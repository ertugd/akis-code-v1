import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from common import compute_metrics, get_device

SYSTEM = ("You are an expert paleographer specializing in Ottoman Turkish documents written in Arabic script. "
          "Your task is to transcribe Ottoman Turkish handwritten or printed text lines into Modern Turkish.")
USER = "Bu Osmanli Turkcesi metin satirini Modern Turkceye transkribe et."


def build_messages(image):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": USER}]},
    ]


@torch.no_grad()
def predict(model, processor, df, device, batch_size, max_new_tokens):
    processor.tokenizer.padding_side = "left"
    preds = []
    for start in tqdm(range(0, len(df), batch_size), desc="tahmin"):
        batch = df.iloc[start:start + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch["path"]]
        texts = [processor.apply_chat_template(build_messages(im), tokenize=False, add_generation_prompt=True)
                 for im in images]
        inputs = processor(text=texts, images=images, padding=True, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        out = out[:, inputs["input_ids"].shape[1]:]
        preds.extend(t.strip() for t in processor.batch_decode(out, skip_special_tokens=True))
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qwen25vl-7b-ottoman")
    ap.add_argument("--processor", default="Qwen/Qwen2-VL-7B-Instruct")
    ap.add_argument("--csv", default="data/akis/test.csv")
    ap.add_argument("--name", default="qwen25vl7b")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = get_device()
    df = pd.read_csv(args.csv, keep_default_na=False)
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=42)

    processor = AutoProcessor.from_pretrained(args.processor)
    config = AutoConfig.from_pretrained(args.model)
    vision = config.vision_config
    if getattr(vision, "out_hidden_size", None):
        vision.hidden_size = vision.out_hidden_size
    for cfg in (config, getattr(config, "text_config", None)):
        rope = getattr(cfg, "rope_parameters", None) if cfg is not None else None
        if rope and not getattr(cfg, "rope_scaling", None):
            cfg.rope_scaling = {"type": "default", "rope_type": "default", "mrope_section": rope["mrope_section"]}
            cfg.rope_theta = rope.get("rope_theta", 1000000.0)
    model = AutoModelForImageTextToText.from_pretrained(args.model, config=config, torch_dtype=torch.bfloat16)
    model = model.to(device).eval()

    df["pred"] = predict(model, processor, df, device, args.batch_size, args.max_new_tokens)
    metrics = {"name": args.name, "model": args.model, "csv": args.csv,
               **compute_metrics(df["text"], df["pred"])}

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    df[["id", "text", "pred"]].to_csv(out / f"{args.name}_predictions.csv", index=False)
    (out / f"{args.name}_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
