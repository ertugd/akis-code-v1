import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from common import LAYOUTS, configure_model, get_device, load_image


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layout", choices=LAYOUTS, default="squash")
    ap.add_argument("--num_beams", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    device = get_device()
    df = pd.read_csv(args.csv, keep_default_na=False)
    processor = TrOCRProcessor.from_pretrained(args.model)
    model = configure_model(VisionEncoderDecoderModel.from_pretrained(args.model), processor).to(device).eval()

    k = args.num_beams
    records = []
    for start in tqdm(range(0, len(df), args.batch_size), desc="n-best"):
        batch = df.iloc[start:start + args.batch_size]
        images = [load_image(p, args.layout).convert("RGB") for p in batch["path"]]
        pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)
        out = model.generate(pixel_values, num_beams=k, num_return_sequences=k,
                             output_scores=True, return_dict_in_generate=True)
        texts = processor.batch_decode(out.sequences, skip_special_tokens=True)
        scores = out.sequences_scores.tolist()
        for i, (_, row) in enumerate(batch.iterrows()):
            cands = [{"text": texts[i * k + j].strip(), "score": scores[i * k + j]} for j in range(k)]
            records.append({"id": row["id"], "text": row["text"], "candidates": cands})
        if device.type == "mps":
            torch.mps.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(records, ensure_ascii=False))
    print(f"{len(records)} satır -> {args.out}")


if __name__ == "__main__":
    main()
