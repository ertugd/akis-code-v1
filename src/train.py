import argparse
import gc
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, get_linear_schedule_with_warmup

from common import LAYOUTS, LineDataset, compute_metrics, configure_model, get_device, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/trocr-base-printed")
    ap.add_argument("--data_dir", default="data/akis")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--layout", choices=LAYOUTS, default="squash")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--max_target_len", type=int, default=128)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--train_limit", type=int, default=None)
    ap.add_argument("--val_limit", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    run_dir = Path(args.out_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    train_df = pd.read_csv(Path(args.data_dir) / "train.csv", keep_default_na=False)
    val_df = pd.read_csv(Path(args.data_dir) / "val.csv", keep_default_na=False)
    if args.train_limit:
        train_df = train_df.sample(n=min(args.train_limit, len(train_df)), random_state=args.seed)
    if args.val_limit:
        val_df = val_df.sample(n=min(args.val_limit, len(val_df)), random_state=args.seed)
    print(f"cihaz={device}  train={len(train_df)}  val={len(val_df)}  layout={args.layout}")

    processor = TrOCRProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model)
    model = configure_model(model, processor, args.max_target_len).to(device)

    train_ds = LineDataset(train_df, processor, layout=args.layout, max_target_len=args.max_target_len,
                           augment=args.augment)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, persistent_workers=args.num_workers > 0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(args.warmup_ratio * total_steps), total_steps)

    def free_memory():
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    def evaluate(tag):
        preds = predict(model, processor, val_df, device, layout=args.layout, batch_size=16, progress=False)
        free_memory()
        m = compute_metrics(val_df["text"], preds)
        mem = torch.mps.driver_allocated_memory() / 2**30 if device.type == "mps" else 0.0
        print(f"[{tag}] val CER={m['CER']:.4f}  WER={m['WER']:.4f}  gpu_mem={mem:.1f}GB  örnek: {preds[0]!r}")
        return m

    best_cer = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        bar = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(bar, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), batch["labels"].reshape(-1), ignore_index=-100
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            if step % 20 == 0:
                bar.set_postfix(loss=f"{running / step:.4f}")

        m = evaluate(f"epoch {epoch}")
        history.append({"epoch": epoch, "train_loss": running / len(loader),
                        "minutes": (time.time() - t0) / 60, **m})
        if m["CER"] < best_cer:
            best_cer = m["CER"]
            model.save_pretrained(run_dir / "best")
            processor.save_pretrained(run_dir / "best")
            print(f"  -> en iyi model kaydedildi (CER={best_cer:.4f})")
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    print(f"Bitti. En iyi val CER={best_cer:.4f}  ({run_dir})")


if __name__ == "__main__":
    main()
