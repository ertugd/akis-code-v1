import argparse
import io
from pathlib import Path

import pandas as pd
from PIL import Image


def load_split(raw_dir: Path, split: str) -> pd.DataFrame:
    files = sorted(raw_dir.glob(f"{split}-*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def export(df: pd.DataFrame, split: str, img_dir: Path) -> pd.DataFrame:
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, r in df.iterrows():
        img = Image.open(io.BytesIO(r["image_path"]["bytes"]))
        path = img_dir / f"{split}_{i:05d}.png"
        img.save(path)
        rows.append({"id": f"{split}_{i:05d}", "path": str(path), "text": r["text"].strip(),
                     "width": img.width, "height": img.height})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--out_dir", default="data/akis")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    raw, out = Path(args.raw_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train = export(load_split(raw, "train"), "train", out / "images")
    test = export(load_split(raw, "test"), "test", out / "images")
    val = train.sample(frac=args.val_frac, random_state=args.seed)
    train = train.drop(val.index)

    for name, df in [("train", train), ("val", val), ("test", test)]:
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"{name:5s}: {len(df):5d} satır, ort. {df['text'].str.len().mean():.0f} karakter, "
              f"ort. genişlik {df['width'].mean():.0f}px")

    train_texts = set(train["text"]) | set(val["text"])
    dup = test["text"].isin(train_texts)
    print(f"test satırlarından eğitimde birebir aynısı olan: {dup.sum()} ({dup.mean():.1%})")
    chars = sorted(set("".join(pd.concat([train, val, test])["text"])))
    print(f"karakter kümesi ({len(chars)}): {''.join(chars)}")


if __name__ == "__main__":
    main()
