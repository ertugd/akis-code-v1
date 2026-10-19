import argparse
import re
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image

TEST_BOOKS = ["SN-IM-01-UHNU-1732", "L-MNBF-01-1888", "R-SS-01-KS-1891", "T-MN-01-TE-1919", "R-SD-01-FG-1924"]
VAL_BOOKS = ["HRG_01"]


def parse_name(stem: str):
    m = re.match(r"(.+)-(\d+[A-Z\-]*)_(\d+)$", stem)
    return (m.group(1), m.group(2), int(m.group(3))) if m else (stem, "", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="akis_dataset_test.zip")
    ap.add_argument("--out_dir", default="data/akis_books")
    args = ap.parse_args()
    out = Path(args.out_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    z = zipfile.ZipFile(args.zip)
    rows = []
    for line in z.read("files.txt").decode("utf-8").splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        img_name, ann_name = parts
        stem = img_name.rsplit(".", 1)[0]
        book, page, line_no = parse_name(stem)
        path = img_dir / img_name
        if not path.exists():
            path.write_bytes(z.read("images/" + img_name))
        with Image.open(path) as im:
            w, h = im.size
        text = z.read("annotations/" + ann_name).decode("utf-8").strip()
        split = "test" if book in TEST_BOOKS else "val" if book in VAL_BOOKS else "train"
        rows.append({"id": stem, "path": str(path), "text": text, "book": book, "page": page,
                     "line": line_no, "width": w, "height": h, "split": split})

    df = pd.DataFrame(rows)
    df = df[df["text"].str.len() > 0]
    for split, part in df.groupby("split"):
        part.drop(columns="split").to_csv(out / f"{split}.csv", index=False)
        print(f"{split:5s}: {len(part):5d} satır, {part['book'].nunique():2d} kitap, "
              f"{part.groupby(['book', 'page']).ngroups:4d} sayfa")

    train_text = set(df.loc[df.split != "test", "text"])
    test = df[df.split == "test"]
    print(f"test satırlarından eğitimde birebir aynısı olan: {test['text'].isin(train_text).sum()}")


if __name__ == "__main__":
    main()
