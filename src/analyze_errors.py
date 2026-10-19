import argparse
import re
import string
from pathlib import Path

import jiwer
import pandas as pd

PUNCT = re.compile(f"[{re.escape(string.punctuation)}’‘“”«»–—]")
CIRCUMFLEX = str.maketrans("âîûÂÎÛ", "aiuAIU")


def no_punct(s: str) -> str:
    return re.sub(r"\s+", " ", PUNCT.sub(" ", s)).strip()


def no_space(s: str) -> str:
    return re.sub(r"\s+", "", s)


def error_type(ref: str, pred: str) -> str:
    if ref == pred:
        return "correct"
    if ref.lower() == pred.lower():
        return "case_only"
    if no_space(ref) == no_space(pred):
        return "space_only"
    if no_punct(ref) == no_punct(pred):
        return "punct_only"
    if ref.translate(CIRCUMFLEX) == pred.translate(CIRCUMFLEX):
        return "circumflex_only"
    return "other"


def analyze(pred_csv: Path) -> dict:
    df = pd.read_csv(pred_csv, keep_default_na=False)
    refs, preds = df["text"].astype(str).tolist(), df["pred"].astype(str).tolist()
    types = pd.Series([error_type(r, p) for r, p in zip(refs, preds)]).value_counts(normalize=True)

    low = lambda xs: [x.lower() for x in xs]
    np_r, np_p = [no_punct(x.lower()) or "-" for x in refs], [no_punct(x.lower()) for x in preds]
    ns_r = [no_space(x) or "-" for x in np_r]
    ns_p = [no_space(x) for x in np_p]
    circ_r = [x.translate(CIRCUMFLEX) for x in np_r]
    circ_p = [x.translate(CIRCUMFLEX) for x in np_p]
    return {
        "file": pred_csv.name,
        "n": len(df),
        "CER_strict": jiwer.cer(refs, preds),
        "WER_strict": jiwer.wer(refs, preds),
        "CER_lower": jiwer.cer(low(refs), low(preds)),
        "CER_no_punct": jiwer.cer(np_r, np_p),
        "WER_no_punct": jiwer.wer(np_r, np_p),
        "CER_no_punct_no_space": jiwer.cer(ns_r, ns_p),
        "CER_no_punct_no_circumflex": jiwer.cer(circ_r, circ_p),
        **{f"lines_{t}": types.get(t, 0.0)
           for t in ["correct", "case_only", "space_only", "punct_only", "circumflex_only", "other"]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_csvs", nargs="+")
    ap.add_argument("--out", default="results/error_analysis.csv")
    args = ap.parse_args()
    table = pd.DataFrame([analyze(Path(p)) for p in args.pred_csvs])
    table.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.set_index("file").T.to_string())


if __name__ == "__main__":
    main()
