import argparse
import json
import re
from collections import Counter
from pathlib import Path

import jiwer
import pandas as pd

TOKEN = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?", re.UNICODE)
ALPHABET = "abcçdefgğhıijklmnoöprsştuüvyzâîû"


def words(text: str):
    return [w.lower() for w in TOKEN.findall(text)]


def build_lexicon(csv: str) -> Counter:
    lex = Counter()
    for t in pd.read_csv(csv, keep_default_na=False)["text"]:
        lex.update(words(t))
    return lex


def oov_rate(text: str, lex: Counter) -> float:
    ws = words(text)
    return sum(w not in lex for w in ws) / len(ws) if ws else 0.0


def rescore(candidates, lex, lam):
    return max(candidates, key=lambda c: c["score"] - lam * oov_rate(c["text"], lex))["text"]


def edits1(w):
    splits = [(w[:i], w[i:]) for i in range(len(w) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    swaps = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
    replaces = [a + c + b[1:] for a, b in splits if b for c in ALPHABET]
    inserts = [a + c + b for a, b in splits for c in ALPHABET]
    return set(deletes + swaps + replaces + inserts)


def fix_oov(text: str, lex: Counter, min_count: int, min_len: int = 4) -> str:
    def repl(m):
        w = m.group(0)
        low = w.lower()
        if low in lex or len(low) < min_len:
            return w
        cands = [(lex[c], c) for c in edits1(low) if lex[c] >= min_count]
        if not cands:
            return w
        best = max(cands)[1]
        return best.capitalize() if w[0].isupper() else best

    return TOKEN.sub(repl, text)


def cer(refs, preds):
    return jiwer.cer(list(refs), list(preds))


def evaluate(records, lex, lam, min_count):
    refs = [r["text"] for r in records]
    preds = [rescore(r["candidates"], lex, lam) for r in records]
    if min_count:
        preds = [fix_oov(p, lex, min_count) for p in preds]
    return cer(refs, preds), preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="data/akis_books/train.csv")
    ap.add_argument("--val_nbest", required=True)
    ap.add_argument("--test_nbest", required=True)
    ap.add_argument("--name", default="books_lexicon")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    lex = build_lexicon(args.train_csv)
    val = json.loads(Path(args.val_nbest).read_text())
    test = json.loads(Path(args.test_nbest).read_text())
    print(f"sözlük: {len(lex)} kelime | test OOV oranı: "
          f"{sum(oov_rate(r['text'], lex) for r in test) / len(test):.1%}")

    grid = []
    for lam in [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
        for mc in [0, 2, 5, 20]:
            grid.append((evaluate(val, lex, lam, mc)[0], lam, mc))
    grid.sort()
    print("doğrulamada en iyi 5 ayar (CER, lambda, min_count):")
    for row in grid[:5]:
        print(f"  {row[0]:.4f}  lambda={row[1]}  min_count={row[2]}")
    base_val = evaluate(val, lex, 0.0, 0)[0]
    best_cer, lam, mc = grid[0]
    print(f"doğrulama: beam-8 en iyi aday {base_val:.4f} -> sözlük {best_cer:.4f}")

    base_test, base_preds = evaluate(test, lex, 0.0, 0)
    test_cer, test_preds = evaluate(test, lex, lam, mc)
    refs = [r["text"] for r in test]
    metrics = {"name": args.name, "lambda": lam, "min_count": mc, "lexicon_size": len(lex),
               "CER_beam8": base_test, "CER": test_cer,
               "WER_beam8": jiwer.wer(refs, base_preds), "WER": jiwer.wer(refs, test_preds),
               "n": len(test)}
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    out = Path(args.out_dir)
    pd.DataFrame({"id": [r["id"] for r in test], "text": refs, "pred": test_preds}).to_csv(
        out / f"{args.name}_predictions.csv", index=False)
    pd.DataFrame({"id": [r["id"] for r in test], "text": refs, "pred": base_preds}).to_csv(
        out / f"{args.name}_beam8_predictions.csv", index=False)
    (out / f"{args.name}_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
