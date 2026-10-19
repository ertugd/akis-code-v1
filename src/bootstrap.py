import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein


def line_stats(pred_csv: str, ids=None):
    df = pd.read_csv(pred_csv, keep_default_na=False)
    if ids is not None:
        df = df.set_index("id").loc[ids].reset_index()
    dist = np.array([Levenshtein.distance(r, p) for r, p in zip(df["text"], df["pred"])], dtype=float)
    length = df["text"].str.len().to_numpy(dtype=float)
    return df["id"].tolist(), dist, length


def ci(values, alpha=0.05):
    return float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))


def bootstrap_pair(base_csv, new_csv, n_boot=10000, seed=0, mask=None):
    ids, d_base, n = line_stats(base_csv)
    _, d_new, _ = line_stats(new_csv, ids)
    if mask is not None:
        d_base, d_new, n = d_base[mask], d_new[mask], n[mask]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(n), size=(n_boot, len(n)))
    cer_base = d_base[idx].sum(1) / n[idx].sum(1)
    cer_new = d_new[idx].sum(1) / n[idx].sum(1)
    diff = cer_new - cer_base
    return {
        "n_lines": int(len(n)),
        "CER_base": float(d_base.sum() / n.sum()),
        "CER_new": float(d_new.sum() / n.sum()),
        "diff": float((d_new.sum() - d_base.sum()) / n.sum()),
        "diff_ci95": ci(diff),
        "CER_base_ci95": ci(cer_base),
        "CER_new_ci95": ci(cer_new),
        "p_value": float((diff >= 0).mean()),
        "lines_better": int((d_new < d_base).sum()),
        "lines_worse": int((d_new > d_base).sum()),
        "lines_same": int((d_new == d_base).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--test_csv", default="data/akis_books/test.csv")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--out", default="results/bootstrap.json")
    args = ap.parse_args()
    r = Path(args.results)
    pairs = {
        "beam4_vs_greedy": ("books_squash_e20", "books_squash_e20_beam4"),
        "lexicon_vs_beam4": ("books_squash_e20_beam4", "books_lexicon"),
        "aug_vs_noaug_greedy": ("books_squash_e20", "books_squash_aug_e20"),
        "aug_vs_noaug_beam4": ("books_squash_e20_beam4", "books_squash_aug_e20_beam4"),
        "full_vs_greedy_baseline": ("books_squash_e20", "books_squash_aug_e20_beam4"),
    }
    test = pd.read_csv(args.test_csv, keep_default_na=False)
    ids, _, _ = line_stats(r / "books_squash_e20_beam4_predictions.csv")
    books = test.set_index("id").loc[ids, "book"].to_numpy()

    out = {}
    for name, (base, new) in pairs.items():
        out[name] = bootstrap_pair(r / f"{base}_predictions.csv", r / f"{new}_predictions.csv", args.n_boot)
        out[name].update({"base": base, "new": new})
    per_book = {}
    for b in sorted(set(books)):
        per_book[b] = bootstrap_pair(r / "books_squash_e20_beam4_predictions.csv",
                                     r / "books_squash_aug_e20_beam4_predictions.csv", args.n_boot, mask=books == b)
    out["aug_vs_noaug_beam4_per_book"] = per_book

    singles = {}
    for name in ["books_qwen2vl7b_kranidiotis", "books_squash_e20", "books_squash_e20_beam4", "books_lexicon",
                 "books_squash_aug_e20", "books_squash_aug_e20_beam4", "ft_squash", "ft_squash_beam4",
                 "ft_fold", "ft_fold_beam4", "qwen2vl7b_kranidiotis"]:
        if not (r / f"{name}_predictions.csv").exists():
            continue
        _, d, n = line_stats(r / f"{name}_predictions.csv")
        idx = np.random.default_rng(1).integers(0, len(n), size=(args.n_boot, len(n)))
        singles[name] = {"CER": float(d.sum() / n.sum()), "CER_ci95": ci(d[idx].sum(1) / n[idx].sum(1)),
                         "n_lines": int(len(n))}
    out["cer_with_ci"] = singles

    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for name in pairs:
        x = out[name]
        print(f"{name:22s} {x['CER_base']:.4f} -> {x['CER_new']:.4f}  diff={x['diff']:+.4f} "
              f"CI95=[{x['diff_ci95'][0]:+.4f}, {x['diff_ci95'][1]:+.4f}]  p={x['p_value']:.4f}  "
              f"daha iyi/kötü satır={x['lines_better']}/{x['lines_worse']}")
    print("kitap bazında (veri artırma + beam4 vs beam4):")
    for b, x in per_book.items():
        print(f"  {b:22s} {x['CER_base']:.4f} -> {x['CER_new']:.4f}  "
              f"CI95=[{x['diff_ci95'][0]:+.4f}, {x['diff_ci95'][1]:+.4f}]  p={x['p_value']:.4f}")


if __name__ == "__main__":
    main()
