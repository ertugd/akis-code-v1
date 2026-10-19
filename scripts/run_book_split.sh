#!/bin/bash
set -e
cd "$(dirname "$0")/.."
PY="python -W ignore"
D="data/akis_books"
mkdir -p logs results/nbest

step() { echo "=== $(date '+%H:%M') $*"; }

step "prepare"
$PY src/prepare_akis_books.py --zip data/akis_dataset_test.zip --out_dir $D

step "baseline: 20 epochs"
$PY src/train.py --run_name books_squash_e20 --data_dir $D --layout squash --epochs 20 --max_target_len 96
$PY src/evaluate.py --model outputs/books_squash_e20/best --csv $D/test.csv --name books_squash_e20
$PY src/evaluate.py --model outputs/books_squash_e20/best --csv $D/test.csv --name books_squash_e20_beam4 --num_beams 4 --batch_size 16

step "lexicon rescoring"
$PY src/nbest.py --model outputs/books_squash_e20/best --csv $D/val.csv --out results/nbest/books_val.json
$PY src/nbest.py --model outputs/books_squash_e20/best --csv $D/test.csv --out results/nbest/books_test.json
$PY src/lexicon_rescore.py --train_csv $D/train.csv --val_nbest results/nbest/books_val.json \
  --test_nbest results/nbest/books_test.json --name books_lexicon

step "augmentation: 20 epochs"
$PY src/train.py --run_name books_squash_aug_e20 --data_dir $D --layout squash --augment --epochs 20 --max_target_len 96
$PY src/evaluate.py --model outputs/books_squash_aug_e20/best --csv $D/test.csv --name books_squash_aug_e20
$PY src/evaluate.py --model outputs/books_squash_aug_e20/best --csv $D/test.csv --name books_squash_aug_e20_beam4 --num_beams 4 --batch_size 16

step "error analysis and bootstrap"
$PY src/analyze_errors.py results/books_{squash_e20,squash_e20_beam4,lexicon,squash_aug_e20,squash_aug_e20_beam4}_predictions.csv \
  --out results/error_analysis_book_split.csv
$PY src/bootstrap.py --results results --test_csv $D/test.csv

step "done"
