#!/bin/bash
set -e
cd "$(dirname "$0")/.."
PY="python -W ignore"
D="data/akis"
COMMON="--epochs 8 --max_target_len 96"
mkdir -p logs results

step() { echo "=== $(date '+%H:%M') $*"; }

step "prepare"
$PY src/prepare_akis.py --raw_dir data/raw --out_dir $D

step "squash"
$PY src/train.py --run_name ft_squash --data_dir $D --layout squash $COMMON
$PY src/evaluate.py --model outputs/ft_squash/best --csv $D/test.csv --layout squash --name ft_squash
$PY src/evaluate.py --model outputs/ft_squash/best --csv $D/test.csv --layout squash --name ft_squash_beam4 --num_beams 4 --batch_size 16

step "fold"
$PY src/train.py --run_name ft_fold --data_dir $D --layout fold $COMMON
$PY src/evaluate.py --model outputs/ft_fold/best --csv $D/test.csv --layout fold --name ft_fold
$PY src/evaluate.py --model outputs/ft_fold/best --csv $D/test.csv --layout fold --name ft_fold_beam4 --num_beams 4 --batch_size 16

step "error analysis"
$PY src/analyze_errors.py results/{ft_squash,ft_squash_beam4,ft_fold,ft_fold_beam4}_predictions.csv \
  --out results/error_analysis_random_split.csv

step "done"
