#!/bin/bash
set -e
cd "$(dirname "$0")/.."
mkdir -p data/raw

curl -L --retry 5 -C - -o data/akis_dataset_test.zip \
  "https://huggingface.co/datasets/OttomanNLP/Akis-Ottoman-Dataset/resolve/main/akis_dataset_test.zip"

if [ "$1" == "--random-split" ]; then
  for f in test-00000-of-00001 train-00000-of-00002 train-00001-of-00002; do
    curl -L --retry 5 -C - -o data/raw/$f.parquet \
      "https://huggingface.co/datasets/asparius/ottoman-ocr-8k/resolve/main/data/$f.parquet"
  done
fi
