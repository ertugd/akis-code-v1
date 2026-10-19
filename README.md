# Ottoman Turkish Printed Text Recognition on the Akis Dataset

Training and evaluation code for transcribing printed Ottoman Turkish text lines (Arabic script, Naskh) directly into Latin-script Turkish with [TrOCR](https://huggingface.co/microsoft/trocr-base-printed).

Everything uses publicly available data and models only.

## Main results

Book-disjoint test set: 937 lines from 5 books that never appear in training.

| Model | CER | 95% CI | WER |
|---|---|---|---|
| Qwen2-VL-7B + LoRA ([kranidiotis/ottoman-ocr-qwen2.5-vl-7b](https://huggingface.co/kranidiotis/ottoman-ocr-qwen2.5-vl-7b)) | 84.93% | 84.0–86.0 | 106.0% |
| TrOCR fine-tuned | 22.27% | 21.3–23.3 | 61.2% |
| + beam search (4) | 21.69% | 20.7–22.7 | 60.5% |
| + beam search (8) + training-set lexicon rescoring | 21.40% | 20.4–22.4 | 59.9% |
| + print augmentation | 20.26% | 19.3–21.2 | 56.9% |
| **+ print augmentation + beam search (4)** | **19.75%** | **18.9–20.7** | **56.1%** |

Print augmentation lowers CER by 1.9 points (paired bootstrap, 95% CI −2.4 to −1.5, p < 0.001) and improves all five test books.

The public random split ([asparius/ottoman-ocr-8k](https://huggingface.co/datasets/asparius/ottoman-ocr-8k)) gives 13.06% CER with the same model setup. On that split, 86% of test pages also appear in training, so it overestimates performance on unseen books.

Full numbers are in `results/`.

## Requirements

- Python 3.9 or newer
- About 3 GB of disk space for the data, plus 17 GB if you also evaluate the Qwen2-VL model
- A GPU (NVIDIA CUDA or Apple Silicon). Training one model takes about 4.5 hours on an Apple M4 Max.

## Installation

```bash
git clone https://github.com/<user>/akis-code-v1.git
cd akis-code-v1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Download the public Akis line images (584 MB, Apache-2.0):

```bash
bash scripts/download_data.sh
```

Add `--random-split` to also download the asparius random split.

The file names in the Akis archive encode the book, page and line, for example `T-EC-01-S-1896-323_05` is book `T-EC-01-S-1896`, page 323, line 5. `src/prepare_akis_books.py` uses them to build the book-disjoint split:

| Split | Lines | Books | Pages |
|---|---|---|---|
| train | 6,868 | 7 | 2,662 |
| val | 231 | 1 (`HRG_01`) | 109 |
| test | 937 | 5 | 408 |

Test books: `SN-IM-01-UHNU-1732`, `L-MNBF-01-1888`, `R-SS-01-KS-1891`, `T-MN-01-TE-1919`, `R-SD-01-FG-1924`. The line ids of each split are in `splits/`.

## Usage

Run all book-split experiments (about 10 hours):

```bash
bash scripts/run_book_split.sh
```

Run the random-split experiments (about 4 hours):

```bash
bash scripts/run_random_split.sh
```

### Single steps

Prepare the book split:

```bash
python src/prepare_akis_books.py --zip data/akis_dataset_test.zip
```

Train:

```bash
python src/train.py --run_name books_aug --data_dir data/akis_books --augment --epochs 20 --max_target_len 96
```

Evaluate:

```bash
python src/evaluate.py --model outputs/books_aug/best --csv data/akis_books/test.csv --name books_aug --num_beams 4
```

Error analysis and significance tests:

```bash
python src/analyze_errors.py results/books_aug_predictions.csv
python src/bootstrap.py --results results/predictions --test_csv data/akis_books/test.csv
```

### Training options

| Option | Description | Default |
|---|---|---|
| `--model` | Starting checkpoint | `microsoft/trocr-base-printed` |
| `--layout` | `squash` (resize the line to 384×384), `pad` (pad to a square first) or `fold` (cut the line right-to-left and stack the pieces) | `squash` |
| `--augment` | Print augmentation: affine jitter, stroke width, blur, contrast and noise | off |
| `--epochs` | Number of epochs | 5 |
| `--lr` | Learning rate | 5e-5 |
| `--batch_size` | Batch size | 16 |
| `--max_target_len` | Maximum label length in tokens | 128 |

### Evaluating the Qwen2-VL model

```bash
huggingface-cli download kranidiotis/ottoman-ocr-qwen2.5-vl-7b --local-dir models/qwen25vl-7b-ottoman
python src/evaluate_qwen.py --csv data/akis_books/test.csv --batch_size 1 --name books_qwen
```

The checkpoint was saved with transformers 5 and contains Qwen2-VL weights despite its name. `evaluate_qwen.py` loads the tokenizer from `Qwen/Qwen2-VL-7B-Instruct` and maps the vision and RoPE settings to the transformers 4.57 format. Use `--batch_size 1`: batched generation produces empty outputs on Apple Silicon.

## Project structure

```
src/
  prepare_akis_books.py  book-disjoint split from the Akis archive
  prepare_akis.py        random split from asparius/ottoman-ocr-8k
  common.py              dataset, line layouts, print augmentation, metrics
  train.py               fine-tuning
  evaluate.py            CER / WER evaluation
  nbest.py               n-best list generation
  lexicon_rescore.py     lexicon rescoring tuned on the validation book
  evaluate_qwen.py       evaluation of the Qwen2-VL checkpoint
  analyze_errors.py      error types and relaxed CER variants
  bootstrap.py           paired bootstrap tests and confidence intervals
scripts/
  download_data.sh
  run_book_split.sh
  run_random_split.sh
splits/                  line ids of the book-disjoint split
results/
  *_metrics.json         metrics of each run
  bootstrap.json         confidence intervals and paired tests
  error_analysis_*.csv   error analysis
  predictions/           line-level predictions
  training_curves/       validation CER per epoch
```

Result files starting with `books_` use the book split, files starting with `ft_` or `qwen2vl7b_` use the random split.

## Notes

- In transformers 4.57, `VisionEncoderDecoderModel` shifts the labels a second time inside its built-in loss. `train.py` computes cross-entropy from the logits directly.
- On Apple Silicon, `train.py` clears the MPS cache after each validation pass. Without it, memory grows past 40 GB and training stalls.
- All runs use a single seed (42).

## Data and citation

Akis dataset (Apache-2.0): <https://huggingface.co/datasets/OttomanNLP/Akis-Ottoman-Dataset>

> Tasdemir, E.F.B., Tandoğan, Z., Akansu, S.D., Kızılırmak, F., Şen, M.U., Akcan, A., Kuru, M., Yanıkoğlu, B. (2024). Automatic Transcription of Ottoman Documents Using Deep Learning. In: *Document Analysis Systems (DAS 2024)*, LNCS 14994, pp. 422–435. Springer. https://doi.org/10.1007/978-3-031-70442-0_26

TrOCR:

> Li, M., Lv, T., Chen, J., Cui, L., Lu, Y., Florencio, D., Zhang, C., Li, Z., Wei, F. (2023). TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models. *AAAI*, 37(11), 13094–13102.
