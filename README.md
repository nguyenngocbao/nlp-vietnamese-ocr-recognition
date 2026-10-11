# HCMUS NLP Vietnamese OCR

PaddleOCR text-recognition experiments for the HCMUS NLP Vietnamese recognition dataset.

## Thành viên nhóm

| Họ và tên | Mã số học viên | Công việc | Đóng góp (%) |
|---|---|---|---:|
| Bùi Viễn Dương | 25C02015 | _(điền)_ | _(điền)_ |
| Nguyễn Minh Thúy Vy | 25C02030 | _(điền)_ | _(điền)_ |
| Nguyễn Ngọc Bảo | 25C02031 | _(điền)_ | _(điền)_ |

## Layout

```text
configs/     Experiment settings and immutable upstream references
notebooks/   Kaggle notebooks, in report order
src/         Shared metric and JSONL utilities
results/     Small versioned metrics and prediction files
reports/     Error-analysis artifacts and figures
scripts/     Repeatable command-line helpers
```

Large artifacts are excluded from Git: image data, pretrained weights, checkpoints and exported models. Record their links in `docs/experiment_log.md`.

## Kaggle data

Attach `baonguyenngoc0902/hcmus-nlp-vietnamese-recognition-resources` to every notebook, and turn on GPU and Internet. Kaggle mounts it read-only at:

```text
/kaggle/input/datasets/baonguyenngoc0902/hcmus-nlp-vietnamese-recognition-resources
```

(older mounts use `/kaggle/input/hcmus-nlp-vietnamese-recognition-resources`; the notebooks try both). The archive is already extracted: the dataset contains `vi_dict.txt` and the folder `vi_rec_100k/` with `rec_train.txt`, `rec_val.txt`, `rec_test.txt` and the images. Never write to `/kaggle/input`; small outputs go to `/kaggle/working/results`.

Notebook 01 is self-contained (no clone). Download Kaggle outputs and commit them from a local clone; notebooks do not push to GitHub. Fine-tuning notebooks will additionally clone the official PaddleOCR source at tag `v3.3.0`.

## Required notebooks

1. `01_baseline_paddleocr_original.ipynb`
2. `02_finetune_ppocrv5_mobile_2epochs.ipynb`
3. `03_finetune_ppocrv5_mobile_4epochs.ipynb`

The two fine-tuning runs start from the same pretrained checkpoint and differ only in `Global.epoch_num`.

The baseline is the unmodified official `latin_PP-OCRv5_mobile_rec` model (the same model that is fine-tuned), run with PaddlePaddle 3.2.1, PaddleOCR 3.3.0 and PaddleX 3.3.0 (see `requirements.txt`). Seed is 42 everywhere. All reported numbers are measured on `rec_test.txt` only.

## JSONL evaluation

```bash
python eval.py --predictions results/pred_baseline_ppocrv5_mobile.jsonl
```

The utility reports exact-match `acc` and `norm_edit_dis`.
