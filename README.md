# HCMUS NLP Vietnamese OCR

PaddleOCR text-recognition experiments for the HCMUS NLP Vietnamese recognition dataset.

## Thành viên nhóm

| Họ và tên | Mã số học viên |
|---|---|
| Bùi Viễn Dương | 25C02015 |
| Nguyễn Minh Thúy Vy | 25C02030 |
| Nguyễn Ngọc Bảo | 25C02031 |

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

Attach `baonguyenngoc0902/hcmus-nlp-vietnamese-recognition-resources` to every notebook. Kaggle mounts it at:

```text
/kaggle/input/hcmus-nlp-vietnamese-recognition-resources
```

Extract `vi_rec_100k.zip` into `/kaggle/working/data`; never write to `/kaggle/input`.

Each notebook clones the pinned official PaddleOCR source and this project repository into `/kaggle/working`. Commit and push code/configuration changes before starting a Kaggle run. Download Kaggle outputs and commit them from a local clone; notebooks do not push to GitHub.

## Required notebooks

1. `01_baseline_paddleocr_original.ipynb`
2. `02_finetune_ppocrv5_mobile_2epochs.ipynb`
3. `03_finetune_ppocrv5_mobile_4epochs.ipynb`

The two fine-tuning runs start from the same pretrained checkpoint and differ only in `Global.epoch_num`.

## JSONL evaluation

```bash
python eval.py --predictions results/pred_baseline_latin_ppocrv5_mobile.jsonl
```

The utility reports exact-match `acc` and `norm_edit_dis`.
