# Experiment log

Record one entry per completed Kaggle run; never overwrite a result. All runs: seed 42, Python 3.13, paddlepaddle-gpu 3.2.1 (cu126), paddleocr 3.3.0, paddlex 3.3.0, Kaggle 2x Tesla T4. Test metrics are on `rec_test.txt` (3,003 lines) with NFC normalization; `acc` / `norm_edit_dis` keep spaces.

| Run ID | Date | Notebook | Config | Train time | Test acc | Test norm_edit_dis | Result files / checkpoint |
|---|---|---|---|---:|---:|---:|---|
| baseline | 2026-10-11 | `01_baseline_paddleocr_original.ipynb` | `configs/baseline/paddleocr_original_ppocrv5_mobile.yml` | n/a (not trained) | 0.0450 | 0.8549 | `results/baseline/` |
| ft-2ep | _pending_ | `02_finetune_ppocrv5_mobile_2epochs.ipynb` | `configs/ppocrv5_mobile/ppocrv5_mobile_2epochs.yml` | | | | |
| ft-4ep | _pending_ | `03_finetune_ppocrv5_mobile_4epochs.ipynb` | `configs/ppocrv5_mobile/ppocrv5_mobile_4epochs.yml` | | | | |

Baseline note: the unmodified `latin_PP-OCRv5_mobile_rec` inference model (assignment item 2.1), repo commit `7ce973c`; its Latin dictionary lacks many Vietnamese characters, which explains the very low exact-match accuracy. Prediction time 76.4 s for 3,003 images (batch size 1).

## Required controlled experiment (assignment 2.4)

- Model: `PP-OCRv5_mobile_rec`, initialized from `PP-OCRv5_mobile_rec_pretrained.pdparams`.
- Constant: initial checkpoint, data split, image size, batch size, learning rate, augmentation, seed and software versions (see the deviation table in `configs/README.md`).
- Variable: `Global.epoch_num` — 2 versus 4.
- Evaluated checkpoint: `latest` (the final epoch) of each run. Final metrics: `rec_test.txt` only; `rec_val.txt` is used only for validation during training.
- Train time = wall time of `tools/train.py` including periodic validation, measured identically in both notebooks.
