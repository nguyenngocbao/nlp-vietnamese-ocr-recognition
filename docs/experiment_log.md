# Experiment log

Record one entry per completed Kaggle run; never overwrite a result. All runs: seed 42, Python 3.13, paddlepaddle-gpu 3.2.1 (cu126), paddleocr 3.3.0, paddlex 3.3.0, Kaggle 2x Tesla T4. Test metrics are on `rec_test.txt` (3,003 lines) with NFC normalization; `acc` / `norm_edit_dis` keep spaces.

| Run ID | Date | Notebook | Config | Train time | Test acc | Test norm_edit_dis | Result files / checkpoint |
|---|---|---|---|---:|---:|---:|---|
| baseline | 2026-10-11 | `01_baseline_paddleocr_original.ipynb` | `configs/baseline/paddleocr_original_ppocrv5_mobile.yml` | n/a (not trained) | 0.0450 | 0.8549 | `results/baseline/` |
| ft-2ep-prepatch (invalid) | 2026-10-11 | `02_finetune_ppocrv5_mobile_2epochs.ipynb` | `configs/ppocrv5_mobile/ppocrv5_mobile_2epochs.yml` | 6884.5 s (1 h 54 min 45 s) | 0.0440 | 0.8097 | `results/ft_2epochs_prepatch/`; checkpoint: https://huggingface.co/nngocbao090/hcmus-vi-paddleocr-rec/tree/main/ppocrv5_mobile_2epochs |
| ft-2ep | 2026-10-12 | `02_finetune_ppocrv5_mobile_2epochs.ipynb` (patched) | `configs/ppocrv5_mobile/ppocrv5_mobile_2epochs.yml` | 6567.5 s (1 h 49 min 27 s) | 0.5511 (ignore space 0.5721) | 0.9705 (ignore space 0.9682) | `results/ft_2epochs/`; checkpoint: https://huggingface.co/nngocbao090/hcmus-vi-paddleocr-rec/tree/main/ppocrv5_mobile_2epochs |
| ft-4ep | 2026-10-12 | `03_finetune_ppocrv5_mobile_4epochs.ipynb` (patched) | `configs/ppocrv5_mobile/ppocrv5_mobile_4epochs.yml` | 14059.2 s (3 h 54 min 19 s) | 0.5698 (ignore space 0.5921) | 0.9723 (ignore space 0.9700) | `results/ft_4epochs/`; checkpoint: https://huggingface.co/nngocbao090/hcmus-vi-paddleocr-rec/tree/main/ppocrv5_mobile_4epochs |

Baseline note: the unmodified `latin_PP-OCRv5_mobile_rec` inference model (assignment item 2.1), repo commit `7ce973c`; its Latin dictionary lacks many Vietnamese characters, which explains the very low exact-match accuracy. Prediction time 76.4 s for 3,003 images (batch size 1).

Fine-tune 2-epoch note: repo commit `733b4ba` (`first_bs` 24 after an out-of-memory failure at 48), evaluated checkpoint `latest`, prediction time 62.3 s (batch size 1). Ignoring spaces the model is slightly better than the baseline (acc 0.0623 vs 0.0476, norm_edit_dis 0.8273 vs 0.8243), but with spaces kept it is worse (norm_edit_dis 0.8097 vs 0.8549); the first sample shows dropped characters, i.e. the model is still under-trained after 2 epochs.

Invalid run, kept for the record: `ft-2ep-prepatch` was trained with the unpatched PaddleOCR v3.3.0 backbone, which forces the CTC to 40 time steps in training whatever the image width (inference uses W/8). About 74% of our lines (median 57, maximum 80 characters) cannot be aligned in 40 steps, so the run did not converge: training loss stayed around 9-10, train acc 0.0 at the last step, validation acc 0.0003-0.055, test acc 0.0440. It must not be used to compare with the baseline. Notebooks 02 and 03 now patch `rec_lcnetv3.py` (`[1, 40]` -> `[1, W//2]`, see `configs/README.md`) and both fine-tune runs are repeated with the patch; the same test set from another project (`quachthanhhmd/vietnamese-ocr-nlp`, same data split) reaches about 0.62-0.66 acc (spaces ignored) after patching the same backbone.

## Required controlled experiment (assignment 2.4)

- Model: `PP-OCRv5_mobile_rec`, initialized from `PP-OCRv5_mobile_rec_pretrained.pdparams`.
- Constant: initial checkpoint, data split, image size, batch size, learning rate, augmentation, seed and software versions (see the deviation table in `configs/README.md`).
- Variable: `Global.epoch_num` — 2 versus 4.
- Evaluated checkpoint: `latest` (the final epoch) of each run. Final metrics: `rec_test.txt` only; `rec_val.txt` is used only for validation during training.
- Train time = wall time of `tools/train.py` including periodic validation, measured identically in both notebooks.

Fine-tune 4-epoch note: repo commit `fb973bd` (patched backbone), interactive Kaggle session (the commit run was cancelled to save GPU quota), evaluated checkpoint `latest` (final epoch), prediction time 69.9 s (batch size 1). Versus 2 epochs (same config except `epoch_num`): acc +0.0187 (0.5511 -> 0.5698), norm_edit_dis +0.0017 (0.9705 -> 0.9723), at 2.1x the training time (6567.5 s -> 14059.2 s). Best validation acc during training 0.5603 (epoch 4).
