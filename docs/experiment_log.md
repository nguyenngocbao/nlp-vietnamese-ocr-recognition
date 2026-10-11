# Experiment log

Record one entry per completed Kaggle run; never overwrite a result.

| Run ID | Date | Notebook | Config | Seed | PaddleOCR | PaddlePaddle | GPU | Train time | Test acc | Test norm_edit_dis | Checkpoint / output link |
|---|---|---|---|---:|---|---|---|---:|---:|---:|---|
| | | | | | | | | | | | |

## Required controlled experiment

- Model: `PP-OCRv5_mobile_rec`
- Constant: checkpoint, data split, image size, batch size, learning rate, augmentation, seed and software versions.
- Variable: `Global.epoch_num` — 2 versus 4.
- Final metrics: `rec_test.txt` only.
