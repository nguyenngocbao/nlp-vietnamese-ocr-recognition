# Experiment configurations

- Every runnable YAML is a complete copy of the exact upstream PaddleOCR config used in Kaggle.
- Do not alter a config after recording results; create a new clearly named file.
- The required 2-epoch and 4-epoch configs may differ only in `Global.epoch_num` and output path.
- `Train` uses `rec_train.txt`; `Eval` uses `rec_val.txt`. Final reporting uses `rec_test.txt` only.

## Setup

1. Record the PaddleOCR Git commit in `docs/experiment_log.md`.
2. Save the untouched upstream config in `upstream_reference/`.
3. Make a complete executable copy in the model directory and change dataset/dictionary/pretrained-model/output paths.
4. Make the two controlled-experiment configs from the same complete base, then check their diff.

The current two fine-tuning YAMLs are intentional placeholders until the exact PaddleOCR revision is installed in Kaggle.
