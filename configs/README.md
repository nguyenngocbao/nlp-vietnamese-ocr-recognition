# Experiment configurations

- `upstream_reference/latin_PP-OCRv5_mobile_rec.yml` — the upstream PaddleOCR **v3.3.0** training config the fine-tuning configs are derived from. Reference only; never edit it.
- `ppocrv5_mobile/ppocrv5_mobile_{2,4}epochs.yml` — the two runnable, complete configs. They are **generated** by `scripts/make_finetune_configs.py` (`python scripts/make_finetune_configs.py`), which applies the same asserted text edits to the upstream file. Do not hand-edit them; change the script and regenerate.
- The 2-epoch and 4-epoch files differ only in `Global.epoch_num`, `Global.save_model_dir` and `Global.save_res_path` (notebooks 02/03 assert this with a diff before training).
- `baseline/` — settings of the unmodified baseline run (no training).
- `Train` uses `rec_train.txt`; `Eval` (validation during training) uses `rec_val.txt`. Final reporting uses `rec_test.txt` only.
- Do not alter a config after recording results; create a new clearly named file.

## Why the config deviates from upstream

Both runs share every deviation below, so none of them is an experiment variable. Upstream is tuned for short Latin text (25 characters, 320 px wide).

| Setting | Upstream | Ours | Reason |
|---|---|---|---|
| dictionary | `ppocrv5_latin_dict.txt` | `vi_dict.txt` | Vietnamese characters (given by the assignment) |
| `max_text_length` (NRTRHead, RecConAug, train MultiLabelEncode) | 25 | 100 | Test labels reach 80 characters; longer labels are dropped in training |
| image width (`d2s_train_image_shape`, `RecConAug`, sampler scales, eval `RecResizeImg`) | 320 | 960 | ~120 CTC steps for 80-character lines; 320 gives only ~40 |
| `first_bs` | 128 | 48 | keeps pixels per batch near upstream at 3x the width, to fit a T4 |
| `warmup_epoch` | 5 | 0.5 | upstream warm-up is longer than the whole 2-epoch run |
| `save_epoch_step` | 10 | 1 | a checkpoint after every epoch |
| `eval_batch_step` | `[0, 500]` | `[0, 1000]` | roughly one validation pass per epoch |
| `print_batch_step`, `num_workers` | 10, 8 | 100, 4 | shorter log; Kaggle has 4 CPU cores |
| `seed` | (not set, defaults to 1024) | 42 | project-wide seed |
| `pretrained_model` | `PP-OCRv5_mobile_rec_pretrained.pdparams` | same file, stored in `/kaggle/working/pretrained/` | the Latin config also starts from this generic checkpoint |

The notebook's data-statistics cell (section 7) prints label-length and image-width percentiles so these numbers can be checked against the data before training.
