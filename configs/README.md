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
| image width (`d2s_train_image_shape`, `RecConAug`, sampler scales, eval `RecResizeImg`) | 320 | 960 | the model emits W/8 CTC steps: 120 for 960 px (enough for 80-character lines), only 40 for 320 px (needs the backbone patch below) |
| `first_bs` | 128 | 24 | 3x the width needs a smaller batch; 48 ran out of memory on a 15 GB T4 (2 GPUs, so 48 per step overall) |
| `warmup_epoch` | 5 | 0.5 | upstream warm-up is longer than the whole 2-epoch run |
| `save_epoch_step` | 10 | 1 | a checkpoint after every epoch |
| `eval_batch_step` | `[0, 500]` | `[0, 1000]` | roughly one validation pass per epoch |
| `print_batch_step`, `num_workers` | 10, 8 | 100, 4 | shorter log; Kaggle has 4 CPU cores |
| `seed` | (not set, defaults to 1024) | 42 | project-wide seed |
| `pretrained_model` | `PP-OCRv5_mobile_rec_pretrained.pdparams` | same file, stored in `/kaggle/working/pretrained/` | the Latin config also starts from this generic checkpoint |

The notebook's data-statistics cell (section 7) prints label-length and image-width percentiles so these numbers can be checked against the data before training.

## Required patch of the PaddleOCR source (backbone)

Upstream `PPLCNetV3` (`ppocr/modeling/backbones/rec_lcnetv3.py`, v3.3.0) ends with `if self.training: x = F.adaptive_avg_pool2d(x, [1, 40])` and, in eval mode, `F.avg_pool2d(x, [3, 2])`. So in training the CTC always sees **40 time steps**, whatever the image width, while inference sees W/8. At 320 px both are 40, which is why upstream never notices. Our labels have median length 57 and maximum 80 characters, so about 74% of the lines have a CTC target that cannot be aligned in 40 steps and training does not converge (first 2-epoch run, before the patch: loss stayed around 9-10, test acc 0.044).

Notebooks 02 and 03 therefore replace `40` by `max(1, x.shape[3] // 2)` (W/8 steps, identical to the inference branch) right after cloning PaddleOCR, assert that the edit is in place, and record it in `ENV_INFO["paddleocr_patch"]`. The patch is shared by both runs, so it is not an experiment variable. It does not change the exported inference model, which already uses the eval branch.
