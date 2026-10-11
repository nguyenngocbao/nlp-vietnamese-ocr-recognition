"""Generate notebooks 02 (2 epochs) and 03 (4 epochs) from ONE template.

Usage:  python scripts/make_finetune_notebooks.py

The two notebooks differ only in EPOCHS (and the names derived from it).
Install / environment / repo-clone / dataset cells are copied from notebook 01
so that all notebooks set up Kaggle in exactly the same way.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
BASE_NB = NB_DIR / "01_baseline_paddleocr_original.ipynb"

base_cells = {c["id"]: c for c in json.loads(BASE_NB.read_text(encoding="utf-8"))["cells"]}


def src(text):
    lines = text.strip("\n").split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def md(cell_id, text):
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": src(text)}


def code(cell_id, text):
    return {"cell_type": "code", "execution_count": None, "id": cell_id, "metadata": {},
            "outputs": [], "source": src(text)}


def reuse(cell_id):
    cell = json.loads(json.dumps(base_cells[cell_id]))  # deep copy, outputs cleared
    if cell["cell_type"] == "code":
        cell["outputs"], cell["execution_count"] = [], None
    return cell


def build(epochs):
    other = 4 if epochs == 2 else 2
    run = f"ppocrv5_mobile_{epochs}epochs"
    cells = []
    cells.append(md("f01-title", f'''
# Fine-tune PP-OCRv5 mobile — {epochs} epoch

Fine-tune `PP-OCRv5_mobile_rec` trên `rec_train.txt` ({epochs} epoch), chọn checkpoint cuối, rồi đánh giá trên `rec_test.txt` bằng đúng `src/metrics.py` đã dùng cho baseline. Hai notebook fine-tune (2 và 4 epoch) được sinh từ **cùng một template** (`scripts/make_finetune_notebooks.py`) và dùng hai config chỉ khác `Global.epoch_num` và đường dẫn output, nên chênh lệch kết quả chỉ do số epoch.

**Chuẩn bị trên Kaggle:** gắn Input `baonguyenngoc0902/hcmus-nlp-vietnamese-recognition-resources`, bật **GPU T4 x2** và **Internet**. Tuỳ chọn: thêm Kaggle Secret `huggingface.co` (quyền ghi) để cell cuối đẩy checkpoint lên Hugging Face; không có thì cell tự bỏ qua. Dùng **Save & Run All** để chạy trọn pipeline.
'''))
    for cid in ("b02-install-md", "b03-install", "b04-env-md", "b05-env", "b05a-repo-md", "b05b-repo",
                "b06-data-md", "b07-data", "b08-labels-md", "3ea17b70"):
        cells.append(reuse(cid))

    cells.append(md("f06-cfg-md", f'''
## 6. Đọc config fine-tune

Config nằm trong repository (`configs/ppocrv5_mobile/`), không viết tay trong notebook. Cell kiểm tra: số epoch đúng bản này, đường dẫn dữ liệu/từ điển khớp thư mục Input, và **so với config của bản {other} epoch thì chỉ khác `epoch_num`, `save_model_dir`, `save_res_path`**. Đây là bằng chứng cho yêu cầu 2.4 (hai cấu hình chỉ khác số epoch).
'''))
    cells.append(code("f07-cfg", f'''
import difflib
import shutil

import yaml

EPOCHS = {epochs}
OTHER_EPOCHS = {other}
RUN_NAME = f"ppocrv5_mobile_{{EPOCHS}}epochs"
CONFIG_DIR = REPO_DIR / "configs" / "ppocrv5_mobile"
CONFIG_PATH = CONFIG_DIR / f"ppocrv5_mobile_{{EPOCHS}}epochs.yml"
OTHER_PATH = CONFIG_DIR / f"ppocrv5_mobile_{{OTHER_EPOCHS}}epochs.yml"

config_text = CONFIG_PATH.read_text(encoding="utf-8")
config = yaml.safe_load(config_text)

assert config["Global"]["epoch_num"] == EPOCHS
assert Path(config["Global"]["character_dict_path"]) == DICT_PATH
assert Path(config["Train"]["dataset"]["data_dir"]) == TRAIN_LABELS.parent
assert Path(config["Train"]["dataset"]["label_file_list"][0]) == TRAIN_LABELS
assert Path(config["Eval"]["dataset"]["label_file_list"][0]) == VAL_LABELS
assert config["Global"]["save_epoch_step"] == 1 and config["Global"]["seed"] == SEED

diff_lines = [
    line for line in difflib.unified_diff(
        config_text.splitlines(), OTHER_PATH.read_text(encoding="utf-8").splitlines(), lineterm="", n=0)
    if line[:1] in "+-" and line[:3] not in ("+++", "---")
]
print("\\n".join(diff_lines))
allowed = ("epoch_num", "save_model_dir", "save_res_path", "# Fine-tuning config")
assert all(any(key in line for key in allowed) for line in diff_lines), "Configs differ beyond epoch/output paths"

MAX_TEXT_LEN = config["Architecture"]["Head"]["head_list"][1]["NRTRHead"]["max_text_length"]
IMG_W = config["Eval"]["dataset"]["transforms"][2]["RecResizeImg"]["image_shape"][2]
print("epoch_num:", EPOCHS, "| max_text_length:", MAX_TEXT_LEN, "| image width:", IMG_W,
      "| first_bs:", config["Train"]["sampler"]["first_bs"], "| lr:", config["Optimizer"]["lr"])
'''))

    cells.append(md("f08-stats-md", '''
## 7. Thống kê dữ liệu và kiểm tra từ điển

Hai kiểm tra trước khi tốn thời gian huấn luyện:

- **Độ dài nhãn và độ rộng ảnh.** Config gốc của PaddleOCR giới hạn 25 ký tự và ảnh rộng 320 px (khoảng 40 bước CTC), trong khi dòng tiếng Việt dài tới ~80 ký tự. Cell in phân vị độ dài nhãn và độ rộng ảnh sau khi co về cao 48 px để chắc rằng `max_text_length` và chiều rộng trong config đủ lớn. Nhãn dài hơn `max_text_length` sẽ bị PaddleOCR bỏ qua khi huấn luyện.
- **Từ điển `vi_dict.txt`.** Ký tự nằm ngoài từ điển không mô hình nào học được; cell đếm tỷ lệ ký tự nhãn train bị thiếu trong từ điển (config có `use_space_char: true`, tự thêm khoảng trắng).
'''))
    cells.append(code("f09-stats", '''
import random
from collections import Counter

from PIL import Image

def percentile(sorted_values, p):
    return sorted_values[min(len(sorted_values) - 1, int(len(sorted_values) * p))]

train_labels = [line.split("\\t", 1)[1] for line in split_lines["train"]]
lengths = sorted(len(label) for label in train_labels)
print("Label length p50/p90/p99/max:", [percentile(lengths, p) for p in (0.5, 0.9, 0.99)], lengths[-1])
too_long = sum(n > MAX_TEXT_LEN for n in lengths)
print(f"Labels longer than max_text_length={MAX_TEXT_LEN}: {too_long} ({too_long / len(lengths):.4%})")
assert too_long / len(lengths) < 0.001, "Raise max_text_length in the config generator"

rng = random.Random(SEED)
widths = []
for line in rng.sample(split_lines["train"], 300):
    with Image.open(TRAIN_LABELS.parent / line.split("\\t", 1)[0]) as image:
        widths.append(image.width * 48 / image.height)  # chiều rộng khi co về cao 48 px
widths.sort()
print("Width at height 48 (300 samples) p50/p90/p99/max:", [round(percentile(widths, p)) for p in (0.5, 0.9, 0.99)], round(widths[-1]))
print(f"Share wider than IMG_W={IMG_W}: {sum(w > IMG_W for w in widths) / len(widths):.1%} (bị nén ngang)")

dict_lines = DICT_PATH.read_text(encoding="utf-8").splitlines()
charset = set(dict_lines) | {" "}
print("Dictionary entries:", len(dict_lines), "| unique:", len(set(dict_lines)), "| has space line:", " " in dict_lines)
missing = Counter(ch for label in train_labels for ch in label if ch not in charset)
total_chars = sum(len(label) for label in train_labels)
print(f"Train characters outside dictionary: {sum(missing.values())} / {total_chars} ({sum(missing.values()) / total_chars:.4%})")
print("Most frequent missing:", missing.most_common(10))
'''))

    cells.append(md("f10-src-md", '''
## 8. Mã nguồn PaddleOCR và trọng số tiền huấn luyện

Huấn luyện dùng `tools/train.py` của PaddleOCR nên cần mã nguồn đúng tag `v3.3.0` (cùng phiên bản với thư viện đã cài). Cell ghi lại mã commit vào `ENV_INFO`, **vá một lỗi của backbone** (xem bên dưới), cài vài thư viện phụ trợ của PaddleOCR rồi kiểm tra `import ppocr`, sau đó tải `PP-OCRv5_mobile_rec_pretrained.pdparams`. Đây là điểm khởi đầu fine-tune; lớp đầu ra có số lớp khác (từ điển tiếng Việt) nên PaddleOCR tự bỏ qua tham số không khớp kích thước và khởi tạo lại các lớp đó.

**Bản vá bắt buộc.** Ở cuối backbone `PPLCNetV3` (`ppocr/modeling/backbones/rec_lcnetv3.py`, v3.3.0), lúc huấn luyện mã gốc luôn ép số bước thời gian của CTC về đúng 40 (`F.adaptive_avg_pool2d(x, [1, 40])`), còn lúc suy luận số bước là `W/8`. Config gốc rộng 320 px nên hai con số trùng nhau (320/8 = 40). Nhãn tiếng Việt dài trung vị 57 và tối đa 80 ký tự, nên với mã gốc khoảng 74% dòng có đích CTC không thể căn chỉnh (CTC cần số bước ≥ độ dài nhãn) và mô hình không hội tụ. Cell bên dưới thay `40` bằng `W/8` (cùng hình dạng với nhánh suy luận); với ảnh rộng 960 ta có 120 bước, đủ cho nhãn dài nhất. Bản vá này dùng chung cho cả hai lần chạy 2 và 4 epoch nên không phải biến thí nghiệm.
'''))
    cells.append(code("f11-src", '''
import urllib.request

PADDLEOCR_REF = "v3.3.0"
PADDLEOCR_DIR = Path("/kaggle/working/PaddleOCR")
if not (PADDLEOCR_DIR / "tools" / "train.py").is_file():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", PADDLEOCR_REF,
                    "https://github.com/PaddlePaddle/PaddleOCR.git", str(PADDLEOCR_DIR)], check=True)
ENV_INFO["paddleocr_source_ref"] = PADDLEOCR_REF
ENV_INFO["paddleocr_source_commit"] = subprocess.check_output(
    ["git", "-C", str(PADDLEOCR_DIR), "rev-parse", "HEAD"], text=True).strip()
print("PaddleOCR source commit:", ENV_INFO["paddleocr_source_commit"])

# Fix: training forces CTC to 40 time steps whatever the image width (inference uses W/8).
BACKBONE = PADDLEOCR_DIR / "ppocr" / "modeling" / "backbones" / "rec_lcnetv3.py"
OLD_POOL = "x = F.adaptive_avg_pool2d(x, [1, 40])"
NEW_POOL = "x = F.adaptive_avg_pool2d(x, [1, max(1, x.shape[3] // 2)])"
src = BACKBONE.read_text(encoding="utf-8")
if OLD_POOL in src:
    BACKBONE.write_text(src.replace(OLD_POOL, NEW_POOL), encoding="utf-8")
src = BACKBONE.read_text(encoding="utf-8")
assert NEW_POOL in src and OLD_POOL not in src, "rec_lcnetv3.py was not patched"
ENV_INFO["paddleocr_patch"] = "rec_lcnetv3.py: adaptive_avg_pool2d([1, 40]) -> [1, W_feat // 2] (CTC time steps = W/8 in training, as in inference)"
print("Patched:", ENV_INFO["paddleocr_patch"])

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(PADDLEOCR_DIR / "requirements.txt")], check=True)
check = subprocess.run([sys.executable, "-c", "import ppocr.data, ppocr.modeling.architectures"],
                       cwd=PADDLEOCR_DIR, capture_output=True, text=True)
assert check.returncode == 0, check.stderr[-2000:]

PRETRAINED_URL = "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams"
PRETRAINED_PATH = Path(config["Global"]["pretrained_model"])
PRETRAINED_PATH.parent.mkdir(parents=True, exist_ok=True)
if not PRETRAINED_PATH.is_file():
    urllib.request.urlretrieve(PRETRAINED_URL, PRETRAINED_PATH)
assert PRETRAINED_PATH.stat().st_size > 1_000_000, "Pretrained download looks truncated"
print("Pretrained:", PRETRAINED_PATH, f"({PRETRAINED_PATH.stat().st_size / 1e6:.1f} MB)")
'''))

    cells.append(md("f12-train-md", f'''
## 9. Huấn luyện

Chạy `tools/train.py` đúng với config trong repository, không ghi đè tham số nào, trên 2 GPU T4 (`paddle.distributed.launch`). Checkpoint lưu sau mỗi epoch (`save_epoch_step: 1`); PaddleOCR luôn ghi thêm `latest` (epoch cuối) và `best_accuracy` (tốt nhất trên `rec_val.txt`, đánh giá mỗi ~1000 bước). **Bài báo cáo dùng `latest`**, tức mô hình sau đúng {epochs} epoch, để so sánh 2 và 4 epoch công bằng; `best_accuracy` chỉ để tham khảo.

Thời gian huấn luyện là tổng thời gian chạy lệnh (gồm cả các lần đánh giá trên tập val) và được lưu vào `train_time.json`. Cả hai notebook đo theo cùng một cách, trên cùng loại GPU.
'''))
    cells.append(code("f13-train", '''
GPUS = "0,1"
OUTPUT_DIR = Path(config["Global"]["save_model_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_LOG = OUTPUT_DIR / "train_stdout.log"
TRAIN_TIME_PATH = OUTPUT_DIR / "train_time.json"
shutil.copy(CONFIG_PATH, OUTPUT_DIR / "config_used.yml")

run_env = os.environ.copy()
run_env["PYTHONPATH"] = f"{PADDLEOCR_DIR}:{run_env.get('PYTHONPATH', '')}"
run_env["PYTHONUNBUFFERED"] = "1"

train_cmd = [sys.executable, "-m", "paddle.distributed.launch", "--gpus", GPUS,
             "tools/train.py", "-c", str(CONFIG_PATH)]
print(" ".join(train_cmd))

train_start = time.perf_counter()
with TRAIN_LOG.open("w", encoding="utf-8") as log, subprocess.Popen(
        train_cmd, cwd=PADDLEOCR_DIR, env=run_env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
    for line in proc.stdout:
        log.write(line)
        print(line, end="")
    returncode = proc.wait()
train_seconds = time.perf_counter() - train_start

assert returncode == 0, f"Training failed (exit {returncode}); see {TRAIN_LOG}"
assert (OUTPUT_DIR / "latest.pdparams").is_file(), "latest.pdparams missing"
TRAIN_TIME_PATH.write_text(json.dumps(
    {"epochs": EPOCHS, "gpus": GPUS, "train_seconds": train_seconds}, indent=2) + "\\n", encoding="utf-8")
print(f"Training time: {train_seconds:.1f} s ({train_seconds / 60:.1f} min)")
'''))

    cells.append(md("f14-export-md", '''
## 10. Xuất mô hình suy luận

`tools/export_model.py` chuyển checkpoint `latest` thành mô hình suy luận (đã gắn từ điển và kích thước ảnh từ config). Mô hình này được nạp lại bằng `TextRecognition` của PaddleOCR, đúng đường dự đoán đã dùng cho baseline, nên hai bên so sánh được trực tiếp.
'''))
    cells.append(code("f15-export", '''
INFER_DIR = OUTPUT_DIR / "inference"
subprocess.run(
    [sys.executable, "tools/export_model.py", "-c", str(CONFIG_PATH),
     "-o", f"Global.pretrained_model={OUTPUT_DIR / 'latest'}", f"Global.save_inference_dir={INFER_DIR}"],
    cwd=PADDLEOCR_DIR, env=run_env, check=True)
print(sorted(path.name for path in INFER_DIR.iterdir()))
'''))

    cells.append(md("f16-resolve-md", '''
## 11. Ghép nhãn với ảnh test

Giống notebook baseline: ghép từng dòng `rec_test.txt` với đường dẫn ảnh thật (3.003 dòng).
'''))
    cells.append(reuse("5b96a998"))

    cells.append(md("f17-predict-md", '''
## 12. Dự đoán toàn bộ tập test và lưu JSONL

Nạp mô hình đã fine-tune và dự đoán 3.003 ảnh với `batch_size=1`, giống hệt baseline (batch khác nhau làm đệm ảnh khác đi và đổi kết quả). Mỗi dòng JSONL có `image`, `gt`, `pred`, đúng định dạng mà `eval.py` và `src/jsonl.py` yêu cầu.
'''))
    cells.append(code("f18-predict", '''
from paddleocr import TextRecognition
from tqdm.auto import tqdm

MODEL_NAME = config["Global"]["model_name"]
BATCH_SIZE = 1

model_init_start = time.perf_counter()
recognizer = TextRecognition(model_name=MODEL_NAME, model_dir=str(INFER_DIR), device="gpu:0")
model_init_seconds = time.perf_counter() - model_init_start

def text_from_result(result) -> str:
    payload = result.json
    if isinstance(payload, str):
        payload = json.loads(payload)
    return str(payload["res"]["rec_text"])

prediction_start = time.perf_counter()
prediction_rows = []
for start in tqdm(range(0, len(test_examples), BATCH_SIZE), desc="Predicting test set"):
    batch = test_examples[start : start + BATCH_SIZE]
    results = recognizer.predict(input=[row["path"] for row in batch], batch_size=len(batch))
    assert len(results) == len(batch), "Prediction count differs from input count."
    for example, result in zip(batch, results):
        prediction_rows.append({"image": example["image"], "gt": example["gt"], "pred": text_from_result(result)})
prediction_seconds = time.perf_counter() - prediction_start
assert len(prediction_rows) == len(test_examples)

PREDICTIONS_PATH = RESULTS_DIR / f"pred_{RUN_NAME}.jsonl"
with PREDICTIONS_PATH.open("w", encoding="utf-8") as file:
    for row in prediction_rows:
        file.write(json.dumps(row, ensure_ascii=False) + "\\n")

with PREDICTIONS_PATH.open(encoding="utf-8") as file:
    saved_rows = [json.loads(line) for line in file if line.strip()]
assert len(saved_rows) == len(prediction_rows)
assert all({"image", "gt", "pred"} <= row.keys() for row in saved_rows)
print(f"Prediction time: {prediction_seconds:.2f} s | Predictions: {PREDICTIONS_PATH}")
print("Sample:", prediction_rows[0])
'''))

    cells.append(md("f19-metrics-md", '''
## 13. Tính metric báo cáo

Dùng `src/metrics.py` (cùng bản với baseline): `acc` và `norm_edit_dis` giữ khoảng trắng là số liệu chính; hai metric `*_ignore_space` là số liệu phụ. Cell cũng lưu thời gian huấn luyện, phiên bản thư viện, seed, GPU, commit repo và commit PaddleOCR để điền bảng so sánh (2.3) và `docs/experiment_log.md`.
'''))
    cells.append(code("f20-metrics", '''
from src.metrics import normalize_text, recognition_metrics

train_info = json.loads(TRAIN_TIME_PATH.read_text(encoding="utf-8"))
metrics = {
    "model": MODEL_NAME,
    "run": RUN_NAME,
    "split": "rec_test.txt",
    "finetune_epochs": EPOCHS,
    "checkpoint": "latest (final epoch)",
    **recognition_metrics(prediction_rows),
    "batch_size": BATCH_SIZE,
    "train_seconds": train_info["train_seconds"],
    "train_gpus": train_info["gpus"],
    "model_init_seconds": model_init_seconds,
    "prediction_seconds": prediction_seconds,
    "config": f"configs/ppocrv5_mobile/ppocrv5_mobile_{EPOCHS}epochs.yml",
    "environment": ENV_INFO,
}
METRICS_PATH = RESULTS_DIR / f"metrics_{RUN_NAME}.json"
METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
print(json.dumps(metrics, ensure_ascii=False, indent=2))
'''))

    cells.append(md("f21-xcheck-md", '''
## 14. Đối chiếu bằng công cụ đánh giá của PaddleOCR (tuỳ chọn)

Chạy `tools/eval.py` trên `rec_test.txt` với checkpoint `latest`. Số `acc` của PaddleOCR không chuẩn hóa NFC và không dùng `src/metrics.py`, nên chỉ để kiểm tra đường suy luận đã xuất không bị lệch (kỳ vọng gần bằng `acc` ở trên). Lỗi ở cell này không ảnh hưởng kết quả chính.
'''))
    cells.append(code("f22-xcheck", '''
try:
    xcheck = subprocess.run(
        [sys.executable, "tools/eval.py", "-c", str(CONFIG_PATH),
         "-o", f"Global.checkpoints={OUTPUT_DIR / 'latest'}", f"Eval.dataset.label_file_list=[{TEST_LABELS}]"],
        cwd=PADDLEOCR_DIR, env=run_env, capture_output=True, text=True, timeout=1800)
    print("\\n".join((xcheck.stdout + xcheck.stderr).splitlines()[-12:]))
except Exception as exc:  # không chặn pipeline
    print("Cross-check skipped:", exc)
'''))

    cells.append(md("f23-upload-md", f'''
## 15. Lưu checkpoint lên Hugging Face và liệt kê file cần lấy

Nếu có Kaggle Secret `huggingface.co`, đẩy thư mục `{run}/` (checkpoint `latest`, `best_accuracy`, mô hình suy luận, config, log) lên repo `nngocbao090/hcmus-vi-paddleocr-rec`; file `*.pdopt` (trạng thái optimizer, không cần để suy luận) bị bỏ qua. Không có token thì bỏ qua, checkpoint vẫn nằm trong Output của notebook. Hai file kết quả nhỏ trong `results/` (JSONL và metrics) là thứ cần đưa vào repository.
'''))
    cells.append(code("f24-upload", '''
HF_REPO = "nngocbao090/hcmus-vi-paddleocr-rec"
try:
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret("huggingface.co")
except Exception as exc:
    hf_token = None
    print("No huggingface.co secret, skipping upload:", exc)

if hf_token:
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    api.create_repo(HF_REPO, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=HF_REPO, repo_type="model", folder_path=str(OUTPUT_DIR),
                      path_in_repo=RUN_NAME, ignore_patterns=["*.pdopt"])
    print(f"Uploaded to https://huggingface.co/{HF_REPO}/tree/main/{RUN_NAME}")

print("Files to put in the repository (results/):")
for path in (PREDICTIONS_PATH, METRICS_PATH):
    print(f"{path} ({path.stat().st_size / 1024:.1f} KB)")
'''))
    return cells


def write(epochs, filename):
    nb = {"cells": build(epochs),
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    path = NB_DIR / filename
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("wrote", path.relative_to(ROOT), len(nb["cells"]), "cells")


if __name__ == "__main__":
    write(2, "02_finetune_ppocrv5_mobile_2epochs.ipynb")
    write(4, "03_finetune_ppocrv5_mobile_4epochs.ipynb")
