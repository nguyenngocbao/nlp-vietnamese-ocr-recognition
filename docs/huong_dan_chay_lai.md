# Hướng dẫn chạy lại thí nghiệm (Kaggle)

Tài liệu này hướng dẫn từng bước để một thành viên khác chạy lại cả ba notebook (baseline, fine-tune 2 epoch, fine-tune 4 epoch) trên Kaggle, lấy kết quả về và kiểm tra lại bằng `eval.py`. Mọi số liệu trong báo cáo đều đo trên tập test `rec_test.txt` (3.003 dòng) với seed 42.

## 0. Cần chuẩn bị

- Tài khoản Kaggle đã xác minh số điện thoại (để bật được GPU và Internet).
- Quota GPU còn đủ. Kaggle cho khoảng 30 giờ GPU mỗi tuần; chạy đủ ba notebook tốn khoảng 6 giờ (baseline vài phút, 2 epoch khoảng 1 giờ 50 phút, 4 epoch khoảng 3 giờ 54 phút).
- Quyền truy cập dataset `baonguyenngoc0902/hcmus-nlp-vietnamese-recognition-resources`. Nếu không tìm thấy khi thêm vào notebook, nhờ chủ dataset (Bảo) chia sẻ hoặc chuyển sang public.
- Repo GitHub `nguyenngocbao/nlp-vietnamese-ocr-recognition` phải là bản đã push mới nhất, vì cả ba notebook đều clone repo này về Kaggle để dùng `src/metrics.py` và các file config.
- Ba file notebook trong thư mục `notebooks/` của repo:
  1. `01_baseline_paddleocr_original.ipynb`
  2. `02_finetune_ppocrv5_mobile_2epochs.ipynb`
  3. `03_finetune_ppocrv5_mobile_4epochs.ipynb`

## 1. Import notebook lên Kaggle

Làm lần lượt cho từng file (mỗi file là một notebook riêng trên Kaggle):

1. Vào kaggle.com, bấm **Create → New Notebook**.
2. Trong notebook mới, chọn **File → Import Notebook**, kéo thả file `.ipynb` vào (hoặc chọn từ máy).
3. Đặt tên rõ ràng, ví dụ `HCMUS NLP Baseline - PaddleOCR`, `HCMUS NLP Fine-tune PP-OCRv5 - epoch 2`, `HCMUS NLP Fine-tune PP-OCRv5 - epoch 4`.

## 2. Cài đặt notebook (làm cho cả ba)

Mở bảng **Notebook** ở cột bên phải (hoặc **Settings** trên thanh menu):

1. **Add Input** → tìm `hcmus-nlp-vietnamese-recognition-resources` → bấm dấu **+**. Sau khi thêm, dữ liệu nằm ở `/kaggle/input/datasets/baonguyenngoc0902/hcmus-nlp-vietnamese-recognition-resources`; notebook tự dò đường dẫn này (cả kiểu mount cũ), không cần sửa code.
2. **Session options → Accelerator**: chọn **GPU T4 x2**. Notebook 02 và 03 huấn luyện trên 2 GPU (`GPUS = "0,1"`); notebook 01 chỉ cần 1 GPU nhưng nên chọn cùng loại để điều kiện đo giống nhau.
3. **Session options → Internet**: bật **On**. Internet chỉ dùng để cài thư viện, clone repo, tải mô hình và checkpoint tiền huấn luyện; dữ liệu không đổi.

## 3. Chạy notebook baseline (01)

1. Chọn **Run All** (hoặc chạy từng cell từ trên xuống). Không đổi thứ tự cell.
2. Cell 1 cài PaddlePaddle GPU 3.2.1, PaddleOCR 3.3.0 và PaddleX 3.3.0 (file cài nặng khoảng 1,6 GB nên mất vài phút). Cell 6 clone repo vào `/kaggle/working/repo` và in mã commit; ghi lại mã này.
3. Chờ đến khi cell dự đoán (mục 8) chạy xong 3.003 ảnh (khoảng 1 đến 2 phút) và cell metric (mục 9) in ra JSON.
4. Kết quả đúng phải khớp với bảng ở mục 6 bên dưới.

## 4. Chạy notebook fine-tune (02 và 03)

Hai notebook giống hệt nhau, chỉ khác số epoch (2 và 4), nên làm cùng một cách:

1. Khuyến nghị dùng **Save Version → Save & Run All (Commit)** để Kaggle chạy nền; đóng trình duyệt vẫn không sao, và kết quả được lưu ở tab **Output** của phiên bản đó. Nếu muốn theo dõi trực tiếp thì chọn **Run All** trong phiên tương tác, nhưng phải giữ máy và tab không bị ngắt.
2. Các cell chạy theo thứ tự: cài đặt môi trường, ghi nhận seed và phần cứng, clone repo, tìm dữ liệu, kiểm tra nhãn, đọc config (`configs/ppocrv5_mobile/ppocrv5_mobile_{2,4}epochs.yml`), thống kê dữ liệu, lấy mã nguồn PaddleOCR đúng tag `v3.3.0`, huấn luyện, xuất mô hình suy luận, dự đoán trên test, tính metric.
3. Ở bước lấy mã nguồn PaddleOCR (mục 8 của notebook), notebook tự vá một dòng trong `rec_lcnetv3.py` (số bước thời gian CTC lúc huấn luyện phải bằng lúc suy luận với ảnh rộng 960 px). Đừng bỏ cell này và đừng sửa tay: thiếu bản vá thì mô hình gần như không học được (acc khoảng 0,04).
4. Huấn luyện là bước lâu nhất (khoảng 1 giờ 50 phút với 2 epoch, 3 giờ 54 phút với 4 epoch). Checkpoint lưu sau mỗi epoch, nên nếu phiên bị ngắt thì checkpoint đã lưu vẫn còn.
5. Cell cuối (mục "Lưu checkpoint lên Hugging Face") là tuỳ chọn, xem mục 7 bên dưới. Không có token thì nó chỉ in `No huggingface.co secret, skipping upload` và bỏ qua, không phải lỗi.

## 5. Lấy kết quả về

1. Mở notebook đã chạy xong, vào tab **Output** (với bản Commit thì mở trang version, chọn tab **Output**).
2. Tải hai file nhỏ cho mỗi lần chạy, nằm trong `/kaggle/working/results/`:
   - `pred_*.jsonl`: mỗi dòng gồm đường dẫn ảnh, nhãn `gt` và dự đoán `pred`.
   - `metrics_*.json`: các chỉ số cùng thông tin môi trường (phiên bản thư viện, seed, GPU, mã commit repo, thời gian huấn luyện và dự đoán).
3. Đặt vào repo theo cấu trúc sau (chỉ commit hai loại file nhỏ này, không commit checkpoint hay dữ liệu ảnh):

| Lần chạy | Thư mục trong repo |
|---|---|
| Baseline | `results/baseline/` |
| Fine-tune 2 epoch | `results/ft_2epochs/` |
| Fine-tune 4 epoch | `results/ft_4epochs/` |

   Thư mục `results/ft_2epochs_prepatch/` là một lần chạy cũ bị lỗi CTC, chỉ giữ để làm bằng chứng; đừng ghi đè hay dùng nó để so sánh.
4. Ghi dòng kết quả mới vào `docs/experiment_log.md` (ngày chạy, thời gian huấn luyện, `acc`, `norm_edit_dis`, đường dẫn thư mục kết quả).

## 6. Kiểm tra lại bằng `eval.py` (không cần Kaggle)

`eval.py` chấm điểm trực tiếp từ file `.jsonl`, dùng đúng hàm `recognition_metrics` mà notebook đã dùng, nên số phải khớp với `metrics_*.json`. Chỉ cần Python 3 và bản repo đã clone:

```bash
git clone https://github.com/nguyenngocbao/nlp-vietnamese-ocr-recognition.git
cd nlp-vietnamese-ocr-recognition
python eval.py --predictions results/baseline/pred_baseline_ppocrv5_mobile.jsonl
python eval.py --predictions results/ft_2epochs/pred_ppocrv5_mobile_2epochs.jsonl
python eval.py --predictions results/ft_4epochs/pred_ppocrv5_mobile_4epochs.jsonl
```

(Thêm `--output ten_file.json` nếu muốn lưu kết quả ra file; có thể cài `pip install python-Levenshtein` cho nhanh hơn, không cài thì code tự dùng bản thay thế.)

Số liệu tham chiếu trên 3.003 dòng test (giữ khoảng trắng):

| Mô hình | `acc` | `norm_edit_dis` |
|---|---:|---:|
| Baseline (`latin_PP-OCRv5_mobile_rec` gốc) | 0,0450 | 0,8549 |
| Fine-tune 2 epoch | 0,5511 | 0,9705 |
| Fine-tune 4 epoch | 0,5698 | 0,9723 |

Baseline gần như không có tính ngẫu nhiên, nên số của bạn phải trùng. Hai bản fine-tune có thể lệch nhẹ (cỡ vài phần nghìn đến khoảng 1 điểm `acc`) do huấn luyện trên GPU không hoàn toàn tất định, và đó là bình thường. Nếu `acc` của fine-tune chỉ quanh 0,04 thì bản vá CTC chưa được áp dụng (xem mục 4, bước 3).

## 7. Lưu checkpoint lên Hugging Face (tuỳ chọn)

Cell cuối của notebook 02 và 03 đẩy checkpoint lên repo Hugging Face `nngocbao090/hcmus-vi-paddleocr-rec`. Chỉ chủ repo mới có quyền ghi vào repo đó. Nếu muốn tự đẩy checkpoint của mình:

1. Tạo token Hugging Face quyền **write** của riêng bạn.
2. Trong notebook Kaggle: **Add-ons → Secrets → Add secret**, tên đúng là `huggingface.co`, dán token vào, rồi bật công tắc **Attached** cho notebook.
3. Sửa dòng `HF_REPO = "..."` ở cell cuối thành repo của bạn (dạng `ten-ban/ten-repo`).
4. Không bao giờ dán token vào ô code, ô markdown hay commit lên GitHub.

Nếu chỉ cần chạy lại để lấy số liệu thì bỏ qua mục này; checkpoint vẫn nằm trong tab **Output** của notebook.

## 8. Lỗi thường gặp

- **Không thấy dataset khi Add Input**: dataset chưa được chia sẻ cho tài khoản của bạn (xem mục 0).
- **Lỗi `git clone` hoặc `pip download`**: Internet đang tắt; bật lại ở Session options.
- **Notebook báo thiếu file `src/metrics.py`**: repo trên GitHub chưa được push bản mới nhất, hoặc `REPO_REF` (mặc định `main`) trỏ sai nhánh. Có thể đổi `REPO_REF` thành mã commit để cố định đúng phiên bản code.
- **`ModuleNotFoundError: langchain.docstore`**: Kaggle cài sẵn `langchain` bản mới làm `import paddleocr` lỗi. Cell cài đặt đã xử lý việc này; hãy chạy lại cell 1 đầy đủ rồi khởi động lại phiên (Restart) nếu vẫn lỗi.
- **Tab Log bị treo khi huấn luyện** (log tqdm rất dài): tải lại trang. Phiên và quá trình huấn luyện vẫn chạy bình thường ở phía sau. Dùng bản **Save & Run All (Commit)** sẽ tránh được tình trạng này.
- **Phiên bị ngắt giữa chừng** (hết giờ hoặc mất kết nối): chạy lại notebook từ đầu. Notebook không có bước tự tiếp tục huấn luyện từ checkpoint dở dang.
- **Tuyệt đối không bấm ⟳ Factory reset** của Kaggle: nó đặt lại phiên và xoá dữ liệu trong `/kaggle/working`, nên mọi kết quả chưa tải về sẽ mất.
- **Hết quota GPU**: xem phần Settings → Usage của Kaggle; quota reset theo tuần. Có thể chạy từng notebook vào các tuần khác nhau.

## 9. Các thông số cố định (để báo cáo nêu rõ điều kiện đo)

- Phần cứng: Kaggle, 2 × NVIDIA Tesla T4, Python 3.13.
- Phiên bản: `paddlepaddle-gpu==3.2.1` (build cu126), `paddleocr==3.3.0`, `paddlex==3.3.0`, mã nguồn PaddleOCR tag `v3.3.0` (xem `requirements.txt`).
- Seed: 42. Dự đoán dùng `batch_size=1` cho cả baseline và fine-tune.
- Hai bản fine-tune chỉ khác nhau đúng một yếu tố là số epoch (2 so với 4); mọi tham số khác nằm trong `configs/ppocrv5_mobile/`.
- Không so sánh số trên tập validation (`rec_val.txt`) với số trên tập test; mọi số trong bảng so sánh đều đo trên `rec_test.txt`.
