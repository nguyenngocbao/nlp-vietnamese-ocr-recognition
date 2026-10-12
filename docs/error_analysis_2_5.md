# Mục 2.5 — Phân tích lỗi (mô hình fine-tune 4 epoch, tập test `rec_test.txt`)

Đề yêu cầu lấy 20 dòng đọc sai và chia thành ba loại: sai dấu thanh, sai chữ cái, lặp ký tự. Nguồn dữ liệu là `results/ft_4epochs/pred_ppocrv5_mobile_4epochs.jsonl`. Trên 3.003 dòng test có 1.292 dòng sai (so khớp chính xác sau NFC, tính cả khoảng trắng). 20 dòng được chọn ngẫu nhiên từ 1.292 dòng sai (theo thứ tự trong file pred) bằng `random.seed(42)` và `random.sample(..., 20)`, rồi được đối chiếu gt-pred từng ký tự và phân loại bằng tay.

Cách phân loại: (i) sai dấu thanh khi hai ký tự chỉ khác một trong năm dấu thanh (sắc, huyền, hỏi, ngã, nặng); (ii) sai chữ cái khi thay, thêm hoặc bớt một chữ cái, kể cả các chữ có dấu phụ khác nhau như a/â/ă, o/ô/ơ, d/đ; (iii) lặp ký tự khi mô hình thêm một ký tự lặp liền kề (kiểu `nhưnng`, `nguười`). Một dòng có nhiều lỗi được xếp theo loại có nhiều lỗi nhất. Dòng chỉ sai khoảng trắng, dấu câu hoặc mất cả cụm cuối dòng không thuộc ba loại của đề nên được đếm riêng ở nhóm "khác".

## Bảng 20 dòng

| # | Loại | Nhãn (gt) | Mô hình đọc (pred) |
|---|---|---|---|
| 1 | khác | …có “chính” chứ không | …có “chính”) chứ không |
| 2 | chữ cái | Tròi dát bī ròi lai thai 1. | Trời dất bĩ rồi lại thái 1. |
| 3 | dấu thanh | Bộ binh phải bắt 300 tên biển | Bộ binh phái bắt 300 tên biền |
| 4 | chữ cái | Đáng ta chỉ đạo còng nhàn và | Đảng ta chỉ đạo công nhân và |
| 5 | dấu thanh | tuy nhiêu, mà xét các quyền trúng | tuy nhiều, mà xét các quyển trúng |
| 6 | khác | …vạn náu ở trong lũng bên cạnh Mã Ấp, | …vạn náu ở trong |
| 7 | dấu thanh | …đến phía ta đằng trước… | …đến phía tả đằng trước… |
| 8 | chữ cái | tháng II, … chỉ huy | tháng 1I, … chỉ-huy |
| 9 | dấu thanh | …tháng tám năm Bính | …tháng tám năm Binh- |
| 10 | chữ cái | CHÚ GIẢI CỦA DỊCH GIẢ: | CHÔ GIẢI CỦA DỊCH GIẢ: |
| 11 | khác | – Quận công nha 21 người, – Hoàng… | Quận công nha 21 người, - Hoàng… |
| 12 | khác | Quân ngũ: nam và bắc… | Quân ngũ : nam và bắc… |
| 13 | dấu thanh | Việt sứ liêu án, … Bảnđánh máy cùa | Việt sử tiêu án, … Bản đánh máy của |
| 14 | chữ cái | đánh hãng. Nác Thu | đánh hăng. Nặc Thu |
| 15 | chữ cái | Nhận là … phách lợi | nhận là … phúc lợi |
| 16 | chữ cái | Túr suong chi huy sur. | Tú suong chỉ huy sú. |
| 17 | chữ cái | Mỹ vui Việt Nam (1980-1975), Ninh Công an | Mỹ và Việt Nam (1950-1975), Nxb. Công an |
| 18 | chữ cái | dều dạ rau cả lên | đều dạ ran cả lên |
| 19 | khác | giữ công việc hai ti, | giữ công việc hai ti', |
| 20 | dấu thanh | « thứ II), sứ Tàu | « thứ II), sử Tàu |

## Thống kê

| Loại lỗi | 20 dòng mẫu | Tỷ lệ |
|---|---:|---:|
| (i) Sai dấu thanh | 6 | 30% |
| (ii) Sai chữ cái | 9 | 45% |
| (iii) Lặp ký tự | 0 | 0% |
| Khác (khoảng trắng, dấu câu, mất cụm chữ) | 5 | 25% |

Nếu chỉ tính 15 dòng thuộc ba loại của đề thì sai dấu thanh chiếm 40%, sai chữ cái 60% và lặp ký tự 0%.

## Nhận xét

Lỗi chiếm nhiều nhất là sai chữ cái (khoảng 45%), thường là nhầm giữa các chữ có hình dạng gần nhau như `a`/`â`/`ă`, `o`/`ô`/`ơ`, `u`/`n` hoặc `I`/`1`. Sai dấu thanh đứng thứ hai (khoảng 30%), phù hợp với việc dấu thanh là chi tiết nhỏ nhất trên ảnh dòng chữ độ phân giải thấp. Lặp ký tự không xuất hiện trong mẫu, trái với kiểu lỗi mà đề nêu làm ví dụ. Điều này hợp lý vì sau khi vá backbone, CTC có 120 bước thời gian cho tối đa 80 ký tự nên mô hình không bị dồn ép phải lặp ký tự để lấp bước thời gian. Một phần tư số dòng mẫu thuộc nhóm khác (khoảng trắng, dấu câu, mất cụm chữ); bỏ qua khoảng trắng chỉ nâng acc từ 0,5698 lên 0,5921, tức lỗi khoảng trắng không phải nguyên nhân chính.

Một điểm cần lưu ý khi đọc các con số này là chất lượng nhãn. Đọc bằng mắt trên văn bản, khoảng một nửa số dòng mẫu (2, 4, 5, 7, 13, 14, 15, 16, 17, 18) có nhãn `gt` chứa lỗi OCR gốc của sách quét (ví dụ `Đáng ta … còng nhàn`, `Mỹ vui Việt Nam (1980-1975)`) trong khi `pred` đúng hoặc đúng hơn. Vì vậy acc 0,5698 là cận dưới của chất lượng thật và một phần lỗi "sai chữ cái" thực chất là nhiễu nhãn. Nhận định này dựa trên văn bản, chưa đối chiếu lại với ảnh gốc.
