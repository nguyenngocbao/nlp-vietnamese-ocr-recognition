# Mục 2.5 — Phân tích 20 dòng nhận diện sai (mô hình fine-tune 4 epoch)

Nguồn: `results/ft_4epochs/pred_ppocrv5_mobile_4epochs.jsonl`. Trên 3.003 dòng test có 1.292 dòng sai (so khớp chính xác sau NFC, có tính khoảng trắng). 20 dòng được chọn ngẫu nhiên từ 1.292 dòng sai bằng `random.seed(42)` (`scripts/error_analysis.py`), sau đó xếp nhóm bằng tay.

Ba nhóm lỗi: (i) sai hoặc mất dấu/thanh tiếng Việt, chữ cái gốc đúng; (ii) sai khoảng trắng; (iii) lỗi còn lại, gồm nhầm ký tự, thừa/thiếu ký tự hoặc dấu câu, mất cả cụm chữ.

## Bảng 20 dòng

| # | Nhóm | Nhãn (gt) | Mô hình đọc (pred) | Ghi chú |
|---|---|---|---|---|
| 1 | iii | …có “chính” chứ không | …có “chính”) chứ không | thừa dấu `)` |
| 2 | i | Tròi dát bī ròi lai thai 1. | Trời dất bĩ rồi lại thái 1. | nhãn nhiễu, pred đúng hơn |
| 3 | i | Bộ binh phải bắt 300 tên biển | Bộ binh phái bắt 300 tên biền | pred sai thanh |
| 4 | i | Đáng ta chỉ đạo còng nhàn | Đảng ta chỉ đạo công nhân | nhãn nhiễu, pred đúng |
| 5 | i | tuy nhiêu … các quyền trúng | tuy nhiều … các quyển trúng | nhãn nhiễu |
| 6 | iii | …lũng bên cạnh Mã Ấp, | …náu ở trong | mất cụm cuối dòng |
| 7 | i | …ngồi thay vong / phía ta đằng trước | …phía tả đằng trước | nhãn nhiễu |
| 8 | iii | tháng II, … chỉ huy | tháng 1I, … chỉ-huy | nhầm `I`/`1`, thêm gạch nối |
| 9 | i | năm Bính | năm Binh- | mất dấu sắc (gạch nối cuối dòng không có trong nhãn) |
| 10 | iii | CHÚ GIẢI | CHÔ GIẢI | nhầm chữ hoa Ú/Ô |
| 11 | iii | – Quận công nha 21 … | Quận công nha 21 … - Hoàng … | mất gạch đầu dòng, đổi `–`/`-` |
| 12 | ii | Quân ngũ: nam | Quân ngũ : nam | thừa khoảng trắng trước `:` |
| 13 | iii | Việt sứ liêu án, Bảnđánh máy cùa | Việt sử tiêu án, Bản đánh máy của … | nhãn nhiễu, nhiều loại lỗi |
| 14 | i | đánh hãng. Nác Thu | đánh hăng. Nặc Thu | nhãn nhiễu |
| 15 | iii | Nhận là … phách lợi | nhận là … phúc lợi | sai chữ hoa/thường, nhãn nhiễu |
| 16 | iii | Túr suong chi huy sur. | Tú suong chỉ huy sú. | dòng ngắn, nhãn nhiễu |
| 17 | iii | Mỹ vui Việt Nam (1980-1975), Ninh Công an | Mỹ và Việt Nam (1950-1975), Nxb. Công an … 2004. | nhãn nhiễu, pred đúng |
| 18 | iii | dều dạ rau cả lên | đều dạ ran cả lên | nhãn nhiễu |
| 19 | iii | giữ công việc hai ti, | giữ công việc hai ti', | thừa dấu `'` |
| 20 | i | « thứ II), sứ Tàu | « thứ II), sử Tàu | pred sai thanh (đúng là "sứ") |

## Thống kê 20 dòng

| Nhóm | Số dòng | Tỷ lệ |
|---|---:|---:|
| (i) sai hoặc mất dấu/thanh | 8 | 40% |
| (ii) sai khoảng trắng | 1 | 5% |
| (iii) nhầm ký tự, thừa/thiếu ký tự hoặc dấu câu, mất cụm chữ | 11 | 55% |

Phân loại tự động thô (cùng quy tắc, chạy trên toàn bộ 1.292 dòng sai) cho (i) 41,3%, (ii) 5,2%, (iii) 53,6%, gần trùng với mẫu 20 dòng, nên mẫu này khá đại diện.

## Nhận xét

Lỗi chiếm đa số là nhóm (iii), nhưng nhìn vào từng dòng thì phần lớn là nhầm ký tự nhỏ (`I`/`1`, `Ú`/`Ô`, `u`/`n`), dấu câu thừa hoặc thiếu (`)`, `'`, gạch đầu dòng) và chỉ một dòng (số 6) bị mất cả cụm chữ cuối dòng. Nhóm (i) đứng thứ hai (40%) và là lỗi đặc thù tiếng Việt: mô hình đọc đúng chữ cái gốc nhưng sai thanh (phải/phái, sử/sứ), phù hợp với việc từ điển có 233 ký tự mà nhiều ký tự mang dấu rất giống nhau. Lỗi khoảng trắng (ii) hiếm (5%), nên bỏ qua khoảng trắng chỉ nâng acc từ 0,5698 lên 0,5921.

Điểm đáng chú ý nhất là chất lượng nhãn: đọc bằng mắt trên văn bản, khoảng một nửa số dòng (2, 4, 5, 7, 13, 14, 15, 16, 17, 18) là trường hợp nhãn `gt` chứa lỗi OCR gốc của sách quét (ví dụ `Đáng ta … còng nhàn`, `Mỹ vui Việt Nam (1980-1975)`) trong khi `pred` lại đúng hoặc đúng hơn. Vì vậy acc 0,5698 là cận dưới của chất lượng thật, và một phần "lỗi" thực chất là nhiễu nhãn. Cần lưu ý đây là nhận định dựa trên văn bản, chưa đối chiếu lại với ảnh gốc.
