"""Recognition metrics used by notebooks and eval.py.

Đây là bản chuẩn để đánh giá mọi mô hình (baseline và fine-tune). Nếu sửa file này,
phải sửa hàm chép lại trong các notebook cho khớp.

Mọi so sánh đều chuẩn hóa Unicode về NFC trước, vì tiếng Việt có hai cách mã hóa
cùng một chữ (dựng sẵn và tổ hợp) mà nhìn giống hệt nhau.
  - acc, norm_edit_dis: giữ khoảng trắng (số liệu chính của báo cáo).
  - acc_ignore_space, norm_edit_dis_ignore_space: bỏ mọi khoảng trắng (số liệu phụ,
    để đối chiếu với các nhóm/công cụ bỏ khoảng trắng như RecMetric mặc định).
"""

from __future__ import annotations
from collections.abc import Iterable
import unicodedata
from typing import Mapping

try:
    from Levenshtein import distance as levenshtein_distance
except ImportError:
    def levenshtein_distance(left: str, right: str) -> int:
        rows = list(range(len(right) + 1))
        for index, char in enumerate(left, 1):
            next_rows = [index]
            for right_index, right_char in enumerate(right, 1):
                next_rows.append(min(rows[right_index] + 1, next_rows[-1] + 1, rows[right_index - 1] + (char != right_char)))
            rows = next_rows
        return rows[-1]


def normalize_text(text: str) -> str:
    """Chuẩn hóa Unicode về NFC; không đổi hoa/thường và không đụng khoảng trắng."""
    return unicodedata.normalize("NFC", text)


def remove_spaces(text: str) -> str:
    """Bỏ mọi khoảng trắng (space, tab, ...)."""
    return "".join(text.split())


def normalized_edit_similarity(gt: str, pred: str) -> float:
    denominator = max(len(gt), len(pred))
    return 1.0 if denominator == 0 else 1.0 - levenshtein_distance(gt, pred) / denominator


def recognition_metrics(rows: Iterable[Mapping[str, str]]) -> dict[str, float | int]:
    rows = list(rows)
    if not rows:
        raise ValueError("Prediction file has no rows.")
    pairs = [(normalize_text(row["gt"]), normalize_text(row["pred"])) for row in rows]
    pairs_no_space = [(remove_spaces(gt), remove_spaces(pred)) for gt, pred in pairs]
    total = len(pairs)
    return {
        "num_samples": total,
        "acc": sum(gt == pred for gt, pred in pairs) / total,
        "norm_edit_dis": sum(normalized_edit_similarity(gt, pred) for gt, pred in pairs) / total,
        "acc_ignore_space": sum(gt == pred for gt, pred in pairs_no_space) / total,
        "norm_edit_dis_ignore_space": sum(normalized_edit_similarity(gt, pred) for gt, pred in pairs_no_space) / total,
    }
