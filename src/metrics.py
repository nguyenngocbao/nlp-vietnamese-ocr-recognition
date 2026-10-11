"""Recognition metrics used by notebooks and eval.py."""

from __future__ import annotations
from collections.abc import Iterable
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


def normalized_edit_similarity(gt: str, pred: str) -> float:
    denominator = max(len(gt), len(pred))
    return 1.0 if denominator == 0 else 1.0 - levenshtein_distance(gt, pred) / denominator


def recognition_metrics(rows: Iterable[Mapping[str, str]]) -> dict[str, float | int]:
    rows = list(rows)
    if not rows:
        raise ValueError("Prediction file has no rows.")
    return {
        "num_samples": len(rows),
        "acc": sum(row["gt"] == row["pred"] for row in rows) / len(rows),
        "norm_edit_dis": sum(normalized_edit_similarity(row["gt"], row["pred"]) for row in rows) / len(rows),
    }
