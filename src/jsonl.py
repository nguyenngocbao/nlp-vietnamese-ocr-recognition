"""Read assignment-format prediction JSONL files."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Iterator

REQUIRED_FIELDS = {"image", "gt", "pred"}


def read_predictions(path: str | Path) -> Iterator[dict[str, str]]:
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = REQUIRED_FIELDS - row.keys()
            if missing:
                raise ValueError(f"Line {line_number} missing: {sorted(missing)}")
            yield {key: str(row[key]) for key in REQUIRED_FIELDS}
