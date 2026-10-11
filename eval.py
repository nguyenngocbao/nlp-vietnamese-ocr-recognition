#!/usr/bin/env python3
"""Evaluate a prediction JSONL file."""

import argparse
import json
from pathlib import Path
from src.jsonl import read_predictions
from src.metrics import recognition_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--predictions", required=True, type=Path)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
metrics = recognition_metrics(read_predictions(args.predictions))
print(json.dumps(metrics, ensure_ascii=False, indent=2))
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
