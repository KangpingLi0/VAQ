#!/usr/bin/env python3
"""Aggregate LAT-AWQ per-group JSONL diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def _mean(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.fmean(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with args.jsonl.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"No LAT-AWQ diagnostics in {args.jsonl}")

    for row in rows:
        for key in ("best_loss", "scale_min", "scale_max"):
            if not math.isfinite(float(row[key])):
                raise ValueError(f"Non-finite {key} in {args.jsonl}: {row}")

    summary = {
        "debug_path": str(args.jsonl),
        "groups": len(rows),
        "layers": len({int(row["layer_idx"]) for row in rows}),
        "mean_best_loss": _mean(rows, "best_loss"),
        "median_best_loss": statistics.median(float(row["best_loss"]) for row in rows),
        "mean_selected_alpha": _mean(rows, "selected_alpha"),
        "mean_saliency_cosine": _mean(rows, "saliency_cosine"),
        "mean_saliency_rank_cosine": _mean(rows, "saliency_rank_cosine"),
        "mean_saliency_relative_l1": _mean(rows, "saliency_relative_l1"),
        "groups_with_saliency_change": sum(
            float(row.get("saliency_relative_l1", 0.0)) > 1e-6 for row in rows
        ),
        "selected_alpha": [
            {
                "layer_idx": row["layer_idx"],
                "group": row["group"],
                "alpha": row["selected_alpha"],
                "loss": row["best_loss"],
            }
            for row in rows
        ],
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

