#!/usr/bin/env python3
"""Collect LAT-AWQ and existing QIG/MBQ lmms-eval metrics into one table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


TASK_METRICS = {
    "vizwiz_vqa_val": ("exact_match,none", "VizWiz val"),
    "mmmu_val": ("mmmu_acc,none", "MMMU val"),
    "chartqa": ("relaxed_overall,none", "ChartQA"),
    "ai2d": ("exact_match,flexible-extract", "AI2D"),
    "scienceqa": ("exact_match,none", "ScienceQA"),
    "textvqa_val": ("exact_match,none", "TextVQA val"),
    "ocrbench": ("ocrbench_accuracy,none", "OCRBench"),
    "seedbench": ("seed_image,none", "SEED-Bench image"),
}

BASELINE_METHODS = (
    "fp16",
    "w3a16_awq",
    "w3a16_mbq",
    "w3a16_qig",
    "w4a8_mbq",
    "w4a8_qig",
)


def newest_result(root: Path) -> Path | None:
    paths = list(root.rglob("*_results.json")) if root.is_dir() else []
    return max(paths, key=lambda path: path.stat().st_mtime_ns) if paths else None


def read_task(root: Path, task: str) -> dict[str, Any]:
    path = newest_result(root)
    if path is None:
        return {"status": "MISSING", "score": None, "samples": None, "result_path": None}
    data = json.loads(path.read_text(encoding="utf-8"))
    metric_key, _ = TASK_METRICS[task]
    task_metrics = data.get("results", {}).get(task, {})
    samples = data.get("n-samples", {}).get(task, {}).get("effective")
    score = task_metrics.get(metric_key)
    return {
        "status": "DONE" if score is not None else "INVALID",
        "score": score,
        "samples": samples,
        "result_path": str(path),
    }


def collect_method(method: str, roots: dict[str, Path]) -> dict[str, Any]:
    row: dict[str, Any] = {"method": method, "tasks": {}}
    scores = []
    for task in TASK_METRICS:
        result = read_task(roots[task], task)
        row["tasks"][task] = result
        if isinstance(result["score"], (int, float)):
            scores.append(float(result["score"]))
    row["mean_score"] = sum(scores) / len(scores) if len(scores) == len(TASK_METRICS) else None
    row["status"] = "DONE" if row["mean_score"] is not None else "INCOMPLETE"
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat-root", type=Path, required=True)
    parser.add_argument("--lat-method", default="lat_awq_w4a16_n128_l05")
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        collect_method(
            args.lat_method,
            {task: args.lat_root / task for task in TASK_METRICS},
        )
    ]
    if args.baseline_root:
        for method in BASELINE_METHODS:
            rows.append(
                collect_method(
                    method,
                    {task: args.baseline_root / method / f"eval_{task}" for task in TASK_METRICS},
                )
            )

    payload = {
        "metric_scale": "fraction",
        "primary_metrics": {task: metric for task, (metric, _) in TASK_METRICS.items()},
        "rows": rows,
    }
    for path in (args.output_json, args.output_csv, args.output_md):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = ["method", "status", *TASK_METRICS.keys(), "mean_score"]
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "method": row["method"],
                    "status": row["status"],
                    **{task: row["tasks"][task]["score"] for task in TASK_METRICS},
                    "mean_score": row["mean_score"],
                }
            )

    headers = ["Method", "Status", *[label for _, label in TASK_METRICS.values()], "Mean"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        values = [row["method"], row["status"]]
        for task in TASK_METRICS:
            value = row["tasks"][task]["score"]
            values.append("" if value is None else f"{100.0 * float(value):.2f}")
        mean = row["mean_score"]
        values.append("" if mean is None else f"{100.0 * float(mean):.2f}")
        lines.append("| " + " | ".join(values) + " |")
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"summary_json={args.output_json}")
    print(f"summary_csv={args.output_csv}")
    print(f"summary_md={args.output_md}")


if __name__ == "__main__":
    main()
