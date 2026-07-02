#!/usr/bin/env python3
"""Collect QIG experiment metrics into CSV and Markdown summaries."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_ROOT = Path("/work/model/lkp/experiments/qig")
COLUMNS = [
    "model",
    "method",
    "w_bit",
    "a_bit",
    "calib_size",
    "eval_size",
    "dataset",
    "metric",
    "score",
    "scale_path",
    "runtime",
    "source",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Experiment root to scan.")
    parser.add_argument("--out", type=Path, default=None, help="CSV output path. Defaults to ROOT/summary.csv.")
    return parser.parse_args()


def safe_text(path: Path, limit: int = 2_000_000) -> str:
    data = path.read_bytes()[:limit]
    return data.decode("utf-8", errors="replace")


def numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def clean_score(value: Any) -> Optional[str]:
    if numeric(value):
        return f"{float(value):.8g}"
    if isinstance(value, str):
        try:
            return f"{float(value.strip()):.8g}"
        except ValueError:
            return None
    return None


def infer_common_from_path(path: Path) -> Dict[str, str]:
    text = str(path).lower()
    row: Dict[str, str] = {}

    for model in ("qwen2vl", "qwen2_vl", "qwen25vl", "qwen2_5_vl", "qwen3vl", "qwen3_vl", "internvl2", "llava_onevision"):
        if model in text:
            row["model"] = model
            break

    for method in ("qig", "mbq", "awq", "smoothquant", "rtn", "gptq"):
        if method in text:
            row["method"] = method
            break

    match = re.search(r"w(\d+)a(\d+)", text)
    if match:
        row["w_bit"], row["a_bit"] = match.groups()

    if "scale" in text and path.suffix in {".pt", ".pth"}:
        row["scale_path"] = str(path)

    return row


def update_from_mapping(row: Dict[str, str], mapping: Dict[str, Any]) -> None:
    aliases = {
        "model": ("model", "arch", "model_name"),
        "method": ("method",),
        "w_bit": ("w_bit", "weight_bit"),
        "a_bit": ("a_bit", "activation_bit"),
        "calib_size": ("calib_size", "n_samples", "n_calib"),
        "eval_size": ("eval_size", "limit", "n_eval"),
        "dataset": ("dataset", "task", "tasks"),
        "scale_path": ("scale_path",),
        "runtime": ("runtime", "elapsed", "elapsed_time", "total_time"),
    }
    for dst, keys in aliases.items():
        for key in keys:
            if key in mapping and mapping[key] not in (None, ""):
                row.setdefault(dst, str(mapping[key]))
                break


def flatten_numeric(prefix: str, value: Any) -> Iterable[tuple[str, Any]]:
    if numeric(value):
        yield prefix, value
    elif isinstance(value, dict):
        for key, nested in value.items():
            key_text = str(key)
            if key_text in {"samples", "sample_logs"}:
                continue
            next_prefix = f"{prefix}.{key_text}" if prefix else key_text
            yield from flatten_numeric(next_prefix, nested)


def rows_from_json(path: Path) -> List[Dict[str, str]]:
    base = infer_common_from_path(path)
    base["source"] = str(path)
    try:
        data = json.loads(safe_text(path))
    except json.JSONDecodeError:
        return []

    rows: List[Dict[str, str]] = []
    if isinstance(data, dict):
        update_from_mapping(base, data)
        for key in ("meta", "config", "quant"):
            if isinstance(data.get(key), dict):
                update_from_mapping(base, data[key])
        if isinstance(data.get("meta"), dict) and isinstance(data["meta"].get("quant"), dict):
            update_from_mapping(base, data["meta"]["quant"])

        metric_root = data.get("results", data)
        for metric, score in flatten_numeric("", metric_root):
            if metric.startswith("config.") or metric.startswith("meta."):
                continue
            row = dict(base)
            parts = metric.split(".")
            if len(parts) >= 2:
                row.setdefault("dataset", parts[0])
                row["metric"] = ".".join(parts[1:])
            else:
                row["metric"] = metric
            row["score"] = clean_score(score) or str(score)
            rows.append(row)

        if not rows and isinstance(data.get("results"), list):
            row = dict(base)
            row["metric"] = "num_results"
            row["score"] = str(len(data["results"]))
            rows.append(row)

    elif isinstance(data, list):
        row = dict(base)
        row["metric"] = "num_items"
        row["score"] = str(len(data))
        rows.append(row)

    return rows


def rows_from_csv(path: Path) -> List[Dict[str, str]]:
    base = infer_common_from_path(path)
    base["source"] = str(path)
    rows: List[Dict[str, str]] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            for item in reader:
                row = dict(base)
                update_from_mapping(row, item)
                metric = item.get("metric") or item.get("Metric") or item.get("name")
                score = item.get("score") or item.get("Score") or item.get("value")
                if metric and clean_score(score) is not None:
                    row["metric"] = str(metric)
                    row["score"] = clean_score(score) or ""
                    rows.append(row)
    except csv.Error:
        return []
    return rows


KEY_VALUE_RE = re.compile(r"\b(model|method|w_bit|a_bit|n_samples|calib_size|eval_size|dataset|tasks|scale_path|runtime)\s*[:=]\s*([^\s,;]+)")
METRIC_RE = re.compile(r"^\s*([A-Za-z0-9_./ -]{2,80})\s*[:=]\s*(-?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s*$", re.IGNORECASE)


def rows_from_text(path: Path) -> List[Dict[str, str]]:
    base = infer_common_from_path(path)
    base["source"] = str(path)
    text = safe_text(path)
    found = dict(KEY_VALUE_RE.findall(text))
    update_from_mapping(base, found)
    rows: List[Dict[str, str]] = []
    for line in text.splitlines():
        match = METRIC_RE.match(line)
        if not match:
            continue
        metric, score = match.groups()
        metric = metric.strip()
        if metric.lower() in {"epoch", "step", "rank", "world_size"}:
            continue
        row = dict(base)
        row["metric"] = metric
        row["score"] = clean_score(score) or score
        rows.append(row)
    return rows


def collect(root: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not root.exists():
        return rows
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".json":
            rows.extend(rows_from_json(path))
        elif suffix == ".csv":
            if path.name != "summary.csv":
                rows.extend(rows_from_csv(path))
        elif suffix in {".txt", ".log", ".md"}:
            rows.extend(rows_from_text(path))
    return rows


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in COLUMNS})


def write_markdown(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# QIG Result Summary\n\n")
        if not rows:
            handle.write("No metrics found.\n")
            return
        handle.write("| " + " | ".join(COLUMNS) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(COLUMNS)) + " |\n")
        for row in rows:
            values = [str(row.get(column, "")).replace("|", "\\|") for column in COLUMNS]
            handle.write("| " + " | ".join(values) + " |\n")


def main() -> None:
    args = parse_args()
    out_csv = args.out or args.root / "summary.csv"
    out_md = out_csv.with_suffix(".md")
    rows = collect(args.root)
    write_csv(out_csv, rows)
    write_markdown(out_md, rows)
    print(f"[OK] wrote {len(rows)} rows to {out_csv}")
    print(f"[OK] wrote markdown summary to {out_md}")


if __name__ == "__main__":
    main()
