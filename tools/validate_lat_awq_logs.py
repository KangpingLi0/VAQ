#!/usr/bin/env python3
"""Validate completed LAT-AWQ diagnostics and emit a machine-readable report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {}

    for path in sorted(args.log_dir.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if not rows:
            continue
        for row in rows:
            if len(row["x_shape"]) != 3:
                raise ValueError(f"Expected X=[B,T,C] in {path}: {row['x_shape']}")
            if row["token_w_shape"] is not None and row["token_w_shape"] != row["x_shape"][:2]:
                raise ValueError(f"Token shape mismatch in {path}: {row}")
            if abs(row["selected_alpha"] * 20 - round(row["selected_alpha"] * 20)) > 1e-6:
                raise ValueError(f"Alpha is off the 0.05 grid in {path}: {row['selected_alpha']}")
            if not 0 <= row["selected_alpha"] <= 0.95:
                raise ValueError(f"Alpha outside [0,.95] in {path}: {row['selected_alpha']}")
            for key in ("best_loss", "scale_min", "scale_max", "a_global_mean", "a_imp_mean"):
                if not math.isfinite(float(row[key])):
                    raise ValueError(f"Non-finite {key} in {path}")
        token_rows = [row for row in rows if row["token_w_shape"] is not None]
        report[path.stem] = {
            "groups": len(rows),
            "layers": len({row["layer_idx"] for row in rows}),
            "all_shapes_valid": True,
            "all_finite": True,
            "all_alphas_on_grid": True,
            "groups_with_changed_saliency": sum(
                float(row["saliency_relative_l1"]) > 1e-6 for row in rows
            ),
            "unique_token_argmax_patterns": len(
                {tuple(row["token_w_argmax"]) for row in token_rows}
            ),
            "unique_token_l2_rounded_8dp": len(
                {round(float(row["token_w_l2"]), 8) for row in token_rows}
            ),
        }

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

