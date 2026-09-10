#!/usr/bin/env python3
"""Create a deterministic, right-trimmed calibration-cache subset."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--indices", default="0,2")
    args = parser.parse_args()

    indices = [int(item) for item in args.indices.split(",") if item.strip()]
    if not indices:
        raise ValueError("At least one calibration index is required")
    cache = torch.load(args.source, map_location="cpu")
    attention_mask = cache["prompt_kwargs"]["attention_mask"][indices]
    nonzero = (attention_mask > 0).nonzero(as_tuple=False)
    if nonzero.numel() == 0:
        raise ValueError("Selected calibration samples contain no valid tokens")
    max_token = int(nonzero[:, 1].max()) + 1

    def select(value):
        if not torch.is_tensor(value):
            return value
        selected = value[indices].clone().contiguous()
        if selected.ndim >= 2 and selected.shape[1] == attention_mask.shape[1]:
            selected = selected[:, :max_token].clone().contiguous()
        return selected

    subset = {
        "prompt_inputs": {key: select(value) for key, value in cache["prompt_inputs"].items()},
        "prompt_kwargs": {key: select(value) for key, value in cache["prompt_kwargs"].items()},
        "meta": {
            **cache.get("meta", {}),
            "n_samples": len(indices),
            "lat_awq_source": str(args.source),
            "lat_awq_indices": indices,
            "lat_awq_sequence_length": max_token,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(subset, args.output)
    print(
        f"Saved {len(indices)} samples with sequence length {max_token} "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
