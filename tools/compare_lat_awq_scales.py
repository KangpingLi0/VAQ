#!/usr/bin/env python3
"""Compare two AWQ-compatible scale caches by module name."""

from __future__ import annotations

import argparse
import json

import torch


def load(path):
    payload = torch.load(path, map_location="cpu")
    return {
        (prev, tuple(layers)): scale.float()
        for prev, layers, scale in payload["scale"]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    args = parser.parse_args()
    reference = load(args.reference)
    candidate = load(args.candidate)
    common = sorted(reference.keys() & candidate.keys())
    if not common:
        raise ValueError("Scale caches have no common groups")
    rel = []
    exact = 0
    for key in common:
        left, right = reference[key], candidate[key]
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch for {key}: {left.shape} vs {right.shape}")
        exact += int(torch.equal(left, right))
        rel.append(float((left - right).abs().mean() / left.abs().mean().clamp_min(1e-8)))
    print(json.dumps({
        "reference_groups": len(reference),
        "candidate_groups": len(candidate),
        "common_groups": len(common),
        "exact_groups": exact,
        "mean_relative_l1": sum(rel) / len(rel),
        "max_relative_l1": max(rel),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

