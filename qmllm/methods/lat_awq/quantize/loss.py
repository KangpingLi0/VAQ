"""Strict token-weighted reconstruction objective for LAT-AWQ."""

from __future__ import annotations

import torch
import torch.nn as nn

from qmllm.methods.lat_awq.quantize.saliency import EPS
from qmllm.utils.device import empty_cache, move_to_device, slice_batch


@torch.no_grad()
def token_weighted_mse_in_batches(
    baseline_module: nn.Module,
    baseline_x: torch.Tensor,
    test_module: nn.Module,
    token_w: torch.Tensor,
    kwargs: dict | None = None,
    batch_size: int = 1,
    device: torch.device | str | None = None,
) -> float:
    """Compute sum(I*mean_C((Y_fp-Y_q)^2))/sum(I) with strict shapes."""
    if baseline_x.ndim != 3:
        raise ValueError(f"baseline_x must be [B,T,C], got {tuple(baseline_x.shape)}")
    if token_w.ndim != 2 or tuple(token_w.shape) != tuple(baseline_x.shape[:2]):
        raise ValueError(
            f"token_w must be [B,T]={tuple(baseline_x.shape[:2])}, "
            f"got {tuple(token_w.shape)}"
        )
    if not torch.isfinite(token_w).all():
        raise FloatingPointError("token_w contains NaN or Inf")

    target = torch.device(device) if device is not None else baseline_x.device
    kwargs = kwargs or {}
    total = baseline_x.shape[0]
    batch_size = min(max(1, int(batch_size)), total)
    numerator = 0.0
    denominator = 0.0

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        mini_x = torch.nan_to_num(
            baseline_x[start:end].to(target),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        mini_kwargs = move_to_device(slice_batch(kwargs, start, end, total), target)
        y_fp = baseline_module(mini_x, **mini_kwargs)
        y_q = test_module(mini_x, **mini_kwargs)
        if isinstance(y_fp, tuple):
            y_fp = y_fp[0]
        if isinstance(y_q, tuple):
            y_q = y_q[0]
        if y_fp.ndim != 3 or y_q.ndim != 3 or tuple(y_fp.shape) != tuple(y_q.shape):
            raise ValueError(
                f"LAT-AWQ requires matching Y_fp,Y_q=[B,T,C], got "
                f"{tuple(y_fp.shape)} and {tuple(y_q.shape)}"
            )
        mini_w = token_w[start:end].to(target, dtype=torch.float32)
        if tuple(mini_w.shape) != tuple(y_fp.shape[:2]):
            raise ValueError(
                f"token_w batch shape {tuple(mini_w.shape)} does not match "
                f"output tokens {tuple(y_fp.shape[:2])}"
            )
        # Attention kernels can emit non-finite values at fully masked/padded
        # positions.  Those tokens carry no calibration weight and must not
        # poison the complete scale candidate.  Exclude every token whose FP
        # or quantized output is non-finite, then compute the error from safe
        # values.  The final denominator check still rejects a batch/run with
        # no usable weighted tokens.
        finite_tokens = torch.isfinite(y_fp).all(dim=-1) & torch.isfinite(y_q).all(dim=-1)
        mini_w = torch.where(finite_tokens, mini_w, torch.zeros_like(mini_w))
        y_fp = torch.nan_to_num(y_fp.float(), nan=0.0, posinf=0.0, neginf=0.0)
        y_q = torch.nan_to_num(y_q.float(), nan=0.0, posinf=0.0, neginf=0.0)
        token_error = (y_fp - y_q).pow(2).mean(dim=-1)
        if not torch.isfinite(token_error).all():
            raise FloatingPointError(
                "per-token reconstruction error remains non-finite after masking"
            )
        numerator += float((token_error * mini_w).sum().cpu())
        denominator += float(mini_w.sum().cpu())
        del mini_x, mini_kwargs, y_fp, y_q, mini_w, token_error
        empty_cache(target)

    if denominator <= EPS:
        raise FloatingPointError("token_w.sum() is zero after sanitization")
    loss = numerator / denominator
    if not torch.isfinite(torch.tensor(loss)):
        raise FloatingPointError("token-weighted reconstruction loss is not finite")
    return loss
