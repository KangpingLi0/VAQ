"""Numerically safe token-aware channel saliency for LAT-AWQ."""

from __future__ import annotations

import torch


EPS = 1e-8


def _validate_3d_input(x: torch.Tensor) -> None:
    if not torch.is_tensor(x) or x.ndim != 3:
        raise ValueError(
            f"LAT-AWQ expects X=[B,T,C], got {type(x)!r} "
            f"shape={getattr(x, 'shape', None)}"
        )


@torch.no_grad()
def get_act_scale(
    x: torch.Tensor,
    token_w: torch.Tensor | None = None,
    mix_lambda: float = 0.0,
):
    """Return global, importance-weighted and mixed FP32 channel saliency."""
    _validate_3d_input(x)
    if not 0.0 <= float(mix_lambda) <= 1.0:
        raise ValueError(f"saliency_mix_lambda must be in [0, 1], got {mix_lambda}")

    x_abs = torch.nan_to_num(
        x.detach().float(), nan=0.0, posinf=0.0, neginf=0.0
    ).abs()
    a_global = x_abs.reshape(-1, x.shape[-1]).mean(0)
    if token_w is None:
        a_imp = a_global.clone()
    else:
        if token_w.ndim != 2 or tuple(token_w.shape) != tuple(x.shape[:2]):
            raise ValueError(
                f"token_w must be [B,T]={tuple(x.shape[:2])}, got {tuple(token_w.shape)}"
            )
        weights = token_w.detach().to(device=x_abs.device, dtype=torch.float32)
        if not torch.isfinite(weights).all() or float(weights.sum()) <= EPS:
            a_imp = a_global.clone()
        else:
            weights = torch.where(
                torch.isfinite(weights), weights, torch.zeros_like(weights)
            ).clamp_min_(0)
            a_imp = (x_abs * weights.unsqueeze(-1)).sum(dim=(0, 1))
            a_imp = a_imp / weights.sum().clamp_min(EPS)
            if not torch.isfinite(a_imp).all():
                a_imp = a_global.clone()

    a_final = (1.0 - float(mix_lambda)) * a_global + float(mix_lambda) * a_imp
    if not torch.isfinite(a_final).all():
        raise FloatingPointError("LAT-AWQ produced non-finite channel saliency")
    return a_global, a_imp, a_final
