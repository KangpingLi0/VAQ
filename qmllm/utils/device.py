"""Small device helpers used by quantization methods."""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn


def module_device(module: nn.Module, default: str | torch.device = "cpu") -> torch.device:
    for tensor in module.parameters(recurse=True):
        return tensor.device
    for tensor in module.buffers(recurse=True):
        return tensor.device
    return torch.device(default)


def object_device(obj: Any, default: str | torch.device | None = "cpu") -> torch.device | None:
    if torch.is_tensor(obj):
        return obj.device
    if isinstance(obj, nn.Module):
        return module_device(obj, default or "cpu")
    if isinstance(obj, dict):
        for value in obj.values():
            device = object_device(value, None)
            if device is not None:
                return device
    if isinstance(obj, (list, tuple)):
        for value in obj:
            device = object_device(value, None)
            if device is not None:
                return device
    return torch.device(default) if default is not None else None


def move_to_device(obj: Any, device: torch.device | str | None = None) -> Any:
    target = torch.device(device) if device is not None else object_device(obj)
    if target is None:
        return obj
    if torch.is_tensor(obj):
        return obj.to(target)
    if isinstance(obj, nn.Module):
        return obj.to(target)
    if isinstance(obj, dict):
        return {key: move_to_device(value, target) for key, value in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(value, target) for value in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(value, target) for value in obj)
    return obj


def slice_batch(obj: Any, start: int, end: int, total: int) -> Any:
    """Slice tensors whose leading dimension matches the calibration batch."""
    if torch.is_tensor(obj):
        if obj.dim() > 0 and obj.shape[0] == total:
            return obj[start:end]
        if obj.dim() > 1 and obj.shape[1] == total:
            return obj[:, start:end]
        return obj
    if isinstance(obj, dict):
        return {key: slice_batch(value, start, end, total) for key, value in obj.items()}
    if isinstance(obj, list):
        return [slice_batch(value, start, end, total) for value in obj]
    if isinstance(obj, tuple):
        return tuple(slice_batch(value, start, end, total) for value in obj)
    return obj


def get_scale_search_batch_size(total: int | None = None, default: int = 8) -> int:
    """Mini-batch size for scale-search forwards over cached calibration chunks."""
    value = os.environ.get("QIG_SCALE_SEARCH_BATCH_SIZE", "").strip()
    try:
        batch_size = int(value) if value else int(default)
    except ValueError:
        batch_size = int(default)
    batch_size = max(1, batch_size)
    if total is not None:
        batch_size = min(batch_size, max(1, int(total)))
    return batch_size


def _channel_view(scales: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if target.dim() <= 1:
        return scales.view(-1)
    return scales.view(*([1] * (target.dim() - 1)), -1)


@torch.no_grad()
def get_act_scale_in_batches(
    x: torch.Tensor,
    batch_size: int | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Compute per-channel activation scale without materializing abs(x) for all chunks."""
    target = torch.device(device) if device is not None else object_device(x, "cpu")
    total = x.shape[0]
    batch_size = get_scale_search_batch_size(total) if batch_size is None else min(max(1, batch_size), total)
    acc = torch.zeros(x.shape[-1], dtype=torch.float32, device=target)
    count = 0

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        mini_x = x[start:end].to(target)
        flat = mini_x.float().abs().reshape(-1, mini_x.shape[-1])
        acc += flat.sum(dim=0)
        count += flat.shape[0]
        del mini_x, flat
        empty_cache(target)

    return acc.div(max(1, count)).to(dtype=x.dtype)


@torch.no_grad()
def reconstruction_loss_in_batches(
    baseline_module: nn.Module,
    baseline_x: torch.Tensor,
    test_x: torch.Tensor | None = None,
    kwargs: dict[str, Any] | None = None,
    test_module: nn.Module | None = None,
    input_scales: torch.Tensor | None = None,
    ans_mask: torch.Tensor | None = None,
    vis_mask: torch.Tensor | None = None,
    token_w: torch.Tensor | None = None,
    reweight_ratio: float | torch.Tensor | None = None,
    loss_mode: str = "mae",
    batch_size: int | None = None,
    device: torch.device | str | None = None,
) -> float:
    """Forward decoder blocks in calibration mini-batches and aggregate reconstruction loss."""
    target = torch.device(device) if device is not None else module_device(baseline_module)
    test_module = test_module or baseline_module
    test_x = baseline_x if test_x is None else test_x
    kwargs = kwargs or {}
    total = baseline_x.shape[0]
    batch_size = get_scale_search_batch_size(total) if batch_size is None else min(max(1, batch_size), total)
    loss_mode = (loss_mode or "mae").lower()

    num = 0.0
    den = 0.0
    ans_num = 0.0
    ans_den = 0.0
    vis_num = 0.0
    vis_den = 0.0
    weighted_mse_masks = (
        token_w is None
        and loss_mode == "mse"
        and ans_mask is not None
        and vis_mask is not None
        and reweight_ratio is not None
    )

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        mini_kwargs = move_to_device(slice_batch(kwargs, start, end, total), target)
        base_in = baseline_x[start:end].to(target)
        test_in = test_x[start:end].to(target)
        if input_scales is not None:
            scales = input_scales.to(device=target, dtype=test_in.dtype)
            test_in = test_in / _channel_view(scales, test_in)

        org_out = baseline_module(base_in, **mini_kwargs)
        out = test_module(test_in, **mini_kwargs)
        if isinstance(org_out, tuple):
            org_out = org_out[0]
        if isinstance(out, tuple):
            out = out[0]

        if loss_mode == "mse":
            diff = (org_out - out).float().pow(2)
        else:
            diff = (org_out - out).float().abs()

        if token_w is not None:
            mini_w = move_to_device(slice_batch(token_w, start, end, total), target).float()
            pos_change = diff.mean(dim=-1)
            num += float((pos_change * mini_w).sum().detach().cpu())
            den += float(mini_w.sum().detach().cpu())
        elif ans_mask is not None and vis_mask is not None:
            ans = move_to_device(slice_batch(ans_mask, start, end, total), target).unsqueeze(-1).expand_as(diff).float()
            vis = move_to_device(slice_batch(vis_mask, start, end, total), target).unsqueeze(-1).expand_as(diff).float()
            if reweight_ratio is not None:
                ratio = float(reweight_ratio.detach().cpu()) if torch.is_tensor(reweight_ratio) else float(reweight_ratio)
                if weighted_mse_masks:
                    ans_num += float((diff * ans).sum().detach().cpu())
                    ans_den += float(ans.sum().detach().cpu())
                    vis_num += float((diff * vis).sum().detach().cpu())
                    vis_den += float(vis.sum().detach().cpu())
                else:
                    num += float(((diff * ans).sum() + ratio * (diff * vis).sum()).detach().cpu())
                    den += float((ans.sum() + vis.sum()).detach().cpu())
            else:
                num += float(diff.sum().detach().cpu())
                den += float(diff.numel())
        elif ans_mask is not None:
            ans = move_to_device(slice_batch(ans_mask, start, end, total), target).unsqueeze(-1).expand_as(diff).float()
            num += float((diff * ans).sum().detach().cpu())
            den += float(ans.sum().detach().cpu())
        else:
            num += float(diff.sum().detach().cpu())
            den += float(diff.numel())

        del mini_kwargs, base_in, test_in, org_out, out, diff
        empty_cache(target)

    if weighted_mse_masks:
        ans_loss = ans_num / max(ans_den, 1e-12)
        vis_loss = vis_num / max(vis_den, 1e-12)
        ratio = float(reweight_ratio.detach().cpu()) if torch.is_tensor(reweight_ratio) else float(reweight_ratio)
        return ans_loss + ratio * vis_loss

    return num / max(den, 1e-12)


@torch.no_grad()
def forward_module_in_batches(
    module: nn.Module,
    inps: torch.Tensor,
    kwargs: dict[str, Any] | None = None,
    batch_size: int = 1,
    device: torch.device | str | None = None,
    output_index: int | None = 0,
    collect_output: bool = True,
    desc: str | None = None,
) -> torch.Tensor | None:
    """Forward a decoder block over calibration rows without one huge activation."""
    target = torch.device(device) if device is not None else module_device(module)
    kwargs = kwargs or {}
    total = inps.shape[0]
    outputs = []

    iterator = range(0, total, batch_size)
    if desc:
        try:
            import tqdm

            iterator = tqdm.tqdm(iterator, desc=desc)
        except Exception:
            pass

    for start in iterator:
        end = min(start + batch_size, total)
        mini_inps = inps[start:end].to(target)
        mini_kwargs = move_to_device(slice_batch(kwargs, start, end, total), target)
        mini_out = module(mini_inps, **mini_kwargs)
        if collect_output:
            if output_index is not None:
                mini_out = mini_out[output_index]
            outputs.append(mini_out.detach().cpu())
        del mini_inps, mini_kwargs, mini_out
        empty_cache(target)

    if collect_output:
        return torch.cat(outputs, dim=0)
    return None


def empty_cache(device: torch.device | str | Any | None = None) -> None:
    resolved = object_device(device, None) if not isinstance(device, (str, torch.device, type(None))) else device
    if resolved is None:
        return
    resolved = torch.device(resolved)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif resolved.type == "npu" and hasattr(torch, "npu"):
        torch.npu.empty_cache()


def synchronize(device: torch.device | str | Any | None = None) -> None:
    resolved = object_device(device, None) if not isinstance(device, (str, torch.device, type(None))) else device
    if resolved is None:
        return
    resolved = torch.device(resolved)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif resolved.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize()


def accelerator_device_count(device: torch.device | str | None = None) -> int:
    if device is not None:
        resolved = torch.device(device)
        if resolved.type == "npu" and hasattr(torch, "npu") and torch.npu.is_available():
            return torch.npu.device_count()
        if resolved.type == "cuda" and torch.cuda.is_available():
            return torch.cuda.device_count()
        return 0
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.npu.device_count()
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0
