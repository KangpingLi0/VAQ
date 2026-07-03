"""Small device helpers used by quantization methods."""

from __future__ import annotations

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
