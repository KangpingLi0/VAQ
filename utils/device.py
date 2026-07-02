"""Device helpers for CUDA, Ascend NPU, and CPU execution."""

from __future__ import annotations

from typing import Any

import torch

try:  # noqa: F401 - importing registers the NPU backend on torch.
    import torch_npu  # type: ignore
except Exception:  # pragma: no cover - depends on the runtime environment.
    torch_npu = None


def _npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def get_device(device_arg: Any = "auto") -> torch.device:
    """Resolve auto/npu/cuda/cpu into a concrete torch.device."""
    if isinstance(device_arg, torch.device):
        return device_arg

    requested = "auto" if device_arg is None else str(device_arg).lower()
    if requested in {"auto", ""}:
        if _npu_available():
            return torch.device("npu:0")
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    if requested.startswith("npu"):
        if not _npu_available():
            raise RuntimeError("Ascend NPU requested, but torch_npu is not available.")
        return torch.device(requested if ":" in requested else "npu:0")

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda is not available.")
        return torch.device(requested if ":" in requested else "cuda:0")

    if requested == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device: {device_arg}")


def move_to_device(obj: Any, device: torch.device) -> Any:
    """Recursively move tensors inside common Python containers."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(value, device) for value in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(value, device) for value in obj)
    return obj


def empty_cache(device: torch.device | str | None = None) -> None:
    """Clear accelerator cache when the backend supports it."""
    resolved = get_device(device or "auto")
    if resolved.type == "npu" and hasattr(torch, "npu"):
        torch.npu.empty_cache()
    elif resolved.type == "cuda":
        torch.cuda.empty_cache()
