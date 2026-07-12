"""Compatibility helpers for Hugging Face model layout changes."""

from __future__ import annotations

from typing import Iterable


def _get_attr_path(obj, path: str):
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def _set_attr_path(obj, path: str, value) -> bool:
    parts = path.split(".")
    parent = _get_attr_path(obj, ".".join(parts[:-1])) if len(parts) > 1 else obj
    if parent is None or not hasattr(parent, parts[-1]):
        return False
    setattr(parent, parts[-1], value)
    return True


def get_qwen_vl_layers(model):
    """Return Qwen2/2.5-VL language decoder layers across transformers versions."""
    for path in (
        "model.layers",
        "model.language_model.layers",
        "language_model.layers",
        "language_model.model.layers",
    ):
        layers = _get_attr_path(model, path)
        if layers is not None:
            return layers
    raise AttributeError(
        "Cannot locate Qwen-VL language layers. "
        f"outer={type(model)}, inner={type(getattr(model, 'model', None))}"
    )


def _move_first_existing(model, paths: Iterable[str], device) -> bool:
    for path in paths:
        module = _get_attr_path(model, path)
        if module is not None:
            _set_attr_path(model, path, module.to(device))
            return True
    return False


def move_qwen_vl_embeddings(model, device) -> None:
    """Move Qwen-VL token/rotary/norm modules when they exist."""
    moved = False

    if hasattr(model, "get_input_embeddings"):
        try:
            emb = model.get_input_embeddings()
            if emb is not None:
                emb.to(device)
                moved = True
        except Exception:
            pass

    moved = _move_first_existing(
        model,
        (
            "model.embed_tokens",
            "model.language_model.embed_tokens",
            "language_model.embed_tokens",
            "language_model.model.embed_tokens",
        ),
        device,
    ) or moved

    for paths in (
        ("model.rotary_emb", "model.language_model.rotary_emb", "language_model.rotary_emb", "language_model.model.rotary_emb"),
        ("model.norm", "model.language_model.norm", "language_model.norm", "language_model.model.norm"),
    ):
        _move_first_existing(model, paths, device)

    if not moved:
        raise AttributeError(
            "Cannot locate Qwen-VL input embedding layer. "
            f"outer={type(model)}, inner={type(getattr(model, 'model', None))}"
        )


def move_qwen_vl_rotary(model, device) -> None:
    """Move Qwen-VL rotary embedding modules when they exist."""
    _move_first_existing(
        model,
        (
            "model.rotary_emb",
            "model.language_model.rotary_emb",
            "language_model.rotary_emb",
            "language_model.model.rotary_emb",
        ),
        device,
    )
