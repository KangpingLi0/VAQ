import gc
import copy
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bloom.modeling_bloom import BloomBlock, BloomGelu
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.activations import GELUActivation
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm  # Used in apply_scale RMSNorm branch

from qmllm.utils.search import get_op_by_name, get_op_name, set_op_by_name
from qmllm.quantization.quant_funcs import pseudo_quantize_tensor
from qmllm.quantization.qlinear import WALinear
from qmllm.methods.lat_awq.quantize.loss import token_weighted_mse_in_batches
from qmllm.methods.lat_awq.quantize.saliency import EPS, get_act_scale
from qmllm.utils.device import (
    empty_cache,
    get_act_scale_in_batches,
    get_scale_search_batch_size,
    move_to_device,
    normalized_power_scales,
    reconstruction_loss_in_batches,
    slice_batch,
)

__all__ = ["auto_scale_block", "apply_scale"]


@torch.no_grad()
def scale_ln_fcs(ln: nn.Module, fcs, scales: torch.Tensor):
    """
    Apply channel-wise scaling across (Norm -> Linear(s)):
    - Divide LN/RMSNorm weights (and bias) by scales
    - Multiply following Linear weights by scales
    """
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device)

    ln.weight.div_(scales)
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    # NaN safety checks
    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_fc_fc(fc1: nn.Linear, fc2: nn.Linear, scales: torch.Tensor):
    """
    Apply channel-wise scaling across (Linear -> Linear).
    Note: keeps the original behavior of only scaling the last `scales.size(0)` rows of fc1.
    """
    assert isinstance(fc1, nn.Linear) and isinstance(fc2, nn.Linear)
    scales = scales.to(fc1.weight.device)

    fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    fc2.weight.mul_(scales.view(1, -1))

    # NaN safety checks
    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_gelu_fc(gelu: nn.Module, fc: nn.Linear, scales: torch.Tensor):
    """Apply channel-wise scaling across (GELU-like activation -> Linear)."""
    assert isinstance(gelu, (nn.GELU, BloomGelu, GELUActivation))
    assert isinstance(fc, nn.Linear)

    fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))

    for p in fc.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def auto_scale_block(
    module,
    module_kwargs,
    w_bit,
    a_bit,
    q_config,
    input_feat,
    ans_mask,
    vis_mask,
    reweight_ratio_dict,
    loss_mode="mae",
    layer_idx=0,
    token_aware_saliency=False,
    token_weighted_loss=False,
    saliency_mix_lambda=1.0,
    wa_quant=False,
    lat_debug=False,
    debug_path=None,
):
    """
    Run adaptive scale search for a Transformer block under weight-only or
    weight-activation fake quantization. Token-aware channel saliency and the
    token-weighted reconstruction objective are shared by both paths.
    """

    # === Quantization helpers ===
    if w_bit is not None:

        def w_quantize_func(p):
            return pseudo_quantize_tensor(p, n_bits=w_bit, **q_config).detach()

    else:

        def w_quantize_func(p):
            return p

    def _wa_quantize_linear(linear):
        return WALinear.from_float(
            linear,
            weight_quant="per_channel",
            act_quant="per_token",
            w_bit=w_bit,
            a_bit=a_bit,
        )

    def _wa_quantize_module(module_to_quantize):
        """Return a copy-compatible module whose Linear layers perform W/A fake quantization."""
        if isinstance(module_to_quantize, nn.Linear):
            return _wa_quantize_linear(module_to_quantize)
        linears = [
            (name, child)
            for name, child in module_to_quantize.named_modules()
            if name and isinstance(child, nn.Linear)
        ]
        for name, child in linears:
            set_op_by_name(module_to_quantize, name, _wa_quantize_linear(child))
        return module_to_quantize

    if "use_cache" in module_kwargs:
        module_kwargs.pop("use_cache")

    # === Search scales for a single module (with token-importance reweighting) ===
    def _search_module_scale(
        block,
        linears2scale: list,
        x,
        reweight_ratio=None,
        kwargs={},
        compute_token_importance=False,
    ):
        # x: [B, T, C]
        device = next(block.parameters()).device
        kwargs = kwargs or {}
        batch_size = get_scale_search_batch_size(x.shape[0])

        def _compute_token_importance_weights(block, x, x_q, kwargs, block_q=None):  # noqa: keep signature
            """
            IGQ-aligned version:
            - Compute integrated gradients of the (y_fp - y_wq) signal
            - Baseline: all-zero input x0
            - Output: token-wise weights (IQR-clipped and normalized)
            """
            device = next(block.parameters()).device
            param_dtype = next(block.parameters()).dtype
            x = torch.nan_to_num(
                x.to(device, dtype=param_dtype),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            kwargs = move_to_device(kwargs or {}, device)
            B, T, H = x.shape

            # === Baseline input ===
            x0 = torch.zeros_like(x)

            owns_block_q = block_q is None
            if owns_block_q:
                block_q = copy.deepcopy(block).to(device)
                if wa_quant:
                    block_q = _wa_quantize_module(block_q)
                else:
                    for m in block_q.modules():
                        if isinstance(m, nn.Linear):
                            m.weight.data = pseudo_quantize_tensor(
                                m.weight.data, n_bits=w_bit, **q_config
                            ).detach()

            was_training_fp = block.training
            was_training_q = block_q.training
            block.eval()
            block_q.eval()

            steps = 32
            alphas = torch.linspace(0.0, 1.0, steps, device=device, dtype=torch.float32)
            weights = torch.ones_like(alphas) / steps

            total_grad = torch.zeros_like(x, dtype=torch.float32, device=device)

            for i, alpha in enumerate(alphas):
                x_interp = (x0 + alpha * (x - x0)).detach().clone().requires_grad_(True)

                with torch.enable_grad():
                    y_fp = block(x_interp, **kwargs)
                    y_q = block_q(x_interp, **kwargs)
                    if isinstance(y_fp, tuple):
                        y_fp = y_fp[0]
                    if isinstance(y_q, tuple):
                        y_q = y_q[0]

                    # diff: [B, T, H]
                    diff = y_fp - y_q
                    F_tok = diff.abs().mean(dim=-1)  # [B, T]

                    grad_outputs = torch.ones_like(F_tok, device=F_tok.device, dtype=F_tok.dtype)
                    grad = torch.autograd.grad(
                        outputs=F_tok,
                        inputs=x_interp,
                        grad_outputs=grad_outputs,
                        retain_graph=False,
                        create_graph=False,
                    )[0]  # [B, T, H]

                if grad is None:
                    grad = torch.zeros_like(x_interp)

                grad = torch.nan_to_num(
                    grad.detach().to(torch.float32),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                total_grad += grad * weights[i]

            # === Integrated gradients & token importance ===
            ig = (x - x0).to(torch.float32) * total_grad
            token_imp = torch.nan_to_num(
                ig.abs().mean(dim=-1), nan=0.0, posinf=0.0, neginf=0.0
            )  # [B, T]

            # === IQR clipping ===
            eps = 1e-8
            token_imp = token_imp / token_imp.sum(dim=1, keepdim=True).clamp_min(eps)

            q1 = torch.quantile(token_imp, 0.25, dim=1, keepdim=True)
            q3 = torch.quantile(token_imp, 0.75, dim=1, keepdim=True)
            iqr = q3 - q1
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr

            clipped = torch.minimum(torch.maximum(token_imp, lo), hi)
            token_w = clipped / clipped.sum(dim=1, keepdim=True).clamp_min(eps)

            iqr_vis = {
                "weights_raw": token_imp.detach(),
                "weights_clipped": token_w.detach(),
                "lower_bound": lo.detach(),
                "upper_bound": hi.detach(),
                "iqr_alpha": 1.5,
            }

            # Restore original training/eval modes
            if was_training_fp:
                block.train()
            if was_training_q:
                block_q.train()

            if owns_block_q:
                del block_q
            empty_cache(device)
            return token_w.detach(), iqr_vis

        def _padding_mask(x):
            """Return a [B,T] non-padding mask when the model exposes one."""
            attention_mask = (kwargs or {}).get("attention_mask")
            if not torch.is_tensor(attention_mask):
                return torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)

            attention_mask = attention_mask.to(x.device)
            if attention_mask.ndim == 2 and tuple(attention_mask.shape) == tuple(x.shape[:2]):
                return attention_mask > 0
            if attention_mask.ndim == 4 and attention_mask.shape[0] == x.shape[0]:
                diagonal = attention_mask[:, 0].diagonal(dim1=-2, dim2=-1)
                if tuple(diagonal.shape) == tuple(x.shape[:2]):
                    return torch.isfinite(diagonal) & (diagonal > -1e4)
            return torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)

        def _valid_mask(x, ans_mask, vis_mask):
            masks = []
            for mask in (ans_mask, vis_mask):
                if mask is None:
                    continue
                mask = mask.to(device=x.device)
                if mask.ndim != 2 or tuple(mask.shape) != tuple(x.shape[:2]):
                    raise ValueError(
                        f"token mask must be [B,T]={tuple(x.shape[:2])}, got {tuple(mask.shape)}"
                    )
                masks.append(mask > 0)
            if masks:
                valid = masks[0]
                for mask in masks[1:]:
                    valid = valid | mask
            else:
                valid = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
            return valid & _padding_mask(x)

        def _sanitize_token_weights(token_w, x, ans_mask, vis_mask):
            if token_w.ndim != 2 or tuple(token_w.shape) != tuple(x.shape[:2]):
                raise ValueError(
                    f"token_w must be [B,T]={tuple(x.shape[:2])}, got {tuple(token_w.shape)}"
                )
            valid = _valid_mask(x, ans_mask, vis_mask)
            token_w = token_w.to(device=x.device, dtype=torch.float32)
            token_w = torch.where(torch.isfinite(token_w), token_w, torch.zeros_like(token_w))
            token_w = token_w.clamp_min(0) * valid.float()
            row_sum = token_w.sum(dim=1, keepdim=True)
            normalized = token_w / row_sum.clamp_min(EPS)

            # Fall back independently per row. Prefer the caption/vision valid set;
            # if that set is empty, use non-padding tokens, and only use every token
            # when the attention mask itself marks no valid positions.
            fallback_mask = valid
            fallback_count = fallback_mask.sum(dim=1, keepdim=True)
            padding_mask = _padding_mask(x)
            fallback_mask = torch.where(
                fallback_count > 0, fallback_mask, padding_mask
            )
            fallback_count = fallback_mask.sum(dim=1, keepdim=True)
            fallback_mask = torch.where(
                fallback_count > 0,
                fallback_mask,
                torch.ones_like(fallback_mask),
            )
            fallback = fallback_mask.float()
            fallback = fallback / fallback.sum(dim=1, keepdim=True).clamp_min(EPS)
            token_w = torch.where(row_sum > EPS, normalized, fallback)
            if not torch.isfinite(token_w).all():
                raise FloatingPointError("LAT-AWQ could not construct finite token weights")
            return token_w

        def _uniform_weights(x, ans_mask, vis_mask):
            """
            Uniformly assign weights over tokens where ans_mask==1 or vis_mask==1;
            all other tokens receive weight 0.
            """
            device = x.device
            dtype = torch.float32
            B, T, _ = x.shape

            valid_mask = _valid_mask(x, ans_mask, vis_mask).to(dtype)

            # Uniform distribution over valid tokens (normalized per batch)
            valid_count = valid_mask.sum(dim=1, keepdim=True).clamp_min(1e-8)
            token_w = valid_mask / valid_count
            return _sanitize_token_weights(token_w, x, ans_mask, vis_mask)

        def _compute_token_importance_weights_batched(block, x, kwargs):
            total = x.shape[0]
            chunks = []
            block_q = copy.deepcopy(block).to(device)
            if wa_quant:
                block_q = _wa_quantize_module(block_q)
            else:
                for m in block_q.modules():
                    if isinstance(m, nn.Linear):
                        m.weight.data = pseudo_quantize_tensor(
                            m.weight.data, n_bits=w_bit, **q_config
                        ).detach()
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                x_mb = x[start:end].to(device)
                kwargs_mb = move_to_device(slice_batch(kwargs or {}, start, end, total), device)
                token_mb, _ = _compute_token_importance_weights(block, x_mb, x_q=None, kwargs=kwargs_mb, block_q=block_q)
                chunks.append(token_mb.detach().cpu())
                del x_mb, kwargs_mb, token_mb
                empty_cache(device)
            del block_q
            empty_cache(device)
            token_w = torch.cat(chunks, dim=0)
            return _sanitize_token_weights(token_w, x, ans_mask, vis_mask).cpu(), None

        use_token_importance = bool(token_aware_saliency or token_weighted_loss)
        if use_token_importance:
            token_w, _ = _compute_token_importance_weights_batched(block, x, kwargs)
        else:
            token_w = None

        # ---- Grid search with token-weighted loss ----
        # Compute both statistics in FP32. The activation itself is never reweighted
        # before a forward pass.
        a_global, a_imp, x_max = get_act_scale(
            x,
            token_w=token_w if token_aware_saliency else None,
            mix_lambda=saliency_mix_lambda if token_aware_saliency else 0.0,
        )
        best_error = float("inf")
        best_ratio = -1
        best_scales = None
        n_grid = 20
        history = []

        # Save/restore parameters for each grid candidate
        baseline_block = copy.deepcopy(block).to(device).eval()
        org_sd = {k: v.detach().cpu().clone() for k, v in block.state_dict().items()}

        linear_names = []
        if wa_quant and not isinstance(block, nn.Linear):
            linear_names = [get_op_name(block, fc) for fc in linears2scale]
            if any(not name for name in linear_names):
                raise ValueError("Could not resolve a target Linear name for W/A scale search")

        for ratio in range(n_grid):
            ratio = ratio / n_grid
            scales = normalized_power_scales(x_max.to(device=device), ratio)
            for fc in linears2scale:
                if scales.numel() != fc.in_features:
                    raise ValueError(
                        f"scale channels ({scales.numel()}) != in_features "
                        f"({fc.in_features}) for {get_op_name(module, fc)}"
                    )
            if not torch.isfinite(scales).all():
                raise FloatingPointError("LAT-AWQ produced non-finite scales")

            test_module = block
            if wa_quant:
                if isinstance(block, nn.Linear):
                    fc = linears2scale[0]
                    fc_scales = scales.to(device=fc.weight.device, dtype=fc.weight.dtype).view(1, -1)
                    fc.weight.mul_(fc_scales)
                    test_module = _wa_quantize_linear(fc)
                else:
                    for fc, fc_name in zip(linears2scale, linear_names):
                        fc_scales = scales.to(device=fc.weight.device, dtype=fc.weight.dtype).view(1, -1)
                        fc.weight.mul_(fc_scales)
                        set_op_by_name(block, fc_name, _wa_quantize_linear(fc))
            else:
                for fc in linears2scale:
                    fc_scales = scales.to(device=fc.weight.device, dtype=fc.weight.dtype).view(1, -1)
                    fc.weight.mul_(fc_scales)
                    fc.weight.data = w_quantize_func(fc.weight.data) / fc_scales

            if token_weighted_loss and wa_quant:
                loss_val = reconstruction_loss_in_batches(
                    baseline_block,
                    baseline_x=x,
                    kwargs=kwargs,
                    test_module=test_module,
                    input_scales=scales,
                    token_w=token_w,
                    loss_mode="mse",
                    batch_size=batch_size,
                    device=device,
                )
            elif token_weighted_loss:
                loss_val = token_weighted_mse_in_batches(
                    baseline_block,
                    baseline_x=x,
                    kwargs=kwargs,
                    test_module=test_module,
                    token_w=token_w,
                    batch_size=batch_size,
                    device=device,
                )
            else:
                loss_val = reconstruction_loss_in_batches(
                    baseline_block,
                    baseline_x=x,
                    kwargs=kwargs,
                    test_module=test_module,
                    input_scales=scales if wa_quant else None,
                    # Match the repository AWQ objective: it passes vision_mask
                    # through its ans_mask argument.
                    ans_mask=vis_mask,
                    loss_mode="mse",
                    batch_size=batch_size,
                    device=device,
                )

            if not torch.isfinite(torch.tensor(loss_val)):
                raise FloatingPointError(f"non-finite scale-search loss at alpha={ratio}")
            history.append((ratio, loss_val))

            if loss_val < best_error:
                best_error = loss_val
                best_ratio = ratio
                best_scales = scales

            # Restore modules first, then their original parameters.
            if wa_quant and not isinstance(block, nn.Linear):
                for fc, fc_name in zip(linears2scale, linear_names):
                    set_op_by_name(block, fc_name, fc)
            block.load_state_dict(org_sd, strict=True)
            if wa_quant and isinstance(block, nn.Linear):
                del test_module

        if best_ratio == -1 or best_scales is None:
            print("Scale search history:", history)
            raise RuntimeError("Failed to find best ratio.")

        group_name = "+".join(get_op_name(module, fc) for fc in linears2scale)
        global_rank = torch.argsort(torch.argsort(a_global))
        imp_rank = torch.argsort(torch.argsort(a_imp))
        global_rank = global_rank.float() - global_rank.float().mean()
        imp_rank = imp_rank.float() - imp_rank.float().mean()
        rank_cosine = F.cosine_similarity(global_rank, imp_rank, dim=0)
        relative_l1 = (a_global - a_imp).abs().mean() / a_global.abs().mean().clamp_min(EPS)
        stats = {
            "layer_idx": int(layer_idx),
            "group": group_name,
            "x_shape": list(x.shape),
            "token_w_shape": list(token_w.shape) if token_w is not None else None,
            "token_w_min": float(token_w.min()) if token_w is not None else None,
            "token_w_max": float(token_w.max()) if token_w is not None else None,
            "token_w_mean": float(token_w.mean()) if token_w is not None else None,
            "token_w_std": float(token_w.std(unbiased=False)) if token_w is not None else None,
            "token_w_l2": float(token_w.float().pow(2).sum().sqrt()) if token_w is not None else None,
            "token_w_argmax": token_w.argmax(dim=1).tolist() if token_w is not None else None,
            "a_global_min": float(a_global.min()),
            "a_global_max": float(a_global.max()),
            "a_global_mean": float(a_global.mean()),
            "a_imp_min": float(a_imp.min()),
            "a_imp_max": float(a_imp.max()),
            "a_imp_mean": float(a_imp.mean()),
            "saliency_cosine": float(F.cosine_similarity(a_global, a_imp, dim=0)),
            "saliency_rank_cosine": float(rank_cosine),
            "saliency_relative_l1": float(relative_l1),
            "selected_alpha": float(best_ratio),
            "best_loss": float(best_error),
            "scale_min": float(best_scales.min()),
            "scale_max": float(best_scales.max()),
            "token_aware_saliency": bool(token_aware_saliency),
            "token_weighted_loss": bool(token_weighted_loss),
            "saliency_mix_lambda": float(saliency_mix_lambda),
        }
        if lat_debug:
            print("[LAT-AWQ] " + json.dumps(stats, sort_keys=True), flush=True)
        if debug_path:
            os.makedirs(os.path.dirname(debug_path) or ".", exist_ok=True)
            with open(debug_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(stats, sort_keys=True) + "\n")

        del baseline_block
        return best_scales.view(-1).detach()

    # === Wrapper: produce (prev_op_name, layer_names, scales_cpu) ===
    def _auto_get_scale(
        prev_op,
        layers,
        inp,
        reweight_ratio=None,
        module2inspect=None,
        kwargs={},
        compute_token_importance=False,
    ):
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        scales = _search_module_scale(
            module2inspect,
            layers,
            inp,
            reweight_ratio,
            kwargs,
            compute_token_importance=compute_token_importance,
        ).detach().cpu()

        return (
            get_op_name(module, prev_op),
            tuple([get_op_name(module, m) for m in layers]),
            scales,
        )

    # -------------------------
    # Model-specific branches (including Qwen2 / InternLM2)
    # -------------------------

    def _get_feat(d, *keys):
        """Return the first existing key from input_feat, otherwise raise."""
        for k in keys:
            if k in d:
                return d[k]
        raise KeyError(f"input_feat missing keys: {keys}")

    scales_list = []

    if isinstance(module, OPTDecoderLayer):
        # Attention input projections
        scales_list.append(
            _auto_get_scale(
                prev_op=module.self_attn_layer_norm,
                layers=[module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )
        # Attention output projection
        scales_list.append(
            _auto_get_scale(
                prev_op=module.self_attn.v_proj,
                layers=[module.self_attn.out_proj],
                inp=input_feat["self_attn.out_proj"],
            )
        )
        # MLP fc1
        scales_list.append(
            _auto_get_scale(
                prev_op=module.final_layer_norm,
                layers=[module.fc1],
                inp=input_feat["fc1"],
                module2inspect=module,
                kwargs=module_kwargs,
            )
        )
        # MLP fc2
        scales_list.append(
            _auto_get_scale(
                prev_op=module.fc1,
                layers=[module.fc2],
                inp=input_feat["fc2"],
            )
        )

    elif isinstance(module, LlamaDecoderLayer):
        # Attention input projections
        scales_list.append(
            _auto_get_scale(
                prev_op=module.input_layernorm,
                layers=[module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj],
                inp=input_feat["self_attn.q_proj"],
                reweight_ratio=reweight_ratio_dict.get("attn", None),
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )
        # Attention output projection (only if shapes match)
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    inp=input_feat["self_attn.o_proj"],
                    reweight_ratio=reweight_ratio_dict.get("attn", None),
                )
            )
        # MLP input (gate + up)
        scales_list.append(
            _auto_get_scale(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                inp=input_feat["mlp.gate_proj"],
                reweight_ratio=reweight_ratio_dict.get("mlp", None),
                module2inspect=module.mlp,
            )
        )
        # MLP output
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                inp=input_feat["mlp.down_proj"],
                reweight_ratio=reweight_ratio_dict.get("mlp", None),
            )
        )

    elif isinstance(module, BloomBlock):
        # Attention input
        scales_list.append(
            _auto_get_scale(
                prev_op=module.input_layernorm,
                layers=[module.self_attention.query_key_value],
                inp=input_feat["self_attention.query_key_value"],
                module2inspect=module.self_attention,
                kwargs=module_kwargs,
            )
        )
        # Attention output (only if shapes match)
        if module.self_attention.query_key_value.weight.shape == module.self_attention.dense.weight.shape:
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.self_attention.query_key_value,
                    layers=[module.self_attention.dense],
                    inp=input_feat["self_attention.dense"],
                )
            )
        # MLP fc1
        scales_list.append(
            _auto_get_scale(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.dense_h_to_4h],
                inp=input_feat["mlp.dense_h_to_4h"],
                module2inspect=module,
                kwargs=module_kwargs,
            )
        )
        # MLP fc2
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.gelu_impl,
                layers=[module.mlp.dense_4h_to_h],
                inp=input_feat["mlp.dense_4h_to_h"],
            )
        )

    elif "mpt" in str(module.__class__).lower():
        scales_list.append(
            _auto_get_scale(
                prev_op=module.norm_1,
                layers=[module.attn.Wqkv],
                inp=input_feat["attn.Wqkv"],
                module2inspect=module.attn,
                kwargs=module_kwargs,
            )
        )
        scales_list.append(
            _auto_get_scale(
                prev_op=module.attn.Wqkv,
                layers=[module.attn.out_proj],
                inp=input_feat["attn.out_proj"],
            )
        )
        scales_list.append(
            _auto_get_scale(
                prev_op=module.norm_2,
                layers=[module.ffn.up_proj, module.ffn.gate_proj],
                inp=input_feat["ffn.up_proj"],
                module2inspect=module.ffn,
            )
        )
        scales_list.append(
            _auto_get_scale(
                prev_op=module.ffn.up_proj,
                layers=[module.ffn.down_proj],
                inp=input_feat["ffn.down_proj"],
            )
        )

    elif "gptneox" in str(module.__class__).lower() or "neox" in str(module.__class__).lower():
        scales_list.append(
            _auto_get_scale(
                prev_op=module.input_layernorm,
                layers=[module.attention.query_key_value],
                inp=input_feat["attention.query_key_value"],
                module2inspect=module.attention,
                kwargs=module_kwargs,
            )
        )
        if module.attention.query_key_value.weight.shape == module.attention.dense.weight.shape:
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.attention.query_key_value,
                    layers=[module.attention.dense],
                    inp=input_feat["attention.dense"],
                )
            )
        scales_list.append(
            _auto_get_scale(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.dense_h_to_4h],
                inp=input_feat["mlp.dense_h_to_4h"],
                module2inspect=module.mlp,
            )
        )
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.dense_h_to_4h,
                layers=[module.mlp.dense_4h_to_h],
                inp=input_feat["mlp.dense_4h_to_h"],
            )
        )

    elif module.__class__.__name__ in ("Qwen2DecoderLayer", "Qwen2VLDecoderLayer", "Qwen2_5_VLDecoderLayer"):
        # Attention input
        scales_list.append(
            _auto_get_scale(
                prev_op=module.input_layernorm,
                layers=[module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj],
                inp=_get_feat(input_feat, "self_attn.q_proj", "attention.q_proj"),
                reweight_ratio=reweight_ratio_dict.get("attn", None),
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
                compute_token_importance=True,
            )
        )
        # Attention output (only if shapes match)
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    inp=_get_feat(input_feat, "self_attn.o_proj", "attention.o_proj"),
                    reweight_ratio=reweight_ratio_dict.get("attn", None),
                    compute_token_importance=True,
                )
            )
        # MLP input (gate + up)
        scales_list.append(
            _auto_get_scale(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                inp=_get_feat(input_feat, "mlp.gate_proj", "mlp.up_proj"),
                reweight_ratio=reweight_ratio_dict.get("mlp", None),
                module2inspect=module.mlp,
                compute_token_importance=True,
            )
        )
        # MLP output (down_proj)
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                inp=_get_feat(input_feat, "mlp.down_proj"),
                reweight_ratio=reweight_ratio_dict.get("mlp", None),
                compute_token_importance=True,
            )
        )

    elif (
        ("internlm2" in str(module.__class__).lower())
        or ("internvl2" in str(module.__class__).lower())
        or (module.__class__.__name__ in ("InternLM2DecoderLayer", "InternVL2DecoderLayer"))
    ):
        # InternLM2 branch: fused wqkv/wo and FFN w1, w3, w2
        attn = getattr(module, "attention", getattr(module, "self_attn", None))
        feed_forward = getattr(module, "feed_forward", getattr(module, "mlp", None))
        attn_norm = getattr(module, "attention_norm", getattr(module, "input_layernorm", None))
        ffn_norm = getattr(module, "ffn_norm", getattr(module, "post_attention_layernorm", None))
        assert (
            attn is not None and feed_forward is not None and attn_norm is not None and ffn_norm is not None
        ), "Unexpected InternLM2 layer structure"

        # Attention input: wqkv
        scales_list.append(
            _auto_get_scale(
                prev_op=attn_norm,
                layers=[attn.wqkv],
                inp=_get_feat(input_feat, "attention.wqkv", "self_attn.wqkv", "attn.wqkv"),
                reweight_ratio=reweight_ratio_dict.get("attn", None),
                module2inspect=attn,
                kwargs=module_kwargs,
                compute_token_importance=False,
            )
        )
        # Attention output: wo (only if present and shapes match)
        if hasattr(attn, "wo") and attn.wqkv.weight.shape == attn.wo.weight.shape:
            scales_list.append(
                _auto_get_scale(
                    prev_op=attn.wqkv,
                    layers=[attn.wo],
                    inp=_get_feat(input_feat, "attention.wo", "self_attn.wo", "attn.wo"),
                    reweight_ratio=reweight_ratio_dict.get("attn", None),
                    module2inspect=attn.wo,
                    compute_token_importance=False,
                )
            )
        # MLP input: w1 + w3 (gate/up)
        scales_list.append(
            _auto_get_scale(
                prev_op=ffn_norm,
                layers=[feed_forward.w1, feed_forward.w3],
                inp=_get_feat(input_feat, "feed_forward.w1", "mlp.w1", "mlp.gate_proj"),
                reweight_ratio=reweight_ratio_dict.get("mlp", None),
                module2inspect=feed_forward,
                compute_token_importance=False,
            )
        )
        # MLP output: w2 (down)
        scales_list.append(
            _auto_get_scale(
                prev_op=feed_forward.w3,
                layers=[feed_forward.w2],
                inp=_get_feat(input_feat, "feed_forward.w2", "mlp.w2", "mlp.down_proj"),
                reweight_ratio=reweight_ratio_dict.get("mlp", None),
                compute_token_importance=False,
            )
        )

    else:
        raise NotImplementedError(f"Unsupported block type: {type(module)}")

    gc.collect()
    return scales_list


# =========================
# Apply scales (external interface unchanged)
# =========================
@torch.no_grad()
def apply_scale(module, scales_list, input_feat_dict=None):
    target_device = next(module.parameters()).device
    for prev_op_name, layer_names, scales in scales_list:
        prev_op = get_op_by_name(module, prev_op_name)
        layers = [get_op_by_name(module, name) for name in layer_names]

        prev_op.to(target_device)
        for layer in layers:
            layer.to(target_device)
        scales = scales.to(target_device)

        if isinstance(prev_op, nn.Linear):
            assert len(layers) == 1
            scale_fc_fc(prev_op, layers[0], scales)
        elif isinstance(prev_op, (nn.LayerNorm, LlamaRMSNorm)) or prev_op.__class__.__name__ in (
            "InternLM2RMSNorm",
            "Qwen2RMSNorm",
        ):
            scale_ln_fcs(prev_op, layers, scales)
        elif isinstance(prev_op, (nn.GELU, BloomGelu, GELUActivation)):
            # Lazy import to avoid circular dependencies
            from qmllm.methods.lat_awq.quantize.qmodule import ScaledActivation  # noqa: WPS433

            new_module = ScaledActivation(prev_op, scales)
            set_op_by_name(module, prev_op_name, new_module)
            scale_gelu_fc(prev_op, layers[0], scales)
        else:
            raise NotImplementedError(f"prev_op {type(prev_op)} not supported yet!")

        # Optionally apply scaling to cached input features
        if input_feat_dict is not None:
            for layer_name in layer_names:
                inp = input_feat_dict[layer_name]
                inp.div_(scales.view(1, -1).to(inp.device))

        prev_op.cpu()
        for layer in layers:
            layer.cpu()
        scales.cpu()
