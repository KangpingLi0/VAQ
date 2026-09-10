import os
import torch

from utils.device import get_device
from qmllm.methods.lat_awq.quantize.pre_quant import run_lat_awq, apply_lat_awq
from qmllm.methods.lat_awq.quantize.quantizer import pseudo_quantize_model_weight, pseudo_quantize_model_weight_act


def lat_awq_entry(
    model,
    prompt_inputs,
    prompt_kwargs,
    run_lat_awq_process: bool,
    pseudo_quant: bool,
    scale_path: str = None,
    zero_point: bool = True,
    q_group_size: int = 128,
    w_bit: int = 4,
    a_bit: int = 16,
    wa_quant: bool = False,
    reweight: bool = False,
    distort: bool = False,
    loss_mode: str = "mse",
    device=None,
    token_aware_saliency: bool = False,
    token_weighted_loss: bool = False,
    saliency_mix_lambda: float = 1.0,
    lat_debug: bool = False,
    debug_path: str = None,
):
    '''
    model: here the model is the LLM, you have to extract the LLM first! 
    prompt_tokens: the prompt tokens
    prompt_mask: the prompt mask, mask the answer language tokens
    run_lat_awq_process: whether to run the QIG process
    '''
    q_config = {
        "zero_point": zero_point,  # by default True
        "q_group_size": q_group_size,  # whether to use group quantization
    }

    if scale_path is None:
        raise ValueError("--scale_path is required for lat_awq")
    device = get_device(device or getattr(model, "device", "auto"))
    if hasattr(model, "set_device"):
        model.set_device(device)

    scale_exist = os.path.exists(scale_path)
    expected_metadata = {
        "method": "lat_awq",
        "w_bit": int(w_bit),
        "a_bit": int(a_bit),
        "group_size": int(q_group_size),
        "token_aware_saliency": bool(token_aware_saliency),
        "token_weighted_loss": bool(token_weighted_loss),
        "saliency_mix_lambda": float(saliency_mix_lambda),
    }
    if scale_exist:
        cached = torch.load(scale_path, map_location="cpu")
        cached_metadata = cached.get("metadata", {}) if isinstance(cached, dict) else {}
        mismatch = {
            key: (cached_metadata.get(key), value)
            for key, value in expected_metadata.items()
            if cached_metadata.get(key) != value
        }
        if mismatch:
            raise ValueError(
                f"LAT-AWQ scale cache {scale_path!r} does not match this run: {mismatch}. "
                "Use a distinct --scale_path for each ablation."
            )
    # reparameterization
    if run_lat_awq_process and not scale_exist:
        model.to_cpu()
        lat_awq_results = run_lat_awq(
            model,
            prompt_inputs,
            prompt_kwargs,
            w_bit=w_bit,
            a_bit=a_bit,
            q_config=q_config,
            auto_scale=True,
            loss_mode=loss_mode,
            wa_quant=wa_quant,
            reweight=reweight,
            distort=distort,
            device=device,
            token_aware_saliency=token_aware_saliency,
            token_weighted_loss=token_weighted_loss,
            saliency_mix_lambda=saliency_mix_lambda,
            lat_debug=lat_debug,
            debug_path=debug_path,
        )
        
        dirpath = os.path.dirname(scale_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        
        torch.save(lat_awq_results, scale_path)
        print("LAT-AWQ results saved at", scale_path)
        cached = lat_awq_results
        scale_exist = True

    if pseudo_quant:
        if not scale_exist:
            raise FileNotFoundError(f"LAT-AWQ scale cache not found: {scale_path}")
        lat_awq_results = cached
        model.to_cpu()
        apply_lat_awq(model.model, lat_awq_results)

        if not wa_quant:
            # weight quantization
            pseudo_quantize_model_weight(model.model, w_bit=w_bit, q_config=q_config)
        else:
            # weight activation quantization
            pseudo_quantize_model_weight_act(model.model, w_bit=w_bit, a_bit=a_bit)

    if hasattr(model, "to_device"):
        model.to_device(device)
    else:
        model.to_cuda()
    return model
