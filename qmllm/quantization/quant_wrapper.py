import json
import os
import time

from qmllm.methods.awq.entry import awq_entry
from qmllm.methods.smoothquant.entry import smoothquant_entry
from qmllm.methods.mbq.entry import mbq_entry
from qmllm.methods.qig.entry import qig_entry
from qmllm.methods.lat_awq.entry import lat_awq_entry
from qmllm.methods.rtn.entry import rtn_entry
from qmllm.methods.gptq.entry import gptq_entry
from qmllm.utils.device import accelerator_device_count

def qwrapper(model, prompt_inputs, prompt_kwargs, args):
    if args.method == "awq":
        model = awq_entry(model, prompt_inputs, prompt_kwargs, run_awq_process=args.run_process, pseudo_quant=args.pseudo_quant, scale_path=args.scale_path, q_group_size=args.w_group, w_bit=args.w_bit)
    elif args.method == "smoothquant":
        model = smoothquant_entry(model, prompt_inputs, prompt_kwargs, run_sq_process=args.run_process, pseudo_quant=args.pseudo_quant, scale_path=args.scale_path, w_bit=args.w_bit, a_bit=args.a_bit, alpha=args.alpha)
    elif args.method == "mbq":
        wa_quant = args.w_bit < 16 and args.a_bit < 16
        model = mbq_entry(model, prompt_inputs, prompt_kwargs, 
                                run_mbq_process=args.run_process, 
                                pseudo_quant=args.pseudo_quant, 
                                scale_path=args.scale_path, 
                                q_group_size=args.w_group, 
                                w_bit=args.w_bit, 
                                a_bit=args.a_bit, 
                                wa_quant=wa_quant, 
                                reweight=args.reweight,
                                distort=args.distort,
                                loss_mode=args.loss_mode)
    elif args.method == "qig":
        wa_quant = args.w_bit < 16 and args.a_bit < 16
        model = qig_entry(model, prompt_inputs, prompt_kwargs, 
                                run_qig_process=args.run_process, 
                                pseudo_quant=args.pseudo_quant, 
                                scale_path=args.scale_path, 
                                q_group_size=args.w_group, 
                                w_bit=args.w_bit, 
                                a_bit=args.a_bit, 
                                wa_quant=wa_quant, 
                                reweight=args.reweight,
                                distort=args.distort,
                                loss_mode=args.loss_mode,
                                device=getattr(args, "torch_device", None))        
    elif args.method == "lat_awq":
        wa_quant = args.w_bit < 16 and args.a_bit < 16
        log_dir = getattr(args, "lat_log_dir", "logs/lat_awq")
        output_dir = getattr(args, "lat_output_dir", "outputs/lat_awq")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        run_name = os.path.splitext(os.path.basename(args.scale_path or "lat_awq"))[0]
        debug_path = (
            os.path.join(log_dir, f"{run_name}.jsonl")
            if getattr(args, "run_process", False) and getattr(args, "lat_debug", False)
            else None
        )
        if (
            debug_path is not None
            and not os.path.exists(args.scale_path or "")
            and os.path.exists(debug_path)
        ):
            os.remove(debug_path)
        started = time.time()
        model = lat_awq_entry(
            model,
            prompt_inputs,
            prompt_kwargs,
            run_lat_awq_process=args.run_process,
            pseudo_quant=args.pseudo_quant,
            scale_path=args.scale_path,
            q_group_size=args.w_group,
            w_bit=args.w_bit,
            a_bit=args.a_bit,
            wa_quant=wa_quant,
            reweight=args.reweight,
            distort=args.distort,
            loss_mode="mse",
            device=getattr(args, "torch_device", None),
            token_aware_saliency=getattr(args, "token_aware_saliency", False),
            token_weighted_loss=getattr(args, "token_weighted_loss", False),
            saliency_mix_lambda=getattr(args, "saliency_mix_lambda", 1.0),
            lat_debug=getattr(args, "lat_debug", False),
            debug_path=debug_path,
        )
        metadata = {
            "method": "lat_awq",
            "model": getattr(args, "model_args", ""),
            "w_bit": args.w_bit,
            "a_bit": args.a_bit,
            "group_size": args.w_group,
            "n_samples": (
                getattr(args, "effective_n_samples", None)
                if getattr(args, "run_process", False)
                else None
            ),
            "phase": (
                "search_and_apply"
                if getattr(args, "run_process", False) and getattr(args, "pseudo_quant", False)
                else "search"
                if getattr(args, "run_process", False)
                else "apply"
                if getattr(args, "pseudo_quant", False)
                else "load_only"
            ),
            "saliency_mix_lambda": getattr(args, "saliency_mix_lambda", 1.0),
            "token_aware_saliency": getattr(args, "token_aware_saliency", False),
            "token_weighted_loss": getattr(args, "token_weighted_loss", False),
            "seed": getattr(args, "seed", 42),
            "scale_path": args.scale_path,
            "debug_path": debug_path,
            "quantization_time_seconds": time.time() - started,
        }
        with open(os.path.join(output_dir, f"{run_name}.json"), "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
    elif args.method == "rtn":
        wa_quant = args.w_bit < 16 and args.a_bit < 16
        model = rtn_entry(model, pseudo_quant=args.pseudo_quant, wa_quant=wa_quant, q_group_size=args.w_group, w_bit=args.w_bit, a_bit=args.a_bit)
    elif args.method == "gptq":
        if accelerator_device_count(getattr(args, "torch_device", None)) > 1:
            model = gptq_entry(
                model,
                prompt_inputs,
                prompt_kwargs,
                pseudo_quant=args.pseudo_quant,
                w_bit=args.w_bit,
                q_group_size=args.w_group,
                percdamp=getattr(args, "percdamp", 0.01),
                model_args=args.model_args,
                model_type=args.model,
            )
        else:
            model = gptq_entry(
                model,
                prompt_inputs,
                prompt_kwargs,
                pseudo_quant=args.pseudo_quant,
                w_bit=args.w_bit,
                q_group_size=args.w_group,
                percdamp=getattr(args, "percdamp", 0.01),
            )
    else:
        raise NotImplementedError

    return model
