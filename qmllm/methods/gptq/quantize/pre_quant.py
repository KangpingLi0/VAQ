import torch
import torch.nn as nn
import tqdm
import copy
import gc
import functools
from collections import defaultdict
from typing import List, Any
import logging
from copy import deepcopy

import numpy as np
from torch.nn import CrossEntropyLoss
from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from qmllm.methods.gptq.quantize.qmodule import find_qlayers, WeightQuantizer
from qmllm.methods.gptq.quantize.quantizer import GPTQ, cleanup_memory
from qmllm.utils.device import empty_cache, forward_module_in_batches, move_to_device
from qmllm.utils.hf_compat import get_qwen_vl_layers, move_qwen_vl_embeddings

__all__ = ["run_gptq"]


def _default_device():
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


DEV = _default_device()
NUM_GPU = torch.cuda.device_count() if DEV.type == "cuda" and torch.cuda.is_available() else 0

def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


def get_blocks(model):
    if model.__class__.__name__ == "LlamaForCausalLM":
        layers = model.model.layers
    elif model.__class__.__name__ == "LlavaLlamaForCausalLM":
        # layers = [model.model.layers, model.model.vision_tower.vision_tower.vision_model.encoder.layers]
        layers = model.model.layers
    elif model.__class__.__name__ == "LlavaQwenForCausalLM":
        layers = model.model.layers
    elif model.__class__.__name__ == "InternLM2ForCausalLM":
        layers = model.model.layers
    elif model.__class__.__name__ == "InternVLChatModel":
        layers = model.language_model.model.layers
    elif model.__class__.__name__ == "Qwen2VLForConditionalGeneration":
        layers = get_qwen_vl_layers(model)
    elif model.__class__.__name__ == "Qwen2_5_VLForConditionalGeneration":
        layers = get_qwen_vl_layers(model)
    elif model.__class__.__name__ == "LlavaLlamaModel":
        layers = model.llm.model.layers
    elif isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, BloomForCausalLM):
        layers = model.transformer.h
    elif "mpt" in str(model.__class__).lower():
        layers = model.transformer.blocks
    elif "falcon" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "bigcode" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "neox" in str(model.__class__).lower():
        layers = model.gpt_neox.layers
    else:
        raise NotImplementedError(type(model))
    return layers


def move_embed(model, device):
    if isinstance(model, LlamaForCausalLM):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    elif isinstance(model, OPTForCausalLM):
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
            device
        )
    elif isinstance(model, BloomForCausalLM):
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
        model.transformer.word_embeddings_layernorm = (
            model.transformer.word_embeddings_layernorm.to(device)
        )
    elif "mpt" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.emb_drop = model.transformer.emb_drop.to(device)
    elif "falcon" in str(model.__class__).lower():
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
    elif "bigcode" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.wpe = model.transformer.wpe.to(device)
        model.transformer.drop = model.transformer.drop.to(device)
    elif "neox" in str(model.__class__).lower():
        model.gpt_neox.embed_in = model.gpt_neox.embed_in.to(device)
        model.gpt_neox.emb_dropout = model.gpt_neox.emb_dropout.to(device)
        model.embed_out = model.embed_out.to(device)
    elif model.__class__.__name__ == "LlavaLlamaForCausalLM":
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.vision_tower.vision_tower.vision_model.embeddings.to(device)
    elif model.__class__.__name__ == "LlavaQwenForCausalLM":
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    elif model.__class__.__name__ == "InternLM2ForCausalLM":
        model.model.tok_embeddings = model.model.tok_embeddings.to(device)
    elif model.__class__.__name__ == "InternVLChatModel":
        model.language_model.model.tok_embeddings = model.language_model.model.tok_embeddings.to(device)  
    elif model.__class__.__name__ == "Qwen2VLForConditionalGeneration":
        move_qwen_vl_embeddings(model, device)
    elif model.__class__.__name__ == "Qwen2_5_VLForConditionalGeneration":
        move_qwen_vl_embeddings(model, device)
    elif model.__class__.__name__ == "LlavaLlamaModel":
        model.llm.model.embed_tokens = model.llm.model.embed_tokens.to(device)
    elif model.__class__.__name__ == "Qwen2_VL":
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    else:
        raise NotImplementedError(type(model))


def process_input(prompt_inputs, prompt_kwargs):
    inputs = {**prompt_inputs, **prompt_kwargs}
    inputs["use_cache"] = False
    vision_mask = inputs.pop("vision_mask", None)
    caption_mask = inputs.pop("caption_mask", None)
    
    return inputs, vision_mask, caption_mask



@torch.no_grad()
def run_gptq(model,
             prompt_inputs,
             prompt_kwargs,
             w_bit,
             w_asym,
             w_clip,
             w_groupsize,
             percdamp,
             act_order,
             w_static_groups,
):
    '''
    From GPTQ repo 
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    logging.info('-----GPTQ Quantization-----')
    # if not torch.cuda.is_available() or torch.cuda.device_count() > 1:
    #     distribute_model(model)
    
    device = torch.device(getattr(model, "device", DEV))
    layers = get_blocks(model.model)

    layers[0] = layers[0].to(device)
    move_embed(model.model, device)

    dtype = next(iter(model.model.parameters())).dtype

    inputs, vision_mask, caption_mask = process_input(prompt_inputs, prompt_kwargs)
    inputs = move_to_device(inputs, device)
    # nsamples = inputs['inputs_embeds'].shape[0]

    inps = []
    layer_kwargs = {}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            if hasattr(module, "attention_type"):
                self.attention_type = module.attention_type
        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError
    layers[0] = Catcher(layers[0])
        
    try:
        if NUM_GPU > 1:
            model.to_cpu()
            inputs = {k: v.to('cpu') if torch.is_tensor(v) else v for k, v in inputs.items()}
        # print(model.__class__.__name__)
        model(**inputs)
    except ValueError: # work with early exit
        # print("False")
        pass
    layers[0] = layers[0].module
    layer_kwargs["use_cache"] = False
    layer_kwargs = move_to_device(layer_kwargs, "cpu")
    layers[0] = layers[0].cpu()
    move_embed(model.model, 'cpu')
    inputs = move_to_device(inputs, "cpu")
    empty_cache(device)

    outs = deepcopy(inps)
    
    inps = inps[0].detach().cpu()
    # print(model.__class__.__name__)

    for i in tqdm.tqdm(range(len(layers)), desc="(GPTQ Quant.) Layers"):
        # model.model.language_model.rotary_emb = model.model.language_model.rotary_emb.to('cuda')
        layer = layers[i].to(device)
        named_linears = get_named_linears(layer)
        for name in named_linears:
            gptq = {}
            layer_weight_bits = w_bit
            layer_weight_sym = not (w_asym)
            if 'lm_head' in name:
                layer_weight_bits = 16
                continue
            gptq[name] = GPTQ(named_linears[name])
            gptq[name].quantizer = WeightQuantizer()
            gptq[name].quantizer.configure(
                layer_weight_bits, perchannel=True, sym=layer_weight_sym, mse=w_clip
            )
            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            handles.append(named_linears[name].register_forward_hook(add_batch(name)))
            layer_device = next(layer.parameters()).device
            outs = forward_module_in_batches(layer, inps, layer_kwargs, batch_size=1, device=layer_device)
            for h in handles:
                h.remove()
            
            layer_w_groupsize = w_groupsize
            gptq[name].fasterquant(
                percdamp=percdamp, groupsize=layer_w_groupsize,
                actorder=act_order, static_groups=w_static_groups
            )
            gptq[name].free()

        layer_device = next(layer.parameters()).device
        outs = forward_module_in_batches(layer, inps, layer_kwargs, batch_size=1, device=layer_device)
        layers[i] = layer.cpu()
        del layer
        del gptq
        empty_cache(device)

        inps, outs = outs, inps

    # model.config.use_cache = use_cache
    cleanup_memory(verbos=True, device=device)
    logging.info('-----GPTQ Quantization Done-----')
    

def distribute_model(model) -> None:
    from accelerate import dispatch_model, infer_auto_device_map
    from accelerate.utils import get_balanced_memory
    no_split_module_classes = getattr(model.model, "_no_split_modules", [])
    if len(no_split_module_classes) == 0:
        raise ValueError("Model has no _no_split_modules attribute for distribute_model.")
        
    print(no_split_module_classes)
    
    # no_split_module_classes = ['LlamaDecoderLayer']
    max_memory = get_balanced_memory(
        model.model,
        no_split_module_classes=no_split_module_classes,
    )

    device_map = infer_auto_device_map(
        model.model, max_memory=max_memory, no_split_module_classes=no_split_module_classes
    )

    dispatch_model(
        model.model,
        device_map=device_map,
        offload_buffers=True,
        offload_dir="offload",
        state_dict=model.model.state_dict(),
    )

    cleanup_memory()
