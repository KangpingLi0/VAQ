import copy
import os
import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLDecoderLayer,
    Qwen2_5_VLRotaryEmbedding,
)

from qmllm.methods.mbq.quantize.auto_scale import apply_scale
from qmllm.methods.mbq.quantize.auto_scale_wa import auto_scale_block_wa
from qmllm.quantization.qlinear import WALinear


@patch.dict(os.environ, {"QIG_SCALE_SEARCH_BATCH_SIZE": "2"})
@torch.no_grad()
def check_qwen25vl_mbq_w4a8(num_key_value_heads):
    torch.manual_seed(42)
    config = Qwen2_5_VLTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=num_key_value_heads,
        rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3]},
    )
    config._attn_implementation = "sdpa"
    layer = Qwen2_5_VLDecoderLayer(config, layer_idx=0).eval()
    inputs = torch.randn(4, 7, config.hidden_size)
    positions = torch.arange(7).view(1, 1, 7).expand(3, 4, 7)
    kwargs = {
        "position_embeddings": Qwen2_5_VLRotaryEmbedding(config)(inputs, positions),
        "attention_mask": torch.full((4, 1, 7, 7), float("-inf")).triu(1),
        "use_cache": False,
    }
    features = {}

    def capture(name):
        def hook(module, args):
            features[name] = args[0].detach().clone()
        return hook

    hooks = [
        module.register_forward_pre_hook(capture(name))
        for name, module in layer.named_modules()
        if isinstance(module, nn.Linear)
    ]
    reference = layer(inputs, **kwargs)[0]
    for hook in hooks:
        hook.remove()
    original_state = copy.deepcopy(layer.state_dict())
    vision_mask = torch.zeros(4, 7, dtype=torch.bool)
    vision_mask[:, :4] = True

    scales = auto_scale_block_wa(
        layer,
        dict(kwargs),
        w_bit=4,
        a_bit=8,
        q_config={"zero_point": True, "q_group_size": 128},
        input_feat=features,
        ans_mask=~vision_mask,
        vis_mask=vision_mask,
        reweight_ratio_dict={"attn": 1.5, "mlp": 2.0},
        loss_mode="mae",
    )

    expected_paths = ["input_layernorm", "post_attention_layernorm", "mlp.up_proj"]
    if num_key_value_heads == config.num_attention_heads:
        expected_paths.insert(1, "self_attn.v_proj")
    assert [path for path, _, _ in scales] == expected_paths
    assert all(torch.isfinite(scale).all() and (scale > 0).all() for _, _, scale in scales)
    # Search must restore the original modules and weights after each candidate.
    assert all(torch.equal(value, original_state[name]) for name, value in layer.state_dict().items())

    apply_scale(layer, scales)
    torch.testing.assert_close(layer(inputs, **kwargs)[0], reference, rtol=1e-5, atol=1e-6)

    for name, module in list(layer.named_modules()):
        if isinstance(module, nn.Linear):
            parent_name, _, child_name = name.rpartition(".")
            parent = layer.get_submodule(parent_name)
            setattr(parent, child_name, WALinear.from_float(module, w_bit=4, a_bit=8))
    assert sum(isinstance(module, WALinear) for module in layer.modules()) == 7
    assert torch.isfinite(layer(inputs, **kwargs)[0]).all()


class TestMBQWeightActivation(unittest.TestCase):
    def test_qwen25vl_grouped_query_attention(self):
        check_qwen25vl_mbq_w4a8(num_key_value_heads=2)

    def test_qwen25vl_multi_head_attention(self):
        check_qwen25vl_mbq_w4a8(num_key_value_heads=4)
