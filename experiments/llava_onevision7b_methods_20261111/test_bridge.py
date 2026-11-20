import unittest
from types import SimpleNamespace

import bootstrap
import torch
from transformers import LlavaOnevisionConfig, LlavaOnevisionForConditionalGeneration, Qwen2Config, SiglipVisionConfig
from model_bridge import decoder_view


def tiny_model():
    text = Qwen2Config(vocab_size=256, hidden_size=128, intermediate_size=256,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2)
    vision = SiglipVisionConfig(hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, image_size=28, patch_size=14)
    config = LlavaOnevisionConfig(text_config=text, vision_config=vision,
        image_token_index=254, video_token_index=255, image_grid_pinpoints=[[28,28]])
    config._attn_implementation = 'sdpa'
    return LlavaOnevisionForConditionalGeneration(config).eval()


class TestDecoderBridge(unittest.TestCase):
    @torch.no_grad()
    def test_shared_decoder_preserves_causal_forward_and_loss(self):
        torch.manual_seed(42)
        full = tiny_model()
        decoder = decoder_view(full)
        ids = torch.tensor([[1,2,3,4,5,6]])
        mask = torch.ones_like(ids)
        labels = ids.clone()
        labels[:,:3] = -100
        a = full(input_ids=ids, attention_mask=mask, labels=labels)
        b = decoder(input_ids=ids, attention_mask=mask, labels=labels)
        torch.testing.assert_close(a.logits, b.logits)
        torch.testing.assert_close(a.loss, b.loss)
        self.assertIs(decoder.model, full.model.language_model)
        self.assertIs(decoder.lm_head, full.lm_head)

    @torch.no_grad()
    def test_rtn_quantizes_only_decoder_and_complete_state_reloads(self):
        from qmllm.methods.rtn.quantizer import pseudo_quantize_model_weight_act
        from qmllm.quantization.qlinear import WALinear
        full = tiny_model()
        decoder = decoder_view(full)
        protected = {k:v.clone() for k,v in full.state_dict().items() if 'language_model.layers.' not in k}
        pseudo_quantize_model_weight_act(decoder, w_bit=4, a_bit=8)
        self.assertEqual(sum(isinstance(m,WALinear) for m in full.modules()), 7)
        for name, value in protected.items():
            torch.testing.assert_close(full.state_dict()[name], value, rtol=0, atol=0)
        restored = tiny_model()
        pseudo_quantize_model_weight_act(decoder_view(restored), w_bit=4, a_bit=8)
        restored.load_state_dict(full.state_dict(), strict=True)
        ids = torch.tensor([[1,2,3,4]])
        torch.testing.assert_close(full(input_ids=ids).logits,restored(input_ids=ids).logits)


if __name__ == '__main__':
    unittest.main()
