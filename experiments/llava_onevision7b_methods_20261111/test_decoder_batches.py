"""Regression: modern Qwen2 layers return tensors, older decoders tuples."""
import unittest

import bootstrap
import torch
from transformers import Qwen2Config, Qwen2Model
from qmllm.utils.device import forward_module_in_batches


class TestDecoderBatches(unittest.TestCase):
    @torch.no_grad()
    def test_two_qwen_layers_keep_batch_and_sequence(self):
        torch.manual_seed(42)
        config = Qwen2Config(vocab_size=256, hidden_size=128, intermediate_size=256,
                            num_hidden_layers=2, num_attention_heads=4,
                            num_key_value_heads=2)
        config._attn_implementation = 'eager'
        model = Qwen2Model(config).eval()
        x = torch.randn(2, 8, 128)
        positions = torch.arange(8).unsqueeze(0).expand(2, -1)
        kwargs = {'position_embeddings': model.rotary_emb(x, positions),
                  'attention_mask': None, 'use_cache': False}
        expected = x
        actual = x
        for layer in model.layers:
            expected = layer(expected, **kwargs)
            actual = forward_module_in_batches(layer, actual, kwargs, batch_size=1)
            self.assertEqual(actual.shape, (2, 8, 128))
            torch.testing.assert_close(actual, expected)

    def test_legacy_tuple_output(self):
        class Legacy(torch.nn.Module):
            def forward(self, x):
                return (x + 1, None)
        x = torch.randn(3, 8, 128)
        actual = forward_module_in_batches(Legacy(), x, batch_size=2)
        torch.testing.assert_close(actual, x + 1)


if __name__ == '__main__':
    unittest.main()
