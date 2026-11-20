import bootstrap
import unittest
from unittest.mock import patch
import torch
from test_bridge import tiny_model
from model_bridge import QuantizationAdapter
from memory_compat import install


class TestMemoryCompatibility(unittest.TestCase):
    def test_checkpointing_preserves_loss_and_input_gradient(self):
        torch.manual_seed(42);full=tiny_model();adapter=QuantizationAdapter(full)
        for parameter in full.parameters():parameter.requires_grad_(False)
        x=torch.randn(1,8,128,requires_grad=True)
        labels=torch.tensor([[-100,-100,-100,4,5,6,7,8]])
        ref=adapter(inputs_embeds=x,labels=labels,use_cache=False)
        ref.loss.backward();expected=x.grad.clone();x.grad=None
        adapter.model.gradient_checkpointing_enable()
        from torch.utils.checkpoint import checkpoint
        with patch('torch.utils.checkpoint.checkpoint',wraps=checkpoint) as wrapped:
            output=adapter(inputs_embeds=x,labels=labels,use_cache=False)
            output.loss.backward()
            self.assertEqual(wrapped.call_count,2)
        torch.testing.assert_close(ref.loss,output.loss)
        torch.testing.assert_close(x.grad,expected)
        self.assertFalse(adapter.model.training)

    @torch.no_grad()
    def test_query_chunks_preserve_masked_prefill_logits_and_cached_decode(self):
        torch.manual_seed(42);full=tiny_model()
        ids=torch.tensor([[1,2,3,4,5,6,7,8],[1,1,1,4,5,6,7,8]])
        mask=torch.ones_like(ids);mask[1,:3]=0
        ref=full.language_model(input_ids=ids,attention_mask=mask,use_cache=True)
        next_ids=ref.logits[:,-1,:].argmax(-1,keepdim=True)
        next_mask=torch.cat([mask,torch.ones_like(next_ids)],dim=1)
        ref_decode=full.language_model(input_ids=next_ids,attention_mask=next_mask,past_key_values=ref.past_key_values,use_cache=True)
        install(full,query_chunk=3)
        out=full.language_model(input_ids=ids,attention_mask=mask,use_cache=True)
        decoded=full.language_model(input_ids=next_ids,attention_mask=next_mask,past_key_values=out.past_key_values,use_cache=True)
        torch.testing.assert_close(out.logits,ref.logits,atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(decoded.logits,ref_decode.logits,atol=1e-5,rtol=1e-5)


if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main()
