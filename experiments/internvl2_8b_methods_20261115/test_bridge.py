import bootstrap
import copy
import unittest
import torch
from PIL import Image
from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from model_bridge import decoder_view
from multimodal import prepare, greedy_generate


def tiny_model():
    config=AutoConfig.from_pretrained(str(bootstrap.MODEL),trust_remote_code=True,local_files_only=True)
    text=config.llm_config
    text.hidden_size=128;text.intermediate_size=256;text.num_hidden_layers=2
    text.num_attention_heads=4;text.num_key_value_heads=2
    vision=config.vision_config
    vision.hidden_size=32;vision.intermediate_size=64;vision.num_hidden_layers=1
    vision.num_attention_heads=4;vision.image_size=28
    config.force_image_size=28;config.min_dynamic_patch=1;config.max_dynamic_patch=2
    cls=get_class_from_dynamic_module('modeling_internvl_chat.InternVLChatModel',str(bootstrap.MODEL),local_files_only=True)
    return cls(config,use_flash_attn=False).eval()


def tokenizer():
    return AutoTokenizer.from_pretrained(str(bootstrap.MODEL),trust_remote_code=True,
        local_files_only=True,use_fast=False)


class TestNativeInternVL(unittest.TestCase):
    @torch.no_grad()
    def test_multimodal_embeddings_match_native_forward_and_labels(self):
        torch.manual_seed(42);full=tiny_model();tok=tokenizer()
        full.img_context_token_id=tok.convert_tokens_to_ids('<IMG_CONTEXT>')
        value=prepare(full,tok,'Describe the image.',[Image.new('RGB',(28,28),'red')],answer='A red square.')
        captured={}
        def capture(module,args,kwargs):captured['embeds']=kwargs['inputs_embeds'].detach().clone()
        hook=full.language_model.register_forward_pre_hook(capture,with_kwargs=True)
        native=full(pixel_values=value['pixel_values'],input_ids=value['input_ids'],
            attention_mask=value['attention_mask'],image_flags=torch.ones(1,1,dtype=torch.long),labels=value['labels'])
        hook.remove()
        torch.testing.assert_close(captured['embeds'],value['inputs_embeds'],rtol=0,atol=0)
        decoder=full.language_model(inputs_embeds=value['inputs_embeds'],attention_mask=value['attention_mask'],labels=value['labels'])
        torch.testing.assert_close(native.logits,decoder.logits)
        torch.testing.assert_close(native.loss,decoder.loss)
        self.assertTrue(value['caption_mask'].any())
        self.assertFalse((value['caption_mask']&value['vision_mask']).any())

    @torch.no_grad()
    def test_native_cached_greedy_equals_uncached_full_forward(self):
        torch.manual_seed(42);full=tiny_model();tok=tokenizer()
        full.img_context_token_id=tok.convert_tokens_to_ids('<IMG_CONTEXT>')
        for images in [[],[Image.new('RGB',(28,28),'red')],
                       [Image.new('RGB',(28,28),'red'),Image.new('RGB',(28,28),'blue')]]:
            value=prepare(full,tok,'Describe the images.',images)
            value['eos_token_id']=-1
            generated=greedy_generate(full,value,3)
            embeds=value['inputs_embeds'];mask=value['attention_mask'];expected=[]
            for step in range(3):
                out=full.language_model(inputs_embeds=embeds,attention_mask=mask,use_cache=False)
                token=out.logits[:,-1,:].argmax(-1,keepdim=True);expected.append(token)
                embeds=torch.cat([embeds,full.language_model.get_input_embeddings()(token)],dim=1)
                mask=torch.cat([mask,torch.ones_like(token)],dim=1)
            torch.testing.assert_close(generated,torch.cat(expected,dim=1),rtol=0,atol=0)

    @torch.no_grad()
    def test_quantization_scope_and_full_state_reload(self):
        from qmllm.methods.rtn.quantizer import pseudo_quantize_model_weight_act
        torch.manual_seed(42);full=tiny_model()
        protected={k:v.clone() for k,v in full.state_dict().items() if not k.startswith('language_model.model.layers.')}
        pseudo_quantize_model_weight_act(decoder_view(full),w_bit=4,a_bit=8)
        self.assertEqual(sum(type(m).__name__=='WALinear' for m in full.modules()),10)
        for key,value in protected.items():torch.testing.assert_close(full.state_dict()[key],value,rtol=0,atol=0)
        restored=tiny_model();pseudo_quantize_model_weight_act(decoder_view(restored),w_bit=4,a_bit=8)
        restored.load_state_dict(full.state_dict(),strict=True)
        ids=torch.tensor([[1,2,3,4]])
        torch.testing.assert_close(full.language_model(input_ids=ids).logits,
                                   restored.language_model(input_ids=ids).logits,rtol=0,atol=0)


if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main()
