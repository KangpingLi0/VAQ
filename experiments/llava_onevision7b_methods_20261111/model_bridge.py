"""Expose HF OneVision's actual Qwen2 decoder to the existing quantizers."""
import bootstrap
import torch
import torch_npu  # registers the NPU backend
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, Qwen2ForCausalLM


class LlavaQwenForCausalLM(Qwen2ForCausalLM):
    """The existing LLaVA quantizer dispatch expects this decoder class name.

    No weights or forward equations are converted: the model and head below
    are the actual HF OneVision modules, shared by reference.
    """


def decoder_view(full_model):
    config = full_model.config.text_config
    with torch.device('meta'):
        view = LlavaQwenForCausalLM(config)
    view.model = full_model.model.language_model
    view.lm_head = full_model.lm_head
    view.eval()
    assert view.model is full_model.model.language_model
    assert view.lm_head is full_model.lm_head
    return view


class QuantizationAdapter:
    def __init__(self, full_model, processor=None):
        self.model = decoder_view(full_model)
        self.processor = processor
        self.tokenizer = processor.tokenizer if processor is not None else None
        self.device = torch.device('npu:0')
        self.num_params = sum(p.numel() for p in self.model.parameters())
        self.lm_head = self.model.lm_head

    def set_device(self, device):
        self.device = torch.device(device)

    def to_device(self, device=None):
        self.model.to(device or self.device)

    def to_cuda(self):
        self.to_device()

    def to_cpu(self):
        self.model.cpu()

    def __call__(self, **kwargs):
        return self.model(**kwargs)


def load_model(state_path=None, w_bit=16, a_bit=16, device='npu:0'):
    torch.set_num_threads(8)
    torch.npu.set_device(0)
    full = LlavaOnevisionForConditionalGeneration.from_pretrained(
        str(bootstrap.MODEL), torch_dtype=torch.float16,
        attn_implementation='sdpa', local_files_only=True,
    ).eval()
    if state_path:
        if a_bit < 16:
            from qmllm.methods.rtn.quantizer import pseudo_quantize_model_weight_act
            pseudo_quantize_model_weight_act(decoder_view(full), w_bit=w_bit, a_bit=a_bit)
        state = torch.load(str(state_path), map_location='cpu', weights_only=True, mmap=True)
        full.load_state_dict(state, strict=True)
        del state
        print('Strictly loaded complete OneVision state:', state_path, flush=True)
    full.to(device)
    processor = AutoProcessor.from_pretrained(str(bootstrap.MODEL), use_fast=False, local_files_only=True)
    processor.tokenizer.padding_side = 'left'
    print('OneVision model and processor ready on', device, flush=True)
    return full, processor
