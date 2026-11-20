"""Native pinned InternVL/InternLM2 modules, with an accelerator adapter."""
import bootstrap
import torch
import torch_npu
from transformers import AutoModel, AutoTokenizer


def decoder_view(full):
    return full.language_model


class QuantizationAdapter:
    def __init__(self, full_model, processor=None):
        self.model = full_model.language_model
        self.tokenizer = processor
        self.device = torch.device('npu:0')
        self.num_params = sum(p.numel() for p in self.model.parameters())
        self.lm_head = self.model.get_output_embeddings()

    def to_cpu(self): self.model.cpu()
    def to_cuda(self): self.model.to(self.device)
    def to_device(self, device=None): self.model.to(device or self.device)
    def set_device(self, device): self.device = torch.device(device)
    def __call__(self, **kwargs):
        # The native InternLM2 forward activates checkpointing only in train
        # mode. Its decoder has no dropout; enable it only for gradient passes.
        was_training=self.model.training
        if torch.is_grad_enabled() and getattr(self.model.model,'gradient_checkpointing',False):
            self.model.train()
        try:return self.model(**kwargs)
        finally:self.model.train(was_training)


def load_model(state_path=None, w_bit=16, a_bit=16, device='npu:0'):
    torch.set_num_threads(8)
    if str(device).startswith('npu'):torch.npu.set_device(0)
    full = AutoModel.from_pretrained(str(bootstrap.MODEL), trust_remote_code=True,
        local_files_only=True, torch_dtype=torch.float16, use_flash_attn=False,
        attn_implementation='eager').eval()
    from memory_compat import install
    install(full)
    if state_path:
        if a_bit < 16:
            from qmllm.methods.rtn.quantizer import pseudo_quantize_model_weight_act
            pseudo_quantize_model_weight_act(decoder_view(full), w_bit=w_bit, a_bit=a_bit)
        state = torch.load(str(state_path), map_location='cpu', weights_only=True, mmap=True)
        full.load_state_dict(state, strict=True)
        del state
    tokenizer = AutoTokenizer.from_pretrained(str(bootstrap.MODEL), trust_remote_code=True,
        local_files_only=True, use_fast=False)
    tokenizer.padding_side='left'
    full.img_context_token_id=tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
    full.to(device)
    print('Native InternVL2 model and tokenizer ready on', device, flush=True)
    return full, tokenizer
