"""Run unchanged lmms-eval tasks/scoring through an HF OneVision NPU adapter."""
import bootstrap
import importlib.util
import re

import torch
from tqdm import tqdm
from lmms_eval.api.model import lmms
from model_bridge import load_model


class OneVisionNPU(lmms):
    def __init__(self, pretrained=None, device='npu:0', batch_size=1, torch_dtype='float16',
                 attn_implementation='sdpa', state_path=None, w_bit=16, a_bit=16, **kwargs):
        super().__init__()
        assert not kwargs, kwargs
        assert int(batch_size) == 1 and torch_dtype == 'float16' and attn_implementation == 'sdpa'
        assert pretrained == str(bootstrap.MODEL)
        self._device = torch.device(device)
        self._model, self.processor = load_model(state_path, int(w_bit), int(a_bit), device=self._device)
        self._tokenizer = self.processor.tokenizer
        self._config = self._model.config
        self.batch_size_per_gpu = 1

    @property
    def model(self): return self._model

    @property
    def config(self): return self._config

    @property
    def tokenizer(self): return self._tokenizer

    @property
    def device(self): return self._device

    @property
    def batch_size(self): return 1

    @property
    def eot_token_id(self): return self.tokenizer.eos_token_id

    @property
    def max_length(self): return self.config.text_config.max_position_embeddings

    def loglikelihood(self, requests):
        raise NotImplementedError('These eight tasks use generation')

    def generate_until_multi_round(self, requests):
        raise NotImplementedError('These eight tasks use single-round generation')

    @torch.inference_mode()
    def generate_until(self, requests):
        answers = []
        for request in tqdm(requests, desc='Model Responding', disable=self.rank != 0):
            context, gen_kwargs, doc_to_visual, doc_id, task, split = request.args
            visuals = doc_to_visual(self.task_dict[task][split][doc_id]) or []
            if not isinstance(visuals, (list, tuple)):
                visuals = [visuals]
            context = re.sub(r'<image(?:\s+\d+)?>', '', context).strip()
            content = [{'type': 'image'} for _ in visuals] + [{'type': 'text', 'text': context}]
            text = self.processor.apply_chat_template([{'role': 'user', 'content': content}],
                tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=text, images=[list(visuals)] if visuals else None,
                return_tensors='pt').to(self.device, torch.float16)
            generated = self.model.generate(**inputs, do_sample=False, max_new_tokens=int(gen_kwargs.get('max_new_tokens',256)),
                num_beams=1, use_cache=True, logits_to_keep=1,
                pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id,
                temperature=None, top_p=None, top_k=None)
            answer = self.processor.decode(generated[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            until = gen_kwargs.get('until', [])
            if isinstance(until, str): until = [until]
            for stop in until:
                if stop: answer = answer.split(stop)[0]
            answers.append(answer)
            self.cache_hook.add_partial('generate_until', (context, gen_kwargs), answer)
        return answers


if __name__ == '__main__':
    spec = importlib.util.spec_from_file_location('original_evaluation_cli', bootstrap.SOURCE/'main.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    original_get_model = cli.get_model
    cli.get_model = lambda name, **kwargs: OneVisionNPU if name == 'llava_onevision' else original_get_model(name, **kwargs)
    cli.cli_evaluate()
