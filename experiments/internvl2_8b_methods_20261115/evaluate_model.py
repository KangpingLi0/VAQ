"""Run unchanged lmms-eval tasks/scoring through an native InternVL2 NPU adapter."""
import bootstrap
import importlib.util
import re

import torch
from tqdm import tqdm
from lmms_eval.api.model import lmms
from model_bridge import load_model


class InternVLNPU(lmms):
    def __init__(self, pretrained=None, device='npu:0', batch_size=1, torch_dtype='float16',
                 attn_implementation='eager', state_path=None, w_bit=16, a_bit=16, **kwargs):
        super().__init__()
        assert not kwargs, kwargs
        assert int(batch_size) == 1 and torch_dtype == 'float16' and attn_implementation == 'eager'
        assert pretrained == str(bootstrap.MODEL)
        self._device = torch.device(device)
        self._model, self.processor = load_model(state_path, int(w_bit), int(a_bit), device=self._device)
        self._tokenizer = self.processor
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
    def max_length(self): return self.config.llm_config.max_position_embeddings

    def loglikelihood(self, requests):
        raise NotImplementedError('These thirteen tasks use generation')

    def generate_until_multi_round(self, requests):
        raise NotImplementedError('These thirteen tasks use single-round generation')

    @torch.inference_mode()
    def generate_until(self, requests):
        answers = []
        for request in tqdm(requests, desc='Model Responding', disable=self.rank != 0):
            context, gen_kwargs, doc_to_visual, doc_id, task, split = request.args
            visuals = doc_to_visual(self.task_dict[task][split][doc_id]) or []
            if not isinstance(visuals, (list, tuple)):
                visuals = [visuals]
            from multimodal import prepare, greedy_generate
            inputs=prepare(self.model,self.tokenizer,context,visuals)
            generated=greedy_generate(self.model,inputs,int(gen_kwargs.get('max_new_tokens',256)))
            answer=self.tokenizer.decode(generated[0],skip_special_tokens=True).strip()
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
    cli.get_model = lambda name, **kwargs: InternVLNPU if name == 'internvl2' else original_get_model(name, **kwargs)
    cli.cli_evaluate()
