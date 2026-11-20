"""OneVision calibration and all requested repository quantization methods."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import bootstrap
import torch
from PIL import Image
from torch.nn.functional import pad

from model_bridge import QuantizationAdapter, decoder_view, load_model


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def variant(args):
    return f'{args.method}_w{args.w}a{args.a}' + ('_smoke' if args.n == 2 else '')


def cache_path(n):
    return bootstrap.OUT / f'calibration_n{n}_seed42.pt'


class CapturedDecoderInput(Exception):
    pass


@torch.no_grad()
def capture_case(model, processor, row):
    image = Image.open(row['image']).convert('RGB')
    conversation = row['conversations']
    question = next(v['value'] for v in conversation if v['from'] == 'human').replace('<image>', '').strip()
    answer = next(v['value'] for v in conversation if v['from'] == 'gpt')
    user = {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': question}]}
    prompt = processor.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
    complete = prompt + answer + processor.tokenizer.eos_token
    prefix_ids = processor(text=prompt, images=[image], return_tensors='pt')['input_ids']
    prefix_length = prefix_ids.shape[1]
    inputs = processor(text=complete, images=[image], return_tensors='pt').to('npu:0', torch.float16)
    assert torch.equal(inputs['input_ids'][:, :prefix_length].cpu(), prefix_ids), 'Assistant boundary tokenization changed'
    result = {}

    def capture(module, args, kwargs):
        result['inputs_embeds'] = kwargs['inputs_embeds'].detach().cpu()
        raise CapturedDecoderInput

    hook = model.model.language_model.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        model(**inputs, use_cache=False)
    except CapturedDecoderInput:
        pass
    finally:
        hook.remove()
    assert result, 'Did not capture the actual multimodal decoder input'
    ids = inputs['input_ids'].cpu()
    labels = ids.clone()
    labels[:, :prefix_length] = -100
    vision = ids == model.config.image_token_index
    assert vision.any() and (labels != -100).any()
    assert not ((labels != -100) & vision).any()
    assert torch.isfinite(result['inputs_embeds']).all()
    return {**result, 'labels': labels, 'attention_mask': inputs['attention_mask'].cpu(),
            'vision_mask': vision, 'caption_mask': labels != -100}


def calibrate(args):
    if cache_path(args.n).exists() and cache_path(args.n).with_suffix('.meta.json').exists():
        return
    model, processor = load_model()
    selection = bootstrap.OUT / 'calibration_n128_seed42.json'
    rows = json.loads(selection.read_text())[:args.n]
    assert len(rows) == args.n
    samples = []
    for index, row in enumerate(rows):
        samples.append(capture_case(model, processor, row))
        print(f'Calibration {index + 1}/{args.n}, tokens={samples[-1]["labels"].shape[1]}', flush=True)
    length = max(sample['labels'].shape[1] for sample in samples)
    combined = {}
    for name in samples[0]:
        if name == 'inputs_embeds':
            combined[name] = torch.cat([pad(s[name], (0, 0, 0, length-s[name].shape[1])) for s in samples])
        else:
            combined[name] = torch.cat([pad(s[name], (0, length-s[name].shape[1]),
                value=-100 if name == 'labels' else 0) for s in samples])
    value = {'prompt_inputs': {'inputs_embeds': combined.pop('inputs_embeds')}, 'prompt_kwargs': combined}
    temporary = cache_path(args.n).with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(cache_path(args.n))
    save(cache_path(args.n).with_suffix('.meta.json'), {
        'model': str(bootstrap.MODEL), 'cases': args.n, 'seed': 42,
        'shape': list(value['prompt_inputs']['inputs_embeds'].shape),
        'selection_sha256': hashlib.sha256(selection.read_bytes()).hexdigest(),
        'preprocessing': 'Pinned HF processor, official chat template, native anyres_max_9; actual expanded multimodal embeddings',
        'labels': 'Assistant continuation only; image and prompt tokens ignored',
    })


def quantize(args):
    from qmllm.quantization.quant_wrapper import qwrapper
    torch.manual_seed(42)
    run = bootstrap.OUT / variant(args)
    run.mkdir(exist_ok=True)
    if (run/'quantization.json').exists() and (run/'quantized_state.pt').exists():
        return
    model, processor = load_model()
    adapter = QuantizationAdapter(model, processor)
    cache = None if args.method == 'rtn' else torch.load(str(cache_path(args.n)), map_location='cpu', weights_only=True, mmap=True)
    options = SimpleNamespace(
        method=args.method, model='llava_onevision', model_args=f'pretrained={bootstrap.MODEL}',
        w_bit=args.w, a_bit=args.a, w_group=128, alpha=.5, run_process=True,
        pseudo_quant=True, reweight=args.method in ('mbq', 'qig'), distort=False,
        loss_mode='mae', scale_path=str(run/'scales.pt'), torch_device=torch.device('npu:0'),
        percdamp=.01, token_aware_saliency=args.method == 'lat_awq',
        token_weighted_loss=args.method == 'lat_awq', saliency_mix_lambda=.5,
        lat_debug=True, lat_log_dir=str(run/'search_logs'), lat_output_dir=str(run/'search_metadata'),
        effective_n_samples=0 if args.method == 'rtn' else args.n, seed=42,
    )
    before = adapter.model.model.layers[0].self_attn.q_proj.weight[:1].detach().cpu().clone()
    save(run/'configuration.json', {
        **{k:v for k,v in vars(options).items() if k != 'torch_device'},
        'calibration_selection_sha256': None if args.method == 'rtn' else hashlib.sha256((bootstrap.OUT/'calibration_n128_seed42.json').read_bytes()).hexdigest(),
        'scope': 'Actual Qwen2 decoder linear layers only; vision, projector, embeddings and LM head remain FP16',
        'decoder_bridge': 'Qwen2ForCausalLM view sharing the actual HF OneVision decoder and head; no weight conversion',
        'preprocessing': 'Same pinned processor, native anyres_max_9 and official chat template for every method',
    })
    started = time.time()
    if args.method in ('awq', 'mbq', 'qig', 'lat_awq'):
        options.pseudo_quant = False
        qwrapper(adapter, cache['prompt_inputs'], cache['prompt_kwargs'], options)
        adapter.to_cpu()
        from transformers import LlavaOnevisionForConditionalGeneration
        original = LlavaOnevisionForConditionalGeneration.from_pretrained(str(bootstrap.MODEL),
            torch_dtype=torch.float16, attn_implementation='sdpa', local_files_only=True)
        adapter.model.load_state_dict(decoder_view(original).state_dict(), strict=True)
        del original
        gc.collect()
        options.run_process = False
        options.pseudo_quant = True
        qwrapper(adapter, None, None, options)
    else:
        qwrapper(adapter, cache['prompt_inputs'] if cache else None, cache['prompt_kwargs'] if cache else None, options)
    after = adapter.model.model.layers[0].self_attn.q_proj.weight[:1].detach().cpu()
    assert not torch.equal(before, after), 'Quantization did not change weights'
    for layer in adapter.model.model.layers:
        for module in layer.modules():
            if hasattr(module, 'weight'):
                assert torch.isfinite(module.weight).all(), 'Non-finite quantized weights'
    assert adapter.model.model is model.model.language_model
    wrappers = sum(type(module).__name__ == 'WALinear' for module in model.modules())
    assert wrappers == (7*len(adapter.model.model.layers) if args.a < 16 else 0), wrappers
    temporary = run/'quantized_state.tmp'
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()}, temporary)
    temporary.replace(run/'quantized_state.pt')
    save(run/'quantization.json', {'complete': True, 'seconds': time.time()-started,
        'method': args.method, 'w_bit': args.w, 'a_bit': args.a, 'calibration_cases': 0 if args.method == 'rtn' else args.n,
        'finite_weights_verified': True, 'reference_weight_changed': True, 'activation_wrappers': wrappers,
        'search_applied_once_to_fresh_decoder': args.method in ('awq', 'mbq', 'qig', 'lat_awq'),
        'representation': 'Fake quantization, tensors stored in FP16'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['calibration', 'quant'])
    parser.add_argument('--n', type=int, default=128)
    parser.add_argument('--method', choices=['rtn', 'awq', 'gptq', 'mbq', 'qig', 'lat_awq', 'smoothquant'])
    parser.add_argument('--w', type=int, default=3)
    parser.add_argument('--a', type=int, default=16)
    args = parser.parse_args()
    if args.mode == 'calibration':
        calibrate(args)
    else:
        assert (args.a == 16 and args.w == 3 and args.method != 'smoothquant') or (args.a == 8 and args.w == 4 and args.method not in ('awq', 'gptq'))
        quantize(args)
