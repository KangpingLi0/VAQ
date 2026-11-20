"""Checkpoint-1660 endoscopy quantization and reproducible full-set evaluation."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
DEFAULT_OUT = REPO / "outputs/neijing_checkpoint1660_20261107"
CHECKPOINT = "/data/lkp/最后使用版本/checkpoint-1660"


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text())


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model(args):
    import torch
    import torch_npu  # registers the backend
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    torch.set_num_threads(24 if getattr(args, "method", None) == "gptq" else 8)
    torch.manual_seed(42)
    torch.npu.set_device(0)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CHECKPOINT, torch_dtype=torch.float16, attn_implementation="sdpa",
        local_files_only=True,
    ).eval().to("npu:0")
    processor = AutoProcessor.from_pretrained(
        CHECKPOINT, min_pixels=224 * 224, max_pixels=224 * 224,
        local_files_only=True, use_fast=False,
    )
    processor.tokenizer.padding_side = "left"
    model.config.use_cache = True
    model.config.text_config.use_cache = True
    return model, processor


def calibration(args):
    import torch
    from qmllm.models import get_process_model
    from qmllm.calibration.coco_vl import get_multimodal_calib_dataset

    model, processor = load_model(args)
    adapter = get_process_model("qwen2_5_vl")(model, processor.tokenizer, processor)
    adapter.set_device("npu:0")
    count = args.limit or len(load_json(args.output / "calibration.json"))
    inputs, kwargs = get_multimodal_calib_dataset(
        str(args.output / "calibration.json"), "", adapter,
        n_samples=count, micro_bs=1, image_size=224, shuffle=False,
    )
    path = args.output / ("calibration_smoke.pt" if args.limit else "calibration.pt")
    temporary = path.with_suffix(".tmp")
    torch.save({"prompt_inputs": inputs, "prompt_kwargs": kwargs}, temporary)
    temporary.replace(path)
    save_json(path.with_suffix(".meta.json"), {
        "cases": count, "input_shape": list(inputs["inputs_embeds"].shape),
        "image_size": 224, "all_images": True, "checkpoint": CHECKPOINT,
    })


def variant_name(args):
    return "fp16" if args.method == "fp16" else f"{args.method}_w{args.w_bit}a{args.a_bit}"


def quantize(args, model, processor):
    import torch
    from types import SimpleNamespace
    from qmllm.models import get_process_model
    from qmllm.quantization.quant_wrapper import qwrapper

    if args.a_bit < 16 and args.method in ("awq", "gptq"):
        raise ValueError(f"{args.method} only supports weight quantization in this repository")
    if args.method == "smoothquant" and args.a_bit >= 16:
        raise ValueError("SmoothQuant is restricted to weight-and-activation quantization for this experiment")
    adapter = get_process_model("qwen2_5_vl")(model, processor.tokenizer, processor)
    adapter.set_device("npu:0")
    run = args.output / variant_name(args)
    run.mkdir(parents=True, exist_ok=True)
    state_path = run / "quantized_state.pt"
    if state_path.exists():
        manifest = load_json(run / "quantization.json")
        # Recreate activation wrappers before loading their buffers.
        if args.a_bit < 16 or args.method == "smoothquant":
            from qmllm.methods.rtn.quantizer import pseudo_quantize_model_weight_act
            pseudo_quantize_model_weight_act(model, args.w_bit, args.a_bit)
        state = torch.load(str(state_path), map_location="cpu", weights_only=True, mmap=True)
        model.load_state_dict(state, strict=True)
        del state
        return model
    cache_path = args.output / ("calibration_smoke.pt" if args.smoke else "calibration.pt")
    cache = None if args.method == "rtn" else torch.load(str(cache_path), map_location="cpu", weights_only=True, mmap=True)
    options = SimpleNamespace(
        method=args.method, w_bit=args.w_bit, a_bit=args.a_bit, w_group=128,
        alpha=0.5, run_process=True, pseudo_quant=True, reweight=True,
        distort=False, loss_mode="mae", scale_path=str(run / "scales.pt"),
        torch_device=torch.device("npu:0"), percdamp=0.01,
        model="qwen2_5_vl", model_args=f"pretrained={CHECKPOINT}",
        token_aware_saliency=args.method == "lat_awq",
        token_weighted_loss=args.method == "lat_awq", saliency_mix_lambda=1.0,
        lat_debug=False, lat_log_dir=str(run / "logs"),
        lat_output_dir=str(run), effective_n_samples=2 if args.smoke else 512,
        seed=42,
    )
    started = time.time()
    from qmllm.utils.hf_compat import get_qwen_vl_layers
    before = get_qwen_vl_layers(model)[0].self_attn.q_proj.weight[:1].detach().cpu().clone()
    if args.method in ("awq", "mbq", "qig", "lat_awq"):
        # Search applies scales in-place while traversing the layers. Restore
        # the original checkpoint before applying the saved scales exactly once.
        options.pseudo_quant = False
        qwrapper(adapter, cache["prompt_inputs"], cache["prompt_kwargs"], options)
        adapter.to_cpu()
        from transformers import Qwen2_5_VLForConditionalGeneration
        original = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            CHECKPOINT, torch_dtype=torch.float16, attn_implementation="sdpa", local_files_only=True,
        )
        adapter.model.load_state_dict(original.state_dict(), strict=True)
        del original
        options.run_process = False
        options.pseudo_quant = True
        qwrapper(adapter, None, None, options)
    else:
        qwrapper(adapter, cache["prompt_inputs"] if cache else None,
                 cache["prompt_kwargs"] if cache else None, options)
    model = adapter.model
    after = get_qwen_vl_layers(model)[0].self_attn.q_proj.weight[:1].detach().cpu()
    assert not torch.equal(before, after), "quantization did not change the reference weight"
    for layer in get_qwen_vl_layers(model):
        for module in layer.modules():
            if hasattr(module, "weight"):
                assert torch.isfinite(module.weight).all(), "non-finite quantized weights"
    # Save buffers as well as parameters: W4A8 linears store weights in buffers.
    temporary = state_path.with_suffix(".tmp")
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, temporary)
    temporary.replace(state_path)
    save_json(run / "quantization.json", {
        **{k: v for k, v in vars(options).items() if k != "torch_device"},
        "seconds": time.time() - started,
        "representation": "fake_quantized_fp16_tensors",
        "scope": "language decoder linear layers; vision, embeddings and lm_head remain FP16",
        "activation_wrappers": sum(type(m).__name__ == "WALinear" for m in model.modules()),
        "calibration_cache": str(cache_path) if cache else None,
        "checkpoint": CHECKPOINT,
        "reference_weight_changed": True,
        "calibration_cases": 0 if args.method == "rtn" else (2 if args.smoke else 512),
        "search_applied_to_fresh_checkpoint": args.method in ("awq", "mbq", "qig", "lat_awq"),
        "scale_search_batch_size": int(os.environ.get("QIG_SCALE_SEARCH_BATCH_SIZE", "8")),
    })
    return model


def build_case_cache(model, processor, row, path):
    import torch
    from PIL import Image

    images = []
    for source in row["images"]:
        with Image.open(source) as image:
            images.append(image.convert("RGB").resize((224, 224), Image.Resampling.LANCZOS))
    messages = [{"role": "user", "content":
                 [{"type": "image"} for _ in images] + [{"type": "text", "text": row["question"]}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = processor(text=[text], images=images, padding=True, return_tensors="pt")
    with torch.inference_mode():
        features = model.visual(batch["pixel_values"].to("npu:0", dtype=torch.float16),
                                grid_thw=batch["image_grid_thw"].to("npu:0")).cpu()
    assert torch.isfinite(features).all(), f"non-finite image features: {row['id']}"
    cache = {k: v for k, v in batch.items() if k != "pixel_values"}
    cache["image_features"] = features
    temporary = path.with_suffix(".tmp")
    torch.save(cache, temporary)
    temporary.replace(path)
    return cache


def infer(args):
    import torch

    model, processor = load_model(args)
    if args.method != "fp16":
        model = quantize(args, model, processor)
    if args.mode == "quant":
        return
    model.eval()
    def check_logits(module, inputs, logits):
        if not torch.isfinite(logits).all():
            raise FloatingPointError("non-finite generation logits")
    model.lm_head.register_forward_hook(check_logits)
    rows = load_json(args.output / "test.json")
    if args.limit:
        rows = rows[:args.limit]
    run = args.output / variant_name(args)
    run.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output / "test_vision_cache"
    cache_dir.mkdir(exist_ok=True)
    save_json(run / f"inference.part{args.shard:02d}.json", {
        "method": args.method, "checkpoint": CHECKPOINT, "dtype": "float16",
        "image_size": [224, 224], "all_images": True,
        "processor": "slow", "batch_size": args.batch_size,
        "do_sample": False, "max_new_tokens": 1024, "repetition_penalty": 1.05,
        "shard": args.shard, "shards": args.shards,
        "test_limit": args.limit or None,
    })
    prediction_path = run / f"predictions.part{args.shard:02d}.jsonl"
    done = {}
    if prediction_path.exists():
        for line in prediction_path.read_text().splitlines():
            item = json.loads(line)
            done[item["id"]] = item
    selected = [(i, row) for i, row in enumerate(rows)
                if i % args.shards == args.shard and row["id"] not in done]
    original_forward = model.visual.forward
    started = time.time()
    with prediction_path.open("a", buffering=1) as output:
        for step in range(0, len(selected), args.batch_size):
            current = selected[step:step + args.batch_size]
            caches = []
            for index, row in current:
                cache_path = cache_dir / f"{index:05d}.pt"
                caches.append(torch.load(cache_path, map_location="cpu", weights_only=True)
                              if cache_path.exists() else build_case_cache(model, processor, row, cache_path))
            max_length = max(c["input_ids"].shape[1] for c in caches)
            from torch.nn.functional import pad
            batch = {
                "input_ids": torch.cat([pad(c["input_ids"], (max_length - c["input_ids"].shape[1], 0),
                                            value=processor.tokenizer.pad_token_id) for c in caches]).to("npu:0"),
                "attention_mask": torch.cat([pad(c["attention_mask"], (max_length - c["input_ids"].shape[1], 0))
                                              for c in caches]).to("npu:0"),
                "image_grid_thw": torch.cat([c["image_grid_thw"] for c in caches]).to("npu:0"),
            }
            features = torch.cat([c["image_features"] for c in caches]).to("npu:0")
            batch["pixel_values"] = torch.zeros(1, device="npu:0", dtype=torch.float16)
            model.visual.forward = lambda *a, **kw: features
            with torch.inference_mode():
                generated = model.generate(
                    **batch, do_sample=False, max_new_tokens=1024, use_cache=True,
                    repetition_penalty=1.05, temperature=None, top_p=None, top_k=None,
                    logits_to_keep=1,
                )
            model.visual.forward = original_forward
            eos_ids = model.generation_config.eos_token_id
            eos_ids = [eos_ids] if isinstance(eos_ids, int) else eos_ids
            for position, (index, row) in enumerate(current):
                token_ids = generated[position, max_length:].cpu().tolist()
                end = next((i + 1 for i, token in enumerate(token_ids) if token in eos_ids), len(token_ids))
                token_ids = token_ids[:end]
                answer = processor.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                record = {"id": row["id"], "image_id": index, "pred": answer,
                          "label": row["label"], "date": row.get("date"),
                          "generated_tokens": len(token_ids),
                          "hit_token_limit": len(token_ids) >= 1024 and token_ids[-1] not in eos_ids}
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            del generated, batch, caches, features
            completed = len(done) + step + len(current)
            save_json(run / f"progress.part{args.shard:02d}.json", {
                "completed": completed, "assigned": len(done) + len(selected),
                "seconds_this_session": time.time() - started,
                "last_id": row["id"], "last_output_tokens": len(token_ids),
            })
            print(f"{variant_name(args)} shard {args.shard}: {completed}/{len(done) + len(selected)} "
                  f"{time.time() - started:.1f}s", flush=True)
    save_json(run / f"done.part{args.shard:02d}.json", {"complete": True, "cases": len(done) + len(selected)})


def metrics(args):
    run = args.output / variant_name(args)
    reference = load_json(args.output / "test.json")
    if args.limit:
        reference = reference[:args.limit]
    predictions = {}
    for path in sorted(run.glob("predictions.part*.jsonl")):
        for line in path.read_text().splitlines():
            item = json.loads(line)
            if item["id"] in predictions:
                raise ValueError(f"duplicate prediction {item['id']}")
            predictions[item["id"]] = item
    assert set(predictions) == {row["id"] for row in reference}, (len(predictions), len(reference))
    data = [predictions[row["id"]] for row in reference]
    assert all(item["label"] == row["label"] for item, row in zip(data, reference))
    save_json(run / "predictions.json", data)
    clinical = load_module("endoscopy_multilevel", Path(args.eval_dir) / "eval_multi_level.py")
    lines = clinical.compute_multi_level_metrics(data, *clinical.load_keyword_data())
    (run / "medical_metrics.txt").write_text(f"sample_count: {len(data)}\n" + "\n".join(lines) + "\n")
    save_json(run / "medical_metrics.json", {"sample_count": len(data), "lines": lines,
              "evaluator": str(Path(args.eval_dir) / "eval_multi_level.py")})
    if args.nlp:
        nlg = load_module("endoscopy_nlg", Path(args.eval_dir) / "NLG指标计算.py")
        from caption_metrics import StandardCaptionEvaluator
        nlg.COCOEvalCap = StandardCaptionEvaluator
        scores = nlg.coco_caption_eval(data)
        save_json(run / "nlg_metrics.json", scores)
        save_json(run / "nlg_status.json", {
            "standard_caption_metrics": "complete",
            "SPICE": "not_computed",
            "reason": "The supplied SPICE dependency uses an English constituency parser; full Chinese reports caused parser out-of-memory and invalid parses. No SPICE score is imputed.",
            "metrics": list(scores), "sample_count": len(data),
        })
    print("\n".join(lines[:11]), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["calibration", "infer", "quant", "metrics"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--method", default="fp16",
                        choices=["fp16", "rtn", "awq", "gptq", "mbq", "qig", "lat_awq", "smoothquant"])
    parser.add_argument("--w-bit", type=int, default=4)
    parser.add_argument("--a-bit", type=int, default=16)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--eval-dir", default="/data/lkp/eval")
    parser.add_argument("--nlp", action="store_true")
    args = parser.parse_args()
    if args.mode == "calibration":
        calibration(args)
    elif args.mode == "metrics":
        metrics(args)
    else:
        infer(args)


if __name__ == "__main__":
    main()
