"""Original Qwen CLI/state reload; budget memory when sharing with InternVL."""
from pathlib import Path
import importlib.util
import os
import sys

BASE=Path(__file__).resolve().parent
SOURCE=BASE/'source'
sys.path[:0]=[str(SOURCE/'3rdparty/lmms-eval-new'),str(SOURCE)]
os.environ.setdefault('HF_DATASETS_CACHE','/data/lkp/datasets/llava_onevision_hf_cache')
import torch
import torch_npu
torch.npu.set_device(0)
torch.npu.set_per_process_memory_fraction(float(os.environ.get('QWEN_EXTRA_MEMORY_FRACTION','0.44')),0)
spec=importlib.util.spec_from_file_location('qwen_evaluation_cli',SOURCE/'main.py')
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
cli.cli_evaluate()
