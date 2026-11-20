import importlib
import os
import sys

import hf_transfer
from loguru import logger

from qmllm.utils.registry import MODEL_REGISTRY

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logger.remove()
logger.add(sys.stdout, level="WARNING")

def get_process_model(model_name):
    # Import only the requested adapter. Optional model packages need not be
    # installed to quantize an unrelated architecture.
    module_name = {"llava": "llava_v15"}.get(model_name, model_name)
    importlib.import_module(f"qmllm.models.{module_name}")
    return MODEL_REGISTRY[model_name]
