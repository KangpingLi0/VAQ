from pathlib import Path
import os
import sys

# Full MME-RealWorld Arrow materialization exceeds the root partition's space.
# Keep this campaign's dataset cache on the data volume.
os.environ.setdefault('HF_DATASETS_CACHE', '/data/lkp/datasets/llava_onevision_hf_cache')

BASE = Path(__file__).resolve().parent
SOURCE = BASE / 'source'
OUT = Path(os.environ.get('LKP_ONEVISION_OUTPUT', str(BASE.parents[1] / 'outputs' / BASE.name)))
MODEL = Path(os.environ.get('LKP_ONEVISION_MODEL_DIR', '/data/lkp/models/llava-onevision-qwen2-7b-ov-hf'))
for path in (SOURCE / '3rdparty/lmms-eval-new', SOURCE, BASE):
    sys.path.insert(0, str(path))
