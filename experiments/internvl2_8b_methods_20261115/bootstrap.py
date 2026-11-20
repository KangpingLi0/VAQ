from pathlib import Path
import os
import sys

BASE = Path(__file__).resolve().parent
SOURCE = BASE / 'source'
OUT = Path(os.environ.get('LKP_INTERNVL_OUTPUT', str(BASE.parents[1] / 'outputs' / BASE.name)))
MODEL = Path(os.environ.get('LKP_INTERNVL_MODEL_DIR', '/data/lkp/models/InternVL2-8B'))
os.environ.setdefault('HF_DATASETS_CACHE', '/data/lkp/datasets/llava_onevision_hf_cache')
os.environ.setdefault('HF_MODULES_CACHE', '/data/lkp/models/hf_modules_cache')
for path in (SOURCE / '3rdparty/lmms-eval-new', SOURCE, BASE, BASE/'deps'):
    sys.path.insert(0, str(path))
