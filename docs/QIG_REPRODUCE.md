# QIG Reproduction For VAQ

## Repository

- Official QIG source: https://github.com/ucas-xiang/QIG
- Target GitHub repository: https://github.com/KangpingLi0/VAQ
- Local repository directory: `/work/model/lkp/QIG`
- Branch: `qig_reproduce`

Do not clone repositories, save model weights, save datasets, write caches, write logs, write scale files, or write experiment outputs under `/home/lkp`. The only valid working root for this reproduction is `/work/model/lkp`.

## Directory Layout

```text
/work/model/lkp/QIG                    # repository code
/work/model/lkp/cache                  # all caches
/work/model/lkp/models                 # local model directories
/work/model/lkp/datasets               # calibration and evaluation data
/work/model/lkp/experiments/qig        # QIG outputs, logs, scale files
```

The repository `.gitignore` excludes model weights, datasets, caches, experiment outputs, logs, scale caches, and checkpoint formats such as `*.pt`, `*.pth`, `*.bin`, `*.safetensors`, and `*.ckpt`.

## Cache Environment

All QIG run scripts source `scripts/qig/env_qig.sh` before doing any work. The script sets:

```bash
WORK_ROOT=/work/model/lkp
REPO_DIR=/work/model/lkp/QIG
CACHE_ROOT=/work/model/lkp/cache
MODEL_ROOT=/work/model/lkp/models
DATA_ROOT=/work/model/lkp/datasets
EXP_ROOT=/work/model/lkp/experiments/qig
```

It redirects `XDG_CACHE_HOME`, `HF_HOME`, `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE`, `TORCH_HOME`, `TRITON_CACHE_DIR`, `PIP_CACHE_DIR`, `WANDB_DIR`, `WANDB_CACHE_DIR`, and `TMPDIR` into `/work/model/lkp/cache`.

If any required root path or the current working directory is under `/home/lkp`, the environment script exits with an error.

## Environment Installation

Follow the official QIG installation flow from the repository root:

```bash
cd /work/model/lkp/QIG
source scripts/qig/env_qig.sh

conda create -n qig python=3.11
conda activate qig

pip install -e 3rdparty/LLaVA-NeXT
pip install -e 3rdparty/lmms-eval
pip install -r requirements.txt
pip install -e .
```

Because `PIP_CACHE_DIR` is set by `env_qig.sh`, pip cache files are written to `/work/model/lkp/cache/pip`.

## Model Paths

Place or symlink local models under `/work/model/lkp/models`:

```text
/work/model/lkp/models/qwen2-vl
/work/model/lkp/models/qwen2.5-vl
/work/model/lkp/models/qwen3-vl
```

Each run script supports `MODEL_PATH=...` override, but the override must not point under `/home/lkp`.

## Data Paths

Default data locations:

```text
/work/model/lkp/datasets/calib/calib.json
/work/model/lkp/datasets/images
/work/model/lkp/datasets/eval/eval.json
```

The calibration JSON should follow the COCO/ShareGPT4V-style format expected by `qmllm.calibration.coco_vl.get_multimodal_calib_dataset`. Image paths in JSON are resolved relative to `IMAGE_FOLDER`.

The smoke evaluation JSON is read by `inference.py --infer_pairs` and should contain items like:

```json
[
  {
    "id": "sample-0",
    "images": ["example.jpg"],
    "question": "Describe the image briefly."
  }
]
```

## Qwen2-VL Smoke Test

Run a tiny Qwen2-VL + QIG W4A16 search, confirm that the scale file is generated, then run pseudo quant inference:

```bash
cd /work/model/lkp/QIG
bash scripts/qig/run_smoke_qwen2vl.sh
```

Useful overrides:

```bash
MODEL_PATH=/work/model/lkp/models/qwen2-vl \
CALIB_DATA=/work/model/lkp/datasets/calib/calib.json \
IMAGE_FOLDER=/work/model/lkp/datasets/images \
EVAL_DATA=/work/model/lkp/datasets/eval/eval.json \
CALIB_SIZE=2 \
EVAL_SIZE=2 \
bash scripts/qig/run_smoke_qwen2vl.sh
```

Default outputs:

```text
/work/model/lkp/experiments/qig/qwen2vl_smoke/scale/qig_w4a16.pt
/work/model/lkp/experiments/qig/qwen2vl_smoke/pseudo_quant_preds.json
```

## Qwen2-VL QIG W4A16 Baseline

Run QIG scale search and lmms-eval pseudo quant evaluation:

```bash
cd /work/model/lkp/QIG
bash scripts/qig/run_qwen2vl_w4a16_qig.sh
```

Default output directory:

```text
/work/model/lkp/experiments/qig/qwen2vl_w4a16_qig
```

Common overrides:

```bash
TASKS=mmmu_val \
CALIB_SIZE=128 \
EVAL_LIMIT=100 \
LOG_SAMPLES=1 \
bash scripts/qig/run_qwen2vl_w4a16_qig.sh
```

Set `RUN_SEARCH=0` to reuse an existing scale file. Set `RUN_EVAL=0` to only search scales.

## Qwen2.5-VL Adaptation Plan

`qmllm/models/qwen2_5_vl` exists, but this reproduction keeps Qwen2-VL intact and treats Qwen2.5-VL as an experimental template until adapter compatibility is verified.

TODO:

- Check qwen2_vl adapter / loader.
- Check processor.
- Check `image_grid_thw`.
- Check visual token positioning logic.
- Check whether QIG forward hooks are compatible with Qwen2.5-VL.

Template:

```bash
bash scripts/qig/run_qwen25vl_w4a16_qig.sh
```

The script defaults to `DRY_RUN=1`. Run with `DRY_RUN=0` only after the TODO items are validated.

## New lmms-eval Reference For Qwen2.5-VL / Qwen3-VL

The current QIG checkout originally did not include `3rdparty/lmms-eval`. A newer lmms-eval checkout has been cloned to:

```text
/work/model/lkp/QIG/3rdparty/lmms-eval-new
```

This directory is only a reference for later Qwen2.5-VL and Qwen3-VL downstream evaluation adaptation. It does not replace the current QIG quantization or evaluation flow, and it is not installed into the original `vtiq-py311` environment.

For future integration, use it as a guide for Qwen2.5-VL / Qwen3-VL model adapters, processors, chat templates, and image/video input handling. When QIG support is added, migrate only the necessary logic into `qmllm/models/` or a separate evaluation entrypoint.

## Unified QIG + Latest lmms-eval Environment

The unified environment is:

```text
/work/data/lkp/miniconda3/envs/qig-lmms-py311
```

The original NPU environment remains unchanged:

```text
/work/data/lkp/miniconda3/envs/vtiq-py311
```

`qig-lmms-py311` was cloned from `vtiq-py311` to keep the working QIG NPU stack, including `torch`, `torch_npu`, the CANN-compatible packages, and existing QIG dependencies. The latest lmms-eval reference at `/work/model/lkp/QIG/3rdparty/lmms-eval-new` is installed into the new environment in editable mode.

This environment is intended for:

- Current QIG NPU quantization runs.
- Later Qwen2.5-VL and Qwen3-VL downstream evaluation adaptation.

Use it by setting:

```bash
export PYTHON_BIN=/work/data/lkp/miniconda3/envs/qig-lmms-py311/bin/python
```

pip freeze records are stored outside the repository-tracked files:

```text
/work/model/lkp/experiments/qig/qig_lmms_py311_freeze_before.txt
/work/model/lkp/experiments/qig/qig_lmms_py311_freeze_after.txt
```

## Qwen3-VL Adaptation Plan

The current official QIG code does not include a Qwen3-VL adapter. Do not modify Qwen2-VL behavior while adding Qwen3-VL support.

Qwen3-VL adapter TODO:

- model class
- processor
- chat template
- image/video input format
- visual token index
- activation hook
- calibration forward

Template:

```bash
bash scripts/qig/run_qwen3vl_w4a16_qig.sh
```

The script defaults to `DRY_RUN=1` and is only a command template until adapter support is implemented.

## Result Collection

Collect JSON, TXT, LOG, and CSV outputs into summary files:

```bash
cd /work/model/lkp/QIG
python tools/collect_qig_results.py \
  --root /work/model/lkp/experiments/qig/qwen2vl_w4a16_qig \
  --out /work/model/lkp/experiments/qig/qwen2vl_w4a16_qig/summary.csv
```

This writes both `summary.csv` and `summary.md`.

## Current Status

Completed:

- Official QIG repository cloned under `/work/model/lkp/QIG`.
- `qig_reproduce` branch prepared.
- `upstream` points to official QIG.
- `origin` points to VAQ.
- Cache and output policy centralized in `scripts/qig/env_qig.sh`.
- Qwen2-VL smoke and W4A16 QIG baseline scripts added.
- Qwen2.5-VL and Qwen3-VL template scripts added.
- Result collection tool added.

TODO:

- Install runtime dependencies.
- Place Qwen2-VL model and datasets under `/work/model/lkp/models` and `/work/model/lkp/datasets`.
- Run Qwen2-VL smoke test.
- Run Qwen2-VL W4A16 QIG baseline.
- Validate Qwen2.5-VL adapter compatibility.
- Implement and validate Qwen3-VL adapter.
- Build later MBQ vs QIG vs VAQ experiment tables.
