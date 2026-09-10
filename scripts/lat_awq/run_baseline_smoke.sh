#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/../qig/env_qig.sh"

PYTHON_BIN=${PYTHON_BIN:-/work/data/lkp/miniconda3/envs/qig-lmms-py311/bin/python}
MODEL_PATH=${MODEL_PATH:-${REPO_DIR}/neijing/models/qwen2.5vl3b-checkpoint-1270}
CALIB_CACHE=${CALIB_CACHE:-${REPO_DIR}/outputs/lat_awq/qwen25vl3b_w4a16_n2/calib/calib_indices_0_2_trimmed.pt}
RUN_ROOT=${RUN_ROOT:-${REPO_DIR}/outputs/lat_awq/qwen25vl3b_w4a16_n2/baselines}
MODEL_ARGS=${MODEL_ARGS:-pretrained=${MODEL_PATH},torch_dtype=float16,min_pixels=50176,max_pixels=50176}
NPU_ID=${NPU_ID:-1}

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-${NPU_ID}}
export PYTHONPATH="${REPO_DIR}/3rdparty/lmms-eval-new:${PYTHONPATH:-}"
export QIG_SCALE_SEARCH_BATCH_SIZE=${QIG_SCALE_SEARCH_BATCH_SIZE:-1}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "${RUN_ROOT}/scale" "${RUN_ROOT}/logs"
cd "${REPO_DIR}"

common=(
  -W ignore main_quant.py
  --model qwen2_5_vl
  --model_args "${MODEL_ARGS}"
  --device npu
  --calib_cache_path "${CALIB_CACHE}"
  --run_process
  --w_bit 4
  --a_bit 16
  --w_group 128
  --seed 42
)

if [[ ! -f "${RUN_ROOT}/scale/awq.pt" ]]; then
  "${PYTHON_BIN}" "${common[@]}" --method awq \
    --scale_path "${RUN_ROOT}/scale/awq.pt" \
    2>&1 | tee "${RUN_ROOT}/logs/awq.log"
fi

if [[ ! -f "${RUN_ROOT}/scale/qig.pt" ]]; then
  "${PYTHON_BIN}" "${common[@]}" --method qig --reweight --loss_mode mse \
    --scale_path "${RUN_ROOT}/scale/qig.pt" \
    2>&1 | tee "${RUN_ROOT}/logs/qig.log"
fi

echo "[LAT-AWQ baseline smoke] AWQ and QIG completed under ${RUN_ROOT}"

