#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/../qig/env_qig.sh"

PYTHON_BIN=${PYTHON_BIN:-/work/data/lkp/miniconda3/envs/qig-lmms-py311/bin/python}
MODEL_PATH=${MODEL_PATH:-${REPO_DIR}/neijing/models/qwen2.5vl3b-checkpoint-1270}
CALIB_CACHE=${CALIB_CACHE:-${EXP_ROOT}/huaxi_qwen25vl3b_checkpoint1270_full_224_128/calib_inputs_224_full_128.pt}
RUN_ROOT=${RUN_ROOT:-${REPO_DIR}/outputs/lat_awq/qwen25vl3b_w4a16_n128/formal_D_full_l05}
LOG_ROOT=${LOG_ROOT:-${REPO_DIR}/logs/lat_awq/qwen25vl3b_w4a16_n128/formal_D_full_l05}
MODEL_ARGS=${MODEL_ARGS:-pretrained=${MODEL_PATH},torch_dtype=float16,min_pixels=50176,max_pixels=50176}
NPU_ID=${NPU_ID:-0}
N_SAMPLES=${N_SAMPLES:-128}
QIG_SCALE_SEARCH_BATCH_SIZE=${QIG_SCALE_SEARCH_BATCH_SIZE:-1}
SEED=${SEED:-42}

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-${NPU_ID}}
export PYTHONPATH="${REPO_DIR}/3rdparty/lmms-eval-new:${PYTHONPATH:-}"
export QIG_SCALE_SEARCH_BATCH_SIZE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "${RUN_ROOT}/scale" "${RUN_ROOT}/summary" "${RUN_ROOT}/run_metadata" "${LOG_ROOT}"
cd "${REPO_DIR}"

for required in "${PYTHON_BIN}" "${MODEL_PATH}" "${CALIB_CACHE}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[ERROR] Required path does not exist: ${required}" >&2
    exit 1
  fi
done

NAME=D_full_l05_n128
SCALE_PATH=${RUN_ROOT}/scale/${NAME}.pt
DEBUG_PATH=${LOG_ROOT}/${NAME}.jsonl
CONSOLE_LOG=${LOG_ROOT}/${NAME}.log

if [[ -f "${SCALE_PATH}" ]]; then
  echo "[LAT-AWQ formal] Reusing completed scale cache ${SCALE_PATH}"
else
  echo "[LAT-AWQ formal] Starting ${NAME} on NPU ${NPU_ID}"
  echo "[LAT-AWQ formal] n=${N_SAMPLES}, W4A16, group=128, lambda=0.5, seed=${SEED}"
  "${PYTHON_BIN}" -W ignore main_quant.py \
    --model qwen2_5_vl \
    --model_args "${MODEL_ARGS}" \
    --device npu \
    --calib_cache_path "${CALIB_CACHE}" \
    --calib_cache_n_samples "${N_SAMPLES}" \
    --method lat_awq \
    --run_process \
    --w_bit 4 \
    --a_bit 16 \
    --w_group 128 \
    --seed "${SEED}" \
    --token_aware_saliency \
    --token_weighted_loss \
    --saliency_mix_lambda 0.5 \
    --lat_debug \
    --lat_output_dir "${RUN_ROOT}/run_metadata" \
    --lat_log_dir "${LOG_ROOT}" \
    --scale_path "${SCALE_PATH}" \
    2>&1 | tee "${CONSOLE_LOG}"
fi

if [[ -f "${DEBUG_PATH}" ]]; then
  "${PYTHON_BIN}" tools/summarize_lat_awq.py "${DEBUG_PATH}" \
    --output "${RUN_ROOT}/summary/${NAME}.json" >/dev/null
fi

echo "[LAT-AWQ formal] Completed ${NAME}"
