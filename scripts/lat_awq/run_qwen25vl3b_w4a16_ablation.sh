#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/../qig/env_qig.sh"

PYTHON_BIN=${PYTHON_BIN:-/work/data/lkp/miniconda3/envs/qig-lmms-py311/bin/python}
MODEL_PATH=${MODEL_PATH:-${REPO_DIR}/neijing/models/qwen2.5vl3b-checkpoint-1270}
SOURCE_CACHE=${SOURCE_CACHE:-${EXP_ROOT}/huaxi_qwen25vl3b_checkpoint1270_full_224_128/calib_inputs_224_full_128.pt}
RUN_ROOT=${RUN_ROOT:-${REPO_DIR}/outputs/lat_awq/qwen25vl3b_w4a16_n2}
LOG_ROOT=${LOG_ROOT:-${REPO_DIR}/logs/lat_awq/qwen25vl3b_w4a16_n2}
CALIB_CACHE=${CALIB_CACHE:-${RUN_ROOT}/calib/calib_indices_0_2_trimmed.pt}
MODEL_ARGS=${MODEL_ARGS:-pretrained=${MODEL_PATH},torch_dtype=float16,min_pixels=50176,max_pixels=50176}
NPU_ID=${NPU_ID:-0}
QIG_SCALE_SEARCH_BATCH_SIZE=${QIG_SCALE_SEARCH_BATCH_SIZE:-1}
SEED=${SEED:-42}

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-${NPU_ID}}
export PYTHONPATH="${REPO_DIR}/3rdparty/lmms-eval-new:${PYTHONPATH:-}"
export QIG_SCALE_SEARCH_BATCH_SIZE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "${RUN_ROOT}/scale" "${RUN_ROOT}/summary" "${LOG_ROOT}"
cd "${REPO_DIR}"

for required in "${PYTHON_BIN}" "${MODEL_PATH}" "${SOURCE_CACHE}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[ERROR] Required path does not exist: ${required}" >&2
    exit 1
  fi
done

if [[ ! -f "${CALIB_CACHE}" ]]; then
  "${PYTHON_BIN}" tools/make_lat_awq_calib_subset.py \
    "${SOURCE_CACHE}" "${CALIB_CACHE}" --indices 0,2
fi

"${PYTHON_BIN}" -m pytest -q tests/test_lat_awq.py | tee "${LOG_ROOT}/unit_tests.log"

run_one() {
  local name=$1
  local lambda=$2
  local saliency=$3
  local weighted=$4
  local scale_path="${RUN_ROOT}/scale/${name}.pt"
  local console_log="${LOG_ROOT}/${name}.log"
  local args=(
    -W ignore main_quant.py
    --model qwen2_5_vl
    --model_args "${MODEL_ARGS}"
    --device npu
    --calib_cache_path "${CALIB_CACHE}"
    --method lat_awq
    --run_process
    --w_bit 4
    --a_bit 16
    --w_group 128
    --seed "${SEED}"
    --saliency_mix_lambda "${lambda}"
    --lat_debug
    --lat_output_dir "${RUN_ROOT}/run_metadata"
    --lat_log_dir "${LOG_ROOT}"
    --scale_path "${scale_path}"
  )
  if [[ "${saliency}" == 1 ]]; then
    args+=(--token_aware_saliency)
  fi
  if [[ "${weighted}" == 1 ]]; then
    args+=(--token_weighted_loss)
  fi

  if [[ ! -f "${scale_path}" ]]; then
    echo "[LAT-AWQ] Starting ${name}: saliency=${saliency}, loss=${weighted}, lambda=${lambda}"
    "${PYTHON_BIN}" "${args[@]}" 2>&1 | tee "${console_log}"
  else
    echo "[LAT-AWQ] Reusing completed scale cache ${scale_path}"
  fi
  "${PYTHON_BIN}" tools/summarize_lat_awq.py "${LOG_ROOT}/${name}.jsonl" \
    --output "${RUN_ROOT}/summary/${name}.json" >/dev/null
}

# Smoke gates use the exact A and D configurations later reported in the 2x2.
run_one A_awq_like 1.0 0 0
run_one D_full_l1 1.0 1 1

# Remaining core 2x2 cells.
run_one B_loss_only 1.0 0 1
# Kept as an independent post-implementation validation cache for the strict
# no-broadcast reconstruction helper.
run_one B_loss_only_strict_validation 1.0 0 1
run_one C_saliency_only_l1 1.0 1 0
# Explicit degeneration check requested by the method definition.
run_one C_saliency_only_l0 0.0 1 0

# Full-method lambda trend; lambda=1 is D_full_l1 above.
run_one D_full_l0 0.0 1 1
run_one D_full_l025 0.25 1 1
run_one D_full_l05 0.5 1 1
run_one D_full_l075 0.75 1 1

echo "[LAT-AWQ] All requested n=2 ablations completed under ${RUN_ROOT}"
