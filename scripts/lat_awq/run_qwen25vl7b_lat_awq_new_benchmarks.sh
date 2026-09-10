#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../qig/env_qig.sh"

PYTHON_BIN=${PYTHON_BIN:-/work/data/lkp/miniconda3/envs/qig-lmms-py311/bin/python}
MODEL_PATH=${MODEL_PATH:-${MODEL_ROOT}/qwen2.5-vl}
DATASET_CACHE_ROOT=${DATASET_CACHE_ROOT:-${REPO_DIR}/outputs/lat_awq/dataset_cache_union}
CONFIG=${1:?usage: $0 w3a16|w4a8 CARD}
CARD=${2:?usage: $0 w3a16|w4a8 CARD}

case "${CONFIG}" in
  w3a16) W_BIT=3; A_BIT=16 ;;
  w4a8)  W_BIT=4; A_BIT=8 ;;
  *) echo "unsupported config: ${CONFIG}" >&2; exit 2 ;;
esac

RUN_ROOT=${REPO_DIR}/outputs/lat_awq/qwen25vl7b_sharegpt4v_${CONFIG}_n128/formal_D_full_l05
SCALE_PATH=${RUN_ROOT}/scale/D_full_l05_n128.pt
EVAL_ROOT=${RUN_ROOT}/eval_qig_mbq_full
LOG_ROOT=${REPO_DIR}/logs/lat_awq/qwen25vl7b_sharegpt4v_${CONFIG}_n128/formal_D_full_l05/new_benchmarks
TASKS=(mmstar blink cv_bench)
EXPECTED=(1500 1901 2638)

export PYTHONPATH="${REPO_DIR}/3rdparty/lmms-eval-new:${PYTHONPATH:-}"
export HF_DATASETS_CACHE="${DATASET_CACHE_ROOT}"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 TRANSFORMERS_OFFLINE=1
export ASCEND_GLOBAL_LOG_LEVEL=${ASCEND_GLOBAL_LOG_LEVEL:-3}
export ASCEND_SLOG_PRINT_TO_STDOUT=${ASCEND_SLOG_PRINT_TO_STDOUT:-0}

mkdir -p "${LOG_ROOT}" "${EVAL_ROOT}"
cd "${REPO_DIR}"

validate_result() {
  local out=$1 task=$2 expected=$3
  "${PYTHON_BIN}" - "${out}" "${task}" "${expected}" <<'PY'
import json, sys
from pathlib import Path
root, task, expected = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
paths = list(root.rglob("*_results.json"))
if not paths:
    raise FileNotFoundError(root)
path = max(paths, key=lambda p: p.stat().st_mtime_ns)
data = json.loads(path.read_text(encoding="utf-8"))
if task not in data.get("results", {}) and task not in data.get("groups", {}):
    raise KeyError(task)
samples = data.get("n-samples", {})
actual = samples.get(task, {}).get("effective")
if actual is None:
    subtasks = data.get("group_subtasks", {}).get(task, [])
    actual = sum(samples.get(name, {}).get("effective", 0) for name in subtasks)
if actual != expected:
    raise ValueError(f"{task}: expected {expected}, got {actual}")
print(path)
PY
}

test -s "${SCALE_PATH}"
for idx in "${!TASKS[@]}"; do
  task=${TASKS[$idx]}
  expected=${EXPECTED[$idx]}
  out=${EVAL_ROOT}/${task}
  mkdir -p "${out}/run_metadata" "${out}/logs"
  exec {lock_fd}>"${out}/eval.lock"
  flock "${lock_fd}"
  if validate_result "${out}" "${task}" "${expected}" >"${out}/result_path.txt" 2>/dev/null; then
    touch "${out}/eval.done"
    echo "[$(date '+%F %T')] SKIP ${CONFIG} ${task}" | tee -a "${LOG_ROOT}/status.log"
    exec {lock_fd}>&-
    continue
  fi
  echo "[$(date '+%F %T')] START ${CONFIG} ${task} card=${CARD}" | tee -a "${LOG_ROOT}/status.log"
  ASCEND_RT_VISIBLE_DEVICES="${CARD}" "${PYTHON_BIN}" -W ignore main.py \
    --device npu --model qwen2_5_vl --model_args "pretrained=${MODEL_PATH}" \
    --tasks "${task}" --batch_size 1 --output_path "${out}" \
    --method lat_awq --pseudo_quant --w_bit "${W_BIT}" --a_bit "${A_BIT}" --w_group 128 \
    --scale_path "${SCALE_PATH}" --token_aware_saliency --token_weighted_loss \
    --saliency_mix_lambda 0.5 --lat_output_dir "${out}/run_metadata" --lat_log_dir "${out}/logs" \
    >"${LOG_ROOT}/${task}.log" 2>&1
  validate_result "${out}" "${task}" "${expected}" >"${out}/result_path.txt"
  touch "${out}/eval.done"
  echo "[$(date '+%F %T')] DONE ${CONFIG} ${task}" | tee -a "${LOG_ROOT}/status.log"
  exec {lock_fd}>&-
done
echo "[$(date '+%F %T')] DONE ${CONFIG} all_new_benchmarks" | tee -a "${LOG_ROOT}/status.log"
