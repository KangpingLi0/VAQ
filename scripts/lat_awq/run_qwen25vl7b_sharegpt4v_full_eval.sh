#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/../qig/env_qig.sh"

PYTHON_BIN=${PYTHON_BIN:-/work/data/lkp/miniconda3/envs/qig-lmms-py311/bin/python}
MODEL_NAME=${MODEL_NAME:-qwen2_5_vl}
case "${MODEL_NAME}" in
  qwen2_5_vl)
    DEFAULT_MODEL_PATH=${MODEL_ROOT}/qwen2.5-vl
    DEFAULT_MODEL_TAG=qwen25vl7b
    DEFAULT_BASELINE_ROOT=${EXP_ROOT}/qwen25vl_7b_npu_methods
    ;;
  qwen2_vl)
    DEFAULT_MODEL_PATH=${MODEL_ROOT}/qwen2-vl
    DEFAULT_MODEL_TAG=qwen2vl7b
    DEFAULT_BASELINE_ROOT=${EXP_ROOT}/table2_qwen2vl_7b_npu_newlmms
    ;;
  *)
    echo "[ERROR] Unsupported MODEL_NAME=${MODEL_NAME}; expected qwen2_5_vl or qwen2_vl" >&2
    exit 2
    ;;
esac
MODEL_PATH=${MODEL_PATH:-${DEFAULT_MODEL_PATH}}
MODEL_TAG=${MODEL_TAG:-${DEFAULT_MODEL_TAG}}
CALIB_DATA=${CALIB_DATA:-${DATA_ROOT}/sharegpt4v/sharegpt4v_coco.json}
IMAGE_FOLDER=${IMAGE_FOLDER:-${DATA_ROOT}}
N_SAMPLES=${N_SAMPLES:-128}
W_BIT=${W_BIT:-4}
A_BIT=${A_BIT:-16}
QUANT_CARD=${QUANT_CARD:-0}
EVAL_CARDS=${EVAL_CARDS:-0,1,2,3,4}
QIG_SCALE_SEARCH_BATCH_SIZE=${QIG_SCALE_SEARCH_BATCH_SIZE:-1}
SEARCH_ONLY=${SEARCH_ONLY:-0}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
DATASET_CACHE_ROOT=${DATASET_CACHE_ROOT:-${HF_DATASETS_CACHE}}

QUANT_TAG=w${W_BIT}a${A_BIT}
RUN_BASE=${RUN_BASE:-${REPO_DIR}/outputs/lat_awq/${MODEL_TAG}_sharegpt4v_${QUANT_TAG}_n128}
RUN_ROOT=${RUN_ROOT:-${RUN_BASE}/formal_D_full_l05}
LOG_ROOT=${LOG_ROOT:-${REPO_DIR}/logs/lat_awq/${MODEL_TAG}_sharegpt4v_${QUANT_TAG}_n128/formal_D_full_l05}
CALIB_CACHE=${CALIB_CACHE:-${RUN_BASE}/calib/sharegpt4v_coco_n128.pt}
SCALE_PATH=${SCALE_PATH:-${RUN_ROOT}/scale/D_full_l05_n128.pt}
SMOKE_SCALE_PATH=${SMOKE_SCALE_PATH:-${RUN_ROOT}/smoke/scale/D_full_l05_n2.pt}
EVAL_ROOT=${EVAL_ROOT:-${RUN_ROOT}/eval_qig_mbq_full}
BASELINE_ROOT=${BASELINE_ROOT:-${DEFAULT_BASELINE_ROOT}}
STATUS_FILE=${STATUS_FILE:-${RUN_ROOT}/pipeline_status.log}

TASKS=(vizwiz_vqa_val mmmu_val chartqa ai2d scienceqa textvqa_val ocrbench seedbench)
EXPECTED_COUNTS=(4319 900 2500 3088 4241 5000 1000 17990)
ERROR_PATTERN='Traceback|ERROR|RetryError|RuntimeError|NonMatchingSplitsSizesError|FileNotFoundError|AssertionError|out of memory|OOM'

export PYTHONPATH="${REPO_DIR}/3rdparty/lmms-eval-new:${PYTHONPATH:-}"
export QIG_SCALE_SEARCH_BATCH_SIZE
export HF_DATASETS_CACHE="${DATASET_CACHE_ROOT}"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export TRANSFORMERS_OFFLINE=1
export ASCEND_GLOBAL_LOG_LEVEL=${ASCEND_GLOBAL_LOG_LEVEL:-3}
export ASCEND_SLOG_PRINT_TO_STDOUT=${ASCEND_SLOG_PRINT_TO_STDOUT:-0}

mkdir -p \
  "${RUN_ROOT}/scale" \
  "${RUN_ROOT}/run_metadata" \
  "$(dirname "${CALIB_CACHE}")" \
  "$(dirname "${SMOKE_SCALE_PATH}")" \
  "${RUN_ROOT}/smoke/run_metadata" \
  "${EVAL_ROOT}" \
  "${LOG_ROOT}/smoke" \
  "${LOG_ROOT}/eval"

cd "${REPO_DIR}"

exec 9>"${RUN_ROOT}/pipeline.lock"
if ! flock -n 9; then
  echo "[ERROR] Another LAT-AWQ 7B full-evaluation pipeline already holds ${RUN_ROOT}/pipeline.lock" >&2
  exit 9
fi

log_status() {
  echo "[$(date '+%F %T')] $*" | tee -a "${STATUS_FILE}"
}

has_log_error() {
  local path=$1
  [[ -f "${path}" ]] && grep -Eiq "${ERROR_PATTERN}" "${path}"
}

validate_scale() {
  local path=$1
  local expected_n=$2
  "${PYTHON_BIN}" - "${path}" "${expected_n}" "${W_BIT}" "${A_BIT}" <<'PY'
import sys
from pathlib import Path

import torch

path = Path(sys.argv[1])
expected_n = int(sys.argv[2])
expected_w_bit = int(sys.argv[3])
expected_a_bit = int(sys.argv[4])
obj = torch.load(str(path), map_location="cpu")
if not isinstance(obj, dict) or not isinstance(obj.get("scale"), list):
    raise TypeError(f"Malformed LAT-AWQ scale cache: {path}")
meta = obj.get("metadata", {})
expected = {
    "method": "lat_awq",
    "w_bit": expected_w_bit,
    "a_bit": expected_a_bit,
    "group_size": 128,
    "token_aware_saliency": True,
    "token_weighted_loss": True,
    "saliency_mix_lambda": 0.5,
}
bad = {key: (meta.get(key), value) for key, value in expected.items() if meta.get(key) != value}
if bad:
    raise ValueError(f"Scale metadata mismatch for {path}: {bad}")
if meta.get("n_samples") not in (None, expected_n):
    raise ValueError(f"Expected n_samples={expected_n}, got {meta.get('n_samples')}")
scales = obj["scale"]
if len(scales) != 84:
    raise ValueError(f"A supported 28-layer VL-7B model must have 28x3=84 scale groups, got {len(scales)}")
for idx, entry in enumerate(scales):
    if not isinstance(entry, (list, tuple)) or len(entry) != 3 or not torch.is_tensor(entry[2]):
        raise TypeError(f"Malformed scale entry {idx}: {type(entry)}")
    if not torch.isfinite(entry[2]).all():
        raise ValueError(f"Non-finite values in scale entry {idx}")
print(f"validated_scale={path}")
print(f"groups={len(scales)} n_samples={meta.get('n_samples')}")
PY
}

preflight() {
  for required in "${PYTHON_BIN}" "${MODEL_PATH}" "${CALIB_DATA}" "${IMAGE_FOLDER}"; do
    if [[ ! -e "${required}" ]]; then
      echo "[ERROR] Required path does not exist: ${required}" >&2
      return 1
    fi
  done

  "${PYTHON_BIN}" - "${MODEL_PATH}" "${MODEL_NAME}" <<'PY'
import json
import sys
from pathlib import Path

model_path, model_name = Path(sys.argv[1]), sys.argv[2]
config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
expected_type = {"qwen2_5_vl": "qwen2_5_vl", "qwen2_vl": "qwen2_vl"}[model_name]
if config.get("model_type") != expected_type:
    raise ValueError(f"{model_path}: expected model_type={expected_type}, got {config.get('model_type')}")
text = config.get("text_config") or config
if text.get("num_hidden_layers") != 28 or text.get("hidden_size") != 3584:
    raise ValueError(f"{model_path}: expected 28 layers and hidden_size 3584, got {text}")
print(f"model={model_path} model_type={expected_type} layers=28 hidden=3584")
PY

  "${PYTHON_BIN}" - "${CALIB_DATA}" "${IMAGE_FOLDER}" "${N_SAMPLES}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

data_path, image_root, n_samples = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
rows = json.loads(data_path.read_text(encoding="utf-8"))
if len(rows) < n_samples:
    raise ValueError(f"Calibration set has {len(rows)} rows, need {n_samples}")
rng = np.random.default_rng(seed=42)
rng.shuffle(rows)
selected = rows[:n_samples]
images = []
for row in selected:
    value = row.get("image", row.get("images", []))
    images.extend(value if isinstance(value, list) else [value])
missing = [str(image_root / value) for value in images if not (image_root / value).is_file()]
if missing:
    raise FileNotFoundError(f"Missing calibration images: {missing[:10]}")
print(f"calibration_rows={len(selected)} images={len(images)} missing=0")
PY

  "${PYTHON_BIN}" - <<'PY'
from datasets import load_dataset

checks = [
    ("VizWiz", "/work/model/lkp/PTQ/data/hf_cache/VizWiz-VQA", None, "val", 4319),
    ("MMMU", "/work/model/lkp/PTQ/data/hf_cache/MMMU", None, "validation", 900),
    ("ChartQA", "lmms-lab/ChartQA", None, "test", 2500),
    ("AI2D", "lmms-lab/ai2d", None, "test", 3088),
    ("ScienceQA", "/work/model/lkp/PTQ/data/hf_cache/ScienceQA", "ScienceQA-FULL", "test", 4241),
    ("TextVQA", "/work/model/lkp/PTQ/data/hf_cache/textvqa", None, "validation", 5000),
    ("OCRBench", "/work/model/lkp/PTQ/data/hf_cache/OCRBench", None, "test", 1000),
    ("SEED-Bench", "/work/model/lkp/PTQ/data/hf_cache/SEED-Bench", None, "test", 17990),
]
for label, path, subset, split, expected in checks:
    dataset = load_dataset(path, subset) if subset else load_dataset(path)
    actual = len(dataset[split])
    if actual != expected:
        raise ValueError(f"{label}/{split}: expected {expected}, got {actual}")
    print(f"{label}/{split}={actual}")
PY

  local cards_csv=",${EVAL_CARDS},"
  if [[ "${cards_csv}" == *",6,"* || "${QUANT_CARD}" == "6" ]]; then
    echo "[ERROR] NPU 6 is reserved by another running job; choose other cards." >&2
    return 1
  fi
}

run_search() {
  local name=$1
  local n_samples=$2
  local scale_path=$3
  local cache_path=$4
  local metadata_dir=$5
  local debug_dir=$6
  local console_log=$7

  if [[ -f "${scale_path}" ]]; then
    validate_scale "${scale_path}" "${n_samples}"
    log_status "SKIP_SEARCH ${name} valid scale already exists: ${scale_path}"
    return 0
  fi

  local calib_args=(
    --calib_data coco
    --data_path "${CALIB_DATA}"
    --image_folder "${IMAGE_FOLDER}"
    --n_samples "${n_samples}"
    --micro_batch_size 1
  )
  if [[ -n "${cache_path}" ]]; then
    calib_args+=(--calib_cache_path "${cache_path}" --calib_cache_n_samples "${n_samples}")
  fi

  mkdir -p "$(dirname "${scale_path}")" "${metadata_dir}" "${debug_dir}"
  log_status "START_SEARCH ${name} card=${QUANT_CARD} n=${n_samples}"
  set +e
  ASCEND_RT_VISIBLE_DEVICES="${QUANT_CARD}" "${PYTHON_BIN}" -W ignore main_quant.py \
    --device npu \
    --model "${MODEL_NAME}" \
    --model_args "pretrained=${MODEL_PATH}" \
    "${calib_args[@]}" \
    --method lat_awq \
    --run_process \
    --w_bit "${W_BIT}" \
    --a_bit "${A_BIT}" \
    --w_group 128 \
    --seed 42 \
    --token_aware_saliency \
    --token_weighted_loss \
    --saliency_mix_lambda 0.5 \
    --lat_debug \
    --lat_output_dir "${metadata_dir}" \
    --lat_log_dir "${debug_dir}" \
    --scale_path "${scale_path}" \
    2>&1 | tee -a "${console_log}"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ ${rc} -ne 0 ]] || has_log_error "${console_log}"; then
    log_status "FAILED_SEARCH ${name} exit=${rc} log=${console_log}"
    return 1
  fi
  validate_scale "${scale_path}" "${n_samples}"
  log_status "DONE_SEARCH ${name} scale=${scale_path}"
}

validate_eval_result() {
  local out_dir=$1
  local task=$2
  local expected_count=$3
  "${PYTHON_BIN}" - "${out_dir}" "${task}" "${expected_count}" <<'PY'
import json
import sys
from pathlib import Path

root, task, expected = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
paths = list(root.rglob("*_results.json"))
if not paths:
    raise FileNotFoundError(f"No aggregated results JSON below {root}")
path = max(paths, key=lambda item: item.stat().st_mtime_ns)
data = json.loads(path.read_text(encoding="utf-8"))
if task not in data.get("results", {}):
    raise KeyError(f"Task {task!r} absent from {path}")
effective = data.get("n-samples", {}).get(task, {}).get("effective")
if effective != expected:
    raise ValueError(f"{task}: expected {expected} effective samples, got {effective}")
print(path)
PY
}

run_eval_task() {
  local task=$1
  local expected_count=$2
  local card=$3
  local out_dir=${EVAL_ROOT}/${task}
  local log_file=${LOG_ROOT}/eval/${task}.log
  local done_file=${out_dir}/eval.done

  mkdir -p "${out_dir}/run_metadata" "${out_dir}/logs"
  if [[ -f "${done_file}" ]] && validate_eval_result "${out_dir}" "${task}" "${expected_count}" >/dev/null; then
    log_status "SKIP_EVAL ${task} valid full result already exists"
    return 0
  fi

  log_status "START_EVAL ${task} card=${card} expected=${expected_count}"
  set +e
  ASCEND_RT_VISIBLE_DEVICES="${card}" "${PYTHON_BIN}" -W ignore main.py \
    --device npu \
    --model "${MODEL_NAME}" \
    --model_args "pretrained=${MODEL_PATH}" \
    --tasks "${task}" \
    --batch_size 1 \
    --output_path "${out_dir}" \
    --method lat_awq \
    --pseudo_quant \
    --w_bit "${W_BIT}" \
    --a_bit "${A_BIT}" \
    --w_group 128 \
    --scale_path "${SCALE_PATH}" \
    --token_aware_saliency \
    --token_weighted_loss \
    --saliency_mix_lambda 0.5 \
    --lat_output_dir "${out_dir}/run_metadata" \
    --lat_log_dir "${out_dir}/logs" \
    2>&1 | tee -a "${log_file}"
  local rc=${PIPESTATUS[0]}
  set -e

  if [[ ${rc} -ne 0 ]] || has_log_error "${log_file}"; then
    log_status "FAILED_EVAL ${task} exit=${rc} log=${log_file}"
    return 1
  fi
  validate_eval_result "${out_dir}" "${task}" "${expected_count}" >"${out_dir}/result_path.txt"
  touch "${done_file}"
  log_status "DONE_EVAL ${task} card=${card}"
}

summarize() {
  "${PYTHON_BIN}" tools/summarize_lat_awq_lmms.py \
    --lat-root "${EVAL_ROOT}" \
    --lat-method "lat_awq_${QUANT_TAG}_n${N_SAMPLES}_l05" \
    --baseline-root "${BASELINE_ROOT}" \
    --output-json "${EVAL_ROOT}/metrics_summary.json" \
    --output-csv "${EVAL_ROOT}/metrics_summary.csv" \
    --output-md "${EVAL_ROOT}/metrics_summary.md"
}

main() {
  log_status "PIPELINE_START pid=$$ model=${MODEL_PATH}"
  log_status "PROTOCOL calibration=ShareGPT4V-COCO n=${N_SAMPLES} quant=${QUANT_TAG} group=128 tasks=${TASKS[*]}"
  preflight | tee -a "${RUN_ROOT}/preflight.log"
  log_status "PREFLIGHT_DONE"

  if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
    log_status "PREFLIGHT_ONLY_DONE"
    return 0
  fi

  run_search \
    D_full_l05_n2 \
    2 \
    "${SMOKE_SCALE_PATH}" \
    "" \
    "${RUN_ROOT}/smoke/run_metadata" \
    "${LOG_ROOT}/smoke" \
    "${LOG_ROOT}/smoke/D_full_l05_n2.log"

  run_search \
    D_full_l05_n128 \
    "${N_SAMPLES}" \
    "${SCALE_PATH}" \
    "${CALIB_CACHE}" \
    "${RUN_ROOT}/run_metadata" \
    "${LOG_ROOT}" \
    "${LOG_ROOT}/D_full_l05_n128.log"

  if [[ -f "${LOG_ROOT}/D_full_l05_n128.jsonl" ]]; then
    "${PYTHON_BIN}" tools/summarize_lat_awq.py "${LOG_ROOT}/D_full_l05_n128.jsonl" \
      --output "${RUN_ROOT}/scale_search_summary.json"
  fi

  if [[ "${SEARCH_ONLY}" == "1" ]]; then
    touch "${RUN_ROOT}/search.done"
    log_status "SEARCH_ONLY_DONE scale=${SCALE_PATH}"
    return 0
  fi

  IFS=',' read -r -a cards <<<"${EVAL_CARDS}"
  if [[ ${#cards[@]} -lt ${#TASKS[@]} ]]; then
    echo "[ERROR] Need at least ${#TASKS[@]} evaluation cards, got ${#cards[@]}" >&2
    exit 2
  fi

  local pids=()
  local labels=()
  local idx
  for idx in "${!TASKS[@]}"; do
    run_eval_task "${TASKS[$idx]}" "${EXPECTED_COUNTS[$idx]}" "${cards[$idx]}" &
    pids+=("$!")
    labels+=("${TASKS[$idx]}")
    sleep 10
  done

  local failed=0
  for idx in "${!pids[@]}"; do
    if wait "${pids[$idx]}"; then
      log_status "JOIN_OK ${labels[$idx]}"
    else
      log_status "JOIN_FAILED ${labels[$idx]}"
      failed=1
    fi
  done
  if [[ ${failed} -ne 0 ]]; then
    log_status "PIPELINE_FAILED one_or_more_evaluations_failed"
    return 1
  fi

  summarize | tee -a "${EVAL_ROOT}/summarize.log"
  touch "${RUN_ROOT}/pipeline.done"
  log_status "PIPELINE_DONE summary=${EVAL_ROOT}/metrics_summary.md"
}

main "$@"
