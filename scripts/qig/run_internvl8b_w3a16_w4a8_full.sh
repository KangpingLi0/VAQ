#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/env_qig.sh"

PYTHON_BIN=${PYTHON_BIN:-/work/data/lkp/miniconda3/envs/qig-lmms-py311/bin/python}
CARD=${CARD:-3}
CALIB_DATA=${CALIB_DATA:-${DATA_ROOT}/sharegpt4v/sharegpt4v_coco.json}
IMAGE_FOLDER=${IMAGE_FOLDER:-${DATA_ROOT}}
DATASET_CACHE_ROOT=${DATASET_CACHE_ROOT:-${REPO_DIR}/outputs/lat_awq/dataset_cache_union}
STATUS_ROOT=${STATUS_ROOT:-${REPO_DIR}/logs/internvl8b_full}

# Keep newly requested benchmarks first so a resumed campaign starts producing
# their results immediately.  BLINK is an lmms-eval group of 14 subtasks.
TASKS=(mmstar blink cv_bench vizwiz_vqa_val mmmu_val chartqa ai2d scienceqa textvqa_val ocrbench seedbench)
EXPECTED=(1500 1901 2638 4319 900 2500 3088 4241 5000 1000 17990)
METHODS=(w3a16_rtn w3a16_awq w3a16_mbq w3a16_qig w3a16_lat_awq w3a16_gptq w4a8_rtn w4a8_sq w4a8_mbq w4a8_qig w4a8_lat_awq)

export PYTHONPATH="${REPO_DIR}/3rdparty/lmms-eval-new:${PYTHONPATH:-}"
export HF_DATASETS_CACHE="${DATASET_CACHE_ROOT}"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 TRANSFORMERS_OFFLINE=1
export QIG_SCALE_SEARCH_BATCH_SIZE=${QIG_SCALE_SEARCH_BATCH_SIZE:-1}
export ASCEND_GLOBAL_LOG_LEVEL=${ASCEND_GLOBAL_LOG_LEVEL:-3}
export ASCEND_SLOG_PRINT_TO_STDOUT=${ASCEND_SLOG_PRINT_TO_STDOUT:-0}

mkdir -p "${STATUS_ROOT}"
cd "${REPO_DIR}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${STATUS_ROOT}/status.log"; }

model_values() {
  case "$1" in
    internvl2_8b) echo "${MODEL_ROOT}/InternVL2-8B|${EXP_ROOT}/internvl2_8b_methods" ;;
    internvl2_5_8b) echo "${MODEL_ROOT}/InternVL2_5-8B|${EXP_ROOT}/internvl2_5_8b_methods" ;;
    *) return 2 ;;
  esac
}

method_values() {
  local root=$1 label=$2
  case "${label}" in
    w3a16_rtn) echo "rtn|3|16||" ;;
    w3a16_gptq) echo "gptq|3|16||--calib_data coco --data_path ${CALIB_DATA} --image_folder ${IMAGE_FOLDER} --n_samples 128 --percdamp 0.01" ;;
    w3a16_awq) echo "awq|3|16|${root}/${label}/scale/awq_w3a16.pt|" ;;
    w3a16_mbq) echo "mbq|3|16|${root}/${label}/scale/mbq_w3a16.pt|--reweight" ;;
    w3a16_qig) echo "qig|3|16|${root}/${label}/scale/qig_w3a16.pt|--reweight" ;;
    w3a16_lat_awq) echo "lat_awq|3|16|${root}/${label}/scale/lat_awq_w3a16.pt|--token_aware_saliency --token_weighted_loss --saliency_mix_lambda 0.5" ;;
    w4a8_rtn) echo "rtn|4|8||" ;;
    w4a8_sq) echo "smoothquant|4|8|${root}/${label}/scale/sq_w4a8.pt|" ;;
    w4a8_mbq) echo "mbq|4|8|${root}/${label}/scale/mbq_w4a8.pt|--reweight" ;;
    w4a8_qig) echo "qig|4|8|${root}/${label}/scale/qig_w4a8.pt|--reweight" ;;
    w4a8_lat_awq) echo "lat_awq|4|8|${root}/${label}/scale/lat_awq_w4a8.pt|--token_aware_saliency --token_weighted_loss --saliency_mix_lambda 0.5" ;;
    *) return 2 ;;
  esac
}

validate_result() {
  local out_dir=$1 task=$2 expected=$3
  "${PYTHON_BIN}" - "${out_dir}" "${task}" "${expected}" <<'PY'
import json, sys
from pathlib import Path
root, task, expected = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
paths = list(root.rglob("*_results.json"))
if not paths:
    raise FileNotFoundError(root)
path = max(paths, key=lambda p: p.stat().st_mtime_ns)
data = json.loads(path.read_text(encoding="utf-8"))
results = data.get("results", {})
groups = data.get("groups", {})
if task not in results and task not in groups:
    raise KeyError(f"{task} absent from {path}")
n_samples = data.get("n-samples", {})
if task in n_samples:
    actual = n_samples[task].get("effective")
else:
    subtasks = data.get("group_subtasks", {}).get(task, [])
    if not subtasks:
        raise KeyError(f"{task} has no n-samples or group subtasks in {path}")
    missing = [name for name in subtasks if name not in n_samples]
    if missing:
        raise KeyError(f"{task} missing subtask sample counts: {missing}")
    actual = sum(n_samples[name].get("effective", 0) for name in subtasks)
if actual != expected:
    raise ValueError(f"{task}: expected {expected}, got {actual}")
print(path)
PY
}

run_search() {
  local tag=$1 model_path=$2 root=$3 label=$4 method=$5 w_bit=$6 a_bit=$7 scale=$8 extra=$9
  [[ -z "${scale}" ]] && return 0
  if [[ -s "${scale}" && -f "${root}/${label}/search.done" ]]; then
    log "SKIP_SEARCH model=${tag} method=${label}"
    return 0
  fi
  mkdir -p "$(dirname "${scale}")"
  local cache=${root}/calib/sharegpt4v_coco_n128.pt
  mkdir -p "$(dirname "${cache}")"
  log "START_SEARCH card=${CARD} model=${tag} method=${label}"
  # extra is assembled only by method_values.
  ASCEND_RT_VISIBLE_DEVICES="${CARD}" "${PYTHON_BIN}" -W ignore main_quant.py \
    --device npu --model internvl2 --model_args "pretrained=${model_path}" \
    --calib_data coco --data_path "${CALIB_DATA}" --image_folder "${IMAGE_FOLDER}" \
    --n_samples 128 --micro_batch_size 1 --calib_cache_path "${cache}" \
    --method "${method}" --run_process --w_bit "${w_bit}" --a_bit "${a_bit}" --w_group 128 \
    --scale_path "${scale}" ${extra} >"${root}/${label}/search.log" 2>&1
  [[ -s "${scale}" ]]
  touch "${root}/${label}/search.done"
  log "DONE_SEARCH model=${tag} method=${label} scale=${scale}"
}

run_eval() {
  local tag=$1 model_path=$2 root=$3 label=$4 method=$5 w_bit=$6 a_bit=$7 scale=$8 extra=$9 task=${10} expected=${11}
  local out=${root}/${label}/eval_${task}
  mkdir -p "${out}"
  if validate_result "${out}" "${task}" "${expected}" >"${out}/result_path.txt" 2>/dev/null; then
    touch "${out}/eval.done"
    log "SKIP_EVAL model=${tag} method=${label} task=${task}"
    return 0
  fi
  local scale_arg=""
  [[ -n "${scale}" ]] && scale_arg="--scale_path ${scale}"
  log "START_EVAL card=${CARD} model=${tag} method=${label} task=${task} expected=${expected}"
  # scale_arg and extra are assembled only by this script.
  ASCEND_RT_VISIBLE_DEVICES="${CARD}" "${PYTHON_BIN}" -W ignore main.py \
    --device npu --model internvl2 --model_args "pretrained=${model_path}" \
    --tasks "${task}" --batch_size 1 --output_path "${out}" \
    --method "${method}" --pseudo_quant --w_bit "${w_bit}" --a_bit "${a_bit}" --w_group 128 \
    ${scale_arg} ${extra} >"${out}/eval.log" 2>&1
  validate_result "${out}" "${task}" "${expected}" >"${out}/result_path.txt"
  touch "${out}/eval.done"
  log "DONE_EVAL model=${tag} method=${label} task=${task}"
}

main() {
  [[ "${CARD}" == 3 ]] || { echo "This campaign is restricted to NPU 3" >&2; exit 2; }
  test -d "${MODEL_ROOT}/InternVL2-8B"
  test -d "${MODEL_ROOT}/InternVL2_5-8B"
  exec 9>"${STATUS_ROOT}/campaign.lock"
  flock -n 9 || { echo "InternVL campaign already running" >&2; exit 9; }
  log "CAMPAIGN_START card=3 models=internvl2_8b,internvl2_5_8b methods=${METHODS[*]}"
  local tag mv model_path root label qv method w_bit a_bit scale extra idx
  for tag in internvl2_8b internvl2_5_8b; do
    mv=$(model_values "${tag}"); IFS='|' read -r model_path root <<<"${mv}"
    for label in "${METHODS[@]}"; do
      qv=$(method_values "${root}" "${label}"); IFS='|' read -r method w_bit a_bit scale extra <<<"${qv}"
      mkdir -p "${root}/${label}"
      run_search "${tag}" "${model_path}" "${root}" "${label}" "${method}" "${w_bit}" "${a_bit}" "${scale}" "${extra}"
      for idx in "${!TASKS[@]}"; do
        run_eval "${tag}" "${model_path}" "${root}" "${label}" "${method}" "${w_bit}" "${a_bit}" "${scale}" "${extra}" "${TASKS[$idx]}" "${EXPECTED[$idx]}"
      done
    done
  done
  touch "${STATUS_ROOT}/campaign.done"
  log "CAMPAIGN_DONE"
}

main "$@"
