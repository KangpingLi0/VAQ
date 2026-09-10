#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/env_qig.sh"

PYTHON_BIN=${PYTHON_BIN:-/work/data/lkp/miniconda3/envs/qig-lmms-py311/bin/python}
CALIB_DATA=${CALIB_DATA:-${DATA_ROOT}/sharegpt4v/sharegpt4v_coco.json}
IMAGE_FOLDER=${IMAGE_FOLDER:-${DATA_ROOT}}
DATASET_CACHE_ROOT=${DATASET_CACHE_ROOT:-${REPO_DIR}/outputs/lat_awq/dataset_cache_union}
CAMPAIGN_ROOT=${CAMPAIGN_ROOT:-${REPO_DIR}/logs/baselines/vl7b_missing_union_eval}
CARDS=${CARDS:-0,1,2,4,5}

TASKS=(vizwiz_vqa_val mmmu_val chartqa ai2d scienceqa textvqa_val ocrbench seedbench)
EXPECTED=(4319 900 2500 3088 4241 5000 1000 17990)

export PYTHONPATH="${REPO_DIR}/3rdparty/lmms-eval-new:${PYTHONPATH:-}"
export HF_DATASETS_CACHE="${DATASET_CACHE_ROOT}"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export TRANSFORMERS_OFFLINE=1
export ASCEND_GLOBAL_LOG_LEVEL=${ASCEND_GLOBAL_LOG_LEVEL:-3}
export ASCEND_SLOG_PRINT_TO_STDOUT=${ASCEND_SLOG_PRINT_TO_STDOUT:-0}

mkdir -p "${CAMPAIGN_ROOT}"
cd "${REPO_DIR}"

log() {
  local lane=$1
  shift
  echo "[$(date '+%F %T')] lane=${lane} $*" | tee -a "${CAMPAIGN_ROOT}/campaign_status.log" "${CAMPAIGN_ROOT}/${lane}.status"
}

model_values() {
  case "$1" in
    qwen25)
      echo "qwen2_5_vl|${MODEL_ROOT}/qwen2.5-vl|${EXP_ROOT}/qwen25vl_7b_npu_methods"
      ;;
    qwen2)
      echo "qwen2_vl|${MODEL_ROOT}/qwen2-vl|${EXP_ROOT}/table2_qwen2vl_7b_npu_newlmms"
      ;;
    *) return 2 ;;
  esac
}

method_values() {
  local root=$1
  local label=$2
  case "${label}" in
    w3a16_rtn)  echo "rtn|3|16|" ;;
    w3a16_gptq) echo "gptq|3|16|--calib_data coco --data_path ${CALIB_DATA} --image_folder ${IMAGE_FOLDER} --n_samples 128 --percdamp 0.01" ;;
    w3a16_awq)  echo "awq|3|16|--scale_path ${root}/w3a16_awq/scale/awq_w3a16.pt" ;;
    w3a16_mbq)  echo "mbq|3|16|--scale_path ${root}/w3a16_mbq/scale/mbq_w3a16.pt" ;;
    w3a16_qig)  echo "qig|3|16|--scale_path ${root}/w3a16_qig/scale/qig_w3a16.pt" ;;
    w4a8_rtn)   echo "rtn|4|8|" ;;
    w4a8_sq)    echo "smoothquant|4|8|--scale_path ${root}/w4a8_sq/scale/sq_w4a8.pt" ;;
    w4a8_mbq)   echo "mbq|4|8|--scale_path ${root}/w4a8_mbq/scale/mbq_w4a8.pt" ;;
    w4a8_qig)   echo "qig|4|8|--scale_path ${root}/w4a8_qig/scale/qig_w4a8.pt" ;;
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
if task not in data.get("results", {}):
    raise KeyError(f"{task} absent from {path}")
actual = data.get("n-samples", {}).get(task, {}).get("effective")
if actual != expected:
    raise ValueError(f"{task}: expected {expected}, got {actual}")
print(path)
PY
}

run_one() {
  local lane=$1 card=$2 model_tag=$3 label=$4 task=$5 expected=$6
  local mv model_name model_path root qv method w_bit a_bit extra out_dir log_file rc
  mv=$(model_values "${model_tag}")
  IFS='|' read -r model_name model_path root <<<"${mv}"
  qv=$(method_values "${root}" "${label}")
  IFS='|' read -r method w_bit a_bit extra <<<"${qv}"
  out_dir=${root}/${label}/eval_${task}
  log_file=${out_dir}/baseline_completion.log
  mkdir -p "${out_dir}"

  if validate_result "${out_dir}" "${task}" "${expected}" >"${out_dir}/result_path.txt" 2>/dev/null; then
    touch "${out_dir}/eval.done"
    log "${lane}" "SKIP_VALID model=${model_tag} method=${label} task=${task}"
    return 0
  fi

  if [[ "${extra}" == *"--scale_path "* ]]; then
    local scale_path=${extra#*--scale_path }
    scale_path=${scale_path%% *}
    if [[ ! -s "${scale_path}" ]]; then
      log "${lane}" "FAILED_MISSING_SCALE model=${model_tag} method=${label} scale=${scale_path}"
      return 1
    fi
  fi

  rm -f "${out_dir}/eval.done" "${out_dir}/eval.failed"
  log "${lane}" "START card=${card} model=${model_tag} method=${label} task=${task} expected=${expected}"
  set +e
  # extra is assembled exclusively by method_values above.
  ASCEND_RT_VISIBLE_DEVICES="${card}" "${PYTHON_BIN}" -W ignore main.py \
    --device npu --model "${model_name}" --model_args "pretrained=${model_path}" \
    --tasks "${task}" --batch_size 1 --output_path "${out_dir}" \
    --method "${method}" --pseudo_quant --w_bit "${w_bit}" --a_bit "${a_bit}" --w_group 128 \
    ${extra} >"${log_file}" 2>&1
  rc=$?
  set -e
  if [[ ${rc} -eq 0 ]] && validate_result "${out_dir}" "${task}" "${expected}" >"${out_dir}/result_path.txt" 2>>"${log_file}"; then
    touch "${out_dir}/eval.done"
    log "${lane}" "DONE card=${card} model=${model_tag} method=${label} task=${task}"
  else
    touch "${out_dir}/eval.failed"
    log "${lane}" "FAILED card=${card} model=${model_tag} method=${label} task=${task} rc=${rc} log=${log_file}"
    return 1
  fi
}

run_lane() {
  local lane=$1 card=$2
  local labels=()
  case "${lane}" in
    rtn)  labels=(w3a16_rtn w4a8_rtn) ;;
    awq_sq) labels=(w3a16_awq w4a8_sq) ;;
    mbq)  labels=(w3a16_mbq w4a8_mbq) ;;
    qig)  labels=(w3a16_qig w4a8_qig) ;;
    gptq) labels=(w3a16_gptq) ;;
    *) echo "Unknown lane ${lane}" >&2; return 2 ;;
  esac
  log "${lane}" "LANE_START card=${card} labels=${labels[*]}"
  local failed=0 model label idx
  for model in qwen25 qwen2; do
    for label in "${labels[@]}"; do
      for idx in "${!TASKS[@]}"; do
        run_one "${lane}" "${card}" "${model}" "${label}" "${TASKS[$idx]}" "${EXPECTED[$idx]}" || failed=1
      done
    done
  done
  if [[ ${failed} -eq 0 ]]; then
    touch "${CAMPAIGN_ROOT}/${lane}.done"
    log "${lane}" "LANE_DONE"
  else
    touch "${CAMPAIGN_ROOT}/${lane}.failed"
    log "${lane}" "LANE_FINISHED_WITH_FAILURES"
    return 1
  fi
}

preflight() {
  test -x "${PYTHON_BIN}"
  test -d "${MODEL_ROOT}/qwen2.5-vl"
  test -d "${MODEL_ROOT}/qwen2-vl"
  test -f "${CALIB_DATA}"
  test -d "${DATASET_CACHE_ROOT}"
  IFS=',' read -r -a cards_array <<<"${CARDS}"
  [[ ${#cards_array[@]} -eq 5 ]]
  for card in "${cards_array[@]}"; do
    [[ "${card}" != 3 && "${card}" != 6 && "${card}" != 7 ]]
  done
}

main() {
  preflight
  if [[ -n "${LANE:-}" ]]; then
    run_lane "${LANE}" "${CARD:?CARD is required with LANE}"
    return
  fi

  exec 9>"${CAMPAIGN_ROOT}/campaign.lock"
  flock -n 9 || { echo "Campaign already running" >&2; exit 9; }
  : >"${CAMPAIGN_ROOT}/campaign_status.log"
  IFS=',' read -r -a cards_array <<<"${CARDS}"
  local lanes=(rtn awq_sq mbq qig gptq) pids=() idx failed=0
  log master "CAMPAIGN_START cards=${CARDS} protocol=ShareGPT4V_COCO_n128 tasks=${TASKS[*]}"
  for idx in "${!lanes[@]}"; do
    LANE="${lanes[$idx]}" CARD="${cards_array[$idx]}" bash "$0" >"${CAMPAIGN_ROOT}/${lanes[$idx]}.console.log" 2>&1 &
    pids+=("$!")
    echo "$!" >"${CAMPAIGN_ROOT}/${lanes[$idx]}.pid"
    log master "SPAWN lane=${lanes[$idx]} card=${cards_array[$idx]} pid=$!"
    sleep 3
  done
  for idx in "${!pids[@]}"; do
    wait "${pids[$idx]}" || failed=1
  done
  if [[ ${failed} -eq 0 ]]; then
    touch "${CAMPAIGN_ROOT}/campaign.done"
    log master "CAMPAIGN_DONE"
  else
    touch "${CAMPAIGN_ROOT}/campaign.failed"
    log master "CAMPAIGN_FINISHED_WITH_FAILURES"
    return 1
  fi
}

main "$@"
