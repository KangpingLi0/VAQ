#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env_qig.sh"

PYTHON_BIN=${PYTHON_BIN:-/work/data/lkp/miniconda3/envs/qig-lmms-py311/bin/python}
DATASET_CACHE_ROOT=${DATASET_CACHE_ROOT:-${REPO_DIR}/outputs/lat_awq/dataset_cache_union}
CALIB_DATA=${CALIB_DATA:-${DATA_ROOT}/sharegpt4v/sharegpt4v_coco.json}
IMAGE_FOLDER=${IMAGE_FOLDER:-${DATA_ROOT}}
MODEL_7B=${MODEL_7B:-${MODEL_ROOT}/qwen2.5-vl}
MODEL_3B=${MODEL_3B:-${WORK_ROOT}/qwen2.5vl3b/Qwen2.5-VL-3B-Instruct}
ROOT_7B=${ROOT_7B:-${EXP_ROOT}/qwen25vl_7b_npu_methods}
ROOT_3B=${ROOT_3B:-${EXP_ROOT}/qwen25vl_3b_npu_methods}
STATUS_ROOT=${STATUS_ROOT:-${REPO_DIR}/logs/qwen25vl7b_new_then_3b_selected}

TASKS_7B=(mmmu_val ai2d scienceqa textvqa_val seedbench mmstar blink cv_bench vizwiz_vqa_val chartqa ocrbench)
EXPECTED_7B=(900 3088 4241 5000 17990 1500 1901 2638 4319 2500 1000)
TASKS_3B=(mmmu_val ai2d scienceqa textvqa_val seedbench mmstar blink cv_bench)
EXPECTED_3B=(900 3088 4241 5000 17990 1500 1901 2638)
TASKS_NEW=(mmstar blink cv_bench)
EXPECTED_NEW=(1500 1901 2638)

export PYTHONPATH="${REPO_DIR}/3rdparty/lmms-eval-new:${PYTHONPATH:-}"
export HF_DATASETS_CACHE="${DATASET_CACHE_ROOT}"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 TRANSFORMERS_OFFLINE=1
export QIG_SCALE_SEARCH_BATCH_SIZE=${QIG_SCALE_SEARCH_BATCH_SIZE:-1}
export ASCEND_GLOBAL_LOG_LEVEL=${ASCEND_GLOBAL_LOG_LEVEL:-3}
export ASCEND_SLOG_PRINT_TO_STDOUT=${ASCEND_SLOG_PRINT_TO_STDOUT:-0}

mkdir -p "${STATUS_ROOT}" "${ROOT_3B}"
cd "${REPO_DIR}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${STATUS_ROOT}/status.log"
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
if task not in data.get("results", {}) and task not in data.get("groups", {}):
    raise KeyError(f"{task} absent from {path}")
n_samples = data.get("n-samples", {})
if task in n_samples:
    actual = n_samples[task].get("effective")
else:
    subtasks = data.get("group_subtasks", {}).get(task, [])
    if not subtasks or any(name not in n_samples for name in subtasks):
        raise KeyError(f"Incomplete group metadata for {task} in {path}")
    actual = sum(n_samples[name].get("effective", 0) for name in subtasks)
if actual != expected:
    raise ValueError(f"{task}: expected {expected}, got {actual}")
print(path)
PY
}

method_values() {
  local root=$1 label=$2
  case "${label}" in
    fp16)          echo "fp16|16|16||" ;;
    w3a16_rtn)     echo "rtn|3|16||" ;;
    w3a16_gptq)    echo "gptq|3|16||--calib_data coco --data_path ${CALIB_DATA} --image_folder ${IMAGE_FOLDER} --n_samples 128 --percdamp 0.01" ;;
    w3a16_awq)     echo "awq|3|16|${root}/${label}/scale/awq_w3a16.pt|" ;;
    w3a16_mbq)     echo "mbq|3|16|${root}/${label}/scale/mbq_w3a16.pt|--reweight" ;;
    w3a16_qig)     echo "qig|3|16|${root}/${label}/scale/qig_w3a16.pt|--reweight" ;;
    w3a16_lat_awq) echo "lat_awq|3|16|${root}/${label}/scale/lat_awq_w3a16.pt|--token_aware_saliency --token_weighted_loss --saliency_mix_lambda 0.5" ;;
    w4a8_rtn)      echo "rtn|4|8||" ;;
    w4a8_sq)       echo "smoothquant|4|8|${root}/${label}/scale/sq_w4a8.pt|" ;;
    w4a8_mbq)      echo "mbq|4|8|${root}/${label}/scale/mbq_w4a8.pt|--reweight" ;;
    w4a8_qig)      echo "qig|4|8|${root}/${label}/scale/qig_w4a8.pt|--reweight" ;;
    w4a8_lat_awq)  echo "lat_awq|4|8|${root}/${label}/scale/lat_awq_w4a8.pt|--token_aware_saliency --token_weighted_loss --saliency_mix_lambda 0.5" ;;
    *) return 2 ;;
  esac
}

run_search() {
  local card=$1 model_path=$2 root=$3 label=$4 method=$5 w_bit=$6 a_bit=$7 scale=$8 extra=$9
  [[ -z "${scale}" ]] && return 0
  if [[ -s "${scale}" ]]; then
    log "SKIP_SEARCH model=$(basename "${root}") method=${label}"
    return 0
  fi
  mkdir -p "$(dirname "${scale}")" "${root}/${label}/calib"
  local cache=${root}/${label}/calib/sharegpt4v_coco_n128.pt
  log "START_SEARCH card=${card} model=$(basename "${root}") method=${label}"
  # extra is assembled only by method_values.
  if ! ASCEND_RT_VISIBLE_DEVICES="${card}" "${PYTHON_BIN}" -W ignore main_quant.py \
      --device npu --model qwen2_5_vl --model_args "pretrained=${model_path}" \
      --calib_data coco --data_path "${CALIB_DATA}" --image_folder "${IMAGE_FOLDER}" \
      --n_samples 128 --micro_batch_size 1 --calib_cache_path "${cache}" \
      --method "${method}" --run_process --w_bit "${w_bit}" --a_bit "${a_bit}" --w_group 128 \
      --scale_path "${scale}" ${extra} >"${root}/${label}/search.log" 2>&1; then
    log "FAILED_SEARCH card=${card} model=$(basename "${root}") method=${label}"
    return 1
  fi
  if [[ ! -s "${scale}" ]]; then
    log "FAILED_SEARCH_NO_SCALE card=${card} model=$(basename "${root}") method=${label}"
    return 1
  fi
  touch "${root}/${label}/search.done"
  log "DONE_SEARCH model=$(basename "${root}") method=${label}"
}

run_eval() {
  local card=$1 model_path=$2 root=$3 label=$4 task=$5 expected=$6
  local qv method w_bit a_bit scale extra out scale_arg="" eval_fd
  qv=$(method_values "${root}" "${label}")
  IFS='|' read -r method w_bit a_bit scale extra <<<"${qv}"
  out=${root}/${label}/eval_${task}
  mkdir -p "${out}"
  exec {eval_fd}>"${out}/eval.lock"
  flock "${eval_fd}"
  if validate_result "${out}" "${task}" "${expected}" >"${out}/result_path.txt" 2>/dev/null; then
    touch "${out}/eval.done"
    log "SKIP_EVAL model=$(basename "${root}") method=${label} task=${task}"
    exec {eval_fd}>&-
    return 0
  fi
  if [[ "${label}" != fp16 ]]; then
    run_search "${card}" "${model_path}" "${root}" "${label}" "${method}" "${w_bit}" "${a_bit}" "${scale}" "${extra}"
  fi
  [[ -n "${scale}" ]] && scale_arg="--scale_path ${scale}"
  log "START_EVAL card=${card} model=$(basename "${root}") method=${label} task=${task} expected=${expected}"
  if [[ "${label}" == fp16 ]]; then
    ASCEND_RT_VISIBLE_DEVICES="${card}" "${PYTHON_BIN}" -W ignore main.py \
      --device npu --model qwen2_5_vl --model_args "pretrained=${model_path}" \
      --tasks "${task}" --batch_size 1 --output_path "${out}" >"${out}/eval.log" 2>&1
  else
    # scale_arg and extra are assembled only by this script.
    ASCEND_RT_VISIBLE_DEVICES="${card}" "${PYTHON_BIN}" -W ignore main.py \
      --device npu --model qwen2_5_vl --model_args "pretrained=${model_path}" \
      --tasks "${task}" --batch_size 1 --output_path "${out}" \
      --method "${method}" --pseudo_quant --w_bit "${w_bit}" --a_bit "${a_bit}" --w_group 128 \
      ${scale_arg} ${extra} >"${out}/eval.log" 2>&1
  fi
  if ! validate_result "${out}" "${task}" "${expected}" >"${out}/result_path.txt"; then
    log "FAILED_EVAL_VALIDATION card=${card} model=$(basename "${root}") method=${label} task=${task}"
    exec {eval_fd}>&-
    return 1
  fi
  touch "${out}/eval.done"
  log "DONE_EVAL model=$(basename "${root}") method=${label} task=${task}"
  exec {eval_fd}>&-
}

run_task_set() {
  local card=$1 model_path=$2 root=$3 label=$4 set_name=$5
  local -n tasks_ref="TASKS_${set_name}"
  local -n expected_ref="EXPECTED_${set_name}"
  local idx
  for idx in "${!tasks_ref[@]}"; do
    if ! run_eval "${card}" "${model_path}" "${root}" "${label}" "${tasks_ref[$idx]}" "${expected_ref[$idx]}"; then
      log "FAILED_TASK_SET card=${card} model=$(basename "${root}") method=${label} task=${tasks_ref[$idx]}"
      return 1
    fi
  done
}

wait_for_pid() {
  local lane=$1 pid=$2
  local state
  while true; do
    state=$(ps -o stat= -p "${pid}" 2>/dev/null | awk '{print $1}' || true)
    [[ -z "${state}" || "${state}" == Z* ]] && break
    log "WAIT_EXISTING lane=${lane} pid=${pid}"
    sleep 60
  done
  log "EXISTING_DONE lane=${lane} pid=${pid}"
}

run_7b_lane() {
  local lane=$1 card=$2 parent=$3 child=$4
  wait_for_pid "${lane}" "${child}"
  kill -KILL "${parent}" 2>/dev/null || true
  local labels=()
  case "${lane}" in
    rtn) labels=(w3a16_rtn w4a8_rtn) ;;
    awq_sq) labels=(w3a16_awq w4a8_sq) ;;
    mbq) labels=(w3a16_mbq w4a8_mbq) ;;
    qig) labels=(w3a16_qig w4a8_qig) ;;
    gptq) labels=(w3a16_gptq) ;;
  esac
  local label
  for label in "${labels[@]}"; do
    run_task_set "${card}" "${MODEL_7B}" "${ROOT_7B}" "${label}" 7B
  done
  touch "${STATUS_ROOT}/7b_${lane}.done"
  log "LANE_7B_DONE lane=${lane} card=${card}"
}

run_7b_helper() {
  local card=3 label
  # No W4A8 RTN job is currently live, so complete its missing original tasks
  # immediately instead of leaving NPU 3 idle.
  run_task_set "${card}" "${MODEL_7B}" "${ROOT_7B}" w4a8_rtn 7B
  # Work-steal the new benchmarks from lanes that are still finishing
  # SeedBench. Per-evaluation locks make handoff race-free.
  for label in w3a16_awq w3a16_mbq w3a16_qig w3a16_rtn w4a8_sq w4a8_mbq w4a8_qig; do
    run_task_set "${card}" "${MODEL_7B}" "${ROOT_7B}" "${label}" NEW
  done
  touch "${STATUS_ROOT}/7b_helper_card3.done"
  log "LANE_7B_DONE lane=helper card=3"
}

run_3b_lane() {
  local lane=$1 card=$2 labels=()
  case "${lane}" in
    rtn) labels=(fp16 w3a16_rtn w4a8_rtn) ;;
    awq_sq) labels=(w3a16_awq w4a8_sq) ;;
    mbq) labels=(w3a16_mbq w4a8_mbq) ;;
    qig) labels=(w3a16_qig w4a8_qig) ;;
    gptq_lat) labels=(w3a16_gptq w3a16_lat_awq w4a8_lat_awq) ;;
  esac
  local label
  for label in "${labels[@]}"; do
    run_task_set "${card}" "${MODEL_3B}" "${ROOT_3B}" "${label}" 3B
  done
  touch "${STATUS_ROOT}/3b_${lane}.done"
  log "LANE_3B_DONE lane=${lane} card=${card}"
}

main() {
  test -x "${PYTHON_BIN}"
  test -d "${MODEL_7B}"
  test -f "${MODEL_3B}/config.json"
  test -f "${CALIB_DATA}"
  exec 9>"${STATUS_ROOT}/campaign.lock"
  flock -n 9 || { echo "Campaign already running" >&2; exit 9; }
  echo "$$" >"${STATUS_ROOT}/campaign.pid"
  log "CAMPAIGN_START phase=7B_new_then_3B_selected cards=0,1,2,4,5"

  run_7b_lane rtn 0 3334441 3334517 >"${STATUS_ROOT}/7b_rtn.console.log" 2>&1 & local p0=$!
  run_7b_lane awq_sq 1 2701370 3900073 >"${STATUS_ROOT}/7b_awq_sq.console.log" 2>&1 & local p1=$!
  run_7b_lane mbq 2 2702016 3320632 >"${STATUS_ROOT}/7b_mbq.console.log" 2>&1 & local p2=$!
  run_7b_helper >"${STATUS_ROOT}/7b_helper_card3.console.log" 2>&1 & local p3=$!
  run_7b_lane qig 4 2702018 3320812 >"${STATUS_ROOT}/7b_qig.console.log" 2>&1 & local p4=$!
  run_7b_lane gptq 5 2702020 3069220 >"${STATUS_ROOT}/7b_gptq.console.log" 2>&1 & local p5=$!
  local failed=0
  for pid in "${p0}" "${p1}" "${p2}" "${p3}" "${p4}" "${p5}"; do wait "${pid}" || failed=1; done
  [[ ${failed} -eq 0 ]] || { log "CAMPAIGN_FAILED phase=7B"; return 1; }
  touch "${STATUS_ROOT}/7b.done"
  log "PHASE_DONE phase=7B; START phase=3B"

  run_3b_lane rtn 0 >"${STATUS_ROOT}/3b_rtn.console.log" 2>&1 & p0=$!
  run_3b_lane awq_sq 1 >"${STATUS_ROOT}/3b_awq_sq.console.log" 2>&1 & p1=$!
  run_3b_lane mbq 2 >"${STATUS_ROOT}/3b_mbq.console.log" 2>&1 & p2=$!
  run_3b_lane qig 4 >"${STATUS_ROOT}/3b_qig.console.log" 2>&1 & p4=$!
  run_3b_lane gptq_lat 5 >"${STATUS_ROOT}/3b_gptq_lat.console.log" 2>&1 & p5=$!
  failed=0
  for pid in "${p0}" "${p1}" "${p2}" "${p4}" "${p5}"; do wait "${pid}" || failed=1; done
  [[ ${failed} -eq 0 ]] || { log "CAMPAIGN_FAILED phase=3B"; return 1; }
  touch "${STATUS_ROOT}/campaign.done"
  log "CAMPAIGN_DONE"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
