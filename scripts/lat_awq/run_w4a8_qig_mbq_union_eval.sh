#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/../qig/env_qig.sh"

RUNNER=${REPO_DIR}/scripts/lat_awq/run_both_vl7b_sharegpt4v_full_eval.sh
UNION_CACHE=${REPO_DIR}/outputs/lat_awq/dataset_cache_union
STATUS_ROOT=${REPO_DIR}/logs/lat_awq/w4a8_qig_mbq_union_full
STATUS_FILE=${STATUS_ROOT}/campaign_status.log
SEARCH_STATUS_ROOT=${REPO_DIR}/logs/lat_awq/w4a8_formal_campaign
EVAL_CARDS=${EVAL_CARDS:-0,1,3,4,5,0,1,3}
mkdir -p "${STATUS_ROOT}"

exec 9>"${STATUS_ROOT}/campaign.lock"
if ! flock -n 9; then
  echo "[ERROR] W4A8 full campaign is already running" >&2
  exit 9
fi

log_status() {
  echo "[$(date '+%F %T')] $*" | tee -a "${STATUS_FILE}"
}

wait_for_search() {
  local model_tag=$1
  local pid_file=$2
  local run_root=${REPO_DIR}/outputs/lat_awq/${model_tag}_sharegpt4v_w4a8_n128/formal_D_full_l05
  local done_file=${run_root}/search.done
  local scale_file=${run_root}/scale/D_full_l05_n128.pt

  while [[ ! -f "${done_file}" ]]; do
    if [[ ! -f "${pid_file}" ]]; then
      log_status "FAILED missing search PID file: ${pid_file}"
      return 1
    fi
    local pid
    pid=$(<"${pid_file}")
    if ! kill -0 "${pid}" 2>/dev/null; then
      log_status "FAILED ${model_tag} search pid=${pid} exited without ${done_file}"
      return 1
    fi
    log_status "WAIT_SEARCH ${model_tag} pid=${pid}"
    sleep 30
  done

  if [[ ! -s "${scale_file}" ]]; then
    log_status "FAILED missing scale after search: ${scale_file}"
    return 1
  fi
  log_status "SEARCH_READY ${model_tag} scale=${scale_file}"
}

echo "$$" >"${STATUS_ROOT}/campaign.pid"
log_status "START pid=$$ quant=W4A8 tasks=QIG_MBQ_union_8"

wait_for_search qwen25vl7b "${SEARCH_STATUS_ROOT}/qwen25_search.pid"
wait_for_search qwen2vl7b "${SEARCH_STATUS_ROOT}/qwen2_search.pid"

W_BIT=4 \
A_BIT=8 \
EVAL_CARDS="${EVAL_CARDS}" \
DATASET_CACHE_ROOT="${UNION_CACHE}" \
  bash "${RUNNER}"

tasks=(vizwiz_vqa_val mmmu_val chartqa ai2d scienceqa textvqa_val ocrbench seedbench)
for model in qwen25vl7b qwen2vl7b; do
  root=${REPO_DIR}/outputs/lat_awq/${model}_sharegpt4v_w4a8_n128/formal_D_full_l05/eval_qig_mbq_full
  for task in "${tasks[@]}"; do
    if [[ ! -f "${root}/${task}/eval.done" ]]; then
      echo "[ERROR] Missing completion marker: ${root}/${task}/eval.done" >&2
      exit 1
    fi
  done
done

touch "${STATUS_ROOT}/campaign.done"
log_status "DONE quant=W4A8 tasks=QIG_MBQ_union_8"
