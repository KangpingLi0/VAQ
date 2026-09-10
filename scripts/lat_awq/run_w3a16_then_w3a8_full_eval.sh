#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/../qig/env_qig.sh"

PIPELINE=${REPO_DIR}/scripts/lat_awq/run_both_vl7b_sharegpt4v_full_eval.sh
CAMPAIGN_ROOT=${REPO_DIR}/logs/lat_awq/w3a16_then_w3a8_sharegpt4v_n128
STATUS_FILE=${CAMPAIGN_ROOT}/campaign_status.log
mkdir -p "${CAMPAIGN_ROOT}"

exec 9>"${CAMPAIGN_ROOT}/campaign.lock"
if ! flock -n 9; then
  echo "[ERROR] Campaign is already running: ${CAMPAIGN_ROOT}" >&2
  exit 9
fi

echo "$$" >"${CAMPAIGN_ROOT}/campaign.pid"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${STATUS_FILE}"
}

run_stage() {
  local a_bit=$1
  local tag=w3a${a_bit}
  local done_file=${REPO_DIR}/logs/lat_awq/both_vl7b_sharegpt4v_${tag}_n128/orchestrator.done

  if [[ -f "${done_file}" ]]; then
    log "SKIP ${tag}: validated orchestrator completion marker exists"
    return 0
  fi

  log "START ${tag}: two-model quantization and five-task full evaluation"
  W_BIT=3 A_BIT="${a_bit}" bash "${PIPELINE}"
  if [[ ! -f "${done_file}" ]]; then
    log "FAILED ${tag}: pipeline returned without completion marker"
    return 1
  fi
  log "DONE ${tag}"
}

log "CAMPAIGN_START pid=$$ calibration=ShareGPT4V-COCO n=128 order=w3a16,w3a8"
run_stage 16
run_stage 8
touch "${CAMPAIGN_ROOT}/campaign.done"
log "CAMPAIGN_DONE"
