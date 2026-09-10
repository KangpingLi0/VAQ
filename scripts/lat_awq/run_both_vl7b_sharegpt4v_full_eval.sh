#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/../qig/env_qig.sh"

PIPELINE=${REPO_DIR}/scripts/lat_awq/run_qwen25vl7b_sharegpt4v_full_eval.sh
W_BIT=${W_BIT:-4}
A_BIT=${A_BIT:-16}
QUANT_TAG=w${W_BIT}a${A_BIT}
EVAL_CARDS=${EVAL_CARDS:-0,1,2,3,4,0,1,2}
DATASET_CACHE_ROOT=${DATASET_CACHE_ROOT:-${HF_DATASETS_CACHE}}
ORCH_ROOT=${REPO_DIR}/logs/lat_awq/both_vl7b_sharegpt4v_${QUANT_TAG}_n128
mkdir -p "${ORCH_ROOT}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${ORCH_ROOT}/orchestrator.log"
}

log "START qwen2.5-vl ${QUANT_TAG} full pipeline on quant NPU0 / eval NPU0-4"
MODEL_NAME=qwen2_5_vl \
W_BIT="${W_BIT}" \
A_BIT="${A_BIT}" \
DATASET_CACHE_ROOT="${DATASET_CACHE_ROOT}" \
QUANT_CARD=0 \
EVAL_CARDS="${EVAL_CARDS}" \
  "${PIPELINE}" 2>&1 | tee -a "${ORCH_ROOT}/qwen25vl7b.log" &
pid_qwen25=$!

sleep 10
log "START qwen2-vl ${QUANT_TAG} scale search on NPU5 while qwen2.5-vl continues"
MODEL_NAME=qwen2_vl \
W_BIT="${W_BIT}" \
A_BIT="${A_BIT}" \
DATASET_CACHE_ROOT="${DATASET_CACHE_ROOT}" \
QUANT_CARD=5 \
SEARCH_ONLY=1 \
  "${PIPELINE}" 2>&1 | tee -a "${ORCH_ROOT}/qwen2vl7b_search.log" &
pid_qwen2_search=$!

set +e
wait "${pid_qwen25}"
rc_qwen25=$?
wait "${pid_qwen2_search}"
rc_qwen2_search=$?
set -e

if [[ ${rc_qwen25} -ne 0 || ${rc_qwen2_search} -ne 0 ]]; then
  log "FAILED first stage qwen25=${rc_qwen25} qwen2_search=${rc_qwen2_search}"
  exit 1
fi

log "START qwen2-vl full evaluation on NPU0-4 (reusing completed scale)"
MODEL_NAME=qwen2_vl \
W_BIT="${W_BIT}" \
A_BIT="${A_BIT}" \
DATASET_CACHE_ROOT="${DATASET_CACHE_ROOT}" \
QUANT_CARD=5 \
EVAL_CARDS="${EVAL_CARDS}" \
  "${PIPELINE}" 2>&1 | tee -a "${ORCH_ROOT}/qwen2vl7b_eval.log"

touch "${ORCH_ROOT}/orchestrator.done"
log "DONE both VL-7B models"
