#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/../qig/env_qig.sh"

RUNNER=${REPO_DIR}/scripts/lat_awq/run_both_vl7b_sharegpt4v_full_eval.sh
UNION_CACHE=${REPO_DIR}/outputs/lat_awq/dataset_cache_union
STATUS_ROOT=${REPO_DIR}/logs/lat_awq/w3a16_qig_mbq_union_full
STATUS_FILE=${STATUS_ROOT}/campaign_status.log
mkdir -p "${STATUS_ROOT}"

exec 9>"${STATUS_ROOT}/campaign.lock"
if ! flock -n 9; then
  echo "[ERROR] Extended W3A16 evaluation is already running" >&2
  exit 9
fi

echo "$$" >"${STATUS_ROOT}/campaign.pid"
echo "[$(date '+%F %T')] START pid=$$ tasks=QIG_MBQ_union_8" | tee -a "${STATUS_FILE}"

W_BIT=3 \
A_BIT=16 \
EVAL_CARDS=0,1,2,3,4,0,1,2 \
DATASET_CACHE_ROOT="${UNION_CACHE}" \
  bash "${RUNNER}"

for model in qwen25vl7b qwen2vl7b; do
  root=${REPO_DIR}/outputs/lat_awq/${model}_sharegpt4v_w3a16_n128/formal_D_full_l05/eval_qig_mbq_full
  for task in textvqa_val ocrbench seedbench; do
    if [[ ! -f "${root}/${task}/eval.done" ]]; then
      echo "[ERROR] Missing completion marker: ${root}/${task}/eval.done" >&2
      exit 1
    fi
  done
done

touch "${STATUS_ROOT}/campaign.done"
echo "[$(date '+%F %T')] DONE tasks=QIG_MBQ_union_8" | tee -a "${STATUS_FILE}"
