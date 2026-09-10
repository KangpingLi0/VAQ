#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/../qig/env_qig.sh"

RUN_ROOT=${RUN_ROOT:-${REPO_DIR}/outputs/lat_awq/qwen25vl3b_w4a16_n128/formal_D_full_l05}
LOG_ROOT=${LOG_ROOT:-${REPO_DIR}/logs/lat_awq/qwen25vl3b_w4a16_n128/formal_D_full_l05}
SCALE_PATH=${RUN_ROOT}/scale/D_full_l05_n128.pt
PID_FILE=${LOG_ROOT}/formal.pid
LAUNCH_LOG=${LOG_ROOT}/launch.log

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

if [[ -f "${SCALE_PATH}" ]]; then
  echo "[LAT-AWQ formal] Already completed: ${SCALE_PATH}"
  exit 0
fi

if [[ -f "${PID_FILE}" ]]; then
  read -r existing_pid <"${PID_FILE}" || existing_pid=""
  if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    echo "[LAT-AWQ formal] Already running with PID ${existing_pid}"
    exit 0
  fi
fi

nohup bash "${REPO_DIR}/scripts/lat_awq/run_formal_qwen25vl3b_w4a16_n128.sh" \
  >"${LAUNCH_LOG}" 2>&1 </dev/null &
formal_pid=$!
printf '%s\n' "${formal_pid}" >"${PID_FILE}"

echo "[LAT-AWQ formal] Launched PID ${formal_pid}"
echo "[LAT-AWQ formal] Launch log: ${LAUNCH_LOG}"
