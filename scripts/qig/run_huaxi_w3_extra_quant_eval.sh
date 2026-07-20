#!/usr/bin/env bash
set -uo pipefail

cd /work/model/lkp/QIG
source scripts/qig/env_qig.sh
set -uo pipefail

export PYTHON_BIN="${PYTHON_BIN:-/work/data/lkp/miniconda3/envs/qig-lmms-py311/bin/python}"
export MODEL_PATH="${MODEL_PATH:-/work/model/lkp/vlm-compressor/checkpoint-1660}"
export RUN_ROOT="${RUN_ROOT:-/work/model/lkp/experiments/qig/huaxi_qwen25vl_checkpoint1660_full_224_128}"
export EVAL_ROOT="${EVAL_ROOT:-${RUN_ROOT}/eval_test3100}"
export CALIB_DATA="${CALIB_DATA:-/work/model/lkp/datasets/huaxi/huaxi_sampled_512_qig.json}"
export IMAGE_FOLDER="${IMAGE_FOLDER:-/}"
export CALIB_CACHE="${CALIB_CACHE:-/work/model/lkp/experiments/qig/huaxi_qwen25vl_checkpoint1660_chunk8_224/calib_inputs_224_full_128_v2.pt}"
export PAIRS_JSON="${PAIRS_JSON:-${EVAL_ROOT}/test_pairs.json}"
export MODEL_ARGS="${MODEL_ARGS:-pretrained=${MODEL_PATH},torch_dtype=float16,min_pixels=50176,max_pixels=50176}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
export QIG_SCALE_SEARCH_BATCH_SIZE="${QIG_SCALE_SEARCH_BATCH_SIZE:-16}"
export CARDS_CSV="${CARDS_CSV:-1,2,7}"

export PYTHONPATH="/work/model/lkp/QIG/3rdparty/lmms-eval-new:${PYTHONPATH:-}"
export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_HUB_CACHE="${CACHE_ROOT}/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/huggingface/hub"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/huggingface/transformers"
export HF_DATASETS_CACHE="${CACHE_ROOT}/huggingface/datasets"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TORCH_HOME="${CACHE_ROOT}/torch"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export TMPDIR="${CACHE_ROOT}/tmp"
export ASCEND_PROCESS_LOG_PATH="${CACHE_ROOT}/ascend/plog"
export ASCEND_WORK_PATH="${CACHE_ROOT}/ascend/work"
export ASCEND_CACHE_PATH="${CACHE_ROOT}/ascend/cache"
export TE_PARALLEL_COMPILER_CACHE="${CACHE_ROOT}/ascend/te"
export TUNE_BANK_PATH="${CACHE_ROOT}/ascend/tune"
export ASCEND_GLOBAL_LOG_LEVEL="${ASCEND_GLOBAL_LOG_LEVEL:-3}"
export ASCEND_SLOG_PRINT_TO_STDOUT="${ASCEND_SLOG_PRINT_TO_STDOUT:-0}"

mkdir -p \
  "${RUN_ROOT}" \
  "${EVAL_ROOT}" \
  "${HF_HOME}" \
  "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${XDG_CACHE_HOME}" \
  "${TORCH_HOME}" \
  "${TRITON_CACHE_DIR}" \
  "${PIP_CACHE_DIR}" \
  "${TMPDIR}" \
  "${ASCEND_PROCESS_LOG_PATH}" \
  "${ASCEND_WORK_PATH}" \
  "${ASCEND_CACHE_PATH}" \
  "${TE_PARALLEL_COMPILER_CACHE}" \
  "${TUNE_BANK_PATH}"

STATUS_FILE="${RUN_ROOT}/w3_extra_quant_eval_status.txt"
: > "${STATUS_FILE}"

IFS=',' read -r -a CARDS <<< "${CARDS_CSV}"

log_status() {
  echo "[$(date '+%F %T')] $*" | tee -a "${STATUS_FILE}"
}

fail_fast_checks() {
  test -x "${PYTHON_BIN}"
  test -d "${MODEL_PATH}"
  test -f "${CALIB_CACHE}"
  test -f "${PAIRS_JSON}"
  test -f /work/model/lkp/QIG/neijing/eval/eval_multi_level.py
  if [[ "${#CARDS[@]}" -lt 3 ]]; then
    echo "Need at least 3 cards in CARDS_CSV, got ${CARDS_CSV}" >&2
    return 2
  fi
}

has_fatal_error_log() {
  local path="$1"
  grep -E "Traceback|RuntimeError|AssertionError|FileNotFoundError|out of memory|OOM|ModuleNotFoundError|ImportError" "${path}" >/dev/null 2>&1
}

scale_path_for() {
  local name="$1"
  case "${name}" in
    w3a8_smoothquant) echo "${RUN_ROOT}/w3a8_smoothquant/scale/sq_w3a8.pt" ;;
    w3a8_mbq) echo "${RUN_ROOT}/w3a8_mbq/scale/mbq_w3a8.pt" ;;
    w3a8_qig) echo "${RUN_ROOT}/w3a8_qig/scale/qig_w3a8.pt" ;;
    *) echo "NONE" ;;
  esac
}

quant_args_for() {
  local name="$1"
  local scale_path
  scale_path="$(scale_path_for "${name}")"
  case "${name}" in
    w3a8_smoothquant)
      echo "--method smoothquant --run_process --w_bit 3 --a_bit 8 --scale_path ${scale_path}"
      ;;
    w3a8_mbq)
      echo "--method mbq --run_process --w_bit 3 --a_bit 8 --w_group 128 --reweight --distort --loss_mode mae --scale_path ${scale_path}"
      ;;
    w3a8_qig)
      echo "--method qig --run_process --w_bit 3 --a_bit 8 --w_group 128 --reweight --distort --loss_mode mae --scale_path ${scale_path}"
      ;;
    *)
      echo "unknown quant task: ${name}" >&2
      return 2
      ;;
  esac
}

eval_args_for() {
  local name="$1"
  case "${name}" in
    w3a16_rtn)
      echo "--method rtn --pseudo_quant --w_bit 3 --a_bit 16 --w_group 128 --calib_data none"
      ;;
    w3a16_gptq)
      echo "--method gptq --pseudo_quant --w_bit 3 --a_bit 16 --w_group 128 --percdamp 0.01 --calib_cache_path ${CALIB_CACHE} --calib_cache_n_samples 128"
      ;;
    w3a8_rtn)
      echo "--method rtn --pseudo_quant --w_bit 3 --a_bit 8 --w_group 128 --calib_data none"
      ;;
    w3a8_smoothquant)
      echo "--method smoothquant --pseudo_quant --w_bit 3 --a_bit 8 --calib_data none --scale_path ${RUN_ROOT}/w3a8_smoothquant/scale/sq_w3a8.pt"
      ;;
    w3a8_mbq)
      echo "--method mbq --pseudo_quant --w_bit 3 --a_bit 8 --w_group 128 --reweight --distort --loss_mode mae --calib_data none --scale_path ${RUN_ROOT}/w3a8_mbq/scale/mbq_w3a8.pt"
      ;;
    w3a8_qig)
      echo "--method qig --pseudo_quant --w_bit 3 --a_bit 8 --w_group 128 --reweight --distort --loss_mode mae --calib_data none --scale_path ${RUN_ROOT}/w3a8_qig/scale/qig_w3a8.pt"
      ;;
    *)
      echo "unknown eval task: ${name}" >&2
      return 2
      ;;
  esac
}

postprocess_predictions() {
  local raw_json="$1"
  local pred_json="$2"
  "${PYTHON_BIN}" - "$raw_json" "$pred_json" "$PAIRS_JSON" <<'PY'
import json
import sys
from pathlib import Path

raw_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
pairs_path = Path(sys.argv[3])

raw = json.load(raw_path.open("r", encoding="utf-8"))
results = raw.get("results", raw) if isinstance(raw, dict) else raw
pairs = json.load(pairs_path.open("r", encoding="utf-8"))
label_by_id = {str(item.get("id")): item.get("label", "") for item in pairs}

out = []
for idx, item in enumerate(results):
    item_id = item.get("id", idx)
    pred = item.get("pred", item.get("answer", item.get("predict", "")))
    row = dict(item)
    row["id"] = item_id
    row["pred"] = pred
    row["predict"] = pred
    row["label"] = row.get("label", label_by_id.get(str(item_id), ""))
    out.append(row)

out_path.parent.mkdir(parents=True, exist_ok=True)
json.dump(out, out_path.open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"saved_eval_json={out_path}")
print(f"sample_count={len(out)}")
PY
}

run_quant() {
  local card="$1"
  local name="$2"
  local out_dir="${RUN_ROOT}/${name}"
  local scale_path
  local extra

  scale_path="$(scale_path_for "${name}")"
  mkdir -p "${out_dir}/scale"

  if [[ -f "${out_dir}/quant.done" && -f "${scale_path}" ]]; then
    log_status "SKIP_QUANT ${name} existing scale"
    return 0
  fi

  rm -f "${out_dir}/quant.done" "${out_dir}/quant.failed" "${scale_path}"
  log_status "START_QUANT ${name} card=${card}"
  extra="$(quant_args_for "${name}")" || return 2

  {
    echo "[START] ${name} card=${card} $(date)"
    echo "PYTHON_BIN=${PYTHON_BIN}"
    echo "MODEL_PATH=${MODEL_PATH}"
    echo "CALIB_CACHE=${CALIB_CACHE}"
    echo "MODEL_ARGS=${MODEL_ARGS}"
    echo "SCALE_PATH=${scale_path}"
    echo "EXTRA_ARGS=${extra}"
  } > "${out_dir}/screen.out"

  set +e
  ASCEND_RT_VISIBLE_DEVICES="${card}" "${PYTHON_BIN}" -W ignore main_quant.py \
    --device npu \
    --model qwen2_5_vl \
    --model_args "${MODEL_ARGS}" \
    --calib_data coco \
    --data_path "${CALIB_DATA}" \
    --image_folder "${IMAGE_FOLDER}" \
    --n_samples 128 \
    --micro_batch_size 1 \
    --calib_image_size 224 \
    --calib_image_chunk_size 0 \
    --calib_cache_path "${CALIB_CACHE}" \
    ${extra} \
    > "${out_dir}/quant.log" 2>&1
  local status=$?
  set -uo pipefail

  echo "[EXIT] ${name} status=${status} $(date)" >> "${out_dir}/screen.out"
  if [[ ${status} -eq 0 && -f "${scale_path}" ]] && ! has_fatal_error_log "${out_dir}/quant.log"; then
    touch "${out_dir}/quant.done"
    echo "[DONE] ${name} $(date)" >> "${out_dir}/screen.out"
    log_status "DONE_QUANT ${name}"
    return 0
  fi

  touch "${out_dir}/quant.failed"
  echo "[FAILED] ${name} $(date)" >> "${out_dir}/screen.out"
  log_status "FAILED_QUANT ${name} status=${status}"
  return 1
}

run_eval() {
  local card="$1"
  local name="$2"
  local out_dir="${EVAL_ROOT}/${name}"
  local raw_json="${out_dir}/predictions_raw.json"
  local pred_json="${out_dir}/predictions.json"
  local infer_log="${out_dir}/inference.log"
  local metrics_log="${out_dir}/metrics.log"
  local metrics_txt="${out_dir}/metrics.txt"
  local scale_path
  local extra

  mkdir -p "${out_dir}"
  scale_path="$(scale_path_for "${name}")"

  if [[ -f "${out_dir}/eval.done" && -f "${pred_json}" && -f "${metrics_txt}" ]]; then
    log_status "SKIP_EVAL ${name} existing eval.done"
    return 0
  fi
  if [[ "${scale_path}" != "NONE" && ! -f "${scale_path}" ]]; then
    touch "${out_dir}/eval.failed"
    log_status "FAILED_EVAL ${name} missing_scale ${scale_path}"
    return 1
  fi

  rm -f "${out_dir}/eval.done" "${out_dir}/eval.failed"
  log_status "START_EVAL ${name} card=${card}"
  extra="$(eval_args_for "${name}")" || return 2

  {
    echo "[START] ${name} card=${card} $(date)"
    echo "PYTHON_BIN=${PYTHON_BIN}"
    echo "MODEL_PATH=${MODEL_PATH}"
    echo "MODEL_ARGS=${MODEL_ARGS}"
    echo "PAIRS_JSON=${PAIRS_JSON}"
    echo "RAW_JSON=${raw_json}"
    echo "PRED_JSON=${pred_json}"
    echo "EXTRA_ARGS=${extra}"
  } > "${out_dir}/screen.out"

  set +e
  ASCEND_RT_VISIBLE_DEVICES="${card}" "${PYTHON_BIN}" -W ignore inference.py \
    --device npu \
    --model qwen2_5_vl \
    --model_args "${MODEL_ARGS}" \
    --batch_size 1 \
    ${extra} \
    --infer_pairs "${PAIRS_JSON}" \
    --save_path "${raw_json}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature 0.0 \
    > "${infer_log}" 2>&1
  local infer_status=$?
  set -uo pipefail

  if [[ ${infer_status} -ne 0 || ! -f "${raw_json}" ]] || has_fatal_error_log "${infer_log}"; then
    touch "${out_dir}/eval.failed"
    echo "[FAILED inference] status=${infer_status} $(date)" >> "${out_dir}/screen.out"
    log_status "FAILED_EVAL ${name} inference status=${infer_status}"
    return 1
  fi

  : > "${metrics_log}"
  postprocess_predictions "${raw_json}" "${pred_json}" >> "${metrics_log}" 2>&1
  (
    cd "${out_dir}" && \
      "${PYTHON_BIN}" /work/model/lkp/QIG/neijing/eval/eval_multi_level.py "${pred_json}" \
        >> "${metrics_log}" 2>&1
  )
  local metric_status=$?

  if [[ ${metric_status} -ne 0 ]] || has_fatal_error_log "${metrics_log}"; then
    touch "${out_dir}/eval.failed"
    echo "[FAILED metrics] status=${metric_status} $(date)" >> "${out_dir}/screen.out"
    log_status "FAILED_EVAL ${name} metrics status=${metric_status}"
    return 1
  fi

  cp "${out_dir}/results_eval/predictions.txt" "${metrics_txt}"
  touch "${out_dir}/eval.done"
  echo "[DONE] ${name} $(date)" >> "${out_dir}/screen.out"
  log_status "DONE_EVAL ${name}"
}

run_card_chain() {
  local card="$1"
  shift
  local status=0

  for task in "$@"; do
    local phase="${task%%:*}"
    local name="${task#*:}"
    if [[ "${phase}" == "quant" ]]; then
      run_quant "${card}" "${name}" || status=1
    elif [[ "${phase}" == "eval" ]]; then
      run_eval "${card}" "${name}" || status=1
    else
      log_status "UNKNOWN_TASK ${task}"
      status=1
    fi
  done
  return "${status}"
}

generate_excel() {
  "${PYTHON_BIN}" tools/export_huaxi_quant_summary_xlsx.py \
    --run-root "${RUN_ROOT}" \
    --eval-root "${EVAL_ROOT}" \
    --out "${EVAL_ROOT}/quantization_results_summary_latest.xlsx"
}

main() {
  fail_fast_checks
  log_status "DRIVER_START model=${MODEL_PATH} cards=${CARDS_CSV}"

  local status=0
  run_card_chain "${CARDS[0]}" \
    "eval:w3a16_rtn" \
    "eval:w3a8_rtn" \
    "quant:w3a8_smoothquant" \
    "eval:w3a8_smoothquant" &
  local p0=$!

  sleep 10
  run_card_chain "${CARDS[1]}" \
    "eval:w3a16_gptq" \
    "quant:w3a8_mbq" \
    "eval:w3a8_mbq" &
  local p1=$!

  sleep 10
  run_card_chain "${CARDS[2]}" \
    "quant:w3a8_qig" \
    "eval:w3a8_qig" &
  local p2=$!

  for pid in "${p0}" "${p1}" "${p2}"; do
    wait "${pid}" || status=1
  done

  generate_excel | tee -a "${RUN_ROOT}/w3_extra_excel.log"
  log_status "DRIVER_DONE status=${status}"
  return "${status}"
}

main "$@"
