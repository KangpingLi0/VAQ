#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/../qig/env_qig.sh"

PYTHON_BIN=${PYTHON_BIN:-/work/data/lkp/miniconda3/envs/qig-lmms-py311/bin/python}
MODEL_PATH=${MODEL_PATH:-${REPO_DIR}/neijing/models/qwen2.5vl3b-checkpoint-1270}
SCALE_ROOT=${SCALE_ROOT:-${REPO_DIR}/outputs/lat_awq/qwen25vl3b_w4a16_n2/scale}
SOURCE_PAIRS=${SOURCE_PAIRS:-${EXP_ROOT}/huaxi_qwen25vl3b_checkpoint1270_full_224_128/eval_test3100/test_pairs.json}
EVAL_ROOT=${EVAL_ROOT:-${REPO_DIR}/outputs/lat_awq/qwen25vl3b_w4a16_n2/tiny_eval_20}
MODEL_ARGS=${MODEL_ARGS:-pretrained=${MODEL_PATH},torch_dtype=float16,min_pixels=50176,max_pixels=50176}
EVAL_SAMPLES=${EVAL_SAMPLES:-20}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}

export PYTHONPATH="${REPO_DIR}/3rdparty/lmms-eval-new:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "${EVAL_ROOT}"
cd "${REPO_DIR}"

PAIRS=${EVAL_ROOT}/pairs.json
"${PYTHON_BIN}" - "${SOURCE_PAIRS}" "${PAIRS}" "${EVAL_SAMPLES}" <<'PY'
import json
import sys
from pathlib import Path

source, output, count = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
rows = json.loads(source.read_text(encoding="utf-8"))[:count]
missing = [path for row in rows for path in row.get("images", []) if not Path(path).exists()]
if missing:
    raise FileNotFoundError(f"Missing evaluation images, first examples: {missing[:5]}")
output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Saved {len(rows)} evaluation pairs to {output}")
PY

run_eval() {
  local card=$1
  local name=$2
  local scale=$3
  local saliency=$4
  local weighted=$5
  local lambda=${6:-1.0}
  local out=${EVAL_ROOT}/${name}
  mkdir -p "${out}/logs"
  if [[ -f "${out}/metrics.txt" ]]; then
    echo "[tiny-eval] Reusing ${out}/metrics.txt"
    return 0
  fi
  while [[ ! -f "${scale}" ]]; do
    echo "[tiny-eval] ${name} waiting for ${scale}"
    sleep 10
  done
  local flags=()
  if [[ "${saliency}" == 1 ]]; then flags+=(--token_aware_saliency); fi
  if [[ "${weighted}" == 1 ]]; then flags+=(--token_weighted_loss); fi
  ASCEND_RT_VISIBLE_DEVICES=${card} "${PYTHON_BIN}" -W ignore inference.py \
    --device npu \
    --model qwen2_5_vl \
    --model_args "${MODEL_ARGS}" \
    --batch_size 1 \
    --method lat_awq \
    --pseudo_quant \
    --w_bit 4 \
    --a_bit 16 \
    --w_group 128 \
    --scale_path "${scale}" \
    --saliency_mix_lambda "${lambda}" \
    "${flags[@]}" \
    --lat_output_dir "${out}/run_metadata" \
    --lat_log_dir "${out}/logs" \
    --infer_pairs "${PAIRS}" \
    --save_path "${out}/predictions_raw.json" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature 0.0 \
    >"${out}/logs/inference.log" 2>&1

  "${PYTHON_BIN}" - "${out}/predictions_raw.json" "${out}/predictions.json" <<'PY'
import json
import sys
from pathlib import Path

raw = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
rows = raw.get("results", raw) if isinstance(raw, dict) else raw
for row in rows:
    pred = row.get("pred", row.get("answer", ""))
    row["pred"] = pred
    row["predict"] = pred
Path(sys.argv[2]).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
PY
  (
    cd "${out}"
    "${PYTHON_BIN}" "${REPO_DIR}/neijing/eval/eval_multi_level.py" predictions.json \
      >logs/metrics.log 2>&1
    cp results_eval/predictions.txt metrics.txt
  )
  echo "[tiny-eval] ${name} completed"
}

run_eval 2 A_awq_like "${SCALE_ROOT}/A_awq_like.pt" 0 0 1.0 &
pid_a=$!
run_eval 3 B_loss_only "${SCALE_ROOT}/B_loss_only.pt" 0 1 1.0 &
pid_b=$!
run_eval 4 C_saliency_only "${SCALE_ROOT}/C_saliency_only_l1.pt" 1 0 1.0 &
pid_c=$!
run_eval 5 D_full "${SCALE_ROOT}/D_full_l1.pt" 1 1 1.0 &
pid_d=$!

status=0
for pid in "${pid_a}" "${pid_b}" "${pid_c}" "${pid_d}"; do
  wait "${pid}" || status=1
done

# Lambda sweep. D_full_l1 above is the lambda=1 endpoint.
run_eval 2 D_full_l0 "${SCALE_ROOT}/D_full_l0.pt" 1 1 0.0 &
pid_l0=$!
run_eval 3 D_full_l025 "${SCALE_ROOT}/D_full_l025.pt" 1 1 0.25 &
pid_l025=$!
run_eval 4 D_full_l05 "${SCALE_ROOT}/D_full_l05.pt" 1 1 0.5 &
pid_l05=$!
run_eval 5 D_full_l075 "${SCALE_ROOT}/D_full_l075.pt" 1 1 0.75 &
pid_l075=$!
for pid in "${pid_l0}" "${pid_l025}" "${pid_l05}" "${pid_l075}"; do
  wait "${pid}" || status=1
done
exit "${status}"
