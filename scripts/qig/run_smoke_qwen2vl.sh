#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env_qig.sh"

MODEL_PATH=${MODEL_PATH:-${MODEL_ROOT}/qwen2-vl}
CALIB_DATA=${CALIB_DATA:-${DATA_ROOT}/calib/calib.json}
IMAGE_FOLDER=${IMAGE_FOLDER:-${DATA_ROOT}/images}
EVAL_DATA=${EVAL_DATA:-${DATA_ROOT}/eval/eval.json}
CALIB_SIZE=${CALIB_SIZE:-2}
EVAL_SIZE=${EVAL_SIZE:-2}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PYTHON_BIN=${PYTHON_BIN:-python}
SCALE_PATH=${SCALE_PATH:-${EXP_ROOT}/qwen2vl_smoke/scale/qig_w4a16.pt}
OUT_DIR=${OUT_DIR:-${EXP_ROOT}/qwen2vl_smoke}
MODEL_ARGS=${MODEL_ARGS:-pretrained=${MODEL_PATH}}
W_GROUP=${W_GROUP:-128}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}

for path_value in "$MODEL_PATH" "$CALIB_DATA" "$IMAGE_FOLDER" "$EVAL_DATA" "$SCALE_PATH" "$OUT_DIR"; do
  if [[ "$path_value" == /home/lkp* ]]; then
    echo "[ERROR] Runtime paths must not be under /home/lkp: $path_value"
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES
mkdir -p "$(dirname "$SCALE_PATH")" "$OUT_DIR" "$OUT_DIR/logs" "$OUT_DIR/tmp"
cd "$REPO_DIR"

if [[ ! -e "$MODEL_PATH" ]]; then
  echo "[ERROR] MODEL_PATH does not exist: $MODEL_PATH"
  exit 1
fi
if [[ ! -f "$CALIB_DATA" ]]; then
  echo "[ERROR] CALIB_DATA does not exist: $CALIB_DATA"
  exit 1
fi
if [[ ! -d "$IMAGE_FOLDER" ]]; then
  echo "[ERROR] IMAGE_FOLDER does not exist: $IMAGE_FOLDER"
  exit 1
fi
if [[ ! -f "$EVAL_DATA" ]]; then
  echo "[ERROR] EVAL_DATA does not exist: $EVAL_DATA"
  exit 1
fi

EVAL_SUBSET="${OUT_DIR}/tmp/eval_subset.json"
"$PYTHON_BIN" - "$EVAL_DATA" "$EVAL_SUBSET" "$EVAL_SIZE" <<'PY'
import json
import sys

src, dst, size_raw = sys.argv[1], sys.argv[2], sys.argv[3]
limit = int(size_raw)

if src.endswith(".jsonl"):
    rows = []
    with open(src, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
else:
    with open(src, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    rows = loaded if isinstance(loaded, list) else loaded.get("data", [])

if limit > 0:
    rows = rows[:limit]

with open(dst, "w", encoding="utf-8") as handle:
    json.dump(rows, handle, ensure_ascii=False, indent=2)
PY

echo "[QIG smoke] Step 1/2: Qwen2-VL QIG W4A16 scale search"
"$PYTHON_BIN" -W ignore main_quant.py \
  --model qwen2_vl \
  --model_args "$MODEL_ARGS" \
  --calib_data coco \
  --data_path "$CALIB_DATA" \
  --image_folder "$IMAGE_FOLDER" \
  --n_samples "$CALIB_SIZE" \
  --method qig \
  --run_process \
  --w_bit 4 \
  --a_bit 16 \
  --w_group "$W_GROUP" \
  --reweight \
  --loss_mode mae \
  --scale_path "$SCALE_PATH" \
  2>&1 | tee "$OUT_DIR/logs/scale_search.log"

if [[ ! -f "$SCALE_PATH" ]]; then
  echo "[ERROR] QIG scale file was not created: $SCALE_PATH"
  exit 1
fi

echo "[QIG smoke] Step 2/2: pseudo quant inference eval"
"$PYTHON_BIN" -W ignore inference.py \
  --model qwen2_vl \
  --model_args "$MODEL_ARGS" \
  --calib_data coco \
  --data_path "$CALIB_DATA" \
  --image_folder "$IMAGE_FOLDER" \
  --n_samples "$CALIB_SIZE" \
  --method qig \
  --pseudo_quant \
  --w_bit 4 \
  --a_bit 16 \
  --w_group "$W_GROUP" \
  --scale_path "$SCALE_PATH" \
  --infer_pairs "$EVAL_SUBSET" \
  --save_path "$OUT_DIR/pseudo_quant_preds.json" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  2>&1 | tee "$OUT_DIR/logs/pseudo_quant_eval.log"

echo "[QIG smoke] Done. Scale: $SCALE_PATH"
echo "[QIG smoke] Output: $OUT_DIR/pseudo_quant_preds.json"
