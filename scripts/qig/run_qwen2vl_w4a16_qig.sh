#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env_qig.sh"

MODEL_PATH=${MODEL_PATH:-${MODEL_ROOT}/qwen2-vl}
CALIB_DATA=${CALIB_DATA:-${DATA_ROOT}/calib/calib.json}
IMAGE_FOLDER=${IMAGE_FOLDER:-${DATA_ROOT}/images}
CALIB_SIZE=${CALIB_SIZE:-128}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PYTHON_BIN=${PYTHON_BIN:-python}
OUT_DIR=${OUT_DIR:-${EXP_ROOT}/qwen2vl_w4a16_qig}
SCALE_PATH=${SCALE_PATH:-${OUT_DIR}/scale/qig_w4a16.pt}
MODEL_ARGS=${MODEL_ARGS:-pretrained=${MODEL_PATH}}
TASKS=${TASKS:-mmmu_val}
BATCH_SIZE=${BATCH_SIZE:-1}
EVAL_LIMIT=${EVAL_LIMIT:-}
W_GROUP=${W_GROUP:-128}
RUN_SEARCH=${RUN_SEARCH:-1}
RUN_EVAL=${RUN_EVAL:-1}
LOG_SAMPLES=${LOG_SAMPLES:-0}

for path_value in "$MODEL_PATH" "$CALIB_DATA" "$IMAGE_FOLDER" "$SCALE_PATH" "$OUT_DIR"; do
  if [[ "$path_value" == /home/lkp* ]]; then
    echo "[ERROR] Runtime paths must not be under /home/lkp: $path_value"
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES
mkdir -p "$(dirname "$SCALE_PATH")" "$OUT_DIR/logs" "$OUT_DIR/eval"
cd "$REPO_DIR"

if [[ "$RUN_SEARCH" == "1" ]]; then
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

  echo "[QIG baseline] Qwen2-VL QIG W4A16 scale search"
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
fi

if [[ "$RUN_EVAL" == "1" ]]; then
  if [[ ! -f "$SCALE_PATH" ]]; then
    echo "[ERROR] SCALE_PATH does not exist: $SCALE_PATH"
    exit 1
  fi

  eval_args=(
    -W ignore main.py
    --model qwen2_vl
    --model_args "$MODEL_ARGS"
    --tasks "$TASKS"
    --batch_size "$BATCH_SIZE"
    --output_path "$OUT_DIR/eval"
    --method qig
    --pseudo_quant
    --w_bit 4
    --a_bit 16
    --w_group "$W_GROUP"
    --scale_path "$SCALE_PATH"
  )
  if [[ -n "$EVAL_LIMIT" ]]; then
    eval_args+=(--limit "$EVAL_LIMIT")
  fi
  if [[ "$LOG_SAMPLES" == "1" ]]; then
    eval_args+=(--log_samples --log_samples_suffix qwen2vl_w4a16_qig)
  fi

  echo "[QIG baseline] Qwen2-VL QIG W4A16 pseudo quant lmms-eval"
  "$PYTHON_BIN" "${eval_args[@]}" 2>&1 | tee "$OUT_DIR/logs/eval.log"
fi

echo "[QIG baseline] Done. Output directory: $OUT_DIR"
