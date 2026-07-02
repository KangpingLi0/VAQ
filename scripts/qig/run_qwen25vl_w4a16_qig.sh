#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env_qig.sh"

# TODO(qwen2.5-vl): check qwen2_vl adapter / loader.
# TODO(qwen2.5-vl): check processor.
# TODO(qwen2.5-vl): check image_grid_thw.
# TODO(qwen2.5-vl): check visual token positioning logic.
# TODO(qwen2.5-vl): check whether forward hooks are compatible with Qwen2.5-VL.

MODEL_PATH=${MODEL_PATH:-${MODEL_ROOT}/qwen2.5-vl}
CALIB_DATA=${CALIB_DATA:-${DATA_ROOT}/calib/calib.json}
IMAGE_FOLDER=${IMAGE_FOLDER:-${DATA_ROOT}/images}
CALIB_SIZE=${CALIB_SIZE:-128}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PYTHON_BIN=${PYTHON_BIN:-python}
OUT_DIR=${OUT_DIR:-${EXP_ROOT}/qwen25vl_w4a16_qig}
SCALE_PATH=${SCALE_PATH:-${OUT_DIR}/scale/qig_w4a16.pt}
MODEL_NAME=${MODEL_NAME:-qwen2_5_vl}
MODEL_ARGS=${MODEL_ARGS:-pretrained=${MODEL_PATH}}
DRY_RUN=${DRY_RUN:-1}
W_GROUP=${W_GROUP:-128}

for path_value in "$MODEL_PATH" "$CALIB_DATA" "$IMAGE_FOLDER" "$SCALE_PATH" "$OUT_DIR"; do
  if [[ "$path_value" == /home/lkp* ]]; then
    echo "[ERROR] Runtime paths must not be under /home/lkp: $path_value"
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES
mkdir -p "$(dirname "$SCALE_PATH")" "$OUT_DIR/logs"
cd "$REPO_DIR"

cmd=(
  "$PYTHON_BIN" -W ignore main_quant.py
  --model "$MODEL_NAME"
  --model_args "$MODEL_ARGS"
  --calib_data coco
  --data_path "$CALIB_DATA"
  --image_folder "$IMAGE_FOLDER"
  --n_samples "$CALIB_SIZE"
  --method qig
  --run_process
  --w_bit 4
  --a_bit 16
  --w_group "$W_GROUP"
  --reweight
  --loss_mode mae
  --scale_path "$SCALE_PATH"
)

echo "[QIG template] Qwen2.5-VL adapter validation is still TODO."
printf '[QIG template] Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[QIG template] DRY_RUN=1, command not executed. Set DRY_RUN=0 after adapter checks."
  exit 0
fi

"${cmd[@]}" 2>&1 | tee "$OUT_DIR/logs/scale_search.log"
