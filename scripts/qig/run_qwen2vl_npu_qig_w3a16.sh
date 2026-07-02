#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env_qig.sh"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-4}"
PYTHON_BIN=${PYTHON_BIN:-/work/data/lkp/miniconda3/envs/vtiq-py311/bin/python}
MODEL_PATH=${MODEL_PATH:-/work/model/lkp/models/qwen2-vl}
CALIB_DATA=${CALIB_DATA:-/work/model/lkp/datasets/sharegpt4v/sharegpt4v_coco.json}
IMAGE_FOLDER=${IMAGE_FOLDER:-/work/model/lkp/datasets}
OUT_DIR=${OUT_DIR:-/work/model/lkp/experiments/qig/npu_adapt/qwen2vl_w3a16_qig}
SCALE_PATH=${SCALE_PATH:-${OUT_DIR}/scale/qig_w3a16.pt}

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

for path_value in "$MODEL_PATH" "$CALIB_DATA" "$IMAGE_FOLDER" "$OUT_DIR" "$SCALE_PATH"; do
  if [[ "$path_value" == /home/lkp* ]]; then
    echo "[ERROR] Runtime paths must not be under /home/lkp: $path_value"
    exit 1
  fi
done

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] PYTHON_BIN is not executable: $PYTHON_BIN"
  exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
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

mkdir -p "${OUT_DIR}/scale" "${OUT_DIR}"
cd "$REPO_DIR"

"$PYTHON_BIN" -W ignore main_quant.py \
  --device npu \
  --model qwen2_vl \
  --model_args pretrained="${MODEL_PATH}" \
  --calib_data coco \
  --data_path "${CALIB_DATA}" \
  --image_folder "${IMAGE_FOLDER}" \
  --n_samples 128 \
  --method qig \
  --run_process \
  --w_bit 3 \
  --a_bit 16 \
  --w_group 128 \
  --reweight \
  --loss_mode mae \
  --scale_path "${SCALE_PATH}" \
  2>&1 | tee "${OUT_DIR}/quant.log"
