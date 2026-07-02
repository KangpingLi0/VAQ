#!/usr/bin/env bash
set -e

export WORK_ROOT="${WORK_ROOT:-/work/model/lkp}"
export REPO_DIR="${REPO_DIR:-${WORK_ROOT}/QIG}"
export CACHE_ROOT="${CACHE_ROOT:-${WORK_ROOT}/cache}"
export MODEL_ROOT="${MODEL_ROOT:-${WORK_ROOT}/models}"
export DATA_ROOT="${DATA_ROOT:-${WORK_ROOT}/datasets}"
export EXP_ROOT="${EXP_ROOT:-${WORK_ROOT}/experiments/qig}"

if [[ "$WORK_ROOT" == /home/lkp* || "$REPO_DIR" == /home/lkp* || "$CACHE_ROOT" == /home/lkp* || "$MODEL_ROOT" == /home/lkp* || "$DATA_ROOT" == /home/lkp* || "$EXP_ROOT" == /home/lkp* ]]; then
  echo "[ERROR] Paths must not be under /home/lkp."
  exit 1
fi

mkdir -p "$CACHE_ROOT" "$MODEL_ROOT" "$DATA_ROOT" "$EXP_ROOT"

export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_HUB_CACHE="${CACHE_ROOT}/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/huggingface/hub"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/huggingface/transformers"
export HF_DATASETS_CACHE="${CACHE_ROOT}/huggingface/datasets"
export TORCH_HOME="${CACHE_ROOT}/torch"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export WANDB_DIR="${CACHE_ROOT}/wandb"
export WANDB_CACHE_DIR="${CACHE_ROOT}/wandb/cache"
export TMPDIR="${CACHE_ROOT}/tmp"

mkdir -p \
  "$XDG_CACHE_HOME" \
  "$HF_HOME" \
  "$HF_HUB_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$TORCH_HOME" \
  "$TRITON_CACHE_DIR" \
  "$PIP_CACHE_DIR" \
  "$WANDB_DIR" \
  "$WANDB_CACHE_DIR" \
  "$TMPDIR"

if [[ "$PWD" == /home/lkp* ]]; then
  echo "[ERROR] Current working directory is under /home/lkp. Please run under /work/model/lkp/QIG."
  exit 1
fi
