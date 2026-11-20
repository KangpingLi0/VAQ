#!/usr/bin/env bash
set -euo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH="/data/lkp/envs/java/jdk8u512-b09-jre/bin:${PATH}"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
cd /data/lkp/VAQ
exec /data/lkp/envs/vaq-neijing/bin/python -u scripts/neijing/schedule.py "$@"
