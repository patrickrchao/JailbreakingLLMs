#!/usr/bin/env bash
# Convenience launcher for run_ucsd_pair_variant.py
# - activates the local conda env
# - points OPENAI_API_KEY / OPENAI_BASE_URL at the UCSD TritonAI gateway
# - keeps all caches under this project folder (not /home)
#
# Usage examples:
#   ./run_variant.sh --dry-run                 # validate config, no API calls
#   ./run_variant.sh --limit 1                 # run 1 behavior (smoke test)
#   ./run_variant.sh --full --resume           # run all 100 behaviors
set -euo pipefail

PROJ="/data/fengfei/JailbreakingLLMs"
source /data/fengfei/RLUQ/miniconda3/etc/profile.d/conda.sh
conda activate "$PROJ/envs/pair"

export OPENAI_API_KEY="$(tr -d '[:space:]' < "$PROJ/api-key.txt")"
export OPENAI_BASE_URL="https://tritonai-api.ucsd.edu/v1"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PIP_CACHE_DIR="$PROJ/.pipcache"
export TMPDIR="$PROJ/.tmp"
export HF_HOME="$PROJ/.hf"
export WANDB_DIR="$PROJ/.wandb"
mkdir -p "$TMPDIR" "$HF_HOME" "$WANDB_DIR"

cd "$PROJ"
exec python run_ucsd_pair_variant.py \
  --api-key-file "$PROJ/api-key.txt" \
  --log-dir "$PROJ/logs/pair_ucsd_variant" \
  "$@"
