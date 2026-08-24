#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export NCCL_MIN_P2P_NCHANNELS=1
export NCCL_MAX_P2P_NCHANNELS=32
export NCCL_P2P_NVL_CHUNKSIZE=524288
export NCCL_P2P_LL_THRESHOLD=16384
export NCCL_GRAPH_REGISTER=1
export NCCL_LOCAL_REGISTER=1

exec bash "${SCRIPT_DIR}/run_te_nccl_baseline.sh" \
  --mode ulysses_forward \
  --global-seq 4096 \
  --hidden 8192 \
  --batch 1 \
  --q-heads 64 \
  --kv-heads 8 \
  --head-dim 128 \
  --warmup 10 \
  --iters 50 \
  --check \
  --include-te \
  --no-include-source \
  --cuda-graph \
  --pack-backend triton \
  --pack-block 1024 \
  --pack-warps 4 \
  --nccl-high-priority \
  --json-out "${REPO_ROOT}/results/te_cublas_nccl_frozen_latest.json" \
  "$@"
