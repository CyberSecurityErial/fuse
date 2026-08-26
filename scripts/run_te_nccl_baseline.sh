#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/chen/miniforge3/envs/mmunlearner/bin/python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MODE="${MODE:-oproj_a2a_gemm}"

case "${MODE}" in
  qkv_gemm_a2a)
    BASELINE="${REPO_ROOT}/benchmarks/QKVproj+a2a/te_nccl_baseline.py"
    ;;
  oproj_a2a_gemm)
    BASELINE="${REPO_ROOT}/benchmarks/a2a+Oproj/te_nccl_baseline.py"
    ;;
  *)
    echo "Unsupported MODE: ${MODE}" >&2
    exit 2
    ;;
esac

bash "${SCRIPT_DIR}/require_idle_gpus.sh"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

exec "${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc-per-node="${NPROC_PER_NODE}" \
  "${BASELINE}" \
  --mode "${MODE}" \
  "$@"
