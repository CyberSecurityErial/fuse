#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/chen/miniforge3/envs/mmunlearner/bin/python}"
FLUX_ROOT="${FLUX_ROOT:-/home/chen/workspace/source_code/flux}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NVSHMEM_HOME="${NVSHMEM_HOME:-/home/chen/miniforge3/envs/mmunlearner/lib/python3.10/site-packages/nvidia/nvshmem}"

test -f "${FLUX_ROOT}/python/flux_ths_pybind.cpython-310-x86_64-linux-gnu.so"
test -f "${FLUX_ROOT}/build/lib/libflux_cuda.so"
test -f "${FLUX_ROOT}/build/lib/libflux_cuda_ths_op.so"

bash "${SCRIPT_DIR}/require_idle_gpus.sh"
export PYTHONPATH="${FLUX_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${NVSHMEM_HOME}/lib:${FLUX_ROOT}/build/lib:/home/chen/miniforge3/envs/mmunlearner/lib/python3.10/site-packages/torch/lib:/home/chen/miniforge3/envs/mmunlearner/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export NVSHMEM_HOME
export FLUX_USE_NVSHMEM="${FLUX_USE_NVSHMEM:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

exec "${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc-per-node="${NPROC_PER_NODE}" \
  "${REPO_ROOT}/benchmarks/flux_baseline.py" \
  "$@"
