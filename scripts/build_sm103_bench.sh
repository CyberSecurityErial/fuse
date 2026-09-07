#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build/sm103}"
cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" -G Ninja \
  -DFUSE_ARCH=sm103 -DFUSE_BUILD_KERNELS=OFF -DFUSE_BUILD_BASELINES=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
cmake --build "${BUILD_DIR}" --parallel "${BUILD_JOBS:-4}"
