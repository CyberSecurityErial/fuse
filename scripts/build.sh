#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"

if [[ -z "${CUTLASS_ROOT:-}" ]]; then
  for candidate in \
    "${REPO_ROOT}/../TransformerEngine/3rdparty/cutlass" \
    "${REPO_ROOT}/../source_code/TransformerEngine/3rdparty/cutlass"; do
    [[ -f "${candidate}/include/cutlass/cutlass.h" ]] && CUTLASS_ROOT="${candidate}" && break
  done
fi
CUTLASS_ROOT="${CUTLASS_ROOT:-${REPO_ROOT}/../TransformerEngine/3rdparty/cutlass}"

if [[ ! -f "${CUTLASS_ROOT}/include/cutlass/cutlass.h" ]]; then
  echo "CUTLASS not found: ${CUTLASS_ROOT}" >&2
  echo "Set CUTLASS_ROOT to a CUTLASS source tree." >&2
  exit 2
fi
cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="${CUDACXX:-/usr/local/cuda/bin/nvcc}" \
  -DCUTLASS_ROOT="${CUTLASS_ROOT}"
cmake --build "${BUILD_DIR}" --parallel "${BUILD_JOBS:-8}"
