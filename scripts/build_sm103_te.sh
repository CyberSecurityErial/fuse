#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TE_ROOT="${TE_ROOT:-/root/workspace_wct/deps/TransformerEngine-a7aec214eb5c3969984a40c3accb6d66987d8f25}"
PYTHON_BIN="${PYTHON_BIN:-/root/workspace_wct/bench-env/bin/python}"
PYTHON_HEADERS="${PYTHON_HEADERS:-/root/workspace_wct/deps/python-headers-3.12.7}"
NCCL_HEADERS="${NCCL_HEADERS:-/root/workspace_wct/bench-env/lib64/python3.12/site-packages/nvidia/nccl/include}"
NCCL_LIBS="${NCCL_LIBS:-/root/workspace_wct/bench-env/lib64/python3.12/site-packages/nvidia/nccl/lib}"
CUDNN_HEADERS="${CUDNN_HEADERS:-/root/workspace_wct/bench-env/lib64/python3.12/site-packages/nvidia/cudnn/include}"
CUDNN_LIBS="${CUDNN_LIBS:-/root/workspace_wct/bench-env/lib64/python3.12/site-packages/nvidia/cudnn/lib}"
BUILD_JOBS="${BUILD_JOBS:-4}"
TE_BUILD_JOBS="${BUILD_JOBS}"
PATCH_FILE="${REPO_ROOT}/third_party/transformer_engine/a7aec214-sm103-userbuffers.patch"
EXPORT_PATCH_FILE="${REPO_ROOT}/third_party/transformer_engine/a7aec214-sm103-userbuffers-export.patch"
COMM_PATCH_FILE="${REPO_ROOT}/third_party/transformer_engine/a7aec214-sm103-userbuffers-communicator.patch"
P2P_PATCH_FILE="${REPO_ROOT}/third_party/transformer_engine/a7aec214-sm103-userbuffers-p2p-correctness.patch"
GRID_PATCH_FILE="${REPO_ROOT}/third_party/transformer_engine/a7aec214-sm103-userbuffers-arbitrary-grid.patch"

test -x "${PYTHON_BIN}"
test -f "${PYTHON_HEADERS}/Python.h"
test -f "${NCCL_HEADERS}/nccl.h"
test -f "${NCCL_LIBS}/libnccl.so.2"
test -f "${CUDNN_HEADERS}/cudnn.h"
test -f "${CUDNN_LIBS}/libcudnn.so.9"
test -f "${TE_ROOT}/setup.py"
test -f "${TE_ROOT}/3rdparty/cutlass/include/cutlass/cutlass.h"
test -f "${PATCH_FILE}"
test -f "${EXPORT_PATCH_FILE}"
test -f "${COMM_PATCH_FILE}"
test -f "${P2P_PATCH_FILE}"
test -f "${GRID_PATCH_FILE}"

if ! grep -q "configure_userbuffers_p2p" \
    "${TE_ROOT}/transformer_engine/pytorch/csrc/extensions.h"; then
  patch -d "${TE_ROOT}" -p1 < "${PATCH_FILE}"
fi
if ! grep -Fq '*userbuffers_send*' \
    "${TE_ROOT}/transformer_engine/common/libtransformer_engine.version"; then
  patch -d "${TE_ROOT}" -p1 < "${EXPORT_PATCH_FILE}"
fi

if ! grep -q 'CommOverlapCore::fuse_userbuffers_communicator' \
    "${TE_ROOT}/transformer_engine/common/comm_gemm_overlap/comm_gemm_overlap.cpp"; then
  patch --dry-run -d "${TE_ROOT}" -p1 < "${COMM_PATCH_FILE}"
  patch -d "${TE_ROOT}" -p1 < "${COMM_PATCH_FILE}"
fi

if ! grep -q 'Each push-copy CTA publishes one completion' \
    "${TE_ROOT}/transformer_engine/common/comm_gemm_overlap/userbuffers/userbuffers.cu"; then
  patch --dry-run -d "${TE_ROOT}" -p1 < "${P2P_PATCH_FILE}"
  patch -d "${TE_ROOT}" -p1 < "${P2P_PATCH_FILE}"
fi

if ! grep -q 'fuse arbitrary-grid copy stride' \
    "${TE_ROOT}/transformer_engine/common/comm_gemm_overlap/userbuffers/userbuffers.cu"; then
  patch --dry-run -d "${TE_ROOT}" -p1 < "${GRID_PATCH_FILE}"
  patch -d "${TE_ROOT}" -p1 < "${GRID_PATCH_FILE}"
fi

source /root/workspace_wct/env.sh
export NVTE_FRAMEWORK=pytorch
export NVTE_CUDA_ARCHS=100
export NVTE_WITH_NCCL_EP=0
export NVTE_BUILD_MAX_JOBS="${TE_BUILD_JOBS}"
export MAX_JOBS="${TE_BUILD_JOBS}"
export NVTE_SKIP_SUBMODULE_CHECKS_DURING_BUILD=1
export NVTE_PYTHON_INCLUDE_DIR="${PYTHON_HEADERS}"
# torch.utils.cpp_extension does not inherit the CMake-discovered Python/NCCL
# include directories. Keep both header and linker overrides local to this
# build process rather than changing the host toolchain configuration.
export CPATH="${PYTHON_HEADERS}:${NCCL_HEADERS}:${CUDNN_HEADERS}${CPATH:+:${CPATH}}"
export LIBRARY_PATH="${NCCL_LIBS}:${CUDNN_LIBS}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${NCCL_LIBS}:${CUDNN_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cd "${TE_ROOT}"
exec "${PYTHON_BIN}" -m pip install --no-build-isolation --no-deps -v .
