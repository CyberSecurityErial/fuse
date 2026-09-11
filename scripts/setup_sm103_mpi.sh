#!/usr/bin/env bash
# Offline, workspace-local MPICH dev tools. Never changes system packages or PATH.
set -euo pipefail

TASK_WORKSPACE=${FUSE_WORKSPACE:-/root/workspace_wct}
case "$TASK_WORKSPACE" in
  /root/workspace_wct|/home/*/workspace_wct) ;;
  *) echo "Expected a named fuse user workspace" >&2; exit 2 ;;
esac
[[ "$(readlink -f "$TASK_WORKSPACE")" == "$TASK_WORKSPACE" ]]
[[ -O "$TASK_WORKSPACE" ]]
MPI_PREFIX=${TASK_WORKSPACE}/toolchain/mpich-5.0.1.post1
MPI_WHEEL_SHA=edb42832e4d04fe3da78056edf74ef01d0405ad375dd990379d8d9bf2507b386
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MPI_TEST_SOURCE=${SCRIPT_DIR}/../benchmarks/sm103/mpi_smoke.cpp
if [[ $# -ne 1 || ! -f "$1" ]]; then
  echo "Usage: bash setup_sm103_mpi.sh /path/to/mpich-5.0.1.post1-...x86_64.whl" >&2
  exit 2
fi
case "$(hostname)" in
  l20d-xerkjfcp-0001|l20d-ebed3kz6-0000) ;;
  *) echo "Not a configured fuse L20D node" >&2; exit 2 ;;
esac
[[ "$(uname -m)" == x86_64 ]]
[[ "$(readlink -f "${TASK_WORKSPACE}/toolchain")" == "${TASK_WORKSPACE}/toolchain" ]]
[[ ! -L "${MPI_PREFIX}" && -f "${MPI_TEST_SOURCE}" ]]
[[ "$(sha256sum "$1" | cut -d ' ' -f 1)" == "${MPI_WHEEL_SHA}" ]]

# Share the controller lock; environment work cannot overlap a managed job.
exec 9>"${TASK_WORKSPACE}/.l20d/workspace.lock"
flock -n 9
source "${TASK_WORKSPACE}/env.sh"
if [[ ! -e "${MPI_PREFIX}" ]]; then
  "${TASK_WORKSPACE}/bench-env/bin/python" -m pip install \
    --no-index --no-deps --no-compile --ignore-installed --disable-pip-version-check \
    --prefix "${MPI_PREFIX}" "$1"
fi
# Existing or interrupted installations are inspected, never silently replaced.
test -f "${MPI_PREFIX}/include/mpi.h"
test -x "${MPI_PREFIX}/bin/mpicxx"
test -x "${MPI_PREFIX}/bin/mpiexec"
export MPICH_CXX="$(command -v g++)"
# This harness uses MPI only for same-host CPU control messages. GPU buffers
# use explicit CUDA IPC, not UCX. Avoid irrelevant RDMA bootstrap selection.
# https://openucx.readthedocs.io/en/master/faq.html#which-transports-does-ucx-use
export UCX_TLS=sm,self
"${MPI_PREFIX}/bin/mpicxx" -show
"${MPI_PREFIX}/bin/mpicxx" -std=c++17 -O2 "${MPI_TEST_SOURCE}" \
  -o "${MPI_PREFIX}/mpi_cpu_smoke"
ldd "${MPI_PREFIX}/mpi_cpu_smoke"
for MPI_WORLD in 2 8; do
  timeout 30 "${MPI_PREFIX}/bin/mpiexec" -launcher fork -n "${MPI_WORLD}" \
    "${MPI_PREFIX}/mpi_cpu_smoke" "${MPI_WORLD}"
done
du -sh "${MPI_PREFIX}"
echo "MPI workspace setup passed; CUDA IPC validation remains separate."
