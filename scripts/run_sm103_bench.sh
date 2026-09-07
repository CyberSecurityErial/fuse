#!/usr/bin/env bash
# Run in a child process so offline JIT headers never alter the login shell.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/workspace_wct/bench-env/bin/python}"
PYTHON_HEADERS="${PYTHON_HEADERS:-/root/workspace_wct/deps/python-headers-3.12.7}"
source /root/workspace_wct/env.sh
export CPATH="${PYTHON_HEADERS}${CPATH:+:${CPATH}}"
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}"
cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -u benchmarks/sm103/bench.py --python "${PYTHON_BIN}" "$@"
