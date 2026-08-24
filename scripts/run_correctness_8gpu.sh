#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

bash "${SCRIPT_DIR}/require_idle_gpus.sh"
bash "${SCRIPT_DIR}/build.sh"
"${REPO_ROOT}/build/fuse_smoke"
compute-sanitizer --tool memcheck --error-exitcode 99 \
  "${REPO_ROOT}/build/fuse_smoke"
compute-sanitizer --tool racecheck --error-exitcode 99 \
  "${REPO_ROOT}/build/fuse_smoke"
