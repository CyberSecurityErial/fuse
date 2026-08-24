#!/usr/bin/env bash
set -euo pipefail

mapfile -t pids < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    | tr -d ' ' \
    | awk '/^[0-9]+$/ { if (!seen[$1]++) print $1 }'
)

for pid in "${pids[@]}"; do
  [[ -r "/proc/${pid}/status" ]] || continue
  [[ "$(stat -c %U "/proc/${pid}")" == "$(id -un)" ]] || continue
  ps -o pid=,ppid=,pgid=,etime=,cmd= -p "${pid}" || true
  kill -KILL -- "${pid}" 2>/dev/null || true
done

