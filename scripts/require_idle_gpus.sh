#!/usr/bin/env bash
set -euo pipefail

mapfile -t gpu_pids < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null |
    awk 'NF {print $1}' | sort -nu
)

if ((${#gpu_pids[@]} == 0)); then
  exit 0
fi

echo "GPU benchmark not started: existing compute processes:" >&2
for pid in "${gpu_pids[@]}"; do
  ps -o user,pid,ppid,lstart,etime,stat,cmd -p "${pid}" >&2 || true
  if [[ -e "/proc/${pid}/cwd" ]]; then
    echo "cwd=$(readlink -f "/proc/${pid}/cwd")" >&2
  fi
done
exit 75
