#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 ]]; then
  echo "Usage: CUDA_VISIBLE_DEVICES=0 bash $0 <micro-binary> <new-output-directory>" >&2
  exit 2
fi
binary="$1"
output="$2"
test -x "$binary"
if [[ -e "$output" ]]; then
  echo "Output already exists; select a new directory: $output" >&2
  exit 2
fi
mkdir -p "$output"
ncu --version > "$output/versions.txt"
nsys --version >> "$output/versions.txt"
ncu --set basic --clock-control none --cache-control none --launch-count 1 \
  --export "$output/micro" "$binary" > "$output/ncu.log" 2>&1
ncu --import "$output/micro.ncu-rep" --page raw --csv > "$output/ncu_metrics.csv"
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --output="$output/micro" "$binary" > "$output/nsys.log" 2>&1
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv \
  "$output/micro.nsys-rep" > "$output/nsys_stats.csv" 2> "$output/nsys_stats.log"
echo "Profile reports and exported metrics: $output"
