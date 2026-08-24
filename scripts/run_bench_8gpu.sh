#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

bash "${SCRIPT_DIR}/require_idle_gpus.sh"
bash "${SCRIPT_DIR}/build.sh"
mkdir -p "${REPO_ROOT}/results"

"${REPO_ROOT}/build/fuse_bench" \
  --mode a2a_gemm --m 2048 --n 5120 --k 4096 \
  --comm-ctas 12 --raster n --swizzle 1 --warmup 10 --iterations 50 \
  --json-out "${REPO_ROOT}/results/cp8_bf16_medium_projection_final.json"

"${REPO_ROOT}/build/fuse_bench" \
  --mode a2a_gemm --m 4096 --n 10240 --k 8192 \
  --comm-ctas 10 --raster n --swizzle 1 --warmup 10 --iterations 50 \
  --json-out "${REPO_ROOT}/results/cp8_bf16_large_projection_final.json"

"${REPO_ROOT}/build/fuse_bench" \
  --mode a2a_gemm_fp8 --m 4096 --n 10240 --k 8192 \
  --comm-ctas 8 --raster n --swizzle 1 --warmup 10 --iterations 50 \
  --json-out "${REPO_ROOT}/results/cp8_fp8_a2a_projection_final.json"

"${REPO_ROOT}/build/fuse_bench" \
  --mode gemm_a2a --m 4096 --n 128 --k 4096 \
  --batch 1 --local-heads 8 --comm-ctas 32 \
  --raster m --swizzle 1 --warmup 10 --iterations 50 \
  --json-out "${REPO_ROOT}/results/cp8_bf16_pv_gemm_a2a_final.json"

"${REPO_ROOT}/build/fuse_bench" \
  --mode qkv_gemm_a2a_fp8 --m 4096 --n 10240 --k 8192 \
  --q-heads 64 --kv-heads 8 --head-dim 128 --comm-ctas 40 \
  --raster m --swizzle 1 --warmup 10 --iterations 50 \
  --json-out "${REPO_ROOT}/results/cp8_fp8_qkv_projection_a2a_final.json"
