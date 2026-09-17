# SM103 BF16 Dispatch + Grouped GEMM — v25.0

User-accepted baseline release. This version publishes **Dispatch -> expert FC1**
only. The existing Combine baseline and swapAB experiments remain development
code; neither is part of v25's feature/performance acceptance. Projection and
SM90 kernels are unchanged.

[Release results, configurations and coverage](../../results/sm103/v25.0/README.md)

## Operator boundary

External routing supplies source (rank, token, branch slot) for each expert row.
The fused kernel gathers/A2A-transfers those rows and computes
`D_e[M_e,2F] = A_e[M_e,H] @ W_e[2F,H]^T`.
Inputs, weights and outputs are BF16; accumulation is FP32; alpha=1, beta=0.
Gate/up are packed into one FC1 GEMM. Router decisions, activation, shared experts,
top-k weighting/reduction, ETP and backward are outside this boundary.

Actual per-expert counts are device metadata, not padded capacity. Empty experts
and partial tiles are supported; useful FLOPs count actual rows only.
The benchmark is a reproducible synthetic routing workload at model-derived
geometries, not a recorded training router trace or end-to-end model validation.
[Matrices, model configuration sources and routing generator](grouped_shapes.py).

## Architecture and delivery contract

### Frozen catalog provenance

These are pinned benchmark geometries, not claims of full model execution.
H is hidden width, F is routed expert intermediate width, E excludes shared
experts. Equal physical tuples share a measurement; aliases do not add weight.

| Model / pinned configuration | H | F | E | top-k |
|---|---:|---:|---:|---:|
| [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/config.json) | 5120 | 2304 | 384 | 6 |
| [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/config.json) | 4096 | 2048 | 256 | 6 |
| [DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/config.json) | 7168 | 3072 | 384 | 6 |
| [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/a7e62ac04ecb2c0a54d736dc46601c5606cf10a6/config.json) | 7168 | 2048 | 256 | 8 |
| [GLM-5](https://huggingface.co/zai-org/GLM-5/blob/c183ef8c61faee82855eca1ed9bb3a9a7ce3b0b2/config.json) | 6144 | 2048 | 256 | 8 |
| [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2/blob/cf457fa734ab149ffef225f80893eb38c6ff5cdc/config.json) | 6144 | 2048 | 256 | 8 |
| [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3/blob/aca966e4e02791568aa6a4ced368624b3d897f42/config.json) | 6144 | 2048 | 256 | 8 |
| [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/config.json) | 4096 | 2048 | 288 | 8 |
| [GLM-4.7](https://huggingface.co/zai-org/GLM-4.7/blob/602d01efcdd332c5238ca4bcede555defbe83eb7/config.json) | 5120 | 1536 | 160 | 8 |
| [GLM-4.5](https://huggingface.co/zai-org/GLM-4.5/blob/cbb2c7cfb52fa128a9660cb1a7a78e017899e115/config.json) | 5120 | 1536 | 160 | 8 |
| [GLM-4.5-Air](https://huggingface.co/zai-org/GLM-4.5-Air/blob/a24ceef6ce4f3536971efe9b778bdaa1bab18daa/config.json) | 4096 | 1408 | 128 | 8 |
| [GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash/blob/7dd20894a642a0aa287e9827cb1a1f7f91386b67/config.json) | 2048 | 1536 | 64 | 4 |
| [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/8472618112abcbd45acbcdc58436aff4233c23f7/config.json) | 4096 | 1024 | 512 | 10 |
| [Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B/blob/dc4d348443bc740c68e2d77492492c11606384d5/config.json) | 3072 | 1024 | 256 | 8 |
| [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/8efa61729e24bd65b1d152b5ab5409052aa80e65/config.json) | 4096 | 1536 | 128 | 8 |
| [Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5/blob/4d01dfe0332d63057c186e0b262165819efb6611/config.json) | 7168 | 2048 | 384 | 8 |
| [MiMo-V2-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash/blob/1afd314a2406c282e0956375c34a676501c78649/config.json) | 4096 | 2048 | 256 | 8 |
| [MiMo-V2.5](https://huggingface.co/XiaomiMiMo/MiMo-V2.5/blob/63651580ca774f8504f676040460aed3e1244ac1/config.json) | 4096 | 2048 | 256 | 8 |
| [MiMo-V2.5-Pro](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro/blob/21d1ecfecd7bd70f31be25ca49d7edd21f003659/config.json) | 6144 | 2048 | 384 | 8 |
| [Mixtral-8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/fc7ac94680e38d7348cfa806e51218e6273104b0/config.json) | 4096 | 14336 | 8 | 2 |
| [Mixtral-8x22B](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1/blob/e1cd34ff1747406fb2277635ed25242803009bc2/config.json) | 6144 | 16384 | 8 | 2 |

### Pipeline

Persistent CTASP separates communication and compute CTA pools; CUTLASS retains
its grouped scheduler, SM100 UMMA/TMEM collective and internal warp specialization.

```text
source GPU tokens (scattered rows)
          |
          v
communication CTA: gather complete rows through K
  warp0: G2S -> local S2G -> full completion
  warp1:   G2S -> local S2G -> full completion
          |
    CTA join / last shared-tail arrival
          |
    ONE ready for [expert, up to128 rows, full K]
          |
compute CTA: acquire -> all dependent N tiles -> BF16 output
```

No within-panel K streaming or extra consumer ready checks. The destination write
must finish before ready publication. Async-proxy fences, source-buffer reuse,
destination drain and cross-rank epoch completion are separate obligations.

Retained optimizations:

- Eight independent warp leaders stage complete scattered rows through
  24KiB slots, bounded by the existing GEMM SMEM allocation. No additional
  communication CTAs or reduction in GEMM stages.
- Batch size is derived from row bytes and actual valid rows. Short stripes
  spread work over available leaders instead of filling just one large slot.
- Resolve row routes/peer addresses cooperatively before leaders issue bulk
  copies. Only addresses are prefetched; no later delivery block jumps ahead.
- Vector LD/ST fallback keeps eight independent16-byte loads per thread for
  small experts or rows that do not fit staging.
- Optional tail balancing shares only the final incomplete communication batch
  in eight-row stripes; the last contributor publishes the original full-K ready.
  Ordinary complete batches keep their original ownership and avoid atomics.

Default receive storage is full-sized. `dispatch_buffer_rows` optionally reuses
a bounded number of rows per expert, waiting until every dependent N tile has
consumed the old contents. **Memory-pressure fallback only:** it can be much
slower; no automatic OOM retry or silently changed shape.

## API and lifetime

- [Public plan/parameters](../../include/fuse/operators/primitives/grouped_gemm.h).
- [MoE semantic wrapper](../../include/fuse/operators/semantics/moe/grouped.h).
- `create_bf16_a2a_grouped_gemm` allocates metadata/workspace outside capture.
- `launch_bf16_grouped_gemm` is capture-safe and reads current GPU counts/routes;
  it allocates nothing and performs no host count readback.
- Keep pointers/capacities/configuration fixed during plan lifetime; contents,
  counts and routes may change between Graph replays.
- All ranks participate once per collective on serial streams. Initialize
  dedicated peer epoch arrays to zero and retain them until all uses finish.
  Recreate all plans/state before 32-bit epoch wraparound.
- Caller enables peer access and supplies valid, nonduplicated branch routes.

GEMM N128/256, K64/128, Along, swizzle and positive communication/compute budgets
are explicit. **No production CTA/GEMM autotune is claimed; zero is not Auto.**
The current automatic transport batching uses work/resource bounds, not model
names or timing searches. Future selection must remain robust across token loads:
small loads emphasize latency, large loads protect compute throughput; use actual
router counts, not allocation capacity. Communication WASP remains deferred.

## Module ownership

```text
include/fuse/operators/primitives/grouped_gemm.h   plan ABI
include/fuse/operators/semantics/moe/grouped.h     upper-level MoE semantics
csrc/operators/sm103/grouped_entry.cu              independent translation unit
csrc/operators/sm103/api/grouped.cuh               validation and plan lifecycle
csrc/operators/sm103/detail/grouped/
  gemm.cuh                                       CUTLASS collective configurations
  persistent_gemm.cuh                             grouped scheduling and CTA roles
  cutlass_pipeline.cuh                            acquire/publication adapters
  producer_consumer.cuh                           tile order and bounded windows
  a2a_gemm.cuh                                    Dispatch gather and delivery
  gemm_a2a.cuh                                    experimental Combine baseline
  communication.cuh                              private peer epoch completion
include/fuse/profiling/grouped.cuh                 diagnostic records
```

Grouped owns its private adaptation; it does not reuse Projection's hot-path
module. CUTLASS and low-level architecture primitives are shared dependencies.
No general transport framework or additional public precision-switch family.

## Build and measurement

CUDA13, a Blackwell-capable CUTLASS checkout, SM103a and peer-accessible GPUs:

```bash
cmake -S . -B build/sm103-grouped -DFUSE_ARCH=sm103 \
  -DCUTLASS_ROOT=/path/to/cutlass -DFUSE_BUILD_KERNELS=OFF \
  -DFUSE_BUILD_BASELINES=OFF -DFUSE_BUILD_GROUPED_KERNELS=ON
cmake --build build/sm103-grouped --target grouped_bf16
```

Example syntax (replace geometry/policy from the release row; CLI flag order is
significant). Local CUDA visibility selects the EP4 devices:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 build/sm103-grouped/grouped_bf16 \
  --case 4 4096 1408 128 8 192 samples.csv \
  --policy 128,64,0,8,20,128 --direction dispatch --tail-balance --fused-only
```

Formal timing uses Graph10+50 for each of two reproducible nonzero payloads,
independent complete numerical/route/tail validation before and after, and
per-sample maximum rank latency. Report the mean of the two payload p50s.
Original stability checks remain; do not select the fastest retry.

Strong reference is the stronger valid bounded-search CUTLASS/DeepGEMM pure
grouped GEMM at148SM, **without communication**. Only its winner is retained in
the release table. The native DeepGEMM adapter is benchmark-only and pinned to
78b69000794d0937b47ae3387eff7663410264d1; it is not a claim about every library's
best possible tuning or Python-default heuristic. Same-budget own GEMM and
transport-only timings are diagnostics, not replacement denominators.

Profiler rules are in [PROFILE_PROTOCOL.md](../../PROFILE_PROTOCOL.md).
All stamps use the local GPU global timer. Ready-poll duration is load-warp time,
not Tensor Core idle or directly recoverable E2E loss. Detailed per-panel traces
require tail sharing off; lightweight role/ready summaries support it.

## Coverage and release limits

The release table separates latest-code-equivalent EP4 measurements from the
earlier complete EP8 transport snapshot. No old/new fastest-per-point mixing;
source, case index and archive hashes identify every result. Missing memory and
reference entries stay explicit. Do not infer a latest-code EP8 remeasurement.

The user accepted this baseline despite the original per-point 15% gain / 60%
pure-GEMM-retention target not being universally met. The release does not claim
that target, arbitrary-routing performance guarantees, production Auto, Combine,
swapAB, backward or whole-model training acceptance. Rejected tuning history,
raw logs, profiles and dependency bundles are not release assets.
