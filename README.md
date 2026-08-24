# GQA + Ulysses CP CTA-specialized kernels

## MegaMoE design work

The H200 CTA-specialized MegaMoE follow-up now has an audited
[design blueprint](megamoe/CTA_SPECIALIZED_BLUEPRINT.md), deriving a Hopper role
split and synchronization protocol from the [DeepGEMM SM90/SM100 reference
extraction](megamoe/REFERENCE_EXTRACTION.md). It is ready for implementation;
no MegaMoE CUDA implementation or performance result is claimed yet.

This directory contains BF16 and FP8 Hopper implementations of two operator
families. Each operator is one cooperative, monolithic CUDA launch with fixed
communication and CUTLASS GEMM CTA roles:

- `A2A -> GEMM`: communication CTAs pull peer feature shards into the final local A layout, publish a system-scope epoch, and CUTLASS persistent GEMM CTAs consume ready M tiles.
- `GEMM -> A2A`: CUTLASS persistent GEMM CTAs write normal local D tiles and publish epochs; communication CTAs route ready tiles directly into peer final layouts.

GEMM uses the local CUTLASS 3.x SM90 TMA/WGMMA collective. The project does not contain a custom GEMM mainloop and does not use a two-stream fallback as the fused result.

## Build

Requirements: CUDA 12.8, CMake, Ninja, 8 mutually P2P-accessible SM90 GPUs, and the local CUTLASS checkout. No separate environment is created.

```bash
bash scripts/build.sh
```

Override CUTLASS when needed:

```bash
cmake -S . -B build -G Ninja \
  -DCUTLASS_ROOT=/path/to/cutlass \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8
```

## Correctness and sanitizer

The full smoke matrix covers CP=2/4/8, B=1/2/4, L-batched GEMM, explicit
padded strides, M/N tails, GQA ratios 1/2/4/8, head dimensions
64/80/96/128/256, forward and inverse routes, production Q/K/V three-segment
packing, and eight consecutive epochs without resetting flags. It also covers
peer K shards that are 16-byte aligned but not aligned to CUTLASS TileK, the
vector fallback for partial M tiles, and swizzle-generated dummy tiles.

```bash
bash scripts/run_correctness_8gpu.sh
```

The quick matrix is intended for fast tool runs. The current delivery also
passes both tools on the full matrix:

```bash
compute-sanitizer --tool memcheck --error-exitcode 99 ./build/fuse_smoke --quick
compute-sanitizer --tool racecheck --error-exitcode 99 ./build/fuse_smoke --quick
```

Full memcheck reports `0 errors`; full racecheck reports `0 hazards`,
`0 errors`, and `0 warnings`.

## Benchmarks

The C++ benchmark reports the maximum rank time for every sample, GEMM-only FLOPS, route payload bandwidth, sequential two-kernel latency, the monolithic kernel, and the task-defined overlap ratio. It also writes structured JSON.

```bash
bash scripts/run_bench_8gpu.sh
```

Equivalent TE/cuBLAS + NCCL baselines use the existing `mmunlearner` environment:

```bash
bash scripts/run_te_nccl_baseline.sh \
  --mode a2a_gemm --gemm-m 4096 --gemm-n 10240 --gemm-k 8192 \
  --batch 1 --warmup 10 --iters 50 \
  --json-out results/cp8_te_nccl_bf16_large_final.json

bash scripts/run_te_nccl_baseline.sh \
  --mode gemm_a2a --gemm-m 4096 --gemm-n 128 --gemm-k 4096 \
  --batch 1 --local-heads 8 --warmup 10 --iters 50 \
  --json-out results/cp8_te_nccl_bf16_pv_final.json

bash scripts/run_te_nccl_baseline.sh \
  --mode a2a_gemm --gemm-m 4096 --gemm-n 10240 --gemm-k 8192 \
  --batch 1 --dtype fp8 --warmup 10 --iters 50 \
  --json-out results/cp8_cublaslt_fp8_a2a_projection_final.json

bash scripts/run_te_nccl_baseline.sh \
  --mode qkv_gemm_a2a --gemm-m 4096 --gemm-n 10240 --gemm-k 8192 \
  --dtype fp8 --q-heads 64 --kv-heads 8 --head-dim 128 \
  --warmup 10 --iters 50 \
  --json-out results/cp8_cublaslt_fp8_qkv_gqa_a2a_final.json
```

The benchmark also supports a real 64K ragged packing contract.  Input mode
uses a packed row map instead of treating the sequences as one dense sequence;
PV mode builds one grouped persistent grid containing the true
`(S_i,128,S_i)` problems:

```bash
lengths=32768,16384,8192,4096,2048,1024,512,256,128,128

./build/fuse_bench --mode packed_a2a_gemm \
  --n 10240 --k 8192 --lengths "$lengths" \
  --packed-rows 16 --comm-ctas 8 --warmup 10 --iterations 50

./build/fuse_bench --mode packed_pv_a2a \
  --lengths "$lengths" --local-heads 8 --head-dim 128 \
  --pv-policy cooperative --comm-ctas 16 --warmup 10 --iterations 50

bash scripts/run_te_nccl_baseline.sh \
  --mode packed_a2a_gemm --gemm-m 8192 --gemm-n 10240 --gemm-k 8192 \
  --lengths "$lengths" --warmup 10 --iters 50

bash scripts/run_te_nccl_baseline.sh \
  --mode packed_pv_a2a --gemm-m 65536 --gemm-n 128 --gemm-k 65536 \
  --local-heads 8 --head-dim 128 --lengths "$lengths" \
  --warmup 10 --iters 50
```

Six highly heterogeneous distributions, CP2/4/8 exact checks, racecheck,
policy sweeps, and the final two-boundary table are collected in
[`PACKED64K_REPORT.md`](PACKED64K_REPORT.md).  Flux is not assigned a number in
that table because its current fused interfaces do not accept a ragged source
map or an equivalent grouped varlen PV problem.

The local Flux 1.1.2 baseline is also driven from this repository and emits the
same maximum-rank JSON statistics.  The first command is exactly shape/layout
equivalent.  Flux has no strided-batched PV operator, so the reverse command is
explicitly labelled a dense analog with equal aggregate FLOPs and A2A payload:

```bash
bash scripts/run_flux_baseline.sh \
  --mode a2a_gemm --gemm-m 2048 --gemm-n 5120 --gemm-k 4096 \
  --head-dim 128 --num-comm-sm 16 --warmup 10 --iters 50 \
  --json-out results/cp8_flux_a2a_gemm_m2048_n5120_k4096_comm16_final.json

bash scripts/run_flux_baseline.sh \
  --mode gemm_a2a_dense_analog --gemm-m 4096 --gemm-n 1024 --gemm-k 4096 \
  --head-dim 128 --num-comm-sm 48 --warmup 10 --iters 50 \
  --json-out results/flux_dense_analog_m4096_n1024_k4096_cp8.json
```

Flux 1.1.2 does not instantiate either Ulysses fused operator for FP8.  The two
exact CP8 capability probes below exercise the real constructors on all eight
ranks and save the identical BF16-only rejection from every rank:

```bash
bash scripts/run_flux_baseline.sh \
  --mode fp8_a2a_gemm_capability \
  --gemm-m 4096 --gemm-n 10240 --gemm-k 8192 --head-dim 128 \
  --num-comm-sm 8 --warmup 0 --iters 1 \
  --json-out results/cp8_flux_fp8_a2a_projection_capability.json

bash scripts/run_flux_baseline.sh \
  --mode fp8_qkv_gemm_a2a_capability \
  --gemm-m 4096 --gemm-n 10240 --gemm-k 8192 --head-dim 128 \
  --num-comm-sm 40 --warmup 0 --iters 1 \
  --json-out results/cp8_flux_fp8_qkv_gemm_a2a_capability.json
```

The QKV three-segment output route has its own benchmark mode; `N` is derived
from `(Hq + 2*Hkv)*D`:

```bash
./build/fuse_bench --mode qkv_gemm_a2a \
  --m 512 --k 8192 --batch 1 --q-heads 64 --kv-heads 8 --head-dim 128 \
  --comm-ctas 24 --warmup 10 --iterations 30 --raster m --swizzle 1 \
  --json-out results/qkv_gemm_a2a_m512_n10240_k8192_cp8.json
```

Latest formal results on the attached 132-SM SM90/NVLink node use 10 warmup
and 50 measured iterations. Every sample is the maximum-rank CUDA-event
critical path; throughput is per GPU.  The task's forward path is
`A2A -> QKV projection -> QK -> PV -> A2A`; consequently the first table below
contains only the two requested fused boundaries.

| required operator boundary / exact shape | fused | fastest pure GEMM | TE or standard GEMM+NCCL | Flux 1.1.2 |
|---|---:|---:|---:|---:|
| BF16 A2A -> projection, CP4 `(2048,5120,4096,1)` | **0.1470 ms / 584.4 T** | cuBLASLt 763.0 T | TE 226.4 T; cuBLAS 330.9 T | 370.5 T |
| BF16 A2A -> projection, CP8 `(2048,5120,4096,1)` | **0.1888 ms / 455.0 T** | cuBLASLt 578.9 T | TE 280.8 T; cuBLAS 302.5 T | 323.6 T |
| BF16 A2A -> projection, CP8 `(4096,10240,8192,1)` | **1.1468 ms / 599.2 T** | cuBLAS 676.7 T | TE 492.9 T; cuBLAS 495.4 T | 551.6 T |
| FP8 A2A -> projection, CP8 `(4096,10240,8192,1)` | **0.6602 ms / 1041.0 T** | cuBLASLt 1216.6 T | cuBLASLt FP8 + NCCL 846.8 T | fused op is BF16-only; FP8 rejected at runtime |
| BF16 batched PV -> A2A, CP8 `(4096,128,4096,8)` | **0.1129 ms / 304.5 T** | CUTLASS 403.9 T | cuBLAS batched PV + NCCL 142.7 T | no equivalent batched PV op |

The following is an additional generic `GEMM -> A2A` capability test, not a
stage in the task's forward path.  It uses dense QKV projection only because
the available cuBLASLt+NCCL FP8 Ulysses reference has this order and gives an
exact, independently implemented comparison for the output-side routing
machinery:

| additional test / exact shape | fused | fastest pure GEMM | exact separated baseline | Flux 1.1.2 |
|---|---:|---:|---:|---:|
| FP8 dense QKV projection -> GQA-pack A2A, CP8 `(4096,10240,8192,1)` | **0.8675 ms / 792.1 T** | CUTLASS 1283.0 T | cuBLASLt FP8 + NCCL 560.5 T | fused op is BF16-only; FP8 rejected at runtime |

The H200 nominal dense roofline is approximately 989.5 TFLOPS for BF16 and
1979 TFLOPS for FP8. The measured single-fast-GPU BF16 target remains
848.3 TFLOPS. See `OPTIMIZATION_LOG.md` for all sweeps, rejected policies, and
invalidated measurements.  “BF16-only” above is a capability result, not a
zero-throughput measurement: Flux pure FP8 GEMM is not substituted for a
missing fused A2A operator because that would duplicate the separated baseline.

## API and layouts

`include/fuse/problem.h` describes GEMM independently from communication:
`(M,N,K,L)`, explicit A/B/D element strides, dtype contract, raster, and
swizzle. The BF16 paths use FP32 accumulation and BF16 output. QKV also has an
E4M3 x E4M3, FP32-accumulate, BF16-output path with explicit epilogue scale.
All paths use row-major A/D and stored `B_nt=[N,K,L]`; batch stride zero is
accepted for broadcast B.

The production CUTLASS policies are separate by shape: BF16 projection uses
`128x256x64`, cooperative mainloop/epilogue, stage 4, and cluster `(2,1,1)`;
BF16 batched PV uses `128x128x64` ping-pong; FP8 QKV uses
`128x128x128` cooperative FastAccum. The API comments in `kernels.h` specify
the exact ready-buffer capacity for each direction.

`include/fuse/route.h` describes CP/GQA routing. Implemented hot paths are:

- input `HEAD_TO_SEQUENCE` for `A2A -> GEMM`, including L-batched input;
- output `HEAD_TO_SEQUENCE/FORWARD`;
- output `SEQUENCE_TO_HEAD/INVERSE`;
- output `QKV_GQA_PACK/FORWARD`, which splits the unequal Q/K/V segments by
  contiguous head ownership and writes the final
  `[B,S_global,(Hq/CP+2*Hkv/CP)*D]` peer layout.

For forward output, local D is `[B,H_local,S_global,D]` and peer final output is `[B,S_local,H_global,D]`. The inverse exchanges `[B,H_global,S_local,D]` into `[B,H_local,S_global,D]`.

Current limitations are explicit:

- Q/K/V head counts must split evenly across CP. Uneven splits return `cudaErrorNotSupported`; K/V are never replicated to imitate MHA.
- Transposed A/B template instances are not yet compiled; non-default transpose flags return unsupported.
- Multi-node, PCIe-only, and pre-SM90 paths are out of scope.

## Synchronization and deadlock proof

Signals live in global memory and use monotonically increasing epochs. Producers publish with system-scope release semantics and consumers poll with system-scope acquire semantics. The GEMM epilogue waits for the stock TMA store before publishing D readiness. Epoch zero is reserved; wraparound requires a quiescent host-side reset.

The kernels use 384 threads. BF16 projection and FP8 use 214016 bytes dynamic
shared memory; BF16 PV uses 205824 bytes. This yields one resident CTA per SM
on the target. Cluster-1 kernels prove cooperative residency with active blocks
per SM. The BF16 projection uses cluster `(2,1,1)` and queries
`cudaOccupancyMaxActiveClusters` with the real block shape and dynamic shared
memory before launch. It rejects the launch unless:

```text
communication_ctas % cluster_x == 0
compute_ctas % cluster_x == 0
total_grid_clusters <= max_active_clusters
```

Communication CTAs occupy a complete-cluster prefix; compute CTAs occupy the
suffix. The x-flattened CUTLASS worker IDs preserve physical cluster rank because
both offsets are cluster aligned. A mixed communication/compute cluster is
therefore impossible. Communication never waits on compute in A2A->GEMM, and
GEMM->A2A communication only waits on an epilogue release from an already
resident compute CTA. Thus the dependency graph has no cycle; a grid that
cannot be resident is rejected before launch rather than silently split into
two kernels.

## Source map

- `csrc/cutlass_kernels_sm90.cu`: both formal kernel families and standalone controls.
- `include/fuse/monolithic_gemm.cuh`: cooperative mixed-role wrapper and residency check.
- `include/fuse/cutlass_scheduler.cuh`: CUTLASS worker offset/flattened persistent stride.
- `include/fuse/cutlass_collectives.cuh`: ready-aware mainloop and signaling epilogue.
- `include/fuse/ready_k_iterator.cuh`: peer/K-group gating around the stock CUTLASS iterator.
- `benchmarks/fuse_smoke.cu`: independent CPU route + cuBLAS numerical reference.
- `benchmarks/te_nccl_baseline.py`: TE/cuBLAS + NCCL reference and baseline.
- `benchmarks/flux_baseline.py`: local Flux equivalent/analog baseline with explicit scope metadata.
