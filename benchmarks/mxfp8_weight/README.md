# MXFP8 weight forward baseline: SM90 matrix contract

For the parked development branch, retained results and known limitations, see
[SNAPSHOT.md](SNAPSHOT.md).

For the new branch's production 2F2B implementation, portable published tables
and combined reproduction driver, start with [BENCHMARK.md](BENCHMARK.md).
This document describes the standalone forward baseline that produced the
historical snapshot; statements about not editing production refer to that run.

This document and `full_v2/` cover the **two forward operators only**. The separate
[backward contract](BACKWARD.md) covers QKV/OProj backward with FP32 main_grad,
as explicitly selected for the new benchmark. Forward completion must not be
reported as four-operator completion.

The separate backward suite is now complete and jointly audited with this
forward suite. See [the four-operator report](../../results/mxfp8_weight/backward_full_v1/FOUR_OPERATORS.md)
for the complete 2304-row scope and the explicitly different FP32 backward
main_grad semantics.

This is a benchmark, not a production Fuse operator. First validate one case;
the full deliverable is **192 settings / 768 best-tested timing rows**. A pilot
result is never labelled full coverage.

## Exact coverage

`matrix.py` imports the original shape registries instead of copying shapes:

| Dimension | Contract |
|---|---|
| Gemm→A2A | All 8 models in `../QKVproj+a2a/qkv_shape_bench.py` |
| A2A→Gemm | All 8 models in `../a2a+Oproj/oproj_shape_bench.py` |
| Global sequence | 1024, 4096, 16384, 131072, 262144, 524288 |
| Parallelism | CP4 and CP8, **not TP** |
| Flattened token count / local GEMM M | global sequence / global sequence divided by CP |
| CP4 physical GPUs | 0,2,4,5 |
| CP8 physical GPUs | 0,1,2,3,4,5,6,7 |
| Backends | cuBLASLt+NCCL; repository-style TE Userbuffers P2P+cuBLASLt |
| Execution | Eager and single-operation CUDA Graph replay |
| Formal timing | 10 warmups, 50 samples, sample-wise maximum across ranks |

The two model lists have different geometries: similarly named synthetic models
must not be substituted across directions. Hidden size, Q/KV head counts, head
dimension, M/N/K, CP, GPU group and output layout are attached to every result.
No max-context clipping, sequence substitutions, or smaller stress-case shapes.
Missing/failed settings must remain visible; no silent reduction of the denominator.

## Only the weight representation changes

Offline BF16 weight → E4M3 payload with one unsigned E8M0 scale per 32 K-elements.
Scale is `2**ceil(log2(amax/448))`, finite benchmark inputs, RNE payload conversion.
Every measured invocation dequantizes to BF16 and uses BF16 GEMM, BF16 A2A and
BF16 output. No activation quantization or communication compression. Offline
quantization is excluded, runtime dequantization is included. This standalone
baseline materializes a BF16 weight workspace; tile-fused dequantization is future
operator work. It is not native MXFP8 Tensor Core execution.

Gemm→A2A keeps the original planar Q/K/V, rank-major sequence output. A2A→Gemm
uses causal-paired sequence routing and one full-K GEMM, without BF16 partial-sum
accumulation. Correctness uses independently dequantized weights; quantization
error against the pre-quantization weights is reported separately.

## Execution efficiency and evidence

One distributed process group runs many shapes: do not launch once per sample
or candidate. Buffers, timing events, cuBLASLt planning, JIT and graph setup are
outside timed regions. CPU barriers align samples; rank timing vectors are
gathered once per measurement and reduced sample-wise afterwards. This preserves
the old rank-max statistic, but the host synchronization harness differs; eager
numbers can include host launch gaps and old timings should not be assumed to
be a same-run A/B. Graph timing removes most such launch overhead.

Results retain raw per-rank samples, p50/p95, algorithm metadata, UB parameters,
hardware identity and source hashes. Output is checkpointed after each setting.
The `gemm_plan` metadata is from the JSON-writing rank (rank 0); plans are locally
autotuned on each rank. Timing vectors and reduced correctness gates cover all
ranks, not just rank 0.
The v2 implementation tunes cuBLASLt using the existing up-to-64-heuristic C++
runner and measures the following finite candidate set. "Best" means an
independently remeasured winner within this declared set, not a global optimum.

| Backend | Short-sweep candidates |
|---|---|
| NCCL | Auto; 24 channels / 128 KiB chunks / 64 KiB LL threshold; 32 channels / 512 KiB chunks / 64 KiB LL threshold |
| TEUB, both directions | Full GEMM path, communication SM counts 4/8/16 |
| TEUB, Gemm→A2A | Destination-slab GEMM followed immediately by send, SM counts 4/8/16, math-SM target 132 minus communication SMs |
| TEUB, A2A→Gemm | 2/4 row chunks, receive later chunks concurrently with full-K GEMM on earlier chunks, communication SM count 8 |
| Legacy policy seed | Re-test the matching old per-shape TEUB winning communication/packing configuration; old timing values are never reused |

The QKV peer-GEMM path stores the offline payload and scales in destination-major
row order, matching the old TEUB weight layout. Its complete weight dequantization
still executes on every measured invocation. The OProj row pipeline preserves
full-K accumulation per output row. The old OProj partial-K BF16 accumulation path
is deliberately not treated as equivalent arithmetic.

Each TEUB candidate gets 3 warmups / 12 samples in Eager and Graph. Select one
winner per mode, then run fresh 10/50 formal samples. NCCL's three settings run
in separate processes so environment caching cannot invalidate the sweep; each
process handles all shapes for its CP. Select per-shape/per-mode winners from
3/12 samples, then regroup shapes by winner for separate 10/50 runs. cuBLASLt
plans are reused across matching candidates within a shape. No production
operator is edited. Reference GEMM computes only the receiver's needed QKV
columns, and error reduction uses bounded scratch memory even for 524288 tokens.

Correctness includes poisoning of every staging/output buffer after capture and
warmup. Validation-only cross-rank synchronization prevents a peer's valid write
from racing another rank's poison operation. Non-finite failures are reduced
explicitly (floating-point MAX alone can hide NaN). `test_correctness_gate.py`
injects NaN/Inf/finite corruption on only one rank and requires all ranks to fail.

## Commands

Use the installed TE-enabled environment (the TE extension needs the repository's
public Userbuffers P2P additions). Build the unchanged C++ runner:

```bash
mkdir -p build-mxfp8-bench
/usr/local/cuda/bin/nvcc -shared -Xcompiler -fPIC -O3 -lineinfo \
  csrc/baselines/cublaslt_runner.cu -lcublasLt -lcudart \
  -o build-mxfp8-bench/libfuse_cublaslt_runner.so

/home/chen/miniforge3/envs/mmunlearner/bin/python benchmarks/mxfp8_weight/test_matrix.py
/home/chen/miniforge3/envs/mmunlearner/bin/python benchmarks/mxfp8_weight/bench.py --list-only

CUDA_VISIBLE_DEVICES=0,2,4,5 OMP_NUM_THREADS=1 \
  /home/chen/miniforge3/envs/mmunlearner/bin/torchrun --standalone --nproc-per-node=4 \
  benchmarks/mxfp8_weight/bench.py --model production_qwen_dense --seqs 4096 \
  --output results/mxfp8_weight/first_cp4.json
```

`--full` selects all 96 settings of the process group's CP. The suite driver runs
the entire sweep, formal selection and coverage audit sequentially:

```bash
/home/chen/miniforge3/envs/mmunlearner/bin/python benchmarks/mxfp8_weight/run_suite.py
```

Outputs live in `results/mxfp8_weight/full_v2/`. `matrix.json` contains all expected
shapes; `nccl_policy_cp*.json` stores selection decisions; `combined_cp*.json`
contains selected formal rows with source-report hashes; `coverage.json` and
`coverage.csv` are the final audit/table. Raw candidate outputs and per-rank sample
vectors remain available. Run the same command to resume after a **confirmed
terminal** job. Resume refuses changed source/protocol/device fingerprints.
Do not start a second runner while the first is live. The suite never runs CP4
and CP8 simultaneously. Together they cover 192 settings, not 192 per CP.

After completion, render the full 192-setting side-by-side table (this reruns the
strict 768-row audit first):

```bash
/home/chen/miniforge3/envs/mmunlearner/bin/python benchmarks/mxfp8_weight/report.py \
  results/mxfp8_weight/full_v2
```

This creates `REPORT.md` and `statistics.json`; speedups compare TEUB versus
cuBLASLt+NCCL under the same MXFP8-weight semantics, not quantized versus BF16
weights and not a production Fuse speedup.

Audit formal outputs against all 768 expected keys (also emits a CSV). Omit
`--require-full` to inspect partial progress without requiring completion:

```bash
/home/chen/miniforge3/envs/mmunlearner/bin/python benchmarks/mxfp8_weight/audit.py \
  results/mxfp8_weight/first_cp4.json results/mxfp8_weight/first_cp8.json \
  --output results/mxfp8_weight/coverage.json
```

The initial 2026-09-06 pilot collected the production_qwen_dense / global S4096
case in both directions and both CP groups: **4/192 settings, 16/768 timing rows**.
All tested configurations passed independent BF16 output checks, including
post-capture poisoning of workspaces. These are 10/50 samples, but the TEUB
best-tested values only select among communication SM counts 4/8/16 for the
initial full-GEMM path. They are not the final optimal/full-matrix benchmark.
