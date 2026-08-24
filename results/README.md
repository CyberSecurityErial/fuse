# Archived benchmark results

The committed result set is the Golden dataset for the fused
`A2A -> O-projection GEMM` operator on the local 132-SM H200-class NVLink
node.

Measurement contract:

- BF16, CP4 and CP8;
- 10 warmup iterations and 50 measured iterations;
- every sample reports the maximum CUDA-event latency across ranks;
- the table uses p50 latency;
- the fused implementation uses `--lhs-policy auto --comm-ctas 0`;
- the production runtime selected `comm_ctas=4` for all 18 cases;
- the tile is selected from the maintained fixed CUTLASS candidates by the
  shared runtime cost model, with no per-shape manual tile override.

Committed directories:

- `oproj_shape_bench`: CP4 formal fused results, selected formal external
  baselines, summaries, and shape definitions;
- `oproj_shape_bench_cp8`: the corresponding CP8 dataset.

The exhaustive NCCL and execution-mode search points are committed alongside
the formal reruns, so the selected baseline can be independently audited.
They are reproducible with `benchmarks/oproj_shape_bench.py`.

The code snapshot is tagged `oproj-a2a-golden-auto-20260825`.
