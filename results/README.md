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
- the production runtime selected `comm_ctas=4` for all 36 cases;
- the tile is selected from the maintained fixed CUTLASS candidates by the
  shared runtime cost model, with no per-shape manual tile override.

Committed directories:

- `oproj_shape_bench` and `oproj_shape_bench_cp8`: selected CP4/CP8 external
  baseline configurations, fused results, and shape definitions;
- `oproj_shape_bench_longseq_cp4` and `oproj_shape_bench_longseq_cp8`: the
  corresponding 128K, 256K, and 512K aggregates, plus fused-only 1M entries;
- `te_userbuffers_shape_bench` and `te_userbuffers_shape_bench_longseq`: the
  selected TE Userbuffers configurations and formal results.

Each O-projection directory keeps only `baseline_summary.*`,
`fused_summary.*`, and `shape_matrix.*`; each Userbuffers directory keeps
`summary.*`. The thousands of intermediate sweep and formal-run JSON files
are intentionally omitted.

The long-sequence SOTA comparison excludes 1M because no formal external
TE/NCCL or cuBLASLt/NCCL rerun was requested for that length. Fused internal
exact checks passed. A small number of very large external cases use
`--no-check` only for the materialized PyTorch reference, whose single-launch
indexing limit is exceeded; the timed baseline path is unchanged. The scope
and selected parameters are recorded in the aggregate JSON files.

The NCCL/execution-mode search is reproducible with
`benchmarks/oproj_shape_bench.py`; the TE Userbuffers search is reproducible
with `benchmarks/te_userbuffers_shape_bench.py`. Reproduction writes raw
search points to the caller-selected result directory without adding them to
Git.

The original Golden measurement snapshot is tagged
`oproj-a2a-golden-auto-20260825`.
