# Archived benchmark results

The committed result set contains the v4.0 benchmark snapshots for
`A2A -> O-projection GEMM` and `QKV projection -> A2A` on the local 132-SM
H200-class NVLink node.

Measurement contract:

- BF16, CP4 and CP8;
- 10 warmup iterations and 50 measured iterations;
- every sample reports the maximum CUDA-event latency across ranks;
- the table uses p50 latency;
- the v3 snapshot fixes the v2 kernel and keeps `--lhs-policy auto`;
- 29 rows use the v2 automatic `comm_ctas=4/6/8` result, while seven rows
  use a formally measured external `comm_ctas` winner;
- the tile is selected from the maintained fixed CUTLASS candidates by the
  shared runtime cost model, with no per-shape manual tile override.

Committed directories:

- `a2a-Oproj/oproj_cluster_wave_bench`: current v2 fused 36-case summary after the
  cluster-frontier scheduler and narrow-shard 3D TMA store optimizations;
- `a2a-Oproj/oproj_shape_bench` and `a2a-Oproj/oproj_shape_bench_cp8`: selected CP4/CP8 external
  baseline configurations, fused results, and shape definitions;
- `a2a-Oproj/oproj_shape_bench_longseq_cp4` and `a2a-Oproj/oproj_shape_bench_longseq_cp8`: the
  corresponding 128K, 256K, and 512K aggregates, plus fused-only 1M entries;
- `a2a-Oproj/te_userbuffers_shape_bench` and `a2a-Oproj/te_userbuffers_shape_bench_longseq`: the
  selected TE Userbuffers configurations and formal results.
- `a2a-Oproj/oproj_v3_manual_comm_bench`: the complete 36-case v3 snapshot. Its
  `result_source` field distinguishes 29 `v2_auto_inherited` rows from seven
  `manual_comm_ctas` rows; all options other than the external communication
  CTA count remain fixed.
- `a2a-Oproj/oproj_mixed_shape_bench`: the v4.0 96-setting OProj matrix. It
  combines 36 exact v3 Golden rows, 24 same-geometry model labels, and 36
  newly calibrated real-model rows.
- `a2a-Oproj/te_userbuffers_mixed_shape_bench`: the matching adapted TE
  Userbuffers winners for all 96 OProj settings.
- `QKVproj-a2a/qkv_shape_bench`: the v4.0 96-setting QKV matrix, including
  tuned TE/cuBLAS/cuBLASLt+NCCL baselines and the current fused implementation.
- `QKVproj-a2a/te_userbuffers_shape_bench`: the matching adapted TE
  Userbuffers QKV winners.

Each v4 operator directory keeps only `baseline_summary.*`, the final fused
summary, `comparison_summary.*`, and `shape_matrix.*`; each Userbuffers
directory keeps `summary.*`. The thousands of intermediate sweep and
formal-run JSON files are intentionally omitted.

The v2 fused directory intentionally keeps only `fused_summary.json` and
`fused_summary.csv`. The superseded M256 probes, manual wide-N probes, raw
per-case JSON, and unaligned-policy A/B files are not part of the Golden
archive.

The v3 directory is deliberately separate from the v2 automatic-policy
archive. Its 36/36 TE Userbuffers p50 result combines 29 unchanged v2 rows
with seven per-shape communication-CTA winners; it must not be described as
zero-tuning default behavior.

The long-sequence SOTA comparison excludes 1M because no formal external
TE/NCCL or cuBLASLt/NCCL rerun was requested for that length. Fused internal
exact checks passed. A small number of very large external cases use
`--no-check` only for the materialized PyTorch reference, whose single-launch
indexing limit is exceeded; the timed baseline path is unchanged. The scope
and selected parameters are recorded in the aggregate JSON files.

The OProj searches are reproducible with
`benchmarks/a2a+Oproj/oproj_shape_bench.py` and
`benchmarks/a2a+Oproj/te_userbuffers_shape_bench.py`. The QKV equivalents are
in `benchmarks/QKVproj+a2a`. Reproduction writes raw search points to the
caller-selected result directory without adding them to Git.

The original Golden measurement snapshot is tagged
`oproj-a2a-golden-auto-20260825`.
