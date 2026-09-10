# SM103 BF16 backward baseline

The public QKV/OProj backward semantics and BF16 correctness harness are shared
with SM90. This is a real reverse route, not a relabelled forward measurement.

| Phase | Boundary | Stored operands |
|---|---|---|
| QKV B | planar dQ/dK/dV → inverse A2A → packed dY → dX | NN: dY × W |
| OProj B | dX = dY × W → head-shard A2A | NN |
| QKV/OProj W | local partial dW = dYᵀ × saved input | TN |

CP weight-gradient reduction is outside these APIs. B and W are measured
separately; their summed samples are not a directly timed immediate B+W result.
The shared small smoke also checks immediate/deferred execution and beta=1
accumulation. Production baseline uses beta=0, BF16 inputs/output, FP32
accumulation, causal dual-chunk routing, fixed 16 communication CTAs and
128×128×64 GEMM. It does not invoke forward autotune or search backward winners.

QKV uses the SM90 source-push inverse route and a system-scope acquire in the
Blackwell mainloop. Ready still covers the whole `(128-row tile, head)` unit.
OProj reuses the SM103 output communication role with one real head segment,
without synthetic K/V heads. Both use the persistent mixed-role dispatcher.

## Measurement

`backward_smoke` and `backward_mpi_bench` are built in an isolated backward build
directory by `scripts/l20d.py --backward`. The MPI harness is shared source;
SM103-only random-input and full-reference checks are conditional on
`FUSE_ARCH_SM103`. SM90 device code and its default benchmark behavior are not
changed. CUDA 13 Graph API argument adaptation is version-guarded.

- Graph: 10 warmup + 50 max-rank event samples, original samples retained.
- Two deterministic independent hashed-random payloads: every GEMM output is
  compared with cuBLAS in bounded 128-row slabs; every route element is checked
  exactly. Fixed tolerance is `abs(error) <= 0.02 + 0.02 * abs(reference)`.
- Per-rank device startup records and controller GPU memory/clock/power telemetry
  accompany every batch. Sampling drift above 5% is reported as unstable, not
  silently accepted as formal throughput or replaced with a best-of-retries.
- MPI ranks persist across shapes. Per-case collective free-memory checks skip
  insufficient-memory cases before allocation. No resident processes are killed.
- Same geometry and route aliases share one physical measurement. The complete
  catalog retains unsupported segmented routes/nondivisible heads as blank rows.

```bash
python3 scripts/l20d.py run fused-build --backward --mpi --node 09
python3 benchmarks/sm103/backward/fused_baseline.py plan --output <local-output>
python3 benchmarks/sm103/backward/fused_baseline.py run --output <local-output>
python3 benchmarks/sm103/backward/fused_baseline.py report --output <local-output> \
  --pure <backward-pure-table/summary.json>
```

Review implementation in `csrc/operators/sm103/api/backward.cuh`,
`detail/backward.cuh`, and the layout/system-ready specializations in
`detail/{gemm,cutlass_pipeline,gemm_a2a}.cuh`. Numerical validation lives in
`validation.cuh`; catalog and full-table orchestration are in this directory.

TODO: backward GEMM tile tuning and a separately validated production/consumption
performance model. Special KDA/MLA segment replication and adjoint reductions
remain explicit route TODOs rather than ordinary-A2A approximations.
