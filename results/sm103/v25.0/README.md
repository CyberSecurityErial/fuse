# v25.0 — SM103 BF16 Dispatch + Grouped GEMM

User-accepted **Dispatch -> expert FC1 forward baseline** on the B300/SM103
development environment. BF16 operands/output, FP32 accumulation. Full routing
and grouped GEMM are included; the pure-GEMM reference does **not** communicate.

[Full tables](table.md) · [CSV](table.csv) · [Exact parameters and evidence](results.json) ·
[API, build and module guide](../../../benchmarks/sm103/GROUPED_GEMM_STUDY.md)

## Results and measurement versions

17 unique physical geometries (21 model labels), EP4/8 and target expert row
counts192/256/384/512/1024/2048/4096/8192. M is expert token count, not sequence
length. Aliases do not add statistical weight.

| Measurement cohort | Valid / requested | PFLOPS/GPU geometric mean | Pure-GEMM retention | Historical fusion improvement |
|---|---:|---:|---:|---:|
| EP4, release-code equivalent | 128 /136 | 0.904028 | 68.5577% | +38.6095% |
| EP8, preceding complete transport snapshot | 134 /136 | 0.822435 | 60.5107% | +40.5067% |

EP4 has8 memory skips. EP8 has2 memory skips and2 measured points without a
reference/before value; those2 are excluded only from the ratios, not hidden.
The reference is the stronger valid finite-candidate CUTLASS/DeepGEMM grouped
GEMM at148SM. Only the winner/configuration is retained per point; this is not
an exhaustive global-optimum claim. Historical improvement is not a fresh
full-matrix same-round A/B; rows identify the prior measurement and any tail-off
fallback explicitly.

**Do not combine these cohorts into a current-version average.** The last complete
EP8 snapshot predates cooperative row-address prefetch. This additional change
was fully measured in EP4; EP8 latest-code remeasurement remains incomplete
because of shared-GPU memory availability. Published source/case/archive hashes
belong to the measured snapshots, not retrospectively to the tag.

The user accepted the current baseline. Original per-point15% improvement and
60% retention goals are **not universally achieved**. No production CTA/GEMM
Auto, arbitrary-router performance guarantee or end-to-end model-training
acceptance is claimed. The workload is synthetic routing at pinned model shapes.

## Included functionality and effective optimizations

- Private Grouped CTASP implementation, separate from Projection. Device-resident
  variable expert counts/routes, empty experts and row tails; capture-safe plans.
- Eight warp leaders use24KiB staging slots for scattered whole-row bulk reads
  and contiguous local bulk writes, within the original GEMM shared-memory budget.
- Derive batch size from actual row count and row bytes; resolve route/peer
  addresses cooperatively before issuing copies. Vector LD/ST fallback batches
  eight independent16-byte loads per thread.
- Optional incomplete-tail sharing balances eight-row stripes while retaining
  one ready per original128-row/full-K delivery and the original acquire protocol.
- Full receive buffers by default. Bounded-buffer recycling is an explicit,
  potentially much slower memory-pressure fallback, not a throughput default.
- Preserve full destination-write completion, proxy ordering, epoch advancement,
  numerical tolerances and cross-rank completion.
- Keep only accepted paths; remove compile-disabled cp.async/two-stage transport.
  Existing Combine and swapAB development paths are retained unchanged but are
  **outside this release's feature/performance acceptance**. Projection/SM90
  algorithms are unchanged.

## Validation and provenance

Each formal result uses two reproducible nonzero payloads, each Graph10+50,
independent complete before/after numerical, routing and untouched-tail checks.
Each sample uses max-rank latency; the two payload p50s are averaged.
Original stability checks and first-stable-round selection remain.

Release cleanup build: `20260917-015023-1d285a`. All72 native build inputs match
the published implementation. The4 fused kernel instruction streams and machine
encodings exactly match the measured row-address-prefetch build
`20260917-003733-a018d7`; cleanup is not counted as a speedup.
EP4 dynamic Graph/bounded-buffer checks for both Along directions:
`20260917-004615-22b344`, `20260917-004627-a69b5a`.
Those do not establish latest-code EP8 coverage.

Local checks:738 SM103 tests exercised (5 dependency skips); the added benchmark
driver-link expectation was corrected and its test rerun successfully.
122 controller tests and7 shape/routing tests passed. Host tests are not GPU
execution claims. Raw measurement archives remain in the existing external
experiment storage, identified by SHA256/run IDs in results.json.

No profiling traces, dependencies, raw sample arrays or tuning histories are
release assets. `PROFILE_PROTOCOL.md` defines optional diagnostic usage and its
interpretation limits.
