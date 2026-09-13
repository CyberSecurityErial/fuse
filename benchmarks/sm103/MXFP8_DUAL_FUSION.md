# MXFP8 dual-fusion development (after v22.0)

Mainline boundaries remain QKV GEMM+A2A and OProj A2A+GEMM. Norm/RoPE is an
optional, disabled experimental extension, not part of these benchmarks.
Targets are measured separately: forward QKV/OProj each2.2 PFLOPS/GPU;
corresponding complete backward each2.0 PFLOPS/GPU. None is claimed achieved.

## Baselines and scope

Preserve the legal physical-point catalogs and aliases in
[QKV v20](../../results/sm103/v20.0/README.md) and
[OProj v21](../../results/sm103/v21.0/README.md). They contain33 and36 physical
long-sequence points respectively; aliases must not duplicate weight in means.
Retain the broader catalog's missing/unsupported entries explicitly. These are
existing projection workloads, not proof that every special attention route
is supported. All legal large-model CP4/8 x128K/256K/512K points remain in scope.

Before changing algorithms, replay representative fixed configurations on the
current binary with norm/RoPE disabled, and measure the independent same-budget
GEMM and communication controls. Historical performance is a reference, not a
same-session A/B. Formal timing stays Graph10+50 with two nonzero payloads and
full numerical/routing validation. Primary comparison remains full-device
cuBLASLt; a reduced-SM GEMM is diagnostic, not a lowered acceptance ceiling.

## Reusable historical work

| Commit | Applicable work |
|---|---|
| `a6687ea` (v19) | BF16 real reverse routes, separate B/W APIs and independent backward checks |
| `e6810f4` (v20) | MXFP8 K32 weight quantization, packed conversion, aggregated full-panel publication, QKV service-based communication selector |
| `91904b7` (v21) | OProj A/W warp cohorts, cp.async/TMA mixed transport, grouped SFA loads, shared bounded M/N traversal, explicit measured budgets |
| `ef6d49f` (v22) | Default-off experimental norm/RoPE and audited profiling/measurement tools; not a replacement for the original dual-fusion tables |

Do not infer a new benefit from a commit message: inspect its code and retained
measurements. Preserve GEMM tile/Along/swizzle as inputs to producer ordering;
communication budget and windows must adapt without changing ready semantics.
Pure GEMM tuning and concurrent service tuning are distinct experiments.

### Communication efficiency before reducing its CTA budget

Treat `C = device_SM_count - communication_CTAs` as a compute budget, not a
guaranteed throughput. Measure pure GEMM at that budget and the complete fused
boundary independently. A faster pure GEMM does not establish a faster fusion.
The communication cohort also quantizes W: its useful service includes both
activation/SFA delivery and complete W-panel publication.

The first controlled CP8/128K budget sweep (four geometries, fixed GEMM and
windows) illustrates the constraint:20 communication CTAs allow128 compute
CTAs and2.36--2.43P pure GEMM, but fusion reaches only1.39--1.50P. These are
diagnostic measurements, not new accepted results or a universal scaling law.
Full-device cuBLASLt remains the main comparison.

Investigate per-CTA service efficiency before making a smaller budget the
default. For example, capacity rounding currently gives these payloads:

| Peer K | Four48-KiB slots: payload capacity | Three64-KiB slots: payload capacity |
|---:|---:|---:|
|1024|4 x32 KiB|3 x64 KiB|
|1536|4 x48 KiB|3 x48 KiB|
|2048|4 x32 KiB|3 x64 KiB|

These are geometry-derived capacities, **not measured bandwidth**. The latter
arrangement also assigns five instead of four warps to W, so an experiment
tests the whole packing/cohort arrangement, not an isolated TMA improvement.
It must not become a universal default on the strength of a power-of-two K
case:1536 is an explicit counterexample to its capacity advantage. Preserve
the complete M128-peer ready unit and derive all arrival counts from actual
chunking. Reject stale service calibration when producer configuration changes.

## Startup participation experiment (not a new default)

Weight quantization uses CUDA-core work even in an input-side fusion. Increasing
its worker pool temporarily is distinct from reserving more CTAs throughout GEMM.
The explicit OProj `all` experiment assigns communication W warps first, then
all compute warps, one dense worker interval for unique chunk ownership. A warps
start transport immediately. Each compute CTA finishes its W contribution and
joins locally before entering CUTLASS; full-panel release/acquire still gates W
consumption. There is no extra global W-completion join in this variant.

This is an experiment, not a proven win. The preceding global-join control
helped low-communication-budget cases but only one of four CP8/128K geometries
exceeded the original configuration's best throughput. A remains a potential
limiter; quantization acceleration alone does not establish a fused speedup.
The producer-only reference launches the same startup participants and includes
their work in timing. Pure-copy reference has no W source and no extra CTAs.
Default communication-only preparation and its existing automatic policy are
unchanged; uncalibrated all-CTA preparation requires an explicit budget.

After full validation, a next **unmeasured** budget experiment can bracket the
intersection using existing measured copy/compute references: from a tested
budget `c`, estimate `c * T_copy(c) / T_compute(148-c)` and check nearby budgets.
This is only a way to prioritize offline candidates. It assumes inverse copy
scaling and ignores changed compute speed, startup and concurrent contention;
it is not a runtime selector or a predicted fused latency. Measure those effects
and keep the original budget as a control before accepting a smaller one.
For the current CP8/128K all-CTA measurements at20 communication CTAs, the copy
to compute time ratios are about1.32 (Qwen geometry),0.63 (Kimi),0.72
(representative large),0.42 (BLOOM),0.34 (Llama405). Thus reducing every case to
the same small budget is not justified: large output widths have more arithmetic
per delivered activation byte. These ratios are observations of this setup,
not constants to embed by model name.

## Backward implementation boundary

Existing SM103 backward production bindings are BF16. MXFP8 backward is not
implemented merely because the two forward MXFP8 directions exist.

| Projection | B/data-gradient phase | W/weight-gradient phase |
|---|---|---|
| QKV | Inverse planar dQ/dK/dV A2A to packed dY; dX=dY W | dW=dY^T X |
| OProj | dA=dY W, then head-shard A2A | dW=dY^T A |

The existing API measures B and W separately and leaves cross-CP dW reduction
to the caller. A sum of independently timed B/W is not a directly measured
immediate full-backward boundary. New MXFP8 reporting must state exactly which
boundary is timed and preserve any required saved-operand lifetime.

MXFP8 scales describe groups of32 along a specific reduction axis. Transposing
FP8 bytes and reusing their original scales does not generally produce a
valid K32 representation for a transposed GEMM. Before implementing the new
interfaces, explicitly define source/saved representations, gradient and master
weight quantization, transposed scale construction, accumulation dtype and the
independent reference. Quantization/transpose work may not silently disappear
from timing or be replaced by a BF16 result labelled MXFP8.

## Evidence and iteration

Keep one compact current table/configuration/evidence file in
`fuse_midfile/mxfp8-v23`, not new per-trial source scripts. Preserve formal raw
artifacts referenced by run/source/binary hashes. Profile only selected
diagnostic cases using the repository protocol; use ordinary uninstrumented
launches for performance decisions. Existing BF16/SM90 behavior and the
experimental norm/RoPE contract must not be silently changed.
