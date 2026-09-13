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

This remains an explicit experiment, not an automatic policy. The preceding global-join control
helped low-communication-budget cases but only one of four CP8/128K geometries
exceeded the original configuration's best throughput. A remains a potential
limiter; quantization acceleration alone does not establish a fused speedup.
The producer-only reference launches the same startup participants and includes
their work in timing. Pure-copy reference has no W source and no extra CTAs.
Default communication-only preparation and its existing automatic policy are
unchanged; uncalibrated all-CTA preparation requires an explicit budget.

After full validation, a budget experiment can bracket the
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

The compute-assisted variant passed all36 physical points. The finite-candidate
geometric mean improved from1.9125 to2.0263P against the old configuration
replayed on the same binary. These winners have not received independent final
confirmation; this is neither an Auto result nor the2.2P target.

The subsequent22-candidate budget bracket across six CP8/128K geometries did
not generally justify smaller budgets. Llama405 illustrates why:

| Communication CTAs | Independent compute P | Fused P | A-end range across ranks, diagnostic us | Per-rank median CTA A wait, diagnostic us |
|---:|---:|---:|---:|---:|
|20|2.3697|2.0272|1189--1277|298--374|
|8|2.4285|1.8811|2540--2677|911--1004|

Formal F/C values are Graph10+50 from run115611-a0f3d2. Diagnostic captures
120406-6b6d03 and120612-a8fcec use the separately instrumented same source,
Eager10+1. W-ready waits remain small (about9 versus19us per-CTA median).
These waits overlap other warps and asynchronous MMA; do not add them or subtract
them from formal latency to claim an exact critical-path attribution.

This motivates increasing A/SFA service per communication CTA before further
reducing its count. Balanced contiguous SFA slices now distribute a complete
M128-peer shard over its existing A chunks, without changing the ready unit.
Independent confirmation of all36 frozen configurations on the alignment-hardened
implementation gives2.084644P (previous compute-assisted2.026349P, +2.8769%).
This is the current accepted finite-candidate baseline, not Auto or the2.2P target.

An unaccepted W-to-A handoff experiment uses six32-KiB or eight24-KiB private
slots in the same192KiB allocation. Copy alone accelerates, but three of four
representative fused cases regress. The paired Llama405 CP8/128K/C20 profiles
show nearly unchanged startup and W waiting, but larger A waiting: late W warps
own chunks of the first GEMM wave even before they can begin copying. A faster
final A completion is therefore insufficient. The next controlled variant
reserves the exact initial compute-wave M prefix for immediate A warps and
hands only the suffix to all active A warps. The prefix follows the resolved
GEMM order/budget, not a model-specific latency guess. It added no completion
barrier or ready granularity. This restored Llama near its accepted baseline
and improved representative-large to2.00835P at20 communication CTAs, but
Qwen/Kimi still regressed; it was not adopted.

The current experiment instead starts all A workers immediately. With
compute-assisted W production, use six32-KiB A slots and two W warps only when
row rounding gives more resident A payload than four48-KiB slots; otherwise
keep four A/four W. The dense W worker interval is recomputed from this actual
cohort plus the compute startup workers. No late handoff or protected-prefix
queue remains. This trades a small fraction of the augmented W pool for more
immediately usable A workers, not additional communication CTAs. All36 physical
points pass the original Graph10+50/F/C/R/P/two-payload checks. Finite-candidate
GM is2.120467P (+1.7184% versus the fixed2.084644P baseline);15 points select
fewer communication CTAs. Four observed regressions are0.11--0.50%; they remain
in the result set. This is not independent frozen confirmation or the2.2P goal.

### Revisit the existing GEMM window after changing W startup

The optional H/P window changes the GEMM traversal itself, not just A delivery.
Increasing P reuses an A M-tile over more N tiles, reducing the distinct M rows
needed by the initial compute wave, but requiring more distinct W panels.
For128K/CP8 Llama geometry with140 compute CTAs, H64 and swizzle8, host execution
of the actual mapping gives P4/8/16 initial A demands of80/48/32MiB, respectively;
the corresponding W demands are16/32/64MiB (FP8 payloads, scales excluded).
These are data requirements, not measured transfer times or a speed prediction.

Compute-assisted W startup changes the tradeoff that motivated a small P.
The four representative P8/P16 tests improve each best F result; some pure C
results also improve, so this is not solely a communication optimization.
Pure C references intentionally use the same H/P mapping as F. The ordinary
independent GEMM search does not set that optional window; an additional H=P=0
control tests its original traversal without changing the collective or ready
unit. No new runtime selector is inferred from these offline candidates.

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
