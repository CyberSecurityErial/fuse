# MXFP8 dual-fusion development (after v22.0)

Mainline boundaries remain QKV GEMM+A2A and OProj A2A+GEMM. Norm/RoPE is an
optional, disabled experimental extension, not part of these benchmarks.
Targets are measured separately: forward QKV/OProj each2.2 PFLOPS/GPU;
corresponding complete backward each2.0 PFLOPS/GPU. OProj forward is independently
confirmed at2.256462P across all36 physical points with explicit offline plans.
The other three targets remain unachieved; this is not a new Auto or release.

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

## QKV startup controls after the OProj confirmation

On the same frozen native source as the OProj confirmation, six128K QKV
representatives passed36 candidates and an independent90-boundary F/C/R/Q
audit (Graph10+50, original two-payload validation). The existing explicit
`all` mode quantizes the whole weight with all CTAs, then joins the grid;
it is not the OProj compute-assisted schedule. Its best tested throughput
versus ordinary `comm` was:

| Projection geometry | comm P | all P |
|---|---:|---:|
| QwenDense, CP8 |1.1681|1.1916|
| Qwen3 235B, CP4 |1.8102|1.9929|
| Qwen72/Llama70, CP8 |1.8698|2.0506|
| Llama405, CP8 |1.9123|2.1986|
| Kimi QKV-only, CP8 |1.8874|2.0411|
| BLOOM, CP8 |1.9732|2.2127|

These are finite candidate controls, not the full33-point result or Auto.
C/R/Q are measured only in `comm` mode; the `all` controls measure F only.
QwenDense's independent R at20/72 CTAs is226/218us, already close to its
fused duration. R includes cross-rank completion, so it does not establish
raw TMA latency or link bandwidth. Inspect local route service and finalize
separately before changing the transport. No QKV device changes are accepted
from this comparison alone.

The same-source Dense service probes subsequently measured GPU0 local R
(before cross-rank finalize) at181.152us with20 CTAs and178.080us with72.
Prepared compute moved from114.976us to181.056us as its budget shrank.
These are individual diagnostic captures, not independent repeated timings.
The complete `all`,20-CTA trace ended compute at145.472us and route at219.456us
on GPU0, with kernel completion at236.864us. Every rank showed route ending
later than compute. Thus extra communication CTAs alone have little measured
benefit here; inspect per-copy transport/control before consuming more SMs.
This does not identify physical NVLink saturation. The service probe cannot
excite a noninitial W panel for this geometry (all16 panels are in the first
compute wave); unavailable delay services remain explicitly unavailable.
Raw probes: `20260913-154439-92fbdb` and `20260913-154739-2faf80`;
full trace: `20260913-154529-279069`. All original numerical/route checks passed.

Replacing only the local64x128 load with warp-cooperative cp.async did not
help. Dense independent R stayed218/226/401us at72/20/8 CTAs. With whole-W
startup, F was1.191/1.069/0.524P versus original1.192/1.130/0.623P; ordinary
comm also regressed. The M136 tail and all seven candidates passed their
original checks, so the native experiment was reverted for performance, not
relaxed validation. Preserve the source/run evidence, not an unused transport
option. Runs `20260913-155954-79cbcf`, `160010-230625`, `160033-12d5cd`.

The existing rank N-band rotation was also tested in an isolated build. All
eight Dense trace ranks had identical8192 task-to-destination mappings; this
motivates testing correlated traffic, not claiming measured incast. Seven
originally validated candidates across Dense/Llama405/Qwen3 gave best
1.1840/2.2022/1.9842P, versus nonrotated1.1916/2.1986/1.9929P. There is no
material best-candidate gain to adopt. Keep rotation off and do not extend
the existing TMA Auto calibration to it. Runs `20260913-161046-d8d4c9`,
`161110-63f282`, `161133-a292d9`; the temporary MXFP8 experiment gateway was
removed. Neither rejected trial changes the accepted OProj implementation.

The full all-startup control on one restored binary passes all33 physical
points /97 finite candidates and an independent audit. Its selected geometric
mean is2.037745P, not yet independently frozen/confirmed. Dense's six-point
mean is1.461139P; the other families range2.034425--2.301891P. No OOM or
missing point was removed. This remains below the separate QKV2.2P target.

An all-startup `NoQkvInputWork` route control also failed to improve the best
tested configurations materially. It removed inactive producer/progress polling
but reused every acquire, copy and final drain. Seven full-check candidates
passed: Dense best1.1921->1.1895P, Qwen3 CP4 2.0246->2.0208P and Llama405
CP8 2.2347->2.2447P. Dense at20 communication CTAs fell2.49%; this is not a
transport solution. Revert the extra route instantiation instead of retaining
another inactive control. Source66cc5109/build163328-a75407 and runs
`163708-e4948e`, `163731-702be9`, `163747-0277a3` retain the evidence.

Two all-startup transport controls used six/seven route warps, each with two
16KiB stages. Each complete-box store committed one group; wait.read<1> retired
the previous stage before reuse, while the final full remote drain was kept.
Multi-group row tails retained the original single-stage route. Seventeen
candidates, including lifecycle/tail tests, passed the original full checks.
Six warps retained the220160-byte production SMEM floor; seven required229504
after the launch helper's128-byte rounding. The latter passed actual device
capacity checks. A host assertion initially forgot that rounding; the already
successful M128 result was reaudited, not rerun or relabelled a GPU failure.

Neither raised the best result materially. Six/seven-warps' best throughput
was Dense1.1933/1.1943P, Qwen3 2.0131/2.0198P, Llama405 2.2149/2.2231P.
Dense C20 gained2.66/1.80%, but C8 lost10.68/7.08%. The transport/worker-count
tradeoff is not a demonstrated solution. Both implementations, their temporary
tests and all-mode telemetry refusal were reverted; accepted profiling remains
unchanged. Builds164604-6c8b81 and165226-a0ab4c plus current JSON retain evidence.
Before another route rewrite, bracket the intermediate communication budgets:
the earlier all-startup scan used8/20/original comm-mode budget, not a dense
search of the different all-startup service balance.

The restored route's Dense budget bracket passes29 candidates over all six
long-sequence CP4/8 points. C48 beats the same-run old C56 by2.797/4.418% at
CP4 128/256K; the other four selected budgets do not change. The six-point
finite-search mean is1.475347P versus the same-run original-budget1.458036P
(+1.187%). This is not an independently fixed confirmation. The unique QKV
current table applies these six updates to the original27 non-Dense rows and
labels the different build identities explicitly; its descriptive mean is
2.041334P, still short of the goal. No fast repeat replaces a slower control.

Dense512K standalone component checks also pass at C48/CP4 and C64/CP8.
Same-run comm-mode F/C/R/Q times are respectively1.26850/1.13038/1.41437/.04678ms
and.74358/.62560/.82434/.04286ms. Prior all-mode F on the matching binary is
1.29774/.74461ms. Isolated R actually exceeds F: a max(C,R) hard lower-bound
interpretation is contradicted by these observations. Fresh producer data,
traffic timing and concurrent hardware behavior differ, but this measurement
does not isolate their individual effects or prove NVLink saturation.

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
This is the earlier confirmed finite-candidate checkpoint, not Auto or the2.2P target.

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

The CP8/128K unwindowed controls pass all original numerical and routing checks.
Llama405 and BLOOM reach2.264/2.271P with four communication CTAs; Qwen reaches
2.161P with sixteen. Kimi, representative-large and Llama70 instead favor P16
among the tested layouts. These are finite representative candidates, not a
universal layout rule or independently confirmed full-matrix performance.
The remaining30 physical points now compare H64/P16 against H=P=0, retaining
their original tile/raster/swizzle and measuring F/C/R/P separately. Reduced
distinct-M demand can permit a smaller communication budget, but pure GEMM
reuse also changes; neither isolated copy timing nor geometry proves overlap.

Bulk C/P balance remains an approximation. At the same twelve communication
CTAs, the Kimi projection's P8/P16 controls measure pure C at2.434/2.399P but
F at1.990/2.120P. Executing the actual host mapping shows that the first compute
wave needs20/12 distinct A M-tiles (30/18MiB, excluding scales). Its last required
A chunk falls in ideal static copy rounds14/8 of86 total rounds. These counts
assume equal task service and are not measured delays. They expose a positional
factor missing from a total-service-only score; no fitted correction or new
runtime selector is enabled from this diagnostic.

The completed layout/budget search covers all36 physical points and yields a
finite-candidate geometric mean of2.259959P (+6.5783% versus the same-source P4
search). Seven individual points remain below2.2P. All original F/C/R/P and
two-payload checks pass, with no missing or OOM points. Each winner was
frozen for an independent confirmation; the search mean is not the acceptance
result. Existing same-configuration repeats are deduplicated by first valid run,
not by choosing the fastest repeat. The unique current table retains the full
configuration and source/run/artifact provenance.

Independent Graph10+50 confirmation of the frozen36 plans gives2.256462P:
-0.1547% versus search, +8.2421% versus the earlier2.084644P fixed checkpoint.
Every original numerical/routing/F/C/R/P audit passes. Relative to published
v21 OProj history, improvement is17.8498% (current/history minus one).
Current throughput retains83.0603% of the historical full-device cuBLASLt
geometric mean. These historical comparisons are not paired new measurements.
No point is replaced by its faster search observation. The six physical-family
means are2.2172--2.3006P; seven individual points remain below2.2P. The forward
OProj geometric-mean target is met, not a guarantee that each point exceeds it.

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
