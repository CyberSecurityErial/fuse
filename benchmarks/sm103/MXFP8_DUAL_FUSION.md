# MXFP8 dual-fusion development (after v22.0)

Mainline boundaries remain QKV GEMM+A2A and OProj A2A+GEMM. Norm/RoPE is an
optional, disabled experimental extension, not part of these benchmarks.
Targets are measured separately: forward QKV/OProj each2.2 PFLOPS/GPU;
corresponding complete backward each2.0 PFLOPS/GPU. OProj forward is independently
confirmed at2.256462P across all36 physical points with explicit offline plans.
OProj complete backward is also confirmed at2.092341P across36 physical points;
the two QKV targets remain unachieved. The OProj forward milestone is
published separately in [v23.0](../../results/sm103/v23.0/README.md); this is
not a new Auto policy or completion of the remaining development goal.

The current QKV backward fixed33 replay is1.836677P, versus the preceding
finite selection's1.830344P (+0.346%). Its full-head512B scale read now precedes
the existing FP8/BF16 asynchronous copies, with unchanged final scale stores,
system release, communication budgets and GEMM settings. All33 pass both
payloads and original full-boundary checks; Dense's small gains repeat, while
other cases are mostly unchanged (worst observed regression0.84%). This is a
small retained scheduling improvement, not the2P target. Source9e5b124e,
build20260914-064042-a3f88c; the unique current table keeps all33 new results.

The pure cuBLASLt ceiling requires its own two-timed-payload validation,
separate from the native operator comparisons below. Historical records that
timed only payload0 must not be described as two-payload timing results.

The backward harness uses a payload-scoped independent-reference
cache: keep the original K32 decoder, pedantic chunk accumulation, tolerance
and full pre/post comparison for every component. Only repeated reference
generation is skipped while inputs remain immutable. Each new payload rebuilds
both outputs. All ranks fall back to bounded scratch if any lacks cache space
plus2GiB headroom. CPU-oracle smoke now cross-checks fresh/cached references and
injects a poisoned actual value that the cached checker must reject. This
harness change passed CP4/8 CPU-FP64/corruption checks for both routes, small
causal MPI QKV/OProj, and six-component Llama70/Kimi-QKV/Dense controls from
frozen source83d58293 (build20260914-011217-0d2481, nativefe8cd13). Those three
large controls change full throughput by less than0.2%; no operator gain is
attributed to caching. It is separate from the parallel-ready experiment.
That experiment batches ready acquires by the actual pipeline stage count,
then shares visibility with a warp join before the elected TMA issuer fences.
It passes CP4/8 CPU-oracle checks (both routes, Eager/Graph, E32/E64), small
causal MPI validation, and the five full-boundary controls below. It is retained
for the large BLOOM gain and reduced already-ready adapter cost, not as a
claim of full-matrix target attainment.
The warmup watchdog becomes30s after the former5s limit expired on Llama405
CP8/256K, CP8/512K and Kimi CP8/512K. The three per-rank5% convergence windows,
>=100ms GPU warmup,10+50 and first-stable-round rule are unchanged. The new
deadline is logged explicitly; these failed old runs remain failures, not
accepted results. Llama405 CP8/256K run20260914-011836-65c095 still fails
all-rank convergence within30s despite passing pre-numeric/route validation;
extra settling is not a demonstrated fix and no timing is accepted for it.

Parallel-ready sourcee88b416d/build20260914-012438-b46cbe versus cache-only
source83d58293: C20, dX E32/sw8/AlongN, dW E32/sw8/AlongM, Graph10+50,
two payloads and independent full pre/post checks for all six components.

| Controlled case | Previous full P/GPU | Parallel ready P/GPU | Change |
|---|---:|---:|---:|
| Llama70 CP8/128K | 1.781211 | 1.793783 | +0.71% |
| Kimi QKV-only CP8/128K | 1.685295 | 1.697695 | +0.74% |
| Dense CP4/128K | 0.697016 | 0.697720 | +0.10% |
| Qwen3 CP4/128K | 1.236854 | 1.235236 | -0.13% |
| BLOOM CP4/256K | 1.762887 | 2.201077 | +24.86% |

New runs013128-0b61c1/013156-bf13e1/013230-383bc5/013248-a8d233/
013311-20140b on20260914. Kimi already-ready dX drops4.971640→3.993368ms,
but complete B only6.356744→6.307096ms: removing adapter cost does not by
itself remove the input-supply limitation. BLOOM B drops55.062912→38.057784ms.
E32 complete/prepared/bare register counts are112/112/99, with no local spills.
Separately, BLOOM's six-component harness wall time falls434.6→108.9s with
the independent reference cache; this is testing efficiency, not operator speed.

Rejected transport trial aa7ecaa1: replace the two64-row FP8/BF16 cp.async
slices with paired2D TMA loads into the same private stage, normal eviction,
same SFA/full-head release and C20. CPU CP4/8 and all five controls pass, but
full P/GPU is1.770997/1.616185/0.654029/1.215440/2.209439 in the table's order:
four regressions, only negligible BLOOM improvement. Runs014842-4efaf4,
014910-3e6017,014943-52a248,015001-bbeb2c,015025-8b5318 (20260914).
The TMA code/descriptors are removed; this rejects this concrete transfer
organization, not TMA in general. Keep the measured parallel-ready baseline.

The subsequent frozen full-boundary confirmation passes all33 legal physical
points at1.782729P geometric mean, still below the separate2.0P QKV backward
target. Sourcee88b416d, Graph10+50, two timed payloads and original independent
pre/post numeric/route checks; complete B+dW, not a sum of component timings.
Fourteen finite communication-budget trials supplied explicit plans for this
confirmation; dX remains E32/sw8/AlongN and dW E32/sw8/AlongM. The tested
representatives inherit their communication budgets across sequence/CP only
for this validation, not as a model-name-based runtime selector or new Auto.
Compared with the previous source34feb/C20 matrix's30 valid paired points,
geometric-mean speedup is21.886%; all30 improve (minimum6.919%). Three old
warmup failures now pass, but their failed records are not converted to old
performance values. This comparison combines ready and budget improvements;
the controlled table above isolates the ready change. Runs20260914-021307-fb6662
through023129-f1e221 and the current JSON retain every configuration and receipt.

Transpose preparation now groups four independent K32 reductions for coalesced
writeback: two K64 FP8 stages and complete512B scale atoms. The actual compiled
occupancy determines its grid, not a fixed CTA/SM multiplier. The numerical
reductions, input masters, ready protocol, GEMM and communication budgets do
not change. CP4/8 CPU-FP64 and small causal MPI checks pass. The smaller
writeback variant beats the original in three controlled complete-QKV cases:
Dense1.061493→1.103325P, Qwen31.724069→1.793259P, Kimi1.826088→1.866554P.
All six component checks pass; W minus prepared-W is not an isolated quantizer
timer. The cleaned implementation removes unused non-K128 fallback machinery;
compiled registers48, static shared18944B, dynamic shared20480B, no local spill.
Its fixed QKV33-point replay at source0fa41585 passes all33, GM1.827869P,
+2.532% against the preceding fixed matrix, with no regressions. It retains
70.63% of the matched full148SM cuBLASLt equivalent compute throughput; that
reference adds two separately measured GEMM times and excludes preparation
and communication. The separate2P QKV target is still unmet. The shared
quantizer's OProj36-point regression also passes:2.038881→2.092341P (+2.622%),
all36 improve. All69 fixed points pass original two-payload full-boundary
checks; no configuration or sample is dropped. No new core/public header is
introduced. The unique O/Q backward current tables and both provenance lists
are in fuse_midfile/mxfp8-v23; full replay source-run20260914-033522-f255cf.

## Backward component checkpoint

QKV forward large-box control: whole-W startup, complete M128/N256 ready
unchanged, replace four64x128 copies by two128x128 copies. Six/seven route
warps passed full/tail checks and7 same-budget controls each against a fresh
baseline. DenseCP4 best gained1.97/1.74%, but DenseCP8 best was unchanged
(low-C20 fell4.49/4.68%); Qwen3 and Llama405 best changes were under0.5%.
The bounded improvement does not justify another transport organization in
the current core. Both experiments and their temporary tests were removed;
this is not evidence that larger boxes or seven warps always lose. Builds
20260914-024640-754598/025354-dc64bb and current JSON preserve the controls.

Register-owned K32 preparation trial: BF16 shared32x256 tile,16B coalesced
input vectors, one whole K32 per thread, four local maximum chains and32B
packed output. There are still two CTA barriers, now per8192 values. Scale
padding remains128 rows, independently masked from the wider256-row tile.
No shuffle reduction, new quantization rule, GEMM or communication change.
Tail M/H128 CP8 run193559-8c7eb0 passes complete validation. Qwen3 CP8/128K
193622-4ef864 with B sw8/N/C16 and dW sw8/M passes all four Graph10+50
two-payload boundaries: full1.184432ms/1.856606P, B0.541904ms, W0.630936ms,
prepared dW0.464896ms/2.365070P. Against the identical-config subgroup run
192759-d2b729, full throughput improves43.79%; exact dW time is essentially
unchanged, locating most improvement in preparation. Five other families are
being checked; this is not full-matrix or2P acceptance yet.

Exact native dW control192348-baf22c (Qwen3 CP8/128K, same C16/E32/sw8/N):
full1.725632ms, B0.618024ms, W1.131832ms, prepared native dW0.480200ms/2.289695P.
All four components have independent full pre/post two-payload checks and
Graph10+50 audits. The W-to-prepared-dW difference is0.651632ms; it indicates
substantial preparation cost but can include changed cache/launch effects.
The compute reference uses the EXACT native adapter/GEMM and original BF16
oracle, not the separate search wrapper. Preparation is explicitly excluded
only from this diagnostic. Full production remains the five-kernel boundary.
Completed full W must immediately precede this reference with unchanged
scratch; B cannot overwrite it between the two. No private Params are exposed.

Independent dW layout options now preserve dA/communication settings:
`--backward-weight-epilogue-n`, `--backward-weight-swizzle`,
`--backward-weight-raster`; absent options inherit dA, as before.
Qwen3 CP8/128K dW AlongM/sw8 (192759-d2b729) improves exact dW
0.480200→0.463624ms and full1.725632→1.703096ms (+1.32%). AlongM/sw4
(192828-e091ed) is1.708704ms full, not preferred. All eight component
boundaries passed independent audits. B remains AlongN/sw8/C16; this is not
communication retuning or a new Auto model. Preparation remains dominant.

OProj backward Qwen3 CP8/128K, fixed C16/E32/sw8/AlongN: actual complete
five-kernel Graph is1.999480ms/1.099798P. Separate B (W transpose quant,
dA and inverse A2A) is0.648888ms; W (dY/A transpose quant and dW) is1.372248ms.
Both payloads and all pre/post numerical and route checks pass Graph10+50;
run20260913-184756-a0df65, source0deb948f. The W phase dominates this point,
but these measurements do not isolate quantization from the dW GEMM itself.
Independent phase times are not added to replace the actual complete timing.
`--calibrate` selects this diagnostic; it preserves native B epochs across
component Graphs, while ordinary dW launches use a separate software counter.

## Baselines and scope

V24 transpose-quantization trial: an8-lane subgroup owns K32 with four values
per thread; four groups execute concurrently per warp. Padded shared transpose,
K32 scales and the two CTA barriers are unchanged, with packed FP8 conversion
and4-byte stores. Qwen3 CP8/128K run191442-1b11da independently passes all
three full/B/W boundaries, two payloads and Graph10+50. At unchanged C16/E32/
sw8/AlongN, full1.999480→1.727992ms (+15.71% throughput), B0.648888→0.617968ms,
W1.372248→1.132424ms. All six physical families now pass all18 full/B/W
boundaries; full B+dW geometric mean1.355396→1.505587P (+11.08%). This is
CP8/128K coverage, not a full36-point result or a2P claim. No GEMM or
communication change. The finite prep optimization is retained.

| CP8/128K geometry | Initial full P | Subgroup quantization full P | Change |
|---|---:|---:|---:|
| Qwen3 235B | 1.098861 | 1.272589 | +15.81% |
| BLOOM 176B | 1.500435 | 1.619615 | +7.94% |
| Llama405B | 1.577801 | 1.709454 | +8.34% |
| Representative large | 1.377749 | 1.529994 | +11.05% |
| Kimi KDA projection geometry | 1.328494 | 1.472623 | +10.85% |
| Llama70B/Qwen72B geometry | 1.302134 | 1.467221 | +12.68% |

Runs191442-1b11da,191651-10dda5,191734-40c568,191823-cc27e0,
191856-bfd8b9,191924-a3072e; native source4cc9e514. Exact component timings,
previous results and fingerprints are in current JSON. These are successive
same-configuration measurements, not a claim of simultaneous paired execution.

Pure dW CUTLASS search190812-025432:254 checked grid/neighbor candidates across
six physical matrices,148compute CTAs, no quantization/communication. Best
2.5035–2.5956P; all select M128/N256/K128/E32, varying Along/swizzle. This
supports inspecting preparation next, but pure prepared GEMM does not measure
the current adapter's exact in-boundary dW duration. Configurations and receipts
are retained in the single current JSON; no online selection was introduced.

Post-v23 continuation: full-device pure cuBLASLt backward geometry probe
20260913-190415-fe8d68 passed all11 physical MNK records (six dA/dW families;
Llama405 dA and dW share one matrix). Source091e7206, Graph10+50, both payloads
checked; generation0 alone is timed, unlike the two-payload backward sampler.
Independent artifact/geometry/check/sample audits pass. Range2.6484–2.7626P.
Qwen3 CP8/128K dA=0.409984ms, dW=0.406224ms. Its backward W phase above is
1.372248ms including two transpose-quantization kernels; it cannot yet be
attributed solely to dW GEMM or communication. Complete full-device pure
GEMM remains the reference; same-budget CUTLASS will isolate our compute path.

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

### OProj low-point follow-up

On the unchanged accepted implementation, four sub-2.2P windowed points compare
H64/P16 with P32, at the original communication budget and four fewer CTAs.
All16 candidates pass independent F/C/R/P re-audits, Graph10+50 and both payloads.
P32 reduces initial distinct A rows for two geometries, but increases W-panel
demand and changes pure GEMM traversal. It is not adopted: the best fused results
regress for three points, while Kimi512K gains only0.14%. Less initial A demand
does not establish an end-to-end benefit.

The retained candidate keeps P16 for the H8192/K8192, CP8/128K projection and
reduces communication CTAs20→16. Independent fixed paired confirmation
`20260913-174152-a24643` gives2.118781→2.188344P (+3.2832%); independent compute
is2.504537→2.551545P. This is a measured offline configuration, not a model-name
branch or a new Auto rule. The original36-point fixed table remains intact;
this later single-point confirmation is recorded separately, not substituted
into the earlier full-run mean. No new kernel code is needed for this result.

## Backward implementation boundary

The released backward bindings are BF16. On the development branch, complete
MXFP8 OProj B/W passes all36 physical points at1.961974P geometric mean. QKV
MXFP8 B/W passes bounded CP4/8 independent CPU-oracle tests; its formal MPI
coverage is in progress. Neither backward performance target is achieved yet.

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
interfaces, the source/saved representations, gradient and master weight
quantization, transposed scale construction, accumulation dtype and independent
reference must be explicit. Quantization/transpose work may not silently
disappear from timing or be replaced by a BF16 result labelled MXFP8.

### Explicit OProj MXFP8 backward baseline

The API declares a straight-through low-precision training convention: ordinary
linear-layer backward GEMMs, with each required operand orientation quantized
from its original BF16 source. It does not differentiate rounding/amax or claim
full-model training convergence. Independently quantizing both K32 orientations
from high precision is consistent with [Transformer Engine's MXFP8 recipe](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/common.html).

| Boundary | Sources / operations | Output |
|---|---|---|
| B/data gradient | Upstream K-H-quantized dY[M,H]; inside B, BF16 W[H,A]→K32 MXFP8 W^T[A,H]; GEMM then inverse OProj A2A | BF16 head-sharded dA |
| W/weight gradient | Inside W, original BF16 dY[M,H] and saved sequence-local BF16 A[M,A] independently transpose/quantize along M; GEMM with alpha/beta | BF16 local partial dW[H,A] |
| Immediate | Actual B then W launches on one stream; preparation included | Both gradients; cross-CP dW reduction remains caller-owned |

Upstream dY quantization is the declared external boundary, not W preparation.
Saved A must be the original BF16 tensor, not dequantized forward FP8 bytes.
Deferred W preserves the existing operand-lifetime requirement. All GEMMs use
FP32 accumulation; BF16 gradient accumulation/output matches the older B/W API.
The first implementation uses explicit transpose/quantization kernels, a padded
32x32 shared tile, and the existing persistent output-tile publisher and inverse
head route. It does not yet claim preparation/communication overlap. Forward
BF16/SM90 and the accepted forward MXFP8 algorithms are unchanged.

Build180040-d0f579 and checks180105-5ab606/180136-448f38 pass CPU-oracle bring-up
at CP4/8, M128/H128/A1024: two Philox payloads with row/column-varying dynamic
range, ordinary/causal-balanced routing, E32/E64 with different raster/swizzle,
full dA/dW numerical checks and byte-exact inverse routing. Deferred B leaves
dW untouched; subsequent beta1 W passes. All16 case summaries and both terminal
checks were independently audited against archived receipts and hashes.
The subsequent M256/H256/A1024 CP8 run180737-631610 (build180625-992008) also
passes all eight cases plus deferred/beta1, exercising multiple dA output and
dW K tiles. Native operator code is identical; only the bounded oracle harness
accepts the larger dimensions. Its receipt and case coverage were re-audited.
This is explicitly not a Graph10+50 long-sequence throughput result. Next,
extend the MPI/full-matrix measurement and independent represented-operand
reference; preserve true immediate B+W timing rather than summing isolated times.

## Evidence and iteration

### OProj backward Graph boundary and large-matrix measurement

`backward/mxfp8_mpi_bench.cu` measures the complete immediate B+W boundary in
one five-kernel CUDA Graph. `fused_graph.cuh` validates its linear dependency
chain and each kernel's cooperative/dynamic-SMEM contract; every accepted
launch advances the native epoch. The original one/two-kernel forward Graph
contracts remain unchanged. Upstream dY quantization is outside the declared
input boundary, while all three backward transpose/quantization preparations
are timed. Work is `4*M*H*A` per rank; independent B and W times are not added
to manufacture an immediate result. Cross-CP dW reduction is still caller-owned.

`backward/mxfp8_reference.cuh` independently reconstructs the represented
operands from original BF16 masters and uses FP32-accumulating cuBLAS, with
128-row/4096-K bounded scratch. It neither reads production scales nor
constructs a numerical reference from actual gradients. Full dA/dW numerical
and separate byte-exact inverse-route checks run before and after measurement
for both Philox payloads, at the original `0.01 + 0.01*abs(reference)` tolerance.
Formal sampling keeps 10+50, per-rank 100ms/three-window convergence and the
first stable round; both payload times are equally weighted, not selected by
speed. Original rank logs, input statistics and device telemetry are retained.

CP8 M256/H256/A1024 run182103-b01728 passed 16 CPU-oracle Eager/Graph cases,
including consecutive Graph updates without resetting ready. Run182829-065a63
also cross-checked the bounded GPU reference against CPU-checked results.
Qwen3 geometry CP8/128K run183608-799fb3 then passed independent archived
receipt/sample/numerical/routing audit: complete B+W **2.001184ms / 1.098861P**.
This is an untuned backward baseline, not the 2P target or a forward regression
result. Initial run183405-3468d5 computed correctly but failed collection because
its startup record did not match the common MPI controller protocol; that failed
receipt is retained, not promoted to a successful formal result. The startup
record was aligned and the successful run above is a fresh confirmation.

The first six physical large-matrix geometries at **CP8/128K** pass the same
audit, with fixed C16/E32/swizzle8/AlongN for both gradients (not tuned winners):

| Geometry | Complete B+W ms | PFLOPS/rank | Run suffix |
|---|---:|---:|---|
| Qwen3 235B | 2.001184 | 1.098861 | 183608-799fb3 |
| BLOOM 176B | 8.976744 | 1.500435 | 183734-e0b724 |
| Llama 3.1 405B | 11.149816 | 1.577801 | 183802-66ed3a |
| representative_large | 5.586344 | 1.377749 | 183835-b3bbc5 |
| Kimi K3 KDA projection geometry | 4.345096 | 1.328494 | 183903-169b3f |
| Llama70B / Qwen72B shared geometry | 3.377568 | 1.302134 | 183931-6c23cb |

Geometric mean **1.355396P**, below the 2P backward target. This is not full
CP4/8 × 128/256/512K coverage, and projection geometry checks do not validate
special KDA model routing. A backward full-device cuBLASLt reference is still
missing; the forward pure-GEMM column cannot be reused for B+W.
The long-K/CP4 cross-check, Qwen3 geometry at global512K (M131072), also passes:
run184057-aa95d0, **15.038944ms / 1.169775P**. It is reported separately, not
mixed into the six-point CP8/128K geometric mean.

The separate read-only `scripts/summarize_sm103_mxfp8_backward.py` verifies
hashes, per-rank ownership, full coverage and raw statistics; it does not relax
the forward result parser or insert backward rows into the forward table.

Keep one compact current table/configuration/evidence file in
`fuse_midfile/mxfp8-v23`, not new per-trial source scripts. Preserve formal raw
artifacts referenced by run/source/binary hashes. Profile only selected
diagnostic cases using the repository protocol; use ordinary uninstrumented
launches for performance decisions. Existing BF16/SM90 behavior and the
experimental norm/RoPE contract must not be silently changed.

### v24 backward register-owned quantization checkpoint

The transpose preparation now loads a BF16 K32 x 256-row tile with aligned
16-byte vectors; each thread owns one entire K32 group and reduces locally.
The scale rule, native padding, original BF16 source and whole-ready contract
are unchanged. This is a preparation optimization, not omission of preparation
from the complete B+W boundary. Six CP8/128K points independently pass all four
full/B/W/prepared-dW audits (two payloads, original numerical/route checks,
Graph10+50). Source-run `20260913-193535-9fe821`, native `98a9e47`:

| Physical geometry | Complete B+W PFLOPS/card | Run suffix |
|---|---:|---|
| Qwen3 235B | 1.856606 | 193622-4ef864 |
| BLOOM 176B | 2.000852 | 193826-d64471 |
| Llama 3.1 405B | 2.007219 | 193915-9b24ee |
| representative_large | 1.964440 | 194009-bc8d39 |
| Kimi K3 KDA projection geometry | 1.953442 | 194047-3830e8 |
| Llama70B / Qwen72B shared geometry | 1.943575 | 194120-ce2c59 |

Six-point geometric mean **1.953717P**. W raster/swizzle is independently
selected from the finite pure-dW search; B remains C16/E32/AlongN/swizzle8.
The subsequent remaining30 CP4/8 x 128/256/512K points used this frozen source,
without another search. All **36/36** now pass the independent full-boundary
audit with one source/binary, no missing/OOM points; geometric mean
**1.961974P**, still below the 2P backward goal. The full table is
`fuse_midfile/mxfp8-v23/oproj-backward-current.md`, with configurations and
raw references in `current.json.oproj_backward_register_full`.
The v23.0 O-forward release/table stays unchanged.

QKV MXFP8 backward development is isolated from that frozen GPU queue. It
receives prequantized planar Q/K/V gradients for dX and saves their ORIGINAL
BF16 values during inverse routing for dW. Head ownership is reconstructed in
the real packed `[all Q][all K][all V]` order; W uses the same shared transposed
K32 preparation as O. The first-use M queue is derived from the actual GEMM
layout/budget, and each ready unit covers a complete M128/head including SFA.
Upstream quantization/visibility and the all-rank input lease are caller-owned;
the complete local immediate boundary includes W preparation, inverse route,
dX, both dW preparations and dW. No replicated-KV or special KDA routing is
claimed. The independent CPU-oracle harness passed CP4 M/H128
(`201027-2f1b8f`, 8 complete B+W cases) and CP8 M/H256 (`201121-19fce7`,
16 cases including causal/noncausal routing). Both use two nonzero Philox
payloads, Eager/Graph, epilogue32/64, full dX/dW numerical checks and byte-exact
original BF16 staging checks. Standalone QKV W also passed alpha=.75 and
beta=0/1 across two packed Q/K/V widths. Receipt/source/binary hashes and
the complete validation records were independently re-audited.
`201349-52c151` additionally verifies that deferred B leaves dW byte-identical,
retains the original BF16 staging lease, and the later W applies beta=1.
This establishes bounded correctness, not large-matrix throughput, full-model
training or the 2P goal. These new entries are not part of published v23.0.

QKV MPI measurement now dispatches the same complete five-kernel boundary,
with separate B, W and prepared-native-W diagnostics. Its independent bounded
GPU reference reconstructs original peer BF16 Q/K/V planes, rather than using
production staging as a numerical oracle. Run203211-298c84 (CP8 M/H256,
source7f792e3a) cross-checks this reference against CPU FP64 for all16 cases,
and verifies prepared W and deferred beta1. Large-matrix performance remains
pending; upstream gradient quantization and cross-CP dW reduction stay explicit
caller boundaries, while original-BF16 inverse routing is included in timing.
Formal MPI CP4 M/H256 Q16/KV8 causal run203605-03b2ec passes independent
source/binary/rank/sample audits for full, B, W and prepared-W Graph10+50,
with complete pre/post checks on both Philox payloads. This bounded contract
test is not included in the long-sequence performance matrix.

QKV inverse transport uses a private6KiB stage per communication warp to issue
FP8 and original-BF16 strided loads with cp.async before dependent local stores.
Sixteen-row transport slices do not change the M128/head ready unit, SFA atom,
GEMM settings, system publication or first-use queue. CP4 causal bounded
Graph10+50 run204432-09bca8 passes all four independent component audits.
Llama70/Qwen72 CP8/128K, unchanged C16/E32/N/sw8, improves complete backward
17.751368→4.552848ms (0.309698→1.207499P); B16.429200→3.216848ms and W stays
1.417ms. Source67b2e513/run204503-7d32e1. At the same source, C32
(204617-3db180) reaches1.496586P, C64 (204651-8e9eaf) falls to1.254960P;
this finite budget comparison does not establish global optimality.
Kimi QKV-only CP8/128K C64 (204725-206968) passes complete validation at
1.151694P, not a full-matrix result or a measured improvement versus its old
transport. Further data-gradient compute/communication diagnosis is required.

Prepared dX reference keeps the SAME GEMM, acquire/TMA adapter, raster/swizzle
and compute budget, but preserves completed B's packed inputs/ready flags and
omits W preparation and communication. It must run after B and BEFORE W reuses
scratch. Its single cooperative Graph is not a full B performance claim.
Sourcea94c7e9a passes bounded CP4 causal test205353-167d7b and all five
independent large-component audits:

| CP8/128K | C | Full ms | B ms | Prepared dX ms | Prepared dX P |
|---|---:|---:|---:|---:|---:|
| Llama70/Qwen72 (205429-e21320) | 32 | 3.674536 | 2.349040 | 2.081272 | 1.320721 |
| Llama70/Qwen72 (205508-6b7ff3) | 64 | 4.381728 | 3.060480 | 2.741368 | 1.002703 |
| Kimi QKV-only (205547-3311f0) | 32 | 12.591144 | 8.226440 | 7.514200 | 1.152305 |

Already-ready dX remains most of B time. This rules out blaming most remaining
time on waiting for delivery; it does not separate arithmetic from the head-wise
acquire/load adapter. B minus prepared dX includes preparation, transport and
concurrent/cache effects, not a measured sum of semaphore wait intervals.

Single-GPU pure CUTLASS search210038-f5ff48 explicitly enforces116 compute CTAs
and includes the exact dX tile/epilogue/raster/swizzle as a measured candidate.
Both matrices have42 fully verified grid/neighbor candidates, with Graph10+50
and a second nonzero payload check. At M128N256K128/E32/sw8/N, Llama70/Qwen72
dX is1.212000ms/2.267970P and Kimi dX4.128928ms/2.097071P. Finite winners are
sw8/M at1.178272ms/2.332890P and sw4/M at3.706992ms/2.335763P respectively.
These are compute-only, single-GPU measurements with independent random inputs,
not substitutes for the eight-rank fused boundary. The gap to prepared dX
warrants testing repeated head-wise acquire/load control and N-tile input reuse;
it is not by itself a profiler attribution of every microsecond to fences.

A tagged32-bit completed-M cache was tested and removed: source6d50a0e7,
Llama C32 run211140-150ee2 full3.669848ms versus3.674536ms before (negligible),
Kimi C32 run211219-671057 full13.075896ms versus12.591144ms (regression).
All five component audits and bounded CP4 causal test210955-a2c282 passed,
but semantic correctness without a useful full-boundary gain is insufficient
to retain this extra state. No completed-row cache remains in production.

Already-ready contiguous-prefix coalescing was also tested and removed:
source677885c2, bounded causal test212011-15bfea and both large five-component
audits pass, but Llama run212127-740c11 rises to3.876848ms and Kimi
run212206-549a4e to13.231496ms. No prefix probe/cache is retained. Static
resource inspection212818-919314 shows zero stack/local bytes for the QKV
backward entries; CUTLASS device_kernel already uses GRID_CONSTANT. This
does not establish a register-spill explanation for the ready-input dX gap.

Fixed ready geometry specialization213814-5dd5d5 (sourceaf1f55f5) retains every
whole-head acquire/proxy fence and the original producer order, but makes
D128/K128=1 a compile-time indexing ratio. CP4 causal214129-85950e and both
large CP8/128K five-component audits pass. Llama214201-188884 full is3.449656ms
(1.593654P), B2.049952ms, prepared dX1.789192ms. Kimi214253-4705cc full is
11.717064ms (1.477956P), B7.176856ms, prepared dX6.414864ms. Relative to the
original same-config controls, full throughput improves6.52% and7.46%; static
register/stack/local resource counts are unchanged. This is retained indexing
specialization, not a change in ready granularity or a full-matrix claim.

Removing QKV backward's unused WeightReadyMainloop wrapper (W is fully prepared
by the preceding stream operation) gives another controlled improvement at
source de5bc712: Llama214757-c21339 full3.373160ms/1.629795P, prepared dX
1.703264ms; Kimi214852-04d8de full11.470896ms/1.509674P, prepared dX6.078216ms.
All five component audits pass. Input head acquire/proxy fence and communication
are unchanged. Relative to the original controls these two complete boundaries
are8.93% and9.77% faster; this still does not meet the2P full-matrix target.

The independent dX raster/swizzle winners do not transfer directly: at the
same source and C32, Llama sw8/M215045-4d7b62 full4.389008ms (B3.069024,
prepared dX1.823488), and Kimi sw4/M215148-f78256 full15.184480ms
(B10.986160, prepared dX6.072960). Both pass all five audits, but full boundaries
regress versus sw8/N; retain N for these controls. Kimi's prepared dX is almost
unchanged while B grows, implicating the coupled production/consumption path,
not a measured arithmetic slowdown. No model-specific dispatch is introduced.

Shared dW beta-zero specialization uses the source-free epilogue only when
beta==0; all nonzero beta values keep the original BF16 C kernel. This gives
E32 four mainloop stages instead of three without changing explicit tuning.
Source4945f245 QKV Llama215754-d9f949 full3.242272ms/1.695588P versus
3.373160ms/1.629795P, with unchanged dX and dW1.421472→1.305960ms;
prepared dW1.177592→1.074280ms. All five component audits pass.
OProj Qwen3215646-42d7d9 measures full1.143040ms/1.923837P and prepared
dW0.416712ms, but uses noncausal routing while the previous formal O matrix
uses causal routing: its full result is NOT a paired speedup or a replacement
for that table. Complete matched O-matrix replay remains required.
CP8 M/H256 contract220127-05f105 independently verifies both operators,
two payloads, E32/E64, eager/Graph, ordinary/causal routes, original BF16 masters,
prepared W, deferred beta1, and CPU FP64 alpha=.75/beta0,1 weight references.

Matched OProj backward full replay v24-beta-zero-oproj-full-01..36 completes
all36 physical points at source4945f245/binarydc64b589. Each configuration,
including causal routing and B/W layouts, is checked against the previous
row. Graph10+50/two-payload/full pre-post audits all pass; GM1.961974→2.038881P
(+3.92%),36 improvements and no missing points. The aggregate2P target is met,
not an assertion that every point exceeds2P. The unique current table is
fuse_midfile/mxfp8-v23/oproj-backward-current.md; raw provenance is in current.json.
Core CUDA consolidation: both reverse operators share the transpose quantizer,
operand/workspace handling and dW kernel/dispatch; only route-specific B adapters
remain separate. No new core or public header is needed for these optimizations.

QKV fixed-head ReadyKIterator (source5e24057b) acquires on the elected TMA
issuer's dereference, then fences in that same thread; whole-warp ++ never
polls and returning the end iterator cannot acquire an out-of-range head.
The four A/B/SFA/SFB dereferences share a per-issuer K cache. This uses one
stock CUTLASS load loop, not one Base::load call per head. Publication grain,
K order, and system-scope readiness remain unchanged. Host-extracted iterator
tests cover rotating issuers, prologue/remainder and end sentinel. CP4 causal
222809-7673d1 and both CP8/128K cases pass all five original component audits.
Llama222857-5f208e full3.094520ms/1.776546P (+4.77% over the beta-zero control),
prepared dX1.558416ms; dW is unchanged. Kimi222958-512a4d full10.807920ms/
1.602279P, prepared dX5.575232ms and dW4.304376ms. Its previous direct-collective
control still used the old dW epilogue, so its full gain cannot be attributed
solely to the iterator. No new CUDA/public file is introduced.

A private two-slot cp.async transport trial (sourcebbcc20c5, original whole-head
publication) is rejected: same C32 Llama224207-d805e8 is1.786597P (+0.57%),
Kimi224356-75bcf9 is1.587751P (-0.91%). Llama C20 224540-8b6a48 is1.780842P,
but has no same-C20 single-slot control. All five component audits pass; mixed
gains do not justify doubling communication storage48→96KiB. The extra slots,
stage logic and trial-specific test were removed, not retained behind a flag.

Whole-head M-cohort narrowing also does not transfer the independent AlongM
winner. Source15cf21e3 uses max(1,8*comm/heads), original5e24057b uses8*comm;
all other parameters match. Llama225906-0e6de8→230420-c0e0fc full1.336796→
1.369708P (+2.46%), Kimi225944-55f06b→230459-16a356 1.222639→1.202496P
(-1.65%). All five audits pass. Both remain below the existing AlongN controls;
revert the cohort formula and trial-specific tests instead of adding a knob.

Local GPU-acquire/SYSTEM-release control sourced9ae9e99 has no meaningful
gain: Llama231718-459471 full1.778878P/prepared dX1.552168ms, Kimi231751-8f358e
full1.602419P/prepared dX5.582312ms; all five original audits pass, as does
CP4 causal231700-9a8fbc. The original SYSTEM consumer is restored. Earlier
231202-b9c603 was rejected before kernel launch because the old parameter
guard coupled head counts above8 to SYSTEM scope; it was not a numerical
failure or valid performance sample. Neither the scope change nor its extra
guard/tests are retained. The fixed-head iterator remains the measured winner.

At C20, increasing each copy warp's private slice16→64 rows (same complete
M128/head release, queue and GEMM) is effective. Source0a8ab986 C20 Llama
232749-abd475 full3.086632ms/1.781086P, Kimi232828-1b3760 full10.264272ms/
1.687144P. Same-C20 original16-row controls at sourcec0b98927 are
233413-acc7c4 full3.824120ms/1.437601P and233451-475d98 full12.817288ms/
1.351090P. Prepared dX is essentially unchanged (Llama1.340072/1.345040ms,
Kimi4.968960/4.969416ms), isolating the transport improvement. The controls
add a separately timed bare-GEMM diagnostic; their production path is the
original16-row transport. All original five components pass independent
audits; new bare diagnostics also pass original BF16 references on all ranks.
C32 controls do not show the same gain: new1.782278/1.579973P versus previous
1.776546/1.602279P. Thus24–25% is a SAME-C20 gain, not a gain over the previous
best configuration. Retain64-row transport and evaluate finite CTA budgets.

The new `data_gemm` diagnostic uses the same completed-B operands, actual
compute budget, CTA/SMEM reservation and simultaneous MPI ranks, but removes
the ready adapter. It runs after B and before W overwrites scratch; it never
bypasses readiness in production. In the C20 controls, Llama adapter1.345040ms
vs bare1.155800ms, Kimi4.969416ms vs4.050744ms. These measure total adapter
overhead under that environment, not the latency of any one fence/instruction.
CP4 causal233355-aa194d passes all six boundaries. Full backward still counts
the same five kernels; neither reference substitutes for full performance.
