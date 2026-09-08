# SM103 BF16 fused operators

This is the first **self-written fused** implementation, not a wrapper around
the independent cuBLASLt/NCCL or TE Userbuffers benchmarks. Source review was
approved on 2026-09-06; node2 CUDA bring-up and measured optimization are now
authorized. CPU checks do not establish CUDA compilation, numerical
correctness on B300, or performance. SM90 stays unchanged during this phase.

## OProj accumulation-direction experiments (review first)

“通信和计算在累积方向上的一致性” treats production and consumption in
the same original `A[M,K]` coordinates. Align their direction, granularity,
and ready publication so produced data forms a usable next compute step.
It is a multidimensional head-of-line dependency principle, not a promise
that the smallest directional angle gives the fastest kernel: transfer rates,
GEMM efficiency, synchronization cost and concurrent completion still matter.

Two explicit `FUSE_SM103_OPROJ_POLICY` values are experimental; default/auto
and the existing five policies retain their original compute/copy paths:

| Policy | Physical GEMM tile | Original-A ready unit | Owner |
|---|---|---|---|
| `kslice_m128n256k64` | 128x256x64, normal, E32 | 128 rows x K128 | one communication slot per K slice |
| `row_m128n32k64` | 128x32x64, SwapAB, auto epilogue | 32 rows x full peer K | one slot per row/peer block |

Both use the same rectangular TMA transfer implementation and preserve local
staging reuse across projection tiles. RowBlock can exceed its 48 KiB slot;
the owner then streams capacity-bounded K rectangles before one release.
It removes cross-owner fan-in, not the need to finish the advertised region.
SwapAB computes `D^T = W^T A^T` through views and a column-major epilogue into
the original output buffer, without a separate transpose or split-K reduction.

Supergroup integration is derived, not a second communication tuning knob:
lower CUTLASS's actual raster/effective swizzle and compute CTA count; map
physical tiles back to original sequence/projection axes; group original
sequence tiles by their first-use compute window; order communication within
each window by peer, K slice, then sequence tile. A supergroup can cross a
compute window, and a window can cross a supergroup. Neither adds a runtime
completion barrier. Swapping axes also swaps which physical axis defines the
first-use window. Existing `max_swizzle_size` remains the explicit search input.

These plans currently require sequence/batch/causal boundaries aligned to their
ready row tile and peer K divisible by their advertised K unit. Unsupported
opt-in geometry returns an error, never silently times a legacy fallback.
Ready storage must be queried for the selected policy; changing policy/layout
requires cleared counters and a restarted epoch sequence after prior work has
finished. The benchmark resets this state at each candidate/payload boundary.

PROFILING retains logical coordinates and indexes records by ready K slice.
For rectangular copy path 3, a row owner's span can include multiple G2S/S2G
pairs and must be displayed as a combined pipeline, not pure S2G latency.
Host arithmetic tests cover both orientations, supergroups, padding, reduced
compute budgets and exact transfer ownership. CUDA compilation, device
correctness and speedups remain unverified until the user reviews this change;
do not promote these candidates into an automatic performance model yet.

## Measured development log

- `20260906-100534-446741`, node2, production SM103a build: failed at
  `A2ALhsReadyMainloop::load`. NVCC rejects assignment of CUTLASS's
  `ForwardCoordIterator`, which contains a reference member. Preserve the
  referenced shape and copy only the advanced coordinate returned by the
  base load. No synchronization or peer-segmentation strategy was changed.
  Rebuild and device validation are still required; this is not a performance
  conclusion. Raw build evidence is in that run's fetched `control/attempt1.log`.
- `20260906-100757-b8c7c7`: the iterator fix passed CUDA device compilation,
  then the host compiler rejected the generated scheduler argument cast.
  Replaced redundant explicit base casts with normal derived-to-base
  conversion/reference binding; scheduling semantics remain unchanged.
  Production SM103a compilation succeeded in `20260906-101116-4adf22`
  (55.6 s total); the generalized harness-only rebuild
  `20260906-101436-6376e6` took 6.8 s, reusing the compiled kernel library.
- `20260906-101446-032ee2`, CP8, S=2048, H=1024, Q=32, KV=8, D=128,
  QKV N128 and comm CTA=8: both production directions passed GEMM and full
  bitwise route checks on all eight ranks, including changed input payloads.
  GEMM reference max error was zero. Micro CUDA-event max-rank p50 was
  0.125216 ms (GEMM+A2A) and 0.036512 ms (A2A+GEMM), each 10 warmups +
  50 samples; half-sample drift 1.72% / 0.70%. These sequential-host-launch
  micro measurements are not same-boundary strong-baseline comparisons or
  long-sequence performance claims.
- `20260906-101736-a445a5`: CP8 macro-instrumented smoke passed both
  directions, complete CTA/peer record validation and changed payloads. The
  first instrumented QKV epoch showed rank-0 local roles done in 15.872 us
  but a 412.448 us total CTA span, mostly after completion publication;
  rank-7 span was 20.800 us. This is a cold diagnostic observation, not a
  steady-state bottleneck or a production speedup. The harness now warms
  the distinct instrumented specialization and records host launch API time
  separately before drawing a host-overhead conclusion.
- Long causal CP8 production bring-up passed both complete boundaries,
  routes and changed payloads with N128 / comm CTA=8:
  `20260906-102231-d2522b`, S=65536, H=2048, Q=16, KV=8, D=128,
  p50 0.344432 / 0.266544 ms (GEMM+A2A / A2A+GEMM);
  `20260906-102711-1631b2`, S=131072, H=4096, Q=32, KV=8, D=128,
  p50 1.269424 / 0.736368 ms. All half-sample drifts were below 1%.
  These are the existing single-process eager 10+50 measurements, not
  matched strong-baseline comparisons; input distribution, OProj causal
  layout and reference warmup protocol must be aligned for that claim.
- `20260906-103001-d8fa40`, S=131072 / H=4096 / Q=32 / KV=8 / D=128,
  CP8, N128, comm CTA=8: independently warmed macro profiling passed full
  numerical/routes and CTA/peer checks. QKV production host launch API took
  9.804--13.660 us/rank in the diagnostic pair. Within each GPU, compute
  roles finished at 803.968--857.504 us while communication roles finished
  at 1100.864--1150.112 us (relative to its earliest CTA start). Overall
  QKV CTA spans were 1154.944--1243.392 us, including peer completion waits.
  A2A communication ended at 721.376--753.344 us and compute at
  733.152--761.248 us. These observations justify testing more communication
  CTAs first; they do not yet prove why each copy is slow. Descriptor caching
  is deferred because measured aggregate host launch time is much smaller
  than the long-case device boundary. No cross-GPU timer offset was assumed.

First manual tradeoff experiment, same immutable source `a8806483`, random
inputs, geometry and old single-process 10+50 collector throughout:

| QKV tile N | comm CTA | GEMM+A2A p50 ms | A2A+GEMM p50 ms | Run |
|---|---:|---:|---:|---|
| 128 | 8 | 1.269424 | 0.736368 | `102711-1631b2` |
| 128 | 16 | 1.076048 | 0.474064 | `103509-593899` |
| 128 | 32 | 1.088816 | 0.577648 | `103627-5c160b` |
| 256 | 16 | 0.979584 | 0.474608 | `103840-e03c5c` |
| 256 | 32 | 1.023680 | 0.573968 | `103958-984699` |

All run IDs have prefix `20260906-`; geometry is S131072/H4096/Q32/KV8/D128,
CP8 causal. A2A's tile remains N128 in every row; QKV tile bindings also carry
their documented communication-task defaults. Every run passed both full
numerical/routes and changed-input checks. More communication CTAs are not
monotonically better: 32 loses to 16 in these tested configurations. This is
a measured candidate comparison, not a general policy, model fit, comparison
with the H2048/Q16 Qwen baseline, or proof of a globally optimal CTA count.
Next measurements use the aligned input distribution and converged sampler;
the old collector's results remain clearly separate diagnostic evidence.

Aligned-input first same-node comparison, Qwen S131072/H2048/Q16/KV8/D128,
CP8 eager (a different geometry from the calibration table above):

- Fused `20260906-104604-61d88c`, N256 / comm CTA16: QKV p50/p95
  0.396128/0.400515 ms, OProj 0.273728/0.280171 ms. Both complete numerical,
  route and changed-input checks passed. Every rank accumulated at least
  100 ms measured warmup and converged; ten sample-cadence warmups preceded
  the first stable 50-sample round. Activations U(-.125,.125), weights
  U(-.02,.02), reproducible recorded seeds. Continuous GPU telemetry was
  captured; active observations reported 2032 MHz on all eight GPUs.
- Node1 QKV completed-group snapshot selected one NCCL and one UB config,
  then `20260906-104857-313a38` replayed exactly these two configurations on
  node2. No communication search or node1 measurement import occurred.
  The first node2 GEMM algorithm setup had disk-cache misses, not fake cross-
  node cache hits. All 16 rank records passed the original measurement reader.
  NCCL p50/p95 = 0.631264/1.272645 ms; UB = 0.567776/0.676150 ms.
  Both have zero reported numerical/route errors. NCCL's wide tail remains
  visible; no fastest-round selection or removal of slow samples was applied.
- Relative to the faster same-node QKV baseline (UB), this fused point is
  **1.433x** faster by p50. It is one manually configured point, not a fitted
  general policy or a full-matrix geometric-mean milestone. Fused still uses
  sequential host rank launches while references use one process per rank;
  both report complete eager max-rank event boundaries. OProj is not compared
  here: its causal input-layout equivalence still needs explicit resolution.

Same-process candidate reuse, measured before GPU-validation optimization:

- CP4 `20260906-105700-ef7955`, the same Qwen S131072 geometry, N256 / CTA16:
  QKV p50/p95 0.669616/0.679656 ms, OProj 0.369072/0.378626 ms. Both directions
  passed full numerical/routes and changed-payload checks; this is not yet a
  CP4 strong-baseline comparison.
- CP8 `20260906-110226-de01f5`, source `066f7d95`: one process reused inputs
  and references for QKV N128/N256 x CTA8/16, and OProj N128 x CTA8/16.
  All six unique candidates passed both input generations. Best observed
  QKV/OProj p50 was 0.396320/0.273584 ms; acceptance is recorded only after
  the second-generation checks, not when the timing summary first appears.
- Measured program stage totals: setup 9.169 s, input generation 17.451 s,
  reference 0.691 s, full CPU validation 30.887 s, benchmark 2.549 s,
  cleanup 0.064 s; program total 60.812 s, controller task 69.6 s.
  CPU validation accounts for about 51% of the measured program stages.
  This directly motivates full-coverage GPU validation with small reductions;
  it does not justify sampling fewer elements or relaxing tolerances.
- `20260906-111920-81afe9` kept the same `066f7d95` source and changed global
  sequence to 262144, extending the explicit CTA list to 8/16/32. All nine
  unique candidates passed both payloads. QKV N128 p50 for those CTA counts:
  1.162464/0.823568/0.787168 ms; QKV N256: 1.087824/0.733040/0.713616 ms;
  OProj N128: 0.955648/0.505680/0.359360 ms. Every candidate selected its
  first stable sampling round. Stage totals: setup 8.726 s, input 34.842 s,
  reference 1.137 s, CPU validation 90.808 s, benchmark 2.636 s, cleanup
  0.088 s; controller task 147.2 s. These are manual calibration candidates,
  not a 256K strong-baseline result or a universal CTA32 policy.

GPU full-validation optimization (operator and sampler unchanged):

- Build `20260906-112712-a73ab8` passed SM103a compilation in 7.9 s. The
  private `benchmarks/sm103/fused_validation.cuh` performs complete element
  checks and deterministic reductions with 16,512 B scratch/rank; only
  two 64 B statistics records return to the CPU. Numerical tolerances are
  unchanged; route checks remain bitwise and nonfinite values always fail.
- Small diagnostic runs `112827-de1fa7` (CP8, M258, causal) and
  `112915-b33fe2` (CP4, M257, rank-major) passed full CPU/GPU oracle agreement,
  13 injected-error detection/restoration checks each, and all six candidates
  across two payloads. These self-tests intentionally accept no performance
  samples. They exercise real GPU/P2P/reduction and non-tile-aligned M tails.
- Same-workload A/B against `111920-81afe9`: `113032-98ef3f` uses source
  `c9e632f5`, identical Qwen S256K/CP8 inputs, nine configurations, two
  payloads and converged 10+50 protocol. Full validation stage fell from
  90.807912 s to 0.140892 s; whole controller task from 147.2 s to 57.6 s
  (2.56x). All nine candidates passed; corresponding production p50 changes
  were below 0.5%. Input generation remained 34.8 s in both runs and is now
  the largest measured host stage. This is workflow acceleration, not a
  kernel speedup, reduced correctness coverage or a new performance model.

GPU input preparation (explicit `--input-generator gpu_philox`):

- Production build `20260906-114911-29c437` actually recompiled CUDA in
  56.6 s. The preceding `114549-321406` receipt is invalid: a restored zero
  source mtime caused Ninja to skip a changed harness. No GPU validation ran
  on that mistaken receipt; the controller now refreshes changed-file mtimes
  only, with an explicit workspace-local `--rebuild` repair path.
- `115056-8fad0a` (CP8 causal M258/H1032) and `115144-1fc75e` (CP4 rank-major
  M257/H1032) each passed eight candidates, two payloads, 13 fault injections,
  full CPU/GPU checker agreement, full cross-rank weight equality, same-seed
  repeatability and independent CPU/GPU OProj gather equality on every rank.
  These diagnostics accept no performance samples.
- Philox uses the existing CUDA header, fixed grid/subsequence mapping,
  unchanged BF16 distribution/ranges, bounded 12,336 B scratch/rank and small
  full-input statistics. It generates **different random sequences** from the
  default CPU mt19937; this is not a bit-identical input A/B. The production
  kernels, complete two-payload checks and converged 10+50 protocol are unchanged.
- The same 42 candidates at S128K/CP8 (`115256-893896`) and S256K/CP8
  (`115344-90794d`) all passed. Whole tasks were 39.2/30.9 s versus
  57.7/70.4 s with CPU preparation; total activation-input stages were
  0.00854/0.01580 s versus 17.4466/34.8147 s. This is measured workflow
  acceleration, not a kernel speedup. GPU preparation avoids large host
  activation/reference-input vectors in normal mode; CPU oracle remains
  available for bounded diagnostics.
- Best observed configurations did not change: QKV N128/CTA24 at 128K
  (0.356096 ms), N256/CTA24 at 256K (0.709120 ms); OProj N256/CTA32
  (0.167664/0.309888 ms). These remain finite-candidate observations, not a
  fitted model or a full-matrix milestone.

OProj comparison contract: the fused causal route selects chunks
`[rank, 2*CP-rank-1]`. Historical NCCL/UB workers have different legacy
chunk orders on unchanged input. SM103's explicit baseline replay option
`--oproj-layout causal_dual_chunk_v1` adapts the timed NCCL/UB route to this
contract, without pre-permuting inputs outside the timer or changing SM90.
Legacy node1 OProj timings are not same-layout comparison evidence; only their
configuration may be consumed and measured anew on node2. Fused comparisons
must explicitly use `--causal` as well.

`20260906-111358-afc404` replayed three completed CP4 winner groups on node2
(12 rank receipts independently reread). QKV NCCL p50/p95 was
1.177280/1.190661 ms and UB 0.852640/0.892202 ms, both numerical/routes zero.
The fused CP4 point above is 1.273x the faster UB p50, still only one point.
Canonical OProj NCCL passed full inverse-route mismatch=0 and GEMM error=0,
p50/p95 0.658608/0.716483 ms; raw and all ranks carry the canonical layout.
This is actual installed-TE inverse-route execution, not just an import check.
At that snapshot the corresponding OProj UB group was pending on node1;
the later CP4 UB validation below supersedes that pending state. Canonical
Graph remains separately unverified.

`20260906-121834-e715a3` (72.6 s) replayed all four newly completed Qwen
OProj S128K/S256K CP4 eager groups on node2, explicitly canonical layout.
NCCL p50/p95 was 0.655536/0.699763 ms and 1.240080/1.296342 ms;
TE UB was 0.399664/0.408222 ms and 0.696240/0.790101 ms.
Actual UB JIT/execution passed numerical max-abs 0.001953125 and zero
self-pack/remote-route mismatches; all 16 rank records independently passed
the original measurement reader and carried `causal_dual_chunk_v1`.
This establishes two same-layout CP4 strong-reference points, not CP8/Graph
coverage or permission to reuse legacy OProj times as canonical measurements.

`20260906-120148-631400` replayed the two completed Qwen S256K/CP8 QKV
winner groups on node2 (88.3 s including fresh local GEMM plan work). NCCL
p50/p95 was 1.182352/1.192563 ms; UB was 0.962224/1.036395 ms. All 16 rank
records passed the original reader, with zero numerical/route errors. The
finite-candidate fused N256/CTA24 point is about 1.357x the faster UB p50;
this is still one eager point, not a generalized model or a matrix milestone.

Independent C/R/F calibration, v10 (2026-09-06):

- Twelve node2 runs, `123410-4510b3` through `125417-881f29`, cover
  H/Q/KV/D = 2048/16/8/128, 4096/32/8/128 and 16384/128/8/128,
  global S128K/S256K, CP4/8. Each explicitly tests comm CTA8/12/16/24/32/48,
  QKV N64/128/160/192/256 and OProj N128/256: 504 fused candidates plus
  504 independent compute and 504 independent communication measurements.
  All 1512 records passed two-payload component-specific validation and
  converged 10+50 sampling. The local audited dataset is
  `fuse_midfile/sm103-cfr-calibration-v10/summary.{json,csv}`; all runs share
  source `cce45a17`, binary `2a865f1f`, environment `604f4464`, GPU Philox,
  per-GPU host workers and collector `per_epoch_rank_events_v3_eventsync`.
- Reference compute uses the selected stock BF16 family and actual compute
  subgrid, with the production shared-memory floor; reference communication
  uses real routing and, for QKV, full cross-rank finalization. Independent
  flags and invocation epochs prevent calibration from advancing production
  ready state. C/R are independent observations, not an exact decomposition
  or lower bound for F. For example, `124959-e6d508`, QKV N64/CTA16:
  F=33.661520 ms, C=41.502720 ms. There are 23 such negative
  `F-max(C,R)` residuals among 504 configurations; none were discarded.
  Selecting by minimum `max(C,R)` instead of measured F loses 10.12%/6.74%
  geometrically on these QKV/OProj calibration points, respectively.
- Half-sample stability does not guarantee a narrow tail: `124959-e6d508`
  QKV N64/CTA8 has p50/p95 34.224943/39.699322 ms while half-drift is 1.65%.
  Raw samples and unsuccessful sampling rounds remain evidence. These are
  finite-grid training observations, not a deployed selector, held-out
  validation or a performance milestone.

Host and GPU macro diagnosis, v11:

- Profile build `130208-605b08` and six runs `130326-6d5369` through
  `130652-5609d5` passed complete validation. The compact local summary is
  `fuse_midfile/sm103-host-profile-v11`; no large trace is copied into it.
  Its 4400 `host_stage` records are production-kernel **diagnostics**, not
  formal samples. Each direction validates the instrumented output before
  these 10+50 diagnostic epochs, then validates their production output;
  `profile_schema=host_stages_v1` requires both explicit phases.
- Qwen S128K/CP8, CTA24, sequential `130340-f91632` versus parallel
  `130428-f231e0`: host API-start skew p50 falls 81.257 to 3.438 us for QKV
  and 60.246 to 2.902 us for OProj. All-enqueued p50 falls 92.655 to 59.365 us
  and 68.594 to 33.278 us. However pooled rank/API p50 grows 8.292 to
  38.584 us for QKV, including peer-descriptor preparation 2.665 to
  24.097 us; OProj API grows 5.260 to 14.741 us. Parallel submission reduces
  skew but makes individual calls more expensive here. Internal driver
  contention is a hypothesis, not established by these host timestamps.
- OProj CTA32, N128 `130516-7b398c` versus N256 `130604-97e3be`: normalize
  timestamps to each GPU's own earliest CTA start, then take rank medians.
  Communication ends at 137.008/146.256 us, final ready publication is
  observed at 136.560/145.824 us, and compute ends at 257.040/179.488 us.
  N256's improvement is not earlier input completion in this trace. The
  consumer M-window (8/15), tile count and GEMM resources change together;
  this does not isolate one cause. Instrumentation strongly distorts this
  comparison: paired instrumented-minus-production event differences have
  rank medians 76.896/24.432 us. Do not treat the macro compute-time difference
  as native speedup; the profile-build production p50 is 201.696/179.520 us.
- QKV N128, CTA16 `130652-5609d5` versus CTA24 `130428-f231e0`: locally
  normalized compute role completion is 229.472/234.560 us, communication
  role completion 290.592/257.904 us, kernel end 305.920/276.144 us. More
  communication CTAs finish fused communication earlier here despite the
  opposite isolated-R trend. Finalization did not explain this improvement.
  QKV has no per-tile ready-frontier record in this experiment. OProj uses
  CTA `end`, not its unset `role_done`; acquire/release are observations,
  not actual wait durations, and absolute timers are never subtracted
  across GPUs. All profile-build timings remain separate from v10 results.

Same-node Llama QKV reference replay, `130741-a5bcac`:

- Sixteen original winner configurations (eight geometries, NCCL and UB)
  were replayed, not searched again: H4096/Q32 and H16384/Q128, KV8/D128,
  S128K/S256K, CP4/8. All 96 rank records passed the existing measurement
  reader; archive-backed raw samples, first stable rounds, actual MNK,
  random BF16 inputs and complete packed rank-major QKV boundaries were
  independently checked. Reported numerical and route errors are zero.
- Against the faster same-node reference by p50, v10 finite-grid fused best
  achieves geometric-mean baseline/F ratios of **0.982933x** for the four
  8B points and **0.795353x** for the four 405B points (all eight: 0.884183x).
  None of these eight fused points wins yet. For example, 405B S128K/CP4:
  NCCL 16.242864/16.777611 ms, UB 15.258624/15.876771 ms p50/p95,
  versus fused N256/CTA8 p50 20.033936 ms. UB wins seven reference points;
  NCCL wins 405B S256K/CP8. Wide tails are retained, not used to change
  the p50 selection rule. These are calibration-only comparisons, not
  runtime-model scores or full-matrix milestones. Baseline and fused
  collectors/host orchestration and random sequences differ, while the
  complete eager max-rank boundary, geometry and input distributions align.

Direct CUDA-driver entry calls, v12:

- `cmake/sm103.cmake` privately enables CUTLASS's direct-driver-call path
  and links `CUDA::cuda_driver`; it does not alter SM90 or the operator
  algorithm. Profile build `131929-b17ad1` and production build
  `132150-0c8466` record `libcuda.so.1 => /lib64/libcuda.so.1`, not a toolkit
  stub. The v11/v12 operator and harness file hashes match. Seven successful
  runs `132048-d12c6d` through `132412-eeedf5` passed the complete local
  artifact audit, including CP4/8 CPU-oracle and small independent C/R checks.
  `fuse_midfile/sm103-direct-driver-v12` contains 36 component records;
  source `b331b416`, profile binary `62747c21` and production binary
  `db586b08` are explicit and remain separate from the v10 fitting dataset.
- Qwen S128K/CP8, CTA24: parallel profile runs `130428-f231e0` versus
  `132127-abe650` reduce pooled rank/API p50 from 38.584 to 7.468 us for
  QKV and 14.741 to 6.110 us for OProj. QKV peer-descriptor preparation
  falls 24.097 to 1.294 us; parameter lowering falls 5.832 to 0.601 us
  for QKV and 6.960 to 0.668 us for OProj. Parallel all-enqueued p50
  falls 59.365 to 30.622 us / 33.278 to 27.974 us. Sequential QKV API
  also falls 8.292 to 6.369 us; sequential OProj changes only
  5.260 to 4.935 us, with small regressions on individual ranks retained.
- Production `132327-c6fa7a` and `132349-775492` repeat 16 same-config
  Qwen candidates at S128K/S256K, CP8, N128/N256 and CTA24/32. Relative
  to v10, all p50 values are lower in this run, by 0.89--7.09%. Examples:
  S128K QKV N128/CTA24 p50/p95 0.319088/0.324117 to
  0.301344/0.306915 ms; OProj N256/CTA32 0.181280/0.187386 to
  0.172896/0.178954 ms. S256K QKV N256/CTA24 changes
  0.688832/0.694898 to 0.675296/0.677883 ms.
- Large-GEMM benefit is not established: `132412-eeedf5`, 405B S128K/CP8,
  N256/CTA8, changes QKV p50 9.980960 to 9.922752 ms (-0.58%) and OProj
  9.239920 to 9.132752 ms (-1.16%). The observed CPU improvement supports
  retaining this small host-side change, not claiming better tensor-core
  throughput or closing the 405B baseline gap. A/B runs were not interleaved
  repeated trials; whole-subprocess telemetry includes idle clocks and
  cannot prove identical per-candidate clock/thermal conditions. Profile
  event timings remain diagnostic and are not pooled with production F.

K-step and epilogue experiments, v13:

- Build `133045-f9e788` and five runs `133219-e59981` through
  `133432-97d619` passed the complete artifact audit: two CP4/8 tail cases
  with CPU oracles, then Qwen/8B/405B S128K/CP8, 330 F/C/R records total.
  `fuse_midfile/sm103-k-epi-v13/summary.{json,csv}` records source
  `debec64d` and production binary `4537b2b8`. The long runs compare five
  real policies at CTA8/16/32; actual MNK and shared-memory traits are
  checked on every rank. C/R resource fields still distinguish production
  reference values from unreported actual reference-kernel occupancy.
- At fixed N256/K64 and communication CTA count, explicit 128x32 epilogue
  tiles reduce QKV p50 at all nine long points by 2.76--12.65% versus Auto.
  OProj p50 is also lower at all nine points, by 0.15--3.71%; the smallest
  differences are not evidence of a robust general speedup. Raising K64
  to K128 with this epilogue makes all nine points slower in both directions.
  N128/K128 is not generally better either, although Qwen OProj/CTA32
  improves 8.97% versus N128/K64. These are matched-configuration results,
  not proof that larger K steps are inherently inefficient on Blackwell.
- Best measured QKV uses N256/K64/e32 at CTA32/16/16 for Qwen/8B/405B:
  p50/p95 = 0.300144/0.307070, 0.837616/0.847251 and
  9.108576/9.291088 ms. Against the already verified same-node stronger
  reference (UB p50 0.567776, 0.927440 and 8.306272 ms), baseline/F is
  **1.891679x / 1.107238x / 0.911918x**. These are finite-grid winners,
  not runtime-model choices or a full-matrix milestone. Canonical CP8
  OProj replays were still pending at this snapshot; the following replay
  supplies them without substituting legacy-layout timings.
- At 405B QKV/CTA16, the winning e32 policy has F/C/R =
  9.108576/8.850624/0.903424 ms. Independent C itself exceeds the complete
  UB boundary by 0.544352 ms, while F-C is 0.257952 ms. This prioritizes
  investigating the compute path and SM budget, without treating C as a
  strict lower bound or decomposing the total gap causally. For OProj at
  the same CTA count, K64/e32 to K128/e32 changes F from 9.379888 to
  15.851296 ms, C from 8.673792 to 9.691280 ms, with R nearly unchanged
  at 1.374 ms: bare-compute slowdown alone does not explain the fusion
  regression. Its pipeline/ready interaction remains a hypothesis to test.

Canonical same-node OProj replay, `140309-89a5c3`:

- Six existing winner configurations, Qwen/8B/405B S128K/CP8 and both
  backends, completed in 171.4 s. All 48 rank records and 131 archive-backed
  files passed local validation, including random inputs, converged warmup,
  10+50 first-stable-round samples and explicit `causal_dual_chunk_v1`.
  Routing mismatches are zero. NCCL numerical max-abs is zero; UB max-abs
  is 0.001953125/0.0029296875/0.005859375, within its checked tolerance.
- NCCL p50/p95 is 0.425744/1.693792, 0.806944/1.092658 and
  7.140688/7.378241 ms; UB is 0.315184/0.345698, 0.663712/0.683238 and
  7.123776/7.263093 ms. UB wins all three by p50; wide NCCL tails remain
  recorded. Against v13 finite-grid F = 0.172384/0.494336/9.085168 ms,
  baseline/F is **1.828383x / 1.342633x / 0.784111x**. The three-point
  geometric mean 1.243943x is calibration-only, not a matrix milestone.
  This replays historical winner configurations on the new layout, not
  a fresh canonical-layout parameter search. Raw evidence remains in the
  original run; no duplicate summary is created.

CTA-only versus full diagnostics, v14:

- Five runs `140045-e96bf9` through `140217-d8d26c` passed the complete
  artifact/phase audit, including CP4 full and CP8 CTA-only CPU-oracle tails.
  `fuse_midfile/sm103-cta-profile-v14` is a compact diagnostic summary;
  its production-event samples belong to profile binary `445ad552` and
  are not pooled into the production performance dataset.
- At 405B S128K/CP8, CTA16, old N256 versus e32 CTA-only traces have
  per-GPU-relative rank-median QKV compute completion 9624.512/8227.664 us,
  communication completion 9667.200/8279.200 us, and end 9940.320/8549.488 us.
  The e32 end-minus-last-local-role median is 271.152 us, but individual
  ranks range 5.824--842.752 us while acknowledging other ranks. This is
  not 271 us of removable critical-path overhead. Timers are normalized
  within each GPU, never subtracted across GPUs.
- Instrumentation is not necessarily a positive additive cost: paired
  OProj production/instrumented event medians are 9252.272/7966.448 us for
  e32 CTA-only and 9278.112/8206.960 us for full traces. Actual production/
  telemetry registers are 96/114, with 230400 B dynamic shared memory.
  Full traces observe final ready at 1747.776 us and compute end at
  8192.992 us, but cannot establish native production throughput or actual
  acquire-wait duration. Specialization/code generation and repeated paired
  measurements need investigation before assigning a cause to this gap.

No-residual epilogue and extended calibration, v15:

- Production source `d07bac7a`, binary `e26bf8b0`, makes epilogue C void;
  A/B/D remain BF16 and accumulation/compute FP32. N256/K64's real mainloop
  now has four stages, with unchanged production ready/finalize protocols.
  Build `141337-7830f7`, two CPU-oracle tails `141632-28f935` /
  `141659-06d31a`, and twelve long runs `141732-c4917b` through
  `143943-05fa92` passed all component validation and sampling checks.
  `fuse_midfile/sm103-cvoid-v15-full/summary.{json,csv}` contains 762 rows:
  254 F/C/R each, including 228 fused long candidates. The old five-run
  partial summary was removed only after exact superset verification;
  all original run evidence remains available.
- Long coverage is H2048/4096/16384, S128K/S256K, CP4/8, both directions,
  N128 and N256/K64/e32. The first three S128K/CP8 runs test CTA8/12/16/32;
  the other nine also test CTA24. This is not a uniform full Cartesian
  sweep or held-out model validation. All 24 direction/geometry winners
  here use e32, but no per-shape winner table is installed as a selector.
- Against matched v13 policies/CTAs, including original-C CTA12 checks
  `140935-660c0b` / `140958-c82f32` / `141020-827e41`, QKV changes are small:
  N128 F ranges -1.37% to +0.75%, e32 -0.86% to +0.51%. OProj e32 improves
  8B/CTA16 from 0.494336 to 0.451008 ms and 405B/CTA16 from 9.379888 to
  8.759184 ms; corresponding C changes only 0.405968 to 0.398944 ms and
  8.673792 to 8.669248 ms. The fused improvement is not explained by the
  same-sized bare-compute improvement. These are sequential runs, not
  repeated interleaved A/B trials; small regressions are retained.
- Same-node complete-reference pairs currently exist for 11/12 QKV and
  5/12 OProj geometries: their respective partial-coverage geometric-mean
  baseline/F ratios are 1.128713x and 1.361918x, not stage completion.
  Qwen S256K/CP4 QKV and seven OProj comparisons are missing; node1 times
  do not fill them. All four 405B QKV points still lose to the stronger
  same-node baseline, and their independent C already exceeds its complete
  boundary. For S128K/CP8, best QKV F/C/R is 8.895440/8.702736/1.171296 ms
  versus UB 8.306272 ms; OProj is 8.493536/8.306960/2.581904 ms versus
  canonical UB 7.123776 ms. This prioritizes compute-path/budget diagnosis,
  without treating independent C as a strict F lower bound. The S256K/CP4
  QKV p50/p95 35.722048/37.674335 ms also retains its broad tail.

MPI measurement and broader GEMM scheduling, v16/v17 (2026-09-06):

- Optional `fused_bf16_mpi` uses one process per GPU, CUDA IPC, per-sample
  barriers and `mpi_rank_events_v1` max-rank CUDA events. CP4/8 two-payload
  CPU-oracle tails and full C/R checks passed; the full historical matrix
  is still being filled. Old single-process results remain diagnostics,
  not silently relabelled MPI measurements. Node09's 405B S512K/CP4 exceeds
  available memory with both directions' reference buffers live; its
  preflight stopped before CUDA launch. The same full case passed on node0a
  (`153930-f803e6`), where its baseline/pure-reference pairing is required.
- Pure Lt coverage now has 66 distinct MNK per node, Eager/Graph separately:
  264 accepted measurements and 13,200 raw samples (`150629-1a1ac4`,
  `152108-a0abd1`, `152423-fb7a09`, `152511-9c37d6`). This is single-GPU
  pure GEMM with bounded numerical checks, not CP max-rank fusion. Graph
  reuses the Eager-selected Lt algorithm. Extra relocated MNK are separate.
- v17 adds explicit `--max-swizzle-size` and per-direction raster arguments
  to the harness; defaults and operator APIs are unchanged. F/C use the
  same actual scheduling. Source `102a6387`, seven validated MPI runs and
  42 F/C/R rows are in `fuse_midfile/sm103-schedule-v17/summary.json` with
  a compact `comparison.md`. Requested cap, effective swizzle, padded tile
  grid and per-rank resources are checked, including small padded tails.
- The primary experiment is QKV, 405B S128K/CP8, N256/K64/e32, comm12.
  Swizzle4 gives F/C/R 7.890864/7.576144/1.169792 ms; bracket controls give
  F 8.993088/8.963312 and C 8.697968/8.678544 ms. F improves 1.136–1.140x;
  R stays about 1.17 ms. Swizzle8 is slower than4 here; AlongN regresses.
  This establishes a useful scheduling dimension, not its cache cause,
  universal dispatch choice or a phase milestone. Incidental OProj C
  improves more than F, so bare-compute gains cannot be assumed to transfer.
- Broader hypotheses were checked against fixed official BF16 sources:
  [DeepGEMM SM100](https://github.com/deepseek-ai/DeepGEMM/blob/559d79fb6994a58b8a15b4b93bf13ccc16edf247/csrc/jit_kernels/heuristics/sm100.hpp)
  and [ThunderKittens B200](https://github.com/HazyResearch/ThunderKittens/blob/be0e7e57e90858dfa2bbeab7296ff252755f8a37/kernels/gemm/bf16_b200/bf16_b200_gemm.cu).
  Our N256/N128 already have two/four accumulator stages; adding generic
  “TMEM double buffering” is not a new proposal. Next candidates include
  matched stock 1-SM/2-SM pure GEMM and measured epilogue/full-store-drain
  costs. The global-completion wait before QKV ready must remain correct.
  [Tencent HPC-Ops](https://github.com/Tencent/hpc-ops/blob/f39028d9f5ab77f71906fbf929d1b611859ab6b7/README.md)
  is mainly H20/SM90 inference, not an established B300 dense-BF16 training
  comparator. No external library's published score is treated as a local measurement.
- The optional stock CUTLASS pure-GEMM comparison is now measured, not just
  compiled (`161950-6b9d59`, `162003-625c26`, source `c4ad34f3`). It uses
  identical BF16 input/weight/output buffers for Lt and static-persistent
  1-SM/2-SM plans, two opposite execution orders, stable 100 ms warmup and
  10+50 sampling. All 42 blocks / 2,100 raw samples passed the audit in
  `fuse_midfile/sm103-cutlass-compare-v18/audited-swizzle1.md`; large numerical
  checks cover 4,096 values, not the complete matrix or a CP boundary.
  With swizzle fixed at 1 / AlongM, QKV M16384/N18432/K16384 gives Lt
  6.856/6.725 ms, 1-SM 8.223/8.246 ms and 2-SM 8.093/8.091 ms. OProj's
  corresponding square GEMM gives Lt 5.774/5.726, 1-SM 7.306/7.297 and
  2-SM 7.160/7.119 ms. These controls show only a small 2-SM improvement
  under this scheduling, not that 2-SM plus tuned locality cannot help.
  Both CUTLASS plans use all 148 SMs; neither is the fused C reference with
  reserved communication CTAs. Do not substitute these times for C in a
  same-resource overlap model or select only the faster repetition.
- Matched-source swizzle1/4 controls (`164746-768ab8`, `164834-30e492`,
  source `178a78da`) change this picture: the two long square/near-square
  GEMMs improve 1.124–1.128x with 1-SM and 1.069–1.071x with 2-SM; 1-SM
  is faster under swizzle4, though Lt still wins. The 64K held-out QKV gains
  only 1.028x/1.009x. Keep both execution-order blocks and the measured Lt
  control drift; these separate jobs are not an interleaved causal estimate.
  AB stages are 4/6 and cluster/tile geometry also differs between 1/2-SM.
  Full evidence: `sm103-cutlass-compare-v18/audited-swizzle4.{json,md}`.
- Private QKV epilogue diagnostics (`165211-da3a7a` through `165331-bf23fe`,
  node09, source `e21c3b6d`) pass CP4/8 tails and the long 405B geometry.
  Production/old-role/private instrumentation use 92/96/125 registers, zero
  reported local allocation and 214016 B shared memory. Long-case explicit
  store drain plus warp join is 0.31–0.43% of each compute CTA's own role
  duration (per-rank medians), not Tensor Core idle percentage. Base::store
  includes accumulator waiting; do not call its interval pure epilogue work.
  Swizzle4 production p50 7.795744 ms versus bracket controls 8.846208 and
  8.865040 ms confirms a scheduling gain without removing completion fences.
  The private ordered role timestamp uses a CTA popcount-result dependency;
  actual SASS confirms that dependency, not a compiler/hardware defect in
  the earlier timestamp experiment. Same-node0a reproduction remains pending.
  Evidence: `sm103-epilogue-v19/audited-v21.{json,md}`; diagnostics do not
  enter formal performance or the full-matrix milestone calculation.
- Explicit QKV `m128n256k64e64` was compiled and measured in v23 (source
  `96c53fc4`), with F and C using the same no-residual family. It retains
  AB4/TMEM2 but raises native 1-SM registers 106→148 and dynamic SMEM
  214016→231424 B; no local allocation is reported. Five pure-GEMM shapes
  show only 0.999–1.005x E32/E64 ratios. Four fused geometries, with both
  candidate orders, show no consistent benefit; 8B and 64K include regressions.
  E64 remains an explicit tested candidate, not a new default. The 405B E64
  F p95 of 11.82476 ms is retained despite passing the existing half-drift
  rule. Details: `sm103-cutlass-compare-v18/v23.{json,md}`.
- Counter-only v24b (`173724-78c25e`, `173839-85e46b`, `174056-413aed`,
  source `1419e866`, node09) isolates one warmed stock 1-SM QKV invocation
  in an NVTX range. At M16384/N18432/K16384, swizzle1→4→1 gives HBM reads
  34.576→12.535→34.575 decimal GB and L2 aggregate hit rates
  36.464→69.493→36.533%. Tensor-pipeline activity is 74.839→87.336→74.196%,
  with SM clocks 1.600→1.591→1.617 GHz. This supports better memory reuse,
  not a clock increase, as an optimization direction; it does not identify
  A versus B reuse or prove every cause of the formal speedup. NCU 2025.3
  needs three replay passes for the six requested metrics. Clock/cache
  controls stay `none`; output NaN fill before the target perturbs cache
  state, and multi-pass metrics need not describe one identical cache state.
  Profiler duration is diagnostic only, never a production performance row.
  The held-out M8192/N18432/K16384 pair (`174513-bbcc1b`, `174600-1fc506`)
  reduces HBM reads by 30.17%, versus 63.74% for the larger bracketed case.
  At M16384/N4096/K2048 (`174817-50b827`, `174905-59f7b3`), L2 read requests
  still fall 9.82% but HBM reads rise 18.28%, with essentially unchanged
  Tensor activity. Locality benefit is geometry-dependent; this smaller
  point changes both N and K, so it does not isolate K as the cause.
  These two extra pairs have no after-control bracket and do not establish
  formal speedups. Keep default scheduling unchanged until a general
  selection rule passes production timing and held-out validation.
  All seven counter runs: `sm103-cutlass-compare-v18/v24b-counters.{json,md}`.
- MPI Eager v25 (source `6d323401`, node09) records matched F/C/R for
  N128 and N256/E32, swizzle1/4 and explicit communication budgets. Reducing
  compute CTAs 140→116 increases large-QKV C latency only 7.54–9.63%, versus
  20.69% from constant per-CTA throughput; that inverse-budget assumption is
  not an adequate calibrated model. With swizzle4, halving M nearly halves
  C, and halving K at fixed M/N gives 0.509–0.516 of C. C/R are independent
  reference totals, not measured Tensor Core service or stall intervals.
  A fixed empirical C-latency candidate `L + (2MNK/1e12)*(A/compute_ctas+B)`
  predicts four candidates at a new geometry within 0.893%, without refitting.
  It still misses a small-N/K training point by about 9%; its intercept and
  shared-resource term are not independently measured launch/HBM times.
  Validation uses identical source/build inputs/environment after an ELF
  rebuild, not the same binary hash. No runtime selector/default changes.
  Evidence: `sm103-cutlass-compare-v18/primitive-v25/`.
- Native v26 (source `a2a24e6a`, node09) tests explicit 1-SM MMA with
  cluster M2, separately from 2-SM MMA. All small/odd-tail/multi-wave outputs
  are fully checked. Actual AB4/TMEM2, 214016 B SMEM and 256 threads remain;
  registers are 99 versus cluster1's 106, with 74 resident clusters reported.
  At requested swizzle4, the four principal large matrices regress about
  5–7% against both cluster1 bracket runs. This configuration is not promoted
  to a default; equal swizzle requests do not imply equal physical tile
  grouping across cluster sizes. Full evidence: `sm103-cutlass-compare-v18/v26.{json,md}`.

## Layout and scope

Both architecture backends use this private structure:

```text
csrc/operators/{sm90,sm103}/
  entry.cu
  api/{forward,policy}.cuh
  detail/{gemm,cutlass_pipeline,persistent_gemm,a2a_gemm,gemm_a2a,launch}.cuh
```

SM90 retains its real backward, heterogeneous, reference and Hopper scheduling
extensions. They are not duplicated as empty SM103 files. Public headers in
`include/fuse/operators` remain architecture-neutral; CMake selects exactly one
backend. Precision remains explicit through the existing BF16/FP8 parameter
types and entry points, without a new quantization framework.

The file responsibilities and namespaces match SM90:

| File | Responsibility |
|---|---|
| `detail/gemm.cuh` | Geometry, scalar helpers and CUTLASS GEMM types |
| `detail/cutlass_pipeline.cuh` | Mainloop acquire and epilogue publication adapters |
| `detail/persistent_gemm.cuh` | Scheduler, CTA roles, residency and cooperative launch |
| `detail/a2a_gemm.cuh` | Input routing, copy and arrival publication |
| `detail/gemm_a2a.cuh` | Output routing, copy and cross-rank completion |
| `detail/launch.cuh` | Concrete bindings, device queries and argument assembly |
| `api/{forward,policy}.cuh` | Public launches and supported trait queries |

Control templates use `fuse::detail`; private operator code uses `namespace
fuse { namespace { ... } }`. The suffix `T` means a template on both backends:
`A2ALhsInputCommT` / `A2ALhsInputComm` and `QkvGqaPackCommT` / `QkvGqaPackComm`.
A2A copy paths stay in `operator()`; QKV paths stay in `run(wait_for_gemm)`,
following the respective SM90 implementations.

The SM103 library implements `launch_a2a_gemm_cutlass` (inverse head-to-sequence
A2A then output projection) and `launch_gemm_a2a_cutlass` (QKV/GQA projection
then forward A2A). Inputs, weights and outputs are BF16; accumulation is FP32.
Role telemetry is also implemented under `FUSE_ENABLE_PROFILING`. SM90-only
FP8, backward, reference, automatic policy selection and Heterogeneous weighted
CP planning are not implemented here. Common declarations do not imply support
for those additional capabilities.

## Execution and tuning

- CUTLASS `arch::Sm100` BF16 collective, compiled for **sm_103a**, uses Blackwell
  `tcgen05` MMA and TMEM. `Sm100` here names the minimum collective architecture,
  not the target binary. No Hopper WGMMA collective is linked into this backend.
- GEMM+A2A has five BF16 tiles: 128x{64,128,160,192,256}x64. A2A+GEMM
  has 128x{128,256}x64. All use one-SM MMA, 1x1x1 clusters and 256 threads per CTA.
  Among legacy unsuffixed policies, N160 uses an explicit 128x32 epilogue
  tile and the other widths use CUTLASS Auto. Explicit K/epilogue variants
  retain separate policy identities; the measured QKV E64 candidate is not
  selected by default.
  Communication uses four warp slots for A2A input pulls and eight for QKV
  output routing. CUTLASS's `SM90_*` TMA/bulk-copy wrapper names denote copy
  instructions also supported on Blackwell, not Hopper GEMM instructions.
  Both comm types expose local `SM100_*` aliases for readability; the aliases
  bind to those same CUTLASS types without changing the generated operation.
- One cooperative kernel contains a communication-CTA prefix and persistent
  compute CTAs. The static scheduler subtracts that prefix and advances by the
  compute grid size. There is no separate communication kernel/GEMM boundary.
- `num_comm_ctas` is mandatory and positive, below the device SM count; **0 does
  not enable automatic tuning**. No power-of-two restriction is imposed on it.
  The runtime queries the actual mixed kernel's occupancy, reserves enough
  shared memory for at most one CTA per SM, and requires the whole grid resident.
  Thus a communication CTA corresponds to a dedicated SM during its execution.
- `lhs_policy` must remain `kAuto`: it selects the SM103 A2A+GEMM binding, not
  SM90's automatic model. Hopper-specific A2A tile requests are rejected. Both
  `recommended_*_comm_ctas` queries return 0 to mean no recommendation.
  Raster and swizzle remain explicit GEMM arguments; the default is AlongN
  for A2A+GEMM and AlongM for GEMM+A2A.
- Traits report the actual dynamic shared-memory reservation after device
  lookup, including occupancy padding. A zero SMEM field means lookup failed
  or no supported device is available. Launch still returns a CUDA error.

For GEMM+A2A, `FUSE_QKV_GEMM_POLICY=m128n64|m128n128|m128n160|m128n192|m128n256`
explicitly selects a compile-time binding. An unset value or `auto` means fixed N128,
not a search or a performance-model decision. Unknown requests are rejected;
interleaved QKV uses the N128 specialization. Keep the selection unchanged from
the selected-geometry query through all rank launches and buffer reuse.

For A2A+GEMM, `FUSE_SM103_OPROJ_POLICY=auto|m128n128|m128n256` selects a
private one-SM binding (default/auto N128), never Hopper's ClusterM2 policy.
The mainloop, epilogue, communication window, production/profile launch and
resource query all use the selected tile. M128/K64 and the ready allocation
are unchanged across these two N widths. The harness has independent
`--qkv-policy-list` / `--oproj-policy-list`: each direction expands only its
own tile x CTA candidates, not a cross-product of both directions' tiles.

N256 A2A build `20260906-113326-5a25f4` passed in 57.0 s. Tail diagnostic
`20260906-113629-c55b85` passed CP8 causal M258/H1032/Q16/KV8/D128, both
N128/N256 and CTA8/16, two payloads and full CPU/GPU checks. Both M and
OProj N have partial final tiles. Thirteen injected errors were detected and
restored. This diagnostic does not benchmark; long performance and N256
macro-instrumented resource/record validation remain separate steps.

Finite tile/CTA calibration, production source `cdfd1c92`, Qwen geometry CP8:
`20260906-113813-17d787` (S128K, 57.7 s) and `113931-e6599e` (S256K,
70.4 s) each passed all 42 candidates across two payloads. Candidates are five
QKV tiles and two OProj tiles, each with CTA8/12/16/24/32/48. No direction
cross-product, shape winner lookup, model fitting or node1 search expansion.

| Global S | Direction | Best observed tile / CTA | p50 / p95 ms |
|---|---|---|---|
| 128K | QKV | N128 / 24 | 0.355424 / 0.360048 |
| 128K | OProj | N256 / 32 | 0.167408 / 0.173816 |
| 256K | QKV | N256 / 24 | 0.711616 / 0.717490 |
| 256K | OProj | N256 / 32 | 0.309792 / 0.313306 |

At 128K, the QKV point is 1.597x the earlier same-node UB p50; this is still
an individually selected point, not a model-selected/full-matrix milestone.
OProj N256 improves over N128 at CTA32 by 1.163x at 128K and about 1.159x
at 256K. Wider QKV tiles are not uniformly better, nor is a larger CTA budget.
The table records experiments; it must never become runtime shape dispatch.

N256 macro profiling also passed actual compilation (`114121-54380e`) and
both complete diagnostic runs (`114239-089138`, CTA24; `114327-dd7afe`, CTA32).
Both use S128K/CP8, QKV N128 and OProj N256. N256 resource query reports
256 threads, production/profile registers 172/174, dynamic SMEM 215040 B,
cluster size 1; all CTA/peer records and full numerical/routes passed.
Within each GPU, QKV computation/communication role ends relative to its
first CTA were 228.8--235.1 / 253.2--257.6 us at CTA24, versus
246.5--262.0 / 258.4--274.2 us at CTA32. A2A communication ends improved from
179.3--185.7 us to 140.9--146.2 us, while compute ends changed from
191.5--205.3 us to 177.6--181.5 us. These are overlapped role spans, not pure
compute/copy costs or exact ready-wait durations.

The CTA24 QKV diagnostic has rank0 local communication done at 257.6 us but
total span 339.6 us, versus rank7 253.2/258.1 us. Host launch APIs took
8.0--10.5 us/rank; launches are sequential. Host launch skew is therefore a
testable candidate explanation for the early-rank completion tail, not an
established exclusive cause. A parallel-host-launch experiment is planned;
no cross-GPU absolute timer subtraction is used in this observation.

Host enqueue experiment, source `9c2d3175`, actual build
`20260906-120811-e4ddfb` (9.3 s): explicit `--host-launch per_gpu_thread`
uses one persistent host worker per GPU. Sequential remains the default;
both modes share the same per-rank start-event/launch/end-event seam and
full max-rank boundary. The main thread waits only after every worker has
successfully enqueued its current end event. Mailbox generations are separate
from GPU epochs; partial enqueue errors fatal-exit without a blocking join.
No CPU affinity, global driver setting or SM90 changes are involved.

CP8 causal and CP4 rank-major tail diagnostics (`120939-a7cd84`,
`121027-97648b`) each passed eight candidates, two payloads, full CPU/GPU
checks and all 13 corruption/restoration cases. Long A/B uses the same binary,
Philox seeds/inputs, two tiles per direction and CTA24/32, eight candidates
per run. S128K ran sequential then parallel (`121108-5e88e9`, `121156-9bef44`);
S256K reversed the order (`121244-6fff6b`, `121332-6e86de`). All passed.

| Global S | Fixed direction / tile / CTA | Sequential p50 ms | Parallel p50 ms |
|---|---|---:|---:|
| 128K | QKV / N128 / 24 | 0.361168 | 0.325024 |
| 128K | OProj / N256 / 32 | 0.168704 | 0.180464 |
| 256K | QKV / N256 / 24 | 0.719104 | 0.673952 |
| 256K | OProj / N256 / 32 | 0.309600 | 0.318832 |

Parallel enqueue improved all four tested QKV configurations by 9.9--11.4%
at 128K and 3.0--7.3% at 256K, but OProj became slower by 4.0--7.0% and
2.3--3.1%, respectively. These are measured boundary changes, not proof of
more NVLink bandwidth. Do not silently mix launch modes or pick a mode per
shape/direction to inflate a comparison. CPU enqueue skew and GPU-local
communication/finalize spans require the pending macro-profile A/B for
attribution; no claim that the entire previously observed tail disappeared.

Macro-profile A/B compiled in `121540-6855f0` (72.7 s), then passed both
S128K CP8 CTA24 QKV-N128/OProj-N256 runs: `121658-daa4d4` sequential and
`121746-f0fd44` parallel. In the production diagnostic dispatch, QKV CPU
API-start skew fell from 87.582 to 1.921 us, while per-rank API duration
increased from 8.476--10.226 to 35.946--48.088 us. GPU-local compute role
ends stayed 229.7--236.8 versus 230.5--236.7 us. QKV rank0's finalize tail
after local grid completion fell from 81.248 to 3.872 us; the maximum
parallel-rank tail was still 30.56 us. These measurements support launch
skew as one contributor, not an exclusive bottleneck or fabric-speed claim.
OProj host API time rose from 5.064--6.652 to 16.522--22.978 us. Its local
communication/compute ends were 175.4--188.6 / 192.3--209.0 us sequential,
181.95--190.50 / 195.3--208.9 us parallel. The call-internal source of the
host increase is not yet measured; descriptor work or driver serialization
must remain hypotheses until separately instrumented.

The initial communication defaults follow the corresponding Hopper bindings:

| GEMM N | Vector M tiles/task | Copy N |
|---|---:|---:|
| 64 | 4 | 128 |
| 128 | 4 | 128 |
| 160 | 4 | 160 |
| 192 | 4 | 192 |
| 256 | 1 | 256 |

These are compile-time pairings, not separate copies of the communication
algorithm. `Bf16GemmTypes<N>` generates CUTLASS types/parameters; the binding
checks that producer-ready geometry matches the communication consumer.
TODO: Tune the tile and task defaults using complete B300 GEMM+A2A measurements.
No Hopper calibration table, cost model or model-based autotuning is imported.
All five QKV tiles and both OProj tiles have now compiled and passed the finite
BF16 validation sweeps above; that is not coverage of other precision families.
Performance-model autotuning remains a required feature of this operator library,
not an abandoned capability. TODO: collect independent B300 compute/route costs,
calibrate a Blackwell model and connect joint tile/communication-SM selection.
This first version keeps the manual path for baseline measurement and review.

Planned model-data split, fixed before fitting: use Qwen dense, Llama3-8B and
Llama3.1-405B geometries at global S128K/S256K, CP4/8 for calibration. Keep
other historical model geometries, S64K/S512K, and short-sequence regression
cases out of coefficient fitting. Existing micro/debug timings are diagnostic,
not calibration rows unless they satisfy the current measurement contract.
Model inputs will be dimensions, tile/resource geometry and communication
volume, never model names or a per-shape winner lookup. Held-out results must
report prediction error, selection regret against measured candidates, and
strong-baseline coverage separately. This defines a data split, not a chosen
cost formula or a claim that a fitted policy exists yet.

Independent BF16 calibration APIs compiled in `120440-5bfebb` (63.9 s) and
the unchanged production path passed the CP8 tail regression
`120636-8611f8` (19.5 s). These facts do **not** validate execution of the
new reference APIs yet. They use the selected tile and compute budget
`SM_count - reserved_comm_ctas`; route-only QKV retains full cross-rank
finalization, while A2A route-only retains real arrival publication.
Callers must use separate calibration ready/done buffers and communication
epochs: pure compute must never advance a cumulative communication counter.
Preparation and counter resets stay outside samples. Fully materialized C and
unbackpressured R differ from F in readiness/cache/contention; neither their
sum nor `max(C,R)` is an established fusion model or strict bound.

Actual C/R execution and sampler repair:

- The first tail calibration `122339-234b90` failed the unchanged 5 s
  warmup deadline during OProj pure compute, not numerical/route correctness
  or OOM. Approximately 4,742 short calls accumulated only 98--106 ms per
  GPU: the old `wait_all` slept 1 ms whenever any end event was pending.
  This failed run is retained and cannot produce accepted performance rows.
- Build `123005-065234` (9.4 s) changes completion waiting only: after all
  ranks record their **current** end event, the main thread waits on those
  CUDA events directly. The process SIGALRM watchdog remains; no device
  reset, extra GPU barrier or reduced warmup/sample threshold. The collector
  is explicitly `per_epoch_rank_events_v3_eventsync`; do not merge it with
  v2 for calibration or assume unchanged GPU duty cycle.
- `123147-ad9517` (CP8 parallel, M258/H1032 causal, 21.8 s) and
  `123236-8eb3a2` (CP4 sequential, M257/H1032 rank-major, 14.8 s) passed
  all four tile/CTA candidates x F/C/R, full two-payload verification and
  32 complete CPU/GPU oracle checks each. Each of the 12 component timings
  retained converged 100 ms warmup, 10 cadence warmups and 50 samples.
  Maximum warmup wall time was 0.577/0.353 s. Actual compute/copy APIs are
  now exercised, including independent arrival/done counters; these small
  tail diagnostics are not part of the pre-registered long-shape fit.

The first two long calibration runs, `123410-4510b3` and
`123457-d18ce8`, passed all 42 configurations x F/C/R at Qwen CP8
S128K/S256K. Source `cce45a17`, production binary `2a865f1f`, parallel
host enqueue and the v3 event collector are fixed for this calibration set.
The local fused summarizer independently checked the archive, both payloads,
all ranks and 50-sample percentiles (252 accepted component records).

| Global S | Finite-grid fused winner | F / C / R, us |
|---|---|---:|
| 128K | QKV N128, CTA24 | 319.088 / 249.008 / 265.952 |
| 256K | QKV N256, CTA24 | 688.832 / 581.968 / 485.328 |
| 128K | OProj N256, CTA32 | 181.280 / 139.104 / 158.608 |
| 256K | OProj N256, CTA32 | 322.640 / 255.584 / 288.944 |

These measurements reject a naive component-winner policy: at OProj CTA32,
N128 -> N256 lowers F by 9.64% / 13.58%, although **both** independent C
and R become slightly slower. Selecting `min max(C,R)` instead of measured
F loses 6.62--30.06% across these four direction/sequence points. QKV R
also gets slower from CTA16 to CTA24 in all five tiles at both lengths,
while F prefers CTA24. This is evidence against a simple monotone
bytes/CTA route model, not proof of the cause. Profile input-ready,
compute/communication frontiers and finalization before assigning the
interaction to window size, bandwidth or resource contention. These are
finite-grid calibration observations, not a fitted policy or a full-matrix
strong-baseline milestone.

`kA2ALhsCommRows=32` is used by the vector fallback and determines its arrival
count. Bulk A2A chooses rows dynamically from the 48 KiB stage and peer-row size.
`kQkvBulkRows=64` with 128 columns defines each TMA transfer, its shared-memory
stage and task grid. These are active communication constants, not GEMM tiles
or unused tuning remnants.

### Precision extension boundary

SM90 and the common parameter headers are unchanged by this extension. SM103
keeps an explicitly named BF16 GEMM family; additional precision families must
provide their actual CUTLASS operand/accumulator/epilogue contract when implemented,
instead of adding `IsFp8`, `IsMxfp8`, etc. to this BF16 builder.

GEMM input precision and routed output format are separate: a block-scaled GEMM
with BF16 output can still use BF16 output communication. If the routed payload
itself is block-scaled, its scale tensors, block layout and readiness/completion
must be implemented together with the data path. A scalar element type, TMA
datatype or `alpha` cannot express that contract. No unimplemented FP8/MXFP8 or
other-precision bindings are declared as supported here.

Only device attributes and kernel resource reservations are cached. Tensor
maps, pointers, epochs and alpha are rebuilt from each launch's arguments;
there is no stale-pointer plan cache. As with other CUDA launch-resource
caches, callers must not reset/recreate the CUDA context during their lifetime.

## Tensor and synchronization contract

The public parameter structs retain their SM90 meanings:

- `gemm.l == 1`, `gemm.m == batch * seq_local`. Flatten the batch into M;
  independent batched GEMMs are not silently reinterpreted. `rhs_nt` is physical
  row-major `[N,K]`; A and D have unit inner stride, aligned row strides and
  16-byte-aligned addresses. Input staging for A2A is packed `[M,K]`.
- Uniform CP groups of up to eight peers are supported, including CP4/8.
  `global_seq == seq_local * world_size`; GQA head splits must be even across
  peers. OProj's K shard per peer is a multiple of 64. P2P-accessible input,
  output and control pointers must already be mapped by the caller.
- QKV output rows remain **rank-major**, including when the causal flag is set,
  matching SM90's forward QKV benchmark. OProj supports standard and causal
  two-chunk gathers; its causal local sequence length must be even.
  OProj `cyclic_peer_order` requires the caller
  to prepack **the weight K axis in exactly the same rank-local cyclic order**,
  as stated in the existing public route header. No online weight permutation
  is performed. Packed-varlen and extra channel modes are explicitly rejected,
  not silently ignored. Interleaved QKV input to the communication stage is
  supported; as in SM90, interleaved plus AlongN raster is rejected.
- Ordinary GEMM+A2A writes Q, K and V as **three contiguous tensors** in each
  `peer_output`: segment shape `[batch * global_seq, local_segment_width]`.
  It is not row-interleaved QKV. Communication waits for every producer M/N
  tile its copy spans. The issuing epilogue warp fully drains TMA destination
  writes before releasing the tile epoch; `.read` completion alone is not enough.
- With `defer_v`, only Q/K are routed in SM90's row-interleaved Q/K layout;
  the kernel also waits for all local V producer flags before optional
  `completion_epoch` publication. That flag alone does not mean all Q/K routing
  has finished; grid/finalize completion provides that guarantee. This mode
  does not promise received remote V.
- QKV `ready` needs `ceil(M/tile_m) * ceil(N/tile_n) * kReadyFlagStride` uint32
  entries for the selected binding. The problem-only traits query returns the
  finest N64 geometry for conservative allocation; the route/comm-aware query
  returns the selected geometry and must be used for profiling interpretation.
  Each rank's `peer_route_done_epoch` allocation needs
  `world_size * kReadyFlagStride` entries. Initialize both to zero. After the
  local cooperative grid sync, send-completion epochs are published to every
  destination; the kernel waits for every source's completion before returning.
  **Enqueue all participating ranks before waiting for any rank to finish.**
- A2A `ready` sizing uses `a2a_lhs_gemm_ready_elements`. Its flags are cumulative
  arrival counters indexed by `(M tile, K peer)`, not overwritten tile epochs.
  Even empty tail chunks publish their expected arrival. The GEMM TMA producer
  acquires each peer's complete arrivals before loading that K shard, with an
  async-proxy fence between the acquire and TMA.
- Begin at epoch 1 with zeroed flags, then use consecutive epochs with unchanged
  geometry/arrival count. Use the same epoch on all ranks; do not overlap launches
  sharing the same staging/output/control storage. Before overflow or changing
  shape/layout, wait for all ranks, reset the flags and restart at epoch 1.
  A2A cumulative counter overflow is rejected; the caller owns epoch continuity.
- Optional A2A `input_epoch` waits on each `peer_input_ready` using system scope.
  Without it, peer input must already be complete. OProj return only guarantees
  this rank's pulls/GEMM are complete: wait for **all ranks** before overwriting
  shared peer input. Ordinary QKV return additionally includes all received
  Q/K/V; deferred V retains the distinct completion contract above.

**Repeated replay of a graph containing one fixed epoch is not supported.**
Old ready/completion flags can satisfy a later replay before its new producer
finishes. Use eager launches with increasing epochs for this first version.
A future replay-safe protocol requires its own implementation and validation;
existing cuBLASLt/TE graph results do not establish fused graph correctness.

## Profiling

`FUSE_ENABLE_PROFILING=ON` adds the existing public diagnostic interfaces:
`launch_a2a_gemm_cutlass_role_telemetry`, `launch_gemm_a2a_role_telemetry` and
`query_a2a_gemm_role_resources`. No new architecture-specific public ABI is
introduced. The production and instrumented kernels are separate template
instances; timer operations are excluded from production paths. Register/SASS
equivalence and actual resource values still require CUDA compilation.

The harness warms the instrumented specialization for ten epochs before
recording a diagnostic timeline. It also records host launch-API duration
(argument setup plus enqueue) and CUDA-event duration for one production and
one instrumented epoch. These diagnostic samples do not replace the 50-sample
production distribution. All output is delayed until every rank is enqueued.

- Allocate at least the device SM count of `A2AGemmCtaTimeline` records.
  Clear diagnostic arrays before **each diagnostic launch**, independently of
  ready flags; continue the normal next epoch, not an arbitrary reset.
- A2A peer telemetry is optional. When supplied, its capacity must cover both
  `M_tiles * world_size` and `L * M_tiles * N_tiles` (currently `L == 1`).
  Publication/communication fields use `[M, peer]`; acquire/tile metadata use
  `[L, M, N]` in disjoint fields of the same allocation. It is not one unified
  index space that can be zipped directly by record number.
- The mainloop TMA producer (warp 2/lane 0) records peer acquires, including
  cached-ready observations on later N tiles. First-write atomics preserve the
  first acquire across prologue/remainder calls. CTA `active_start` is the first
  input-ready observation, not an MMA instruction timestamp.
- A2A communication records copy stages and the last arrival's publication.
  `release` is sampled **after** publishing the flag, so a fast consumer can
  have an earlier `acquire` timestamp. Do not interpret that difference alone
  as negative transport latency. Timers are comparable across SMs on one GPU,
  not assumed synchronized across GPUs.
- QKV records CTA role completion, grid synchronization and CTA0's system
  fence/publication/per-source completion phases. Diagnostic CTA end stamps
  follow full-CTA rejoin, including communication's otherwise inactive warps.
- The resource query uses the actual outer kernel and padded launch SMEM.
  The legacy `compute_active_warps` field reports the **physical** eight-warp
  CTA budget, not measured active warps; some CUTLASS roles can be idle.

External ncu/nsys remains useful, but does not replace these kernel events.

## Checks and later node2 execution

Local checks, without CUDA or a cluster connection:

```bash
python3 -m unittest scripts.test_sm103_fusion_contract scripts.test_sm103_fusion_routes -v
bash -n scripts/build.sh scripts/build_sm103_bench.sh
git diff --check
```

The CPU tests cover scheduler/arrival/copy arithmetic and compile/execute the
real public C++ head-mapping header. They do not run the CUDA implementation or
prove GPU memory ordering. The SM90 move is separately checked against Git:
the twelve private implementation/scheduling files are unchanged apart from
documented comments, and entry-point changes are include paths only.

After source review approval, on **node2 only**, with an explicit free-memory
and utilization check and the existing workspace lock:

```bash
source /root/workspace_wct/env.sh
cd /root/workspace_wct/fuse
FUSE_ARCH=sm103 BUILD_DIR="$PWD/build/sm103-fused" \
  CUTLASS_ROOT=/root/workspace_wct/deps/cutlass-57e3cfb47a2d9e0d46eb6335c3dc411498efa198 \
  bash scripts/build.sh
build/sm103-fused/fused_bf16 --world 4 --comm-sm 8 --seq-local 256
build/sm103-fused/fused_bf16 --world 8 --comm-sm 8 --seq-local 256
build/sm103-fused/fused_bf16 --world 8 --comm-sm 12 --seq-local 258 --causal
# Explicit tile comparison, not a model-based autotuner:
for qkv_policy in m128n64 m128n128 m128n160 m128n192 m128n256; do
  FUSE_QKV_GEMM_POLICY="$qkv_policy" \
    build/sm103-fused/fused_bf16 --world 8 --comm-sm 8 --seq-local 256
done
```

Build the diagnostic variant separately, after the same review approval:

```bash
cmake -S . -B build/sm103-fused-profile -G Ninja \
  -DFUSE_ARCH=sm103 -DFUSE_BUILD_KERNELS=ON -DFUSE_BUILD_BASELINES=OFF \
  -DFUSE_ENABLE_PROFILING=ON -DCMAKE_BUILD_TYPE=Release \
  -DCUTLASS_ROOT=/root/workspace_wct/deps/cutlass-57e3cfb47a2d9e0d46eb6335c3dc411498efa198
cmake --build build/sm103-fused-profile --parallel 8
build/sm103-fused-profile/fused_bf16 --world 8 --comm-sm 8 --profile
```

`fused_bf16` is a small two-direction correctness/microbenchmark, not the
full strong-baseline sweep. It uses deterministic random data, at least 10
warmups and 50 samples, max-rank CUDA-event latency, and a changed-input check
for stale flags. Cross-rank launch skew and host setup can affect these early
measurements; report raw samples and do not infer production speedup from them.
Each run prints its requested QKV policy and resolved per-rank tile/resources
before timing, so results from different tile candidates remain identifiable.
The default geometry is Q=32, KV=8, head dimension=128 and hidden=1024;
`--hidden`, `--q-heads`, `--kv-heads`, `--head-dim` and either `--global-seq`
or `--seq-local` select other valid CP4/8 geometries. Default CPU input
preparation has shape-dependent host costs; explicit `--input-generator
gpu_philox` uses bounded GPU generation/statistics and an independent GPU
reference gather. Full validation stays outside the timed GPU boundary.
`--causal` exercises OProj's two-chunk gather, not a different QKV output order.
`--profile` adds a separately instrumented epoch, prints the raw CTA/peer events
and checks both numerical/route correctness again. The watchdog defaults to
60 seconds (`--timeout-seconds` allows longer full validations) and exits
only this benchmark process on a hang; it never resets GPUs or stops peers'
unrelated applications. Interleaved/defer-V/cyclic-weight modes are not exercised
by this small harness and still require separate device validation.
The separate `build_sm103_bench.sh` explicitly disables fused-kernel builds,
so reference-benchmark maintenance does not accidentally test unreviewed code.

## Coordinated producer / ready / copy order (2026-09-07)

`detail/producer_consumer.cuh` is the shared one-CTA-cluster ordering contract.
`PublishedTile<M,N>` describes the whole output tile whose completed GMEM stores
allow one ready release. It is not the MMA instruction or epilogue subtile.
The producer and consumer use `ProducerTileOrder` with CUTLASS-resolved padding,
raster and swizzle. The static persistent compute stride and GEMM math do not change.

The 64x128 TMA route enumerates copy rectangles at their latest logical ready
dependency, rather than assigning a permanent destination peer to each CTA.
Each rectangle is owned once; overlapping N64/160/192 publication tiles and
padding create skipped candidate slots. The destination peer, local head and
Q/K/V segment are decoded from the physical source column. Existing N128
peer-interleaved and deferred-V mappings retain their address semantics.
The non-TMA vector fallback keeps its existing task order in this revision.

Concurrency constraints:

- A later logical tile is not necessarily completed later. Acquire **all**
  intersecting ready flags, not just the dependency that owns the copy task.
- No global sorted-completion barrier, shared work queue or new ready counters.
  Each warp owns a disjoint static sequence of candidate slots. Completion
  reordering is harmless because output rectangles do not overlap.
- Preserve async-proxy ordering, SMEM-read completion before stage reuse, full
  destination-write drain, cooperative join and source-complete epoch protocol.
- No bounded ready lookahead yet. Add one only if new traces show residual
  out-of-order producer completion blocking useful work; measure polling cost
  and final max-rank boundary time rather than summing overlapping waits.

Changing ready granularity independently of the whole GEMM output tile requires
a new publication protocol; this implementation does not pretend that changing
a template constant would safely aggregate or subdivide releases. There are no
model-name/sequence-length rules and no new precision switches.

CPU checks: `scripts/test_sm103_producer_consumer.py` host-compiles the actual
arithmetic and checks N64/128/160/192/256, both rasters, swizzle1/2/4/8, tails,
complete unique copy coverage, dependency ownership and comm CTA1/3/7/16/32/147.
These are mapping proofs against an independent enumeration, not GPU timings.

Device checks on node09: CP8 comm CTA7/16 and CP4 CTA13, all five N tiles,
AlongM/swizzle4 and AlongN/swizzle8, passed full numeric/route checks for two
payload generations (`20260907-172414-214c4c`, `20260907-172456-7c41f3`).
N160/comm7 full profiling also verified skipped slots and exporter coverage
(`20260907-173448-b4658c`). SM90 was not changed.

Fixed-config CP8 Graph comparison, MPI max-rank p50, profiling OFF, random
Philox input and at least 10+50 samples. Both revisions use N256/K64/e32,
AlongM/swizzle4; Dense has 32 communication CTAs, Qwen25 has 16.
These are one paired run per point, not an independently repeated speedup claim.

| Model | Global S | Before ms | After ms | Before / after |
|---|---:|---:|---:|---:|
| production_qwen_dense | 131072 | 0.269904 | 0.272768 | 0.9895x |
| production_qwen_dense | 262144 | 0.523632 | 0.524304 | 0.9987x |
| production_qwen_dense | 524288 | 1.037232 | 1.043408 | 0.9941x |
| qwen25_14b_32b | 131072 | 1.028368 | 0.968640 | 1.0617x |
| qwen25_14b_32b | 262144 | 2.042976 | 1.932000 | 1.0574x |
| qwen25_14b_32b | 524288 | 4.070144 | 3.871584 | 1.0513x |

Production build before/after: `20260907-171443-d2fb26` /
`20260907-172224-474d80`. Corresponding before run IDs, in table order:
`171621-0c9dee`, `171645-796356`, `171707-eb4acf`, `171731-c30526`,
`171754-726398`, `171817-ac9a01`; after: `172520-98a1cd`, `172543-02a49e`,
`172606-631599`, `172633-bf4b34`, `172655-b4761a`, `172722-89493c`.
All abbreviated IDs have prefix `20260907-`. The last before point's sample
half drift was 4.52%; all passed the existing 5% rule, but this limits the
strength of a single-pair attribution. Dense end-to-end time did not improve.

Separate full diagnostic traces show the expected ordering change. A
conservative observed HOL event below means: a ready wait lasts >5 us and
another copy later in that same warp's queue depends on a tile already observed
ready before this wait started. These six N256 cases have one ready dependency
per copy. Each GPU uses its own time origin; counts aggregate eight GPUs, not
elapsed time. First-wait p50 is across all route warps on all eight GPUs.

| Model / S | First wait p50 us, before -> after | Observed HOL, before -> after |
|---|---:|---:|
| Dense / 131072 | 44.016 -> 21.248 | 6287 -> 0 |
| Dense / 262144 | 68.016 -> 22.144 | 13783 -> 0 |
| Dense / 524288 | 121.568 -> 21.792 | 28374 -> 4 |
| Qwen25 / 131072 | 199.824 -> 47.008 | 2552 -> 33 |
| Qwen25 / 262144 | 347.824 -> 42.816 | 5241 -> 97 |
| Qwen25 / 524288 | 680.096 -> 39.520 | 12276 -> 123 |

The remaining events are consistent with concurrent producer completion not
following logical tile order; they are not proof that another scheduling
mechanism would improve the end-to-end boundary. Do not sum overlapping waits
or treat diagnostic kernel timings as production throughput. Latest six traces
remain at `fuse_midfile/qkv-profile/`, with each role followed by its own warps.
Their provenance includes the immutable raw artifact hash and run/config IDs.

### Experimental rank-dependent N-band rotation

`FUSE_SM103_QKV_RANK_SWIZZLE=ON` rotates padded N bands by
`source_rank % band_count`, retaining the resolved group-local swizzle. GEMM
decode and the copy queue's dependency inverse share `NBandSwizzle`; actual
head destinations, physical ready addresses and release/acquire/drain semantics
are unchanged. This tests whether rank-synchronous destination concentration
affects the full boundary. It does not establish that NVLink congestion is the
dominant cause of a long TMA interval. The non-TMA path is not rotated.

The option defaults OFF and changes no SM90 code or public parameter layout.
For the workspace controller, add `--qkv-rank-swizzle` to BOTH `fused-build`
and `fused-smoke`. It uses a separate `*-rank-swizzle` build directory, records
the option in the executable's config and build receipt, and is rejected for
profiling tasks until rotated diagnostic ownership has been integrated.

The first six CP8 pairs used MPI Graph, profiling OFF, full two-generation
numeric/route validation, Philox input, 10 warmups and 50 samples, N256/K64/E32,
AlongM/swizzle4; Dense used 32 comm CTAs and Qwen25 used 16. Node09 observed
speedups of 1.0362/1.0179/1.0285x (Dense 128K/256K/512K) and
1.0154/1.0108/1.0206x (Qwen25); geometric mean 1.0215x. These are single
same-node A/B pairs, not repeated trials or a universal performance guarantee.
The complete 95/96-point comparison has geometric mean 1.0061x overall and
1.0129x for the 47/48 long-sequence points, with both gains and regressions.
The option therefore remains OFF by default. Configuration, paired samples
and evidence hashes are archived in [v15.0 results](../../../results/sm103/v15.0/README.md).
