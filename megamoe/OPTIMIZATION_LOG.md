# SM90 MegaMoE CTASP optimization log

Last updated: 2026-08-19

This is the append-only performance journal for the H200-oriented prefill / training
forward path.  Decode is intentionally out of scope.  Every material optimization,
including rejected variants, records correctness, resource use, latency, and the
keep / revert decision.

## Fixed test point and lower bounds

- Hardware target: H200 / SM90, 132 SMs, 8 ranks.  The current host reports an
  L20X/GH100-compatible device; final H200 qualification must repeat these runs.
- Main shape: hidden=7168, intermediate=2048, experts=256, top-k=8,
  1024 tokens/rank, about 8160 routed rows/rank.
- Tuned original DeepGEMM SM90 golden at the main shape: **1578.2 us**
  (ping-pong, experts-per-wave=32, N-major=0, default 7 stages, 132 SMs).
- Current ideal two-monolithic-cuBLASLt-GEMM lower bound at routed M=8192:
  FC1 **297.552 us**, FC2 **170.784 us**, sum **468.352 us**.  This deliberately
  excludes routing, SwiGLU, quantization, ring traffic, scatter and combine.
- Current accepted CTASP V22 at the standalone M=1024 point: **1634.6 us**
  (descriptor hoist plus four TMA stages), **56.4 us / 3.57%** behind the tuned
  WASP golden and 3.49x the deliberately optimistic two-GEMM lower bound.

Tuned / best-observed shape baseline for the original WASP implementation:

| tokens/rank | WASP latency |
|---:|---:|
| 256 | 595.0 us (ping-pong, default EPW, N-major=1, default 7 stages, 132 SMs) |
| 1024 | 1578.2 us (ping-pong, EPW=32, N-major=0, default 7 stages, 132 SMs) |
| 4096 | 5064.2 us best observed (cooperative, EPW=16); sweep stopped here |

## Optimization history

| ID | Material change | M=1024 | Resource / correctness | Decision |
|---|---|---:|---|---|
| V1 | First CTASP: all-CTA dispatch, fixed FC1/FC2 CTA pools, FC1 global ring, phased per-expert scatter | 2129.5 us best; M256 921.7 us; M4096 7149.5 us | Correct | Keep architecture, replace scatter schedule |
| V2 | Completed FC1 CTAs join scatter while FC2 CTAs remain GEMM-only | 1933.2 us; M256 767.0 us; M4096 6772.3 us | Correct | Keep |
| V3 | Flatten scatter across routed rows instead of restarting a stride for every expert | 1956.7 us with vector combine in the then-current pipeline | Correct | Keep flattened work distribution |
| V4 | Dedicated early-scatter CTAs plus dynamic row queue; fold expert top-k weight into FC1 requantization | S0 1894.0 us, S2 1888.9 us, **S4 1883.8 us**, S8 1889.4 us | Correct; S4 gives the best compute/communication split | Keep S4 default |
| V5 | Progressive dispatch: each FC1 block waits on its own release/acquire arrival count; remove global dispatch-to-GEMM gate | **1682.4 us** (10 runs), 1679.9 us (30-run cached A/B) | 168 registers, 16 barriers, 0 spill; zero output diff | Keep; -201.4 us / -10.7% versus V4 |
| V6 | TMA/SMEM combine instead of flat vector combine | 1691.7 us (10 runs), 1694.6 us (30-run cached A/B) | Correct | Revert; vector combine is 14.7 us faster |
| V7 | Reconfirmed progressive dispatch + vector combine from a fresh JIT cache | **1680.6 us** (30 runs) | 168 registers, 16 barriers, 0 spill | Current accepted baseline; +102.4 us / +6.5% versus tuned WASP golden |
| V8 | Full 128-row CTA-owned scatter block: one atomic, expert scan and readiness acquire per block | **1964.5 us** (3 runs) | Correct benchmark run; 168 registers, 12 B spill stores, 32 B spill loads | Reject; -8x queue operations does not offset the loss of row-level warp parallelism, +283.9 us versus V7 |
| V9 | CTA claims an 8-row batch with one queue atomic, then assigns one row to each epilogue warp | **1716.3 us** (10 runs) | 168 registers, 16 barriers, 0 spill | Reject; two CTA barriers per batch cost more than the saved atomics, +35.7 us versus V7 |
| V10 | Warp-prefix expert lookup plus explicit peer-base hoisting in scatter | **1710.5 us** (10 runs) | 168 registers, 16 barriers, 0 spill | Reject; longer live state/address pressure outweighs fewer expert scans, +29.9 us |
| V11 | Explicitly hoist the mapped peer destination base only | **1704.0 us** (10 runs) | 168 registers, 16 barriers, 0 spill | Reject; compiler already schedules the mapping effectively; extended pointer lifetime costs +23.4 us |
| V12 | Each warp claims two routed rows per queue atomic | **1744.2 us** (10 runs) | 168 registers, 16 barriers, 0 spill | Reject; reserving a second not-yet-ready row creates head-of-line waiting, +63.6 us |
| V13 | Two-way scatter copy ILP: each lane keeps and writes two uint4 values per loop | **1686.5 us** (10 runs) | 168 registers, 16 barriers, 0 spill | Reject; extra live value/address issue pressure costs +5.9 us despite identical bytes moved |
| V14 | Lane 0 loads warp-uniform FC1/FC2 weight SF and shuffles to the other lanes | **1771.7 us** (10 runs) | 168 registers, 16 barriers, 0 spill | Reject; Hopper already coalesces uniform loads, while shuffles serialize the software-pipelined next-SF prefetch, +91.1 us |
| V15 | SM100-style FC1 epilogue front end: FP32 accumulator pairs narrow to BF16, packed BF16x2 clamp, then FP32 SwiGLU/amax/FP8 quantization | **1664.5 us** (10 runs) | 168 registers, 16 barriers, 0 spill; 8-rank L1 smoke diff=0.0001 | Keep; -16.1 us / -1.0% versus V7. CUDA 12.8 lacks SM100's FP32x2 intrinsics, so only the supported packed-BF16 part is used |
| V16 | Ordinary-DeepGEMM-style cooperative weight-scale staging in SMEM | No valid timing: launch hit illegal memory access | Compiled at 168 registers, 16 barriers, 0 spill, but added 512 B beyond the host/JIT dynamic-SMEM launch contract | Reject and fully revert. This transplant requires a coordinated host SMEM-size change and adds a full-math-CTA barrier per tile; it is not a safe local hot-loop optimization |
| V17 | Hoist invariant A/B GMMA SMEM descriptors out of the scheduler/K loop; update only the encoded 16-byte stage/K address, following ordinary DeepGEMM SM90 GEMM | **1654.9 us** (30 runs), versus V15 1664.5 us | 168 registers, 16 barriers, 0 spill; descriptor address is algebraically identical | Keep; -9.6 us / -0.58%. In a max-tokens=4096 shape sweep: M256 +1.9 us, M1024 -0.4 us, M4096 -27.0 us |
| V18 / V22 | Reduce the CTASP TMA pipeline from five stages to four; CTASP defaults to four while WASP retains its original maximum-stage policy and the environment override still wins | **1634.6 us** (30 runs); direct 20-run A/B 1634.3 versus 1654.9 us | 168 registers, 16 barriers, 0 spill; 183,536 B dynamic + 1,024 B static SMEM; 8-rank benchmark completes | Keep; -20.6 us / -1.24%. Four stages cover TMA without paying the fifth-stage bookkeeping/SMEM cost in this one-CTA/SM kernel |
| V19 study | Packed W4 weights with A8 activations | No kernel timing: native-instruction feasibility gate failed | SM90 WGMMA has FP8/FP16/BF16/TF32/INT8 forms but no INT4-by-FP8 form; packed B would need producer-side unpack/dequant to FP8 | Do not put in the accepted path. Stable DRAM read is only about 50%, while unpack adds producer, SMEM and barrier pressure; retain as a separate accuracy/format experiment |

The source was returned to the V7 warp-granular, one-row-at-a-time queue and
single-uint4 copy loop after V8-V13.

## Original WASP golden sweep

The golden is measured, not the first default run.  Main M=1024 highlights:

- Cooperative N-major=0, EPW 1/2/4/8/16/32: 2334.7 / 1759.4 / 1647.7 /
  1602.5 / 1607.2 / 1581.5 us.  Stage 4 gave 1579.3 us; stage 3 was an
  invalid launch for this resource footprint.
- Ping-pong EPW=32, N-major=0: **1578.2 us**; N-major=1: 1578.9 us.
  Stage 6 was 1581.2 us.  Reducing SM count from 132 to 130 regressed to
  1609.8 us.
- At M=256, ping-pong N-major=1/default EPW/default stages was **595.0 us**
  over 30 runs.  At M=4096 the best completed point was cooperative EPW=16
  at **5064.2 us**.
- A post-V22 golden recheck was attempted but not recorded: an external VLLM
  job was concurrently holding about 137 GB on each GPU, and the eight-rank
  symmetric-memory setup/teardown returned CUDA driver `invalid argument`.
  No partial number from that run replaces the previously completed golden.

## V7 NSYS phase decomposition (M=1024)

The profiled launch is 1677.8 us (50-us GPU-metric sampling; CUDA-event run is
1680.6 us):

| Approximate interval | Duration | Observation |
|---|---:|---|
| Progressive dispatch + strong FC1/FC2 compute | 0-1050 us | Tensor activity starts at kernel entry and stays about 44-47%; the old ~200-us dispatch gate is gone |
| FC2 tail overlapping scatter | 1050-1240 us | Tensor activity falls while NVLink response traffic rises |
| Scatter drain | 1240-1340 us | Tensor activity is near zero, peer traffic remains active |
| Cross-rank tail / barrier wait | 1340-1590 us | SMs are resident but issue/tensor/L2 activity are nearly zero |
| Vector combine | 1590-1678 us | Local L2/SM activity resumes for about 90 us |

The remaining optimization target is therefore scatter tail variance and FC2 pool
balance, not another global dispatch synchronization change.

## V15 NSYS bottleneck audit (M=1024)

The aligned V15 launch is 1645-1651 us under NSYS; the normal CUDA-event result
is 1664.5 us.  The 20-kHz GPU-metric report is
`results/nsys_megamoe/ctasp_v15_bf16epi_m1024_20khz.nsys-rep`.

| Interval | Dominant work | Measured utilization / interpretation |
|---|---|---|
| 0-200 us | Progressive dispatch plus GEMM ramp | Tensor rises 15% -> 45%; incoming/outgoing NVLink response traffic reaches about 50%. Dispatch is overlapped rather than globally gated. |
| 200-1050 us | Stable FC1/FC2 GEMM | Tensor 46-47%, SM issue 40%, DRAM read 45-55%. This is a mixed WGMMA + software block-scale pipeline, not a bandwidth-saturated or tensor-saturated GEMM. |
| 1050-1100 us | GEMM pool tail | Tensor falls 39% -> 14%; fixed FC1/FC2 CTA allocation is draining unevenly. |
| 1100-1300 us | Scatter | Outgoing NVLink request payload peaks at 73%, incoming request payload at 45%; SM issue falls 16% -> 1%. The strong part is network-bound. |
| 1300-1550 us | Cross-rank tail | SM residency remains 100% but issue is about 1%, tensor is 0%, local TX reaches 0 while incoming RX request payload remains 23-32%. Rank 0 has finished its own rows and is waiting for slower ranks still writing it. |
| 1550-1645 us | Combine | Tensor 0%, issue rises only 4% -> 11%, DRAM read reaches 37% and write 12%. This is a short serial-top-k/dependency-limited reduction, not HBM saturation. |

Whole-kernel averages are Tensor 30.3%, SM issue 27.2%, DRAM read 34.0%,
DRAM write 4.2%, with SMs active 99.4%.  The last number is misleading by
itself: the kernel uses 168 registers/thread x 384 threads = 64,512 registers
per CTA (98.4% of an SM register file) and 217,344 B dynamic plus 1,024 B
static shared memory.  Both resources force one CTA/SM; only 12 of the SM's 64
warp slots can be resident (18.75% occupancy), and only eight are math warps.

The original cooperative WASP NSYS launch is 1601.4 us.  It averages Tensor
30.4%, issue 27.2%, L2 bandwidth 39.6%, and L2 hit rate 34.3%.  The equivalent
V7 CTASP topology averaged Tensor 30.1%, issue 26.8%, L2 bandwidth 41.1%, and
L2 hit rate 30.1%.  CTASP makes the GEMM plateau smoother and scatter TX peaks
higher (74% versus WASP's 47%), but its compute/scatter drain extends to about
1300 us versus about 1200 us for WASP.  Its cross-rank wait is slightly shorter,
so the net remaining loss is primarily fixed-pool GEMM balance plus ring traffic,
not an under-driven scatter link.

At routed M=8192, the two pure cuBLASLt GEMMs take 468.352 us and sustain about
1540.6 effective TFLOP/s.  V15 takes 3.554x that lower bound and sustains about
433.5 effective TFLOP/s end-to-end.  The compute-active span alone is about
1.2-1.25 ms (2.6-2.7x the two pure GEMMs), leaving clear GEMM/dequant/epilogue
headroom before communication and combine are counted.

## V17/V22 GEMM optimization and current phase audit (M=1024)

V17 follows the ordinary DeepGEMM SM90 FP8 GEMM descriptor pattern without
copying its MegaMoE implementation.  Each math warpgroup creates the invariant
A/B SMEM descriptor once and shuffles its low word once.  A stage/K iteration
then changes only the encoded 16-byte address field.  This removes repeated
descriptor construction from both FC1 and FC2 hot loops without changing the
TMA/WGMMA schedule or CTA specialization.

V22 reduces the CTASP pipeline to four stages.  A direct 20-run A/B measured
1634.3 us at four stages versus 1654.9 us at five.  The rebuilt default produced
1634.6 us over 30 runs.  A single max-tokens=4096 template showed the following
shape sensitivity:

| tokens/rank | V17 five-stage | V22 four-stage | Change | Tuned WASP golden |
|---:|---:|---:|---:|---:|
| 256 | 722.2 us | **721.0 us** | -1.2 us / -0.17% | 595.0 us |
| 1024 | 1724.3 us | **1704.6 us** | -19.7 us / -1.14% | 1578.2 us |
| 4096 | 5868.7 us | **5809.5 us** | -59.2 us / -1.01% | 5064.2 us |

The current 20-kHz NSYS report is
`results/nsys_megamoe/ctasp_v22_s4_m1024_20khz.nsys-rep`.  The aligned rank-0
kernel is 1634.466 us:

| Interval | Duration | Tensor / issue / DRAM read | Interpretation |
|---|---:|---:|---|
| 0-200 us | 200 us | 36.8% / 29.8% / 35.0% | Progressive dispatch and GEMM ramp |
| 200-1050 us | 850 us | 46.8% / 36.7% / 49.8% | Stable FC1/FC2 GEMM; neither tensor nor HBM is saturated |
| 1050-1100 us | 50 us | 30.0% / 26.0% / 30.0% | Fixed FC1/FC2 pool drain |
| 1100-1300 us | 200 us | 7.5% / 9.0% / 13.0% | Scatter, with TX request traffic averaging 72.8% |
| 1300-1550 us | 250 us | 0.0% / 1.0% / 1.0% | Cross-rank tail; RX request traffic remains 26.0% |
| 1550-1634 us | 84 us | 0.0% / 8.0% / 24.5% | Local top-k combine |

The aligned semantic profiler was replayed separately in phase-only and
wait-only modes, after a warmup allocation/JIT capture and an eight-rank host
barrier.  All traces have zero dropped events and rank 0 passes
`inspect --require-complete`.  The detailed report and diagram are:

- `results/ctasp_v22_s4_m1024_stage_report.md`
- `results/ctasp_v22_s4_m1024_stage_timeline.svg`
- `results/ctasp_v22_s4_m1024_mkprof_phase.{json,md}`
- `results/ctasp_v22_s4_m1024_mkprof_wait8.{json,md}`

The phase-only replay is intentionally instrumented and spans 2522.368 us on
rank 0, so it is used for entity distributions rather than production wall
time.  FC1 tile P50/P95 is 43.072/59.232 us, with its fused
SwiGLU/weight/quant/ring epilogue at 2.592/4.032 us.  FC2 tile P50/P95 is
16.320/21.440 us, with its local-BF16 epilogue at 2.304/3.872 us.

The decisive wait results are:

| Wait | P50 | P95 | Max | Conclusion |
|---|---:|---:|---:|---|
| Pre-combine cross-rank barrier | 666.352 us | 812.064 us | 835.232 us | Dominant instrumented tail |
| Scatter FC2-ready | 0.640 us | 146.496 us | 209.792 us | Large long tail; pool balance/order is next |
| FC2 activation-ready | 0.832 us | 1.248 us | 95.712 us | Median is healthy, rare dependency tail remains |
| FC1 dispatch-ready | 0.512 us | 0.736 us | 27.584 us | Progressive dispatch is healthy |
| FC1 ring reuse | 0.448 us | 0.704 us | 0.960 us | Ring depth 32 is sufficient |

The profiler macros fully compile out.  The profiler-off cubin and the
pre-profiler V20 descriptor-hoist cubin have byte-identical complete SASS
(SHA-256 `eaf35294...`) and identical 168-register/16-barrier/zero-spill
resources in the five-stage audit.  The profiler therefore remains a replay
diagnostic and adds no production-path gate.

### W4A8 decision

The per-rank FP8 expert weights total about 1.409 GB at this shape; packed W4
would save about 704.6 MB of reads if consumed natively.  Hopper WGMMA does not
provide an INT4-by-FP8 instruction, however: the available families are FP8,
FP16/BF16, TF32 and INT8.  Preserving the FP8 math path means unpacking and
dequantizing W4 B tiles into FP8 SMEM before WGMMA.  That adds exactly the
producer/SMEM/synchronization pressure that is already constrained at one
CTA/SM, while the stable production interval uses only 49.8% DRAM read
bandwidth.  W4A8 is therefore not a justified mainline optimization for the
measured bottleneck; it needs a separate weight-format, accuracy and unpack
microbenchmark before reconsideration.

### NCU limitation

No NCU counter result is claimed.  Kernel replay deadlocks because replaying one
rank prevents the other seven ranks from completing the in-kernel NVLink
barrier.  Application replay failed kernel matching when peers remained live;
profiling the entire eight-process tree, including a CUDA start/stop app-range,
also stalled in pass 1 because NCU serialized related rank ranges.  The temporary
benchmark app-range gate was removed.  NSYS's one-pass GPU metric sampler is the
valid hardware evidence for this multi-rank monolithic kernel on the installed
NCU 2025.1 build.

## Correctness and consistency invariants

- Dispatch writes token data, SF, top-k weight and metadata before a release-add
  to the FC1 block arrival counter.  The FC1 A-loader uses an acquire load and
  cannot observe a ready count before those payloads are visible.
- FC1 publishes every post-SwiGLU FP8 N slice through the L2-ready mask.  FC2
  waits for the complete mask before reading the bounded ring slot.
- FC2 publishes each BF16 N slice before incrementing the high 16 bits of the
  block completion counter.  Scatter acquires the full N-slice count before any
  peer copy.
- Ring reuse waits for the previous logical occupant's full FC2 completion count.
- The top-k expert weight is applied in FC1 requantization.  This is valid because
  FC2 is linear: `FC2(weight * activation) = weight * FC2(activation)`.
