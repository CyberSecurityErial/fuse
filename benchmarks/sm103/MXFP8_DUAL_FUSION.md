# SM103 MXFP8 dual-fusion contracts and optimization summary

The mainline operators are QKV GEMM→A2A and OProj A2A→GEMM, with their
corresponding complete backward B+dW boundaries. Norm/RoPE remains a separate,
default-off experiment, not part of the mainline performance tables.
Final measurements and explicit configurations live in
[v24.0](../../results/sm103/v24.0/README.md). Historical releases stay immutable.

## Numerical and lifetime contract

- Activations supplied to forward are MXFP8 E4M3 with K32 UE8M0 scales.
  BF16 master weights are dynamically quantized inside the measured operator.
  GEMMs accumulate in FP32 and produce BF16. Upstream activation quantization
  is an explicit external boundary.
- Backward uses a straight-through low-precision training convention: ordinary
  linear-layer gradient GEMMs with each operand orientation independently
  quantized from its original BF16 master. It does not differentiate amax or
  rounding, or establish full-model training convergence.
- A transpose changes the reduction axis. Transposing FP8 bytes and keeping
  forward scales is not valid K32 preparation. Both transposed operands are
  prepared inside the appropriate backward phase.
- QKV B reads upstream quantized planar dQ/dK/dV and also transports the
  ORIGINAL BF16 gradients into packed sequence-local dQKV for dW. The dW
  reference and production input never come from dequantizing dX inputs.
- Peer inputs remain immutable until ALL ranks finish B. Deferred W additionally
  leases packed dQKV/saved X (or dY/saved A for OProj) until W completes.
  Concurrent invocations require disjoint workspace/control slots.
- BF16 dW supports alpha/beta accumulation. Cross-CP dW summation is owned by
  the caller and excluded from these local-partial-gradient measurements.

| Operator | B / data gradient | W / weight gradient |
|---|---|---|
| QKV | inverse planar Q/K/V A2A, then dX = dQKV W | dW = dQKVᵀ X |
| OProj | dA = dY W, then head-shard A2A | dW = dYᵀ A |

Immediate mode launches B then W on one stream. Its measured five-kernel Graph
includes three transpose/quantization preparations and both GEMMs, with the
required route and synchronization. FLOPs/card = 4*M*H*A. It is NOT the sum of
independently timed B and W. Forward FLOPs/card = 2*M*N*K.

QKV requires evenly partitionable Q and KV heads: Qwen3 CP8 KV replication is
not implemented. Kimi entries describe the supported packed QKV projection
geometry, not a complete special KDA route. MLA, special KDA routing and
norm/RoPE backward are not added by this release.

## Retained optimization ideas

These mechanisms are general geometry/resource rules; explicit finite-search
winners remain caller-selected, not model-name runtime branches.

| Mechanism | What it changes | Constraint retained |
|---|---|---|
| Register-owned K32 quantization | Coalesced BF16 input into padded shared transpose; one thread reduces an entire K32 group with four local maximum chains and packed FP8 conversion | Original K32 scale/rounding rule and tail masking |
| Coalesced transpose writeback | Four K32 groups produce two K64 FP8 stages and complete scale atoms; launch grid follows compiled occupancy | Same represented operands, no extra GEMM work |
| Input prefetch within quantization | Once all current-group input reads finish, reuse the input SMEM tile for next K32 via cp.async while current register values are reduced/converted | Reader CTA join, async wait plus CTA join, separate output staging; no added ready checks |
| QKV backward input transport | cp.async batches FP8 and original BF16 payloads; complete-head scale reads precede these copies | Full-head publication and original-gradient lease |
| Ready adapter | Acquire complete heads in parallel according to actual pipeline stages, then warp join and async-proxy ordering before elected TMA issue | No early partial-head consumption or weakened fence |
| Fresh dW epilogue | beta=0 uses the source-free epilogue; beta!=0 retains accumulation | alpha/beta semantics and original tolerance |
| Independent dW layout | dW epilogue/raster/swizzle tuned independently of dX and communication | GEMM tuning does not silently change the fusion budget |
| OProj forward startup and transport | Compute CTAs assist W preparation while communication starts A/SFA; balanced scales and capacity-aware mixed copy slots | Complete-panel readiness and same dynamic quantization boundary |
| Producer/consumer layout | Resolve Along/swizzle/tile first, then enumerate copy tasks at their GEMM dependency frontier and match communication windows | Acquire all overlapping dependencies; concurrent completions may be out of order |

The prefetch is internal to each quantizer, not overlap of the whole
preparation kernel with the subsequent GEMM. Generated SASS confirms next-group
LDGSTS precedes current-group reductions; it does not by itself measure hidden
latency. Compiled registers change from 48 to 56, with unchanged shared storage
and no local spill in the inspected build; no unchanged-occupancy claim is made.

Useful rejected directions: replacing the backward payload copies wholesale
with TMA, stronger/batched publication rewrites, unconditional destination
rotation, and wider QKV route boxes did not give reliable general end-to-end
benefits in their controlled tests. Those trial implementations are removed.
This rejects the measured organizations, not the underlying instructions.
Detailed historical experiments remain recoverable from Git and hashed local
receipts rather than being repeated in final benchmark tables.

## Tuning and diagnostic boundaries

GEMM settings and actual compute-CTA budget are inputs; fusion transport,
quantization ordering and communication budget must adapt to those inputs.
Use geometry, actual resource occupancy and measured component evidence.
Do not fit model-name/shape winner tables or hide an online timing search.

Backward dW flags are independent:
`--backward-weight-epilogue-n`, `--backward-weight-swizzle`,
`--backward-weight-raster`. When absent they inherit dA settings.
The release uses explicit positive communication CTA counts, not a new Auto.

`--calibrate` measures B, W and prepared compute references independently.
Prepared dW must immediately follow completed W with its inputs untouched.
Prepared dX references require completed B and its preserved scratch/ready;
W must not overwrite that scratch in between. The bare dX reference removes
the ready adapter but keeps the compute budget and outer SMEM reservation.
These are diagnostic boundaries, never substitute full fusion measurements.

Independent component differences can include changed cache/launch/contention
effects. Neither W minus prepared-W nor release-to-acquire is automatically an
isolated quantizer timer. A long ready lag can mean early production or late
consumption. Compare matching tile/peer times relative to one GPU's kernel
origin; do not subtract absolute timestamps across GPUs or sum overlapped waits.

## Measurement and validation

Formal uninstrumented MPI Graph: two reproducible nonzero Philox payloads,
each independently converged, 10 warmups + 50 actual samples, complete
independent numeric/route checks before and after measurement. Each sample
uses the maximum rank time. Aggregate latency is the equal-weight mean of
payload p50 values; p95, when reported, is the mean of payload p95 values,
not a pooled percentile. Preserve all real samples and first-stable-round
selection; never take the faster payload or faster source version.

The backward oracle reconstructs K32 operands from original BF16 masters and
uses independent FP32-accumulating cuBLAS with bounded scratch. It does not read
production scales or build its reference from actual output. The original
0.01 + 0.01*abs(reference) tolerance and separate byte-exact route checks remain.
A payload-scoped reference cache is invalidated when inputs change; bounded
fallback remains available. CPU-FP64 and poisoned-output tests validate it.

The native epoch must advance on every Graph replay. The harness checks actual
kernel dependency chains and updates epochs; replaying a fixed stale epoch is
not safe. Full backward includes all five kernels; Graph preparation is not an
excuse to exclude dynamic quantization.

Main performance reference remains full-device cuBLASLt. Same-budget CUTLASS
and cuBLASLt are diagnostics. Backward's pure-compute equivalent adds the two
separately measured GEMM times and excludes quantization/communication.
Historical one-timed-payload results stay labelled as such; newer two-timed
payload confirmations do not silently relabel old releases.

Use repository macro profiling and [PROFILE_PROTOCOL](../../PROFILE_PROTOCOL.md);
Perfetto exporters retain role/tile/peer ownership. Profiling samples are not
production timing. Preserve raw provenance by run/source/binary/environment/
archive hashes, but do not ship profile traces, dependency packages or search
history as benchmark results.
