# SM90 MXFP8-weight 2F2B baseline review

Scope: `feat/sm90-mxfp8-2f2b`, based on remote main
`16de0b2b0f49534361277d352b9a815cb1df3133`. No KDA code or results are included.
The original dirty worktree is preserved separately.

## Structure and readability

| Concern | Implementation |
|---|---|
| Weight representation / scratch contract | `include/fuse/layout/mxfp8.h` |
| Forward public parameters | New MXFP8 structs beside BF16/FP8 in `operators/primitives/{gemm_a2a,a2a_gemm}.h` |
| Backward B/W and immediate/deferred contracts | New MXFP8 structs in the existing `operators/ulysses/{qkv_backward,oproj_backward}.h` |
| Semantic registration | Overloads in the existing four specifications in `semantics/ulysses/projection.h` |
| Device weight conversion | `ulysses_sm90/detail/mxfp8.cuh` |
| Shared-memory-aware FP32 wgrad | Existing `detail/core.cuh` and `detail/launch.cuh`, parameterized by output type |
| Public launches | Forward/backward stay in their original `api/` files; weight conversion in `api/mxfp8.cuh` |
| Baselines, table export and tests | `benchmarks/mxfp8_weight/`; no dependency from production back into this directory |

The SM90 assembly file remains one translation unit. No new communication
kernel, model-specific dispatch table, hidden weight cache, per-call allocation,
cuBLAS/TE dependency, or training-batch metadata system was added. Shared
forward route structs retain the legacy `batch` field fixed to 1; new backward
parameters expose only flattened `local_tokens`. The incoming token extent is
`T`, and `M=T/CP`. No per-sequence boundary processing is performed.

An independent read-only review and an explicit source comparison confirmed
that the old BF16/FP8 public parameter bodies and forward/backward entry
implementations are unchanged. Existing BF16 wgrad keeps its four-stage type;
only the new FP32-output specialization uses epilogue-aware stage selection.

## Findings corrected during review and testing

1. **FP32 wgrad shared-memory overflow.** Simply changing BF16 output to FP32
   with four mainloop stages requested 264,192 bytes of dynamic shared memory.
   The new FP32 specialization now uses `StageCountAutoCarveout`, as the existing
   FP8 path does; the resulting request is 215,040 bytes. A compile-time SM90
   capacity assertion guards this resource contract.
2. **QKV forward layout boundary.** Existing QKV forward routing is rank-major;
   it does not implement the causal-paired mapping used by OProj backward.
   The new entry rejects that flag before DQ or peer writes. It does not modify
   the old BF16/FP8 route to invent a different semantic boundary.
3. **Prepacked forward weights incompatible with shared F/B weights.** QKV
   peer-interleaved rows and OProj cyclic columns need corresponding backward
   transforms. The new forward entries reject those two options rather than
   accepting a layout that backward would interpret differently. Negative
   tests cover both. No extra reorder system was added.

Weight decoding preserves the original forward input-feature scale axis in B.
The decoder handles all E4M3/E8M0 encodings, including signed zero, subnormal
rounding, NaN and overflow, using integer bit construction to avoid fast-math
flush-to-zero changing tiny scales. Independent reference decoding is used in
tests; quantization error is not confused with implementation error.

## Verification

Initial verification: **PASS** for 16 production operator/CP/shape/layout cases,
all 65,536 E4M3×E8M0 encoding pairs, 144 CPU tests, and all four unchanged
legacy executables (`backward_smoke`, `fp8_smoke`, `qkvproj_a2a_smoke`,
`fuse_smoke`). Production source fingerprints match the final implementation.
The independent static re-review found no remaining must-fix item within this
baseline's canonical-layout scope.

The W-tile candidate extension additionally passes 48 GPU cases: forward is
not duplicated, while both backward paths cover auto and four explicit W
policies with the full Eager/Graph and beta0/1 contract. Native resource queries
report stages 3/5/2/3, shared storage 215040/231424/198656/215040 bytes, and
168 registers/thread for the four actual kernels. Smaller N does not imply
lower compiled register allocation. Auto still resolves to the initial kernel;
candidate correctness does not establish a performance improvement.

The additional N192/K64/C2 and N256/K32/C2 candidates pass 32 GPU cases
(auto plus the two new W policies), including all encoding pairs. Actual
stage counts are 4/6, shared storage 231424/215040 bytes, and register count
remains 168/thread. Formal CP8 paired measurements do not justify replacing
the initial W kernel. The CPU suite now has 170 passing checks, including
explicit-model validation and independent service-model fitting contracts.

The calibrated OProj selector is opt-in; the production default is unchanged.
Native/Python parity passes 204 route/layout/domain checks and 288 checks that
the other operators are unaffected. A separate default-path 16-case GPU test
passes. Full CP4/8 OProj forward model-path A/B now covers 96 original settings,
both Eager/Graph, both execution/capture orders, and three 10+50 pairs per order.
All output, route and DQ checks and strict sample/config/provenance audits pass.
Opposite-order balancing is required because an unchanged CP4 configuration
showed order-dependent timing bias; partial reruns never replace full rows.
That OProj verification had 214 passing CPU checks. These OProj-specific
results do not establish a 2F2B milestone.

The independent OProj copy calibration uses the fused frontier-window formula,
shared with the production path. The original two-argument copy reference
retains its fixed-window behavior and ABI. Matched compute references use the
same tile with either the fused compute SM budget or all SMs, without ready
waits. Primitive timings exclude DQ/preloading and are not full operator timings.

QKV backward service references now reuse the production RowMajor-B argument
builder, N256/C2 policy, monolithic scheduler and twelve-slot TMA route. Bare
and preloaded-ready compute keep the fused SMEM reservation and use the same
flattened grid; fullgrid changes only the compute SM budget. The shared builder
extraction passed 16 multi-GPU operator checks and all four legacy BF16/FP8
smokes after rebuilding. Unsupported reference families return not-supported.
All used ready flags are restored as int32 epochs outside every timed sample;
copy output is checked only after all ranks finish, without preloading it.

The original copy reference is C1/196704 B; the additional
`copy_fused_reservation` reference launches the same entry with C2/214016 B.
Both use the actual 38-register copy kernel, not the fused 168-register entry.
The function-level dynamic-SMEM permission stays at the larger bound so
interleaved Graph capture cannot lower it; each launch/query retains its own
actual allocation. Neither copy reference contains concurrent compute/finalize.
Measured copy-only and fused route envelopes differ, so they are not silently
combined into one calibration. No QKV production dispatch has been changed.

Final verification results are recorded alongside the code in
[`results/mxfp8_weight/validation/`](../../results/mxfp8_weight/validation/).
The production correctness report includes source/library hashes; regression
results identify the tested executables. These files are correctness evidence,
not formal performance measurements.

CPU checks cover exact old shape registries, frozen route helpers, negative
audit cases, publication hashes, complete 768/1536-row key sets and selected
configurations. GPU tests cover flattened token extents, CP4/CP8, canonical
layouts, Eager/Graph, immediate/deferred W, poisoned scratch/output, FP32 dW,
and beta=1 twice from nonzero main_grad.

The new distributed performance runner uses CUDA IPC allocations independent
of PyTorch's caching allocator. It clears ready/done state on every rank before
replaying a fixed-epoch graph; clearing and host barriers are outside timing.
The classic cuBLAS helper passed all 16 combinations of transpose flags,
BF16/FP32 output and beta=0/1, including two CUDA Graph replays. The CP4 pilot
covered all four operators; the CP8 pilot additionally covered 512K tokens.
Pilot samples are diagnostic only and are rejected by the formal table builder.
The pilots have now been superseded and their raw vectors removed: formal
CP4/CP8 coverage is complete, 384 settings / 2,304 records, with the strict
source/sample audit archived under `operators/full_v1/audit.json`.

The initial production code and full comparison are frozen in local commit
`d5b1317`; subsequent profiling bridge changes do not rewrite that run's source
hashes. The initial shared library is retained beside its raw measurements.
The formal runner rejects a profiling-enabled shared library before launching
any measured operator. Diagnostic CTA traces use the same IPC/reset/reference
helpers, independent rank clock origins, and bounded output; their marker
spans include observation overhead and are never published as formal timings.

Tables reuse the existing BF16/FP8 units, wide Eager/Graph columns, configuration
spelling and shape-registry order. The original selected baseline JSON is
unchanged by rendering. CPU tests guard winner matching, FP32/beta semantics,
real sequential totals, FLOP formulas, missing rows and duplicate rejection.

## Deliberate limitations

- This is **software DQ to a full BF16 workspace + existing BF16 persistent
  GEMM/A2A**, not in-Tensor-Core MXFP8 or on-chip tile-fused dequantization.
- Callers own scratch and saved B-to-W operands. Epoch and lifetime rules are
  inherited from the existing BF16 operators; this is not a graph scheduler.
- The 2,304 published rows are imported TEUB/cuBLASLt baseline results with
  matching configurations and provenance. They are not timings of the new
  Fuse implementation, and no speedup for the new Fuse code is claimed.
- The production Fuse/classic cuBLAS sweep has its own raw records and coverage
  audit. Only completed formal 10+50 rows enter the comparison; missing points
  stay visible. No optimizer, gradient reduction or training E2E is included.
