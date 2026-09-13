# v23.0 — MXFP8 OProj forward 2.2P milestone

The **original A2A + OProj double fusion**, without norm/RoPE, reaches
**2.256462 PFLOPS/GPU geometric mean** over all36 predefined physical points:
CP4/8 ×128K/256K/512K, six physical projection families. All36 fixed
configurations passed; no OOM, missing or numerical-failure points were removed.
Seven individual points remain below2.2P. Llama70B/Qwen72B aliases count once.

Relative to the historical v21 release: **+17.85%** throughput. Retention of
historical full-device pure cuBLASLt is **83.1%** (2.7167P geometric mean).
These are historical comparisons, not same-run paired causal measurements.
Same-compute-budget CUTLASS is a diagnostic column, not the performance ceiling.

[Full table](table.md) · [CSV](table.csv) ·
[Exact timings, configurations, replay commands and fingerprints](results.json)

## Effective optimizations

- Compute CTAs assist initial weight quantization while activation communication
  starts immediately; each compute CTA then starts GEMM without a global
  weight-completion join. Full-panel readiness is preserved.
- Vectorized scale delivery is balanced across the actual activation chunks;
  one chunk no longer carries an entire disproportionate scale workload.
- Copy-slot capacity follows actual rounded payload geometry. Immediate
  communication workers and compute-assisted quantization reuse the existing
  shared-memory budget rather than adding a late worker handoff.
- Explicit offline communication budgets and M/N windows follow the selected
  GEMM consumption order. There is no model-name device branch, new online
  timing search, new Auto calibration or global-optimality claim.

## Boundary, scope and validation

Input activations are already MXFP8 E4M3 with K32 UE8M0 scales. Dynamic BF16
master-weight quantization, A/SFA communication, GEMM and required synchronization
are inside the measured boundary; upstream activation quantization is outside.
FP32 accumulation produces BF16 output. PFLOPS/GPU is2*M*N*K / full boundary time.

Formal MPI Graph10+50, converged warmup, reproducible nonzero Philox payloads,
per-sample maximum-rank event time, independent complete numeric/route checks
before and after measurement. Original tolerances and ready/fence contracts
are unchanged. All36 F/C/R/P results (144 component boundaries) were independently
re-audited from hashed receipts and per-rank logs before release.

The published whole-matrix measurements are the original fixed confirmation
using sourcefa9d9a87 and binarye59d6bd4; per-row full hashes identify them.
The release adds subsequent backward/benchmark work and receives targeted
forward integration regressions, not a newly measured36-point table on that
integrated binary. No later single-point faster retest is mixed into the mean.
Source-run and raw evidence stay in local `fuse_midfile/l20d/<run_id>`;
they are not bundled as large release assets.

These are operator-level projection geometries, not full-model training
validation. Kimi's geometry does not imply full special KDA routing. Historical
unsupported MLA/KDA/KV-replication cases remain unsupported, not silently folded
into the36-point catalog. v22 norm/RoPE is still default-off and experimental;
its old numerical failures are not claimed fixed by this release.

## Reproduction

Build this checkout with the existing private toolchain:

```bash
python3 scripts/l20d.py run fused-build --node 09 \
  --workspace /home/work/workspace_wct --mpi --mxfp8
```

Then use a row's `replay` command in `results.json`. The commands select explicit
positive communication budgets, `weight_preparation=all`, M128/N256/K128,
epilogue N32, AlongN, swizzle and optional H/P window. H=P=0 disables the window.
They rebuild/reuse this checkout normally, not a private historical source-run.
`--calibrate` separately measures C/R/P diagnostics; only F is the fused result.

## Saved development progress; remaining goal continues

- QKV forward is still being optimized toward2.2P; no new confirmed full QKV
  table is claimed by this OProj release.
- Complete OProj MXFP8 backward B+dW has an initial, untuned CP8/128K baseline:
  six physical geometries GM1.355396P. It is not full long-sequence coverage or
  a2P result. Immediate/deferred interfaces preserve FP32 accumulation and BF16
  gradients; cross-CP dW summation is explicitly caller-owned. The stated
  straight-through quantization convention is not model-training validation.
- Its actual five-kernel Graph includes W transpose quantization, dA+inverse
  A2A, dY transpose quantization, saved-A transpose quantization and dW.
  Separate B/W diagnostics do not substitute summed times for full-boundary time.
- QKV MXFP8 backward is not yet implemented. Both backward2P goals and QKV
  forward2.2P remain active after publication.

See [development contracts](../../../benchmarks/sm103/MXFP8_DUAL_FUSION.md)
for backward interfaces and baseline status. Final tables contain results and
configuration only; search histories, dependency packages and profiles are not
release attachments.

## Release integration checks

289 related host tests pass (94 MXFP8,109 controller,9 Graph,59 Perfetto,18
OProj planner). The planner's BF16 exporter now preserves both independent
MXFP8 calibration sections; no calibration values changed. The old structural
QKV test now checks the already-existing postprocess exclusion as well.

Current integrated forward CUDA build20260913-185611-e721a5 passed. Three
CP8/128K Graph10+50 regressions with both-payload full pre/post validation
were independently audited: OProj without window190137-9f7d19 (2.115096P),
OProj H64/P16190200-e5fb96 (2.206118P), QKV190223-8a9768 (2.110912P).
They use H8192/Q64/KV8/D128 and C16, and are integration checks, not new
per-point winners or a replacement for the36-point primary confirmation.
The initial regression185955-97db7a was rejected before measurement because
its invocation omitted the required explicit QKV tile-list argument; that
failed receipt remains retained and is not an accepted performance result.
