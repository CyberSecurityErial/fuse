# v24.0 — SM103 MXFP8 forward and complete backward baselines

This release packages the accepted QKVProj / OProj double-fusion operators,
including both complete MXFP8 backward paths. Norm/RoPE remains separate,
default-off and experimental. All predefined legal long-sequence points are
retained; aliases count once.

## 典型泛化场景 / Representative generalization scenarios

| Operator | Headline physical points | Geometric mean PFLOPS/GPU | Target | Status in this cohort |
|---|---:|---:|---:|---|
| OProj forward | 36 | 2.249637 | 2.2 | Met |
| OProj complete backward | 36 | 2.122491 | 2.0 | Met |
| QKVProj forward | 27 | 2.200574 | 2.2 | Met in revised cohort |
| QKVProj complete backward | 27 | 2.081297 | 2.0 | Met in revised cohort |

QwenDense remains a supplementary scenario: all six points per QKV direction
remain in the detailed tables but do not contribute to the headline summary.
The original all-measured QKV means remain 2.047223P forward and 1.870026P
backward (33 points each), below the original full-matrix targets. OProj is
unchanged. This user-requested post-release cohort revision changes no code,
measurement or per-point configuration, and does not prove universal
generalization. results.json preserves both summary cohorts.

CP4/8 × 128K/256K/512K, fixed configurations, two actual nonzero Philox payloads,
each independently warmed and sampled with Graph10+50. No OOM/missing/numerical
failure was removed from these predefined matrices. Individual points can be
below their operator's aggregate target.

[Full table](table.md) · [CSV](table.csv) ·
[Exact configurations, replay arguments and evidence](results.json)

## What changed

- **Complete QKV MXFP8 backward:** inverse planar gradient A2A and dX, followed
  by the independently quantized dW path. Original BF16 gradients are retained
  for the W phase; immediate/deferred and alpha/beta semantics are preserved.
- **Faster transpose preparation:** coalesced input into a padded shared tile,
  register-owned K32 reduction, packed FP8 conversion and coalesced FP8/scale
  writeback. The launch grid follows actual compiled occupancy.
- **Input prefetch:** after all current K32 input reads finish, reuse the same
  input shared tile for the next group through cp.async while reducing and
  converting current register values. Output staging stays separate. No extra
  shared stage, ready granularity change or weakened synchronization.
- **QKV backward transport/ready adapter:** batched asynchronous payload copies,
  early complete-head scale reads, and parallel complete-head acquires matched
  to the actual pipeline stages before elected TMA issue.
- **Independent GEMM tuning:** dW raster/swizzle/epilogue is independent of dX
  and communication; beta=0 uses a source-free epilogue while accumulation keeps
  its original semantics. This release uses explicit finite-search plans, not
  a new Auto policy or global-optimality claim.
- **Code organization:** keep quantization in its existing private header and
  profiling in include/fuse/profiling. Two outdated development logs were
  condensed from 2,122 to 258 lines of module ownership, contracts and useful
  optimization conclusions. No new core/public file was introduced.

The following prefetch comparison retains ALL measured points, including
supplementary QwenDense. On the headline 27-point QKV cohort the corresponding
change is 2.057202P → 2.081297P (+1.17%).

The final prefetch confirmation keeps the same per-point GEMM/communication
configuration as its immediate predecessor:

| Full backward boundary | Before prefetch | With prefetch | GM gain | Observed per-point gains |
|---|---:|---:|---:|---:|
| OProj, 36 points | 2.092341 | 2.122491 | +1.44% | +0.55% … +3.02% |
| QKV, 33 points | 1.836677 | 1.870026 | +1.82% | +0.35% … +5.79% |

All69 observed changes are positive; these are sequential same-node fixed
confirmations, not a guarantee that every sub-percent difference is noiseless.
The inspected quantizer changes 48→56 registers, retains shared storage, and
has no local spill. SASS verifies the prefetch issue order, not the precise
amount of latency hidden.

## Boundary and correctness

Forward includes dynamic BF16 W→MXFP8 K32 quantization, GEMM, A2A and required
synchronization. Input activation quantization is upstream. MXFP8 uses
E4M3/UE8M0, FP32 accumulation and BF16 output; forward work is 2*M*N*K/card.

Complete backward is the actual immediate B+dW five-kernel Graph, including
all three orientation-specific transpose/quantization preparations, the two
GEMMs and the required route. Work is 4*M*H*A/card. Independently timed B/W
components are NOT added to create this fused time. Upstream gradient
quantization and cross-CP dW reduction are declared caller-owned boundaries.

Quantized transposes are reconstructed from ORIGINAL BF16 masters along the
new K32 reduction axis. Forward scale bytes are not reused for a transposed
GEMM. The straight-through convention does not claim gradients of rounding or
amax, or full-model training convergence. Deferred W and all-rank peer-input
leases remain part of the public contract.

Both payloads pass independent whole-output numeric and byte-exact route
checks before/after measurement. Original tolerance, ready/fence and Graph
epoch rules are unchanged. Latency is the mean of the two payload p50 values,
not the faster one. Forward p95 is the mean of payload p95s, not pooled p95.
Raw samples stay in the hashed formal archives rather than release assets.

Forward rows retain the earlier fixed two-payload confirmation
(source010ba02f); they are not presented as a new full forward measurement of
the prefetch binary. Prefetch changes only backward preparation. The final
release runs targeted forward integration checks; their identities and status
are recorded separately in results.json. Backward rows use the newly confirmed
prefetch sourcee2982801, build20260914-090006-5fe8c8. Source snapshots and remote
binary attestations identify actual measurements; tag/commit labels do not
replace those fingerprints.

## Reference and scope limitations

Primary reference remains full148SM cuBLASLt. Forward pure-GEMM values are
historical; backward pure reference run20260914-023303-0cbfa4 contains93
independently audited represented-operand geometries, each with two actual
10+50 payloads. Its equivalent backward throughput adds two independent GEMM
times and excludes quantization/communication: it is NOT a measured fused E2E.
The reference configurations are retained without the search history.

Qwen3 CP8 QKV needs unimplemented KV replication: all three sequences remain
explicitly unsupported in forward and backward. Kimi denotes supported
projection geometry, not full special KDA routing. No new MLA/KDA-special
route, norm/RoPE backward or model-training validation is claimed. The old
experimental norm/RoPE failures are not declared fixed. SM90 implementation
and historical release tables remain unchanged.

## Reproduction and audit

Use the current checkout and the existing private SM103a toolchain:

```bash
# Forward
python3 scripts/l20d.py run fused-build --node 09 \
  --workspace /home/work/workspace_wct --mpi --mxfp8
# Backward
python3 scripts/l20d.py run fused-build --node 09 \
  --workspace /home/work/workspace_wct --mpi --mxfp8 --backward
```

Choose a row in results.json. Append its replay_args to replay.run_prefix.
These arguments include the direction-specific raster, GEMM layout,
communication budget and windows; no historical source-run or online search
is needed to execute the release. Reproduction needs available B300 memory
and the original measurement/validation protocol, not a promise of identical
clocks or runtime.

Local audit:706 SM103 tests pass with5 dependency-conditioned skips;112 controller
tests pass. All138 exported replay commands pass isolated CLI/geometry/binding
validation without external actions. Both CP4/CP8 independent CPU-FP64 backward checks cover Q/O,
E32/E64, Eager/Graph, alpha/beta and deferred W. All19 private headers have
include consumers, local quoted includes resolve, and101 compiled input/build
files match the frozen candidate snapshot. Full138 fusion records and93 pure
reference rows are independently re-audited from their receipts.

See [compact optimization and numerical contracts](../../../benchmarks/sm103/MXFP8_DUAL_FUSION.md)
and [module ownership](../../../csrc/operators/sm103/README.md).
Final results contain configuration/evidence only; no profiles, dependencies,
large raw logs or per-candidate tuning histories are shipped.
