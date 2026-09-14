# v24.1 — QKV quantization maintenance release

Only measured, accepted QKV preparation changes are retained. OProj and both
backward implementations/configurations/results remain at v24.0. No new Auto,
GEMM tuning, communication policy or norm/RoPE change is included.

## Results

| Operator / cohort | Points | PFLOPS/GPU | Evidence |
|---|---:|---:|---|
| QKV forward, 典型泛化场景 | 27 | 2.227942 | New full fixed-configuration confirmation |
| QKV forward, all measured | 33 | 2.077387 | Includes supplementary QwenDense |
| OProj forward | 36 | 2.249637 | Retained v24.0 |
| OProj complete backward | 36 | 2.122491 | Retained v24.0 |
| QKV complete backward, 典型泛化场景 | 27 | 2.081297 | Retained v24.0 |
| QKV complete backward, all measured | 33 | 1.870026 | Retained v24.0 |

QKV forward improves +1.24% / +1.47% against the corresponding historical
v24.0 cohort, not a fresh paired full-matrix comparison. All33 new points are
reported; none is replaced by an older faster value. Two tiny regressions
remain visible. The table measures **all**, not standalone quantization.

[QKV full table](table.md) · [CSV](table.csv) · [Parameters and evidence](results.json) ·
[Unchanged operators](../v24.0/README.md)

## Compact optimization summary

- QKV all-mode already completes every W chunk before GEMM begins. Remove
  redundant per-worker panel arrival atomics, retain the original full-grid
  join, then publish each complete panel once. Incremental producers keep the
  existing arrival protocol and consumer async-proxy ordering.
- Distribute all-mode work by contiguous physical1024-value chunks, avoiding
  repeated panel/step decoding. K32 scales, values, padding and output layout
  are unchanged; no extra ready checks or GEMM layout changes.
- The standalone QKV W preparation reuses the register-owned K32 quantizer
  and sizes its launch using actual occupancy. The explicit benchmark option
  `--mxfp8-weight-preparation separate` times both preparation and GEMM+A2A in
  the same two-kernel Graph on every invocation; this is not cached-weight GEMM.
- Add the performance warning at the public mode selector and the all-mode
  branch: fusion inherits GEMM/route parallel-resource limits. In the observed
  comparisons separate preparation is preferable; fewer launches alone do not
  imply higher effective parallelism. It is not a universal speed guarantee.
- OProj incremental addressing and backward barrier-coalescing trials are not
  retained. Their intermediate tuning records are not release assets.

Default preparation selection is unchanged. Standalone preparation remains an
explicit comparison path, not a new default or an automatically chosen winner.
The published33 points use all-mode's accepted optimizations.

## Validation and limitations

Local release checks:714 SM103 tests completed with5 dependency skips;
113 controller tests passed. Dispatch regression compiles the actual wrapper
with profiling both enabled and disabled and checks exactly one operator launch.
These are host checks, not a new GPU performance/profile validation.

The full33 measurements use frozen source
`a85987fd744cd34f84530dd8db8adf7e3870996f24d654aff54c01d2826ebc7b`,
production binary
`daedcd99031c9ec326e1dceadadbb2ac0b9bdae7f50f2f5964437b3c0d37d1e5`.
Run IDs, exact geometry/configuration, replay arguments and drift are retained
per row. CP4/8,128K/256K/512K use the original independent whole-output and route
validation with both actual random payloads, each10+50; upstream A quantization
is excluded. Qwen3 CP8 KV replication remains unsupported, as in v24.0.
Kimi denotes the supported projection geometry, not complete KDA routing.

Release-only wrapper scoping prevents a profiling call from also launching the
ordinary operator. The updated all-mode detailed quantization/release exporter
is not yet adapted to post-grid-join publication: requesting that detailed
trace now explicitly returns cudaErrorNotSupported. Existing incremental
profiling is unchanged; new all-mode Perfetto support is not claimed.
The release comments/diagnostic guards do not change the measured production
kernel. GPU performance receipts identify the actual measured snapshot, not a
retrospectively assigned tag hash.

No raw logs, profiles, dependencies or per-candidate tuning histories are
shipped. Numerical, communication and SM90 contracts remain unchanged.
