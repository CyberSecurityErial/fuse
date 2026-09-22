# v27.0 — SM103 BF16 Dispatch + Grouped GEMM

User-accepted final BF16 Dispatch baseline. No additional tuning is scheduled.

[Complete table](table.md) · [Configurations, replay commands and evidence](results.json)

## Core changes

- Preserve the offline GEMM winner's scheduler, tile, raster, swizzle and 1-/2-CTA UMMA width when integrating fusion. Native and CUTLASS grouped schedulers are explicit choices.
- Derive communication order and first-consumer coverage from the actual GEMM traversal. A two-CTA compute cluster remains a complete pair; full-K delivery uses the selected 128-/256-row compute tile. Communication never takes one member of an active pair.
- Select measured communication budgets 20/32/40 for small-token results. This is a finite offline choice, not a newly enabled production Auto or a global optimum.

## Measured performance

| Scope | Fused valid | PFLOPS/GPU GM | Actual strong-GEMM retention | vs v26.0 | Paired points |
|---|---:|---:|---:|---:|---:|
| EP4 all | 255/255 | 0.366667 | 79.71% | +11.23% | 249 |
| EP4 M<192 | 119/119 | 0.111346 | 80.48% | +7.24% | 119 |
| EP4 M>=192 | 136/136 | 1.040325 | 79.04% | +15.00% | 130 |

The geometric mean includes M=1 through 8192; it is not a large-token-only peak.
The only paired regression exceeding 3% is Mixtral 8x22B M8, -3.81%.
EP8 is **not remeasured** in v27; its separately labelled historical results
remain in [v26.0](../v26.0/README.md), without copying them into the current cohort.

## Overhead-adjusted diagnostics

| Scope | Including overhead / overlap reference | Overhead removed / overlap reference | Overhead removed / same-SM GEMM | Overhead removed / 148SM strong GEMM |
|---|---:|---:|---:|---:|
| EP4 all | 93.54% | 93.17% | 92.91% | 85.16% |
| EP4 M<192 | 91.57% | 90.98% | 90.63% | 85.78% |
| EP4 M>=192 | 95.22% | 95.02% | 94.95% | 84.63% |

For measured times F=fusion, G=same-configuration/same-SM own GEMM,
R=transport, B=148SM strong GEMM, and H=overhead estimate, the four columns are
`max(G+H,R)/F`, `max(G,R)/(F-H)`, `G/(F-H)`, and `B/(F-H)`.
All aggregate ratios are geometric means of per-point ratios. The first two
columns use 249/113/136 valid points; the last two use 255/119/136. Six unstable
transport measurements remain missing; do not multiply columns across different sets.

H is the median of three same-configuration lightweight profiles. On the rank
with the longest local-role envelope, it compares the maximum compute-CTA end
time with the maximum after subtracting each CTA's first ready wait. This is a
fixed-duration sensitivity estimate, **not** a proven removable cost or hardware
bound. The 23 overlap estimates above 100% are retained; 12 points have H ranges
exceeding 20% of their median. Profile-adjusted ratios are not achieved throughput.

## Validation, scope and replay

Formal timing uses two reproducible nonzero payloads, Graph10+50, the mean of
per-payload max-rank p50s, and complete numerical/routing/tail validation. Small-token
budget winners have not had a separate full winner-only replay; large-token
winners were independently replayed. Component timings repaired from later data
use the first stable measurement of the identical configuration, with provenance.
The strong reference is the historical faster valid full148SM CUTLASS/DeepGEMM
pure grouped GEMM, **without communication**, not a same-run A/B or global optimum.

166 local tests pass, including randomized schedule/route properties. The 116
CUDA/C++ inputs match the validated GPU build; native/CUTLASS × 1-/2-CTA × both
rasters passed eight device property jobs with 16 changing-route Graph replays.
Production/profile builds and evidence hashes are recorded in results.json.
Detailed multi-producer panel profiling remains unsupported; lightweight ready
and overhead summaries support the measured configurations.

Build using [the grouped implementation guide](../../../benchmarks/sm103/GROUPED_GEMM_STUDY.md).
Each row's `replay_argv` preserves its full policy, buffer and tail settings;
select four idle, peer-accessible GPUs through `CUDA_VISIBLE_DEVICES`. Do not
substitute the old six-field policy or assume all rows use tail balancing.

Projection/SM90 algorithms are unchanged. Combine and swapAB are not new v27
acceptance claims. Only final results/configurations/evidence identifiers are
published: no tuning history, profile traces, raw sample arrays or dependencies.
