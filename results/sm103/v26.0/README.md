# v26.0 — SM103 BF16 Dispatch + Grouped GEMM

Selectable grouped-dispatch transport on B300/SM103: full-warp `cp.async` for
small expert rows and TMA for larger rows, with explicit override retained.
The CTASP performance model may lend idle compute CTAs to small dispatch tails;
the full-K / 128-row ready contract is unchanged.

[Table](table.md) · [CSV](table.csv) · [Exact configurations and evidence](results.json) ·
[API and implementation guide](../../../benchmarks/sm103/GROUPED_GEMM_STUDY.md)

| Scope | Valid / requested | PFLOPS/GPU GM | Strong-GEMM retention | Perfect-overlap attainment | vs v25.1 |
|---|---:|---:|---:|---:|---:|
| EP4 all | 249 / 255 | 0.320239 | 71.3460% | 84.3847% | +4.7248% |
| EP4 M<=128 | 119 / 119 | 0.103828 | 75.0443% | 82.2123% | +11.1725% |
| EP8 all | 255 / 255 | 0.284977 | 62.4600% | 76.4389% | +5.1099% |
| EP8 M<=128 | 119 / 119 | 0.084154 | 64.3663% | 71.2261% | +10.8872% |
| Combined | 504 / 510 | — | 66.7287% | 80.2663% | +4.9184% |

Strong reference is the faster valid full-148-SM CUTLASS/DeepGEMM grouped GEMM.
Perfect-overlap attainment is `max(matched GEMM, transport) / fused`; it measures
the remaining fusion/overlap gap independently of the GEMM implementation gap.

All 510 points were rerun from one corrected source build. Measurements use two
nonzero reproducible payloads, Graph 10+50, max-rank latency, and complete
numerical/routing/tail validation. Six EP4 points are marked
`insufficient_memory`; three valid EP8 points lack a historical strong-reference
denominator. The recorded policy is `cp.async` for M<=128 and TMA for M>=192.

No profiles, raw sample arrays, dependencies, or tuning history are included.
