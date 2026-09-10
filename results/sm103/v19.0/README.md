# v19.0 — Final SM103 BF16 results

BF16 QKV/OProj forward is inherited from v18.0. This version adds the reverse
routes, NN data gradients and TN local weight gradients, with explicit per-shape
GEMM configuration. The cancelled QKV forward autotune is not included.

- [Final backward table](table.md): all 540 logical points, including gaps.
- [Final values and configurations](table.csv): GEMM geometry, tile, epilogue,
  raster/swizzle, communication budget, validity and measurement run IDs.
- [Pure cuBLASLt reference](pure-gemm.csv): 1080 NN/TN results and selected library
  algorithm configuration; full148-SM device, no communication.
- [Inherited forward results](../v18.0/README.md).

Only final results are stored here: no candidate searches, before/after tables,
raw timing samples, profiling files or intermediate tuning logs.

B means fused data-gradient work, W means local weight-gradient work without
cross-CP reduction. These are separate phases, not a measured B+W end-to-end time.
BF16 input/output, FP32 accumulation, random inputs, Graph10+50. Fused numerical
and routing validation covers two payloads. Unstable or unavailable results stay
blank; where a new configuration was not validated, the prior stable configuration
is retained. Each recorded value is paired with its actual configuration.

Special MLA/KDA/gated routes and non-divisible heads remain TODO. Available GPU
memory limited some large-shape coverage. A complete catalog is not a claim that
every row ran successfully. Pure cuBLASLt validation uses a bounded numerical
sample, distinct from the fused full-output/routing validation.

Per-shape N256/K64 GEMM settings are explicit; defaults retain N128/K64. No
runtime per-shape winner lookup or communication-budget autotune is added to
backward. MXFP8 is subsequent work.
