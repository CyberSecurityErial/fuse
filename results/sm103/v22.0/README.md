# v22.0 — Experimental SM103 MXFP8 norm/RoPE forward fusion

Optional forward features, **disabled by default**. The original GEMM+A2A and
A2A+GEMM remain separate operating modes; their published results remain in
[v20.0](../v20.0/README.md) and [v21.0](../v21.0/README.md). This archive does not
replace their shorter measurement boundaries with augmented-kernel timings.

| New measured boundary | Passed / planned legal points | Geometric mean PFLOPS/GPU |
|---|---:|---:|
| Qwen3 QKV → Q/K head RMSNorm → RoPE → A2A | 3 / 3 | 1.785490 |
| Llama QKV → RoPE → A2A | 12 / 12 | 1.953058 |
| A2A → OProj → residual add → full-hidden RMSNorm | 16 / 18 | 1.743187 |

The combined QKV15-point mean is **1.918331P**, not evidence that both QKV
categories exceed1.9P. OProj remains below1.9P. Llama405B CP4/256K and512K
remain numerical failures and have no accepted throughput, not OOM skips.
The OProj mean describes the16 accepted points, not a complete18-point pass.

[Complete table](table.md) · [CSV](table.csv) ·
[Exact configurations, samples, replay commands and evidence](results.json) ·
[Model semantics and implementation](../../../benchmarks/sm103/ATTENTION_FUSION.md)

## Boundary and validation

PFLOPS/GPU = `2*M*N*K / complete augmented boundary time`. Dynamic BF16 master
weight quantization, MXFP8 GEMM, communication and requested postprocessing are
all included. MXFP8 activation/SFA and position-selected BF16 cos/sin arrive
from upstream; activation quantization and table construction are excluded.
Attention/MLP and the input RMSNorm preceding QKV remain outside these kernels.
The OProj residual sum is retained separately for the next MLP residual path.

Measurements use MPI Graph10+50, stable warmup, reproducible nonzero two-payload
inputs, per-sample maximum-rank time and its median/p95. Independent full
numeric/routing and postmeasurement validation use unchanged tolerance
`0.01 + 0.01*abs(reference)`; disputed OProj rows may use an independent FP64
reference refinement, never the actual output as an oracle. Remaining errors
are failures. Repeat checks are evidence for tested payloads, not a universal
bitwise-equivalence guarantee to BF16 model execution.

These are operator-level tests, not an integrated training or model-quality
claim. Sequences beyond checkpoint context limits are operator workloads only.

## Separate-operation controls

The JSON separately preserves31 matched pairs with identical per-pair GEMM,
communication and postprocessing settings:

| Direction | Pairs | Fused / separate throughput geometric mean |
|---|---:|---:|
| QKV | 15 | 1.090129x |
| OProj | 16 | 1.030788x |

The separate control runs the original projection/communication followed by
an optimized standalone postprocessing kernel, both timed in one Graph. It is
not the expensive correctness oracle or a globally tuned external norm library.
Two OProj control pairs regress slightly; improvement is not universal.
Some final primary configurations differ from these controls: do not divide
primary times by older separated times and call the result a matched speedup.
Historical v20/v21 entries exclude norm/RoPE and are explicitly different-boundary
references, not causal fusion gains.

## Configuration and reproduction

Use the explicit configuration and `replay` command of each JSON row after
building this checkout:

```bash
python3 scripts/l20d.py run fused-build --node 09 \
  --workspace /home/work/workspace_wct --mpi --mxfp8
```

No norm/RoPE flags means original behavior. `--qkv-postprocess rope` selects
RoPE-only; `--qkv-postprocess qknorm_rope` selects Q/K norm then RoPE.
`--oproj-postnorm` selects residual/full-hidden RMSNorm; the recorded
`--oproj-postnorm-overlap` additionally selects row-ready scheduling.
Positive communication budgets are required for these experimental services;
their costs are not silently substituted into the original Auto model.

QKV postprocessing currently requires complete128-dimensional heads; OProj
requires full-hidden width<=16384 and checked shared-memory capacity. Qwen3
CP8 QKV requires unimplemented KV replication. Qwen2.5 bias, BLOOM, KDA/MLA
special operations and other norm/residual orderings are not claimed supported.
MXFP8 and norm/RoPE backward are not included in this release.

Each direction's primary results use one attested binary, but QKV and OProj
binaries differ. Per-row hashes identify source, binary, environment and raw
artifacts retained under `fuse_midfile/l20d/<run_id>`. Reproduction builds the
released checkout, not a private frozen-source run. The final native files
match the accepted OProj build; later OProj-only changes did not rerun the whole
QKV matrix on that binary. No new full SM90/BF16/original-mode GPU regression
claim is made. Raw logs, profiles, search histories and dependencies are not
bundled with this release.

Release checks:363 related host tests passed; all31 primary and62 matched-control
measurements were re-audited from their local receipts/artifacts;70 native
`csrc/` and `include/` files match the accepted080035 build snapshot byte-for-byte.
