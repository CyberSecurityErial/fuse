# SM90 MXFP8-weight temporary development snapshot

Development is parked while work moves to the B-series GPU operators. This is
a continuation checkpoint, not a production release or a completed performance
milestone. No GPU benchmark was rerun to package this branch.

## Code and semantics

- Source checkpoint: `08450d0`, plus the uncommitted `operator_sweep.py` external
  diagnostic-trace flag and its CPU test. The original development branch is
  `feat/sm90-mxfp8-2f2b`; it remains available locally with its measurement history.
- Remote snapshot: `tmp/sm90-mxfp8-2f2b-20260906`, a single commit on
  `16de0b2b0f49534361277d352b9a815cb1df3133` (`origin/main` when packaged).
  The development commits containing raw captures are not ancestors of this
  snapshot.
- Four SM90 operators: QKV GEMM -> A2A forward, A2A -> OProj GEMM forward,
  and the corresponding QKV/OProj backward data and weight gradients.
- Weights use offline E4M3 values and E8M0 scales per 32 original forward-K
  elements. Runtime software dequantization produces BF16 for BF16 Tensor Core
  GEMM. Activations, communication and data gradients remain BF16; dW is FP32,
  with beta=0 overwrite and beta=1 accumulation.
- Flattened global tokens are the operator input length; no training batch,
  optimizer, KDA, or end-to-end model execution is added.

Start with [BENCHMARK.md](BENCHMARK.md) for build and execution commands and
[SEARCH_PROTOCOL.md](SEARCH_PROTOCOL.md) for baseline search. Reusable profiling
code and its tests are retained; generated captures are not.

## Retained measurements

Under `results/mxfp8_weight/`:

- `published/`: complete forward/backward baseline tables, best measured
  configurations, and export hashes.
- `operators/full_model_v1/comparison/`: the latest complete four-operator
  comparison, including Eager/Graph, CP4/CP8, all six lengths, both backward beta
  modes, pure cuBLAS references, and the full-table/badcase handoff documents.
- `policy_ab/*summary/`: compact policy A/B reports; the QKV forward report
  includes both paired execution orders.
- Selected `calibration/`, `sweep/`, and `validation/` JSON summaries retain the
  frozen models, accepted/rejected candidate conclusions and correctness evidence.

Raw per-rank timing archives, intermediate searches, logs, Perfetto/NSYS/NCU
captures, SQLite exports, binaries and build directories are intentionally not
published here. No file under `results/mxfp8_weight/profile/` is included.
Original local evidence is not deleted by this packaging operation.

Historical paths and source/DSO hashes in reports refer to the original runs;
they have not been rewritten to suggest that this snapshot was measured.
Some historical document links therefore point to local-only evidence. The
reports can be read from a fresh checkout, but re-auditing every raw sample or
refitting a model from its original profile input requires those local archives
or a new run. The files listed above are the portable handoff artifacts.

## Measured status and remaining work

The last complete long-sequence comparison is 1.151840x with boundary-equal
weighting, or 1.169852x with four-operator-equal weighting. The whole matrix,
including short sequences, is 1.038383x with boundary-equal weighting. These
are not current-HEAD reruns and are not end-to-end training speedups.

CP4 QKV forward's explicitly enabled frozen communication model achieved
1.065537x versus same-run auto in the balanced paired test. Against the archived
external baseline its long-sequence GM is 1.166200x. Thirteen boundaries remain
below the original scope mean: all Q3/Q8 long Eager/Graph points and Q7/128K
Graph. The model is opt-in; it is not a blanket default change.

CP4 QKV backward Q3/Q8, Graph, both beta modes: reducing communication CTAs
from 12 to 4 improved complete B -> W by 1.019534x versus same-run auto in the
balanced paired test. This is not an external-baseline or full-matrix speedup.
The Q2/Q7 c8 candidate was rejected: 0.995826x overall, worst
Q7/128K/beta0 0.952185x. Backward production defaults were not changed.

Large-matrix backward profiling observed high W Tensor activity (about 96-97%)
and substantial loaded-clock variability. This does not establish a hardware
ceiling or prove that all remaining loss is communication. The next uncompleted
diagnostic was to reproduce Q7/128K/beta0 under the full paired B -> W execution
history; it was not launched before handoff.

The latest CPU suite passed 299 tests before packaging. The added sweep
`--diagnostic-trace` flag is covered by CPU contract tests but has not yet been
GPU-validated. It preserves execution counts/order, adds NVTX arm ranges, and
marks externally traced output non-formal; never publish its timing as formal
performance. Previous GPU correctness and performance records remain historical.

Resume one operator scope at a time, prioritize 128K/256K/512K badcases, measure
compute/communication service times and the resource crossover, and validate
generic tile/CTA models without model-name or per-shape winner dispatch.
