# Experimental norm/RoPE forward fusion (v22.0)

Optional, disabled by default. Original GEMM+communication modes and their
v20/v21 benchmark tables remain separate; these experimental features do not
replace them. MXFP8/norm/RoPE backward is not implemented in this release.

## Legal model boundaries

References: Transformers **v4.51.3**
[Qwen3 MoE](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py),
[Llama](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/llama/modeling_llama.py).

```text
upstream input RMSNorm -> quantized activation
  -> QKV GEMM -> Q/K head RMSNorm [Qwen3 only] -> Q/K RoPE -> A2A
  -> attention [outside these operators]
  -> A2A -> OProj GEMM -> residual add -> full-hidden RMSNorm
  -> MLP [outside these operators]
```

A2A permutes whole heads/tokens. Their independent norm/rotation can precede
that permutation using the original token's position tables. Llama has RoPE
but no Q/K norm. There is **no OProj RoPE**. The retained BF16 residual sum
is a separate output needed by the subsequent MLP residual connection.
These are forward kernels, not norm/RoPE backward implementations.

The operator consumes upstream position-selected BF16 cos/sin, not guessed CP
positions. Tests use rank-major global positions. Qwen3 uses base1e6 and
epsilon1e-6. Llama3.1 uses base500000, factor8, low1/high4, original8192 and
epsilon1e-5, following its [configuration](https://huggingface.co/nvidia/Llama-3.1-405B-Instruct-FP8/blob/b2cbce00b6238d2fe04939ee8c9851c78ac83046/config.json)
and [RoPE implementation](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/modeling_rope_utils.py).
Beyond-checkpoint-context sequences are operator extrapolation workloads,
not model quality/context-length claims.

Qwen3-235B has H4096/Q64/KV4/D128: CP8 QKV needs KV replication and is not
silently replaced by eight independent KV heads. Qwen2.5 QKV bias and
BLOOM/KDA's different position/norm operations are outside this prototype.

## Numerical contract

MXFP8 A/SFA arrives prequantized; BF16 master W is dynamically quantized
inside the persistent kernel. FP32 accumulation produces BF16 projection.
Residual add rounds to BF16. Norm reduces FP32, rounds normalized values to
BF16, then multiplies BF16 gamma. Split-half RoPE preserves BF16 product/add
rounding, without FMA contraction. V and raw local QKV projection are retained.

The QKV route oracle independently evaluates the specified FP32 head tree
and BF16 arithmetic with scalar arrays; transformed routing is byte-exact.
Raw GEMM is checked independently against cuBLAS on represented operands.
OProj uses a separate full-hidden FP64 norm reference after cuBLAS GEMM.
The numerical tolerance remains **0.01 + 0.01 |reference|**.

Rare GEMM accumulation differences near BF16 halfway points can be amplified
by residual cancellation and norm. An independent FP64 GEMM + norm reference
may refine at most eight disputed complete rows/rank/check. It never uses
actual output to construct expected values. Original disagreements and
refinements are logged; surviving errors remain failures, not waived points.
The tests additionally compare both normalized output and residual sum
byte-for-byte across three repeated launches of each payload and after Graph
measurement against the pre-measurement snapshot. These checks are untimed.

## Implementation

QKV transforms a complete BF16 head in its existing communication SMEM slot:

```text
GEMM ready -> TMA G2S -> Q/K norm -> RoPE -> TMA S2G
                           V bypasses both ----^
```

Two lanes own each head. Each lane keeps both split-half RoPE partners in
registers; sixteen rows run concurrently per warp. Packed BF16 pairs avoid
repeated scalar conversions. Only the norm crosses lanes. An async-proxy
fence publishes generic SMEM writes before TMA reads. GEMM, weight
quantization, route ownership and producer-ready granularity are unchanged.

OProj has a full-grid norm tail control and an explicit
`overlap_postnorm=true` experimental row-ready schedule:

```text
A/W CTA finishes its own work --\
                                +-> claim 8 rows -> all N ready -> full norm
GEMM CTA finishes its own work -/
```

Every worker uses the same virtual256-thread reduction tree. The current
candidate normally groups four rows per CTA:64 physical threads/row each own
four independent virtual-thread partials. When dual inputs for four rows
exceed existing SMEM but two rows fit, it selects two rows to retain cp.async
input staging;128 physical threads/row each own two virtual partials. This is
a capacity check, not model-name dispatch or an online timing search. A local CTA barrier
precedes SMEM reuse. Eight-row tasks never cross an M128 panel. Readiness is
acquired for every N256 output tile, published only after its output TMA has
fully drained. No worker waits for norm with its own original production or
GEMM tasks still unissued. The single atomic row queue and flags are reset at
each existing initialization barrier, including reused Graph epochs. Original
A/W/GEMM schedules, H/P windows and ready units remain unchanged.

Explicit communication budgets only: prior Auto calibration did not measure
these added services. The API rejects unsupported automatic postnorm choices
instead of silently reusing an unrelated cost model.

## Optional modes and final results

No postprocessing flags means original GEMM+communication behavior.
QKV selects `--qkv-postprocess rope` (Llama) or
`--qkv-postprocess qknorm_rope` (Qwen3); OProj selects
`--oproj-postnorm`, with `--oproj-postnorm-overlap` for row-ready scheduling.
Public parameter structs default their optional postprocessing pointers to null.
A mode describes explicit semantics, not a model-name switch in the kernel.
These are the supported boundaries, not a universal norm/RoPE ordering for
Gemma, MLA or every model sharing a projection shape.

Final results are maintained in [v22.0](../../results/sm103/v22.0/README.md):
QKV15/15 combined GM1.918331P (Qwen3 three points1.785490P, Llama twelve
points1.953058P), OProj16/18 accepted points GM1.743187P. The two
Llama405B CP4/256K and512K numerical failures remain explicit. This does not
claim every category exceeds1.9P or all planned OProj points pass.

Count original GEMM FLOPs over the complete augmented boundary, per GPU:
dynamic W quantization, A2A and requested postprocessing are included.
Activation quantization and position-table construction are upstream.
Use uninstrumented MPI Graph10+50 and two reproducible nonzero payloads.
Original projection-only tables, augmented tables and matched separate-operation
controls remain distinct. Final configurations are explicit, not a new Auto
rule or a claim of global optimality.

## Separated controls

QKV `--qkv-postprocess-separate` runs the original dynamic-W GEMM/A2A then
an optimized received-head norm/RoPE kernel in one ordered two-kernel Graph.
It uses the two-lane arithmetic on strided GMEM, not the scalar correctness
oracle. V and raw local projection are retained. Destination position tables
cover all rank-major global rows, constructed before the measured boundary.

The second QKV kernel is cooperative with an occupancy-bounded grid. After a
grid join it reuses all-rank release/acquire completion. Epoch e maps to raw2e
and post2e+1, so the next A2A cannot overwrite a slower rank's postprocessing.
The additional completion is timed, not replaced by an untimed host barrier.

OProj `--oproj-postnorm-separate` runs the original A2A/GEMM then an optimized
ordinary-grid residual/RMSNorm kernel, both timed in one Graph.
Per-pair GEMM/communication/window settings match. Neither separate reference
is a claim of globally optimal standalone norm/RoPE.
The final archive keeps these controls separately because later primary
configuration choices can differ.

## Profiling

Follow [PROFILE_PROTOCOL.md](../../PROFILE_PROTOCOL.md). Profiling results
never substitute for uninstrumented formal performance.
The whole-collective epilogue observer records its return after publication;
captures identify `whole_collective_v2`. Legacy traces use different phase
boundaries and must not be mixed with it. Norm phase sums are metadata,
not fabricated child spans. Large traces are not part of the release.
