# Large-model attention projection coverage

## Shared-input packing (current production comparison)

Use `projection_shapes.py --grouped` for the primary pure-GEMM comparison.
The default native-Linear catalog remains a diagnostic inventory, not the
optimized execution grouping. Concatenate weights along N only when the
projections consume the same tensor. Equal K alone is insufficient.

- Kimi-K3 KDA: Q/K/V + F-down + beta + full-rank G, N=49376, K=7168.
- Kimi-Linear KDA: Q/K/V + F-down + beta + G-down, N=12576, K=2304.
- KDA F-up and G-up consume different latent tensors and remain separate.
- MLA input groups: DeepSeek-V3 Q-down+KV-down (2112×7168), GLM-5
  Q-down+KV-down (2624×6144), Kimi-K3 Q-down+KV-down+G (14400×7168),
  Kimi-Linear full Q+KV-down (6720×2304). Normalized Q/KV up-projections
  are distinct inputs, not a single concatenated-weight GEMM.
- Qwen3.5 GDN QKV/Z/beta/a share the hidden-state input; full-attention
  Q+gate/K/V form a separate packed group in their own layer type.

The existing `kimi_*_kda` boundary registration measures QKV-only exchange.
It does **not** implement the six-projection boundary above. Keep its prior
measurements as QKV-only diagnostics; do not publish them as full KDA input
projection results. Gate/latent ownership and packing require a routing adapter.
Grouped catalog arithmetic has CPU tests; GPU validation is recorded separately.

### Runtime reference and CP ownership

Runtime reference: SGLang revision `5aab054ec8ce6b6100fbfb7aafe67d632a7df3aa`,
`python/sglang/srt/models/{kimi_linear,kimi_k3,deepseek_v2}.py`.
Kimi-Linear's `fused_qkvbfg_a_proj` concatenates Q/K/V/beta/F-down/G-down;
Q/K/V/beta are column-sharded, F-down/G-down are replicated. Its two up
projections use `ColumnParallelBatchedLinear`, not weight concatenation over
one common latent input. K3 CUDA packs Q/K/V/G but separates F-down/beta;
the ROCm whole-input option packs all six. Our six-way K3 probe follows the
user-requested legal shared-input packing, not a claim of CUDA runtime parity.
DeepSeek-style MLA packs Q-down+KV-down as a replicated projection and keeps
normalized up projections separate. K3 output gate is separate in SGLang;
Kimi-Linear's no-Q-LoRA MLA uses separate full-Q and KV-down. The catalog's
larger shared-input MLA groups are legal candidate GEMMs, not those exact
runtime launch groups.

For CP adaptation, source ranks own token shards and full projection weights.
Head-owned outputs route to their destination head rank; shared low-rank outputs
must be available on every consuming head rank, following the runtime's
replicated ownership. This is an adapter design, not yet a measured fused path.
The backward adjoint reverses the head permutation and **sums** contributions
for replicated outputs; a copy-only reverse would be incorrect. Projection
grouping/weight offsets must remain identical in forward and backward.

The registry in `projection_shapes.py` is opt-in and SM103-only. Historical
SM90 code and released performance tables stay intact. The production boundary
default now excludes 1K/4K: 128 points across both directions. Historical
192-point readers and explicit small correctness diagnostics remain available.
These are BF16, bias-free projection **geometry** tests, not full-model runs.
No weights or remote model code are downloaded by the benchmark.

## Pinned sources and shapes

All dimensions below are `N × K` in `Y[M,N] = X[M,K] W[N,K]^T`.
Set `M = global_sequence / CP`; no tensor-parallel division is implied.
Revisions are pinned in `SOURCES`, including each cited config and corresponding
`modeling_deepseek.py`, `modeling_kimi_linear.py` or `modeling_kimi.py` implementation.

| Model | Attention | Main input projections N × K | Output projection N × K |
|---|---|---|---|
| Qwen3.5-397B | Gated DeltaNet + gated full attention | GDN packed QKV 12288×4096; Z 8192×4096; full Q+gate 16384×4096; full K/V each 512×4096 | 4096×8192 |
| GLM-5 | MLA + sparse indexer | Q-down 2048×6144; Q-up 16384×2048; KV-down 576×6144; KV-up 28672×512 | 6144×16384 |
| Qwen2.5-72B | GQA | Q 8192×8192; K/V each 1024×8192 | 8192×8192 |
| Qwen3-235B-A22B | GQA | Q 8192×4096; K/V each 512×4096 | 4096×8192 |
| BLOOM-176B | MHA | Native packed QKV 43008×14336 | 14336×14336 |
| DeepSeek-V3 | MLA | Q-down 1536×7168; Q-up 24576×1536; KV-down 576×7168; KV-up 32768×512 | 7168×16384 |
| Kimi-K3 KDA layers | KDA | Q/K/V each 12288×7168; full-rank G 12288×7168 | 7168×12288 |
| Kimi-Linear-48B KDA layers | KDA | Q/K/V each 4096×2304; G-down 128×2304; G-up 4096×128 | 2304×4096 |

The KDA registry also includes decay F-down/F-up and beta projections; Kimi's
hybrid MLA layers and K3 MLA output gate are included separately. Names retain
every logical operation even when N/K coincide. The existing GEMM runner groups
exact M/N/K and retains aliases, so equal Q/K/V shapes do not launch three tunings.
Do not sum isolated GEMM times as a complete fused attention latency.

- [Qwen2.5-72B config](https://huggingface.co/Qwen/Qwen2.5-72B/blob/efba10c8e54e91e0d9570ab5f7b51a958474d4cb/config.json)
- [Qwen3.5 config](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/8472618112abcbd45acbcdc58436aff4233c23f7/config.json) and [projection implementation](https://github.com/huggingface/transformers/blob/8eaf75f84e0ef68ccdaac14b739ace53a962bbee/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py)
- [GLM-5 config](https://huggingface.co/zai-org/GLM-5/blob/c183ef8c61faee82855eca1ed9bb3a9a7ce3b0b2/config.json) and [projection implementation](https://github.com/huggingface/transformers/blob/8eaf75f84e0ef68ccdaac14b739ace53a962bbee/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py)
- [Qwen3-235B config](https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/8efa61729e24bd65b1d152b5ab5409052aa80e65/config.json)
- [BLOOM config](https://huggingface.co/bigscience/bloom/blob/7f10a99ce7c08f03c7719a586cb2cbda1433ac05/config.json)
- [DeepSeek-V3 implementation](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/modeling_deepseek.py)
- [Kimi-K3 implementation](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py)
- [Kimi-Linear implementation](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/e1df551a447157d4658b573f9a695d57658590e9/modeling_kimi.py)

## Run with existing tools

Long sequences default to 64K/128K/256K/512K, CP4/8. These are training-oriented
geometry stress lengths, not claims that every checkpoint supports those contexts.
Export one model per matrix to respect the existing 256-logical-row bound:

```bash
python3 benchmarks/sm103/projection_shapes.py --models kimi_k3
```

This prints `sm103_gemm_matrix_v1` JSON. Save the output in the normal midfile
directory and pass it to the existing workflow, e.g.:

```bash
python3 scripts/l20d.py run gemm-probe --node 09 \
  --gemm-matrix /path/to/kimi-k3-matrix.json \
  --launches graph --experiment large-projections-v1
```

The controller selects only the first GPU of its device list for this pure-GEMM
stage (GPU0 by default); it does not launch a CP4/8 process group.
The existing sampler retains random inputs, 10+50, convergence checks, numerical
checks, and same-process shape reuse. A successful pure-GEMM probe is **not** a
validated distributed fusion or TEUB comparison.

For conventional head-routing boundary plans, `bench.py --models` now accepts
`qwen25_72b`, `qwen3_235b`, `bloom_176b`, `kimi_k3_kda`, and
`kimi_linear_48b_kda`. KDA entries pack only the three equal-width Q/K/V
projections then perform canonical token/head exchange; OProj uses the inverse
exchange. They exclude short convolution, decay/gating and the recurrent core.
This is a projection-boundary benchmark, not a complete KDA layer measurement.
The independent catalog retains the native three separate Linear operations.
Qwen3 QKV supports CP4 in the current
planner, but rejects CP8 because four KV heads cannot be split into eight equal
nonempty head shards. OProj has no such KV constraint. Do not duplicate KV heads
just to bypass this check. BLOOM uses benchmark canonical Q/K/V packing, not its
checkpoint's head-interleaved native packed ordering; bias is excluded.

MLA and the remaining KDA/gated projections are present in the pure-GEMM bench.
Their additional communication adapters are not implemented here: rank bottlenecks, intervening
normalization/activation, KDA convolution placement and gate delivery need their
actual producer/consumer contracts. MQA needs replicated/shared-KV semantics and
must not be faked as GQA with more KV heads. Registry inclusion alone is not a
claim of device validation, tuning completion, or a new released performance result.

## Initial device validation (2026-09-08)

- Node09 GPU0, run `20260908-101612-575cd9`: all eight catalog models at
  M=16384 (64K/CP4 token-shard geometry), 59 logical projections / 38 unique
  MNK, BF16 Graph, 10+50 and numerical spot checks passed. This is a pure GEMM
  probe, not 59 distributed tests or a full long-sequence sweep. Raw receipt:
  `fuse_midfile/l20d/<run-id>/artifacts-attempt1/control/gemm-probe.json`.
- Node09 CP4/64K full fused numeric/route/two-input checks passed for Qwen2.5-72B
  (`20260908-101754-7bbc43`) and BLOOM MHA geometry (`20260908-101832-2ef00e`).
  Fixed N256/K64/E32, comm CTA16, Graph, rank rotation OFF; these are first
  validated configurations, not per-shape tuned winners.
- Node09 also passed the previously memory-limited Llama405B CP4/512K pair
  (`20260908-100443-395699`, comm CTA8). Do not overwrite old release tables.
- Node0a's Qwen CP8 attempt was rejected before launch when utilization rose
  to 65%; it is not a successful result. No service was killed during testing.
- Same-layout TEUB/cuBLASLt+NCCL measurements for the newly added boundary
  cases initially remained unmeasured. The latest scope keeps existing TEUB/
  cuBLASLt+NCCL records but does not fill missing reference-communication rows;
  new measurements compare self-developed fusion with same-M/N/K pure cuBLASLt.

## Forward/backward grouping contract

Use the same ordered projection segments, offsets and rank ownership for forward
and backward. The input-gradient path reverses the forward exchange and applies
the transposed GEMM; weight gradients are a separate required GEMM, not included
implicitly in that reversed path. A forward replicated segment requires gradient
summation in backward, not merely the inverse copy. Different branches with
different post-normalization/activation inputs must not be concatenated just
because their dimensions match. Layout/group metadata should be shared between
the two directions; tile and communication CTA configurations may differ.

The non-profile fused harness accepts `--fused-direction qkv|oproj|both` (default
`both`). A single-direction run owns only that direction's inputs, weights,
outputs and full reference buffers. In particular, OProj does not require KV
heads divisible by CP. This does not make the corresponding QKV boundary valid.
Profiling retains its existing `--profile-direction` contract.
