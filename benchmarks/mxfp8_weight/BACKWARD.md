# Backward benchmark: completed finite-candidate baseline

The branch includes portable selected results and a separate production Fuse
implementation. Start with [BENCHMARK.md](BENCHMARK.md). The raw-run paths below
refer to the original archive recorded in the published metadata, not files
that must already exist in a fresh checkout.

The full run in `results/mxfp8_weight/backward_full_v1/` has passed the strict
192-setting / 1536-row FP32 audit. The four-operator audit also passes: 768
forward + 1536 backward = 2304 formal rows. This is finite-candidate benchmark
completion, not an exhaustive performance ceiling and not a new Fuse kernel.

* [Full backward table](../../results/mxfp8_weight/backward_full_v1/REPORT.md)
* [All 1536 rows as CSV](../../results/mxfp8_weight/backward_full_v1/coverage.csv)
* [Four-operator scope and evidence](../../results/mxfp8_weight/backward_full_v1/FOUR_OPERATORS.md)
* [Plain-language search protocol](SEARCH_PROTOCOL.md)

The finished forward suite is **not** four-operator coverage. Backward completion
requires QKV and OProj, 96 shape/sequence/CP settings each, from the **backward**
SM90 registry. Global sequences are 1024/4096/16384/131072/262144/524288; CP4/8,
batch=1, M=S/CP. Formal layout is causal paired for both backward operators.
Rank-major and batch=2 causal are additional correctness cases, not substitutes.

| Operator | B boundary | W boundary |
|---|---|---|
| QKV backward | inverse Q/K/V pack + A2A + unpack + runtime weight DQ + dX | dQKV transpose × saved X |
| OProj backward | runtime weight DQ + dA + pack + A2A + unpack | dY transpose × saved attention |

Weights remain stored in their original forward [output,input] orientation with
one E8M0 scale per 32 original input-axis values. Dgrad uses these exact weights,
not a separately requantized transpose. GEMMs consume BF16; offline quantization
is excluded; DQ is inside every B/full invocation and graph replay. W-only does
not consume a weight and must not acquire an artificial DQ charge.

## Important dtype correction

The published SM90 backward docs and `backward_te_nccl_baseline.py` actually use
**BF16 dW/main_grad**, with FP32 GEMM accumulation. They do not store main_grad in
FP32. Earlier chat wording claiming otherwise was incorrect.

The user explicitly confirmed that the **new formal main table uses FP32
dW/main_grad**. Shape, CP, sequence, layout and timing boundaries match the old
SM90 suite, but the different gradient-storage precision is explicitly labelled.

* `grad_dtype=fp32`: required new formal main table (default).
* `grad_dtype=bf16`: optional historical-semantic diagnostic, not required full
  coverage. Never claim an FP32-vs-historical-BF16 comparison is quantization-only.

Both have ordinary immediate B→W (beta=0) and deferred B/W (beta=1). Nonzero
initial main_grad and two consecutive beta=1 updates must pass correctness. The
rounding boundary is the selected output dtype on every update. No optimizer or
gradient all-reduce is added: neither is inside the old operator boundary.

192 settings × 2 backends × 2 launches × 2 scheduling modes = **1536 formal rows
for the required FP32 table**. Each row includes separately measured B,
W and full sequential B→W statistics; a sum of isolated times is labelled as such
and is not claimed to be a measured combined time or ZeroBubble scheduling gain.

Backends: cuBLASLt+NCCL and TE Userbuffers P2P+cuBLASLt, same meaning as the forward
suite (not native TE MXFP8 GEMM). Preserve 10 warmups/50 formal samples, Eager and
Graph separately, sample-wise rank-max. Reuse processes, preallocated buffers and
events, bounded tuning and independent formal samples. All-rank finite checks,
reference route/dgrad/wgrad checks, raw rank samples and source hashes are required.

The criteria above are verified by the raw formal outputs, `coverage.json`, and
`four_operator_coverage.json`. CP4/CP8 validation covers both layouts and batch
sizes, actual autograd, nonzero two-step accumulation, and one-rank corruption.

## Reproduction

```bash
/usr/local/cuda/bin/nvcc -shared -Xcompiler -fPIC -O3 -lineinfo \
  benchmarks/mxfp8_weight/backward_gemm.cu -lcublasLt -lcudart \
  -o build-mxfp8-bench/libmxfp8_backward.so

/home/chen/miniforge3/envs/mmunlearner/bin/python benchmarks/mxfp8_weight/test_backward_matrix.py
/home/chen/miniforge3/envs/mmunlearner/bin/python benchmarks/mxfp8_weight/test_backward_report.py
/home/chen/miniforge3/envs/mmunlearner/bin/python benchmarks/mxfp8_weight/test_backward_sources.py
/home/chen/miniforge3/envs/mmunlearner/bin/python benchmarks/mxfp8_weight/test_backward_gemm.py

# Run correctness serially for CP4 and CP8 (change CUDA_VISIBLE_DEVICES and nproc).
CUDA_VISIBLE_DEVICES=0,2,4,5 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 \
  /home/chen/miniforge3/envs/mmunlearner/bin/torchrun --standalone --nproc-per-node=4 \
  benchmarks/mxfp8_weight/backward_bench.py --validation --warmup 2 --iterations 3 \
  --sweep-warmup 2 --sweep-iters 3 --candidates 8 --ub-sms 8 \
  --output results/mxfp8_weight/backward_validation/cp4.json

# Full finite tuning + independently remeasured formal table; serialized GPU jobs.
/home/chen/miniforge3/envs/mmunlearner/bin/python benchmarks/mxfp8_weight/backward_suite.py

# Re-audit completed forward + backward + distributed validation evidence.
/home/chen/miniforge3/envs/mmunlearner/bin/python benchmarks/mxfp8_weight/four_operator_report.py
```

NCCL: the three environment profiles from the forward suite, each in a fresh
process. Select minimum B p50 per shape/launch using 3/12, regroup by winner,
and run fresh 10/50. TEUB: communication SM counts 4/8/16, same short/final
separation. Since W consumes no quantized weights or communication, B policy
selection is shared across gradient modes. cuBLASLt evaluates up to 32 heuristics
with 2 warmups/6 batched repetitions, tuning the actual beta and output dtype.
"Best" means the winner of this finite set, not an exhaustive optimum.

Pure BF16 and quantized-effective weights are not same-run paired A/B here.
Reported backend speedups compare TEUB/NCCL under the **same new FP32** semantics.
`backward_suite.py` resumes completed case checkpoints but rejects changed source,
device, environment or protocol fingerprints. It locks its output directory;
do not run another GPU suite concurrently. Raw reports retain all rank samples,
source hashes and rank0 algorithm metadata. Eager times can include CPU submission
gaps; CUDA Graph uses one full invocation per replay.
