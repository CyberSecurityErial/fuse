# Frozen TE/cuBLAS + NCCL baseline

这份基线固定正确的 Ulysses 前向边界：

```text
QKV projection -> Q/K A2A
V A2A -> PV BMM
```

## 固定问题规模

| 项目 | 数值 |
|---|---:|
| GPU / CP | 8 × H200-class，SM90，132 SM/GPU |
| 模型参数 | B=1，S=4096，hidden=8192，Hq=64，Hkv=8，D=128 |
| QKV GEMM | M=512，N=10240，K=8192，L=1 |
| Q/K A2A | 9 MiB/GPU，总计 1.125 MiB/peer |
| V A2A | 1 MiB/GPU，总计 128 KiB/peer |
| PV BMM | M=4096，N=128，K=4096，L=8 |
| dtype | BF16 input/output，FP32 accumulate |

每个样本先在各 rank 记录 CUDA event，再对 8 个 rank 的耗时取最大值。正式数据为三次独立运行，每次 10 次 warmup 和 50 次采样。

## 冻结实现和参数

- TE：`transformer_engine.pytorch.Linear`，无 bias，inference mode。
- cuBLAS：`torch.mm(..., out=...)`；PV 使用 `torch.bmm(..., out=...)`。
- NCCL：`torch.distributed.all_to_all_single`，NCCL grouped send/recv。
- Q/K 和 V pack：一个 Triton kernel，`BLOCK=1024`、4 warps。
- B=1 且每 rank 一个 KV head 时，NCCL receive buffer 已经是所需 global-sequence 顺序，unpack 只做 view，不再启动 copy kernel。
- 整条边界使用 CUDA Graph，NCCL 使用高优先级 stream。
- NCCL 固定参数：

```text
NCCL_MIN_P2P_NCHANNELS=1
NCCL_MAX_P2P_NCHANNELS=32
NCCL_P2P_NVL_CHUNKSIZE=524288
NCCL_P2P_LL_THRESHOLD=16384
NCCL_GRAPH_REGISTER=1
NCCL_LOCAL_REGISTER=1
NCCL_IB_DISABLE=1
```

NCCL 2.27.5 会把 P2P channel 数向上取为 2 的幂，因此 20/24/28 的测试实际上仍落到 32；有效 channel 候选是 8/16/32。最终还覆盖了 128/256/512/1024 KiB NVLink chunk，以及 16/64/128 KiB LL threshold。16 channel、较小 chunk 和强制小消息走 LL 均退化。

## 最终数据

主值是三轮 p50 的中位数；尾延迟是三轮 p95 的中位数。每一轮和每个原始样本都保留在 JSON 中，没有删除离群点。

| 路径 | 三轮中位 p50 (ms) | 三轮中位 p95 (ms) |
|---|---:|---:|
| TE QKV GEMM | 0.1502 | 0.1532 |
| cuBLAS QKV GEMM | 0.1510 | 0.1545 |
| NCCL Q/K A2A，含 pack/view | 0.0614 | 0.0633 |
| TE QKV GEMM + Q/K A2A | 0.2107 | 0.2135 |
| cuBLAS QKV GEMM + Q/K A2A | 0.2102 | 0.2147 |
| NCCL V A2A，含 pack/view | 0.0277 | 0.0310 |
| cuBLAS PV BMM | 0.0915 | 0.0933 |
| NCCL V A2A + cuBLAS PV BMM | 0.1163 | 0.1209 |
| **TE + NCCL 两个边界联合** | **0.3118** | **0.3180** |
| **cuBLAS + NCCL 两个边界联合** | **0.3149** | **0.3193** |

当前 fused boundary pair 的同机旧正式值是 p50 0.3192 ms、p95 0.3225 ms。强化后的 TE+NCCL 用时是它的 97.7%，cuBLAS+NCCL 用时是它的 98.6%。因此旧的 0.36–0.42 ms 外部基线不再用于胜负结论；后续融合优化必须超过这份冻结基线。

## 复现和证据

单次正式运行：

```bash
bash scripts/run_frozen_te_cublas_nccl.sh \
  --json-out results/te_cublas_nccl_frozen_run.json
```

三轮聚合：

```bash
python scripts/summarize_frozen_baseline.py \
  results/te_cublas_nccl_frozen_graph_triton1024_highprio_10w50i_run1.json \
  results/te_cublas_nccl_frozen_graph_triton1024_highprio_10w50i_run2.json \
  results/te_cublas_nccl_frozen_graph_triton1024_highprio_10w50i_run3.json \
  --json-out results/te_cublas_nccl_frozen_graph_triton1024_highprio_3x10w50i_summary.json
```

冻结配置哈希：`59e410abb19164945ba6be50707ea5dcfbfa58b2e5adf19d410d43be6d471d3a`。

正式原始数据：

- `results/te_cublas_nccl_frozen_graph_triton1024_highprio_10w50i_run1.json`
- `results/te_cublas_nccl_frozen_graph_triton1024_highprio_10w50i_run2.json`
- `results/te_cublas_nccl_frozen_graph_triton1024_highprio_10w50i_run3.json`
- `results/te_cublas_nccl_frozen_graph_triton1024_highprio_3x10w50i_summary.json`
