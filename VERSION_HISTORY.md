# 版本演进手册

本文记录每个版本对 A2A + O-projection 的有效改动、代码差异、测量结果和已知边界。正式数据统一使用 BF16、10 次 warmup + 50 次采样，并先取每次采样的跨 rank 最大延迟，再统计 p50。

## v1.0：可复现的 A2A + O-projection Golden

状态：已发布。

核心改动：

- 将 inverse A2A 和 O-projection GEMM 放进单个 cooperative persistent kernel。
- 通信 CTA 把远端 head shard 直接写入 GEMM 的最终 A layout；GEMM CTA 按 peer/K shard 等待 ready epoch。
- 运行时从 `M64N128`、`M128N128`、`M128N160`、`M128N256 cluster-M2` 中选择 tile policy，并自动选择通信 CTA 数。
- 固化 TE、cuBLASLt、NCCL 和 TE Userbuffers 的搜索与正式复测流程，形成 CP4/CP8、三种宽度、六档序列长度的 36-case Golden。

效果：

- 36/36 exact correctness 通过。
- 相对调优后的 TE+NCCL/cuBLASLt+NCCL 分离实现，36 个 setting 全部加速。
- Golden tag、源码、复现命令和结果目录一并归档。

## v2.0：cluster-aware frontier 与自动通信 CTA

状态：已发布。

### 实验性通信 CTA 模型

v2 源码保留了一套覆盖短、长序列的通信 CTA 成本模型，默认关闭：

```bash
export FUSE_A2A_LHS_COMM_POLICY=experimental_model
```

启用后，运行时会枚举 `comm_ctas={2,4,6,8,10,12,14,16,20,24}`，联合估算 GEMM tile/wave、通信 task wave、peer shard 行宽、NVLink 带宽和通信 CTA 对计算资源的占用，再选择最低分。模型参数来自 CP4/CP8 的36-shape sweep；目前跨硬件和未覆盖 shape 的泛化性证据不足，因此不属于 Golden，也不作为版本性能结论。默认路径继续使用可审计的 `M` 与通信/计算比规则；显式传入 `--comm-ctas N` 时，手工值优先于两种自动策略。

### 问题

旧策略用 compute CTA 数量计算 wave。`cluster-M2` 中两个 CTA 是一个不可拆分的调度单元，因此真实容量应按 compute cluster 计算。

以 CP8、`M×N×K=16384×5120×5120`、`comm_ctas=4` 为例：

- 128 个 compute CTA 对应 64 个 cluster；
- `N256` 产生 20 个 N tile，`64 % 20 != 0`；
- 一个 wave 会在同一 M tile 的 N frontier 中间截断，已经到达的数据要跨 wave 才能被完整消费；
- `N320` 产生 16 个 N tile，`64 / 16 = 4`，每个 wave 恰好处理 4 个完整 M frontier。

### 核心 diff

策略计数从 CTA 改成 cluster：

```diff
- wave_capacity = compute_ctas
- tile_count = m_tiles * n_tiles * batch
- waves = ceil_div(tile_count, compute_ctas)
+ compute_clusters = compute_ctas / cluster_m
+ cluster_tile_count = ceil_div(m_tiles, cluster_m) * n_tiles * batch
+ waves = ceil_div(cluster_tile_count, compute_clusters)
```

增加 frontier 对齐判定：

```diff
+ frontier_aligned =
+     waves == 1 ||
+     compute_clusters % n_tiles == 0 ||
+     n_tiles % compute_clusters == 0;
+
+ full_last_wave =
+     cluster_tile_count % compute_clusters == 0;
```

扩充自动候选并调整选择优先级：

```diff
- candidates = {M64N128, M128N128, M128N160, M128N256-C2}
+ candidates = {M64N128, M128N128, M128N160, M128N256-C2,
+               M128N320-C2}
+
+ selection priority:
+   1. frontier_aligned
+   2. estimated_cycles，包含 wave efficiency、L1/L2 流量和 tile 开销
+
+ N320 仅在对齐 frontier 或改善非满尾 wave 时进入自动选择。
```

同时将 `compute_clusters`、`n_tiles`、`last_wave_clusters`、`frontier_aligned` 和 `full_last_wave` 写入正式结果，便于复核策略选择；smoke test 增加 aligned frontier、unaligned mature fallback 和 partial-wave 三类回归测试。

代码落点：

| 文件 | 改动 |
|---|---|
| `include/fuse/kernels.h` | 增加 N320 policy 和 cluster/frontier 诊断字段 |
| `csrc/cutlass_kernels_sm90.cu` | 实例化 N320 collective；按 cluster 计算 wave；实现 frontier-aware 自动选择 |
| `benchmarks/fuse_smoke.cu` | 固化对齐、回退和 partial-wave 的策略回归测试 |
| `benchmarks/oproj_shape_bench.py` | 将 cluster/frontier 字段写入正式 JSON/CSV |
| `BENCHMARK.md` | 更新 36-case 正式结果和 TE Userbuffers 对照 |

### 收益

第一阶段完成了 36-case 复测；下列数据保留该阶段相对 v1 的独立结论。正式结果目录现已更新为第二阶段的最新主线结果。

| 指标 | 结果 |
|---|---:|
| exact correctness | 36/36 PASS |
| 优于 v1 Golden | 24/36 |
| 相对 v1 Golden 几何平均 | 1.0113× |
| 最大单点提升 | 1.1173× |
| 对 TE Userbuffers 胜场 | 22/36 |
| 相对 TE Userbuffers 几何平均 | 1.0304× |

目标组 `N=K=5120`：

| CP | 全局 S | v1 p50 | v2 p50 | 提升 | v2 policy |
|---:|---:|---:|---:|---:|---|
| 4 | 16K | 0.364 ms | 0.354 ms | 3.02% | `M128N320 C2` |
| 4 | 128K | 2.722 ms | 2.608 ms | 4.35% | `M128N320 C2` |
| 4 | 256K | 5.435 ms | 5.199 ms | 4.54% | `M128N320 C2` |
| 4 | 512K | 10.813 ms | 10.530 ms | 2.69% | `M128N320 C2` |
| 8 | 16K | 0.282 ms | 0.252 ms | 11.73% | `M128N320 C2` |
| 8 | 128K | 1.834 ms | 1.776 ms | 3.28% | `M128N320 C2` |
| 8 | 256K | 3.568 ms | 3.508 ms | 1.72% | `M128N320 C2` |
| 8 | 512K | 7.107 ms | 6.964 ms | 2.06% | `M128N320 C2` |

CP8 中宽度长序列仍慢于 TE Userbuffers，但差距缩小：128K、256K、512K 的吞吐分别从 TE UB 的 86.97%、84.44%、84.48% 提升到 89.82%、85.89%、86.23%。

### 第二阶段：窄 peer shard 的 3D TMA store

原通信热路径先把远端数据搬进每个通信 warp 的 SMEM stage，再对每一行单独发一条 S2G TMA。peer shard 较窄时，单条 TMA 的 payload 太小。

主线现在按真实 shard 字节数选择写回方式：

```text
peer_row_bytes = K / CP * sizeof(BF16)

peer_row_bytes <= 1024 B:
    将多行合并为约 8 KiB 的 SM90 3D TMA store
otherwise:
    保持原逐行 S2G TMA
```

该条件只依赖数据布局，没有 CP、模型名称或 benchmark shape 的硬编码。3D TMA 只改变 SMEM 到最终 GEMM A layout 的写回粒度，G2S、peer 发布顺序、ready epoch 和 GEMM 消费协议均不变。

正式 36-case 复测中，当前模型矩阵只有 `K=4096, CP8` 的六个 setting 命中新路径：

| 全局 S | 第一阶段 p50 | 当前 p50 | 提升 |
|---:|---:|---:|---:|
| 1K | 0.032096 ms | 0.031840 ms | 0.80% |
| 4K | 0.063024 ms | 0.062000 ms | 1.65% |
| 16K | 0.195904 ms | 0.191728 ms | 2.18% |
| 128K | 1.365728 ms | 1.309536 ms | 4.29% |
| 256K | 2.687024 ms | 2.597632 ms | 3.44% |
| 512K | 5.354336 ms | 5.086144 ms | 5.27% |

- 新路径 6/6 提升，没有命中后的退化点。
- 其余 30 个 setting 继续执行原写回路径；同一时段交叉 A/B 的差异为 -0.08% 到 +0.23%。
- 额外 `K=2048` 泛化检查中，CP4/CP8 长序列分别提升约 4.5%/27.7%。
- 完整编译、quick smoke、代表性 10+50 性能回归均通过，`exact_mismatches=0`。

最新 36-case 汇总保存在 [`results/oproj_cluster_wave_bench`](results/oproj_cluster_wave_bench)。目录只保留 `fused_summary.json/csv`；失败的 M256 policy、手工 probe、逐 case JSON 和临时 A/B 文件已经删除。

### 发布版自动通信 CTA

Golden 入口仍为 `--comm-ctas 0 --lhs-policy auto`。默认规则只使用可审计的 shape 和硬件量：

```text
M < 32768: comm4
M >= 32768:
  pressure = (CP - 1) / (CP * N) * 900 / NVLink_GBps
  pressure >= 1.65e-4: comm8
  otherwise:             comm6
```

H100/H200 默认按 900 GB/s 双向 NVLink，H800 按 400 GB/s；其他拓扑可用 `FUSE_NVLINK_BIDIR_GBPS` 显式覆盖。该规则来自 CP4 长序列 sweep，9 个点命中 8 个 oracle comm，未命中点的延迟损失不超过 0.022%。

最终 36-case（10 warmup + 50 samples，max-rank p50）：

| 指标 | 结果 |
|---|---:|
| exact correctness | 36/36 PASS |
| 通信 CTA 分布 | `comm4/6/8 = 21/8/7` |
| 优于 TE+NCCL/cuBLASLt+NCCL 最强分离实现 | 36/36 |
| 相对最强分离实现 | 1.160×–2.958×，几何平均 1.662× |
| 优于 TE Userbuffers | 29/36 |
| 相对 TE Userbuffers 几何平均 | 1.110× |
| 相对上一版 comm4 Golden | CP4 1.080×，CP8 1.070×，全部 1.075× |

原本已经领先 TE Userbuffers 的 case 没有因自动 comm 规则变成落后。完整逐点数据和对手调优参数见 [`BENCHMARK.md`](BENCHMARK.md)。

### 边界

- `N=7168` 在当前候选集中没有与 64 compute cluster 整除的 N tile 数；策略会在成熟 `N256` 和 partial-wave `N320` 之间按完整评分选择。
- 尾 wave 是否满载仍由 wave efficiency 计入成本，不作为硬门禁；硬性优先级只用于避免 frontier 被 wave 边界切开。
- 小宽度长序列仍是主要短板，不能从本次改动推断 GEMM、通信或调度中的单一瓶颈。
- profiling 协议和字段定义见 [`PROFILE_PROTOCOL.md`](PROFILE_PROTOCOL.md)。
