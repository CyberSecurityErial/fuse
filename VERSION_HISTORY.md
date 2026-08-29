# 版本演进手册

本文记录 A2A + O-projection 与 QKV Projection + A2A 的版本改动、测量结果和已知边界。正式数据统一使用 BF16、10 次 warmup + 50 次采样，并先取每次采样的跨 rank 最大延迟，再统计 p50。

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

当前版本：v2.0

| 文件 | 改动 |
|---|---|
| `include/fuse/operators/a2a_gemm.h` | 增加 N320 policy 和 cluster/frontier 诊断字段 |
| `csrc/operators/ulysses_sm90.cu` | 实例化 N320 collective；按 cluster 计算 wave；实现 frontier-aware 自动选择 |
| `benchmarks/a2a+Oproj/fuse_smoke.cu` | 固化对齐、回退和 partial-wave 的策略回归测试 |
| `benchmarks/a2a+Oproj/oproj_shape_bench.py` | 将 cluster/frontier 字段写入正式 JSON/CSV |
| `BENCHMARK.md` | 更新 36-case 正式结果和 TE Userbuffers 对照 |

### 收益

第一阶段完成了 36-case 复测；下列数据保留该阶段相对 v1 的独立结论。正式结果目录现已更新为第二阶段的最新主线结果。

本节第一阶段和第二阶段的逐点数字都是开发过程快照，用于解释每步改动；v2.0 的最终发布结论只使用后文“发布版自动通信 CTA”中的正式36-case结果。

当前版本：v2.0

| 指标 | 结果 |
|---|---:|
| exact correctness | 36/36 PASS |
| 优于 v1 Golden | 24/36 |
| 相对 v1 Golden 几何平均 | 1.0113× |
| 最大单点提升 | 1.1173× |
| 对 TE Userbuffers 胜场 | 22/36 |
| 相对 TE Userbuffers 几何平均 | 1.0304× |

目标组 `N=K=5120`：

当前版本：v2.0

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

当前版本：v2.0

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

最新 36-case 汇总保存在 [`results/a2a-Oproj/oproj_cluster_wave_bench`](results/a2a-Oproj/oproj_cluster_wave_bench)。目录只保留 `fused_summary.json/csv`；失败的 M256 policy、手工 probe、逐 case JSON 和临时 A/B 文件已经删除。

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

当前版本：v2.0

| 指标 | 结果 |
|---|---:|
| exact correctness | 36/36 PASS |
| 通信 CTA 分布 | `comm4/6/8 = 21/8/7` |
| 优于 TE+NCCL/cuBLASLt+NCCL 最强分离实现 | 36/36 |
| 相对最强分离实现 | 1.160×–2.958×，几何平均 1.662× |
| 优于 TE Userbuffers | 29/36 |
| 相对 TE Userbuffers 几何平均 | 1.110× |
| 优于 v1.0 Golden | 25/36 |
| 相对 v1.0 Golden | CP4 1.086×，CP8 1.092×，全部 1.089×；最大提升 1.531× |

原本已经领先 TE Userbuffers 的 case 没有因自动 comm 规则变成落后。完整逐点数据和对手调优参数见 [`BENCHMARK.md`](BENCHMARK.md)。

### 边界

- `N=7168` 在当前候选集中没有与 64 compute cluster 整除的 N tile 数；策略会在成熟 `N256` 和 partial-wave `N320` 之间按完整评分选择。
- 尾 wave 是否满载仍由 wave efficiency 计入成本，不作为硬门禁；硬性优先级只用于避免 frontier 被 wave 边界切开。
- 长序列剩余短板集中在 CP8 128K：`16384×4096×4096` 和 `16384×5120×5120`；其余16个长序列点均领先 TE Userbuffers。不能仅凭总延迟把这两个点归因于 GEMM、通信或调度中的某一个环节。
- profiling 协议和字段定义见 [`PROFILE_PROTOCOL.md`](PROFILE_PROTOCOL.md)。

## v3.0：外部通信 CTA 标定

状态：已发布。

v3.0 不改 A2A + O-projection 的计算、通信或自动 tile 逻辑。它固定 v2.0 kernel，只把外部 `comm_ctas` 作为唯一调参量：`--lhs-policy auto --raster n --swizzle 1` 全程不变。29 个已经领先 TE Userbuffers 的 setting 原样沿用 v2 自动结果，7 个原落后点使用正式扫描得到的通信 CTA 数。

当前版本：v3.0

| CP | S | GEMM M×N×K | v2 自动 p50 | v3 p50 / comm | TE UB p50 | v2→v3 | v3 相对 TE |
|---:|---:|---|---:|---:|---:|---:|---:|
| 4 | 4K | `1024×4096×4096` | 0.0928 | 0.0736 / c10 | 0.0792 | 1.261× | 1.076× |
| 4 | 16K | `4096×4096×4096` | 0.2976 | 0.2280 / c16 | 0.2507 | 1.305× | 1.100× |
| 4 | 4K | `1024×5120×5120` | 0.1126 | 0.1036 / c13 | 0.1063 | 1.086× | 1.026× |
| 4 | 1K | `256×7168×16384` | 0.1345 | 0.1156 / c16 | 0.1202 | 1.164× | 1.040× |
| 8 | 4K | `512×7168×16384` | 0.2631 | 0.2365 / c20 | 0.2507 | 1.112× | 1.060× |
| 8 | 128K | `16384×4096×4096` | 1.3070 | 0.9361 / c12 | 1.1210 | 1.396× | 1.197× |
| 8 | 128K | `16384×5120×5120` | 1.7831 | 1.3930 / c6 | 1.5952 | 1.280× | 1.145× |

发布口径：

- 7/7 exact correctness 通过；与其余29个 v2 setting 合并后，36/36 个 p50 快于 TE Userbuffers。
- 相对 TE Userbuffers 的36点 p50 延迟比几何平均为1.154×；相对 v2.0 的36点 p50 几何平均为1.040×。
- `256×7168×16384, CP4` 的两次独立 p50 为0.1134/0.1156 ms，均快于 TE 的0.1202 ms；最新一轮 p95 为0.1258 ms，慢于 TE 的0.1217 ms。因此36/36只描述正式 p50，不扩展为尾延迟结论。
- `comm_ctas` 的 winner 属于 shape 与硬件拓扑标定值。默认 `--comm-ctas 0` 仍是通用生产入口；v3.0 没有把7个 shape 写进运行时规则。
- profiling 字段全部由 `FUSE_ENABLE_PROFILING` 编译期开关隔离。Release 默认关闭，关闭构建不携带 timeline 参数、时间戳读取或 diagnostic atomic。

完整36点的 v3.0 归档位于 [`results/a2a-Oproj/oproj_v3_manual_comm_bench`](results/a2a-Oproj/oproj_v3_manual_comm_bench)，每行通过 `result_source` 标明 `v2_auto_inherited` 或 `manual_comm_ctas`；上面的7点表只展示相对 v2.0 有变化的标定项。完整36点的 v2 原始结果继续保存在 `results/a2a-Oproj/oproj_cluster_wave_bench`，避免把手工标定冒充自动策略结果。

## v4.0：真实模型矩阵与强基线

状态：已发布。

v4.0 的重点是调参与 benchmark 固化，没有修改 QKVProj+A2A 的算子性能逻辑：

- A2A+OProj 从36个代表 setting 扩展到96个 setting：三个人工 Golden、五组真实 GQA 模型、六档全局序列长度、CP4/CP8；
- QKVProj+A2A 建立同样的96点正式矩阵，完成 TE、经典 cuBLAS、cuBLASLt、NCCL 和适配版 TE Userbuffers 的强基线搜索，作为 v5 优化前的固定起点。

OProj 的融合侧只允许 `comm_ctas` 变化：先扫 `{2,4,6,8,10,12,14,16,20,24}`，复查短程 winner 的相邻整数，再把 top-3 固定为10+50正式复测。三组代表点逐字段复用 v3 Golden；Llama-3-8B 与 Qwen2.5 的同几何标签复用同一次物理测量；生产 Qwen、Nanbeige4.2-3B 和 Llama-3.1-405B 重新执行完整标定。

当前版本：v4.0

| 项目 | 结果 |
|---|---:|
| OProj 正式 setting | 96 |
| OProj 相对 TE Userbuffers p50 胜场 | 95/96 |
| Nanbeige4.2-3B OProj 胜场 | 12/12 |
| OProj 唯一 p50 落后点 | Llama-3.1-405B，CP4，S=1K |
| QKV 正式 setting | 96 |
| 正式采样 | 10 warmup + 50 samples，max-rank p50/p95 |

分离基线对每个 setting 扫描48个独立 NCCL tuple，并继续搜索 Graph/eager、stream 优先级与 pack 参数；cuBLASLt 使用64 MiB workspace做本地 heuristic 调优。TE Userbuffers 对通信 SM、streams、push/pull、CE/SM、pack 参数与方向做结构搜索，top-3 统一正式复测。不同 NCCL tuple 使用独立进程组，避免环境参数被第一个 communicator 缓存。

两条算子的完整口径、逐点数据和复现命令分别见 [`benchmarks/a2a+Oproj/BENCHMARK.md`](benchmarks/a2a+Oproj/BENCHMARK.md) 与 [`benchmarks/QKVproj+a2a/BENCHMARK.md`](benchmarks/QKVproj+a2a/BENCHMARK.md)。v5 最终先收口正式启动与计时口径，QKVProj+A2A 的专项性能优化后移到 v6。

## v5.0：Eager/Graph 双口径与可复现调参

状态：已发布。

v5.0 是 benchmark release。A2A+OProj 与 QKVProj+A2A 的生产热路径沿用
v4.0，本版本集中修正和固化以下内容：

- 两条算子分别给出 Eager 与 CUDA Graph 的10+50正式结果，两列独立报告；
- Graph capture、instantiate 与显式 upload 都在采样外，结果记录 epoch 模式与启动配置；
- QKV 正式 runner 使用 MPI 一进程一卡和 CUDA IPC，逐样本对各 rank 的 CUDA event 做 `MPI_MAX`；单进程串行多卡 runner 只保留作诊断；
- QKV Graph 由一个预上传 graph 承载10个 warmup和50个正式 monotonic epoch kernel node，只 replay 一次；
- 两条算子的96点配置 manifest 固化 `comm_ctas`、policy、raster 和 swizzle，TE/cuBLASLt+NCCL 与 TE Userbuffers 的 winner 参数同步归档；
- TE Userbuffers 的计时边界完成静态审计：event 覆盖完整 pack/GEMM/通信/unpack 边界，所有通信 stream 在 stop 前回到主 stream，跨 rank 使用最大值；
- `PROFILE_PROTOCOL.md` 升级为通算融合统一协议，补齐 GEMM→A2A finalize、正式启动和 Graph 计时规则。

当前版本：v5.0

| 项目 | Eager | CUDA Graph |
|---|---:|---:|
| A2A+OProj 正式 setting | 96 | 96 |
| A2A+OProj 对最强外部基线胜场 | 95/96 | 94/96 |
| A2A+OProj CP8 胜场 | 48/48 | 48/48 |
| QKVProj+A2A 正式 setting | 96 | 96 |
| QKVProj+A2A 对 TE Userbuffers 胜场 | 69/96 | 72/96 |
| QKVProj+A2A 对最强外部基线胜场 | 63/96 | 68/96 |

所有正式结果使用 BF16、10 warmup + 50 samples，并先逐样本取跨 rank 最大延迟，
再统计 p50/p95。OProj 两套结果的 exact mismatch 均为0；QKV 全部 case 通过
epoch lifecycle 检查，并按每种模型 geometry 额外完成 fused 与 separated reference
对照。v6 从这份不再漂移的基线开始 QKVProj+A2A 专项性能优化。

## v6.0：QKV wave-time tile选择

状态：已发布。

v6.0第一次修改QKVProj+A2A热路径。实现保留`M128N128`、`M128N160`、
`M128N256 cluster-M2`和`M128N320 cluster-M2`四种BF16 CUTLASS policy，运行时
使用下面的可审计评分选择tile：

```text
score(policy) = ceil(work_units / persistent_workers)
              × calibrated_one_wave_time(K, policy, comm_ctas)
```

整波时间来自H200、132 SM上的纯CUTLASS compute-subgrid：分别标定
`comm_ctas={24,32}`、`K={2048,3072,4096,5120,16384}`和四种policy，共40行。
运行时M/N只计算work unit和wave数量，TE Userbuffers结果与96点逐case winner不进入
模型。K在相邻标定点间线性插值；设备SM数、comm、K或peer-interleaved超出标定域时
回退v5策略。原始标定表保存在
[`benchmarks/QKVproj+a2a/qkv_wave_calibration.csv`](benchmarks/QKVproj+a2a/qkv_wave_calibration.csv)。

默认路径无需环境变量；`FUSE_QKV_GEMM_POLICY=legacy`可复现v5，
`m128n128/m128n160/m128n256/m128n320`用于固定policy消融，
`wave_time_model`显式选择v6模型。

当前版本：v6.0

| QKVProj+A2A，96个setting | v5 Eager | v6 Eager | v5 Graph | v6 Graph |
|---|---:|---:|---:|---:|
| 对TE Userbuffers胜场 | 69/96 | 73/96 | 72/96 | 74/96 |
| 对最强外部基线胜场 | 63/96 | 64/96 | 68/96 | 69/96 |
| 相对v5 p50几何平均 | 1.0000× | 1.0155× | 1.0000× | 1.0164× |
| 23个换tile setting的几何平均 | 1.0000× | 1.0649× | 1.0000× | 1.0688× |

96点继续使用MPI一进程一卡、10 warmup + 50 samples、逐样本max-rank p50/p95，
Eager与Graph分列。73个没有换tile的setting，Eager p50几何平均变化只有+0.045%，
可作为本轮采样稳定性对照。已知取舍是CP4人工中型S=4K：Eager由111.7 μs
变为128.2 μs，慢于TE Userbuffers的117.4 μs；v6保留这一结果，没有加入逐shape
特判。完整数据和复现命令见
[`benchmarks/QKVproj+a2a/BENCHMARK.md`](benchmarks/QKVproj+a2a/BENCHMARK.md)。

## v7.0：QKV联合流水与CTA独立前进

状态：已发布。

v7把通信CTA和GEMM tile放进同一个启动前模型。候选通信CTA为`4..32`的偶数，
候选tile为`M128N128`、`M128N160`、`M128N192`、
`M128N256 cluster-M2`和`M128N320 cluster-M2`。评分只读取M/N/K、CP、
head geometry、SM数、远端字节数、tile/cluster几何和H200原语标定：

```text
compute = persistent_waves × calibrated_wave_time(tile, K)
route   = max(16KiB_task_waves, remote_bytes / one_way_NVLink_rate)
score   = compute + max(one_task_wave, route - later_compute_waves)
```

N192是本轮新增的cluster1 policy。它与cluster-M2需要相同wave数且原始GEMM速度接近时，
可以保留两个CTA独立前进，减少通信竞争下的cluster级互相等待。运行时模型不读取模型名、
TE Userbuffers结果、逐case winner，也没有96点shape查表。超出当前设备、K、head dim或
布局标定域时回退成熟路径。

QKV finalize由thread0串行发布、串行等待8个source epoch，改为第一warp的8个lane
并行发布和并行等待；system fence、epoch语义与退出条件不变。benchmark默认统一传入
`comm_ctas=0/raster=n/swizzle=1`，由同一个模型解析实际通信CTA和tile；历史逐case
manifest只保留作v5/v6复现。

当前融合版本：v7.0

| QKVProj+A2A，96个setting | v6 Eager | v7 Eager | v6 Graph | v7 Graph |
|---|---:|---:|---:|---:|
| 对TE Userbuffers胜场 | 73/96 | 96/96 | 74/96 | 96/96 |
| 对TE Userbuffers几何平均 | — | 1.268× | — | 1.319× |
| 对最强外部基线胜场 | 64/96 | 93/96 | 69/96 | 96/96 |
| 对最强外部基线几何平均 | — | 1.201× | — | 1.250× |
| 融合吞吐 / 经典cuBLAS吞吐，中位数 | — | 90.8% | — | 92.0% |
| 相对v6 p50几何平均 | 1.000× | 1.147× | 1.000× | 1.156× |

正式口径继续使用MPI一进程一卡、10次warmup + 50次采样，并逐样本取跨rank最大值。
Eager相对v6有3个setting回退0.3%～1.9%，三点仍分别比TE Userbuffers快26.0%～
35.3%；Graph全96点也全部超过最强外部基线。完整逐点数据与复现命令见
[`benchmarks/QKVproj+a2a/BENCHMARK.md`](benchmarks/QKVproj+a2a/BENCHMARK.md)。

## v8.0：QKV N64 与稳定自动启动

状态：已发布。

### N64进入统一选择

v8新增`M128N64 cluster-M1`。它与`M128N128`、`M128N160`、`M128N192`、
`M128N256 cluster-M2`和`M128N320 cluster-M2`使用同一套启动前计算：先根据
真实M/N/K算出每种tile需要多少个计算wave，再结合独立测得的整波时间和通信时间，
选择预计总时间最短的组合。

N64没有模型名、benchmark shape或`comm_ctas`阈值特判；尤其不存在“通信CTA至少
为16才能使用N64”的条件。H200整波测量保存在
[`qkv_wave_calibration.csv`](benchmarks/QKVproj+a2a/qkv_wave_calibration.csv)，
TE Userbuffers结果和96点逐case winner不参与选择。

### 修复H800 Eager每层重复选择

旧实现收到`comm_ctas=0`时，每次Eager调用都会重新读取GPU信息，并重新遍历通信CTA
和GEMM tile组合。H800上读取设备信息可能等待前面的GPU工作，所以训练中的每一层都
可能重复付出这段时间；显式写死通信CTA时不会走这条路径。CUDA Graph只在capture时
执行一次主机侧选择，因此这个问题在Graph replay里不明显。

v8在第一次调用时生成完整启动配置，记住实际通信CTA、tile、SM数和设备。之后遇到
相同设备、shape、route和环境配置时直接复用，不再逐层读取设备属性或重新搜索。
NVLink带宽信息也按设备读取一次。数据指针、epoch和alpha仍使用每次调用的新值，
不会被缓存。
上层无论是直接把`comm_ctas=0`交给launcher，还是先调用公开的CTA推荐接口，
都会复用同一类已解析结果，不会从另一个入口重复付这段开销。

H800原问题已经用同一训练case稳定复现；当前机器没有H800，因此修复后的H800
训练step仍需在对应环境复验。H200上使用同一shape连续三次测试，自动入口与等价
显式配置的p50中位数分别为43.440 μs和43.248 μs，只差0.44%，说明后续调用已走
复用路径。

| 启动方式 | 修复前 | 修复后 | 显式通信CTA对照 |
|---|---:|---:|---:|
| Eager训练step | 855～1250 ms | 待H800环境复验 | 约662.6 ms；legacy auto约668.3 ms |

### 数据写完以后再发布ready

TMA store原有等待只保证异步引擎已经读完源共享内存，不能保证目标显存已经写完。
旧代码随后立即发布ready，消费者理论上可能先看到“数据可读”，再遇到尚未完成的
目标写入。v8改为完整等待目标显存中的整块数据写完，再发布ready。该规则同时覆盖
QKV GEMM输出和A2A+OProj输入；epoch与system-scope内存顺序保持不变。

为确认这项修复没有拖慢A2A+OProj，v8又按原口径重跑了96个setting的Eager和
CUDA Graph：所有结果均为零误差，几何平均延迟相对旧结果分别变化`+0.11%`和
`-0.06%`，没有系统性退化。唯一绝对时间波动超过5%的长序列点，同一次运行中的
纯GEMM也慢了约8%，因此判断为测试环境波动，而不是融合路径效率下降。

### smoke按真实tile分配ready缓冲区

旧smoke按默认tile大小计算ready数组。自动策略选择N64后，N方向tile数量会增加，
继续按N128分配会使测试缓冲区偏小。v8先用真实route、通信CTA、SM数和自动选择结果
取得实际BM/BN，再据此分配ready数组。这是测试脚手架的正确性修复，不改变生产接口。

### 正式结果

当前融合版本：v8.0

| QKVProj+A2A，96个setting | v7 Eager | v8 Eager | v7 Graph | v8 Graph |
|---|---:|---:|---:|---:|
| 对TE Userbuffers胜场 | 96/96 | 96/96 | 96/96 | 96/96 |
| 对最强外部基线胜场 | 93/96 | 96/96 | 96/96 | 96/96 |
| 相对v7 p50几何平均 | 1.000× | 1.0116× | 1.000× | 1.0135× |
| 融合吞吐 / 经典cuBLAS吞吐，中位数 | 90.8% | 91.0% | 92.0% | 92.2% |
| N64实际命中 | — | 9/96 | — | 9/96 |

H200正式结果包含192份raw、96个唯一setting；Eager和Graph均为96/96领先
TE Userbuffers。N64命中9点，6点提升、3点回退，子集相对v7几何平均提升
1.1368×/1.1503×；3个回退点仍全部领先TE Userbuffers。8卡完整smoke覆盖
CP2/4/8、BF16/FP8、完整QKV/defer-V和padding，均通过。H200的性能数字与H800的
Eager启动问题分开报告，不把H200吞吐数字直接当作H800性能承诺。

## v9.0：锁频异构 CP

状态：已发布。

v9.0 新增两个独立的 BF16 算子：weighted QKV Projection+A2A 和 weighted A2A+OProj。原有均匀 QKV/OProj 的接口、tile策略和热路径不变。

调用方为每个 CP rank 提供相对 SM、HBM 和 NVLink 能力。规划器不读取频率、设备名、模型名、外部基线或逐 case winner；它在冷路径枚举256-row对齐的连续序列分区，并结合现有通信CTA和GEMM tile的物理模型，使预测最慢rank的完成时间最短。正式自动入口不需要alpha；结果中的`equivalent_alpha`仅用于解释最终分区距离均分和纯SM比例端点有多远。

每个rank仍只拥有一个连续token区间并启动一个persistent kernel，不做跨GPU动态偷任务。QKV根据`global_sequence_begin`把本地生成的Q/K/V直接写到全局正确位置；OProj直接从所有peer读取本rank新token区间对应的head shard。全局数学结果不变，但local sequence length可不同，因此框架必须让相同分区贯穿依赖该序列布局的后续计算。

长时间运行的QKV会让原本较快的卡撞到功耗墙，固定频率比例不再能代表真实算力。因此默认只在全局`S≤16K`时允许QKV重分；更长QKV直接使用原均匀算子。调用方只有在已经测得稳定的长QKV有效能力时，才可显式开启实验覆盖。OProj长序列在本轮仍稳定受益，所以不使用该长度限制。任何候选只要模型没有预测到严格收益，就不启动weighted kernel而直接回退uniform。

正式矩阵使用三张1500 MHz卡和1980 MHz参考卡，HBM均为3201 MHz；覆盖CP2/4/6/8、不同慢卡数量、本地M=2048/16384，使用BF16、5次warmup + 30次采样，并逐样本先取max-rank再统计p50。短QKV实际启用的7点全部提升`1.0619×～1.0901×`，长QKV全部回退`1.0000×`；OProj启用的13点全部提升`1.1593×～1.4216×`。其余点主动回退，所有启用点均通过逐元素完全一致检查。

SM锁频是本版本唯一完成性能验收的异构来源。HBM/NVLink比例已进入API和成本模型，但尚无对应降频硬件数据；自动DVFS、温度、功耗墙和动态争用不属于v9.0保证范围。完整口径见[`benchmarks/heterogeneous_cp/BENCHMARK.md`](benchmarks/heterogeneous_cp/BENCHMARK.md)。

## v9.1：异构 shape 功耗安全回退

状态：已发布。

v9.1 保留 v9.0 的两个 weighted 算子、连续 token 分区和冷路径规划模型，不改原有均匀 QKV/OProj 热路径。补充真实模型宽度后发现：当单 rank GEMM 足够重时，原本 1980 MHz 的参考卡也会撞到约 700 W 功耗墙并降至约 1500～1650 MHz。此时 1500/1980 的标称频率比不再代表实际吞吐，继续按它搬工作反而可能变慢。

规划器现在用 `2MNKL / 989 TFLOPS` 计算 H200 BF16 纯 GEMM 的理论最低时间。默认只在该值不超过 `0.75 ms` 时使用标称锁频比例；超过后 QKV 和 OProj 都直接复用 uniform 路径。这个边界不读取模型名或逐 case winner。若调用方提供的是同一 workload 下测得的有效吞吐比，而非标称频率比，可显式设置 `allow_power_limited_redistribution` 放开。

补充矩阵覆盖 production Qwen、Llama-3 8B、Nanbeige 4.2-3B、Qwen2.5 14B/32B、Llama-3.1 405B 和原 v9 控制 shape，共 18 个 setting、36 个算子点，包含 CP4/CP8 与本地 `M=2048/16384`。19 个实际启用 weighted 的点全部提升并通过 BF16 逐元素完全一致检查，另外 17 个点安全回退；QKV 和 OProj 启用点的加速比几何平均分别为 `1.1101×` 和 `1.3915×`，没有退化点。

这组验证用于说明所选模型宽度内的安全泛化，不是所有 MNK、功耗上限或动态 DVFS 状态的全量保证。SM 锁频仍是唯一完成正式性能验收的异构来源；HBM/NVLink 降频接口继续保留为实验能力。完整口径和逐点结果见 [`benchmarks/heterogeneous_cp/BENCHMARK.md`](benchmarks/heterogeneous_cp/BENCHMARK.md)。

## v10.0：QKV 与 OProj 反向融合

状态：已发布。

v10 新增两条 BF16 反向算子，不改变 v9.1 的四条前向/weighted 接口。

QKV 反向 B 先把各 source rank 的平面 `dQ/dK/dV` 直接写成 destination rank 的
`[M,QKV]`，再计算 `dX[M,H]`；W 计算
`dWqkv[QKV,H]=dQKVᵀ[QKV,M]×X[M,H]`。OProj 反向 B 先计算
`dA[M,A]=dY[M,H]×stored_Wo[H,A]`，再把最终 head shard 直接写到各 peer；W
计算 `dWo[H,A]=dYᵀ[H,M]×saved_A[M,A]`。两条 B 路径的通信与 GEMM 顺序分别
是前向的自然逆序，接口不要求框架插入 `cat/index_select/permute/contiguous`。

普通入口在同一 stream 内严格执行 B→W，使用 `beta=0` 写权重梯度。ZeroBubble
入口把 B 和 W 拆开，W 用 `beta=1` 直接累加 BF16 `main_grad`；B 与 W 之间，
调用方必须保留 W 真正读取的两个 buffer，并为每个未完成区间提供独立 slot。
如果框架能延长原 buffer 寿命，就不需要为了接口再复制一份；不能保留时，stash
是调度延迟本身带来的存储成本。多个 W 不能并发写同一个梯度地址。

反向自动策略只读取 M/N/K、CP、head geometry、route task、SM 数和合法 tile，
不读取模型名、前向/TE 结果或逐 case winner。QKV 与 OProj 使用各自的参数和计划
缓存；共享 CUTLASS ready adapter 新增的 system-scope 路径是默认关闭的编译期分支，
只由 QKV 反向的跨 GPU producer 显式启用。v9 的 43 个旧 device-kernel 指令体在
v10 Release 构建中保持一致。

旧前向还用 CP8 Nanbeige、全局 S=16K 做了两轮同机 10+50 A/B：QKV 的 v9/v10
p50 中位为 `0.2402/0.2408 ms`，OProj 为 `0.2643/0.2657 ms`，变化分别为
`+0.25%/+0.53%`，落在短程复测波动内；两边逐元素检查均为零误差。

正式矩阵对每条算子覆盖 3 个人工 shape、5 组真实模型、6 档全局序列长度和
CP4/CP8，共 96 个 setting。Eager/Graph 与普通/ZeroBubble 分别做 10 warmup +
50 samples，逐样本先取 max-rank；两条算子合计 768 份融合 raw。另用同一卡组、
真实 B/W MNK 和匹配 `beta` 完成 192 份经典 cuBLAS 纯 GEMM测量。

| 算子 | 调度 | Eager 相对前向几何平均 | Graph 相对前向几何平均 | Eager / Graph 经典cuBLAS中位 |
|---|---|---:|---:|---:|
| QKV backward | ZeroBubble，`beta=1` | 100.5% | 98.0% | 91.0% / 90.4% |
| QKV backward | 普通 B→W，`beta=0` | 104.3% | 102.3% | 88.8% / 89.6% |
| OProj backward | ZeroBubble，`beta=1` | 99.6% | 100.1% | 89.7% / 90.3% |
| OProj backward | 普通 B→W，`beta=0` | 104.0% | 104.6% | 89.5% / 90.1% |

这里的“相对前向”是 B+W 总 FLOPs 吞吐除以同 shape 已发布前向融合吞吐；四种
几何平均均高于 98%。小矩阵 exact smoke 覆盖 CP4/CP8、rank-major/causal、
batch=2、普通/分离和 `beta=1`；正式 S=1K 还检查跨 rank route、epoch 和完整写入。
完整 96 行 ZeroBubble/普通表、TFLOPS、989T MFU、经典 cuBLAS 对照与复现命令见
[`QKV backward`](benchmarks/QKVproj-backward/BENCHMARK.md) 和
[`OProj backward`](benchmarks/Oproj-backward/BENCHMARK.md)。
