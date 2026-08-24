# GQA + Ulysses CP 双向通算融合 Kernel 任务书

状态：两个 monolithic kernel family 已实现并通过完整 correctness/sanitizer；
状态：已完成并归档。生产实现、CP2/4/8 正确性与 sanitizer、10+50 性能矩阵、
TE/cuBLASLt/NCCL/Flux 对照及完整优化日志均已收口；双 stream 仍只作为 baseline，
未用于替代最终 monolithic 架构。

目标平台：单机 8×H200、NVLink、SM90（132 SM/GPU）

本文先冻结问题定义、算子边界、调度原则、验证口径和交付物。审核通过前不进入正式实现和调优。

实现附加原则：

- 核心代码使用纯 CUDA/C++；允许把本地 CUTLASS/CuTe 作为 GEMM building block，但不依赖 PyTorch extension 承担核心调度；
- 性能优先、开发速度优先、代码简洁，shape/dtype/layout 检查集中在 host launch 边界，hot path 不放重复或无效门禁；
- 不实现 autograd，也不处理 forward/backward 的高层求导图语义；只实现两类算子融合本身需要的 GEMM transpose/stride/route 变体；
- 复用本机 CUDA 12.8、cuBLAS、NCCL、Transformer Engine 和已有 CUTLASS 源码，不新建独立环境，除非现有依赖被证明确实无法使用；
- 正式测量前枚举并终止占用目标 GPU 的其他计算进程，测试进程使用可获得的最高 CUDA stream priority；
- 每次结构性改动、参数扫描、正确性结果和性能结果都写入 `OPTIMIZATION_LOG.md`，包括失败方案，不只保留最终数字。

当前实现快照（2026-08-17）：

- `A2A→GEMM` 已是单个 cooperative monolithic grid；BF16 projection 使用
  CUTLASS `128x256x64` Cooperative、stage 4、cluster `(2,1,1)`，FP8 QKV
  使用 `128x128x128` Cooperative FastAccum；
- `GEMM→A2A` 已是单个 cooperative monolithic grid；BF16 strided-batched
  PV 使用 `128x128x64` Pingpong，FP8 dense QKV 输出使用 FastAccum；
- ready 协议已细化到 `[L,M-tile,peer/K-group]`，通信生产顺序使用自适应
  M window；输出方向在 stock CUTLASS TMA epilogue 完成后只发布 epoch；
- projection cluster2 使用 cluster-aware extended launch，并在 host 上证明
  communication/compute prefix 对齐及 active-cluster residency；不存在双-stream
  fallback 冒充 fused 结果；
- 完整 CP2/4/8 correctness、连续 8 epoch、full memcheck 和 full racecheck
  已通过；正式性能与 TE、cuBLASLt/cuBLAS+NCCL、Flux 的结果见 README 和
  `OPTIMIZATION_LOG.md`。

## 1. 背景与目标

Ulysses Context Parallel（CP）会在 sequence-sharded 与 head-sharded 布局之间执行 All-to-All（A2A）。目标不是单纯缩短一次 GEMM，也不是把多个现成 kernel 放进同一条 stream，而是开发两个对偶的 CTA-specialized persistent CUDA kernel，让通信 CTA 与 GEMM CTA 以 tile 为粒度持续生产/消费，隐藏 A2A，并直接生成消费者需要的最终布局。

本任务需要完成两个 kernel family：

1. `A2A → GEMM`：通信 tile 是生产者，GEMM tile 是消费者。
   - forward 的主要实例是 A2A 后执行 QKV projection；
   - backward/recompute 中同一调度骨架还必须能承载 `P·V=O` 一类 strided-batched GEMM，不能把它写死成 projection。
2. `GEMM → A2A`：GEMM tile 是生产者，通信 tile 是消费者。
   - forward 的主要实例是 attention 的 `P·V=O`，随后执行 Ulysses A2A；
   - backward 通过转置、batch stride 和反向 route descriptor 复用同一骨架。

核心优化对象是生产者和消费者之间的 tile 管道，包括：tile 首次可用时间、channel 排布、信号开销、通信 CTA 占用的 SM 数量，以及 GEMM 峰值损失与 NVLink 饱和之间的甜点区。GEMM 主循环优先复用 CUTLASS/CuTe 等成熟实现。

## 2. 计算流与本任务位置

当前训练计算流按以下链路理解：

```text
forward:   A2A → QKV projection → QK → PV → A2A
recompute: A2A → QKV projection → QK → PV → A2A
backward:  A2A → dKV → dQK → dQKV projection → A2A
```

开启 zero-bubble 后，backward 还会穿插权重梯度落盘：

```text
A2A → dKV → store(wgrad_kv)
    → dQK → store(wgrad_qk)
    → dQKV projection → store(wgrad_proj)
    → A2A
```

第一阶段只实现和验证两类 `A2A↔GEMM` primitive。QK、softmax、完整 attention 融合、autograd，以及 zero-bubble 的三处 WGRAD store 不进入第一阶段 hot path；但参数和调度接口必须允许后续接入，不能因 forward 的一个固定 shape 阻塞 backward 的 GEMM/route 变体。

## 3. 通用代数接口

### 3.1 GEMM 统一表示

两个 kernel 都用逻辑问题 `(M, N, K, L)` 表示 GEMM：

```text
D[m, n, l] = sum_k A[m, k, l] * B[k, n, l]
```

- `L=1`：普通 dense projection，例如 QKV projection；
- `L=B×H_local`：strided-batched attention GEMM，例如每个 batch/head 上的 `P·V=O`；
- A/B/D 各自携带显式 stride；
- 通过 `transpose_a`、`transpose_b`、stride 和反向 route 支持 dgrad/wgrad/PV 的转置变体；
- kernel 模板可以按 dtype、tile shape 和布局特化，但运行时 API 不绑定某个开源模型参数。

典型 shape 仅用于说明，不构成硬编码：

```text
QKV projection:
  M = B * S_local
  K = hidden_size
  N = (Hq + 2 * Hkv) * D
  L = 1

PV = O:
  M = query_sequence
  K = key_value_sequence
  N = head_dim
  L = B * heads_local
```

两者的 `M/N/K/L`、访存方向和 batch 结构明显不同，必须进入不同的 CUTLASS policy/template 实例，但共享 persistent 调度、信号协议和 A2A route 层。

### 3.2 Ulysses/GQA 路由统一表示

记：

- `P`：CP world size，目标平台固定为 8，但接口不硬编码 8；
- `B`：batch size；
- `S`：global sequence length；
- `Hq`：query head 数；
- `Hkv`：key/value head 数；
- `D`：head dimension；
- `G=Hq/Hkv`：GQA ratio。

路由层描述 sequence-sharded 与 head-sharded 之间的 tile 映射，而不是散落在 GEMM 主循环里。至少支持：

- `SEQ_TO_HEAD`；
- `HEAD_TO_SEQ`；
- Q/K/V 三段不等长的 `QKV_GQA_PACK`；
- 上述 route 的 backward/inverse 形式；
- source rank、destination rank、channel、logical head、sequence range、batch index 到线性地址的显式映射；
- 尾块和非默认 `head_dim`，不假设 `D=128`；
- `Hq != Hkv`，不复制 K/V 来伪装 MHA。

候选 descriptor：

```cpp
struct GemmProblem {
  int64_t m, n, k, l;
  Stride4D stride_a, stride_b, stride_d;
  DType input_dtype, weight_dtype, output_dtype, accum_dtype;
  bool transpose_a, transpose_b;
};

struct UlyssesRoute {
  int32_t world_size, rank;
  int32_t batch, global_seq;
  int32_t q_heads, kv_heads, head_dim;
  RouteKind kind;
  Direction direction;
  int32_t channel_count;
};
```

最终字段可按 CUTLASS/CuTe 所需的 layout 类型调整，但 GEMM 问题和通信路由必须保持两个独立概念。

## 4. Kernel A：`A2A → GEMM`

### 4.1 CTA 职责

一个 persistent grid 内区分两类 CTA：

- 通信 CTA：负责从 peer/symmetric buffer 拉取或接收 route 对应的 tile，写入本地 staging/final-consumable layout，并发布 ready epoch；
- GEMM CTA：执行 CUTLASS/CuTe mainloop，只消费已经由通信 CTA 发布的 tile；
- 调度 CTA 数量由 `num_comm_ctas` 或 `num_comm_sms` 控制，剩余资源给 GEMM；
- 不让每个 GEMM warp 自行扫描所有 peer/channel，也不让 GEMM 一次搬运大 chunk 后才开始计算。

### 4.2 调度原则

`A2A → GEMM` 的困难在于 GEMM 是消费者。直接让所有 GEMM CTA 轮询远端 ready 会损失计算效率，而用大 chunk 降低轮询开销又会增加 pipeline bubble。因此采用“通信中心化发布、GEMM 按 ready 顺序消费”的方式：

1. 通信 CTA 按 channel 负责固定 peer/tile 子集；
2. 每完成一个可独立消费的 tile/stage，就用 system-scope release 发布 epoch；
3. GEMM persistent scheduler 优先分发已经 ready 的 tile，等待发生在 CTA/stage 边界，不进入 MMA 内循环；
4. tile 大小需要同时满足 NVLink transaction 效率、GEMM K-stage 利用率和较小首包延迟；
5. 对 projection 与 batched PV 使用不同 tile policy，但保留同一 ready-queue/epoch 协议。

### 4.3 主要实例

- forward/recompute：A2A 输入 tile → dense QKV projection；
- backward：A2A 输入 tile → strided-batched PV/PV-transpose 类 GEMM；
- 后续扩展：A2A 输入 tile → projection dgrad/wgrad。

## 5. Kernel B：`GEMM → A2A`

### 5.1 CTA 职责

- GEMM CTA：按固定 persistent tile 顺序生产 D tile；
- epilogue 只完成本地 D tile 的正常落盘，并写一个很小的 global-memory ready epoch；
- 通信 CTA：用 acquire 观察 ready epoch，把 tile 直接搬运到目标 rank 的最终 Ulysses 布局；
- 不在 GEMM epilogue 中发起远端 TMA/大块 P2P store，避免额外占用 epilogue shared memory，降低 GEMM occupancy；
- 不增加仅用于 pack/unpack 的完整中间张量。

### 5.2 GEMM 中心化消费排布

GEMM tile 的生产顺序可由 persistent scheduler 和 raster order 确定。通信调度以此为自变量：

1. 建立 `gemm_tile_id → route/channel/destination` 的确定映射；
2. 每个通信 CTA 负责若干 channel，优先排空已出现 ready tile 的 channel；
3. 避免某个 channel 的早期 tile 因静态轮询顺序长期滞留；
4. 当 `num_comm_ctas < world_size` 时，一个 CTA 轮转多个 peer；当更多时，一个 peer 可由多个 CTA 分担连续 tile；
5. 本 rank tile 与 remote tile 走同一逻辑 route，但本地 copy 可以单独走更低开销路径。

### 5.3 主要实例

- forward/recompute：strided-batched `P·V=O` tile → Ulysses A2A；
- backward：projection/PV 的反向 GEMM tile → inverse A2A；
- QKV 三段 route 作为通用 routing policy，而不是 kernel 唯一支持的输出格式。

## 6. 同步与内存语义

跨 CTA、跨 kernel role、跨 GPU 可见性的协议必须是显式的：

- 数据写完成后使用 `st.global.release.sys` 或等价 CUDA/CUTLASS system-scope release；
- 消费方使用 `ld.global.acquire.sys` 或等价 acquire；
- ready flag 使用单调递增 epoch，而不是每轮清零的布尔值，避免跨 iteration 的 ABA 和 reset kernel；
- epoch 溢出策略在 host 端管理，steady-state benchmark 不插入全量 memset；
- signal 存在 global memory，不占用 GEMM shared-memory stage；
- 每个 signal 对应 GEMM/通信真正可独立消费的最小 tile 集合，partial tile 的所有 producer 完成后才能 release；
- kernel launch 前执行一次 rank 间 epoch 对齐；异常退出必须有超时/诊断路径，避免永久 spin；
- CUDA IPC/VMM 或等价 symmetric allocation 用于建立 peer 可访问地址，启动时校验 8 卡 P2P access 和原子/内存可见性能力。

## 7. GEMM 实现策略

GEMM 本身不是本任务的主要创新点：

- dense projection 优先使用 CUTLASS 3.x Hopper warp-specialized/persistent GEMM；
- PV 优先使用 CUTLASS strided-batched/grouped GEMM collective；
- 通过自定义 epilogue visitor/callback 发布 tile epoch；
- pure GEMM 对照必须复用相同 tile policy，或明确使用 cuBLASLt 作为硬件上界，防止把 GEMM 实现差异误算成通信融合收益；
- 第一版不在 epilogue 中做远端直写或额外 TMA pipeline；
- shared memory 优先留给 MMA mainloop，通信 CTA 使用独立、较小的 staging；
- 若单个 monolithic grid 无法保证跨 role 的 residency/deadlock 安全，必须先给出资源证明或失败证据，不能静默退化成顺序执行的两个 kernel。任何双-stream fallback 都要单列为 baseline，不冒充 CTA-specialized 单-kernel 结果。

## 8. Dtype 范围

建议分两步：

1. BF16：先覆盖两个方向、dense 与 strided-batched 两种 GEMM，验证 route、同步和调度；
2. FP8：QKV projection 增加 E4M3×E4M3、FP32 accumulate、BF16 output，与 Transformer Engine FP8 baseline 对齐。

PV 的 softmax probability 与 V 的实际 dtype 需按目标训练配置决定；第一阶段默认 BF16 输入/输出、FP32 accumulate。FP8 PV 只有在存在等价数值契约和基准时再加入，不为追求 FLOPS 直接改语义。

## 9. Shape 泛化与约束

实现不得固定到单一开源模型。测试矩阵至少覆盖：

- `B ∈ {1, 2, 4}`；
- 多个 global sequence length，包含非 tile 整倍数；
- `Hq/Hkv ∈ {1, 2, 4, 8}` 的 MHA/GQA 组合；
- `head_dim ∈ {64, 80, 96, 128, 256}` 中硬件/算法允许的值；
- projection 的多组 hidden size 与 QKV width；
- PV 的 causal/non-causal 产生的有效 `M/K` 范围；
- CP=8 为性能主目标，CP=2/4 用于缩小正确性诊断；
- `Hq` 或 `Hkv` 不能整除 CP 的情况必须明确选择：支持 uneven split，或在 API 校验阶段给出可读错误，不能越界或静默复制。

允许为常见 tile shape 编译若干模板实例；运行时选择策略不能只识别某个模型的 `(M,N,K)` 三元组。

## 10. 正确性验证

### 10.1 参考实现

使用独立、易审计的 PyTorch/NCCL reference：

- A2A 使用 `all_to_all_single` 或等价 split API；
- dense GEMM 使用 `torch.matmul`/cuBLAS；
- PV 使用 `torch.bmm` 或显式 einsum；
- Q/K/V route 使用纯 reshape/permute/copy，不能调用待测 kernel 的地址映射函数；
- backward 分别验证 dP、dV、projection dgrad/wgrad 的代数结果与 inverse route。

### 10.2 布局验证

先用带 rank/head/sequence 编码的整数可精确表示张量做 exact layout test，确认每个输出元素来源；再做随机数值测试。至少验证：

- Q/K/V 三段在 GQA 下的 head offset；
- sequence shard 拼接顺序；
- batch/head batch-stride；
- local-rank 与 remote-rank route；
- tail tile；
- 连续多 iteration 的 epoch 正确性；
- 改变 `num_comm_ctas` 不改变结果。

### 10.3 数值标准

- BF16/FP32 accumulation：报告 max abs、max relative、relative L2，并使用与参考误差尺度匹配的 `atol/rtol`；
- FP8：量化 scale、fast accumulation、输出 dtype 必须与 TE/cuBLASLt reference 一致；
- 所有 rank 在 warmup 前和计时后各做一次正确性检查；
- 检查 NaN/Inf、越界（compute-sanitizer 小 shape）和 race（racecheck 能覆盖的路径）。

## 11. 性能基准与 FLOPS 口径

每个 operator 分别比较：

1. pure cuBLASLt/CUTLASS GEMM：相同 `(M,N,K,L)`、dtype、transpose 和 accumulation，代表无通信上界；
2. Transformer Engine + NCCL A2A：等价数学与布局语义的非融合基线；
3. GEMM + NCCL A2A 显式基线：用于拆分 TE module 开销；
4. CTA-specialized fused kernel；
5. 可选双-stream tile-overlap baseline：只用于判断单-kernel specialization 的额外价值。

吞吐只计算 GEMM 算术 FLOPS：

```text
FLOPS = 2 * M * N * K * L
effective TFLOPS/GPU = FLOPS / critical_path_time
aggregate TFLOPS = world_size * FLOPS / critical_path_time
```

同时报告：

- 每轮所有 rank 最大耗时的 mean/P50/P95/min/max；
- 相对 pure GEMM 的 TFLOPS retained、绝对 gap 和 latency slowdown；
- A2A algorithmic/remote payload bandwidth；
- 首个通信 tile ready、首个 GEMM tile start、最后一个 GEMM tile done、最后一个通信 tile done；
- `overlap_ratio = 1 - (fused_time - max(gemm_only, a2a_only)) / min(gemm_only, a2a_only)`，并注明这是调度指标而非硬件利用率；
- pack/unpack/staging bytes 是否被真正消除；
- persistent workspace 和额外显存。

TE 与 cuBLAS 对比必须分别对 dense projection 和 batched PV 给出，不能用 QKV 的 FLOPS/shape 代表 PV。

## 12. 调优顺序

调优按以下顺序进行：

1. 固定正确的 route 和同步协议；
2. 找 pure GEMM 的合理 CUTLASS tile/mainloop baseline；
3. 扫描通信 CTA/SM 数量，观察 GEMM 降速曲线与 NVLink 带宽曲线的交点；
4. 调整通信 tile 大小、每 CTA channel 数和 channel 轮询顺序；
5. 调整 GEMM raster order，使生产顺序匹配 route；
6. 调整 signal 粒度，权衡首包延迟与 acquire/release 开销；
7. 分别调 dense projection 与 batched PV，不共享未经验证的固定最优参数；
8. 最后才考虑 FP8 scale/epilogue、CUDA Graph 和 zero-bubble WGRAD store 扩展。

期望存在一个 sweet spot：给 GEMM 少分配若干 SM，只产生可接受的 FLOPS 下降；省出的 SM 由通信 CTA 使用并刚好打满 NVLink。调优输出应是按 shape class 选择的策略，而非单一模型 lookup。

## 13. 预期目录结构与交付物

```text
fuse/
├── TASK.md                         # 本任务书
├── OPTIMIZATION_LOG.md             # 按时间记录方案、测量、失败与结论
├── README.md                       # 构建、运行、支持矩阵和当前结果
├── CMakeLists.txt / pyproject.toml
├── include/fuse/
│   ├── problem.h                   # (M,N,K,L)、stride、dtype
│   ├── route.h                     # Ulysses/GQA route descriptor
│   └── semaphore.h                 # system-scope epoch 协议
├── csrc/
│   ├── a2a_gemm_sm90.cu            # Kernel A
│   ├── gemm_a2a_sm90.cu            # Kernel B
│   ├── route_sm90.cuh
│   ├── cutlass_epilogue_signal.cuh
│   └── torch_extension.cpp
├── python/fuse_attn_cp/
│   ├── ops.py
│   └── reference.py
├── tests/
│   ├── test_route.py
│   ├── test_forward.py
│   ├── test_backward.py
│   └── test_epoch_stress.py
├── benchmarks/
│   ├── bench_a2a_gemm.py
│   ├── bench_gemm_a2a.py
│   └── common.py
├── scripts/
│   ├── build.sh
│   ├── run_correctness_8gpu.sh
│   └── run_bench_8gpu.sh
└── results/
    └── *.json
```

最终必须交付源代码、可重复构建脚本、算子级 reference、dense/batched/transpose/route 测试、TE/cuBLAS 基准、结构化 JSON 结果、完整优化日志和设计/限制说明。只提供 benchmark wrapper 或直接调用现有 Flux operator 不算完成两个 kernel。

## 14. 阶段与验收门槛

### Phase 0：环境与融合必要性

- 冻结 8×H200 拓扑、P2P、NVLink、SM、CUDA/CUTLASS/TE/NCCL 版本；
- 分别测 pure GEMM、A2A、TE+NCCL，并量化未利用的算力/通信窗口；
- 若某个 shape 没有可重叠窗口，明确记录，不为了“融合”而融合。

### Phase 1：通用接口与 BF16 正确性

- 两个 kernel 都支持 dense 与 strided-batched 问题描述；
- forward route 与至少一个 backward inverse route 通过 CP=2/4/8；
- 连续 epoch stress 无错误或死锁。

### Phase 2：Hopper persistent pipeline

- 单个 primitive 使用一个 CTA-specialized persistent launch；
- 通信与 GEMM 存在可观测的 tile 级并行区间；
- GEMM epilogue 不执行远端大块直写，只发布 signal；
- compute-sanitizer 小 shape 通过。

### Phase 3：性能与调优

- 分别给出 QKV projection `A2A→GEMM` 与 PV `GEMM→A2A` 的结果；
- 结果与 TE+NCCL、显式 GEMM+NCCL、pure cuBLASLt/CUTLASS 比较；
- 报告通信 SM sweep 和 sweet spot；
- 性能结论以 8 卡 max-rank critical path 为准，不以单 rank 平均掩盖 straggler。

### Phase 4：backward shape/route 与 zero-bubble 扩展

- 复用两个 kernel family 覆盖 dKV/dQK/dQKV projection 所需的 transpose/stride/route；不负责 autograd 图或高层梯度调度；
- 评估三处 WGRAD store 是否能用独立 CTA/stream 隐藏；
- 不破坏 forward 的通用接口。

## 15. 非目标与禁止的捷径

- 第一阶段不实现完整 FlashAttention、softmax 或整个 Transformer block；
- 不针对某个开源模型固定 hidden size、head 数、sequence length 或 tile 表；
- 不把三个独立 Q/K/V NCCL collective 包装成 Python 类后称为 fused kernel；
- 不把顺序的 GEMM 和 A2A 放入 CUDA Graph 后称为 tile overlap；
- 不用不同 dtype、不同量化开销或不同输出布局制造不公平 FLOPS 对比；
- 不从 L20X/H200 产品名猜架构，正式运行按 compute capability 9.0、132 SM 和实际 P2P/NVLink capability 校验；
- 暂不支持多机、IB、PCIe-only 和非 Hopper 性能路径。

## 16. 审核时需要确认的理解

以下是当前按上下文作出的假设，请审核时直接修改或批注：

1. 你所说的 `PV=0` 按 `PV=O`（字母 O，attention output）理解。
2. 两个“kernel”指每个方向各一个 CTA-specialized persistent CUDA launch；双-stream 的 GEMM kernel + comm kernel 只能作为对照或 fallback，不能作为最终实现冒充单 kernel。
3. `A2A→GEMM` 在 forward 的主实例确实是 QKV projection，而不是常见 Ulysses 顺序中的 `QKV projection→A2A`；具体输入/输出 endpoint layout 将以 route descriptor 明确，不凭惯例反转。
4. `GEMM→A2A` 在 forward 的 GEMM 是 strided-batched `P·V=O`，不是 attention output projection dense GEMM。
5. backward 第一阶段要求证明通用接口和至少一个实际 inverse-route/PV 变体，不要求一次完成含 QK、softmax、全部 WGRAD store 的完整 attention backward。
6. 性能主目标是单机 8×H200 NVLink；CP=2/4 仅用于诊断和泛化测试。
7. QKV projection 的正式 TE 对比需要 FP8；PV 第一版按 BF16 语义实现，除非目标训练配置另有要求。
8. `/home/chen/workspace/source_code/kill_te` 的 TE FP8 QKV + NCCL Ulysses 结果可作为已有 baseline，但新基准会在 `fuse/` 中按两个 kernel 的各自 shape 重新组织，不能只复用一个 QKV 数字。
