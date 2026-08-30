# Ulysses GEMM + All-to-All Fusion

单机 Ulysses Context Parallel 的 GEMM/All-to-All 融合算子。A2A+O-projection 已完成优化；QKV Projection+A2A 在 v8.0 联合选择通信 CTA 与 GEMM tile，并复用已经算好的自动配置。v9.1 提供两个面向已知锁频差异的 BF16 加权序列算子。v10.0 新增 QKV 和 OProj 的两条 BF16 反向融合路径，同时支持普通同流 B→W 与 ZeroBubble 分离 B/W；v11.0 补齐 PyTorch autograd 正确性、TE 强基线和前后向 trace；v11.2 验证四条融合边界在完整 CUDA Graph 训练中的 E2E 收益。

Attention 输出按 head 分片：

```text
[B, H/CP, S, D]
        │ inverse All-to-All
        ▼
[B, S/CP, H, D]
        │ O-projection GEMM
        ▼
[B, S/CP, hidden]
```

对应 GEMM：

```text
M = B × S / CP
K = H × D
N = hidden
```

## 实现

- 单个 cooperative persistent kernel，通信 CTA 与 CUTLASS GEMM CTA 同时驻留。
- 通信 CTA 将远端 head shard 直接写入 GEMM 的最终 A layout，并按 peer/K shard 发布 ready epoch。
- 窄 peer shard 自动将多行写回合并为约8KiB的3D TMA store，宽 shard沿用逐行路径。
- GEMM CTA 使用 system-scope acquire 消费已到达的数据，省去独立 permutation kernel 和 kernel 间同步。
- `auto` 策略在五个成熟 policy 中选择：`M64N128`、`M128N128`、`M128N160`、`M128N256 cluster-M2`、`M128N320 cluster-M2`；wave 以 cluster 为调度单元，并优先保证 N frontier 不被 wave 边界切开。

QKV Projection+A2A 提供 `M128N64/N128/N160/N192/N256/N320` 六种 policy。H200 默认
用GEMM wave成本、QKV通信量、NVLink下界和cluster前进代价联合选择通信CTA与tile；
模型名称和逐case winner不参与选择；不支持的shape或布局自动回退成熟策略。
132-SM设备复用H200整波标定；H800的通信估算使用400 GB/s双向NVLink，
其他设备按900 GB/s。完整口径见
[`benchmarks/QKVproj+a2a/BENCHMARK.md`](benchmarks/QKVproj+a2a/BENCHMARK.md)。

`comm_ctas=0` 只在首次遇到某种设备和 shape 时计算通信 CTA 与 tile；后续 Eager
调用直接复用结果，但仍使用本次调用自己的地址和 epoch。这样避免 H800 训练中每层
重复读取设备信息和搜索配置。QKV 与 A2A+OProj 都会等目标显存写完后再发布 ready。

默认通信策略对短中序列使用 `comm4`，长序列根据 CP、`N` 和设备 NVLink 带宽选择 `comm6/8`。实验性全量成本模型可通过 `FUSE_A2A_LHS_COMM_POLICY=experimental_model` 启用；它默认关闭，尚不属于 Golden。模型口径和边界见 [`VERSION_HISTORY.md`](VERSION_HISTORY.md)。

### 锁频异构 CP（v9.1）

v9.0 新增独立的 weighted QKV Projection+A2A 与 weighted A2A+OProj；v9.1 保留这套接口和调度模型，并补上跨 shape 的功耗安全边界。调用方在启动前为每个 rank 提供相对 SM、HBM 和 NVLink 能力；规划器以 256-row 对齐的连续 token 区间为单位，联合选择每张卡的行数、通信 CTA 和既有 GEMM tile，使预测最慢 rank 的完成时间最短。它不读取实时频率、模型名、逐 case winner 或外部基线，也不需要调用方提供 alpha。

这不是跨 GPU 动态偷任务：每个 rank 仍只执行一个连续区间和一个 persistent kernel。全局数学结果不变，但每张卡的 local sequence length 可以不同，因此框架必须让同一分区贯穿依赖该序列布局的后续计算。QKV 默认只在全局 `S≤16K` 时允许重分；更长 QKV 回退原均匀算子。OProj 不使用固定的序列长度限制。

v9.1 的 shape 复核确认：把 MNK 按 H200 的 989 TFLOPS 理论峰值换算后，当单 rank 纯 GEMM 最低时间接近 1 ms 时，1980 MHz 参考卡本身也会撞 700 W 功耗墙并降到约 1500 MHz，此时标称频率比已经失效。v9.1 默认只在实测的 0.75 ms 功耗安全域内重分配，超出后 QKV 和 OProj 都会回退 uniform；若调用方提供同一负载下的有效吞吐比，可显式放开。完整约束、18 点补充结果和复现命令见 [`benchmarks/heterogeneous_cp/BENCHMARK.md`](benchmarks/heterogeneous_cp/BENCHMARK.md)。

### 反向融合（v10.0）

v10 把两个前向融合边界按反向依赖倒过来实现：

```text
QKV B:   Head→Sequence A2A(dQ,dK,dV) -> dX GEMM
QKV W:   dWqkv = dQKVᵀ × X

OProj B: dA GEMM -> Sequence→Head A2A
OProj W: dWo = dYᵀ × saved_A
```

通信直接读写最终布局，接口不要求框架先做 `cat`、`index_select`、`permute` 或
`contiguous`。普通模式在同一 stream 中严格执行 B→W，并用 `beta=0` 写权重梯度；
ZeroBubble 模式把 B 和 W 拆成两个入口，W 用 `beta=1` 累加 BF16 `main_grad`。分离
期间调用方必须保留 W 真正需要的两个输入；能延长原 buffer 寿命时无需额外复制，
不能保留时才需要显式 stash。

两条反向算子的参数结构、自动策略和生产入口互相独立，只共用采样脚手架。完整
MNK、生命周期约束、96 行逐点结果和复现命令见
[`QKV backward`](benchmarks/QKVproj-backward/BENCHMARK.md) 与
[`OProj backward`](benchmarks/Oproj-backward/BENCHMARK.md)。

## 开箱运行

依赖 CUDA 12.8、CMake、Ninja 和 CUTLASS。Golden 使用以下源码版本：

```text
TransformerEngine  a7aec214eb5c3969984a40c3accb6d66987d8f25
```

```bash
git clone https://github.com/CyberSecurityErial/fuse.git
cd fuse

CUTLASS_ROOT=/path/to/TransformerEngine/3rdparty/cutlass bash scripts/build.sh

./build/fuse_smoke --quick
```

### 默认入口与 SOTA 复现

下面是 CP8、中型宽度、全局 S=4K 的 Golden 命令。直接照抄，不要改 tile、通信 CTA、raster 或 swizzle：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./build/fuse_bench \
  --mode a2a_gemm_lhs \
  --m 512 --n 5120 --k 5120 \
  --batch 1 --q-heads 40 --head-dim 128 \
  --comm-ctas 0 --lhs-policy auto \
  --raster n --swizzle 1 \
  --warmup 10 --iterations 50 \
  --json-out oproj_cp8.json
```

正确输出应包含：

```text
comm_ctas=4 policy=m128n160 tile=128x160
fused p50 ≈ 82.9 μs
```

这是不需要逐 shape 调参的生产入口。v4.0 的极限表保持 `--lhs-policy auto --raster n --swizzle 1`，只扫描通信 CTA 数；核心 kernel 和 tile 选择逻辑没有修改。具体映射见 [`benchmarks/a2a+Oproj/BENCHMARK.md`](benchmarks/a2a+Oproj/BENCHMARK.md)。

相同软件和硬件下，短程复跑落在 Golden 的 ±5% 可视为正常。明显偏慢时按顺序检查：

1. `bash scripts/require_idle_gpus.sh` 必须通过；
2. `nvidia-smi topo -m` 中参与设备必须走 NVLink/P2P；
3. 各卡 SM clock 保持稳定，结果由最慢 rank 决定；
4. 依赖 commit 与上文一致，使用 Release/SM90a 构建；
5. 默认模式保留 `--comm-ctas 0 --lhs-policy auto --raster n --swizzle 1`；复现 v4.0 极限表时只替换 `--comm-ctas`。

完整 shape matrix：

```bash
# OProj：正式表分别运行 eager 与 CUDA Graph。
python3 'benchmarks/a2a+Oproj/oproj_shape_bench.py' \
  --phase fuse-launch-formal --phase fuse-launch-aggregate \
  --phase shape-table \
  --models representative_small,representative_medium,representative_large \
  --seqs 1024,4096,16384 --cps 8 \
  --results results/reproduce_cp8

# QKV：正式表必须使用 MPI 一进程一卡；eager 与 Graph 分列。
unset FUSE_QKV_COMM_POLICY FUSE_QKV_GEMM_POLICY
python3 'benchmarks/QKVproj+a2a/qkv_shape_bench.py' \
  --phase fuse-mpi-formal --phase fuse-mpi-aggregate \
  --phase comparison-table
```

## 性能

实验设置：单机 8×H200、NVLink、每卡 132 SM、BF16、CUDA 12.8；10 次 warmup + 50 次采样，表内延迟为跨 rank 最大值的 p50。最优分离实现取调优后的 TE+NCCL 与 cuBLASLt+NCCL 中较快者；纯 GEMM 百分比固定对比经典 cuBLAS。吞吐只计算 GEMM FLOPs，延迟包含通信。

当前版本：v11.2（生产 kernel 不变；新增四算子完整训练 E2E 验证）

| 启动口径 | CP4 对最强外部 | CP8 对最强外部 | 总胜场 | 纯 GEMM 中位数（CP4 / CP8） |
|---|---:|---:|---:|---:|
| Eager | 47/48，1.110× 中位 | 48/48，1.178× 中位 | 95/96 | 86.6% / 84.9% |
| CUDA Graph | 46/48，1.130× 中位 | 48/48，1.199× 中位 | 94/96 | 86.8% / 86.7% |

“最强外部”逐 setting 取 `min(TE Userbuffers, 最强 TE/cuBLASLt+NCCL 分离方案)`。
Eager 与 Graph 各自做10+50正式采样，不拿两列之间的差值当算子收益。Graph 在采样前
完成 capture、instantiate 和显式 upload。上述极限表包含 per-shape `comm_ctas`
单变量标定；默认自动入口不承诺零调参复现全部极限点。

QKV的96点v8结果中，Eager/Graph对TE Userbuffers均为`96/96`胜场，几何平均
分别领先`1.282×`和`1.337×`；对最强外部基线也均为`96/96`，几何平均分别领先
`1.215×`和`1.267×`。相对v7的全量p50几何平均提升为`1.0116×`和`1.0135×`。
融合吞吐达到经典cuBLAS的全量中位数为Eager `91.0%`、Graph `92.2%`；CP4分别为
`92.6%/93.4%`，CP8为`88.8%/90.0%`。所有case使用同一自动入口，运行时不读取
逐shape winner或TE结果。

v9 锁频异构矩阵使用三张 1500 MHz 卡与 1980 MHz 参考卡，HBM 均为 3201 MHz。CP2/4/6/8 共 22 个 setting 使用 5 次 warmup + 30 次正式采样并逐样本取 max-rank p50。短 QKV 实际启用的 7 点全部提升 `1.0619×～1.0901×`；长 QKV 全部回退为 `1.0000×`。OProj 启用的 13 点全部提升 `1.1593×～1.4216×`。所有启用点均通过 BF16 逐元素完全一致检查。

v10 反向矩阵每个算子覆盖 96 个 setting，并分别测 Eager/Graph 与普通/ZeroBubble，
共 768 份融合结果。下表的“前向占比”是 B+W 总 FLOPs 吞吐除以同 shape 已发布
前向融合吞吐；“cuBLAS 中位”使用本轮同卡组、同 MNK、匹配 `beta` 的两次经典
cuBLAS 纯 GEMM。

| 反向算子 | 调度 | Eager 前向占比几何平均 | Graph 前向占比几何平均 | Eager / Graph cuBLAS 中位 | Eager / Graph 989T MFU中位 |
|---|---|---:|---:|---:|---:|
| QKV | ZeroBubble B/W分离，`beta=1` | 100.5% | 98.0% | 91.0% / 90.4% | 63.7% / 64.0% |
| QKV | 普通同流 B→W，`beta=0` | 104.3% | 102.3% | 88.8% / 89.6% | 64.7% / 64.6% |
| OProj | ZeroBubble B/W分离，`beta=1` | 99.6% | 100.1% | 89.7% / 90.3% | 62.4% / 62.4% |
| OProj | 普通同流 B→W，`beta=0` | 104.0% | 104.6% | 89.5% / 90.1% | 63.1% / 63.4% |

所有四组几何平均都达到同 shape 前向吞吐的 98.0% 以上。小矩阵 exact smoke
覆盖 CP4/CP8、rank-major/causal、batch=2、同流/分离和 `beta=1`；正式 S=1K
再检查跨 rank route、epoch 与完整写入。v9 的 43 个旧前向 device-kernel 指令体
在 v10 Release 构建中保持一致。

适配版 TE Userbuffers 的 96 点正式强基线也已完成。QKV 的 Eager 普通/
ZeroBubble 几何平均加速为 `1.463×/1.475×`，Graph 为 `1.222×/1.216×`；OProj
对应为 `1.428×/1.430×` 与 `1.146×/1.146×`。PyTorch 原生 forward→autograd
backward 对照覆盖两种布局、CP4/CP8、batch=2、宽 GQA、普通与 ZeroBubble 共
16 组，全部通过；ZeroBubble 连续两次 `beta=1` 累加的最大绝对误差为
`0.0009765625`。

v11.2 的完整训练对照同时替换 QKV/OProj 前向和反向，原生侧与融合侧均开启
full-iteration CUDA Graph。Nanbeige、Llama-3 8B geometry、Qwen2.5 7B
geometry 的 1K–128K 共 15 个 setting 全部提速，完整 step 吞吐几何平均提升
`2.09%`，最大提升为 Nanbeige 16K 的 `4.26%`。该时间包含前向、激活重计算、
反向与优化器，不是把四个 microbenchmark 的收益简单相加。

完整数据与复现流程：

- [A2A + O-projection benchmark](benchmarks/a2a+Oproj/BENCHMARK.md)
- [QKV Projection + A2A benchmark](benchmarks/QKVproj+a2a/BENCHMARK.md)
- [锁频异构 CP benchmark](benchmarks/heterogeneous_cp/BENCHMARK.md)
- [QKV Projection backward benchmark](benchmarks/QKVproj-backward/BENCHMARK.md)
- [Output Projection backward benchmark](benchmarks/Oproj-backward/BENCHMARK.md)
- [四算子完整训练 E2E benchmark](benchmarks/e2e/BENCHMARK.md)
- [版本演进](VERSION_HISTORY.md)
