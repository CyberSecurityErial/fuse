# Ulysses GEMM + All-to-All Fusion

单机 Ulysses Context Parallel 的 GEMM/All-to-All 融合算子。A2A+O-projection 已完成优化；QKV Projection+A2A 在v7.0联合选择通信CTA与GEMM tile。

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

QKV Projection+A2A提供`M128N128/N160/N192/N256/N320`五种policy。H200默认
用GEMM wave成本、QKV通信量、NVLink下界和cluster前进代价联合选择通信CTA与tile；
模型名称和逐case winner不参与选择，标定范围外自动回退成熟策略。完整口径见
[`benchmarks/QKVproj+a2a/BENCHMARK.md`](benchmarks/QKVproj+a2a/BENCHMARK.md)。

默认通信策略对短中序列使用 `comm4`，长序列根据 CP、`N` 和设备 NVLink 带宽选择 `comm6/8`。实验性全量成本模型可通过 `FUSE_A2A_LHS_COMM_POLICY=experimental_model` 启用；它默认关闭，尚不属于 Golden。模型口径和边界见 [`VERSION_HISTORY.md`](VERSION_HISTORY.md)。

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

当前融合版本：v7.0（OProj热路径沿用v4.0；QKV使用v7联合流水策略）

| 启动口径 | CP4 对最强外部 | CP8 对最强外部 | 总胜场 | 纯 GEMM 中位数（CP4 / CP8） |
|---|---:|---:|---:|---:|
| Eager | 47/48，1.110× 中位 | 48/48，1.178× 中位 | 95/96 | 86.6% / 84.9% |
| CUDA Graph | 46/48，1.130× 中位 | 48/48，1.199× 中位 | 94/96 | 86.8% / 86.7% |

“最强外部”逐 setting 取 `min(TE Userbuffers, 最强 TE/cuBLASLt+NCCL 分离方案)`。
Eager 与 Graph 各自做10+50正式采样，不拿两列之间的差值当算子收益。Graph 在采样前
完成 capture、instantiate 和显式 upload。上述极限表包含 per-shape `comm_ctas`
单变量标定；默认自动入口不承诺零调参复现全部极限点。

QKV的96点结果中，v7对TE Userbuffers的Eager/Graph胜场均为`96/96`，几何平均
分别领先`1.268×`和`1.319×`；对最强外部基线的胜场为`93/96`和`96/96`，几何
平均分别领先`1.201×`和`1.250×`。相对v6的全量p50几何平均提升为`1.147×`和
`1.156×`。融合GEMM-equivalent吞吐达到经典cuBLAS的全量中位数为Eager `90.8%`、
Graph `92.0%`；CP4分别为`93.1%/93.6%`，CP8为`86.7%/88.2%`。所有case使用
同一自动入口，运行时不读取逐shape winner或TE结果。

完整数据与复现流程：

- [A2A + O-projection benchmark](benchmarks/a2a+Oproj/BENCHMARK.md)
- [QKV Projection + A2A benchmark](benchmarks/QKVproj+a2a/BENCHMARK.md)
- [版本演进](VERSION_HISTORY.md)
