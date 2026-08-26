# A2A + O-projection

单机 Ulysses Context Parallel 的 fused A2A + O-projection CUDA kernel。

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

默认通信策略对短中序列使用 `comm4`，长序列根据 CP、`N` 和设备 NVLink 带宽选择 `comm6/8`。实验性全量成本模型可通过 `FUSE_A2A_LHS_COMM_POLICY=experimental_model` 启用；它默认关闭，尚不属于 Golden。模型口径和边界见 [`VERSION_HISTORY.md`](VERSION_HISTORY.md)。

## 开箱运行

依赖 CUDA 12.8、CMake、Ninja、CUTLASS 和 DeepGEMM headers。Golden 使用以下源码版本：

```text
TransformerEngine  a7aec214eb5c3969984a40c3accb6d66987d8f25
DeepGEMM           ec757bd0ba89bc3458d7c47456e197b10179bc8c
```

```bash
git clone https://github.com/CyberSecurityErial/fuse.git
cd fuse

CUTLASS_ROOT=/path/to/TransformerEngine/3rdparty/cutlass \
DEEPGEMM_ROOT=/path/to/DeepGEMM \
bash scripts/build.sh

./build/fuse_smoke --quick
```

### SOTA 复现

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

相同软件和硬件下，短程复跑落在 Golden 的 ±5% 可视为正常。明显偏慢时按顺序检查：

1. `bash scripts/require_idle_gpus.sh` 必须通过；
2. `nvidia-smi topo -m` 中参与设备必须走 NVLink/P2P；
3. 各卡 SM clock 保持稳定，结果由最慢 rank 决定；
4. 依赖 commit 与上文一致，使用 Release/SM90a 构建；
5. 命令保留 `--comm-ctas 0 --lhs-policy auto --raster n --swizzle 1`。

完整 shape matrix：

```bash
python3 benchmarks/oproj_shape_bench.py \
  --phase fuse-formal --phase shape-table \
  --models representative_small,representative_medium,representative_large \
  --seqs 1024,4096,16384 --cps 8 \
  --results results/reproduce_cp8
```

## 性能

实验设置：单机 8×H200、NVLink、每卡 132 SM、BF16、CUDA 12.8；10 次 warmup + 50 次采样，表内延迟为跨 rank 最大值的 p50。最优分离实现取调优后的 TE+NCCL 与 cuBLASLt+NCCL 中较快者；纯 GEMM 百分比固定对比经典 cuBLAS。吞吐只计算 GEMM FLOPs，延迟包含通信。

| CP | case 数 | 相对最优分离 | 纯 GEMM 的百分比 |
|---:|---:|---:|---:|
| 4 | 18 | 1.17×–2.21× | 54.5%–100.5% |
| 8 | 18 | 1.16×–2.96× | 61.3%–95.7% |

全局序列 1K 到 512K 的完整数据、调优空间和复现流程见 [`BENCHMARK.md`](BENCHMARK.md)；版本级代码差异和收益见 [`VERSION_HISTORY.md`](VERSION_HISTORY.md)。
