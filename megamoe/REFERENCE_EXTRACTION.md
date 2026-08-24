# DeepGEMM MegaMoE reference extraction

本文只提取数据流、角色、同步和 benchmark 口径，不复制 DeepGEMM kernel 源码。
后续实现可以持续对照原文件，也能避免把一个仍在变化的 PR 快照悄悄混入本项目。

## 版本锚点

参考仓库：

```text
/home/chen/workspace/source_code/DeepGEMM
```

本次审计固定在：

```text
HEAD                  ec757bd0ba89bc3458d7c47456e197b10179bc8c
local ref             upstream/pr/360
SM100 base            88965b078186ee7510ab9fc4f1d5ebc19adfa8d1
```

工作目录里还存在 `upstream/pr/360-current=f983307` 和
`upstream/pr/360-latest=e481caf`。它们是经过重写且结构不同的版本，不属于本蓝图的
对手。所有性能结果必须同时记录 commit，不能只写“PR360”。

PR360 在 `88965b0..ec757bd` 之间新增的是 SM90 路径；SM100 MegaMoE core 与
`88965b0` 相同。

## 源码地图

| 部分 | SM90 PR360 | 原版 SM100 |
|---|---|---|
| device kernel | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe_pingpong.cuh` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh` |
| device kernel 2 | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe_cooperative.cuh` | — |
| scheduler | `deep_gemm/include/deep_gemm/scheduler/sm90_mega_moe.cuh` | `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh` |
| heuristic | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp` | `csrc/jit_kernels/heuristics/mega_moe.hpp` |
| JIT runtime | `csrc/jit_kernels/impls/sm90_fp8_mega_moe_{pingpong,cooperative}.hpp` | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp` |
| public API | `csrc/apis/mega.hpp`, `deep_gemm/mega/__init__.py` | 同左 |
| shared layout | `deep_gemm/include/deep_gemm/layout/mega_moe.cuh` | 同左 |
| correctness | `tests/test_mega_moe_sm90.py` | `tests/test_mega_moe.py` |
| performance | `tests/bench_mega_moe_sm90.py` | `tests/test_mega_moe.py` |

在 `ec757bd0` 上可直接对照的行段：

| 路径 | 行段 | 内容 |
|---|---:|---|
| `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe_pingpong.cuh` | 357–625 | count、metadata、remote pull |
| 同上 | 627–755 | A/SFA 与 B producer |
| 同上 | 756–1024 | FP8 WGMMA/scales |
| 同上 | 1026–1212 | L1 SwiGLU/requant epilogue |
| 同上 | 1214–1289 | L2 peer scatter |
| 同上 | 1293–1413 | top-k combine |
| `deep_gemm/include/deep_gemm/scheduler/sm90_mega_moe.cuh` | 47–216 | persistent state machine/count wait |
| `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh` | 329–642 | dispatch |
| 同上 | 643–756 | TMA producers |
| 同上 | 757–869 | TCGen05 UMMA |
| 同上 | 874–1205 | L1/L2 epilogue |
| 同上 | 1213–1351 | combine |
| `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh` | 118–219 | SM100 wave scheduler/count wait |

## 统一算子契约

两条实现都覆盖完整 EP MoE：

```text
FP8 input x + top-k expert/weight
  -> EP dispatch
  -> expert-local Linear 1: [M_e,H] x [2I,H]^T
  -> clamp + SwiGLU + top-k weight + requant
  -> expert-local Linear 2: [M_e,I] x [H,I]^T
  -> EP combine scatter
  -> top-k reduction
  -> BF16 y
```

符号：

- `T`：每 rank 输入 token 数；
- `H`：hidden；
- `I`：expert intermediate hidden；
- `E`：全局 expert 数；
- `E_local=E/EP`；
- `Ktop`：每 token 选择的 expert 数；
- `M_e`：路由后落到本 rank 某个 local expert 的 token 数。

PR360 SM90 的具体 tensor/dtype 合同：

```text
x            E4M3 [T, H]
x_sf         FP32 [T, H/128]
w1           E4M3 [E_local, 2I, H]
w1_sf        FP32 [E_local, 2I/128, H/128]
w2           E4M3 [E_local, H, I]
w2_sf        FP32 [E_local, H/128, I/128]
topk_idx     INT64 [T, Ktop]
topk_weight  FP32 [T, Ktop]
y            BF16 [T, H]
```

`H`、`I` 都要求能被 128 整除，当前 API还要求 `I<=4096`。L1 input SF 是 per-row/per-128，
L1 产生的 L2 activation SF 是 per-row/per-64。默认 benchmark 的 `T` 是每 rank token数，
所以 EP8 全局 source token数为 `8*T`，路由 contribution总数为 `8*T*Ktop`（扣除 mask）。

SM90 transform只对 W1 的 FP8 data做 gate/up granularity-8 interleave；W1 SF保持上述自然
连续布局（gate全部 block在前、up全部 block在后，K block为内层），W2 data/SF也不做
这项 interleave。实现不能把 data transform机械套到 scale tensor。

共享 workspace 的关键内容：

```text
grid/NVLink barrier state
expert send count
expert recv count and sum
L1 arrival count per expert-pool M tile
L2 arrival mask per expert-pool M tile
(expert, source rank, slot) -> source token/top-k index
pool token -> {source rank, source token, top-k slot}
```

token pool 按 expert 排列，每个 expert 的容量向 `BLOCK_M` 对齐。这样 grouped GEMM
无需再做 pack，但会产生尾块浪费。

## 原版 SM100 数据流

### 物理组织

- 单个 persistent kernel；
- `grid = num_sms`；
- cluster size 2；
- 软件 grid barrier 假定整个 grid 同时驻留；
- 每 CTA 有 4 dispatch warps、4 non-epilogue warps，以及 1 或 2 个
  epilogue warpgroups。

non-epilogue warps 的职责：

1. activation + SFA TMA；
2. weight + SFB TMA；
3. cluster leader 发 2-CTA TCGen05 UMMA；
4. mainloop 中保留/no-op。

TMEM allocate发生在 setup 阶段，由全局 `warp_idx==3` 执行；它是第4个 dispatch warp，
不属于上面的 non-epilogue warp 4。

accumulator 在 TMEM，MMA issue warp 与 epilogue warpgroups 可以通过 TMEM
full/empty barrier 解耦。

### 端到端时序

```text
1. 每 CTA 统计自己负责 token 的 top-k expert count
2. 本 GPU global atomic 预留 per-expert 无冲突 slot；跨 rank recv-count/barrier再用
   system-scope 操作保证可见性
3. 将 source token/top-k index 写入目标 rank symmetric workspace
4. grid + rank barrier，finalize local expert token count
5. dispatch warps 在 source ranks 间 round-robin remote TMA pull
6. 写 local expert token/SF/weight/metadata pool，发布 L1 arrival count
7. scheduler 对一个 expert wave 发完全部 L1 tiles
8. L1: FP8 x FP4 UMMA -> BF16 gate/up -> SwiGLU -> top-k weight
9. 每 token/per-32 求 scale，量化 FP8，写 L2 pool，发布 L2 arrival mask
10. 同一 expert wave 发全部 L2 tiles
11. L2: FP8 x FP4 UMMA -> BF16 contribution，直接写回 source rank/top-k slot
12. rank barrier
13. 本 rank 双缓冲读取所有 top-k slot，FP32 累加，BF16 写 y
14. dispatch warps 同时清 workspace
```

scheduler 的 wave 顺序是：

```text
wave 0: all L1 tiles -> all L2 tiles
wave 1: all L1 tiles -> all L2 tiles
...
```

每个 CTA 从 `blockIdx.x` 起步，以 `num_sms` 为 stride。wave 宽度同时考虑
SM 填充、routing imbalance、尾浪和 expert weight 的 L2 locality。

### 可以迁移到 H200 的思想

- symmetric address mapping；
- count/index 两阶段 dispatch；
- source-rank round-robin pull；
- expert-padded token pool；
- L1 arrival count 和 L2 arrival mask；
- persistent expert-wave scheduler；
- dispatch、compute、combine 的流水；
- TMA + mbarrier + system-scope release/acquire 的协议范式；
- L1 epilogue提前乘 top-k weight；
- 最终 FP32 top-k reduction；
- workspace cleanup 与 combine 重叠。

### 不能搬到 H200 的机制

- FP8 x FP4 TCGen05 block-scale MMA；
- `SM100_TMA_2SM_LOAD_2D`；Hopper cluster2须改为 leader-issued
  `SM90_TMA_LOAD_MULTICAST_2D`；
- TMEM allocator 和 accumulator 双缓冲；
- 2-CTA UMMA；
- UMMA multicast arrival；
- UTCCP 将 scale 搬到 TMEM；
- TMEM-load epilogue；
- 动态 UMMA-N；
- SM100 8-bit STSM 路径。

Hopper WGMMA 的 accumulator 属于发出 MMA 的 math warpgroup。L1 的 SwiGLU、
amax 和量化因此应留在同一个 compute CTA；把它交给另一个 CTA 会先产生一次昂贵的
accumulator 落盘。

## SM90 PR360 对手

源码中没有名为 `WASP` 的类型。准确结构是 persistent single-kernel + CTA 内
warp-group specialization。

### CTA 内角色

```text
1 CTA / SM, cluster=1, 384 threads

hardware WG0:
  2 dispatch warps
  1 A+SFA TMA warp
  1 B TMA warp

hardware WG1:
  math/epilogue warpgroup 0

hardware WG2:
  math/epilogue warpgroup 1
```

WG0 使用 48 registers/thread，两个 math WG 使用 224 registers/thread；
`__launch_bounds__(384,1)` 加约 200 KiB 以上 dynamic SMEM，使其保持 1 CTA/SM。

### 两个 kernel policy

| policy | tile | 组织 | 默认适用区间 |
|---|---|---|---|
| pingpong | `M64 N128 K128` | 两个 math WG 交替处理不同 tile，一个做 MMA 时另一个做 epilogue | `T < 256` |
| cooperative | `M128 N128 K128` | 两个 math WG 各负责 64 行，共享 B tile | `T >= 256` |

auto threshold 默认为 256，可用 `DG_SM90_MOE_COOPERATIVE_THRESHOLD` 修改。
L2 在估算 tokens/expert 大于等于 256 时切 N-major，以提高 W2 weight 的 L2 reuse。

### SM90 数据流

```text
dispatch warps:
  count/slot reservation/index exchange
  -> rank barrier
  -> remote token+SF pull
  -> local expert pool
  -> release L1 arrival count

TMA producers + math WGs:
  acquire L1 arrival
  -> L1 FP8 WGMMA
  -> SwiGLU * top-k weight
  -> per-row/per-64 FP8 requant
  -> local L2 pool
  -> release L2 bit
  -> L2 FP8 WGMMA
  -> BF16 direct peer scatter

math WGs:
  rank barrier
  -> top-k slot pull/reduce
  -> BF16 y
```

这条实现已经给出了三段可复用的 producer/consumer 语义：

```text
dispatch release -> L1 acquire
L1 release       -> L2 acquire
L2 peer store    -> combine acquire/barrier
```

CTA-specialized 版本需要重新定义参与者数量和前进条件。现有 barrier 模板把参与者写死为
`num_sms`，named barrier 也假定 dispatch 和 math warps 在同一 CTA，不能机械拆分。

## 当前对手 benchmark 口径

SM90 harness 默认：

```text
EP=8
H=7168
I=2048
E=256
Ktop=8
T in {1,2,4,8,16,32}
activation_clamp=10
fast_math=1
```

虽然最大被测 `T` 只有 32，`get_symm_buffer_for_mega_moe` 按 LCM 384对齐，因此 exact
harness 的 `num_max_tokens_per_rank=384`；JIT specialization和 workspace容量也按384，
不能用32替代。bench还传入非空 `cum_stats`，cleanup会对每个 local expert执行统计
atomic。

一次被测调用包含：

- 将 `x/x_sf/topk_idx/topk_weight` copy 到 symmetric buffer；
- `torch.empty(y)`；
- fused kernel call。

JIT 和首次 warmup 不计。默认只有 5 次 warmup 和 1 次 timed iteration；每个 rank 用
本地 CUDA event计时，通常只有 rank0打印，并没有逐 sample收集 all-rank max。其 rough
TFLOPS 为：

```text
2 * received_tokens * H * I * 3 / time
```

rough HBM 没有计 scale、routing metadata 和 NVLink，不能当硬件 roofline。
PR 源码没有正式 SM90 结果表。

后续公平对比必须同时提供：

1. 原 harness 的 exact 口径；
2. kernel-only 口径；
3. 10 warmup + 50 samples 的所有 rank 最大值；
4. auto、forced pingpong、forced cooperative 三条对手结果；
5. 同一 routing seed、mask、量化和 symmetric buffer layout。
