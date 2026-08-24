# 优化日志

本文件按实际发生顺序记录 GQA + Ulysses CP 两个 persistent kernel 的开发过程。每项记录包括假设、代码变化、测试 shape、编译参数、正确性、性能、观察和下一步。失败或回退方案不删除。

## 2026-08-16：任务冻结与环境审计

### 已知环境

- 目标合同：单机 8×H200、NVLink、SM90、132 SM/GPU；
- CUDA toolkit：12.8；
- PyTorch/TE baseline 环境：`/home/chen/miniforge3/envs/mmunlearner`；
- PyTorch 2.9.0+cu128、NCCL 2.27.5、Transformer Engine 2.19.0.dev0；
- 可复用 CUTLASS：`/home/chen/workspace/source_code/TransformerEngine/3rdparty/cutlass` 或 `/home/chen/workspace/source_code/flux/3rdparty/cutlass`；
- 已有 QKV FP8 baseline：`/home/chen/workspace/source_code/kill_te`；
- 设备 API 报告 compute capability 9.0、132 SM、约 140 GiB/GPU；`nvidia-smi` 的 8.9 字段与 CUDA runtime 不一致，编译和运行选择以 CUDA runtime 的 SM90 为准。

### 冻结的实现方向

- Kernel A：`A2A→GEMM`，通信 CTA 发布输入 tile，GEMM CTA 消费；
- Kernel B：`GEMM→A2A`，GEMM CTA 本地落盘并 release，通信 CTA acquire 后写最终 Ulysses 布局；
- 一个方向对应一个 CTA-specialized persistent launch；
- projection 使用 dense `(M,N,K,L=1)`，PV 使用 strided-batched `(M,N,K,L=B×H)`；
- GEMM algebra 与 Ulysses/GQA route descriptor 解耦；
- 第一实现以 BF16/FP32 accumulate 建立正确且可测的调度骨架，随后为 QKV projection 增加与 TE 对齐的 FP8 path。

### 待测基线

- pure cuBLASLt/CUTLASS GEMM；
- standalone A2A route；
- 显式 GEMM + NCCL A2A；
- TE + NCCL A2A；
- fused persistent kernel 的通信 CTA 数量、tile、raster sweep。

## 2026-08-17：BF16 自研 WGMMA 原型与淘汰结论

### v0：单级搬运的 monolithic CTA-specialized 原型

假设：先用一个最小 CuTe/WGMMA mainloop 验证单 grid 内通信 CTA 与 GEMM CTA 的角色划分、P2P UVA route 和 system-scope epoch，再逐步补齐 GEMM pipeline。

- tile：`BM=128, BN=128, BK=64`；
- dtype：BF16 输入/输出，FP32 accumulate；
- 一个 persistent grid 内固定区分通信 CTA 与 GEMM CTA；
- 两个方向均使用 global-memory epoch 的 system-scope release/acquire；
- 8 卡 correctness smoke：
  - `A2A→GEMM max_abs_diff=0`；
  - `GEMM→A2A max_abs_diff=0`；
- 性能 shape：`M=2048, N=5120, K=4096, L=1`，`num_comm_ctas=8`；
- pure cuBLAS BF16：`0.1459 ms`，`588.7 TFLOPS/GPU`；
- 原型：`1.6743 ms`，`51.3 TFLOPS/GPU`，只保留 pure GEMM 的 `8.7%`。

结论：同步/route 骨架能工作，但该自研 GEMM 不是可交付 mainloop。它把 GEMM 实现差异放大成了所谓“融合代价”，违反公平比较口径。

### v1：三阶段 `cp.async` ring

假设：v0 的主要损失来自全局内存搬运未形成流水；把 A/B 搬运改成三阶段 `cp.async` ring 后应显著恢复 WGMMA 吞吐。

- 相同 tile、shape、dtype 和通信 CTA 配额；
- 8 卡 correctness smoke 继续 exact pass；
- pure cuBLAS BF16：`0.1475 ms`，`582.5 TFLOPS/GPU`；
- 三阶段原型：`1.6644 ms`，`51.6 TFLOPS/GPU`，保留率 `8.9%`。

观察：相对 v0 的变化落在噪声范围，说明瓶颈不是再加一层简单搬运流水可以修复，而是整个 mainloop、warp specialization、TMA/WGMMA pipeline 和 persistent scheduler 都不具备成熟 GEMM 的质量。

决定：停止优化自写 GEMM。正式实现直接复用 CUTLASS 3 Hopper warp-specialized/persistent collective，并参考 Flux 的 CUTLASS policy、输入 barrier 和 epilogue visitor；本任务只修改 GEMM 与通信相接的 tile 边界。

### 架构偏航与纠正

审计 Flux 时一度把 Flux 的“双 stream、独立 GEMM/通信 grid”实现结构误当成最终架构。这与 `TASK.md` 已冻结的单-launch 条件冲突，现明确撤销：

- 最终每个方向仍必须是一个 monolithic CTA-specialized persistent CUDA launch；
- Flux/CUTLASS 只作为成熟 GEMM collective、tile policy、barrier/EVT 写法的来源；
- 双-stream Flux 风格实现只允许进入 baseline，不得以 fused final 名义报告；
- 统一 grid 使用固定通信/GEMM CTA 角色，并先用 cooperative residency、occupancy 上界和所有 role 同时驻留证明排除 persistent CTA 互等死锁；
- 如果标准 CUTLASS collective 无法安全嵌入统一 grid，必须留下编译/资源/性能证据后再讨论下一种单-launch 设计，不能静默回退。

## 2026-08-17：CUTLASS collective 嵌入 monolithic grid

### 正式 GEMM 路径

把自写 mainloop 替换为本地 CUTLASS 3.x 的 Hopper collective：

- BF16 输入/输出、FP32 accumulate；
- `TileShape=128×128×64`，`ClusterShape=1×1×1`；
- `KernelTmaWarpSpecializedPingpong` mainloop；
- `TmaWarpSpecialized` epilogue；
- stock persistent tile mapping 外包一层 scheduler，仅改 worker 的物理 block offset 和 persistent stride；
- `A2A→GEMM` 在 mainloop tile load 前做一次 system acquire；
- `GEMM→A2A` 在 stock epilogue TMA store 完成后发布 tile epoch；
- 通信 CTA 和 GEMM CTA 在同一个 cooperative launch 中占固定角色，没有双 stream fallback。

当前 kernel 每 CTA 使用 384 threads、205824 B dynamic shared memory。在本机 132 SM 上 occupancy 为 1 CTA/SM；launch 前查询 `cudaOccupancyMaxActiveBlocksPerMultiprocessor`，且强制
`communication_ctas + compute_ctas <= active_blocks_per_sm × sm_count`。因此所有通信/计算 role 可同时驻留，persistent wait 不依赖尚未调度的 CTA。

### 输入通信版本与失败记录

1. `uint4` 向量化 P2P load/store：正确，但 384-thread communication CTA 明显受线程数限制。大 shape 上 384-thread copy 为约 `2.09 ms`，临时 1024-thread standalone copy 为约 `1.11 ms`；正式 CUTLASS kernel 的 block shape 固定为 384 threads，不能用后者冒充可嵌入实现。
2. bulk v0：每个 task 仅搬 `K` 方向 256 元素、64 行，G2S/S2G transaction 太碎；standalone 从向量版本的约 `0.56 ms` 退化到 `1.008 ms`，fused 约 `2.570 ms`，淘汰。
3. bulk v1：每 task 从一个 peer 搬完整 `K/world` shard、32 行；一个 G2S contiguous bulk transaction，随后逐行 S2G 到 packed staging。standalone 大 shape 降至约 `0.69 ms`（不同 sweep 曾测到 `0.37 ms`，后续统一按 critical-rank event 口径重测）。该版本进入正式路径。

### scheduler 展平错误：一次缺算导致的虚假高吞吐

CUTLASS stock scheduler 的物理 grid 依 raster 可能是 `(workers,1,1)`，也可能是 `(1,workers,1)`。monolithic wrapper 最初只把 `grid.x` 当成 compute CTA 数；当 `AlongN` 返回 `(1,workers,1)` 时，实际上只启动了 1 个 GEMM CTA：

- 小 smoke 的第二个 N tile 全零，`A2A→GEMM max_abs_diff=2.01562`；
- staging 本身 `max_abs_diff=0`，由此排除 A2A route/TMA 可见性问题；
- 先前大 shape 报出的 `653.6 TFLOPS/GPU`、`97.6% retained` 是缺算后的无效结果，明确作废，不进入任何结论。

修复：把 CUTLASS 二维/三维物理 worker grid 展平为 `grid.x × grid.y × grid.z`，custom scheduler 仍用展平后的 compute worker id 和 compute-subgrid stride。修复后 8 卡 smoke：

- `A2A staging max_abs_diff=0`；
- `A2A→GEMM max_abs_diff=0`；
- `GEMM→A2A max_abs_diff=0`。

修复后的首个真实大-shape 结果（`M=4096,N=10240,K=8192,L=1,CP=8,comm_ctas=8`，3 warmup + 10 iterations，critical rank）：

- cuBLAS BF16：`1.0286 ms`, `668.1 TFLOPS/GPU`；
- 同 tile/collective、同 `AlongN` raster 的 pure CUTLASS：`1.3017 ms`, `527.9 TFLOPS/GPU`；
- bulk A2A：`0.6880 ms`；
- fused monolithic：`1.4561 ms`, `471.9 TFLOPS/GPU`；
- fused 相对 cuBLAS 保留 `70.6%`，相对同 policy CUTLASS 保留 `89.4%`。

这说明通信已与 GEMM 大量重叠，但 `AlongN` 为了让同一 M tile 的 N consumers 一起等待/启动，牺牲了 pure GEMM raster 效率。下一步不能再用缺算数据判断甜点区；需要在正确 worker 数下联合 sweep raster、producer 发布顺序和通信 CTA 数。

### 修复后 dense 通信 CTA sweep

统一 shape `M=4096,N=10240,K=8192,L=1,CP=8`，`AlongN,swizzle=1`，所有数字均为 8 rank critical path。完整计算修复后的 sweep：

| comm CTA | bulk A2A (ms) | fused (ms) | fused TFLOPS/GPU | vs cuBLAS | vs same-policy |
|---:|---:|---:|---:|---:|---:|
| 4  | 1.1833 | 1.6766 | 409.9 | 61.7% | 78.2% |
| 8  | 0.6918 | 1.4659 | 468.8 | 70.5% | 85.8% |
| 12 | 0.5124 | 1.4984 | 458.6 | 69.0% | 86.5% |
| 16 | 0.4351 | 1.4914 | 460.8 | 69.0% | 87.1% |
| 24 | 0.3913 | 1.5536 | 442.3 | 66.4% | 85.4% |
| 32 | 0.3715 | 1.6529 | 415.8 | 62.0% | 79.0% |

结论：通信单独带宽继续随 CTA 增加，但让出的 GEMM SM 代价在 8 CTA 后超过收益；当前大 dense shape 的绝对延迟甜点是 8 CTA。

### raster/swizzle 联合观察

同一 dense shape、comm=8：

- `AlongM,swizzle=1`：pure CUTLASS `1.0522 ms / 653.1 TFLOPS`，但 producer 逐 M 发布时大量不同 M 的计算 CTA 自旋，fused `2.0440 ms / 336.2 TFLOPS`；
- `AlongN,swizzle=1`：pure 约 `1.30 ms / 526 TFLOPS`，fused 约 `1.45–1.47 ms / 469–475 TFLOPS`，绝对最快；
- `AlongN,swizzle=2`：pure 恢复到约 `1.03 ms / 663 TFLOPS`，但 ready 同时覆盖更多 M，fused `1.5380 ms`；comm=12/16/24 也未超过 swizzle=1、comm=8；
- `AlongN,swizzle=4/8`：pure 约 `1.04/1.01 ms`，fused 退化到 `1.67/1.65 ms`。

结论：只优化 pure GEMM 的 raster 并不等于优化融合；首批 consumer 的 M 分布必须和通信发布次序共同选择。

## 2026-08-17：通用 problem/route、batched 与 inverse route

### 接口与正式路径收敛

- 新增独立 `GemmProblem`：`M/N/K/L`、A/B/D 元素 stride、dtype、transpose 合同、raster、swizzle；
- 新增独立 `UlyssesRoute`：CP rank/world、B/S、Q/KV/local heads、head dim、route kind/direction；
- CUTLASS A/B/D tensor stride 改为显式运行时 stride，支持 batch padding；B 的 batch stride 允许 0 表示 broadcast；
- `A2A→GEMM` 从 `L==1` 扩展为任意 `L>0`，ready 索引为 `[L,m_tile]`，peer input/staging 都带 L offset；
- bulk staging 固定为 65536 BF16 元素，并按 `K/world` 自适应 32/16/8 行 task，支持 CP=2 时更大的 shard；
- route 尾块不能安全形成 contiguous bulk transaction 时，在同一个 monolithic kernel family 内选择 vector communication policy；不会退化成独立通信+GEMM 两个 kernel；
- 已从构建和公开 API 删除被淘汰的自写 WGMMA 实验，正式库只编译 CUTLASS collective 版本。

### 正确性矩阵

所有 case 连续运行 8 个单调 epoch，不在迭代间清零 signal，结果均 `max_abs_diff=0`：

- `A2A→GEMM`：CP=2/4/8，`B=1,L=1` packed；
- `A2A→GEMM`：CP=4，`B=2,L=3`，非 tile 整倍数 `seq_local=150`，A/B/D 都有显式 padding/batch stride；该 case 走 vector tail policy；
- `GEMM→A2A HEAD_TO_SEQUENCE`：
  - CP=2, B=1, GQA=1, D=64；
  - CP=2, B=1, GQA=2, D=96；
  - CP=4, B=2, GQA=4, D=80，A/B/D padding stride；
  - CP=8, B=1, GQA=8, D=128；
  - CP=4, B=1, GQA=4, D=256；
- `GEMM→A2A SEQUENCE_TO_HEAD/inverse`：CP=4, B=2, GQA=4, D=80。

这覆盖 CP=2/4/8、`B=1/2`、GQA ratio 1/2/4/8、head dim 64/80/96/128/256、M/N tail、L-batched 和 inverse route。当前还需增加 B=4、大规模随机数值统计和 sanitizer。

### PV 方向真实 sweep 与 overlap

shape `M=4096,N=128,K=4096,L=8,CP=8`，`AlongM,swizzle=1`：

| comm CTA | fused ms | TFLOPS/GPU | vs cuBLAS | vs same-policy |
|---:|---:|---:|---:|---:|
| 8  | 0.2313 | 148.6 | 38.5% | 37.6% |
| 16 | 0.1516 | 226.6 | 58.3% | 57.6% |
| 24 | 0.1233 | 278.7 | 72.8% | 70.5% |
| 32 | 0.1125 | 305.5 | 79.9% | 77.3% |
| 40 | 0.1127 | 305.0 | 79.9% | 78.5% |
| 48 | 0.1247 | 275.5 | 70.8% | 68.9% |

甜点为 32–40 CTA，选择 32 作为较少占用 SM 的默认候选。加入 standalone route 后的复测（comm=32）：

- cuBLAS：`0.0900 ms`, `381.7 TFLOPS/GPU`；
- same-policy CUTLASS：`0.0865 ms`, `397.2 TFLOPS/GPU`；
- standalone route：`0.0510 ms`, payload `164.5 GB/s/GPU`，估算 remote `143.9 GB/s/GPU`；
- 顺序 CUTLASS+route：`0.1420 ms`；
- fused：`0.1129 ms`, `304.3 TFLOPS/GPU`；
- 按任务书公式 overlap ratio：`48.2%`。

同样加入完整基线后的 dense 复测（comm=8）：cuBLAS `1.0305 ms`，same-policy CUTLASS `1.3304 ms`，bulk route `0.7069 ms`，顺序 `2.0261 ms`，fused `1.4669 ms / 468.5 TFLOPS/GPU`，相对 same-policy 保留 `90.7%`，overlap ratio `80.7%`。

### bulk v2：单通信 CTA 双 TMA slot

观察：大 dense shape 的 `K/world=1024`，一个 32-row task 只占 64 KiB stage，而通信 CTA 已因统一 CUTLASS block shape 固定为 384 threads、共享内存预算约 201 KiB。原实现只让 thread 0 串行使用一半 stage。

改动：每个通信 CTA 建立两套互不重叠的 64 KiB stage + mbarrier，由两个 issuer thread 分别维护独立 G2S/S2G bulk group；不增加通信 CTA/SM，也不改变 GEMM shared-storage 上界。

结果（同一 dense shape、comm=8）：

- correctness 全矩阵与 8-epoch stress 继续 exact；
- standalone bulk route 从约 `0.707 ms` 降至 `0.471 ms`；
- payload 从约 `94.9` 提高到 `142.5 GB/s/GPU`，估算 remote 从 `83.1` 提高到 `124.7 GB/s/GPU`；
- fused 从典型 `1.467 ms` 小幅改善到正式 20-iteration 结果 `1.4393 ms / 477.4 TFLOPS/GPU`；
- compute-subgrid CUTLASS（只给 124 SM）为 `1.3369 ms`，fused 相对它保留 `92.9%`。

说明 route 带宽瓶颈被明显缓解，但 fused 的最终时间已主要由 AlongN GEMM 与共享 HBM/L2 争用决定；继续增加 TMA slot 不太可能线性改善融合时间。

## 2026-08-17：TE/NCCL 与显式 NCCL 基线

新增本地环境直接运行的 `te_nccl_baseline.py`，用 NCCL `all_to_all_single` 实现与两个 kernel 相同的布局：

- dense：head/feature-sharded global sequence → NCCL A2A + pack → local sequence/full K → TE BF16 Linear 或 torch/cuBLAS GEMM；
- PV：Batch Matrix Multiplication (BMM)/strided-batched PV GEMM → pack + NCCL A2A +
  unpack → `[B,S_local,H_global,D]`；等价 shape 为
  `P[8,4096,4096] × V[8,4096,128] → O[8,4096,128]`；
- 每个 sample 均取 8 rank CUDA event 最大值，barrier/all-reduce 本身不计入被测 event；
- 输出结构化 JSON；复用 `/home/chen/miniforge3/envs/mmunlearner`，没有新环境。

正式 dense shape `M=4096,N=10240,K=8192,CP=8`：

- NCCL A2A route（含 pack）：`0.3827 ms`；
- TE BF16 GEMM：`1.1535 ms`；
- cuBLAS BF16 GEMM：`1.0526 ms`；
- TE BF16 A2A→GEMM：`1.3786 ms`, `498.5 TFLOPS/GPU`；
- cuBLAS BF16 A2A→GEMM：`1.3695 ms`, `501.8 TFLOPS/GPU`；
- fused v2：`1.4393 ms`, `477.4 TFLOPS/GPU`。

结论：该 dense BF16 shape 的 fused 当前比 TE+NCCL 慢约 `4.4%`，比 cuBLAS+NCCL 慢约 `5.1%`。这个 shape 暂不能宣称融合有收益；原因是高效 NCCL route 很短，而为 ready-friendly 的 raster 降低了 pure GEMM 效率。仍保留该结果作为“融合必要性依 shape 而定”的负例。

正式 PV shape `M=4096,N=128,K=4096,L=8,CP=8`：

- NCCL route：`0.1246 ms`；
- torch/cuBLAS BF16 batched PV GEMM：`0.1116 ms`；
- batched PV GEMM + NCCL A2A（等价基线）：`0.2146 ms`, `160.1 TFLOPS/GPU`；
- same-policy CUTLASS：`0.0870 ms`；compute-subgrid（100 SM）：`0.0962 ms`；
- fused：`0.1127 ms`, `304.9 TFLOPS/GPU`。

结论：PV fused 相比 batched PV GEMM + NCCL A2A 等价基线延迟缩短约
`47.5%`（约 `1.90×`），这是当前最明确的融合收益场景。

### sanitizer

`compute-sanitizer` quick matrix覆盖 A2A tail、forward route、inverse route：

- memcheck：`ERROR SUMMARY: 0 errors`；
- racecheck：`0 hazards displayed (0 errors, 0 warnings)`。

### device timeline telemetry

计时路径的 telemetry 指针为空，不含 instrumentation。另起一个非计时 launch，用 `%globaltimer` 和可选 global atomics 记录每个 rank 的 tile milestone，再报告各 milestone 相对该 rank 最早 role 的最大值。

- dense `M4096,N10240,K8192,comm=8`：first comm tile `0 us`，first GEMM tile start `16.736 us`，first GEMM tile done `81.440 us`，last comm tile done `779.520 us`，last GEMM tile done `1439.232 us`；存在约 763 us 的明确通算并行区间。
- PV `M4096,N128,K4096,L8,comm=32`：first GEMM start `0 us`，first GEMM tile done `34.272 us`，first communication tile `38.624 us`，last GEMM tile done `97.600 us`，last communication tile done `106.432 us`；通信在首个 D tile 完成后启动，并与剩余 GEMM 重叠约 59 us。

这给出了单个 monolithic grid 内 role 并行的直接设备侧证据，而不是用双 stream 时间差推断。

### bulk v3：按 shard 大小使用最多 4 个 slot

`K/world=512` 时每个 32-row task 只占 32 KiB，128 KiB stage 可以安全容纳 4 个独立 slot。把 slot 上限从 2 提到 4 后，dense shape `M=2048,N=5120,K=4096` 的 route 从约 `0.257 ms` 降至 `0.175 ms`（comm=8）。随后重扫通信 CTA：

| comm CTA | bulk route ms | fused ms | fused TFLOPS/GPU |
|---:|---:|---:|---:|
| 4  | 0.3322 | 0.4136 | 207.7 |
| 8  | 0.1748 | 0.2478 | 346.6 |
| 12 | 0.1315 | 0.2032 | 422.6 |
| 16 | 0.1127 | 0.2086 | 411.8 |

该 shape 的甜点为 12 CTA。正式 20-iteration 复测：

- cuBLAS `0.1472 ms`；same-policy CUTLASS `0.1487 ms`；compute-subgrid CUTLASS `0.1724 ms`；
- bulk route `0.1315 ms`，payload `127.6 GB/s/GPU`，remote `111.7 GB/s/GPU`；
- 顺序 CUTLASS+route `0.2848 ms`；
- fused `0.2009 ms / 427.5 TFLOPS/GPU`，相对 full-SM same-policy 保留 `74.0%`；
- device timeline：first GEMM start `13.824 us`，last route `152.832 us`，last GEMM `182.816 us`。

等价 NCCL baseline 为 TE+NCCL `0.3000 ms`、cuBLAS+NCCL `0.2853 ms`。因此该中等 dense shape 的 fused 分别快约 `1.49×` 和 `1.42×`。这与大 dense shape 的负结果共同说明：融合是否必要取决于通信/计算比例和 ready-friendly raster 的 GEMM 代价，不能只给单一结论。

## 2026-08-17：本地 Flux 1.1.2 基线补全

### 构建审计与失败记录

本地 `/home/chen/workspace/source_code/flux` 初始只有 `build/lib/libflux_cuda.so`，没有
`libflux_cuda_ths_op.so` 和 `flux_ths_pybind`，不能直接运行两条公开 Python 算子。全程复用
`/home/chen/miniforge3/envs/mmunlearner` 和已有源码/build tree，没有创建新环境。

依次暴露并处理了以下本地构建问题；这些都属于 Flux baseline 准备，不是本项目 kernel 改动：

1. 旧 cache 为 `CUDAARCHS=80`，且 `libnvrtc.so` 指向已经消失的
   `/tmp/myflux_cublasmp_py/...`。该产物不能作为 H200 基线。重放 CMake 时固定
   `CUDAARCHS=90`、`GPU_SM_CORES=132`、CUDA 12.8 本地路径，生成 SM90a/H800 GEMMv3 注册。
2. Flux 在 `ENABLE_NVSHMEM=OFF` 时仍把 `coll/ths_op/isendrecv.cc` 放入统一 Torch 桥接目标，
   先报 `nvshmem.h` 缺失，随后因缺 `FLUX_SHM_USE_NVSHMEM` 报
   `nvshmem_create_tensor` 未声明。机器已有 pip NVSHMEM 头、host so 和 device archive，
   因而只补本地 include/definition/link dependency。
3. 只把宏加入编译会在 unified CUDA device-link 中产生
   `nvshmemi_device_state_d` 未定义；补入现成 `libnvshmem_device.a` 后 device-link 成功。
   该 archive 还必须进入 host-link，否则 `dlopen` 缺 `__fatbinwrap_*init_device*`。
4. 完整 pybind target 清单包含 `flux_coll_op`，但对应 `DisScatterForward` 在当前
   `ENABLE_NVSHMEM=OFF` build 中没有实现，导入时报未定义符号。最终 pybind 只编入本基线需要的
   `ths_op/gemm_only/a2a_transpose_gemm/gemm_a2a_transpose` 四个绑定；没有把无关 MoE/DisScatter
   依赖带入。临时 CMake dependency patch 已恢复，Flux 算法源码没有遗留修改。
5. 上游 `test_a2a_transpose_gemm.py --verify` 会把对齐的最大 local sequence 随机裁为
   `1..M`，但正式通信实现硬性检查 `local_seq_len % kTileM == 0`，其中 `kTileM=256`，
   因而上游随机-tail verify 自身失败。固定对齐 perf 路径末尾的 Torch-vs-Flux 检查成功；
   本仓 baseline 另外输出 max-rank `max_abs/max_rel/rel_l2/bitwise/passed`。这也是本项目
   支持 M tail、Flux 当前不支持的明确边界。

`benchmarks/flux_baseline.py` 和 `scripts/run_flux_baseline.sh` 现提供统一 JSON 口径：每个
sample 在事件外做 rank barrier，事件结束后对 elapsed 做 8-rank MAX。`FLUX_USE_NVSHMEM=0`
让单机 baseline 使用 CUDA IPC/NVLink 路径；`init_flux_shm` 的一次性初始化不计入时间。
Flux 实现作为成熟开源 baseline 使用，不冒充本项目要求的 monolithic CTA-specialized 单 grid。

### Flux communication-SM sweep

所有点为 3 warmup + 10 samples，单位 ms，correctness 均 passed。

等价 medium `A2A→GEMM, M2048,N5120,K4096,CP8`：

| Flux comm SM | 4 | 8 | 12 | 16 | 24 | 32 | 40 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| mean | 0.3966 | 0.3428 | 0.3285 | **0.2462** | 0.2520 | 0.3309 | 0.2898 |
| p50 | 0.3701 | 0.2677 | 0.2592 | **0.2446** | 0.2492 | 0.2717 | 0.2775 |

等价 large `A2A→GEMM, M4096,N10240,K8192,CP8`：

| Flux comm SM | 4 | 8 | 12 | 16 | 24 | 32 |
|---:|---:|---:|---:|---:|---:|---:|
| mean | 1.3352 | 1.2939 | 1.2536 | **1.2531** | 1.4068 | 1.4364 |
| p50 | 1.2973 | **1.2450** | 1.2560 | 1.2473 | 1.3350 | 1.4281 |

reverse dense analog `M4096,N1024,K4096,CP8`（总 FLOPs 和 A2A 元素数与
`PV M4096,N128,K4096,L8` 相同，但 GEMM 数据复用/布局不同）：

| Flux comm SM | 8 | 16 | 24 | 32 | 40 | 48 | 56 | 64 | 72 | 80 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mean | 0.3179 | 0.2994 | 0.2392 | 0.1996 | 0.2215 | **0.1766** | 0.2566 | 0.2461 | 0.2367 | 0.2457 |
| p50 | 0.2966 | 0.2313 | 0.1746 | 0.1971 | 0.1731 | **0.1680** | 0.1765 | 0.1807 | 0.2248 | 0.2105 |

### Flux 正式结果

10 warmup + 30 samples，8-rank critical mean：

- medium 等价，Flux comm=16：`0.2728 ms / 314.9 TFLOPS/GPU`，p50 `0.2583 ms`，
  bitwise exact；本项目 `0.2009 ms`，按 mean 约快 `1.36×`。
- large 等价，Flux comm=16：`1.2548 ms / 547.7 TFLOPS/GPU`，p50 `1.2490 ms`；
  `max_abs=0.0078125, rel_l2=3.96e-5`，passed。Flux 比本项目 `1.4393 ms` 快约 `14.7%`，
  因而 large negative case 在加入 Flux 后更明确。
- reverse dense analog，Flux comm=48：`0.1880 ms / 182.7 TFLOPS/GPU`，p50
  `0.1790 ms`，bitwise exact；本项目真正 batched PV 为 `0.1127 ms / 304.9 TFLOPS/GPU`。
  由于 Flux 没有 strided-batched PV collective，这一项只说明同 FLOPs/同 payload 的 dense
  方向参照，不能声称是等价 `1.67×` 胜出。

## 2026-08-17：QKV GQA 三段 production route 与 producer-order 调度

此前 `QKV_GQA_PACK` 只有 host address mapping；本轮把它接入正式
`GEMM→A2A` monolithic family。GEMM 仍是同一 CUTLASS SM90 collective，新增内容只在独立
communication CTA 中：等待 epilogue 的 `(M-tile,N-tile)` system-acquire epoch，将本地
`[B*S_local,(Hq+2Hkv)*D]` 按 Q/K/V 三段连续 head ownership 写到各 peer 最终
`[B,S_global,(Hq/CP+2Hkv/CP)*D]`。KV 不复制；`Hq` 或 `Hkv` 不能整除 CP 时 launcher 明确返回
`cudaErrorNotSupported`。

正确性新增独立 CPU reference（没有调用被测 mapping helper）并覆盖：

- CP=2：`Hq=8,Hkv=4,D=64`；
- CP=4：`B=2,S_local=150,Hq=16,Hkv=4,D=80`，A/B/D padded stride 和 M/N tail；
- CP=8：`Hq=64,Hkv=8,D=128`；
- quick case 为 CP=2、`S_local=150,D=80`；完整矩阵运行 8 个连续 epoch。

所有 case 相对 cuBLAS local GEMM + 独立 CPU 三段重排均逐元素 exact。新增路径再次通过
compute-sanitizer：memcheck `0 errors`，racecheck `0 hazards`。

### v0：ready-index 顺序造成 head-of-line blocking

代表 shape：CP=8，`M=512,N=10240,K=8192,Hq=64,Hkv=8,D=128`。最初 communication
任务直接按 ready 数组的 `(m*n_tiles+n)` 顺序分给 CTA；但输出 GEMM 默认 `AlongM`，CUTLASS
实际按 M tile 最快、N tile 最慢生产。结果是启动的多数 CTA 先等待很晚才会生产的 N tile，
已经 ready 的其他 M tile 被堵在各 CTA 的后续任务中。

初始 comm=8：same-policy CUTLASS `0.1722 ms`，standalone route `0.2634 ms / 39.8 GB/s`，
fused `0.4413 ms`；device timeline 的 last GEMM/last comm 分别约 `163/431 us`。扫到 comm=24
仍只有 `0.2695 ms`，相对 sequential 没有有效 overlap。

### v1：按 CUTLASS producer raster 排列 consumer task

consumer work-id 现在按 GEMM 的 effective raster 转为 `(tile_m,tile_n)`，再访问原 ready
signal；默认/AlongM 使用 M-fast，AlongN 使用 N-fast。该变化不轮询多个 flag，也不改变 epoch
协议，却消除了宽 N 的 CTA 队头阻塞。comm=24 的 fused 从约 `0.270 ms` 降到约
`0.213 ms`，task 定序本身贡献约 21%。PV 的 `N=128` 只有一个 N tile，语义不变。

### v2：消除每个 16-byte transfer 的重复 row route 整除

原 QKV copy 每个 `uint4` 都重复计算 batch/local-sequence/global-sequence。改为每 tile 用
最多 128 个 thread 计算一次 row destination base，放入约 1 KiB CTA-local shared memory；
每个 thread 在完整 128-column tile 上保持固定 feature-vector index，因此 Q/K/V feature mapping
只算一次并寄存器复用，tail 不能保持固定 index 时保留通用 fallback。GEMM shared-memory
mainloop 不变，occupancy 查询仍证明 1 CTA/SM 和全 role residency。

一次尝试把 16 个 feature owner/local-offset 也缓存到 function-local shared array，性能可到
comm=20 fused `0.1961 ms`，但 `D=80` tail/GQA case 出现确定性 vector permutation
（max abs `11.0156`），因此立即回退；没有以错误结果保留该优化。寄存器复用版本恢复所有
case exact，性能基本相同。

最终通信 CTA sweep（3 warmup + 10 samples，单位 ms）：

| comm CTA | 16 | 18 | 20 | 22 | 24 | 26 |
|---:|---:|---:|---:|---:|---:|---:|
| standalone route | 0.1060 | 0.0944 | 0.0862 | 0.0833 | 0.0802 | 0.0753 |
| fused | 0.2030 | 0.2011 | 0.1972 | 0.1990 | **0.1951** | 0.2262 |

comm=26 的退化不是 route 变慢，而是 compute workers 从 108 降到 106 后，320 个 GEMM tile
由 3 个 persistent waves 跨到 4 waves；compute-subgrid 从约 `0.166` 跳到 `0.216 ms`。
因此宽 QKV route 默认选 24 CTA，而不是一味增加通信 SM。

正式 10 warmup + 30 samples、comm=24：cuBLAS `0.1542 ms`（p50 `0.1461`），
same-policy CUTLASS `0.1682 ms`，compute-subgrid CUTLASS `0.1657 ms`，standalone route
`0.0801 ms / 131.0 GB/s`，sequential `0.2551 ms`，fused `0.2001 ms / 429.3 TFLOPS/GPU`。
相对 same-policy GEMM+route sequential 延迟降低 `21.6%`，任务书 overlap ratio 为 `60.1%`。
device timeline：first GEMM done/first comm/last GEMM/last comm 约
`55.6/58.5/163.0/192.5 us`，直接证明三段 P2P route 与剩余 GEMM 重叠。

## 2026-08-17：cuBLASLt SOTA 重标定与 8 卡时钟审计

### 旧 `cuBLAS` 数字不能单独代表 pure-GEMM SOTA

旧 C++ 基线实际调用
`cublasGemmStridedBatchedEx(..., CUBLAS_GEMM_DEFAULT_TENSOR_OP)`；它是有效的官方库基线，
但是没有做 cuBLASLt heuristic candidate 搜索，因而不足以回答“当前 shape 的
pure GEMM 上界是多少”。为此新增了两层重标定：

- `benchmarks/cublaslt_bench.cu`：单 GPU 独立基准，BF16 row-major
  `A[M,K] × B[N,K]^T`，FP32 accumulate/BF16 output；向 cuBLASLt 请求 64 个
  heuristic candidate（本机返回 8 个），对每个可用算法实测后选最快者，
  candidate tuning 不计入最终延迟；
- `fuse_bench` 集成同样的 row-major cuBLASLt 路径，每 GPU 预留
  64 MiB workspace，并在 JSON 中同时输出 `cublaslt_autotuned` 和
  `pure_sota`；`pure_sota` 永远取当次可比 pure 实现中的最快者，
  不强行把 cuBLASLt 标成胜者。

NVIDIA H200 公开规格的 BF16 `1,979 TFLOPS` 带 sparsity 脚注，对应 dense
理论值约 `989.5 TFLOPS/GPU`。本机快卡负载时钟为 `1,980 MHz`，按
`132 SM × 4096 BF16 FLOP/cycle/SM × 1.98 GHz` 换算的 clock-scaled dense
理论上限约 `1,071 TFLOPS/GPU`。两者都是硬件 roofline，不是库可达性声明；
实际优化目标使用实测 pure SOTA。

| shape | 单快卡 pure SOTA | 8 卡并发 max-rank pure | 当前目标口径 |
|---|---:|---:|---:|
| dense medium `M2048,N5120,K4096` | cuBLASLt `0.1013 ms / 848.3 T` | cuBLASLt `0.1434 ms / 598.9 T` | dense 冲刺目标 `848 T/GPU` |
| dense large `M4096,N10240,K8192` | cuBLASLt `0.8166 ms / 841.5 T` | legacy cuBLAS `1.0291 ms / 667.8 T`，cuBLASLt `1.0296 ms / 667.4 T` | 正式单卡值 `841.5 T/GPU`；不采用短 tuning 中约 `899 T` 的瞬时值 |
| batched PV `M4096,N128,K4096,L8` | cuBLASLt `0.0741 ms / 463.8 T` | same-policy CUTLASS `0.0882 ms / 389.6 T`，cuBLASLt `0.0886 ms / 388.0 T` | batched PV 目标 `463.8 T/GPU` |

上表的 8 卡 pure 列不含通信，只是所有 rank 同时运行后取最慢 rank；
fused 和 TE/Flux/NCCL 端到端列才包含通信。后续报告同时列出硬件 dense
peak、单快卡 pure SOTA 和当次多卡 max-rank pure，不再用一个含糊的
`“cuBLAS”` 数字代表所有上界。

### 8 卡并发只有约 600 T 的主因

在 8 GPU 上长时并发 pure GEMM，并用 `nvidia-smi dmon` 同时采样后，观察到：

- 物理 GPU `0/2/4/5/7` 负载下保持 `1,980 MHz`；
- 物理 GPU `1/3/6` 负载下固定在 `1,500 MHz`；
- sparse operation mode 为 disabled；温度约 `40–54 °C`，慢卡功耗约
  `340–516 W`，低于 `700 W` limit，未观察到温度或功耗墙证据。

`848.3 × 1500 / 1980 ≈ 642.7 T`；8-rank max 的长测约 `598.9 T`已能用慢卡
时钟解释大部分，其余约 `7%` 是算法选择、持续频率和测量抖动的合并效应。
这是当前机器的环境异构，不是把 dense 目标从 `848 T` 下调的理由。

因为当前 launcher 支持 CP=2/4/8，而快卡有 5 张，开发 sweep 固定选
`CUDA_VISIBLE_DEVICES=0,2,4,5` 运行 CP=4：这可以避免慢 rank 遮蔽 kernel 调度改动，
但不会冒充 CP=8 交付结果。CP=4 的 remote fraction 为 `3/4`，CP=8 为
`7/8`，拓扑和通信量都不同；最终 CP=8 仍需单独复测。

## 2026-08-17：standalone bulk reference 启动错误与旧结论撤销

### 根因与修复

bulk communication functor 的安全实现是“每个 active slot 一个完整 warp”：
`slot = threadIdx.x >> 5`，lane 0 发 TMA，其余 lane 参与完整 `__syncwarp()`。
但是旧 standalone BF16/FP8 bulk reference 都误启动为 32 threads，只有 slot 0
存在；functor 的 task stride 却仍按全部 slot 计算，因而一部分 task 根本没有执行。

已修为：

- BF16 standalone bulk：`kMaxBulkSlots × 32 = 128` threads；
- FP8 safe standalone bulk：`2 × 32 = 64` threads；
- mbarrier init 后执行 `fence_barrier_init`，等待前和 slot 复用前使用整 warp
  synchronization；FP8 当前最多 2 个 active slot。

monolithic fused launch 始终使用 CUTLASS 的 384-thread CTA，因此该 standalone
launch 错误不改变历史 fused 计时；这只是排除 reference launch 这一个影响，
不代表曾有 racecheck warning 的 FP8 fused 版本重新有效，FP8 交付仍必须以
racecheck clean 的整 warp/slot 安全版为准。该错误使以下历史结论全部失效：

- 修复前的 standalone BF16/FP8 `a2a_bulk_route` 延迟和带宽；
- 用该错误 reference 串行执行的 `sequential`；
- 所有以上述 route/sequential 推导的 overlap ratio；
- 如果 correctness 只在重复迭代后检查 staging，未清零区可能由早先迭代
  残留；这种结果不能作为 standalone reference 完整覆盖的证据。

因此，本文前文中 dense large 的 `0.471/0.707 ms`、dense medium 的
`0.1315 ms` 等旧 standalone bulk 数字仅保留为开发历史，明确撤销其
带宽和 overlap 含义。修复后的 `fast4_fixedroute_*` 及之后文件才可用于
standalone/sequential/overlap 对比。

## 2026-08-17：4 快卡 CP4 调度冲刺

统一 shape 为 BF16 dense `A2A→GEMM, M2048,N5120,K4096,L1,CP4`，统一取
4 rank CUDA event max。快卡 pure SOTA 在这些运行中约为 `746–763 T/GPU`；
单卡独占的 `848.3 T/GPU` 仍作为 dense 冲刺目标，不用多卡并发值取代。

### 修正 standalone launch 后的 CP4 baseline

`fast4_fixedroute` 短 sweep 为 3 warmup + 10 iterations；额外 comm=4 的
5+20 复测为 route `0.1571 ms`、fused `0.2345 ms`，与表中短测一致。

| comm CTA | corrected bulk route (ms) | compute-subgrid (ms) | fused (ms) | fused TFLOPS/GPU | 为当次 pure SOTA 的 | 为 848.3 T 目标的 |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.1575 | 0.1298 | 0.2330 | 368.7 | 48.6% | 43.5% |
| 6 | 0.1194 | 0.1443 | 0.1860 | 461.7 | 61.9% | 54.4% |
| 8 | 0.1013 | 0.1431 | 0.1733 | 495.7 | 65.7% | 58.4% |
| 10 | 0.0899 | 0.1425 | 0.1753 | 489.9 | 64.5% | 57.7% |
| 12 | 0.0877 | 0.1430 | 0.1719 | 499.6 | 65.7% | 58.9% |
| 16 | 0.0788 | 0.1391 | 0.1804 | 476.2 | 62.8% | 56.1% |

CP4 当前甜点为 comm=12。comm=4 中 standalone route `0.1575 ms` 和
compute-subgrid `0.1298 ms` 都不长，但 fused 为 `0.2330 ms`，只交叠了一小部分；
这证明下一阶段的主问题是 tile-ready 生产顺序/头阻塞，而不是只把
standalone route 带宽再做高。

### `rows=16, slots<=8` 失败实验

假设：把每个 TMA task 从 64 rows 缩到 16 rows，并把 slot 上限提到 8，
可以用更多独立 issuer 降低单 CTA 的长 transaction 气泡。实际上 CP4
`K/world=1024`时，16-row task 占 32 KiB，受 128 KiB stage 限制只能激活
4 slots；每个 M tile 的 ready arrival 从 8 增至 32。

| comm CTA | rows64 baseline route/fused (ms) | rows16 route/fused (ms) | 结果 |
|---:|---:|---:|---|
| 4 | 0.1575 / 0.2330 | 0.1668 / 0.2604 | route 和 fused 都更慢 |
| 8 | 0.1013 / 0.1733 | 0.1021 / 0.1770 | route 近中性，fused 为基线的 97.9% |
| 12 | 0.0877 / 0.1719 | 0.0867 / 0.1732 | route 近中性，fused 为基线的 99.3% |

结论：更细的 task 并没有增加有效 NVLink 带宽，反而增加 mbarrier/TMA
发射、system atomic 和整 tile 最后一个 arrival 的尾延迟。该配置淘汰，
当前 best 已回到 rows=64。

### slot-major/channel-pinned 映射：低 comm-CTA 数有效

旧 CTA-major 起始索引为 `comm_id * slots + slot`，一个 CTA 的多个 slot
会集中处理相邻 subtask。新映射改为：

```text
task start = slot * comm_ctas + comm_id
peer       = subtask % world_size
row_group  = subtask / world_size
```

即先按 row-group 展平、peer 最快，再按 slot-major 分发。当
`comm_ctas % world_size == 0` 时，每个 warp 在迭代间固定 peer，CTA 中各 slot
也稳定落在 NVLink channel，同一 issue wave 先覆盖所有 peer，不让一条 channel
的数据长时滞留。

| comm CTA | row-major/no-drain fused (ms) | channel-pinned fused (ms) | channel-pinned route (ms) |
|---:|---:|---:|---:|
| 4 | 0.2580 | 0.2330 | 0.1566 |
| 8 | 0.1753 | 0.1732 | 0.1019 |
| 12 | 0.1742 | 0.1733 | 0.0901 |

comm=4 改善最明显，fused 延迟为旧映射的 `90.3%`；comm=8/12 分别为
`98.8%/99.5%`。这说明改动主要消除低并发下的 channel 头阻塞，在通信
CTA 已足够时收益自然变小。该映射保留。

### 无信号 epilogue drain 条件化

`SignalingEpilogue` 旧版每个 D tile 都执行 `tma_store_wait<0>()`，即使
`ready == nullptr && telemetry == nullptr`。对 `A2A→GEMM` 计时路径，没有下游通信
消费 output tile，因而这个额外 drain 不提供顺序保证。现在只在
`ready || telemetry` 时 drain：

- `GEMM→A2A` 仍在 D tile 全局可见后 release signal，顺序语义不变；
- `A2A→GEMM` 的无信号计时路径去掉不必要的每 tile drain；
- 该改动与 row-major decode 在同一批 sweep 中落地，不能用这批数据单独
  归因。相比 rows16 版，comm=4/8/12 的 fused 从
  `0.2604/0.1770/0.1732 ms` 变为 `0.2580/0.1753/0.1742 ms`，总体近中性。

这个条件化保留，原因是它更符合数据依赖，而不是声称一个超出噪声的性能收益。

### ready flag 从紧密排列改为 128-byte 间距：近中性

旧 `kReadyFlagStride=1`，16 个 32-bit flag 挤在同一 64-byte cache line；系统域
producer atomic add 和多个 consumer acquire 会有 false-sharing 可能。新值为
`kReadyFlagStride=32`，每个 flag 相距 128 bytes。channel-pinned 5+20 A/B 结果：

| comm CTA | tight flags fused (ms) | 128-byte stride fused (ms) | 新版为旧版的 |
|---:|---:|---:|---:|
| 4 | 0.2330 | 0.2323 | 99.7% |
| 8 | 0.1732 | 0.1743 | 100.7% |
| 12 | 0.1733 | 0.1716 | 99.0% |

差异在约 `±1%` 内，该 shape 没有显示 signal cache-line 是主瓶颈。
128-byte 间距仍保留：它占用很小，可避免在更多 M/L tile 并发时引入明显的
系统域 false sharing；不将其记为性能胜点。

### Flux-like `128×256×64` projection policy

密集 projection family 从通用 `128×128×64` ping-pong policy 分离，改为参考
Flux Hopper projection family 的：

- `TileShape=(128,256,64)`；
- `ClusterShape=(1,1,1)`，本轮没有引入 cluster=2；
- `KernelTmaWarpSpecializedCooperative` mainloop；
- `TmaWarpSpecializedCooperative` epilogue；
- 显式 `StageCount<4>`。

batched PV 仍使用窄 `128×128×64` policy，不为了 dense projection 强行改变
PV tile。初版 3+10 sweep（仍使用 rows16 通信粒度）：

| comm CTA | compute-subgrid (ms) | corrected route (ms) | fused (ms) | fused TFLOPS/GPU |
|---:|---:|---:|---:|---:|
| 4 | 0.1340 | 0.1581 | 0.2445 | 351.3 |
| 6 | 0.1341 | 0.1200 | 0.1919 | 447.6 |
| 8 | 0.1336 | 0.1002 | 0.1751 | 490.6 |
| 10 | 0.1338 | 0.0958 | 0.1720 | 499.4 |
| 12 | 0.1330 | 0.0864 | 0.1711 | 502.0 |
| 16 | 0.1323 | 0.0772 | 0.1725 | 497.9 |

comm=12 的 reserved compute 从旧 policy 约 `0.143 ms` 改为 `0.133 ms`，为旧版的
`93.0%`；但 fused 仅从 ready128 的 `0.1716 ms` 到 `0.1711 ms`，为旧版的
`99.7%`，属于噪声级。这说明 compute-subgrid 本身已不是唯一约束；
communication readiness 和共享 HBM/L2 争用没有因更宽 N tile 自动解决。

随后把通信粒度回到 rows64，做 5+20 复测：

| comm CTA | corrected route (ms) | compute-subgrid (ms) | fused (ms) | fused TFLOPS/GPU |
|---:|---:|---:|---:|---:|
| 8 | 0.1138 | 0.1339 | 0.1858 | 462.2 |
| 10 | 0.1014 | 0.1335 | 0.1797 | 478.0 |
| 12 | 0.0924 | 0.1330 | **0.1696** | **506.4** |
| 16 | 0.0828 | 0.1333 | 0.1829 | 469.6 |

当前 CP4 best 是 rows64 + projection policy + comm=12：`0.16963 ms / 506.4 T/GPU`，
为当次多卡 pure SOTA `755.6 T/GPU` 的 `67.0%`，为单快卡 `848.3 T/GPU`
目标的 `59.7%`。它比 corrected sequential `0.2264 ms` 更快，但这里的
`63.4%` overlap ratio 仅使用修正后的 standalone route 计算，不与旧失效
overlap 数字混用。

### spin backoff `64→256` 失败

假设：增大 ready acquire 轮询的 `__nanosleep` 可以降低 system/L2 读流量，
让出更多资源给 TMA 和 GEMM。在 rows64 + `128×256×64` + CP4 上将
sleep 从 64 cycles 改为 256 cycles，comm=12 fused 从 `0.1696 ms / 506.4 T`
变为 `0.1738 ms / 494.2 T`，只有旧版吞吐的 `97.6%`；comm=8 也从
`0.1858 ms` 变为 `0.1865 ms`。更长 backoff 增加 ready 反应延迟，未换来可见的
带宽收益，因而回退到 64 cycles。

### 本轮公平对比口径

- hardware peak 只是 roofline；dense 实测目标为单快卡 `848.3 T/GPU`；
- pure SOTA 不含通信，每个 shape 在 cuBLASLt、legacy cuBLAS 和同 policy
  CUTLASS 中取当次最快者；
- TE/Flux/NCCL 和 fused 端到端数字都包含该路径定义的通信/重排；
- PV 等价对照统一写为“batched PV GEMM + NCCL A2A（等价基线）”，
  首次定义为 Batch Matrix Multiplication/strided-batched GEMM；不再用含糊的
  旧缩写标签；
- CP4 快卡 sweep 是调度开发 oracle，不与 CP8 延迟直接比值；
- standalone/sequential/overlap 只接受修正启动线程数之后的数据；旧数字保留为
  失败记录，不再进入性能结论。

## 2026-08-17：FP8 FastAccum 与可交付 bulk-mbarrier 协议

### GEMM policy 与通信 slot 安全性

FP8 E4M3 路径的 CUTLASS mainloop 改用
`KernelTmaWarpSpecializedCooperativeFP8FastAccum`，与 TE baseline 中
`fast_accum=true` 的数值/性能口径对齐。输入 bulk communication 同时完成了以下
mbarrier 修正：

- FP8 使用 `kFp8CommRows=32`；CP8 large 的 `K/world=1024` 时，每个 task
  占 `32 KiB`，128 KiB communication stage 正好激活 4 个 slot；
- 每 slot 由一个完整 warp 拥有，barrier 只在 task loop 之前初始化一次，
  随后执行 `fence_barrier_init()`；
- 每轮设置 transaction bytes，等待当前 `barrier_phase`，完成后用
  `barrier_phase ^= 1` 切换 phase，不再在循环内重复 init 同一 mbarrier；
- G2S wait 完成后执行 `tma_store_fence()`，然后才从该 slot 发 S2G；
  S2G `tma_store_wait<0>()` 完成后才对 tile-ready counter 做 system release；
- slot 退出 task loop 后 invalidate barrier。

新版 quick correctness 为逐元素 exact（`max_abs_diff=0`）；quick racecheck 为
`0 errors / 0 warnings`。之前带 racecheck warning 的 `0.7375 ms` 不是可交付结果，
继续明确作废；下文 `0.6837 ms` 是第一个同时满足完整计算、
correctness exact 和 racecheck clean 的 FP8 `A2A→GEMM` 正式结果。

### FP8 `A2A→GEMM` large CP8

shape：`M=4096,N=10240,K=8192,L=1,CP=8`。短 sweep 为 3 warmup +
10 iterations，全部为 8-rank CUDA-event max：

| comm CTA | compute-subgrid (ms) | corrected bulk route (ms) | fused (ms) | fused TFLOPS/GPU |
|---:|---:|---:|---:|---:|
| 4 | 0.6045 | 0.4786 | 0.7218 | 952.0 |
| 6 | 0.6134 | 0.3337 | 0.6868 | 1000.6 |
| 8 | 0.6052 | 0.2523 | **0.6806** | **1009.7** |
| 10 | 0.6024 | 0.2177 | 0.6905 | 995.2 |
| 12 | 0.6248 | 0.1981 | 0.7085 | 969.9 |

route 继续随 comm CTA 增加而变快，但 comm>8 后计算 SM 配额和通算争用
已超过 route 收益，因而甜点为 comm=8。10 warmup + 30 iterations
正式复测：

- same-policy CUTLASS：`0.6029 ms / 1139.8 TFLOPS/GPU`；
- compute-subgrid CUTLASS（124 compute SM）：`0.6135 ms / 1120.1 TFLOPS/GPU`；
- corrected bulk route：`0.2535 ms`，payload `132.4 GB/s/GPU`，remote
  `115.8 GB/s/GPU`；
- corrected sequential：`0.8605 ms / 798.6 TFLOPS/GPU`；
- fused：`0.6837 ms / 1005.1 TFLOPS/GPU`，为 same-policy pure 的 `88.2%`，
  为 compute-subgrid pure 的 `89.7%`，修正后 overlap ratio 为 `68.1%`。

device timeline 以首个 communication milestone 为原点：
`first_comm=0 us`、`first_gemm_start=8.512 us`、`first_gemm_done=38.176 us`、
`last_comm_done=328.320 us`、`last_gemm_done=669.152 us`。通信在整个 GEMM 结束前
约 `340.8 us` 已完成，首个 GEMM tile 在启动后 `8.5 us` 就开始，证明
ready wave 与主计算实际重叠。

同 shape 的 TE FP8 QKV + NCCL Ulysses 外部参照为
`1.1725 ms / 586.1 TFLOPS/GPU`；新 fused 吞吐为它的 `171.5%`，延迟比约
`1.71×`。严格语义上 TE 这条是 `QKV GEMM→Ulysses A2A`，与本节
`A2A→GEMM` 方向相反；二者 GEMM FLOPs 和 payload 相同，因而这里只将它作为
同规模端到端参照，不冒充严格同向 baseline。

### FP8 QKV `GEMM→A2A` large CP8

同一 `M4096,N10240,K8192,CP8`，QKV 三段 GQA production route 的短 sweep：

| comm CTA | compute-subgrid (ms) | QKV A2A route (ms) | fused (ms) | fused TFLOPS/GPU |
|---:|---:|---:|---:|---:|
| 24 | 0.6311 | 0.9373 | 1.0496 | 654.7 |
| 32 | 0.6830 | 0.8186 | 0.9144 | 751.6 |
| 40 | 0.7351 | 0.7566 | **0.8673** | **792.4** |
| 44 | 0.7855 | 0.7534 | 0.9124 | 753.2 |
| 48 | 0.8152 | 0.7205 | 0.9242 | 743.6 |
| 52 | 0.8405 | 0.7252 | 0.9587 | 716.8 |
| 56 | 0.8895 | 0.7032 | 1.0037 | 684.7 |

standalone route 在 comm=56 仍略有收益，但从 compute role 拿走的 SM 已使
compute-subgrid 从 comm=40 的 `0.7351 ms` 变为 comm=56 的 `0.8895 ms`，
因而端到端甜点为 comm=40。10+30 正式复测：

- pure/same-policy CUTLASS：`0.5333 ms / 1288.5 TFLOPS/GPU`；
- compute-subgrid CUTLASS（92 compute SM）：`0.7360 ms / 933.7 TFLOPS/GPU`；
- standalone QKV route：`0.7599 ms`，payload `110.4 GB/s/GPU`，remote
  `96.6 GB/s/GPU`；
- sequential：`1.1718 ms / 586.4 TFLOPS/GPU`；
- fused：`0.8625 ms / 796.7 TFLOPS/GPU`，为 pure SOTA 的 `61.8%`，为
  compute-subgrid pure 的 `85.3%`，overlap ratio 为 `80.8%`。

device timeline 的关键 milestone：`first_comm=31.136 us`、
`last_gemm_done=773.824 us`、`last_comm_done=785.824 us`。即通信在第一个 output tile
完成后立即开始，与剩余 GEMM 重叠约 `742.7 us`，GEMM 完成后只留约
`12.0 us` 通信尾波。

该方向与 TE FP8 QKV + NCCL Ulysses 是等价的 QKV projection +
Ulysses route；相对 TE 的 `1.1725 ms / 586.1 T`，fused 吞吐为 `135.9%`，
延迟比约 `1.36×`。

## 2026-08-17：BF16 wide projection 扩大 comm 配额的新 best

上一轮 rows64 + `128×256×64` projection policy 只扫到 comm=16，当时 best
为 comm=12 的 `0.1696 ms / 506.4 T`。继续扩展 CP4
`M2048,N5120,K4096` 的 comm 配额，3+10 短 sweep 为：

| comm CTA | compute-subgrid (ms) | corrected bulk route (ms) | fused (ms) | fused TFLOPS/GPU |
|---:|---:|---:|---:|---:|
| 20 | 0.1324 | 0.0794 | **0.1636** | **525.1** |
| 22 | 0.1309 | 0.0804 | 0.1673 | 513.4 |
| 24 | 0.1310 | 0.0776 | 0.1669 | 514.7 |
| 25 | 0.1319 | 0.0782 | 0.1659 | 517.9 |

comm=20 是新甜点。10+30 正式复测：

- cuBLASLt/pure SOTA：`0.1131 ms / 759.3 TFLOPS/GPU`；
- same-policy CUTLASS：`0.1355 ms / 634.2 TFLOPS/GPU`；
- compute-subgrid CUTLASS（112 compute SM）：`0.1319 ms / 651.4 TFLOPS/GPU`；
- corrected bulk route：`0.0791 ms`，payload `212.0 GB/s/GPU`，remote
  `159.0 GB/s/GPU`；
- corrected sequential：`0.2148 ms / 399.9 TFLOPS/GPU`；
- fused：`0.1637 ms / 524.8 TFLOPS/GPU`，为当次 local pure SOTA 的 `69.1%`，
  为单快卡 `848.3 T/GPU` 目标的 `61.9%`，修正后 overlap ratio 为
  `64.3%`。

该结果将上一轮 CP4 best 刷新为原延迟的 `96.5%`、原吞吐的 `103.6%`。
device timeline：`first_gemm_start=9.056 us`、`first_gemm_done=50.208 us`、
`last_comm_done=65.952 us`、`last_gemm_done=148.192 us`。通信尾波已不是最后边界，
当前剩余的主要气泡是 GEMM 最后 M panel/wave 的利用率。

下一个 A/B 已开始验证 `128×128×64` Cooperative + explicit
`StageCount<6>`：假设是更窄 N tile 可以用更多 worker 缩短最后 M panel 的尾波，
而 Cooperative stage-6 补回旧 ping-pong policy 的 pure-GEMM 效率。该实验尚无数据，
当前 winner 仍是 `128×256×64` + rows64 + comm=20。

## 2026-08-17：peer/K-ready 和 M-window=2 生产顺序

### 从整 M tile ready 细化到 peer/K group ready

旧 `A2A→GEMM` 协议对每个 `[l,m_tile]` 只有一个 counter；所有 peer 的 K shard
全部落到 staging 后，CUTLASS mainloop 才可以开始该 M tile 的第一个 K stage。
这保证了正确性，但丢失了“peer 0 已 ready 时先消费它的 K group，同时继续收
peer 1..P-1”的管道化机会。

新协议不改 CUTLASS MMA/epilogue，只修改 ready 粒度和 K iterator 边界：

- signal 布局改为 `[l,m_tile,peer]`，bulk/vector communication 完成一个 peer
  的所有 row-group/chunk 后，只对该 peer flag 做 system release；
- 每个 flag 的目标值为 `epoch * arrivals_per_peer`，不再等待
  `world_size * arrivals_per_peer`的整 tile 总和；
- 当 `k_per_rank % TileK == 0` 时，`ReadyKIterator` 包装 stock CUTLASS K iterator：
  首次解引用前等待当前 peer，K iterator 跨过 peer group 边界时再等待下一 peer；
- wait 只在 CUTLASS 已有 `elect_one_sync()` 的 TMA issuer lane 内执行，不在
  只有单 lane 进入的分支中错加 warp barrier；
- 当 peer 边界不与 `TileK` 对齐时，不强行切开 CUTLASS K stage；在
  mainloop tile 起点依次等待所有 peer，保留通用 shape 的安全 fallback。

peer/K-ready 单独落地后性能基本中性：

- BF16 CP4 comm=20 quick 为 `0.1639 ms / 524.0 T`，与上一个正式 best
  `0.1637 ms / 524.8 T` 一致；comm=12 quick 为 `0.1671 ms / 514.0 T`；
- FP8 CP8 comm=8 正式为 `0.6839 ms / 1004.8 T`，与旧正式
  `0.6837 ms / 1005.1 T` 一致。

原因是当时 communication task 仍倾向先让一个 M tile 的所有 peer 在很短间隔内
全部 ready；虽然 consumer 可以在 K 边界分段等待，producer 没有制造足够长的
peer-ready lead，所以还没有转化为端到端收益。

### M-window=2：让 peer-ready lead 实际暴露给 GEMM

下一步不增加门禁或 CTA，只改变 bulk task 的线性顺序。对每个连续的
2-M-tile window，先完成 peer 0 在这两个 M tile 上的所有 row group，再处理
peer 1，依次前进。这样可以同时启动两组 M consumer，而不会像全 M 范围
peer-major 那样让后续 peer 等待过久。如果 M tile 数不能整除 2，最后一个
window 用实际剩余 tile 数解码，不读写虚构 M tile。

quick sweep 全部正确性 PASS；其中峰值只用于选择正式复测点，不作为
交付性能：

| dtype/world | comm CTA | fused ms | fused TFLOPS/GPU |
|---|---:|---:|---:|
| BF16/CP4 | 8 | 0.1807 | 475.5 |
| BF16/CP4 | 10 | 0.1641 | 523.5 |
| BF16/CP4 | 12 | **0.1588** | **541.0** |
| BF16/CP4 | 14 | 0.1663 | 516.5 |
| BF16/CP4 | 16 | 0.1704 | 504.1 |
| BF16/CP4 | 20 | 0.1593 | 539.4 |
| FP8/CP8 | 4 | 0.7021 | 978.7 |
| FP8/CP8 | 6 | 0.6542 | 1050.4 |
| FP8/CP8 | 8 | **0.6415** | **1071.2** |
| FP8/CP8 | 10 | 0.6559 | 1047.8 |
| FP8/CP8 | 12 | 0.6575 | 1045.2 |

因此 `541.0 T` 和 `1071.2 T` 都标记为 quick-only candidate。正式口径仍是
10 warmup + 30 iterations、所有 rank CUDA-event max。

### M-window=2 正式结果

BF16 `M2048,N5120,K4096,CP4,comm=12`：

- cuBLASLt/pure SOTA：`0.1129 ms / 761.0 TFLOPS/GPU`；
- same-policy CUTLASS：`0.1359 ms / 632.3 TFLOPS/GPU`；
- compute-subgrid CUTLASS（120 compute SM）：`0.1332 ms / 644.9 TFLOPS/GPU`；
- corrected bulk route：`0.0956 ms`，payload `175.5 GB/s/GPU`，remote
  `131.7 GB/s/GPU`；
- corrected sequential：`0.2281 ms / 376.6 TFLOPS/GPU`；
- fused：`0.1602 ms / 536.1 TFLOPS/GPU`，为当次 pure SOTA 的 `70.4%`，
  为 `848.3 T/GPU` 单快卡目标的 `63.2%`，overlap ratio 为 `74.5%`。

与旧正式 best `0.1637 ms / 524.8 T` 相比，新正式吞吐为它的
`102.2%`（约提高 `2.2%`），延迟为它的 `97.9%`。正式 `536.1 T`
而不是 quick `541.0 T` 进入交付表。

FP8 `M4096,N10240,K8192,CP8,comm=8`：

- same-policy CUTLASS/pure：`0.6047 ms / 1136.4 TFLOPS/GPU`；
- compute-subgrid CUTLASS（124 compute SM）：`0.6082 ms / 1129.8 TFLOPS/GPU`；
- corrected bulk route：`0.2571 ms`，payload `130.5 GB/s/GPU`，remote
  `114.2 GB/s/GPU`；
- corrected sequential：`0.8674 ms / 792.3 TFLOPS/GPU`；
- fused：`0.6636 ms / 1035.6 TFLOPS/GPU`，为 same-policy pure 的 `91.1%`，
  为 compute-subgrid pure 的 `91.7%`，overlap ratio 为 `77.1%`。

与旧安全正式 `0.6837 ms / 1005.1 T` 相比，新正式吞吐为它的
`103.0%`（约提高 `3.0%`），延迟为它的 `97.1%`。正式 `1035.6 T`
而不是 quick `1071.2 T` 进入交付表。

### sanitizer 边界与通用性修复

- peer/K-ready-only 版本的 targeted racecheck 为 `0 hazards`（`0 errors / 0 warnings`）；
- M-window=2 只改 communication task 解码/顺序，quick correctness 已 exact PASS，
  但当前尚未运行 racecheck。在它通过前，不将 peer-only 版本的
  `0 hazards` 结论外推给 M-window=2；
- 每个 `k_per_rank` 现在独立要求 16-byte 对齐，而不是只验证 global K；
  BF16 对应 8 个 element，FP8 对应 16 个 element，确保每个 peer shard
  的 vector/TMA 起点合法；
- `comm_rows_for` 把 stage capacity 下取为不超过 dtype 上限的最大 2 次幂，
  防止通用 `k_per_rank` 产生任意 rows 后破坏 row-group/window 展平；
- bulk fast path 要求 M 整 tile 和 seq/rows 可整除；M tail 或不可形成完整
  bulk transaction 时，launcher 回退到同一 monolithic family 的 vector
  communication policy，不拆成通信/GEMM 两个 kernel；
- CUTLASS swizzle 可能向 mainloop 传入 dummy/invalid work tile。`ReadyMainloop` 现在先检查
  `m < 0 || m >= m_tiles`；dummy tile 直接走 stock `Base::load`，不计算 ready 索引，
  避免越界 system-acquire。

本轮结论：peer/K-ready 只提供粒度，单独使用近中性；M-window=2
让 producer 顺序与该粒度匹配后，BF16/FP8 才分别得到约 `2.2%/3.0%`
的正式收益。当前性能 winner 是 M-window=2，但安全交付状态仍等待
M-window=2 targeted racecheck。

## 2026-08-17：自适应 M-window、projection cluster2 与最终正式矩阵

### ready producer/consumer 收敛

固定 `M-window=2` 后继续把 window 改为运行时工作前沿：

```text
window = ceil(ceil(compute_ctas / gemm_n_tiles) / gemm_m_tiles_per_ready)
window = clamp(window, 1, min(8, producer_m_tiles))
```

bulk task 采用 `[L,window,peer,M-in-window,row-group]` 顺序；consumer 的
`ReadyKIterator` 在 peer/K group 边界等待。首个 peer 的 system acquire 已从
`Base::load` 内部移到 CUTLASS pipeline acquire 之前，后续 peer 仍由 iterator
的 `operator++` 在下一个 producer acquire 前等待。这样不占住一个空 mainloop
stage，也不在每次 A/B iterator dereference 上重复首包判断。

本轮还固定了以下通用性条件：

- input-ready 容量是
  `L * ceil(M/128) * world * kReadyFlagStride` 个 `uint32_t`；输出方向是
  `L * ceil(M/128) * ceil(N/128) * kReadyFlagStride`；公开 API 已写明；
- 每个 peer 的 `K/world` 必须独立满足 16-byte alignment；当 peer 边界不与
  CUTLASS TileK 对齐时，mainloop 在 tile 起点等待所有 peer，而不是切开一个
  WGMMA K stage；
- bulk rows 向下取不超过 staging capacity 的 2 次幂，始终整除 BM128；M 尾块
  回退到同一 monolithic vector CommOp，不拆 kernel；
- swizzle 生成的 dummy M tile 直接进入 stock CUTLASS predicate path，不访问
  real-ready allocation；
- input GEMM 的无 signal epilogue 不再无条件执行 `tma_store_wait<0>()`，避免把
  fused 与 pure 的 epilogue pipeline 人为做成不等价。

完整 smoke 新增 BF16 `K=192` 与 FP8 `K=384` 的 CP2/4/8 case，覆盖
16-byte aligned 但非 TileK-aligned peer shard；另加入 M=150 vector fallback、
swizzle=4 dummy tile 和自动 comm CTA 选择。全部数值 exact。生产版本的完整
`fuse_smoke` 连续 8 epoch 通过；full memcheck 为 `0 errors`；full racecheck 为
`0 hazards / 0 errors / 0 warnings`。因此上一节“等待 M-window racecheck”的状态
在此正式关闭。

### Flux-style projection cluster `(2,1,1)`

BF16 projection 保留 `128x256x64`、Cooperative mainloop/epilogue、stage 4，
cluster 从 `(1,1,1)` 改为 Flux H800 tuned config 使用的 `(2,1,1)`。这不是只改
template tag：最终 cubin 的 pure、bulk-monolithic、vector-monolithic entry 均为
168 registers/thread、0 stack、0 local spill，并生成
`SM90_TMA_LOAD_MULTICAST`。

monolithic launch 同时设置 `cudaLaunchAttributeClusterDimension` 与
`cudaLaunchAttributeCooperative`。communication CTAs 占 x-flattened grid 的完整
cluster prefix；host 明确要求 `comm_ctas % 2 == 0` 与 `compute_ctas % 2 == 0`。
由于 `block_offset` 和 compute stride 都是 cluster size 的倍数，physical
`block_id_in_cluster.x` 与 CUTLASS logical worker rank 一致。若让一个 physical
cluster 混入 comm/GEMM 两种 role，CUTLASS cluster arrive/wait 会确定死锁；当前
host gate 明确排除该情况。

residency 不再用 cluster1 的 blocks/SM 公式外推。host 使用实际 384-thread block、
214016-byte dynamic SMEM 和 cluster attr 调用
`cudaOccupancyMaxActiveClusters`，并证明 total grid clusters 全部可同时驻留；本机
132-CTA grid 实际通过 66-cluster gate。pure reference 同样使用 extended cluster
launch，确保 same-policy 对照没有漏掉 cluster multicast。

同环境 A/B：

| shape / config | cluster1 fused | cluster2 fused | 结论 |
|---|---:|---:|---|
| BF16 CP4 `(2048,5120,4096,1)`, comm12, 10+50 | `0.1482 ms / 579.7 T` | `0.1470 ms / 584.4 T` | cluster2 高约 0.8% |
| BF16 CP8 `(4096,10240,8192,1)`, comm10, 10+50 | `1.1475 ms / 598.9 T` | `1.1472 ms / 599.0 T` | 实质持平 |

因此生产保留 Flux-style cluster2；它在 medium 有小幅正收益、large 不退化，并为
纯 projection 的 B multicast 保留成熟路径。

### 失败 policy：`64x256x64` Pingpong

该候选可编译、correctness exact、无实质 spill，但 quick pure 只有
`0.1946 ms / 441.4 T`，compute-subgrid `0.2101 ms / 408.9 T`，fused
`0.2380 ms / 360.9 T`；相对恢复后的 `128x256` Cooperative fused 仅约
`62.4%`。失败不是 wave tail：两者约 214 KiB SMEM、384 threads，均为
1 CTA/SM；`64x256` 把 tile 数从 320 翻到 640，重复 B panel load 和 fixed
scheduler/epilogue 开销，并由 Pingpong ordered barrier 让两个 math warp group
按整 tile 交替。该 policy 已淘汰，不进入后续 comm sweep。

### 最终 comm CTA 甜点与默认 heuristic

cluster2 BF16 medium CP4 的 5+15 sweep：

| comm CTA | route ms | fused ms | fused TFLOPS/GPU |
|---:|---:|---:|---:|
| 8 | 0.1181 | 0.1744 | 492.5 |
| 10 | 0.1044 | 0.1592 | 539.6 |
| 12 | 0.0942 | **0.1460** | **588.2** |
| 14 | 0.0903 | 0.1494 | 575.0 |
| 16 | 0.0833 | 0.1496 | 574.4 |
| 18 | 0.0824 | 0.1511 | 568.6 |
| 20 | 0.0781 | 0.1532 | 560.6 |
| 22 | 0.0760 | 0.1536 | 559.1 |
| 24 | 0.0808 | 0.1518 | 565.9 |

large CP8 的候选中 comm10 正式 `1.1472 ms / 599.0 T`；comm8 正式
`1.1675 ms / 588.6 T`，comm14 以后 compute-subgrid 损失明显。默认 heuristic
因此使用 BF16 `N>=8192 -> 10`、BF16 `N>=2048 -> 12`；FP8 large 输入继续用 8；
FP8 large QKV 输出用 40；batched PV 用 32。它们是 shape-family heuristic，
不是对某个模型名做固定 dispatch。

### 10+50 最终性能与严格基线

所有 C++/Python 正式结果均为 10 warmup + 50 measured iterations，单次样本取所有
rank 的 CUDA-event 最大值。CP4 使用物理 GPU `0,2,4,5`；CP8 使用全部 8 卡。
任务指定的 forward 是 `A2A→QKV projection→QK→PV→A2A`，所以下表只列两个
正式融合边界；不会把额外的 dense QKV 输出测试误写成 forward 的第一步。

| 正式融合边界 / dtype / CP / `(M,N,K,L)` | fused | fastest pure | 标准分离方案（含通信） | Flux |
|---|---:|---:|---:|---:|
| A2A→projection, BF16, CP4, `(2048,5120,4096,1)` | **0.1470 ms / 584.4 T** | cuBLASLt 763.0 T | TE+NCCL 226.4 T；cuBLAS+NCCL 330.9 T | 370.5 T |
| A2A→projection, BF16, CP8, `(2048,5120,4096,1)` | **0.1888 ms / 455.0 T** | cuBLASLt 578.9 T | TE+NCCL 280.8 T；cuBLAS+NCCL 302.5 T | 323.6 T |
| A2A→projection, BF16, CP8, `(4096,10240,8192,1)` | **1.1468 ms / 599.2 T** | cuBLAS 676.7 T | TE+NCCL 492.9 T；cuBLAS+NCCL 495.4 T | 551.6 T |
| A2A→projection, FP8, CP8, `(4096,10240,8192,1)` | **0.6602 ms / 1041.0 T** | cuBLASLt 1216.6 T | cuBLASLt FP8+NCCL 846.8 T | 融合算子仅支持 BF16；FP8 运行时拒绝 |
| BF16 batched PV→A2A, CP8, `(4096,128,4096,8)` | **0.1129 ms / 304.5 T** | CUTLASS 403.9 T | PyTorch/cuBLAS batched PV+NCCL 142.7 T | 无等价 batched PV op |

下面一行只是 `GEMM→A2A` 通用 route 的附加能力测试，不属于上述 forward 的计算
顺序。选 dense QKV projection 是因为现有 TE FP8 Ulysses reference 正好提供该顺序，
可以严格比较相同 GQA 三段最终布局；该 reference 的 GEMM 实际由
`torch._scaled_mm`/cuBLASLt 执行：

| 附加泛化测试 / dtype / CP / `(M,N,K,L)` | fused | fastest pure | 精确分离方案（含通信） | Flux |
|---|---:|---:|---:|---:|
| dense QKV projection→GQA-pack A2A, FP8, CP8, `(4096,10240,8192,1)` | **0.8675 ms / 792.1 T** | CUTLASS 1283.0 T | cuBLASLt FP8+NCCL 560.5 T | 融合算子仅支持 BF16；FP8 运行时拒绝 |

对应的明确比例：

- BF16 CP4 medium：fused/pure `76.6%`；TE+NCCL/pure `29.7%`；
  cuBLAS+NCCL/pure `43.4%`；fused/最强分离方案 `176.6%`；
- BF16 CP8 medium：fused/pure `78.6%`；TE+NCCL/pure `48.5%`；
  cuBLAS+NCCL/pure `52.3%`；fused/最强分离方案 `150.4%`；
- BF16 CP8 large：fused/pure `88.5%`；TE+NCCL/pure `72.8%`；
  fused/最强分离方案 `121.0%`；
- FP8 A2A→projection：fused/cuBLASLt pure `85.6%`；标准分离方案/pure
  `69.6%`；fused/标准分离方案 `122.9%`；
- 附加 FP8 dense QKV projection→GQA-pack A2A：fused/same-policy pure `61.7%`；exact-layout
  cuBLASLt+NCCL/cuBLASLt pure `46.5%`；fused/标准分离方案 `141.3%`；
- BF16 batched PV→A2A：fused/same-policy pure `75.4%`；PyTorch/cuBLAS
  batched PV+NCCL 相对其自身 pure `45.5%`；fused/标准分离方案 `213.4%`。

H200 nominal dense roofline 继续按 BF16 989.5 T、FP8 1979 T 报告；BF16
单快卡实测冲刺目标为 848.3 T。CP8 的 max-rank pure 受三张 1.5 GHz 卡限制，
不据此下调目标。

### baseline 口径修正

`benchmarks/te_nccl_baseline.py` 新增精确的 `qkv_gemm_a2a`：先做 dense
QKV GEMM，再把 Q/K/V 三段按 `Hq/Hkv/world` 切片并执行 NCCL A2A，最终布局与
production `QKV_GQA_PACK` 完全一致。FP8 pure 使用
`torch._scaled_mm(..., use_fast_accum=True)`，即 cuBLASLt 路径；不把它错误命名为
TE。最终重放得到 FP8 output 精确标准方案 `1.2260 ms / 560.5 T`。

旧日志中用 `GEMM→A2A` 的 TE FP8 数字去参照反方向 `A2A→GEMM` 的
`171.5%` 只保留为历史“同 FLOPs/同 payload、反方向”观察，不再进入最终精确表。
最终同向 FP8 input 基线是 cuBLASLt FP8 GEMM+NCCL A2A：
`0.8115 ms / 846.8 T`，当前 fused 为其 `122.9%`。

较早一轮、用于确认配置甜点的正式 JSON（最终汇总表已由下方归档重放替代）：

- `results/fast4_projection_cluster2_comm12_formal.json`；
- `results/cp8_bf16_medium_projection_cluster2_comm12_final.json`；
- `results/cp8_bf16_large_projection_cluster2_comm10_final.json`；
- `results/cp8_fp8_a2a_projection_comm8_final.json`；
- `results/cp8_fp8_qkv_projection_a2a_comm40_final.json`；
- `results/cp8_bf16_pv_gemm_a2a_m4096_n128_k4096_l8_final.json`；
- `results/cp8_cublaslt_fp8_qkv_gqa_a2a_final.json`；
- `results/cp8_flux_a2a_gemm_m2048_n5120_k4096_comm16_final.json`；
- `results/cp8_flux_a2a_gemm_m4096_n10240_k8192_comm16_final.json`；
- `results/cp8_flux_fp8_a2a_projection_capability.json`；
- `results/cp8_flux_fp8_qkv_gemm_a2a_capability.json`。

归档前又完整执行了一次 `scripts/run_bench_8gpu.sh`，并分别重放五条标准
TE/cuBLASLt+NCCL 命令。上述表采用这批最新 10+50 mean；对应文件为：

- `results/cp8_bf16_medium_projection_final.json`；
- `results/cp8_bf16_large_projection_final.json`；
- `results/cp8_fp8_a2a_projection_final.json`；
- `results/cp8_bf16_pv_gemm_a2a_final.json`；
- `results/cp8_fp8_qkv_projection_a2a_final.json`；
- `results/cp8_te_nccl_bf16_medium_final.json`；
- `results/cp8_te_nccl_bf16_large_final.json`；
- `results/cp8_cublaslt_fp8_a2a_projection_final.json`；
- `results/cp8_te_nccl_bf16_pv_final.json`；
- `results/cp8_cublaslt_fp8_qkv_gqa_a2a_final.json`。

CP4 行另外来自：

- `results/fast4_projection_cluster2_comm12_formal.json`；
- `results/fast4_te_nccl_bf16_m2048_n5120_k4096_cp4_formal.json`；
- `results/fast4_flux_a2a_gemm_m2048_n5120_k4096_comm8_final.json`。

### Flux FP8 两个缺项的能力边界

最终表原先两格写成 `n/a`，容易被误读为漏测。本轮在相同 CP8 large shape
`(M,N,K,L)=(4096,10240,8192,1)` 上直接调用 Flux 1.1.2 的真实融合算子构造器，
而不是只读测试脚本：

- `AllToAllTransposeGemm` 的 8 个 rank 一致在
  `all_to_all_transpose_gemm_kernel.cc:361` 拒绝 E4M3 输入，错误为
  `A2A Gemm Only support BF16 input`；
- `GemmAllToAllTranspose` 配置为 `QKVPackA2A, gqa=8` 后，8 个 rank 一致在
  `gemm_all2all_transpose.cc:315` 拒绝 E4M3 输入，错误为
  `gemm + all2all + transpose only accept BF16 input`。

两边的 forward 还都拒绝 scale 参数，generator space 也只有 BF16 dtype config；
远端官方 `main`（探测时 HEAD `19831ca2d820e3e782ed1d15d8b52d0898b78b26`）仍保留
同样的 BF16 gate 和 BF16-only generator space。因此这不是参数没调到，而是 Flux
没有可运行的等价 FP8 融合实现。最终表把两格明确写成“BF16-only / runtime rejected”；
不把 Flux `GemmOnly` FP8 加 NCCL 冒充 Flux fusion，因为其语义就是已经列出的分离方案。

## 2026-08-21：64K 极端变长 packing

### 语义与 workload

本轮不把多条 packed sequence 当成一张 64K dense attention。输入边界固定为
70B GQA/CP8 的 `(M,N,K,L)=(8192,10240,8192,1)`；PV 边界对每条序列构造真实
`(S_i,128,S_i)` BMM，local heads=8，单卡 FLOPs 精确为
`2*sum(S_i^2)*128*8`。六组正式分布均含 65536 个有效 token：

- geometric：`32768/16384/8192/4096/2048/1024/512/256/128/128`；
- single-heavy：`49152 + 4x4096`；
- bimodal：`2x16384 + 16x2048`；
- long-tail：`32768 + 256x128`；
- irregular：`24576/12288/7168/5120/4096/3072/2048/1536/1024/512/4096`；
- uniform control：`8x8192`。

所有正式结果是 10 warmup + 50 measured iterations，每个 sample 取 8 rank
CUDA-event 最大值。JSON 现在显式记录 warmup、iterations、packed rows 与 policy。

### packed A2A 到 projection

`UlyssesRoute` 增加 `packed_source_row` 与 `packed_row_granularity`。通信 CTA 根据
packed map 直接从 peer token-major 输入搬到 GEMM staging；bulk G2S 允许一个
task 跨 sequence segment，完成全部 segment 后才 S2G/publish ready。benchmark
新增与“完整 bulk route + 同 policy CUTLASS GEMM”的 bitwise 对照，六组 CP8 和
额外 CP2/4/8 全部为 0 mismatch。

失败 A/B：packed rows 32/64 会把 BF16 128 KiB staging 的独立 slot 从 4 降为
2/1。geometric comm8 quick 的 route 从 rows16 约 `1.05 ms` 退到 rows32
`1.16 ms`、rows64 `1.66 ms`。最终固定 rows16，comm sweep 的六组赢家一致为 8。
正式 fused 为 `2.1380-2.1545 ms / 637.9-642.8 T`，fastest pure 为
`2.0022-2.0079 ms / 684.5-686.5 T`；fused 吞吐为 pure 的 `93.1%-93.8%`。
TE/cuBLAS+packing+NCCL 正式为 `2.7488-2.7725 ms`，fused 耗时为其
`77.4%-78.1%`，即 `1.281x-1.292x`。

### grouped PV 到 packed A2A

新增 pointer-array grouped CUTLASS kernel、host-built persistent work list、
group epilogue epoch publish 与 `PackedPvOutputComm`。一个 cooperative grid 的前缀
CTA 等待真实 `(sequence,head,M-tile)` ready 后，直接 scatter 到目标
rank/token/head；其余 CTA 执行 grouped BMM。work list 按 sequence K 成本降序并
循环分配给 workers，避免 packing 的输入顺序造成 static persistent load imbalance。
同一 geometric multiset 的降序/升序/交错正式值为
`6.7827/6.7727/6.7450 ms`，差异小于 0.6%，均为 0 mismatch。

Pingpong/Cooperative 与 comm CTA 宽扫表明通信配额必须看
`route_time / sum(S_i^2)`：最终 geometric/single-heavy/bimodal/long-tail/
irregular/uniform 分别使用 16/16/24/20/20/20 个通信 CTA。bimodal quick 曾在
comm40 出现 `2.94 ms` 短样本；三次 10+50 复测 comm40 为
`3.24-3.27 ms`，comm24 为 `3.11-3.13 ms`，正式采用独立 comm24 重跑
`3.1307 ms`，没有用短程最低值。

正式 PV fused：geometric `6.7827 ms`、single-heavy `11.4151 ms`、bimodal
`3.1307 ms`、long-tail `5.6454 ms`、irregular `4.2197 ms`、uniform
`2.5651 ms`。fused 吞吐为 fastest pure 的 `82.6%-89.8%`。外部基线已从
逐 `(destination,sequence)` copy 改为一次预计算索引、单次 `index_select` pack、
NCCL A2A、一次输出整理；另外用“C++ fastest pure + 实测 packed NCCL route”
构造更偏向 baseline 的 component sum，避免 257 段 long-tail 的 Python launch
开销夸大收益。相对该强基线，PV 单边为 `0.986x-1.278x`；single-heavy 是唯一
低于 1 的 case，因为 97.3% FLOPs 集中在一条 49152 序列，通信不足以覆盖 grouped
GEMM 对 cuBLAS 的差距。

### 两个融合边界合计

前后边界相加后，相对更强 component baseline：

| packing | fused total ms | strongest separated ms | fused 耗时占比 | speedup |
|---|---:|---:|---:|---:|
| geometric | 8.9372 | 9.6811 | 92.3% | 1.083x |
| single-heavy | 13.5639 | 14.0200 | 96.7% | 1.034x |
| bimodal | 5.2822 | 6.3589 | 83.1% | 1.204x |
| long-tail | 7.7854 | 8.4827 | 91.8% | 1.090x |
| irregular | 6.3577 | 7.4615 | 85.2% | 1.174x |
| uniform | 4.7078 | 6.0477 | 77.8% | 1.285x |

完整口径、分方向表和文件索引见 `PACKED64K_REPORT.md`。正式数据位于
`results/packed64k_formal/`，quick/policy sweep 位于
`results/packed64k_sweep/`。packed input 与 grouped PV 的 CP2 racecheck 均为
`0 hazards / 0 errors / 0 warnings`；CP2/4/8 exact 全部为 0 mismatch。

## 2026-08-23：正确 Ulysses 前向的小 M QKV GEMM 审计

正确前向 shape 为 QKV projection `(M,N,K,L)=(512,10240,8192,1)`，随后只路由
Q/K；V 留在 projection 输出，供后续 `V A2A→PV GEMM` 使用。正式 10+50 数据中，
最初 `AlongN` 正式数据中，QKV 的相同 policy CUTLASS 为
`0.1903 ms / 451.4 T`，compute-subgrid 为 `0.1753 ms / 490.1 T`，自动调优
cuBLASLt 为 `0.1457 ms / 589.8 T`。PV 的相同
policy CUTLASS 为 `0.0842 ms / 408.1 T`，已略快于 cuBLAS/cuBLASLt；纯 GEMM
缺口集中在 QKV。

本地 Flux profiler 对同一 QKV shape 的候选排序为：Stream-K
`128x256x64 Cooperative cluster(2,1)` 约 `0.153 ms`；最佳非 Stream-K 为
`128x128x64 Pingpong cluster(1,2)` 约 `0.166 ms`。Stream-K 会让多个 CTA 共同
产生一个输出 tile，在最终归约前不能发布 Q/K ready，不能直接复用当前
one-CTA/one-tile 通信协议。

实际接入 `cluster(1,2)` 后，CP2/4/8 exact 全部 0 mismatch。纯 GEMM 改善到约
`0.182-0.184 ms`，但通信 CTA sweep `8/12/16/20/24/28` 的 fused 最好只有约
`0.204 ms`，低于原 `cluster(1,1), comm24` 的正式 `0.1880 ms`。相邻 N tile 的
A multicast 只改善纯计算，cluster 内两 CTA 的共同前进又放大了融合场景的
TMA/调度争用。该候选及为它增加的二维 monolithic grid 支持已完整回退，生产版本
继续使用 `128x128x64 Pingpong cluster(1,1)`。

性能报告改为两个独立口径：融合收益只用相同 CUTLASS collective、tile policy、
raster 与 route 的 separated/fused A/B；cuBLASLt+NCCL 与 TE 只报告绝对墙钟，
不用于归因融合收益。当前相同 policy 联合路径为 `0.4318→0.3544 ms`，融合时间是
分离版的 `82.1%`，延迟减少 `17.9%`。TE 未做 shape/kernel/NCCL sweep，只保留为
标准实现参考。

随后固定生产 kernel 为已验证的 `cluster(1,1)`，扫描
`AlongM/AlongN × swizzle 1/2/4/8`。`AlongM + swizzle1` 胜出：正式 10+50 的
same-policy/compute-subgrid/route/fused 分别为
`0.1698/0.1659/0.0705/0.1888 ms`；相同 policy sequential 为 `0.2464 ms`。
`AlongN + swizzle1` 的 quick fused 约 `0.211 ms`，swizzle4/8 进一步退到约
`0.238 ms`。原因是 consumer task 必须跟随真实 producer raster；沿 M 让同一 N tile
的行块更早连续完成，route CTA 不会先占住晚到的 ready。

联合 benchmark 原来硬编码 `AlongN`，现已改为：CUTLASS separated/fused 都用
`AlongM`，保证受控 A/B；cuBLASLt 完成整张输出后没有 producer-order 约束，其
standalone route 独立使用更快的 `AlongN`，不削弱外部基线。新正式联合结果为
same-policy separated `0.4056 ms`、cuBLASLt+custom routes `0.3845 ms`、fused
`0.3189 ms`，四个 linked exact mismatch 均为0。受控融合时间为分离版的
`78.6%`，联合延迟减少 `21.4%`。

同一 shape 又测试 `128x128x64 Cooperative cluster(1,1)`，保持 one-CTA/one-tile
协议不变。quick exact PASS；pure/fused 为 `0.1714/0.1895 ms`，均略慢于
Pingpong 的正式 `0.1698/0.1888 ms`。该候选已回退。

## 2026-08-24：PV 最后一组 WG A/B 与角色资源审计

按约定，PV 只再完成最后一个 WG 候选，随后停止扩展性能 policy。候选为
`256x128x64 Cooperative cluster(2,1)`，其 3+20 quick 的 same-policy pure 为
`0.08264 ms / 415.8 T`，fused 为 `0.09573 ms / 358.9 T`；当前生产
`128x128x64 Pingpong cluster(2,1)` 的对应 quick 为约
`0.0816/0.0927 ms`。大 M cooperative 没有改善纯 GEMM，又使 fused 慢约 3.3%，
已完整回退。数据文件为
`results/pv_rowmajor_m256n128_cooperative_cluster2_ready128_tma_q4_3w20i.json`；
回退复测为 `results/pv_current_k64_restored_q4_3w20i.json`。

为直接观察同一 persistent grid 中通信/计算角色的资源互补，新增一个可独立删除的
诊断实例 `RoleTelemetryKernel<A2AGemmKernel>`。正式入口与正式 kernel 参数完全不变；
只有显式指定 `--role-telemetry` 时，才启动诊断实例，每个 CTA 写一次 `%globaltimer`
开始时间，并在全 CTA 汇合后写结束时间。生产计时样本不含这些写入和额外 barrier。

典型 PV shape 为 `(M,N,K,L)=(4096,128,4096,8), CP=8, comm=4`。3+12 性能
复测为 pure CUTLASS `0.0825 ms / 416.3 T`、route `0.0218 ms`、sequential
`0.1050 ms`、fused `0.0930 ms / 369.6 T`，与回退前的最优结果一致。静态资源查询与
诊断 codegen 均得到 168 registers/thread，说明这个只运行一次的 wrapper 没有提高
寄存器档位：

| role | CTA 数 | 活跃/分配 warp | 分配 registers/CTA | 分配 dynamic SMEM | 实际角色 staging |
|---|---:|---:|---:|---:|---:|
| communication | 4 | 4/12 | 64,512 | 201.0 KiB | 128.0 KiB |
| compute | 128 | 12/12 | 64,512 | 201.0 KiB | 201.0 KiB |

这是一个 monolithic kernel 的两类 CTA，编译器和硬件按同一个重型 block envelope
收费。通信 CTA 虽只执行 4 个 warp，也仍占 384 threads、64,512 个寄存器和完整
201 KiB dynamic SMEM；空出的寄存器/SMEM不能转给别的 SM。互补发生在整卡层面：
4 个 SM 主要推进 TMA/NVLink，另外 128 个 SM 推进 WGMMA。

单次设备时间线如下。`comm_overlapped` 表示通信角色活动区间中，同时存在计算角色的
比例；各 GPU 都是 100%。

| rank | comm span us | compute span us | overlap us | compute tail after comm us |
|---:|---:|---:|---:|---:|
| 0 | 13.18 | 75.52 | 13.18 | 62.30 |
| 1 | 14.94 | 81.60 | 14.94 | 66.62 |
| 2 | 12.61 | 78.72 | 12.61 | 66.08 |
| 3 | 15.04 | 82.78 | 15.04 | 67.71 |
| 4 | 13.34 | 75.84 | 13.34 | 62.37 |
| 5 | 13.57 | 79.97 | 13.57 | 66.37 |
| 6 | 16.80 | 80.96 | 16.80 | 64.13 |
| 7 | 14.27 | 77.66 | 14.27 | 63.36 |

因此当前 PV 的剩余差距已经不能归因于“通信 CTA 结束太晚”：通信活动期全部藏在
计算活动期内，最后仍有 62--68 us 的计算尾巴。由 standalone route/pure/fused
三项墙钟反推的 52.1% overlap 会混入两次独立 launch 的固定成本，也无法看见角色
何时结束；这次 per-CTA 时间线是更直接的证据。下一步若恢复性能探索，应针对融合
环境中的 WGMMA/TMA 争用、ready 等待给 compute CTA 带来的效率变化和 launch 固定
成本，而无需继续增加通信 CTA 或另换 WG policy。
