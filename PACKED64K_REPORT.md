# 64K 变长 packing：A2A/GEMM 双边融合报告

## 测试口径

- 模型参数按 70B GQA：hidden `8192`、Q heads `64`、KV heads `8`、head dim
  `128`、QKV width `10240`。
- CP=8，单次 packing 的有效 token 总数固定为 `65536`。
- 输入边界是 packed Ulysses `A2A -> QKV projection`。每卡 GEMM 为
  `(M,N,K,L)=(8192,10240,8192,1)`，每卡计算量 `1.374 TFLOP`，A2A payload
  `128 MiB`。
- 输出边界是每条真实序列各自的 `PV -> packed A2A`。每个 local head 的 GEMM
  为 `(S_i,128,S_i)`，local heads 为 8；没有把多条序列伪装成一张
  `65536 x 65536` dense BMM。每卡 FLOPs 为
  `2 * sum(S_i^2) * 128 * 8`。
- 正式数据统一为 10 次 warmup + 50 次 measured iteration。每个 sample 取 8 个
  rank 的 CUDA event 最大值，再统计 mean/p50/p95。
- 本机 runtime 报告 8 张 `NVIDIA L20X`、SM90、132 SM、全 NVLink；按任务约定
  对标 H200。三张卡运行时上限为 1.5 GHz，因此所有表都保留 max-rank 口径，
  不挑快卡结果。

六组正式 packing：

| 名称 | 序列长度 | 序列数 | `sum(S_i^2)` | 最长序列 FLOPs 占比 |
|---|---|---:|---:|---:|
| geometric | `32768/16384/8192/4096/2048/1024/512/256/128/128` | 10 | 1,431,666,688 | 75.0% |
| single-heavy | `49152 + 4 x 4096` | 5 | 2,483,027,968 | 97.3% |
| bimodal | `2 x 16384 + 16 x 2048` | 18 | 603,979,776 | 44.4% |
| long-tail | `32768 + 256 x 128` | 257 | 1,077,936,128 | 99.6% |
| irregular | `24576/12288/7168/5120/4096/3072/2048/1536/1024/512/4096` | 11 | 883,425,280 | 68.4% |
| uniform control | `8 x 8192` | 8 | 536,870,912 | 12.5% |

额外把 geometric 的同一组长度排成降序、升序和大小交错。PV 正式融合分别为
`6.7827/6.7727/6.7450 ms`，差异小于 0.6%，三组均为 0 mismatch。

## 实现和调优

输入方向增加 packed row map。通信 CTA 直接按 map 从各 peer 的 token-major
输入读取，写入本地 GEMM staging；不先启动单独的 pack kernel。bulk TMA 可以在
一个 task 内跨 sequence segment，ready 仍按 GEMM M tile 和 peer shard 发布。

PV 方向增加 pointer-array grouped CUTLASS GEMM 和单个 persistent grid。host 只在
plan 构造时生成 `(sequence, head, M tile)` work list；运行时 compute CTA 持久消费
work list，epilogue 发布 tile epoch，communication CTA 随即把真实 output tile
scatter 到目标 rank/token/head。work list 按 K 成本降序后循环分配给 persistent
workers，消除 packing 顺序导致的长短任务失衡。

关键搜索结论：

- packed input 的 16-row bulk task 保留 4 个独立 TMA issuer；32/64-row 会降低
  issuer 数，route 反而变慢。六组都由 `packed_rows=16, comm_ctas=8` 胜出。
- PV 的通信/计算占比随 `sum(S_i^2)` 变化，不能固定同一个通信 CTA 数。正式配置为：
  geometric 16、single-heavy 16、bimodal 24、long-tail 20、irregular 20、uniform
  20；GEMM policy 使用 Hopper cooperative pointer-array collective。
- bimodal 的短 sweep 曾把 comm40 误判成 `2.94 ms`。三次 10+50 复测显示 comm40
  稳定在 `3.24-3.27 ms`，comm24 稳定在 `3.11-3.13 ms`。正式表使用独立命名的
  comm24 重跑 `3.1307 ms`，没有挑短程最好值。

正确性：64K CP8 六组的输入/PV 均 bitwise 0 mismatch；另用较小的极端 packing
覆盖 CP=2/4/8，六项均为 0 mismatch。Compute Sanitizer racecheck 分别覆盖 packed
input 和 grouped PV，均为 `0 hazards / 0 errors / 0 warnings`。

## 正式结果

输入表中的 pure 不含通信；“标准方案”包含一次 packing、NCCL A2A、输出整理和
GEMM。最后一列的百分比是融合耗时占最快标准方案的比例。

| packing | fused ms / TFLOPS | fastest pure ms / TFLOPS | fused 吞吐为 pure 的 | TE+NCCL ms | cuBLAS+NCCL ms | fused 耗时 / 最快标准方案 |
|---|---:|---:|---:|---:|---:|---:|
| geometric | **2.1545 / 637.9** | 2.0061 / 685.1 | 93.1% | 2.7650 | 2.7639 | **78.0% / 1.283x** |
| single-heavy | **2.1488 / 639.6** | 2.0022 / 686.5 | 93.2% | 2.7721 | 2.7637 | **77.8% / 1.286x** |
| bimodal | **2.1515 / 638.8** | 2.0079 / 684.5 | 93.3% | 2.7669 | 2.7563 | **78.1% / 1.281x** |
| long-tail | **2.1401 / 642.2** | 2.0079 / 684.5 | 93.8% | 2.7488 | 2.7725 | **77.9% / 1.284x** |
| irregular | **2.1380 / 642.8** | 2.0053 / 685.4 | 93.8% | 2.7498 | 2.7615 | **77.8% / 1.286x** |
| uniform | **2.1427 / 641.4** | 2.0048 / 685.6 | 93.6% | 2.7683 | 2.7708 | **77.4% / 1.292x** |

PV 的外部标准方案使用真实 varlen cuBLAS BMM、单次 `index_select` packing、NCCL
A2A 和一次输出整理。为了避免 Python 逐 shape launch 在 257 段长尾 case 中放大
收益，另列一个更强的 component baseline：本地 C++ 最快 pure GEMM 加实测 packed
NCCL route。它是偏向 baseline 的乐观分量和，不是另一个伪装成端到端的实测值。

| packing | fused ms / TFLOPS | fastest pure ms / TFLOPS | fused 吞吐为 pure 的 | cuBLAS BMM+NCCL 实测 ms | 最强 component baseline ms | fused 耗时 / 最强 component | 相对同 policy 顺序执行 |
|---|---:|---:|---:|---:|---:|---:|---:|
| geometric | **6.7827 / 432.3** | 5.8889 / 497.9 | 86.8% | 7.1810 | 6.9173 | **98.1% / 1.020x** | 1.164x |
| single-heavy | **11.4151 / 445.5** | 10.2482 / 496.2 | 89.8% | 12.8906 | 11.2563 | 101.4% / 0.986x | 1.066x |
| bimodal | **3.1307 / 395.1** | 2.5873 / 478.1 | 82.6% | 3.5997 | 3.6026 | **86.9% / 1.151x** | 1.149x |
| long-tail | **5.6454 / 391.0** | 4.7307 / 466.7 | 83.8% | 6.5527 | 5.7339 | **98.5% / 1.016x** | 1.036x |
| irregular | **4.2197 / 428.8** | 3.6760 / 492.2 | 87.1% | 4.7006 | 4.7117 | **89.6% / 1.117x** | 1.181x |
| uniform | **2.5651 / 428.6** | 2.2663 / 485.2 | 88.4% | 3.2848 | 3.2794 | **78.2% / 1.278x** | 1.362x |

single-heavy 的 PV 单边是明确边界：相对真实 Python/cuBLAS+NCCL 仍为 1.129x，
但相对“最快 C++ pure + NCCL route”的乐观分量和只有 98.6%。该分布 97.3% 的 PV
FLOPs 集中在一条 49152 序列，通信很难覆盖 grouped GEMM 相对 cuBLAS 的差距。

把前后两个边界相加，最后一列仍使用更强的 component baseline：

| packing | 两个 fused 总 ms | 有效 TFLOPS/GPU | 外部标准方案实测总 ms | 相对外部实测 | 最强分离方案 ms | fused 耗时 / 最强分离方案 |
|---|---:|---:|---:|---:|---:|---:|
| geometric | **8.9372** | 481.9 | 9.9448 | 1.113x | 9.6811 | **92.3% / 1.083x** |
| single-heavy | **13.5639** | 476.2 | 15.6543 | 1.154x | 14.0200 | **96.7% / 1.034x** |
| bimodal | **5.2822** | 494.4 | 6.3559 | 1.203x | 6.3589 | **83.1% / 1.204x** |
| long-tail | **7.7854** | 460.1 | 9.3015 | 1.195x | 8.4827 | **91.8% / 1.090x** |
| irregular | **6.3577** | 500.8 | 7.4504 | 1.172x | 7.4615 | **85.2% / 1.174x** |
| uniform | **4.7078** | 525.5 | 6.0531 | 1.286x | 6.0477 | **77.8% / 1.285x** |

结论：64K packing 的两个边界合计，在最偏向 baseline 的比较里仍为原耗时的
`77.8%-96.7%`，对应 `1.034x-1.285x`。收益随序列平方和与路由占比变化；序列越
均匀，PV 计算越少，通信重叠价值越高。Flux 当前接口没有 ragged packed source map
或等价 grouped varlen PV 路径，因此不拿 uniform-shape Flux 数字冒充这组语义。

正式 JSON 和 stdout 位于 `results/packed64k_formal/`；CP2/4/8 exact 与 racecheck
日志位于 `results/packed64k_correctness/`，quick/CTA policy sweep 位于
`results/packed64k_sweep/`。
