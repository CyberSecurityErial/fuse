# Ulysses 前向融合边界报告

## 测试口径

硬件为单机 8 卡，BF16，CP=8。模型参数为 `B=1, S=4096, hidden=8192,
Hq=64, Hkv=8, D=128`。对应两个 GEMM：

| 边界 | GEMM `(M,N,K,L)` | 通信 |
|---|---:|---|
| QKV projection → Q/K A2A | `(512,10240,8192,1)` | Q/K 从 sequence shard 转为 head shard |
| V A2A → PV | `(4096,128,4096,8)` | 每个 rank 拉取一个 KV head，8 个 Q head 共享 RHS |

所有正式数据均为 10 次预热、50 次测量。每个样本取 8 卡最大 CUDA event
时间。Q/K、V、TE source Q/K/V 重排和 PV 输出均做 exact correctness，mismatch
为 0。测试只覆盖两
个融合边界，QK、softmax 和模型其他层不在计时范围内。

Transformer Engine 源码版本固定为
`a7aec214eb5c3969984a40c3accb6d66987d8f25`。外部强基线已经单独完成
NCCL channel/chunk/protocol、pack kernel、unpack、CUDA Graph和stream优先级搜索，
固定口径见 `FROZEN_TE_CUBLAS_NCCL_BASELINE.md`。TE 有两种对照：

- frozen matched：TE Linear 后合并 Q/K 做一次 NCCL A2A，V 在 PV 前单独 A2A；
  它与本实现的数据依赖和通信量一致，并使用调优后的 Triton pack、零拷贝 unpack、
  CUDA Graph与NCCL配置。
- source：直接调用 TE `flash_attn_a2a_communicate`，按源码为 Q、K、V
  分别建立 contiguous send buffer、receive buffer和异步 A2A。

## 正式结果

### 受控实验：相同 CUTLASS GEMM

融合版与分离版复用完全相同的 CUTLASS mainloop、tile policy、raster 和数值类型。
分离版依次执行 GEMM 与同一份 route 代码，融合版只改变 CTA 分工、tile ready 和
执行重叠。分离版 GEMM 可以使用整卡 SM，融合版还要给通信 CTA 留 SM，因此这个
对照不会偏袒融合版。

| 边界 | 相同 policy 分离执行 | CTA-specialized 融合 | 融合时间为分离版的 | 延迟减少 |
|---|---:|---:|---:|---:|
| QKV GEMM → Q/K A2A | 0.2464 ms | 0.1888 ms | 76.6% | 23.4% |
| V A2A → PV GEMM | 0.1422 ms | 0.1157 ms | 81.4% | 18.6% |
| 两个边界联合 | 0.4056 ms | 0.3189 ms | 78.6% | 21.4% |

这里的 `21.4%` 是当前可以直接归因给通算融合的联合收益。

### 纯 GEMM 质量

| GEMM | shape `(M,N,K,L)` | 相同 policy CUTLASS | 最快 vendor GEMM | CUTLASS 吞吐为 vendor 的 |
|---|---:|---:|---:|---:|
| QKV projection | `(512,10240,8192,1)` | 0.1698 ms | cuBLASLt 0.1465 ms | 86.3% |
| PV | `(4096,128,4096,8)` | 0.0842 ms | cuBLAS 0.0872 ms | 103.6% |

当前 GEMM 缺口集中在 QKV projection。它的真实 `M=512` 会选择
`128×128×64 Pingpong`，没有使用此前为大 M shape 调过的
`128×256×64 Cooperative`。`AlongM + swizzle1` 将同一 policy CUTLASS 从此前
`AlongN` 的约 `0.190 ms` 降到 `0.1698 ms`；QKV mainloop 仍比 cuBLASLt 慢
约 15.9%。PV policy 没有这个问题。

### 外部实现：冻结后的强基线

下面的 GEMM 实现不同，只能回答“整条实现谁更快”，不能用来计算融合收益。
表内均为三次独立 10+50 运行的 p50 中位数；本实现仍列自己的正式 10+50 p50。

| 实现 | QKV GEMM | Q/K A2A | QKV+Q/K | V A2A | PV BMM | V A2A+PV |
|---|---:|---:|---:|---:|---:|---:|
| TE + frozen NCCL | 0.1502 ms | 0.0614 ms | 0.2107 ms | 0.0277 ms | 0.0915 ms | 0.1163 ms |
| cuBLAS + frozen NCCL | 0.1510 ms | 0.0614 ms | 0.2102 ms | 0.0277 ms | 0.0915 ms | 0.1163 ms |
| 本实现 | 0.1659 ms¹ | 0.0701 ms | 0.1887 ms | 0.0864 ms¹ | 0.0454 ms | 0.1154 ms |

¹ 本实现使用给通信 CTA 预留相同 SM 数后的 compute-subgrid GEMM 时间，以免把
通信 CTA 占用的 SM 算成通信开销。

按 `hidden = compute + communication - combined` 做辅助观察：

| 实现 | QKV 边界通信被覆盖 | V/PV 边界通信被覆盖 |
|---|---:|---:|
| TE + frozen NCCL | 1.3% | 10.5% |
| cuBLAS + frozen NCCL | 3.5% | 10.5% |
| 本实现 | 67.5% | 36.2% |

TE 和 cuBLAS frozen case 是算术口径。源码依赖要求 projection 完成后才能 pack
Q/K，V A2A 完成后才能启动 BMM，因此这里没有 tile 级 GEMM/A2A 并发；小于两项
之和来自 CUDA Graph、NCCL调度、cache、max-rank合成和测量边界。真正的 tile 级
并发只存在于本实现的单 kernel producer/consumer 流水。

### 外部实现联合测量

| 联合 case | mean | p50 | p95 | 相对本实现 |
|---|---:|---:|---:|---:|
| cuBLASLt + 同一 custom routes | 0.3845 ms | 0.3844 ms | 0.3875 ms | 120.6% |
| **TE + frozen NCCL**² | **0.3128 ms** | **0.3118 ms** | **0.3180 ms** | **98.1%** |
| **cuBLAS + frozen NCCL**² | **0.3152 ms** | **0.3149 ms** | **0.3193 ms** | **98.8%** |
| TE source Q/K/V 三次 A2A | 0.9479 ms | 0.9456 ms | 0.9681 ms | 297.2% |
| 本实现双边界融合 | 0.3189 ms | 0.3192 ms | 0.3225 ms | 100.0% |

² frozen 行的 mean/p50/p95分别取三轮对应统计量的中位数；每轮包含50个max-rank
样本，150个原始样本全部保存在汇总JSON中，没有删除离群点。

TE source case 包含源码中的三次 A2A 和重排，只用于说明当前 TE CP 源码路径的
实际成本。冻结基线完成了有效P2P channel、128/256/512/1024 KiB chunk、LL阈值、
pack block/warp、CUDA Graph和高优先级stream搜索。当前本实现p50比TE frozen慢
约2.3%，比cuBLAS frozen慢约1.4%；后续优化必须以这两行作为外部胜负线。

## Buffer 生命周期与数据依赖

TE 在 attention 前将 Q、K、V 依次转换为
`[CP,B,S_local,H_local,D]` contiguous send buffer。每个张量随后分配同尺寸
receive buffer并启动 `all_to_all_single(async_op=True)`。前一个 request 在
`cp_stream` 上 wait，receive buffer再经过 contiguous 和 sequence chunk reorder。
三个结果都完成后，当前计算 stream执行 `wait_stream(cp_stream)`，attention才能读取
Q、K、V。

因此 TE 的生命周期是：

```text
local Q/K/V
  → contiguous send buffers
  → NCCL receive buffers
  → reordered attention buffers
  → attention 完成后释放
```

其依赖为：

```text
QKV projection 完成
  → Q/K/V pack
  → 三次 A2A 与各自 reorder
  → current stream 等 cp_stream
  → attention
```

本实现把生命周期改为：

```text
QKV GEMM tile
  ├─ Q/K：tile ready 后直接写目标 rank，保留到 QK/softmax 消费
  └─ V：留在源 rank 的 QKV output，保留到所有 peer 完成 V pull

V peer shard
  → 直接转置到一个共享 RHS [128,4096]
  → peer-ready 后 PV 开始对应 K 区间
  → A2AGemm 完成后 RHS staging 可复用
```

典型 CP8/GQA shape 中，每个 rank 只有一个 KV head，8 个本地 Q head共享同一个
V。`stride_b.batch=0` 让 8 个 GEMM batch直接读取同一份 RHS，避免将 V staging
复制 8 次。

## 基于生命周期的后续优化

1. **压缩长期存活的 V。** 当前完整 10 MiB local QKV output要一直活到 V pull
   完成，真正需要长期保留的 V 只有 1 MiB。可在 projection 完成时把 V 写入紧凑
   buffer，Q/K tile完成远端发送后即可释放或循环复用其空间。
2. **删除全 rank 的阶段交接。** 当前联合 C++ case在两个 kernel间显式等待所有
   rank。可由每个 rank在 QKV 完成时发布 V epoch，A2AGemm按 peer独立等待，避免
   最慢 rank阻塞所有 peer。独立融合 kernel之和约0.305 ms，联合 case为0.319 ms，
   目前约14 us是交接优化空间。
3. **利用 QK/softmax 的空档预取 V。** V 在 softmax前不会参与计算，可以在单独
   stream或 persistent communication CTA中提前填充下一阶段共享 RHS。需要与当前
   just-in-time A2AGemm做完整 attention E2E A/B，选择更短关键路径的方案。
4. **继续补齐 QKV GEMM。** `AlongM` 已经收回约20 us。下一步候选必须同时测
   pure、compute-subgrid和fused；`cluster(1,2)` 虽改善 pure，却使 fused退化，已回退。
5. **继续优化 A2AGemm尾部。** QKV 边界覆盖67.5%的通信；V/PV只有35.6%。
   下一轮围绕更早的 peer ready、减少最后一个 peer后的 GEMM tail和消除 ready
   polling开销。

数据文件：

- `results/correct_qkv_gemm_a2a_alongm_cp8_s4096_bf16_comm24_10w50i.json`
- `results/correct_a2a_gemm_shared_rhs_cp8_s4096_bf16_comm4_10w50i.json`
- `results/correct_ulysses_forward_alongm_shared_rhs_cp8_s4096_bf16_10w50i.json`
- `results/correct_ulysses_forward_te_source_and_matched_cp8_s4096_bf16_10w50i.json`
- `results/correct_te_source_qkv_exact_smoke.json`
- `results/te_cublas_nccl_frozen_graph_triton1024_highprio_3x10w50i_summary.json`
