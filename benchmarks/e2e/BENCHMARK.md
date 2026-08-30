# Four-operator E2E training benchmark

当前版本：v11.2

这组测试回答一个直接问题：QKV Projection、Output Projection 的前向和反向
四条融合路径接入真实训练循环后，完整训练 step 是否仍然更快。

## 对比边界

原生侧和融合侧使用同一个 Megatron Core v0.16.1 训练入口、同一份随机初始化、
同一批 mock token、相同优化器、全量激活重计算、BF16 和同一卡组。两边都启用
`full_iteration` CUDA Graph。融合侧只替换以下四个边界：

- QKV Projection + Head→Sequence A2A 前向；
- A2A + Output Projection 前向；
- QKV Projection 反向；
- Output Projection 反向。

测试关闭了 checkpoint 前向复用、attention 状态复用、额外 WGrad stream overlap 等
框架级加速，避免把它们算作四个算子的收益。Qwen2.5 7B geometry 的 128K 点为了
装入显存，融合侧复用不同层之间不会同时使用的临时通信工作区；这只减少临时显存，
不跳过计算，也不改变数据依赖。

## 测量方法

- Transformer Engine：`a7aec214eb5c3969984a40c3accb6d66987d8f25`；
- 融合扩展 SHA256：
  `7c945ed1ed496beac77ce932ac5b9847a7bbd1f5679d53f753757f4be70eb886`；
- 每次运行 10 个完整训练 step；第 4 步完成 Graph capture，统计第 5–10 步；
- 表中时间是框架打印的完整 step 时间，包含前向、重计算、反向和优化器；
- 加速比定义为 `原生时间 / 融合时间`，提升百分比为 `加速比 - 1`；
- CP8 使用 GPU 0–7，CP4 使用 GPU 0–3。GPU 1、3、6 的 SM 锁在
  1500 MHz，其余为 1980 MHz；所有 GPU 的显存频率均为 3201 MHz；
- 所有有效运行均为 0 次 skipped iteration、0 次 NaN iteration。

Nanbeige 128K 原生第一次采样期间出现了一个外部 `bwtest` 进程，该次结果不进入
发布表；v11.2 使用随后独占卡组的完整复测。

## 结果

| 模型 | CP | 全局序列 | 原生 Graph | 四算子融合 Graph | E2E 提升 |
|---|---:|---:|---:|---:|---:|
| Nanbeige4.2-3B | 8 | 1K | 125.917 ms | 124.017 ms | 1.53% |
| Nanbeige4.2-3B | 8 | 4K | 159.133 ms | 156.033 ms | 1.99% |
| Nanbeige4.2-3B | 8 | 16K | 343.083 ms | 329.067 ms | 4.26% |
| Nanbeige4.2-3B | 8 | 64K | 1979.667 ms | 1928.633 ms | 2.65% |
| Nanbeige4.2-3B | 8 | 128K | 6415.000 ms | 6305.967 ms | 1.73% |
| Llama-3 8B geometry | 8 | 1K | 229.967 ms | 227.283 ms | 1.18% |
| Llama-3 8B geometry | 8 | 4K | 281.883 ms | 274.617 ms | 2.65% |
| Llama-3 8B geometry | 8 | 16K | 535.017 ms | 519.433 ms | 3.00% |
| Llama-3 8B geometry | 8 | 64K | 2425.250 ms | 2378.700 ms | 1.96% |
| Llama-3 8B geometry | 8 | 128K | 7149.900 ms | 7059.483 ms | 1.28% |
| Qwen2.5 7B geometry | 4 | 1K | 229.100 ms | 226.833 ms | 1.00% |
| Qwen2.5 7B geometry | 4 | 4K | 319.967 ms | 313.933 ms | 1.92% |
| Qwen2.5 7B geometry | 4 | 16K | 772.583 ms | 749.933 ms | 3.02% |
| Qwen2.5 7B geometry | 4 | 64K | 3837.400 ms | 3763.633 ms | 1.96% |
| Qwen2.5 7B geometry | 4 | 128K | 11311.883 ms | 11176.400 ms | 1.21% |

15 个 setting 全部由融合侧获胜。Nanbeige、Llama-3 8B geometry、Qwen2.5 7B
geometry 的五点几何平均提升分别为 2.43%、2.01% 和 1.82%；全部 15 点的几何
平均提升为 2.09%。最大提升是 Nanbeige 16K 的 4.26%。

这里的 Llama 和 Qwen 条目表示完整层数、hidden/FFN、Q/KV heads 和词表规模都
对应目标模型的训练 geometry；测试使用随机初始化和 mock token，不声称复现真实
checkpoint 的收敛曲线。权重数值不会改变本次要比较的 CUDA 执行路径。
