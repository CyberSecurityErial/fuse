# Benchmark Index

当前版本：v13.1

从 v4.0 开始，每个通算融合算子独立维护 benchmark、调优空间、正式结果与复现命令：

- [A2A + O-projection](benchmarks/a2a+Oproj/BENCHMARK.md)
- [QKV Projection + A2A](benchmarks/QKVproj+a2a/BENCHMARK.md)
- [锁频异构 CP（v9.1 实验特性）](benchmarks/heterogeneous_cp/BENCHMARK.md)
- [QKV Projection backward](benchmarks/QKVproj-backward/BENCHMARK.md)
- [Output Projection backward](benchmarks/Oproj-backward/BENCHMARK.md)
- [四算子完整训练 E2E](benchmarks/e2e/BENCHMARK.md)
- [四算子 FP8](benchmarks/fp8/BENCHMARK.md)

两个均匀算子文档使用同一口径：BF16、10 次 warmup + 50 次正式采样、逐样本先取跨 rank 最大延迟，再报告 p50/p95；eager 与 CUDA Graph 分别采样、分别成列。Graph capture、instantiate 与显式 upload 均在正式采样外。QKV 正式数据固定 MPI 一进程一卡，单进程多卡只用于诊断。TE、经典 cuBLAS、cuBLASLt、NCCL 和适配版 TE Userbuffers 的调优方法与 winner 配置都在对应文档中逐点列出。

v9.0 引入锁频异构 CP 独立路径；v9.1 增加跨 shape 的功耗安全回退。调用方提供每张卡的稳定资源能力，规划器改变连续 token ownership，并联合选择行数、通信 CTA 和 tile。其正式矩阵使用 5 次 warmup + 30 次采样，不替换或改写 v8 的两条均匀算子 Golden。v9.1 补测五种真实模型宽度和原 v9 控制 shape 共 18 点；会让参考卡自身撞功耗墙的 shape 默认回退 uniform，避免误用标称频率比。

v8.0 的 QKV 结果额外保存自动请求值、最终通信 CTA、实际 `BM/BN/cluster`、
流水模型版本、自动方案缓存版本、经典 cuBLAS 吞吐以及融合吞吐相对 cuBLAS 的百分比。正式 runner
把 `comm_ctas=0` 原样传入算子，因此也覆盖自动配置缓存，而不是提前替库固定配置。

v9.0 的原始异构矩阵见 [`results/heterogeneous-cp/locked_frequency_summary.csv`](results/heterogeneous-cp/locked_frequency_summary.csv)；v9.1 的跨 shape 复核见 [`results/heterogeneous-cp/shape_generalization_summary.csv`](results/heterogeneous-cp/shape_generalization_summary.csv)。两份归档都保存 CP、资源配置、uniform/weighted p50 和逐点加速比。

v10.0 的两条反向算子各覆盖三个人工 shape、五组真实模型、六档序列长度和
CP4/CP8，共 96 个 setting。每点分别报告 Eager/Graph 与普通同流 B→W/
ZeroBubble 分离 B/W；普通 W 使用 `beta=0`，ZeroBubble W 使用 `beta=1` 直接累加
BF16 `main_grad`。raw 与汇总都保存真实 B/W MNK，正式数据为 MPI 一进程一卡、
10 warmup + 50 samples、逐样本 max-rank p50/p95。经典 cuBLAS 使用同卡组、同
MNK 和匹配 beta，只测纯 GEMM；它不包含 A2A。

反向总吞吐相对同 shape 前向融合吞吐的几何平均，在 QKV 四种口径中为
`98.0%～104.3%`，在 OProj 中为 `99.6%～104.6%`。完整的 96 行普通表、96 行
ZeroBubble 表、TFLOPS、989T MFU 和经典 cuBLAS 占比均放在各自算子文档与
`results/QKVproj-backward`、`results/Oproj-backward`。

反向外部主对照使用适配版 TE Userbuffers：每个 setting 对结构参数做搜索，短采样
前三名统一进行 10+50 正式复测。QKV 的 Eager 普通/ZeroBubble 几何平均分别领先
`1.463×/1.475×`，Graph 为 `1.222×/1.216×`；OProj 对应为
`1.428×/1.430×` 与 `1.146×/1.146×`。轻量 TE+NCCL 另取两类 shape、三档序列、
CP4/CP8 共 12 点，只用于证明分离基线口径和复现链路，不作为全量调参结论。

v11.2 把 QKV/OProj 的前向和反向四条融合边界接入完整 Megatron 训练，并与同配置
原生 TE 做 full-iteration CUDA Graph 对照。Nanbeige、Llama-3 8B geometry、
Qwen2.5 7B geometry 各覆盖 1K/4K/16K/64K/128K，共 15 个 setting，融合侧
`15/15` 获胜，完整 step 吞吐几何平均提升 `2.09%`，最大提升 `4.26%`。逐点
时间和测试边界见 [`四算子 E2E benchmark`](benchmarks/e2e/BENCHMARK.md)。

v12.0 给同样四条边界增加纯 E4M3 路径。输入、权重、通信数据和输出均为 E4M3，
Tensor Core 内部使用 FP32 累加；量化、amax 和 scale 的生成由调用方负责，不计入
本轮 kernel 延迟。正式 QKV 使用 MPI 一进程一卡、10+50 和预上传 CUDA Graph；
自动策略只读取 shape、SM 数、通信量和 FP8 wave 标定，不读取模型名或逐点赢家。
接口、正确性和复现方式见 [`四算子 FP8 benchmark`](benchmarks/fp8/BENCHMARK.md)。

v13.0/v13.1 是纯结构重构，不生成新的性能结果，也不改写任何历史 Golden。重构版与
v12.0 干净 Release 基线的 69 个 device function 逐函数 SASS 完全一致，58 个公开
符号完全一致；Release/profiling 构建以及 BF16/FP8 前向、反向 smoke 均通过。
因此本页所有性能数字继续使用对应算子的已发布归档。

版本级改动与历史结果见 [VERSION_HISTORY.md](VERSION_HISTORY.md)。
