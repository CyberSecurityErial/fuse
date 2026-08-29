# Benchmark Index

当前版本：v9.0

从 v4.0 开始，每个通算融合算子独立维护 benchmark、调优空间、正式结果与复现命令：

- [A2A + O-projection](benchmarks/a2a+Oproj/BENCHMARK.md)
- [QKV Projection + A2A](benchmarks/QKVproj+a2a/BENCHMARK.md)
- [锁频异构 CP（v9.0 实验特性）](benchmarks/heterogeneous_cp/BENCHMARK.md)

两个均匀算子文档使用同一口径：BF16、10 次 warmup + 50 次正式采样、逐样本先取跨 rank 最大延迟，再报告 p50/p95；eager 与 CUDA Graph 分别采样、分别成列。Graph capture、instantiate 与显式 upload 均在正式采样外。QKV 正式数据固定 MPI 一进程一卡，单进程多卡只用于诊断。TE、经典 cuBLAS、cuBLASLt、NCCL 和适配版 TE Userbuffers 的调优方法与 winner 配置都在对应文档中逐点列出。

v9.0 的锁频异构 CP 是独立实验路径：调用方提供每张卡的稳定资源能力，规划器改变连续 token ownership，并联合选择行数、通信 CTA 和 tile。其正式矩阵使用 5 次 warmup + 30 次采样；它不替换或改写 v8 的两条均匀算子 Golden。

v8.0 的 QKV 结果额外保存自动请求值、最终通信 CTA、实际 `BM/BN/cluster`、
流水模型版本、自动方案缓存版本、经典 cuBLAS 吞吐以及融合吞吐相对 cuBLAS 的百分比。正式 runner
把 `comm_ctas=0` 原样传入算子，因此也覆盖自动配置缓存，而不是提前替库固定配置。

v9.0 的异构归档保存 CP、慢卡位置、资源比例、连续行分区、最终通信 CTA、uniform/weighted p50 和逐点加速比；见 [`results/heterogeneous-cp/locked_frequency_summary.csv`](results/heterogeneous-cp/locked_frequency_summary.csv)。

版本级改动与历史结果见 [VERSION_HISTORY.md](VERSION_HISTORY.md)。
