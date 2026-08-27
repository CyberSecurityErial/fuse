# Benchmark Index

当前版本：v7.0

从 v4.0 开始，每个通算融合算子独立维护 benchmark、调优空间、正式结果与复现命令：

- [A2A + O-projection](benchmarks/a2a+Oproj/BENCHMARK.md)
- [QKV Projection + A2A](benchmarks/QKVproj+a2a/BENCHMARK.md)

两份文档使用同一口径：BF16、10 次 warmup + 50 次正式采样、逐样本先取跨 rank 最大延迟，再报告 p50/p95；eager 与 CUDA Graph 分别采样、分别成列。Graph capture、instantiate 与显式 upload 均在正式采样外。QKV 正式数据固定 MPI 一进程一卡，单进程多卡只用于诊断。TE、经典 cuBLAS、cuBLASLt、NCCL 和适配版 TE Userbuffers 的调优方法与 winner 配置都在对应文档中逐点列出。

v7.0的QKV结果额外保存policy请求值、实际`BM/BN/cluster`、联合流水模型版本、
经典cuBLAS吞吐以及融合吞吐相对cuBLAS的百分比，避免只凭`auto`无法复核运行时选择。

版本级改动与历史结果见 [VERSION_HISTORY.md](VERSION_HISTORY.md)。
