# Benchmark Index

从 v4.0 开始，每个通算融合算子独立维护 benchmark、调优空间、正式结果与复现命令：

- [A2A + O-projection](benchmarks/a2a+Oproj/BENCHMARK.md)
- [QKV Projection + A2A](benchmarks/QKVproj+a2a/BENCHMARK.md)

两份文档使用同一口径：BF16、10 次 warmup + 50 次正式采样、逐样本先取跨 rank 最大延迟，再报告 p50/p95。TE、经典 cuBLAS、cuBLASLt、NCCL 和适配版 TE Userbuffers 的调优方法与 winner 配置都在对应文档中逐点列出。

版本级改动与历史结果见 [VERSION_HISTORY.md](VERSION_HISTORY.md)。
