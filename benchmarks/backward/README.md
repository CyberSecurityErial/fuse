# Backward benchmark harness

这个目录只放两条反向算子共用的测试脚手架，不是第三个生产算子。

- `backward_smoke.cu` 用独立参考实现检查 QKV 与 OProj 的 B/W 数学、跨卡布局、普通同流模式、ZeroBubble 分离模式和梯度累加；
- `backward_mpi_bench.cu` 负责一进程一卡的 Eager/Graph 计时与 CUDA IPC；
- `backward_cublas_mpi_bench.cu` 用相同卡组和反向 MNK 测两次经典 cuBLAS 纯 GEMM；
- `backward_shape_bench.py` 枚举相同的模型、序列长度和 CP 矩阵，并分别生成两个算子的结果。

每份正式 raw 都直接记录 B/W MNK；聚合器会重新核对 shape、50 个有限样本、
普通模式的 `beta=0`、ZeroBubble 的 `beta=1`、自动请求和最终 tile。发布目录只保留
96 行 JSON/CSV/Markdown 汇总，逐次 MPI raw 留在用户指定的临时结果目录。

生产接口和参数结构仍完全分开：`qkv_backward.h` 不依赖 OProj 参数，`oproj_backward.h` 也不依赖 QKV 参数。共用 runner 只用于保证两边使用同一采样口径，不能把其中一个算子的 shape 或策略套到另一个算子。

完整语义、反向 MNK、复现命令和正式结果分别见：

- [QKV Projection backward](../QKVproj-backward/BENCHMARK.md)
- [Output Projection backward](../Oproj-backward/BENCHMARK.md)
