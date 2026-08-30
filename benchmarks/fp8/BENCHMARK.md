# FP8 四算子 Benchmark

当前版本：v12.0

## 算子与数据口径

v12.0 为下面四条已有融合边界增加纯 E4M3 实现：

- `QKV Projection -> A2A`；
- `A2A -> OProj`；
- `QKV backward` 的 B/W 两阶段；
- `OProj backward` 的 B/W 两阶段。

输入、权重、通信数据和输出都已经是 E4M3，Tensor Core 使用 FP32 累加，`alpha`
在结果写回 E4M3 前生效。算子不计算 amax、scale，也不执行 BF16→FP8 量化；调用方
负责准备量化张量和组合后的 `alpha`。因此这里报告的是四条投影/通信边界本身，
不是完整混合精度训练 step。

普通反向在同一 stream 内执行 B→W，并用 `beta=0` 写权重梯度；ZeroBubble 仍把
B/W 分成两个入口，W 可用 `beta=1` 累加。调用方必须保留 W 所需的两个输入，直到
延迟的 W 完成。v12 不改变 BF16 接口和布局。

## 正式计时规则

- H200/SM90a，profiling 关闭；
- QKV 使用 MPI 一进程一卡，CP4/CP8；
- 3 个人工宽度和 5 组真实模型几何，6 档全局序列，共 96 个 setting；
- 10 次 warmup + 50 次正式采样；
- 每个样本先取跨 rank 最大 CUDA Event 时间，再计算 p50/p95；
- CUDA Graph 在采样前完成 capture、instantiate 和 upload，正式 graph 只 replay 一次，
  graph 内每个 epoch 单调递增；
- 正式自动入口固定 `comm_ctas=0`，不设置手工 tile 或流水环境变量。

QKV 的自动模型在 N64/N128/N256 中联合选择 tile 与通信 CTA。评分只读取 M/N/K、
CP/head 几何、SM 数、通信量和 H200 FP8 wave 标定，不读取模型名、BF16 结果或逐点
赢家。N64 的 ready 发布粒度更细；超过标定覆盖的 8 个物理 wave 时自动回到更宽
tile，避免把短点结论外推到长序列。

## 性能结果

最终完整 96 点 CUDA Graph 中，FP8 相对同 shape BF16 Graph 的 p50 几何平均为
`1.763×`，`85/96` 达到 `1.5×`。这是端到端融合边界的 FLOPS 加速，不是只比较
纯 GEMM；剩余点主要是短矩阵固定通信/同步成本占比高。N64 长波边界修正后的最终
96 点结果保存在 [`results/fp8/qkv_graph_summary.md`](../../results/fp8/qkv_graph_summary.md)，
JSON/CSV 是同一张表的机器可读版本。

`fp8_smoke` 对四条路径做确定性 E4M3 对照，覆盖 routed 输出、dX、dW、普通写入和
`beta=1` 累加。参考路径和生产路径都从相同的已量化输入出发，因此能够检查算子
语义，但不替代训练框架对 scale/amax 生命周期的验证。

## 复现

```bash
cmake -S . -B build-v12 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DFUSE_ENABLE_PROFILING=OFF
cmake --build build-v12 -j

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./build-v12/fp8_smoke

unset FUSE_FP8_GEMM_TILE FUSE_FP8_GEMM_PIPELINE
python3 benchmarks/QKVproj+a2a/qkv_shape_bench.py \
  --phase fuse-mpi-formal --phase fuse-mpi-aggregate \
  --precision fp8 --mpi-launches both --mpi-auto-comm \
  --formal-warmup 10 --formal-iters 50 \
  --mpi-bench build-v12/qkvproj_a2a_mpi_bench \
  --results results/reproduce_fp8_qkv --no-resume

python3 benchmarks/fp8/oproj_fp8_shape_bench.py \
  --bench build-v12/fuse_bench \
  --results results/reproduce_fp8_oproj --no-resume
```

FP8 profile 沿用 [`PROFILE_PROTOCOL.md`](../../PROFILE_PROTOCOL.md) 的条带定义；
profile 构建只能解释 compute/route/finalize 时序，不能写进正式性能表。
