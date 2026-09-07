# BF16 / FP8 / FP4 benchmark 入口与合同

检索日期：2026-09-05。独立 cuBLASLt 的 BF16、tensor-wise E4M3 FP8 和
NVFP4 已在 L20D/B300 上完成非平凡输入的数值 smoke、Eager 和 CUDA Graph
验证。分布式 Userbuffers 边界目前只定义 BF16；FP4 没有 TE Ulysses A2A
合同，故不生成伪基线。

## 一次收齐的入口

| 类别 | 源码/入口 | 用途与限制 |
|---|---|---|
| 本仓 SM103 GEMM | `GEMM/cublaslt_bench.py`、`csrc/baselines/sm103/cublaslt_training.cu` | BF16/E4M3/NVFP4；实际计时全部 heuristic 候选并保存 raw samples、算法元数据、数值误差和库哈希 |
| 本仓 BF16 QKV/A2A、A2A/OProj | 两方向的 `run_bench.py`，由 `sm103/bench.py` 调度 | 完整边界；保留 max-rank 样本计时；依赖适配版 TE |
| 本仓历史 E4M3 | [v12 FP8 说明](../sm90/fp8/BENCHMARK.md)、SM90 QKV shape runner 的 `--precision fp8` | 旧 SM90a 四算子，不能当作 SM103 产物 |
| 官方 cuBLASLt BF16 | 本仓 BF16 实现；另参考官方 cuBLASLt layout/algorithm 样例 | 不能把名字含 H 的 FP16 样例当成 BF16 测量结果 |
| 官方 cuBLASLt FP8 | [LtFp8Matmul][fp8]、[LtFp8CustomFind][fp8find] | FP8 descriptor/scale，以及候选算法搜索参考 |
| 官方 cuBLASLt MXFP8 | [LtMxfp8Matmul][mxfp8] | Blackwell block-scaled FP8 API 参考 |
| 官方 cuBLASLt NVFP4 | [LtNvfp4Matmul][nvfp4] | E2M1 packed 数据及 block scale 描述符参考；样例不等于完整调优器 |
| 官方 TE 全精度 GEMM | [benchmark_gemm.py][tebench] | BF16、FP8 current/delayed/block、MXFP8、NVFP4；纯 GEMM / 含量化模式，**不含 Ulysses A2A** |

cuBLASLt 参考固定在 CUDALibrarySamples commit
`a94482ebecf8b16d5b83ab276b7db3a84979f0e5`。源码只从 Mac 获取，再经云盘离线传输。
TE 文档当前显示 2.18.0；链接的 benchmark 为 main，使用前须固定其 commit、
匹配的 TE/PyTorch/CUDA 版本与构建参数，不自动覆盖现有适配版 TE。

## FP8 / FP4 不是一个 dtype 参数就够

| 精度合同 | 值格式 | scale 语义 | 输出和通信需单独声明 |
|---|---|---|---|
| BF16 | BF16 | 无量化 scale | 当前 BF16 边界的输入/权重/输出/通信均 BF16 |
| 旧 v12 FP8 | E4M3 | 外部准备已量化输入和组合 alpha | 输入、权重、通信、输出均 E4M3；计时不含量化/amax |
| FP8 tensor-wise | E4M3（前向） | FP32 per-tensor；current/delayed 量化算法不同 | BF16 输出与 E4M3 输出不可混表 |
| MXFP8 | E4M3/E5M2 | 32 个值一组、E8M0 scale | scale 布局、转置后的重分块及通信 payload 必须明确 |
| NVFP4 | packed E2M1 | 16 个值一组、E4M3 block scale，加 FP32 tensor scale | 通信若为 FP4，须计入 scales、padding、pack/unpack；GEMM 为 FP4 不代表输出/A2A 为 FP4 |

格式定义见 [TE FP8 primer][primer] 与 [NVFP4 文档][tefp4]。
NVFP4 的 GEMM / all-gather 支持不能推导出旧版定制 P2P A2A UB 已支持 NVFP4。
必须检查实际扩展、量化缓冲区生命周期和 stream 同步，不能只设置 `use_fp8=True`。

## 官方 TE GEMM 的两种测量口径

下面命令在**已固定版本且包含该脚本的 TE checkout** 中运行；当前 L20D 尚未
安装依赖，未执行这些命令。`--shapes` 次序是 **M x K x N**，不是 M x N x K。
这里选择与本次单卡 cuBLASLt 诊断相同的 M=512、N=4096、K=2048：

```bash
# 默认启用该版本支持的多种 recipe，含逐步量化开销。
CUDA_VISIBLE_DEVICES=0 /root/workspace_wct/bench-env/bin/python \
  benchmarks/gemm/benchmark_gemm.py --shapes 512x2048x4096 \
  -o /root/workspace_wct/profiles/te-gemm-autocast.png

# 输入在计时外量化，观察 GEMM 本身。必须与上面的结果分表。
CUDA_VISIBLE_DEVICES=0 /root/workspace_wct/bench-env/bin/python \
  benchmarks/gemm/benchmark_gemm.py --shapes 512x2048x4096 --pre-quantize \
  -o /root/workspace_wct/profiles/te-gemm-prequantized.png
```

该工具的 FP8 DelayedScaling 即使指定 `--pre-quantize` 仍包含动态量化，
不能归入纯 GEMM 同口径比较。官方也说明 FP8 Block 在 Blackwell 上通过 MXFP8
兼容实现，故两者不能重复计为两个独立原生格式。[官方教程][tutorial]

## 接到完整 A2A 边界前的验收

1. 单卡每精度先验证原生 kernel dispatch、scale/layout、误差阈值与非平凡输入；
   profile 验证不能只看 Python recipe 名称，避免无声回退。
2. 分别固定 GEMM 输入/权重/accumulator/output dtype、通信 dtype、量化计时边界。
   FP4 输出重分块和 OProj beta=1 累加路径须有独立正确性检查。
3. 先两卡 smoke，再扩到 CP4/8；发送量需包含 scale 与 padding。跨 rank 比较同一
   已量化输入的数值参考，并另外记录对 BF16 参考的量化误差。
4. 每精度、每 shape、每方向、Eager/Graph 独立调优 cuBLASLt 与通信配置。
   最终沿用 10 warmup、50 个 max-rank 样本和 p50/p95，不能沿用 H200 winner。
5. 表中分开列纯 GEMM、含量化 GEMM、完整分离边界、完整 UB 重叠边界。
   未实现/不支持/失败使用状态字段，禁止用 BF16 运行结果填充 FP8/FP4 行。

当前 `sm103/bench.py --precision` 只接受 `bf16`，因此低精度误调用会立即报错。
三精度纯 GEMM 使用 `sm103/GEMM/cublaslt_bench.py --precisions bf16,fp8,fp4`；
量化在计时区外，输出统一为 BF16，不能与“低精度 A2A payload”混为一谈。

[fp8]: https://github.com/NVIDIA/CUDALibrarySamples/tree/a94482ebecf8b16d5b83ab276b7db3a84979f0e5/cuBLASLt/LtFp8Matmul
[fp8find]: https://github.com/NVIDIA/CUDALibrarySamples/tree/a94482ebecf8b16d5b83ab276b7db3a84979f0e5/cuBLASLt/LtFp8CustomFind
[mxfp8]: https://github.com/NVIDIA/CUDALibrarySamples/tree/a94482ebecf8b16d5b83ab276b7db3a84979f0e5/cuBLASLt/LtMxfp8Matmul
[nvfp4]: https://github.com/NVIDIA/CUDALibrarySamples/tree/a94482ebecf8b16d5b83ab276b7db3a84979f0e5/cuBLASLt/LtNvfp4Matmul
[tebench]: https://github.com/NVIDIA/TransformerEngine/blob/main/benchmarks/gemm/benchmark_gemm.py
[primer]: https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html
[tefp4]: https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html
[tutorial]: https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/gemm_profiling/gemm_profiling.html
