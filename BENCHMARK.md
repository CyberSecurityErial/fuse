# A2A + O-projection Benchmark

## 口径

- shape：三组模型宽度 × 三个全局序列长度 × CP4/CP8，共18个 case；
- BF16，10次 warmup + 50次正式采样；
- 每次采样先取跨 rank 最大延迟，再报告 p50；
- 融合吞吐只计算 GEMM FLOPs，耗时包含 A2A；
- correctness 失败的配置不进入比较。

## 基线如何调优

纯 GEMM 同时测 TE Linear、经典 cuBLAS 和 cuBLASLt。cuBLASLt 使用64 MiB workspace，请求最多64个 heuristic；每个可运行候选做5次 warmup + 30次计时，保留本地实测最快项。

TE+NCCL 与 cuBLASLt+NCCL 对每个 shape 独立搜索：

```text
NCCL channels       = 8, 16, 24, 32
P2P chunk           = 128, 256, 512, 1024 KiB
LL threshold        = 16, 64, 128 KiB
第一轮              = 48个配置，Graph + 高优先级 stream
第二轮              = winner × {Graph,eager} × {高优先级,普通优先级}
最终                = winner 重新跑10+50
```

CP4和CP8各扫描432个第一轮配置。正式分离基线取 TE+NCCL 与 cuBLASLt+NCCL 中更快的一项。

因此表中的“最优分离”不是默认 NCCL，也不是单次偶然值：每个 shape 都先完成通信参数与执行模式搜索，再用 winner 做正式复测。`baseline_summary.json` 保存最终选择，原始 JSON 可以重新验证该选择。

融合侧固定使用生产入口：

```text
--comm-ctas 0 --lhs-policy auto --raster n --swizzle 1
```

运行时在 `M64N128`、`M128N128`、`M128N160`、`M128N256 cluster-M2` 中选择；18个 Golden case 都自动得到 `comm_ctas=4`，没有逐 shape 手工覆盖。

使用者要达到表中融合性能，只需保持自动入口；手工指定 tile 或 `comm_ctas` 属于实验模式，不是 Golden 配置。

## 复现

完整基线搜索需要安装 TransformerEngine，并让 `torchrun` 指向该环境：

```bash
export FUSE_CP4_DEVICES=0,1,2,3
export FUSE_CP8_DEVICES=0,1,2,3,4,5,6,7

python3 benchmarks/oproj_shape_bench.py \
  --phase baseline-nccl-sweep \
  --phase baseline-mode-sweep \
  --phase baseline-formal \
  --phase baseline-aggregate \
  --phase fuse-formal \
  --phase shape-table \
  --models representative_small,representative_medium,representative_large \
  --seqs 1024,4096,16384 --cps 4,8 \
  --torchrun "$(command -v torchrun)" \
  --results results/reproduce
```

原始 sweep、正式结果和汇总分别保存在 `baseline_nccl_sweep`、`baseline_mode_sweep`、`baseline_formal`、`fuse_formal` 和 `baseline_summary.json`。聚合逻辑位于 `benchmarks/oproj_shape_bench.py`，可以直接审计 winner 的选择过程。

## 最终结果

实验设置：单机8×H200、NVLink、每卡132 SM、CUDA 12.8。Golden 的 CP4 使用设备 `0,2,4,5`，CP8 使用全部设备；可通过 `FUSE_CP4_DEVICES` 和 `FUSE_CP8_DEVICES` 修改。

| CP | 规模 | 全局 S | GEMM M×N×K | 融合延迟 | 融合吞吐 | 相对最优分离 | 纯 GEMM 的百分比 | 自动 tile |
|---:|---|---:|---|---:|---:|---:|---:|---|
| 4 | 小 | 1K | `256×4096×4096` | 40.10 μs | 214.2 T | 2.23× | 50.9% | `M64N128` |
| 4 | 小 | 4K | `1024×4096×4096` | 92.51 μs | 371.4 T | 1.54× | 53.1% | `M128N128` |
| 4 | 小 | 16K | `4096×4096×4096` | 297.62 μs | 461.7 T | 1.17× | 58.8% | `M128N256 C2` |
| 4 | 中 | 1K | `256×5120×5120` | 43.66 μs | 307.4 T | 1.71× | 67.5% | `M128N128` |
| 4 | 中 | 4K | `1024×5120×5120` | 111.87 μs | 479.9 T | 1.68× | 65.5% | `M128N160` |
| 4 | 中 | 16K | `4096×5120×5120` | 364.48 μs | 589.2 T | 1.72× | 71.4% | `M128N256 C2` |
| 4 | 大 | 1K | `256×7168×16384` | 133.90 μs | 448.8 T | 1.46× | 71.2% | `M128N128` |
| 4 | 大 | 4K | `1024×7168×16384` | 382.29 μs | 629.2 T | 1.68× | 79.6% | `M128N256 C2` |
| 4 | 大 | 16K | `4096×7168×16384` | 1494.77 μs | 643.6 T | 1.43× | 80.1% | `M128N256 C2` |
| 8 | 小 | 1K | `128×4096×4096` | 32.10 μs | 133.8 T | 2.11× | 63.6% | `M64N128` |
| 8 | 小 | 4K | `512×4096×4096` | 63.22 μs | 271.8 T | 2.86× | 58.5% | `M128N128` |
| 8 | 小 | 16K | `2048×4096×4096` | 196.48 μs | 349.8 T | 1.68× | 59.8% | `M128N256 C2` |
| 8 | 中 | 1K | `128×5120×5120` | 38.75 μs | 173.2 T | 1.55× | 65.6% | `M64N128` |
| 8 | 中 | 4K | `512×5120×5120` | 82.77 μs | 324.3 T | 1.35× | 63.1% | `M128N160` |
| 8 | 中 | 16K | `2048×5120×5120` | 281.86 μs | 381.0 T | 1.55× | 63.3% | `M128N256 C2` |
| 8 | 大 | 1K | `128×7168×16384` | 108.16 μs | 278.0 T | 1.41× | 75.7% | `M64N128` |
| 8 | 大 | 4K | `512×7168×16384` | 265.09 μs | 453.7 T | 1.56× | 74.5% | `M128N128` |
| 8 | 大 | 16K | `2048×7168×16384` | 835.52 μs | 575.7 T | 1.65× | 83.2% | `M128N256 C2` |

归档目录：`results/oproj_shape_bench`、`results/oproj_shape_bench_cp8`。
