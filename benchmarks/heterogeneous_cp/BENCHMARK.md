# Locked-frequency heterogeneous CP benchmark

当前版本：v9.0。状态：已发布的实验特性；原有均匀 QKV/OProj 算子保持 v8.0 行为。

## 口径

- BF16，测试两个独立算子：QKV Projection + Ulysses A2A，以及 Ulysses A2A + OProj；
- 只处理调用方已经知道、并且在负载期间保持稳定的锁频差异；算子不读取频率，也不预测 GPU 自动降频；
- QKV 使用 `M×9216×4096`、`Hq/Hkv/D=24/24/128`，OProj 使用 `M×4096×3072`；
- CP2/4/6/8，覆盖当前八卡能够组成的不同慢卡数量；
- 每个结果使用 5 次 warmup + 30 次采样，每个样本先取所有 rank 的最大延迟，再报告 p50；
- uniform 与 weighted 使用相同输入、权重和最终布局；执行后的 BF16 输出逐元素完全相同；
- 本轮设备的物理 GPU 1/3/6 锁定 1500 MHz，其余卡的最高 SM 频率为 1980 MHz；所有 HBM 均为 3201 MHz，NVLink 配置未变。

## 调度规则

调用方为每个 rank 提供三个无量纲能力比：SM、HBM 和 NVLink。纯 SM 锁频且频率稳定时，1500 MHz 相对 1980 MHz 可写成 `sm=1500/1980`、`hbm=1`、`nvlink=1`。这些值只在冷路径参与规划，kernel 热路径不查询设备状态。

规划器枚举整数行数、通信 CTA 和现有 tile，目标是让预测最慢 rank 的时间最短。行数必须满足统一的 `row_quantum`，所有 rank 的连续区间仍完整覆盖同一全局序列。`equivalent_alpha` 仅用于解释结果，不参与选择，也不是调用方超参数。

QKV 的默认适用范围为全局 `S≤16K`。更长的 QKV 会持续压满 Tensor Core，未锁定的参考卡可能撞到功耗上限，使标称频率不再代表实际算力；因此 `S>16K` 默认直接使用原来的 uniform 算子。只有调用方已经测得稳定的长 QKV 能力比时，才可显式使用 `--allow-long-qkv` 做实验。

OProj 在本轮长序列中仍有稳定收益，因此不使用这个长度限制。任何算子只要物理模型没有预测到严格收益，都会直接回退 uniform；回退时不会启动 weighted kernel。

## 复现

```bash
cmake -S . -B build-v9 -DCMAKE_BUILD_TYPE=Release
cmake --build build-v9 --parallel 8 \
  --target heterogeneous_cp_plan_test heterogeneous_cp_weighted_bench

./build-v9/heterogeneous_cp_plan_test

python3 benchmarks/heterogeneous_cp/weighted_sweep.py \
  --binary ./build-v9/heterogeneous_cp_weighted_bench \
  --output /tmp/heterogeneous_cp_locked_frequency.json \
  --worlds 2,4,6,8 --m-values 2048,16384 \
  --slow-ratio 0.7575757576 --auto-plan \
  --warmup 5 --iterations 30 --check
```

`--alpha` 只保留给人工消融；正式自动规划使用 `--auto-plan`，不读取 alpha。

## 完整结果

仓内精简结果：[`results/heterogeneous-cp/locked_frequency_summary.csv`](../../results/heterogeneous-cp/locked_frequency_summary.csv)。它保存22个setting的设备组合、资源比例、uniform/weighted p50、实际分区、通信CTA和加速比；完整逐rank诊断可由上面的命令重新生成。

| CP | 慢卡数 | 本地 M | 全局 S | QKV 路径 | QKV 加速 | OProj 路径 | OProj 加速 |
|---:|---:|---:|---:|---|---:|---|---:|
| 2 | 0 | 2048 | 4096 | uniform 回退 | 1.0000× | uniform 回退 | 1.0000× |
| 2 | 0 | 16384 | 32768 | uniform 回退 | 1.0000× | uniform 回退 | 1.0000× |
| 2 | 1 | 2048 | 4096 | weighted | 1.0901× | weighted | 1.2059× |
| 2 | 1 | 16384 | 32768 | uniform 回退 | 1.0000× | weighted | 1.2804× |
| 2 | 2 | 2048 | 4096 | uniform 回退 | 1.0000× | uniform 回退 | 1.0000× |
| 2 | 2 | 16384 | 32768 | uniform 回退 | 1.0000× | uniform 回退 | 1.0000× |
| 4 | 0 | 2048 | 8192 | uniform 回退 | 1.0000× | uniform 回退 | 1.0000× |
| 4 | 0 | 16384 | 65536 | uniform 回退 | 1.0000× | uniform 回退 | 1.0000× |
| 4 | 1 | 2048 | 8192 | weighted | 1.0808× | weighted | 1.1766× |
| 4 | 1 | 16384 | 65536 | uniform 回退 | 1.0000× | weighted | 1.2933× |
| 4 | 2 | 2048 | 8192 | weighted | 1.0659× | weighted | 1.1948× |
| 4 | 2 | 16384 | 65536 | uniform 回退 | 1.0000× | weighted | 1.3000× |
| 4 | 3 | 2048 | 8192 | uniform 回退 | 1.0000× | uniform 回退 | 1.0000× |
| 4 | 3 | 16384 | 65536 | uniform 回退 | 1.0000× | weighted | 1.2141× |
| 6 | 1 | 2048 | 12288 | weighted | 1.0731× | uniform 回退 | 1.0000× |
| 6 | 1 | 16384 | 98304 | uniform 回退 | 1.0000× | weighted | 1.3949× |
| 6 | 2 | 2048 | 12288 | weighted | 1.0619× | uniform 回退 | 1.0000× |
| 6 | 2 | 16384 | 98304 | uniform 回退 | 1.0000× | weighted | 1.3986× |
| 6 | 3 | 2048 | 12288 | weighted | 1.0705× | weighted | 1.2293× |
| 6 | 3 | 16384 | 98304 | uniform 回退 | 1.0000× | weighted | 1.4216× |
| 8 | 3 | 2048 | 16384 | weighted | 1.0740× | weighted | 1.1593× |
| 8 | 3 | 16384 | 131072 | uniform 回退 | 1.0000× | weighted | 1.4178× |

混合频率组合中，短 QKV 启用的 7 个 setting 全部提升，范围为 `1.0619×～1.0901×`；长 QKV 全部回退为 `1.0000×`。OProj 启用的 13 个 setting 全部提升，范围为 `1.1593×～1.4216×`。其余 setting 均由规划器主动回退，没有性能退化。

## 限制

- 当前正式数据只验证了 SM 锁频；HBM/NVLink 能力比已经进入模型和 API，但没有对应的降频硬件可做性能验收，因此仍属于实验接口；
- weighted ownership 会改变每个 CP rank 的本地序列长度，框架必须让相同分区贯穿依赖该序列布局的后续计算；
- 自动 DVFS、温度变化、功耗墙和共享机器上的动态争用不属于本版本保证范围。
