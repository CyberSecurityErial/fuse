# MXFP8 权重四算子 Benchmark

当前版本：SM90 MXFP8-weight 2F2B baseline

## 口径

- 沿用已有 QKV Projection→A2A、A2A→OProj、QKV backward、OProj backward 四条融合边界；
- 只离线量化权重：E4M3 payload，每 32 个原始输入维元素共享一个 E8M0 scale；
- 前向和反向 B 每次调用先软件反量化，再执行 BF16 Tensor Core GEMM，FP32 累加；不是原生 MXFP8 GEMM；
- 激活、通信、前向输出与数据梯度保持 BF16；W 阶段写 FP32 `main_grad`，不读取量化权重，也没有权重反量化；
- 每个算子沿用原有 8 组模型几何、6 档全局序列长度和 CP4/CP8，共 96 个 setting；四算子共 384 个 setting；
- 表内 `S` 是展平后的 token 数 `T`，`M=T/CP`，不增加训练 batch 维度。历史 registry 的 `batch=1` 仅保留作兼容字段；
- 每个正式结果使用 10 次 warmup + 50 次采样；一进程一卡，每个样本先取跨 rank 最大 CUDA Event 时间，再报告 p50/p95；
- Eager 与 CUDA Graph 分别采样、分别成列；正文时间单位为 ms，吞吐单位为 TFLOPS/GPU；
- 前向与独立 B/W 吞吐只计算对应 GEMM 的 `2MNK`；反向总吞吐计算两次 GEMM，总计 `4×M×weight_rows×weight_columns`。

本目录不包含 KDA。本次是完整 BF16 weight workspace 的反量化基线，
不是片上 tile-fused DQ。调用方分配和管理 workspace；每个并发调用需要独立空间。
生产入口不分配内存，不缓存反量化权重，不创建 cuBLAS handle，也不执行 host 同步。

### 测试设备

H200/SM90a，profiling 关闭。CP4 固定使用物理卡 `0,2,4,5`；CP8 使用 `0–7`，
与旧 BF16/FP8 benchmark 相同。运行时原始设备名为 `NVIDIA L20X`，
CC 9.0、132 SM、约 140 GiB 显存；raw JSON 保留原始身份，不改写测量元数据。
不同卡组只在各自 CP 内比较，不拿 CP4/CP8 的吞吐差异直接解释并行收益。

## Shape

Shape 直接导入已有 SM90 registry，不重新维护模型列表，不按原生上下文长度裁剪。
全局序列固定为 1K/4K/16K/128K/256K/512K。

| 算子 | GEMM M×N×K | 布局 | 原有 Shape 表 |
|---|---|---|---|
| QKV forward | `M×QKV×H` | rank-major，完整 Q/K/V | [QKV Projection + A2A](../QKVproj+a2a/BENCHMARK.md#shape) |
| OProj forward | `M×H×A` | causal-paired inverse A2A | [A2A + OProj](../a2a+Oproj/BENCHMARK.md#shape) |
| QKV backward B / W | `M×H×QKV` / `QKV×H×M` | causal-paired inverse A2A | [QKV backward](../QKVproj-backward/BENCHMARK.md#反向-shape-和顺序) |
| OProj backward B / W | `M×A×H` / `H×A×M` | causal-paired Sequence→Head A2A | [OProj backward](../Oproj-backward/BENCHMARK.md#反向-shape-和顺序) |

`QKV=(Hq+2×Hkv)×D`，`A=Hq×D`。即使 MNK 相同，Hq/Hkv/D、CP 和布局也必须匹配。
前向权重按 `[QKV,H]` / `[H,A]` 保存；反向 B 读取同一原始轴权重，
不预先转置后重新量化。QKV forward 不支持 causal-paired、peer-interleaved 权重行；
OProj forward 不支持 cyclic 权重列，入口显式拒绝这些布局。

## 普通反向和 ZeroBubble

普通模式在同一 stream 中执行 B→W，W 使用 `beta=0`。分离模式分别调用 B/W，
W 使用 `beta=1` 累加 FP32 `main_grad`；调用方必须保留 W 的两个输入直到 W 完成。
本次测量的分离模式总时间仍是实际顺序 B→W，不是 ZeroBubble 调度器的流水收益，
也不是两个独立 p50 相加。

没有优化器、权重更新或梯度 collective。旧 BF16 反向使用 BF16 `main_grad`，
本次是 FP32，因此不能把历史反向差值宣称为只改变权重量化带来的收益。

## 基线如何调优

TE Userbuffers 与 cuBLASLt+NCCL 继续使用已经完成的 MXFP8-weight 搜索结果，
不重复搜索。TE 在这里提供 Userbuffers P2P 通信，GEMM 使用 cuBLASLt，
不是 TE 原生 FP8 模块。短程选择与正式 10+50 复测分开；“最优”只指已声明候选集的 winner。

前向搜索完整 GEMM、QKV destination-slab GEMM/send、OProj row pipeline、
通信 SM 与 NCCL profile；反向按原有 B/W 矩阵、FP32 dW 和匹配 beta 搜索。
具体候选与复测规则见 [SEARCH_PROTOCOL.md](SEARCH_PROTOCOL.md)、
[前向说明](README.md)、[反向说明](BACKWARD.md)，不把旧 BF16 的候选数量冒充本次搜索数量。

最强外部基线逐 setting、逐启动方式、逐反向模式取
`min(TE Userbuffers, cuBLASLt+NCCL)`。反向以真实 total p50 选择 backend，
B/W 明细使用同一个 winner，不拼接不同 backend 的最低阶段时间。

### 经典 cuBLAS 纯 GEMM

沿用旧 benchmark 的经典 `cublasGemmEx` 对照，不是 cuBLASLt。
只测相同 MNK 的 BF16 GEMM，不含反量化、pack 或 A2A；
转置由 cuBLAS 的操作标志表达，不在计时前后物化一个不同的矩阵。

反向 B 输出 BF16，W 输出 FP32，并分别匹配 `beta=0/1`。
Fuse 与经典 cuBLAS 在本轮同一卡组测量。表内“cuBLAS 吞吐占比”
等于 Fuse TFLOPS/GPU 除以经典 cuBLAS TFLOPS/GPU，
只作计算吞吐参照，不是硬件 MFU，也不是相同完整边界的端到端加速比。

## 正确性门槛

- CP4/CP8，展平 local M=128/256，四算子共 16 个基础用例；
- Eager、Graph 重放、独立 routed 输出、数据梯度和 FP32 dW；
- 非零 `main_grad` 上连续两次 `beta=1`；输出和 workspace poison；
- 全部 65,536 个 E4M3×E8M0 编码组合，包括 subnormal、NaN 和 overflow；
- workspace 容量、别名、形状和不支持布局的拒绝检查；
- 原有 BF16/FP8 forward/backward smoke 保持通过；
- 正式性能 runner 另外检查每个 shape 的有效 BF16 权重、路由、输出和 FP32 dW。

这些检查不是性能数据。代码结构、修正项和验证范围见 [REVIEW.md](REVIEW.md)。

## 复现

使用已安装 PyTorch、Triton、CUDA 12.8 和 TE Userbuffers P2P 扩展的环境。

```bash
cmake -S . -B build-mxfp8 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DFUSE_ENABLE_PROFILING=OFF
cmake --build build-mxfp8 --target fuse_mxfp8_torch_bridge \
  backward_smoke fp8_smoke qkvproj_a2a_smoke fuse_smoke -j 2
python benchmarks/mxfp8_weight/run_checks.py --scope all

# 复现外部基线的全量搜索；已有归档时不需要重跑。
python benchmarks/mxfp8_weight/run_baselines.py --nvcc /usr/local/cuda/bin/nvcc

# 只补测 Fuse 与经典 cuBLAS，不重复搜索外部基线。
python benchmarks/mxfp8_weight/run_operator_bench.py
python benchmarks/mxfp8_weight/audit_operators.py \
  results/mxfp8_weight/operators/full_v1/cp4.json \
  results/mxfp8_weight/operators/full_v1/cp8.json --require-full \
  --source-revision d5b1317 \
  --artifact build-mxfp8/libfuse_mxfp8_torch_bridge.so=results/mxfp8_weight/operators/full_v1/libfuse_mxfp8_baseline.so
python benchmarks/mxfp8_weight/comparison_report.py --measurements \
  results/mxfp8_weight/operators/full_v1/cp4.json \
  results/mxfp8_weight/operators/full_v1/cp8.json
```

外部基线的新输出放在 `results/mxfp8_weight/reproduce/`，不覆盖已发布归档。
同一进程组处理多个 shape，仅 NCCL 环境 profile 变化时重启。
进程启动、分配、JIT、cuBLAS planning、Graph capture 和正确性检查均不进入计时。

新 Fuse runner 使用独立 cudaMalloc/IPC peer 缓冲区，一进程一卡，
同一次启动连续测多个 shape；不重新搜索已完成的外部基线。
单次 Graph capture 后重复 replay，每次 replay 前跨 rank 重置 ready/done；
重置与 Gloo host barrier 不计时。该方式不同于旧原生 MPI runner 的单调 epoch 节点链，
但都保证旧 epoch 不能冒充新一轮完成，不将两种 harness 的差异宣称为算子优化。

## 宏打点诊断

沿用 [PROFILE_PROTOCOL.md](../../PROFILE_PROTOCOL.md) 的 CTA 角色计时。
同一权重 DQ 后调用现有 BF16 role-telemetry 入口，反向再调用本次的 FP32 W，
不替换成旧 BF16 `main_grad`。正式 runner 拒绝 profiling-enabled 库。

```bash
cmake -S . -B build-mxfp8-profile -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DFUSE_ENABLE_PROFILING=ON
cmake --build build-mxfp8-profile --target fuse_mxfp8_torch_bridge -j 2
CUDA_VISIBLE_DEVICES=0,2,4,5 OMP_NUM_THREADS=1 \
  python -m torch.distributed.run --standalone --nproc-per-node=4 \
  benchmarks/mxfp8_weight/profile_operators.py --cp 4 --seqs 131072 \
  --output results/mxfp8_weight/profile/initial_cp4
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
  python -m torch.distributed.run --standalone --nproc-per-node=8 \
  benchmarks/mxfp8_weight/profile_operators.py --cp 8 --seqs 131072 \
  --output results/mxfp8_weight/profile/initial_cp8
```

默认每个 CP 只选一个几何的四算子、Graph 重放，并校验输出/路由/dW。
每个样本仅保留 `_summary.json` 与 `_perfetto.json`，不额外导出逐 tile 记录。
每卡独立时间原点；角色包络包含等待，不是裸 Tensor Core/NVLink 时间；
并行 source waits 取最大而非相加。DQ/W 的 marker 区间包含打点开销。
初轮 128K 的 CP4/CP8 共八个样本已经通过验证，仍只作瓶颈诊断。

多 CTA 校准使用 `--comm-ctas 0,8,16,24 --aggregate-only`；此模式只保留一个
逐项 checkpoint 的 `calibration_summary.json`，不写原始 CTA 或 Perfetto。
`--max-cases` 按 shape×CTA 请求数量计数，同一 shape 复用 IPC/timeline/marker。

BF16/MXFP8 QKV B 的首个 ready acquire 现与上述 CTA timeline 接通；字段
`first_ready_wait` 包含 CTA 启动工作，不是纯等待或后续 K/head 的累计等待。
旧记录的 observed=0/null 仍表示未观测，不改写成零。
`profile/qkv_backward_cp4_first_ready_v1` 为 Q2/Q3/Q4/Q8 × 128K/512K ×
c4/12/24 的 24 个 aggregate 诊断点；同名前缀 `first_ready_reverse_v1`
另保留 Q8 的六个反序点，不能当作正式性能或与旧宏数据混合拟合。

小矩阵的实际 grid 可能小于 SM 数。QKV B 诊断按现有静态 scheduler 的
cluster1/2、AlongN/swizzle1 几何声明 `grid_ctas`，要求未使用的分配尾部全零；
reporter 仍逐个校验已启动 CTA，不以丢弃缺失时间戳绕过检查。
1K/16K、Eager beta1 的四点回归与紧凑来源见
`validation/qkv_backward_ready_grid_v1/gpu_checks.json`，不是性能表。

## 通用候选与独立原语校准

### 完整反向 CTA 对照

`operator_sweep.py --backward-phase total` 只允许 backward 的 comm sweep，
对每个 `--weight-modes immediate,deferred` 实测完整 B→W。默认仍是独立 B；
原 comm/WGrad sweep 行为不变。beta0 使用原生组合入口，beta1 与正式
`operator_bench.py` 共用 `launch_boundary()`，依次发起 B 和 W，包含在同一个
event/Graph 边界内；不得把 deferred 的 B-only 入口误标为 total。
两者的 main_grad 初始化在计时外，短搜后检查 beta0 非零覆盖、beta1 连续
两次累加；末轮正式采样后再次验证输出、路由和 FP32 weight gradient。

`--pair-order candidate-auto` 交换正式 A/B 执行顺序，报告仍为 auto/candidate
比值，并保存每轮真实执行顺序。capture 和短搜仍按 auto、候选顺序，每臂
捕获一次；不能把它描述成 capture 顺序也已交换。两种执行顺序应分别保留，
不要拿单一顺序的小幅收益直接写生产策略。

```bash
env -u FUSE_QKV_COMM_POLICY -u FUSE_QKV_GEMM_POLICY \
  -u FUSE_A2A_LHS_COMM_POLICY -u FUSE_NVLINK_BIDIR_GBPS \
  CUDA_VISIBLE_DEVICES=0,2,4,5 OMP_NUM_THREADS=1 \
  python -m torch.distributed.run --standalone --nproc-per-node=4 \
  benchmarks/mxfp8_weight/operator_sweep.py --cp 4 --operators qkv_backward \
  --backward-phase total --model artificial_large,llama31_405b \
  --seqs 131072,262144,524288 --comm-ctas 0,4 \
  --pair-order candidate-auto \
  --output results/mxfp8_weight/sweep/reproduce_qkv_backward_complete_reverse.json
```

本例测 Q3/Q8 的三个长序列、Graph、两种 beta，不是全四算子、全几何或 CP8
验收。基线为原 auto（本例实际 c12），候选手动 c4；不重搜外部基线。

### 独立阶段候选

`operator_sweep.py` 的默认 `--sweep-kind comm` 测 F 或独立 B；
`--sweep-kind wgrad` 测 FP32 W，默认两种 beta，B 的准备与验证不进入 W 计时。
两者共用 2+7 短搜、候选 Graph 复用和三轮交错 10+50 A/B。短搜只能筛选，
逐阶段收益不能替代完整四算子验收。没有按模型名/shape 生成生产 winner 表。

`--sweep-kind qkv_tile --fixed-comm-ctas <偶数>` 复用同一个 runner，
只测 QKV F，并固定所有候选的通信 CTA 数。`--qkv-tiles` 的 benchmark-local
编号为 0:auto、1:N64、2:N128、3:N160、4:N192、5:N256、6:N320；
M128/K64 固定，N256/N320 为 C2，其余 C1。切换使用既有原生环境配置，
原生 launch cache 包含该配置；每臂重新查询实际 tile，检测静默回退。
环境切换、配置检查、capture、ready reset 均不进入计时；结束或报错恢复
原环境和 CTA override。没有改 BF16/FP8 内核或任何生产默认。

此实验的 auto 是“固定通信预算下自动选择 tile”，**不是**完整原 auto
dispatch，也不是 TEUB/cuBLASLt 外部基线。tile 改变还可能改变 cluster、
SMEM 和 stage 数，不能把完整时延差直接称为裸 Tensor Core 差异。
每个 case 共享分配/IPC/reference，每候选只捕获一次；2+7 只筛选两个
不同于 auto 的配置，再做三轮 10+50，保留输掉的正式候选。
当前仅完成 CP4 Q3/Q8 的 4K 正确性和三个长序列测量，未证明 CP8 性能。

```bash
env -u FUSE_QKV_COMM_POLICY -u FUSE_QKV_GEMM_POLICY \
  -u FUSE_A2A_LHS_COMM_POLICY -u FUSE_NVLINK_BIDIR_GBPS \
  CUDA_VISIBLE_DEVICES=0,2,4,5 OMP_NUM_THREADS=1 \
  python -m torch.distributed.run --standalone --nproc-per-node=4 \
  benchmarks/mxfp8_weight/operator_sweep.py --cp 4 --sweep-kind qkv_tile \
  --fixed-comm-ctas 4 --qkv-tiles 0,1,2,3,4,6 \
  --model artificial_large,llama31_405b --seqs 131072,262144,524288 \
  --output results/mxfp8_weight/sweep/reproduce_qkv_tiles.json
```

```bash
python benchmarks/mxfp8_weight/test_operators.py --wgrad-policies 0,1,2,3,4 \
  --output results/mxfp8_weight/validation/wgrad_candidate_correctness.json
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
  python -m torch.distributed.run --standalone --nproc-per-node=8 \
  benchmarks/mxfp8_weight/operator_sweep.py --cp 8 --sweep-kind wgrad \
  --model production_qwen_dense,llama3_8b --seqs 131072 \
  --output results/mxfp8_weight/sweep/reproduce_wgrad.json
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
  python -m torch.distributed.run --standalone --nproc-per-node=8 \
  benchmarks/mxfp8_weight/calibrate_operators.py --cp 8 \
  --model production_qwen_dense,llama3_8b --seqs 131072 \
  --comm-ctas 4,8,16,24,32 --max-cases 10 \
  --output results/mxfp8_weight/calibration/reproduce_primitives.json
```

独立原语复用现有 BF16 reference，不重新写 GEMM。compute-subgrid/fullgrid
使用同一个所选 tile，只改变 SM budget；copy 与融合路径共用 frontier-window
公式，记录原生 `comm_m_window`，不使用旧 reference 的固定窗口冒充同调度。
独立 GEMM 前的 BF16 权重和 routed 输入由独立参考预装，DQ/路由不藏进其计时。
记录的 bytes 是逻辑远端 payload/强制访存下界，不是硬件 DRAM 计数器。
只有完整算子 bench 才包含每次软件 DQ + BF16 GEMM + BF16 通信。

`operator_model.py` 只拟合上述独立原语，不读取 fused 短搜或正式 A/B 的 winner。
每个校准文件需至少三个不同的实际 `(N,K)`，输出包含留一几何误差、输入哈希、
显式卡组系数及适用 tile。CP 本身不是硬件服务率，不能跨卡组直接推广系数。

```bash
python benchmarks/mxfp8_weight/operator_model.py \
  --calibration results/mxfp8_weight/calibration/cp4_primitives_v1.json \
                results/mxfp8_weight/calibration/cp8_primitives_v1.json \
  --output results/mxfp8_weight/calibration/reproduce_service_model.json
```

模型默认关闭；`operator_bench.py` 和 `profile_operators.py` 支持显式
`--oproj-comm-model <JSON>`，只影响 OProj F。模型文件在 worker 初始化时加载，
选择基于实际原生 tile 与几何，不使用模型名。Graph capture 后无重复 host 决策；
Eager 仍执行真实 selector，不能宣称它没有 host overhead。手动 CTA 优先，
越域回退到旧策略；只有预测时间下降至少 10% 才切换。5µs 截距是固定先验，
不是实测 launch；流水填充项是选择假设，不是完整的等待时间分解。
`operator_sweep.py` 拒绝模型参数；comm/WGrad sweep 的 auto 表示未修改的基线。
QKV tile sweep 则显式固定通信预算，其不同基线含义见上文，不能混用标签。

`test_native_operator_model.py` 只做原生 metadata/Python 选择一致性检查，不分配
tensor 或发起 kernel，不能当作多卡正确性或性能证据。随后
`operator_policy_ab.py` 对全部原 OProj F setting 实测冻结策略：每 launch 三轮
baseline/model 配对，每臂 10+50，保留未切换/回退点，绝不按计时选择参数。
同一 shape 复用数据、IPC 和参考，每个 arm 的 Graph 仅捕获一次。

```bash
python benchmarks/mxfp8_weight/test_native_operator_model.py \
  --library build-mxfp8/libfuse_mxfp8_torch_bridge.so \
  --model results/mxfp8_weight/calibration/oproj_service_model_v1.json \
  --output results/mxfp8_weight/validation/reproduce_native_model.json
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
  python -m torch.distributed.run --standalone --nproc-per-node=8 \
  benchmarks/mxfp8_weight/operator_policy_ab.py --cp 8 --full \
  --oproj-comm-model results/mxfp8_weight/calibration/oproj_service_model_v1.json \
  --output results/mxfp8_weight/policy_ab/reproduce_cp8.json
```

CP4 使用同样命令，改为 `CUDA_VISIBLE_DEVICES=0,2,4,5`、4 个 worker 和 `--cp 4`。
这仍然仅是 OProj F 的策略验收，不是四算子全量里程碑。

正式汇总同时测两种顺序：原命令默认 `--arm-order baseline-model`，再用
`--arm-order model-baseline` 和不同输出文件测完整矩阵。此参数同时控制 capture
和每轮执行顺序，比值方向始终为 baseline/model。两顺序各三轮等权，共六个
arm p50 的几何平均，不是 pooled median。不能用某次 partial 重测覆盖退化点。

```bash
python benchmarks/mxfp8_weight/operator_policy_report.py \
  results/mxfp8_weight/policy_ab/cp4_frozen_v1.json \
  results/mxfp8_weight/policy_ab/cp4_frozen_reverse_v1.json \
  results/mxfp8_weight/policy_ab/cp8_frozen_v1.json \
  results/mxfp8_weight/policy_ab/cp8_frozen_reverse_v1.json \
  --balance-orders --output results/mxfp8_weight/policy_ab/summary
```

汇总默认拒覆盖；重建上述已有派生表需显式 `--overwrite`。平衡模式要求每 CP
恰好一份完整正序和一份完整反序，模型、native 源码、库与实际配置一致。
现有平衡结果：OProj F 长序列对原策略 CP4/CP8 分别为 1.04019/1.16183×，
合并 1.09933×；这不是相对最优外部基线的四算子成绩。

查看当前四算子完整效果时，使用同一全量 launcher 显式启用已冻结的 OProj
策略，其余算子仍走原自动配置，不追加调参搜索或多轮 A/B：

```bash
python benchmarks/mxfp8_weight/run_operator_bench.py --skip-build \
  --oproj-comm-model results/mxfp8_weight/calibration/oproj_service_model_v1.json \
  --output results/mxfp8_weight/operators/reproduce_full_model
```

仍为 384 settings/2304 records、CP4/CP8、Eager/Graph、前向和完整反向两种
模式。当前验收目标按用户更新为长序列 1.2×，1.3× 冲刺；1.4/1.5× 非硬指标。
优化摘要同时报告匹配的纯 cuBLAS 计算参考，假设通信和 DQ 都免费，不能把它
当作硬件硬上限，也不能把独立 B/W 中位数相加代替实际 total。

## QKV F 长序列通信资源模型实验

当前只验收 CP4 的 QKV Forward→A2A，沿用原八种几何及 T=128K/256K/512K，
共 24 settings、48 个 Eager/Graph 边界。原 scope 的外部基线加速 GM 为
1.123583×，其中 27/48 低于该固定门槛；不会根据新结果移动 badcase 门槛。

`qkv_forward_service_model.py` 从固定 N256/C2 的 80 个宏 profile 样本拟合
计算/路由角色，按物理 (N,K) 留一验证，保留长度漂移与 Q8 计算残差。
公式为 C=a·(2MNK/1e9)/(132−c)、R=tau·(MN/8192)/(12c)，最小化 max(C,R)。
不使用完整 F winner 拟合，也不把带 producer 等待的路由包络称为裸 NVLink。
模型仅接受已校准的物理范围/实际 tile，手动 CTA 优先，默认和域外策略不变。
当前冻结文件是实验候选，不是已接受的生产默认或四算子性能里程碑。

```bash
python benchmarks/mxfp8_weight/qkv_forward_service_model.py \
  --calibration results/mxfp8_weight/profile/qkv_forward_cp4_balance_v1/calibration_summary.json \
  --output results/mxfp8_weight/calibration/reproduce_qkv_forward_service.json
CUDA_VISIBLE_DEVICES=0,2,4,5 OMP_NUM_THREADS=1 \
  python -m torch.distributed.run --standalone --nproc-per-node=4 \
  benchmarks/mxfp8_weight/operator_policy_ab.py --cp 4 \
  --model artificial_small,artificial_medium,artificial_large,production_qwen_dense,nanbeige42_3b,llama3_8b,qwen25_14b_32b,llama31_405b \
  --seqs 131072,262144,524288 --launches eager,graph \
  --qkv-comm-model results/mxfp8_weight/calibration/qkv_forward_service_model_v1.json \
  --output results/mxfp8_weight/policy_ab/reproduce_qkv_forward_cp4.json
```

不设 `FUSE_QKV_GEMM_POLICY` 或通信 override，实际 tile 由原自动策略选择。
再用 `--arm-order model-baseline` 和另一输出文件重复同一矩阵；两个顺序都保留，
各三轮、每臂 10+50。同一配置复用 IPC/数据/参考和每臂一次 Graph capture，
不重复网格搜索。对原 auto 的配对收益与对归档 TEUB/cuBLASLt 的收益分开报告。
新 QKV A/B 使用独立 schema，不能误用旧 OProj 专属解析器。

汇总器已支持独立 QKV schema；下面将完整两序与归档外部表关联，不重跑基线：

```bash
python benchmarks/mxfp8_weight/operator_policy_report.py \
  results/mxfp8_weight/policy_ab/qkv_forward_cp4_long_v1.json \
  results/mxfp8_weight/policy_ab/qkv_forward_cp4_long_reverse_v1.json \
  --balance-orders --require-full \
  --reference-comparison results/mxfp8_weight/operators/full_model_v1/comparison/comparison_summary.json \
  --output results/mxfp8_weight/policy_ab/reproduce_qkv_summary
```

已完成两序共 576 个正式 arm/28,800 个 rank-max 样本。对原 auto 的 scope
GM 为 1.065537×，改变配置的 24 个 Eager/Graph 边界全部改善、GM 1.134815×；
24 个不变边界 GM 1.000488×。对归档外部最佳基线 GM 1.166200×，原 27 个
低于均值的边界有 14 个达到原 GM。当前 auto 相对原 Fuse 的时延 GM 慢约
2.66%，因此不能把旧表直接乘以 paired 收益。Q3/Q8 的计算问题未解决，也不
据这一个 scope 宣称四算子达到 1.2×。

后续宏诊断可显式使用 `profile_operators.py --qkv-comm-model <同一JSON>
--comm-ctas 0 --warmup 10 --aggregate-only`，沿用上面的完整模型名/长度列表、
CP4 与 `--max-cases 24`。默认 profile warmup 仍为 2；额外调用复用已捕获的
Graph，不产生额外 trace，输出记录 warmup 合同；单次诊断不是正式 10+50。

## QKV B 独立服务诊断

这不是完整 B 或四算子性能表。权重已在计时外解码到 BF16，compute 输入
和 int32 ready epoch 也在计时外准备；copy 不执行 finalize，不预填路由输出。
`compute_bare_subgrid` 与 `compute_ready_preloaded_subgrid` 保留相同 tile、
SM 预算和 fused 共享内存预留；二者差值不是 producer 等待时间。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python -m torch.distributed.run --standalone --nproc-per-node=8 \
  benchmarks/mxfp8_weight/calibrate_operators.py --cp 8 --operator qkv_backward \
  --model production_qwen_dense,artificial_small,nanbeige42_3b,artificial_medium \
  --seqs 131072 --comm-ctas 4,8,12,16,24,32 --max-cases 24 \
  --primitives compute_bare_subgrid,copy,compute_ready_preloaded_subgrid \
  --output results/mxfp8_weight/calibration/reproduce_qkv_cp8.json
```

CP4 改卡组为 `0,2,4,5`，worker/CP 改为 4。单独复测
`--primitives copy_fused_reservation` 可匹配 fused 的 cluster/动态共享内存，
但它仍不包含 concurrent compute。原 `copy` 的 C1 启动保持可复现，禁止将
不同 launch variant 混拟合。所有输出默认拒绝覆盖。已有校准保存在
`calibration/cp{4,8}_qkv_primitives_v1.json` 和
`calibration/cp{4,8}_qkv_copy_reservation_v1.json`，生产 QKV 策略仍未变更。

## QKV F 独立服务诊断

本 scope 是 CP4、rank-major 的 GEMM→A2A 前向，长序列 128K/256K/512K。
使用 profiling OFF 的同一校准 runner；不把独立服务时间作为完整 F 加速比。
首个匹配 reference 支持原策略的 M128N256K64/C2/stage4，其他实际 tile
明确返回 NotSupported，不能静默换 tile。下面固定 tile，只改变通信 SM 预留：

```bash
env -u FUSE_QKV_COMM_POLICY FUSE_QKV_GEMM_POLICY=m128n256 \
  CUDA_VISIBLE_DEVICES=0,2,4,5 OMP_NUM_THREADS=1 \
  python -m torch.distributed.run --standalone --nproc-per-node=4 \
  benchmarks/mxfp8_weight/calibrate_operators.py --cp 4 --operator qkv_forward \
  --model artificial_large,nanbeige42_3b,llama31_405b --seqs 131072 \
  --comm-ctas 4 --max-cases 3 \
  --output results/mxfp8_weight/calibration/reproduce_qkv_forward_cp4.json
```

| 原语 | 测量边界 |
|---|---|
| `producer_signaling_subgrid` | 原 GEMM/scheduler，逐 tile 排空 TMA store 并发布 ready；使用 SM−c 预算 |
| `producer_no_signal_subgrid` | 同一 kernel entry、同一准备，只把 epilogue ready 指针置空 |
| `producer_signaling_fullgrid` / `producer_no_signal_fullgrid` | 保留选中的 tile，SM 预算改为全卡 |
| `copy_fused_reservation` | 原 12-slot QKV route，不等 producer；保留 fused cluster/动态共享内存，无 finalize |

所有原语都排除 DQ、另一个角色、分配、IPC、Graph capture 和准备，保留
10 warmup/50 sample-wise rank-max。signaling 两组的差值包含逐 tile 排空与
发布，不是裸原子耗时；独立 copy 没有并发计算，不能从融合 route 包络相减
就称为 ready 等待。full/subgrid 的比值是本实现资源参考，不是硬件理论上限。
实际 grid 可能被短问题的 tile 数限幅，输出分开记录 SM 预算和真实 launch grid。

每个几何复用一份 IPC/state 和验证缓冲，每原语/CTA 复用 Graph 与 event。
compute 每次在计时外预载相同 BF16 权重；copy 的只读、确定性 BF16 payload
在 warmup/poison 验证前准备，不每个样本重拷数 GB 源数据。copy 使用独立
bit-exact Q/K/V 路由期望值；producer 检查原始 local GEMM 和完整 strided
ready 区，signaling 应发布 epoch，no-signal/copy 应保持零。poison 验证与
采样后验证均等待所有 peer 完成。校准只输出一个紧凑 JSON，不生成额外 trace。

## 完整结果

新版 `operators/full_model_v1` 已完成并通过源码版本 `d89e40e` 的完整审计：
384 settings、2,304 records、1,152 个匹配完整边界，无缺项。只显式启用冻结
OProj F 模型，其余三个算子保持原自动策略；本轮没有重搜外部基线。
profiling OFF、10 warmup / 50 sample-wise rank-max，覆盖原始六档 T、CP4/8、
Eager/Graph 和反向 beta0/1。全量边界等权 / 四算子等权为 1.038383× / 1.046918×；
长序列为 1.151840× / 1.169852×，463/576 个长边界获胜，整体 1.2× 尚未达到。

| 算子 | CP4 全量 | CP8 全量 | CP4 长序列 | CP8 长序列 |
|---|---:|---:|---:|---:|
| QKV F | 1.020186× | 1.056107× | 1.123583× | 1.262458× |
| OProj F | 1.083687× | 1.135057× | 1.209184× | 1.315452× |
| QKV B+W | 1.035499× | 0.979171× | 1.115616× | 1.084518× |
| OProj B+W | 1.047016× | 1.025722× | 1.121011× | 1.146299× |

比例为历史最优外部基线 / 当前 Fuse；不是同轮外部 A/B。长序列纯 GEMM 参考
为 1.329338× / 1.350255×，假设通信和 DQ 都免费，不是仅隐藏通信的硬件上限。
完整逐点数值、最优配置与原始采样分别保留在下表及本轮 CP4/8 原始 JSON 中。

外部基线已完成：前向 192 个 setting、768 条记录；反向 192 个 setting、
1,536 条记录。它们是已完成归档的导出，不冒充本分支新 Fuse 的性能。
初版 Fuse 与纯 cuBLAS 已完成全部 384 个 setting、2,304 条记录，并通过完整
rank-max 采样、源码/库哈希、配置和正确性审计，无缺项。
长序列 128K/256K/512K 的 576 个 Fuse 完整边界相对最优外部基线几何平均为
1.132×；先合并反向两种模式、再四算子等权为 1.140×。全量边界等权为 1.024×。
这些是优化前的起点，尚未达到 1.2× 阶段目标。

| 范围 | 最优吞吐与对应配置 |
|---|---|
| 新版最强外部基线 / Fuse / 经典 cuBLAS | [Markdown](../../results/mxfp8_weight/operators/full_model_v1/comparison/comparison_summary.md) · [CSV](../../results/mxfp8_weight/operators/full_model_v1/comparison/comparison_summary.csv) · [JSON](../../results/mxfp8_weight/operators/full_model_v1/comparison/comparison_summary.json) |
| 新版全量 / 长序列及纯 GEMM 参考 | [Markdown](../../results/mxfp8_weight/operators/full_model_v1/comparison/optimization_summary.md) · [审计](../../results/mxfp8_weight/operators/full_model_v1/audit.json) |
| 两个前向算子 | [Markdown](../../results/mxfp8_weight/published/forward_best.md) · [CSV](../../results/mxfp8_weight/published/forward_best.csv) · [JSON](../../results/mxfp8_weight/published/forward_best.json) |
| 两个反向算子 | [Markdown](../../results/mxfp8_weight/published/backward_best.md) · [CSV](../../results/mxfp8_weight/published/backward_best.csv) · [JSON](../../results/mxfp8_weight/published/backward_best.json) |
| 初版最强外部基线 / Fuse / 经典 cuBLAS | [Markdown](../../results/mxfp8_weight/comparison/comparison_summary.md) · [CSV](../../results/mxfp8_weight/comparison/comparison_summary.csv) · [JSON](../../results/mxfp8_weight/comparison/comparison_summary.json) |
| 初版全量 / 长序列与逐算子优化汇总 | [Markdown](../../results/mxfp8_weight/comparison/optimization_summary.md) · [JSON](../../results/mxfp8_weight/comparison/optimization_summary.json) |

表格沿用旧 BF16/FP8 的 `CP / 模型 / S / GEMM M×N×K / p50 ms / TFLOPS/GPU / 配置`
列，融合对比将 Eager/Graph 横向成列；反向按普通/分离模式分别展示 B/W/总时间。
JSON 保留原始微秒值、匹配配置、rank-0 GEMM plan、正确性、来源文件与 SHA256；
各 rank 的 GEMM 调优独立完成。原始候选和逐 rank 采样向量保留在原运行归档。

[metadata.json](../../results/mxfp8_weight/published/metadata.json) 记录归档来源、硬件、
原始 source fingerprints 和发布文件哈希；导出不会改写旧 source hash 来伪装成本分支复测。
