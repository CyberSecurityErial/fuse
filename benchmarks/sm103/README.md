# SM103 Ulysses forward baseline bench

新增可选的[大模型/近期模型投影矩阵](PROJECTION_SHAPES.md)，覆盖 GQA、MHA、
MLA、KDA 与 Gated DeltaNet。完整融合边界和独立 GEMM 分开记录；新增模型显式选择。
生产默认序列已移除 1K/4K，保留 16K/128K/256K/512K；小尺寸仅用于 smoke/回归。

本目录为 B300 / SM103a 上的两条前向边界准备强基线：
QKV Projection → 完整 Q/K/V A2A，以及 inverse A2A → O-projection。
独立基线由本目录的 Python runner 测量；自研 BF16 融合算子使用
`fused_bf16.cu` harness，内核在 `csrc/operators/sm103`。当前节点2的编译、
正确性、profiling 和调优实测见[融合开发记录](../../csrc/operators/sm103/README.md)。
不把 SM90a/WGMMA 的编译产物视为 Blackwell 实现，也不将局部结果视为全量达标。

BF16、FP8（tensor-wise 与 MXFP8）、FP4（NVFP4）的本地/官方 benchmark
入口和精度合同见 [PRECISIONS.md](PRECISIONS.md)。独立 cuBLASLt GEMM 已覆盖
BF16/E4M3/NVFP4；分布式计划器只接受 BF16，因为 TE Ulysses Userbuffers
没有 FP4 A2A 合同。

## 已验证状态（2026-09-05）

- L20D 节点 `e01-cn-4jg4wglzy09` / `l20d-xerkjfcp-0001`，CUDA Runtime
  报告 compute 10.3、148 SM；nvidia-smi 显示 L20D / 8.9。
- CUDA 13.0.48 + GCC 12 + CMake 3.31.6 + Ninja：独立 `sm_103a` Release 构建通过。
- `cublaslt_smoke` 在 GPU 0 通过：本地 heuristic 调优、BF16 beta=0/1、
  三次 CUDA Graph replay，每次逐元素检查 32,768 个结果。
- 既有 `blackwell_probe_sm103a` 的 NCU 2025.3.0、NSYS 2025.3.2 采集和
  报告导出通过。NCU 采到一个 kernel（basic set，10 passes）；NSYS 报告
  `probe(int *)` 一次执行、1.600 μs。这仅证明采集链路工作，单 CTA micro test
  的时长和利用率不代表 GEMM 性能。
- CPU 测试覆盖 shape、通信 SM 预算、Eager/Graph 独立配置、NCCL 搜索空间、
  max-rank 样本分位数与无效结果拒绝。
- `fuse_sm103_cublaslt` 已在 GPU 0 完成 BF16、tensor-wise E4M3 FP8、NVFP4
  的 Eager/Graph 数值 smoke；训练代表 QKV 形状保存候选与 10+50 样本。
- 分布式边界依赖 workspace-local PyTorch cu130 和固定 commit、带 P2P adapter
  的 TE 构建；只有实际生成 correctness 与 max-rank raw samples 后才称为结果。

## 历史 bench 的构建与沿用关系

| 历史部分 | 构建/入口 | SM103a 本次处理 |
|---|---|---|
| v1–v3 OProj Golden | CMake `fuse_kernels`、`fuse_smoke`、`fuse_bench`；外部 CTA 标定 | 保留 Hopper 内核及 Golden |
| v4–v5 两条前向强基线 | CMake cuBLASLt `.so` + Python shape/tuning runner；QKV 正式 launcher 为 MPI 一进程一卡 | 沿用 shape、完整边界和计时合同，新建结果目录与调优计划 |
| v6–v8 QKV 优化 | `qkvproj_a2a_mpi_bench`、六种 tile/自动 CTA/计划缓存 | 不复用 H200 wave 标定、132 SM 或旧 winner |
| v9/9.1 weighted CP | `heterogeneous_cp_plan_test`、`heterogeneous_cp_weighted_bench` + Python sweep | 不在本次均匀前向范围 |
| v10–v11.1 backward | `backward_smoke`、两个 MPI bench、可选 Torch bridge；TE UB / autograd Python 验证 | 不在本次范围 |
| v11.2 E2E | 外部 Megatron/TE 完整训练脚手架及结果归档 | 不把 E2E 数据混入边界 microbench |
| v12 FP8 | `fp8_smoke` + FP8 shape runner，QKV MPI 支持 FP8 | 保留 E4M3 四边界合同；Blackwell 低精度入口见 PRECISIONS |
| v13/13.1 | 单 CUDA translation unit 的结构重构 | 沿用重构后的 baseline 文件，不触碰生产内核 |

当前 main `16de0b2` 相对 v13.1 的差异是移除 MegaMoE prototype，原 Ulysses
bench 保留。新构建由根 CMake 的 `FUSE_ARCH=sm103` 分支启动，
仅复用 `csrc/baselines/cublaslt_runner.cu` 和 `benchmarks/GEMM/cublaslt_bench.cu`。

## 构建与基础验证

在 L20D 的现有主 screen 中：

```bash
cd /root/workspace_wct/fuse
(
  source /root/workspace_wct/env.sh
  bash scripts/build_sm103_bench.sh
)
CUDA_VISIBLE_DEVICES=0 ./build/sm103/cublaslt_smoke
CUDA_VISIBLE_DEVICES=0 ./build/sm103/cublaslt_bench \
  --m 512 --n 4096 --k 2048 --workspace-mib 64 --candidates 64 \
  --tune-warmup 5 --tune-iterations 30 --warmup 10 --iterations 50

CUDA_VISIBLE_DEVICES=0 /root/workspace_wct/bench-env/bin/python \
  benchmarks/sm103/GEMM/cublaslt_bench.py \
  --m 512 --n 4096 --k 2048 --precisions bf16,fp8,fp4 \
  --launches eager,graph --output results/sm103/gemm/qwen-qkv.json
```

最后一个命令仅为单卡纯 GEMM 诊断，不包含 A2A，不是正式分布式边界结果。
`BUILD_DIR`、`BUILD_JOBS`、`CUDACXX`、`CUDAARCHS` 可在构建子 shell 中覆盖。
默认产物位于 `build/sm103`；不需要 CUTLASS、Torch、MPI 或 TE 即可完成 C++ 构建。

## 两条强基线

1. **cuBLASLt + NCCL 完整分离边界**：复用旧 `te_nccl_baseline.py` 中的
   `cublaslt_packed_qkv_gemm_a2a` / `cublaslt_nccl_oproj_boundary`。
   该 Python 脚手架仍依赖 TE 的 Linear/route 工具，即使仅选择 cuBLASLt 指标，
   也需要安装 TE。QKV packed 结果是 segment-major / rank-major 完整 Q/K/V；
   OProj 是 TE causal inverse route。保留旧 worker 的正确性检查。
2. **适配版 TransformerEngine Userbuffers**：复用旧 UB worker，用精调
   cuBLASLt 计算每个 slab / K shard，TE P2P 原语与之重叠。QKV 包含全部
   slab GEMM、send/recv、unpack；OProj 包含 pack、send/recv 与全部 beta=0/1
   GEMM 累加。所有通信 stream 在 stop event 前 join 回主 stream。
   这是你原来的 UB 适配方案，不是 TE 发布包直接提供的 Ulysses 融合算子。

BF16 NT cuBLASLt 使用 64 MiB workspace、最多 64 个 heuristic 候选，
每个候选至少 10 次预热 + 50 次本地计时。UB 的计算 SM 预算为
`实际 SM 数 - comm_sm`，本机为 `148 - comm_sm`，不写死 132。
当前沿用旧 heuristic 候选调优器，不声称穷举了 cuBLASLt 全部算法配置。
SM103 计划器通过每作业 `FUSE_CUBLASLT_TUNE_GRAPH=0/1` 分别在 Eager 或
Graph replay 下计时并选择 GEMM 候选，不把 Eager 选出的算法直接当作 Graph winner。
每 rank 的元数据保存实际调优模式；未设置该变量的历史 SM90 调用仍保持 Eager 调优。

## 计划、调优与正式复测

默认复用每方向 8 个模型/人工 geometry × 4 个 S（16K/128K/256K/512K）× CP4/8 = 64 点，
完整记录 Hq/Hkv/D 和 MNK。Eager/Graph 分开调优、分开选择 winner。

- `smoke`：固定小型 production Qwen geometry、S=1024、默认 CP8，每条边界、
  每种基线、每种 launch 各一次配置，共 8 个作业。
- `sweep`：NCCL 扫 48 个 channels/chunk/LL tuple；UB 扫 6 个 comm_sm。
  扫描 `4,8,12,16,20,24`；适配版支持1–32的任意整数通信 SM预算。
  固定上游版本的 P2P 拷贝分块使用了仅适用于2的幂的位运算，且 push 接收完成
  计数误用了布尔或；本仓补丁分别改为整数除法对齐和正确的发送CTA计数。
  不把这些实现缺陷作为永久的策略搜索限制；必须使用带修复的 TE 构建。
  每个配置分别测 Eager 与 Graph。全默认矩阵共 13,824 个作业；建议先用
  `--models production_qwen_dense --seqs 1024 --cps 2` 验证整个流程。
- `refine`：各组前三名继续搜索 pack block/warps 和 stream priority；UB 搜索
  streams、push/pull、CE/SM、pack、方向及 QKV local-first。
- `formal`：合并 sweep/refine 后每组 top-3 统一 10 warmup + 50 samples。

2026-09-06 的随机输入/稳定性合同同时适用于 smoke、sweep、refine 和 formal：
每候选至少 10 次预热、50 次采样；Graph 捕获前后分别预热。输入为 BF16
`U(-0.125,0.125)`，权重为 `U(-0.02,0.02)`；QKV seed=3109+rank、
OProj seed=2701+rank，权重统一广播 rank0。预热/计时使用同一批随机数据，
不在计时区生成随机数。OProj UB 在 A2A 之前选 GEMM 算法时，先用真实随机
activation 填充临时接收 slab，禁止读取空白或未初始化数据。逐rank记录输入
非零比例、采样均值/标准差、范围、形状和种子；算法缓存仍独立于 launch/SM预算。

每进程首次使用或距上次计时超过2秒时，增加至少100ms CUDA事件时间的预热；
常驻批次不为每候选重复支付此预算，但始终保留逐候选预热和三窗口收敛检查。
各rank最后三窗口波动须≤5%，上限5秒。再按正式 barrier/event 节奏预热，
收集50样本，要求前后半段p50差异≤5%；最多三轮，取首个稳定轮而非最快轮，
所有原始轮次保存在rank metadata。该判据是可审计的稳态筛查，不等于长期热平衡证明。
每批独立记录200ms间隔的时钟、功耗、功率上限、温度、利用率；不锁频、不改功率限制。

最长序列的参考校验按tile/独立A2A处理，不在每卡all-gather全部输入；
大输出FP32误差比较分块执行。完整QKV边界不分配未使用的QK/V-only辅助缓存。
这些调整不删除shape、不放宽数值阈值，也不改变被计时的GEMM/A2A边界。
旧3+12、默认TE随机权重及未记录预热稳定性的结果保留为探索数据，不混入新合同winner。
- `summary`：只读当前 formal 原始记录，重算 p50/p95，输出每组最快候选与
  raw 路径。TFLOPS 使用 p50 对应的 `2MNK / time`。

每个作业用独立 torchrun 进程组，防止 NCCL 环境参数被之前的 communicator
缓存。正式数据是逐样本 CUDA event → `dist.MAX` → p50/p95；Graph capture
和预热 replay 在采样外（沿用旧 Python worker 的 Graph 初始化流程）。
没有额外的 SM103a fused launcher，因此不输出融合加速比。

```bash
# Mac 可以离线生成计划，不导入 Torch/TE，也不启动 GPU 任务。
python3 benchmarks/sm103/bench.py --stage smoke \
  --results /Users/admin/workspace/fuse_midfile/sm103-plan

# 依赖补齐后在 L20D 上执行；Python 指向自有目录的环境。
python3 benchmarks/sm103/bench.py --stage smoke --execute \
  --python /root/workspace_wct/bench-env/bin/python \
  --results /root/workspace_wct/fuse/results/sm103/baselines

# --stage sweep/refine/formal/summary 使用同一组 shape/设备/路径参数。
# 不传 --execute 时，前三个阶段只生成命令计划。
```

`--directions qkv,oproj`、`--backends cublaslt_nccl,te_ub`、`--models`、
`--seqs`、`--cps`、`--devices`、`--sm-count`、`--library`、`--te-root`
均可显式指定。CUDA Runtime preflight 会验证 compute 10.3 和 SM 数；
只规划时 `--sm-count` 是预期值，实际执行不会仅凭 nvidia-smi 判断架构。

执行前采样所选 GPU 的利用率与可用显存，并记录 GPU UUID、时钟与功率。
允许模型驻留，不停止其他进程；有计算负载时退出。短时检查无法保证后续没有
其他计算进入；这类共享设备结果应结合原始观察记录复核。2 GiB 余量检查只是
基本门槛，不是大 shape 的精确显存预算，长序列需按当前可用显存安排。

计划对 baseline 源码、共享库、设备列表和预期 SM 数做指纹校验；各 rank 保存
实际 CUDA/torch/TE 版本、模块路径和设备属性。修改源码、重编译、换软件环境
或卡组后使用新的结果目录，不续用旧调优结果。历史 H200 Golden 从不作为输入。

## TE UB 固定版本与适配

旧 worker 需要 `CommOverlapP2P` 上的 `configure_userbuffers_p2p`、
`userbuffers_p2p_send`、`userbuffers_p2p_recv`、`get_userbuffers_send_stream`
等扩展。基线固定官方 TE commit `a7aec214...`，并在 native Userbuffers
`CommOverlapP2P` 上增加经过范围、peer、对齐和 stream 下标检查的薄 adapter；
保留 Userbuffers 传输方案，同时修复该固定版本 P2P 的任意 grid 拷贝边界和
接收完成计数；结果明确属于适配版 TE，不冒充未经修改的官方发布包。
构建设置 `NVTE_CUDA_ARCHS=100`，TE 将其展开为
Blackwell generic 与 `100a/103a` 专用目标。`worker.py --preflight-only` 会验证
包、接口、Runtime compute 10.3 和 148 SM，不悄悄替换成另一条基线。

## Profiling

复用原 micro test 的命令：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/profile_sm103a_micro.sh \
  /root/workspace_wct/transfers/bootstrap-20260905/blackwell_probe_sm103a \
  /root/workspace_wct/profiles/micro-new-run
```

输出必须是新目录；生成 `.ncu-rep`、`.nsys-rep`、SQLite 和 CSV。
NCU 禁用自动锁频与 cache flush；NSYS 只采应用的 CUDA/NVTX，不做 CPU
采样或系统范围 context-switch 跟踪。NCU 默认重放适用于本次独立 micro test；
未来跨 GPU ready/epoch 融合内核不能直接套用单 kernel replay，须另行设计采样。

本次报告已通过云盘取回：
`/Users/admin/workspace/fuse_midfile/profiles/micro-20260905/`。
对应远端 `/root/workspace_wct/profiles/micro-20260905/`，云盘
`arsenal-it-bucket/arsenal-it-bucket/wct/fuse/profiles/micro-20260905/`。
这些 profile 数据仅用于诊断，不参与正式性能统计。

```bash
python3 -m unittest discover -s benchmarks/sm103 -p 'test_*.py' -v
```
