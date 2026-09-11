# SM103 Ulysses forward baseline bench

## MXFP8 QKV forward baseline (v20 development)

The public operands are **prequantized MXFP8 activations and BF16 master weights**:

```text
MXFP8 X [M,K] + scales ────────────────────────────────┐
                                                     ├─ MXFP8 GEMM
BF16 W [N,K] ── communication-CTA quantization ─ ready ┘
                    FP32 accumulation → BF16 epilogue → Q/K/V A2A
                          ↑                                  │
          quantize subsequent W while waiting/routing outputs ┘
```

W is freshly quantized on every full call; both sources remain unmodified. Each
32-element block uses E4M3 values and one UE8M0 power-of-two scale, choosing
the smallest representable scale that avoids overflow for finite inputs.
The scale tensors are written directly in the pinned CUTLASS native layout.
GEMM uses native block-scaled MXFP8 Tensor Cores, not BF16 computation.
The optional activation adapter runs upstream, outside this operator's boundary.
The public activation-size query includes native SFA padding. Arbitrary compact
scale tensors cannot be passed without conversion to the required layout.

The first implementation uses an explicit M128/N256/K128 collective with an
M128/N64 epilogue, 1-SM cluster, and caller-selected communication CTA budget,
raster and swizzle. It reuses the existing BF16 output routing and ready
protocol; it does not reuse BF16 performance curves to select MXFP8 parameters.
The K dimension must be divisible by 128, head_dim must be 128 for this TMA
communication-side quantization path, and existing GQA/CP route constraints
still apply. Special MLA/KDA routing is not added by this precision path.

`fused_mxfp8[_mpi]` reuses the forward harness rather than duplicating its
validation or sampling loop. The normal Graph boundary is **one persistent
kernel including weight quantization, GEMM and A2A**. It does not include
upstream activation quantization. The optional `--mxfp8-prequantized` diagnostic
prepares W separately and excludes weight preparation; report it separately.
There is no implicit persistent weight cache. The `precision,` record identifies
the actual collective and whether quantization is included.

Default `--mxfp8-weight-preparation comm` reuses all communication warps: bounded
SIMT chunks execute during output-ready waits and after issuing local/peer TMA.
Every warp drains its remaining assigned weight work before role completion,
even when it owns no output tasks. A full N256-by-K panel publishes one ready
flag; all M tiles reuse that panel, and the mainloop caches its acquire across
prologue/remainder. No per-32-element/K-stage wait is added to GEMM.
`--mxfp8-weight-preparation all` is an explicit control: all resident CTAs
quantize first, synchronize, then start their normal roles. It is not overlapped
quantization/GEMM. Per-call counters are reset inside the cooperative kernel;
scratch need not be zeroed and Graph replay cannot reuse stale panel readiness.
First-use N order comes from the resolved raster/swizzle geometry, with optional
N-band rotation; publication still requires all chunks, not logical task order.
No model-name or per-shape weight cache is introduced. BF16 bindings retain
their original no-input-work path and need no new prologue or grid barrier.

The complete output is checked against cuBLAS using independently reconstructed
quantized operands, then the full A2A route is checked. This validates arithmetic
and routing on the represented MXFP8 values, **not** model-quality equivalence
to unquantized BF16. Two reproducible random payloads change both X and W to
detect stale quantized storage. Formal sampling remains 10 warmups + 50 samples;
BF16-specific tuning diagnostics are not accepted by this harness. MXFP8 uses
the existing QKV profiling protocol and supports explicit C/R calibration below.

Through the existing Mac → mc → main screen controller:

```bash
python3 scripts/l20d.py run fused-build --node 09 --mpi --mxfp8 \
  --experiment v20-mxfp8-bringup
python3 scripts/l20d.py run fused-smoke --node 09 --mpi --mxfp8 \
  --fused-launch graph --fused-direction qkv --world 8 \
  --global-seq 2048 --hidden 1024 --q-heads 32 --kv-heads 8 --head-dim 128 \
  --comm-sm 8 --qkv-policy m128n256 --input-generator gpu_philox \
  --experiment v20-mxfp8-bringup
```

Validation status: local host contracts pass; CUDA compilation, multi-GPU
correctness and performance are **pending**, not a released baseline result.

## OProj ready / MMA / epilogue diagnostic

`l20d.py run fused-smoke --profile --profile-detail full --directions oproj
--oproj-pipeline-probe` adds a private pipeline probe; supply the same explicit
shape, tile, communication CTA budget, raster, and swizzle as the uninstrumented
comparison. First build with `l20d.py run fused-build --profile`. The probe is
single-process Eager only, never an accepted MPI Graph performance sample.

Only GPU0 workers 0, 1, and the start of the final swizzle-sized worker group
are sampled (deduplicated for small grids). Buffers are bounded to 64 MiB.
Other GPUs still run and undergo complete numeric/route validation. Ready
checks retain the original whole-peer granularity; K-stage timestamps observe
existing MMA input waits without adding ready checks, fences, or GPU barriers.
The selected BF16 CUTLASS `mma()` needs a small diagnostic mirror because its
concrete pipeline parameter would slice an observer subclass. Keep this mirror
aligned with the pinned CUTLASS implementation when upgrading dependencies.

`scripts/export_sm103_oproj_perfetto.py` exports the verified receipt to a new
JSON. Load-warp checks, MMA stages, TMEM-slot waits, and epilogue phases appear
directly below their owning CTA. Submission timestamps are **not** tensor-active
intervals; epilogue observation of the existing accumulator barrier is an upper
bound on MMA completion. An input-wait span may overlap earlier asynchronous MMA.

Cross-check with the existing **unprofiled** `--calibrate --fused-counters fused`
NCU path (GPU0 only; all peers active), separately for rows/columns with matched
geometry and budgets. Preserve concurrent range replay, no clock locking or
cache flushing. Tensor-active, memory and NVLink counters describe the measured
GPU/range, not a particular CTA or a time-aligned slice of the separate Perfetto
capture. Quantify probe overhead before attributing differences to production.

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

纯 GEMM 的计算预算对照：`scripts/l20d.py run gemm-probe --launches graph`
支持 `--gemm-sm-budget N`，其中 N 应取待比较融合配置的实际计算 CTA 预算
（例如 Runtime 148 SM、通信 16 CTA 时为 132）。不传该参数保留满卡对照。
每个预算重新选 cuBLASLt 算法，仍使用随机输入、选算法及正式采样各至少 10+50，
不按 SM 比例缩放已有耗时。只比较同物理 GPU、同 M/N/K/精度/布局/Graph 边界；
单 GPU Lt 不能直接与自研跨 rank 最大耗时作同卡 GEMM 归因。

`--gemm-sm-budget` 创建进程内 CUDA green context，查询并要求实际分配 SM 数
与 N 完全相同，同时核实 stream 绑定的资源。选算法、Graph capture/replay 和计时
均使用该 stream。无法精确分配时失败，不静默取整。正常分组不能表示 N 时使用
`IGNORE_SM_COSCHEDULING`，记录 split flags 与实际 co-scheduling alignment；
该选择可能限制大 cluster 算法，结果不能隐藏这项条件。没有配置 MIG/MPS 或系统
环境；已设置 MPS active-thread override 的进程拒绝这项实验。

只想验证库的提示参数可用 `--cublaslt-sm-target N`（显式 0 是满卡审计）：
`CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET` 只是启发式提示，不是硬件 SM 分区。
两种显式入口都在计时后保存**实际测量 Graph**的 verbose DOT，包含 kernel
grid/block 等信息；CTA 网格不等于驻留 SM 数。资源限制版记录
`sm_budget_diagnostic`，仅提示版记录 `sm_target_diagnostic`。
非零 `tuning.math_sms` 的诊断不会通过满卡基线表的导入检查，不覆盖既有全量参考。

2026-09-08 验证：Qwen3-235B OProj 每 GPU M16384/N4096/K8192、BF16 Graph，
node09 GPU0，green context 实际分配 132 SM、stream 资源复核通过，split flags=1、
co-scheduling alignment=2，8/8 Lt 算法可运行。随机输入、10+50、4096 点数值校验
通过；run `20260908-222829-c5a6c7`。这不是完整矩阵更新，也不是融合 CP8 测量。

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
# 独立 OProj GEMM 片间空隙诊断

独立本体调参使用 `fused-smoke --mpi --fused-launch graph --fused-direction oproj
--calibrate --compute-only`：只启动 `compute_reference`，跳过 F 与 copy reference。
输入提前物化，计算 CTA 预算由 `148-comm_sm` 明确给定；此处 comm_sm 是预留预算，
并不启动通信 CTA。同 shape 的 tile 候选复用进程、输入和校验器，每个候选两种
随机 payload 完整校验、至少10+50；独立选优不能声称融合/overlap也有相同收益。

`l20d.py run fused-smoke --profile --oproj-gap-probe --directions oproj`
复用既有单进程 Eager profiling 入口、完整校验和显式 tile/计算 CTA 预算。
本选项额外物化完整 lhs，测独立 `compute_reference`，不把融合的 ready 等待
混入独立 GEMM。GPU0 的全部计算 CTA/逻辑 tile 都记录；其余 GPU 不插细粒度点。
只记录 tile 边界，不逐 K-stage 输出，不生成大型 Perfetto 文件。

每条 CTA 链定义 `A_i=第一份 K 输入可消费的时间`、
`B_i=epilogue 观察 accumulator 可消费的时间`；片间间隔为
`A_(i+1)-B_i`，保留负值以表示观察区间重叠。`B_i-A_i` 仍含 tile 内部
输入/流水等待，不是纯 Tensor Core 执行时间。输出每条链的 service、signed gap、
正 gap、首尾时间与 tile 数，逐链校验时间闭合，禁止把并行 CTA 的 gap 相加当 E2E。
采用最后观察完成的 CTA 作为计算关键链的近似，并不证明它就是 kernel 关键路径。

插桩前后各 10+50，额外至少 100 ms 预热；输出扰动和样本半段稳定性。
`scripts/summarize_sm103_gemm_gaps.py --log <log> --output <json>` 汇总。
`--table <abc.json> --run-root <l20d目录> --output <json>` 按 M/N/K/预算映射
历史表，另出 CSV。此处是 Eager 诊断与历史 Graph 的数量级比较：
`gap占比/(1-b/a)` **不是已测出的损失归因比例或严格上界**，因为没有测到
cuBLASLt 自身逐 tile 的 gap，且插桩、epoch、launch mode 均可能带来差异。

`l20d.py run gemm-probe --cublaslt-counters --launches graph` 可对单一 BF16
geometry 的已预热、已选算法 Graph 采集 NCU，并保留 `--gemm-sm-budget`
green context。NCU 使用现有六项计数器及同一导出/校验器（输出文件沿用
`cutlass-counters.csv` 名称，contract 明确 backend=cublaslt）。结果是 diagnostic，
不可替换正式性能表；Tensor 活跃度不能直接换算成片间 gap 时间。

## OProj：生产消费统一性能模型

v17.0 保留离线选优和显式参数；v18.0 在此基础上提供运行时通信 CTA 模型，
[验证结果](../../results/sm103/v18.0/README.md)与 v17 发布表分开。
SM90、ready 粒度、device kernel 与同步协议保持不变。
主机选择器不使用模型名或逐 shape 赢家表，也不在每次 launch 前实跑候选。

核心同时约束两件事：**数值上的生产/消费速率**，以及**位置上的交货/消费
布局和顺序**。SM103 的旧固定 GEMM `auto` 入口不等于运行期成本模型；
本版通过通信预算 0 显式请求新模型，不改变旧的默认 GEMM tile。

`fused_model.py::score_oproj_schedule` 保留 tile、计算 CTA、整数波、启动与收尾项，
并显式展开与算子一致的调度：

```text
candidate = GEMM tile/raster/swizzle + communication CTAs + cohort
  compute CTAs = physical SMs - communication CTAs
  GEMM schedule -> first-use windows -> copy-slot-bounded delivery queue
  copy slots -> last chunk of each (M, peer) -> complete-ready time
  compute worker: tile c, c+compute_CTAs, ...; consume peer 0,1,... in K order
  predicted finish = critical dependency path, not a sum of parallel waits
```

每个预算使用**对应实测的 GEMM 服务曲线**，不假设吞吐与 SM 数线性增长。
增加通信 CTA 同时会改变计算预算、窗口、copy 槽位与 cohort，必须重新评分。
通信服务标定包含独立 A 搬运（本地和远端），不是仅远端 NVLink 带宽。
按真实队列计算各 copy slot 的字节量，使用 `max(slot_bytes) / R_time` 得到
等效单槽服务率，目标队列再计算自己的槽位尾差；不能先用 `A_bytes/R_time`
作为槽位总服务率、再重复计入标定尺寸原有的槽位不均衡。
同一个 `(M, peer)` 被多个 N tile 复用，交货量只计一次；所有原 chunk 完成
才可发布 ready，不引入 K 细粒度信号、额外 fence 或全局队列屏障。

多 N-band 的 AlongM 在第一段就需要全部 A，但只完成约 `swizzle/N_tiles`
的 GEMM；因此总量上的 `T_comm < T_gemm` 不足以证明不会断供。模型同时输出
这个阶段的生产/消费比，以及真实队列映射形成的预测关键路径等待。AlongN
按实际 M 分组的 N 复用轨迹处理，不套用 AlongM 的前段放大系数。

校准只使用独立计算 C、独立通信 R 或明确定义的 profile 服务数据；融合 F
仅作预测验证，不按模型名、序列标签或融合赢家表拟合。完整 C/R 边界折算为
有效周期和带宽时，启动/收尾仅被摊销，不能再额外相加或称为已测得零开销。
各 peer 的计算服务均分、固定 copy-slot 服务、未模拟资源竞争反馈是当前
明确的近似；预测误差与留出尺寸结果必须单独报告。Profile 诊断与 Graph
性能不直接相减，1+5 快筛不能覆盖为正式 10+50 bench。

### v18 主机选择与显式覆盖

实现保持三个职责：`detail/model_calibration.cuh` 只存独立 C/R 服务点；
`detail/performance_model.cuh` 复用算子的生产消费映射计算完成时间；
`detail/autotune.cuh` 枚举允许的候选并缓存选择结果。CPU/Python 使用同一
评分语义，冷选择只发生于新物理请求，16 项线程局部缓存不包含 tensor 指针或 epoch。

当前标定域：Runtime CC 10.3、148 SM、BF16/FP32/BF16、CP4/8、causal rows-pull；
GEMM 为 `m128n256` 或 `m128n256k64e32`，有效 swizzle 为 4/8，通信 CTA 为
8/16/24/32/48。K 仅在 8192–16384 的独立标定点间插值，每个通信/计算预算
使用自己的服务曲线；M/N 改变会重算实际 tile、padding、队列和波数，不改变
标定数据。K 域外、不同实际计算预算或未标定 collective 不偷偷外推。

公开入口第一步只自动选择通信 CTA，GEMM tile/epilogue、raster/swizzle 仍由
调用者显式指定。`A2AGemmParams::num_comm_ctas > 0` 保持原路径；设为 0 时
以当前 GEMM 配置请求模型选择。`recommended_a2a_lhs_gemm_comm_ctas` 与生产、
role telemetry、copy reference 共用 resolver，查询返回 0 表示不可自动选择；
自动 launch 对域外请求返回 `cudaErrorNotSupported`。不要把查询失败的 0
当作一个合法显式预算。默认 GEMM `auto` 仍选 N128，不暗改旧的 shapeless
traits/内存分配合同；N128 当前需要显式通信预算。

独立 GEMM reference 的 `reserved_comm_ctas=0` 仍表示全 SM，不表示自动通信。
同预算对照需先查询正预算并显式传给 reference。输入的外部 ready acquire
保持不变；标定以所有上游输入已经发布为前提，不估算未完成上游的额外等待。

内部主机选择器也能联合预测两种 collective × 两种 raster × 两种 swizzle ×
五档通信 CTA（最多40候选）。它的联合预测与公开入口“固定 GEMM、自动 CTA”
是两种范围，不能混称。TODO：留出尺寸验证后，通过 shape-aware 的公开 plan
接口统一 allocation/query/launch 的联合选择；再扩展 QKV、精度和新标定域。
无 shape 的资源查询不能依赖“上次某个 shape 的缓存结果”。

验证先冻结 C/R 标定、候选集和预测，再运行未参与标定的矩阵。固定 GEMM 的
五档通信实测可衡量 Top1/Top2 相对该五档最优的损失；仅测联合 Top2 不能声称
40候选穷举最优。控制点与留出点分开，显存不足留空，预测误差不回填成该点的
特殊规则。所有性能数值必须来自已完成的完整边界校验，主机测试不替代 GPU 验证。

本轮固定 GEMM 的公开自动入口与 v17 最终显式预算重放完成23点、46次A/B，
Graph 1+5、两组随机payload完整校验。Auto相对显式配置几何平均−1.21%；
Qwen3 CP8/512K的32→48 CTA选择退化10.62%，没有按模型名覆盖该选择。
7点存在采样漂移，7个历史缺测未重试；这些是快测，不是正式稳定性能承诺。
控制器可用 `--auto-oproj-comm` 请求真实零预算入口，不能同时给显式通信预算，
也不能与profiling/独立C/R标定混用。首次主机查询和Graph构建不计入Graph吞吐。
# MXFP8 QKV independent GEMM search

`l20d.py run build --mxfp8-gemm-search` builds an isolated target; run
`gemm-probe --gemm-precision mxfp8 --mxfp8-gemm-search --gemm-sm-budget 132
--launches graph --gemm-matrix <matrix.json>` through the normal controller.
Use the current ordinary-user workspace explicitly on both commands.

The matrix is `{schema: "sm103_gemm_matrix_v1", shapes: [{id,m,n,k}, ...]}`.
The search reuses the production MXFP8 collective and persistent scheduler,
with prequantized operands, BF16 output and no quantization/communication or
ready polling. The 132-CTA cap and one-resident-CTA SMEM allocation model the
compute budget, not fixed physical SM IDs or communication interference.
Production remains M128/N256/K128/E64/automatic stages and is not overwritten.

Grid: M128 × N{128,256} × K{128,256} × epilogue-N{32,64} ×
Along{M,N} × swizzle{1,4}, automatic mainloop stages: 32 candidates.
Top two stable candidates expand to missing swizzles 2/8 and explicit stages
2/3, at most two neighborhood rounds, deduplicated across rounds. K256 only
admits explicit stage2 at N128; N256/K256 stays on Auto (its E64/stage2
combination exceeds SMEM capacity).
Each candidate uses converged warmup, Graph10+50, the first stable round
(half-sample drift ≤5%), and full decoded-operand GEMM validation on two
random payloads. Results are not accepted until the second payload passes.
Allocations, operands and reference are reused within each shape; candidates
and shapes share one process. Resource/memory skips and unstable candidates
are explicit; failures are not converted to winners. Reports call the result
candidate-set best, not a global optimum or measured fused speedup.
# MXFP8 GEMM-driven fused tuning

`scripts/tune_sm103_mxfp8_fused.py` consumes the audited pure-GEMM search
summary and the existing forward matrix. Its default invocation only prints a
plan; `--execute` runs the normal L20D controller. Supply `--current`, `--gemm`,
`--output`, and the current ordinary-user `--workspace` explicitly.

The MXFP8 public parameters expose `epilogue_n=32|64` (default 64 keeps the old
baseline); `projection.gemm.raster/max_swizzle_size` and
`projection.num_comm_ctas` remain explicit. The registered M128/N256/K128
families use auto stages, and do not change complete-panel/tile ready units.
The same selected family is used for launch, resource queries and profiling.

Like the BF16 GEMM-driven fusion path, the resolved producer tile order drives
the output-copy dependency inverse. Weight panels follow their first-use N
order in that same schedule. Raster/swizzle and communication budget are
joint offline candidates, not independent winners pasted together: start at
the pure-GEMM winner, scan communication budgets, probe neighboring layouts
with nearby budgets, then refine the best fused candidate's budget. Each
candidate uses Graph 10+50, nonzero random operands and full two-payload
numerical/routing validation. A same-binary E64/M/sw1/comm16 control makes
the old/new comparison explicit. `summarize_sm103_mxfp8_fused.py` audits raw
samples, archive receipts and per-rank ownership before accepting a result.

This is bounded offline tuning, not a global-optimality claim or an MXFP8
runtime performance model. BF16 service-time coefficients are not reused:
MXFP8 additionally produces quantized weights on communication workers, and
would require its own measured quantization/compute/copy calibration. No
model-name lookup or fused winner is embedded in the kernel.

For an explicitly shortened search, `--resume --fast` reuses completed,
same-binary/same-environment candidate evidence and prioritizes up to two
layouts already successful for the same N/K family. It probes neighboring
budgets at every remaining shape and expands regressing cases. This reduces
candidate count, not matrix coverage, warmup, samples or validation. Such rows
are marked `family_seeded_joint_pool`; this narrower search is not exhaustive.

## MXFP8 QKV communication autotune

The interface follows BF16: `projection.num_comm_ctas=0` requests automatic
communication-budget selection; positive values are strict overrides.
`recommended_gemm_a2a_mxfp8_comm_ctas(problem, route, epilogue_n)` uses the
same resolver as production and telemetry, returns a positive budget on
success, and returns zero for unsupported geometry/calibration/device state.
The query is shape-only. It does not inspect tensor pointers, allocate GPU
memory, encode descriptors, benchmark candidates, or launch kernels.

**The offline policy passed the declared same-run acceptance matrix on
2026-09-11.** All 33 physical points have independently confirmed manual
configurations replayed beside true Auto in the same binary/job. Graph 10+50,
two random payloads and full numeric/routing validation passed. Auto/manual
geometric mean is 96.6929%; the minimum is 90.0858% (QwenDense CP4/512K).
The user's report-only gate is geometric mean >=95% and every point >90%.
The minimum is close to the boundary, not a guaranteed 90% performance floor.
The 11 calibration-sequence points average 95.4402%; the 22 held-out sequence
points average 97.3253%. These are measured retention ratios, not prediction
accuracy or proof for unmeasured shapes.

Keep the two unique result pairs under `mxfp8-v20/autotune/` in the Mac
midfile directory: `acceptance-current.json/md` and
`manual-best-current.json/md`. The latter retains confirmed offline SOTA,
exact configurations, all 50 samples and provenance without search history.
Historical measurements remain separate: Dense CP4/512K retains only 88.3915%
against its older saved throughput, versus 90.0858% against this run's replay.
Do not claim every point exceeds 90% against the historical snapshot.
BLOOM CP4/512K's previous OOM left no manual reference; it is now filled by a
finite 16/24/32-budget search and independent paired confirmation, without
changing the policy. Qwen3 CP8 remains outside the adapted routing scope;
Kimi is QKV-only. Qwen72/Llama70 share a physical measurement, not two samples.

The formal binary SHA256 is
`14ba624ba3bdcde405a8aa53837fc3b2eb83cde326c8d061511f6ad0b966041b`.
Its calibration identity retains the original `validation_pending_6714eeeb...`
label so the tested binary is not modified just to rename a provenance marker;
the acceptance artifact, not that label, records the validation outcome.

The
registered physical layouts are calibrated independently at communication
budgets 16/32/64; the maintained calibration artifact records actual coverage,
unavailable observations and binary identity. S128K is the calibration
sequence; S256K/S512K are predeclared holdouts, not inputs used to fit a winner.
Unsupported physical keys return `cudaErrorNotSupported`; the manual path
remains available. Calibration, Auto performance and bitwise-repeatability are
separate checks. Do not insert synthetic coefficients, BF16 timings or fused
winner tables to fill missing domains.

The maintained `acceptance-current.json/md` artifact records the actual formal
coverage, paired same-run manual budget, historical reference, exact layouts,
model version and missing rows. A model query succeeding does not certify its
performance prediction or generalization. Probe diagnostics
still distinguish Q-only and publishing mixed slots, but individual publication
latencies do not enter the new decision. They depend on concurrent traffic;
more context fields do not prove coverage of future concurrent states. Formal
held-out comparisons remain necessary. No correction fitted to fused winners
is enabled.
Online tuning is explicitly out of scope: the first production invocation must
resolve its budget from the compiled offline strategy without candidate GPU
launches, timing feedback or runtime calibration.

The coarse offline policy uses directly observed compute tile
first/cycle services for `C`, and whole local Q+R/R services for `P`. For a
calibration row count `M0` and a larger row count `M` at the same N/K/layout,
the rule uses `P(M) = QR(M0) + (M/M0 - 1) * R(M0)` and
`C(M) = tile_first + (ceil(valid_tiles/(148-c))-1) * tile_cycle`.
The score is `startup + max(C(M), P(M))`. Weight work is fixed, not multiplied by sequence
length. `QR-R` is not labeled quantization latency and is not clamped to zero:
the mixed workload may change traffic pacing. R here is an aggregate effective
output service, never a per-TMA latency; C uses individual observed tile
intervals, not whole-C divided by waves. Local Q/R completion excludes cross-rank
finalization. Extrapolation to 256K/512K remains a hypothesis requiring formal
validation. It assumes sufficient overlap: neither first-panel stalls nor
changing fused contention are exactly simulated. Unmodeled stage predictions
are reported as unknown, not zero. The implementation and its host reference
must agree before formal GPU acceptance; host tests alone are not acceptance.

The private API mirrors SM103 BF16's `TuningRequest`, `TuningResult`,
`select_*_plan` and bounded success-only cache. GEMM tile/epilogue, resolved
stages/raster/swizzle and resource footprint are inputs, not auto outputs.
Each candidate communication count `c` selects independently measured services
for compute stride `C=148-c` and communication-warp stride `8*c`. The kernel
continues to use the original shared scheduling map:

```text
GEMM scheduler                    shared communication warp
  tile j, j+C, ...                  wait output -> quantize one chunk
          ^                         issue G2S -> one chunk -> wait G2S
    full N256*K ready               issue S2G -> one chunk -> wait slot
          |                         next output; finally drain own W work
  last panel contribution <---------+
          |
          +-> GEMM -> full output tile ready -> BF16 route -> remote drain
```

`ProducerTileOrder` supplies the resolved CUTLASS mapping; `ConsumerTileOrder`
assigns copies to their last logical dependency, while execution still acquires
all dependencies. No dynamic work queue, new device polling point, changed
publication granularity or new WASP mode is introduced. Current registered
geometry is M128/N256/K128 with E32/E64. Other GEMM N tiles require an explicit
weight-panel dependency adapter before entering this model.

The host scorer evaluates all applicable measured budgets using constant-work
arithmetic per candidate. It does not simulate publication/DMA events or launch
GPU candidate trials. Only three direct compute/startup intervals and whole
R/QR spans enter the compiled decision table; finer diagnostic observations
remain in calibration JSON. A success-only bounded cache avoids repeated queries.

Missing physical anchors are not guessed. Communication budgets and layouts
must match measured calibration; K interpolation requires compatible explicit
brackets and a common interpolation group. Prequantized diagnostics require a
positive budget: query auto production first and reuse its result, rather than
score a no-quantization boundary with quantization services. Formal validation
must use the existing 10+50/random/full-routing protocol and compare against
held-out hand-tuned cases; synthetic host tests prove model contracts only.

### Calibration is measurement, not fitting

The selection rule is `argmin_c predicted_completion_time(c)`. There is no
trainable weight, penalty, shape-specific winner, or threshold selected from
fused performance. Every candidate reconstructs the actual scheduler with
`compute_ctas=148-c` and `quant/copy_workers=8*c`. Model names and sequence
labels belong only to the report, never to the selection key or score.

Service values must be durations of identified operations in independent
experiments. For example, two successive complete output releases by the
same compute CTA measure its release interval; dividing whole-kernel time by
waves does not measure that interval. A deliberately delayed weight panel
measures response after a dependency becomes ready, not an unexplained
coefficient inferred from fused slowdown. Delay lengths are experimental
stimuli, not parameters optimized to select a preferred communication budget.

For diagnostic mixed quantization/TMA slots, both branches use the same time
origin; nested intervals are not added twice. The coarse decision instead uses
the measured whole QR span once. The original complete-panel ready unit is unchanged.

Keep the primitive evidence and held-out fused validation separate. A new
M/N reconstructs its work and dependencies instead of choosing a recorded
winner. Expanding a calibration domain needs independent evidence; unsupported
resources are not silently extrapolated. A failed prediction is reported and
investigated through an observable missing dependency or resource interaction,
never corrected by fitting its error into a scale factor. The acceptance table
must distinguish fixed-GEMM Auto/manual comparisons from a historical offline
winner that used a different GEMM layout, and must preserve missing results.

### Independent MXFP8 C/Q/R measurements

Add `--calibrate` to a manual-budget MXFP8 MPI Graph run. The same process and
buffers measure four distinct boundaries for every candidate and validate
both random payloads, with pre/post checks around formal 10+50 sampling:

- `fused`: dynamic W quantization + MXFP8 GEMM + BF16 route.
- `compute_reference`: prepared MXFP8 A/W, no communication/quantization; the
  production collective retains full-output-tile drain/ready publication.
- `copy_reference`: already materialized BF16 output, no GEMM/quantization or
  GEMM-ready waits; includes actual routing and cross-rank completion.
- `quantize_reference`: original communication-warp producer only, including
  its reset/barrier and panel publication, with exactly `comm*8` workers.
  No TMA/GEMM runs inside this timed boundary. Afterwards, C consumes the actual
  Q output for numerical validation against independently reconstructed
  operands, without preparing W again. This validation is outside Q timing;
  it is not a standalone bitwise quantization oracle. Q reset uses `comm`
  CTAs rather than the full fused grid, so it cannot provide F's startup time.

C uses `148-comm` compute CTAs; R uses `comm` communication CTAs. E32/E64,
raster/swizzle, threads and actual dynamic SMEM match the requested production
configuration; separate reference ready/done flags prevent epoch contamination.
Matching these resources does not assert identical register allocation/SASS.
Preparation, allocations, graph capture and validation are outside timing.
Calibration currently requires ordinary `comm` mode and dynamic-weight F.
These aggregate C/Q/R durations diagnose budget tradeoffs; they cannot uniquely
identify the model's first-tile/cycle/publication/mixed-window service fields.
In particular, C divided by waves is not a measured tile-ready latency.
