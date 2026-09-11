# 通算融合 Perfetto profile 协议

这套协议用于定位两个方向的单 kernel 时序：

- `A2A -> GEMM`：通信发布、ready handoff 和 GEMM 消费；
- `GEMM -> A2A`：GEMM 生产、QKV route、cooperative grid barrier 和跨 rank finalize。

v10 的两条反向算子沿用这两个物理方向，但不沿用前向 shape：QKV 反向 B 是
`Head→Sequence A2A -> dX GEMM`，OProj 反向 B 是 `dA GEMM -> Sequence→Head A2A`。
权重梯度 W 是单独的纯 GEMM，不应混进 B 阶段的通信轨道。

正式性能数据必须使用默认构建；profile 数据只解释相对时序。

## 正式启动与计时规则

正式 benchmark 固定区分 `eager` 和 `graph`，两列独立报告，不把两者的差值
当作算子优化收益。

- 正式 MPI benchmark 必须使用 `FUSE_ENABLE_PROFILING=OFF` 构建；profiling
  构建产生的计时和 JSON 只能用于诊断，不能写入 Golden/BENCHMARK。
- `QKV Projection -> A2A` 必须使用 MPI/torchrun 一进程一卡；每个样本先在各
  rank 记录 CUDA event，再对 elapsed time 做 `MPI_MAX`。单进程按 rank 顺序
  launch 会把后续 GPU 的 host 提交时间计入前面 rank 的跨源 finalize，只能用于
  diagnostic trace，不能进入正式性能表。
- 自动配置的正式测试必须把 `comm_ctas=0` 原样传给算子。runner 可以提前查看
  实际选择，用它计算 ready 缓冲区大小并写入 JSON，但不能把这个数替换回计时
  参数；结果必须同时保存 `requested_comm_ctas=0`、最终 `comm_ctas` 和自动方案
  缓存版本。这样 `--resume` 不会把修复缓存前后的结果混在一起。
- QKV Graph 使用一个预上传的 graph、一次 replay；graph 内含 10 个 warmup epoch
  和 50 个带独立 event pair 的正式 epoch。每个 kernel node 使用不同且单调递增的
  epoch。capture、instantiate、upload 与 MPI barrier 全部在计时外。
- `A2A -> OProj` 也分别报告 eager/Graph。其正式 runner、shape、通信 CTA、tile、
  raster、swizzle、warmup 与 iterations 必须写入结果表或配置 manifest。
- TE Userbuffers 的 start/stop event 必须包住完整边界，并在 stop 前把所有通信
  stream join 回主 stream；逐 rank elapsed time 再做 `dist.MAX`。Graph capture 与
  额外 warmup replay 均在正式采样外。
- 所有正式数据均为“逐样本跨 rank 最大值，再计算 p50/p95”，禁止先对每个 rank
  求分位数后再取最大。
- v10 反向正式表必须同时保存 B 与 W 的真实 MNK。普通模式在同一 stream 内记录
  完整 `B→W`；ZeroBubble 模式分别记录 B 和 W，再报告两者之和。ZeroBubble 的 W
  使用 `beta=1` 累加 BF16 `main_grad`，普通模式使用 `beta=0`。
- 反向 Graph 与前向 QKV 一样，把 10 个预热 epoch 和 50 个正式 epoch 放进一个
  预上传 graph，并只 replay 一次。框架若要重复 replay，必须统一重置跨卡状态，或
  更新 graph 中的 epoch；旧 ready/done 不能直接复用。
- 反向正式结果还要测同一卡组、同 B/W MNK 的经典 cuBLAS 纯 GEMM。普通模式的
  W 对照用 `beta=0`，ZeroBubble 的 W 对照用 `beta=1`；cuBLAS 不含通信，只用来
  回答融合总吞吐达到同语义纯计算上限的百分之多少。总时间按相同样本编号，把
  B 与匹配 beta 的 W 两个独立 max-rank 时间相加，不能误写成同一个组合 kernel。

## 构建开关

`FUSE_ENABLE_PROFILING` 默认关闭。关闭时不会实例化 diagnostic kernel，生产 kernel 的参数、类型和热路径均不含打点。

```bash
cmake -S . -B build -DFUSE_ENABLE_PROFILING=ON
cmake --build build --parallel 8
```

采样结束后恢复正式构建：

```bash
cmake -S . -B build -DFUSE_ENABLE_PROFILING=OFF
cmake --build build --parallel 8
```

## 固定采样流程

1. 确认参与 GPU 空闲、P2P/NVLink 正常，记录 `CUDA_VISIBLE_DEVICES`。
2. 固定 `M/N/K`、CP、`comm_ctas`、tile policy、raster 和 swizzle；一次只改一个待比较变量。QKV 必须同时记录GEMM/通信policy请求值、policy模型版本和最终解析出的`BM/BN/cluster`，不能只写`auto`。
3. 使用 `--trace-out`。benchmark 会在普通测量结束后先预热一次独立的
   diagnostic kernel，再清空全部 ready/epoch，以 epoch 1 采集一次正式 trace；
   这样不会把逐 GPU 的首次模块装载误记为跨 rank 等待。
4. exact correctness 必须通过；命令退出码必须为 0。
5. 只保留 Perfetto JSON，不把终端输出当成归档数据。

本轮 `S=128K, N=K=5120` 的标准命令如下；`M=S/CP`：

```bash
# CP4: CUDA_VISIBLE_DEVICES=0,1,2,3; M=32768
# CP8: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; M=16384
CUDA_VISIBLE_DEVICES=<devices> ./build/fuse_bench \
  --mode a2a_gemm_lhs --m <M> --n 5120 --k 5120 \
  --batch 1 --q-heads 40 --head-dim 128 \
  --comm-ctas 4 --lhs-policy <m128n256c2|m128n320c2> \
  --raster n --swizzle 1 --warmup 1 --iterations 1 \
  --trace-out /home/chen/workspace/<name>_perfetto.json
```

QKV Projection -> A2A 同样使用独立 diagnostic launch。下面命令中的
`N=(Hq+2*Hkv)*D`，`M=S/CP`：

前向 QKV 正式路径保持两个环境变量未设置；下列变量只用于固定 policy 的消融或回退复现。

```bash
CUDA_VISIBLE_DEVICES=<devices> \
FUSE_QKV_COMM_POLICY=<pipeline|legacy> \
FUSE_QKV_GEMM_POLICY=<wave_time_model|legacy|m128n64|m128n128|m128n160|m128n192|m128n256|m128n320> \
./build/qkvproj_a2a_bench \
  --mode qkv_gemm_a2a --m <M> --k <K> \
  --batch 1 --q-heads <Hq> --kv-heads <Hkv> --head-dim <D> \
  --comm-ctas <comm> --raster <m|n|auto> --swizzle <1|2|4|8> \
  --warmup 1 --iterations 1 \
  --trace-out /home/chen/workspace/<name>_perfetto.json
```

### QKV MPI role summary

`qkvproj_a2a_mpi_bench` 还提供轻量的 rank 级角色时间汇总。它仅在
`FUSE_ENABLE_PROFILING=ON` 时存在，并且在普通计时与正确性检查结束后运行：先在
每个 rank 预热一次独立 diagnostic kernel，再经 MPI barrier 启动一次使用下一单调
epoch 的采样 kernel。预热和采样都不进入正式样本。

```bash
CUDA_VISIBLE_DEVICES=<devices> mpirun --bind-to none -np <CP> \
  ./build/qkvproj_a2a_mpi_bench \
  --m <M> --k <K> --batch 1 \
  --q-heads <Hq> --kv-heads <Hkv> --head-dim <D> \
  --comm-ctas <comm> --raster <m|n|auto> --swizzle <1|2|4|8> \
  --launch <eager|graph> --warmup 10 --iterations 50 --check \
  --json-out <diagnostic_measurement.json> \
  --role-profile --role-profile-json <role_profile.json>
```

`--role-profile` 打印每个 rank 的汇总；`--role-profile-json` 同时开启采样并写入
结构化结果。两个 JSON 路径必须不同，且都不得作为正式 benchmark 输入。

## 事件定义

所有设备端时间戳来自 SM90 `%globaltimer`，原始单位为 ns；JSON 中 `ts` 和 `dur` 按 Perfetto/Chrome trace 约定写成 μs。

协议版本：10（对应 v12.0）

### v12 FP8 边界

v12 的四条 FP8 路径沿用同一组通信、ready、compute、finalize 和 B→W 条带，
但数据口径是纯 E4M3：输入、权重、通信数据和输出都已经量化为 E4M3，矩阵乘在
FP32 中累加。profile 不包含 BF16→FP8 转换、amax 或 scale 计算；这些步骤由调用方
在算子外完成。比较 FP8 与 BF16 trace 时，必须使用相同的 M/N/K、CP、Graph
边界和 rank 启动方式，不能把量化准备时间只计在其中一侧。

### v10 反向算子的对应关系

| 算子 | B 阶段 | W 阶段 | profile 解释 |
|---|---|---|---|
| QKV backward | 各 source 的平面 dQ/dK/dV 直接写成 destination 的完整 `[M,QKV]`，随后计算 `dX[M,H]` | `dWqkv[QKV,H]=dQKVᵀ[QKV,M]×X[M,H]` | B 使用 `A2A -> GEMM`；ready 从远端 GPU 发布，因此 acquire 是 system scope |
| OProj backward | `dA[M,A]=dY[M,H]×Wo[H,A]`，随后直接把 head 切片写到各 peer | `dWo[H,A]=dYᵀ[H,M]×saved_A[M,A]` | B 使用 `GEMM -> A2A`；W 是独立纯 GEMM |

QKV B 的通信顺序固定为“全部 Q、全部 K、全部 V”。profile 或框架接入不得先用
PyTorch `cat/index_select/permute/contiguous` 重新造一份 `[M,QKV]`，否则测到的是
框架重排加算子，而不是 v10 接口本身。OProj B 也直接发布最终 head-sharded 输出，
中间不应增加独立 layout kernel。

QKV backward 和 OProj backward 都提供 profiling-only 的 role telemetry，覆盖各自
自动策略能够选择的全部 tile。它与模型名称无关，生产 kernel 和自动策略不读取
profile 参数。profile 构建仍不得进入正式表。

### v10 backward B→W Perfetto

`backward_mpi_bench --trace-out` 使用 MPI 一进程一卡同步发射一个独立 diagnostic
epoch，并把 B 与紧随其后的 W 放在同一 rank 的时间线上。B 使用 kernel 内
`%globaltimer`；W 前后各放一个极小的时间戳 marker kernel，因此 W 条带适合看执行
顺序和大致跨度，但正式 W 延迟仍以 profiling 关闭后的 CUDA event 10+50 为准。

通用人工中型 shape、CP8、S=16K 的标准反向命令如下。QKV 使用
`H=5120,Hq=24,Hkv=8,D=128`，OProj 使用 `H=5120,Hq=40,D=128`；两边的
projection width 都是 5120，便于在不绑定任何模型的情况下对照两种数据方向。
命令保留 `comm_ctas=0` 和 `gemm_policy=auto`，验证 profiling 入口与生产自动选择
使用同一通用策略：

```bash
# QKV backward: B=2048x5120x5120, W=5120x5120x2048
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpirun --bind-to none -np 8 \
  ./build-v10-profile/backward_mpi_bench \
  --operator qkv --m 2048 --hidden 5120 --batch 1 \
  --q-heads 24 --kv-heads 8 --head-dim 128 \
  --comm-ctas 0 --gemm-policy auto --weight-mode immediate \
  --weight-beta 0 --launch eager --causal-load-balanced \
  --warmup 1 --iterations 1 --check \
  --trace-out /home/chen/workspace/fuse_v10_cp8_s16k_medium_traces/qkv_backward_perfetto.json

# OProj backward: B=2048x5120x5120, W=5120x5120x2048
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpirun --bind-to none -np 8 \
  ./build-v10-profile/backward_mpi_bench \
  --operator oproj --m 2048 --hidden 5120 --batch 1 \
  --q-heads 40 --kv-heads 8 --head-dim 128 \
  --comm-ctas 0 --gemm-policy auto --weight-mode immediate \
  --weight-beta 0 --launch eager --causal-load-balanced \
  --warmup 1 --iterations 1 --check \
  --trace-out /home/chen/workspace/fuse_v10_cp8_s16k_medium_traces/oproj_backward_perfetto.json
```

反向 Perfetto 的条带含义：

| 条带 | 含义 |
|---|---|
| `B boundary` | QKV 是 `Head→Sequence route + dX GEMM`；OProj 是 `dA GEMM + Sequence→Head route` |
| `compute envelope` | 所有 compute CTA 从最早进入到最后完成本地 GEMM 角色的包络；QKV 中包含 ready 等待 |
| `route envelope` | 所有 comm CTA 的本地存活包络；包含任务排队和 ready 等待，不是裸 NVLink 时间 |
| `compute / route overlap` | 两个包络的交集，只表示单 kernel 内角色同时活跃的时间 |
| `local roles -> cooperative grid sync` | 本 rank 最慢本地角色结束后，等待所有 CTA 到达 grid barrier 的尾差 |
| `system fence` | 确保本 source 的 routed peer writes 在发布完成 epoch 前可见 |
| `publish this source...` | 本 source 向每个 destination 发布本轮完成标记 |
| `wait for source N...` | 本 destination 并行等待各 source；各条会重叠，不能相加 |
| `B kernel completion -> WGrad marker` | B 完全退出到 W 前 marker 执行的交接空隙 |
| `WGrad GEMM` | 独立权重梯度 GEMM；普通模式 beta=0，ZeroBubble 累加模式 beta=1 |
| `route/compute CTA N` | 单个 persistent CTA 的本地角色；名称写明它读取、等待、计算或发布什么 |

### A2A -> GEMM

| 轨道 | 起点 | 终点 | 含义 |
|---|---|---|---|
| `remote A2A` | 通信 CTA 进入 kernel | 该 CTA 全部 warp 汇合并退出 | 一个 persistent 通信 CTA 的完整存活区间 |
| `ready wait` | 计算 CTA 进入 kernel | 第一次观察到可消费 peer shard | 计算 CTA 的首包等待 |
| `GEMM` | 第一次 ready acquire | 计算 CTA 退出 | 该 persistent CTA 的后续区间，包含 WGMMA、后续 peer wait、epilogue 和可能的多个逻辑 tile |
| `release->acquire` | 某 `[ready_m, source_rank]` 最后一次通信发布 | 指定 `[m_tile,n_tile]` GEMM CTA 观察该 peer | 数据已发布到消费者真正读取之间的间隔 |
| `final publisher` | 最后完成该 `[ready_m, peer]` 的通信 chunk 开始执行 | ready atomic 完成 | 决定该 peer shard 发布时间的关键 chunk |
| `task setup / input-ready wait` | 关键 chunk 开始执行 | task 解码完成且源 rank 输入 epoch 可见 | 地址计算与可能存在的上游输入生命周期等待 |
| `remote G2S` | 发起 peer GMEM 到通信 CTA SMEM 的 bulk copy | mbarrier wait 返回 | 远端读取与 G2S TMA 阶段 |
| `local S2G` | 发起通信 CTA SMEM 到 `input_staging` 的 store | 目的端写入完成等待返回 | 本地 GEMM 输入已完整写好，随后才能发布 ready |
| `ready atomic` | 目标显存写入完成后，发起 ready counter 原子加 | 返回旧值并完成 diagnostic 时间戳 | 发布“这一片数据可以读取”；正式 kernel 使用不返回值的 reduction |

通信分段只记录使 `[ready_m, peer]` 计数达到目标值的最后一个 chunk。每个
task 会读取若干次 `%globaltimer`，但只有最终 chunk 写回 timeline；因此它用于
区分任务排队、G2S、S2G 和发布原子，不作为正式延迟。

`release->acquire` 变长不自动表示调度停顿：固定 K 顺序下，CTA 在观察后续 peer 前会先计算已经拿到的 K shard。

### GEMM -> A2A

| 轨道 | 起点 | 终点 | 含义 |
|---|---|---|---|
| `GEMM role` | compute CTA 进入 kernel | 该 CTA 的 CUTLASS persistent scheduler 和 epilogue 全部返回 | 本地 GEMM 生产阶段；无任务 CTA 会很短 |
| `QKV route role` | comm CTA 进入 kernel | 该 CTA 分配到的所有 ready wait、G2S/S2G 或 vector route 完成 | 本地输出路由阶段，包含等待 GEMM tile 发布 |
| `grid barrier / finalize` | 本 CTA 本地角色结束 | 本 CTA 退出 kernel | cooperative grid barrier；CTA0 还包含跨 rank source-complete 发布与等待 |
| `all local roles done -> kernel complete` | 本 rank 最后一个本地角色结束 | 本 rank 最后一个 CTA 退出 | 本地计算和通信都已完成后仍暴露在关键路径上的尾部 |

`GEMM role` 与 `QKV route role` 的交集是 CTA-specialized 的实际本地重叠。
`QKV route role` 包含 ready wait，因此不能直接当作裸 NVLink 时间；裸 route 仍由默认构建的 standalone route 测量。

`all local roles done -> kernel complete` 在 CTA0 的 `finalize` 轨道中进一步拆成：

| 子阶段 | 含义 |
|---|---|
| `local roles -> grid sync` | rank 内所有 persistent CTA 到达 cooperative grid barrier 的尾差 |
| `fence.sc.sys` | route 写入完成后的 system-scope fence |
| `publish source-complete epochs` | CTA0 第一 warp 的 lane 0..`world-1` 并行向各 destination rank 发布本轮 route 完成标记 |
| `wait source N epoch` | CTA0 的 lane N 从并行轮询开始，到 source N 完成标记可见的独立等待；各 source 使用独立 Perfetto 轨道 |
| `kernel retire` | 最后一个 source 就绪后，到 rank 内最后一个 CTA 退出 |

`wait source N epoch` 的起点是本 rank 完成全部并行 publish 的时刻，终点是本地
acquire 观察到 source N 的 epoch；它包含 source N 的剩余计算、路由和网络传播，不能
单独解释为裸 NVLink 延迟。所有 source wait 会重叠，finalize 的暴露等待取其最大值，
不能相加。

MPI role summary 将同一批 CTA 时间戳压缩成以下字段：

| 字段 | 定义 |
|---|---|
| `compute_role_us` | 最早 compute CTA 启动，到最晚 compute CTA 完成本地角色的包络 |
| `route_role_us` | 最早 route CTA 启动，到最晚 route CTA 完成本地角色的包络；包含 ready wait |
| `overlap_us` | 上述 compute 与 route 两个包络的交集 |
| `grid_sync_us` | 本 rank 最后一个本地角色完成，到 CTA0 通过 cooperative grid barrier |
| `finalize_us` | CTA0 通过 grid barrier，到本 rank 最后一个 CTA 退出；包含 fence、跨 rank 发布与 source-complete 等待 |
| `kernel_us` | 本 rank 最早 CTA 启动，到最后一个 CTA 退出的完整 diagnostic kernel 包络 |

这些字段来自 `%globaltimer`，只能比较同一个 rank 内的起止和跨度；不同 GPU 的
绝对时钟值不能互相相减。它们是单次 diagnostic kernel 的角色分解，不是正式 E2E
延迟，也不能替代 profiling 关闭后的跨 rank 最大值 10+50 结果。

### SM103 QKV 逐 route tile 诊断

SM103 `fused_bf16 --profile --profile-direction qkv --profile-detail full`
在原 CTA/finalize 时间线上增加每个通信 warp 的逐任务记录。目前只接受
64×128 BF16 tensor-copy 路径，其他路径拒绝而非输出缺失记录。每个 task
附带源 row/column、大小、目标 rank、Q/K/V segment，以及生产 GEMM tile 坐标。

- `ready wait`：开始等待生产 tile 到 ready acquire 与原有 warp join 返回。
- `local G2S`：TMA load 发起前至 transaction mbarrier 等待返回。
- `peer S2G (SMEM read complete)`：TMA store 发起前至原有 `.read` wait 返回；
  **仅表示源 SMEM 可复用，不表示远端 GMEM 写入完成**。
- `all peer writes drain`：每个 warp 完成全部任务后，原有完整 wait-group
  的区间；到此该 warp 发起的目标 GMEM 写入全部完成。

不为逐任务打点添加完整 store wait 或新同步。阶段间隙包括地址计算、fence、
原有 warp 同步和记录写回；不能把三段时间之和当作完整 route role。
记录按 task 索引唯一写入，验收全覆盖、warp 内单调不重叠和 CTA role 边界。
SM103 `producer_ready_v1` 的 task 是生产顺序中的候选槽位，padding 或归属
另一个依赖 tile 的槽位不发起拷贝，因此 task 编号允许有空洞。
`profile_qkv_order` 单独记录槽位数和实际拷贝数；逐几何块验收唯一覆盖，
末尾 drain 从槽位数开始编号，不能用有效记录数推断 drain 的起始编号。
`profile_qkv_order.route_warps` 记录每 CTA 的实际 route warp 数；旧日志省略时
按 8 处理。copy 的 owner 为 `task % (comm_ctas * route_warps)`，对应
`cta=owner % comm_ctas`、物理 `warp=owner / comm_ctas`。drain 编号为
`slots + warp * comm_ctas + cta`，必须逐 rank 验收全部 copy 和
`comm_ctas * route_warps` 个 drain，不能通过减少轨道数掩盖缺记录。
生产参数结构不增加记录指针；开关关闭时不实例化详细诊断路径。诊断专用
kernel 预热10次，清空记录后采一个连续 epoch，保留前后生产/诊断 event 时间。
只采一个 epoch，不把这些时间替代正式10+50结果。使用已有
`scripts/export_sm103_qkv_perfetto.py` 流式导出，验证后替换当前六份 JSON，
不保留多轮重复 trace。

TMA 条带的 `args` 记录 `src_gpu`、`dst_gpu`、`route_peer`、`bytes` 和 `task`。
GPU 编号为 trace 的逻辑 rank，不是 PCI ordinal。local G2S 两端在同一 GPU，
表示本地 GMEM→通信 SMEM；peer S2G 表示源通信 SMEM→目标 GMEM，若两端 rank
相同则为本地写入。`route_peer` 是该 tile 最终路由目标；`bytes` 是 BF16 payload，
不是实测 NVLink 线上字节数。S2G 终点仍为源 SMEM read complete，不生成虚构的
接收端到达时间。旧六份 JSON 可用 `export_sm103_qkv_perfetto.py --annotate-trace
<path>` 原位补充这些派生信息，不改变时间戳、轨道或 kernel，不需重新采集。

### SM103 MXFP8 QKV 权重量化诊断

沿用上面的 QKV CTA/warp、route 三阶段、finalize 与独立 GPU 时钟原点。
激活在入口已是 MXFP8；本 trace 包含持久化 kernel 内 BF16 master weight 的
量化，不包含上游激活量化。`FUSE_ENABLE_PROFILING=OFF` 时无这些记录与打点。
诊断构建、10 次诊断预热、清空记录后一个连续 epoch、完整数值/路由校验和
原导出器不变。单进程 per-GPU-thread 发射仅用于诊断，不替代 MPI Graph 正式计时。

| 条带 | 测量边界 |
|---|---|
| `W quantize BF16 -> MXFP8` | 原有一个量化 progress chunk 的 32 个 group：读取 BF16、amax 归约、UE8M0 scale、E4M3 转换和写入；不是纯转换指令时间 |
| `W local bookkeeping (no publication)` | 非发布 chunk 的量化结束到本地记账结束；没有 panel 原子计数或 ready 发布，不画伪 atomic/fence 子条带 |
| `W arrival counter + warp join` | 仅在本 worker 当前 panel 的最后一个有效 chunk：量化结束到聚合到达计数、可选 ready 发布和 warp 汇合结束；没有额外的 SC device fence |
| ↳ `W first warp join` | `quant_done→warp_join_done`：原有第一次 `__syncwarp()`，汇合本 warp 的 32 个 lane |
| ↳ `W arrival counter (address + atomic return)` | `warp_join_done→arrival_done`：lane 0 的 panel 计数寻址和返回旧值的 `fetch_add(arrival_chunks, acq_rel)`；不能单独称为纯原子指令延迟 |
| ↳ `W ready publication + final warp join` | `arrival_done→end`：最后到达者可选的 ready 发布，以及原有最后一次 warp 汇合；非最后到达者不发布 ready |
| `W panel ready published (post-store stamp)` | 最后到达的 worker 发布完整 N256×K 权重 panel 后的时间戳；每 panel 一个点，不是每 K32 一个 ready |
| `GEMM waits W panel + warp join + proxy fence` | 实际发生的新 N panel acquire 至原有 warp 汇合/async proxy fence 结束；缓存命中的复用不伪造事件 |

量化与通信细节紧随所属 CTA/warp 排列，GEMM 权重等待紧随所属计算 CTA。
记录附带 panel/chunk/M tile、CTA/warp、group 范围及精度。固定槽位唯一写入，
不添加队列 claim 原子或新的同步，不改变量化生产/消费粒度。导出检查量化
chunk 全覆盖、每 panel 一次发布、时间戳有序且位于对应 CTA 角色内。
当前 `warp_panel_acq_rel_v3` 将每 worker 对同 panel 的到达贡献合并一次，
不改每次 progress 的 1024 值处理量或完整 panel ready 粒度。每 chunk 仍独立记录
量化时间；`arrival_chunks=0` 时 `warp_join_done/arrival_done/release` 必须全为0；
正值表示本次实际聚合贡献的有效 chunk 数。导出按 `(rank, CTA, warp, panel)`
检查正贡献恰在该 worker 最后一个有效 chunk，且等于它的有效 chunk 数；再检查
各 panel 的贡献总数等于该 panel 的有效 chunk 数。没有任务的 worker 不贡献，
N 尾部按原128行 padding 计数，不把无效调度槽位算入贡献。
profile 记录为72字节；新增贡献字段和子阶段时间戳都在该段结束后统一写回，
没有在原子热路径插入新的 profile 全局写。precision 行与每条记录的协议一致。
当前 `mxfp8_weight_preparation=comm_warp` 使用单向交接：256-thread 通信 CTA
的物理 warp 0..3 只做 route；warp 4..7 各自完成所有量化/发布后转做 route，
不再返回量化，不增加交接 barrier 或动态队列。量化仍有 `comm_ctas*4` 个
worker；route 保留原 `comm_ctas*8` 静态 ownership，每个 warp 都须有 drain。
`profile_qkv_order` 明确记录 `route_warps=8,weight_schedule=warp_then_route_v1`。
CTA 标为 `MXFP8 route + quant role`，route 展示实际 warp 0..7，量化仍按物理
warp 4..7 展示，不重新编号。全细节导出利用已有时间戳验收：同 rank/CTA/warp
最后量化 `end` 不晚于首次 route `begin`（无copy时检查drain），因此同warp没有
量化穿插在任意route条带内；不同warp的量化与通信仍可并行。该检查不推断
TMA远端到达时间，CTA-only日志不宣称完成此时间线验收。

历史固定4+4日志以 `route_warps=4` 识别，并标记 `fixed_split_legacy_v1`，仅
展示0..3的route轨道；不将旧数据解释为单向交接。导出交叉检查job模式、各rank
的route_warps和weight_schedule，拒绝元数据缺失而无法区分的新旧comm_warp，
以及矛盾/跨rank不一致或量化落在非通信CTA、非物理warp4..7的记录。
旧 `comm` 的8-warp交替复用及 `all` 格式保持兼容；省略schedule不推断新交接。
各 warp 的量化时间可以重叠，不能相加当关键路径；ready 是发布后的打点，
消费者可能在该打点之前已经观察到 flag，不能据此生成负的传播时间。

三个发布子阶段仅在 `arrival_chunks>0` 的 compound 条带内、同一 `tid` 嵌套显示，不增加独立轨道；
它们连续覆盖父条带。新增边界先保存在寄存器，整个发布段结束后再写入记录，
不在 warp join/atomic 之间插入新的全局 profile 写入或同步。当前导出要求
正贡献满足 `quant_done <= warp_join_done <= arrival_done <= end`，不记录 `fence_done`，
也不伪造零耗时的 fence 条带。`args.publication_protocol=warp_panel_acq_rel_v3`
标识聚合发布协议；warp 汇合与 `fetch_add(acq_rel)` 的发布语义仍保留。
`publication_subphase_chunks` 只统计实际发布的 chunk，不再等于全部量化 chunk；
`aggregate_publications` 是有任务的 worker-panel 组合数。

旧日志没有 `arrival_chunks`、只有 `warp_join_done/arrival_done` 时，继续按
`warp_join_acq_rel_v2` 每 chunk 显示三个真实子阶段，不将它解释为聚合发布。
旧日志同时含 `fence_done`、`warp_join_done`、`arrival_done` 时，继续使用
`W fence + arrival counter + warp join` 父条带和原四个子阶段：先测
`quant_done→fence_done` 的 `W device fence`，再从 `fence_done` 开始测第一次
warp 汇合。此时标识为 `sc_fence_acq_rel_v1`，并验证全部五个边界有序。
旧日志缺少全部三个细分字段时，标识为 `legacy_unsplit`，仅保留原父条带。
v3 缺贡献/子阶段字段、混合新旧协议、其余部分字段缺失或任何边界乱序均拒绝。
发布段不含 GEMM 消费者等待，不是 release→acquire 传播时间。计数器可能存在
同 panel 的竞争，但条带长度本身不能证明竞争或其占比；新增打点也有诊断开销，
不能替代关闭 profiling 的正式计时。

### TE Userbuffers QKV 对照（CUDA Event）

TE 对照不使用 `nsys --cuda-graph-trace=node`。逐 node 的 CUPTI 回调会显著放大
由多个短 GEMM 和 P2P kernel 组成的 Graph。本协议在显式传入 `--trace-out` 时，
只向 diagnostic Graph 加入少量 external CUDA timing event：

| 轨道 | 含义 |
|---|---|
| `8 QKV slab GEMMs` | 主 stream 上八个 destination slab GEMM 的总跨度 |
| `remote sends` | Userbuffers send stream 的首个发送到全部发送完成 |
| `remote receives` | Userbuffers recv stream 的首个接收到全部接收完成 |
| `send + recv envelope` | send/recv 两段的并集外框，作为 `T_comm` |
| `unpack / dependency tail` | 计算和通信依赖满足后，到输出 unpack 完成 |
| `TE UB boundary` | 完整 Graph 边界 |

`T_overlap` 是 GEMM 区间与通信外框的交集；通信掩盖比例为
`T_overlap / T_comm`。Event 版本仍是 diagnostic Graph，绝对性能继续引用不带
`--trace-out` 的正式 10+50 结果。

```bash
CUDA_VISIBLE_DEVICES=<devices> \
PYTHONPATH=/home/chen/workspace/source_code/TransformerEngine \
LD_LIBRARY_PATH=/home/chen/workspace/source_code/TransformerEngine:/usr/local/cuda/lib64 \
/home/chen/miniforge3/envs/mmunlearner/bin/python -B -m torch.distributed.run \
  --standalone --nproc-per-node=<CP> \
  benchmarks/QKVproj+a2a/te_userbuffers_qkv.py \
  <正式 winner 参数> --cuda-graph --trace-out <perfetto.json>
```

## 完整性检查

每个 rank 都必须满足：

- CTA timeline 容量等于该 GPU 的 SM 数；
- release 记录数为 `ceil(M / ready_BM) * world`；
- GEMM tile 记录数为 `ceil(M / BM) * ceil(N / BN) * L`；
- acquire 记录数为 GEMM tile 记录数乘 `world`；
- JSON 可以完整解析，所有 duration 非负；
- 四个比较 case 使用相同通信 CTA 数和相同数据初始化。

GEMM -> A2A 还必须满足：

- 每个 CTA 都有 `start <= role_done <= end`；
- CTA `[0, comm_ctas)` 是 route role，其余是 compute role；
- QKV trace保存`FUSE_QKV_GEMM_POLICY`、`FUSE_QKV_COMM_POLICY`请求值、模型版本以及实际`tile_m/tile_n/cluster_m`；
- `all local roles done` 取所有 CTA 的最大 `role_done`；
- 最终 kernel 时间取所有 CTA 的最大 `end`，不能用 rank 内平均值。

## 解读边界

SM103 MXFP8 独立服务标定使用 `--mxfp8-service-probe`，sidecar 为
`services-rank-0.jsonl`（`sm103_mxfp8_services_v1`）。只在 GPU0 记录逐 tile、
量化 chunk、完整 route slot 与发布区间，其他 rank 仍执行真实通信并完整校验。
每个服务边界只保留一个原始 capture；tile/warp 内部样本彼此相关，不宣称为
50 次独立 trace。另存的 control/instrumented 10+50 是包含主机 API 准备等待的
Eager CUDA-event 诊断，不能当纯 kernel 扰动，也不能拿该比值校正服务系数。
服务时长仅来自同 rank、同 capture 的 globaltimer 差值。正式 Auto 验收仍须
profiling 关闭、真实 `comm_ctas=0`、MPI Graph 10+50，与手工配置独立对照。

发布 chunk 在独立 Q、G2S 重叠、S2G 重叠下分别统计。混合场景同时保留
slot 起点到 quant.end（含发布）与 slot 完成的时长，不能把 Q-only 发布常数
叠加到已包含发布的混合区间。测到某个上下文的区间不意味着能预测任意并发
流量中的区间；探针不证明已覆盖与 GEMM 并发的全部资源竞争状态。

整段服务模型使用同一 GPU/capture 的起止差：起点为最早 CTA 的 setup_done，
Q 终点为全部量化 warp 的最大 quant.end；R/QR 终点包含所有最终远端 drain
和剩余量化完成，不含跨 rank finalize。不能拿 thread0 的 CTA 退出代表其他
量化 warp 完成。R/QR 是整段有效服务，不是裸 TMA 延迟或融合执行时间的严格界。
计算首 tile/周期仍由逐 tile 记录直接统计，不用整段 C 除 waves 代替。
细分发布记录保留用于诊断，不作为新整段模型的决策系数。

- 不比较不同 rank 的绝对 `%globaltimer` 值；每个 rank 在 JSON 中使用自己的时间原点。
- `GEMM` 轨道不是纯 Tensor Core 时间，不能直接拿它计算 WGMMA 吞吐。
- profile kernel 多写 global-memory 时间戳，数值会受观测开销影响；正式延迟仍以 profiling 关闭后的 10+50 benchmark 为准。
- Perfetto 先看同一 rank 内的通信完成、peer 发布顺序、首包等待和 CTA 长尾，再用正式 benchmark 判断这些现象是否影响端到端时间。
- TE Userbuffers 对照使用正式 winner 配置和 CUDA Event 阶段时间线。TE 是多 stream 边界，不能用单个 Graph 外框代替计算、通信和 unpack 三段。
