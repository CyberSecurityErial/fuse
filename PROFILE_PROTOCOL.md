# 通算融合 Perfetto profile 协议

这套协议用于定位两个方向的单 kernel 时序：

- `A2A -> GEMM`：通信发布、ready handoff 和 GEMM 消费；
- `GEMM -> A2A`：GEMM 生产、QKV route、cooperative grid barrier 和跨 rank finalize。

正式性能数据必须使用默认构建；profile 数据只解释相对时序。

## 正式启动与计时规则

正式 benchmark 固定区分 `eager` 和 `graph`，两列独立报告，不把两者的差值
当作算子优化收益。

- `QKV Projection -> A2A` 必须使用 MPI/torchrun 一进程一卡；每个样本先在各
  rank 记录 CUDA event，再对 elapsed time 做 `MPI_MAX`。单进程按 rank 顺序
  launch 会把后续 GPU 的 host 提交时间计入前面 rank 的跨源 finalize，只能用于
  diagnostic trace，不能进入正式性能表。
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

v7正式路径保持两个环境变量未设置；下列变量只用于固定policy的消融或回退复现。

```bash
CUDA_VISIBLE_DEVICES=<devices> \
FUSE_QKV_COMM_POLICY=<pipeline|legacy> \
FUSE_QKV_GEMM_POLICY=<wave_time_model|legacy|m128n128|m128n160|m128n192|m128n256|m128n320> \
./build/qkvproj_a2a_bench \
  --mode qkv_gemm_a2a --m <M> --k <K> \
  --batch 1 --q-heads <Hq> --kv-heads <Hkv> --head-dim <D> \
  --comm-ctas <comm> --raster <m|n|auto> --swizzle <1|2|4|8> \
  --warmup 1 --iterations 1 \
  --trace-out /home/chen/workspace/<name>_perfetto.json
```

## 事件定义

所有设备端时间戳来自 SM90 `%globaltimer`，原始单位为 ns；JSON 中 `ts` 和 `dur` 按 Perfetto/Chrome trace 约定写成 μs。

协议版本：6

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
| `local S2G` | 发起通信 CTA SMEM 到 `input_staging` 的 store | `tma_store_wait<0>()` 返回 | 本地 GEMM 输入落盘阶段 |
| `ready atomic` | 发起 ready counter 原子加 | 返回旧值并完成 diagnostic 时间戳 | 发布指令本身；正式 kernel 使用不返回值的 reduction |

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

### TE Userbuffers QKV 对照

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

- 不比较不同 rank 的绝对 `%globaltimer` 值；每个 rank 在 JSON 中使用自己的时间原点。
- `GEMM` 轨道不是纯 Tensor Core 时间，不能直接拿它计算 WGMMA 吞吐。
- profile kernel 多写 global-memory 时间戳，数值会受观测开销影响；正式延迟仍以 profiling 关闭后的 10+50 benchmark 为准。
- Perfetto 先看同一 rank 内的通信完成、peer 发布顺序、首包等待和 CTA 长尾，再用正式 benchmark 判断这些现象是否影响端到端时间。
- TE Userbuffers 对照使用正式 winner 配置和 CUDA Event 阶段时间线。TE 是多 stream 边界，不能用单个 Graph 外框代替计算、通信和 unpack 三段。
