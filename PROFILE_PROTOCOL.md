# A2A + GEMM Perfetto profile 协议

这套协议只用于定位单 kernel 内通信、ready handoff 和 GEMM CTA 的时序。正式性能数据必须使用默认构建；profile 数据只解释相对时序。

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
2. 固定 `M/N/K`、CP、`comm_ctas`、tile policy、raster 和 swizzle；一次只改一个待比较变量。
3. 使用 `--trace-out`。benchmark 会在普通测量结束后清空 ready flag，以 epoch 1 单独启动一次 diagnostic kernel。
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

## 事件定义

所有设备端时间戳来自 SM90 `%globaltimer`，原始单位为 ns；JSON 中 `ts` 和 `dur` 按 Perfetto/Chrome trace 约定写成 μs。

| 轨道 | 起点 | 终点 | 含义 |
|---|---|---|---|
| `remote A2A` | 通信 CTA 进入 kernel | 该 CTA 全部 warp 汇合并退出 | 一个 persistent 通信 CTA 的完整存活区间 |
| `ready wait` | 计算 CTA 进入 kernel | 第一次观察到可消费 peer shard | 计算 CTA 的首包等待 |
| `GEMM` | 第一次 ready acquire | 计算 CTA 退出 | 该 persistent CTA 的后续区间，包含 WGMMA、后续 peer wait、epilogue 和可能的多个逻辑 tile |
| `release->acquire` | 某 `[ready_m, source_rank]` 最后一次通信发布 | 指定 `[m_tile,n_tile]` GEMM CTA 观察该 peer | 数据已发布到消费者真正读取之间的间隔 |

`release->acquire` 变长不自动表示调度停顿：固定 K 顺序下，CTA 在观察后续 peer 前会先计算已经拿到的 K shard。

## 完整性检查

每个 rank 都必须满足：

- CTA timeline 容量等于该 GPU 的 SM 数；
- release 记录数为 `ceil(M / ready_BM) * world`；
- GEMM tile 记录数为 `ceil(M / BM) * ceil(N / BN) * L`；
- acquire 记录数为 GEMM tile 记录数乘 `world`；
- JSON 可以完整解析，所有 duration 非负；
- 四个比较 case 使用相同通信 CTA 数和相同数据初始化。

## 解读边界

- 不比较不同 rank 的绝对 `%globaltimer` 值；每个 rank 在 JSON 中使用自己的时间原点。
- `GEMM` 轨道不是纯 Tensor Core 时间，不能直接拿它计算 WGMMA 吞吐。
- profile kernel 多写 global-memory 时间戳，数值会受观测开销影响；正式延迟仍以 profiling 关闭后的 10+50 benchmark 为准。
- Perfetto 先看同一 rank 内的通信完成、peer 发布顺序、首包等待和 CTA 长尾，再用正式 benchmark 判断这些现象是否影响端到端时间。
