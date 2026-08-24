# H200 CTA-specialized MegaMoE 设计蓝图

状态：设计阶段。本文冻结数据流、角色边界、同步协议、失败条件和落地顺序；具体 tile、
CTA 数量和 scheduler 参数留给逐项实验。

参考提取见 [REFERENCE_EXTRACTION.md](REFERENCE_EXTRACTION.md)。直接性能对手固定为
DeepGEMM `ec757bd0` 的 SM90 PR360。

## 1. 目标

在单机 8×H200 NVLink 上实现一个纯 CUDA/CuTe/CUTLASS 的单 launch MegaMoE：

```text
EP dispatch
  -> Linear 1
  -> SwiGLU + top-k weight + FP8 requant
  -> Linear 2
  -> EP combine scatter
  -> top-k reduction
```

第一性能目标是超过 PR360 SM90 auto kernel；第二目标是找出 CTA specialization 在
token/expert、routing skew 和模型 shape 上的有效区间。

不在第一版做：router logits/top-k 本身、训练反向、跨节点 IB、FP4、模型框架集成。

## 2. 先回答“为什么值得拆 CTA”

PR360 每个 SM 都固定携带 dispatch、TMA、MMA、epilogue 和 combine：

- dispatch 完成后，相关 warps 主要做 cleanup；
- combine 阶段 tensor core 空闲；
- 每个角色复制同一份 scheduler 状态并独立自旋；
- dispatch buffer 和 barrier 常驻 shared memory；
- L2 epilogue承担 peer scatter；
- 全 rank barrier 把流式到达压成阶段边界。

CTA specialization 的假设是：少量 RouteCTA 可以打满 NVLink，把其余 SM 留给纯
ComputeCTA；ComputeCTA 执行路径不再保持 dispatch/combine 状态存活，并通过 shared
storage union复用对应空间，同时 RouteCTA 能在 dispatch 后切到 output scatter/reduce。

这个假设必须由数据证明。静态拿走 `R` 个 SM 会令 GEMM 只剩 `132-R` 个 SM。
单入口 kernel 中两类 CTA 共用同一个 `blockDim`、寄存器上限和 dynamic SMEM 大小；
RouteCTA 即使只执行少量指令，也仍占一个完整 resident CTA/SM。shared storage 应写成
`union { ComputeStorage; RouteStorage; }`，让 ComputeCTA 复用 dispatch 区域扩 mainloop
stage，不能把它解释成 RouteCTA 获得了独立的轻量 occupancy。第一轮必须同时测：

```text
通信隐藏收益
vs
GEMM 少 R 个 SM 的损失
vs
compute CTA 多一个 pipeline stage/更少控制逻辑的收益
```

若三者没有交点，CTA specialization 不成立，不能靠较弱 baseline 制造胜利。

SM100 结构落到 H200 的对应关系先冻结为：

| SM100 原语/角色 | H200 v0 决策 |
|---|---|
| 4 dispatch warps | 独立 RouteCTA pool，负责 dispatch/scatter/combine |
| A/SFA、B TMA + TCGen05 issue | ComputeCTA 内的 SM90 TMA/WGMMA producer；SFB由 math WG `__ldg` |
| TMEM accumulator + 独立 epi WG | 不迁移；FP32 accumulator epilogue留在 math WG/同一 CTA |
| cluster2、2-CTA UMMA | 不迁移；v0 cluster1 |
| L1 arrival / L2 mask | dispatch epoch + 两个 L1 half-ready epoch |
| L2 direct peer write | compact local ring + RouteCTA vector scatter，保留 direct-peer A/B |
| L2 后 rank barrier | full-contribution ready epoch；只保留 input-lifetime handshake |

## 3. 推荐的第一版物理结构

### 3.1 单个 cooperative persistent grid

```text
grid.x = resident_ctas <= 132
blockDim = compute policy 所需线程数
cluster = 1                         // v0

[0, R)                 RouteCTA
[R, resident_ctas)     ComputeCTA
```

首轮扫描：

```text
R in {8, 12, 16, 20}
G = resident_ctas - R
```

必须在 launch 前用真实 blockDim、register、dynamic SMEM 做 cooperative residency
检查，证明所有 RouteCTA 和 ComputeCTA 同时驻留。任何双-stream/two-kernel 版本只作
baseline，不冒充单 kernel CTA specialization。

### 3.2 RouteCTA

RouteCTA 是一个阶段自适应的通信 worker pool，不分配专门 ControlCTA：

1. 统计 top-k expert count；
2. 预留 remote slot，交换 source token/top-k metadata；
3. 从 peers 拉 token、SF、top-k weight，构造 local expert pool；
4. 发布 dispatch-ready；
5. 在 remote-pull 循环中穿插消费 L2 output-ready，将 BF16 tile scatter 到 source
   rank/top-k slot；
6. 消费本 rank combine-ready，FP32 top-k reduce，写 BF16 y；
7. 回收 epoch/workspace。

RouteCTA0 的一个 warp 可以做 prefix/epoch 控制；不为这些短任务占一个完整 CTA。

硬规则：`R_dispatch_min >= 1` 个 RouteCTA 在 `dispatch_done` 前只推进 dispatch，永远不等
ComputeCTA。第一批 L2 output 可能出现后，再固定 `R_ring_min >= 1` 个 RouteCTA 优先
drain ring，直到 `scattered_tiles == total_l2_tiles`；它不能领取未 ready 的 combine token
后阻塞。其余 RouteCTA 使用有限 burst，例如处理 8 个 pull task 后轮询一次 ring 和
ready combine queue。优先级固定为 `dispatch progress / ring drain > ready combine >
cleanup`，combine 只能 try/poll。dispatch 完成后，全部 RouteCTA 转入 scatter/combine。
这样既允许 L2 输出边到边发，也不会形成 route 全体等待 compute 的环。
正式 fused policy 因此要求 `R>=2`；`R=1` 只允许作为无 overlap 的协议测试。

### 3.3 ComputeCTA

ComputeCTA 只做：

```text
TMA A/SFA + B; math WG global-load SFB
  -> WGMMA
  -> register accumulator epilogue
```

v0 的 block固定为 384 threads/12 warps，保持三个完整、128-thread 对齐的 hardware WG：

```text
WG0 (warp 0..3): A/SFA producer, B producer, scheduler/control, idle/control
WG1 (warp 4..7): math/epilogue WG0
WG2 (warp 8..11): math/epilogue WG1
```

WG0 四个 warp都必须参加同一个 `setmaxnreg.*.sync.aligned`，即使其中两个 warp没有
mainloop copy；不能把 ComputeCTA 缩写成“2 producer warps + 8 math warps”的 320-thread
block。WGMMA 和 warpgroup register reallocation都依赖完整 WG 对齐。

它动态执行 L1 或 L2 task：

- L1：FP8×FP8、FP32 accumulate、clamp/SwiGLU/top-k weight、per-row/per-64
  E4M3 quant，写 L2 activation pool；
- L2：FP8×FP8、FP32 accumulate，BF16 写 bounded local output ring，发布给
  RouteCTA。

L1 epilogue必须留在 ComputeCTA。Hopper 没有 TMEM，把 accumulator 交给另一个 CTA
会先落一次完整 FP32 tile，代价太大。

v0 复用与对手相同的两种 GEMM policy，先隔离调度收益：

```text
small T: M64 N128 K128 pingpong
large T: M128 N128 K128 cooperative
```

后续可以独立改变 tile，但 pure GEMM 对照必须同步更新。

## 4. 完整数据流

```text
source rank symmetric input
    |
    | RouteCTA: count + metadata exchange
    v
final expert counts / pool offsets
    |
    | RouteCTA: peer-round-robin TMA pull
    v
expert_x_pool + x_sf + topk_weight + source_metadata
    |
    | release dispatch_ready_epoch[e,m]
    v
ComputeCTA L1
    |
    | SwiGLU + requant, release l1_to_l2_ready_epoch[e,m,k_group,half]
    v
l2_activation_pool
    |
    | ComputeCTA L2
    v
local BF16 output ring
    |
    | store_release slot_seq[slot] = ticket + 1
    v
RouteCTA peer scatter
    |
    | all N tiles scattered, release combine_ready_epoch[token,topk]
    v
source rank combine slots
    |
    | RouteCTA FP32 reduce
    v
BF16 y
```

这条链路取消 L2 完成后的全 rank compute barrier，每个 token 的 top-k 贡献到齐即可
reduce；输入复用所需的早期 dispatch-pull handshake仍然保留，见 5.8。

## 5. Buffer 与信号协议

### 5.1 继续沿用的 buffer

- registered symmetric input `x/x_sf/topk_idx/topk_weight`；
- expert-contiguous L1 token/SF pool；
- top-k weight pool；
- pool token 的 source metadata；
- L2 FP8 activation/SF pool；
- source rank 的 `[topk_slot, token, hidden]` combine buffer；
- BF16 output `y`。

### 5.2 新增的 buffer

- bounded BF16 L2 output ring；
- ring slot descriptor `{expert,m,n,valid_m,epoch}`；
- 双缓冲 dispatch arrival counter 与 ready epoch；
- 双缓冲 pull-task completion counter 与 rank done epoch；
- 每个 L1→L2 K-group 的两个 half-ready epoch；
- 64-bit output-ring producer/consumer ticket 与 per-slot sequence；
- 双缓冲 scatter-N completion counter；
- 每个 `(token, topk_slot)` 的 combine-ready epoch。

ring 必须是有界的，不新建完整第二份 L2 output tensor。slot 数以 RouteCTA 能及时排空
为准，首轮扫描 2/4/8 个 tile/RouteCTA。

### 5.3 内存序

固定 arrival 数的信号使用单调 epoch：

```text
producer writes payload
producer st/atom.release.<scope> ready = epoch or counter
consumer ld.acquire.<scope> ready
consumer reads payload
```

所有 rank使用同一个非零 epoch，首轮从 1 开始。steady state 不清零 ready-epoch
array；双缓冲 counter按协议初始化。epoch wrap 在 host 端处理。

routing count、尾 tile `valid_m`、masked top-k mask 都是每次调用可变的，不能套用
`epoch * 固定target`。这类状态使用以下二选一协议：

1. 64-bit `{epoch,count_or_mask}`，首次更新用 CAS 安装本 epoch；
2. 双缓冲 workspace，按 `epoch & 1` 选槽，并在允许任何 peer 发布前完成本槽初始化。

v0 优先选双缓冲，协议更短。L1→L2 的唯一生产者 flag 直接写 epoch；ring sequence
使用不回退的 64-bit ticket。

scope 按数据真正跨越的边界选择，热路径不统一付 `.sys` 成本：

| 信号 | scope |
|---|---|
| dispatch pool ready、L1→L2、ring seq、wave barrier、local task queue | `.gpu` |
| 跨 rank count/metadata、peer combine-ready | `.sys` |
| 汇总多个 peer store 后再远端发布的 scatter completion RMW 链 | `.sys` |

实现时按这张表逐个关闭状态机：

| 状态 | 完成条件 | 复用协议 |
|---|---|---|
| expert count | 每 expert收到 `R*EP` 个无条件 participant arrival | epoch双缓冲 |
| metadata done | remote index写完后收到 `R*EP` 个 arrival | epoch双缓冲 |
| dispatch ready | 当前 `(e,m)` 收到 `valid_m` 行 | counter双缓冲 + ready epoch |
| L1→L2 | 两个唯一 half都等于本 epoch | ready epoch，不清零 |
| output ring | slot sequence等于当前 ticket状态 | 64-bit monotonic sequence |
| scatter done | 当前 `(e,m)` 收到 `H/BN` 个 N tile | counter双缓冲 |
| combine ready | 当前 `(token,topk_slot)` 等于本 epoch | ready epoch，不清零 |
| wave done | 每 wave收到 `G` 个 ComputeCTA arrival | counter双缓冲 + ready epoch |
| local pulls done | 当前 rank收到 `num_pull_tasks` 个 task completion | counter双缓冲 |
| input reusable | 所有 EP rank发布 dispatch-pull done | ready epoch，不清零 |

双缓冲初始化也有固定协议：每个物理 counter只有其所在 rank 的 controller CTA能清零；
清零后发布 `counter_init_ready[kind,parity]=epoch`。本地 updater 用 gpu-scope acquire，
会从 peer更新该 counter 的 updater 用 system-scope acquire，之后才能做第一次 RMW。
`epoch n+2` 清同一 parity前，owner必须已经观察到 epoch n 对应阶段的 done handshake；
本地 counter可由本地 kernel/stage completion证明，跨 rank count/metadata counter必须由
上一轮 metadata-done handshake证明。不能只依赖“时间应该够了”。

### 5.4 dispatch-ready

count/slot reservation只证明 offset不冲突，还不能证明 remote source metadata 已经可读。
每个 RouteCTA 写完自己负责的全部 `source_token/topk_slot` metadata 后，都要无条件参加
一次 system-scope `metadata_done` handshake；零 token stripe也必须参加。目标 rank只有
在 expected participant全部到齐后才开始按 metadata remote pull。v0 保留这个安全阶段，
以后若做分块 metadata/pull流水，需要为每个 chunk单独给出发布 target。

`dispatch_ready_epoch[e,m]` 表示该 expert M tile 的 token、input SF、top-k weight 和
source metadata 全部可见。对应的双缓冲 `dispatch_arrivals[epoch&1][e,m]` 在任何 peer
发布前清零，尾 tile 的 target 是 `valid_m`，不能固定等 BLOCK_M。

每个 pull task 等异步 peer copy 完成后，按实际写入行数执行：

```text
old = atomicAdd.acq_rel.gpu(dispatch_arrivals[buf][e,m], rows_written)
if old + rows_written == valid_m:
    store.release.gpu(dispatch_ready_epoch[e,m], epoch)
```

同一 counter 上的 acq_rel RMW 链把所有生产者的 payload write 汇入最后一个发布者。
ComputeCTA 用 acquire 等 `dispatch_ready_epoch == epoch`。`valid_m==0` 的 tile不进入任务
表。初始化 buffer 的跨 rank barrier 必须发生在任何 pull/publish 之前。

若整 tile 发布造成首包过晚，再细分：

```text
dispatch_ready_epoch[e,m,peer_or_row_group]
```

ComputeCTA 的 A loader只在 K-independent 的 token rows 到齐后启动；不让 math WG 在
MMA 内循环轮询通信。

### 5.5 L1→L2 ready

L1 的一个 `N=128` gate/up tile产生 64 个 post-SwiGLU channels。L2 的一个
`K=128` group需要相邻两个 L1 output tile。

推荐把原来的单个 `uint64` mask泛化成两个唯一生产者 flag：

```text
l1_to_l2_ready_epoch[e,m,k_group][half] = epoch, half in {0,1}
```

每个 L1 N128 task唯一拥有一个 half；完成 activation/SF TMA store 并
`tma_store_wait` 后做 release store。L2 对两个 half做 acquire wait 后再发该 K128 stage。
不需要并发 reset、atomic add 或 `{epoch,count}`，连续 invocation 也不会继承旧 arrival。

优点：

- 不再限制 `I/64 <= 64`；
- L2 可以按 K group 等待；
- 与之前 A2A→GEMM 的 Ready-K iterator 经验一致；
- 每个 per-64 scale 由唯一 ComputeCTA 产生，无跨 CTA amax。

### 5.6 L2 output ring

ComputeCTA 的 accumulator ready 后先进入有界 MPMC ring，领取并等到 free slot，再把
BF16 epilogue写进该 slot；不在 epilogue 发 peer TMA。ring 使用 per-slot sequence协议，
不能只放一对 producer/consumer epoch：

```text
workspace init: slot_seq[s] = s, enqueue_ticket = dequeue_ticket = 0

producer:
  t = atomicAdd(enqueue_ticket, 1)       // 必须满足 t < call_end
  s = t % num_slots
  wait_acquire(slot_seq[s] == t)         // 上一轮 consumer 已回收
  epilogue writers write payload; one thread writes descriptor; drain async store
  CTA-wide producer/math rendezvous
  one publisher: store_release.gpu(slot_seq[s], t + 1)

consumer:
  用 CAS 在 dequeue_ticket < call_end 时领取 t
  s = t % num_slots
  wait_acquire(slot_seq[s] == t + 1)
  read descriptor + payload; 完成 peer scatter
  one publisher after completion RMW:
    store_release.gpu(slot_seq[s], t + num_slots)
```

count finalized 后可精确计算本次 L2 tile 总数，令 `call_end=call_begin+num_tiles`；生产者
和消费者各自恰领取一次 `[call_begin,call_end)`。ticket/sequence 都用 64 bit，调用间不
清零；到 wrap 安全边界前由 host 停机重建 workspace。连续 epoch 测试必须覆盖 ring
多次回绕、多个 ComputeCTA 同时 enqueue、多个 RouteCTA 同时 dequeue。

kernel退出前必须满足 `enqueue_ticket == dequeue_ticket == call_end`，且区间内每个 slot
都已由 consumer发布到下一次 free sequence；只等 ticket相等而不等最后一次 scatter/
slot release也不够。

slot容量按 `BLOCK_M*BLOCK_N` 预留，实际 payload只能写/读/scatter
`valid_m*BLOCK_N*sizeof(BF16)`；尾行用 row-major compact store 或 runtime-size 1D bulk
copy。默认小 T 时一个 M64 tile常只有约8个有效行，若固定搬完整 M64 padding，ring
write+read 会被放大约8倍。full-padded ring只保留为反例 baseline，不进入生产 policy。

v0 的可信 scatter 路径是 RouteCTA 对每个有效行做 16-byte vectorized local load + peer
global store。所有 writer线程完成 store后先做 CTA-wide barrier；随后 thread0只发一次
`fence.sc.sys` 和 system-scope RMW，把完成性汇入 scatter counter。禁止每个 writer各发
一次 system fence。不能把 local-global ring
描述成能直接 TMA 到 peer。TMA variant 至少需要：

```text
local ring G2S -> mbarrier wait -> peer S2G -> tma_store_wait -> completion RMW
```

TMA issuer完成 wait 后还要与 CTA内参与 staging 的线程 rendezvous，再由唯一 publisher
做 completion RMW。它要在 `RouteStorage` 中放双缓冲 staging，并先通过独立
correctness/NVLink bandwidth microbench；证明优于 vector copy 后才进入 fused sweep。

这个设计增加一次 local BF16 write + read，必须单独测成本。它可能命中 L2，也可能与
W1/W2 weight traffic 争抢。若这项税超过 overlap 收益，保留两条后备路线：

1. L2 compute epilogue继续 direct peer scatter，只把 dispatch/combine CTA-specialize；
2. cluster/DSM handoff，ComputeCTA 将 tile 放 shared memory，RouteCTA 从 DSM 读取。

第二条不进入 v0：1:1 cluster 会损失一半 compute SM，cluster 4/8 又显著提高调度和
residency复杂度。

### 5.7 scatter completion 与 combine-ready

一个 ring slot 只覆盖 `(expert,pool_m,n)` 的一个 `N=128` 列块；默认 `H=7168` 时一个
完整 contribution 有 56 个 N tile。第一个 tile完成后绝不能发布整个 top-k slot。

v0 为每个 `(expert,pool_m)` 设置双缓冲
`scatter_n_done[epoch&1][expert,pool_m]`。RouteCTA 对该 ring tile 的所有有效行完成 peer
vector store 后，以 system release/acq_rel语义只做一次；若启用 TMA variant，则必须先
完成对应 `tma_store_wait`：

```text
old = atomicAdd.acq_rel.sys(scatter_n_done[buf][expert,pool_m], 1)
if old + 1 == ceil(H / L2_BLOCK_N):
    for each valid row in this pool_m:
        store.release.sys(
          combine_ready_epoch[source_rank][source_token,topk_slot], epoch)
```

同一 counter 的 RMW 链把全部 N tile 的 remote payload write 汇入最后一个 RouteCTA；
它再按 source metadata 为每一行发布一次完整 contribution。这样每个 N tile只有一个
completion atomic，并且不会为每行每列块做 system atomic。

本 rank按 `token % num_combine_workers` 唯一拥有 token，对每个有效 top-k slot分别
acquire 等：

```text
combine_ready_epoch[token, topk_slot] == epoch
```

全部有效 slot到齐后，严格按 top-k slot index升序对完整 `H` 做 FP32 累加，再写 BF16
`y`；不能按 contribution ready到达顺序归约。masked slot不等待。
双缓冲 scatter counter 必须在任何 peer scatter 前清零；ready epoch不清零。异步 peer
store 未完成时禁止发布 ready。

### 5.8 symmetric input 生命周期

取消 L2 后的全 rank barrier，不等于 source-owned input 可以提前复用。某个快 rank 的
本地 kernel结束时，慢 rank 仍可能在 remote pull 它的 `x/x_sf/topk_idx/topk_weight`；
host 若立刻 copy 下一次输入，会覆盖仍在读的 payload。

v0 保留一个只约束输入生命周期的 `dispatch_pull_done` rank handshake。每个 pull task先
drain 自己的 TMA，再完成 CTA内发布汇聚，最后由唯一线程对本 rank
`pull_tasks_done` 做 `atomicAdd.acq_rel.gpu`；数值最大的 task ID或队列暂时为空都不能
代表全部完成。最后完成者才向所有 rank发布 system-scope `dispatch_pull_done=epoch`；
`num_pull_tasks==0` 的 rank也必须无条件发布。

该握手不阻塞本 epoch的 L1/L2，可与后续 compute 重叠，但 kernel 返回前必须 acquire
确认所有 rank都已完成 pull。替代方案是 input ping-pong buffer，或 host 在下一次 copy
前做 rank barrier。三者至少实现一个；仅有 full-contribution combine-ready不足以保护输入。

## 6. Scheduler 蓝图

### 6.1 count 与 expert pool offset

沿用 packed 64-bit count：

```text
low32  = token count
high32 = completed RouteCTA/rank participants
```

expected participant 数改成 `num_route_ctas * num_ranks`。不能继续使用 PR360 写死的
`num_sms * num_ranks`。

这个 target成立有一个严格前提：每个 rank 的 RouteCTA `r` 只处理
`token = r + k*num_route_ctas` 的输入 stripe；完成自己的全部 low32 token count 后，
它必须对每个 expert无条件贡献一次 high32 arrival，哪怕该 stripe 对这个 expert 的
count 为零。所有 arrival 都是 system-scope release/acq_rel，finalizer acquire 等到
`high32 == num_route_ctas*num_ranks` 后才读取 low32。若实现不满足这个参与规则，就必须
重写 target，不能只替换常数。count workspace 也按 epoch双缓冲，并在任何 rank更新前
完成初始化握手。

count finalized 后做 expert block prefix。每个 expert pool 起点按当前 ComputeCTA 的
BLOCK_M 对齐。pingpong/cooperative policy若 BLOCK_M 不同，buffer capacity 和 offset
必须由本次 launch policy决定。

prefix/task table 只能由一个 owner发布。RouteCTA0 写完 expert offsets、每个 M tile 的
`valid_m`、wave/task bounds、`total_l2_tiles` 和 ring `call_end` 后，执行
`store.release.gpu(layout_ready_epoch,epoch)`。所有 RouteCTA 在做 pull/slot映射前、所有
ComputeCTA 在读取 scheduler前、所有 ring consumer 在领取 ticket前，都必须
`ld.acquire.gpu(layout_ready_epoch)==epoch`。不能用“count finalized”口头代替这条
payload publication。

### 6.2 v0：安全的 wave scheduler

先保留：

```text
wave: all L1 -> all L2
```

区别是 block stripe 改为：

```text
compute_id = blockIdx.x - num_route_ctas
stride     = num_compute_ctas
```

每个 ComputeCTA 完成自己的所有 L1 task 后，先 drain WGMMA、TMA load/store，并让本
CTA 的 producer/math角色完成一次 CTA-wide rendezvous；随后恰好一个线程对双缓冲
`l1_wave_arrivals[epoch&1][wave]` 做 `atomicAdd.acq_rel.gpu`。即使本 wave 没分到 task也
必须 arrive 一次。最后一个 RMW 继承前面 CTA 的完成性，再
`store.release.gpu(l1_wave_ready_epoch[wave],epoch)`；其余 CTA acquire 后进入 L2。

L2 wave结束和 activation-pool 复用采用同样的 `l2_wave_arrivals/ready_epoch` 协议，
arrival 前也要 drain 所有 L2 read/WGMMA/output store并完成 CTA-wide rendezvous。两个
arrival buffer 都按 5.3 的 init-ready协议清零。这个等待无环，因为每个 CTA 在 arrive
前已经完成全部静态责任，且所有 ComputeCTA 已被 cooperative launch证明常驻。

v0 已经可以重叠：

- 当前 wave compute 与后续 dispatch；
- L2 compute 与前一批 output scatter；
- output scatter 与已到齐 token 的 combine；
- combine 与 workspace cleanup。

### 6.3 v1：L1/L2 K-group 流水

目标是 L1 每完成两个 post-SwiGLU 64-column tile，L2 对应 K128 stage 就能消费。

风险：如果所有 ComputeCTA 都领取 L2 task并停在未 ready 的 K group，而仍有 L1 task
未执行，会永久死锁。

只允许以下两种实现：

1. 固定保留一部分 L1 workers；L1:L2 初值按 FLOPs 约 `2:1`；
2. admission credits：最多允许 `G-R_l1` 个 L2 waiter，始终保留 `R_l1` 个 CTA
   处理 L1。

禁止无 credit 的统一 atomic queue。

### 6.4 tile 顺序

L1 的两个相邻 N tile应连续产生，以尽早完成一个 L2 K group。为了让 L2 N tiles填满
`G` 个 ComputeCTA，再同时推进一个小 M window：

```text
frontier_m = ceil(num_compute_ctas / (H / L2_BLOCK_N))
```

以默认 `H=7168, BN=128, G≈120` 为例，`H/BN=56`，首轮候选
`frontier_m in {2,3,4}`。

大 tokens/expert 下，L2 继续测试 N-major：固定 W2 N tile，扫多个 M tile，提高 weight
L2 reuse。小 tokens/expert 下优先 M completion，尽早触发 output scatter/combine。

### 6.5 routing skew

heuristic 不能只看本 rank 输入 `T`。至少使用：

- 实际 local expert token count；
- active experts；
- max/mean tokens per expert；
- tail wave fill；
- local/remote token 比例。

v0 的 M64 pingpong/M128 cooperative 是不同 kernel entry、shared layout 和 descriptor，
只能由 host 在 launch 前按已知 `T/H/I` 选择。count finalized 后的 routing统计只能在
当前固定 policy 内调整 task顺序、wave大小和 active-expert 列表；也可以写入 telemetry
供下一 step/autotuner选择。若把两套 compute path编进一个 fat kernel，会按最大寄存器、
SMEM 和代码路径付费，必须单列 A/B，不能假装 device 端能免费切换预编译 kernel。

## 7. H200 compute policy

### 7.1 数据类型

第一版严格匹配 SM90 PR360：

- input/weights：E4M3；
- accumulator：FP32；
- W1 SF：FP32 `[E_local,2I/128,H/128]`，自然连续布局；
- W2 SF：FP32 `[E_local,H/128,I/128]`，自然连续布局；
- input SF：float per-token/per-128 K；
- L2 activation SF：float per-token/per-64 K；
- output：BF16；
- 只有 W1 FP8 data做 gate/up granularity-8 interleave；W1 SF和W2 data/SF不 interleave；
- activation clamp 和 fast-math 开关一致。

### 7.2 L1 compute

一个 L1 N128 tile必须同时包含完整 gate/up pair，epilogue产出 64 channels：

```text
WGMMA accumulator
  -> apply FP32 scale
  -> gate upper-clamp; up symmetric clamp
  -> SwiGLU
  -> lane-local amax of unweighted SwiGLU
  -> multiply values by top-k weight; amax *= abs(weight)
  -> per-row/per-64 amax reduction
  -> sf = max(amax, 1e-4) / 448; sf_inv = fast_rcp(sf)
  -> E4M3 + float SF
  -> TMA store
  -> release K-group half arrival
```

数值顺序严格跟 SM90 `ec757bd0`：scaled FP32 accumulator 直接进入 clamp/SwiGLU，
先求未加权局部 amax，再给 value乘 top-k weight并给 amax乘 `abs(weight)`，随后做行归约；
gate只做上限 clamp，up做双边 clamp；fast-math SiLU 使用 `__expf + fast_rcp`，量化用
`sf_inv=fast_rcp(sf)`。不插入显式 BF16 round-trip。SM100 路径中的 FP32→BF16 步骤
不能搬来改变对手合同。

先复用成熟的 SM90 WGMMA mainloop/scale顺序，创新点放在 CTA 间流水。pure L1 GEMM+
epilogue 必须单测，防止把较弱计算实现归因给 CTA scheduling。

### 7.3 L2 compute

L2 K128 stage读取两个 per-64 activation SF。v0 只保留 PR360 的 math-WG global
`__ldg` software-prefetch 路径，SFB 不进 SMEM。后续再独立 A/B “TMA producer搬 SFB
到 SMEM并广播”；两条路径不能在同一个结果里混写。

L2 output 首版写 local BF16 ring；pure control 还要提供 direct peer scatter policy。

### 7.4 cluster

v0 固定 cluster1。SM100 的 `SM100_TMA_2SM_LOAD_2D` 不存在于 Hopper；cluster2实验必须
改成 leader-issued `SM90_TMA_LOAD_MULTICAST_2D`。cluster2 以后只用于两个目的：

1. A tile multicast，两个 ComputeCTA 分 N；
2. ComputeCTA→RouteCTA DSM handoff。

Hopper 没有 2-CTA WGMMA。任何 cluster2 配置都必须证明：

- cluster 内全 CTA角色完整；
- prefix offset 与 cluster size 对齐；
- physical cluster rank 与 logical scheduler rank一致；
- cluster barrier没有角色缺席；
- active cluster 数足以覆盖 cooperative grid。

## 8. 前进与死锁证明

v0 必须同时满足：

1. 所有 CTA 同时驻留；
2. RouteCTA 不依赖 ComputeCTA 才能完成 dispatch；
3. ComputeCTA 只在 dispatch release 后读取 input；
4. wave L1 barrier 前，每个 ComputeCTA 已完成自己的静态 L1 task；
5. L2 output ring满时，至少一个 RouteCTA 只做 ring drain，不阻塞等 combine；
6. consumer只领取 `[call_begin,call_end)`，slot必须经过 free→ready→free sequence；
7. combine 只等待有效 top-k slot，且每个 slot全部 N tile完成后才发布；
8. cleanup/双缓冲初始化不覆盖当前 epoch payload或 counter；
9. peer 退出前完成最后一个 system-scope release；
10. kernel 返回/host复用 input 前，完成 dispatch-pull rank handshake；
11. timeout 只作诊断，不能掩盖 protocol error。

等待图应保持单向：

```text
Route dispatch -> Compute L1 -> Compute L2 -> Route scatter -> Route combine
```

RouteCTA 的阶段自适应不能生成反向依赖。

## 9. 最容易失败的四个点

### 9.1 少掉的 compute SM 比通信收益更大

这是第一风险。必须扫 RouteCTA 数，并测 same-policy compute-subgrid。若
`G=132-R` 的 pure L1/L2 已经比 PR360 全 SM math慢很多，后续调度不会救回来。

### 9.2 L2 local ring 的 HBM/L2 税

PR360 从 epilogue 直接 peer scatter；我方多一次 BF16 local write/read。必须测：

```text
direct peer epilogue
local ring + RouteCTA
DSM handoff（后续）
```

### 9.3 L1/L2 过早流水导致 compute CTA 饥饿或死锁

v0 先用 wave barrier；只有 telemetry 证明 L1→L2 bubble 足够大，才上 K-group admission
credit。

### 9.4 sparse expert 尾块

默认 benchmark 中期望 tokens/expert 只有 `T/4`，T≤32 时不超过 8。WGMMA M64 的
有效行很少，路由、barrier、weight load 和尾浪都可能比 tensor core更重要。必须单列
useful-row ratio，不能只看 nominal TFLOPS。

## 10. 落地顺序

### M0：对手冻结

- [ ] 记录 PR360 commit、submodule、CUDA、driver、GPU clock；
- [ ] 原样跑 auto/pingpong/cooperative；
- [ ] exact run保留 token-capacity 384与非空 `cum_stats`；
- [ ] 将原 harness 改为 10+50 rank-max，但保留原始结果；
- [ ] NCU 拆 dispatch、L1、L1 epi、L2、scatter、barrier、combine；
- [ ] 记录每 shape 的 active experts、tokens/expert 分布。

### M1：协议与 reference

- [ ] 在 `fuse/megamoe` 定义独立 problem/layout/route descriptor；
- [ ] 复用相同 quantization contract 和 transformed weights；
- [ ] 写 CPU/PyTorch reference；
- [ ] symmetric buffer、epoch、P2P atomic smoke；
- [ ] 单独验证 count/metadata/dispatch 和 combine。

### M2：pure compute

- [ ] L1 M64 pingpong；
- [ ] L1 M128 cooperative；
- [ ] SwiGLU/top-k/per64 quant exactness；
- [ ] L2 FP8 GEMM；
- [ ] same-policy full-SM 和 `132-R` SM throughput；
- [ ] ptxas register/SMEM/spill 报告。

### M3：CTA-specialized v0

- [ ] monolithic cooperative role split；
- [ ] RouteCTA dispatch -> L1 ready；
- [ ] safe expert-wave L1 -> L2；
- [ ] L2 local ring -> RouteCTA scatter；
- [ ] full-contribution combine-ready，无 L2 全 rank compute barrier；
- [ ] 连续 epoch correctness；
- [ ] memcheck/racecheck/initcheck/synccheck。

### M4：性能冲刺

- [ ] RouteCTA `{8,12,16,20}`；
- [ ] ring slots `{2,4,8}`；
- [ ] M frontier `{2,3,4}`；
- [ ] L2 M-major/N-major；
- [ ] L1/L2 admission credits；
- [ ] direct-peer vs local-ring；
- [ ] cluster2 multicast；
- [ ] DSM handoff，仅在 local-ring 税被证实后尝试。

## 11. 测试矩阵

### 11.1 必须复现的 PR360 默认点

```text
EP8, H7168, I2048, E256, topk8
T = 1,2,4,8,16,32
```

### 11.2 补充 token bands

```text
T = 64,128,256,512,1024,4096,8192
```

覆盖 auto policy threshold、compute-bound 区域和长尾。

### 11.3 模型 shape

至少再加：

```text
H4096, I2048, E256, topk6
H7168, I2048, E256, topk8
H7168, I3072, E384, topk6
```

另加一行待确定的目标模型 shape；不为某一个模型把 API 写死。

### 11.4 routing distribution

- balanced；
- random uniform；
- Zipf/skew；
- all-to-one hot expert；
- local-rank heavy；
- remote-rank heavy；
- masked top-k；
- zero token；
- expert M tail。

## 12. 性能报告口径

DeepGEMM 原 harness 是各 rank 的本地 CUDA-event，通常只由 rank0打印；它没有对每个
sample做 all-rank max。最终报告必须分开三张表，不能用改过的计时器冒充“原样复现”：

1. **exact harness**：保留原来的 input copy、`torch.empty(y)`、fused call 和 5+1 采样；
   `T<=32` 时仍使用原代码的 `num_max_tokens_per_rank=384` JIT/workspace specialization，
   并保留非空 `cum_stats` 及每 local expert 的 cleanup统计 atomic；
2. **rank-critical E2E**：同样包含 input copy，每个 sample all-gather 各 rank latency后
   取最大值，10 warmup + 50 samples；
3. **kernel-only**：双方输入已在各自合法 symmetric workspace，同样按 per-sample
   rank-max 统计。

rank-critical 与 kernel-only 表每行同时报告：

| 字段 | 说明 |
|---|---|
| max-rank mean/p50/p95 | 10 warmup + 50 samples |
| tokens/s | 固定 global token 口径 |
| useful TFLOPS | 只按真实 received tokens 计三次 GEMM |
| HBM bytes/s | 明确是否解析值或 NCU counter |
| NVLink bytes/s | 区分 payload 与 remote payload |
| pure compute | full 132 SM |
| compute subgrid | 与 fused 相同 `G=132-R` |
| route only | dispatch 和 combine 分列 |
| fused / PR360 | 最终核心指标 |
| fused / pure | CTA-specialization 对 GEMM 上界的保持比例 |

正式对手：

```text
DeepGEMM ec757bd auto
DeepGEMM ec757bd forced pingpong
DeepGEMM ec757bd forced cooperative
```

源码附带的 `DeepEP + grouped GEMMs + TileLang activation + combine` 使用 per-128 L1
requant，PR360 fused中间 activation是 per-64，因此只列作 **legacy non-equivalent perf
reference**。只有重写成相同 per-64 scale合同后，才能进入正式胜负表。

“打赢”的最低标准：相同输入 copy 口径、相同 dtype/scale、相同 routing seed、相同
rank-critical timing 下，每个主 token band 都优于 PR360 auto；不能只挑一个稀疏点。

## 13. 需要逐项决定的问题

按以下顺序做决定：

1. [ ] v0 选择 compact-valid-row BF16 ring，还是保留 L2 direct-peer scatter？
2. [ ] RouteCTA 初始数量从 8 还是 12 开始？
3. [ ] v0 是否完全保留 PR360 的 M64/M128 compute policy？
4. [ ] expert wave 是否先严格 L1-all -> L2-all？
5. [ ] combine 第一版使用完整 contribution epoch，还是保留 PR360 rank barrier？
6. [ ] L1→L2 使用两个 half-ready epoch，还是使用带 epoch 的旧 uint64 mask？
7. [ ] 目标主模型除了 PR360 默认 H7168/I2048 外选哪一个？
8. [ ] 何时引入 cluster2/DSM；触发条件应是 local-ring 税的 NCU 证据。

推荐默认答案：`compact-valid-row bounded ring / R=8 起扫 / 保留原 compute policy /
保留安全 wave / 完整 contribution epoch / 两个 half-ready epoch / 先跑默认 shape /
DSM 最后`。
