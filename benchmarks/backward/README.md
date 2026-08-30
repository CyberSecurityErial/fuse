# Backward benchmark harness

这个目录只放两条反向算子共用的测试脚手架，不是第三个生产算子。

- `backward_smoke.cu` 用独立参考实现检查 QKV 与 OProj 的 B/W 数学、跨卡布局、普通同流模式、ZeroBubble 分离模式和梯度累加；
- `backward_mpi_bench.cu` 负责一进程一卡的 Eager/Graph 计时与 CUDA IPC；
- `backward_cublas_mpi_bench.cu` 用相同卡组和反向 MNK 测两次经典 cuBLAS 纯 GEMM；
- `backward_shape_bench.py` 枚举相同的模型、序列长度和 CP 矩阵，并分别生成两个算子的结果。
- `backward_torch_autograd.py` 先用 `torch.nn.functional.linear` 做前向，
  再让 PyTorch autograd 生成真正的 `dX/dW`，用来核对融合反向，而不是拿另一份
  手写矩阵乘当“参考答案”。用例覆盖 CP4/CP8、rank-major/causal、batch=2，
  以及 Q40/KV8、QKV 宽度 7168 的宽 GQA shape；参考侧关闭 TF32、启用
  PyTorch 确定性算法，并固定 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；
- `backward_te_nccl_baseline.py` 和 `backward_te_shape_bench.py` 是轻量调参的
  TE GEMM + NCCL 对照；每组 NCCL 环境变量都在新进程里生效。正式版本固定
  cuBLAS 确定性配置，并在 S=1K 把 route、DGrad、WGrad 各连续执行两次，要求
  输出逐元素一致；
- `backward_te_userbuffers.py` 和它的 shape driver 是重点 TE Userbuffers 对照。
  正式数值先收齐完整路由结果，再做一次完整 BF16 GEMM；逐 peer 的 BF16
  `beta=1` 分段累加只保留作诊断，不能用不同的舍入顺序换取一个看起来更快的数字。

每份正式 raw 都直接记录 B/W MNK；聚合器会重新核对 shape、50 个有限样本、
普通模式的 `beta=0`、ZeroBubble 的 `beta=1`、自动请求和最终 tile。发布目录只保留
96 行 JSON/CSV/Markdown 汇总，逐次 MPI raw 留在用户指定的临时结果目录。

TE Userbuffers 的通信资源会搜索 `4/8/12/16/20/24` 个 SM；stream 数、
push/pull、Copy Engine、pack block/warp
和 peer 顺序会先做短采样搜索，winner 再用 10 次预热和 50 次正式采样重测。
单变量搜索后还会把各轴的实测 winner 合并，只复测四种 push/pull×CE/SM
传输组合；这样能覆盖主要参数交互，又不需要为每个 shape 穷举 144 个组合。
短采样排名靠前的三组配置都会重新做正式 10+50，最终表只取正式复测 winner，
不会把一次 12-sample 的偶然低值直接当成强基线。
Userbuffers 对象每个测试项重新创建；cuBLASLt 的纯算法计划只在相邻的同 shape
测试间复用，避免 Eager/Graph、普通/ZeroBubble 四列重复做完全相同的离线调优。
这些 setup 都不在 CUDA event 计时区间内。

两个 shape driver 都支持 `--publish-results`。搜索和逐次正式 raw 仍留在调用方指定
的临时目录；只有选出的 compact JSON/CSV/Markdown 会写入两个算子自己的
`results/*-backward/*_backward_shape_bench`，避免把几千个调参文件混进发布归档。

QKV 对照中的 Triton pack 对 Q、K、V 分别使用独立 load mask。原因很直接：
`tl.where` 只选择结果，并不会阻止另一侧的 load 执行；如果三块输入共用一个地址
表达式，Q 的较宽 stride 会在长序列下读出 K/V allocation 边界。raw JSON 用
`qkv_pack_kernel=branch_masked_v1` 标明已修复实现，旧结果不会被 resume 接受。

Userbuffers unpack 对本地 `send` 和远端 `recv` 也使用互斥 load mask。旧写法虽然
最终选择的数值正确，但会把两个缓冲区都读一遍，白白增加一倍 unpack 读取流量。
raw JSON 用 `ub_unpack_kernel=branch_masked_v1` 区分修复后的实现，搜索与正式结果
不会混入旧实现。

生产接口和参数结构仍完全分开：`qkv_backward.h` 不依赖 OProj 参数，`oproj_backward.h` 也不依赖 QKV 参数。共用 runner 只用于保证两边使用同一采样口径，不能把其中一个算子的 shape 或策略套到另一个算子。

PyTorch autograd 对照可独立复现：

```bash
cmake -S . -B build-v10-validation \
  -DCMAKE_BUILD_TYPE=Release -DFUSE_ENABLE_PROFILING=OFF
cmake --build build-v10-validation --parallel 8 \
  --target fuse_backward_torch_bridge

python3 benchmarks/backward/backward_torch_autograd.py \
  --bridge build-v10-validation/libfuse_backward_torch_bridge.so \
  --json-out /tmp/fuse-v10-torch-autograd.json
```

这个脚本默认同时运行两条算子、普通 B→W 和 ZeroBubble 分离 B/W。它还会从非零
`main_grad` 连续累加两次，确认 `beta=1` 不是只写在参数里而是真正生效。

本次正式验证运行了 4 组输入：CP4 rank-major、CP8 causal、CP4 batch=2 causal，
以及 CP8 宽 GQA rank-major。两条算子分别覆盖普通与 ZeroBubble，共 16 组；全部
通过，普通模式最大绝对误差为 0，ZeroBubble 连续累加的最大绝对误差为
`0.0009765625`。完整逐张量检查保存在
[`results/backward_autograd_validation.json`](../../results/backward_autograd_validation.json)。

完整语义、反向 MNK、复现命令和正式结果分别见：

- [QKV Projection backward](../QKVproj-backward/BENCHMARK.md)
- [Output Projection backward](../Oproj-backward/BENCHMARK.md)
