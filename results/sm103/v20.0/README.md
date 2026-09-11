# v20.0 — MXFP8 QKVProj 前向最终结果

- [完整性能表](table.md)：Auto、同场手工、保存的手工最佳、纯 cuBLASLt。
- [机器可读表](table.csv)：全部尺寸、吞吐和实际 GEMM/通信配置。
- [独立手工最佳 SOTA](manual-best.md)：保留已确认配置，不用最快重测覆盖历史。
- [最终结果与证据](results.json)：配置、p50/p95、校验、运行及构建哈希。

只发布最终测量，不包含候选搜索过程、profile、原始逐样本日志或安装包。
完整原始证据留在本地 fuse_midfile 的对应 run 与唯一验收/手工最佳归档。

## 边界与验收

融合输入为预量化 MXFP8 activation 与 BF16 master weight；一次持久化 kernel
内部量化权重、MXFP8 E4M3/UE8M0 GEMM（FP32 累加）及 BF16 输出 A2A。
激活值的上游量化不计时。纯 cuBLASLt 是相同 M/N/K、预量化操作数、满设备
148 SM 可用预算的历史纯 GEMM，不含权重量化和通信，故百分比不是硬件峰值 MFU。
每卡工作量口径为 `2*M*N*K/(p50_ms*1e12)`；融合取每样本跨 rank 最大耗时。

33/33 物理点全部完成 Graph10+50、两个可复现随机 payload 的完整数值/路由
检查。Qwen2.5 72B 与 Llama 3.1 70B 同 shape，表中分名展示，统计不重复。
Qwen3 CP8 未适配，不冒充已测；Kimi 仅 QKV-only，不包括额外门控投影。

Auto 实际向 API 传入 `comm_ctas=0`，所有 rank 查询与 launch 预算一致；
手工对照重放独立离线选出的固定配置，使用同一个二进制、job、GEMM 和输入。
33 点 Auto/同场手工几何平均 **96.6929%**，最低 **90.0858%**，达到用户给定
的平均至少95%、每点严格大于90%的验收线。最低点接近边界，不能保证重复测量
或其他共享负载下仍高于90%。最大已接受半段漂移4.2029%，均遵守原5%测量门限。

11个128K标定序列点平均95.4402%；22个未参与标定的256K/512K点平均97.3253%。
这些是吞吐保留率，不是模型预测误差，更不是所有未见 N/K/布局上的泛化证明。
模型只使用固定 GEMM 输入及独立服务数据；无在线试跑、融合赢家反拟合或阈值拟合。

历史原值与同场重测分列：Dense CP4/512K 对保存的手工吞吐只有 **88.3915%**，
对同场手工为90.0858%，不可混称两种口径逐点都通过。历史手工最佳只代表有限
候选搜索并独立确认，不宣称全局最优。原先 BLOOM CP4/512K 因显存缺少分母，
本次补16/24/32预算选择和独立确认后纳入全部33点；模型没有因此修改。

## 配置与复现

当前受验收 GEMM：M128/N256/K128、epilogue N32、自动 stages（实际4）、
AlongM/AlongN 和 swizzle 按表中固定输入；Auto 只选通信 CTA，不自动搜索 GEMM。
显式正通信预算可复现手工 SOTA；`0` 使用模型；没有匹配标定则明确拒绝。
数值校验通过不代表与 BF16 或其他 cuBLASLt 算法 bitwise 相同。

按仓库正常 SM103a CUDA13 + pinned CUTLASS/MPICH 构建；参见
[测量接口](../../../benchmarks/sm103/README.md) 与
[公开入口](../../../include/fuse/operators/primitives/gemm_a2a_mxfp8.h)。
构建入口：`python3 scripts/l20d.py run build --node 09 --workspace /home/work/workspace_wct --mpi`。
测试模板（其他 shape/GEMM 参数见 CSV）：

```bash
python3 scripts/l20d.py run fused-smoke --node 09 \
  --workspace /home/work/workspace_wct --mpi --mxfp8 --auto-mxfp8-comm \
  --fused-direction qkv --directions qkv --fused-launch graph \
  --input-generator gpu_philox --qkv-policy-list m128n256 \
  --mxfp8-weight-preparation comm --mxfp8-epilogue-n 32 \
  --qkv-raster along_m --max-swizzle-size 8 --comm-sm 32 \
  --world 8 --global-seq 131072 --hidden 8192 \
  --q-heads 64 --kv-heads 8 --head-dim 128 \
  --experiment v20-qwen72-cp8-128k-replay
```

`--comm-sm 32 --auto-mxfp8-comm` 测显式32及真实Auto两个候选，默认正式10+50；
不是先测候选再把赢家传入Auto。控制器读取当前 screen 并检查设备资源。
正式 binary SHA256：`14ba624ba3bdcde405a8aa53837fc3b2eb83cde326c8d061511f6ad0b966041b`。
校准版本保留 `validation_pending_6714eeeb...` 原始标识以保持被测二进制身份；
验收结果由本文及结果哈希证明，不通过改字符串冒充新的测试。

发布前主机回归716项，5项按既有条件跳过，其余通过。91个CUDA/C++/构建文件
与正式测量构建快照逐字节一致；后续只补最终表、说明与主机测试，不改已测kernel。

MXFP8 A2A→OProj、反向及新的特殊路由留待下一版本；既有 BF16 结果继续引用
[v19.0](../v19.0/README.md)，不复制或伪造新测试。
