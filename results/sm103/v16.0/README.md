# 新模型前向融合 vs 纯 GEMM

v16.0 为 benchmark-only 发布：算子实现与 v15.0 完全一致。

## 数据与复现

- [逐点配置 CSV](table.csv)、[分组纯 GEMM 表](pure-gemm-grouped.md)。
- [融合胜者样本](fused-samples.json)：56个胜者各50个原始max-rank耗时（ms）。
- [纯GEMM样本](pure-samples.json)：主表使用的去重M/N/K、算法ID及原始样本；
  扩展纯投影目录的其余数据只收录汇总，不复制全部调优过程。
- [证据索引](audit.json)：节点、源码、环境、二进制及外部归档哈希。
  CSV中的external-raw是外部原始归档标识，不是仓库内文件。
  全候选/rank校验留在外部归档；这里的胜者样本不替代完整原始证据。
- [模型形状来源与分组限制](../../../benchmarks/sm103/PROJECTION_SHAPES.md)、
  [构建和测量合同](../../../benchmarks/sm103/README.md)。

使用既有环境和授权节点重测示例；其他点替换模型尺寸及CSV中的配置：

```bash
python3 scripts/l20d.py run fused-build --node 09 --mpi --experiment reproduce-v16
python3 scripts/l20d.py run fused-smoke --node 09 --mpi --fused-launch graph \
  --fused-direction oproj --experiment reproduce-v16 --world 8 --global-seq 131072 \
  --hidden 4096 --q-heads 64 --kv-heads 4 --head-dim 128 \
  --comm-sm-list 8,16,24 --oproj-policy-list m128n128,m128n256k64e32 \
  --input-generator gpu_philox --causal --max-swizzle-size 4 --oproj-raster along_n \
  --timeout 600 --timeout-seconds 120
python3 benchmarks/sm103/projection_shapes.py --models deepseek_v3 --grouped
```

独立方向入口只改变测试资源分配，不修改算子算法；默认both保留。
扩展模型的特殊输入路由不在本版实现范围内。

## 完整表

BF16 / Graph / 节点09 / 全局 S=64K～512K / CP4、8。PFLOPS均为单卡口径；融合采用跨 rank 最大延迟的 p50，纯 cuBLASLt 来自 GPU0 同 M/N/K 测量。保留比例 = 融合 PFLOPS / 纯 GEMM PFLOPS，不是相对硬件理论峰值的 MFU。

已验收 38 个自研运行、336 个候选，填入 **56/80** 项。每方向两档 tile × 通信 CTA 8/16/24；选候选内最小 p50，不声称全局最优，也未追加独立胜者复测。随机输入、至少10+50，融合完整数值/路由和两组 payload 校验；纯 GEMM 为已有数值 spot-check 协议。

缺失24项：KDA六路输入路由暂缓16项；Qwen3-235B的QKV在CP8下KV头数不整除4项；BLOOM 512K双方向/CP4、8因先前成对测试显存限制跳过4项，未重试，也不据此断言独立OProj一定OOM。MLA/GDN特殊输入路由不扩展，纯投影数据见 pure-gemm-grouped.md；KDA旧QKV-only不冒充六路。

TEUB/Lt+NCCL已有列原地保留，空白不补测。逐点 tile、通信 CTA、raster/swizzle、p50/p95、源码/run、纯库算法及原始记录定位见table.csv；Graph、causal_dual_chunk_v1、gpu_philox、seed=20260906、rank旋转关闭。低于纯 GEMM 80%的已测点加粗，阈值仅为阅读标记。

| 模型 | S | CP | 方向 | 自研 PFLOPS | 纯 GEMM PFLOPS | 保留比例 | 相对TEUB | 相对Lt+NCCL |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| qwen25_72b | 65536 | 4 | qkv | 1.242 | 1.526 | 81.4% | 1.113× | 1.204× |
| qwen25_72b | 65536 | 4 | oproj | 1.161 | 1.434 | 81.0% | 0.948× | 1.072× |
| qwen25_72b | 65536 | 8 | qkv | 1.259 | 1.405 | 89.7% |  |  |
| qwen25_72b | 65536 | 8 | oproj | 1.126 | 1.401 | 80.4% |  |  |
| qwen25_72b | 131072 | 4 | qkv | 1.257 | 1.505 | 83.6% | 1.111× | 1.205× |
| qwen25_72b | 131072 | 4 | oproj | 1.170 | 1.516 | **77.2%** | 0.903× | 1.091× |
| qwen25_72b | 131072 | 8 | qkv | 1.250 | 1.526 | 82.0% |  |  |
| qwen25_72b | 131072 | 8 | oproj | 1.130 | 1.434 | **78.8%** |  |  |
| qwen25_72b | 262144 | 4 | qkv | 1.246 | 1.494 | 83.4% | 1.110× | 1.193× |
| qwen25_72b | 262144 | 4 | oproj | 1.179 | 1.492 | **79.0%** | 0.967× | 1.092× |
| qwen25_72b | 262144 | 8 | qkv | 1.254 | 1.505 | 83.4% |  |  |
| qwen25_72b | 262144 | 8 | oproj | 1.142 | 1.516 | **75.4%** |  |  |
| qwen25_72b | 524288 | 4 | qkv | 1.224 | 1.478 | 82.8% | 1.106× | 1.128× |
| qwen25_72b | 524288 | 4 | oproj | 1.183 | 1.479 | 80.0% | 0.966× | 1.098× |
| qwen25_72b | 524288 | 8 | qkv | 1.245 | 1.494 | 83.3% |  |  |
| qwen25_72b | 524288 | 8 | oproj | 1.139 | 1.492 | **76.3%** |  |  |
| qwen3_235b | 65536 | 4 | qkv | 1.174 | 1.441 | 81.5% | 1.237× | 1.501× |
| qwen3_235b | 65536 | 4 | oproj | 1.140 | 1.439 | **79.2%** | 0.999× | 1.512× |
| qwen3_235b | 65536 | 8 | qkv |  | 1.395 |  |  |  |
| qwen3_235b | 65536 | 8 | oproj | 1.102 | 1.312 | 84.0% |  |  |
| qwen3_235b | 131072 | 4 | qkv | 1.167 | 1.486 | **78.5%** | 1.236× | 1.494× |
| qwen3_235b | 131072 | 4 | oproj | 1.139 | 1.415 | 80.5% | 1.002× | 1.411× |
| qwen3_235b | 131072 | 8 | qkv |  | 1.441 |  |  |  |
| qwen3_235b | 131072 | 8 | oproj | 1.131 | 1.439 | **78.6%** |  |  |
| qwen3_235b | 262144 | 4 | qkv | 1.175 | 1.493 | **78.7%** | 1.253× | 1.460× |
| qwen3_235b | 262144 | 4 | oproj | 1.141 | 1.428 | **79.9%** | 0.998× | 1.400× |
| qwen3_235b | 262144 | 8 | qkv |  | 1.486 |  |  |  |
| qwen3_235b | 262144 | 8 | oproj | 1.132 | 1.415 | **80.0%** |  |  |
| qwen3_235b | 524288 | 4 | qkv | 1.154 | 1.476 | **78.2%** | 1.257× | 1.445× |
| qwen3_235b | 524288 | 4 | oproj | 1.139 | 1.482 | **76.9%** | 1.033× | 1.361× |
| qwen3_235b | 524288 | 8 | qkv |  | 1.493 |  |  |  |
| qwen3_235b | 524288 | 8 | oproj | 1.112 | 1.428 | **77.9%** |  |  |
| bloom_176b | 65536 | 4 | qkv | 1.278 | 1.464 | 87.3% |  |  |
| bloom_176b | 65536 | 4 | oproj | 1.125 | 1.491 | **75.5%** |  |  |
| bloom_176b | 65536 | 8 | qkv | 1.284 | 1.482 | 86.7% |  |  |
| bloom_176b | 65536 | 8 | oproj | 1.033 | 1.470 | **70.3%** |  |  |
| bloom_176b | 131072 | 4 | qkv | 1.265 | 1.462 | 86.5% |  |  |
| bloom_176b | 131072 | 4 | oproj | 1.125 | 1.489 | **75.6%** |  |  |
| bloom_176b | 131072 | 8 | qkv | 1.276 | 1.464 | 87.1% |  |  |
| bloom_176b | 131072 | 8 | oproj | 1.071 | 1.491 | **71.8%** |  |  |
| bloom_176b | 262144 | 4 | qkv | 1.255 | 1.444 | 86.9% |  |  |
| bloom_176b | 262144 | 4 | oproj | 1.117 | 1.473 | **75.8%** |  |  |
| bloom_176b | 262144 | 8 | qkv | 1.265 | 1.462 | 86.5% |  |  |
| bloom_176b | 262144 | 8 | oproj | 1.035 | 1.489 | **69.5%** |  |  |
| bloom_176b | 524288 | 4 | qkv |  | 1.439 |  |  |  |
| bloom_176b | 524288 | 4 | oproj |  | 1.465 |  |  |  |
| bloom_176b | 524288 | 8 | qkv |  | 1.444 |  |  |  |
| bloom_176b | 524288 | 8 | oproj |  | 1.473 |  |  |  |
| kimi_k3_kda | 65536 | 4 | qkv |  | 1.461 |  |  |  |
| kimi_k3_kda | 65536 | 4 | oproj | 1.158 | 1.444 | 80.2% |  |  |
| kimi_k3_kda | 65536 | 8 | qkv |  | 1.524 |  |  |  |
| kimi_k3_kda | 65536 | 8 | oproj | 1.128 | 1.361 | 82.9% |  |  |
| kimi_k3_kda | 131072 | 4 | qkv |  | 1.474 |  |  |  |
| kimi_k3_kda | 131072 | 4 | oproj | 1.171 | 1.489 | **78.6%** |  |  |
| kimi_k3_kda | 131072 | 8 | qkv |  | 1.461 |  |  |  |
| kimi_k3_kda | 131072 | 8 | oproj | 1.148 | 1.444 | **79.5%** |  |  |
| kimi_k3_kda | 262144 | 4 | qkv |  | 1.472 |  |  |  |
| kimi_k3_kda | 262144 | 4 | oproj | 1.175 | 1.487 | **79.0%** |  |  |
| kimi_k3_kda | 262144 | 8 | qkv |  | 1.474 |  |  |  |
| kimi_k3_kda | 262144 | 8 | oproj | 1.154 | 1.489 | **77.5%** |  |  |
| kimi_k3_kda | 524288 | 4 | qkv |  | 1.471 |  |  |  |
| kimi_k3_kda | 524288 | 4 | oproj | 1.197 | 1.476 | 81.1% |  |  |
| kimi_k3_kda | 524288 | 8 | qkv |  | 1.472 |  |  |  |
| kimi_k3_kda | 524288 | 8 | oproj | 1.175 | 1.487 | **79.0%** |  |  |
| kimi_linear_48b_kda | 65536 | 4 | qkv |  | 1.442 |  |  |  |
| kimi_linear_48b_kda | 65536 | 4 | oproj | 0.936 | 1.324 | **70.7%** |  |  |
| kimi_linear_48b_kda | 65536 | 8 | qkv |  | 1.378 |  |  |  |
| kimi_linear_48b_kda | 65536 | 8 | oproj | 0.843 | 1.209 | **69.7%** |  |  |
| kimi_linear_48b_kda | 131072 | 4 | qkv |  | 1.443 |  |  |  |
| kimi_linear_48b_kda | 131072 | 4 | oproj | 0.975 | 1.372 | **71.1%** |  |  |
| kimi_linear_48b_kda | 131072 | 8 | qkv |  | 1.442 |  |  |  |
| kimi_linear_48b_kda | 131072 | 8 | oproj | 0.908 | 1.324 | **68.5%** |  |  |
| kimi_linear_48b_kda | 262144 | 4 | qkv |  | 1.446 |  |  |  |
| kimi_linear_48b_kda | 262144 | 4 | oproj | 0.918 | 1.448 | **63.4%** |  |  |
| kimi_linear_48b_kda | 262144 | 8 | qkv |  | 1.443 |  |  |  |
| kimi_linear_48b_kda | 262144 | 8 | oproj | 0.937 | 1.372 | **68.3%** |  |  |
| kimi_linear_48b_kda | 524288 | 4 | qkv |  | 1.433 |  |  |  |
| kimi_linear_48b_kda | 524288 | 4 | oproj | 0.919 | 1.435 | **64.0%** |  |  |
| kimi_linear_48b_kda | 524288 | 8 | qkv |  | 1.446 |  |  |  |
| kimi_linear_48b_kda | 524288 | 8 | oproj | 0.915 | 1.448 | **63.2%** |  |  |
