# v17.0：OProj 长序列融合归档

本版范围为节点09（用户说明硬件为B300；CUDA Runtime 10.3、148 SM）、BF16/FP32累加/BF16输出、Graph、causal_dual_chunk_v1、五模型 × S=128K/256K/512K × CP4/8，共30点。23点完成数值与路由校验；7点融合结果留空。

在同一轮、同进程、固定该shape的GEMM配置下，模型Top2通信CTA候选实测选优，相对原通信预算的吞吐几何平均提升 **15.7655%**。这不是相对v16或历史融合run的提升。23个有效点相对同M/N/K、GPU0满148SM可用预算的历史纯cuBLASLt分母，吞吐几何保留 **85.8277%**。两个统计均对23个逻辑点等权，不把空白计成零，也不代表全30点覆盖。

融合采用明确授权的Graph **1次预热+5次采样快测**，没有收敛门禁，formal_eligible=false；23点中5点的baseline或winner至少一方半段p50漂移超过5%（其中winner自身4点）。数值/路由verified不代表正式性能稳定性通过。Top2在候选实测后选最小p50，没有独立胜者复测，不声称全局最优；runtime自动选优留到下一版。

纯cuBLASLt采用历史Graph独立选算法和至少10+50协议，来自两个历史run；保留全部已用算法及原始测量样本，不视为本轮同场重测。满SM分母的math_sms=0表示不限制本卡148个SM，与缩减compute CTA预算的独立GEMM不同。两边输入生成协议独立，未声称随机数逐位相同。

SM90未改；新QKV全量未测，本归档不扩大到QKV、KDA六路特殊路由或新精度的性能结论。Kimi仅为已适配的OProj边界。

## 数据

- [table.csv](table.csv)：30点最新表、完整复现尺寸、GEMM配置、原预算/胜者通信预算、漂移、run及样本ID。
- [samples.json](samples.json)：43份去重baseline/winner记录，保留每份1次预热和5次采样的逐rank及max-rank毫秒值；19个去重纯Lt几何记录，保留10次采样节奏预热、50样本、算法与校验。
- [audit.json](audit.json)：源码/环境/二进制远端证明、外部原始归档与提取文件SHA256、候选选择核对、输入统计和整进程遥测。哈希索引指向外部保留证据，不包含大profile、安装包或完整调优过程。二进制哈希为远端构建证明，不是本地重新编译结果。

样本ID可从CSV直接关联JSON。p50/p95由max-rank样本线性分位数重算；PFLOPS=2MNK/(p50_ms×10^12)，保留比例=fused_pflops/pure_pflops。原预算baseline和winner使用同一GEMM配置，改变的是通信预算；原预算也可能属于Top2，同一实测记录只保存一次。

7个空白包括3个原先缺失/已知OOM点，以及4个本轮显存预检不足、kernel未启动点；原因逐点保留。Llama405B CP4/512K在指定历史表中也没有纯Lt分母，因此纯列同样留空。

## 复现

从仓库根目录，使用已有环境与授权节点重测。以下对应Qwen2.5-72B、CP4、128K：固定GEMM配置，原预算8，模型Top2为32/24。

```bash
python3 scripts/l20d.py run fused-build --node 09 --mpi --experiment reproduce-v17
python3 scripts/l20d.py run fused-smoke --node 09 --mpi --quick \
  --fused-launch graph --fused-direction oproj --experiment reproduce-v17 \
  --world 4 --global-seq 131072 --hidden 8192 --q-heads 64 --kv-heads 8 --head-dim 128 \
  --comm-sm-list 8,32,24 --oproj-policy m128n256 --oproj-raster along_m \
  --max-swizzle-size 8 --oproj-comm-layout rows --input-generator gpu_philox --causal \
  --timeout 600 --timeout-seconds 300
python3 scripts/l20d.py run gemm-probe --node 09 --experiment reproduce-v17-pure \
  --directions oproj --launches graph --world 4 --global-seq 131072 \
  --hidden 8192 --q-heads 64 --kv-heads 8 --head-dim 128 --cublaslt-sm-target 0
```

其他点替换CSV中的尺寸、tile/raster/swizzle、原预算及Top2预算。去掉--quick可按默认10+50合同追加正式测量，新结果必须独立记账。已有外部run可用scripts/summarize_sm103_fused.py的本地audit_run重新验收；该接口只读回传材料。本次归档已逐run重新验收，不产生新GPU测量。

## 最新结果

PFLOPS为单卡工作量口径；融合延迟使用逐样本跨rank最大值的p50。`*`表示baseline或winner漂移超过5%。

| 模型 | S | CP | 原预算 PFLOPS | 胜者 PFLOPS | 纯Lt PFLOPS | 保留 | 通信CTA 原→胜 |
|---|---:|---:|---:|---:|---:|---:|---:|
| llama31_405b* | 131072 | 4 | 1.407 | 1.407 | 1.503 | 93.58% | 8→8 |
| llama31_405b | 131072 | 8 | 1.076 | 1.282 | 1.478 | 86.75% | 8→24 |
| llama31_405b* | 262144 | 4 | 1.071 | 1.213 | 1.478 | 82.02% | 8→24 |
| llama31_405b | 262144 | 8 |  |  | 1.503 |  |  |
| llama31_405b | 524288 | 4 |  |  |  |  |  |
| llama31_405b | 524288 | 8 |  |  | 1.478 |  |  |
| qwen25_72b | 131072 | 4 | 0.947 | 1.395 | 1.516 | 92.04% | 8→24 |
| qwen25_72b | 131072 | 8 | 1.141 | 1.269 | 1.434 | 88.48% | 16→32 |
| qwen25_72b | 262144 | 4 | 0.955 | 1.384 | 1.492 | 92.73% | 8→24 |
| qwen25_72b | 262144 | 8 | 1.163 | 1.286 | 1.516 | 84.85% | 16→32 |
| qwen25_72b* | 524288 | 4 | 0.949 | 1.353 | 1.479 | 91.49% | 8→24 |
| qwen25_72b | 524288 | 8 |  |  | 1.492 |  |  |
| qwen3_235b | 131072 | 4 | 1.075 | 1.306 | 1.415 | 92.30% | 16→32 |
| qwen3_235b | 131072 | 8 | 0.920 | 1.173 | 1.439 | 81.50% | 16→32 |
| qwen3_235b | 262144 | 4 | 1.387 | 1.387 | 1.428 | 97.13% | 16→16 |
| qwen3_235b | 262144 | 8 | 0.944 | 0.987 | 1.415 | 69.75% | 24→32 |
| qwen3_235b | 524288 | 4 | 1.359 | 1.359 | 1.482 | 91.68% | 16→16 |
| qwen3_235b | 524288 | 8 | 1.178 | 1.245 | 1.428 | 87.22% | 24→32 |
| bloom_176b | 131072 | 4 | 1.047 | 1.295 | 1.489 | 86.98% | 8→24 |
| bloom_176b | 131072 | 8 | 1.095 | 1.344 | 1.491 | 90.13% | 8→24 |
| bloom_176b* | 262144 | 4 | 1.033 | 1.242 | 1.473 | 84.33% | 8→24 |
| bloom_176b | 262144 | 8 | 1.044 | 1.269 | 1.489 | 85.25% | 8→24 |
| bloom_176b | 524288 | 4 |  |  | 1.465 |  |  |
| bloom_176b | 524288 | 8 |  |  | 1.473 |  |  |
| kimi_k3_kda | 131072 | 4 | 1.127 | 1.205 | 1.489 | 80.91% | 16→32 |
| kimi_k3_kda | 131072 | 8 | 1.071 | 1.129 | 1.444 | 78.18% | 16→32 |
| kimi_k3_kda | 262144 | 4 | 1.128 | 1.232 | 1.487 | 82.85% | 16→32 |
| kimi_k3_kda | 262144 | 8 | 1.081 | 1.151 | 1.489 | 77.30% | 16→24 |
| kimi_k3_kda* | 524288 | 4 | 1.047 | 1.208 | 1.476 | 81.89% | 16→24 |
| kimi_k3_kda | 524288 | 8 |  |  | 1.487 |  |  |
