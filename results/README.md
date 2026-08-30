# Archived benchmark results

当前归档包含A2A -> O-projection历史Golden，以及132-SM H200 NVLink节点上的
QKV projection -> A2A结果，以及v9.0/v9.1锁频异构CP的精简正式矩阵。QKV外部基线继续使用独立调优的v4/v5归档；均匀融合侧Eager/Graph结果和对比表为v8.0正式96点采样。

v13.0 只重构源码和公开头文件，没有生成或替换结果文件。它与 v12.0 的 69 个
device function 逐函数 SASS 一致，因此这里继续保留原有 Golden，不复制一套相同数据。

v8归档需要同时保存用户请求的`comm_ctas=0`、实际解析出的通信CTA、实际tile和
policy模型版本。正式runner会把0原样传给launcher，第一次prime完成自动选择，后续
Eager样本复用该结果；不能在计时前把0替换成实际CTA，否则测不到真实自动入口。

v8还包含两项正确性修复：发布ready之前先等TMA目标显存完整写完；smoke按真实自动
tile分配ready缓冲区，覆盖N64。正式结果必须来自profiling关闭的构建，diagnostic
role profile不得混入Golden。

Measurement contract:

- BF16, CP4 and CP8;
- 10 warmup iterations and 50 measured iterations;
- every sample reports the maximum CUDA-event latency across ranks;
- the table uses p50 latency;
- the v3 snapshot fixes the v2 kernel and keeps `--lhs-policy auto`;
- 29 rows use the v2 automatic `comm_ctas=4/6/8` result, while seven rows
  use a formally measured external `comm_ctas` winner;
- the tile is selected from the maintained fixed CUTLASS candidates by the
  shared runtime cost model, with no per-shape manual tile override.

Committed directories:

- `a2a-Oproj/oproj_cluster_wave_bench`: current v2 fused 36-case summary after the
  cluster-frontier scheduler and narrow-shard 3D TMA store optimizations;
- `a2a-Oproj/oproj_shape_bench` and `a2a-Oproj/oproj_shape_bench_cp8`: selected CP4/CP8 external
  baseline configurations, fused results, and shape definitions;
- `a2a-Oproj/oproj_shape_bench_longseq_cp4` and `a2a-Oproj/oproj_shape_bench_longseq_cp8`: the
  corresponding 128K, 256K, and 512K aggregates, plus fused-only 1M entries;
- `a2a-Oproj/te_userbuffers_shape_bench` and `a2a-Oproj/te_userbuffers_shape_bench_longseq`: the
  selected TE Userbuffers configurations and formal results.
- `a2a-Oproj/oproj_v3_manual_comm_bench`: the complete 36-case v3 snapshot. Its
  `result_source` field distinguishes 29 `v2_auto_inherited` rows from seven
  `manual_comm_ctas` rows; all options other than the external communication
  CTA count remain fixed.
- `a2a-Oproj/oproj_mixed_shape_bench`: the v4.0 96-setting OProj matrix. It
  combines 36 exact v3 Golden rows, 24 same-geometry model labels, and 36
  newly calibrated real-model rows.
- `a2a-Oproj/te_userbuffers_mixed_shape_bench`: the matching adapted TE
  Userbuffers winners for all 96 OProj settings.
- `QKVproj-a2a/qkv_shape_bench`: 96个QKV setting。它保留独立调优的
  TE/cuBLAS/cuBLASLt+NCCL归档，并保存v8 Eager/Graph结果、N64在内的实际policy
  信息、自动方案缓存版本、经典cuBLAS吞吐和融合吞吐占比。
- `QKVproj-a2a/te_userbuffers_shape_bench`: the matching adapted TE
  Userbuffers QKV winners.
- `QKVproj-backward/qkv_backward_shape_bench`: v10 的 96 个 QKV 反向
  setting。汇总同时保存 B/W MNK、Eager/Graph、普通同流/ZeroBubble、匹配
  beta 的经典 cuBLAS、TFLOPS、989T MFU 和相对同 shape 前向吞吐；同目录还保存
  TE Userbuffers 96 点正式对照与 TE+NCCL 12 点轻量对照。
- `Oproj-backward/oproj_backward_shape_bench`: v10 的 96 个 OProj 反向
  setting，字段和采样口径与 QKV 反向相同；同目录也保存对应的 TE Userbuffers 与
  TE+NCCL 对照。两个算子的生产接口与自动策略仍独立。
- `backward_autograd_validation.json`: PyTorch forward→autograd backward 的逐张量
  正确性记录，覆盖 CP4/CP8、两种布局、batch=2、宽 GQA、普通与 ZeroBubble。
- `e2e/four_operator_graph_summary.csv` 与 `e2e/metadata.json`: v11.2 的完整训练
  CUDA Graph 对照。它同时替换 QKV/OProj 前向和反向，覆盖三种模型 geometry、
  五档序列长度和 CP4/CP8，保存原生/融合完整 step 时间、加速比、采样步骤和
  Qwen 128K 的临时工作区复用说明。
- `fp8/qkv_graph_summary.{json,csv,md}` 与 `fp8/metadata.json`：v12.0 的纯 E4M3 QKV Projection+A2A
  CUDA Graph 96 点汇总。输入、权重、通信数据和输出均为 E4M3，FP32 累加；
  量化与 scale 计算不在计时边界内。
- `heterogeneous-cp/locked_frequency_summary.csv`: v9.0 两个独立 weighted 算子的22点锁频矩阵。它覆盖CP2/4/6/8、不同慢卡数量和本地M=2048/16384，保存实际设备组合、连续行分区、通信CTA、uniform/weighted p50及加速比。该矩阵为BF16、5次warmup + 30次采样、逐样本max-rank，并启用逐元素完全一致检查；它不覆盖自动DVFS或未经实测的HBM/NVLink降频。
- `heterogeneous-cp/shape_generalization_summary.csv`: v9.1 选取五种真实模型宽度与原v9控制shape的18点复核，覆盖CP4/CP8和本地M=2048/16384。启用weighted的19个算子点全部提升并通过exact BF16检查，17个点安全回退；该复核用于限定功耗安全域，不宣称全量MNK覆盖。

Each operator directory keeps only `baseline_summary.*`, the final fused
summary, `comparison_summary.*`, and `shape_matrix.*`; each Userbuffers
directory keeps `summary.*`. The thousands of intermediate sweep and
formal-run JSON files are intentionally omitted.

The v10 backward directories keep the fused/classic-cuBLAS summary triplet,
the compact TE Userbuffers comparison/summary files, and the compact light
TE+NCCL comparison files. Their thousands of sweep and per-run JSON inputs are
not release artifacts.

The v2 fused directory intentionally keeps only `fused_summary.json` and
`fused_summary.csv`. The superseded M256 probes, manual wide-N probes, raw
per-case JSON, and unaligned-policy A/B files are not part of the Golden
archive.

The v3 directory is deliberately separate from the v2 automatic-policy
archive. Its 36/36 TE Userbuffers p50 result combines 29 unchanged v2 rows
with seven per-shape communication-CTA winners; it must not be described as
zero-tuning default behavior.

The long-sequence SOTA comparison excludes 1M because no formal external
TE/NCCL or cuBLASLt/NCCL rerun was requested for that length. Fused internal
exact checks passed. A small number of very large external cases use
`--no-check` only for the materialized PyTorch reference, whose single-launch
indexing limit is exceeded; the timed baseline path is unchanged. The scope
and selected parameters are recorded in the aggregate JSON files.

The OProj searches are reproducible with
`benchmarks/a2a+Oproj/oproj_shape_bench.py` and
`benchmarks/a2a+Oproj/te_userbuffers_shape_bench.py`. The QKV equivalents are
in `benchmarks/QKVproj+a2a`. Reproduction writes raw search points to the
caller-selected result directory without adding them to Git.

The original Golden measurement snapshot is tagged
`oproj-a2a-golden-auto-20260825`.
