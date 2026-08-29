# v9.0/v9.1 locked-frequency heterogeneous CP results

`locked_frequency_summary.csv` 是 v9.0 的22点精简归档，`metadata.json`保存统一采样口径、正确性范围、汇总统计和未裁剪raw文件的SHA-256。完整实验使用BF16、5次warmup + 30次采样，每个样本先取所有rank的最大CUDA event延迟，再报告p50；运行时启用了uniform与weighted输出的逐元素完全一致检查。

测试卡中物理GPU 1/3/6锁定1500 MHz，其余参与卡的参考上限为1980 MHz；所有HBM均为3201 MHz，slow SM ratio为`0.7575757576`。CSV中的`physical_devices`给出每个setting的可见设备顺序，`logical_slow_ranks`给出其中的慢rank。

`qkv_impl`与`oproj_impl`为`weighted`时，`*_rows`和`*_comm`保存每个rank最终采用的连续行数和通信CTA；`uniform_fallback`表示规划器没有启动新kernel，weighted列直接复用旧均匀路径。因此回退点的加速比严格为`1.0`。

完整逐rank物理模型诊断、p95和sample数组不提交到仓库，可使用[`weighted_sweep.py`](../../benchmarks/heterogeneous_cp/weighted_sweep.py)重新生成。

`shape_generalization_summary.csv` 是 v9.1 的18点补充复核：五种真实模型宽度加原v9控制shape，覆盖CP4/CP8和本地M=2048/16384。它使用同样的5+30和max-rank p50；19个实际启用weighted的算子点均通过exact BF16检查，另外17个点直接复用uniform。复核发现，宽到足以让参考卡也撞700 W功耗墙时，1500/1980不再是有效算力比；v9.1 因此加入保守功耗回退。最终没有退化点。
