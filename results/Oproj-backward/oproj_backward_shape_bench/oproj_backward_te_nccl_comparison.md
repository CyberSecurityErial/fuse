# OPROJ backward：fused 与 TE+NCCL

比值为 `TE+NCCL p50 / fused p50`；大于 1 表示 fused 更快。

| CP | 模型 | S | 模式 | fused Eager / TE / 加速 | fused Graph / TE / 加速 |
|---:|---|---:|---|---:|---:|
| 4 | nanbeige42_3b | 1K | immediate | 0.0735 / 0.1736 / 2.362x | 0.0673 / 0.0950 / 1.411x |
| 4 | nanbeige42_3b | 1K | deferred | 0.0816 / 0.1913 / 2.345x | 0.0752 / 0.1008 / 1.339x |
| 4 | nanbeige42_3b | 16K | immediate | 0.4529 / 0.6783 / 1.498x | 0.4440 / 0.6574 / 1.481x |
| 4 | nanbeige42_3b | 16K | deferred | 0.4673 / 0.7201 / 1.541x | 0.4569 / 0.6678 / 1.462x |
| 4 | nanbeige42_3b | 128K | immediate | 3.5345 / 4.9313 / 1.395x | 3.5526 / 4.9258 / 1.387x |
| 4 | nanbeige42_3b | 128K | deferred | 3.5097 / 5.0226 / 1.431x | 3.5252 / 4.9433 / 1.402x |
| 4 | representative_medium | 1K | immediate | 0.0887 / 0.1835 / 2.068x | 0.0826 / 0.1053 / 1.275x |
| 4 | representative_medium | 1K | deferred | 0.1044 / 0.2090 / 2.002x | 0.0984 / 0.1164 / 1.183x |
| 4 | representative_medium | 16K | immediate | 0.6145 / 0.8050 / 1.310x | 0.6169 / 0.7701 / 1.248x |
| 4 | representative_medium | 16K | deferred | 0.6278 / 0.8328 / 1.326x | 0.6239 / 0.7788 / 1.248x |
| 4 | representative_medium | 128K | immediate | 4.8404 / 6.0651 / 1.253x | 4.9610 / 6.0344 / 1.216x |
| 4 | representative_medium | 128K | deferred | 4.8151 / 6.0733 / 1.261x | 5.0232 / 6.0562 / 1.206x |
| 8 | nanbeige42_3b | 1K | immediate | 0.0667 / 0.1738 / 2.606x | 0.0613 / 0.0904 / 1.474x |
| 8 | nanbeige42_3b | 1K | deferred | 0.0756 / 0.1918 / 2.537x | 0.0686 / 0.0910 / 1.326x |
| 8 | nanbeige42_3b | 16K | immediate | 0.3341 / 0.4643 / 1.390x | 0.3233 / 0.4370 / 1.351x |
| 8 | nanbeige42_3b | 16K | deferred | 0.3451 / 0.5059 / 1.466x | 0.3374 / 0.4509 / 1.336x |
| 8 | nanbeige42_3b | 128K | immediate | 2.0835 / 2.9401 / 1.411x | 2.0538 / 2.9222 / 1.423x |
| 8 | nanbeige42_3b | 128K | deferred | 2.0942 / 3.0007 / 1.433x | 2.0852 / 2.9423 / 1.411x |
| 8 | representative_medium | 1K | immediate | 0.0821 / 0.1921 / 2.338x | 0.0762 / 0.0985 / 1.293x |
| 8 | representative_medium | 1K | deferred | 0.0963 / 0.2253 / 2.340x | 0.0898 / 0.1098 / 1.223x |
| 8 | representative_medium | 16K | immediate | 0.4323 / 0.5294 / 1.225x | 0.4259 / 0.5000 / 1.174x |
| 8 | representative_medium | 16K | deferred | 0.4496 / 0.5701 / 1.268x | 0.4391 / 0.5167 / 1.177x |
| 8 | representative_medium | 128K | immediate | 2.7829 / 3.4654 / 1.245x | 2.7758 / 3.4354 / 1.238x |
| 8 | representative_medium | 128K | deferred | 2.8029 / 3.5222 / 1.257x | 2.7934 / 3.4770 / 1.245x |

immediate：fused 对 TE+NCCL 的 Eager 胜点 12/12，几何平均 1.611x；Graph 胜点 12/12，几何平均 1.327x。
TE+NCCL 吞吐相对前向的中位数为 74.5%/76.9% (Eager/Graph)，相对经典 cuBLAS 纯 B+W 为 60.7%/65.4%。

deferred：fused 对 TE+NCCL 的 Eager 胜点 12/12，几何平均 1.627x；Graph 胜点 12/12，几何平均 1.293x。
TE+NCCL 吞吐相对前向的中位数为 70.9%/75.3% (Eager/Graph)，相对经典 cuBLAS 纯 B+W 为 58.7%/66.6%。
