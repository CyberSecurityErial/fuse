# SM90 MXFP8-weight 逐边界 badcase 清单

来源 full_model_v1 完整快照；并非后续局部优化后的全量。

收录规则：所有长度 speedup<1.1，或长序列 speedup<冻结的算子×CP长序列GM。两个条件取并集，不重复记行；负优化另标。短序列不套用长序列平均阈值。数字未舍入时筛选。

完整配置与 B/W 分项通过原表的 operator/model/T/CP/beta/launch 对应。

## 长序列概览

| 算子 | CP | 长序列 GM | <1.0× | <1.1× | 低于 scope GM | 低于均值的 shape（边界数） |
|---|---:|---:|---:|---:|---:|---|
| QKV Forward | 4 | 1.123583 | 6/48 | 25/48 | 27/48 | Q2(4), Q3(6), Q4(1), Q5(4), Q7(6), Q8(6) |
| QKV Forward | 8 | 1.262458 | 5/48 | 12/48 | 15/48 | Q3(6), Q4(2), Q7(1), Q8(6) |
| OProj Forward | 4 | 1.209184 | 6/48 | 12/48 | 23/48 | O2(6), O3(6), O7(5), O8(6) |
| OProj Forward | 8 | 1.315452 | 4/48 | 6/48 | 20/48 | O2(4), O3(6), O7(4), O8(6) |
| QKV Backward B→W | 4 | 1.115616 | 24/96 | 27/96 | 34/96 | Q2(9), Q3(12), Q7(1), Q8(12) |
| QKV Backward B→W | 8 | 1.084518 | 36/96 | 41/96 | 40/96 | Q1(4), Q3(12), Q4(12), Q8(12) |
| OProj Backward B→W | 4 | 1.121011 | 19/96 | 48/96 | 48/96 | O2(12), O3(12), O7(12), O8(12) |
| OProj Backward B→W | 8 | 1.146299 | 13/96 | 30/96 | 48/96 | O2(12), O3(12), O7(12), O8(12) |

## 逐边界清单

| 算子 | CP | Shape | 模型 | T | β | 启动 | 外部后端 | 外部 ms | Fuse ms | 加速 | <1 | <1.1 | 低于长 scope GM |
|---|---:|---|---|---:|---:|---|---|---:|---:|---:|---|---|---|
| QKV Forward | 4 | Q1 | artificial_small | 1K | — | graph | cublaslt_nccl | 0.068864 | 0.086624 | 0.794976 | 是 | 是 |  |
| QKV Forward | 4 | Q2 | artificial_medium | 1K | — | eager | cublaslt_nccl | 0.122624 | 0.126192 | 0.971726 | 是 | 是 |  |
| QKV Forward | 4 | Q2 | artificial_medium | 1K | — | graph | teub | 0.093136 | 0.121376 | 0.767335 | 是 | 是 |  |
| QKV Forward | 4 | Q3 | artificial_large | 1K | — | eager | cublaslt_nccl | 0.240304 | 0.422784 | 0.568385 | 是 | 是 |  |
| QKV Forward | 4 | Q3 | artificial_large | 1K | — | graph | teub | 0.212640 | 0.417904 | 0.508825 | 是 | 是 |  |
| QKV Forward | 4 | Q4 | production_qwen_dense | 1K | — | graph | teub | 0.046112 | 0.058288 | 0.791106 | 是 | 是 |  |
| QKV Forward | 4 | Q5 | nanbeige42_3b | 1K | — | eager | cublaslt_nccl | 0.128128 | 0.129552 | 0.989008 | 是 | 是 |  |
| QKV Forward | 4 | Q5 | nanbeige42_3b | 1K | — | graph | cublaslt_nccl | 0.086112 | 0.123056 | 0.699779 | 是 | 是 |  |
| QKV Forward | 4 | Q6 | llama3_8b | 1K | — | graph | teub | 0.082256 | 0.116736 | 0.704633 | 是 | 是 |  |
| QKV Forward | 4 | Q7 | qwen25_14b_32b | 1K | — | eager | cublaslt_nccl | 0.124672 | 0.153648 | 0.811413 | 是 | 是 |  |
| QKV Forward | 4 | Q7 | qwen25_14b_32b | 1K | — | graph | cublaslt_nccl | 0.103440 | 0.153920 | 0.672037 | 是 | 是 |  |
| QKV Forward | 4 | Q8 | llama31_405b | 1K | — | eager | cublaslt_nccl | 0.504576 | 0.988080 | 0.510663 | 是 | 是 |  |
| QKV Forward | 4 | Q8 | llama31_405b | 1K | — | graph | cublaslt_nccl | 0.497136 | 0.986416 | 0.503982 | 是 | 是 |  |
| QKV Forward | 4 | Q1 | artificial_small | 4K | — | eager | cublaslt_nccl | 0.144016 | 0.136432 | 1.055588 |  | 是 |  |
| QKV Forward | 4 | Q1 | artificial_small | 4K | — | graph | cublaslt_nccl | 0.117216 | 0.133952 | 0.875060 | 是 | 是 |  |
| QKV Forward | 4 | Q2 | artificial_medium | 4K | — | eager | cublaslt_nccl | 0.187744 | 0.184160 | 1.019461 |  | 是 |  |
| QKV Forward | 4 | Q2 | artificial_medium | 4K | — | graph | cublaslt_nccl | 0.165696 | 0.180496 | 0.918004 | 是 | 是 |  |
| QKV Forward | 4 | Q3 | artificial_large | 4K | — | eager | teub | 0.462928 | 0.642144 | 0.720910 | 是 | 是 |  |
| QKV Forward | 4 | Q3 | artificial_large | 4K | — | graph | teub | 0.460896 | 0.639680 | 0.720510 | 是 | 是 |  |
| QKV Forward | 4 | Q4 | production_qwen_dense | 4K | — | graph | cublaslt_nccl | 0.095648 | 0.087312 | 1.095474 |  | 是 |  |
| QKV Forward | 4 | Q5 | nanbeige42_3b | 4K | — | eager | cublaslt_nccl | 0.192880 | 0.195424 | 0.986982 | 是 | 是 |  |
| QKV Forward | 4 | Q5 | nanbeige42_3b | 4K | — | graph | cublaslt_nccl | 0.176960 | 0.194432 | 0.910138 | 是 | 是 |  |
| QKV Forward | 4 | Q6 | llama3_8b | 4K | — | eager | cublaslt_nccl | 0.191120 | 0.184176 | 1.037703 |  | 是 |  |
| QKV Forward | 4 | Q6 | llama3_8b | 4K | — | graph | teub | 0.145504 | 0.183360 | 0.793543 | 是 | 是 |  |
| QKV Forward | 4 | Q7 | qwen25_14b_32b | 4K | — | eager | teub | 0.216000 | 0.243696 | 0.886350 | 是 | 是 |  |
| QKV Forward | 4 | Q7 | qwen25_14b_32b | 4K | — | graph | teub | 0.187264 | 0.238480 | 0.785240 | 是 | 是 |  |
| QKV Forward | 4 | Q8 | llama31_405b | 4K | — | eager | teub | 1.126512 | 1.624224 | 0.693569 | 是 | 是 |  |
| QKV Forward | 4 | Q8 | llama31_405b | 4K | — | graph | cublaslt_nccl | 1.189008 | 1.632528 | 0.728323 | 是 | 是 |  |
| QKV Forward | 4 | Q1 | artificial_small | 16K | — | graph | teub | 0.307488 | 0.283168 | 1.085885 |  | 是 |  |
| QKV Forward | 4 | Q2 | artificial_medium | 16K | — | eager | teub | 0.457184 | 0.433312 | 1.055092 |  | 是 |  |
| QKV Forward | 4 | Q2 | artificial_medium | 16K | — | graph | teub | 0.436192 | 0.426032 | 1.023848 |  | 是 |  |
| QKV Forward | 4 | Q3 | artificial_large | 16K | — | eager | teub | 1.591408 | 1.640464 | 0.970096 | 是 | 是 |  |
| QKV Forward | 4 | Q3 | artificial_large | 16K | — | graph | teub | 1.636720 | 1.604448 | 1.020114 |  | 是 |  |
| QKV Forward | 4 | Q6 | llama3_8b | 16K | — | graph | teub | 0.430208 | 0.397888 | 1.081229 |  | 是 |  |
| QKV Forward | 4 | Q7 | qwen25_14b_32b | 16K | — | eager | teub | 0.601616 | 0.577648 | 1.041492 |  | 是 |  |
| QKV Forward | 4 | Q7 | qwen25_14b_32b | 16K | — | graph | teub | 0.579312 | 0.572816 | 1.011340 |  | 是 |  |
| QKV Forward | 4 | Q8 | llama31_405b | 16K | — | eager | teub | 4.024736 | 4.439120 | 0.906652 | 是 | 是 |  |
| QKV Forward | 4 | Q8 | llama31_405b | 16K | — | graph | cublaslt_nccl | 4.062464 | 4.446848 | 0.913560 | 是 | 是 |  |
| QKV Forward | 4 | Q3 | artificial_large | 128K | — | eager | teub | 11.920400 | 11.773984 | 1.012436 |  | 是 | 是 |
| QKV Forward | 4 | Q3 | artificial_large | 128K | — | graph | teub | 12.242224 | 11.885168 | 1.030042 |  | 是 | 是 |
| QKV Forward | 4 | Q7 | qwen25_14b_32b | 128K | — | eager | teub | 4.407584 | 4.011088 | 1.098850 |  | 是 | 是 |
| QKV Forward | 4 | Q7 | qwen25_14b_32b | 128K | — | graph | teub | 4.377376 | 4.057344 | 1.078877 |  | 是 | 是 |
| QKV Forward | 4 | Q8 | llama31_405b | 128K | — | eager | teub | 32.054192 | 32.301008 | 0.992359 | 是 | 是 | 是 |
| QKV Forward | 4 | Q8 | llama31_405b | 128K | — | graph | teub | 31.906976 | 33.157536 | 0.962284 | 是 | 是 | 是 |
| QKV Forward | 4 | Q2 | artificial_medium | 256K | — | eager | teub | 6.369904 | 5.756464 | 1.106565 |  |  | 是 |
| QKV Forward | 4 | Q2 | artificial_medium | 256K | — | graph | teub | 6.406864 | 5.768448 | 1.110674 |  |  | 是 |
| QKV Forward | 4 | Q3 | artificial_large | 256K | — | eager | teub | 24.201600 | 23.658944 | 1.022937 |  | 是 | 是 |
| QKV Forward | 4 | Q3 | artificial_large | 256K | — | graph | teub | 24.227839 | 24.157023 | 1.002931 |  | 是 | 是 |
| QKV Forward | 4 | Q4 | production_qwen_dense | 256K | — | eager | teub | 2.631008 | 2.408928 | 1.092190 |  | 是 | 是 |
| QKV Forward | 4 | Q5 | nanbeige42_3b | 256K | — | eager | teub | 6.867872 | 6.529424 | 1.051834 |  | 是 | 是 |
| QKV Forward | 4 | Q5 | nanbeige42_3b | 256K | — | graph | teub | 6.926048 | 6.539584 | 1.059096 |  | 是 | 是 |
| QKV Forward | 4 | Q7 | qwen25_14b_32b | 256K | — | eager | teub | 8.967968 | 8.454128 | 1.060780 |  | 是 | 是 |
| QKV Forward | 4 | Q7 | qwen25_14b_32b | 256K | — | graph | teub | 9.257232 | 8.846736 | 1.046401 |  | 是 | 是 |
| QKV Forward | 4 | Q8 | llama31_405b | 256K | — | eager | teub | 63.173313 | 66.559761 | 0.949122 | 是 | 是 | 是 |
| QKV Forward | 4 | Q8 | llama31_405b | 256K | — | graph | teub | 63.896626 | 66.499344 | 0.960861 | 是 | 是 | 是 |
| QKV Forward | 4 | Q2 | artificial_medium | 512K | — | eager | teub | 12.747152 | 11.939840 | 1.067615 |  | 是 | 是 |
| QKV Forward | 4 | Q2 | artificial_medium | 512K | — | graph | teub | 12.847600 | 12.018352 | 1.068998 |  | 是 | 是 |
| QKV Forward | 4 | Q3 | artificial_large | 512K | — | eager | teub | 49.367985 | 49.003616 | 1.007436 |  | 是 | 是 |
| QKV Forward | 4 | Q3 | artificial_large | 512K | — | graph | teub | 50.030529 | 49.461567 | 1.011503 |  | 是 | 是 |
| QKV Forward | 4 | Q5 | nanbeige42_3b | 512K | — | eager | teub | 13.934816 | 13.064464 | 1.066620 |  | 是 | 是 |
| QKV Forward | 4 | Q5 | nanbeige42_3b | 512K | — | graph | teub | 14.100688 | 13.130880 | 1.073857 |  | 是 | 是 |
| QKV Forward | 4 | Q7 | qwen25_14b_32b | 512K | — | eager | teub | 18.299727 | 17.265360 | 1.059910 |  | 是 | 是 |
| QKV Forward | 4 | Q7 | qwen25_14b_32b | 512K | — | graph | teub | 18.593472 | 17.595488 | 1.056718 |  | 是 | 是 |
| QKV Forward | 4 | Q8 | llama31_405b | 512K | — | eager | teub | 127.897713 | 134.137840 | 0.953480 | 是 | 是 | 是 |
| QKV Forward | 4 | Q8 | llama31_405b | 512K | — | graph | teub | 126.673328 | 134.348976 | 0.942868 | 是 | 是 | 是 |
| QKV Forward | 8 | Q1 | artificial_small | 1K | — | graph | cublaslt_nccl | 0.078528 | 0.088608 | 0.886241 | 是 | 是 |  |
| QKV Forward | 8 | Q2 | artificial_medium | 1K | — | graph | teub | 0.089248 | 0.121568 | 0.734141 | 是 | 是 |  |
| QKV Forward | 8 | Q3 | artificial_large | 1K | — | eager | cublaslt_nccl | 0.231136 | 0.462224 | 0.500052 | 是 | 是 |  |
| QKV Forward | 8 | Q3 | artificial_large | 1K | — | graph | teub | 0.208608 | 0.459104 | 0.454381 | 是 | 是 |  |
| QKV Forward | 8 | Q4 | production_qwen_dense | 1K | — | graph | teub | 0.055696 | 0.065520 | 0.850061 | 是 | 是 |  |
| QKV Forward | 8 | Q5 | nanbeige42_3b | 1K | — | eager | cublaslt_nccl | 0.138896 | 0.146720 | 0.946674 | 是 | 是 |  |
| QKV Forward | 8 | Q5 | nanbeige42_3b | 1K | — | graph | teub | 0.096000 | 0.142816 | 0.672194 | 是 | 是 |  |
| QKV Forward | 8 | Q6 | llama3_8b | 1K | — | eager | cublaslt_nccl | 0.137328 | 0.141088 | 0.973350 | 是 | 是 |  |
| QKV Forward | 8 | Q6 | llama3_8b | 1K | — | graph | teub | 0.085280 | 0.141568 | 0.602396 | 是 | 是 |  |
| QKV Forward | 8 | Q7 | qwen25_14b_32b | 1K | — | eager | cublaslt_nccl | 0.134912 | 0.169536 | 0.795772 | 是 | 是 |  |
| QKV Forward | 8 | Q7 | qwen25_14b_32b | 1K | — | graph | teub | 0.105488 | 0.166032 | 0.635347 | 是 | 是 |  |
| QKV Forward | 8 | Q8 | llama31_405b | 1K | — | eager | cublaslt_nccl | 0.461472 | 1.071600 | 0.430638 | 是 | 是 |  |
| QKV Forward | 8 | Q8 | llama31_405b | 1K | — | graph | cublaslt_nccl | 0.454944 | 1.069936 | 0.425207 | 是 | 是 |  |
| QKV Forward | 8 | Q1 | artificial_small | 4K | — | graph | teub | 0.108224 | 0.118672 | 0.911959 | 是 | 是 |  |
| QKV Forward | 8 | Q2 | artificial_medium | 4K | — | eager | cublaslt_nccl | 0.161296 | 0.167728 | 0.961652 | 是 | 是 |  |
| QKV Forward | 8 | Q2 | artificial_medium | 4K | — | graph | teub | 0.138640 | 0.165136 | 0.839550 | 是 | 是 |  |
| QKV Forward | 8 | Q3 | artificial_large | 4K | — | eager | cublaslt_nccl | 0.370272 | 0.604928 | 0.612093 | 是 | 是 |  |
| QKV Forward | 8 | Q3 | artificial_large | 4K | — | graph | cublaslt_nccl | 0.349280 | 0.592464 | 0.589538 | 是 | 是 |  |
| QKV Forward | 8 | Q4 | production_qwen_dense | 4K | — | graph | teub | 0.088096 | 0.083456 | 1.055598 |  | 是 |  |
| QKV Forward | 8 | Q5 | nanbeige42_3b | 4K | — | eager | cublaslt_nccl | 0.175104 | 0.187040 | 0.936185 | 是 | 是 |  |
| QKV Forward | 8 | Q5 | nanbeige42_3b | 4K | — | graph | teub | 0.143424 | 0.188272 | 0.761791 | 是 | 是 |  |
| QKV Forward | 8 | Q6 | llama3_8b | 4K | — | eager | cublaslt_nccl | 0.182704 | 0.185904 | 0.982787 | 是 | 是 |  |
| QKV Forward | 8 | Q6 | llama3_8b | 4K | — | graph | cublaslt_nccl | 0.142352 | 0.181568 | 0.784015 | 是 | 是 |  |
| QKV Forward | 8 | Q7 | qwen25_14b_32b | 4K | — | eager | cublaslt_nccl | 0.188800 | 0.228704 | 0.825521 | 是 | 是 |  |
| QKV Forward | 8 | Q7 | qwen25_14b_32b | 4K | — | graph | teub | 0.171920 | 0.222224 | 0.773634 | 是 | 是 |  |
| QKV Forward | 8 | Q8 | llama31_405b | 4K | — | eager | cublaslt_nccl | 0.796224 | 1.436144 | 0.554418 | 是 | 是 |  |
| QKV Forward | 8 | Q8 | llama31_405b | 4K | — | graph | cublaslt_nccl | 0.772000 | 1.427040 | 0.540980 | 是 | 是 |  |
| QKV Forward | 8 | Q1 | artificial_small | 16K | — | graph | teub | 0.235536 | 0.226992 | 1.037640 |  | 是 |  |
| QKV Forward | 8 | Q2 | artificial_medium | 16K | — | eager | cublaslt_nccl | 0.344016 | 0.318800 | 1.079097 |  | 是 |  |
| QKV Forward | 8 | Q2 | artificial_medium | 16K | — | graph | teub | 0.321504 | 0.315824 | 1.017985 |  | 是 |  |
| QKV Forward | 8 | Q3 | artificial_large | 16K | — | eager | cublaslt_nccl | 0.952304 | 1.179200 | 0.807585 | 是 | 是 |  |
| QKV Forward | 8 | Q3 | artificial_large | 16K | — | graph | teub | 0.914432 | 1.165600 | 0.784516 | 是 | 是 |  |
| QKV Forward | 8 | Q5 | nanbeige42_3b | 16K | — | graph | teub | 0.334912 | 0.333984 | 1.002779 |  | 是 |  |
| QKV Forward | 8 | Q6 | llama3_8b | 16K | — | eager | teub | 0.353104 | 0.337488 | 1.046271 |  | 是 |  |
| QKV Forward | 8 | Q6 | llama3_8b | 16K | — | graph | teub | 0.318336 | 0.332560 | 0.957229 | 是 | 是 |  |
| QKV Forward | 8 | Q7 | qwen25_14b_32b | 16K | — | eager | cublaslt_nccl | 0.435024 | 0.428736 | 1.014666 |  | 是 |  |
| QKV Forward | 8 | Q7 | qwen25_14b_32b | 16K | — | graph | cublaslt_nccl | 0.404288 | 0.423520 | 0.954590 | 是 | 是 |  |
| QKV Forward | 8 | Q8 | llama31_405b | 16K | — | eager | teub | 2.270608 | 2.782544 | 0.816019 | 是 | 是 |  |
| QKV Forward | 8 | Q8 | llama31_405b | 16K | — | graph | teub | 2.232128 | 2.754320 | 0.810410 | 是 | 是 |  |
| QKV Forward | 8 | Q3 | artificial_large | 128K | — | eager | cublaslt_nccl | 6.339120 | 5.981872 | 1.059722 |  | 是 | 是 |
| QKV Forward | 8 | Q3 | artificial_large | 128K | — | graph | cublaslt_nccl | 6.346880 | 6.011568 | 1.055778 |  | 是 | 是 |
| QKV Forward | 8 | Q8 | llama31_405b | 128K | — | eager | teub | 16.062560 | 16.182160 | 0.992609 | 是 | 是 | 是 |
| QKV Forward | 8 | Q8 | llama31_405b | 128K | — | graph | teub | 16.165135 | 16.901984 | 0.956405 | 是 | 是 | 是 |
| QKV Forward | 8 | Q3 | artificial_large | 256K | — | eager | cublaslt_nccl | 12.572096 | 11.751728 | 1.069808 |  | 是 | 是 |
| QKV Forward | 8 | Q3 | artificial_large | 256K | — | graph | cublaslt_nccl | 12.543920 | 11.928128 | 1.051625 |  | 是 | 是 |
| QKV Forward | 8 | Q8 | llama31_405b | 256K | — | eager | teub | 32.330687 | 32.290768 | 1.001236 |  | 是 | 是 |
| QKV Forward | 8 | Q8 | llama31_405b | 256K | — | graph | cublaslt_nccl | 32.146160 | 33.340849 | 0.964167 | 是 | 是 | 是 |
| QKV Forward | 8 | Q3 | artificial_large | 512K | — | eager | cublaslt_nccl | 25.529536 | 23.859408 | 1.069999 |  | 是 | 是 |
| QKV Forward | 8 | Q3 | artificial_large | 512K | — | graph | cublaslt_nccl | 25.268080 | 24.366784 | 1.036989 |  | 是 | 是 |
| QKV Forward | 8 | Q4 | production_qwen_dense | 512K | — | eager | teub | 3.277056 | 2.665472 | 1.229447 |  |  | 是 |
| QKV Forward | 8 | Q4 | production_qwen_dense | 512K | — | graph | teub | 3.238752 | 2.671040 | 1.212543 |  |  | 是 |
| QKV Forward | 8 | Q7 | qwen25_14b_32b | 512K | — | graph | teub | 9.906048 | 7.859376 | 1.260412 |  |  | 是 |
| QKV Forward | 8 | Q8 | llama31_405b | 512K | — | eager | cublaslt_nccl | 64.857086 | 65.941059 | 0.983561 | 是 | 是 | 是 |
| QKV Forward | 8 | Q8 | llama31_405b | 512K | — | graph | cublaslt_nccl | 64.792736 | 66.481457 | 0.974599 | 是 | 是 | 是 |
| OProj Forward | 4 | O1 | representative_small | 1K | — | graph | teub | 0.060656 | 0.083456 | 0.726802 | 是 | 是 |  |
| OProj Forward | 4 | O2 | representative_medium | 1K | — | graph | teub | 0.077104 | 0.109664 | 0.703093 | 是 | 是 |  |
| OProj Forward | 4 | O3 | representative_large | 1K | — | eager | cublaslt_nccl | 0.274112 | 0.422576 | 0.648669 | 是 | 是 |  |
| OProj Forward | 4 | O3 | representative_large | 1K | — | graph | teub | 0.227152 | 0.418656 | 0.542574 | 是 | 是 |  |
| OProj Forward | 4 | O5 | nanbeige42_3b | 1K | — | graph | teub | 0.069328 | 0.093392 | 0.742333 | 是 | 是 |  |
| OProj Forward | 4 | O6 | llama3_8b | 1K | — | graph | teub | 0.061344 | 0.082816 | 0.740726 | 是 | 是 |  |
| OProj Forward | 4 | O7 | qwen25_14b_32b | 1K | — | graph | teub | 0.082000 | 0.109008 | 0.752238 | 是 | 是 |  |
| OProj Forward | 4 | O8 | llama31_405b | 1K | — | eager | cublaslt_nccl | 0.479824 | 0.948352 | 0.505956 | 是 | 是 |  |
| OProj Forward | 4 | O8 | llama31_405b | 1K | — | graph | teub | 0.425328 | 0.945440 | 0.449873 | 是 | 是 |  |
| OProj Forward | 4 | O1 | representative_small | 4K | — | graph | teub | 0.103712 | 0.134608 | 0.770474 | 是 | 是 |  |
| OProj Forward | 4 | O2 | representative_medium | 4K | — | eager | cublaslt_nccl | 0.198512 | 0.181152 | 1.095831 |  | 是 |  |
| OProj Forward | 4 | O2 | representative_medium | 4K | — | graph | teub | 0.153760 | 0.178080 | 0.863432 | 是 | 是 |  |
| OProj Forward | 4 | O3 | representative_large | 4K | — | eager | teub | 0.549760 | 0.666096 | 0.825346 | 是 | 是 |  |
| OProj Forward | 4 | O3 | representative_large | 4K | — | graph | teub | 0.481216 | 0.663568 | 0.725195 | 是 | 是 |  |
| OProj Forward | 4 | O4 | production_qwen_dense | 4K | — | graph | teub | 0.054816 | 0.056800 | 0.965070 | 是 | 是 |  |
| OProj Forward | 4 | O5 | nanbeige42_3b | 4K | — | eager | cublaslt_nccl | 0.171088 | 0.179488 | 0.953200 | 是 | 是 |  |
| OProj Forward | 4 | O5 | nanbeige42_3b | 4K | — | graph | teub | 0.121664 | 0.175952 | 0.691461 | 是 | 是 |  |
| OProj Forward | 4 | O6 | llama3_8b | 4K | — | graph | teub | 0.103232 | 0.133504 | 0.773250 | 是 | 是 |  |
| OProj Forward | 4 | O7 | qwen25_14b_32b | 4K | — | eager | cublaslt_nccl | 0.187072 | 0.179712 | 1.040954 |  | 是 |  |
| OProj Forward | 4 | O7 | qwen25_14b_32b | 4K | — | graph | teub | 0.148672 | 0.177248 | 0.838780 | 是 | 是 |  |
| OProj Forward | 4 | O8 | llama31_405b | 4K | — | eager | teub | 1.028048 | 1.409728 | 0.729253 | 是 | 是 |  |
| OProj Forward | 4 | O8 | llama31_405b | 4K | — | graph | teub | 0.935328 | 1.405136 | 0.665649 | 是 | 是 |  |
| OProj Forward | 4 | O1 | representative_small | 16K | — | graph | teub | 0.293776 | 0.287664 | 1.021247 |  | 是 |  |
| OProj Forward | 4 | O2 | representative_medium | 16K | — | eager | teub | 0.469936 | 0.435296 | 1.079578 |  | 是 |  |
| OProj Forward | 4 | O2 | representative_medium | 16K | — | graph | teub | 0.431136 | 0.430752 | 1.000891 |  | 是 |  |
| OProj Forward | 4 | O3 | representative_large | 16K | — | eager | teub | 1.664944 | 1.790224 | 0.930020 | 是 | 是 |  |
| OProj Forward | 4 | O3 | representative_large | 16K | — | graph | teub | 1.694320 | 1.805104 | 0.938627 | 是 | 是 |  |
| OProj Forward | 4 | O4 | production_qwen_dense | 16K | — | graph | teub | 0.124320 | 0.115312 | 1.078119 |  | 是 |  |
| OProj Forward | 4 | O5 | nanbeige42_3b | 16K | — | graph | teub | 0.374128 | 0.366544 | 1.020691 |  | 是 |  |
| OProj Forward | 4 | O6 | llama3_8b | 16K | — | graph | teub | 0.294336 | 0.287280 | 1.024561 |  | 是 |  |
| OProj Forward | 4 | O7 | qwen25_14b_32b | 16K | — | eager | teub | 0.470512 | 0.433584 | 1.085169 |  | 是 |  |
| OProj Forward | 4 | O7 | qwen25_14b_32b | 16K | — | graph | teub | 0.419200 | 0.430784 | 0.973109 | 是 | 是 |  |
| OProj Forward | 4 | O8 | llama31_405b | 16K | — | eager | teub | 3.479328 | 3.982512 | 0.873652 | 是 | 是 |  |
| OProj Forward | 4 | O8 | llama31_405b | 16K | — | graph | teub | 3.424160 | 3.950736 | 0.866714 | 是 | 是 |  |
| OProj Forward | 4 | O2 | representative_medium | 128K | — | eager | teub | 3.064784 | 2.565024 | 1.194836 |  |  | 是 |
| OProj Forward | 4 | O2 | representative_medium | 128K | — | graph | teub | 3.121904 | 2.590096 | 1.205324 |  |  | 是 |
| OProj Forward | 4 | O3 | representative_large | 128K | — | eager | teub | 12.852288 | 12.025200 | 1.068780 |  | 是 | 是 |
| OProj Forward | 4 | O3 | representative_large | 128K | — | graph | teub | 12.883360 | 12.075552 | 1.066896 |  | 是 | 是 |
| OProj Forward | 4 | O7 | qwen25_14b_32b | 128K | — | graph | teub | 3.114032 | 2.581808 | 1.206144 |  |  | 是 |
| OProj Forward | 4 | O8 | llama31_405b | 128K | — | eager | teub | 27.990592 | 29.363312 | 0.953251 | 是 | 是 | 是 |
| OProj Forward | 4 | O8 | llama31_405b | 128K | — | graph | teub | 28.027472 | 30.389585 | 0.922272 | 是 | 是 | 是 |
| OProj Forward | 4 | O2 | representative_medium | 256K | — | eager | teub | 6.105888 | 5.102768 | 1.196583 |  |  | 是 |
| OProj Forward | 4 | O2 | representative_medium | 256K | — | graph | teub | 6.182416 | 5.145856 | 1.201436 |  |  | 是 |
| OProj Forward | 4 | O3 | representative_large | 256K | — | eager | teub | 25.742288 | 24.238560 | 1.062039 |  | 是 | 是 |
| OProj Forward | 4 | O3 | representative_large | 256K | — | graph | teub | 26.396895 | 24.817921 | 1.063622 |  | 是 | 是 |
| OProj Forward | 4 | O7 | qwen25_14b_32b | 256K | — | eager | teub | 6.197312 | 5.149904 | 1.203384 |  |  | 是 |
| OProj Forward | 4 | O7 | qwen25_14b_32b | 256K | — | graph | teub | 6.205632 | 5.147232 | 1.205625 |  |  | 是 |
| OProj Forward | 4 | O8 | llama31_405b | 256K | — | eager | teub | 55.693840 | 58.004784 | 0.960159 | 是 | 是 | 是 |
| OProj Forward | 4 | O8 | llama31_405b | 256K | — | graph | teub | 55.912830 | 59.247631 | 0.943714 | 是 | 是 | 是 |
| OProj Forward | 4 | O2 | representative_medium | 512K | — | eager | teub | 12.049040 | 10.266000 | 1.173684 |  |  | 是 |
| OProj Forward | 4 | O2 | representative_medium | 512K | — | graph | teub | 12.171552 | 10.538128 | 1.155001 |  |  | 是 |
| OProj Forward | 4 | O3 | representative_large | 512K | — | eager | teub | 52.497313 | 49.363503 | 1.063484 |  | 是 | 是 |
| OProj Forward | 4 | O3 | representative_large | 512K | — | graph | teub | 53.155231 | 49.751215 | 1.068421 |  | 是 | 是 |
| OProj Forward | 4 | O7 | qwen25_14b_32b | 512K | — | eager | teub | 12.079776 | 10.323136 | 1.170165 |  |  | 是 |
| OProj Forward | 4 | O7 | qwen25_14b_32b | 512K | — | graph | teub | 12.234384 | 10.434448 | 1.172499 |  |  | 是 |
| OProj Forward | 4 | O8 | llama31_405b | 512K | — | eager | teub | 110.171791 | 116.121555 | 0.948763 | 是 | 是 | 是 |
| OProj Forward | 4 | O8 | llama31_405b | 512K | — | graph | teub | 110.853649 | 116.035648 | 0.955341 | 是 | 是 | 是 |
| OProj Forward | 8 | O1 | representative_small | 1K | — | graph | cublaslt_nccl | 0.075648 | 0.083552 | 0.905400 | 是 | 是 |  |
| OProj Forward | 8 | O2 | representative_medium | 1K | — | graph | teub | 0.087536 | 0.116208 | 0.753270 | 是 | 是 |  |
| OProj Forward | 8 | O3 | representative_large | 1K | — | eager | cublaslt_nccl | 0.270304 | 0.448208 | 0.603077 | 是 | 是 |  |
| OProj Forward | 8 | O3 | representative_large | 1K | — | graph | teub | 0.217232 | 0.445472 | 0.487645 | 是 | 是 |  |
| OProj Forward | 8 | O5 | nanbeige42_3b | 1K | — | graph | teub | 0.082944 | 0.094992 | 0.873168 | 是 | 是 |  |
| OProj Forward | 8 | O6 | llama3_8b | 1K | — | graph | teub | 0.069920 | 0.083264 | 0.839739 | 是 | 是 |  |
| OProj Forward | 8 | O7 | qwen25_14b_32b | 1K | — | graph | cublaslt_nccl | 0.088640 | 0.117056 | 0.757244 | 是 | 是 |  |
| OProj Forward | 8 | O8 | llama31_405b | 1K | — | eager | cublaslt_nccl | 0.452560 | 1.055616 | 0.428717 | 是 | 是 |  |
| OProj Forward | 8 | O8 | llama31_405b | 1K | — | graph | teub | 0.391120 | 1.051536 | 0.371951 | 是 | 是 |  |
| OProj Forward | 8 | O1 | representative_small | 4K | — | graph | teub | 0.102784 | 0.113328 | 0.906960 | 是 | 是 |  |
| OProj Forward | 8 | O2 | representative_medium | 4K | — | graph | teub | 0.130192 | 0.157024 | 0.829122 | 是 | 是 |  |
| OProj Forward | 8 | O3 | representative_large | 4K | — | eager | cublaslt_nccl | 0.416016 | 0.604944 | 0.687693 | 是 | 是 |  |
| OProj Forward | 8 | O3 | representative_large | 4K | — | graph | teub | 0.342560 | 0.595248 | 0.575491 | 是 | 是 |  |
| OProj Forward | 8 | O5 | nanbeige42_3b | 4K | — | graph | teub | 0.108256 | 0.136048 | 0.795719 | 是 | 是 |  |
| OProj Forward | 8 | O6 | llama3_8b | 4K | — | graph | teub | 0.103568 | 0.113024 | 0.916336 | 是 | 是 |  |
| OProj Forward | 8 | O7 | qwen25_14b_32b | 4K | — | graph | teub | 0.126720 | 0.157856 | 0.802757 | 是 | 是 |  |
| OProj Forward | 8 | O8 | llama31_405b | 4K | — | eager | cublaslt_nccl | 0.753472 | 1.206704 | 0.624405 | 是 | 是 |  |
| OProj Forward | 8 | O8 | llama31_405b | 4K | — | graph | teub | 0.689872 | 1.201360 | 0.574243 | 是 | 是 |  |
| OProj Forward | 8 | O1 | representative_small | 16K | — | graph | teub | 0.223232 | 0.221152 | 1.009405 |  | 是 |  |
| OProj Forward | 8 | O2 | representative_medium | 16K | — | eager | cublaslt_nccl | 0.342848 | 0.313440 | 1.093823 |  | 是 |  |
| OProj Forward | 8 | O2 | representative_medium | 16K | — | graph | teub | 0.292576 | 0.309616 | 0.944964 | 是 | 是 |  |
| OProj Forward | 8 | O3 | representative_large | 16K | — | eager | teub | 1.070752 | 1.141360 | 0.938137 | 是 | 是 |  |
| OProj Forward | 8 | O3 | representative_large | 16K | — | graph | teub | 1.046144 | 1.132080 | 0.924090 | 是 | 是 |  |
| OProj Forward | 8 | O4 | production_qwen_dense | 16K | — | eager | cublaslt_nccl | 0.155952 | 0.143200 | 1.089050 |  | 是 |  |
| OProj Forward | 8 | O4 | production_qwen_dense | 16K | — | graph | teub | 0.105696 | 0.140752 | 0.750938 | 是 | 是 |  |
| OProj Forward | 8 | O5 | nanbeige42_3b | 16K | — | graph | teub | 0.271728 | 0.256272 | 1.060311 |  | 是 |  |
| OProj Forward | 8 | O6 | llama3_8b | 16K | — | graph | teub | 0.219216 | 0.224768 | 0.975299 | 是 | 是 |  |
| OProj Forward | 8 | O7 | qwen25_14b_32b | 16K | — | eager | cublaslt_nccl | 0.338832 | 0.313488 | 1.080845 |  | 是 |  |
| OProj Forward | 8 | O7 | qwen25_14b_32b | 16K | — | graph | teub | 0.295200 | 0.308960 | 0.955463 | 是 | 是 |  |
| OProj Forward | 8 | O8 | llama31_405b | 16K | — | eager | teub | 2.034560 | 2.397408 | 0.848650 | 是 | 是 |  |
| OProj Forward | 8 | O8 | llama31_405b | 16K | — | graph | teub | 1.973648 | 2.399312 | 0.822589 | 是 | 是 |  |
| OProj Forward | 8 | O2 | representative_medium | 128K | — | eager | teub | 1.898704 | 1.458928 | 1.301438 |  |  | 是 |
| OProj Forward | 8 | O2 | representative_medium | 128K | — | graph | cublaslt_nccl | 1.899552 | 1.472896 | 1.289672 |  |  | 是 |
| OProj Forward | 8 | O3 | representative_large | 128K | — | eager | teub | 7.124400 | 6.284416 | 1.133661 |  |  | 是 |
| OProj Forward | 8 | O3 | representative_large | 128K | — | graph | cublaslt_nccl | 7.425120 | 6.313776 | 1.176019 |  |  | 是 |
| OProj Forward | 8 | O7 | qwen25_14b_32b | 128K | — | eager | teub | 1.900240 | 1.459840 | 1.301677 |  |  | 是 |
| OProj Forward | 8 | O7 | qwen25_14b_32b | 128K | — | graph | cublaslt_nccl | 1.901120 | 1.467104 | 1.295832 |  |  | 是 |
| OProj Forward | 8 | O8 | llama31_405b | 128K | — | eager | teub | 14.428064 | 14.157472 | 1.019113 |  | 是 | 是 |
| OProj Forward | 8 | O8 | llama31_405b | 128K | — | graph | cublaslt_nccl | 14.412032 | 14.284752 | 1.008910 |  | 是 | 是 |
| OProj Forward | 8 | O2 | representative_medium | 256K | — | eager | cublaslt_nccl | 3.682896 | 2.816256 | 1.307728 |  |  | 是 |
| OProj Forward | 8 | O2 | representative_medium | 256K | — | graph | cublaslt_nccl | 3.662048 | 2.808368 | 1.303977 |  |  | 是 |
| OProj Forward | 8 | O3 | representative_large | 256K | — | eager | teub | 14.171424 | 12.156352 | 1.165763 |  |  | 是 |
| OProj Forward | 8 | O3 | representative_large | 256K | — | graph | teub | 14.055408 | 12.117696 | 1.159908 |  |  | 是 |
| OProj Forward | 8 | O7 | qwen25_14b_32b | 256K | — | eager | cublaslt_nccl | 3.678512 | 2.815680 | 1.306438 |  |  | 是 |
| OProj Forward | 8 | O7 | qwen25_14b_32b | 256K | — | graph | cublaslt_nccl | 3.656272 | 2.811232 | 1.300594 |  |  | 是 |
| OProj Forward | 8 | O8 | llama31_405b | 256K | — | eager | teub | 28.847327 | 29.582191 | 0.975159 | 是 | 是 | 是 |
| OProj Forward | 8 | O8 | llama31_405b | 256K | — | graph | cublaslt_nccl | 29.146704 | 30.542032 | 0.954314 | 是 | 是 | 是 |
| OProj Forward | 8 | O3 | representative_large | 512K | — | eager | cublaslt_nccl | 29.256240 | 24.042880 | 1.216836 |  |  | 是 |
| OProj Forward | 8 | O3 | representative_large | 512K | — | graph | cublaslt_nccl | 29.284272 | 24.347872 | 1.202745 |  |  | 是 |
| OProj Forward | 8 | O8 | llama31_405b | 512K | — | eager | cublaslt_nccl | 58.357567 | 59.011343 | 0.988921 | 是 | 是 | 是 |
| OProj Forward | 8 | O8 | llama31_405b | 512K | — | graph | teub | 58.855824 | 59.817823 | 0.983918 | 是 | 是 | 是 |
| QKV Backward B→W | 4 | Q1 | artificial_small | 1K | 0 | graph | teub | 0.098016 | 0.125472 | 0.781178 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q1 | artificial_small | 1K | 1 | graph | teub | 0.124512 | 0.141392 | 0.880616 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q2 | artificial_medium | 1K | 0 | graph | teub | 0.124832 | 0.171568 | 0.727595 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q2 | artificial_medium | 1K | 1 | graph | teub | 0.163248 | 0.195360 | 0.835627 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q3 | artificial_large | 1K | 0 | eager | cublaslt_nccl | 0.425488 | 0.623120 | 0.682835 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q3 | artificial_large | 1K | 0 | graph | teub | 0.361840 | 0.619808 | 0.583794 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q3 | artificial_large | 1K | 1 | eager | cublaslt_nccl | 0.629216 | 0.741968 | 0.848037 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q3 | artificial_large | 1K | 1 | graph | teub | 0.564992 | 0.736496 | 0.767135 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q4 | production_qwen_dense | 1K | 0 | graph | teub | 0.074432 | 0.087024 | 0.855304 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q4 | production_qwen_dense | 1K | 1 | graph | teub | 0.086688 | 0.096672 | 0.896723 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q5 | nanbeige42_3b | 1K | 0 | graph | teub | 0.132672 | 0.177344 | 0.748105 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q5 | nanbeige42_3b | 1K | 1 | graph | teub | 0.169184 | 0.198544 | 0.852123 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q6 | llama3_8b | 1K | 0 | graph | teub | 0.124368 | 0.166528 | 0.746829 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q6 | llama3_8b | 1K | 1 | graph | teub | 0.161056 | 0.191472 | 0.841147 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q7 | qwen25_14b_32b | 1K | 0 | eager | cublaslt_nccl | 0.221136 | 0.228336 | 0.968468 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q7 | qwen25_14b_32b | 1K | 0 | graph | teub | 0.159360 | 0.226352 | 0.704036 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q7 | qwen25_14b_32b | 1K | 1 | eager | cublaslt_nccl | 0.279232 | 0.263328 | 1.060396 |  | 是 |  |
| QKV Backward B→W | 4 | Q7 | qwen25_14b_32b | 1K | 1 | graph | teub | 0.217072 | 0.259808 | 0.835509 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q8 | llama31_405b | 1K | 0 | eager | cublaslt_nccl | 0.909568 | 1.537520 | 0.591581 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q8 | llama31_405b | 1K | 0 | graph | teub | 0.856384 | 1.529168 | 0.560033 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q8 | llama31_405b | 1K | 1 | eager | cublaslt_nccl | 1.427136 | 1.843344 | 0.774210 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q8 | llama31_405b | 1K | 1 | graph | teub | 1.377600 | 1.841952 | 0.747902 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q1 | artificial_small | 4K | 0 | graph | teub | 0.172784 | 0.201984 | 0.855434 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q1 | artificial_small | 4K | 1 | graph | teub | 0.187216 | 0.215680 | 0.868027 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q2 | artificial_medium | 4K | 0 | eager | cublaslt_nccl | 0.276656 | 0.296432 | 0.933287 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q2 | artificial_medium | 4K | 0 | graph | teub | 0.237200 | 0.290592 | 0.816265 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q2 | artificial_medium | 4K | 1 | eager | cublaslt_nccl | 0.295488 | 0.316848 | 0.932586 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q2 | artificial_medium | 4K | 1 | graph | teub | 0.257888 | 0.311888 | 0.826861 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q3 | artificial_large | 4K | 0 | eager | cublaslt_nccl | 0.861136 | 1.127952 | 0.763451 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q3 | artificial_large | 4K | 0 | graph | cublaslt_nccl | 0.843696 | 1.118528 | 0.754291 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q3 | artificial_large | 4K | 1 | eager | cublaslt_nccl | 0.925920 | 1.229504 | 0.753084 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q3 | artificial_large | 4K | 1 | graph | teub | 0.889008 | 1.223792 | 0.726437 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q4 | production_qwen_dense | 4K | 0 | graph | teub | 0.124080 | 0.126576 | 0.980281 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q4 | production_qwen_dense | 4K | 1 | graph | teub | 0.132240 | 0.134512 | 0.983109 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q5 | nanbeige42_3b | 4K | 0 | eager | cublaslt_nccl | 0.290160 | 0.289408 | 1.002598 |  | 是 |  |
| QKV Backward B→W | 4 | Q5 | nanbeige42_3b | 4K | 0 | graph | teub | 0.253632 | 0.285616 | 0.888018 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q5 | nanbeige42_3b | 4K | 1 | eager | cublaslt_nccl | 0.306080 | 0.310688 | 0.985168 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q5 | nanbeige42_3b | 4K | 1 | graph | teub | 0.271152 | 0.305424 | 0.887789 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q6 | llama3_8b | 4K | 0 | eager | cublaslt_nccl | 0.286496 | 0.293264 | 0.976922 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q6 | llama3_8b | 4K | 0 | graph | teub | 0.238160 | 0.286560 | 0.831100 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q6 | llama3_8b | 4K | 1 | eager | cublaslt_nccl | 0.305536 | 0.313632 | 0.974186 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q6 | llama3_8b | 4K | 1 | graph | teub | 0.259216 | 0.308608 | 0.839952 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q7 | qwen25_14b_32b | 4K | 0 | eager | cublaslt_nccl | 0.352816 | 0.393408 | 0.896820 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q7 | qwen25_14b_32b | 4K | 0 | graph | cublaslt_nccl | 0.326096 | 0.389936 | 0.836281 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q7 | qwen25_14b_32b | 4K | 1 | eager | cublaslt_nccl | 0.379072 | 0.422944 | 0.896270 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q7 | qwen25_14b_32b | 4K | 1 | graph | teub | 0.351040 | 0.419040 | 0.837724 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q8 | llama31_405b | 4K | 0 | eager | teub | 2.198528 | 2.869296 | 0.766226 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q8 | llama31_405b | 4K | 0 | graph | teub | 2.149776 | 2.829728 | 0.759711 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q8 | llama31_405b | 4K | 1 | eager | teub | 2.339840 | 3.068416 | 0.762556 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q8 | llama31_405b | 4K | 1 | graph | teub | 2.298352 | 3.082320 | 0.745656 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q1 | artificial_small | 16K | 0 | graph | teub | 0.518992 | 0.473808 | 1.095364 |  | 是 |  |
| QKV Backward B→W | 4 | Q1 | artificial_small | 16K | 1 | graph | teub | 0.527648 | 0.485728 | 1.086303 |  | 是 |  |
| QKV Backward B→W | 4 | Q2 | artificial_medium | 16K | 0 | eager | teub | 0.812544 | 0.741856 | 1.095285 |  | 是 |  |
| QKV Backward B→W | 4 | Q2 | artificial_medium | 16K | 0 | graph | teub | 0.790960 | 0.749520 | 1.055289 |  | 是 |  |
| QKV Backward B→W | 4 | Q2 | artificial_medium | 16K | 1 | graph | teub | 0.809536 | 0.759808 | 1.065448 |  | 是 |  |
| QKV Backward B→W | 4 | Q3 | artificial_large | 16K | 0 | eager | cublaslt_nccl | 3.149920 | 3.500704 | 0.899796 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q3 | artificial_large | 16K | 0 | graph | cublaslt_nccl | 3.072496 | 3.510240 | 0.875295 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q3 | artificial_large | 16K | 1 | eager | teub | 3.094848 | 3.561408 | 0.868996 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q3 | artificial_large | 16K | 1 | graph | teub | 3.180784 | 3.681360 | 0.864024 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q7 | qwen25_14b_32b | 16K | 0 | graph | cublaslt_nccl | 1.102160 | 1.007680 | 1.093760 |  | 是 |  |
| QKV Backward B→W | 4 | Q7 | qwen25_14b_32b | 16K | 1 | graph | teub | 1.112976 | 1.028944 | 1.081668 |  | 是 |  |
| QKV Backward B→W | 4 | Q8 | llama31_405b | 16K | 0 | eager | teub | 8.050208 | 9.214624 | 0.873634 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q8 | llama31_405b | 16K | 0 | graph | cublaslt_nccl | 8.346864 | 9.268416 | 0.900571 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q8 | llama31_405b | 16K | 1 | eager | teub | 8.104336 | 9.508096 | 0.852362 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q8 | llama31_405b | 16K | 1 | graph | cublaslt_nccl | 8.562640 | 9.063888 | 0.944698 | 是 | 是 |  |
| QKV Backward B→W | 4 | Q2 | artificial_medium | 128K | 0 | eager | teub | 6.116816 | 5.563408 | 1.099473 |  | 是 | 是 |
| QKV Backward B→W | 4 | Q2 | artificial_medium | 128K | 0 | graph | cublaslt_nccl | 6.139568 | 5.832800 | 1.052594 |  | 是 | 是 |
| QKV Backward B→W | 4 | Q2 | artificial_medium | 128K | 1 | eager | cublaslt_nccl | 6.343328 | 5.742672 | 1.104595 |  |  | 是 |
| QKV Backward B→W | 4 | Q2 | artificial_medium | 128K | 1 | graph | cublaslt_nccl | 6.166000 | 5.575888 | 1.105833 |  |  | 是 |
| QKV Backward B→W | 4 | Q3 | artificial_large | 128K | 0 | eager | teub | 24.682976 | 26.038400 | 0.947945 | 是 | 是 | 是 |
| QKV Backward B→W | 4 | Q3 | artificial_large | 128K | 0 | graph | teub | 24.508320 | 26.137200 | 0.937680 | 是 | 是 | 是 |
| QKV Backward B→W | 4 | Q3 | artificial_large | 128K | 1 | eager | cublaslt_nccl | 24.772881 | 26.244385 | 0.943931 | 是 | 是 | 是 |
| QKV Backward B→W | 4 | Q3 | artificial_large | 128K | 1 | graph | teub | 24.760912 | 26.209439 | 0.944733 | 是 | 是 | 是 |
| QKV Backward B→W | 4 | Q7 | qwen25_14b_32b | 128K | 0 | eager | cublaslt_nccl | 8.687152 | 7.824736 | 1.110217 |  |  | 是 |
| QKV Backward B→W | 4 | Q8 | llama31_405b | 128K | 0 | eager | cublaslt_nccl | 64.171150 | 66.566418 | 0.964017 | 是 | 是 | 是 |
| QKV Backward B→W | 4 | Q8 | llama31_405b | 128K | 0 | graph | teub | 63.401697 | 66.797871 | 0.949157 | 是 | 是 | 是 |
| QKV Backward B→W | 4 | Q8 | llama31_405b | 128K | 1 | eager | cublaslt_nccl | 63.424496 | 66.918846 | 0.947782 | 是 | 是 | 是 |
| QKV Backward B→W | 4 | Q8 | llama31_405b | 128K | 1 | graph | teub | 63.589823 | 66.883072 | 0.950761 | 是 | 是 | 是 |
| QKV Backward B→W | 4 | Q2 | artificial_medium | 256K | 0 | graph | cublaslt_nccl | 12.502080 | 11.667184 | 1.071559 |  | 是 | 是 |
| QKV Backward B→W | 4 | Q2 | artificial_medium | 256K | 1 | eager | cublaslt_nccl | 12.702432 | 11.522560 | 1.102397 |  |  | 是 |
| QKV Backward B→W | 4 | Q2 | artificial_medium | 256K | 1 | graph | teub | 12.879696 | 11.588096 | 1.111459 |  |  | 是 |
| QKV Backward B→W | 4 | Q3 | artificial_large | 256K | 0 | eager | cublaslt_nccl | 49.220209 | 52.094208 | 0.944831 | 是 | 是 | 是 |
| QKV Backward B→W | 4 | Q3 | artificial_large | 256K | 0 | graph | cublaslt_nccl | 49.333887 | 52.110592 | 0.946715 | 是 | 是 | 是 |
| QKV Backward B→W | 4 | Q3 | artificial_large | 256K | 1 | eager | cublaslt_nccl | 49.732960 | 52.220160 | 0.952371 | 是 | 是 | 是 |
| QKV Backward B→W | 4 | Q3 | artificial_large | 256K | 1 | graph | teub | 49.514992 | 52.165344 | 0.949193 | 是 | 是 | 是 |
| QKV Backward B→W | 4 | Q8 | llama31_405b | 256K | 0 | eager | teub | 124.911983 | 132.515228 | 0.942624 | 是 | 是 | 是 |
| QKV Backward B→W | 4 | Q8 | llama31_405b | 256K | 0 | graph | teub | 125.569023 | 132.639610 | 0.946693 | 是 | 是 | 是 |
| QKV Backward B→W | 4 | Q8 | llama31_405b | 256K | 1 | eager | teub | 125.318306 | 132.424156 | 0.946340 | 是 | 是 | 是 |
| QKV Backward B→W | 4 | Q8 | llama31_405b | 256K | 1 | graph | teub | 125.395107 | 132.314331 | 0.947706 | 是 | 是 | 是 |
| QKV Backward B→W | 4 | Q2 | artificial_medium | 512K | 0 | graph | teub | 25.392688 | 23.011344 | 1.103486 |  |  | 是 |
| QKV Backward B→W | 4 | Q2 | artificial_medium | 512K | 1 | eager | cublaslt_nccl | 25.618080 | 23.051744 | 1.111329 |  |  | 是 |
| QKV Backward B→W | 4 | Q3 | artificial_large | 512K | 0 | eager | teub | 98.039791 | 103.250957 | 0.949529 | 是 | 是 | 是 |
| QKV Backward B→W | 4 | Q3 | artificial_large | 512K | 0 | graph | teub | 98.409489 | 103.266560 | 0.952966 | 是 | 是 | 是 |
| QKV Backward B→W | 4 | Q3 | artificial_large | 512K | 1 | eager | teub | 98.022350 | 103.394035 | 0.948046 | 是 | 是 | 是 |
| QKV Backward B→W | 4 | Q3 | artificial_large | 512K | 1 | graph | teub | 98.075985 | 103.363552 | 0.948845 | 是 | 是 | 是 |
| QKV Backward B→W | 4 | Q8 | llama31_405b | 512K | 0 | eager | cublaslt_nccl | 250.000671 | 262.399170 | 0.952749 | 是 | 是 | 是 |
| QKV Backward B→W | 4 | Q8 | llama31_405b | 512K | 0 | graph | cublaslt_nccl | 250.474113 | 262.661087 | 0.953602 | 是 | 是 | 是 |
| QKV Backward B→W | 4 | Q8 | llama31_405b | 512K | 1 | eager | cublaslt_nccl | 249.336205 | 262.449783 | 0.950034 | 是 | 是 | 是 |
| QKV Backward B→W | 4 | Q8 | llama31_405b | 512K | 1 | graph | cublaslt_nccl | 249.448563 | 262.569855 | 0.950027 | 是 | 是 | 是 |
| QKV Backward B→W | 8 | Q1 | artificial_small | 1K | 0 | graph | teub | 0.097056 | 0.136224 | 0.712474 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q1 | artificial_small | 1K | 1 | graph | teub | 0.124512 | 0.152208 | 0.818038 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q2 | artificial_medium | 1K | 0 | eager | cublaslt_nccl | 0.205664 | 0.192048 | 1.070899 |  | 是 |  |
| QKV Backward B→W | 8 | Q2 | artificial_medium | 1K | 0 | graph | teub | 0.137408 | 0.193344 | 0.710692 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q2 | artificial_medium | 1K | 1 | graph | teub | 0.175296 | 0.217152 | 0.807250 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q3 | artificial_large | 1K | 0 | eager | cublaslt_nccl | 0.411936 | 0.604960 | 0.680931 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q3 | artificial_large | 1K | 0 | graph | teub | 0.348128 | 0.601296 | 0.578963 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q3 | artificial_large | 1K | 1 | eager | cublaslt_nccl | 0.632416 | 0.721360 | 0.876700 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q3 | artificial_large | 1K | 1 | graph | teub | 0.565696 | 0.720080 | 0.785602 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q4 | production_qwen_dense | 1K | 0 | graph | teub | 0.077664 | 0.110512 | 0.702765 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q4 | production_qwen_dense | 1K | 1 | graph | teub | 0.095856 | 0.118000 | 0.812339 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q5 | nanbeige42_3b | 1K | 0 | eager | cublaslt_nccl | 0.197360 | 0.199152 | 0.991002 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q5 | nanbeige42_3b | 1K | 0 | graph | teub | 0.130112 | 0.197104 | 0.660119 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q5 | nanbeige42_3b | 1K | 1 | eager | cublaslt_nccl | 0.233536 | 0.227456 | 1.026730 |  | 是 |  |
| QKV Backward B→W | 8 | Q5 | nanbeige42_3b | 1K | 1 | graph | teub | 0.169088 | 0.223632 | 0.756099 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q6 | llama3_8b | 1K | 0 | graph | teub | 0.136592 | 0.176512 | 0.773840 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q6 | llama3_8b | 1K | 1 | graph | teub | 0.172992 | 0.201392 | 0.858982 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q7 | qwen25_14b_32b | 1K | 0 | eager | cublaslt_nccl | 0.220432 | 0.244208 | 0.902640 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q7 | qwen25_14b_32b | 1K | 0 | graph | teub | 0.165312 | 0.243088 | 0.680050 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q7 | qwen25_14b_32b | 1K | 1 | eager | cublaslt_nccl | 0.283648 | 0.281456 | 1.007788 |  | 是 |  |
| QKV Backward B→W | 8 | Q7 | qwen25_14b_32b | 1K | 1 | graph | cublaslt_nccl | 0.222528 | 0.278336 | 0.799494 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q8 | llama31_405b | 1K | 0 | eager | cublaslt_nccl | 0.896352 | 1.461408 | 0.613348 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q8 | llama31_405b | 1K | 0 | graph | teub | 0.789728 | 1.457280 | 0.541919 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q8 | llama31_405b | 1K | 1 | eager | cublaslt_nccl | 1.409440 | 1.781120 | 0.791322 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q8 | llama31_405b | 1K | 1 | graph | cublaslt_nccl | 1.357696 | 1.772864 | 0.765821 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q1 | artificial_small | 4K | 0 | graph | teub | 0.135360 | 0.179088 | 0.755829 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q1 | artificial_small | 4K | 1 | graph | teub | 0.158048 | 0.190736 | 0.828622 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q2 | artificial_medium | 4K | 0 | eager | cublaslt_nccl | 0.255664 | 0.257872 | 0.991438 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q2 | artificial_medium | 4K | 0 | graph | teub | 0.187424 | 0.254768 | 0.735665 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q2 | artificial_medium | 4K | 1 | eager | cublaslt_nccl | 0.282336 | 0.283840 | 0.994701 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q2 | artificial_medium | 4K | 1 | graph | teub | 0.214400 | 0.282640 | 0.758562 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q3 | artificial_large | 4K | 0 | eager | cublaslt_nccl | 0.621984 | 0.949808 | 0.654852 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q3 | artificial_large | 4K | 0 | graph | teub | 0.573248 | 0.947008 | 0.605325 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q3 | artificial_large | 4K | 1 | eager | cublaslt_nccl | 0.762784 | 1.048144 | 0.727747 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q3 | artificial_large | 4K | 1 | graph | teub | 0.713472 | 1.044848 | 0.682848 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q4 | production_qwen_dense | 4K | 0 | graph | teub | 0.101680 | 0.126928 | 0.801084 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q4 | production_qwen_dense | 4K | 1 | graph | teub | 0.122784 | 0.137840 | 0.890772 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q5 | nanbeige42_3b | 4K | 0 | eager | cublaslt_nccl | 0.248704 | 0.261600 | 0.950703 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q5 | nanbeige42_3b | 4K | 0 | graph | teub | 0.197360 | 0.258336 | 0.763966 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q5 | nanbeige42_3b | 4K | 1 | eager | cublaslt_nccl | 0.270480 | 0.287632 | 0.940368 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q5 | nanbeige42_3b | 4K | 1 | graph | teub | 0.220624 | 0.281136 | 0.784759 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q6 | llama3_8b | 4K | 0 | eager | cublaslt_nccl | 0.243856 | 0.238928 | 1.020625 |  | 是 |  |
| QKV Backward B→W | 8 | Q6 | llama3_8b | 4K | 0 | graph | teub | 0.196592 | 0.236784 | 0.830259 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q6 | llama3_8b | 4K | 1 | eager | cublaslt_nccl | 0.264352 | 0.263280 | 1.004072 |  | 是 |  |
| QKV Backward B→W | 8 | Q6 | llama3_8b | 4K | 1 | graph | teub | 0.219664 | 0.258896 | 0.848464 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q7 | qwen25_14b_32b | 4K | 0 | eager | cublaslt_nccl | 0.291680 | 0.325568 | 0.895911 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q7 | qwen25_14b_32b | 4K | 0 | graph | teub | 0.246800 | 0.315840 | 0.781408 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q7 | qwen25_14b_32b | 4K | 1 | eager | cublaslt_nccl | 0.329520 | 0.358384 | 0.919461 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q7 | qwen25_14b_32b | 4K | 1 | graph | teub | 0.285984 | 0.352432 | 0.811459 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q8 | llama31_405b | 4K | 0 | eager | cublaslt_nccl | 1.381632 | 2.316864 | 0.596337 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q8 | llama31_405b | 4K | 0 | graph | teub | 1.355136 | 2.316240 | 0.585059 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q8 | llama31_405b | 4K | 1 | eager | cublaslt_nccl | 1.755472 | 2.589824 | 0.677834 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q8 | llama31_405b | 4K | 1 | graph | teub | 1.722512 | 2.585040 | 0.666339 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q1 | artificial_small | 16K | 0 | eager | cublaslt_nccl | 0.402672 | 0.392128 | 1.026889 |  | 是 |  |
| QKV Backward B→W | 8 | Q1 | artificial_small | 16K | 0 | graph | teub | 0.340432 | 0.383680 | 0.887281 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q1 | artificial_small | 16K | 1 | eager | cublaslt_nccl | 0.411968 | 0.404720 | 1.017909 |  | 是 |  |
| QKV Backward B→W | 8 | Q1 | artificial_small | 16K | 1 | graph | teub | 0.342800 | 0.396640 | 0.864260 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q2 | artificial_medium | 16K | 0 | eager | cublaslt_nccl | 0.534768 | 0.537392 | 0.995117 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q2 | artificial_medium | 16K | 0 | graph | teub | 0.504624 | 0.532992 | 0.946776 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q2 | artificial_medium | 16K | 1 | eager | cublaslt_nccl | 0.548016 | 0.558832 | 0.980645 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q2 | artificial_medium | 16K | 1 | graph | teub | 0.511056 | 0.562512 | 0.908525 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q3 | artificial_large | 16K | 0 | eager | teub | 1.693600 | 2.062480 | 0.821147 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q3 | artificial_large | 16K | 0 | graph | teub | 1.652720 | 2.056560 | 0.803633 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q3 | artificial_large | 16K | 1 | eager | cublaslt_nccl | 1.711072 | 2.150656 | 0.795605 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q3 | artificial_large | 16K | 1 | graph | teub | 1.671472 | 2.143616 | 0.779744 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q4 | production_qwen_dense | 16K | 1 | graph | teub | 0.247040 | 0.226960 | 1.088474 |  | 是 |  |
| QKV Backward B→W | 8 | Q5 | nanbeige42_3b | 16K | 0 | eager | cublaslt_nccl | 0.563904 | 0.556592 | 1.013137 |  | 是 |  |
| QKV Backward B→W | 8 | Q5 | nanbeige42_3b | 16K | 0 | graph | teub | 0.523200 | 0.542640 | 0.964175 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q5 | nanbeige42_3b | 16K | 1 | eager | cublaslt_nccl | 0.564336 | 0.566480 | 0.996215 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q5 | nanbeige42_3b | 16K | 1 | graph | teub | 0.534960 | 0.559888 | 0.955477 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q6 | llama3_8b | 16K | 0 | eager | cublaslt_nccl | 0.541824 | 0.528784 | 1.024660 |  | 是 |  |
| QKV Backward B→W | 8 | Q6 | llama3_8b | 16K | 0 | graph | teub | 0.504720 | 0.532096 | 0.948551 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q6 | llama3_8b | 16K | 1 | eager | cublaslt_nccl | 0.548992 | 0.551568 | 0.995330 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q6 | llama3_8b | 16K | 1 | graph | teub | 0.514800 | 0.549120 | 0.937500 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q7 | qwen25_14b_32b | 16K | 0 | eager | teub | 0.701472 | 0.717712 | 0.977373 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q7 | qwen25_14b_32b | 16K | 0 | graph | teub | 0.666128 | 0.705600 | 0.944059 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q7 | qwen25_14b_32b | 16K | 1 | eager | cublaslt_nccl | 0.703184 | 0.734528 | 0.957328 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q7 | qwen25_14b_32b | 16K | 1 | graph | teub | 0.681072 | 0.729104 | 0.934122 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q8 | llama31_405b | 16K | 0 | eager | teub | 4.161920 | 5.056336 | 0.823110 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q8 | llama31_405b | 16K | 0 | graph | teub | 4.135456 | 5.052416 | 0.818511 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q8 | llama31_405b | 16K | 1 | eager | cublaslt_nccl | 4.473536 | 5.353840 | 0.835575 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q8 | llama31_405b | 16K | 1 | graph | cublaslt_nccl | 4.206240 | 5.307392 | 0.792525 | 是 | 是 |  |
| QKV Backward B→W | 8 | Q1 | artificial_small | 128K | 0 | eager | cublaslt_nccl | 2.202944 | 2.038544 | 1.080646 |  | 是 | 是 |
| QKV Backward B→W | 8 | Q1 | artificial_small | 128K | 0 | graph | cublaslt_nccl | 2.167280 | 2.042288 | 1.061202 |  | 是 | 是 |
| QKV Backward B→W | 8 | Q1 | artificial_small | 128K | 1 | eager | cublaslt_nccl | 2.203728 | 2.052944 | 1.073448 |  | 是 | 是 |
| QKV Backward B→W | 8 | Q1 | artificial_small | 128K | 1 | graph | cublaslt_nccl | 2.167808 | 2.049600 | 1.057674 |  | 是 | 是 |
| QKV Backward B→W | 8 | Q3 | artificial_large | 128K | 0 | eager | teub | 12.647232 | 12.964624 | 0.975519 | 是 | 是 | 是 |
| QKV Backward B→W | 8 | Q3 | artificial_large | 128K | 0 | graph | cublaslt_nccl | 12.495296 | 13.277296 | 0.941102 | 是 | 是 | 是 |
| QKV Backward B→W | 8 | Q3 | artificial_large | 128K | 1 | eager | teub | 12.428480 | 13.169776 | 0.943712 | 是 | 是 | 是 |
| QKV Backward B→W | 8 | Q3 | artificial_large | 128K | 1 | graph | teub | 12.711120 | 13.443680 | 0.945509 | 是 | 是 | 是 |
| QKV Backward B→W | 8 | Q4 | production_qwen_dense | 128K | 0 | eager | cublaslt_nccl | 1.445920 | 1.544528 | 0.936157 | 是 | 是 | 是 |
| QKV Backward B→W | 8 | Q4 | production_qwen_dense | 128K | 0 | graph | cublaslt_nccl | 1.412640 | 1.550320 | 0.911193 | 是 | 是 | 是 |
| QKV Backward B→W | 8 | Q4 | production_qwen_dense | 128K | 1 | eager | cublaslt_nccl | 1.440208 | 1.554144 | 0.926689 | 是 | 是 | 是 |
| QKV Backward B→W | 8 | Q4 | production_qwen_dense | 128K | 1 | graph | teub | 1.416224 | 1.546160 | 0.915962 | 是 | 是 | 是 |
| QKV Backward B→W | 8 | Q8 | llama31_405b | 128K | 0 | eager | teub | 32.820496 | 34.214975 | 0.959244 | 是 | 是 | 是 |
| QKV Backward B→W | 8 | Q8 | llama31_405b | 128K | 0 | graph | teub | 32.729248 | 33.983921 | 0.963080 | 是 | 是 | 是 |
| QKV Backward B→W | 8 | Q8 | llama31_405b | 128K | 1 | eager | cublaslt_nccl | 32.755024 | 34.318592 | 0.954440 | 是 | 是 | 是 |
| QKV Backward B→W | 8 | Q8 | llama31_405b | 128K | 1 | graph | cublaslt_nccl | 32.738958 | 34.236624 | 0.956255 | 是 | 是 | 是 |
| QKV Backward B→W | 8 | Q1 | artificial_small | 256K | 0 | graph | cublaslt_nccl | 4.252800 | 3.871584 | 1.098465 |  | 是 |  |
| QKV Backward B→W | 8 | Q3 | artificial_large | 256K | 0 | eager | teub | 25.265024 | 26.270576 | 0.961723 | 是 | 是 | 是 |
| QKV Backward B→W | 8 | Q3 | artificial_large | 256K | 0 | graph | teub | 24.737967 | 26.298704 | 0.940653 | 是 | 是 | 是 |
| QKV Backward B→W | 8 | Q3 | artificial_large | 256K | 1 | eager | teub | 25.115567 | 26.329663 | 0.953889 | 是 | 是 | 是 |
| QKV Backward B→W | 8 | Q3 | artificial_large | 256K | 1 | graph | cublaslt_nccl | 25.112624 | 26.332191 | 0.953685 | 是 | 是 | 是 |
| QKV Backward B→W | 8 | Q4 | production_qwen_dense | 256K | 0 | eager | cublaslt_nccl | 2.763504 | 2.936976 | 0.940935 | 是 | 是 | 是 |
| QKV Backward B→W | 8 | Q4 | production_qwen_dense | 256K | 0 | graph | cublaslt_nccl | 2.730048 | 2.934928 | 0.930193 | 是 | 是 | 是 |
| QKV Backward B→W | 8 | Q4 | production_qwen_dense | 256K | 1 | eager | cublaslt_nccl | 2.759264 | 2.971376 | 0.928615 | 是 | 是 | 是 |
| QKV Backward B→W | 8 | Q4 | production_qwen_dense | 256K | 1 | graph | cublaslt_nccl | 2.727024 | 2.962704 | 0.920451 | 是 | 是 | 是 |
| QKV Backward B→W | 8 | Q8 | llama31_405b | 256K | 0 | eager | cublaslt_nccl | 64.865105 | 67.222481 | 0.964932 | 是 | 是 | 是 |
| QKV Backward B→W | 8 | Q8 | llama31_405b | 256K | 0 | graph | cublaslt_nccl | 64.681362 | 67.070034 | 0.964385 | 是 | 是 | 是 |
| QKV Backward B→W | 8 | Q8 | llama31_405b | 256K | 1 | eager | cublaslt_nccl | 64.967407 | 67.527489 | 0.962088 | 是 | 是 | 是 |
| QKV Backward B→W | 8 | Q8 | llama31_405b | 256K | 1 | graph | cublaslt_nccl | 64.894814 | 67.498734 | 0.961423 | 是 | 是 | 是 |
| QKV Backward B→W | 8 | Q3 | artificial_large | 512K | 0 | eager | teub | 49.695776 | 52.651951 | 0.943854 | 是 | 是 | 是 |
| QKV Backward B→W | 8 | Q3 | artificial_large | 512K | 0 | graph | cublaslt_nccl | 49.924305 | 52.744465 | 0.946532 | 是 | 是 | 是 |
| QKV Backward B→W | 8 | Q3 | artificial_large | 512K | 1 | eager | cublaslt_nccl | 50.316015 | 52.832209 | 0.952374 | 是 | 是 | 是 |
| QKV Backward B→W | 8 | Q3 | artificial_large | 512K | 1 | graph | cublaslt_nccl | 50.529663 | 52.773281 | 0.957486 | 是 | 是 | 是 |
| QKV Backward B→W | 8 | Q4 | production_qwen_dense | 512K | 0 | eager | cublaslt_nccl | 5.387088 | 5.781968 | 0.931705 | 是 | 是 | 是 |
| QKV Backward B→W | 8 | Q4 | production_qwen_dense | 512K | 0 | graph | cublaslt_nccl | 5.358064 | 5.733248 | 0.934560 | 是 | 是 | 是 |
| QKV Backward B→W | 8 | Q4 | production_qwen_dense | 512K | 1 | eager | cublaslt_nccl | 5.378512 | 5.809360 | 0.925836 | 是 | 是 | 是 |
| QKV Backward B→W | 8 | Q4 | production_qwen_dense | 512K | 1 | graph | cublaslt_nccl | 5.377360 | 5.827376 | 0.922776 | 是 | 是 | 是 |
| QKV Backward B→W | 8 | Q8 | llama31_405b | 512K | 0 | eager | cublaslt_nccl | 127.862720 | 134.166695 | 0.953014 | 是 | 是 | 是 |
| QKV Backward B→W | 8 | Q8 | llama31_405b | 512K | 0 | graph | cublaslt_nccl | 128.021358 | 133.711967 | 0.957441 | 是 | 是 | 是 |
| QKV Backward B→W | 8 | Q8 | llama31_405b | 512K | 1 | eager | cublaslt_nccl | 127.244144 | 134.430466 | 0.946542 | 是 | 是 | 是 |
| QKV Backward B→W | 8 | Q8 | llama31_405b | 512K | 1 | graph | teub | 128.071953 | 132.940338 | 0.963379 | 是 | 是 | 是 |
| OProj Backward B→W | 4 | O1 | representative_small | 1K | 0 | graph | teub | 0.096752 | 0.120816 | 0.800821 | 是 | 是 |  |
| OProj Backward B→W | 4 | O1 | representative_small | 1K | 1 | graph | teub | 0.122448 | 0.136400 | 0.897713 | 是 | 是 |  |
| OProj Backward B→W | 4 | O2 | representative_medium | 1K | 0 | graph | teub | 0.123104 | 0.174464 | 0.705613 | 是 | 是 |  |
| OProj Backward B→W | 4 | O2 | representative_medium | 1K | 1 | eager | cublaslt_nccl | 0.215392 | 0.208064 | 1.035220 |  | 是 |  |
| OProj Backward B→W | 4 | O2 | representative_medium | 1K | 1 | graph | teub | 0.163552 | 0.197824 | 0.826755 | 是 | 是 |  |
| OProj Backward B→W | 4 | O3 | representative_large | 1K | 0 | eager | cublaslt_nccl | 0.413856 | 0.627424 | 0.659611 | 是 | 是 |  |
| OProj Backward B→W | 4 | O3 | representative_large | 1K | 0 | graph | teub | 0.385184 | 0.623344 | 0.617932 | 是 | 是 |  |
| OProj Backward B→W | 4 | O3 | representative_large | 1K | 1 | eager | cublaslt_nccl | 0.617392 | 0.742976 | 0.830972 | 是 | 是 |  |
| OProj Backward B→W | 4 | O3 | representative_large | 1K | 1 | graph | teub | 0.594112 | 0.739168 | 0.803758 | 是 | 是 |  |
| OProj Backward B→W | 4 | O4 | production_qwen_dense | 1K | 0 | graph | teub | 0.046208 | 0.054832 | 0.842720 | 是 | 是 |  |
| OProj Backward B→W | 4 | O4 | production_qwen_dense | 1K | 1 | graph | teub | 0.048800 | 0.055216 | 0.883802 | 是 | 是 |  |
| OProj Backward B→W | 4 | O5 | nanbeige42_3b | 1K | 0 | graph | teub | 0.108480 | 0.129984 | 0.834564 | 是 | 是 |  |
| OProj Backward B→W | 4 | O5 | nanbeige42_3b | 1K | 1 | graph | teub | 0.137344 | 0.150464 | 0.912803 | 是 | 是 |  |
| OProj Backward B→W | 4 | O6 | llama3_8b | 1K | 0 | graph | teub | 0.088496 | 0.121280 | 0.729683 | 是 | 是 |  |
| OProj Backward B→W | 4 | O6 | llama3_8b | 1K | 1 | graph | teub | 0.116496 | 0.136144 | 0.855682 | 是 | 是 |  |
| OProj Backward B→W | 4 | O7 | qwen25_14b_32b | 1K | 0 | eager | cublaslt_nccl | 0.167488 | 0.171248 | 0.978044 | 是 | 是 |  |
| OProj Backward B→W | 4 | O7 | qwen25_14b_32b | 1K | 0 | graph | teub | 0.122784 | 0.177248 | 0.692724 | 是 | 是 |  |
| OProj Backward B→W | 4 | O7 | qwen25_14b_32b | 1K | 1 | eager | cublaslt_nccl | 0.213696 | 0.205216 | 1.041322 |  | 是 |  |
| OProj Backward B→W | 4 | O7 | qwen25_14b_32b | 1K | 1 | graph | teub | 0.166016 | 0.199104 | 0.833816 | 是 | 是 |  |
| OProj Backward B→W | 4 | O8 | llama31_405b | 1K | 0 | eager | cublaslt_nccl | 0.795344 | 1.377376 | 0.577434 | 是 | 是 |  |
| OProj Backward B→W | 4 | O8 | llama31_405b | 1K | 0 | graph | teub | 0.760912 | 1.370576 | 0.555177 | 是 | 是 |  |
| OProj Backward B→W | 4 | O8 | llama31_405b | 1K | 1 | eager | cublaslt_nccl | 1.268032 | 1.644464 | 0.771091 | 是 | 是 |  |
| OProj Backward B→W | 4 | O8 | llama31_405b | 1K | 1 | graph | teub | 1.233600 | 1.638800 | 0.752746 | 是 | 是 |  |
| OProj Backward B→W | 4 | O1 | representative_small | 4K | 0 | eager | cublaslt_nccl | 0.200272 | 0.199552 | 1.003608 |  | 是 |  |
| OProj Backward B→W | 4 | O1 | representative_small | 4K | 0 | graph | teub | 0.167264 | 0.193184 | 0.865827 | 是 | 是 |  |
| OProj Backward B→W | 4 | O1 | representative_small | 4K | 1 | eager | cublaslt_nccl | 0.216528 | 0.212176 | 1.020511 |  | 是 |  |
| OProj Backward B→W | 4 | O1 | representative_small | 4K | 1 | graph | teub | 0.182848 | 0.204400 | 0.894560 | 是 | 是 |  |
| OProj Backward B→W | 4 | O2 | representative_medium | 4K | 0 | eager | cublaslt_nccl | 0.256256 | 0.294016 | 0.871572 | 是 | 是 |  |
| OProj Backward B→W | 4 | O2 | representative_medium | 4K | 0 | graph | teub | 0.226640 | 0.289680 | 0.782381 | 是 | 是 |  |
| OProj Backward B→W | 4 | O2 | representative_medium | 4K | 1 | eager | cublaslt_nccl | 0.280624 | 0.315184 | 0.890350 | 是 | 是 |  |
| OProj Backward B→W | 4 | O2 | representative_medium | 4K | 1 | graph | teub | 0.250928 | 0.309984 | 0.809487 | 是 | 是 |  |
| OProj Backward B→W | 4 | O3 | representative_large | 4K | 0 | eager | teub | 0.904592 | 1.125328 | 0.803847 | 是 | 是 |  |
| OProj Backward B→W | 4 | O3 | representative_large | 4K | 0 | graph | teub | 0.897136 | 1.118272 | 0.802252 | 是 | 是 |  |
| OProj Backward B→W | 4 | O3 | representative_large | 4K | 1 | eager | cublaslt_nccl | 0.985552 | 1.245088 | 0.791552 | 是 | 是 |  |
| OProj Backward B→W | 4 | O3 | representative_large | 4K | 1 | graph | teub | 0.948080 | 1.222048 | 0.775812 | 是 | 是 |  |
| OProj Backward B→W | 4 | O4 | production_qwen_dense | 4K | 0 | graph | teub | 0.072832 | 0.078256 | 0.930689 | 是 | 是 |  |
| OProj Backward B→W | 4 | O4 | production_qwen_dense | 4K | 1 | graph | teub | 0.079888 | 0.079904 | 0.999800 | 是 | 是 |  |
| OProj Backward B→W | 4 | O5 | nanbeige42_3b | 4K | 0 | eager | cublaslt_nccl | 0.223776 | 0.223344 | 1.001934 |  | 是 |  |
| OProj Backward B→W | 4 | O5 | nanbeige42_3b | 4K | 0 | graph | teub | 0.196960 | 0.225936 | 0.871751 | 是 | 是 |  |
| OProj Backward B→W | 4 | O5 | nanbeige42_3b | 4K | 1 | eager | cublaslt_nccl | 0.238960 | 0.238768 | 1.000804 |  | 是 |  |
| OProj Backward B→W | 4 | O5 | nanbeige42_3b | 4K | 1 | graph | teub | 0.214832 | 0.242112 | 0.887325 | 是 | 是 |  |
| OProj Backward B→W | 4 | O6 | llama3_8b | 4K | 0 | eager | cublaslt_nccl | 0.199232 | 0.195184 | 1.020739 |  | 是 |  |
| OProj Backward B→W | 4 | O6 | llama3_8b | 4K | 0 | graph | teub | 0.167776 | 0.191552 | 0.875877 | 是 | 是 |  |
| OProj Backward B→W | 4 | O6 | llama3_8b | 4K | 1 | eager | cublaslt_nccl | 0.214000 | 0.207536 | 1.031146 |  | 是 |  |
| OProj Backward B→W | 4 | O6 | llama3_8b | 4K | 1 | graph | teub | 0.181088 | 0.203024 | 0.891954 | 是 | 是 |  |
| OProj Backward B→W | 4 | O7 | qwen25_14b_32b | 4K | 0 | eager | cublaslt_nccl | 0.257488 | 0.290320 | 0.886911 | 是 | 是 |  |
| OProj Backward B→W | 4 | O7 | qwen25_14b_32b | 4K | 0 | graph | teub | 0.232000 | 0.283872 | 0.817270 | 是 | 是 |  |
| OProj Backward B→W | 4 | O7 | qwen25_14b_32b | 4K | 1 | eager | cublaslt_nccl | 0.279808 | 0.310992 | 0.899727 | 是 | 是 |  |
| OProj Backward B→W | 4 | O7 | qwen25_14b_32b | 4K | 1 | graph | teub | 0.253040 | 0.305424 | 0.828488 | 是 | 是 |  |
| OProj Backward B→W | 4 | O8 | llama31_405b | 4K | 0 | eager | teub | 1.906720 | 2.520384 | 0.756520 | 是 | 是 |  |
| OProj Backward B→W | 4 | O8 | llama31_405b | 4K | 0 | graph | teub | 1.910560 | 2.580560 | 0.740366 | 是 | 是 |  |
| OProj Backward B→W | 4 | O8 | llama31_405b | 4K | 1 | eager | teub | 2.051440 | 2.749584 | 0.746091 | 是 | 是 |  |
| OProj Backward B→W | 4 | O8 | llama31_405b | 4K | 1 | graph | teub | 2.045216 | 2.740816 | 0.746207 | 是 | 是 |  |
| OProj Backward B→W | 4 | O1 | representative_small | 16K | 0 | graph | teub | 0.517168 | 0.476400 | 1.085575 |  | 是 |  |
| OProj Backward B→W | 4 | O1 | representative_small | 16K | 1 | graph | teub | 0.524208 | 0.486752 | 1.076951 |  | 是 |  |
| OProj Backward B→W | 4 | O2 | representative_medium | 16K | 0 | eager | teub | 0.801536 | 0.732688 | 1.093966 |  | 是 |  |
| OProj Backward B→W | 4 | O2 | representative_medium | 16K | 0 | graph | teub | 0.771984 | 0.734480 | 1.051062 |  | 是 |  |
| OProj Backward B→W | 4 | O2 | representative_medium | 16K | 1 | eager | teub | 0.811808 | 0.756400 | 1.073252 |  | 是 |  |
| OProj Backward B→W | 4 | O2 | representative_medium | 16K | 1 | graph | teub | 0.771200 | 0.751536 | 1.026165 |  | 是 |  |
| OProj Backward B→W | 4 | O3 | representative_large | 16K | 0 | eager | teub | 3.283552 | 3.461696 | 0.948539 | 是 | 是 |  |
| OProj Backward B→W | 4 | O3 | representative_large | 16K | 0 | graph | cublaslt_nccl | 3.258432 | 3.511488 | 0.927935 | 是 | 是 |  |
| OProj Backward B→W | 4 | O3 | representative_large | 16K | 1 | eager | teub | 3.289568 | 3.540944 | 0.929009 | 是 | 是 |  |
| OProj Backward B→W | 4 | O3 | representative_large | 16K | 1 | graph | teub | 3.309376 | 3.698128 | 0.894879 | 是 | 是 |  |
| OProj Backward B→W | 4 | O6 | llama3_8b | 16K | 1 | graph | teub | 0.525360 | 0.482096 | 1.089741 |  | 是 |  |
| OProj Backward B→W | 4 | O7 | qwen25_14b_32b | 16K | 0 | eager | teub | 0.784560 | 0.732944 | 1.070423 |  | 是 |  |
| OProj Backward B→W | 4 | O7 | qwen25_14b_32b | 16K | 0 | graph | teub | 0.749920 | 0.735216 | 1.020000 |  | 是 |  |
| OProj Backward B→W | 4 | O7 | qwen25_14b_32b | 16K | 1 | eager | teub | 0.814000 | 0.759648 | 1.071549 |  | 是 |  |
| OProj Backward B→W | 4 | O7 | qwen25_14b_32b | 16K | 1 | graph | teub | 0.781744 | 0.757504 | 1.032000 |  | 是 |  |
| OProj Backward B→W | 4 | O8 | llama31_405b | 16K | 0 | eager | teub | 6.997312 | 7.938912 | 0.881394 | 是 | 是 |  |
| OProj Backward B→W | 4 | O8 | llama31_405b | 16K | 0 | graph | cublaslt_nccl | 7.010624 | 8.276880 | 0.847013 | 是 | 是 |  |
| OProj Backward B→W | 4 | O8 | llama31_405b | 16K | 1 | eager | teub | 7.117408 | 8.015472 | 0.887959 | 是 | 是 |  |
| OProj Backward B→W | 4 | O8 | llama31_405b | 16K | 1 | graph | teub | 7.069840 | 8.405456 | 0.841101 | 是 | 是 |  |
| OProj Backward B→W | 4 | O2 | representative_medium | 128K | 0 | eager | teub | 5.879536 | 5.471168 | 1.074640 |  | 是 | 是 |
| OProj Backward B→W | 4 | O2 | representative_medium | 128K | 0 | graph | teub | 5.878608 | 5.742288 | 1.023740 |  | 是 | 是 |
| OProj Backward B→W | 4 | O2 | representative_medium | 128K | 1 | eager | cublaslt_nccl | 6.064768 | 5.700512 | 1.063899 |  | 是 | 是 |
| OProj Backward B→W | 4 | O2 | representative_medium | 128K | 1 | graph | teub | 5.967568 | 5.429536 | 1.099094 |  | 是 | 是 |
| OProj Backward B→W | 4 | O3 | representative_large | 128K | 0 | eager | cublaslt_nccl | 26.075680 | 26.223231 | 0.994373 | 是 | 是 | 是 |
| OProj Backward B→W | 4 | O3 | representative_large | 128K | 0 | graph | teub | 25.759888 | 26.100688 | 0.986943 | 是 | 是 | 是 |
| OProj Backward B→W | 4 | O3 | representative_large | 128K | 1 | eager | teub | 25.944400 | 26.210912 | 0.989832 | 是 | 是 | 是 |
| OProj Backward B→W | 4 | O3 | representative_large | 128K | 1 | graph | teub | 25.974928 | 26.364593 | 0.985220 | 是 | 是 | 是 |
| OProj Backward B→W | 4 | O7 | qwen25_14b_32b | 128K | 0 | eager | teub | 5.898576 | 5.460864 | 1.080154 |  | 是 | 是 |
| OProj Backward B→W | 4 | O7 | qwen25_14b_32b | 128K | 0 | graph | teub | 5.902720 | 5.771264 | 1.022778 |  | 是 | 是 |
| OProj Backward B→W | 4 | O7 | qwen25_14b_32b | 128K | 1 | eager | teub | 6.098784 | 5.626240 | 1.083989 |  | 是 | 是 |
| OProj Backward B→W | 4 | O7 | qwen25_14b_32b | 128K | 1 | graph | teub | 5.897664 | 5.596176 | 1.053874 |  | 是 | 是 |
| OProj Backward B→W | 4 | O8 | llama31_405b | 128K | 0 | eager | teub | 55.914223 | 59.642817 | 0.937485 | 是 | 是 | 是 |
| OProj Backward B→W | 4 | O8 | llama31_405b | 128K | 0 | graph | teub | 56.083647 | 59.443905 | 0.943472 | 是 | 是 | 是 |
| OProj Backward B→W | 4 | O8 | llama31_405b | 128K | 1 | eager | teub | 56.180992 | 59.416769 | 0.945541 | 是 | 是 | 是 |
| OProj Backward B→W | 4 | O8 | llama31_405b | 128K | 1 | graph | cublaslt_nccl | 55.920401 | 59.034704 | 0.947246 | 是 | 是 | 是 |
| OProj Backward B→W | 4 | O2 | representative_medium | 256K | 0 | eager | teub | 12.064512 | 11.244368 | 1.072938 |  | 是 | 是 |
| OProj Backward B→W | 4 | O2 | representative_medium | 256K | 0 | graph | teub | 11.906512 | 11.425552 | 1.042095 |  | 是 | 是 |
| OProj Backward B→W | 4 | O2 | representative_medium | 256K | 1 | eager | teub | 12.086352 | 11.314016 | 1.068264 |  | 是 | 是 |
| OProj Backward B→W | 4 | O2 | representative_medium | 256K | 1 | graph | teub | 12.069072 | 11.360288 | 1.062391 |  | 是 | 是 |
| OProj Backward B→W | 4 | O3 | representative_large | 256K | 0 | eager | teub | 52.047825 | 52.157488 | 0.997897 | 是 | 是 | 是 |
| OProj Backward B→W | 4 | O3 | representative_large | 256K | 0 | graph | cublaslt_nccl | 52.213232 | 52.166990 | 1.000886 |  | 是 | 是 |
| OProj Backward B→W | 4 | O3 | representative_large | 256K | 1 | eager | teub | 52.106575 | 52.252560 | 0.997206 | 是 | 是 | 是 |
| OProj Backward B→W | 4 | O3 | representative_large | 256K | 1 | graph | teub | 52.263742 | 52.040512 | 1.004290 |  | 是 | 是 |
| OProj Backward B→W | 4 | O7 | qwen25_14b_32b | 256K | 0 | eager | cublaslt_nccl | 12.129056 | 11.194192 | 1.083513 |  | 是 | 是 |
| OProj Backward B→W | 4 | O7 | qwen25_14b_32b | 256K | 0 | graph | teub | 11.916096 | 11.260640 | 1.058208 |  | 是 | 是 |
| OProj Backward B→W | 4 | O7 | qwen25_14b_32b | 256K | 1 | eager | cublaslt_nccl | 12.139808 | 11.412576 | 1.063722 |  | 是 | 是 |
| OProj Backward B→W | 4 | O7 | qwen25_14b_32b | 256K | 1 | graph | teub | 12.077808 | 11.349632 | 1.064159 |  | 是 | 是 |
| OProj Backward B→W | 4 | O8 | llama31_405b | 256K | 0 | eager | cublaslt_nccl | 111.848675 | 119.264992 | 0.937816 | 是 | 是 | 是 |
| OProj Backward B→W | 4 | O8 | llama31_405b | 256K | 0 | graph | cublaslt_nccl | 112.664608 | 119.526035 | 0.942595 | 是 | 是 | 是 |
| OProj Backward B→W | 4 | O8 | llama31_405b | 256K | 1 | eager | teub | 111.364754 | 119.357456 | 0.933036 | 是 | 是 | 是 |
| OProj Backward B→W | 4 | O8 | llama31_405b | 256K | 1 | graph | teub | 111.646851 | 119.654926 | 0.933074 | 是 | 是 | 是 |
| OProj Backward B→W | 4 | O2 | representative_medium | 512K | 0 | eager | teub | 24.209455 | 22.548800 | 1.073647 |  | 是 | 是 |
| OProj Backward B→W | 4 | O2 | representative_medium | 512K | 0 | graph | teub | 24.245136 | 22.662096 | 1.069854 |  | 是 | 是 |
| OProj Backward B→W | 4 | O2 | representative_medium | 512K | 1 | eager | teub | 24.507264 | 22.776672 | 1.075981 |  | 是 | 是 |
| OProj Backward B→W | 4 | O2 | representative_medium | 512K | 1 | graph | teub | 24.383232 | 22.751663 | 1.071712 |  | 是 | 是 |
| OProj Backward B→W | 4 | O3 | representative_large | 512K | 0 | eager | teub | 106.873409 | 105.016350 | 1.017684 |  | 是 | 是 |
| OProj Backward B→W | 4 | O3 | representative_large | 512K | 0 | graph | cublaslt_nccl | 104.390034 | 103.312241 | 1.010432 |  | 是 | 是 |
| OProj Backward B→W | 4 | O3 | representative_large | 512K | 1 | eager | cublaslt_nccl | 105.375809 | 104.920559 | 1.004339 |  | 是 | 是 |
| OProj Backward B→W | 4 | O3 | representative_large | 512K | 1 | graph | cublaslt_nccl | 104.686626 | 104.845005 | 0.998489 | 是 | 是 | 是 |
| OProj Backward B→W | 4 | O7 | qwen25_14b_32b | 512K | 0 | eager | teub | 24.237792 | 22.537872 | 1.075425 |  | 是 | 是 |
| OProj Backward B→W | 4 | O7 | qwen25_14b_32b | 512K | 0 | graph | teub | 24.088672 | 22.795199 | 1.056743 |  | 是 | 是 |
| OProj Backward B→W | 4 | O7 | qwen25_14b_32b | 512K | 1 | eager | teub | 24.387312 | 22.735712 | 1.072643 |  | 是 | 是 |
| OProj Backward B→W | 4 | O7 | qwen25_14b_32b | 512K | 1 | graph | teub | 24.347456 | 22.645872 | 1.075139 |  | 是 | 是 |
| OProj Backward B→W | 4 | O8 | llama31_405b | 512K | 0 | eager | cublaslt_nccl | 221.515099 | 239.195869 | 0.926082 | 是 | 是 | 是 |
| OProj Backward B→W | 4 | O8 | llama31_405b | 512K | 0 | graph | teub | 222.582283 | 238.660240 | 0.932632 | 是 | 是 | 是 |
| OProj Backward B→W | 4 | O8 | llama31_405b | 512K | 1 | eager | cublaslt_nccl | 222.463570 | 238.486786 | 0.932813 | 是 | 是 | 是 |
| OProj Backward B→W | 4 | O8 | llama31_405b | 512K | 1 | graph | teub | 223.729347 | 238.536140 | 0.937926 | 是 | 是 | 是 |
| OProj Backward B→W | 8 | O1 | representative_small | 1K | 0 | graph | cublaslt_nccl | 0.099216 | 0.133520 | 0.743080 | 是 | 是 |  |
| OProj Backward B→W | 8 | O1 | representative_small | 1K | 1 | graph | cublaslt_nccl | 0.129104 | 0.156112 | 0.826996 | 是 | 是 |  |
| OProj Backward B→W | 8 | O2 | representative_medium | 1K | 0 | eager | cublaslt_nccl | 0.189984 | 0.181088 | 1.049125 |  | 是 |  |
| OProj Backward B→W | 8 | O2 | representative_medium | 1K | 0 | graph | cublaslt_nccl | 0.134064 | 0.175840 | 0.762420 | 是 | 是 |  |
| OProj Backward B→W | 8 | O2 | representative_medium | 1K | 1 | graph | teub | 0.174016 | 0.201728 | 0.862627 | 是 | 是 |  |
| OProj Backward B→W | 8 | O3 | representative_large | 1K | 0 | eager | cublaslt_nccl | 0.391712 | 0.611088 | 0.641007 | 是 | 是 |  |
| OProj Backward B→W | 8 | O3 | representative_large | 1K | 0 | graph | teub | 0.351888 | 0.604960 | 0.581672 | 是 | 是 |  |
| OProj Backward B→W | 8 | O3 | representative_large | 1K | 1 | eager | cublaslt_nccl | 0.615888 | 0.728224 | 0.845740 | 是 | 是 |  |
| OProj Backward B→W | 8 | O3 | representative_large | 1K | 1 | graph | teub | 0.580448 | 0.723488 | 0.802291 | 是 | 是 |  |
| OProj Backward B→W | 8 | O4 | production_qwen_dense | 1K | 0 | graph | teub | 0.063024 | 0.071456 | 0.881997 | 是 | 是 |  |
| OProj Backward B→W | 8 | O4 | production_qwen_dense | 1K | 1 | graph | teub | 0.062784 | 0.070624 | 0.888990 | 是 | 是 |  |
| OProj Backward B→W | 8 | O5 | nanbeige42_3b | 1K | 0 | graph | teub | 0.109200 | 0.133632 | 0.817170 | 是 | 是 |  |
| OProj Backward B→W | 8 | O5 | nanbeige42_3b | 1K | 1 | graph | teub | 0.142016 | 0.155536 | 0.913075 | 是 | 是 |  |
| OProj Backward B→W | 8 | O6 | llama3_8b | 1K | 0 | graph | cublaslt_nccl | 0.102992 | 0.125040 | 0.823672 | 是 | 是 |  |
| OProj Backward B→W | 8 | O6 | llama3_8b | 1K | 1 | graph | teub | 0.132512 | 0.139888 | 0.947272 | 是 | 是 |  |
| OProj Backward B→W | 8 | O7 | qwen25_14b_32b | 1K | 0 | eager | cublaslt_nccl | 0.178288 | 0.182240 | 0.978314 | 是 | 是 |  |
| OProj Backward B→W | 8 | O7 | qwen25_14b_32b | 1K | 0 | graph | teub | 0.132592 | 0.175376 | 0.756044 | 是 | 是 |  |
| OProj Backward B→W | 8 | O7 | qwen25_14b_32b | 1K | 1 | graph | teub | 0.170992 | 0.198768 | 0.860259 | 是 | 是 |  |
| OProj Backward B→W | 8 | O8 | llama31_405b | 1K | 0 | eager | cublaslt_nccl | 0.732448 | 1.311792 | 0.558357 | 是 | 是 |  |
| OProj Backward B→W | 8 | O8 | llama31_405b | 1K | 0 | graph | teub | 0.700800 | 1.308256 | 0.535675 | 是 | 是 |  |
| OProj Backward B→W | 8 | O8 | llama31_405b | 1K | 1 | eager | cublaslt_nccl | 1.247888 | 1.586672 | 0.786481 | 是 | 是 |  |
| OProj Backward B→W | 8 | O8 | llama31_405b | 1K | 1 | graph | teub | 1.210224 | 1.579792 | 0.766065 | 是 | 是 |  |
| OProj Backward B→W | 8 | O1 | representative_small | 4K | 0 | eager | cublaslt_nccl | 0.193280 | 0.181040 | 1.067609 |  | 是 |  |
| OProj Backward B→W | 8 | O1 | representative_small | 4K | 0 | graph | teub | 0.141856 | 0.169776 | 0.835548 | 是 | 是 |  |
| OProj Backward B→W | 8 | O1 | representative_small | 4K | 1 | graph | teub | 0.166032 | 0.184592 | 0.899454 | 是 | 是 |  |
| OProj Backward B→W | 8 | O2 | representative_medium | 4K | 0 | eager | cublaslt_nccl | 0.215440 | 0.270032 | 0.797831 | 是 | 是 |  |
| OProj Backward B→W | 8 | O2 | representative_medium | 4K | 0 | graph | teub | 0.181568 | 0.264160 | 0.687341 | 是 | 是 |  |
| OProj Backward B→W | 8 | O2 | representative_medium | 4K | 1 | eager | cublaslt_nccl | 0.247136 | 0.294912 | 0.837999 | 是 | 是 |  |
| OProj Backward B→W | 8 | O2 | representative_medium | 4K | 1 | graph | teub | 0.211632 | 0.286128 | 0.739641 | 是 | 是 |  |
| OProj Backward B→W | 8 | O3 | representative_large | 4K | 0 | eager | teub | 0.641664 | 0.957856 | 0.669896 | 是 | 是 |  |
| OProj Backward B→W | 8 | O3 | representative_large | 4K | 0 | graph | teub | 0.603392 | 0.944352 | 0.638948 | 是 | 是 |  |
| OProj Backward B→W | 8 | O3 | representative_large | 4K | 1 | eager | teub | 0.784192 | 1.062048 | 0.738377 | 是 | 是 |  |
| OProj Backward B→W | 8 | O3 | representative_large | 4K | 1 | graph | teub | 0.743920 | 1.053088 | 0.706418 | 是 | 是 |  |
| OProj Backward B→W | 8 | O4 | production_qwen_dense | 4K | 0 | graph | teub | 0.076112 | 0.077184 | 0.986111 | 是 | 是 |  |
| OProj Backward B→W | 8 | O4 | production_qwen_dense | 4K | 1 | graph | teub | 0.077312 | 0.081104 | 0.953245 | 是 | 是 |  |
| OProj Backward B→W | 8 | O5 | nanbeige42_3b | 4K | 0 | eager | cublaslt_nccl | 0.192208 | 0.201696 | 0.952959 | 是 | 是 |  |
| OProj Backward B→W | 8 | O5 | nanbeige42_3b | 4K | 0 | graph | teub | 0.164608 | 0.199120 | 0.826677 | 是 | 是 |  |
| OProj Backward B→W | 8 | O5 | nanbeige42_3b | 4K | 1 | eager | cublaslt_nccl | 0.218720 | 0.218544 | 1.000805 |  | 是 |  |
| OProj Backward B→W | 8 | O5 | nanbeige42_3b | 4K | 1 | graph | teub | 0.188768 | 0.221952 | 0.850490 | 是 | 是 |  |
| OProj Backward B→W | 8 | O6 | llama3_8b | 4K | 0 | eager | cublaslt_nccl | 0.191664 | 0.178592 | 1.073195 |  | 是 |  |
| OProj Backward B→W | 8 | O6 | llama3_8b | 4K | 0 | graph | teub | 0.144320 | 0.173824 | 0.830265 | 是 | 是 |  |
| OProj Backward B→W | 8 | O6 | llama3_8b | 4K | 1 | graph | teub | 0.167088 | 0.187712 | 0.890130 | 是 | 是 |  |
| OProj Backward B→W | 8 | O7 | qwen25_14b_32b | 4K | 0 | eager | cublaslt_nccl | 0.224592 | 0.269152 | 0.834443 | 是 | 是 |  |
| OProj Backward B→W | 8 | O7 | qwen25_14b_32b | 4K | 0 | graph | teub | 0.184384 | 0.262112 | 0.703455 | 是 | 是 |  |
| OProj Backward B→W | 8 | O7 | qwen25_14b_32b | 4K | 1 | eager | cublaslt_nccl | 0.245696 | 0.290048 | 0.847087 | 是 | 是 |  |
| OProj Backward B→W | 8 | O7 | qwen25_14b_32b | 4K | 1 | graph | teub | 0.218080 | 0.286640 | 0.760815 | 是 | 是 |  |
| OProj Backward B→W | 8 | O8 | llama31_405b | 4K | 0 | eager | teub | 1.235440 | 2.071664 | 0.596352 | 是 | 是 |  |
| OProj Backward B→W | 8 | O8 | llama31_405b | 4K | 0 | graph | teub | 1.195392 | 2.068480 | 0.577908 | 是 | 是 |  |
| OProj Backward B→W | 8 | O8 | llama31_405b | 4K | 1 | eager | cublaslt_nccl | 1.564432 | 2.312304 | 0.676568 | 是 | 是 |  |
| OProj Backward B→W | 8 | O8 | llama31_405b | 4K | 1 | graph | teub | 1.525488 | 2.306960 | 0.661255 | 是 | 是 |  |
| OProj Backward B→W | 8 | O1 | representative_small | 16K | 0 | eager | teub | 0.383216 | 0.372752 | 1.028072 |  | 是 |  |
| OProj Backward B→W | 8 | O1 | representative_small | 16K | 0 | graph | teub | 0.342672 | 0.370608 | 0.924621 | 是 | 是 |  |
| OProj Backward B→W | 8 | O1 | representative_small | 16K | 1 | eager | cublaslt_nccl | 0.398816 | 0.384368 | 1.037589 |  | 是 |  |
| OProj Backward B→W | 8 | O1 | representative_small | 16K | 1 | graph | teub | 0.350656 | 0.380656 | 0.921189 | 是 | 是 |  |
| OProj Backward B→W | 8 | O2 | representative_medium | 16K | 0 | eager | cublaslt_nccl | 0.510080 | 0.538448 | 0.947315 | 是 | 是 |  |
| OProj Backward B→W | 8 | O2 | representative_medium | 16K | 0 | graph | teub | 0.478672 | 0.532176 | 0.899462 | 是 | 是 |  |
| OProj Backward B→W | 8 | O2 | representative_medium | 16K | 1 | eager | cublaslt_nccl | 0.519888 | 0.559584 | 0.929062 | 是 | 是 |  |
| OProj Backward B→W | 8 | O2 | representative_medium | 16K | 1 | graph | teub | 0.490336 | 0.564416 | 0.868749 | 是 | 是 |  |
| OProj Backward B→W | 8 | O3 | representative_large | 16K | 0 | eager | cublaslt_nccl | 1.797728 | 2.068960 | 0.868904 | 是 | 是 |  |
| OProj Backward B→W | 8 | O3 | representative_large | 16K | 0 | graph | teub | 1.771536 | 2.069408 | 0.856059 | 是 | 是 |  |
| OProj Backward B→W | 8 | O3 | representative_large | 16K | 1 | eager | cublaslt_nccl | 1.814960 | 2.150384 | 0.844017 | 是 | 是 |  |
| OProj Backward B→W | 8 | O3 | representative_large | 16K | 1 | graph | teub | 1.789520 | 2.145696 | 0.834004 | 是 | 是 |  |
| OProj Backward B→W | 8 | O4 | production_qwen_dense | 16K | 0 | graph | teub | 0.137616 | 0.131568 | 1.045969 |  | 是 |  |
| OProj Backward B→W | 8 | O4 | production_qwen_dense | 16K | 1 | graph | teub | 0.140608 | 0.135872 | 1.034856 |  | 是 |  |
| OProj Backward B→W | 8 | O5 | nanbeige42_3b | 16K | 0 | eager | teub | 0.437952 | 0.422128 | 1.037486 |  | 是 |  |
| OProj Backward B→W | 8 | O5 | nanbeige42_3b | 16K | 0 | graph | teub | 0.398912 | 0.417776 | 0.954847 | 是 | 是 |  |
| OProj Backward B→W | 8 | O5 | nanbeige42_3b | 16K | 1 | eager | teub | 0.439376 | 0.437776 | 1.003655 |  | 是 |  |
| OProj Backward B→W | 8 | O5 | nanbeige42_3b | 16K | 1 | graph | teub | 0.413488 | 0.434704 | 0.951194 | 是 | 是 |  |
| OProj Backward B→W | 8 | O6 | llama3_8b | 16K | 0 | eager | teub | 0.387840 | 0.379472 | 1.022052 |  | 是 |  |
| OProj Backward B→W | 8 | O6 | llama3_8b | 16K | 0 | graph | teub | 0.353264 | 0.370448 | 0.953613 | 是 | 是 |  |
| OProj Backward B→W | 8 | O6 | llama3_8b | 16K | 1 | eager | teub | 0.397184 | 0.389520 | 1.019675 |  | 是 |  |
| OProj Backward B→W | 8 | O6 | llama3_8b | 16K | 1 | graph | teub | 0.360336 | 0.389200 | 0.925838 | 是 | 是 |  |
| OProj Backward B→W | 8 | O7 | qwen25_14b_32b | 16K | 0 | eager | cublaslt_nccl | 0.505952 | 0.537568 | 0.941187 | 是 | 是 |  |
| OProj Backward B→W | 8 | O7 | qwen25_14b_32b | 16K | 0 | graph | teub | 0.483504 | 0.531040 | 0.910485 | 是 | 是 |  |
| OProj Backward B→W | 8 | O7 | qwen25_14b_32b | 16K | 1 | eager | cublaslt_nccl | 0.516912 | 0.550112 | 0.939649 | 是 | 是 |  |
| OProj Backward B→W | 8 | O7 | qwen25_14b_32b | 16K | 1 | graph | teub | 0.496480 | 0.553616 | 0.896795 | 是 | 是 |  |
| OProj Backward B→W | 8 | O8 | llama31_405b | 16K | 0 | eager | teub | 3.644352 | 4.518640 | 0.806515 | 是 | 是 |  |
| OProj Backward B→W | 8 | O8 | llama31_405b | 16K | 0 | graph | teub | 3.615888 | 4.502048 | 0.803165 | 是 | 是 |  |
| OProj Backward B→W | 8 | O8 | llama31_405b | 16K | 1 | eager | cublaslt_nccl | 3.660784 | 4.918720 | 0.744255 | 是 | 是 |  |
| OProj Backward B→W | 8 | O8 | llama31_405b | 16K | 1 | graph | cublaslt_nccl | 3.790912 | 4.722496 | 0.802735 | 是 | 是 |  |
| OProj Backward B→W | 8 | O2 | representative_medium | 128K | 0 | eager | cublaslt_nccl | 3.185360 | 2.894768 | 1.100385 |  |  | 是 |
| OProj Backward B→W | 8 | O2 | representative_medium | 128K | 0 | graph | teub | 3.180256 | 2.878000 | 1.105023 |  |  | 是 |
| OProj Backward B→W | 8 | O2 | representative_medium | 128K | 1 | eager | cublaslt_nccl | 3.209664 | 2.901920 | 1.106048 |  |  | 是 |
| OProj Backward B→W | 8 | O2 | representative_medium | 128K | 1 | graph | cublaslt_nccl | 3.172992 | 2.901968 | 1.093393 |  | 是 | 是 |
| OProj Backward B→W | 8 | O3 | representative_large | 128K | 0 | eager | cublaslt_nccl | 13.205088 | 12.904064 | 1.023328 |  | 是 | 是 |
| OProj Backward B→W | 8 | O3 | representative_large | 128K | 0 | graph | cublaslt_nccl | 13.349008 | 13.155040 | 1.014745 |  | 是 | 是 |
| OProj Backward B→W | 8 | O3 | representative_large | 128K | 1 | eager | cublaslt_nccl | 13.289248 | 13.153264 | 1.010338 |  | 是 | 是 |
| OProj Backward B→W | 8 | O3 | representative_large | 128K | 1 | graph | teub | 13.306976 | 13.471664 | 0.987775 | 是 | 是 | 是 |
| OProj Backward B→W | 8 | O7 | qwen25_14b_32b | 128K | 0 | eager | cublaslt_nccl | 3.185856 | 2.897184 | 1.099639 |  | 是 | 是 |
| OProj Backward B→W | 8 | O7 | qwen25_14b_32b | 128K | 0 | graph | cublaslt_nccl | 3.162864 | 2.894368 | 1.092765 |  | 是 | 是 |
| OProj Backward B→W | 8 | O7 | qwen25_14b_32b | 128K | 1 | eager | teub | 3.225360 | 2.916624 | 1.105854 |  |  | 是 |
| OProj Backward B→W | 8 | O7 | qwen25_14b_32b | 128K | 1 | graph | cublaslt_nccl | 3.165552 | 2.908464 | 1.088393 |  | 是 | 是 |
| OProj Backward B→W | 8 | O8 | llama31_405b | 128K | 0 | eager | cublaslt_nccl | 28.534016 | 29.689040 | 0.961096 | 是 | 是 | 是 |
| OProj Backward B→W | 8 | O8 | llama31_405b | 128K | 0 | graph | teub | 28.604352 | 30.346080 | 0.942605 | 是 | 是 | 是 |
| OProj Backward B→W | 8 | O8 | llama31_405b | 128K | 1 | eager | teub | 28.835808 | 30.241776 | 0.953509 | 是 | 是 | 是 |
| OProj Backward B→W | 8 | O8 | llama31_405b | 128K | 1 | graph | cublaslt_nccl | 28.682719 | 30.421568 | 0.942842 | 是 | 是 | 是 |
| OProj Backward B→W | 8 | O2 | representative_medium | 256K | 0 | eager | cublaslt_nccl | 6.189312 | 5.591792 | 1.106857 |  |  | 是 |
| OProj Backward B→W | 8 | O2 | representative_medium | 256K | 0 | graph | cublaslt_nccl | 6.195952 | 5.670208 | 1.092720 |  | 是 | 是 |
| OProj Backward B→W | 8 | O2 | representative_medium | 256K | 1 | eager | cublaslt_nccl | 6.295376 | 5.625648 | 1.119049 |  |  | 是 |
| OProj Backward B→W | 8 | O2 | representative_medium | 256K | 1 | graph | cublaslt_nccl | 6.210016 | 5.620736 | 1.104840 |  |  | 是 |
| OProj Backward B→W | 8 | O3 | representative_large | 256K | 0 | eager | cublaslt_nccl | 26.645968 | 26.464688 | 1.006850 |  | 是 | 是 |
| OProj Backward B→W | 8 | O3 | representative_large | 256K | 0 | graph | cublaslt_nccl | 26.511712 | 26.367936 | 1.005453 |  | 是 | 是 |
| OProj Backward B→W | 8 | O3 | representative_large | 256K | 1 | eager | cublaslt_nccl | 26.363487 | 26.324160 | 1.001494 |  | 是 | 是 |
| OProj Backward B→W | 8 | O3 | representative_large | 256K | 1 | graph | cublaslt_nccl | 26.502144 | 26.424144 | 1.002952 |  | 是 | 是 |
| OProj Backward B→W | 8 | O7 | qwen25_14b_32b | 256K | 0 | eager | cublaslt_nccl | 6.224704 | 5.596704 | 1.112209 |  |  | 是 |
| OProj Backward B→W | 8 | O7 | qwen25_14b_32b | 256K | 0 | graph | cublaslt_nccl | 6.210800 | 5.657520 | 1.097795 |  | 是 | 是 |
| OProj Backward B→W | 8 | O7 | qwen25_14b_32b | 256K | 1 | eager | cublaslt_nccl | 6.286464 | 5.644960 | 1.113642 |  |  | 是 |
| OProj Backward B→W | 8 | O7 | qwen25_14b_32b | 256K | 1 | graph | cublaslt_nccl | 6.217872 | 5.628512 | 1.104710 |  |  | 是 |
| OProj Backward B→W | 8 | O8 | llama31_405b | 256K | 0 | eager | cublaslt_nccl | 57.026497 | 60.212831 | 0.947082 | 是 | 是 | 是 |
| OProj Backward B→W | 8 | O8 | llama31_405b | 256K | 0 | graph | cublaslt_nccl | 57.025888 | 60.167057 | 0.947793 | 是 | 是 | 是 |
| OProj Backward B→W | 8 | O8 | llama31_405b | 256K | 1 | eager | cublaslt_nccl | 56.798496 | 60.112638 | 0.944868 | 是 | 是 | 是 |
| OProj Backward B→W | 8 | O8 | llama31_405b | 256K | 1 | graph | cublaslt_nccl | 56.531311 | 60.063616 | 0.941191 | 是 | 是 | 是 |
| OProj Backward B→W | 8 | O2 | representative_medium | 512K | 0 | eager | cublaslt_nccl | 12.607472 | 11.241280 | 1.121534 |  |  | 是 |
| OProj Backward B→W | 8 | O2 | representative_medium | 512K | 0 | graph | cublaslt_nccl | 12.449408 | 11.209296 | 1.110632 |  |  | 是 |
| OProj Backward B→W | 8 | O2 | representative_medium | 512K | 1 | eager | cublaslt_nccl | 12.515344 | 11.304592 | 1.107103 |  |  | 是 |
| OProj Backward B→W | 8 | O2 | representative_medium | 512K | 1 | graph | cublaslt_nccl | 12.472384 | 11.289472 | 1.104780 |  |  | 是 |
| OProj Backward B→W | 8 | O3 | representative_large | 512K | 0 | eager | cublaslt_nccl | 52.779873 | 52.629744 | 1.002853 |  | 是 | 是 |
| OProj Backward B→W | 8 | O3 | representative_large | 512K | 0 | graph | cublaslt_nccl | 52.805456 | 52.744816 | 1.001150 |  | 是 | 是 |
| OProj Backward B→W | 8 | O3 | representative_large | 512K | 1 | eager | cublaslt_nccl | 52.896080 | 52.788752 | 1.002033 |  | 是 | 是 |
| OProj Backward B→W | 8 | O3 | representative_large | 512K | 1 | graph | cublaslt_nccl | 52.788832 | 52.735792 | 1.001006 |  | 是 | 是 |
| OProj Backward B→W | 8 | O7 | qwen25_14b_32b | 512K | 0 | eager | cublaslt_nccl | 12.670720 | 11.256880 | 1.125598 |  |  | 是 |
| OProj Backward B→W | 8 | O7 | qwen25_14b_32b | 512K | 0 | graph | cublaslt_nccl | 12.420784 | 11.274160 | 1.101704 |  |  | 是 |
| OProj Backward B→W | 8 | O7 | qwen25_14b_32b | 512K | 1 | eager | cublaslt_nccl | 12.559792 | 11.350016 | 1.106588 |  |  | 是 |
| OProj Backward B→W | 8 | O7 | qwen25_14b_32b | 512K | 1 | graph | teub | 12.599072 | 11.312864 | 1.113694 |  |  | 是 |
| OProj Backward B→W | 8 | O8 | llama31_405b | 512K | 0 | eager | cublaslt_nccl | 113.379536 | 119.533855 | 0.948514 | 是 | 是 | 是 |
| OProj Backward B→W | 8 | O8 | llama31_405b | 512K | 0 | graph | cublaslt_nccl | 113.500992 | 119.527149 | 0.949583 | 是 | 是 | 是 |
| OProj Backward B→W | 8 | O8 | llama31_405b | 512K | 1 | eager | teub | 113.686291 | 119.626049 | 0.950347 | 是 | 是 | 是 |
| OProj Backward B→W | 8 | O8 | llama31_405b | 512K | 1 | graph | cublaslt_nccl | 113.074306 | 119.840782 | 0.943538 | 是 | 是 | 是 |

合计 697 个边界。全量 <1.1：642/1152；长序列 <1.1：201/576。

## 后续 CP4 QKV Forward：单列，不替换全量快照

同一冻结外部基线 / 当前模型开启时延。时延来自正反顺序各三轮 p50 的几何平均，外部基线未同轮重测。当前模型为显式启用，不是默认已切换。

| Shape | 128K E/G | 256K E/G | 512K E/G |
|---|---:|---:|---:|
| Q1 | 1.272 / 1.207 | 1.245 / 1.247 | 1.219 / 1.308 |
| Q2 | 1.174 / 1.181 | 1.165 / 1.157 | 1.170 / 1.172 |
| Q3 | 0.986 / 1.005 | 0.994 / 0.991 | 0.998 / 1.011 |
| Q4 | 1.559 / 1.563 | 1.464 / 1.438 | 1.417 / 1.381 |
| Q5 | 1.228 / 1.234 | 1.195 / 1.198 | 1.215 / 1.227 |
| Q6 | 1.208 / 1.218 | 1.161 / 1.177 | 1.168 / 1.166 |
| Q7 | 1.134 / 1.105 | 1.143 / 1.181 | 1.184 / 1.194 |
| Q8 | 0.971 / 0.963 | 0.959 / 0.968 | 0.956 / 0.946 |

当前模型相对同轮 auto：GM 1.065537×；归档外部基线 / 当前模型：GM 1.166200×。原 scope 阈值 1.123583×，仍低于阈值的 13 个边界：

- Q3 / 128K / eager: 0.986366×
- Q3 / 128K / graph: 1.004720×
- Q3 / 256K / eager: 0.994434×
- Q3 / 256K / graph: 0.990944×
- Q3 / 512K / eager: 0.998013×
- Q3 / 512K / graph: 1.011385×
- Q7 / 128K / graph: 1.105084×
- Q8 / 128K / eager: 0.971051×
- Q8 / 128K / graph: 0.962852×
- Q8 / 256K / eager: 0.958970×
- Q8 / 256K / graph: 0.967832×
- Q8 / 512K / eager: 0.955936×
- Q8 / 512K / graph: 0.945805×

## 后续 CP4 QKV Backward：候选验证，不替换全量快照

Q3/Q8、Graph、128K/256K/512K、beta0/1，实际完整 B→W；c12→c4 对当前 auto 双顺序配对 GM 1.019534×，不是相对外部基线的倍数。

| Shape | β | 128K vs auto | 256K vs auto | 512K vs auto |
|---|---:|---:|---:|---:|
| Q3 | 0 | 1.018997 | 1.019341 | 1.017885 |
| Q3 | 1 | 1.017002 | 1.021310 | 1.018911 |
| Q8 | 0 | 1.019067 | 1.023646 | 1.020272 |
| Q8 | 1 | 1.015343 | 1.022143 | 1.020523 |

Q2/Q7、Graph、完整 B→W c12→c8：单顺序三轮 GM 0.995826×，最差 Q7/128K/beta0 为 0.952185×，未接受此候选；没有改生产默认。
