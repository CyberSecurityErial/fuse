# QKV backward：fused 与 TE+NCCL

比值为 `TE+NCCL p50 / fused p50`；大于 1 表示 fused 更快。

| CP | 模型 | S | 模式 | fused Eager / TE / 加速 | fused Graph / TE / 加速 |
|---:|---|---:|---|---:|---:|
| 4 | artificial_medium | 1K | immediate | 0.0944 / 0.1884 / 1.997x | 0.0884 / 0.1063 / 1.203x |
| 4 | artificial_medium | 1K | deferred | 0.1154 / 0.2441 / 2.114x | 0.1011 / 0.1182 / 1.169x |
| 4 | artificial_medium | 16K | immediate | 0.6152 / 0.8248 / 1.341x | 0.6087 / 0.8026 / 1.319x |
| 4 | artificial_medium | 16K | deferred | 0.6352 / 0.8536 / 1.344x | 0.6422 / 0.8068 / 1.256x |
| 4 | artificial_medium | 128K | immediate | 4.9124 / 6.2687 / 1.276x | 4.9201 / 6.2912 / 1.279x |
| 4 | artificial_medium | 128K | deferred | 4.7795 / 6.3222 / 1.323x | 4.8893 / 6.2928 / 1.287x |
| 4 | nanbeige42_3b | 1K | immediate | 0.1032 / 0.1906 / 1.848x | 0.0970 / 0.1164 / 1.200x |
| 4 | nanbeige42_3b | 1K | deferred | 0.1130 / 0.2190 / 1.938x | 0.1076 / 0.1296 / 1.204x |
| 4 | nanbeige42_3b | 16K | immediate | 0.5983 / 0.8937 / 1.494x | 0.5953 / 0.8624 / 1.449x |
| 4 | nanbeige42_3b | 16K | deferred | 0.6100 / 0.9229 / 1.513x | 0.6054 / 0.8713 / 1.439x |
| 4 | nanbeige42_3b | 128K | immediate | 4.5825 / 6.7791 / 1.479x | 4.5268 / 6.7857 / 1.499x |
| 4 | nanbeige42_3b | 128K | deferred | 4.6306 / 6.8407 / 1.477x | 4.5544 / 6.8160 / 1.497x |
| 8 | artificial_medium | 1K | immediate | 0.0934 / 0.1947 / 2.084x | 0.0869 / 0.0972 / 1.118x |
| 8 | artificial_medium | 1K | deferred | 0.1064 / 0.2191 / 2.059x | 0.1003 / 0.1072 / 1.069x |
| 8 | artificial_medium | 16K | immediate | 0.4463 / 0.5488 / 1.230x | 0.4301 / 0.5180 / 1.204x |
| 8 | artificial_medium | 16K | deferred | 0.4633 / 0.5843 / 1.261x | 0.4525 / 0.5360 / 1.185x |
| 8 | artificial_medium | 128K | immediate | 2.8631 / 3.6749 / 1.284x | 2.8411 / 3.6527 / 1.286x |
| 8 | artificial_medium | 128K | deferred | 2.8756 / 3.7295 / 1.297x | 2.8613 / 3.6743 / 1.284x |
| 8 | nanbeige42_3b | 1K | immediate | 0.1055 / 0.1867 / 1.770x | 0.0968 / 0.1093 / 1.129x |
| 8 | nanbeige42_3b | 1K | deferred | 0.1161 / 0.2208 / 1.903x | 0.1102 / 0.1218 / 1.105x |
| 8 | nanbeige42_3b | 16K | immediate | 0.4497 / 0.5929 / 1.318x | 0.4252 / 0.5606 / 1.319x |
| 8 | nanbeige42_3b | 16K | deferred | 0.4634 / 0.6260 / 1.351x | 0.4547 / 0.5777 / 1.270x |
| 8 | nanbeige42_3b | 128K | immediate | 2.6201 / 3.9468 / 1.506x | 2.5669 / 3.9144 / 1.525x |
| 8 | nanbeige42_3b | 128K | deferred | 2.6162 / 4.0031 / 1.530x | 2.5980 / 3.9556 / 1.523x |

immediate：fused 对 TE+NCCL 的 Eager 胜点 12/12，几何平均 1.527x；Graph 胜点 12/12，几何平均 1.288x。
TE+NCCL 吞吐相对前向的中位数为 75.9%/80.9% (Eager/Graph)，相对经典 cuBLAS 纯 B+W 为 59.1%/66.6%。

deferred：fused 对 TE+NCCL 的 Eager 胜点 12/12，几何平均 1.565x；Graph 胜点 12/12，几何平均 1.267x。
TE+NCCL 吞吐相对前向的中位数为 74.1%/75.8% (Eager/Graph)，相对经典 cuBLAS 纯 B+W 为 57.7%/65.8%。
