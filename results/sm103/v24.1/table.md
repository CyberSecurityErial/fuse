# v24.1 — QKV all-mode quantization

PFLOPS/GPU; dynamic W quantization + GEMM + A2A. Two actual Philox payloads, each Graph10+50, full pre/post numeric and route checks. Comparison to historical v24.0, not a same-round full A/B.

| Scope | Points | v24.0 | v24.1 | Change |
|---|---:|---:|---:|---:|
| All measured | 33 | 2.047223 | 2.077387 | +1.47% |
| 典型泛化场景 | 27 | 2.200574 | 2.227942 | +1.24% |

QwenDense remains supplementary and visible. Unchanged OProj and both backward tables: [v24.0](../v24.0/table.md).

| Model | CP | Sequence | v24.0 | v24.1 | Change |
|---|---:|---:|---:|---:|---:|
| QwenDense | 4 | 128K | 1.5329 | 1.5924 | +3.88% |
| QwenDense | 4 | 256K | 1.6801 | 1.7110 | +1.84% |
| QwenDense | 4 | 512K | 1.7080 | 1.7071 | -0.05% |
| QwenDense | 8 | 128K | 1.1854 | 1.2384 | +4.47% |
| QwenDense | 8 | 256K | 1.3607 | 1.4048 | +3.24% |
| QwenDense | 8 | 512K | 1.4757 | 1.5018 | +1.77% |
| Qwen3 235B | 4 | 128K | 2.0025 | 2.0353 | +1.64% |
| Qwen3 235B | 4 | 256K | 2.0469 | 2.1086 | +3.02% |
| Qwen3 235B | 4 | 512K | 2.0696 | 2.0966 | +1.30% |
| Qwen2.5 72B / Llama 3.1 70B | 4 | 128K | 2.2041 | 2.2587 | +2.47% |
| Qwen2.5 72B / Llama 3.1 70B | 4 | 256K | 2.1545 | 2.1854 | +1.43% |
| Qwen2.5 72B / Llama 3.1 70B | 4 | 512K | 2.1499 | 2.1549 | +0.23% |
| Qwen2.5 72B / Llama 3.1 70B | 8 | 128K | 2.0655 | 2.1592 | +4.54% |
| Qwen2.5 72B / Llama 3.1 70B | 8 | 256K | 2.1550 | 2.1937 | +1.80% |
| Qwen2.5 72B / Llama 3.1 70B | 8 | 512K | 2.2384 | 2.2569 | +0.83% |
| Llama 3.1 405B | 4 | 128K | 2.2909 | 2.3179 | +1.18% |
| Llama 3.1 405B | 4 | 256K | 2.3394 | 2.3540 | +0.63% |
| Llama 3.1 405B | 4 | 512K | 2.3793 | 2.3788 | -0.02% |
| Llama 3.1 405B | 8 | 128K | 2.2247 | 2.2794 | +2.46% |
| Llama 3.1 405B | 8 | 256K | 2.2491 | 2.2576 | +0.37% |
| Llama 3.1 405B | 8 | 512K | 2.3616 | 2.3742 | +0.53% |
| Kimi K3 QKV-only | 4 | 128K | 2.1271 | 2.1525 | +1.19% |
| Kimi K3 QKV-only | 4 | 256K | 2.1789 | 2.2021 | +1.07% |
| Kimi K3 QKV-only | 4 | 512K | 2.2117 | 2.2260 | +0.65% |
| Kimi K3 QKV-only | 8 | 128K | 2.0498 | 2.1312 | +3.97% |
| Kimi K3 QKV-only | 8 | 256K | 2.0886 | 2.1074 | +0.90% |
| Kimi K3 QKV-only | 8 | 512K | 2.1333 | 2.1450 | +0.55% |
| BLOOM 176B | 4 | 128K | 2.2993 | 2.3218 | +0.98% |
| BLOOM 176B | 4 | 256K | 2.3269 | 2.3344 | +0.32% |
| BLOOM 176B | 4 | 512K | 2.3747 | 2.3803 | +0.24% |
| BLOOM 176B | 8 | 128K | 2.1812 | 2.1918 | +0.48% |
| BLOOM 176B | 8 | 256K | 2.2849 | 2.2992 | +0.62% |
| BLOOM 176B | 8 | 512K | 2.2988 | 2.3073 | +0.37% |

[Exact parameters, replay arguments and measured source/binary identifiers](results.json). No per-point old/new cherry-picking.
