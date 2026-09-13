# v23.0 — MXFP8 A2A + OProj

36/36 physical points; Graph10+50. Geometric mean **2.256462 PFLOPS/GPU**. Explicit independently confirmed configurations. Pure cuBLASLt and v21 are historical references, not newly paired measurements.

Llama70B/Qwen72B share one physical geometry. Kimi denotes the supported projection geometry, not a claim of complete special KDA routing. No norm/RoPE or upstream activation quantization is timed.

| Model / geometry | CP | Sequence | Fused P | Same-budget CUTLASS P | Full-SM cuBLASLt (historical) P | Fused / cuBLASLt | Gain vs v21 (historical) | C / H / P / swizzle |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| bloom_176b | 4 | 128K | 2.275 | 2.436 | 2.726 | 83.4% | +16.92% | 4 / 0 / 0 / 8 |
| bloom_176b | 4 | 256K | 2.342 | 2.464 | 2.708 | 86.5% | +18.24% | 8 / 0 / 0 / 8 |
| bloom_176b | 4 | 512K | 2.355 | 2.465 | 2.700 | 87.2% | +17.71% | 8 / 0 / 0 / 8 |
| bloom_176b | 8 | 128K | 2.263 | 2.446 | 2.691 | 84.1% | +20.49% | 4 / 0 / 0 / 8 |
| bloom_176b | 8 | 256K | 2.239 | 2.434 | 2.726 | 82.1% | +18.13% | 4 / 0 / 0 / 8 |
| bloom_176b | 8 | 512K | 2.297 | 2.467 | 2.708 | 84.8% | +16.91% | 8 / 0 / 0 / 8 |
| llama31_405b | 4 | 128K | 2.273 | 2.459 | 2.697 | 84.3% | +15.45% | 8 / 0 / 0 / 8 |
| llama31_405b | 4 | 256K | 2.341 | 2.475 | 2.697 | 86.8% | +17.72% | 8 / 0 / 0 / 8 |
| llama31_405b | 4 | 512K | 2.366 | 2.457 | 2.691 | 87.9% | +18.47% | 8 / 0 / 0 / 8 |
| llama31_405b | 8 | 128K | 2.247 | 2.458 | 2.723 | 82.5% | +18.05% | 4 / 0 / 0 / 8 |
| llama31_405b | 8 | 256K | 2.267 | 2.454 | 2.697 | 84.1% | +18.53% | 4 / 0 / 0 / 8 |
| llama31_405b | 8 | 512K | 2.311 | 2.481 | 2.697 | 85.7% | +17.64% | 8 / 0 / 0 / 8 |
| qwen3_235b | 4 | 128K | 2.185 | 2.535 | 2.810 | 77.7% | +17.03% | 16 / 0 / 0 / 2 |
| qwen3_235b | 4 | 256K | 2.316 | 2.523 | 2.724 | 85.0% | +20.43% | 16 / 0 / 0 / 8 |
| qwen3_235b | 4 | 512K | 2.308 | 2.564 | 2.694 | 85.7% | +18.95% | 16 / 0 / 0 / 8 |
| qwen3_235b | 8 | 128K | 2.117 | 2.565 | 2.612 | 81.1% | +31.18% | 16 / 0 / 0 / 8 |
| qwen3_235b | 8 | 256K | 2.150 | 2.460 | 2.810 | 76.5% | +14.81% | 16 / 0 / 0 / 2 |
| qwen3_235b | 8 | 512K | 2.263 | 2.520 | 2.724 | 83.1% | +20.55% | 16 / 0 / 0 / 8 |
| representative_large | 4 | 128K | 2.280 | 2.405 | 2.672 | 85.3% | +16.87% | 12 / 64 / 16 / 4 |
| representative_large | 4 | 256K | 2.232 | 2.337 | 2.674 | 83.4% | +14.43% | 12 / 64 / 16 / 4 |
| representative_large | 4 | 512K | 2.282 | 2.334 | 2.671 | 85.4% | +15.07% | 12 / 0 / 0 / 4 |
| representative_large | 8 | 128K | 2.181 | 2.383 | 2.672 | 81.6% | +19.57% | 12 / 64 / 16 / 4 |
| representative_large | 8 | 256K | 2.262 | 2.400 | 2.672 | 84.7% | +20.18% | 12 / 64 / 16 / 4 |
| representative_large | 8 | 512K | 2.219 | 2.351 | 2.674 | 83.0% | +18.41% | 12 / 64 / 16 / 4 |
| kimi_k3_kda | 4 | 128K | 2.243 | 2.414 | 2.747 | 81.7% | +12.78% | 20 / 64 / 16 / 4 |
| kimi_k3_kda | 4 | 256K | 2.253 | 2.402 | 2.734 | 82.4% | +16.60% | 12 / 0 / 0 / 4 |
| kimi_k3_kda | 4 | 512K | 2.302 | 2.423 | 2.713 | 84.9% | +15.97% | 12 / 0 / 0 / 4 |
| kimi_k3_kda | 8 | 128K | 2.134 | 2.410 | 2.750 | 77.6% | +17.88% | 20 / 64 / 16 / 4 |
| kimi_k3_kda | 8 | 256K | 2.212 | 2.359 | 2.747 | 80.5% | +17.74% | 12 / 0 / 0 / 4 |
| kimi_k3_kda | 8 | 512K | 2.163 | 2.408 | 2.734 | 79.1% | +14.77% | 12 / 64 / 16 / 4 |
| llama31_70b / qwen25_72b | 4 | 128K | 2.307 | 2.491 | 2.777 | 83.1% | +17.93% | 16 / 64 / 16 / 8 |
| llama31_70b / qwen25_72b | 4 | 256K | 2.348 | 2.519 | 2.728 | 86.1% | +17.52% | 8 / 64 / 16 / 8 |
| llama31_70b / qwen25_72b | 4 | 512K | 2.303 | 2.456 | 2.715 | 84.8% | +14.82% | 8 / 64 / 16 / 8 |
| llama31_70b / qwen25_72b | 8 | 128K | 2.118 | 2.456 | 2.789 | 76.0% | +18.03% | 20 / 64 / 16 / 8 |
| llama31_70b / qwen25_72b | 8 | 256K | 2.242 | 2.505 | 2.777 | 80.7% | +18.63% | 8 / 64 / 16 / 8 |
| llama31_70b / qwen25_72b | 8 | 512K | 2.271 | 2.458 | 2.728 | 83.2% | +19.37% | 8 / 0 / 0 / 8 |

All points: M128/N256/K128, epilogue N32, AlongN, weight_preparation=all. H=P=0 disables the optional window. Seven points remain below2.2P; the target is the physical-point geometric mean.

See [results.json](results.json) for exact configurations, timing, provenance and replay commands; [README](README.md) for boundaries and release validation.
