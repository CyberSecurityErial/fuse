# v22.0 — Final augmented-forward results

PFLOPS/GPU = original GEMM FLOPs / complete augmented boundary. Graph10+50, maximum-rank latency. Explicit offline configurations; not Auto or global-optimality claims.

| Model | Operation | CP | S | PFLOPS/GPU | p50 ms | Comm CTA | Along/sw | H/P |
|---|---|---:|---:|---:|---:|---:|---|---|
| llama31_405b | a2a_oproj_residual_rmsnorm | 4 | 128K | 1.876 | 9.3775 | 48 | n/8 | 64/4 |
| llama31_405b | a2a_oproj_residual_rmsnorm | 8 | 128K | 1.823 | 4.8250 | 40 | n/8 | 64/4 |
| llama31_405b | a2a_oproj_residual_rmsnorm | 8 | 256K | 1.845 | 9.5350 | 40 | n/8 | 64/4 |
| llama31_405b | a2a_oproj_residual_rmsnorm | 8 | 512K | 1.851 | 19.0092 | 48 | n/8 | 64/4 |
| llama31_70b | a2a_oproj_residual_rmsnorm | 4 | 128K | 1.788 | 2.4595 | 48 | n/8 | 64/4 |
| llama31_70b | a2a_oproj_residual_rmsnorm | 4 | 256K | 1.771 | 4.9658 | 32 | n/8 | 64/4 |
| llama31_70b | a2a_oproj_residual_rmsnorm | 4 | 512K | 1.789 | 9.8335 | 48 | n/8 | 64/4 |
| llama31_70b | a2a_oproj_residual_rmsnorm | 8 | 128K | 1.681 | 1.3079 | 40 | m/8 | 64/8 |
| llama31_70b | a2a_oproj_residual_rmsnorm | 8 | 256K | 1.761 | 2.4977 | 40 | m/8 | 64/8 |
| llama31_70b | a2a_oproj_residual_rmsnorm | 8 | 512K | 1.752 | 5.0198 | 40 | m/8 | 64/8 |
| qwen3_235b | a2a_oproj_residual_rmsnorm | 4 | 128K | 1.711 | 1.2856 | 48 | n/2 | 64/4 |
| qwen3_235b | a2a_oproj_residual_rmsnorm | 4 | 256K | 1.730 | 2.5416 | 40 | n/2 | 64/4 |
| qwen3_235b | a2a_oproj_residual_rmsnorm | 4 | 512K | 1.687 | 5.2148 | 40 | n/2 | 64/4 |
| qwen3_235b | a2a_oproj_residual_rmsnorm | 8 | 128K | 1.540 | 0.7140 | 40 | n/2 | 64/4 |
| qwen3_235b | a2a_oproj_residual_rmsnorm | 8 | 256K | 1.640 | 1.3408 | 48 | n/2 | 64/4 |
| qwen3_235b | a2a_oproj_residual_rmsnorm | 8 | 512K | 1.679 | 2.6191 | 40 | n/2 | 64/4 |
| llama31_405b | qkv_rope_a2a | 4 | 128K | 1.981 | 9.9909 | 24 | m/8 | 0/0 |
| llama31_405b | qkv_rope_a2a | 4 | 256K | 2.025 | 19.5515 | 24 | m/8 | 0/0 |
| llama31_405b | qkv_rope_a2a | 4 | 512K | 2.039 | 38.8290 | 24 | m/8 | 0/0 |
| llama31_405b | qkv_rope_a2a | 8 | 128K | 1.874 | 5.2814 | 40 | m/8 | 0/0 |
| llama31_405b | qkv_rope_a2a | 8 | 256K | 1.860 | 10.6396 | 48 | m/8 | 0/0 |
| llama31_405b | qkv_rope_a2a | 8 | 512K | 1.998 | 19.8078 | 40 | m/8 | 0/0 |
| llama31_70b | qkv_rope_a2a | 4 | 128K | 2.027 | 2.7124 | 40 | m/8 | 0/0 |
| llama31_70b | qkv_rope_a2a | 4 | 256K | 2.015 | 5.4572 | 40 | m/8 | 0/0 |
| llama31_70b | qkv_rope_a2a | 4 | 512K | 1.996 | 11.0187 | 40 | m/8 | 0/0 |
| llama31_70b | qkv_rope_a2a | 8 | 128K | 1.801 | 1.5264 | 40 | m/8 | 0/0 |
| llama31_70b | qkv_rope_a2a | 8 | 256K | 1.858 | 2.9586 | 48 | m/8 | 0/0 |
| llama31_70b | qkv_rope_a2a | 8 | 512K | 1.983 | 5.5438 | 40 | m/8 | 0/0 |
| qwen3_235b | qkv_qknorm_rope_a2a | 4 | 128K | 1.718 | 1.4401 | 44 | n/2 | 0/0 |
| qwen3_235b | qkv_qknorm_rope_a2a | 4 | 256K | 1.816 | 2.7248 | 44 | n/2 | 0/0 |
| qwen3_235b | qkv_qknorm_rope_a2a | 4 | 512K | 1.825 | 5.4230 | 44 | n/2 | 0/0 |
| llama31_405b | a2a_oproj_residual_rmsnorm | 4 | 256K | — | — | — | numerical failure | — |
| llama31_405b | a2a_oproj_residual_rmsnorm | 4 | 512K | — | — | — | numerical failure | — |

Direction geometric means: QKV **1.918331P** (15/15), OProj **1.743187P** (16/18 accepted subset).

QKV operation names distinguish Qwen3 Q/K RMSNorm+RoPE from Llama RoPE-only. OProj has residual add and full-hidden RMSNorm, never RoPE. Missing numerical points are excluded, not zero-filled.

[Configurations, samples, replay commands and evidence](results.json).
