# SM103 QKV projection + A2A

训练前向边界：先做本 rank 的 QKV projection GEMM，再按 causal Ulysses
合同执行 A2A。`run_bench.py` 固定方向并调用 SM103 分阶段计划器；后端包括
精调 cuBLASLt + NCCL 和适配后的 TransformerEngine Userbuffers。

```bash
/root/workspace_wct/bench-env/bin/python \
  benchmarks/sm103/QKVproj+a2a/run_bench.py --stage smoke --execute \
  --python /root/workspace_wct/bench-env/bin/python --devices 0,1
```

FP4 只进入独立 cuBLASLt GEMM 基线；TE Userbuffers 没有 FP4 A2A 合同，因此
不会伪造 FP4 融合边界结果。
