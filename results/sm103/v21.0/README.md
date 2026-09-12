# v21.0 — SM103 MXFP8 OProj manual SOTA

Final results only: 36 physical points / 42 model rows, CP4/8 × 128K/256K/512K.
All pass Graph10+50, reproducible nonzero random two-payload numerical checks,
and complete FP8/SFA byte-routing validation. Model aliases share measurements.

- Fusion: prequantized MXFP8 activation → A2A + BF16 W quantization + MXFP8 GEMM → BF16 output; FP32 accumulation. Upstream activation quantization is excluded.
- Optimizations: separate copy/quantization warps; FP8 cp.async→TMA and grouped scale loads; shared bounded M/N traversal; measured communication budgets.
- Final configuration: M128N256K128/E32/AlongN, H64/P4, per-point swizzle and comm20/32/48. The table is the current measured manual SOTA, not a global-optimum claim.
- Geometric means: fusion **1.914693 PFLOPS/GPU**, same-budget independent GEMM **2.174387**, full-device pure cuBLASLt **2.716657**. Ratios: **88.06% / 70.48%**. Compared with the historical table, +20.97%; this is not an isolated same-binary window ablation.
- Defaults remain H/P=0/0. Current OProj Auto lacks matching producer-revision calibration and returns NotSupported for comm=0. Use the explicit released configurations.

[Table](table.md) · [CSV](table.csv) · [Exact configurations and evidence](results.json)

Build the current checkout with `python3 scripts/l20d.py run fused-build --node 09 --workspace /home/work/workspace_wct --mpi --mxfp8`, then use each row's `replay` command in results.json. Commands use this checkout, not private frozen-source identifiers. GPU availability checks remain enabled.

The JSON records the tested source, binary/environment hashes and each selected
run/candidate/artifact hash. Raw evidence stays in the original validated L20D
artifacts (local `fuse_midfile/l20d/<run_id>/artifacts-attempt1`); no profiles,
raw logs, tuning candidates or dependency packages are included in this release.
The released native implementation matches the tested build; the only later
native-harness change adds a help sentence. Existing SM90/BF16/QKV behavior is
retained, without claiming a new full GPU performance regression suite for them.
