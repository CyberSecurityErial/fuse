# SM90 benchmark archive and active entry points

The original benchmark sources now live under this directory. Directory names,
operator contracts and historical golden data are preserved. The legacy paths
`benchmarks/GEMM`, `benchmarks/QKVproj+a2a`, `benchmarks/a2a+Oproj`, etc. resolve
here through compatibility symlinks. Python repository-root discovery accounts
for the extra architecture directory.

- `GEMM/`: cuBLASLt standalone GEMM.
- `QKVproj+a2a/`: QKV projection then complete Q/K/V all-to-all.
- `a2a+Oproj/`: inverse all-to-all then output projection.
- `backward/`, `QKVproj-backward/`, `Oproj-backward/`: backward tests and reports.
- `fp8/`: v12 E4M3 four-boundary benchmarks.
- `heterogeneous_cp/`, `e2e/`: weighted CP and end-to-end reports.

Build from the repository root with `-DFUSE_ARCH=sm90` (the compatibility default).
Use a separate build tree, e.g. `build/sm90`; no SM103 kernel is substituted.
