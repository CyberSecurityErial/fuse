# Architecture-specific benchmarks

| Family | Canonical directory | CUDA target | Scope |
|---|---|---|---|
| Hopper | [sm90](sm90/README.md) | sm_90a | Historical fusion kernels, baselines and golden contracts |
| Blackwell Ultra | [sm103](sm103/README.md) | sm_103a | Training GEMM / QKV→A2A / A2A→OProj strong baselines |

Both families organize benchmarks by `GEMM`, `QKVproj+a2a`, `a2a+Oproj`.
Top-level legacy benchmark directories are compatibility symlinks to `sm90/`;
`sm103a/` is a compatibility symlink to `sm103/`. New code should use canonical paths.
Historical SM90 results are not moved or reused as SM103 tuning results.
Root CMake selects the family with `-DFUSE_ARCH=sm90|sm103`; build directories must differ.
