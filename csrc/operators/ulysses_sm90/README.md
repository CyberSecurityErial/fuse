# SM90 Ulysses implementation layout

`csrc/operators/ulysses_sm90.cu` is only the assembly point. The implementation
stays in one CUDA translation unit so moving code between these files does not
duplicate CUTLASS kernels or change device code generation.

## Public operator hierarchy

- `operators/primitives/gemm_a2a.h` and `a2a_gemm.h` define the two physical
  dataflow primitives.
- `operators/ulysses/projection_dataflow.h` binds projection semantics and
  forward/backward direction to those primitives with templates.
- `operators/ulysses/qkv_backward.h` and `oproj_backward.h` own the independent
  weight-gradient phase and its immediate/deferred ZeroBubble contract.
- The former flat headers remain compatibility includes only.

## Private implementation

- `detail/core.cuh`: scalar helpers, kernel geometry, calibrated policy data,
  and CUTLASS type construction.
- `detail/a2a_gemm.cuh`: peer input staging and ready publication for
  A2A-then-GEMM.
- `detail/gemm_a2a.cuh`: Q/K/V routing and cross-rank completion for
  GEMM-then-A2A.
- `detail/backward.cuh`: device-side backward routing.
- `detail/launch.cuh`: shared launch helpers, policy-to-kernel bindings, traits,
  and bounded launch-plan caches.

## Public API implementation

- `api/forward.cuh`: production forward launch entry points.
- `api/backward.cuh`: dgrad, wgrad, and deferred-wgrad entry points.
- `api/policy.cuh`: automatic tile and communication-CTA selection.
- `api/heterogeneous.cuh`: weighted CP planning and launch.
- `api/reference.cuh`: independent GEMM and communication reference paths.

## Rules for new code

1. Add a policy-to-kernel mapping once in `detail/launch.cuh`; production,
   profiling, traits, and reference paths should visit that same binding.
2. Use `BoundedLaunchCache` for immutable launch plans. Do not add another
   hand-written thread-local/global cache pair.
3. Keep pointer values, epochs, alpha/beta, and streams out of cached plans.
4. Device communication belongs in the matching dataflow file; validation and
   launch glue belongs in `api/`.
5. A change that alters kernel parameters or a CUTLASS type needs correctness
   and performance validation. A file move alone must preserve generated SASS.
