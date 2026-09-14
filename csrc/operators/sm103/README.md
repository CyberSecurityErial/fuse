# SM103 operator backend

B300 uses the Blackwell CUTLASS Sm100 collective compiled for sm_103a.
That collective name does not mean Hopper WGMMA; it uses tcgen05 MMA and TMEM.
The backend supports the released BF16 and MXFP8 projection paths. See the
public headers for exact precision-specific interfaces and supported policies,
and [v24 results](../../../results/sm103/v24.0/README.md) for measured coverage.
A declaration, host test or successful build alone does not prove GPU support.

## Module ownership

SM90 and SM103 keep the same architecture-level organization. Precision is
explicit in parameter/entry names, not a collection of IsFp8/IsMxfp8 switches.
There are no empty files for unimplemented architecture features.

| Location | Responsibility |
|---|---|
| entry.cu | Assemble the selected architecture's single CUDA translation unit |
| api/forward.cuh, api/forward_mxfp8.cuh | Validate and launch precision-specific forward operators |
| api/backward.cuh, api/backward_mxfp8.cuh | Data/weight gradient entry points and workspace/lifetime validation |
| api/policy.cuh, api/reference.cuh | Supported policy/resource queries and diagnostic boundaries |
| detail/gemm.cuh | CUTLASS families, geometry and precision-specific GEMM types |
| detail/cutlass_pipeline.cuh | Input-ready acquire and epilogue publication adapters |
| detail/persistent_gemm.cuh | Persistent scheduler, CTA roles and cooperative residency |
| detail/a2a_gemm.cuh, detail/gemm_a2a.cuh, detail/backward.cuh | Transport, routing, publication and completion |
| detail/producer_consumer.cuh | Shared GEMM/copy traversal and dependency mapping |
| detail/quantization.cuh | MXFP8 K32 quantization, packing and preparation |
| detail/launch.cuh | Concrete bindings and launch assembly |
| detail/autotune.cuh, detail/performance_model.cuh, detail/model_calibration.cuh | Selection logic, model and compiled calibration data |
| detail/attention_postprocess.cuh | Default-off experimental norm/RoPE/postnorm |
| include/fuse/profiling/sm103/ | Host, OProj and epilogue profiling support, outside operator detail |

Control templates use fuse::detail; private operator helpers use
namespace fuse { namespace { ... } }. The T suffix consistently denotes a
template, with ordinary aliases where needed. Local SM100_* copy aliases bind
CUTLASS's compatible SM90_* bulk/TMA instructions; they do not change operations.
Common public headers stay architecture-neutral and CMake selects one backend.
SM90 is not modified to support these SM103 changes.

## Execution and selection

Persistent mixed kernels reserve communication CTAs and give the remaining
resident budget to GEMM. The actual compiled resource/occupancy checks must
admit the cooperative grid. Do not substitute the full-device SM count when
comparing compute under a reduced budget.

Tile, Along direction, swizzle, epilogue and communication budget are coupled
through scheduling, not through model-name branches. Public policy APIs and
validation define which explicit combinations or automatic modes are supported.
Do not infer that zero means Auto for every precision/direction: in particular
the current MXFP8 backward requires a positive explicit communication budget.
Published manual configurations are finite-search results, not universal optima.

The selected GEMM traversal drives communication windows and dependency
ownership. PublishedTile represents a whole output tile, not an MMA or epilogue
subtile. Copy rectangles are assigned at their latest logical producer dependency:

    GEMM tile order (tile / Along / swizzle / compute budget)
           -> ready dependency frontier -> independent copy-worker queues

A later logical tile may finish earlier under concurrency. Every copy still
acquires ALL intersecting flags; ownership order is not proof of completion.
No global completion sort or new synchronization granularity is implied.
Preserve async-proxy ordering, stage-read completion, destination-write drain
and final cross-rank completion.

BF16 OProj's optional rows/columns transport layout changes copy rectangles,
not full-peer ready granularity. Keep its layout/arrival count fixed across
epochs; unsupported tails are rejected. The optional rank-dependent QKV
N-band swizzle stays OFF by default; its mixed historical benefits do not
justify making it unconditional. See [v15](../../../results/sm103/v15.0/README.md).

The vector fallback's kA2ALhsCommRows controls arrivals; kQkvBulkRows controls
the TMA route stage/task grid. They are active transport parameters, not dead
GEMM tile constants. Model calibration tables and reference entry points are
also active consumers, not trial files to delete.

## Tensor, epoch and lifetime rules

- rhs_nt is physical row-major [N,K]. Batch is flattened into M; independent
  batched GEMMs are not silently substituted. Use the declared aligned row
  strides, addresses and peer mappings.
- QKV emits three planar Q/K/V tensors; the route is not row-interleaved QKV.
  Standard and causal OProj mappings retain their public contract. Special
  KDA/MLA/KV-replication routes require explicit implementation.
- Query ready storage for the actual selected geometry. QKV publishes output
  tile epochs after destination-store completion. A2A uses cumulative arrivals
  for complete [M tile, peer-K] units. A rectangle-layout change cannot silently
  change the expected arrivals while reusing old counters.
- Start zeroed flags at epoch1 and advance consistently across ranks. Do not
  overlap operations sharing staging/control/output. Before overflow or a
  geometry/arrival-layout change, finish all ranks, reset and restart.
- Every Graph replay must advance the native epoch; repeatedly replaying a
  fixed epoch is unsafe. The benchmark Graph machinery checks and updates the
  appropriate kernel nodes; it does not weaken the kernel protocol.
- Enqueue all participating ranks before waiting on one. OProj pull completion
  alone does not release other ranks' peer-input leases. Preserve immutable
  source data until ALL readers finish.
- Deferred V and deferred weight gradients have separate lifetime/completion
  rules in their public headers; do not interpret a local completion marker
  as all remote tensors being available.
- MXFP8 data and scale layout/readiness travel together. A transpose requires
  new K32 quantization from the original BF16 orientation, not reused scales.
  See [MXFP8 contracts](../../../benchmarks/sm103/MXFP8_DUAL_FUSION.md).

## Profiling and verification

Follow [PROFILE_PROTOCOL](../../../PROFILE_PROTOCOL.md) and the existing
include/fuse/profiling components. Production and instrumented specializations
are separate; instrumentation timing is not a formal benchmark.

Clear diagnostic arrays each diagnostic launch without resetting native epochs.
Publication records use [M,peer], while GEMM tile records use [L,M,N]; never zip
them by array index. The release timestamp is sampled after publication, so
a consumer may already have observed the flag. Compare one GPU's own timebase,
not absolute timestamps across ranks. Role spans include waits and are not
pure MMA instruction time. Legacy physical warp counts are not active-warp
measurements.

Host tests check geometry, policy, mappings and exporter ownership. Independent
GPU numeric/route tests, CPU-oracle cross-checks and formal Graph10+50 evidence
remain distinct. Retain original tolerances, full-boundary work and explicit
missing/unsupported cases. Historical development traces and rejected trials
remain recoverable in Git and hashed artifacts instead of this module guide.
