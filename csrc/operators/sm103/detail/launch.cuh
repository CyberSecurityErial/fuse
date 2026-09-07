// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "host_profiling.cuh"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace fuse {
namespace {

template <class T>
struct TypeTag {
  using type = T;
};

// A2A readiness is indexed by M tile and peer, independent of the output N
// width. K policies share the copy algorithm but validate their own peer-K
// divisibility, so adding K128 never narrows the existing K64 input contract.
template <int BlockN, int BlockK = 64, int EpilogueN = 0>
struct A2ALhsKernelBinding {
  using Types = A2ALhsGemmTypes<BlockN, BlockK, EpilogueN>;
  using Gemm = typename Types::Gemm;
  using PureGemm = typename Types::PureGemm;
  using TileShape = typename Gemm::TileShape;
  using Comm = A2ALhsInputCommT<cute::size<0>(TileShape{}), cute::size<2>(TileShape{})>;
  static_assert(cute::size<0>(TileShape{}) == Comm::kReadyBlockM);
  static_assert(cute::size<2>(TileShape{}) == Comm::kTileK);
  using Kernel = detail::MonolithicGemm<Gemm, Comm>;
#if FUSE_ENABLE_PROFILING
  using TelemetryGemm = typename Types::TelemetryGemm;
  using TelemetryComm = A2ALhsInputCommT<
      cute::size<0>(TileShape{}), cute::size<2>(TileShape{}), true>;
  using TelemetryKernel = detail::RoleTelemetryKernel<
      detail::MonolithicGemm<TelemetryGemm, TelemetryComm>>;
#endif
};

using OprojForwardN128Binding = A2ALhsKernelBinding<128>;
using OprojForwardN256Binding = A2ALhsKernelBinding<256>;
using OprojForwardN128K128Binding = A2ALhsKernelBinding<128, 128>;
using OprojForwardN256K64E32Binding = A2ALhsKernelBinding<256, 64, 32>;
using OprojForwardN256K128E32Binding = A2ALhsKernelBinding<256, 128, 32>;

enum class OprojGemmPolicy {
  kM128N128,
  kM128N256,
  kM128N128K128,
  kM128N256K64E32,
  kM128N256K128E32,
};

inline cudaError_t select_oproj_gemm_policy(OprojGemmPolicy* policy) {
  // Private one-SM policies must not reuse Hopper's ClusterM2 enum values.
  // TODO: Select tile and communication CTAs jointly with independently
  // calibrated B300 performance costs. Auto stays N128 until that model is
  // validated; explicit communication CTAs are never changed here.
  const char* value = std::getenv("FUSE_SM103_OPROJ_POLICY");
  if (!value || std::strcmp(value, "auto") == 0 ||
      std::strcmp(value, "m128n128") == 0) {
    *policy = OprojGemmPolicy::kM128N128;
  } else if (std::strcmp(value, "m128n256") == 0) {
    *policy = OprojGemmPolicy::kM128N256;
  } else if (std::strcmp(value, "m128n128k128") == 0) {
    *policy = OprojGemmPolicy::kM128N128K128;
  } else if (std::strcmp(value, "m128n256k64e32") == 0) {
    *policy = OprojGemmPolicy::kM128N256K64E32;
  } else if (std::strcmp(value, "m128n256k128e32") == 0) {
    *policy = OprojGemmPolicy::kM128N256K128E32;
  } else {
    return cudaErrorNotSupported;
  }
  return cudaSuccess;
}

template <class Visitor>
cudaError_t visit_oproj_forward_policy(OprojGemmPolicy policy, Visitor& visitor) {
  switch (policy) {
    case OprojGemmPolicy::kM128N128:
      return visitor(TypeTag<OprojForwardN128Binding>{});
    case OprojGemmPolicy::kM128N256:
      return visitor(TypeTag<OprojForwardN256Binding>{});
    case OprojGemmPolicy::kM128N128K128:
      return visitor(TypeTag<OprojForwardN128K128Binding>{});
    case OprojGemmPolicy::kM128N256K64E32:
      return visitor(TypeTag<OprojForwardN256K64E32Binding>{});
    case OprojGemmPolicy::kM128N256K128E32:
      return visitor(TypeTag<OprojForwardN256K128E32Binding>{});
    default:
      return cudaErrorInvalidValue;
  }
}

// One pairing is used by production, telemetry and selected-geometry queries.
// The geometry is the actual producer-ready tile, not the MMA atom or cluster.
template <class GemmTypes, class CommType>
struct GemmA2AKernelBinding {
  using Gemm = typename GemmTypes::OutputGemm;
  using Comm = CommType;
  using TileShape = typename Gemm::TileShape;
  // Fused and pure compute must share K, epilogue and stage choices, not
  // reconstruct a default collective from only the producer's N width.
  using PureGemm = typename GemmTypes::PureGemm;
  static_assert(cute::size<0>(TileShape{}) == Comm::kBlockM);
  static_assert(cute::size<1>(TileShape{}) == Comm::kBlockN);
  using Kernel = detail::MonolithicGemm<Gemm, Comm>;
#if FUSE_ENABLE_PROFILING
  using TelemetryKernel = GemmA2ARoleTelemetryKernel<Gemm, Comm>;
#endif
};

using QkvForwardN64Binding = GemmA2AKernelBinding<Bf16GemmTypes<64>, QkvGqaPackCommN64>;
using QkvForwardN128Binding = GemmA2AKernelBinding<Bf16GemmTypes<128>, QkvGqaPackComm>;
using QkvForwardN128InterleavedBinding =
    GemmA2AKernelBinding<Bf16GemmTypes<128>, QkvGqaPackCommSmallInterleaved>;
using QkvForwardN160Binding = GemmA2AKernelBinding<Bf16GemmTypes<160>, QkvGqaPackCommN160>;
using QkvForwardN192Binding = GemmA2AKernelBinding<Bf16GemmTypes<192>, QkvGqaPackCommN192>;
using QkvForwardN256Binding =
    GemmA2AKernelBinding<Bf16GemmTypes<256>, QkvGqaPackCommWide>;
using QkvForwardN128K128Binding =
    GemmA2AKernelBinding<Bf16GemmTypes<128, 128>, QkvGqaPackComm>;
using QkvForwardN256K64E32Binding =
    GemmA2AKernelBinding<Bf16GemmTypes<256, 64, 32>, QkvGqaPackCommWide>;
using QkvForwardN256K128E32Binding =
    GemmA2AKernelBinding<Bf16GemmTypes<256, 128, 32>, QkvGqaPackCommWide>;
using QkvForwardN256K64E64Binding =
    GemmA2AKernelBinding<Bf16GemmTypes<256, 64, 64>, QkvGqaPackCommWide>;
using GemmA2AKernel = typename QkvForwardN128Binding::Kernel;

enum class QkvGemmPolicy {
  kM128N64,
  kM128N128,
  kM128N160,
  kM128N192,
  kM128N256,
  kM128N128K128,
  kM128N256K64E32,
  kM128N256K128E32,
  kM128N256K64E64,
};

inline cudaError_t select_qkv_gemm_policy(QkvGemmPolicy* policy) {
  // Keep the Hopper override spelling; no H200 cost model is imported.
  // TODO: Implement performance-model-based joint tile/communication-SM tuning
  // with independently measured B300 compute/route costs. This is a required
  // library capability, deferred for the first BF16 version, not removed.
  // Auto currently means fixed N128 and communication SMs remain explicit.
  const char* value = std::getenv("FUSE_QKV_GEMM_POLICY");
  if (!value || std::strcmp(value, "auto") == 0 ||
      std::strcmp(value, "m128n128") == 0) {
    *policy = QkvGemmPolicy::kM128N128;
  } else if (std::strcmp(value, "m128n64") == 0) {
    *policy = QkvGemmPolicy::kM128N64;
  } else if (std::strcmp(value, "m128n160") == 0) {
    *policy = QkvGemmPolicy::kM128N160;
  } else if (std::strcmp(value, "m128n192") == 0) {
    *policy = QkvGemmPolicy::kM128N192;
  } else if (std::strcmp(value, "m128n256") == 0) {
    *policy = QkvGemmPolicy::kM128N256;
  } else if (std::strcmp(value, "m128n128k128") == 0) {
    *policy = QkvGemmPolicy::kM128N128K128;
  } else if (std::strcmp(value, "m128n256k64e32") == 0) {
    *policy = QkvGemmPolicy::kM128N256K64E32;
  } else if (std::strcmp(value, "m128n256k128e32") == 0) {
    *policy = QkvGemmPolicy::kM128N256K128E32;
  } else if (std::strcmp(value, "m128n256k64e64") == 0) {
    *policy = QkvGemmPolicy::kM128N256K64E64;
  } else {
    return cudaErrorNotSupported;
  }
  return cudaSuccess;
}

template <class Visitor>
cudaError_t visit_qkv_forward_policy(
    QkvGemmPolicy policy, bool peer_interleaved, Visitor& visitor) {
  // Match Hopper's existing interleaved specialization, which uses N128.
  if (peer_interleaved && policy != QkvGemmPolicy::kM128N128) {
    return cudaErrorNotSupported;
  }
  switch (policy) {
    case QkvGemmPolicy::kM128N64:
      return visitor(TypeTag<QkvForwardN64Binding>{});
    case QkvGemmPolicy::kM128N128:
      return peer_interleaved
          ? visitor(TypeTag<QkvForwardN128InterleavedBinding>{})
          : visitor(TypeTag<QkvForwardN128Binding>{});
    case QkvGemmPolicy::kM128N160:
      return visitor(TypeTag<QkvForwardN160Binding>{});
    case QkvGemmPolicy::kM128N192:
      return visitor(TypeTag<QkvForwardN192Binding>{});
    case QkvGemmPolicy::kM128N256:
      return visitor(TypeTag<QkvForwardN256Binding>{});
    case QkvGemmPolicy::kM128N128K128:
      return visitor(TypeTag<QkvForwardN128K128Binding>{});
    case QkvGemmPolicy::kM128N256K64E32:
      return visitor(TypeTag<QkvForwardN256K64E32Binding>{});
    case QkvGemmPolicy::kM128N256K128E32:
      return visitor(TypeTag<QkvForwardN256K128E32Binding>{});
    case QkvGemmPolicy::kM128N256K64E64:
      return visitor(TypeTag<QkvForwardN256K64E64Binding>{});
    default:
      return cudaErrorInvalidValue;
  }
}

using DeviceInfo = detail::DeviceInfo;

#if FUSE_ENABLE_PROFILING
// This isolated diagnostic binding is not part of the production policy pool.
// Unlike ordinary CTA-only telemetry, it includes an ordered role-join
// reduction before the timestamp. Account for that extra diagnostic overhead.
using QkvEpilogueProbeKernel = GemmA2ARoleTelemetryKernel<
    QkvEpilogueProbeGemm, typename QkvForwardN256K64E32Binding::Comm, true>;
static_assert(QkvEpilogueProbeKernel::SharedStorageSize ==
              QkvForwardN256K64E32Binding::Kernel::SharedStorageSize);
static_assert(QkvEpilogueProbeKernel::MaxThreadsPerBlock ==
              QkvForwardN256K64E32Binding::Kernel::MaxThreadsPerBlock);
#endif

inline cudaError_t device_info(DeviceInfo* result) {
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return status;
  }
  // Only immutable device attributes are cached. No pointers, epochs,
  // streams, descriptors, or user launch arguments are cached.
  static thread_local std::vector<DeviceInfo> devices;
  for (const auto& info : devices) {
    if (info.device == device) {
      *result = info;
      return cudaSuccess;
    }
  }
  cudaDeviceProp properties{};
  status = cudaGetDeviceProperties(&properties, device);
  if (status != cudaSuccess) {
    return status;
  }
  // The binary contains architecture-specific sm_103a instructions. Use
  // CUDA Runtime's capability, not the marketing name reported by NVML.
  if (properties.major != 10 || properties.minor != 3 ||
      !properties.cooperativeLaunch) {
    return cudaErrorNotSupported;
  }
  *result = {device, properties.multiProcessorCount,
             static_cast<int>(properties.sharedMemPerBlockOptin),
             static_cast<int>(properties.sharedMemPerMultiprocessor)};
  devices.push_back(*result);
  return cudaSuccess;
}

template <class Kernel>
typename Kernel::Arguments gemm_arguments(
    const GemmProblem& problem, const Bf16* lhs, const Bf16* rhs_nt,
    Bf16* output, float alpha, int32_t num_comm_ctas,
    const DeviceInfo& info, GemmRaster fallback) {
  typename Kernel::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape = cute::make_shape(problem.m, problem.n, problem.k, 1);
  args.mainloop.ptr_A = lhs;
  args.mainloop.dA = cute::make_stride(a_row_stride(problem),
                                      cute::_1{}, int64_t{0});
  args.mainloop.ptr_B = rhs_nt;
  // rhs_nt is physically [N,K], CUTLASS B uses the logical (N,K,L) view.
  args.mainloop.dB = cute::make_stride(b_row_stride(problem),
                                      cute::_1{}, int64_t{0});
  args.epilogue.thread.alpha = alpha;
  args.epilogue.thread.beta = 0.0f;
  args.epilogue.ptr_C = nullptr;
  args.epilogue.dC = cute::make_stride(d_row_stride(problem),
                                      cute::_1{}, int64_t{0});
  args.epilogue.ptr_D = output;
  args.epilogue.dD = args.epilogue.dC;
  args.hw_info.device_id = info.device;
  args.hw_info.sm_count = info.sm_count - num_comm_ctas;
  args.scheduler.block_offset = num_comm_ctas;
  args.scheduler.max_swizzle_size = problem.max_swizzle_size;
  args.scheduler.raster_order = raster_option(problem.raster, fallback);
  return args;
}

template <class Kernel>
KernelTraits kernel_traits() {
  DeviceInfo info{};
  size_t smem = 0;
  // Report the real launch reservation, including occupancy padding. A zero
  // value means resource lookup failed/unavailable, not a zero-SMEM kernel.
  if (device_info(&info) != cudaSuccess ||
      detail::launch_shared_memory<Kernel>(info, &smem) != cudaSuccess) {
    smem = 0;
  }
  using ProducerTile = typename Kernel::TileShape;
  return {static_cast<int32_t>(cute::size<0>(ProducerTile{})),
          static_cast<int32_t>(cute::size<1>(ProducerTile{})),
          static_cast<int32_t>(cute::size<2>(ProducerTile{})),
          Kernel::MaxThreadsPerBlock,
          static_cast<int32_t>(smem)};
}

template <class Kernel>
cudaError_t launch_monolithic(
    const typename Kernel::Arguments& args,
    const DeviceInfo& info,
    cudaStream_t stream) {
  FUSE_SM103_HOST_MARK(kImplementWorkspace);
  if (!Kernel::can_implement(args) || Kernel::get_workspace_size(args) != 0) {
    return cudaErrorNotSupported;
  }
  if (Kernel::initialize_workspace(args, nullptr, stream) != cutlass::Status::kSuccess) {
    return cudaErrorInitializationError;
  }
#if FUSE_ENABLE_PROFILING
  FUSE_SM103_HOST_MARK(kLowerParameters);
  const auto params = Kernel::to_underlying_arguments(args, nullptr);
  FUSE_SM103_HOST_MARK(kLaunchSetup);
  return detail::launch_cooperative<Kernel>(params, info, stream);
#else
  return detail::launch_cooperative<Kernel>(
      Kernel::to_underlying_arguments(args, nullptr), info, stream);
#endif
}

template <class Gemm, class Kernel, class Comm, bool Instrumented = false>
cudaError_t launch_a2a_lhs_gemm_policy(
    const A2AGemmParams& params, cudaStream_t stream
#if FUSE_ENABLE_PROFILING
    , A2AGemmCtaTimeline* timeline = nullptr, int32_t timeline_capacity = 0,
    A2AGemmPeerTimeline* peer_timeline = nullptr, int32_t peer_timeline_capacity = 0
#endif
    ) {
  if (!supported_problem(params.gemm) || !std::isfinite(params.alpha) ||
      params.epoch == 0 || params.num_comm_ctas <= 0 ||
      params.lhs_policy != A2ALhsGemmPolicy::kAuto) {
    return cudaErrorInvalidValue;
  }
  DeviceInfo info{};
  cudaError_t status = device_info(&info);
  if (status != cudaSuccess) {
    return status;
  }
  if (params.num_comm_ctas >= info.sm_count) {
    return cudaErrorInvalidValue;
  }
  typename Comm::Arguments comm{};
  comm.params = params;
#if FUSE_ENABLE_PROFILING
  if constexpr (Instrumented) {
    if (!timeline || timeline_capacity < info.sm_count || peer_timeline_capacity < 0) {
      return cudaErrorInvalidValue;
    }
    comm.peer_timeline = peer_timeline;
    comm.peer_timeline_capacity = peer_timeline_capacity;
  }
#endif
  FUSE_SM103_HOST_MARK(kCommunicationPrepare);
  status = Comm::initialize(comm);
  if (status != cudaSuccess) {
    return status;
  }
  FUSE_SM103_HOST_MARK(kArguments);
  using ConsumerTile = typename Gemm::TileShape;
  constexpr int32_t tile_m = cute::size<0>(ConsumerTile{});
  constexpr int32_t tile_n = cute::size<1>(ConsumerTile{});
  constexpr int32_t tile_k = cute::size<2>(ConsumerTile{});
  static_assert(tile_m == Comm::kReadyBlockM);
  static_assert(tile_k == Comm::kTileK);
  const int32_t n_tiles = ceil_div(params.gemm.n, tile_n);
  comm.m_window = ceil_div(info.sm_count - params.num_comm_ctas, n_tiles);
  auto args = gemm_arguments<Gemm>(
      params.gemm, params.input_staging, params.rhs_nt, params.output,
      params.alpha, params.num_comm_ctas, info, GemmRaster::kAlongN);
  args.mainloop.ready = params.ready;
  args.mainloop.world_size = params.route.world_size;
  args.mainloop.m_tiles = ceil_div(params.gemm.m, tile_m);
  args.mainloop.arrivals_per_peer = Comm::arrivals_per_peer(comm);
  args.mainloop.k_tiles_per_peer = params.gemm.k / params.route.world_size / tile_k;
  args.mainloop.epoch = params.epoch;
#if FUSE_ENABLE_PROFILING
  if constexpr (Instrumented) {
    args.mainloop.timeline = timeline;
    args.mainloop.timeline_capacity = timeline_capacity;
    args.mainloop.peer_timeline = peer_timeline;
    args.mainloop.peer_timeline_capacity = peer_timeline_capacity;
    args.mainloop.n_tiles = n_tiles;
  }
#endif
  typename Kernel::Arguments launch_args{};
  launch_args.gemm = args;
  launch_args.comm = comm;
  launch_args.num_comm_ctas = params.num_comm_ctas;
#if FUSE_ENABLE_PROFILING
  if constexpr (Instrumented) {
    launch_args.timeline = timeline;
    launch_args.timeline_capacity = timeline_capacity;
  }
#endif
  return launch_monolithic<Kernel>(launch_args, info, stream);
}

template <bool Instrumented = false>
cudaError_t launch_oproj_forward_policy(
    const A2AGemmParams& params, cudaStream_t stream
#if FUSE_ENABLE_PROFILING
    , A2AGemmCtaTimeline* timeline = nullptr, int32_t timeline_capacity = 0,
    A2AGemmPeerTimeline* peer_timeline = nullptr, int32_t peer_timeline_capacity = 0
#endif
    ) {
  FUSE_SM103_HOST_BEGIN();
  OprojGemmPolicy policy{};
  const cudaError_t status = select_oproj_gemm_policy(&policy);
  if (status != cudaSuccess) {
    FUSE_SM103_HOST_RETURN(status);
  }
  auto launch = [&](auto binding_tag) {
    using Binding = typename decltype(binding_tag)::type;
#if FUSE_ENABLE_PROFILING
    using Gemm = std::conditional_t<Instrumented,
        typename Binding::TelemetryGemm, typename Binding::Gemm>;
    using Kernel = std::conditional_t<Instrumented,
        typename Binding::TelemetryKernel, typename Binding::Kernel>;
    using Comm = std::conditional_t<Instrumented,
        typename Binding::TelemetryComm, typename Binding::Comm>;
    return launch_a2a_lhs_gemm_policy<Gemm, Kernel, Comm, Instrumented>(
        params, stream, timeline, timeline_capacity,
        peer_timeline, peer_timeline_capacity);
#else
    static_assert(!Instrumented);
    return launch_a2a_lhs_gemm_policy<
        typename Binding::Gemm, typename Binding::Kernel, typename Binding::Comm>(
            params, stream);
#endif
  };
  FUSE_SM103_HOST_RETURN(visit_oproj_forward_policy(policy, launch));
}

template <class Gemm, class Kernel, class Comm, bool Instrumented = false,
          bool EpilogueInstrumented = false>
cudaError_t launch_gemm_a2a_impl(
    const GemmA2AParams& params, cudaStream_t stream
#if FUSE_ENABLE_PROFILING
    , A2AGemmCtaTimeline* timeline = nullptr, int32_t timeline_capacity = 0,
    detail::QkvEpilogueRecord* epilogue_records = nullptr, int32_t epilogue_record_capacity = 0,
    QkvRouteTimeline* route_timeline = nullptr, int32_t route_capacity = 0
#endif
    ) {
  if (!supported_problem(params.gemm) || !std::isfinite(params.alpha) ||
      params.epoch == 0 || params.num_comm_ctas <= 0) {
    return cudaErrorInvalidValue;
  }
  DeviceInfo info{};
  cudaError_t status = device_info(&info);
  if (status != cudaSuccess) {
    return status;
  }
  if (params.num_comm_ctas >= info.sm_count) {
    return cudaErrorInvalidValue;
  }
  typename Comm::Arguments comm{};
  comm.params = params;
#if FUSE_ENABLE_PROFILING
  if constexpr (Instrumented) {
    if (!timeline || timeline_capacity < info.sm_count) {
      return cudaErrorInvalidValue;
    }
  }
  if constexpr (EpilogueInstrumented) {
    static_assert(Instrumented);
    if (!epilogue_records || epilogue_record_capacity < info.sm_count ||
        reinterpret_cast<uintptr_t>(epilogue_records) % alignof(detail::QkvEpilogueRecord) != 0) {
      return cudaErrorInvalidValue;
    }
  }
#endif
  // Resolve the communication traversal to the same raster as the producer.
  if (comm.params.gemm.raster == GemmRaster::kHeuristic) {
    comm.params.gemm.raster = GemmRaster::kAlongM;
  }
  FUSE_SM103_HOST_MARK(kCommunicationPrepare);
  status = Comm::initialize(comm);
  if (status != cudaSuccess) {
    return status;
  }
#if FUSE_ENABLE_PROFILING
  if constexpr (Instrumented) {
    if (route_timeline) {
      // Detailed records currently describe the tensor-copy path, not vector
      // fallback. Reject unsupported geometry rather than emit partial data.
      const int64_t tasks = Comm::route_slots(params);
      if (!comm.use_tma || !comm.use_tma_store ||
          route_capacity < tasks + params.num_comm_ctas * Comm::kQkvBulkSlots) {
        return cudaErrorInvalidValue;
      }
    }
  }
#endif
  FUSE_SM103_HOST_MARK(kArguments);
  auto args = gemm_arguments<Gemm>(
      params.gemm, params.lhs, params.rhs_nt, params.local_output,
      params.alpha, params.num_comm_ctas, info, GemmRaster::kAlongM);
#if FUSE_SM103_QKV_RANK_SWIZZLE
  // Only rotate when the consumer follows the shared tile-order contract.
  // The scalar/vector fallback retains the original producer order.
  args.scheduler.n_band_rank = comm.use_tma ? params.route.rank : 0;
#endif
  args.epilogue.ready = params.ready;
  using ProducerTile = typename Gemm::TileShape;
  static_assert(cute::size<0>(ProducerTile{}) == Comm::kBlockM);
  static_assert(cute::size<1>(ProducerTile{}) == Comm::kBlockN);
  args.epilogue.m_tiles = ceil_div(params.gemm.m, Comm::kBlockM);
  args.epilogue.n_tiles = ceil_div(params.gemm.n, Comm::kBlockN);
  args.epilogue.epoch = params.epoch;
#if FUSE_ENABLE_PROFILING
  if constexpr (EpilogueInstrumented) {
    args.epilogue.records = epilogue_records;
    args.epilogue.record_capacity = epilogue_record_capacity;
  }
#endif
  typename Kernel::Arguments launch_args{};
  launch_args.gemm = args;
  launch_args.comm = comm;
  launch_args.num_comm_ctas = params.num_comm_ctas;
#if FUSE_ENABLE_PROFILING
  if constexpr (Instrumented) {
    launch_args.timeline = timeline;
    launch_args.timeline_capacity = timeline_capacity;
    launch_args.route_timeline = route_timeline;
  }
#endif
  return launch_monolithic<Kernel>(launch_args, info, stream);
}

template <bool Instrumented = false>
cudaError_t launch_qkv_forward_policy(
    const GemmA2AParams& params, cudaStream_t stream
#if FUSE_ENABLE_PROFILING
    , A2AGemmCtaTimeline* timeline = nullptr, int32_t timeline_capacity = 0,
    QkvRouteTimeline* route_timeline = nullptr, int32_t route_capacity = 0
#endif
    ) {
  FUSE_SM103_HOST_BEGIN();
  QkvGemmPolicy policy{};
  const cudaError_t status = select_qkv_gemm_policy(&policy);
  if (status != cudaSuccess) {
    FUSE_SM103_HOST_RETURN(status);
  }
  auto launch = [&](auto binding_tag) {
    using Binding = typename decltype(binding_tag)::type;
#if FUSE_ENABLE_PROFILING
    using Kernel = std::conditional_t<Instrumented,
        typename Binding::TelemetryKernel, typename Binding::Kernel>;
    return launch_gemm_a2a_impl<
        typename Binding::Gemm, Kernel, typename Binding::Comm, Instrumented>(
            params, stream, timeline, timeline_capacity,
            nullptr, 0, route_timeline, route_capacity);
#else
    static_assert(!Instrumented);
    return launch_gemm_a2a_impl<
        typename Binding::Gemm, typename Binding::Kernel, typename Binding::Comm>(
            params, stream);
#endif
  };
  FUSE_SM103_HOST_RETURN(visit_qkv_forward_policy(
      policy, params.route.qkv_peer_interleaved, launch));
}

}  // namespace
}  // namespace fuse
