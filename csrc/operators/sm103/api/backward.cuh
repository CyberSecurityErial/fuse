// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include "fuse/profiling/backward.cuh"

namespace fuse {
namespace {

// First BF16 baseline: one explicit tile, no forward-autotune reuse. Reverse
// operands are NN (dgrad) and TN (wgrad), exactly as in the public SM90 API.
// TODO: tune backward GEMM tiles and recalibrate the producer/consumer model.
using BackwardTypes = Bf16GemmTypes<128, 64, 0, LayoutA, cutlass::layout::RowMajor>;
using BackwardWeightTypes = Bf16GemmTypes<128, 64, 0,
    cutlass::layout::ColumnMajor, cutlass::layout::RowMajor, Bf16>;
using BackwardReadyMainloop = detail::A2ALhsReadyMainloop<
    typename BackwardTypes::Mainloop, typename BackwardTypes::TileShape,
#if FUSE_ENABLE_PROFILING
    false,
#endif
    true>;
using BackwardReadyGemm = cutlass::gemm::kernel::GemmUniversal<ProblemShape,
    BackwardReadyMainloop, typename BackwardTypes::Epilogue, detail::MonolithicPersistentScheduler>;
using QkvBackwardPushComm = QkvBackwardPushCommT<128>;
using QkvBackwardDataKernel = detail::MonolithicGemm<BackwardReadyGemm, QkvBackwardPushComm>;
using OprojBackwardHeadComm = QkvGqaPackCommT<OprojBackwardKernelParams, 128, 128>;
using OprojBackwardDataKernel = detail::MonolithicGemm<typename BackwardTypes::OutputGemm, OprojBackwardHeadComm>;

template <int TileN, int EpilogueN>
struct BackwardDataBinding {
  using Types = Bf16GemmTypes<TileN,64,EpilogueN,LayoutA,cutlass::layout::RowMajor>;
  using ReadyMainloop = detail::A2ALhsReadyMainloop<typename Types::Mainloop,typename Types::TileShape,
#if FUSE_ENABLE_PROFILING
      false,
#endif
      true>;
  using ReadyGemm = cutlass::gemm::kernel::GemmUniversal<ProblemShape,ReadyMainloop,
      typename Types::Epilogue,detail::MonolithicPersistentScheduler>;
  using OutputGemm = typename Types::OutputGemm;
  using OutputComm = QkvGqaPackCommT<OprojBackwardKernelParams,128,TileN>;
  using QkvKernel = detail::MonolithicGemm<ReadyGemm,QkvBackwardPushComm>;
  using OprojKernel = detail::MonolithicGemm<OutputGemm,OutputComm>;
};

template <class P, class Fn>
auto dispatch_backward_binding(const P& p, Fn&& fn) {
  if (p.gemm_policy == BackwardGemmPolicy::kM128N256) {
    if (p.gemm_tuning.epilogue_n == 64) return fn(BackwardDataBinding<256,64>{});
    return fn(BackwardDataBinding<256,0>{});
  }
  return fn(BackwardDataBinding<128,0>{});
}

inline bool backward_policy_supported(BackwardGemmPolicy policy) {
  return policy == BackwardGemmPolicy::kAuto || policy == BackwardGemmPolicy::kM128N128 ||
      policy == BackwardGemmPolicy::kM128N256;
}

template <class P>
bool backward_tuning_supported(const P& p) {
  const auto& t=p.gemm_tuning;
  return backward_policy_supported(p.gemm_policy) &&
      (t.epilogue_n==0 || (t.epilogue_n==64 && p.gemm_policy==BackwardGemmPolicy::kM128N256)) &&
      (t.max_swizzle_size==1 || t.max_swizzle_size==2 || t.max_swizzle_size==4 || t.max_swizzle_size==8);
}

template <class P>
bool backward_shape_supported(const P& p) {
  return p.world_size > 0 && p.world_size <= kMaxWorldSize && p.rank >= 0 &&
      p.rank < p.world_size && p.batch > 0 && p.local_tokens > 0 &&
      p.local_tokens % p.batch == 0 && p.hidden > 0 && p.q_heads > 0 &&
      p.q_heads % p.world_size == 0 && p.head_dim > 0 && p.head_dim % 64 == 0 &&
      static_cast<int64_t>(p.local_tokens) * p.world_size <= INT32_MAX &&
      static_cast<int64_t>(p.q_heads) * p.head_dim <= INT32_MAX &&
      (!p.causal_load_balanced || (p.local_tokens / p.batch) % 2 == 0) &&
      std::isfinite(p.alpha) && p.epoch > 0 && backward_tuning_supported(p);
}

template <class P>
UlyssesRoute backward_route(const P& p, bool qkv) {
  UlyssesRoute route{};
  route.world_size = p.world_size;
  route.rank = p.rank;
  route.batch = p.batch;
  route.seq_local = p.local_tokens / p.batch;
  route.global_seq = route.seq_local * p.world_size;
  route.q_heads = p.q_heads;
  route.local_heads = p.q_heads / p.world_size;
  route.head_dim = p.head_dim;
  route.causal_load_balanced = p.causal_load_balanced;
  route.kind = qkv ? RouteKind::kQkvGqaPack : RouteKind::kHeadToSequence;
  route.direction = qkv ? RouteDirection::kInverse : RouteDirection::kForward;
  return route;
}

template <class Gemm, bool WeightGradient = false>
typename Gemm::Arguments backward_gemm_arguments(int m, int n, int k,
    const Bf16* a, const Bf16* b, Bf16* d, float alpha, float beta,
    int comm_ctas, const DeviceInfo& info) {
  typename Gemm::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape = cute::make_shape(m, n, k, 1);
  args.mainloop.ptr_A = a;
  if constexpr (WeightGradient) args.mainloop.dA = cute::make_stride(cute::_1{}, int64_t{m}, int64_t{0});
  else args.mainloop.dA = cute::make_stride(int64_t{k}, cute::_1{}, int64_t{0});
  args.mainloop.ptr_B = b;
  args.mainloop.dB = cute::make_stride(cute::_1{}, int64_t{n}, int64_t{0});
  args.epilogue.thread.alpha = alpha;
  args.epilogue.thread.beta = beta;
  args.epilogue.ptr_C = WeightGradient ? d : nullptr;
  args.epilogue.ptr_D = d;
  args.epilogue.dC = cute::make_stride(int64_t{n}, cute::_1{}, int64_t{0});
  args.epilogue.dD = args.epilogue.dC;
  args.hw_info.device_id = info.device;
  args.hw_info.sm_count = info.sm_count - comm_ctas;
  args.scheduler.block_offset = comm_ctas;
  args.scheduler.max_swizzle_size = 1;
  args.scheduler.raster_order = raster_option(GemmRaster::kAlongN, GemmRaster::kAlongN);
  return args;
}

template <class Binding = BackwardDataBinding<128,0>, bool Profile = false>
cudaError_t qkv_backward_data_impl(const QkvBackwardDataParams& p, cudaStream_t stream
#if FUSE_ENABLE_PROFILING
    , A2AGemmCtaTimeline* timeline = nullptr, int capacity = 0
#endif
    ) {
  if (!backward_shape_supported(p) || p.kv_heads <= 0 || p.q_heads % p.kv_heads ||
      p.kv_heads % p.world_size ||
      (static_cast<int64_t>(p.q_heads) + 2LL*p.kv_heads)*p.head_dim > INT32_MAX)
    return cudaErrorInvalidValue;
  DeviceInfo info{};
  auto status = device_info(&info);
  if (status != cudaSuccess) return status;
  const int comm = p.num_comm_ctas ? p.num_comm_ctas : 16;
  if (comm <= 0 || comm >= info.sm_count) return cudaErrorInvalidValue;
  QkvBackwardKernelParams normalized{};
  normalized.local_q = p.grad_q; normalized.local_k = p.grad_k; normalized.local_v = p.grad_v;
  normalized.weight = p.weight; normalized.grad_input = p.grad_input;
  normalized.route = backward_route(p, true); normalized.route.kv_heads = p.kv_heads;
  normalized.gemm = {p.local_tokens, p.hidden, (p.q_heads + 2*p.kv_heads)*p.head_dim, 1};
  normalized.gemm.transpose_b = true; normalized.gemm.stride_b.row = p.hidden;
  normalized.num_comm_ctas = comm; normalized.epoch = p.epoch; normalized.alpha = p.alpha;
  for (int peer = 0; peer < p.world_size; ++peer) {
    normalized.peer_staging[peer] = p.peer_dqkv_staging[peer];
    normalized.peer_ready[peer] = p.peer_ready[peer];
    normalized.peer_done_epoch[peer] = p.peer_done_epoch[peer];
  }
  using BackwardReadyGemm = typename Binding::ReadyGemm;
  using Production = typename Binding::QkvKernel;
#if FUSE_ENABLE_PROFILING
  using Kernel = std::conditional_t<Profile,
      GemmA2ARoleTelemetryKernel<BackwardReadyGemm, QkvBackwardPushComm, true, false>, Production>;
#else
  using Kernel = Production;
#endif
  typename Kernel::Arguments args{};
  args.comm.params = normalized;
  if (!QkvBackwardPushComm::can_implement(args.comm)) return cudaErrorNotSupported;
  status = QkvBackwardPushComm::initialize(args.comm);
  if (status != cudaSuccess) return status;
  args.gemm = backward_gemm_arguments<BackwardReadyGemm>(normalized.gemm.m,
      normalized.gemm.n, normalized.gemm.k, p.peer_dqkv_staging[p.rank], p.weight,
      p.grad_input, p.alpha, 0, comm, info);
  args.gemm.scheduler.max_swizzle_size=p.gemm_tuning.max_swizzle_size;
  args.gemm.scheduler.raster_order=raster_option(
      p.gemm_tuning.along_m?GemmRaster::kAlongM:GemmRaster::kAlongN,GemmRaster::kAlongN);
  // N/raster/swizzle change the output-tile visit order, not the complete
  // (M128, head) input-release unit. Every consumer still acquires its own M
  // and logical K-head before loading; multiple N tiles share that release.
  // The source push covers all such units independently of consumer order.
  // The generic ready adapter counts logical K groups: peers for forward
  // OProj, individual Q/K/V heads here. One release covers both row chunks.
  args.gemm.mainloop.ready = p.peer_ready[p.rank];
  args.gemm.mainloop.world_size = p.q_heads + 2*p.kv_heads;
  args.gemm.mainloop.m_tiles = ceil_div(p.local_tokens, 128);
  args.gemm.mainloop.arrivals_per_peer = 1;
  args.gemm.mainloop.k_tiles_per_peer = p.head_dim / 64;
  args.gemm.mainloop.epoch = p.epoch;
  args.num_comm_ctas = comm;
#if FUSE_ENABLE_PROFILING
  if constexpr (Profile) {
    if (!timeline || capacity < info.sm_count) return cudaErrorInvalidValue;
    args.timeline = timeline; args.timeline_capacity = capacity;
  }
#endif
  return launch_monolithic<Kernel>(args, info, stream);
}

template <class Binding = BackwardDataBinding<128,0>, bool Profile = false>
cudaError_t oproj_backward_data_impl(const OprojBackwardDataParams& p, cudaStream_t stream
#if FUSE_ENABLE_PROFILING
    , A2AGemmCtaTimeline* timeline = nullptr, int capacity = 0
#endif
    ) {
  if (!backward_shape_supported(p)) return cudaErrorInvalidValue;
  DeviceInfo info{};
  auto status = device_info(&info);
  if (status != cudaSuccess) return status;
  const int comm = p.num_comm_ctas ? p.num_comm_ctas : 16;
  if (comm <= 0 || comm >= info.sm_count) return cudaErrorInvalidValue;
  OprojBackwardKernelParams normalized{};
  normalized.lhs = p.grad_output; normalized.rhs_nt = p.weight;
  normalized.local_output = p.local_grad_attention; normalized.ready = p.ready;
  normalized.gemm = {p.local_tokens, p.q_heads*p.head_dim, p.hidden, 1};
  normalized.gemm.transpose_b = true; normalized.gemm.stride_b.row = normalized.gemm.n;
  normalized.gemm.raster = p.gemm_tuning.along_m?GemmRaster::kAlongM:GemmRaster::kAlongN;
  normalized.gemm.max_swizzle_size = p.gemm_tuning.max_swizzle_size;
  normalized.route = backward_route(p, false);
  normalized.num_comm_ctas = comm; normalized.epoch = p.epoch; normalized.alpha = p.alpha;
  for (int peer = 0; peer < p.world_size; ++peer) {
    normalized.peer_output[peer] = p.peer_grad_attention[peer];
    normalized.peer_route_done_epoch[peer] = p.peer_done_epoch[peer];
  }
  using BackwardTypes = typename Binding::Types;
  using OprojBackwardHeadComm = typename Binding::OutputComm;
  using Production = typename Binding::OprojKernel;
#if FUSE_ENABLE_PROFILING
  using Kernel = std::conditional_t<Profile,
      GemmA2ARoleTelemetryKernel<typename BackwardTypes::OutputGemm, OprojBackwardHeadComm, true>, Production>;
#else
  using Kernel = Production;
#endif
  typename Kernel::Arguments args{};
  args.comm.params = normalized;
  status = OprojBackwardHeadComm::initialize(args.comm);
  if (status != cudaSuccess) return status;
  args.gemm = backward_gemm_arguments<typename BackwardTypes::OutputGemm>(normalized.gemm.m,
      normalized.gemm.n, normalized.gemm.k, p.grad_output, p.weight,
      p.local_grad_attention, p.alpha, 0, comm, info);
  args.gemm.scheduler.max_swizzle_size=normalized.gemm.max_swizzle_size;
  args.gemm.scheduler.raster_order=raster_option(normalized.gemm.raster,GemmRaster::kAlongN);
  // One geometry drives all three views:
  //   GEMM (M128,Ntile,raster,swizzle) -> ready[Mtile,Ntile]
  //                                -> route producer_schedule -> head slices
  // E32/E64 only changes internal epilogue staging, never the ready unit.
  // CopyOrder is instantiated with the same Ntile and follows this producer
  // schedule; no fixed N128 flag indexing survives a switch to N256.
  args.gemm.epilogue.ready = p.ready;
  args.gemm.epilogue.m_tiles = ceil_div(p.local_tokens, 128);
  args.gemm.epilogue.n_tiles = ceil_div(normalized.gemm.n, BackwardTypes::kTileN);
  args.gemm.epilogue.epoch = p.epoch;
  args.num_comm_ctas = comm;
#if FUSE_ENABLE_PROFILING
  if constexpr (Profile) {
    if (!timeline || capacity < info.sm_count) return cudaErrorInvalidValue;
    args.timeline = timeline; args.timeline_capacity = capacity;
  }
#endif
  return launch_monolithic<Kernel>(args, info, stream);
}

cudaError_t backward_weight_impl(int m, int n, int k, const Bf16* dy,
    const Bf16* x, Bf16* dw, float alpha, float beta, cudaStream_t stream) {
  if (m <= 0 || n <= 0 || k <= 0 || !dy || !x || !dw ||
      !std::isfinite(alpha) || !std::isfinite(beta)) return cudaErrorInvalidValue;
  DeviceInfo info{};
  auto status = device_info(&info);
  if (status != cudaSuccess) return status;
  using Gemm = typename BackwardWeightTypes::PureGemm;
  auto args = backward_gemm_arguments<Gemm, true>(m,n,k,dy,x,dw,alpha,beta,0,info);
  using Adapter = cutlass::gemm::device::GemmUniversalAdapter<Gemm>;
  if (Adapter::can_implement(args) != cutlass::Status::kSuccess || Adapter::get_workspace_size(args))
    return cudaErrorNotSupported;
  Adapter op;
  if (op.initialize(args, nullptr, stream) != cutlass::Status::kSuccess ||
      op.run(stream) != cutlass::Status::kSuccess) return cudaErrorLaunchFailure;
  return cudaGetLastError();
}

}  // namespace

int32_t recommended_qkv_backward_comm_ctas(const QkvBackwardDataParams&) { return 16; }
int32_t recommended_oproj_backward_comm_ctas(const OprojBackwardDataParams&) { return 16; }
BackwardGemmPolicy recommended_qkv_backward_gemm_policy(const QkvBackwardDataParams& p, int32_t, int32_t) {
  return p.gemm_policy == BackwardGemmPolicy::kAuto ? BackwardGemmPolicy::kM128N128 : p.gemm_policy;
}
BackwardGemmPolicy recommended_oproj_backward_gemm_policy(const OprojBackwardDataParams& p, int32_t, int32_t) {
  return p.gemm_policy == BackwardGemmPolicy::kAuto ? BackwardGemmPolicy::kM128N128 : p.gemm_policy;
}
KernelTraits qkv_backward_kernel_traits(const QkvBackwardDataParams& p, int32_t, int32_t) {
  if (!backward_tuning_supported(p)) return {};
  return dispatch_backward_binding(p,[](auto binding) {
    using B=decltype(binding); using K=typename B::QkvKernel;
    return KernelTraits{128,B::Types::kTileN,64,K::MaxThreadsPerBlock,sizeof(typename K::SharedStorage)};
  });
}
KernelTraits oproj_backward_kernel_traits(const OprojBackwardDataParams& p, int32_t, int32_t) {
  if (!backward_tuning_supported(p)) return {};
  return dispatch_backward_binding(p,[](auto binding) {
    using B=decltype(binding); using K=typename B::OprojKernel;
    return KernelTraits{128,B::Types::kTileN,64,K::MaxThreadsPerBlock,sizeof(typename K::SharedStorage)};
  });
}
int64_t qkv_backward_ready_elements(const QkvBackwardDataParams& p) {
  return p.local_tokens > 0 && p.q_heads > 0 && p.kv_heads > 0
      ? static_cast<int64_t>(ceil_div(p.local_tokens,128))*(p.q_heads+2LL*p.kv_heads)*kReadyFlagStride : 0;
}
int64_t oproj_backward_ready_elements(const OprojBackwardDataParams& p) {
  const int64_t n = static_cast<int64_t>(p.q_heads)*p.head_dim;
  return p.local_tokens > 0 && n > 0 && n <= INT32_MAX
      ? static_cast<int64_t>(ceil_div(p.local_tokens,128))*ceil_div(static_cast<int>(n),
          p.gemm_policy==BackwardGemmPolicy::kM128N256?256:128)*kReadyFlagStride : 0;
}
cudaError_t launch_qkv_backward_data(const QkvBackwardDataParams& p, cudaStream_t s) {
  return dispatch_backward_binding(p,[&](auto b){return qkv_backward_data_impl<decltype(b)>(p,s);});
}
cudaError_t launch_oproj_backward_data(const OprojBackwardDataParams& p, cudaStream_t s) {
  return dispatch_backward_binding(p,[&](auto b){return oproj_backward_data_impl<decltype(b)>(p,s);});
}
cudaError_t launch_qkv_backward_weight(const QkvBackwardWeightParams& p, cudaStream_t s) {
  const int64_t n = (p.q_heads+2LL*p.kv_heads)*p.head_dim;
  if (n <= 0 || n > INT32_MAX) return cudaErrorInvalidValue;
  return backward_weight_impl(static_cast<int>(n),p.hidden,p.local_tokens,p.dqkv_staging,p.saved_input,p.grad_weight,p.alpha,p.beta,s);
}
cudaError_t launch_oproj_backward_weight(const OprojBackwardWeightParams& p, cudaStream_t s) {
  const int64_t n = static_cast<int64_t>(p.q_heads)*p.head_dim;
  if (n <= 0 || n > INT32_MAX) return cudaErrorInvalidValue;
  return backward_weight_impl(p.hidden,static_cast<int>(n),p.local_tokens,p.grad_output,p.saved_attention,p.grad_weight,p.alpha,p.beta,s);
}
cudaError_t launch_qkv_backward(const QkvBackwardParams& p, cudaStream_t s) {
  if (p.weight_mode != WeightGradientMode::kImmediate && p.weight_mode != WeightGradientMode::kDeferred) return cudaErrorInvalidValue;
  const auto status = launch_qkv_backward_data(p.data,s);
  return status == cudaSuccess && p.weight_mode == WeightGradientMode::kImmediate ? launch_qkv_backward_weight(p.weight,s) : status;
}
cudaError_t launch_oproj_backward(const OprojBackwardParams& p, cudaStream_t s) {
  if (p.weight_mode != WeightGradientMode::kImmediate && p.weight_mode != WeightGradientMode::kDeferred) return cudaErrorInvalidValue;
  const auto status = launch_oproj_backward_data(p.data,s);
  return status == cudaSuccess && p.weight_mode == WeightGradientMode::kImmediate ? launch_oproj_backward_weight(p.weight,s) : status;
}
#if FUSE_ENABLE_PROFILING
namespace {
template <class Production, int TileN, int TileK, int EpilogueN>
cudaError_t backward_gemm_reference_impl(int m, int n, int k,
    const Bf16* a, const Bf16* b, Bf16* d, int reserved, cudaStream_t stream,
    int swizzle, bool along_m) {
  DeviceInfo info{};
  auto status = reference_device_info(&info,reserved);
  if (status != cudaSuccess) return status;
  using Types = Bf16GemmTypes<TileN, TileK, EpilogueN, LayoutA, cutlass::layout::RowMajor>;
  using Gemm = typename Types::PureGemm;
  using Kernel = detail::GemmReferenceKernel<Gemm,Production>;
  auto args = backward_gemm_arguments<Gemm>(m,n,k,a,b,d,1,0,reserved,info);
  args.scheduler.block_offset = 0;
  args.scheduler.max_swizzle_size = swizzle;
  args.scheduler.raster_order = raster_option(
      along_m ? GemmRaster::kAlongM : GemmRaster::kAlongN, GemmRaster::kAlongN);
  if (!Gemm::can_implement(args) || Gemm::get_workspace_size(args)!=0) return cudaErrorNotSupported;
  if (Gemm::initialize_workspace(args,nullptr,stream)!=cutlass::Status::kSuccess) return cudaErrorInitializationError;
  return detail::launch_reference_cooperative<Kernel>(
      Gemm::to_underlying_arguments(args,nullptr),info,info.sm_count-reserved,stream);
}
}  // namespace
cudaError_t launch_backward_gemm_reference(bool qkv, int m, int n, int k,
    const Bf16* a, const Bf16* b, Bf16* d, int reserved, cudaStream_t stream,
    int tile_n, int tile_k, int epilogue_n, int swizzle, bool along_m) {
  if (swizzle != 1 && swizzle != 2 && swizzle != 4 && swizzle != 8) return cudaErrorInvalidValue;
  // Search compute geometry only. Neither production ready granularity nor
  // communication scheduling is changed by this diagnostic dispatch.
#define FUSE_BACKWARD_REFERENCE(N, K, E) \
  if (tile_n == N && tile_k == K && epilogue_n == E) \
    return qkv ? backward_gemm_reference_impl<QkvBackwardDataKernel,N,K,E>(m,n,k,a,b,d,reserved,stream,swizzle,along_m) \
               : backward_gemm_reference_impl<OprojBackwardDataKernel,N,K,E>(m,n,k,a,b,d,reserved,stream,swizzle,along_m)
  FUSE_BACKWARD_REFERENCE(128,64,0);
  FUSE_BACKWARD_REFERENCE(192,64,0);
  FUSE_BACKWARD_REFERENCE(256,64,0);
  FUSE_BACKWARD_REFERENCE(128,128,0);
  FUSE_BACKWARD_REFERENCE(256,128,32);
  FUSE_BACKWARD_REFERENCE(256,64,64);
#undef FUSE_BACKWARD_REFERENCE
  return cudaErrorNotSupported;
}
cudaError_t launch_qkv_backward_data_role_telemetry(const QkvBackwardDataParams& p,
    A2AGemmCtaTimeline* t, int32_t capacity, cudaStream_t s) {
  return dispatch_backward_binding(p,[&](auto b){return qkv_backward_data_impl<decltype(b),true>(p,s,t,capacity);});
}
cudaError_t launch_oproj_backward_data_role_telemetry(const OprojBackwardDataParams& p,
    A2AGemmCtaTimeline* t, int32_t capacity, cudaStream_t s) {
  return dispatch_backward_binding(p,[&](auto b){return oproj_backward_data_impl<decltype(b),true>(p,s,t,capacity);});
}
#endif
}  // namespace fuse
