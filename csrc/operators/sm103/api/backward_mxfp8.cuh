// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <type_traits>

namespace fuse {
namespace {

// Backward keeps BF16 masters alive: W and saved A have different K32 axes
// from forward. Explicit preparation kernels establish the first baseline.
// Their stream edges and runtime belong to B/W, not an untimed setup phase.
// TODO: overlap transposed W preparation with routing only after profiling
// this complete boundary; do not reuse forward rowwise scales as a shortcut.
struct Mxfp8BackwardWeightWorkspace {
  Mxfp8Workspace rhs{};
  Fp8E4m3* lhs = nullptr;
  cutlass::float_ue8m0_t* sfa = nullptr;
  size_t bytes = 0;

  static Mxfp8BackwardWeightWorkspace make(const GemmProblem& g, void* memory) {
    Mxfp8BackwardWeightWorkspace w;
    w.rhs = Mxfp8Workspace::make(g, memory);
    const size_t data = Mxfp8Workspace::align(size_t(g.m) * g.k);
    const auto shape = cute::make_shape(g.m, g.n, g.k, 1);
    const size_t scales = Mxfp8Workspace::align(cute::size(cute::filter_zeros(
        Mxfp8ScaleConfig::tile_atom_to_shape_SFA(shape))));
    w.bytes = w.rhs.bytes + data + scales;
    if (memory) {
      auto* base = static_cast<unsigned char*>(memory) + w.rhs.bytes;
      w.lhs = reinterpret_cast<Fp8E4m3*>(base);
      w.sfa = reinterpret_cast<cutlass::float_ue8m0_t*>(base + data);
    }
    return w;
  }
};

struct Mxfp8OprojBackwardComm : QkvGqaPackCommT<OprojBackwardKernelParams, 128, 256> {
  using Base = QkvGqaPackCommT<OprojBackwardKernelParams, 128, 256>;
  static bool can_implement(const Arguments& args) {
    // Input pointers are explicitly MXFP8, not the BF16 fields of the shared
    // output-route metadata. Keep all output/flag/route validation unchanged.
    return Base::can_implement(args, false);
  }
  static cudaError_t initialize(Arguments& args) { return Base::initialize(args, false); }
};

template <class P>
bool mxfp8_backward_dimensions(const P& p) {
  const int64_t width = int64_t{p.q_heads} * p.head_dim;
  return p.local_tokens > 0 && p.local_tokens % 128 == 0 &&
      p.hidden > 0 && p.hidden % 128 == 0 && p.q_heads > 0 &&
      p.head_dim == 128 && width > 0 && width <= INT32_MAX - 127;
}

bool mxfp8_backward_tuning(const BackwardGemmTuning& t) {
  return (t.epilogue_n == 0 || t.epilogue_n == 32 || t.epilogue_n == 64) &&
      (t.max_swizzle_size == 1 || t.max_swizzle_size == 2 ||
       t.max_swizzle_size == 4 || t.max_swizzle_size == 8);
}

GemmProblem mxfp8_backward_weight_problem(const OprojBackwardWeightParams& p,
                                        const BackwardGemmTuning& t = {}) {
  GemmProblem g{p.hidden, p.q_heads * p.head_dim, p.local_tokens, 1};
  g.max_swizzle_size = t.max_swizzle_size;
  g.raster = t.along_m ? GemmRaster::kAlongM : GemmRaster::kAlongN;
  return g;
}

cudaError_t validate_mxfp8_backward_weight(const Mxfp8OprojBackwardWeightParams& p) {
  const auto& g = p.projection;
  if (!mxfp8_backward_dimensions(g) || !mxfp8_backward_tuning(p.gemm_tuning) ||
      !std::isfinite(g.alpha) || !std::isfinite(g.beta) ||
      !mxfp8_aligned(g.grad_output, 16) || !mxfp8_aligned(g.saved_attention, 16) ||
      !mxfp8_aligned(g.grad_weight, 16) || !mxfp8_aligned(p.workspace))
    return cudaErrorInvalidValue;
  const auto w = Mxfp8BackwardWeightWorkspace::make(mxfp8_backward_weight_problem(g), p.workspace);
  return p.workspace_bytes >= w.bytes ? cudaSuccess : cudaErrorInvalidValue;
}

cudaError_t validate_mxfp8_backward_data(const Mxfp8OprojBackwardDataParams& p) {
  auto basic = p.projection;
  // Reuse BF16 route validation, without applying its epilogue whitelist to
  // the distinct MXFP8 collective. No BF16 validation is relaxed globally.
  basic.gemm_tuning.epilogue_n = 0;
  if (!mxfp8_backward_dimensions(basic) || !backward_shape_supported(basic) ||
      !mxfp8_backward_tuning(p.projection.gemm_tuning) ||
      (basic.gemm_policy != BackwardGemmPolicy::kAuto &&
       basic.gemm_policy != BackwardGemmPolicy::kM128N256) ||
      basic.num_comm_ctas <= 0 || !mxfp8_aligned(basic.weight, 16) ||
      !mxfp8_aligned(p.workspace)) return cudaErrorInvalidValue;
  const GemmProblem g{basic.local_tokens, basic.q_heads * basic.head_dim, basic.hidden, 1};
  if (!valid_mxfp8_activation(g, p.grad_output) ||
      p.workspace_bytes < Mxfp8Workspace::make(g).bytes) return cudaErrorInvalidValue;
  OprojBackwardKernelParams normalized{};
  normalized.gemm = g;
  normalized.gemm.transpose_b = true;
  normalized.gemm.stride_b.row = g.n;
  normalized.route = backward_route(basic, false);
  normalized.num_comm_ctas = basic.num_comm_ctas;
  normalized.epoch = basic.epoch;
  normalized.local_output = basic.local_grad_attention;
  normalized.ready = basic.ready;
  for (int peer = 0; peer < basic.world_size; ++peer) {
    normalized.peer_output[peer] = basic.peer_grad_attention[peer];
    normalized.peer_route_done_epoch[peer] = basic.peer_done_epoch[peer];
  }
  using Comm = Mxfp8OprojBackwardComm;
  typename Comm::Arguments comm{};
  comm.params = normalized;
  return Comm::can_implement(comm) ? cudaSuccess : cudaErrorNotSupported;
}

template <class Layout>
cudaError_t prepare_mxfp8_transpose(const Bf16* source, Fp8E4m3* output,
    cutlass::float_ue8m0_t* scales, int rows, int k, int64_t source_stride,
    Layout layout, const DeviceInfo& info, cudaStream_t stream) {
  // Public backward dimensions are K128-aligned. The K32 numerical groups
  // share a K128 scale atom and two K64 writeback stages; no fallback layout
  // or per-model tuning is needed. Static + dynamic SMEM stays below 48KiB.
  const int64_t tiles = ((int64_t{rows} + kMxfp8TransposeRows - 1) / kMxfp8TransposeRows) *
      (k / (32 * kMxfp8TransposeGroups));
  const size_t shared_bytes = size_t{kMxfp8TransposeRows} *
      (kMxfp8TransposeStoreGroups * 2 + 1) * sizeof(uint4);
  // Use the compiled kernel's resource limit, not a fixed CTA/SM multiplier.
  int active = 0;
  auto status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,
      quantize_mxfp8_transposed_operand<Layout>, 256, shared_bytes);
  if (status != cudaSuccess) return status;
  if (active <= 0) return cudaErrorInvalidConfiguration;
  const int blocks = int(std::min<int64_t>(tiles, int64_t{info.sm_count} * active));
  quantize_mxfp8_transposed_operand<<<blocks, 256, shared_bytes, stream>>>(
      source, output, scales, rows, k, source_stride, layout);
  return cudaGetLastError();
}

template <int EpilogueN>
cudaError_t oproj_backward_mxfp8_data_impl(const Mxfp8OprojBackwardDataParams& p,
                                        cudaStream_t stream) {
  auto status = validate_mxfp8_backward_data(p);
  if (status != cudaSuccess) return status;
  DeviceInfo info{};
  status = device_info(&info);
  if (status != cudaSuccess) return status;
  const auto& d = p.projection;
  if (d.num_comm_ctas >= info.sm_count) return cudaErrorInvalidValue;
  GemmProblem g{d.local_tokens, d.q_heads * d.head_dim, d.hidden, 1};
  g.max_swizzle_size = d.gemm_tuning.max_swizzle_size;
  g.raster = d.gemm_tuning.along_m ? GemmRaster::kAlongM : GemmRaster::kAlongN;
  const auto w = Mxfp8Workspace::make(g, p.workspace);
  using Types = Mxfp8GemmFamily<256, 128, EpilogueN>;
  using Gemm = typename Types::OutputGemm;
  using Comm = Mxfp8OprojBackwardComm;
  using Kernel = detail::MonolithicGemm<Gemm, Comm>;
  typename Kernel::Arguments args{};
  auto& route = args.comm.params;
  route.gemm = g;
  // Physical BF16 master W is [H,A]. Routing only consumes BF16 output;
  // the GEMM below instead reads its freshly packed MXFP8 [A,H] representation.
  route.gemm.transpose_b = true;
  route.gemm.stride_b.row = g.n;
  route.route = backward_route(d, false);
  route.local_output = d.local_grad_attention;
  route.ready = d.ready;
  route.num_comm_ctas = d.num_comm_ctas;
  route.epoch = d.epoch;
  for (int peer = 0; peer < d.world_size; ++peer) {
    route.peer_output[peer] = d.peer_grad_attention[peer];
    route.peer_route_done_epoch[peer] = d.peer_done_epoch[peer];
  }
  status = Comm::initialize(args.comm);
  if (status != cudaSuccess) return status;
  args.gemm = gemm_arguments<Gemm>(g, p.grad_output.data, w.b,
      d.local_grad_attention, d.alpha, d.num_comm_ctas, info, GemmRaster::kAlongN);
  args.gemm.mainloop.ptr_SFA = reinterpret_cast<const cutlass::float_ue8m0_t*>(p.grad_output.scales);
  args.gemm.mainloop.ptr_SFB = w.sfb;
  args.gemm.mainloop.layout_SFA = Mxfp8ScaleConfig::tile_atom_to_shape_SFA(args.gemm.problem_shape);
  args.gemm.mainloop.layout_SFB = Mxfp8ScaleConfig::tile_atom_to_shape_SFB(args.gemm.problem_shape);
  // Stream-ordered preparation below completes the whole weight. No stale
  // forward ready flags are reused; only output full-tile publication is live.
  args.gemm.mainloop.weight_ready = nullptr;
  args.gemm.epilogue.ready = d.ready;
  args.gemm.epilogue.m_tiles = ceil_div(g.m, 128);
  args.gemm.epilogue.n_tiles = ceil_div(g.n, 256);
  args.gemm.epilogue.epoch = d.epoch;
  args.num_comm_ctas = d.num_comm_ctas;
  if (!Kernel::can_implement(args) || Kernel::get_workspace_size(args) != 0)
    return cudaErrorNotSupported;
  status = prepare_mxfp8_transpose(d.weight, w.b, w.sfb, g.n, g.k, g.n,
      args.gemm.mainloop.layout_SFB, info, stream);
  return status == cudaSuccess ? launch_monolithic<Kernel>(args, info, stream) : status;
}

// Both projection gradients are the same physical operation G^T * X. Resolve
// semantic dimensions at the entry, then share preparation and CUTLASS without
// inventing a different model's head count or copying the three-kernel path.
struct Mxfp8WeightGradientOperands {
  GemmProblem gemm{};
  const Bf16* gradient = nullptr;  // Original [K,M], quantized along reduction K.
  const Bf16* input = nullptr;     // Original [K,N], independently quantized.
  Bf16* output = nullptr;          // [M,N].
  float alpha = 1.f, beta = 0.f;
};

template <int EpilogueN, bool Prepare, class SourceElement>
cudaError_t backward_mxfp8_weight_kernel(const Mxfp8WeightGradientOperands& d,
                                       void* workspace, cudaStream_t stream) {
  DeviceInfo info{};
  auto status = device_info(&info);
  if (status != cudaSuccess) return status;
  const auto& g = d.gemm;
  const auto w = Mxfp8BackwardWeightWorkspace::make(g, workspace);
  using Gemm = typename Mxfp8GemmFamily<256, 128, EpilogueN, 0, SourceElement>::PureGemm;
  using Adapter = cutlass::gemm::device::GemmUniversalAdapter<Gemm>;
  auto args = gemm_arguments<Gemm>(g, w.lhs, w.rhs.b, d.output,
      d.alpha, 0, info, GemmRaster::kAlongN);
  args.mainloop.ptr_SFA = w.sfa;
  args.mainloop.ptr_SFB = w.rhs.sfb;
  args.mainloop.layout_SFA = Mxfp8ScaleConfig::tile_atom_to_shape_SFA(args.problem_shape);
  args.mainloop.layout_SFB = Mxfp8ScaleConfig::tile_atom_to_shape_SFB(args.problem_shape);
  args.epilogue.thread.beta = d.beta;
  if constexpr (!std::is_void_v<SourceElement>) args.epilogue.ptr_C = d.output;
  if (Adapter::can_implement(args) != cutlass::Status::kSuccess || Adapter::get_workspace_size(args))
    return cudaErrorNotSupported;
  Adapter op;
  if (op.initialize(args, nullptr, stream) != cutlass::Status::kSuccess)
    return cudaErrorInitializationError;
  if constexpr (Prepare) {
    status = prepare_mxfp8_transpose(d.gradient, w.lhs, w.sfa, g.m, g.k, g.m,
        args.mainloop.layout_SFA, info, stream);
    if (status != cudaSuccess) return status;
    status = prepare_mxfp8_transpose(d.input, w.rhs.b, w.rhs.sfb, g.n, g.k, g.n,
        args.mainloop.layout_SFB, info, stream);
    if (status != cudaSuccess) return status;
  }
  if (op.run(stream) != cutlass::Status::kSuccess) return cudaErrorLaunchFailure;
  return cudaGetLastError();
}

template <int EpilogueN, bool Prepare = true>
cudaError_t backward_mxfp8_weight_impl(const Mxfp8WeightGradientOperands& d,
                                     void* workspace, cudaStream_t stream) {
  // beta=0 starts a local gradient: no old C values participate. A source-free
  // epilogue also removes their compile-time SMEM/pipeline requirements.
  // Accumulating calls keep the original BF16 C path, including beta!=1.
  // Dispatch is scalar semantics, never a shape-specific performance choice.
  if (d.beta == 0.f)
    return backward_mxfp8_weight_kernel<EpilogueN, Prepare, void>(d, workspace, stream);
  return backward_mxfp8_weight_kernel<EpilogueN, Prepare, Bf16>(d, workspace, stream);
}

template <int EpilogueN, bool Prepare = true>
cudaError_t oproj_backward_mxfp8_weight_impl(const Mxfp8OprojBackwardWeightParams& p,
                                          cudaStream_t stream) {
  auto status = validate_mxfp8_backward_weight(p);
  if (status != cudaSuccess) return status;
  const auto& d = p.projection;
  return backward_mxfp8_weight_impl<EpilogueN, Prepare>(
      {mxfp8_backward_weight_problem(d, p.gemm_tuning), d.grad_output,
       d.saved_attention, d.grad_weight, d.alpha, d.beta}, p.workspace, stream);
}

bool mxfp8_qkv_backward_weight_dimensions(const QkvBackwardWeightParams& p) {
  const int64_t heads = int64_t{p.q_heads} + 2LL * p.kv_heads;
  return p.local_tokens > 0 && p.local_tokens % 128 == 0 &&
      p.hidden > 0 && p.hidden % 128 == 0 && p.head_dim == 128 &&
      p.q_heads > 0 && p.kv_heads > 0 && p.q_heads % p.kv_heads == 0 &&
      heads * p.head_dim <= INT32_MAX - 127;
}

GemmProblem mxfp8_qkv_backward_weight_problem(const QkvBackwardWeightParams& p,
                                             const BackwardGemmTuning& t = {}) {
  GemmProblem g{(p.q_heads + 2 * p.kv_heads) * p.head_dim,
                p.hidden, p.local_tokens, 1};
  g.max_swizzle_size = t.max_swizzle_size;
  g.raster = t.along_m ? GemmRaster::kAlongM : GemmRaster::kAlongN;
  return g;
}

cudaError_t validate_mxfp8_qkv_backward_weight(const Mxfp8QkvBackwardWeightParams& p) {
  const auto& d = p.projection;
  if (!mxfp8_qkv_backward_weight_dimensions(d) || !mxfp8_backward_tuning(p.gemm_tuning) ||
      !std::isfinite(d.alpha) || !std::isfinite(d.beta) ||
      !mxfp8_aligned(d.dqkv_staging, 16) || !mxfp8_aligned(d.saved_input, 16) ||
      !mxfp8_aligned(d.grad_weight, 16) || !mxfp8_aligned(p.workspace))
    return cudaErrorInvalidValue;
  const auto g = mxfp8_qkv_backward_weight_problem(d, p.gemm_tuning);
  if (p.workspace_bytes < Mxfp8BackwardWeightWorkspace::make(g, nullptr).bytes)
    return cudaErrorInvalidValue;
  return cudaSuccess;
}

template <int EpilogueN, bool Prepare = true>
cudaError_t qkv_backward_mxfp8_weight_impl(const Mxfp8QkvBackwardWeightParams& p,
                                        cudaStream_t stream) {
  auto status=validate_mxfp8_qkv_backward_weight(p);
  if(status!=cudaSuccess)return status;
  const auto& d=p.projection;
  const auto g=mxfp8_qkv_backward_weight_problem(d,p.gemm_tuning);
  return backward_mxfp8_weight_impl<EpilogueN, Prepare>(
      {g, d.dqkv_staging, d.saved_input, d.grad_weight, d.alpha, d.beta}, p.workspace, stream);
}

bool mxfp8_qkv_backward_data_dimensions(const QkvBackwardDataParams& p) {
  auto basic=p;
  basic.gemm_tuning.epilogue_n=0;
  basic.epoch=1;  // A workspace-size query does not consume a launch epoch.
  return backward_shape_supported(basic) && mxfp8_backward_tuning(p.gemm_tuning) &&
      (p.gemm_policy==BackwardGemmPolicy::kAuto || p.gemm_policy==BackwardGemmPolicy::kM128N256) &&
      p.hidden%128==0 && p.head_dim==128 && p.kv_heads>0 &&
      p.q_heads%p.kv_heads==0 && p.kv_heads%p.world_size==0 &&
      (int64_t{p.q_heads}+2LL*p.kv_heads)*128<=INT32_MAX-127 &&
      (p.local_tokens/p.batch)%(p.causal_load_balanced?256:128)==0;
}

GemmProblem mxfp8_qkv_backward_data_problem(const QkvBackwardDataParams& p) {
  GemmProblem g{p.local_tokens,p.hidden,(p.q_heads+2*p.kv_heads)*128,1};
  g.max_swizzle_size=p.gemm_tuning.max_swizzle_size;
  g.raster=p.gemm_tuning.along_m?GemmRaster::kAlongM:GemmRaster::kAlongN;
  return g;
}

cudaError_t validate_mxfp8_qkv_backward_data(const Mxfp8QkvBackwardDataParams& p) {
  const auto& d=p.projection;
  if(!mxfp8_qkv_backward_data_dimensions(d) || d.num_comm_ctas<=0 || d.epoch==0 ||
      !mxfp8_aligned(d.weight,16) || !mxfp8_aligned(d.grad_input,16) ||
      !mxfp8_aligned(d.peer_dqkv_staging[d.rank],16) ||
      !mxfp8_aligned(d.peer_ready[d.rank],4) || !mxfp8_aligned(p.workspace))
    return cudaErrorInvalidValue;
  const auto g=mxfp8_qkv_backward_data_problem(d);
  if(p.workspace_bytes<Mxfp8BackwardWeightWorkspace::make(g,nullptr).bytes)
    return cudaErrorInvalidValue;
  for(int peer=0;peer<d.world_size;++peer) {
    const auto& a=p.peer_input[peer];
    const GemmProblem q{d.local_tokens*d.world_size,d.hidden,d.q_heads/d.world_size*128,1};
    const GemmProblem kv{q.m,q.n,d.kv_heads/d.world_size*128,1};
    if(!valid_mxfp8_activation(q,a.grad_q) || !valid_mxfp8_activation(kv,a.grad_k) ||
        !valid_mxfp8_activation(kv,a.grad_v) || !mxfp8_aligned(a.master_q,16) ||
        !mxfp8_aligned(a.master_k,16) || !mxfp8_aligned(a.master_v,16))
      return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

template <int EpilogueN, bool Prepare = true, bool WaitInput = true>
cudaError_t qkv_backward_mxfp8_data_impl(const Mxfp8QkvBackwardDataParams& p,cudaStream_t stream) {
  static_assert(WaitInput || !Prepare,"Bare GEMM is only valid for completed B scratch");
  auto status=validate_mxfp8_qkv_backward_data(p);
  if(status!=cudaSuccess)return status;
  DeviceInfo info{};
  status=device_info(&info);
  if(status!=cudaSuccess)return status;
  const auto& d=p.projection;
  if(d.num_comm_ctas>=info.sm_count)return cudaErrorInvalidValue;
  const auto g=mxfp8_qkv_backward_data_problem(d);
  const auto w=Mxfp8BackwardWeightWorkspace::make(g,p.workspace);
  using Types=Mxfp8GemmFamily<256,128,EpilogueN>;
  // Logical K heads, not physical CP peers. Keep the existing backward
  // system-scope full-head protocol and issuing-thread async-proxy ordering.
  // W transpose/quantization is a preceding stream operation, not a
  // concurrent panel producer. Use the plain collective beneath input-ready
  // adaptation: no weight-panel predicate is needed at each head load.
  using ReadyMainloop=detail::A2ALhsReadyMainloop<
      typename Types::Mainloop,typename Types::TileShape
#if FUSE_ENABLE_PROFILING
      ,false
#endif
      ,true,true,1>;
  using Mainloop=std::conditional_t<WaitInput,ReadyMainloop,typename Types::Mainloop>;
  using Gemm=cutlass::gemm::kernel::GemmUniversal<ProblemShape,Mainloop,
      typename Types::Epilogue,detail::MonolithicPersistentScheduler>;
  using Comm=Mxfp8QkvBackwardPullComm<!Prepare>;
  using Base=detail::MonolithicGemm<Gemm,Comm>;
  using Kernel=detail::InputProductionKernel<Base,Comm>;
  typename Kernel::Arguments args{};
  args.gemm=gemm_arguments<Gemm>(g,w.lhs,w.rhs.b,d.grad_input,d.alpha,
      d.num_comm_ctas,info,GemmRaster::kAlongN);
  auto& main=args.gemm.mainloop;
  main.ptr_SFA=w.sfa;main.ptr_SFB=w.rhs.sfb;
  main.layout_SFA=Mxfp8ScaleConfig::tile_atom_to_shape_SFA(args.gemm.problem_shape);
  main.layout_SFB=Mxfp8ScaleConfig::tile_atom_to_shape_SFB(args.gemm.problem_shape);
  if constexpr (WaitInput) {
    main.ready=d.peer_ready[d.rank];main.world_size=d.q_heads+2*d.kv_heads;
    main.m_tiles=d.local_tokens/128;main.arrivals_per_peer=1;main.k_tiles_per_peer=1;main.epoch=1;
  }
  auto& comm=args.comm;
  comm.params=d;comm.route=backward_route(d,true);comm.route.kv_heads=d.kv_heads;
  for(int peer=0;peer<d.world_size;++peer)comm.peer_input[peer]=p.peer_input[peer];
  comm.activation=w.lhs;comm.scales=w.sfa;comm.destination_scales=main.layout_SFA;
  for(int kind=0;kind<2;++kind)comm.source_scales[kind]=Mxfp8ScaleConfig::tile_atom_to_shape_SFA(
      cute::make_shape(d.local_tokens*d.world_size,d.hidden,
          (kind?d.kv_heads:d.q_heads)/d.world_size*128,1));
  comm.input_order=a2a_input_order<Gemm>(args.gemm);
  comm.input_order.ready_group_m_tiles=std::max(1,d.num_comm_ctas*8);
  args.num_comm_ctas=d.num_comm_ctas;
  if(!Kernel::can_implement(args) || Kernel::get_workspace_size(args))return cudaErrorNotSupported;
  if constexpr (Prepare)
    status=prepare_mxfp8_transpose(d.weight,w.rhs.b,w.rhs.sfb,g.n,g.k,g.n,
        main.layout_SFB,info,stream);
  return status==cudaSuccess?launch_monolithic<Kernel>(args,info,stream):status;
}

}  // namespace

cudaError_t qkv_backward_mxfp8_data_workspace_size(const QkvBackwardDataParams& p,size_t* bytes) {
  if(!bytes || !mxfp8_qkv_backward_data_dimensions(p))return cudaErrorInvalidValue;
  *bytes=Mxfp8BackwardWeightWorkspace::make(mxfp8_qkv_backward_data_problem(p),nullptr).bytes;
  return cudaSuccess;
}
cudaError_t launch_qkv_backward_mxfp8_data(const Mxfp8QkvBackwardDataParams& p,cudaStream_t stream) {
  return p.projection.gemm_tuning.epilogue_n==64
      ?qkv_backward_mxfp8_data_impl<64>(p,stream):qkv_backward_mxfp8_data_impl<32>(p,stream);
}
cudaError_t launch_qkv_backward_mxfp8_data_compute_reference(
    const Mxfp8QkvBackwardDataParams& p,cudaStream_t stream) {
  return p.projection.gemm_tuning.epilogue_n==64?qkv_backward_mxfp8_data_impl<64,false>(p,stream):
      qkv_backward_mxfp8_data_impl<32,false>(p,stream);
}

cudaError_t launch_qkv_backward_mxfp8_data_gemm_reference(
    const Mxfp8QkvBackwardDataParams& p,cudaStream_t stream) {
  return p.projection.gemm_tuning.epilogue_n==64?qkv_backward_mxfp8_data_impl<64,false,false>(p,stream):
      qkv_backward_mxfp8_data_impl<32,false,false>(p,stream);
}

cudaError_t qkv_backward_mxfp8_weight_workspace_size(const QkvBackwardWeightParams& p, size_t* bytes) {
  if (!bytes || !mxfp8_qkv_backward_weight_dimensions(p)) return cudaErrorInvalidValue;
  *bytes = Mxfp8BackwardWeightWorkspace::make(mxfp8_qkv_backward_weight_problem(p), nullptr).bytes;
  return cudaSuccess;
}
cudaError_t launch_qkv_backward_mxfp8_weight(const Mxfp8QkvBackwardWeightParams& p, cudaStream_t stream) {
  return p.gemm_tuning.epilogue_n == 64
      ? qkv_backward_mxfp8_weight_impl<64>(p, stream) : qkv_backward_mxfp8_weight_impl<32>(p, stream);
}
cudaError_t launch_qkv_backward_mxfp8_weight_compute_reference(
    const Mxfp8QkvBackwardWeightParams& p,cudaStream_t stream) {
  return p.gemm_tuning.epilogue_n==64?qkv_backward_mxfp8_weight_impl<64,false>(p,stream):
      qkv_backward_mxfp8_weight_impl<32,false>(p,stream);
}
cudaError_t launch_qkv_backward_mxfp8(const Mxfp8QkvBackwardParams& p,cudaStream_t stream) {
  if(p.weight_mode!=WeightGradientMode::kImmediate && p.weight_mode!=WeightGradientMode::kDeferred)
    return cudaErrorInvalidValue;
  auto status=validate_mxfp8_qkv_backward_data(p.data);
  if(status!=cudaSuccess)return status;
  if(p.weight_mode==WeightGradientMode::kImmediate) {
    const auto& d=p.data.projection;const auto& w=p.weight.projection;
    if(d.local_tokens!=w.local_tokens || d.hidden!=w.hidden || d.q_heads!=w.q_heads ||
        d.kv_heads!=w.kv_heads || d.head_dim!=w.head_dim ||
        d.peer_dqkv_staging[d.rank]!=w.dqkv_staging)return cudaErrorInvalidValue;
    status=validate_mxfp8_qkv_backward_weight(p.weight);
    if(status!=cudaSuccess)return status;
  }
  status=launch_qkv_backward_mxfp8_data(p.data,stream);
  return status==cudaSuccess && p.weight_mode==WeightGradientMode::kImmediate
      ?launch_qkv_backward_mxfp8_weight(p.weight,stream):status;
}

cudaError_t oproj_backward_mxfp8_data_workspace_size(const OprojBackwardDataParams& p, size_t* bytes) {
  if (!bytes || !mxfp8_backward_dimensions(p)) return cudaErrorInvalidValue;
  *bytes = Mxfp8Workspace::make({p.local_tokens, p.q_heads * p.head_dim, p.hidden, 1}).bytes;
  return cudaSuccess;
}
cudaError_t oproj_backward_mxfp8_weight_workspace_size(const OprojBackwardWeightParams& p, size_t* bytes) {
  if (!bytes || !mxfp8_backward_dimensions(p)) return cudaErrorInvalidValue;
  *bytes = Mxfp8BackwardWeightWorkspace::make(mxfp8_backward_weight_problem(p), nullptr).bytes;
  return cudaSuccess;
}
cudaError_t launch_oproj_backward_mxfp8_data(const Mxfp8OprojBackwardDataParams& p, cudaStream_t stream) {
  return p.projection.gemm_tuning.epilogue_n == 64
      ? oproj_backward_mxfp8_data_impl<64>(p, stream) : oproj_backward_mxfp8_data_impl<32>(p, stream);
}
cudaError_t launch_oproj_backward_mxfp8_weight(const Mxfp8OprojBackwardWeightParams& p, cudaStream_t stream) {
  return p.gemm_tuning.epilogue_n == 64
      ? oproj_backward_mxfp8_weight_impl<64>(p, stream) : oproj_backward_mxfp8_weight_impl<32>(p, stream);
}
cudaError_t launch_oproj_backward_mxfp8_weight_compute_reference(
    const Mxfp8OprojBackwardWeightParams& p, cudaStream_t stream) {
  return p.gemm_tuning.epilogue_n == 64
      ? oproj_backward_mxfp8_weight_impl<64, false>(p, stream)
      : oproj_backward_mxfp8_weight_impl<32, false>(p, stream);
}
cudaError_t launch_oproj_backward_mxfp8(const Mxfp8OprojBackwardParams& p, cudaStream_t stream) {
  if (p.weight_mode != WeightGradientMode::kImmediate && p.weight_mode != WeightGradientMode::kDeferred)
    return cudaErrorInvalidValue;
  auto status = validate_mxfp8_backward_data(p.data);
  if (status != cudaSuccess) return status;
  if (p.weight_mode == WeightGradientMode::kImmediate) {
    const auto& d = p.data.projection;
    const auto& w = p.weight.projection;
    if (d.local_tokens != w.local_tokens || d.hidden != w.hidden ||
        d.q_heads != w.q_heads || d.head_dim != w.head_dim) return cudaErrorInvalidValue;
    status = validate_mxfp8_backward_weight(p.weight);
    if (status != cudaSuccess) return status;
  }
  status = launch_oproj_backward_mxfp8_data(p.data, stream);
  return status == cudaSuccess && p.weight_mode == WeightGradientMode::kImmediate
      ? launch_oproj_backward_mxfp8_weight(p.weight, stream) : status;
}

}  // namespace fuse
