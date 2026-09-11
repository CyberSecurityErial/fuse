// SPDX-License-Identifier: BSD-3-Clause
#pragma once

namespace fuse {
namespace {

struct Mxfp8ProjectionInput {
  Mxfp8GemmA2AParams params{};
  Mxfp8Workspace workspace{};
  bool dynamic_weight = true;
#if FUSE_ENABLE_PROFILING
  Mxfp8ProfileView probe{};
#endif

  void configure_communication(Mxfp8QkvGqaPackComm::Arguments& comm) const {
    const auto& g = params.projection.gemm;
    comm.weights.source = dynamic_weight ? params.projection.rhs_nt : nullptr;
    comm.weights.workspace = workspace;
    comm.weights.scales = Mxfp8ScaleConfig::tile_atom_to_shape_SFB(
        cute::make_shape(g.m, g.n, g.k, 1));
    comm.weights.n = g.n;
    comm.weights.k = g.k;
    comm.weights.row_stride = b_row_stride(g);
    comm.weights.epoch = params.projection.epoch;
    comm.weights.all_ctas = params.weight_preparation == Mxfp8WeightPreparation::kAllCtas;
    comm.weights.warp_specialized =
        params.weight_preparation == Mxfp8WeightPreparation::kCommunicationWarps;
#if FUSE_ENABLE_PROFILING
    comm.weights.probe = probe;
#endif
  }

  template <class Gemm>
  auto arguments(const GemmA2AParams& p, const DeviceInfo& info) const {
    GemmProblem packed = p.gemm;
    packed.stride_a.row = packed.k;
    packed.stride_b.row = packed.k;
    auto args = gemm_arguments<Gemm>(packed, params.activation.data, workspace.b, p.local_output,
        p.alpha, p.num_comm_ctas, info, GemmRaster::kAlongM);
    args.mainloop.ptr_SFA = reinterpret_cast<const cutlass::float_ue8m0_t*>(params.activation.scales);
    args.mainloop.ptr_SFB = workspace.sfb;
    args.mainloop.layout_SFA = Mxfp8ScaleConfig::tile_atom_to_shape_SFA(args.problem_shape);
    args.mainloop.layout_SFB = Mxfp8ScaleConfig::tile_atom_to_shape_SFB(args.problem_shape);
    args.mainloop.weight_ready = dynamic_weight ? workspace.ready : nullptr;
    args.mainloop.weight_panels = workspace.panels;
    args.mainloop.weight_epoch = p.epoch;
#if FUSE_ENABLE_PROFILING
    args.mainloop.weight_probe = probe;
#endif
    return args;
  }
};

inline bool mxfp8_aligned(const void* p, size_t alignment = 256) {
  return p && reinterpret_cast<uintptr_t>(p) % alignment == 0;
}

inline bool valid_mxfp8_activation(const GemmProblem& g, const Mxfp8Activation& a) {
  size_t data = 0, scales = 0;
  return gemm_a2a_mxfp8_activation_size(g, &data, &scales) == cudaSuccess &&
      mxfp8_aligned(a.data) && mxfp8_aligned(a.scales) &&
      a.data_bytes >= data && a.scale_bytes >= scales;
}

inline cudaError_t validate_mxfp8(const Mxfp8GemmA2AParams& p, Mxfp8Workspace* workspace) {
  if ((p.epilogue_n != 32 && p.epilogue_n != 64) ||
      !supported_mxfp8_problem(p.projection.gemm) || !std::isfinite(p.projection.alpha) ||
      !valid_mxfp8_activation(p.projection.gemm, p.activation) ||
      !mxfp8_aligned(p.projection.rhs_nt, 16) || !mxfp8_aligned(p.workspace) ||
      (p.weight_preparation != Mxfp8WeightPreparation::kCommunicationCtas &&
       p.weight_preparation != Mxfp8WeightPreparation::kAllCtas &&
       p.weight_preparation != Mxfp8WeightPreparation::kCommunicationWarps)) return cudaErrorInvalidValue;
  *workspace = Mxfp8Workspace::make(p.projection.gemm, p.workspace);
  if (p.workspace_bytes < workspace->bytes) return cudaErrorInvalidValue;
  Mxfp8QkvGqaPackComm::Arguments comm{};
  comm.params = p.projection;
  if (!Mxfp8QkvGqaPackComm::can_implement(comm)) return cudaErrorNotSupported;
  DeviceInfo info{};
  auto status = device_info(&info);
  if (status != cudaSuccess) return status;
  return p.projection.num_comm_ctas < info.sm_count ? cudaSuccess : cudaErrorInvalidValue;
}

template <bool Profile = false, int EpilogueN = 64>
cudaError_t launch_mxfp8(const Mxfp8GemmA2AParams& params, cudaStream_t stream, bool dynamic_weight
#if FUSE_ENABLE_PROFILING
    , A2AGemmCtaTimeline* timeline = nullptr, int32_t capacity = 0,
    Mxfp8ProfileView probe = {}, QkvRouteTimeline* routes = nullptr, int32_t route_capacity = 0
#endif
    ) {
  FUSE_SM103_HOST_BEGIN();
  // Production and telemetry resolve the same plan before pointer/epoch
  // validation and descriptor construction. The diagnostic prequantized path
  // has a different boundary: reuse a queried positive budget, never score it
  // as if weight quantization were still present.
  if (!dynamic_weight && params.projection.num_comm_ctas == 0) {
    FUSE_SM103_HOST_RETURN(cudaErrorNotSupported);
  }
  Mxfp8GemmA2AParams p{};
  auto status = resolve_mxfp8_qkv_communication(params, &p);
  if (status != cudaSuccess) FUSE_SM103_HOST_RETURN(status);
  Mxfp8Workspace workspace;
  status = validate_mxfp8(p, &workspace);
  if (status != cudaSuccess) FUSE_SM103_HOST_RETURN(status);
  using Binding = Mxfp8QkvBinding<EpilogueN>;
  // GEMM owns traversal: projection.gemm raster/swizzle feed both CUTLASS
  // and the communication producer_schedule. Weight quantization follows that
  // same N-panel order; output routing expands each produced tile into copies.
  //
  // GEMM tile order -> weight panel order -> complete N256 x K ready
  //                -> output tile order  -> BF16 route copies
  //
  // E32/E64 only partition epilogue stores inside an M128 x N256 tile.
  // SignalingEpilogue publishes after the full tile's stores complete: neither
  // weight readiness nor output readiness becomes a finer-grained protocol.
  // Communication CTA count changes worker ownership/stride, not ready units.
#if FUSE_ENABLE_PROFILING
  using Base = std::conditional_t<Profile,
      typename Binding::TelemetryKernel, typename Binding::Kernel>;
#else
  using Base = typename Binding::Kernel;
#endif
  using Kernel = detail::InputProductionKernel<Base, Mxfp8WeightProducer, Profile>;
  FUSE_SM103_HOST_RETURN((launch_gemm_a2a_impl<typename Binding::Gemm, Kernel,
      Mxfp8QkvGqaPackComm, Profile, false, Mxfp8ProjectionInput>(p.projection, stream,
#if FUSE_ENABLE_PROFILING
      timeline, capacity, nullptr, 0, routes, route_capacity,
#endif
      Mxfp8ProjectionInput{p, workspace, dynamic_weight
#if FUSE_ENABLE_PROFILING
          , probe
#endif
      })));
}

// The generic reference launcher enforces a production SMEM floor. These
// primitive services require exact equality as well, so a different reference
// occupancy/layout cannot quietly become the fused kernel's calibration.
template <class Kernel>
cudaError_t launch_mxfp8_reference(const typename Kernel::Params& params,
    const DeviceInfo& info, int32_t budget, cudaStream_t stream) {
  size_t production_smem = 0, reference_smem = 0;
  auto status = detail::launch_shared_memory<typename Kernel::ProductionKernel>(info, &production_smem);
  if (status != cudaSuccess) return status;
  status = detail::launch_shared_memory<Kernel>(info, &reference_smem);
  if (status != cudaSuccess) return status;
  if (production_smem != reference_smem) return cudaErrorInvalidConfiguration;
  return detail::launch_reference_cooperative<Kernel>(params, info, budget, stream);
}

template <int EpilogueN>
cudaError_t launch_mxfp8_compute_reference(
    const Mxfp8GemmA2AParams& p, cudaStream_t stream) {
  if (p.projection.num_comm_ctas == 0) return cudaErrorNotSupported;
  if (p.projection.num_comm_ctas < 0) return cudaErrorInvalidValue;
  Mxfp8Workspace workspace;
  auto status = validate_mxfp8(p, &workspace);
  if (status != cudaSuccess) return status;
  DeviceInfo info{};
  status = device_info(&info);
  if (status != cudaSuccess) return status;
  using Binding = Mxfp8QkvBinding<EpilogueN>;
  // This is OutputGemm, deliberately NOT PureGemm: the measured computation
  // retains the production epilogue's full global-store drain and ready
  // publication. The weight adapter remains present with its ready pointer
  // disabled, since operand preparation is outside this timed boundary.
  using Gemm = typename Binding::Gemm;
  using Production = detail::InputProductionKernel<typename Binding::Kernel, Mxfp8WeightProducer>;
  using Kernel = detail::GemmReferenceKernel<Gemm, Production>;
  const Mxfp8ProjectionInput input{p, workspace, false};
  auto args = input.template arguments<Gemm>(p.projection, info);
  args.scheduler.block_offset = 0;
#if FUSE_SM103_QKV_RANK_SWIZZLE
  // The ordinary MXFP8 route uses TMA for this aligned head_dim=128 domain,
  // and rotates the GEMM and copy mappings together when the option is built.
  args.scheduler.n_band_rank = p.projection.route.rank;
#endif
  args.epilogue.ready = p.projection.ready;
  args.epilogue.m_tiles = ceil_div(p.projection.gemm.m, Binding::Comm::kBlockM);
  args.epilogue.n_tiles = ceil_div(p.projection.gemm.n, Binding::Comm::kBlockN);
  args.epilogue.epoch = p.projection.epoch;
  if (!Gemm::can_implement(args) || Gemm::get_workspace_size(args) != 0) return cudaErrorNotSupported;
  if (Gemm::initialize_workspace(args, nullptr, stream) != cutlass::Status::kSuccess) {
    return cudaErrorInitializationError;
  }
  const auto lowered = Gemm::to_underlying_arguments(args, nullptr);
  if (lowered.scheduler.block_offset != 0) return cudaErrorInvalidConfiguration;
  return launch_mxfp8_reference<Kernel>(
      lowered, info, info.sm_count - p.projection.num_comm_ctas, stream);
}

template <int EpilogueN>
cudaError_t launch_mxfp8_copy_reference(
    const Mxfp8GemmA2AParams& p, cudaStream_t stream) {
  const auto& projection = p.projection;
  if (projection.num_comm_ctas == 0) return cudaErrorNotSupported;
  if (projection.num_comm_ctas < 0 || projection.epoch == 0 ||
      (p.epilogue_n != 32 && p.epilogue_n != 64) ||
      !supported_mxfp8_problem(projection.gemm) || !std::isfinite(projection.alpha)) {
    return cudaErrorInvalidValue;
  }
  if (projection.route.defer_v_a2a || projection.route.head_dim != 128) return cudaErrorNotSupported;
  DeviceInfo info{};
  auto status = device_info(&info);
  if (status != cudaSuccess) return status;
  if (projection.num_comm_ctas >= info.sm_count) return cudaErrorInvalidValue;
  using Binding = Mxfp8QkvBinding<EpilogueN>;
  using Production = detail::InputProductionKernel<typename Binding::Kernel, Mxfp8WeightProducer>;
  // Route-only has no weight producer at all. Reuse exactly the BF16 output
  // route that the MXFP8 comm adapter inherits, with the MXFP8 resource floor.
  using Comm = QkvGqaPackCommWide;
  using Kernel = detail::CopyReferenceKernel<Comm, Production>;
  static_assert(Comm::kNeedsGridFinalize);
  typename Comm::Arguments comm{};
  comm.params = projection;
  if (comm.params.gemm.raster == GemmRaster::kHeuristic) comm.params.gemm.raster = GemmRaster::kAlongM;
  status = Comm::initialize(comm, false); // MXFP8 A/W are not BF16 route operands.
  if (status != cudaSuccess) return status;
  return launch_mxfp8_reference<Kernel>(
      Comm::to_underlying_arguments(comm), info, projection.num_comm_ctas, stream);
}

template <int EpilogueN>
cudaError_t launch_mxfp8_quantize_reference(
    const Mxfp8GemmA2AParams& p, cudaStream_t stream) {
  if (p.projection.num_comm_ctas == 0) return cudaErrorNotSupported;
  if (p.projection.num_comm_ctas < 0) return cudaErrorInvalidValue;
  if (p.weight_preparation != Mxfp8WeightPreparation::kCommunicationCtas) return cudaErrorNotSupported;
  Mxfp8Workspace workspace;
  auto status = validate_mxfp8(p, &workspace);
  if (status != cudaSuccess) return status;
  DeviceInfo info{};
  status = device_info(&info);
  if (status != cudaSuccess) return status;
  using Binding = Mxfp8QkvBinding<EpilogueN>;
  using Production = detail::InputProductionKernel<typename Binding::Kernel, Mxfp8WeightProducer>;
  using Kernel = detail::InputReferenceKernel<
      Mxfp8QkvGqaPackComm::Arguments, Mxfp8WeightProducer, Production>;
  typename Kernel::Params args{};
  args.params = p.projection;
  if (args.params.gemm.raster == GemmRaster::kHeuristic) args.params.gemm.raster = GemmRaster::kAlongM;
  const Mxfp8ProjectionInput input{p, workspace, true};
  input.configure_communication(args);
  // Resolve only the shared queue mapping. Quantization has no TMA descriptors
  // to encode and does not need to initialize the output communication route.
  args.producer_order = Mxfp8QkvGqaPackComm::producer_schedule(args.params);
#if FUSE_SM103_QKV_RANK_SWIZZLE
  args.n_band_swizzle = detail::NBandSwizzle::make(args.producer_order, args.params.route.rank);
#endif
  return launch_mxfp8_reference<Kernel>(args, info, p.projection.num_comm_ctas, stream);
}

#if FUSE_ENABLE_PROFILING
template <class Schedule>
std::vector<uint8_t> mxfp8_recovery_panels(
    const Schedule& schedule, int m_tiles, int panels) {
  // bit 0: some CTA's first valid tile uses this panel.
  // bit 1: some CTA first computes another panel, then uses this one.
  // AlongN may put EVERY panel in the global first wave, yet still supplies
  // steady-state consumers through other CTAs. Only their already-completed
  // prior outputs qualify for recovery-latency measurement in the exporter.
  std::vector<uint8_t> result(panels, 0);
  if (schedule.compute_grid_size == 0) return result;
  for (uint64_t worker = 0; worker < schedule.compute_grid_size; ++worker) {
    int first_panel = -1;
    for (uint64_t linear = worker; linear < schedule.blocks_per_problem_;
         linear += schedule.compute_grid_size) {
      const auto tile = detail::ProducerTileOrder::decode(
          schedule, linear, schedule.n_band_swizzle);
      if (!tile.valid || tile.m < 0 || tile.m >= m_tiles || tile.n < 0 || tile.n >= panels)
        continue;
      if (first_panel < 0) {
        first_panel = tile.n;
        result[tile.n] |= 1;
      } else if (tile.n != first_panel) {
        result[tile.n] |= 2;
      }
    }
  }
  return result;
}

template <int EpilogueN>
cudaError_t mxfp8_service_resources(
    const Mxfp8GemmA2AParams& p, Mxfp8ServiceResources* resources) {
  if (!resources || p.projection.num_comm_ctas < 0) return cudaErrorInvalidValue;
  if (p.projection.num_comm_ctas == 0 ||
      p.weight_preparation != Mxfp8WeightPreparation::kCommunicationCtas ||
      p.projection.route.defer_v_a2a) return cudaErrorNotSupported;
  Mxfp8Workspace workspace;
  auto status = validate_mxfp8(p, &workspace);
  if (status != cudaSuccess) return status;
  DeviceInfo info{};
  status = device_info(&info);
  if (status != cudaSuccess) return status;
  using Binding = Mxfp8QkvBinding<EpilogueN>;
  using Gemm = typename Binding::Gemm;
  using Mainloop = typename Binding::Types::Mainloop;
  using Production = detail::InputProductionKernel<typename Binding::Kernel, Mxfp8WeightProducer>;
  size_t smem = 0;
  status = detail::launch_shared_memory<Production>(info, &smem);
  if (status != cudaSuccess) return status;
  const Mxfp8ProjectionInput input{p, workspace, true};
  auto args = input.template arguments<Gemm>(p.projection, info);
#if FUSE_SM103_QKV_RANK_SWIZZLE
  args.scheduler.n_band_rank = p.projection.route.rank;
#endif
  using Scheduler = typename Gemm::TileScheduler;
  const auto schedule = Scheduler::to_underlying_arguments(args.problem_shape,
      typename Gemm::TileShape{}, typename Gemm::AtomThrShapeMNK{},
      typename Gemm::ClusterShape{}, args.hw_info, args.scheduler);
  *resources = {Mainloop::DispatchPolicy::Stages, static_cast<int32_t>(smem),
      static_cast<int32_t>(schedule.compute_grid_size), -1, 1 << schedule.log_swizzle_size_};
  const auto recovery = mxfp8_recovery_panels(schedule, ceil_div(p.projection.gemm.m, 128), workspace.panels);
  for (int panel = 0; panel < workspace.panels; ++panel) {
    if (recovery[panel] == 2) {
      resources->delayed_panel = panel;
      break;
    }
  }
  if (resources->delayed_panel < 0) {
    for (int panel = 0; panel < workspace.panels; ++panel) {
      if (recovery[panel] & 2) { resources->delayed_panel = panel; break; }
    }
  }
  return cudaSuccess;
}

template <int EpilogueN>
cudaError_t launch_mxfp8_service(const Mxfp8GemmA2AParams& p,
    const Mxfp8ServiceConfig& config, Mxfp8ServiceView view, cudaStream_t stream) {
  Mxfp8ServiceResources resources{};
  auto status = mxfp8_service_resources<EpilogueN>(p, &resources);
  if (status != cudaSuccess) return status;
  const bool compute = config.mode == Mxfp8ServiceMode::kCompute;
  const bool route = config.mode == Mxfp8ServiceMode::kRoute ||
      config.mode == Mxfp8ServiceMode::kQuantizeRoute;
  if (!compute && !route && config.mode != Mxfp8ServiceMode::kQuantize)
    return cudaErrorInvalidValue;
  if (config.quant_phase_steps < 0 || config.quant_phase_steps > 1 ||
      (config.quant_phase_steps != 0 && config.mode != Mxfp8ServiceMode::kQuantizeRoute))
    return cudaErrorInvalidValue;
  if ((!compute && (config.delayed_panel != -1 || config.delay_ns != 0)) ||
      config.delayed_panel < -1 || (config.delayed_panel == -1 && config.delay_ns != 0) ||
      (config.delayed_panel >= 0 && config.delay_ns == 0)) return cudaErrorInvalidValue;
  const Mxfp8Workspace workspace = Mxfp8Workspace::make(p.projection.gemm, p.workspace);
  const int64_t tiles = int64_t{ceil_div(p.projection.gemm.m, 128)} * workspace.panels;
  const int64_t quant_steps = int64_t{workspace.panels} * (p.projection.gemm.k / 4);
  const int ctas = p.projection.num_comm_ctas + (compute ? resources.compute_ctas : 0);
  if (view.tile_capacity < 0 || view.cta_capacity < 0 || view.panel_capacity < 0 ||
      view.weight.quant_capacity < 0 || view.weight.wait_capacity < 0 || view.route_capacity < 0 ||
      (view.tiles && view.tile_capacity < tiles) ||
      (view.ctas && view.cta_capacity < ctas) ||
      ((view.panel_release || view.panel_release_begin) && view.panel_capacity < workspace.panels) ||
      (view.weight.quant && view.weight.quant_capacity < quant_steps) ||
      (view.weight.waits && view.weight.wait_capacity < tiles)) return cudaErrorInvalidValue;
  if (route && view.routes) {
    const uint64_t slots = Mxfp8QkvGqaPackComm::route_slots(p.projection) +
        8 * uint64_t{static_cast<uint32_t>(p.projection.num_comm_ctas)};
    if (slots > static_cast<uint64_t>(view.route_capacity)) return cudaErrorInvalidValue;
  }
  DeviceInfo info{};
  status = device_info(&info);
  if (status != cudaSuccess) return status;
  using Binding = Mxfp8QkvBinding<EpilogueN>;
  using Production = detail::InputProductionKernel<typename Binding::Kernel, Mxfp8WeightProducer>;
  if (compute) {
    using Gemm = typename Binding::Types::ServiceGemm;
    using Kernel = detail::InputComputeReferenceKernel<Gemm, Production>;
    const Mxfp8ProjectionInput input{p, workspace, true, view.weight};
    auto args = input.template arguments<Gemm>(p.projection, info);
#if FUSE_SM103_QKV_RANK_SWIZZLE
    args.scheduler.n_band_rank = p.projection.route.rank;
#endif
    args.mainloop.service = view;
    args.epilogue.service = view;
    args.epilogue.ready = p.projection.ready;
    args.epilogue.m_tiles = ceil_div(p.projection.gemm.m, 128);
    args.epilogue.n_tiles = workspace.panels;
    args.epilogue.epoch = p.projection.epoch;
    if (!Gemm::can_implement(args) || Gemm::get_workspace_size(args) != 0)
      return cudaErrorNotSupported;
    if (Gemm::initialize_workspace(args, nullptr, stream) != cutlass::Status::kSuccess)
      return cudaErrorInitializationError;
    typename Kernel::Params lowered{};
    lowered.gemm = Gemm::to_underlying_arguments(args, nullptr);
    if (config.delayed_panel >= workspace.panels) return cudaErrorInvalidValue;
    if (config.delayed_panel >= 0) {
      const auto recovery = mxfp8_recovery_panels(
          lowered.gemm.scheduler, args.epilogue.m_tiles, workspace.panels);
      if (!(recovery[config.delayed_panel] & 2)) return cudaErrorInvalidValue;
    }
    lowered.weight_ready = workspace.ready;
    lowered.weight_arrivals = workspace.arrivals;
    lowered.panels = workspace.panels;
    lowered.comm_ctas = p.projection.num_comm_ctas;
    lowered.compute_ctas = static_cast<int32_t>(lowered.gemm.scheduler.compute_grid_size);
    lowered.epoch = p.projection.epoch;
    lowered.config = config;
    lowered.service = view;
    if (lowered.gemm.scheduler.block_offset != lowered.comm_ctas ||
        lowered.compute_ctas != resources.compute_ctas) return cudaErrorInvalidConfiguration;
    return launch_mxfp8_reference<Kernel>(lowered, info, info.sm_count, stream);
  }
  using Comm = Mxfp8QkvGqaPackComm;
  using Kernel = detail::InputCopyReferenceKernel<Comm, Mxfp8WeightProducer, Production>;
  typename Kernel::Params args{};
  args.comm.params = p.projection;
  if (args.comm.params.gemm.raster == GemmRaster::kHeuristic)
    args.comm.params.gemm.raster = GemmRaster::kAlongM;
  const bool quantize = config.mode != Mxfp8ServiceMode::kRoute;
  const Mxfp8ProjectionInput input{p, workspace, quantize, view.weight};
  input.configure_communication(args.comm);
  if (route) {
    status = Comm::initialize(args.comm);
    if (status != cudaSuccess) return status;
    if (!args.comm.use_tma || !args.comm.use_tma_store) return cudaErrorNotSupported;
  } else {
    args.comm.producer_order = Comm::producer_schedule(args.comm.params);
#if FUSE_SM103_QKV_RANK_SWIZZLE
    args.comm.n_band_swizzle = detail::NBandSwizzle::make(
        args.comm.producer_order, args.comm.params.route.rank);
#endif
  }
  args.route = route;
  args.quant_phase_steps = config.quant_phase_steps;
  args.service = view;
  return launch_mxfp8_reference<Kernel>(args, info, p.projection.num_comm_ctas, stream);
}
#endif

}  // namespace

KernelTraits mxfp8_qkv_cutlass_kernel_traits(int32_t epilogue_n) {
  if (epilogue_n == 32) return kernel_traits<detail::InputProductionKernel<
      Mxfp8QkvBinding<32>::Kernel, Mxfp8WeightProducer>>();
  if (epilogue_n == 64) return kernel_traits<detail::InputProductionKernel<
      Mxfp8QkvBinding<64>::Kernel, Mxfp8WeightProducer>>();
  return {};
}

cudaError_t gemm_a2a_mxfp8_activation_size(
    const GemmProblem& g, size_t* data_bytes, size_t* scale_bytes) {
  if (!data_bytes || !scale_bytes || !supported_mxfp8_problem(g)) return cudaErrorInvalidValue;
  *data_bytes = Mxfp8Workspace::align(size_t{static_cast<uint32_t>(g.m)} * g.k);
  *scale_bytes = Mxfp8Workspace::align(cute::size(cute::filter_zeros(
      Mxfp8ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(g.m, g.n, g.k, 1)))));
  return cudaSuccess;
}

cudaError_t quantize_gemm_a2a_mxfp8_activation(const GemmProblem& g,
    const Bf16* source, const Mxfp8Activation& output, cudaStream_t stream) {
  if (!valid_mxfp8_activation(g, output) || !mxfp8_aligned(source, 16)) return cudaErrorInvalidValue;
  quantize_mxfp8_operand<<<256, 256, 0, stream>>>(source,
      const_cast<Fp8E4m3*>(output.data),
      reinterpret_cast<cutlass::float_ue8m0_t*>(const_cast<uint8_t*>(output.scales)),
      g.m, g.k, a_row_stride(g),
      Mxfp8ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(g.m, g.n, g.k, 1)));
  return cudaGetLastError();
}

cudaError_t gemm_a2a_mxfp8_workspace_size(const GemmProblem& problem, size_t* bytes) {
  if (!bytes || !supported_mxfp8_problem(problem)) return cudaErrorInvalidValue;
  *bytes = Mxfp8Workspace::make(problem).bytes;
  return cudaSuccess;
}

cudaError_t prepare_gemm_a2a_mxfp8(const Mxfp8GemmA2AParams& p, cudaStream_t stream) {
  Mxfp8Workspace w;
  auto status = validate_mxfp8(p, &w);
  if (status != cudaSuccess) return status;
  const auto& g = p.projection.gemm;
  quantize_mxfp8_operand<<<256, 256, 0, stream>>>(p.projection.rhs_nt, w.b, w.sfb,
      g.n, g.k, b_row_stride(g),
      Mxfp8ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(g.m, g.n, g.k, 1)));
  return cudaGetLastError();
}

cudaError_t launch_gemm_a2a_mxfp8_prequantized(
    const Mxfp8GemmA2AParams& params, cudaStream_t stream) {
  return params.epilogue_n == 32 ? launch_mxfp8<false, 32>(params, stream, false)
                                : launch_mxfp8<false, 64>(params, stream, false);
}

cudaError_t launch_gemm_a2a_mxfp8_cutlass(
    const Mxfp8GemmA2AParams& params, cudaStream_t stream) {
  return params.epilogue_n == 32 ? launch_mxfp8<false, 32>(params, stream, true)
                                : launch_mxfp8<false, 64>(params, stream, true);
}

cudaError_t launch_gemm_a2a_mxfp8_compute_reference(
    const Mxfp8GemmA2AParams& params, cudaStream_t stream) {
  return params.epilogue_n == 32 ? launch_mxfp8_compute_reference<32>(params, stream)
                                : launch_mxfp8_compute_reference<64>(params, stream);
}

cudaError_t launch_gemm_a2a_mxfp8_copy_reference(
    const Mxfp8GemmA2AParams& params, cudaStream_t stream) {
  return params.epilogue_n == 32 ? launch_mxfp8_copy_reference<32>(params, stream)
                                : launch_mxfp8_copy_reference<64>(params, stream);
}

cudaError_t launch_gemm_a2a_mxfp8_quantize_reference(
    const Mxfp8GemmA2AParams& params, cudaStream_t stream) {
  return params.epilogue_n == 32 ? launch_mxfp8_quantize_reference<32>(params, stream)
                                : launch_mxfp8_quantize_reference<64>(params, stream);
}

#if FUSE_ENABLE_PROFILING
cudaError_t launch_gemm_a2a_mxfp8_service(const Mxfp8GemmA2AParams& params,
    const Mxfp8ServiceConfig& config, Mxfp8ServiceView view, cudaStream_t stream) {
  return params.epilogue_n == 32 ? launch_mxfp8_service<32>(params, config, view, stream)
                                : launch_mxfp8_service<64>(params, config, view, stream);
}

cudaError_t query_gemm_a2a_mxfp8_service_resources(
    const Mxfp8GemmA2AParams& params, Mxfp8ServiceResources* resources) {
  return params.epilogue_n == 32 ? mxfp8_service_resources<32>(params, resources)
                                : mxfp8_service_resources<64>(params, resources);
}

cudaError_t launch_gemm_a2a_mxfp8_role_telemetry(
    const Mxfp8GemmA2AParams& params, A2AGemmCtaTimeline* timeline,
    int32_t capacity, cudaStream_t stream, Mxfp8ProfileView probe,
    QkvRouteTimeline* routes, int32_t route_capacity) {
  return params.epilogue_n == 32
      ? launch_mxfp8<true, 32>(params, stream, true, timeline, capacity, probe, routes, route_capacity)
      : launch_mxfp8<true, 64>(params, stream, true, timeline, capacity, probe, routes, route_capacity);
}
#endif

}  // namespace fuse
