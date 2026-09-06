// SPDX-License-Identifier: BSD-3-Clause
// Independent QKV-forward services, assembled after forward.cuh.
// Reuse the original signaling GEMM, scheduler and QKV route. Weight DQ,
// producer input preparation and cross-rank completion are caller-owned.

namespace fuse {
namespace {

// The production scheduler addresses a linear compute subgrid. As in the
// dgrad service reference, flatten CUTLASS's ordinary grid and preserve the
// fused reservation even though no communication CTAs run beside producers.
template <class Gemm, class FusedKernel>
struct QkvForwardProducerReference : Gemm {
  using SharedStorage = typename FusedKernel::SharedStorage;
  static constexpr int SharedStorageSize = sizeof(SharedStorage);
  static_assert(SharedStorageSize >= Gemm::SharedStorageSize);

  static dim3 get_grid_shape(const typename Gemm::Params& params) {
    const dim3 grid = Gemm::get_grid_shape(params);
    return dim3(grid.x * grid.y * grid.z, 1, 1);
  }
};

// A separate thin entry keeps its fused SMEM permission independent of the
// legacy copy reference, which still uses the smaller Comm-only reservation.
// The communication loop itself is exactly the production Comm::run path.
template <class Comm, class FusedKernel>
struct QkvForwardCopyReference {
  using Params = typename Comm::Params;
  using ClusterShape = typename FusedKernel::ClusterShape;
  using SharedStorage = typename FusedKernel::SharedStorage;
  static constexpr int MaxThreadsPerBlock = 384;
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static_assert(sizeof(SharedStorage) >= Comm::SharedStorageBytes);

  static dim3 get_grid_shape(const Params& params) {
    return dim3(params.params.num_comm_ctas, 1, 1);
  }
  static dim3 get_block_shape() { return dim3(384, 1, 1); }

  CUTLASS_DEVICE void operator()(const Params& params, char*) {
    Comm{}.run(params, static_cast<int32_t>(blockIdx.x),
               static_cast<int32_t>(gridDim.x), false);
  }
};

// Share production plan resolution between the launch and resource query.
// Full-grid mode changes the SM budget only after tile selection. Keeping
// this first reference family explicit avoids replacing an unsupported auto
// tile with a different GEMM under the same calibration label.
template <class Visitor>
cudaError_t visit_qkv_forward_reference(
    const GemmA2AParams& params,
    QkvForwardReference primitive,
    Visitor& visitor) {
  if (static_cast<int32_t>(primitive) < 0 ||
      static_cast<int32_t>(primitive) > 4 ||
      !supported_problem(params.gemm) ||
      params.route.kind != RouteKind::kQkvGqaPack ||
      params.route.direction != RouteDirection::kForward ||
      params.route.causal_load_balanced || params.route.qkv_peer_interleaved ||
      params.route.defer_v_a2a || params.route.batch != 1 || params.gemm.l != 1) {
    return cudaErrorInvalidValue;
  }
  QkvLaunchPlan plan{};
  cudaError_t status = cached_qkv_launch_plan(params, &plan);
  if (status != cudaSuccess) {
    return status;
  }
  GemmA2AParams resolved = params;
  resolved.num_comm_ctas = plan.num_comm_ctas;
  auto visit = [&](auto binding_tag) -> cudaError_t {
    using Binding = typename decltype(binding_tag)::type;
    if constexpr (!std::is_same_v<Binding, QkvForwardN256Binding>) {
      return cudaErrorNotSupported;
    } else {
      using Gemm = typename Binding::Gemm;
      using Comm = typename Binding::Comm;
      static_assert(cute::size<0>(typename Gemm::TileShape{}) == 128 &&
                    cute::size<1>(typename Gemm::TileShape{}) == 256 &&
                    cute::size<2>(typename Gemm::TileShape{}) == 64 &&
                    cute::size<0>(typename Gemm::ClusterShape{}) == 2);
      static_assert(Gemm::CollectiveMainloop::DispatchPolicy::Stages == 4);
      if (resolved.num_comm_ctas % 2 != 0 ||
          (plan.sm_count - resolved.num_comm_ctas) % 2 != 0) {
        return cudaErrorInvalidConfiguration;
      }
      typename Comm::Arguments comm{};
      comm.params = resolved;
      if (!Comm::can_implement(comm)) {
        return cudaErrorNotSupported;
      }
      cudaError_t result = Comm::initialize(comm);
      if (result != cudaSuccess) {
        return result;
      }
      if (!comm.use_tma) {
        return cudaErrorNotSupported;
      }
      return visitor(binding_tag, resolved, comm, plan.sm_count, plan.device);
    }
  };
  return visit_qkv_forward_policy(plan.policy, false, visit);
}

// This is the GEMM argument contract of launch_gemm_a2a_impl. The reference
// changes only the compute budget, the removed communication-prefix offset,
// and (for the no-signal control) the epilogue ready pointer. In particular,
// original [N,K] BF16 weight rows and the production raster are retained.
template <class Gemm>
typename Gemm::Arguments make_qkv_forward_producer_arguments(
    const GemmA2AParams& params,
    int32_t compute_ctas,
    int32_t device,
    bool signaling) {
  typename Gemm::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape =
      make_shape(params.gemm.m, params.gemm.n, params.gemm.k, params.gemm.l);
  args.mainloop.ptr_A = params.lhs;
  args.mainloop.dA = make_stride(
      a_row_stride(params.gemm), _1{}, a_batch_stride(params.gemm));
  args.mainloop.ptr_B = params.rhs_nt;
  args.mainloop.dB = make_stride(
      b_row_stride(params.gemm), _1{}, b_batch_stride(params.gemm));
  args.epilogue.thread.alpha = params.alpha;
  args.epilogue.thread.beta = 0.0f;
  args.epilogue.ptr_C = nullptr;
  args.epilogue.dC = make_stride(
      d_row_stride(params.gemm), _1{}, d_batch_stride(params.gemm));
  args.epilogue.ptr_D = params.local_output;
  args.epilogue.dD = make_stride(
      d_row_stride(params.gemm), _1{}, d_batch_stride(params.gemm));
  args.epilogue.ready = signaling ? params.ready : nullptr;
  args.epilogue.m_tiles = ceil_div(
      params.gemm.m, cute::size<0>(typename Gemm::TileShape{}));
  args.epilogue.n_tiles = ceil_div(
      params.gemm.n, cute::size<1>(typename Gemm::TileShape{}));
  args.epilogue.epoch = params.epoch;
  args.hw_info.device_id = device;
  args.hw_info.sm_count = compute_ctas;
  args.scheduler.max_swizzle_size = params.gemm.max_swizzle_size;
  args.scheduler.block_offset = 0;
  args.scheduler.raster_order =
      raster_option(params.gemm.raster, RasterOptions::AlongM);
  return args;
}

template <class Binding, class Visitor>
cudaError_t visit_qkv_forward_producer_reference(
    const GemmA2AParams& params,
    QkvForwardReference primitive,
    int32_t sm_count,
    int32_t device,
    Visitor& visitor) {
  using Gemm = typename Binding::Gemm;
  using Kernel = QkvForwardProducerReference<Gemm, typename Binding::Kernel>;
  const bool full_grid =
      primitive == QkvForwardReference::kProducerSignalingFullgrid ||
      primitive == QkvForwardReference::kProducerNoSignalFullgrid;
  const bool signaling =
      primitive == QkvForwardReference::kProducerSignalingSubgrid ||
      primitive == QkvForwardReference::kProducerSignalingFullgrid;
  const int32_t compute_ctas = sm_count - (full_grid ? 0 : params.num_comm_ctas);
  constexpr int32_t cluster_m = cute::size<0>(typename Gemm::ClusterShape{});
  if (compute_ctas <= 0 || compute_ctas % cluster_m != 0) {
    return cudaErrorInvalidConfiguration;
  }
  const auto args = make_qkv_forward_producer_arguments<Gemm>(
      params, compute_ctas, device, signaling);
  if (!Kernel::can_implement(args) || Kernel::get_workspace_size(args) != 0) {
    return cudaErrorNotSupported;
  }
  return visitor(TypeTag<Kernel>{}, args);
}

template <class Kernel>
cudaError_t query_qkv_forward_reference_entry(
    const typename Kernel::Params& native,
    QkvForwardReferenceResources* resources) {
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes, cutlass::device_kernel<Kernel>);
  if (status != cudaSuccess) {
    return status;
  }
  resources->primitive_dynamic_smem_bytes = sizeof(typename Kernel::SharedStorage);
  resources->primitive_registers_per_thread = attributes.numRegs;
  resources->primitive_grid_x = Kernel::get_grid_shape(native).x;
  resources->primitive_launch_cluster_m =
      cute::size<0>(typename Kernel::ClusterShape{});
  resources->threads_per_cta = Kernel::get_block_shape().x;
  return cudaSuccess;
}

}  // namespace

cudaError_t launch_qkv_forward_reference(
    const GemmA2AParams& params,
    QkvForwardReference primitive,
    cudaStream_t stream) {
  auto launch = [&](auto binding_tag, const GemmA2AParams& resolved,
                    const auto& comm, int32_t sm_count, int32_t device) {
    using Binding = typename decltype(binding_tag)::type;
    using Comm = typename Binding::Comm;
    if (primitive == QkvForwardReference::kCopyFusedReservation) {
      using Kernel = QkvForwardCopyReference<Comm, typename Binding::Kernel>;
      return launch_regular<Kernel>(Comm::to_underlying_arguments(comm), stream);
    }
    auto compute = [&](auto kernel_tag, const auto& args) {
      using Kernel = typename decltype(kernel_tag)::type;
      if (Kernel::initialize_workspace(args, nullptr, stream) !=
          cutlass::Status::kSuccess) {
        return cudaErrorInitializationError;
      }
      return launch_regular<Kernel>(
          Kernel::to_underlying_arguments(args, nullptr), stream);
    };
    return visit_qkv_forward_producer_reference<Binding>(
        resolved, primitive, sm_count, device, compute);
  };
  return visit_qkv_forward_reference(params, primitive, launch);
}

cudaError_t query_qkv_forward_reference(
    const GemmA2AParams& params,
    QkvForwardReference primitive,
    QkvForwardReferenceResources* resources) {
  if (resources == nullptr) {
    return cudaErrorInvalidValue;
  }
  *resources = {};
  auto query = [&](auto binding_tag, const GemmA2AParams& resolved,
                   const auto& comm, int32_t sm_count, int32_t device) {
    using Binding = typename decltype(binding_tag)::type;
    using Gemm = typename Binding::Gemm;
    using Comm = typename Binding::Comm;
    resources->tile_m = cute::size<0>(typename Gemm::TileShape{});
    resources->tile_n = cute::size<1>(typename Gemm::TileShape{});
    resources->tile_k = cute::size<2>(typename Gemm::TileShape{});
    resources->selected_cluster_m = cute::size<0>(typename Gemm::ClusterShape{});
    resources->selected_gemm_stages = Gemm::CollectiveMainloop::DispatchPolicy::Stages;
    resources->ready_flag_stride = kReadyFlagStride;
    resources->ready_m_tiles = ceil_div(resolved.gemm.m, resources->tile_m);
    resources->ready_n_tiles = ceil_div(resolved.gemm.n, resources->tile_n);
    if (primitive == QkvForwardReference::kCopyFusedReservation) {
      using Kernel = QkvForwardCopyReference<Comm, typename Binding::Kernel>;
      resources->copy_use_tma = comm.use_tma;
      resources->copy_use_tma_store = comm.use_tma_store;
      resources->copy_slots = kQkvBulkSlots;
      return query_qkv_forward_reference_entry<Kernel>(
          Comm::to_underlying_arguments(comm), resources);
    }
    auto compute = [&](auto kernel_tag, const auto& args) {
      using Kernel = typename decltype(kernel_tag)::type;
      return query_qkv_forward_reference_entry<Kernel>(
          Kernel::to_underlying_arguments(args, nullptr), resources);
    };
    return visit_qkv_forward_producer_reference<Binding>(
        resolved, primitive, sm_count, device, compute);
  };
  return visit_qkv_forward_reference(params, primitive, query);
}

}  // namespace fuse
