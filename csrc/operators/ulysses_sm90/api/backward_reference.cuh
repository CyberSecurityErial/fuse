// SPDX-License-Identifier: BSD-3-Clause
// Independent QKV dgrad service references, assembled after backward.cuh.
// Reuse the production plan, RowMajor-B GEMM, head-ready adapter and route.
// No weight DQ, optimizer work or cross-rank finalization is performed here.

namespace fuse {
namespace {

// The monolithic scheduler interprets blockIdx.x as a linear compute index.
// CUTLASS's ordinary AlongN grid is two-dimensional; flatten it even when the
// communication prefix is absent. Keep the fused CTA's SMEM reservation so
// bare/ready comparisons do not silently change shared-memory occupancy.
template <class Gemm, class FusedKernel>
struct QkvBackwardComputeReference : Gemm {
  using SharedStorage = typename FusedKernel::SharedStorage;
  static constexpr int SharedStorageSize = sizeof(SharedStorage);
  static_assert(SharedStorageSize >= Gemm::SharedStorageSize);

  static dim3 get_grid_shape(const typename Gemm::Params& params) {
    const dim3 grid = Gemm::get_grid_shape(params);
    return dim3(grid.x * grid.y * grid.z, 1, 1);
  }
};

// Both launch and resource query use this exact production-plan resolution.
// Unsupported families are explicit: no geometry-name exceptions or silent
// replacement of the selected tile, including for full-grid references.
template <class Visitor>
cudaError_t visit_qkv_backward_reference(
    const QkvBackwardDataParams& params,
    QkvBackwardReference primitive,
    Visitor& visitor) {
  if (static_cast<int32_t>(primitive) < 0 ||
      static_cast<int32_t>(primitive) > 5) {
    return cudaErrorInvalidValue;
  }
  QkvBackwardKernelParams resolved{};
  int32_t sm_count = 0, device = 0;
  cudaError_t status = prepare_qkv_backward_data_launch(
      params, &resolved, &sm_count, &device);
  if (status != cudaSuccess) {
    return status;
  }
  auto visit = [&](auto binding_tag) -> cudaError_t {
    using Binding = typename decltype(binding_tag)::type;
    if constexpr (!std::is_same_v<Binding, QkvBackwardN256Binding>) {
      return cudaErrorNotSupported;
    } else {
      using Gemm = typename Binding::Gemm;
      using Comm = typename Binding::Comm;
      static_assert(cute::size<0>(typename Gemm::TileShape{}) == 128 &&
                    cute::size<1>(typename Gemm::TileShape{}) == 256 &&
                    cute::size<2>(typename Gemm::TileShape{}) == 64 &&
                    cute::size<0>(typename Gemm::ClusterShape{}) == 2);
      static_assert(Gemm::CollectiveMainloop::DispatchPolicy::Stages == 4);
      static_assert(kBlockM == 128 && kQkvBulkSlots == 12 &&
                    Comm::SharedStorageBytes == 196704);
      if (resolved.route.head_dim % 64 != 0) {
        return cudaErrorNotSupported;
      }
      typename Comm::Arguments comm{};
      comm.params = resolved;
      cudaError_t result = Comm::initialize(comm);
      if (result != cudaSuccess) {
        return result;
      }
      // The fitted copy service must be the actual twelve-slot TMA path,
      // not a vector-copy fallback under the same high-level shape label.
      if (!Comm::can_implement(comm) || !comm.use_tma) {
        return cudaErrorNotSupported;
      }
      return visitor(binding_tag, resolved, comm, sm_count, device);
    }
  };
  return visit_qkv_backward_policy(resolved.gemm_policy, visit);
}

bool is_qkv_backward_copy_reference(QkvBackwardReference primitive) {
  return primitive == QkvBackwardReference::kCopy ||
      primitive == QkvBackwardReference::kCopyFusedReservation;
}

struct QkvBackwardCopyReservation {
  int32_t smem_bytes;
  int32_t cluster_m;
};

// One launch contract for both resource reporting and kernel submission.
// This changes the reservation only: the copy entry and its twelve slots
// remain identical. Neither reference contains concurrent compute/finalize.
template <class Binding>
QkvBackwardCopyReservation qkv_backward_copy_reservation(
    QkvBackwardReference primitive) {
  if (primitive == QkvBackwardReference::kCopyFusedReservation) {
    using Kernel = typename Binding::Kernel;
    static_assert(sizeof(typename Kernel::SharedStorage) >=
                  Binding::Comm::SharedStorageBytes);
    return {sizeof(typename Kernel::SharedStorage),
            cute::size<0>(typename Kernel::ClusterShape{})};
  }
  return {Binding::Comm::SharedStorageBytes, 1};
}

template <class Binding, class Visitor>
cudaError_t visit_qkv_backward_compute_reference(
    const QkvBackwardKernelParams& params,
    QkvBackwardReference primitive,
    int32_t sm_count,
    int32_t device,
    Visitor& visitor) {
  const bool full_grid =
      primitive == QkvBackwardReference::kComputeBareFullgrid ||
      primitive == QkvBackwardReference::kComputeReadyPreloadedFullgrid;
  const bool with_ready =
      primitive == QkvBackwardReference::kComputeReadyPreloadedSubgrid ||
      primitive == QkvBackwardReference::kComputeReadyPreloadedFullgrid;
  const int32_t compute_ctas = sm_count - (full_grid ? 0 : params.num_comm_ctas);
  constexpr int32_t cluster_m =
      cute::size<0>(typename Binding::Gemm::ClusterShape{});
  if (compute_ctas <= 0 || compute_ctas % cluster_m != 0) {
    return cudaErrorInvalidConfiguration;
  }
  auto visit = [&](auto gemm_tag) -> cudaError_t {
    using Gemm = typename decltype(gemm_tag)::type;
    using Kernel = QkvBackwardComputeReference<Gemm, typename Binding::Kernel>;
    constexpr bool kWithReady = std::is_same_v<Gemm, typename Binding::Gemm>;
    const auto args = make_qkv_backward_gemm_arguments<Gemm, kWithReady>(
        params, compute_ctas, device, 0);
    if (!Kernel::can_implement(args) || Kernel::get_workspace_size(args) != 0) {
      return cudaErrorNotSupported;
    }
    return visitor(TypeTag<Kernel>{}, args);
  };
  if (with_ready) {
    return visit(TypeTag<typename Binding::Gemm>{});
  }
  return visit(TypeTag<typename Binding::PureGemm>{});
}

}  // namespace

cudaError_t launch_qkv_backward_reference(
    const QkvBackwardDataParams& params,
    QkvBackwardReference primitive,
    cudaStream_t stream) {
  auto launch = [&](auto binding_tag, const QkvBackwardKernelParams& resolved,
                    const auto& comm, int32_t sm_count, int32_t device) {
    using Binding = typename decltype(binding_tag)::type;
    using Comm = typename Binding::Comm;
    if (is_qkv_backward_copy_reference(primitive)) {
      const auto reservation = qkv_backward_copy_reservation<Binding>(primitive);
      const int32_t smem_bytes = reservation.smem_bytes;
      auto entry = a2a_lhs_input_copy_reference_kernel<Comm>;
      // Both references use one entry, so keep its function-level permission
      // stable across interleaved graph capture/replay. This is only the
      // allowed upper bound, not the allocation: primitive 1 still launches
      // with Comm::SharedStorageBytes, while primitive 5 reserves fused SMEM.
      constexpr size_t max_smem_bytes =
          sizeof(typename Binding::Kernel::SharedStorage);
      cudaError_t status = cudaFuncSetAttribute(
          entry, cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(max_smem_bytes));
      if (status != cudaSuccess) {
        return status;
      }
      // This existing wrapper calls Comm::operator() only. It drains the
      // route's TMA stores and publishes head-ready flags, but intentionally
      // does not call Comm::finalize() or wait for source-done acknowledgments.
      if (primitive == QkvBackwardReference::kCopyFusedReservation) {
        if (resolved.num_comm_ctas % reservation.cluster_m != 0) {
          return cudaErrorInvalidConfiguration;
        }
        cudaLaunchAttribute attribute{};
        attribute.id = cudaLaunchAttributeClusterDimension;
        attribute.val.clusterDim = {
            static_cast<unsigned int>(reservation.cluster_m), 1, 1};
        cudaLaunchConfig_t config{};
        config.gridDim = dim3(resolved.num_comm_ctas, 1, 1);
        config.blockDim = dim3(384, 1, 1);
        config.dynamicSmemBytes = smem_bytes;
        config.stream = stream;
        config.attrs = &attribute;
        config.numAttrs = 1;
        return cudaLaunchKernelEx(
            &config, entry, Comm::to_underlying_arguments(comm));
      }
      entry<<<resolved.num_comm_ctas, 384, smem_bytes, stream>>>(
          Comm::to_underlying_arguments(comm));
      return cudaGetLastError();
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
    return visit_qkv_backward_compute_reference<Binding>(
        resolved, primitive, sm_count, device, compute);
  };
  return visit_qkv_backward_reference(params, primitive, launch);
}

cudaError_t query_qkv_backward_reference(
    const QkvBackwardDataParams& params,
    QkvBackwardReference primitive,
    QkvBackwardReferenceResources* resources) {
  if (resources == nullptr) {
    return cudaErrorInvalidValue;
  }
  *resources = {};
  auto query = [&](auto binding_tag, const QkvBackwardKernelParams& resolved,
                   const auto& comm, int32_t sm_count, int32_t device) {
    using Binding = typename decltype(binding_tag)::type;
    using Gemm = typename Binding::Gemm;
    using Comm = typename Binding::Comm;
    resources->selected_gemm_stages = Gemm::CollectiveMainloop::DispatchPolicy::Stages;
    resources->ready_block_m = cute::size<0>(typename Gemm::TileShape{});
    resources->ready_flag_stride = kReadyFlagStride;
    resources->packed_heads = resolved.route.q_heads + 2 * resolved.route.kv_heads;
    resources->k_tiles_per_head = resolved.route.head_dim /
        cute::size<2>(typename Gemm::TileShape{});
    if (is_qkv_backward_copy_reference(primitive)) {
      const auto reservation = qkv_backward_copy_reservation<Binding>(primitive);
      cudaFuncAttributes attributes{};
      cudaError_t status = cudaFuncGetAttributes(
          &attributes, a2a_lhs_input_copy_reference_kernel<Comm>);
      if (status != cudaSuccess) {
        return status;
      }
      resources->primitive_dynamic_smem_bytes = reservation.smem_bytes;
      resources->primitive_registers_per_thread = attributes.numRegs;
      resources->primitive_grid_x = resolved.num_comm_ctas;
      resources->primitive_launch_cluster_m = reservation.cluster_m;
      resources->threads_per_cta = 384;
      resources->copy_use_tma = comm.use_tma;
      resources->copy_slots = kQkvBulkSlots;
      return cudaSuccess;
    }
    auto compute = [&](auto kernel_tag, const auto& args) {
      using Kernel = typename decltype(kernel_tag)::type;
      cudaFuncAttributes attributes{};
      cudaError_t status = cudaFuncGetAttributes(
          &attributes, cutlass::device_kernel<Kernel>);
      if (status != cudaSuccess) {
        return status;
      }
      const auto native = Kernel::to_underlying_arguments(args, nullptr);
      const dim3 grid = Kernel::get_grid_shape(native);
      resources->primitive_dynamic_smem_bytes = sizeof(typename Kernel::SharedStorage);
      resources->primitive_registers_per_thread = attributes.numRegs;
      resources->primitive_grid_x = grid.x;
      resources->primitive_launch_cluster_m =
          cute::size<0>(typename Kernel::ClusterShape{});
      resources->threads_per_cta = Kernel::get_block_shape().x;
      return cudaSuccess;
    };
    return visit_qkv_backward_compute_reference<Binding>(
        resolved, primitive, sm_count, device, compute);
  };
  return visit_qkv_backward_reference(params, primitive, query);
}

}  // namespace fuse
