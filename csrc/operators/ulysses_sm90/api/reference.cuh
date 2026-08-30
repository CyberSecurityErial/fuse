// SPDX-License-Identifier: BSD-3-Clause
// Public API implementation; assembled by csrc/operators/ulysses_sm90.cu.
// Declarations: fuse/operators/primitives/a2a_gemm.h and gemm_a2a.h.
//
// Module index:
//   - BF16/FP8 independent GEMM reference launches
//   - A2A-input staging copy references
//   - QKV route copy references selected through production policy bindings

namespace fuse {

cudaError_t launch_batched_cutlass_reference(
    const GemmA2AParams& params,
    cudaStream_t stream,
    int32_t reserved_comm_ctas) {
  int32_t sm_count = 0;
  int32_t device = 0;
  cudaError_t status = device_sm_count(&sm_count, &device);
  if (status != cudaSuccess) {
    return status;
  }
  if (reserved_comm_ctas < 0 || reserved_comm_ctas >= sm_count ||
      params.route.kind != RouteKind::kQkvGqaPack ||
      params.route.direction != RouteDirection::kForward) {
    return cudaErrorInvalidValue;
  }

  const int32_t policy_comm_ctas = params.num_comm_ctas == 0
      ? recommended_gemm_a2a_comm_ctas(params.gemm, params.route)
      : params.num_comm_ctas;
  const QkvGemmPolicy policy = select_qkv_gemm_policy(
      params.gemm,
      policy_comm_ctas,
      sm_count,
      params.route.qkv_peer_interleaved);
  auto launch = [&](auto binding_tag) {
    using Binding = typename decltype(binding_tag)::type;
    return launch_gemm_reference_impl<typename Binding::PureGemm>(
        params,
        stream,
        sm_count,
        device,
        reserved_comm_ctas,
        RasterOptions::AlongN);
  };
  return visit_qkv_forward_policy(
      policy, params.route.qkv_peer_interleaved, launch);
}


cudaError_t launch_a2a_gemm_cutlass_reference(
    const A2AGemmParams& params,
    cudaStream_t stream,
    int32_t reserved_comm_ctas) {
  int32_t sm_count = 0;
  int32_t device = 0;
  cudaError_t status = device_sm_count(&sm_count, &device);
  if (status != cudaSuccess) {
    return status;
  }
  if (reserved_comm_ctas < 0 || reserved_comm_ctas >= sm_count ||
      !params.input_staging || !params.rhs_nt || !params.output ||
      !supported_problem(params.gemm)) {
    return cudaErrorInvalidValue;
  }

  const auto selected = select_a2a_lhs_policy_impl(
      params.gemm,
      params.num_comm_ctas,
      sm_count,
      params.lhs_policy);
  auto launch = [&](auto binding_tag) {
    using Binding = typename decltype(binding_tag)::type;
    constexpr int32_t cluster_ctas =
        cute::size<0>(typename Binding::Kernel::ClusterShape{}) *
        cute::size<1>(typename Binding::Kernel::ClusterShape{}) *
        cute::size<2>(typename Binding::Kernel::ClusterShape{});
    if (cluster_ctas > 1 &&
        (reserved_comm_ctas % cluster_ctas != 0 ||
         (sm_count - reserved_comm_ctas) % cluster_ctas != 0)) {
      return cudaErrorInvalidValue;
    }
    return launch_a2a_lhs_reference_impl<typename Binding::PureGemm>(
        params, stream, sm_count, device, reserved_comm_ctas);
  };
  return visit_oproj_forward_policy(selected.policy, launch);
}

cudaError_t launch_a2a_gemm_fp8_cutlass_reference(
    const Fp8A2AGemmParams& params,
    cudaStream_t stream,
    int32_t reserved_comm_ctas) {
  int32_t sm_count = 0;
  int32_t device = 0;
  cudaError_t status = device_sm_count(&sm_count, &device);
  if (status != cudaSuccess) {
    return status;
  }
  if (reserved_comm_ctas < 0 || reserved_comm_ctas >= sm_count ||
      !params.input_staging || !params.rhs_nt || !params.output ||
      !supported_fp8_problem(params.gemm)) {
    return cudaErrorInvalidValue;
  }
  if (selected_fp8_pipeline(params.gemm) ==
      Fp8PipelinePolicy::kCooperative) {
    return launch_a2a_lhs_reference_impl<Fp8CooperativePureGemm>(
        params, stream, sm_count, device, reserved_comm_ctas);
  }
  return launch_a2a_lhs_reference_impl<Fp8PureGemm>(
      params, stream, sm_count, device, reserved_comm_ctas);
}

template <class Gemm>
cudaError_t launch_dense_fp8_reference_impl(
    const Fp8GemmA2AParams& params,
    cudaStream_t stream,
    int32_t sm_count,
    int32_t device,
    int32_t reserved_comm_ctas) {
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
  args.hw_info.device_id = device;
  args.hw_info.sm_count = sm_count - reserved_comm_ctas;
  args.scheduler.max_swizzle_size = params.gemm.max_swizzle_size;
  args.scheduler.raster_order =
      raster_option(params.gemm.raster, RasterOptions::AlongM);
  if (!Gemm::can_implement(args) || Gemm::get_workspace_size(args) != 0) {
    return cudaErrorNotSupported;
  }
  if (Gemm::initialize_workspace(args, nullptr, stream) !=
      cutlass::Status::kSuccess) {
    return cudaErrorInitializationError;
  }
  return launch_regular<Gemm>(
      Gemm::to_underlying_arguments(args, nullptr), stream);
}

cudaError_t launch_dense_fp8_cutlass_reference(
    const Fp8GemmA2AParams& params,
    cudaStream_t stream,
    int32_t reserved_comm_ctas) {
  int32_t sm_count = 0;
  int32_t device = 0;
  cudaError_t status = device_sm_count(&sm_count, &device);
  if (status != cudaSuccess) {
    return status;
  }
  if (reserved_comm_ctas < 0 || reserved_comm_ctas >= sm_count ||
      !params.lhs || !params.rhs_nt || !params.local_output ||
      !supported_fp8_problem(params.gemm)) {
    return cudaErrorInvalidValue;
  }

  const Fp8TilePolicy tile = select_fp8_qkv_tile(
      params.gemm,
      params.route,
      reserved_comm_ctas,
      sm_count,
      device);
  if (tile == Fp8TilePolicy::kM128N64) {
    return launch_dense_fp8_reference_impl<Fp8N64PureGemm>(
        params, stream, sm_count, device, reserved_comm_ctas);
  }
  if (tile == Fp8TilePolicy::kM128N256ClusterM2) {
    return launch_dense_fp8_reference_impl<Fp8WidePureGemm>(
        params, stream, sm_count, device, reserved_comm_ctas);
  }
  if (selected_fp8_pipeline(params.gemm) ==
      Fp8PipelinePolicy::kCooperative) {
    return launch_dense_fp8_reference_impl<Fp8CooperativePureGemm>(
        params, stream, sm_count, device, reserved_comm_ctas);
  }
  return launch_dense_fp8_reference_impl<Fp8PureGemm>(
      params, stream, sm_count, device, reserved_comm_ctas);
}

template <class Comm, class ParamsType>
cudaError_t launch_a2a_lhs_copy_reference_impl(
    const ParamsType& params, cudaStream_t stream) {
    typename Comm::Arguments args{};
    args.params = params;
    cudaError_t status = Comm::initialize(args);
    if (status != cudaSuccess) {
      return status;
    }
    if (params.num_comm_ctas <= 0 ||
        !Comm::can_implement(args)) {
      return cudaErrorNotSupported;
    }
    constexpr size_t smem_bytes =
        kA2ALhsBulkSlots * kA2ALhsBulkStageBytes +
        kA2ALhsBulkSlots * sizeof(uint64_t);
    auto entry = a2a_lhs_input_copy_reference_kernel<Comm>;
    status = cudaFuncSetAttribute(
        entry,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes));
    if (status != cudaSuccess) {
      return status;
    }
    entry<<<
        params.num_comm_ctas, 384, smem_bytes, stream>>>(
            Comm::to_underlying_arguments(args));
    return cudaGetLastError();
}

cudaError_t launch_a2a_gemm_copy_reference(
    const A2AGemmParams& params,
    cudaStream_t stream) {
  int32_t sm_count = 0;
  int32_t device = 0;
  cudaError_t status = device_sm_count(&sm_count, &device);
  if (status != cudaSuccess) {
    return status;
  }
  const auto selected = select_a2a_lhs_policy_impl(
      params.gemm,
      params.num_comm_ctas,
      sm_count,
      params.lhs_policy);
  if (selected.policy == A2ALhsGemmPolicy::kM64N128) {
    return launch_a2a_lhs_copy_reference_impl<A2ALhsM64InputComm>(
        params, stream);
  }
  return launch_a2a_lhs_copy_reference_impl<A2ALhsInputComm>(
      params, stream);
}

cudaError_t launch_a2a_gemm_fp8_copy_reference(
    const Fp8A2AGemmParams& params,
    cudaStream_t stream) {
  if (params.num_comm_ctas <= 0) {
    return cudaErrorInvalidValue;
  }
  return launch_a2a_lhs_copy_reference_impl<Fp8A2ALhsInputComm>(
      params, stream);
}

cudaError_t launch_gemm_a2a_copy_reference(
    const GemmA2AParams& params,
    cudaStream_t stream) {
  if (params.num_comm_ctas <= 0 ||
      params.route.kind != RouteKind::kQkvGqaPack ||
      params.route.direction != RouteDirection::kForward) {
    return cudaErrorInvalidValue;
  }
  int32_t sm_count = 0;
  int32_t device = 0;
  cudaError_t status = device_sm_count(&sm_count, &device);
  if (status != cudaSuccess || params.num_comm_ctas >= sm_count) {
    return status == cudaSuccess ? cudaErrorInvalidValue : status;
  }
  const QkvGemmPolicy policy = select_qkv_gemm_policy(
      params.gemm,
      params.num_comm_ctas,
      sm_count,
      params.route.qkv_peer_interleaved);
  auto copy = [&](auto binding_tag) {
    using Binding = typename decltype(binding_tag)::type;
    typename Binding::Comm::Arguments args{};
    args.params = params;
    return launch_qkv_gqa_copy_reference<typename Binding::Comm>(
        args, params.num_comm_ctas, stream);
  };
  return visit_qkv_forward_policy(
      policy, params.route.qkv_peer_interleaved, copy);
}


cudaError_t launch_gemm_a2a_fp8_copy_reference(
    const Fp8GemmA2AParams& params,
    cudaStream_t stream) {
  if (params.num_comm_ctas <= 0 ||
      params.route.kind != RouteKind::kQkvGqaPack ||
      params.route.direction != RouteDirection::kForward) {
    return cudaErrorInvalidValue;
  }
  int32_t sm_count = 0;
  int32_t device = 0;
  cudaError_t status = device_sm_count(&sm_count, &device);
  if (status != cudaSuccess) {
    return status;
  }
  const Fp8TilePolicy tile = select_fp8_qkv_tile(
      params.gemm,
      params.route,
      params.num_comm_ctas,
      sm_count,
      device);
  if (tile == Fp8TilePolicy::kM128N64) {
    Fp8QkvGqaPackCommN64::Arguments args{};
    args.params = params;
    return launch_qkv_gqa_copy_reference<Fp8QkvGqaPackCommN64>(
        args, params.num_comm_ctas, stream);
  }
  if (tile == Fp8TilePolicy::kM128N256ClusterM2) {
    Fp8QkvGqaPackCommWide::Arguments args{};
    args.params = params;
    return launch_qkv_gqa_copy_reference<Fp8QkvGqaPackCommWide>(
        args, params.num_comm_ctas, stream);
  }
  Fp8QkvGqaPackComm::Arguments args{};
  args.params = params;
  return launch_qkv_gqa_copy_reference<Fp8QkvGqaPackComm>(
      args, params.num_comm_ctas, stream);
}

}  // namespace fuse
