// SPDX-License-Identifier: BSD-3-Clause
// Test-only decomposition of QKV/OProj forward and QKV backward. Use the BF16
// reference kernels with the same selected tile and reserved compute grid.
// Weight DQ and staging initialization are intentionally caller-owned here.
namespace {

cudaError_t make_qkv_forward_reference_params(
    const TestArguments* args, fuse::GemmA2AParams* params) {
  if (!args || !params || args->geometry[0] != 0 ||
      args->geometry[8] != 1 || args->geometry[9] != 0 ||
      args->route_flags != 0) {
    return cudaErrorInvalidValue;
  }
  const auto& g = args->geometry;
  const auto& t = args->tensors;
  if (g[2] < 2 || g[2] > 8 || g[1] < 0 || g[1] >= g[2]) {
    return cudaErrorInvalidValue;
  }
  *params = fuse::GemmA2AParams{};
  auto& p = *params;
  p.lhs = pointer<const fuse::Bf16>(t[0]);
  p.rhs_nt = pointer<const fuse::Bf16>(t[6]);
  p.local_output = pointer<fuse::Bf16>(t[7]);
  p.ready = pointer<uint32_t>(t[10]);
  for (int peer = 0; peer < g[2]; ++peer) {
    p.peer_output[peer] = pointer<fuse::Bf16>(args->peer_data[peer]);
    p.peer_route_done_epoch[peer] = pointer<uint32_t>(args->peer_done[peer]);
  }
  p.gemm = {g[3], (g[5] + 2 * g[6]) * g[7], g[4], 1};
  p.gemm.raster = fuse::GemmRaster::kAlongN;
  p.route.kind = fuse::RouteKind::kQkvGqaPack;
  p.route.direction = fuse::RouteDirection::kForward;
  p.route.world_size = g[2];
  p.route.rank = g[1];
  p.route.batch = 1;
  p.route.global_seq = g[3] * g[2];
  p.route.seq_local = g[3];
  p.route.q_heads = g[5];
  p.route.kv_heads = g[6];
  p.route.local_heads = g[5] / g[2];
  p.route.head_dim = g[7];
  p.num_comm_ctas = mxfp8_test_qkv_forward_comm_request(p.gemm, p.route);
  p.epoch = g[10];
  p.alpha = args->alpha;
  return cudaSuccess;
}

cudaError_t make_oproj_reference_params(
    const TestArguments* args, fuse::A2AGemmParams* params) {
  if (!args || !params || args->geometry[0] != 1 ||
      args->geometry[8] != 1 || args->route_flags != 0) {
    return cudaErrorInvalidValue;
  }
  const auto& g = args->geometry;
  const auto& t = args->tensors;
  if (g[2] < 2 || g[2] > 8 || g[1] < 0 || g[1] >= g[2]) {
    return cudaErrorInvalidValue;
  }
  *params = fuse::A2AGemmParams{};
  auto& p = *params;
  for (int peer = 0; peer < g[2]; ++peer) {
    p.peer_input[peer] = pointer<const fuse::Bf16>(args->peer_data[peer]);
  }
  p.rhs_nt = pointer<fuse::Bf16>(t[6]);
  p.input_staging = pointer<fuse::Bf16>(t[7]);
  p.output = pointer<fuse::Bf16>(t[8]);
  p.ready = pointer<uint32_t>(t[10]);
  p.gemm = {g[3], g[4], g[5] * g[7], 1};
  p.gemm.raster = fuse::GemmRaster::kAlongN;
  p.route.kind = fuse::RouteKind::kHeadToSequence;
  p.route.direction = fuse::RouteDirection::kInverse;
  p.route.world_size = g[2];
  p.route.rank = g[1];
  p.route.batch = 1;
  p.route.global_seq = g[3] * g[2];
  p.route.seq_local = g[3];
  p.route.q_heads = g[5];
  p.route.kv_heads = g[6];
  p.route.local_heads = g[5] / g[2];
  p.route.head_dim = g[7];
  p.route.causal_load_balanced = g[9] != 0;
  p.num_comm_ctas = mxfp8_test_oproj_comm_request(p.gemm, p.route);
  if (p.num_comm_ctas == 0) {
    p.num_comm_ctas = fuse::recommended_a2a_lhs_gemm_comm_ctas(p.gemm, p.route);
  }
  p.epoch = g[10];
  p.alpha = args->alpha;
  return cudaSuccess;
}

cudaError_t make_qkv_reference_params(
    const TestArguments* args, fuse::QkvBackwardDataParams* params) {
  if (!args || !params || args->geometry[0] != 2 ||
      args->geometry[8] != 1 || args->route_flags != 0) {
    return cudaErrorInvalidValue;
  }
  const auto& g = args->geometry;
  const auto& t = args->tensors;
  if (g[2] < 2 || g[2] > 8 || g[1] < 0 || g[1] >= g[2]) {
    return cudaErrorInvalidValue;
  }
  *params = fuse::QkvBackwardDataParams{};
  auto& p = *params;
  p.grad_q = pointer<const fuse::Bf16>(t[0]);
  p.grad_k = pointer<const fuse::Bf16>(t[1]);
  p.grad_v = pointer<const fuse::Bf16>(t[2]);
  for (int peer = 0; peer < g[2]; ++peer) {
    p.peer_dqkv_staging[peer] = pointer<fuse::Bf16>(args->peer_data[peer]);
    p.peer_ready[peer] = pointer<uint32_t>(args->peer_ready[peer]);
    p.peer_done_epoch[peer] = pointer<uint32_t>(args->peer_done[peer]);
  }
  // DQ, destination-owned staging and ready prepublication are all prepared
  // outside the timed primitive. The original [QKV,H] weight is not transposed.
  p.weight = pointer<const fuse::Bf16>(t[6]);
  p.grad_input = pointer<fuse::Bf16>(t[8]);
  p.local_tokens = g[3];
  p.hidden = g[4];
  p.q_heads = g[5];
  p.kv_heads = g[6];
  p.head_dim = g[7];
  p.batch = 1;
  p.world_size = g[2];
  p.rank = g[1];
  p.num_comm_ctas = mxfp8_test_comm_ctas;
  p.epoch = g[10];
  p.causal_load_balanced = g[9] != 0;
  p.alpha = args->alpha;
  return cudaSuccess;
}

}  // namespace

// 0/1: same signaling GEMM entry with real/null ready, compute subgrid;
// 2/3: corresponding full-grid controls without reselecting the tile;
// 4: the original route loop without producer waits/finalize, fused reservation.
extern "C" int fuse_mxfp8_test_qkv_forward_reference(
    const TestArguments* args, int32_t primitive) {
  fuse::GemmA2AParams params{};
  const auto status = make_qkv_forward_reference_params(args, &params);
  if (status != cudaSuccess) return status;
  return fuse::launch_qkv_forward_reference(
      params, static_cast<fuse::QkvForwardReference>(primitive),
      reinterpret_cast<cudaStream_t>(args->tensors[12]));
}

// Independent sixteen-int diagnostic ABI. Existing forward policy and QKV-B
// resource ABIs are unchanged; all resource values come from the native plan.
extern "C" int fuse_mxfp8_test_qkv_forward_reference_config(
    const TestArguments* args, int32_t primitive, int32_t out[16]) {
  if (!out) return cudaErrorInvalidValue;
  fuse::GemmA2AParams params{};
  auto status = make_qkv_forward_reference_params(args, &params);
  if (status != cudaSuccess) return status;
  fuse::QkvForwardReferenceResources resources{};
  status = fuse::query_qkv_forward_reference(
      params, static_cast<fuse::QkvForwardReference>(primitive), &resources);
  if (status != cudaSuccess) return status;
  out[0] = resources.tile_m;
  out[1] = resources.tile_n;
  out[2] = resources.tile_k;
  out[3] = resources.selected_cluster_m;
  out[4] = resources.selected_gemm_stages;
  out[5] = resources.primitive_dynamic_smem_bytes;
  out[6] = resources.primitive_registers_per_thread;
  out[7] = resources.primitive_grid_x;
  out[8] = resources.primitive_launch_cluster_m;
  out[9] = resources.threads_per_cta;
  out[10] = resources.ready_flag_stride;
  out[11] = resources.ready_m_tiles;
  out[12] = resources.ready_n_tiles;
  out[13] = resources.copy_use_tma;
  out[14] = resources.copy_use_tma_store;
  out[15] = resources.copy_slots;
  return cudaSuccess;
}

// One native value: the ready-M window actually used by the matched route.
extern "C" int fuse_mxfp8_test_oproj_reference_config(
    const TestArguments* args, int32_t out[1]) {
  if (!out) return cudaErrorInvalidValue;
  fuse::A2AGemmParams params{};
  const auto status = make_oproj_reference_params(args, &params);
  if (status != cudaSuccess) return status;
  return fuse::a2a_gemm_comm_window(params, out);
}

extern "C" int fuse_mxfp8_test_oproj_reference(
    const TestArguments* args, int32_t primitive) {
  if (primitive < 0 || primitive > 2) return cudaErrorInvalidValue;
  fuse::A2AGemmParams p{};
  const auto status = make_oproj_reference_params(args, &p);
  if (status != cudaSuccess) return status;
  const auto stream = reinterpret_cast<cudaStream_t>(args->tensors[12]);
  if (primitive == 1) {
    return fuse::launch_a2a_gemm_copy_reference(p, stream, true);
  }
  // 0: exactly the compute subgrid available to the persistent fused kernel.
  // 2: same tile on all SMs, to measure the cost of reserving route CTAs.
  return fuse::launch_a2a_gemm_cutlass_reference(
      p, stream, primitive == 0 ? p.num_comm_ctas : 0);
}

// 0/2: bare/ready-preloaded compute subgrid; 1: copy without finalize;
// 3/4: corresponding full-grid compute, retaining the original selected tile.
// 5: the same copy kernel with the fused cluster and dynamic-SMEM reservation.
extern "C" int fuse_mxfp8_test_qkv_reference(
    const TestArguments* args, int32_t primitive) {
  fuse::QkvBackwardDataParams params{};
  const auto status = make_qkv_reference_params(args, &params);
  if (status != cudaSuccess) return status;
  return fuse::launch_qkv_backward_reference(
      params, static_cast<fuse::QkvBackwardReference>(primitive),
      reinterpret_cast<cudaStream_t>(args->tensors[12]));
}

// Fixed twelve-int diagnostic ABI, distinct from the existing policy query.
extern "C" int fuse_mxfp8_test_qkv_reference_config(
    const TestArguments* args, int32_t primitive, int32_t out[12]) {
  if (!out) return cudaErrorInvalidValue;
  fuse::QkvBackwardDataParams params{};
  auto status = make_qkv_reference_params(args, &params);
  if (status != cudaSuccess) return status;
  fuse::QkvBackwardReferenceResources resources{};
  status = fuse::query_qkv_backward_reference(
      params, static_cast<fuse::QkvBackwardReference>(primitive), &resources);
  if (status != cudaSuccess) return status;
  out[0] = resources.selected_gemm_stages;
  out[1] = resources.primitive_dynamic_smem_bytes;
  out[2] = resources.primitive_registers_per_thread;
  out[3] = resources.primitive_grid_x;
  out[4] = resources.primitive_launch_cluster_m;
  out[5] = resources.threads_per_cta;
  out[6] = resources.copy_use_tma;
  out[7] = resources.copy_slots;
  out[8] = resources.ready_block_m;
  out[9] = resources.ready_flag_stride;
  out[10] = resources.packed_heads;
  out[11] = resources.k_tiles_per_head;
  return cudaSuccess;
}
