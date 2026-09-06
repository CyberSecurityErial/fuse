// SPDX-License-Identifier: BSD-3-Clause
// Included by the test bridge after TestArguments and its ordinary launch.
// Diagnostic composition only; existing BF16 role telemetry remains the owner
// of compute/route/ready/finalize instrumentation. No production kernel here.
#if FUSE_ENABLE_PROFILING

namespace {
__global__ void mxfp8_profile_timer(uint64_t* output) {
  if (threadIdx.x == 0) *output = fuse::detail::read_global_timer();
}
}  // namespace

extern "C" uint64_t fuse_mxfp8_test_timeline_bytes() {
  return sizeof(fuse::A2AGemmCtaTimeline);
}

extern "C" int fuse_mxfp8_test_profile(
    const TestArguments* args, uint64_t timeline_address,
    int32_t timeline_capacity, uint64_t markers_address) {
  if (!args || !timeline_address || !markers_address || args->route_flags != 0) {
    return cudaErrorInvalidValue;
  }
  const auto& g = args->geometry;
  const auto& t = args->tensors;
  const int op = g[0], rank = g[1], world = g[2], m = g[3];
  const int h = g[4], q = g[5], kv = g[6], d = g[7];
  if (op < 0 || op > 3 || world < 2 || world > 8 || rank < 0 ||
      rank >= world || g[8] != 1 || g[12] != 0 || (op == 0 && g[9])) {
    return cudaErrorInvalidValue;
  }
  const int a = q * d, packed = (q + 2 * kv) * d;
  const bool is_qkv = op == 0 || op == 2;
  const int wr = is_qkv ? packed : h, wc = is_qkv ? h : a;
  const auto stream = reinterpret_cast<cudaStream_t>(t[12]);
  auto* timeline = pointer<fuse::A2AGemmCtaTimeline>(timeline_address);
  auto* markers = pointer<uint64_t>(markers_address);
  auto* weight = pointer<fuse::Bf16>(t[6]);

  mxfp8_profile_timer<<<1, 1, 0, stream>>>(markers);
  auto status = fuse::launch_mxfp8_weight_dequant(
      {pointer<const fuse::Fp8E4m3>(t[4]), pointer<const uint8_t>(t[5]), wr, wc},
      {weight, size_t(wr) * wc * sizeof(fuse::Bf16)}, stream);
  if (status != cudaSuccess) return status;
  mxfp8_profile_timer<<<1, 1, 0, stream>>>(markers + 1);

  fuse::UlyssesRoute route{};
  route.world_size = world;
  route.rank = rank;
  route.batch = 1;
  route.global_seq = m * world;
  route.seq_local = m;
  route.q_heads = q;
  route.kv_heads = kv;
  route.local_heads = q / world;
  route.head_dim = d;
  route.causal_load_balanced = g[9] != 0;

  if (op == 0) {
    fuse::GemmA2AParams p{};
    p.lhs = pointer<const fuse::Bf16>(t[0]);
    p.rhs_nt = weight;
    p.local_output = pointer<fuse::Bf16>(t[7]);
    p.ready = pointer<uint32_t>(t[10]);
    for (int peer = 0; peer < world; ++peer) {
      p.peer_output[peer] = pointer<fuse::Bf16>(args->peer_data[peer]);
      p.peer_route_done_epoch[peer] = pointer<uint32_t>(args->peer_done[peer]);
    }
    p.gemm = {m, packed, h, 1};
    p.gemm.raster = fuse::GemmRaster::kAlongN;
    p.route = route;
    p.route.kind = fuse::RouteKind::kQkvGqaPack;
    p.num_comm_ctas = mxfp8_test_qkv_forward_comm_request(p.gemm, p.route);
    p.epoch = g[10];
    p.alpha = args->alpha;
    status = fuse::launch_gemm_a2a_role_telemetry(p, timeline, timeline_capacity, stream);
  } else if (op == 1) {
    fuse::A2AGemmParams p{};
    for (int peer = 0; peer < world; ++peer) {
      p.peer_input[peer] = pointer<const fuse::Bf16>(args->peer_data[peer]);
    }
    p.rhs_nt = weight;
    p.input_staging = pointer<fuse::Bf16>(t[7]);
    p.output = pointer<fuse::Bf16>(t[8]);
    p.ready = pointer<uint32_t>(t[10]);
    p.gemm = {m, h, a, 1};
    p.route = route;
    p.route.kind = fuse::RouteKind::kHeadToSequence;
    p.route.direction = fuse::RouteDirection::kInverse;
    p.num_comm_ctas = mxfp8_test_oproj_comm_request(p.gemm, p.route);
    p.epoch = g[10];
    p.alpha = args->alpha;
    // CTA-role summary does not require a per-tile/per-peer trace array.
    // Null/zero is supported by both instrumented mainloop and publisher.
    status = fuse::launch_a2a_gemm_cutlass_role_telemetry(
        p, timeline, timeline_capacity, nullptr, 0, stream);
  } else if (op == 2) {
    fuse::QkvBackwardDataParams p{};
    p.grad_q = pointer<const fuse::Bf16>(t[0]);
    p.grad_k = pointer<const fuse::Bf16>(t[1]);
    p.grad_v = pointer<const fuse::Bf16>(t[2]);
    for (int peer = 0; peer < world; ++peer) {
      p.peer_dqkv_staging[peer] = pointer<fuse::Bf16>(args->peer_data[peer]);
      p.peer_ready[peer] = pointer<uint32_t>(args->peer_ready[peer]);
      p.peer_done_epoch[peer] = pointer<uint32_t>(args->peer_done[peer]);
    }
    p.weight = weight;
    p.grad_input = pointer<fuse::Bf16>(t[8]);
    p.local_tokens = m;
    p.hidden = h;
    p.q_heads = q;
    p.kv_heads = kv;
    p.head_dim = d;
    p.world_size = world;
    p.rank = rank;
    p.num_comm_ctas = mxfp8_test_comm_ctas;
    p.epoch = g[10];
    p.causal_load_balanced = g[9] != 0;
    p.alpha = args->alpha;
    status = fuse::launch_qkv_backward_data_role_telemetry(p, timeline, timeline_capacity, stream);
  } else {
    fuse::OprojBackwardDataParams p{};
    p.grad_output = pointer<const fuse::Bf16>(t[0]);
    p.weight = weight;
    p.local_grad_attention = pointer<fuse::Bf16>(t[7]);
    for (int peer = 0; peer < world; ++peer) {
      p.peer_grad_attention[peer] = pointer<fuse::Bf16>(args->peer_data[peer]);
      p.peer_done_epoch[peer] = pointer<uint32_t>(args->peer_done[peer]);
    }
    p.ready = pointer<uint32_t>(t[10]);
    p.local_tokens = m;
    p.hidden = h;
    p.q_heads = q;
    p.head_dim = d;
    p.world_size = world;
    p.rank = rank;
    p.num_comm_ctas = mxfp8_test_comm_ctas;
    p.epoch = g[10];
    p.causal_load_balanced = g[9] != 0;
    p.alpha = args->alpha;
    status = fuse::launch_oproj_backward_data_role_telemetry(p, timeline, timeline_capacity, stream);
  }
  if (status != cudaSuccess) return status;
  if (op >= 2) {
    mxfp8_profile_timer<<<1, 1, 0, stream>>>(markers + 2);
    auto weight_args = *args;
    weight_args.geometry[12] = 1;
    // Preserve the new FP32 dW and matching beta. The old BF16 W path is not
    // semantically interchangeable, even though its input matrices match.
    status = static_cast<cudaError_t>(fuse_mxfp8_test_launch(&weight_args));
    if (status != cudaSuccess) return status;
    mxfp8_profile_timer<<<1, 1, 0, stream>>>(markers + 3);
  }
  return cudaGetLastError();
}
#endif  // FUSE_ENABLE_PROFILING
