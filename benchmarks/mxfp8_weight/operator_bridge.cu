// SPDX-License-Identifier: BSD-3-Clause
// Test-only raw-pointer ABI, matching backward_torch_bridge.cu. No PyTorch,
// cuBLAS, or benchmark implementation is linked into the production operator.
#include "fuse/fuse.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace {

// Test-only override, like explicit CTA options in the existing MPI benches.
// Zero preserves production auto dispatch; the production library owns no
// global tuning state and receives the resolved request through its params.
thread_local int32_t mxfp8_test_comm_ctas = 0;
thread_local fuse::Mxfp8WgradPolicy mxfp8_test_wgrad_policy =
    fuse::Mxfp8WgradPolicy::kAuto;
thread_local fuse::Mxfp8OprojCommModel mxfp8_test_oproj_comm_model{};
thread_local fuse::Mxfp8QkvForwardCommModel mxfp8_test_qkv_forward_comm_model{};

int32_t mxfp8_test_qkv_forward_comm_request(
    const fuse::GemmProblem& problem, const fuse::UlyssesRoute& route) {
  if (mxfp8_test_comm_ctas != 0) return mxfp8_test_comm_ctas;
  if (mxfp8_test_qkv_forward_comm_model.world_size != 0) {
    return fuse::select_mxfp8_qkv_forward_comm_ctas(
        problem, route, mxfp8_test_qkv_forward_comm_model);
  }
  return 0;  // Preserve production auto; the model is explicitly opt-in.
}

int32_t mxfp8_test_oproj_comm_request(
    const fuse::GemmProblem& problem, const fuse::UlyssesRoute& route) {
  if (mxfp8_test_comm_ctas != 0) return mxfp8_test_comm_ctas;
  if (mxfp8_test_oproj_comm_model.world_size != 0) {
    return fuse::select_mxfp8_oproj_comm_ctas(problem, route, mxfp8_test_oproj_comm_model);
  }
  return 0;  // Preserve production auto; the model is explicitly opt-in.
}

template <class T>
T* pointer(uint64_t address) {
  return reinterpret_cast<T*>(static_cast<uintptr_t>(address));
}

}  // namespace

// Same argument block for each dataflow, interpreted explicitly below.
// Geometry: op (0=QKV F,1=OProj F,2=QKV B,3=OProj B), rank, CP, M,
// H, Q heads, KV heads, D, reserved=1, causal, epoch, weight_mode, W-only.
struct TestArguments {
  int32_t geometry[13];
  // lhs/grad_q, grad_k, grad_v, saved_input, payload, scales, workspace,
  // local staging, output/dX, main_grad, ready, done, stream.
  uint64_t tensors[13];
  uint64_t peer_data[8];
  uint64_t peer_ready[8];
  uint64_t peer_done[8];
  float alpha;
  float beta;
  int32_t route_flags;
};

extern "C" int fuse_mxfp8_test_profiling_enabled() {
  return FUSE_ENABLE_PROFILING;
}

// Explicit caller calibration, never infer universal coefficients from CP.
// Values: compute TFLOP, compute tile, copy MiB, copy task-wave, prior, margin.
extern "C" int fuse_mxfp8_test_set_oproj_comm_model(
    int32_t world_size, int32_t sm_count, const double values[6]) {
  if (world_size == 0) {
    mxfp8_test_oproj_comm_model = {};
    return cudaSuccess;
  }
  if (!values || (world_size != 4 && world_size != 8) || sm_count != 132) {
    return cudaErrorInvalidValue;
  }
  for (int i = 0; i < 6; ++i) {
    if (!std::isfinite(values[i]) || values[i] < 0.0) return cudaErrorInvalidValue;
  }
  if (values[0] == 0.0 || values[2] == 0.0 || values[3] == 0.0 || values[5] >= 1.0) {
    return cudaErrorInvalidValue;
  }
  mxfp8_test_oproj_comm_model = {
      world_size, sm_count, values[0], values[1], values[2], values[3], values[4], values[5]};
  return cudaSuccess;
}

extern "C" int fuse_mxfp8_test_get_oproj_comm_model(double values[8]) {
  if (!values) return cudaErrorInvalidValue;
  const auto& model = mxfp8_test_oproj_comm_model;
  values[0] = model.world_size;
  values[1] = model.sm_count;
  values[2] = model.compute_flop_us;
  values[3] = model.compute_tile_us;
  values[4] = model.copy_mib_us;
  values[5] = model.copy_task_wave_us;
  values[6] = model.launch_prior_us;
  values[7] = model.minimum_gain;
  return cudaSuccess;
}

// Values: compute GFLOP/SM coefficient, route slot-task coefficient, margin.
extern "C" int fuse_mxfp8_test_set_qkv_forward_comm_model(
    int32_t world_size, int32_t sm_count, const double values[3]) {
  if (world_size == 0) {
    mxfp8_test_qkv_forward_comm_model = {};
    return cudaSuccess;
  }
  if (!values || world_size != 4 || sm_count != 132) return cudaErrorInvalidValue;
  for (int i = 0; i < 3; ++i) {
    if (!std::isfinite(values[i]) || values[i] < 0.0) return cudaErrorInvalidValue;
  }
  if (values[0] == 0.0 || values[1] == 0.0 || values[2] >= 1.0) {
    return cudaErrorInvalidValue;
  }
  mxfp8_test_qkv_forward_comm_model = {
      world_size, sm_count, values[0], values[1], values[2]};
  return cudaSuccess;
}

extern "C" int fuse_mxfp8_test_get_qkv_forward_comm_model(double values[5]) {
  if (!values) return cudaErrorInvalidValue;
  const auto& model = mxfp8_test_qkv_forward_comm_model;
  values[0] = model.world_size;
  values[1] = model.sm_count;
  values[2] = model.compute_gflop_sm_us;
  values[3] = model.route_slot_task_us;
  values[4] = model.minimum_gain;
  return cudaSuccess;
}

extern "C" int fuse_mxfp8_test_set_comm_ctas(int32_t count) {
  if (count < 0 || count % 2 != 0) return cudaErrorInvalidValue;
  if (count > 0) {
    int device = 0, sm = 0;
    auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) return status;
    status = cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, device);
    if (status != cudaSuccess) return status;
    if (count >= sm) return cudaErrorInvalidValue;
  }
  mxfp8_test_comm_ctas = count;
  return cudaSuccess;
}

extern "C" int fuse_mxfp8_test_set_wgrad_policy(int32_t policy) {
  const auto requested = static_cast<fuse::Mxfp8WgradPolicy>(policy);
  switch (requested) {
    case fuse::Mxfp8WgradPolicy::kAuto:
    case fuse::Mxfp8WgradPolicy::kM128N256K64ClusterM2:
    case fuse::Mxfp8WgradPolicy::kM128N128K64ClusterM2:
    case fuse::Mxfp8WgradPolicy::kM128N128K128ClusterM2:
    case fuse::Mxfp8WgradPolicy::kM128N256K64ClusterM1:
    case fuse::Mxfp8WgradPolicy::kM128N192K64ClusterM2:
    case fuse::Mxfp8WgradPolicy::kM128N256K32ClusterM2:
      mxfp8_test_wgrad_policy = requested;
      return cudaSuccess;
    default:
      return cudaErrorInvalidValue;
  }
}

// Separate W metadata ABI; the existing ten-int F/B query is unchanged.
// actual policy, M, N, K, cluster M, stages, dynamic SMEM, registers/thread.
extern "C" int fuse_mxfp8_test_wgrad_config(
    const TestArguments* args, int32_t out[8]) {
  if (!args || !out || (args->geometry[0] != 2 && args->geometry[0] != 3)) {
    return cudaErrorInvalidValue;
  }
  fuse::Mxfp8WgradKernelTraits traits{};
  const auto status = fuse::mxfp8_wgrad_kernel_traits(
      mxfp8_test_wgrad_policy, &traits);
  if (status != cudaSuccess) return status;
  out[0] = static_cast<int32_t>(traits.policy);
  out[1] = traits.block_m;
  out[2] = traits.block_n;
  out[3] = traits.block_k;
  out[4] = traits.cluster_m;
  out[5] = traits.stages;
  out[6] = traits.dynamic_smem_bytes;
  out[7] = traits.registers_per_thread;
  return cudaSuccess;
}

extern "C" int fuse_mxfp8_test_dequant(
    uint64_t payload, uint64_t scales, uint64_t output,
    int rows, int columns, uint64_t bytes, uint64_t stream) {
  return fuse::launch_mxfp8_weight_dequant(
      {pointer<const fuse::Fp8E4m3>(payload), pointer<const uint8_t>(scales),
       rows, columns},
      {pointer<fuse::Bf16>(output), static_cast<size_t>(bytes)},
      reinterpret_cast<cudaStream_t>(stream));
}

// Metadata only: use the same public selectors and explicit test overrides
// as the timed entry. This query does not mutate the active policy.
extern "C" int fuse_mxfp8_test_config(const TestArguments* args, int32_t* out) {
  if (!args || !out) return cudaErrorInvalidValue;
  const auto& g = args->geometry;
  const int op = g[0], m = g[3], h = g[4], q = g[5], kv = g[6], d = g[7];
  int device = 0, sm = 0;
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  status = cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, device);
  if (status != cudaSuccess) return status;
  fuse::KernelTraits traits{};
  int comm = 0, cluster = 1, policy = 0;
  if (op < 2) {
    fuse::GemmProblem problem{m, op == 0 ? (q + 2 * kv) * d : h,
                              op == 0 ? h : q * d, 1};
    problem.raster = fuse::GemmRaster::kAlongN;
    fuse::UlyssesRoute route{};
    route.world_size = g[2];
    route.rank = g[1];
    route.batch = 1;
    route.global_seq = m * g[2];
    route.seq_local = m;
    route.q_heads = q;
    route.kv_heads = kv;
    route.local_heads = q / g[2];
    route.head_dim = d;
    route.causal_load_balanced = g[9] != 0;
    if (op == 0) {
      route.kind = fuse::RouteKind::kQkvGqaPack;
      route.qkv_peer_interleaved = (args->route_flags & 1) != 0;
      route.cyclic_peer_order = (args->route_flags & 2) != 0;
      comm = mxfp8_test_qkv_forward_comm_request(problem, route);
      if (comm == 0) comm = fuse::recommended_gemm_a2a_comm_ctas(problem, route);
      traits = fuse::qkv_cutlass_kernel_traits(problem, route, comm, sm);
      cluster = traits.block_n >= 256 ? 2 : 1;
    } else {
      route.kind = fuse::RouteKind::kHeadToSequence;
      route.direction = fuse::RouteDirection::kInverse;
      comm = mxfp8_test_oproj_comm_request(problem, route);
      if (comm == 0) comm = fuse::recommended_a2a_lhs_gemm_comm_ctas(problem, route);
      const auto info = fuse::select_a2a_lhs_gemm_policy(problem, comm, sm);
      traits.block_m = info.tile_m;
      traits.block_n = info.tile_n;
      traits.block_k = info.tile_k;
      cluster = info.cluster_m;
      policy = static_cast<int>(info.policy);
    }
  } else if (op == 2) {
    fuse::QkvBackwardDataParams p{};
    p.local_tokens = m;
    p.hidden = h;
    p.q_heads = q;
    p.kv_heads = kv;
    p.head_dim = d;
    p.world_size = g[2];
    p.rank = g[1];
    p.causal_load_balanced = g[9] != 0;
    comm = mxfp8_test_comm_ctas ? mxfp8_test_comm_ctas :
        fuse::recommended_qkv_backward_comm_ctas(p);
    const auto selected = fuse::recommended_qkv_backward_gemm_policy(p, comm, sm);
    traits = fuse::qkv_backward_kernel_traits(p, comm, sm);
    policy = static_cast<int>(selected);
    cluster = selected == fuse::BackwardGemmPolicy::kM128N256 ||
                      selected == fuse::BackwardGemmPolicy::kM128N64ClusterM2 ? 2 : 1;
  } else if (op == 3) {
    fuse::OprojBackwardDataParams p{};
    p.local_tokens = m;
    p.hidden = h;
    p.q_heads = q;
    p.head_dim = d;
    p.world_size = g[2];
    p.rank = g[1];
    p.causal_load_balanced = g[9] != 0;
    comm = mxfp8_test_comm_ctas ? mxfp8_test_comm_ctas :
        fuse::recommended_oproj_backward_comm_ctas(p);
    const auto selected = fuse::recommended_oproj_backward_gemm_policy(p, comm, sm);
    traits = fuse::oproj_backward_kernel_traits(p, comm, sm);
    policy = static_cast<int>(selected);
    cluster = selected == fuse::BackwardGemmPolicy::kM128N256 ||
                      selected == fuse::BackwardGemmPolicy::kM128N64ClusterM2 ? 2 : 1;
  } else {
    return cudaErrorInvalidValue;
  }
  out[0] = comm;
  out[1] = traits.block_m;
  out[2] = traits.block_n;
  out[3] = traits.block_k;
  out[4] = cluster;
  out[5] = sm;
  out[6] = policy;
  const auto weight_traits = fuse::projection_cutlass_kernel_traits();
  out[7] = weight_traits.block_m;
  out[8] = weight_traits.block_n;
  out[9] = weight_traits.block_k;
  return cudaSuccess;
}

extern "C" int fuse_mxfp8_test_launch(const TestArguments* args) {
  if (!args) return cudaErrorInvalidValue;
  const auto& g = args->geometry;
  const auto& t = args->tensors;
  const int op = g[0], rank = g[1], world = g[2], m = g[3];
  const int h = g[4], q = g[5], kv = g[6], d = g[7];
  if (world < 2 || world > 8 || rank < 0 || rank >= world || g[8] != 1) {
    return cudaErrorInvalidValue;
  }
  const int a = q * d, packed = (q + 2 * kv) * d;
  const bool is_qkv = op == 0 || op == 2;
  const int wr = is_qkv ? packed : h, wc = is_qkv ? h : a;
  const fuse::Mxfp8Weight weight{
      pointer<const fuse::Fp8E4m3>(t[4]), pointer<const uint8_t>(t[5]), wr, wc};
  const fuse::Mxfp8WeightWorkspace workspace{
      pointer<fuse::Bf16>(t[6]), size_t(wr) * wc * sizeof(fuse::Bf16)};
  const auto stream = reinterpret_cast<cudaStream_t>(t[12]);
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
  route.qkv_peer_interleaved = (args->route_flags & 1) != 0;
  route.cyclic_peer_order = (args->route_flags & 2) != 0;

  if (op == 0) {
    fuse::Mxfp8GemmA2AParams p{};
    p.lhs = pointer<const fuse::Bf16>(t[0]);
    p.weight = weight;
    p.weight_workspace = workspace;
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
    return fuse::ulysses::QkvForward::launch(p, stream);
  }
  if (op == 1) {
    fuse::Mxfp8A2AGemmParams p{};
    for (int peer = 0; peer < world; ++peer) {
      p.peer_input[peer] = pointer<const fuse::Bf16>(args->peer_data[peer]);
    }
    p.weight = weight;
    p.weight_workspace = workspace;
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
    return fuse::ulysses::OprojForward::launch(p, stream);
  }
  if (op == 2) {
    fuse::Mxfp8QkvBackwardParams p{};
    auto& b = p.data;
    b.grad_q = pointer<const fuse::Bf16>(t[0]);
    b.grad_k = pointer<const fuse::Bf16>(t[1]);
    b.grad_v = pointer<const fuse::Bf16>(t[2]);
    for (int peer = 0; peer < world; ++peer) {
      b.peer_dqkv_staging[peer] = pointer<fuse::Bf16>(args->peer_data[peer]);
      b.peer_ready[peer] = pointer<uint32_t>(args->peer_ready[peer]);
      b.peer_done_epoch[peer] = pointer<uint32_t>(args->peer_done[peer]);
    }
    b.weight = weight;
    b.weight_workspace = workspace;
    b.grad_input = pointer<fuse::Bf16>(t[8]);
    b.local_tokens = m;
    b.hidden = h;
    b.q_heads = q;
    b.kv_heads = kv;
    b.head_dim = d;
    b.world_size = world;
    b.rank = rank;
    b.num_comm_ctas = mxfp8_test_comm_ctas;
    b.epoch = g[10];
    b.causal_load_balanced = g[9] != 0;
    b.alpha = args->alpha;
    p.weight.saved_input = pointer<const fuse::Bf16>(t[3]);
    p.weight.grad_weight = pointer<float>(t[9]);
    p.weight.alpha = args->alpha;
    p.weight.beta = args->beta;
    p.weight.gemm_policy = mxfp8_test_wgrad_policy;
    p.weight_mode = static_cast<fuse::WeightGradientMode>(g[11]);
    if (g[12]) {
      p.weight.dqkv_staging = pointer<const fuse::Bf16>(t[7]);
      p.weight.local_tokens = m;
      p.weight.hidden = h;
      p.weight.q_heads = q;
      p.weight.kv_heads = kv;
      p.weight.head_dim = d;
      return fuse::launch_qkv_backward_mxfp8_weight(p.weight, stream);
    }
    return fuse::launch_qkv_backward_mxfp8(p, stream);
  }
  if (op == 3) {
    fuse::Mxfp8OprojBackwardParams p{};
    auto& b = p.data;
    b.grad_output = pointer<const fuse::Bf16>(t[0]);
    b.weight = weight;
    b.weight_workspace = workspace;
    b.local_grad_attention = pointer<fuse::Bf16>(t[7]);
    for (int peer = 0; peer < world; ++peer) {
      b.peer_grad_attention[peer] = pointer<fuse::Bf16>(args->peer_data[peer]);
      b.peer_done_epoch[peer] = pointer<uint32_t>(args->peer_done[peer]);
    }
    b.ready = pointer<uint32_t>(t[10]);
    b.local_tokens = m;
    b.hidden = h;
    b.q_heads = q;
    b.head_dim = d;
    b.world_size = world;
    b.rank = rank;
    b.num_comm_ctas = mxfp8_test_comm_ctas;
    b.epoch = g[10];
    b.causal_load_balanced = g[9] != 0;
    b.alpha = args->alpha;
    p.weight.saved_attention = pointer<const fuse::Bf16>(t[3]);
    p.weight.grad_weight = pointer<float>(t[9]);
    p.weight.alpha = args->alpha;
    p.weight.beta = args->beta;
    p.weight.gemm_policy = mxfp8_test_wgrad_policy;
    p.weight_mode = static_cast<fuse::WeightGradientMode>(g[11]);
    if (g[12]) {
      p.weight.grad_output = b.grad_output;
      p.weight.local_tokens = m;
      p.weight.hidden = h;
      p.weight.q_heads = q;
      p.weight.head_dim = d;
      return fuse::launch_oproj_backward_mxfp8_weight(p.weight, stream);
    }
    return fuse::launch_oproj_backward_mxfp8(p, stream);
  }
  return cudaErrorInvalidValue;
}

#include "operator_profile.cuh"
#include "operator_reference.cuh"
