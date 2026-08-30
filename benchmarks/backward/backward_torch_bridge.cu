// SPDX-License-Identifier: BSD-3-Clause
// Benchmark-only C ABI used by backward_torch_autograd.py.  This file keeps
// PyTorch out of the production library: Torch owns the tensors and streams,
// while this adapter only translates raw addresses into the public fuse API.

#include "fuse/operators/oproj_backward.h"
#include "fuse/operators/qkv_backward.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace {

thread_local std::string g_last_error;

int finish(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) {
    g_last_error.clear();
    return 0;
  }
  g_last_error = std::string(operation) + ": " + cudaGetErrorString(status);
  return static_cast<int>(status);
}

template <class T>
T* pointer(uint64_t address) {
  return reinterpret_cast<T*>(static_cast<uintptr_t>(address));
}

cudaStream_t stream(uint64_t address) {
  return reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(address));
}

}  // namespace

extern "C" {

const char* fuse_backward_bridge_last_error() {
  return g_last_error.c_str();
}

int fuse_backward_bridge_enable_peer_access(int world_size) {
  for (int source = 0; source < world_size; ++source) {
    cudaError_t status = cudaSetDevice(source);
    if (status != cudaSuccess) {
      return finish(status, "cudaSetDevice");
    }
    for (int destination = 0; destination < world_size; ++destination) {
      if (source == destination) {
        continue;
      }
      int supported = 0;
      status = cudaDeviceCanAccessPeer(&supported, source, destination);
      if (status != cudaSuccess) {
        return finish(status, "cudaDeviceCanAccessPeer");
      }
      if (!supported) {
        g_last_error = "CUDA peer access is unavailable";
        return static_cast<int>(cudaErrorPeerAccessUnsupported);
      }
      status = cudaDeviceEnablePeerAccess(destination, 0);
      if (status == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
      } else if (status != cudaSuccess) {
        return finish(status, "cudaDeviceEnablePeerAccess");
      }
    }
  }
  g_last_error.clear();
  return 0;
}

int64_t fuse_backward_bridge_qkv_ready_elements(
    int local_tokens,
    int hidden,
    int batch,
    int q_heads,
    int kv_heads,
    int head_dim,
    int world_size,
    int num_comm_ctas,
    int causal_load_balanced) {
  fuse::QkvBackwardDataParams params{};
  params.local_tokens = local_tokens;
  params.hidden = hidden;
  params.batch = batch;
  params.q_heads = q_heads;
  params.kv_heads = kv_heads;
  params.head_dim = head_dim;
  params.world_size = world_size;
  params.num_comm_ctas = num_comm_ctas;
  params.causal_load_balanced = causal_load_balanced != 0;
  return fuse::qkv_backward_ready_elements(params);
}

int64_t fuse_backward_bridge_oproj_ready_elements(
    int local_tokens,
    int hidden,
    int batch,
    int q_heads,
    int head_dim,
    int world_size,
    int num_comm_ctas,
    int causal_load_balanced) {
  fuse::OprojBackwardDataParams params{};
  params.local_tokens = local_tokens;
  params.hidden = hidden;
  params.batch = batch;
  params.q_heads = q_heads;
  params.head_dim = head_dim;
  params.world_size = world_size;
  params.num_comm_ctas = num_comm_ctas;
  params.causal_load_balanced = causal_load_balanced != 0;
  return fuse::oproj_backward_ready_elements(params);
}

int fuse_backward_bridge_qkv_data(
    int device,
    uint64_t grad_q,
    uint64_t grad_k,
    uint64_t grad_v,
    const uint64_t* peer_staging,
    const uint64_t* peer_ready,
    const uint64_t* peer_done,
    uint64_t weight,
    uint64_t grad_input,
    int local_tokens,
    int hidden,
    int batch,
    int q_heads,
    int kv_heads,
    int head_dim,
    int world_size,
    int rank,
    int num_comm_ctas,
    uint32_t epoch,
    int causal_load_balanced,
    uint64_t stream_address) {
  cudaError_t status = cudaSetDevice(device);
  if (status != cudaSuccess) {
    return finish(status, "cudaSetDevice");
  }
  fuse::QkvBackwardDataParams params{};
  params.grad_q = pointer<const fuse::Bf16>(grad_q);
  params.grad_k = pointer<const fuse::Bf16>(grad_k);
  params.grad_v = pointer<const fuse::Bf16>(grad_v);
  for (int peer = 0; peer < world_size; ++peer) {
    params.peer_dqkv_staging[peer] = pointer<fuse::Bf16>(peer_staging[peer]);
    params.peer_ready[peer] = pointer<uint32_t>(peer_ready[peer]);
    params.peer_done_epoch[peer] = pointer<uint32_t>(peer_done[peer]);
  }
  params.weight = pointer<const fuse::Bf16>(weight);
  params.grad_input = pointer<fuse::Bf16>(grad_input);
  params.local_tokens = local_tokens;
  params.hidden = hidden;
  params.batch = batch;
  params.q_heads = q_heads;
  params.kv_heads = kv_heads;
  params.head_dim = head_dim;
  params.world_size = world_size;
  params.rank = rank;
  params.num_comm_ctas = num_comm_ctas;
  params.epoch = epoch;
  params.causal_load_balanced = causal_load_balanced != 0;
  return finish(
      fuse::launch_qkv_backward_data(params, stream(stream_address)),
      "launch_qkv_backward_data");
}

int fuse_backward_bridge_qkv_weight(
    int device,
    uint64_t dqkv_staging,
    uint64_t saved_input,
    uint64_t grad_weight,
    int local_tokens,
    int hidden,
    int q_heads,
    int kv_heads,
    int head_dim,
    float beta,
    uint64_t stream_address) {
  cudaError_t status = cudaSetDevice(device);
  if (status != cudaSuccess) {
    return finish(status, "cudaSetDevice");
  }
  fuse::QkvBackwardWeightParams params{};
  params.dqkv_staging = pointer<const fuse::Bf16>(dqkv_staging);
  params.saved_input = pointer<const fuse::Bf16>(saved_input);
  params.grad_weight = pointer<fuse::Bf16>(grad_weight);
  params.local_tokens = local_tokens;
  params.hidden = hidden;
  params.q_heads = q_heads;
  params.kv_heads = kv_heads;
  params.head_dim = head_dim;
  params.beta = beta;
  return finish(
      fuse::launch_qkv_backward_weight(params, stream(stream_address)),
      "launch_qkv_backward_weight");
}

int fuse_backward_bridge_oproj_data(
    int device,
    uint64_t grad_output,
    uint64_t weight,
    uint64_t local_grad_attention,
    const uint64_t* peer_grad_attention,
    const uint64_t* peer_done,
    uint64_t ready,
    int local_tokens,
    int hidden,
    int batch,
    int q_heads,
    int head_dim,
    int world_size,
    int rank,
    int num_comm_ctas,
    uint32_t epoch,
    int causal_load_balanced,
    uint64_t stream_address) {
  cudaError_t status = cudaSetDevice(device);
  if (status != cudaSuccess) {
    return finish(status, "cudaSetDevice");
  }
  fuse::OprojBackwardDataParams params{};
  params.grad_output = pointer<const fuse::Bf16>(grad_output);
  params.weight = pointer<const fuse::Bf16>(weight);
  params.local_grad_attention = pointer<fuse::Bf16>(local_grad_attention);
  for (int peer = 0; peer < world_size; ++peer) {
    params.peer_grad_attention[peer] =
        pointer<fuse::Bf16>(peer_grad_attention[peer]);
    params.peer_done_epoch[peer] = pointer<uint32_t>(peer_done[peer]);
  }
  params.ready = pointer<uint32_t>(ready);
  params.local_tokens = local_tokens;
  params.hidden = hidden;
  params.batch = batch;
  params.q_heads = q_heads;
  params.head_dim = head_dim;
  params.world_size = world_size;
  params.rank = rank;
  params.num_comm_ctas = num_comm_ctas;
  params.epoch = epoch;
  params.causal_load_balanced = causal_load_balanced != 0;
  return finish(
      fuse::launch_oproj_backward_data(params, stream(stream_address)),
      "launch_oproj_backward_data");
}

int fuse_backward_bridge_oproj_weight(
    int device,
    uint64_t grad_output,
    uint64_t saved_attention,
    uint64_t grad_weight,
    int local_tokens,
    int hidden,
    int q_heads,
    int head_dim,
    float beta,
    uint64_t stream_address) {
  cudaError_t status = cudaSetDevice(device);
  if (status != cudaSuccess) {
    return finish(status, "cudaSetDevice");
  }
  fuse::OprojBackwardWeightParams params{};
  params.grad_output = pointer<const fuse::Bf16>(grad_output);
  params.saved_attention = pointer<const fuse::Bf16>(saved_attention);
  params.grad_weight = pointer<fuse::Bf16>(grad_weight);
  params.local_tokens = local_tokens;
  params.hidden = hidden;
  params.q_heads = q_heads;
  params.head_dim = head_dim;
  params.beta = beta;
  return finish(
      fuse::launch_oproj_backward_weight(params, stream(stream_address)),
      "launch_oproj_backward_weight");
}

}  // extern "C"
