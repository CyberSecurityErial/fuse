#include "fuse/operators/oproj_backward.h"
#include "fuse/operators/qkv_backward.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using fuse::Bf16;

#define CUDA_CHECK(expr) check_cuda((expr), #expr, __FILE__, __LINE__)

void check_cuda(cudaError_t status, const char* expr, const char* file, int line) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(file) + ":" + std::to_string(line) + " " + expr +
        ": " + cudaGetErrorString(status));
  }
}

template <class T>
T* allocate(int device, size_t elements) {
  CUDA_CHECK(cudaSetDevice(device));
  T* pointer = nullptr;
  CUDA_CHECK(cudaMalloc(&pointer, elements * sizeof(T)));
  return pointer;
}

template <class T>
void upload(int device, T* destination, const std::vector<T>& source) {
  CUDA_CHECK(cudaSetDevice(device));
  CUDA_CHECK(cudaMemcpy(
      destination,
      source.data(),
      source.size() * sizeof(T),
      cudaMemcpyHostToDevice));
}

template <class T>
std::vector<T> download(int device, const T* source, size_t elements) {
  CUDA_CHECK(cudaSetDevice(device));
  std::vector<T> result(elements);
  CUDA_CHECK(cudaMemcpy(
      result.data(),
      source,
      elements * sizeof(T),
      cudaMemcpyDeviceToHost));
  return result;
}

std::vector<Bf16> values(size_t elements, int seed) {
  std::vector<Bf16> result(elements);
  for (size_t index = 0; index < elements; ++index) {
    const int value = static_cast<int>((index * 17 + seed * 23) % 13) - 6;
    result[index] = Bf16(static_cast<float>(value) / 64.0f);
  }
  return result;
}

void expect_close(
    const std::vector<Bf16>& actual,
    const std::vector<Bf16>& expected,
    float tolerance,
    const std::string& label) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error(label + ": size mismatch");
  }
  float maximum = 0.0f;
  size_t worst = 0;
  for (size_t index = 0; index < actual.size(); ++index) {
    const float error = std::abs(
        static_cast<float>(actual[index]) -
        static_cast<float>(expected[index]));
    if (error > maximum) {
      maximum = error;
      worst = index;
    }
  }
  if (maximum > tolerance) {
    throw std::runtime_error(
        label + ": max_abs=" + std::to_string(maximum) +
        " index=" + std::to_string(worst) +
        " actual=" + std::to_string(static_cast<float>(actual[worst])) +
        " expected=" + std::to_string(static_cast<float>(expected[worst])));
  }
}

std::vector<Bf16> scaled(
    const std::vector<Bf16>& values,
    float scale) {
  std::vector<Bf16> result(values.size());
  for (size_t index = 0; index < values.size(); ++index) {
    result[index] = Bf16(static_cast<float>(values[index]) * scale);
  }
  return result;
}

std::vector<Bf16> matmul(
    const std::vector<Bf16>& lhs,
    const std::vector<Bf16>& rhs,
    int m,
    int n,
    int k) {
  std::vector<Bf16> result(static_cast<size_t>(m) * n);
  for (int row = 0; row < m; ++row) {
    for (int column = 0; column < n; ++column) {
      float accumulator = 0.0f;
      for (int inner = 0; inner < k; ++inner) {
        accumulator += static_cast<float>(
            lhs[static_cast<size_t>(row) * k + inner]) *
            static_cast<float>(rhs[static_cast<size_t>(inner) * n + column]);
      }
      result[static_cast<size_t>(row) * n + column] = Bf16(accumulator);
    }
  }
  return result;
}

std::vector<Bf16> transpose_left_matmul(
    const std::vector<Bf16>& lhs,
    const std::vector<Bf16>& rhs,
    int tokens,
    int rows,
    int columns) {
  std::vector<Bf16> result(static_cast<size_t>(rows) * columns);
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < columns; ++column) {
      float accumulator = 0.0f;
      for (int token = 0; token < tokens; ++token) {
        accumulator += static_cast<float>(
            lhs[static_cast<size_t>(token) * rows + row]) *
            static_cast<float>(rhs[static_cast<size_t>(token) * columns + column]);
      }
      result[static_cast<size_t>(row) * columns + column] = Bf16(accumulator);
    }
  }
  return result;
}

struct Runtime {
  cudaStream_t stream = nullptr;
};

std::vector<Runtime> make_runtimes(int world) {
  std::vector<Runtime> result(world);
  for (int rank = 0; rank < world; ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaStreamCreateWithFlags(
        &result[rank].stream, cudaStreamNonBlocking));
  }
  return result;
}

void enable_peer_access(int world) {
  for (int source = 0; source < world; ++source) {
    CUDA_CHECK(cudaSetDevice(source));
    for (int peer = 0; peer < world; ++peer) {
      if (source == peer) {
        continue;
      }
      int supported = 0;
      CUDA_CHECK(cudaDeviceCanAccessPeer(&supported, source, peer));
      if (!supported) {
        throw std::runtime_error("peer access is unavailable");
      }
      const cudaError_t status = cudaDeviceEnablePeerAccess(peer, 0);
      if (status == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
      } else {
        CUDA_CHECK(status);
      }
    }
  }
}

void synchronize(const std::vector<Runtime>& runtimes) {
  for (int rank = 0; rank < static_cast<int>(runtimes.size()); ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaStreamSynchronize(runtimes[rank].stream));
  }
}

int global_sequence_row(
    int owner,
    int local_row,
    int local_tokens,
    int batch,
    int world,
    bool causal_load_balanced) {
  const int sequence_local = local_tokens / batch;
  const int batch_index = local_row / sequence_local;
  const int row_in_batch = local_row - batch_index * sequence_local;
  if (!causal_load_balanced) {
    return batch_index * sequence_local * world +
        owner * sequence_local + row_in_batch;
  }
  const int chunk_rows = sequence_local / 2;
  const int chunk = row_in_batch < chunk_rows
      ? owner
      : 2 * world - owner - 1;
  return batch_index * sequence_local * world + chunk * chunk_rows +
      (row_in_batch < chunk_rows
           ? row_in_batch
           : row_in_batch - chunk_rows);
}

void smoke_qkv_backward(
    int world,
    const std::vector<Runtime>& runtimes,
    bool causal_load_balanced,
    int batch,
    int local_tokens = 16,
    int hidden = 256) {
  constexpr int q_heads = 16;
  constexpr int kv_heads = 8;
  constexpr int head_dim = 128;
  constexpr int comm_ctas = 8;
  const int global_tokens = local_tokens * world;
  const int q_local_heads = q_heads / world;
  const int kv_local_heads = kv_heads / world;
  const int q_local_width = q_local_heads * head_dim;
  const int kv_local_width = kv_local_heads * head_dim;
  const int packed_width = (q_heads + 2 * kv_heads) * head_dim;
  const size_t staging_elements =
      static_cast<size_t>(local_tokens) * packed_width;
  const size_t input_elements = static_cast<size_t>(local_tokens) * hidden;
  const size_t weight_elements = static_cast<size_t>(packed_width) * hidden;

  std::vector<std::vector<Bf16>> host_q(world);
  std::vector<std::vector<Bf16>> host_k(world);
  std::vector<std::vector<Bf16>> host_v(world);
  std::vector<std::vector<Bf16>> host_x(world);
  std::vector<std::vector<Bf16>> host_weight(world);
  std::vector<Bf16*> grad_q(world), grad_k(world), grad_v(world);
  std::vector<Bf16*> staging(world), weight(world), x(world), dx(world), dw(world);
  std::vector<uint32_t*> ready(world), done(world);

  fuse::QkvBackwardDataParams shape{};
  shape.local_tokens = local_tokens;
  shape.hidden = hidden;
  shape.batch = batch;
  shape.q_heads = q_heads;
  shape.kv_heads = kv_heads;
  shape.head_dim = head_dim;
  shape.world_size = world;
  const int64_t ready_elements = fuse::qkv_backward_ready_elements(shape);
  if (ready_elements <= 0) {
    throw std::runtime_error("QKV backward ready shape failed");
  }

  for (int rank = 0; rank < world; ++rank) {
    host_q[rank] = values(
        static_cast<size_t>(global_tokens) * q_local_width, 100 + rank);
    host_k[rank] = values(
        static_cast<size_t>(global_tokens) * kv_local_width, 200 + rank);
    host_v[rank] = values(
        static_cast<size_t>(global_tokens) * kv_local_width, 300 + rank);
    host_x[rank] = values(input_elements, 400 + rank);
    host_weight[rank] = values(weight_elements, 500);
    grad_q[rank] = allocate<Bf16>(rank, host_q[rank].size());
    grad_k[rank] = allocate<Bf16>(rank, host_k[rank].size());
    grad_v[rank] = allocate<Bf16>(rank, host_v[rank].size());
    staging[rank] = allocate<Bf16>(rank, staging_elements);
    weight[rank] = allocate<Bf16>(rank, weight_elements);
    x[rank] = allocate<Bf16>(rank, input_elements);
    dx[rank] = allocate<Bf16>(rank, input_elements);
    dw[rank] = allocate<Bf16>(rank, weight_elements);
    ready[rank] = allocate<uint32_t>(rank, ready_elements);
    done[rank] = allocate<uint32_t>(
        rank, world * fuse::kReadyFlagStride);
    upload(rank, grad_q[rank], host_q[rank]);
    upload(rank, grad_k[rank], host_k[rank]);
    upload(rank, grad_v[rank], host_v[rank]);
    upload(rank, weight[rank], host_weight[rank]);
    upload(rank, x[rank], host_x[rank]);
    CUDA_CHECK(cudaMemset(staging[rank], 0, staging_elements * sizeof(Bf16)));
    CUDA_CHECK(cudaMemset(dx[rank], 0, input_elements * sizeof(Bf16)));
    CUDA_CHECK(cudaMemset(dw[rank], 0, weight_elements * sizeof(Bf16)));
    CUDA_CHECK(cudaMemset(ready[rank], 0, ready_elements * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(
        done[rank], 0, world * fuse::kReadyFlagStride * sizeof(uint32_t)));
  }

  std::vector<fuse::QkvBackwardParams> params(world);
  for (int rank = 0; rank < world; ++rank) {
    auto& data = params[rank].data;
    data.grad_q = grad_q[rank];
    data.grad_k = grad_k[rank];
    data.grad_v = grad_v[rank];
    for (int peer = 0; peer < world; ++peer) {
      data.peer_dqkv_staging[peer] = staging[peer];
      data.peer_ready[peer] = ready[peer];
      data.peer_done_epoch[peer] = done[peer];
    }
    data.weight = weight[rank];
    data.grad_input = dx[rank];
    data.local_tokens = local_tokens;
    data.hidden = hidden;
    data.batch = batch;
    data.q_heads = q_heads;
    data.kv_heads = kv_heads;
    data.head_dim = head_dim;
    data.world_size = world;
    data.rank = rank;
    data.num_comm_ctas = comm_ctas;
    data.epoch = 1;
    data.causal_load_balanced = causal_load_balanced;
    params[rank].weight.dqkv_staging = staging[rank];
    params[rank].weight.saved_input = x[rank];
    params[rank].weight.grad_weight = dw[rank];
    params[rank].weight.local_tokens = local_tokens;
    params[rank].weight.hidden = hidden;
    params[rank].weight.q_heads = q_heads;
    params[rank].weight.kv_heads = kv_heads;
    params[rank].weight.head_dim = head_dim;
    params[rank].weight_mode = fuse::WeightGradientMode::kDeferred;
  }

  for (int rank = 0; rank < world; ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(fuse::launch_qkv_backward(
        params[rank], runtimes[rank].stream));
  }
  synchronize(runtimes);

  for (int destination = 0; destination < world; ++destination) {
    std::vector<Bf16> expected(staging_elements);
    for (int source = 0; source < world; ++source) {
      for (int row = 0; row < local_tokens; ++row) {
        const int global_row = global_sequence_row(
            destination,
            row,
            local_tokens,
            batch,
            world,
            causal_load_balanced);
        for (int local_head = 0; local_head < q_local_heads; ++local_head) {
          const int global_head = source * q_local_heads + local_head;
          std::copy_n(
              host_q[source].data() +
                  static_cast<size_t>(global_row) * q_local_width +
                  local_head * head_dim,
              head_dim,
              expected.data() + static_cast<size_t>(row) * packed_width +
                  global_head * head_dim);
        }
        for (int local_head = 0; local_head < kv_local_heads; ++local_head) {
          const int global_head = source * kv_local_heads + local_head;
          std::copy_n(
              host_k[source].data() +
                  static_cast<size_t>(global_row) * kv_local_width +
                  local_head * head_dim,
              head_dim,
              expected.data() + static_cast<size_t>(row) * packed_width +
                  (q_heads + global_head) * head_dim);
          std::copy_n(
              host_v[source].data() +
                  static_cast<size_t>(global_row) * kv_local_width +
                  local_head * head_dim,
              head_dim,
              expected.data() + static_cast<size_t>(row) * packed_width +
                  (q_heads + kv_heads + global_head) * head_dim);
        }
      }
    }
    expect_close(
        download(destination, staging[destination], staging_elements),
        expected,
        0.0f,
        "QKV backward route rank " + std::to_string(destination));
    expect_close(
        download(destination, dx[destination], input_elements),
        matmul(
            expected,
            host_weight[destination],
            local_tokens,
            hidden,
            packed_width),
        0.125f,
        "QKV backward dgrad rank " + std::to_string(destination));

    const auto expected_dw = transpose_left_matmul(
        expected,
        host_x[destination],
        local_tokens,
        packed_width,
        hidden);
    CUDA_CHECK(cudaSetDevice(destination));
    CUDA_CHECK(fuse::launch_qkv_backward_weight(
        params[destination].weight, runtimes[destination].stream));
    CUDA_CHECK(cudaStreamSynchronize(runtimes[destination].stream));
    expect_close(
        download(destination, dw[destination], weight_elements),
        expected_dw,
        0.125f,
        "QKV backward delayed wgrad rank " + std::to_string(destination));

    params[destination].weight.beta = 1.0f;
    CUDA_CHECK(fuse::launch_qkv_backward_weight(
        params[destination].weight, runtimes[destination].stream));
    CUDA_CHECK(cudaStreamSynchronize(runtimes[destination].stream));
    expect_close(
        download(destination, dw[destination], weight_elements),
        scaled(expected_dw, 2.0f),
        0.25f,
        "QKV backward beta accumulation rank " +
            std::to_string(destination));
    params[destination].weight.beta = 0.0f;
  }


  for (int rank = 0; rank < world; ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaMemsetAsync(
        dw[rank], 0, weight_elements * sizeof(Bf16), runtimes[rank].stream));
    params[rank].data.epoch = 2;
    params[rank].weight_mode = fuse::WeightGradientMode::kImmediate;
    CUDA_CHECK(fuse::launch_qkv_backward(
        params[rank], runtimes[rank].stream));
  }
  synchronize(runtimes);
  for (int rank = 0; rank < world; ++rank) {
    std::vector<Bf16> expected(staging_elements);
    for (int source = 0; source < world; ++source) {
      for (int row = 0; row < local_tokens; ++row) {
        const int global_row = global_sequence_row(
            rank,
            row,
            local_tokens,
            batch,
            world,
            causal_load_balanced);
        for (int local_head = 0; local_head < q_local_heads; ++local_head) {
          const int global_head = source * q_local_heads + local_head;
          std::copy_n(
              host_q[source].data() +
                  static_cast<size_t>(global_row) * q_local_width +
                  local_head * head_dim,
              head_dim,
              expected.data() + static_cast<size_t>(row) * packed_width +
                  global_head * head_dim);
        }
        for (int local_head = 0; local_head < kv_local_heads; ++local_head) {
          const int global_head = source * kv_local_heads + local_head;
          std::copy_n(
              host_k[source].data() +
                  static_cast<size_t>(global_row) * kv_local_width +
                  local_head * head_dim,
              head_dim,
              expected.data() + static_cast<size_t>(row) * packed_width +
                  (q_heads + global_head) * head_dim);
          std::copy_n(
              host_v[source].data() +
                  static_cast<size_t>(global_row) * kv_local_width +
                  local_head * head_dim,
              head_dim,
              expected.data() + static_cast<size_t>(row) * packed_width +
                  (q_heads + kv_heads + global_head) * head_dim);
        }
      }
    }
    expect_close(
        download(rank, dw[rank], weight_elements),
        transpose_left_matmul(
            expected,
            host_x[rank],
            local_tokens,
            packed_width,
            hidden),
        0.125f,
        "QKV backward immediate wgrad rank " + std::to_string(rank));
  }
  std::cout << "QKV backward CP" << world
            << " batch=" << batch
            << " M=" << local_tokens
            << (causal_load_balanced ? " causal" : " rank-major")
            << " B/W split: PASS\n";
}

void smoke_oproj_backward(
    int world,
    const std::vector<Runtime>& runtimes,
    bool causal_load_balanced,
    int batch,
    int local_tokens = 16,
    int hidden = 256) {
  constexpr int q_heads = 16;
  constexpr int head_dim = 128;
  constexpr int comm_ctas = 8;
  const int attention_width = q_heads * head_dim;
  const int local_width = attention_width / world;
  const int global_tokens = local_tokens * world;
  const size_t dy_elements = static_cast<size_t>(local_tokens) * hidden;
  const size_t attention_elements =
      static_cast<size_t>(local_tokens) * attention_width;
  const size_t peer_elements =
      static_cast<size_t>(global_tokens) * local_width;
  const size_t weight_elements =
      static_cast<size_t>(hidden) * attention_width;

  std::vector<std::vector<Bf16>> host_dy(world), host_attention(world), host_weight(world);
  std::vector<Bf16*> dy(world), attention(world), local_da(world), peer_da(world);
  std::vector<Bf16*> weight(world), dw(world);
  std::vector<uint32_t*> ready(world), done(world);
  fuse::OprojBackwardDataParams shape{};
  shape.local_tokens = local_tokens;
  shape.hidden = hidden;
  shape.batch = batch;
  shape.q_heads = q_heads;
  shape.head_dim = head_dim;
  shape.world_size = world;
  const int64_t ready_elements = fuse::oproj_backward_ready_elements(shape);

  for (int rank = 0; rank < world; ++rank) {
    host_dy[rank] = values(dy_elements, 600 + rank);
    host_attention[rank] = values(attention_elements, 700 + rank);
    host_weight[rank] = values(weight_elements, 800);
    dy[rank] = allocate<Bf16>(rank, dy_elements);
    attention[rank] = allocate<Bf16>(rank, attention_elements);
    local_da[rank] = allocate<Bf16>(rank, attention_elements);
    peer_da[rank] = allocate<Bf16>(rank, peer_elements);
    weight[rank] = allocate<Bf16>(rank, weight_elements);
    dw[rank] = allocate<Bf16>(rank, weight_elements);
    ready[rank] = allocate<uint32_t>(rank, ready_elements);
    done[rank] = allocate<uint32_t>(
        rank, world * fuse::kReadyFlagStride);
    upload(rank, dy[rank], host_dy[rank]);
    upload(rank, attention[rank], host_attention[rank]);
    upload(rank, weight[rank], host_weight[rank]);
    CUDA_CHECK(cudaMemset(local_da[rank], 0, attention_elements * sizeof(Bf16)));
    CUDA_CHECK(cudaMemset(peer_da[rank], 0, peer_elements * sizeof(Bf16)));
    CUDA_CHECK(cudaMemset(dw[rank], 0, weight_elements * sizeof(Bf16)));
    CUDA_CHECK(cudaMemset(ready[rank], 0, ready_elements * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(
        done[rank], 0, world * fuse::kReadyFlagStride * sizeof(uint32_t)));
  }

  std::vector<fuse::OprojBackwardParams> params(world);
  for (int rank = 0; rank < world; ++rank) {
    auto& data = params[rank].data;
    data.grad_output = dy[rank];
    data.weight = weight[rank];
    data.local_grad_attention = local_da[rank];
    for (int peer = 0; peer < world; ++peer) {
      data.peer_grad_attention[peer] = peer_da[peer];
      data.peer_done_epoch[peer] = done[peer];
    }
    data.ready = ready[rank];
    data.local_tokens = local_tokens;
    data.hidden = hidden;
    data.batch = batch;
    data.q_heads = q_heads;
    data.head_dim = head_dim;
    data.world_size = world;
    data.rank = rank;
    data.num_comm_ctas = comm_ctas;
    data.epoch = 1;
    data.causal_load_balanced = causal_load_balanced;
    params[rank].weight.grad_output = dy[rank];
    params[rank].weight.saved_attention = attention[rank];
    params[rank].weight.grad_weight = dw[rank];
    params[rank].weight.local_tokens = local_tokens;
    params[rank].weight.hidden = hidden;
    params[rank].weight.q_heads = q_heads;
    params[rank].weight.head_dim = head_dim;
    params[rank].weight_mode = fuse::WeightGradientMode::kDeferred;
  }

  for (int rank = 0; rank < world; ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(fuse::launch_oproj_backward(
        params[rank], runtimes[rank].stream));
  }
  synchronize(runtimes);

  std::vector<std::vector<Bf16>> expected_da(world);
  for (int source = 0; source < world; ++source) {
    expected_da[source] = matmul(
        host_dy[source],
        host_weight[source],
        local_tokens,
        attention_width,
        hidden);
    expect_close(
        download(source, local_da[source], attention_elements),
        expected_da[source],
        0.125f,
        "OProj backward dgrad rank " + std::to_string(source));
  }
  for (int destination = 0; destination < world; ++destination) {
    std::vector<Bf16> expected(peer_elements);
    for (int source = 0; source < world; ++source) {
      for (int row = 0; row < local_tokens; ++row) {
        std::copy_n(
            expected_da[source].data() +
                static_cast<size_t>(row) * attention_width +
                destination * local_width,
            local_width,
            expected.data() +
                static_cast<size_t>(global_sequence_row(
                    source,
                    row,
                    local_tokens,
                    batch,
                    world,
                    causal_load_balanced)) * local_width);
      }
    }
    expect_close(
        download(destination, peer_da[destination], peer_elements),
        expected,
        0.125f,
        "OProj backward route rank " + std::to_string(destination));
    const auto expected_dw = transpose_left_matmul(
        host_dy[destination],
        host_attention[destination],
        local_tokens,
        hidden,
        attention_width);
    CUDA_CHECK(cudaSetDevice(destination));
    CUDA_CHECK(fuse::launch_oproj_backward_weight(
        params[destination].weight, runtimes[destination].stream));
    CUDA_CHECK(cudaStreamSynchronize(runtimes[destination].stream));
    expect_close(
        download(destination, dw[destination], weight_elements),
        expected_dw,
        0.125f,
        "OProj backward delayed wgrad rank " + std::to_string(destination));

    params[destination].weight.beta = 1.0f;
    CUDA_CHECK(fuse::launch_oproj_backward_weight(
        params[destination].weight, runtimes[destination].stream));
    CUDA_CHECK(cudaStreamSynchronize(runtimes[destination].stream));
    expect_close(
        download(destination, dw[destination], weight_elements),
        scaled(expected_dw, 2.0f),
        0.25f,
        "OProj backward beta accumulation rank " +
            std::to_string(destination));
    params[destination].weight.beta = 0.0f;
  }


  for (int rank = 0; rank < world; ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaMemsetAsync(
        dw[rank], 0, weight_elements * sizeof(Bf16), runtimes[rank].stream));
    params[rank].data.epoch = 2;
    params[rank].weight_mode = fuse::WeightGradientMode::kImmediate;
    CUDA_CHECK(fuse::launch_oproj_backward(
        params[rank], runtimes[rank].stream));
  }
  synchronize(runtimes);
  for (int rank = 0; rank < world; ++rank) {
    expect_close(
        download(rank, dw[rank], weight_elements),
        transpose_left_matmul(
            host_dy[rank],
            host_attention[rank],
            local_tokens,
            hidden,
            attention_width),
        0.125f,
        "OProj backward immediate wgrad rank " + std::to_string(rank));
  }
  std::cout << "OProj backward CP" << world
            << " batch=" << batch
            << " M=" << local_tokens
            << (causal_load_balanced ? " causal" : " rank-major")
            << " B/W split: PASS\n";
}

}  // namespace

int main() {
  try {
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices < 8) {
      throw std::runtime_error("backward_smoke requires eight GPUs");
    }
    constexpr int maximum_world = 8;
    enable_peer_access(maximum_world);
    const auto runtimes = make_runtimes(maximum_world);
    smoke_qkv_backward(4, runtimes, false, 1);
    smoke_oproj_backward(4, runtimes, false, 1);
    smoke_qkv_backward(8, runtimes, true, 1);
    smoke_oproj_backward(8, runtimes, true, 1);
    // Batch-two with 128 rows per local sequence exercises the TMA route,
    // including the batch offset in the causal dual-chunk mapping.
    smoke_qkv_backward(4, runtimes, true, 2, 256, 16);
    smoke_oproj_backward(4, runtimes, true, 2, 256, 16);
    std::cout << "backward_smoke: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "backward_smoke: FAIL: " << error.what() << "\n";
    return 1;
  }
}
