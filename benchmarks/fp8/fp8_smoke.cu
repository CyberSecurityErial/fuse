#include "fuse/operators/a2a_gemm.h"
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

using fuse::Fp8E4m3;

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

std::vector<Fp8E4m3> fp8_values(size_t elements, int seed) {
  std::vector<Fp8E4m3> result(elements);
  for (size_t index = 0; index < elements; ++index) {
    const int value = static_cast<int>((index * 17 + seed * 23) % 13) - 6;
    result[index] = Fp8E4m3(static_cast<float>(value) / 32.0f);
  }
  return result;
}

void expect_fp8_equal(
    const std::vector<Fp8E4m3>& actual,
    const std::vector<Fp8E4m3>& expected,
    const std::string& label) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error(label + ": size mismatch");
  }
  for (size_t index = 0; index < actual.size(); ++index) {
    if (static_cast<float>(actual[index]) !=
        static_cast<float>(expected[index])) {
      throw std::runtime_error(
          label + ": mismatch at " + std::to_string(index));
    }
  }
}

std::vector<Fp8E4m3> matmul_rhs_nt(
    const std::vector<Fp8E4m3>& lhs,
    const std::vector<Fp8E4m3>& rhs_nt,
    int m,
    int n,
    int k) {
  std::vector<Fp8E4m3> result(static_cast<size_t>(m) * n);
  for (int row = 0; row < m; ++row) {
    for (int column = 0; column < n; ++column) {
      float accumulator = 0.0f;
      for (int inner = 0; inner < k; ++inner) {
        accumulator += static_cast<float>(
            lhs[static_cast<size_t>(row) * k + inner]) *
            static_cast<float>(
                rhs_nt[static_cast<size_t>(column) * k + inner]);
      }
      result[static_cast<size_t>(row) * n + column] = Fp8E4m3(accumulator);
    }
  }
  return result;
}

std::vector<Fp8E4m3> wgrad_from_transposes(
    const std::vector<Fp8E4m3>& gradient_t,
    const std::vector<Fp8E4m3>& input_t,
    int rows,
    int columns,
    int tokens) {
  std::vector<Fp8E4m3> result(static_cast<size_t>(rows) * columns);
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < columns; ++column) {
      float accumulator = 0.0f;
      for (int token = 0; token < tokens; ++token) {
        accumulator += static_cast<float>(
            gradient_t[static_cast<size_t>(row) * tokens + token]) *
            static_cast<float>(
                input_t[static_cast<size_t>(column) * tokens + token]);
      }
      result[static_cast<size_t>(row) * columns + column] =
          Fp8E4m3(accumulator);
    }
  }
  return result;
}

std::vector<Fp8E4m3> scaled(
    const std::vector<Fp8E4m3>& input,
    float scale) {
  std::vector<Fp8E4m3> result(input.size());
  for (size_t index = 0; index < input.size(); ++index) {
    result[index] = Fp8E4m3(static_cast<float>(input[index]) * scale);
  }
  return result;
}

std::vector<Fp8E4m3> transpose_fp8(
    const std::vector<Fp8E4m3>& input,
    int rows,
    int columns) {
  std::vector<Fp8E4m3> result(static_cast<size_t>(rows) * columns);
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < columns; ++column) {
      result[static_cast<size_t>(column) * rows + row] =
          input[static_cast<size_t>(row) * columns + column];
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
      if (source == peer) continue;
      int supported = 0;
      CUDA_CHECK(cudaDeviceCanAccessPeer(&supported, source, peer));
      if (!supported) throw std::runtime_error("peer access unavailable");
      const cudaError_t status = cudaDeviceEnablePeerAccess(peer, 0);
      if (status == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
      } else {
        CUDA_CHECK(status);
      }
    }
  }
}

void synchronize(const std::vector<Runtime>& runtimes, int world) {
  for (int rank = 0; rank < world; ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaStreamSynchronize(runtimes[rank].stream));
  }
}

int global_sequence_row(int owner, int local_row, int local_tokens) {
  return owner * local_tokens + local_row;
}

void smoke_fp8_oproj_forward(
    int world,
    const std::vector<Runtime>& runtimes,
    int local_tokens) {
  constexpr int q_heads = 4;
  constexpr int head_dim = 128;
  constexpr int hidden = 128;
  constexpr int comm_ctas = 8;
  const int attention_width = q_heads * head_dim;
  const int local_width = attention_width / world;
  const int global_tokens = local_tokens * world;
  const size_t peer_elements =
      static_cast<size_t>(global_tokens) * local_width;
  const size_t staging_elements =
      static_cast<size_t>(local_tokens) * attention_width;
  const size_t output_elements =
      static_cast<size_t>(local_tokens) * hidden;
  const size_t weight_elements =
      static_cast<size_t>(hidden) * attention_width;

  std::vector<std::vector<Fp8E4m3>> host_peer(world), host_weight(world);
  std::vector<Fp8E4m3*> peer(world), staging(world), weight_nt(world);
  std::vector<Fp8E4m3*> output(world);
  std::vector<uint32_t*> ready(world);
  fuse::GemmProblem problem{local_tokens, hidden, attention_width, 1};
  problem.input_dtype = fuse::DType::kFloat8E4M3;
  problem.weight_dtype = fuse::DType::kFloat8E4M3;
  problem.output_dtype = fuse::DType::kFloat8E4M3;
  fuse::UlyssesRoute route{};
  route.world_size = world;
  route.batch = 1;
  route.seq_local = local_tokens;
  route.global_seq = global_tokens;
  route.q_heads = q_heads;
  route.local_heads = q_heads / world;
  route.head_dim = head_dim;
  route.kind = fuse::RouteKind::kHeadToSequence;
  route.direction = fuse::RouteDirection::kInverse;
  const int64_t ready_elements =
      fuse::a2a_lhs_gemm_ready_elements(problem, route);

  for (int rank = 0; rank < world; ++rank) {
    host_peer[rank] = fp8_values(peer_elements, 10 + rank);
    host_weight[rank] = fp8_values(weight_elements, 20);
    peer[rank] = allocate<Fp8E4m3>(rank, peer_elements);
    staging[rank] = allocate<Fp8E4m3>(rank, staging_elements);
    weight_nt[rank] = allocate<Fp8E4m3>(rank, weight_elements);
    output[rank] = allocate<Fp8E4m3>(rank, output_elements);
    ready[rank] = allocate<uint32_t>(rank, ready_elements);
    upload(rank, peer[rank], host_peer[rank]);
    upload(rank, weight_nt[rank], host_weight[rank]);
    CUDA_CHECK(cudaMemset(staging[rank], 0, staging_elements));
    CUDA_CHECK(cudaMemset(
        output[rank], 0, output_elements * sizeof(Fp8E4m3)));
    CUDA_CHECK(cudaMemset(ready[rank], 0, ready_elements * sizeof(uint32_t)));
  }

  std::vector<fuse::Fp8A2AGemmParams> params(world);
  for (int rank = 0; rank < world; ++rank) {
    auto& p = params[rank];
    for (int source = 0; source < world; ++source) {
      p.peer_input[source] = peer[source];
    }
    p.input_staging = staging[rank];
    p.rhs_nt = weight_nt[rank];
    p.output = output[rank];
    p.ready = ready[rank];
    p.gemm = problem;
    p.route = route;
    p.route.rank = rank;
    p.num_comm_ctas = comm_ctas;
    p.lhs_policy = fuse::A2ALhsGemmPolicy::kM128N128;
    p.epoch = 1;
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(fuse::launch_a2a_gemm_fp8_cutlass(
        p, runtimes[rank].stream));
  }
  synchronize(runtimes, world);

  float maximum = 0.0f;
  for (int rank = 0; rank < world; ++rank) {
    std::vector<Fp8E4m3> expected_staging(staging_elements);
    for (int row = 0; row < local_tokens; ++row) {
      const int global_row = global_sequence_row(rank, row, local_tokens);
      for (int source = 0; source < world; ++source) {
        std::copy_n(
            host_peer[source].data() +
                static_cast<size_t>(global_row) * local_width,
            local_width,
            expected_staging.data() +
                static_cast<size_t>(row) * attention_width +
                source * local_width);
      }
    }
    expect_fp8_equal(
        download(rank, staging[rank], staging_elements),
        expected_staging,
        "FP8 OProj forward route rank " + std::to_string(rank));
    maximum = std::max(
        maximum,
        ([&]() {
          const auto actual = download(rank, output[rank], output_elements);
          const auto expected = matmul_rhs_nt(
              expected_staging,
              host_weight[rank],
              local_tokens,
              hidden,
              attention_width);
          expect_fp8_equal(
              actual,
              expected,
              "FP8 OProj forward rank " + std::to_string(rank));
          float max_abs = 0.0f;
          for (size_t index = 0; index < actual.size(); ++index) {
            max_abs = std::max(
                max_abs,
                std::abs(static_cast<float>(actual[index]) -
                         static_cast<float>(expected[index])));
          }
          return max_abs;
        })());
  }
  std::cout << "FP8 A2A+OProj CPU reference M=" << local_tokens
            << ": PASS max_abs=" << maximum << "\n";
}

void smoke_fp8_qkv_backward(
    int world,
    const std::vector<Runtime>& runtimes) {
  constexpr int local_tokens = 128;
  constexpr int hidden = 128;
  constexpr int q_heads = 4;
  constexpr int kv_heads = 2;
  constexpr int head_dim = 128;
  constexpr int comm_ctas = 8;
  const int global_tokens = local_tokens * world;
  const int q_local_width = q_heads / world * head_dim;
  const int kv_local_width = kv_heads / world * head_dim;
  const int packed_width = (q_heads + 2 * kv_heads) * head_dim;
  const size_t staging_elements =
      static_cast<size_t>(local_tokens) * packed_width;
  const size_t input_elements = static_cast<size_t>(local_tokens) * hidden;
  const size_t weight_elements = static_cast<size_t>(packed_width) * hidden;

  std::vector<std::vector<Fp8E4m3>> host_q(world), host_k(world), host_v(world);
  std::vector<std::vector<Fp8E4m3>> host_weight_nt(world), host_x(world);
  std::vector<Fp8E4m3*> q(world), k(world), v(world), staging(world);
  std::vector<Fp8E4m3*> weight_nt(world), dqkv_t(world), x_t(world);
  std::vector<Fp8E4m3*> dx(world), dw(world);
  std::vector<uint32_t*> ready(world), done(world);
  fuse::Fp8QkvBackwardDataParams shape{};
  shape.local_tokens = local_tokens;
  shape.hidden = hidden;
  shape.q_heads = q_heads;
  shape.kv_heads = kv_heads;
  shape.head_dim = head_dim;
  shape.world_size = world;
  const int64_t ready_elements = fuse::qkv_backward_fp8_ready_elements(shape);

  for (int rank = 0; rank < world; ++rank) {
    host_q[rank] = fp8_values(
        static_cast<size_t>(global_tokens) * q_local_width, 30 + rank);
    host_k[rank] = fp8_values(
        static_cast<size_t>(global_tokens) * kv_local_width, 40 + rank);
    host_v[rank] = fp8_values(
        static_cast<size_t>(global_tokens) * kv_local_width, 50 + rank);
    host_weight_nt[rank] = fp8_values(weight_elements, 60);
    host_x[rank] = fp8_values(input_elements, 70 + rank);
    q[rank] = allocate<Fp8E4m3>(rank, host_q[rank].size());
    k[rank] = allocate<Fp8E4m3>(rank, host_k[rank].size());
    v[rank] = allocate<Fp8E4m3>(rank, host_v[rank].size());
    staging[rank] = allocate<Fp8E4m3>(rank, staging_elements);
    weight_nt[rank] = allocate<Fp8E4m3>(rank, weight_elements);
    dqkv_t[rank] = allocate<Fp8E4m3>(rank, staging_elements);
    x_t[rank] = allocate<Fp8E4m3>(rank, input_elements);
    dx[rank] = allocate<Fp8E4m3>(rank, input_elements);
    dw[rank] = allocate<Fp8E4m3>(rank, weight_elements);
    ready[rank] = allocate<uint32_t>(rank, ready_elements);
    done[rank] = allocate<uint32_t>(
        rank, world * fuse::kReadyFlagStride);
    upload(rank, q[rank], host_q[rank]);
    upload(rank, k[rank], host_k[rank]);
    upload(rank, v[rank], host_v[rank]);
    upload(rank, weight_nt[rank], host_weight_nt[rank]);
    CUDA_CHECK(cudaMemset(staging[rank], 0, staging_elements));
    CUDA_CHECK(cudaMemset(dx[rank], 0, input_elements * sizeof(Fp8E4m3)));
    CUDA_CHECK(cudaMemset(dw[rank], 0, weight_elements * sizeof(Fp8E4m3)));
    CUDA_CHECK(cudaMemset(ready[rank], 0, ready_elements * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(
        done[rank], 0, world * fuse::kReadyFlagStride * sizeof(uint32_t)));
  }

  std::vector<fuse::Fp8QkvBackwardParams> params(world);
  for (int rank = 0; rank < world; ++rank) {
    auto& data = params[rank].data;
    data.grad_q = q[rank];
    data.grad_k = k[rank];
    data.grad_v = v[rank];
    for (int peer = 0; peer < world; ++peer) {
      data.peer_dqkv_staging[peer] = staging[peer];
      data.peer_ready[peer] = ready[peer];
      data.peer_done_epoch[peer] = done[peer];
    }
    data.weight_nt = weight_nt[rank];
    data.grad_input = dx[rank];
    data.local_tokens = local_tokens;
    data.hidden = hidden;
    data.q_heads = q_heads;
    data.kv_heads = kv_heads;
    data.head_dim = head_dim;
    data.world_size = world;
    data.rank = rank;
    data.num_comm_ctas = comm_ctas;
    data.gemm_policy = fuse::BackwardGemmPolicy::kM128N128;
    data.epoch = 1;
    params[rank].weight.grad_weight = dw[rank];
    params[rank].weight.local_tokens = local_tokens;
    params[rank].weight.hidden = hidden;
    params[rank].weight.q_heads = q_heads;
    params[rank].weight.kv_heads = kv_heads;
    params[rank].weight.head_dim = head_dim;
    params[rank].weight_mode = fuse::WeightGradientMode::kDeferred;
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(fuse::launch_qkv_backward_fp8(
        params[rank], runtimes[rank].stream));
  }
  synchronize(runtimes, world);

  float dgrad_maximum = 0.0f;
  float wgrad_maximum = 0.0f;
  for (int destination = 0; destination < world; ++destination) {
    std::vector<Fp8E4m3> expected(staging_elements);
    for (int source = 0; source < world; ++source) {
      for (int row = 0; row < local_tokens; ++row) {
        const int global_row = global_sequence_row(
            destination, row, local_tokens);
        std::copy_n(
            host_q[source].data() +
                static_cast<size_t>(global_row) * q_local_width,
            q_local_width,
            expected.data() + static_cast<size_t>(row) * packed_width +
                source * q_local_width);
        std::copy_n(
            host_k[source].data() +
                static_cast<size_t>(global_row) * kv_local_width,
            kv_local_width,
            expected.data() + static_cast<size_t>(row) * packed_width +
                q_heads * head_dim + source * kv_local_width);
        std::copy_n(
            host_v[source].data() +
                static_cast<size_t>(global_row) * kv_local_width,
            kv_local_width,
            expected.data() + static_cast<size_t>(row) * packed_width +
                (q_heads + kv_heads) * head_dim +
                source * kv_local_width);
      }
    }
    expect_fp8_equal(
        download(destination, staging[destination], staging_elements),
        expected,
        "FP8 QKV backward route rank " + std::to_string(destination));
    dgrad_maximum = std::max(
        dgrad_maximum,
        ([&]() {
          const auto actual = download(
              destination, dx[destination], input_elements);
          const auto reference = matmul_rhs_nt(
              expected,
              host_weight_nt[destination],
              local_tokens,
              hidden,
              packed_width);
          expect_fp8_equal(
              actual,
              reference,
              "FP8 QKV backward dgrad rank " +
                  std::to_string(destination));
          return 0.0f;
        })());

    const auto expected_t = transpose_fp8(expected, local_tokens, packed_width);
    const auto input_t = transpose_fp8(
        host_x[destination], local_tokens, hidden);
    upload(destination, dqkv_t[destination], expected_t);
    upload(destination, x_t[destination], input_t);
    params[destination].weight.dqkv_t = dqkv_t[destination];
    params[destination].weight.saved_input_t = x_t[destination];
    CUDA_CHECK(cudaSetDevice(destination));
    CUDA_CHECK(fuse::launch_qkv_backward_fp8_weight(
        params[destination].weight, runtimes[destination].stream));
    CUDA_CHECK(cudaStreamSynchronize(runtimes[destination].stream));
    const auto expected_dw = wgrad_from_transposes(
        expected_t, input_t, packed_width, hidden, local_tokens);
    wgrad_maximum = std::max(
        wgrad_maximum,
        ([&]() {
          const auto actual = download(
              destination, dw[destination], weight_elements);
          expect_fp8_equal(
              actual,
              expected_dw,
              "FP8 QKV backward wgrad rank " +
                  std::to_string(destination));
          return 0.0f;
        })());
    params[destination].weight.beta = 1.0f;
    CUDA_CHECK(fuse::launch_qkv_backward_fp8_weight(
        params[destination].weight, runtimes[destination].stream));
    CUDA_CHECK(cudaStreamSynchronize(runtimes[destination].stream));
    expect_fp8_equal(
        download(destination, dw[destination], weight_elements),
        scaled(expected_dw, 2.0f),
        "FP8 QKV backward beta rank " + std::to_string(destination));
  }
  std::cout << "FP8 QKV backward CPU reference: PASS dgrad_max_abs="
            << dgrad_maximum << " wgrad_max_abs=" << wgrad_maximum << "\n";
}

void smoke_fp8_oproj_backward(
    int world,
    const std::vector<Runtime>& runtimes) {
  constexpr int local_tokens = 128;
  constexpr int hidden = 128;
  constexpr int q_heads = 4;
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

  std::vector<std::vector<Fp8E4m3>> host_dy(world), host_weight_nt(world);
  std::vector<std::vector<Fp8E4m3>> host_attention(world);
  std::vector<Fp8E4m3*> dy(world), weight_nt(world), dy_t(world), attention_t(world);
  std::vector<Fp8E4m3*> local_da(world), peer_da(world), dw(world);
  std::vector<uint32_t*> ready(world), done(world);
  fuse::Fp8OprojBackwardDataParams shape{};
  shape.local_tokens = local_tokens;
  shape.hidden = hidden;
  shape.q_heads = q_heads;
  shape.head_dim = head_dim;
  shape.world_size = world;
  const int64_t ready_elements = fuse::oproj_backward_fp8_ready_elements(shape);

  for (int rank = 0; rank < world; ++rank) {
    host_dy[rank] = fp8_values(dy_elements, 80 + rank);
    host_weight_nt[rank] = fp8_values(weight_elements, 90);
    host_attention[rank] = fp8_values(attention_elements, 100 + rank);
    dy[rank] = allocate<Fp8E4m3>(rank, dy_elements);
    weight_nt[rank] = allocate<Fp8E4m3>(rank, weight_elements);
    dy_t[rank] = allocate<Fp8E4m3>(rank, dy_elements);
    attention_t[rank] = allocate<Fp8E4m3>(rank, attention_elements);
    local_da[rank] = allocate<Fp8E4m3>(rank, attention_elements);
    peer_da[rank] = allocate<Fp8E4m3>(rank, peer_elements);
    dw[rank] = allocate<Fp8E4m3>(rank, weight_elements);
    ready[rank] = allocate<uint32_t>(rank, ready_elements);
    done[rank] = allocate<uint32_t>(
        rank, world * fuse::kReadyFlagStride);
    upload(rank, dy[rank], host_dy[rank]);
    upload(rank, weight_nt[rank], host_weight_nt[rank]);
    upload(rank, dy_t[rank], transpose_fp8(host_dy[rank], local_tokens, hidden));
    upload(
        rank,
        attention_t[rank],
        transpose_fp8(host_attention[rank], local_tokens, attention_width));
    CUDA_CHECK(cudaMemset(
        local_da[rank], 0, attention_elements * sizeof(Fp8E4m3)));
    CUDA_CHECK(cudaMemset(
        peer_da[rank], 0, peer_elements * sizeof(Fp8E4m3)));
    CUDA_CHECK(cudaMemset(dw[rank], 0, weight_elements * sizeof(Fp8E4m3)));
    CUDA_CHECK(cudaMemset(ready[rank], 0, ready_elements * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(
        done[rank], 0, world * fuse::kReadyFlagStride * sizeof(uint32_t)));
  }

  std::vector<fuse::Fp8OprojBackwardParams> params(world);
  for (int rank = 0; rank < world; ++rank) {
    auto& data = params[rank].data;
    data.grad_output = dy[rank];
    data.weight_nt = weight_nt[rank];
    data.local_grad_attention = local_da[rank];
    for (int peer = 0; peer < world; ++peer) {
      data.peer_grad_attention[peer] = peer_da[peer];
      data.peer_done_epoch[peer] = done[peer];
    }
    data.ready = ready[rank];
    data.local_tokens = local_tokens;
    data.hidden = hidden;
    data.q_heads = q_heads;
    data.head_dim = head_dim;
    data.world_size = world;
    data.rank = rank;
    data.num_comm_ctas = comm_ctas;
    data.gemm_policy = fuse::BackwardGemmPolicy::kM128N128;
    data.epoch = 1;
    auto& weight = params[rank].weight;
    weight.grad_output_t = dy_t[rank];
    weight.saved_attention_t = attention_t[rank];
    weight.grad_weight = dw[rank];
    weight.local_tokens = local_tokens;
    weight.hidden = hidden;
    weight.q_heads = q_heads;
    weight.head_dim = head_dim;
    params[rank].weight_mode = fuse::WeightGradientMode::kDeferred;
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(fuse::launch_oproj_backward_fp8(
        params[rank], runtimes[rank].stream));
  }
  synchronize(runtimes, world);

  float dgrad_maximum = 0.0f;
  float wgrad_maximum = 0.0f;
  std::vector<std::vector<Fp8E4m3>> expected_da(world);
  for (int source = 0; source < world; ++source) {
    expected_da[source] = matmul_rhs_nt(
        host_dy[source],
        host_weight_nt[source],
        local_tokens,
        attention_width,
        hidden);
    dgrad_maximum = std::max(
        dgrad_maximum,
        ([&]() {
          const auto actual = download(
              source, local_da[source], attention_elements);
          expect_fp8_equal(
              actual,
              expected_da[source],
              "FP8 OProj backward dgrad rank " +
                  std::to_string(source));
          return 0.0f;
        })());
  }
  for (int destination = 0; destination < world; ++destination) {
    std::vector<Fp8E4m3> expected_peer(peer_elements);
    for (int source = 0; source < world; ++source) {
      for (int row = 0; row < local_tokens; ++row) {
        std::copy_n(
            expected_da[source].data() +
                static_cast<size_t>(row) * attention_width +
                destination * local_width,
            local_width,
            expected_peer.data() +
                static_cast<size_t>(global_sequence_row(
                    source, row, local_tokens)) * local_width);
      }
    }
    expect_fp8_equal(
        download(destination, peer_da[destination], peer_elements),
        expected_peer,
        "FP8 OProj backward route rank " + std::to_string(destination));
    CUDA_CHECK(cudaSetDevice(destination));
    CUDA_CHECK(fuse::launch_oproj_backward_fp8_weight(
        params[destination].weight, runtimes[destination].stream));
    CUDA_CHECK(cudaStreamSynchronize(runtimes[destination].stream));
    const auto expected_dw = wgrad_from_transposes(
        transpose_fp8(host_dy[destination], local_tokens, hidden),
        transpose_fp8(
            host_attention[destination], local_tokens, attention_width),
        hidden,
        attention_width,
        local_tokens);
    wgrad_maximum = std::max(
        wgrad_maximum,
        ([&]() {
          const auto actual = download(
              destination, dw[destination], weight_elements);
          expect_fp8_equal(
              actual,
              expected_dw,
              "FP8 OProj backward wgrad rank " +
                  std::to_string(destination));
          return 0.0f;
        })());
    params[destination].weight.beta = 1.0f;
    CUDA_CHECK(fuse::launch_oproj_backward_fp8_weight(
        params[destination].weight, runtimes[destination].stream));
    CUDA_CHECK(cudaStreamSynchronize(runtimes[destination].stream));
    expect_fp8_equal(
        download(destination, dw[destination], weight_elements),
        scaled(expected_dw, 2.0f),
        "FP8 OProj backward beta rank " + std::to_string(destination));
  }
  std::cout << "FP8 OProj backward CPU reference: PASS dgrad_max_abs="
            << dgrad_maximum << " wgrad_max_abs=" << wgrad_maximum << "\n";
}

}  // namespace

int main() {
  try {
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices < 2) {
      throw std::runtime_error("fp8_smoke requires at least two GPUs");
    }
    constexpr int world = 2;
    enable_peer_access(world);
    const auto runtimes = make_runtimes(world);
    // M=128 exercises the bulk/TMA route; M=150 forces the vector fallback.
    smoke_fp8_oproj_forward(world, runtimes, 128);
    smoke_fp8_oproj_forward(world, runtimes, 150);
    smoke_fp8_qkv_backward(world, runtimes);
    smoke_fp8_oproj_backward(world, runtimes);
    std::cout << "fp8_smoke: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "fp8_smoke: FAIL: " << error.what() << "\n";
    return 1;
  }
}
