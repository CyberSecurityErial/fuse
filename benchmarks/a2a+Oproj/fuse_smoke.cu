#include "fuse/kernels.h"

#include <cublas_v2.h>
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
using fuse::Fp8E4m3;
uint32_t g_epoch_iterations = 8;

#define CUDA_CHECK(expr) check_cuda((expr), #expr, __FILE__, __LINE__)
#define CUBLAS_CHECK(expr) check_cublas((expr), #expr, __FILE__, __LINE__)

void check_cuda(cudaError_t status, const char* expr, const char* file, int line) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(file) + ":" + std::to_string(line) + " " + expr + ": " +
        cudaGetErrorString(status));
  }
}

void check_cublas(cublasStatus_t status, const char* expr, const char* file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
        std::string(file) + ":" + std::to_string(line) + " " + expr +
        ": cublas status " + std::to_string(static_cast<int>(status)));
  }
}

template <class T>
T* device_alloc(int device, size_t elements) {
  CUDA_CHECK(cudaSetDevice(device));
  T* ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&ptr, elements * sizeof(T)));
  return ptr;
}

template <class T>
void upload(int device, T* destination, const std::vector<T>& source) {
  CUDA_CHECK(cudaSetDevice(device));
  CUDA_CHECK(cudaMemcpy(
      destination, source.data(), source.size() * sizeof(T), cudaMemcpyHostToDevice));
}

template <class T>
std::vector<T> download(int device, const T* source, size_t elements) {
  CUDA_CHECK(cudaSetDevice(device));
  std::vector<T> host(elements);
  CUDA_CHECK(cudaMemcpy(host.data(), source, elements * sizeof(T), cudaMemcpyDeviceToHost));
  return host;
}

std::vector<Bf16> make_values(size_t elements, int seed) {
  std::vector<Bf16> result(elements);
  for (size_t i = 0; i < elements; ++i) {
    const int value = static_cast<int>((i * 17 + seed * 13) % 19) - 9;
    result[i] = Bf16(static_cast<float>(value) / 32.0f);
  }
  return result;
}

std::vector<Fp8E4m3> make_fp8_values(size_t elements, int seed) {
  std::vector<Fp8E4m3> result(elements);
  for (size_t i = 0; i < elements; ++i) {
    const int value = static_cast<int>((i * 17 + seed * 13) % 17) - 8;
    result[i] = Fp8E4m3(static_cast<float>(value) / 4.0f);
  }
  return result;
}

struct RankRuntime {
  cudaStream_t stream = nullptr;
  cublasHandle_t blas = nullptr;
};

std::vector<RankRuntime> create_runtimes(int world) {
  std::vector<RankRuntime> runtimes(world);
  for (int rank = 0; rank < world; ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    int least_priority = 0;
    int greatest_priority = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
    CUDA_CHECK(cudaStreamCreateWithPriority(
        &runtimes[rank].stream, cudaStreamNonBlocking, greatest_priority));
    CUBLAS_CHECK(cublasCreate(&runtimes[rank].blas));
    CUBLAS_CHECK(cublasSetStream(runtimes[rank].blas, runtimes[rank].stream));
    CUBLAS_CHECK(cublasSetMathMode(runtimes[rank].blas, CUBLAS_TENSOR_OP_MATH));
  }
  return runtimes;
}

void enable_peer_access(int world) {
  for (int source = 0; source < world; ++source) {
    CUDA_CHECK(cudaSetDevice(source));
    for (int peer = 0; peer < world; ++peer) {
      if (source == peer) {
        continue;
      }
      int can_access = 0;
      CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, source, peer));
      if (!can_access) {
        throw std::runtime_error(
            "P2P unavailable from GPU " + std::to_string(source) + " to " +
            std::to_string(peer));
      }
      cudaError_t status = cudaDeviceEnablePeerAccess(peer, 0);
      if (status == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
      } else {
        CUDA_CHECK(status);
      }
    }
  }
}

void sync_all(const std::vector<RankRuntime>& runtimes) {
  for (int rank = 0; rank < static_cast<int>(runtimes.size()); ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaStreamSynchronize(runtimes[rank].stream));
  }
}

void cublas_nt(
    cublasHandle_t handle,
    int m,
    int n,
    int k,
    int batches,
    const Bf16* a,
    const Bf16* b_nt,
    Bf16* d,
    int64_t a_row = 0,
    int64_t a_batch = 0,
    int64_t b_row = 0,
    int64_t b_batch = 0,
    int64_t d_row = 0,
    int64_t d_batch = 0) {
  a_row = a_row == 0 ? k : a_row;
  a_batch = a_batch == 0 ? static_cast<int64_t>(m) * a_row : a_batch;
  b_row = b_row == 0 ? k : b_row;
  b_batch = b_batch == 0 ? static_cast<int64_t>(n) * b_row : b_batch;
  d_row = d_row == 0 ? n : d_row;
  d_batch = d_batch == 0 ? static_cast<int64_t>(m) * d_row : d_batch;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  CUBLAS_CHECK(cublasGemmStridedBatchedEx(
      handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      n,
      m,
      k,
      &alpha,
      b_nt,
      CUDA_R_16BF,
      b_row,
      b_batch,
      a,
      CUDA_R_16BF,
      a_row,
      a_batch,
      &beta,
      d,
      CUDA_R_16BF,
      d_row,
      d_batch,
      batches,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

float max_abs_diff(const std::vector<Bf16>& lhs, const std::vector<Bf16>& rhs) {
  if (lhs.size() != rhs.size()) {
    throw std::runtime_error("comparison size mismatch");
  }
  float maximum = 0.0f;
  for (size_t i = 0; i < lhs.size(); ++i) {
    maximum = std::max(maximum, std::abs(static_cast<float>(lhs[i]) - static_cast<float>(rhs[i])));
  }
  return maximum;
}

struct ErrorMetrics {
  float max_abs = 0.0f;
  float max_relative = 0.0f;
  double relative_l2 = 0.0;
};

ErrorMetrics error_metrics(
    const std::vector<Bf16>& actual,
    const std::vector<Bf16>& expected) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error("comparison size mismatch");
  }
  ErrorMetrics result{};
  double squared_error = 0.0;
  double squared_reference = 0.0;
  for (size_t index = 0; index < actual.size(); ++index) {
    const float got = static_cast<float>(actual[index]);
    const float want = static_cast<float>(expected[index]);
    const float error = std::abs(got - want);
    result.max_abs = std::max(result.max_abs, error);
    result.max_relative =
        std::max(result.max_relative, error / std::max(std::abs(want), 1.0e-6f));
    squared_error += static_cast<double>(error) * error;
    squared_reference += static_cast<double>(want) * want;
  }
  result.relative_l2 =
      std::sqrt(squared_error / std::max(squared_reference, 1.0e-30));
  return result;
}

void smoke_qkv_gqa_route() {
  struct Case {
    int world;
    int q_heads;
    int kv_heads;
    int head_dim;
  };
  for (const Case test : {
           Case{2, 8, 8, 64},
           Case{2, 8, 4, 96},
           Case{4, 16, 4, 80},
           Case{4, 32, 4, 128}}) {
    fuse::UlyssesRoute route{};
    route.world_size = test.world;
    route.q_heads = test.q_heads;
    route.kv_heads = test.kv_heads;
    route.head_dim = test.head_dim;
    route.kind = fuse::RouteKind::kQkvGqaPack;
    const int width = (test.q_heads + 2 * test.kv_heads) * test.head_dim;
    for (int feature = 0; feature < width; ++feature) {
      const auto address = fuse::map_qkv_gqa_feature(route, feature);
      if (!address.valid() ||
          fuse::qkv_gqa_global_feature(
              route, address.owner_rank, address.local_feature) != feature) {
        throw std::runtime_error(
            "QKV GQA pack mapping is not invertible at feature=" +
            std::to_string(feature) + " owner=" +
            std::to_string(address.owner_rank) + " local=" +
            std::to_string(address.local_feature) + " inverse=" +
            std::to_string(fuse::qkv_gqa_global_feature(
                route, address.owner_rank, address.local_feature)) +
            " world=" + std::to_string(route.world_size) +
            " q=" + std::to_string(route.q_heads) +
            " kv=" + std::to_string(route.kv_heads) +
            " d=" + std::to_string(route.head_dim));
      }
    }
  }
  fuse::UlyssesRoute uneven{};
  uneven.world_size = 4;
  uneven.q_heads = 10;
  uneven.kv_heads = 4;
  uneven.head_dim = 128;
  if (fuse::map_qkv_gqa_feature(uneven, 0).valid()) {
    throw std::runtime_error("uneven QKV split must be rejected");
  }
  std::cout << "QKV GQA pack route: PASS\n";
}

void smoke_qkv_gqa_pack(
    int world,
    const std::vector<RankRuntime>& runtimes,
    int batch,
    int seq_local,
    int q_heads,
    int kv_heads,
    int head_dim,
    bool padded,
    bool defer_v = true) {
  constexpr int k = 256;
  constexpr int num_comm_ctas = 8;
  const int m = batch * seq_local;
  const int n = (q_heads + 2 * kv_heads) * head_dim;
  const int global_seq = seq_local * world;
  const int q_local_width = q_heads / world * head_dim;
  const int kv_local_width = kv_heads / world * head_dim;
  const int local_width =
      q_local_width + (defer_v ? 1 : 2) * kv_local_width;
  const int64_t a_row = padded ? k + 16 : k;
  const int64_t a_batch = a_row * m + (padded ? 16 : 0);
  const int64_t b_row = padded ? k + 16 : k;
  const int64_t b_batch = b_row * n + (padded ? 16 : 0);
  const int64_t d_row = padded ? n + 16 : n;
  const int64_t d_batch = d_row * m + (padded ? 16 : 0);
  const fuse::GemmProblem qkv_problem{m, n, k, 1};
  const auto traits = fuse::qkv_cutlass_kernel_traits(qkv_problem);
  const int ready_count =
      ((m + traits.block_m - 1) / traits.block_m) *
      ((n + traits.block_n - 1) / traits.block_n) * fuse::kReadyFlagStride;
  const size_t peer_elements =
      static_cast<size_t>(batch) * global_seq * local_width;

  std::vector<Bf16*> lhs(world);
  std::vector<Bf16*> rhs_nt(world);
  std::vector<Bf16*> local_output(world);
  std::vector<Bf16*> reference_local(world);
  std::vector<Bf16*> peer_output(world);
  std::vector<uint32_t*> ready(world);
  std::vector<uint32_t*> consumed_epoch(world);

  for (int rank = 0; rank < world; ++rank) {
    std::vector<Bf16> host_lhs(static_cast<size_t>(a_batch));
    std::vector<Bf16> host_rhs(static_cast<size_t>(b_batch));
    const auto packed_lhs = make_values(static_cast<size_t>(m) * k, rank + 401);
    const auto packed_rhs = make_values(static_cast<size_t>(n) * k, rank + 503);
    for (int row = 0; row < m; ++row) {
      std::copy_n(
          packed_lhs.data() + static_cast<size_t>(row) * k,
          k,
          host_lhs.data() + static_cast<size_t>(row) * a_row);
    }
    const int q_width = q_heads * head_dim;
    const int k_width = kv_heads * head_dim;
    for (int row = 0; row < n; ++row) {
      int logical_row = row;
      if (row < q_width + k_width) {
        const int segment = row < q_width ? 0 : 1;
        const int segment_base = segment == 0 ? 0 : q_width;
        const int physical_feature = row - segment_base;
        const int physical_head = physical_feature / head_dim;
        const int head_offset = physical_feature % head_dim;
        const int global_heads = segment == 0 ? q_heads : kv_heads;
        const int local_heads = global_heads / world;
        const int owner = physical_head % world;
        const int local_head = physical_head / world;
        const int logical_head = owner * local_heads + local_head;
        logical_row =
            segment_base + logical_head * head_dim + head_offset;
      }
      std::copy_n(
          packed_rhs.data() + static_cast<size_t>(logical_row) * k,
          k,
          host_rhs.data() + static_cast<size_t>(row) * b_row);
    }
    lhs[rank] = device_alloc<Bf16>(rank, host_lhs.size());
    rhs_nt[rank] = device_alloc<Bf16>(rank, host_rhs.size());
    local_output[rank] = device_alloc<Bf16>(rank, static_cast<size_t>(d_batch));
    reference_local[rank] = device_alloc<Bf16>(rank, static_cast<size_t>(d_batch));
    peer_output[rank] = device_alloc<Bf16>(rank, peer_elements);
    ready[rank] = device_alloc<uint32_t>(rank, ready_count);
    consumed_epoch[rank] = device_alloc<uint32_t>(
        rank, world * fuse::kReadyFlagStride);
    upload(rank, lhs[rank], host_lhs);
    upload(rank, rhs_nt[rank], host_rhs);
    CUDA_CHECK(cudaMemsetAsync(
        local_output[rank],
        0,
        static_cast<size_t>(d_batch) * sizeof(Bf16),
        runtimes[rank].stream));
    CUDA_CHECK(cudaMemsetAsync(
        reference_local[rank],
        0,
        static_cast<size_t>(d_batch) * sizeof(Bf16),
        runtimes[rank].stream));
    CUDA_CHECK(cudaMemsetAsync(
        peer_output[rank],
        0,
        peer_elements * sizeof(Bf16),
        runtimes[rank].stream));
    CUDA_CHECK(cudaMemsetAsync(
        ready[rank],
        0,
        ready_count * sizeof(uint32_t),
        runtimes[rank].stream));
    CUDA_CHECK(cudaMemsetAsync(
        consumed_epoch[rank],
        0,
        world * fuse::kReadyFlagStride * sizeof(uint32_t),
        runtimes[rank].stream));
    CUDA_CHECK(cudaSetDevice(rank));
    cublas_nt(
        runtimes[rank].blas,
        m,
        n,
        k,
        1,
        lhs[rank],
        rhs_nt[rank],
        reference_local[rank],
        a_row,
        a_batch,
        b_row,
        b_batch,
        d_row,
        d_batch);
  }
  sync_all(runtimes);

  std::vector<fuse::GemmA2AParams> params(world);
  for (int rank = 0; rank < world; ++rank) {
    params[rank].lhs = lhs[rank];
    params[rank].rhs_nt = rhs_nt[rank];
    params[rank].local_output = local_output[rank];
    for (int peer = 0; peer < world; ++peer) {
      params[rank].peer_output[peer] = peer_output[peer];
      params[rank].peer_route_done_epoch[peer] = consumed_epoch[peer];
    }
    params[rank].ready = ready[rank];
    params[rank].gemm = {m, n, k, 1};
    params[rank].gemm.stride_a = {a_row, 1, a_batch};
    params[rank].gemm.stride_b = {b_row, 1, b_batch};
    params[rank].gemm.stride_d = {d_row, 1, d_batch};
    params[rank].route.world_size = world;
    params[rank].route.rank = rank;
    params[rank].route.batch = batch;
    params[rank].route.seq_local = seq_local;
    params[rank].route.global_seq = global_seq;
    params[rank].route.q_heads = q_heads;
    params[rank].route.kv_heads = kv_heads;
    params[rank].route.head_dim = head_dim;
    params[rank].route.qkv_peer_interleaved = true;
    params[rank].route.defer_v_a2a = defer_v;
    params[rank].route.kind = fuse::RouteKind::kQkvGqaPack;
    params[rank].route.direction = fuse::RouteDirection::kForward;
    params[rank].num_comm_ctas = num_comm_ctas;
    params[rank].epoch = 1;
  }

  for (uint32_t epoch = 1; epoch <= g_epoch_iterations; ++epoch) {
    for (int rank = 0; rank < world; ++rank) {
      CUDA_CHECK(cudaSetDevice(rank));
      params[rank].epoch = epoch;
      CUDA_CHECK(fuse::launch_gemm_a2a_cutlass(
          params[rank], runtimes[rank].stream));
    }
    sync_all(runtimes);
  }

  std::vector<std::vector<Bf16>> host_reference_local(world);
  ErrorMetrics local_worst{};
  for (int rank = 0; rank < world; ++rank) {
    host_reference_local[rank] =
        download(rank, reference_local[rank], static_cast<size_t>(d_batch));
    const auto actual_local =
        download(rank, local_output[rank], static_cast<size_t>(d_batch));
    const ErrorMetrics errors = error_metrics(actual_local, host_reference_local[rank]);
    if (errors.max_abs > 0.25f) {
      int shown = 0;
      for (size_t index = 0; index < actual_local.size() && shown < 8; ++index) {
        const float got = static_cast<float>(actual_local[index]);
        const float want = static_cast<float>(host_reference_local[rank][index]);
        if (std::abs(got - want) > 0.25f) {
          std::cout << "local QKV mismatch rank=" << rank
                    << " i=" << index << " actual=" << got
                    << " expected=" << want << "\n";
          ++shown;
        }
      }
    }
    local_worst.max_abs = std::max(local_worst.max_abs, errors.max_abs);
    local_worst.max_relative =
        std::max(local_worst.max_relative, errors.max_relative);
    local_worst.relative_l2 = std::max(local_worst.relative_l2, errors.relative_l2);
  }

  ErrorMetrics worst{};
  for (int destination = 0; destination < world; ++destination) {
    std::vector<Bf16> expected(peer_elements);
    for (int source = 0; source < world; ++source) {
      for (int batch_index = 0; batch_index < batch; ++batch_index) {
        for (int sequence = 0; sequence < seq_local; ++sequence) {
          const int source_row = batch_index * seq_local + sequence;
          const int global_sequence = source * seq_local + sequence;
          for (int segment = 0; segment < (defer_v ? 2 : 3); ++segment) {
            const int segment_heads = segment == 0 ? q_heads : kv_heads;
            const int heads_per_rank = segment_heads / world;
            const int global_feature_base = segment == 0
                ? 0
                : (q_heads + (segment == 2 ? kv_heads : 0)) * head_dim;
            for (int local_head = 0; local_head < heads_per_rank; ++local_head) {
              const int physical_head = segment < 2
                  ? local_head * world + destination
                  : destination * heads_per_rank + local_head;
              const Bf16* source_feature =
                  host_reference_local[source].data() +
                  static_cast<size_t>(source_row) * d_row + global_feature_base +
                  physical_head * head_dim;
              const size_t logical_row =
                  static_cast<size_t>(batch_index) * global_seq + global_sequence;
              Bf16* destination_feature = nullptr;
              if (defer_v) {
                const int local_feature_base =
                    segment == 0 ? 0 : q_local_width;
                destination_feature = expected.data() +
                    logical_row * local_width + local_feature_base +
                    local_head * head_dim;
              } else {
                const size_t segment_rows =
                    static_cast<size_t>(batch) * global_seq;
                const size_t segment_base = segment == 0
                    ? 0
                    : segment_rows * q_local_width +
                        (segment == 2 ? segment_rows * kv_local_width : 0);
                const int segment_width =
                    segment == 0 ? q_local_width : kv_local_width;
                destination_feature = expected.data() + segment_base +
                    logical_row * segment_width + local_head * head_dim;
              }
              std::copy_n(source_feature, head_dim, destination_feature);
            }
          }
        }
      }
    }
    const auto actual =
        download(destination, peer_output[destination], peer_elements);
    const ErrorMetrics errors = error_metrics(actual, expected);
    if (errors.max_abs > 0.25f && destination == 0) {
      int shown = 0;
      for (size_t index = 0; index < actual.size() && shown < 8; ++index) {
        const float got = static_cast<float>(actual[index]);
        const float want = static_cast<float>(expected[index]);
        if (std::abs(got - want) > 0.25f) {
          std::cout << "QKV mismatch i=" << index << " actual=" << got
                    << " expected=" << want << "\n";
          ++shown;
        }
      }
    }
    worst.max_abs = std::max(worst.max_abs, errors.max_abs);
    worst.max_relative = std::max(worst.max_relative, errors.max_relative);
    worst.relative_l2 = std::max(worst.relative_l2, errors.relative_l2);
  }
  std::cout << "QKV-GQA GEMM->A2A CP=" << world << " B=" << batch
            << " S_local=" << seq_local << " Hq/Hkv="
            << q_heads / kv_heads << " D=" << head_dim
            << " padded=" << padded << " defer_v=" << defer_v
            << " route_max_abs=" << worst.max_abs
            << " route_rel_l2=" << worst.relative_l2
            << " local_qkv_max_abs=" << local_worst.max_abs
            << " local_qkv_rel_l2=" << local_worst.relative_l2 << "\n";
  if (worst.max_abs > 0.25f || local_worst.max_abs > 0.25f) {
    throw std::runtime_error("QKV-GQA GEMM->A2A correctness failed");
  }

  fuse::GemmA2AParams uneven = params[0];
  uneven.route.q_heads += 1;
  CUDA_CHECK(cudaSetDevice(0));
  if (fuse::launch_gemm_a2a_cutlass(uneven, runtimes[0].stream) !=
      cudaErrorNotSupported) {
    throw std::runtime_error("uneven production QKV split must be rejected");
  }
}

void smoke_fp8_qkv_gqa_pack(
    int world,
    const std::vector<RankRuntime>& runtimes,
    int batch,
    int seq_local,
    int q_heads,
    int kv_heads,
    int head_dim,
    bool padded) {
  constexpr int k = 256;
  constexpr int num_comm_ctas = 8;
  const int m = batch * seq_local;
  const int n = (q_heads + 2 * kv_heads) * head_dim;
  const int global_seq = seq_local * world;
  const int q_local_width = q_heads / world * head_dim;
  const int kv_local_width = kv_heads / world * head_dim;
  const int local_width = q_local_width + kv_local_width;
  const int64_t a_row = padded ? k + 16 : k;
  const int64_t a_batch = a_row * m + (padded ? 16 : 0);
  const int64_t b_row = padded ? k + 16 : k;
  const int64_t b_batch = b_row * n + (padded ? 16 : 0);
  const int64_t d_row = padded ? n + 16 : n;
  const int64_t d_batch = d_row * m + (padded ? 16 : 0);
  const auto traits = fuse::fp8_cutlass_kernel_traits();
  const int ready_count =
      ((m + traits.block_m - 1) / traits.block_m) *
      ((n + traits.block_n - 1) / traits.block_n) * fuse::kReadyFlagStride;
  const size_t peer_elements =
      static_cast<size_t>(batch) * global_seq * local_width;

  std::vector<Fp8E4m3*> lhs(world);
  std::vector<Fp8E4m3*> rhs_nt(world);
  std::vector<Bf16*> local_output(world);
  std::vector<Bf16*> reference_local(world);
  std::vector<Bf16*> peer_output(world);
  std::vector<uint32_t*> ready(world);
  std::vector<uint32_t*> consumed_epoch(world);
  std::vector<fuse::Fp8GemmA2AParams> params(world);

  for (int rank = 0; rank < world; ++rank) {
    std::vector<Fp8E4m3> host_lhs(static_cast<size_t>(a_batch));
    std::vector<Fp8E4m3> host_rhs(static_cast<size_t>(b_batch));
    const auto packed_lhs = make_fp8_values(static_cast<size_t>(m) * k, rank + 809);
    const auto packed_rhs = make_fp8_values(static_cast<size_t>(n) * k, rank + 907);
    for (int row = 0; row < m; ++row) {
      std::copy_n(
          packed_lhs.data() + static_cast<size_t>(row) * k,
          k,
          host_lhs.data() + static_cast<size_t>(row) * a_row);
    }
    for (int row = 0; row < n; ++row) {
      std::copy_n(
          packed_rhs.data() + static_cast<size_t>(row) * k,
          k,
          host_rhs.data() + static_cast<size_t>(row) * b_row);
    }
    lhs[rank] = device_alloc<Fp8E4m3>(rank, host_lhs.size());
    rhs_nt[rank] = device_alloc<Fp8E4m3>(rank, host_rhs.size());
    local_output[rank] = device_alloc<Bf16>(rank, static_cast<size_t>(d_batch));
    reference_local[rank] = device_alloc<Bf16>(rank, static_cast<size_t>(d_batch));
    peer_output[rank] = device_alloc<Bf16>(rank, peer_elements);
    ready[rank] = device_alloc<uint32_t>(rank, ready_count);
    consumed_epoch[rank] = device_alloc<uint32_t>(
        rank, world * fuse::kReadyFlagStride);
    upload(rank, lhs[rank], host_lhs);
    upload(rank, rhs_nt[rank], host_rhs);
    CUDA_CHECK(cudaMemsetAsync(
        local_output[rank],
        0,
        static_cast<size_t>(d_batch) * sizeof(Bf16),
        runtimes[rank].stream));
    CUDA_CHECK(cudaMemsetAsync(
        reference_local[rank],
        0,
        static_cast<size_t>(d_batch) * sizeof(Bf16),
        runtimes[rank].stream));
    CUDA_CHECK(cudaMemsetAsync(
        peer_output[rank],
        0,
        peer_elements * sizeof(Bf16),
        runtimes[rank].stream));
    CUDA_CHECK(cudaMemsetAsync(
        ready[rank],
        0,
        ready_count * sizeof(uint32_t),
        runtimes[rank].stream));
    CUDA_CHECK(cudaMemsetAsync(
        consumed_epoch[rank],
        0,
        world * fuse::kReadyFlagStride * sizeof(uint32_t),
        runtimes[rank].stream));

    params[rank].lhs = lhs[rank];
    params[rank].rhs_nt = rhs_nt[rank];
    params[rank].local_output = local_output[rank];
    for (int peer = 0; peer < world; ++peer) {
      params[rank].peer_output[peer] = peer_output[peer];
    }
    params[rank].ready = ready[rank];
    params[rank].gemm = {m, n, k, 1};
    params[rank].gemm.stride_a = {a_row, 1, a_batch};
    params[rank].gemm.stride_b = {b_row, 1, b_batch};
    params[rank].gemm.stride_d = {d_row, 1, d_batch};
    params[rank].gemm.input_dtype = fuse::DType::kFloat8E4M3;
    params[rank].gemm.weight_dtype = fuse::DType::kFloat8E4M3;
    params[rank].route.world_size = world;
    params[rank].route.rank = rank;
    params[rank].route.batch = batch;
    params[rank].route.seq_local = seq_local;
    params[rank].route.global_seq = global_seq;
    params[rank].route.q_heads = q_heads;
    params[rank].route.kv_heads = kv_heads;
    params[rank].route.head_dim = head_dim;
    params[rank].route.defer_v_a2a = true;
    params[rank].route.kind = fuse::RouteKind::kQkvGqaPack;
    params[rank].route.direction = fuse::RouteDirection::kForward;
    params[rank].num_comm_ctas = num_comm_ctas;
    params[rank].epoch = 1;
    params[rank].alpha = 0.5f;

    fuse::Fp8GemmA2AParams reference_params = params[rank];
    reference_params.local_output = reference_local[rank];
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(fuse::launch_dense_fp8_cutlass_reference(
        reference_params, runtimes[rank].stream));
  }
  sync_all(runtimes);

  for (int rank = 0; rank < world; ++rank) {
    for (int peer = 0; peer < world; ++peer) {
      params[rank].peer_output[peer] = peer_output[peer];
      params[rank].peer_route_done_epoch[peer] = consumed_epoch[peer];
    }
  }

  for (uint32_t epoch = 1; epoch <= g_epoch_iterations; ++epoch) {
    for (int rank = 0; rank < world; ++rank) {
      CUDA_CHECK(cudaSetDevice(rank));
      params[rank].epoch = epoch;
      CUDA_CHECK(fuse::launch_gemm_a2a_fp8_cutlass(
          params[rank], runtimes[rank].stream));
    }
    sync_all(runtimes);
  }

  std::vector<std::vector<Bf16>> host_reference_local(world);
  std::vector<Bf16> rank0_actual_local;
  float local_worst = 0.0f;
  for (int rank = 0; rank < world; ++rank) {
    host_reference_local[rank] =
        download(rank, reference_local[rank], static_cast<size_t>(d_batch));
    const auto actual_local =
        download(rank, local_output[rank], static_cast<size_t>(d_batch));
    if (rank == 0) {
      rank0_actual_local = actual_local;
    }
    local_worst = std::max(
        local_worst, max_abs_diff(actual_local, host_reference_local[rank]));
  }

  ErrorMetrics cpu_gemm_error{};
  if (n <= 2048) {
    const auto host_lhs =
        download(0, lhs[0], static_cast<size_t>(a_batch));
    const auto host_rhs =
        download(0, rhs_nt[0], static_cast<size_t>(b_batch));
    std::vector<Bf16> cpu_output(static_cast<size_t>(d_batch));
    for (int row = 0; row < m; ++row) {
      for (int column = 0; column < n; ++column) {
        float accumulator = 0.0f;
        for (int inner = 0; inner < k; ++inner) {
          accumulator +=
              static_cast<float>(host_lhs[static_cast<size_t>(row) * a_row + inner]) *
              static_cast<float>(host_rhs[static_cast<size_t>(column) * b_row + inner]);
        }
        cpu_output[static_cast<size_t>(row) * d_row + column] =
            Bf16(params[0].alpha * accumulator);
      }
    }
    cpu_gemm_error = error_metrics(rank0_actual_local, cpu_output);
  }

  ErrorMetrics worst{};
  for (int destination = 0; destination < world; ++destination) {
    std::vector<Bf16> expected(peer_elements);
    for (int source = 0; source < world; ++source) {
      for (int batch_index = 0; batch_index < batch; ++batch_index) {
        for (int sequence = 0; sequence < seq_local; ++sequence) {
          const int source_row = batch_index * seq_local + sequence;
          const int global_sequence = source * seq_local + sequence;
          for (int segment = 0; segment < 2; ++segment) {
            const int segment_heads = segment == 0 ? q_heads : kv_heads;
            const int heads_per_rank = segment_heads / world;
            const int global_feature_base =
                segment == 0 ? 0 : q_heads * head_dim;
            const int local_feature_base = segment == 0 ? 0 : q_local_width;
            for (int local_head = 0; local_head < heads_per_rank; ++local_head) {
              const int global_head = destination * heads_per_rank + local_head;
              const Bf16* source_feature =
                  host_reference_local[source].data() +
                  static_cast<size_t>(source_row) * d_row + global_feature_base +
                  global_head * head_dim;
              Bf16* destination_feature =
                  expected.data() +
                  (static_cast<size_t>(batch_index) * global_seq + global_sequence) *
                      local_width +
                  local_feature_base + local_head * head_dim;
              std::copy_n(source_feature, head_dim, destination_feature);
            }
          }
        }
      }
    }
    const auto actual =
        download(destination, peer_output[destination], peer_elements);
    const ErrorMetrics errors = error_metrics(actual, expected);
    worst.max_abs = std::max(worst.max_abs, errors.max_abs);
    worst.max_relative = std::max(worst.max_relative, errors.max_relative);
    worst.relative_l2 = std::max(worst.relative_l2, errors.relative_l2);
  }
  std::cout << "FP8 QKV-GQA GEMM->A2A CP=" << world << " B=" << batch
            << " S_local=" << seq_local << " Hq/Hkv=" << q_heads / kv_heads
            << " D=" << head_dim << " padded=" << padded
            << " local_max_abs=" << local_worst
            << " cpu_max_abs=" << cpu_gemm_error.max_abs
            << " cpu_rel_l2=" << cpu_gemm_error.relative_l2
            << " qk_route_max_abs=" << worst.max_abs
            << " qk_route_rel_l2=" << worst.relative_l2 << "\n";
  if (local_worst != 0.0f || worst.max_abs != 0.0f ||
      cpu_gemm_error.relative_l2 > 1.0e-3) {
    throw std::runtime_error("FP8 QKV-GQA GEMM->A2A correctness failed");
  }
}

void smoke_a2a_lhs_gemm(
    int world,
    const std::vector<RankRuntime>& runtimes,
    int batch,
    int seq_local,
    int q_heads,
    int head_dim,
    int hidden,
    bool causal_load_balanced,
    bool cyclic_peer_order) {
  if (q_heads % world != 0) {
    throw std::runtime_error("A2A LHS smoke requires even head sharding");
  }
  const int local_heads = q_heads / world;
  const int global_seq = seq_local * world;
  const int m = batch * seq_local;
  const int k = q_heads * head_dim;
  const int n = hidden;
  const int64_t peer_input_elements =
      static_cast<int64_t>(batch) * global_seq * local_heads * head_dim;
  const int64_t staging_elements = static_cast<int64_t>(m) * k;
  const int64_t weight_elements = static_cast<int64_t>(n) * k;
  const int64_t output_elements = static_cast<int64_t>(m) * n;

  fuse::UlyssesRoute ready_route{};
  ready_route.world_size = world;
  ready_route.batch = batch;
  ready_route.global_seq = global_seq;
  ready_route.seq_local = seq_local;
  ready_route.q_heads = q_heads;
  ready_route.local_heads = local_heads;
  ready_route.head_dim = head_dim;
  ready_route.kind = fuse::RouteKind::kHeadToSequence;
  ready_route.direction = fuse::RouteDirection::kInverse;
  ready_route.causal_load_balanced = causal_load_balanced;
  ready_route.cyclic_peer_order = cyclic_peer_order;
  fuse::GemmProblem problem{m, n, k, 1};
  problem.raster = fuse::GemmRaster::kAlongN;
  const int64_t ready_count =
      fuse::a2a_lhs_gemm_ready_elements(problem, ready_route);

  std::vector<Bf16*> peer_input(world);
  std::vector<Bf16*> staging(world);
  std::vector<Bf16*> weight(world);
  std::vector<Bf16*> output(world);
  std::vector<Bf16*> reference(world);
  std::vector<uint32_t*> ready(world);
  std::vector<std::vector<Bf16>> host_peer_input(world);
  std::vector<fuse::A2AGemmParams> params(world);

  for (int rank = 0; rank < world; ++rank) {
    host_peer_input[rank] = make_values(peer_input_elements, 701 + rank);
    const auto host_weight = make_values(weight_elements, 809 + rank);
    peer_input[rank] = device_alloc<Bf16>(rank, peer_input_elements);
    staging[rank] = device_alloc<Bf16>(rank, staging_elements);
    weight[rank] = device_alloc<Bf16>(rank, weight_elements);
    output[rank] = device_alloc<Bf16>(rank, output_elements);
    reference[rank] = device_alloc<Bf16>(rank, output_elements);
    ready[rank] = device_alloc<uint32_t>(rank, ready_count);
    upload(rank, peer_input[rank], host_peer_input[rank]);
    upload(rank, weight[rank], host_weight);
    CUDA_CHECK(cudaMemsetAsync(
        staging[rank],
        0,
        staging_elements * sizeof(Bf16),
        runtimes[rank].stream));
    CUDA_CHECK(cudaMemsetAsync(
        output[rank],
        0,
        output_elements * sizeof(Bf16),
        runtimes[rank].stream));
    CUDA_CHECK(cudaMemsetAsync(
        reference[rank],
        0,
        output_elements * sizeof(Bf16),
        runtimes[rank].stream));
    CUDA_CHECK(cudaMemsetAsync(
        ready[rank],
        0,
        ready_count * sizeof(uint32_t),
        runtimes[rank].stream));
  }

  for (int rank = 0; rank < world; ++rank) {
    auto& p = params[rank];
    for (int peer = 0; peer < world; ++peer) {
      p.peer_input[peer] = peer_input[peer];
    }
    p.input_staging = staging[rank];
    p.rhs_nt = weight[rank];
    p.output = output[rank];
    p.ready = ready[rank];
    p.gemm = problem;
    p.route = ready_route;
    p.route.rank = rank;
    p.num_comm_ctas = 8;
    p.epoch = 1;
  }

  for (uint32_t epoch = 1; epoch <= g_epoch_iterations; ++epoch) {
    for (int rank = 0; rank < world; ++rank) {
      CUDA_CHECK(cudaSetDevice(rank));
      params[rank].epoch = epoch;
      CUDA_CHECK(fuse::launch_a2a_gemm_cutlass(
          params[rank], runtimes[rank].stream));
    }
    sync_all(runtimes);
  }

  float route_error = 0.0f;
  for (int rank = 0; rank < world; ++rank) {
    std::vector<Bf16> expected(staging_elements);
    const int chunk_rows = seq_local / 2;
    for (int b = 0; b < batch; ++b) {
      for (int local_sequence = 0; local_sequence < seq_local;
           ++local_sequence) {
        int source_sequence = rank * seq_local + local_sequence;
        if (causal_load_balanced) {
          const int chunk = local_sequence < chunk_rows
              ? rank
              : 2 * world - rank - 1;
          const int row_in_chunk = local_sequence < chunk_rows
              ? local_sequence
              : local_sequence - chunk_rows;
          source_sequence = chunk * chunk_rows + row_in_chunk;
        }
        const int destination_row = b * seq_local + local_sequence;
        for (int peer_slot = 0; peer_slot < world; ++peer_slot) {
          const int source_peer = cyclic_peer_order
              ? (rank + peer_slot) % world
              : peer_slot;
          const int64_t src =
              (static_cast<int64_t>(b) * global_seq + source_sequence) *
              local_heads * head_dim;
          const int64_t dst = static_cast<int64_t>(destination_row) * k +
              peer_slot * local_heads * head_dim;
          std::copy_n(
              host_peer_input[source_peer].data() + src,
              local_heads * head_dim,
              expected.data() + dst);
        }
      }
    }
    const auto actual = download(rank, staging[rank], staging_elements);
    route_error = std::max(route_error, max_abs_diff(actual, expected));

    // The pure reference consumes the exact expected inverse-A2A matrix and
    // uses the same CUTLASS GEMM policy as the fused compute role.
    upload(rank, staging[rank], expected);
    auto reference_params = params[rank];
    reference_params.output = reference[rank];
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(fuse::launch_a2a_gemm_cutlass_reference(
        reference_params, runtimes[rank].stream));
  }
  sync_all(runtimes);

  ErrorMetrics worst{};
  for (int rank = 0; rank < world; ++rank) {
    const auto actual = download(rank, output[rank], output_elements);
    const auto expected = download(rank, reference[rank], output_elements);
    const ErrorMetrics errors = error_metrics(actual, expected);
    worst.max_abs = std::max(worst.max_abs, errors.max_abs);
    worst.max_relative = std::max(worst.max_relative, errors.max_relative);
    worst.relative_l2 = std::max(worst.relative_l2, errors.relative_l2);
  }
  std::cout << "HeadToSequence A2A->GEMM CP=" << world
            << " B=" << batch << " S_local=" << seq_local
            << " Hq=" << q_heads << " D=" << head_dim
            << " N=" << hidden
            << " causal_lb=" << causal_load_balanced
            << " cyclic_peer=" << cyclic_peer_order
            << " route_max_abs=" << route_error
            << " output_max_abs=" << worst.max_abs
            << " output_rel_l2=" << worst.relative_l2 << "\n";
  if (route_error != 0.0f || worst.max_abs != 0.0f) {
    throw std::runtime_error("HeadToSequence A2A->GEMM correctness failed");
  }
}

void smoke_a2a_lhs_policy_selection() {
  struct Case {
    fuse::GemmProblem problem;
    int32_t comm_ctas;
    fuse::A2ALhsGemmPolicy expected;
    int32_t expected_waves;
  };
  const Case cases[] = {
      {{1024, 2048, 2048, 1}, 14,
       fuse::A2ALhsGemmPolicy::kM128N160, 1},
      {{2048, 5120, 4096, 1}, 12,
       fuse::A2ALhsGemmPolicy::kM128N256ClusterM2, 3},
      {{4096, 10240, 8192, 1}, 4,
       fuse::A2ALhsGemmPolicy::kM128N320ClusterM2, 8},
  };
  for (const auto& test : cases) {
    const auto selected = fuse::select_a2a_lhs_gemm_policy(
        test.problem, test.comm_ctas, 132);
    if (selected.policy != test.expected ||
        selected.waves != test.expected_waves) {
      throw std::runtime_error(
          "A2A LHS wave-policy selection failed: policy=" +
          std::to_string(static_cast<int32_t>(selected.policy)) +
          " waves=" + std::to_string(selected.waves) +
          " clusters=" + std::to_string(selected.compute_clusters) +
          " n_tiles=" + std::to_string(selected.n_tiles));
    }
  }

  const fuse::GemmProblem long_sequence{16384, 5120, 5120, 1};
  const auto split_frontier = fuse::select_a2a_lhs_gemm_policy(
      long_sequence,
      4,
      132,
      fuse::A2ALhsGemmPolicy::kM128N256ClusterM2);
  if (split_frontier.compute_clusters != 64 ||
      split_frontier.n_tiles != 20 || split_frontier.waves != 20 ||
      split_frontier.frontier_aligned || !split_frontier.full_last_wave) {
    throw std::runtime_error("A2A LHS split-frontier accounting failed");
  }
  const auto aligned = fuse::select_a2a_lhs_gemm_policy(
      long_sequence, 4, 132, fuse::A2ALhsGemmPolicy::kAuto);
  if (aligned.policy != fuse::A2ALhsGemmPolicy::kM128N320ClusterM2 ||
      aligned.compute_clusters != 64 || aligned.n_tiles != 16 ||
      aligned.waves != 16 || !aligned.frontier_aligned ||
      !aligned.full_last_wave) {
    throw std::runtime_error("A2A LHS aligned-frontier selection failed");
  }

  const auto unaligned_full = fuse::select_a2a_lhs_gemm_policy(
      {16384, 7168, 16384, 1}, 4, 132, fuse::A2ALhsGemmPolicy::kAuto);
  if (unaligned_full.policy !=
          fuse::A2ALhsGemmPolicy::kM128N256ClusterM2 ||
      unaligned_full.frontier_aligned || !unaligned_full.full_last_wave) {
    throw std::runtime_error("A2A LHS mature unaligned policy fallback failed");
  }
  const auto partial_wave = fuse::select_a2a_lhs_gemm_policy(
      {2048, 7168, 16384, 1}, 4, 132, fuse::A2ALhsGemmPolicy::kAuto);
  if (partial_wave.policy !=
          fuse::A2ALhsGemmPolicy::kM128N320ClusterM2 ||
      partial_wave.frontier_aligned || partial_wave.full_last_wave ||
      partial_wave.waves != 3) {
    throw std::runtime_error("A2A LHS partial-wave N320 selection failed");
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const bool quick = argc == 2 && std::string(argv[1]) == "--quick";
    if (argc > 1 && !quick) {
      throw std::runtime_error("usage: fuse_smoke [--quick]");
    }
    g_epoch_iterations = quick ? 1 : 8;
    smoke_qkv_gqa_route();
    smoke_a2a_lhs_policy_selection();
    int world = 0;
    CUDA_CHECK(cudaGetDeviceCount(&world));
    world = std::min(world, fuse::kMaxWorldSize);
    if (world != 2 && world != 4 && world != 8) {
      throw std::runtime_error("fuse_smoke requires 2, 4, or 8 visible GPUs");
    }
    enable_peer_access(world);
    auto runtimes = create_runtimes(world);
    const auto cutlass_traits = fuse::cutlass_kernel_traits();
    std::cout << "BF16 CUTLASS traits: BM=" << cutlass_traits.block_m
              << " BN=" << cutlass_traits.block_n
              << " BK=" << cutlass_traits.block_k
              << " threads=" << cutlass_traits.threads
              << " smem=" << cutlass_traits.dynamic_smem_bytes << "\n";
    const auto projection_traits = fuse::projection_cutlass_kernel_traits();
    std::cout << "BF16 projection CUTLASS traits: BM="
              << projection_traits.block_m
              << " BN=" << projection_traits.block_n
              << " BK=" << projection_traits.block_k
              << " threads=" << projection_traits.threads
              << " smem=" << projection_traits.dynamic_smem_bytes << "\n";
    const auto fp8_traits = fuse::fp8_cutlass_kernel_traits();
    std::cout << "FP8 CUTLASS traits: BM=" << fp8_traits.block_m
              << " BN=" << fp8_traits.block_n
              << " BK=" << fp8_traits.block_k
              << " threads=" << fp8_traits.threads
              << " smem=" << fp8_traits.dynamic_smem_bytes << "\n";
    if (quick) {
      smoke_a2a_lhs_gemm(2, runtimes, 1, 150, 8, 64, 512, true, true);
      smoke_qkv_gqa_pack(2, runtimes, 1, 150, 8, 4, 80, true);
      smoke_qkv_gqa_pack(2, runtimes, 1, 128, 8, 4, 64, false, false);
      smoke_fp8_qkv_gqa_pack(2, runtimes, 1, 128, 8, 4, 64, false);
    } else {
      smoke_a2a_lhs_gemm(2, runtimes, 1, 128, 8, 64, 512, false, false);
      smoke_qkv_gqa_pack(2, runtimes, 1, 128, 8, 4, 64, false);
      smoke_fp8_qkv_gqa_pack(2, runtimes, 1, 128, 8, 4, 64, false);
      if (world >= 4) {
        smoke_a2a_lhs_gemm(4, runtimes, 2, 150, 16, 64, 768, true, true);
        smoke_qkv_gqa_pack(4, runtimes, 2, 150, 16, 4, 80, true);
        // Production-size aligned rows exercise the warp-private QKV bulk
        // route instead of the generic vector fallback above.
        smoke_qkv_gqa_pack(4, runtimes, 1, 1024, 64, 8, 128, false, true);
        smoke_qkv_gqa_pack(4, runtimes, 1, 1024, 64, 8, 128, false, false);
      }
      if (world >= 8) {
        smoke_a2a_lhs_gemm(8, runtimes, 1, 128, 64, 128, 1024, true, true);
        smoke_qkv_gqa_pack(8, runtimes, 1, 128, 64, 8, 128, false);
        smoke_qkv_gqa_pack(8, runtimes, 1, 128, 64, 8, 128, false, false);
        smoke_fp8_qkv_gqa_pack(8, runtimes, 1, 128, 64, 8, 128, true);
      }
    }
    std::cout << "fuse_smoke: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "fuse_smoke: FAIL: " << error.what() << "\n";
    return 1;
  }
}
