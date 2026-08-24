#include "fuse/kernels.h"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <sys/resource.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using fuse::Bf16;
using fuse::Fp8E4m3;

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

struct Options {
  std::string mode = "a2a_gemm";
  int m = 4096;
  int n = 128;
  int k = 4096;
  int batch = 1;
  int q_heads = 64;
  int kv_heads = 8;
  int head_dim = 128;
  int comm_ctas = 4;
  int qkv_comm_ctas = 24;
  int pv_comm_ctas = 4;
  int output_comm_ctas = 32;
  int ffn_hidden = 28672;
  int layers = 80;
  int warmup = 5;
  int iterations = 20;
  fuse::GemmRaster raster = fuse::GemmRaster::kHeuristic;
  int swizzle = 1;
  bool role_telemetry = false;
  bool defer_v_a2a = false;
  fuse::A2ALhsGemmPolicy lhs_policy = fuse::A2ALhsGemmPolicy::kAuto;
  std::string json_out;
  std::string trace_out;
};

int parse_int(const std::string& value, const char* name) {
  const int parsed = std::stoi(value);
  if (parsed <= 0) {
    throw std::runtime_error(std::string(name) + " must be positive");
  }
  return parsed;
}

int parse_nonnegative_int(const std::string& value, const char* name) {
  const int parsed = std::stoi(value);
  if (parsed < 0) {
    throw std::runtime_error(std::string(name) + " must be nonnegative");
  }
  return parsed;
}

const char* lhs_policy_name(fuse::A2ALhsGemmPolicy policy) {
  switch (policy) {
    case fuse::A2ALhsGemmPolicy::kOneWaveAuto:
      return "one-wave";
    case fuse::A2ALhsGemmPolicy::kM64N128:
      return "m64n128";
    case fuse::A2ALhsGemmPolicy::kM128N128:
      return "m128n128";
    case fuse::A2ALhsGemmPolicy::kM128N160:
      return "m128n160";
    case fuse::A2ALhsGemmPolicy::kM128N256ClusterM2:
      return "m128n256_cluster_m2";
    default:
      return "auto";
  }
}

fuse::A2ALhsGemmPolicy parse_lhs_policy(const std::string& value) {
  if (value == "auto") return fuse::A2ALhsGemmPolicy::kAuto;
  if (value == "one-wave") return fuse::A2ALhsGemmPolicy::kOneWaveAuto;
  if (value == "m64n128") return fuse::A2ALhsGemmPolicy::kM64N128;
  if (value == "m128n128") return fuse::A2ALhsGemmPolicy::kM128N128;
  if (value == "m128n160") return fuse::A2ALhsGemmPolicy::kM128N160;
  if (value == "m128n256c2") {
    return fuse::A2ALhsGemmPolicy::kM128N256ClusterM2;
  }
  throw std::runtime_error(
      "--lhs-policy must be auto, one-wave, m64n128, m128n128, m128n160, "
      "or m128n256c2");
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    auto take = [&](const char* name) {
      if (++i == argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      return std::string(argv[i]);
    };
    if (argument == "--mode") {
      options.mode = take("--mode");
    } else if (argument == "--m") {
      options.m = parse_int(take("--m"), "--m");
    } else if (argument == "--n") {
      options.n = parse_int(take("--n"), "--n");
    } else if (argument == "--k") {
      options.k = parse_int(take("--k"), "--k");
    } else if (argument == "--batch") {
      options.batch = parse_int(take("--batch"), "--batch");
    } else if (argument == "--q-heads") {
      options.q_heads = parse_int(take("--q-heads"), "--q-heads");
    } else if (argument == "--kv-heads") {
      options.kv_heads = parse_int(take("--kv-heads"), "--kv-heads");
    } else if (argument == "--head-dim") {
      options.head_dim = parse_int(take("--head-dim"), "--head-dim");
    } else if (argument == "--comm-ctas") {
      options.comm_ctas =
          parse_nonnegative_int(take("--comm-ctas"), "--comm-ctas");
    } else if (argument == "--lhs-policy") {
      options.lhs_policy = parse_lhs_policy(take("--lhs-policy"));
    } else if (argument == "--qkv-comm-ctas") {
      options.qkv_comm_ctas =
          parse_int(take("--qkv-comm-ctas"), "--qkv-comm-ctas");
    } else if (argument == "--pv-comm-ctas") {
      options.pv_comm_ctas =
          parse_int(take("--pv-comm-ctas"), "--pv-comm-ctas");
    } else if (argument == "--output-comm-ctas") {
      options.output_comm_ctas =
          parse_int(take("--output-comm-ctas"), "--output-comm-ctas");
    } else if (argument == "--ffn-hidden") {
      options.ffn_hidden = parse_int(take("--ffn-hidden"), "--ffn-hidden");
    } else if (argument == "--layers") {
      options.layers = parse_int(take("--layers"), "--layers");
    } else if (argument == "--warmup") {
      options.warmup = parse_int(take("--warmup"), "--warmup");
    } else if (argument == "--iterations") {
      options.iterations = parse_int(take("--iterations"), "--iterations");
    } else if (argument == "--raster") {
      const std::string value = take("--raster");
      if (value == "auto") {
        options.raster = fuse::GemmRaster::kHeuristic;
      } else if (value == "m") {
        options.raster = fuse::GemmRaster::kAlongM;
      } else if (value == "n") {
        options.raster = fuse::GemmRaster::kAlongN;
      } else {
        throw std::runtime_error("--raster must be auto, m, or n");
      }
    } else if (argument == "--swizzle") {
      options.swizzle = parse_int(take("--swizzle"), "--swizzle");
      if (options.swizzle != 1 && options.swizzle != 2 &&
          options.swizzle != 4 && options.swizzle != 8) {
        throw std::runtime_error("--swizzle must be 1, 2, 4, or 8");
      }
    } else if (argument == "--json-out") {
      options.json_out = take("--json-out");
    } else if (argument == "--trace-out") {
      options.trace_out = take("--trace-out");
    } else if (argument == "--role-telemetry") {
      options.role_telemetry = true;
    } else if (argument == "--defer-v-a2a") {
      options.defer_v_a2a = true;
    } else {
      throw std::runtime_error("unknown argument: " + argument);
    }
  }
  return options;
}

struct Runtime {
  cudaStream_t stream = nullptr;
  cublasHandle_t blas = nullptr;
  cublasLtHandle_t blas_lt = nullptr;
  void* blas_lt_workspace = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
};

template <class T>
T* allocate(int device, int64_t elements) {
  CUDA_CHECK(cudaSetDevice(device));
  T* pointer = nullptr;
  CUDA_CHECK(cudaMalloc(&pointer, static_cast<size_t>(elements) * sizeof(T)));
  return pointer;
}

void synchronize(const std::vector<Runtime>& runtime);

struct RoleSpan {
  uint64_t first = std::numeric_limits<uint64_t>::max();
  uint64_t last = 0;
  uint64_t elapsed_sum = 0;
  int32_t count = 0;

  void add(const fuse::A2AGemmCtaTimeline& sample) {
    if (sample.start == 0 || sample.end < sample.start) {
      return;
    }
    first = std::min(first, sample.start);
    last = std::max(last, sample.end);
    elapsed_sum += sample.end - sample.start;
    ++count;
  }

  double span_us() const {
    return count == 0 ? 0.0 : static_cast<double>(last - first) / 1000.0;
  }

  double average_us() const {
    return count == 0
        ? 0.0
        : static_cast<double>(elapsed_sum) / count / 1000.0;
  }
};

void report_a2a_role_telemetry(
    int world,
    const std::vector<Runtime>& runtime,
    std::vector<fuse::A2AGemmParams>& params,
    const std::vector<uint32_t*>& ready,
    int64_t ready_count) {
  std::vector<int32_t> capacities(world);
  std::vector<fuse::A2AGemmCtaTimeline*> device_timeline(world);
  for (int rank = 0; rank < world; ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaDeviceGetAttribute(
        &capacities[rank], cudaDevAttrMultiProcessorCount, rank));
    device_timeline[rank] = allocate<fuse::A2AGemmCtaTimeline>(
        rank, capacities[rank]);
    CUDA_CHECK(cudaMemsetAsync(
        device_timeline[rank],
        0,
        static_cast<size_t>(capacities[rank]) *
            sizeof(fuse::A2AGemmCtaTimeline),
        runtime[rank].stream));
    CUDA_CHECK(cudaMemsetAsync(
        ready[rank],
        0,
        static_cast<size_t>(ready_count) * sizeof(uint32_t),
        runtime[rank].stream));
    params[rank].epoch = 1;
  }
  for (int rank = 0; rank < world; ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(fuse::launch_a2a_gemm_cutlass_role_telemetry(
        params[rank],
        device_timeline[rank],
        capacities[rank],
        nullptr,
        0,
        runtime[rank].stream));
  }
  synchronize(runtime);

  fuse::A2AGemmRoleResources resources{};
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(fuse::query_a2a_gemm_role_resources(&resources));
  const int64_t allocated_registers =
      static_cast<int64_t>(resources.threads_per_cta) *
      resources.registers_per_thread;
  std::cout << "\nA2A->GEMM persistent-role resources (production kernel)\n"
            << "role       CTAs  active/allocated warps  registers/CTA  "
               "dynamic SMEM allocated  role working SMEM\n";
  std::cout << "comm       " << std::setw(4) << params[0].num_comm_ctas
            << "  " << resources.comm_active_warps << "/"
            << resources.threads_per_cta / 32 << "                    "
            << allocated_registers << "          "
            << std::fixed << std::setprecision(1)
            << resources.dynamic_smem_bytes / 1024.0 << " KiB"
            << "              " << resources.comm_working_smem_bytes / 1024.0
            << " KiB\n";
  std::cout << "compute    " << std::setw(4)
            << capacities[0] - params[0].num_comm_ctas
            << "  " << resources.compute_active_warps << "/"
            << resources.threads_per_cta / 32 << "                   "
            << allocated_registers << "          "
            << resources.dynamic_smem_bytes / 1024.0 << " KiB"
            << "              " << resources.dynamic_smem_bytes / 1024.0
            << " KiB\n";
  std::cout << "cluster=" << resources.cluster_ctas
            << " CTAs, production regs/thread="
            << resources.registers_per_thread
            << ", telemetry regs/thread="
            << resources.telemetry_registers_per_thread
            << ", static SMEM=" << resources.static_smem_bytes
            << " B\n";

  std::cout << "\nOne-shot role timeline (%globaltimer, diagnostic kernel)\n"
            << "rank  comm_CTAs compute_CTAs comm_span_us compute_span_us "
               "overlap_us comm_overlapped compute_tail_us\n";
  for (int rank = 0; rank < world; ++rank) {
    std::vector<fuse::A2AGemmCtaTimeline> host(capacities[rank]);
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaMemcpy(
        host.data(),
        device_timeline[rank],
        host.size() * sizeof(host[0]),
        cudaMemcpyDeviceToHost));
    RoleSpan comm;
    RoleSpan compute;
    for (int32_t cta = 0; cta < capacities[rank]; ++cta) {
      if (cta < params[rank].num_comm_ctas) {
        comm.add(host[cta]);
      } else {
        compute.add(host[cta]);
      }
    }
    const uint64_t overlap_begin = std::max(comm.first, compute.first);
    const uint64_t overlap_end = std::min(comm.last, compute.last);
    const double overlap_us =
        comm.count && compute.count && overlap_end > overlap_begin
        ? static_cast<double>(overlap_end - overlap_begin) / 1000.0
        : 0.0;
    const double comm_overlap =
        comm.span_us() > 0.0 ? 100.0 * overlap_us / comm.span_us() : 0.0;
    const double compute_tail =
        comm.count && compute.count && compute.last > comm.last
        ? static_cast<double>(compute.last - comm.last) / 1000.0
        : 0.0;
    std::cout << std::setw(4) << rank << "  " << std::setw(9) << comm.count
              << " " << std::setw(12) << compute.count << " "
              << std::setw(12) << std::setprecision(2) << comm.span_us()
              << " " << std::setw(15) << compute.span_us() << " "
              << std::setw(10) << overlap_us << " " << std::setw(13)
              << std::setprecision(1) << comm_overlap << "% "
              << std::setw(15) << std::setprecision(2) << compute_tail
              << "\n";
    std::cout << "      avg CTA duration: comm=" << comm.average_us()
              << " us, compute=" << compute.average_us() << " us\n";
  }
  std::cout << "Static allocation is identical for both roles; complementarity "
               "is across SMs (TMA/NVLink versus WGMMA), not pooled registers "
               "or SMEM inside one SM.\n";

  for (int rank = 0; rank < world; ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaFree(device_timeline[rank]));
  }
}

void report_a2a_lhs_role_trace(
    int world,
    const std::vector<Runtime>& runtime,
    std::vector<fuse::A2AGemmParams>& params,
    const std::vector<uint32_t*>& ready,
    int64_t ready_count,
    const std::string& trace_path) {
  std::vector<int32_t> capacities(world);
  std::vector<fuse::A2AGemmCtaTimeline*> device_timeline(world);
  std::vector<std::vector<fuse::A2AGemmCtaTimeline>> timeline(world);
  std::vector<int32_t> peer_capacities(world);
  std::vector<fuse::A2AGemmPeerTimeline*> device_peer_timeline(world);
  std::vector<std::vector<fuse::A2AGemmPeerTimeline>> peer_timeline(world);
  for (int rank = 0; rank < world; ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaDeviceGetAttribute(
        &capacities[rank], cudaDevAttrMultiProcessorCount, rank));
    device_timeline[rank] = allocate<fuse::A2AGemmCtaTimeline>(
        rank, capacities[rank]);
    const int32_t compute_ctas = capacities[rank] - params[rank].num_comm_ctas;
    const int32_t m128_tiles = (params[rank].gemm.m + 127) / 128;
    const int32_t n128_tiles = (params[rank].gemm.n + 127) / 128;
    const bool use_m64 = params[rank].gemm.n < 4096 &&
        static_cast<int64_t>(m128_tiles) * n128_tiles * params[rank].gemm.l <
            compute_ctas;
    const int32_t tile_m = use_m64 ? 64 : 128;
    const int32_t m_tiles = (params[rank].gemm.m + tile_m - 1) / tile_m;
    const int32_t n_tiles = params[rank].gemm.n >= 4096
        ? (params[rank].gemm.n + 255) / 256
        : n128_tiles;
    peer_capacities[rank] = std::max(
        m_tiles * n_tiles * params[rank].gemm.l,
        m_tiles * params[rank].route.world_size);
    device_peer_timeline[rank] = allocate<fuse::A2AGemmPeerTimeline>(
        rank, peer_capacities[rank]);
    CUDA_CHECK(cudaMemsetAsync(
        device_timeline[rank],
        0,
        static_cast<size_t>(capacities[rank]) *
            sizeof(fuse::A2AGemmCtaTimeline),
        runtime[rank].stream));
    CUDA_CHECK(cudaMemsetAsync(
        device_peer_timeline[rank],
        0,
        static_cast<size_t>(peer_capacities[rank]) *
            sizeof(fuse::A2AGemmPeerTimeline),
        runtime[rank].stream));
    CUDA_CHECK(cudaMemsetAsync(
        ready[rank],
        0,
        static_cast<size_t>(ready_count) * sizeof(uint32_t),
        runtime[rank].stream));
    params[rank].epoch = 1;
  }
  for (int rank = 0; rank < world; ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(fuse::launch_a2a_gemm_cutlass_role_telemetry(
        params[rank],
        device_timeline[rank],
        capacities[rank],
        device_peer_timeline[rank],
        peer_capacities[rank],
        runtime[rank].stream));
  }
  synchronize(runtime);

  std::cout << "\nO-proj CTA timeline (%globaltimer, diagnostic kernel)\n"
            << "rank  comm_span_us  gemm_span_us  overlap_us  "
               "compute_wait_us  fused_span_us\n";
  for (int rank = 0; rank < world; ++rank) {
    timeline[rank].resize(capacities[rank]);
    peer_timeline[rank].resize(peer_capacities[rank]);
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaMemcpy(
        timeline[rank].data(),
        device_timeline[rank],
        timeline[rank].size() * sizeof(timeline[rank][0]),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        peer_timeline[rank].data(),
        device_peer_timeline[rank],
        peer_timeline[rank].size() * sizeof(peer_timeline[rank][0]),
        cudaMemcpyDeviceToHost));
    int32_t release_records = 0;
    int32_t valid_records = 0;
    int32_t acquire_records = 0;
    for (const auto& event : peer_timeline[rank]) {
      release_records += event.release != 0;
      valid_records += event.valid != 0;
      for (uint64_t value : event.acquire) {
        acquire_records += value != 0;
      }
    }
    std::cout << "      peer timing records: release=" << release_records
              << " gemm_tiles=" << valid_records
              << " acquire=" << acquire_records << "\n";

    uint64_t first = UINT64_MAX;
    uint64_t last = 0;
    uint64_t comm_first = UINT64_MAX;
    uint64_t comm_last = 0;
    uint64_t gemm_first = UINT64_MAX;
    uint64_t gemm_last = 0;
    double wait_sum_ns = 0.0;
    int32_t wait_count = 0;
    for (int32_t cta = 0; cta < capacities[rank]; ++cta) {
      const auto& event = timeline[rank][cta];
      if (event.start == 0 || event.end <= event.start) {
        continue;
      }
      first = std::min(first, event.start);
      last = std::max(last, event.end);
      if (cta < params[rank].num_comm_ctas && event.end > event.start) {
        comm_first = std::min(comm_first, event.start);
        comm_last = std::max(comm_last, event.end);
      }
      if (event.active_start > 0 && event.end > event.active_start) {
        gemm_first = std::min(gemm_first, event.active_start);
        gemm_last = std::max(gemm_last, event.end);
        wait_sum_ns += static_cast<double>(event.active_start - event.start);
        ++wait_count;
      }
    }
    const uint64_t overlap_begin = std::max(comm_first, gemm_first);
    const uint64_t overlap_end = std::min(comm_last, gemm_last);
    const double comm_us = comm_last > comm_first
        ? static_cast<double>(comm_last - comm_first) / 1000.0
        : 0.0;
    const double gemm_us = gemm_last > gemm_first
        ? static_cast<double>(gemm_last - gemm_first) / 1000.0
        : 0.0;
    const double overlap_us = overlap_end > overlap_begin
        ? static_cast<double>(overlap_end - overlap_begin) / 1000.0
        : 0.0;
    const double wait_us = wait_count > 0
        ? wait_sum_ns / wait_count / 1000.0
        : 0.0;
    const double fused_us = last > first
        ? static_cast<double>(last - first) / 1000.0
        : 0.0;
    std::cout << std::setw(4) << rank << "  " << std::setw(12)
              << std::fixed << std::setprecision(2) << comm_us << "  "
              << std::setw(12) << gemm_us << "  " << std::setw(10)
              << overlap_us << "  " << std::setw(15) << wait_us << "  "
              << std::setw(13) << fused_us << "\n";
  }

  std::ofstream trace(trace_path);
  if (!trace) {
    throw std::runtime_error("cannot open trace output: " + trace_path);
  }
  trace << "{\n  \"displayTimeUnit\": \"ns\",\n  \"traceEvents\": [\n";
  bool first_event = true;
  auto emit = [&](int rank, int tid, const std::string& name,
                  const std::string& category,
                  uint64_t begin, uint64_t end, uint64_t origin) {
    if (begin == 0 || end <= begin) {
      return;
    }
    trace << (first_event ? "" : ",\n");
    first_event = false;
    trace << "    {\"name\":\"" << name << "\",\"cat\":\""
          << category << "\",\"ph\":\"X\",\"pid\":" << rank
          << ",\"tid\":" << tid << ",\"ts\":"
          << std::fixed << std::setprecision(3)
          << static_cast<double>(begin - origin) / 1000.0
          << ",\"dur\":" << static_cast<double>(end - begin) / 1000.0
          << "}";
  };
  for (int rank = 0; rank < world; ++rank) {
    uint64_t origin = UINT64_MAX;
    for (const auto& event : timeline[rank]) {
      if (event.start != 0) {
        origin = std::min(origin, event.start);
      }
    }
    if (origin == UINT64_MAX) {
      continue;
    }
    for (int32_t cta = 0; cta < capacities[rank]; ++cta) {
      const auto& event = timeline[rank][cta];
      if (cta < params[rank].num_comm_ctas) {
        emit(rank, cta, "remote A2A", "communication",
             event.start, event.end, origin);
      } else {
        emit(rank, cta, "ready wait", "dependency",
             event.start, event.active_start, origin);
        emit(rank, cta, "GEMM", "compute",
             event.active_start, event.end, origin);
      }
    }
    const int32_t compute_ctas = capacities[rank] - params[rank].num_comm_ctas;
    const int32_t m128_tiles = (params[rank].gemm.m + 127) / 128;
    const int32_t n128_tiles = (params[rank].gemm.n + 127) / 128;
    const bool use_m64 = params[rank].gemm.n < 4096 &&
        static_cast<int64_t>(m128_tiles) * n128_tiles * params[rank].gemm.l <
            compute_ctas;
    const int32_t ready_block_m = use_m64 ? 64 : 128;
    const int32_t ready_m_tiles =
        (params[rank].gemm.m + ready_block_m - 1) / ready_block_m;
    const int32_t peer_count = params[rank].route.world_size;
    for (int32_t tile = 0; tile < peer_capacities[rank]; ++tile) {
      const auto& event = peer_timeline[rank][tile];
      if (!event.valid) {
        continue;
      }
      const int32_t ready_m = event.m_tile;
      if (ready_m < 0 || ready_m >= ready_m_tiles) {
        continue;
      }
      for (int32_t peer_slot = 0; peer_slot < peer_count; ++peer_slot) {
        const int32_t release_index = ready_m * peer_count + peer_slot;
        if (release_index >= peer_capacities[rank]) {
          continue;
        }
        const uint64_t release = peer_timeline[rank][release_index].release;
        const uint64_t acquire = event.acquire[peer_slot];
        if (release == 0 || acquire <= release) {
          continue;
        }
        const int32_t source_rank = params[rank].route.cyclic_peer_order
            ? (rank + peer_slot) % peer_count
            : peer_slot;
        const std::string name =
            "m_tile=" + std::to_string(event.m_tile) +
            " n_tile=" + std::to_string(event.n_tile) +
            " peer_slot=" + std::to_string(peer_slot) +
            " source_rank=" + std::to_string(source_rank) +
            " release->acquire";
        emit(
            rank,
            1000 + tile * peer_count + peer_slot,
            name,
            "peer_ready",
            release,
            acquire,
            origin);
      }
    }
  }
  trace << "\n  ]\n}\n";
  trace.close();
  std::cout << "Perfetto trace: " << trace_path << "\n";

  for (int rank = 0; rank < world; ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaFree(device_timeline[rank]));
    CUDA_CHECK(cudaFree(device_peer_timeline[rank]));
  }
}

__global__ void fill_kernel(Bf16* data, int64_t elements, int seed) {
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < elements;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int value = static_cast<int>((index * 17 + seed * 13) % 19) - 9;
    data[index] = Bf16(static_cast<float>(value) / 32.0f);
  }
}

__global__ void fill_weight_kernel(Bf16* data, int64_t elements, int seed) {
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < elements;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int value = static_cast<int>((index * 17 + seed * 13) % 19) - 9;
    data[index] = Bf16(static_cast<float>(value) / 256.0f);
  }
}

__global__ void fill_norm_weight_kernel(Bf16* data, int64_t elements, int seed) {
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < elements;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int value = static_cast<int>((index * 17 + seed * 13) % 19) - 9;
    data[index] = Bf16(1.0f + static_cast<float>(value) / 256.0f);
  }
}

__global__ void fill_fp8_kernel(Fp8E4m3* data, int64_t elements, int seed) {
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < elements;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int value = static_cast<int>((index * 17 + seed * 13) % 17) - 8;
    data[index] = Fp8E4m3(static_cast<float>(value) / 4.0f);
  }
}

__global__ void count_u16_mismatch_kernel(
    const uint16_t* lhs,
    const uint16_t* rhs,
    int64_t elements,
    unsigned long long* mismatches) {
  unsigned long long local = 0;
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    local += lhs[index] != rhs[index];
  }
  if (local) {
    atomicAdd(mismatches, local);
  }
}

__global__ void rmsnorm_kernel(
    const Bf16* input,
    const Bf16* weight,
    Bf16* output,
    int rows,
    int hidden,
    float epsilon) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  float sum = 0.0f;
  for (int column = threadIdx.x; column < hidden; column += blockDim.x) {
    const float value = static_cast<float>(input[static_cast<int64_t>(row) * hidden + column]);
    sum += value * value;
  }
  __shared__ float reduction[256];
  reduction[threadIdx.x] = sum;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      reduction[threadIdx.x] += reduction[threadIdx.x + offset];
    }
    __syncthreads();
  }
  const float inverse_rms = rsqrtf(reduction[0] / hidden + epsilon);
  for (int column = threadIdx.x; column < hidden; column += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * hidden + column;
    output[index] = Bf16(
        static_cast<float>(input[index]) * inverse_rms *
        static_cast<float>(weight[column]));
  }
}

__global__ void unpack_qk_kernel(
    const Bf16* packed,
    Bf16* query,
    Bf16* key,
    int sequence,
    int local_q_heads,
    int head_dim) {
  const int64_t q_elements =
      static_cast<int64_t>(sequence) * local_q_heads * head_dim;
  const int64_t total = q_elements + static_cast<int64_t>(sequence) * head_dim;
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    if (index < q_elements) {
      const int channel = index % head_dim;
      const int64_t logical = index / head_dim;
      const int row = logical % sequence;
      const int head = logical / sequence;
      query[index] = packed[
          static_cast<int64_t>(row) * (local_q_heads + 1) * head_dim +
          head * head_dim + channel];
    } else {
      const int64_t key_index = index - q_elements;
      const int channel = key_index % head_dim;
      const int row = key_index / head_dim;
      key[key_index] = packed[
          static_cast<int64_t>(row) * (local_q_heads + 1) * head_dim +
          local_q_heads * head_dim + channel];
    }
  }
}

__global__ void causal_softmax_kernel(
    Bf16* scores,
    int sequence,
    int rows,
    float scale) {
  const int logical_row = static_cast<int>(blockIdx.x);
  if (logical_row >= rows) {
    return;
  }
  const int query_row = logical_row % sequence;
  const int64_t base = static_cast<int64_t>(logical_row) * sequence;
  float maximum = -3.402823466e+38F;
  for (int column = threadIdx.x; column < sequence; column += blockDim.x) {
    if (column <= query_row) {
      maximum = fmaxf(maximum, static_cast<float>(scores[base + column]) * scale);
    }
  }
  __shared__ float reduction[256];
  reduction[threadIdx.x] = maximum;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      reduction[threadIdx.x] =
          fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + offset]);
    }
    __syncthreads();
  }
  maximum = reduction[0];
  float sum = 0.0f;
  for (int column = threadIdx.x; column < sequence; column += blockDim.x) {
    if (column <= query_row) {
      sum += __expf(static_cast<float>(scores[base + column]) * scale - maximum);
    }
  }
  reduction[threadIdx.x] = sum;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      reduction[threadIdx.x] += reduction[threadIdx.x + offset];
    }
    __syncthreads();
  }
  const float inverse_sum = 1.0f / reduction[0];
  for (int column = threadIdx.x; column < sequence; column += blockDim.x) {
    const float probability = column <= query_row
        ? __expf(static_cast<float>(scores[base + column]) * scale - maximum) * inverse_sum
        : 0.0f;
    scores[base + column] = Bf16(probability);
  }
}

__global__ void swiglu_kernel(
    const Bf16* gate_up,
    Bf16* output,
    int64_t elements,
    int ffn_hidden) {
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t row = index / ffn_hidden;
    const int column = index - row * ffn_hidden;
    const int64_t gate_base = row * 2 * ffn_hidden;
    const float gate = static_cast<float>(gate_up[gate_base + column]);
    const float up = static_cast<float>(gate_up[gate_base + ffn_hidden + column]);
    output[index] = Bf16((gate / (1.0f + __expf(-gate))) * up);
  }
}

__global__ void residual_add_kernel(
    const Bf16* lhs,
    const Bf16* rhs,
    Bf16* output,
    int64_t elements) {
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    output[index] = Bf16(static_cast<float>(lhs[index]) + static_cast<float>(rhs[index]));
  }
}

__global__ void count_nonfinite_kernel(
    const Bf16* data,
    int64_t elements,
    unsigned long long* count) {
  unsigned long long local = 0;
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    local += !isfinite(static_cast<float>(data[index]));
  }
  if (local != 0) {
    atomicAdd(count, local);
  }
}

void fill(int device, cudaStream_t stream, Bf16* pointer, int64_t elements, int seed) {
  CUDA_CHECK(cudaSetDevice(device));
  const int blocks = static_cast<int>(std::min<int64_t>((elements + 255) / 256, 4096));
  fill_kernel<<<blocks, 256, 0, stream>>>(pointer, elements, seed);
  CUDA_CHECK(cudaGetLastError());
}

void fill_weight(
    int device,
    cudaStream_t stream,
    Bf16* pointer,
    int64_t elements,
    int seed) {
  CUDA_CHECK(cudaSetDevice(device));
  const int blocks = static_cast<int>(
      std::min<int64_t>((elements + 255) / 256, 4096));
  fill_weight_kernel<<<blocks, 256, 0, stream>>>(pointer, elements, seed);
  CUDA_CHECK(cudaGetLastError());
}

void fill_norm_weight(
    int device,
    cudaStream_t stream,
    Bf16* pointer,
    int64_t elements,
    int seed) {
  CUDA_CHECK(cudaSetDevice(device));
  const int blocks = static_cast<int>(
      std::min<int64_t>((elements + 255) / 256, 4096));
  fill_norm_weight_kernel<<<blocks, 256, 0, stream>>>(pointer, elements, seed);
  CUDA_CHECK(cudaGetLastError());
}

void fill_fp8(
    int device,
    cudaStream_t stream,
    Fp8E4m3* pointer,
    int64_t elements,
    int seed) {
  CUDA_CHECK(cudaSetDevice(device));
  const int blocks = static_cast<int>(std::min<int64_t>((elements + 255) / 256, 4096));
  fill_fp8_kernel<<<blocks, 256, 0, stream>>>(pointer, elements, seed);
  CUDA_CHECK(cudaGetLastError());
}

constexpr size_t kCublasLtWorkspaceBytes = 64ull << 20;

std::vector<Runtime> initialize_runtime(int world) {
  std::vector<Runtime> result(world);
  for (int rank = 0; rank < world; ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    int least_priority = 0;
    int greatest_priority = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
    CUDA_CHECK(cudaStreamCreateWithPriority(
        &result[rank].stream, cudaStreamNonBlocking, greatest_priority));
    CUBLAS_CHECK(cublasCreate(&result[rank].blas));
    CUBLAS_CHECK(cublasSetStream(result[rank].blas, result[rank].stream));
    CUBLAS_CHECK(cublasSetMathMode(result[rank].blas, CUBLAS_TENSOR_OP_MATH));
    CUBLAS_CHECK(cublasLtCreate(&result[rank].blas_lt));
    CUDA_CHECK(cudaMalloc(
        &result[rank].blas_lt_workspace, kCublasLtWorkspaceBytes));
    CUDA_CHECK(cudaEventCreate(&result[rank].start));
    CUDA_CHECK(cudaEventCreate(&result[rank].stop));
  }
  return result;
}

void enable_p2p(int world) {
  for (int device = 0; device < world; ++device) {
    CUDA_CHECK(cudaSetDevice(device));
    for (int peer = 0; peer < world; ++peer) {
      if (peer == device) {
        continue;
      }
      int supported = 0;
      CUDA_CHECK(cudaDeviceCanAccessPeer(&supported, device, peer));
      if (!supported) {
        throw std::runtime_error("all-pairs P2P is required");
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

void synchronize(const std::vector<Runtime>& runtime) {
  for (int rank = 0; rank < static_cast<int>(runtime.size()); ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaStreamSynchronize(runtime[rank].stream));
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
    int64_t b_batch_stride = -1) {
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
      k,
      b_batch_stride < 0 ? static_cast<long long>(n) * k : b_batch_stride,
      a,
      CUDA_R_16BF,
      k,
      static_cast<long long>(m) * k,
      &beta,
      d,
      CUDA_R_16BF,
      n,
      static_cast<long long>(m) * n,
      batches,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

struct CublasLtBf16Plan {
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t a = nullptr;
  cublasLtMatrixLayout_t b = nullptr;
  cublasLtMatrixLayout_t c = nullptr;
  cublasLtMatrixLayout_t d = nullptr;
  cublasLtMatmulAlgo_t algorithm{};
  size_t algorithm_workspace = 0;
};

int cublaslt_algo_attribute(
    const cublasLtMatmulAlgo_t& algorithm,
    cublasLtMatmulAlgoConfigAttributes_t attribute) {
  int value = -1;
  size_t written = 0;
  const cublasStatus_t status = cublasLtMatmulAlgoConfigGetAttribute(
      &algorithm, attribute, &value, sizeof(value), &written);
  return status == CUBLAS_STATUS_SUCCESS ? value : -1;
}

int cublaslt_algo_u16_attribute(
    const cublasLtMatmulAlgo_t& algorithm,
    cublasLtMatmulAlgoConfigAttributes_t attribute) {
  uint16_t value = 0;
  size_t written = 0;
  const cublasStatus_t status = cublasLtMatmulAlgoConfigGetAttribute(
      &algorithm, attribute, &value, sizeof(value), &written);
  return status == CUBLAS_STATUS_SUCCESS ? static_cast<int>(value) : -1;
}

uint64_t cublaslt_algo_numerical_flags(
    const cublasLtMatmulAlgo_t& algorithm) {
  uint64_t value = 0;
  size_t written = 0;
  const cublasStatus_t status = cublasLtMatmulAlgoCapGetAttribute(
      &algorithm,
      CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,
      &value,
      sizeof(value),
      &written);
  return status == CUBLAS_STATUS_SUCCESS ? value : 0;
}

cublasStatus_t cublaslt_nt_unchecked(
    const Runtime& runtime,
    const CublasLtBf16Plan& plan,
    const Bf16* a,
    const Bf16* b_nt,
    Bf16* d,
    const cublasLtMatmulAlgo_t& algorithm) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  return cublasLtMatmul(
      runtime.blas_lt,
      plan.operation,
      &alpha,
      a,
      plan.a,
      b_nt,
      plan.b,
      &beta,
      d,
      plan.c,
      d,
      plan.d,
      &algorithm,
      runtime.blas_lt_workspace,
      kCublasLtWorkspaceBytes,
      runtime.stream);
}

void cublaslt_nt(
    const Runtime& runtime,
    const CublasLtBf16Plan& plan,
    const Bf16* a,
    const Bf16* b_nt,
    Bf16* d) {
  CUBLAS_CHECK(cublaslt_nt_unchecked(
      runtime, plan, a, b_nt, d, plan.algorithm));
}

float time_cublaslt_candidate(
    const Runtime& runtime,
    const CublasLtBf16Plan& plan,
    const cublasLtMatmulAlgo_t& algorithm,
    const Bf16* a,
    const Bf16* b_nt,
    Bf16* d) {
  constexpr int warmup = 5;
  constexpr int iterations = 30;
  for (int iteration = 0; iteration < warmup; ++iteration) {
    if (cublaslt_nt_unchecked(runtime, plan, a, b_nt, d, algorithm) !=
        CUBLAS_STATUS_SUCCESS) {
      return std::numeric_limits<float>::infinity();
    }
  }
  if (cudaStreamSynchronize(runtime.stream) != cudaSuccess) {
    cudaGetLastError();
    return std::numeric_limits<float>::infinity();
  }
  CUDA_CHECK(cudaEventRecord(runtime.start, runtime.stream));
  for (int iteration = 0; iteration < iterations; ++iteration) {
    if (cublaslt_nt_unchecked(runtime, plan, a, b_nt, d, algorithm) !=
        CUBLAS_STATUS_SUCCESS) {
      return std::numeric_limits<float>::infinity();
    }
  }
  CUDA_CHECK(cudaEventRecord(runtime.stop, runtime.stream));
  if (cudaEventSynchronize(runtime.stop) != cudaSuccess) {
    cudaGetLastError();
    return std::numeric_limits<float>::infinity();
  }
  float elapsed = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed, runtime.start, runtime.stop));
  return elapsed / iterations;
}

CublasLtBf16Plan autotune_cublaslt_bf16(
    const Runtime& runtime,
    int m,
    int n,
    int k,
    int batches,
    const Bf16* a,
    const Bf16* b_nt,
    Bf16* d,
    int64_t b_batch_stride = -1) {
  CUDA_CHECK(cudaSetDevice(0));
  CublasLtBf16Plan plan;
  CUBLAS_CHECK(cublasLtMatmulDescCreate(
      &plan.operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  const cublasOperation_t trans_a = CUBLAS_OP_N;
  const cublasOperation_t trans_b = CUBLAS_OP_T;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      plan.operation, CUBLASLT_MATMUL_DESC_TRANSA,
      &trans_a, sizeof(trans_a)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      plan.operation, CUBLASLT_MATMUL_DESC_TRANSB,
      &trans_b, sizeof(trans_b)));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
      &plan.a, CUDA_R_16BF, m, k, k));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
      &plan.b, CUDA_R_16BF, n, k, k));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
      &plan.c, CUDA_R_16BF, m, n, n));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
      &plan.d, CUDA_R_16BF, m, n, n));
  const cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
  for (auto layout : {plan.a, plan.b, plan.c, plan.d}) {
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        layout, CUBLASLT_MATRIX_LAYOUT_ORDER,
        &row_order, sizeof(row_order)));
  }
  if (batches > 1) {
    const int32_t batch_count = batches;
    const int64_t stride_a = static_cast<int64_t>(m) * k;
    const int64_t stride_b = b_batch_stride < 0
        ? static_cast<int64_t>(n) * k
        : b_batch_stride;
    const int64_t stride_d = static_cast<int64_t>(m) * n;
    for (auto layout : {plan.a, plan.b, plan.c, plan.d}) {
      CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
          layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
          &batch_count, sizeof(batch_count)));
    }
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        plan.a, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &stride_a, sizeof(stride_a)));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        plan.b, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &stride_b, sizeof(stride_b)));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        plan.c, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &stride_d, sizeof(stride_d)));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        plan.d, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &stride_d, sizeof(stride_d)));
  }

  cublasLtMatmulPreference_t preference = nullptr;
  CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
  CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &kCublasLtWorkspaceBytes, sizeof(kCublasLtWorkspaceBytes)));
  constexpr int requested = 64;
  std::vector<cublasLtMatmulHeuristicResult_t> heuristics(requested);
  int returned = 0;
  CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      runtime.blas_lt,
      plan.operation,
      plan.a,
      plan.b,
      plan.c,
      plan.d,
      preference,
      requested,
      heuristics.data(),
      &returned));
  CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
  if (returned == 0) {
    throw std::runtime_error("cuBLASLt returned no BF16 heuristic candidates");
  }

  struct Candidate {
    float milliseconds;
    int index;
  };
  std::vector<Candidate> measured;
  for (int index = 0; index < returned; ++index) {
    if (heuristics[index].state != CUBLAS_STATUS_SUCCESS ||
        heuristics[index].workspaceSize > kCublasLtWorkspaceBytes) {
      continue;
    }
    const float milliseconds = time_cublaslt_candidate(
        runtime, plan, heuristics[index].algo, a, b_nt, d);
    if (std::isfinite(milliseconds)) {
      measured.push_back({milliseconds, index});
    }
  }
  if (measured.empty()) {
    throw std::runtime_error("all cuBLASLt BF16 candidates failed");
  }
  std::sort(measured.begin(), measured.end(), [](const Candidate& lhs, const Candidate& rhs) {
    return lhs.milliseconds < rhs.milliseconds;
  });
  const int best_index = measured.front().index;
  plan.algorithm = heuristics[best_index].algo;
  plan.algorithm_workspace = heuristics[best_index].workspaceSize;
  std::cout << "cuBLASLt autotune: returned=" << returned
            << " valid=" << measured.size()
            << " best_algo=" << cublaslt_algo_attribute(
                   plan.algorithm, CUBLASLT_ALGO_CONFIG_ID)
            << " tile=" << cublaslt_algo_attribute(
                   plan.algorithm, CUBLASLT_ALGO_CONFIG_TILE_ID)
            << " stages=" << cublaslt_algo_attribute(
                   plan.algorithm, CUBLASLT_ALGO_CONFIG_STAGES_ID)
            << " split_k=" << cublaslt_algo_attribute(
                   plan.algorithm, CUBLASLT_ALGO_CONFIG_SPLITK_NUM)
            << " reduction=" << cublaslt_algo_attribute(
                   plan.algorithm, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME)
            << " cta_swizzle=" << cublaslt_algo_attribute(
                   plan.algorithm, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING)
            << " custom=" << cublaslt_algo_attribute(
                   plan.algorithm, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION)
            << " inner=" << cublaslt_algo_u16_attribute(
                   plan.algorithm, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID)
            << " cluster=" << cublaslt_algo_u16_attribute(
                   plan.algorithm, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID)
            << " numerical_flags=0x" << std::hex
            << cublaslt_algo_numerical_flags(plan.algorithm) << std::dec
            << " waves=" << heuristics[best_index].wavesCount
            << " tune_ms=" << std::fixed << std::setprecision(4)
            << measured.front().milliseconds
            << " workspace=" << plan.algorithm_workspace << "\n";
  return plan;
}

void destroy_cublaslt_plan(CublasLtBf16Plan& plan) {
  if (plan.d) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(plan.d));
  if (plan.c) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(plan.c));
  if (plan.b) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(plan.b));
  if (plan.a) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(plan.a));
  if (plan.operation) CUBLAS_CHECK(cublasLtMatmulDescDestroy(plan.operation));
}

using Launch = std::function<void(int, uint32_t)>;

std::vector<float> time_all_ranks(
    const std::vector<Runtime>& runtime,
    int warmup,
    int iterations,
    uint32_t& epoch,
    const Launch& launch) {
  const int world = static_cast<int>(runtime.size());
  for (int iteration = 0; iteration < warmup; ++iteration) {
    ++epoch;
    for (int rank = 0; rank < world; ++rank) {
      launch(rank, epoch);
    }
    synchronize(runtime);
  }

  std::vector<float> samples;
  samples.reserve(iterations);
  for (int iteration = 0; iteration < iterations; ++iteration) {
    ++epoch;
    for (int rank = 0; rank < world; ++rank) {
      CUDA_CHECK(cudaSetDevice(rank));
      CUDA_CHECK(cudaEventRecord(runtime[rank].start, runtime[rank].stream));
      launch(rank, epoch);
      CUDA_CHECK(cudaEventRecord(runtime[rank].stop, runtime[rank].stream));
    }
    float critical = 0.0f;
    for (int rank = 0; rank < world; ++rank) {
      CUDA_CHECK(cudaSetDevice(rank));
      CUDA_CHECK(cudaEventSynchronize(runtime[rank].stop));
      float milliseconds = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(
          &milliseconds, runtime[rank].start, runtime[rank].stop));
      critical = std::max(critical, milliseconds);
    }
    samples.push_back(critical);
  }
  return samples;
}

std::vector<float> time_two_stage_all_ranks(
    const std::vector<Runtime>& runtime,
    int warmup,
    int iterations,
    uint32_t& epoch,
    const Launch& stage_one,
    const Launch& stage_two) {
  const int world = static_cast<int>(runtime.size());
  std::vector<cudaEvent_t> stage_one_done(world);
  for (int rank = 0; rank < world; ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaEventCreateWithFlags(
        &stage_one_done[rank], cudaEventDisableTiming));
  }

  auto submit = [&](bool measured) {
    ++epoch;
    for (int rank = 0; rank < world; ++rank) {
      CUDA_CHECK(cudaSetDevice(rank));
      if (measured) {
        CUDA_CHECK(cudaEventRecord(runtime[rank].start, runtime[rank].stream));
      }
      stage_one(rank, epoch);
      CUDA_CHECK(cudaEventRecord(stage_one_done[rank], runtime[rank].stream));
    }
    for (int rank = 0; rank < world; ++rank) {
      CUDA_CHECK(cudaSetDevice(rank));
      for (int peer = 0; peer < world; ++peer) {
        CUDA_CHECK(cudaStreamWaitEvent(
            runtime[rank].stream, stage_one_done[peer], 0));
      }
      stage_two(rank, epoch);
      if (measured) {
        CUDA_CHECK(cudaEventRecord(runtime[rank].stop, runtime[rank].stream));
      }
    }
  };

  for (int iteration = 0; iteration < warmup; ++iteration) {
    submit(false);
    synchronize(runtime);
  }

  std::vector<float> samples;
  samples.reserve(iterations);
  for (int iteration = 0; iteration < iterations; ++iteration) {
    submit(true);
    float critical = 0.0f;
    for (int rank = 0; rank < world; ++rank) {
      CUDA_CHECK(cudaSetDevice(rank));
      CUDA_CHECK(cudaEventSynchronize(runtime[rank].stop));
      float milliseconds = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(
          &milliseconds, runtime[rank].start, runtime[rank].stop));
      critical = std::max(critical, milliseconds);
    }
    samples.push_back(critical);
  }

  for (int rank = 0; rank < world; ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaEventDestroy(stage_one_done[rank]));
  }
  return samples;
}

// Both stages are submitted on each rank's local stream. The first stage
// publishes device-side peer epochs and the second stage waits on those
// epochs, so no host-side all-rank event fan-in is needed.
std::vector<float> time_peer_ready_two_stage_all_ranks(
    const std::vector<Runtime>& runtime,
    int warmup,
    int iterations,
    uint32_t& epoch,
    const Launch& stage_one,
    const Launch& stage_two) {
  const int world = static_cast<int>(runtime.size());

  auto submit = [&](bool measured) {
    ++epoch;
    for (int rank = 0; rank < world; ++rank) {
      CUDA_CHECK(cudaSetDevice(rank));
      if (measured) {
        CUDA_CHECK(cudaEventRecord(runtime[rank].start, runtime[rank].stream));
      }
      stage_one(rank, epoch);
    }
    for (int rank = 0; rank < world; ++rank) {
      CUDA_CHECK(cudaSetDevice(rank));
      stage_two(rank, epoch);
      if (measured) {
        CUDA_CHECK(cudaEventRecord(runtime[rank].stop, runtime[rank].stream));
      }
    }
  };

  for (int iteration = 0; iteration < warmup; ++iteration) {
    submit(false);
    synchronize(runtime);
  }

  std::vector<float> samples;
  samples.reserve(iterations);
  for (int iteration = 0; iteration < iterations; ++iteration) {
    submit(true);
    float critical = 0.0f;
    for (int rank = 0; rank < world; ++rank) {
      CUDA_CHECK(cudaSetDevice(rank));
      CUDA_CHECK(cudaEventSynchronize(runtime[rank].stop));
      float milliseconds = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(
          &milliseconds, runtime[rank].start, runtime[rank].stop));
      critical = std::max(critical, milliseconds);
    }
    samples.push_back(critical);
  }
  return samples;
}

float percentile(std::vector<float> values, double q) {
  std::sort(values.begin(), values.end());
  const double position = (values.size() - 1) * q;
  const size_t lower = static_cast<size_t>(std::floor(position));
  const size_t upper = static_cast<size_t>(std::ceil(position));
  const double weight = position - lower;
  return static_cast<float>(values[lower] * (1.0 - weight) + values[upper] * weight);
}

struct Summary {
  double mean;
  float p50;
  float p95;
  float minimum;
  float maximum;
};

Summary summarize(const std::vector<float>& samples) {
  return {
      std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size(),
      percentile(samples, 0.50),
      percentile(samples, 0.95),
      *std::min_element(samples.begin(), samples.end()),
      *std::max_element(samples.begin(), samples.end())};
}

void print_result(
    const char* name,
    const std::vector<float>& samples,
    double flops,
    int world,
    double pure_sota_ms,
    double policy_ms = 0.0) {
  const Summary stats = summarize(samples);
  const double tflops = flops / stats.mean / 1.0e9;
  std::cout << std::left << std::setw(20) << name << std::right << " mean=" << std::fixed
            << std::setprecision(4) << stats.mean << " ms p50=" << stats.p50
            << " p95=" << stats.p95 << " TFLOPS/GPU=" << std::setprecision(1) << tflops
            << " aggregate=" << tflops * world;
  if (pure_sota_ms > 0.0) {
    std::cout << " vs_pure_SOTA=" << std::setprecision(1)
              << pure_sota_ms / stats.mean * 100.0 << "%";
  }
  if (policy_ms > 0.0) {
    std::cout << " vs_policy=" << std::setprecision(1)
              << policy_ms / stats.mean * 100.0 << "%";
  }
  std::cout << "\n";
}

void print_copy_result(
    const char* name,
    const std::vector<float>& samples,
    double payload_bytes,
    int world) {
  const Summary stats = summarize(samples);
  const double payload_gbps = payload_bytes / stats.mean / 1.0e6;
  const double remote_gbps = payload_gbps * (world - 1.0) / world;
  std::cout << std::left << std::setw(20) << name << std::right << " mean=" << std::fixed
            << std::setprecision(4) << stats.mean << " ms p50=" << stats.p50
            << " p95=" << stats.p95 << " payload_GB/s/GPU=" << std::setprecision(1)
            << payload_gbps << " remote_GB/s/GPU=" << remote_gbps << "\n";
}

double overlap_ratio(double fused_ms, double gemm_ms, double route_ms) {
  const double shorter = std::min(gemm_ms, route_ms);
  return shorter > 0.0
      ? 1.0 - (fused_ms - std::max(gemm_ms, route_ms)) / shorter
      : 0.0;
}

void write_json(
    const Options& options,
    int world,
    int l,
    double flops,
    double route_payload_bytes,
    const char* route_name,
    double overlap,
    const std::vector<std::pair<std::string, const std::vector<float>*>>& results,
    const fuse::A2ALhsPolicyInfo* lhs_policy = nullptr) {
  if (options.json_out.empty()) {
    return;
  }
  std::ofstream output(options.json_out);
  if (!output) {
    throw std::runtime_error("cannot open JSON output: " + options.json_out);
  }
  output << std::setprecision(10)
         << "{\n  \"mode\": \"" << options.mode << "\",\n"
         << "  \"shape\": {\"m\": " << options.m << ", \"n\": " << options.n
         << ", \"k\": " << options.k << ", \"l\": " << l << "},\n"
         << "  \"world_size\": " << world << ",\n"
         << "  \"comm_ctas\": " << options.comm_ctas << ",\n"
         << "  \"warmup\": " << options.warmup << ",\n"
         << "  \"iterations\": " << options.iterations << ",\n"
         << "  \"dtype\": \""
         << (options.mode == "qkv_gemm_a2a_fp8" ||
                     options.mode == "a2a_gemm_fp8"
                 ? "e4m3xe4m3_fp32acc_bfloat16out"
                 : "bfloat16")
         << "\",\n"
         << "  \"timing\": \"max-rank critical path\",\n"
         << "  \"overlap_ratio\": " << overlap << ",\n";
  if (lhs_policy != nullptr) {
    output << "  \"lhs_policy\": {\"name\": \""
           << lhs_policy_name(lhs_policy->policy)
           << "\", \"tile_m\": " << lhs_policy->tile_m
           << ", \"tile_n\": " << lhs_policy->tile_n
           << ", \"tile_k\": " << lhs_policy->tile_k
           << ", \"cluster_m\": " << lhs_policy->cluster_m
           << ", \"compute_ctas\": " << lhs_policy->compute_ctas
           << ", \"tile_count\": " << lhs_policy->tile_count
           << ", \"waves\": " << lhs_policy->waves
           << ", \"last_wave_ctas\": " << lhs_policy->last_wave_ctas
           << ", \"estimated_cycles\": " << lhs_policy->estimated_cycles
           << "},\n";
  }
  output << "  \"results\": {\n";
  for (size_t index = 0; index < results.size(); ++index) {
    const auto& [name, samples] = results[index];
    const Summary stats = summarize(*samples);
    const double tflops = flops / stats.mean / 1.0e9;
    output << "    \"" << name << "\": {\"mean_ms\": " << stats.mean
           << ", \"p50_ms\": " << stats.p50 << ", \"p95_ms\": " << stats.p95
           << ", \"min_ms\": " << stats.minimum << ", \"max_ms\": " << stats.maximum
           << ", \"tflops_per_gpu\": " << tflops
           << ", \"aggregate_tflops\": " << tflops * world;
    if (name == route_name) {
      const double payload_gbps = route_payload_bytes / stats.mean / 1.0e6;
      output << ", \"payload_gbps_per_gpu\": " << payload_gbps
             << ", \"remote_gbps_per_gpu\": "
             << payload_gbps * (world - 1.0) / world;
    }
    output << "}" << (index + 1 == results.size() ? "\n" : ",\n");
  }
  output << "  }\n}\n";
}

void benchmark_a2a_gemm(
    const Options& options,
    int world,
    const std::vector<Runtime>& runtime) {
  const int m = options.m;
  const int n = options.n;
  const int k = options.k;
  if (m != k || n != options.head_dim || m % world != 0 ||
      options.q_heads % world != 0 || options.kv_heads % world != 0 ||
      options.q_heads % options.kv_heads != 0) {
    throw std::runtime_error(
        "a2a_gemm model adapter requires M=K=global_seq, N=head_dim, and divisible GQA heads");
  }
  const int seq_local = m / world;
  const int local_q_heads = options.q_heads / world;
  const int local_kv_heads = options.kv_heads / world;
  const int l = options.batch * local_q_heads;
  const bool shared_rhs = options.batch == 1 && local_kv_heads == 1;
  fuse::A2AGemmLayout ready_layout{};
  ready_layout.world_size = world;
  ready_layout.outer_batch = options.batch;
  ready_layout.rows_per_peer = seq_local;
  ready_layout.global_rows = m;
  ready_layout.global_input_groups = options.kv_heads;
  ready_layout.local_gemm_groups = local_q_heads;
  ready_layout.group_width = n;
  const int64_t ready_count = fuse::a2a_gemm_ready_elements(ready_layout);
  const int64_t probability_elements = static_cast<int64_t>(l) * m * k;
  const int64_t peer_value_elements =
      static_cast<int64_t>(options.batch) * seq_local * options.kv_heads * n;
  const int64_t staging_elements = static_cast<int64_t>(
      shared_rhs ? 1 : l) * n * k;
  const int64_t output_elements = static_cast<int64_t>(l) * m * n;

  std::vector<Bf16*> probability(world);
  std::vector<Bf16*> peer_value(world);
  std::vector<Bf16*> staging(world);
  std::vector<Bf16*> output(world);
  std::vector<Bf16*> cutlass_output(world);
  std::vector<Bf16*> cublas_output(world);
  std::vector<uint32_t*> ready(world);
  std::vector<fuse::A2AGemmParams> params(world);
  for (int rank = 0; rank < world; ++rank) {
    probability[rank] = allocate<Bf16>(rank, probability_elements);
    peer_value[rank] = allocate<Bf16>(rank, peer_value_elements);
    staging[rank] = allocate<Bf16>(rank, staging_elements);
    output[rank] = allocate<Bf16>(rank, output_elements);
    cutlass_output[rank] = allocate<Bf16>(rank, output_elements);
    cublas_output[rank] = allocate<Bf16>(rank, output_elements);
    ready[rank] = allocate<uint32_t>(rank, ready_count);
    fill(rank, runtime[rank].stream, probability[rank], probability_elements, 701 + rank);
    fill(rank, runtime[rank].stream, peer_value[rank], peer_value_elements, 809 + rank);
    CUDA_CHECK(cudaMemsetAsync(
        staging[rank], 0, staging_elements * sizeof(Bf16), runtime[rank].stream));
    CUDA_CHECK(cudaMemsetAsync(
        ready[rank], 0, ready_count * sizeof(uint32_t), runtime[rank].stream));
  }
  synchronize(runtime);

  for (int rank = 0; rank < world; ++rank) {
    auto& p = params[rank];
    p.lhs = probability[rank];
    for (int peer = 0; peer < world; ++peer) {
      p.peer_rhs[peer] = peer_value[peer];
    }
    p.rhs_nt = staging[rank];
    p.output = output[rank];
    p.ready = ready[rank];
    p.gemm = {m, n, k, l};
    if (shared_rhs) {
      p.gemm.stride_b.batch = 0;
    }
    p.gemm.raster = options.raster;
    p.gemm.max_swizzle_size = options.swizzle;
    p.layout.world_size = world;
    p.layout.rank = rank;
    p.layout.outer_batch = options.batch;
    p.layout.rows_per_peer = seq_local;
    p.layout.global_rows = m;
    p.layout.global_input_groups = options.kv_heads;
    p.layout.local_gemm_groups = local_q_heads;
    p.layout.group_width = n;
    p.num_comm_ctas = options.comm_ctas;
  }

  uint32_t fused_epoch = 0;
  const auto fused = time_all_ranks(
      runtime,
      options.warmup,
      options.iterations,
      fused_epoch,
      [&](int rank, uint32_t epoch) {
        CUDA_CHECK(cudaSetDevice(rank));
        params[rank].epoch = epoch;
        params[rank].output = output[rank];
        CUDA_CHECK(fuse::launch_a2a_gemm_cutlass(
            params[rank], runtime[rank].stream));
      });

  uint32_t cutlass_epoch = 0;
  const auto cutlass = time_all_ranks(
      runtime,
      options.warmup,
      options.iterations,
      cutlass_epoch,
      [&](int rank, uint32_t) {
        CUDA_CHECK(cudaSetDevice(rank));
        params[rank].output = cutlass_output[rank];
        CUDA_CHECK(fuse::launch_a2a_gemm_cutlass_reference(
            params[rank], runtime[rank].stream));
      });

  uint32_t reserved_epoch = 0;
  const auto reserved_cutlass = time_all_ranks(
      runtime,
      options.warmup,
      options.iterations,
      reserved_epoch,
      [&](int rank, uint32_t) {
        CUDA_CHECK(cudaSetDevice(rank));
        params[rank].output = cutlass_output[rank];
        CUDA_CHECK(fuse::launch_a2a_gemm_cutlass_reference(
            params[rank], runtime[rank].stream, options.comm_ctas));
      });

  auto cublaslt_plan = autotune_cublaslt_bf16(
      runtime[0], m, n, k, l,
      probability[0], staging[0], cublas_output[0],
      shared_rhs ? 0 : -1);
  uint32_t cublaslt_epoch = 0;
  const auto cublaslt = time_all_ranks(
      runtime,
      options.warmup,
      options.iterations,
      cublaslt_epoch,
      [&](int rank, uint32_t) {
        CUDA_CHECK(cudaSetDevice(rank));
        cublaslt_nt(
            runtime[rank], cublaslt_plan,
            probability[rank], staging[rank], cublas_output[rank]);
      });

  uint32_t cublas_epoch = 0;
  const auto cublas = time_all_ranks(
      runtime,
      options.warmup,
      options.iterations,
      cublas_epoch,
      [&](int rank, uint32_t) {
        CUDA_CHECK(cudaSetDevice(rank));
        cublas_nt(
            runtime[rank].blas,
            m,
            n,
            k,
            l,
            probability[rank],
            staging[rank],
            cublas_output[rank],
            shared_rhs ? 0 : -1);
      });

  uint32_t route_epoch = fused_epoch;
  const auto route = time_all_ranks(
      runtime,
      options.warmup,
      options.iterations,
      route_epoch,
      [&](int rank, uint32_t epoch) {
        CUDA_CHECK(cudaSetDevice(rank));
        params[rank].epoch = epoch;
        CUDA_CHECK(fuse::launch_a2a_gemm_copy_reference(
            params[rank], runtime[rank].stream));
      });

  uint32_t sequential_epoch = route_epoch;
  const auto sequential = time_all_ranks(
      runtime,
      options.warmup,
      options.iterations,
      sequential_epoch,
      [&](int rank, uint32_t epoch) {
        CUDA_CHECK(cudaSetDevice(rank));
        params[rank].epoch = epoch;
        params[rank].output = cutlass_output[rank];
        CUDA_CHECK(fuse::launch_a2a_gemm_copy_reference(
            params[rank], runtime[rank].stream));
        CUDA_CHECK(fuse::launch_a2a_gemm_cutlass_reference(
            params[rank], runtime[rank].stream));
      });

  const double flops = 2.0 * m * n * k * l;
  const double cutlass_mean = summarize(cutlass).mean;
  const double pure_mean = std::min(
      {cutlass_mean, summarize(cublaslt).mean, summarize(cublas).mean});
  const double payload_bytes =
      static_cast<double>(options.batch) * m * local_kv_heads * n *
      sizeof(Bf16);
  const double overlap = overlap_ratio(
      summarize(fused).mean, cutlass_mean, summarize(route).mean);
  std::cout << "V A2A->PV shape M=" << m << " N=" << n << " K=" << k
            << " L=" << l << " CP=" << world
            << " Hq/Hkv=" << options.q_heads << "/" << options.kv_heads
            << " comm_ctas=" << options.comm_ctas << "\n";
  print_result("cuBLAS BF16", cublas, flops, world, pure_mean);
  print_result("cuBLASLt BF16", cublaslt, flops, world, pure_mean);
  print_result("same-policy CUTLASS", cutlass, flops, world, pure_mean);
  print_result(
      "compute-subgrid CUTLASS",
      reserved_cutlass,
      flops,
      world,
      pure_mean,
      cutlass_mean);
  print_copy_result("V A2A route", route, payload_bytes, world);
  print_result(
      "A2A+PV sequential", sequential, flops, world, pure_mean, cutlass_mean);
  print_result("fused A2A->PV", fused, flops, world, pure_mean, cutlass_mean);
  std::cout << "overlap_ratio=" << std::fixed << std::setprecision(1)
            << overlap * 100.0 << "% (same-policy PV + V route)\n";
  write_json(
      options,
      world,
      l,
      flops,
      payload_bytes,
      "v_a2a_route",
      overlap,
      {{"cublas", &cublas},
       {"cublaslt_autotuned", &cublaslt},
       {"same_policy_cutlass", &cutlass},
       {"compute_subgrid_cutlass", &reserved_cutlass},
       {"v_a2a_route", &route},
       {"sequential", &sequential},
       {"fused", &fused}});
  if (options.role_telemetry) {
    report_a2a_role_telemetry(
        world, runtime, params, ready, ready_count);
  }
  destroy_cublaslt_plan(cublaslt_plan);
}

void benchmark_a2a_lhs_gemm(
    const Options& options,
    int world,
    const std::vector<Runtime>& runtime) {
  const int m = options.m;
  const int n = options.n;
  const int k = options.k;
  if (m % options.batch != 0 || options.q_heads % world != 0 ||
      k != options.q_heads * options.head_dim) {
    throw std::runtime_error(
        "a2a_gemm_lhs requires M divisible by B, Hq divisible by CP, "
        "and K=Hq*D");
  }
  const int seq_local = m / options.batch;
  if (seq_local % 2 != 0) {
    throw std::runtime_error(
        "a2a_gemm_lhs causal load-balanced mapping requires even S_local");
  }
  const int global_seq = seq_local * world;
  const int local_heads = options.q_heads / world;
  const int64_t peer_input_elements =
      static_cast<int64_t>(options.batch) * global_seq * local_heads *
      options.head_dim;
  const int64_t staging_elements = static_cast<int64_t>(m) * k;
  const int64_t weight_elements = static_cast<int64_t>(n) * k;
  const int64_t output_elements = static_cast<int64_t>(m) * n;

  fuse::UlyssesRoute route{};
  route.world_size = world;
  route.batch = options.batch;
  route.global_seq = global_seq;
  route.seq_local = seq_local;
  route.q_heads = options.q_heads;
  route.local_heads = local_heads;
  route.head_dim = options.head_dim;
  route.causal_load_balanced = true;
  route.cyclic_peer_order = true;
  route.kind = fuse::RouteKind::kHeadToSequence;
  route.direction = fuse::RouteDirection::kInverse;
  fuse::GemmProblem problem{m, n, k, 1};
  problem.raster = options.raster;
  problem.max_swizzle_size = options.swizzle;
  Options launch_options = options;
  if (launch_options.lhs_policy == fuse::A2ALhsGemmPolicy::kOneWaveAuto &&
      launch_options.comm_ctas == 0) {
    throw std::runtime_error(
        "--lhs-policy one-wave requires an explicit positive --comm-ctas");
  }
  if (launch_options.comm_ctas == 0) {
    launch_options.comm_ctas =
        fuse::recommended_a2a_lhs_gemm_comm_ctas(problem);
  }
  const int64_t ready_count =
      fuse::a2a_lhs_gemm_ready_elements(problem, route);

  std::vector<Bf16*> peer_input(world);
  std::vector<Bf16*> staging(world);
  std::vector<Bf16*> weight(world);
  std::vector<Bf16*> fused_output(world);
  std::vector<Bf16*> cutlass_output(world);
  std::vector<Bf16*> cublas_output(world);
  std::vector<uint32_t*> ready(world);
  std::vector<unsigned long long*> mismatch_count(world);
  std::vector<fuse::A2AGemmParams> params(world);

  for (int rank = 0; rank < world; ++rank) {
    peer_input[rank] = allocate<Bf16>(rank, peer_input_elements);
    staging[rank] = allocate<Bf16>(rank, staging_elements);
    weight[rank] = allocate<Bf16>(rank, weight_elements);
    fused_output[rank] = allocate<Bf16>(rank, output_elements);
    cutlass_output[rank] = allocate<Bf16>(rank, output_elements);
    cublas_output[rank] = allocate<Bf16>(rank, output_elements);
    ready[rank] = allocate<uint32_t>(rank, ready_count);
    mismatch_count[rank] = allocate<unsigned long long>(rank, 1);
    fill(
        rank,
        runtime[rank].stream,
        peer_input[rank],
        peer_input_elements,
        1901 + rank);
    fill_weight(
        rank,
        runtime[rank].stream,
        weight[rank],
        weight_elements,
        2003);
    CUDA_CHECK(cudaMemsetAsync(
        staging[rank], 0, staging_elements * sizeof(Bf16),
        runtime[rank].stream));
    CUDA_CHECK(cudaMemsetAsync(
        ready[rank], 0, ready_count * sizeof(uint32_t),
        runtime[rank].stream));
  }
  synchronize(runtime);

  for (int rank = 0; rank < world; ++rank) {
    auto& p = params[rank];
    p.input_operand = fuse::A2AGemmOperand::kLhs;
    for (int peer = 0; peer < world; ++peer) {
      p.peer_input[peer] = peer_input[peer];
    }
    p.input_staging = staging[rank];
    p.rhs_nt = weight[rank];
    p.output = fused_output[rank];
    p.ready = ready[rank];
    p.gemm = problem;
    p.route = route;
    p.route.rank = rank;
    p.num_comm_ctas = launch_options.comm_ctas;
    p.lhs_policy = options.lhs_policy;
  }

  int sm_count = 0;
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaDeviceGetAttribute(
      &sm_count, cudaDevAttrMultiProcessorCount, 0));
  const auto policy_info = fuse::select_a2a_lhs_gemm_policy(
      problem, launch_options.comm_ctas, sm_count, options.lhs_policy);

  uint32_t fused_epoch = 0;
  const auto fused = time_all_ranks(
      runtime,
      options.warmup,
      options.iterations,
      fused_epoch,
      [&](int rank, uint32_t epoch) {
        CUDA_CHECK(cudaSetDevice(rank));
        params[rank].epoch = epoch;
        params[rank].output = fused_output[rank];
        CUDA_CHECK(fuse::launch_a2a_gemm_cutlass(
            params[rank], runtime[rank].stream));
      });

  uint32_t cutlass_epoch = 0;
  const auto cutlass = time_all_ranks(
      runtime,
      options.warmup,
      options.iterations,
      cutlass_epoch,
      [&](int rank, uint32_t) {
        CUDA_CHECK(cudaSetDevice(rank));
        params[rank].output = cutlass_output[rank];
        CUDA_CHECK(fuse::launch_a2a_gemm_cutlass_reference(
            params[rank], runtime[rank].stream));
      });

  uint32_t reserved_epoch = 0;
  const auto reserved_cutlass = time_all_ranks(
      runtime,
      options.warmup,
      options.iterations,
      reserved_epoch,
      [&](int rank, uint32_t) {
        CUDA_CHECK(cudaSetDevice(rank));
        params[rank].output = cutlass_output[rank];
        CUDA_CHECK(fuse::launch_a2a_gemm_cutlass_reference(
            params[rank], runtime[rank].stream, launch_options.comm_ctas));
      });

  auto cublaslt_plan = autotune_cublaslt_bf16(
      runtime[0], m, n, k, 1,
      staging[0], weight[0], cublas_output[0]);
  uint32_t cublaslt_epoch = 0;
  const auto cublaslt = time_all_ranks(
      runtime,
      options.warmup,
      options.iterations,
      cublaslt_epoch,
      [&](int rank, uint32_t) {
        CUDA_CHECK(cudaSetDevice(rank));
        cublaslt_nt(
            runtime[rank], cublaslt_plan,
            staging[rank], weight[rank], cublas_output[rank]);
      });

  uint32_t cublas_epoch = 0;
  const auto cublas = time_all_ranks(
      runtime,
      options.warmup,
      options.iterations,
      cublas_epoch,
      [&](int rank, uint32_t) {
        CUDA_CHECK(cudaSetDevice(rank));
        cublas_nt(
            runtime[rank].blas, m, n, k, 1,
            staging[rank], weight[rank], cublas_output[rank]);
      });

  uint32_t route_epoch = fused_epoch;
  const auto inverse_route = time_all_ranks(
      runtime,
      options.warmup,
      options.iterations,
      route_epoch,
      [&](int rank, uint32_t epoch) {
        CUDA_CHECK(cudaSetDevice(rank));
        params[rank].epoch = epoch;
        CUDA_CHECK(fuse::launch_a2a_gemm_copy_reference(
            params[rank], runtime[rank].stream));
      });

  uint32_t sequential_epoch = route_epoch;
  const auto sequential = time_all_ranks(
      runtime,
      options.warmup,
      options.iterations,
      sequential_epoch,
      [&](int rank, uint32_t epoch) {
        CUDA_CHECK(cudaSetDevice(rank));
        params[rank].epoch = epoch;
        params[rank].output = cutlass_output[rank];
        CUDA_CHECK(fuse::launch_a2a_gemm_copy_reference(
            params[rank], runtime[rank].stream));
        CUDA_CHECK(fuse::launch_a2a_gemm_cutlass_reference(
            params[rank], runtime[rank].stream));
      });

  uint32_t cublaslt_sequential_epoch = sequential_epoch;
  const auto cublaslt_sequential = time_all_ranks(
      runtime,
      options.warmup,
      options.iterations,
      cublaslt_sequential_epoch,
      [&](int rank, uint32_t epoch) {
        CUDA_CHECK(cudaSetDevice(rank));
        params[rank].epoch = epoch;
        CUDA_CHECK(fuse::launch_a2a_gemm_copy_reference(
            params[rank], runtime[rank].stream));
        cublaslt_nt(
            runtime[rank], cublaslt_plan,
            staging[rank], weight[rank], cublas_output[rank]);
      });

  unsigned long long exact_mismatches = 0;
  for (int rank = 0; rank < world; ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaMemsetAsync(
        mismatch_count[rank], 0, sizeof(unsigned long long),
        runtime[rank].stream));
    const int blocks = static_cast<int>(
        std::min<int64_t>((output_elements + 255) / 256, 4096));
    count_u16_mismatch_kernel<<<blocks, 256, 0, runtime[rank].stream>>>(
        reinterpret_cast<const uint16_t*>(fused_output[rank]),
        reinterpret_cast<const uint16_t*>(cutlass_output[rank]),
        output_elements,
        mismatch_count[rank]);
    CUDA_CHECK(cudaGetLastError());
  }
  synchronize(runtime);
  for (int rank = 0; rank < world; ++rank) {
    unsigned long long rank_mismatches = 0;
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaMemcpy(
        &rank_mismatches,
        mismatch_count[rank],
        sizeof(rank_mismatches),
        cudaMemcpyDeviceToHost));
    exact_mismatches += rank_mismatches;
  }
  if (exact_mismatches != 0) {
    throw std::runtime_error("fused A2A->GEMM output mismatch");
  }

  const double flops = 2.0 * m * n * k;
  const double cutlass_mean = summarize(cutlass).mean;
  const double reserved_mean = summarize(reserved_cutlass).mean;
  const double pure_mean = std::min(
      {cutlass_mean, summarize(cublaslt).mean, summarize(cublas).mean});
  const double payload_bytes = static_cast<double>(m) * k * sizeof(Bf16);
  const double overlap = overlap_ratio(
      summarize(fused).mean,
      cutlass_mean,
      summarize(inverse_route).mean);
  const double subgrid_overlap = overlap_ratio(
      summarize(fused).mean,
      reserved_mean,
      summarize(inverse_route).mean);

  std::cout << "HeadToSequence A2A->GEMM shape M=" << m
            << " N=" << n << " K=" << k << " L=1 CP=" << world
            << " B=" << options.batch << " S_global=" << global_seq
            << " Hq=" << options.q_heads << " D=" << options.head_dim
            << " comm_ctas=" << launch_options.comm_ctas
            << " policy=" << lhs_policy_name(policy_info.policy)
            << " tile=" << policy_info.tile_m << "x" << policy_info.tile_n
            << " waves=" << policy_info.waves
            << " last_wave=" << policy_info.last_wave_ctas
            << " exact_mismatches=" << exact_mismatches << "\n";
  print_result("cuBLAS BF16", cublas, flops, world, pure_mean);
  print_result("cuBLASLt BF16", cublaslt, flops, world, pure_mean);
  print_result("same-policy CUTLASS", cutlass, flops, world, pure_mean);
  print_result(
      "compute-subgrid CUTLASS",
      reserved_cutlass,
      flops,
      world,
      pure_mean,
      cutlass_mean);
  print_copy_result("inverse A2A route", inverse_route, payload_bytes, world);
  print_result(
      "CUTLASS sequential",
      sequential,
      flops,
      world,
      pure_mean,
      cutlass_mean);
  print_result(
      "cuBLASLt sequential",
      cublaslt_sequential,
      flops,
      world,
      pure_mean,
      summarize(cublaslt).mean);
  print_result(
      "fused A2A->GEMM",
      fused,
      flops,
      world,
      pure_mean,
      cutlass_mean);
  std::cout << "fused_time_as_same_policy_sequential=" << std::fixed
            << std::setprecision(1)
            << summarize(fused).mean / summarize(sequential).mean * 100.0
            << "%\n"
            << "overlap_ratio_full_sm=" << overlap * 100.0
            << "% overlap_ratio_same_compute_subgrid="
            << subgrid_overlap * 100.0 << "%\n";

  write_json(
      launch_options,
      world,
      1,
      flops,
      payload_bytes,
      "inverse_a2a_route",
      overlap,
      {{"cublas", &cublas},
       {"cublaslt_autotuned", &cublaslt},
       {"same_policy_cutlass", &cutlass},
       {"compute_subgrid_cutlass", &reserved_cutlass},
       {"inverse_a2a_route", &inverse_route},
       {"same_policy_sequential", &sequential},
       {"cublaslt_sequential", &cublaslt_sequential},
       {"fused", &fused}},
      &policy_info);
  if (options.role_telemetry) {
    const std::string trace_path = options.trace_out.empty()
        ? "/home/chen/workspace/oproj_overlap_trace.json"
        : options.trace_out;
    report_a2a_lhs_role_trace(
        world, runtime, params, ready, ready_count, trace_path);
  }
  destroy_cublaslt_plan(cublaslt_plan);
}

void benchmark_qkv_gemm_a2a(
    const Options& options, int world, const std::vector<Runtime>& runtime) {
  const int m = options.m;
  const int n = (options.q_heads + 2 * options.kv_heads) * options.head_dim;
  const int k = options.k;
  const fuse::GemmProblem problem{m, n, k, 1};
  const auto traits = fuse::qkv_cutlass_kernel_traits(problem);
  const int seq_local = m / options.batch;
  const int global_seq = seq_local * world;
  const int local_width =
      (options.q_heads / world +
       (options.defer_v_a2a ? 1 : 2) * options.kv_heads / world) *
      options.head_dim;
  const int ready_count =
      ((m + traits.block_m - 1) / traits.block_m) *
      ((n + traits.block_n - 1) / traits.block_n) * fuse::kReadyFlagStride;

  std::vector<Bf16*> lhs(world);
  std::vector<Bf16*> rhs_nt(world);
  std::vector<Bf16*> local_output(world);
  std::vector<Bf16*> peer_output(world);
  std::vector<uint32_t*> ready(world);
  std::vector<uint32_t*> consumed_epoch(world);
  std::vector<fuse::GemmA2AParams> params(world);
  for (int rank = 0; rank < world; ++rank) {
    lhs[rank] = allocate<Bf16>(rank, static_cast<int64_t>(m) * k);
    rhs_nt[rank] = allocate<Bf16>(rank, static_cast<int64_t>(n) * k);
    local_output[rank] = allocate<Bf16>(rank, static_cast<int64_t>(m) * n);
    peer_output[rank] = allocate<Bf16>(
        rank, static_cast<int64_t>(options.batch) * global_seq * local_width);
    ready[rank] = allocate<uint32_t>(rank, ready_count);
    consumed_epoch[rank] = allocate<uint32_t>(
        rank, world * fuse::kReadyFlagStride);
    fill(rank, runtime[rank].stream, lhs[rank], static_cast<int64_t>(m) * k, rank + 601);
    fill(rank, runtime[rank].stream, rhs_nt[rank], static_cast<int64_t>(n) * k, rank + 701);
    CUDA_CHECK(cudaMemsetAsync(
        ready[rank], 0, ready_count * sizeof(uint32_t), runtime[rank].stream));
    CUDA_CHECK(cudaMemsetAsync(
        consumed_epoch[rank],
        0,
        world * fuse::kReadyFlagStride * sizeof(uint32_t),
        runtime[rank].stream));
  }
  synchronize(runtime);

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
    params[rank].gemm.raster = options.raster;
    params[rank].gemm.max_swizzle_size = options.swizzle;
    params[rank].route.world_size = world;
    params[rank].route.rank = rank;
    params[rank].route.batch = options.batch;
    params[rank].route.q_heads = options.q_heads;
    params[rank].route.kv_heads = options.kv_heads;
    params[rank].route.head_dim = options.head_dim;
    params[rank].route.seq_local = seq_local;
    params[rank].route.global_seq = global_seq;
    params[rank].route.qkv_peer_interleaved =
        m < 2048 && options.head_dim == 128 &&
        options.raster == fuse::GemmRaster::kAlongM;
    params[rank].route.kind = fuse::RouteKind::kQkvGqaPack;
    params[rank].route.defer_v_a2a = options.defer_v_a2a;
    params[rank].route.direction = fuse::RouteDirection::kForward;
    params[rank].num_comm_ctas = options.comm_ctas;
  }

  uint32_t epoch = 0;
  const auto fused = time_all_ranks(
      runtime,
      options.warmup,
      options.iterations,
      epoch,
      [&](int rank, uint32_t current_epoch) {
        CUDA_CHECK(cudaSetDevice(rank));
        params[rank].epoch = current_epoch;
        CUDA_CHECK(fuse::launch_gemm_a2a_cutlass(params[rank], runtime[rank].stream));
      });

  const bool deepgemm_fused_supported =
      m == 512 && n == 10240 && k == 8192 &&
      (options.comm_ctas == 8 || options.comm_ctas == 12 ||
       options.comm_ctas == 16 || options.comm_ctas == 20 ||
       options.comm_ctas == 24);
  std::vector<float> deepgemm_fused;
  std::vector<float> deepgemm_compute_only;
  if (deepgemm_fused_supported) {
    for (int rank = 0; rank < world; ++rank) {
      CUDA_CHECK(cudaSetDevice(rank));
      CUDA_CHECK(cudaMemsetAsync(
          ready[rank], 0, ready_count * sizeof(uint32_t),
          runtime[rank].stream));
    }
    synchronize(runtime);
    uint32_t deepgemm_fused_epoch = 0;
    deepgemm_fused = time_all_ranks(
        runtime,
        options.warmup,
        options.iterations,
        deepgemm_fused_epoch,
        [&](int rank, uint32_t current_epoch) {
          CUDA_CHECK(cudaSetDevice(rank));
          params[rank].epoch = current_epoch;
          CUDA_CHECK(fuse::launch_gemm_a2a_deepgemm(
              params[rank], runtime[rank].stream));
        });
    uint32_t deepgemm_compute_only_epoch = 0;
    deepgemm_compute_only = time_all_ranks(
        runtime,
        options.warmup,
        options.iterations,
        deepgemm_compute_only_epoch,
        [&](int rank, uint32_t current_epoch) {
          CUDA_CHECK(cudaSetDevice(rank));
          params[rank].epoch = current_epoch;
          CUDA_CHECK(fuse::launch_gemm_a2a_deepgemm_compute_only(
              params[rank], runtime[rank].stream));
        });
  }

  auto cublaslt_plan = autotune_cublaslt_bf16(
      runtime[0], m, n, k, 1, lhs[0], rhs_nt[0], local_output[0]);
  uint32_t cublaslt_epoch = 0;
  const auto pure_cublaslt = time_all_ranks(
      runtime,
      options.warmup,
      options.iterations,
      cublaslt_epoch,
      [&](int rank, uint32_t) {
        CUDA_CHECK(cudaSetDevice(rank));
        cublaslt_nt(
            runtime[rank], cublaslt_plan,
            lhs[rank], rhs_nt[rank], local_output[rank]);
      });

  uint32_t cublas_epoch = 0;
  const auto pure = time_all_ranks(
      runtime,
      options.warmup,
      options.iterations,
      cublas_epoch,
      [&](int rank, uint32_t) {
        CUDA_CHECK(cudaSetDevice(rank));
        cublas_nt(
            runtime[rank].blas,
            m,
            n,
            k,
            1,
            lhs[rank],
            rhs_nt[rank],
            local_output[rank]);
      });

  uint32_t cutlass_epoch = 0;
  const auto cutlass = time_all_ranks(
      runtime,
      options.warmup,
      options.iterations,
      cutlass_epoch,
      [&](int rank, uint32_t) {
        CUDA_CHECK(cudaSetDevice(rank));
        CUDA_CHECK(fuse::launch_batched_cutlass_reference(
            params[rank], runtime[rank].stream));
      });

  uint32_t reserved_cutlass_epoch = 0;
  const auto reserved_cutlass = time_all_ranks(
      runtime,
      options.warmup,
      options.iterations,
      reserved_cutlass_epoch,
      [&](int rank, uint32_t) {
        CUDA_CHECK(cudaSetDevice(rank));
        CUDA_CHECK(fuse::launch_batched_cutlass_reference(
            params[rank], runtime[rank].stream, options.comm_ctas));
      });

  const bool deepgemm_supported =
      m == 512 && n == 10240 && k == 8192 &&
      (options.comm_ctas == 4 || options.comm_ctas == 8 ||
       options.comm_ctas == 12 || options.comm_ctas == 16 ||
       options.comm_ctas == 20 || options.comm_ctas == 24);
  std::vector<float> deepgemm;
  std::vector<float> reserved_deepgemm;
  if (deepgemm_supported) {
    uint32_t deepgemm_epoch = 0;
    deepgemm = time_all_ranks(
        runtime,
        options.warmup,
        options.iterations,
        deepgemm_epoch,
        [&](int rank, uint32_t) {
          CUDA_CHECK(cudaSetDevice(rank));
          CUDA_CHECK(fuse::launch_deepgemm_bf16_reference(
              params[rank], runtime[rank].stream));
        });
    uint32_t reserved_deepgemm_epoch = 0;
    reserved_deepgemm = time_all_ranks(
        runtime,
        options.warmup,
        options.iterations,
        reserved_deepgemm_epoch,
        [&](int rank, uint32_t) {
          CUDA_CHECK(cudaSetDevice(rank));
          CUDA_CHECK(fuse::launch_deepgemm_bf16_reference(
              params[rank], runtime[rank].stream, options.comm_ctas));
        });
  }

  uint32_t copy_epoch = 0;
  const auto copy = time_all_ranks(
      runtime,
      options.warmup,
      options.iterations,
      copy_epoch,
      [&](int rank, uint32_t) {
        CUDA_CHECK(cudaSetDevice(rank));
        CUDA_CHECK(fuse::launch_gemm_a2a_copy_reference(
            params[rank], runtime[rank].stream));
      });

  std::vector<float> deepgemm_copy;
  if (deepgemm_fused_supported) {
    uint32_t deepgemm_copy_epoch = 0;
    deepgemm_copy = time_all_ranks(
        runtime,
        options.warmup,
        options.iterations,
        deepgemm_copy_epoch,
        [&](int rank, uint32_t) {
          CUDA_CHECK(cudaSetDevice(rank));
          CUDA_CHECK(fuse::launch_gemm_a2a_deepgemm_copy_reference(
              params[rank], runtime[rank].stream));
        });
  }

  uint32_t sequential_epoch = 0;
  const auto sequential = time_all_ranks(
      runtime,
      options.warmup,
      options.iterations,
      sequential_epoch,
      [&](int rank, uint32_t) {
        CUDA_CHECK(cudaSetDevice(rank));
        CUDA_CHECK(fuse::launch_batched_cutlass_reference(
            params[rank], runtime[rank].stream));
        CUDA_CHECK(fuse::launch_gemm_a2a_copy_reference(
            params[rank], runtime[rank].stream));
      });

  const double flops = 2.0 * m * n * k;
  const double cublas_mean = summarize(pure).mean;
  const double cublaslt_mean = summarize(pure_cublaslt).mean;
  const double cutlass_mean = summarize(cutlass).mean;
  double pure_mean = std::min({cublas_mean, cublaslt_mean, cutlass_mean});
  const std::vector<float>* pure_sota = &pure_cublaslt;
  if (cublas_mean == pure_mean) pure_sota = &pure;
  if (cutlass_mean == pure_mean) pure_sota = &cutlass;
  if (deepgemm_supported && summarize(deepgemm).mean < pure_mean) {
    pure_mean = summarize(deepgemm).mean;
    pure_sota = &deepgemm;
  }
  const double reserved_cutlass_mean = summarize(reserved_cutlass).mean;
  const double copy_payload_bytes = static_cast<double>(m) *
      (options.q_heads +
       (options.defer_v_a2a ? 1 : 2) * options.kv_heads) *
      options.head_dim * sizeof(Bf16);
  std::cout << "QKV-GQA GEMM->A2A shape M=" << m << " N=" << n
            << " K=" << k << " Hq=" << options.q_heads
            << " Hkv=" << options.kv_heads << " D=" << options.head_dim
            << " world=" << world << " comm_ctas=" << options.comm_ctas << "\n";
  print_result("cuBLAS BF16", pure, flops, world, pure_mean);
  print_result("cuBLASLt BF16", pure_cublaslt, flops, world, pure_mean);
  print_result("same-policy CUTLASS", cutlass, flops, world, pure_mean);
  print_result(
      "compute-subgrid CUTLASS",
      reserved_cutlass,
      flops,
      world,
      pure_mean,
      cutlass_mean);
  if (deepgemm_supported) {
    print_result("DeepGEMM BF16", deepgemm, flops, world, pure_mean);
    print_result(
        "subgrid DeepGEMM",
        reserved_deepgemm,
        flops,
        world,
        pure_mean,
        summarize(deepgemm).mean);
  }
  print_copy_result(
      options.defer_v_a2a ? "Q/K A2A route" : "Q/K/V A2A route",
      copy,
      copy_payload_bytes,
      world);
  if (deepgemm_fused_supported) {
    print_copy_result(
        "Q/K A2A route candidate", deepgemm_copy, copy_payload_bytes, world);
  }
  print_result("GEMM+A2A sequential", sequential, flops, world, pure_mean, cutlass_mean);
  print_result("fused QKV GEMM->A2A", fused, flops, world, pure_mean, cutlass_mean);
  if (deepgemm_fused_supported) {
    print_result(
        "DeepGEMM signaled compute",
        deepgemm_compute_only,
        flops,
        world,
        pure_mean,
        summarize(reserved_deepgemm).mean);
    print_result(
        "DeepGEMM fused",
        deepgemm_fused,
        flops,
        world,
        pure_mean,
        summarize(reserved_deepgemm).mean);
    std::cout << "deepgemm_overlap_ratio=" << std::fixed
              << std::setprecision(1)
              << overlap_ratio(
                     summarize(deepgemm_fused).mean,
                     summarize(reserved_deepgemm).mean,
                     summarize(copy).mean) *
                     100.0
              << "% (subgrid DeepGEMM + QKV route)\n";
  }
  std::cout << "fused_vs_compute_subgrid=" << std::fixed << std::setprecision(1)
            << reserved_cutlass_mean / summarize(fused).mean * 100.0 << "%\n";
  const double overlap =
      overlap_ratio(summarize(fused).mean, cutlass_mean, summarize(copy).mean);
  std::cout << "overlap_ratio=" << std::fixed << std::setprecision(1)
            << overlap * 100.0 << "% (same-policy GEMM + QKV route)\n";
  Options resolved = options;
  resolved.n = n;
  std::vector<std::pair<std::string, const std::vector<float>*>> results = {
      {"cublas", &pure},
      {"cublaslt_autotuned", &pure_cublaslt},
      {"pure_sota", pure_sota},
      {"same_policy_cutlass", &cutlass},
      {"compute_subgrid_cutlass", &reserved_cutlass}};
  if (deepgemm_supported) {
    results.emplace_back("deepgemm", &deepgemm);
    results.emplace_back("compute_subgrid_deepgemm", &reserved_deepgemm);
  }
  results.emplace_back("qk_a2a_route", &copy);
  if (deepgemm_fused_supported) {
    results.emplace_back("qk_a2a_route_candidate", &deepgemm_copy);
  }
  results.emplace_back("sequential", &sequential);
  results.emplace_back("fused", &fused);
  if (deepgemm_fused_supported) {
    results.emplace_back(
        "deepgemm_signaled_compute", &deepgemm_compute_only);
    results.emplace_back("deepgemm_fused", &deepgemm_fused);
  }
  write_json(
      resolved,
      world,
      1,
      flops,
      copy_payload_bytes,
      "qk_a2a_route",
      overlap,
      results);
  destroy_cublaslt_plan(cublaslt_plan);
}

void benchmark_fp8_qkv_gemm_a2a(
    const Options& options, int world, const std::vector<Runtime>& runtime) {
  const auto traits = fuse::fp8_cutlass_kernel_traits();
  const int m = options.m;
  const int n = (options.q_heads + 2 * options.kv_heads) * options.head_dim;
  const int k = options.k;
  const int seq_local = m / options.batch;
  const int global_seq = seq_local * world;
  const int local_width =
      (options.q_heads / world + options.kv_heads / world) *
      options.head_dim;
  const int ready_count =
      ((m + traits.block_m - 1) / traits.block_m) *
      ((n + traits.block_n - 1) / traits.block_n) * fuse::kReadyFlagStride;

  std::vector<Fp8E4m3*> lhs(world);
  std::vector<Fp8E4m3*> rhs_nt(world);
  std::vector<Bf16*> local_output(world);
  std::vector<Bf16*> peer_output(world);
  std::vector<uint32_t*> ready(world);
  std::vector<uint32_t*> consumed_epoch(world);
  std::vector<fuse::Fp8GemmA2AParams> params(world);
  for (int rank = 0; rank < world; ++rank) {
    lhs[rank] = allocate<Fp8E4m3>(rank, static_cast<int64_t>(m) * k);
    rhs_nt[rank] = allocate<Fp8E4m3>(rank, static_cast<int64_t>(n) * k);
    local_output[rank] = allocate<Bf16>(rank, static_cast<int64_t>(m) * n);
    peer_output[rank] = allocate<Bf16>(
        rank, static_cast<int64_t>(options.batch) * global_seq * local_width);
    ready[rank] = allocate<uint32_t>(rank, ready_count);
    consumed_epoch[rank] = allocate<uint32_t>(
        rank, world * fuse::kReadyFlagStride);
    fill_fp8(
        rank, runtime[rank].stream, lhs[rank], static_cast<int64_t>(m) * k,
        rank + 1009);
    fill_fp8(
        rank, runtime[rank].stream, rhs_nt[rank], static_cast<int64_t>(n) * k,
        rank + 1103);
    CUDA_CHECK(cudaMemsetAsync(
        ready[rank], 0, ready_count * sizeof(uint32_t), runtime[rank].stream));
    CUDA_CHECK(cudaMemsetAsync(
        consumed_epoch[rank],
        0,
        world * fuse::kReadyFlagStride * sizeof(uint32_t),
        runtime[rank].stream));
  }
  synchronize(runtime);

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
    params[rank].gemm.input_dtype = fuse::DType::kFloat8E4M3;
    params[rank].gemm.weight_dtype = fuse::DType::kFloat8E4M3;
    params[rank].gemm.raster = options.raster;
    params[rank].gemm.max_swizzle_size = options.swizzle;
    params[rank].route.world_size = world;
    params[rank].route.rank = rank;
    params[rank].route.batch = options.batch;
    params[rank].route.q_heads = options.q_heads;
    params[rank].route.kv_heads = options.kv_heads;
    params[rank].route.head_dim = options.head_dim;
    params[rank].route.seq_local = seq_local;
    params[rank].route.global_seq = global_seq;
    params[rank].route.kind = fuse::RouteKind::kQkvGqaPack;
    params[rank].route.defer_v_a2a = true;
    params[rank].route.direction = fuse::RouteDirection::kForward;
    params[rank].num_comm_ctas = options.comm_ctas;
  }

  uint32_t fused_epoch = 0;
  const auto fused = time_all_ranks(
      runtime,
      options.warmup,
      options.iterations,
      fused_epoch,
      [&](int rank, uint32_t current_epoch) {
        CUDA_CHECK(cudaSetDevice(rank));
        params[rank].epoch = current_epoch;
        CUDA_CHECK(fuse::launch_gemm_a2a_fp8_cutlass(
            params[rank], runtime[rank].stream));
      });

  uint32_t cutlass_epoch = 0;
  const auto cutlass = time_all_ranks(
      runtime,
      options.warmup,
      options.iterations,
      cutlass_epoch,
      [&](int rank, uint32_t) {
        CUDA_CHECK(cudaSetDevice(rank));
        CUDA_CHECK(fuse::launch_dense_fp8_cutlass_reference(
            params[rank], runtime[rank].stream));
      });

  uint32_t reserved_epoch = 0;
  const auto reserved_cutlass = time_all_ranks(
      runtime,
      options.warmup,
      options.iterations,
      reserved_epoch,
      [&](int rank, uint32_t) {
        CUDA_CHECK(cudaSetDevice(rank));
        CUDA_CHECK(fuse::launch_dense_fp8_cutlass_reference(
            params[rank], runtime[rank].stream, options.comm_ctas));
      });

  uint32_t copy_epoch = 0;
  const auto copy = time_all_ranks(
      runtime,
      options.warmup,
      options.iterations,
      copy_epoch,
      [&](int rank, uint32_t) {
        CUDA_CHECK(cudaSetDevice(rank));
        CUDA_CHECK(fuse::launch_gemm_a2a_fp8_copy_reference(
            params[rank], runtime[rank].stream));
      });

  uint32_t sequential_epoch = 0;
  const auto sequential = time_all_ranks(
      runtime,
      options.warmup,
      options.iterations,
      sequential_epoch,
      [&](int rank, uint32_t) {
        CUDA_CHECK(cudaSetDevice(rank));
        CUDA_CHECK(fuse::launch_dense_fp8_cutlass_reference(
            params[rank], runtime[rank].stream));
        CUDA_CHECK(fuse::launch_gemm_a2a_fp8_copy_reference(
            params[rank], runtime[rank].stream));
      });

  const double flops = 2.0 * m * n * k;
  const double cutlass_mean = summarize(cutlass).mean;
  const double reserved_mean = summarize(reserved_cutlass).mean;
  const double qk_a2a_bytes = static_cast<double>(m) *
      (options.q_heads + options.kv_heads) * options.head_dim * sizeof(Bf16);
  const double copy_payload_bytes = qk_a2a_bytes;
  std::cout << "FP8 QKV-GQA GEMM->A2A shape M=" << m << " N=" << n
            << " K=" << k << " Hq=" << options.q_heads
            << " Hkv=" << options.kv_heads << " D=" << options.head_dim
            << " world=" << world << " comm_ctas=" << options.comm_ctas << "\n";
  print_result("same-policy FP8 CUTLASS", cutlass, flops, world, 0.0);
  print_result(
      "compute-subgrid CUTLASS",
      reserved_cutlass,
      flops,
      world,
      0.0,
      cutlass_mean);
  print_copy_result("Q/K BF16 A2A route", copy, copy_payload_bytes, world);
  print_result("FP8 GEMM+A2A seq", sequential, flops, world, 0.0, cutlass_mean);
  print_result("fused FP8 QKV+A2A", fused, flops, world, 0.0, cutlass_mean);
  std::cout << "fused_vs_compute_subgrid=" << std::fixed << std::setprecision(1)
            << reserved_mean / summarize(fused).mean * 100.0 << "%\n";
  const double overlap =
      overlap_ratio(summarize(fused).mean, cutlass_mean, summarize(copy).mean);
  std::cout << "overlap_ratio=" << std::fixed << std::setprecision(1)
            << overlap * 100.0 << "% (FP8 CUTLASS + BF16 QKV route)\n";
  Options resolved = options;
  resolved.n = n;
  write_json(
      resolved,
      world,
      1,
      flops,
      copy_payload_bytes,
      "qk_a2a_route",
      overlap,
      {{"same_policy_cutlass", &cutlass},
       {"compute_subgrid_cutlass", &reserved_cutlass},
       {"qk_a2a_route", &copy},
       {"sequential", &sequential},
       {"fused", &fused}});
}

void benchmark_ulysses_forward_boundaries(
    const Options& options,
    int world,
    const std::vector<Runtime>& runtime) {
  const int global_seq = options.m;
  const int hidden = options.k;
  if ((static_cast<int64_t>(options.batch) * global_seq) % world != 0 ||
      global_seq % world != 0 || options.q_heads % world != 0 ||
      options.kv_heads % world != 0 ||
      options.q_heads % options.kv_heads != 0 || options.head_dim != 128) {
    throw std::runtime_error(
        "ulysses_forward requires divisible B*S/CP, S/CP, GQA heads, and D=128");
  }
  const int seq_local = global_seq / world;
  const int qkv_m = options.batch * seq_local;
  const int qkv_n =
      (options.q_heads + 2 * options.kv_heads) * options.head_dim;
  const int qk_local_width =
      (options.q_heads / world + options.kv_heads / world) * options.head_dim;
  const int v_offset =
      (options.q_heads + options.kv_heads) * options.head_dim;
  const int local_q_heads = options.q_heads / world;
  const int local_kv_heads = options.kv_heads / world;
  const bool shared_pv_rhs = options.batch == 1 && local_kv_heads == 1;
  const int pv_l = options.batch * local_q_heads;
  const fuse::GemmProblem qkv_problem{qkv_m, qkv_n, hidden, 1};
  const auto projection_traits = fuse::qkv_cutlass_kernel_traits(qkv_problem);
  const int qkv_ready_count =
      ((qkv_m + projection_traits.block_m - 1) / projection_traits.block_m) *
      ((qkv_n + projection_traits.block_n - 1) / projection_traits.block_n) *
      fuse::kReadyFlagStride;
  fuse::A2AGemmLayout pv_ready_layout{};
  pv_ready_layout.world_size = world;
  pv_ready_layout.outer_batch = options.batch;
  pv_ready_layout.rows_per_peer = seq_local;
  pv_ready_layout.global_rows = global_seq;
  pv_ready_layout.global_input_groups = options.kv_heads;
  pv_ready_layout.local_gemm_groups = local_q_heads;
  pv_ready_layout.group_width = options.head_dim;
  const int64_t pv_ready_count =
      fuse::a2a_gemm_ready_elements(pv_ready_layout);

  const int64_t qkv_lhs_elements = static_cast<int64_t>(qkv_m) * hidden;
  const int64_t qkv_weight_elements = static_cast<int64_t>(qkv_n) * hidden;
  const int64_t qkv_output_elements = static_cast<int64_t>(qkv_m) * qkv_n;
  const int64_t qk_output_elements =
      static_cast<int64_t>(options.batch) * global_seq * qk_local_width;
  const int64_t probability_elements =
      static_cast<int64_t>(pv_l) * global_seq * global_seq;
  const int64_t value_nt_elements =
      static_cast<int64_t>(shared_pv_rhs ? 1 : pv_l) *
      options.head_dim * global_seq;
  const int64_t pv_output_elements =
      static_cast<int64_t>(pv_l) * global_seq * options.head_dim;

  std::vector<Bf16*> qkv_lhs(world);
  std::vector<Bf16*> qkv_weight_nt(world);
  std::vector<Bf16*> qkv_output(world);
  std::vector<Bf16*> qkv_reference(world);
  std::vector<Bf16*> qk_output(world);
  std::vector<Bf16*> qk_reference(world);
  std::vector<uint32_t*> qkv_ready(world);
  std::vector<uint32_t*> qkv_consumed_epoch(world);
  std::vector<uint32_t*> v_ready_epoch(world);
  std::vector<Bf16*> probability(world);
  std::vector<Bf16*> value_nt(world);
  std::vector<Bf16*> value_nt_reference(world);
  std::vector<Bf16*> pv_output(world);
  std::vector<Bf16*> pv_reference(world);
  std::vector<uint32_t*> pv_ready(world);
  std::vector<uint32_t*> pv_reference_ready(world);
  std::vector<uint32_t*> pv_policy_ready(world);
  std::vector<uint32_t*> pv_cublaslt_ready(world);
  std::vector<unsigned long long*> mismatch_count(world);
  std::vector<fuse::GemmA2AParams> qkv_params(world);
  std::vector<fuse::GemmA2AParams> qkv_reference_params(world);
  std::vector<fuse::A2AGemmParams> pv_params(world);
  std::vector<fuse::A2AGemmParams> pv_reference_params(world);

  for (int rank = 0; rank < world; ++rank) {
    qkv_lhs[rank] = allocate<Bf16>(rank, qkv_lhs_elements);
    qkv_weight_nt[rank] = allocate<Bf16>(rank, qkv_weight_elements);
    qkv_output[rank] = allocate<Bf16>(rank, qkv_output_elements);
    qkv_reference[rank] = allocate<Bf16>(rank, qkv_output_elements);
    qk_output[rank] = allocate<Bf16>(rank, qk_output_elements);
    qk_reference[rank] = allocate<Bf16>(rank, qk_output_elements);
    qkv_ready[rank] = allocate<uint32_t>(rank, qkv_ready_count);
    qkv_consumed_epoch[rank] = allocate<uint32_t>(
        rank, world * fuse::kReadyFlagStride);
    v_ready_epoch[rank] = allocate<uint32_t>(rank, fuse::kReadyFlagStride);
    probability[rank] = allocate<Bf16>(rank, probability_elements);
    value_nt[rank] = allocate<Bf16>(rank, value_nt_elements);
    value_nt_reference[rank] = allocate<Bf16>(rank, value_nt_elements);
    pv_output[rank] = allocate<Bf16>(rank, pv_output_elements);
    pv_reference[rank] = allocate<Bf16>(rank, pv_output_elements);
    pv_ready[rank] = allocate<uint32_t>(rank, pv_ready_count);
    pv_reference_ready[rank] = allocate<uint32_t>(rank, pv_ready_count);
    pv_policy_ready[rank] = allocate<uint32_t>(rank, pv_ready_count);
    pv_cublaslt_ready[rank] = allocate<uint32_t>(rank, pv_ready_count);
    mismatch_count[rank] = allocate<unsigned long long>(rank, 1);
    fill(rank, runtime[rank].stream, qkv_lhs[rank], qkv_lhs_elements, 1201 + rank);
    fill(
        rank,
        runtime[rank].stream,
        qkv_weight_nt[rank],
        qkv_weight_elements,
        1301 + rank);
    fill(
        rank,
        runtime[rank].stream,
        probability[rank],
        probability_elements,
        1409 + rank);
    CUDA_CHECK(cudaMemsetAsync(
        qkv_ready[rank], 0, qkv_ready_count * sizeof(uint32_t),
        runtime[rank].stream));
    CUDA_CHECK(cudaMemsetAsync(
        qkv_consumed_epoch[rank],
        0,
        world * fuse::kReadyFlagStride * sizeof(uint32_t),
        runtime[rank].stream));
    CUDA_CHECK(cudaMemsetAsync(
        v_ready_epoch[rank], 0,
        fuse::kReadyFlagStride * sizeof(uint32_t), runtime[rank].stream));
    CUDA_CHECK(cudaMemsetAsync(
        pv_ready[rank], 0, pv_ready_count * sizeof(uint32_t),
        runtime[rank].stream));
    CUDA_CHECK(cudaMemsetAsync(
        pv_reference_ready[rank], 0, pv_ready_count * sizeof(uint32_t),
        runtime[rank].stream));
    CUDA_CHECK(cudaMemsetAsync(
        pv_policy_ready[rank], 0, pv_ready_count * sizeof(uint32_t),
        runtime[rank].stream));
    CUDA_CHECK(cudaMemsetAsync(
        pv_cublaslt_ready[rank], 0, pv_ready_count * sizeof(uint32_t),
        runtime[rank].stream));
  }
  synchronize(runtime);

  for (int rank = 0; rank < world; ++rank) {
    auto& qkv = qkv_params[rank];
    qkv.lhs = qkv_lhs[rank];
    qkv.rhs_nt = qkv_weight_nt[rank];
    qkv.local_output = qkv_output[rank];
    for (int peer = 0; peer < world; ++peer) {
      qkv.peer_output[peer] = qk_output[peer];
      qkv.peer_route_done_epoch[peer] = qkv_consumed_epoch[peer];
    }
    qkv.ready = qkv_ready[rank];
    qkv.completion_epoch = v_ready_epoch[rank];
    qkv.gemm = {qkv_m, qkv_n, hidden, 1};
    qkv.gemm.raster = fuse::GemmRaster::kAlongM;
    qkv.gemm.max_swizzle_size = options.swizzle;
    qkv.route.world_size = world;
    qkv.route.rank = rank;
    qkv.route.batch = options.batch;
    qkv.route.seq_local = seq_local;
    qkv.route.global_seq = global_seq;
    qkv.route.q_heads = options.q_heads;
    qkv.route.kv_heads = options.kv_heads;
    qkv.route.head_dim = options.head_dim;
    qkv.route.kind = fuse::RouteKind::kQkvGqaPack;
    qkv.route.defer_v_a2a = true;
    qkv.route.direction = fuse::RouteDirection::kForward;
    qkv.num_comm_ctas = options.qkv_comm_ctas;
    qkv.epoch = 1;

    qkv_reference_params[rank] = qkv;
    qkv_reference_params[rank].completion_epoch = nullptr;
    qkv_reference_params[rank].local_output = qkv_reference[rank];
    for (int peer = 0; peer < world; ++peer) {
      qkv_reference_params[rank].peer_output[peer] = qk_reference[peer];
    }

    auto& pv = pv_params[rank];
    pv.lhs = probability[rank];
    for (int peer = 0; peer < world; ++peer) {
      pv.peer_rhs[peer] = qkv_output[peer] + v_offset;
      pv.peer_input_ready[peer] = v_ready_epoch[peer];
    }
    pv.peer_rhs_row_stride = qkv_n;
    pv.rhs_nt = value_nt[rank];
    pv.output = pv_output[rank];
    pv.ready = pv_ready[rank];
    pv.gemm = {global_seq, options.head_dim, global_seq, pv_l};
    if (shared_pv_rhs) {
      pv.gemm.stride_b.batch = 0;
    }
    pv.gemm.raster = fuse::GemmRaster::kAlongM;
    pv.gemm.max_swizzle_size = options.swizzle;
    pv.layout.world_size = world;
    pv.layout.rank = rank;
    pv.layout.outer_batch = options.batch;
    pv.layout.rows_per_peer = seq_local;
    pv.layout.global_rows = global_seq;
    pv.layout.global_input_groups = options.kv_heads;
    pv.layout.local_gemm_groups = local_q_heads;
    pv.layout.group_width = options.head_dim;
    pv.num_comm_ctas = options.pv_comm_ctas;
    pv.epoch = 1;
    pv.input_epoch = 1;

    pv_reference_params[rank] = pv;
    pv_reference_params[rank].input_epoch = 0;
    for (int peer = 0; peer < world; ++peer) {
      pv_reference_params[rank].peer_rhs[peer] =
          qkv_reference[peer] + v_offset;
    }
    pv_reference_params[rank].rhs_nt = value_nt_reference[rank];
    pv_reference_params[rank].output = pv_reference[rank];
    pv_reference_params[rank].ready = pv_reference_ready[rank];
  }

  // Exact linked reference: pure QKV, the identical Q/K route, the identical
  // strided V route, and the identical PV collective.
  for (int rank = 0; rank < world; ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(fuse::launch_batched_cutlass_reference(
        qkv_reference_params[rank], runtime[rank].stream));
  }
  synchronize(runtime);
  for (int rank = 0; rank < world; ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(fuse::launch_gemm_a2a_copy_reference(
        qkv_reference_params[rank], runtime[rank].stream));
    CUDA_CHECK(fuse::launch_a2a_gemm_copy_reference(
        pv_reference_params[rank], runtime[rank].stream));
    CUDA_CHECK(fuse::launch_a2a_gemm_cutlass_reference(
        pv_reference_params[rank], runtime[rank].stream));
  }
  synchronize(runtime);

  // Exact linked fused path: submit all producers, then all consumers. Peer
  // epochs provide the only cross-device dependency.
  for (int rank = 0; rank < world; ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(fuse::launch_gemm_a2a_cutlass(
        qkv_params[rank], runtime[rank].stream));
  }
  for (int rank = 0; rank < world; ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(fuse::launch_a2a_gemm_cutlass(
        pv_params[rank], runtime[rank].stream));
  }
  synchronize(runtime);

  auto exact_mismatches = [&](
      const char* label,
      const std::vector<Bf16*>& actual,
      const std::vector<Bf16*>& expected,
      int64_t elements) {
    unsigned long long total = 0;
    for (int rank = 0; rank < world; ++rank) {
      CUDA_CHECK(cudaSetDevice(rank));
      CUDA_CHECK(cudaMemsetAsync(
          mismatch_count[rank], 0, sizeof(unsigned long long),
          runtime[rank].stream));
      const int blocks = static_cast<int>(
          std::min<int64_t>((elements + 255) / 256, 4096));
      count_u16_mismatch_kernel<<<blocks, 256, 0, runtime[rank].stream>>>(
          reinterpret_cast<const uint16_t*>(actual[rank]),
          reinterpret_cast<const uint16_t*>(expected[rank]),
          elements,
          mismatch_count[rank]);
      CUDA_CHECK(cudaGetLastError());
    }
    synchronize(runtime);
    for (int rank = 0; rank < world; ++rank) {
      unsigned long long rank_count = 0;
      CUDA_CHECK(cudaSetDevice(rank));
      CUDA_CHECK(cudaMemcpy(
          &rank_count,
          mismatch_count[rank],
          sizeof(rank_count),
          cudaMemcpyDeviceToHost));
      total += rank_count;
    }
    std::cout << label << "_exact_mismatches=" << total << "\n";
    if (total != 0) {
      throw std::runtime_error(std::string(label) + " linked correctness failed");
    }
  };
  exact_mismatches("qkv_local", qkv_output, qkv_reference, qkv_output_elements);
  exact_mismatches("qk_a2a", qk_output, qk_reference, qk_output_elements);
  exact_mismatches("v_a2a", value_nt, value_nt_reference, value_nt_elements);
  exact_mismatches("pv_output", pv_output, pv_reference, pv_output_elements);

  uint32_t fused_epoch = 1;
  const auto fused = time_peer_ready_two_stage_all_ranks(
      runtime,
      options.warmup,
      options.iterations,
      fused_epoch,
      [&](int rank, uint32_t epoch) {
        CUDA_CHECK(cudaSetDevice(rank));
        qkv_params[rank].epoch = epoch;
        CUDA_CHECK(fuse::launch_gemm_a2a_cutlass(
            qkv_params[rank], runtime[rank].stream));
      },
      [&](int rank, uint32_t epoch) {
        CUDA_CHECK(cudaSetDevice(rank));
        pv_params[rank].epoch = epoch;
        pv_params[rank].input_epoch = epoch;
        CUDA_CHECK(fuse::launch_a2a_gemm_cutlass(
            pv_params[rank], runtime[rank].stream));
      });

  std::vector<fuse::A2AGemmParams> pv_policy_params = pv_params;
  for (int rank = 0; rank < world; ++rank) {
    pv_policy_params[rank].ready = pv_policy_ready[rank];
    pv_policy_params[rank].input_epoch = 0;
  }
  uint32_t policy_epoch = 0;
  const auto policy_sequential = time_two_stage_all_ranks(
      runtime,
      options.warmup,
      options.iterations,
      policy_epoch,
      [&](int rank, uint32_t epoch) {
        CUDA_CHECK(cudaSetDevice(rank));
        qkv_params[rank].epoch = epoch;
        CUDA_CHECK(fuse::launch_batched_cutlass_reference(
            qkv_params[rank], runtime[rank].stream));
        CUDA_CHECK(fuse::launch_gemm_a2a_copy_reference(
            qkv_params[rank], runtime[rank].stream));
      },
      [&](int rank, uint32_t epoch) {
        CUDA_CHECK(cudaSetDevice(rank));
        pv_policy_params[rank].epoch = epoch;
        CUDA_CHECK(fuse::launch_a2a_gemm_copy_reference(
            pv_policy_params[rank], runtime[rank].stream));
        CUDA_CHECK(fuse::launch_a2a_gemm_cutlass_reference(
            pv_policy_params[rank], runtime[rank].stream));
      });

  auto qkv_lt_plan = autotune_cublaslt_bf16(
      runtime[0], qkv_m, qkv_n, hidden, 1,
      qkv_lhs[0], qkv_weight_nt[0], qkv_output[0]);
  auto pv_lt_plan = autotune_cublaslt_bf16(
      runtime[0], global_seq, options.head_dim, global_seq, pv_l,
      probability[0], value_nt[0], pv_output[0],
      shared_pv_rhs ? 0 : -1);
  std::vector<fuse::A2AGemmParams> pv_cublaslt_params = pv_params;
  std::vector<fuse::GemmA2AParams> qkv_cublaslt_params = qkv_params;
  for (int rank = 0; rank < world; ++rank) {
    pv_cublaslt_params[rank].ready = pv_cublaslt_ready[rank];
    pv_cublaslt_params[rank].input_epoch = 0;
    qkv_cublaslt_params[rank].completion_epoch = nullptr;
    // Once cuBLASLt has completed the whole projection, the standalone route
    // is no longer constrained by GEMM producer order. Use its fastest order
    // instead of weakening the external baseline to match fused scheduling.
    qkv_cublaslt_params[rank].gemm.raster = fuse::GemmRaster::kAlongN;
  }
  uint32_t cublaslt_epoch = 0;
  const auto cublaslt_sequential = time_two_stage_all_ranks(
      runtime,
      options.warmup,
      options.iterations,
      cublaslt_epoch,
      [&](int rank, uint32_t epoch) {
        CUDA_CHECK(cudaSetDevice(rank));
        qkv_cublaslt_params[rank].epoch = epoch;
        cublaslt_nt(
            runtime[rank], qkv_lt_plan,
            qkv_lhs[rank], qkv_weight_nt[rank], qkv_output[rank]);
        CUDA_CHECK(fuse::launch_gemm_a2a_copy_reference(
            qkv_cublaslt_params[rank], runtime[rank].stream));
      },
      [&](int rank, uint32_t epoch) {
        CUDA_CHECK(cudaSetDevice(rank));
        pv_cublaslt_params[rank].epoch = epoch;
        CUDA_CHECK(fuse::launch_a2a_gemm_copy_reference(
            pv_cublaslt_params[rank], runtime[rank].stream));
        cublaslt_nt(
            runtime[rank], pv_lt_plan,
            probability[rank], value_nt[rank], pv_output[rank]);
      });

  uint32_t qkv_math_epoch = 0;
  const auto qkv_math = time_all_ranks(
      runtime,
      options.warmup,
      options.iterations,
      qkv_math_epoch,
      [&](int rank, uint32_t) {
        CUDA_CHECK(cudaSetDevice(rank));
        cublaslt_nt(
            runtime[rank], qkv_lt_plan,
            qkv_lhs[rank], qkv_weight_nt[rank], qkv_output[rank]);
      });
  uint32_t pv_math_epoch = 0;
  const auto pv_math = time_all_ranks(
      runtime,
      options.warmup,
      options.iterations,
      pv_math_epoch,
      [&](int rank, uint32_t) {
        CUDA_CHECK(cudaSetDevice(rank));
        cublaslt_nt(
            runtime[rank], pv_lt_plan,
            probability[rank], value_nt[rank], pv_output[rank]);
      });
  std::vector<float> pure_math(qkv_math.size());
  for (size_t sample = 0; sample < pure_math.size(); ++sample) {
    pure_math[sample] = qkv_math[sample] + pv_math[sample];
  }

  const double qkv_flops = 2.0 * qkv_m * qkv_n * hidden;
  const double pv_flops = 2.0 * global_seq * options.head_dim * global_seq * pv_l;
  const double total_flops = qkv_flops + pv_flops;
  const double qk_bytes = static_cast<double>(qkv_m) *
      (options.q_heads + options.kv_heads) * options.head_dim * sizeof(Bf16);
  const double v_bytes = static_cast<double>(options.batch) * global_seq *
      local_kv_heads * options.head_dim * sizeof(Bf16);
  const double pure_math_mean = summarize(pure_math).mean;
  std::cout << "Ulysses forward fusion-boundary pair B=" << options.batch
            << " S_global=" << global_seq << " S_local=" << seq_local
            << " hidden=" << hidden << " Hq/Hkv=" << options.q_heads << "/"
            << options.kv_heads << " D=" << options.head_dim << " CP=" << world
            << " qkv_comm=" << options.qkv_comm_ctas
            << " pv_comm=" << options.pv_comm_ctas << "\n";
  std::cout << "QKV GEMM=(" << qkv_m << "," << qkv_n << "," << hidden
            << ",1) PV GEMM=(" << global_seq << "," << options.head_dim << ","
            << global_seq << "," << pv_l << ")\n";
  print_result(
      "pure cuBLASLt math sum",
      pure_math,
      total_flops,
      world,
      pure_math_mean);
  print_result(
      "same-policy separated",
      policy_sequential,
      total_flops,
      world,
      pure_math_mean);
  print_result(
      "cuBLASLt + routes",
      cublaslt_sequential,
      total_flops,
      world,
      pure_math_mean);
  print_result(
      "fused boundary pair",
      fused,
      total_flops,
      world,
      pure_math_mean);
  std::cout << "fused_vs_best_separated=" << std::fixed << std::setprecision(1)
            << std::min(
                   summarize(policy_sequential).mean,
                   summarize(cublaslt_sequential).mean) /
                   summarize(fused).mean * 100.0
            << "%\n";
  Options resolved = options;
  resolved.n = options.head_dim;
  write_json(
      resolved,
      world,
      pv_l,
      total_flops,
      qk_bytes + v_bytes,
      "qk_a2a_plus_v_a2a",
      0.0,
      {{"pure_cublaslt_math_sum", &pure_math},
       {"same_policy_separated", &policy_sequential},
       {"cublaslt_routes_separated", &cublaslt_sequential},
       {"fused_boundary_pair", &fused}});
  destroy_cublaslt_plan(qkv_lt_plan);
  destroy_cublaslt_plan(pv_lt_plan);
}

void benchmark_dense_forward(
    const Options& options,
    int world,
    const std::vector<Runtime>& runtime) {
  const int global_seq = options.m;
  const int hidden = options.k;
  const int seq_local = global_seq / world;
  const int local_q_heads = options.q_heads / world;
  const int local_kv_heads = options.kv_heads / world;
  const int qkv_m = options.batch * seq_local;
  const int qkv_n =
      (options.q_heads + 2 * options.kv_heads) * options.head_dim;
  const int qk_local_width =
      (local_q_heads + local_kv_heads) * options.head_dim;
  const int v_offset =
      (options.q_heads + options.kv_heads) * options.head_dim;
  const int pv_l = options.batch * local_q_heads;
  if (options.batch != 1 || local_kv_heads != 1 || options.head_dim != 128 ||
      hidden != options.q_heads * options.head_dim ||
      options.ffn_hidden % 128 != 0) {
    throw std::runtime_error(
        "dense_forward currently fixes the optimized 70B B1/GQA CP8 contract");
  }

  const fuse::GemmProblem qkv_problem{qkv_m, qkv_n, hidden, 1};
  const auto qkv_traits = fuse::qkv_cutlass_kernel_traits(qkv_problem);
  const int64_t qkv_ready_count =
      static_cast<int64_t>((qkv_m + qkv_traits.block_m - 1) / qkv_traits.block_m) *
      ((qkv_n + qkv_traits.block_n - 1) / qkv_traits.block_n) *
      fuse::kReadyFlagStride;
  fuse::A2AGemmLayout pv_ready_layout{};
  pv_ready_layout.world_size = world;
  pv_ready_layout.outer_batch = options.batch;
  pv_ready_layout.rows_per_peer = seq_local;
  pv_ready_layout.global_rows = global_seq;
  pv_ready_layout.global_input_groups = options.kv_heads;
  pv_ready_layout.local_gemm_groups = local_q_heads;
  pv_ready_layout.group_width = options.head_dim;
  const int64_t pv_ready_count =
      fuse::a2a_gemm_ready_elements(pv_ready_layout);

  const int64_t hidden_elements = static_cast<int64_t>(qkv_m) * hidden;
  const int64_t qkv_weight_elements = static_cast<int64_t>(qkv_n) * hidden;
  const int64_t qkv_output_elements = static_cast<int64_t>(qkv_m) * qkv_n;
  const int64_t qk_output_elements =
      static_cast<int64_t>(options.batch) * global_seq * qk_local_width;
  const int64_t query_elements =
      static_cast<int64_t>(pv_l) * global_seq * options.head_dim;
  const int64_t key_elements =
      static_cast<int64_t>(global_seq) * options.head_dim;
  const int64_t probability_elements =
      static_cast<int64_t>(pv_l) * global_seq * global_seq;
  const int64_t pv_output_elements =
      static_cast<int64_t>(pv_l) * global_seq * options.head_dim;
  const int64_t value_staging_elements =
      static_cast<int64_t>(global_seq) * options.head_dim;
  const int64_t output_weight_elements =
      static_cast<int64_t>(hidden) * hidden;
  const int64_t gate_up_elements =
      static_cast<int64_t>(qkv_m) * 2 * options.ffn_hidden;
  const int64_t gate_up_weight_elements =
      static_cast<int64_t>(2) * options.ffn_hidden * hidden;
  const int64_t mlp_elements =
      static_cast<int64_t>(qkv_m) * options.ffn_hidden;
  const int64_t down_weight_elements =
      static_cast<int64_t>(hidden) * options.ffn_hidden;

  std::vector<Bf16*> state(world), norm(world), residual(world), projection(world);
  std::vector<Bf16*> attention_norm_weight(world), mlp_norm_weight(world);
  std::vector<Bf16*> qkv_weight(world), qkv_output(world), qk_output(world);
  std::vector<Bf16*> query(world), key(world), probability(world);
  std::vector<Bf16*> value_staging(world), pv_output(world), attention_tokens(world);
  std::vector<Bf16*> output_weight(world), gate_up_weight(world), gate_up(world);
  std::vector<Bf16*> mlp_middle(world), down_weight(world);
  std::vector<uint32_t*> qkv_ready(world), qkv_consumed_epoch(world);
  std::vector<uint32_t*> pv_ready(world), v_ready(world);
  std::vector<fuse::GemmA2AParams> qkv_params(world), output_route(world);
  std::vector<fuse::A2AGemmParams> pv_params(world);
  std::vector<cudaEvent_t> qkv_done(world), route_done(world);

  for (int rank = 0; rank < world; ++rank) {
    state[rank] = allocate<Bf16>(rank, hidden_elements);
    norm[rank] = allocate<Bf16>(rank, hidden_elements);
    residual[rank] = allocate<Bf16>(rank, hidden_elements);
    projection[rank] = allocate<Bf16>(rank, hidden_elements);
    attention_norm_weight[rank] = allocate<Bf16>(rank, hidden);
    mlp_norm_weight[rank] = allocate<Bf16>(rank, hidden);
    qkv_weight[rank] = allocate<Bf16>(rank, qkv_weight_elements);
    qkv_output[rank] = allocate<Bf16>(rank, qkv_output_elements);
    qk_output[rank] = allocate<Bf16>(rank, qk_output_elements);
    query[rank] = allocate<Bf16>(rank, query_elements);
    key[rank] = allocate<Bf16>(rank, key_elements);
    probability[rank] = allocate<Bf16>(rank, probability_elements);
    value_staging[rank] = allocate<Bf16>(rank, value_staging_elements);
    pv_output[rank] = allocate<Bf16>(rank, pv_output_elements);
    attention_tokens[rank] = allocate<Bf16>(rank, hidden_elements);
    output_weight[rank] = allocate<Bf16>(rank, output_weight_elements);
    gate_up_weight[rank] = allocate<Bf16>(rank, gate_up_weight_elements);
    gate_up[rank] = allocate<Bf16>(rank, gate_up_elements);
    mlp_middle[rank] = allocate<Bf16>(rank, mlp_elements);
    down_weight[rank] = allocate<Bf16>(rank, down_weight_elements);
    qkv_ready[rank] = allocate<uint32_t>(rank, qkv_ready_count);
    qkv_consumed_epoch[rank] = allocate<uint32_t>(
        rank, world * fuse::kReadyFlagStride);
    pv_ready[rank] = allocate<uint32_t>(rank, pv_ready_count);
    v_ready[rank] = allocate<uint32_t>(rank, 1);
    fill(rank, runtime[rank].stream, state[rank], hidden_elements, 5101 + rank);
    fill_norm_weight(
        rank, runtime[rank].stream, attention_norm_weight[rank], hidden, 5601);
    fill_norm_weight(
        rank, runtime[rank].stream, mlp_norm_weight[rank], hidden, 5701);
    fill_weight(
        rank, runtime[rank].stream, qkv_weight[rank],
        qkv_weight_elements, 5201);
    fill_weight(
        rank, runtime[rank].stream, output_weight[rank],
        output_weight_elements, 5301);
    fill_weight(
        rank, runtime[rank].stream, gate_up_weight[rank],
        gate_up_weight_elements, 5401);
    fill_weight(
        rank, runtime[rank].stream, down_weight[rank],
        down_weight_elements, 5501);
    CUDA_CHECK(cudaMemsetAsync(
        qkv_ready[rank], 0, qkv_ready_count * sizeof(uint32_t), runtime[rank].stream));
    CUDA_CHECK(cudaMemsetAsync(
        qkv_consumed_epoch[rank],
        0,
        world * fuse::kReadyFlagStride * sizeof(uint32_t),
        runtime[rank].stream));
    CUDA_CHECK(cudaMemsetAsync(
        pv_ready[rank], 0, pv_ready_count * sizeof(uint32_t), runtime[rank].stream));
    CUDA_CHECK(cudaMemsetAsync(v_ready[rank], 0, sizeof(uint32_t), runtime[rank].stream));
    CUDA_CHECK(cudaEventCreateWithFlags(&qkv_done[rank], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&route_done[rank], cudaEventDisableTiming));
  }
  synchronize(runtime);

  for (int rank = 0; rank < world; ++rank) {
    auto& qkv = qkv_params[rank];
    qkv.lhs = norm[rank];
    qkv.rhs_nt = qkv_weight[rank];
    qkv.local_output = qkv_output[rank];
    for (int peer = 0; peer < world; ++peer) {
      qkv.peer_output[peer] = qk_output[peer];
      qkv.peer_route_done_epoch[peer] = qkv_consumed_epoch[peer];
    }
    qkv.ready = qkv_ready[rank];
    qkv.completion_epoch = v_ready[rank];
    qkv.gemm = {qkv_m, qkv_n, hidden, 1};
    qkv.gemm.raster = fuse::GemmRaster::kAlongM;
    qkv.route.world_size = world;
    qkv.route.rank = rank;
    qkv.route.batch = options.batch;
    qkv.route.seq_local = seq_local;
    qkv.route.global_seq = global_seq;
    qkv.route.q_heads = options.q_heads;
    qkv.route.kv_heads = options.kv_heads;
    qkv.route.head_dim = options.head_dim;
    qkv.route.qkv_peer_interleaved = true;
    qkv.route.kind = fuse::RouteKind::kQkvGqaPack;
    qkv.route.defer_v_a2a = true;
    qkv.route.direction = fuse::RouteDirection::kForward;
    qkv.num_comm_ctas = options.qkv_comm_ctas;

    auto& pv = pv_params[rank];
    pv.lhs = probability[rank];
    for (int peer = 0; peer < world; ++peer) {
      pv.peer_rhs[peer] = qkv_output[peer] + v_offset;
      pv.peer_input_ready[peer] = v_ready[peer];
    }
    pv.peer_rhs_row_stride = qkv_n;
    pv.rhs_nt = value_staging[rank];
    pv.output = pv_output[rank];
    pv.ready = pv_ready[rank];
    pv.gemm = {global_seq, options.head_dim, global_seq, pv_l};
    pv.gemm.stride_b.batch = 0;
    pv.gemm.raster = fuse::GemmRaster::kAlongM;
    pv.layout = pv_ready_layout;
    pv.layout.rank = rank;
    pv.num_comm_ctas = options.pv_comm_ctas;

    auto& route = output_route[rank];
    route.lhs = pv_output[rank];
    route.rhs_nt = pv_output[rank];
    route.local_output = pv_output[rank];
    for (int peer = 0; peer < world; ++peer) {
      route.peer_output[peer] = attention_tokens[peer];
    }
    route.ready = qkv_ready[rank];
    route.gemm = {global_seq, options.head_dim, options.head_dim, pv_l};
    route.gemm.raster = fuse::GemmRaster::kAlongM;
    route.route.world_size = world;
    route.route.rank = rank;
    route.route.batch = options.batch;
    route.route.seq_local = seq_local;
    route.route.global_seq = global_seq;
    route.route.q_heads = options.q_heads;
    route.route.kv_heads = options.kv_heads;
    route.route.local_heads = local_q_heads;
    route.route.head_dim = options.head_dim;
    route.route.kind = fuse::RouteKind::kHeadToSequence;
    route.route.direction = fuse::RouteDirection::kForward;
    route.num_comm_ctas = options.output_comm_ctas;
    route.epoch = 1;
  }

  auto qk_plan = autotune_cublaslt_bf16(
      runtime[0], global_seq, global_seq, options.head_dim, pv_l,
      query[0], key[0], probability[0], 0);
  auto output_plan = autotune_cublaslt_bf16(
      runtime[0], qkv_m, hidden, hidden, 1,
      attention_tokens[0], output_weight[0], projection[0]);
  auto gate_up_plan = autotune_cublaslt_bf16(
      runtime[0], qkv_m, 2 * options.ffn_hidden, hidden, 1,
      norm[0], gate_up_weight[0], gate_up[0]);
  auto down_plan = autotune_cublaslt_bf16(
      runtime[0], qkv_m, hidden, options.ffn_hidden, 1,
      mlp_middle[0], down_weight[0], projection[0]);

  uint32_t epoch = 0;
  auto submit_forward = [&]() {
    for (int layer = 0; layer < options.layers; ++layer) {
      ++epoch;
      for (int rank = 0; rank < world; ++rank) {
        CUDA_CHECK(cudaSetDevice(rank));
        rmsnorm_kernel<<<qkv_m, 256, 0, runtime[rank].stream>>>(
            state[rank], attention_norm_weight[rank], norm[rank],
            qkv_m, hidden, 1.0e-5f);
        qkv_params[rank].epoch = epoch;
        CUDA_CHECK(fuse::launch_gemm_a2a_cutlass(
            qkv_params[rank], runtime[rank].stream));
        CUDA_CHECK(cudaEventRecord(qkv_done[rank], runtime[rank].stream));
      }
      for (int rank = 0; rank < world; ++rank) {
        CUDA_CHECK(cudaSetDevice(rank));
        for (int peer = 0; peer < world; ++peer) {
          CUDA_CHECK(cudaStreamWaitEvent(runtime[rank].stream, qkv_done[peer], 0));
        }
        const int64_t qk_elements = query_elements + key_elements;
        const int blocks = static_cast<int>(
            std::min<int64_t>((qk_elements + 255) / 256, 4096));
        unpack_qk_kernel<<<blocks, 256, 0, runtime[rank].stream>>>(
            qk_output[rank], query[rank], key[rank], global_seq,
            local_q_heads, options.head_dim);
        cublaslt_nt(
            runtime[rank], qk_plan, query[rank], key[rank], probability[rank]);
        causal_softmax_kernel<<<
            pv_l * global_seq, 256, 0, runtime[rank].stream>>>(
            probability[rank], global_seq, pv_l * global_seq,
            1.0f / std::sqrt(static_cast<float>(options.head_dim)));
        pv_params[rank].epoch = epoch;
        pv_params[rank].input_epoch = epoch;
        CUDA_CHECK(fuse::launch_a2a_gemm_cutlass(
            pv_params[rank], runtime[rank].stream));
        CUDA_CHECK(fuse::launch_gemm_a2a_copy_reference(
            output_route[rank], runtime[rank].stream));
        CUDA_CHECK(cudaEventRecord(route_done[rank], runtime[rank].stream));
      }
      for (int rank = 0; rank < world; ++rank) {
        CUDA_CHECK(cudaSetDevice(rank));
        for (int peer = 0; peer < world; ++peer) {
          CUDA_CHECK(cudaStreamWaitEvent(runtime[rank].stream, route_done[peer], 0));
        }
        cublaslt_nt(
            runtime[rank], output_plan,
            attention_tokens[rank], output_weight[rank], projection[rank]);
        residual_add_kernel<<<
            static_cast<int>(std::min<int64_t>((hidden_elements + 255) / 256, 4096)),
            256, 0, runtime[rank].stream>>>(
            state[rank], projection[rank], residual[rank], hidden_elements);
        rmsnorm_kernel<<<qkv_m, 256, 0, runtime[rank].stream>>>(
            residual[rank], mlp_norm_weight[rank], norm[rank],
            qkv_m, hidden, 1.0e-5f);
        cublaslt_nt(
            runtime[rank], gate_up_plan,
            norm[rank], gate_up_weight[rank], gate_up[rank]);
        swiglu_kernel<<<
            static_cast<int>(std::min<int64_t>((mlp_elements + 255) / 256, 4096)),
            256, 0, runtime[rank].stream>>>(
            gate_up[rank], mlp_middle[rank], mlp_elements, options.ffn_hidden);
        cublaslt_nt(
            runtime[rank], down_plan,
            mlp_middle[rank], down_weight[rank], projection[rank]);
        residual_add_kernel<<<
            static_cast<int>(std::min<int64_t>((hidden_elements + 255) / 256, 4096)),
            256, 0, runtime[rank].stream>>>(
            residual[rank], projection[rank], state[rank], hidden_elements);
      }
    }
  };

  for (int iteration = 0; iteration < options.warmup; ++iteration) {
    submit_forward();
    synchronize(runtime);
  }
  std::vector<float> samples;
  samples.reserve(options.iterations);
  for (int iteration = 0; iteration < options.iterations; ++iteration) {
    for (int rank = 0; rank < world; ++rank) {
      CUDA_CHECK(cudaSetDevice(rank));
      CUDA_CHECK(cudaEventRecord(runtime[rank].start, runtime[rank].stream));
    }
    submit_forward();
    for (int rank = 0; rank < world; ++rank) {
      CUDA_CHECK(cudaSetDevice(rank));
      CUDA_CHECK(cudaEventRecord(runtime[rank].stop, runtime[rank].stream));
    }
    float critical = 0.0f;
    for (int rank = 0; rank < world; ++rank) {
      CUDA_CHECK(cudaSetDevice(rank));
      CUDA_CHECK(cudaEventSynchronize(runtime[rank].stop));
      float elapsed = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(
          &elapsed, runtime[rank].start, runtime[rank].stop));
      critical = std::max(critical, elapsed);
    }
    samples.push_back(critical);
  }

  unsigned long long nonfinite_count = 0;
  for (int rank = 0; rank < world; ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    auto* device_count = allocate<unsigned long long>(rank, 1);
    CUDA_CHECK(cudaMemsetAsync(
        device_count, 0, sizeof(unsigned long long), runtime[rank].stream));
    const int blocks = static_cast<int>(
        std::min<int64_t>((hidden_elements + 255) / 256, 4096));
    count_nonfinite_kernel<<<blocks, 256, 0, runtime[rank].stream>>>(
        state[rank], hidden_elements, device_count);
    CUDA_CHECK(cudaGetLastError());
    unsigned long long rank_count = 0;
    CUDA_CHECK(cudaMemcpyAsync(
        &rank_count, device_count, sizeof(rank_count), cudaMemcpyDeviceToHost,
        runtime[rank].stream));
    CUDA_CHECK(cudaStreamSynchronize(runtime[rank].stream));
    nonfinite_count += rank_count;
    CUDA_CHECK(cudaFree(device_count));
  }

  const Summary result = summarize(samples);
  std::cout << "Dense 70B forward (current explicit-attention implementation)"
            << " layers=" << options.layers << " S=" << global_seq
            << " CP=" << world << "\n";
  std::cout << "mean=" << std::fixed << std::setprecision(4) << result.mean
            << " ms p50=" << result.p50 << " p95=" << result.p95
            << " per_layer=" << result.mean / options.layers << " ms"
            << " nonfinite_count=" << nonfinite_count << "\n";
  if (!options.json_out.empty()) {
    std::ofstream output(options.json_out);
    output << std::setprecision(10)
           << "{\n  \"scope\": \"connected dense forward using fused QKV/QK route, explicit causal QK-softmax, fused V A2A-PV, output A2A, output projection and SwiGLU MLP\",\n"
           << "  \"layers\": " << options.layers << ",\n"
           << "  \"world_size\": " << world << ",\n"
           << "  \"global_seq\": " << global_seq << ",\n"
           << "  \"hidden\": " << hidden << ",\n"
           << "  \"ffn_hidden\": " << options.ffn_hidden << ",\n"
           << "  \"warmup\": " << options.warmup << ",\n"
           << "  \"iterations\": " << options.iterations << ",\n"
           << "  \"nonfinite_count\": " << nonfinite_count << ",\n"
           << "  \"mean_ms\": " << result.mean << ",\n"
           << "  \"p50_ms\": " << result.p50 << ",\n"
           << "  \"p95_ms\": " << result.p95 << ",\n"
           << "  \"per_layer_mean_ms\": " << result.mean / options.layers << ",\n"
           << "  \"samples_ms\": [";
    for (size_t index = 0; index < samples.size(); ++index) {
      output << (index == 0 ? "" : ", ") << samples[index];
    }
    output << "]\n}\n";
  }

  destroy_cublaslt_plan(qk_plan);
  destroy_cublaslt_plan(output_plan);
  destroy_cublaslt_plan(gate_up_plan);
  destroy_cublaslt_plan(down_plan);
}

void validate_fast_path(const Options& options, int world) {
  const auto traits = fuse::cutlass_kernel_traits();
  const bool qkv = options.mode == "qkv_gemm_a2a" ||
      options.mode == "qkv_gemm_a2a_fp8";
  const bool lhs_input = options.mode == "a2a_gemm_lhs";
  const bool e2e = options.mode == "ulysses_forward" ||
      options.mode == "dense_forward";
  const int n = qkv
      ? (options.q_heads + 2 * options.kv_heads) * options.head_dim
      : (e2e ? options.head_dim : options.n);
  const bool route_shape = qkv
      ? options.m % options.batch == 0 &&
          options.q_heads % options.kv_heads == 0 &&
          options.q_heads % world == 0 && options.kv_heads % world == 0
      : lhs_input
      ? options.m % options.batch == 0 &&
          (options.m / options.batch) % 2 == 0 &&
          options.q_heads % world == 0 &&
          options.k == options.q_heads * options.head_dim
      : options.k % world == 0 && options.m % world == 0 &&
          (!e2e ||
           (static_cast<int64_t>(options.batch) * options.m) % world == 0);
  if (options.m % traits.block_m != 0 || n % traits.block_n != 0 ||
      options.k % traits.block_k != 0 || !route_shape) {
    throw std::runtime_error("fast benchmark requires tiled M/N/K and world divisibility");
  }
  int device = 0;
  int sm_count = 0;
  CUDA_CHECK(cudaSetDevice(device));
  CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
  if (options.comm_ctas >= sm_count) {
    throw std::runtime_error("comm CTAs must leave at least one compute CTA");
  }
  const bool wide_qkv = e2e &&
      static_cast<int64_t>(options.batch) * options.m / world >= 2048;
  if (e2e &&
      (options.qkv_comm_ctas >= sm_count || options.pv_comm_ctas >= sm_count ||
       (wide_qkv && options.qkv_comm_ctas % 2 != 0))) {
    throw std::runtime_error(
        "ulysses_forward requires both roles below SM count and an even QKV "
        "comm count for the wide cluster-2 policy");
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    if (setpriority(PRIO_PROCESS, 0, -20) != 0) {
      std::cerr << "note: CPU nice=-20 unavailable; CUDA streams still use highest priority\n";
    }
    int world = 0;
    CUDA_CHECK(cudaGetDeviceCount(&world));
    world = std::min(world, fuse::kMaxWorldSize);
    if (world != 2 && world != 4 && world != 8) {
      throw std::runtime_error("fuse_bench requires 2, 4, or 8 visible GPUs");
    }
    validate_fast_path(options, world);
    enable_p2p(world);
    const auto runtime = initialize_runtime(world);
    if (options.mode == "a2a_gemm") {
      benchmark_a2a_gemm(options, world, runtime);
    } else if (options.mode == "a2a_gemm_lhs") {
      benchmark_a2a_lhs_gemm(options, world, runtime);
    } else if (options.mode == "qkv_gemm_a2a") {
      benchmark_qkv_gemm_a2a(options, world, runtime);
    } else if (options.mode == "qkv_gemm_a2a_fp8") {
      benchmark_fp8_qkv_gemm_a2a(options, world, runtime);
    } else if (options.mode == "ulysses_forward") {
      benchmark_ulysses_forward_boundaries(options, world, runtime);
    } else if (options.mode == "dense_forward") {
      benchmark_dense_forward(options, world, runtime);
    } else {
      throw std::runtime_error(
          "--mode must be a2a_gemm, a2a_gemm_lhs, qkv_gemm_a2a, "
          "qkv_gemm_a2a_fp8, ulysses_forward, or dense_forward");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "fuse_bench: " << error.what() << "\n";
    return 1;
  }
}
