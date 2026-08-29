// SPDX-License-Identifier: BSD-3-Clause

#include "fuse/fuse.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    const cudaError_t status_ = (call);                                         \
    if (status_ != cudaSuccess) {                                               \
      std::fprintf(                                                             \
          stderr, "%s:%d: %s\n", __FILE__, __LINE__,                         \
          cudaGetErrorString(status_));                                         \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

namespace {

struct Options {
  int world = 4;
  int local_rows = 2048;
  int warmup = 5;
  int iterations = 20;
  int row_quantum = 256;
  double slow_ratio = 1500.0 / 1980.0;
  double slow_hbm_ratio = 1.0;
  double slow_nvlink_ratio = 1.0;
  double baseline_nvlink_bidirectional_gbps = 900.0;
  double alpha = 1.0;
  std::vector<int> slow_ranks;
  std::string operation = "both";
  int qkv_fast_comm = 0;
  int qkv_slow_comm = 0;
  int oproj_fast_comm = 0;
  int oproj_slow_comm = 0;
  bool check = true;
  bool calibrate = false;
  bool auto_plan = false;
  bool allow_long_qkv = false;
};

bool parse_int(const char* text, int* value) {
  char* end = nullptr;
  const long parsed = std::strtol(text, &end, 10);
  if (end == text || *end != '\0') {
    return false;
  }
  *value = static_cast<int>(parsed);
  return true;
}

bool parse_double(const char* text, double* value) {
  char* end = nullptr;
  const double parsed = std::strtod(text, &end);
  if (end == text || *end != '\0' || !std::isfinite(parsed)) {
    return false;
  }
  *value = parsed;
  return true;
}

std::vector<int> parse_rank_list(const std::string& text) {
  std::vector<int> ranks;
  if (text.empty() || text == "none" || text == "-") {
    return ranks;
  }
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    int rank = -1;
    if (!parse_int(item.c_str(), &rank)) {
      std::fprintf(stderr, "invalid rank list: %s\n", text.c_str());
      std::exit(2);
    }
    ranks.push_back(rank);
  }
  std::sort(ranks.begin(), ranks.end());
  ranks.erase(std::unique(ranks.begin(), ranks.end()), ranks.end());
  return ranks;
}

void usage(const char* program) {
  std::fprintf(
      stderr,
      "usage: %s --world P --m LOCAL_ROWS --slow-ranks LIST "
      "[--slow-ratio R] [--slow-hbm-ratio R] "
      "[--slow-nvlink-ratio R] [--nvlink-bidir-gbps G] "
      "[--alpha A | --auto-plan] [--warmup N] [--iterations N] "
      "[--operation both|qkv|oproj] [--qkv-fast-comm N] "
      "[--qkv-slow-comm N] [--oproj-fast-comm N] "
      "[--oproj-slow-comm N] [--allow-long-qkv] [--no-check]\n",
      program);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index];
    auto take = [&]() -> const char* {
      if (++index >= argc) {
        usage(argv[0]);
        std::exit(2);
      }
      return argv[index];
    };
    if (arg == "--world") {
      if (!parse_int(take(), &options.world)) std::exit(2);
    } else if (arg == "--m") {
      if (!parse_int(take(), &options.local_rows)) std::exit(2);
    } else if (arg == "--slow-ranks") {
      options.slow_ranks = parse_rank_list(take());
    } else if (arg == "--slow-ratio") {
      if (!parse_double(take(), &options.slow_ratio)) std::exit(2);
    } else if (arg == "--slow-hbm-ratio") {
      if (!parse_double(take(), &options.slow_hbm_ratio)) std::exit(2);
    } else if (arg == "--slow-nvlink-ratio") {
      if (!parse_double(take(), &options.slow_nvlink_ratio)) std::exit(2);
    } else if (arg == "--nvlink-bidir-gbps") {
      if (!parse_double(
              take(), &options.baseline_nvlink_bidirectional_gbps)) {
        std::exit(2);
      }
    } else if (arg == "--alpha") {
      if (!parse_double(take(), &options.alpha)) std::exit(2);
    } else if (arg == "--warmup") {
      if (!parse_int(take(), &options.warmup)) std::exit(2);
    } else if (arg == "--iterations") {
      if (!parse_int(take(), &options.iterations)) std::exit(2);
    } else if (arg == "--operation") {
      options.operation = take();
    } else if (arg == "--qkv-fast-comm") {
      if (!parse_int(take(), &options.qkv_fast_comm)) std::exit(2);
    } else if (arg == "--qkv-slow-comm") {
      if (!parse_int(take(), &options.qkv_slow_comm)) std::exit(2);
    } else if (arg == "--oproj-fast-comm") {
      if (!parse_int(take(), &options.oproj_fast_comm)) std::exit(2);
    } else if (arg == "--oproj-slow-comm") {
      if (!parse_int(take(), &options.oproj_slow_comm)) std::exit(2);
    } else if (arg == "--no-check") {
      options.check = false;
    } else if (arg == "--calibrate") {
      options.calibrate = true;
    } else if (arg == "--auto-plan") {
      options.auto_plan = true;
    } else if (arg == "--allow-long-qkv") {
      options.allow_long_qkv = true;
    } else {
      usage(argv[0]);
      std::exit(2);
    }
  }
  if (options.world < 2 || options.world > fuse::kMaxWorldSize ||
      options.local_rows <= 0 || options.local_rows % options.row_quantum != 0 ||
      options.warmup < 0 || options.iterations <= 0 ||
      !(options.slow_ratio > 0.0 && options.slow_ratio <= 1.0) ||
      !(options.slow_hbm_ratio > 0.0 && options.slow_hbm_ratio <= 1.0) ||
      !(options.slow_nvlink_ratio > 0.0 &&
        options.slow_nvlink_ratio <= 1.0) ||
      !(options.baseline_nvlink_bidirectional_gbps > 0.0) ||
      !(options.alpha >= 0.0 && options.alpha <= 1.0) ||
      (options.operation != "both" && options.operation != "qkv" &&
       options.operation != "oproj")) {
    usage(argv[0]);
    std::exit(2);
  }
  for (int rank : options.slow_ranks) {
    if (rank < 0 || rank >= options.world) {
      std::fprintf(stderr, "slow rank %d is outside world=%d\n", rank, options.world);
      std::exit(2);
    }
  }
  return options;
}

bool is_slow(const Options& options, int rank) {
  return std::binary_search(
      options.slow_ranks.begin(), options.slow_ranks.end(), rank);
}

struct Partition {
  std::vector<int> rows;
  std::vector<int> begin;
};

Partition make_partition(const Options& options) {
  const int total_units =
      options.world * options.local_rows / options.row_quantum;
  const double capacity_sum =
      (options.world - static_cast<int>(options.slow_ranks.size())) +
      options.slow_ranks.size() * options.slow_ratio;
  std::vector<int> units(options.world, 0);
  std::vector<double> remainder(options.world, 0.0);
  int assigned = 0;
  for (int rank = 0; rank < options.world; ++rank) {
    const double capacity = is_slow(options, rank) ? options.slow_ratio : 1.0;
    const double equal_units = static_cast<double>(total_units) / options.world;
    const double proportional_units = total_units * capacity / capacity_sum;
    const double target =
        (1.0 - options.alpha) * equal_units +
        options.alpha * proportional_units;
    units[rank] = static_cast<int>(std::floor(target));
    remainder[rank] = target - units[rank];
    assigned += units[rank];
  }
  while (assigned < total_units) {
    int best = 0;
    for (int rank = 1; rank < options.world; ++rank) {
      if (remainder[rank] > remainder[best] ||
          (remainder[rank] == remainder[best] && rank < best)) {
        best = rank;
      }
    }
    ++units[best];
    remainder[best] = -1.0;
    ++assigned;
  }
  Partition partition;
  partition.rows.resize(options.world);
  partition.begin.resize(options.world);
  int cursor = 0;
  for (int rank = 0; rank < options.world; ++rank) {
    if (units[rank] <= 0) {
      std::fprintf(stderr, "partition produced an empty rank\n");
      std::exit(2);
    }
    partition.begin[rank] = cursor;
    partition.rows[rank] = units[rank] * options.row_quantum;
    cursor += partition.rows[rank];
  }
  if (cursor != options.world * options.local_rows) {
    std::fprintf(stderr, "partition does not cover the global sequence\n");
    std::exit(2);
  }
  return partition;
}

fuse::WeightedCpPlannerOptions make_planner_options(const Options& options) {
  fuse::WeightedCpPlannerOptions planner{};
  planner.world_size = options.world;
  planner.uniform_local_rows = options.local_rows;
  planner.row_quantum = options.row_quantum;
  planner.sm_count = 132;
  planner.baseline_nvlink_bidirectional_gbps =
      options.baseline_nvlink_bidirectional_gbps;
  planner.allow_long_qkv_redistribution = options.allow_long_qkv;
  for (int rank = 0; rank < options.world; ++rank) {
    if (is_slow(options, rank)) {
      planner.rank[rank].sm = options.slow_ratio;
      planner.rank[rank].hbm = options.slow_hbm_ratio;
      planner.rank[rank].nvlink = options.slow_nvlink_ratio;
    }
  }
  return planner;
}

Partition partition_from_plan(const fuse::WeightedCpPlan& plan) {
  Partition partition;
  partition.rows.resize(plan.world_size);
  partition.begin.resize(plan.world_size);
  for (int rank = 0; rank < plan.world_size; ++rank) {
    partition.rows[rank] = plan.rank[rank].rows;
    partition.begin[rank] = plan.rank[rank].global_sequence_begin;
  }
  return partition;
}

void print_plan(const char* operation, const fuse::WeightedCpPlan& plan) {
  std::printf(
      "PLAN op=%s weighted=%d predicted_speedup=%.6f "
      "uniform_us=%.6f weighted_us=%.6f redistributed_rows=%lld "
      "equivalent_alpha=%.6f uniform_bottleneck=%d "
      "weighted_bottleneck=%d\n",
      operation,
      plan.weighted ? 1 : 0,
      plan.predicted_speedup,
      plan.uniform_critical_us,
      plan.weighted_critical_us,
      static_cast<long long>(plan.redistributed_rows),
      plan.equivalent_alpha,
      plan.uniform_bottleneck_rank,
      plan.weighted_bottleneck_rank);
  for (int rank = 0; rank < plan.world_size; ++rank) {
    const auto& decision = plan.rank[rank];
    std::printf(
        "PLAN_RANK op=%s rank=%d rows=%d begin=%d comm=%d "
        "tile=%dx%d cluster_m=%d waves=%d compute_us=%.6f "
        "route_us=%.6f critical_us=%.6f\n",
        operation,
        rank,
        decision.rows,
        decision.global_sequence_begin,
        decision.comm_ctas,
        decision.tile_m,
        decision.tile_n,
        decision.cluster_m,
        decision.waves,
        decision.compute_us,
        decision.route_us,
        decision.critical_us);
  }
}

fuse::WeightedCpPlan make_qkv_plan(const Options& options) {
  constexpr int q_heads = 24;
  constexpr int kv_heads = 24;
  constexpr int head_dim = 128;
  fuse::GemmProblem problem{};
  problem.m = options.local_rows;
  problem.n = (q_heads + 2 * kv_heads) * head_dim;
  problem.k = 4096;
  problem.raster = fuse::GemmRaster::kAlongN;
  fuse::UlyssesRoute route{};
  route.world_size = options.world;
  route.batch = 1;
  route.global_seq = options.world * options.local_rows;
  route.seq_local = options.local_rows;
  route.q_heads = q_heads;
  route.kv_heads = kv_heads;
  route.head_dim = head_dim;
  route.kind = fuse::RouteKind::kQkvGqaPack;
  route.direction = fuse::RouteDirection::kForward;
  fuse::WeightedCpPlan plan{};
  CUDA_CHECK(fuse::plan_weighted_gemm_a2a(
      problem, route, make_planner_options(options), &plan));
  return plan;
}

fuse::WeightedCpPlan make_oproj_plan(const Options& options) {
  constexpr int heads = 24;
  constexpr int head_dim = 128;
  fuse::GemmProblem problem{};
  problem.m = options.local_rows;
  problem.n = 4096;
  problem.k = heads * head_dim;
  problem.raster = fuse::GemmRaster::kAlongN;
  fuse::UlyssesRoute route{};
  route.world_size = options.world;
  route.batch = 1;
  route.global_seq = options.world * options.local_rows;
  route.seq_local = options.local_rows;
  route.q_heads = heads;
  route.local_heads = heads / options.world;
  route.head_dim = head_dim;
  route.kind = fuse::RouteKind::kHeadToSequence;
  route.direction = fuse::RouteDirection::kInverse;
  fuse::WeightedCpPlan plan{};
  CUDA_CHECK(fuse::plan_weighted_a2a_gemm(
      problem, route, make_planner_options(options), &plan));
  return plan;
}

class HostBarrier {
 public:
  explicit HostBarrier(int participants) : participants_(participants) {}
  void wait() {
    const int generation = generation_.load(std::memory_order_acquire);
    if (arrived_.fetch_add(1, std::memory_order_acq_rel) == participants_ - 1) {
      arrived_.store(0, std::memory_order_relaxed);
      generation_.fetch_add(1, std::memory_order_release);
      return;
    }
    while (generation_.load(std::memory_order_acquire) == generation) {
      std::this_thread::yield();
    }
  }

 private:
  int participants_;
  std::atomic<int> arrived_{0};
  std::atomic<int> generation_{0};
};

struct Timing {
  std::vector<float> critical_ms;
  std::vector<std::vector<float>> rank_ms;
};

using Launch = std::function<cudaError_t(int, uint32_t)>;

Timing time_ranks(
    int world,
    const std::vector<cudaStream_t>& streams,
    int warmup,
    int iterations,
    uint32_t first_epoch,
    const Launch& launch) {
  const int total = warmup + iterations;
  HostBarrier begin(world);
  HostBarrier finish(world);
  std::vector<std::vector<float>> samples(
      world, std::vector<float>(total, 0.0f));
  std::vector<std::thread> workers;
  for (int rank = 0; rank < world; ++rank) {
    workers.emplace_back([&, rank] {
      CUDA_CHECK(cudaSetDevice(rank));
      cudaEvent_t start = nullptr;
      cudaEvent_t stop = nullptr;
      CUDA_CHECK(cudaEventCreate(&start));
      CUDA_CHECK(cudaEventCreate(&stop));
      for (int step = 0; step < total; ++step) {
        begin.wait();
        CUDA_CHECK(cudaEventRecord(start, streams[rank]));
        CUDA_CHECK(launch(rank, first_epoch + step));
        CUDA_CHECK(cudaEventRecord(stop, streams[rank]));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&samples[rank][step], start, stop));
        finish.wait();
      }
      CUDA_CHECK(cudaEventDestroy(stop));
      CUDA_CHECK(cudaEventDestroy(start));
    });
  }
  for (auto& worker : workers) worker.join();

  Timing timing;
  timing.rank_ms.resize(world);
  for (int rank = 0; rank < world; ++rank) {
    timing.rank_ms[rank].assign(
        samples[rank].begin() + warmup, samples[rank].end());
  }
  for (int step = warmup; step < total; ++step) {
    float critical = 0.0f;
    for (int rank = 0; rank < world; ++rank) {
      critical = std::max(critical, samples[rank][step]);
    }
    timing.critical_ms.push_back(critical);
  }
  return timing;
}

float percentile(std::vector<float> values, double fraction) {
  std::sort(values.begin(), values.end());
  const double index = fraction * (values.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(index));
  const size_t hi = static_cast<size_t>(std::ceil(index));
  const double weight = index - lo;
  return static_cast<float>(values[lo] * (1.0 - weight) + values[hi] * weight);
}

__global__ void fill_rows(
    fuse::Bf16* data,
    int64_t elements,
    int columns,
    int global_row_begin,
    int seed) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < elements;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = index / columns + global_row_begin;
    const int64_t column = index % columns;
    const int value = static_cast<int>(
        (row * 17 + column * 13 + static_cast<int64_t>(seed) * 7) % 31) - 15;
    data[index] = fuse::Bf16(static_cast<float>(value) / 64.0f);
  }
}

template <class T>
T* allocate(int device, int64_t elements) {
  CUDA_CHECK(cudaSetDevice(device));
  T* pointer = nullptr;
  CUDA_CHECK(cudaMalloc(&pointer, static_cast<size_t>(elements) * sizeof(T)));
  return pointer;
}

void fill(
    int device,
    cudaStream_t stream,
    fuse::Bf16* data,
    int rows,
    int columns,
    int global_row_begin,
    int seed) {
  CUDA_CHECK(cudaSetDevice(device));
  const int64_t elements = static_cast<int64_t>(rows) * columns;
  fill_rows<<<std::min<int64_t>(4096, (elements + 255) / 256), 256, 0, stream>>>(
      data, elements, columns, global_row_begin, seed);
  CUDA_CHECK(cudaGetLastError());
}

void enable_peer_access(int world) {
  for (int device = 0; device < world; ++device) {
    CUDA_CHECK(cudaSetDevice(device));
    for (int peer = 0; peer < world; ++peer) {
      if (peer == device) continue;
      int supported = 0;
      CUDA_CHECK(cudaDeviceCanAccessPeer(&supported, device, peer));
      if (!supported) {
        std::fprintf(stderr, "GPU %d cannot access GPU %d\n", device, peer);
        std::exit(1);
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

void synchronize(int world, const std::vector<cudaStream_t>& streams) {
  for (int rank = 0; rank < world; ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaStreamSynchronize(streams[rank]));
  }
}

void compare_exact(
    int device,
    const fuse::Bf16* expected,
    const fuse::Bf16* actual,
    int64_t elements,
    const char* label) {
  CUDA_CHECK(cudaSetDevice(device));
  std::vector<uint16_t> lhs(elements);
  std::vector<uint16_t> rhs(elements);
  CUDA_CHECK(cudaMemcpy(
      lhs.data(), expected, elements * sizeof(uint16_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(
      rhs.data(), actual, elements * sizeof(uint16_t), cudaMemcpyDeviceToHost));
  int64_t mismatches = 0;
  for (int64_t index = 0; index < elements; ++index) {
    mismatches += lhs[index] != rhs[index];
  }
  if (mismatches != 0) {
    std::fprintf(
        stderr, "%s rank %d: %lld/%lld values differ\n", label, device,
        static_cast<long long>(mismatches), static_cast<long long>(elements));
    std::exit(1);
  }
}

int qkv_comm(const Options& options, int rank, int fallback) {
  const int requested = is_slow(options, rank)
      ? options.qkv_slow_comm
      : options.qkv_fast_comm;
  return requested > 0 ? requested : fallback;
}

int oproj_comm(const Options& options, int rank, int fallback) {
  const int requested = is_slow(options, rank)
      ? options.oproj_slow_comm
      : options.oproj_fast_comm;
  return requested > 0 ? requested : fallback;
}

void print_result(
    const char* operation,
    const char* policy,
    const char* implementation,
    const Options& options,
    const Partition& partition,
    const Timing& timing) {
  const float p50 = percentile(timing.critical_ms, 0.50);
  const float p95 = percentile(timing.critical_ms, 0.95);
  const float mean = std::accumulate(
      timing.critical_ms.begin(), timing.critical_ms.end(), 0.0f) /
      timing.critical_ms.size();
  std::printf(
      "RESULT op=%s policy=%s implementation=%s world=%d slow=%zu "
      "ratio=%.6f alpha=%.3f "
      "m=%d mean_ms=%.6f p50_ms=%.6f p95_ms=%.6f rows=",
      operation, policy, implementation, options.world,
      options.slow_ranks.size(),
      options.slow_ratio, options.alpha, options.local_rows, mean, p50, p95);
  for (int rank = 0; rank < options.world; ++rank) {
    std::printf("%s%d", rank == 0 ? "" : ",", partition.rows[rank]);
  }
  std::printf(" rank_p50=");
  for (int rank = 0; rank < options.world; ++rank) {
    std::printf(
        "%s%.6f", rank == 0 ? "" : ",",
        percentile(timing.rank_ms[rank], 0.50));
  }
  std::printf("\n");
}

void print_calibration(
    const char* operation,
    const char* stage,
    const Options& options,
    const Timing& timing) {
  std::printf(
      "CALIBRATION op=%s stage=%s world=%d slow=%zu m=%d critical_p50_ms=%.6f "
      "rank_p50=",
      operation,
      stage,
      options.world,
      options.slow_ranks.size(),
      options.local_rows,
      percentile(timing.critical_ms, 0.50));
  for (int rank = 0; rank < options.world; ++rank) {
    std::printf(
        "%s%.6f",
        rank == 0 ? "" : ",",
        percentile(timing.rank_ms[rank], 0.50));
  }
  std::printf("\n");
}

void run_qkv(
    const Options& options,
    const Partition& partition,
    const std::vector<cudaStream_t>& streams,
    const fuse::WeightedCpPlan* planner = nullptr) {
  const bool fallback_uniform = planner != nullptr && !planner->weighted;
  // This artificial geometry is deliberately divisible by CP2/4/6/8 so the
  // same mathematical problem can isolate CP degree and slow-rank count.
  constexpr int q_heads = 24;
  constexpr int kv_heads = 24;
  constexpr int head_dim = 128;
  constexpr int k = 4096;
  constexpr int n = (q_heads + 2 * kv_heads) * head_dim;
  const int global_rows = options.world * options.local_rows;
  const int local_width =
      (q_heads / options.world + 2 * (kv_heads / options.world)) * head_dim;
  const int64_t routed_elements =
      static_cast<int64_t>(global_rows) * local_width;
  const int64_t weight_elements = static_cast<int64_t>(n) * k;
  const int64_t done_elements =
      static_cast<int64_t>(options.world) * fuse::kReadyFlagStride;

  struct RankData {
    fuse::Bf16* uniform_lhs = nullptr;
    fuse::Bf16* weighted_lhs = nullptr;
    fuse::Bf16* weight = nullptr;
    fuse::Bf16* uniform_local = nullptr;
    fuse::Bf16* weighted_local = nullptr;
    fuse::Bf16* uniform_output = nullptr;
    fuse::Bf16* weighted_output = nullptr;
    uint32_t* uniform_ready = nullptr;
    uint32_t* weighted_ready = nullptr;
    uint32_t* uniform_done = nullptr;
    uint32_t* weighted_done = nullptr;
    int uniform_comm = 0;
    int weighted_comm = 0;
  };
  std::vector<RankData> rank(options.world);

  for (int r = 0; r < options.world; ++r) {
    fuse::GemmProblem uniform_problem{};
    uniform_problem.m = options.local_rows;
    uniform_problem.n = n;
    uniform_problem.k = k;
    uniform_problem.raster = fuse::GemmRaster::kAlongN;
    fuse::UlyssesRoute uniform_route{};
    uniform_route.world_size = options.world;
    uniform_route.rank = r;
    uniform_route.batch = 1;
    uniform_route.global_seq = global_rows;
    uniform_route.seq_local = options.local_rows;
    uniform_route.q_heads = q_heads;
    uniform_route.kv_heads = kv_heads;
    uniform_route.head_dim = head_dim;
    uniform_route.kind = fuse::RouteKind::kQkvGqaPack;
    uniform_route.direction = fuse::RouteDirection::kForward;
    CUDA_CHECK(cudaSetDevice(r));
    const int uniform_auto =
        fuse::recommended_gemm_a2a_comm_ctas(uniform_problem, uniform_route);
    rank[r].uniform_comm = uniform_auto;

    auto weighted_problem = uniform_problem;
    weighted_problem.m = partition.rows[r];
    auto weighted_route = uniform_route;
    weighted_route.seq_local = partition.rows[r];
    const int weighted_auto =
        fuse::recommended_gemm_a2a_comm_ctas(weighted_problem, weighted_route);
    rank[r].weighted_comm = fallback_uniform
        ? rank[r].uniform_comm
        : (planner != nullptr
               ? planner->rank[r].comm_ctas
               : qkv_comm(options, r, weighted_auto));

    const int64_t uniform_ready_elements =
        static_cast<int64_t>((options.local_rows + 127) / 128) *
        ((n + 63) / 64) * fuse::kReadyFlagStride;
    const int64_t weighted_ready_elements =
        static_cast<int64_t>((partition.rows[r] + 127) / 128) *
        ((n + 63) / 64) * fuse::kReadyFlagStride;
    rank[r].uniform_lhs = allocate<fuse::Bf16>(
        r, static_cast<int64_t>(options.local_rows) * k);
    rank[r].weighted_lhs = allocate<fuse::Bf16>(
        r, static_cast<int64_t>(partition.rows[r]) * k);
    rank[r].weight = allocate<fuse::Bf16>(r, weight_elements);
    rank[r].uniform_local = allocate<fuse::Bf16>(
        r, static_cast<int64_t>(options.local_rows) * n);
    rank[r].weighted_local = allocate<fuse::Bf16>(
        r, static_cast<int64_t>(partition.rows[r]) * n);
    rank[r].uniform_output = allocate<fuse::Bf16>(r, routed_elements);
    rank[r].weighted_output = allocate<fuse::Bf16>(r, routed_elements);
    rank[r].uniform_ready = allocate<uint32_t>(r, uniform_ready_elements);
    rank[r].weighted_ready = allocate<uint32_t>(r, weighted_ready_elements);
    rank[r].uniform_done = allocate<uint32_t>(r, done_elements);
    rank[r].weighted_done = allocate<uint32_t>(r, done_elements);
    fill(r, streams[r], rank[r].uniform_lhs, options.local_rows, k,
         r * options.local_rows, 101);
    fill(r, streams[r], rank[r].weighted_lhs, partition.rows[r], k,
         partition.begin[r], 101);
    fill(r, streams[r], rank[r].weight, n, k, 0, 7);
    CUDA_CHECK(cudaMemsetAsync(
        rank[r].uniform_output, 0, routed_elements * sizeof(fuse::Bf16), streams[r]));
    CUDA_CHECK(cudaMemsetAsync(
        rank[r].weighted_output, 0, routed_elements * sizeof(fuse::Bf16), streams[r]));
    CUDA_CHECK(cudaMemsetAsync(
        rank[r].uniform_ready, 0, uniform_ready_elements * sizeof(uint32_t), streams[r]));
    CUDA_CHECK(cudaMemsetAsync(
        rank[r].weighted_ready, 0, weighted_ready_elements * sizeof(uint32_t), streams[r]));
    CUDA_CHECK(cudaMemsetAsync(
        rank[r].uniform_done, 0, done_elements * sizeof(uint32_t), streams[r]));
    CUDA_CHECK(cudaMemsetAsync(
        rank[r].weighted_done, 0, done_elements * sizeof(uint32_t), streams[r]));
  }
  synchronize(options.world, streams);

  auto uniform_launch = [&](int r, uint32_t epoch) {
    fuse::GemmA2AParams params{};
    params.lhs = rank[r].uniform_lhs;
    params.rhs_nt = rank[r].weight;
    params.local_output = rank[r].uniform_local;
    for (int peer = 0; peer < options.world; ++peer) {
      params.peer_output[peer] = rank[peer].uniform_output;
      params.peer_route_done_epoch[peer] = rank[peer].uniform_done;
    }
    params.ready = rank[r].uniform_ready;
    params.gemm.m = options.local_rows;
    params.gemm.n = n;
    params.gemm.k = k;
    params.gemm.raster = fuse::GemmRaster::kAlongN;
    params.route.world_size = options.world;
    params.route.rank = r;
    params.route.batch = 1;
    params.route.global_seq = global_rows;
    params.route.seq_local = options.local_rows;
    params.route.q_heads = q_heads;
    params.route.kv_heads = kv_heads;
    params.route.head_dim = head_dim;
    params.route.kind = fuse::RouteKind::kQkvGqaPack;
    params.route.direction = fuse::RouteDirection::kForward;
    params.num_comm_ctas = rank[r].uniform_comm;
    params.epoch = epoch;
    return fuse::launch_gemm_a2a_cutlass(params, streams[r]);
  };
  auto weighted_launch = [&](int r, uint32_t epoch) {
    fuse::WeightedGemmA2AParams params{};
    params.lhs = rank[r].weighted_lhs;
    params.rhs_nt = rank[r].weight;
    params.local_output = rank[r].weighted_local;
    for (int peer = 0; peer < options.world; ++peer) {
      params.peer_output[peer] = rank[peer].weighted_output;
      params.peer_route_done_epoch[peer] = rank[peer].weighted_done;
    }
    params.ready = rank[r].weighted_ready;
    params.gemm.m = partition.rows[r];
    params.gemm.n = n;
    params.gemm.k = k;
    params.gemm.raster = fuse::GemmRaster::kAlongN;
    params.route.world_size = options.world;
    params.route.rank = r;
    params.route.batch = 1;
    params.route.global_seq = global_rows;
    params.route.seq_local = partition.rows[r];
    params.route.q_heads = q_heads;
    params.route.kv_heads = kv_heads;
    params.route.head_dim = head_dim;
    params.route.kind = fuse::RouteKind::kQkvGqaPack;
    params.route.direction = fuse::RouteDirection::kForward;
    params.global_sequence_begin = partition.begin[r];
    params.num_comm_ctas = rank[r].weighted_comm;
    params.epoch = epoch;
    return fuse::launch_weighted_gemm_a2a_cutlass(params, streams[r]);
  };

  const auto uniform_timing = time_ranks(
      options.world, streams, options.warmup, options.iterations, 1,
      uniform_launch);
  const auto weighted_timing = fallback_uniform
      ? uniform_timing
      : time_ranks(
            options.world, streams, options.warmup, options.iterations, 1,
            weighted_launch);
  if (options.check && !fallback_uniform) {
    for (int r = 0; r < options.world; ++r) {
      compare_exact(
          r, rank[r].uniform_output, rank[r].weighted_output,
          routed_elements, "weighted QKVProj+A2A");
    }
  }
  Partition equal = partition;
  std::fill(equal.rows.begin(), equal.rows.end(), options.local_rows);
  print_result(
      "qkv", "uniform", "uniform", options, equal, uniform_timing);
  print_result(
      "qkv",
      "weighted",
      fallback_uniform ? "uniform_fallback" : "weighted",
      options,
      partition,
      weighted_timing);
  std::printf(
      "SPEEDUP op=qkv value=%.6f comm_uniform=",
      percentile(uniform_timing.critical_ms, 0.50) /
          percentile(weighted_timing.critical_ms, 0.50));
  for (int r = 0; r < options.world; ++r) {
    std::printf("%s%d", r ? "," : "", rank[r].uniform_comm);
  }
  std::printf(" comm_weighted=");
  for (int r = 0; r < options.world; ++r) {
    std::printf("%s%d", r ? "," : "", rank[r].weighted_comm);
  }
  std::printf("\n");

  if (options.calibrate) {
    auto route_launch = [&](int r, uint32_t epoch) {
      fuse::GemmA2AParams params{};
      params.lhs = rank[r].uniform_lhs;
      params.rhs_nt = rank[r].weight;
      params.local_output = rank[r].uniform_local;
      for (int peer = 0; peer < options.world; ++peer) {
        params.peer_output[peer] = rank[peer].uniform_output;
        params.peer_route_done_epoch[peer] = rank[peer].uniform_done;
      }
      params.ready = rank[r].uniform_ready;
      params.gemm.m = options.local_rows;
      params.gemm.n = n;
      params.gemm.k = k;
      params.gemm.raster = fuse::GemmRaster::kAlongN;
      params.route.world_size = options.world;
      params.route.rank = r;
      params.route.batch = 1;
      params.route.global_seq = global_rows;
      params.route.seq_local = options.local_rows;
      params.route.q_heads = q_heads;
      params.route.kv_heads = kv_heads;
      params.route.head_dim = head_dim;
      params.route.kind = fuse::RouteKind::kQkvGqaPack;
      params.route.direction = fuse::RouteDirection::kForward;
      params.num_comm_ctas = rank[r].weighted_comm;
      params.epoch = epoch;
      return fuse::launch_gemm_a2a_copy_reference(params, streams[r]);
    };
    const auto route = time_ranks(
        options.world, streams, options.warmup, options.iterations, 2000,
        route_launch);
    print_calibration("qkv", "pure_route", options, route);
  }
}

void run_oproj(
    const Options& options,
    const Partition& partition,
    const std::vector<cudaStream_t>& streams,
    const fuse::WeightedCpPlan* planner = nullptr) {
  const bool fallback_uniform = planner != nullptr && !planner->weighted;
  constexpr int heads = 24;
  constexpr int head_dim = 128;
  constexpr int k = heads * head_dim;
  constexpr int n = 4096;
  const int global_rows = options.world * options.local_rows;
  const int shard_width = k / options.world;
  const int64_t input_elements =
      static_cast<int64_t>(global_rows) * shard_width;
  const int64_t weight_elements = static_cast<int64_t>(n) * k;
  const int64_t done_elements =
      static_cast<int64_t>(options.world) * fuse::kReadyFlagStride;

  struct RankData {
    fuse::Bf16* input = nullptr;
    fuse::Bf16* weight = nullptr;
    fuse::Bf16* uniform_staging = nullptr;
    fuse::Bf16* weighted_staging = nullptr;
    fuse::Bf16* uniform_output = nullptr;
    fuse::Bf16* weighted_output = nullptr;
    fuse::Bf16* expected_weighted = nullptr;
    uint32_t* uniform_ready = nullptr;
    uint32_t* weighted_ready = nullptr;
    uint32_t* weighted_done = nullptr;
    int uniform_comm = 0;
    int weighted_comm = 0;
  };
  std::vector<RankData> rank(options.world);

  for (int r = 0; r < options.world; ++r) {
    fuse::GemmProblem uniform_problem{};
    uniform_problem.m = options.local_rows;
    uniform_problem.n = n;
    uniform_problem.k = k;
    uniform_problem.raster = fuse::GemmRaster::kAlongN;
    fuse::UlyssesRoute uniform_route{};
    uniform_route.world_size = options.world;
    uniform_route.rank = r;
    uniform_route.batch = 1;
    uniform_route.global_seq = global_rows;
    uniform_route.seq_local = options.local_rows;
    uniform_route.q_heads = heads;
    uniform_route.local_heads = heads / options.world;
    uniform_route.head_dim = head_dim;
    uniform_route.kind = fuse::RouteKind::kHeadToSequence;
    uniform_route.direction = fuse::RouteDirection::kInverse;
    CUDA_CHECK(cudaSetDevice(r));
    const int uniform_auto =
        fuse::recommended_a2a_lhs_gemm_comm_ctas(uniform_problem, uniform_route);
    rank[r].uniform_comm = uniform_auto;

    auto weighted_problem = uniform_problem;
    weighted_problem.m = partition.rows[r];
    auto weighted_route = uniform_route;
    weighted_route.seq_local = partition.rows[r];
    const int weighted_auto =
        fuse::recommended_a2a_lhs_gemm_comm_ctas(weighted_problem, weighted_route);
    rank[r].weighted_comm = fallback_uniform
        ? rank[r].uniform_comm
        : (planner != nullptr
               ? planner->rank[r].comm_ctas
               : oproj_comm(options, r, weighted_auto));
    const int64_t uniform_ready_elements =
        fuse::a2a_lhs_gemm_ready_elements(uniform_problem, uniform_route);
    const int64_t weighted_ready_elements =
        fuse::a2a_lhs_gemm_ready_elements(weighted_problem, weighted_route);

    rank[r].input = allocate<fuse::Bf16>(r, input_elements);
    rank[r].weight = allocate<fuse::Bf16>(r, weight_elements);
    rank[r].uniform_staging = allocate<fuse::Bf16>(
        r, static_cast<int64_t>(options.local_rows) * k);
    rank[r].weighted_staging = allocate<fuse::Bf16>(
        r, static_cast<int64_t>(partition.rows[r]) * k);
    rank[r].uniform_output = allocate<fuse::Bf16>(
        r, static_cast<int64_t>(options.local_rows) * n);
    rank[r].weighted_output = allocate<fuse::Bf16>(
        r, static_cast<int64_t>(partition.rows[r]) * n);
    rank[r].expected_weighted = allocate<fuse::Bf16>(
        r, static_cast<int64_t>(partition.rows[r]) * n);
    rank[r].uniform_ready = allocate<uint32_t>(r, uniform_ready_elements);
    rank[r].weighted_ready = allocate<uint32_t>(r, weighted_ready_elements);
    rank[r].weighted_done = allocate<uint32_t>(r, done_elements);
    fill(r, streams[r], rank[r].input, global_rows, shard_width, 0, 401 + r);
    fill(r, streams[r], rank[r].weight, n, k, 0, 17);
    CUDA_CHECK(cudaMemsetAsync(
        rank[r].uniform_ready, 0, uniform_ready_elements * sizeof(uint32_t), streams[r]));
    CUDA_CHECK(cudaMemsetAsync(
        rank[r].weighted_ready, 0, weighted_ready_elements * sizeof(uint32_t), streams[r]));
    CUDA_CHECK(cudaMemsetAsync(
        rank[r].weighted_done, 0, done_elements * sizeof(uint32_t), streams[r]));
  }
  synchronize(options.world, streams);

  auto uniform_launch = [&](int r, uint32_t epoch) {
    fuse::A2AGemmParams params{};
    for (int peer = 0; peer < options.world; ++peer) {
      params.peer_input[peer] = rank[peer].input;
    }
    params.input_staging = rank[r].uniform_staging;
    params.rhs_nt = rank[r].weight;
    params.output = rank[r].uniform_output;
    params.ready = rank[r].uniform_ready;
    params.gemm.m = options.local_rows;
    params.gemm.n = n;
    params.gemm.k = k;
    params.gemm.raster = fuse::GemmRaster::kAlongN;
    params.route.world_size = options.world;
    params.route.rank = r;
    params.route.batch = 1;
    params.route.global_seq = global_rows;
    params.route.seq_local = options.local_rows;
    params.route.q_heads = heads;
    params.route.local_heads = heads / options.world;
    params.route.head_dim = head_dim;
    params.route.kind = fuse::RouteKind::kHeadToSequence;
    params.route.direction = fuse::RouteDirection::kInverse;
    params.num_comm_ctas = rank[r].uniform_comm;
    params.epoch = epoch;
    return fuse::launch_a2a_gemm_cutlass(params, streams[r]);
  };
  auto weighted_launch = [&](int r, uint32_t epoch) {
    fuse::WeightedA2AGemmParams params{};
    for (int peer = 0; peer < options.world; ++peer) {
      params.peer_input[peer] = rank[peer].input;
      params.peer_done_epoch[peer] = rank[peer].weighted_done;
    }
    params.input_staging = rank[r].weighted_staging;
    params.rhs_nt = rank[r].weight;
    params.output = rank[r].weighted_output;
    params.ready = rank[r].weighted_ready;
    params.gemm.m = partition.rows[r];
    params.gemm.n = n;
    params.gemm.k = k;
    params.gemm.raster = fuse::GemmRaster::kAlongN;
    params.route.world_size = options.world;
    params.route.rank = r;
    params.route.batch = 1;
    params.route.global_seq = global_rows;
    params.route.seq_local = partition.rows[r];
    params.route.q_heads = heads;
    params.route.local_heads = heads / options.world;
    params.route.head_dim = head_dim;
    params.route.kind = fuse::RouteKind::kHeadToSequence;
    params.route.direction = fuse::RouteDirection::kInverse;
    params.global_sequence_begin = partition.begin[r];
    params.num_comm_ctas = rank[r].weighted_comm;
    params.epoch = epoch;
    return fuse::launch_weighted_a2a_gemm_cutlass(params, streams[r]);
  };

  const auto uniform_timing = time_ranks(
      options.world, streams, options.warmup, options.iterations, 1,
      uniform_launch);
  const auto weighted_timing = fallback_uniform
      ? uniform_timing
      : time_ranks(
            options.world, streams, options.warmup, options.iterations, 1,
            weighted_launch);
  if (options.check && !fallback_uniform) {
    for (int r = 0; r < options.world; ++r) {
      int global_cursor = partition.begin[r];
      int local_cursor = 0;
      int remaining = partition.rows[r];
      CUDA_CHECK(cudaSetDevice(r));
      while (remaining > 0) {
        const int owner = global_cursor / options.local_rows;
        const int owner_offset = global_cursor % options.local_rows;
        const int rows = std::min(remaining, options.local_rows - owner_offset);
        CUDA_CHECK(cudaMemcpyPeerAsync(
            rank[r].expected_weighted + static_cast<int64_t>(local_cursor) * n,
            r,
            rank[owner].uniform_output + static_cast<int64_t>(owner_offset) * n,
            owner,
            static_cast<size_t>(rows) * n * sizeof(fuse::Bf16),
            streams[r]));
        global_cursor += rows;
        local_cursor += rows;
        remaining -= rows;
      }
    }
    synchronize(options.world, streams);
    for (int r = 0; r < options.world; ++r) {
      compare_exact(
          r, rank[r].expected_weighted, rank[r].weighted_output,
          static_cast<int64_t>(partition.rows[r]) * n,
          "weighted A2A+OProj");
    }
  }
  Partition equal = partition;
  std::fill(equal.rows.begin(), equal.rows.end(), options.local_rows);
  print_result(
      "oproj", "uniform", "uniform", options, equal, uniform_timing);
  print_result(
      "oproj",
      "weighted",
      fallback_uniform ? "uniform_fallback" : "weighted",
      options,
      partition,
      weighted_timing);
  std::printf(
      "SPEEDUP op=oproj value=%.6f comm_uniform=",
      percentile(uniform_timing.critical_ms, 0.50) /
          percentile(weighted_timing.critical_ms, 0.50));
  for (int r = 0; r < options.world; ++r) {
    std::printf("%s%d", r ? "," : "", rank[r].uniform_comm);
  }
  std::printf(" comm_weighted=");
  for (int r = 0; r < options.world; ++r) {
    std::printf("%s%d", r ? "," : "", rank[r].weighted_comm);
  }
  std::printf("\n");

  if (options.calibrate) {
    auto route_launch = [&](int r, uint32_t epoch) {
      fuse::A2AGemmParams params{};
      for (int peer = 0; peer < options.world; ++peer) {
        params.peer_input[peer] = rank[peer].input;
      }
      params.input_staging = rank[r].uniform_staging;
      params.rhs_nt = rank[r].weight;
      params.output = rank[r].uniform_output;
      params.ready = rank[r].uniform_ready;
      params.gemm.m = options.local_rows;
      params.gemm.n = n;
      params.gemm.k = k;
      params.gemm.raster = fuse::GemmRaster::kAlongN;
      params.route.world_size = options.world;
      params.route.rank = r;
      params.route.batch = 1;
      params.route.global_seq = global_rows;
      params.route.seq_local = options.local_rows;
      params.route.q_heads = heads;
      params.route.local_heads = heads / options.world;
      params.route.head_dim = head_dim;
      params.route.kind = fuse::RouteKind::kHeadToSequence;
      params.route.direction = fuse::RouteDirection::kInverse;
      params.num_comm_ctas = rank[r].weighted_comm;
      params.epoch = epoch;
      return fuse::launch_a2a_gemm_copy_reference(params, streams[r]);
    };
    const auto route = time_ranks(
        options.world, streams, options.warmup, options.iterations, 2000,
        route_launch);
    print_calibration("oproj", "pure_route", options, route);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const Options options = parse_options(argc, argv);
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count < options.world) {
    std::fprintf(
        stderr, "need %d visible GPUs, found %d\n", options.world, device_count);
    return 2;
  }
  if (24 % options.world != 0) {
    std::fprintf(stderr, "the common 24-head geometry does not divide CP%d\n", options.world);
    return 2;
  }
  const Partition diagnostic_partition = make_partition(options);
  std::printf(
      "CONFIG world=%d slow=%zu sm_ratio=%.6f hbm_ratio=%.6f "
      "nvlink_ratio=%.6f alpha=%.3f auto_plan=%d m=%d rows=",
      options.world, options.slow_ranks.size(), options.slow_ratio,
      options.slow_hbm_ratio, options.slow_nvlink_ratio, options.alpha,
      options.auto_plan ? 1 : 0, options.local_rows);
  for (int rank = 0; rank < options.world; ++rank) {
    std::printf(
        "%s%d", rank ? "," : "", diagnostic_partition.rows[rank]);
  }
  std::printf("\n");

  enable_peer_access(options.world);
  std::vector<cudaStream_t> streams(options.world);
  for (int rank = 0; rank < options.world; ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaStreamCreate(&streams[rank]));
  }
  if (options.operation == "both" || options.operation == "qkv") {
    if (options.auto_plan) {
      const auto plan = make_qkv_plan(options);
      const auto partition = partition_from_plan(plan);
      print_plan("qkv", plan);
      run_qkv(options, partition, streams, &plan);
    } else {
      run_qkv(options, diagnostic_partition, streams);
    }
  }
  if (options.operation == "both" || options.operation == "oproj") {
    if (options.auto_plan) {
      const auto plan = make_oproj_plan(options);
      const auto partition = partition_from_plan(plan);
      print_plan("oproj", plan);
      run_oproj(options, partition, streams, &plan);
    } else {
      run_oproj(options, diagnostic_partition, streams);
    }
  }
  for (int rank = 0; rank < options.world; ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaStreamDestroy(streams[rank]));
  }
  return 0;
}
