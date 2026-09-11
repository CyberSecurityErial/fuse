// SPDX-License-Identifier: BSD-3-Clause
// Shared CP4/8 full-validation benchmark; optional one-process-per-GPU MPI.
#include "fuse/operators/primitives/a2a_gemm.h"
#include "fuse/operators/primitives/gemm_a2a.h"
#include "fuse/profiling/qkv_route.cuh"
#include "fused_validation.cuh"
#include "fused_inputs.cuh"
#include "fused_launch.cuh"
#include "fused_mpi.cuh"
#if FUSE_BENCH_MXFP8
#include "fused_mxfp8.cuh"
#include "../../csrc/operators/sm103/detail/model_calibration.cuh"
#endif
#if FUSE_BENCH_MPI
#include "fused_graph.cuh"
#endif
#if FUSE_ENABLE_PROFILING
#include "fuse/profiling/sm103/host.cuh"
#include "fuse/profiling/sm103/epilogue.cuh"
#include "fuse/profiling/sm103/oproj.cuh"
#endif

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <unistd.h>

namespace {

using fuse::Bf16;
constexpr int kWarmup = 10;
constexpr int kSamples = 50;
constexpr uint64_t kCpuOracleElements = 4 * 1024 * 1024;

enum class MeasurementComponent { kFused, kComputeReference, kCopyReference, kQuantizeReference };

bool component_has_route(MeasurementComponent component) {
  return component == MeasurementComponent::kFused || component == MeasurementComponent::kCopyReference;
}

const char* component_name(MeasurementComponent component) {
  switch (component) {
    case MeasurementComponent::kFused: return "fused";
    case MeasurementComponent::kComputeReference: return "compute_reference";
    case MeasurementComponent::kCopyReference: return "copy_reference";
    case MeasurementComponent::kQuantizeReference: return "quantize_reference";
  }
  throw std::runtime_error("invalid measurement component");
}

struct Options {
#if FUSE_BENCH_MXFP8
  bool mxfp8_prequantized = false;
  std::string mxfp8_weight_preparation = "comm";
  int mxfp8_epilogue_n = 64;
#endif
  int world = 4;
  int comm_sm = 8;
  int seq_local = 256;
  int hidden = 1024;
  int q_heads = 32;
  int kv_heads = 8;
  int head_dim = 128;
  unsigned int timeout_seconds = 60;
  uint32_t seed = 20260906;
  bool causal = false;
  bool profile = false;
  bool qkv_epilogue_probe = false;
  bool oproj_pipeline_probe = false;
  bool oproj_gap_probe = false;
  bool cpu_oracle = false;
  bool validation_self_test = false;
  bool calibrate = false;
  bool compute_only = false;
  bool auto_oproj_comm = false;
  bool auto_mxfp8_comm = false;
  bool mxfp8_service_probe = false;
  std::string mxfp8_service_output;
  bool quick = false;
  std::string counter_component;
  std::string counter_direction;
  MeasurementComponent component = MeasurementComponent::kFused;
  std::string input_generator = "cpu_mt19937";
  std::string host_launch = "sequential";
  std::string launch = "eager";
  std::string profile_detail = "full";
  std::string profile_direction = "both";
  std::string fused_direction = "both";
  int max_swizzle_size = 1;
  std::string qkv_raster = "heuristic";
  std::string oproj_raster = "heuristic";
  std::string oproj_comm_layout = "rows";
  std::vector<int> comm_sm_list;
  std::vector<std::string> qkv_policy_list;
  std::vector<std::string> oproj_policy_list;

  // parse_options validates these products before any CUDA/host allocation.
  int global_seq() const { return seq_local * world; }
  int q_width() const { return q_heads * head_dim; }
  int kv_width() const { return kv_heads * head_dim; }
  int projection_width() const { return q_width() + 2 * kv_width(); }
  bool run_qkv() const {
    return fused_direction != "oproj" && !(profile && profile_direction == "oproj");
  }
  bool run_oproj() const { return fused_direction != "qkv"; }
};

int ceil_div(int value, int divisor) {
  return value / divisor + (value % divisor != 0);
}

size_t checked_product(size_t lhs, size_t rhs) {
  if (rhs != 0 && lhs > std::numeric_limits<size_t>::max() / rhs) {
    throw std::runtime_error("tensor size exceeds size_t");
  }
  return lhs * rhs;
}

template <class T>
size_t checked_bytes(size_t count) {
  const size_t bytes = checked_product(count, sizeof(T));
  if (bytes > static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
    throw std::runtime_error("tensor byte size exceeds addressable object size");
  }
  return bytes;
}

enum class Direction { kQkv, kOproj };

const char* direction_name(Direction direction) {
  return direction == Direction::kQkv ? "GEMM_A2A" : "A2A_GEMM";
}

// Scheduling is a per-run experiment, not another implicit candidate grid.
// Keep the public problem's heuristic value: the two launch paths retain
// their existing AlongM (QKV) / AlongN (OProj) defaults.
template <class Problem>
void apply_schedule(Problem& problem, const Options& options, Direction direction) {
  using Raster = decltype(problem.raster);
  const auto& requested = direction == Direction::kQkv ? options.qkv_raster : options.oproj_raster;
  problem.raster = requested == "along_m" ? Raster::kAlongM
      : requested == "along_n" ? Raster::kAlongN : Raster::kHeuristic;
  problem.max_swizzle_size = options.max_swizzle_size;
}

template <class Problem>
const char* effective_raster(const Problem& problem, Direction direction) {
  using Raster = decltype(problem.raster);
  const bool along_m = problem.raster == Raster::kAlongM ||
      (problem.raster == Raster::kHeuristic && direction == Direction::kQkv);
  return along_m ? "along_m" : "along_n";
}

struct ScheduleGeometry {
  int effective_swizzle_size;
  int padded_m_tiles;
  int padded_n_tiles;
  int64_t tiles() const { return int64_t{padded_m_tiles} * padded_n_tiles; }
};

ScheduleGeometry schedule_geometry(int m_tiles, int n_tiles, int max_swizzle_size) {
  // Mirrors the pinned CUTLASS 57e3cfb static scheduler, cluster=1:
  // tile_scheduler_params.h:get_log_swizzle_size and initialize. The requested
  // maximum is a cap; small tile grids can select a smaller actual swizzle.
  const int minimum = std::min(m_tiles, n_tiles);
  const int swizzle = max_swizzle_size >= 8 && minimum >= 6 ? 8
      : max_swizzle_size >= 4 && minimum >= 3 ? 4
      : max_swizzle_size >= 2 && minimum >= 2 ? 2 : 1;
  return {swizzle, ceil_div(m_tiles, swizzle) * swizzle,
                  ceil_div(n_tiles, swizzle) * swizzle};
}

struct Candidate {
  Direction direction;
  int comm_sm;
  std::string tile_policy;
  bool auto_comm = false;
};

std::vector<Candidate> make_candidates(const Options& options) {
  std::vector<Candidate> candidates;
  for (const auto& policy : options.qkv_policy_list) {
    for (int comm_sm : options.comm_sm_list) {
      candidates.push_back({Direction::kQkv, comm_sm, policy, options.auto_mxfp8_comm && comm_sm == 0});
    }
  }
  for (const auto& policy : options.oproj_policy_list) {
    for (int comm_sm : options.comm_sm_list) {
      candidates.push_back({Direction::kOproj, comm_sm, policy, options.auto_oproj_comm});
    }
  }
  candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
      [&](const Candidate& c) {
        return c.direction == Direction::kQkv ? !options.run_qkv() : !options.run_oproj();
      }), candidates.end());
  if (options.profile && options.profile_direction != "both") {
    const auto selected = options.profile_direction == "qkv" ? Direction::kQkv : Direction::kOproj;
    candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
        [selected](const Candidate& candidate) { return candidate.direction != selected; }), candidates.end());
  }
  return candidates;
}

std::string candidate_context(size_t index, const Candidate& candidate) {
  return ",candidate=" + std::to_string(index + 1) +
      ",comm_sm=" + std::to_string(candidate.comm_sm) + ",tile=" + candidate.tile_policy;
}

std::vector<std::string> split_list(const std::string& value) {
  std::vector<std::string> result;
  size_t begin = 0;
  do {
    const size_t end = value.find(',', begin);
    const std::string item = value.substr(begin, end - begin);
    if (item.empty()) throw std::runtime_error("list entries must not be empty");
    result.push_back(item);
    if (end == std::string::npos) return result;
    begin = end + 1;
  } while (true);
}

std::string normalize_qkv_policy(const std::string& policy) {
  if (policy == "auto" || policy == "m128n128") return "m128n128";
  if (policy == "m128n64" || policy == "m128n160" ||
      policy == "m128n192" || policy == "m128n256" ||
      policy == "m128n128k128" || policy == "m128n256k64e32" ||
      policy == "m128n256k128e32" || policy == "m128n256k64e64") return policy;
  throw std::runtime_error("unsupported QKV policy: " + policy);
}

std::string normalize_oproj_policy(const std::string& policy) {
  if (policy == "auto" || policy == "m128n128") return "m128n128";
  if (policy == "m128n256" || policy == "m128n128k128" ||
      policy == "m128n256k64e32" || policy == "m128n256k128e32") return policy;
  throw std::runtime_error("unsupported OProj policy: " + policy);
}

void set_tile_policy(Direction direction, const std::string& policy) {
  const char* variable = direction == Direction::kQkv
      ? "FUSE_QKV_GEMM_POLICY" : "FUSE_SM103_OPROJ_POLICY";
  if (::setenv(variable, policy.c_str(), 1) != 0) {
    throw std::runtime_error(std::string("cannot set ") + variable);
  }
}

void check_cuda(cudaError_t status, const char* expression) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(expression) + ": " + cudaGetErrorString(status));
  }
}

void check_cublas(cublasStatus_t status, const char* expression) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(expression) + ": cuBLAS status " +
                             std::to_string(static_cast<int>(status)));
  }
}

#define CUDA_CHECK(expression) check_cuda((expression), #expression)
#define CUBLAS_CHECK(expression) check_cublas((expression), #expression)

// A failed peer launch can leave another GPU waiting inside finalize. Exit
// only this benchmark process, without blocking in CUDA cleanup or resetting
// devices. The alarm also covers a host CUDA call that does not return.
void timeout_handler(int) {
  constexpr char message[] = "fused_bf16: watchdog expired; exiting this process\n";
  const auto ignored = ::write(STDERR_FILENO, message, sizeof(message) - 1);
  (void)ignored;
  ::_exit(124);
}

Options parse_options(int argc, char** argv) {
  Options options;
  if (const char* layout = std::getenv("FUSE_SM103_OPROJ_COMM_LAYOUT")) {
    options.oproj_comm_layout = layout;
  }
#if FUSE_BENCH_MPI
  options.world = fused_mpi::process_world;
  options.host_launch = "mpi_process";
#endif
  bool has_seq_local = false, has_global_seq = false;
  bool has_comm_sm = false, has_comm_sm_list = false;
  bool has_profile_detail = false;
  uint64_t global_seq = 0;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto number = [&]() -> uint64_t {
      if (++index == argc || argv[index][0] == '-') {
        throw std::runtime_error("missing positive value for " + argument);
      }
      size_t consumed = 0;
      const std::string value = argv[index];
      const uint64_t result = std::stoull(value, &consumed);
      if (consumed != value.size() || result > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("invalid value for " + argument);
      }
      return result;
    };
    auto positive_int = [&]() {
      const uint64_t value = number();
      if (value == 0 || value > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error("requires a positive int32 value for " + argument);
      }
      return static_cast<int>(value);
    };
    if (argument == "--world") {
      const uint64_t value = number();
      if (value != 4 && value != 8) throw std::runtime_error("--world must be 4 or 8");
      options.world = static_cast<int>(value);
    } else if (argument == "--world4" || argument == "--world8") {
      options.world = argument == "--world4" ? 4 : 8;
    } else if (argument == "--auto-oproj-comm") {
      options.auto_oproj_comm = true;
    } else if (argument == "--auto-mxfp8-comm") {
      options.auto_mxfp8_comm = true;
    } else if (argument == "--mxfp8-service-probe") {
      options.mxfp8_service_probe = true;
    } else if (argument == "--mxfp8-service-output") {
      if (++index == argc || argv[index][0] == '-') throw std::runtime_error("missing MXFP8 service output path");
      options.mxfp8_service_output = argv[index];
    } else if (argument == "--comm-sm") {
      const uint64_t value = number();
      if (value == 0 || value > 1024) throw std::runtime_error("invalid --comm-sm");
      options.comm_sm = static_cast<int>(value);
      has_comm_sm = true;
    } else if (argument == "--comm-sm-list" || argument == "--qkv-policy-list" ||
               argument == "--oproj-policy-list") {
      if (++index == argc) throw std::runtime_error("missing value for " + argument);
      const auto items = split_list(argv[index]);
      if (argument == "--comm-sm-list") {
        has_comm_sm_list = true;
        for (const auto& item : items) {
          if (item.find_first_not_of("0123456789") != std::string::npos) {
            throw std::runtime_error("--comm-sm-list requires positive integers");
          }
          const uint64_t value = std::stoull(item);
          if (value == 0 || value > 1024) throw std::runtime_error("invalid --comm-sm-list");
          const int comm_sm = static_cast<int>(value);
          if (std::find(options.comm_sm_list.begin(), options.comm_sm_list.end(), comm_sm) ==
              options.comm_sm_list.end()) options.comm_sm_list.push_back(comm_sm);
        }
      } else {
        const bool qkv = argument == "--qkv-policy-list";
        auto& policies = qkv ? options.qkv_policy_list : options.oproj_policy_list;
        for (const auto& item : items) {
          const auto policy = qkv ? normalize_qkv_policy(item) : normalize_oproj_policy(item);
          if (std::find(policies.begin(), policies.end(), policy) == policies.end()) {
            policies.push_back(policy);
          }
        }
      }
    } else if (argument == "--seq-local") {
      options.seq_local = positive_int();
      has_seq_local = true;
    } else if (argument == "--global-seq") {
      global_seq = positive_int();
      has_global_seq = true;
    } else if (argument == "--hidden") {
      options.hidden = positive_int();
    } else if (argument == "--q-heads") {
      options.q_heads = positive_int();
    } else if (argument == "--kv-heads") {
      options.kv_heads = positive_int();
    } else if (argument == "--head-dim") {
      options.head_dim = positive_int();
    } else if (argument == "--timeout-seconds") {
      options.timeout_seconds = static_cast<unsigned int>(positive_int());
    } else if (argument == "--max-swizzle-size") {
      options.max_swizzle_size = positive_int();
      if (options.max_swizzle_size != 1 && options.max_swizzle_size != 2 &&
          options.max_swizzle_size != 4 && options.max_swizzle_size != 8) {
        throw std::runtime_error("--max-swizzle-size requires 1, 2, 4, or 8");
      }
    } else if (argument == "--qkv-raster" || argument == "--oproj-raster") {
      if (++index == argc) throw std::runtime_error("missing value for " + argument);
      std::string value = argv[index];
      if (value != "heuristic" && value != "along_m" && value != "along_n") {
        throw std::runtime_error(argument + " requires heuristic, along_m, or along_n");
      }
      (argument == "--qkv-raster" ? options.qkv_raster : options.oproj_raster) = value;
    } else if (argument == "--oproj-comm-layout") {
      if (++index == argc) throw std::runtime_error("missing value for " + argument);
      options.oproj_comm_layout = argv[index];
    } else if (argument == "--seed") {
      options.seed = static_cast<uint32_t>(number());
    } else if (argument == "--input-generator") {
      if (++index == argc) throw std::runtime_error("missing value for " + argument);
      options.input_generator = argv[index];
      if (options.input_generator != "cpu_mt19937" && options.input_generator != "gpu_philox") {
        throw std::runtime_error("--input-generator requires cpu_mt19937 or gpu_philox");
      }
    } else if (argument == "--launch") {
      if (++index == argc) throw std::runtime_error("missing value for " + argument);
      options.launch = argv[index];
      if (options.launch != "eager" && options.launch != "graph") {
        throw std::runtime_error("--launch requires eager or graph");
      }
    } else if (argument == "--host-launch") {
#if FUSE_BENCH_MPI
      throw std::runtime_error("MPI target owns one GPU per process; --host-launch is single-process only");
#endif
      if (++index == argc) throw std::runtime_error("missing value for " + argument);
      options.host_launch = argv[index];
      if (options.host_launch != "sequential" && options.host_launch != "per_gpu_thread") {
        throw std::runtime_error("--host-launch requires sequential or per_gpu_thread");
      }
    } else if (argument == "--causal") {
      options.causal = true;
#if FUSE_BENCH_MXFP8
    } else if (argument == "--mxfp8-epilogue-n") {
      options.mxfp8_epilogue_n = positive_int();
      if (options.mxfp8_epilogue_n != 32 && options.mxfp8_epilogue_n != 64)
        throw std::runtime_error("--mxfp8-epilogue-n requires 32 or 64");
    } else if (argument == "--mxfp8-prequantized") {
      options.mxfp8_prequantized = true;
    } else if (argument == "--mxfp8-weight-preparation") {
      if (++index == argc) throw std::runtime_error("missing MXFP8 weight preparation");
      options.mxfp8_weight_preparation = argv[index];
      if (options.mxfp8_weight_preparation != "comm" && options.mxfp8_weight_preparation != "all" &&
          options.mxfp8_weight_preparation != "comm_warp")
        throw std::runtime_error("--mxfp8-weight-preparation requires comm, all, or comm_warp");
#endif
    } else if (argument == "--profile") {
      options.profile = true;
    } else if (argument == "--fused-direction") {
      if (++index == argc) throw std::runtime_error("missing fused direction");
      options.fused_direction = argv[index];
      if (options.fused_direction != "both" && options.fused_direction != "qkv" &&
          options.fused_direction != "oproj") throw std::runtime_error("invalid fused direction");
    } else if (argument == "--profile-direction") {
      if (++index == argc) throw std::runtime_error("missing profile direction");
      options.profile_direction = argv[index];
      if (options.profile_direction != "both" && options.profile_direction != "qkv" &&
          options.profile_direction != "oproj") throw std::runtime_error("invalid profile direction");
    } else if (argument == "--qkv-epilogue-probe") {
      options.qkv_epilogue_probe = true;
    } else if (argument == "--oproj-pipeline-probe") {
      options.oproj_pipeline_probe = true;
    } else if (argument == "--oproj-gap-probe") {
      options.oproj_gap_probe = true;
    } else if (argument == "--profile-detail") {
      if (++index == argc) throw std::runtime_error("missing value for " + argument);
      options.profile_detail = argv[index];
      has_profile_detail = true;
      if (options.profile_detail != "full" && options.profile_detail != "cta") {
        throw std::runtime_error("--profile-detail requires full or cta");
      }
    } else if (argument == "--calibrate") {
      options.calibrate = true;
    } else if (argument == "--compute-only") {
      options.compute_only = true;
    } else if (argument == "--quick") {
      options.quick = true;
    } else if (argument == "--counter-component" || argument == "--counter-direction") {
      if (++index == argc) throw std::runtime_error("missing value for " + argument);
      (argument == "--counter-component" ? options.counter_component : options.counter_direction) = argv[index];
    } else if (argument == "--cpu-oracle") {
      options.cpu_oracle = true;
    } else if (argument == "--validation-self-test") {
      options.validation_self_test = true;
      options.cpu_oracle = true;
    } else if (argument == "--help") {
      std::cout << "fused_bf16 [--world 4|8] [--comm-sm 8 | --comm-sm-list 8,16] "
                   "[--qkv-policy-list m128n128,m128n256] [--oproj-policy-list m128n128,m128n256] "
                   "[--seq-local 256 | --global-seq S] "
                   "[--hidden 1024] [--q-heads 32] [--kv-heads 8] [--head-dim 128] "
                   "[--seed 20260906] [--causal] [--profile [--profile-detail full|cta]] [--timeout-seconds 60] "
                   "[--input-generator cpu_mt19937|gpu_philox] "
                   "[--max-swizzle-size 1|2|4|8] "
                   "[--qkv-raster heuristic|along_m|along_n] [--oproj-raster heuristic|along_m|along_n] "
                   "[--oproj-comm-layout rows|columns] "
                   "[--host-launch sequential|per_gpu_thread] [--launch eager|graph] "
                   "[--cpu-oracle] [--validation-self-test] [--calibrate]\n"
#if FUSE_BENCH_MXFP8
                   "--mxfp8-weight-preparation comm|all|comm_warp: comm_warp uses warp_then_route_v1; "
                   "warps 0..3 route immediately, warps 4..7 quantize their weights then join routing.\n"
#endif
                   "--auto-oproj-comm: runtime CTA selection; Graph OProj causal rows, explicit N256/e32 and sw4/8 only; excludes --comm-sm/list.\n"
                   "--auto-mxfp8-comm: MXFP8 Graph QKV, ordinary comm, explicit raster; optional --comm-sm/list pairs manual budgets with runtime auto.\n"
                   "--mxfp8-service-probe: CP4/8 full-profile C/Q/R/QR diagnostics; requires internal --mxfp8-service-output path.\n"
                   "BF16 GQA; warmup=10, samples=50. Global sequence must divide evenly across ranks.\n"
                   "--causal changes OProj gather; QKV output remains rank-major, matching SM90.\n"
                   "Candidates reuse one shape and two payload generations; each direction expands tile x comm.\n"
                   "--profile requires one communication budget and one policy per direction.\n"
                   "--profile-detail defaults to full; cta omits peer traces but retains instrumentation overhead.\n"
                   "--qkv-epilogue-probe adds a private three-mode diagnostic; requires --profile --profile-detail cta "
                   "and QKV m128n256k64e32. Not an MPI performance result.\n"
                   "--calibrate adds independently validated compute/copy references; excludes profile/self-test.\n"
                   "QKV policies: auto,m128n64,m128n128,m128n160,m128n192,m128n256,"
                   "m128n128k128,m128n256k64e32,m128n256k128e32,m128n256k64e64.\n"
                   "OProj policies: auto,m128n128,m128n256,m128n128k128,m128n256k64e32,m128n256k128e32.\n"
                   "e32/e64 select epilogue subtiles 128x32/128x64; E64 is QKV-only.\n"
                   "Unsuffixed policies retain K64/original epilogue.\n"
                   "Each list overrides its direction's environment.\n"
                   "OProj communication layout is fixed for the run, independent of GEMM tile policy; CLI overrides environment.\n"
                   "Input generators use different reproducible sequences; gpu_philox avoids full host input buffers.\n"
                   "Scheduling is shared by production and compute calibration; defaults are swizzle=1, QKV AlongM, OProj AlongN.\n"
                   "--profile requires no M/N tile padding from the effective swizzle; production/calibration allow padding.\n"
                   "CPU oracle/self-test: at most 4194304 activation/output elements per rank; self-test does not benchmark.\n"
                   "Full host/reference buffers are retained; large shapes require sufficient host and GPU memory.\n"
                   "Use CUDA_VISIBLE_DEVICES to select GPUs; --profile requires FUSE_ENABLE_PROFILING.\n"
                   "Optional fused_bf16_mpi: CP4/8 on one host, Eager or Graph; no --host-launch or --profile.\n"
                   "Graph recaptures updated epochs before each sample; host preparation is not GPU event time.\n";
      fused_mpi::finalize();
      std::exit(0);
    } else {
      throw std::runtime_error("unknown option: " + argument);
    }
  }
  if (has_comm_sm && has_comm_sm_list) {
    throw std::runtime_error("--comm-sm and --comm-sm-list are mutually exclusive");
  }
  if (options.auto_oproj_comm && (has_comm_sm || has_comm_sm_list)) {
    throw std::runtime_error("--auto-oproj-comm and --comm-sm/list are mutually exclusive");
  }
  if (options.auto_mxfp8_comm && options.auto_oproj_comm)
    throw std::runtime_error("MXFP8 and BF16 OProj automatic modes are mutually exclusive");
#if !FUSE_BENCH_MXFP8
  if (options.auto_mxfp8_comm) throw std::runtime_error("--auto-mxfp8-comm requires the MXFP8 target");
  if (options.mxfp8_service_probe) throw std::runtime_error("--mxfp8-service-probe requires the MXFP8 target");
#endif
  if (options.quick && (options.profile || options.validation_self_test ||
      options.oproj_gap_probe || !options.counter_component.empty())) {
    throw std::runtime_error("--quick is a non-profile performance screening mode");
  }
  if (options.oproj_comm_layout != "rows" && options.oproj_comm_layout != "columns") {
    throw std::runtime_error("--oproj-comm-layout requires rows or columns");
  }
#if FUSE_BENCH_MPI
  if (options.world != fused_mpi::process_world) throw std::runtime_error("--world differs from MPI size");
  if (options.profile) throw std::runtime_error("MPI profiling is not implemented; use the single-process target");
#endif
  if (options.launch == "graph" && (!fused_mpi::enabled || options.profile || options.qkv_epilogue_probe)) {
    throw std::runtime_error("Graph requires the MPI target and excludes profiling/epilogue diagnostics");
  }
  if (options.auto_oproj_comm) options.comm_sm = 0; // Unresolved request, never a measured 8-CTA default.
  if (options.auto_mxfp8_comm && !has_comm_sm && !has_comm_sm_list) options.comm_sm = 0;
  if (options.comm_sm_list.empty()) options.comm_sm_list.push_back(options.comm_sm);
  if (options.auto_mxfp8_comm && options.comm_sm_list.back() != 0) options.comm_sm_list.push_back(0);
  options.comm_sm = options.comm_sm_list.front();
  if (options.qkv_policy_list.empty()) {
    const char* policy = std::getenv("FUSE_QKV_GEMM_POLICY");
    options.qkv_policy_list.push_back(normalize_qkv_policy(policy ? policy : "auto"));
  }
  if (options.oproj_policy_list.empty()) {
    const char* policy = std::getenv("FUSE_SM103_OPROJ_POLICY");
    options.oproj_policy_list.push_back(normalize_oproj_policy(policy ? policy : "auto"));
  }
  if (options.auto_oproj_comm && (options.profile || options.validation_self_test || options.compute_only ||
      !options.counter_component.empty() || options.fused_direction != "oproj" || options.launch != "graph" ||
      !options.causal || options.oproj_comm_layout != "rows" ||
      (options.max_swizzle_size != 4 && options.max_swizzle_size != 8) ||
      std::any_of(options.oproj_policy_list.begin(), options.oproj_policy_list.end(), [](const auto& policy) {
        return policy != "m128n256" && policy != "m128n256k64e32";
      }))) {
    throw std::runtime_error("--auto-oproj-comm requires non-profile OProj Graph causal rows with explicit N256/e32 and sw4/8");
  }
  if (options.profile && (options.comm_sm_list.size() != 1 || options.qkv_policy_list.size() != 1 ||
                         options.oproj_policy_list.size() != 1)) {
    throw std::runtime_error("--profile requires one communication budget and one policy per direction");
  }
  if (has_profile_detail && !options.profile) {
    throw std::runtime_error("--profile-detail requires --profile");
  }
  if (options.oproj_pipeline_probe && (!options.profile || options.profile_detail != "full" ||
      options.profile_direction != "oproj" || fused_mpi::enabled || options.qkv_epilogue_probe)) {
    throw std::runtime_error("OProj pipeline probe requires single-process full OProj profiling");
  }
  if (options.oproj_gap_probe && (!options.profile || options.profile_detail != "full" ||
      options.profile_direction != "oproj" || options.oproj_pipeline_probe || fused_mpi::enabled)) {
    throw std::runtime_error("OProj gap probe requires standalone full OProj profiling without the K-stage probe");
  }
  if (options.qkv_epilogue_probe && (!options.profile || options.profile_detail != "cta" ||
                                    options.qkv_policy_list.front() != "m128n256k64e32")) {
    throw std::runtime_error("--qkv-epilogue-probe requires --profile --profile-detail cta and QKV m128n256k64e32");
  }
  if (options.calibrate && (options.profile || options.validation_self_test)) {
    throw std::runtime_error("--calibrate is mutually exclusive with --profile and --validation-self-test");
  }
  if (options.compute_only && (!options.calibrate || options.run_qkv() || options.profile ||
      !options.counter_component.empty())) {
    throw std::runtime_error("compute-only requires OProj calibration without profiling/counters");
  }
  if (!options.counter_component.empty() || !options.counter_direction.empty()) {
    if (!options.calibrate || options.profile || options.validation_self_test ||
        options.launch != "eager" ||
        options.host_launch != (fused_mpi::enabled ? "mpi_process" : "sequential") ||
        options.comm_sm_list.size() != 1 || options.qkv_policy_list.size() != 1 ||
        options.oproj_policy_list.size() != 1 ||
        (options.counter_direction != "qkv" && options.counter_direction != "oproj") ||
        (options.counter_component != "fused" && options.counter_component != "compute_reference" &&
         options.counter_component != "copy_reference")) {
      throw std::runtime_error("Counters require eager calibration, one direction/component and one candidate");
    }
  }
  if (has_seq_local && has_global_seq) {
    throw std::runtime_error("--seq-local and --global-seq are mutually exclusive");
  }
  if (has_global_seq) {
    if (global_seq % options.world != 0) {
      throw std::runtime_error("--global-seq must be divisible by --world");
    }
    options.seq_local = static_cast<int>(global_seq / options.world);
  }
  const int64_t q_width = static_cast<int64_t>(options.q_heads) * options.head_dim;
#if !FUSE_BENCH_MXFP8
  if (options.profile && options.fused_direction != "both") {
    throw std::runtime_error("profiling uses --profile-direction, not --fused-direction");
  }
#endif
  const int64_t kv_width = static_cast<int64_t>(options.kv_heads) * options.head_dim;
  const int64_t int_limit = std::numeric_limits<int32_t>::max();
  if (static_cast<int64_t>(options.seq_local) * options.world > int_limit ||
      q_width > int_limit || kv_width > int_limit ||
      q_width + 2 * kv_width > int_limit) {
    throw std::runtime_error("global sequence or packed projection width exceeds int32");
  }
  if ((options.run_qkv() && (options.q_heads % options.kv_heads != 0 ||
      options.kv_heads % options.world != 0)) || options.q_heads % options.world != 0 || options.head_dim % 8 != 0 ||
      options.hidden % 8 != 0 || (q_width / options.world) % 64 != 0) {
    throw std::runtime_error("requires GQA/CP-divisible heads, BF16 8-element alignment, and OProj K shards divisible by 64");
  }
  const int64_t m_tiles = ceil_div(options.seq_local, 128);
  if (options.auto_oproj_comm && (options.seq_local % 256 || q_width < 8192 || q_width > 16384 ||
      schedule_geometry(static_cast<int>(m_tiles), ceil_div(options.hidden, 256), options.max_swizzle_size)
          .effective_swizzle_size < 4)) {
    throw std::runtime_error("--auto-oproj-comm geometry is outside the calibrated ready/K/swizzle domain");
  }
  if (m_tiles * ceil_div(options.projection_width(), 64) > int_limit ||
      m_tiles * ceil_div(options.hidden, 128) > int_limit ||
      m_tiles * options.world > int_limit) {
    throw std::runtime_error("tile grid or profiling capacity exceeds int32");
  }
  for (const auto& candidate : make_candidates(options)) {
    // Policy names were normalized above; all registered SM103 policies use
    // physical M=128. Kernel-trait queries still audit the actual launch later.
    const int tile_n = std::stoi(candidate.tile_policy.substr(5));
    const int width = candidate.direction == Direction::kQkv
        ? options.projection_width() : options.hidden;
    const int n_tiles = ceil_div(width, tile_n);
    const auto schedule = schedule_geometry(static_cast<int>(m_tiles), n_tiles,
                                            options.max_swizzle_size);
    if (schedule.tiles() > int_limit) throw std::runtime_error("padded tile grid exceeds int32");
    if (options.profile && !options.mxfp8_service_probe &&
        (schedule.padded_m_tiles != m_tiles || schedule.padded_n_tiles != n_tiles)) {
      throw std::runtime_error("--profile does not support swizzle-padded M/N tiles; "
                               "use an unpadded geometry or smaller swizzle (production/calibration remain supported)");
    }
  }
  for (const int width : {options.hidden, options.q_width(), options.projection_width()}) {
    if (options.cpu_oracle && checked_product(options.seq_local, width) > kCpuOracleElements) {
      throw std::runtime_error("CPU oracle/self-test requires at most 4194304 activation/output elements per rank");
    }
    checked_bytes<Bf16>(checked_product(options.seq_local, width));
    checked_bytes<Bf16>(checked_product(options.hidden, width));
  }
  if (options.causal && options.seq_local % 2 != 0) {
    throw std::runtime_error("--causal requires an even --seq-local");
  }
#if !FUSE_ENABLE_PROFILING
  if (options.profile) throw std::runtime_error("rebuild with FUSE_ENABLE_PROFILING=ON for --profile");
#endif
#if FUSE_BENCH_MXFP8
  if (options.mxfp8_service_probe && (!options.profile || fused_mpi::enabled ||
      (options.world != 4 && options.world != 8) ||
      options.profile_detail != "full" || options.launch != "eager" || options.auto_mxfp8_comm ||
      options.host_launch != "per_gpu_thread" ||
      options.calibrate || options.mxfp8_prequantized || options.mxfp8_weight_preparation != "comm" ||
      options.mxfp8_service_output.empty() ||
      (options.qkv_raster != "along_m" && options.qkv_raster != "along_n")))
    throw std::runtime_error("MXFP8 services require CP4/8 single-process full profile, explicit ordinary comm/layout, and output path");
  if (!options.mxfp8_service_probe && !options.mxfp8_service_output.empty())
    throw std::runtime_error("--mxfp8-service-output requires --mxfp8-service-probe");
  if (options.auto_mxfp8_comm && (!fused_mpi::enabled || options.launch != "graph" || options.profile ||
      options.mxfp8_prequantized || options.mxfp8_weight_preparation != "comm" ||
      (options.qkv_raster != "along_m" && options.qkv_raster != "along_n")))
    throw std::runtime_error("MXFP8 automatic CTAs require MPI Graph, dynamic-weight ordinary comm, and explicit raster");
  if (options.fused_direction != "qkv" ||
      options.cpu_oracle || options.compute_only || options.validation_self_test ||
      !options.counter_component.empty() || options.quick)
    throw std::runtime_error("MXFP8 baseline requires QKV-only, full 10+50, without BF16-specific diagnostics");
  if (options.profile && (options.profile_direction != "qkv" || options.mxfp8_prequantized))
    throw std::runtime_error("MXFP8 profile requires dynamic-weight QKV");
  if (options.calibrate && (options.mxfp8_prequantized || options.mxfp8_weight_preparation != "comm"))
    throw std::runtime_error("MXFP8 C/Q/R calibration requires dynamic-weight ordinary communication warps");
  if (options.qkv_policy_list.empty()) options.qkv_policy_list = {"m128n256"};
  if (options.qkv_policy_list != std::vector<std::string>{"m128n256"} || options.hidden % 128 || options.head_dim != 128)
    throw std::runtime_error("MXFP8 baseline requires M128/N256/K128, hidden divisible by 128, and head_dim=128");
#endif
  return options;
}

struct RankRuntime {
  // The vector remains indexed by logical rank. Only owned entries have a
  // device/stream/allocations; other entries are non-owning IPC peer views.
  int device = -1;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t end = nullptr;
  cublasHandle_t blas = nullptr;
  int sm_count = 0;
  std::vector<void*> allocations;
  std::vector<int> enabled_peers;
  std::vector<void*> ipc_mappings;
  fuse::GemmA2AParams qkv{};
#if FUSE_BENCH_MXFP8
  void* mxfp8_workspace = nullptr;
  size_t mxfp8_workspace_bytes = 0;
  fuse::Mxfp8Activation mxfp8_activation{};
  fuse::Mxfp8WeightPreparation mxfp8_weight_preparation = fuse::Mxfp8WeightPreparation::kCommunicationCtas;
  int mxfp8_epilogue_n = 64;
  Bf16* mxfp8_reference_a = nullptr;
  Bf16* mxfp8_reference_b = nullptr;
  bool mxfp8_prequantized = false;
#if FUSE_ENABLE_PROFILING
  fuse::Mxfp8ProfileView mxfp8_probe{};
  fuse::Mxfp8ServiceView mxfp8_service{};
#endif
#endif
  fuse::A2AGemmParams oproj{};
  Bf16* peer_input = nullptr;
  Bf16* peer_output = nullptr;
  uint32_t* input_ready = nullptr;
  uint32_t* route_done = nullptr;
  Bf16* qkv_reference = nullptr;
  Bf16* oproj_reference = nullptr;
  Bf16* reference_lhs = nullptr;
  std::vector<Bf16> qkv_input;
  std::vector<Bf16> oproj_input;
  std::vector<Bf16> reference_input;
  fused_validation::Scratch* validation = nullptr;
  fused_inputs::Scratch* inputs = nullptr;
  fused_launch::Team* launch_team = nullptr;
#if FUSE_BENCH_MPI
  std::unique_ptr<fused_graph::Operation> graph;
  uint64_t graph_prepare_calls = 0;
  double graph_prepare_wall_s = 0;
  uint32_t graph_bound_epoch = 0;
#endif
  uint32_t* calibration_qkv_ready = nullptr;
  uint32_t* calibration_oproj_ready = nullptr;
  uint32_t* calibration_route_done = nullptr;
  size_t qkv_ready_elements = 0;
  size_t oproj_ready_elements = 0;
#if FUSE_ENABLE_PROFILING
  fuse::A2AGemmCtaTimeline* timeline = nullptr;
  fuse::QkvRouteTimeline* qkv_route_timeline = nullptr;
  int qkv_route_capacity = 0;
  fuse::A2AGemmPeerTimeline* peer_timeline = nullptr;
  fuse::detail::QkvEpilogueRecord* qkv_epilogue = nullptr;
  fuse::detail::OprojPipelineView oproj_pipeline{};
  int peer_capacity = 0;
#endif
};

void check_enqueue(const fused_launch::Result& result) {
  if (!result.error) return;
  // Another rank may already be waiting in CUDA. Do not unwind a live job,
  // join workers, or reset devices after a partial cross-rank submission.
  try {
    std::rethrow_exception(result.error);
  } catch (const std::exception& error) {
    std::cerr << "fused_bf16: enqueue failed, rank=" << result.rank << ": " << error.what() << '\n';
  } catch (...) {
    std::cerr << "fused_bf16: enqueue failed, rank=" << result.rank << ": unknown exception\n";
  }
  std::cerr << std::flush;
  ::_exit(1);
}

std::unique_ptr<fused_launch::Team> start_launch_team(
    std::vector<RankRuntime>& runtimes, const Options& options) {
  if (fused_mpi::enabled || options.host_launch == "sequential") return nullptr;
  auto team = std::make_unique<fused_launch::Team>(options.world);
  check_enqueue(team->dispatch([](void* context, int rank) {
    auto& owned = *static_cast<std::vector<RankRuntime>*>(context);
    CUDA_CHECK(cudaSetDevice(owned[rank].device));
  }, &runtimes));
  for (auto& runtime : runtimes) runtime.launch_team = team.get();
  return team;
}

template <class T>
T* allocate(RankRuntime& runtime, size_t count) {
  T* pointer = nullptr;
  const size_t bytes = checked_bytes<T>(count);
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer), bytes));
  runtime.allocations.push_back(pointer);
  CUDA_CHECK(cudaMemsetAsync(pointer, 0, bytes, runtime.stream));
  return pointer;
}

template <class T>
void upload(T* destination, const std::vector<T>& source, cudaStream_t stream) {
  CUDA_CHECK(cudaMemcpyAsync(destination, source.data(), checked_bytes<T>(source.size()), cudaMemcpyHostToDevice, stream));
  // Sources can be temporary pageable vectors. Keep their lifetime explicit
  // and do not rely on legacy-default-stream ordering with nonblocking streams.
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template <class T>
std::vector<T> download(const T* source, size_t count) {
  const size_t bytes = checked_bytes<T>(count);
  std::vector<T> result(count);
  CUDA_CHECK(cudaMemcpy(result.data(), source, bytes, cudaMemcpyDeviceToHost));
  return result;
}

std::vector<Bf16> make_values(size_t count, uint32_t seed, const std::string& label,
                              float magnitude) {
  checked_bytes<Bf16>(count);
  std::mt19937 random(seed);
  std::vector<Bf16> result(count);
  double sum = 0.0, square_sum = 0.0;
  float minimum = std::numeric_limits<float>::infinity(), maximum = -minimum;
  size_t nonzero = 0;
  for (auto& value : result) {
    // mt19937's integer sequence is specified; do not depend on a library's
    // uniform_real_distribution implementation for reproducibility.
    value = Bf16((static_cast<float>(random() >> 8) / 16777216.0f - 0.5f) * (2 * magnitude));
    const float scalar = static_cast<float>(value);
    if (!std::isfinite(scalar)) throw std::runtime_error(label + " nonfinite input");
    nonzero += scalar != 0;
    sum += scalar;
    square_sum += scalar * scalar;
    minimum = std::min(minimum, scalar);
    maximum = std::max(maximum, scalar);
  }
  if (nonzero == 0) throw std::runtime_error(label + " all-zero input");
  const double mean = sum / count, mean_square = square_sum / count;
  std::cout << "input," << label << ",seed=" << seed << ",count=" << count
            << ",distribution=uniform,lower=" << -magnitude << ",upper=" << magnitude
            << ",min=" << minimum << ",max=" << maximum << ",mean=" << mean
            << ",rms=" << std::sqrt(mean_square)
            << ",std=" << std::sqrt(std::max(0.0, mean_square - mean * mean))
            << ",nonzero_fraction=" << static_cast<double>(nonzero) / count << '\n';
  return result;
}

void make_gpu_values(RankRuntime& runtime, Bf16* output, size_t count, uint32_t seed,
                       const std::string& label, float magnitude) {
  CUDA_CHECK(fused_inputs::generate(reinterpret_cast<uint16_t*>(output), count, seed,
                                    magnitude, runtime.inputs, runtime.stream));
  CUDA_CHECK(cudaStreamSynchronize(runtime.stream));
  const auto stats = download(&runtime.inputs->result, 1).front();
  const float bound = static_cast<float>(Bf16(magnitude));
  if (stats.count != count || stats.finite != count || stats.nonzero == 0 ||
      stats.minimum < -bound || stats.maximum > bound ||
      !std::isfinite(stats.sum) || !std::isfinite(stats.square_sum)) {
    throw std::runtime_error(label + " invalid GPU input statistics");
  }
  const double mean = stats.sum / count, mean_square = stats.square_sum / count;
  std::cout << "input," << label << ",seed=" << seed << ",count=" << count
            << ",generator=gpu_philox,algorithm=curand_philox4x32_10,mapping=thread_subsequence_v1"
            << ",blocks=" << fused_inputs::kBlocks << ",threads=" << fused_inputs::kThreads
            << ",offset=0,distribution=uniform,lower=" << -magnitude << ",upper=" << magnitude
            << ",min=" << stats.minimum << ",max=" << stats.maximum << ",mean=" << mean
            << ",rms=" << std::sqrt(mean_square)
            << ",std=" << std::sqrt(std::max(0.0, mean_square - mean * mean))
            << ",finite=" << stats.finite
            << ",nonzero_fraction=" << static_cast<double>(stats.nonzero) / count << '\n';
}

fuse::UlyssesRoute make_route(const Options& options, int rank, Direction direction) {
  fuse::UlyssesRoute route{};
  route.world_size = options.world;
  route.rank = rank;
  route.batch = 1;
  route.global_seq = options.global_seq();
  route.seq_local = options.seq_local;
  route.q_heads = options.q_heads;
  route.kv_heads = options.kv_heads;
  route.local_heads = options.q_heads / options.world;
  route.head_dim = options.head_dim;
  route.causal_load_balanced = options.causal;
  route.kind = direction == Direction::kQkv ? fuse::RouteKind::kQkvGqaPack
                                           : fuse::RouteKind::kHeadToSequence;
  route.direction = direction == Direction::kQkv ? fuse::RouteDirection::kForward
                                                : fuse::RouteDirection::kInverse;
  return route;
}

// Independent CPU ownership mapping; do not use the operator's route helper.
int global_sequence(const Options& options, int rank, int local_row) {
  if (!options.causal) return rank * options.seq_local + local_row;
  const int half = options.seq_local / 2;
  return local_row < half ? rank * half + local_row
                         : (2 * options.world - 1 - rank) * half + (local_row - half);
}

void wait_all(std::vector<RankRuntime>& runtimes) {
  // Callers have already recorded every rank's CURRENT end event. Waiting
  // here cannot prevent a peer launch; SIGALRM still bounds this process.
  // A 1 ms polling sleep starved short kernels of the required GPU warmup
  // time (measured in C/R calibration). Let CUDA wait for actual completion.
  for (const int rank : fused_mpi::owned_ranks(runtimes.size())) {
    CUDA_CHECK(cudaSetDevice(runtimes[rank].device));
    CUDA_CHECK(cudaEventSynchronize(runtimes[rank].end));
  }
}

void finish_all(std::vector<RankRuntime>& runtimes) {
  for (const int rank : fused_mpi::owned_ranks(runtimes.size())) {
    CUDA_CHECK(cudaSetDevice(runtimes[rank].device));
    CUDA_CHECK(cudaEventRecord(runtimes[rank].end, runtimes[rank].stream));
  }
  wait_all(runtimes);
  // Input publication, poison, validation and cleanup are cross-rank stages.
  fused_mpi::barrier();
}

void bind_graph(std::vector<RankRuntime>& runtimes, const Options& options, uint32_t committed_epoch) {
#if FUSE_BENCH_MPI
  if (options.launch != "graph") return;
  finish_all(runtimes);
  for (const int rank : fused_mpi::owned_ranks(options.world)) {
    auto& runtime = runtimes[rank];
    CUDA_CHECK(cudaSetDevice(runtime.device));
    if (runtime.graph) runtime.graph->reset(committed_epoch);
    else runtime.graph = std::make_unique<fused_graph::Operation>(
        runtime.device, runtime.stream, committed_epoch);
    runtime.graph_bound_epoch = committed_epoch;
    runtime.graph_prepare_calls = 0;
    runtime.graph_prepare_wall_s = 0;
  }
#else
  (void)runtimes;
  (void)options;
  (void)committed_epoch;
#endif
}

void report_graph_preparation(const std::vector<RankRuntime>& runtimes, const Options& options,
                               Direction direction, const std::string& context) {
#if FUSE_BENCH_MPI
  if (options.launch != "graph") return;
  for (const int rank : fused_mpi::owned_ranks(options.world)) {
    const auto& runtime = runtimes[rank];
    if (!runtime.graph || runtime.graph_prepare_calls == 0) throw std::runtime_error("missing Graph preparation");
    std::cout << "graph_prepare," << direction_name(direction) << context << ",rank=" << rank
              << ",launch=graph,graph_epoch_mode=recapture_update_v1"
              << ",calls=" << runtime.graph_prepare_calls
              << ",first_epoch=" << runtime.graph_bound_epoch + 1
              << ",last_epoch=" << runtime.graph->committed_epoch()
              << ",wall_s=" << runtime.graph_prepare_wall_s
              << ",includes=capture_inspect_instantiate_initial_upload_sync_update"
              << ",gpu_sample_time=0\n";
  }
  std::cout << std::flush;
#else
  (void)runtimes;
  (void)options;
  (void)direction;
  (void)context;
#endif
}

void exchange_peer_buffers(std::vector<RankRuntime>& runtimes, const Options& options) {
#if FUSE_BENCH_MPI
  enum Buffer { kQkvSource, kQkvDestination, kRouteDone, kOprojInput,
                kInputReady, kCalibrationDone, kQkvWeight, kOprojWeight, kCount };
  struct Handle { cudaIpcMemHandle_t memory{}; int present = 0; };
  using Handles = std::array<Handle, kCount>;
  std::vector<Handles> handles(options.world);
  auto& owner = runtimes[fused_mpi::process_rank];
  const bool share_weights = options.cpu_oracle && options.input_generator == "gpu_philox";
  const void* pointers[kCount] = {
      owner.qkv.local_output, owner.peer_output, owner.route_done, owner.peer_input,
      owner.input_ready, owner.calibration_route_done,
      share_weights ? owner.qkv.rhs_nt : nullptr, share_weights ? owner.oproj.rhs_nt : nullptr};
  for (int field = 0; field < kCount; ++field) {
    auto& handle = handles[fused_mpi::process_rank][field];
    if (!pointers[field]) continue;
    handle.present = 1;
    CUDA_CHECK(cudaIpcGetMemHandle(&handle.memory, const_cast<void*>(pointers[field])));
  }
  fused_mpi::gather_owned(handles);
  for (int peer = 0; peer < options.world; ++peer) {
    if (fused_mpi::owns(peer)) continue;
    std::array<void*, kCount> mapped{};
    for (int field = 0; field < kCount; ++field) {
      if (!handles[peer][field].present) continue;
      CUDA_CHECK(cudaIpcOpenMemHandle(&mapped[field], handles[peer][field].memory,
                                     cudaIpcMemLazyEnablePeerAccess));
      owner.ipc_mappings.push_back(mapped[field]);
    }
    auto& view = runtimes[peer];
    view.qkv.local_output = static_cast<Bf16*>(mapped[kQkvSource]);
    view.peer_output = static_cast<Bf16*>(mapped[kQkvDestination]);
    view.route_done = static_cast<uint32_t*>(mapped[kRouteDone]);
    view.peer_input = static_cast<Bf16*>(mapped[kOprojInput]);
    view.input_ready = static_cast<uint32_t*>(mapped[kInputReady]);
    view.calibration_route_done = static_cast<uint32_t*>(mapped[kCalibrationDone]);
    view.qkv.rhs_nt = static_cast<Bf16*>(mapped[kQkvWeight]);
    view.oproj.rhs_nt = static_cast<Bf16*>(mapped[kOprojWeight]);
  }
  fused_mpi::barrier();
#else
  (void)runtimes;
  (void)options;
#endif
}

std::vector<RankRuntime> create_runtimes(const Options& options) {
  ::alarm(options.timeout_seconds);
  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  if (!fused_mpi::enabled && count < options.world) throw std::runtime_error("not enough visible CUDA devices");
  const bool gpu_inputs = options.input_generator == "gpu_philox";
  const size_t qkv_weight_count = checked_product(options.projection_width(), options.hidden);
  const size_t oproj_weight_count = checked_product(options.hidden, options.q_width());
  const std::string weight_rank = fused_mpi::enabled ? ",rank=" + std::to_string(fused_mpi::process_rank) : "";
  const auto qkv_weight = gpu_inputs || !options.run_qkv() ? std::vector<Bf16>{}
      : make_values(qkv_weight_count, options.seed + 11, "QKV-weight" + weight_rank, 0.02f);
  const auto oproj_weight = gpu_inputs || !options.run_oproj() ? std::vector<Bf16>{}
      : make_values(oproj_weight_count, options.seed + 17, "OProj-weight" + weight_rank, 0.02f);
  std::vector<RankRuntime> runtimes(options.world);
  for (const int rank : fused_mpi::owned_ranks(options.world)) {
    auto& runtime = runtimes[rank];
    runtime.device = fused_mpi::device_for(rank);
    CUDA_CHECK(cudaSetDevice(runtime.device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, runtime.device));
    if (properties.major != 10 || properties.minor != 3 || !properties.cooperativeLaunch ||
        *std::max_element(options.comm_sm_list.begin(), options.comm_sm_list.end()) >=
            properties.multiProcessorCount) {
      throw std::runtime_error("requires cooperative SM103 GPUs and 0 < comm-sm < SM count");
    }
    runtime.sm_count = properties.multiProcessorCount;
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    std::cout << "device,rank=" << rank << ",cuda_device=" << runtime.device << ",name=" << properties.name
              << ",runtime_cc=" << properties.major << '.' << properties.minor
              << ",sms=" << runtime.sm_count << ",free_bytes=" << free_bytes
              << ",total_bytes=" << total_bytes << '\n';
    CUDA_CHECK(cudaStreamCreateWithFlags(&runtime.stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&runtime.start));
    CUDA_CHECK(cudaEventCreate(&runtime.end));
    CUBLAS_CHECK(cublasCreate(&runtime.blas));
    CUBLAS_CHECK(cublasSetStream(runtime.blas, runtime.stream));
    CUBLAS_CHECK(cublasSetMathMode(runtime.blas, CUBLAS_DEFAULT_MATH));
    for (int peer = 0; peer < options.world; ++peer) {
      if (fused_mpi::enabled) break; // CUDA IPC opens enable directed peer access.
      if (peer == rank) continue;
      int accessible = 0;
      CUDA_CHECK(cudaDeviceCanAccessPeer(&accessible, rank, peer));
      if (!accessible) throw std::runtime_error("full directed P2P access is required");
      const cudaError_t status = cudaDeviceEnablePeerAccess(peer, 0);
      if (status == cudaErrorPeerAccessAlreadyEnabled) (void)cudaGetLastError();
      else {
        CUDA_CHECK(status);
        runtime.enabled_peers.push_back(peer);
      }
    }
    const size_t m = options.seq_local;
    if (gpu_inputs) runtime.inputs = allocate<fused_inputs::Scratch>(runtime, 1);
    // Directions own independent resources: an OProj-only run must not allocate
    // or validate an unrelated QKV boundary (e.g. KV heads smaller than CP).
    if (options.run_qkv()) {
      runtime.qkv.route = make_route(options, rank, Direction::kQkv);
      runtime.qkv.gemm.m = options.seq_local;
      runtime.qkv.gemm.n = options.projection_width();
      runtime.qkv.gemm.k = options.hidden;
      apply_schedule(runtime.qkv.gemm, options, Direction::kQkv);
      runtime.qkv.num_comm_ctas = options.comm_sm;
      runtime.qkv.lhs = allocate<Bf16>(runtime, checked_product(m, options.hidden));
      Bf16* qkv_rhs = allocate<Bf16>(runtime, qkv_weight_count);
      if (gpu_inputs) {
        make_gpu_values(runtime, qkv_rhs, qkv_weight_count, options.seed + 11,
                         "QKV-weight,rank=" + std::to_string(rank), 0.02f);
      } else upload(qkv_rhs, qkv_weight, runtime.stream);
      runtime.qkv.rhs_nt = qkv_rhs;
      runtime.qkv.local_output = allocate<Bf16>(runtime, checked_product(m, options.projection_width()));
      runtime.peer_output = allocate<Bf16>(runtime, checked_product(m, options.projection_width()));
      runtime.route_done = allocate<uint32_t>(runtime, options.world * fuse::kReadyFlagStride);
      const auto traits =
#if FUSE_BENCH_MXFP8
          fuse::mxfp8_qkv_cutlass_kernel_traits(options.mxfp8_epilogue_n);
#else
          fuse::qkv_cutlass_kernel_traits(runtime.qkv.gemm);
#endif
      if (traits.block_m <= 0 || traits.block_n <= 0) throw std::runtime_error("invalid QKV tile query");
      const size_t qkv_tiles = checked_product(ceil_div(options.seq_local, traits.block_m),
                                               ceil_div(options.projection_width(), traits.block_n));
      const size_t qkv_flags = checked_product(qkv_tiles, fuse::kReadyFlagStride);
      runtime.qkv_ready_elements = qkv_flags;
      runtime.qkv.ready = allocate<uint32_t>(runtime, qkv_flags);
      runtime.qkv_reference = allocate<Bf16>(runtime, checked_product(m, options.projection_width()));
#if FUSE_BENCH_MXFP8
      runtime.mxfp8_prequantized = options.mxfp8_prequantized;
      runtime.mxfp8_epilogue_n = options.mxfp8_epilogue_n;
      runtime.mxfp8_weight_preparation = options.mxfp8_weight_preparation == "all"
          ? fuse::Mxfp8WeightPreparation::kAllCtas
          : (options.mxfp8_weight_preparation == "comm_warp"
              ? fuse::Mxfp8WeightPreparation::kCommunicationWarps
              : fuse::Mxfp8WeightPreparation::kCommunicationCtas);
      auto& activation = runtime.mxfp8_activation;
      CUDA_CHECK(fuse::gemm_a2a_mxfp8_activation_size(runtime.qkv.gemm,
          &activation.data_bytes, &activation.scale_bytes));
      activation.data = reinterpret_cast<fuse::Fp8E4m3*>(allocate<uint8_t>(runtime, activation.data_bytes));
      activation.scales = allocate<uint8_t>(runtime, activation.scale_bytes);
      CUDA_CHECK(fuse::gemm_a2a_mxfp8_workspace_size(runtime.qkv.gemm, &runtime.mxfp8_workspace_bytes));
      runtime.mxfp8_workspace = allocate<uint8_t>(runtime, runtime.mxfp8_workspace_bytes);
      runtime.mxfp8_reference_a = allocate<Bf16>(runtime, checked_product(m, options.hidden));
      runtime.mxfp8_reference_b = allocate<Bf16>(runtime, qkv_weight_count);
#endif
    }
    if (options.run_oproj()) {
      runtime.oproj.route = make_route(options, rank, Direction::kOproj);
      runtime.oproj.gemm.m = options.seq_local;
      runtime.oproj.gemm.n = options.hidden;
      runtime.oproj.gemm.k = options.q_width();
      apply_schedule(runtime.oproj.gemm, options, Direction::kOproj);
      runtime.oproj.num_comm_ctas = options.comm_sm;
      runtime.peer_input = allocate<Bf16>(runtime, checked_product(m, options.q_width()));
      runtime.input_ready = allocate<uint32_t>(runtime, fuse::kReadyFlagStride);
      runtime.oproj.input_staging = allocate<Bf16>(runtime, checked_product(m, options.q_width()));
      runtime.oproj.rhs_nt = allocate<Bf16>(runtime, oproj_weight_count);
      if (gpu_inputs) {
        make_gpu_values(runtime, runtime.oproj.rhs_nt, oproj_weight_count, options.seed + 17,
                         "OProj-weight,rank=" + std::to_string(rank), 0.02f);
      } else upload(runtime.oproj.rhs_nt, oproj_weight, runtime.stream);
      runtime.oproj.output = allocate<Bf16>(runtime, checked_product(m, options.hidden));
      const int64_t oproj_flags = fuse::a2a_lhs_gemm_ready_elements(runtime.oproj.gemm, runtime.oproj.route);
      if (oproj_flags <= 0) throw std::runtime_error("invalid OProj ready query");
      runtime.oproj_ready_elements = static_cast<size_t>(oproj_flags);
      runtime.oproj.ready = allocate<uint32_t>(runtime, static_cast<size_t>(oproj_flags));
      runtime.oproj_reference = allocate<Bf16>(runtime, checked_product(m, options.hidden));
      runtime.reference_lhs = allocate<Bf16>(runtime, checked_product(m, options.q_width()));
    }
    runtime.validation = allocate<fused_validation::Scratch>(runtime, 1);
    if (options.calibrate || options.oproj_gap_probe || options.mxfp8_service_probe) {
      if (options.run_qkv()) {
        runtime.calibration_qkv_ready = allocate<uint32_t>(runtime, runtime.qkv_ready_elements);
        runtime.calibration_route_done = allocate<uint32_t>(runtime, options.world * fuse::kReadyFlagStride);
      }
      if (options.run_oproj()) {
      runtime.calibration_oproj_ready = allocate<uint32_t>(runtime, runtime.oproj_ready_elements);
      }
    }
#if FUSE_ENABLE_PROFILING
    if (options.profile && (!options.mxfp8_service_probe || rank == 0)) {
      runtime.timeline = allocate<fuse::A2AGemmCtaTimeline>(runtime, runtime.sm_count);
#if FUSE_BENCH_MXFP8
      runtime.mxfp8_probe.quant_capacity = ceil_div(options.projection_width(), 256) * (256 * (options.hidden / 32) / 32);
      runtime.mxfp8_probe.wait_capacity = ceil_div(options.seq_local, 128) * ceil_div(options.projection_width(), 256);
      runtime.mxfp8_probe.quant = allocate<fuse::Mxfp8QuantRecord>(runtime, runtime.mxfp8_probe.quant_capacity);
      runtime.mxfp8_probe.waits = allocate<fuse::Mxfp8WaitRecord>(runtime, runtime.mxfp8_probe.wait_capacity);
#endif
      if (options.qkv_epilogue_probe) {
        runtime.qkv_epilogue = allocate<fuse::detail::QkvEpilogueRecord>(runtime, runtime.sm_count);
      }
      if (options.profile_detail == "full") {
        if (options.profile_direction != "oproj" && !options.qkv_epilogue_probe) {
          CUDA_CHECK(fuse::query_gemm_a2a_route_timeline_capacity(runtime.qkv, &runtime.qkv_route_capacity));
          runtime.qkv_route_timeline = allocate<fuse::QkvRouteTimeline>(runtime, runtime.qkv_route_capacity);
        }
#if FUSE_BENCH_MXFP8
        if (options.mxfp8_service_probe) {
          auto& service = runtime.mxfp8_service;
          service.tile_capacity = runtime.mxfp8_probe.wait_capacity;
          service.tiles = allocate<fuse::Mxfp8ServiceTileRecord>(runtime, service.tile_capacity);
          service.cta_capacity = runtime.sm_count;
          service.ctas = allocate<fuse::Mxfp8ServiceCtaRecord>(runtime, service.cta_capacity);
          service.panel_capacity = ceil_div(options.projection_width(), 256);
          service.panel_release = allocate<uint64_t>(runtime, service.panel_capacity);
          service.panel_release_begin = allocate<uint64_t>(runtime, service.panel_capacity);
          service.weight = runtime.mxfp8_probe;
          service.routes = runtime.qkv_route_timeline;
          service.route_capacity = runtime.qkv_route_capacity;
        }
#endif
        const auto oproj_traits = fuse::cutlass_kernel_traits();
        if (oproj_traits.block_m <= 0 || oproj_traits.block_n <= 0) {
          throw std::runtime_error("invalid OProj tile query");
        }
        const int m_tiles = ceil_div(options.seq_local, oproj_traits.block_m);
        const int n_tiles = ceil_div(options.hidden, oproj_traits.block_n);
        runtime.peer_capacity = std::max(m_tiles * n_tiles, m_tiles * options.world);
        runtime.peer_timeline = allocate<fuse::A2AGemmPeerTimeline>(runtime, runtime.peer_capacity);
        if ((options.oproj_pipeline_probe || options.oproj_gap_probe) && rank == 0) {
          auto& probe = runtime.oproj_pipeline;
          probe.m_tiles = m_tiles;
          probe.n_tiles = n_tiles;
          probe.k_tiles = options.q_width() / oproj_traits.block_k;
          probe.comm_ctas = options.oproj_gap_probe ? 0 : options.comm_sm;
          probe.all_workers = options.oproj_gap_probe;
          probe.compute_ctas = std::min(m_tiles * n_tiles, runtime.sm_count - options.comm_sm);
          probe.swizzle = options.max_swizzle_size;
          const size_t tiles = checked_product(m_tiles, n_tiles);
          const size_t stages = options.oproj_gap_probe ? 0 : checked_product(tiles, probe.k_tiles);
          const size_t bytes = tiles * sizeof(fuse::detail::OprojPipelineRecord) +
              stages * sizeof(fuse::detail::OprojMmaStageRecord);
          if (bytes > (64 << 20)) throw std::runtime_error("OProj pipeline probe exceeds 64 MiB diagnostic limit");
          probe.tiles = allocate<fuse::detail::OprojPipelineRecord>(runtime, tiles);
          if (stages) probe.stages = allocate<fuse::detail::OprojMmaStageRecord>(runtime, stages);
        }
      }
    }
#endif
  }
  finish_all(runtimes);
  exchange_peer_buffers(runtimes, options);
  for (const int rank : fused_mpi::owned_ranks(options.world)) {
    auto& runtime = runtimes[rank];
    for (int peer = 0; peer < options.world; ++peer) {
      runtime.qkv.peer_output[peer] = runtimes[peer].peer_output;
      runtime.qkv.peer_route_done_epoch[peer] = runtimes[peer].route_done;
      runtime.oproj.peer_input[peer] = runtimes[peer].peer_input;
      runtime.oproj.peer_input_ready[peer] = runtimes[peer].input_ready;
    }
  }
  finish_all(runtimes);
  if (gpu_inputs && options.cpu_oracle) {
    for (const int rank : fused_mpi::owned_ranks(options.world)) {
      CUDA_CHECK(cudaSetDevice(runtimes[rank].device));
      auto& runtime = runtimes[rank];
      for (int direction = 0; direction < 2; ++direction) {
        if (direction == 0 ? !options.run_qkv() : !options.run_oproj()) continue;
        const auto* actual = direction == 0 ? runtime.qkv.rhs_nt : runtime.oproj.rhs_nt;
        const auto* expected = direction == 0 ? runtimes[0].qkv.rhs_nt : runtimes[0].oproj.rhs_nt;
        CUDA_CHECK(fused_validation::launch<false>(reinterpret_cast<const uint16_t*>(actual),
            fused_validation::DenseOracle{reinterpret_cast<const uint16_t*>(expected)},
            direction == 0 ? qkv_weight_count : oproj_weight_count,
            runtime.validation, direction, runtime.stream));
      }
    }
    finish_all(runtimes);
    for (const int rank : fused_mpi::owned_ranks(options.world)) {
      CUDA_CHECK(cudaSetDevice(runtimes[rank].device));
      const auto results = download(runtimes[rank].validation->result, 2);
      for (int direction = 0; direction < 2; ++direction) {
        if (direction == 0 ? !options.run_qkv() : !options.run_oproj()) continue;
        const auto& result = results[direction];
        if (result.checked != (direction == 0 ? qkv_weight_count : oproj_weight_count) ||
            result.mismatches || result.nonfinite) {
          throw std::runtime_error("Philox weights differ across ranks");
        }
      }
    }
    fused_mpi::barrier();
    fused_mpi::root_output() << "input_oracle,generator=gpu_philox,check=cross_rank_weights,full_bitwise_match=1\n";
  }
  return runtimes;
}

void set_inputs(std::vector<RankRuntime>& runtimes, const Options& options, uint32_t generation) {
  ::alarm(options.timeout_seconds);
  // Every preceding epoch has completed on ALL ranks before any peer-owned
  // input is overwritten. Explicit stream completion for H2D publication
  // here does not benchmark overlap with an upstream input producer.
  for (const int rank : fused_mpi::owned_ranks(options.world)) {
    CUDA_CHECK(cudaSetDevice(runtimes[rank].device));
    auto& runtime = runtimes[rank];
    const uint32_t seed = options.seed + generation * 100003u + rank * 101u;
    const std::string label = "generation=" + std::to_string(generation) + ",rank=" + std::to_string(rank);
#if FUSE_BENCH_MXFP8
    // Change the BF16 master weight too, with the same seed on all ranks.
    // The second payload therefore detects accidental activation OR weight reuse.
    const auto weight_count = checked_product(options.hidden, options.projection_width());
    const auto weight_seed = options.seed + generation * 100003u + 11u;
    if (options.input_generator == "gpu_philox") {
      make_gpu_values(runtime, const_cast<Bf16*>(runtime.qkv.rhs_nt), weight_count,
          weight_seed, "MXFP8-weight," + label, 0.02f);
    } else {
      upload(const_cast<Bf16*>(runtime.qkv.rhs_nt),
          make_values(weight_count, weight_seed, "MXFP8-weight," + label, 0.02f), runtime.stream);
    }
#endif
    if (options.input_generator == "gpu_philox") {
      const size_t qkv_count = checked_product(options.seq_local, options.hidden);
      const size_t oproj_count = checked_product(options.seq_local, options.q_width());
      if (options.run_qkv()) make_gpu_values(runtime, const_cast<Bf16*>(runtime.qkv.lhs), qkv_count,
                        seed + 1000, "QKV-activation," + label, 0.125f);
      if (options.run_oproj()) make_gpu_values(runtime, runtime.peer_input, oproj_count,
                        seed + 2000, "OProj-activation," + label, 0.125f);
      if (options.cpu_oracle) {
        if (options.run_qkv()) runtime.qkv_input = download(runtime.qkv.lhs, qkv_count);
        if (options.run_oproj()) runtime.oproj_input = download(runtime.peer_input, oproj_count);
        if (generation == 0 && rank == 0) {
          const bool qkv = options.run_qkv();
          auto* input = qkv ? const_cast<Bf16*>(runtime.qkv.lhs) : runtime.peer_input;
          const auto count = qkv ? qkv_count : oproj_count;
          const auto& original = qkv ? runtime.qkv_input : runtime.oproj_input;
          make_gpu_values(runtime, input, count, seed + (qkv ? 1000 : 2000),
                          std::string(qkv ? "QKV-repeat," : "OProj-repeat,") + label, 0.125f);
          const auto repeated = download(input, count);
          if (std::memcmp(repeated.data(), original.data(), checked_bytes<Bf16>(count)) != 0) {
            throw std::runtime_error("Philox same-seed repeat differs");
          }
          std::cout << "input_oracle,generator=gpu_philox,check=same_seed_repeat,full_bitwise_match=1\n";
        }
      }
    } else {
      if (options.run_qkv()) {
      runtime.qkv_input = make_values(checked_product(options.seq_local, options.hidden),
                                      seed + 1000, "QKV-activation," + label, 0.125f);
      upload(const_cast<Bf16*>(runtime.qkv.lhs), runtime.qkv_input, runtime.stream);
      }
      if (options.run_oproj()) {
      runtime.oproj_input = make_values(checked_product(options.seq_local, options.q_width()),
                                        seed + 2000, "OProj-activation," + label, 0.125f);
      upload(runtime.peer_input, runtime.oproj_input, runtime.stream);
      }
    }
    if (options.run_oproj()) {
    runtime.oproj.input_epoch = generation + 1;
    CUDA_CHECK(cudaMemcpyAsync(runtime.input_ready, &runtime.oproj.input_epoch,
                               sizeof(uint32_t), cudaMemcpyHostToDevice, runtime.stream));
    }
  }
  // Complete every payload and input-ready upload before any rank consumes it.
  finish_all(runtimes);
#if FUSE_BENCH_MXFP8
  // Upstream activation adapter: run once per changed payload, outside timing.
  // Full calls below MUST freshly quantize W inside the persistent kernel.
  for (const int rank : fused_mpi::owned_ranks(options.world)) {
    auto& r = runtimes[rank];
    CUDA_CHECK(cudaSetDevice(r.device));
    CUDA_CHECK(fuse::quantize_gemm_a2a_mxfp8_activation(
        r.qkv.gemm, r.qkv.lhs, r.mxfp8_activation, r.stream));
    if (options.mxfp8_prequantized) {
      fuse::Mxfp8GemmA2AParams p{r.qkv, r.mxfp8_workspace, r.mxfp8_workspace_bytes,
                                r.mxfp8_activation, r.mxfp8_weight_preparation, r.mxfp8_epilogue_n};
      p.projection.epoch = 1;
      CUDA_CHECK(fuse::prepare_gemm_a2a_mxfp8(p, r.stream));
    }
  }
  finish_all(runtimes);
#endif
}

void poison_outputs(std::vector<RankRuntime>& runtimes, const Options& options, Direction direction) {
  ::alarm(options.timeout_seconds);
  for (const int rank : fused_mpi::owned_ranks(options.world)) {
    CUDA_CHECK(cudaSetDevice(runtimes[rank].device));
    auto& runtime = runtimes[rank];
    if (direction == Direction::kQkv) {
      const size_t bytes = checked_bytes<Bf16>(checked_product(options.seq_local, options.projection_width()));
      CUDA_CHECK(cudaMemsetAsync(runtime.qkv.local_output, 0xff, bytes, runtime.stream));
      CUDA_CHECK(cudaMemsetAsync(runtime.peer_output, 0xff, bytes, runtime.stream));
    } else {
      CUDA_CHECK(cudaMemsetAsync(runtime.oproj.input_staging, 0xff,
          checked_bytes<Bf16>(checked_product(options.seq_local, options.q_width())), runtime.stream));
      CUDA_CHECK(cudaMemsetAsync(runtime.oproj.output, 0xff,
          checked_bytes<Bf16>(checked_product(options.seq_local, options.hidden)), runtime.stream));
    }
  }
  // Keep cumulative ready/route_done flags. No rank may overwrite another
  // rank's poison before all destination initialization is complete.
  finish_all(runtimes);
}

void select_candidate(std::vector<RankRuntime>& runtimes, const Candidate& candidate,
                        const std::string& context) {
  // Overrides change only between completed candidates; MPI ranks agreed on
  // the complete list. No outstanding launch reads the environment concurrently.
  set_tile_policy(candidate.direction, candidate.tile_policy);
  for (const int rank : fused_mpi::owned_ranks(static_cast<int>(runtimes.size()))) {
    CUDA_CHECK(cudaSetDevice(runtimes[rank].device));
    auto& runtime = runtimes[rank];
    const bool qkv = candidate.direction == Direction::kQkv;
    if (qkv) runtime.qkv.num_comm_ctas = candidate.auto_comm ? 0 : candidate.comm_sm;
    else runtime.oproj.num_comm_ctas = candidate.auto_comm ? 0 : candidate.comm_sm;
    const auto& problem = qkv ? runtime.qkv.gemm : runtime.oproj.gemm;
    const auto traits =
#if FUSE_BENCH_MXFP8
        fuse::mxfp8_qkv_cutlass_kernel_traits(runtime.mxfp8_epilogue_n);
#else
        qkv
        ? fuse::qkv_cutlass_kernel_traits(runtime.qkv.gemm, runtime.qkv.route,
                                          candidate.comm_sm, runtime.sm_count)
        : fuse::cutlass_kernel_traits();
#endif
    if (traits.block_m <= 0 || traits.block_n <= 0 || traits.block_k <= 0) {
      throw std::runtime_error("unsupported candidate or invalid geometry query");
    }
    const auto schedule = schedule_geometry(ceil_div(problem.m, traits.block_m),
                                            ceil_div(problem.n, traits.block_n), problem.max_swizzle_size);
    std::cout << "candidate," << direction_name(candidate.direction) << context
              << ",state=resolved,rank=" << rank << ",tile_m=" << traits.block_m
              << ",tile_n=" << traits.block_n << ",tile_k=" << traits.block_k
              << ",threads=" << traits.threads << ",dynamic_smem=" << traits.dynamic_smem_bytes
              << ",raster=" << effective_raster(problem, candidate.direction)
              << ",max_swizzle_size=" << problem.max_swizzle_size
              << ",effective_swizzle_size=" << schedule.effective_swizzle_size
              << ",padded_m_tiles=" << schedule.padded_m_tiles << ",padded_n_tiles=" << schedule.padded_n_tiles
              << ",scheduled_compute_ctas=" << std::min(schedule.tiles(), int64_t{runtime.sm_count - candidate.comm_sm});
    if (!qkv) std::cout << ",oproj_comm_layout=" << std::getenv("FUSE_SM103_OPROJ_COMM_LAYOUT");
    std::cout << '\n';
  }
  std::cout << std::flush;
}

void resolve_auto_candidates(std::vector<RankRuntime>& runtimes, std::vector<Candidate>& candidates) {
  // Query once per tile/rank before either payload generation. Keep the actual
  // positive budget in result/reference metadata, but leave production params
  // at zero so the real API resolver is exercised. MXFP8 C/Q/R diagnostics use
  // the explicit resolved budget; the existing BF16 copy-reference path is unchanged.
  for (size_t index = 0; index < candidates.size(); ++index) {
    auto& candidate = candidates[index];
    if (!candidate.auto_comm) continue;
    set_tile_policy(candidate.direction, candidate.tile_policy);
    int resolved = 0;
    for (int rank : fused_mpi::owned_ranks(static_cast<int>(runtimes.size()))) {
      const auto& runtime = runtimes[rank];
      CUDA_CHECK(cudaSetDevice(runtime.device));
      const auto begin = std::chrono::steady_clock::now();
#if FUSE_BENCH_MXFP8
      if (candidate.direction != Direction::kQkv) throw std::runtime_error("MXFP8 auto requires QKV");
      const auto query = [&]() { return fuse::recommended_gemm_a2a_mxfp8_comm_ctas(
          runtime.qkv.gemm, runtime.qkv.route, runtime.mxfp8_epilogue_n); };
      const int comm = query();
#else
      const int comm = fuse::recommended_a2a_lhs_gemm_comm_ctas(runtime.oproj.gemm, runtime.oproj.route);
#endif
      const auto first = std::chrono::steady_clock::now();
#if FUSE_BENCH_MXFP8
      const int repeat = query();
#else
      const int repeat = fuse::recommended_a2a_lhs_gemm_comm_ctas(runtime.oproj.gemm, runtime.oproj.route);
#endif
      const auto end = std::chrono::steady_clock::now();
      if (comm <= 0 || comm >= runtime.sm_count || repeat != comm || (resolved && resolved != comm)) {
        throw std::runtime_error("automatic CTA query is unsupported or disagrees across ranks/repeated calls");
      }
      resolved = comm;
      candidate.comm_sm = comm;
      std::cout << "auto_comm," << direction_name(candidate.direction) << candidate_context(index, candidate) << ",rank=" << rank
#if FUSE_BENCH_MXFP8
                << ",model_version=" << fuse::detail::kMxfp8QkvCalibrationVersion
                << ",requested_comm=0,resolved_comm=" << comm
#endif
                << ",launch_comm=0,query_us=" << std::chrono::duration<double, std::micro>(first - begin).count()
                << ",repeat_query_us=" << std::chrono::duration<double, std::micro>(end - first).count() << '\n';
    }
    fused_mpi::agree("auto_comm" + candidate_context(index, candidate));
  }
  std::cout << std::flush;
}

void describe_component(std::vector<RankRuntime>& runtimes, const Options& options,
                          Direction direction, const std::string& context) {
  for (const int rank : fused_mpi::owned_ranks(options.world)) {
    CUDA_CHECK(cudaSetDevice(runtimes[rank].device));
    const auto& runtime = runtimes[rank];
    const bool qkv = direction == Direction::kQkv;
    const auto& problem = qkv ? runtime.qkv.gemm : runtime.oproj.gemm;
    const auto traits =
#if FUSE_BENCH_MXFP8
        fuse::mxfp8_qkv_cutlass_kernel_traits(runtime.mxfp8_epilogue_n);
#else
        qkv
        ? fuse::qkv_cutlass_kernel_traits(problem, runtime.qkv.route, options.comm_sm, runtime.sm_count)
        : fuse::cutlass_kernel_traits();
#endif
    if (traits.block_m <= 0 || traits.block_n <= 0) throw std::runtime_error("invalid component geometry");
    const int budget = runtime.sm_count - options.comm_sm;
    const auto schedule = schedule_geometry(ceil_div(problem.m, traits.block_m),
                                            ceil_div(problem.n, traits.block_n), problem.max_swizzle_size);
    std::cout << "component_resources," << direction_name(direction) << context << ",rank=" << rank
              << ",tile_m=" << traits.block_m << ",tile_n=" << traits.block_n << ",tile_k=" << traits.block_k
              << ",raster=" << effective_raster(problem, direction)
              << ",max_swizzle_size=" << problem.max_swizzle_size
              << ",effective_swizzle_size=" << schedule.effective_swizzle_size
              << ",padded_m_tiles=" << schedule.padded_m_tiles << ",padded_n_tiles=" << schedule.padded_n_tiles
              << ",compute_budget=" << budget
              << ",scheduled_compute_ctas=" << ((options.component == MeasurementComponent::kCopyReference ||
                                                  options.component == MeasurementComponent::kQuantizeReference)
                                                   ? int64_t{0} : std::min(schedule.tiles(), int64_t{budget}))
              << ",scheduled_comm_ctas=" << (options.component == MeasurementComponent::kComputeReference
                                                ? 0 : options.comm_sm)
              << ",production_threads=" << traits.threads
              << ",production_dynamic_smem=" << traits.dynamic_smem_bytes
              << ",reference_resources=" << (options.component == MeasurementComponent::kFused ? "not_applicable" : "unknown");
#if FUSE_BENCH_MXFP8
    std::cout << ",reference_precision=mxfp8,reference_weight_preparation="
              << (options.component == MeasurementComponent::kFused ? "not_applicable" :
                  (options.component == MeasurementComponent::kQuantizeReference ? "inside_timing" : "outside_timing"))
              << ",epilogue_n=" << runtime.mxfp8_epilogue_n
              << ",reference_resource_contract=production_threads_and_dynamic_smem";
#endif
    if (!qkv) std::cout << ",oproj_comm_layout=" << options.oproj_comm_layout;
    std::cout << '\n';
  }
}

void reset_calibration(std::vector<RankRuntime>& runtimes, const Options& options, Direction direction) {
  ::alarm(options.timeout_seconds);
  finish_all(runtimes);
  for (const int rank : fused_mpi::owned_ranks(options.world)) {
    CUDA_CHECK(cudaSetDevice(runtimes[rank].device));
    auto& runtime = runtimes[rank];
    const bool qkv = direction == Direction::kQkv;
    CUDA_CHECK(cudaMemsetAsync(qkv ? runtime.calibration_qkv_ready : runtime.calibration_oproj_ready,
        0, checked_bytes<uint32_t>(qkv ? runtime.qkv_ready_elements : runtime.oproj_ready_elements), runtime.stream));
    if (qkv) CUDA_CHECK(cudaMemsetAsync(runtime.calibration_route_done, 0,
        options.world * fuse::kReadyFlagStride * sizeof(uint32_t), runtime.stream));
  }
  finish_all(runtimes);
}

void prepare_component(std::vector<RankRuntime>& runtimes, const Options& options, Direction direction) {
  ::alarm(options.timeout_seconds);
  const bool compute = options.component == MeasurementComponent::kComputeReference;
  for (const int rank : fused_mpi::owned_ranks(options.world)) {
    CUDA_CHECK(cudaSetDevice(runtimes[rank].device));
    auto& runtime = runtimes[rank];
    Bf16* destination;
    int width;
    if (direction == Direction::kQkv) {
      destination = compute ? runtime.qkv.local_output : runtime.peer_output;
      width = options.projection_width();
      // The copy reference consumes the current candidate's validated local
      // output. Only its peer destinations are poisoned, never that source.
#if FUSE_BENCH_MXFP8
      if (compute) {
        // Independent C measures represented MXFP8 operands, not BF16 GEMM
        // and not weight preparation. Refresh W outside Graph capture/timing;
        // the production F boundary still quantizes W on every epoch.
        fuse::Mxfp8GemmA2AParams params{runtime.qkv, runtime.mxfp8_workspace,
            runtime.mxfp8_workspace_bytes, runtime.mxfp8_activation,
            runtime.mxfp8_weight_preparation, runtime.mxfp8_epilogue_n};
        params.projection.num_comm_ctas = options.comm_sm; // C preparation is explicit, F remains auto.
        CUDA_CHECK(fuse::prepare_gemm_a2a_mxfp8(params, runtime.stream));
      } else if (options.component == MeasurementComponent::kQuantizeReference) {
        // Q must produce every value itself. Its subsequent C-only numerical
        // validation must NOT call prepare(), which would hide missing Q work.
        CUDA_CHECK(cudaMemsetAsync(runtime.mxfp8_workspace, 0xff,
            runtime.mxfp8_workspace_bytes, runtime.stream));
        destination = runtime.qkv.local_output;
      }
#endif
    } else {
      destination = compute ? runtime.oproj.output : runtime.oproj.input_staging;
      width = compute ? options.hidden : options.q_width();
      if (compute) CUDA_CHECK(cudaMemcpyAsync(runtime.oproj.input_staging, runtime.reference_lhs,
          checked_bytes<Bf16>(checked_product(options.seq_local, options.q_width())),
          cudaMemcpyDeviceToDevice, runtime.stream));
    }
    CUDA_CHECK(cudaMemsetAsync(destination, 0xff,
        checked_bytes<Bf16>(checked_product(options.seq_local, width)), runtime.stream));
  }
  finish_all(runtimes);
}

struct HostLaunchTiming {
  // CPU steady-clock offsets from this dispatch, not cross-GPU timestamps.
  std::vector<double> api_begin_us;
  std::vector<double> api_end_us;
  double all_enqueued_us = 0;
};

struct RankLaunch {
  std::vector<RankRuntime>& runtimes;
  Direction direction;
  uint32_t epoch;
  bool profile;
  MeasurementComponent component;
  int reserved_comm_ctas;
  std::vector<double>* launch_us;
  HostLaunchTiming* timing;
  std::chrono::steady_clock::time_point dispatch_begin;
#if FUSE_BENCH_MPI
  bool graph = false;
#endif
#if FUSE_ENABLE_PROFILING
  fuse::detail::HostLaunchRecord* host_stages = nullptr;
  bool qkv_epilogue_probe = false;
#endif
};

// Only the real operation: reusable by Eager enqueue and Graph capture.
// The caller owns the CUDA device/stream; there are no events or MPI calls.
void enqueue_operation(const RankLaunch& job, int rank) {
  auto& runtime = job.runtimes[rank];
  if (job.component != MeasurementComponent::kFused) {
    if (job.direction == Direction::kQkv) {
      auto params = runtime.qkv;
      params.epoch = job.epoch;
      params.ready = runtime.calibration_qkv_ready;
      for (size_t peer = 0; peer < job.runtimes.size(); ++peer) {
        params.peer_route_done_epoch[peer] = job.runtimes[peer].calibration_route_done;
      }
#if FUSE_BENCH_MXFP8
      fuse::Mxfp8GemmA2AParams packed{params, runtime.mxfp8_workspace,
          runtime.mxfp8_workspace_bytes, runtime.mxfp8_activation,
          runtime.mxfp8_weight_preparation, runtime.mxfp8_epilogue_n};
      packed.projection.num_comm_ctas = job.reserved_comm_ctas; // Diagnostic C/Q/R use the resolved budget.
      packed.projection.lhs = nullptr;  // Both C and F use the MXFP8 activation.
      if (job.component == MeasurementComponent::kQuantizeReference) {
        CUDA_CHECK(fuse::launch_gemm_a2a_mxfp8_quantize_reference(packed, runtime.stream));
      } else CUDA_CHECK(job.component == MeasurementComponent::kComputeReference
          ? fuse::launch_gemm_a2a_mxfp8_compute_reference(packed, runtime.stream)
          : fuse::launch_gemm_a2a_mxfp8_copy_reference(packed, runtime.stream));
#else
      if (job.component == MeasurementComponent::kComputeReference) {
        CUDA_CHECK(fuse::launch_batched_cutlass_reference(params, runtime.stream, job.reserved_comm_ctas));
      } else CUDA_CHECK(fuse::launch_gemm_a2a_copy_reference(params, runtime.stream));
#endif
    } else {
      auto params = runtime.oproj;
      params.epoch = job.epoch;
      params.ready = runtime.calibration_oproj_ready;
      if (job.component == MeasurementComponent::kComputeReference) {
#if FUSE_ENABLE_PROFILING
        fuse::detail::OprojPipelineBinding binding(
            job.profile && runtime.oproj_pipeline.tiles ? &runtime.oproj_pipeline : nullptr);
#endif
        CUDA_CHECK(fuse::launch_a2a_gemm_cutlass_reference(params, runtime.stream, job.reserved_comm_ctas));
      } else CUDA_CHECK(fuse::launch_a2a_gemm_copy_reference(params, runtime.stream));
    }
  } else if (job.direction == Direction::kQkv) {
    runtime.qkv.epoch = job.epoch;
#if FUSE_BENCH_MXFP8
    fuse::Mxfp8GemmA2AParams params{runtime.qkv, runtime.mxfp8_workspace,
        runtime.mxfp8_workspace_bytes, runtime.mxfp8_activation, runtime.mxfp8_weight_preparation,
        runtime.mxfp8_epilogue_n};
    params.projection.lhs = nullptr; // Prove the fused input is the MXFP8 view.
#if FUSE_ENABLE_PROFILING
    if (job.profile) CUDA_CHECK(fuse::launch_gemm_a2a_mxfp8_role_telemetry(
        params, runtime.timeline, runtime.sm_count, runtime.stream,
        runtime.mxfp8_probe, runtime.qkv_route_timeline, runtime.qkv_route_capacity));
    else
#endif
    CUDA_CHECK(runtime.mxfp8_prequantized
        ? fuse::launch_gemm_a2a_mxfp8_prequantized(params, runtime.stream)
        : fuse::launch_gemm_a2a_mxfp8_cutlass(params, runtime.stream));
#else
#if FUSE_ENABLE_PROFILING
    if (job.qkv_epilogue_probe) CUDA_CHECK(fuse::detail::launch_qkv_epilogue_telemetry(
        runtime.qkv, runtime.timeline, runtime.sm_count,
        runtime.qkv_epilogue, runtime.sm_count, runtime.stream));
    else if (job.profile && runtime.qkv_route_timeline) CUDA_CHECK(fuse::launch_gemm_a2a_route_telemetry(
        runtime.qkv, runtime.timeline, runtime.sm_count,
        runtime.qkv_route_timeline, runtime.qkv_route_capacity, runtime.stream));
    else if (job.profile) CUDA_CHECK(fuse::launch_gemm_a2a_role_telemetry(
        runtime.qkv, runtime.timeline, runtime.sm_count, runtime.stream));
    else
#endif
    CUDA_CHECK(fuse::launch_gemm_a2a_cutlass(runtime.qkv, runtime.stream));
#endif
  } else {
    runtime.oproj.epoch = job.epoch;
#if FUSE_ENABLE_PROFILING
    if (job.profile) {
      fuse::detail::OprojPipelineBinding binding(
          runtime.oproj_pipeline.tiles ? &runtime.oproj_pipeline : nullptr);
      CUDA_CHECK(fuse::launch_a2a_gemm_cutlass_role_telemetry(
        runtime.oproj, runtime.timeline, runtime.sm_count,
        runtime.peer_timeline, runtime.peer_capacity, runtime.stream));
    }
    else
#endif
    CUDA_CHECK(fuse::launch_a2a_gemm_cutlass(runtime.oproj, runtime.stream));
  }
}

// Shared sequential/parallel seam: enqueue a complete rank event boundary,
// without any GPU wait. Only the owning rank's parameters and records change.
void enqueue_rank(void* context, int rank) {
  const auto& job = *static_cast<RankLaunch*>(context);
  auto& runtime = job.runtimes[rank];
  CUDA_CHECK(cudaSetDevice(runtime.device));
  CUDA_CHECK(cudaEventRecord(runtime.start, runtime.stream));
#if FUSE_ENABLE_PROFILING
  // Install the library's single external TLS sink on this GPU's owning
  // thread. The binding restores it even when CUDA_CHECK throws.
  fuse::detail::HostLaunchBinding host_binding(
      job.host_stages ? &job.host_stages[rank] : nullptr);
#endif
  const bool timed = job.launch_us || job.timing
#if FUSE_ENABLE_PROFILING
      || job.host_stages
#endif
      ;
  const auto launch_begin = timed ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
#if FUSE_ENABLE_PROFILING
  if (job.host_stages) job.host_stages[rank].outer_begin_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(launch_begin.time_since_epoch()).count();
#endif
#if FUSE_BENCH_MPI
  if (job.graph) {
    if (!runtime.graph) throw std::runtime_error("missing candidate Graph binding");
    runtime.graph->launch();
  } else
#endif
  enqueue_operation(job, rank);
  if (timed) {
    const auto launch_end = std::chrono::steady_clock::now();
#if FUSE_ENABLE_PROFILING
    if (job.host_stages) job.host_stages[rank].outer_end_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(launch_end.time_since_epoch()).count();
#endif
    if (job.launch_us) {
      (*job.launch_us)[rank] = std::chrono::duration<double, std::micro>(launch_end - launch_begin).count();
    }
    if (job.timing) {
      job.timing->api_begin_us[rank] =
          std::chrono::duration<double, std::micro>(launch_begin - job.dispatch_begin).count();
      job.timing->api_end_us[rank] =
          std::chrono::duration<double, std::micro>(launch_end - job.dispatch_begin).count();
    }
  }
  CUDA_CHECK(cudaEventRecord(runtime.end, runtime.stream));
}

std::vector<float> run_epoch(std::vector<RankRuntime>& runtimes, const Options& options, Direction direction,
                             uint32_t epoch, bool profile = false,
                             std::vector<double>* launch_us = nullptr, HostLaunchTiming* timing = nullptr
#if FUSE_ENABLE_PROFILING
                             , fuse::detail::HostLaunchRecord* host_stages = nullptr,
                             bool qkv_epilogue_probe = false
#endif
                             ) {
  ::alarm(options.timeout_seconds);
  if (launch_us) launch_us->resize(runtimes.size());
  if (timing) {
    timing->api_begin_us.resize(runtimes.size());
    timing->api_end_us.resize(runtimes.size());
  }
  RankLaunch job{runtimes, direction, epoch, profile, options.component, options.comm_sm, launch_us, timing,
                timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}};
#if FUSE_ENABLE_PROFILING
  job.host_stages = host_stages;
  if (qkv_epilogue_probe && (!profile || direction != Direction::kQkv ||
                            options.component != MeasurementComponent::kFused)) {
    throw std::runtime_error("epilogue probe requires an instrumented QKV epoch");
  }
  job.qkv_epilogue_probe = qkv_epilogue_probe;
#endif
#if FUSE_BENCH_MPI
  job.graph = options.launch == "graph";
  if (job.graph) {
    if (profile) throw std::runtime_error("Graph profiling is not implemented");
    for (const int rank : fused_mpi::owned_ranks(options.world)) {
      auto& runtime = runtimes[rank];
      CUDA_CHECK(cudaSetDevice(runtime.device));
      if (!runtime.graph) throw std::runtime_error("missing candidate Graph binding");
      const auto started = std::chrono::steady_clock::now();
      runtime.graph->prepare(epoch, [&](uint32_t captured_epoch, cudaStream_t stream) {
        if (captured_epoch != job.epoch || stream != runtime.stream) {
          throw std::runtime_error("Graph capture differs from the requested epoch/stream");
        }
        enqueue_operation(job, rank);
        return cudaSuccess;
      });
      if (runtime.graph_prepare_calls == 0) CUDA_CHECK(cudaStreamSynchronize(runtime.stream));
      runtime.graph_prepare_wall_s +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
      ++runtime.graph_prepare_calls;
    }
  }
#endif
  // MPI callers own one device. Every rank reaches this boundary before any
  // rank waits for its kernel, including QKV's device-side peer finalizer.
  fused_mpi::barrier();
  if (options.host_launch == "per_gpu_thread") {
    if (!runtimes.front().launch_team) throw std::runtime_error("missing persistent launch team");
    check_enqueue(runtimes.front().launch_team->dispatch(enqueue_rank, &job));
  } else {
    for (const int rank : fused_mpi::owned_ranks(runtimes.size())) enqueue_rank(&job, rank);
  }
  if (timing) {
    timing->all_enqueued_us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - job.dispatch_begin).count();
  }
  // Every end event belongs to this epoch before any query can observe it.
  // QKV finalize still acquires completion from every source rank.
  wait_all(runtimes);
  std::vector<float> times(runtimes.size());
  for (const int rank : fused_mpi::owned_ranks(static_cast<int>(runtimes.size()))) {
    CUDA_CHECK(cudaSetDevice(runtimes[rank].device));
    CUDA_CHECK(cudaEventElapsedTime(&times[rank], runtimes[rank].start, runtimes[rank].end));
    if (!std::isfinite(times[rank]) || times[rank] <= 0) throw std::runtime_error("invalid CUDA event sample");
  }
  fused_mpi::gather_owned(times);
  return times;
}

void cublas_nt(RankRuntime& runtime, int m, int n, int k,
                const Bf16* lhs, const Bf16* rhs_nt, Bf16* output) {
  const float alpha = 1.0f, beta = 0.0f;
  // Row-major D=A*B_nt^T is column-major D^T=B_nt*A^T.
  CUBLAS_CHECK(cublasGemmEx(runtime.blas, CUBLAS_OP_T, CUBLAS_OP_N,
      n, m, k, &alpha, rhs_nt, CUDA_R_16BF, k, lhs, CUDA_R_16BF, k,
      &beta, output, CUDA_R_16BF, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void prepare_cpu_gather(std::vector<Bf16>& expected, const std::vector<RankRuntime>& runtimes,
                         const Options& options, int rank) {
  expected.resize(checked_product(options.seq_local, options.q_width()));
  const int shard_width = options.q_width() / options.world;
  for (int peer = 0; peer < options.world; ++peer) {
    // MPI peers retain inputs only on device. Row-aligned scratch is bounded
    // by max(1M elements, one shard row), never by the global activation size.
    const int chunk_rows = std::max(1, 1048576 / shard_width);
    for (int row = 0; row < options.seq_local;) {
      int rows = std::min(chunk_rows, options.seq_local - row);
      if (options.causal && row < options.seq_local / 2) rows = std::min(rows, options.seq_local / 2 - row);
      const size_t offset = checked_product(global_sequence(options, rank, row), shard_width);
      const auto copy = fused_mpi::enabled
          ? download(runtimes[peer].peer_input + offset, checked_product(rows, shard_width))
          : std::vector<Bf16>{};
      const Bf16* source = fused_mpi::enabled ? copy.data() : runtimes[peer].oproj_input.data() + offset;
      for (int local = 0; local < rows; ++local) {
        std::copy_n(source + static_cast<size_t>(local) * shard_width, shard_width,
            expected.data() + static_cast<size_t>(row + local) * options.q_width() + peer * shard_width);
      }
      row += rows;
    }
  }
}

void prepare_references(std::vector<RankRuntime>& runtimes, const Options& options) {
  ::alarm(options.timeout_seconds);
  const bool gpu_inputs = options.input_generator == "gpu_philox";
  for (const int rank : fused_mpi::owned_ranks(options.world)) {
    CUDA_CHECK(cudaSetDevice(runtimes[rank].device));
    auto& runtime = runtimes[rank];
    if (options.run_qkv()) {
#if FUSE_BENCH_MXFP8
      fused_mxfp8::reference_operand<<<256, 256, 0, runtime.stream>>>(runtime.qkv.lhs,
          runtime.mxfp8_reference_a, int64_t{options.seq_local} * options.hidden);
      CUDA_CHECK(cudaGetLastError());
      fused_mxfp8::reference_operand<<<256, 256, 0, runtime.stream>>>(runtime.qkv.rhs_nt,
          runtime.mxfp8_reference_b, int64_t{options.projection_width()} * options.hidden);
      CUDA_CHECK(cudaGetLastError());
      cublas_nt(runtime, options.seq_local, options.projection_width(), options.hidden,
          runtime.mxfp8_reference_a, runtime.mxfp8_reference_b, runtime.qkv_reference);
#else
      cublas_nt(runtime, options.seq_local, options.projection_width(), options.hidden,
          runtime.qkv.lhs, runtime.qkv.rhs_nt, runtime.qkv_reference);
#endif
    }
    if (!options.run_oproj()) continue;
    auto& expected = runtime.reference_input;
    if (!gpu_inputs || options.cpu_oracle) {
      prepare_cpu_gather(expected, runtimes, options, rank);
    }
    if (gpu_inputs) {
      fused_inputs::OprojGather mapping{};
      mapping.seq_local = options.seq_local;
      mapping.q_width = options.q_width();
      mapping.world = options.world;
      mapping.destination = rank;
      mapping.causal = options.causal;
      for (int peer = 0; peer < options.world; ++peer) {
        mapping.source[peer] = reinterpret_cast<const uint16_t*>(runtimes[peer].peer_input);
      }
      CUDA_CHECK(fused_inputs::gather(reinterpret_cast<uint16_t*>(runtime.reference_lhs), mapping, runtime.stream));
      if (options.cpu_oracle) {
        CUDA_CHECK(cudaStreamSynchronize(runtime.stream));
        const auto gathered = download(runtime.reference_lhs, expected.size());
        if (std::memcmp(gathered.data(), expected.data(), checked_bytes<Bf16>(expected.size())) != 0) {
          throw std::runtime_error("GPU reference gather disagrees with independent CPU gather");
        }
        std::cout << "input_oracle,generator=gpu_philox,check=oproj_reference_gather,rank=" << rank
                  << ",elements=" << expected.size() << ",full_cpu_bitwise_match=1\n";
      }
    } else upload(runtime.reference_lhs, expected, runtime.stream);
    cublas_nt(runtime, options.seq_local, options.hidden, options.q_width(),
              runtime.reference_lhs, runtime.oproj.rhs_nt, runtime.oproj_reference);
  }
  finish_all(runtimes);
  // Only this generation's independent OProj oracle is needed on the host.
  // Inputs remain on device, and the next generation is regenerated once.
  for (auto& runtime : runtimes) {
    std::vector<Bf16>().swap(runtime.qkv_input);
    std::vector<Bf16>().swap(runtime.oproj_input);
    if (!options.cpu_oracle) std::vector<Bf16>().swap(runtime.reference_input);
  }
}

uint16_t bits(Bf16 value) {
  uint16_t result;
  static_assert(sizeof(result) == sizeof(value));
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

using ValidationResult = std::array<fused_validation::Stats, 2>;

std::vector<ValidationResult> gpu_validation(
    std::vector<RankRuntime>& runtimes, const Options& options, Direction direction) {
  ::alarm(options.timeout_seconds);
  const bool numeric = options.component != MeasurementComponent::kCopyReference;
  const bool route = component_has_route(options.component);
  for (const int rank : fused_mpi::owned_ranks(options.world)) {
    CUDA_CHECK(cudaSetDevice(runtimes[rank].device));
    auto& runtime = runtimes[rank];
    const bool qkv = direction == Direction::kQkv;
    const auto* actual = reinterpret_cast<const uint16_t*>(qkv ? runtime.qkv.local_output : runtime.oproj.output);
    const auto* expected = reinterpret_cast<const uint16_t*>(qkv ? runtime.qkv_reference : runtime.oproj_reference);
    if (numeric) {
      CUDA_CHECK(fused_validation::launch<true>(actual, fused_validation::DenseOracle{expected},
          checked_product(options.seq_local, qkv ? options.projection_width() : options.hidden),
          runtime.validation, 0, runtime.stream));
    }
    if (!route) continue;
    if (qkv) {
      fused_validation::QkvOracle oracle{};
      oracle.seq_local = options.seq_local;
      oracle.q_width = options.q_width();
      oracle.kv_width = options.kv_width();
      oracle.world = options.world;
      oracle.destination = rank;
      for (int source = 0; source < options.world; ++source) {
        oracle.source[source] = reinterpret_cast<const uint16_t*>(runtimes[source].qkv.local_output);
      }
      CUDA_CHECK(fused_validation::launch<false>(reinterpret_cast<const uint16_t*>(runtime.peer_output),
          oracle, checked_product(options.seq_local, options.projection_width()),
          runtime.validation, 1, runtime.stream));
    } else {
      CUDA_CHECK(fused_validation::launch<false>(reinterpret_cast<const uint16_t*>(runtime.oproj.input_staging),
          fused_validation::DenseOracle{reinterpret_cast<const uint16_t*>(runtime.reference_lhs)},
          checked_product(options.seq_local, options.q_width()), runtime.validation, 1, runtime.stream));
    }
  }
  finish_all(runtimes);
  std::vector<ValidationResult> results(options.world,
      {fused_validation::Stats::zero(), fused_validation::Stats::zero()});
  for (const int rank : fused_mpi::owned_ranks(options.world)) {
    CUDA_CHECK(cudaSetDevice(runtimes[rank].device));
    if (numeric && route) {
      CUDA_CHECK(cudaMemcpy(results[rank].data(), runtimes[rank].validation->result,
                            sizeof(ValidationResult), cudaMemcpyDeviceToHost));
      continue;
    }
    for (int check = 0; check < 2; ++check) {
      if ((check == 0 && !numeric) || (check == 1 && !route)) continue;
      CUDA_CHECK(cudaMemcpy(&results[rank][check], runtimes[rank].validation->result + check,
                            sizeof(fused_validation::Stats), cudaMemcpyDeviceToHost));
    }
  }
  fused_mpi::gather_owned(results);
  return results;
}

std::vector<ValidationResult> cpu_validation(
    std::vector<RankRuntime>& runtimes, const Options& options, Direction direction) {
  ::alarm(options.timeout_seconds);
  const size_t m = options.seq_local;
  const bool numeric = options.component != MeasurementComponent::kCopyReference;
  const bool route = component_has_route(options.component);
  std::vector<ValidationResult> results(options.world,
      {fused_validation::Stats::zero(), fused_validation::Stats::zero()});
  std::vector<std::vector<Bf16>> local_outputs(options.world);
  for (const int rank : fused_mpi::owned_ranks(options.world)) {
    CUDA_CHECK(cudaSetDevice(runtimes[rank].device));
    auto& runtime = runtimes[rank];
    if (direction == Direction::kOproj && route) {
      const auto& expected = runtime.reference_input;
      if (expected.size() != checked_product(m, options.q_width())) {
        throw std::runtime_error("OProj reference is not prepared for this shape");
      }
      const auto staging = download(runtime.oproj.input_staging, expected.size());
      for (size_t index = 0; index < expected.size(); ++index) {
        results[rank][1].observe_bits(bits(staging[index]), bits(expected[index]), index);
      }
    }
  }
  for (const int rank : fused_mpi::owned_ranks(options.world)) {
    CUDA_CHECK(cudaSetDevice(runtimes[rank].device));
    auto& runtime = runtimes[rank];
    const bool qkv = direction == Direction::kQkv;
    if (!numeric && !qkv) continue;
    local_outputs[rank] = download(qkv ? runtime.qkv.local_output : runtime.oproj.output,
                                    checked_product(m, qkv ? options.projection_width() : options.hidden));
    if (numeric) {
      const auto expected = download(qkv ? runtime.qkv_reference : runtime.oproj_reference, local_outputs[rank].size());
      for (size_t index = 0; index < expected.size(); ++index) {
        results[rank][0].observe_numeric(bits(local_outputs[rank][index]), bits(expected[index]), index);
      }
    }
  }
  if (direction != Direction::kQkv || !route) {
    fused_mpi::gather_owned(results);
    return results;
  }
  if (fused_mpi::enabled) {
    // CPU oracle is explicitly small-shape only. Read actual IPC source
    // outputs; never substitute cuBLAS references for route correctness.
    for (int source = 0; source < options.world; ++source) {
      if (fused_mpi::owns(source)) continue;
      local_outputs[source] = download(runtimes[source].qkv.local_output,
                                       checked_product(m, options.projection_width()));
    }
  }
  const size_t global_rows = m * options.world;
  for (const int destination : fused_mpi::owned_ranks(options.world)) {
    CUDA_CHECK(cudaSetDevice(runtimes[destination].device));
    const auto routed = download(runtimes[destination].peer_output, checked_product(m, options.projection_width()));
    for (int source = 0; source < options.world; ++source) {
      for (int row = 0; row < options.seq_local; ++row) {
        // FUSE QKV packed outputs stay rank-major even with causal CP. Only
        // OProj's inverse gather interprets the two causal sequence chunks.
        const int global_row = source * options.seq_local + row;
        for (int segment = 0; segment < 3; ++segment) {
          const int global_width = segment == 0 ? options.q_width() : options.kv_width();
          const int local_width = global_width / options.world;
          const int source_base = segment == 0 ? 0 : options.q_width() + (segment == 2 ? options.kv_width() : 0);
          const size_t destination_base = segment == 0 ? 0 : global_rows *
              (options.q_width() / options.world + (segment == 2 ? options.kv_width() / options.world : 0));
          for (int feature = 0; feature < local_width; ++feature) {
            const auto expected = local_outputs[source][static_cast<size_t>(row) * options.projection_width() +
                source_base + destination * local_width + feature];
            const size_t index = destination_base + static_cast<size_t>(global_row) * local_width + feature;
            results[destination][1].observe_bits(bits(routed[index]), bits(expected), index);
          }
        }
      }
    }
  }
  fused_mpi::gather_owned(results);
  return results;
}

void compare_validation_oracles(const std::vector<ValidationResult>& gpu,
                                const std::vector<ValidationResult>& cpu) {
  auto close = [](double a, double b) {
    return a == b || (std::isfinite(a) && std::isfinite(b) &&
        std::abs(a - b) <= 1e-10 * std::max({1.0, std::abs(a), std::abs(b)}));
  };
  for (size_t rank = 0; rank < gpu.size(); ++rank) {
    for (int check = 0; check < 2; ++check) {
      const auto& a = gpu[rank][check];
      const auto& b = cpu[rank][check];
      if (a.checked != b.checked || a.mismatches != b.mismatches || a.nonfinite != b.nonfinite ||
          a.first_index != b.first_index || a.first_actual != b.first_actual || a.first_expected != b.first_expected ||
          !close(a.max_abs, b.max_abs) || !close(a.error_square, b.error_square) ||
          !close(a.reference_square, b.reference_square)) {
        throw std::runtime_error("GPU validation statistics disagree with full CPU oracle");
      }
    }
  }
}

void validate(std::vector<RankRuntime>& runtimes, const Options& options, Direction direction,
                const std::string& context = "") {
#if FUSE_BENCH_MXFP8
  if (options.component == MeasurementComponent::kQuantizeReference) {
    // Read Q's actual packed A/W through the same compute collective, outside
    // the measured Q Graph. No weight preparation here: the independent BF16
    // reconstructed-operand oracle must see any quantizer omissions/errors.
    for (const int rank : fused_mpi::owned_ranks(options.world)) {
      auto& runtime = runtimes[rank];
      CUDA_CHECK(cudaSetDevice(runtime.device));
      fuse::Mxfp8GemmA2AParams packed{runtime.qkv, runtime.mxfp8_workspace,
          runtime.mxfp8_workspace_bytes, runtime.mxfp8_activation,
          runtime.mxfp8_weight_preparation, runtime.mxfp8_epilogue_n};
      packed.projection.num_comm_ctas = options.comm_sm; // Q validation is an explicit C diagnostic.
      packed.projection.lhs = nullptr;
      packed.projection.ready = runtime.calibration_qkv_ready;
      packed.projection.epoch = 1;
      CUDA_CHECK(fuse::launch_gemm_a2a_mxfp8_compute_reference(packed, runtime.stream));
    }
    finish_all(runtimes);
    fused_mpi::root_output() << "quant_validation," << direction_name(direction) << context
        << ",method=represented_operands_gemm,prepare_repeated=0,compute_outside_timing=1\n";
  }
#endif
  const auto results = gpu_validation(runtimes, options, direction);
  if (options.cpu_oracle) {
    compare_validation_oracles(results, cpu_validation(runtimes, options, direction));
    fused_mpi::root_output() << "validation_oracle," << direction_name(direction) << context << ",full_cpu_match=1\n";
  }
  bool failed = false;
  for (int rank = 0; rank < options.world; ++rank) {
    for (int check = 0; check < 2; ++check) {
      if ((check == 0 && options.component == MeasurementComponent::kCopyReference) ||
          (check == 1 && !component_has_route(options.component))) continue;
      const auto& stats = results[rank][check];
      const bool numeric = check == 0, qkv = direction == Direction::kQkv;
      const uint64_t elements = checked_product(options.seq_local,
          qkv ? options.projection_width() : (numeric ? options.hidden : options.q_width()));
      failed = failed || stats.checked != elements || stats.mismatches != 0 || stats.nonfinite != 0;
      fused_mpi::root_output() << (numeric ? "correctness," : "route,") << direction_name(direction) << context
                << ",rank=" << rank << ",validator=gpu_full,elements=" << elements
                << ",checked=" << stats.checked << ",nonfinite=" << stats.nonfinite;
      if (numeric) {
        const double relative_l2 = stats.nonfinite ? std::numeric_limits<double>::infinity()
            : std::sqrt(stats.error_square / std::max(stats.reference_square, 1e-30));
        fused_mpi::root_output() << ",max_abs=" << stats.max_abs << ",relative_l2=" << relative_l2
                  << ",mismatches=" << stats.mismatches << ",atol=0.01,rtol=0.01";
      } else {
        fused_mpi::root_output() << ",bitwise_mismatches=" << stats.mismatches;
      }
      fused_mpi::root_output() << '\n';
      if (stats.mismatches) {
        fused_mpi::root_output() << "validation_error," << direction_name(direction) << context << ",rank=" << rank
                  << ",check=" << (numeric ? "numeric" : "route") << ",index=" << stats.first_index
                  << ",actual=" << fused_validation::bf16_to_float(stats.first_actual)
                  << ",expected=" << fused_validation::bf16_to_float(stats.first_expected)
                  << ",actual_bits=" << stats.first_actual << ",expected_bits=" << stats.first_expected << '\n';
      }
    }
  }
  fused_mpi::root_output() << std::flush;
  if (failed) throw std::runtime_error("full GPU numeric or route validation failed");
}

void validation_self_test(std::vector<RankRuntime>& runtimes, const Options& options,
                           Direction direction, const std::string& context) {
  enum class Fault { kBitFlip, kNaN, kFinite };
  auto inject = [&](Bf16* buffer, uint64_t index, Fault fault, int check, const std::string& label) {
    const bool owner = fused_mpi::owns(0);
    uint16_t* location = owner ? reinterpret_cast<uint16_t*>(buffer) + index : nullptr;
    uint16_t saved = 0;
    if (owner) {
      CUDA_CHECK(cudaSetDevice(runtimes[0].device));
      CUDA_CHECK(cudaMemcpy(&saved, location, sizeof(saved), cudaMemcpyDeviceToHost));
    }
    const uint16_t replacement = fault == Fault::kBitFlip ? static_cast<uint16_t>(saved ^ 1u)
        : (fault == Fault::kNaN ? 0x7fc0u : (saved & 0x8000u ? 0x7f7fu : 0xff7fu));
    auto restore = [&]() {
      if (!owner) return;
      CUDA_CHECK(cudaSetDevice(runtimes[0].device));
      CUDA_CHECK(cudaMemcpy(location, &saved, sizeof(saved), cudaMemcpyHostToDevice));
    };
    try {
      if (owner) CUDA_CHECK(cudaMemcpy(location, &replacement, sizeof(replacement), cudaMemcpyHostToDevice));
      fused_mpi::barrier();
      const auto failed = gpu_validation(runtimes, options, direction);
      compare_validation_oracles(failed, cpu_validation(runtimes, options, direction));
      if (failed[0][check].mismatches == 0) throw std::runtime_error("validator missed injected " + label);
    } catch (...) {
      restore();
      throw;
    }
    restore();
    fused_mpi::barrier();
    const auto clean = gpu_validation(runtimes, options, direction);
    compare_validation_oracles(clean, cpu_validation(runtimes, options, direction));
    for (int rank = 0; rank < options.world; ++rank) {
      for (int kind = 0; kind < 2; ++kind) {
        const auto& stats = clean[rank][kind];
        const uint64_t elements = checked_product(options.seq_local, direction == Direction::kQkv
            ? options.projection_width() : (kind == 0 ? options.hidden : options.q_width()));
        if (stats.checked != elements || stats.mismatches || stats.nonfinite) {
          throw std::runtime_error("validator self-test failed to restore clean data");
        }
      }
    }
    fused_mpi::root_output() << "validation_self_test," << direction_name(direction) << context
              << ",fault=" << label << ",rank=0,index=" << index
              << ",gpu_cpu_detected=1,restored_full_clean=1,diagnostic_only=1\n" << std::flush;
  };
  auto& runtime = runtimes[0];
  if (direction == Direction::kQkv) {
    const uint64_t q = checked_product(options.seq_local, options.q_width());
    const uint64_t kv = checked_product(options.seq_local, options.kv_width());
    inject(runtime.qkv.local_output, 0, Fault::kFinite, 0, "local_finite_error");
    inject(runtime.qkv.local_output, q + 2 * kv - 1, Fault::kNaN, 0, "local_nan");
    for (uint64_t index : {uint64_t{0}, q - 1, q, q + kv - 1, q + kv, q + 2 * kv - 1}) {
      inject(runtime.peer_output, index, Fault::kBitFlip, 1, "route_segment_boundary");
    }
    inject(runtime.peer_output, q + 2 * kv - 1, Fault::kNaN, 1, "route_nan");
  } else {
    inject(runtime.oproj.input_staging, checked_product(options.seq_local, options.q_width()) - 1,
            Fault::kBitFlip, 1, "staging_tail_bit");
    inject(runtime.oproj.input_staging, 0, Fault::kNaN, 1, "staging_nan");
    inject(runtime.oproj.output, checked_product(options.seq_local, options.hidden) - 1,
            Fault::kFinite, 0, "output_tail_finite_error");
    inject(runtime.oproj.output, 0, Fault::kNaN, 0, "output_nan");
  }
}

struct StageTimer {
  const char* stage;
  std::string context;
  std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
  int exceptions = std::uncaught_exceptions();

  ~StageTimer() {
    std::cout << "stage_time,stage=" << stage << context;
    if (fused_mpi::enabled) std::cout << ",process_rank=" << fused_mpi::process_rank;
    std::cout << ",wall_s="
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
              << ",status=" << (std::uncaught_exceptions() > exceptions ? "failed" : "complete")
              << '\n' << std::flush;
  }
};

template <class T>
double percentile(std::vector<T> values, double quantile) {
  std::sort(values.begin(), values.end());
  const double index = quantile * (values.size() - 1);
  const size_t low = static_cast<size_t>(index), high = std::min(low + 1, values.size() - 1);
  return static_cast<double>(values[low]) +
      (static_cast<double>(values[high]) - values[low]) * (index - low);
}

bool stable_window(const std::vector<double>& values) {
  if (values.size() < 3) return false;
  const std::vector<double> recent(values.end() - 3, values.end());
  for (double value : recent) if (!std::isfinite(value) || value <= 0) return false;
  return (*std::max_element(recent.begin(), recent.end()) -
          *std::min_element(recent.begin(), recent.end())) / percentile(recent, 0.5) <= 0.05;
}

double sample_half_drift(const std::vector<float>& samples) {
  const auto middle = samples.begin() + samples.size() / 2;
  const std::vector<float> first(samples.begin(), middle), second(middle, samples.end());
  return std::abs(percentile(first, 0.5) - percentile(second, 0.5)) / percentile(samples, 0.5);
}

void benchmark(std::vector<RankRuntime>& runtimes, const Options& options,
                 Direction direction, uint32_t& epoch, const std::string& context = "") {
  const bool counter_mode = !options.counter_component.empty();
  if (counter_mode && (options.counter_component != component_name(options.component) ||
      options.counter_direction != (direction == Direction::kQkv ? "qkv" : "oproj"))) return;
  auto collect = [&](int count, const char* phase, int round) {
    const uint32_t initial_epoch = epoch;
    std::vector<std::vector<float>> rank_samples;
    rank_samples.reserve(count);
    std::vector<float> samples;
    samples.reserve(count);
    auto print_samples = [&]() {
      for (size_t index = 0; index < rank_samples.size(); ++index) {
        fused_mpi::root_output() << (round < 0 ? "warmup," : "sample,") << direction_name(direction)
                  << context << ",host_launch=" << options.host_launch
                  << ",phase=" << phase << ",round=" << round << ",index=" << index
                  << ",epoch=" << initial_epoch + index + 1 << ",maxrank_ms=" << samples[index];
        for (int rank = 0; rank < options.world; ++rank) {
          fused_mpi::root_output() << ",rank" << rank << "_ms=" << rank_samples[index][rank];
        }
        fused_mpi::root_output() << '\n';
      }
      fused_mpi::root_output() << std::flush;
    };
    try {
      for (int iteration = 0; iteration < count; ++iteration) {
        rank_samples.push_back(run_epoch(runtimes, options, direction, ++epoch));
        const auto& times = rank_samples.back();
        samples.push_back(*std::max_element(times.begin(), times.end()));
      }
    } catch (...) {
      print_samples();
      throw;
    }
    // Preserve rejected rounds and completed samples before C++ exceptions;
    // defer stdout until after timed calls. A watchdog _exit cannot flush here.
    print_samples();
    return samples;
  };

  if (options.quick) {
    // User-requested screening: exactly 1 warmup + 5 samples, no convergence
    // loop, cadence rewarm or noise retries. Full two-payload validation outside
    // this sampler is unchanged. These samples are NOT a formal benchmark.
    const auto warmup = collect(1, "initial", -1);
    const auto samples = collect(5, "measurement", 0);
    fused_mpi::root_output() << "summary," << direction_name(direction) << context
              << ",host_launch=" << options.host_launch
              << ",verification=pending,sampling_mode=quick_1_5,formal_eligible=0"
              << ",warmup=1,samples=5,additional_warmup_calls=0,minimum_warmup_cuda_ms=0"
              << ",converged_all_ranks=0,sample_cadence_warmup=0,selected_round=0"
              << ",warmup_p50_ms=" << warmup.front() << ",warmup_p95_ms=" << warmup.front()
              << ",p50_ms=" << percentile(samples, 0.5) << ",p95_ms=" << percentile(samples, 0.95)
              << ",half_drift=" << sample_half_drift(samples)
              << ",collector=" << (options.launch == "graph" ? "mpi_graph_rank_events_v1" :
                  (fused_mpi::enabled ? "mpi_rank_events_v1" : "per_epoch_rank_events_v3_eventsync"))
              << ",boundary=" << (options.launch == "graph" ? "mpi_graph_maxrank_cudaevent" :
                  (fused_mpi::enabled ? "mpi_eager_maxrank_cudaevent" : "single_process_eager_maxrank_cudaevent"));
    if (options.launch == "graph") {
      fused_mpi::root_output() << ",launch=graph,graph_epoch_mode=recapture_update_v1";
    }
    fused_mpi::root_output() << '\n' << std::flush;
    return;
  }
  const auto initial_warmup = collect(kWarmup, "initial", -1);
  const auto warm_started = std::chrono::steady_clock::now();
  auto warm_wall_seconds = [&]() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - warm_started).count();
  };
  std::vector<double> accumulated_ms(options.world, 0);
  std::vector<std::vector<double>> windows(options.world);
  int count = kWarmup, calls = 0;
  while (true) {
    std::vector<double> elapsed(options.world, 0);
    int window_calls = 0;
    do {
      const auto times = run_epoch(runtimes, options, direction, ++epoch);
      for (int rank = 0; rank < options.world; ++rank) elapsed[rank] += times[rank];
      ++window_calls;
    } while (window_calls < count && !fused_mpi::any(warm_wall_seconds() >= 5));
    calls += window_calls;
    bool converged = true;
    int next_count = 1000;
    for (int rank = 0; rank < options.world; ++rank) {
      accumulated_ms[rank] += elapsed[rank];
      windows[rank].push_back(elapsed[rank] / window_calls);
      const bool ready = accumulated_ms[rank] >= 100 && stable_window(windows[rank]);
      converged = converged && ready;
      next_count = std::min(next_count, static_cast<int>(std::clamp(
          std::ceil(20 / std::max(windows[rank].back(), 0.001)), 10.0, 1000.0)));
      fused_mpi::root_output() << "warmup," << direction_name(direction) << context << ",phase=convergence,rank=" << rank
                << ",host_launch=" << options.host_launch
                << ",window=" << windows[rank].size() - 1 << ",calls=" << window_calls
                << ",epoch=" << epoch << ",ms_per_call=" << windows[rank].back()
                << ",accumulated_cuda_ms=" << accumulated_ms[rank] << ",ready=" << ready << '\n';
    }
    fused_mpi::root_output() << std::flush;
    if (converged) break;
    if (fused_mpi::any(warm_wall_seconds() >= 5)) throw std::runtime_error("warmup did not converge within 5s");
    count = next_count;
  }
  const double warm_wall_s = warm_wall_seconds();
  // Use the same synchronized per-epoch cadence as sampling. Unlike the Python
  // distributed collector, each window here sums individual rank CUDA events.
  collect(kWarmup, "sample_cadence", -1);
  if (counter_mode) {
    // app-range replays the application, not individual kernels. All peers
    // remain concurrent; only device 0's context is selected by the profiler.
    // No poisoning, validation or allocation belongs inside this range.
    if (fused_mpi::root()) {
      CUDA_CHECK(cudaSetDevice(runtimes.front().device));
      nvtxRangePushA("fuse_communication_counters");
      CUDA_CHECK(cudaProfilerStart());
    }
    const auto times = run_epoch(runtimes, options, direction, ++epoch);
    if (fused_mpi::root()) {
      CUDA_CHECK(cudaSetDevice(runtimes.front().device));
      CUDA_CHECK(cudaProfilerStop());
      nvtxRangePop();
    }
    validate(runtimes, options, direction, context + ",validation_phase=counter_post");
    fused_mpi::root_output() << "counter_epoch," << direction_name(direction) << context
              << ",epoch=" << epoch << ",profiled_device=0,active_devices=" << options.world
              << ",counter_component=" << options.counter_component
              << ",warmup=" << kWarmup << ",additional_warmup_calls=" << calls
              << ",minimum_warmup_cuda_ms=100,converged_all_ranks=1,selected_ranges=1"
              << ",diagnostic_only=1,performance_accepted=0,maxrank_event_ms="
              << *std::max_element(times.begin(), times.end()) << '\n' << std::flush;
    return;
  }
  for (int round = 0; round < 3; ++round) {
    fused_mpi::root_output() << "sample," << direction_name(direction) << context << ",round=" << round
              << ",host_launch=" << options.host_launch
              << ",state=collecting,count=" << kSamples << '\n' << std::flush;
    const auto samples = collect(kSamples, "measurement", round);
    const double drift = sample_half_drift(samples);
    fused_mpi::root_output() << "sample," << direction_name(direction) << context << ",round=" << round
              << ",host_launch=" << options.host_launch
              << ",state=complete,half_drift=" << drift << ",stable_5pct=" << (drift <= 0.05)
              << '\n' << std::flush;
    if (drift > 0.05) continue;
    fused_mpi::root_output() << "summary," << direction_name(direction) << context
              << ",host_launch=" << options.host_launch
              << ",verification=pending,warmup=" << kWarmup
              << ",additional_warmup_calls=" << calls << ",minimum_warmup_cuda_ms=100"
              << ",warmup_wall_s=" << warm_wall_s << ",converged_all_ranks=1"
              << ",sample_cadence_warmup=" << kWarmup << ",samples=" << kSamples
              << ",selected_round=" << round
              << ",warmup_p50_ms=" << percentile(initial_warmup, 0.5)
              << ",warmup_p95_ms=" << percentile(initial_warmup, 0.95)
              << ",p50_ms=" << percentile(samples, 0.5) << ",p95_ms=" << percentile(samples, 0.95)
              << ",half_drift=" << drift << ",stable_5pct=1"
              << ",collector=" << (options.launch == "graph" ? "mpi_graph_rank_events_v1" :
                  (fused_mpi::enabled ? "mpi_rank_events_v1" : "per_epoch_rank_events_v3_eventsync"))
              << ",boundary=" << (options.launch == "graph" ? "mpi_graph_maxrank_cudaevent" :
                  (fused_mpi::enabled ? "mpi_eager_maxrank_cudaevent" : "single_process_eager_maxrank_cudaevent"));
    if (options.launch == "graph") {
      fused_mpi::root_output() << ",launch=graph,graph_epoch_mode=recapture_update_v1";
    }
    fused_mpi::root_output() << '\n' << std::flush;
    return; // Select the first stable round, never the fastest round.
  }
  throw std::runtime_error("measurement drift exceeds 5% in all 3 rounds; raw rank samples retained");
}

#if FUSE_ENABLE_PROFILING
// This is a bounded diagnostic, separate from accepted benchmark samples.
// One final CTA record per GPU is retained, not a per-tile/per-epoch trace.
void profile_qkv_epilogue(std::vector<RankRuntime>& runtimes, const Options& options,
                          uint32_t& epoch) {
  if (!options.qkv_epilogue_probe || !options.profile || options.profile_detail != "cta") {
    throw std::runtime_error("QKV epilogue collection requires its explicit diagnostic option");
  }
  constexpr const char* modes[] = {"production", "role_telemetry", "epilogue_telemetry"};
  using Resources = fuse::detail::QkvEpilogueResources;
  for (int rank = 0; rank < options.world; ++rank) {
    auto& runtime = runtimes[rank];
    CUDA_CHECK(cudaSetDevice(runtime.device));
    Resources resources{};
    CUDA_CHECK(fuse::detail::query_qkv_epilogue_resources(runtime.qkv, &resources));
    const cudaFuncAttributes attributes[] = {
        resources.production, resources.role_telemetry, resources.epilogue_telemetry};
    if (resources.dynamic_smem_bytes <= 0 || resources.tile_m != 128 ||
        resources.tile_n != 256 || resources.tile_k != 64 || resources.cluster_ctas != 1) {
      throw std::runtime_error("unexpected QKV epilogue diagnostic resources");
    }
    for (int mode = 0; mode < 3; ++mode) {
      const auto& attr = attributes[mode];
      if (attr.numRegs <= 0 || attr.maxThreadsPerBlock <= 0) {
        throw std::runtime_error("empty QKV epilogue function attributes");
      }
      std::cout << "epilogue_resources,rank=" << rank << ",kind=" << modes[mode]
                << ",schema=" << Resources::kSchema << ",clock=" << Resources::kClock
                << ",clock_unit=" << Resources::kClockUnit
                << ",store_interval=" << Resources::kStoreInterval
                << ",drain_interval=" << Resources::kDrainInterval
                << ",role_timestamp_join=" << (mode == 2 ? Resources::kRoleTimestampJoin
                    : mode == 1 ? "legacy_bar_sync" : "none")
                << ",record_bytes=" << sizeof(fuse::detail::QkvEpilogueRecord)
                << ",regs=" << attr.numRegs << ",local_bytes=" << attr.localSizeBytes
                << ",static_smem=" << attr.sharedSizeBytes
                << ",dynamic_smem=" << resources.dynamic_smem_bytes
                << ",max_threads=" << attr.maxThreadsPerBlock << ",cluster_ctas=1"
                << ",tile_m=128,tile_n=256,tile_k=64,performance_accepted=0\n";
    }
  }
  auto clear = [&]() {
    for (auto& runtime : runtimes) {
      CUDA_CHECK(cudaSetDevice(runtime.device));
      CUDA_CHECK(cudaMemsetAsync(runtime.timeline, 0,
          runtime.sm_count * sizeof(*runtime.timeline), runtime.stream));
      CUDA_CHECK(cudaMemsetAsync(runtime.qkv_epilogue, 0,
          runtime.sm_count * sizeof(*runtime.qkv_epilogue), runtime.stream));
    }
    finish_all(runtimes); // Do not clear ready flags or reset epochs.
  };
  auto launch = [&](int mode) {
    return run_epoch(runtimes, options, Direction::kQkv, ++epoch, mode != 0,
                     nullptr, nullptr, nullptr, mode == 2);
  };
  // All three paths get their own warmup. Alternate the forward/reverse order
  // during collection, retaining every sample instead of the fastest repeat.
  for (int iteration = 0; iteration < kWarmup; ++iteration) {
    for (int mode = 0; mode < 3; ++mode) { clear(); launch(mode); }
  }
  struct Sample {
    std::array<std::vector<float>, 3> times;
    std::array<uint32_t, 3> epochs{};
  };
  std::array<Sample, kSamples> collected;
  for (int sample = 0; sample < kSamples; ++sample) {
    for (int slot = 0; slot < 3; ++slot) {
      const int mode = sample % 2 ? 2 - slot : slot;
      clear();
      if (sample == kSamples - 1) poison_outputs(runtimes, options, Direction::kQkv);
      collected[sample].times[mode] = launch(mode);
      collected[sample].epochs[mode] = epoch;
      // Validate each mode's final timed output before the next mode replaces
      // it. This is full numeric/route validation, not a rerun of that kernel.
      if (sample == kSamples - 1) {
        validate(runtimes, options, Direction::kQkv,
                 std::string(",profile_phase=epilogue_") + modes[mode]);
      }
    }
  }
  clear();
  poison_outputs(runtimes, options, Direction::kQkv);
  launch(2); // One explicit final epoch owns the bounded record below.
  validate(runtimes, options, Direction::kQkv, ",profile_phase=epilogue_record");
  // Do not put terminal I/O between sampled modes. The final sample per mode
  // is explicitly poisoned outside its event boundary; retain that marker.
  for (int sample = 0; sample < kSamples; ++sample) {
    for (int slot = 0; slot < 3; ++slot) {
      const int mode = sample % 2 ? 2 - slot : slot;
      for (int rank = 0; rank < options.world; ++rank) {
        std::cout << "epilogue_sample,rank=" << rank << ",sample=" << sample
                  << ",kind=" << modes[mode] << ",epoch=" << collected[sample].epochs[mode]
                  << ",warmup=" << kWarmup << ",samples=" << kSamples
                  << ",host_launch=" << options.host_launch << ",launch=eager,process_layout=single_process"
                  << ",performance_accepted=0,final_sample_poisoned=" << (sample == kSamples - 1)
                  << ",event_ms=" << collected[sample].times[mode][rank] << '\n';
      }
    }
  }
  for (int rank = 0; rank < options.world; ++rank) {
    auto& runtime = runtimes[rank];
    CUDA_CHECK(cudaSetDevice(runtime.device));
    const auto records = download(runtime.qkv_epilogue, runtime.sm_count);
    const auto timeline = download(runtime.timeline, runtime.sm_count);
    const int m_tiles = ceil_div(options.seq_local, 128);
    const int n_tiles = ceil_div(options.projection_width(), 256);
    const int total_tiles = m_tiles * n_tiles;
    const int compute_ctas = std::min(total_tiles, runtime.sm_count - options.comm_sm);
    uint64_t observed_tiles = 0;
    for (int cta = 0; cta < runtime.sm_count; ++cta) {
      const auto& record = records[cta];
      const auto& event = timeline[cta];
      const bool compute = cta >= options.comm_sm && cta < options.comm_sm + compute_ctas;
      if (!compute) {
        const fuse::detail::QkvEpilogueRecord zero{};
        if (std::memcmp(&record, &zero, sizeof(record)) != 0) {
          throw std::runtime_error("unexpected QKV epilogue record outside compute CTAs");
        }
        continue;
      }
      const uint64_t expected = ceil_div(total_tiles - (cta - options.comm_sm), compute_ctas);
      if (record.tile_count != expected || record.epoch != epoch ||
          record.first_m_tile < 0 || record.first_m_tile >= m_tiles ||
          record.first_n_tile < 0 || record.first_n_tile >= n_tiles || record.first_batch != 0 ||
          !event.start || record.first_store_begin < event.start ||
          record.first_store_end < record.first_store_begin ||
          record.first_drain_end < record.first_store_end ||
          record.first_ready_after < record.first_drain_end ||
          record.last_ready_after < record.first_ready_after ||
          event.role_done < record.last_ready_after || event.end < event.role_done ||
          record.store_ns_max > record.store_ns_sum || record.drain_ns_max > record.drain_ns_sum ||
          record.first_store_end - record.first_store_begin > record.store_ns_max ||
          record.first_drain_end - record.first_store_end > record.drain_ns_max ||
          record.store_ns_sum > event.role_done - event.start ||
          record.drain_ns_sum > event.role_done - event.start - record.store_ns_sum ||
          record.store_ns_sum / expected + (record.store_ns_sum % expected != 0) > record.store_ns_max ||
          record.drain_ns_sum / expected + (record.drain_ns_sum % expected != 0) > record.drain_ns_max) {
        std::cerr << "epilogue_record_rejected,rank=" << rank << ",cta=" << cta
                  << ",expected_tiles=" << expected << ",tile_count=" << record.tile_count
                  << ",expected_epoch=" << epoch << ",epoch=" << record.epoch
                  << ",m=" << record.first_m_tile << ",n=" << record.first_n_tile << ",batch=" << record.first_batch
                  << ",start=" << event.start << ",role_done=" << event.role_done << ",end=" << event.end
                  << ",t0=" << record.first_store_begin << ",t1=" << record.first_store_end
                  << ",t2=" << record.first_drain_end << ",t3=" << record.first_ready_after
                  << ",last=" << record.last_ready_after << ",store_sum=" << record.store_ns_sum
                  << ",store_max=" << record.store_ns_max << ",drain_sum=" << record.drain_ns_sum
                  << ",drain_max=" << record.drain_ns_max << '\n';
        throw std::runtime_error("incomplete or unordered QKV epilogue CTA record");
      }
      observed_tiles += record.tile_count;
      std::cout << "epilogue_cta,rank=" << rank << ",cta=" << cta << ",epoch=" << record.epoch
                << ",schema=" << Resources::kSchema << ",clock=globaltimer,clock_unit=ns"
                << ",tile_count=" << record.tile_count << ",first_m_tile=" << record.first_m_tile
                << ",first_n_tile=" << record.first_n_tile << ",first_batch=" << record.first_batch
                << ",store_ns_sum=" << record.store_ns_sum << ",store_ns_max=" << record.store_ns_max
                << ",drain_ns_sum=" << record.drain_ns_sum << ",drain_ns_max=" << record.drain_ns_max
                << ",first_store_begin=" << record.first_store_begin << ",first_store_end=" << record.first_store_end
                << ",first_drain_end=" << record.first_drain_end << ",first_ready_after=" << record.first_ready_after
                << ",last_ready_after=" << record.last_ready_after << ",cta_start=" << event.start
                << ",cta_role_done=" << event.role_done << ",cta_end=" << event.end
                << ",performance_accepted=0\n";
    }
    if (observed_tiles != static_cast<uint64_t>(total_tiles)) {
      throw std::runtime_error("QKV epilogue records do not cover the physical tile grid");
    }
  }
  std::cout << std::flush;
}

void profile_host_stages(std::vector<RankRuntime>& runtimes, const Options& options,
                         Direction direction, uint32_t& epoch) {
  if (!options.profile || options.component != MeasurementComponent::kFused) {
    throw std::runtime_error("host stages require explicit fused profiling");
  }
  struct Sample {
    std::array<fuse::detail::HostLaunchRecord, fuse::kMaxWorldSize> ranks{};
    HostLaunchTiming dispatch;
    std::vector<double> launch_us;
    std::vector<float> event_ms;
    uint32_t epoch = 0;
  };
  std::array<Sample, kSamples> samples{};
  for (auto& sample : samples) {
    sample.launch_us.resize(options.world);
    sample.dispatch.api_begin_us.resize(options.world);
    sample.dispatch.api_end_us.resize(options.world);
  }
  // Keep the normal sampler and its accepted performance records untouched.
  // These 10+50 diagnostic epochs use the production kernel, not 50 large
  // instrumented CTA/peer traces; cumulative ready epochs remain contiguous.
  for (int iteration = 0; iteration < kWarmup; ++iteration) {
    run_epoch(runtimes, options, direction, ++epoch);
  }
  for (auto& sample : samples) {
    sample.epoch = ++epoch;
    sample.event_ms = run_epoch(runtimes, options, direction, sample.epoch, false,
        &sample.launch_us, &sample.dispatch, sample.ranks.data());
  }
  constexpr const char* names[] = {
      "policy_device_validation", "communication_prepare", "arguments",
      "implement_workspace", "lower_parameters", "launch_setup", "cuda_enqueue"};
  bool valid = true;
  for (int index = 0; index < kSamples; ++index) {
    const auto& sample = samples[index];
    const auto begin = std::minmax_element(sample.dispatch.api_begin_us.begin(), sample.dispatch.api_begin_us.end());
    const auto end = std::minmax_element(sample.dispatch.api_end_us.begin(), sample.dispatch.api_end_us.end());
    for (int rank = 0; rank < options.world; ++rank) {
      const auto& record = sample.ranks[rank];
      bool ordered = record.outer_begin_ns <= record.stamp_ns[0] &&
          record.stamp_ns[7] <= record.api_return_ns && record.api_return_ns <= record.outer_end_ns;
      for (int stage = 0; stage < 7; ++stage) {
        ordered = ordered && record.stamp_ns[stage] <= record.stamp_ns[stage + 1];
      }
      const bool complete = record.stage_mask == 255 && record.status == cudaSuccess &&
          !record.protocol_error && ordered;
      valid = valid && complete;
      // Convert operands before subtracting so incomplete records cannot wrap
      // uint64_t and masquerade as a very large positive duration.
      auto elapsed_us = [](uint64_t stop, uint64_t start) {
        return (static_cast<int64_t>(stop) - static_cast<int64_t>(start)) * 0.001;
      };
      std::cout << "host_stage," << direction_name(direction) << ",sample=" << index
                << ",rank=" << rank << ",epoch=" << sample.epoch
                << ",host_launch=" << options.host_launch << ",warmup=" << kWarmup << ",samples=" << kSamples
                << ",kind=production_kernel_diagnostic,performance_accepted=0,clock=steady_clock"
                << ",profile_phase=host_stages,profile_schema=host_stages_v1"
                << ",profile_detail=" << options.profile_detail
                << ",clock_overhead_subtracted=0"
                << ",api_boundary=after_start_event_before_end_event"
                << ",stage_begin=policy_entry,descriptor_timing=subset_of_communication_prepare"
                << ",status=" << record.status << ",stage_mask=" << record.stage_mask
                << ",complete=" << complete << ",api_us=" << sample.launch_us[rank]
                << ",api_prefix_us=" << elapsed_us(record.stamp_ns[0], record.outer_begin_ns)
                << ",library_return_us=" << elapsed_us(record.api_return_ns, record.stamp_ns[7])
                << ",api_suffix_us=" << elapsed_us(record.outer_end_ns, record.api_return_ns);
      for (int stage = 0; stage < 7; ++stage) {
        std::cout << ',' << names[stage] << "_us="
                  << elapsed_us(record.stamp_ns[stage + 1], record.stamp_ns[stage]);
      }
      std::cout << ",local_descriptor_us=" << record.descriptor_ns[0] * 0.001
                << ",peer_descriptors_us=" << record.descriptor_ns[1] * 0.001
                << ",local_descriptor_count=" << record.descriptor_count[0]
                << ",peer_descriptor_count=" << record.descriptor_count[1]
                << ",api_begin_us=" << sample.dispatch.api_begin_us[rank]
                << ",api_end_us=" << sample.dispatch.api_end_us[rank]
                << ",api_begin_skew_us=" << *begin.second - *begin.first
                << ",api_end_skew_us=" << *end.second - *end.first
                << ",all_enqueued_us=" << sample.dispatch.all_enqueued_us
                << ",event_us=" << sample.event_ms[rank] * 1000.0 << '\n';
    }
  }
  std::cout << std::flush;
  if (!valid) throw std::runtime_error("incomplete or unordered host-stage diagnostic records");
}

void profile_compute_gaps(std::vector<RankRuntime>& runtimes, const Options& options) {
  // This is independent C, not F: materialize the whole lhs before launching.
  // Retain the production tile/scheduler and reduced compute CTA budget.
  Options reference = options;
  reference.component = MeasurementComponent::kComputeReference;
  uint32_t epoch = 0;
  prepare_component(runtimes, reference, Direction::kOproj);
  run_epoch(runtimes, reference, Direction::kOproj, ++epoch);
  validate(runtimes, reference, Direction::kOproj, ",gap_phase=pre");
  benchmark(runtimes, reference, Direction::kOproj, epoch, ",component=compute_reference,gap_phase=warm");
  std::vector<float> plain, observed;
  for (bool instrumented : {false, true}) {
    double warm_ms = 0;
    const auto begun = std::chrono::steady_clock::now();
    for (int i = 0; i < kWarmup || warm_ms < 100; ++i) {
      const auto times = run_epoch(runtimes, reference, Direction::kOproj, ++epoch, instrumented);
      warm_ms += times[0];
      if (std::chrono::duration<double>(std::chrono::steady_clock::now() - begun).count() > 5)
        throw std::runtime_error("gap probe warmup timeout");
    }
    auto& samples = instrumented ? observed : plain;
    for (int i = 0; i < 50; ++i) {
      const auto times = run_epoch(runtimes, reference, Direction::kOproj, ++epoch, instrumented);
      samples.push_back(times[0]);
      std::cout << "profile_gemm_gap_sample,component=compute_reference,rank=0,instrumented="
                << instrumented << ",sample=" << i << ",event_ms=" << times[0] << '\n';
    }
    validate(runtimes, reference, Direction::kOproj,
             instrumented ? ",gap_phase=instrumented" : ",gap_phase=plain");
  }
  CUDA_CHECK(cudaSetDevice(runtimes[0].device));
  const auto& probe = runtimes[0].oproj_pipeline;
  const auto records = download(probe.tiles, probe.m_tiles * probe.n_tiles);
  size_t visited = 0;
  // Never sum parallel CTA gaps into E2E time. Save each chain independently;
  // its signed gaps include overlap (negative values) and observer latency.
  // Even the last-finishing chain's gap is not a measured a-b contribution:
  // cuBLASLt has its own gaps, and removing one chain's waits can change the
  // critical path. The analysis must label any zero-Lt-gap estimate explicitly.
  for (int cta = 0; cta < probe.compute_ctas; ++cta) {
    std::vector<const fuse::detail::OprojPipelineRecord*> tiles;
    for (const auto& record : records) if (record.first_input && record.cta == cta) tiles.push_back(&record);
    if (tiles.empty()) throw std::runtime_error("missing compute CTA gap records");
    std::sort(tiles.begin(), tiles.end(), [](auto lhs, auto rhs) { return lhs->first_input < rhs->first_input; });
    int64_t service = 0, gaps = 0, positive = 0;
    for (size_t i = 0; i < tiles.size(); ++i) {
      const auto& tile = *tiles[i];
      if (tile.acc_wait_end < tile.first_input) throw std::runtime_error("invalid tile observation order");
      service += tile.acc_wait_end - tile.first_input;
      if (i) {
        const int64_t gap = static_cast<int64_t>(tile.first_input) - static_cast<int64_t>(tiles[i-1]->acc_wait_end);
        gaps += gap;
        positive += std::max(int64_t{0}, gap);
      }
    }
    const auto span = tiles.back()->acc_wait_end - tiles.front()->first_input;
    if (service + gaps != static_cast<int64_t>(span)) throw std::runtime_error("gap timeline does not close");
    std::cout << "profile_gemm_gap,component=compute_reference,rank=0,cta=" << cta
              << ",tiles=" << tiles.size() << ",first_input_ns=" << tiles.front()->first_input
              << ",last_completion_ns=" << tiles.back()->acc_wait_end
              << ",service_ns=" << service << ",signed_gap_ns=" << gaps
              << ",positive_gap_ns=" << positive << ",span_ns=" << span
              << ",trace_event_ms=" << observed.back() << '\n';
    visited += tiles.size();
  }
  if (visited != records.size()) throw std::runtime_error("incomplete logical tile coverage");
}

void profile(std::vector<RankRuntime>& runtimes, const Options& options, Direction direction, uint32_t& epoch) {
  ::alarm(options.timeout_seconds);
  auto clear_diagnostics = [&]() {
    for (const int rank : fused_mpi::owned_ranks(options.world)) {
      CUDA_CHECK(cudaSetDevice(runtimes[rank].device));
      auto& runtime = runtimes[rank];
      CUDA_CHECK(cudaMemsetAsync(runtime.timeline, 0, runtime.sm_count * sizeof(*runtime.timeline), runtime.stream));
#if FUSE_BENCH_MXFP8
      CUDA_CHECK(cudaMemsetAsync(runtime.mxfp8_probe.quant, 0,
          runtime.mxfp8_probe.quant_capacity * sizeof(fuse::Mxfp8QuantRecord), runtime.stream));
      CUDA_CHECK(cudaMemsetAsync(runtime.mxfp8_probe.waits, 0,
          runtime.mxfp8_probe.wait_capacity * sizeof(fuse::Mxfp8WaitRecord), runtime.stream));
#endif
      if (runtime.qkv_route_timeline) {
        CUDA_CHECK(cudaMemsetAsync(runtime.qkv_route_timeline, 0,
            runtime.qkv_route_capacity * sizeof(*runtime.qkv_route_timeline), runtime.stream));
      }
      if (options.profile_detail == "full") {
        CUDA_CHECK(cudaMemsetAsync(runtime.peer_timeline, 0, runtime.peer_capacity * sizeof(*runtime.peer_timeline), runtime.stream));
      }
      if (runtime.oproj_pipeline.tiles) {
        const auto& probe = runtime.oproj_pipeline;
        const size_t count = static_cast<size_t>(probe.m_tiles) * probe.n_tiles;
        CUDA_CHECK(cudaMemsetAsync(probe.tiles, 0, count * sizeof(*probe.tiles), runtime.stream));
        CUDA_CHECK(cudaMemsetAsync(probe.stages, 0, count * probe.k_tiles * sizeof(*probe.stages), runtime.stream));
      }
    }
    finish_all(runtimes); // Never reset cumulative ready flags.
  };
  for (const int rank : fused_mpi::owned_ranks(options.world)) {
    CUDA_CHECK(cudaSetDevice(runtimes[rank].device));
    if (direction == Direction::kOproj) {
      fuse::A2AGemmRoleResources resources{};
      CUDA_CHECK(fuse::query_a2a_gemm_role_resources(&resources));
      std::cout << "profile_resources,rank=" << rank << ",threads=" << resources.threads_per_cta
                << ",profile_detail=" << options.profile_detail
                << ",host_launch=" << options.host_launch
                << ",regs=" << resources.registers_per_thread << ",telemetry_regs=" << resources.telemetry_registers_per_thread
                << ",static_smem=" << resources.static_smem_bytes << ",dynamic_smem=" << resources.dynamic_smem_bytes
                << ",cluster_ctas=" << resources.cluster_ctas << ",comm_warps=" << resources.comm_active_warps
                << ",compute_warps=" << resources.compute_active_warps << ",comm_smem=" << resources.comm_working_smem_bytes << '\n';
      if (resources.threads_per_cta <= 0 || resources.registers_per_thread <= 0 ||
          resources.telemetry_registers_per_thread <= 0 || resources.dynamic_smem_bytes <= 0) {
        throw std::runtime_error("empty role resource query");
      }
    }
  }
  // The instrumented kernel is a distinct specialization with its own first-
  // use resource setup. Warm it before attributing cross-rank waits to GPU
  // work; production warmup alone does not initialize this path. Ready epochs
  // stay contiguous across production and diagnostic launches.
  for (int iteration = 0; iteration < kWarmup; ++iteration) {
    clear_diagnostics();
    run_epoch(runtimes, options, direction, ++epoch, true);
  }
  clear_diagnostics();
  std::vector<double> production_host_us, instrumented_host_us;
  HostLaunchTiming production_dispatch, instrumented_dispatch;
  const auto production_times = run_epoch(runtimes, options, direction, ++epoch, false,
                                          &production_host_us, &production_dispatch);
  const auto instrumented_times = run_epoch(runtimes, options, direction, ++epoch, true,
                                            &instrumented_host_us, &instrumented_dispatch);
  auto print_dispatch = [&](const char* kind, const HostLaunchTiming& timing, uint32_t recorded_epoch) {
    const auto begin = std::minmax_element(timing.api_begin_us.begin(), timing.api_begin_us.end());
    const auto end = std::minmax_element(timing.api_end_us.begin(), timing.api_end_us.end());
    std::cout << "profile_dispatch," << direction_name(direction) << ",host_launch=" << options.host_launch
              << ",profile_detail=" << options.profile_detail
              << ",kind=" << kind << ",epoch=" << recorded_epoch
              << ",clock=steady_clock,origin=dispatch"
              << ",api_begin_skew_us=" << *begin.second - *begin.first
              << ",api_end_skew_us=" << *end.second - *end.first
              << ",all_enqueued_us=" << timing.all_enqueued_us << '\n';
  };
  print_dispatch("production", production_dispatch, epoch - 1);
  print_dispatch("instrumented", instrumented_dispatch, epoch);
  for (int rank = 0; rank < options.world; ++rank) {
    std::cout << "profile_host," << direction_name(direction) << ",rank=" << rank
              << ",profile_detail=" << options.profile_detail
              << ",host_launch=" << options.host_launch
              << ",instrumented_warmup=" << kWarmup
              << ",production_launch_us=" << production_host_us[rank]
              << ",instrumented_launch_us=" << instrumented_host_us[rank]
              << ",production_api_begin_us=" << production_dispatch.api_begin_us[rank]
              << ",production_api_end_us=" << production_dispatch.api_end_us[rank]
              << ",instrumented_api_begin_us=" << instrumented_dispatch.api_begin_us[rank]
              << ",instrumented_api_end_us=" << instrumented_dispatch.api_end_us[rank]
              << ",production_event_us=" << production_times[rank] * 1000.0
              << ",instrumented_event_us=" << instrumented_times[rank] * 1000.0 << '\n';
  }
  for (const int rank : fused_mpi::owned_ranks(options.world)) {
    CUDA_CHECK(cudaSetDevice(runtimes[rank].device));
    const auto& runtime = runtimes[rank];
    const auto timeline = download(runtime.timeline, runtime.sm_count);
#if FUSE_BENCH_MXFP8
    const auto traits = fuse::mxfp8_qkv_cutlass_kernel_traits(runtime.mxfp8_epilogue_n);
    const auto quant = download(runtime.mxfp8_probe.quant, runtime.mxfp8_probe.quant_capacity);
    for (int i = 0; i < static_cast<int>(quant.size()); ++i) {
      const auto& r = quant[i];
      if (!r.begin) continue; // Only native padding beyond the final N panel can be absent.
      std::cout << "profile_mxfp8_quant,rank=" << rank << ",index=" << i
                << ",cta=" << r.cta << ",warp=" << r.warp << ",panel=" << r.panel
                << ",groups=" << r.groups << ",begin=" << r.begin << ",quant_done=" << r.quant_done
                << ",arrival_chunks=" << r.arrival_chunks
                << ",warp_join_done=" << r.warp_join_done
                << ",arrival_done=" << r.arrival_done
                << ",end=" << r.end << ",release=" << r.release << '\n';
    }
    const auto waits = download(runtime.mxfp8_probe.waits, runtime.mxfp8_probe.wait_capacity);
    for (int i = 0; i < static_cast<int>(waits.size()); ++i) {
      const auto& r = waits[i];
      if (!r.begin) continue; // Repeated N panel uses reuse the mainloop's acquired panel.
      std::cout << "profile_mxfp8_wait,rank=" << rank << ",index=" << i
                << ",cta=" << r.cta << ",warp=" << r.warp << ",m=" << i / ceil_div(options.projection_width(), 256)
                << ",panel=" << i % ceil_div(options.projection_width(), 256)
                << ",begin=" << r.begin << ",end=" << r.end << '\n';
    }
#else
    const auto traits = direction == Direction::kQkv
        ? fuse::qkv_cutlass_kernel_traits(runtime.qkv.gemm, runtime.qkv.route,
                                         options.comm_sm, runtime.sm_count)
        : fuse::cutlass_kernel_traits();
#endif
    if (traits.block_m <= 0 || traits.block_n <= 0) throw std::runtime_error("invalid profiling tile query");
    const int m_tiles = ceil_div(options.seq_local, traits.block_m);
    const int n = direction == Direction::kQkv ? options.projection_width() : options.hidden;
    const int n_tiles = ceil_div(n, traits.block_n);
    // Cluster stays 1; parse_options rejects profile-only swizzle padding.
    // Thus every scheduled CTA here owns at least one unpadded physical tile.
    const int expected_compute = std::min(m_tiles * n_tiles, runtime.sm_count - options.comm_sm);
    int comm_records = 0, compute_records = 0;
    for (int cta = 0; cta < runtime.sm_count; ++cta) {
      const auto& event = timeline[cta];
      if ((cta < options.comm_sm + expected_compute) != (event.start != 0)) {
        throw std::runtime_error("missing or unexpected physical CTA record");
      }
      if (!event.start) continue; // Small shapes can launch fewer CTAs than SMs.
      if (event.end < event.start) throw std::runtime_error("invalid CTA end timestamp");
      if (cta < options.comm_sm) ++comm_records;
      else {
        ++compute_records;
        if (direction == Direction::kOproj &&
            (event.active_start < event.start || event.active_start > event.end)) {
          throw std::runtime_error("missing compute active_start");
        }
      }
      if (direction == Direction::kQkv &&
          (event.role_done < event.start || event.grid_sync_done < event.role_done || event.end < event.grid_sync_done)) {
        throw std::runtime_error("invalid QKV role/finalize ordering");
      }
      std::cout << "profile_cta," << direction_name(direction) << ",rank=" << rank << ",cta=" << cta
                << ",profile_detail=" << options.profile_detail
                << ",host_launch=" << options.host_launch
                << ",start=" << event.start << ",end=" << event.end << ",active_start=" << event.active_start
                << ",role_done=" << event.role_done << ",grid_sync_done=" << event.grid_sync_done
                << ",fence_done=" << event.fence_done << ",publish_done=" << event.publish_done;
      for (int peer = 0; peer < options.world; ++peer) std::cout << ",source_ready" << peer << '=' << event.source_ready[peer];
      std::cout << '\n';
    }
    if (comm_records != options.comm_sm || compute_records != expected_compute) {
      throw std::runtime_error("incomplete CTA role records");
    }
    if (direction == Direction::kQkv) {
      const auto& event = timeline[0];
      if (event.fence_done < event.grid_sync_done || event.publish_done < event.fence_done || event.end < event.publish_done) {
        throw std::runtime_error("missing QKV finalize timestamps");
      }
      for (int peer = 0; peer < options.world; ++peer) {
        if (event.source_ready[peer] < event.grid_sync_done || event.source_ready[peer] > event.end) {
          throw std::runtime_error("missing QKV source-completion timestamp");
        }
      }
      if (runtime.qkv_route_timeline) {
        const int route_warps = 8;
        const int expected_copies = ceil_div(options.seq_local, 64) *
            (options.q_heads + (runtime.qkv.route.defer_v_a2a ? 1 : 2) * options.kv_heads);
        // Every physical warp owns its original route stride and drain slot.
        // In comm_warp mode, warps 4..7 begin routing after their weight work.
        const int tasks = runtime.qkv_route_capacity - options.comm_sm * 8;
        const auto routes = download(runtime.qkv_route_timeline, tasks + options.comm_sm * 8);
        std::vector<uint64_t> previous(options.comm_sm * route_warps, 0);
        std::vector<bool> copied(ceil_div(options.seq_local, 64) * (n / 128), false);
        int copy_count = 0;
        std::cout << "profile_qkv_order,rank=" << rank << ",version=producer_ready_v1,slots=" << tasks
                  << ",copies=" << expected_copies << ",comm_sm=" << options.comm_sm
                  << ",route_warps=" << route_warps;
#if FUSE_BENCH_MXFP8
        if (options.mxfp8_weight_preparation == "comm_warp")
          std::cout << ",weight_schedule=warp_then_route_v1";
#endif
        std::cout << '\n';
        for (int index = 0; index < static_cast<int>(routes.size()); ++index) {
          const auto& r = routes[index];
          const bool drain = index >= tasks;
          if (!drain && !r.begin) continue; // Padding or copy owned by a later dependency.
          const int owner = (drain ? index - tasks : index) % (options.comm_sm * route_warps);
          if (r.cta != owner % options.comm_sm || r.warp != owner / options.comm_sm ||
              r.begin < timeline[r.cta].start || r.s2g_read_done < r.begin ||
              r.s2g_read_done > timeline[r.cta].role_done || r.begin < previous[owner]) {
            throw std::runtime_error("missing, overlapping or invalid QKV route task/drain");
          }
          previous[owner] = r.s2g_read_done;
          if (!drain && !(r.begin <= r.ready && r.ready <= r.g2s_begin &&
              r.g2s_begin <= r.g2s_done && r.g2s_done <= r.s2g_begin &&
              r.s2g_begin <= r.s2g_read_done && r.peer >= 0 && r.peer < options.world &&
              r.row >= 0 && r.row + r.rows <= options.seq_local &&
              r.column >= 0 && r.column + r.columns <= n && r.rows == 64 &&
              r.columns == 128 && r.segment >= 0 && r.segment < 3)) {
            throw std::runtime_error("invalid QKV route tile phases or coordinates");
          }
          if (!drain) {
            const int id = r.row / 64 * (n / 128) + r.column / 128;
            if (r.row % 64 || r.column % 128 || copied[id]) throw std::runtime_error("duplicate/misaligned QKV copy");
            copied[id] = true;
            ++copy_count;
          }
          std::cout << "profile_qkv_route,rank=" << rank << ",task=" << index
                    << ",drain=" << drain << ",cta=" << r.cta << ",warp=" << r.warp
                    << ",begin=" << r.begin << ",ready=" << r.ready
                    << ",g2s_begin=" << r.g2s_begin << ",g2s_done=" << r.g2s_done
                    << ",s2g_begin=" << r.s2g_begin << ",s2g_read_done=" << r.s2g_read_done
                    << ",row=" << r.row << ",column=" << r.column
                    << ",rows=" << r.rows << ",columns=" << r.columns
                    << ",peer=" << r.peer << ",segment=" << r.segment << '\n';
        }
        if (copy_count != expected_copies) throw std::runtime_error("incomplete QKV route coverage");
      }
      continue;
    }
    if (options.profile_detail == "cta") continue;
    if (runtime.oproj_pipeline.tiles) {
      const auto& probe = runtime.oproj_pipeline;
      const int count = probe.m_tiles * probe.n_tiles;
      const auto records = download(probe.tiles, count);
      const auto stages = download(probe.stages, static_cast<size_t>(count) * probe.k_tiles);
      for (int tile = 0; tile < count; ++tile) {
        const auto& r = records[tile];
        if (!r.mma_begin) continue;
        if (!(r.mma_begin <= r.tmem_acquired && r.tmem_acquired <= r.mma_return &&
              r.epi_begin <= r.acc_wait_begin && r.acc_wait_begin <= r.acc_wait_end &&
              r.acc_wait_end <= r.tmem_release_begin && r.tmem_release_begin <= r.tmem_release_end &&
              r.tmem_release_end <= r.epi_return)) throw std::runtime_error("invalid OProj pipeline phase order");
        std::cout << "profile_oproj_pipeline,rank=" << rank << ",index=" << tile << ",cta=" << r.cta
                  << ",mma_begin=" << r.mma_begin << ",tmem_acquired=" << r.tmem_acquired
                  << ",mma_return=" << r.mma_return << ",epi_begin=" << r.epi_begin
                  << ",acc_wait_begin=" << r.acc_wait_begin << ",acc_wait_end=" << r.acc_wait_end
                  << ",tmem_release_begin=" << r.tmem_release_begin << ",tmem_release_end=" << r.tmem_release_end
                  << ",epi_return=" << r.epi_return << ",k_tiles=" << probe.k_tiles;
        for (int peer = 0; peer < options.world; ++peer) {
          const auto& ready = r.ready[peer];
          if (!ready.begin || ready.begin > ready.end || ready.end > ready.joined)
            throw std::runtime_error("invalid OProj ready check interval");
          std::cout << ",ready_begin" << peer << '=' << ready.begin << ",ready_end" << peer << '=' << ready.end
                    << ",ready_joined" << peer << '=' << ready.joined << ",cache_hit" << peer << '=' << ready.cache_hit;
        }
        std::cout << '\n';
        for (int k = 0; k < probe.k_tiles; ++k) {
          const auto& s = stages[static_cast<size_t>(tile) * probe.k_tiles + k];
          if (!s.wait_begin || s.wait_begin > s.wait_end || s.wait_end > s.issue_end)
            throw std::runtime_error("invalid OProj MMA stage interval");
          std::cout << "profile_oproj_stage,rank=" << rank << ",index=" << tile << ",k=" << k
                    << ",wait_begin=" << s.wait_begin << ",wait_end=" << s.wait_end
                    << ",issue_end=" << s.issue_end << '\n';
        }
      }
    }
    const auto peers = download(runtime.peer_timeline, runtime.peer_capacity);
    for (int index = 0; index < runtime.peer_capacity; ++index) {
      const auto& event = peers[index];
      if (index < m_tiles * options.world) {
        if (!event.comm_valid || !event.task_begin || event.input_ready < event.task_begin ||
            event.publish_issue < event.input_ready || event.release < event.publish_issue ||
            event.copy_rows < 0 || event.copy_path < 0 || event.copy_path > 3) {
          throw std::runtime_error("missing or invalid peer release record");
        }
        // Empty bulk tails can leave all copy phases zero. The vector path
        // still samples its no-op loop/barrier, which must remain ordered.
        const bool has_copy_timestamps =
            event.g2s_issue || event.g2s_done || event.s2g_issue || event.s2g_done;
        if (event.copy_rows > 0 || has_copy_timestamps) {
          if (event.g2s_issue < event.input_ready || event.g2s_done < event.g2s_issue) {
            throw std::runtime_error("invalid G2S phase ordering");
          }
          if (event.copy_path == 0) {
            if (event.s2g_issue || event.s2g_done || event.publish_issue < event.g2s_done) {
              throw std::runtime_error("invalid vector-copy phase ordering");
            }
          } else if (event.s2g_issue < event.g2s_done || event.s2g_done < event.s2g_issue ||
                     event.publish_issue < event.s2g_done) {
            throw std::runtime_error("invalid bulk-copy phase ordering");
          }
        }
      }
      if (index < m_tiles * n_tiles) {
        if (!event.valid || event.m_tile != index / n_tiles || event.n_tile != index % n_tiles || event.batch != 0) {
          throw std::runtime_error("missing logical GEMM tile record");
        }
        for (int peer = 0; peer < options.world; ++peer) {
          if (!event.acquire[peer]) throw std::runtime_error("missing peer acquire timestamp");
        }
      }
      // release is sampled AFTER publication; a racing acquire may precede
      // that sample. Absolute globaltimer values are not compared across GPUs.
      std::cout << "profile_peer,rank=" << rank << ",index=" << index << ",release=" << event.release
                << ",profile_detail=" << options.profile_detail
                << ",host_launch=" << options.host_launch
                << ",task_begin=" << event.task_begin << ",input_ready=" << event.input_ready
                << ",g2s_issue=" << event.g2s_issue << ",g2s_done=" << event.g2s_done
                << ",s2g_issue=" << event.s2g_issue << ",s2g_done=" << event.s2g_done
                << ",publish_issue=" << event.publish_issue << ",m=" << event.m_tile << ",n=" << event.n_tile
                << ",batch=" << event.batch << ",valid=" << event.valid << ",comm_cta=" << event.comm_cta
                << ",comm_slot=" << event.comm_slot << ",task_id=" << event.task_id << ",row_chunk=" << event.row_chunk
                << ",copy_rows=" << event.copy_rows << ",source_rank=" << event.source_rank
                << ",copy_path=" << event.copy_path << ",comm_valid=" << event.comm_valid;
      for (int peer = 0; peer < options.world; ++peer) std::cout << ",acquire" << peer << '=' << event.acquire[peer];
      std::cout << '\n';
    }
  }
  // Validate the instrumented epoch before production diagnostics overwrite
  // its outputs; timestamp coverage alone does not establish correctness.
  validate(runtimes, options, direction, ",profile_phase=instrumented");
  profile_host_stages(runtimes, options, direction, epoch);
  validate(runtimes, options, direction, ",profile_phase=host_stages");
}
#endif

#if FUSE_BENCH_MXFP8 && FUSE_ENABLE_PROFILING
// Service probes keep the production queues and resources, but remove their
// dependency one at a time. Only GPU0 records device-local timestamps. The
// other seven GPUs still execute and validate the same communication protocol.
// Empty-view and instrumented timings are diagnostics, not Graph performance.
// Their Eager CUDA events include host API preparation/enqueue idle time.
// Derive services ONLY from device-local timestamps below, never by correcting
// them with this launch-inclusive event ratio or by dividing C time by waves.
void profile_mxfp8_services(std::vector<RankRuntime>& runtimes, const Options& options,
                            const Candidate& candidate) {
  select_candidate(runtimes, candidate, ",service_probe=1");
  auto packed = [&](int rank, uint32_t epoch) {
    auto& runtime = runtimes[rank];
    fuse::Mxfp8GemmA2AParams params{runtime.qkv, runtime.mxfp8_workspace,
        runtime.mxfp8_workspace_bytes, runtime.mxfp8_activation,
        runtime.mxfp8_weight_preparation, runtime.mxfp8_epilogue_n};
    params.projection.num_comm_ctas = candidate.comm_sm;
    params.projection.epoch = epoch;
    params.projection.lhs = nullptr;
    return params;
  };
  fuse::Mxfp8ServiceResources resources{};
  for (int rank = 0; rank < options.world; ++rank) {
    CUDA_CHECK(cudaSetDevice(runtimes[rank].device));
    fuse::Mxfp8ServiceResources current{};
    CUDA_CHECK(fuse::query_gemm_a2a_mxfp8_service_resources(packed(rank, 1), &current));
    if (rank == 0) resources = current;
    else if (current.stages != resources.stages || current.dynamic_smem_bytes != resources.dynamic_smem_bytes ||
             current.compute_ctas != resources.compute_ctas || current.delayed_panel != resources.delayed_panel ||
             current.resolved_swizzle != resources.resolved_swizzle)
      throw std::runtime_error("MXFP8 service resources differ across ranks");
  }
  CUDA_CHECK(cudaSetDevice(runtimes[0].device));
  cudaDeviceProp device{};
  CUDA_CHECK(cudaGetDeviceProperties(&device, runtimes[0].device));
  std::ostringstream uuid;
  uuid << "GPU-" << std::hex << std::setfill('0');
  for (int i = 0; i < 16; ++i) {
    if (i == 4 || i == 6 || i == 8 || i == 10) uuid << '-';
    uuid << std::setw(2) << static_cast<unsigned>(static_cast<unsigned char>(device.uuid.bytes[i]));
  }
  std::ofstream output(options.mxfp8_service_output);
  if (!output) throw std::runtime_error("cannot open MXFP8 service sidecar");
  output << std::setprecision(12);
  auto& view = runtimes[0].mxfp8_service;
  auto clear_records = [&]() {
    CUDA_CHECK(cudaSetDevice(runtimes[0].device));
    const auto stream = runtimes[0].stream;
    auto clear = [&](auto* ptr, int capacity) {
      using Value = std::remove_pointer_t<decltype(ptr)>;
      if (ptr) CUDA_CHECK(cudaMemsetAsync(ptr, 0, checked_bytes<Value>(capacity), stream));
    };
    clear(view.tiles, view.tile_capacity); clear(view.ctas, view.cta_capacity);
    clear(view.panel_release, view.panel_capacity); clear(view.panel_release_begin, view.panel_capacity);
    clear(view.weight.quant, view.weight.quant_capacity); clear(view.weight.waits, view.weight.wait_capacity);
    clear(view.routes, view.route_capacity);
    finish_all(runtimes); // Clearing is outside both CUDA event intervals.
  };
  uint32_t epoch = 0;
  auto run = [&](const fuse::Mxfp8ServiceConfig& config, bool instrumented) {
    ::alarm(options.timeout_seconds);
    if (instrumented) clear_records();
    struct Job {
      std::vector<RankRuntime>* runtimes;
      std::vector<fuse::Mxfp8GemmA2AParams> params;
      fuse::Mxfp8ServiceConfig config;
      bool instrumented;
    } job{&runtimes, {}, config, instrumented};
    ++epoch;
    for (int rank = 0; rank < options.world; ++rank) job.params.push_back(packed(rank, epoch));
    auto enqueue = +[](void* opaque, int rank) {
      auto& job = *static_cast<Job*>(opaque);
      auto& runtime = (*job.runtimes)[rank];
      CUDA_CHECK(cudaSetDevice(runtime.device));
      CUDA_CHECK(cudaEventRecord(runtime.start, runtime.stream));
      CUDA_CHECK(fuse::launch_gemm_a2a_mxfp8_service(job.params[rank], job.config,
          job.instrumented && rank == 0 ? runtime.mxfp8_service : fuse::Mxfp8ServiceView{}, runtime.stream));
      CUDA_CHECK(cudaEventRecord(runtime.end, runtime.stream));
    };
    if (runtimes.front().launch_team)
      check_enqueue(runtimes.front().launch_team->dispatch(enqueue, &job));
    else for (int rank = 0; rank < options.world; ++rank) enqueue(&job, rank);
    wait_all(runtimes);
    CUDA_CHECK(cudaSetDevice(runtimes[0].device));
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, runtimes[0].start, runtimes[0].end));
    if (!std::isfinite(milliseconds) || milliseconds <= 0)
      throw std::runtime_error("invalid service diagnostic sample");
    return milliseconds;
  };
  struct Stage { const char* name; fuse::Mxfp8ServiceMode mode; int delay_multiple; int quant_phase; };
  const Stage stages[] = {
      {"C_allready", fuse::Mxfp8ServiceMode::kCompute, 0, 0},
      {"C_delay1", fuse::Mxfp8ServiceMode::kCompute, 2, 0},
      {"C_delay2", fuse::Mxfp8ServiceMode::kCompute, 8, 0},
      {"Q", fuse::Mxfp8ServiceMode::kQuantize, 0, 0},
      {"R", fuse::Mxfp8ServiceMode::kRoute, 0, 0},
      {"QR", fuse::Mxfp8ServiceMode::kQuantizeRoute, 0, 0},
      {"QR_phase1", fuse::Mxfp8ServiceMode::kQuantizeRoute, 0, 1}};
  uint64_t compute_span_ns = 0;
  for (uint32_t generation = 0; generation < 2; ++generation) {
    set_inputs(runtimes, options, generation);
    prepare_references(runtimes, options);
    for (const auto& stage : stages) {
      if (stage.delay_multiple && resources.delayed_panel < 0) {
        if (!generation) std::cout << "service_unavailable,stage=" << stage.name
            << ",reason=all_panels_in_initial_compute_wave\n";
        continue;
      }
      std::cout << "service_progress,stage=" << stage.name << ",generation=" << generation
          << ",comm_sm=" << candidate.comm_sm << '\n' << std::flush;
      const bool compute = stage.mode == fuse::Mxfp8ServiceMode::kCompute;
      const bool quantize = stage.mode == fuse::Mxfp8ServiceMode::kQuantize ||
          stage.mode == fuse::Mxfp8ServiceMode::kQuantizeRoute;
      const bool route = stage.mode == fuse::Mxfp8ServiceMode::kRoute ||
          stage.mode == fuse::Mxfp8ServiceMode::kQuantizeRoute;
      Options check_options = options;
      check_options.comm_sm = candidate.comm_sm;
      check_options.component = compute ? MeasurementComponent::kComputeReference
          : (route ? MeasurementComponent::kCopyReference : MeasurementComponent::kQuantizeReference);
      // This independent path has no preceding F epoch to seed the runtime.
      // Preparation validates the same positive-epoch contract as production;
      // the actual diagnostic launch below advances its own epoch as usual.
      for (auto& runtime : runtimes) runtime.qkv.epoch = epoch + 1;
      prepare_component(runtimes, check_options, Direction::kQkv);
      if (quantize && route) {
        // Preserve C's validated local output for R while proving QR writes
        // every weight itself. Validation below does not re-prepare weights.
        for (int rank = 0; rank < options.world; ++rank) {
          CUDA_CHECK(cudaSetDevice(runtimes[rank].device));
          CUDA_CHECK(cudaMemsetAsync(runtimes[rank].mxfp8_workspace, 0xff,
              runtimes[rank].mxfp8_workspace_bytes, runtimes[rank].stream));
        }
        finish_all(runtimes);
      }
      fuse::Mxfp8ServiceConfig config{};
      config.mode = stage.mode;
      config.quant_phase_steps = stage.quant_phase;
      if (stage.delay_multiple) {
        if (!compute_span_ns || compute_span_ns > std::numeric_limits<uint64_t>::max() / stage.delay_multiple)
          throw std::runtime_error("invalid measured C delay interval");
        config.delayed_panel = resources.delayed_panel;
        config.delay_ns = compute_span_ns * stage.delay_multiple;
      }
      std::vector<float> control_samples, instrumented_samples;
      if (generation == 0) {
        for (bool instrumented : {false, true}) {
          for (int i = 0; i < kWarmup; ++i) run(config, instrumented);
          auto& samples = instrumented ? instrumented_samples : control_samples;
          for (int i = 0; i < kSamples; ++i) samples.push_back(run(config, instrumented));
        }
        // A single fresh, fully cleared epoch is the sole raw trace retained.
        // Fifty perturbation timings above are not fifty service captures.
        run(config, true);
        auto prefix = [&](const char* kind) -> std::ostream& {
          return output << "{\"schema\":\"sm103_mxfp8_services_v1\",\"kind\":\"" << kind
              << "\",\"stage\":\"" << stage.name << "\",\"iteration\":0,\"rank\":0";
        };
        prefix("config") << ",\"captured_rank\":0,\"capture_epochs\":1,\"validation_generations\":2"
            << ",\"event_boundary\":\"host_api_inclusive_eager_cuda_event\""
            << ",\"warmup\":" << kWarmup << ",\"samples\":" << kSamples
            << ",\"clock\":\"globaltimer\",\"clock_unit\":\"ns\",\"gpu_uuid\":\"" << uuid.str()
            << "\",\"m\":" << options.seq_local << ",\"n\":" << options.projection_width()
            << ",\"k\":" << options.hidden << ",\"world\":" << options.world
            << ",\"sm_count\":" << runtimes[0].sm_count << ",\"capability\":" << device.major * 10 + device.minor
            << ",\"comm_ctas\":" << candidate.comm_sm << ",\"compute_ctas\":" << resources.compute_ctas
            << ",\"tile_m\":128,\"tile_n\":256,\"tile_k\":128,\"epilogue_n\":" << options.mxfp8_epilogue_n
            << ",\"stages\":" << resources.stages << ",\"cluster_ctas\":1,\"raster\":" << (options.qkv_raster == "along_n" ? 1 : 0)
            << ",\"resolved_swizzle\":" << resources.resolved_swizzle << ",\"max_swizzle_size\":" << options.max_swizzle_size
            << ",\"dynamic_smem_bytes\":" << resources.dynamic_smem_bytes << ",\"q_heads\":" << options.q_heads
            << ",\"kv_heads\":" << options.kv_heads << ",\"head_dim\":" << options.head_dim
            << ",\"weight_preparation\":\"comm\",\"rank_swizzle\":\"off\",\"delayed_panel\":" << config.delayed_panel
            << ",\"delay_ns\":" << config.delay_ns << ",\"quant_phase_steps\":" << config.quant_phase_steps;
        auto array = [&](const char* name, const std::vector<float>& values) {
          output << ",\"" << name << "\":[";
          for (size_t i = 0; i < values.size(); ++i) { if (i) output << ','; output << values[i]; }
          output << ']';
        };
        array("control_samples_ms", control_samples); array("instrumented_samples_ms", instrumented_samples);
        output << "}\n";
        const auto ctas = download(view.ctas, view.cta_capacity);
        uint64_t first = std::numeric_limits<uint64_t>::max(), last = 0;
        for (size_t i = 0; i < ctas.size(); ++i) {
          const auto& r = ctas[i]; if (!r.begin) continue;
          if (r.setup_done < r.begin || r.end < r.setup_done) throw std::runtime_error("invalid service CTA timestamps");
          first = std::min(first, r.begin); last = std::max(last, r.end);
          prefix("cta") << ",\"index\":" << i << ",\"begin\":" << r.begin
              << ",\"setup_done\":" << r.setup_done << ",\"end\":" << r.end << "}\n";
        }
        if (compute && !stage.delay_multiple) {
          if (last <= first) throw std::runtime_error("missing service C span");
          compute_span_ns = last - first;
        }
        const auto tiles = download(view.tiles, view.tile_capacity);
        const auto waits = download(view.weight.waits, view.weight.wait_capacity);
        for (size_t i = 0; i < tiles.size(); ++i) {
          const auto& r = tiles[i]; if (!r.first_load && !r.ready_after) continue;
          prefix("tile") << ",\"index\":" << i << ",\"cta\":" << r.cta << ",\"warp\":" << r.warp
              << ",\"m\":" << r.m << ",\"n\":" << r.n << ",\"first_load\":" << r.first_load
              << ",\"load_return\":" << r.load_return << ",\"store_begin\":" << r.store_begin
              << ",\"ready_after\":" << r.ready_after << ",\"wait_begin\":" << waits[i].begin
              << ",\"wait_end\":" << waits[i].end << "}\n";
        }
        auto panels = download(view.panel_release, view.panel_capacity);
        const auto panel_begin = download(view.panel_release_begin, view.panel_capacity);
        const auto quant = download(view.weight.quant, view.weight.quant_capacity);
        for (size_t i = 0; i < quant.size(); ++i) {
          const auto& r = quant[i]; if (!r.begin) continue;
          if (r.release && r.panel >= 0 && r.panel < static_cast<int>(panels.size())) panels[r.panel] = r.release;
          prefix("quant") << ",\"index\":" << i << ",\"begin\":" << r.begin
              << ",\"quant_done\":" << r.quant_done << ",\"end\":" << r.end << ",\"release\":" << r.release
              << ",\"warp_join_done\":" << r.warp_join_done << ",\"arrival_done\":" << r.arrival_done
              << ",\"cta\":" << r.cta << ",\"warp\":" << r.warp << ",\"panel\":" << r.panel
              << ",\"groups\":" << r.groups << ",\"arrival_chunks\":" << r.arrival_chunks << "}\n";
        }
        for (size_t i = 0; i < panels.size(); ++i) if (panels[i])
          prefix("panel") << ",\"index\":" << i << ",\"release_begin\":" << panel_begin[i]
              << ",\"release\":" << panels[i] << "}\n";
        const auto routes = download(view.routes, view.route_capacity);
        for (size_t i = 0; i < routes.size(); ++i) {
          const auto& r = routes[i]; if (!r.begin) continue;
          prefix("route") << ",\"index\":" << i << ",\"begin\":" << r.begin << ",\"ready\":" << r.ready
              << ",\"g2s_begin\":" << r.g2s_begin << ",\"g2s_done\":" << r.g2s_done
              << ",\"s2g_begin\":" << r.s2g_begin << ",\"s2g_read_done\":" << r.s2g_read_done
              << ",\"copy_end\":" << r.copy_end << ",\"cta\":" << r.cta << ",\"warp\":" << r.warp
              << ",\"row\":" << r.row << ",\"column\":" << r.column << ",\"rows\":" << r.rows
              << ",\"columns\":" << r.columns << ",\"peer\":" << r.peer << ",\"segment\":" << r.segment << "}\n";
        }
        output.flush();
        if (!output) throw std::runtime_error("cannot write MXFP8 service sidecar");
      } else run(config, false);
      const std::string context = ",service_stage=" + std::string(stage.name) + ",generation=" + std::to_string(generation);
      validate(runtimes, check_options, Direction::kQkv, context);
      if (quantize && route) {
        check_options.component = MeasurementComponent::kQuantizeReference;
        validate(runtimes, check_options, Direction::kQkv, context + ",quant_workspace_validation=1");
      }
    }
  }
  for (const auto& stage : stages) {
    if (stage.delay_multiple && resources.delayed_panel < 0) continue;
    std::cout << "service_stage_verified,stage=" << stage.name
        << ",payload_generations=2,full_numeric=" << (stage.mode != fuse::Mxfp8ServiceMode::kRoute)
        << ",full_route=" << (stage.mode == fuse::Mxfp8ServiceMode::kRoute || stage.mode == fuse::Mxfp8ServiceMode::kQuantizeRoute)
        << ",performance_accepted=0\n";
  }
}
#endif

void destroy_runtimes(std::vector<RankRuntime>& runtimes, const Options& options) {
  ::alarm(options.timeout_seconds);
  finish_all(runtimes);
#if FUSE_BENCH_MPI
  auto& owner = runtimes[fused_mpi::process_rank];
  CUDA_CHECK(cudaSetDevice(owner.device));
  if (owner.graph) {
    owner.graph->reset(owner.graph->committed_epoch());
    owner.graph.reset();
  }
  for (void* mapping : owner.ipc_mappings) CUDA_CHECK(cudaIpcCloseMemHandle(mapping));
  owner.ipc_mappings.clear();
  // Owners cannot free a buffer until every importer has closed its mapping.
  fused_mpi::barrier();
#endif
  for (const int rank : fused_mpi::owned_ranks(static_cast<int>(runtimes.size()))) {
    CUDA_CHECK(cudaSetDevice(runtimes[rank].device));
    auto& runtime = runtimes[rank];
    CUBLAS_CHECK(cublasDestroy(runtime.blas));
    for (void* allocation : runtime.allocations) CUDA_CHECK(cudaFree(allocation));
    for (int peer : runtime.enabled_peers) CUDA_CHECK(cudaDeviceDisablePeerAccess(peer));
    CUDA_CHECK(cudaEventDestroy(runtime.start));
    CUDA_CHECK(cudaEventDestroy(runtime.end));
    CUDA_CHECK(cudaStreamDestroy(runtime.stream));
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGALRM, timeout_handler);
  // Keep ownership outside the try: fatal errors must reach _exit without
  // first unwinding through a potentially blocked worker join.
  std::unique_ptr<fused_launch::Team> launch_team;
  try {
    ::alarm(60);
    fused_mpi::initialize(argc, argv);
    const Options options = parse_options(argc, argv);
    ::alarm(options.timeout_seconds);
    auto candidates = make_candidates(options);
    std::ostringstream contract;
    contract << options.quick << ' ' << options.world << ' ' << options.seq_local << ' ' << options.hidden << ' '
             << options.q_heads << ' ' << options.kv_heads << ' ' << options.head_dim << ' '
             << options.seed << ' ' << options.causal << ' ' << options.input_generator << ' '
             << options.host_launch << ' ' << options.calibrate << ' ' << options.cpu_oracle << ' '
             << options.validation_self_test << ' ' << options.profile << ' ' << options.timeout_seconds << ' '
             << options.max_swizzle_size << ' ' << options.qkv_raster << ' ' << options.oproj_raster
             << ' ' << options.launch << ' ' << options.fused_direction << ' ' << options.oproj_comm_layout
             << ' ' << options.auto_oproj_comm << ' ' << options.auto_mxfp8_comm;
    for (size_t index = 0; index < candidates.size(); ++index) {
      contract << ' ' << direction_name(candidates[index].direction) << candidate_context(index, candidates[index]);
    }
    fused_mpi::agree(contract.str());
#if FUSE_BENCH_MXFP8
    fused_mpi::agree(options.mxfp8_prequantized ? "mxfp8_prequantized" : "mxfp8_dynamic");
    fused_mpi::agree(options.mxfp8_weight_preparation);
    fused_mpi::agree(std::to_string(options.mxfp8_epilogue_n));
    fused_mpi::agree(fuse::detail::kMxfp8QkvCalibrationVersion);
    fused_mpi::root_output() << "precision,mxfp8,input=mxfp8,weight=bf16,output=bf16,accumulator=fp32"
        << ",scale=ue8m0,group_k=32,tile=128x256x128,includes_activation_quantization=0"
        << ",includes_weight_quantization=" << !options.mxfp8_prequantized
        << ",weight_preparation=" << options.mxfp8_weight_preparation
        << ",epilogue_n=" << options.mxfp8_epilogue_n;
    if (options.mxfp8_weight_preparation == "comm_warp")
      fused_mpi::root_output() << ",weight_schedule=warp_then_route_v1";
    fused_mpi::root_output() << ",publication_protocol=warp_panel_acq_rel_v3,kernel_nodes=1\n";
#endif
    // One communication layout for every candidate/payload; never mutate the
    // ready arrival geometry while reusing this run's cumulative epochs.
    if (::setenv("FUSE_SM103_OPROJ_COMM_LAYOUT", options.oproj_comm_layout.c_str(), 1) != 0) {
      throw std::runtime_error("cannot set OProj communication layout");
    }
    std::cout << std::setprecision(9); // Also retain full input statistics on non-root ranks.
    fused_mpi::root_output() << std::setprecision(9) << "config,world=" << options.world << ",comm_sm=" << options.comm_sm
              << ",auto_oproj_comm=" << options.auto_oproj_comm
#if FUSE_BENCH_MXFP8
              << ",auto_mxfp8_comm=" << options.auto_mxfp8_comm
#endif
              << ",global_seq=" << options.global_seq() << ",seq_local=" << options.seq_local
              << ",q_heads=" << options.q_heads << ",kv_heads=" << options.kv_heads
              << ",head_dim=" << options.head_dim << ",hidden=" << options.hidden << ",causal=" << options.causal
              << ",seed=" << options.seed << ",warmup=" << (options.quick ? 1 : kWarmup)
              << ",samples=" << (options.quick ? 5 : kSamples)
              << ",sampling_mode=" << (options.quick ? "quick_1_5" : "formal_10_50")
              << ",input_generator=" << options.input_generator
              << ",host_launch=" << options.host_launch
              << ",max_swizzle_size=" << options.max_swizzle_size
#if FUSE_SM103_QKV_RANK_SWIZZLE
              << ",qkv_rank_swizzle=rank_n_band_v1"
#else
              << ",qkv_rank_swizzle=off"
#endif
              << ",qkv_raster=" << options.qkv_raster << ",oproj_raster=" << options.oproj_raster
              << ",oproj_comm_layout=" << options.oproj_comm_layout
              << ",qkv_effective_raster=" << (options.qkv_raster == "heuristic" ? "along_m" : options.qkv_raster)
              << ",oproj_effective_raster=" << (options.oproj_raster == "heuristic" ? "along_n" : options.oproj_raster)
              << ",process_layout=" << (fused_mpi::enabled ? "mpi_one_process_per_gpu" : "single_process")
              << ",launch=" << options.launch
              << ",fused_direction=" << options.fused_direction
              << ",calibrate=" << options.calibrate
              << ",profile=" << options.profile << ",timeout_seconds=" << options.timeout_seconds
              << ",profile_detail=" << (options.profile ? options.profile_detail : "none")
              << ",qkv_epilogue_probe=" << options.qkv_epilogue_probe
#if FUSE_ENABLE_PROFILING
              << ",profile_schema=" << (options.profile ? "host_stages_v1" : "none")
#endif
              << ",cpu_oracle=" << options.cpu_oracle << ",validation_self_test=" << options.validation_self_test
              << ",validation_scratch_bytes=" << sizeof(fused_validation::Scratch)
              << ",input_scratch_bytes=" << (options.input_generator == "gpu_philox" ? sizeof(fused_inputs::Scratch) : 0)
              << ",candidates=" << candidates.size();
    if (options.launch == "graph") {
      fused_mpi::root_output() << ",collector=mpi_graph_rank_events_v1,graph_epoch_mode=recapture_update_v1";
    }
    fused_mpi::root_output() << '\n' << std::flush;
    std::vector<RankRuntime> runtimes;
    {
      StageTimer timer{"setup", options.input_generator == "gpu_philox"
          ? ",includes=gpu_weight_rng_statistics_cuda_resources_alloc_init"
          : ",includes=weight_rng_cuda_resources_alloc_upload_init"};
      set_tile_policy(Direction::kQkv, options.qkv_policy_list.front());
      set_tile_policy(Direction::kOproj, options.oproj_policy_list.front());
      runtimes = create_runtimes(options);
      launch_team = start_launch_team(runtimes, options);
    }
    if (options.auto_oproj_comm || options.auto_mxfp8_comm) resolve_auto_candidates(runtimes, candidates);
#if FUSE_BENCH_MXFP8 && FUSE_ENABLE_PROFILING
    if (options.mxfp8_service_probe) {
      profile_mxfp8_services(runtimes, options, candidates.front());
      launch_team.reset();
      destroy_runtimes(runtimes, options);
      runtimes.clear();
      fused_mpi::finalize();
      ::alarm(0);
      std::cout << "PASS: MXFP8 service diagnostics, full numeric/routes, two payloads; not production performance\n";
      return 0;
    }
#endif
    uint32_t qkv_epoch = 0, oproj_epoch = 0;
    uint32_t reference_epochs[2][3]{}; // Per direction: compute, copy, MXFP8 quantization.
    bool self_tested[2] = {false, false};
    // Generation-major reuse avoids retaining two copies of a large shape.
    // A candidate is not accepted until BOTH payload generations pass.
    for (uint32_t generation = 0; generation < 2; ++generation) {
      const std::string payload_context = ",generation=" + std::to_string(generation);
      {
        StageTimer timer{"input", payload_context + (options.input_generator == "gpu_philox"
            ? ",includes=gpu_activation_rng_statistics_publish"
            : ",includes=activation_rng_statistics_upload_publish")};
        set_inputs(runtimes, options, generation);
      }
      {
        StageTimer timer{"reference", payload_context + (options.input_generator == "gpu_philox"
            ? ",includes=gpu_scalar_gather_cublas_completion"
            : ",includes=cpu_gather_upload_cublas_completion")};
        prepare_references(runtimes, options);
      }
      for (size_t index = 0; index < candidates.size(); ++index) {
        const auto& candidate = candidates[index];
        const std::string candidate_payload = candidate_context(index, candidate) + payload_context;
        const std::string context = candidate_payload + ",component=fused";
        Options candidate_options = options;
        candidate_options.comm_sm = candidate.comm_sm;
        uint32_t& epoch = candidate.direction == Direction::kQkv ? qkv_epoch : oproj_epoch;
        const int direction_index = candidate.direction == Direction::kQkv ? 0 : 1;
        // Pure-GEMM search does not launch communication roles or select by F.
        // Keep the same tile, scheduler and reserved compute budget for C.
        if (options.compute_only) select_candidate(runtimes, candidate, context);
        if (!options.compute_only) {
        {
          StageTimer timer{"validate", context + ",includes=select_poison_fused_epoch_gpu_full_checks_small_stats"
              + (options.cpu_oracle ? "_and_full_cpu_oracle" : "")};
          select_candidate(runtimes, candidate, context);
          bind_graph(runtimes, candidate_options, epoch);
          if (options.calibrate) describe_component(runtimes, candidate_options, candidate.direction, context);
          poison_outputs(runtimes, candidate_options, candidate.direction);
          run_epoch(runtimes, candidate_options, candidate.direction, ++epoch);
          validate(runtimes, candidate_options, candidate.direction, context);
        }
        if (options.validation_self_test && !self_tested[direction_index]) {
          StageTimer timer{"validation_self_test", context + ",includes=fault_detection_restore_full_gpu_cpu_checks"};
          validation_self_test(runtimes, candidate_options, candidate.direction, context);
          self_tested[direction_index] = true;
        }
        if (generation == 0 && !options.validation_self_test) {
          StageTimer timer{"benchmark", context + ",includes=convergence_sampling_logging"};
          benchmark(runtimes, candidate_options, candidate.direction, epoch, context);
        }
        if (options.launch == "graph" && generation == 0 && !options.validation_self_test) {
          StageTimer timer{"validate", context + ",validation_phase=post,includes=last_graph_sample_full_checks"};
          validate(runtimes, candidate_options, candidate.direction, context + ",validation_phase=post");
        }
        report_graph_preparation(runtimes, candidate_options, candidate.direction, context);
        }
        if (options.calibrate) {
          // Only separate calibration flags are reset. Production counters
          // continue across candidates and both payload generations.
          {
            StageTimer timer{"calibration_setup", candidate_payload + ",includes=completion_clear_independent_flags"};
            reset_calibration(runtimes, candidate_options, candidate.direction);
          }
          reference_epochs[direction_index][0] = 0;
          reference_epochs[direction_index][1] = 0;
          reference_epochs[direction_index][2] = 0;
          for (const auto component : {MeasurementComponent::kComputeReference,
                                       MeasurementComponent::kCopyReference
#if FUSE_BENCH_MXFP8
                                       , MeasurementComponent::kQuantizeReference
#endif
                                       }) {
            if (options.compute_only && component != MeasurementComponent::kComputeReference) continue;
            Options reference_options = candidate_options;
            reference_options.component = component;
            const std::string reference_context = candidate_payload + ",component=" + component_name(component);
            uint32_t& reference_epoch = reference_epochs[direction_index]
                [component == MeasurementComponent::kComputeReference ? 0 :
                    (component == MeasurementComponent::kCopyReference ? 1 : 2)];
            describe_component(runtimes, reference_options, candidate.direction, reference_context);
            {
              StageTimer timer{"validate", reference_context + ",validation_phase=pre,includes=prepare_reference_epoch_full_checks"};
              prepare_component(runtimes, reference_options, candidate.direction);
              bind_graph(runtimes, reference_options, reference_epoch);
              run_epoch(runtimes, reference_options, candidate.direction, ++reference_epoch);
              validate(runtimes, reference_options, candidate.direction, reference_context + ",validation_phase=pre");
            }
            if (generation == 0) {
              {
                StageTimer timer{"benchmark", reference_context + ",includes=convergence_sampling_logging"};
                benchmark(runtimes, reference_options, candidate.direction, reference_epoch, reference_context);
              }
              StageTimer timer{"validate", reference_context + ",validation_phase=post,includes=last_sample_full_checks"};
              validate(runtimes, reference_options, candidate.direction, reference_context + ",validation_phase=post");
            }
            report_graph_preparation(runtimes, reference_options, candidate.direction, reference_context);
          }
        }
      }
    }
#if FUSE_ENABLE_PROFILING
    if (options.profile) {
      StageTimer timer{"profile", ",includes=instrumented_warmup_host_timing_full_validation"};
      for (size_t index = 0; index < candidates.size(); ++index) {
        const auto& candidate = candidates[index];
        select_candidate(runtimes, candidate, candidate_context(index, candidate));
        uint32_t& epoch = candidate.direction == Direction::kQkv ? qkv_epoch : oproj_epoch;
        if (options.oproj_gap_probe) profile_compute_gaps(runtimes, options);
        else profile(runtimes, options, candidate.direction, epoch);
        if (options.qkv_epilogue_probe && candidate.direction == Direction::kQkv) {
          profile_qkv_epilogue(runtimes, options, epoch);
        }
      }
    }
#endif
    {
      StageTimer timer{"cleanup", ",includes=cuda_completion_device_and_host_resources"};
      launch_team.reset();
      destroy_runtimes(runtimes, options);
      runtimes.clear();
    }
    fused_mpi::finalize();
    ::alarm(0);
    for (size_t index = 0; index < candidates.size(); ++index) {
      for (const auto component : {MeasurementComponent::kFused, MeasurementComponent::kComputeReference,
                                   MeasurementComponent::kCopyReference
#if FUSE_BENCH_MXFP8
                                   , MeasurementComponent::kQuantizeReference
#endif
                                   }) {
        if (!options.calibrate && component != MeasurementComponent::kFused) continue;
        if (options.compute_only && component != MeasurementComponent::kComputeReference) continue;
        fused_mpi::root_output() << "candidate_verified," << direction_name(candidates[index].direction)
                  << candidate_context(index, candidates[index]) << ",component=" << component_name(component)
                  << ",host_launch=" << options.host_launch << ",payload_generations=2"
                  << ",full_numeric=" << (component != MeasurementComponent::kCopyReference)
                  << ",full_route=" << component_has_route(component)
                  << ",performance_accepted=" << (!options.profile && !options.validation_self_test && options.counter_component.empty());
        if (options.launch == "graph") {
          fused_mpi::root_output() << ",launch=graph,graph_epoch_mode=recapture_update_v1";
        }
        fused_mpi::root_output() << '\n';
      }
    }
    fused_mpi::root_output() << "PASS: selected BF16 boundaries, complete routes, changed payloads"
              << (options.profile ? ", and diagnostic timelines" : "") << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "fused_bf16: rank=" << fused_mpi::process_rank << ": " << error.what() << '\n' << std::flush;
    fused_mpi::abort(1);
    // No cudaDeviceReset or unbounded teardown after a partial multi-GPU launch.
    ::_exit(1);
  }
}
