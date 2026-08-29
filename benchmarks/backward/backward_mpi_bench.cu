#include "fuse/operators/oproj_backward.h"
#include "fuse/operators/qkv_backward.h"

#include <cuda_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using fuse::Bf16;

#define CUDA_CHECK(expr) check_cuda((expr), #expr, __FILE__, __LINE__)
#define MPI_CHECK(expr) check_mpi((expr), #expr, __FILE__, __LINE__)

void check_cuda(
    cudaError_t status,
    const char* expression,
    const char* file,
    int line) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(file) + ":" + std::to_string(line) + " " + expression +
        ": " + cudaGetErrorString(status));
  }
}

void check_mpi(
    int status,
    const char* expression,
    const char* file,
    int line) {
  if (status != MPI_SUCCESS) {
    char message[MPI_MAX_ERROR_STRING]{};
    int length = 0;
    MPI_Error_string(status, message, &length);
    throw std::runtime_error(
        std::string(file) + ":" + std::to_string(line) + " " + expression +
        ": " + std::string(message, static_cast<size_t>(length)));
  }
}

enum class OperatorKind {
  kQkv,
  kOproj,
};

enum class WeightMode {
  kImmediate,
  kDeferred,
};

enum class LaunchMode {
  kEager,
  kGraph,
};

struct Options {
  OperatorKind operator_kind = OperatorKind::kQkv;
  WeightMode weight_mode = WeightMode::kImmediate;
  LaunchMode launch_mode = LaunchMode::kEager;
  int m = 128;
  int hidden = 4096;
  int batch = 1;
  int q_heads = 16;
  int kv_heads = 8;
  int head_dim = 128;
  int comm_ctas = 0;
  int weight_beta = -1;
  fuse::BackwardGemmPolicy gemm_policy = fuse::BackwardGemmPolicy::kAuto;
  int warmup = 10;
  int iterations = 50;
  bool causal_load_balanced = false;
  bool check = true;
  bool help = false;
  bool run_all_modes = false;
#if FUSE_ENABLE_PROFILING
  bool role_profile = false;
#endif
  std::string json_out;
  std::string json_prefix;
};

const char* operator_name(OperatorKind kind) {
  return kind == OperatorKind::kQkv
      ? "qkv_backward_a2a_dgrad"
      : "oproj_backward_dgrad_a2a";
}

const char* weight_mode_name(WeightMode mode) {
  return mode == WeightMode::kImmediate ? "immediate" : "deferred";
}

const char* launch_mode_name(LaunchMode mode) {
  return mode == LaunchMode::kGraph ? "graph" : "eager";
}

const char* gemm_policy_name(fuse::BackwardGemmPolicy policy) {
  switch (policy) {
    case fuse::BackwardGemmPolicy::kAuto:
      return "auto";
    case fuse::BackwardGemmPolicy::kM128N64:
      return "m128n64";
    case fuse::BackwardGemmPolicy::kM128N128:
      return "m128n128";
    case fuse::BackwardGemmPolicy::kM128N160:
      return "m128n160";
    case fuse::BackwardGemmPolicy::kM128N192:
      return "m128n192";
    case fuse::BackwardGemmPolicy::kM128N256:
      return "m128n256";
    case fuse::BackwardGemmPolicy::kM128N64ClusterM2:
      return "m128n64c2";
    default:
      return "invalid";
  }
}

int parse_positive(const std::string& value, const char* name) {
  const long long parsed = std::stoll(value);
  if (parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
    throw std::runtime_error(std::string(name) + " must be a positive int");
  }
  return static_cast<int>(parsed);
}

int parse_nonnegative(const std::string& value, const char* name) {
  const long long parsed = std::stoll(value);
  if (parsed < 0 || parsed > std::numeric_limits<int>::max()) {
    throw std::runtime_error(std::string(name) + " must be a nonnegative int");
  }
  return static_cast<int>(parsed);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto take = [&](const char* name) {
      if (++index == argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      return std::string(argv[index]);
    };
    if (argument == "--operator") {
      const std::string value = take("--operator");
      if (value == "qkv") {
        options.operator_kind = OperatorKind::kQkv;
      } else if (value == "oproj") {
        options.operator_kind = OperatorKind::kOproj;
      } else {
        throw std::runtime_error("--operator must be qkv or oproj");
      }
    } else if (argument == "--weight-mode") {
      const std::string value = take("--weight-mode");
      if (value == "immediate") {
        options.weight_mode = WeightMode::kImmediate;
      } else if (value == "deferred") {
        options.weight_mode = WeightMode::kDeferred;
      } else {
        throw std::runtime_error(
            "--weight-mode must be immediate or deferred");
      }
    } else if (argument == "--launch") {
      const std::string value = take("--launch");
      if (value == "eager") {
        options.launch_mode = LaunchMode::kEager;
      } else if (value == "graph") {
        options.launch_mode = LaunchMode::kGraph;
      } else {
        throw std::runtime_error("--launch must be eager or graph");
      }
    } else if (argument == "--cuda-graph") {
      options.launch_mode = LaunchMode::kGraph;
    } else if (argument == "--m") {
      options.m = parse_positive(take("--m"), "--m");
    } else if (argument == "--hidden") {
      options.hidden = parse_positive(take("--hidden"), "--hidden");
    } else if (argument == "--batch") {
      options.batch = parse_positive(take("--batch"), "--batch");
    } else if (argument == "--q-heads") {
      options.q_heads = parse_positive(take("--q-heads"), "--q-heads");
    } else if (argument == "--kv-heads") {
      options.kv_heads = parse_positive(take("--kv-heads"), "--kv-heads");
    } else if (argument == "--head-dim") {
      options.head_dim = parse_positive(take("--head-dim"), "--head-dim");
    } else if (argument == "--comm-ctas") {
      options.comm_ctas =
          parse_nonnegative(take("--comm-ctas"), "--comm-ctas");
    } else if (argument == "--weight-beta") {
      options.weight_beta =
          parse_nonnegative(take("--weight-beta"), "--weight-beta");
      if (options.weight_beta > 1) {
        throw std::runtime_error("--weight-beta must be 0 or 1");
      }
    } else if (argument == "--gemm-policy") {
      const std::string value = take("--gemm-policy");
      if (value == "auto") {
        options.gemm_policy = fuse::BackwardGemmPolicy::kAuto;
      } else if (value == "m128n64") {
        options.gemm_policy = fuse::BackwardGemmPolicy::kM128N64;
      } else if (value == "m128n128") {
        options.gemm_policy = fuse::BackwardGemmPolicy::kM128N128;
      } else if (value == "m128n160") {
        options.gemm_policy = fuse::BackwardGemmPolicy::kM128N160;
      } else if (value == "m128n192") {
        options.gemm_policy = fuse::BackwardGemmPolicy::kM128N192;
      } else if (value == "m128n256") {
        options.gemm_policy = fuse::BackwardGemmPolicy::kM128N256;
      } else if (value == "m128n64c2") {
        options.gemm_policy = fuse::BackwardGemmPolicy::kM128N64ClusterM2;
      } else {
        throw std::runtime_error("unknown --gemm-policy value: " + value);
      }
    } else if (argument == "--warmup") {
      options.warmup =
          parse_nonnegative(take("--warmup"), "--warmup");
    } else if (argument == "--iterations") {
      options.iterations =
          parse_positive(take("--iterations"), "--iterations");
    } else if (argument == "--causal-load-balanced") {
      options.causal_load_balanced = true;
    } else if (argument == "--check") {
      options.check = true;
    } else if (argument == "--no-check") {
      options.check = false;
    } else if (argument == "--json-out") {
      options.json_out = take("--json-out");
    } else if (argument == "--json-prefix") {
      options.json_prefix = take("--json-prefix");
    } else if (argument == "--run-all-modes") {
      options.run_all_modes = true;
#if FUSE_ENABLE_PROFILING
    } else if (argument == "--role-profile") {
      options.role_profile = true;
#endif
    } else if (argument == "--help" || argument == "-h") {
      options.help = true;
    } else {
      throw std::runtime_error("unknown argument: " + argument);
    }
  }
  if (options.weight_beta < 0) {
    options.weight_beta =
        options.weight_mode == WeightMode::kDeferred ? 1 : 0;
  }
  return options;
}

void print_usage(const char* program) {
  std::cout
      << "Usage: mpirun -np <2..8> " << program << " [options]\n"
      << "  --operator qkv|oproj\n"
      << "  --weight-mode immediate|deferred\n"
      << "  --launch eager|graph\n"
      << "  --m N --hidden N --batch N\n"
      << "  --q-heads N --kv-heads N --head-dim N\n"
      << "  --comm-ctas N          0 selects the operator recommendation\n"
      << "  --weight-beta 0|1      defaults: immediate=0, deferred=1\n"
      << "  --gemm-policy auto|m128n64|m128n64c2|m128n128|m128n160|m128n192|m128n256\n"
      << "  --warmup N --iterations N   defaults: 10 and 50\n"
      << "  --causal-load-balanced --check|--no-check --json-out PATH\n"
      << "  --run-all-modes --json-prefix PATH  run all four launch/mode pairs\n"
#if FUSE_ENABLE_PROFILING
      << "  --role-profile         one diagnostic QKV B epoch after timing\n"
#endif
      ;
}

struct RankContext {
  MPI_Comm local_comm = MPI_COMM_NULL;
  int world_rank = 0;
  int world_size = 0;
  int rank = 0;
  int world = 0;
  int device = 0;
};

RankContext initialize_rank_context() {
  RankContext context;
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &context.world_rank));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &context.world_size));
  MPI_CHECK(MPI_Comm_split_type(
      MPI_COMM_WORLD,
      MPI_COMM_TYPE_SHARED,
      context.world_rank,
      MPI_INFO_NULL,
      &context.local_comm));
  MPI_CHECK(MPI_Comm_set_errhandler(context.local_comm, MPI_ERRORS_RETURN));
  MPI_CHECK(MPI_Comm_rank(context.local_comm, &context.rank));
  MPI_CHECK(MPI_Comm_size(context.local_comm, &context.world));
  if (context.world != context.world_size) {
    throw std::runtime_error(
        "CUDA IPC benchmark requires all MPI ranks on one host");
  }
  if (context.world < 2 || context.world > fuse::kMaxWorldSize) {
    throw std::runtime_error("MPI world size must be between 2 and 8");
  }

  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count == 1) {
    context.device = 0;
  } else {
    if (device_count < context.world) {
      throw std::runtime_error(
          "each rank must see one GPU or the same world-sized GPU set");
    }
    context.device = context.rank;
  }
  CUDA_CHECK(cudaSetDevice(context.device));

  std::array<char, 32> local_bus_id{};
  CUDA_CHECK(cudaDeviceGetPCIBusId(
      local_bus_id.data(),
      static_cast<int>(local_bus_id.size()),
      context.device));
  std::vector<char> all_bus_ids(
      static_cast<size_t>(context.world) * local_bus_id.size());
  MPI_CHECK(MPI_Allgather(
      local_bus_id.data(),
      static_cast<int>(local_bus_id.size()),
      MPI_BYTE,
      all_bus_ids.data(),
      static_cast<int>(local_bus_id.size()),
      MPI_BYTE,
      context.local_comm));
  for (int lhs = 0; lhs < context.world; ++lhs) {
    for (int rhs = lhs + 1; rhs < context.world; ++rhs) {
      const char* a = all_bus_ids.data() +
          static_cast<size_t>(lhs) * local_bus_id.size();
      const char* b = all_bus_ids.data() +
          static_cast<size_t>(rhs) * local_bus_id.size();
      if (std::equal(a, a + local_bus_id.size(), b)) {
        throw std::runtime_error("two ranks selected the same physical GPU");
      }
    }
  }
  return context;
}

struct Runtime {
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
};

Runtime initialize_runtime() {
  Runtime runtime;
  int least_priority = 0;
  int greatest_priority = 0;
  CUDA_CHECK(cudaDeviceGetStreamPriorityRange(
      &least_priority, &greatest_priority));
  CUDA_CHECK(cudaStreamCreateWithPriority(
      &runtime.stream, cudaStreamNonBlocking, greatest_priority));
  CUDA_CHECK(cudaEventCreate(&runtime.start));
  CUDA_CHECK(cudaEventCreate(&runtime.stop));
  return runtime;
}

template <class T>
T* allocate(int64_t elements) {
  if (elements <= 0 ||
      static_cast<uint64_t>(elements) >
          std::numeric_limits<size_t>::max() / sizeof(T)) {
    throw std::runtime_error("invalid allocation size");
  }
  T* pointer = nullptr;
  CUDA_CHECK(cudaMalloc(
      &pointer, static_cast<size_t>(elements) * sizeof(T)));
  return pointer;
}

__global__ void fill_kernel(Bf16* data, int64_t elements, int seed) {
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int value = static_cast<int>((index * 17 + seed * 13) % 19) - 9;
    data[index] = Bf16(static_cast<float>(value) / 128.0f);
  }
}

__global__ void count_pattern_kernel(
    const uint16_t* data,
    int64_t elements,
    uint16_t pattern,
    unsigned long long* count) {
  unsigned long long local = 0;
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    local += data[index] == pattern;
  }
  if (local != 0) {
    atomicAdd(count, local);
  }
}

void fill(Bf16* pointer, int64_t elements, int seed, cudaStream_t stream) {
  const int blocks = static_cast<int>(
      std::min<int64_t>((elements + 255) / 256, 4096));
  fill_kernel<<<blocks, 256, 0, stream>>>(pointer, elements, seed);
  CUDA_CHECK(cudaGetLastError());
}

struct IpcHandles {
  cudaIpcMemHandle_t output{};
  cudaIpcMemHandle_t ready{};
  cudaIpcMemHandle_t epoch{};
};

static_assert(std::is_trivially_copyable_v<IpcHandles>);

struct Buffers {
  Bf16* grad_q = nullptr;
  Bf16* grad_k = nullptr;
  Bf16* grad_v = nullptr;
  Bf16* grad_output = nullptr;
  Bf16* weight = nullptr;
  Bf16* saved_input = nullptr;
  Bf16* local_intermediate = nullptr;
  Bf16* grad_input = nullptr;
  Bf16* grad_weight = nullptr;
  Bf16* owned_output = nullptr;
  uint32_t* owned_ready = nullptr;
  uint32_t* owned_epoch = nullptr;
  int64_t intermediate_elements = 0;
  int64_t output_elements = 0;
  int64_t grad_input_elements = 0;
  int64_t grad_weight_elements = 0;
  int64_t ready_elements = 0;
  std::vector<Bf16*> peer_output;
  std::vector<uint32_t*> peer_ready;
  std::vector<uint32_t*> peer_epoch;
};

void initialize_sentinel(
    Bf16* pointer,
    int64_t elements,
    cudaStream_t stream) {
  CUDA_CHECK(cudaMemsetAsync(
      pointer,
      0xff,
      static_cast<size_t>(elements) * sizeof(Bf16),
      stream));
}

Buffers allocate_and_exchange_buffers(
    const Options& options,
    const RankContext& context,
    const Runtime& runtime,
    int projection_width,
    int64_t ready_elements) {
  Buffers buffers;
  buffers.ready_elements = ready_elements;
  const int64_t m = options.m;
  const int64_t hidden = options.hidden;
  const int64_t width = projection_width;
  const int64_t global_rows = m * context.world;

  if (options.operator_kind == OperatorKind::kQkv) {
    const int64_t q_local_width =
        options.q_heads / context.world * options.head_dim;
    const int64_t kv_local_width =
        options.kv_heads / context.world * options.head_dim;
    buffers.grad_q = allocate<Bf16>(global_rows * q_local_width);
    buffers.grad_k = allocate<Bf16>(global_rows * kv_local_width);
    buffers.grad_v = allocate<Bf16>(global_rows * kv_local_width);
    buffers.weight = allocate<Bf16>(width * hidden);
    buffers.saved_input = allocate<Bf16>(m * hidden);
    buffers.grad_input = allocate<Bf16>(m * hidden);
    buffers.grad_weight = allocate<Bf16>(width * hidden);
    buffers.owned_output = allocate<Bf16>(m * width);
    buffers.local_intermediate = buffers.owned_output;
    buffers.intermediate_elements = m * width;
    buffers.output_elements = m * width;
    buffers.grad_input_elements = m * hidden;
    buffers.grad_weight_elements = width * hidden;
    fill(buffers.grad_q, global_rows * q_local_width, context.rank + 101,
         runtime.stream);
    fill(buffers.grad_k, global_rows * kv_local_width, context.rank + 201,
         runtime.stream);
    fill(buffers.grad_v, global_rows * kv_local_width, context.rank + 301,
         runtime.stream);
  } else {
    buffers.grad_output = allocate<Bf16>(m * hidden);
    buffers.weight = allocate<Bf16>(hidden * width);
    buffers.saved_input = allocate<Bf16>(m * width);
    buffers.local_intermediate = allocate<Bf16>(m * width);
    buffers.grad_input = buffers.local_intermediate;
    buffers.grad_weight = allocate<Bf16>(hidden * width);
    buffers.owned_output = allocate<Bf16>(m * width);
    buffers.intermediate_elements = m * width;
    buffers.output_elements = m * width;
    buffers.grad_input_elements = m * width;
    buffers.grad_weight_elements = hidden * width;
    fill(buffers.grad_output, m * hidden, context.rank + 401, runtime.stream);
  }
  fill(
      buffers.weight,
      buffers.grad_weight_elements,
      context.rank + 501,
      runtime.stream);
  fill(
      buffers.saved_input,
      options.operator_kind == OperatorKind::kQkv ? m * hidden : m * width,
      context.rank + 601,
      runtime.stream);
  initialize_sentinel(
      buffers.local_intermediate,
      buffers.intermediate_elements,
      runtime.stream);
  initialize_sentinel(
      buffers.grad_input,
      buffers.grad_input_elements,
      runtime.stream);
  if (options.weight_beta == 0) {
    initialize_sentinel(
        buffers.grad_weight,
        buffers.grad_weight_elements,
        runtime.stream);
  } else {
    CUDA_CHECK(cudaMemsetAsync(
        buffers.grad_weight,
        0,
        static_cast<size_t>(buffers.grad_weight_elements) * sizeof(Bf16),
        runtime.stream));
  }
  if (buffers.owned_output != buffers.local_intermediate) {
    initialize_sentinel(
        buffers.owned_output, buffers.output_elements, runtime.stream);
  }
  buffers.owned_ready = allocate<uint32_t>(ready_elements);
  buffers.owned_epoch = allocate<uint32_t>(
      static_cast<int64_t>(context.world) * fuse::kReadyFlagStride);
  CUDA_CHECK(cudaMemsetAsync(
      buffers.owned_ready,
      0,
      static_cast<size_t>(ready_elements) * sizeof(uint32_t),
      runtime.stream));
  CUDA_CHECK(cudaMemsetAsync(
      buffers.owned_epoch,
      0,
      static_cast<size_t>(context.world) * fuse::kReadyFlagStride *
          sizeof(uint32_t),
      runtime.stream));
  CUDA_CHECK(cudaStreamSynchronize(runtime.stream));

  IpcHandles local_handles{};
  CUDA_CHECK(cudaIpcGetMemHandle(
      &local_handles.output, buffers.owned_output));
  CUDA_CHECK(cudaIpcGetMemHandle(
      &local_handles.ready, buffers.owned_ready));
  CUDA_CHECK(cudaIpcGetMemHandle(
      &local_handles.epoch, buffers.owned_epoch));
  std::vector<IpcHandles> handles(context.world);
  MPI_CHECK(MPI_Allgather(
      &local_handles,
      static_cast<int>(sizeof(IpcHandles)),
      MPI_BYTE,
      handles.data(),
      static_cast<int>(sizeof(IpcHandles)),
      MPI_BYTE,
      context.local_comm));

  buffers.peer_output.resize(context.world);
  buffers.peer_ready.resize(context.world);
  buffers.peer_epoch.resize(context.world);
  for (int peer = 0; peer < context.world; ++peer) {
    if (peer == context.rank) {
      buffers.peer_output[peer] = buffers.owned_output;
      buffers.peer_ready[peer] = buffers.owned_ready;
      buffers.peer_epoch[peer] = buffers.owned_epoch;
      continue;
    }
    CUDA_CHECK(cudaIpcOpenMemHandle(
        reinterpret_cast<void**>(&buffers.peer_output[peer]),
        handles[peer].output,
        cudaIpcMemLazyEnablePeerAccess));
    CUDA_CHECK(cudaIpcOpenMemHandle(
        reinterpret_cast<void**>(&buffers.peer_ready[peer]),
        handles[peer].ready,
        cudaIpcMemLazyEnablePeerAccess));
    CUDA_CHECK(cudaIpcOpenMemHandle(
        reinterpret_cast<void**>(&buffers.peer_epoch[peer]),
        handles[peer].epoch,
        cudaIpcMemLazyEnablePeerAccess));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  MPI_CHECK(MPI_Barrier(context.local_comm));
  return buffers;
}

struct OperatorState {
  fuse::QkvBackwardParams qkv{};
  fuse::OprojBackwardParams oproj{};
  int resolved_comm_ctas = 0;
  fuse::BackwardGemmPolicy resolved_gemm_policy =
      fuse::BackwardGemmPolicy::kAuto;
  fuse::KernelTraits traits{};
};

OperatorState make_operator_state(
    const Options& options,
    const RankContext& context,
    const Buffers& buffers) {
  OperatorState state;
  if (options.operator_kind == OperatorKind::kQkv) {
    auto& data = state.qkv.data;
    data.grad_q = buffers.grad_q;
    data.grad_k = buffers.grad_k;
    data.grad_v = buffers.grad_v;
    for (int peer = 0; peer < context.world; ++peer) {
      data.peer_dqkv_staging[peer] = buffers.peer_output[peer];
      data.peer_ready[peer] = buffers.peer_ready[peer];
      data.peer_done_epoch[peer] = buffers.peer_epoch[peer];
    }
    data.weight = buffers.weight;
    data.grad_input = buffers.grad_input;
    data.local_tokens = options.m;
    data.hidden = options.hidden;
    data.batch = options.batch;
    data.q_heads = options.q_heads;
    data.kv_heads = options.kv_heads;
    data.head_dim = options.head_dim;
    data.world_size = context.world;
    data.rank = context.rank;
    data.num_comm_ctas = options.comm_ctas;
    data.gemm_policy = options.gemm_policy;
    data.epoch = 1;
    data.causal_load_balanced = options.causal_load_balanced;
    state.resolved_comm_ctas = options.comm_ctas == 0
        ? fuse::recommended_qkv_backward_comm_ctas(data)
        : options.comm_ctas;
    int sm_count = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &sm_count, cudaDevAttrMultiProcessorCount, context.device));
    state.resolved_gemm_policy =
        fuse::recommended_qkv_backward_gemm_policy(
            data, state.resolved_comm_ctas, sm_count);
    state.traits = fuse::qkv_backward_kernel_traits(
        data, state.resolved_comm_ctas, sm_count);
    state.qkv.weight.dqkv_staging = buffers.owned_output;
    state.qkv.weight.saved_input = buffers.saved_input;
    state.qkv.weight.grad_weight = buffers.grad_weight;
    state.qkv.weight.local_tokens = options.m;
    state.qkv.weight.hidden = options.hidden;
    state.qkv.weight.q_heads = options.q_heads;
    state.qkv.weight.kv_heads = options.kv_heads;
    state.qkv.weight.head_dim = options.head_dim;
    state.qkv.weight.beta = static_cast<float>(options.weight_beta);
    state.qkv.weight_mode = options.weight_mode == WeightMode::kImmediate
        ? fuse::WeightGradientMode::kImmediate
        : fuse::WeightGradientMode::kDeferred;
  } else {
    auto& data = state.oproj.data;
    data.grad_output = buffers.grad_output;
    data.weight = buffers.weight;
    data.local_grad_attention = buffers.local_intermediate;
    for (int peer = 0; peer < context.world; ++peer) {
      data.peer_grad_attention[peer] = buffers.peer_output[peer];
      data.peer_done_epoch[peer] = buffers.peer_epoch[peer];
    }
    data.ready = buffers.owned_ready;
    data.local_tokens = options.m;
    data.hidden = options.hidden;
    data.batch = options.batch;
    data.q_heads = options.q_heads;
    data.head_dim = options.head_dim;
    data.world_size = context.world;
    data.rank = context.rank;
    data.num_comm_ctas = options.comm_ctas;
    data.gemm_policy = options.gemm_policy;
    data.epoch = 1;
    data.causal_load_balanced = options.causal_load_balanced;
    state.resolved_comm_ctas = options.comm_ctas == 0
        ? fuse::recommended_oproj_backward_comm_ctas(data)
        : options.comm_ctas;
    int sm_count = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &sm_count, cudaDevAttrMultiProcessorCount, context.device));
    state.resolved_gemm_policy =
        fuse::recommended_oproj_backward_gemm_policy(
            data, state.resolved_comm_ctas, sm_count);
    state.traits = fuse::oproj_backward_kernel_traits(
        data, state.resolved_comm_ctas, sm_count);
    state.oproj.weight.grad_output = buffers.grad_output;
    state.oproj.weight.saved_attention = buffers.saved_input;
    state.oproj.weight.grad_weight = buffers.grad_weight;
    state.oproj.weight.local_tokens = options.m;
    state.oproj.weight.hidden = options.hidden;
    state.oproj.weight.q_heads = options.q_heads;
    state.oproj.weight.head_dim = options.head_dim;
    state.oproj.weight.beta = static_cast<float>(options.weight_beta);
    state.oproj.weight_mode = options.weight_mode == WeightMode::kImmediate
        ? fuse::WeightGradientMode::kImmediate
        : fuse::WeightGradientMode::kDeferred;
  }
  return state;
}

void launch_data(
    const Options& options,
    OperatorState& state,
    uint32_t epoch,
    cudaStream_t stream) {
  if (options.operator_kind == OperatorKind::kQkv) {
    state.qkv.data.epoch = epoch;
    CUDA_CHECK(fuse::launch_qkv_backward_data(state.qkv.data, stream));
  } else {
    state.oproj.data.epoch = epoch;
    CUDA_CHECK(fuse::launch_oproj_backward_data(state.oproj.data, stream));
  }
}

void launch_weight(
    const Options& options,
    OperatorState& state,
    cudaStream_t stream) {
  if (options.operator_kind == OperatorKind::kQkv) {
    CUDA_CHECK(fuse::launch_qkv_backward_weight(state.qkv.weight, stream));
  } else {
    CUDA_CHECK(fuse::launch_oproj_backward_weight(state.oproj.weight, stream));
  }
}

void launch_immediate(
    const Options& options,
    OperatorState& state,
    uint32_t epoch,
    cudaStream_t stream) {
  if (options.operator_kind == OperatorKind::kQkv) {
    state.qkv.data.epoch = epoch;
    CUDA_CHECK(fuse::launch_qkv_backward(state.qkv, stream));
  } else {
    state.oproj.data.epoch = epoch;
    CUDA_CHECK(fuse::launch_oproj_backward(state.oproj, stream));
  }
}

template <class Launch>
std::vector<float> time_eager(
    const Options& options,
    const RankContext& context,
    const Runtime& runtime,
    Launch launch) {
  for (int step = 0; step < options.warmup; ++step) {
    MPI_CHECK(MPI_Barrier(context.local_comm));
    launch(step);
    CUDA_CHECK(cudaStreamSynchronize(runtime.stream));
  }
  std::vector<float> samples;
  samples.reserve(options.iterations);
  for (int step = 0; step < options.iterations; ++step) {
    MPI_CHECK(MPI_Barrier(context.local_comm));
    CUDA_CHECK(cudaEventRecord(runtime.start, runtime.stream));
    launch(options.warmup + step);
    CUDA_CHECK(cudaEventRecord(runtime.stop, runtime.stream));
    CUDA_CHECK(cudaEventSynchronize(runtime.stop));
    float local_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(
        &local_ms, runtime.start, runtime.stop));
    float critical_ms = 0.0f;
    MPI_CHECK(MPI_Allreduce(
        &local_ms,
        &critical_ms,
        1,
        MPI_FLOAT,
        MPI_MAX,
        context.local_comm));
    samples.push_back(critical_ms);
  }
  return samples;
}

// Capture all warmup and measured steps into one linear graph and replay it
// once. Each measured step gets its own event pair inside the graph. This is
// the same timing boundary as the forward MPI benchmark: graph construction,
// instantiation, upload, and the host cudaGraphLaunch call are excluded, while
// the B/W kernels and their cross-rank completion waits remain included.
template <class Launch>
std::vector<float> time_graph(
    const Options& options,
    const RankContext& context,
    const Runtime& runtime,
    int kernels_per_step,
    Launch launch) {
  if (kernels_per_step <= 0) {
    throw std::runtime_error("kernels_per_step must be positive");
  }
  std::vector<cudaEvent_t> starts(options.iterations, nullptr);
  std::vector<cudaEvent_t> stops(options.iterations, nullptr);
  for (int sample = 0; sample < options.iterations; ++sample) {
    CUDA_CHECK(cudaEventCreate(&starts[sample]));
    CUDA_CHECK(cudaEventCreate(&stops[sample]));
  }

  const int total_steps = options.warmup + options.iterations;
  const size_t expected_kernels =
      static_cast<size_t>(total_steps) * kernels_per_step;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(
      runtime.stream, cudaStreamCaptureModeThreadLocal));
  for (int step = 0; step < total_steps; ++step) {
    launch(step);
  }
  CUDA_CHECK(cudaStreamEndCapture(runtime.stream, &graph));

  size_t node_count = 0;
  CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &node_count));
  std::vector<cudaGraphNode_t> nodes(node_count);
  CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &node_count));
  if (node_count != expected_kernels) {
    throw std::runtime_error(
        "captured graph node count mismatch: nodes=" +
        std::to_string(node_count) + " expected kernels=" +
        std::to_string(expected_kernels));
  }
  for (cudaGraphNode_t node : nodes) {
    cudaGraphNodeType type{};
    CUDA_CHECK(cudaGraphNodeGetType(node, &type));
    if (type != cudaGraphNodeTypeKernel) {
      throw std::runtime_error("captured graph contains a non-kernel node");
    }
  }

  // cudaGraphGetNodes does not promise capture order. Recover the one linear
  // stream-capture chain before inserting event nodes around whole B or B+W
  // steps.
  size_t edge_count = 0;
  CUDA_CHECK(cudaGraphGetEdges(graph, nullptr, nullptr, &edge_count));
  std::vector<cudaGraphNode_t> edge_from(edge_count);
  std::vector<cudaGraphNode_t> edge_to(edge_count);
  CUDA_CHECK(cudaGraphGetEdges(
      graph, edge_from.data(), edge_to.data(), &edge_count));
  if (edge_count + 1 != node_count) {
    throw std::runtime_error("captured kernels do not form one linear chain");
  }
  std::vector<int> indegree(node_count, 0);
  std::vector<int> successor(node_count, -1);
  auto node_index = [&](cudaGraphNode_t target) {
    const auto found = std::find(nodes.begin(), nodes.end(), target);
    if (found == nodes.end()) {
      throw std::runtime_error("graph edge references an unknown node");
    }
    return static_cast<int>(found - nodes.begin());
  };
  for (size_t edge = 0; edge < edge_count; ++edge) {
    const int from = node_index(edge_from[edge]);
    const int to = node_index(edge_to[edge]);
    if (successor[from] != -1) {
      throw std::runtime_error("captured graph node has multiple successors");
    }
    successor[from] = to;
    ++indegree[to];
  }
  int current = -1;
  for (size_t node = 0; node < node_count; ++node) {
    if (indegree[node] == 0) {
      if (current != -1) {
        throw std::runtime_error("captured graph has multiple roots");
      }
      current = static_cast<int>(node);
    }
  }
  std::vector<cudaGraphNode_t> kernels;
  kernels.reserve(node_count);
  while (current != -1) {
    kernels.push_back(nodes[current]);
    current = successor[current];
  }
  if (kernels.size() != node_count) {
    throw std::runtime_error("captured graph kernel chain is incomplete");
  }
  if (edge_count != 0) {
    CUDA_CHECK(cudaGraphRemoveDependencies(
        graph, edge_from.data(), edge_to.data(), edge_count));
  }

  cudaGraphNode_t previous = nullptr;
  for (int step = 0; step < total_steps; ++step) {
    const int sample = step - options.warmup;
    if (sample >= 0) {
      cudaGraphNode_t start_node = nullptr;
      CUDA_CHECK(cudaGraphAddEventRecordNode(
          &start_node,
          graph,
          previous ? &previous : nullptr,
          previous ? 1 : 0,
          starts[sample]));
      previous = start_node;
    }
    for (int part = 0; part < kernels_per_step; ++part) {
      const cudaGraphNode_t kernel =
          kernels[static_cast<size_t>(step) * kernels_per_step + part];
      if (previous) {
        CUDA_CHECK(cudaGraphAddDependencies(graph, &previous, &kernel, 1));
      }
      previous = kernel;
    }
    if (sample >= 0) {
      cudaGraphNode_t stop_node = nullptr;
      CUDA_CHECK(cudaGraphAddEventRecordNode(
          &stop_node, graph, &previous, 1, stops[sample]));
      previous = stop_node;
    }
  }

  node_count = 0;
  CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &node_count));
  nodes.resize(node_count);
  CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &node_count));
  size_t kernel_nodes = 0;
  size_t event_nodes = 0;
  for (cudaGraphNode_t node : nodes) {
    cudaGraphNodeType type{};
    CUDA_CHECK(cudaGraphNodeGetType(node, &type));
    kernel_nodes += type == cudaGraphNodeTypeKernel;
    event_nodes += type == cudaGraphNodeTypeEventRecord;
  }
  if (kernel_nodes != expected_kernels ||
      event_nodes != static_cast<size_t>(2 * options.iterations)) {
    throw std::runtime_error(
        "final graph structure mismatch: kernels=" +
        std::to_string(kernel_nodes) + " event_records=" +
        std::to_string(event_nodes));
  }

  CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  CUDA_CHECK(cudaGraphUpload(graph_exec, runtime.stream));
  CUDA_CHECK(cudaStreamSynchronize(runtime.stream));
  MPI_CHECK(MPI_Barrier(context.local_comm));
  CUDA_CHECK(cudaGraphLaunch(graph_exec, runtime.stream));
  CUDA_CHECK(cudaStreamSynchronize(runtime.stream));

  std::vector<float> local(options.iterations);
  std::vector<float> critical(options.iterations);
  for (int sample = 0; sample < options.iterations; ++sample) {
    CUDA_CHECK(cudaEventElapsedTime(
        &local[sample], starts[sample], stops[sample]));
  }
  MPI_CHECK(MPI_Allreduce(
      local.data(),
      critical.data(),
      options.iterations,
      MPI_FLOAT,
      MPI_MAX,
      context.local_comm));

  CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
  CUDA_CHECK(cudaGraphDestroy(graph));
  for (int sample = 0; sample < options.iterations; ++sample) {
    CUDA_CHECK(cudaEventDestroy(starts[sample]));
    CUDA_CHECK(cudaEventDestroy(stops[sample]));
  }
  return critical;
}

float percentile(std::vector<float> values, double quantile) {
  std::sort(values.begin(), values.end());
  const double position = (values.size() - 1) * quantile;
  const size_t lower = static_cast<size_t>(std::floor(position));
  const size_t upper = static_cast<size_t>(std::ceil(position));
  const double weight = position - lower;
  return static_cast<float>(
      values[lower] * (1.0 - weight) + values[upper] * weight);
}

struct Summary {
  double mean = 0.0;
  float p50 = 0.0f;
  float p95 = 0.0f;
  float minimum = 0.0f;
  float maximum = 0.0f;
};

Summary summarize(const std::vector<float>& samples) {
  return {
      std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size(),
      percentile(samples, 0.50),
      percentile(samples, 0.95),
      *std::min_element(samples.begin(), samples.end()),
      *std::max_element(samples.begin(), samples.end())};
}

#if FUSE_ENABLE_PROFILING
struct RoleProfileSummary {
  int32_t rank = 0;
  int32_t observed_ctas = 0;
  int32_t compute_ctas = 0;
  int32_t route_ctas = 0;
  double compute_role_us = 0.0;
  double route_role_us = 0.0;
  double overlap_us = 0.0;
  double grid_sync_us = 0.0;
  double finalize_us = 0.0;
  double kernel_us = 0.0;
};

static_assert(std::is_trivially_copyable_v<RoleProfileSummary>);

double timer_span_us(uint64_t begin, uint64_t end) {
  return end > begin ? static_cast<double>(end - begin) / 1000.0 : 0.0;
}

RoleProfileSummary summarize_role_profile(
    int rank,
    int comm_ctas,
    int world,
    const std::vector<fuse::A2AGemmCtaTimeline>& timeline) {
  uint64_t first = std::numeric_limits<uint64_t>::max();
  uint64_t last = 0;
  uint64_t compute_first = std::numeric_limits<uint64_t>::max();
  uint64_t compute_done = 0;
  uint64_t route_first = std::numeric_limits<uint64_t>::max();
  uint64_t route_done = 0;
  uint64_t local_roles_done = 0;
  RoleProfileSummary summary{};
  summary.rank = rank;
  for (int32_t cta = 0; cta < static_cast<int32_t>(timeline.size()); ++cta) {
    const auto& event = timeline[cta];
    if (event.start == 0 || event.end <= event.start ||
        event.role_done < event.start || event.role_done > event.end) {
      continue;
    }
    ++summary.observed_ctas;
    first = std::min(first, event.start);
    last = std::max(last, event.end);
    local_roles_done = std::max(local_roles_done, event.role_done);
    if (cta < comm_ctas) {
      ++summary.route_ctas;
      route_first = std::min(route_first, event.start);
      route_done = std::max(route_done, event.role_done);
    } else {
      ++summary.compute_ctas;
      compute_first = std::min(compute_first, event.start);
      compute_done = std::max(compute_done, event.role_done);
    }
  }
  summary.compute_role_us = timer_span_us(compute_first, compute_done);
  summary.route_role_us = timer_span_us(route_first, route_done);
  summary.overlap_us = timer_span_us(
      std::max(compute_first, route_first),
      std::min(compute_done, route_done));
  summary.kernel_us = timer_span_us(first, last);
  if (timeline.empty()) {
    throw std::runtime_error("empty role-profile timeline");
  }
  const auto& root = timeline.front();
  summary.grid_sync_us = timer_span_us(local_roles_done, root.grid_sync_done);
  summary.finalize_us = timer_span_us(root.grid_sync_done, last);
  bool monotonic = root.grid_sync_done >= local_roles_done &&
      root.fence_done >= root.grid_sync_done &&
      root.publish_done >= root.fence_done;
  uint64_t sources_done = 0;
  for (int source = 0; source < world; ++source) {
    monotonic = monotonic &&
        root.source_ready[source] >= root.publish_done;
    sources_done = std::max(sources_done, root.source_ready[source]);
  }
  monotonic = monotonic && last >= sources_done;
  if (!monotonic || summary.route_ctas != comm_ctas ||
      summary.compute_ctas == 0 ||
      summary.observed_ctas != summary.route_ctas + summary.compute_ctas) {
    throw std::runtime_error(
        "invalid QKV backward role profile on rank " +
        std::to_string(rank));
  }
  return summary;
}

void run_qkv_role_profile(
    const Options& options,
    const RankContext& context,
    const Runtime& runtime,
    OperatorState& state,
    uint32_t& epoch) {
  if (options.operator_kind != OperatorKind::kQkv) {
    throw std::runtime_error("--role-profile currently supports QKV only");
  }
  int timeline_capacity = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(
      &timeline_capacity,
      cudaDevAttrMultiProcessorCount,
      context.device));
  auto* device_timeline =
      allocate<fuse::A2AGemmCtaTimeline>(timeline_capacity);
  auto params = state.qkv.data;
  params.num_comm_ctas = state.resolved_comm_ctas;
  params.gemm_policy = state.resolved_gemm_policy;

  auto launch_once = [&] {
    CUDA_CHECK(cudaMemsetAsync(
        device_timeline,
        0,
        static_cast<size_t>(timeline_capacity) *
            sizeof(fuse::A2AGemmCtaTimeline),
        runtime.stream));
    CUDA_CHECK(cudaStreamSynchronize(runtime.stream));
    ++epoch;
    params.epoch = epoch;
    MPI_CHECK(MPI_Barrier(context.local_comm));
    CUDA_CHECK(fuse::launch_qkv_backward_data_role_telemetry(
        params, device_timeline, timeline_capacity, runtime.stream));
    CUDA_CHECK(cudaStreamSynchronize(runtime.stream));
    MPI_CHECK(MPI_Barrier(context.local_comm));
  };

  // First use warms the separate diagnostic kernel; only the second epoch is
  // reported. Both remain outside formal Eager/Graph timing.
  launch_once();
  launch_once();
  std::vector<fuse::A2AGemmCtaTimeline> timeline(timeline_capacity);
  CUDA_CHECK(cudaMemcpy(
      timeline.data(),
      device_timeline,
      timeline.size() * sizeof(timeline[0]),
      cudaMemcpyDeviceToHost));
  const RoleProfileSummary local = summarize_role_profile(
      context.rank, state.resolved_comm_ctas, context.world, timeline);
  std::vector<RoleProfileSummary> ranks(
      context.rank == 0 ? static_cast<size_t>(context.world) : 0);
  MPI_CHECK(MPI_Gather(
      &local,
      static_cast<int>(sizeof(local)),
      MPI_BYTE,
      context.rank == 0 ? ranks.data() : nullptr,
      static_cast<int>(sizeof(local)),
      MPI_BYTE,
      0,
      context.local_comm));
  if (context.rank == 0) {
    std::cout
        << "\nQKV backward B role profile epoch=" << epoch << "\n"
        << "rank compute_ctas route_ctas compute_us route_us overlap_us "
           "grid_sync_us finalize_us kernel_us\n";
    for (const auto& rank : ranks) {
      std::cout << std::setw(4) << rank.rank << " "
                << std::setw(12) << rank.compute_ctas << " "
                << std::setw(10) << rank.route_ctas << " "
                << std::fixed << std::setprecision(2)
                << std::setw(10) << rank.compute_role_us << " "
                << std::setw(8) << rank.route_role_us << " "
                << std::setw(10) << rank.overlap_us << " "
                << std::setw(12) << rank.grid_sync_us << " "
                << std::setw(11) << rank.finalize_us << " "
                << std::setw(9) << rank.kernel_us << "\n";
    }
  }
  CUDA_CHECK(cudaFree(device_timeline));
}
#endif

unsigned long long count_pattern(
    Bf16* pointer,
    int64_t elements,
    uint16_t pattern,
    cudaStream_t stream) {
  auto* device_count = allocate<unsigned long long>(1);
  CUDA_CHECK(cudaMemsetAsync(
      device_count, 0, sizeof(unsigned long long), stream));
  const int blocks = static_cast<int>(
      std::min<int64_t>((elements + 255) / 256, 4096));
  count_pattern_kernel<<<blocks, 256, 0, stream>>>(
      reinterpret_cast<const uint16_t*>(pointer),
      elements,
      pattern,
      device_count);
  CUDA_CHECK(cudaGetLastError());
  unsigned long long result = 0;
  CUDA_CHECK(cudaMemcpyAsync(
      &result,
      device_count,
      sizeof(result),
      cudaMemcpyDeviceToHost,
      stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaFree(device_count));
  return result;
}

Bf16 generated_value(int64_t index, int seed) {
  const int value =
      static_cast<int>((index * 17 + seed * 13) % 19) - 9;
  return Bf16(static_cast<float>(value) / 128.0f);
}

int source_sequence_row(
    const Options& options,
    const RankContext& context,
    int destination_rank,
    int local_sequence) {
  const int seq_local = options.m / options.batch;
  if (!options.causal_load_balanced) {
    return destination_rank * seq_local + local_sequence;
  }
  const int chunk_rows = seq_local / 2;
  const int chunk = local_sequence < chunk_rows
      ? destination_rank
      : 2 * context.world - destination_rank - 1;
  return chunk * chunk_rows + local_sequence % chunk_rows;
}

// The formal S=1K cases are small enough to reconstruct the inverse QKV route
// exactly on the host. This validates the TMA path's head ownership, planar
// Q/K/V offsets, rank-major or causal row mapping, and packed [Q,K,V] order
// independently of the fused kernel. Larger cases retain the protocol and
// full-overwrite check without allocating another multi-GiB host tensor.
void validate_qkv_route_exact(
    const Options& options,
    const RankContext& context,
    const Runtime& runtime,
    const Buffers& buffers) {
  if (options.operator_kind != OperatorKind::kQkv ||
      options.m * context.world != 1024) {
    return;
  }
  const int seq_local = options.m / options.batch;
  const int global_seq = seq_local * context.world;
  const int q_local_heads = options.q_heads / context.world;
  const int kv_local_heads = options.kv_heads / context.world;
  const int q_local_width = q_local_heads * options.head_dim;
  const int kv_local_width = kv_local_heads * options.head_dim;
  const int packed_width =
      (options.q_heads + 2 * options.kv_heads) * options.head_dim;
  const int64_t elements =
      static_cast<int64_t>(options.m) * packed_width;
  std::vector<Bf16> actual(static_cast<size_t>(elements));
  std::vector<Bf16> expected(static_cast<size_t>(elements));
  CUDA_CHECK(cudaMemcpyAsync(
      actual.data(),
      buffers.owned_output,
      static_cast<size_t>(elements) * sizeof(Bf16),
      cudaMemcpyDeviceToHost,
      runtime.stream));
  CUDA_CHECK(cudaStreamSynchronize(runtime.stream));

  for (int source = 0; source < context.world; ++source) {
    for (int row = 0; row < options.m; ++row) {
      const int batch = row / seq_local;
      const int local_sequence = row - batch * seq_local;
      const int source_row = batch * global_seq + source_sequence_row(
          options, context, context.rank, local_sequence);
      for (int local_head = 0; local_head < q_local_heads; ++local_head) {
        const int global_head = source * q_local_heads + local_head;
        for (int column = 0; column < options.head_dim; ++column) {
          const int64_t source_index =
              static_cast<int64_t>(source_row) * q_local_width +
              local_head * options.head_dim + column;
          const int64_t destination_index =
              static_cast<int64_t>(row) * packed_width +
              global_head * options.head_dim + column;
          expected[static_cast<size_t>(destination_index)] =
              generated_value(source_index, source + 101);
        }
      }
      for (int local_head = 0; local_head < kv_local_heads; ++local_head) {
        const int global_head = source * kv_local_heads + local_head;
        for (int column = 0; column < options.head_dim; ++column) {
          const int64_t source_index =
              static_cast<int64_t>(source_row) * kv_local_width +
              local_head * options.head_dim + column;
          const int64_t k_index =
              static_cast<int64_t>(row) * packed_width +
              (options.q_heads + global_head) * options.head_dim + column;
          const int64_t v_index =
              static_cast<int64_t>(row) * packed_width +
              (options.q_heads + options.kv_heads + global_head) *
                  options.head_dim +
              column;
          expected[static_cast<size_t>(k_index)] =
              generated_value(source_index, source + 201);
          expected[static_cast<size_t>(v_index)] =
              generated_value(source_index, source + 301);
        }
      }
    }
  }

  unsigned long long local_errors = 0;
  for (int64_t index = 0; index < elements; ++index) {
    local_errors += std::memcmp(
        &actual[static_cast<size_t>(index)],
        &expected[static_cast<size_t>(index)],
        sizeof(Bf16)) != 0;
  }
  unsigned long long global_errors = 0;
  MPI_CHECK(MPI_Allreduce(
      &local_errors,
      &global_errors,
      1,
      MPI_UNSIGNED_LONG_LONG,
      MPI_SUM,
      context.local_comm));
  if (global_errors != 0) {
    throw std::runtime_error(
        "exact QKV inverse-route check failed: mismatches=" +
        std::to_string(global_errors));
  }
}

void validate_protocol(
    const Options& options,
    const RankContext& context,
    const Runtime& runtime,
    const Buffers& buffers,
    uint32_t final_epoch) {
  std::vector<uint32_t> ready(
      static_cast<size_t>(buffers.ready_elements));
  std::vector<uint32_t> epochs(
      static_cast<size_t>(context.world) * fuse::kReadyFlagStride);
  CUDA_CHECK(cudaMemcpyAsync(
      ready.data(),
      buffers.owned_ready,
      ready.size() * sizeof(uint32_t),
      cudaMemcpyDeviceToHost,
      runtime.stream));
  CUDA_CHECK(cudaMemcpyAsync(
      epochs.data(),
      buffers.owned_epoch,
      epochs.size() * sizeof(uint32_t),
      cudaMemcpyDeviceToHost,
      runtime.stream));
  CUDA_CHECK(cudaStreamSynchronize(runtime.stream));

  unsigned long long local_ready_errors = 0;
  for (size_t offset = 0;
       offset < ready.size();
       offset += fuse::kReadyFlagStride) {
    local_ready_errors += ready[offset] != final_epoch;
  }
  unsigned long long local_epoch_errors = 0;
  for (int source = 0; source < context.world; ++source) {
    local_epoch_errors +=
        epochs[static_cast<size_t>(source) * fuse::kReadyFlagStride] !=
        final_epoch;
  }
  unsigned long long local_errors[5] = {
      local_ready_errors,
      local_epoch_errors,
      count_pattern(
          buffers.owned_output,
          buffers.output_elements,
          0xffffu,
          runtime.stream),
      count_pattern(
          buffers.grad_input,
          buffers.grad_input_elements,
          0xffffu,
          runtime.stream),
      0};
  const unsigned long long weight_pattern_count = count_pattern(
      buffers.grad_weight,
      buffers.grad_weight_elements,
      options.weight_beta == 0 ? 0xffffu : 0u,
      runtime.stream);
  local_errors[4] = options.weight_beta == 0
      ? weight_pattern_count
      : weight_pattern_count ==
          static_cast<unsigned long long>(buffers.grad_weight_elements);
  unsigned long long global_errors[5]{};
  MPI_CHECK(MPI_Allreduce(
      local_errors,
      global_errors,
      5,
      MPI_UNSIGNED_LONG_LONG,
      MPI_SUM,
      context.local_comm));
  if (std::any_of(
          std::begin(global_errors),
          std::end(global_errors),
          [](unsigned long long value) { return value != 0; })) {
    throw std::runtime_error(
        "protocol/full-overwrite check failed: ready=" +
        std::to_string(global_errors[0]) + " epoch=" +
        std::to_string(global_errors[1]) + " output=" +
        std::to_string(global_errors[2]) + " dgrad=" +
        std::to_string(global_errors[3]) + " wgrad=" +
        std::to_string(global_errors[4]));
  }
}

void write_summary_fields(
    std::ostream& output,
    const char* prefix,
    const Summary& summary,
    const std::vector<float>& samples,
    bool trailing_comma) {
  output << "  \"" << prefix << "\": {\n"
         << "    \"mean_ms\": " << summary.mean << ",\n"
         << "    \"p50_ms\": " << summary.p50 << ",\n"
         << "    \"p95_ms\": " << summary.p95 << ",\n"
         << "    \"min_ms\": " << summary.minimum << ",\n"
         << "    \"max_ms\": " << summary.maximum << ",\n"
         << "    \"samples_ms\": [";
  for (size_t index = 0; index < samples.size(); ++index) {
    output << (index == 0 ? "" : ", ") << samples[index];
  }
  output << "]\n  }" << (trailing_comma ? "," : "") << "\n";
}

void write_json(
    const Options& options,
    const RankContext& context,
    int projection_width,
    int resolved_comm_ctas,
    fuse::BackwardGemmPolicy resolved_gemm_policy,
    const fuse::KernelTraits& traits,
    const Summary& data,
    const std::vector<float>& data_samples,
    const Summary& weight,
    const std::vector<float>& weight_samples,
    const Summary& total,
    const std::vector<float>& total_samples) {
  if (context.rank != 0 || options.json_out.empty()) {
    return;
  }
  std::ofstream output(options.json_out);
  if (!output) {
    throw std::runtime_error("cannot open JSON output: " + options.json_out);
  }
  output << std::setprecision(10)
         << "{\n"
         << "  \"mode\": \"" << operator_name(options.operator_kind)
         << "_mpi\",\n"
         << "  \"launch\": \"" << launch_mode_name(options.launch_mode)
         << "\",\n"
         << "  \"weight_mode\": \"" << weight_mode_name(options.weight_mode)
         << "\",\n"
         << "  \"backward_schema\": \"v10_b_w_split_v1\",\n"
         << "  \"backward_policy_model\": "
            "\"route_task_wave_compute_residency_v3\",\n"
         << "  \"launch_plan_cache\": \"per_process_v1\",\n"
         << "  \"zero_bubble_contract\": \""
         << (options.weight_mode == WeightMode::kDeferred
                 ? "separate_B_and_W_with_operand_lease"
                 : "same_stream_B_then_W")
         << "\",\n"
         << "  \"weight_accumulation_beta\": "
         << options.weight_beta << ",\n"
#if FUSE_ENABLE_PROFILING
         << "  \"profiling_build\": true,\n"
         << "  \"role_profile_requested\": "
         << (options.role_profile ? "true" : "false") << ",\n"
#else
         << "  \"profiling_build\": false,\n"
         << "  \"role_profile_requested\": false,\n"
#endif
         << "  \"shape\": {\"local_tokens\": " << options.m
         << ", \"hidden\": " << options.hidden
         << ", \"projection_width\": " << projection_width << "},\n"
         << "  \"b_mnk\": [" << options.m << ", "
         << (options.operator_kind == OperatorKind::kQkv
                 ? options.hidden
                 : projection_width)
         << ", "
         << (options.operator_kind == OperatorKind::kQkv
                 ? projection_width
                 : options.hidden)
         << "],\n"
         << "  \"w_mnk\": ["
         << (options.operator_kind == OperatorKind::kQkv
                 ? projection_width
                 : options.hidden)
         << ", "
         << (options.operator_kind == OperatorKind::kQkv
                 ? options.hidden
                 : projection_width)
         << ", " << options.m << "],\n"
         << "  \"head_geometry\": {\"batch\": " << options.batch
         << ", \"q_heads\": " << options.q_heads
         << ", \"kv_heads\": " << options.kv_heads
         << ", \"head_dim\": " << options.head_dim << "},\n"
         << "  \"world_size\": " << context.world << ",\n"
         << "  \"warmup\": " << options.warmup << ",\n"
         << "  \"iterations\": " << options.iterations << ",\n"
         << "  \"requested_comm_ctas\": " << options.comm_ctas << ",\n"
         << "  \"comm_ctas\": " << resolved_comm_ctas << ",\n"
         << "  \"requested_gemm_policy\": \""
         << gemm_policy_name(options.gemm_policy) << "\",\n"
         << "  \"gemm_policy\": \""
         << gemm_policy_name(resolved_gemm_policy) << "\",\n"
         << "  \"kernel_traits\": {\"tile_m\": " << traits.block_m
         << ", \"tile_n\": " << traits.block_n
         << ", \"tile_k\": " << traits.block_k
         << ", \"threads\": " << traits.threads
         << ", \"dynamic_smem_bytes\": " << traits.dynamic_smem_bytes
         << "},\n"
         << "  \"causal_load_balanced\": "
         << (options.causal_load_balanced ? "true" : "false") << ",\n"
         << "  \"correctness\": \""
         << (!options.check
                 ? "not_run"
                 : (options.operator_kind == OperatorKind::kQkv &&
                            options.m * context.world == 1024
                        ? (options.weight_beta == 0
                               ? "exact_qkv_route_protocol_and_full_overwrite"
                               : "exact_qkv_route_protocol_outputs_and_main_grad_touched")
                        : (options.weight_beta == 0
                               ? "protocol_and_full_overwrite"
                               : "protocol_outputs_and_main_grad_touched")))
         << "\",\n"
         << "  \"timing\": \""
         << (options.launch_mode == LaunchMode::kGraph
                 ? "in-graph event nodes, per-sample max-rank"
                 : "per-sample max-rank CUDA event")
         << "\",\n";
  write_summary_fields(
      output, "data_phase", data, data_samples, true);
  write_summary_fields(
      output, "weight_phase", weight, weight_samples, true);
  write_summary_fields(
      output, "total", total, total_samples, false);
  output << "}\n";
}

void cleanup_buffers(
    const Options& options,
    const RankContext& context,
    Buffers& buffers) {
  CUDA_CHECK(cudaDeviceSynchronize());
  MPI_CHECK(MPI_Barrier(context.local_comm));
  for (int peer = 0; peer < context.world; ++peer) {
    if (peer == context.rank) {
      continue;
    }
    CUDA_CHECK(cudaIpcCloseMemHandle(buffers.peer_output[peer]));
    CUDA_CHECK(cudaIpcCloseMemHandle(buffers.peer_ready[peer]));
    CUDA_CHECK(cudaIpcCloseMemHandle(buffers.peer_epoch[peer]));
  }
  MPI_CHECK(MPI_Barrier(context.local_comm));
  CUDA_CHECK(cudaFree(buffers.owned_epoch));
  CUDA_CHECK(cudaFree(buffers.owned_ready));
  CUDA_CHECK(cudaFree(buffers.owned_output));
  if (buffers.local_intermediate != buffers.owned_output) {
    CUDA_CHECK(cudaFree(buffers.local_intermediate));
  }
  CUDA_CHECK(cudaFree(buffers.grad_weight));
  if (buffers.grad_input != buffers.local_intermediate) {
    CUDA_CHECK(cudaFree(buffers.grad_input));
  }
  CUDA_CHECK(cudaFree(buffers.saved_input));
  CUDA_CHECK(cudaFree(buffers.weight));
  if (options.operator_kind == OperatorKind::kQkv) {
    CUDA_CHECK(cudaFree(buffers.grad_v));
    CUDA_CHECK(cudaFree(buffers.grad_k));
    CUDA_CHECK(cudaFree(buffers.grad_q));
  } else {
    CUDA_CHECK(cudaFree(buffers.grad_output));
  }
}

void validate_shape(const Options& options, const RankContext& context) {
  if (options.m % options.batch != 0 || options.m % 16 != 0 ||
      options.hidden % 16 != 0 || options.head_dim % 16 != 0) {
    throw std::runtime_error(
        "M, hidden and head dimension must satisfy the BF16 tile alignment");
  }
  if (options.q_heads % context.world != 0) {
    throw std::runtime_error("Q heads must divide the CP world size");
  }
  if (options.operator_kind == OperatorKind::kQkv &&
      (options.kv_heads <= 0 ||
       options.q_heads % options.kv_heads != 0 ||
       options.kv_heads % context.world != 0)) {
    throw std::runtime_error(
        "QKV backward requires valid GQA heads divisible by CP");
  }
  if (options.causal_load_balanced &&
      (options.m / options.batch) % 2 != 0) {
    throw std::runtime_error(
        "causal load balancing requires an even local sequence length");
  }
}

int run(
    const Options& options,
    RankContext& context) {
  validate_shape(options, context);
  Runtime runtime = initialize_runtime();
  const int projection_width = options.operator_kind == OperatorKind::kQkv
      ? (options.q_heads + 2 * options.kv_heads) * options.head_dim
      : options.q_heads * options.head_dim;

  int64_t ready_elements = 0;
  if (options.operator_kind == OperatorKind::kQkv) {
    fuse::QkvBackwardDataParams shape{};
    shape.local_tokens = options.m;
    shape.hidden = options.hidden;
    shape.batch = options.batch;
    shape.q_heads = options.q_heads;
    shape.kv_heads = options.kv_heads;
    shape.head_dim = options.head_dim;
    shape.world_size = context.world;
    shape.num_comm_ctas = options.comm_ctas;
    shape.gemm_policy = options.gemm_policy;
    ready_elements = fuse::qkv_backward_ready_elements(shape);
  } else {
    fuse::OprojBackwardDataParams shape{};
    shape.local_tokens = options.m;
    shape.hidden = options.hidden;
    shape.batch = options.batch;
    shape.q_heads = options.q_heads;
    shape.head_dim = options.head_dim;
    shape.world_size = context.world;
    shape.num_comm_ctas = options.comm_ctas;
    shape.gemm_policy = options.gemm_policy;
    ready_elements = fuse::oproj_backward_ready_elements(shape);
  }
  if (ready_elements <= 0) {
    throw std::runtime_error("invalid ready-buffer shape");
  }

  Buffers buffers = allocate_and_exchange_buffers(
      options,
      context,
      runtime,
      projection_width,
      ready_elements);
  OperatorState state = make_operator_state(options, context, buffers);
  if (state.resolved_comm_ctas <= 0) {
    throw std::runtime_error("failed to resolve communication CTAs");
  }

  uint32_t epoch = 1;
  MPI_CHECK(MPI_Barrier(context.local_comm));
  if (options.weight_mode == WeightMode::kImmediate) {
    launch_immediate(options, state, epoch, runtime.stream);
  } else {
    launch_data(options, state, epoch, runtime.stream);
    launch_weight(options, state, runtime.stream);
  }
  CUDA_CHECK(cudaStreamSynchronize(runtime.stream));
  MPI_CHECK(MPI_Barrier(context.local_comm));

  std::vector<float> data_samples;
  std::vector<float> weight_samples;
  std::vector<float> total_samples;
  Summary data_stats{};
  Summary weight_stats{};
  Summary total_stats{};
  if (options.weight_mode == WeightMode::kImmediate) {
    const uint32_t first_epoch = epoch + 1;
    auto launch = [&](int step) {
      launch_immediate(
          options,
          state,
          first_epoch + static_cast<uint32_t>(step),
          runtime.stream);
    };
    total_samples = options.launch_mode == LaunchMode::kGraph
        ? time_graph(options, context, runtime, 2, launch)
        : time_eager(options, context, runtime, launch);
    epoch += static_cast<uint32_t>(options.warmup + options.iterations);
    total_stats = summarize(total_samples);
  } else {
    const uint32_t first_epoch = epoch + 1;
    auto launch_data_phase = [&](int step) {
      launch_data(
          options,
          state,
          first_epoch + static_cast<uint32_t>(step),
          runtime.stream);
    };
    data_samples = options.launch_mode == LaunchMode::kGraph
        ? time_graph(options, context, runtime, 1, launch_data_phase)
        : time_eager(options, context, runtime, launch_data_phase);
    epoch += static_cast<uint32_t>(options.warmup + options.iterations);
    data_stats = summarize(data_samples);
    auto launch_weight_phase = [&](int) {
      launch_weight(options, state, runtime.stream);
    };
    weight_samples = options.launch_mode == LaunchMode::kGraph
        ? time_graph(options, context, runtime, 1, launch_weight_phase)
        : time_eager(options, context, runtime, launch_weight_phase);
    weight_stats = summarize(weight_samples);
    total_samples.resize(options.iterations);
    for (int index = 0; index < options.iterations; ++index) {
      total_samples[index] = data_samples[index] + weight_samples[index];
    }
    total_stats = summarize(total_samples);
  }

#if FUSE_ENABLE_PROFILING
  if (options.role_profile) {
    run_qkv_role_profile(
        options, context, runtime, state, epoch);
  }
#endif

  if (options.check) {
    validate_qkv_route_exact(options, context, runtime, buffers);
    validate_protocol(options, context, runtime, buffers, epoch);
  }

  if (context.rank == 0) {
    const double phase_flops =
        2.0 * options.m * projection_width * options.hidden;
    std::cout << operator_name(options.operator_kind)
              << " MPI launch=" << launch_mode_name(options.launch_mode)
              << " weight_mode="
              << weight_mode_name(options.weight_mode)
              << " weight_beta=" << options.weight_beta
              << " world=" << context.world
              << " B_MNK=" << options.m << "x"
              << (options.operator_kind == OperatorKind::kQkv
                      ? options.hidden
                      : projection_width)
              << "x"
              << (options.operator_kind == OperatorKind::kQkv
                      ? projection_width
                      : options.hidden)
              << " W_MNK="
              << (options.operator_kind == OperatorKind::kQkv
                      ? projection_width
                      : options.hidden)
              << "x"
              << (options.operator_kind == OperatorKind::kQkv
                      ? options.hidden
                      : projection_width)
              << "x" << options.m
              << " comm_ctas=" << state.resolved_comm_ctas
              << " policy=" << gemm_policy_name(state.resolved_gemm_policy)
              << " tile=" << state.traits.block_m << "x"
              << state.traits.block_n << "\n";
    std::cout << std::fixed << std::setprecision(4);
    if (options.weight_mode == WeightMode::kDeferred) {
      std::cout << "B: p50=" << data_stats.p50 << " ms p95="
                << data_stats.p95 << " ms TFLOPS/GPU="
                << std::setprecision(1)
                << phase_flops / data_stats.p50 / 1.0e9 << "\n"
                << std::setprecision(4)
                << "W: p50=" << weight_stats.p50 << " ms p95="
                << weight_stats.p95 << " ms TFLOPS/GPU="
                << std::setprecision(1)
                << phase_flops / weight_stats.p50 / 1.0e9 << "\n";
    }
    std::cout << std::setprecision(4)
              << "B+W: p50=" << total_stats.p50 << " ms p95="
              << total_stats.p95 << " ms TFLOPS/GPU="
              << std::setprecision(1)
              << (2.0 * phase_flops) / total_stats.p50 / 1.0e9
              << "\n";
  }
  write_json(
      options,
      context,
      projection_width,
      state.resolved_comm_ctas,
      state.resolved_gemm_policy,
      state.traits,
      data_stats,
      data_samples,
      weight_stats,
      weight_samples,
      total_stats,
      total_samples);

  cleanup_buffers(options, context, buffers);
  CUDA_CHECK(cudaEventDestroy(runtime.stop));
  CUDA_CHECK(cudaEventDestroy(runtime.start));
  CUDA_CHECK(cudaStreamDestroy(runtime.stream));
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  int initialized = 0;
  int finalized = 0;
  int rank = -1;
  try {
    int provided = MPI_THREAD_SINGLE;
    MPI_CHECK(MPI_Init_thread(
        &argc, &argv, MPI_THREAD_FUNNELED, &provided));
    initialized = 1;
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    if (provided < MPI_THREAD_FUNNELED) {
      throw std::runtime_error("MPI does not provide MPI_THREAD_FUNNELED");
    }
    const Options options = parse_options(argc, argv);
    if (options.help) {
      if (rank == 0) {
        print_usage(argv[0]);
      }
      MPI_CHECK(MPI_Finalize());
      return 0;
    }
    RankContext context = initialize_rank_context();
    int result = 0;
    if (options.run_all_modes) {
      if (options.json_prefix.empty() || !options.json_out.empty()) {
        throw std::runtime_error(
            "--run-all-modes requires --json-prefix and no --json-out");
      }
      for (LaunchMode launch : {LaunchMode::kEager, LaunchMode::kGraph}) {
        for (WeightMode weight :
             {WeightMode::kDeferred, WeightMode::kImmediate}) {
          Options variant = options;
          variant.run_all_modes = false;
          variant.launch_mode = launch;
          variant.weight_mode = weight;
          variant.weight_beta =
              weight == WeightMode::kDeferred ? 1 : 0;
          variant.json_out = options.json_prefix + "_" +
              launch_mode_name(launch) + "_" + weight_mode_name(weight) +
              ".json";
          result = run(variant, context);
          if (result != 0) {
            break;
          }
        }
        if (result != 0) {
          break;
        }
      }
    } else {
      result = run(options, context);
    }
    MPI_CHECK(MPI_Comm_free(&context.local_comm));
    MPI_CHECK(MPI_Finalize());
    return result;
  } catch (const std::exception& error) {
    std::cerr << "backward_mpi_bench"
              << (rank >= 0 ? " rank=" + std::to_string(rank) : "")
              << ": " << error.what() << "\n";
    if (initialized) {
      MPI_Finalized(&finalized);
      if (!finalized) {
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
    }
    return 1;
  }
}
