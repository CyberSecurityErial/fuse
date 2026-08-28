#include "fuse/operators/gemm_a2a.h"

#include <cuda_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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

void check_cuda(cudaError_t status, const char* expression, const char* file, int line) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(file) + ":" + std::to_string(line) + " " + expression +
        ": " + cudaGetErrorString(status));
  }
}

void check_mpi(int status, const char* expression, const char* file, int line) {
  if (status != MPI_SUCCESS) {
    char message[MPI_MAX_ERROR_STRING]{};
    int length = 0;
    MPI_Error_string(status, message, &length);
    throw std::runtime_error(
        std::string(file) + ":" + std::to_string(line) + " " + expression +
        ": " + std::string(message, static_cast<size_t>(length)));
  }
}

enum class LaunchMode {
  kEager,
  kGraph,
};

struct Options {
  int m = 128;
  int k = 4096;
  int batch = 1;
  int q_heads = 16;
  int kv_heads = 8;
  int head_dim = 128;
  int comm_ctas = 16;
  int warmup = 10;
  int iterations = 50;
  fuse::GemmRaster raster = fuse::GemmRaster::kAlongM;
  int swizzle = 8;
  bool defer_v_a2a = false;
  bool check = true;
  bool help = false;
  LaunchMode launch_mode = LaunchMode::kEager;
  std::string json_out;
#if FUSE_ENABLE_PROFILING
  bool role_profile = false;
  std::string role_profile_json;
#endif
};

const char* launch_mode_name(LaunchMode mode) {
  return mode == LaunchMode::kGraph ? "graph" : "eager";
}

const char* raster_name(fuse::GemmRaster raster) {
  switch (raster) {
    case fuse::GemmRaster::kAlongM:
      return "m";
    case fuse::GemmRaster::kAlongN:
      return "n";
    default:
      return "auto";
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
    if (argument == "--m") {
      options.m = parse_positive(take("--m"), "--m");
    } else if (argument == "--k") {
      options.k = parse_positive(take("--k"), "--k");
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
    } else if (argument == "--warmup") {
      options.warmup = parse_nonnegative(take("--warmup"), "--warmup");
    } else if (argument == "--iterations") {
      options.iterations =
          parse_positive(take("--iterations"), "--iterations");
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
      options.swizzle = parse_positive(take("--swizzle"), "--swizzle");
      if (options.swizzle != 1 && options.swizzle != 2 &&
          options.swizzle != 4 && options.swizzle != 8) {
        throw std::runtime_error("--swizzle must be 1, 2, 4, or 8");
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
    } else if (argument == "--defer-v-a2a") {
      options.defer_v_a2a = true;
    } else if (argument == "--check") {
      options.check = true;
    } else if (argument == "--no-check") {
      options.check = false;
    } else if (argument == "--json-out") {
      options.json_out = take("--json-out");
#if FUSE_ENABLE_PROFILING
    } else if (argument == "--role-profile") {
      options.role_profile = true;
    } else if (argument == "--role-profile-json") {
      options.role_profile_json = take("--role-profile-json");
      options.role_profile = true;
#endif
    } else if (argument == "--help" || argument == "-h") {
      options.help = true;
    } else {
      throw std::runtime_error("unknown argument: " + argument);
    }
  }
  return options;
}

void print_usage(const char* program) {
  std::cout
      << "Usage: mpirun -np <2..8> " << program << " [options]\n"
      << "  --m N --k N --batch N\n"
      << "  --q-heads N --kv-heads N --head-dim N\n"
      << "  --comm-ctas N          0 selects the kernel heuristic\n"
      << "  --raster auto|m|n --swizzle 1|2|4|8\n"
      << "  --launch eager|graph   graph is one pre-uploaded replay containing\n"
      << "                         warmup+sample kernels with monotonic epochs\n"
      << "  --warmup N --iterations N   defaults: 10 and 50\n"
      << "  --defer-v-a2a --check|--no-check --json-out PATH\n"
#if FUSE_ENABLE_PROFILING
      << "  --role-profile         one synchronized diagnostic epoch per rank\n"
      << "  --role-profile-json PATH  also write the per-rank role summary\n"
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
    throw std::runtime_error("CUDA IPC benchmark requires all MPI ranks on one host");
  }
  if (context.world < 2 || context.world > fuse::kMaxWorldSize) {
    throw std::runtime_error("MPI world size must be between 2 and 8");
  }

  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count == 1) {
    // Supports launchers that give every rank a distinct one-GPU visibility set.
    context.device = 0;
  } else {
    if (device_count < context.world) {
      throw std::runtime_error(
          "each rank must see either one GPU or the same world-sized GPU set");
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
      const char* lhs_bus_id = all_bus_ids.data() +
          static_cast<size_t>(lhs) * local_bus_id.size();
      const char* rhs_bus_id = all_bus_ids.data() +
          static_cast<size_t>(rhs) * local_bus_id.size();
      if (std::equal(
              lhs_bus_id,
              lhs_bus_id + local_bus_id.size(),
              rhs_bus_id)) {
        throw std::runtime_error("two MPI ranks selected the same physical GPU");
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
  CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
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
    data[index] = Bf16(static_cast<float>(value) / 32.0f);
  }
}

__global__ void fill_weight_kernel(Bf16* data, int64_t elements, int seed) {
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int value = static_cast<int>((index * 17 + seed * 13) % 19) - 9;
    data[index] = Bf16(static_cast<float>(value) / 256.0f);
  }
}

__global__ void count_sentinel_kernel(
    const uint16_t* data,
    int64_t elements,
    unsigned long long* count) {
  unsigned long long local = 0;
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    local += data[index] == 0xffffu;
  }
  if (local != 0) {
    atomicAdd(count, local);
  }
}

template <class Kernel>
void launch_fill(Kernel kernel, Bf16* pointer, int64_t elements, int seed,
                 cudaStream_t stream) {
  const int blocks = static_cast<int>(
      std::min<int64_t>((elements + 255) / 256, 4096));
  kernel<<<blocks, 256, 0, stream>>>(pointer, elements, seed);
  CUDA_CHECK(cudaGetLastError());
}

struct IpcHandles {
  cudaIpcMemHandle_t output{};
  cudaIpcMemHandle_t epoch{};
};

static_assert(std::is_trivially_copyable_v<IpcHandles>);

struct Buffers {
  Bf16* lhs = nullptr;
  Bf16* rhs_nt = nullptr;
  Bf16* local_output = nullptr;
  Bf16* owned_output = nullptr;
  uint32_t* ready = nullptr;
  uint32_t* owned_epoch = nullptr;
  int64_t output_elements = 0;
  int64_t ready_elements = 0;
  std::vector<Bf16*> peer_output;
  std::vector<uint32_t*> peer_epoch;
};

Buffers allocate_and_exchange_buffers(
    const RankContext& context,
    const Runtime& runtime,
    int m,
    int n,
    int k,
    int64_t output_elements,
    int ready_elements) {
  Buffers buffers;
  buffers.output_elements = output_elements;
  buffers.ready_elements = ready_elements;
  buffers.lhs = allocate<Bf16>(static_cast<int64_t>(m) * k);
  buffers.rhs_nt = allocate<Bf16>(static_cast<int64_t>(n) * k);
  buffers.local_output = allocate<Bf16>(static_cast<int64_t>(m) * n);
  buffers.owned_output = allocate<Bf16>(output_elements);
  buffers.ready = allocate<uint32_t>(ready_elements);
  buffers.owned_epoch = allocate<uint32_t>(
      static_cast<int64_t>(context.world) * fuse::kReadyFlagStride);

  launch_fill(
      fill_kernel,
      buffers.lhs,
      static_cast<int64_t>(m) * k,
      context.rank + 601,
      runtime.stream);
  launch_fill(
      fill_weight_kernel,
      buffers.rhs_nt,
      static_cast<int64_t>(n) * k,
      context.rank + 701,
      runtime.stream);
  CUDA_CHECK(cudaMemsetAsync(
      buffers.local_output,
      0,
      static_cast<size_t>(m) * n * sizeof(Bf16),
      runtime.stream));
  CUDA_CHECK(cudaMemsetAsync(
      buffers.owned_output,
      0xff,
      static_cast<size_t>(output_elements) * sizeof(Bf16),
      runtime.stream));
  CUDA_CHECK(cudaMemsetAsync(
      buffers.ready,
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
  buffers.peer_epoch.resize(context.world);
  for (int peer = 0; peer < context.world; ++peer) {
    if (peer == context.rank) {
      buffers.peer_output[peer] = buffers.owned_output;
      buffers.peer_epoch[peer] = buffers.owned_epoch;
      continue;
    }
    CUDA_CHECK(cudaIpcOpenMemHandle(
        reinterpret_cast<void**>(&buffers.peer_output[peer]),
        handles[peer].output,
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

fuse::GemmA2AParams make_params(
    const Options& options,
    const RankContext& context,
    const Buffers& buffers,
    int n,
    int launch_comm_ctas) {
  fuse::GemmA2AParams params{};
  params.lhs = buffers.lhs;
  params.rhs_nt = buffers.rhs_nt;
  params.local_output = buffers.local_output;
  for (int peer = 0; peer < context.world; ++peer) {
    params.peer_output[peer] = buffers.peer_output[peer];
    params.peer_route_done_epoch[peer] = buffers.peer_epoch[peer];
  }
  params.ready = buffers.ready;
  params.gemm = {options.m, n, options.k, 1};
  params.gemm.raster = options.raster;
  params.gemm.max_swizzle_size = options.swizzle;
  params.route.world_size = context.world;
  params.route.rank = context.rank;
  params.route.batch = options.batch;
  params.route.seq_local = options.m / options.batch;
  params.route.global_seq = params.route.seq_local * context.world;
  params.route.q_heads = options.q_heads;
  params.route.kv_heads = options.kv_heads;
  params.route.head_dim = options.head_dim;
  params.route.qkv_peer_interleaved =
      options.m < 2048 && options.head_dim == 128 &&
      options.raster == fuse::GemmRaster::kAlongM;
  params.route.kind = fuse::RouteKind::kQkvGqaPack;
  params.route.direction = fuse::RouteDirection::kForward;
  params.route.defer_v_a2a = options.defer_v_a2a;
  params.num_comm_ctas = launch_comm_ctas;
  params.epoch = 1;
  return params;
}

void launch_fused(
    fuse::GemmA2AParams& params,
    uint32_t epoch,
    cudaStream_t stream) {
  params.epoch = epoch;
  CUDA_CHECK(fuse::launch_gemm_a2a_cutlass(params, stream));
}

void prime(
    const RankContext& context,
    const Runtime& runtime,
    fuse::GemmA2AParams& params,
    uint32_t& epoch) {
  ++epoch;
  MPI_CHECK(MPI_Barrier(context.local_comm));
  launch_fused(params, epoch, runtime.stream);
  CUDA_CHECK(cudaStreamSynchronize(runtime.stream));
  MPI_CHECK(MPI_Barrier(context.local_comm));
}

std::vector<float> time_eager(
    const Options& options,
    const RankContext& context,
    const Runtime& runtime,
    fuse::GemmA2AParams& params,
    uint32_t& epoch) {
  for (int step = 0; step < options.warmup; ++step) {
    ++epoch;
    MPI_CHECK(MPI_Barrier(context.local_comm));
    launch_fused(params, epoch, runtime.stream);
    CUDA_CHECK(cudaStreamSynchronize(runtime.stream));
  }

  std::vector<float> samples;
  samples.reserve(options.iterations);
  for (int step = 0; step < options.iterations; ++step) {
    ++epoch;
    MPI_CHECK(MPI_Barrier(context.local_comm));
    CUDA_CHECK(cudaEventRecord(runtime.start, runtime.stream));
    launch_fused(params, epoch, runtime.stream);
    CUDA_CHECK(cudaEventRecord(runtime.stop, runtime.stream));
    CUDA_CHECK(cudaEventSynchronize(runtime.stop));
    float local_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&local_ms, runtime.start, runtime.stop));
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

// One graph replay contains all warmup and measured epochs.  This leaves only
// one MPI-barrier-to-cudaGraphLaunch host skew; each fused kernel's finalizer
// keeps subsequent epochs aligned across ranks.  Event nodes around every
// measured kernel retain the usual per-sample max-rank timing definition.
std::vector<float> time_graph(
    const Options& options,
    const RankContext& context,
    const Runtime& runtime,
    fuse::GemmA2AParams& params,
    uint32_t& epoch) {
  std::vector<cudaEvent_t> starts(options.iterations, nullptr);
  std::vector<cudaEvent_t> stops(options.iterations, nullptr);
  for (int sample = 0; sample < options.iterations; ++sample) {
    CUDA_CHECK(cudaEventCreate(&starts[sample]));
    CUDA_CHECK(cudaEventCreate(&stops[sample]));
  }

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  const uint32_t first_epoch = epoch + 1;
  const int total = options.warmup + options.iterations;
  CUDA_CHECK(cudaStreamBeginCapture(
      runtime.stream, cudaStreamCaptureModeThreadLocal));
  for (int step = 0; step < total; ++step) {
    launch_fused(
        params,
        first_epoch + static_cast<uint32_t>(step),
        runtime.stream);
  }
  CUDA_CHECK(cudaStreamEndCapture(runtime.stream, &graph));

  size_t graph_node_count = 0;
  CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &graph_node_count));
  std::vector<cudaGraphNode_t> graph_nodes(graph_node_count);
  CUDA_CHECK(cudaGraphGetNodes(
      graph, graph_nodes.data(), &graph_node_count));
  if (graph_node_count != static_cast<size_t>(total)) {
    throw std::runtime_error(
        "captured graph node count mismatch: nodes=" +
        std::to_string(graph_node_count) + " expected kernels=" +
        std::to_string(total));
  }
  for (cudaGraphNode_t node : graph_nodes) {
    cudaGraphNodeType type{};
    CUDA_CHECK(cudaGraphNodeGetType(node, &type));
    if (type != cudaGraphNodeTypeKernel) {
      throw std::runtime_error("captured graph contains a non-kernel node");
    }
  }

  // Stream capture gives a linear kernel chain, but cudaGraphGetNodes does not
  // promise capture order. Recover that order from the dependency edges before
  // replacing the chain with kernel/event/kernel segments.
  size_t edge_count = 0;
  CUDA_CHECK(cudaGraphGetEdges(graph, nullptr, nullptr, &edge_count));
  std::vector<cudaGraphNode_t> edge_from(edge_count);
  std::vector<cudaGraphNode_t> edge_to(edge_count);
  CUDA_CHECK(cudaGraphGetEdges(
      graph, edge_from.data(), edge_to.data(), &edge_count));
  if (edge_count + 1 != graph_node_count) {
    throw std::runtime_error("captured kernels do not form one linear chain");
  }
  std::vector<int> indegree(graph_node_count, 0);
  std::vector<int> successor(graph_node_count, -1);
  auto node_index = [&](cudaGraphNode_t target) {
    const auto found = std::find(graph_nodes.begin(), graph_nodes.end(), target);
    if (found == graph_nodes.end()) {
      throw std::runtime_error("graph edge references an unknown node");
    }
    return static_cast<int>(found - graph_nodes.begin());
  };
  for (size_t edge = 0; edge < edge_count; ++edge) {
    const int from = node_index(edge_from[edge]);
    const int to = node_index(edge_to[edge]);
    if (successor[from] != -1) {
      throw std::runtime_error("captured graph kernel has multiple successors");
    }
    successor[from] = to;
    ++indegree[to];
  }
  int current = -1;
  for (size_t node = 0; node < graph_node_count; ++node) {
    if (indegree[node] == 0) {
      if (current != -1) {
        throw std::runtime_error("captured graph has multiple roots");
      }
      current = static_cast<int>(node);
    }
  }
  std::vector<cudaGraphNode_t> kernels;
  kernels.reserve(graph_node_count);
  while (current != -1) {
    kernels.push_back(graph_nodes[current]);
    current = successor[current];
  }
  if (kernels.size() != graph_node_count) {
    throw std::runtime_error("captured graph kernel chain is incomplete");
  }
  if (!edge_from.empty()) {
    CUDA_CHECK(cudaGraphRemoveDependencies(
        graph, edge_from.data(), edge_to.data(), edge_count));
  }

  cudaGraphNode_t previous = nullptr;
  for (int step = 0; step < total; ++step) {
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
    if (previous) {
      const cudaGraphNode_t kernel = kernels[step];
      CUDA_CHECK(cudaGraphAddDependencies(graph, &previous, &kernel, 1));
    }
    previous = kernels[step];
    if (sample >= 0) {
      cudaGraphNode_t stop_node = nullptr;
      CUDA_CHECK(cudaGraphAddEventRecordNode(
          &stop_node, graph, &previous, 1, stops[sample]));
      previous = stop_node;
    }
  }

  graph_node_count = 0;
  CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &graph_node_count));
  graph_nodes.resize(graph_node_count);
  CUDA_CHECK(cudaGraphGetNodes(
      graph, graph_nodes.data(), &graph_node_count));
  size_t kernel_nodes = 0;
  size_t event_record_nodes = 0;
  for (cudaGraphNode_t node : graph_nodes) {
    cudaGraphNodeType type{};
    CUDA_CHECK(cudaGraphNodeGetType(node, &type));
    kernel_nodes += type == cudaGraphNodeTypeKernel;
    event_record_nodes += type == cudaGraphNodeTypeEventRecord;
  }
  if (kernel_nodes != static_cast<size_t>(total) ||
      event_record_nodes != static_cast<size_t>(2 * options.iterations)) {
    throw std::runtime_error(
        "final graph structure mismatch: kernels=" +
        std::to_string(kernel_nodes) + " event_records=" +
        std::to_string(event_record_nodes));
  }
  CUDA_CHECK(cudaGraphInstantiate(
      &graph_exec, graph, nullptr, nullptr, 0));
  CUDA_CHECK(cudaGraphUpload(graph_exec, runtime.stream));
  CUDA_CHECK(cudaStreamSynchronize(runtime.stream));

  MPI_CHECK(MPI_Barrier(context.local_comm));
  CUDA_CHECK(cudaGraphLaunch(graph_exec, runtime.stream));
  // Synchronize the replay stream so an asynchronous graph/kernel failure is
  // reported at its source instead of surfacing as an unrecorded-event error.
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
  epoch += static_cast<uint32_t>(total);

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
  if (!timeline.empty()) {
    const auto& root = timeline.front();
    summary.grid_sync_us = timer_span_us(
        local_roles_done, root.grid_sync_done);
    summary.finalize_us = timer_span_us(root.grid_sync_done, last);
    uint64_t sources_done = 0;
    bool valid_finalize = root.grid_sync_done >= local_roles_done &&
        root.fence_done >= root.grid_sync_done &&
        root.publish_done >= root.fence_done;
    for (int source = 0; source < world; ++source) {
      valid_finalize = valid_finalize &&
          root.source_ready[source] >= root.publish_done;
      sources_done = std::max(sources_done, root.source_ready[source]);
    }
    valid_finalize = valid_finalize && last >= sources_done;
    if (!valid_finalize) {
      throw std::runtime_error(
          "role-profile finalize timestamps are not monotonic on rank " +
          std::to_string(rank));
    }
  }
  if (summary.route_ctas != comm_ctas || summary.compute_ctas == 0 ||
      summary.observed_ctas != summary.route_ctas + summary.compute_ctas) {
    throw std::runtime_error(
        "role-profile CTA accounting mismatch on rank " +
        std::to_string(rank));
  }
  return summary;
}

void write_role_profile_json(
    const Options& options,
    const RankContext& context,
    int n,
    int resolved_comm_ctas,
    const fuse::KernelTraits& traits,
    uint32_t epoch,
    const std::vector<RoleProfileSummary>& ranks) {
  if (context.rank != 0 || options.role_profile_json.empty()) {
    return;
  }
  std::ofstream output(options.role_profile_json);
  if (!output) {
    throw std::runtime_error(
        "cannot open role-profile JSON output: " +
        options.role_profile_json);
  }
  const char* gemm_policy = std::getenv("FUSE_QKV_GEMM_POLICY");
  const char* comm_policy = std::getenv("FUSE_QKV_COMM_POLICY");
  const bool known_gemm_policy = gemm_policy != nullptr &&
      (std::strcmp(gemm_policy, "legacy") == 0 ||
       std::strcmp(gemm_policy, "wave_time_model") == 0 ||
       std::strcmp(gemm_policy, "m128n64") == 0 ||
       std::strcmp(gemm_policy, "m128n128") == 0 ||
       std::strcmp(gemm_policy, "m128n160") == 0 ||
       std::strcmp(gemm_policy, "m128n192") == 0 ||
       std::strcmp(gemm_policy, "m128n256") == 0 ||
       std::strcmp(gemm_policy, "m128n320") == 0);
  const bool wave_model = gemm_policy == nullptr ||
      (known_gemm_policy &&
       std::strcmp(gemm_policy, "wave_time_model") == 0);
  const bool pipeline_model = comm_policy == nullptr ||
      std::strcmp(comm_policy, "pipeline") == 0 ||
      std::strcmp(comm_policy, "roofline") == 0;
  const bool known_comm_policy = comm_policy == nullptr || pipeline_model ||
      std::strcmp(comm_policy, "legacy") == 0;
  const char* policy_model = !wave_model
      ? (known_gemm_policy ? "manual_override" : "unrecognized")
      : (pipeline_model ? "calibrated_pipeline_independent_progress_v3"
                        : (gemm_policy != nullptr
                              ? "calibrated_wave_time_independent_progress_v3"
                              : "calibrated_wave_time_v1"));
  output << std::setprecision(10)
         << "{\n"
         << "  \"mode\": \"qkv_gemm_a2a_mpi_role_profile\",\n"
         << "  \"profiling_build\": true,\n"
         << "  \"diagnostic_launch\": \"mpi_synchronized_eager\",\n"
         << "  \"production_timing_launch\": \""
         << launch_mode_name(options.launch_mode) << "\",\n"
         << "  \"clock\": \"sm90_globaltimer\",\n"
         << "  \"epoch\": " << epoch << ",\n"
         << "  \"shape\": {\"m\": " << options.m << ", \"n\": " << n
         << ", \"k\": " << options.k << "},\n"
         << "  \"head_geometry\": {\"batch\": " << options.batch
         << ", \"q_heads\": " << options.q_heads
         << ", \"kv_heads\": " << options.kv_heads
         << ", \"head_dim\": " << options.head_dim << "},\n"
         << "  \"world_size\": " << context.world << ",\n"
         << "  \"requested_comm_ctas\": " << options.comm_ctas << ",\n"
         << "  \"comm_ctas\": " << resolved_comm_ctas << ",\n"
         << "  \"kernel_traits\": {\"tile_m\": " << traits.block_m
         << ", \"tile_n\": " << traits.block_n
         << ", \"tile_k\": " << traits.block_k
         << ", \"cluster_m\": " << (traits.block_n >= 256 ? 2 : 1)
         << ", \"threads\": " << traits.threads
         << ", \"dynamic_smem_bytes\": " << traits.dynamic_smem_bytes
         << "},\n"
         << "  \"qkv_policy_request\": \""
         << (gemm_policy == nullptr
                 ? "auto"
                 : (known_gemm_policy ? gemm_policy : "unrecognized"))
         << "\",\n"
         << "  \"qkv_comm_policy_request\": \""
         << (comm_policy == nullptr
                 ? "auto"
                 : (known_comm_policy ? comm_policy : "unrecognized"))
         << "\",\n"
         << "  \"qkv_policy_model\": \"" << policy_model << "\",\n"
         << "  \"launch_plan_cache\": \"per_process_v1\",\n"
         << "  \"raster\": \"" << raster_name(options.raster) << "\",\n"
         << "  \"swizzle\": " << options.swizzle << ",\n"
         << "  \"defer_v_a2a\": "
         << (options.defer_v_a2a ? "true" : "false") << ",\n"
         << "  \"telemetry_finalize_policy\": \"diagnostic_whole_grid\",\n"
         << "  \"ranks\": [\n";
  for (size_t index = 0; index < ranks.size(); ++index) {
    const auto& rank = ranks[index];
    output << "    {\"rank\": " << rank.rank
           << ", \"observed_ctas\": " << rank.observed_ctas
           << ", \"compute_ctas\": " << rank.compute_ctas
           << ", \"route_ctas\": " << rank.route_ctas
           << ", \"compute_role_us\": " << rank.compute_role_us
           << ", \"route_role_us\": " << rank.route_role_us
           << ", \"overlap_us\": " << rank.overlap_us
           << ", \"grid_sync_us\": " << rank.grid_sync_us
           << ", \"finalize_us\": " << rank.finalize_us
           << ", \"kernel_us\": " << rank.kernel_us << "}"
           << (index + 1 == ranks.size() ? "\n" : ",\n");
  }
  output << "  ]\n}\n";
}

void run_role_profile(
    const Options& options,
    const RankContext& context,
    const Runtime& runtime,
    const fuse::GemmA2AParams& params,
    int n,
    int resolved_comm_ctas,
    const fuse::KernelTraits& traits,
    uint32_t& epoch) {
  auto diagnostic_params = params;
  diagnostic_params.num_comm_ctas = resolved_comm_ctas;
  int timeline_capacity = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(
      &timeline_capacity,
      cudaDevAttrMultiProcessorCount,
      context.device));
  auto* device_timeline =
      allocate<fuse::A2AGemmCtaTimeline>(timeline_capacity);
  CUDA_CHECK(cudaMemsetAsync(
      device_timeline,
      0,
      static_cast<size_t>(timeline_capacity) *
          sizeof(fuse::A2AGemmCtaTimeline),
      runtime.stream));
  CUDA_CHECK(cudaStreamSynchronize(runtime.stream));

  // The diagnostic wrapper is a separate kernel instantiation. Warm it once
  // on every rank so lazy module setup cannot masquerade as source waiting.
  ++epoch;
  diagnostic_params.epoch = epoch;
  MPI_CHECK(MPI_Barrier(context.local_comm));
  CUDA_CHECK(fuse::launch_gemm_a2a_role_telemetry(
      diagnostic_params,
      device_timeline,
      timeline_capacity,
      runtime.stream));
  CUDA_CHECK(cudaStreamSynchronize(runtime.stream));
  MPI_CHECK(MPI_Barrier(context.local_comm));

  CUDA_CHECK(cudaMemsetAsync(
      device_timeline,
      0,
      static_cast<size_t>(timeline_capacity) *
          sizeof(fuse::A2AGemmCtaTimeline),
      runtime.stream));
  CUDA_CHECK(cudaStreamSynchronize(runtime.stream));

  // Preserve the production monotonic-epoch protocol. Every MPI process
  // submits exactly one measured diagnostic kernel after the same barrier.
  ++epoch;
  diagnostic_params.epoch = epoch;
  MPI_CHECK(MPI_Barrier(context.local_comm));
  CUDA_CHECK(fuse::launch_gemm_a2a_role_telemetry(
      diagnostic_params,
      device_timeline,
      timeline_capacity,
      runtime.stream));
  CUDA_CHECK(cudaStreamSynchronize(runtime.stream));

  std::vector<fuse::A2AGemmCtaTimeline> timeline(timeline_capacity);
  CUDA_CHECK(cudaMemcpy(
      timeline.data(),
      device_timeline,
      timeline.size() * sizeof(timeline[0]),
      cudaMemcpyDeviceToHost));
  const RoleProfileSummary local = summarize_role_profile(
      context.rank, resolved_comm_ctas, context.world, timeline);
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
        << "\nQKV GEMM->A2A MPI role profile epoch=" << epoch
        << " (%globaltimer, one measured diagnostic kernel per rank; "
           "whole-grid finalize)\n"
        << "rank  compute_ctas  route_ctas  compute_role_us  route_role_us  "
           "overlap_us  grid_sync_us  finalize_us  kernel_us\n";
    for (const auto& rank : ranks) {
      std::cout << std::setw(4) << rank.rank << "  "
                << std::setw(12) << rank.compute_ctas << "  "
                << std::setw(10) << rank.route_ctas << "  "
                << std::setw(15) << std::fixed << std::setprecision(2)
                << rank.compute_role_us << "  "
                << std::setw(13) << rank.route_role_us << "  "
                << std::setw(10) << rank.overlap_us << "  "
                << std::setw(12) << rank.grid_sync_us << "  "
                << std::setw(11) << rank.finalize_us << "  "
                << std::setw(9) << rank.kernel_us << "\n";
    }
  }
  write_role_profile_json(
      options,
      context,
      n,
      resolved_comm_ctas,
      traits,
      epoch,
      ranks);
  CUDA_CHECK(cudaFree(device_timeline));
}
#endif

void validate_result(
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
      buffers.ready,
      ready.size() * sizeof(uint32_t),
      cudaMemcpyDeviceToHost,
      runtime.stream));
  CUDA_CHECK(cudaMemcpyAsync(
      epochs.data(),
      buffers.owned_epoch,
      epochs.size() * sizeof(uint32_t),
      cudaMemcpyDeviceToHost,
      runtime.stream));

  auto* sentinel_count = allocate<unsigned long long>(1);
  CUDA_CHECK(cudaMemsetAsync(
      sentinel_count, 0, sizeof(unsigned long long), runtime.stream));
  const int blocks = static_cast<int>(std::min<int64_t>(
      (buffers.output_elements + 255) / 256, 4096));
  count_sentinel_kernel<<<blocks, 256, 0, runtime.stream>>>(
      reinterpret_cast<const uint16_t*>(buffers.owned_output),
      buffers.output_elements,
      sentinel_count);
  CUDA_CHECK(cudaGetLastError());
  unsigned long long local_sentinel = 0;
  CUDA_CHECK(cudaMemcpyAsync(
      &local_sentinel,
      sentinel_count,
      sizeof(local_sentinel),
      cudaMemcpyDeviceToHost,
      runtime.stream));
  CUDA_CHECK(cudaStreamSynchronize(runtime.stream));
  CUDA_CHECK(cudaFree(sentinel_count));

  unsigned long long local_ready_errors = 0;
  for (size_t signal = 0;
       signal < ready.size();
       signal += fuse::kReadyFlagStride) {
    local_ready_errors += ready[signal] != final_epoch;
  }
  unsigned long long local_epoch_errors = 0;
  for (int source = 0; source < context.world; ++source) {
    local_epoch_errors +=
        epochs[static_cast<size_t>(source) * fuse::kReadyFlagStride] !=
        final_epoch;
  }
  unsigned long long local_errors[3] = {
      local_ready_errors, local_epoch_errors, local_sentinel};
  unsigned long long global_errors[3]{};
  MPI_CHECK(MPI_Allreduce(
      local_errors,
      global_errors,
      3,
      MPI_UNSIGNED_LONG_LONG,
      MPI_SUM,
      context.local_comm));
  if (global_errors[0] != 0 || global_errors[1] != 0 ||
      global_errors[2] != 0) {
    throw std::runtime_error(
        "lightweight IPC check failed: ready_errors=" +
        std::to_string(global_errors[0]) + " epoch_errors=" +
        std::to_string(global_errors[1]) + " untouched_output_elements=" +
        std::to_string(global_errors[2]));
  }
  if (context.rank == 0) {
    std::cout << "IPC check: all source epochs reached " << final_epoch
              << ", routed output fully overwritten\n";
  }
}

void validate_numerics(
    const RankContext& context,
    const Runtime& runtime,
    Buffers& buffers,
    const fuse::GemmA2AParams& params,
    int resolved_comm_ctas) {
  auto reference_params = params;
  reference_params.num_comm_ctas = resolved_comm_ctas;
  std::vector<Bf16> actual(static_cast<size_t>(buffers.output_elements));
  std::vector<Bf16> reference(static_cast<size_t>(buffers.output_elements));
  CUDA_CHECK(cudaMemcpyAsync(
      actual.data(),
      buffers.owned_output,
      actual.size() * sizeof(Bf16),
      cudaMemcpyDeviceToHost,
      runtime.stream));
  CUDA_CHECK(cudaStreamSynchronize(runtime.stream));

  // Every rank independently recomputes its local projection, then routes it
  // through the same imported peer pointers. The post-sync barrier guarantees
  // all remote reference stores have completed before destinations download.
  MPI_CHECK(MPI_Barrier(context.local_comm));
  CUDA_CHECK(fuse::launch_batched_cutlass_reference(
      reference_params, runtime.stream));
  CUDA_CHECK(fuse::launch_gemm_a2a_copy_reference(
      reference_params, runtime.stream));
  CUDA_CHECK(cudaStreamSynchronize(runtime.stream));
  MPI_CHECK(MPI_Barrier(context.local_comm));

  CUDA_CHECK(cudaMemcpyAsync(
      reference.data(),
      buffers.owned_output,
      reference.size() * sizeof(Bf16),
      cudaMemcpyDeviceToHost,
      runtime.stream));
  CUDA_CHECK(cudaStreamSynchronize(runtime.stream));

  constexpr float kTolerance = 0.25f;
  unsigned long long local_mismatches = 0;
  float local_max_abs = 0.0f;
  for (size_t index = 0; index < actual.size(); ++index) {
    const float difference = std::abs(
        static_cast<float>(actual[index]) -
        static_cast<float>(reference[index]));
    if (!std::isfinite(difference)) {
      local_max_abs = std::numeric_limits<float>::infinity();
      ++local_mismatches;
    } else {
      local_max_abs = std::max(local_max_abs, difference);
      local_mismatches += difference > kTolerance;
    }
  }

  unsigned long long global_mismatches = 0;
  float global_max_abs = 0.0f;
  MPI_CHECK(MPI_Allreduce(
      &local_mismatches,
      &global_mismatches,
      1,
      MPI_UNSIGNED_LONG_LONG,
      MPI_SUM,
      context.local_comm));
  MPI_CHECK(MPI_Allreduce(
      &local_max_abs,
      &global_max_abs,
      1,
      MPI_FLOAT,
      MPI_MAX,
      context.local_comm));
  if (global_mismatches != 0 || global_max_abs > kTolerance) {
    throw std::runtime_error(
        "fused vs separated reference failed: mismatches=" +
        std::to_string(global_mismatches) + " max_abs=" +
        std::to_string(global_max_abs));
  }
  if (context.rank == 0) {
    std::cout << "Numerical check: fused vs separated max_abs="
              << global_max_abs << " mismatches=" << global_mismatches
              << "\n";
  }
}

void write_json(
    const Options& options,
    const RankContext& context,
    int n,
    int resolved_comm_ctas,
    const fuse::KernelTraits& traits,
    const Summary& stats,
    const std::vector<float>& samples) {
  if (context.rank != 0 || options.json_out.empty()) {
    return;
  }
  std::ofstream output(options.json_out);
  if (!output) {
    throw std::runtime_error("cannot open JSON output: " + options.json_out);
  }
  const char* policy_override = std::getenv("FUSE_QKV_GEMM_POLICY");
  const char* comm_policy_override = std::getenv("FUSE_QKV_COMM_POLICY");
  const bool known_policy = policy_override != nullptr &&
      (std::strcmp(policy_override, "legacy") == 0 ||
       std::strcmp(policy_override, "wave_time_model") == 0 ||
       std::strcmp(policy_override, "m128n64") == 0 ||
       std::strcmp(policy_override, "m128n128") == 0 ||
       std::strcmp(policy_override, "m128n160") == 0 ||
       std::strcmp(policy_override, "m128n192") == 0 ||
       std::strcmp(policy_override, "m128n256") == 0 ||
       std::strcmp(policy_override, "m128n320") == 0);
  const bool pipeline_model = comm_policy_override == nullptr ||
      std::strcmp(comm_policy_override, "pipeline") == 0 ||
      std::strcmp(comm_policy_override, "roofline") == 0;
  const bool calibrated_tile_model = policy_override == nullptr ||
      (known_policy && std::strcmp(policy_override, "wave_time_model") == 0);
  const char* policy_model = !calibrated_tile_model
      ? (known_policy ? "manual_override" : "unrecognized")
      : (pipeline_model ? "calibrated_pipeline_independent_progress_v3"
                        : (policy_override != nullptr
                                  ? "calibrated_wave_time_independent_progress_v3"
                                  : "calibrated_wave_time_v1"));
  const bool known_comm_policy = comm_policy_override == nullptr ||
      std::strcmp(comm_policy_override, "pipeline") == 0 ||
      std::strcmp(comm_policy_override, "roofline") == 0 ||
      std::strcmp(comm_policy_override, "legacy") == 0;
  output << std::setprecision(10)
         << "{\n"
         << "  \"mode\": \"qkv_gemm_a2a_mpi\",\n"
#if FUSE_ENABLE_PROFILING
         << "  \"profiling_build\": true,\n"
         << "  \"role_profile_requested\": "
         << (options.role_profile ? "true" : "false") << ",\n"
#else
         << "  \"profiling_build\": false,\n"
         << "  \"role_profile_requested\": false,\n"
#endif
         << "  \"launch\": \"" << launch_mode_name(options.launch_mode)
         << "\",\n"
         << "  \"graph_epoch_mode\": \""
         << (options.launch_mode == LaunchMode::kGraph
                 ? "single_replay_monotonic_epoch_nodes"
                 : "eager_monotonic_epochs")
         << "\",\n"
         << "  \"shape\": {\"m\": " << options.m << ", \"n\": " << n
         << ", \"k\": " << options.k << "},\n"
         << "  \"head_geometry\": {\"batch\": " << options.batch
         << ", \"q_heads\": " << options.q_heads
         << ", \"kv_heads\": " << options.kv_heads
         << ", \"head_dim\": " << options.head_dim << "},\n"
         << "  \"world_size\": " << context.world << ",\n"
         << "  \"warmup\": " << options.warmup << ",\n"
         << "  \"iterations\": " << options.iterations << ",\n"
         << "  \"requested_comm_ctas\": " << options.comm_ctas << ",\n"
         << "  \"comm_ctas\": " << resolved_comm_ctas << ",\n"
         << "  \"qkv_policy_request\": \""
         << (policy_override == nullptr
                 ? "auto"
                 : (known_policy ? policy_override : "unrecognized"))
         << "\",\n"
         << "  \"qkv_comm_policy_request\": \""
         << (comm_policy_override == nullptr
                 ? "auto"
                 : (known_comm_policy ? comm_policy_override : "unrecognized"))
         << "\",\n"
         << "  \"qkv_policy_model\": \""
         << policy_model
         << "\",\n"
         << "  \"launch_plan_cache\": \"per_process_v1\",\n"
         << "  \"kernel_traits\": {\"tile_m\": " << traits.block_m
         << ", \"tile_n\": " << traits.block_n
         << ", \"tile_k\": " << traits.block_k
         << ", \"cluster_m\": " << (traits.block_n >= 256 ? 2 : 1)
         << ", \"threads\": " << traits.threads
         << ", \"dynamic_smem_bytes\": " << traits.dynamic_smem_bytes
         << "},\n"
         << "  \"raster\": \"" << raster_name(options.raster) << "\",\n"
         << "  \"swizzle\": " << options.swizzle << ",\n"
         << "  \"defer_v_a2a\": "
         << (options.defer_v_a2a ? "true" : "false") << ",\n"
         << "  \"correctness\": \""
         << (options.check ? "fused_vs_separated_reference" : "not_run")
         << "\",\n"
         << "  \"timing\": \"per-sample max-rank CUDA event\",\n"
         << "  \"mean_ms\": " << stats.mean << ",\n"
         << "  \"p50_ms\": " << stats.p50 << ",\n"
         << "  \"p95_ms\": " << stats.p95 << ",\n"
         << "  \"min_ms\": " << stats.minimum << ",\n"
         << "  \"max_ms\": " << stats.maximum << ",\n"
         << "  \"samples_ms\": [";
  for (size_t index = 0; index < samples.size(); ++index) {
    output << (index == 0 ? "" : ", ") << samples[index];
  }
  output << "]\n}\n";
}

void cleanup_buffers(const RankContext& context, Buffers& buffers) {
  CUDA_CHECK(cudaDeviceSynchronize());
  MPI_CHECK(MPI_Barrier(context.local_comm));
  for (int peer = 0; peer < context.world; ++peer) {
    if (peer == context.rank) {
      continue;
    }
    CUDA_CHECK(cudaIpcCloseMemHandle(buffers.peer_output[peer]));
    CUDA_CHECK(cudaIpcCloseMemHandle(buffers.peer_epoch[peer]));
  }
  MPI_CHECK(MPI_Barrier(context.local_comm));
  CUDA_CHECK(cudaFree(buffers.owned_epoch));
  CUDA_CHECK(cudaFree(buffers.ready));
  CUDA_CHECK(cudaFree(buffers.owned_output));
  CUDA_CHECK(cudaFree(buffers.local_output));
  CUDA_CHECK(cudaFree(buffers.rhs_nt));
  CUDA_CHECK(cudaFree(buffers.lhs));
}

void validate_shape(
    const Options& options,
    const RankContext& context,
    int n,
    const fuse::KernelTraits& traits) {
  if (options.m % options.batch != 0) {
    throw std::runtime_error("--m must be divisible by --batch");
  }
  if (options.q_heads % options.kv_heads != 0 ||
      options.q_heads % context.world != 0 ||
      options.kv_heads % context.world != 0) {
    throw std::runtime_error(
        "Q heads and KV heads must define valid GQA and divide world size");
  }
  if (options.m % traits.block_m != 0 ||
      options.k % traits.block_k != 0) {
    throw std::runtime_error(
        "fast benchmark requires M/K divisible by the selected kernel tiles");
  }
}

int run(const Options& options, RankContext& context) {
#if FUSE_ENABLE_PROFILING
  if (options.role_profile && !options.json_out.empty() &&
      options.json_out == options.role_profile_json) {
    throw std::runtime_error(
        "--json-out and --role-profile-json must name different files");
  }
#endif
  const int n =
      (options.q_heads + 2 * options.kv_heads) * options.head_dim;
  const fuse::GemmProblem problem{options.m, n, options.k, 1};
  Runtime runtime = initialize_runtime();
  const int seq_local = options.m / options.batch;
  const int global_seq = seq_local * context.world;
  fuse::UlyssesRoute route{};
  route.world_size = context.world;
  route.rank = context.rank;
  route.batch = options.batch;
  route.seq_local = seq_local;
  route.global_seq = global_seq;
  route.q_heads = options.q_heads;
  route.kv_heads = options.kv_heads;
  route.head_dim = options.head_dim;
  route.kind = fuse::RouteKind::kQkvGqaPack;
  route.direction = fuse::RouteDirection::kForward;
  route.defer_v_a2a = options.defer_v_a2a;
  route.qkv_peer_interleaved =
      options.m < 2048 && options.head_dim == 128 &&
      options.raster == fuse::GemmRaster::kAlongM;
  const int resolved_comm_ctas = options.comm_ctas == 0
      ? fuse::recommended_gemm_a2a_comm_ctas(problem, route)
      : options.comm_ctas;
  int sm_count = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(
      &sm_count, cudaDevAttrMultiProcessorCount, context.device));
  if (resolved_comm_ctas <= 0 || resolved_comm_ctas >= sm_count) {
    throw std::runtime_error("comm CTAs must leave at least one compute CTA");
  }
  const auto traits = fuse::qkv_cutlass_kernel_traits(
      problem, route, resolved_comm_ctas, sm_count);
  validate_shape(options, context, n, traits);

  const int local_width =
      (options.q_heads / context.world +
       (options.defer_v_a2a ? 1 : 2) * options.kv_heads / context.world) *
      options.head_dim;
  const int64_t output_elements64 =
      static_cast<int64_t>(options.batch) * global_seq * local_width;
  const int64_t ready_elements64 =
      static_cast<int64_t>((options.m + traits.block_m - 1) / traits.block_m) *
      ((n + traits.block_n - 1) / traits.block_n) * fuse::kReadyFlagStride;
  if (static_cast<uint64_t>(output_elements64) >
          std::numeric_limits<size_t>::max() / sizeof(Bf16) ||
      ready_elements64 > std::numeric_limits<int>::max()) {
    throw std::runtime_error("shape exceeds the focused benchmark's index range");
  }

  Buffers buffers = allocate_and_exchange_buffers(
      context,
      runtime,
      options.m,
      n,
      options.k,
      output_elements64,
      static_cast<int>(ready_elements64));
  // Preserve an auto request in the production launch path. The priming
  // launch populates the internal launch-plan cache; Eager warmup/timing and
  // Graph capture then exercise its cache-hit path.
  auto params = make_params(
      options, context, buffers, n, options.comm_ctas);

  uint32_t epoch = 0;
  prime(context, runtime, params, epoch);
  // The priming launch validates lazy module/IPC setup but must not satisfy the
  // post-run output check on behalf of the timed eager/graph path.
  CUDA_CHECK(cudaMemsetAsync(
      buffers.owned_output,
      0xff,
      static_cast<size_t>(buffers.output_elements) * sizeof(Bf16),
      runtime.stream));
  CUDA_CHECK(cudaStreamSynchronize(runtime.stream));
  MPI_CHECK(MPI_Barrier(context.local_comm));
  const auto samples = options.launch_mode == LaunchMode::kGraph
      ? time_graph(options, context, runtime, params, epoch)
      : time_eager(options, context, runtime, params, epoch);
  const Summary stats = summarize(samples);

  if (options.check) {
    validate_result(context, runtime, buffers, epoch);
    validate_numerics(
        context, runtime, buffers, params, resolved_comm_ctas);
  }
  if (context.rank == 0) {
    const double flops = 2.0 * options.m * n * options.k;
    const double tflops = flops / stats.p50 / 1.0e9;
    std::cout << "QKV GEMM->A2A MPI/IPC launch="
              << launch_mode_name(options.launch_mode)
              << " world=" << context.world << " shape=" << options.m << "x"
              << n << "x" << options.k << " comm_ctas="
              << resolved_comm_ctas << " raster=" << raster_name(options.raster)
              << " swizzle=" << options.swizzle << "\n"
              << "timing=" << options.warmup << " warmup + "
              << options.iterations
              << " samples, per-sample max-rank CUDA event\n"
              << std::fixed << std::setprecision(4)
              << "mean=" << stats.mean << " ms p50=" << stats.p50
              << " ms p95=" << stats.p95 << " ms min=" << stats.minimum
              << " ms max=" << stats.maximum << " ms TFLOPS/GPU(p50)="
              << std::setprecision(1) << tflops << " aggregate="
              << tflops * context.world << "\n";
  }
  write_json(
      options, context, n, resolved_comm_ctas, traits, stats, samples);

#if FUSE_ENABLE_PROFILING
  // Keep the diagnostic kernel outside the formal timing, correctness, and
  // result JSON. This prevents profile builds from silently becoming Golden
  // benchmark inputs while still preserving the monotonic epoch protocol.
  if (options.role_profile) {
    run_role_profile(
        options,
        context,
        runtime,
        params,
        n,
        resolved_comm_ctas,
        traits,
        epoch);
  }
#endif

  cleanup_buffers(context, buffers);
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
    const int result = run(options, context);
    MPI_CHECK(MPI_Comm_free(&context.local_comm));
    MPI_CHECK(MPI_Finalize());
    return result;
  } catch (const std::exception& error) {
    std::cerr << "qkvproj_a2a_mpi_bench"
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
