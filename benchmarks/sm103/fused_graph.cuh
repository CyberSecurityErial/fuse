// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Benchmark-only Graph preparation. No production entry point caches or
// exposes its private kernel Params. The caller supplies the actual launch.
namespace fused_graph {

class Operation {
 public:
  // The caller owns this device/stream and every captured tensor/IPC mapping.
  // Complete the stream before reset/destruction, then release IPC resources.
  // All methods, including destruction, must have one serialized owner.
  Operation(int device, cudaStream_t stream, uint32_t committed_epoch = 0,
            bool separate_postprocess = false, bool cooperative_postprocess = false)
      : device_(device), stream_(stream), separate_postprocess_(separate_postprocess),
        cooperative_postprocess_(cooperative_postprocess),
        committed_epoch_(committed_epoch) {
    if (device < 0 || stream == nullptr || (cooperative_postprocess && !separate_postprocess)) {
      throw std::invalid_argument("Graph operation requires a device and explicit stream");
    }
  }

  Operation(const Operation&) = delete;
  Operation& operator=(const Operation&) = delete;
  Operation(Operation&&) = delete;
  Operation& operator=(Operation&&) = delete;

  // Deliberately no implicit synchronization: a failed peer must not strand
  // another process in destructor-driven CUDA waits. Use reset() on the
  // normal path to check completion and report destruction errors.
  ~Operation() noexcept { destroy_unchecked(); }

  uint32_t committed_epoch() const { return committed_epoch_; }

  // Call outside the sample events and MPI barrier. Capture does not execute
  // the operation or commit its epoch. callable(epoch, stream) must return a
  // cudaError_t and pass that epoch to the real public launch without changing
  // the candidate, pointers, input-publication generation or scheduler policy.
  // Recreate/reset this object when any of those bindings change.
  template <class Callable>
  void prepare(uint32_t epoch, Callable&& callable) {
    check_owner();
    require_idle();
    if (poisoned_) throw std::runtime_error("Graph operation is poisoned; reset after completion");
    if (prepared_) throw std::runtime_error("Graph operation already has an unlaunched epoch");
    if (committed_epoch_ == std::numeric_limits<uint32_t>::max() ||
        epoch == 0 || epoch != committed_epoch_ + 1) {
      throw std::runtime_error("Graph epoch must advance by exactly one actual launch");
    }

    cudaGraph_t captured = nullptr;
    cudaGraphExec_t initial_exec = nullptr;
    bool capturing = false;
    try {
      check(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal), "begin capture");
      capturing = true;
      check(std::forward<Callable>(callable)(epoch, stream_), "captured public launch");
      const cudaError_t end_status = cudaStreamEndCapture(stream_, &captured);
      capturing = false;
      check(end_status, "end capture");
      const auto signature = inspect(captured);
      if (!exec_) {
        check(cudaGraphInstantiateWithFlags(&initial_exec, captured, 0), "instantiate");
        // Initial upload is preparation, not a sample or an executed epoch.
        // The caller completes preparation before entering its timed cadence.
        check(cudaGraphUpload(initial_exec, stream_), "initial upload");
        graph_ = captured;
        exec_ = initial_exec;
        signature_ = signature;
        captured = nullptr;
        initial_exec = nullptr;
      } else {
        if (!same_signature(signature_, signature)) {
          throw std::runtime_error("Graph kernel function, geometry or launch attributes changed");
        }
        cudaGraphExecUpdateResultInfo result{};
        const cudaError_t status = cudaGraphExecUpdate(exec_, captured, &result);
        if (status != cudaSuccess || result.result != cudaGraphExecUpdateSuccess) {
          throw std::runtime_error(std::string("Graph update failed: ") + cudaGetErrorString(status) +
              ", result=" + std::to_string(static_cast<int>(result.result)));
        }
        check(cudaGraphDestroy(captured), "destroy update graph");
        captured = nullptr;
      }
      prepared_epoch_ = epoch;
      prepared_ = true;
    } catch (...) {
      // Even an invalidated capture must be ended on its originating thread.
      if (capturing) (void)cudaStreamEndCapture(stream_, &captured);
      if (initial_exec) (void)cudaGraphExecDestroy(initial_exec);
      if (captured) (void)cudaGraphDestroy(captured);
      poisoned_ = true;
      throw;
    }
  }

  // Exactly one operation belongs between the caller's start/end events.
  // Commit only an accepted CUDA enqueue; asynchronous failures remain fatal
  // when the caller synchronizes its end event, never retries of this epoch.
  void launch() {
    check_owner();
    if (poisoned_ || !prepared_ || !exec_) {
      throw std::runtime_error("Graph launch requires one successfully prepared epoch");
    }
    const cudaError_t status = cudaGraphLaunch(exec_, stream_);
    if (status != cudaSuccess) {
      poisoned_ = true;
      check(status, "launch");
    }
    committed_epoch_ = prepared_epoch_;
    prepared_ = false;
  }

  // A completed candidate/component/generation may bind a new epoch baseline.
  // This never clears production or calibration flags; their owner decides
  // whether this baseline is the continuing count or a separately reset count.
  void reset(uint32_t completed_epoch = 0) {
    check_owner();
    require_idle();
    if (exec_) {
      check(cudaGraphExecDestroy(exec_), "destroy executable");
      exec_ = nullptr;
    }
    if (graph_) {
      check(cudaGraphDestroy(graph_), "destroy original graph");
      graph_ = nullptr;
    }
    committed_epoch_ = completed_epoch;
    prepared_epoch_ = 0;
    prepared_ = false;
    poisoned_ = false;
    signature_ = {};
  }

 private:
  struct Signature {
    void* function = nullptr;
    dim3 grid{};
    dim3 block{};
    unsigned int shared_mem = 0;
    int cooperative = 0;
    unsigned int cluster_x = 0, cluster_y = 0, cluster_z = 0;
    int cluster_scheduling = 0;
  };

  static void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
      throw std::runtime_error(std::string("Graph ") + operation + ": " + cudaGetErrorString(status));
    }
  }

  void check_owner() const {
    int current = -1;
    check(cudaGetDevice(&current), "get device");
    if (current != device_) throw std::runtime_error("Graph operation used on a different CUDA device");
  }

  void require_idle() const {
    check(cudaStreamQuery(stream_), "requires completed owner stream");
  }

  static bool same_dim(dim3 a, dim3 b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
  }

  static bool same_signature(const Signature& a, const Signature& b) {
    return a.function == b.function && same_dim(a.grid, b.grid) && same_dim(a.block, b.block) &&
        a.shared_mem == b.shared_mem && a.cooperative == b.cooperative &&
        a.cluster_x == b.cluster_x && a.cluster_y == b.cluster_y && a.cluster_z == b.cluster_z &&
        a.cluster_scheduling == b.cluster_scheduling;
  }

  static bool same_signature(const std::vector<Signature>& a, const std::vector<Signature>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) if (!same_signature(a[i],b[i])) return false;
    return true;
  }

  std::vector<Signature> inspect(cudaGraph_t graph) const {
    size_t count = 0;
    check(cudaGraphGetNodes(graph, nullptr, &count), "count nodes");
    const size_t expected = separate_postprocess_ ? 2 : 1;
    if (count != expected) throw std::runtime_error("Graph operation kernel count differs from boundary");
    std::vector<cudaGraphNode_t> nodes(count);
    check(cudaGraphGetNodes(graph,nodes.data(),&count),"get kernel nodes");
    if (count != expected) throw std::runtime_error("Graph nodes changed during inspection");
    if (separate_postprocess_) {
      // Explicit reference only: exactly base-F -> standalone norm, not two
      // unordered nodes or hidden memcpy/memset work. Both belong to ONE graph.
      size_t edges = 0;
      check(cudaGraphGetEdges(graph,nullptr,nullptr,nullptr,&edges),"count reference edges");
      if (edges != 1) throw std::runtime_error("Separate postprocess requires one dependency edge");
      cudaGraphNode_t from = nullptr, to = nullptr;
      cudaGraphEdgeData dependency{};
      check(cudaGraphGetEdges(graph,&from,&to,&dependency,&edges),"get reference edge");
      if (edges != 1 || from == to ||
          dependency.type != 0 || dependency.from_port != 0 || dependency.to_port != 0 ||
          !((from == nodes[0] && to == nodes[1]) || (from == nodes[1] && to == nodes[0])))
        throw std::runtime_error("Invalid separate postprocess dependency");
      nodes = {from,to};
    }
    std::vector<Signature> result;
    for (size_t i = 0; i < count; ++i) result.push_back(inspect_node(nodes[i],i == 1));
    return result;
  }

  Signature inspect_node(cudaGraphNode_t node, bool postprocess) const {
    if (node == nullptr) throw std::runtime_error("Null Graph kernel node");
    cudaGraphNodeType type{};
    check(cudaGraphNodeGetType(node, &type), "get node type");
    if (type != cudaGraphNodeTypeKernel) throw std::runtime_error("Graph operation contains a non-kernel node");
    cudaKernelNodeParams params{};
    check(cudaGraphKernelNodeGetParams(node, &params), "get kernel parameters");
    // kernelParams/extra and their pointed-to values are node-owned. Never
    // write them or guess the layout of the private CUTLASS argument object.
    cudaKernelNodeAttrValue cooperative{}, cluster{}, scheduling{};
    check(cudaGraphKernelNodeGetAttribute(node, cudaKernelNodeAttributeCooperative, &cooperative),
          "get cooperative attribute");
    check(cudaGraphKernelNodeGetAttribute(node, cudaKernelNodeAttributeClusterDimension, &cluster),
          "get cluster attribute");
    check(cudaGraphKernelNodeGetAttribute(node, cudaKernelNodeAttributeClusterSchedulingPolicyPreference,
                                        &scheduling), "get cluster scheduling attribute");
    const bool implicit_cluster = cluster.clusterDim.x == 0 && cluster.clusterDim.y == 0 && cluster.clusterDim.z == 0;
    const bool single_cluster = cluster.clusterDim.x == 1 && cluster.clusterDim.y == 1 && cluster.clusterDim.z == 1;
    if (!params.func || params.gridDim.x == 0 || params.gridDim.y != 1 || params.gridDim.z != 1 ||
        params.blockDim.x != 256 || params.blockDim.y != 1 || params.blockDim.z != 1 ||
        (postprocess && cooperative_postprocess_ ? params.sharedMemBytes != 0 : params.sharedMemBytes == 0) ||
        cooperative.cooperative != (postprocess ? int(cooperative_postprocess_) : 1) ||
        (!implicit_cluster && !single_cluster)) {
      throw std::runtime_error("Graph requires the declared SM103 base/postprocess launch");
    }
    return {params.func, params.gridDim, params.blockDim, params.sharedMemBytes, cooperative.cooperative,
        cluster.clusterDim.x, cluster.clusterDim.y, cluster.clusterDim.z,
        static_cast<int>(scheduling.clusterSchedulingPolicyPreference)};
  }

  void destroy_unchecked() noexcept {
    if (exec_) (void)cudaGraphExecDestroy(exec_);
    if (graph_) (void)cudaGraphDestroy(graph_);
  }

  const int device_;
  const cudaStream_t stream_;
  const bool separate_postprocess_;
  const bool cooperative_postprocess_;
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t exec_ = nullptr;
  std::vector<Signature> signature_{};
  uint32_t committed_epoch_ = 0;
  uint32_t prepared_epoch_ = 0;
  bool prepared_ = false;
  bool poisoned_ = false;
};

}  // namespace fused_graph
