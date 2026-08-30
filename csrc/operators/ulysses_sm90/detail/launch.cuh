// SPDX-License-Identifier: BSD-3-Clause
// Private implementation; assembled by csrc/operators/ulysses_sm90.cu.
//
// Module index:
//   - device lookup, regular/cooperative launch, and reference launch helpers
//   - WordArrayHash and BoundedLaunchCache shared by all automatic plans
//   - policy-to-kernel bindings shared by launch, profiling, traits, and refs
//   - BF16/FP8 forward, backward-data, wgrad, and weighted dispatch helpers

namespace fuse {
namespace {

// Shared host launch helpers. Keeping both dataflow directions in one TU
// avoids duplicate registration of the CUTLASS reference kernels they share.
cudaError_t device_sm_count(int32_t* count, int32_t* device) {
  cudaError_t status = cudaGetDevice(device);
  if (status != cudaSuccess) {
    return status;
  }
  return cudaDeviceGetAttribute(count, cudaDevAttrMultiProcessorCount, *device);
}

template <class Key>
struct WordArrayHash {
  size_t operator()(const Key& key) const {
    size_t result = 0xcbf29ce484222325ull;
    for (uint64_t word : key.words) {
      result ^= std::hash<uint64_t>{}(word) + 0x9e3779b97f4a7c15ull +
          (result << 6) + (result >> 2);
    }
    return result;
  }
};

// Every automatic launch path needs the same two-level cache: a one-entry
// thread-local fast path followed by a small process-wide map. Keeping the
// implementation here prevents forward/backward policy code from growing its
// own subtly different locking, eviction and cold-error behavior.
template <class Key, class Value, class Hash, size_t MaxEntries = 256>
class BoundedLaunchCache {
 public:
  bool find(const Key& key, Value* result) {
    if (result == nullptr) {
      return false;
    }
    Recent& recent = last();
    if (recent.valid && recent.owner == this && recent.key == key) {
      *result = recent.value;
      return true;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = entries_.find(key);
    if (found == entries_.end()) {
      return false;
    }
    remember(key, found->second);
    *result = found->second;
    return true;
  }

  Value store(const Key& key, const Value& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = entries_.find(key);
    if (found == entries_.end()) {
      if (entries_.size() >= MaxEntries) {
        entries_.erase(entries_.begin());
      }
      found = entries_.emplace(key, value).first;
    }
    remember(key, found->second);
    return found->second;
  }

  template <class Factory>
  cudaError_t get_or_create(
      const Key& key, Value* result, Factory& factory) {
    if (result == nullptr) {
      return cudaErrorInvalidValue;
    }
    Recent& recent = last();
    if (recent.valid && recent.owner == this && recent.key == key) {
      *result = recent.value;
      return cudaSuccess;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = entries_.find(key);
    if (found != entries_.end()) {
      remember(key, found->second);
      *result = found->second;
      return cudaSuccess;
    }

    Value value{};
    cudaError_t status = factory(&value);
    if (status != cudaSuccess) {
      return status;
    }
    if (entries_.size() >= MaxEntries) {
      entries_.erase(entries_.begin());
    }
    entries_.emplace(key, value);
    remember(key, value);
    *result = value;
    return cudaSuccess;
  }

 private:
  struct Recent {
    const BoundedLaunchCache* owner = nullptr;
    bool valid = false;
    Key key{};
    Value value{};
  };

  static Recent& last() {
    thread_local Recent recent;
    return recent;
  }

  void remember(const Key& key, const Value& value) {
    last() = {this, true, key, value};
  }

  std::mutex mutex_;
  std::unordered_map<Key, Value, Hash> entries_;
};

template <class Kernel>
cudaError_t launch_regular(const typename Kernel::Params& params, cudaStream_t stream) {
  constexpr size_t smem_bytes = sizeof(typename Kernel::SharedStorage);
  constexpr int cluster_x = cute::size<0>(typename Kernel::ClusterShape{});
  constexpr int cluster_y = cute::size<1>(typename Kernel::ClusterShape{});
  constexpr int cluster_z = cute::size<2>(typename Kernel::ClusterShape{});
  constexpr int cluster_size = cluster_x * cluster_y * cluster_z;
  auto entry = cutlass::device_kernel<Kernel>;
  cudaError_t status = cudaFuncSetAttribute(
      entry, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes));
  if (status != cudaSuccess) {
    return status;
  }
  const dim3 grid = Kernel::get_grid_shape(params);
  if constexpr (cluster_size == 1) {
    entry<<<grid, Kernel::get_block_shape(), smem_bytes, stream>>>(params);
    return cudaGetLastError();
  } else {
    if (grid.x % cluster_x != 0 || grid.y % cluster_y != 0 ||
        grid.z % cluster_z != 0) {
      return cudaErrorInvalidConfiguration;
    }
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim = {cluster_x, cluster_y, cluster_z};
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = Kernel::get_block_shape();
    config.dynamicSmemBytes = smem_bytes;
    config.stream = stream;
    config.attrs = &attribute;
    config.numAttrs = 1;
    return cudaLaunchKernelEx(&config, entry, params);
  }
}

template <class Kernel>
cudaError_t launch_gemm_reference_impl(
    const GemmA2AParams& params,
    cudaStream_t stream,
    int32_t sm_count,
    int32_t device,
    int32_t reserved_comm_ctas,
    RasterOptions fallback) {
  typename Kernel::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape =
      make_shape(params.gemm.m, params.gemm.n, params.gemm.k, params.gemm.l);
  args.mainloop.ptr_A = params.lhs;
  args.mainloop.dA = make_stride(
      a_row_stride(params.gemm), _1{}, a_batch_stride(params.gemm));
  args.mainloop.ptr_B = params.rhs_nt;
  args.mainloop.dB = make_stride(
      b_row_stride(params.gemm), _1{}, b_batch_stride(params.gemm));
  args.epilogue.thread.alpha = params.alpha;
  args.epilogue.thread.beta = 0.0f;
  args.epilogue.ptr_C = nullptr;
  args.epilogue.dC = make_stride(
      d_row_stride(params.gemm), _1{}, d_batch_stride(params.gemm));
  args.epilogue.ptr_D = params.local_output;
  args.epilogue.dD = make_stride(
      d_row_stride(params.gemm), _1{}, d_batch_stride(params.gemm));
  args.hw_info.device_id = device;
  args.hw_info.sm_count = sm_count - reserved_comm_ctas;
  args.scheduler.max_swizzle_size = params.gemm.max_swizzle_size;
  args.scheduler.raster_order = raster_option(params.gemm.raster, fallback);
  if (!Kernel::can_implement(args) || Kernel::get_workspace_size(args) != 0) {
    return cudaErrorNotSupported;
  }
  if (Kernel::initialize_workspace(args, nullptr, stream) !=
      cutlass::Status::kSuccess) {
    return cudaErrorInitializationError;
  }
  return launch_regular<Kernel>(
      Kernel::to_underlying_arguments(args, nullptr), stream);
}

template <class Kernel, class ParamsType>
cudaError_t launch_a2a_lhs_reference_impl(
    const ParamsType& params,
    cudaStream_t stream,
    int32_t sm_count,
    int32_t device,
    int32_t reserved_comm_ctas) {
  typename Kernel::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape =
      make_shape(params.gemm.m, params.gemm.n, params.gemm.k, params.gemm.l);
  args.mainloop.ptr_A = params.input_staging;
  args.mainloop.dA = make_stride(
      a_row_stride(params.gemm), _1{}, a_batch_stride(params.gemm));
  args.mainloop.ptr_B = params.rhs_nt;
  args.mainloop.dB = make_stride(
      b_row_stride(params.gemm), _1{}, b_batch_stride(params.gemm));
  args.epilogue.thread.alpha = params.alpha;
  args.epilogue.thread.beta = 0.0f;
  args.epilogue.ptr_C = nullptr;
  args.epilogue.dC = make_stride(
      d_row_stride(params.gemm), _1{}, d_batch_stride(params.gemm));
  args.epilogue.ptr_D = params.output;
  args.epilogue.dD = make_stride(
      d_row_stride(params.gemm), _1{}, d_batch_stride(params.gemm));
  args.hw_info.device_id = device;
  args.hw_info.sm_count = sm_count - reserved_comm_ctas;
  args.scheduler.max_swizzle_size = params.gemm.max_swizzle_size;
  args.scheduler.raster_order =
      raster_option(params.gemm.raster, RasterOptions::AlongN);
  if (!Kernel::can_implement(args) || Kernel::get_workspace_size(args) != 0) {
    return cudaErrorNotSupported;
  }
  if (Kernel::initialize_workspace(args, nullptr, stream) !=
      cutlass::Status::kSuccess) {
    return cudaErrorInitializationError;
  }
  return launch_regular<Kernel>(
      Kernel::to_underlying_arguments(args, nullptr), stream);
}

template <
    class InputGemm,
    class Kernel,
    class Comm = A2ALhsInputComm
#if FUSE_ENABLE_PROFILING
    ,
    bool Instrumented = false
#endif
    ,
    class ParamsType = A2AGemmParams>
cudaError_t launch_a2a_lhs_gemm_policy(
    const ParamsType& params,
    cudaStream_t stream,
    int32_t sm_count,
    int32_t device
#if FUSE_ENABLE_PROFILING
    , A2AGemmCtaTimeline* timeline = nullptr,
    int32_t timeline_capacity = 0,
    A2AGemmPeerTimeline* peer_timeline = nullptr,
    int32_t peer_timeline_capacity = 0
#endif
    ) {
  typename Comm::Arguments comm_args{};
  comm_args.params = params;
  cudaError_t status = Comm::initialize(comm_args);
  if (status != cudaSuccess) {
    return status;
  }
  if (!Comm::can_implement(comm_args)) {
    return cudaErrorNotSupported;
  }

  constexpr int32_t tile_m = static_cast<int32_t>(
      cute::size<0>(typename InputGemm::TileShape{}));
  constexpr int32_t tile_n = static_cast<int32_t>(
      cute::size<1>(typename InputGemm::TileShape{}));
  constexpr int32_t tile_k = static_cast<int32_t>(
      cute::size<2>(typename InputGemm::TileShape{}));
  const int32_t m_tiles = ceil_div(params.gemm.m, tile_m);
  const int32_t n_tiles = ceil_div(params.gemm.n, tile_n);
  static_assert(Comm::kReadyBlockM % tile_m == 0);
  constexpr int32_t m_tiles_per_ready = Comm::kReadyBlockM / tile_m;
  const int32_t ready_m_tiles = ceil_div(params.gemm.m, Comm::kReadyBlockM);
  const int32_t compute_m_frontier =
      ceil_div(sm_count - params.num_comm_ctas, n_tiles);
  comm_args.m_window = min(
      ready_m_tiles,
      max(1, ceil_div(compute_m_frontier, m_tiles_per_ready)));
  const int32_t k_per_peer = params.gemm.k / params.route.world_size;
  if (params.gemm.k % params.route.world_size != 0 ||
      k_per_peer % tile_k != 0) {
    return cudaErrorNotSupported;
  }

  typename Kernel::Arguments args{};
  args.num_comm_ctas = params.num_comm_ctas;
#if FUSE_ENABLE_PROFILING
  if constexpr (Instrumented) {
    args.timeline = timeline;
    args.timeline_capacity = timeline_capacity;
  }
#endif
  args.comm = comm_args;
#if FUSE_ENABLE_PROFILING
  if constexpr (Instrumented) {
    args.comm.peer_timeline = peer_timeline;
    args.comm.peer_timeline_capacity = peer_timeline_capacity;
  }
#endif
  args.gemm.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.gemm.problem_shape =
      make_shape(params.gemm.m, params.gemm.n, params.gemm.k, params.gemm.l);
  args.gemm.mainloop.ptr_A = params.input_staging;
  args.gemm.mainloop.dA = make_stride(
      a_row_stride(params.gemm), _1{}, a_batch_stride(params.gemm));
  args.gemm.mainloop.ptr_B = params.rhs_nt;
  args.gemm.mainloop.dB = make_stride(
      b_row_stride(params.gemm), _1{}, b_batch_stride(params.gemm));
  args.gemm.mainloop.ready = params.ready;
  args.gemm.mainloop.world_size = params.route.world_size;
  args.gemm.mainloop.m_tiles = m_tiles;
  args.gemm.mainloop.arrivals_per_peer =
      Comm::arrivals_per_peer(comm_args);
  args.gemm.mainloop.k_tiles_per_peer = k_per_peer / tile_k;
  args.gemm.mainloop.epoch = params.epoch;
#if FUSE_ENABLE_PROFILING
  if constexpr (Instrumented) {
    args.gemm.mainloop.timeline = timeline;
    args.gemm.mainloop.timeline_capacity = timeline_capacity;
    args.gemm.mainloop.peer_timeline = peer_timeline;
    args.gemm.mainloop.peer_timeline_capacity = peer_timeline_capacity;
    args.gemm.mainloop.n_tiles = n_tiles;
  }
#endif
  args.gemm.epilogue.thread.alpha = params.alpha;
  args.gemm.epilogue.thread.beta = 0.0f;
  args.gemm.epilogue.ptr_C = nullptr;
  args.gemm.epilogue.dC = make_stride(
      d_row_stride(params.gemm), _1{}, d_batch_stride(params.gemm));
  args.gemm.epilogue.ptr_D = params.output;
  args.gemm.epilogue.dD = make_stride(
      d_row_stride(params.gemm), _1{}, d_batch_stride(params.gemm));
  args.gemm.hw_info.device_id = device;
  args.gemm.hw_info.sm_count = sm_count - params.num_comm_ctas;
  args.gemm.scheduler.max_swizzle_size = params.gemm.max_swizzle_size;
  args.gemm.scheduler.block_offset = params.num_comm_ctas;
  args.gemm.scheduler.raster_order =
      raster_option(params.gemm.raster, RasterOptions::AlongN);
  if (!Kernel::can_implement(args)) {
    return cudaErrorNotSupported;
  }
  if (Kernel::get_workspace_size(args) != 0) {
    return cudaErrorNotSupported;
  }
  return detail::launch_cooperative<Kernel>(
      Kernel::to_underlying_arguments(args, nullptr),
      stream,
      sm_count);
}

// Runtime policy model for A2A -> GEMM; the candidate kernels stay finite and
// precompiled above.
struct LhsPolicyCandidate {
  A2ALhsGemmPolicy policy;
  int32_t tile_m;
  int32_t tile_n;
  int32_t cluster_m;
};

constexpr std::array<LhsPolicyCandidate, 5> kLhsPolicyCandidates{{
    {A2ALhsGemmPolicy::kM64N128, 64, 128, 1},
    {A2ALhsGemmPolicy::kM128N128, 128, 128, 1},
    {A2ALhsGemmPolicy::kM128N160, 128, 160, 1},
    {A2ALhsGemmPolicy::kM128N256ClusterM2, 128, 256, 2},
    {A2ALhsGemmPolicy::kM128N320ClusterM2, 128, 320, 2},
}};

constexpr LhsPolicyCandidate kWideN320ManualCandidate{
    A2ALhsGemmPolicy::kM128N320ClusterM2, 128, 320, 2};

A2ALhsPolicyInfo score_a2a_lhs_policy(
    const GemmProblem& problem,
    int32_t num_comm_ctas,
    int32_t sm_count,
    const LhsPolicyCandidate& candidate) {
  A2ALhsPolicyInfo info{};
  info.policy = candidate.policy;
  info.tile_m = candidate.tile_m;
  info.tile_n = candidate.tile_n;
  info.tile_k = 64;
  info.cluster_m = candidate.cluster_m;
  info.compute_ctas = sm_count - num_comm_ctas;
  if (problem.m <= 0 || problem.n <= 0 || problem.k <= 0 || problem.l <= 0 ||
      info.compute_ctas <= 0 ||
      (candidate.cluster_m > 1 &&
       (num_comm_ctas % candidate.cluster_m != 0 ||
        info.compute_ctas % candidate.cluster_m != 0))) {
    info.estimated_cycles = std::numeric_limits<double>::infinity();
    return info;
  }
  info.compute_clusters = info.compute_ctas / candidate.cluster_m;

  const int64_t m_tiles = ceil_div(problem.m, candidate.tile_m);
  const int64_t n_tiles = ceil_div(problem.n, candidate.tile_n);
  const int64_t m_cluster_tiles = ceil_div(m_tiles, candidate.cluster_m);
  info.n_tiles = static_cast<int32_t>(n_tiles);
  info.tile_count = m_tiles * n_tiles * problem.l;
  info.cluster_tile_count = m_cluster_tiles * n_tiles * problem.l;
  info.waves = static_cast<int32_t>(
      ceil_div(
          info.cluster_tile_count,
          static_cast<int64_t>(info.compute_clusters)));
  info.last_wave_clusters = static_cast<int32_t>(
      info.cluster_tile_count - static_cast<int64_t>(info.waves - 1) *
          info.compute_clusters);
  info.last_wave_ctas = info.last_wave_clusters * candidate.cluster_m;
  // A cluster is the indivisible scheduler work unit.  A wave boundary is
  // frontier-aligned only when it contains an integer number of complete
  // N-frontiers, or one frontier occupies an integer number of whole waves.
  // This prevents an already-published M frontier from being split across
  // two persistent-CTA waves merely because its N fan-out does not divide the
  // resident cluster budget.
  info.frontier_aligned =
      info.waves == 1 || info.compute_clusters % n_tiles == 0 ||
      n_tiles % info.compute_clusters == 0;
  info.full_last_wave =
      info.cluster_tile_count % info.compute_clusters == 0;

  // SM90 wave model.  Tensor-core work and HBM bytes are
  // invariant across candidates; this compares the variable L1/L2 traffic
  // after accounting for cluster-B multicast and partial-wave occupancy.
  constexpr double kL2BytesPerCycle = 8.0e6 / 1.3e3;
  const double l2_bandwidth = std::min(
      64.0 * info.compute_ctas, kL2BytesPerCycle);
  const double l1_bandwidth = 128.0 * info.compute_ctas;
  constexpr double element_bytes = sizeof(Element);
  const double l2_bytes_per_tile =
      static_cast<double>(problem.k) *
          (candidate.tile_m +
           static_cast<double>(candidate.tile_n) / candidate.cluster_m) *
          element_bytes +
      static_cast<double>(candidate.tile_m) * candidate.tile_n *
          element_bytes;
  const double l1_bytes_per_tile =
      static_cast<double>(problem.k) *
          (candidate.tile_m + candidate.tile_n) * element_bytes +
      static_cast<double>(problem.k) *
          (std::max(64, candidate.tile_m) + candidate.tile_n) *
          element_bytes +
      2.0 * candidate.tile_m * candidate.tile_n * element_bytes;
  const double wave_efficiency =
      static_cast<double>(info.cluster_tile_count) /
      (static_cast<double>(info.waves) * info.compute_clusters);
  const double l2_cycles =
      l2_bytes_per_tile * info.tile_count / l2_bandwidth;
  const double l1_cycles =
      l1_bytes_per_tile * info.tile_count / l1_bandwidth;
  info.estimated_cycles =
      std::max(l1_cycles, l2_cycles) / wave_efficiency;
  return info;
}

A2ALhsPolicyInfo select_a2a_lhs_policy_impl(
    const GemmProblem& problem,
    int32_t num_comm_ctas,
    int32_t sm_count,
    A2ALhsGemmPolicy requested) {
  A2ALhsPolicyInfo best{};
  best.estimated_cycles = std::numeric_limits<double>::infinity();
  if (requested == A2ALhsGemmPolicy::kM128N320ClusterM2) {
    return score_a2a_lhs_policy(
        problem, num_comm_ctas, sm_count, kWideN320ManualCandidate);
  }
  for (const auto& candidate : kLhsPolicyCandidates) {
    if (requested != A2ALhsGemmPolicy::kAuto &&
        requested != candidate.policy) {
      continue;
    }
    const auto current = score_a2a_lhs_policy(
        problem, num_comm_ctas, sm_count, candidate);
    if (requested != A2ALhsGemmPolicy::kAuto) {
      return current;
    }
    // A two-CTA multicast cluster makes a one-wave input consumer advance at
    // the slower ready tile in each pair. Independent CTAs are stronger when
    // the whole GEMM already fits in one wave.
    if (candidate.cluster_m > 1 && current.waves == 1) {
      continue;
    }
    // N320 is admitted automatically only when it fixes a scheduling
    // geometry problem: aligning an M frontier or improving a partial wave.
    // If an unaligned candidate already ends on a full wave, keep the mature
    // N256 family instead of treating larger N as a generic GEMM tuning knob.
    if (candidate.policy == A2ALhsGemmPolicy::kM128N320ClusterM2 &&
        !current.frontier_aligned && current.full_last_wave) {
      continue;
    }
    const bool better_frontier =
        current.frontier_aligned > best.frontier_aligned;
    const bool equal_frontier =
        current.frontier_aligned == best.frontier_aligned;
    // Frontier splitting directly delays ready-data consumption and is a
    // structural scheduling hazard.  A partial final wave is softer: its cost
    // is already represented by wave_efficiency, so never choose an otherwise
    // weak GEMM tile solely to make the last wave full.
    if (better_frontier ||
        (equal_frontier &&
         current.estimated_cycles < best.estimated_cycles)) {
      best = current;
    }
  }
  return best;
}

template <class T>
struct TypeTag {
  using type = T;
};

// The semantic axes are part of the binding, not encoded in alias names.  A
// visitor can therefore share one registration across production, profiling,
// traits and reference paths without reconstructing what the kernel means.
template <
    class ProjectionType,
    class PassType,
    class GemmType,
    class CommType,
    class PureGemmType = void,
    class TelemetryGemmType = GemmType,
    class TelemetryCommType = CommType>
struct DataflowKernelBinding {
  using Projection = ProjectionType;
  using Pass = PassType;
  using Gemm = GemmType;
  using Comm = CommType;
  using Kernel = detail::MonolithicGemm<Gemm, Comm>;
  using PureGemm = PureGemmType;
#if FUSE_ENABLE_PROFILING
  using TelemetryGemm = TelemetryGemmType;
  using TelemetryComm = TelemetryCommType;
  using TelemetryBase = detail::MonolithicGemm<TelemetryGemm, TelemetryComm>;
  using TelemetryKernel = detail::RoleTelemetryKernel<TelemetryBase>;
#endif
};

#if FUSE_ENABLE_PROFILING
using OprojForwardM64Binding = DataflowKernelBinding<
    ulysses::OutputProjection, ulysses::Forward,
    A2ALhsM64Gemm, A2ALhsM64InputComm, M64PureGemm,
    A2ALhsM64TelemetryGemm, A2ALhsM64TelemetryInputComm>;
using OprojForwardN128Binding = DataflowKernelBinding<
    ulysses::OutputProjection, ulysses::Forward,
    A2ALhsInputGemm, A2ALhsInputComm, ::fuse::PureGemm,
    A2ALhsTelemetryGemm, A2ALhsTelemetryInputComm>;
using OprojForwardN160Binding = DataflowKernelBinding<
    ulysses::OutputProjection, ulysses::Forward,
    A2ALhsN160Gemm, A2ALhsInputComm, N160PureGemm,
    A2ALhsN160TelemetryGemm, A2ALhsTelemetryInputComm>;
using OprojForwardN256Binding = DataflowKernelBinding<
    ulysses::OutputProjection, ulysses::Forward,
    A2ALhsProjectionGemm, A2ALhsInputComm, ProjectionPureGemm,
    A2ALhsProjectionTelemetryGemm, A2ALhsTelemetryInputComm>;
using OprojForwardN320Binding = DataflowKernelBinding<
    ulysses::OutputProjection, ulysses::Forward,
    A2ALhsWideN320Gemm, A2ALhsInputComm, WideN320PureGemm,
    A2ALhsWideN320TelemetryGemm, A2ALhsTelemetryInputComm>;
#else
using OprojForwardM64Binding = DataflowKernelBinding<
    ulysses::OutputProjection, ulysses::Forward,
    A2ALhsM64Gemm, A2ALhsM64InputComm, M64PureGemm>;
using OprojForwardN128Binding = DataflowKernelBinding<
    ulysses::OutputProjection, ulysses::Forward,
    A2ALhsInputGemm, A2ALhsInputComm, ::fuse::PureGemm>;
using OprojForwardN160Binding = DataflowKernelBinding<
    ulysses::OutputProjection, ulysses::Forward,
    A2ALhsN160Gemm, A2ALhsInputComm, N160PureGemm>;
using OprojForwardN256Binding = DataflowKernelBinding<
    ulysses::OutputProjection, ulysses::Forward,
    A2ALhsProjectionGemm, A2ALhsInputComm, ProjectionPureGemm>;
using OprojForwardN320Binding = DataflowKernelBinding<
    ulysses::OutputProjection, ulysses::Forward,
    A2ALhsWideN320Gemm, A2ALhsInputComm, WideN320PureGemm>;
#endif

template <class Visitor>
cudaError_t visit_oproj_forward_policy(
    A2ALhsGemmPolicy policy, Visitor& visitor) {
  switch (policy) {
    case A2ALhsGemmPolicy::kM64N128:
      return visitor(TypeTag<OprojForwardM64Binding>{});
    case A2ALhsGemmPolicy::kM128N128:
      return visitor(TypeTag<OprojForwardN128Binding>{});
    case A2ALhsGemmPolicy::kM128N160:
      return visitor(TypeTag<OprojForwardN160Binding>{});
    case A2ALhsGemmPolicy::kM128N256ClusterM2:
      return visitor(TypeTag<OprojForwardN256Binding>{});
    case A2ALhsGemmPolicy::kM128N320ClusterM2:
      return visitor(TypeTag<OprojForwardN320Binding>{});
    default:
      return cudaErrorNotSupported;
  }
}

cudaError_t launch_a2a_lhs_gemm_impl(
    const A2AGemmParams& params,
    cudaStream_t stream,
    int32_t sm_count,
    int32_t device) {
  const auto selected = select_a2a_lhs_policy_impl(
      params.gemm,
      params.num_comm_ctas,
      sm_count,
      params.lhs_policy);
  auto launch = [&](auto binding_tag) {
    using Binding = typename decltype(binding_tag)::type;
    return launch_a2a_lhs_gemm_policy<
        typename Binding::Gemm,
        typename Binding::Kernel,
        typename Binding::Comm>(params, stream, sm_count, device);
  };
  return visit_oproj_forward_policy(selected.policy, launch);
}

template <
    class GemmKernel,
    class Comm,
    class Params
#if FUSE_ENABLE_PROFILING
    , bool Instrumented = false
#endif
    >
cudaError_t launch_gemm_a2a_impl(
    const Params& params,
    cudaStream_t stream,
    int32_t sm_count,
    int32_t device
#if FUSE_ENABLE_PROFILING
    , A2AGemmCtaTimeline* timeline = nullptr,
    int32_t timeline_capacity = 0
#endif
    ) {
#if FUSE_ENABLE_PROFILING
  using Kernel = std::conditional_t<
      Instrumented,
      GemmA2ARoleTelemetryKernel<GemmKernel, Comm>,
      detail::MonolithicGemm<GemmKernel, Comm>>;
#else
  using Kernel = detail::MonolithicGemm<GemmKernel, Comm>;
#endif
  constexpr int32_t tile_m =
      static_cast<int32_t>(cute::size<0>(typename GemmKernel::TileShape{}));
  constexpr int32_t tile_n =
      static_cast<int32_t>(cute::size<1>(typename GemmKernel::TileShape{}));
  const int32_t m_tiles = ceil_div(params.gemm.m, tile_m);
  const int32_t n_tiles = ceil_div(params.gemm.n, tile_n);
  typename Kernel::Arguments args{};
#if FUSE_ENABLE_PROFILING
  if constexpr (Instrumented) {
    args.timeline = timeline;
    args.timeline_capacity = timeline_capacity;
  }
#endif
  args.num_comm_ctas = params.num_comm_ctas;
  args.comm.params = params;
  cudaError_t status = Comm::initialize(args.comm);
  if (status != cudaSuccess) {
    return status;
  }
  args.gemm.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.gemm.problem_shape =
      make_shape(params.gemm.m, params.gemm.n, params.gemm.k, params.gemm.l);
  args.gemm.mainloop.ptr_A = params.lhs;
  args.gemm.mainloop.dA = make_stride(
      a_row_stride(params.gemm), _1{}, a_batch_stride(params.gemm));
  args.gemm.mainloop.ptr_B = params.rhs_nt;
  args.gemm.mainloop.dB = make_stride(
      b_row_stride(params.gemm), _1{}, b_batch_stride(params.gemm));
  args.gemm.epilogue.thread.alpha = params.alpha;
  args.gemm.epilogue.thread.beta = 0.0f;
  args.gemm.epilogue.ptr_C = nullptr;
  args.gemm.epilogue.dC = make_stride(
      d_row_stride(params.gemm), _1{}, d_batch_stride(params.gemm));
  args.gemm.epilogue.ptr_D = params.local_output;
  args.gemm.epilogue.dD = make_stride(
      d_row_stride(params.gemm), _1{}, d_batch_stride(params.gemm));
  args.gemm.epilogue.ready = params.ready;
  args.gemm.epilogue.m_tiles = m_tiles;
  args.gemm.epilogue.n_tiles = n_tiles;
  args.gemm.epilogue.epoch = params.epoch;
  args.gemm.hw_info.device_id = device;
  args.gemm.hw_info.sm_count = sm_count - params.num_comm_ctas;
  args.gemm.scheduler.max_swizzle_size = params.gemm.max_swizzle_size;
  args.gemm.scheduler.block_offset = params.num_comm_ctas;
  args.gemm.scheduler.raster_order =
      raster_option(params.gemm.raster, RasterOptions::AlongM);
  if (!Kernel::can_implement(args) || Kernel::get_workspace_size(args) != 0) {
    return cudaErrorNotSupported;
  }
  if (Kernel::initialize_workspace(args, nullptr, stream) != cutlass::Status::kSuccess) {
    return cudaErrorInitializationError;
  }
  const auto kernel_params = Kernel::to_underlying_arguments(args, nullptr);
  return detail::launch_cooperative<Kernel>(kernel_params, stream, sm_count);
}

using QkvForwardN64Binding = DataflowKernelBinding<
    ulysses::QkvProjection, ulysses::Forward,
    N64OutputGemm, QkvGqaPackCommN64, N64PureGemm>;
using QkvForwardN128Binding = DataflowKernelBinding<
    ulysses::QkvProjection, ulysses::Forward,
    OutputGemm, QkvGqaPackCommSmall, ::fuse::PureGemm>;
using QkvForwardN128InterleavedBinding = DataflowKernelBinding<
    ulysses::QkvProjection, ulysses::Forward,
    OutputGemm, QkvGqaPackCommSmallInterleaved, ::fuse::PureGemm>;
using QkvForwardN160Binding = DataflowKernelBinding<
    ulysses::QkvProjection, ulysses::Forward,
    N160OutputGemm, QkvGqaPackCommN160, N160PureGemm>;
using QkvForwardN192Binding = DataflowKernelBinding<
    ulysses::QkvProjection, ulysses::Forward,
    N192OutputGemm, QkvGqaPackCommN192, N192PureGemm>;
using QkvForwardN256Binding = DataflowKernelBinding<
    ulysses::QkvProjection, ulysses::Forward,
    ProjectionOutputGemm, QkvGqaPackCommWide, ProjectionPureGemm>;
using QkvForwardN320Binding = DataflowKernelBinding<
    ulysses::QkvProjection, ulysses::Forward,
    WideN320OutputGemm, QkvGqaPackCommN320, WideN320PureGemm>;

template <class Visitor>
cudaError_t visit_qkv_forward_policy(
    QkvGemmPolicy policy, bool peer_interleaved, Visitor& visitor) {
  switch (policy) {
    case QkvGemmPolicy::kM128N64:
      return visitor(TypeTag<QkvForwardN64Binding>{});
    case QkvGemmPolicy::kM128N128:
      return peer_interleaved
          ? visitor(TypeTag<QkvForwardN128InterleavedBinding>{})
          : visitor(TypeTag<QkvForwardN128Binding>{});
    case QkvGemmPolicy::kM128N160:
      return visitor(TypeTag<QkvForwardN160Binding>{});
    case QkvGemmPolicy::kM128N192:
      return visitor(TypeTag<QkvForwardN192Binding>{});
    case QkvGemmPolicy::kM128N256ClusterM2:
      return visitor(TypeTag<QkvForwardN256Binding>{});
    case QkvGemmPolicy::kM128N320ClusterM2:
      return visitor(TypeTag<QkvForwardN320Binding>{});
    default:
      return cudaErrorInvalidValue;
  }
}

template <
    bool Instrumented = false
#if !FUSE_ENABLE_PROFILING
    , class = void
#endif
    >
cudaError_t launch_qkv_forward_policy(
    const GemmA2AParams& params,
    QkvGemmPolicy policy,
    cudaStream_t stream,
    int32_t sm_count,
    int32_t device
#if FUSE_ENABLE_PROFILING
    , A2AGemmCtaTimeline* timeline = nullptr,
    int32_t timeline_capacity = 0
#endif
    ) {
  auto launch = [&](auto binding_tag) {
    using Binding = typename decltype(binding_tag)::type;
#if FUSE_ENABLE_PROFILING
    return launch_gemm_a2a_impl<
        typename Binding::Gemm,
        typename Binding::Comm,
        GemmA2AParams,
        Instrumented>(
            params,
            stream,
            sm_count,
            device,
            timeline,
            timeline_capacity);
#else
    static_assert(!Instrumented);
    return launch_gemm_a2a_impl<
        typename Binding::Gemm, typename Binding::Comm>(
            params, stream, sm_count, device);
#endif
  };
  return visit_qkv_forward_policy(
      policy, params.route.qkv_peer_interleaved, launch);
}

using QkvBackwardN64ClusterM2Binding = DataflowKernelBinding<
    ulysses::QkvProjection, ulysses::Backward,
    BackwardN64ClusterM2ReadyGemm, QkvBackwardPushComm>;
using QkvBackwardN64Binding = DataflowKernelBinding<
    ulysses::QkvProjection, ulysses::Backward,
    BackwardN64ReadyGemm, QkvBackwardPushComm>;
using QkvBackwardN128Binding = DataflowKernelBinding<
    ulysses::QkvProjection, ulysses::Backward,
    BackwardN128ReadyGemm, QkvBackwardPushComm>;
using QkvBackwardN160Binding = DataflowKernelBinding<
    ulysses::QkvProjection, ulysses::Backward,
    BackwardN160ReadyGemm, QkvBackwardPushComm>;
using QkvBackwardN192Binding = DataflowKernelBinding<
    ulysses::QkvProjection, ulysses::Backward,
    BackwardN192ReadyGemm, QkvBackwardPushComm>;
using QkvBackwardN256Binding = DataflowKernelBinding<
    ulysses::QkvProjection, ulysses::Backward,
    BackwardA2ALhsGemm, QkvBackwardPushComm>;

template <class Visitor>
cudaError_t visit_qkv_backward_policy(
    BackwardGemmPolicy policy, Visitor& visitor) {
  switch (policy) {
    case BackwardGemmPolicy::kM128N64ClusterM2:
      return visitor(TypeTag<QkvBackwardN64ClusterM2Binding>{});
    case BackwardGemmPolicy::kM128N64:
      return visitor(TypeTag<QkvBackwardN64Binding>{});
    case BackwardGemmPolicy::kM128N128:
      return visitor(TypeTag<QkvBackwardN128Binding>{});
    case BackwardGemmPolicy::kM128N160:
      return visitor(TypeTag<QkvBackwardN160Binding>{});
    case BackwardGemmPolicy::kM128N192:
      return visitor(TypeTag<QkvBackwardN192Binding>{});
    case BackwardGemmPolicy::kM128N256:
      return visitor(TypeTag<QkvBackwardN256Binding>{});
    default:
      return cudaErrorInvalidValue;
  }
}

using OprojBackwardN64Binding = DataflowKernelBinding<
    ulysses::OutputProjection, ulysses::Backward,
    BackwardN64SignalingGemm, OprojBackwardHeadCommN64>;
using OprojBackwardN128Binding = DataflowKernelBinding<
    ulysses::OutputProjection, ulysses::Backward,
    BackwardN128SignalingGemm, OprojBackwardHeadCommN128>;
using OprojBackwardN160Binding = DataflowKernelBinding<
    ulysses::OutputProjection, ulysses::Backward,
    BackwardN160SignalingGemm, OprojBackwardHeadCommN160>;
using OprojBackwardN192Binding = DataflowKernelBinding<
    ulysses::OutputProjection, ulysses::Backward,
    BackwardN192SignalingGemm, OprojBackwardHeadCommN192>;
using OprojBackwardN256Binding = DataflowKernelBinding<
    ulysses::OutputProjection, ulysses::Backward,
    BackwardProjectionOutputGemm, OprojBackwardHeadCommN256>;

template <class Visitor>
cudaError_t visit_oproj_backward_policy(
    BackwardGemmPolicy policy, Visitor& visitor) {
  switch (policy) {
    case BackwardGemmPolicy::kM128N64:
      return visitor(TypeTag<OprojBackwardN64Binding>{});
    case BackwardGemmPolicy::kM128N128:
      return visitor(TypeTag<OprojBackwardN128Binding>{});
    case BackwardGemmPolicy::kM128N160:
      return visitor(TypeTag<OprojBackwardN160Binding>{});
    case BackwardGemmPolicy::kM128N192:
      return visitor(TypeTag<OprojBackwardN192Binding>{});
    case BackwardGemmPolicy::kM128N256:
      return visitor(TypeTag<OprojBackwardN256Binding>{});
    default:
      return cudaErrorInvalidValue;
  }
}

template <
    class Gemm,
    class BaseKernel,
    class KernelParamsType
#if FUSE_ENABLE_PROFILING
    , bool Instrumented = false
#endif
    >
cudaError_t launch_qkv_backward_data_policy(
    const KernelParamsType& params,
    cudaStream_t stream,
    int32_t sm_count,
    int32_t device
#if FUSE_ENABLE_PROFILING
    , A2AGemmCtaTimeline* timeline = nullptr,
    int32_t timeline_capacity = 0
#endif
    ) {
  using Comm = std::conditional_t<
      std::is_same_v<KernelParamsType, Fp8QkvBackwardKernelParams>,
      Fp8QkvBackwardPushComm,
      QkvBackwardPushComm>;
#if FUSE_ENABLE_PROFILING
  using Kernel = std::conditional_t<
      Instrumented,
      GemmA2ARoleTelemetryKernel<Gemm, Comm>,
      BaseKernel>;
#else
  using Kernel = BaseKernel;
#endif
  typename Comm::Arguments comm_args{};
  comm_args.params = params;
  cudaError_t status = Comm::initialize(comm_args);
  if (status != cudaSuccess || !Comm::can_implement(comm_args)) {
    return status != cudaSuccess ? status : cudaErrorNotSupported;
  }

  constexpr int32_t tile_m = static_cast<int32_t>(
      cute::size<0>(typename Gemm::TileShape{}));
  constexpr int32_t tile_k = static_cast<int32_t>(
      cute::size<2>(typename Gemm::TileShape{}));
  const int32_t packed_heads =
      params.route.q_heads + 2 * params.route.kv_heads;
  if (params.route.head_dim % tile_k != 0) {
    return cudaErrorNotSupported;
  }

  typename Kernel::Arguments args{};
#if FUSE_ENABLE_PROFILING
  if constexpr (Instrumented) {
    args.timeline = timeline;
    args.timeline_capacity = timeline_capacity;
  }
#endif
  args.num_comm_ctas = params.num_comm_ctas;
  args.comm = comm_args;
  args.gemm.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.gemm.problem_shape = make_shape(
      params.gemm.m, params.gemm.n, params.gemm.k, 1);
  args.gemm.mainloop.ptr_A = params.peer_staging[params.route.rank];
  args.gemm.mainloop.dA = make_stride(
      static_cast<int64_t>(params.gemm.k),
      _1{},
      static_cast<int64_t>(params.gemm.m) * params.gemm.k);
  args.gemm.mainloop.ptr_B = params.weight;
  if constexpr (std::is_same_v<
                    KernelParamsType, Fp8QkvBackwardKernelParams>) {
    // FP8 dgrad consumes the TN-friendly [N,K] quantized transpose copy.
    args.gemm.mainloop.dB = make_stride(
        static_cast<int64_t>(params.gemm.k),
        _1{},
        static_cast<int64_t>(params.gemm.k) * params.gemm.n);
  } else {
    // BF16 consumes the original row-major forward weight [K,N].
    args.gemm.mainloop.dB = make_stride(
        _1{},
        static_cast<int64_t>(params.gemm.n),
        static_cast<int64_t>(params.gemm.k) * params.gemm.n);
  }
  args.gemm.mainloop.ready = params.peer_ready[params.route.rank];
  args.gemm.mainloop.world_size = packed_heads;
  args.gemm.mainloop.m_tiles = ceil_div(params.gemm.m, tile_m);
  args.gemm.mainloop.arrivals_per_peer = 1;
  args.gemm.mainloop.k_tiles_per_peer = params.route.head_dim / tile_k;
  args.gemm.mainloop.epoch = params.epoch;
  args.gemm.epilogue.thread.alpha = params.alpha;
  args.gemm.epilogue.thread.beta = 0.0f;
  args.gemm.epilogue.ptr_C = nullptr;
  args.gemm.epilogue.dC = make_stride(
      static_cast<int64_t>(params.gemm.n),
      _1{},
      static_cast<int64_t>(params.gemm.m) * params.gemm.n);
  args.gemm.epilogue.ptr_D = params.grad_input;
  args.gemm.epilogue.dD = make_stride(
      static_cast<int64_t>(params.gemm.n),
      _1{},
      static_cast<int64_t>(params.gemm.m) * params.gemm.n);
  args.gemm.hw_info.device_id = device;
  args.gemm.hw_info.sm_count = sm_count - params.num_comm_ctas;
  args.gemm.scheduler.max_swizzle_size = 1;
  args.gemm.scheduler.block_offset = params.num_comm_ctas;
  args.gemm.scheduler.raster_order = RasterOptions::AlongN;
  if (!Kernel::can_implement(args) || Kernel::get_workspace_size(args) != 0) {
    return cudaErrorNotSupported;
  }
  if (Kernel::initialize_workspace(args, nullptr, stream) !=
      cutlass::Status::kSuccess) {
    return cudaErrorInitializationError;
  }
  return detail::launch_cooperative<Kernel>(
      Kernel::to_underlying_arguments(args, nullptr), stream, sm_count);
}

#if FUSE_ENABLE_PROFILING
cudaError_t launch_qkv_backward_data_telemetry_impl(
    const QkvBackwardKernelParams& params,
    A2AGemmCtaTimeline* timeline,
    int32_t timeline_capacity,
    cudaStream_t stream,
    int32_t sm_count,
    int32_t device) {
  if (timeline == nullptr || timeline_capacity < sm_count) {
    return cudaErrorInvalidValue;
  }
  auto launch = [&](auto binding_tag) {
    using Binding = typename decltype(binding_tag)::type;
    return launch_qkv_backward_data_policy<
        typename Binding::Gemm,
        typename Binding::Kernel,
        QkvBackwardKernelParams,
        true>(
            params, stream, sm_count, device, timeline, timeline_capacity);
  };
  return visit_qkv_backward_policy(params.gemm_policy, launch);
}
#endif

cudaError_t launch_qkv_backward_data_impl(
    const QkvBackwardKernelParams& params,
    cudaStream_t stream,
    int32_t sm_count,
    int32_t device) {
  auto launch = [&](auto binding_tag) {
    using Binding = typename decltype(binding_tag)::type;
    return launch_qkv_backward_data_policy<
        typename Binding::Gemm, typename Binding::Kernel>(
            params, stream, sm_count, device);
  };
  return visit_qkv_backward_policy(params.gemm_policy, launch);
}

cudaError_t launch_qkv_backward_fp8_data_impl(
    const Fp8QkvBackwardKernelParams& params,
    cudaStream_t stream,
    int32_t sm_count,
    int32_t device) {
  return launch_qkv_backward_data_policy<
      Fp8BackwardReadyGemm,
      Fp8QkvBackwardDataKernel>(
          params, stream, sm_count, device);
}

#if FUSE_ENABLE_PROFILING
cudaError_t launch_qkv_backward_fp8_data_telemetry_impl(
    const Fp8QkvBackwardKernelParams& params,
    A2AGemmCtaTimeline* timeline,
    int32_t timeline_capacity,
    cudaStream_t stream,
    int32_t sm_count,
    int32_t device) {
  if (timeline == nullptr || timeline_capacity < sm_count) {
    return cudaErrorInvalidValue;
  }
  return launch_qkv_backward_data_policy<
      Fp8BackwardReadyGemm,
      Fp8QkvBackwardDataKernel,
      Fp8QkvBackwardKernelParams,
      true>(
          params,
          stream,
          sm_count,
          device,
          timeline,
          timeline_capacity);
}
#endif

template <
    class Gemm,
    class Comm,
    class BaseKernel,
    class KernelParamsType
#if FUSE_ENABLE_PROFILING
    , bool Instrumented = false
#endif
    >
cudaError_t launch_oproj_backward_data_policy(
    const KernelParamsType& params,
    cudaStream_t stream,
    int32_t sm_count,
    int32_t device
#if FUSE_ENABLE_PROFILING
    , A2AGemmCtaTimeline* timeline = nullptr,
    int32_t timeline_capacity = 0
#endif
    ) {
#if FUSE_ENABLE_PROFILING
  using Kernel = std::conditional_t<
      Instrumented,
      GemmA2ARoleTelemetryKernel<Gemm, Comm>,
      BaseKernel>;
#else
  using Kernel = BaseKernel;
#endif
  constexpr int32_t tile_m = static_cast<int32_t>(
      cute::size<0>(typename Gemm::TileShape{}));
  constexpr int32_t tile_n = static_cast<int32_t>(
      cute::size<1>(typename Gemm::TileShape{}));
  const int32_t m_tiles = ceil_div(params.gemm.m, tile_m);
  const int32_t n_tiles = ceil_div(params.gemm.n, tile_n);

  typename Kernel::Arguments args{};
#if FUSE_ENABLE_PROFILING
  if constexpr (Instrumented) {
    args.timeline = timeline;
    args.timeline_capacity = timeline_capacity;
  }
#endif
  args.num_comm_ctas = params.num_comm_ctas;
  args.comm.params = params;
  cudaError_t status = Comm::initialize(args.comm);
  if (status != cudaSuccess) {
    return status;
  }
  args.gemm.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.gemm.problem_shape = make_shape(
      params.gemm.m, params.gemm.n, params.gemm.k, 1);
  args.gemm.mainloop.ptr_A = params.lhs;
  args.gemm.mainloop.dA = make_stride(
      static_cast<int64_t>(params.gemm.k),
      _1{},
      static_cast<int64_t>(params.gemm.m) * params.gemm.k);
  args.gemm.mainloop.ptr_B = params.rhs_nt;
  if constexpr (std::is_same_v<
                    KernelParamsType, Fp8OprojBackwardKernelParams>) {
    args.gemm.mainloop.dB = make_stride(
        static_cast<int64_t>(params.gemm.k),
        _1{},
        static_cast<int64_t>(params.gemm.k) * params.gemm.n);
  } else {
    args.gemm.mainloop.dB = make_stride(
        _1{},
        static_cast<int64_t>(params.gemm.n),
        static_cast<int64_t>(params.gemm.k) * params.gemm.n);
  }
  args.gemm.epilogue.thread.alpha = params.alpha;
  args.gemm.epilogue.thread.beta = 0.0f;
  args.gemm.epilogue.ptr_C = nullptr;
  args.gemm.epilogue.dC = make_stride(
      static_cast<int64_t>(params.gemm.n),
      _1{},
      static_cast<int64_t>(params.gemm.m) * params.gemm.n);
  args.gemm.epilogue.ptr_D = params.local_output;
  args.gemm.epilogue.dD = make_stride(
      static_cast<int64_t>(params.gemm.n),
      _1{},
      static_cast<int64_t>(params.gemm.m) * params.gemm.n);
  args.gemm.epilogue.ready = params.ready;
  args.gemm.epilogue.m_tiles = m_tiles;
  args.gemm.epilogue.n_tiles = n_tiles;
  args.gemm.epilogue.epoch = params.epoch;
  args.gemm.hw_info.device_id = device;
  args.gemm.hw_info.sm_count = sm_count - params.num_comm_ctas;
  args.gemm.scheduler.max_swizzle_size = 1;
  args.gemm.scheduler.block_offset = params.num_comm_ctas;
  args.gemm.scheduler.raster_order = RasterOptions::AlongN;
  if (!Kernel::can_implement(args) || Kernel::get_workspace_size(args) != 0) {
    return cudaErrorNotSupported;
  }
  if (Kernel::initialize_workspace(args, nullptr, stream) !=
      cutlass::Status::kSuccess) {
    return cudaErrorInitializationError;
  }
  return detail::launch_cooperative<Kernel>(
      Kernel::to_underlying_arguments(args, nullptr), stream, sm_count);
}

#if FUSE_ENABLE_PROFILING
cudaError_t launch_oproj_backward_data_telemetry_impl(
    const OprojBackwardKernelParams& params,
    BackwardGemmPolicy policy,
    A2AGemmCtaTimeline* timeline,
    int32_t timeline_capacity,
    cudaStream_t stream,
    int32_t sm_count,
    int32_t device) {
  if (timeline == nullptr || timeline_capacity < sm_count) {
    return cudaErrorInvalidValue;
  }
  auto launch = [&](auto binding_tag) {
    using Binding = typename decltype(binding_tag)::type;
    return launch_oproj_backward_data_policy<
        typename Binding::Gemm,
        typename Binding::Comm,
        typename Binding::Kernel,
        OprojBackwardKernelParams,
        true>(
            params,
            stream,
            sm_count,
            device,
            timeline,
            timeline_capacity);
  };
  return visit_oproj_backward_policy(policy, launch);
}
#endif

cudaError_t launch_oproj_backward_data_impl(
    const OprojBackwardKernelParams& params,
    BackwardGemmPolicy policy,
    cudaStream_t stream,
    int32_t sm_count,
    int32_t device) {
  auto launch = [&](auto binding_tag) {
    using Binding = typename decltype(binding_tag)::type;
    return launch_oproj_backward_data_policy<
        typename Binding::Gemm,
        typename Binding::Comm,
        typename Binding::Kernel>(params, stream, sm_count, device);
  };
  return visit_oproj_backward_policy(policy, launch);
}

cudaError_t launch_oproj_backward_fp8_data_impl(
    const Fp8OprojBackwardKernelParams& params,
    cudaStream_t stream,
    int32_t sm_count,
    int32_t device) {
  return launch_oproj_backward_data_policy<
      Fp8BackwardSignalingGemm,
      Fp8OprojBackwardHeadComm,
      Fp8OprojBackwardDataKernel>(
          params, stream, sm_count, device);
}

#if FUSE_ENABLE_PROFILING
cudaError_t launch_oproj_backward_fp8_data_telemetry_impl(
    const Fp8OprojBackwardKernelParams& params,
    A2AGemmCtaTimeline* timeline,
    int32_t timeline_capacity,
    cudaStream_t stream,
    int32_t sm_count,
    int32_t device) {
  if (timeline == nullptr || timeline_capacity < sm_count) {
    return cudaErrorInvalidValue;
  }
  return launch_oproj_backward_data_policy<
      Fp8BackwardSignalingGemm,
      Fp8OprojBackwardHeadComm,
      Fp8OprojBackwardDataKernel,
      Fp8OprojBackwardKernelParams,
      true>(
          params,
          stream,
          sm_count,
          device,
          timeline,
          timeline_capacity);
}
#endif

cudaError_t launch_backward_wgrad_impl(
    const Bf16* output_gradient,
    const Bf16* saved_input,
    Bf16* grad_weight,
    int32_t weight_rows,
    int32_t weight_columns,
    int32_t tokens,
    float alpha,
    float beta,
    cudaStream_t stream,
    int32_t sm_count,
    int32_t device) {
  if (!output_gradient || !saved_input || !grad_weight ||
      weight_rows <= 0 || weight_columns <= 0 || tokens <= 0 ||
      weight_rows % kAlignment != 0 ||
      weight_columns % kAlignment != 0 || tokens % kAlignment != 0) {
    return cudaErrorInvalidValue;
  }
  typename BackwardWgradGemm::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape = make_shape(
      weight_rows, weight_columns, tokens, 1);
  args.mainloop.ptr_A = output_gradient;
  // output_gradient is [tokens,weight_rows] row-major and is consumed as its
  // zero-copy column-major transpose [weight_rows,tokens].
  args.mainloop.dA = make_stride(
      _1{},
      static_cast<int64_t>(weight_rows),
      static_cast<int64_t>(tokens) * weight_rows);
  args.mainloop.ptr_B = saved_input;
  // CUTLASS B coordinates are [N,K]; saved_input is [K,N] row-major.
  args.mainloop.dB = make_stride(
      _1{},
      static_cast<int64_t>(weight_columns),
      static_cast<int64_t>(tokens) * weight_columns);
  args.epilogue.thread.alpha = alpha;
  args.epilogue.thread.beta = beta;
  args.epilogue.ptr_C = beta == 0.0f ? nullptr : grad_weight;
  args.epilogue.dC = make_stride(
      static_cast<int64_t>(weight_columns),
      _1{},
      static_cast<int64_t>(weight_rows) * weight_columns);
  args.epilogue.ptr_D = grad_weight;
  args.epilogue.dD = make_stride(
      static_cast<int64_t>(weight_columns),
      _1{},
      static_cast<int64_t>(weight_rows) * weight_columns);
  args.hw_info.device_id = device;
  args.hw_info.sm_count = sm_count;
  args.scheduler.max_swizzle_size = 1;
  args.scheduler.raster_order = RasterOptions::AlongN;
  if (!BackwardWgradGemm::can_implement(args) ||
      BackwardWgradGemm::get_workspace_size(args) != 0) {
    return cudaErrorNotSupported;
  }
  if (BackwardWgradGemm::initialize_workspace(args, nullptr, stream) !=
      cutlass::Status::kSuccess) {
    return cudaErrorInitializationError;
  }
  return launch_regular<BackwardWgradGemm>(
      BackwardWgradGemm::to_underlying_arguments(args, nullptr), stream);
}

cudaError_t launch_backward_fp8_wgrad_impl(
    const Fp8Element* output_gradient,
    const Fp8Element* saved_input,
    Fp8Element* grad_weight,
    int32_t weight_rows,
    int32_t weight_columns,
    int32_t tokens,
    float alpha,
    float beta,
    cudaStream_t stream,
    int32_t sm_count,
    int32_t device) {
  if (!output_gradient || !saved_input || !grad_weight ||
      weight_rows <= 0 || weight_columns <= 0 || tokens <= 0 ||
      weight_rows % kFp8Alignment != 0 ||
      weight_columns % kFp8Alignment != 0 ||
      tokens % kFp8Alignment != 0) {
    return cudaErrorInvalidValue;
  }
  typename Fp8BackwardWgradGemm::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape = make_shape(
      weight_rows, weight_columns, tokens, 1);
  args.mainloop.ptr_A = output_gradient;
  // FP8 wgrad receives the quantized transpose [weight_rows,tokens].
  args.mainloop.dA = make_stride(
      static_cast<int64_t>(tokens),
      _1{},
      static_cast<int64_t>(tokens) * weight_rows);
  args.mainloop.ptr_B = saved_input;
  // saved_input_t is [weight_columns,tokens].
  args.mainloop.dB = make_stride(
      static_cast<int64_t>(tokens),
      _1{},
      static_cast<int64_t>(tokens) * weight_columns);
  args.epilogue.thread.alpha = alpha;
  args.epilogue.thread.beta = beta;
  args.epilogue.ptr_C = beta == 0.0f ? nullptr : grad_weight;
  args.epilogue.dC = make_stride(
      static_cast<int64_t>(weight_columns),
      _1{},
      static_cast<int64_t>(weight_rows) * weight_columns);
  args.epilogue.ptr_D = grad_weight;
  args.epilogue.dD = make_stride(
      static_cast<int64_t>(weight_columns),
      _1{},
      static_cast<int64_t>(weight_rows) * weight_columns);
  args.hw_info.device_id = device;
  args.hw_info.sm_count = sm_count;
  args.scheduler.max_swizzle_size = 1;
  args.scheduler.raster_order = RasterOptions::AlongN;
  if (!Fp8BackwardWgradGemm::can_implement(args) ||
      Fp8BackwardWgradGemm::get_workspace_size(args) != 0) {
    return cudaErrorNotSupported;
  }
  if (Fp8BackwardWgradGemm::initialize_workspace(args, nullptr, stream) !=
      cutlass::Status::kSuccess) {
    return cudaErrorInitializationError;
  }
  return launch_regular<Fp8BackwardWgradGemm>(
      Fp8BackwardWgradGemm::to_underlying_arguments(args, nullptr), stream);
}

BackwardGemmPolicy backward_policy_from_qkv_policy(QkvGemmPolicy policy) {
  switch (policy) {
    case QkvGemmPolicy::kM128N64:
      return BackwardGemmPolicy::kM128N64;
    case QkvGemmPolicy::kM128N128:
      return BackwardGemmPolicy::kM128N128;
    case QkvGemmPolicy::kM128N160:
      return BackwardGemmPolicy::kM128N160;
    case QkvGemmPolicy::kM128N192:
      return BackwardGemmPolicy::kM128N192;
    case QkvGemmPolicy::kM128N256ClusterM2:
      return BackwardGemmPolicy::kM128N256;
    case QkvGemmPolicy::kM128N320ClusterM2:
      return BackwardGemmPolicy::kM128N256;
    default:
      return BackwardGemmPolicy::kM128N128;
  }
}

QkvComputeEstimate estimate_backward_compute(
    const GemmProblem& problem,
    int32_t num_comm_ctas,
    int32_t sm_count) {
  QkvComputeEstimate best{};
  if (problem.m <= 0 || problem.n <= 0 || problem.k <= 0 ||
      problem.l <= 0 || num_comm_ctas <= 0 ||
      num_comm_ctas >= sm_count) {
    return best;
  }
  const int32_t compute_ctas = sm_count - num_comm_ctas;
  int64_t best_score = std::numeric_limits<int64_t>::max();
  int32_t best_tie_rank = std::numeric_limits<int32_t>::max();
  QkvComputeEstimate best_independent{};
  int64_t best_independent_score = std::numeric_limits<int64_t>::max();

  // N320 is deliberately absent: SM90's row-major-B dgrad mainloop cannot
  // legally instantiate that GMMA geometry without first transposing the
  // stored forward weight. The first five entries are all zero-copy layouts.
  for (size_t index = 0; index < 5; ++index) {
    const auto& candidate = kQkvPolicyGeometries[index];
    if (candidate.cluster_m == 2 &&
        (num_comm_ctas % 2 != 0 || compute_ctas % 2 != 0)) {
      continue;
    }
    int64_t wave_ns = interpolate_qkv_wave_ns(
        kQkvWaveCalibrationComm24,
        max(
            kQkvWaveCalibrationComm24.front().k,
            min(problem.k, kQkvWaveCalibrationComm24.back().k)),
        index);
    if (wave_ns <= 0) {
      continue;
    }
    if (problem.k > kQkvWaveCalibrationComm24.back().k) {
      // Above the last calibrated K, tensor-core service is dominated by the
      // K loop. Scale every legal tile by the same physical work ratio rather
      // than falling back to an unrelated legacy tile.
      wave_ns =
          (wave_ns * problem.k +
           kQkvWaveCalibrationComm24.back().k / 2) /
          kQkvWaveCalibrationComm24.back().k;
    }
    const int64_t workers = compute_ctas / candidate.cluster_m;
    if (workers <= 0) {
      continue;
    }
    const int64_t m_tiles = ceil_div(problem.m, kBlockM);
    const int64_t work_units =
        ceil_div(m_tiles, candidate.cluster_m) *
        ceil_div(problem.n, candidate.tile_n) * problem.l;
    const int64_t waves = ceil_div(work_units, workers);
    const int64_t score = waves * wave_ns;
    QkvComputeEstimate estimate{};
    estimate.policy = candidate.policy;
    estimate.tile_n = candidate.tile_n;
    estimate.cluster_m = candidate.cluster_m;
    estimate.waves = waves;
    estimate.wave_ns = wave_ns;
    estimate.total_ns = score;
    estimate.valid = true;
    if (candidate.cluster_m == 1 && score < best_independent_score) {
      best_independent = estimate;
      best_independent_score = score;
    }
    const int32_t tie_rank = static_cast<int32_t>(index);
    if (score < best_score ||
        (score == best_score && tie_rank < best_tie_rank)) {
      best = estimate;
      best_score = score;
      best_tie_rank = tie_rank;
    }
  }
  if (best.valid && best.cluster_m == 2 && best_independent.valid &&
      best_independent.waves == best.waves &&
      best_independent.total_ns <=
          best.total_ns + best.cluster_m * kQkvRouteTaskWaveNs) {
    return best_independent;
  }
  return best;
}

BackwardGemmPolicy select_backward_gemm_policy(
    BackwardGemmPolicy request,
    int32_t m,
    int32_t n,
    int32_t k,
    int32_t comm_ctas,
    int32_t sm_count) {
  if (request != BackwardGemmPolicy::kAuto) {
    return request;
  }
  GemmProblem problem{m, n, k, 1};
  problem.raster = GemmRaster::kAlongN;
  const auto estimate = estimate_backward_compute(
      problem, comm_ctas, sm_count);
  return backward_policy_from_qkv_policy(
      estimate.valid ? estimate.policy : legacy_qkv_gemm_policy(problem));
}

template <class Kernel, class Gemm>
KernelTraits backward_traits() {
  return {
      static_cast<int32_t>(cute::size<0>(typename Gemm::TileShape{})),
      static_cast<int32_t>(cute::size<1>(typename Gemm::TileShape{})),
      static_cast<int32_t>(cute::size<2>(typename Gemm::TileShape{})),
      static_cast<int32_t>(Kernel::get_block_shape().x),
      static_cast<int32_t>(sizeof(typename Kernel::SharedStorage))};
}

KernelTraits qkv_backward_traits_for_policy(BackwardGemmPolicy policy) {
  KernelTraits traits{};
  auto read_traits = [&](auto binding_tag) {
    using Binding = typename decltype(binding_tag)::type;
    traits = backward_traits<
        typename Binding::Kernel, typename Binding::Gemm>();
    return cudaSuccess;
  };
  return visit_qkv_backward_policy(policy, read_traits) == cudaSuccess
      ? traits
      : KernelTraits{};
}

KernelTraits oproj_backward_traits_for_policy(BackwardGemmPolicy policy) {
  KernelTraits traits{};
  auto read_traits = [&](auto binding_tag) {
    using Binding = typename decltype(binding_tag)::type;
    traits = backward_traits<
        typename Binding::Kernel, typename Binding::Gemm>();
    return cudaSuccess;
  };
  return visit_oproj_backward_policy(policy, read_traits) == cudaSuccess
      ? traits
      : KernelTraits{};
}

template <class InputGemm, int32_t ReadyBlockM>
cudaError_t launch_weighted_a2a_policy(
    const WeightedA2AGemmKernelParams& segment,
    cudaStream_t stream,
    int32_t sm_count,
    int32_t device) {
  using Comm = WeightedA2ALhsInputComm<ReadyBlockM, true>;
  using Kernel = detail::MonolithicGemm<InputGemm, Comm>;
#if FUSE_ENABLE_PROFILING
  return launch_a2a_lhs_gemm_policy<
      InputGemm,
      Kernel,
      Comm,
      false,
      WeightedA2AGemmKernelParams>(
          segment, stream, sm_count, device);
#else
  return launch_a2a_lhs_gemm_policy<
      InputGemm,
      Kernel,
      Comm,
      WeightedA2AGemmKernelParams>(
          segment, stream, sm_count, device);
#endif
}

}  // namespace
}  // namespace fuse
