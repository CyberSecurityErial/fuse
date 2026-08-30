// SPDX-License-Identifier: BSD-3-Clause
// Private implementation; assembled by csrc/operators/ulysses_sm90.cu.
//
// Module index:
//   - optional A2A-input profiling samples
//   - TMA descriptor construction for peer input staging
//   - A2ALhsInputCommT and weighted A2A-then-GEMM communication roles

namespace fuse {
namespace {

// Pull the sequence rows owned by this destination rank from every
// head-sharded peer output and place them directly in the row-major GEMM A
// staging matrix. A peer completes one contiguous K interval for each M tile;
// the CUTLASS producer can therefore advance peer by peer without waiting for
// the full inverse A2A.
#if FUSE_ENABLE_PROFILING
template <bool Instrumented>
struct A2ALhsCommTimelineArguments {};

template <>
struct A2ALhsCommTimelineArguments<true> {
  A2AGemmPeerTimeline* peer_timeline = nullptr;
  int32_t peer_timeline_capacity = 0;
};
#endif

#if FUSE_ENABLE_PROFILING
template <bool Instrumented>
struct A2ALhsCommStageSample {};

template <>
struct A2ALhsCommStageSample<true> {
  uint64_t task_begin = 0;
  uint64_t input_ready = 0;
  uint64_t g2s_issue = 0;
  uint64_t g2s_done = 0;
  uint64_t s2g_issue = 0;
  uint64_t s2g_done = 0;
  int32_t comm_cta = 0;
  int32_t comm_slot = 0;
  int32_t task_id = 0;
  int32_t row_chunk = 0;
  int32_t copy_rows = 0;
  int32_t source_rank = 0;
  int32_t copy_path = 0;
};
#endif

cudaError_t make_a2a_lhs_store_tma_3d(
    CUtensorMap* tensor_map,
    void* pointer,
    int32_t inner_u64,
    int32_t total_groups,
    int32_t rows,
    int32_t peer_groups,
    int32_t box_rows) {
  const uint64_t global_dims[3] = {
      static_cast<uint64_t>(inner_u64),
      static_cast<uint64_t>(total_groups),
      static_cast<uint64_t>(rows)};
  const uint64_t global_strides[2] = {
      static_cast<uint64_t>(inner_u64) * sizeof(uint64_t),
      static_cast<uint64_t>(inner_u64) * total_groups * sizeof(uint64_t)};
  const uint32_t box_dims[3] = {
      static_cast<uint32_t>(inner_u64),
      static_cast<uint32_t>(peer_groups),
      static_cast<uint32_t>(box_rows)};
  constexpr uint32_t element_strides[3] = {1, 1, 1};
  const CUresult result =
      CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
          tensor_map,
          CU_TENSOR_MAP_DATA_TYPE_UINT64,
          3,
          pointer,
          global_dims,
          global_strides,
          box_dims,
          element_strides,
          CU_TENSOR_MAP_INTERLEAVE_NONE,
          CU_TENSOR_MAP_SWIZZLE_NONE,
          CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return result == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

// A2A -> GEMM: remote input staging and per-peer K-shard publication.
template <
    int32_t ReadyBlockM,
    class ParamsType = A2AGemmParams,
    bool Heterogeneous = false,
    bool FinalizeAcrossRanks = false
#if FUSE_ENABLE_PROFILING
    , bool Instrumented = false
#endif
    >
struct A2ALhsInputCommT {
  using CommElement = std::remove_cv_t<std::remove_pointer_t<
      decltype(ParamsType{}.input_staging)>>;
  static constexpr bool kFp8Input =
      std::is_same_v<CommElement, Fp8Element>;
  static constexpr int32_t kCommElementsPerVector =
      16 / sizeof(CommElement);
  static constexpr int32_t kReadyBlockM = ReadyBlockM;
  static constexpr size_t SharedStorageBytes =
      kA2ALhsBulkSlots * kA2ALhsBulkStageBytes +
      kA2ALhsBulkSlots * sizeof(uint64_t);
  struct Arguments
#if FUSE_ENABLE_PROFILING
      : A2ALhsCommTimelineArguments<Instrumented>
#endif
  {
    ParamsType params{};
    CUtensorMap store_tma_full{};
    int32_t comm_rows = kA2ALhsCommRows;
    int32_t m_window = 2;
    int32_t store_peer_groups = 0;
    int32_t store_rows = 0;
    bool use_bulk = false;
    bool use_tensor_store = false;
  };
  using Params = Arguments;
  static constexpr bool kNeedsGridFinalize = FinalizeAcrossRanks;

  CUTLASS_HOST_DEVICE static int32_t logical_source_rank(
      const ParamsType& p) {
    if constexpr (Heterogeneous) {
      return p.logical_source_rank;
    }
    return p.route.rank;
  }

  CUTLASS_HOST_DEVICE static int32_t source_row_begin(
      const ParamsType& p) {
    if constexpr (Heterogeneous) {
      return p.source_row_begin;
    }
    return 0;
  }

  CUTLASS_HOST_DEVICE static int32_t executor_rank(const ParamsType& p) {
    if constexpr (Heterogeneous) {
      return p.executor_rank;
    }
    return p.route.rank;
  }

  static cudaError_t initialize(Arguments& args) {
    const auto& p = args.params;
    const auto& route = p.route;
    if (route.world_size <= 0) {
      return cudaErrorInvalidValue;
    }
    const int32_t shard_width = route.local_heads * route.head_dim;
    const int32_t row_bytes = shard_width * sizeof(CommElement);
    const int32_t max_rows = row_bytes > 0
        ? kA2ALhsBulkStageBytes / row_bytes
        : 0;
    args.comm_rows = min(ReadyBlockM, max_rows);
    args.use_bulk = args.comm_rows > 0 && row_bytes % 16 == 0 &&
        route.seq_local % ReadyBlockM == 0 &&
        source_row_begin(p) % ReadyBlockM == 0 &&
        (!route.causal_load_balanced ||
         (route.seq_local / 2) % ReadyBlockM == 0);
    if (!args.use_bulk) {
      args.comm_rows = kA2ALhsCommRows;
      return cudaSuccess;
    }

    constexpr int32_t kTensorStoreBytes = 8 * 1024;
    if (row_bytes > 1024) {
      return cudaSuccess;
    }
    args.store_rows = min(args.comm_rows, kTensorStoreBytes / row_bytes);
    if (args.store_rows < 2) {
      return cudaSuccess;
    }

    // Store a narrow peer shard as a small 3D tensor instead of one TMA per
    // row. The descriptor treats either precision as raw 64-bit groups; the
    // first two dimensions factor the contiguous K shard and the third is M.
    const int32_t shard_u64 = row_bytes / sizeof(uint64_t);
    int32_t inner_u64 = 0;
    for (int32_t candidate = 256; candidate >= 2; candidate >>= 1) {
      if (shard_u64 % candidate == 0) {
        inner_u64 = candidate;
        break;
      }
    }
    const int32_t peer_groups = inner_u64 > 0
        ? shard_u64 / inner_u64
        : 0;
    if (inner_u64 == 0 || peer_groups <= 0 || peer_groups > 256) {
      return cudaSuccess;
    }

    const int32_t total_groups = peer_groups * route.world_size;
    const cudaError_t status = make_a2a_lhs_store_tma_3d(
        &args.store_tma_full,
        p.input_staging,
        inner_u64,
        total_groups,
        p.gemm.m,
        peer_groups,
        args.store_rows);
    if (status != cudaSuccess) {
      return cudaSuccess;
    }
    args.store_peer_groups = peer_groups;
    args.use_tensor_store = true;
    return cudaSuccess;
  }

  static bool can_implement(const Arguments& args) {
    const auto& p = args.params;
    const auto& route = p.route;
    bool pointers = p.input_staging && p.rhs_nt && p.output && p.ready;
    for (int32_t peer = 0; peer < route.world_size && pointers; ++peer) {
      pointers = p.peer_input[peer] != nullptr &&
          (p.input_epoch == 0 || p.peer_input_ready[peer] != nullptr);
      if constexpr (Heterogeneous) {
        pointers = pointers && p.peer_done_epoch[peer] != nullptr;
      }
    }
    bool rows_supported = p.gemm.m == route.batch * route.seq_local;
    bool sequence_supported =
        route.global_seq == route.seq_local * route.world_size;
    if constexpr (Heterogeneous) {
      const bool rank_supported = p.executor_rank >= 0 &&
          p.executor_rank < route.world_size &&
          p.logical_source_rank >= 0 &&
          p.logical_source_rank < route.world_size;
      if (p.weighted_partition) {
        rows_supported = rank_supported &&
            p.gemm.m == route.batch * route.seq_local &&
            p.source_row_begin == 0 && p.global_sequence_begin >= 0;
        sequence_supported = !route.causal_load_balanced &&
            p.global_sequence_begin + route.seq_local <= route.global_seq;
      } else {
        rows_supported = rank_supported && p.source_row_begin >= 0 &&
            p.source_row_begin + p.gemm.m <=
                route.batch * route.seq_local;
      }
    }
    const bool problem_supported = kFp8Input
        ? supported_fp8_problem(p.gemm)
        : supported_problem(p.gemm);
    return pointers && p.epoch > 0 && problem_supported &&
        supported_route_base(route) &&
        route.kind == RouteKind::kHeadToSequence &&
        route.direction == RouteDirection::kInverse &&
        route.q_heads > 0 && route.local_heads > 0 &&
        route.q_heads == route.local_heads * route.world_size &&
        route.head_dim > 0 &&
        route.head_dim % kCommElementsPerVector == 0 &&
        sequence_supported &&
        (!route.causal_load_balanced || route.seq_local % 2 == 0) &&
        p.gemm.l == 1 && rows_supported &&
        p.gemm.k == route.q_heads * route.head_dim &&
        a_row_stride(p.gemm) == p.gemm.k;
  }

  static Params to_underlying_arguments(const Arguments& args) { return args; }

  __host__ __device__ static int32_t arrivals_per_peer(
      const Params& args) {
    return ceil_div(ReadyBlockM, args.comm_rows);
  }

  CUTLASS_DEVICE static void publish_ready(
      const Params& args,
      int32_t tile_m,
      int32_t peer_slot) {
    const auto& p = args.params;
    const auto& route = p.route;
    auto* ready = p.ready +
        (static_cast<int64_t>(tile_m) * route.world_size + peer_slot) *
            kReadyFlagStride;
    detail::add_release_gpu(ready);
  }

#if FUSE_ENABLE_PROFILING
  CUTLASS_DEVICE static void publish_ready_instrumented(
      const Params& args,
      int32_t tile_m,
      int32_t peer_slot,
      const A2ALhsCommStageSample<true>& sample) {
    const auto& p = args.params;
    const auto& route = p.route;
    auto* ready = p.ready +
        (static_cast<int64_t>(tile_m) * route.world_size + peer_slot) *
            kReadyFlagStride;
    const uint64_t release_issue = static_cast<unsigned long long>(
        detail::read_global_timer());
    const uint32_t old = detail::add_release_gpu_fetch_old(ready);
    const uint64_t release_done = static_cast<unsigned long long>(
        detail::read_global_timer());
    const uint32_t target = p.epoch * arrivals_per_peer(args);
    const int32_t index = tile_m * route.world_size + peer_slot;
    if (old + 1 == target && args.peer_timeline && index >= 0 &&
        index < args.peer_timeline_capacity) {
      auto& event = args.peer_timeline[index];
      event.release = release_done;
      event.task_begin = sample.task_begin;
      event.input_ready = sample.input_ready;
      event.g2s_issue = sample.g2s_issue;
      event.g2s_done = sample.g2s_done;
      event.s2g_issue = sample.s2g_issue;
      event.s2g_done = sample.s2g_done;
      event.publish_issue = release_issue;
      event.comm_cta = sample.comm_cta;
      event.comm_slot = sample.comm_slot;
      event.task_id = sample.task_id;
      event.row_chunk = sample.row_chunk;
      event.copy_rows = sample.copy_rows;
      event.source_rank = sample.source_rank;
      event.copy_path = sample.copy_path;
      event.comm_valid = 1;
    }
  }
#endif

  CUTLASS_DEVICE static int32_t global_sequence_row(
      const ParamsType& p,
      int32_t local_sequence) {
    const auto& route = p.route;
    if constexpr (Heterogeneous) {
      if (p.weighted_partition) {
        return p.global_sequence_begin + local_sequence;
      }
    }
    if (!route.causal_load_balanced) {
      return logical_source_rank(p) * route.seq_local + local_sequence;
    }
    const int32_t chunk_rows = route.seq_local / 2;
    const int32_t chunk = local_sequence < chunk_rows
        ? logical_source_rank(p)
        : 2 * route.world_size - logical_source_rank(p) - 1;
    const int32_t row_in_chunk = local_sequence < chunk_rows
        ? local_sequence
        : local_sequence - chunk_rows;
    return chunk * chunk_rows + row_in_chunk;
  }

  CUTLASS_DEVICE void operator()(
      const Params& args,
      int32_t comm_id,
      int32_t comm_ctas) {
    const auto& p = args.params;
    const auto& route = p.route;
    const int32_t m_tiles = ceil_div(p.gemm.m, ReadyBlockM);
    const int32_t chunks_per_tile = arrivals_per_peer(args);
    const int32_t tasks_per_m = route.world_size * chunks_per_tile;
    const int32_t tasks = m_tiles * tasks_per_m;
    const int32_t shard_width = route.local_heads * route.head_dim;
    constexpr int32_t elements_per_vector = kCommElementsPerVector;
    const int32_t vectors_per_row = shard_width / elements_per_vector;
    int32_t waited_peer = -1;

    if (args.use_bulk) {
      extern __shared__ char dynamic_smem[];
      auto* barriers = reinterpret_cast<uint64_t*>(
          dynamic_smem + kA2ALhsBulkSlots * kA2ALhsBulkStageBytes);
      const int32_t lane = static_cast<int32_t>(threadIdx.x) & 31;
      const int32_t slot = static_cast<int32_t>(threadIdx.x) >> 5;
      if (slot >= kA2ALhsBulkSlots) {
        return;
      }
      auto* stage = reinterpret_cast<CommElement*>(
          dynamic_smem + slot * kA2ALhsBulkStageBytes);
      uint64_t* barrier = barriers + slot;
      int32_t phase = 0;
      if (lane == 0) {
        cute::initialize_barrier(*barrier, 1);
        if (args.use_tensor_store) {
          cute::prefetch_tma_descriptor(&args.store_tma_full);
        }
      }
      __syncwarp();

      const int32_t task_peer_count = route.world_size;
      const int32_t bulk_tasks =
          m_tiles * task_peer_count * chunks_per_tile;
      const int32_t task_stride = comm_ctas * kA2ALhsBulkSlots;
      for (int32_t task = slot * comm_ctas + comm_id;
           task < bulk_tasks;
           task += task_stride) {
#if FUSE_ENABLE_PROFILING
        A2ALhsCommStageSample<Instrumented> sample{};
        if constexpr (Instrumented) {
          if (lane == 0) {
            sample.task_begin = detail::read_global_timer();
            sample.comm_cta = comm_id;
            sample.comm_slot = slot;
            sample.task_id = task;
            sample.copy_path = args.use_tensor_store ? 2 : 1;
          }
        }
#endif
        const int32_t m_window = args.m_window;
        const int32_t full_windows = m_tiles / m_window;
        const int32_t tasks_per_window =
            m_window * task_peer_count * chunks_per_tile;
        const int32_t full_window_tasks = full_windows * tasks_per_window;
        int32_t tile_m = 0;
        int32_t peer_slot = 0;
        int32_t row_chunk = 0;
        if (task < full_window_tasks) {
          const int32_t window = task / tasks_per_window;
          const int32_t task_in_window = task - window * tasks_per_window;
          peer_slot = task_in_window / (m_window * chunks_per_tile);
          const int32_t task_in_peer =
              task_in_window - peer_slot * m_window * chunks_per_tile;
          const int32_t m_in_window = task_in_peer / chunks_per_tile;
          row_chunk = task_in_peer - m_in_window * chunks_per_tile;
          tile_m = window * m_window + m_in_window;
        } else {
          const int32_t tail = task - full_window_tasks;
          const int32_t tail_m_tiles = m_tiles - full_windows * m_window;
          peer_slot = tail / (tail_m_tiles * chunks_per_tile);
          const int32_t task_in_peer =
              tail - peer_slot * tail_m_tiles * chunks_per_tile;
          const int32_t m_in_window = task_in_peer / chunks_per_tile;
          row_chunk = task_in_peer - m_in_window * chunks_per_tile;
          tile_m = full_windows * m_window + m_in_window;
        }
        const int32_t row_in_tile = row_chunk * args.comm_rows;
        const int32_t source_peer = route.cyclic_peer_order
            ? (logical_source_rank(p) + peer_slot) % route.world_size
            : peer_slot;
        const int32_t m_begin = tile_m * ReadyBlockM + row_in_tile;
        const int32_t copy_rows = max(
            0,
            min(min(args.comm_rows, ReadyBlockM - row_in_tile),
                p.gemm.m - m_begin));

#if FUSE_ENABLE_PROFILING
        if constexpr (Instrumented) {
          if (lane == 0) {
            sample.row_chunk = row_chunk;
            sample.copy_rows = copy_rows;
            sample.source_rank = source_peer;
          }
        }
#endif

        if (p.input_epoch != 0 && source_peer != waited_peer) {
          detail::wait_acquire_system(
              p.peer_input_ready[source_peer], p.input_epoch, lane);
          waited_peer = source_peer;
        }

#if FUSE_ENABLE_PROFILING
        if constexpr (Instrumented) {
          if (lane == 0) {
            sample.input_ready = detail::read_global_timer();
          }
        }
#endif

        if (lane == 0 && copy_rows > 0) {
          const int32_t logical_row = source_row_begin(p) + m_begin;
          const int32_t batch = logical_row / route.seq_local;
          const int32_t local_sequence =
              logical_row - batch * route.seq_local;
          const int32_t source_sequence =
              global_sequence_row(p, local_sequence);
          const auto* source = p.peer_input[source_peer] +
              (static_cast<int64_t>(batch) * route.global_seq +
               source_sequence) * shard_width;
          const int32_t copy_bytes =
              copy_rows * shard_width * sizeof(CommElement);
          cute::set_barrier_transaction_bytes(*barrier, copy_bytes);
#if FUSE_ENABLE_PROFILING
          if constexpr (Instrumented) {
            sample.g2s_issue = detail::read_global_timer();
          }
#endif
          cute::SM90_BULK_COPY_G2S::copy(
              source, barrier, stage, copy_bytes);
          cute::wait_barrier(*barrier, phase);
#if FUSE_ENABLE_PROFILING
          if constexpr (Instrumented) {
            sample.g2s_done = detail::read_global_timer();
          }
#endif
          phase ^= 1;
          cute::tma_store_fence();
#if FUSE_ENABLE_PROFILING
          if constexpr (Instrumented) {
            sample.s2g_issue = detail::read_global_timer();
          }
#endif
          if (args.use_tensor_store) {
            int32_t row = 0;
            for (; row + args.store_rows <= copy_rows;
                 row += args.store_rows) {
              cute::SM90_TMA_STORE_3D::copy(
                  &args.store_tma_full,
                  stage + static_cast<int64_t>(row) * shard_width,
                  0,
                  peer_slot * args.store_peer_groups,
                  m_begin + row);
            }
            if (row > 0) {
              cute::tma_store_arrive();
            }
            int32_t residual_ops = 0;
            for (; row < copy_rows; ++row) {
              cute::SM90_BULK_COPY_S2G::copy(
                  stage + static_cast<int64_t>(row) * shard_width,
                  p.input_staging +
                      static_cast<int64_t>(m_begin + row) * p.gemm.k +
                      peer_slot * shard_width,
                  shard_width * sizeof(CommElement));
              if (++residual_ops == 8) {
                cute::tma_store_arrive();
                residual_ops = 0;
              }
            }
            if (residual_ops != 0) {
              cute::tma_store_arrive();
            }
          } else {
            for (int32_t row = 0; row < copy_rows; ++row) {
              cute::SM90_BULK_COPY_S2G::copy(
                  stage + static_cast<int64_t>(row) * shard_width,
                  p.input_staging +
                      static_cast<int64_t>(m_begin + row) * p.gemm.k +
                      peer_slot * shard_width,
                  shard_width * sizeof(CommElement));
              if ((row & 7) == 7) {
                cute::tma_store_arrive();
              }
            }
            if ((copy_rows & 7) != 0) {
              cute::tma_store_arrive();
            }
          }
          // The ready flag is consumed by another CTA.  Waiting only for the
          // source-SMEM read would allow that consumer to observe ready before
          // the destination-global writes have completed.
          detail::tma_store_wait_all();
#if FUSE_ENABLE_PROFILING
          if constexpr (Instrumented) {
            sample.s2g_done = detail::read_global_timer();
            publish_ready_instrumented(args, tile_m, peer_slot, sample);
          } else
#endif
          {
            publish_ready(args, tile_m, peer_slot);
          }
        } else if (lane == 0) {
#if FUSE_ENABLE_PROFILING
          if constexpr (Instrumented) {
            publish_ready_instrumented(args, tile_m, peer_slot, sample);
          } else
#endif
          {
            publish_ready(args, tile_m, peer_slot);
          }
        }
        __syncwarp();
      }
      if (lane == 0) {
        cutlass::arch::ClusterBarrier::invalidate(barrier);
      }
      return;
    }

    for (int32_t task = comm_id; task < tasks; task += comm_ctas) {
#if FUSE_ENABLE_PROFILING
      A2ALhsCommStageSample<Instrumented> sample{};
      if constexpr (Instrumented) {
        if (threadIdx.x == 0) {
          sample.task_begin = detail::read_global_timer();
          sample.comm_cta = comm_id;
          sample.comm_slot = -1;
          sample.task_id = task;
          sample.copy_path = 0;
        }
      }
#endif
      const int32_t tile_m = task / tasks_per_m;
      const int32_t task_in_m = task - tile_m * tasks_per_m;
      const int32_t peer_slot = task_in_m / chunks_per_tile;
      const int32_t row_chunk = task_in_m - peer_slot * chunks_per_tile;
      const int32_t source_peer = route.cyclic_peer_order
          ? (logical_source_rank(p) + peer_slot) % route.world_size
          : peer_slot;
      const int32_t m_begin = tile_m * ReadyBlockM +
          row_chunk * args.comm_rows;
      const int32_t copy_rows = max(
          0, min(args.comm_rows, p.gemm.m - m_begin));

#if FUSE_ENABLE_PROFILING
      if constexpr (Instrumented) {
        if (threadIdx.x == 0) {
          sample.row_chunk = row_chunk;
          sample.copy_rows = copy_rows;
          sample.source_rank = source_peer;
        }
      }
#endif

      if (p.input_epoch != 0 && source_peer != waited_peer) {
        if (threadIdx.x < 32) {
          detail::wait_acquire_system(
              p.peer_input_ready[source_peer], p.input_epoch,
              static_cast<int32_t>(threadIdx.x));
        }
        __syncthreads();
        waited_peer = source_peer;
      }
#if FUSE_ENABLE_PROFILING
      if constexpr (Instrumented) {
        if (threadIdx.x == 0) {
          sample.input_ready = detail::read_global_timer();
          sample.g2s_issue = sample.input_ready;
        }
      }
#endif

      const auto* source =
          reinterpret_cast<const uint4*>(p.peer_input[source_peer]);
      auto* destination = reinterpret_cast<uint4*>(p.input_staging);
      const int32_t vector_count = copy_rows * vectors_per_row;
      for (int32_t index = static_cast<int32_t>(threadIdx.x);
           index < vector_count;
           index += static_cast<int32_t>(blockDim.x)) {
        const int32_t row = index / vectors_per_row;
        const int32_t vector_k = index - row * vectors_per_row;
        const int32_t destination_row = m_begin + row;
        const int32_t logical_row =
            source_row_begin(p) + destination_row;
        const int32_t batch = logical_row / route.seq_local;
        const int32_t local_sequence =
            logical_row - batch * route.seq_local;
        const int32_t source_sequence =
            global_sequence_row(p, local_sequence);
        const int64_t src =
            (static_cast<int64_t>(batch) * route.global_seq +
             source_sequence) *
                vectors_per_row +
            vector_k;
        const int64_t dst =
            static_cast<int64_t>(destination_row) *
                (p.gemm.k / elements_per_vector) +
            peer_slot * vectors_per_row + vector_k;
        destination[dst] = source[src];
      }
      __syncthreads();
      if (threadIdx.x == 0) {
#if FUSE_ENABLE_PROFILING
        if constexpr (Instrumented) {
          sample.g2s_done = detail::read_global_timer();
          publish_ready_instrumented(args, tile_m, peer_slot, sample);
        } else
#endif
        {
          publish_ready(args, tile_m, peer_slot);
        }
      }
      __syncthreads();
    }
  }

  CUTLASS_DEVICE void finalize(const Params& args) {
    if constexpr (Heterogeneous) {
      const auto& p = args.params;
      finalize_weighted_cp_epoch(
          p.peer_done_epoch,
          p.route.world_size,
          executor_rank(p),
          p.epoch);
    }
  }
};

}  // namespace
}  // namespace fuse
