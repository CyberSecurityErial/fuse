// SPDX-License-Identifier: BSD-3-Clause
// Private implementation; assembled by csrc/operators/ulysses_sm90.cu.
//
// Module index:
//   - QkvGqaPackCommT: ready wait, Q/K/V routing, and epoch finalization
//   - forward, OProj-backward, FP8, and weighted communication aliases
//   - role-telemetry wrapper and standalone route reference launch

namespace fuse {
namespace {

// GEMM -> A2A: consume published output tiles and route Q/K/V to peers.
template <
    class ParamsType,
    bool IsFp8,
    int32_t BlockM,
    int32_t BlockN,
    int32_t MTilesPerTask = 1,
    bool PeerInterleaved = false,
    int32_t CopyBlockN = BlockN,
    bool FinalizeAcrossRanks = true,
    bool Heterogeneous = false>
struct QkvGqaPackCommT {
  using CommElement = std::remove_cv_t<std::remove_pointer_t<
      decltype(ParamsType{}.local_output)>>;
  static constexpr int32_t kCommAlignment = 16 / sizeof(CommElement);
  static constexpr int32_t kBulkStageElements =
      kQkvBulkRows * kQkvBulkColumns;
  static constexpr int32_t kBulkStageBytes =
      kBulkStageElements * sizeof(CommElement);
  struct Arguments {
    ParamsType params{};
    cute::TmaDescriptor local_output_tma{};
    cute::TmaDescriptor peer_output_tma[kMaxWorldSize][3]{};
    bool use_tma = false;
    bool use_tma_store = false;
  };
  using Params = Arguments;
  static constexpr size_t SharedStorageBytes =
      kQkvBulkSlots * kBulkStageBytes +
      kQkvBulkSlots * sizeof(uint64_t);
  static constexpr bool kNeedsGridFinalize = FinalizeAcrossRanks;
  static constexpr bool kUniformHeads =
      std::is_same_v<ParamsType, OprojBackwardKernelParams> ||
      std::is_same_v<ParamsType, Fp8OprojBackwardKernelParams>;

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

  CUTLASS_DEVICE static int64_t destination_row(
      const ParamsType& p,
      int32_t source_row) {
    const int32_t logical_row = source_row_begin(p) + source_row;
    const int32_t batch = logical_row / p.route.seq_local;
    const int32_t local_sequence =
        logical_row - batch * p.route.seq_local;
    int32_t sequence_begin = logical_source_rank(p) * p.route.seq_local;
    if constexpr (kUniformHeads) {
      if (p.route.causal_load_balanced) {
        const int32_t chunk_rows = p.route.seq_local / 2;
        const int32_t source_rank = logical_source_rank(p);
        const int32_t chunk = local_sequence < chunk_rows
            ? source_rank
            : 2 * p.route.world_size - source_rank - 1;
        const int32_t row_in_chunk = local_sequence < chunk_rows
            ? local_sequence
            : local_sequence - chunk_rows;
        return static_cast<int64_t>(batch) * p.route.global_seq +
            chunk * chunk_rows + row_in_chunk;
      }
    }
    if constexpr (Heterogeneous) {
      if (p.weighted_partition) {
        sequence_begin = p.global_sequence_begin;
      }
    }
    return static_cast<int64_t>(batch) * p.route.global_seq +
        sequence_begin + local_sequence;
  }

  static cudaError_t initialize(Arguments& args) {
    const auto& p = args.params;
    args.use_tma = p.route.head_dim == kQkvBulkColumns &&
        d_row_stride(p.gemm) >= p.gemm.n &&
        d_row_stride(p.gemm) % kCommAlignment == 0;
    if (!args.use_tma) {
      return cudaSuccess;
    }
    const uint64_t global_dims[2] = {
        static_cast<uint64_t>(p.gemm.n),
        static_cast<uint64_t>(p.gemm.m)};
    const uint64_t global_strides[1] = {
        static_cast<uint64_t>(d_row_stride(p.gemm)) * sizeof(CommElement)};
    constexpr uint32_t box_dims[2] = {
        kQkvBulkColumns, kQkvBulkRows};
    constexpr uint32_t element_strides[2] = {1, 1};
    CUresult result =
        CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &args.local_output_tma,
            IsFp8 ? CU_TENSOR_MAP_DATA_TYPE_UINT8
                  : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,
            const_cast<CommElement*>(p.local_output),
            global_dims,
            global_strides,
            box_dims,
            element_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (result != CUDA_SUCCESS) {
      return cudaErrorInvalidValue;
    }

    args.use_tma_store = p.gemm.m % kQkvBulkRows == 0 &&
        p.route.seq_local % kQkvBulkRows == 0 &&
        source_row_begin(p) % kQkvBulkRows == 0;
    if (!args.use_tma_store) {
      return cudaSuccess;
    }
    const int32_t q_local_width =
        p.route.q_heads / p.route.world_size * p.route.head_dim;
    const int32_t kv_local_width =
        p.route.kv_heads / p.route.world_size * p.route.head_dim;
    constexpr bool uniform_heads = kUniformHeads;
    const uint64_t output_rows =
        static_cast<uint64_t>(p.route.batch) * p.route.global_seq;
    for (int32_t peer = 0; peer < p.route.world_size; ++peer) {
      const int32_t descriptor_count =
          uniform_heads || p.route.defer_v_a2a ? 1 : 3;
      for (int32_t segment = 0; segment < descriptor_count; ++segment) {
        const int32_t segment_width = uniform_heads
            ? q_local_width
            : p.route.defer_v_a2a
            ? q_local_width + kv_local_width
            : (segment == 0 ? q_local_width : kv_local_width);
        const int64_t segment_offset =
            uniform_heads || p.route.defer_v_a2a || segment == 0
            ? 0
            : static_cast<int64_t>(output_rows) * q_local_width +
                (segment == 2
                     ? static_cast<int64_t>(output_rows) * kv_local_width
                     : 0);
        const uint64_t destination_dims[2] = {
            static_cast<uint64_t>(segment_width), output_rows};
        const uint64_t destination_strides[1] = {
            static_cast<uint64_t>(segment_width) * sizeof(CommElement)};
        result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &args.peer_output_tma[peer][segment],
            IsFp8 ? CU_TENSOR_MAP_DATA_TYPE_UINT8
                  : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,
            const_cast<CommElement*>(p.peer_output[peer] + segment_offset),
            destination_dims,
            destination_strides,
            box_dims,
            element_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (result != CUDA_SUCCESS) {
          return cudaErrorInvalidValue;
        }
      }
    }
    return cudaSuccess;
  }

  static bool can_implement(const Arguments& args) {
    const auto& p = args.params;
    bool pointers = supported_route_base(p.route) && p.lhs && p.rhs_nt &&
        p.local_output && p.ready;
    for (int32_t peer = 0; peer < p.route.world_size && pointers; ++peer) {
      pointers = p.peer_output[peer] && p.peer_route_done_epoch[peer];
    }
    const int32_t packed_heads = p.route.q_heads +
        (kUniformHeads ? 0 : 2 * p.route.kv_heads);
    const bool problem_supported = IsFp8
        ? (p.gemm.transpose_b ? supported_fp8_transpose_b_problem(p.gemm)
                              : supported_fp8_problem(p.gemm))
        : (p.gemm.transpose_b ? supported_transpose_b_problem(p.gemm)
                              : supported_problem(p.gemm));
    bool rows_supported =
        p.gemm.m == p.route.batch * p.route.seq_local;
    bool sequence_supported =
        p.route.global_seq == p.route.seq_local * p.route.world_size;
    if constexpr (Heterogeneous) {
      const bool rank_supported = !IsFp8 && !p.route.defer_v_a2a &&
          p.executor_rank >= 0 &&
          p.executor_rank < p.route.world_size &&
          p.logical_source_rank >= 0 &&
          p.logical_source_rank < p.route.world_size;
      if (p.weighted_partition) {
        rows_supported = rank_supported &&
            p.gemm.m == p.route.batch * p.route.seq_local &&
            p.source_row_begin == 0 && p.global_sequence_begin >= 0;
        sequence_supported =
            p.global_sequence_begin + p.route.seq_local <=
                p.route.global_seq;
      } else {
        rows_supported = rank_supported && p.source_row_begin >= 0 &&
            p.source_row_begin + p.gemm.m <=
                p.route.batch * p.route.seq_local;
      }
    }
    const bool route_supported = kUniformHeads
        ? p.route.kind == RouteKind::kHeadToSequence &&
              p.route.direction == RouteDirection::kForward
        : p.route.kind == RouteKind::kQkvGqaPack &&
              p.route.direction == RouteDirection::kForward;
    return pointers && p.epoch > 0 && problem_supported && route_supported &&
        p.gemm.l == 1 && p.route.q_heads > 0 &&
        (kUniformHeads ||
         (p.route.kv_heads > 0 &&
          p.route.q_heads % p.route.kv_heads == 0)) &&
        p.route.q_heads % p.route.world_size == 0 &&
        (kUniformHeads || p.route.kv_heads % p.route.world_size == 0) &&
        p.route.head_dim > 0 &&
        p.route.head_dim % kCommAlignment == 0 &&
        rows_supported &&
        sequence_supported &&
        p.gemm.n == packed_heads * p.route.head_dim &&
        p.gemm.n % kCommAlignment == 0 &&
        (!kUniformHeads ||
         (!p.route.defer_v_a2a && !p.route.qkv_peer_interleaved)) &&
        p.route.qkv_peer_interleaved == PeerInterleaved &&
        (!PeerInterleaved || p.gemm.raster != GemmRaster::kAlongN);
  }

  static Params to_underlying_arguments(const Arguments& args) { return args; }

  CUTLASS_DEVICE static int32_t source_feature(
      const UlyssesRoute& route,
      int32_t destination_rank,
      int32_t segment,
      int32_t local_head,
      int32_t head_offset) {
    const int32_t q_width = route.q_heads * route.head_dim;
    const int32_t kv_width = route.kv_heads * route.head_dim;
    const int32_t segment_base =
        segment == 0 ? 0 : q_width + (segment == 2 ? kv_width : 0);
    const int32_t global_heads =
        segment == 0 ? route.q_heads : route.kv_heads;
    const int32_t local_heads = global_heads / route.world_size;
    const int32_t physical_head =
        PeerInterleaved && segment < 2
        ? local_head * route.world_size + destination_rank
        : destination_rank * local_heads + local_head;
    return segment_base + physical_head * route.head_dim + head_offset;
  }

  CUTLASS_DEVICE void run(
      const Params& args,
      int32_t comm_id,
      int32_t comm_ctas,
      bool wait_for_gemm) {
    const auto& p = args.params;
    const int32_t m_tiles = ceil_div(p.gemm.m, BlockM);
    const int32_t m_groups = ceil_div(m_tiles, MTilesPerTask);
    const int32_t output_n_tiles = ceil_div(p.gemm.n, BlockN);
    const int32_t q_local_heads = p.route.q_heads / p.route.world_size;
    const int32_t kv_local_heads = p.route.kv_heads / p.route.world_size;
    const int32_t route_heads = kUniformHeads
        ? q_local_heads
        : q_local_heads +
            (p.route.defer_v_a2a ? 1 : 2) * kv_local_heads;
    const int32_t chunks_per_head = ceil_div(p.route.head_dim, CopyBlockN);
    const int32_t head_chunks = route_heads * chunks_per_head;
    const int32_t tasks =
        p.route.world_size * m_groups * head_chunks;
    const int64_t source_row_vectors =
        d_row_stride(p.gemm) / kCommAlignment;
    const int32_t q_local_width = q_local_heads * p.route.head_dim;
    const int32_t kv_local_width = kv_local_heads * p.route.head_dim;
    const int64_t segment_rows =
        static_cast<int64_t>(p.route.batch) * p.route.global_seq;

    if (args.use_tma) {
      static_assert(CopyBlockN >= kQkvBulkColumns);
      extern __shared__ char dynamic_smem[];
      auto* stages = reinterpret_cast<CommElement*>(dynamic_smem);
      auto* barriers = reinterpret_cast<uint64_t*>(
          dynamic_smem + kQkvBulkSlots * kBulkStageBytes);
      const int32_t lane = static_cast<int32_t>(threadIdx.x) & 31;
      const int32_t slot = static_cast<int32_t>(threadIdx.x) >> 5;
      if (slot < kQkvBulkSlots) {
        CommElement* stage = stages +
            static_cast<int64_t>(slot) * kBulkStageElements;
        uint64_t* barrier = barriers + slot;
        int32_t phase = 0;
        if (lane == 0) {
          cute::initialize_barrier(*barrier, 1);
        }
        __syncwarp();

        const int32_t m_chunks = ceil_div(p.gemm.m, kQkvBulkRows);
        const int32_t bulk_tasks =
            p.route.world_size * m_chunks * head_chunks;
        const int32_t task_stride = comm_ctas * kQkvBulkSlots;
        for (int32_t work = slot * comm_ctas + comm_id;
             work < bulk_tasks;
             work += task_stride) {
          const bool along_n = p.gemm.raster == GemmRaster::kAlongN;
          const int32_t destination_rank = work % p.route.world_size;
          const int32_t rank_work = work / p.route.world_size;
          const int32_t m_chunk = along_n
              ? rank_work / head_chunks
              : rank_work % m_chunks;
          const int32_t head_chunk = along_n
              ? rank_work % head_chunks
              : rank_work / m_chunks;
          const int32_t head_slot = head_chunk / chunks_per_head;
          const int32_t chunk = head_chunk - head_slot * chunks_per_head;
          const int32_t segment = head_slot < q_local_heads
              ? 0
              : (head_slot < q_local_heads + kv_local_heads ? 1 : 2);
          const int32_t local_head = segment == 0
              ? head_slot
              : head_slot - q_local_heads -
                  (segment == 2 ? kv_local_heads : 0);
          const int32_t head_offset = chunk * CopyBlockN;
          const int32_t copy_n =
              min(CopyBlockN, p.route.head_dim - head_offset);
          const int32_t physical_feature = source_feature(
              p.route,
              destination_rank,
              segment,
              local_head,
              head_offset);
          const int32_t m_begin = m_chunk * kQkvBulkRows;
          const int32_t copy_m =
              min(kQkvBulkRows, p.gemm.m - m_begin);

          if (wait_for_gemm) {
            const int32_t producer_m = m_begin / BlockM;
            const int32_t first_producer_n = physical_feature / BlockN;
            const int32_t last_producer_n =
                (physical_feature + copy_n - 1) / BlockN;
            for (int32_t producer_n = first_producer_n;
                 producer_n <= last_producer_n;
                 ++producer_n) {
              detail::wait_acquire_system(
                  p.ready +
                      (producer_m * output_n_tiles + producer_n) *
                          kReadyFlagStride,
                  p.epoch,
                  lane);
            }
          }
          __syncwarp();

          if (lane == 0) {
            cute::set_barrier_transaction_bytes(
                *barrier, kBulkStageBytes);
            cute::SM90_TMA_LOAD_2D::copy(
                &args.local_output_tma,
                barrier,
                0x12f0000000000000ull,
                stage,
                physical_feature,
                m_begin);
            cute::wait_barrier(*barrier, phase);
            phase ^= 1;
            cute::tma_store_fence();

            auto* destination = p.peer_output[destination_rank];
            const int32_t segment_width =
                segment == 0 ? q_local_width : kv_local_width;
            const int32_t local_segment_base =
                segment == 0 ? 0 : q_local_width +
                    (segment == 2 ? kv_local_width : 0);
            const int64_t segment_offset = segment == 0
                ? 0
                : segment_rows * q_local_width +
                    (segment == 2 ? segment_rows * kv_local_width : 0);
            const int32_t local_feature =
                local_segment_base + local_head * p.route.head_dim +
                head_offset;
            if (args.use_tma_store) {
              const int64_t destination_row_value =
                  destination_row(p, m_begin);
              const int32_t destination_feature = p.route.defer_v_a2a
                  ? local_feature
                  : local_feature - local_segment_base;
              const int32_t descriptor =
                  p.route.defer_v_a2a ? 0 : segment;
              cute::SM90_TMA_STORE_2D::copy(
                  &args.peer_output_tma[destination_rank][descriptor],
                  stage,
                  destination_feature,
                  destination_row_value);
              cute::tma_store_arrive();
            } else {
              for (int32_t row = 0; row < copy_m; ++row) {
                const int32_t source_row = m_begin + row;
                const int64_t destination_row_value =
                    destination_row(p, source_row);
                const int64_t dst = p.route.defer_v_a2a
                    ? destination_row_value *
                          (q_local_width + kv_local_width) +
                        local_feature
                    : segment_offset +
                        destination_row_value * segment_width +
                        local_feature - local_segment_base;
                cute::SM90_BULK_COPY_S2G::copy(
                    stage + static_cast<int64_t>(row) * kQkvBulkColumns,
                    destination + dst,
                    kQkvBulkColumns * sizeof(CommElement));
                if ((row & 7) == 7) {
                  cute::tma_store_arrive();
                }
              }
              if ((copy_m & 7) != 0) {
                cute::tma_store_arrive();
              }
            }
            cute::tma_store_wait<0>();
          }
          __syncwarp();
        }
        if (lane == 0) {
          // Per-task `.read` waits above make each stage reusable.  Before
          // this CTA publishes completion, also wait for every destination
          // global write issued by this lane to finish.
          detail::tma_store_wait_all();
          cutlass::arch::ClusterBarrier::invalidate(barrier);
        }
      }
      __syncthreads();
      if (wait_for_gemm && p.route.defer_v_a2a && p.completion_epoch &&
          comm_id == 0) {
        wait_and_publish_completion(args);
      }
      return;
    }

    for (int32_t work = comm_id; work < tasks; work += comm_ctas) {
      const bool along_n = p.gemm.raster == GemmRaster::kAlongN;
      const int32_t destination_rank = work % p.route.world_size;
      const int32_t rank_work = work / p.route.world_size;
      const int32_t m_group = along_n
          ? rank_work / head_chunks
          : rank_work % m_groups;
      const int32_t head_chunk = along_n
          ? rank_work % head_chunks
          : rank_work / m_groups;
      const int32_t head_slot = head_chunk / chunks_per_head;
      const int32_t chunk = head_chunk - head_slot * chunks_per_head;
      const int32_t segment = head_slot < q_local_heads
          ? 0
          : (head_slot < q_local_heads + kv_local_heads ? 1 : 2);
      const int32_t local_head = segment == 0
          ? head_slot
          : head_slot - q_local_heads -
              (segment == 2 ? kv_local_heads : 0);
      const int32_t head_offset = chunk * CopyBlockN;
      const int32_t copy_n =
          min(CopyBlockN, p.route.head_dim - head_offset);
      const int32_t physical_feature = source_feature(
          p.route, destination_rank, segment, local_head, head_offset);
      if (wait_for_gemm) {
        const int32_t lane = static_cast<int32_t>(threadIdx.x) & 31;
        const int32_t warp = static_cast<int32_t>(threadIdx.x) >> 5;
        const int32_t warps = static_cast<int32_t>(blockDim.x) >> 5;
        const int32_t first_producer_n = physical_feature / BlockN;
        const int32_t last_producer_n =
            (physical_feature + copy_n - 1) / BlockN;
        const int32_t producer_n_tiles =
            last_producer_n - first_producer_n + 1;
        const int32_t waits = MTilesPerTask * producer_n_tiles;
        for (int32_t item = warp; item < waits; item += warps) {
          const int32_t local_m_tile = item / producer_n_tiles;
          const int32_t tile_m =
              m_group * MTilesPerTask + local_m_tile;
          if (tile_m < m_tiles) {
            const int32_t producer_n =
                first_producer_n + item % producer_n_tiles;
            const int32_t signal =
                tile_m * output_n_tiles + producer_n;
            detail::wait_acquire_system(
                p.ready + signal * kReadyFlagStride,
                p.epoch,
                lane);
          }
        }
      }
      __syncthreads();

      const int32_t m_begin = m_group * MTilesPerTask * BlockM;
      const int32_t copy_m =
          min(BlockM * MTilesPerTask, p.gemm.m - m_begin);
      const int32_t vectors_per_row = copy_n / kCommAlignment;
      const int32_t vector_count = copy_m * vectors_per_row;
      const auto* source = reinterpret_cast<const uint4*>(
          p.local_output);
      auto* destination =
          reinterpret_cast<uint4*>(p.peer_output[destination_rank]);
      const int32_t segment_width =
          segment == 0 ? q_local_width : kv_local_width;
      const int32_t local_segment_base =
          segment == 0 ? 0 : q_local_width +
              (segment == 2 ? kv_local_width : 0);
      const int64_t segment_offset = segment == 0
          ? 0
          : segment_rows * q_local_width +
              (segment == 2 ? segment_rows * kv_local_width : 0);
      const int32_t local_feature =
          local_segment_base + local_head * p.route.head_dim + head_offset;
      for (int32_t index = threadIdx.x; index < vector_count; index += blockDim.x) {
        const int32_t row = index / vectors_per_row;
        const int32_t vector_n = index - row * vectors_per_row;
        const int32_t source_row = m_begin + row;
        const int64_t src =
            static_cast<int64_t>(source_row) * source_row_vectors +
            physical_feature / kCommAlignment + vector_n;
        const int64_t destination_row_value =
            destination_row(p, source_row);
        const int64_t dst = p.route.defer_v_a2a
            ? (destination_row_value *
                   (q_local_width + kv_local_width) +
               local_feature) /
                    kCommAlignment +
                vector_n
            : (segment_offset +
               destination_row_value * segment_width +
               local_feature - local_segment_base) /
                    kCommAlignment +
                vector_n;
        destination[dst] = source[src];
      }
    }

    // The downstream A2A->GEMM consumes V but not the routed Q/K buffer. A
    // single communication CTA waits only for V's epilogue tiles and publishes
    // their transitive completion after the Q/K route has drained.
    if (wait_for_gemm && p.route.defer_v_a2a && p.completion_epoch &&
        comm_id == 0) {
      wait_and_publish_completion(args);
    }

  }

  CUTLASS_DEVICE static void wait_and_publish_completion(const Params& args) {
    const auto& p = args.params;
    if (!p.completion_epoch) {
      return;
    }
    const int32_t m_tiles = ceil_div(p.gemm.m, BlockM);
    const int32_t output_n_tiles = ceil_div(p.gemm.n, BlockN);
    const int32_t v_begin =
        (p.route.q_heads + p.route.kv_heads) * p.route.head_dim;
    const int32_t v_first_tile = v_begin / BlockN;
    const int32_t v_n_tiles = output_n_tiles - v_first_tile;
    const int32_t v_signals = m_tiles * v_n_tiles;
    const int32_t lane = static_cast<int32_t>(threadIdx.x) & 31;
    const int32_t warp = static_cast<int32_t>(threadIdx.x) >> 5;
    const int32_t warps = static_cast<int32_t>(blockDim.x) >> 5;
    for (int32_t item = warp; item < v_signals; item += warps) {
      const int32_t tile_m = item / v_n_tiles;
      const int32_t tile_n = v_first_tile + item - tile_m * v_n_tiles;
      const int32_t signal = tile_m * output_n_tiles + tile_n;
      detail::wait_acquire_system(
          p.ready + signal * kReadyFlagStride, p.epoch, lane);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      detail::store_release_system(p.completion_epoch, p.epoch);
    }
  }

  CUTLASS_DEVICE void operator()(
      const Params& args,
      int32_t comm_id,
      int32_t comm_ctas) {
    run(args, comm_id, comm_ctas, true);
  }

  // All local compute and communication CTAs have crossed the cooperative
  // grid barrier before this hook runs, so every remote store issued by this
  // source rank is complete. Publish one source-complete epoch into every
  // destination-owned array, then hold this rank's kernel open until all
  // sources have completed its routed output.
  CUTLASS_DEVICE void finalize(const Params& args) {
    if constexpr (Heterogeneous) {
      const auto& p = args.params;
      finalize_weighted_cp_epoch(
          p.peer_route_done_epoch,
          p.route.world_size,
          executor_rank(p),
          p.epoch);
      return;
    }
    if (blockIdx.x != 0 || threadIdx.x >= 32) {
      return;
    }
    const auto& p = args.params;
    const int32_t lane = static_cast<int32_t>(threadIdx.x);
    if (lane == 0) {
      detail::fence_system();
    }
    __syncwarp();
    if (lane < p.route.world_size) {
      detail::store_release_system(
          p.peer_route_done_epoch[lane] +
              p.route.rank * kReadyFlagStride,
          p.epoch);
    }
    __syncwarp();
    const uint32_t* local_sources_done =
        p.peer_route_done_epoch[p.route.rank];
    if (lane < p.route.world_size) {
      while (detail::load_acquire_system(
                 local_sources_done + lane * kReadyFlagStride) <
             p.epoch) {
        __nanosleep(64);
      }
    }
    __syncwarp();
  }

#if FUSE_ENABLE_PROFILING
  // Diagnostic copy of finalize() with phase boundaries.  Production Params,
  // control flow, and generated kernel remain untouched when profiling is off.
  CUTLASS_DEVICE void finalize_profile(
      const Params& args,
      A2AGemmCtaTimeline* event) {
    if (blockIdx.x != 0 || threadIdx.x >= 32 || event == nullptr) {
      return;
    }
    const auto& p = args.params;
    const int32_t lane = static_cast<int32_t>(threadIdx.x);
    if (lane == 0) {
      detail::fence_system();
      event->fence_done = detail::read_global_timer();
    }
    __syncwarp();
    if (lane < p.route.world_size) {
      detail::store_release_system(
          p.peer_route_done_epoch[lane] +
              p.route.rank * kReadyFlagStride,
          p.epoch);
    }
    __syncwarp();
    if (lane == 0) {
      event->publish_done = detail::read_global_timer();
    }
    const uint32_t* local_sources_done =
        p.peer_route_done_epoch[p.route.rank];
    if (lane < p.route.world_size) {
      while (detail::load_acquire_system(
                 local_sources_done + lane * kReadyFlagStride) <
             p.epoch) {
        __nanosleep(64);
      }
      event->source_ready[lane] = detail::read_global_timer();
    }
    __syncwarp();
  }
#endif
};

using QkvGqaPackCommWide = QkvGqaPackCommT<
    GemmA2AParams,
    false,
    static_cast<int32_t>(cute::size<0>(ProjectionTileShape{})),
    static_cast<int32_t>(cute::size<1>(ProjectionTileShape{}))>;
using QkvGqaPackCommN64 = QkvGqaPackCommT<
    GemmA2AParams,
    false,
    static_cast<int32_t>(cute::size<0>(N64TileShape{})),
    static_cast<int32_t>(cute::size<1>(N64TileShape{})),
    4,
    false,
    kQkvBulkColumns>;
using QkvGqaPackCommN160 = QkvGqaPackCommT<
    GemmA2AParams,
    false,
    static_cast<int32_t>(cute::size<0>(N160TileShape{})),
    static_cast<int32_t>(cute::size<1>(N160TileShape{})),
    4>;
using QkvGqaPackCommN192 = QkvGqaPackCommT<
    GemmA2AParams,
    false,
    static_cast<int32_t>(cute::size<0>(N192TileShape{})),
    static_cast<int32_t>(cute::size<1>(N192TileShape{})),
    4>;
using QkvGqaPackCommN320 = QkvGqaPackCommT<
    GemmA2AParams,
    false,
    static_cast<int32_t>(cute::size<0>(WideN320TileShape{})),
    static_cast<int32_t>(cute::size<1>(WideN320TileShape{}))>;
using QkvGqaPackCommSmall = QkvGqaPackCommT<
    GemmA2AParams,
    false,
    static_cast<int32_t>(cute::size<0>(TileShape{})),
    static_cast<int32_t>(cute::size<1>(TileShape{})),
    4>;
using QkvGqaPackCommSmallInterleaved = QkvGqaPackCommT<
    GemmA2AParams,
    false,
    static_cast<int32_t>(cute::size<0>(TileShape{})),
    static_cast<int32_t>(cute::size<1>(TileShape{})),
    4,
    true>;
using Fp8QkvGqaPackComm = QkvGqaPackCommT<
    Fp8GemmA2AParams,
    true,
    static_cast<int32_t>(cute::size<0>(Fp8TileShape{})),
    static_cast<int32_t>(cute::size<1>(Fp8TileShape{}))>;
using Fp8QkvGqaPackCommN64 = QkvGqaPackCommT<
    Fp8GemmA2AParams,
    true,
    static_cast<int32_t>(cute::size<0>(Fp8N64TileShape{})),
    static_cast<int32_t>(cute::size<1>(Fp8N64TileShape{})),
    4,
    false,
    kQkvBulkColumns>;
using Fp8QkvGqaPackCommWide = QkvGqaPackCommT<
    Fp8GemmA2AParams,
    true,
    static_cast<int32_t>(cute::size<0>(Fp8WideTileShape{})),
    static_cast<int32_t>(cute::size<1>(Fp8WideTileShape{})),
    1,
    false,
    kQkvBulkColumns>;
template <
    int32_t BlockM,
    int32_t BlockN,
    int32_t MTilesPerTask,
    int32_t CopyBlockN>
using OprojBackwardHeadCommT = QkvGqaPackCommT<
    OprojBackwardKernelParams,
    false,
    BlockM,
    BlockN,
    MTilesPerTask,
    false,
    CopyBlockN,
    true,
    false>;
using OprojBackwardHeadCommN64 = OprojBackwardHeadCommT<128, 64, 4, 128>;
using OprojBackwardHeadCommN128 = OprojBackwardHeadCommT<128, 128, 4, 128>;
using OprojBackwardHeadCommN160 = OprojBackwardHeadCommT<128, 160, 4, 160>;
using OprojBackwardHeadCommN192 = OprojBackwardHeadCommT<128, 192, 4, 192>;
using OprojBackwardHeadCommN256 = OprojBackwardHeadCommT<128, 256, 1, 256>;
using Fp8OprojBackwardHeadComm = QkvGqaPackCommT<
    Fp8OprojBackwardKernelParams,
    true,
    128,
    128,
    4,
    false,
    128,
    true,
    false>;

template <bool Finalize>
using WeightedQkvGqaPackCommN64 = QkvGqaPackCommT<
    WeightedGemmA2AKernelParams,
    false,
    128,
    64,
    4,
    false,
    kQkvBulkColumns,
    Finalize,
    true>;
template <bool Finalize>
using WeightedQkvGqaPackCommN128 = QkvGqaPackCommT<
    WeightedGemmA2AKernelParams,
    false,
    128,
    128,
    4,
    false,
    128,
    Finalize,
    true>;
template <bool Finalize>
using WeightedQkvGqaPackCommN128Interleaved = QkvGqaPackCommT<
    WeightedGemmA2AKernelParams,
    false,
    128,
    128,
    4,
    true,
    128,
    Finalize,
    true>;
template <bool Finalize>
using WeightedQkvGqaPackCommN160 = QkvGqaPackCommT<
    WeightedGemmA2AKernelParams,
    false,
    128,
    160,
    4,
    false,
    160,
    Finalize,
    true>;
template <bool Finalize>
using WeightedQkvGqaPackCommN192 = QkvGqaPackCommT<
    WeightedGemmA2AKernelParams,
    false,
    128,
    192,
    4,
    false,
    192,
    Finalize,
    true>;
template <bool Finalize>
using WeightedQkvGqaPackCommN256 = QkvGqaPackCommT<
    WeightedGemmA2AKernelParams,
    false,
    128,
    256,
    1,
    false,
    256,
    Finalize,
    true>;
template <bool Finalize>
using WeightedQkvGqaPackCommN320 = QkvGqaPackCommT<
    WeightedGemmA2AKernelParams,
    false,
    128,
    320,
    1,
    false,
    320,
    Finalize,
    true>;

using QkvGemmA2AKernelWide =
    detail::MonolithicGemm<ProjectionOutputGemm, QkvGqaPackCommWide>;
using QkvGemmA2AKernelN64 =
    detail::MonolithicGemm<N64OutputGemm, QkvGqaPackCommN64>;
using QkvGemmA2AKernelN160 =
    detail::MonolithicGemm<N160OutputGemm, QkvGqaPackCommN160>;
using QkvGemmA2AKernelN192 =
    detail::MonolithicGemm<N192OutputGemm, QkvGqaPackCommN192>;
using QkvGemmA2AKernelN320 =
    detail::MonolithicGemm<WideN320OutputGemm, QkvGqaPackCommN320>;
using QkvGemmA2AKernelSmall =
    detail::MonolithicGemm<OutputGemm, QkvGqaPackCommSmall>;
using Fp8GemmA2AKernel =
    detail::MonolithicGemm<Fp8OutputGemm, Fp8QkvGqaPackComm>;
using Fp8CooperativeGemmA2AKernel = detail::MonolithicGemm<
    Fp8CooperativeOutputGemm, Fp8QkvGqaPackComm>;
using Fp8N64GemmA2AKernel =
    detail::MonolithicGemm<Fp8N64OutputGemm, Fp8QkvGqaPackCommN64>;
using Fp8WideGemmA2AKernel =
    detail::MonolithicGemm<Fp8WideOutputGemm, Fp8QkvGqaPackCommWide>;
using Fp8OprojBackwardDataKernel = detail::MonolithicGemm<
    Fp8BackwardSignalingGemm, Fp8OprojBackwardHeadComm>;
template <class Gemm, class Comm>
using OprojBackwardDataKernelT = detail::MonolithicGemm<Gemm, Comm>;
using OprojBackwardDataKernelN64 = OprojBackwardDataKernelT<
    BackwardN64SignalingGemm, OprojBackwardHeadCommN64>;
using OprojBackwardDataKernelN128 = OprojBackwardDataKernelT<
    BackwardN128SignalingGemm, OprojBackwardHeadCommN128>;
using OprojBackwardDataKernelN160 = OprojBackwardDataKernelT<
    BackwardN160SignalingGemm, OprojBackwardHeadCommN160>;
using OprojBackwardDataKernelN192 = OprojBackwardDataKernelT<
    BackwardN192SignalingGemm, OprojBackwardHeadCommN192>;
using OprojBackwardDataKernelN256 = OprojBackwardDataKernelT<
    BackwardProjectionOutputGemm, OprojBackwardHeadCommN256>;

#if FUSE_ENABLE_PROFILING
// Diagnostic-only GEMM->A2A wrapper. Unlike the generic outer telemetry
// wrapper, this records the local role completion before the cooperative grid
// barrier, which separates GEMM/route work from the cross-rank finalize tail.
template <class GemmKernel, class CommOp>
struct GemmA2ARoleTelemetryKernel
    : detail::MonolithicGemm<GemmKernel, CommOp> {
  using BaseKernel = detail::MonolithicGemm<GemmKernel, CommOp>;
  using ArchTag = typename BaseKernel::ArchTag;
  using ClusterShape = typename BaseKernel::ClusterShape;
  using SharedStorage = typename BaseKernel::SharedStorage;
  static constexpr int MaxThreadsPerBlock = BaseKernel::MaxThreadsPerBlock;
  static constexpr int MinBlocksPerMultiprocessor =
      BaseKernel::MinBlocksPerMultiprocessor;

  struct Arguments : BaseKernel::Arguments {
    A2AGemmCtaTimeline* timeline = nullptr;
    int32_t timeline_capacity = 0;
  };

  struct Params : BaseKernel::Params {
    A2AGemmCtaTimeline* timeline = nullptr;
    int32_t timeline_capacity = 0;
  };

  static bool can_implement(const Arguments& args) {
    return args.timeline != nullptr && args.timeline_capacity > 0 &&
        BaseKernel::can_implement(
            static_cast<const typename BaseKernel::Arguments&>(args));
  }

  static size_t get_workspace_size(const Arguments& args) {
    return BaseKernel::get_workspace_size(
        static_cast<const typename BaseKernel::Arguments&>(args));
  }

  static cutlass::Status initialize_workspace(
      const Arguments& args,
      void* workspace,
      cudaStream_t stream) {
    return BaseKernel::initialize_workspace(
        static_cast<const typename BaseKernel::Arguments&>(args),
        workspace,
        stream);
  }

  static Params to_underlying_arguments(const Arguments& args, void* workspace) {
    Params params{};
    static_cast<typename BaseKernel::Params&>(params) =
        BaseKernel::to_underlying_arguments(
            static_cast<const typename BaseKernel::Arguments&>(args),
            workspace);
    params.timeline = args.timeline;
    params.timeline_capacity = args.timeline_capacity;
    return params;
  }

  static dim3 get_grid_shape(const Params& params) {
    return BaseKernel::get_grid_shape(
        static_cast<const typename BaseKernel::Params&>(params));
  }

  static dim3 get_block_shape() { return BaseKernel::get_block_shape(); }

  CUTLASS_DEVICE void operator()(const Params& params, char* smem) {
    const int32_t cta = static_cast<int32_t>(blockIdx.x);
    if (threadIdx.x == 0 && cta < params.timeline_capacity) {
      params.timeline[cta].start = detail::read_global_timer();
    }

    const bool is_comm = cta < params.num_comm_ctas;
    if (is_comm) {
      CommOp{}(params.comm, cta, params.num_comm_ctas);
    } else {
      GemmKernel{}(params.gemm, smem);
    }

    __syncthreads();
    if (threadIdx.x == 0 && cta < params.timeline_capacity) {
      params.timeline[cta].role_done = detail::read_global_timer();
    }
    __syncthreads();

    if constexpr (detail::NeedsGridFinalize<CommOp>::value) {
      cooperative_groups::this_grid().sync();
      A2AGemmCtaTimeline* event =
          cta < params.timeline_capacity ? params.timeline + cta : nullptr;
      if (threadIdx.x == 0 && event != nullptr) {
        event->grid_sync_done = detail::read_global_timer();
      }
      CommOp{}.finalize_profile(params.comm, event);
    }

    __syncthreads();
    if (threadIdx.x == 0 && cta < params.timeline_capacity) {
      params.timeline[cta].end = detail::read_global_timer();
    }
  }
};
#endif

static_assert(
    sizeof(typename QkvGemmA2AKernelWide::SharedStorage) >=
        QkvGqaPackCommWide::SharedStorageBytes);
static_assert(
    sizeof(typename QkvGemmA2AKernelN64::SharedStorage) >=
        QkvGqaPackCommN64::SharedStorageBytes);
static_assert(N64OutputGemm::MaxThreadsPerBlock == 384);
static_assert(
    sizeof(typename QkvGemmA2AKernelN160::SharedStorage) >=
        QkvGqaPackCommN160::SharedStorageBytes);
static_assert(
    sizeof(typename QkvGemmA2AKernelN320::SharedStorage) >=
        QkvGqaPackCommN320::SharedStorageBytes);
static_assert(
    sizeof(typename QkvGemmA2AKernelSmall::SharedStorage) >=
        QkvGqaPackCommSmall::SharedStorageBytes);
static_assert(
    sizeof(typename Fp8GemmA2AKernel::SharedStorage) >=
        Fp8QkvGqaPackComm::SharedStorageBytes);
static_assert(
    sizeof(typename Fp8CooperativeGemmA2AKernel::SharedStorage) >=
        Fp8QkvGqaPackComm::SharedStorageBytes);
static_assert(
    sizeof(typename Fp8N64GemmA2AKernel::SharedStorage) >=
        Fp8QkvGqaPackCommN64::SharedStorageBytes);
static_assert(
    sizeof(typename Fp8WideGemmA2AKernel::SharedStorage) >=
        Fp8QkvGqaPackCommWide::SharedStorageBytes);
static_assert(
    sizeof(typename Fp8A2ALhsGemmKernel::SharedStorage) >=
        Fp8A2ALhsInputComm::SharedStorageBytes);
static_assert(
    sizeof(typename Fp8CooperativeA2ALhsGemmKernel::SharedStorage) >=
        Fp8A2ALhsInputComm::SharedStorageBytes);
static_assert(
    sizeof(typename Fp8QkvBackwardDataKernel::SharedStorage) >=
        Fp8QkvBackwardPushComm::SharedStorageBytes);
static_assert(
    sizeof(typename Fp8OprojBackwardDataKernel::SharedStorage) >=
        Fp8OprojBackwardHeadComm::SharedStorageBytes);

template <class Comm>
__global__ __launch_bounds__(384)
void a2a_lhs_input_copy_reference_kernel(
    CUTLASS_GRID_CONSTANT const typename Comm::Params params) {
  Comm{}(
      params, static_cast<int32_t>(blockIdx.x),
      static_cast<int32_t>(gridDim.x));
}

template <class Comm>
__global__ __launch_bounds__(384) void qkv_gqa_copy_reference_kernel(
    CUTLASS_GRID_CONSTANT const typename Comm::Params params) {
  Comm{}.run(
      params,
      static_cast<int32_t>(blockIdx.x),
      static_cast<int32_t>(gridDim.x),
      false);
}

template <class Comm>
cudaError_t launch_qkv_gqa_copy_reference(
    typename Comm::Arguments& args,
    int32_t comm_ctas,
    cudaStream_t stream) {
  cudaError_t status = Comm::initialize(args);
  if (status != cudaSuccess) {
    return status;
  }
  if (!Comm::can_implement(args)) {
    return cudaErrorNotSupported;
  }
  auto entry = qkv_gqa_copy_reference_kernel<Comm>;
  status = cudaFuncSetAttribute(
      entry,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(Comm::SharedStorageBytes));
  if (status != cudaSuccess) {
    return status;
  }
  entry<<<comm_ctas, 384, Comm::SharedStorageBytes, stream>>>(
      Comm::to_underlying_arguments(args));
  return cudaGetLastError();
}

}  // namespace
}  // namespace fuse
