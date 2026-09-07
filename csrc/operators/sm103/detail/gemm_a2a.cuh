// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "host_profiling.cuh"
#include "producer_consumer.cuh"

#include "fuse/arch/common.cuh"
#include "fuse/operators/primitives/gemm_a2a.h"
#include "fuse/profiling/qkv_route.cuh"

#include <cute/arch/copy_sm90.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cuda_host_adapter.hpp>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace fuse {
namespace {

// Independent communication choices: producer tile N is not the copy width.
// Keep the first implementation local to SM103; SM90 remains unchanged.
template <
    int32_t MTilesPerTask_,
    int32_t CopyBlockN_,
    bool PeerInterleaved_ = false,
    bool FinalizeAcrossRanks_ = true>
struct QkvCommConfig {
  static_assert(MTilesPerTask_ > 0 && CopyBlockN_ > 0);
  static constexpr int32_t kMTilesPerTask = MTilesPerTask_;
  static constexpr int32_t kCopyBlockN = CopyBlockN_;
  static constexpr bool kPeerInterleaved = PeerInterleaved_;
  static constexpr bool kFinalizeAcrossRanks = FinalizeAcrossRanks_;
};

// QKV projection followed by non-heterogeneous head routing. The parameter
// type describes the payload; producer geometry and communication task sizes
// are independent. Only the BF16 projection binding is implemented here.
// GEMM input precision must not select the wire format: a future block-scaled
// GEMM may still produce BF16 output. A quantized routed payload would need its
// own data/scale layout and publication contract, not an IsFp8/IsMxfp8 switch.
template <
    class ParamsType,
    int32_t BlockM,
    int32_t BlockN,
    class CommConfig = QkvCommConfig<1, BlockN>>
struct QkvGqaPackCommT {
  // Blackwell-facing names for CUTLASS's shared SM90-era copy primitives.
  // These are type aliases only; they do not select different instructions.
  using SM100_TMA_LOAD_2D = cute::SM90_TMA_LOAD_2D;
  using SM100_TMA_STORE_2D = cute::SM90_TMA_STORE_2D;
  using SM100_BULK_COPY_S2G = cute::SM90_BULK_COPY_S2G;
  using CommElement = std::remove_cv_t<std::remove_pointer_t<
      decltype(ParamsType{}.local_output)>>;
  static_assert(std::is_same_v<ParamsType, GemmA2AParams>,
                "SM103 currently implements BF16 QKV projection parameters.");
  static_assert(std::is_same_v<CommElement, Bf16>);
  static constexpr int32_t kBlockM = BlockM;
  static constexpr int32_t kBlockN = BlockN;
  static constexpr int32_t MTilesPerTask = CommConfig::kMTilesPerTask;
  static constexpr int32_t CopyBlockN = CommConfig::kCopyBlockN;
  static constexpr bool PeerInterleaved = CommConfig::kPeerInterleaved;
  static constexpr bool FinalizeAcrossRanks = CommConfig::kFinalizeAcrossRanks;
  static_assert(BlockM == 128 && BlockN > 0 && BlockN % 8 == 0);
  // TMA transfer geometry: descriptor box, stage bytes and task traversal must
  // agree. It is independent of the GEMM N tile and vector task grouping.
  static constexpr int32_t kQkvBulkRows = 64;
  static constexpr int32_t kQkvBulkColumns = 128;
  // Blackwell's physical GEMM CTA has eight warps, not Hopper's twelve.
  static constexpr int32_t kQkvBulkSlots = 8;
  static constexpr int32_t kCommAlignment = 16 / sizeof(CommElement);
  static constexpr CUtensorMapDataType kTmaDataType = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  static_assert(CopyBlockN >= kQkvBulkColumns &&
                CopyBlockN % kCommAlignment == 0);
  static constexpr int32_t kBulkStageElements = kQkvBulkRows * kQkvBulkColumns;
  static constexpr int32_t kBulkStageBytes = kBulkStageElements * sizeof(CommElement);
  static constexpr int32_t kMinThreads = kQkvBulkSlots * 32;
  static constexpr size_t SharedStorageBytes =
      kQkvBulkSlots * kBulkStageBytes + kQkvBulkSlots * sizeof(uint64_t);
  static constexpr bool kNeedsGridFinalize = FinalizeAcrossRanks;
  using ReadyTile = detail::PublishedTile<BlockM, BlockN>;
  using CopyOrder = detail::ConsumerTileOrder<ReadyTile, kQkvBulkRows, kQkvBulkColumns>;
  using SchedulerParams = cutlass::gemm::kernel::detail::StaticPersistentTileScheduler100::Params;

  static SchedulerParams producer_schedule(const ParamsType& p) {
    SchedulerParams result{};
    result.initialize(dim3(ceil_div(p.gemm.m, BlockM), ceil_div(p.gemm.n, BlockN), 1),
        cutlass::gemm::GemmCoord(1, 1, 1), cutlass::KernelHardwareInfo{}, p.gemm.max_swizzle_size,
        p.gemm.raster == GemmRaster::kAlongN ? SchedulerParams::RasterOrderOptions::AlongN
                                           : SchedulerParams::RasterOrderOptions::AlongM);
    return result;
  }

  static uint64_t route_slots(const ParamsType& p) {
    return producer_schedule(p).blocks_per_problem_ * CopyOrder::kSlots;
  }

  struct Arguments {
    ParamsType params{};
    cute::TmaDescriptor local_output_tma{};
    cute::TmaDescriptor peer_output_tma[kMaxWorldSize][3]{};
    bool use_tma = false;
    bool use_tma_store = false;
    SchedulerParams producer_order{};
    detail::NBandSwizzle n_band_swizzle{};
  };
  using Params = Arguments;

  CUTLASS_DEVICE static int64_t destination_row(
      const ParamsType& p, int32_t source_row) {
    const auto& route = p.route;
    const int32_t batch = source_row / route.seq_local;
    const int32_t local_sequence = source_row - batch * route.seq_local;
    const int32_t sequence_begin = route.rank * route.seq_local;
    return static_cast<int64_t>(batch) * route.global_seq +
        sequence_begin + local_sequence;
  }

  static cudaError_t initialize(Arguments& args) {
    if (!can_implement(args)) {
      return cudaErrorNotSupported;
    }
    const auto& p = args.params;
    args.producer_order = producer_schedule(p);
    args.use_tma_store = false;
    args.use_tma = p.route.head_dim == kQkvBulkColumns &&
        d_row_stride(p.gemm) >= p.gemm.n &&
        d_row_stride(p.gemm) % kCommAlignment == 0;
    if (!args.use_tma) {
      return cudaSuccess;
    }
#if FUSE_SM103_QKV_RANK_SWIZZLE
    args.n_band_swizzle = detail::NBandSwizzle::make(args.producer_order, p.route.rank);
#endif
    const uint64_t global_dims[2] = {
        static_cast<uint64_t>(p.gemm.n),
        static_cast<uint64_t>(p.gemm.m)};
    const uint64_t global_strides[1] = {
        static_cast<uint64_t>(d_row_stride(p.gemm)) * sizeof(CommElement)};
    constexpr uint32_t box_dims[2] = {
        kQkvBulkColumns, kQkvBulkRows};
    constexpr uint32_t element_strides[2] = {1, 1};
    CUresult result;
    {
      FUSE_SM103_HOST_DESCRIPTOR_SCOPE(local_descriptor, 0);
      FUSE_SM103_HOST_DESCRIPTOR_ATTEMPT(local_descriptor);
      result =
        CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &args.local_output_tma,
            kTmaDataType,
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
    }
    if (result != CUDA_SUCCESS) {
      return cudaErrorInvalidValue;
    }

    args.use_tma_store = p.gemm.m % kQkvBulkRows == 0 &&
        p.route.seq_local % kQkvBulkRows == 0;
    if (!args.use_tma_store) {
      return cudaSuccess;
    }
    const int32_t q_local_width =
        p.route.q_heads / p.route.world_size * p.route.head_dim;
    const int32_t kv_local_width =
        p.route.kv_heads / p.route.world_size * p.route.head_dim;
    const uint64_t output_rows =
        static_cast<uint64_t>(p.route.batch) * p.route.global_seq;
    FUSE_SM103_HOST_DESCRIPTOR_SCOPE(peer_descriptors, 1);
    for (int32_t peer = 0; peer < p.route.world_size; ++peer) {
      const int32_t descriptor_count =
          p.route.defer_v_a2a ? 1 : 3;
      for (int32_t segment = 0; segment < descriptor_count; ++segment) {
        const int32_t segment_width = p.route.defer_v_a2a
            ? q_local_width + kv_local_width
            : (segment == 0 ? q_local_width : kv_local_width);
        const int64_t segment_offset =
            p.route.defer_v_a2a || segment == 0
            ? 0
            : static_cast<int64_t>(output_rows) * q_local_width +
                (segment == 2
                     ? static_cast<int64_t>(output_rows) * kv_local_width
                     : 0);
        const uint64_t destination_dims[2] = {
            static_cast<uint64_t>(segment_width), output_rows};
        const uint64_t destination_strides[1] = {
            static_cast<uint64_t>(segment_width) * sizeof(CommElement)};
        FUSE_SM103_HOST_DESCRIPTOR_ATTEMPT(peer_descriptors);
        result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &args.peer_output_tma[peer][segment],
                kTmaDataType,
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
    const auto& route = p.route;
    if (route.world_size <= 0 || route.world_size > kMaxWorldSize ||
        route.rank < 0 || route.rank >= route.world_size || route.batch <= 0 ||
        route.seq_local <= 0 || route.global_seq <= 0 || route.q_heads <= 0 ||
        route.kv_heads <= 0 || route.head_dim <= 0 || p.num_comm_ctas <= 0 ||
        p.epoch == 0 || !supported_problem(p.gemm) ||
        route.kind != RouteKind::kQkvGqaPack ||
        route.direction != RouteDirection::kForward || route.channel_count != 1 ||
        route.cyclic_peer_order || route.packed_source_row ||
        route.packed_row_granularity != 0 ||
        (p.completion_epoch && !route.defer_v_a2a) ||
        route.qkv_peer_interleaved != PeerInterleaved ||
        (route.qkv_peer_interleaved && p.gemm.raster == GemmRaster::kAlongN) ||
        route.q_heads % route.kv_heads != 0 || route.q_heads % route.world_size != 0 ||
        route.kv_heads % route.world_size != 0 || route.head_dim % kCommAlignment != 0) {
      return false;
    }
    if (static_cast<int64_t>(route.seq_local) * route.world_size != route.global_seq ||
        static_cast<int64_t>(route.batch) * route.seq_local != p.gemm.m ||
        static_cast<int64_t>(route.batch) * route.global_seq >
            std::numeric_limits<int32_t>::max() ||
        (static_cast<int64_t>(route.q_heads) + 2 * static_cast<int64_t>(route.kv_heads)) *
            route.head_dim != p.gemm.n) {
      return false;
    }
    auto aligned = [](const void* pointer) {
      return pointer != nullptr && reinterpret_cast<uintptr_t>(pointer) % 16 == 0;
    };
    if (!aligned(p.lhs) || !aligned(p.rhs_nt) || !aligned(p.local_output) ||
        p.ready == nullptr || reinterpret_cast<uintptr_t>(p.ready) % 4 != 0) {
      return false;
    }
    for (int32_t peer = 0; peer < route.world_size; ++peer) {
      if (!aligned(p.peer_output[peer]) || p.peer_route_done_epoch[peer] == nullptr ||
          reinterpret_cast<uintptr_t>(p.peer_route_done_epoch[peer]) % 4 != 0) {
        return false;
      }
    }
    if (p.completion_epoch &&
        reinterpret_cast<uintptr_t>(p.completion_epoch) % alignof(uint32_t) != 0) {
      return false;
    }
    return true;
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

  template <bool TraceTasks = false>
  CUTLASS_DEVICE void run(
      const Params& args,
      char* smem,
      int32_t comm_id,
      int32_t comm_ctas,
      bool wait_for_gemm
#if FUSE_ENABLE_PROFILING
      , QkvRouteTimeline* route_timeline = nullptr
#endif
      ) {
    const auto& p = args.params;
    const int32_t m_tiles = ceil_div(p.gemm.m, BlockM);
    const int32_t m_groups = ceil_div(m_tiles, MTilesPerTask);
    const int32_t output_n_tiles = ceil_div(p.gemm.n, BlockN);
    const int32_t q_local_heads = p.route.q_heads / p.route.world_size;
    const int32_t kv_local_heads = p.route.kv_heads / p.route.world_size;
    const int32_t route_heads = q_local_heads +
        (p.route.defer_v_a2a ? 1 : 2) * kv_local_heads;
    const int32_t chunks_per_head = ceil_div(p.route.head_dim, CopyBlockN);
    const int32_t head_chunks = route_heads * chunks_per_head;
    const int64_t tasks =
        static_cast<int64_t>(p.route.world_size) * m_groups * head_chunks;
    const int64_t source_row_vectors =
        d_row_stride(p.gemm) / kCommAlignment;
    const int32_t q_local_width = q_local_heads * p.route.head_dim;
    const int32_t kv_local_width = kv_local_heads * p.route.head_dim;
    const int64_t segment_rows =
        static_cast<int64_t>(p.route.batch) * p.route.global_seq;

    if (args.use_tma) {
      static_assert(CopyBlockN >= kQkvBulkColumns);
      auto* stages = reinterpret_cast<CommElement*>(smem);
      auto* barriers = reinterpret_cast<uint64_t*>(
          smem + kQkvBulkSlots * kBulkStageBytes);
      const int32_t lane = static_cast<int32_t>(threadIdx.x) & 31;
      const int32_t slot = static_cast<int32_t>(threadIdx.x) >> 5;
      if (slot < kQkvBulkSlots) {
        CommElement* stage = stages +
            static_cast<int64_t>(slot) * kBulkStageElements;
        uint64_t* barrier = barriers + slot;
        int32_t phase = 0;
        if (lane == 0) {
          cute::initialize_barrier(*barrier, 1);
          cutlass::arch::fence_barrier_init();
          cute::prefetch_tma_descriptor(&args.local_output_tma);
        }
        __syncwarp();

        // Follow resolved GEMM production order, not a peer-major copy queue.
        // CopyOrder owns each rectangle at its last logical ready dependency;
        // the waits below still acquire every dependency because CTA completion
        // can be out of order. See producer_consumer.cuh for the HOL contract.
        const int64_t bulk_tasks = args.producer_order.blocks_per_problem_ * CopyOrder::kSlots;
        const int64_t task_stride = static_cast<int64_t>(comm_ctas) * kQkvBulkSlots;
        for (int64_t work = static_cast<int64_t>(slot) * comm_ctas + comm_id;
             work < bulk_tasks;
             work += task_stride) {
          const auto task = CopyOrder::decode(args.producer_order, work, p.gemm.m, p.gemm.n
#if FUSE_SM103_QKV_RANK_SWIZZLE
              , args.n_band_swizzle
#endif
              );
          if (!task.valid) continue;
          const int32_t q_width = p.route.q_heads * p.route.head_dim;
          const int32_t kv_width = p.route.kv_heads * p.route.head_dim;
          const int32_t segment = task.column < q_width
              ? 0 : (task.column < q_width + kv_width ? 1 : 2);
          if (p.route.defer_v_a2a && segment == 2) continue;
          const int32_t base = segment == 0 ? 0 : q_width + (segment == 2 ? kv_width : 0);
          const int32_t physical_head = (task.column - base) / p.route.head_dim;
          const int32_t local_heads = segment == 0 ? q_local_heads : kv_local_heads;
          const int32_t destination_rank = PeerInterleaved && segment < 2
              ? physical_head % p.route.world_size : physical_head / local_heads;
          const int32_t local_head = PeerInterleaved && segment < 2
              ? physical_head / p.route.world_size : physical_head % local_heads;
          const int32_t head_offset = 0; // TMA path: one 128-column head per copy.
          const int32_t physical_feature = task.column;
          const int32_t m_begin = task.row;
          const int32_t copy_m = task.rows;
          const int32_t copy_n = task.columns;

#if FUSE_ENABLE_PROFILING
          QkvRouteTimeline sample{};
          if constexpr (TraceTasks) {
            if (lane == 0) sample.begin = detail::read_global_timer();
          }
#endif

          if (wait_for_gemm) {
            // Latest logical dependency assigns ownership, not completion.
            // Independent producers can finish out of order: acquire them all.
            for (int32_t producer_m = task.first_m; producer_m <= task.last_m; ++producer_m) {
              for (int32_t producer_n = task.first_n; producer_n <= task.last_n; ++producer_n) {
                detail::wait_acquire_system(p.ready +
                    ReadyTile::index(producer_m, producer_n, output_n_tiles) * kReadyFlagStride,
                    p.epoch, lane);
              }
            }
          }
          __syncwarp();

          if (lane == 0) {
#if FUSE_ENABLE_PROFILING
            if constexpr (TraceTasks) sample.ready = detail::read_global_timer();
#endif
            detail::fence_proxy_async_global();
            cute::set_barrier_transaction_bytes(
                *barrier, kBulkStageBytes);
#if FUSE_ENABLE_PROFILING
            if constexpr (TraceTasks) sample.g2s_begin = detail::read_global_timer();
#endif
            SM100_TMA_LOAD_2D::copy(
                &args.local_output_tma,
                barrier,
                0x12f0000000000000ull,
                stage,
                physical_feature,
                m_begin);
            cute::wait_barrier(*barrier, phase);
#if FUSE_ENABLE_PROFILING
            if constexpr (TraceTasks) sample.g2s_done = detail::read_global_timer();
#endif
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
#if FUSE_ENABLE_PROFILING
              if constexpr (TraceTasks) sample.s2g_begin = detail::read_global_timer();
#endif
              SM100_TMA_STORE_2D::copy(
                  &args.peer_output_tma[destination_rank][descriptor],
                  stage,
                  destination_feature,
                  destination_row_value);
              cute::tma_store_arrive();
            } else {
#if FUSE_ENABLE_PROFILING
              if constexpr (TraceTasks) sample.s2g_begin = detail::read_global_timer();
#endif
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
                SM100_BULK_COPY_S2G::copy(
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
#if FUSE_ENABLE_PROFILING
            if constexpr (TraceTasks) {
              sample.s2g_read_done = detail::read_global_timer();
              sample.cta = comm_id; sample.warp = slot;
              sample.row = m_begin; sample.column = physical_feature;
              sample.rows = copy_m; sample.columns = copy_n;
              sample.peer = destination_rank; sample.segment = segment;
              route_timeline[work] = sample;
            }
#endif
          }
          __syncwarp();
        }
        if (lane == 0) {
          // Per-task `.read` waits above make each stage reusable.  Before
          // this CTA publishes completion, also wait for every destination
          // global write issued by this lane to finish.
#if FUSE_ENABLE_PROFILING
          uint64_t drain_begin = 0;
          if constexpr (TraceTasks) drain_begin = detail::read_global_timer();
#endif
          detail::tma_store_wait_all();
#if FUSE_ENABLE_PROFILING
          if constexpr (TraceTasks) {
            auto& drain = route_timeline[bulk_tasks + slot * comm_ctas + comm_id];
            drain.s2g_read_done = detail::read_global_timer();
            drain.begin = drain_begin;
            drain.cta = comm_id; drain.warp = slot;
          }
#endif
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

    for (int64_t work = comm_id; work < tasks; work += comm_ctas) {
      const bool along_n = p.gemm.raster == GemmRaster::kAlongN;
      const int32_t destination_rank = static_cast<int32_t>(work % p.route.world_size);
      const int64_t rank_work = work / p.route.world_size;
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
          const int32_t tile_m = m_group * MTilesPerTask + local_m_tile;
          if (tile_m < m_tiles) {
            const int32_t producer_n = first_producer_n + item % producer_n_tiles;
            const int64_t signal =
                static_cast<int64_t>(tile_m) * output_n_tiles + producer_n;
            detail::wait_acquire_system(
                p.ready + signal * kReadyFlagStride, p.epoch, lane);
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
      __syncthreads();
    }
    // Order every vector writer before handing completion to CTA zero.
    detail::fence_system();

    // This CTA has drained its own Q/K tasks. The epoch publishes V producer
    // completion only; all Q/K routing completes at the later grid/finalize.
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
    const int64_t v_signals = static_cast<int64_t>(m_tiles) * v_n_tiles;
    const int32_t lane = static_cast<int32_t>(threadIdx.x) & 31;
    const int32_t warp = static_cast<int32_t>(threadIdx.x) >> 5;
    const int32_t warps = static_cast<int32_t>(blockDim.x) >> 5;
    for (int64_t item = warp; item < v_signals; item += warps) {
      const int32_t tile_m = static_cast<int32_t>(item / v_n_tiles);
      const int32_t tile_n = v_first_tile + static_cast<int32_t>(item % v_n_tiles);
      const int64_t signal = static_cast<int64_t>(tile_m) * output_n_tiles + tile_n;
      detail::wait_acquire_system(
          p.ready + signal * kReadyFlagStride, p.epoch, lane);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      detail::store_release_system(p.completion_epoch, p.epoch);
    }
  }

  CUTLASS_DEVICE void operator()(
      const Params& args, char* smem, int32_t comm_id, int32_t comm_ctas) {
    run(args, smem, comm_id, comm_ctas, true);
  }

  CUTLASS_DEVICE void finalize(const Params& args) {
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

// Compile-time GEMM/communication pairings, not separate route algorithms.
// Start with Hopper's per-tile task grouping and copy width. In particular,
// a 128-column copy may span several N64 producer tiles. MTilesPerTask groups the
// vector fallback only; the TMA path retains its 64x128 transfer tiles.
// TODO: Tune these starting values on B300 using full GEMM+A2A measurements.
using QkvGqaPackCommN64 = QkvGqaPackCommT<
    GemmA2AParams,
    static_cast<int32_t>(cute::size<0>(N64TileShape{})),
    static_cast<int32_t>(cute::size<1>(N64TileShape{})),
    QkvCommConfig<4, 128>>;
using QkvGqaPackCommSmall = QkvGqaPackCommT<
    GemmA2AParams,
    static_cast<int32_t>(cute::size<0>(TileShape{})),
    static_cast<int32_t>(cute::size<1>(TileShape{})),
    QkvCommConfig<4, 128>>;
using QkvGqaPackCommSmallInterleaved = QkvGqaPackCommT<
    GemmA2AParams,
    static_cast<int32_t>(cute::size<0>(TileShape{})),
    static_cast<int32_t>(cute::size<1>(TileShape{})),
    QkvCommConfig<4, 128, true>>;
using QkvGqaPackCommN160 = QkvGqaPackCommT<
    GemmA2AParams,
    static_cast<int32_t>(cute::size<0>(N160TileShape{})),
    static_cast<int32_t>(cute::size<1>(N160TileShape{})),
    QkvCommConfig<4, 160>>;
using QkvGqaPackCommN192 = QkvGqaPackCommT<
    GemmA2AParams,
    static_cast<int32_t>(cute::size<0>(N192TileShape{})),
    static_cast<int32_t>(cute::size<1>(N192TileShape{})),
    QkvCommConfig<4, 192>>;
using QkvGqaPackCommWide = QkvGqaPackCommT<
    GemmA2AParams,
    static_cast<int32_t>(cute::size<0>(ProjectionTileShape{})),
    static_cast<int32_t>(cute::size<1>(ProjectionTileShape{})),
    QkvCommConfig<1, 256>>;
using QkvGqaPackComm = QkvGqaPackCommSmall;

#if FUSE_ENABLE_PROFILING
template <class GemmKernel, class CommOp, bool OrderedRoleTimestamp = false>
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
    QkvRouteTimeline* route_timeline = nullptr;
  };

  struct Params : BaseKernel::Params {
    A2AGemmCtaTimeline* timeline = nullptr;
    int32_t timeline_capacity = 0;
    QkvRouteTimeline* route_timeline = nullptr;
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
    params.route_timeline = args.route_timeline;
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
      if (params.route_timeline) {
        CommOp{}.template run<true>(params.comm, smem, cta,
            params.num_comm_ctas, true, params.route_timeline);
      } else {
        CommOp{}(params.comm, smem, cta, params.num_comm_ctas);
      }
    } else if (detail::PersistentTileSchedulerSm100Monolithic::valid_initial_worker(
                   params.gemm.scheduler, blockIdx.x)) {
      GemmKernel{}(params.gemm, smem);
    }

    if constexpr (OrderedRoleTimestamp) {
      // Private epilogue probe only: make the timestamp depend on the CTA
      // reduction result. BAR.SYNC.DEFER_BLOCKING followed by an independent
      // timer read was observed before the epilogue's last-ready timestamp.
      // This probe includes extra ordered-join overhead; inspect its emitted
      // predicate dependency and validate on GPU before interpreting timings.
      const int arrived = __syncthreads_count(1);
      if (threadIdx.x == 0 && cta < params.timeline_capacity &&
          arrived == MaxThreadsPerBlock) {
        params.timeline[cta].role_done = detail::read_global_timer();
      }
    } else {
      __syncthreads();
      if (threadIdx.x == 0 && cta < params.timeline_capacity) {
        params.timeline[cta].role_done = detail::read_global_timer();
      }
    }
    __syncthreads();

    if constexpr (CommOp::kNeedsGridFinalize) {
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

}  // namespace
}  // namespace fuse
