// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "fuse/profiling/sm103/host.cuh"
#include "producer_consumer.cuh"

#include "fuse/arch/common.cuh"
#include "fuse/operators/primitives/a2a_gemm.h"

#include <cute/arch/copy_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cuda_host_adapter.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace fuse {
namespace {

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

// Receiver-side pull and per-peer K-shard publication.
template <
    int32_t ReadyBlockM, int32_t TileK = 64
#if FUSE_ENABLE_PROFILING
    , bool Instrumented = false
#endif
    >
struct A2ALhsInputCommT {
  // Blackwell-facing names for CUTLASS's shared SM90-era copy primitives.
  // These are type aliases only; they do not select different instructions.
  using SM100_BULK_COPY_G2S = cute::SM90_BULK_COPY_G2S;
  using SM100_BULK_COPY_S2G = cute::SM90_BULK_COPY_S2G;
  using SM100_TMA_LOAD_3D = cute::SM90_TMA_LOAD_3D;
  using SM100_TMA_STORE_3D = cute::SM90_TMA_STORE_3D;
  using CommElement = Bf16;
  static constexpr int32_t kReadyBlockM = ReadyBlockM;
  static_assert(ReadyBlockM == 128);
  // Only the consumer's peer-K segmentation depends on TileK. Copy rows,
  // arrival counts and byte routing remain identical across GEMM K policies.
  static_assert(TileK == 64 || TileK == 128);
  static constexpr int32_t kTileK = TileK;
  static constexpr int32_t kCommElementsPerVector = 16 / sizeof(CommElement);
  // Vector-fallback rows per task, not the GEMM tile height. Bulk pulls choose
  // comm_rows at initialization from the peer-row bytes and stage capacity.
  // This task size also determines the number of arrivals per ready tile.
  static constexpr int32_t kA2ALhsCommRows = 32;
  static constexpr int32_t kA2ALhsBulkSlots = 4;
  static constexpr int32_t kA2ALhsBulkStageBytes = 48 * 1024;
  static constexpr int32_t kMinThreads = kA2ALhsBulkSlots * 32;
  static constexpr size_t SharedStorageBytes =
      kA2ALhsBulkSlots * kA2ALhsBulkStageBytes +
      kA2ALhsBulkSlots * sizeof(uint64_t);
  static constexpr bool kNeedsGridFinalize = false;

  // Layout-only experiment: rows are short-M/full-peer-K rectangles; columns
  // cover all 128 M rows and advance along K, like the GEMM's loads. Both still
  // publish ONE ready unit [128, full peer-K] after its LAST rectangle completes.
  // No K-slice ready, extra consumer acquire, GEMM transpose or scheduler change.
  // CopyK=192 fills the existing 48 KiB slot; it is independent of GEMM TileK.
  // Keep the layout fixed for a ready buffer's epoch sequence: the cumulative
  // counter's arrivals-per-epoch may differ. Switching requires a fresh buffer.
  struct ColumnCopy {
    int32_t width = 0;
    int32_t chunks = 0;
    int32_t tail = 0;

    CUTLASS_HOST_DEVICE static ColumnCopy make(int32_t peer_k) {
      if (peer_k <= 0) return {};
      const int32_t width = peer_k < 192 ? peer_k : 192;
      return {width, peer_k / width + (peer_k % width != 0), peer_k % width};
    }

    CUTLASS_HOST_DEVICE int32_t width_at(int32_t chunk) const {
      return tail != 0 && chunk == chunks - 1 ? tail : width;
    }
  };

  struct Arguments
#if FUSE_ENABLE_PROFILING
      : A2ALhsCommTimelineArguments<Instrumented>
#endif
  {
    A2AGemmParams params{};
    CUtensorMap store_tma_full{};
    CUtensorMap load_tma_full[kMaxWorldSize]{};
    CUtensorMap load_tma_tail[kMaxWorldSize]{};
    CUtensorMap store_tma_tail{};
    ColumnCopy columns{};
    int32_t comm_rows = kA2ALhsCommRows;
    detail::A2AInputTileOrder input_order{};
    int32_t store_peer_groups = 0;
    int32_t store_rows = 0;
    bool use_bulk = false;
    bool use_tensor_store = false;
    bool use_columns = false;
  };
  using Params = Arguments;

  static bool supported_params(const A2AGemmParams& p) {
    const auto& route = p.route;
    if (route.world_size <= 0 || route.world_size > kMaxWorldSize ||
        route.rank < 0 || route.rank >= route.world_size || route.batch <= 0 ||
        route.seq_local <= 0 || route.global_seq <= 0 || route.q_heads <= 0 ||
        route.local_heads <= 0 || route.head_dim <= 0 || p.num_comm_ctas <= 0 ||
        p.epoch == 0 || !supported_problem(p.gemm) ||
        route.kind != RouteKind::kHeadToSequence ||
        route.direction != RouteDirection::kInverse || route.channel_count != 1 ||
        route.qkv_peer_interleaved || route.defer_v_a2a || route.packed_source_row ||
        route.packed_row_granularity != 0 ||
        (route.causal_load_balanced && route.seq_local % 2 != 0)) {
      return false;
    }
    const int64_t shard_width = static_cast<int64_t>(route.local_heads) * route.head_dim;
    if (static_cast<int64_t>(route.seq_local) * route.world_size != route.global_seq ||
        static_cast<int64_t>(route.batch) * route.seq_local != p.gemm.m ||
        static_cast<int64_t>(route.local_heads) * route.world_size != route.q_heads ||
        static_cast<int64_t>(route.q_heads) * route.head_dim != p.gemm.k ||
        shard_width % kTileK != 0 || route.head_dim % kCommElementsPerVector != 0 ||
        a_row_stride(p.gemm) != p.gemm.k) {
      return false;
    }
    auto aligned = [](const void* pointer) {
      return pointer != nullptr && reinterpret_cast<uintptr_t>(pointer) % 16 == 0;
    };
    if (!aligned(p.input_staging) || !aligned(p.rhs_nt) || !aligned(p.output) ||
        p.ready == nullptr || reinterpret_cast<uintptr_t>(p.ready) % 4 != 0) {
      return false;
    }
    for (int32_t peer = 0; peer < route.world_size; ++peer) {
      if (!aligned(p.peer_input[peer]) ||
          (p.input_epoch != 0 &&
           (p.peer_input_ready[peer] == nullptr ||
            reinterpret_cast<uintptr_t>(p.peer_input_ready[peer]) % 4 != 0))) {
        return false;
      }
    }
    return true;
  }

  static cudaError_t initialize(Arguments& args) {
    if (!supported_params(args.params)) {
      return cudaErrorNotSupported;
    }
    const auto& p = args.params;
    const auto& route = p.route;
    const int64_t row_bytes =
        static_cast<int64_t>(route.local_heads) * route.head_dim * sizeof(Bf16);
    const char* layout = std::getenv("FUSE_SM103_OPROJ_COMM_LAYOUT");
    args.use_columns = layout && std::strcmp(layout, "columns") == 0;
    if (layout && *layout && !args.use_columns && std::strcmp(layout, "rows") != 0) {
      return cudaErrorInvalidValue;
    }
    args.use_bulk = false;
    args.use_tensor_store = false;
    args.store_rows = 0;
    args.store_peer_groups = 0;
    args.comm_rows = static_cast<int32_t>(
        std::min<int64_t>(kReadyBlockM, kA2ALhsBulkStageBytes / row_bytes));
    if (args.use_columns) {
      return initialize_columns(args);
    }
    args.use_bulk = args.comm_rows > 0 && route.seq_local % kReadyBlockM == 0 &&
        (!route.causal_load_balanced || (route.seq_local / 2) % kReadyBlockM == 0);
    if (!args.use_bulk) {
      args.comm_rows = kA2ALhsCommRows;
    }
    if (!can_implement(args)) {
      return cudaErrorNotSupported;
    }
    initialize_input_order(args);
    if (!args.use_bulk || row_bytes > 1024) {
      return cudaSuccess;
    }

    // Narrow peer shards use one 3D TMA store for several rows. A raw u64
    // descriptor factors the K shard without changing its BF16 representation.
    args.store_rows = std::min(args.comm_rows, static_cast<int32_t>(8192 / row_bytes));
    const int32_t shard_u64 = static_cast<int32_t>(row_bytes / sizeof(uint64_t));
    int32_t inner_u64 = 0;
    for (int32_t candidate = 256; candidate >= 2; candidate >>= 1) {
      if (shard_u64 % candidate == 0) {
        inner_u64 = candidate;
        break;
      }
    }
    if (inner_u64 == 0 || args.store_rows < 2) {
      return cudaSuccess;
    }
    const int32_t peer_groups = shard_u64 / inner_u64;
    if (peer_groups > 256) {
      return cudaSuccess;
    }
    cudaError_t status;
    {
      FUSE_SM103_HOST_DESCRIPTOR_SCOPE(local_descriptor, 0);
      FUSE_SM103_HOST_DESCRIPTOR_ATTEMPT(local_descriptor);
      status = make_a2a_lhs_store_tma_3d(
          &args.store_tma_full, p.input_staging, inner_u64,
          peer_groups * route.world_size, p.gemm.m, peer_groups, args.store_rows);
    }
    if (status == cudaSuccess) {
      args.store_peer_groups = peer_groups;
      args.use_tensor_store = true;
    }
    // Descriptor-ineligible narrow shapes retain the bulk row-store path.
    return cudaSuccess;
  }

  static cudaError_t initialize_columns(Arguments& args) {
    const auto& p = args.params;
    const auto& route = p.route;
    const int32_t peer_k = route.local_heads * route.head_dim;
    const int64_t source_rows = static_cast<int64_t>(route.batch) * route.global_seq;
    // Full-M rectangles must not straddle a batch or the causal routing jump.
    // Explicit columns requests fail instead of silently benchmarking rows.
    if (route.seq_local % ReadyBlockM != 0 ||
        (route.causal_load_balanced && (route.seq_local / 2) % ReadyBlockM != 0) ||
        source_rows > INT32_MAX) {
      return cudaErrorNotSupported;
    }
    args.columns = ColumnCopy::make(peer_k);
    args.comm_rows = ReadyBlockM;
    args.use_bulk = true;
    args.use_tensor_store = true;
    if (!can_implement(args)) return cudaErrorNotSupported;
    initialize_input_order(args);

    // UINT64 [16, K/64, M] is only a byte-preserving view of BF16. Tensor G2S
    // is required: consecutive source rows have stride peer-K, not CopyK;
    // destination rows have stride full-K. A separate tail box prevents a
    // partial final column from overwriting the next peer's staging region.
    for (int32_t tail = 0; tail <= (args.columns.tail != 0); ++tail) {
      const int32_t width = tail ? args.columns.tail : args.columns.width;
      auto* store = tail ? &args.store_tma_tail : &args.store_tma_full;
      {
        FUSE_SM103_HOST_DESCRIPTOR_SCOPE(local_descriptor, 0);
        FUSE_SM103_HOST_DESCRIPTOR_ATTEMPT(local_descriptor);
        const auto status = make_a2a_lhs_store_tma_3d(
            store, p.input_staging, 16, p.gemm.k / 64, p.gemm.m,
            width / 64, ReadyBlockM);
        if (status != cudaSuccess) return status;
      }
      for (int32_t peer = 0; peer < route.world_size; ++peer) {
        FUSE_SM103_HOST_DESCRIPTOR_SCOPE(peer_descriptor, 1);
        FUSE_SM103_HOST_DESCRIPTOR_ATTEMPT(peer_descriptor);
        auto* load = tail ? &args.load_tma_tail[peer] : &args.load_tma_full[peer];
        const auto status = make_a2a_lhs_store_tma_3d(
            load, const_cast<CommElement*>(p.peer_input[peer]), 16, peer_k / 64,
            static_cast<int32_t>(source_rows), width / 64, ReadyBlockM);
        if (status != cudaSuccess) return status;
      }
    }
    return cudaSuccess;
  }

  static void initialize_input_order(Arguments& args) {
    // Share capacity lowering between fusion and the independent copy reference.
    // Rows/columns and peer width have already resolved the real chunk count.
    // Bulk has four independent warp slots; vector fallback one task per CTA.
    // No GEMM layout, CTA budget or complete (M,peer) publication is changed.
    const int32_t copy_slots = args.params.num_comm_ctas *
        (args.use_bulk ? kA2ALhsBulkSlots : 1);
    args.input_order.ready_group_m_tiles = std::max(
        int32_t{1}, copy_slots / arrivals_per_peer(args));
  }

  static bool can_implement(const Arguments& args) {
    return supported_params(args.params) && args.comm_rows > 0 &&
        args.comm_rows <= kReadyBlockM && args.input_order.compute_ctas > 0 &&
        args.input_order.m_tiles == ceil_div(args.params.gemm.m, kReadyBlockM) &&
        arrivals_per_peer(args) > 0 &&
        args.params.epoch <= std::numeric_limits<uint32_t>::max() /
            static_cast<uint32_t>(arrivals_per_peer(args));
  }

  static Params to_underlying_arguments(const Arguments& args) { return args; }

  CUTLASS_HOST_DEVICE static int32_t arrivals_per_peer(const Params& args) {
    return args.use_columns ? args.columns.chunks : ceil_div(ReadyBlockM, args.comm_rows);
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
    const int64_t index = static_cast<int64_t>(tile_m) * route.world_size + peer_slot;
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
      const A2AGemmParams& p, int32_t local_sequence) {
    const auto& route = p.route;
    if (!route.causal_load_balanced) {
      return route.rank * route.seq_local + local_sequence;
    }
    const int32_t chunk_rows = route.seq_local / 2;
    const bool first = local_sequence < chunk_rows;
    const int32_t chunk = first ? route.rank : 2 * route.world_size - route.rank - 1;
    return chunk * chunk_rows + (first ? local_sequence : local_sequence - chunk_rows);
  }

  CUTLASS_DEVICE void operator()(
      const Params& args,
      char* smem,
      int32_t comm_id,
      int32_t comm_ctas) {
    const auto& p = args.params;
    const auto& route = p.route;
    const int32_t m_tiles = ceil_div(p.gemm.m, ReadyBlockM);
    const int32_t chunks_per_tile = arrivals_per_peer(args);
    const int64_t tasks_per_m = static_cast<int64_t>(route.world_size) * chunks_per_tile;
    const int64_t tasks = static_cast<int64_t>(m_tiles) * tasks_per_m;
    const int32_t shard_width = route.local_heads * route.head_dim;
    constexpr int32_t elements_per_vector = kCommElementsPerVector;
    const int32_t vectors_per_row = shard_width / elements_per_vector;
    int32_t waited_peer = -1;

    if (args.use_bulk) {
      auto* barriers = reinterpret_cast<uint64_t*>(
          smem + kA2ALhsBulkSlots * kA2ALhsBulkStageBytes);
      const int32_t lane = static_cast<int32_t>(threadIdx.x) & 31;
      const int32_t slot = static_cast<int32_t>(threadIdx.x) >> 5;
      if (slot >= kA2ALhsBulkSlots) {
        return;
      }
      auto* stage = reinterpret_cast<CommElement*>(
          smem + slot * kA2ALhsBulkStageBytes);
      uint64_t* barrier = barriers + slot;
      int32_t phase = 0;
      if (lane == 0) {
        cute::initialize_barrier(*barrier, 1);
        cutlass::arch::fence_barrier_init();
        if (args.use_tensor_store) {
          cute::prefetch_tma_descriptor(&args.store_tma_full);
          if (args.use_columns && args.columns.tail != 0) {
            cute::prefetch_tma_descriptor(&args.store_tma_tail);
          }
        }
      }
      __syncwarp();

      const int32_t task_peer_count = route.world_size;
      const int64_t bulk_tasks =
          static_cast<int64_t>(m_tiles) * task_peer_count * chunks_per_tile;
      const int64_t task_stride = static_cast<int64_t>(comm_ctas) * kA2ALhsBulkSlots;
      for (int64_t task = static_cast<int64_t>(slot) * comm_ctas + comm_id;
           task < bulk_tasks;
           task += task_stride) {
#if FUSE_ENABLE_PROFILING
        A2ALhsCommStageSample<Instrumented> sample{};
        if constexpr (Instrumented) {
          if (lane == 0) {
            sample.task_begin = detail::read_global_timer();
            sample.comm_cta = comm_id;
            sample.comm_slot = slot;
            sample.task_id = task <= INT32_MAX ? static_cast<int32_t>(task) : -1;
            sample.copy_path = args.use_columns ? 3 : (args.use_tensor_store ? 2 : 1);
          }
        }
#endif
        const auto input_task = args.input_order.decode(task, task_peer_count, chunks_per_tile);
        const int32_t tile_m = input_task.m;
        const int32_t peer_slot = input_task.peer;
        const int32_t row_chunk = input_task.chunk;
        // The queue/window is unchanged. Only the rectangle WITHIN each ready
        // unit changes; all four slots still independently contribute arrivals.
        const int32_t row_in_tile = args.use_columns ? 0 : row_chunk * args.comm_rows;
        const int32_t column = args.use_columns ? row_chunk * args.columns.width : 0;
        const int32_t copy_k = args.use_columns ? args.columns.width_at(row_chunk) : shard_width;
        const bool column_tail = args.use_columns && copy_k != args.columns.width;
        const int32_t source_peer = route.cyclic_peer_order
            ? (route.rank + peer_slot) % route.world_size
            : peer_slot;
        const int32_t m_begin = tile_m * ReadyBlockM + row_in_tile;
        const int32_t copy_rows = max(
            0,
            min(min(args.comm_rows, ReadyBlockM - row_in_tile),
                p.gemm.m - m_begin));

#if FUSE_ENABLE_PROFILING
        if constexpr (Instrumented) {
          if (lane == 0) {
            // Historical schema field: copy_path=3 interprets this as a column.
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
          const int32_t logical_row = m_begin;
          const int32_t batch = logical_row / route.seq_local;
          const int32_t local_sequence =
              logical_row - batch * route.seq_local;
          const int32_t source_sequence =
              global_sequence_row(p, local_sequence);
          const auto* source = p.peer_input[source_peer] +
              (static_cast<int64_t>(batch) * route.global_seq +
               source_sequence) * shard_width;
          const int32_t copy_bytes =
              copy_rows * copy_k * sizeof(CommElement);
          detail::fence_proxy_async_global();
          cute::set_barrier_transaction_bytes(*barrier, copy_bytes);
#if FUSE_ENABLE_PROFILING
          if constexpr (Instrumented) {
            sample.g2s_issue = detail::read_global_timer();
          }
#endif
          if (args.use_columns) {
            const auto* load = column_tail
                ? &args.load_tma_tail[source_peer] : &args.load_tma_full[source_peer];
            SM100_TMA_LOAD_3D::copy(
                load, barrier, static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
                stage, 0, column / 64,
                batch * route.global_seq + source_sequence);
          } else {
            SM100_BULK_COPY_G2S::copy(source, barrier, stage, copy_bytes);
          }
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
          if (args.use_columns) {
            const auto* store = column_tail ? &args.store_tma_tail : &args.store_tma_full;
            SM100_TMA_STORE_3D::copy(
                store, stage, 0, (peer_slot * shard_width + column) / 64, m_begin);
            cute::tma_store_arrive();
          } else if (args.use_tensor_store) {
            int32_t row = 0;
            for (; row + args.store_rows <= copy_rows;
                 row += args.store_rows) {
              SM100_TMA_STORE_3D::copy(
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
              SM100_BULK_COPY_S2G::copy(
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
              SM100_BULK_COPY_S2G::copy(
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

    for (int64_t task = comm_id; task < tasks; task += comm_ctas) {
#if FUSE_ENABLE_PROFILING
      A2ALhsCommStageSample<Instrumented> sample{};
      if constexpr (Instrumented) {
        if (threadIdx.x == 0) {
          sample.task_begin = detail::read_global_timer();
          sample.comm_cta = comm_id;
          sample.comm_slot = -1;
          sample.task_id = task <= INT32_MAX ? static_cast<int32_t>(task) : -1;
          sample.copy_path = 0;
        }
      }
#endif
      // Match the bulk path's dependency order, including partial-M fallback.
      const auto input_task = args.input_order.decode(task, route.world_size, chunks_per_tile);
      const int32_t tile_m = input_task.m;
      const int32_t peer_slot = input_task.peer;
      const int32_t row_chunk = input_task.chunk;
      const int32_t source_peer = route.cyclic_peer_order
          ? (route.rank + peer_slot) % route.world_size
          : peer_slot;
      const int32_t m_begin = tile_m * ReadyBlockM +
          row_chunk * args.comm_rows;
      const int32_t copy_rows = max(
          0, min(min(args.comm_rows, ReadyBlockM - row_chunk * args.comm_rows),
                 p.gemm.m - m_begin));

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
      const int64_t vector_count = static_cast<int64_t>(copy_rows) * vectors_per_row;
      for (int64_t index = static_cast<int32_t>(threadIdx.x);
           index < vector_count;
           index += static_cast<int32_t>(blockDim.x)) {
        const int32_t row = index / vectors_per_row;
        const int32_t vector_k = index - row * vectors_per_row;
        const int32_t destination_row = m_begin + row;
        const int32_t logical_row =
            destination_row;
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

};

using A2ALhsInputComm = A2ALhsInputCommT<kTileM>;
#if FUSE_ENABLE_PROFILING
using A2ALhsTelemetryInputComm = A2ALhsInputCommT<kTileM, kTileK, true>;
#endif

}  // namespace
}  // namespace fuse
