// SPDX-License-Identifier: BSD-3-Clause
// Private implementation; assembled by csrc/operators/ulysses_sm90.cu.
//
// Module index:
//   - QkvBackwardPushCommT: natural dQ/dK/dV to packed peer staging
//   - BF16/FP8 backward communication aliases
//   - QKV backward monolithic kernel aliases

namespace fuse {
namespace {

// QKV backward source-push route. Each rank reads its planar local-head
// dQ/dK/dV tensors and writes every destination's [M,QKV] staging matrix in
// the exact forward projection order [all Q][all K][all V].
template <
    int32_t ReadyBlockM,
    class KernelParamsType = QkvBackwardKernelParams>
struct QkvBackwardPushCommT {
  using CommElement = std::remove_cv_t<std::remove_pointer_t<
      decltype(KernelParamsType{}.local_q)>>;
  static constexpr bool kFp8Input =
      std::is_same_v<CommElement, Fp8Element>;
  static constexpr int32_t kElementsPerVector =
      16 / static_cast<int32_t>(sizeof(CommElement));
  static constexpr size_t SharedStorageBytes =
      kQkvBulkSlots * kQkvBulkStageBytes +
      kQkvBulkSlots * sizeof(uint64_t);
  static constexpr bool kNeedsGridFinalize = true;

  struct Arguments {
    KernelParamsType params{};
    cute::TmaDescriptor source_tma[3]{};
    cute::TmaDescriptor peer_staging_tma[kMaxWorldSize]{};
    bool use_tma = false;
  };
  using Params = Arguments;

  static cudaError_t initialize(Arguments& args) {
    const auto& p = args.params;
    const auto& route = p.route;
    args.use_tma = false;
    if (ReadyBlockM != 2 * kQkvBulkRows ||
        route.head_dim != kQkvBulkColumns ||
        route.world_size <= 0 || route.world_size > kMaxWorldSize ||
        route.seq_local <= 0 || route.seq_local % ReadyBlockM != 0 ||
        p.gemm.m <= 0 || p.gemm.m % ReadyBlockM != 0) {
      return cudaSuccess;
    }

    constexpr uint32_t box_dims[2] = {
        kQkvBulkColumns, kQkvBulkRows};
    constexpr uint32_t element_strides[2] = {1, 1};
    auto encode = [&](cute::TmaDescriptor* descriptor,
                      CommElement* pointer,
                      int32_t columns,
                      int32_t rows) {
      const uint64_t global_dims[2] = {
          static_cast<uint64_t>(columns),
          static_cast<uint64_t>(rows)};
      const uint64_t global_strides[1] = {
          static_cast<uint64_t>(columns) * sizeof(CommElement)};
      return CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
          descriptor,
          kFp8Input
              ? CU_TENSOR_MAP_DATA_TYPE_UINT8
              : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
          2,
          pointer,
          global_dims,
          global_strides,
          box_dims,
          element_strides,
          CU_TENSOR_MAP_INTERLEAVE_NONE,
          CU_TENSOR_MAP_SWIZZLE_NONE,
          CU_TENSOR_MAP_L2_PROMOTION_NONE,
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
    };

    const int32_t q_local_width =
        route.q_heads / route.world_size * route.head_dim;
    const int32_t kv_local_width =
        route.kv_heads / route.world_size * route.head_dim;
    const int32_t global_rows = route.batch * route.global_seq;
    if (!encode(
            &args.source_tma[0],
            const_cast<CommElement*>(p.local_q),
            q_local_width,
            global_rows) ||
        !encode(
            &args.source_tma[1],
            const_cast<CommElement*>(p.local_k),
            kv_local_width,
            global_rows) ||
        !encode(
            &args.source_tma[2],
            const_cast<CommElement*>(p.local_v),
            kv_local_width,
            global_rows)) {
      return cudaSuccess;
    }
    for (int32_t peer = 0; peer < route.world_size; ++peer) {
      if (!encode(
              &args.peer_staging_tma[peer],
              p.peer_staging[peer],
              p.gemm.k,
              p.gemm.m)) {
        return cudaSuccess;
      }
    }
    args.use_tma = true;
    return cudaSuccess;
  }

  static bool can_implement(const Arguments& args) {
    const auto& p = args.params;
    const auto& route = p.route;
    bool pointers = p.local_q && p.local_k && p.local_v && p.weight &&
        p.grad_input && p.peer_staging[route.rank] &&
        p.peer_ready[route.rank];
    for (int32_t peer = 0; peer < route.world_size && pointers; ++peer) {
      pointers = p.peer_staging[peer] && p.peer_ready[peer] &&
          p.peer_done_epoch[peer];
    }
    const int32_t packed_heads = route.q_heads + 2 * route.kv_heads;
    return pointers && p.epoch > 0 && supported_route_base(route) &&
        route.kind == RouteKind::kQkvGqaPack &&
        route.direction == RouteDirection::kInverse &&
        route.q_heads > 0 && route.kv_heads > 0 &&
        route.q_heads % route.kv_heads == 0 &&
        route.q_heads % route.world_size == 0 &&
        route.kv_heads % route.world_size == 0 &&
        route.head_dim > 0 && route.head_dim % kElementsPerVector == 0 &&
        route.global_seq == route.seq_local * route.world_size &&
        (!route.causal_load_balanced || route.seq_local % 2 == 0) &&
        p.gemm.l == 1 && p.gemm.m == route.batch * route.seq_local &&
        p.gemm.k == packed_heads * route.head_dim && p.gemm.n > 0 &&
        (kFp8Input
             ? supported_fp8_problem(p.gemm)
             : (p.gemm.transpose_b &&
                supported_transpose_b_problem(p.gemm)));
  }

  static Params to_underlying_arguments(const Arguments& args) { return args; }

  __host__ __device__ static int32_t arrivals_per_peer(const Params&) {
    return 1;
  }

  CUTLASS_DEVICE static int32_t global_sequence_row(
      const UlyssesRoute& route,
      int32_t destination_rank,
      int32_t local_sequence) {
    if (!route.causal_load_balanced) {
      return destination_rank * route.seq_local + local_sequence;
    }
    const int32_t chunk_rows = route.seq_local / 2;
    const int32_t chunk = local_sequence < chunk_rows
        ? destination_rank
        : 2 * route.world_size - destination_rank - 1;
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
    const int32_t q_local_heads = route.q_heads / route.world_size;
    const int32_t kv_local_heads = route.kv_heads / route.world_size;
    const int32_t local_packed_heads =
        q_local_heads + 2 * kv_local_heads;
    const int32_t packed_heads = route.q_heads + 2 * route.kv_heads;
    const int32_t m_tiles = ceil_div(p.gemm.m, ReadyBlockM);
    const int32_t tasks =
        m_tiles * route.world_size * local_packed_heads;
    constexpr int32_t elements_per_vector = kElementsPerVector;
    const int32_t vectors_per_head = route.head_dim / elements_per_vector;

    if (args.use_tma) {
      extern __shared__ char dynamic_smem[];
      auto* stages = reinterpret_cast<CommElement*>(dynamic_smem);
      auto* barriers = reinterpret_cast<uint64_t*>(
          dynamic_smem + kQkvBulkSlots * kQkvBulkStageBytes);
      const int32_t lane = static_cast<int32_t>(threadIdx.x) & 31;
      const int32_t slot = static_cast<int32_t>(threadIdx.x) >> 5;
      if (slot < kQkvBulkSlots) {
        auto* stage = stages +
            static_cast<int64_t>(slot) *
                kQkvBulkRows * kQkvBulkColumns;
        uint64_t* barrier = barriers + slot;
        int32_t phase = 0;
        if (lane == 0) {
          cute::initialize_barrier(*barrier, 1);
        }
        __syncwarp();

        const int32_t task_stride = comm_ctas * kQkvBulkSlots;
        for (int32_t task = slot * comm_ctas + comm_id;
             task < tasks;
             task += task_stride) {
          const int32_t local_head = task % local_packed_heads;
          const int32_t destination_work = task / local_packed_heads;
          const int32_t destination_rank =
              destination_work % route.world_size;
          const int32_t tile_m = destination_work / route.world_size;

          int32_t segment = 0;
          int32_t segment_head = local_head;
          int32_t global_head = 0;
          if (local_head >= q_local_heads + kv_local_heads) {
            segment = 2;
            segment_head =
                local_head - q_local_heads - kv_local_heads;
            global_head = route.q_heads + route.kv_heads +
                route.rank * kv_local_heads + segment_head;
          } else if (local_head >= q_local_heads) {
            segment = 1;
            segment_head = local_head - q_local_heads;
            global_head = route.q_heads +
                route.rank * kv_local_heads + segment_head;
          } else {
            global_head =
                route.rank * q_local_heads + segment_head;
          }

          if (lane == 0) {
            for (int32_t chunk = 0; chunk < 2; ++chunk) {
              const int32_t destination_row =
                  tile_m * ReadyBlockM + chunk * kQkvBulkRows;
              const int32_t batch = destination_row / route.seq_local;
              const int32_t local_sequence =
                  destination_row - batch * route.seq_local;
              const int32_t source_row = batch * route.global_seq +
                  global_sequence_row(
                      route, destination_rank, local_sequence);
              cute::set_barrier_transaction_bytes(
                  *barrier,
                  kQkvBulkRows * kQkvBulkColumns * sizeof(CommElement));
              cute::SM90_TMA_LOAD_2D::copy(
                  &args.source_tma[segment],
                  barrier,
                  0x12f0000000000000ull,
                  stage,
                  segment_head * route.head_dim,
                  source_row);
              cute::wait_barrier(*barrier, phase);
              phase ^= 1;
              cute::tma_store_fence();
              cute::SM90_TMA_STORE_2D::copy(
                  &args.peer_staging_tma[destination_rank],
                  stage,
                  global_head * route.head_dim,
                  destination_row);
              cute::tma_store_arrive();
              // The stage can be reused once the async engine has consumed
              // it; final destination completion is enforced below before
              // publishing the ready epoch.
              cute::tma_store_wait<0>();
            }
            detail::tma_store_wait_all();
            detail::store_release_system(
                p.peer_ready[destination_rank] +
                    (static_cast<int64_t>(tile_m) * packed_heads +
                     global_head) * kReadyFlagStride,
                p.epoch);
          }
          __syncwarp();
        }
        if (lane == 0) {
          cutlass::arch::ClusterBarrier::invalidate(barrier);
        }
      }
      return;
    }

    for (int32_t task = comm_id; task < tasks; task += comm_ctas) {
      const int32_t local_head = task % local_packed_heads;
      const int32_t destination_work = task / local_packed_heads;
      const int32_t destination_rank =
          destination_work % route.world_size;
      const int32_t tile_m = destination_work / route.world_size;

      int32_t segment_head = local_head;
      int32_t global_head = 0;
      int32_t segment_width = q_local_heads * route.head_dim;
      const CommElement* segment = p.local_q;
      if (local_head >= q_local_heads + kv_local_heads) {
        segment_head = local_head - q_local_heads - kv_local_heads;
        global_head = route.q_heads + route.kv_heads +
            route.rank * kv_local_heads + segment_head;
        segment_width = kv_local_heads * route.head_dim;
        segment = p.local_v;
      } else if (local_head >= q_local_heads) {
        segment_head = local_head - q_local_heads;
        global_head = route.q_heads +
            route.rank * kv_local_heads + segment_head;
        segment_width = kv_local_heads * route.head_dim;
        segment = p.local_k;
      } else {
        global_head = route.rank * q_local_heads + segment_head;
      }

      const int32_t m_begin = tile_m * ReadyBlockM;
      const int32_t copy_rows =
          max(0, min(ReadyBlockM, p.gemm.m - m_begin));
      const auto* source = reinterpret_cast<const uint4*>(segment);
      auto* destination = reinterpret_cast<uint4*>(
          p.peer_staging[destination_rank]);
      const int32_t vector_count = copy_rows * vectors_per_head;
      for (int32_t index = static_cast<int32_t>(threadIdx.x);
           index < vector_count;
           index += static_cast<int32_t>(blockDim.x)) {
        const int32_t row = index / vectors_per_head;
        const int32_t vector_n = index - row * vectors_per_head;
        const int32_t destination_row = m_begin + row;
        const int32_t batch = destination_row / route.seq_local;
        const int32_t local_sequence =
            destination_row - batch * route.seq_local;
        const int32_t source_sequence = global_sequence_row(
            route, destination_rank, local_sequence);
        const int64_t source_element =
            (static_cast<int64_t>(batch) * route.global_seq +
             source_sequence) * segment_width +
            segment_head * route.head_dim;
        const int64_t destination_element =
            static_cast<int64_t>(destination_row) * p.gemm.k +
            global_head * route.head_dim;
        destination[destination_element / elements_per_vector + vector_n] =
            source[source_element / elements_per_vector + vector_n];
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        // The destination GEMM runs on another GPU and may consume this tile
        // immediately after observing the flag.
        detail::fence_system();
        detail::store_release_system(
            p.peer_ready[destination_rank] +
                (static_cast<int64_t>(tile_m) * packed_heads + global_head) *
                    kReadyFlagStride,
            p.epoch);
      }
      __syncthreads();
    }
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
          p.peer_done_epoch[lane] + p.route.rank * kReadyFlagStride,
          p.epoch);
    }
    __syncwarp();
    const uint32_t* local_done = p.peer_done_epoch[p.route.rank];
    if (lane < p.route.world_size) {
      while (detail::load_acquire_system(
                 local_done + lane * kReadyFlagStride) < p.epoch) {
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
          p.peer_done_epoch[lane] + p.route.rank * kReadyFlagStride,
          p.epoch);
    }
    __syncwarp();
    if (lane == 0) {
      event->publish_done = detail::read_global_timer();
    }
    const uint32_t* local_done = p.peer_done_epoch[p.route.rank];
    if (lane < p.route.world_size) {
      while (detail::load_acquire_system(
                 local_done + lane * kReadyFlagStride) < p.epoch) {
        __nanosleep(64);
      }
      event->source_ready[lane] = detail::read_global_timer();
    }
    __syncwarp();
  }
#endif
};

using A2ALhsInputComm = A2ALhsInputCommT<kBlockM>;
using A2ALhsM64InputComm = A2ALhsInputCommT<64>;
using Fp8A2ALhsInputComm =
    A2ALhsInputCommT<kBlockM, Fp8A2AGemmParams>;
using QkvBackwardPushComm = QkvBackwardPushCommT<kBlockM>;
using Fp8QkvBackwardPushComm =
    QkvBackwardPushCommT<kBlockM, Fp8QkvBackwardKernelParams>;
template <int32_t ReadyBlockM, bool Finalize>
using WeightedA2ALhsInputComm = A2ALhsInputCommT<
    ReadyBlockM,
    WeightedA2AGemmKernelParams,
    true,
    Finalize>;
#if FUSE_ENABLE_PROFILING
using A2ALhsTelemetryInputComm =
    A2ALhsInputCommT<kBlockM, A2AGemmParams, false, false, true>;
using A2ALhsM64TelemetryInputComm =
    A2ALhsInputCommT<64, A2AGemmParams, false, false, true>;
#endif

using A2ALhsGemmKernel =
    detail::MonolithicGemm<A2ALhsInputGemm, A2ALhsInputComm>;
using Fp8A2ALhsGemmKernel =
    detail::MonolithicGemm<Fp8A2ALhsGemm, Fp8A2ALhsInputComm>;
using Fp8CooperativeA2ALhsGemmKernel = detail::MonolithicGemm<
    Fp8CooperativeA2ALhsGemm, Fp8A2ALhsInputComm>;
using A2ALhsN160GemmKernel =
    detail::MonolithicGemm<A2ALhsN160Gemm, A2ALhsInputComm>;
#if FUSE_ENABLE_PROFILING
using A2ALhsN160TelemetryBase = detail::MonolithicGemm<
    A2ALhsN160TelemetryGemm, A2ALhsTelemetryInputComm>;
using A2ALhsN160TelemetryKernel =
    detail::RoleTelemetryKernel<A2ALhsN160TelemetryBase>;
using A2ALhsTelemetryBase =
    detail::MonolithicGemm<A2ALhsTelemetryGemm, A2ALhsTelemetryInputComm>;
using A2ALhsTelemetryKernel =
    detail::RoleTelemetryKernel<A2ALhsTelemetryBase>;
#endif
using A2ALhsProjectionGemmKernel =
    detail::MonolithicGemm<A2ALhsProjectionGemm, A2ALhsInputComm>;
using A2ALhsWideN320GemmKernel =
    detail::MonolithicGemm<A2ALhsWideN320Gemm, A2ALhsInputComm>;

template <class Gemm>
using QkvBackwardDataKernelT =
    detail::MonolithicGemm<Gemm, QkvBackwardPushComm>;
using QkvBackwardDataKernelN64 =
    QkvBackwardDataKernelT<BackwardN64ReadyGemm>;
using QkvBackwardDataKernelN128 =
    QkvBackwardDataKernelT<BackwardN128ReadyGemm>;
using QkvBackwardDataKernelN160 =
    QkvBackwardDataKernelT<BackwardN160ReadyGemm>;
using QkvBackwardDataKernelN192 =
    QkvBackwardDataKernelT<BackwardN192ReadyGemm>;
using QkvBackwardDataKernelN256 =
    QkvBackwardDataKernelT<BackwardA2ALhsGemm>;
using QkvBackwardDataKernelN64ClusterM2 =
    QkvBackwardDataKernelT<BackwardN64ClusterM2ReadyGemm>;
using Fp8QkvBackwardDataKernel = detail::MonolithicGemm<
    Fp8BackwardReadyGemm, Fp8QkvBackwardPushComm>;
#if FUSE_ENABLE_PROFILING
using A2ALhsProjectionTelemetryBase = detail::MonolithicGemm<
    A2ALhsProjectionTelemetryGemm, A2ALhsTelemetryInputComm>;
using A2ALhsProjectionTelemetryKernel =
    detail::RoleTelemetryKernel<A2ALhsProjectionTelemetryBase>;
using A2ALhsWideN320TelemetryBase = detail::MonolithicGemm<
    A2ALhsWideN320TelemetryGemm, A2ALhsTelemetryInputComm>;
using A2ALhsWideN320TelemetryKernel =
    detail::RoleTelemetryKernel<A2ALhsWideN320TelemetryBase>;
#endif
using A2ALhsM64GemmKernel =
    detail::MonolithicGemm<A2ALhsM64Gemm, A2ALhsM64InputComm>;
#if FUSE_ENABLE_PROFILING
using A2ALhsM64TelemetryBase =
    detail::MonolithicGemm<
        A2ALhsM64TelemetryGemm, A2ALhsM64TelemetryInputComm>;
using A2ALhsM64TelemetryKernel =
    detail::RoleTelemetryKernel<A2ALhsM64TelemetryBase>;
#endif
static_assert(
    sizeof(typename A2ALhsGemmKernel::SharedStorage) >=
        kA2ALhsBulkSlots * kA2ALhsBulkStageBytes +
            kA2ALhsBulkSlots * sizeof(uint64_t),
    "A2A LHS monolithic shared storage must hold every bulk slot");
static_assert(
    sizeof(typename A2ALhsN160GemmKernel::SharedStorage) >=
        kA2ALhsBulkSlots * kA2ALhsBulkStageBytes +
            kA2ALhsBulkSlots * sizeof(uint64_t),
    "N160 A2A LHS monolithic shared storage must hold every bulk slot");
static_assert(
    sizeof(typename A2ALhsProjectionGemmKernel::SharedStorage) >=
        kA2ALhsBulkSlots * kA2ALhsBulkStageBytes +
            kA2ALhsBulkSlots * sizeof(uint64_t),
    "wide A2A LHS monolithic shared storage must hold every bulk slot");
static_assert(
    sizeof(typename A2ALhsWideN320GemmKernel::SharedStorage) >=
        kA2ALhsBulkSlots * kA2ALhsBulkStageBytes +
            kA2ALhsBulkSlots * sizeof(uint64_t),
    "N320 A2A LHS monolithic shared storage must hold every bulk slot");
static_assert(
    sizeof(typename A2ALhsM64GemmKernel::SharedStorage) >=
        kA2ALhsBulkSlots * kA2ALhsBulkStageBytes +
            kA2ALhsBulkSlots * sizeof(uint64_t),
    "M64 A2A LHS monolithic shared storage must hold every bulk slot");

}  // namespace
}  // namespace fuse
