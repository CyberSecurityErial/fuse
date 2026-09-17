// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "fuse/operators/primitives/grouped_gemm.h"
#include "../detail/grouped/a2a_gemm.cuh"
#include "../detail/grouped/gemm_a2a.cuh"
#include <cutlass/device_kernel.h>
#include <cutlass/util/packed_stride.hpp>
#include <memory>
#include <new>
#include <limits>

namespace fuse {

// Only host launch dispatch is polymorphic; GPU role dispatch and collectives
// remain statically specialized. No virtual calls occur inside a kernel.
struct Bf16GroupedGemmPlan {
  int device = 0;
  virtual ~Bf16GroupedGemmPlan() = default;
  virtual cudaError_t launch(cudaStream_t) = 0;
};

namespace detail {

struct GroupedDeviceStorage {
  void* ptr = nullptr;
  ~GroupedDeviceStorage() { if (ptr) cudaFree(ptr); }
};

inline void check_grouped_cuda(cudaError_t status) {
  if (status != cudaSuccess) throw status;
}

// Explicit private layout keeps CUTLASS types out of the public parameter ABI.
// Reserve with overflow checks before allocating anything on the GPU.
template <class T>
size_t reserve_grouped_storage(size_t& bytes, size_t count) {
  constexpr size_t limit = std::numeric_limits<size_t>::max();
  if (bytes > limit - 255) throw cudaErrorInvalidValue;
  const size_t offset = (bytes + 255) & ~size_t(255);
  if (count > (limit - offset) / sizeof(T)) throw cudaErrorInvalidValue;
  bytes = offset + count * sizeof(T);
  return offset;
}

template <class SA, class SB, class SD>
__global__ void initialize_grouped_strides(SA* a, SB* b, SD* d,
    int experts, int capacity, int n, int k) {
  for (int e = blockIdx.x * blockDim.x + threadIdx.x;
       e < experts; e += blockDim.x * gridDim.x) {
    a[e] = cutlass::make_cute_packed_stride(SA{}, cute::make_shape(capacity,k,1));
    b[e] = cutlass::make_cute_packed_stride(SB{}, cute::make_shape(n,k,1));
    d[e] = cutlass::make_cute_packed_stride(SD{}, cute::make_shape(capacity,n,1));
  }
}

template <bool Combine, int TileN, int TileK>
struct GroupedBf16PlanImpl final : Bf16GroupedGemmPlan {
  using Types = Bf16GroupedGemmTypes<TileN,TileK>;
  using Kernel = std::conditional_t<Combine, GroupedGemmA2A<TileN,TileK>, A2AGroupedGemm<TileN,TileK>>;
  using G = std::conditional_t<Combine, typename Types::CombineGemm, typename Types::DispatchGemm>;
  using Shape = GroupedProblemShape::UnderlyingProblemShape;
  using SA = typename G::InternalStrideA;
  using SB = typename G::InternalStrideB;
  using SD = typename G::InternalStrideD;
  GroupedDeviceStorage metadata, workspace;
  typename Kernel::Params params{};
  GroupedPrepareParams prepare{};

  void initialize(const Bf16GroupedGemmParams& p) {
    check_grouped_cuda(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    check_grouped_cuda(cudaGetDeviceProperties(&prop, device));
    if (prop.major != 10 || prop.minor != 3 || !prop.cooperativeLaunch)
      throw cudaErrorNotSupported;
    const auto& policy = p.policy;
    if (!p.lhs || !p.weight_nt || !p.output || !p.row_offsets || !p.source ||
        p.experts <= 0 || p.expert_row_capacity <= 0 || p.n <= 0 || p.k <= 0 ||
        p.n % 8 || p.k % 8 || p.world_size < 1 || p.world_size > kMaxWorldSize ||
        p.rank < 0 || p.rank >= p.world_size || p.topk <= 0 ||
        policy.num_comm_ctas <= 0 || policy.num_compute_ctas <= 0 ||
        int64_t(policy.num_comm_ctas) + policy.num_compute_ctas > prop.multiProcessorCount ||
        (policy.swizzle != 1 && policy.swizzle != 2 && policy.swizzle != 4 && policy.swizzle != 8))
      throw cudaErrorInvalidValue;
    for (int r = 0; r < p.world_size; ++r)
      if (!p.peer_started[r] || !p.peer_done[r] ||
          (Combine ? !p.peer_output[r] : !p.peer_input[r])) throw cudaErrorInvalidValue;

    const size_t experts = p.experts;
    if (p.dispatch_buffer_rows < 0 ||
        (p.dispatch_buffer_rows && (Combine || p.dispatch_buffer_rows % 128)))
      throw cudaErrorInvalidValue;
    const int buffer_m = p.dispatch_buffer_rows && p.dispatch_buffer_rows < p.expert_row_capacity
        ? p.dispatch_buffer_rows/128 : 0;
    if (Combine && policy.balance_dispatch_tail) throw cudaErrorInvalidValue;
    const int n_tiles = int((int64_t(p.n) + TileN - 1) / TileN);
    const size_t row_tiles = experts * ((size_t(p.expert_row_capacity) + 127) / 128);
#if FUSE_ENABLE_PROFILING
    if (policy.balance_dispatch_tail && p.profile.panels) throw cudaErrorNotSupported;
    // Existing handoff exporter assumes the original unwindowed task order.
    if (buffer_m && p.profile.panels) throw cudaErrorNotSupported;
    if (p.profile.panels && (Combine || !p.profile.tiles || !p.profile.roles ||
        p.profile.panel_capacity < int64_t(row_tiles) || p.profile.n_tiles != n_tiles))
      throw cudaErrorInvalidValue;
#endif
    if (Combine && row_tiles > std::numeric_limits<size_t>::max() / size_t(n_tiles))
      throw cudaErrorInvalidValue;
    const size_t ready_tiles = Combine ? row_tiles * size_t(n_tiles) : row_tiles;
    if (ready_tiles > std::numeric_limits<size_t>::max() / kReadyFlagStride)
      throw cudaErrorInvalidValue;
    size_t bytes = 0;
    const auto shape_offset = reserve_grouped_storage<Shape>(bytes, experts);
    const auto sa_offset = reserve_grouped_storage<SA>(bytes, experts);
    const auto sb_offset = reserve_grouped_storage<SB>(bytes, experts);
    const auto sd_offset = reserve_grouped_storage<SD>(bytes, experts);
    const auto tile_offset = reserve_grouped_storage<int64_t>(bytes, experts + 1);
    const auto ready_offset = reserve_grouped_storage<uint32_t>(bytes, ready_tiles * kReadyFlagStride);
    const auto epoch_offset = reserve_grouped_storage<uint32_t>(bytes, 1);
    const auto arrivals_offset = reserve_grouped_storage<uint32_t>(bytes, Combine ? 0 : row_tiles);
    const auto consumed_offset = reserve_grouped_storage<uint32_t>(bytes, buffer_m ? row_tiles : 0);
    const auto input_offset = reserve_grouped_storage<const Bf16*>(bytes, p.world_size);
    const auto output_offset = reserve_grouped_storage<Bf16*>(bytes, p.world_size);
    check_grouped_cuda(cudaMalloc(&metadata.ptr, bytes));
    check_grouped_cuda(cudaMemset(metadata.ptr, 0, bytes));
    auto* base = static_cast<char*>(metadata.ptr);
    auto* shapes = reinterpret_cast<Shape*>(base + shape_offset);
    auto* sa = reinterpret_cast<SA*>(base + sa_offset);
    auto* sb = reinterpret_cast<SB*>(base + sb_offset);
    auto* sd = reinterpret_cast<SD*>(base + sd_offset);
    auto* tiles = reinterpret_cast<int64_t*>(base + tile_offset);
    auto* ready = reinterpret_cast<uint32_t*>(base + ready_offset);
    auto* epoch = reinterpret_cast<uint32_t*>(base + epoch_offset);
    auto* arrivals = Combine ? nullptr : reinterpret_cast<uint32_t*>(base + arrivals_offset);
    auto* consumed = buffer_m ? reinterpret_cast<uint32_t*>(base + consumed_offset) : nullptr;
    auto* peer_input = reinterpret_cast<const Bf16**>(base + input_offset);
    auto* peer_output = reinterpret_cast<Bf16**>(base + output_offset);
    check_grouped_cuda(cudaMemcpy(peer_input, p.peer_input,
        p.world_size * sizeof(Bf16*), cudaMemcpyHostToDevice));
    check_grouped_cuda(cudaMemcpy(peer_output, p.peer_output,
        p.world_size * sizeof(Bf16*), cudaMemcpyHostToDevice));
    initialize_grouped_strides<<<(int64_t(p.experts) + 255) / 256, 256>>>(
        sa, sb, sd, p.experts, p.expert_row_capacity, p.n, p.k);
    check_grouped_cuda(cudaGetLastError());

    typename Kernel::Arguments args{};
#if FUSE_ENABLE_PROFILING
    args.profile = p.profile;
    args.comm.params.profile = p.profile;
    if constexpr (!Combine) args.gemm.mainloop.profile = p.profile;
#endif
    auto& g = args.gemm;
    g.mode = cutlass::gemm::GemmUniversalMode::kGrouped;
    g.problem_shape.num_groups = p.experts;
    g.problem_shape.problem_shapes = shapes;
    // CUTLASS's ptr-array ABI lacks top-level const on pointer entries, but
    // these collectives only read the arrays. Keep the public arrays read-only.
    g.mainloop.ptr_A = const_cast<const Bf16**>(reinterpret_cast<const Bf16* const*>(p.lhs));
    g.mainloop.ptr_B = const_cast<const Bf16**>(p.weight_nt);
    g.mainloop.dA = sa; g.mainloop.dB = sb;
    g.epilogue.ptr_D = const_cast<Bf16**>(p.output); g.epilogue.dD = sd;
    g.epilogue.thread.alpha = 1.f; g.epilogue.thread.beta = 0.f;
    g.hw_info.device_id = device;
    g.hw_info.sm_count = prop.multiProcessorCount;
    g.scheduler.row_tile_offsets = tiles; g.scheduler.n = p.n;
    g.scheduler.compute_ctas = policy.num_compute_ctas;
    g.scheduler.block_offset = policy.num_comm_ctas;
    g.scheduler.window_m = buffer_m;
    g.scheduler.max_swizzle_size = policy.swizzle;
    using Raster = typename G::TileScheduler::RasterOrderOptions;
    g.scheduler.raster_order = policy.along_n ? Raster::AlongN : Raster::AlongM;
    GroupedTileOrder order{tiles, p.experts, n_tiles, policy.swizzle, policy.along_n};
    if constexpr (Combine) {
      g.epilogue.order = order; g.epilogue.ready = ready; g.epilogue.epoch_ptr = epoch;
    } else {
      g.mainloop.row_tile_offsets = tiles;
      g.mainloop.ready = ready; g.mainloop.epoch_ptr = epoch;
      g.mainloop.buffer_m = buffer_m;
      g.epilogue.row_tile_offsets = tiles; g.epilogue.consumed = consumed;
    }
    args.num_comm_ctas = policy.num_comm_ctas;
    auto& c = args.comm.params;
    c.order = order; c.row_offsets = p.row_offsets; c.source = p.source; c.ready = ready;
    c.arrivals = arrivals;
    c.buffer_m = buffer_m; c.consumed = consumed;
    c.balance_tail = policy.balance_dispatch_tail;
    c.input = Combine ? reinterpret_cast<const Bf16* const*>(p.output) : peer_input;
    c.output = Combine ? peer_output : p.lhs;
    c.columns = Combine ? p.n : p.k;
    c.topk = p.topk; c.rank = p.rank; c.world_size = p.world_size;
    c.num_comm_ctas = policy.num_comm_ctas; c.epoch_ptr = epoch;
    for (int r = 0; r < p.world_size; ++r) {
      c.peer_done[r] = p.peer_done[r]; prepare.peer_started[r] = p.peer_started[r];
    }
    prepare.row_offsets = p.row_offsets; prepare.row_tile_offsets = tiles;
    prepare.shapes = shapes; prepare.epoch = epoch;
    prepare.arrivals = arrivals; prepare.num_comm_ctas = policy.num_comm_ctas;
    prepare.consumed = consumed;
    prepare.balance_tail = policy.balance_dispatch_tail;
    prepare.experts = p.experts; prepare.n = p.n; prepare.k = p.k;
    prepare.rank = p.rank; prepare.world_size = p.world_size;
    prepare.row_capacity = int64_t(p.experts) * p.expert_row_capacity;
    prepare.expert_row_capacity = p.expert_row_capacity;
    if (!Kernel::can_implement(args)) throw cudaErrorInvalidValue;
    check_grouped_cuda(cudaMalloc(&workspace.ptr, Kernel::get_workspace_size(args)));
    if (Kernel::initialize_workspace(args, workspace.ptr, nullptr) != cutlass::Status::kSuccess)
      throw cudaErrorUnknown;
    params = Kernel::to_underlying_arguments(args, workspace.ptr);
    auto fn = cutlass::device_kernel<Kernel>;
    check_grouped_cuda(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
        Kernel::SharedStorageSize));
    int resident = 0;
    check_grouped_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, fn,
        Kernel::MaxThreadsPerBlock, Kernel::SharedStorageSize));
    if (resident < 1) throw cudaErrorInvalidConfiguration;
    // Initialization is explicitly outside Graph/timing; make metadata visible
    // before launch on any caller-selected nonblocking stream.
    check_grouped_cuda(cudaStreamSynchronize(nullptr));
  }

  cudaError_t launch(cudaStream_t stream) override {
    prepare_grouped_invocation<<<1,256,0,stream>>>(prepare);
    auto status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    void* packed[] = {&params};
    return cudaLaunchCooperativeKernel(reinterpret_cast<void*>(cutlass::device_kernel<Kernel>),
        Kernel::get_grid_shape(params), Kernel::get_block_shape(), packed,
        Kernel::SharedStorageSize, stream);
  }
};

template <bool Combine, int TileN, int TileK>
cudaError_t create_grouped_tiled_plan(const Bf16GroupedGemmParams& params, Bf16GroupedGemmPlan** out) {
  if (!out) return cudaErrorInvalidValue;
  *out = nullptr;
  try {
    auto plan = std::make_unique<GroupedBf16PlanImpl<Combine,TileN,TileK>>();
    plan->initialize(params);
    *out = plan.release();
    return cudaSuccess;
  } catch (cudaError_t status) { return status; }
    catch (const std::bad_alloc&) { return cudaErrorMemoryAllocation; }
}

template <bool Combine>
cudaError_t create_grouped_plan(const Bf16GroupedGemmParams& params, Bf16GroupedGemmPlan** out) {
  if(!out) return cudaErrorInvalidValue;
  *out=nullptr;
  const auto& p=params.policy;
  if(p.tile_n==128 && p.tile_k==64) return create_grouped_tiled_plan<Combine,128,64>(params,out);
  if(p.tile_n==128 && p.tile_k==128) return create_grouped_tiled_plan<Combine,128,128>(params,out);
  if(p.tile_n==256 && p.tile_k==64) return create_grouped_tiled_plan<Combine,256,64>(params,out);
  if(p.tile_n==256 && p.tile_k==128) return create_grouped_tiled_plan<Combine,256,128>(params,out);
  return cudaErrorInvalidValue;
}

}  // namespace detail

cudaError_t create_bf16_a2a_grouped_gemm(const Bf16GroupedGemmParams& p, Bf16GroupedGemmPlan** out) {
  return detail::create_grouped_plan<false>(p, out);
}
cudaError_t create_bf16_grouped_gemm_a2a(const Bf16GroupedGemmParams& p, Bf16GroupedGemmPlan** out) {
  return detail::create_grouped_plan<true>(p, out);
}
cudaError_t launch_bf16_grouped_gemm(Bf16GroupedGemmPlan* plan, cudaStream_t stream) {
  if (!plan) return cudaErrorInvalidValue;
  int current = -1;
  auto status = cudaGetDevice(&current);
  if (status != cudaSuccess) return status;
  if (current != plan->device) return cudaErrorInvalidDevice;
  return plan->launch(stream);
}
cudaError_t destroy_bf16_grouped_gemm(Bf16GroupedGemmPlan* plan) {
  if (!plan) return cudaSuccess;
  int current = -1;
  auto status = cudaGetDevice(&current);
  if (status != cudaSuccess) return status;
  status = cudaSetDevice(plan->device);
  if (status != cudaSuccess) return status;
  delete plan;
  return cudaSetDevice(current);
}

}  // namespace fuse
