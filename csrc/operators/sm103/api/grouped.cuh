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
#include <type_traits>

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

template <bool SwapAB, class SA, class SB, class SD>
__global__ void initialize_grouped_strides(SA* a, SB* b, SD* d,
    int experts, int capacity, int n, int k) {
  for (int e = blockIdx.x * blockDim.x + threadIdx.x;
       e < experts; e += blockDim.x * gridDim.x) {
    a[e] = cutlass::make_cute_packed_stride(SA{}, cute::make_shape(SwapAB?n:capacity,k,1));
    b[e] = cutlass::make_cute_packed_stride(SB{}, cute::make_shape(SwapAB?capacity:n,k,1));
    d[e] = cutlass::make_cute_packed_stride(SD{}, cute::make_shape(SwapAB?n:capacity,SwapAB?capacity:n,1));
  }
}

template <bool Combine, int TileN, int TileK, bool SwapAB = false, bool TrimTokens = true,
    class Selector = GroupedExplicitPreparePolicy, bool StockScheduler = false,
    int SmMode = StockScheduler ? 2 : 1>
struct GroupedBf16PlanImpl final : Bf16GroupedGemmPlan {
  static_assert(!Combine || !SwapAB);
  static constexpr bool SeparateRoles = StockScheduler || SmMode == 2;
  static_assert(!SeparateRoles || (!Combine && !SwapAB));
  static constexpr bool DynamicPolicy = !std::is_same_v<Selector, GroupedExplicitPreparePolicy>;
  static_assert(!Combine || !DynamicPolicy, "device policy selection currently supports Dispatch only");
  static_assert(!SeparateRoles || !DynamicPolicy,
      "separate communication/GEMM branches use an explicit measured policy");
  static_assert(std::is_trivially_copyable_v<Selector>, "device selectors must be kernel arguments");
  static constexpr int TileM=128*SmMode;
  using Types = std::conditional_t<StockScheduler,
      Bf16GroupedStockGemmTypes<TileN,TileK,SmMode>,
      Bf16GroupedGemmTypes<TileN,TileK,SwapAB,TrimTokens,SmMode>>;
  using Kernel = std::conditional_t<Combine, GroupedGemmA2A<TileN,TileK>,
      GroupedMonolithicGemm<typename Types::DispatchGemm,
          GroupedDispatchComm<TileM>,Selector,SeparateRoles>>;
  using G = std::conditional_t<Combine, typename Types::CombineGemm, typename Types::DispatchGemm>;
  using Shape = GroupedProblemShape::UnderlyingProblemShape;
  using SA = typename G::InternalStrideA;
  using SB = typename G::InternalStrideB;
  using SD = typename G::InternalStrideD;
  GroupedDeviceStorage metadata, workspace;
  typename Kernel::Params params{};
  typename Kernel::Params gemm_params{}, comm_params{};
  cudaStream_t comm_stream = nullptr;
  cudaEvent_t fork_event = nullptr, comm_done_event = nullptr;
  GroupedPrepareParams prepare{};
  using PreparePolicy = std::conditional_t<DynamicPolicy,
      GroupedPreparePolicyPatch<typename Kernel::Params,Selector>,GroupedExplicitPreparePolicy>;
  PreparePolicy prepare_policy{};

  ~GroupedBf16PlanImpl() override {
    if constexpr (SeparateRoles) {
      int previous = 0;
      cudaGetDevice(&previous);
      cudaSetDevice(device);
      if (fork_event) cudaEventDestroy(fork_event);
      if (comm_done_event) cudaEventDestroy(comm_done_event);
      if (comm_stream) cudaStreamDestroy(comm_stream);
      cudaSetDevice(previous);
    }
  }

  void initialize(const Bf16GroupedGemmParams& p, const Selector& selector = {}) {
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
    if (policy.mma_sm_count != SmMode) throw cudaErrorInvalidValue;
    if constexpr (SeparateRoles) {
      if (policy.num_comm_ctas % SmMode || policy.num_compute_ctas % SmMode ||
          policy.num_comm_ctas + policy.num_compute_ctas != prop.multiProcessorCount ||
          p.dispatch_buffer_rows) throw cudaErrorInvalidValue;
    }
    for (int r = 0; r < p.world_size; ++r)
      if (!p.peer_started[r] || !p.peer_done[r] ||
          (Combine ? !p.peer_output[r] : !p.peer_input[r])) throw cudaErrorInvalidValue;

    const size_t experts = p.experts;
    if (p.dispatch_buffer_rows < 0 ||
        (p.dispatch_buffer_rows && (Combine || p.dispatch_buffer_rows % 128)))
      throw cudaErrorInvalidValue;
    const int buffer_m = p.dispatch_buffer_rows && p.dispatch_buffer_rows < p.expert_row_capacity
        ? p.dispatch_buffer_rows/TileM : 0;
    if (Combine && policy.balance_dispatch_tail) throw cudaErrorInvalidValue;
    const int n_tiles = int((int64_t(p.n) + TileN - 1) / TileN);
    const size_t row_tiles = experts * ((size_t(p.expert_row_capacity) + TileM-1) / TileM);
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
    const auto first_wave_offset = reserve_grouped_storage<int32_t>(bytes, Combine ? 0 : 1);
    const auto effective_comm_offset = reserve_grouped_storage<int32_t>(bytes, Combine ? 0 : 1);
    const auto arrivals_offset = reserve_grouped_storage<uint32_t>(bytes, Combine ? 0 : row_tiles);
    const auto consumed_offset = reserve_grouped_storage<uint32_t>(bytes, buffer_m ? row_tiles : 0);
    const auto input_offset = reserve_grouped_storage<const Bf16*>(bytes, p.world_size);
    const auto output_offset = reserve_grouped_storage<Bf16*>(bytes, p.world_size);
    size_t invocation_offset = 0;
    if constexpr (DynamicPolicy) {
      static_assert(alignof(typename Kernel::Params) <= 256);
      invocation_offset = reserve_grouped_storage<typename Kernel::Params>(bytes, 1);
    }
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
    auto* first_wave = Combine ? nullptr : reinterpret_cast<int32_t*>(base + first_wave_offset);
    auto* effective_comm = Combine ? nullptr : reinterpret_cast<int32_t*>(base + effective_comm_offset);
    auto* arrivals = Combine ? nullptr : reinterpret_cast<uint32_t*>(base + arrivals_offset);
    auto* consumed = buffer_m ? reinterpret_cast<uint32_t*>(base + consumed_offset) : nullptr;
    auto* peer_input = reinterpret_cast<const Bf16**>(base + input_offset);
    auto* peer_output = reinterpret_cast<Bf16**>(base + output_offset);
    check_grouped_cuda(cudaMemcpy(peer_input, p.peer_input,
        p.world_size * sizeof(Bf16*), cudaMemcpyHostToDevice));
    check_grouped_cuda(cudaMemcpy(peer_output, p.peer_output,
        p.world_size * sizeof(Bf16*), cudaMemcpyHostToDevice));
    initialize_grouped_strides<SwapAB><<<(int64_t(p.experts) + 255) / 256, 256>>>(
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
    // Row-major [M,N] and column-major [N,M] share the same physical output.
    // Swap only GEMM operands/shapes; producer order and ready IDs stay logical.
    if constexpr (SwapAB) std::swap(g.mainloop.ptr_A,g.mainloop.ptr_B);
    g.mainloop.dA = sa; g.mainloop.dB = sb;
    g.epilogue.ptr_D = const_cast<Bf16**>(p.output); g.epilogue.dD = sd;
    g.epilogue.thread.alpha = 1.f; g.epilogue.thread.beta = 0.f;
    g.hw_info.device_id = device;
    g.hw_info.sm_count = StockScheduler
        ? policy.num_compute_ctas : prop.multiProcessorCount;
    if constexpr (!StockScheduler) {
      g.scheduler.row_tile_offsets = tiles; g.scheduler.n = p.n;
      g.scheduler.compute_ctas = policy.num_compute_ctas;
      g.scheduler.block_offset = SeparateRoles ? 0 : policy.num_comm_ctas;
      g.scheduler.window_m = buffer_m;
    }
    g.scheduler.max_swizzle_size = policy.swizzle;
    using Raster = typename G::TileScheduler::RasterOrderOptions;
    g.scheduler.raster_order = policy.along_n ? Raster::AlongN : Raster::AlongM;
    GroupedTileOrder order{tiles, p.experts, n_tiles, policy.swizzle, policy.along_n, buffer_m};
    order = grouped_consumer_order(order, StockScheduler);
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
    c.first_wave_panels = first_wave;
    c.effective_comm_ctas = effective_comm;
    c.buffer_m = buffer_m; c.consumed = consumed;
    c.balance_tail = policy.balance_dispatch_tail;
    c.cp_async_g2s = policy.dispatch_copy == GroupedDispatchCopy::CpAsync;
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
    prepare.first_wave_panels = first_wave; prepare.order = order;
    prepare.mma_sm_count = SmMode;
    prepare.effective_comm_ctas = effective_comm;
    prepare.num_compute_ctas = policy.num_compute_ctas;
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
    if constexpr (SeparateRoles) {
      gemm_params=params; gemm_params.split_role=1;
      comm_params=params; comm_params.split_role=2;
#if FUSE_ENABLE_PROFILING
      gemm_params.profile.cta_offset=policy.num_comm_ctas;
      gemm_params.gemm.mainloop.profile.cta_offset=policy.num_comm_ctas;
#endif
      check_grouped_cuda(cudaStreamCreateWithFlags(&comm_stream,cudaStreamNonBlocking));
      check_grouped_cuda(cudaEventCreateWithFlags(&fork_event,cudaEventDisableTiming));
      check_grouped_cuda(cudaEventCreateWithFlags(&comm_done_event,cudaEventDisableTiming));
    }
    if constexpr (DynamicPolicy) {
      auto* invocation = reinterpret_cast<typename Kernel::Params*>(base + invocation_offset);
      // Copy immutable descriptors/pointers once. Prepare later patches only
      // policy scalars, while this host envelope keeps its fixed launch grid.
      check_grouped_cuda(cudaMemcpy(invocation, &params, sizeof(params), cudaMemcpyHostToDevice));
      params.invocation_params = invocation;
      prepare_policy.invocation_params = invocation;
      prepare_policy.selector = selector;
      prepare_policy.initial = {policy.num_comm_ctas, policy.num_compute_ctas,
          policy.swizzle, policy.along_n};
      prepare_policy.launch_ctas = params.num_comm_ctas + params.compute_ctas;
    }
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
    prepare_grouped_invocation<TileM,SwapAB><<<1,256,0,stream>>>(prepare,prepare_policy);
    auto status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    void* packed[] = {&params};
    if constexpr (!SeparateRoles)
      return cudaLaunchCooperativeKernel(reinterpret_cast<void*>(cutlass::device_kernel<Kernel>),
          Kernel::get_grid_shape(params), Kernel::get_block_shape(), packed,
          Kernel::SharedStorageSize, stream);
    else {
      status=cudaEventRecord(fork_event,stream);
      if(status!=cudaSuccess) return status;
      status=cudaStreamWaitEvent(comm_stream,fork_event);
      if(status!=cudaSuccess) return status;
      cudaLaunchAttribute attributes[2]{};
      attributes[0].id=cudaLaunchAttributeCooperative;
      attributes[0].val.cooperative=1;
      attributes[1].id=cudaLaunchAttributeClusterDimension;
      attributes[1].val.clusterDim={SmMode,1,1};
      cudaLaunchConfig_t config{};
      config.gridDim=dim3(comm_params.num_comm_ctas,1,1);
      config.blockDim=Kernel::get_block_shape();
      config.dynamicSmemBytes=Kernel::SharedStorageSize;
      config.stream=comm_stream;
      config.attrs=attributes;
      config.numAttrs=2;
      status=cudaLaunchKernelEx(&config,cutlass::device_kernel<Kernel>,comm_params);
      if(status!=cudaSuccess) return status;
      cudaLaunchAttribute cluster{};
      cluster.id=cudaLaunchAttributeClusterDimension;
      cluster.val.clusterDim={SmMode,1,1};
      // Stock grouped scheduling uses the physical x/y grid to recover the
      // CTA's position within a raster/cluster. Flattening (2, C/2) into
      // (C, 1) preserves the CTA count but changes its logical M coordinate.
      config.gridDim=G::get_grid_shape(gemm_params.gemm);
      config.stream=stream;
      config.attrs=&cluster;
      config.numAttrs=1;
      status=cudaLaunchKernelEx(&config,cutlass::device_kernel<Kernel>,gemm_params);
      if(status!=cudaSuccess) return status;
      status=cudaEventRecord(comm_done_event,comm_stream);
      if(status!=cudaSuccess) return status;
      return cudaStreamWaitEvent(stream,comm_done_event);
    }
  }
};

template <bool Combine, int TileN, int TileK, bool SwapAB = false, bool TrimTokens = true,
    bool StockScheduler = false, class Selector = GroupedExplicitPreparePolicy,
    int SmMode = StockScheduler ? 2 : 1>
cudaError_t create_grouped_tiled_plan(const Bf16GroupedGemmParams& params, Bf16GroupedGemmPlan** out,
    const Selector& selector = {}) {
  if (!out) return cudaErrorInvalidValue;
  *out = nullptr;
  try {
    auto plan = std::make_unique<GroupedBf16PlanImpl<Combine,TileN,TileK,SwapAB,TrimTokens,
        Selector,StockScheduler,SmMode>>();
    plan->initialize(params, selector);
    *out = plan.release();
    return cudaSuccess;
  } catch (cudaError_t status) { return status; }
    catch (const std::bad_alloc&) { return cudaErrorMemoryAllocation; }
}

template <bool StockScheduler, int SmMode, class Selector>
cudaError_t create_grouped_explicit_scheduler(const Bf16GroupedGemmParams& params,
    Bf16GroupedGemmPlan** out, const Selector& selector) {
  const auto& p=params.policy;
  if(p.tile_n==128 && p.tile_k==64)
    return create_grouped_tiled_plan<false,128,64,false,true,StockScheduler,Selector,SmMode>(params,out,selector);
  if(p.tile_n==128 && p.tile_k==128)
    return create_grouped_tiled_plan<false,128,128,false,true,StockScheduler,Selector,SmMode>(params,out,selector);
  if(p.tile_n==256 && p.tile_k==64)
    return create_grouped_tiled_plan<false,256,64,false,true,StockScheduler,Selector,SmMode>(params,out,selector);
  if(p.tile_n==256 && p.tile_k==128)
    return create_grouped_tiled_plan<false,256,128,false,true,StockScheduler,Selector,SmMode>(params,out,selector);
  return cudaErrorInvalidValue;
}

template <bool Combine, class Selector = GroupedExplicitPreparePolicy>
cudaError_t create_grouped_plan(const Bf16GroupedGemmParams& params, Bf16GroupedGemmPlan** out,
    const Selector& selector = {}) {
  if(!out) return cudaErrorInvalidValue;
  *out=nullptr;
  const auto& p=params.policy;
  if((p.mma_sm_count!=1 && p.mma_sm_count!=2) ||
      (p.scheduler!=GroupedGemmScheduler::Default &&
       p.scheduler!=GroupedGemmScheduler::Native &&
       p.scheduler!=GroupedGemmScheduler::Cutlass)) return cudaErrorInvalidValue;
  const bool stock=p.scheduler==GroupedGemmScheduler::Cutlass ||
      (p.scheduler==GroupedGemmScheduler::Default && p.mma_sm_count==2);
  if(p.mma_sm_count==2 || stock) {
    // Scheduler identity is independent of the UMMA width. Keep these explicit
    // offline choices out of the native-1SM device-policy template graph.
    if constexpr (Combine || !std::is_same_v<Selector,GroupedExplicitPreparePolicy>) {
      return cudaErrorInvalidValue;
    } else {
      if(p.swap_ab) return cudaErrorInvalidValue;
      if(!stock) return create_grouped_explicit_scheduler<false,2>(params,out,selector);
      if(p.mma_sm_count==1) return create_grouped_explicit_scheduler<true,1>(params,out,selector);
      return create_grouped_explicit_scheduler<true,2>(params,out,selector);
    }
  }
  if(p.swap_ab) {
    if constexpr (Combine) return cudaErrorInvalidValue;
    else {
      if(p.tile_n!=128) return cudaErrorInvalidValue;
      if(!p.trim_swap_tokens) {
        if(p.tile_k==64) return create_grouped_tiled_plan<false,128,64,true,false>(params,out,selector);
        if(p.tile_k==128) return create_grouped_tiled_plan<false,128,128,true,false>(params,out,selector);
        return cudaErrorInvalidValue;
      }
      if(p.tile_k==64) return create_grouped_tiled_plan<false,128,64,true>(params,out,selector);
      if(p.tile_k==128) return create_grouped_tiled_plan<false,128,128,true>(params,out,selector);
      return cudaErrorInvalidValue;
    }
  }
  if(p.tile_n==128 && p.tile_k==64) return create_grouped_tiled_plan<Combine,128,64>(params,out,selector);
  if(p.tile_n==128 && p.tile_k==128) return create_grouped_tiled_plan<Combine,128,128>(params,out,selector);
  if(p.tile_n==256 && p.tile_k==64) return create_grouped_tiled_plan<Combine,256,64>(params,out,selector);
  if(p.tile_n==256 && p.tile_k==128) return create_grouped_tiled_plan<Combine,256,128>(params,out,selector);
  return cudaErrorInvalidValue;
}

}  // namespace detail

}  // namespace fuse
