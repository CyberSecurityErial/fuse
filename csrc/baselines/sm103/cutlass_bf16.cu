// SPDX-License-Identifier: BSD-3-Clause
// Independent pure BF16 GEMM plans; never a fused-operator or cuBLASLt result.
// X[M,K], W[N,K], D[M,N] are contiguous row-major, D = X * W^T.
// create() binds device pointers and prepares descriptors/resources. run() only
// launches the initialized plan, including during CUDA Graph capture. The caller
// must keep all buffers and this plan alive until streams/graphs finish using it.
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/kernel/tile_scheduler.hpp>
#include <cutlass/util/packed_stride.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

thread_local std::string last_error;

void check(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

void check(cutlass::Status status) {
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(cutlass::cutlassGetStatusString(status));
  }
}

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::invalid_argument(message);
  }
}

void check_pointer(const void* pointer, int device) {
  require(pointer && reinterpret_cast<uintptr_t>(pointer) % 16 == 0,
          "GEMM buffers must be non-null and 16-byte aligned");
  cudaPointerAttributes attributes{};
  check(cudaPointerGetAttributes(&attributes, pointer));
  require(attributes.type == cudaMemoryTypeDevice && attributes.device == device,
          "GEMM buffers must belong to the current CUDA device");
}

bool overlaps(const void* a, uint64_t a_bytes, const void* b, uint64_t b_bytes) {
  const auto x = reinterpret_cast<uintptr_t>(a);
  const auto y = reinterpret_cast<uintptr_t>(b);
  require(a_bytes <= std::numeric_limits<uintptr_t>::max() - x &&
          b_bytes <= std::numeric_limits<uintptr_t>::max() - y,
          "GEMM pointer range overflows");
  return x < y + b_bytes && y < x + a_bytes;
}

template <int SmMode, int EpilogueN, int ClusterM = SmMode>
struct Bf16Gemm {
  static_assert(SmMode == 1 || SmMode == 2);
  static_assert(EpilogueN == 32 || EpilogueN == 64);
  static_assert(ClusterM == 1 || ClusterM == 2);
  static_assert(SmMode == 1 || ClusterM == 2);
  static_assert(SmMode != 1 || ClusterM != 2 || EpilogueN == 32);
  using Element = cutlass::bfloat16_t;
  using Tile = cute::Shape<cute::Int<128 * SmMode>, cute::_256, cute::_64>;
  using Cluster = cute::Shape<cute::Int<ClusterM>, cute::_1, cute::_1>;
  // MMA scope and multicast cluster are independent. Keep the explicit 1-SM
  // schedule: Auto would choose 2-SM MMA when ClusterM is even. The optional
  // 1-SM cluster2 path multicasts B, but retains per-CTA A/B storage and TMEM.
  using MainloopSchedule = std::conditional_t<SmMode == 1,
      cutlass::gemm::KernelTmaWarpSpecialized1SmSm100,
      cutlass::gemm::KernelTmaWarpSpecialized2SmSm100>;
  using EpilogueSchedule = std::conditional_t<SmMode == 1,
      cutlass::epilogue::TmaWarpSpecialized1Sm,
      cutlass::epilogue::TmaWarpSpecialized2Sm>;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      Tile, Cluster, cute::Shape<cute::_128, cute::Int<EpilogueN>>, float, float,
      void, cutlass::layout::RowMajor, 8,
      Element, cutlass::layout::RowMajor, 8, EpilogueSchedule>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      Element, cutlass::layout::RowMajor, 8,
      Element, cutlass::layout::ColumnMajor, 8,
      float, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      MainloopSchedule>::CollectiveOp;
  // Do not use the default PersistentScheduler: on Blackwell it selects CLC.
  // The initial A/B changes MMA/cluster organization, not the scheduler family.
  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, Mainloop, Epilogue,
      cutlass::gemm::StaticPersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  static_assert(!Kernel::TileScheduler::IsDynamicPersistent);
  static_assert(cute::size(typename Mainloop::AtomThrShapeMNK{}) == SmMode);
  static_assert(cute::size<0>(typename Mainloop::CtaShape_MNK{}) == 128);
};

// This experiment changes multicast/cluster scheduling, not the MMA tile or
// pipeline depth. These are real CUTLASS types, not assumed runtime resources.
using SingleCtaGemm = Bf16Gemm<1, 32, 1>;
using MulticastGemm = Bf16Gemm<1, 32, 2>;
static_assert(MulticastGemm::Mainloop::DispatchPolicy::Stages == 4);
static_assert(MulticastGemm::Mainloop::DispatchPolicy::Stages ==
              SingleCtaGemm::Mainloop::DispatchPolicy::Stages);
static_assert(MulticastGemm::Kernel::AccumulatorPipelineStageCount == 2);
static_assert(std::is_same_v<typename MulticastGemm::Mainloop::TiledMma,
                             typename SingleCtaGemm::Mainloop::TiledMma>);
static_assert(MulticastGemm::Kernel::SharedStorageSize == SingleCtaGemm::Kernel::SharedStorageSize);
static_assert(MulticastGemm::Kernel::MaxThreadsPerBlock == SingleCtaGemm::Kernel::MaxThreadsPerBlock);

struct Plan {
  int device = -1;
  std::string info;
  virtual ~Plan() = default;
  virtual cutlass::Status run(cudaStream_t stream) = 0;
};

template <int SmMode, int EpilogueN, int ClusterM = SmMode>
struct TypedPlan final : Plan {
  using Types = Bf16Gemm<SmMode, EpilogueN, ClusterM>;
  using Kernel = typename Types::Kernel;
  using Gemm = typename Types::Gemm;
  using Mainloop = typename Types::Mainloop;
  using Epilogue = typename Types::Epilogue;
  static constexpr int ClusterSize = cute::size(typename Types::Cluster{});
  Gemm gemm;

  TypedPlan(int m, int n, int k, int max_swizzle_size,
            const void* lhs, const void* rhs, void* output,
            cudaStream_t stream, int device_id, const cudaDeviceProp& properties, int sm_budget) {
    device = device_id;
    const auto kernel = reinterpret_cast<const void*>(cutlass::device_kernel<Kernel>);
    const dim3 block = Kernel::get_block_shape();
    const int smem = Kernel::SharedStorageSize;
    if (smem >= (48 << 10)) {
      check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    }
    cudaFuncAttributes attributes{};
    check(cudaFuncGetAttributes(&attributes, kernel));
    int active_blocks = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, kernel, block.x * block.y * block.z, smem));
    require(active_blocks >= 1, "GEMM has no resident CTA on this device");

    cutlass::KernelHardwareInfo hardware{};
    hardware.device_id = device;
    // A persistent CTA budget, not an SM affinity partition. Physical L2 stays shared.
    hardware.sm_count = sm_budget == 0 ? properties.multiProcessorCount : sm_budget;
    int active_clusters = 0;
    if constexpr (ClusterSize > 1) {
      require(properties.clusterLaunch && hardware.sm_count % ClusterSize == 0,
              "Clustered GEMM requires cluster launch and a divisible physical SM count");
      cudaLaunchAttribute attribute{};
      attribute.id = cudaLaunchAttributeClusterDimension;
      attribute.val.clusterDim.x = ClusterM;
      attribute.val.clusterDim.y = attribute.val.clusterDim.z = 1;
      cudaLaunchConfig_t config{};
      config.gridDim = dim3(hardware.sm_count, 1, 1);
      config.blockDim = block;
      config.dynamicSmemBytes = smem;
      config.stream = stream;
      config.attrs = &attribute;
      config.numAttrs = 1;
      check(cudaOccupancyMaxActiveClusters(&active_clusters, kernel, &config));
      require(active_clusters > 0, "GEMM has no resident cluster");
      // A physical SM budget is not an SM affinity mask. Limit the persistent
      // grid to at most one CTA per available SM and real cluster residency.
      hardware.max_active_clusters = std::min(active_clusters, hardware.sm_count / ClusterSize);
    }

    const auto stride_a = cutlass::make_cute_packed_stride(
        typename Kernel::StrideA{}, cute::make_shape(m, k, 1));
    const auto stride_b = cutlass::make_cute_packed_stride(
        typename Kernel::StrideB{}, cute::make_shape(n, k, 1));
    const auto stride_d = cutlass::make_cute_packed_stride(
        typename Kernel::StrideD{}, cute::make_shape(m, n, 1));
    using Element = typename Types::Element;
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k, 1},
        {static_cast<const Element*>(lhs), stride_a,
         static_cast<const Element*>(rhs), stride_b},
        {{}, nullptr, stride_d, static_cast<Element*>(output), stride_d}, hardware};
    args.epilogue.thread.alpha = 1.0f;
    args.epilogue.thread.beta = 0.0f;
    args.scheduler.max_swizzle_size = max_swizzle_size;
    using Raster = typename Kernel::TileScheduler::RasterOrderOptions;
    args.scheduler.raster_order = Raster::AlongM;
    check(Gemm::can_implement(args));
    // Static dense GEMM needs no workspace. Fail explicitly if that changes;
    // neither initialization nor run may silently allocate a different path.
    require(Gemm::get_workspace_size(args) == 0, "Unexpected CUTLASS workspace requirement");
    check(gemm.initialize(args, nullptr, stream));
    const auto& scheduler = gemm.params().scheduler;
    const int effective_swizzle_size = 1 << scheduler.log_swizzle_size_;
    // These are padded work coordinates, not the resident launch grid below.
    // CUTLASS stores cluster counts in problem_tiles_m/n_, including for 2-SM.
    const uint64_t padded_ctas_m = uint64_t(scheduler.problem_tiles_m_) * scheduler.cluster_shape_m_;
    const uint64_t padded_ctas_n = uint64_t(scheduler.problem_tiles_n_) * scheduler.cluster_shape_n_;
    require(effective_swizzle_size <= max_swizzle_size &&
            padded_ctas_m * padded_ctas_n * scheduler.problem_tiles_l_ == scheduler.blocks_per_problem_,
            "Lowered scheduler swizzle/padded work grid is inconsistent");
    const dim3 grid = Gemm::get_grid_shape(gemm.params());
    const uint64_t ctas = uint64_t(grid.x) * grid.y * grid.z;
    require(ctas > 0 && ctas <= uint64_t(hardware.sm_count) && grid.x % ClusterM == 0,
            "Actual persistent grid violates the physical SM/cluster budget");
    if constexpr (ClusterSize > 1) {
      require(ctas / ClusterSize <= uint64_t(active_clusters), "Grid exceeds measured cluster residency");
    }

    std::ostringstream text;
    text << "{\"schema\":\"sm103_cutlass_bf16_plan_v1\","
         << "\"backend\":\"cutlass_" << SmMode << "sm\",\"sm_mode\":" << SmMode
         << ",\"cluster_m\":" << ClusterM << ",\"cluster_size\":" << ClusterSize
         << ",\"mma_sm_count\":" << int(cute::size(typename Mainloop::AtomThrShapeMNK{}))
         << ",\"precision\":\"bf16\",\"accumulator\":\"fp32\",\"output_dtype\":\"bf16\","
         << "\"element_c\":\"void\",\"alpha\":1,\"beta\":0,\"layout\":\"row_major_x_wt_d\","
         << "\"scheduler\":\"static_persistent\",\"raster\":\"along_m\",\"max_swizzle_size\":"
         << max_swizzle_size << ",\"effective_swizzle_size\":" << effective_swizzle_size
         << ",\"padded_work_grid_ctas\":[" << padded_ctas_m << ',' << padded_ctas_n << ','
         << scheduler.problem_tiles_l_ << "],\"padded_work_ctas\":" << scheduler.blocks_per_problem_ << ','
         << "\"searched\":false,\"workspace_bytes\":0,\"launch_with_pdl\":false,"
         << "\"distributed_boundary_measured\":false,\"production_matched_reference\":false,"
         << "\"m\":" << m << ",\"n\":" << n << ",\"k\":" << k
         << ",\"mma_tile\":[" << 128 * SmMode << ",256,64],\"physical_cta_tile\":["
         << int(cute::size<0>(typename Mainloop::CtaShape_MNK{})) << ','
         << int(cute::size<1>(typename Mainloop::CtaShape_MNK{})) << ','
         << int(cute::size<2>(typename Mainloop::CtaShape_MNK{})) << "],\"cluster\":[" << ClusterM << ",1,1],"
         << "\"epilogue_n\":" << EpilogueN << ",\"epilogue_tile\":["
         << int(cute::size<0>(typename Epilogue::EpilogueTile{})) << ','
         << int(cute::size<1>(typename Epilogue::EpilogueTile{})) << "],\"ab_stages\":"
         << Mainloop::DispatchPolicy::Stages
         << ",\"accumulator_stages\":" << Kernel::AccumulatorPipelineStageCount
         << ",\"tmem_columns\":" << Kernel::TmemAllocator::Sm100TmemCapacityColumns
         << ",\"epilogue_c_stages\":" << Epilogue::DispatchPolicy::StagesC
         << ",\"epilogue_d_stages\":" << Epilogue::DispatchPolicy::StagesD
         << ",\"device\":" << device << ",\"physical_sm_count\":" << properties.multiProcessorCount
         << ",\"runtime_reported_l2_cache_bytes\":" << properties.l2CacheSize
         << ",\"requested_sm_budget\":" << hardware.sm_count << ",\"sm_affinity\":false,"
         << "\"grid\":[" << grid.x << ',' << grid.y << ',' << grid.z << "],\"grid_ctas\":" << ctas
         << ",\"block\":[" << block.x << ',' << block.y << ',' << block.z << "],"
         << "\"dynamic_smem_bytes\":" << smem << ",\"static_smem_bytes\":" << attributes.sharedSizeBytes
         << ",\"registers_per_thread\":" << attributes.numRegs
         << ",\"local_bytes_per_thread\":" << attributes.localSizeBytes
         << ",\"active_blocks_per_sm\":" << active_blocks << ",\"active_clusters\":";
    if constexpr (ClusterSize > 1) {
      text << active_clusters;
    } else {
      text << "null";
    }
    text << ",\"cutlass_builder_sha256\":\"" << FUSE_SM103_CUTLASS_BUILDER_SHA256 << "\"}";
    info = text.str();
  }

  cutlass::Status run(cudaStream_t stream) override {
    // Uses prepared Params. No initialize/update, descriptor lowering, allocation,
    // synchronization or graph construction occurs in the measured callable.
    return gemm.run(stream, nullptr, false);
  }
};

}  // namespace

extern "C" const char* sm103_cutlass_last_error() {
  return last_error.c_str();
}

extern "C" const char* sm103_cutlass_plan_info(void* handle) {
  if (!handle) {
    last_error = "Null CUTLASS plan";
    return nullptr;
  }
  return static_cast<Plan*>(handle)->info.c_str();
}

extern "C" void sm103_cutlass_destroy(void* handle) {
  delete static_cast<Plan*>(handle);
}

extern "C" int sm103_cutlass_run(void* handle, void* stream) {
  try {
    require(handle != nullptr, "Null CUTLASS plan");
    auto& plan = *static_cast<Plan*>(handle);
    int device = -1;
    check(cudaGetDevice(&device));
    require(device == plan.device, "CUTLASS plan used on a different CUDA device");
    check(plan.run(static_cast<cudaStream_t>(stream)));
    return 0;
  } catch (const std::exception& error) {
    last_error = error.what();
    return -1;
  }
}

// Version the changed private ABI so a new wrapper cannot miscall an old .so.
extern "C" void* sm103_cutlass_create_v5(
    int sm_mode, int max_swizzle_size, int epilogue_n, int cluster_m, int sm_budget,
    int64_t m, int64_t n, int64_t k,
    const void* lhs, const void* rhs, void* output, void* stream_pointer) {
  try {
    require(sm_mode == 1 || sm_mode == 2, "CUTLASS sm_mode must be 1 or 2");
    require(max_swizzle_size == 1 || max_swizzle_size == 2 ||
            max_swizzle_size == 4 || max_swizzle_size == 8,
            "CUTLASS max_swizzle_size must be 1, 2, 4 or 8");
    require(epilogue_n == 32 || epilogue_n == 64, "CUTLASS epilogue_n must be 32 or 64");
    require((sm_mode == 1 && (cluster_m == 1 || cluster_m == 2)) ||
            (sm_mode == 2 && cluster_m == 2), "CUTLASS cluster_m is incompatible with sm_mode");
    require(sm_mode != 1 || cluster_m != 2 || epilogue_n == 32,
            "CUTLASS 1-SM cluster2 comparison requires epilogue_n=32");
    for (const int64_t dimension : {m, n, k}) {
      require(dimension > 0 && dimension <= std::numeric_limits<int>::max(),
              "CUTLASS dimensions must be positive int32 values");
    }
    int device = -1;
    check(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, device));
    require(sm_budget >= 0 && sm_budget <= properties.multiProcessorCount,
            "CUTLASS SM budget must be zero (all) or within the physical SM count");
    require(sm_budget == 0 || sm_budget % cluster_m == 0,
            "CUTLASS SM budget must be divisible by cluster size");
    require(properties.major == 10 && properties.minor == 3,
            "This isolated plan requires Runtime compute capability 10.3");
    for (const void* pointer : {lhs, rhs, static_cast<const void*>(output)}) {
      check_pointer(pointer, device);
    }
    require(!overlaps(lhs, uint64_t(m) * k * 2, output, uint64_t(m) * n * 2) &&
            !overlaps(rhs, uint64_t(n) * k * 2, output, uint64_t(m) * n * 2),
            "CUTLASS output must not alias an input");
    auto stream = static_cast<cudaStream_t>(stream_pointer);
    cudaStreamCaptureStatus capture{};
    check(cudaStreamIsCapturing(stream, &capture));
    require(capture == cudaStreamCaptureStatusNone, "Create CUTLASS plans before graph capture");
    // The epilogue carveout may also change AB stages. Report actual resources
    // for each type instead of treating N32/N64 as guaranteed equal-stage A/B.
    if (sm_mode == 1) {
      if (cluster_m == 2) {
        return new TypedPlan<1, 32, 2>(int(m), int(n), int(k), max_swizzle_size,
                                       lhs, rhs, output, stream, device, properties, sm_budget);
      }
      if (epilogue_n == 32) {
        return new TypedPlan<1, 32>(int(m), int(n), int(k), max_swizzle_size,
                                    lhs, rhs, output, stream, device, properties, sm_budget);
      }
      return new TypedPlan<1, 64>(int(m), int(n), int(k), max_swizzle_size,
                                  lhs, rhs, output, stream, device, properties, sm_budget);
    }
    if (epilogue_n == 32) {
      return new TypedPlan<2, 32>(int(m), int(n), int(k), max_swizzle_size,
                                  lhs, rhs, output, stream, device, properties, sm_budget);
    }
    return new TypedPlan<2, 64>(int(m), int(n), int(k), max_swizzle_size,
                                lhs, rhs, output, stream, device, properties, sm_budget);
  } catch (const std::exception& error) {
    last_error = error.what();
    return nullptr;
  }
}
