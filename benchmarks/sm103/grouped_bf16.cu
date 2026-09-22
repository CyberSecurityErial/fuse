// SPDX-License-Identifier: BSD-3-Clause
// Incremental grouped correctness harness: local ready adapters by default;
// --ep 4|8 validates both complete peer-transport boundaries under Graph replay.
// Neither mode is a tuned performance benchmark.
#include "../../csrc/operators/sm103/detail/grouped/a2a_gemm.cuh"
#include "../../csrc/operators/sm103/detail/grouped/gemm_a2a.cuh"
#include "../../csrc/operators/sm103/api/grouped.cuh"
#include <cutlass/device_kernel.h>
#include <cutlass/util/packed_stride.hpp>
#include <cublas_v2.h>
#include "fuse/operators/semantics/moe/grouped.h"
#include "fused_inputs.cuh"
#include "grouped_validation.cuh"
#include "grouped_measurement.cuh"
#include <map>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>
#include <memory>
#include <cstring>
#include <sstream>
#include <limits>
#include <set>
#include <thread>

// Reuse the published pure-GEMM tuning ABI, not Projection operator internals.
extern "C" {
const char* sm103_last_error();
const char* sm103_plan_info(void*);
void* sm103_create(int,int64_t,int64_t,int64_t,const void*,const void*,void*,
                  const void*,const void*,void*,int,int,int,int,int,int,float);
int sm103_run(void*,const void*,const void*,void*,void*);
void sm103_destroy(void*);
#if FUSE_GROUPED_EXTERNAL
const char* grouped_deepgemm_error();
void* grouped_deepgemm_create(int,const int64_t*,int,int,int,const void*,const void*,int,cudaStream_t);
cudaError_t grouped_deepgemm_launch(void*,cudaStream_t);
cudaError_t grouped_deepgemm_read(void*,void*,int,cudaStream_t);
void grouped_deepgemm_destroy(void*);
#endif
}

using Types = fuse::detail::Bf16GroupedGemmTypes<>;

struct GroupedCudaError : std::runtime_error {
  cudaError_t status;
  explicit GroupedCudaError(cudaError_t value) : std::runtime_error(cudaGetErrorString(value)), status(value) {}
};
static void check(cudaError_t status) {
  if (status != cudaSuccess) throw GroupedCudaError(status);
}
static void check(cublasStatus_t status) {
  if(status==CUBLAS_STATUS_ALLOC_FAILED) throw GroupedCudaError(cudaErrorMemoryAllocation);
  if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS correctness reference failed");
}
template <class T> struct Buffer {
  T* ptr = nullptr;
  size_t size;
  int device = 0;
  bool owned = true;
  explicit Buffer(size_t n, T* external = nullptr) : ptr(external), size(n), owned(!external) {
    check(cudaGetDevice(&device));
    if(owned) check(cudaMalloc(reinterpret_cast<void**>(&ptr), std::max(size_t(1), n) * sizeof(T)));
  }
  Buffer(const Buffer&) = delete;
  ~Buffer() {
    int current = 0; cudaGetDevice(&current); cudaSetDevice(device);
    if(owned) cudaFree(ptr); cudaSetDevice(current);
  }
  void copy(const std::vector<T>& v) {
    if (v.size() > size) throw std::runtime_error("host input exceeds allocated capacity");
    if (v.empty()) return;
    check(cudaMemcpy(ptr, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
};

// Native grouped cuBLAS is one pure-GEMM candidate, not a claim of tuned best.
// Equal M_e values form one group; pointer order is regrouped without copying
// matrices. All geometry arrays stay on host, matrix pointer arrays on device.
struct GroupedNativeReference {
  Buffer<const void*> a, b;
  Buffer<void*> c;
  Buffer<fuse::Bf16> output;
  Buffer<unsigned char> workspace;
  cudaGraph_t graph{};
  int device;
  GroupedNativeReference(cublasHandle_t handle, cudaStream_t stream,
      const std::vector<int64_t>& offsets, int capacity, int n, int k,
      const fuse::Bf16* weights, const fuse::Bf16* inputs, fuse::Bf16* shared_output=nullptr)
      : a(offsets.size()-1),b(offsets.size()-1),c(offsets.size()-1),
        output((offsets.size()-1)*size_t(capacity)*n,shared_output),workspace(32<<20) {
    check(cudaGetDevice(&device));
    check(cudaMemset(output.ptr,0x7f,output.size*sizeof(fuse::Bf16)));
    std::map<int,std::vector<int>> groups;
    for (int e=0;e<int(offsets.size())-1;++e) {
      const int rows=offsets[e+1]-offsets[e];
      if (rows) groups[rows].push_back(e);
    }
    std::vector<const void*> pa,pb; std::vector<void*> pc;
    std::vector<int> m,cols,ks,lda,ldb,ldc,sizes;
    std::vector<float> alpha,beta;
    std::vector<cublasOperation_t> ta,tb;
    for (const auto& group:groups) {
      m.push_back(n); cols.push_back(group.first); ks.push_back(k);
      lda.push_back(k); ldb.push_back(k); ldc.push_back(n);
      sizes.push_back(group.second.size()); alpha.push_back(1); beta.push_back(0);
      ta.push_back(CUBLAS_OP_T); tb.push_back(CUBLAS_OP_N);
      for (int e:group.second) {
        pa.push_back(weights+size_t(e)*n*k);
        pb.push_back(inputs+size_t(e)*capacity*k);
        pc.push_back(output.ptr+size_t(e)*capacity*n);
      }
    }
    a.copy(pa); b.copy(pb); c.copy(pc);
    check(cublasSetStream(handle,stream));
    check(cublasSetWorkspace(handle,workspace.ptr,workspace.size));
    auto launch=[&] {
      if (m.empty()) return;
      check(cublasGemmGroupedBatchedEx(handle,ta.data(),tb.data(),m.data(),cols.data(),ks.data(),
          alpha.data(),a.ptr,CUDA_R_16BF,lda.data(),b.ptr,CUDA_R_16BF,ldb.data(),
          beta.data(),c.ptr,CUDA_R_16BF,ldc.data(),m.size(),sizes.data(),CUBLAS_COMPUTE_32F));
    };
    launch(); check(cudaStreamSynchronize(stream));
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    launch();
    check(cudaStreamEndCapture(stream,&graph));
    check(cublasSetStream(handle,nullptr));
  }
  ~GroupedNativeReference() {
    cudaSetDevice(device);
    if(graph) cudaGraphDestroy(graph);
  }
};

struct GroupedLtPlan {
  void* handle=nullptr;
  ~GroupedLtPlan() { if(handle) sm103_destroy(handle); }
};

// One tuned plan per distinct actual M_e; equal-sized experts reuse it. The
// rank owns this cache across payloads, so payload 1 verifies/replays the same
// selected algorithms instead of selecting a faster repeat. Captured GEMMs run
// sequentially on GPU, with no host launch gap and no padded expert arithmetic.
// This is explicitly an Lt Graph sequence, not a native grouped Lt API.
struct GroupedLtReference {
  Buffer<fuse::Bf16> output;
  cudaGraph_t graph{};
  int device;
  GroupedLtReference(cudaStream_t stream, const std::vector<int64_t>& rows,
      int capacity,int n,int k,const fuse::Bf16* weights,const fuse::Bf16* inputs,
      std::map<int,std::unique_ptr<GroupedLtPlan>>& plans,fuse::Bf16* shared_output=nullptr)
      : output((rows.size()-1)*size_t(capacity)*n,shared_output) {
    check(cudaGetDevice(&device));
    check(cudaMemset(output.ptr,0x7f,output.size*sizeof(fuse::Bf16)));
    for(int e=0;e<int(rows.size())-1;++e) {
      const int m=rows[e+1]-rows[e];
      if(!m || plans.count(m)) continue;
      auto plan=std::make_unique<GroupedLtPlan>();
      plan->handle=sm103_create(16,m,n,k,inputs+size_t(e)*capacity*k,
          weights+size_t(e)*n*k,output.ptr+size_t(e)*capacity*n,nullptr,nullptr,
          stream,32,32,10,50,1,0,0.f);
      if(!plan->handle) throw std::runtime_error(sm103_last_error());
      printf("ALGORITHM grouped rank=%d m=%d n=%d k=%d api=cublasLt_sequence info=%s\n",
          device,m,n,k,sm103_plan_info(plan->handle));
      plans.emplace(m,std::move(plan));
    }
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    int status=0;
    for(int e=0;e<int(rows.size())-1 && !status;++e) {
      const int m=rows[e+1]-rows[e];
      if(m) status=sm103_run(plans.at(m)->handle,inputs+size_t(e)*capacity*k,
          weights+size_t(e)*n*k,output.ptr+size_t(e)*capacity*n,stream);
    }
    const auto end=cudaStreamEndCapture(stream,&graph);
    if(status) {
      if(graph) cudaGraphDestroy(graph);
      graph=nullptr;
      throw std::runtime_error(sm103_last_error());
    }
    check(end);
  }
  ~GroupedLtReference() { cudaSetDevice(device); if(graph) cudaGraphDestroy(graph); }
};

// Same CUTLASS collectives/scheduler as the fused kernels, with pre-delivered
// input and no ready adapters or communication CTAs. Physical device metadata
// stays full-size; only the persistent worker grid uses the requested budget.
template <int TileN=128, int TileK=64, bool Stock=false, bool SwapAB=false,
    bool TrimTokens=true, int SmMode=1>
struct GroupedCutlassReference {
  using Collectives=fuse::detail::Bf16GroupedGemmTypes<
      TileN,TileK,SwapAB,TrimTokens,SmMode>;
  using StockProblem=cutlass::gemm::GroupProblemShape<cute::Shape<int32_t,int32_t,int32_t>>;
  using Kernel=std::conditional_t<Stock,
      cutlass::gemm::kernel::GemmUniversal<StockProblem,typename Collectives::Mainloop,typename Collectives::Epi>,
      typename Collectives::PureGemm>;
  using Shape=fuse::detail::GroupedProblemShape::UnderlyingProblemShape;
  using SA=typename Kernel::InternalStrideA;
  using SB=typename Kernel::InternalStrideB;
  using SD=typename Kernel::InternalStrideD;
  static constexpr int ClusterSize=cute::size(typename Kernel::ClusterShape{});
  Buffer<const fuse::Bf16*> a,b;
  Buffer<fuse::Bf16*> d;
  Buffer<Shape> shapes;
  Buffer<int64_t> tiles;
  Buffer<SA> sa; Buffer<SB> sb; Buffer<SD> sd;
  Buffer<fuse::Bf16> output;
  std::unique_ptr<Buffer<unsigned char>> workspace;
  cudaGraph_t graph{};
  int device;
  GroupedCutlassReference(cudaStream_t stream, const std::vector<int64_t>& rows,
      int capacity, int n, int k, const fuse::Bf16* weights, const fuse::Bf16* inputs,
      int workers, bool along_n, int swizzle, fuse::Bf16* shared_output=nullptr,
      bool host_shapes=true)
      : a(rows.size()-1),b(rows.size()-1),d(rows.size()-1),shapes(rows.size()-1),
        tiles(rows.size()),sa(rows.size()-1),sb(rows.size()-1),sd(rows.size()-1),
        output((rows.size()-1)*size_t(capacity)*n,shared_output) {
    check(cudaGetDevice(&device));
    const int experts=rows.size()-1;
    std::vector<const fuse::Bf16*> pa,pb; std::vector<fuse::Bf16*> pd;
    std::vector<Shape> ps; std::vector<int64_t> pt{0};
    std::vector<SA> va; std::vector<SB> vb; std::vector<SD> vd;
    for(int e=0;e<experts;++e) {
      const int m=rows[e+1]-rows[e];
      const auto* input=inputs+size_t(e)*capacity*k;
      const auto* weight=weights+size_t(e)*n*k;
      pa.push_back(SwapAB?weight:input); pb.push_back(SwapAB?input:weight);
      pd.push_back(output.ptr+size_t(e)*capacity*n);
      ps.push_back(cute::make_shape(SwapAB?n:m,SwapAB?m:n,k));
      pt.push_back(pt.back()+(int64_t(m)+128*SmMode-1)/(128*SmMode));
      va.push_back(cutlass::make_cute_packed_stride(SA{},cute::make_shape(SwapAB?n:capacity,k,1)));
      vb.push_back(cutlass::make_cute_packed_stride(SB{},cute::make_shape(SwapAB?capacity:n,k,1)));
      vd.push_back(cutlass::make_cute_packed_stride(SD{},cute::make_shape(SwapAB?n:capacity,SwapAB?capacity:n,1)));
    }
    a.copy(pa); b.copy(pb); d.copy(pd); shapes.copy(ps); tiles.copy(pt);
    sa.copy(va); sb.copy(vb); sd.copy(vd);
    check(cudaMemset(output.ptr,0x7f,output.size*sizeof(fuse::Bf16)));
    typename Kernel::Arguments args{};
    args.mode=cutlass::gemm::GemmUniversalMode::kGrouped;
    args.problem_shape.num_groups=experts; args.problem_shape.problem_shapes=shapes.ptr;
    // Standalone library/search may use known host shapes to truncate its
    // grid/termination bound. The matched diagnostic must not inherit this
    // shortcut: production accepts device counts that change on Graph replay.
    if constexpr(Stock) if(host_shapes) args.problem_shape.host_problem_shapes=ps.data();
    args.mainloop.ptr_A=a.ptr; args.mainloop.ptr_B=b.ptr;
    args.mainloop.dA=sa.ptr; args.mainloop.dB=sb.ptr;
    args.epilogue.ptr_D=d.ptr; args.epilogue.dD=sd.ptr;
    args.epilogue.thread.alpha=1.f; args.epilogue.thread.beta=0.f;
    cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop,device));
    args.hw_info.device_id=device;
    // The stock scheduler may be used as a same-budget pure-GEMM diagnostic.
    // Our fused scheduler must retain the physical count because its block
    // offset participates in CUTLASS's pointer-array workspace indexing.
    args.hw_info.sm_count=Stock?workers:prop.multiProcessorCount;
    if constexpr(!Stock) {
      args.scheduler.row_tile_offsets=tiles.ptr; args.scheduler.n=n;
      args.scheduler.compute_ctas=workers; args.scheduler.block_offset=0;
    } else if(workers%ClusterSize) {
      throw std::runtime_error("stock CUTLASS budget must contain whole clusters");
    }
    args.scheduler.max_swizzle_size=swizzle;
    using Raster=typename Kernel::TileScheduler::RasterOrderOptions;
    args.scheduler.raster_order=along_n?Raster::AlongN:Raster::AlongM;
    if(workers<=0 || workers>prop.multiProcessorCount || !Kernel::can_implement(args))
      throw std::runtime_error("pure grouped CUTLASS configuration rejected");
    workspace=std::make_unique<Buffer<unsigned char>>(Kernel::get_workspace_size(args));
    if(Kernel::initialize_workspace(args,workspace->ptr,stream)!=cutlass::Status::kSuccess)
      throw std::runtime_error("pure grouped CUTLASS workspace failed");
    auto params=Kernel::to_underlying_arguments(args,workspace->ptr);
    constexpr size_t smem=sizeof(typename Kernel::SharedStorage);
    check(cudaFuncSetAttribute(cutlass::device_kernel<Kernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
    check(cudaStreamSynchronize(stream));
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    if constexpr(ClusterSize==1) {
      cutlass::device_kernel<Kernel><<<Kernel::get_grid_shape(params),
          Kernel::get_block_shape(),smem,stream>>>(params);
    } else {
      cudaLaunchAttribute attribute{};
      attribute.id=cudaLaunchAttributeClusterDimension;
      attribute.val.clusterDim={ClusterSize,1,1};
      cudaLaunchConfig_t config{};
      config.gridDim=Kernel::get_grid_shape(params);
      config.blockDim=Kernel::get_block_shape();
      config.dynamicSmemBytes=smem;
      config.stream=stream;
      config.attrs=&attribute;
      config.numAttrs=1;
      check(cudaLaunchKernelEx(&config,cutlass::device_kernel<Kernel>,params));
    }
    check(cudaGetLastError());
    check(cudaStreamEndCapture(stream,&graph));
  }
  ~GroupedCutlassReference() { cudaSetDevice(device); if(graph) cudaGraphDestroy(graph); }
};

template <class Kernel, int Mode>
void verify_grouped(cublasHandle_t blas, int sm_count, bool along_n, int swizzle, int workers) {
  using Bf16 = fuse::Bf16;
  using Shape = fuse::detail::GroupedProblemShape::UnderlyingProblemShape;
  using SA = typename Kernel::InternalStrideA;
  using SB = typename Kernel::InternalStrideB;
  using SD = typename Kernel::InternalStrideD;
  constexpr int E = 5, capacity = 385, N = 384, K = 64;
  constexpr int mt_capacity = (capacity + 127) / 128;
  Buffer<Bf16> a(E * capacity * K), b(E * N * K), d(E * capacity * N), ref(E * capacity * N);
  Buffer<const Bf16*> ap(E), bp(E);
  Buffer<Bf16*> dp(E);
  Buffer<SA> sa(E); Buffer<SB> sb(E); Buffer<SD> sd(E);
  Buffer<Shape> shapes(E);
  Buffer<int64_t> offsets(E + 1);
  Buffer<uint32_t> flags(E * mt_capacity * (N/128) * fuse::kReadyFlagStride);
  std::vector<Bf16> ha(a.size), hb(b.size);
  std::mt19937 rng(20260916);
  std::uniform_real_distribution<float> random(-0.5f, 0.5f);
  for (auto& x : ha) x = Bf16(random(rng));
  for (auto& x : hb) x = Bf16(random(rng));
  a.copy(ha); b.copy(hb);
  std::vector<const Bf16*> vap(E), vbp(E);
  std::vector<Bf16*> vdp(E);
  std::vector<SA> vsa(E); std::vector<SB> vsb(E); std::vector<SD> vsd(E);
  for (int e = 0; e < E; ++e) {
    vap[e] = a.ptr + e * capacity * K; vbp[e] = b.ptr + e * N * K;
    vdp[e] = d.ptr + e * capacity * N;
    vsa[e] = cutlass::make_cute_packed_stride(SA{}, cute::make_shape(capacity,K,1));
    vsb[e] = cutlass::make_cute_packed_stride(SB{}, cute::make_shape(N,K,1));
    vsd[e] = cutlass::make_cute_packed_stride(SD{}, cute::make_shape(capacity,N,1));
  }
  ap.copy(vap); bp.copy(vbp); dp.copy(vdp); sa.copy(vsa); sb.copy(vsb); sd.copy(vsd);
  typename Kernel::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGrouped;
  args.problem_shape.num_groups = E;
  args.problem_shape.problem_shapes = shapes.ptr;
  args.mainloop.ptr_A = ap.ptr; args.mainloop.ptr_B = bp.ptr;
  args.mainloop.dA = sa.ptr; args.mainloop.dB = sb.ptr;
  args.epilogue.ptr_D = dp.ptr; args.epilogue.dD = sd.ptr;
  args.epilogue.thread.alpha = 1.f; args.epilogue.thread.beta = 0.f;
  args.hw_info.device_id = 0; args.hw_info.sm_count = sm_count;
  args.scheduler.row_tile_offsets = offsets.ptr;
  args.scheduler.n = N; args.scheduler.compute_ctas = workers;
  args.scheduler.max_swizzle_size = swizzle;
  using Raster = typename Kernel::TileScheduler::RasterOrderOptions;
  args.scheduler.raster_order = along_n ? Raster::AlongN : Raster::AlongM;
  if constexpr (Mode == 1) {
    args.mainloop.row_tile_offsets = offsets.ptr;
    args.mainloop.ready = flags.ptr; args.mainloop.epoch = 1;
  }
  if constexpr (Mode == 2) {
    args.epilogue.order = {offsets.ptr, E, N/128, swizzle, along_n};
    args.epilogue.ready = flags.ptr; args.epilogue.epoch = 1;
  }
  if (!Kernel::can_implement(args)) throw std::runtime_error("grouped arguments rejected");
  Buffer<unsigned char> workspace(Kernel::get_workspace_size(args));
  if (Kernel::initialize_workspace(args, workspace.ptr, nullptr) != cutlass::Status::kSuccess)
    throw std::runtime_error("grouped workspace initialization failed");
  auto params = Kernel::to_underlying_arguments(args, workspace.ptr);
  const auto grid = Kernel::get_grid_shape(params);
  constexpr int smem = sizeof(typename Kernel::SharedStorage);
  check(cudaFuncSetAttribute(cutlass::device_kernel<Kernel>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  cudaStream_t stream;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  cutlass::device_kernel<Kernel><<<grid, Kernel::get_block_shape(), smem, stream>>>(params);
  cudaGraph_t graph; cudaGraphExec_t exec;
  check(cudaGetLastError());
  check(cudaStreamEndCapture(stream, &graph));
  check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  int replay = 0;
  for (const std::vector<int>& rows : {std::vector<int>{0,1,129,0,385},
      std::vector<int>{257,0,0,8,127}, std::vector<int>{0,0,0,0,0}}) {
    // Exact same graph and parameter pointers, different GPU M_e each replay.
    std::vector<Shape> hs;
    std::vector<int64_t> ho(1,0);
    for (int m : rows) { hs.push_back(cute::make_shape(m,N,K)); ho.push_back(ho.back()+(m+127)/128); }
    shapes.copy(hs); offsets.copy(ho);
    std::vector<uint32_t> hf(flags.size, Mode == 1 ? 1 : 0);
    flags.copy(hf);
    // Inactive capacity rows are sentinels, not padded GEMM work.
    check(cudaMemset(d.ptr, 0x7f, d.size*sizeof(Bf16)));
    check(cudaGraphLaunch(exec, stream));
    check(cudaStreamSynchronize(stream));
    const float alpha = 1.f, beta = 0.f;
    for (int e = 0; e < E; ++e) if (rows[e])
      check(cublasGemmEx(blas,CUBLAS_OP_T,CUBLAS_OP_N,N,rows[e],K,
          &alpha,vbp[e],CUDA_R_16BF,K,vap[e],CUDA_R_16BF,K,&beta,
          ref.ptr+e*capacity*N,CUDA_R_16BF,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    check(cudaDeviceSynchronize());
    std::vector<Bf16> hd(d.size), hr(ref.size);
    check(cudaMemcpy(hd.data(),d.ptr,d.size*sizeof(Bf16),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(hr.data(),ref.ptr,ref.size*sizeof(Bf16),cudaMemcpyDeviceToHost));
    double max_abs = 0;
    for (int e=0; e<E; ++e) for (int m=0; m<capacity; ++m) for (int n=0; n<N; ++n) {
      const int i=(e*capacity+m)*N+n;
      if (m >= rows[e]) {
        if (hd[i].raw() != 0x7f7f) throw std::runtime_error("GEMM wrote outside expert rows");
      } else {
        const double x=float(hd[i]), y=float(hr[i]), error=std::abs(x-y);
        if (!std::isfinite(x) || error > 0.002 + 0.008*std::abs(y))
          throw std::runtime_error("grouped numeric mismatch against cuBLAS");
        max_abs=std::max(max_abs,error);
      }
    }
    if constexpr (Mode == 2) {
      check(cudaMemcpy(hf.data(),flags.ptr,flags.size*sizeof(uint32_t),cudaMemcpyDeviceToHost));
      for (int64_t i=0; i<int64_t(flags.size); ++i) {
        const bool expected = i % fuse::kReadyFlagStride == 0 &&
            i / fuse::kReadyFlagStride < ho.back()*(N/128);
        if (hf[i] != uint32_t(expected)) throw std::runtime_error("wrong output-ready tile coverage");
      }
    }
    printf("CHECK grouped-local mode=%d along_n=%d swizzle=%d compute=%d replay=%d max_abs=%.6g passed\n",
        Mode,along_n,swizzle,workers,replay++,max_abs);
    fflush(stdout);
  }
  check(cudaGraphExecDestroy(exec)); check(cudaGraphDestroy(graph)); check(cudaStreamDestroy(stream));
}

struct GroupedWorkload {
  int experts = 3, tokens = 257, topk = 2, h = 64, f = 192;
  bool boundary_checks = true;
  const char* timing_csv = nullptr;
  int tile_n = 128, tile_k = 64;
  bool gemm_search = false;
  bool external_search = false;
  const char* trace_out = nullptr;
  bool fused_only = false;
  bool transport_compare = false;
  bool compute_compare = false;
  int buffer_rows = 0;
  bool balance_tail = false, hot_half = false;
  bool ready_summary = false;
  bool swap_ab = false;
  bool trim_swap_tokens = true;
  int mma_sm_count = 1;
  fuse::GroupedGemmScheduler scheduler = fuse::GroupedGemmScheduler::Default;
  fuse::GroupedDispatchCopy dispatch_copy = fuse::GroupedDispatchCopy::CpAsync;
  int64_t property_seed = -1;  // Validation only; never changes formal samples.
  bool property_dynamic = false;  // Validation-only device policy probe.
  bool stock_scheduler() const {
    return scheduler==fuse::GroupedGemmScheduler::Cutlass ||
        (scheduler==fuse::GroupedGemmScheduler::Default && mma_sm_count==2);
  }
};

// This selector is intentionally a validation probe, not the production
// performance model. It makes several legal decisions from the current
// device-resident route counts so Graph replay tests exercise the dynamic
// parameter path. Production thresholds must come from physical calibration.
struct GroupedPropertySelector {
  fuse::detail::GroupedInvocationPolicy* selected = nullptr;

  CUTLASS_DEVICE fuse::detail::GroupedInvocationPolicy operator()(
      const fuse::detail::GroupedPrepareParams& p,
      fuse::detail::GroupedInvocationPolicy initial) const {
    int64_t panels = p.row_tile_offsets[p.experts];
    int64_t max_rows = 0;
    int active = 0;
    for (int e = 0; e < p.experts; ++e) {
      const int64_t rows = p.row_offsets[e + 1] - p.row_offsets[e];
      max_rows = rows > max_rows ? rows : max_rows;
      active += rows != 0;
    }
    auto chosen = initial;
    if (panels == 0) chosen = {8, 140, 1, false};
    else if (max_rows < 128) chosen = {40, 108, 4, true};
    else if (max_rows < 192 && active > 1) chosen = {64, 84, 8, true};
    else chosen = {20, 128, 4, true};
    chosen.borrow_idle_compute_ctas = panels > 0;
    if (selected) *selected = chosen;
    return chosen;
  }
};

template <bool IsCombine>
struct GroupedRank {
  using Bf16 = fuse::Bf16;
  using Source = fuse::GroupedTokenSource;
  using Spec = std::conditional_t<IsCombine, fuse::moe::CombineForwardSpec,
      fuse::moe::DispatchForwardSpec>;
  const int E, T, Top, N, K;
  int rank, world, capacity, buffer_rows;
  bool property_dynamic;
  int mma_sm_count;
  fuse::GroupedGemmScheduler scheduler;
  bool stock_scheduler;
  fuse::GroupedDispatchCopy dispatch_copy;
  Buffer<Bf16> input, a, b, d, output, ref, oracle_a;
  Buffer<const Bf16*> bp;
  Buffer<Bf16*> dp, stage_ap;
  Buffer<int64_t> offsets;
  Buffer<Source> routes;
  Buffer<uint32_t> started, done;
  Buffer<fused_inputs::Scratch> random_scratch;
  Buffer<grouped_validation::Result> validation;
  Buffer<grouped_validation::BranchOwner> branch_owners;
  Buffer<fuse::detail::GroupedInvocationPolicy> selected_policy;
  fuse::Bf16GroupedGemmPlan* plan = nullptr;
  cudaStream_t stream{};
  cudaGraph_t graph{};
  cudaGraphExec_t exec{};
  cublasHandle_t blas{};
  std::map<int,std::unique_ptr<GroupedLtPlan>> lt_plans;
  std::vector<int64_t> host_offsets;
  std::vector<Source> host_routes;
  std::vector<Bf16> host_input, host_a, host_d, host_ref;
#if FUSE_ENABLE_PROFILING
  std::unique_ptr<Buffer<fuse::GroupedPanelTimeline>> profile_panels;
  std::unique_ptr<Buffer<fuse::GroupedTileTimeline>> profile_tiles;
  std::unique_ptr<Buffer<fuse::GroupedRoleTimeline>> profile_roles;
  std::unique_ptr<Buffer<fuse::GroupedReadySummary>> profile_ready;
  std::unique_ptr<Buffer<fuse::GroupedCommSummary>> profile_comm;
#endif

  GroupedRank(int r, int w, const GroupedWorkload& q)
      : E(q.experts), T(q.tokens), Top(q.topk),
        N(IsCombine ? q.h : 2*q.f), K(IsCombine ? q.f : q.h),
        rank(r), world(w),
        capacity(q.boundary_checks ? w*T : int((int64_t(T)*Top*(q.hot_half?2:1)+E-1)/E)),
        buffer_rows(!IsCombine && q.buffer_rows && q.buffer_rows<capacity ? q.buffer_rows : capacity),
      property_dynamic(q.property_dynamic),mma_sm_count(q.mma_sm_count),scheduler(q.scheduler),
      stock_scheduler(q.stock_scheduler()),dispatch_copy(q.dispatch_copy),
      input(size_t(T)*K), a(size_t(E)*buffer_rows*K), b(size_t(E)*N*K), d(size_t(E)*capacity*N),
      output(size_t(T)*Top*N), ref(size_t(E)*capacity*N), oracle_a(size_t(E)*capacity*K),
      bp(E),dp(E),stage_ap(E),offsets(E+1),routes(size_t(E)*capacity),
      started(w*fuse::kReadyFlagStride),done(w*fuse::kReadyFlagStride),
      random_scratch(1),validation(1),branch_owners(size_t(T)*Top),selected_policy(1) {
    try {
#if FUSE_ENABLE_PROFILING
    if (q.trace_out) {
      const size_t panels=size_t(E)*((capacity+127)/128), nt=(N+q.tile_n-1)/q.tile_n;
      if (q.ready_summary)
        profile_ready=std::make_unique<Buffer<fuse::GroupedReadySummary>>(148);
      else {
        profile_panels=std::make_unique<Buffer<fuse::GroupedPanelTimeline>>(panels);
        profile_tiles=std::make_unique<Buffer<fuse::GroupedTileTimeline>>(panels*nt);
        profile_comm=std::make_unique<Buffer<fuse::GroupedCommSummary>>(148*8);
      }
      profile_roles=std::make_unique<Buffer<fuse::GroupedRoleTimeline>>(148);
      clear_profile();
    }
#endif
    check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    check(cublasCreate(&blas));
    check(cudaMemset(started.ptr,0,started.size*sizeof(uint32_t)));
    check(cudaMemset(done.ptr,0,done.size*sizeof(uint32_t)));
    std::vector<const Bf16*> vb(E);
    std::vector<Bf16*> vd(E),vstage(E);
    for(int e=0;e<E;++e) {
      vb[e]=b.ptr+size_t(e)*N*K; vd[e]=d.ptr+size_t(e)*capacity*N;
      vstage[e]=a.ptr+size_t(e)*buffer_rows*K;
    }
    bp.copy(vb); dp.copy(vd); stage_ap.copy(vstage);
    } catch(...) {
      if(blas) cublasDestroy(blas);
      if(stream) cudaStreamDestroy(stream);
      throw;
    }
  }
  ~GroupedRank() {
    cudaSetDevice(rank);
    if(exec) cudaGraphExecDestroy(exec);
    if(graph) cudaGraphDestroy(graph);
    fuse::destroy_bf16_grouped_gemm(plan);
    cublasDestroy(blas); cudaStreamDestroy(stream);
  }

  void randomize(Buffer<Bf16>& buffer, uint32_t seed, const char* name, int payload) {
    check(fused_inputs::generate(reinterpret_cast<uint16_t*>(buffer.ptr), buffer.size,
        seed, 0.5f, random_scratch.ptr, nullptr));
    fused_inputs::Stats stats{};
    check(cudaMemcpy(&stats, &random_scratch.ptr->result, sizeof(stats), cudaMemcpyDeviceToHost));
    if (stats.count != buffer.size || stats.finite != stats.count ||
        stats.nonzero == 0 || stats.square_sum <= 0)
      throw std::runtime_error("invalid random grouped payload");
    printf("INPUT grouped rank=%d payload=%d tensor=%s seed=%u count=%llu nonzero=%llu min=%.6g max=%.6g sum=%.9g square_sum=%.9g generator=philox-v1\n",
        rank,payload,name,seed,(unsigned long long)stats.count,(unsigned long long)stats.nonzero,
        stats.minimum,stats.maximum,stats.sum,stats.square_sum);
  }

  double validate_gpu(const std::vector<std::unique_ptr<GroupedRank>>& ranks) {
    check(cudaSetDevice(rank));
    check(cudaMemset(validation.ptr,0,sizeof(grouped_validation::Result)));
    if(buffer_rows<capacity)
      grouped_validation::check_bounded_lhs<<<256,256>>>(a.ptr,oracle_a.ptr,E,capacity,
          buffer_rows,K,offsets.ptr,validation.ptr);
    else grouped_validation::check_lhs<<<256,256>>>(a.ptr,oracle_a.ptr,a.size,validation.ptr);
    grouped_validation::check_gemm<<<256,256>>>(d.ptr,ref.ptr,E,capacity,N,offsets.ptr,validation.ptr);
    if constexpr (IsCombine) {
      grouped_validation::Peers outputs{};
      for (int r=0;r<world;++r) outputs.data[r]=ranks[r]->d.ptr;
      grouped_validation::check_branches<<<256,256>>>(output.ptr,output.size,
          branch_owners.ptr,outputs,capacity,N,validation.ptr);
    }
    check(cudaGetLastError());
    grouped_validation::Result result{};
    check(cudaMemcpy(&result,validation.ptr,sizeof(result),cudaMemcpyDeviceToHost));
    if (result.numeric_errors || result.route_errors || result.tail_errors)
      throw std::runtime_error("GPU grouped numeric/route/tail validation failed: " +
          std::to_string(result.numeric_errors) + "/" + std::to_string(result.route_errors) +
          "/" + std::to_string(result.tail_errors));
    return result.max_abs;
  }

  void capture(const std::vector<std::unique_ptr<GroupedRank>>& ranks, bool along_n, int sw,
      int comm, int compute, int tile_n, int tile_k, bool balance_tail=false, bool swap_ab=false,
      bool trim_swap_tokens=true) {
    check(cudaSetDevice(rank));
    fuse::Bf16GroupedGemmParams p{};
    p.lhs = stage_ap.ptr; p.weight_nt = bp.ptr; p.output = dp.ptr;
    p.row_offsets = offsets.ptr; p.source = routes.ptr;
    p.experts = E; p.expert_row_capacity = capacity; p.n = N; p.k = K;
    p.rank = rank; p.world_size = world; p.topk = Top;
    p.policy = {comm, compute, sw, along_n, tile_n, tile_k};
    p.policy.balance_dispatch_tail=balance_tail;
    p.policy.swap_ab=swap_ab;
    p.policy.trim_swap_tokens=trim_swap_tokens;
    p.policy.mma_sm_count=mma_sm_count;
    p.policy.scheduler=scheduler;
    p.policy.dispatch_copy=dispatch_copy;
    p.dispatch_buffer_rows=buffer_rows<capacity ? buffer_rows : 0;
#if FUSE_ENABLE_PROFILING
    if (profile_panels)
      p.profile={profile_panels->ptr,profile_tiles->ptr,profile_roles->ptr,
          int64_t(profile_panels->size),(N+tile_n-1)/tile_n};
    if (profile_comm) {
      p.profile.comm_summary=profile_comm->ptr;
      p.profile.comm_summary_capacity=int32_t(profile_comm->size);
    }
    if (profile_ready) {
      p.profile.roles=profile_roles->ptr;
      p.profile.ready_summary=profile_ready->ptr;
    }
#endif
    for (int r = 0; r < world; ++r) {
      p.peer_input[r] = ranks[r]->input.ptr;
      p.peer_output[r] = ranks[r]->output.ptr;
      p.peer_started[r] = ranks[r]->started.ptr;
      p.peer_done[r] = ranks[r]->done.ptr;
    }
    if constexpr (!IsCombine) {
      if (property_dynamic) {
        GroupedPropertySelector selector{selected_policy.ptr};
        check(fuse::detail::create_grouped_plan<false>(p, &plan, selector));
      } else check(Spec::create(p, &plan));
    } else {
      if (property_dynamic)
        throw std::runtime_error("dynamic property policy only supports Dispatch");
      check(Spec::create(p, &plan));
    }
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    check(Spec::launch(plan, stream));
    check(cudaStreamEndCapture(stream,&graph));
    check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
  }
#if FUSE_ENABLE_PROFILING
  void clear_profile() {
    if (profile_panels) check(cudaMemset(profile_panels->ptr,0,profile_panels->size*sizeof(fuse::GroupedPanelTimeline)));
    if (profile_tiles) check(cudaMemset(profile_tiles->ptr,0,profile_tiles->size*sizeof(fuse::GroupedTileTimeline)));
    check(cudaMemset(profile_roles->ptr,0,profile_roles->size*sizeof(fuse::GroupedRoleTimeline)));
    if (profile_ready) check(cudaMemset(profile_ready->ptr,0,profile_ready->size*sizeof(fuse::GroupedReadySummary)));
    if (profile_comm) check(cudaMemset(profile_comm->ptr,0,profile_comm->size*sizeof(fuse::GroupedCommSummary)));
  }
  void dump_profile(const char* prefix,int comm,int compute,int tile_n,int tile_k,int sw,bool along) {
    const int tile_m=128*mma_sm_count;
    if (profile_ready) {
      std::vector<fuse::GroupedRoleTimeline> roles(comm+compute);
      std::vector<fuse::GroupedReadySummary> ready(comm+compute);
      check(cudaMemcpy(roles.data(),profile_roles->ptr,roles.size()*sizeof(roles[0]),cudaMemcpyDeviceToHost));
      check(cudaMemcpy(ready.data(),profile_ready->ptr,ready.size()*sizeof(ready[0]),cudaMemcpyDeviceToHost));
      std::ofstream out(std::string(prefix)+"-rank-"+std::to_string(rank)+".json");
      out<<"{\"schema\":\"grouped-ready-summary-v1\",\"rank\":"<<rank<<",\"world\":"<<world
         <<",\"n\":"<<N<<",\"k\":"<<K<<",\"tile_n\":"<<tile_n<<",\"tile_k\":"<<tile_k
         <<",\"swizzle\":"<<sw<<",\"along_n\":"<<int(along)<<",\"comm\":"<<comm<<",\"compute\":"<<compute
         <<",\"tile_m\":"<<tile_m<<",\"mma_sm_count\":"<<mma_sm_count
         <<",\"stock_scheduler\":"<<int(stock_scheduler)
         <<",\"warmup\":10,\"payload\":1,\"epoch\":13,\"rows\":[";
      for(int e=0;e<E;++e) out<<(e?",":"")<<host_offsets[e+1]-host_offsets[e];
      out<<"],\"ctas\":[";
      for(int c=0;c<comm+compute;++c) {
        const auto r=roles[c]; const auto s=ready[c];
        out<<(c?",":"")<<"["<<r.begin<<","<<r.role_end<<","<<r.end<<","
           <<s.checks<<","<<s.wait_ns<<","<<s.max_wait_ns<<","<<s.first_wait_ns<<","<<s.waits_ge_1us<<"]";
      }
      // Separate extension preserves the established ready-counter schema.
      // Physical-CTA indices also identify producers borrowed from idle GEMM.
      out<<"],\"startup\":[";
      for(int c=0;c<comm+compute;++c) {
        const auto r=roles[c]; const auto s=ready[c];
        out<<(c?",":"")<<"["<<r.first_release_begin<<","<<r.first_release_end
           <<","<<s.first_wait_begin<<","<<s.first_observed<<"]";
      }
      out<<"],\"startup_bytes\":[";
      for(int c=0;c<comm+compute;++c) {
        uint64_t bytes=0,remote=0;
        int64_t panel=roles[c].first_release_panel;
        if(roles[c].first_release_begin) {
          for(int e=0;e<E;++e) {
            const int64_t rows=host_offsets[e+1]-host_offsets[e];
            const int64_t count=(rows+tile_m-1)/tile_m;
            if(panel>=count) {panel-=count;continue;}
            for(int64_t m=panel*tile_m;m<std::min(rows,(panel+1)*tile_m);++m) {
              bytes+=uint64_t(K)*sizeof(Bf16);
              if(host_routes[host_offsets[e]+m].rank!=rank) remote+=uint64_t(K)*sizeof(Bf16);
            }
            break;
          }
        }
        out<<(c?",":"")<<"["<<bytes<<","<<remote<<"]";
      }
      out<<"]}";
      if(!out) throw std::runtime_error("cannot write grouped ready summary");
      return;
    }
    std::vector<fuse::GroupedPanelTimeline> panels(profile_panels->size);
    std::vector<fuse::GroupedTileTimeline> tiles(profile_tiles->size);
    std::vector<fuse::GroupedRoleTimeline> roles(profile_roles->size);
    std::vector<fuse::GroupedCommSummary> comm_warps(profile_comm->size);
    check(cudaMemcpy(panels.data(),profile_panels->ptr,panels.size()*sizeof(panels[0]),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(tiles.data(),profile_tiles->ptr,tiles.size()*sizeof(tiles[0]),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(roles.data(),profile_roles->ptr,roles.size()*sizeof(roles[0]),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(comm_warps.data(),profile_comm->ptr,
        comm_warps.size()*sizeof(comm_warps[0]),cudaMemcpyDeviceToHost));
    int64_t actual=0;
    for(int e=0;e<E;++e) actual+=(host_offsets[e+1]-host_offsets[e]+tile_m-1)/tile_m;
    const int nt=(N+tile_n-1)/tile_n;
    std::ofstream out(std::string(prefix)+"-rank-"+std::to_string(rank)+".json");
    out<<"{\"schema\":\"grouped-handoff-v1\",\"rank\":"<<rank<<",\"world\":"<<world
       <<",\"tile_m\":"<<tile_m<<",\"mma_sm_count\":"<<mma_sm_count
       <<",\"stock_scheduler\":"<<int(stock_scheduler)
       <<",\"n\":"<<N<<",\"k\":"<<K<<",\"tile_n\":"<<tile_n<<",\"tile_k\":"<<tile_k
       <<",\"swizzle\":"<<sw<<",\"along_n\":"<<int(along)<<",\"comm\":"<<comm<<",\"compute\":"<<compute
       <<",\"warmup\":10,\"payload\":1,\"epoch\":13,\"rows\":[";
    for(int e=0;e<E;++e) out<<(e?",":"")<<host_offsets[e+1]-host_offsets[e];
    out<<"],\"roles\":[";
    for(int c=0;c<comm+compute;++c) {auto x=roles[c];out<<(c?",":"")<<"["<<x.begin<<","<<x.role_end<<","<<x.end<<"]";}
    out<<"],\"panels\":[";
    for(int64_t i=0;i<actual;++i) {auto x=panels[i];out<<(i?",":"")<<"["<<x.begin<<","<<x.release_begin<<","<<x.release_end<<","<<x.cta<<","<<x.expert<<","<<x.m<<"]";}
    out<<"],\"tiles\":[";
    for(int64_t i=0;i<actual*nt;++i) {auto x=tiles[i];out<<(i?",":"")<<"["<<x.wait_begin<<","<<x.observed<<","<<x.load_begin<<","<<x.load_end<<","<<x.cta<<","<<x.expert<<","<<x.m<<","<<x.n<<","<<x.polled<<"]";}
    out<<"],\"comm_warps\":[";
    for(int c=0;c<comm*8;++c) {auto x=comm_warps[c];out<<(c?",":"")<<"["
        <<x.panels<<","<<x.batches<<","<<x.bytes<<","<<x.address_ns<<","<<x.store_wait_ns
        <<","<<x.g2s_ns<<","<<x.store_issue_ns<<","<<x.g2s_issue_ns<<","<<x.g2s_wait_ns<<"]";}
    out<<"]}\n";
    if(!out) throw std::runtime_error("cannot write grouped profile");
  }
#endif
};

template <bool IsCombine, int TileN, int SmMode = 1>
__global__ __launch_bounds__(256,1) void grouped_transport_only(
    fuse::detail::GroupedCommArguments args) {
  using Comm=std::conditional_t<IsCombine,fuse::detail::GroupedCombineComm<TileN>,
      fuse::detail::GroupedDispatchComm<128 * SmMode>>;
  extern __shared__ char storage[];
  Comm{}(args,storage,blockIdx.x,gridDim.x);
}

// Diagnostic only: same transport body, CTA budget and reserved SMEM, with
// every input already available. No GEMM or ready stall, no invocation entry/
// exit handshake. Max-rank timing and the existing writer fence cover all peer
// writes; all rank streams finish before independent output validation. When
// Dispatch shares panels, its diagnostic includes a graph memset of the arrival
// counters; production folds that reset into its existing preparation node.
template <bool IsCombine, int TileN, int TileK, int SmMode = 1,
    bool StockScheduler = SmMode == 2>
struct GroupedTransportReference {
  static constexpr int TileM = 128 * SmMode;
  Buffer<const fuse::Bf16*> inputs;
  Buffer<fuse::Bf16*> outputs;
  Buffer<int32_t> first_wave{1};
  Buffer<int32_t> effective_comm_storage{1};
  Buffer<uint32_t> ready, arrivals;
  cudaGraph_t graph{};
  int device;
  GroupedTransportReference(const std::vector<std::unique_ptr<GroupedRank<IsCombine>>>& ranks,
      int rank, const Buffer<int64_t>& tiles, int comm, int compute,
      bool along_n, int sw, bool balance_tail=false)
      : inputs(std::max(ranks.size(),size_t(ranks[rank]->E))),
        outputs(inputs.size), ready((tiles.size-1)*
            size_t((ranks[rank]->capacity+TileM-1)/TileM)*
            (IsCombine?(ranks[rank]->N+TileN-1)/TileN:1)*fuse::kReadyFlagStride),
        arrivals(IsCombine?0:(tiles.size-1)*size_t((ranks[rank]->capacity+TileM-1)/TileM)) {
    check(cudaGetDevice(&device));
    const auto& r=*ranks[rank];
    std::vector<const fuse::Bf16*> pi;
    std::vector<fuse::Bf16*> po;
    if constexpr(IsCombine) {
      for(int e=0;e<r.E;++e) pi.push_back(r.d.ptr+size_t(e)*r.capacity*r.N);
      for(const auto& peer:ranks) po.push_back(peer->output.ptr);
    } else {
      for(const auto& peer:ranks) pi.push_back(peer->input.ptr);
      for(int e=0;e<r.E;++e) po.push_back(r.a.ptr+size_t(e)*r.capacity*r.K);
    }
    inputs.copy(pi); outputs.copy(po);
    // A constant diagnostic epoch, separate from production invocation state.
    ready.copy(std::vector<uint32_t>(ready.size,1));
    fuse::detail::GroupedCommArguments args{};
    auto& p=args.params;
    p.order={tiles.ptr,r.E,(r.N+TileN-1)/TileN,sw,along_n};
    p.order=fuse::detail::grouped_consumer_order(p.order,StockScheduler);
    p.balance_tail=balance_tail;
    p.cp_async_g2s=r.dispatch_copy==fuse::GroupedDispatchCopy::CpAsync;
    p.row_offsets=r.offsets.ptr; p.source=r.routes.ptr;
    p.input=inputs.ptr; p.output=outputs.ptr; p.ready=ready.ptr;
    int effective_comm=comm;
    int64_t panels=0;
    if constexpr(!IsCombine) {
      p.arrivals=arrivals.ptr;
      std::vector<int64_t> prefix(1,0);
      for(int e=0;e<r.E;++e)
        prefix.push_back(prefix.back()+(r.host_offsets[e+1]-r.host_offsets[e]+TileM-1)/TileM);
      panels=prefix.back();
      auto host_order=p.order; host_order.row_tile_offsets=prefix.data();
      const int32_t first=fuse::detail::grouped_use_latency_cohort(
          r.host_offsets.data(),r.E)
          ? fuse::detail::grouped_first_wave_panels(host_order,compute/SmMode) : -1;
      first_wave.copy(std::vector<int32_t>{first}); p.first_wave_panels=first_wave.ptr;
      effective_comm_storage.copy(std::vector<int32_t>{effective_comm});
      p.effective_comm_ctas=effective_comm_storage.ptr;
    }
    p.columns=IsCombine?r.N:r.K; p.topk=r.Top; p.rank=rank;
    p.world_size=r.world; p.num_comm_ctas=effective_comm; p.epoch=1;
    using Types=std::conditional_t<StockScheduler,
        fuse::detail::Bf16GroupedStockGemmTypes<TileN,TileK,SmMode>,
        fuse::detail::Bf16GroupedGemmTypes<TileN,TileK,false,true,SmMode>>;
    using Fused=std::conditional_t<IsCombine,fuse::detail::GroupedGemmA2A<TileN,TileK>,
        fuse::detail::GroupedMonolithicGemm<typename Types::DispatchGemm,
            fuse::detail::GroupedDispatchComm<TileM>,fuse::detail::GroupedExplicitPreparePolicy,
            StockScheduler || SmMode==2>>;
    constexpr size_t smem=Fused::SharedStorageSize;
    check(cudaFuncSetAttribute(grouped_transport_only<IsCombine,TileN,SmMode>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
    check(cudaStreamBeginCapture(r.stream,cudaStreamCaptureModeThreadLocal));
    if constexpr(!IsCombine) {
      int32_t first=0; check(cudaMemcpy(&first,first_wave.ptr,sizeof(first),cudaMemcpyDeviceToHost));
      if(balance_tail || fuse::detail::grouped_dispatch_splits(
          panels,effective_comm,first,int64_t(TileM)*r.K*sizeof(fuse::Bf16),TileM)>1)
        check(cudaMemsetAsync(arrivals.ptr,0,arrivals.size*sizeof(uint32_t),r.stream));
    }
    grouped_transport_only<IsCombine,TileN,SmMode><<<effective_comm,256,smem,r.stream>>>(args);
    check(cudaGetLastError()); check(cudaStreamEndCapture(r.stream,&graph));
  }
  ~GroupedTransportReference() { cudaSetDevice(device); if(graph) cudaGraphDestroy(graph); }
};

template <bool IsCombine>
void validate_grouped_reference(GroupedRank<IsCombine>& r,const fuse::Bf16* data) {
  check(cudaSetDevice(r.rank));
  check(cudaMemset(r.validation.ptr,0,sizeof(grouped_validation::Result)));
  grouped_validation::check_gemm<<<256,256>>>(data,r.ref.ptr,
      r.E,r.capacity,r.N,r.offsets.ptr,r.validation.ptr);
  check(cudaGetLastError());
  grouped_validation::Result result{};
  check(cudaMemcpy(&result,r.validation.ptr,sizeof(result),cudaMemcpyDeviceToHost));
  if(result.numeric_errors || result.tail_errors)
    throw std::runtime_error("pure GEMM disagrees with independent reference");
}

template <bool IsCombine>
void write_grouped_samples(std::ostream& raw,
    const std::vector<std::unique_ptr<GroupedRank<IsCombine>>>& ranks,
    const GroupedWorkload& q,int replay,const char* mode,
    const grouped_measurement::Samples& result,const fuse::GroupedGemmPolicy& p) {
  const int world=ranks.size();
  for(size_t round=0;round<result.rounds.size();++round) {
    const auto& observed=result.rounds[round];
    const bool accepted=round+1==result.rounds.size() && observed.drift<=0.05;
    for(size_t sample=0;sample<observed.ranks_ms.size();++sample)
      for(int rank=0;rank<world;++rank)
        raw<<(IsCombine?"combine":"dispatch")<<','<<q.h<<','<<q.f<<','<<q.experts*world
           <<','<<q.topk<<','<<q.tokens<<','<<world<<','<<replay<<','<<mode<<','<<sample
           <<','<<rank<<','<<ranks[rank]->host_offsets.back()<<','<<observed.ranks_ms[sample][rank]
           <<','<<p.num_comm_ctas<<','<<p.num_compute_ctas<<','<<p.swizzle<<','<<p.along_n
           <<','<<observed.warmup<<','<<observed.drift<<','<<round<<','<<accepted
           <<','<<p.tile_n<<','<<p.tile_k<<','<<ranks[rank]->buffer_rows
           <<','<<q.balance_tail<<','<<q.hot_half<<','<<q.swap_ab<<','<<q.trim_swap_tokens
           <<','<<int(q.dispatch_copy)<<','<<int(q.scheduler)<<'\n';
  }
  if(!raw) throw std::runtime_error("failed to write grouped sample CSV");
}

// Bounded compute-only grid: four tiles x two rasters x four swizzles. Reuse
// payload, routes, reference and CUDA contexts; no Lt retuning or communication
// timing per candidate. Both payloads independently validate the same grid.
template <bool IsCombine,int TileN,int TileK,bool Stock=false,int SmMode=1>
void search_grouped_tile(std::vector<std::unique_ptr<GroupedRank<IsCombine>>>& ranks,
    const GroupedWorkload& q,int replay,int compute,int& candidate,double& best) {
  for(bool along:{false,true}) for(int sw:{1,2,4,8}) {
    // Pinned stock grouped scheduling linearizes the aggregate N extent to
    // one: requested sw=2/4/8 all resolve to effective sw=1. Search that
    // kernel once; retain the historical external-reference grid unchanged.
    if constexpr(Stock) if(!q.external_search && sw!=1) continue;
    std::vector<std::unique_ptr<GroupedCutlassReference<
        TileN,TileK,Stock,false,true,SmMode>>> refs;
    std::vector<grouped_measurement::Operation> ops;
    double flops=0;
    for(auto& r:ranks) {
      check(cudaSetDevice(r->rank));
      refs.push_back(std::make_unique<GroupedCutlassReference<
          TileN,TileK,Stock,false,true,SmMode>>(r->stream,
          r->host_offsets,r->capacity,r->N,r->K,r->b.ptr,r->oracle_a.ptr,compute,along,sw));
      ops.push_back({r->rank,r->stream,refs.back()->graph});
      flops+=2.*r->host_offsets.back()*r->N*r->K;
    }
    { grouped_measurement::Timer initial(ops); initial.sample(); }
    for(auto& r:ranks) validate_grouped_reference(*r,refs[r->rank]->output.ptr);
    // Match the isolated winner replay's clock warmup for every family.
    // Skipping it only for stock kernels changes duty-cycle/clock history
    // between search and replay. Formal samples remain one operation, 10+50.
    const auto result=grouped_measurement::measure(ops);
    for(auto& r:ranks) validate_grouped_reference(*r,refs[r->rank]->output.ptr);
    std::ofstream raw(q.timing_csv,std::ios::app); raw<<std::setprecision(9);
    const char* mode=q.external_search?"cutlass_stock":!Stock?(SmMode==1?"cutlass_search":"cutlass_search_2sm"):
        (SmMode==1?"cutlass_stock_1sm_budget":"cutlass_stock_2sm_budget");
    write_grouped_samples(raw,ranks,q,replay,mode,result,{0,compute,sw,along,TileN,TileK});
    if(result.drift<=.05) best=std::min(best,result.p50);
    using Kernel=typename GroupedCutlassReference<TileN,TileK,Stock,false,true,SmMode>::Kernel;
    cudaFuncAttributes resources{};
    check(cudaFuncGetAttributes(&resources,cutlass::device_kernel<Kernel>));
    printf("RESULT grouped-search direction=%s payload=%d candidate=%d/%d tile_n=%d tile_k=%d "
        "along_n=%d swizzle=%d compute=%d p50=%.6f p95=%.6f best=%.6f pflops=%.6f "
        "regs=%d local_bytes=%zu smem=%zu stable=%d verified=1\n",
        IsCombine?"combine":"dispatch",replay,++candidate,q.external_search?32:80,TileN,TileK,int(along),sw,compute,
        result.p50,result.p95,best,flops/(ranks.size()*result.p50*1e12),resources.numRegs,
        resources.localSizeBytes,sizeof(typename Kernel::SharedStorage),int(result.drift<=.05));
    fflush(stdout);
  }
}

#if FUSE_GROUPED_EXTERNAL
struct GroupedDeepgemmReference {
  int device,capacity;
  void* plan=nullptr;
  cudaGraph_t graph{};
  Buffer<fuse::Bf16> output;
  template<class Rank>
  GroupedDeepgemmReference(Rank& r,int block_m):device(r.rank),capacity(r.capacity),output(r.d.size) {
    plan=grouped_deepgemm_create(r.E,r.host_offsets.data(),r.capacity,r.N,r.K,
        r.oracle_a.ptr,r.b.ptr,block_m,r.stream);
    if(!plan) throw std::runtime_error(grouped_deepgemm_error());
    try {
      check(grouped_deepgemm_launch(plan,r.stream));
      check(cudaStreamSynchronize(r.stream));
      check(cudaStreamBeginCapture(r.stream,cudaStreamCaptureModeThreadLocal));
      check(grouped_deepgemm_launch(plan,r.stream));
      check(cudaStreamEndCapture(r.stream,&graph));
    } catch(...) { grouped_deepgemm_destroy(plan); throw; }
  }
  ~GroupedDeepgemmReference() {
    cudaSetDevice(device); if(graph) cudaGraphDestroy(graph); grouped_deepgemm_destroy(plan);
  }
};
template<bool IsCombine>
void measure_deepgemm(std::vector<std::unique_ptr<GroupedRank<IsCombine>>>& ranks,
    const GroupedWorkload& q,int replay) {
  for(int block_m:{128,256}) {
    std::vector<std::unique_ptr<GroupedDeepgemmReference>> refs;
    std::vector<grouped_measurement::Operation> ops;
    double flops=0;
    for(auto& r:ranks) {
      check(cudaSetDevice(r->rank));
      refs.push_back(std::make_unique<GroupedDeepgemmReference>(*r,block_m));
      ops.push_back({r->rank,r->stream,refs.back()->graph});
      flops+=2.*r->host_offsets.back()*r->N*r->K;
    }
    auto validate=[&] {
      for(auto& r:ranks) {
        check(cudaSetDevice(r->rank)); auto& ref=*refs[r->rank];
        check(grouped_deepgemm_read(ref.plan,ref.output.ptr,r->capacity,r->stream));
        validate_grouped_reference(*r,ref.output.ptr);
      }
    };
    validate();
    const auto result=grouped_measurement::measure(ops);
    validate();
    std::ofstream raw(q.timing_csv,std::ios::app); raw<<std::setprecision(9);
    write_grouped_samples(raw,ranks,q,replay,block_m==128?"deepgemm_m128":"deepgemm_m256",
        result,{0,148,1,false,128,64});
    printf("RESULT external backend=deepgemm_native payload=%d tile_m=%d n=128 k=64 cluster=2 swap_ab=1 compute=148 p50=%.6f pflops=%.6f stable=%d verified=1\n",
        replay,block_m,result.p50,flops/(ranks.size()*result.p50*1e12),int(result.drift<=.05));
    fflush(stdout);
  }
}
#endif

template <bool IsCombine, int TileN, int TileK>
void measure_grouped_case(std::vector<std::unique_ptr<GroupedRank<IsCombine>>>& ranks,
    const GroupedWorkload& q, int replay, bool along_n, int sw, int comm, int compute) {
  const int world=ranks.size();
  // Reference modes execute and validate serially. Share their output capacity
  // instead of keeping three large GEMM results resident; graph/plan lifetimes
  // end before this owning buffer. Never alias the independent oracle or fusion.
  std::vector<std::unique_ptr<Buffer<fuse::Bf16>>> reference_output;
  std::vector<std::unique_ptr<GroupedNativeReference>> refs;
  std::vector<std::unique_ptr<GroupedLtReference>> lt;
  std::vector<std::unique_ptr<GroupedCutlassReference<TileN,TileK>>> cutlass;
  std::vector<std::unique_ptr<GroupedTransportReference<IsCombine,TileN,TileK>>> transport;
  std::vector<grouped_measurement::Operation> pure_ops,cutlass_ops,fused_ops,transport_ops,lt_ops;
  for (auto& r:ranks) {
    check(cudaSetDevice(r->rank));
    reference_output.push_back(std::make_unique<Buffer<fuse::Bf16>>(r->d.size));
    auto* scratch=reference_output.back()->ptr;
    refs.push_back(std::make_unique<GroupedNativeReference>(r->blas,r->stream,
        r->host_offsets,r->capacity,r->N,r->K,r->b.ptr,r->oracle_a.ptr,scratch));
    pure_ops.push_back({r->rank,r->stream,refs.back()->graph});
    lt.push_back(std::make_unique<GroupedLtReference>(r->stream,r->host_offsets,
        r->capacity,r->N,r->K,r->b.ptr,r->oracle_a.ptr,r->lt_plans,scratch));
    lt_ops.push_back({r->rank,r->stream,lt.back()->graph});
    cutlass.push_back(std::make_unique<GroupedCutlassReference<TileN,TileK>>(r->stream,
        r->host_offsets,r->capacity,r->N,r->K,r->b.ptr,r->oracle_a.ptr,compute,along_n,sw,scratch));
    cutlass_ops.push_back({r->rank,r->stream,cutlass.back()->graph});
    transport.push_back(std::make_unique<GroupedTransportReference<IsCombine,TileN,TileK>>(
        ranks,r->rank,cutlass.back()->tiles,comm,compute,along_n,sw));
    transport_ops.push_back({r->rank,r->stream,transport.back()->graph});
    fused_ops.push_back({r->rank,r->stream,r->graph});
  }
  auto validate_reference=[&](int variant=0) {
    for (auto& r:ranks) {
      const auto* data=variant==2?lt[r->rank]->output.ptr:
          (variant==1?cutlass[r->rank]->output.ptr:refs[r->rank]->output.ptr);
      validate_grouped_reference(*r,data);
    }
  };
  { grouped_measurement::Timer initial(pure_ops); initial.sample(); }
  validate_reference();
  const auto pure=grouped_measurement::measure(pure_ops);
  validate_reference();
  { grouped_measurement::Timer initial(lt_ops); initial.sample(); }
  validate_reference(2);
  const auto tuned=grouped_measurement::measure(lt_ops);
  validate_reference(2);
  { grouped_measurement::Timer initial(cutlass_ops); initial.sample(); }
  validate_reference(true);
  const auto own=grouped_measurement::measure(cutlass_ops);
  validate_reference(true);
  const auto fused=grouped_measurement::measure(fused_ops);
  for(auto& r:ranks) r->validate_gpu(ranks);
  const auto copy=grouped_measurement::measure(transport_ops);
  for(auto& r:ranks) r->validate_gpu(ranks);
  std::ofstream raw(q.timing_csv,std::ios::app);
  if(!raw) throw std::runtime_error("cannot open grouped sample CSV");
  raw<<std::setprecision(9);
  double flops=0;
  for(const auto& r:ranks) flops+=2.*r->host_offsets.back()*r->N*r->K;
  for (int variant=0;variant<5;++variant) {
    const auto& result=variant==4?tuned:(variant==3?copy:(variant==2?fused:(variant==1?own:pure)));
    const char* mode=variant==4?"cublaslt_sequence_tuned":(variant==3?"transport_body":
        (variant==2?"fused":(variant==1?"cutlass_matched":"cublas_grouped_default")));
    write_grouped_samples(raw,ranks,q,replay,mode,result,{comm,compute,sw,along_n,TileN,TileK});
  }
  if(!raw) throw std::runtime_error("failed to write grouped sample CSV");
  const double pf=flops/(world*pure.p50*1e12), ff=flops/(world*fused.p50*1e12);
  printf("RESULT grouped direction=%s ep=%d experts=%d tokens=%d n=%d k=%d payload=%d tile_n=%d tile_k=%d "
      "fused_ms=%.6f fused_pflops=%.6f native_ms=%.6f native_pflops=%.6f retention=%.4f "
      "cutlass_ms=%.6f cutlass_pflops=%.6f fused_over_cutlass=%.4f "
      "transport_ms=%.6f drift_transport=%.4f "
      "lt_ms=%.6f lt_pflops=%.6f fused_over_lt=%.4f drift_lt=%.4f "
      "drift_fused=%.4f drift_native=%.4f drift_cutlass=%.4f samples=50 stable=%d native_tuned=0 lt_tuned=1 exploratory=1\n",
      IsCombine?"combine":"dispatch",world,q.experts,q.tokens,ranks[0]->N,ranks[0]->K,replay,TileN,TileK,
      fused.p50,ff,pure.p50,pf,pure.p50/fused.p50,own.p50,flops/(world*own.p50*1e12),own.p50/fused.p50,
      copy.p50,copy.drift,tuned.p50,flops/(world*tuned.p50*1e12),tuned.p50/fused.p50,tuned.drift,
      fused.drift,pure.drift,own.drift,
      int(fused.drift<=0.05&&pure.drift<=0.05&&own.drift<=0.05&&copy.drift<=0.05&&tuned.drift<=0.05));
  fflush(stdout);
}

// Fixed-budget transport experiment: reuse the production copy body, without
// constructing/tuning unrelated GEMM libraries. Keep prefix buffers alive
// through Graph destruction and retain the fused kernel's reserved SMEM.
template<int TileN,int TileK,bool SwapAB=false,bool TrimTokens=true,int SmMode=1,
    bool StockScheduler=SmMode==2>
void measure_dispatch_transport(std::vector<std::unique_ptr<GroupedRank<false>>>& ranks,
    const GroupedWorkload& q,int replay,bool along,int sw,int comm,int compute) {
  std::vector<std::unique_ptr<Buffer<int64_t>>> tiles;
  std::vector<std::unique_ptr<GroupedTransportReference<false,TileN,TileK,SmMode,StockScheduler>>> refs;
  std::vector<grouped_measurement::Operation> ops;
  for(auto& r:ranks) {
    check(cudaSetDevice(r->rank));
    std::vector<int64_t> prefix(1,0);
    for(int e=0;e<r->E;++e)
      prefix.push_back(prefix.back()+(r->host_offsets[e+1]-r->host_offsets[e]+128*SmMode-1)/(128*SmMode));
    tiles.push_back(std::make_unique<Buffer<int64_t>>(prefix.size()));
    tiles.back()->copy(prefix);
    refs.push_back(std::make_unique<GroupedTransportReference<false,TileN,TileK,SmMode,StockScheduler>>(
        ranks,r->rank,*tiles.back(),comm,compute,along,sw,q.balance_tail));
    ops.push_back({r->rank,r->stream,refs.back()->graph});
  }
  const auto measured=grouped_measurement::measure(ops);
  for(auto& r:ranks) r->validate_gpu(ranks);
  std::ofstream raw(q.timing_csv,std::ios::app);raw<<std::setprecision(9);
  write_grouped_samples(raw,ranks,q,replay,"transport_body",measured,
      {comm,compute,sw,along,TileN,TileK});
  printf("RESULT transport-only payload=%d p50=%.6f p95=%.6f stable=%d verified=1\n",
      replay,measured.p50,measured.p95,int(measured.drift<=.05));fflush(stdout);
  if(q.compute_compare) {
    // Diagnose the frozen GEMM, not a new reference search. Inputs are the
    // independently gathered oracle; tile/raster/swizzle/worker budget match
    // fusion exactly. Serial modes reuse production D after fusion validation
    // to avoid allocating another model-sized output. No communication runs.
    // Match scheduler identity independently of UMMA width: a native pair
    // and a stock pair are different kernels even with identical tile sizes.
    std::vector<std::unique_ptr<GroupedCutlassReference<
        TileN,TileK,StockScheduler,SwapAB,TrimTokens,SmMode>>> pure;
    std::vector<grouped_measurement::Operation> pure_ops;
    for(auto& r:ranks) {
      check(cudaSetDevice(r->rank));
      pure.push_back(std::make_unique<GroupedCutlassReference<
          TileN,TileK,StockScheduler,SwapAB,TrimTokens,SmMode>>(
          r->stream,r->host_offsets,r->capacity,r->N,r->K,r->b.ptr,r->oracle_a.ptr,
          compute,along,sw,r->d.ptr,false));
      // Finish default-stream metadata/output poisoning before replaying on
      // the nonblocking reference stream; this setup is outside all samples.
      check(cudaDeviceSynchronize());
      pure_ops.push_back({r->rank,r->stream,pure.back()->graph});
    }
    { grouped_measurement::Timer initial(pure_ops); initial.sample(); }
    for(auto& r:ranks) validate_grouped_reference(*r,r->d.ptr);
    const auto gemm=grouped_measurement::measure(pure_ops);
    for(auto& r:ranks) validate_grouped_reference(*r,r->d.ptr);
    write_grouped_samples(raw,ranks,q,replay,"cutlass_matched",gemm,
        {comm,compute,sw,along,TileN,TileK});
    printf("RESULT matched-gemm payload=%d compute=%d p50=%.6f stable=%d verified=1\n",
        replay,compute,gemm.p50,int(gemm.drift<=.05));fflush(stdout);
  }
  if(!raw) throw std::runtime_error("failed to write dispatch diagnostic samples");
}

template<int SmMode,bool StockScheduler>
void measure_dispatch_configuration(std::vector<std::unique_ptr<GroupedRank<false>>>& ranks,
    const GroupedWorkload& q,int replay,bool along,int sw,int comm,int compute) {
  if(q.tile_n==128 && q.tile_k==64)
    measure_dispatch_transport<128,64,false,true,SmMode,StockScheduler>(ranks,q,replay,along,sw,comm,compute);
  else if(q.tile_n==128 && q.tile_k==128)
    measure_dispatch_transport<128,128,false,true,SmMode,StockScheduler>(ranks,q,replay,along,sw,comm,compute);
  else if(q.tile_n==256 && q.tile_k==64)
    measure_dispatch_transport<256,64,false,true,SmMode,StockScheduler>(ranks,q,replay,along,sw,comm,compute);
  else if(q.tile_n==256 && q.tile_k==128)
    measure_dispatch_transport<256,128,false,true,SmMode,StockScheduler>(ranks,q,replay,along,sw,comm,compute);
  else throw std::runtime_error("unsupported grouped tile");
}

template<bool IsCombine>
void verify_grouped_ep(int world, bool along_n, int sw, int comm, int compute,
    const GroupedWorkload& q = {}) {
  using R=GroupedRank<IsCombine>; using Bf16=fuse::Bf16;
  const int E=q.experts, T=q.tokens, Top=q.topk;
  const int N=IsCombine?q.h:2*q.f, K=IsCombine?q.f:q.h;
  std::vector<std::unique_ptr<R>> ranks;
  for(int r=0;r<world;++r) { check(cudaSetDevice(r)); ranks.push_back(std::make_unique<R>(r,world,q)); }
  for(auto& r:ranks) r->capture(ranks,along_n,sw,comm,compute,q.tile_n,q.tile_k,q.balance_tail,q.swap_ab,q.trim_swap_tokens);
  std::vector<std::vector<Bf16>> previous_branches;
  std::vector<fuse::detail::GroupedInvocationPolicy> previous_policies(world);
  std::set<int> observed_policies;
  // Equal-ish, deliberately skewed (other ranks empty), and all-empty. Change
  // payload and row order as well as M_e; no graph re-capture between replays.
  for(int replay=0;replay<(q.property_seed>=0?16:q.boundary_checks?3:2);++replay) {
    std::mt19937 rng(q.property_seed>=0 ? uint32_t(q.property_seed)+replay/2 : 20260916+replay);
    std::uniform_real_distribution<float> random(-0.5f,0.5f);
    using Source=fuse::detail::GroupedTokenSource;
    std::vector<std::vector<Source>> by_expert(world*E);
    if (q.property_seed>=0) {
      by_expert=grouped_validation::property_routes(world,E,T,Top,uint32_t(q.property_seed),replay);
      printf("PROPERTY grouped seed=%llu replay=%d rows=",(unsigned long long)q.property_seed,replay);
      for(const auto& rows:by_expert) printf("%zu,",rows.size());
      puts(""); fflush(stdout); // Reproduction coordinates survive a GPU failure.
    } else if (q.boundary_checks) {
      if(replay!=2) for(int r=0;r<world;++r) for(int t=0;t<T;++t) {
        const int total=replay==1?E:world*E;
        const int first=(t*17+r*5)%total;
        const int second=(first+1+(t*5)%(total-1))%total;
        by_expert[first].push_back({r,t,0}); by_expert[second].push_back({r,t,1});
      }
    } else {
      // Match the catalog's legal cyclic distinct-top-k route family. An
      // explicit permutation changes destination locality, never branch counts.
      std::vector<int> permutation(world*E);
      for (int e=0;e<world*E;++e) permutation[e]=e;
      std::shuffle(permutation.begin(),permutation.end(),rng);
      const int active_experts=q.hot_half?world*E/2:world*E;
      if(Top>active_experts) throw std::runtime_error("distinct top-k exceeds active experts");
      for (int r=0;r<world;++r) for (int t=0;t<T;++t) for (int slot=0;slot<Top;++slot) {
        const int64_t branch=(int64_t(r)*T+t)*Top+slot;
        by_expert[permutation[branch%active_experts]].push_back({r,t,slot});
      }
    }
    if(q.property_seed<0) for(auto& rows:by_expert) std::shuffle(rows.begin(),rows.end(),rng);
    std::vector<std::vector<grouped_validation::BranchOwner>> owners(
        world,std::vector<grouped_validation::BranchOwner>(size_t(T)*Top));
    for (int e=0;e<world*E;++e) for (size_t row=0;row<by_expert[e].size();++row) {
      const auto src=by_expert[e][row];
      auto& owner=owners[src.rank][size_t(src.token)*Top+src.slot];
      if (owner.rank>=0) throw std::runtime_error("duplicate input branch route");
      owner={e/E,e%E,int(row)};
    }
    for(auto& r:ranks) {
      check(cudaSetDevice(r->rank));
      if (q.boundary_checks) {
        r->host_input.resize(r->input.size);
        for(auto& x:r->host_input) x=Bf16(random(rng));
        r->input.copy(r->host_input);
        std::vector<Bf16> weights(r->b.size);
        for(auto& x:weights) x=Bf16(random(rng));
        r->b.copy(weights);
      } else {
        const uint32_t seed=20260916+replay*1000+r->rank*3;
        r->randomize(r->input,seed,"input",replay);
        r->randomize(r->b,seed+1,"weight",replay);
      }
      r->host_offsets.assign(1,0); r->host_routes.clear();
      for(int e=0;e<E;++e) {
        const auto& rows=by_expert[r->rank*E+e];
        if(rows.size()>size_t(r->capacity)) throw std::runtime_error("expert capacity exceeded");
        r->host_routes.insert(r->host_routes.end(),rows.begin(),rows.end());
        r->host_offsets.push_back(r->host_routes.size());
      }
      r->offsets.copy(r->host_offsets); r->routes.copy(r->host_routes);
      if(q.timing_csv) {
        uint64_t route_hash=14695981039346656037ull;
        for(const auto& route:r->host_routes)
          for(uint32_t value:{uint32_t(route.rank),uint32_t(route.token),uint32_t(route.slot)})
            for(int byte=0;byte<4;++byte)
              route_hash=(route_hash^((value>>(8*byte))&255u))*1099511628211ull;
        printf("ROWS grouped direction=%s rank=%d payload=%d route_seed=%d route_fnv64=%016llx values=",
            IsCombine?"combine":"dispatch",r->rank,replay,20260916+replay,
            (unsigned long long)route_hash);
        for(int e=0;e<E;++e) printf("%s%lld",e?",":"",
            (long long)(r->host_offsets[e+1]-r->host_offsets[e]));
        puts("");
      }
      r->branch_owners.copy(owners[r->rank]);
      check(cudaMemset(r->a.ptr,0x7f,r->a.size*sizeof(Bf16)));
      check(cudaMemset(r->d.ptr,0x7f,r->d.size*sizeof(Bf16)));
      check(cudaMemset(r->output.ptr,0x7f,r->output.size*sizeof(Bf16)));
    }
    for(auto& r:ranks) {
      check(cudaSetDevice(r->rank));
      if (q.boundary_checks) {
      r->host_a.assign(r->oracle_a.size,Bf16::bitcast(0x7f7f));
      for(int e=0;e<E;++e) for(int64_t i=r->host_offsets[e];i<r->host_offsets[e+1];++i) {
        const auto route=r->host_routes[i]; const int row=int(i-r->host_offsets[e]);
        for(int k=0;k<K;++k)
          r->host_a[(size_t(e)*r->capacity+row)*K+k]=IsCombine?Bf16(random(rng)):
              ranks[route.rank]->host_input[size_t(route.token)*K+k];
      }
      r->oracle_a.copy(r->host_a);
      if constexpr(IsCombine) r->a.copy(r->host_a);

      } else {
        if constexpr (IsCombine)
          r->randomize(r->oracle_a,20260918+replay*1000+r->rank*3,"expert_input",replay);
        grouped_validation::Peers inputs{};
        for (int peer=0;peer<world;++peer) inputs.data[peer]=ranks[peer]->input.ptr;
        grouped_validation::reference_lhs<IsCombine><<<256,256>>>(r->oracle_a.ptr,r->a.ptr,
            E,r->capacity,K,r->offsets.ptr,r->routes.ptr,inputs);
        check(cudaGetLastError());
      }
      const float alpha=1.f,beta=0.f;
      for(int e=0;e<E;++e) {
        const int m=int(r->host_offsets[e+1]-r->host_offsets[e]);
        if(m) check(cublasGemmEx(r->blas,CUBLAS_OP_T,CUBLAS_OP_N,N,m,K,
            &alpha,r->b.ptr+size_t(e)*N*K,CUDA_R_16BF,K,
            r->oracle_a.ptr+size_t(e)*r->capacity*K,CUDA_R_16BF,K,&beta,
            r->ref.ptr+size_t(e)*r->capacity*N,CUDA_R_16BF,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      }
      check(cudaDeviceSynchronize());
    }
    for(auto& r:ranks) { check(cudaSetDevice(r->rank)); check(cudaGraphLaunch(r->exec,r->stream)); }
    for(auto& r:ranks) { check(cudaSetDevice(r->rank)); check(cudaStreamSynchronize(r->stream)); }
    if (q.property_dynamic) {
      for (auto& r : ranks) {
        fuse::detail::GroupedInvocationPolicy selected{};
        check(cudaSetDevice(r->rank));
        check(cudaMemcpy(&selected,r->selected_policy.ptr,sizeof(selected),cudaMemcpyDeviceToHost));
        if (selected.num_comm_ctas <= 0 || selected.num_compute_ctas <= 0 ||
            selected.num_comm_ctas + selected.num_compute_ctas != 148 ||
            (selected.swizzle != 1 && selected.swizzle != 2 &&
             selected.swizzle != 4 && selected.swizzle != 8))
          throw std::runtime_error("property: invalid selected device policy");
        const auto& previous=previous_policies[r->rank];
        if (replay % 2 && (selected.num_comm_ctas != previous.num_comm_ctas ||
            selected.num_compute_ctas != previous.num_compute_ctas ||
            selected.swizzle != previous.swizzle || selected.along_n != previous.along_n))
          throw std::runtime_error("property: row permutation changed device policy");
        if (!(replay % 2)) previous_policies[r->rank]=selected;
        observed_policies.insert(selected.num_comm_ctas*1000 + selected.swizzle*10 + selected.along_n);
        printf("PROPERTY_POLICY rank=%d replay=%d comm=%d compute=%d swizzle=%d along_n=%d\n",
            r->rank,replay,selected.num_comm_ctas,selected.num_compute_ctas,
            selected.swizzle,int(selected.along_n));
      }
      fflush(stdout);
    }
    double max_abs=0;
    for (auto& r:ranks) max_abs=std::max(max_abs,r->validate_gpu(ranks));
    // Full host oracle remains only for the bounded boundary suite, cross-checking
    // the new device validator. Sized cases never download whole tensors.
    if (q.boundary_checks) {
    std::vector<std::vector<Bf16>> branch_expected(world,std::vector<Bf16>(size_t(T)*Top*N,Bf16::bitcast(0x7f7f)));
    for(auto& r:ranks) {
      check(cudaSetDevice(r->rank));
      r->host_d.resize(r->d.size); r->host_ref.resize(r->ref.size);
      check(cudaMemcpy(r->host_d.data(),r->d.ptr,r->d.size*sizeof(Bf16),cudaMemcpyDeviceToHost));
      check(cudaMemcpy(r->host_ref.data(),r->ref.ptr,r->ref.size*sizeof(Bf16),cudaMemcpyDeviceToHost));
      std::vector<Bf16> copied(r->a.size);
      check(cudaMemcpy(copied.data(),r->a.ptr,r->a.size*sizeof(Bf16),cudaMemcpyDeviceToHost));
      for(int e=0;e<E;++e) for(int row=0;row<r->buffer_rows;++row) for(int k=0;k<K;++k) {
        const int64_t rows=r->host_offsets[e+1]-r->host_offsets[e];
        const int64_t last=row<rows ? row+(rows-1-row)/r->buffer_rows*r->buffer_rows : row;
        const auto expected=row<rows?r->host_a[(size_t(e)*r->capacity+last)*K+k].raw():0x7f7f;
        if(copied[(size_t(e)*r->buffer_rows+row)*K+k].raw()!=expected)
          throw std::runtime_error("dispatch bytes/untouched buffer capacity mismatch");
      }
      for(int e=0;e<E;++e) for(int row=0;row<r->capacity;++row) for(int n=0;n<N;++n) {
        const size_t idx=(size_t(e)*r->capacity+row)*N+n;
        if(row>=r->host_offsets[e+1]-r->host_offsets[e]) {
          if(r->host_d[idx].raw()!=0x7f7f) throw std::runtime_error("CTASP wrote inactive rows");
        } else {
          const double x=float(r->host_d[idx]),y=float(r->host_ref[idx]),error=std::abs(x-y);
          if(!std::isfinite(x)||error>0.002+0.008*std::abs(y)) throw std::runtime_error("CTASP numeric mismatch");
          max_abs=std::max(max_abs,error);
          const auto route=r->host_routes[r->host_offsets[e]+row];
          branch_expected[route.rank][(size_t(route.token)*Top+route.slot)*N+n]=r->host_d[idx];
        }
      }
    }
    if constexpr(IsCombine) for(auto& r:ranks) {
      check(cudaSetDevice(r->rank)); std::vector<Bf16> received(r->output.size);
      check(cudaMemcpy(received.data(),r->output.ptr,r->output.size*sizeof(Bf16),cudaMemcpyDeviceToHost));
      if(std::memcmp(received.data(),branch_expected[r->rank].data(),received.size()*sizeof(Bf16)))
        throw std::runtime_error("combine branch bytes/untouched capacity mismatch");
    }
    if(q.property_seed>=0) {
      if(replay%2==0) previous_branches=branch_expected;
      else for(int rank=0;rank<world;++rank) for(size_t i=0;i<branch_expected[rank].size();++i) {
        const auto a=branch_expected[rank][i], b=previous_branches[rank][i];
        if(a.raw()==0x7f7f || b.raw()==0x7f7f) {
          if(a.raw()!=b.raw()) throw std::runtime_error("property: row permutation changed branch coverage");
        } else if(!std::isfinite(float(a)) || !std::isfinite(float(b)) ||
            std::abs(float(a)-float(b))>0.002f+0.008f*std::abs(float(b)))
          throw std::runtime_error("property: row permutation changed branch GEMM result");
      }
    }
    }
#if FUSE_ENABLE_PROFILING
    if(q.trace_out && replay==1) {
      // All rank graphs are submitted concurrently, then joined. Preparation
      // retains the original cross-rank input handshake and monotonically
      // increasing epochs; old ready flags cannot satisfy the new invocation.
      auto launch_all=[&]() {
        std::vector<cudaError_t> errors(world,cudaSuccess);
        std::vector<std::thread> threads;
        for(int r=0;r<world;++r) threads.emplace_back([&,r]() {
          auto status=cudaSetDevice(r);
          if(status==cudaSuccess) status=cudaGraphLaunch(ranks[r]->exec,ranks[r]->stream);
          errors[r]=status;
        });
        for(auto& thread:threads) thread.join();
        for(auto status:errors) check(status);
        for(auto& r:ranks) {check(cudaSetDevice(r->rank));check(cudaStreamSynchronize(r->stream));}
      };
      for(int warm=0;warm<10;++warm) launch_all();
      for(auto& r:ranks) {
        check(cudaSetDevice(r->rank));r->clear_profile();
        check(cudaMemset(r->a.ptr,0x7f,r->a.size*sizeof(Bf16)));
        check(cudaMemset(r->d.ptr,0x7f,r->d.size*sizeof(Bf16)));
        // These resets use the default stream, whereas the captured graph
        // uses a nonblocking stream. Finish diagnostic preparation BEFORE
        // launching any rank: default-stream ordering does not cover it.
        check(cudaDeviceSynchronize());
      }
      launch_all();
      for(auto& r:ranks) {
        r->validate_gpu(ranks);
        r->dump_profile(q.trace_out,comm,compute,q.tile_n,q.tile_k,sw,along_n);
      }
      puts("PROFILE grouped: globaltimer, diagnostic Graph, 10 warmups, epoch13, all-rank numeric/route/tail checked");
    }
#endif
    if(q.external_search) {
#if FUSE_GROUPED_EXTERNAL
      int candidate=0; double best=std::numeric_limits<double>::infinity();
      search_grouped_tile<IsCombine,128,64,true>(ranks,q,replay,148,candidate,best);
      search_grouped_tile<IsCombine,128,128,true>(ranks,q,replay,148,candidate,best);
      search_grouped_tile<IsCombine,256,64,true>(ranks,q,replay,148,candidate,best);
      search_grouped_tile<IsCombine,256,128,true>(ranks,q,replay,148,candidate,best);
      measure_deepgemm(ranks,q,replay);
      std::vector<grouped_measurement::Operation> ops;
      for(auto& r:ranks) ops.push_back({r->rank,r->stream,r->graph});
      const auto fused=grouped_measurement::measure(ops,false);
      for(auto& r:ranks) r->validate_gpu(ranks);
      std::ofstream raw(q.timing_csv,std::ios::app); raw<<std::setprecision(9);
      write_grouped_samples(raw,ranks,q,replay,"fused",fused,
          {comm,compute,sw,along_n,q.tile_n,q.tile_k});
#else
      throw std::runtime_error("external references were not built");
#endif
    } else if(q.gemm_search) {
      int candidate=0; double best=std::numeric_limits<double>::infinity();
      search_grouped_tile<IsCombine,128,64>(ranks,q,replay,compute,candidate,best);
      search_grouped_tile<IsCombine,128,128>(ranks,q,replay,compute,candidate,best);
      search_grouped_tile<IsCombine,256,64>(ranks,q,replay,compute,candidate,best);
      search_grouped_tile<IsCombine,256,128>(ranks,q,replay,compute,candidate,best);
      search_grouped_tile<IsCombine,128,64,false,2>(ranks,q,replay,compute,candidate,best);
      search_grouped_tile<IsCombine,128,128,false,2>(ranks,q,replay,compute,candidate,best);
      search_grouped_tile<IsCombine,256,64,false,2>(ranks,q,replay,compute,candidate,best);
      search_grouped_tile<IsCombine,256,128,false,2>(ranks,q,replay,compute,candidate,best);
      // Diagnose whether the remaining gap is the custom grouped traversal or
      // the one-SM MMA shape. These use the SAME persistent CTA budget as the
      // fused GEMM; unlike the external strong reference they do not get 148.
      search_grouped_tile<IsCombine,128,64,true,1>(ranks,q,replay,compute,candidate,best);
      search_grouped_tile<IsCombine,128,128,true,1>(ranks,q,replay,compute,candidate,best);
      search_grouped_tile<IsCombine,256,64,true,1>(ranks,q,replay,compute,candidate,best);
      search_grouped_tile<IsCombine,256,128,true,1>(ranks,q,replay,compute,candidate,best);
      search_grouped_tile<IsCombine,128,64,true,2>(ranks,q,replay,compute,candidate,best);
      search_grouped_tile<IsCombine,128,128,true,2>(ranks,q,replay,compute,candidate,best);
      search_grouped_tile<IsCombine,256,64,true,2>(ranks,q,replay,compute,candidate,best);
      search_grouped_tile<IsCombine,256,128,true,2>(ranks,q,replay,compute,candidate,best);
    } else if(q.timing_csv && q.fused_only) {
      std::vector<grouped_measurement::Operation> ops;
      double flops=0;
      for(auto& r:ranks) {
        ops.push_back({r->rank,r->stream,r->graph});
        flops+=2.*r->host_offsets.back()*r->N*r->K;
      }
      const auto measured=grouped_measurement::measure(ops);
      for(auto& r:ranks) r->validate_gpu(ranks);
      std::ofstream raw(q.timing_csv,std::ios::app);raw<<std::setprecision(9);
      write_grouped_samples(raw,ranks,q,replay,"fused",measured,
          {comm,compute,sw,along_n,q.tile_n,q.tile_k});
      // Finish this writer before the optional transport sampler opens the
      // same CSV. Buffered append streams must not interleave partial rows.
      raw.close();
      if(!raw) throw std::runtime_error("failed to write fused samples");
      printf("RESULT fused-only payload=%d buffer_rows=%d tail_balance=%d hot_half=%d p50=%.6f p95=%.6f pflops=%.6f stable=%d verified=1\n",
          replay,ranks[0]->buffer_rows,q.balance_tail,q.hot_half,measured.p50,measured.p95,
          flops/(world*measured.p50*1e12),int(measured.drift<=.05));fflush(stdout);
      if constexpr(!IsCombine) if(q.transport_compare) {
        if(q.swap_ab && !q.trim_swap_tokens && q.tile_k==64) measure_dispatch_transport<128,64,true,false>(ranks,q,replay,along_n,sw,comm,compute);
        else if(q.swap_ab && !q.trim_swap_tokens && q.tile_k==128) measure_dispatch_transport<128,128,true,false>(ranks,q,replay,along_n,sw,comm,compute);
        else if(q.swap_ab && q.tile_k==64) measure_dispatch_transport<128,64,true>(ranks,q,replay,along_n,sw,comm,compute);
        else if(q.swap_ab && q.tile_k==128) measure_dispatch_transport<128,128,true>(ranks,q,replay,along_n,sw,comm,compute);
        else if(q.mma_sm_count==2 && q.stock_scheduler()) measure_dispatch_configuration<2,true>(ranks,q,replay,along_n,sw,comm,compute);
        else if(q.mma_sm_count==2) measure_dispatch_configuration<2,false>(ranks,q,replay,along_n,sw,comm,compute);
        else if(q.stock_scheduler()) measure_dispatch_configuration<1,true>(ranks,q,replay,along_n,sw,comm,compute);
        else measure_dispatch_configuration<1,false>(ranks,q,replay,along_n,sw,comm,compute);
      }
    } else if(q.timing_csv) {
      if(q.tile_n==128 && q.tile_k==64) measure_grouped_case<IsCombine,128,64>(ranks,q,replay,along_n,sw,comm,compute);
      else if(q.tile_n==128 && q.tile_k==128) measure_grouped_case<IsCombine,128,128>(ranks,q,replay,along_n,sw,comm,compute);
      else if(q.tile_n==256 && q.tile_k==64) measure_grouped_case<IsCombine,256,64>(ranks,q,replay,along_n,sw,comm,compute);
      else if(q.tile_n==256 && q.tile_k==128) measure_grouped_case<IsCombine,256,128>(ranks,q,replay,along_n,sw,comm,compute);
      else throw std::runtime_error("unsupported grouped tile");
    }
    printf("CHECK grouped-ep direction=%s ep=%d experts=%d tokens=%d n=%d k=%d along_n=%d swizzle=%d comm=%d compute=%d replay=%d max_abs=%.6g passed\n",
        IsCombine?"combine":"dispatch",world,E,T,N,K,along_n,sw,comm,compute,replay,max_abs); fflush(stdout);
  }
  if (q.property_dynamic && observed_policies.size() < 2)
    throw std::runtime_error("property: dynamic selector did not exercise multiple policies");
}

void report_grouped_validation(const char* suite,const std::string& direction,
    int policies,int replays,const char* detail) {
  const bool both=direction=="both";
  printf("PASSED grouped-%s: directions=%s policies=%d validation_graph_replays_per_rank=%d "
      "replays_per_configuration=%d; %s\n",suite,both?"dispatch,combine":direction.c_str(),
      policies,policies*(both?2:1)*replays,replays,detail);
}

int run_grouped(int argc,char** argv) {
  try {
    int64_t property_seed=-1;
    if(argc>=3 && std::string(argv[argc-2])=="--property-seed") {
      size_t parsed=0; property_seed=std::stoll(argv[argc-1],&parsed);
      if(parsed!=std::strlen(argv[argc-1]) || property_seed<0 || property_seed>UINT32_MAX)
        throw std::runtime_error("property seed must be uint32");
      argc-=2;
    }
    const bool property_dynamic=argc>=2 && std::string(argv[argc-1])=="--property-dynamic";
    if(property_dynamic) --argc;
    if(property_dynamic && property_seed<0)
      throw std::runtime_error("dynamic property policy requires a property seed");
    const bool compute_compare=argc>=2 && std::string(argv[argc-1])=="--compute-compare";
    if(compute_compare) --argc;
    const bool transport_compare=argc>=2 && std::string(argv[argc-1])=="--transport-compare";
    if(transport_compare) --argc;
    if(compute_compare && !transport_compare)
      throw std::runtime_error("compute comparison requires transport comparison");
    const bool ready_summary=argc>=2 && std::string(argv[argc-1])=="--ready-summary";
    if(ready_summary) --argc;
    const bool fused_only=argc>=2 && std::string(argv[argc-1])=="--fused-only";
    if(fused_only) --argc;
    const bool hot_half=argc>=2 && std::string(argv[argc-1])=="--hot-half";
    if(hot_half) --argc;
    const bool balance_tail=argc>=2 && std::string(argv[argc-1])=="--tail-balance";
    if(balance_tail) --argc;
    int buffer_rows=0;
    if(argc>=3 && std::string(argv[argc-2])=="--dispatch-buffer-rows") {
      size_t parsed=0;buffer_rows=std::stoi(argv[argc-1],&parsed);
      if(parsed!=std::strlen(argv[argc-1]) || buffer_rows<0 || buffer_rows%128)
        throw std::runtime_error("Dispatch buffer rows must be zero or a positive multiple of 128");
      argc-=2;
    }
    const char* trace_out=nullptr;
    if(argc>=3 && std::string(argv[argc-2])=="--trace-out") {
#if FUSE_ENABLE_PROFILING
      trace_out=argv[argc-1];argc-=2;
#else
      throw std::runtime_error("grouped profiling was not compiled");
#endif
    }
    std::string direction="both";
    if(argc>=3 && std::string(argv[argc-2])=="--direction") {
      direction=argv[argc-1]; argc-=2;
      if(direction!="both" && direction!="dispatch" && direction!="combine")
        throw std::runtime_error("invalid grouped direction");
    }
    fuse::GroupedGemmPolicy policy{};
    const bool explicit_policy=argc>=3 && std::string(argv[argc-2])=="--policy";
    if(explicit_policy) {
      std::stringstream input(argv[argc-1]);
      std::vector<int> values;
      for(std::string item;std::getline(input,item,',');) {
        size_t parsed=0; int value=std::stoi(item,&parsed);
        if(parsed!=item.size()) throw std::runtime_error("invalid grouped policy integer");
        values.push_back(value);
      }
      if((values.size()<6 || values.size()>11) || (values[0]!=128 && values[0]!=256) ||
          (values[1]!=64 && values[1]!=128) || (values[2]!=0 && values[2]!=1) ||
          (values[3]!=1 && values[3]!=2 && values[3]!=4 && values[3]!=8) ||
          values[4]<=0 || values[5]<=0 ||
          (values.size()>=7 && (values[6]<0 || values[6]>1 || (values[6] && values[0]!=128))) ||
          (values.size()>=8 && (values[7]<0 || values[7]>1 ||
              (values.size()==8 && !values[6]) ||
              (values.size()>=9 && !values[6] && values[7]!=1))) ||
          (values.size()>=9 && (values[8]<1 || values[8]>2 ||
              (values[8]==2 && (values[6] || values[4]%2 || values[5]%2)))))
        throw std::runtime_error(
            "policy requires N,K,AlongN,swizzle,comm,compute[,swapAB[,trimTokens[,mmaSMs[,copy[,scheduler]]]]]");
      policy={values[4],values[5],values[3],bool(values[2]),values[0],values[1]};
      policy.swap_ab=values.size()>=7 && values[6];
      policy.trim_swap_tokens=values.size()<8 || values[7];
      policy.mma_sm_count=values.size()<9 ? 1 : values[8];
      if(values.size()>=10) {
        if(values[9]<0 || values[9]>1 || direction!="dispatch")
          throw std::runtime_error("invalid grouped copy method");
        policy.dispatch_copy=values[9] ? fuse::GroupedDispatchCopy::Tma :
            fuse::GroupedDispatchCopy::CpAsync;
      }
      if(values.size()==11) {
        if(values[10]<0 || values[10]>2 || direction!="dispatch")
          throw std::runtime_error("invalid grouped scheduler");
        policy.scheduler=static_cast<fuse::GroupedGemmScheduler>(values[10]);
      }
      argc-=2;
    }
    const bool external_search=argc>=2 && std::string(argv[argc-1])=="--external-search";
    if(external_search) --argc;
    const bool gemm_search=argc>=2 && std::string(argv[argc-1])=="--gemm-search";
    if(gemm_search) --argc;
    if(policy.scheduler!=fuse::GroupedGemmScheduler::Default &&
        (policy.swap_ab || gemm_search || external_search || (argc==9 && !fused_only)))
      throw std::runtime_error("explicit scheduler requires unswapped Dispatch; timing requires fused-only");
    if(policy.swap_ab && (direction!="dispatch" || gemm_search || external_search ||
        (argc==9 && !fused_only)))
      throw std::runtime_error("swapAB timing requires fused-only Dispatch; use compute-compare for matched GEMM");
    if((gemm_search || external_search) && (argc!=9 || std::string(argv[1])!="--case"))
      throw std::runtime_error("GEMM search requires a sized case and sample CSV");
    cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop,0));
    if (prop.major != 10 || prop.minor != 3) throw std::runtime_error("SM103 device required");
    if(int64_t(policy.num_comm_ctas)+policy.num_compute_ctas>prop.multiProcessorCount)
      throw std::runtime_error("grouped CTA budget exceeds physical SMs");
    const bool sized_case = (argc==8 || argc==9) && std::string(argv[1])=="--case";
    if(property_seed>=0 && (argc!=3 || std::string(argv[1])!="--ep" ||
        direction!="dispatch" || trace_out || gemm_search || external_search || fused_only))
      throw std::runtime_error("property generation requires untimed Dispatch --ep validation");
    if(trace_out && (!sized_case || argc!=8 || direction!="dispatch" || gemm_search || external_search))
      throw std::runtime_error("grouped trace requires one untimed sized Dispatch case");
    if(ready_summary && !trace_out) throw std::runtime_error("ready summary requires --trace-out");
    if(sized_case || (argc==3 && std::string(argv[1])=="--ep")) {
      const int world=std::stoi(argv[2]);
      int count=0; check(cudaGetDeviceCount(&count));
      if((world!=4&&world!=8)||count!=world) throw std::runtime_error("EP4/8 visible device count mismatch");
      for(int r=0;r<world;++r) for(int p=0;p<world;++p) if(r!=p) {
        check(cudaSetDevice(r)); int access=0; check(cudaDeviceCanAccessPeer(&access,r,p));
        if(!access) throw std::runtime_error("required CUDA peer access unavailable");
        auto status=cudaDeviceEnablePeerAccess(p,0);
        if(status==cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError(); else check(status);
      }
      if (sized_case) {
        const int h=std::stoi(argv[3]), f=std::stoi(argv[4]), experts=std::stoi(argv[5]);
        const int top=std::stoi(argv[6]), target=std::stoi(argv[7]);
        if (h<=0 || f<=0 || h%8 || f%8 || f>INT32_MAX/2 || experts<=0 ||
            experts%world || top<=0 || top>experts || target<=0)
          throw std::runtime_error("invalid grouped case dimensions");
        const int64_t tokens=(int64_t(experts)*target+int64_t(world)*top-1)/(int64_t(world)*top);
        if (tokens>INT32_MAX) throw std::runtime_error("token count exceeds index range");
        GroupedWorkload q{experts/world,int(tokens),top,h,f,false,argc==9?argv[8]:nullptr};
        q.tile_n=policy.tile_n; q.tile_k=policy.tile_k;
        q.swap_ab=policy.swap_ab;
        q.trim_swap_tokens=policy.trim_swap_tokens;
        q.mma_sm_count=policy.mma_sm_count;
        q.scheduler=policy.scheduler;
        q.dispatch_copy=policy.dispatch_copy;
        q.gemm_search=gemm_search;
        q.external_search=external_search;
        q.trace_out=trace_out;
        q.ready_summary=ready_summary;
        q.fused_only=fused_only;
        q.transport_compare=transport_compare;
        q.compute_compare=compute_compare;
        if(transport_compare && (!fused_only || direction!="dispatch" || buffer_rows))
          throw std::runtime_error("transport comparison requires full-buffer fused-only Dispatch timing");
        q.buffer_rows=buffer_rows;
        if(buffer_rows && (direction!="dispatch" || (q.timing_csv && !fused_only) || trace_out || gemm_search || external_search))
          throw std::runtime_error("bounded Dispatch requires Dispatch-only validation or fused-only timing");
        q.balance_tail=balance_tail;q.hot_half=hot_half;
        if((balance_tail || hot_half) && (direction!="dispatch" || (trace_out && !ready_summary) || gemm_search || external_search || (q.timing_csv && !fused_only)))
          throw std::runtime_error("tail/skew probe requires Dispatch validation or fused-only timing");
        if(fused_only && (!q.timing_csv || gemm_search || external_search || trace_out))
          throw std::runtime_error("fused-only requires a sized measured case without profiling/search");
        if(q.timing_csv) {
          std::ofstream out(q.timing_csv);
          if(!out) throw std::runtime_error("cannot create grouped sample CSV");
          out<<"direction,h,f,total_experts,topk,tokens_per_rank,ep,payload,mode,sample,rank,rank_rows,ms,comm_ctas,compute_ctas,swizzle,along_n,warmup,drift,round,accepted,tile_n,tile_k,buffer_rows,tail_balance,hot_half,swap_ab,trim_swap_tokens,dispatch_copy,scheduler\n";
        }
        if(direction!="combine") verify_grouped_ep<false>(world,policy.along_n,policy.swizzle,policy.num_comm_ctas,policy.num_compute_ctas,q);
        if(direction!="dispatch") verify_grouped_ep<true>(world,policy.along_n,policy.swizzle,policy.num_comm_ctas,policy.num_compute_ctas,q);
        report_grouped_validation("case",direction,1,2,q.timing_csv?
            "correctness and exploratory native-reference timing; tuning incomplete":
            "sized dual-payload API correctness; performance NOT measured");
        return 0;
      }
      if(explicit_policy) {
        GroupedWorkload q{}; q.tile_n=policy.tile_n; q.tile_k=policy.tile_k;
        q.property_seed=property_seed;
        q.property_dynamic=property_dynamic;
        q.swap_ab=policy.swap_ab;
        q.trim_swap_tokens=policy.trim_swap_tokens;
        q.mma_sm_count=policy.mma_sm_count;
        q.scheduler=policy.scheduler;
        q.dispatch_copy=policy.dispatch_copy;
        q.balance_tail=balance_tail;
        q.buffer_rows=buffer_rows;
        if(buffer_rows && direction!="dispatch") throw std::runtime_error("buffer only applies to Dispatch");
        if(hot_half || (balance_tail && direction!="dispatch")) throw std::runtime_error("invalid tail/skew probe mode");
        if(direction!="combine") verify_grouped_ep<false>(world,policy.along_n,policy.swizzle,policy.num_comm_ctas,policy.num_compute_ctas,q);
        if(direction!="dispatch") verify_grouped_ep<true>(world,policy.along_n,policy.swizzle,policy.num_comm_ctas,policy.num_compute_ctas,q);
        report_grouped_validation("ep",direction,1,property_seed>=0?16:3,
            property_dynamic ?
            "device-selected invocation policy, dynamic Graph boundaries; performance NOT measured" :
            "explicit tile policy, dynamic Graph boundaries; performance NOT measured");
        return 0;
      }
      for(bool along_n:{false,true}) {
        GroupedWorkload q{};q.buffer_rows=buffer_rows;
        q.property_seed=property_seed;
        q.property_dynamic=property_dynamic;
        if(buffer_rows && direction!="dispatch") throw std::runtime_error("buffer only applies to Dispatch");
        if(direction!="combine") verify_grouped_ep<false>(world,along_n,along_n?4:1,along_n?4:2,8,q);
        if(direction!="dispatch") verify_grouped_ep<true>(world,along_n,along_n?4:1,along_n?4:2,8);
      }
      report_grouped_validation("ep",direction,2,property_seed>=0?16:3,
          "routed bytes, numeric, dynamic Graph boundaries; performance NOT measured");
      return 0;
    }
    if(argc!=1) throw std::runtime_error("usage: grouped_bf16 [--ep 4|8 | --case EP H F EXPERTS TOPK ROWS]");
    cublasHandle_t blas; check(cublasCreate(&blas));
    for (bool along_n : {false,true}) for (int sw : {1,4,8}) {
      const int workers = sw == 1 ? 3 : 8;
      verify_grouped<Types::PureGemm,0>(blas,prop.multiProcessorCount,along_n,sw,workers);
      verify_grouped<Types::DispatchGemm,1>(blas,prop.multiProcessorCount,along_n,sw,workers);
      verify_grouped<Types::CombineGemm,2>(blas,prop.multiProcessorCount,along_n,sw,workers);
    }
    check(cublasDestroy(blas));
    puts("PASSED grouped-local: 54 graph replays; transport NOT tested; performance NOT measured");
  } catch (const GroupedCudaError& e) {
    fprintf(stderr,"ERROR: %s\n",e.what());
    return e.status==cudaErrorMemoryAllocation?2:1;
  } catch (const std::exception& e) { fprintf(stderr,"ERROR: %s\n",e.what()); return 1; }
  return 0;
}

int main(int argc,char** argv) {
  if(argc!=3 || std::string(argv[1])!="--batch") return run_grouped(argc,argv);
  try {
    std::ifstream manifest(argv[2]);
    if(!manifest) throw std::runtime_error("cannot open grouped batch manifest");
    std::vector<std::string> lines;
    for(std::string line;std::getline(manifest,line);) if(!line.empty()) lines.push_back(line);
    if(lines.empty() || lines.size()>64) throw std::runtime_error("invalid grouped batch size");
    for(size_t index=0;index<lines.size();++index) {
      std::istringstream input(lines[index]);
      uint64_t minimum=0; input>>minimum;
      std::vector<std::string> arguments{"grouped_bf16"};
      for(std::string item;input>>item;) arguments.push_back(item);
      if(arguments.size()<9 || arguments[1]!="--case" || !minimum)
        throw std::runtime_error("invalid grouped batch row");
      const int world=std::stoi(arguments[2]);
      int visible=0; check(cudaGetDeviceCount(&visible));
      if((world!=4 && world!=8) || visible!=world) throw std::runtime_error("batch device count mismatch");
      bool enough=true;
      for(int rank=0;rank<world;++rank) {
        check(cudaSetDevice(rank)); size_t free=0,total=0; check(cudaMemGetInfo(&free,&total));
        if(free<minimum) enough=false;
      }
      if(!enough) {
        printf("CASE grouped index=%zu state=skipped reason=insufficient_memory required_bytes=%llu\n",
            index,(unsigned long long)minimum); fflush(stdout); continue;
      }
      printf("CASE grouped index=%zu state=running total=%zu\n",index,lines.size()); fflush(stdout);
      std::vector<char*> args;
      for(auto& argument:arguments) args.push_back(argument.data());
      const int status=run_grouped(args.size(),args.data());
      if(status==2) {
        // Non-sticky allocation failure only; never continue after numerical,
        // synchronization or asynchronous device errors in a shared context.
        cudaGetLastError();
        printf("CASE grouped index=%zu state=skipped reason=cuda_oom\n",index);
      } else if(status) {
        printf("CASE grouped index=%zu state=failed\n",index); fflush(stdout); return status;
      } else printf("CASE grouped index=%zu state=completed\n",index);
      fflush(stdout);
    }
    puts("PASSED grouped-batch: completed or explicitly memory-skipped cases; no device reset");
    return 0;
  } catch(const std::exception& e) { fprintf(stderr,"ERROR: %s\n",e.what()); return 1; }
}
