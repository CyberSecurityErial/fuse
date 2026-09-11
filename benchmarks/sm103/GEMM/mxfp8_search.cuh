// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "fuse/operators/primitives/gemm_a2a.h"
#include "../../../csrc/operators/sm103/detail/cutlass_pipeline.cuh"
#include "../../../csrc/operators/sm103/detail/persistent_gemm.cuh"
#include "../../../csrc/operators/sm103/detail/gemm.cuh"
#include <functional>
#include <set>
#include <tuple>

namespace mxfp8_search {
// Pure compute uses the production collective and static persistent scheduler.
// Materialized FP8 operands/scales are supplied before timing. No weight-ready
// adapter, signaling epilogue, quantization or communication is launched.
// The grid is capped at the requested compute budget; SMEM occupancy enforces
// one resident CTA per SM, not a cuBLAS-style advisory heuristic target.
template<class Gemm> struct Entry {
  using ProductionKernel = Entry;
  using Params = typename Gemm::Params;
  using ArchTag = typename Gemm::ArchTag;
  using ClusterShape = typename Gemm::ClusterShape;
  static constexpr int MaxThreadsPerBlock = Gemm::MaxThreadsPerBlock;
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr size_t SharedStorageSize = sizeof(typename Gemm::SharedStorage);
  static dim3 get_grid_shape(const Params& p) { return Gemm::get_grid_shape(p); }
  static dim3 get_block_shape() { return Gemm::get_block_shape(); }
  CUTLASS_DEVICE void operator()(const Params& p, char* smem) { Gemm{}(p, smem); }
};
struct Input {
  int m,n,k,budget;
  void *a,*b,*d,*sa,*sb;
  cudaStream_t stream;
};
struct Config {
  int n,k,e,stages,swizzle;
  bool along_m;
  auto key() const { return std::make_tuple(n,k,e,stages,swizzle,along_m); }
  std::string name() const {
    return "m128n"+std::to_string(n)+"k"+std::to_string(k)+"e"+std::to_string(e)+
        "s"+std::to_string(stages)+"sw"+std::to_string(swizzle)+(along_m?"M":"N");
  }
};
template<int N,int K,int E,int Stages>
cudaError_t launch(const Input& in,const Config& c) {
  using Types = fuse::Mxfp8GemmFamily<N,K,E,Stages>;
  using Gemm = typename Types::PureGemm;
  using Kernel = Entry<Gemm>;
  using Scale = cutlass::detail::Sm1xxBlockScaledConfig<32>;
  typename Gemm::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape = cute::make_shape(in.m,in.n,in.k,1);
  args.mainloop.ptr_A = static_cast<const cutlass::float_e4m3_t*>(in.a);
  args.mainloop.ptr_B = static_cast<const cutlass::float_e4m3_t*>(in.b);
  args.mainloop.dA = cute::make_stride(int64_t(in.k),cute::_1{},int64_t(0));
  args.mainloop.dB = args.mainloop.dA;
  args.mainloop.ptr_SFA = static_cast<const cutlass::float_ue8m0_t*>(in.sa);
  args.mainloop.ptr_SFB = static_cast<const cutlass::float_ue8m0_t*>(in.sb);
  args.mainloop.layout_SFA = Scale::tile_atom_to_shape_SFA(args.problem_shape);
  args.mainloop.layout_SFB = Scale::tile_atom_to_shape_SFB(args.problem_shape);
  args.epilogue.thread.alpha = 1;
  args.epilogue.thread.beta = 0;
  args.epilogue.ptr_D = static_cast<fuse::Bf16*>(in.d);
  args.epilogue.dD = cute::make_stride(int64_t(in.n),cute::_1{},int64_t(0));
  args.epilogue.dC = args.epilogue.dD;
  cudaDeviceProp props{};
  int device=0;
  auto status=cudaGetDevice(&device);
  if(status!=cudaSuccess) return status;
  status=cudaGetDeviceProperties(&props,device);
  if(status!=cudaSuccess) return status;
  args.hw_info.device_id=device;
  args.hw_info.sm_count=in.budget;
  args.scheduler.block_offset=0;
  args.scheduler.max_swizzle_size=c.swizzle;
  using Raster = typename Gemm::TileScheduler::RasterOrderOptions;
  args.scheduler.raster_order=c.along_m?Raster::AlongM:Raster::AlongN;
  if(!Gemm::can_implement(args)||Gemm::get_workspace_size(args)!=0) return cudaErrorNotSupported;
  auto params=Gemm::to_underlying_arguments(args,nullptr);
  fuse::detail::DeviceInfo info{device,props.multiProcessorCount,
      int(props.sharedMemPerBlockOptin),int(props.sharedMemPerMultiprocessor)};
  return fuse::detail::launch_reference_cooperative<Kernel>(params,info,in.budget,in.stream);
}
inline cudaError_t dispatch(const Input& in,const Config& c) {
#define MX_CASE(N,K,E) if(c.n==N && c.k==K && c.e==E) { \
  if(c.stages==0) return launch<N,K,E,0>(in,c); \
  if constexpr(N==128 || K==128) { if(c.stages==2) return launch<N,K,E,2>(in,c); } \
  if constexpr(K==128) { if(c.stages==3) return launch<N,K,E,3>(in,c); } }
  MX_CASE(128,128,32) MX_CASE(128,128,64)
  MX_CASE(256,128,32) MX_CASE(256,128,64)
  MX_CASE(128,256,32) MX_CASE(128,256,64)
  MX_CASE(256,256,32) MX_CASE(256,256,64)
#undef MX_CASE
  return cudaErrorNotSupported;
}
inline std::vector<Config> grid() {
  // Baseline first; the set removes its duplicate from the Cartesian grid.
  std::vector<Config> out{{256,128,64,0,1,true}};
  for(int n:{128,256}) for(int k:{128,256}) for(int e:{32,64})
    for(bool along:{true,false}) for(int sw:{1,4}) {
      Config c{n,k,e,0,sw,along};
      if(c.key()!=out.front().key()) out.push_back(c);
    }
  return out;
}
inline std::vector<Config> neighbors(const Config& c) {
  // One-axis neighbors of each top-2 grid point: missing swizzles and
  // explicit mainloop stages. No changes to communication or quantization.
  std::vector<Config> out;
  for(int sw:{1,2,4,8}) { auto v=c; v.swizzle=sw; out.push_back(v); }
  for(int stages:{2,3}) if(c.k==128 || (c.n==128 && stages==2)) {
    auto v=c; v.stages=stages; out.push_back(v);
  }
  return out;
}
} // namespace mxfp8_search
