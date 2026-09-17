// SPDX-License-Identifier: BSD-3-Clause
// Benchmark-only native adapter to unmodified DeepGEMM 78b6900 kernels.
// This is NOT the Python API/JIT heuristic: M128/M256, N128/K64, swap-AB,
// two-CTA multicast and full 148-SM budget are explicit measured candidates.
// Descriptor construction follows upstream runtime_utils.hpp; stage selection
// follows SM100ArchSpec::get_pipeline_config. Inputs are pre-delivered. Packing,
// descriptors, allocation and validation copies are OUTSIDE pure-GEMM timing.
#include <deep_gemm/impls/sm100_bf16_gemm.cuh>
#include <cuda_runtime.h>
#include <cuda.h>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string error;
void check(cudaError_t status) {
  if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
CUtensorMap descriptor(void* ptr,int inner,int outer,int box_outer) {
  CUtensorMap map{};
  const cuuint64_t dims[]{cuuint64_t(inner),cuuint64_t(outer)},strides[]{cuuint64_t(inner)*2};
  const cuuint32_t box[]{64,cuuint32_t(box_outer)},element_strides[]{1,1};
  if(cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,ptr,dims,strides,
      box,element_strides,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)!=CUDA_SUCCESS)
    throw std::runtime_error("DeepGEMM BF16 descriptor failed");
  return map;
}
struct Plan {
  void* a=nullptr; void* d=nullptr; int* counts=nullptr;
  CUtensorMap ma{},mb{},md{};
  int device=0,capacity=0,n=0,k=0,block_m=0;
  std::vector<int> rows;
  cudaError_t (*launch)(Plan&,cudaStream_t)=nullptr;
  ~Plan() { cudaSetDevice(device); cudaFree(a); cudaFree(d); cudaFree(counts); }
};

template<int Experts,int M>
cudaError_t launch(Plan& p,cudaStream_t stream) {
  // Upstream BF16 layout: two store stages; reserve worst-case barrier budget.
  constexpr int stages=(232448-(16*128*2*2+32*8*3+2*8*3+8+4))/((M/2+128)*64*2);
  using Storage=deep_gemm::layout::SM100BF16GemmSharedStorage<
      stages,2,2,M/2,128,64,16,128,cutlass::bfloat16_t>;
  auto kernel=deep_gemm::sm100_bf16_gemm_impl<
      cute::UMMA::Major::K,cute::UMMA::Major::K,0,0,0,
      M,128,64,Experts,128,128,128,stages,128,128,2,true,148,128,
      true,true,deep_gemm::GemmType::MGroupedMasked,false,cutlass::bfloat16_t,
      deep_gemm::epilogue::transform::EpilogueIdentity,100>;
  auto status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Storage));
  if(status!=cudaSuccess) return status;
  cudaLaunchAttribute attr{};
  attr.id=cudaLaunchAttributeClusterDimension; attr.val.clusterDim={2,1,1};
  cudaLaunchConfig_t config{};
  config.gridDim=dim3(148); config.blockDim=dim3(256);
  config.dynamicSmemBytes=sizeof(Storage); config.stream=stream;
  config.attrs=&attr; config.numAttrs=1;
  deep_gemm::epilogue::transform::EpilogueIdentity epilogue{};
  return cudaLaunchKernelEx(&config,kernel,p.counts,uint32_t(p.capacity),uint32_t(p.n),
      uint32_t(p.k),epilogue,p.ma,p.mb,p.md);
}
template<int M> void select(Plan& p,int experts) {
  // Local expert counts in the frozen 17-profile EP4/8 catalog.
#define GROUP(E) case E: p.launch=&launch<E,M>; break
  switch(experts) {
    GROUP(1); GROUP(2); GROUP(8); GROUP(16); GROUP(20); GROUP(32); GROUP(36);
    GROUP(40); GROUP(48); GROUP(64); GROUP(72); GROUP(96); GROUP(128);
    default: throw std::runtime_error("DeepGEMM adapter: unsupported local expert count");
  }
#undef GROUP
}
}

extern "C" const char* grouped_deepgemm_error() { return error.c_str(); }
extern "C" void* grouped_deepgemm_create(int experts,const int64_t* offsets,int capacity,
    int n,int k,const void* inputs,const void* weights,int block_m,cudaStream_t stream) {
  Plan* p=nullptr;
  try {
    p=new Plan; check(cudaGetDevice(&p->device));
    cudaDeviceProp props{}; check(cudaGetDeviceProperties(&props,p->device));
    if(props.major!=10 || props.minor!=3 || props.multiProcessorCount!=148 ||
        n%256 || k%64 || (block_m!=128 && block_m!=256))
      throw std::runtime_error("DeepGEMM adapter requires SM103/148SM,N%256,K%64");
    p->capacity=(capacity+255)/256*256; p->n=n; p->k=k; p->block_m=block_m;
    for(int e=0;e<experts;++e) p->rows.push_back(int(offsets[e+1]-offsets[e]));
    check(cudaMalloc(&p->a,size_t(experts)*p->capacity*k*2));
    check(cudaMalloc(&p->d,size_t(experts)*p->capacity*n*2));
    check(cudaMalloc(&p->counts,experts*sizeof(int)));
    check(cudaMemsetAsync(p->a,0,size_t(experts)*p->capacity*k*2,stream));
    check(cudaMemsetAsync(p->d,0x7f,size_t(experts)*p->capacity*n*2,stream));
    check(cudaMemcpyAsync(p->counts,p->rows.data(),experts*sizeof(int),cudaMemcpyHostToDevice,stream));
    for(int e=0;e<experts;++e) if(p->rows[e])
      check(cudaMemcpyAsync(static_cast<char*>(p->a)+size_t(e)*p->capacity*k*2,
          static_cast<const char*>(inputs)+size_t(e)*capacity*k*2,
          size_t(p->rows[e])*k*2,cudaMemcpyDeviceToDevice,stream));
    p->ma=descriptor(p->a,k,p->capacity*experts,block_m/2);
    p->mb=descriptor(const_cast<void*>(weights),k,n*experts,128);
    p->md=descriptor(p->d,n,p->capacity*experts,16);
    if(block_m==128) select<128>(*p,experts); else select<256>(*p,experts);
    check(cudaStreamSynchronize(stream));
    return p;
  } catch(const std::exception& e) { error=e.what(); delete p; return nullptr; }
}
extern "C" cudaError_t grouped_deepgemm_launch(void* plan,cudaStream_t stream) {
  return static_cast<Plan*>(plan)->launch(*static_cast<Plan*>(plan),stream);
}
extern "C" cudaError_t grouped_deepgemm_read(void* plan,void* output,int capacity,cudaStream_t stream) {
  auto& p=*static_cast<Plan*>(plan);
  auto status=cudaMemsetAsync(output,0x7f,p.rows.size()*size_t(capacity)*p.n*2,stream);
  if(status!=cudaSuccess) return status;
  for(size_t e=0;e<p.rows.size();++e) if(p.rows[e]) {
    status=cudaMemcpyAsync(static_cast<char*>(output)+e*capacity*p.n*2,
        static_cast<char*>(p.d)+e*p.capacity*p.n*2,size_t(p.rows[e])*p.n*2,
        cudaMemcpyDeviceToDevice,stream);
    if(status!=cudaSuccess) return status;
  }
  return cudaStreamSynchronize(stream);
}
extern "C" void grouped_deepgemm_destroy(void* p) { delete static_cast<Plan*>(p); }
