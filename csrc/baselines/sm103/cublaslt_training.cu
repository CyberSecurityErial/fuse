#include <cublasLt.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Benchmark-only ABI. BF16 outputs; BF16, tensor-scaled E4M3 or NVFP4 inputs.
// Row-major X[M,K], W[N,K], Y[M,N] map to column-major W^T * X.
namespace {
thread_local std::string error;
void ck(cudaError_t s) { if(s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); }
void ck(cublasStatus_t s) { if(s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS status " + std::to_string(s)); }
void ck(CUresult s) {
  if(s != CUDA_SUCCESS) {
    const char* message = nullptr;
    cuGetErrorString(s, &message);
    throw std::runtime_error(message ? message : "CUDA Driver error");
  }
}
// Benchmark-local resource restriction, not a machine-wide MIG/MPS change.
// Graph tuning, capture and replay must all use this owned nonblocking stream.
struct SmBudget {
  CUgreenCtx context{};
  CUstream stream{};
  std::string info;
  ~SmBudget() {
    if(stream) { cuStreamSynchronize(stream); cuStreamDestroy(stream); }
    if(context) cuGreenCtxDestroy(context);
  }
};
struct Plan {
  cublasLtHandle_t handle{};
  cublasLtMatmulDesc_t op{};
  cublasLtMatrixLayout_t a{}, b{}, c{}, d{};
  void* workspace{};
  size_t capacity{};
  cublasLtMatmulAlgo_t winner{};
  float beta{};
  std::string info;
  ~Plan() {
    if(workspace) cudaFree(workspace);
    for(auto v : {a,b,c,d}) if(v) cublasLtMatrixLayoutDestroy(v);
    if(op) cublasLtMatmulDescDestroy(op);
    if(handle) cublasLtDestroy(handle);
  }
};
int attr(const cublasLtMatmulAlgo_t& a, cublasLtMatmulAlgoConfigAttributes_t k) {
  int result = 0; size_t written = 0;
  if(cublasLtMatmulAlgoConfigGetAttribute(&a,k,&result,sizeof(result),&written)) return -1;
  return result;
}
cublasStatus_t launch(Plan& p, const void* x, const void* w, void* y,
                      cudaStream_t stream, const cublasLtMatmulAlgo_t& algorithm) {
  float alpha=1;
  return cublasLtMatmul(p.handle,p.op,&alpha,w,p.a,x,p.b,&p.beta,y,p.c,y,p.d,
                        &algorithm,p.workspace,p.capacity,stream);
}
float measure(Plan& p, const void* x, const void* w, void* y, cudaStream_t stream,
              const cublasLtMatmulAlgo_t& algorithm, int warmup, int iters, bool graph) {
  cudaGraph_t g{}; cudaGraphExec_t exec{}; cudaEvent_t start{}, stop{};
  try {
    // Reject unsupported algorithms before capture, and surface asynchronous faults.
    ck(launch(p,x,w,y,stream,algorithm)); ck(cudaStreamSynchronize(stream));
    if(graph) {
      ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
      auto status=launch(p,x,w,y,stream,algorithm);
      auto end_status=cudaStreamEndCapture(stream,&g);
      ck(status); ck(end_status);
      ck(cudaGraphInstantiate(&exec,g,0)); ck(cudaGraphUpload(exec,stream));
    }
    auto run=[&]() { if(graph) ck(cudaGraphLaunch(exec,stream)); else ck(launch(p,x,w,y,stream,algorithm)); };
    for(int i=0;i<warmup;i++) run();
    ck(cudaEventCreate(&start)); ck(cudaEventCreate(&stop));
    ck(cudaEventRecord(start,stream));
    for(int i=0;i<iters;i++) run();
    ck(cudaEventRecord(stop,stream)); ck(cudaEventSynchronize(stop));
    float ms; ck(cudaEventElapsedTime(&ms,start,stop));
    cudaEventDestroy(start); cudaEventDestroy(stop);
    if(exec) cudaGraphExecDestroy(exec); if(g) cudaGraphDestroy(g);
    return ms/iters;
  } catch(...) {
    if(start) cudaEventDestroy(start); if(stop) cudaEventDestroy(stop);
    if(exec) cudaGraphExecDestroy(exec); if(g) cudaGraphDestroy(g);
    throw;
  }
}
// cuBLASLt VEC16_UE4M3 scale layout: tiles of 128 rows by 4 scale columns.
__host__ __device__ int64_t sf_index(int64_t row, int64_t block, int64_t k) {
  int64_t tiles_k=(k+63)/64;
  return ((row/128)*tiles_k+block/4)*512+(row%32)*16+((row%128)/32)*4+block%4;
}
__global__ void amax8(const __nv_bfloat16* x, float* maximum, int64_t count) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  float v=0; for(;i<count;i+=int64_t(gridDim.x)*blockDim.x) v=fmaxf(v,fabsf(float(x[i])));
  atomicMax(reinterpret_cast<unsigned*>(maximum),__float_as_uint(v));
}
__global__ void encode8(const __nv_bfloat16* x, __nv_fp8_e4m3* y, const float* maximum,int64_t count) {
  float scale=fmaxf(*maximum/448.0f,1e-12f);
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  for(;i<count;i+=int64_t(gridDim.x)*blockDim.x) y[i]=__nv_fp8_e4m3(float(x[i])/scale);
}
__global__ void finish_scale8(float* scale) { *scale=fmaxf(*scale/448.0f,1e-12f); }
__device__ unsigned encode4(float v) {
  const float values[8]={0,.5f,1,1.5f,2,3,4,6};
  unsigned code=0; float best=fabsf(v);
  for(unsigned i=1;i<8;i++) {
    float d=fabsf(fabsf(v)-values[i]);
    if(d<best || (d==best && (i&1)==0)) {best=d;code=i;}
  }
  return code | ((v<0)?8:0);
}
__global__ void quant4(const __nv_bfloat16* x, unsigned char* y,
                       __nv_fp8_e4m3* scales,int64_t rows,int64_t k) {
  int64_t block=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  for(;block<rows*(k/16);block+=int64_t(gridDim.x)*blockDim.x) {
    int64_t row=block/(k/16), col=block%(k/16), base=block*16;
    float maximum=0;
    for(int j=0;j<16;j++) maximum=fmaxf(maximum,fabsf(float(x[base+j])));
    // Tensor-level scale is exactly 1. Block scale is positive E4M3.
    __nv_fp8_e4m3 encoded(fmaxf(maximum/6.0f,1.0f/512));
    float scale=float(encoded);
    scales[sf_index(row,col,k)]=encoded;
    for(int j=0;j<8;j++) y[block*8+j]=encode4(float(x[base+2*j])/scale) |
                                                (encode4(float(x[base+2*j+1])/scale)<<4);
  }
}
}
extern "C" const char* sm103_last_error() { return error.c_str(); }
extern "C" void* sm103_sm_budget_create(int requested) {
  try {
    if(std::getenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"))
      throw std::runtime_error("MPS active-thread override can expand green-context resources; refusing comparison");
    ck(cudaFree(nullptr));
    CUdevice device{};
    ck(cuCtxGetDevice(&device));
    CUdevResource full{}, group{}, actual{};
    ck(cuDeviceGetDevResource(device, &full, CU_DEV_RESOURCE_TYPE_SM));
    if(requested < 1 || unsigned(requested) > full.sm.smCount)
      throw std::runtime_error("SM budget outside device resource range");
    // Preserve normal cluster co-scheduling whenever the requested count fits.
    // Finer partitions may limit large-cluster algorithms; record this choice,
    // and retune in the real restricted stream rather than reuse a full-card plan.
    unsigned flags = (full.sm.smCoscheduledAlignment &&
                      unsigned(requested) % full.sm.smCoscheduledAlignment)
        ? CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING : 0;
    unsigned groups = 1;
    ck(cuDevSmResourceSplitByCount(&group, &groups, &full, nullptr, flags, requested));
    if(groups != 1 || group.sm.smCount != unsigned(requested))
      throw std::runtime_error("Driver rounded SM budget to " + std::to_string(group.sm.smCount) +
                               "; refusing an unequal-budget result");
    CUdevResourceDesc desc{};
    ck(cuDevResourceGenerateDesc(&desc, &group, 1));
    auto budget = std::make_unique<SmBudget>();
    ck(cuGreenCtxCreate(&budget->context, desc, device, CU_GREEN_CTX_DEFAULT_STREAM));
    ck(cuGreenCtxStreamCreate(&budget->stream, budget->context, CU_STREAM_NON_BLOCKING, 0));
    CUgreenCtx stream_context{};
    ck(cuStreamGetGreenCtx(budget->stream, &stream_context));
    if(stream_context != budget->context) throw std::runtime_error("Stream escaped SM budget");
    ck(cuGreenCtxGetDevResource(stream_context, &actual, CU_DEV_RESOURCE_TYPE_SM));
    if(actual.sm.smCount != unsigned(requested)) throw std::runtime_error("Green context SM count mismatch");
    std::ostringstream info;
    info << "{\"requested_sms\":" << requested << ",\"provisioned_sms\":" << actual.sm.smCount
         << ",\"device_sms\":" << full.sm.smCount << ",\"split_flags\":" << flags
         << ",\"coscheduled_alignment\":" << actual.sm.smCoscheduledAlignment
         << ",\"stream_resource_verified\":true,\"enforcement\":\"cuda_green_context\"}";
    budget->info = info.str();
    return budget.release();
  } catch(const std::exception& e) { error=e.what(); return nullptr; }
}
extern "C" void* sm103_sm_budget_stream(void* budget) { return static_cast<SmBudget*>(budget)->stream; }
extern "C" const char* sm103_sm_budget_info(void* budget) { return static_cast<SmBudget*>(budget)->info.c_str(); }
extern "C" void sm103_sm_budget_destroy(void* budget) { delete static_cast<SmBudget*>(budget); }
extern "C" const char* sm103_plan_info(void* plan) { return static_cast<Plan*>(plan)->info.c_str(); }
extern "C" void sm103_destroy(void* plan) { delete static_cast<Plan*>(plan); }
extern "C" int sm103_run(void* plan,const void* x,const void* w,void* y,void* stream) {
  try { auto& p=*static_cast<Plan*>(plan); ck(launch(p,x,w,y,(cudaStream_t)stream,p.winner));return 0; }
  catch(const std::exception& e) {error=e.what();return -1;}
}
extern "C" int sm103_quantize(int precision,const void* x,void* y,void* scales,
                              int64_t rows,int64_t k,void* stream_ptr) {
  try {
    auto stream=(cudaStream_t)stream_ptr;
    int blocks=int(std::min<int64_t>(4096,(rows*k+255)/256));
    if(precision==8) {
      ck(cudaMemsetAsync(scales,0,sizeof(float),stream));
      amax8<<<blocks,256,0,stream>>>((const __nv_bfloat16*)x,(float*)scales,rows*k);
      encode8<<<blocks,256,0,stream>>>((const __nv_bfloat16*)x,(__nv_fp8_e4m3*)y,(float*)scales,rows*k);
      finish_scale8<<<1,1,0,stream>>>((float*)scales);
    } else if(precision==4 && k%16==0) {
      quant4<<<blocks,256,0,stream>>>((const __nv_bfloat16*)x,(unsigned char*)y,(__nv_fp8_e4m3*)scales,rows,k);
    } else throw std::runtime_error("quantize expects FP8 or FP4 with K multiple of 16");
    ck(cudaGetLastError());return 0;
  } catch(const std::exception& e) {error=e.what();return -1;}
}
extern "C" void* sm103_create_strided(int precision,int64_t m,int64_t n,int64_t k,
    const void* x,const void* w,void* y,const void* xscale,const void* wscale,
    void* stream_ptr,int candidates,int workspace_mib,int warmup,int iters,
    int graph,int math_sms,float beta,int transpose_x,int transpose_w) {
  try {
    if(m<=0||n<=0||k<=0||candidates<1||candidates>1024||iters<1||warmup<0||workspace_mib<0)
      throw std::runtime_error("invalid tuning arguments");
    auto p=std::make_unique<Plan>(); p->beta=beta;
    auto stream=(cudaStream_t)stream_ptr;
    cudaDataType_t dtype=precision==16?CUDA_R_16BF:precision==8?CUDA_R_8F_E4M3:CUDA_R_4F_E2M1;
    if(precision!=16&&precision!=8&&precision!=4) throw std::runtime_error("precision must be 16,8,4");
    if ((transpose_x != 0 && transpose_x != 1) || (transpose_w != 0 && transpose_w != 1) ||
        (precision != 16 && (transpose_x || transpose_w)))
      throw std::runtime_error("transpose views require BF16 operands");
    ck(cublasLtCreate(&p->handle)); ck(cublasLtMatmulDescCreate(&p->op,CUBLAS_COMPUTE_32F,CUDA_R_32F));
    // Row-major Y = X * W^T becomes column-major Y^T = W * X^T.
    // Backward accepts zero-copy transpose views, never a timed/offline copy:
    // dX: X=dY contiguous, W=forward_weight.T; dW: X=dY.T, W=saved_X.T.
    cublasOperation_t ta=transpose_w?CUBLAS_OP_N:CUBLAS_OP_T;
    cublasOperation_t tb=transpose_x?CUBLAS_OP_T:CUBLAS_OP_N;
    ck(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));
    ck(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb)));
    if(math_sms>0) ck(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET,&math_sms,sizeof(math_sms)));
    if(precision!=16) {
      if(!xscale||!wscale) throw std::runtime_error("missing low-precision scales");
      auto mode=precision==8?CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F:CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
      ck(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_A_SCALE_MODE,&mode,sizeof(mode)));
      ck(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_B_SCALE_MODE,&mode,sizeof(mode)));
      ck(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&wscale,sizeof(wscale)));
      ck(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&xscale,sizeof(xscale)));
      int8_t fast_accum=0;
      ck(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_FAST_ACCUM,&fast_accum,sizeof(fast_accum)));
    }
    ck(cublasLtMatrixLayoutCreate(&p->a,dtype,transpose_w?n:k,transpose_w?k:n,transpose_w?n:k));
    ck(cublasLtMatrixLayoutCreate(&p->b,dtype,transpose_x?m:k,transpose_x?k:m,transpose_x?m:k));
    ck(cublasLtMatrixLayoutCreate(&p->c,CUDA_R_16BF,n,m,n));
    ck(cublasLtMatrixLayoutCreate(&p->d,CUDA_R_16BF,n,m,n));
    p->capacity=size_t(workspace_mib)<<20;
    if(p->capacity) ck(cudaMalloc(&p->workspace,p->capacity));
    cublasLtMatmulPreference_t preference{}; ck(cublasLtMatmulPreferenceCreate(&preference));
    ck(cublasLtMatmulPreferenceSetAttribute(preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&p->capacity,sizeof(p->capacity)));
    std::vector<cublasLtMatmulHeuristicResult_t> choices(candidates);int returned=0;
    auto status=cublasLtMatmulAlgoGetHeuristic(p->handle,p->op,p->a,p->b,p->c,p->d,preference,candidates,choices.data(),&returned);
    cublasLtMatmulPreferenceDestroy(preference);ck(status);
    float best=std::numeric_limits<float>::infinity(); int valid=0;
    std::ostringstream records; records<<"[";
    int rejected=0;
    for(int i=0;i<returned;i++) {
      if(choices[i].state!=CUBLAS_STATUS_SUCCESS||choices[i].workspaceSize>p->capacity) continue;
      float ms;
      try {
        ms=measure(*p,x,w,y,stream,choices[i].algo,warmup,iters,graph);
      } catch(const std::exception&) {
        // A heuristic result may be legal for ordinary launch but reject CUDA
        // Graph capture. Keep searching instead of discarding the whole plan.
        ++rejected;
        cudaGetLastError();
        continue;
      }
      if(!std::isfinite(ms)||ms<=0) throw std::runtime_error("invalid candidate time");
      if(valid++) records<<",";
      records<<"{\"index\":"<<i<<",\"algorithm\":"<<attr(choices[i].algo,CUBLASLT_ALGO_CONFIG_ID)
             <<",\"tile\":"<<attr(choices[i].algo,CUBLASLT_ALGO_CONFIG_TILE_ID)
             <<",\"split_k\":"<<attr(choices[i].algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM)
             <<",\"workspace_bytes\":"<<choices[i].workspaceSize<<",\"tune_ms\":"<<ms<<"}";
      if(ms<best) {best=ms;p->winner=choices[i].algo;}
    }
    if(!valid) throw std::runtime_error("no valid cuBLASLt algorithm");
    std::ostringstream info; info<<"{\"precision\":"<<precision<<",\"requested\":"<<candidates
       <<",\"returned\":"<<returned<<",\"valid\":"<<valid<<",\"rejected\":"<<rejected
       <<",\"graph_tuning\":"<<graph
       <<",\"workspace_capacity\":"<<p->capacity<<",\"math_sms\":"<<math_sms
       <<",\"beta\":"<<beta<<",\"transpose_x\":"<<transpose_x<<",\"transpose_w\":"<<transpose_w
       <<",\"best_ms\":"<<best<<",\"algorithm\":"
       <<attr(p->winner,CUBLASLT_ALGO_CONFIG_ID)<<",\"candidates\":"<<records.str()<<"]}";
    p->info=info.str();return p.release();
  } catch(const std::exception& e) {error=e.what();return nullptr;}
}

// Preserve the original contiguous-operand ABI used by published forward baselines.
extern "C" void* sm103_create(int precision,int64_t m,int64_t n,int64_t k,
    const void* x,const void* w,void* y,const void* xscale,const void* wscale,
    void* stream_ptr,int candidates,int workspace_mib,int warmup,int iters,
    int graph,int math_sms,float beta) {
  return sm103_create_strided(precision,m,n,k,x,w,y,xscale,wscale,stream_ptr,
      candidates,workspace_mib,warmup,iters,graph,math_sms,beta,0,0);
}
