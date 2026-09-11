// SPDX-License-Identifier: BSD-3-Clause
// CPU-light pure MXFP8 cuBLASLt matrix runner: no Torch/TE or communication.
// Reuses the existing algorithm search ABI, random generator and full checker.
#include "../fused_inputs.cuh"
#include "../fused_validation.cuh"
#include <cublas_v2.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef FUSE_MXFP8_SEARCH
#include "mxfp8_search.cuh"
#endif

extern "C" {
const char* sm103_last_error();
const char* sm103_plan_info(void*);
void* sm103_create(int,int64_t,int64_t,int64_t,const void*,const void*,void*,
                  const void*,const void*,void*,int,int,int,int,int,int,float);
int sm103_run(void*,const void*,const void*,void*,void*);
void sm103_destroy(void*);
int sm103_quantize(int,const void*,void*,void*,int64_t,int64_t,void*);
int sm103_decode_mxfp8(const void*,const void*,void*,int64_t,int64_t,void*);
}

namespace {
void check(cudaError_t status) {
  if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void check(cublasStatus_t status) {
  if(status!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS status "+std::to_string(status));
}
void native(int status) { if(status) throw std::runtime_error(sm103_last_error()); }
double median(std::vector<float> values) {
  std::sort(values.begin(),values.end());
  return (double(values[(values.size()-1)/2])+values[values.size()/2])/2;
}
struct Shape { std::string id; int64_t m,n,k; };
struct Resources {
  cudaStream_t stream{};
  cublasHandle_t blas{};
  void* plan{};
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  std::vector<void*> buffers;
  std::vector<cudaEvent_t> events;
  ~Resources() {
    if(executable) cudaGraphExecDestroy(executable);
    if(graph) cudaGraphDestroy(graph);
    if(plan) sm103_destroy(plan);
    if(blas) cublasDestroy(blas);
    for(auto event:events) cudaEventDestroy(event);
    for(void* buffer:buffers) cudaFree(buffer);
    if(stream) cudaStreamDestroy(stream);
  }
  template<class T> T* allocate(size_t count) {
    T* pointer{}; check(cudaMalloc(&pointer,count*sizeof(T))); buffers.push_back(pointer); return pointer;
  }
};

void measure(const Shape& s,int candidates,int compute_budget=0) {
  const size_t sa=((s.m+127)/128)*((s.k+127)/128)*512;
  const size_t sb=((s.n+127)/128)*((s.k+127)/128)*512;
  size_t free=0,total=0; check(cudaMemGetInfo(&free,&total));
  const size_t required=3*(s.m*s.k+s.n*s.k)+4*s.m*s.n+sa+sb+(256ull<<20)+(2ull<<30);
  if(free<required) {
    std::cout<<"RESULT {\"id\":"<<std::quoted(s.id)<<",\"status\":\"memory_skip\",\"required_bytes\":"
             <<required<<",\"free_bytes\":"<<free<<"}\n"<<std::flush; return;
  }
  Resources r;
  check(cudaStreamCreateWithFlags(&r.stream,cudaStreamNonBlocking));
  check(cublasCreate(&r.blas)); check(cublasSetStream(r.blas,r.stream));
  auto* x=r.allocate<__nv_bfloat16>(s.m*s.k);
  auto* w=r.allocate<__nv_bfloat16>(s.n*s.k);
  auto* y=r.allocate<__nv_bfloat16>(s.m*s.n);
  auto* reference=r.allocate<__nv_bfloat16>(s.m*s.n);
  auto* x8=r.allocate<uint8_t>(s.m*s.k);
  auto* w8=r.allocate<uint8_t>(s.n*s.k);
  auto* sx=r.allocate<uint8_t>(sa); auto* sw=r.allocate<uint8_t>(sb);
  auto* inputs=r.allocate<fused_inputs::Scratch>(1);
  auto* validation=r.allocate<fused_validation::Scratch>(1);
  std::vector<float> accepted;
  double drift=0;
  auto launch=[&]() { check(cudaGraphLaunch(r.executable,r.stream)); };
  auto collect=[&]() {
    std::vector<float> samples;
    for(int i=0;i<50;++i) {
      check(cudaEventRecord(r.events[2*i],r.stream)); launch();
      check(cudaEventRecord(r.events[2*i+1],r.stream)); check(cudaEventSynchronize(r.events[2*i+1]));
      float ms=0; check(cudaEventElapsedTime(&ms,r.events[2*i],r.events[2*i+1]));
      if(!std::isfinite(ms)||ms<=0) throw std::runtime_error("invalid CUDA event sample");
      samples.push_back(ms);
    }
    return samples;
  };
  auto verify=[&](int generation,const char* phase) {
    check(fused_validation::launch<true>(reinterpret_cast<uint16_t*>(y),
        fused_validation::DenseOracle{reinterpret_cast<uint16_t*>(reference)},s.m*s.n,validation,0,r.stream));
    check(cudaStreamSynchronize(r.stream));
    fused_validation::Stats stats{};
    check(cudaMemcpy(&stats,validation->result,sizeof(stats),cudaMemcpyDeviceToHost));
    std::cout<<"correctness,pure_mxfp8,id="<<s.id<<",generation="<<generation<<",phase="<<phase
             <<",checked="<<stats.checked<<",mismatches="<<stats.mismatches
             <<",nonfinite="<<stats.nonfinite<<",max_abs="<<stats.max_abs<<'\n';
    if(stats.checked!=uint64_t(s.m*s.n)||stats.mismatches||stats.nonfinite)
      throw std::runtime_error("full MXFP8 cuBLASLt GEMM validation failed");
  };
#ifdef FUSE_MXFP8_SEARCH
  struct Trial { mxfp8_search::Config config; double ms,p95,drift; };
  std::vector<Trial> trials;
  auto queue=mxfp8_search::grid();
  std::set<decltype(queue.front().key())> visited;
  mxfp8_search::Input search_input{int(s.m),int(s.n),int(s.k),compute_budget,x8,w8,y,sx,sw,r.stream};
  if(compute_budget) for(int i=0;i<100;++i) {
    cudaEvent_t event{}; check(cudaEventCreate(&event)); r.events.push_back(event);
    check(cudaEventRecord(event,r.stream));
  }
  auto capture=[&](const mxfp8_search::Config& config) {
    if(r.executable) { check(cudaGraphExecDestroy(r.executable)); r.executable=nullptr; }
    if(r.graph) { check(cudaGraphDestroy(r.graph)); r.graph=nullptr; }
    // Resource setup happens before capture. A rejected resource configuration
    // is recorded, never reported as a numerical or performance winner.
    auto status=mxfp8_search::dispatch(search_input,config);
    if(status==cudaErrorNotSupported || status==cudaErrorInvalidConfiguration ||
       status==cudaErrorCooperativeLaunchTooLarge) return false;
    check(status); check(cudaStreamSynchronize(r.stream));
    check(cudaStreamBeginCapture(r.stream,cudaStreamCaptureModeThreadLocal));
    check(mxfp8_search::dispatch(search_input,config));
    check(cudaStreamEndCapture(r.stream,&r.graph));
    check(cudaGraphInstantiate(&r.executable,r.graph,0));
    check(cudaGraphUpload(r.executable,r.stream)); check(cudaStreamSynchronize(r.stream));
    return true;
  };
#endif
  for(int generation=0;generation<2;++generation) {
    for(int operand=0;operand<2;++operand) {
      const int64_t rows=operand?s.n:s.m;
      auto* source=operand?w:x;
      const uint32_t seed=20260906u+(operand?11u:101u)+generation*1009u;
      check(fused_inputs::generate(reinterpret_cast<uint16_t*>(source),rows*s.k,seed,
                                  operand?.02f:.1f,inputs,r.stream));
      check(cudaStreamSynchronize(r.stream));
      fused_inputs::Stats stats{}; check(cudaMemcpy(&stats,&inputs->result,sizeof(stats),cudaMemcpyDeviceToHost));
      if(stats.finite!=uint64_t(rows*s.k)||!stats.nonzero) throw std::runtime_error("invalid random input");
      std::cout<<"input,pure_mxfp8,id="<<s.id<<",generation="<<generation<<",operand="<<operand
               <<",seed="<<seed<<",count="<<stats.count<<",rms="<<std::sqrt(stats.square_sum/stats.count)
               <<",min="<<stats.minimum<<",max="<<stats.maximum<<'\n';
      native(sm103_quantize(32,source,operand?w8:x8,operand?sw:sx,rows,s.k,r.stream));
      // Decode represented operands into the now-unneeded BF16 source buffers.
      // The oracle tests arithmetic, not model-quality loss from quantization.
      native(sm103_decode_mxfp8(operand?w8:x8,operand?sw:sx,source,rows,s.k,r.stream));
    }
    float alpha=1,beta=0;
    check(cublasGemmEx(r.blas,CUBLAS_OP_T,CUBLAS_OP_N,s.n,s.m,s.k,&alpha,w,CUDA_R_16BF,s.k,
        x,CUDA_R_16BF,s.k,&beta,reference,CUDA_R_16BF,s.n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    check(cudaStreamSynchronize(r.stream));
#ifdef FUSE_MXFP8_SEARCH
    if(compute_budget) {
      if(generation==0) {
        // Reuse allocations, represented operands and the independent oracle
        // across all candidates. Coarse grid then bounded top-2 hill climbing.
        for(int pass=0;pass<3;++pass) {
          for(const auto& config:queue) {
            if(!visited.insert(config.key()).second) continue;
            std::cout<<"RUN cutlass_mxfp8,id="<<s.id<<",candidate="<<config.name()
                     <<",pass="<<pass<<",tested="<<visited.size()<<'\n'<<std::flush;
            if(!capture(config)) {
              std::cout<<"SKIP cutlass_mxfp8,config="<<config.name()<<",reason=resources\n"; continue;
            }
            check(cudaMemsetAsync(y,0xff,2*s.m*s.n,r.stream)); launch(); verify(0,"pre");
            std::vector<double> windows; double gpu_ms=0;
            const auto started=std::chrono::steady_clock::now();
            bool converged=false;
            do {
              auto values=collect(); gpu_ms+=std::accumulate(values.begin(),values.end(),0.0);
              windows.push_back(median(values));
              if(windows.size()>=3 && gpu_ms>=100) {
                auto first=windows.end()-3;
                converged=*std::max_element(first,windows.end()) / *std::min_element(first,windows.end())<=1.05;
              }
            } while(!converged && (windows.size()<3 ||
                std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()<5));
            if(!converged) { std::cout<<"SKIP cutlass_mxfp8,reason=warmup_drift\n"; continue; }
            accepted.clear();
            for(int round=0;round<3;++round) {
              for(int i=0;i<10;++i) { launch(); check(cudaStreamSynchronize(r.stream)); }
              auto values=collect();
              drift=std::abs(median({values.begin()+25,values.end()})/
                             median({values.begin(),values.begin()+25})-1);
              std::cout<<"samples,cutlass_mxfp8,id="<<s.id<<",config="<<config.name()<<",round="<<round<<",ms=[";
              for(size_t i=0;i<values.size();++i) std::cout<<(i?",":"")<<values[i];
              std::cout<<"]\n";
              if(drift<=.05) { accepted=std::move(values); break; }
            }
            verify(0,"post");
            if(accepted.empty()) { std::cout<<"SKIP cutlass_mxfp8,reason=sample_drift\n"; continue; }
            const double ms=median(accepted);
            std::sort(accepted.begin(),accepted.end());
            trials.push_back({config,ms,accepted[47],drift});
            const auto best=std::min_element(trials.begin(),trials.end(),[](auto a,auto b){return a.ms<b.ms;});
            std::cout<<"DONE cutlass_mxfp8,id="<<s.id<<",config="<<config.name()<<",p50_ms="<<ms
                     <<",p95_ms="<<accepted[47]<<",pflops="<<2.*s.m*s.n*s.k/ms*1e-12
                     <<",best_ms="<<best->ms<<",verification=pending_payload1\n"<<std::flush;
          }
          auto ranked=trials;
          std::sort(ranked.begin(),ranked.end(),[](auto a,auto b){return a.ms<b.ms;});
          queue.clear();
          for(size_t i=0;i<std::min(size_t(2),ranked.size());++i)
            for(auto c:mxfp8_search::neighbors(ranked[i].config)) if(!visited.count(c.key())) queue.push_back(c);
          if(queue.empty()) break;
        }
      } else for(const auto& trial:trials) {
        std::cout<<"verify_candidate,cutlass_mxfp8,id="<<s.id<<",config="<<trial.config.name()<<'\n';
        if(!capture(trial.config)) throw std::runtime_error("previously accepted candidate became unsupported");
        check(cudaMemsetAsync(y,0xff,2*s.m*s.n,r.stream)); launch(); verify(1,"post");
      }
      continue;
    }
#endif
    if(!r.plan) {
      r.plan=sm103_create(32,s.m,s.n,s.k,x8,w8,y,sx,sw,r.stream,candidates,256,10,50,1,0,0);
      if(!r.plan) throw std::runtime_error(sm103_last_error());
      std::cout<<"plan,pure_mxfp8,id="<<s.id<<",config="<<sm103_plan_info(r.plan)<<'\n';
      check(cudaStreamBeginCapture(r.stream,cudaStreamCaptureModeThreadLocal));
      native(sm103_run(r.plan,x8,w8,y,r.stream));
      check(cudaStreamEndCapture(r.stream,&r.graph));
      check(cudaGraphInstantiate(&r.executable,r.graph,0));
      check(cudaGraphUpload(r.executable,r.stream)); check(cudaStreamSynchronize(r.stream));
      for(int i=0;i<100;++i) {
        cudaEvent_t event{}; check(cudaEventCreate(&event)); r.events.push_back(event);
        check(cudaEventRecord(event,r.stream));
      }
      check(cudaStreamSynchronize(r.stream));
    }
    check(cudaMemsetAsync(y,0xff,2*s.m*s.n,r.stream)); launch(); verify(generation,"pre");
    if(generation==0) {
      // Three converged windows and >=100 ms actual GPU warmup. Large GEMMs
      // can need more than 5 s just to produce those three 50-sample windows;
      // only apply the wall-time watchdog once convergence can be evaluated.
      // Formal 10+50 selects the FIRST stable round, never fastest.
      const auto start=std::chrono::steady_clock::now();
      std::vector<double> windows; double gpu_ms=0;
      while(true) {
        auto values=collect(); gpu_ms+=std::accumulate(values.begin(),values.end(),0.0);
        windows.push_back(median(values));
        if(windows.size()>=3 && gpu_ms>=100) {
          auto first=windows.end()-3;
          if(*std::max_element(first,windows.end()) / *std::min_element(first,windows.end())<=1.05) break;
        }
        if(windows.size()>=3 &&
           std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()>5)
          throw std::runtime_error("warmup did not converge");
      }
      std::cout<<"warmup,pure_mxfp8,id="<<s.id<<",windows="<<windows.size()
               <<",gpu_ms="<<gpu_ms<<",converged=1\n";
      for(int round=0;round<3;++round) {
        for(int i=0;i<10;++i) { launch(); check(cudaStreamSynchronize(r.stream)); }
        auto values=collect();
        const double first=median({values.begin(),values.begin()+25});
        const double second=median({values.begin()+25,values.end()});
        drift=std::abs(second/first-1);
        std::cout<<"samples,pure_mxfp8,id="<<s.id<<",round="<<round<<",drift="<<drift<<",ms=[";
        for(size_t i=0;i<values.size();++i) std::cout<<(i?",":"")<<values[i];
        std::cout<<"]\n";
        if(drift<=.05) { accepted=std::move(values); break; }
      }
      verify(generation,"post");
    }
  }
#ifdef FUSE_MXFP8_SEARCH
  if(compute_budget) {
    if(trials.empty()) throw std::runtime_error("no stable CUTLASS candidate");
    for(const auto& t:trials) std::cout<<"RESULT {\"id\":"<<std::quoted(s.id)
        <<",\"status\":\"passed\",\"backend\":\"cutlass_mxfp8\",\"config\":"<<std::quoted(t.config.name())
        <<",\"m\":"<<s.m<<",\"n\":"<<s.n<<",\"k\":"<<s.k<<",\"compute_ctas\":"<<compute_budget
        <<",\"p50_ms\":"<<t.ms<<",\"p95_ms\":"<<t.p95<<",\"drift\":"<<t.drift
        <<",\"pflops\":"<<2.*s.m*s.n*s.k/t.ms*1e-12<<",\"payloads\":2,\"full_numeric\":true}\n";
    std::cout<<std::flush;
    return;
  }
#endif
  if(accepted.empty()) {
    std::cout<<"RESULT {\"id\":"<<std::quoted(s.id)<<",\"status\":\"unstable\"}\n"<<std::flush; return;
  }
  const double p50=median(accepted);
  std::sort(accepted.begin(),accepted.end());
  std::cout<<"RESULT {\"id\":"<<std::quoted(s.id)<<",\"status\":\"passed\",\"m\":"<<s.m
           <<",\"n\":"<<s.n<<",\"k\":"<<s.k<<",\"p50_ms\":"<<p50<<",\"p95_ms\":"<<accepted[47]
           <<",\"pflops\":"<<2.0*s.m*s.n*s.k/p50*1e-12<<",\"drift\":"<<drift
           <<",\"full_numeric\":true,\"payloads\":2,\"plan\":"<<sm103_plan_info(r.plan)<<"}\n"<<std::flush;
}
}

int main(int argc,char** argv) {
  try {
    if(argc!=3 && argc!=4) throw std::runtime_error("usage: mxfp8_gemm_bench MATRIX.txt CANDIDATES [COMPUTE_CTAS]");
    const int budget=argc==4?std::stoi(argv[3]):0;
#ifndef FUSE_MXFP8_SEARCH
    if(budget) throw std::runtime_error("build the optional CUTLASS search target first");
#endif
    const int candidates=std::stoi(argv[2]);
    if(candidates<1||candidates>256) throw std::runtime_error("candidates must be 1..256");
    std::ifstream input(argv[1]);
    if(!input) throw std::runtime_error("matrix file missing");
    std::vector<Shape> shapes;
    Shape s;
    while(input>>s.id>>s.m>>s.n>>s.k) {
      if(!std::regex_match(s.id,std::regex("[A-Za-z0-9_.-]{1,160}"))||
         std::min({s.m,s.n,s.k})<=0||std::max({s.m,s.n,s.k})>2147483647||s.k%32)
        throw std::runtime_error("invalid matrix record");
      shapes.push_back(s);
    }
    if(!input.eof()||shapes.empty()||shapes.size()>256) throw std::runtime_error("invalid matrix");
    cudaDeviceProp props{}; check(cudaGetDeviceProperties(&props,0));
    if(argc==4 && (budget<1||budget>props.multiProcessorCount)) throw std::runtime_error("invalid compute CTA budget");
    std::cout<<std::setprecision(9)<<"config,pure_mxfp8,launch=graph,warmup=10,samples=50,candidates="<<candidates
             <<(budget ? "" : ",sm_budget=full_device")
             <<",compute_ctas="<<budget<<",device_sms="<<props.multiProcessorCount
             <<",includes_quantization=0,includes_communication=0,output=bf16,group_k=32,scale=ue8m0\n";
    for(const auto& shape:shapes) measure(shape,candidates,budget);
    return 0;
  } catch(const std::exception& e) { std::cerr<<"mxfp8_gemm_bench: "<<e.what()<<'\n'; return 1; }
}
