// SPDX-License-Identifier: BSD-3-Clause
// Small full-element CPU-oracle bring-up. Not a formal throughput benchmark:
// no large-shape coverage or timing claim is inferred from these checks.
#include "fuse/operators/ulysses/oproj_backward.h"
#include "fuse/operators/primitives/gemm_a2a_mxfp8.h"
#include "../fused_graph.cuh"
#include "mxfp8_reference.cuh"
#include <curand_kernel.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using fuse::Bf16;
void check(cudaError_t s, const char* expression) {
  if (s != cudaSuccess) throw std::runtime_error(std::string(expression) + ": " + cudaGetErrorString(s));
}
#define CUDA_CHECK(expr) check((expr), #expr)

__global__ void random_input(Bf16* p, int count, int columns, uint64_t seed) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
    curandStatePhilox4_32_10_t state;
    curand_init(seed, i, 0, &state);
    // Different row/column dynamic ranges make the two K32 orientations
    // numerically distinct; uniform-amplitude random data can miss axis bugs.
    const int exponent = (i / columns) % 7 - 3 + (i % columns) / 32 % 3;
    p[i] = Bf16(ldexpf((curand_uniform(&state) - .5f) * .25f, exponent));
  }
}

struct Rank {
  int device = 0;
  cudaStream_t stream{};
  std::vector<void*> allocations;
  fuse::Mxfp8OprojBackwardParams params{};
  Bf16 *dy{}, *weight{}, *attention{}, *da{}, *routed{}, *dw{};
  std::vector<Bf16> hdy, hw, ha, expected_da, expected_dw;
  mxfp8_reference::Workspace reference;
  template <class T> T* alloc(size_t n) {
    T* p{}; CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
    allocations.push_back(p); CUDA_CHECK(cudaMemset(p, 0, n * sizeof(T))); return p;
  }
  ~Rank() {
    cudaSetDevice(device);
    cudaStreamSynchronize(stream);
    if (reference.handle) cublasDestroy(reference.handle);
    for (auto* p : allocations) cudaFree(p);
    if (stream) cudaStreamDestroy(stream);
  }
};

std::vector<Bf16> download(Rank& r, Bf16* p, size_t count) {
  CUDA_CHECK(cudaSetDevice(r.device));
  std::vector<Bf16> result(count);
  CUDA_CHECK(cudaMemcpy(result.data(), p, count * sizeof(Bf16), cudaMemcpyDeviceToHost));
  return result;
}

// Independent quantization oracle: the group maximum and exponent use scalar
// double frexp/ldexp, not production indexing, exponent bits or GPU scale bytes.
std::vector<float> represented(const std::vector<Bf16>& source, int rows, int k, bool transpose) {
  std::vector<float> result(size_t(rows) * k);
  for (int row = 0; row < rows; ++row) {
    for (int base = 0; base < k; base += 32) {
      double values[32], amax = 0;
      for (int j = 0; j < 32; ++j) {
        values[j] = float(source[transpose ? size_t(base+j)*rows+row : size_t(row)*k+base+j]);
        amax = std::max(amax, std::abs(values[j]));
      }
      int exponent = 0;
      if (amax > 0) {
        std::frexp(amax, &exponent); exponent -= 9;
        if (amax > std::ldexp(448., exponent)) ++exponent;
      }
      exponent = std::max(-127, std::min(127, exponent));
      for (int j = 0; j < 32; ++j)
        result[size_t(row)*k+base+j] = std::ldexp(
            float(fuse::Fp8E4m3(float(std::ldexp(values[j], -exponent)))), exponent);
    }
  }
  return result;
}

std::vector<Bf16> multiply(const std::vector<float>& a, const std::vector<float>& bt,
                           int m, int n, int k) {
  std::vector<Bf16> result(size_t(m)*n);
  for (int i = 0; i < m; ++i) for (int j = 0; j < n; ++j) {
    double sum = 0;
    for (int x = 0; x < k; ++x) sum += double(a[size_t(i)*k+x])*bt[size_t(j)*k+x];
    result[size_t(i)*n+j] = Bf16(float(sum));
  }
  return result;
}

void expect(const std::vector<Bf16>& actual, const std::vector<Bf16>& expected, const char* label) {
  if (actual.size() != expected.size()) throw std::runtime_error("reference size");
  size_t mismatches = 0;
  float error = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    float a = float(actual[i]), e = float(expected[i]), delta = std::abs(a-e);
    error = std::max(error, delta);
    if (!std::isfinite(a) || delta > .01f + .01f*std::abs(e)) ++mismatches;
  }
  if (mismatches) throw std::runtime_error(std::string(label)+" mismatches="+
      std::to_string(mismatches)+" max_abs="+std::to_string(error));
}

void synchronize(std::vector<Rank>& ranks) {
  for (auto& r : ranks) { CUDA_CHECK(cudaSetDevice(r.device)); CUDA_CHECK(cudaStreamSynchronize(r.stream)); }
}

void run(int world, int m, int h) {
  constexpr int heads=8, dim=128, a=heads*dim;
  std::cout<<"backward_config op=oproj_mxfp8 world="<<world<<" M="<<m<<" H="<<h<<" A="<<a
      <<" scope=cpu_oracle_bringup performance=not_measured\n"<<std::flush;
  std::vector<Rank> ranks(world);
  for (int i=0;i<world;++i) {
    auto& r=ranks[i];r.device=i;
    CUDA_CHECK(cudaSetDevice(i));
    for(int peer=0;peer<world;++peer) if(peer!=i) {
      int enabled=0;CUDA_CHECK(cudaDeviceCanAccessPeer(&enabled,i,peer));
      if(!enabled)throw std::runtime_error("peer access unavailable");
      auto status=cudaDeviceEnablePeerAccess(peer,0);
      if(status==cudaErrorPeerAccessAlreadyEnabled)cudaGetLastError();else CUDA_CHECK(status);
    }
    CUDA_CHECK(cudaStreamCreateWithFlags(&r.stream,cudaStreamNonBlocking));
    r.dy=r.alloc<Bf16>(m*h);r.weight=r.alloc<Bf16>(h*a);r.attention=r.alloc<Bf16>(m*a);
    r.da=r.alloc<Bf16>(m*a);r.routed=r.alloc<Bf16>(m*a);r.dw=r.alloc<Bf16>(h*a);
    auto& d=r.params.data.projection;
    d.weight=r.weight;d.local_grad_attention=r.da;d.ready=r.alloc<uint32_t>((m/128)*(a/256)*fuse::kReadyFlagStride);
    d.local_tokens=m;d.hidden=h;d.q_heads=heads;d.head_dim=dim;d.world_size=world;d.rank=i;d.num_comm_ctas=4;
    d.gemm_policy=fuse::BackwardGemmPolicy::kM128N256;
    r.params.weight.projection={r.dy,r.attention,r.dw,m,h,heads,dim,1.f,0.f};
    size_t bbytes=0,wbytes=0;
    CUDA_CHECK(fuse::oproj_backward_mxfp8_data_workspace_size(d,&bbytes));
    CUDA_CHECK(fuse::oproj_backward_mxfp8_weight_workspace_size(r.params.weight.projection,&wbytes));
    auto* scratch=r.alloc<unsigned char>(std::max(bbytes,wbytes));
    r.params.data.workspace=r.params.weight.workspace=scratch;
    r.params.data.workspace_bytes=r.params.weight.workspace_bytes=std::max(bbytes,wbytes);
    size_t data=0,scale=0;
    CUDA_CHECK(fuse::gemm_a2a_mxfp8_activation_size({m,a,h,1},&data,&scale));
    r.params.data.grad_output={r.alloc<fuse::Fp8E4m3>(data),r.alloc<uint8_t>(scale),data,scale};
    d.peer_done_epoch[i]=r.alloc<uint32_t>(world*fuse::kReadyFlagStride);
    r.reference.initialize(a,r.stream,[&r](size_t bytes)->void*{return r.alloc<unsigned char>(bytes);});
  }
  for(int i=0;i<world;++i)for(int peer=0;peer<world;++peer){
    auto& d=ranks[i].params.data.projection;
    d.peer_grad_attention[peer]=ranks[peer].routed;
    d.peer_done_epoch[peer]=ranks[peer].params.data.projection.peer_done_epoch[peer];
  }
  uint32_t epoch=0;
  for(int generation=0;generation<2;++generation){
    for(auto& r:ranks){
      CUDA_CHECK(cudaSetDevice(r.device));
      random_input<<<64,256,0,r.stream>>>(r.dy,m*h,h,1234+generation*100+r.device);
      random_input<<<64,256,0,r.stream>>>(r.weight,h*a,a,5678+generation*100);
      random_input<<<64,256,0,r.stream>>>(r.attention,m*a,a,9012+generation*100+r.device);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(fuse::quantize_gemm_a2a_mxfp8_activation({m,a,h,1},r.dy,r.params.data.grad_output,r.stream));
    }
    synchronize(ranks);
    for(auto& r:ranks){
      r.hdy=download(r,r.dy,m*h);r.hw=download(r,r.weight,h*a);r.ha=download(r,r.attention,m*a);
      r.expected_da=multiply(represented(r.hdy,m,h,false),represented(r.hw,a,h,true),m,a,h);
      r.expected_dw=multiply(represented(r.hdy,h,m,true),represented(r.ha,a,m,true),h,a,m);
    }
    for(bool graph:{false,true})for(bool causal:{false,true})for(int epilogue:{32,64}){
      for(auto& r:ranks){
        CUDA_CHECK(cudaSetDevice(r.device));auto& d=r.params.data.projection;
        d.causal_load_balanced=causal;
        d.gemm_tuning={epilogue,epilogue==32?1:8,epilogue==64};
        r.params.weight.gemm_tuning=d.gemm_tuning;
        r.params.weight_mode=fuse::WeightGradientMode::kImmediate;
        r.params.weight.projection.beta=0;
      }
      std::vector<std::unique_ptr<fused_graph::Operation>> graphs;
      if(graph)for(auto& r:ranks){
        using L=fused_graph::Launch;
        graphs.emplace_back(new fused_graph::Operation(r.device,r.stream,epoch,
            {L::kOrdinaryStatic,L::kCooperativeDynamic,L::kOrdinaryStatic,
             L::kOrdinaryStatic,L::kOrdinaryDynamic}));
      }
      // Exercise instantiate AND update, with a fresh native publication epoch
      // on each replay. Poison destinations without clearing ready counters:
      // replaying an old epoch must not accidentally pass through old outputs.
      for(int replay=0;replay<(graph?2:1);++replay){
        ++epoch;
        for(auto& r:ranks){
          CUDA_CHECK(cudaSetDevice(r.device));
          r.params.data.projection.epoch=epoch;
          if(graph)graphs[r.device]->prepare(epoch,[&r](uint32_t e,cudaStream_t s){
            r.params.data.projection.epoch=e;
            return fuse::launch_oproj_backward_mxfp8(r.params,s);
          });
          CUDA_CHECK(cudaMemsetAsync(r.da,0xff,size_t(m)*a*sizeof(Bf16),r.stream));
          CUDA_CHECK(cudaMemsetAsync(r.routed,0xff,size_t(m)*a*sizeof(Bf16),r.stream));
          CUDA_CHECK(cudaMemsetAsync(r.dw,0xff,size_t(h)*a*sizeof(Bf16),r.stream));
        }
        synchronize(ranks);
        for(auto& r:ranks){
          CUDA_CHECK(cudaSetDevice(r.device));
          if(graph)graphs[r.device]->launch();
          else CUDA_CHECK(fuse::launch_oproj_backward_mxfp8(r.params,r.stream));
        }
        synchronize(ranks);
      }
      for(auto& r:ranks){
        expect(download(r,r.da,m*a),r.expected_da,"dA numeric");
        expect(download(r,r.dw,h*a),r.expected_dw,"dW numeric");
        std::vector<Bf16> expected(m*a);
        for(int src=0;src<world;++src)for(int row=0;row<m;++row){
          int global=causal?(row<m/2?src*m/2+row:(2*world-src-1)*m/2+row-m/2):src*m+row;
          for(int feature=0;feature<a/world;++feature)
            expected[size_t(global)*(a/world)+feature]=ranks[src].expected_da[size_t(row)*a+r.device*(a/world)+feature];
        }
        expect(download(r,r.routed,m*a),expected,"dA inverse route numeric");
        // A separate bitwise route check uses actual source dA only to test
        // transport, never to construct the numerical GEMM oracle above.
        auto actual=download(r,r.routed,m*a);
        for(int src=0;src<world;++src){
          auto source=download(ranks[src],ranks[src].da,m*a);
          for(int row=0;row<m;++row){
            int global=causal?(row<m/2?src*m/2+row:(2*world-src-1)*m/2+row-m/2):src*m+row;
            for(int f=0;f<a/world;++f)
              if(actual[size_t(global)*(a/world)+f].raw()!=source[size_t(row)*a+r.device*(a/world)+f].raw())
                throw std::runtime_error("dA inverse route bytes");
          }
        }
      }
      std::cout<<"backward_validation op=oproj_mxfp8 world="<<world<<" generation="<<generation
          <<" causal="<<causal<<" epilogue="<<epilogue<<" launch="<<(graph?"graph":"eager")
          <<" B=pass W=pass route_bytes=pass\n"<<std::flush;
      for(auto& r:ranks)if(graph){
        CUDA_CHECK(cudaSetDevice(r.device));graphs[r.device]->reset(epoch);
      }
    }
    // Cross-check the scalable GPU reference against the same outputs that
    // passed the independent CPU FP64 oracle above. This reference is used
    // for large matrices; it must not be introduced without a small cross-check.
    for(auto& r:ranks){
      CUDA_CHECK(cudaSetDevice(r.device));
      const auto b=r.reference.validate({r.dy,h,1},{r.weight,1,a},r.da,m,a,h,r.stream);
      const auto w=r.reference.validate({r.dy,1,h},{r.attention,1,a},r.dw,h,a,m,r.stream);
      if(b.checked!=size_t(m)*a || w.checked!=size_t(h)*a ||
          b.mismatches || w.mismatches || b.nonfinite || w.nonfinite)
        throw std::runtime_error("bounded GPU represented-operand reference disagrees with CPU-checked outputs");
    }
    std::cout<<"backward_reference generation="<<generation<<" independent_gpu=pass CPU_FP64=pass\n"<<std::flush;
  }
  // Deferred B leaves dW untouched; a subsequent explicit W applies beta=1.
  ++epoch;
  std::vector<std::vector<Bf16>> previous;
  for(auto& r:ranks){
    previous.push_back(download(r,r.dw,h*a));
    CUDA_CHECK(cudaSetDevice(r.device));r.params.data.projection.epoch=epoch;
    r.params.weight_mode=fuse::WeightGradientMode::kDeferred;
    CUDA_CHECK(fuse::launch_oproj_backward_mxfp8(r.params,r.stream));
  }
  synchronize(ranks);
  for(auto& r:ranks){
    auto unchanged=download(r,r.dw,h*a);
    for(size_t i=0;i<unchanged.size();++i)if(unchanged[i].raw()!=previous[r.device][i].raw())
      throw std::runtime_error("deferred B modified dW");
    CUDA_CHECK(cudaSetDevice(r.device));r.params.weight.projection.beta=1;
    CUDA_CHECK(fuse::launch_oproj_backward_mxfp8_weight(r.params.weight,r.stream));
  }
  synchronize(ranks);
  for(auto& r:ranks){
    auto expected=r.expected_dw;
    for(size_t i=0;i<expected.size();++i)expected[i]=Bf16(float(expected[i])+float(previous[r.device][i]));
    expect(download(r,r.dw,h*a),expected,"dW beta1");
  }
  std::cout<<"backward_validation deferred=pass beta1=pass mode=cpu_oracle_bringup_only performance=not_measured\n";
}

int main(int argc,char** argv){
  try{
    if(argc!=7 || std::string(argv[1])!="--world" || std::string(argv[3])!="--m" ||
        std::string(argv[5])!="--hidden")throw std::runtime_error("expected --world 4|8 --m 128|256 --hidden 128|256");
    int world=std::stoi(argv[2]),m=std::stoi(argv[4]),h=std::stoi(argv[6]),count=0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if((world!=4&&world!=8)||count<world||(m!=128&&m!=256)||(h!=128&&h!=256))
      throw std::runtime_error("requires CP4/8, CPU-oracle bounded M/H128|256");
    run(world,m,h);return 0;
  }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
