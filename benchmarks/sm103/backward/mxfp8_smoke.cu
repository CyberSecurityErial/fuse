// SPDX-License-Identifier: BSD-3-Clause
// Small full-element CPU-oracle bring-up. Not a formal throughput benchmark:
// no large-shape coverage or timing claim is inferred from these checks.
#include "fuse/operators/ulysses/oproj_backward.h"
#include "fuse/operators/ulysses/qkv_backward.h"
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

// Cross-check cached and freshly generated independent references on outputs
// already checked with CPU FP64. A poisoned actual value MUST still fail after
// caching; restore its exact original bytes before any subsequent operation.
template<class Left,class Right>
void check_reference_cache(Rank& r,Left a,Right bt,Bf16* actual,int m,int n,int k){
  auto* cache=r.alloc<Bf16>(size_t(m)*n);
  // Rank::alloc initializes on the default stream; finish that memset before
  // the nonblocking oracle stream writes its independent expected values.
  CUDA_CHECK(cudaStreamSynchronize(nullptr));
  for(bool reuse:{false,true}){
    const auto stats=r.reference.validate_views(a,bt,actual,m,n,k,r.stream,cache,reuse);
    if(stats.checked!=size_t(m)*n || stats.mismatches || stats.nonfinite)
      throw std::runtime_error("cached independent oracle disagrees with CPU-checked output");
  }
  uint16_t saved=0;
  CUDA_CHECK(cudaMemcpy(&saved,actual,sizeof(saved),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemsetAsync(actual,0xff,sizeof(saved),r.stream));
  const auto bad=r.reference.validate_views(a,bt,actual,m,n,k,r.stream,cache,true);
  CUDA_CHECK(cudaMemcpyAsync(actual,&saved,sizeof(saved),cudaMemcpyHostToDevice,r.stream));
  CUDA_CHECK(cudaStreamSynchronize(r.stream));
  if(bad.checked!=size_t(m)*n || (!bad.mismatches && !bad.nonfinite))
    throw std::runtime_error("cached oracle missed poisoned actual output");
}

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
            {L::kOrdinaryDynamic,L::kCooperativeDynamic,L::kOrdinaryDynamic,
             L::kOrdinaryDynamic,L::kOrdinaryDynamic}));
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
      check_reference_cache(r,mxfp8_reference::Operand{r.dy,h,1},
          mxfp8_reference::Operand{r.weight,1,a},r.da,m,a,h);
      check_reference_cache(r,mxfp8_reference::Operand{r.dy,1,h},
          mxfp8_reference::Operand{r.attention,1,a},r.dw,h,a,m);
    }
    std::cout<<"backward_reference generation="<<generation<<" independent_gpu=pass CPU_FP64=pass cache_fault_check=pass\n"<<std::flush;
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

// QKV W is separately checked with its real packed-head width, not by
// relabeling the OProj case. B/A2A is deliberately NOT claimed by this test.
void run_qkv_weight(int m, int h) {
  Rank r;
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaStreamCreateWithFlags(&r.stream,cudaStreamNonBlocking));
  for (int q_heads : {8,16}) {
    constexpr int kv_heads=8, dim=128;
    const int packed=(q_heads+2*kv_heads)*dim;
    auto* gradient=r.alloc<Bf16>(size_t(m)*packed);
    auto* input=r.alloc<Bf16>(size_t(m)*h);
    auto* output=r.alloc<Bf16>(size_t(packed)*h);
    fuse::Mxfp8QkvBackwardWeightParams p{};
    p.projection={gradient,input,output,m,h,q_heads,kv_heads,dim,.75f,0.f};
    CUDA_CHECK(fuse::qkv_backward_mxfp8_weight_workspace_size(p.projection,&p.workspace_bytes));
    p.workspace=r.alloc<unsigned char>(p.workspace_bytes);
    auto invalid=p;
    invalid.workspace_bytes--;
    if(fuse::launch_qkv_backward_mxfp8_weight(invalid,r.stream)!=cudaErrorInvalidValue)
      throw std::runtime_error("QKV W undersized scratch accepted");
    invalid=p;invalid.projection.kv_heads=0;
    if(fuse::launch_qkv_backward_mxfp8_weight(invalid,r.stream)!=cudaErrorInvalidValue)
      throw std::runtime_error("QKV W invalid packed heads accepted");
    for(int generation=0;generation<2;++generation) {
      random_input<<<64,256,0,r.stream>>>(gradient,m*packed,packed,2345+100*generation);
      random_input<<<64,256,0,r.stream>>>(input,m*h,h,6789+100*generation);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaStreamSynchronize(r.stream));
      const auto g=represented(download(r,gradient,size_t(m)*packed),packed,m,true);
      const auto x=represented(download(r,input,size_t(m)*h),h,m,true);
      std::vector<double> product(size_t(packed)*h);
      for(int row=0;row<packed;++row)for(int col=0;col<h;++col) {
        double sum=0;
        for(int k=0;k<m;++k)sum+=double(g[size_t(row)*m+k])*x[size_t(col)*m+k];
        product[size_t(row)*h+col]=sum;
      }
      for(bool graph:{false,true})for(int epilogue:{32,64}) {
        p.gemm_tuning={epilogue,epilogue==32?1:8,epilogue==64};
        std::vector<Bf16> previous(product.size(),Bf16(.125f)),expected(product.size());
        for(float beta:{0.f,1.f}) {
          p.projection.beta=beta;
          using L=fused_graph::Launch;
          std::unique_ptr<fused_graph::Operation> operation;
          if(graph)operation.reset(new fused_graph::Operation(0,r.stream,0,
              {L::kOrdinaryDynamic,L::kOrdinaryDynamic,L::kOrdinaryDynamic}));
          for(int replay=1;replay<=(graph?2:1);++replay) {
            if(graph)operation->prepare(replay,[&](uint32_t,cudaStream_t s){
              return fuse::launch_qkv_backward_mxfp8_weight(p,s);
            });
            // beta=0 must overwrite poison; beta=1 must read the original C.
            if(beta==0)CUDA_CHECK(cudaMemsetAsync(output,0xff,product.size()*sizeof(Bf16),r.stream));
            else CUDA_CHECK(cudaMemcpyAsync(output,previous.data(),product.size()*sizeof(Bf16),
                cudaMemcpyHostToDevice,r.stream));
            if(graph)operation->launch();
            else CUDA_CHECK(fuse::launch_qkv_backward_mxfp8_weight(p,r.stream));
            CUDA_CHECK(cudaStreamSynchronize(r.stream));
            for(size_t i=0;i<product.size();++i)
              expected[i]=Bf16(float(.75*product[i]+beta*float(previous[i])));
            expect(download(r,output,product.size()),expected,"QKV W CPU FP64 numeric");
          }
          if(graph)operation->reset(2);
        }
      }
      std::cout<<"backward_weight_validation op=qkv_mxfp8 M="<<m<<" H="<<h
          <<" Q="<<q_heads<<" KV="<<kv_heads<<" generation="<<generation
          <<" CPU_FP64=pass alpha_beta=pass eager_graph=pass performance=not_measured\n"<<std::flush;
    }
  }
}

struct QkvRank : Rank {
  fuse::Mxfp8QkvBackwardParams qkv{};
  Bf16* gradients[3]{};
  fuse::Mxfp8Activation quantized[3]{};
  std::vector<Bf16> host_gradients[3];
};

void run_qkv(int world,int m,int h) {
  constexpr int q=16,kv=8,dim=128,packed=(q+2*kv)*dim;
  std::vector<QkvRank> ranks(world);
  auto sync=[&]{for(auto& r:ranks){CUDA_CHECK(cudaSetDevice(r.device));
      CUDA_CHECK(cudaStreamSynchronize(r.stream));}};
  for(int rank=0;rank<world;++rank) {
    auto& r=ranks[rank];r.device=rank;
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaStreamCreateWithFlags(&r.stream,cudaStreamNonBlocking));
    r.weight=r.alloc<Bf16>(size_t(packed)*h);r.attention=r.alloc<Bf16>(size_t(m)*h);
    r.da=r.alloc<Bf16>(size_t(m)*h);r.dw=r.alloc<Bf16>(size_t(packed)*h);
    r.routed=r.alloc<Bf16>(size_t(m)*packed);
    r.reference.initialize(h,r.stream,[&r](size_t bytes)->void*{return r.alloc<unsigned char>(bytes);});
    for(int kind=0;kind<3;++kind) {
      const int width=(kind==0?q:kv)/world*dim;
      const fuse::GemmProblem shape{m*world,h,width,1};
      r.gradients[kind]=r.alloc<Bf16>(size_t(m)*world*width);
      size_t data=0,scales=0;
      CUDA_CHECK(fuse::gemm_a2a_mxfp8_activation_size(shape,&data,&scales));
      r.quantized[kind]={r.alloc<fuse::Fp8E4m3>(data),r.alloc<uint8_t>(scales),data,scales};
    }
    auto& d=r.qkv.data.projection;
    d.local_tokens=m;d.hidden=h;d.q_heads=q;d.kv_heads=kv;d.head_dim=dim;
    d.world_size=world;d.rank=rank;d.num_comm_ctas=4;d.epoch=1;
    d.weight=r.weight;d.grad_input=r.da;d.peer_dqkv_staging[rank]=r.routed;
    d.peer_ready[rank]=r.alloc<uint32_t>(fuse::qkv_backward_ready_elements(d));
    d.gemm_policy=fuse::BackwardGemmPolicy::kM128N256;
    r.qkv.weight.projection={r.routed,r.attention,r.dw,m,h,q,kv,dim,1.f,0.f};
    size_t b=0,w=0;
    CUDA_CHECK(fuse::qkv_backward_mxfp8_data_workspace_size(d,&b));
    CUDA_CHECK(fuse::qkv_backward_mxfp8_weight_workspace_size(r.qkv.weight.projection,&w));
    auto* scratch=r.alloc<unsigned char>(std::max(b,w));
    r.qkv.data.workspace=r.qkv.weight.workspace=scratch;
    r.qkv.data.workspace_bytes=r.qkv.weight.workspace_bytes=std::max(b,w);
  }
  // P2P visibility was established by the preceding O bring-up. No host cat
  // feeds production; the host gather below is solely an independent oracle.
  for(auto& r:ranks)for(int peer=0;peer<world;++peer) {
    const auto& source=ranks[peer];
    r.qkv.data.peer_input[peer]={source.quantized[0],source.quantized[1],source.quantized[2],
        source.gradients[0],source.gradients[1],source.gradients[2]};
  }
  for(int generation=0;generation<2;++generation) {
    for(auto& r:ranks) {
      CUDA_CHECK(cudaSetDevice(r.device));
      for(int kind=0;kind<3;++kind) {
        const int width=(kind==0?q:kv)/world*dim;
        random_input<<<64,256,0,r.stream>>>(r.gradients[kind],m*world*width,width,
            12340+1000*generation+100*kind+r.device);
        CUDA_CHECK(fuse::quantize_gemm_a2a_mxfp8_activation({m*world,h,width,1},
            r.gradients[kind],r.quantized[kind],r.stream));
      }
      random_input<<<64,256,0,r.stream>>>(r.weight,packed*h,h,23450+generation*1000);
      random_input<<<64,256,0,r.stream>>>(r.attention,m*h,h,34560+generation*1000+r.device);
      CUDA_CHECK(cudaGetLastError());
    }
    sync();
    for(auto& r:ranks) {
      for(int kind=0;kind<3;++kind)r.host_gradients[kind]=download(r,r.gradients[kind],
          size_t(m)*(kind==0?q:kv)*dim);
      r.hw=download(r,r.weight,size_t(packed)*h);r.ha=download(r,r.attention,size_t(m)*h);
    }
    for(bool causal:{false,true}) {
      if(causal && m%256)continue;
      for(auto& r:ranks) {
        r.hdy.resize(size_t(m)*packed);
        for(int row=0;row<m;++row) {
          const int global=causal?(row<m/2?r.device*m/2+row:
              (2*world-r.device-1)*m/2+row-m/2):r.device*m+row;
          for(int head=0;head<q+2*kv;++head) {
            const int kind=head<q?0:(head<q+kv?1:2);
            const int within=head-(kind==0?0:(kind==1?q:q+kv));
            const int local=(kind==0?q:kv)/world,peer=within/local,own=within%local;
            for(int c=0;c<dim;++c)r.hdy[size_t(row)*packed+head*dim+c]=
                ranks[peer].host_gradients[kind][size_t(global)*local*dim+own*dim+c];
          }
        }
        r.expected_da=multiply(represented(r.hdy,m,packed,false),represented(r.hw,h,packed,true),m,h,packed);
        r.expected_dw=multiply(represented(r.hdy,packed,m,true),represented(r.ha,h,m,true),packed,h,m);
      }
      for(bool graph:{false,true})for(int epilogue:{32,64}) {
        std::vector<std::unique_ptr<fused_graph::Operation>> graphs;
        for(auto& r:ranks) {
          CUDA_CHECK(cudaSetDevice(r.device));
          r.qkv.data.projection.causal_load_balanced=causal;
          r.qkv.data.projection.gemm_tuning={epilogue,epilogue==32?1:8,epilogue==64};
          r.qkv.weight.gemm_tuning=r.qkv.data.projection.gemm_tuning;
          using L=fused_graph::Launch;
          if(graph)graphs.emplace_back(new fused_graph::Operation(r.device,r.stream,0,
              {L::kOrdinaryDynamic,L::kCooperativeDynamic,L::kOrdinaryDynamic,
               L::kOrdinaryDynamic,L::kOrdinaryDynamic}));
        }
        for(int replay=1;replay<=(graph?2:1);++replay) {
          for(auto& r:ranks) {
            CUDA_CHECK(cudaSetDevice(r.device));
            if(graph)graphs[r.device]->prepare(replay,[&](uint32_t e,cudaStream_t s){
              r.qkv.data.projection.epoch=e;return fuse::launch_qkv_backward_mxfp8(r.qkv,s);
            });
            CUDA_CHECK(cudaMemsetAsync(r.da,0xff,size_t(m)*h*sizeof(Bf16),r.stream));
            CUDA_CHECK(cudaMemsetAsync(r.dw,0xff,size_t(packed)*h*sizeof(Bf16),r.stream));
            CUDA_CHECK(cudaMemsetAsync(r.routed,0xff,size_t(m)*packed*sizeof(Bf16),r.stream));
          }
          sync();
          for(auto& r:ranks) {
            CUDA_CHECK(cudaSetDevice(r.device));
            if(graph)graphs[r.device]->launch();
            else CUDA_CHECK(fuse::launch_qkv_backward_mxfp8(r.qkv,r.stream));
          }
          sync();
          for(auto& r:ranks) {
            expect(download(r,r.da,size_t(m)*h),r.expected_da,"QKV dX CPU FP64");
            expect(download(r,r.dw,size_t(packed)*h),r.expected_dw,"QKV dW CPU FP64");
            const auto master=download(r,r.routed,size_t(m)*packed);
            for(size_t i=0;i<master.size();++i)if(master[i].raw()!=r.hdy[i].raw())
              throw std::runtime_error("QKV ORIGINAL BF16 route bytes");
            // Cross-check the bounded GPU oracle against outputs already
            // checked with the independent CPU FP64/host-gather reference.
            mxfp8_reference::QkvGradient original{};
            original.m=m;original.q_heads=q;original.kv_heads=kv;
            original.world=world;original.rank=r.device;original.causal=causal;
            for(int peer=0;peer<world;++peer)for(int kind=0;kind<3;++kind)
              original.source[peer][kind]=ranks[peer].gradients[kind];
            CUDA_CHECK(cudaSetDevice(r.device));
            const auto dx=r.reference.validate_views(original,
                mxfp8_reference::Operand{r.weight,1,h},r.da,m,h,packed,r.stream);
            original.transpose=true;
            const auto dw=r.reference.validate_views(original,
                mxfp8_reference::Operand{r.attention,1,h},r.dw,packed,h,m,r.stream);
            if(dx.checked!=size_t(m)*h || dw.checked!=size_t(packed)*h ||
                dx.mismatches || dw.mismatches || dx.nonfinite || dw.nonfinite)
              throw std::runtime_error("QKV bounded GPU oracle cross-check");
            if(graph && epilogue==64){
              original.transpose=false;
              check_reference_cache(r,original,mxfp8_reference::Operand{r.weight,1,h},r.da,m,h,packed);
              original.transpose=true;
              check_reference_cache(r,original,mxfp8_reference::Operand{r.attention,1,h},r.dw,packed,h,m);
            }
            // Full W just prepared this scratch; preserve it and verify that
            // the compute-only diagnostic executes the same native GEMM.
            CUDA_CHECK(cudaMemsetAsync(r.dw,0xff,size_t(packed)*h*sizeof(Bf16),r.stream));
            CUDA_CHECK(fuse::launch_qkv_backward_mxfp8_weight_compute_reference(r.qkv.weight,r.stream));
            expect(download(r,r.dw,size_t(packed)*h),r.expected_dw,"QKV prepared dW CPU FP64");
          }
        }
        for(auto& r:ranks)if(graph){CUDA_CHECK(cudaSetDevice(r.device));graphs[r.device]->reset(2);}
        std::cout<<"backward_validation op=qkv_mxfp8 world="<<world<<" M="<<m
            <<" H="<<h<<" generation="<<generation<<" causal="<<causal<<" epilogue="<<epilogue
            <<" launch="<<(graph?"graph":"eager")
            <<" B=pass W=pass original_route_bytes=pass bounded_GPU_oracle=pass prepared_W=pass performance=not_measured\n"<<std::flush;
      }
    }
  }
  std::vector<std::vector<Bf16>> previous;
  for(auto& r:ranks) {
    previous.push_back(download(r,r.dw,size_t(packed)*h));
    CUDA_CHECK(cudaSetDevice(r.device));
    r.qkv.weight_mode=fuse::WeightGradientMode::kDeferred;
    CUDA_CHECK(cudaMemsetAsync(r.da,0xff,size_t(m)*h*sizeof(Bf16),r.stream));
    CUDA_CHECK(cudaMemsetAsync(r.routed,0xff,size_t(m)*packed*sizeof(Bf16),r.stream));
    CUDA_CHECK(fuse::launch_qkv_backward_mxfp8(r.qkv,r.stream));
  }
  sync();
  for(auto& r:ranks) {
    const auto dw=download(r,r.dw,size_t(packed)*h);
    for(size_t i=0;i<dw.size();++i)if(dw[i].raw()!=previous[r.device][i].raw())
      throw std::runtime_error("QKV deferred B modified dW");
    expect(download(r,r.da,size_t(m)*h),r.expected_da,"QKV deferred dX");
    const auto master=download(r,r.routed,size_t(m)*packed);
    for(size_t i=0;i<master.size();++i)if(master[i].raw()!=r.hdy[i].raw())
      throw std::runtime_error("QKV deferred W lease is not original BF16");
    CUDA_CHECK(cudaSetDevice(r.device));r.qkv.weight.projection.beta=1;
    CUDA_CHECK(fuse::launch_qkv_backward_mxfp8_weight(r.qkv.weight,r.stream));
  }
  sync();
  for(auto& r:ranks) {
    auto expected=r.expected_dw;
    for(size_t i=0;i<expected.size();++i)
      expected[i]=Bf16(float(expected[i])+float(previous[r.device][i]));
    expect(download(r,r.dw,size_t(packed)*h),expected,"QKV deferred W beta1");
  }
  std::cout<<"backward_validation op=qkv_mxfp8 deferred=pass beta1=pass performance=not_measured\n";
}

int main(int argc,char** argv){
  try{
    if(argc!=7 || std::string(argv[1])!="--world" || std::string(argv[3])!="--m" ||
        std::string(argv[5])!="--hidden")throw std::runtime_error("expected --world 4|8 --m 128|256 --hidden 128|256");
    int world=std::stoi(argv[2]),m=std::stoi(argv[4]),h=std::stoi(argv[6]),count=0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if((world!=4&&world!=8)||count<world||(m!=128&&m!=256)||(h!=128&&h!=256))
      throw std::runtime_error("requires CP4/8, CPU-oracle bounded M/H128|256");
    run(world,m,h);run_qkv_weight(m,h);run_qkv(world,m,h);return 0;
  }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
