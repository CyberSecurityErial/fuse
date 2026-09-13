// SPDX-License-Identifier: BSD-3-Clause
// Complete immediate projection backwards: W preparation -> inverse A2A/dX
// (QKV) or dA/inverse A2A (OProj) -> two saved-master preparations -> dW.
// Upstream gradient quantization and cross-CP dW reduction are caller-owned.
#include "fuse/operators/ulysses/oproj_backward.h"
#include "fuse/operators/ulysses/qkv_backward.h"
#include "fuse/operators/primitives/gemm_a2a_mxfp8.h"
#include "../fused_graph.cuh"
#include "../fused_inputs.cuh"
#include "../fused_mpi.cuh"
#include "mxfp8_reference.cuh"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <numeric>

using fuse::Bf16;
using mxfp8_reference::check;

enum class Component { kFull, kData, kWeight, kWeightCompute, kDataCompute, kDataGemm };
const char* component_name(Component c){
  return c==Component::kDataGemm?"data_gemm":c==Component::kDataCompute?"data_compute":c==Component::kFull?"full":c==Component::kData?"data":
      c==Component::kWeight?"weight":"weight_compute";
}

struct Options {
  int world=8,m=256,h=256,heads=8,comm=16,epilogue=32,swizzle=1;
  bool along_m=false,causal=false,calibrate=false;
  int weight_epilogue=0,weight_swizzle=0;
  std::string weight_raster;
  std::string json;
  int kv_heads=8;
  bool qkv=false;
  int width() const { return (heads+(qkv?2*kv_heads:0))*128; }
};

Options parse(int argc,char** argv) {
  Options o;
  for(int i=1;i<argc;++i){
    const std::string key=argv[i];
    if(key=="--causal"){o.causal=true;continue;}
    if(key=="--calibrate"){o.calibrate=true;continue;}
    if(i+1==argc)throw std::invalid_argument("missing option value");
    const std::string value=argv[++i];
    if(key=="--world")o.world=std::stoi(value);
    else if(key=="--operator" && (value=="qkv" || value=="oproj"))o.qkv=value=="qkv";
    else if(key=="--kv-heads")o.kv_heads=std::stoi(value);
    else if(key=="--m")o.m=std::stoi(value);
    else if(key=="--hidden")o.h=std::stoi(value);
    else if(key=="--q-heads")o.heads=std::stoi(value);
    else if(key=="--comm-ctas")o.comm=std::stoi(value);
    else if(key=="--epilogue-n")o.epilogue=std::stoi(value);
    else if(key=="--swizzle")o.swizzle=std::stoi(value);
    else if(key=="--weight-epilogue-n")o.weight_epilogue=std::stoi(value);
    else if(key=="--weight-swizzle")o.weight_swizzle=std::stoi(value);
    else if(key=="--weight-raster" && (value=="along_m" || value=="along_n"))o.weight_raster=value;
    else if(key=="--raster" && (value=="along_m" || value=="along_n"))o.along_m=value=="along_m";
    else if(key=="--json-out")o.json=value;
    else throw std::invalid_argument("unknown option: "+key);
  }
  if(!o.weight_epilogue)o.weight_epilogue=o.epilogue;
  if(!o.weight_swizzle)o.weight_swizzle=o.swizzle;
  if(o.weight_raster.empty())o.weight_raster=o.along_m?"along_m":"along_n";
  if(o.world!=fused_mpi::process_world || o.m<=0 || o.m%128 || o.h<=0 || o.h%128 ||
      o.heads<=0 || o.heads>16384 || o.heads%o.world || o.comm<=0 || o.comm>=148 ||
      (o.epilogue!=32 && o.epilogue!=64) ||
      (o.swizzle!=1 && o.swizzle!=2 && o.swizzle!=4 && o.swizzle!=8) ||
      (o.weight_epilogue!=32 && o.weight_epilogue!=64) ||
      (o.weight_swizzle!=1 && o.weight_swizzle!=2 && o.weight_swizzle!=4 && o.weight_swizzle!=8) ||
      (o.qkv && (o.kv_heads<=0 || o.kv_heads>o.heads || o.heads%o.kv_heads ||
          o.kv_heads%o.world || (o.causal && o.m%256))))
    throw std::invalid_argument("unsupported backward benchmark geometry/configuration");
  return o;
}

// Independent scalar inverse HeadToSequence mapping. Numerical dA is checked
// from original operands first; actual source dA is used ONLY for byte transport.
struct InverseRoute {
  const uint16_t* source[8]{};
  int m,a,world,rank;
  bool causal;
  __device__ uint16_t operator()(uint64_t index) const {
    const uint64_t global=index/(a/world), feature=index%(a/world);
    int src=int(global/m),row=int(global%m);
    if(causal){
      const int chunk=int(global/(m/2));
      src=chunk<world?chunk:2*world-1-chunk;
      row=int(global%(m/2))+(chunk<world?0:m/2);
    }
    return source[src][uint64_t(row)*a+uint64_t(rank)*(a/world)+feature];
  }
};

struct Runtime {
  Component component=Component::kFull;
  cudaStream_t stream{};
  cudaEvent_t start{},end{};
  std::vector<void*> allocations,imports;
  Bf16 *dy{},*weight{},*attention{},*da{},*routed{},*dw{};
  fuse::Mxfp8OprojBackwardParams params{};
  fused_inputs::Scratch* input_stats{};
  mxfp8_reference::Workspace reference;
  InverseRoute route{};
  bool qkv=false;
  fuse::Mxfp8QkvBackwardParams qkv_params{};
  Bf16* gradient[3]{};
  fuse::Mxfp8Activation quantized[3]{};
  mxfp8_reference::QkvGradient original_qkv{};
  Bf16 *expected_data{}, *expected_weight{};
  int reference_generation=-1;

  void initialize_reference_cache(int m,int n,int weight_rows){
    // Calibration checks the same unchanged payload before/after SIX graphs.
    // Store its two independent oracle outputs once, not twelve recomputations
    // of each GEMM. Keep a 2 GiB reserve; if any rank lacks space, ALL ranks use
    // the original bounded reference. This is outside every timed boundary.
    const uint64_t bytes=2ull*n*(uint64_t(m)+weight_rows);
    size_t free=0,total=0;check(cudaMemGetInfo(&free,&total));
    if(fused_mpi::any(free<bytes+(2ull<<30)))return;
    expected_data=alloc<Bf16>(size_t(m)*n);
    expected_weight=alloc<Bf16>(size_t(weight_rows)*n);
  }

  uint32_t native_epoch() const {
    return qkv?qkv_params.data.projection.epoch:params.data.projection.epoch;
  }

  template<class T> T* alloc(size_t count){
    T* p{};check(cudaMalloc(&p,count*sizeof(T)));allocations.push_back(p);
    check(cudaMemsetAsync(p,0,count*sizeof(T),stream));return p;
  }
  template<class T> std::vector<T*> share(T* local,int world){
    std::vector<cudaIpcMemHandle_t> handles(world);
    check(cudaIpcGetMemHandle(&handles[fused_mpi::process_rank],local));
    fused_mpi::gather_owned(handles);
    std::vector<T*> peers(world);
    for(int p=0;p<world;++p){
      if(p==fused_mpi::process_rank)peers[p]=local;
      else{
        void* mapped{};
        check(cudaIpcOpenMemHandle(&mapped,handles[p],cudaIpcMemLazyEnablePeerAccess));
        imports.push_back(mapped);peers[p]=static_cast<T*>(mapped);
      }
    }
    return peers;
  }
  void initialize(const Options& o){
    qkv=o.qkv;
    if(qkv){initialize_qkv(o);return;}
    const int rank=fused_mpi::process_rank,a=o.heads*128;
    auto& d=params.data.projection;
    d.local_tokens=o.m;d.hidden=o.h;d.q_heads=o.heads;d.head_dim=128;
    d.world_size=o.world;d.rank=rank;d.num_comm_ctas=o.comm;
    d.causal_load_balanced=o.causal;d.gemm_policy=fuse::BackwardGemmPolicy::kM128N256;
    d.gemm_tuning={o.epilogue,o.swizzle,o.along_m};
    params.weight.projection.local_tokens=o.m;params.weight.projection.hidden=o.h;
    params.weight.projection.q_heads=o.heads;params.weight.projection.head_dim=128;
    // Independent dW search must not change dA's GEMM or inverse-A2A schedule.
    params.weight.gemm_tuning={o.weight_epilogue,o.weight_swizzle,o.weight_raster=="along_m"};
    size_t bbytes=0,wbytes=0,data=0,scales=0;
    check(fuse::oproj_backward_mxfp8_data_workspace_size(d,&bbytes));
    check(fuse::oproj_backward_mxfp8_weight_workspace_size(params.weight.projection,&wbytes));
    check(fuse::gemm_a2a_mxfp8_activation_size({o.m,a,o.h,1},&data,&scales));
    // Conservative preallocation guard, including bounded reference and CUDA
    // workspaces. No allocation or admission check is part of a timed Graph.
    const uint64_t needed=6ull*o.m*o.h+6ull*o.h*a+9ull*o.m*a+(2ull<<30);
    size_t free=0,total=0;check(cudaMemGetInfo(&free,&total));
    if(fused_mpi::any(free<needed))throw std::runtime_error("OOM admission: insufficient free memory; no benchmark launched");
    check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    check(cudaEventCreate(&start));check(cudaEventCreate(&end));
    dy=alloc<Bf16>(size_t(o.m)*o.h);weight=alloc<Bf16>(size_t(o.h)*a);
    attention=alloc<Bf16>(size_t(o.m)*a);da=alloc<Bf16>(size_t(o.m)*a);
    routed=alloc<Bf16>(size_t(o.m)*a);dw=alloc<Bf16>(size_t(o.h)*a);
    d.weight=weight;d.local_grad_attention=da;
    d.ready=alloc<uint32_t>(size_t(o.m/128)*(a/256)*fuse::kReadyFlagStride);
    auto* done=alloc<uint32_t>(o.world*fuse::kReadyFlagStride);
    params.weight.projection.grad_output=dy;params.weight.projection.saved_attention=attention;
    params.weight.projection.grad_weight=dw;params.weight.projection.alpha=1;params.weight.projection.beta=0;
    params.weight_mode=fuse::WeightGradientMode::kImmediate;
    auto* workspace=alloc<unsigned char>(std::max(bbytes,wbytes));
    params.data.workspace=params.weight.workspace=workspace;
    params.data.workspace_bytes=params.weight.workspace_bytes=std::max(bbytes,wbytes);
    params.data.grad_output={alloc<fuse::Fp8E4m3>(data),alloc<uint8_t>(scales),data,scales};
    input_stats=alloc<fused_inputs::Scratch>(1);
    reference.initialize(a,stream,[this](size_t bytes)->void*{return alloc<unsigned char>(bytes);});
    check(cudaStreamSynchronize(stream));fused_mpi::barrier();
    const auto output_peers=share(routed,o.world),source_peers=share(da,o.world);
    const auto flag_peers=share(done,o.world);
    route.m=o.m;route.a=a;route.world=o.world;route.rank=rank;route.causal=o.causal;
    for(int p=0;p<o.world;++p){
      d.peer_grad_attention[p]=output_peers[p];d.peer_done_epoch[p]=flag_peers[p];
      route.source[p]=reinterpret_cast<const uint16_t*>(source_peers[p]);
    }
    initialize_reference_cache(o.m,a,o.h);
    cudaDeviceProp device{};check(cudaGetDeviceProperties(&device,fused_mpi::local_device));
    // Common launcher startup record: the controller uses this exact rank
    // identity to distinguish a launched process from an empty log file.
    std::cout<<"device,rank="<<rank<<",sm="<<device.multiProcessorCount
        <<",compute="<<device.major<<'.'<<device.minor<<",free_bytes="<<free<<",admission_bytes="<<needed
        <<",reference_cache="<<bool(expected_data)<<'\n'<<std::flush;
    fused_mpi::barrier();
  }
  void initialize_qkv(const Options& o){
    const int rank=fused_mpi::process_rank,a=o.width();
    route.m=o.m;route.a=a;route.world=o.world;route.rank=rank;route.causal=o.causal;
    auto& d=qkv_params.data.projection;
    d.local_tokens=o.m;d.hidden=o.h;d.q_heads=o.heads;d.kv_heads=o.kv_heads;d.head_dim=128;
    d.world_size=o.world;d.rank=rank;d.num_comm_ctas=o.comm;d.epoch=1;
    d.causal_load_balanced=o.causal;d.gemm_policy=fuse::BackwardGemmPolicy::kM128N256;
    d.gemm_tuning={o.epilogue,o.swizzle,o.along_m};
    auto& w=qkv_params.weight.projection;
    w.local_tokens=o.m;w.hidden=o.h;w.q_heads=o.heads;w.kv_heads=o.kv_heads;w.head_dim=128;
    qkv_params.weight.gemm_tuning={o.weight_epilogue,o.weight_swizzle,o.weight_raster=="along_m"};
    size_t bbytes=0,wbytes=0;
    check(fuse::qkv_backward_mxfp8_data_workspace_size(d,&bbytes));
    check(fuse::qkv_backward_mxfp8_weight_workspace_size(w,&wbytes));
    const uint64_t needed=8ull*o.m*a+6ull*o.h*a+6ull*o.m*o.h+(2ull<<30);
    size_t free=0,total=0;check(cudaMemGetInfo(&free,&total));
    if(fused_mpi::any(free<needed))throw std::runtime_error("OOM admission: insufficient free memory; no benchmark launched");
    check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    check(cudaEventCreate(&start));check(cudaEventCreate(&end));
    weight=alloc<Bf16>(size_t(a)*o.h);attention=alloc<Bf16>(size_t(o.m)*o.h);
    da=alloc<Bf16>(size_t(o.m)*o.h);routed=alloc<Bf16>(size_t(o.m)*a);dw=alloc<Bf16>(size_t(a)*o.h);
    d.weight=weight;d.grad_input=da;d.peer_dqkv_staging[rank]=routed;
    d.peer_ready[rank]=alloc<uint32_t>(fuse::qkv_backward_ready_elements(d));
    w.dqkv_staging=routed;w.saved_input=attention;w.grad_weight=dw;
    auto* scratch=alloc<unsigned char>(std::max(bbytes,wbytes));
    qkv_params.data.workspace=qkv_params.weight.workspace=scratch;
    qkv_params.data.workspace_bytes=qkv_params.weight.workspace_bytes=std::max(bbytes,wbytes);
    input_stats=alloc<fused_inputs::Scratch>(1);
    reference.initialize(o.h,stream,[this](size_t bytes)->void*{return alloc<unsigned char>(bytes);});
    original_qkv.m=o.m;original_qkv.q_heads=o.heads;original_qkv.kv_heads=o.kv_heads;
    original_qkv.world=o.world;original_qkv.rank=rank;original_qkv.causal=o.causal;
    for(int kind=0;kind<3;++kind){
      const int width=(kind==0?o.heads:o.kv_heads)/o.world*128;
      size_t data=0,scales=0;
      check(fuse::gemm_a2a_mxfp8_activation_size({o.m*o.world,o.h,width,1},&data,&scales));
      gradient[kind]=alloc<Bf16>(size_t(o.m)*o.world*width);
      auto* bytes=alloc<fuse::Fp8E4m3>(data);auto* sf=alloc<uint8_t>(scales);
      quantized[kind]={bytes,sf,data,scales};
      check(cudaStreamSynchronize(stream));fused_mpi::barrier();
      const auto masters=share(gradient[kind],o.world);
      const auto fp8=share(bytes,o.world);
      const auto scales_peers=share(sf,o.world);
      for(int peer=0;peer<o.world;++peer){
        auto& input=qkv_params.data.peer_input[peer];
        auto& view=kind==0?input.grad_q:(kind==1?input.grad_k:input.grad_v);
        auto& master=kind==0?input.master_q:(kind==1?input.master_k:input.master_v);
        view={fp8[peer],scales_peers[peer],data,scales};master=masters[peer];
        original_qkv.source[peer][kind]=masters[peer];
      }
    }
    initialize_reference_cache(o.m,o.h,a);
    cudaDeviceProp device{};check(cudaGetDeviceProperties(&device,fused_mpi::local_device));
    std::cout<<"device,rank="<<rank<<",sm="<<device.multiProcessorCount
        <<",compute="<<device.major<<'.'<<device.minor<<",free_bytes="<<free<<",admission_bytes="<<needed
        <<",reference_cache="<<bool(expected_data)<<'\n'<<std::flush;
    fused_mpi::barrier();
  }
  void input(Bf16* target,size_t count,uint32_t seed,const char* name,int generation){
    check(fused_inputs::generate(reinterpret_cast<uint16_t*>(target),count,seed,.25f,input_stats,stream));
    fused_inputs::Stats stats{};
    check(cudaMemcpyAsync(&stats,&input_stats->result,sizeof(stats),cudaMemcpyDeviceToHost,stream));
    check(cudaStreamSynchronize(stream));
    if(stats.count!=count || stats.finite!=count || !stats.nonzero)throw std::runtime_error("invalid random payload");
    std::cout<<"backward_input rank="<<fused_mpi::process_rank<<" generation="<<generation
        <<" tensor="<<name<<" generator=gpu_philox seed="<<seed<<" count="<<stats.count
        <<" finite="<<stats.finite<<" nonzero="<<stats.nonzero<<" sum="<<stats.sum
        <<" square_sum="<<stats.square_sum<<" min="<<stats.minimum<<" max="<<stats.maximum<<'\n';
  }
  void poison(const Options& o){
    if(component==Component::kDataCompute || component==Component::kDataGemm){
      check(cudaMemsetAsync(da,0xff,size_t(o.m)*o.h*sizeof(Bf16),stream));
      check(cudaStreamSynchronize(stream));fused_mpi::barrier();return;
    }
    if(component==Component::kFull || component==Component::kData){
      check(cudaMemsetAsync(da,0xff,size_t(o.m)*(qkv?o.h:route.a)*sizeof(Bf16),stream));
      check(cudaMemsetAsync(routed,0xff,size_t(o.m)*route.a*sizeof(Bf16),stream));
    }
    if(component!=Component::kData)
      check(cudaMemsetAsync(dw,0xff,size_t(o.h)*route.a*sizeof(Bf16),stream));
    check(cudaStreamSynchronize(stream));fused_mpi::barrier();
  }
  void validate(const Options& o,int generation,const char* phase){
    auto b=fused_validation::Stats::zero(),w=fused_validation::Stats::zero();
    // Inputs are generated once per payload, then immutable across components.
    // Changing generation MUST rebuild both reference outputs before reuse.
    const bool reuse=expected_data && reference_generation==generation;
    if(qkv){
      auto transposed=original_qkv;transposed.transpose=true;
      b=reference.validate_views(original_qkv,mxfp8_reference::Operand{weight,1,o.h},da,o.m,o.h,route.a,stream,expected_data,reuse);
      w=reference.validate_views(transposed,mxfp8_reference::Operand{attention,1,o.h},dw,route.a,o.h,o.m,stream,expected_weight,reuse);
    }else{
      b=reference.validate({dy,o.h,1},{weight,1,route.a},da,o.m,route.a,o.h,stream,expected_data,reuse);
      w=reference.validate({dy,1,o.h},{attention,1,route.a},dw,o.h,route.a,o.m,stream,expected_weight,reuse);
    }
    reference_generation=generation;
    fused_mpi::barrier();
    if(qkv)check(fused_validation::launch<false>(reinterpret_cast<const uint16_t*>(routed),original_qkv,
        uint64_t(o.m)*route.a,reference.comparison,0,stream));
    else check(fused_validation::launch<false>(reinterpret_cast<const uint16_t*>(routed),route,
        uint64_t(o.m)*route.a,reference.comparison,0,stream));
    fused_validation::Stats transport{};
    check(cudaMemcpyAsync(&transport,reference.comparison->result,sizeof(transport),cudaMemcpyDeviceToHost,stream));
    check(cudaStreamSynchronize(stream));
    const bool bad=b.checked!=uint64_t(o.m)*(qkv?o.h:route.a) || w.checked!=uint64_t(o.h)*route.a ||
        transport.checked!=uint64_t(o.m)*route.a || b.mismatches || w.mismatches || transport.mismatches;
    std::cout<<"backward_validation rank="<<fused_mpi::process_rank<<" generation="<<generation
        <<" component="<<component_name(component)
        <<" phase="<<phase<<" B_checked="<<b.checked<<" W_checked="<<w.checked
        <<" route_checked="<<transport.checked<<" B_mismatch="<<b.mismatches<<" W_mismatch="<<w.mismatches
        <<" route_mismatch="<<transport.mismatches<<" B_max_abs="<<b.max_abs<<" W_max_abs="<<w.max_abs<<'\n'<<std::flush;
    if(fused_mpi::any(bad))throw std::runtime_error("full backward numerical/transport validation failed");
  }
  std::vector<float> step(fused_graph::Operation& graph){
    graph.prepare(graph.committed_epoch()+1,[this](uint32_t epoch,cudaStream_t s){
      if(qkv){
        if(component==Component::kDataGemm)
          return fuse::launch_qkv_backward_mxfp8_data_gemm_reference(qkv_params.data,s);
        if(component==Component::kDataCompute)
          return fuse::launch_qkv_backward_mxfp8_data_compute_reference(qkv_params.data,s);
        if(component==Component::kWeight)return fuse::launch_qkv_backward_mxfp8_weight(qkv_params.weight,s);
        if(component==Component::kWeightCompute)
          return fuse::launch_qkv_backward_mxfp8_weight_compute_reference(qkv_params.weight,s);
        qkv_params.data.projection.epoch=epoch;
        return component==Component::kData?fuse::launch_qkv_backward_mxfp8_data(qkv_params.data,s):
            fuse::launch_qkv_backward_mxfp8(qkv_params,s);
      }
      // W has no native ready epoch. Its Graph launch index must not advance
      // B's publication epoch. Full/B graphs resume from the last actual B.
      if(component==Component::kWeight)return fuse::launch_oproj_backward_mxfp8_weight(params.weight,s);
      if(component==Component::kWeightCompute)
        return fuse::launch_oproj_backward_mxfp8_weight_compute_reference(params.weight,s);
      params.data.projection.epoch=epoch;
      return component==Component::kData?fuse::launch_oproj_backward_mxfp8_data(params.data,s):
          fuse::launch_oproj_backward_mxfp8(params,s);
    });
    check(cudaStreamSynchronize(stream));fused_mpi::barrier();
    check(cudaEventRecord(start,stream));graph.launch();check(cudaEventRecord(end,stream));
    check(cudaEventSynchronize(end));
    std::vector<float> elapsed(fused_mpi::process_world);
    check(cudaEventElapsedTime(&elapsed[fused_mpi::process_rank],start,end));
    fused_mpi::gather_owned(elapsed);
    for(float t:elapsed)if(!std::isfinite(t)||t<=0)throw std::runtime_error("invalid CUDA event duration");
    return elapsed;
  }
  void release(){
    check(cudaStreamSynchronize(stream));fused_mpi::barrier();
    reference.release();for(auto* p:imports)check(cudaIpcCloseMemHandle(p));
    fused_mpi::barrier();for(auto* p:allocations)check(cudaFree(p));
    check(cudaEventDestroy(start));check(cudaEventDestroy(end));check(cudaStreamDestroy(stream));
  }
};

template<class T> double percentile(std::vector<T> values,double fraction){
  std::sort(values.begin(),values.end());const double at=fraction*(values.size()-1);
  const auto lo=size_t(at),hi=std::min(lo+1,values.size()-1);
  return values[lo]+(values[hi]-values[lo])*(at-lo);
}
struct Result{double p50,p95,drift;int round;};

// Same convergence/cadence contract as fused_bf16: 10 initial calls, >=100ms
// on EACH rank and three <=5% windows, 10 cadence calls, first stable 50-call
// round out of at most three. Rejected rounds remain in the raw log.
// Large eight-rank cases exhausted the former 5s watchdog without converging.
// Allow more settling time, not a looser threshold or a fastest-round choice.
constexpr double kWarmupTimeoutSeconds=30;
Result measure(Runtime& r,fused_graph::Operation& graph,const Options& o,int generation){
  auto collect=[&](int count,const char* phase,int round){
    std::vector<std::vector<float>> raw;
    std::vector<float> maxima;
    const auto initial=graph.committed_epoch();
    for(int i=0;i<count;++i){
      raw.push_back(r.step(graph));maxima.push_back(*std::max_element(raw.back().begin(),raw.back().end()));
    }
    for(int i=0;i<count;++i){
      auto& out=fused_mpi::root_output();
      out<<"backward_sample generation="<<generation<<" phase="<<phase<<" round="<<round
          <<" component="<<component_name(r.component)
          <<" index="<<i<<" epoch="<<initial+i+1<<" maxrank_ms="<<maxima[i];
      for(int rank=0;rank<o.world;++rank)out<<" rank"<<rank<<"_ms="<<raw[i][rank];
      out<<'\n';
    }
    fused_mpi::root_output()<<std::flush;return maxima;
  };
  collect(10,"initial",-1);
  const auto started=std::chrono::steady_clock::now();
  auto expired=[&]{return fused_mpi::any(std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()>=kWarmupTimeoutSeconds);};
  std::vector<double> accumulated(o.world);
  std::vector<std::vector<double>> windows(o.world);
  int count=10;
  while(true){
    std::vector<double> elapsed(o.world);int calls=0;
    do{
      const auto times=r.step(graph);for(int p=0;p<o.world;++p)elapsed[p]+=times[p];++calls;
    }while(calls<count && !expired());
    bool stable=true;int next_count=1000;
    for(int p=0;p<o.world;++p){
      accumulated[p]+=elapsed[p];windows[p].push_back(elapsed[p]/calls);
      bool ready=false;
      if(windows[p].size()>=3){
        const std::vector<double> last(windows[p].end()-3,windows[p].end());
        ready=accumulated[p]>=100 && (*std::max_element(last.begin(),last.end())-
            *std::min_element(last.begin(),last.end()))/percentile(last,.5)<=.05;
      }
      stable=stable&&ready;
      next_count=std::min(next_count,int(std::clamp(std::ceil(20/std::max(windows[p].back(),.001)),10.,1000.)));
      fused_mpi::root_output()<<"backward_warmup generation="<<generation<<" rank="<<p
          <<" component="<<component_name(r.component)
          <<" window="<<windows[p].size()-1<<" calls="<<calls<<" ms_per_call="<<windows[p].back()
          <<" accumulated_cuda_ms="<<accumulated[p]<<" ready="<<ready<<'\n';
    }
    fused_mpi::root_output()<<std::flush;
    if(stable)break;
    if(expired())throw std::runtime_error("backward warmup did not converge within 30s");
    count=next_count;
  }
  collect(10,"sample_cadence",-1);
  for(int round=0;round<3;++round){
    const auto samples=collect(50,"measurement",round);
    const std::vector<float> first(samples.begin(),samples.begin()+25),second(samples.begin()+25,samples.end());
    const double p50=percentile(samples,.5),drift=std::abs(percentile(first,.5)-percentile(second,.5))/p50;
    if(drift<=.05)return {p50,percentile(samples,.95),drift,round};
    fused_mpi::root_output()<<"backward_rejected generation="<<generation<<" component="<<component_name(r.component)
        <<" round="<<round<<" half_drift="<<drift<<'\n'<<std::flush;
  }
  throw std::runtime_error("backward sample drift exceeds 5% in all three rounds");
}

void run(const Options& o){
  Runtime r;r.initialize(o);
  using L=fused_graph::Launch;
  fused_graph::Operation graph(fused_mpi::local_device,r.stream,0,
      {L::kOrdinaryDynamic,L::kCooperativeDynamic,L::kOrdinaryDynamic,L::kOrdinaryDynamic,L::kOrdinaryDynamic});
  const double flops=4.*o.m*o.h*o.width();
  fused_mpi::root_output()<<"backward_config op="<<(o.qkv?"qkv_mxfp8":"oproj_mxfp8")
      <<" M="<<o.m<<" H="<<o.h<<" A="<<o.width()
      <<" q_heads="<<o.heads<<" kv_heads="<<o.kv_heads
      <<" world="<<o.world<<" comm="<<o.comm<<" epilogue="<<o.epilogue<<" swizzle="<<o.swizzle
      <<" weight_epilogue="<<o.weight_epilogue<<" weight_swizzle="<<o.weight_swizzle
      <<" weight_along_m="<<(o.weight_raster=="along_m")
      <<" along_m="<<o.along_m<<" causal="<<o.causal<<" launch=graph kernels=5 weight_mode=immediate"
      <<" warmup_timeout_s="<<kWarmupTimeoutSeconds
      <<" calibrate="<<o.calibrate
      <<" weight_compute_reference="<<o.calibrate
      <<" data_compute_reference="<<(o.qkv && o.calibrate)
      <<" data_gemm_reference="<<(o.qkv && o.calibrate)
      <<(o.qkv?" timed=weight_quant_inverse_QKV_dX_dQKV_quant_X_quant_dW upstream_dQ_dK_dV_quant=excluded original_BF16_route=included input_lease=all_ranks_until_B_complete":
          " timed=weight_quant_dA_route_dY_quant_A_quant_dW upstream_dY_quant=excluded")
      <<" CP_dW_reduce=caller_owned"
      <<" flops_per_rank="<<flops<<'\n'<<std::flush;
  std::vector<Result> results;
  for(int generation=0;generation<2;++generation){
    r.component=Component::kFull;
    const int rank=fused_mpi::process_rank;
    r.input(r.weight,size_t(o.h)*r.route.a,5678+generation*100,"W",generation);
    if(o.qkv){
      for(int kind=0;kind<3;++kind){
        const int width=(kind==0?o.heads:o.kv_heads)/o.world*128;
        r.input(r.gradient[kind],size_t(o.m)*o.world*width,1234+kind*1000+generation*100+rank,
            kind==0?"dQ":(kind==1?"dK":"dV"),generation);
        check(fuse::quantize_gemm_a2a_mxfp8_activation({o.m*o.world,o.h,width,1},
            r.gradient[kind],r.quantized[kind],r.stream));
      }
      r.input(r.attention,size_t(o.m)*o.h,9012+generation*100+rank,"saved_X",generation);
    }else{
      r.input(r.dy,size_t(o.m)*o.h,1234+generation*100+rank,"dY",generation);
      r.input(r.attention,size_t(o.m)*r.route.a,9012+generation*100+rank,"saved_A",generation);
      check(fuse::quantize_gemm_a2a_mxfp8_activation({o.m,r.route.a,o.h,1},r.dy,r.params.data.grad_output,r.stream));
    }
    check(cudaStreamSynchronize(r.stream));fused_mpi::barrier();
    r.poison(o);r.step(graph);r.validate(o,generation,"pre");
    const auto result=measure(r,graph,o,generation);
    r.validate(o,generation,"post");results.push_back(result);
    fused_mpi::root_output()<<"backward_verified generation="<<generation<<" component=full p50_ms="<<result.p50
        <<" p95_ms="<<result.p95<<" half_drift="<<result.drift<<" selected_round="<<result.round
        <<" pflops="<<flops/(result.p50*1e12)<<" verification=pass\n"<<std::flush;
    auto components=std::vector<Component>{Component::kData,Component::kWeight,Component::kWeightCompute};
    if(o.qkv)components.insert(components.begin()+1,Component::kDataCompute);
    if(o.qkv)components.insert(components.begin()+2,Component::kDataGemm);
    if(o.calibrate)for(Component component:components){
      r.component=component;
      const std::vector<L> boundary=(component==Component::kDataCompute || component==Component::kDataGemm)?
          std::vector<L>{L::kCooperativeDynamic}:component==Component::kData?
          std::vector<L>{L::kOrdinaryDynamic,L::kCooperativeDynamic}:
          component==Component::kWeightCompute?std::vector<L>{L::kOrdinaryDynamic}:
          std::vector<L>{L::kOrdinaryDynamic,L::kOrdinaryDynamic,L::kOrdinaryDynamic};
      fused_graph::Operation isolated(fused_mpi::local_device,r.stream,
          component==Component::kData?r.native_epoch():0,boundary);
      // dX compute follows completed B, before W can overwrite its scratch;
      // it preserves the original ready flags, budget and acquire adapter.
      // The following bare dX uses the same inputs/budget/ranks but removes
      // ONLY that adapter. Neither diagnostic may run after W reuses scratch.
      // W compute immediately follows completed full W: its exact prepared
      // operands/scales remain in scratch. Only dW is poisoned; no preparation
      // enters the single-kernel Graph. The oracle still reads original BF16.
      // Only the active component's outputs are poisoned. Checking both full
      // gradients also checks that this isolated component leaves the other
      // gradient valid; it does NOT count the other gradient in timed FLOPs.
      r.poison(o);r.step(isolated);r.validate(o,generation,"pre");
      const auto value=measure(r,isolated,o,generation);
      r.validate(o,generation,"post");
      fused_mpi::root_output()<<"backward_verified generation="<<generation
          <<" component="<<component_name(component)<<" p50_ms="<<value.p50<<" p95_ms="<<value.p95
          <<" half_drift="<<value.drift<<" selected_round="<<value.round
          <<" pflops="<<(flops/2)/(value.p50*1e12)<<" verification=pass\n"<<std::flush;
      isolated.reset(isolated.committed_epoch());
    }
    graph.reset(r.native_epoch());
  }
  if(fused_mpi::root() && !o.json.empty()){
    std::ofstream out(o.json);out<<std::setprecision(12);
    out<<"{\"schema\":\"sm103_mxfp8_"<<(o.qkv?"qkv":"oproj")
        <<"_backward_v1\",\"verified\":true,\"launch\":\"graph\","
        <<"\"boundary\":\"immediate_B_W_five_kernels\",\"M\":"<<o.m<<",\"H\":"<<o.h
        <<",\"A\":"<<r.route.a<<",\"world\":"<<o.world<<",\"comm\":"<<o.comm
        <<",\"epilogue\":"<<o.epilogue<<",\"swizzle\":"<<o.swizzle<<",\"along_m\":"<<int(o.along_m)
        <<",\"weight_epilogue\":"<<o.weight_epilogue<<",\"weight_swizzle\":"<<o.weight_swizzle
        <<",\"weight_along_m\":"<<int(o.weight_raster=="along_m")
        <<",\"causal\":"<<int(o.causal)<<",\"flops_per_rank\":"<<flops<<",\"payloads\":[";
    for(size_t i=0;i<results.size();++i){const auto& x=results[i];if(i)out<<',';
      out<<"{\"generation\":"<<i<<",\"warmup\":10,\"samples\":50,\"p50_ms\":"<<x.p50
          <<",\"p95_ms\":"<<x.p95<<",\"half_drift\":"<<x.drift<<",\"selected_round\":"<<x.round<<'}';}
    out<<"]}\n";out.close();if(!out)throw std::runtime_error("cannot write backward result");
  }
  r.release();fused_mpi::root_output()<<"backward_complete verification=pass payloads=2 boundary=immediate_B_W\n"<<std::flush;
}

int main(int argc,char** argv){
  try{
    fused_mpi::initialize(argc,argv);std::cout<<std::setprecision(12);
    std::string contract;for(int i=1;i<argc;++i)contract+=std::string(argv[i])+"\n";
    fused_mpi::agree(contract);run(parse(argc,argv));fused_mpi::finalize();return 0;
  }catch(const std::exception& e){
    std::cerr<<"FAIL rank="<<fused_mpi::process_rank<<" "<<e.what()<<'\n'<<std::flush;
    fused_mpi::abort(1);return 1;
  }
}
