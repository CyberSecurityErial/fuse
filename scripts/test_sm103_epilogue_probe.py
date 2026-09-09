#!/usr/bin/env python3
"""CPU contracts for the actual private adapter; no CUDA ordering/performance emulation."""

import hashlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class EpilogueProbeContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = shlex.split(os.environ.get('CXX', 'c++'))
        if not cls.compiler or shutil.which(cls.compiler[0]) is None:
            raise unittest.SkipTest('host C++ compiler required')
        cls.temporary = tempfile.TemporaryDirectory(prefix='fuse-epilogue-probe-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        headers = {
            'cuda_runtime.h': '#pragma once\n#include <cstddef>\nusing cudaError_t=int; using cudaStream_t=void*;\n'
                'struct cudaFuncAttributes { int numRegs=0, maxThreadsPerBlock=0; size_t sharedSizeBytes=0, localSizeBytes=0; };\n',
            'cutlass/cutlass.h': '#pragma once\nnamespace cutlass { enum class Status { kSuccess }; struct CudaHostAdapter {}; }\n',
            'cutlass/bfloat16.h': '#pragma once\nnamespace cutlass { using bfloat16_t=unsigned short; }\n',
            'cutlass/float8.h': '#pragma once\nnamespace cutlass { using float_e4m3_t=unsigned char; }\n',
        }
        for name, data in headers.items():
            path = cls.directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(data)
        pipeline = (ROOT / 'csrc/operators/sm103/detail/cutlass_pipeline.cuh').read_text()
        begin = pipeline.index('template <class Base, class TileShape>\nstruct SignalingEpilogue')
        body = pipeline[begin:pipeline.index('\n}  // namespace fuse::detail', begin)]
        ordering = (ROOT / 'csrc/operators/sm103/detail/producer_consumer.cuh').read_text()
        begin = ordering.index('template <int M, int N>\nstruct PublishedTile')
        published_tile = ordering[begin:ordering.index('\n};', begin) + len('\n};')]
        source = r'''
#include "csrc/operators/sm103/detail/epilogue_profiling.cuh"
#include <array>
#include <cstring>
#include <stdexcept>
#include <tuple>
#include <vector>
#define CUTLASS_DEVICE
#define CUTLASS_HOST_DEVICE
#define CHECK(expr) do { if (!(expr)) throw std::runtime_error(#expr); } while (0)
struct { int x=0; } threadIdx, blockIdx;
enum Call { Stamp, Store, Wait, Join, Publish, Tail };
std::vector<Call> calls;
uint64_t clock_value=0;
int store_delay=10, wait_delay=5;
void __syncwarp() { calls.push_back(Join); clock_value+=7; }
namespace cute {
using std::get;
template <int I, class T> constexpr int size(T) { return T::dims[I]; }
template <int N> auto append(const std::array<int,4>& shape, int) { static_assert(N==4); return shape; }
template <class Engine, class Layout> struct Tensor {};
}
namespace fuse::detail {
uint64_t epilogue_timestamp() { calls.push_back(Stamp); return ++clock_value; }
void tma_store_wait_all() { calls.push_back(Wait); clock_value+=wait_delay; }
void store_release_gpu(uint32_t* ptr, uint32_t value) { calls.push_back(Publish); *ptr=value; clock_value+=3; }
''' + published_tile + '\n' + body + r'''
}
struct Tile { static constexpr int dims[3]={128,256,64}; };
struct Base {
  static constexpr int ThreadCount=128;
  struct Arguments { int cookie=0; };
  struct Params { int cookie=0; };
  struct TensorStorage {};
  using LoadPipeline=int; using LoadPipelineState=int;
  using StorePipeline=int; using StorePipelineState=int;
  Base(const Params&,TensorStorage&) {}
  static Params to_underlying_arguments(const std::array<int,4>&,const Arguments& args,void*) { return {args.cookie}; }
  static bool can_implement(const std::array<int,4>&,const Arguments& args) { return args.cookie==7; }
  static size_t get_workspace_size(const std::array<int,4>&,const Arguments&) { return 0; }
  static cutlass::Status initialize_workspace(const std::array<int,4>&,const Arguments&,void*,cudaStream_t,
                                              cutlass::CudaHostAdapter*) { return cutlass::Status::kSuccess; }
  template <bool Reuse, class... Args> auto store(Args&&...) {
    calls.push_back(Store); clock_value+=store_delay;
    return std::make_tuple(3,5,7);
  }
  template <class... Args> void store_tail(Args&&...) { calls.push_back(Tail); }
};
using Probe=fuse::detail::QkvEpilogueProbe<Base,Tile>;
using Record=fuse::detail::QkvEpilogueRecord;
std::array<int,4> shape{256,512,64,1};
std::array<uint32_t,4*fuse::kReadyFlagStride> ready{};
std::array<Record,6> records{};
Base::TensorStorage storage;
Probe::Arguments arguments() {
  Probe::Arguments args{}; args.cookie=7; args.ready=ready.data(); args.m_tiles=2; args.n_tiles=2;
  args.epoch=7; args.records=records.data(); args.record_capacity=6;
  return args;
}
void store_tile(Probe& probe, int m, int n, int l=0) {
  const auto states=probe.store(0,0,0,0,0,0,shape,Tile{},std::array<int,4>{m,n,0,l},
                               0,0,cute::Tensor<int,int>{},storage);
  CHECK(states==std::make_tuple(3,5,7));
}
int main(int argc,char** argv) {
  CHECK(argc==2); const std::string mode=argv[1];
  auto args=arguments();
  auto params=Probe::to_underlying_arguments(shape,args,nullptr);
  CHECK(params.cookie==7 && params.records==records.data() && params.record_capacity==6);
  CHECK(params.ready==ready.data() && params.epoch==7 && params.m_tiles==2 && params.n_tiles==2);
  CHECK(Probe::can_implement(shape,args));
  CHECK(Probe::get_workspace_size(shape,args)==0);
  CHECK(Probe::initialize_workspace(shape,args,nullptr,nullptr)==cutlass::Status::kSuccess);
  threadIdx.x=128; blockIdx.x=3;
  if (mode=="params") {
    args.records=nullptr; CHECK(!Probe::can_implement(shape,args)); args=arguments();
    args.record_capacity=0; CHECK(!Probe::can_implement(shape,args)); args=arguments();
    args.cookie=0; CHECK(!Probe::can_implement(shape,args)); args=arguments();
    shape[3]=2; CHECK(!Probe::can_implement(shape,args));
    CHECK(sizeof(Record)==96 && alignof(Record)==16);
    CHECK(std::strcmp(fuse::detail::QkvEpilogueResources::kClock,"globaltimer")==0);
    CHECK(std::strcmp(fuse::detail::QkvEpilogueResources::kClockUnit,"ns")==0);
  } else if (mode=="aggregate") {
    Probe probe(params,storage);
    store_tile(probe,0,1); CHECK(records[3].tile_count==0);
    CHECK(calls==std::vector<Call>({Stamp,Store,Stamp,Wait,Join,Stamp,Publish,Stamp}));
    store_tile(probe,1,0); CHECK(records[3].tile_count==0);
    probe.store_tail(0,0,0,0,Tile{}); CHECK(calls.back()==Tail);
    const auto& r=records[3];
    CHECK(r.tile_count==2 && r.store_ns_sum==22 && r.store_ns_max==11);
    CHECK(r.drain_ns_sum==26 && r.drain_ns_max==13);
    CHECK(r.first_store_begin==1 && r.first_store_end==12 && r.first_drain_end==25);
    CHECK(r.first_ready_after==29 && r.last_ready_after==58);
    CHECK(r.first_m_tile==0 && r.first_n_tile==1 && r.first_batch==0 && r.epoch==7);
    CHECK(ready[fuse::kReadyFlagStride]==7 && ready[2*fuse::kReadyFlagStride]==7);
    for (int i=0;i<6;++i) if(i!=3) CHECK(records[i].tile_count==0);
    // A new launch/collective does not reuse the previous launch's accumulator.
    params.epoch=8; Probe next(params,storage); store_tile(next,1,1);
    next.store_tail(0,0,0,0,Tile{});
    CHECK(records[3].tile_count==1 && records[3].epoch==8 && records[3].first_m_tile==1);
  } else if (mode=="lanes") {
    for (int lane=0;lane<128;++lane) {
      calls.clear(); records={}; ready={}; threadIdx.x=128+lane;
      Probe probe(params,storage); store_tile(probe,0,0); probe.store_tail(0,0,0,0,Tile{});
      CHECK(std::count(calls.begin(),calls.end(),Store)==1);
      CHECK(std::count(calls.begin(),calls.end(),Wait)==int(lane<32));
      CHECK(std::count(calls.begin(),calls.end(),Join)==int(lane<32));
      CHECK(std::count(calls.begin(),calls.end(),Publish)==int(lane==0));
      CHECK(std::count(calls.begin(),calls.end(),Stamp)==(lane==0?4:0));
      CHECK(records[3].tile_count==uint64_t(lane==0));
    }
  } else if (mode=="bounds") {
    for (auto coord : {std::array<int,3>{-1,0,0},{2,0,0},{0,-1,0},{0,2,0},{0,0,-1}}) {
      calls.clear(); Probe probe(params,storage); store_tile(probe,coord[0],coord[1],coord[2]);
      probe.store_tail(0,0,0,0,Tile{});
      CHECK(std::count(calls.begin(),calls.end(),Wait)==1);
      CHECK(std::count(calls.begin(),calls.end(),Publish)==0 && records[3].tile_count==0);
    }
    params.record_capacity=3;
    Probe probe(params,storage); store_tile(probe,0,0); probe.store_tail(0,0,0,0,Tile{});
    for (const auto& r:records) CHECK(r.tile_count==0);
    CHECK(ready[0]==7); // Logging bounds must not suppress the original ready publication.
  } else return 2;
}
'''
        source = source.replace('#include <array>', '#include <array>\n#include <algorithm>\n#include <string>')
        cls.probe = cls.compile(source, 'adapter')
        launch = (ROOT / 'csrc/operators/sm103/detail/launch.cuh').read_text()
        start = launch.index('template <class Gemm, class Kernel, class Comm, bool Instrumented = false,')
        helper = launch[start:launch.index('\ntemplate <bool Instrumented = false>', start)]
        forward = (ROOT / 'csrc/operators/sm103/api/forward.cuh').read_text()
        entries = forward[forward.index('namespace detail {'):forward.index('}  // namespace detail')]
        cls.entry_probe = cls.compile(r'''
#include "csrc/operators/sm103/detail/epilogue_profiling.cuh"
#include <cmath>
#include <stdexcept>
#include <string>
#define CHECK(expr) do { if (!(expr)) throw std::runtime_error(#expr); } while (0)
#define FUSE_SM103_HOST_BEGIN()
#define FUSE_SM103_HOST_MARK(stage)
#define FUSE_SM103_HOST_RETURN(value) return (value)
enum { cudaSuccess=0, cudaErrorInvalidValue=1, cudaErrorNotSupported=2 };
namespace cute {
template <int I,class T> constexpr int size(T) { return T::dims[I]; }
template <class T> constexpr int size(T) { return T::dims[0]; }
}
namespace cutlass { template <class T> void device_kernel() {} }
namespace fuse {
struct QkvRouteTimeline; // This epilogue-only entry must leave route logging disabled.
enum class GemmRaster { kHeuristic, kAlongM, kAlongN };
struct GemmProblem { int m=256,n=512,k=64,l=1; GemmRaster raster=GemmRaster::kHeuristic; };
struct GemmA2AParams {
  GemmProblem gemm; float alpha=1; uint32_t epoch=7; int num_comm_ctas=12;
  struct { bool qkv_peer_interleaved=false; } route;
  const Bf16 *lhs=nullptr,*rhs_nt=nullptr; Bf16* local_output=nullptr; uint32_t* ready=nullptr;
};
struct DeviceInfo { int sm_count=148; };
enum class QkvGemmPolicy { kM128N256K64E32, kOther };
QkvGemmPolicy selected=QkvGemmPolicy::kM128N256K64E32;
int policy_status=0,device_status=0,comm_status=0,launch_status=0,smem_status=0,attr_failure=0;
int comm_calls=0,launch_calls=0,attr_calls=0;
int select_qkv_gemm_policy(QkvGemmPolicy* p) { *p=selected; return policy_status; }
bool supported_problem(const GemmProblem& p) { return p.l==1 && p.m>0 && p.n>0 && p.k>0; }
int device_info(DeviceInfo* p) { *p={148}; return device_status; }
int ceil_div(int a,int b) { return (a+b-1)/b; }
struct Tile { static constexpr int dims[3]={128,256,64}; };
struct Cluster { static constexpr int dims[1]={1}; };
struct Comm {
  static constexpr int kBlockM=128,kBlockN=256,kQkvBulkSlots=4;
  struct Arguments { GemmA2AParams params; bool use_tma=true,use_tma_store=true; };
  static int initialize(Arguments&) { ++comm_calls; return comm_status; }
  static int64_t route_slots(const GemmA2AParams&) { throw std::runtime_error("unexpected route probe"); }
};
struct QkvEpilogueProbeGemm {
  using TileShape=Tile;
  struct Arguments {
    struct { uint32_t* ready=nullptr; int m_tiles=0,n_tiles=0; uint32_t epoch=0;
      detail::QkvEpilogueRecord* records=nullptr; int record_capacity=0; } epilogue;
  };
};
struct QkvEpilogueProbeKernel {
  using ClusterShape=Cluster;
  struct Arguments {
    QkvEpilogueProbeGemm::Arguments gemm; Comm::Arguments comm; int num_comm_ctas=0;
    A2AGemmCtaTimeline* timeline=nullptr; int timeline_capacity=0;
    QkvRouteTimeline* route_timeline=nullptr;
  };
};
struct ProductionKernel {};
struct RoleKernel {};
struct QkvForwardN256K64E32Binding {
  using Comm=fuse::Comm; using Kernel=ProductionKernel; using TelemetryKernel=RoleKernel; using TileShape=Tile;
};
QkvEpilogueProbeKernel::Arguments captured;
template <class Gemm> typename Gemm::Arguments gemm_arguments(
    const GemmProblem&,const Bf16*,const Bf16*,Bf16*,float,int,const DeviceInfo&,GemmRaster) { return {}; }
template <class Kernel> int launch_monolithic(const typename Kernel::Arguments& args,const DeviceInfo&,cudaStream_t) {
  captured=args; ++launch_calls; return launch_status;
}
namespace detail {
template <class Kernel> int launch_shared_memory(const DeviceInfo&,size_t* bytes) { *bytes=227328; return smem_status; }
}
int cudaFuncGetAttributes(cudaFuncAttributes* result,void (*kernel)()) {
  ++attr_calls;
  if(attr_failure==attr_calls) return 17;
  result->numRegs=kernel==cutlass::device_kernel<ProductionKernel>?80:
                  kernel==cutlass::device_kernel<RoleKernel>?82:96;
  result->localSizeBytes=kernel==cutlass::device_kernel<QkvEpilogueProbeKernel>?16:0;
  result->sharedSizeBytes=1024; result->maxThreadsPerBlock=256;
  return 0;
}
''' + helper + '\n' + entries + r'''
} // detail
} // fuse
int main(int argc,char** argv) {
  CHECK(argc==2); const std::string mode=argv[1];
  using namespace fuse; using namespace fuse::detail;
  GemmA2AParams p; A2AGemmCtaTimeline timeline[148]{}; QkvEpilogueRecord records[148]{};
  auto run=[&](A2AGemmCtaTimeline* t,int tc,QkvEpilogueRecord* r,int rc) {
    return launch_qkv_epilogue_telemetry(p,t,tc,r,rc,nullptr);
  };
  if(mode=="launch") {
    CHECK(run(timeline,148,records,148)==0 && launch_calls==1 && comm_calls==1);
    CHECK(captured.gemm.epilogue.records==records && captured.gemm.epilogue.record_capacity==148);
    CHECK(captured.gemm.epilogue.epoch==7 && captured.gemm.epilogue.m_tiles==2 && captured.gemm.epilogue.n_tiles==2);
    CHECK(captured.timeline==timeline && captured.timeline_capacity==148 && captured.num_comm_ctas==12);
    CHECK(captured.route_timeline==nullptr);
    CHECK(run(nullptr,148,records,148)==cudaErrorInvalidValue);
    CHECK(run(timeline,147,records,148)==cudaErrorInvalidValue);
    CHECK(run(timeline,148,nullptr,148)==cudaErrorInvalidValue);
    CHECK(run(timeline,148,records,147)==cudaErrorInvalidValue);
    CHECK(run(timeline,148,reinterpret_cast<QkvEpilogueRecord*>(reinterpret_cast<char*>(records)+8),148)==cudaErrorInvalidValue);
    CHECK(launch_calls==1 && comm_calls==1);
    selected=QkvGemmPolicy::kOther; CHECK(run(timeline,148,records,148)==cudaErrorNotSupported);
    selected=QkvGemmPolicy::kM128N256K64E32; p.route.qkv_peer_interleaved=true;
    CHECK(run(timeline,148,records,148)==cudaErrorNotSupported); p.route.qkv_peer_interleaved=false;
    p.gemm.l=2; CHECK(run(timeline,148,records,148)==cudaErrorInvalidValue); p.gemm.l=1;
    p.epoch=0; CHECK(run(timeline,148,records,148)==cudaErrorInvalidValue); p.epoch=7;
    p.num_comm_ctas=148; CHECK(run(timeline,148,records,148)==cudaErrorInvalidValue); p.num_comm_ctas=12;
    policy_status=11; CHECK(run(timeline,148,records,148)==11); policy_status=0;
    device_status=12; CHECK(run(timeline,148,records,148)==12); device_status=0;
    comm_status=13; CHECK(run(timeline,148,records,148)==13); comm_status=0;
    launch_status=14; CHECK(run(timeline,148,records,148)==14);
  } else if(mode=="resources") {
    QkvEpilogueResources r;
    CHECK(query_qkv_epilogue_resources(p,&r)==0 && attr_calls==3);
    CHECK(r.production.numRegs==80 && r.role_telemetry.numRegs==82 && r.epilogue_telemetry.numRegs==96);
    CHECK(r.epilogue_telemetry.localSizeBytes==16 && r.production.localSizeBytes==0);
    CHECK(r.dynamic_smem_bytes==227328 && r.tile_m==128 && r.tile_n==256 && r.tile_k==64 && r.cluster_ctas==1);
    CHECK(query_qkv_epilogue_resources(p,nullptr)==cudaErrorInvalidValue);
    selected=QkvGemmPolicy::kOther; CHECK(query_qkv_epilogue_resources(p,&r)==cudaErrorNotSupported);
    selected=QkvGemmPolicy::kM128N256K64E32;
    for(int failure=1;failure<=3;++failure) {
      attr_calls=0; attr_failure=failure; r.tile_m=-7;
      CHECK(query_qkv_epilogue_resources(p,&r)==17 && r.tile_m==-7);
    }
    attr_failure=0; smem_status=18; CHECK(query_qkv_epilogue_resources(p,&r)==18 && r.tile_m==-7);
  } else return 2;
}
''', 'entry')

    @classmethod
    def compile(cls, source, name, profiling=1):
        target = cls.directory / name
        result = subprocess.run([*cls.compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
            '-DFUSE_ENABLE_PROFILING=' + str(profiling), '-I', str(cls.directory), '-I', str(ROOT),
            '-I', str(ROOT / 'include'), '-x', 'c++', '-', '-o', str(target)],
            input=source, text=True, capture_output=True, timeout=60)
        if result.returncode:
            raise AssertionError(result.stderr)
        return target

    def test_real_adapter_preserves_parameter_conversion_and_narrow_capacity_contract(self):
        subprocess.run([str(self.probe), 'params'], check=True, timeout=10)

    def test_real_adapter_accumulates_first_ready_and_flushes_only_at_tail(self):
        subprocess.run([str(self.probe), 'aggregate'], check=True, timeout=10)

    def test_all_issuing_lanes_keep_wait_and_join_only_lane_zero_records(self):
        subprocess.run([str(self.probe), 'lanes'], check=True, timeout=10)

    def test_padded_tiles_and_capacity_bounds_do_not_overwrite_records(self):
        subprocess.run([str(self.probe), 'bounds'], check=True, timeout=10)

    def test_original_production_epilogue_is_byte_identical(self):
        source = (ROOT / 'csrc/operators/sm103/detail/cutlass_pipeline.cuh').read_text()
        begin = source.index('template <class Base, class TileShape>\nstruct SignalingEpilogue')
        end = source.index('\n#if FUSE_ENABLE_PROFILING', begin)
        # Production baseline after factoring the shared PublishedTile index;
        # adapter tests above also exercise its actual ready-address mapping.
        self.assertEqual(hashlib.sha256(source[begin:end].encode()).hexdigest(),
                         '545fbd215fb0f5da78f5b23e196644c21a92e77e11703e1addb457f8c7506b01')

    def test_real_private_entry_and_launch_helper_forward_buffers_and_reject_invalid_scope(self):
        subprocess.run([str(self.entry_probe), 'launch'], check=True, timeout=10)

    def test_real_resource_query_keeps_three_variants_and_propagates_failures(self):
        subprocess.run([str(self.entry_probe), 'resources'], check=True, timeout=10)

    def test_real_telemetry_template_keeps_default_join_and_gates_private_timestamp(self):
        source = (ROOT / 'csrc/operators/sm103/detail/gemm_a2a.cuh').read_text()
        start = source.index('template <class GemmKernel, class CommOp, bool OrderedRoleTimestamp = false>')
        wrapper = source[start:source.index('\n#endif', start)]
        launch = (ROOT / 'csrc/operators/sm103/detail/launch.cuh').read_text()
        start = launch.index('using QkvEpilogueProbeKernel = GemmA2ARoleTelemetryKernel<')
        alias = launch[start:launch.index(';', start) + 1]
        self.assertIn('using TelemetryKernel = GemmA2ARoleTelemetryKernel<Gemm, Comm>;', launch)
        # Compile the real wrapper and private alias, not a two-argument alias
        # stand-in. This checks dispatch/control flow only: CUDA reduction and
        # timer ordering must still be checked in actual SASS and on the GPU.
        probe = self.compile(r'''
#include "csrc/operators/sm103/detail/epilogue_profiling.cuh"
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#define CUTLASS_DEVICE
#define CHECK(expr) do { if (!(expr)) throw std::runtime_error(#expr); } while (0)
struct dim3 { unsigned x=1,y=1,z=1; };
struct { unsigned x=0; } threadIdx,blockIdx;
int arrived_count=256;
uint64_t timer_value=0;
std::vector<std::string> events;
void __syncthreads() { events.push_back("join"); }
int __syncthreads_count(int predicate) { CHECK(predicate==1);events.push_back("count");return arrived_count; }
namespace cooperative_groups {
struct Grid { void sync() { events.push_back("grid"); } };
Grid this_grid() { return {}; }
}
namespace fuse {
struct QkvRouteTimeline;
struct Gemm {
  struct Params { int scheduler=0; };
  void operator()(Params const&,char*) { events.push_back("gemm"); }
};
struct Comm {
  static constexpr bool kNeedsGridFinalize=true;
  void operator()(int,char*,int,int) { events.push_back("comm"); }
  template <bool Instrumented> void run(int,char*,int,int,bool,QkvRouteTimeline*) {
    throw std::runtime_error("unexpected route probe");
  }
  void finalize_profile(int,A2AGemmCtaTimeline*) { events.push_back("finalize"); }
};
namespace detail {
uint64_t read_global_timer() { events.push_back("stamp");return ++timer_value; }
struct PersistentTileSchedulerSm100Monolithic {
  static bool valid_initial_worker(int,unsigned) { return true; }
};
template<class G,class C> struct MonolithicGemm {
  using ArchTag=int;using ClusterShape=int;using SharedStorage=int;
  static constexpr int MaxThreadsPerBlock=256,MinBlocksPerMultiprocessor=1,SharedStorageSize=4;
  struct Arguments {};
  struct Params { int num_comm_ctas=1,comm=0;typename G::Params gemm; };
  static bool can_implement(Arguments const&) { return true; }
  static size_t get_workspace_size(Arguments const&) { return 0; }
  static cutlass::Status initialize_workspace(Arguments const&,void*,cudaStream_t) { return cutlass::Status::kSuccess; }
  static Params to_underlying_arguments(Arguments const&,void*) { return {}; }
  static dim3 get_grid_shape(Params const&) { return {}; }
  static dim3 get_block_shape() { return {256,1,1}; }
};
}
''' + wrapper + r'''
using QkvEpilogueProbeGemm=Gemm;
struct QkvForwardN256K64E32Binding { using Comm=fuse::Comm; };
''' + alias + r'''
using Regular=GemmA2ARoleTelemetryKernel<Gemm,Comm>;
static_assert(std::is_same_v<Regular,GemmA2ARoleTelemetryKernel<Gemm,Comm,false>>);
static_assert(std::is_same_v<QkvEpilogueProbeKernel,GemmA2ARoleTelemetryKernel<Gemm,Comm,true>>);
template<class Kernel> void run(bool ordered,int count,unsigned thread) {
  A2AGemmCtaTimeline timeline[2]{};
  typename Kernel::Params p;p.timeline=timeline;p.timeline_capacity=2;
  blockIdx.x=1;threadIdx.x=thread;arrived_count=count;events.clear();timer_value=0;
  Kernel{}(p,nullptr);
  bool should_stamp_role=thread==0 && (!ordered || count==256);
  CHECK((timeline[1].role_done!=0)==should_stamp_role);
  std::vector<std::string> expected;
  if(thread==0) expected.push_back("stamp");
  expected.push_back("gemm");expected.push_back(ordered?"count":"join");
  if(should_stamp_role) expected.push_back("stamp");
  expected.push_back("join");expected.push_back("grid");
  if(thread==0) expected.push_back("stamp");
  expected.push_back("finalize");expected.push_back("join");
  if(thread==0) expected.push_back("stamp");
  CHECK(events==expected);
}
}
int main() {
  for(int count:{0,128,255,256}) for(unsigned thread:{0u,128u}) {
    fuse::run<fuse::Regular>(false,count,thread);
    fuse::run<fuse::QkvEpilogueProbeKernel>(true,count,thread);
  }
}
''', 'ordered-role')
        subprocess.run([str(probe)], check=True, timeout=10)

    def test_macro_off_private_header_has_no_record_or_entry(self):
        result = subprocess.run([*self.compiler, '-std=c++17', '-DFUSE_ENABLE_PROFILING=0',
            '-I', str(ROOT), '-E', '-P', '-x', 'c++', '-'],
            input='#include "csrc/operators/sm103/detail/epilogue_profiling.cuh"\n',
            text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('QkvEpilogue', result.stdout)
        self.assertNotIn('launch_qkv_epilogue', result.stdout)


if __name__ == '__main__':
    unittest.main()
