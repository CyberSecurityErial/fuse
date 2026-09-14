"""Backward preparation/contract checks; these do not prove CUDA correctness."""
from collections import Counter
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import summarize_sm103_mxfp8_backward as backward_summary

ROOT = Path(__file__).resolve().parents[1]


class Mxfp8BackwardContracts(unittest.TestCase):
    def test_ready_iterator_parallel_heads_join_before_tma(self):
        text=(ROOT/'csrc/operators/sm103/detail/cutlass_pipeline.cuh').read_text()
        start=text.index('  template <class Iterator>\n  struct ReadyKIterator')
        end=text.index('\n  template <class LoadParams, class TileCoord, class KTileIterator>',start)
        iterator=text[start:end]
        compiler=shutil.which('c++')
        if not compiler:self.skipTest('Host C++ compiler unavailable')
        program=r'''
#include <cassert>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>
#define CUTLASS_DEVICE
constexpr int kReadyFlagStride=32;
constexpr int heads=53;
uint32_t flags[32*heads];
std::atomic<int> acquires[heads];
struct ThreadIdx {int x;};
thread_local ThreadIdx threadIdx;
thread_local int fences;
struct WarpBarrier {
  std::mutex mutex;std::condition_variable cv;int count=0,phase=0;
  void wait(){
    std::unique_lock<std::mutex> lock(mutex);int p=phase;
    if(++count==32){count=0;++phase;cv.notify_all();}
    else cv.wait(lock,[&]{return phase!=p;});
  }
} warp;
void __syncwarp(){warp.wait();}
void wait_acquire_system_single_lane(const uint32_t* p,uint32_t target) {
  assert(p>=flags && p<flags+32*heads && *p==target);
  assert((p-flags)%32==0);++acquires[(p-flags)/32];
}
void wait_acquire_gpu_single_lane(const uint32_t* p,uint32_t target) {
  wait_acquire_system_single_lane(p,target);
}
void fence_proxy_async_global(){++fences;}
struct Iterator {
  int coord;const int& limit;
  const int& operator*()const{return coord;}
  Iterator& operator++(){++coord;return *this;}
};
template<bool SystemScope,int StageCount> struct Adapter {
  struct Base {struct DispatchPolicy {static constexpr int Stages=StageCount;};};
'''+iterator+r'''
};
template<bool Scope,int Stages> void check(int prefix,bool rotate) {
  using R=typename Adapter<Scope,Stages>::template ReadyKIterator<Iterator>;
  for(int i=0;i<heads;++i){flags[32*i]=7;acquires[i]=0;}
  std::vector<std::thread> lanes;
  for(int lane=0;lane<32;++lane)lanes.emplace_back([&,lane]{
    threadIdx.x=64+lane;fences=0;int expected_fences=0;
    for(int part=0;part<2;++part){
      const int first=part?prefix:0,last=part?heads:prefix;
      R iter{Iterator{first,last},flags,7,last-first};iter.acquire_batch();
      int last_fenced_end=-1;
      for(int k=first;k<last;++k){
        const int issuer=rotate?(k*7+part*3)%32:0;
        if(lane==issuer){
          const int end=std::min(last,first+((k-first)/Stages+1)*Stages);
          if(end!=last_fenced_end){++expected_fences;last_fenced_end=end;}
          for(int operand=0;operand<4;++operand){
            assert(*iter==k);assert(acquires[k]==1);
            assert(fences==expected_fences);
          }
        }
        ++iter;
      }
      assert(*iter.iterator==last && iter.remaining==0 && iter.batch_remaining==0);
    }
  });
  for(auto& lane:lanes)lane.join();
  for(int i=0;i<heads;++i)assert(acquires[i]==1);
}
int main(){for(int prefix:{0,3,4})for(bool rotate:{false,true}){
  check<true,4>(prefix,rotate);check<false,3>(prefix,rotate);
}}
'''
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'iterator.cpp';binary=Path(directory)/'iterator'
            source.write_text(program)
            subprocess.run([compiler,'-std=c++17','-O2','-pthread',str(source),'-o',str(binary)],check=True,
                           capture_output=True,text=True)
            subprocess.run([str(binary)],check=True,capture_output=True,text=True,timeout=30)

    def test_transposed_quantization_register_groups_cover_k32(self):
        stores, groups = Counter(), Counter()
        for warp in range(8):
            for lane in range(32):
                row = 32 * warp + lane
                for k in range(32):
                    stores[row, k] += 1
                groups[row] += 1
            for k in range(32):
                banks = {}
                for lane in range(32):
                    word = (k*264+warp*32+lane)//2
                    banks.setdefault(word % 32,set()).add(word)
                # Two adjacent BF16 halves share one32-bit word, not two
                # different words contending for the same shared-memory bank.
                self.assertTrue(all(len(words)==1 for words in banks.values()))
        self.assertEqual(set(stores), {(r,k) for r in range(256) for k in range(32)})
        self.assertEqual(set(stores.values()), {1})
        self.assertEqual(groups, Counter(range(256)))

    def test_transpose_async_input_preserves_tail_and_stride(self):
        compiler = shutil.which('c++')
        if not compiler:
            self.skipTest('host C++ compiler required')
        text = (ROOT/'csrc/operators/sm103/detail/quantization.cuh').read_text()
        begin = text.index('CUTLASS_DEVICE void load_mxfp8_transpose_input(')
        helper = text[begin:text.index('\ntemplate <class ScaleLayout>', begin)]
        program = r'''
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>
#define CUTLASS_DEVICE
using Bf16 = uint16_t;
struct alignas(16) uint4 { uint32_t x,y,z,w; };
struct {int x=0;} threadIdx;
constexpr int kMxfp8TransposeRows=256;
int commits=0, async_copies=0;
namespace cutlass {
template<class T,int N,int A> struct alignas(A) AlignedArray {
  T v[N]; void clear(){for(auto& x:v)x=0;} T& operator[](int i){return v[i];}
};
}
namespace cute {
template<class T> struct SM80_CP_ASYNC_CACHEGLOBAL {
  static void copy(const T& src,T& dst){std::memcpy(&dst,&src,sizeof(T));++async_copies;}
};
void cp_async_fence(){++commits;}
}
''' + helper + r'''
int main() {
  for(int rows:{1,33,128,129,256,257,384})for(int padding:{0,1,2,3}) {
    int stride=rows+padding;
    std::vector<uint4> storage((128*stride+7)/8);
    auto* input=reinterpret_cast<Bf16*>(storage.data());
    for(int k=0;k<128;++k)for(int r=0;r<stride;++r)input[k*stride+r]=(k*13+r)%60000+1;
    alignas(16) Bf16 tile[32][264];
    for(int rb=0;rb<rows;rb+=256)for(int kb=0;kb<128;kb+=32) {
      for(auto& line:tile)for(auto& v:line)v=65535;
      int before=commits;
      for(int thread=0;thread<256;++thread) {
        threadIdx.x=thread;load_mxfp8_transpose_input(input,tile,rows,rb,kb,stride);
      }
      assert(commits-before==256);
      for(int k=0;k<32;++k)for(int r=0;r<264;++r)
        assert(tile[k][r]==(r>=256?65535:rb+r<rows?input[(kb+k)*stride+rb+r]:0));
    }
  }
  assert(async_copies>0);
}
'''
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'async_input.cpp';binary=Path(directory)/'async_input'
            source.write_text(program)
            subprocess.run([compiler,'-std=c++17','-O2',str(source),'-o',str(binary)],
                           check=True,capture_output=True,text=True)
            subprocess.run([str(binary)],check=True,capture_output=True,text=True)
        kernel = text[text.index('__global__ void quantize_mxfp8_transposed_operand'):]
        self.assertLess(kernel.index('cute::cp_async_wait<0>();'),
                        kernel.index('values[i] = float(tile[i][threadIdx.x]);'))
        prefetch = kernel[kernel.index('if (group + 1 < kGroups) {'):]
        self.assertLess(prefetch.index('__syncthreads();'),
                        prefetch.index('load_mxfp8_transpose_input('))

    def test_backward_log_parser_rejects_unowned_or_ambiguous_records(self):
        rows=backward_summary.parse(b'device,rank=3,sm=148,compute=10.3\n')
        self.assertEqual(rows,[dict(kind='backward_device',rank='3',sm='148',compute='10.3')])
        for raw in (b'FAIL bad\n',b'backward_unknown rank=0\n',
                    b'backward_verified pflops=nan\n',b'backward_device rank=0 rank=1\n'):
            with self.subTest(raw=raw),self.assertRaises(ValueError):
                backward_summary.parse(raw)

    def test_transpose_tile_ownership_and_padded_scale_rows(self):
        for rows, k in ((1,128), (33,128), (128,256), (129,128), (256,384), (257,128), (384,256)):
            padded = (rows+127)//128*128
            stores, scales = Counter(), Counter()
            for rb in range(0,padded,256):
                for kb in range(0,k,32):
                    tile = {}
                    for warp in range(8):
                        for lane in range(32):
                            for i in range(warp,32,8):
                                # Label each scalar by its original physical
                                # source coordinate, not its floating value.
                                for j in range(8):
                                    r=lane*8+j
                                    tile[i,r] = (kb+i,rb+r) if rb+r<rows else None
                    for thread in range(256):
                        row=rb+thread
                        for i in range(32):
                            col=kb+i
                            self.assertEqual(tile[i,thread],(col,row) if row<rows else None)
                            if row<rows: stores[row,col]+=1
                        if row<padded: scales[row,kb//32]+=1
            self.assertEqual(set(stores),{(r,c) for r in range(rows) for c in range(k)})
            self.assertEqual(set(scales),{(r,c) for r in range(padded) for c in range(k//32)})
            self.assertEqual(set(stores.values()),{1})
            self.assertEqual(set(scales.values()),{1})

    def test_transpose_grouped_writeback_preserves_data_and_scale_bytes(self):
        def offset(row, kb, k):
            return ((row//128)*((k+127)//128)+kb//128)*512 + (row%32)*16 + (row%128)//32*4 + kb%128//32
        for rows, k in ((1,128),(33,128),(128,128),(129,256),(256,384),(257,128),(384,256)):
            groups = 4
            store_groups = 2
            vectors = store_groups*2
            padded = (rows+127)//128*128
            data, scales = Counter(), {}
            for rb in range(0,padded,256):
                for kb in range(0,k,groups*32):
                    for stage in range(0,groups,store_groups):
                        for thread in range(256):
                            for i in range(thread,256*vectors,256):
                                row, col = rb+i//vectors, kb+stage*32+(i%vectors)*16
                                if row < rows:
                                    for j in range(16):data[row,col+j]+=1
                    for thread in range(64):
                        row = rb+(thread//32)*128+thread%32
                        if row >= padded:continue
                        start = offset(row,kb,k)
                        self.assertEqual(start%16,0)
                        for b in range(16):
                            self.assertNotIn(start+b,scales)
                            scales[start+b]=(row+(b//4)*32,kb//32+b%4)
            self.assertEqual(data,Counter({(r,c):1 for r in range(rows) for c in range(k)}))
            self.assertEqual(scales,{offset(r,c*32,k):(r,c) for r in range(padded) for c in range(k//32)})
            # Each16B vector transaction's eight lanes span all32 banks once.
            for first in range(0,32,8):
                banks=[((lane*(vectors+1))*4+j)%32 for lane in range(first,first+8) for j in range(4)]
                self.assertEqual(len(set(banks)),32)
        text=(ROOT/'csrc/operators/sm103/detail/quantization.cuh').read_text()
        self.assertIn('kMxfp8TransposeGroups = 4',text)
        self.assertIn('kMxfp8TransposeStoreGroups = 2',text)
        self.assertIn('extern __shared__ __align__(16) uint4 converted_storage[]',text)
        self.assertIn('reinterpret_cast<uint4 (*)[kVectors + 1]>(converted_storage)',text)
        self.assertIn('scale_words[local_row + 96]',text)
        api=(ROOT/'csrc/operators/sm103/api/backward_mxfp8.cuh').read_text()
        self.assertIn('cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active',api)
        self.assertIn('int64_t{info.sm_count} * active',api)
        for name in ('mxfp8_smoke.cu','mxfp8_mpi_bench.cu'):
            harness=(ROOT/'benchmarks/sm103/backward'/name).read_text()
            self.assertNotIn('L::kOrdinaryStatic',harness)
            self.assertIn('L::kOrdinaryDynamic,L::kCooperativeDynamic',harness)

    def test_backward_boundaries_include_preparation_and_true_weight_gradient(self):
        api=(ROOT/'csrc/operators/sm103/api/backward_mxfp8.cuh').read_text()
        header=(ROOT/'include/fuse/operators/ulysses/oproj_backward.h').read_text()
        self.assertIn('prepare_mxfp8_transpose(d.weight, w.b, w.sfb, g.n, g.k, g.n',api)
        self.assertIn('prepare_mxfp8_transpose(d.gradient, w.lhs, w.sfa, g.m, g.k, g.m',api)
        self.assertIn('prepare_mxfp8_transpose(d.input, w.rhs.b, w.rhs.sfb, g.n, g.k, g.n',api)
        self.assertIn('d.saved_attention, d.grad_weight, d.alpha, d.beta',api)
        self.assertIn('Mxfp8GemmFamily<256, 128, EpilogueN, 0, SourceElement>::PureGemm',api)
        self.assertIn('if (d.beta == 0.f)',api)
        self.assertIn('backward_mxfp8_weight_kernel<EpilogueN, Prepare, void>',api)
        self.assertIn('backward_mxfp8_weight_kernel<EpilogueN, Prepare, Bf16>',api)
        self.assertIn('args.epilogue.thread.beta = d.beta',api)
        self.assertIn('launch_oproj_backward_mxfp8_weight(p.weight, stream)',api)
        self.assertIn('Cross-CP dWo reduction is caller-owned',header)
        self.assertIn('straight-through',header)
        self.assertIn('upstream dY quantization is not',header)

    def test_qkv_weight_shares_preparation_without_relabeling_heads(self):
        api=(ROOT/'csrc/operators/sm103/api/backward_mxfp8.cuh').read_text()
        header=(ROOT/'include/fuse/operators/ulysses/qkv_backward.h').read_text()
        self.assertIn('(p.q_heads + 2 * p.kv_heads) * p.head_dim',api)
        self.assertIn('p.hidden, p.local_tokens, 1',api)
        self.assertIn('d.dqkv_staging, d.saved_input, d.grad_weight, d.alpha, d.beta',api)
        self.assertIn('backward_mxfp8_weight_impl<EpilogueN, Prepare>',api)
        self.assertIn('cross-CP dW summation remains caller-owned',header)
        self.assertIn('This W entry alone does not represent a complete QKV backward',header)

    def test_existing_input_type_and_ready_protocol_are_not_faked(self):
        api=(ROOT/'csrc/operators/sm103/api/backward_mxfp8.cuh').read_text()
        self.assertIn('Base::can_implement(args, false)',api)
        self.assertIn('args.gemm.mainloop.weight_ready = nullptr',api)
        self.assertIn('args.gemm.epilogue.n_tiles = ceil_div(g.n, 256)',api)
        self.assertNotIn('reinterpret_cast<Bf16*>',api)
        self.assertNotIn('cudaMalloc',api)
        self.assertNotIn('cudaDeviceSynchronize',api)

    def test_qkv_inverse_route_packs_heads_without_changing_k32_groups(self):
        for world,q,kv in ((4,8,8),(8,16,8),(4,64,8),(8,128,8)):
            for batch in (1,2):
                for causal in (False,True):
                    seq=256
                    for rank in range(world):
                        destinations=Counter()
                        for row in range(batch*seq):
                            b,local=divmod(row,seq)
                            if causal:
                                c=rank if local<seq//2 else 2*world-rank-1
                                source=b*seq*world+c*(seq//2)+local%(seq//2)
                            else:
                                source=b*seq*world+rank*seq+local
                            for head in range(q+2*kv):
                                kind=0 if head<q else (1 if head<q+kv else 2)
                                group=head-(0 if kind==0 else (q if kind==1 else q+kv))
                                count=(q if kind==0 else kv)//world
                                peer,own=divmod(group,count)
                                self.assertLess(peer,world)
                                # A ready task cannot cross a causal row jump
                                # or change a native aligned M128/K128 SF atom.
                                self.assertEqual(source%128,row%128)
                                destinations[row,head]+=1
                                self.assertEqual(peer*count+own,group)
                        self.assertEqual(len(destinations),batch*seq*(q+2*kv))
                        self.assertEqual(set(destinations.values()),{1})

    def test_qkv_backward_keeps_bf16_w_lease_and_complete_ready(self):
        api=(ROOT/'csrc/operators/sm103/api/backward_mxfp8.cuh').read_text()
        route=(ROOT/'csrc/operators/sm103/detail/backward.cuh').read_text()
        header=(ROOT/'include/fuse/operators/ulysses/qkv_backward.h').read_text()
        self.assertIn('main.k_tiles_per_peer=1;main.epoch=1',api)
        qkv=api[api.index('cudaError_t qkv_backward_mxfp8_data_impl'):]
        self.assertNotIn('WeightReadyMainloop',qkv)
        self.assertNotIn('main.weight_ready',qkv)
        self.assertIn('d.peer_dqkv_staging[d.rank]!=w.dqkv_staging',api)
        self.assertIn('detail::InputProductionKernel<Base,Comm>',api)
        self.assertIn('Mxfp8QkvBackwardPullComm',route)
        self.assertIn('source.master_q',route)
        self.assertIn('cooperative_groups::this_grid().sync()',route)
        self.assertIn('ALL ranks finish B',header)
        self.assertIn('No KV replication/reduction',header)
        self.assertGreater(header.index('struct Mxfp8QkvBackwardInput'),
                           header.index('using Bf16QkvBackwardDataParams'))

    def test_qkv_async_transport_slices_preserve_one_complete_head(self):
        text=(ROOT/'csrc/operators/sm103/detail/backward.cuh').read_text()
        route=text[text.index('struct Mxfp8QkvBackwardPullComm'):]
        copy_rows=int(route.split('kCopyRows = ')[1].split(';')[0])
        self.assertEqual(128%copy_rows,0)
        self.assertLessEqual(8*copy_rows*128*3,192*1024)
        for elements_per_vector in (16,8):
            owners=Counter()
            for start in range(0,128,copy_rows):
                for lane in range(32):
                    for i in range(lane,copy_rows*128//elements_per_vector,32):
                        row=start+i//(128//elements_per_vector)
                        col=i%(128//elements_per_vector)*elements_per_vector
                        for j in range(elements_per_vector):owners[row,col+j]+=1
            self.assertEqual(set(owners),{(r,c) for r in range(128) for c in range(128)})
            self.assertEqual(set(owners.values()),{1})
        self.assertIn('cute::cp_async_wait<0>()',route)
        self.assertEqual(route.count('detail::store_release_system('),1)
        self.assertLess(route.index('cute::cp_async_wait<0>()'),route.index('detail::store_release_system('))

    def test_qkv_scale_prefetch_keeps_the_original_atom_and_publication_boundary(self):
        text=(ROOT/'csrc/operators/sm103/detail/backward.cuh').read_text()
        route=text[text.index('struct Mxfp8QkvBackwardPullComm'):]
        load=route.index('asm volatile("ld.global.v4.u32')
        copies=route.index('for(int slice=0;')
        store=route.index('[lane]=scale_value;')
        release=route.index('detail::store_release_system(')
        self.assertLess(route.index('detail::fence_proxy_async_global()'),load)
        self.assertLess(load,copies)
        self.assertLess(route.index('cute::cp_async_wait<0>()'),store)
        self.assertLess(store,release)
        self.assertIn('__syncwarp();',route[store:release])
        self.assertEqual(route.count('ld.global.v4.u32'),1)
        self.assertIn('kWarpStageBytes = kCopyRows * 128 * 3;',route)
        owners=Counter(lane*16+byte for lane in range(32) for byte in range(16))
        self.assertEqual(owners,Counter(range(512)))

    def test_static_head_index_preserves_every_ready_boundary(self):
        for heads in (24,72,80,144,288,336):
            for prologue in (1,2,3,4):
                dynamic=[]; specialized=[]
                for start,end in ((0,prologue),(prologue,heads)):
                    for k in range(start,end):
                        dynamic.append((k//1,min(end-k,1-k%1)))
                        specialized.append((k,1))
                self.assertEqual(dynamic,specialized)
                self.assertEqual([p for p,_ in specialized],list(range(heads)))
        text=(ROOT/'csrc/operators/sm103/detail/cutlass_pipeline.cuh').read_text()
        self.assertIn('args.k_tiles_per_peer == StaticKTilesPerPeer',text)
        self.assertIn('int StaticKTilesPerPeer = 0',text)

    def test_smoke_cannot_be_mistaken_for_full_performance(self):
        text=(ROOT/'benchmarks/sm103/backward/mxfp8_smoke.cu').read_text()
        self.assertIn('curandStatePhilox4_32_10_t',text)
        self.assertIn('generation<2',text)
        self.assertIn('performance=not_measured',text)
        self.assertIn('double sum = 0',text)
        self.assertIn('dA inverse route bytes',text)
        self.assertIn('deferred B modified dW',text)

    def test_formal_boundary_and_reference_are_independent_and_complete(self):
        text=(ROOT/'benchmarks/sm103/backward/mxfp8_mpi_bench.cu').read_text()
        ref=(ROOT/'benchmarks/sm103/backward/mxfp8_reference.cuh').read_text()
        self.assertIn('const double flops=4.*o.m*o.h*o.width()',text)
        self.assertIn('WeightGradientMode::kImmediate',text)
        self.assertIn('r.validate(o,generation,"pre")',text)
        self.assertIn('r.validate(o,generation,"post")',text)
        self.assertIn('collect(50,"measurement",round)',text)
        self.assertIn('graph.committed_epoch()+1',text)
        self.assertIn('fused_inputs::generate',text)
        self.assertIn('cudaIpcOpenMemHandle',text)
        self.assertIn('reference.validate({dy,o.h,1},{weight,1,route.a}',text)
        self.assertIn('reference.validate({dy,1,o.h},{attention,1,route.a}',text)
        self.assertIn('frexpf(amax, &exponent)',ref)
        self.assertIn('CUBLAS_COMPUTE_32F_PEDANTIC',ref)
        self.assertIn('kRows = 128, kChunk = 4096',ref)
        self.assertNotIn('ptr_SFA',ref)
        self.assertNotIn('ptr_SFB',ref)
        self.assertNotIn('quantize_mxfp8_transposed_operand',ref)

    def test_reference_cache_is_payload_scoped_and_checker_remains_active(self):
        text=(ROOT/'benchmarks/sm103/backward/mxfp8_mpi_bench.cu').read_text()
        ref=(ROOT/'benchmarks/sm103/backward/mxfp8_reference.cuh').read_text()
        smoke=(ROOT/'benchmarks/sm103/backward/mxfp8_smoke.cu').read_text()
        self.assertIn('expected_data && reference_generation==generation',text)
        self.assertIn('fused_mpi::any(free<bytes+(2ull<<30))',text)
        self.assertIn('initialize_reference_cache(o.m,a,o.h)',text)
        self.assertIn('initialize_reference_cache(o.m,o.h,a)',text)
        self.assertIn('stream,expected_data,reuse)',text)
        self.assertIn('stream,expected_weight,reuse)',text)
        self.assertIn('(reuse_cache && !cache)',ref)
        self.assertIn('cache ? cache + int64_t(row)*n : expected',ref)
        self.assertIn('finish<<<256,256,0,stream>>>(accumulator,expected_slab',ref)
        self.assertIn('      }\n      check(fused_validation::launch<true>',ref)
        self.assertIn('cudaMemsetAsync(actual,0xff,sizeof(saved)',smoke)
        self.assertIn('r.stream,cache,true)',smoke)
        self.assertIn('cached oracle missed poisoned actual output',smoke)

    def test_isolated_components_preserve_full_boundary_and_native_epoch(self):
        text=(ROOT/'benchmarks/sm103/backward/mxfp8_mpi_bench.cu').read_text()
        api=(ROOT/'csrc/operators/sm103/api/backward_mxfp8.cuh').read_text()
        self.assertIn('template <int EpilogueN, bool Prepare = true>',api)
        self.assertIn('if constexpr (Prepare)',api)
        self.assertIn('oproj_backward_mxfp8_weight_impl<32, false>',api)
        self.assertIn('Component::kData,Component::kWeight,Component::kWeightCompute',text)
        self.assertIn('component==Component::kFull || component==Component::kData',text)
        weight=text.index('if(component==Component::kWeight)return fuse::launch_oproj_backward_mxfp8_weight')
        epoch=text.index('params.data.projection.epoch=epoch;',weight)
        self.assertLess(weight,epoch)
        self.assertLess(text.index('if(component==Component::kWeightCompute)',weight),epoch)
        self.assertIn('graph.reset(r.native_epoch())',text)
        self.assertIn('component==Component::kData?r.native_epoch():0',text)
        self.assertIn('(flops/2)/(value.p50*1e12)',text)
        self.assertIn('r.validate(o,generation,"post");results.push_back(result)',text)

    def test_prepared_dx_preserves_ready_budget_and_weight_workspace_lease(self):
        api=(ROOT/'csrc/operators/sm103/api/backward_mxfp8.cuh').read_text()
        harness=(ROOT/'benchmarks/sm103/backward/mxfp8_mpi_bench.cu').read_text()
        route=(ROOT/'csrc/operators/sm103/detail/backward.cuh').read_text()
        header=(ROOT/'include/fuse/operators/ulysses/qkv_backward.h').read_text()
        self.assertIn('using Comm=Mxfp8QkvBackwardPullComm<!Prepare>',api)
        self.assertIn('qkv_backward_mxfp8_data_impl<32,false>',api)
        self.assertIn('if constexpr (Prepared) return',route)
        self.assertIn('compute-CTA budget',header)
        self.assertIn('Do not run W on this scratch',header)
        self.assertIn('components.insert(components.begin()+1,Component::kDataCompute)',harness)
        self.assertIn('std::vector<L>{L::kCooperativeDynamic}',harness)

    def test_qkv_formal_reference_reads_original_peer_planes(self):
        text=(ROOT/'benchmarks/sm103/backward/mxfp8_mpi_bench.cu').read_text()
        ref=(ROOT/'benchmarks/sm103/backward/mxfp8_reference.cuh').read_text()
        self.assertIn('struct QkvGradient',ref)
        self.assertIn('source[owner][kind][int64_t(global)*width+column]',ref)
        self.assertIn('reference.validate_views(original_qkv',text)
        self.assertIn('original_BF16_route=included',text)
        self.assertIn('input_lease=all_ranks_until_B_complete',text)
        self.assertIn('(heads+(qkv?2*kv_heads:0))*128',text)

    def test_bare_dx_is_a_separate_completed_input_diagnostic(self):
        api=(ROOT/'csrc/operators/sm103/api/backward_mxfp8.cuh').read_text()
        harness=(ROOT/'benchmarks/sm103/backward/mxfp8_mpi_bench.cu').read_text()
        header=(ROOT/'include/fuse/operators/ulysses/qkv_backward.h').read_text()
        auditor=(ROOT/'scripts/summarize_sm103_mxfp8_backward.py').read_text()
        self.assertIn('bool WaitInput = true',api)
        self.assertIn('static_assert(WaitInput || !Prepare',api)
        self.assertIn('std::conditional_t<WaitInput,ReadyMainloop,typename Types::Mainloop>',api)
        self.assertIn('qkv_backward_mxfp8_data_impl<32,false,false>',api)
        self.assertIn('components.insert(components.begin()+2,Component::kDataGemm)',harness)
        self.assertIn('component==Component::kDataCompute || component==Component::kDataGemm',harness)
        self.assertIn('Never use this entry while input production is in flight',header)
        self.assertIn("c.get('data_gemm_reference')=='1'",auditor)
        self.assertIn('prepared_dX_same_budget_stock_collective_no_adapter_quantization_or_transport',auditor)


if __name__ == '__main__':
    unittest.main()
