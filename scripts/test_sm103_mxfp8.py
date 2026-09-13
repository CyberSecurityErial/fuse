"""MXFP8 quantization contracts; host checks are not CUDA validation."""

import os
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import tarfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


class Mxfp8QuantizationContracts(unittest.TestCase):
    def test_joint_fused_search_starts_from_registered_gemm(self):
        import tune_sm103_mxfp8_fused as search
        seed = search.gemm_config({'id':'example','winner':{'config':'m128n256k128e32s0sw8M'}})
        self.assertEqual(seed, (32,8,'along_m'))
        self.assertEqual(search.layout_neighbors(seed), [(32,8,'along_n'),(32,4,'along_m')])
        self.assertEqual(len(search.layout_neighbors((32,2,'along_n'))),3)
        with self.assertRaises(ValueError):
            search.gemm_config({'id':'unsupported','winner':{'config':'m128n128k128e32s0sw8M'}})

    def test_native_log_keeps_full_device_budget_contract(self):
        compiler = shlex.split(os.environ.get('CXX', 'c++'))
        if not compiler or not shutil.which(compiler[0]):
            self.skipTest('host C++ compiler required')
        text=(ROOT/'benchmarks/sm103/GEMM/mxfp8_bench.cu').read_text()
        start=text.index('    std::cout<<std::setprecision(9)<<"config,pure_mxfp8,')
        log=text[start:text.index(';',start)+1]
        program='#include <iostream>\n#include <iomanip>\nint main(){ int candidates=32; struct {int multiProcessorCount=148;} props; for(int budget:{0,132}) {'+log+'}}'
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'metadata.cpp'; source.write_text(program)
            executable=Path(folder)/'metadata'
            subprocess.run([*compiler,'-std=c++17',str(source),'-o',str(executable)],check=True,capture_output=True)
            lines=subprocess.check_output([str(executable)],text=True).splitlines()
            self.assertIn(',sm_budget=full_device,',lines[0])
            self.assertIn(',compute_ctas=132,',lines[1])
            self.assertNotIn('sm_budget=full_device',lines[1])

    def test_search_report_requires_full_evidence_and_first_stable_round(self):
        import summarize_sm103_mxfp8_search as report
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); control=root/'artifacts-attempt1/control'
            control.mkdir(parents=True)
            config=report.BASELINE
            shape=dict(id='example',m=128,n=256,k=128)
            job=dict(run_id='unit',mxfp8_gemm_search=True,gemm_sm_budget=132,
                     gemm_matrix_payload=dict(shapes=[shape]))
            (control/'job.json').write_text(json.dumps(job))
            (control/'status.json').write_text(json.dumps(dict(work_exit_code=0)))
            (control/'mxfp8-search-build.json').write_text('{}')
            result=dict(**shape,status='passed',backend='cutlass_mxfp8',config=config,
                        compute_ctas=132,p50_ms=1.,p95_ms=1.,drift=0)
            lines=[f'RUN cutlass_mxfp8,id=example,candidate={config}']
            for generation,phase in ((0,'pre'),(0,'post'),(1,'post')):
                lines.append(f'correctness,pure_mxfp8,id=example,generation={generation},phase={phase},'
                             'checked=32768,mismatches=0,nonfinite=0')
            lines += [f'samples,cutlass_mxfp8,id=example,config={config},round=0,ms='+json.dumps([1.]*50),
                      f'samples,cutlass_mxfp8,id=example,config={config},round=1,ms='+json.dumps([.5]*50),
                      'RESULT '+json.dumps(result)]
            def save():
                (control/'attempt1.log').write_text('\n'.join(lines))
                with tarfile.open(root/'artifacts.tar.gz','w:gz') as archive:
                    for p in control.iterdir(): archive.add(p,arcname='control/'+p.name)
                digest=hashlib.sha256((root/'artifacts.tar.gz').read_bytes()).hexdigest()
                (root/'fetched.json').write_text(json.dumps(dict(state='succeeded',exit_code=0,artifact_sha256=digest)))
            save()
            summary=report.audit(root)
            self.assertEqual(summary['rows'][0]['winner']['p50_ms'],1.)
            complete_lines = list(lines)
            missing = 'm128n256k128e32s0sw8N'
            lines += [f'RUN cutlass_mxfp8,id=example,candidate={missing}']
            for generation,phase in ((0,'pre'),(0,'post'),(1,'post')):
                lines.append(f'correctness,pure_mxfp8,id=example,generation={generation},phase={phase},'
                             'checked=32768,mismatches=0,nonfinite=0')
            lines += [f'samples,cutlass_mxfp8,id=example,config={missing},round=0,ms='+json.dumps([.5]*50), '']
            # The first candidate has a RESULT; the faster verified second
            # candidate does not. A printed prefix cannot select the winner.
            save()
            with self.assertRaisesRegex(ValueError,'Incomplete result emission'):
                report.audit(root)
            partial = report.audit(root,allow_boundary_partial=True)
            self.assertEqual(partial['pending'],[shape])
            self.assertEqual(partial['rows'],[])
            lines = complete_lines
            save()
            tail=json.loads(json.dumps(summary))
            tail['run_id']='unit-tail'
            tail['rows'][0]['id']='example-tail'
            first=json.loads(json.dumps(summary))
            first['pending']=[dict(shape,id='example-tail')]
            first['partial']=True
            merged=report.merge(first,tail)
            self.assertEqual(len(merged['rows']),2)
            self.assertFalse(merged['partial'])
            self.assertEqual(merged['rows'][1]['source_run_id'],'unit-tail')
            with self.assertRaises(ValueError):
                report.merge(first,tail | {'compute_ctas':148})
            # A new explicit budget must propagate through every result; an old
            # 132-CTA record cannot be accepted just by relabeling the job.
            job['gemm_sm_budget']=128
            (control/'job.json').write_text(json.dumps(job))
            save()
            with self.assertRaisesRegex(ValueError,'budget mismatch'):
                report.audit(root)
            result['compute_ctas']=128
            lines[-1]='RESULT '+json.dumps(result)
            save()
            updated=report.audit(root)
            self.assertEqual(updated['compute_ctas'],128)
            self.assertIn('128 个计算 CTA',report.render(updated))
            lines=[line for line in lines if 'generation=1' not in line]
            save()
            with self.assertRaisesRegex(ValueError,'dual-payload'):
                report.audit(root)

    def test_cutlass_search_grid_and_neighbors(self):
        compiler = shlex.split(os.environ.get('CXX', 'c++'))
        if not compiler or not shutil.which(compiler[0]):
            self.skipTest('host C++ compiler required')
        text = (ROOT / 'benchmarks/sm103/GEMM/mxfp8_search.cuh').read_text()
        config = text[text.index('struct Config {'):text.index('template<int N,')]
        grid = text[text.index('inline std::vector<Config> grid()'):text.index('} // namespace mxfp8_search')]
        program = '#include <vector>\n#include <string>\n#include <tuple>\n#include <set>\n' + config + grid + r'''
int main() {
  auto all=grid(); std::set<decltype(all.front().key())> keys;
  if(all.size()!=32 || all.front().name()!="m128n256k128e64s0sw1M")return 1;
  for(auto c:all) {
    if(!keys.insert(c.key()).second)return 2;
    for(auto v:neighbors(c)) {
      int delta=(c.n!=v.n)+(c.k!=v.k)+(c.e!=v.e)+(c.stages!=v.stages)+
                (c.swizzle!=v.swizzle)+(c.along_m!=v.along_m);
      if(delta>1 || (v.n==256 && v.k==256 && v.stages!=0))return 3;
    }
  }
}
'''
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'grid.cpp'; source.write_text(program)
            executable=Path(folder)/'grid'
            subprocess.run([*compiler,'-std=c++17',str(source),'-o',str(executable)],check=True,capture_output=True)
            subprocess.run([str(executable)],check=True,capture_output=True)

    def test_register_tile_mapping_padding_and_unaligned_rows(self):
        # Run the actual one-K32-group-per-lane chunk helper. K128 and K384
        # make the same chunk cross multiple source rows. The small
        # integer storage/converter below checks addressing and ownership, not
        # BF16/FP8 rounding; real conversion is validated by the GPU oracle.
        compiler = shlex.split(os.environ.get('CXX', 'c++'))
        if not compiler or not shutil.which(compiler[0]):
            self.skipTest('host C++ compiler required')
        text = (ROOT / 'csrc/operators/sm103/detail/quantization.cuh').read_text()
        begin = text.index('CUTLASS_HOST_DEVICE int mxfp8_scale_exponent(')
        math = text[begin:text.index('// One warp owns', begin)]
        begin = text.index('template <class ScaleLayout>\nCUTLASS_DEVICE void quantize_mxfp8_chunk')
        helper = text[begin:text.index('\ntemplate <class ScaleLayout>', begin + 1)]
        program = r'''
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <tuple>
#include <vector>
#define CUTLASS_HOST_DEVICE
#define CUTLASS_DEVICE
#define CUTLASS_PRAGMA_UNROLL
using Bf16=int16_t;
using Fp8E4m3=uint8_t;
struct {int x=0;} threadIdx;
namespace cute { auto make_coord(int r,int c,int z){return std::make_tuple(r,c,z);} }
uint8_t encode(float v) {
  return uint8_t(int(std::clamp(std::nearbyint(v/4),-127.0f,127.0f))+128);
}
namespace cutlass {
using float_ue8m0_t=uint8_t;
template<class T,int N> struct Array {
  T v[N]; T& operator[](int i){return v[i];} const T& operator[](int i)const{return v[i];}
  void clear(){for(auto& x:v)x=T{};}
};
template<class T,int N,int Align> struct alignas(Align) AlignedArray: Array<T,N> {};
template<class To,class From,int N> struct NumericArrayConverter {
  Array<To,N> operator()(const Array<From,N>& in)const {
    Array<To,N> out; for(int i=0;i<N;++i)out[i]=encode(in[i]); return out;
  }
};
}
std::vector<int> scale_writes;
struct Layout {
  int k;
  int operator()(std::tuple<int,int,int> c)const {
    int row=std::get<0>(c),column=std::get<1>(c),offset=row*(k/32)+column/32;
    if(row<0 || column<0 || column>=k || column%32 || offset>=int(scale_writes.size()))
      throw std::runtime_error("invalid scale coordinate");
    if(++scale_writes[offset]!=1)throw std::runtime_error("duplicate scale writer");
    return offset;
  }
};
float __shfl_xor_sync(uint32_t,float,int,int) {
  throw std::runtime_error("K32-per-lane helper must not use shuffle");
}
struct alignas(16) OutputWords { uint64_t lo,hi; };
''' + math + helper + r'''
int main() {
  for(int rows:{1,127,128,129,256,385})
  for(int k:{128,256,384,640,1024,1536,2048,3072,8192})for(int padding:{0,1,2,3}) {
    int padded=(rows+127)/128*128, stride=k+padding;
    std::vector<Bf16> input(rows*stride,32767);
    for(int r=0;r<rows;++r)for(int c=0;c<k;++c)
      input[r*stride+c]=((r*(k/32)+c/32)%7==0)?0:(r*71+c*13)%3803-1901;
    constexpr uint64_t guard=0xa5a5a5a5a5a5a5a5ull;
    std::vector<OutputWords> storage(rows*k/16+2,{guard,guard});
    auto* output=reinterpret_cast<uint8_t*>(storage.data())+16;
    std::vector<uint8_t> scale_storage(padded*(k/32)+2,0xff);
    auto* scales=scale_storage.data()+1;
    scale_writes.assign(padded*(k/32),0);
    for(int64_t first=0;first<int64_t(padded)*(k/32);first+=32) {
      for(int lane=0;lane<32;++lane) {
        threadIdx.x=lane;
        quantize_mxfp8_chunk(input.data(),output,scales,rows,k,stride,Layout{k},
            int(first/(k/32)),int(first%(k/32))*32);
      }
    }
    for(int g=0;g<padded*(k/32);++g) {
      int r=g/(k/32),c=(g%(k/32))*32; float amax=0;
      if(r<rows)for(int i=0;i<32;++i)amax=std::max(amax,std::fabs(float(input[r*stride+c+i])));
      int exponent=mxfp8_scale_exponent(amax);
      if(scales[g]!=exponent+127)throw std::runtime_error("scale address/value");
      if(scale_writes[g]!=1)throw std::runtime_error("missing scale writer");
      if(r<rows)for(int i=0;i<32;++i) {
        uint8_t value=encode(std::ldexp(float(input[r*stride+c+i]),-exponent));
        if(output[r*k+c+i]!=value)throw std::runtime_error("vector output address/value");
      }
    }
    if(storage.front().lo!=guard || storage.front().hi!=guard ||
        storage.back().lo!=guard || storage.back().hi!=guard)
      throw std::runtime_error("output bounds");
    if(scale_storage.front()!=0xff || scale_storage.back()!=0xff)
      throw std::runtime_error("scale bounds");
  }
}
'''
        with tempfile.TemporaryDirectory(prefix='fuse-mxfp8-vector-') as directory:
            binary = Path(directory) / 'vector'
            built = subprocess.run([*compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                '-x', 'c++', '-', '-o', str(binary)], input=program, text=True,
                capture_output=True, timeout=30)
            self.assertEqual(built.returncode, 0, built.stderr)
            result = subprocess.run([str(binary)], text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_power_of_two_multiply_matches_ldexp_for_all_finite_bf16_values(self):
        compiler = shlex.split(os.environ.get('CXX', 'c++'))
        if not compiler or not shutil.which(compiler[0]):
            self.skipTest('host C++ compiler required')
        text = (ROOT / 'csrc/operators/sm103/detail/quantization.cuh').read_text()
        begin = text.index('CUTLASS_HOST_DEVICE float mxfp8_scaled_value(')
        function = text[begin:text.index('\n}\n', begin) + 3]
        program = r'''
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#define CUTLASS_HOST_DEVICE
''' + function + r'''
int main() {
  for (uint32_t bits=0; bits<65536; ++bits) {
    if ((bits & 0x7f80) == 0x7f80) continue;
    uint32_t raw=bits<<16;
    float value;
    std::memcpy(&value,&raw,sizeof(value));
    for (int exponent=-127; exponent<=127; ++exponent) {
      float actual=mxfp8_scaled_value(value,exponent);
      float expected=std::ldexp(value,-exponent);
      uint32_t a,b;
      std::memcpy(&a,&actual,4); std::memcpy(&b,&expected,4);
      if(a!=b) throw std::runtime_error("power-of-two scaling changed float bits");
    }
  }
}
'''
        with tempfile.TemporaryDirectory(prefix='fuse-mxfp8-scaling-') as directory:
            binary = Path(directory) / 'scaling'
            built = subprocess.run([*compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                '-x', 'c++', '-', '-o', str(binary)], input=program, text=True,
                capture_output=True, timeout=30)
            self.assertEqual(built.returncode, 0, built.stderr)
            result = subprocess.run([str(binary)], text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_every_finite_positive_bf16_amax_selects_minimum_nonoverflow_scale(self):
        compiler = shlex.split(os.environ.get('CXX', 'c++'))
        if not compiler or not shutil.which(compiler[0]):
            self.skipTest('host C++ compiler required')
        text = (ROOT / 'csrc/operators/sm103/detail/quantization.cuh').read_text()
        begin = text.index('CUTLASS_HOST_DEVICE int mxfp8_scale_exponent(')
        function = text[begin:text.index('\n}\n', begin) + 3]
        program = r'''
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#define CUTLASS_HOST_DEVICE
''' + function + r'''
int main() {
  for (uint32_t bits=0; bits<0x7f80; ++bits) {
    uint32_t raw=bits<<16;
    float amax;
    std::memcpy(&amax,&raw,sizeof(amax));
    const int exponent=mxfp8_scale_exponent(amax);
    if(exponent < -127 || exponent > 127) throw std::runtime_error("scale range");
    if(amax == 0) { if(exponent != 0) throw std::runtime_error("zero group"); continue; }
    if(std::ldexp(double(amax),-exponent)>448) throw std::runtime_error("FP8 overflow");
    if(exponent > -127 && std::ldexp(double(amax),1-exponent)<=448)
      throw std::runtime_error("scale needlessly loses precision");
  }
}
'''
        with tempfile.TemporaryDirectory(prefix='fuse-mxfp8-contract-') as directory:
            binary = Path(directory) / 'scale'
            built = subprocess.run([*compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror',
                '-x', 'c++', '-', '-o', str(binary)], input=program, text=True,
                capture_output=True, timeout=30)
            self.assertEqual(built.returncode, 0, built.stderr)
            result = subprocess.run([str(binary)], text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_dynamic_weight_is_inside_one_persistent_kernel(self):
        api = (ROOT / 'csrc/operators/sm103/api/forward_mxfp8.cuh').read_text()
        start = api.index('cudaError_t launch_gemm_a2a_mxfp8_cutlass(')
        launch = api[start:api.index('\n}', start)]
        for epilogue in (32, 64):
            self.assertIn(f'launch_mxfp8<false, {epilogue}>(params, stream, true)', launch)
        self.assertNotIn('prepare_gemm_a2a_mxfp8(', launch)
        self.assertIn('InputProductionKernel<Base, Mxfp8WeightProducer, Profile>', api)
        self.assertIn('params.activation.data', api)
        prepare = api[api.index('cudaError_t prepare_gemm_a2a_mxfp8('):start]
        self.assertIn('p.projection.rhs_nt, w.b, w.sfb', prepare)
        self.assertNotIn('p.projection.lhs', prepare)

    def test_fused_epilogue_selection_preserves_ready_geometry(self):
        api = (ROOT / 'csrc/operators/sm103/api/forward_mxfp8.cuh').read_text()
        bindings = (ROOT / 'csrc/operators/sm103/detail/launch.cuh').read_text()
        self.assertIn('Mxfp8GemmFamily<256, 128, EpilogueN>', bindings)
        self.assertIn('using Binding = Mxfp8QkvBinding<EpilogueN>', api)
        self.assertIn('p.epilogue_n != 32 && p.epilogue_n != 64', api)
        self.assertIn('args.mainloop.weight_panels = workspace.panels', api)
        self.assertIn('p.num_comm_ctas, info, GemmRaster::kAlongM', api)
        header = (ROOT / 'include/fuse/operators/primitives/gemm_a2a_mxfp8.h').read_text()
        self.assertIn('int32_t epilogue_n = 64', header)
        import l20d
        job = dict(stage='fused-smoke', mxfp8=True, mpi=True, fused_direction='qkv',
                   world=8, hidden=2048, q_heads=16, kv_heads=8, head_dim=128,
                   global_seq=131072, qkv_policy_list='m128n256', comm_sm_list='16,24',
                   mxfp8_epilogue_n=32)
        argv = l20d.fused_argv(job)
        self.assertEqual(argv[argv.index('--mxfp8-epilogue-n')+1], '32')

    def test_progress_and_ready_granularity(self):
        comm = (ROOT / 'csrc/operators/sm103/detail/gemm_a2a.cuh').read_text()
        self.assertIn('if constexpr (InputWork::kEnabled && QuantWarps == 0) input_work.drain()', comm)
        self.assertEqual(comm.count('input_work.progress()'), 3)
        pipe = (ROOT / 'csrc/operators/sm103/detail/cutlass_pipeline.cuh').read_text()
        begin = pipe.index('struct WeightReadyMainloop')
        ready = pipe[begin:pipe.index('\n};', begin) + len('\n};')]
        self.assertIn('n != acquired_n_', ready)
        self.assertEqual(ready.count('Base::load('), 1)
        self.assertNotIn('while (', ready)
        harness = (ROOT / 'benchmarks/sm103/fused_bf16.cu').read_text()
        self.assertIn('params.projection.lhs = nullptr', harness)
        self.assertIn('includes_activation_quantization=0', harness)
        self.assertIn('kernel_nodes=1', harness)

    def test_communication_warp_specialization_keeps_independent_queues(self):
        comm = (ROOT / 'csrc/operators/sm103/detail/gemm_a2a.cuh').read_text()
        self.assertIn('static_cast<int64_t>(comm_ctas) * kQkvBulkSlots', comm)
        self.assertEqual(comm.count('if constexpr (InputWork::kEnabled && QuantWarps == 0)'), 4)
        handoff = 'if (slot >= kQkvBulkSlots - QuantWarps) input_work.drain();'
        self.assertIn(handoff, comm)
        route = comm[comm.index(handoff):comm.index('// Follow resolved GEMM production order')]
        self.assertIn('cute::initialize_barrier', route)
        self.assertNotIn('__syncthreads', route)
        self.assertNotIn('atomic', route)
        self.assertIn('run<TraceTasks, Mxfp8WeightProducer, 4, QkvHeadPostprocess>', comm)
        self.assertIn('quant_warps * comm_ctas', comm)
        # Independent strided queues must cover every item exactly once even
        # when the work size is not a multiple of either worker population.
        for ctas in (1, 3, 16, 31):
            route_workers = [warp * ctas + cta for warp in range(8) for cta in range(ctas)]
            quant_workers = [(warp % 4) * ctas + cta for warp in range(4, 8) for cta in range(ctas)]
            for workers in (route_workers, quant_workers):
                self.assertEqual(sorted(workers), list(range(len(workers))))
                for count in (1, 127, 512, 8192, 8193):
                    seen = [item for worker in workers for item in range(worker, count, len(workers))]
                    self.assertEqual(sorted(seen), list(range(count)))

    def test_weight_publication_preserves_warp_device_and_proxy_handoff(self):
        # Guard the audited source protocol, not a substitute for CUDA testing.
        source = (ROOT / 'csrc/operators/sm103/detail/quantization.cuh').read_text()
        producer = source[source.index('struct Mxfp8WeightProducer'):]
        progress = producer[producer.index('CUTLASS_DEVICE bool progress()'):]
        self.assertNotIn('__threadfence(', progress)
        self.assertNotIn('fence_done', progress)
        self.assertEqual(progress.count('__syncwarp();'), 2)
        self.assertLess(progress.index('quantize_mxfp8_chunk('), progress.index('__syncwarp();'))
        self.assertLess(progress.index('__syncwarp();'), progress.index('count.fetch_add('))
        self.assertIn('cuda::atomic_ref<uint32_t, cuda::thread_scope_device>', progress)
        self.assertIn('count.fetch_add(delta, cuda::memory_order_acq_rel)', progress)
        self.assertIn('uint32_t pending_chunks', producer)
        self.assertLess(progress.index('count.fetch_add('), progress.index('if (arrived == expected)'))
        self.assertLess(progress.index('if (arrived == expected)'), progress.index('detail::store_release_gpu('))
        self.assertIn('cooperative_groups::this_grid().sync();', producer)
        pipe = (ROOT / 'csrc/operators/sm103/detail/cutlass_pipeline.cuh').read_text()
        consumer = pipe[pipe.index('struct WeightReadyMainloop'):pipe.index('struct A2ALhsTimelineArguments')]
        self.assertLess(consumer.index('wait_acquire_gpu_single_lane('), consumer.index('__syncwarp();'))
        self.assertLess(consumer.index('__syncwarp();'), consumer.index('fence_proxy_async_global();'))
        self.assertLess(consumer.index('fence_proxy_async_global();'), consumer.index('return Base::load('))

    def test_actual_weight_queue_covers_groups_once_and_publishes_only_complete_panels(self):
        # Execute production queue/counter code with CPU memory-operation stubs.
        # This validates ownership, padding and publication counts, NOT CUDA
        # ordering, numerical conversion or GPU forward-progress guarantees.
        compiler = shlex.split(os.environ.get('CXX', 'c++'))
        if not compiler or not shutil.which(compiler[0]):
            self.skipTest('host C++ compiler required')
        source = (ROOT / 'csrc/operators/sm103/detail/quantization.cuh').read_text()
        producer = source[source.index('struct Mxfp8WeightProducer'):source.index('\n}  // namespace')]
        order_source = (ROOT / 'csrc/operators/sm103/detail/producer_consumer.cuh').read_text()
        order = order_source[order_source.index('struct NBandSwizzle'):order_source.index('struct ProducerTileOrder')]
        program = r'''
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <random>
#define CUTLASS_DEVICE
#define CUTLASS_HOST_DEVICE
using std::min;
using Bf16 = float;
constexpr int kReadyFlagStride=32;
int ceil_div(int a,int b) { return (a+b-1)/b; }
struct Dim { int x=1; } blockIdx, blockDim, gridDim, threadIdx{0};
void __syncwarp() {}
namespace cooperative_groups { struct Grid { void sync() {} }; Grid this_grid(){return {};} }
namespace cute { template<class...T> int make_shape(T...){return 0;} }
struct Mxfp8ScaleConfig { static int tile_atom_to_shape_SFB(int){return 0;} };
struct Mxfp8Workspace { void* b=nullptr; void* sfb=nullptr; uint32_t* arrivals; uint32_t* ready; int panels; };
std::vector<int> seen, completed_chunks, atomic_calls, release_calls;
std::vector<uint32_t> delta_sum, worker_pending, worker_publications;
int panel_groups=0, panel_count=0, current_worker=-1, last_chunk_panel=-1;
uint32_t* flags=nullptr;
uint32_t* arrival_base=nullptr;
void observe_atomic(uint32_t* p,uint32_t delta) {
  int offset=int(p-arrival_base), panel=offset/kReadyFlagStride;
  if(offset<0 || offset%kReadyFlagStride || panel>=panel_count || current_worker<0 || !delta)
    throw std::runtime_error("invalid counter address/worker/delta");
  int id=current_worker*panel_count+panel;
  if(delta!=worker_pending[id] || ++worker_publications[id]!=1)
    throw std::runtime_error("publication must include exactly this worker's panel chunks once");
  worker_pending[id]=0;
  ++atomic_calls[panel]; delta_sum[panel]+=delta;
}
namespace cuda {
enum {thread_scope_device, memory_order_acq_rel};
template<class T,int Scope> struct atomic_ref {
  T& x; atomic_ref(T& v):x(v){}
  T fetch_add(T a,int order) {
    if(order!=memory_order_acq_rel) throw std::runtime_error("counter memory order");
    observe_atomic(&x,a); T old=x; x+=a; return old;
  }
};
}
namespace detail {
void store_release_gpu(uint32_t* p,uint32_t epoch) {
  int panel=(p-flags)/kReadyFlagStride;
  for(int i=panel*panel_groups; i<std::min(int(seen.size()),(panel+1)*panel_groups);++i)
    if(seen[i]!=1) throw std::runtime_error("early publication");
  if(++release_calls[panel]!=1) throw std::runtime_error("duplicate ready publication");
  *p=epoch;
}
''' + order + r'''
}
void quantize_mxfp8_chunk(const float*,void*,void*,int,int k,int64_t,int,int row,int column) {
  if(column<0 || column%128 || column+128>k || row<0)
    throw std::runtime_error("invalid chunk origin or alignment");
  int64_t first=int64_t(row)*(k/32)+column/32;
  for(int64_t g=first; g<first+32; ++g)
    if(g<0 || g>=int64_t(seen.size()) || ++seen[g]!=1) throw std::runtime_error("group coverage");
  last_chunk_panel=int(first/panel_groups);
  ++completed_chunks[last_chunk_panel];
  ++worker_pending[current_worker*panel_count+last_chunk_panel];
}
struct Schedule {
  enum class RasterOrder { AlongM, AlongN };
  RasterOrder raster_order_;
  struct Divide { uint64_t divisor; } divmod_cluster_blk_major_, divmod_batch_;
};
''' + producer + r'''
struct Comm { Mxfp8WeightProducer::Arguments weights; Schedule producer_order; detail::NBandSwizzle n_band_swizzle; };
int main() {
  std::mt19937 rng(20260910);
  float source=1;
  for(int n : {0,1,127,128,129,256,384,385,768,1536,2304})
  for(int k : {128,256,384,640,1024,1536,2048,3072,8192})
  for(int workers : {1,3,64,128,192,384,1184}) for(int along : {0,1}) for(int rotate : {0,1}) {
    int panels=ceil_div(n,256), extent=ceil_div(panels,4)*4;
    std::vector<uint32_t> arrivals(panels*kReadyFlagStride), ready(arrivals.size());
    Mxfp8WeightProducer::Arguments a{};
    a.source=&source; a.n=n; a.k=k; a.row_stride=k; a.epoch=7;
    a.workspace.arrivals=arrivals.data(); a.workspace.ready=ready.data(); a.workspace.panels=panels;
    Schedule s{along?Schedule::RasterOrder::AlongN:Schedule::RasterOrder::AlongM,
        {uint64_t(along?extent:16)}, {uint64_t(16*extent)}};
    detail::NBandSwizzle order{rotate?extent/2:0,extent};
    // Replay the SAME epoch after poison to prove reset is not cumulative.
    for(int replay=0;replay<2;++replay) {
      std::fill(arrivals.begin(),arrivals.end(),0xdead); std::fill(ready.begin(),ready.end(),0xbeef);
      blockIdx.x=0; blockDim.x=gridDim.x=1;
      Mxfp8WeightProducer::initialize_grid(Comm{a,s,order});
      for(int i=0;i<panels;++i)
        if(arrivals[i*kReadyFlagStride] || ready[i*kReadyFlagStride]) throw std::runtime_error("reset failed");
      seen.assign(ceil_div(n,128)*128*(k/32),0); flags=ready.data(); panel_groups=256*(k/32);
      panel_count=panels; arrival_base=arrivals.data();
      completed_chunks.assign(panels,0); atomic_calls.assign(panels,0); release_calls.assign(panels,0);
      delta_sum.assign(panels,0); worker_pending.assign(workers*panels,0);
      worker_publications.assign(workers*panels,0);
      std::vector<int> valid_steps(panels);
      for(int panel=0;panel<panels;++panel)
        valid_steps[panel]=std::min(panel_groups,int(seen.size())-panel*panel_groups)/32;
      std::vector<Mxfp8WeightProducer> work;
      for(int i=0;i<workers;++i) work.emplace_back(a,s,order,i,workers);
      std::vector<int> active; for(int i=0;i<workers;++i) active.push_back(i);
      while(!active.empty()) {
        int slot=rng()%active.size();
        current_worker=active[slot]; last_chunk_panel=-1;
        bool progressed=work[current_worker].progress();
        if(last_chunk_panel>=0) {
          if(!progressed) throw std::runtime_error("valid chunk returned no progress");
          // Check on the final VALID progress itself: no later padding step,
          // exhausted-queue call or drain may be needed to flush a panel.
          int panel=last_chunk_panel;
          if(completed_chunks[panel]==valid_steps[panel] && ready[panel*kReadyFlagStride]!=7)
            throw std::runtime_error("last valid chunk did not publish ready immediately");
        }
        if(!progressed){active[slot]=active.back();active.pop_back();}
      }
      for(int value:seen) if(value!=1) throw std::runtime_error("missing group");
      for(int i=0;i<panels;++i) {
        if(ready[i*kReadyFlagStride]!=7 || release_calls[i]!=1) throw std::runtime_error("missing ready");
        if(atomic_calls[i]!=std::min(workers,valid_steps[i]) || delta_sum[i]!=uint32_t(valid_steps[i]) ||
            arrivals[i*kReadyFlagStride]!=uint32_t(valid_steps[i]))
          throw std::runtime_error("aggregated atomic count/delta sum mismatch");
      }
      for(uint32_t value:worker_pending) if(value) throw std::runtime_error("unflushed worker panel");
      // An exhausted queue and the prequantized diagnostic do no extra work.
      auto previous_calls=atomic_calls; auto previous_seen=seen;
      for(int i=0;i<workers;++i) {
        current_worker=i;
        if(work[i].progress()) throw std::runtime_error("exhausted queue progressed");
      }
      auto prequantized=a; prequantized.source=nullptr;
      Mxfp8WeightProducer no_work(prequantized,s,order,0,workers);
      if(no_work.progress() || atomic_calls!=previous_calls || seen!=previous_seen)
        throw std::runtime_error("empty worker changed output or published");
    }
  }
}
'''
        with tempfile.TemporaryDirectory(prefix='fuse-mxfp8-queue-') as directory:
            binary = Path(directory) / 'queue'
            result = subprocess.run([*compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                '-x', 'c++', '-', '-o', str(binary)], input=program, text=True,
                capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(binary)], text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_actual_mainloop_adapter_preserves_state_and_caches_panel_acquire(self):
        compiler = shlex.split(os.environ.get('CXX', 'c++'))
        if not compiler or not shutil.which(compiler[0]):
            self.skipTest('host C++ compiler required')
        source = (ROOT / 'csrc/operators/sm103/detail/cutlass_pipeline.cuh').read_text()
        begin = source.index('template <class Base>\nstruct WeightReadyMainloop')
        adapter = source[begin:source.index('\n};', begin) + len('\n};')]
        program = r'''
#include <cstdint>
#include <stdexcept>
#include <tuple>
#define CUTLASS_DEVICE
namespace cute { using std::get; }
namespace cutlass { struct KernelHardwareInfo {}; }
struct { int x=0; } threadIdx;
constexpr int kReadyFlagStride=32;
int acquires=0, fences=0;
void __syncwarp() {}
void fence_proxy_async_global() { ++fences; }
void wait_acquire_gpu_single_lane(const uint32_t* p,uint32_t target) {
  if(*p<target) throw std::runtime_error("unready panel");
  ++acquires;
}
struct Base {
  struct Arguments {}; struct Params {};
  struct DispatchPolicy { using ClusterShape=int; };
  using MainloopPipeline=int; using MainloopPipelineState=int;
  Base(const Params&,int,uint32_t) {}
  template<class Problem> static Params to_underlying_arguments(
      const Problem&,const Arguments&,void*,const cutlass::KernelHardwareInfo&) { return {}; }
  template<class Problem> static bool can_implement(const Problem&,const Arguments&) { return true; }
  template<class Input,class Tile,class Iterator>
  auto load(int,int state,const Input&,const Tile&,Iterator k,int count) {
    return std::make_tuple(state+count,k+count);
  }
};
''' + adapter + r'''
int main() {
  uint32_t flags[2*kReadyFlagStride]{}; flags[0]=flags[kReadyFlagStride]=7;
  using Main=WeightReadyMainloop<Base>;
  Main::Arguments args{}; args.weight_ready=flags; args.weight_panels=2; args.weight_epoch=7;
  if(!Main::can_implement(0,args)) return 1;
  auto p=Main::to_underlying_arguments(0,args,nullptr);
  Main main(p,1,0);
  auto tile=std::make_tuple(0,0,0,0);
  auto [state,k]=main.load(0,0,0,tile,0,3);
  auto [state2,k2]=main.load(0,state,0,tile,k,5);
  if(state2!=8 || k2!=8 || acquires!=1 || fences!=1) return 2;
  std::get<0>(tile)=1; main.load(0,state2,0,tile,0,8); // Another M reuses N.
  if(acquires!=1) return 3;
  std::get<1>(tile)=1; main.load(0,state2,0,tile,0,8);
  if(acquires!=2) return 4;
  std::get<1>(tile)=2; main.load(0,state2,0,tile,0,8); // Padded N has no flag.
  if(acquires!=2) return 5;
  args.weight_epoch=0; if(Main::can_implement(0,args)) return 6;
  args.weight_ready=nullptr; p=Main::to_underlying_arguments(0,args,nullptr);
  Main diagnostic(p,1,0); std::get<1>(tile)=0;
  diagnostic.load(0,0,0,tile,0,8); if(acquires!=2) return 7;
}
'''
        with tempfile.TemporaryDirectory(prefix='fuse-mxfp8-mainloop-') as directory:
            binary = Path(directory) / 'mainloop'
            result = subprocess.run([*compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror',
                '-x', 'c++', '-', '-o', str(binary)], input=program, text=True,
                capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(binary)], text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)


class Mxfp8ComponentSummaryContracts(unittest.TestCase):
    """Synthetic log audits reuse the real 10+50 parser; no GPU measurements."""

    def fixture(self):
        import summarize_sm103_mxfp8_fused as report
        from test_sm103_fused_summary import FusedSummaryTests, line
        job = dict(run_id='synthetic', experiment='unit', workspace='/home/work/workspace_wct',
                   stage='fused-smoke', node='09', mxfp8=True, mpi=True, calibrate=True,
                   fused_direction='qkv', fused_launch='graph', world=4, seq_local=256,
                   hidden=1024, q_heads=8, kv_heads=4, head_dim=128, comm_sm=8,
                   qkv_policy='m128n256', source_id='a' * 64,
                   files={'csrc/operators/sm103/detail/quantization.cuh': 'b' * 64})
        records = [line('precision', 'mxfp8', input='mxfp8', weight='bf16', output='bf16',
                        accumulator='fp32', scale='ue8m0', group_k=32, tile='128x256x128',
                        includes_activation_quantization=0, includes_weight_quantization=1,
                        weight_preparation='comm', epilogue_n=64),
                   line('config', world=4, global_seq=1024, seq_local=256, hidden=1024,
                        q_heads=8, kv_heads=4, head_dim=128, seed=20260906, warmup=10, samples=50,
                        input_generator='gpu_philox', sampling_mode='formal_10_50', profile=0,
                        launch='graph', process_layout='mpi_one_process_per_gpu', fused_direction='qkv',
                        calibrate=1, cpu_oracle=0, candidates=1, max_swizzle_size=1, qkv_raster='heuristic',
                        oproj_raster='heuristic', qkv_effective_raster='along_m', oproj_effective_raster='along_n')]
        helpers = FusedSummaryTests()
        for rank in range(4):
            records += [line('device', rank=rank, runtime_cc='10.3', sms=148),
                        helpers.input_row('QKV-weight', 1024 * 2048, 20260917, .02, rank=rank)]
        schedule = dict(tile_m=128, tile_n=256, tile_k=128, raster='along_m',
                        max_swizzle_size=1, effective_swizzle_size=1, padded_m_tiles=2, padded_n_tiles=8)
        for generation in range(2):
            for rank in range(4):
                records += [helpers.input_row('QKV-activation', 256 * 1024,
                            20260906 + generation * 100003 + rank * 101 + 1000, .125,
                            generation=generation, rank=rank),
                            helpers.input_row('MXFP8-weight', 1024 * 2048,
                            20260917 + generation * 100003, .02, generation=generation, rank=rank)]
            context = dict(candidate=1, comm_sm=8, tile='m128n256', generation=generation)
            for rank in range(4):
                records.append(line('candidate', 'GEMM_A2A', **context, component='fused', rank=rank,
                                    state='resolved', threads=256, dynamic_smem=187392,
                                    scheduled_compute_ctas=16, **schedule))
            for component in report.BOUNDARIES:
                scoped = context | {'component': component}
                kinds = ('correctness', 'route') if component == 'fused' else (
                    ('route',) if component == 'copy_reference' else ('correctness',))
                for rank in range(4):
                    records.append(line('component_resources', 'GEMM_A2A', **scoped, rank=rank, **schedule,
                        compute_budget=140, scheduled_compute_ctas=16 if component in ('fused', 'compute_reference') else 0,
                        scheduled_comm_ctas=0 if component == 'compute_reference' else 8,
                        production_threads=256, production_dynamic_smem=187392,
                        reference_resources='not_applicable' if component == 'fused' else 'unknown',
                        reference_precision='mxfp8', reference_weight_preparation=(
                            'inside_timing' if component == 'quantize_reference' else 'outside_timing'),
                        epilogue_n=64, reference_resource_contract='production_threads_and_dynamic_smem'))
                def checks(phase):
                    if component == 'quantize_reference':
                        records.append(line('quant_validation', 'GEMM_A2A', **scoped, validation_phase=phase,
                            method='represented_operands_gemm', prepare_repeated=0, compute_outside_timing=1))
                    for kind in kinds:
                        for rank in range(4):
                            numeric = dict(mismatches=0, max_abs=0, relative_l2=0, atol=.01, rtol=.01)
                            records.append(line(kind, 'GEMM_A2A', **scoped, rank=rank,
                                validation_phase=phase, validator='gpu_full', elements=256 * 2048,
                                checked=256 * 2048, nonfinite=0,
                                **(numeric if kind == 'correctness' else dict(bitwise_mismatches=0))))
                checks('pre')
                if generation == 0:
                    for text in helpers.timing_rows('GEMM_A2A', scoped):
                        text = text.replace('per_epoch_rank_events_v2', 'mpi_graph_rank_events_v1').replace(
                            'single_process_eager_maxrank_cudaevent', 'mpi_graph_maxrank_cudaevent')
                        if text.startswith('summary,'):
                            text += ',launch=graph,graph_epoch_mode=recapture_update_v1'
                        records.append(text)
                    checks('post')
                first, calls = (102 if component == 'fused' and generation == 1 else 1), (101 if generation == 0 else 1)
                for rank in range(4):
                    records.append(line('graph_prepare', 'GEMM_A2A', **scoped, rank=rank, launch='graph',
                        graph_epoch_mode='recapture_update_v1', calls=calls, first_epoch=first,
                        last_epoch=first + calls - 1, wall_s=.01, gpu_sample_time=0,
                        includes='capture_inspect_instantiate_initial_upload_sync_update'))
        for component in report.BOUNDARIES:
            records.append(line('candidate_verified', 'GEMM_A2A', candidate=1, comm_sm=8, tile='m128n256',
                component=component, payload_generations=2, full_numeric=int(component != 'copy_reference'),
                full_route=int(component in ('fused', 'copy_reference')), performance_accepted=1, launch='graph',
                graph_epoch_mode='recapture_update_v1'))
        records.append('PASS: selected BF16 boundaries, complete routes, changed payloads')
        return job, '\n'.join(records)

    def audit(self, component, change=lambda text: text):
        import summarize_sm103_mxfp8_fused as report
        job, log = self.fixture()
        receipts = {'status.json': {'attempt': 1}, 'fused-build.json': {'binary_sha256': 'c' * 64},
                    'environment.json': {'fingerprint': 'd' * 64}, 'gpu-before.json': {}}
        data = {'attempt1.log': change(log).encode(), 'gpu-telemetry.csv': b''}
        evidence = {'artifacts.tar.gz': {'sha256': 'e' * 64}}
        # Receipt/hash ownership has its own MPI fixture tests; here only that
        # boundary is stubbed. Input, timing, Graph, validation and resource
        # parsers all run unchanged against independently generated fake rows.
        with tempfile.TemporaryDirectory(prefix='fuse-mxfp8-component-') as folder:
            (Path(folder) / 'job.json').write_text(json.dumps(job))
            with mock.patch.object(report.sf, 'read_receipts', return_value=(job, receipts, data, evidence)), \
                 mock.patch.object(report.sf, 'audit_telemetry', return_value={}):
                return report.audit_run(folder, component=component)

    def test_fused_compute_copy_quantize_are_separate_verified_boundaries(self):
        results = {component: self.audit(component) for component in (
            'fused', 'compute_reference', 'copy_reference', 'quantize_reference')}
        self.assertEqual(len({r['boundary'] for r in results.values()}), 4)
        self.assertEqual(results['fused']['measurement_role'], 'production')
        for component, result in results.items():
            self.assertEqual(result['component'], component)
            self.assertEqual(len(result['raw_maxrank_ms']), 50)
            self.assertEqual(result['warmup_calls'], 50)
            self.assertEqual(len(result['component_resources']), 8)
        self.assertEqual(results['copy_reference']['executed_gemm_flops'], 0)
        self.assertIsNone(results['copy_reference']['pflops_per_rank'])
        self.assertGreater(results['compute_reference']['pflops_per_rank'], 0)
        self.assertEqual(results['compute_reference']['validation'], 'both_payloads_and_postmeasurement_full_numeric')
        self.assertEqual(results['copy_reference']['validation'], 'both_payloads_and_postmeasurement_full_route')
        self.assertIsNone(results['quantize_reference']['pflops_per_rank'])
        self.assertEqual(results['quantize_reference']['executed_gemm_flops'], 0)
        self.assertEqual(results['quantize_reference']['precision'], 'BF16_to_MXFP8_E4M3_UE8M0')

    def test_reference_requires_both_payloads_postcheck_and_exact_resources(self):
        for component in ('compute_reference', 'copy_reference', 'quantize_reference'):
            for mutation in ('payload', 'post', 'resources', 'raw_sample'):
                def corrupt(text):
                    changed = []
                    for row in text.splitlines():
                        target = f'component={component},' in row
                        if target and mutation == 'payload' and 'generation=1,' in row:
                            continue
                        if target and mutation == 'post' and ',validation_phase=post,' in row:
                            continue
                        if target and mutation == 'raw_sample' and row.startswith('sample,') and ',index=49,' in row:
                            continue
                        if target and mutation == 'resources' and row.startswith('component_resources,'):
                            row = row.replace('production_dynamic_smem=187392', 'production_dynamic_smem=187393')
                        changed.append(row)
                    return '\n'.join(changed)
                with self.subTest(component=component, mutation=mutation), self.assertRaises(ValueError):
                    self.audit(component, corrupt)

    def test_reference_rejects_wrong_precision_or_component(self):
        for before, after in (('reference_precision=mxfp8', 'reference_precision=bf16'),
                              ('reference_weight_preparation=outside_timing', 'reference_weight_preparation=inside_timing'),
                              ('accumulator=fp32', 'accumulator=bf16'), ('calibrate=1', 'calibrate=0')):
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.audit('compute_reference', lambda text: text.replace(before, after))
        with self.assertRaises(ValueError):
            self.audit('unknown')

    def test_quantize_reference_requires_no_overwrite_untimed_compute_validation(self):
        for before, after in (('prepare_repeated=0', 'prepare_repeated=1'),
                              ('compute_outside_timing=1', 'compute_outside_timing=0'),
                              ('method=represented_operands_gemm', 'method=unchecked'),
                              ('reference_weight_preparation=inside_timing', 'reference_weight_preparation=outside_timing')):
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.audit('quantize_reference', lambda text: text.replace(before, after))
        with self.assertRaises(ValueError):
            self.audit('quantize_reference', lambda text: '\n'.join(
                row for row in text.splitlines() if not row.startswith('quant_validation,')))

    def test_bf16_parser_does_not_implicitly_admit_quantization_boundary(self):
        import summarize_sm103_mxfp8_fused as report
        _, log = self.fixture()
        with self.assertRaisesRegex(ValueError, 'Unsupported measurement component'):
            report.sf.parse_log(log)
        rows, diagnostics = report.sf.parse_log(log, components=report.BOUNDARIES)
        self.assertEqual(len([r for r in rows if r['kind'] == 'quant_validation']), 3)
        self.assertFalse(diagnostics)
        self.assertEqual(report.sf.COMPONENTS, ('fused', 'compute_reference', 'copy_reference'))

    def test_aggregate_archive_keeps_samples_and_rejects_mixed_or_missing_boundaries(self):
        import summarize_sm103_mxfp8_fused as report
        results = [self.audit(component) for component in report.BOUNDARIES]
        summary = report.build_service_summary(results)
        self.assertEqual(summary['calibration_scope'], 'aggregate_only')
        self.assertEqual(len(summary['rows']), 1)
        with tempfile.TemporaryDirectory(prefix='fuse-service-summary-') as directory:
            paths = report.write_service_summary(directory, summary)
            restored = json.loads(paths[0].read_text())
            self.assertEqual(restored, summary)
            for row in restored['rows'][0]['components'].values():
                self.assertEqual(len(row['raw_maxrank_ms']), 50)
                self.assertEqual(row['artifact_sha256'], 'e' * 64)
            markdown = paths[1].read_text()
            self.assertIn('F μs | C μs | Q μs | R μs | F P | C P', markdown)
            self.assertIn('aggregate_only', markdown)
        for field in ('binary_sha256', 'environment_fingerprint'):
            corrupted = [dict(r) for r in results]
            corrupted[-1][field] = 'f' * 64
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'binary/environment'):
                report.build_service_summary(corrupted)
        with self.assertRaisesRegex(ValueError, 'Missing F/C/R/Q'):
            report.build_service_summary(results[:-1])
        with self.assertRaisesRegex(ValueError, 'Duplicate physical'):
            report.build_service_summary(results + results[:1])
        other_run = [dict(r) for r in results]
        other_run[-1]['run_id'] = 'another_run'
        with self.assertRaisesRegex(ValueError, 'same audited run'):
            report.build_service_summary(other_run)

    def test_archive_failure_does_not_partially_replace_previous_pair(self):
        import summarize_sm103_mxfp8_fused as report
        results = [self.audit(component) for component in report.BOUNDARIES]
        with tempfile.TemporaryDirectory(prefix='fuse-service-failure-') as directory:
            root = Path(directory)
            output = root / 'output'
            output.mkdir()
            paths = [output / name for name in ('services-current.json', 'services-current.md')]
            for path in paths:
                path.write_text('previous ' + path.suffix)
            previous = [path.read_bytes() for path in paths]
            job, _ = self.fixture()
            inputs = [root / 'run0', root / 'run1']
            for path in inputs:
                path.mkdir()
                (path / 'job.json').write_text(json.dumps(job))
            with mock.patch.object(report, 'audit_run', side_effect=results + [ValueError('last input rejected')]):
                with self.assertRaisesRegex(ValueError, 'last input rejected'):
                    report.archive_services(inputs, output)
            self.assertEqual([path.read_bytes() for path in paths], previous)
            replace = os.replace
            calls = []
            def fail_second(source, destination):
                calls.append(destination)
                if len(calls) == 2:
                    raise OSError('second replacement failed')
                return replace(source, destination)
            with mock.patch.object(report.os, 'replace', side_effect=fail_second):
                with self.assertRaisesRegex(OSError, 'second replacement failed'):
                    report.write_service_summary(output, report.build_service_summary(results))
            self.assertEqual([path.read_bytes() for path in paths], previous)
            self.assertEqual(sorted(p.name for p in output.iterdir()), sorted(p.name for p in paths))


class Mxfp8ServiceProbeContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shlex.split(os.environ.get('CXX', 'c++'))
        if not compiler or not shutil.which(compiler[0]):
            raise unittest.SkipTest('host C++ compiler required')
        cls.temporary = tempfile.TemporaryDirectory(prefix='fuse-mxfp8-service-')
        cls.addClassCleanup(cls.temporary.cleanup)
        folder = Path(cls.temporary.name)
        api = (ROOT / 'csrc/operators/sm103/api/forward_mxfp8.cuh').read_text()
        begin = api.index('template <class Schedule>\nstd::vector<uint8_t> mxfp8_recovery_panels')
        recovery = api[begin:api.index('\n}', begin) + 2]
        order = (ROOT / 'csrc/operators/sm103/detail/producer_consumer.cuh').read_text()
        begin = order.index('struct NBandSwizzle')
        end = order.index('\n};', order.index('struct ProducerTileOrder', begin)) + len('\n};')
        order = order[begin:end]
        program = r'''
#include <cassert>
#include <cstdint>
#include <string>
#include <vector>
#define CUTLASS_HOST_DEVICE
namespace fuse::detail {
''' + order + r'''
}
namespace fuse {
''' + recovery + r'''
}
struct Divmod {
  uint64_t divisor = 1;
  void operator()(uint64_t& q, uint64_t& r, uint64_t x) const { q=x/divisor; r=x%divisor; }
};
struct Schedule {
  enum class RasterOrder { AlongM, AlongN };
  RasterOrder raster_order_ = RasterOrder::AlongM;
  int log_swizzle_size_ = 3;
  uint64_t blocks_per_problem_ = 0, compute_grid_size = 0;
  Divmod divmod_batch_, divmod_cluster_blk_major_;
  fuse::detail::NBandSwizzle n_band_swizzle{};
};
Schedule make(int m, int n, bool along_n, int g, int sw=8) {
  Schedule p;
  p.raster_order_=along_n?Schedule::RasterOrder::AlongN:Schedule::RasterOrder::AlongM;
  p.log_swizzle_size_=0; while ((1<<p.log_swizzle_size_)<sw) ++p.log_swizzle_size_;
  const int major=along_n?n:m, minor=along_n?m:n;
  p.divmod_cluster_blk_major_.divisor=major;
  p.blocks_per_problem_=major*((minor+sw-1)/sw*sw);
  p.divmod_batch_.divisor=p.blocks_per_problem_; p.compute_grid_size=g;
  return p;
}
int main(int argc, char** argv) {
  assert(argc==2); const std::string mode=argv[1];
  if (mode=="along_m") {
    auto r=fuse::mxfp8_recovery_panels(make(128,16,false,132),128,16);
    for (int n=0;n<8;++n) assert(r[n]&1);
    for (int n=8;n<16;++n) assert(r[n]==2);
  } else if (mode=="along_n") {
    auto r=fuse::mxfp8_recovery_panels(make(128,16,true,132),128,16);
    // Every panel has a first-wave consumer, but also an independently
    // eligible CTA that computes a different panel before reaching it.
    for (auto state:r) assert(state==3);
  } else if (mode=="no_recovery") {
    auto r=fuse::mxfp8_recovery_panels(make(128,1,false,8,1),128,1);
    assert(r.size()==1 && r[0]==1);
    r=fuse::mxfp8_recovery_panels(make(128,16,false,0),128,16);
    for (auto state:r) assert(state==0);
  } else if (mode=="padding") {
    for (bool along_n:{false,true}) {
      auto p=make(17,13,along_n,7);
      auto r=fuse::mxfp8_recovery_panels(p,17,13);
      std::vector<uint8_t> expected(13,0);
      for (uint64_t worker=0;worker<p.compute_grid_size;++worker) {
        int first=-1;
        for (uint64_t i=worker;i<p.blocks_per_problem_;i+=p.compute_grid_size) {
          auto tile=fuse::detail::ProducerTileOrder::decode(p,i);
          if (tile.m>=17 || tile.n>=13) continue;
          if(first<0) { first=tile.n; expected[first]|=1; }
          else if(tile.n!=first) expected[tile.n]|=2;
        }
      }
      assert(r==expected && r.size()==13);
    }
  } else return 2;
}
'''
        source = folder / 'recovery.cpp'
        source.write_text(program)
        cls.probe = folder / 'recovery'
        result = subprocess.run([*compiler, '-std=c++17', str(source), '-o', str(cls.probe)],
                                capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stderr)

    def test_delayed_panel_has_real_steady_state_consumers(self):
        for mode in ('along_m', 'along_n', 'no_recovery', 'padding'):
            with self.subTest(mode=mode):
                subprocess.run([str(self.probe), mode], check=True)

    def test_services_preserve_quant_queue_and_seed_real_route_flags(self):
        source = (ROOT / 'csrc/operators/sm103/detail/persistent_gemm.cuh').read_text()
        begin = source.index('struct InputCopyReferenceKernel')
        body = source[begin:source.index('\n};', begin)]
        self.assertEqual(body.count('InputProducer work('), 1)
        self.assertIn('InputProducer::initialize_grid(p.comm)', body)
        self.assertIn('if (p.quant_phase_steps == 1) work.progress()', body)
        self.assertIn('p.service.routes, work)', body)
        self.assertIn('nullptr, work)', body)
        self.assertIn('ready[tile * kReadyFlagStride] = p.comm.params.epoch', body)
        self.assertIn('CommOp{}.finalize(p.comm)', body)
        self.assertIn('if (p.service.routes)', body)

    def test_diagnostic_tile_writers_reject_padded_aliases(self):
        source = (ROOT / 'csrc/operators/sm103/detail/cutlass_pipeline.cuh').read_text()
        for name, guard in (('Mxfp8ServiceMainloop', 'n < params_->weight_panels'),
                            ('Mxfp8ServiceEpilogue', 'n < params_->n_tiles')):
            start = source.index(f'struct {name}')
            body = source[start:source.index('\n};', start)]
            self.assertIn(guard, body)
            self.assertNotIn('store_release_gpu(', body)
            self.assertNotIn('__syncwarp(', body)


class Mxfp8AutoSummaryContracts(unittest.TestCase):
    def fixture(self):
        job = dict(auto_mxfp8_comm=True, comm_sm_list='16,32', world=4,
                   qkv_policy='m128n256', fused_direction='qkv')
        config = dict(auto_mxfp8_comm='1')
        rows = []
        for rank in range(4):
            rows.append(dict(kind='auto_comm', label='GEMM_A2A', candidate='3',
                comm_sm='32', tile='m128n256', rank=str(rank), line=rank + 1,
                model_version='measured_services_v1', requested_comm='0', resolved_comm='32',
                launch_comm='0', query_us='123.0', repeat_query_us='0.1'))
            for generation in range(2):
                rows.append(dict(kind='candidate', candidate='3', rank=str(rank),
                                 line=10 + rank + generation * 4))
        return rows, job, config

    def test_paired_manual_and_auto_preserve_requested_zero_and_resolved_budget(self):
        import summarize_sm103_mxfp8_fused as report
        expected, metadata = report.audit_auto_selection(*self.fixture())
        self.assertEqual(expected, [(16, 'm128n256'), (32, 'm128n256'), (32, 'm128n256')])
        self.assertEqual(set(metadata), {3})
        self.assertEqual(metadata[3]['launch_comm_ctas'], 0)
        self.assertEqual(metadata[3]['resolved_comm_ctas'], 32)
        self.assertEqual(len(metadata[3]['rank_queries']), 4)

    def test_auto_requires_all_rank_agreement_and_native_prelaunch_evidence(self):
        import summarize_sm103_mxfp8_fused as report
        for field, value in (('rank', '1'), ('candidate', '1'), ('comm_sm', '16'),
                             ('resolved_comm', '16'), ('launch_comm', '32'),
                             ('requested_comm', '32'), ('query_us', 'nan'),
                             ('repeat_query_us', '-1'), ('model_version', 'empty_v1'),
                             ('model_version', 'different_version'), ('line', 100)):
            rows, job, config = self.fixture()
            rows[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                report.audit_auto_selection(rows, job, config)
        rows, job, config = self.fixture()
        with self.assertRaisesRegex(ValueError, 'Missing MXFP8 auto rank'):
            report.audit_auto_selection(rows[1:], job, config)

    def test_explicit_does_not_accept_stray_auto_records(self):
        import summarize_sm103_mxfp8_fused as report
        rows, job, config = self.fixture()
        job['auto_mxfp8_comm'], config['auto_mxfp8_comm'] = False, '0'
        with self.assertRaisesRegex(ValueError, 'Unexpected MXFP8 auto evidence'):
            report.audit_auto_selection(rows, job, config)
        expected, metadata = report.audit_auto_selection([], job, config)
        self.assertEqual(expected, [(16, 'm128n256'), (32, 'm128n256')])
        self.assertFalse(metadata)

    def test_candidate_order_is_tile_then_communication(self):
        import summarize_sm103_mxfp8_fused as report
        _, job, _ = self.fixture()
        job.update(auto_mxfp8_comm=False, qkv_policy_list='m128n128,m128n256')
        expected, _ = report.audit_auto_selection([], job, {})
        self.assertEqual(expected, [(16, 'm128n128'), (32, 'm128n128'),
                                    (16, 'm128n256'), (32, 'm128n256')])


if __name__ == '__main__':
    unittest.main()
