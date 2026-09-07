"""Narrow CPU-only CUTLASS plan ABI/lifetime tests; not CUDA compilation."""
import ctypes as ct
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / 'benchmarks/sm103/GEMM/cutlass.py'


class Function:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return self.result(*args) if callable(self.result) else self.result


class Tensor:
    def __init__(self, shape, pointer, *, dtype='bf16', device=0, contiguous=True, cuda=True):
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = types.SimpleNamespace(type='cuda', index=device)
        self.is_cuda = cuda
        self.pointer = pointer
        self.contiguous = contiguous

    def is_contiguous(self):
        return self.contiguous

    def data_ptr(self):
        return self.pointer


class CutlassPlanContracts(unittest.TestCase):
    def setUp(self):
        cuda = types.SimpleNamespace(current_device=lambda: 0,
                                     current_stream=lambda: types.SimpleNamespace(cuda_stream=99))
        self.torch = types.SimpleNamespace(bfloat16='bf16', cuda=cuda)
        spec = importlib.util.spec_from_file_location('isolated_cutlass_plan', ENTRY)
        self.module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {'torch': self.torch}):
            spec.loader.exec_module(self.module)
        self.native = types.SimpleNamespace(
            sm103_cutlass_create_v5=Function(123), sm103_cutlass_run=Function(0),
            sm103_cutlass_plan_info=Function(), sm103_cutlass_destroy=Function(),
            sm103_cutlass_last_error=Function(b'expected native failure'))
        with mock.patch.object(self.module.ct, 'CDLL', return_value=self.native) as load:
            self.lib = self.module.Library('/test/libcutlass.so')
        load.assert_called_once_with('/test/libcutlass.so')
        self.x = types.SimpleNamespace(precision='bf16', rows=256, k=64,
                                       data=Tensor((256, 64), 0x10000))
        self.w = types.SimpleNamespace(precision='bf16', rows=256, k=64,
                                       data=Tensor((256, 64), 0x20000))
        self.y = Tensor((256, 256), 0x30000)
        self.info = {'schema': 'sm103_cutlass_bf16_plan_v1', 'sm_mode': 1,
                     'backend': 'cutlass_1sm', 'm': 256, 'n': 256, 'k': 64,
                     'cluster_m': 1, 'cluster': [1, 1, 1], 'cluster_size': 1, 'mma_sm_count': 1,
                     'grid': [2, 1, 1], 'grid_ctas': 2, 'physical_sm_count': 148, 'requested_sm_budget': 148, 'active_clusters': None,
                     'max_swizzle_size': 1, 'effective_swizzle_size': 1,
                     'epilogue_n': 32, 'epilogue_tile': [128, 32],
                     'padded_work_grid_ctas': [2, 1, 1], 'padded_work_ctas': 2}
        self.native.sm103_cutlass_plan_info.result = lambda handle: json.dumps(self.info).encode()

    def plan(self, mode=1):
        return self.module.Plan(self.lib, self.x, self.w, self.y, sm_mode=mode)

    def test_explicit_sm_budget_is_forwarded_and_grid_enforced(self):
        self.info['requested_sm_budget'] = 132
        plan = self.module.Plan(self.lib, self.x, self.w, self.y, sm_budget=132)
        self.assertEqual(self.native.sm103_cutlass_create_v5.calls[-1][4], 132)
        plan.close()
        for value in (0, 133, 149):
            self.info['requested_sm_budget'] = value
            with self.assertRaises(ValueError):
                self.module.Plan(self.lib, self.x, self.w, self.y, sm_budget=132)
        self.info['requested_sm_budget'] = 132
        self.info['grid'] = [133, 1, 1]
        self.info['grid_ctas'] = 133
        with self.assertRaises(ValueError):
            self.module.Plan(self.lib, self.x, self.w, self.y, sm_budget=132)

    def test_bad_sm_budget_rejected_before_native_create(self):
        for value in (-1, True, 1.5, 2**31):
            with self.assertRaises(ValueError):
                self.module.Plan(self.lib, self.x, self.w, self.y, sm_budget=value)
        self.assertFalse(self.native.sm103_cutlass_create_v5.calls)

    def test_ffi_widths_and_result_types(self):
        self.assertEqual(self.native.sm103_cutlass_create_v5.argtypes,
                         [ct.c_int] * 5 + [ct.c_int64] * 3 + [ct.c_void_p] * 4)
        self.assertIs(self.native.sm103_cutlass_create_v5.restype, ct.c_void_p)
        self.assertEqual(self.native.sm103_cutlass_run.argtypes, [ct.c_void_p] * 2)
        self.assertIs(self.native.sm103_cutlass_run.restype, ct.c_int)
        self.assertIs(self.native.sm103_cutlass_plan_info.restype, ct.c_char_p)
        self.assertIsNone(self.native.sm103_cutlass_destroy.restype)

    def test_default_library_path_matches_isolated_controller_build(self):
        expected = ROOT / 'build/sm103-cutlass/libfuse_sm103_cutlass_bf16.so'
        with mock.patch.object(self.module.ct, 'CDLL', return_value=self.native) as load:
            library = self.module.Library()
        self.assertEqual(library.path, expected)
        load.assert_called_once_with(str(expected))

    def test_resource_stages_use_the_public_actual_dispatch_policy(self):
        source = (ROOT / 'csrc/baselines/sm103/cutlass_bf16.cu').read_text()
        for name in ('StagesC', 'StagesD'):
            self.assertIn(f'Epilogue::DispatchPolicy::{name}', source)
            self.assertNotIn(f'Epilogue::{name}', source)
        self.assertIn('epilogue_c_stages', source)
        self.assertIn('epilogue_d_stages', source)
        for dimension in (0, 1):
            self.assertIn(f'int(cute::size<{dimension}>(typename Epilogue::EpilogueTile{{}}))', source)

    def test_invalid_json_retains_bounded_escaped_raw_and_releases_plan(self):
        malformed = b'{"physical_cta_tile":[_128,_256,_64],"rest":"' + b'x' * 5000
        self.native.sm103_cutlass_plan_info.result = malformed
        with self.assertRaisesRegex(ValueError, 'Invalid CUTLASS plan JSON') as context:
            self.plan()
        message = str(context.exception)
        self.assertIn('[_128,_256,_64]', message)
        self.assertIn(f'raw_bytes={len(malformed)}', message)
        self.assertLess(len(message), 1200)
        self.assertEqual(self.native.sm103_cutlass_destroy.calls, [(123,)])

    def test_actual_cute_host_stream_format_and_native_integer_casts(self):
        cutlass = os.environ.get('CUTLASS_ROOT')
        if not cutlass:
            self.skipTest('set CUTLASS_ROOT for the real CuTe host-compiler regression')
        compiler = shlex.split(os.environ.get('CXX', 'c++'))
        if not compiler or shutil.which(compiler[0]) is None:
            self.skipTest('a host C++ compiler is required')
        source = (ROOT / 'csrc/baselines/sm103/cutlass_bf16.cu').read_text()
        expressions = re.findall(r'<< (int\(cute::size<([012])>\(typename Mainloop::CtaShape_MNK\{\}\)\))', source)
        self.assertEqual([dimension for _, dimension in expressions], ['0', '1', '2'])
        metadata = source.split('std::ostringstream text;')[1].split('info = text.str();')[0]
        self.assertNotIn('<< cute::', metadata)
        # Compile CuTe's actual constant and ostream templates. The full header
        # pulls in CUDA/CCCL unavailable on Mac; these exact source declarations
        # have no CUDA dependency. Do not mock the stream overload responsible
        # for the bug. A second shape rules out hardcoded replacement values.
        header = (Path(cutlass) / 'include/cute/numeric/integral_constant.hpp').read_text()
        start = header.index('template <auto v>\nstruct C {')
        constant = header[start:header.index('\n};', start) + 3]
        start = header.index('template <auto t>\nCUTE_HOST std::ostream& operator<<')
        stream_operator = header[start:header.index('\n}', start) + 2]
        probe = r'''
#include <iostream>
#define CUTE_HOST_DEVICE inline
#define CUTE_HOST inline
namespace cute {
''' + constant + '\n' + stream_operator + r'''
}
template<int M, int N, int K> void emit() {
  std::cout << "[" << cute::C<M>{} << "," << cute::C<N>{} << "," << cute::C<K>{} << "]\n";
  std::cout << "[" << int(cute::C<M>{}) << "," << int(cute::C<N>{}) << ","
            << int(cute::C<K>{}) << "]\n";
}
int main() { emit<128,256,64>(); emit<17,29,31>(); }
'''
        with tempfile.TemporaryDirectory(prefix='fuse-cute-json-test-') as directory:
            binary = Path(directory) / 'probe'
            result = subprocess.run([*compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror',
                                     '-x', 'c++', '-', '-o', str(binary)],
                                    input=probe, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(binary)], text=True, capture_output=True, timeout=10, check=True)
        lines = result.stdout.splitlines()
        self.assertEqual(lines[0], '[_128,_256,_64]')
        self.assertEqual(lines[2], '[_17,_29,_31]')
        self.assertEqual(json.loads(lines[1]), [128, 256, 64])
        self.assertEqual(json.loads(lines[3]), [17, 29, 31])

    def test_create_binds_buffers_once_run_only_calls_prepared_plan(self):
        plan = self.plan()
        self.assertEqual(self.native.sm103_cutlass_create_v5.calls,
                         [(1, 1, 32, 1, 0, 256, 256, 64, 0x10000, 0x20000, 0x30000, 99)])
        self.assertIs(plan.x, self.x)
        self.assertIs(plan.weight, self.w)
        self.assertEqual(plan.info, self.info)
        for _ in range(3):
            self.assertIs(plan.run(), self.y)
        self.assertEqual(self.native.sm103_cutlass_run.calls, [(123, 99)] * 3)
        self.assertEqual(len(self.native.sm103_cutlass_create_v5.calls), 1)
        self.assertEqual(self.native.sm103_cutlass_plan_info.calls, [(123,)])

    def test_both_modes_and_invalid_mode_do_not_silently_fallback(self):
        self.info.update(sm_mode=2, backend='cutlass_2sm', cluster_m=2, cluster=[2, 1, 1],
                         cluster_size=2, mma_sm_count=2, active_clusters=74)
        plan = self.plan(2)
        self.assertEqual(plan.info['sm_mode'], 2)
        plan.close()
        for mode in (0, 3, True, 1.0, '2'):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.plan(mode)
        self.assertEqual(len(self.native.sm103_cutlass_create_v5.calls), 1)

    def test_swizzle_is_explicit_validated_and_can_be_lowered_for_small_shapes(self):
        for requested, effective in ((1, 1), (2, 1), (4, 2), (8, 4)):
            self.info.update(max_swizzle_size=requested, effective_swizzle_size=effective,
                             padded_work_grid_ctas=[effective * 2, effective, 1],
                             padded_work_ctas=effective * effective * 2)
            plan = self.module.Plan(self.lib, self.x, self.w, self.y, max_swizzle_size=requested)
            self.assertEqual(self.native.sm103_cutlass_create_v5.calls[-1][:2], (1, requested))
            plan.close()
        for requested in (0, 3, 16, True, 4.0, '4'):
            with self.subTest(requested=requested), self.assertRaisesRegex(ValueError, 'max_swizzle_size'):
                self.module.Plan(self.lib, self.x, self.w, self.y, max_swizzle_size=requested)
        self.assertEqual(len(self.native.sm103_cutlass_create_v5.calls), 4)

    def test_inconsistent_lowered_grid_or_swizzle_releases_plan(self):
        for change in ({'max_swizzle_size': 2}, {'effective_swizzle_size': 2},
                       {'effective_swizzle_size': True}, {'padded_work_ctas': 3},
                       {'padded_work_grid_ctas': [2, 0, 1]}, {'padded_work_grid_ctas': [2, 1]}):
            original = self.info.copy()
            self.info.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.plan()
            self.info = original
        self.assertEqual(len(self.native.sm103_cutlass_destroy.calls), 6)

    def test_cluster2_retains_one_sm_mma_and_explicit_resources(self):
        self.info.update(cluster_m=2, cluster=[2, 1, 1], cluster_size=2, active_clusters=66)
        plan = self.module.Plan(self.lib, self.x, self.w, self.y, cluster_m=2)
        self.assertEqual(self.native.sm103_cutlass_create_v5.calls[-1][:4], (1, 1, 32, 2))
        self.assertEqual(plan.info['mma_sm_count'], 1)
        self.assertEqual(plan.info['backend'], 'cutlass_1sm')
        plan.close()
        for changes in ({'cluster_m': 0}, {'cluster_m': 4}, {'cluster_m': True},
                        {'cluster_m': 2.0}, {'cluster_m': '2'},
                        {'sm_mode': 2, 'cluster_m': 1}, {'cluster_m': 2, 'epilogue_n': 64}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.module.Plan(self.lib, self.x, self.w, self.y, **changes)
        self.assertEqual(len(self.native.sm103_cutlass_create_v5.calls), 1)

    def test_cluster2_rejects_wrong_scope_odd_grid_and_unknown_residency(self):
        self.info.update(cluster_m=2, cluster=[2, 1, 1], cluster_size=2, active_clusters=66)
        changes = ({'cluster_m': 1}, {'cluster': [1, 1, 1]}, {'cluster_size': 1},
                   {'mma_sm_count': 2}, {'active_clusters': None}, {'active_clusters': 0},
                   {'active_clusters': True}, {'grid': [3, 1, 1], 'grid_ctas': 3},
                   {'grid': [134, 1, 1], 'grid_ctas': 134},
                   {'padded_work_grid_ctas': [3, 1, 1], 'padded_work_ctas': 3})
        for change in changes:
            original = self.info.copy()
            self.info.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.module.Plan(self.lib, self.x, self.w, self.y, cluster_m=2)
            self.info = original
        self.assertEqual(len(self.native.sm103_cutlass_destroy.calls), len(changes))
        self.assertFalse(self.native.sm103_cutlass_run.calls)

    def test_stale_library_fails_at_versioned_symbol_before_create(self):
        del self.native.sm103_cutlass_create_v5
        self.native.sm103_cutlass_create = Function(123)
        self.native.sm103_cutlass_create_v2 = Function(123)
        self.native.sm103_cutlass_create_v3 = Function(123)
        with mock.patch.object(self.module.ct, 'CDLL', return_value=self.native), self.assertRaises(AttributeError):
            self.module.Library('/test/old.so')
        self.assertFalse(self.native.sm103_cutlass_create.calls)
        self.assertFalse(self.native.sm103_cutlass_create_v2.calls)
        self.assertFalse(self.native.sm103_cutlass_create_v3.calls)

    def test_epilogue_n_is_validated_and_bound_for_both_sm_modes(self):
        for mode in (1, 2):
            for epilogue_n in (32, 64):
                self.info.update(sm_mode=mode, backend=f'cutlass_{mode}sm',
                                 cluster_m=mode, cluster=[mode, 1, 1], cluster_size=mode,
                                 mma_sm_count=mode, active_clusters=74 if mode == 2 else None,
                                 epilogue_n=epilogue_n, epilogue_tile=[128, epilogue_n])
                plan = self.module.Plan(self.lib, self.x, self.w, self.y,
                                        sm_mode=mode, epilogue_n=epilogue_n)
                self.assertEqual(self.native.sm103_cutlass_create_v5.calls[-1][:3],
                                 (mode, 1, epilogue_n))
                plan.close()
        for epilogue_n in (0, 16, 128, True, 32.0, '64', None):
            with self.subTest(epilogue_n=epilogue_n), self.assertRaisesRegex(ValueError, 'epilogue_n'):
                self.module.Plan(self.lib, self.x, self.w, self.y, epilogue_n=epilogue_n)
        self.assertEqual(len(self.native.sm103_cutlass_create_v5.calls), 4)

    def test_mismatched_epilogue_metadata_releases_plan_without_launch(self):
        for change in ({'epilogue_n': 64}, {'epilogue_tile': [128, 64]},
                       {'epilogue_tile': [64, 32]}, {'epilogue_tile': None}):
            original = self.info.copy()
            self.info.update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'metadata'):
                self.plan()
            self.info = original
        self.assertEqual(self.native.sm103_cutlass_destroy.calls, [(123,)] * 4)
        self.assertFalse(self.native.sm103_cutlass_run.calls)

    def test_native_five_way_dispatch_with_actual_source_on_host(self):
        compiler = shlex.split(os.environ.get('CXX', 'c++'))
        if not compiler or shutil.which(compiler[0]) is None:
            self.skipTest('a host C++ compiler is required')
        native = (ROOT / 'csrc/baselines/sm103/cutlass_bf16.cu').read_text()
        create = native.split('extern "C" void* sm103_cutlass_create_v5(', 1)[1]
        self.assertIn('int sm_mode, int max_swizzle_size, int epilogue_n, int cluster_m,', create)
        self.assertLess(create.index('require(epilogue_n == 32 || epilogue_n == 64,'),
                        create.index('check(cudaGetDevice(&device))'))
        dispatch = create[create.index('    if (sm_mode == 1) {'):create.index('  } catch')]
        # Compile the actual dispatch, substituting only the CUDA-dependent
        # constructors. This proves ABI argument forwarding/type selection,
        # not CUDA template legality or measured resource counts.
        probe = r'''
#include <cstdint>
#include <iostream>
struct Plan { int mode, epi, cluster, m, n, k, swizzle; virtual ~Plan() = default; };
template<int Mode, int Epi, int ClusterM=Mode> struct TypedPlan : Plan {
 TypedPlan(int m_,int n_,int k_,int swizzle_,const void*,const void*,void*,void*,int,int,int) {
  mode=Mode;epi=Epi;cluster=ClusterM;m=m_;n=n_;k=k_;swizzle=swizzle_;
 }
};
void* create(int sm_mode,int epilogue_n,int max_swizzle_size,int cluster_m) {
 int64_t m=257,n=264,k=520; const void *lhs=nullptr,*rhs=nullptr;
 void *output=nullptr,*stream=nullptr; int device=0,properties=0,sm_budget=0;
''' + dispatch + r'''
}
int main() {
 for(int mode:{1,2}) for(int cluster:{1,2}) for(int epi:{32,64}) for(int swizzle:{1,2,4,8}) {
  if((mode==2&&cluster!=2)||(mode==1&&cluster==2&&epi!=32))continue;
  auto* plan=static_cast<Plan*>(create(mode,epi,swizzle,cluster));
  if(plan->mode!=mode||plan->epi!=epi||plan->swizzle!=swizzle||
     plan->cluster!=cluster||plan->m!=257||plan->n!=264||plan->k!=520)return 1;
  delete plan;
 }
 std::cout<<"20 actual dispatch cases passed\n";
}
'''
        with tempfile.TemporaryDirectory(prefix='fuse-cutlass-epilogue-test-') as directory:
            binary = Path(directory) / 'probe'
            built = subprocess.run([*compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror',
                                    '-x', 'c++', '-', '-o', str(binary)], input=probe,
                                   text=True, capture_output=True, timeout=30)
            self.assertEqual(built.returncode, 0, built.stderr)
            run = subprocess.run([str(binary)], text=True, capture_output=True, check=True, timeout=10)
        self.assertEqual(run.stdout.strip(), '20 actual dispatch cases passed')

    def test_actual_cutlass_host_swizzle_and_cluster_padding(self):
        cutlass = os.environ.get('CUTLASS_ROOT')
        if not cutlass:
            self.skipTest('set CUTLASS_ROOT for the actual scheduler host regression')
        compiler = shlex.split(os.environ.get('CXX', 'c++'))
        if not compiler or shutil.which(compiler[0]) is None:
            self.skipTest('a host C++ compiler is required')
        kernel_dir = Path(cutlass) / 'include/cutlass/gemm/kernel'
        params = (kernel_dir / 'tile_scheduler_params.h').read_text()
        params = params.split('struct PersistentTileSchedulerSm90Params {', 1)[1].split('\n};', 1)[0]
        static = (kernel_dir / 'sm100_static_tile_scheduler.hpp').read_text()
        self.assertIn('using Params = PersistentTileSchedulerSm90Params;', static)
        self.assertIn('arguments.max_swizzle_size', static)

        def method(marker, signature):
            start = params.index(marker)
            brace = params.index('{', start)
            depth, end = 1, brace + 1
            while depth:
                depth += (params[end] == '{') - (params[end] == '}')
                end += 1
            return signature + params[brace:end]

        # Compile the pinned initialize/swizzle/raster methods themselves, not
        # a Python reimplementation. Only CUDA-free scalar types/divmod storage
        # are stood in; their device division operations are never invoked here.
        initialize = method('initialize(\n    dim3 problem_blocks,',
            'void initialize(dim3 problem_blocks,GemmCoord cluster_shape,KernelHardwareInfo const& hw_info,int max_swizzle_size,RasterOrderOptions raster_order_option) ')
        # The first get_log_swizzle_size occurrence is a call inside initialize;
        # select its definition explicitly for the real body.
        definition = params.index('get_log_swizzle_size(int problem_ctas_m')
        saved = params
        params = params[definition:]
        swizzle = method('get_log_swizzle_size(',
            'static int get_log_swizzle_size(int problem_ctas_m,int problem_ctas_n,int max_swizzle_size) ')
        params = saved[saved.index('get_rasterization_order(\n    uint32_t tiles_m,'):]
        raster = method('get_rasterization_order(',
            'static RasterOrder get_rasterization_order(uint32_t tiles_m,uint32_t tiles_n,RasterOrderOptions raster_order_option) ')
        native = (ROOT / 'csrc/baselines/sm103/cutlass_bf16.cu').read_text()
        self.assertIn('args.scheduler.max_swizzle_size = max_swizzle_size;', native)
        self.assertIn('extern "C" void* sm103_cutlass_create_v5(', native)
        self.assertNotIn('extern "C" void* sm103_cutlass_create_v2(', native)
        self.assertNotIn('extern "C" void* sm103_cutlass_create(', native)
        self.assertIn('const auto& scheduler = gemm.params().scheduler;', native)
        assignments = native.split('const int effective_swizzle_size =', 1)[1].split('    require(', 1)[0]
        assignments = 'const int effective_swizzle_size =' + assignments
        self.assertIn('scheduler.problem_tiles_m_) * scheduler.cluster_shape_m_', assignments)
        self.assertIn('scheduler.problem_tiles_n_) * scheduler.cluster_shape_n_', assignments)
        probe = r'''
#include <algorithm>
#include <cstdint>
#include <iostream>
#define CUTLASS_UNUSED(x) (void)(x)
namespace platform { using std::min; }
struct dim3 { uint32_t x,y,z; };
struct GemmCoord { int x,y; int m() const{return x;} int n() const{return y;} };
struct KernelHardwareInfo {};
struct Div { explicit Div(uint64_t=1) {} };
using FastDivmodU64=Div; using FastDivmodU64Pow2=Div;
template<class A,class B> auto round_up(A x,B y) { return ((x+y-1)/y)*y; }
enum class RasterOrder { AlongM,AlongN };
enum class RasterOrderOptions { Heuristic,AlongM,AlongN };
struct Params {
uint64_t blocks_per_problem_=0;
int log_swizzle_size_=0;
RasterOrder raster_order_{};
uint32_t problem_tiles_m_=0,problem_tiles_n_=0,problem_tiles_l_=0,cluster_shape_m_=0,cluster_shape_n_=0;
Div divmod_batch_,divmod_cluster_shape_major_,divmod_cluster_shape_minor_,divmod_cluster_blk_major_;
''' + initialize + '\n' + swizzle + '\n' + raster + r'''
};
void emit(int mode,int cluster_m,int cta_m,int cta_n,int requested) {
 Params scheduler;
 scheduler.initialize({uint32_t(cta_m),uint32_t(cta_n),1},{cluster_m,1},{},requested,RasterOrderOptions::AlongM);
''' + assignments + r'''
 std::cout<<mode<<' '<<cluster_m<<' '<<cta_m<<' '<<cta_n<<' '<<requested<<' '<<effective_swizzle_size<<' '
          <<padded_ctas_m<<' '<<padded_ctas_n<<' '<<scheduler.blocks_per_problem_<<'\n';
}
int main() {
 for(int mode:{1,2}) for(int cluster:{1,2}) for(int requested:{1,2,4,8}) {
  if(mode==2&&cluster!=2)continue;
  emit(mode,cluster,1,1,requested); // tiny/odd M work needs a complete cluster
  emit(mode,cluster,5,6,requested); // odd M tile, including 1-SM+cluster2
  emit(mode,cluster,128,72,requested); // 405B QKV physical work-grid dimensions
 }
}
'''
        with tempfile.TemporaryDirectory(prefix='fuse-cutlass-swizzle-test-') as directory:
            binary = Path(directory) / 'probe'
            built = subprocess.run([*compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror',
                                    '-x', 'c++', '-', '-o', str(binary)], input=probe,
                                   text=True, capture_output=True, timeout=30)
            self.assertEqual(built.returncode, 0, built.stderr)
            run = subprocess.run([str(binary)], text=True, capture_output=True, check=True, timeout=10)
        rows = [tuple(map(int, line.split())) for line in run.stdout.splitlines()]
        self.assertEqual(len(rows), 36)
        for mode, cluster, m, n, requested, effective, pm, pn, total in rows:
            expected = (8 if requested >= 8 and min(m, n) >= 6 else
                        4 if requested >= 4 and min(m, n) >= 3 else
                        2 if requested >= 2 and min(m, n) >= 2 else 1)
            self.assertEqual(effective, expected)
            self.assertEqual(pm, ((m + effective * cluster - 1) // (effective * cluster)) * effective * cluster)
            self.assertEqual(pn, ((n + effective - 1) // effective) * effective)
            self.assertEqual(total, pm * pn)
        self.assertIn((1, 1, 5, 6, 8, 4, 8, 8, 64), rows)
        self.assertIn((1, 2, 5, 6, 8, 4, 8, 8, 64), rows)
        self.assertIn((1, 2, 1, 1, 1, 1, 2, 1, 2), rows)

    def test_native_actual_cluster_occupancy_and_grid_branch_on_host(self):
        compiler = shlex.split(os.environ.get('CXX', 'c++'))
        if not compiler or shutil.which(compiler[0]) is None:
            self.skipTest('a host C++ compiler is required')
        source = (ROOT / 'csrc/baselines/sm103/cutlass_bf16.cu').read_text()
        occupancy = source[source.index('    cutlass::KernelHardwareInfo hardware{};'):
                           source.index('    const auto stride_a =')]
        grid = source[source.index('    const dim3 grid = Gemm::get_grid_shape'):
                      source.index('    std::ostringstream text;')]
        self.assertIn('hardware.max_active_clusters = std::min(active_clusters, hardware.sm_count / ClusterSize);', occupancy)
        # Execute the real native guards and CUDA attribute construction with
        # mocked query results. This is not a GPU occupancy measurement.
        probe = r'''
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
struct dim3 { unsigned x,y,z; dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){} };
enum { cudaLaunchAttributeClusterDimension=7 };
struct cudaLaunchAttribute { int id; struct { dim3 clusterDim; } val; };
struct cudaLaunchConfig_t {
 dim3 gridDim,blockDim; int dynamicSmemBytes; void* stream;
 cudaLaunchAttribute* attrs; unsigned numAttrs;
};
namespace cutlass { struct KernelHardwareInfo { int device_id=0,sm_count=0,max_active_clusters=0; }; }
int queries=0,query_cluster=0;
int cudaOccupancyMaxActiveClusters(int* out,const void*,cudaLaunchConfig_t* c) {
 ++queries;query_cluster=c->attrs[0].val.clusterDim.x;
 if(c->numAttrs!=1||c->attrs[0].id!=7||query_cluster!=2||
    c->attrs[0].val.clusterDim.y!=1||c->attrs[0].val.clusterDim.z!=1||
    c->gridDim.x!=148||c->blockDim.x!=256||c->dynamicSmemBytes!=214016)return 1;
 *out=66;return 0;
}
void check(int e){if(e)throw std::runtime_error("CUDA query");}
void require(bool b,const char* m){if(!b)throw std::runtime_error(m);}
struct Properties { int multiProcessorCount=148; bool clusterLaunch=true; };
struct Gemm { static dim3 get_grid_shape(int count){return dim3(count);} };
struct FakeGemm { int count; int params(){return count;} };
template<int SmMode,int ClusterM> int run(int grid_count) {
 constexpr int ClusterSize=ClusterM;
 Properties properties;int device=0,smem=214016,sm_budget=0;dim3 block(256);
 void *stream=nullptr;const void* kernel=nullptr;FakeGemm gemm{grid_count};
''' + occupancy + grid + r'''
 return hardware.max_active_clusters;
}
int main() {
 if(run<1,1>(148)!=0||queries!=0)return 1;
 if(run<1,2>(132)!=66||queries!=1||query_cluster!=2)return 2;
 if(run<2,2>(132)!=66||queries!=2)return 3;
 for(int count:{131,134,150}) {
  try{run<1,2>(count);return 4;}catch(const std::runtime_error&){}
 }
 std::cout<<"actual cluster occupancy and grid guards passed\n";
}
'''
        with tempfile.TemporaryDirectory(prefix='fuse-cutlass-cluster-test-') as directory:
            binary = Path(directory) / 'probe'
            built = subprocess.run([*compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror',
                                    '-x', 'c++', '-', '-o', str(binary)], input=probe,
                                   text=True, capture_output=True, timeout=30)
            self.assertEqual(built.returncode, 0, built.stderr)
            run = subprocess.run([str(binary)], text=True, capture_output=True, check=True, timeout=10)
        self.assertIn('guards passed', run.stdout)

    def test_multicast_type_keeps_real_one_sm_mma_and_pipeline_contract(self):
        source = (ROOT / 'csrc/baselines/sm103/cutlass_bf16.cu').read_text()
        for guard in ('using MulticastGemm = Bf16Gemm<1, 32, 2>;',
                      'MulticastGemm::Mainloop::DispatchPolicy::Stages == 4',
                      'MulticastGemm::Kernel::AccumulatorPipelineStageCount == 2',
                      'std::is_same_v<typename MulticastGemm::Mainloop::TiledMma,',
                      'MulticastGemm::Kernel::SharedStorageSize == SingleCtaGemm::Kernel::SharedStorageSize',
                      'MulticastGemm::Kernel::MaxThreadsPerBlock == SingleCtaGemm::Kernel::MaxThreadsPerBlock',
                      'Kernel::TmemAllocator::Sm100TmemCapacityColumns',
                      'runtime_reported_l2_cache_bytes', 'properties.l2CacheSize'):
            self.assertIn(guard, source)
        cutlass = os.environ.get('CUTLASS_ROOT')
        if not cutlass:
            self.skipTest('set CUTLASS_ROOT to check pinned builder/synchronization contracts')
        include = Path(cutlass) / 'include'
        common = (include / 'cutlass/gemm/collective/builders/sm100_common.inl').read_text()
        explicit_one = common.split('// MMA_1SM requested', 1)[1].split('// Auto scheduling requested')[0]
        self.assertIn('sm100_make_1sm_trivial_tiled_mma<', explicit_one)
        select_b = common.split('sm100_cluster_shape_to_tma_atom_B(', 1)[1].split('sm100_cluster_shape_to_tma_atom_SFB')[0]
        self.assertIn('sm90_cluster_shape_to_tma_atom(cute::size<0>(cluster_shape_mnk))', select_b)
        kernel = (include / 'cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp').read_text()
        self.assertIn('cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm', kernel)

    def test_close_is_idempotent_and_closed_plan_cannot_launch(self):
        plan = self.plan()
        plan.close()
        plan.close()
        self.assertEqual(self.native.sm103_cutlass_destroy.calls, [(123,)])
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            plan.run()
        self.assertEqual(self.native.sm103_cutlass_run.calls, [])

    def test_changed_buffer_refuses_launch_and_original_is_kept_alive(self):
        plan = self.plan()
        original = self.x.data
        self.x.data = Tensor((256, 64), 0x40000)
        self.assertIs(plan._buffers[0], original)
        with self.assertRaisesRegex(RuntimeError, 'buffers changed'):
            plan.run()
        self.assertFalse(self.native.sm103_cutlass_run.calls)

    def test_native_failures_propagate_without_retry(self):
        self.native.sm103_cutlass_create_v5.result = None
        with self.assertRaisesRegex(RuntimeError, 'expected native failure'):
            self.plan()
        self.assertEqual(self.native.sm103_cutlass_destroy.calls, [])
        self.native.sm103_cutlass_create_v5.result = 123
        plan = self.plan()
        self.native.sm103_cutlass_run.result = -1
        with self.assertRaisesRegex(RuntimeError, 'expected native failure'):
            plan.run()
        self.assertEqual(self.native.sm103_cutlass_run.calls, [(123, 99)])

    def test_bad_or_mismatched_info_releases_native_plan(self):
        for value in (None, b'{', json.dumps(self.info | {'m': 128}).encode(),
                      json.dumps(self.info | {'backend': 'cublaslt'}).encode(),
                      json.dumps(self.info | {'sm_mode': 2}).encode()):
            with self.subTest(value=value):
                self.native.sm103_cutlass_plan_info.result = value
                with self.assertRaises((ValueError, RuntimeError)):
                    self.plan()
        self.assertEqual(self.native.sm103_cutlass_destroy.calls, [(123,)] * 5)

    def test_rejects_precision_shape_layout_and_cross_device_before_native(self):
        invalid = [dict(dtype='fp32'), dict(cuda=False), dict(contiguous=False), dict(device=1)]
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.module.Plan(self.lib, self.x, self.w, Tensor((256, 256), 0x30000, **changes))
        for shape in ((256, 128), (256,), (0, 256)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.module.Plan(self.lib, self.x, self.w, Tensor(shape, 0x30000))
        self.x.precision = 'fp8'
        with self.assertRaisesRegex(ValueError, 'BF16'):
            self.plan()
        self.assertFalse(self.native.sm103_cutlass_create_v5.calls)

    def test_current_device_must_match_without_global_device_mutation(self):
        self.torch.cuda.current_device = lambda: 1
        with self.assertRaisesRegex(ValueError, 'current CUDA device'):
            self.plan()
        self.assertFalse(self.native.sm103_cutlass_create_v5.calls)

    def test_cpp_has_only_five_static_no_residual_bf16_instances(self):
        source = (ROOT / 'csrc/baselines/sm103/cutlass_bf16.cu').read_text()
        self.assertIn('using Element = cutlass::bfloat16_t;', source)
        self.assertIn('void, cutlass::layout::RowMajor, 8,', source)
        self.assertIn('cutlass::gemm::StaticPersistentScheduler>', source)
        self.assertIn('KernelTmaWarpSpecialized1SmSm100', source)
        self.assertIn('KernelTmaWarpSpecialized2SmSm100', source)
        self.assertEqual(source.count('return new TypedPlan<'), 5)
        self.assertEqual(set(re.findall(r'return new TypedPlan<(\d), (\d+)>', source)),
                         {('1', '32'), ('1', '64'), ('2', '32'), ('2', '64')})
        self.assertIn('return new TypedPlan<1, 32, 2>', source)
        self.assertIn('using Types = Bf16Gemm<SmMode, EpilogueN, ClusterM>;', source)
        self.assertIn('Tile, Cluster, cute::Shape<cute::_128, cute::Int<EpilogueN>>', source)
        self.assertIn('static_assert(EpilogueN == 32 || EpilogueN == 64);', source)
        self.assertNotIn('cudaMalloc', source)
        self.assertNotIn('MonolithicPersistentScheduler', source)
        run = source.split('cutlass::Status run(cudaStream_t stream) override {')[1].split('\n  }')[0]
        self.assertIn('return gemm.run(stream, nullptr, false);', run)
        self.assertNotIn('gemm.initialize(', run)
        self.assertNotIn('gemm.update(', run)
        for guard in ('cudaOccupancyMaxActiveClusters', 'cudaFuncGetAttributes',
                      'cudaPointerGetAttributes', 'cudaStreamIsCapturing',
                      'ctas <= uint64_t(hardware.sm_count)', 'grid.x % ClusterM == 0'):
            self.assertIn(guard, source)

    def test_optional_cmake_is_off_and_target_local(self):
        cmake = (ROOT / 'benchmarks/sm103/CMakeLists.txt').read_text()
        self.assertIn('option(FUSE_SM103_BUILD_CUTLASS_BF16 "Build isolated 1-SM/2-SM BF16 GEMM plans" OFF)', cmake)
        branch = cmake.split('if(FUSE_SM103_BUILD_CUTLASS_BF16)')[1]
        self.assertIn('add_library(fuse_sm103_cutlass_bf16 SHARED', branch)
        self.assertIn('CUDA::cudart CUDA::cuda_driver', branch)
        self.assertNotIn('add_compile_definitions', branch)
        self.assertNotIn('FetchContent', branch)


if __name__ == '__main__':
    unittest.main()
