"""Real OProj API/resolver/selector CPU tests; CUDA and CUTLASS plumbing is stubbed.

These execute the real wrapper bodies, problem validator and calibrated model,
not device code, descriptor encoding, CUDA Graphs or runtime GPU attributes.
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
DETAIL = ROOT / 'csrc/operators/sm103/detail'
API = ROOT / 'csrc/operators/sm103/api'


def function(text, signature):
    start = text.index(signature)
    end = text.index('{', start) + 1
    depth = 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


class AutoCommunicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('clang++') or shutil.which('g++')
        if not compiler:
            raise unittest.SkipTest('A host C++17 compiler is required')
        cls.temp = tempfile.TemporaryDirectory(prefix='fuse-auto-api-')
        cls.addClassCleanup(cls.temp.cleanup)
        directory = Path(cls.temp.name)
        (directory / 'cutlass').mkdir()
        (directory / 'cutlass/cutlass.h').write_text(
            '#pragma once\n#define CUTLASS_HOST\n#define CUTLASS_HOST_DEVICE\n')
        (directory / 'cutlass/fast_math.h').write_text(r'''
#pragma once
#include <cstdint>
namespace cutlass {
struct FastDivmodU64 {
  uint64_t divisor = 1;
  FastDivmodU64() = default;
  explicit FastDivmodU64(uint64_t d) : divisor(d) {}
  uint64_t divide(uint64_t v) const { return v / divisor; }
};
}
''')
        policy = (API / 'policy.cuh').read_text()
        forward = (API / 'forward.cuh').read_text()
        reference = (API / 'reference.cuh').read_text()
        launch = (DETAIL / 'launch.cuh').read_text()
        gemm = (DETAIL / 'gemm.cuh').read_text()
        primitive = (ROOT / 'include/fuse/operators/primitives/a2a_gemm.h').read_text()
        source = r'''
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "fuse/layout/gemm.h"
#include "fuse/layout/ulysses.h"
#include "autotune.cuh"
#define FUSE_ENABLE_PROFILING 1
#define __host__
#define __device__
enum cudaError_t {cudaSuccess, cudaErrorInvalidValue, cudaErrorNotSupported, cudaErrorUnknown};
using cudaStream_t = void*;
namespace fuse {
using Bf16 = uint16_t;
constexpr int kMaxWorldSize = 8, kAlignment = 8;
enum class A2ALhsGemmPolicy {kAuto,kM64N128};
PARAMS;
PROBLEM
POLICY
struct DeviceInfo {int device=0,sm_count=148;};
int device_sms=148,device_id=0,lowering_calls=0;
cudaError_t device_error=cudaSuccess;
cudaError_t device_info(DeviceInfo* info) {*info={device_id,device_sms};return device_error;}
struct A2AGemmCtaTimeline {};
struct A2AGemmPeerTimeline {};
struct FakeGemm {};
struct GemmArguments {GemmProblem problem;GemmRaster raster;int budget;};
template<class Gemm>
GemmArguments gemm_arguments(const GemmProblem& p,const Bf16*,const Bf16*,Bf16*,float,
    int budget,const DeviceInfo&,GemmRaster fallback) {
  ++lowering_calls;return {p,p.raster==GemmRaster::kHeuristic?fallback:p.raster,budget};
}
// CPU stand-in for CUTLASS's host geometry lowering. The real resolver must
// consume effective width, including N-limited requested8 -> resolved4/2/1.
template<class Gemm>
detail::A2AInputTileOrder a2a_input_order(const GemmArguments& args) {
  auto& p=args.problem;int minimum=std::min((p.m+127)/128,(p.n+255)/256);
  detail::A2AInputTileOrder order;order.along_n=args.raster==GemmRaster::kAlongN;
  order.log_swizzle=p.max_swizzle_size>=8 && minimum>=6 ? 3:
    p.max_swizzle_size>=4 && minimum>=3 ? 2:p.max_swizzle_size>=2 && minimum>=2 ? 1:0;
  return order;
}
struct FakeComm {
  static constexpr bool kNeedsGridFinalize=false;
  struct Arguments {A2AGemmParams params;detail::A2AInputTileOrder input_order;};
  static cudaError_t initialize(const Arguments&) {return cudaSuccess;}
  static bool can_implement(const Arguments&) {return true;}
  static Arguments to_underlying_arguments(const Arguments& args) {return args;}
};
struct Binding {using Gemm=FakeGemm;using PureGemm=FakeGemm;using TelemetryPureGemm=FakeGemm;
  using Kernel=FakeGemm;using Comm=FakeComm;};
template<class T>struct TypeTag {using type=T;};
template<class Visitor>
cudaError_t visit_oproj_forward_policy(OprojGemmPolicy,Visitor& visitor) {
  return visitor(TypeTag<Binding>{});
}
RESOLVER
QUERY
int last_forward=-1,last_telemetry=-1,last_copy=-1,last_reserved=-1;
template<bool Instrumented=false,class... Args>
cudaError_t launch_oproj_forward_policy(const A2AGemmParams& p,cudaStream_t,Args...) {
  if(p.num_comm_ctas<=0 || p.num_comm_ctas>=device_sms || !p.epoch)return cudaErrorInvalidValue;
  if constexpr(Instrumented)last_telemetry=p.num_comm_ctas;else last_forward=p.num_comm_ctas;
  return cudaSuccess;
}
FORWARD
TELEMETRY
REFERENCE_DEVICE
namespace detail {
const int* oproj_pipeline_sink=nullptr;
template<class Comm,class Kernel>struct CopyReferenceKernel {};
template<class Kernel,class Args>
cudaError_t launch_reference_cooperative(const Args&,const DeviceInfo&,int count,cudaStream_t) {
  last_copy=count;return cudaSuccess;
}
}
template<class B,class G=typename B::PureGemm,bool Instrumented=false>
cudaError_t launch_gemm_reference_impl(const GemmProblem&,const Bf16*,const Bf16*,Bf16*,float,
    int reserved,const DeviceInfo&,GemmRaster,cudaStream_t) {last_reserved=reserved;return cudaSuccess;}
PURE
COPY
}
int main(int argc,char** argv) {
  using namespace fuse;A2AGemmParams p;
  p.gemm.m=32768;p.gemm.n=8192;p.gemm.k=8192;p.gemm.max_swizzle_size=8;
  p.gemm.raster=GemmRaster::kAlongM;p.epoch=1;
  auto& r=p.route;r.world_size=4;r.q_heads=64;r.local_heads=16;r.head_dim=128;
  r.seq_local=p.gemm.m;r.global_seq=r.seq_local*4;
  r.causal_load_balanced=true;r.direction=RouteDirection::kInverse;
  const std::string mode=argc>1?argv[1]:"default";
  if(mode=="explicit")p.num_comm_ctas=13;
  if(mode=="oversized")p.num_comm_ctas=148;
  if(mode=="negative")p.num_comm_ctas=-1;
  if(mode=="world8") {r.world_size=8;r.local_heads=8;r.global_seq=r.seq_local*8;}
  if(mode=="world2") {r.world_size=2;r.local_heads=32;r.global_seq=r.seq_local*2;}
  if(mode=="alongn")p.gemm.raster=GemmRaster::kAlongN;
  if(mode=="heuristic")p.gemm.raster=GemmRaster::kHeuristic;
  if(mode=="sw4")p.gemm.max_swizzle_size=4;
  if(mode=="nlimited4")p.gemm.n=1024;
  if(mode=="nlimited2")p.gemm.n=512;
  if(mode=="invalidsw")p.gemm.max_swizzle_size=3;
  if(mode=="dtype")p.gemm.weight_dtype=DType::kFloat8E4M3;
  if(mode=="transpose")p.gemm.transpose_a=true;
  if(mode=="batchgemm")p.gemm.l=2;
  if(mode=="strided")p.gemm.stride_b.row=p.gemm.k+8;
  if(mode=="stridedout")p.gemm.stride_d.row=p.gemm.n+8;
  if(mode=="noncausal")r.causal_load_balanced=false;
  if(mode=="cyclic")r.cyclic_peer_order=true;
  if(mode=="packed")r.packed_source_row=reinterpret_cast<const int32_t*>(16);
  if(mode=="packgranularity")r.packed_row_granularity=128;
  if(mode=="defer")r.defer_v_a2a=true;
  if(mode=="interleaved")r.qkv_peer_interleaved=true;
  if(mode=="external")p.input_epoch=1;
  if(mode=="direction")r.direction=RouteDirection::kForward;
  if(mode=="channel")r.channel_count=3;
  if(mode=="rank")r.rank=4;
  if(mode=="mapping")r.global_seq++;
  if(mode=="tail") {p.gemm.m=32896;r.seq_local=p.gemm.m;r.global_seq=r.seq_local*4;}
  if(mode=="lowk") {p.gemm.k=4096;r.q_heads=32;r.local_heads=8;}
  if(mode=="highk") {p.gemm.k=32768;r.q_heads=256;r.local_heads=64;}
  if(mode=="largen")p.gemm.n=2147483640;
  if(mode=="largemn") {p.gemm.m=2147483392;p.gemm.n=2147483640;
    r.seq_local=256;r.batch=p.gemm.m/256;r.global_seq=1024;}
  if(mode=="sms")device_sms=132;
  if(mode=="devicefail")device_error=cudaErrorUnknown;
  if(mode=="epoch")p.epoch=0;
  auto recommendation=recommended_a2a_lhs_gemm_comm_ctas(p.gemm,r);
  auto forward=launch_a2a_gemm_cutlass(p,nullptr);
  auto telemetry=launch_a2a_gemm_cutlass_role_telemetry(p,nullptr,148,nullptr,0,nullptr);
  auto copy=launch_a2a_gemm_copy_reference(p,nullptr);
  auto pure=launch_a2a_gemm_cutlass_reference(p,nullptr,0);
  std::cout<<recommendation<<' '<<forward<<' '<<last_forward<<' '<<telemetry<<' '
    <<last_telemetry<<' '<<copy<<' '<<last_copy<<' '<<pure<<' '<<last_reserved<<' '<<p.num_comm_ctas<<' ';
  auto matched=launch_a2a_gemm_cutlass_reference(p,nullptr,recommendation);
  std::cout<<matched<<' '<<last_reserved<<' '<<lowering_calls<<'\n';
}
'''
        substitutions = dict(
            PARAMS=function(primitive, 'struct A2AGemmParams'),
            PROBLEM=gemm[gemm.index('__host__ __device__ constexpr int64_t a_row_stride'):
                         gemm.index('inline auto raster_option')],
            POLICY=launch[launch.index('enum class OprojGemmPolicy'):launch.index('template <class Visitor>')],
            RESOLVER=function(policy, 'inline cudaError_t resolve_oproj_communication'),
            QUERY=function(policy, 'int32_t recommended_a2a_lhs_gemm_comm_ctas'),
            FORWARD=function(forward, 'cudaError_t launch_a2a_gemm_cutlass('),
            TELEMETRY=function(forward, 'cudaError_t launch_a2a_gemm_cutlass_role_telemetry'),
            REFERENCE_DEVICE=function(reference, 'inline cudaError_t reference_device_info'),
            PURE=function(reference, 'cudaError_t launch_a2a_gemm_cutlass_reference('),
            COPY=function(reference, 'cudaError_t launch_a2a_gemm_copy_reference('))
        for key, value in substitutions.items():
            source = source.replace('\n' + key, '\n' + value)
        cls.binary = directory / 'api'
        result = subprocess.run([compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
            '-I', str(directory), '-I', str(ROOT / 'include'), '-I', str(DETAIL),
            '-x', 'c++', '-', '-o', str(cls.binary)], input=source, text=True, capture_output=True, timeout=30)
        if result.returncode:
            raise AssertionError(result.stderr)

    def query(self, mode='default', policy='m128n256', layout=None):
        env = os.environ.copy()
        for name, value in (('FUSE_SM103_OPROJ_POLICY', policy), ('FUSE_SM103_OPROJ_COMM_LAYOUT', layout)):
            env.pop(name, None)
            if value is not None:
                env[name] = value
        run = subprocess.run([str(self.binary), mode], env=env, text=True, capture_output=True, check=True, timeout=10)
        keys = ('recommended', 'forward', 'forward_budget', 'telemetry', 'telemetry_budget', 'copy',
                'copy_budget', 'pure', 'pure_reserved', 'input_budget', 'matched', 'matched_reserved', 'lowering_calls')
        return dict(zip(keys, map(int, run.stdout.split())))

    def test_automatic_budget_shared_by_every_wrapper(self):
        for mode in ('default', 'world8', 'alongn', 'heuristic', 'sw4', 'nlimited4', 'external'):
            for policy in ('m128n256', 'm128n256k64e32'):
                with self.subTest(mode=mode, policy=policy):
                    row = self.query(mode, policy)
                    self.assertIn(row['recommended'], (8, 16, 24, 32, 48))
                    for name in ('forward', 'telemetry', 'copy'):
                        self.assertEqual(row[name], 0)
                        self.assertEqual(row[name + '_budget'], row['recommended'])
                    self.assertEqual(row['input_budget'], 0)
                    self.assertEqual(row['pure_reserved'], 0)
                    self.assertEqual(row['matched_reserved'], row['recommended'])

    def test_explicit_budget_preserves_old_path(self):
        for policy in (None, 'auto', 'm128n256'):
            row = self.query('explicit', policy)
            for name in ('forward', 'telemetry', 'copy'):
                self.assertEqual(row[name], 0)
                self.assertEqual(row[name + '_budget'], 13)
            self.assertEqual(row['input_budget'], 13)
        for name in ('forward', 'telemetry', 'copy'):
            self.assertEqual(self.query('oversized')[name], 1)

    def test_auto_does_not_replace_uncalibrated_collective(self):
        for policy in (None, 'auto', 'm128n128', 'm128n128k128', 'm128n256k128e32'):
            row = self.query(policy=policy)
            self.assertEqual(row['recommended'], 0)
            self.assertEqual(row['forward'], 2)
            self.assertEqual(row['pure'], 0)
            self.assertEqual(row['pure_reserved'], 0)

    def test_calibration_domain_rejections(self):
        for mode in ('world2', 'nlimited2', 'invalidsw', 'dtype', 'transpose', 'batchgemm',
                     'strided', 'stridedout', 'noncausal', 'cyclic', 'packed', 'packgranularity',
                     'defer', 'interleaved', 'direction', 'channel', 'rank', 'mapping', 'tail', 'lowk', 'sms'):
            with self.subTest(mode=mode):
                row = self.query(mode)
                for name in ('forward', 'telemetry', 'copy'):
                    self.assertEqual(row[name], 2)
                    self.assertEqual(row[name + '_budget'], -1)
                self.assertEqual(row['recommended'], 0)

    def test_layout_parser_and_override(self):
        for layout in (None, '', 'rows'):
            self.assertEqual(self.query(layout=layout)['forward'], 0)
        for layout, status in (('columns', 2), ('invalid', 1)):
            row = self.query(layout=layout)
            self.assertEqual(row['recommended'], 0)
            for name in ('forward', 'telemetry', 'copy'):
                self.assertEqual(row[name], status)
        # Positive overrides leave real Comm initialization responsible for
        # checking layouts outside the narrower model domain.
        self.assertEqual(self.query('explicit', layout='columns')['forward_budget'], 13)

    def test_failure_and_epoch_are_not_silently_launched(self):
        for mode, status in (('negative', 1), ('devicefail', 3), ('epoch', 1)):
            row = self.query(mode)
            for name in ('forward', 'telemetry', 'copy'):
                self.assertEqual(row[name], status)
                self.assertEqual(row[name + '_budget'], -1)
        self.assertEqual(self.query('devicefail')['recommended'], 0)

    def test_bad_geometry_rejected_before_scheduler_lowering(self):
        for mode in ('largen', 'largemn', 'invalidsw', 'lowk', 'highk', 'sms', 'devicefail'):
            with self.subTest(mode=mode):
                row = self.query(mode)
                self.assertEqual(row['recommended'], 0)
                self.assertNotEqual(row['forward'], 0)
                self.assertEqual(row['lowering_calls'], 0)


if __name__ == '__main__':
    unittest.main()
