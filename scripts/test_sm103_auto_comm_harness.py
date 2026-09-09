"""Host harness auto-budget flow tests, separate from the public API resolver tests."""

import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class AutoCommHarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shlex.split(os.environ.get('CXX', 'c++'))
        if not compiler or not shutil.which(compiler[0]):
            raise unittest.SkipTest('host C++ compiler required')
        temporary = tempfile.TemporaryDirectory(prefix='fuse-auto-comm-')
        cls.addClassCleanup(temporary.cleanup)
        cls.directory = Path(temporary.name)
        cls.source = (ROOT / 'benchmarks/sm103/fused_bf16.cu').read_text()
        types = cls.source[cls.source.index('constexpr int kWarmup ='):cls.source.index('\nvoid check_cuda(')]
        selection = cls.source[cls.source.index('void select_candidate('):cls.source.index('\nvoid describe_component(')]
        enqueue = cls.source[cls.source.index('void enqueue_operation('):cls.source.index('\n// Shared sequential/parallel seam:')]
        source = r'''
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (0)
#define CUDA_CHECK(x) CHECK((x)==0)
using Bf16=uint16_t;
''' + types + r'''
static_assert(kWarmup==10 && kSamples==50 && kCpuOracleElements==4*1024*1024);
std::string mode;
int device=0, queries=0, launches=0, agreements=0, expected_comm=0, expected_reserved=0;
int cudaSetDevice(int value) { device=value; return 0; }
namespace fused_mpi {
std::vector<int> owned_ranks(int count) { std::vector<int> ranks; for(int r=0;r<count;++r) ranks.push_back(r); return ranks; }
void agree(const std::string& value) { CHECK(value.find("comm_sm=0,")==std::string::npos); ++agreements; }
}
namespace fuse {
enum class Raster { kHeuristic, kAlongM, kAlongN };
struct Problem { int m=32768,n=8192,k=8192,max_swizzle_size=8; Raster raster=Raster::kAlongM; };
struct Route {};
struct Params { Problem gemm; Route route; int num_comm_ctas=0; uint32_t epoch=0;
  uint32_t* ready=nullptr; uint32_t* peer_route_done_epoch[8]{}; };
struct Traits { int block_m=128,block_n=256,block_k=64,threads=256,dynamic_smem_bytes=214016; };
Traits cutlass_kernel_traits() { return {}; }
Traits qkv_cutlass_kernel_traits(const Problem&,const Route&,int,int) { return {}; }
int recommended_a2a_lhs_gemm_comm_ctas(const Problem&,const Route&) {
  ++queries;
  if(mode=="unsupported") return 0;
  const std::string policy=std::getenv("FUSE_SM103_OPROJ_POLICY");
  const int budget=policy=="m128n256"?16:24;
  if(mode=="rank_mismatch" && device==1) return budget+8;
  if(mode=="repeat_mismatch" && queries%2==0) return budget+8;
  return budget;
}
int launch_a2a_gemm_cutlass(const Params& p,void*) { CHECK(p.num_comm_ctas==expected_comm);++launches;return 0; }
int launch_a2a_gemm_copy_reference(const Params& p,void*) { CHECK(p.num_comm_ctas==expected_comm);++launches;return 0; }
int launch_a2a_gemm_cutlass_reference(const Params& p,void*,int reserved) {
  CHECK(p.num_comm_ctas==expected_comm && reserved==expected_reserved);++launches;return 0;
}
int launch_batched_cutlass_reference(const Params&,void*,int) { throw std::runtime_error("unexpected QKV"); }
int launch_gemm_a2a_copy_reference(const Params&,void*) { throw std::runtime_error("unexpected QKV"); }
int launch_gemm_a2a_cutlass(const Params&,void*) { throw std::runtime_error("unexpected QKV"); }
}
struct RankRuntime { int device=0,sm_count=148; fuse::Params qkv,oproj; void* stream=nullptr;
  uint32_t *calibration_qkv_ready=nullptr,*calibration_oproj_ready=nullptr,*calibration_route_done=nullptr; };
''' + selection + r'''
struct RankLaunch { std::vector<RankRuntime>& runtimes; Direction direction; uint32_t epoch;
  bool profile; MeasurementComponent component; int reserved_comm_ctas; };
''' + enqueue + r'''
int main(int argc,char** argv) {
  CHECK(argc==2);mode=argv[1];setenv("FUSE_SM103_OPROJ_COMM_LAYOUT","rows",1);
  std::vector<RankRuntime> runtimes(4);for(int r=0;r<4;++r) runtimes[r].device=r;
  std::vector<Candidate> candidates{{Direction::kOproj,0,"m128n256",true},
                                  {Direction::kOproj,0,"m128n256k64e32",true}};
  try {
    resolve_auto_candidates(runtimes,candidates);
    CHECK(mode=="success" && queries==16 && agreements==2);
    CHECK(candidates[0].comm_sm==16 && candidates[1].comm_sm==24);
    for(const auto& candidate:candidates) {
      select_candidate(runtimes,candidate,",test=auto");
      expected_comm=0;expected_reserved=candidate.comm_sm;
      for(auto component:{MeasurementComponent::kFused,MeasurementComponent::kComputeReference,MeasurementComponent::kCopyReference}) {
        RankLaunch job{runtimes,Direction::kOproj,3,false,component,candidate.comm_sm};
        for(int rank=0;rank<4;++rank) enqueue_operation(job,rank);
      }
    }
    CHECK(launches==24);
    Candidate manual{Direction::kOproj,8,"m128n256",false};
    select_candidate(runtimes,manual,",test=manual");expected_comm=8;expected_reserved=8;
    RankLaunch job{runtimes,Direction::kOproj,4,false,MeasurementComponent::kFused,8};
    enqueue_operation(job,0);
    std::cout << "PASS zero F/R and positive C budget; explicit unchanged\n";
  } catch(const std::exception&) {
    if(mode=="success") throw;
    CHECK(launches==0);std::cout << "PASS rejected before launch\n";
  }
}
'''
        cls.binary = cls.directory / 'harness'
        built = subprocess.run([*compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
            '-DFUSE_ENABLE_PROFILING=0', '-DFUSE_BENCH_MPI=0', '-x', 'c++', '-', '-o', str(cls.binary)],
            input=source, text=True, capture_output=True, timeout=60)
        if built.returncode:
            raise AssertionError(built.stderr)
        parser = cls.source[cls.source.index('Options parse_options('):cls.source.index('\nstruct RankRuntime')]
        parser_source = source[:source.index('std::string mode;')] + r'''
namespace fused_mpi { constexpr bool enabled=true; int process_world=4; void finalize() {} }
''' + parser + r'''
int main(int argc,char** argv) {
  try {
    const auto options=parse_options(argc,argv);
    std::cout << options.auto_oproj_comm << ' ' << options.comm_sm << ' ' << make_candidates(options).size() << '\n';
  } catch(const std::exception& e) { std::cerr << e.what(); return 1; }
}
'''
        cls.parser = cls.directory / 'parser'
        built = subprocess.run([*compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
            '-DFUSE_ENABLE_PROFILING=0', '-DFUSE_BENCH_MPI=1', '-x', 'c++', '-', '-o', str(cls.parser)],
            input=parser_source, text=True, capture_output=True, timeout=60)
        if built.returncode:
            raise AssertionError(built.stderr)

    def test_query_once_per_candidate_rank_records_actual_and_preserves_zero_launch(self):
        result = subprocess.run([str(self.binary), 'success'], capture_output=True, text=True, check=True, timeout=10)
        records = [dict(cell.split('=', 1) for cell in line.split(',')[2:])
                   for line in result.stdout.splitlines() if line.startswith('auto_comm,')]
        self.assertEqual(len(records), 8)
        self.assertEqual({(int(r['candidate']), int(r['rank'])) for r in records},
                         {(c, rank) for c in (1, 2) for rank in range(4)})
        for row in records:
            self.assertEqual(int(row['comm_sm']), 16 if row['candidate'] == '1' else 24)
            self.assertEqual(row['launch_comm'], '0')
            for field in ('query_us', 'repeat_query_us'):
                self.assertTrue(math.isfinite(float(row[field])) and float(row[field]) >= 0)
        self.assertIn('PASS zero F/R and positive C budget; explicit unchanged', result.stdout)

    def test_unsupported_or_disagreeing_queries_never_launch(self):
        for mode in ('unsupported', 'rank_mismatch', 'repeat_mismatch'):
            with self.subTest(mode=mode):
                result = subprocess.run([str(self.binary), mode], capture_output=True, text=True, check=True, timeout=10)
                self.assertIn('PASS rejected before launch', result.stdout)

    def test_real_main_resolves_before_payloads_and_keeps_actual_reference_budget(self):
        main = self.source[self.source.index('int main('):]
        self.assertLess(main.index('resolve_auto_candidates(runtimes, candidates)'), main.index('for (uint32_t generation'))
        self.assertIn('candidate_options.comm_sm = candidate.comm_sm;', main)
        self.assertIn('Options reference_options = candidate_options;', main)
        self.assertIn('options.component, options.comm_sm, launch_us, timing', self.source)
        self.assertIn('runtime.oproj.input_epoch = generation + 1;', self.source)

    def test_real_mpi_parser_accepts_auto_graph_and_rejects_unsupported_combinations(self):
        flags = ['--auto-oproj-comm', '--fused-direction', 'oproj', '--launch', 'graph', '--causal',
                 '--oproj-policy-list', 'm128n256,m128n256k64e32', '--max-swizzle-size', '8',
                 '--q-heads', '64', '--hidden', '8192', '--seq-local', '32768', '--oproj-comm-layout', 'rows']
        environment = {k: v for k, v in os.environ.items() if k not in
                       ('FUSE_QKV_GEMM_POLICY', 'FUSE_SM103_OPROJ_POLICY', 'FUSE_SM103_OPROJ_COMM_LAYOUT')}
        for suffix in ([], ['--calibrate'], ['--quick']):
            result = subprocess.run([str(self.parser), *flags, *suffix], env=environment,
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), '1 0 2')
        for suffix in (['--comm-sm', '8'], ['--comm-sm-list', '8,16'], ['--launch', 'eager'],
                       ['--profile'], ['--calibrate', '--compute-only'], ['--oproj-comm-layout', 'columns'],
                       ['--max-swizzle-size', '2'], ['--oproj-policy-list', 'auto'], ['--seq-local', '128']):
            with self.subTest(suffix=suffix):
                result = subprocess.run([str(self.parser), *flags, *suffix], env=environment,
                                        capture_output=True, text=True, timeout=10)
                self.assertNotEqual(result.returncode, 0)
        result = subprocess.run([str(self.parser)], env=environment, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.stdout.strip(), '0 8 2')


if __name__ == '__main__':
    unittest.main()
