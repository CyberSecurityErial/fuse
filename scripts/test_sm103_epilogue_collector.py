#!/usr/bin/env python3
"""Execute the real epilogue collector with host CUDA/launch stand-ins.

This checks orchestration, record rejection and output-validation order. It
does not emulate CUDA synchronization, numeric correctness or GPU timings.
"""

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class EpilogueCollectorContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        if not compiler or shutil.which(compiler[0]) is None:
            raise unittest.SkipTest("host C++ compiler required")
        temporary = tempfile.TemporaryDirectory(prefix="fuse-epilogue-collector-")
        cls.addClassCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        headers = {
            "cuda_runtime.h": "#pragma once\n#include <cstddef>\n"
                "using cudaError_t=int; using cudaStream_t=void*;\n"
                "struct cudaFuncAttributes { int numRegs=0, maxThreadsPerBlock=0; "
                "size_t sharedSizeBytes=0, localSizeBytes=0; };\n",
            "fuse/types.h": "#pragma once\n#include <cstdint>\n"
                "namespace fuse { constexpr int kMaxWorldSize=8; }\n",
            "cutlass/cutlass.h": "#pragma once\n",
        }
        for name, contents in headers.items():
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents)
        harness = (ROOT / "benchmarks/sm103/fused_bf16.cu").read_text()
        begin = harness.index("void profile_qkv_epilogue(")
        collector = harness[begin:harness.index("void profile_host_stages(", begin)]
        source = r'''
#include "fuse/profiling/sm103/epilogue.cuh"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#define CHECK(value) do { if (!(value)) throw std::runtime_error(#value); } while (0)
#define CUDA_CHECK(value) CHECK((value)==0)
constexpr int kWarmup=10, kSamples=50;
enum class Direction { kQkv, kOproj };
namespace fuse { struct GemmA2AParams {}; }
struct Options {
  bool qkv_epilogue_probe=true, profile=true;
  std::string profile_detail="cta", host_launch="per_gpu_thread";
  int world=4, comm_sm=2, seq_local=513, width=768;
  int projection_width() const { return width; }
};
struct RankRuntime {
  int device=0, sm_count=8;
  cudaStream_t stream=nullptr;
  fuse::GemmA2AParams qkv;
  fuse::A2AGemmCtaTimeline* timeline=nullptr;
  fuse::detail::QkvEpilogueRecord* qkv_epilogue=nullptr;
  uint32_t simulated_ready=17;
};
std::string scenario;
std::vector<RankRuntime>* active_runtimes=nullptr;
std::vector<int> modes, validation_at, poison_at;
std::vector<std::string> validation_contexts;
uint32_t previous_epoch=17, output_epoch=17;
int clear_count=0;
bool cleared=false;
int ceil_div(int n,int d) { CHECK(n>0 && d>0); return (n+d-1)/d; }
int cudaSetDevice(int device) { CHECK(device>=0 && device<8); return 0; }
int cudaMemsetAsync(void* ptr,int value,size_t size,cudaStream_t) {
  CHECK(value==0 && active_runtimes);
  bool matched=false;
  for (const auto& rank:*active_runtimes) {
    matched=matched || (ptr==rank.timeline && size==rank.sm_count*sizeof(*rank.timeline));
    matched=matched || (ptr==rank.qkv_epilogue && size==rank.sm_count*sizeof(*rank.qkv_epilogue));
  }
  CHECK(matched); // Clearing a production ready buffer is never allowed.
  std::memset(ptr,0,size); ++clear_count; return 0;
}
void finish_all(std::vector<RankRuntime>& ranks) {
  CHECK(clear_count==2*int(ranks.size())); clear_count=0; cleared=true;
}
template<class T> std::vector<T> download(const T* ptr,int count) {
  CHECK(count>0); return {ptr,ptr+count};
}
namespace fuse::detail {
cudaError_t query_qkv_epilogue_resources(const GemmA2AParams&,QkvEpilogueResources* result) {
  result->tile_m=128; result->tile_n=256; result->tile_k=64; result->cluster_ctas=1;
  result->dynamic_smem_bytes=230400;
  for (auto* attr:{&result->production,&result->role_telemetry,&result->epilogue_telemetry}) {
    attr->numRegs=120; attr->maxThreadsPerBlock=256;
  }
  if (scenario=="resources") result->tile_n=128;
  return 0;
}
}
void corrupt(fuse::detail::QkvEpilogueRecord& r,fuse::A2AGemmCtaTimeline& t,
             int m_tiles,RankRuntime& rank) {
  if(scenario=="missing") r={};
  if(scenario=="unexpected") rank.qkv_epilogue[0].tile_count=1;
  if(scenario=="count") ++r.tile_count;
  if(scenario=="epoch") --r.epoch;
  if(scenario=="coordinate") r.first_m_tile=m_tiles;
  if(scenario=="batch") r.first_batch=1;
  if(scenario=="timeline") t.start=0;
  if(scenario=="order") r.first_drain_end=r.first_store_end-1;
  if(scenario=="maximum") r.store_ns_max=r.store_ns_sum+1;
  if(scenario=="zero_sum") r.store_ns_sum=r.store_ns_max=0;
  if(scenario=="overflow") {
    r.store_ns_sum=std::numeric_limits<uint64_t>::max()-5;
    r.drain_ns_sum=10; r.drain_ns_max=10;
  }
}
std::vector<float> run_epoch(std::vector<RankRuntime>& ranks,const Options& options,
    Direction direction,uint32_t epoch,bool profile,void*,void*,void*,bool probe) {
  CHECK(direction==Direction::kQkv && cleared && epoch==previous_epoch+1);
  CHECK(!probe || profile);
  cleared=false; previous_epoch=epoch; output_epoch=epoch;
  const int mode=probe?2:int(profile);
  modes.push_back(mode);
  const int m_tiles=ceil_div(options.seq_local,128), n_tiles=ceil_div(options.projection_width(),256);
  const int total=m_tiles*n_tiles;
  for(auto& rank:ranks) {
    CHECK(rank.simulated_ready+1==epoch); rank.simulated_ready=epoch;
    const int compute=std::min(total,rank.sm_count-options.comm_sm);
    for(int worker=0;worker<compute;++worker) {
      const int cta=options.comm_sm+worker;
      auto& t=rank.timeline[cta];
      t.start=1000; t.role_done=1900; t.grid_sync_done=1950; t.end=2000;
      if(!probe) continue;
      auto& r=rank.qkv_epilogue[cta];
      r.tile_count=ceil_div(total-worker,compute); r.epoch=epoch;
      r.first_m_tile=worker%m_tiles; r.first_n_tile=worker/m_tiles;
      r.first_store_begin=1100; r.first_store_end=1120; r.first_drain_end=1130;
      r.first_ready_after=1131; r.last_ready_after=1131+(r.tile_count-1)*100;
      r.store_ns_sum=20*r.tile_count; r.store_ns_max=20;
      r.drain_ns_sum=10*r.tile_count; r.drain_ns_max=10;
    }
    if(modes.size()==181 && rank.device==0) {
      corrupt(rank.qkv_epilogue[options.comm_sm],rank.timeline[options.comm_sm],m_tiles,rank);
    }
  }
  return std::vector<float>(ranks.size(),1.0f+mode*0.1f);
}
void validate(std::vector<RankRuntime>&,const Options&,Direction direction,const std::string& context) {
  CHECK(direction==Direction::kQkv && output_epoch==previous_epoch);
  const char* expected[]={"production","role_telemetry","epilogue_telemetry"};
  CHECK(context==(modes.size()==181 ? ",profile_phase=epilogue_record" :
        std::string(",profile_phase=epilogue_")+expected[modes.back()]));
  if(scenario=="validation") throw std::runtime_error("injected output validation failure");
  validation_at.push_back(int(modes.size())); validation_contexts.push_back(context);
}
void poison_outputs(std::vector<RankRuntime>&,const Options&,Direction direction) {
  CHECK(direction==Direction::kQkv);
  poison_at.push_back(int(modes.size())); output_epoch=0;
}
''' + collector + r'''
int main(int argc,char** argv) {
  CHECK(argc>=2); scenario=argv[1];
  Options options;
  if(argc>2) options.world=std::stoi(argv[2]);
  if(scenario=="small") { options.seq_local=1; options.width=256; }
  if(scenario=="disabled") options.qkv_epilogue_probe=false;
  if(scenario=="nonprofile") options.profile=false;
  if(scenario=="full") options.profile_detail="full";
  std::vector<RankRuntime> ranks(options.world);
  std::vector<std::vector<fuse::A2AGemmCtaTimeline>> timelines(options.world);
  std::vector<std::vector<fuse::detail::QkvEpilogueRecord>> records(options.world);
  for(int rank=0;rank<options.world;++rank) {
    timelines[rank].resize(8); records[rank].resize(8);
    ranks[rank].device=rank; ranks[rank].timeline=timelines[rank].data();
    ranks[rank].qkv_epilogue=records[rank].data();
  }
  active_runtimes=&ranks;
  uint32_t epoch=17;
  try {
    profile_qkv_epilogue(ranks,options,epoch);
    CHECK(epoch==198 && modes.size()==181);
    for(int i=0;i<30;++i) CHECK(modes[i]==i%3);
    for(int sample=0;sample<50;++sample) for(int slot=0;slot<3;++slot) {
      CHECK(modes[30+sample*3+slot]==(sample%2?2-slot:slot));
    }
    CHECK(modes.back()==2 && validation_at==std::vector<int>({178,179,180,181}));
    CHECK(poison_at==std::vector<int>({177,178,179,180}));
    for(const auto& rank:ranks) CHECK(rank.simulated_ready==epoch);
    std::cout<<"PASS launches="<<modes.size()<<" epoch="<<epoch<<'\n';
  } catch(const std::exception& error) {
    std::cerr<<error.what()<<" launches="<<modes.size()<<" validations="<<validation_at.size()<<'\n';
    return 1;
  }
}
'''
        path = directory / "collector.cpp"
        path.write_text(source)
        cls.probe = directory / "collector"
        result = subprocess.run([*compiler, "-std=c++17", "-DFUSE_ENABLE_PROFILING=1",
            "-Wall", "-Wextra", "-Werror", "-I", str(directory), "-I", str(ROOT),
            "-I", str(ROOT / "include"), str(path), "-o", str(cls.probe)],
            text=True, capture_output=True)
        if result.returncode:
            raise AssertionError(result.stderr)

    def run_probe(self, scenario="normal", world=4):
        return subprocess.run([str(self.probe), scenario, str(world)],
                              text=True, capture_output=True, timeout=10)

    def test_real_collector_keeps_three_modes_all_samples_epochs_and_validation_order(self):
        for world in (4, 8):
            with self.subTest(world=world):
                result = self.run_probe(world=world)
                self.assertEqual(result.returncode, 0, result.stderr)
                rows = [line for line in result.stdout.splitlines() if line.startswith("epilogue_sample,")]
                self.assertEqual(len(rows), 150 * world)
                self.assertTrue(all("performance_accepted=0" in row for row in rows))
                self.assertEqual(sum("final_sample_poisoned=1" in row for row in rows), 3 * world)
                self.assertEqual(sum(line.startswith("epilogue_resources,")
                                     for line in result.stdout.splitlines()), 3 * world)
                self.assertIn("PASS launches=181 epoch=198", result.stdout)

    def test_small_grid_keeps_noncompute_records_empty(self):
        result = self.run_probe("small")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sum(line.startswith("epilogue_cta,") for line in result.stdout.splitlines()), 4)

    def test_missing_out_of_bounds_stale_or_unordered_records_are_rejected(self):
        for scenario in ("missing", "unexpected", "count", "epoch", "coordinate", "batch",
                         "timeline", "order", "maximum", "zero_sum", "overflow"):
            with self.subTest(scenario=scenario):
                result = self.run_probe(scenario)
                self.assertNotEqual(result.returncode, 0, f"accepted invalid record: {scenario}")

    def test_output_validation_failure_stops_before_overwrite(self):
        result = self.run_probe("validation")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("injected output validation failure launches=178 validations=0", result.stderr)

    def test_non_diagnostic_invocation_and_wrong_resources_are_rejected_before_launch(self):
        for scenario in ("disabled", "nonprofile", "full", "resources"):
            with self.subTest(scenario=scenario):
                result = self.run_probe(scenario)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("launches=0", result.stderr)


if __name__ == "__main__":
    unittest.main()
