"""Execute production grouped tile arithmetic on CPU; not a CUDA validation."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import re

ROOT = Path(__file__).resolve().parents[1]


class GroupedScheduleTests(unittest.TestCase):
    def test_dispatch_row_address_prefetch_matches_copy_ownership(self):
        body=(ROOT/'csrc/operators/sm103/detail/grouped/a2a_gemm.cuh').read_text()
        for valid in range(1,129):
            for slots in (1,2,3,4,8):
                for batch in (1,2,4,8,16,32,64,128):
                    all_rows=[]
                    for warp in range(slots):
                        expected=[r for begin in range(warp*batch,valid,slots*batch)
                                  for r in range(begin,min(valid,begin+batch))]
                        actual=[]
                        for lane in range(32):
                            owned=lane
                            while True:
                                row=(owned//batch)*slots*batch+warp*batch+owned%batch
                                if row>=valid: break
                                actual.append(row); owned+=32
                        self.assertEqual(sorted(actual),expected)
                        all_rows.extend(actual)
                    self.assertEqual(sorted(all_rows),list(range(valid)))
        self.assertIn('const Bf16* src=row_sources[begin+r];',body)
        self.assertRegex(body,r'__syncwarp\(\);\s+if\(lane!=0\) return;')

    def test_dispatch_staged_rows_are_contiguous_and_covered(self):
        body=(ROOT/'csrc/operators/sm103/detail/grouped/a2a_gemm.cuh').read_text()
        stage_bytes=int(re.search(r'kStageBytes = (\d+) \* 1024',body)[1])*1024
        for k in (8,24,256,1408,2048,3072,4096,5120,6144,7168,32768):
            if k*2>stage_bytes:
                self.assertIn('int64_t(p.columns)*sizeof(Bf16)<=kStageBytes',body)
                continue  # A whole row cannot fit: production uses vector copy.
            batch_rows=1
            while batch_rows<128 and batch_rows*2*k*2<=stage_bytes:
                batch_rows*=2
            self.assertLessEqual(batch_rows*k*2,stage_bytes)
            for valid in (1,7,63,127,128):
                rows=[r for begin in range(0,valid,batch_rows)
                      for r in range(begin,min(valid,begin+batch_rows))]
                self.assertEqual(rows,list(range(valid)))
                for slots in (1,2,3,4,8):
                    selected=batch_rows
                    while selected>1 and (valid+selected-1)//selected<slots:
                        selected//=2
                    copied=[r for slot in range(slots)
                            for begin in range(slot*selected,valid,slots*selected)
                            for r in range(begin,min(valid,begin+selected))]
                    self.assertEqual(sorted(copied),list(range(valid)))
        for valid in (1,7,9,63,127,128):
            for producers in (1,2,4,8,16):
                copied=[r for stripe in range(producers)
                        for start in range(stripe*8,valid,8*producers)
                        for r in range(start,min(valid,start+8))]
                self.assertEqual(sorted(copied),list(range(valid)))
        self.assertIn('rows>TileM',body)
        self.assertIn('offset+=warps*producers',body)
        self.assertIn('mbarrier.inval.shared::cta.b64',body)
        self.assertIn('tma_store_wait_all();',body)

    def test_dispatch_vector_batches_cover_tail_once(self):
        body=(ROOT/'csrc/operators/sm103/detail/grouped/a2a_gemm.cuh').read_text()
        depth=int(re.search(r'kVectorsPerThread = (\d+)',body)[1])
        for k in (8,24,256,1024,1408,2048,3072,4096,5120,6144,7168):
            seen=[]
            for lane in range(32):
                for col in range(lane*8,k,depth*32*8):
                    for i in range(depth):
                        start=col+i*32*8
                        if start<k: seen.extend(range(start,start+8))
            self.assertEqual(sorted(seen),list(range(k)))
        self.assertIn('col += kVectorsPerThread * 32 * 8',body)
        self.assertIn('uint4 values[kVectorsPerThread]',body)

    def test_measurement_accepts_first_stable_not_fastest(self):
        compiler = shutil.which('c++')
        if not compiler:
            self.skipTest('C++ compiler unavailable')
        body = (ROOT / 'benchmarks/sm103/grouped_measurement.cuh').read_text()
        types = body[body.index('inline double percentile'):body.index('inline Samples measure')]
        formal = body[body.index('  Timer timer(ops);'):body.index('\n}  // namespace grouped_measurement')]
        source = r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>
bool all_bad=false;
int calls=0;
struct Timer {
  explicit Timer(int) { calls=0; }
  std::vector<float> sample() {
    const int round=calls/60, index=calls++%60;
    const bool bad=all_bad || round==0;
    const float ms=bad ? (index<35?10.f:20.f) : (round==1?30.f:1.f);
    return {ms*.5f,ms};
  }
};
''' + types + '\nSamples exercise() { int ops=0; Samples out; out.warmup=128;\n' + formal + r'''
int main() {
  auto result=exercise();
  assert(result.rounds.size()==2 && calls==120);
  assert(result.p50==30 && result.drift==0 && result.warmup==148);
  assert(result.rounds[0].drift>.05 && result.rounds[0].ranks_ms.size()==50);
  all_bad=true; result=exercise();
  assert(result.rounds.size()==3 && calls==180 && result.drift>.05);
  assert(result.warmup==158);
}
'''
        with tempfile.TemporaryDirectory(prefix='fuse-grouped-timer-') as tmp:
            binary = str(Path(tmp) / 'timer')
            compiled = subprocess.run([compiler, '-std=c++17', '-O2', '-Wall', '-Werror',
                                       '-x', 'c++', '-', '-o', binary],
                                      input=source, capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            result = subprocess.run([binary], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_private_implementation_layers(self):
        root = ROOT / 'csrc/operators/sm103/detail/grouped'
        for header in root.glob('*.cuh'):
            for dependency in re.findall(r'^#include "([^"]+)"', header.read_text(), re.M):
                if dependency.startswith('fuse/'):
                    self.assertTrue(dependency.startswith(('fuse/arch/', 'fuse/profiling/')) or
                                    dependency == 'fuse/operators/primitives/grouped_gemm.h')
                else:
                    self.assertNotIn('/', dependency, (header.name, dependency))
                    self.assertTrue((root / dependency).is_file(), dependency)
        for filename, role in [('a2a_gemm.cuh', 'GroupedDispatchComm'),
                               ('gemm_a2a.cuh', 'GroupedCombineComm')]:
            body = (root / filename).read_text()
            self.assertIn(role, body)
            self.assertIn('GroupedMonolithicGemm<', body)
            self.assertNotIn('GroupedTokenComm', body)

    def test_tile_order_and_ctasp_coverage(self):
        compiler = shutil.which('c++')
        if not compiler:
            self.skipTest('C++ compiler unavailable')
        body = (ROOT / 'csrc/operators/sm103/detail/grouped/producer_consumer.cuh').read_text()
        body = '\n'.join(line for line in body.splitlines()
                         if not line.startswith(('#include', '#pragma once')))
        source = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>
#define CUTLASS_HOST_DEVICE
''' + body + r'''
using Order = fuse::detail::GroupedTileOrder;

void check(const std::vector<int>& rows, int n, int sw, bool along_n, int window) {
  std::vector<int64_t> offsets(1, 0);
  for (int m : rows) offsets.push_back(offsets.back() + (int64_t(m) + 127) / 128);
  Order order{offsets.data(), int(rows.size()), n, sw, along_n, window};
  std::vector<Order::Tile> expected;
  // Independent nested-loop oracle, not a copy of division-based decode.
  for (int e = 0; e < int(rows.size()); ++e) {
    const int total_m = int(offsets[e+1] - offsets[e]);
    const int step=window?window:std::max(1,total_m);
    for(int first_m=0;first_m<total_m;first_m+=step) {
    const int mt = std::min(step,total_m-first_m);
    const int outer = along_n ? mt : n, middle = along_n ? n : mt;
    for (int band = 0; band < outer; band += sw)
      for (int major = 0; major < middle; ++major)
        for (int minor = band; minor < std::min(band + sw, outer); ++minor)
          expected.push_back({e, first_m+(along_n ? minor : major), along_n ? major : minor, true});
    }
  }
  assert(int64_t(expected.size()) == order.tiles());
  std::vector<int> seen(order.tiles());
  std::vector<int64_t> first(order.row_tiles(), order.tiles());
  for (int64_t q = 0; q < order.tiles(); ++q) {
    const auto t = order.decode(q), r = expected[q];
    assert(t.valid && t.expert == r.expert && t.m == r.m && t.n == r.n);
    assert(order.linear(t.expert, t.m, t.n) == q);
    const auto a = order.input_ready_index(t.expert, t.m);
    const auto d = order.output_ready_index(t.expert, t.m, t.n);
    assert(a >= 0 && a < order.row_tiles() && d >= 0 && d < order.tiles());
    assert(++seen[d] == 1);
    first[a] = std::min(first[a], q);
  }
  assert(std::is_sorted(first.begin(), first.end()));
  // Every reader of an old slot precedes every reader of its replacement.
  // A static CTA queue cannot wait for a new window while retaining an
  // unexecuted old-window task behind it, for either raster or partial bands.
  if(window) for(int e=0;e<int(rows.size());++e)
    for(int m=window;m<offsets[e+1]-offsets[e];++m)
      for(int old_n=0;old_n<n;++old_n) for(int new_n=0;new_n<n;++new_n)
        assert(order.linear(e,m-window,old_n)<order.linear(e,m,new_n));
  assert(!order.decode(-1).valid && !order.decode(order.tiles()).valid);
  for (int comm : {1, 8, 20}) for (int compute : {1, 3, 8, 64, 128}) {
    std::fill(seen.begin(), seen.end(), 0);
    // Physical CTA prefix is COMM, stride is only COMPUTE, including tails.
    for (int block = comm; block < comm + compute; ++block)
      for (int64_t q = block - comm; q < order.tiles(); q += compute)
        assert(++seen[q] == 1);
    for (int count : seen) assert(count == 1);
  }
}

int main() {
  // New tail sharing: each row is copied once; every nonempty stripe arrives
  // once. Empty stripes must not increase the final publisher's target.
  for(int panels:{21,36,64,72,108,144,257}) for(int c:{8,20,40,64}) {
    const int begin=panels-panels%c;
    if(!begin || begin==panels) continue;
    for(int rows:{1,7,17,64,127,128}) {
      const int producers=(rows+7)/8;
      std::vector<int> seen((panels-begin)*rows),arrivals(panels-begin);
      for(int worker=0;worker<c;++worker)
        for(int work=worker;work<(panels-begin)*16;work+=c) {
          const int p=work/16,stripe=work%16;
          if(stripe>=producers) continue;
          ++arrivals[p];
          for(int warp=0;warp<8;++warp)
            for(int row=stripe*8+warp;row<rows;row+=8*producers) ++seen[p*rows+row];
        }
      for(int x:seen) assert(x==1);
      for(int x:arrivals) assert(x==producers);
    }
  }
  // Cooperative Dispatch still copies every row exactly once and requires only
  // one consumer ready per panel, including tails and fewer rows than warps.
  for (int panels : {0,1,2,3,8,17,32,128}) for (int c : {1,8,20,40,64,128}) {
    int splits=fuse::detail::grouped_dispatch_splits(panels,c);
    assert(splits>=1 && splits<=16);
    if(panels>=c || panels==0) assert(splits==1);
    for(int rows : {1,7,8,9,16,31,64,127,128}) {
      const int producers=std::min(splits,(rows+7)/8);
      std::vector<int> seen(rows);
      for(int stripe=0;stripe<splits;++stripe) {
        if(stripe>=producers) continue;
        for(int warp=0;warp<8;++warp)
          for(int row=stripe*8+warp;row<rows;row+=8*producers) ++seen[row];
      }
      for(int visits:seen) assert(visits==1);
    }
  }
  for (const auto& rows : std::vector<std::vector<int>>{
      {}, {0}, {0,0,0}, {1}, {1,8,0,129,256,0}, {8192,1,0,17,127,128,129},
      {3,1001,12,23,384,999,0,0,0,4097}})
    for (int n : {1,3,8,11,32}) for (int sw : {1,2,4,8})
      for (bool along_n : {false,true}) for(int window : {0,1,2,3,8}) check(rows, n, sw, along_n, window);
  // Dynamic replay changes the same offsets storage; no cached host tile count.
  std::vector<int64_t> offsets{0,0,2,2};
  Order dynamic{offsets.data(), 3, 7, 4, true};
  assert(dynamic.decode(0).expert == 1 && dynamic.tiles() == 14);
  offsets = {0,3,3,8};
  dynamic.row_tile_offsets = offsets.data();
  assert(dynamic.decode(0).expert == 0 && dynamic.decode(21).expert == 2);
  assert(dynamic.tiles() == 56);
  // Tile totals and ready indexing must not silently overflow int32.
  int64_t large[] = {0, 1LL<<25, 1LL<<26};
  Order wide{large, 2, 256, 8, false};
  for (int64_t q : {0LL, (1LL<<33)-1, 1LL<<33, (1LL<<34)-1}) {
    const auto t = wide.decode(q);
    assert(t.valid && wide.linear(t.expert,t.m,t.n) == q);
  }
  std::cout << "grouped order: coverage, tails, empty experts, first-use, replay, int64 passed\n";
}
'''
        with tempfile.TemporaryDirectory(prefix='fuse-grouped-test-') as tmp:
            binary = str(Path(tmp) / 'grouped_order')
            compiled = subprocess.run([compiler, '-std=c++17', '-O2', '-Wall',
                                       '-Wextra', '-Werror', '-x', 'c++', '-', '-o', binary],
                                      input=source, capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            result = subprocess.run([binary], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
