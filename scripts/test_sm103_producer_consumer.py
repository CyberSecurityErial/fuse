"""Host-execute the shared tile-order arithmetic against an independent oracle.

CUDA execution, publication ordering and measured performance remain separate
GPU checks. The scheduler-parameter stand-in supplies CUTLASS's resolved padded
geometry; it deliberately does not replicate its swizzle selection policy.
"""
from collections import Counter
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ProducerConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.binary = Path(cls.temp.name) / 'order'
        body = (ROOT / 'csrc/operators/sm103/detail/producer_consumer.cuh').read_text()
        body = '\n'.join(line for line in body.splitlines()
                         if not line.startswith(('#include', '#pragma once')))
        source = r'''
#include <cstdint>
#include <cstdlib>
#include <iostream>
#define CUTLASS_HOST_DEVICE
struct Divmod {
  uint64_t divisor = 1;
  void operator()(uint64_t& q, uint64_t& r, uint64_t v) const {
    q = v / divisor; r = v % divisor;
  }
};
struct Params {
  enum class RasterOrder { AlongM, AlongN };
  RasterOrder raster_order_ = RasterOrder::AlongM;
  uint64_t blocks_per_problem_ = 0;
  int log_swizzle_size_ = 0;
  Divmod divmod_batch_, divmod_cluster_blk_major_;
};
''' + body + r'''
template<int N> void run(int M, int columns, int swizzle, bool along_n) {
  using Ready = fuse::detail::PublishedTile<128, N>;
  using Consumer = fuse::detail::ConsumerTileOrder<Ready, 64, 128>;
  const int mt = (M + 127) / 128, nt = (columns + N - 1) / N;
  const int pm = (mt + swizzle - 1) / swizzle * swizzle;
  const int pn = (nt + swizzle - 1) / swizzle * swizzle;
  Params params;
  params.blocks_per_problem_ = uint64_t(pm) * pn;
  params.divmod_batch_.divisor = params.blocks_per_problem_;
  params.divmod_cluster_blk_major_.divisor = along_n ? pn : pm;
  params.raster_order_ = along_n ? Params::RasterOrder::AlongN : Params::RasterOrder::AlongM;
  while ((1 << params.log_swizzle_size_) < swizzle) ++params.log_swizzle_size_;
  for (uint64_t q = 0; q < params.blocks_per_problem_; ++q) {
    const auto t = fuse::detail::ProducerTileOrder::decode(params, q);
    if (!t.valid || fuse::detail::ProducerTileOrder::linear(params, t.m, t.n, t.batch) != q) std::abort();
  }
  if (fuse::detail::ProducerTileOrder::decode(params, params.blocks_per_problem_).valid) std::abort();
  for (uint64_t u = 0; u < params.blocks_per_problem_ * Consumer::kSlots; ++u) {
    const auto t = Consumer::decode(params, u, M, columns);
    if (t.valid) std::cout << u << ' ' << u / Consumer::kSlots << ' '
        << t.row << ' ' << t.column << ' ' << t.rows << ' ' << t.columns << ' '
        << t.first_m << ' ' << t.last_m << ' ' << t.first_n << ' ' << t.last_n << '\n';
  }
}
int main(int argc, char** argv) {
  if (argc != 6) return 2;
  int n = std::atoi(argv[1]), m = std::atoi(argv[2]), columns = std::atoi(argv[3]);
  int s = std::atoi(argv[4]); bool along_n = std::atoi(argv[5]);
  switch (n) {
    case 64: run<64>(m, columns, s, along_n); break;
    case 128: run<128>(m, columns, s, along_n); break;
    case 160: run<160>(m, columns, s, along_n); break;
    case 192: run<192>(m, columns, s, along_n); break;
    case 256: run<256>(m, columns, s, along_n); break;
    default: return 3;
  }
}
'''
        compiler = shutil.which('clang++') or shutil.which('g++')
        if not compiler: raise unittest.SkipTest('C++ compiler required')
        result = subprocess.run([compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
            '-x', 'c++', '-', '-o', str(cls.binary)], input=source, text=True, capture_output=True)
        if result.returncode: raise AssertionError(result.stderr)

    def test_all_tiles_rasters_swizzles_tails_and_comm_budgets(self):
        for tile_n in (64,128,160,192,256):
            for m,n in ((63,384),(128,4096),(257,896),(1024,7168)):
                for swizzle in (1,2,4,8):
                    for along_n in (False,True):
                        with self.subTest(tile_n=tile_n,m=m,n=n,swizzle=swizzle,along_n=along_n):
                            rows = [tuple(map(int,line.split())) for line in subprocess.check_output(
                                [str(self.binary),str(tile_n),str(m),str(n),str(swizzle),str(int(along_n))],text=True).splitlines()]
                            self.assertEqual(Counter((r[2],r[3]) for r in rows),
                                Counter((x,y) for x in range(0,m,64) for y in range(0,n,128)))
                            pm = ((m+127)//128+swizzle-1)//swizzle*swizzle
                            pn = ((n+tile_n-1)//tile_n+swizzle-1)//swizzle*swizzle
                            # Independent explicit producer-order enumeration.
                            order=[]
                            for group in range(0, pm if along_n else pn, swizzle):
                                for major in range(pn if along_n else pm):
                                    for offset in range(swizzle):
                                        order.append((group+offset,major) if along_n else (major,group+offset))
                            ordinal={tile:i for i,tile in enumerate(order)}
                            for u,q,x,y,h,w,m0,m1,n0,n1 in rows:
                                self.assertEqual((h,w),(min(64,m-x),min(128,n-y)))
                                self.assertEqual((m0,m1,n0,n1),(x//128,(x+h-1)//128,y//tile_n,(y+w-1)//tile_n))
                                self.assertEqual(q,max(ordinal[a,b] for a in range(m0,m1+1) for b in range(n0,n1+1)))
                            for comm in (1,3,7,16,32,147):
                                previous={}
                                for u,q,*_ in rows:
                                    owner=u%(comm*8)
                                    self.assertGreaterEqual(q,previous.get(owner,-1));previous[owner]=q


if __name__ == '__main__':
    unittest.main()
