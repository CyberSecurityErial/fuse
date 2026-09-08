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
  uint64_t compute_grid_size = 1;
  int log_swizzle_size_ = 0;
  Divmod divmod_batch_, divmod_cluster_blk_major_;
};
''' + body + r'''
void run_input(int mt, int nt, int swizzle, bool along_n, int compute, int world,
               int chunks, bool swap_ab = false, bool k_slices = false) {
  const int physical_m = swap_ab ? nt : mt, physical_n = swap_ab ? mt : nt;
  const int pm = (physical_m + swizzle - 1) / swizzle * swizzle;
  const int pn = (physical_n + swizzle - 1) / swizzle * swizzle;
  Params params;
  params.blocks_per_problem_ = uint64_t(pm) * pn;
  params.compute_grid_size = compute < int(params.blocks_per_problem_) ? compute : params.blocks_per_problem_;
  params.divmod_batch_.divisor = params.blocks_per_problem_;
  params.divmod_cluster_blk_major_.divisor = along_n ? pn : pm;
  params.raster_order_ = along_n ? Params::RasterOrder::AlongN : Params::RasterOrder::AlongM;
  while ((1 << params.log_swizzle_size_) < swizzle) ++params.log_swizzle_size_;
  const auto order = fuse::detail::A2AInputTileOrder::make(params, mt, swap_ab);
  for (uint64_t task = 0; task < uint64_t(mt) * world * chunks; ++task) {
    const auto t = order.decode(task, world, chunks, k_slices);
    std::cout << t.m << ' ' << t.peer << ' ' << t.chunk << '\n';
  }
}
template<int N> void run(int M, int columns, int swizzle, bool along_n, int rank) {
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
  const auto rotation = rank < 0 ? fuse::detail::NBandSwizzle{}
      : fuse::detail::NBandSwizzle::make(params, rank);
  for (uint64_t q = 0; q < params.blocks_per_problem_; ++q) {
    const auto t = fuse::detail::ProducerTileOrder::decode(params, q, rotation);
    if (!t.valid || fuse::detail::ProducerTileOrder::linear(params, t.m, t.n, t.batch, rotation) != q) std::abort();
  }
  if (fuse::detail::ProducerTileOrder::decode(params, params.blocks_per_problem_).valid) std::abort();
  for (uint64_t u = 0; u < params.blocks_per_problem_ * Consumer::kSlots; ++u) {
    const auto t = Consumer::decode(params, u, M, columns, rotation);
    if (t.valid) std::cout << u << ' ' << u / Consumer::kSlots << ' '
        << t.row << ' ' << t.column << ' ' << t.rows << ' ' << t.columns << ' '
        << t.first_m << ' ' << t.last_m << ' ' << t.first_n << ' ' << t.last_n << '\n';
  }
}
int main(int argc, char** argv) {
  if (argc == 6 && argv[1][0] == 'g') {
    const auto s = fuse::detail::A2AInputTransferShape::make(
        std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]),
        std::atoi(argv[5]), 48 * 1024);
    std::cout << s.rows << ' ' << s.peer_k << ' ' << s.ready_k << ' '
        << s.copy_k << ' ' << s.inner_u64 << '\n';
    return 0;
  }
  if (argc == 9 || argc == 11) {
    run_input(std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]),
        std::atoi(argv[5]), std::atoi(argv[6]), std::atoi(argv[7]), std::atoi(argv[8]),
        argc == 11 && std::atoi(argv[9]), argc == 11 && std::atoi(argv[10]));
    return 0;
  }
  if (argc != 6 && argc != 7) return 2;
  int n = std::atoi(argv[1]), m = std::atoi(argv[2]), columns = std::atoi(argv[3]);
  int s = std::atoi(argv[4]); bool along_n = std::atoi(argv[5]);
  int rank = argc == 7 ? std::atoi(argv[6]) : -1;
  switch (n) {
    case 64: run<64>(m, columns, s, along_n, rank); break;
    case 128: run<128>(m, columns, s, along_n, rank); break;
    case 160: run<160>(m, columns, s, along_n, rank); break;
    case 192: run<192>(m, columns, s, along_n, rank); break;
    case 256: run<256>(m, columns, s, along_n, rank); break;
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
        self.check_orders((-1,))

    def test_rank_rotation_all_tiles_rasters_tails_and_comm_budgets(self):
        self.check_orders(range(8))

    def test_a2a_input_first_use_windows(self):
        # Oracle enumerates real GEMM work, deduplicates M by FIRST-use wave,
        # then supplies peers in K-consumption order. No inverse formula here.
        for mt, nt in ((1, 1), (3, 5), (13, 7), (128, 16), (128, 28),
                       (128, 32), (128, 56), (128, 64)):
            for swizzle in (1, 2, 4, 8):
                for along_n in (False, True):
                    pm = (mt + swizzle - 1) // swizzle * swizzle
                    pn = (nt + swizzle - 1) // swizzle * swizzle
                    tiles = [(g + o, a) if along_n else (a, g + o)
                             for g in range(0, pm if along_n else pn, swizzle)
                             for a in range(pn if along_n else pm)
                             for o in range(swizzle)]
                    for compute in (1, 3, 8, 124, 132, 140, 147):
                        compute_grid = min(compute, len(tiles))
                        waves, seen = {}, set()
                        for i, (m, n) in enumerate(tiles):
                            if m < mt and n < nt and m not in seen:
                                waves.setdefault(i // compute_grid, []).append(m)
                                seen.add(m)
                        for world, chunks in ((4, 1), (8, 6), (8, 11)):
                            with self.subTest(mt=mt, nt=nt, swizzle=swizzle,
                                              along_n=along_n, compute=compute,
                                              world=world, chunks=chunks):
                                rows = [tuple(map(int, line.split())) for line in subprocess.check_output(
                                    [str(self.binary), 'input', str(mt), str(nt), str(swizzle),
                                     str(int(along_n)), str(compute), str(world), str(chunks)],
                                    text=True).splitlines()]
                                expected = [(m, p, c) for ms in waves.values()
                                            for p in range(world) for m in ms for c in range(chunks)]
                                self.assertEqual(rows, expected)
                                self.assertEqual(len(set(rows)), mt * world * chunks)
                                # Persistent comm workers partition this queue exactly once.
                                for comm in (1, 8, 16, 24):
                                    visited = [t for slot in range(comm * 4)
                                               for t in rows[slot::comm * 4]]
                                    self.assertEqual(Counter(visited), Counter(expected))

    def test_aligned_frontiers_include_supergroup_orientation_and_padding(self):
        # Enumerate the ACTUAL physical supergroups, then map into original
        # sequence/projection coordinates. This oracle shares no inverse math.
        from itertools import product
        for (mt, nt), swizzle, along_n, compute, swap in product(
                ((7, 5), (33, 16)), (1, 4, 8), (False, True),
                (3, 32, 132, 140), (False, True)):
            pm, pn = (nt, mt) if swap else (mt, nt)
            pm = (pm + swizzle - 1) // swizzle * swizzle
            pn = (pn + swizzle - 1) // swizzle * swizzle
            physical = [(g + o, a) if along_n else (a, g + o)
                        for g in range(0, pm if along_n else pn, swizzle)
                        for a in range(pn if along_n else pm) for o in range(swizzle)]
            waves, seen = {}, set()
            for i, (x, y) in enumerate(physical):
                m, n = (y, x) if swap else (x, y)
                if m < mt and n < nt and m not in seen:
                    waves.setdefault(i // min(compute, len(physical)), []).append(m)
                    seen.add(m)
            for world, slices in ((4, 1), (8, 8)):
                actual = [tuple(map(int, line.split())) for line in subprocess.check_output(
                    [str(self.binary), 'input', str(mt), str(nt), str(swizzle),
                     str(int(along_n)), str(compute), str(world), str(slices),
                     str(int(swap)), '1'], text=True).splitlines()]
                expected = [(m, p, k) for ms in waves.values() for p in range(world)
                            for k in range(slices) for m in ms]
                self.assertEqual(actual, expected)
                self.assertEqual(len(set(actual)), mt * world * slices)
                for comm in (8, 16, 24):
                    owners = [t for slot in range(comm * 4) for t in actual[slot::comm * 4]]
                    self.assertEqual(Counter(owners), Counter(expected))

    def test_rectangular_transfer_geometry_capacity_and_exact_k_coverage(self):
        from itertools import product
        for rows, peer in product((32, 128), (128, 1024, 1536, 1792, 2048, 3584, 4096, 7168)):
            ready = peer if rows == 32 else 128
            shape = tuple(map(int, subprocess.check_output(
                [str(self.binary), 'geometry', str(rows), str(peer), str(ready), '64'],
                text=True).split()))
            r, p, k, copy, inner = shape
            self.assertEqual((r, p, k), (rows, peer, ready))
            self.assertGreater(copy, 0)
            self.assertLessEqual(rows * copy * 2, 48 * 1024)
            self.assertEqual(copy % 64, 0)
            self.assertEqual(ready % copy, 0)
            self.assertEqual(peer % (inner * 4), 0)
            self.assertEqual(copy % (inner * 4), 0)
            self.assertLessEqual(copy // (inner * 4), 256)
            rectangles = [(s + o, s + o + copy) for s in range(0, peer, ready)
                          for o in range(0, ready, copy)]
            self.assertEqual([v for a, b in rectangles for v in range(a, b)], list(range(peer)))
            # Each row preserves source peer-K stride and destination full-K
            # stride, independently of the u64 descriptor factorization.
            for world in (4, 8):
                for row in (0, rows - 1):
                    for peer_slot in (0, world - 1):
                        for start, end in rectangles:
                            for col in (start, end - 1):
                                group, inner_element = divmod(col, inner * 4)
                                src = row * peer + group * inner * 4 + inner_element
                                dst = row * peer * world + (peer_slot * peer // (inner * 4) + group) * inner * 4 + inner_element
                                self.assertEqual(src, row * peer + col)
                                self.assertEqual(dst, row * peer * world + peer_slot * peer + col)
        for rows, peer, ready, tile in ((0, 1024, 128, 64), (128, 64, 128, 64),
                                       (32, 1024, 1024, 0), (257, 1024, 128, 64)):
            shape = tuple(map(int, subprocess.check_output(
                [str(self.binary), 'geometry', str(rows), str(peer), str(ready), str(tile)],
                text=True).split()))
            self.assertEqual(shape, (0, 0, 0, 0, 0))

    def check_orders(self, ranks):
        for tile_n in (64,128,160,192,256):
            for m,n in ((63,384),(128,4096),(257,896),(1024,7168)):
                for swizzle in (1,2,4,8):
                    for along_n, rank in ((a,r) for a in (False,True) for r in ranks):
                        with self.subTest(tile_n=tile_n,m=m,n=n,swizzle=swizzle,along_n=along_n,rank=rank):
                            rows = [tuple(map(int,line.split())) for line in subprocess.check_output(
                                [str(self.binary),str(tile_n),str(m),str(n),str(swizzle),str(int(along_n)),str(rank)],text=True).splitlines()]
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
                            offset = (rank % (pn // swizzle)) * swizzle if rank >= 0 else 0
                            ordinal={(x,(y+offset)%pn):i for i,(x,y) in enumerate(order)}
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
