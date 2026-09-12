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
#include <algorithm>
#include <vector>
#define CUTLASS_HOST
#define CUTLASS_HOST_DEVICE
namespace cutlass {
struct FastDivmodU64 {
  uint64_t divisor = 1;
  FastDivmodU64() = default;
  explicit FastDivmodU64(uint64_t d) : divisor(d) {}
  uint64_t divide(uint64_t v) const { return v / divisor; }
};
}
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
void run_input(int mt, int nt, int swizzle, bool along_n, int compute, int world, int chunks, int comm_slots,
               int h = 0, int g = 0, bool window_test = false) {
  const int pm = (mt + swizzle - 1) / swizzle * swizzle;
  const int pn = (nt + swizzle - 1) / swizzle * swizzle;
  Params params;
  params.blocks_per_problem_ = uint64_t(pm) * pn;
  params.compute_grid_size = compute < int(params.blocks_per_problem_) ? compute : params.blocks_per_problem_;
  params.divmod_batch_.divisor = params.blocks_per_problem_;
  params.divmod_cluster_blk_major_.divisor = along_n ? pn : pm;
  params.raster_order_ = along_n ? Params::RasterOrder::AlongN : Params::RasterOrder::AlongM;
  while ((1 << params.log_swizzle_size_) < swizzle) ++params.log_swizzle_size_;
  const auto rectangles = fuse::detail::OprojTileOrder::make(params, h, g);
  auto order = fuse::detail::A2AInputTileOrder::make(params, mt, rectangles);
  order.ready_group_m_tiles = comm_slots / chunks > 0 ? comm_slots / chunks : 1;
  if (window_test) {
    // Independently recover first use by enumerating all real consumers. The
    // Python oracle below checks the whole tile sequence, not only its inverse.
    params.blocks_per_problem_ *= 2;
    std::vector<uint64_t> first(mt, params.blocks_per_problem_);
    for (uint64_t q = 0; q < params.blocks_per_problem_; ++q) {
      const auto t = rectangles.decode(params, q);
      if (!t.valid || rectangles.linear(params, t.m, t.n, t.batch) != q) std::abort();
      std::cout << "T " << t.m << ' ' << t.n << ' ' << t.batch << '\n';
      if (t.batch == 0 && t.m < mt && t.n < nt) first[t.m] = std::min(first[t.m], q);
    }
    if (rectangles.decode(params, params.blocks_per_problem_).valid) std::abort();
    for (int m = 0; m < mt; ++m) {
      if (order.first_use(m) != first[m]) std::abort();
    }
    if (!std::is_sorted(first.begin(), first.end())) std::abort();
    for (uint64_t q = 0; q <= params.blocks_per_problem_; ++q) {
      const int expected = std::lower_bound(first.begin(), first.end(), q) - first.begin();
      if (order.lower_bound(q) != expected) std::abort();
    }
  }
  for (uint64_t task = 0; task < uint64_t(mt) * world * chunks; ++task) {
    const auto t = order.decode(task, world, chunks);
    if (window_test) std::cout << "I ";
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
  if (argc == 12) {
    run_input(std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]),
        std::atoi(argv[5]), std::atoi(argv[6]), std::atoi(argv[7]), std::atoi(argv[8]),
        std::atoi(argv[9]), std::atoi(argv[10]), std::atoi(argv[11]), true);
    return 0;
  }
  if (argc == 10) {
    run_input(std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]),
        std::atoi(argv[5]), std::atoi(argv[6]), std::atoi(argv[7]), std::atoi(argv[8]),
        std::atoi(argv[9]));
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
        # then follows the resolved raster: multi-N-band AlongM completes ready
        # cohorts; AlongN and single-N-band AlongM retain the group/peer diagonal.
        # No prefix inversion formula from the device implementation is used.
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
                        for world, chunks, slots in ((4, 1, 4), (4, 11, 32),
                                                      (4, 11, 64), (4, 16, 64),
                                                      (8, 6, 8), (8, 22, 16)):
                            with self.subTest(mt=mt, nt=nt, swizzle=swizzle,
                                              along_n=along_n, compute=compute,
                                              world=world, chunks=chunks, slots=slots):
                                rows = [tuple(map(int, line.split())) for line in subprocess.check_output(
                                    [str(self.binary), 'input', str(mt), str(nt), str(swizzle),
                                     str(int(along_n)), str(compute), str(world), str(chunks), str(slots)],
                                    text=True).splitlines()]
                                expected = []
                                for ms in waves.values():
                                    use_cohorts = not along_n and nt > swizzle
                                    width = (max(1, slots // chunks) if use_cohorts else
                                             max(swizzle, (len(ms) + world - 1) // world))
                                    groups = [ms[i:i + width] for i in range(0, len(ms), width)]
                                    if not use_cohorts:
                                        for diagonal in range(len(groups) + world - 1):
                                            for peer in range(world):
                                                group = diagonal - peer
                                                if 0 <= group < len(groups):
                                                    expected.extend((m, peer, c) for m in groups[group]
                                                                    for c in range(chunks))
                                    else:
                                        for group in groups:
                                            for peer in range(world):
                                                expected.extend((m, peer, c) for m in group
                                                                for c in range(chunks))
                                self.assertEqual(rows, expected)
                                self.assertEqual(len(set(rows)), mt * world * chunks)
                                # Persistent comm workers partition this queue exactly once.
                                for workers in (1, slots, 96):
                                    visited = [t for slot in range(workers)
                                               for t in rows[slot::workers]]
                                    self.assertEqual(Counter(visited), Counter(expected))

    def test_a2a_cohorts_schedule_all_peers_before_next_group(self):
        # Representative AlongM cases include partial first-window cohorts.
        # These assertions constrain queue priority, not asynchronous completion.
        for nt, swizzle, compute, chunks, slots, width, window in (
                (16, 8, 132, 11, 64, 5, 17),
                (32, 8, 140, 11, 32, 2, 18),
                (28, 4, 132, 16, 64, 4, 33),
                (56, 4, 140, 22, 32, 1, 35),
                (64, 4, 140, 4, 64, 16, 35)):
            with self.subTest(nt=nt, swizzle=swizzle, compute=compute,
                              chunks=chunks, slots=slots):
                rows = [tuple(map(int, line.split())) for line in subprocess.check_output(
                    [str(self.binary), 'input', '256', str(nt), str(swizzle), '0',
                     str(compute), '4', str(chunks), str(slots)],
                    text=True).splitlines()]
                positions = {task: i for i, task in enumerate(rows)}
                expected_window = [(m, peer, chunk)
                                   for begin in range(0, window, width)
                                   for peer in range(4)
                                   for m in range(begin, min(begin + width, window))
                                   for chunk in range(chunks)]
                self.assertEqual(rows[:len(expected_window)], expected_window)
                self.assertLess(positions[width - 1, 3, chunks - 1], positions[width, 0, 0])
                self.assertEqual(rows[len(expected_window)], (window, 0, 0))
                for m in range(256):
                    for peer in range(3):
                        self.assertLess(positions[m, peer, chunks - 1], positions[m, peer + 1, 0])
                self.assertEqual(len(positions), 256 * 4 * chunks)

    def test_a2a_copy_slots_affect_along_m_but_not_along_n(self):
        for along_n in (False, True):
            with self.subTest(along_n=along_n):
                queues = []
                for slots in (1, 8, 32, 64):
                    queues.append([tuple(map(int, line.split())) for line in subprocess.check_output(
                        [str(self.binary), 'input', '128', '28', '4', str(int(along_n)),
                         '132', '4', '11', str(slots)], text=True).splitlines()])
                if along_n:
                    for rows in queues[1:]:
                        self.assertEqual(rows, queues[0])
                else:
                    # Fewer slots than chunks clamp the M cohort to one, then
                    # larger budgets permit two/five M without altering coverage.
                    self.assertEqual(queues[0], queues[1])
                    self.assertNotEqual(queues[0], queues[2])
                    self.assertNotEqual(queues[2], queues[3])
                    for rows in queues[1:]:
                        self.assertEqual(Counter(rows), Counter(queues[0]))

    def test_a2a_single_n_band_along_m_preserves_diagonal(self):
        for nt, swizzle in ((1, 1), (3, 4), (4, 4), (7, 8), (8, 8)):
            with self.subTest(nt=nt, swizzle=swizzle):
                queues = []
                for slots in (1, 32, 64):
                    queues.append([tuple(map(int, line.split())) for line in subprocess.check_output(
                        [str(self.binary), 'input', '128', str(nt), str(swizzle), '0',
                         '132', '4', '11', str(slots)], text=True).splitlines()])
                for rows in queues[1:]:
                    self.assertEqual(rows, queues[0])
                self.assertEqual(len(set(queues[0])), 128 * 4 * 11)

    def test_a2a_single_group_preserves_peer_major(self):
        rows = [tuple(map(int, line.split())) for line in subprocess.check_output(
            [str(self.binary), 'input', '4', '64', '4', '1', '140', '4', '4', '64'],
            text=True).splitlines()]
        self.assertEqual(rows, [(m, p, c) for p in range(4) for m in range(4) for c in range(4)])

    def test_oproj_rectangle_order_and_first_use_windows(self):
        # Cover both padded tails (M/N not multiples of the resolved swizzle)
        # and clipped rectangles (H/P smaller or larger than actual extents).
        for mt, nt, h, group_n in ((3, 5, 64, 4), (67, 13, 64, 4),
                                   (133, 17, 128, 4), (129, 13, 64, 8),
                                   (67, 5, 2, 2), (133, 17, 0, 0)):
            for swizzle in (1, 2, 4, 8):
                for along_n in (False, True):
                    pm = (mt + swizzle - 1) // swizzle * swizzle
                    pn = (nt + swizzle - 1) // swizzle * swizzle
                    height, width = (h, group_n) if h else (pm, pn)
                    effective_swizzle = min(swizzle, height if along_n else width)
                    expected_tiles = []
                    for mb in range(0, pm, height):
                        for nb in range(0, pn, width):
                            ms, ns = min(height, pm - mb), min(width, pn - nb)
                            for band in range(0, ms if along_n else ns, effective_swizzle):
                                for major in range(ns if along_n else ms):
                                    for offset in range(effective_swizzle):
                                        expected_tiles.append((mb + band + offset, nb + major)
                                            if along_n else (mb + major, nb + band + offset))
                    for compute in (84, 100, 116, 128):
                        world, chunks, slots = (4, 3, 12) if compute < 116 else (8, 4, 80)
                        with self.subTest(mt=mt, nt=nt, h=h, p=group_n, sw=swizzle,
                                          along_n=along_n, compute=compute):
                            lines = subprocess.check_output([
                                str(self.binary), 'window', str(mt), str(nt), str(swizzle),
                                str(int(along_n)), str(compute), str(world), str(chunks),
                                str(slots), str(h), str(group_n)], text=True).splitlines()
                            tiles = [tuple(map(int, line.split()[1:])) for line in lines if line.startswith('T ')]
                            rows = [tuple(map(int, line.split()[1:])) for line in lines if line.startswith('I ')]
                            self.assertEqual(tiles, [(m, n, b) for b in range(2) for m, n in expected_tiles])
                            self.assertEqual(len(set(tiles)), pm * pn * 2)
                            grid = min(compute, len(expected_tiles))
                            waves, seen = {}, set()
                            for i, (m, n) in enumerate(expected_tiles):
                                if m < mt and n < nt and m not in seen:
                                    waves.setdefault(i // grid, []).append(m)
                                    seen.add(m)
                            expected = []
                            for ms in waves.values():
                                cohort = not along_n and pn > effective_swizzle
                                size = (max(1, slots // chunks) if cohort else
                                        max(effective_swizzle, (len(ms) + world - 1) // world))
                                groups = [ms[i:i + size] for i in range(0, len(ms), size)]
                                if cohort:
                                    expected.extend((m, peer, c) for members in groups
                                                    for peer in range(world) for m in members
                                                    for c in range(chunks))
                                else:
                                    for diagonal in range(len(groups) + world - 1):
                                        for peer in range(world):
                                            g = diagonal - peer
                                            if 0 <= g < len(groups):
                                                expected.extend((m, peer, c) for m in groups[g]
                                                                for c in range(chunks))
                            self.assertEqual(rows, expected)
                            self.assertEqual(len(set(rows)), mt * world * chunks)
                            for workers in (1, slots, 148):
                                self.assertEqual(Counter(t for w in range(workers) for t in rows[w::workers]),
                                                 Counter(expected))

    def test_oproj_qwen_first_two_waves_reuse_four_w_panels(self):
        for h in (64, 128):
            lines = subprocess.check_output([str(self.binary), 'window', '128', '16', '8', '1',
                '128', '8', '4', '80', str(h), '4'], text=True).splitlines()
            tiles = [tuple(map(int, line.split()[1:])) for line in lines if line.startswith('T ')]
            for wave in (0, 1):
                self.assertEqual(set(tiles[wave * 128:(wave + 1) * 128]),
                    {(m, n, 0) for m in range(wave * 32, (wave + 1) * 32) for n in range(4)})

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


class ColumnCopyTests(unittest.TestCase):
    """Execute the communication helper; CUDA descriptor checks stay separate."""

    PEER_WIDTHS = (64, 128, 192, 256, 512, 1024, 1536, 1792, 2048, 3584, 4096)

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.binary = Path(cls.temp.name) / 'column-copy'
        body = (ROOT / 'csrc/operators/sm103/detail/a2a_gemm.cuh').read_text()
        start = body.index('  struct ColumnCopy {')
        stop = body.index('\n  };', start) + len('\n  };')
        source = r'''
#include <cstdint>
#include <cstdlib>
#include <iostream>
#define CUTLASS_HOST_DEVICE
''' + body[start:stop] + r'''
int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    const int peer_k = std::atoi(argv[i]);
    if (peer_k <= 0) return 2;
    const auto copy = ColumnCopy::make(peer_k);
    std::cout << peer_k << ' ' << copy.width << ' ' << copy.chunks << ' ' << copy.tail;
    for (int chunk = 0; chunk < copy.chunks; ++chunk) {
      std::cout << ' ' << copy.width_at(chunk);
    }
    std::cout << '\n';
  }
}
'''
        compiler = shutil.which('clang++') or shutil.which('g++')
        if not compiler:
            raise unittest.SkipTest('C++ compiler required')
        result = subprocess.run(
            [compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
             '-x', 'c++', '-', '-o', str(cls.binary)],
            input=source, text=True, capture_output=True)
        if result.returncode:
            raise AssertionError(result.stderr)
        output = subprocess.check_output(
            [str(cls.binary), *map(str, cls.PEER_WIDTHS)], text=True)
        cls.copies = {row[0]: row[1:] for line in output.splitlines()
                      if (row := tuple(map(int, line.split())))}

    def test_stage_capacity_and_exact_tail(self):
        self.assertEqual(set(self.copies), set(self.PEER_WIDTHS))
        for peer_k, (width, chunks, tail, *widths) in self.copies.items():
            with self.subTest(peer_k=peer_k):
                self.assertEqual(width, min(peer_k, 192))
                expected = [min(192, peer_k - start)
                            for start in range(0, peer_k, 192)]
                self.assertEqual(widths, expected)
                self.assertEqual(chunks, len(expected))
                self.assertEqual(tail, peer_k % width)
                self.assertEqual(sum(widths), peer_k)
                for actual in widths:
                    self.assertGreater(actual, 0)
                    self.assertLessEqual(128 * actual * 2, 48 * 1024)
                    self.assertEqual(actual % 64, 0)

    def test_cp4_cp8_exact_cover_without_peer_overlap(self):
        for peer_k, (width, _, _, *widths) in self.copies.items():
            for world in (4, 8):
                with self.subTest(peer_k=peer_k, world=world):
                    coverage = Counter()
                    for peer in range(world):
                        peer_begin, peer_end = peer * peer_k, (peer + 1) * peer_k
                        for chunk, actual in enumerate(widths):
                            begin = peer_begin + chunk * width
                            end = begin + actual
                            self.assertGreaterEqual(begin, peer_begin)
                            self.assertLessEqual(end, peer_end)
                            coverage.update(range(begin, end))
                    # Every row of the 128-row rectangle uses these same K
                    # intervals with row stride world*peer_k, not copy.width.
                    self.assertEqual(coverage, Counter(range(world * peer_k)))


if __name__ == '__main__':
    unittest.main()
