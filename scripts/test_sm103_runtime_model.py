"""Host-only C++ / Python service-model parity; no CUDA compiler, GPU or cloud."""

import itertools
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks/sm103'))
import fused_model as model


INPUT_FIELDS = ('m', 'n', 'k', 'world', 'sm_count', 'comm_ctas', 'tile_m', 'tile_n', 'tile_k',
                'resolved_swizzle', 'copy_chunks', 'copy_slots', 'cohort_m_tiles', 'calibrated_compute_ctas', 'cluster_ctas')
OUTPUT_FIELDS = ('compute_ctas', 'critical_worker', 'scheduled_work_tiles', 'valid_work_tiles', 'integer_waves',
    'score_us', 'ideal_overlap_us', 'critical_path_us', 'compute_finish_us', 'copy_finish_us', 'first_ready_us',
    'wave_compute_service_us', 'strided_compute_service_us', 'critical_worker_service_us',
    'critical_worker_feed_wait_us', 'critical_worker_initial_wait_us', 'critical_worker_later_feed_wait_us',
    'exposed_feed_us', 'worker_wait_sum_us', 'feed_phase_ideal_compute_us', 'feed_phase_predicted_finish_us',
    'feed_phase_demand_gb_s', 'effective_delivery_gb_s', 'production_consumption_ratio')


def arguments(**changes):
    row = dict(m=32768, n=8192, k=8192, world=4, sm_count=148, comm_ctas=16,
               tile_m=128, tile_n=256, tile_k=64, cluster_ctas=1, raster='along_m', resolved_swizzle=8,
               tile_cycle_us=32.125, copy_bandwidth_gb_s=444.5)
    row.update(changes)
    copy_rows = min(128, 48 * 1024 // (2 * row['k'] // row['world']))
    chunks = model.ceil_div(128, max(1, copy_rows))
    row.setdefault('copy_chunks', chunks)
    row.setdefault('copy_slots', 4 * row['comm_ctas'])
    row.setdefault('cohort_m_tiles', max(1, row['copy_slots'] // chunks))
    row.setdefault('copy_chunk_bytes', [2 * row['k'] // row['world'] * min(copy_rows, 128 - i * copy_rows)
                                         for i in range(chunks)])
    sw = row['resolved_swizzle']
    pm = model.ceil_div(row['m'] // 128, sw) * sw
    pn = model.ceil_div(model.ceil_div(row['n'], row['tile_n']), sw) * sw
    row.setdefault('calibrated_compute_ctas', min(pm * pn, row['sm_count'] - row['comm_ctas']))
    return row


class RuntimeModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('clang++') or shutil.which('g++')
        if not compiler:
            raise unittest.SkipTest('A local C++17 compiler is required')
        cls.directory = tempfile.TemporaryDirectory(prefix='fuse-runtime-model-')
        cls.addClassCleanup(cls.directory.cleanup)
        root = Path(cls.directory.name)
        (root / 'cutlass').mkdir()
        (root / 'cutlass/cutlass.h').write_text('#pragma once\n#define CUTLASS_HOST_DEVICE\n')
        source = r'''
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include "performance_model.cuh"
#if __has_include("model_calibration.cuh")
#include "model_calibration.cuh"
static_assert(sizeof(fuse::detail::kOprojCalibrationPoints) > 0);
#endif
int main() {
  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream stream(line);
    fuse::detail::OProjModelInput input;
    int raster;
    std::string cycle, bandwidth;
    size_t count;
    if (!(stream INPUT_READ >> raster >> cycle >> bandwidth >> count)) return 2;
    input.raster = static_cast<fuse::detail::OProjModelRaster>(raster);
    input.tile_cycle_us = std::stod(cycle);
    input.copy_bandwidth_gb_s = std::stod(bandwidth);
    input.copy_chunk_bytes.resize(count);
    for (auto& bytes : input.copy_chunk_bytes) if (!(stream >> bytes)) return 3;
    auto result = fuse::detail::score_oproj_schedule(input);
    std::cout << std::setprecision(17) << int(result.status) OUTPUT_WRITE << '\n';
  }
}
'''.replace('INPUT_READ', ''.join(' >> input.' + field for field in INPUT_FIELDS)).replace(
            'OUTPUT_WRITE', ''.join(" << ' ' << result." + field for field in OUTPUT_FIELDS))
        cls.binary = root / 'model'
        include = Path(__file__).resolve().parents[1] / 'csrc/operators/sm103/detail'
        result = subprocess.run([compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                                 '-I', str(root), '-I', str(include), '-x', 'c++', '-', '-o', str(cls.binary)],
                                input=source, text=True, capture_output=True, timeout=30)
        if result.returncode:
            raise AssertionError(result.stderr)

    def native(self, rows):
        lines = []
        for row in rows:
            chunks = row['copy_chunk_bytes'] or []
            raster = row['raster'] if isinstance(row['raster'], int) else int(row['raster'] == 'along_n')
            values = [row[key] for key in INPUT_FIELDS] + [raster, row['tile_cycle_us'],
                     row['copy_bandwidth_gb_s'], len(chunks)] + chunks
            lines.append(' '.join(map(str, values)))
        result = subprocess.run([str(self.binary)], input='\n'.join(lines) + '\n',
                                text=True, capture_output=True, check=True, timeout=30)
        actual = []
        for line in result.stdout.splitlines():
            fields = list(map(float, line.split()))
            actual.append(dict(status=int(fields[0]), **dict(zip(OUTPUT_FIELDS, fields[1:]))))
        self.assertEqual(len(actual), len(rows))
        return actual

    def assert_parity(self, rows):
        for row, native in zip(rows, self.native(rows)):
            with self.subTest(shape=(row['m'], row['n'], row['k']), raster=row['raster'],
                              swizzle=row['resolved_swizzle'], comm=row['comm_ctas']):
                self.assertEqual(native['status'], 0)
                params = {key: value for key, value in row.items() if key not in ('calibrated_compute_ctas', 'cluster_ctas')}
                params['copy_chunk_bytes'] = params['copy_chunk_bytes'] or None
                python = model.score_oproj_schedule(**params, launch_us=0., copy_start_us=0., tail_us=0.,
                                                    service_basis='amortized_full_boundary')
                for key in OUTPUT_FIELDS:
                    self.assertTrue(math.isclose(native[key], python[key], rel_tol=1e-11, abs_tol=1e-8),
                                    f'{key}: native {native[key]} != Python {python[key]}')

    def test_two_rasters_swizzles_cp_and_budget_curves_match(self):
        rows = [arguments(world=world, raster=raster, resolved_swizzle=sw, comm_ctas=comm,
                          k=k, tile_cycle_us={8: 28., 16: 31., 32: 40.}[comm])
                for world, raster, sw, comm, k in itertools.product(
                    (4, 8), ('along_m', 'along_n'), (4, 8), (8, 16, 32), (8192, 14336))]
        self.assert_parity(rows)

    def test_padding_small_worker_grid_and_both_supported_tiles(self):
        rows = [arguments(m=mt*128, n=nt*tn-16, k=8192, tile_n=tn, tile_k=tk,
                          raster=raster, resolved_swizzle=sw, world=world)
                for mt, nt, tn, tk, raster, sw, world in itertools.product(
                    (3, 9), (5, 7), (128, 256), (64, 128), ('along_m', 'along_n'), (4, 8), (4, 8))]
        self.assert_parity(rows)

    def test_swizzle_padding_executes_full_service_including_m_padding(self):
        # Three real M tiles and five real N tiles are scheduled as 4 x 8.
        # With one worker, all 32 MMA pipelines execute, not just 15 outputs.
        rows = [arguments(m=3*128, n=5*256, sm_count=17, comm_ctas=16,
                          resolved_swizzle=4, raster=raster, tile_cycle_us=10.,
                          copy_bandwidth_gb_s=1e12)
                for raster in ('along_m', 'along_n')]
        for result in self.native(rows):
            self.assertEqual(result['status'], 0)
            self.assertEqual(result['valid_work_tiles'], 15)
            self.assertEqual(result['scheduled_work_tiles'], 32)
            self.assertEqual(result['integer_waves'], 32)
            self.assertEqual(result['strided_compute_service_us'], 320.)

    def test_n_padding_still_waits_and_m_padding_does_not_zero_first_ready(self):
        # Same padded grid and same A-copy queue: extending N from five to
        # eight actual tiles changes useful outputs but not MMA/ready events.
        rows = [arguments(m=3*128, n=nt*256, raster=raster,
                          resolved_swizzle=4, copy_bandwidth_gb_s=1.)
                for raster in ('along_m', 'along_n') for nt in (5, 8)]
        results = self.native(rows)
        for left, right in zip(results[::2], results[1::2]):
            self.assertGreater(left['first_ready_us'], 0)
            for key in ('score_us', 'worker_wait_sum_us', 'compute_finish_us',
                        'strided_compute_service_us', 'integer_waves', 'first_ready_us'):
                self.assertEqual(left[key], right[key], key)

    def test_equal_chunks_and_vector_slot_budget(self):
        rows = [arguments(m=4096, n=4096, copy_chunk_bytes=None, copy_chunks=5, copy_slots=16,
                          cohort_m_tiles=cohort, raster=raster, resolved_swizzle=sw)
                for cohort, raster, sw in itertools.product((1, 3, 11), ('along_m', 'along_n'), (1, 2, 4, 8))]
        self.assert_parity(rows)

    def test_invalid_inputs_and_budget_mismatch_fail_before_allocation(self):
        base = arguments()
        changes = [dict(m=0), dict(n=-1), dict(k=8193), dict(m=2**31), dict(world=2),
                   dict(sm_count=0), dict(comm_ctas=148), dict(tile_m=64), dict(tile_k=32),
                   dict(tile_n=192), dict(cluster_ctas=2), dict(raster=2), dict(resolved_swizzle=3),
                   dict(copy_slots=17), dict(cohort_m_tiles=0), dict(copy_chunks=0),
                   dict(calibrated_compute_ctas=148), dict(calibrated_compute_ctas=0),
                   dict(tile_cycle_us=0), dict(tile_cycle_us=float('nan')),
                   dict(copy_bandwidth_gb_s=float('inf')), dict(copy_bandwidth_gb_s=-1),
                   dict(copy_chunk_bytes=[0]*base['copy_chunks']), dict(copy_chunk_bytes=[1]),
                   dict(copy_chunk_bytes=[2**64-1]*base['copy_chunks'])]
        self.assertTrue(all(row['status'] != 0 for row in self.native([base | change for change in changes])))

    def test_work_bound_and_nonfinite_arithmetic_fail_explicitly(self):
        base = arguments()
        rows = [base | dict(m=128*1000000, n=256*1000000),
                base | dict(copy_chunks=1000001, copy_chunk_bytes=None),
                base | dict(copy_slots=1000001, comm_ctas=1000001, sm_count=1000004),
                base | dict(copy_bandwidth_gb_s=1e-307), base | dict(tile_cycle_us=1e308)]
        self.assertTrue(all(row['status'] != 0 for row in self.native(rows)))


if __name__ == '__main__':
    unittest.main()
