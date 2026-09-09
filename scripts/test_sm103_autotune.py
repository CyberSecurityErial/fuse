"""CPU selector/compiled-calibration parity; no CUDA, GPU, cloud or raw result dependency."""

import copy
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest

import plan_sm103_oproj as planner

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('m', 'n', 'k', 'world', 'sm_count', 'device', 'policy_index', 'raster',
          'max_swizzle_size', 'swizzle', 'comm_ctas')


def request(**changes):
    return dict(m=32768, n=8192, k=8192, world=4, sm_count=148, device=0,
                policy_index=-1, raster=-1, max_swizzle_size=8, swizzle=0, comm_ctas=0) | changes


def compiled_services():
    """Adapt the checked-in C/R table to Python planner inputs, without F values.

    Diagnostic provenance placeholders are not measured observations; selection
    uses exactly the two service values and actual compute budget from C++.
    """
    header = (ROOT / 'csrc/operators/sm103/detail/model_calibration.cuh').read_text()
    anchors = {}
    for line in re.findall(r'^  \{([^}]+)\},$', header, re.MULTILINE):
        cells = line.split(', ')
        world, policy = map(int, cells[:2])
        raster = 'along_n' if cells[2] == 'true' else 'along_m'
        swizzle, comm, k, m, n, compute = map(int, cells[3:9])
        cycle, bandwidth = map(float, cells[9:])
        key = (world, 148, planner.POLICIES[policy], raster, swizzle, swizzle, comm)
        anchors.setdefault(key, {})[k] = dict(m=m, n=n, k=k, compute_ctas=compute,
            tile_cycle_us=cycle, copy_bandwidth_gb_s=bandwidth, signature={},
            run_id='compiled-table-test-adapter', source_id='compiled-table', candidate=0,
            c_us=None, r_us=None, copy_service={}, sampling_mode='quick_1_5', formal_eligible=False)
    return dict(anchors=anchors)


class OProjAutotuneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shlex.split(os.environ.get('CXX', 'c++'))
        if not compiler or not shutil.which(compiler[0]):
            raise unittest.SkipTest('host C++ compiler required')
        temporary = tempfile.TemporaryDirectory(prefix='fuse-oproj-autotune-')
        cls.addClassCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        (directory / 'cutlass').mkdir()
        (directory / 'cutlass/cutlass.h').write_text('#pragma once\n#define CUTLASS_HOST_DEVICE\n')
        cls.binary = directory / 'selector'
        source = r'''
#include "csrc/operators/sm103/detail/autotune.cuh"
#include <iomanip>
#include <iostream>
#include <string>
int main(int argc, char** argv) {
  using namespace fuse::detail;
  const bool cached = argc > 1 && std::string(argv[1]) == "cached";
  OProjTuningRequest q;
  while (std::cin >> q.m >> q.n >> q.k >> q.world >> q.sm_count >> q.device
         >> q.policy_index >> q.raster >> q.max_swizzle_size >> q.swizzle >> q.comm_ctas) {
    const auto r = cached ? select_oproj_plan_cached(q) : select_oproj_plan(q);
    const auto& p = r.prediction;
    std::cout << std::setprecision(17) << int(r.status) << ' ' << r.policy_index << ' '
        << r.along_n << ' ' << r.swizzle << ' ' << r.comm_ctas << ' ' << r.compute_ctas << ' '
        << p.score_us << ' ' << p.production_consumption_ratio << ' ' << p.exposed_feed_us << ' '
        << p.copy_finish_us << ' ' << p.compute_finish_us << ' ' << r.cache_hit << '\n';
  }
}
'''
        built = subprocess.run([*compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
            '-I', str(directory), '-I', str(ROOT), '-x', 'c++', '-', '-o', str(cls.binary)],
            input=source, text=True, capture_output=True, timeout=60)
        if built.returncode:
            raise AssertionError(built.stderr)
        cls.calibration = compiled_services()

    def run_requests(self, requests, cached=False):
        output = subprocess.run([str(self.binary)] + (['cached'] if cached else []),
            input=''.join(' '.join(str(q[field]) for field in FIELDS) + '\n' for q in requests),
            text=True, capture_output=True, check=True, timeout=60).stdout.splitlines()
        self.assertEqual(len(output), len(requests))
        names = ('status', 'policy_index', 'along_n', 'swizzle', 'comm_ctas', 'compute_ctas',
                 'score_us', 'production_consumption_ratio', 'exposed_feed_us',
                 'copy_finish_us', 'compute_finish_us', 'cache_hit')
        return [dict(zip(names, map(float, line.split()))) for line in output]

    def expected(self, q):
        results = []
        for policy in range(2):
            for raster in range(2):
                for swizzle in (4, 8):
                    if ((q['policy_index'] >= 0 and q['policy_index'] != policy) or
                        (q['raster'] >= 0 and q['raster'] != raster) or swizzle > q['max_swizzle_size'] or
                        (q['swizzle'] and q['swizzle'] != swizzle) or
                        planner.swizzle_for(q['m'], q['n'], swizzle) != swizzle):
                        continue
                    spec = dict(tile_policy=planner.POLICIES[policy],
                                raster='along_n' if raster else 'along_m', max_swizzle_size=swizzle)
                    for comm in (8, 16, 24, 32, 48):
                        if q['comm_ctas'] and q['comm_ctas'] != comm:
                            continue
                        predicted = planner.predict(self.calibration, q, spec, comm)
                        if predicted['status'] != 'predicted':
                            continue
                        # Do not transfer an anchor measured with more active
                        # workers to a tiny target whose entire grid is smaller.
                        if any(e['compute_ctas'] != predicted['prediction']['compute_ctas'] for e in predicted['evidence']):
                            continue
                        results.append((predicted['prediction']['score_us'], policy, raster, swizzle, comm, predicted))
        return min(results, key=lambda row: row[0])

    def assert_matches_python(self, q, actual):
        _, policy, raster, swizzle, comm, expected = self.expected(q)
        self.assertEqual(actual['status'], 0)
        self.assertEqual([actual[f] for f in ('policy_index', 'along_n', 'swizzle', 'comm_ctas')],
                         [policy, raster, swizzle, comm])
        for field in ('score_us', 'production_consumption_ratio', 'exposed_feed_us', 'copy_finish_us', 'compute_finish_us'):
            value = expected['prediction'][field]
            self.assertAlmostEqual(actual[field], value, delta=max(1e-7, abs(value) * 1e-10), msg=field)

    def test_joint_pool_matches_python_for_anchor_new_geometry_and_padding(self):
        requests = [request(), request(m=65536, n=4096, k=12288),
                    request(m=16384, n=16384, k=16384, world=8),
                    request(m=17 * 128, n=9 * 256, k=12288)]
        for q, result in zip(requests, self.run_requests(requests)):
            with self.subTest(request=q):
                self.assert_matches_python(q, result)

    def test_explicit_overrides_are_preserved_and_use_same_model(self):
        requests = [request(policy_index=1, raster=1, swizzle=4),
                    request(policy_index=0, raster=0, swizzle=8, comm_ctas=8),
                    request(comm_ctas=24), request(max_swizzle_size=4)]
        for q, result in zip(requests, self.run_requests(requests)):
            with self.subTest(request=q):
                self.assert_matches_python(q, result)
                for field in ('policy_index', 'swizzle', 'comm_ctas'):
                    if q[field] >= (0 if field == 'policy_index' else 1):
                        self.assertEqual(q[field], result[field])

    def test_domain_missing_budget_and_unresolved_exact_swizzle_reject(self):
        changes = ({'world': 2}, {'sm_count': 147}, {'k': 4096}, {'k': 32768},
                   {'comm_ctas': 12}, {'m': 32769}, {'max_swizzle_size': 2},
                   {'m': 17 * 128, 'n': 5 * 256, 'swizzle': 8},
                   {'m': 4 * 128, 'n': 3 * 256})
        results = self.run_requests([request(**change) for change in changes])
        self.assertEqual([r['status'] for r in results], [2] * len(changes))
        valid = request(m=17 * 128, n=5 * 256)
        self.assertEqual(self.run_requests([valid])[0]['swizzle'], 4)

    def test_invalid_input_and_work_limit_are_explicit_failures(self):
        requests = [request(m=0), request(swizzle=3), request(policy_index=2),
                    request(raster=2), request(device=-1), request(comm_ctas=148)]
        self.assertEqual([r['status'] for r in self.run_requests(requests)], [1] * len(requests))
        self.assertEqual(self.run_requests([request(m=128 * 1000000)])[0]['status'], 3)

    def test_cache_preserves_results_and_keys_all_physical_fields(self):
        base = request(m=8192, policy_index=0, raster=0, swizzle=4, comm_ctas=16)
        changes = ({'m': 8320}, {'n': 8448}, {'k': 8448}, {'world': 8}, {'device': 1},
                   {'policy_index': 1}, {'raster': 1}, {'max_swizzle_size': 4}, {'swizzle': 8}, {'comm_ctas': 24})
        requests = [base, base] + [base | change for change in changes]
        actual = self.run_requests(requests, cached=True)
        direct = self.run_requests(requests)
        self.assertEqual([r['cache_hit'] for r in actual], [0, 1] + [0] * len(changes))
        for cached, uncached in zip(actual, direct):
            cached = copy.copy(cached); cached['cache_hit'] = 0
            self.assertEqual(cached, uncached)
        invalid = base | {'sm_count': 147}
        self.assertEqual([r['cache_hit'] for r in self.run_requests([invalid, invalid], cached=True)], [0, 0])

    def test_cache_is_bounded_and_evicted_requests_recompute(self):
        requests = [request(m=(17 + i) * 128, n=9 * 256, policy_index=0, raster=0,
                            swizzle=4, comm_ctas=16) for i in range(17)]
        results = self.run_requests(requests + [requests[0], requests[0]], cached=True)
        self.assertEqual([r['cache_hit'] for r in results], [0] * 18 + [1])


if __name__ == '__main__':
    unittest.main()
