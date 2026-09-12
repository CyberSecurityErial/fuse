"""Host-only MXFP8 scheduling/selection contracts; all timings below are SYNTHETIC.

These fixtures test causality and ownership, not a measured GPU performance claim.
No CUDA runtime, remote machine, cloud object or benchmark result is required.
"""

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]

PROGRAM = r'''
#include "csrc/operators/sm103/detail/autotune.cuh"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using namespace fuse::detail;
using MS = Mxfp8QkvModelStatus;
using TS = Mxfp8QkvTuningStatus;

void check(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

// Artificial host fixtures, not B300 measurements.
Mxfp8QkvServices services() { return {0., 1., 1.}; }
int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }
Mxfp8QkvModelInput model_input(int64_t m = 128, int64_t n = 512,
                             int64_t k = 256, int comm = 2, int sm = 148) {
  Mxfp8QkvModelInput q;
  q.m = m; q.n = n; q.k = k; q.world = 4; q.sm_count = sm; q.comm_ctas = comm;
  q.calibrated_compute_ctas = static_cast<int>(std::min(
      ceil_div(m, 128) * ceil_div(n, 256), int64_t{sm - comm}));
  q.services = services(); q.bulk = {m, 100., 200.};
  return q;
}

void mapping_and_padding() {
  for (bool along_n : {false, true}) for (int width : {1, 2, 4, 8}) {
    auto q = model_input(17 * 128, 5 * 256, 384, 16, 148);
    q.along_n = along_n; q.resolved_swizzle = width;
    const int64_t mt = ceil_div(q.m, 128), nt = ceil_div(q.n, 256);
    const int64_t pm = ceil_div(mt, width) * width, pn = ceil_div(nt, width) * width;
    q.calibrated_compute_ctas = static_cast<int>(std::min(pm * pn, int64_t{132}));
    int log = 0;
    while ((1 << log) < width) ++log;
    const OProjModelSchedulerParams p{
        along_n ? OProjModelSchedulerParams::RasterOrder::AlongN
                : OProjModelSchedulerParams::RasterOrder::AlongM,
        log, static_cast<uint64_t>(pm * pn), static_cast<uint64_t>(q.calibrated_compute_ctas),
        {static_cast<uint64_t>(pm * pn)}, {static_cast<uint64_t>(along_n ? pn : pm)}};
    std::set<std::pair<int, int>> tiles;
    std::set<std::pair<int, int>> copies;
    using Copy = ConsumerTileOrder<PublishedTile<128, 256>, 64, 128>;
    for (uint64_t linear = 0; linear < p.blocks_per_problem_; ++linear) {
      const auto tile = ProducerTileOrder::decode(p, linear);
      check(tile.valid && tile.batch == 0, "producer mapping failed");
      check(ProducerTileOrder::linear(p, tile.m, tile.n) == linear, "producer inverse mismatch");
      if (tile.m < mt && tile.n < nt)
        check(tiles.emplace(tile.m, tile.n).second, "duplicate valid GEMM tile");
      for (int slot = 0; slot < Copy::kSlots; ++slot) {
        const auto task = Copy::decode(p, linear * Copy::kSlots + slot, q.m, q.n);
        if (!task.valid) continue;
        check(copies.emplace(task.row, task.column).second, "duplicate copy rectangle");
        const auto owner = ProducerTileOrder::linear(p, task.last_m, task.last_n);
        check(owner == linear, "copy is not assigned to its last producer");
      }
    }
    check(tiles.size() == static_cast<size_t>(mt * nt), "padded grid lost valid tiles");
    check(copies.size() == static_cast<size_t>(mt * nt * 4), "route rectangles lost coverage");
  }
  // Partial ready panels are outside the first registered calibration domain.
  // Do not relax the implementation to make a synthetic fixture pass.
  const auto tail = model_input(128, 384, 384);
  check(score_mxfp8_qkv_bulk(tail).status == MS::UnsupportedGeometry,
        "unsupported partial panel accepted");
}

Mxfp8QkvTuningRequest request() {
  Mxfp8QkvTuningRequest q;
  q.m = 4096; q.n = 4096; q.k = 1024; q.world = 4;
  q.comm_ctas = 16; q.stages = 4; q.dynamic_smem_bytes = 187392;
  q.q_heads = 16; q.kv_heads = 8;
  return q;
}

Mxfp8QkvCalibrationPoint calibration(const Mxfp8QkvTuningRequest& q) {
  Mxfp8QkvCalibrationPoint p;
#define FIELD(name) p.name = q.name
  FIELD(world); FIELD(sm_count); FIELD(capability); FIELD(tile_m); FIELD(tile_n);
  FIELD(tile_k); FIELD(epilogue_n); FIELD(stages); FIELD(cluster_ctas); FIELD(raster);
  FIELD(swizzle); FIELD(comm_ctas); FIELD(k); FIELD(dynamic_smem_bytes);
  FIELD(q_heads); FIELD(kv_heads); FIELD(head_dim);
#undef FIELD
  p.compute_ctas = q.sm_count - q.comm_ctas;
  p.m_min = q.m; p.m_max = q.m * 4; p.n_min = p.n_max = q.n;
  p.services = services();
  p.bulk = {q.m, 100., 200.};
  return p;
}

void selector_budgets() {
  auto q = request();
  auto p = calibration(q);
  const auto before = q;
  auto result = select_mxfp8_qkv_plan(q, &p, 1);
  check(result.status == TS::Success && result.comm_ctas == 16 && result.compute_ctas == 132,
        "manual budget changed or valid calibration rejected");
  check(!result.along_n && result.swizzle == q.swizzle &&
        mxfp8_qkv_request_key(q) == mxfp8_qkv_request_key(before), "selector changed fixed GEMM inputs");
  auto q8 = q; q8.comm_ctas = 8;
  std::vector<Mxfp8QkvCalibrationPoint> points{calibration(q8), p};
  q.comm_ctas = 12;
  check(select_mxfp8_qkv_plan(q, points.data(), points.size()).status == TS::UnsupportedCalibration,
        "invented c=12 services from c=8/16");
  q.comm_ctas = 0;
  result = select_mxfp8_qkv_plan(q, points.data(), points.size());
  check(result.status == TS::Success && (result.comm_ctas == 8 || result.comm_ctas == 16),
        "automatic budget not selected from measured candidates");
  q = request();
  check(select_mxfp8_qkv_plan(q, nullptr, 0).status == TS::UnsupportedCalibration,
        "empty calibration fabricated a result");
  for (int repeat = 0; repeat < 2; ++repeat) {
    result = select_mxfp8_qkv_plan_cached(q);
    check(result.status == TS::UnsupportedCalibration && !result.cache_hit,
          "empty compiled table returned or cached fake calibration");
  }
}

void selector_calibration_identity() {
  const auto base = request();
  const auto anchor = calibration(base);
#define MISMATCH(name, value) { auto p = anchor; p.name = value; \
  check(select_mxfp8_qkv_plan(base, &p, 1).status == TS::UnsupportedCalibration, \
        "mismatched calibration accepted: " #name); }
  MISMATCH(world, 8); MISMATCH(sm_count, 147); MISMATCH(capability, 100);
  MISMATCH(tile_m, 64); MISMATCH(tile_n, 128); MISMATCH(tile_k, 256);
  MISMATCH(epilogue_n, 32); MISMATCH(stages, 3); MISMATCH(cluster_ctas, 2);
  MISMATCH(raster, 1); MISMATCH(swizzle, 2); MISMATCH(compute_ctas, 131);
  MISMATCH(dynamic_smem_bytes, 187264); MISMATCH(q_heads, 8); MISMATCH(kv_heads, 4);
  MISMATCH(head_dim, 64); MISMATCH(m_max, 2048); MISMATCH(n_min, 8192);
#undef MISMATCH
  auto q = base; q.k = 1536;
  auto lower = anchor, upper = anchor; upper.k = 2048;
  std::vector<Mxfp8QkvCalibrationPoint> points{lower, upper};
  check(select_mxfp8_qkv_plan(q, points.data(), points.size()).status == TS::UnsupportedCalibration,
        "K interpolation enabled without calibrated authorization");
  for (auto& p : points) p.k_interpolation_group = 7;
  check(select_mxfp8_qkv_plan(q, points.data(), points.size()).status == TS::Success,
        "authorized same-domain K bracket rejected");
  points[1].k_interpolation_group = 8;
  check(select_mxfp8_qkv_plan(q, points.data(), points.size()).status == TS::UnsupportedCalibration,
        "different K interpolation groups merged");
  points[1].k_interpolation_group = 7;
  points[0].bulk.route_us = -1;
  points[1].bulk.route_us = 3;
  check(select_mxfp8_qkv_plan(q, points.data(), points.size()).status == TS::UnsupportedCalibration,
        "invalid endpoint became a valid interpolated service");
  points = {lower, upper};
  for (auto& p : points) p.k_interpolation_group = 7;
  points.push_back(points.front());
  check(select_mxfp8_qkv_plan(q, points.data(), points.size()).status == TS::UnsupportedCalibration,
        "ambiguous interpolation brackets depend on table order");
  points = {anchor, anchor};
  check(select_mxfp8_qkv_plan(base, points.data(), points.size()).status == TS::UnsupportedCalibration,
        "duplicate exact anchors were silently selected");
}

void request_key_fields() {
  const auto base = request();
#define KEY(name) { auto changed = base; ++changed.name; \
  check(mxfp8_qkv_request_key(base) != mxfp8_qkv_request_key(changed), "cache key omits " #name); }
  KEY(m); KEY(n); KEY(k); KEY(world); KEY(sm_count); KEY(device); KEY(capability);
  KEY(tile_m); KEY(tile_n); KEY(tile_k); KEY(epilogue_n); KEY(stages); KEY(cluster_ctas);
  KEY(raster); KEY(max_swizzle_size); KEY(swizzle); KEY(comm_ctas); KEY(dynamic_smem_bytes);
  KEY(q_heads); KEY(kv_heads); KEY(head_dim);
#undef KEY
}

void bulk_balance() {
  auto q = model_input(4096, 4096, 1024, 16);
  q.services.startup_us = 2;
  q.services.tile_first_us = 5;
  q.services.tile_cycle_us = 3;
  q.bulk = {4096, 20., 15.};  // QR < R is legal: do not clamp QR-R.
  auto first = score_mxfp8_qkv_bulk(q);
  check(first.status == MS::Success, "bulk model not selected");
  check(first.compute_finish_us == 2+5+3*3 && first.copy_finish_us == 2+15,
        "bulk first/cycle service or QR anchor changed");
  q.m *= 4;
  auto large = score_mxfp8_qkv_bulk(q);
  check(large.status == MS::Success && large.copy_finish_us == 2+15+3*20,
        "bulk model scaled W work with M or clamped its pacing difference");
  check(large.score_us == std::max(large.compute_finish_us, large.copy_finish_us),
        "bulk score is not constant-work host arithmetic");
  auto bad = q; bad.m += 128;
  check(score_mxfp8_qkv_bulk(bad).status == MS::UnsupportedGeometry, "bulk extrapolation escaped domain");
  bad = q; bad.bulk.route_us = -1;
  check(score_mxfp8_qkv_bulk(bad).status == MS::InvalidInput, "negative bulk service accepted");
  bad = q; bad.calibrated_compute_ctas--;
  check(score_mxfp8_qkv_bulk(bad).status == MS::InvalidInput, "unmeasured compute budget accepted");
}

int main(int argc, char** argv) {
  try {
    check(argc == 2, "one host test case required");
    const std::string name = argv[1];
    if (name == "mapping") mapping_and_padding();
    else if (name == "budgets") selector_budgets();
    else if (name == "calibration") selector_calibration_identity();
    else if (name == "keys") request_key_fields();
    else if (name == "bulk") bulk_balance();
    else throw std::runtime_error("unknown host test case");
    std::cout << "PASS " << name << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
'''


class Mxfp8AutotuneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shlex.split(os.environ.get('CXX', 'c++'))
        if not compiler or not shutil.which(compiler[0]):
            raise unittest.SkipTest('host C++ compiler required')
        temporary = tempfile.TemporaryDirectory(prefix='fuse-mxfp8-model-')
        cls.addClassCleanup(temporary.cleanup)
        directory = Path(temporary.name)
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
        cls.binary = directory / 'model'
        flags = ['-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror']
        # Probe sanitizer support independently so model compile errors cannot
        # be hidden by an automatic retry without sanitizers.
        sanitizer = ['-fsanitize=undefined', '-fno-sanitize-recover=all']
        probe = subprocess.run([*compiler, *flags, *sanitizer, '-x', 'c++', '-',
                                '-o', str(directory / 'sanitizer-probe')],
                               input='int main() { return 0; }\n', text=True,
                               capture_output=True, timeout=60)
        if not probe.returncode:
            ran = subprocess.run([str(directory / 'sanitizer-probe')], capture_output=True, timeout=30)
            if not ran.returncode:
                flags += sanitizer
        built = subprocess.run([*compiler, *flags, '-I', str(directory), '-I', str(ROOT),
                                '-x', 'c++', '-', '-o', str(cls.binary)],
                               input=PROGRAM, text=True, capture_output=True, timeout=60)
        if built.returncode:
            raise AssertionError(built.stderr)

    def run_case(self, name):
        result = subprocess.run([str(self.binary), name], text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'PASS ' + name)

    def test_production_mapping_both_rasters_swizzles_and_padding(self):
        self.run_case('mapping')

    def test_manual_budget_empty_calibration_and_no_sm_interpolation(self):
        self.run_case('budgets')

    def test_calibration_physical_identity_and_authorized_k_brackets(self):
        self.run_case('calibration')

    def test_cache_key_covers_every_physical_request_field(self):
        self.run_case('keys')

    def test_bulk_model_uses_fixed_weight_and_no_publication_estimates(self):
        self.run_case('bulk')

    def test_model_keeps_pure_gemm_fixed_and_no_event_simulation(self):
        model = (ROOT / 'csrc/operators/sm103/detail/performance_model.cuh').read_text()
        mxfp8 = model[model.index('enum class Mxfp8QkvModelStatus'):]
        self.assertIn('score_mxfp8_qkv_bulk', mxfp8)
        self.assertNotIn('std::priority_queue', mxfp8)
        self.assertNotIn('score_mxfp8_qkv_schedule', mxfp8)
        gemm = (ROOT / 'csrc/operators/sm103/detail/gemm.cuh').read_text()
        self.assertIn('using Mxfp8GemmTypes = Mxfp8GemmFamily<>;', gemm)
        family = gemm[gemm.index('struct Mxfp8GemmFamily'):gemm.index('using Mxfp8GemmTypes')]
        pure = family[family.index('using PureGemm'):family.index('static_assert')]
        self.assertIn('Mainloop, Epilogue', pure)
        self.assertNotIn('WeightReady', pure)
        self.assertNotIn('autotune', pure)

    def test_public_query_launch_and_profile_share_resolver(self):
        policy = (ROOT / 'csrc/operators/sm103/api/policy.cuh').read_text()
        resolver = policy[policy.index('inline cudaError_t resolve_mxfp8_qkv_communication('):]
        resolver = resolver[:resolver.index('}  // namespace')]
        self.assertLess(resolver.index('params.projection.num_comm_ctas > 0'),
                        resolver.index('device_info(&info)'))
        self.assertIn('*resolved = params;', resolver)
        self.assertIn('Mainloop::DispatchPolicy::Stages', resolver)
        self.assertIn('Scheduler::to_underlying_arguments', resolver)
        self.assertIn('select_mxfp8_qkv_plan_cached(request)', resolver)
        self.assertNotIn('cudaMalloc', resolver)
        self.assertNotIn('cudaEvent', resolver)
        self.assertNotIn('<<<', resolver)
        header = (ROOT / 'include/fuse/operators/primitives/gemm_a2a_mxfp8.h').read_text()
        self.assertIn('recommended_gemm_a2a_mxfp8_comm_ctas(', header)
        api = (ROOT / 'csrc/operators/sm103/api/forward_mxfp8.cuh').read_text()
        self.assertIn('resolve_mxfp8_qkv_communication(params, &p)', api)
        self.assertLess(api.index('resolve_mxfp8_qkv_communication(params, &p)'),
                        api.index('status = validate_mxfp8(p, &workspace)'))
        self.assertIn('!dynamic_weight && params.projection.num_comm_ctas == 0', api)
        for epilogue in (32, 64):
            self.assertIn(f'launch_mxfp8<false, {epilogue}>(params, stream, true)', api)
            self.assertIn(f'launch_mxfp8<true, {epilogue}>(params, stream, true,', api)


if __name__ == '__main__':
    unittest.main()
