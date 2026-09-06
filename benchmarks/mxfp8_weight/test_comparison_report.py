"""CPU-only matched-boundary, formula and legacy-table-format checks."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from comparison_report import (SCHEMA, build_report, geometry,
                               select_baselines, wide_rows, write_report)
from export_baselines import write_tables


def timing(median):
    return dict(p50_us=median, p95_us=median * 1.1, mean_us=median)


def case(backward=False):
    result = dict(id='gemm_a2a/test/s512/cp4', direction='gemm_a2a', model='test',
                  global_seq=512, cp=4, batch=1, m=128, n=256, k=64,
                  layout='rank_major', visible_devices='0,2,4,5')
    if backward:
        result.update(id='qkv_backward/test/s512/cp4', operator='qkv',
                      b_mnk=[128, 64, 256], w_mnk=[256, 64, 128])
        del result['direction']
    return result


def baseline(backend='teub', launch='graph', backward=False, mode='immediate',
             median=100, data=20, weight=60):
    record = dict(backend=backend, launch=launch)
    if backward:
        record.update(weight_mode=mode, grad_dtype='fp32', beta=int(mode == 'deferred'),
                      data=timing(data), weight=timing(weight), total=timing(median))
    else:
        record.update(timing(median))
    return dict(case=case(backward), record=record, config=dict(sms=8), source='historical.json')


def measured(base, backend='fuse', median=80, data=15, weight=50):
    result = copy.deepcopy(base['record'])
    result.update(backend=backend, warmup=10, iterations=50,
                  config=dict(comm_ctas=8), correctness=dict(passed=True))
    if 'total' in result:
        result.update(data=timing(data), weight=timing(weight), total=timing(median))
    else:
        for key in ('p50_us', 'p95_us', 'mean_us'):
            result.pop(key)
        result['forward'] = timing(median)
    return result


def report(base, *records):
    return ('new.json', dict(schema=SCHEMA, complete=True,
                            cases=[dict(case=copy.deepcopy(base['case']),
                                        operator='qkv_backward' if 'total' in base['record'] else 'qkv_forward',
                                        records=list(records))]))


class ComparisonReportTest(unittest.TestCase):
    def test_baseline_winner_keeps_same_launch(self):
        rows = [baseline(median=90), baseline('cublaslt_nccl', median=100),
                baseline(launch='eager', median=120),
                baseline('cublaslt_nccl', 'eager', median=110)]
        winners = select_baselines(rows, [])
        self.assertEqual(len(winners), 2)
        by_launch = {key[2]: value['record']['backend'] for key, value in winners.items()}
        self.assertEqual(by_launch, dict(eager='cublaslt_nccl', graph='teub'))

    def test_backward_selects_actual_total_not_fastest_b_or_sum(self):
        rows = [baseline(backward=True, median=100, data=50, weight=60),
                baseline('cublaslt_nccl', backward=True, median=120, data=5, weight=5)]
        output = build_report([], rows)
        self.assertEqual({r['best_backend'] for r in output['rows']}, {'teub'})
        by_phase = {r['phase']: r['best_p50_us'] for r in output['rows']}
        self.assertEqual(by_phase, dict(data=50, weight=60, total=100))

    def test_forward_flops_speedup_and_pure_compute_rate(self):
        base = baseline(median=100)
        output = build_report([base], [], [report(base, measured(base, median=80),
                                                  measured(base, 'pure_cublas', median=40))])
        row = output['rows'][0]
        self.assertEqual(row['flops_per_gpu'], 2 * 128 * 256 * 64)
        self.assertAlmostEqual(row['best_tflops_per_gpu'], 2 * 128 * 256 * 64 / 1e8)
        self.assertEqual(row['baseline_over_fuse'], 1.25)
        self.assertEqual(row['fuse_over_pure_rate_pct'], 50)

    def test_backward_total_flops_actual_total_and_mode(self):
        base = baseline(backward=True, mode='deferred', median=100)
        output = build_report([], [base], [report(base, measured(base, median=80, data=15, weight=50),
                                                 measured(base, 'pure_cublas', median=40, data=5, weight=10))])
        rows = {row['phase']: row for row in output['rows']}
        self.assertEqual(rows['total']['flops_per_gpu'], 4 * 128 * 64 * 256)
        self.assertEqual(rows['total']['fuse_p50_us'], 80)
        self.assertEqual(rows['total']['pure_cublas_p50_us'], 40)
        self.assertEqual(rows['total']['fuse_over_pure_rate_pct'], 50)
        self.assertEqual(rows['total']['beta'], 1)

    def test_missing_measurements_stay_null_and_not_counted(self):
        output = build_report([baseline()], [])
        row = output['rows'][0]
        self.assertIsNone(row['fuse_p50_us'])
        self.assertIsNone(row['pure_cublas_p50_us'])
        self.assertIsNone(row['baseline_over_fuse'])
        self.assertEqual(output['missing_fuse_records'], 1)
        self.assertEqual(output['summary'][0]['matched_fuse_rows'], 0)
        self.assertIsNone(output['summary'][0]['baseline_over_fuse_geomean'])

    def test_launch_not_matched_to_other_launch(self):
        graph, eager = baseline(), baseline(launch='eager')
        output = build_report([graph, eager], [], [report(graph, measured(graph))])
        rows = {row['launch']: row for row in output['rows']}
        self.assertIsNone(rows['eager']['fuse_p50_us'])
        self.assertEqual(rows['graph']['fuse_p50_us'], 80)

    def test_duplicate_baseline_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate baseline'):
            select_baselines([baseline(), baseline()], [])

    def test_duplicate_measurement_rejected(self):
        base = baseline()
        with self.assertRaisesRegex(ValueError, 'duplicate measured'):
            build_report([base], [], [report(base, measured(base), measured(base))])

    def test_mismatched_shape_or_layout_rejected(self):
        base = baseline()
        for key, value in [('n', 128), ('layout', 'causal_paired'), ('visible_devices', '0,1,2,3')]:
            source, raw = report(base, measured(base))
            raw['cases'][0]['case'][key] = value
            with self.assertRaisesRegex(ValueError, 'geometry/layout differs'):
                build_report([base], [], [(source, raw)])

    def test_flattened_geometry_rejects_separate_batch(self):
        value = case()
        value['batch'] = 2
        with self.assertRaisesRegex(ValueError, 'flattened'):
            geometry(value)

    def test_backward_dtype_and_beta_required(self):
        for key, value in [('grad_dtype', 'bf16'), ('beta', 1)]:
            base = baseline(backward=True)
            base['record'][key] = value
            with self.assertRaises(ValueError):
                build_report([], [base])

    def test_backward_total_cannot_be_synthesized(self):
        base = baseline(backward=True)
        del base['record']['total']
        with self.assertRaises(KeyError):
            build_report([], [base])

    def test_invalid_timing_rejected(self):
        for value in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                build_report([baseline(median=value)], [])

    def test_unmatched_measurement_rejected(self):
        base = baseline()
        item = measured(base)
        item['launch'] = 'eager'
        with self.assertRaisesRegex(ValueError, 'unmatched'):
            build_report([base], [], [report(base, item)])

    def test_pilot_rejected_unless_explicitly_diagnostic(self):
        base = baseline()
        item = measured(base)
        item['iterations'] = 3
        with self.assertRaisesRegex(ValueError, '10 warmups / 50 samples'):
            build_report([base], [], [report(base, item)])
        output = build_report([base], [], [report(base, item)], allow_diagnostic=True)
        self.assertTrue(output['diagnostic'])

    def test_summary_counts_only_matched_rows(self):
        base = baseline()
        other = copy.deepcopy(base)
        other['case']['id'] = 'gemm_a2a/other/s512/cp4'
        other['case']['model'] = 'other'
        output = build_report([base, other], [], [report(base, measured(base, median=50))])
        summary = output['summary'][0]
        self.assertEqual((summary['baseline_rows'], summary['matched_fuse_rows'], summary['fuse_wins']), (2, 1, 1))
        self.assertEqual(summary['baseline_over_fuse_geomean'], 2)

    def test_wide_ms_format_and_missing_graph(self):
        base = baseline(launch='eager')
        rows = wide_rows(build_report([base], [], [report(base, measured(base, median=80))]))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['eager_p50_ms'], 0.08)
        self.assertEqual(rows[0]['eager_best_external_p50_ms'], 0.1)
        self.assertIsNone(rows[0]['graph_p50_ms'])

    def test_export_names_columns_null_and_legacy_units(self):
        base = baseline()
        output = build_report([base], [])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_report(output, path)
            rows = json.loads((path / 'comparison_summary.json').read_text())
            self.assertEqual(len(rows), 1)
            with (path / 'comparison_summary.csv').open() as stream:
                csv_row = next(csv.DictReader(stream))
            self.assertEqual(csv_row['graph_p50_ms'], '')
            text = (path / 'comparison_summary.md').read_text()
            self.assertIn('| CP | 模型 | S | GEMM M×N×K |', text)
            self.assertIn('Eager p50/p95 ms / TFLOPS', text)
            self.assertNotIn('| Phase |', text)
            self.assertFalse((path / 'comparison_phases.json').exists())
            write_report(output, path, phase_details=True)
            self.assertTrue((path / 'comparison_phases.json').exists())

    def test_archival_json_unchanged_by_rendering(self):
        rows = [baseline()]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_tables(output, 'forward_best', rows)
            self.assertEqual((output / 'forward_best.json').read_text(), json.dumps(rows, indent=2) + '\n')
            self.assertIn('| CP | 模型 | S | GEMM M×N×K | p50 ms | TFLOPS/GPU | 配置 |',
                          (output / 'forward_best.md').read_text())

    def test_published_full_matrix(self):
        directory = Path(__file__).resolve().parents[2] / 'results/mxfp8_weight/published'
        forward = json.loads((directory / 'forward_best.json').read_text())
        backward = json.loads((directory / 'backward_best.json').read_text())
        output = build_report(forward, backward)
        self.assertEqual((output['baseline_winners'], output['phase_rows']), (1152, 2688))
        self.assertEqual(len(wide_rows(output)), 576)


if __name__ == '__main__':
    unittest.main()
