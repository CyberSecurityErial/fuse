"""CPU-only full-registry coverage and optimization-weighting checks."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from optimization_report import (LONG_TOKENS, build_summary,
                                  expected_rows, markdown, write_summary)


def complete_fixture(speedup=1.6):
    rows = []
    for (identifier, mode), item in expected_rows().items():
        case, operator = item['case'], item['operator']
        backward = operator.endswith('backward')
        row = dict(id=identifier, operator=operator, model=case['model'],
                   global_seq=case['global_seq'], cp=case['cp'], m=case['m'],
                   weight_mode=mode, grad_dtype='fp32' if backward else None,
                   beta=int(mode == 'deferred') if backward else None)
        for field in (('b_mnk', 'w_mnk') if backward else ('n', 'k')):
            row[field] = copy.deepcopy(case[field])
        for launch in ('eager', 'graph'):
            row.update({f'{launch}_best_external_p50_ms': speedup,
                        f'{launch}_p50_ms': 1,
                        f'{launch}_best_external_backend': 'teub',
                        f'{launch}_speedup_over_best_external': speedup})
            if backward:
                row[f'{launch}_total_p50_ms'] = 1
                row[f'{launch}_best_external_total_p50_ms'] = speedup
        rows.append(row)
    metadata = dict(schema='mxfp8-matched-comparison-v1', diagnostic=False,
                    measurement_reports=[dict(source='formal.json', complete=True)],
                    missing_fuse_records=0, baseline_winners=2 * len(rows))
    return rows, metadata


def set_speedup(row, ratio, launches=('eager', 'graph')):
    for launch in launches:
        row[f'{launch}_best_external_p50_ms'] = ratio
        row[f'{launch}_speedup_over_best_external'] = ratio
        if row['operator'].endswith('backward'):
            row[f'{launch}_best_external_total_p50_ms'] = ratio


def set_pure(row, timing, launches=('eager', 'graph')):
    backward = row['operator'].endswith('backward')
    for launch in launches:
        prefix = f'{launch}_cublas'
        row[f'{prefix}_p50_ms'] = timing
        row[f'{prefix}_config'] = dict(
            dq_included=False, communication_included=False,
            data=dict(mnk=copy.deepcopy(row['b_mnk'] if backward else [row['m'], row['n'], row['k']]),
                      output_dtype='torch.bfloat16'),
            weight=dict(mnk=copy.deepcopy(row['w_mnk']), output_dtype='torch.float32', beta=row['beta'])
            if backward else None)
        if backward:
            row[f'{prefix}_total_p50_ms'] = timing
            # Deliberately not the actual total; these medians must be ignored.
            row[f'{prefix}_b_p50_ms'] = timing * .8
            row[f'{prefix}_w_p50_ms'] = timing * .9


class OptimizationReportTest(unittest.TestCase):
    def test_full_registry_coverage(self):
        rows, metadata = complete_fixture()
        result = build_summary(rows, metadata)
        self.assertEqual((result['coverage']['expected_settings'], result['coverage']['expected_wide_rows'],
                          result['coverage']['expected_boundary_records']), (384, 576, 1152))
        self.assertTrue(result['coverage']['complete'])
        self.assertTrue(result['milestone_eligible'])
        self.assertEqual(result['scopes']['long']['aggregate']['boundary_equal']['expected'], 576)
        for scope in result['scopes'].values():
            self.assertEqual(len(scope['groups']), 24)
            for weighting in scope['milestones'].values():
                self.assertTrue(all(weighting.values()))

    def test_boundary_vs_four_operator_weights(self):
        rows, metadata = complete_fixture()
        for row in rows:
            set_speedup(row, 4 if row['operator'].endswith('backward') else 1)
        values = build_summary(rows, metadata)['scopes']['all']['aggregate']
        self.assertAlmostEqual(values['boundary_equal']['geomean'], 4 ** (2 / 3))
        self.assertAlmostEqual(values['operator_equal']['geomean'], 2)
        self.assertEqual(values['operator_equal']['expected_case_launches'], 768)

    def test_backward_modes_paired_before_operator_average(self):
        rows, metadata = complete_fixture()
        for row in rows:
            set_speedup(row, 9 if row['weight_mode'] == 'immediate' else 1)
        result = build_summary(rows, metadata)
        operators = result['scopes']['all']['aggregate']['operator_equal']['operators']
        self.assertAlmostEqual(operators['qkv_backward']['geomean'], 3)
        self.assertAlmostEqual(operators['oproj_backward']['geomean'], 3)
        self.assertAlmostEqual(result['scopes']['all']['aggregate']['operator_equal']['geomean'], math.sqrt(3))

    def test_long_scope_excludes_short_tokens(self):
        rows, metadata = complete_fixture()
        for row in rows:
            set_speedup(row, 2 if row['global_seq'] in LONG_TOKENS else 0.5)
        result = build_summary(rows, metadata)
        self.assertAlmostEqual(result['scopes']['all']['aggregate']['boundary_equal']['geomean'], 1)
        self.assertAlmostEqual(result['scopes']['long']['aggregate']['boundary_equal']['geomean'], 2)
        self.assertEqual(result['scopes']['long']['aggregate']['boundary_equal']['wins'], 576)

    def test_launches_not_mixed_in_group_statistics(self):
        rows, metadata = complete_fixture()
        for row in rows:
            set_speedup(row, 2, ('eager',))
            set_speedup(row, 0.5, ('graph',))
        groups = build_summary(rows, metadata)['scopes']['all']['groups']
        for row in groups:
            self.assertEqual(row['geomean'], 2 if row['launch'] == 'eager' else 0.5)
            self.assertEqual(row['wins'], row['measured'] if row['launch'] == 'eager' else 0)

    def test_group_minimum_maximum_and_wins(self):
        rows, metadata = complete_fixture(1)
        set_speedup(rows[0], 0.5, ('eager',))
        group = next(row for row in build_summary(rows, metadata)['scopes']['all']['groups']
                     if row['operator'] == rows[0]['operator'] and row['cp'] == rows[0]['cp']
                     and row['launch'] == 'eager')
        self.assertEqual((group['minimum'], group['maximum'], group['wins']), (0.5, 1, 0))

    def test_optional_p95_stability_keeps_launches_separate(self):
        rows, metadata = complete_fixture()
        rows[0]['eager_p95_ms'] = 1.25
        rows[0]['graph_p95_ms'] = 1.05
        result = build_summary(rows, metadata)
        groups = [row for row in result['scopes']['all']['groups']
                  if row['operator'] == rows[0]['operator'] and row['cp'] == rows[0]['cp']]
        for row in groups:
            ratio = 1.25 if row['launch'] == 'eager' else 1.05
            self.assertEqual(row['p95_p50_measured'], 1)
            self.assertEqual(row['p95_p50_median'], ratio)
            self.assertEqual(row['p95_p50_maximum'], ratio)
        self.assertTrue(result['milestone_eligible'])

    def test_missing_p95_does_not_block_otherwise_valid_summary(self):
        rows, metadata = complete_fixture()
        result = build_summary(rows, metadata)
        self.assertIsNone(result['scopes']['all']['aggregate']['boundary_equal']['p95_p50_median'])
        self.assertTrue(result['milestone_eligible'])

    def test_missing_wide_row_blocks_all_milestones(self):
        rows, metadata = complete_fixture(10)
        rows.pop()
        result = build_summary(rows, metadata)
        self.assertFalse(result['milestone_eligible'])
        self.assertEqual(result['coverage']['missing_wide_rows'], 1)
        self.assertEqual(result['coverage']['missing_boundary_records'], 2)
        self.assertFalse(any(flag for scope in result['scopes'].values()
                             for weighting in scope['milestones'].values() for flag in weighting.values()))

    def test_missing_timing_is_not_zero_or_borrowed(self):
        rows, metadata = complete_fixture()
        rows[0]['eager_p50_ms'] = None
        result = build_summary(rows, metadata)
        self.assertEqual(result['coverage']['missing_boundary_records'], 1)
        self.assertFalse(result['milestone_eligible'])
        self.assertAlmostEqual(result['scopes']['all']['aggregate']['boundary_equal']['geomean'], 1.6)

    def test_missing_backward_mode_not_given_single_mode_weight(self):
        rows, metadata = complete_fixture()
        missing = next(i for i, row in enumerate(rows) if row['weight_mode'] == 'deferred')
        rows.pop(missing)
        result = build_summary(rows, metadata)['scopes']['all']['aggregate']['operator_equal']
        self.assertEqual(result['mode_paired_case_launches'], result['expected_case_launches'] - 2)
        self.assertFalse(result['complete'])

    def test_diagnostic_unknown_or_incomplete_provenance_cannot_achieve(self):
        rows, metadata = complete_fixture(10)
        variants = [None, dict(metadata, diagnostic=True), dict(metadata, diagnostic=None),
                    dict(metadata, measurement_reports=[]), dict(metadata, missing_fuse_records=1),
                    dict(metadata, measurement_reports=[dict(complete=False)])]
        for meta in variants:
            result = build_summary(rows, meta)
            self.assertFalse(result['milestone_eligible'])
            self.assertFalse(any(result['scopes']['all']['milestones']['operator_equal'].values()))

    def test_thresholds_independent_and_not_rounded_up(self):
        rows, metadata = complete_fixture(1.29999)
        result = build_summary(rows, metadata)
        self.assertEqual(result['milestone_targets'], [1.2, 1.3])
        flags = result['scopes']['all']['milestones']['operator_equal']
        self.assertEqual(flags, {'1.2': True, '1.3': False})
        text = markdown(result).split('## 里程碑', 1)[1].split('## all', 1)[0]
        self.assertIn('| 范围 | 权重 | 1.2× | 1.3× |', text)
        self.assertNotIn('1.4×', text)
        self.assertNotIn('1.5×', text)

    def test_duplicate_or_unknown_row_rejected(self):
        rows, metadata = complete_fixture()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            build_summary(rows + rows[:1], metadata)
        rows[0]['id'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'unexpected'):
            build_summary(rows, metadata)

    def test_registry_geometry_cannot_be_changed(self):
        for field, value in [('n', 1), ('global_seq', 123), ('cp', 16), ('model', 'not-the-registry-model')]:
            rows, metadata = complete_fixture()
            rows[0][field] = value
            with self.assertRaisesRegex(ValueError, 'registry'):
                build_summary(rows, metadata)

    def test_backward_dtype_beta_and_total_are_required(self):
        for field, value in [('grad_dtype', 'bf16'), ('beta', 7), ('eager_total_p50_ms', 3)]:
            rows, metadata = complete_fixture()
            row = next(row for row in rows if row['weight_mode'] == 'immediate')
            row[field] = value
            with self.assertRaises(ValueError):
                build_summary(rows, metadata)

    def test_nonfinite_or_inconsistent_ratio_rejected(self):
        for field, value in [('eager_p50_ms', 0), ('eager_p50_ms', float('nan')),
                             ('eager_best_external_p50_ms', float('inf')),
                             ('eager_speedup_over_best_external', 20)]:
            rows, metadata = complete_fixture()
            rows[0][field] = value
            with self.assertRaises(ValueError):
                build_summary(rows, metadata)

    def test_empty_input_is_an_incomplete_report_not_success(self):
        result = build_summary([])
        self.assertFalse(result['coverage']['complete'])
        self.assertEqual(result['coverage']['missing_boundary_records'], 1152)
        self.assertIsNone(result['scopes']['all']['aggregate']['operator_equal']['geomean'])

    def test_only_two_compact_artifacts(self):
        rows, metadata = complete_fixture()
        result = build_summary(rows, metadata)
        self.assertNotIn('rows', result)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_summary(result, output)
            self.assertEqual(sorted(path.name for path in output.iterdir()),
                             ['optimization_summary.json', 'optimization_summary.md'])
            json.loads((output / 'optimization_summary.json').read_text())
            text = (output / 'optimization_summary.md').read_text()
            self.assertIn('384/384 settings', text)
            self.assertIn('四算子等权几何平均', text)

    def test_missing_pure_is_explicit_and_does_not_change_milestones(self):
        rows, metadata = complete_fixture()
        result = build_summary(rows, metadata)
        reference = result['pure_gemm_reference']
        self.assertFalse(reference['coverage']['complete'])
        self.assertEqual(reference['coverage']['missing_pure_records'], 1152)
        self.assertTrue(result['milestone_eligible'])
        self.assertTrue(all(result['scopes']['all']['milestones']['operator_equal'].values()))
        self.assertIsNone(reference['scopes']['all']['aggregate']['boundary_equal']['base_over_pure_geomean'])
        self.assertIn('参考覆盖不完整', markdown(result))

    def test_pure_reference_ratios_counts_and_actual_backward_total(self):
        rows, metadata = complete_fixture(1.3)
        for row in rows:
            set_pure(row, 1)
        reference = build_summary(rows, metadata)['pure_gemm_reference']
        self.assertTrue(reference['coverage']['complete'])
        self.assertFalse(reference['is_hardware_hard_limit'])
        self.assertEqual(reference['comparison_targets'], [1.2, 1.3, 1.4, 1.5])
        for scope, count in [('all', 1152), ('long', 576)]:
            item = reference['scopes'][scope]
            self.assertEqual(len(item['groups']), 24)
            for row in item['aggregate'].values():
                self.assertAlmostEqual(row['base_over_pure_geomean'], 1.3)
                self.assertAlmostEqual(row['pure_over_base_geomean_percent'], 100 / 1.3)
                self.assertAlmostEqual(row['fuse_over_pure_geomean'], 1)
            boundary = item['aggregate']['boundary_equal']
            self.assertEqual(boundary['targets_exceeding_point_reference'],
                             {'1.2': 0, '1.3': 0, '1.4': count, '1.5': count})
            self.assertEqual(boundary['fuse_faster_than_pure'], 0)
        self.assertIn('实测纯 cuBLAS GEMM 参考≠硬件硬上限', markdown(build_summary(rows, metadata)))

    def test_pure_reference_reuses_operator_weighting_and_mode_pairs(self):
        rows, metadata = complete_fixture(4)
        for row in rows:
            set_pure(row, 1 if row['operator'].endswith('backward') else 4)
        reference = build_summary(rows, metadata)['pure_gemm_reference']['scopes']['all']['aggregate']
        self.assertAlmostEqual(reference['boundary_equal']['base_over_pure_geomean'], 4 ** (2 / 3))
        self.assertAlmostEqual(reference['operator_equal']['base_over_pure_geomean'], 2)
        for row in rows:
            set_pure(row, 4 / 9 if row['weight_mode'] == 'immediate' else 4)
        reference = build_summary(rows, metadata)['pure_gemm_reference']['scopes']['all']['aggregate']
        self.assertAlmostEqual(reference['operator_equal']['base_over_pure_geomean'], math.sqrt(3))
        self.assertAlmostEqual(reference['operator_equal']['operators']['qkv_backward']['base_over_pure_geomean'], 3)
        missing = next(row for row in rows if row['weight_mode'] == 'deferred')
        missing['eager_cublas_total_p50_ms'] = None
        result = build_summary(rows, metadata)
        self.assertTrue(result['milestone_eligible'])
        reference = result['pure_gemm_reference']
        self.assertEqual(reference['coverage']['missing_pure_records'], 1)
        balanced = reference['scopes']['all']['aggregate']['operator_equal']
        self.assertEqual(balanced['mode_paired_case_launches'], balanced['expected_case_launches'] - 1)
        self.assertFalse(balanced['complete'])

    def test_pure_launch_long_scope_and_faster_fuse_are_not_clipped(self):
        rows, metadata = complete_fixture(1)
        for row in rows:
            set_pure(row, 2 if row['global_seq'] in LONG_TOKENS else .5)
        reference = build_summary(rows, metadata)['pure_gemm_reference']
        all_rows = reference['scopes']['all']['aggregate']['boundary_equal']
        long_rows = reference['scopes']['long']['aggregate']['boundary_equal']
        self.assertAlmostEqual(all_rows['base_over_pure_geomean'], 1)
        self.assertEqual(all_rows['fuse_faster_than_pure'], 576)
        self.assertAlmostEqual(long_rows['base_over_pure_geomean'], .5)
        self.assertAlmostEqual(long_rows['pure_over_base_geomean_percent'], 200)
        self.assertAlmostEqual(long_rows['fuse_over_pure_geomean'], .5)
        for row in rows:
            set_pure(row, 2, ('eager',))
            set_pure(row, .5, ('graph',))
        groups = build_summary(rows, metadata)['pure_gemm_reference']['scopes']['all']['groups']
        for group in groups:
            self.assertEqual(group['base_over_pure_geomean'], .5 if group['launch'] == 'eager' else 2)

    def test_invalid_pure_total_config_and_nonfinite_values_are_rejected(self):
        for change in ('wrong_total', 'nonfinite', 'negative', 'boolean', 'DQ', 'communication', 'beta', 'mnk'):
            rows, metadata = complete_fixture()
            row = next(row for row in rows if row['weight_mode'] == 'deferred')
            set_pure(row, 1)
            config = row['eager_cublas_config']
            if change == 'wrong_total': row['eager_cublas_total_p50_ms'] = 2
            elif change == 'nonfinite': row['eager_cublas_p50_ms'] = math.inf
            elif change == 'negative': row['eager_cublas_p50_ms'] = -1
            elif change == 'boolean': row['eager_cublas_p50_ms'] = True
            elif change == 'DQ': config['dq_included'] = True
            elif change == 'communication': config['communication_included'] = True
            elif change == 'beta': config['weight']['beta'] = 0
            elif change == 'mnk': config['weight']['mnk'][0] += 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                build_summary(rows, metadata)

    def test_pure_reference_is_compact_and_missing_fuse_does_not_claim_coverage(self):
        rows, metadata = complete_fixture()
        for row in rows:
            set_pure(row, 1)
        original = copy.deepcopy(rows)
        result = build_summary(rows, metadata)
        self.assertEqual(rows, original)
        self.assertNotIn('rows', result['pure_gemm_reference'])
        json.dumps(result, allow_nan=False)
        rows[0]['eager_p50_ms'] = None
        reference = build_summary(rows, metadata)['pure_gemm_reference']
        self.assertEqual(reference['coverage']['missing_pure_records'], 0)
        self.assertEqual(reference['coverage']['missing_matched_reference_records'], 1)
        self.assertFalse(reference['coverage']['complete'])


if __name__ == '__main__':
    unittest.main()
