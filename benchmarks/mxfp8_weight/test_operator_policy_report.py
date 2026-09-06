"""CPU-only proof of frozen A/B audit, weighting, coverage and table contracts."""

import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile
import unittest

from matrix import full_matrix
from operator_policy_ab import paired_summary
import operator_policy_report as reporter


def timing(case, duration):
    samples = [duration] * 50
    flops = 2 * case['m'] * case['n'] * case['k']
    return dict(p50_us=duration, p95_us=duration, mean_us=duration, samples_us=samples,
                rank_samples_us=[[duration] * 50 for _ in range(case['cp'])],
                flops_per_gpu=flops, tflops_per_gpu=flops / duration / 1e6)


def correctness(operator='oproj_forward'):
    routed = 'local_gemm' if operator == 'qkv_forward' else 'route'
    return {name: dict(all_ranks_finite=True, max_abs=0., relative_rmse=0.)
            for name in ('weight_dequant', routed, 'output')}


def fixture(cp=4, full=False, launches=('eager', 'graph'), speedup=2., operator='oproj_forward'):
    qkv = operator == 'qkv_forward'
    cases = [case for case in full_matrix() if case['direction'] == ('gemm_a2a' if qkv else 'a2a_gemm') and
             case['cp'] == cp and (not qkv or case['global_seq'] >= 131072)]
    if not full:
        cases = cases[:1]
    model = dict(schema='mxfp8-oproj-service-model-v1', world_size=cp, sm_count=132,
                 coefficients=dict(a=1., e=0., b=1., t=1.), launch_prior_us=5., minimum_gain=.1,
                 tile_families=[[128, 256, 64, 2], [128, 320, 64, 2]], source='model.json', sha256='a' * 64)
    if qkv:
        model = dict(schema='mxfp8-qkv-forward-service-model-v1', operator=operator, world_size=cp, sm_count=132,
                     coefficients=dict(compute_gflop_sm_us=171.676018685189, route_slot_task_us=5.913378755546),
                     minimum_gain=.05, source='model.json', sha256='a' * 64,
                     domain=dict(min_m=32768, max_m=131072, min_n=4096, max_n=18432, min_k=2048,
                                 max_k=16384, m_multiple=256, n_multiple=256, k_multiple=64, head_dim=128,
                                 tile_families=[[128, 256, 64, 2]], comm_ctas=list(range(4, 25, 2))),
                     auto_reference='explicit_calibrated_model_or_domain_fallback_not_legacy_auto',
                     scope='qkv_forward_only_manual_comm_ctas_take_precedence')
    scope = reporter.QKV_SCOPE if qkv else reporter.SCOPE
    model_key = 'qkv_forward_comm_model' if qkv else 'oproj_comm_model'
    document = dict(schema='mxfp8-qkv-forward-frozen-policy-ab-v1' if qkv else 'mxfp8-frozen-policy-ab-v1', complete=True,
                    scope=scope, production_policy_written=False, milestone_evidence=False,
                    formal_ab_protocol=True, semantic=reporter.SEMANTIC,
                    token_semantic='T_is_flattened_tokens_no_training_batch',
                    selection='one_explicit_frozen_model_no_search_no_winner_selection',
                    arms=dict(baseline='manual_CTA_zero_native_model_disabled_unchanged_auto',
                              model='manual_CTA_zero_explicit_model_native_selector_or_domain_fallback'),
                    args=dict(cp=cp, launches=','.join(launches), operators=operator, backends='fuse',
                              library='build-mxfp8/libfuse_mxfp8_torch_bridge.so'),
                    timing=dict(rounds=3, warmup_per_arm=10, samples_per_arm=50,
                                sample_statistic='sample_wise_max_across_ranks', clocks='CUDA_events_one_operation_per_sample',
                                ab_order='baseline_model_repeated_three_times_per_case_and_launch'),
                    sources={'model.json': 'a' * 64, 'build-mxfp8/libfuse_mxfp8_torch_bridge.so': 'b' * 64,
                             'benchmarks/mxfp8_weight/operator_policy_ab.py': 'c' * 64,
                             'csrc/operators/ulysses_sm90.cu': 'd' * 64,
                             'csrc/operators/ulysses_sm90/api/policy.cuh': 'e' * 64,
                             'include/fuse/operators/primitives/a2a_gemm.h': 'f' * 64},
                    devices=[dict(rank=rank, cc=[9, 0], sm_count=132) for rank in range(cp)],
                    gpu_clock_snapshot='CPU fixture, not measured clocks', policy_environment={},
                    torch='mock', cuda='mock', **{model_key: model},
                    selected_case_ids=[case['id'] for case in cases], cases=[])
    for case in cases:
        configs = {}
        for arm in reporter.ARMS:
            configs[arm] = dict(requested_comm_ctas=0,
                                comm_ctas=(4 if arm == 'baseline' else 6) if qkv else (8 if arm == 'baseline' else 16),
                                tile_m=128, tile_n=256, tile_k=64, cluster_m=2, sm_count=132, policy_enum=0 if qkv else 4,
                                weight_block_size=32, payload='e4m3', scale='e8m0', weight_axis='original_forward_K',
                                gemm_input='bf16', gemm_accumulator='fp32', raster='n', swizzle=1,
                                gemm_policy_request='auto', route_layout=case['layout'], weight_mnk=None,
                                forward_or_data_mnk=[case['m'], case['n'], case['k']], alpha=1, epoch=1,
                                dispatch='existing_bf16_production_auto' if arm == 'baseline' else 'calibrated_model_or_domain_fallback')
            if arm == 'model':
                configs[arm][model_key] = copy.deepcopy(model)
        item = dict(case=copy.deepcopy(case), operator=operator, records=[])
        for launch in launches:
            record = dict(launch=launch, phase='forward', scope=scope, warmup=10, iterations=50,
                          configs=copy.deepcopy(configs), correctness_after_capture={arm: correctness(operator) for arm in reporter.ARMS},
                          kernel_configuration_changed=True, rounds=[])
            for index in range(1, 4):
                pair = dict(round=index)
                for arm in reporter.ARMS:
                    pair[arm] = dict(arm=arm, config=copy.deepcopy(configs[arm]),
                                     timing=timing(case, (speedup if arm == 'baseline' else 1) * 10))
                    if index == 3:
                        pair[arm]['correctness_after_formal_samples'] = correctness(operator)
                pair['speedup'] = speedup
                record['rounds'].append(pair)
            record['paired_summary'] = paired_summary(record['rounds'])
            item['records'].append(record)
        document['cases'].append(item)
    return declare_order(document, 'baseline-model') if qkv else document


def change_times(document, durations):
    """Set the first record's three baseline/model pairs consistently."""
    case, record = document['cases'][0]['case'], document['cases'][0]['records'][0]
    for pair, (baseline, model) in zip(record['rounds'], durations):
        for arm, value in (('baseline', baseline), ('model', model)):
            pair[arm]['timing'] = timing(case, value)
        pair['speedup'] = baseline / model
    record['paired_summary'] = paired_summary(record['rounds'])


def declare_order(document, order):
    sequence = list(reporter.arm_sequence(order))
    document['args']['arm_order'] = order
    document['timing'].update(capture_order=sequence,
                              ab_order='_'.join(sequence) + '_repeated_three_times_per_case_and_launch')
    for item in document['cases']:
        for record in item['records']:
            record['capture_order'] = list(sequence)
            for pair in record['rounds']:
                pair['execution_order'] = list(sequence)
    return document


def order_pair(cp=4, speedups=(2., .5)):
    return [(f'/cpu_fixture/cp{cp}_forward.json', fixture(cp, full=True, speedup=speedups[0])),
            (f'/cpu_fixture/cp{cp}_reverse.json', declare_order(fixture(cp, full=True, speedup=speedups[1]),
                                                              'model-baseline'))]


class FrozenPolicyReportTest(unittest.TestCase):
    def test_external_trace_is_rejected_for_both_forward_scopes(self):
        for operator in ('qkv_forward', 'oproj_forward'):
            for location in ('top_level', 'args'):
                document = fixture(operator=operator)
                target = document if location == 'top_level' else document['args']
                target['diagnostic_trace'] = True
                # Even an inconsistent formal=True flag must not admit the trace.
                with self.subTest(operator=operator, location=location), self.assertRaisesRegex(
                        ValueError, 'Externally traced'):
                    reporter.build_report([('diagnostic.json', document)])

    def build(self, document=None, require_full=False):
        return reporter.build_report([('/cpu_fixture/input.json', document or fixture())], require_full)

    def test_three_arm_p50_geomeans_not_pooled_medians(self):
        document = fixture(launches=('graph',))
        change_times(document, [(1, 1), (1, 1), (1000, 1)])
        row = self.build(document)['rows'][0]
        self.assertAlmostEqual(row['graph_baseline_p50_geomean_ms'], .01)
        self.assertAlmostEqual(row['graph_model_p50_geomean_ms'], .001)
        self.assertAlmostEqual(row['graph_speedup'], 10)
        self.assertEqual(row['graph_paired_speedups'], [1, 1, 1000])
        self.assertNotEqual(row['graph_baseline_p50_geomean_ms'], statistics.median([1, 1, 1000]) / 1000)

    def test_full_registry_cp4_and_cp8_and_equal_setting_launch_weight(self):
        report = reporter.build_report([('cp4.json', fixture(4, full=True, speedup=4)),
                                        ('cp8.json', fixture(8, full=True, speedup=1))], require_full=True)
        self.assertEqual(len(report['rows']), 96)
        self.assertEqual([row['settings'] for row in report['coverage']], [48, 48])
        self.assertTrue(all(row['complete'] for row in report['coverage']))
        self.assertAlmostEqual(report['summary']['all']['aggregate']['geometric_mean_speedup'], 2)
        self.assertEqual(report['summary']['all']['aggregate']['records'], 192)
        self.assertEqual(report['summary']['long']['aggregate']['records'], 96)
        self.assertFalse(report['milestone_evidence'])
        self.assertEqual(report['scope'], reporter.SCOPE)

    def test_only_requested_cp_must_be_full_and_partial_launches_remain_missing(self):
        report = self.build(fixture(8, full=True), require_full=True)
        self.assertEqual([row['cp'] for row in report['coverage']], [8])
        for document in (fixture(), fixture(full=True, launches=('graph',))):
            self.assertFalse(self.build(document)['coverage'][0]['complete'])
            with self.assertRaisesRegex(ValueError, 'full original registry'):
                self.build(document, require_full=True)
        text = reporter.markdown(self.build(fixture(launches=('graph',))))
        self.assertIn('— / —', text)

    def test_duplicate_geometries_and_unchanged_regressions_are_retained(self):
        document = fixture(full=True, speedup=.5)
        for item in document['cases']:
            for record in item['records']:
                record['configs']['model']['comm_ctas'] = record['configs']['baseline']['comm_ctas']
                for pair in record['rounds']:
                    pair['model']['config'] = copy.deepcopy(record['configs']['model'])
                record['kernel_configuration_changed'] = False
        report = self.build(document, require_full=True)
        geometries = {(row['global_seq'], row['m'], row['n'], row['k']) for row in report['rows']}
        self.assertLess(len(geometries), len(report['rows']))
        self.assertEqual(len(report['rows']), 48)
        stats = report['summary']['all']['aggregate']
        self.assertEqual((stats['wins'], stats['losses'], stats['ties']), (0, 96, 0))
        self.assertEqual(stats['maximum_latency_regression_percent'], 100)
        self.assertEqual(stats['latency_regressions_over_3pct'], 96)
        unchanged = next(row for row in report['summary']['all']['by_changed'] if not row['changed'])
        self.assertEqual(unchanged['records'], 96)

    def test_long_threshold_is_token_count_not_local_m(self):
        document = fixture(8, full=True)
        report = self.build(document)
        expected = sum(item['case']['global_seq'] >= 131072 for item in document['cases']) * 2
        self.assertEqual(report['summary']['long']['aggregate']['records'], expected)
        self.assertEqual(expected, 48)

    def test_incomplete_or_renamed_semantics_are_rejected(self):
        for key, value in (('complete', False), ('complete', 1), ('schema', 'other'),
                           ('scope', '2F2B_total'), ('milestone_evidence', True),
                           ('semantic', 'fp8_tensor_core'), ('production_policy_written', True),
                           ('formal_ab_protocol', False), ('selection', 'fastest_measured_candidate')):
            document = fixture()
            document[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.build(document)

    def test_manifest_unknown_duplicate_registry_geometry_and_launch_changes_fail(self):
        mutations = [lambda d: d['selected_case_ids'].append('missing'),
                     lambda d: d['cases'][0]['case'].update(n=1),
                     lambda d: d['cases'][0]['records'].pop(),
                     lambda d: d['cases'][0]['records'].append(copy.deepcopy(d['cases'][0]['records'][0]))]
        for mutate in mutations:
            document = fixture()
            mutate(document)
            with self.assertRaises(ValueError):
                self.build(document)
        document = fixture()
        with self.assertRaisesRegex(ValueError, 'Duplicate registry'):
            reporter.build_report([('a.json', document), ('b.json', copy.deepcopy(document))])

    def test_formal_counts_round_order_ratios_and_summary_are_recomputed(self):
        mutations = [lambda d, r: d['timing'].update(warmup_per_arm=2),
                     lambda d, r: r.update(iterations=7),
                     lambda d, r: r['rounds'].pop(),
                     lambda d, r: r['rounds'][0].update(round=2),
                     lambda d, r: r['rounds'][0].update(speedup=5),
                     lambda d, r: r['paired_summary'].update(geometric_mean_speedup=5),
                     lambda d, r: r['paired_summary']['paired_speedups'].__setitem__(0, 5)]
        for mutate in mutations:
            document = fixture()
            mutate(document, document['cases'][0]['records'][0])
            with self.assertRaises(ValueError):
                self.build(document)

    def test_both_execution_orders_preserve_semantic_ratio_and_legacy_default(self):
        legacy = self.build(fixture())
        self.assertEqual(legacy['rows'][0]['arm_order'], 'baseline-model')
        for order in ('baseline-model', 'model-baseline'):
            document = declare_order(fixture(), order)
            report = self.build(document)
            self.assertEqual(report['inputs'][0]['arm_order'], order)
            self.assertEqual(report['rows'][0]['arm_order'], order)
            self.assertEqual(report['rows'][0]['eager_paired_speedups'], [2, 2, 2])
            self.assertAlmostEqual(report['summary']['all']['aggregate']['geometric_mean_speedup'], 2)
            self.assertIn(order, reporter.markdown(report))

    def test_declared_argument_capture_and_each_pair_order_must_match(self):
        mutations = [lambda d: d['args'].update(arm_order='baseline-model'),
                     lambda d: d['timing'].update(ab_order='baseline_model_repeated_three_times_per_case_and_launch'),
                     lambda d: d['timing'].update(capture_order=['baseline', 'model']),
                     lambda d: d['cases'][0]['records'][0].update(capture_order=['baseline', 'model']),
                     lambda d: d['cases'][0]['records'][0]['rounds'][1].update(execution_order=['baseline', 'model']),
                     lambda d: d['cases'][0]['records'][0]['rounds'][0].pop('execution_order'),
                     lambda d: d['args'].update(arm_order='random')]
        for mutate in mutations:
            document = declare_order(fixture(), 'model-baseline')
            mutate(document)
            with self.assertRaises(ValueError):
                self.build(document)
        # A reverse declaration cannot silently acquire legacy baseline-first semantics.
        document = declare_order(fixture(), 'model-baseline')
        document['args'].pop('arm_order')
        with self.assertRaises(ValueError):
            self.build(document)

    def test_raw_vectors_medians_percentiles_and_flops_are_strict(self):
        mutations = [lambda t: t['rank_samples_us'].pop(),
                     lambda t: t['rank_samples_us'][0].pop(),
                     lambda t: t['samples_us'].__setitem__(0, 9),
                     lambda t: t['rank_samples_us'][0].__setitem__(0, math.nan),
                     lambda t: t['rank_samples_us'][0].__setitem__(0, -1),
                     lambda t: t['rank_samples_us'][0].__setitem__(0, True),
                     lambda t: t.update(p50_us=1), lambda t: t.update(p95_us=1),
                     lambda t: t.update(mean_us=1), lambda t: t.update(flops_per_gpu=1),
                     lambda t: t.update(tflops_per_gpu=1)]
        for mutate in mutations:
            document = fixture()
            mutate(document['cases'][0]['records'][0]['rounds'][0]['baseline']['timing'])
            with self.assertRaises(ValueError):
                self.build(document)

    def test_compute_bf16_tolerance_and_exact_dequant_route_are_distinct(self):
        document = fixture()
        checks = document['cases'][0]['records'][0]['correctness_after_capture']['model']
        checks['output'].update(max_abs=.01, relative_rmse=.004)
        self.build(document)
        for name, changes in (('output', dict(relative_rmse=.006)), ('output', dict(all_ranks_finite=False)),
                              ('output', dict(max_abs=math.nan)), ('route', dict(max_abs=.01)),
                              ('weight_dequant', dict(relative_rmse=.001))):
            bad = copy.deepcopy(document)
            bad['cases'][0]['records'][0]['correctness_after_capture']['model'][name].update(changes)
            with self.assertRaises(ValueError):
                self.build(bad)
        document['cases'][0]['records'][0]['rounds'][2]['baseline']['correctness_after_formal_samples'].pop('route')
        with self.assertRaises(ValueError):
            self.build(document)

    def test_native_config_arm_and_changed_flags_cannot_drift(self):
        mutations = [lambda r: r['configs']['baseline'].update(oproj_comm_model=r['configs']['model']['oproj_comm_model']),
                     lambda r: r['configs']['model'].update(requested_comm_ctas=16),
                     lambda r: r['configs']['model'].update(gemm_input='fp8'),
                     lambda r: r['configs']['model'].update(policy_enum=3),
                     lambda r: r['rounds'][0]['model']['config'].update(comm_ctas=24),
                     lambda r: r['rounds'][0]['model'].update(arm='baseline'),
                     lambda r: r.update(kernel_configuration_changed=False)]
        for mutate in mutations:
            document = fixture(launches=('graph',))
            mutate(document['cases'][0]['records'][0])
            with self.assertRaises(ValueError):
                self.build(document)

    def test_source_library_model_and_device_fingerprints_are_preserved_and_checked(self):
        document = fixture()
        inputs = self.build(document)['inputs'][0]
        self.assertEqual(inputs['recorded_sources'], document['sources'])
        self.assertEqual(inputs['gpu_clock_snapshot'], document['gpu_clock_snapshot'])
        mutations = [lambda d: d['sources'].update({'model.json': 'd' * 64}),
                     lambda d: d['sources'].pop('build-mxfp8/libfuse_mxfp8_torch_bridge.so'),
                     lambda d: d['sources'].update({'x': 'not-a-sha'}),
                     lambda d: d['devices'][0].update(sm_count=120),
                     lambda d: d['oproj_comm_model'].update(world_size=8)]
        for mutate in mutations:
            bad = copy.deepcopy(document)
            mutate(bad)
            with self.assertRaises(ValueError):
                self.build(bad)

    def test_writer_has_three_outputs_and_refuses_implicit_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'report'
            report = self.build()
            reporter.write_report(report, output)
            self.assertEqual({path.name for path in output.iterdir()},
                             {'policy_summary.json', 'policy_summary.csv', 'policy_summary.md'})
            with self.assertRaises(FileExistsError):
                reporter.write_report(report, output)
            reporter.write_report(report, output, overwrite=True)
            with (output / 'policy_summary.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(json.loads(rows[0]['eager_model_config'])['comm_ctas'], 16)
            text = (output / 'policy_summary.md').read_text()
            for label in ('| CP | 模型 | S | GEMM M×N×K |', 'Eager baseline / model ms',
                          'Graph baseline / model ms', '不是 pooled median', '不构成四算子 2F2B milestone'):
                self.assertIn(label, text)

    def test_main_records_exact_input_sha_and_never_overwrites_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / 'input.json'
            raw = json.dumps(fixture()).encode()
            source.write_bytes(raw)
            output = directory / 'out'
            reporter.main([str(source), '--output', str(output)])
            report = json.loads((output / 'policy_summary.json').read_text())
            self.assertEqual(report['input_sha256'][str(source)], hashlib.sha256(raw).hexdigest())
            self.assertEqual(source.read_bytes(), raw)
            report['inputs'][0]['source'] = str(output / 'policy_summary.json')
            with self.assertRaisesRegex(ValueError, 'replace an input'):
                reporter.write_report(report, output, overwrite=True)

    def test_balance_orders_has_six_equal_pairs_and_keeps_each_order(self):
        report = reporter.build_report(order_pair(), balance_orders=True)
        self.assertTrue(report['balance_orders'])
        self.assertTrue(report['require_full'])
        self.assertEqual(len(report['rows']), 48)
        self.assertEqual(report['coverage'][0]['original_order_records'], 192)
        self.assertEqual(report['coverage'][0]['paired_rounds_per_setting_launch'], 6)
        self.assertEqual(report['summary']['all']['aggregate']['records'], 96)
        self.assertAlmostEqual(report['summary']['all']['aggregate']['geometric_mean_speedup'], 1)
        row = report['rows'][0]
        self.assertEqual(row['arm_order'], 'balanced_equal_orders')
        self.assertEqual(row['eager_paired_speedups'], [2.] * 3 + [.5] * 3)
        self.assertAlmostEqual(row['eager_baseline_p50_geomean_ms'], .01)
        self.assertAlmostEqual(row['eager_model_p50_geomean_ms'], .01)
        self.assertEqual(row['by_order']['baseline-model']['eager_paired_speedups'], [2.] * 3)
        self.assertEqual(row['by_order']['model-baseline']['eager_paired_speedups'], [.5] * 3)
        self.assertAlmostEqual(report['summary_by_order']['model-baseline']['all']['aggregate']['geometric_mean_speedup'], .5)
        self.assertEqual(len(report['inputs']), 2)
        self.assertIn('six arm p50s', report['latency_statistic'])
        text = reporter.markdown(report)
        self.assertIn('六个 p50', text)
        self.assertIn('不是 pooled median', text)
        self.assertIn('原始顺序分项（不选赢家）', text)

    def test_balanced_both_cp_setting_weight_and_nonpooled_statistic(self):
        report = reporter.build_report(order_pair(4, (4., 1.)) + order_pair(8, (1., 1.)), balance_orders=True)
        self.assertEqual(len(report['rows']), 96)
        self.assertEqual(report['summary']['all']['aggregate']['records'], 192)
        self.assertEqual(report['summary']['long']['aggregate']['records'], 96)
        self.assertAlmostEqual(report['summary']['all']['aggregate']['geometric_mean_speedup'], math.sqrt(2))
        self.assertFalse(report['milestone_evidence'])
        documents = order_pair(speedups=(1., 1.))
        change_times(documents[0][1], [(1, 1), (1, 1), (1000, 1)])
        change_times(documents[1][1], [(1, 1)] * 3)
        row = reporter.build_report(documents, balance_orders=True)['rows'][0]
        self.assertAlmostEqual(row['eager_speedup'], math.sqrt(10))
        self.assertAlmostEqual(row['eager_baseline_p50_geomean_ms'], math.sqrt(10) / 1000)

    def test_balance_rejects_missing_same_order_partial_and_extra_controls(self):
        documents = order_pair()
        bad_inputs = [documents[:1], [documents[0], ('duplicate.json', copy.deepcopy(documents[0][1]))],
                      [documents[0], ('partial.json', declare_order(fixture(), 'model-baseline'))],
                      documents + [('variance.json', fixture())],
                      [documents[0], ('graph-only.json', declare_order(fixture(full=True, launches=('graph',)),
                                                                      'model-baseline'))]]
        for bad in bad_inputs:
            with self.assertRaises(ValueError):
                reporter.build_report(bad, balance_orders=True)
        with self.assertRaisesRegex(ValueError, 'Duplicate registry'):
            reporter.build_report(documents)

    def test_balance_reuses_strict_samples_and_rounds_validation(self):
        documents = order_pair()
        documents[1][1]['cases'][0]['records'][0]['rounds'].pop()
        with self.assertRaisesRegex(ValueError, 'Three ordered'):
            reporter.build_report(documents, balance_orders=True)
        documents = order_pair()
        documents[1][1]['cases'][0]['records'][0]['rounds'][0]['model']['timing']['rank_samples_us'][0][0] = math.nan
        with self.assertRaises(ValueError):
            reporter.build_report(documents, balance_orders=True)

    def test_balance_rejects_native_library_model_versions_and_source_sets(self):
        for path in ('csrc/operators/ulysses_sm90.cu', 'csrc/operators/ulysses_sm90/api/policy.cuh',
                     'include/fuse/operators/primitives/a2a_gemm.h', 'build-mxfp8/libfuse_mxfp8_torch_bridge.so'):
            documents = order_pair()
            documents[1][1]['sources'][path] = '0' * 64
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'different native/library/model'):
                reporter.build_report(documents, balance_orders=True)
        documents = order_pair()
        documents[1][1]['sources']['new_native.cuh'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'different native/library/model'):
            reporter.build_report(documents, balance_orders=True)
        documents = order_pair()
        reverse = documents[1][1]
        reverse['sources']['model.json'] = reverse['oproj_comm_model']['sha256'] = '0' * 64
        for item in reverse['cases']:
            for record in item['records']:
                record['configs']['model']['oproj_comm_model']['sha256'] = '0' * 64
                for pair in record['rounds']:
                    pair['model']['config']['oproj_comm_model']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'different native/library/model'):
            reporter.build_report(documents, balance_orders=True)

    def test_balance_allows_and_records_python_runner_hash_changes(self):
        documents = order_pair()
        path = 'benchmarks/mxfp8_weight/operator_policy_ab.py'
        documents[1][1]['sources'][path] = '0' * 64
        report = reporter.build_report(documents, balance_orders=True)
        self.assertEqual(report['recorded_source_differences'],
                         {path: {documents[0][0]: 'c' * 64, documents[1][0]: '0' * 64}})
        self.assertEqual(report['inputs'][1]['recorded_sources'][path], '0' * 64)

    def test_balance_rejects_config_card_group_and_runtime_differences(self):
        documents = order_pair()
        for item in documents[1][1]['cases']:
            for record in item['records']:
                record['configs']['model']['comm_ctas'] = 24
                for pair in record['rounds']:
                    pair['model']['config']['comm_ctas'] = 24
        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            reporter.build_report(documents, balance_orders=True)
        for change in ('card', 'runtime', 'environment'):
            documents = order_pair()
            if change == 'card':
                documents[1][1]['devices'][0]['name'] = 'different card group'
            elif change == 'runtime':
                documents[1][1]['torch'] = 'other-version'
            else:
                documents[1][1]['policy_environment']['FUSE_A2A_LHS_COMM_POLICY'] = 'experimental_model'
            with self.subTest(change=change), self.assertRaises(ValueError):
                reporter.build_report(documents, balance_orders=True)

    def test_main_balance_flag_writes_only_three_outputs_and_full_source_shas(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for source, document in order_pair():
                path = root / Path(source).name
                path.write_text(json.dumps(document))
                paths.append(path)
            output = root / 'summary'
            reporter.main([*map(str, paths), '--balance-orders', '--output', str(output)])
            result = json.loads((output / 'policy_summary.json').read_text())
            self.assertTrue(result['balance_orders'])
            self.assertTrue(result['require_full'])
            self.assertEqual(len(result['input_sha256']), 2)
            self.assertEqual(len(result['rows']), 48)
            self.assertEqual(len(list(output.iterdir())), 3)


class QkvFrozenPolicyReportTest(unittest.TestCase):
    def build(self, document=None, require_full=False):
        return reporter.build_report([('qkv.json', document or fixture(operator='qkv_forward'))], require_full)

    def orders(self, full=False, launches=('eager', 'graph')):
        first = fixture(operator='qkv_forward', full=full, launches=launches, speedup=4.)
        second = declare_order(fixture(operator='qkv_forward', full=full, launches=launches, speedup=1.),
                               'model-baseline')
        return [('forward.json', first), ('reverse.json', second)]

    def test_full_qkv_coverage_is_24_cp4_long_settings_and_48_boundaries(self):
        report = self.build(fixture(operator='qkv_forward', full=True), require_full=True)
        self.assertEqual(report['schema'], 'mxfp8-qkv-forward-frozen-policy-summary-v1')
        self.assertEqual(report['scope'], reporter.QKV_SCOPE)
        self.assertEqual(len(report['rows']), 24)
        self.assertEqual(report['coverage'], [dict(cp=4, settings=24, expected_settings=24,
                                                  records=48, expected_records=48, complete=True)])
        self.assertEqual(report['summary']['all'], report['summary']['long'])
        self.assertTrue(all(row['id'].startswith('gemm_a2a/') for row in report['rows']))
        self.assertEqual({row['global_seq'] for row in report['rows']}, {131072, 262144, 524288})
        self.assertEqual(report['baseline_reference'], 'unchanged_native_auto_not_external_baseline')
        self.assertFalse(report['external_baseline_comparison_included'])
        text = reporter.markdown(report)
        self.assertIn('QKV F 冻结策略 A/B', text)
        self.assertIn('不是相对 TEUB/cuBLASLt', text)
        self.assertNotIn('只测 OProj', text)

    def test_qkv_bf16_local_gemm_is_not_bit_exact_route(self):
        document = fixture(operator='qkv_forward')
        checks = document['cases'][0]['records'][0]['correctness_after_capture']['model']
        checks['local_gemm'].update(max_abs=.02, relative_rmse=.004)
        checks['output'].update(max_abs=.01, relative_rmse=.005)
        self.build(document)
        for name, values in (('local_gemm', dict(relative_rmse=.0051)),
                             ('local_gemm', dict(all_ranks_finite=False)),
                             ('local_gemm', dict(max_abs=-1)), ('local_gemm', dict(max_abs=math.inf)),
                             ('weight_dequant', dict(max_abs=.01))):
            bad = copy.deepcopy(document)
            bad['cases'][0]['records'][0]['correctness_after_capture']['model'][name].update(values)
            with self.subTest(name=name, values=values), self.assertRaises(ValueError):
                self.build(bad)
        bad = copy.deepcopy(document)
        check = bad['cases'][0]['records'][0]['rounds'][2]['model']['correctness_after_formal_samples']
        check['route'] = check.pop('local_gemm')
        with self.assertRaisesRegex(ValueError, 'local_gemm'):
            self.build(bad)

    def test_qkv_zero_policy_placeholder_and_real_fallback_tile_are_retained(self):
        document = fixture(operator='qkv_forward', launches=('graph',))
        record = document['cases'][0]['records'][0]
        self.assertEqual(self.build(document)['rows'][0]['graph_model_config']['policy_enum'], 0)
        for config in record['configs'].values():
            config.update(tile_n=320, comm_ctas=4)
        for pair in record['rounds']:
            for arm in reporter.ARMS:
                pair[arm]['config'] = copy.deepcopy(record['configs'][arm])
        record['kernel_configuration_changed'] = False
        report = self.build(document)
        self.assertFalse(report['rows'][0]['graph_changed'])
        self.assertEqual(report['rows'][0]['graph_model_config']['tile_n'], 320)
        for values in (dict(policy_enum=4), dict(policy_enum=False), dict(tile_m=64),
                       dict(tile_k=128), dict(tile_n=192), dict(cluster_m=1)):
            bad = copy.deepcopy(document)
            bad['cases'][0]['records'][0]['configs']['model'].update(values)
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.build(bad)

    def test_qkv_rejects_cp8_short_scope_wrong_schema_and_oproj_metadata(self):
        with self.assertRaises(ValueError):
            self.build(fixture(cp=8, operator='qkv_forward'))
        for mutation in (lambda d: d.update(schema='mxfp8-frozen-policy-ab-v1'),
                         lambda d: d.update(scope=reporter.SCOPE),
                         lambda d: d['args'].pop('arm_order'),
                         lambda d: d['cases'][0]['records'][0].update(scope=reporter.SCOPE),
                         lambda d: d['cases'][0]['records'][0]['configs']['baseline'].update(
                             qkv_forward_comm_model=d['qkv_forward_comm_model']),
                         lambda d: d['cases'][0]['records'][0]['configs']['model'].update(oproj_comm_model={})):
            document = fixture(operator='qkv_forward')
            mutation(document)
            with self.assertRaises(ValueError):
                self.build(document)
        document = fixture(operator='qkv_forward')
        short = next(c for c in full_matrix() if c['direction'] == 'gemm_a2a' and c['cp'] == 4)
        document['cases'][0]['case'] = short
        document['selected_case_ids'] = [short['id']]
        with self.assertRaisesRegex(ValueError, 'original registry'):
            self.build(document)
        with self.assertRaisesRegex(ValueError, 'mixed scope'):
            reporter.build_report([('oproj.json', fixture()), ('qkv.json', fixture(operator='qkv_forward'))])

    def test_qkv_model_domain_coefficients_and_source_are_strict(self):
        mutations = [lambda m: m.update(operator='oproj_forward'), lambda m: m.update(world_size=8),
                     lambda m: m.update(sm_count=132.), lambda m: m.update(sha256='0' * 64),
                     lambda m: m['domain'].update(tile_families=[[128, 320, 64, 2]]),
                     lambda m: m['domain'].update(head_dim=128.),
                     lambda m: m['domain'].update(comm_ctas=[4, 6]),
                     lambda m: m['coefficients'].update(compute_gflop_sm_us=0),
                     lambda m: m['coefficients'].update(route_slot_task_us=math.nan),
                     lambda m: m['coefficients'].update(route_slot_task_us=True),
                     lambda m: m.update(minimum_gain=1), lambda m: m.update(minimum_gain=math.inf),
                     lambda m: m.update(scope='2F2B_total')]
        for mutation in mutations:
            document = fixture(operator='qkv_forward')
            mutation(document['qkv_forward_comm_model'])
            with self.assertRaises(ValueError):
                self.build(document)

    def test_qkv_balances_matching_complete_subset_without_claiming_full_registry(self):
        report = reporter.build_report(self.orders(), balance_orders=True)
        self.assertEqual(len(report['rows']), 1)
        self.assertFalse(report['require_full'])
        self.assertFalse(report['coverage'][0]['complete'])
        self.assertTrue(report['coverage'][0]['paired_subset_complete'])
        self.assertEqual(report['coverage'][0]['expected_settings'], 24)
        row = report['rows'][0]
        self.assertEqual(row['graph_paired_speedups'], [4.] * 3 + [1.] * 3)
        self.assertAlmostEqual(row['graph_speedup'], 2.)
        self.assertAlmostEqual(row['graph_baseline_p50_geomean_ms'], .02)
        self.assertEqual(len(report['summary_by_order']), 2)
        with self.assertRaisesRegex(ValueError, 'full original registry'):
            reporter.build_report(self.orders(), require_full=True, balance_orders=True)
        full = reporter.build_report(self.orders(full=True), require_full=True, balance_orders=True)
        self.assertTrue(full['coverage'][0]['complete'])
        self.assertTrue(full['require_full'])
        self.assertEqual(full['summary']['all']['aggregate']['records'], 48)

    def test_qkv_balances_graph_only_subset_and_preserves_unmeasured_eager(self):
        documents = self.orders(launches=('graph',))
        change_times(documents[0][1], [(1, 1), (1, 1), (1000, 1)])
        change_times(documents[1][1], [(1, 1)] * 3)
        report = reporter.build_report(documents, balance_orders=True)
        row = report['rows'][0]
        self.assertIsNone(row['eager_speedup'])
        self.assertIsNone(row['eager_paired_speedups'])
        self.assertAlmostEqual(row['graph_speedup'], math.sqrt(10))
        self.assertEqual(report['coverage'][0]['records'], 1)
        self.assertIn('— / —', reporter.markdown(report))

    def test_qkv_balance_rejects_manifest_launch_and_any_source_difference(self):
        for path in ('benchmarks/mxfp8_weight/operator_policy_ab.py', 'model.json',
                     'csrc/operators/ulysses_sm90/api/policy.cuh', 'build-mxfp8/libfuse_mxfp8_torch_bridge.so'):
            documents = self.orders()
            documents[1][1]['sources'][path] = '0' * 64
            with self.subTest(path=path), self.assertRaises(ValueError):
                reporter.build_report(documents, balance_orders=True)
        documents = self.orders()
        documents[1][1]['sources']['additional.py'] = '1' * 64
        with self.assertRaisesRegex(ValueError, 'source sets'):
            reporter.build_report(documents, balance_orders=True)
        documents = self.orders()
        documents[1] = ('reverse.json', declare_order(fixture(operator='qkv_forward', full=True), 'model-baseline'))
        with self.assertRaisesRegex(ValueError, 'same complete setting/launch subset'):
            reporter.build_report(documents, balance_orders=True)
        documents = self.orders()
        documents[1] = ('reverse.json', declare_order(fixture(operator='qkv_forward', launches=('graph',)), 'model-baseline'))
        with self.assertRaisesRegex(ValueError, 'same complete setting/launch subset'):
            reporter.build_report(documents, balance_orders=True)

    def test_qkv_balance_rejects_config_device_runtime_and_extra_control_drift(self):
        documents = self.orders()
        for record in documents[1][1]['cases'][0]['records']:
            record['configs']['model']['comm_ctas'] = 8
            for pair in record['rounds']:
                pair['model']['config']['comm_ctas'] = 8
        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            reporter.build_report(documents, balance_orders=True)
        for mutation in (lambda d: d['devices'][0].update(uuid='other'), lambda d: d.update(cuda='different'),
                         lambda d: d['policy_environment'].update(FUSE_QKV_GEMM_POLICY='m128n320')):
            documents = self.orders()
            mutation(documents[1][1])
            with self.assertRaises(ValueError):
                reporter.build_report(documents, balance_orders=True)
        documents = self.orders()
        with self.assertRaises(ValueError):
            reporter.build_report(documents + [('extra.json', copy.deepcopy(documents[0][1]))], balance_orders=True)

    def test_qkv_strict_timing_is_reused_and_cli_writes_only_existing_three_outputs(self):
        documents = self.orders()
        documents[1][1]['cases'][0]['records'][0]['rounds'][0]['model']['timing']['rank_samples_us'][0][0] = math.nan
        with self.assertRaises(ValueError):
            reporter.build_report(documents, balance_orders=True)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            paths = []
            for name, document in self.orders():
                path = directory / name
                path.write_text(json.dumps(document))
                paths.append(str(path))
            output = directory / 'report'
            reporter.main([*paths, '--balance-orders', '--output', str(output)])
            self.assertEqual({p.name for p in output.iterdir()},
                             {'policy_summary.json', 'policy_summary.csv', 'policy_summary.md'})
            report = json.loads((output / 'policy_summary.json').read_text())
            self.assertEqual(report['scope'], reporter.QKV_SCOPE)
            self.assertEqual(len(report['input_sha256']), 2)


def archived_comparison_fixture():
    from test_optimization_report import complete_fixture

    rows, metadata = complete_fixture()
    metadata['input_sha256'] = {'formal.json': '1' * 64, 'published/forward_best.json': '2' * 64,
                                'published/backward_best.json': '3' * 64}
    scope = reporter.registry_for('qkv_forward')
    count = 0
    for row in rows:
        if row['id'] not in scope:
            continue
        ratio = 1. if count < len(scope) // 2 else 4.
        count += 1
        for launch in reporter.LAUNCHES:
            row.update({launch + '_p50_ms': .01, launch + '_best_external_p50_ms': ratio * .01,
                        launch + '_speedup_over_best_external': ratio,
                        launch + '_config': dict(comm_ctas=4, tile_m=128, tile_n=256, tile_k=64, cluster_m=2),
                        launch + '_best_external_config': dict(operator=dict(chunks=2, sms=16),
                                                               gemm_plan=dict(algo_id=7), nccl_environment={})})
    return rows, metadata


class QkvArchivedReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents = [('forward.json', fixture(operator='qkv_forward', full=True, speedup=4.)),
                         ('reverse.json', declare_order(fixture(operator='qkv_forward', full=True, speedup=1.),
                                                        'model-baseline'))]
        cls.report = reporter.build_report(cls.documents, balance_orders=True, require_full=True)
        cls.archived_rows, cls.metadata = archived_comparison_fixture()

    def join(self, report=None, rows=None, metadata=None):
        return reporter.add_reference_comparison(self.report if report is None else report,
                                                 self.archived_rows if rows is None else rows,
                                                 self.metadata if metadata is None else metadata)

    def test_full_archive_and_fixed_original_scope_mean_not_current_winner_mean(self):
        result = self.join()
        reference = result['reference_comparison']
        self.assertEqual(reference['archived_coverage']['measured_boundary_records'], 1152)
        self.assertEqual(len(reference['boundaries']), 48)
        self.assertEqual(len(result['rows']), 24)
        self.assertAlmostEqual(reference['original_scope_mean'], 2.)
        summary = reference['summary']
        self.assertEqual(summary['original_below_scope_mean'], 24)
        self.assertEqual(summary['current_model_reaches_original_scope_mean'], 24)
        self.assertEqual(summary['original_bad_reaching_scope_mean'], 0)
        self.assertEqual([group['records'] for group in reference['by_launch']], [24, 24])
        self.assertEqual(reference['recorded_archive_input_sha256'], self.metadata['input_sha256'])
        self.assertFalse(reference['milestone_evidence'])
        self.assertFalse(reference['hardware_upper_bound'])
        self.assertFalse(self.report['external_baseline_comparison_included'])
        self.assertNotIn('reference_comparison', self.report)

    def test_raw_latency_drift_is_not_multiplied_into_archived_speedup(self):
        reference = self.join()['reference_comparison']
        boundary = reference['boundaries'][0]
        self.assertAlmostEqual(boundary['paired_speedup_over_current_auto'], 2.)
        self.assertAlmostEqual(boundary['current_auto_six_p50_geomean_ms'], .02)
        self.assertAlmostEqual(boundary['current_model_six_p50_geomean_ms'], .01)
        self.assertEqual(boundary['original_fuse_p50_ms'], .01)
        self.assertAlmostEqual(boundary['original_external_over_fuse'], 1.)
        self.assertAlmostEqual(boundary['archived_external_over_current_model'], 1.)
        self.assertAlmostEqual(boundary['archived_external_over_current_auto'], .5)
        drift = boundary['auto_latency_drift']
        self.assertAlmostEqual(drift['current_auto_over_original_fuse'], 2.)
        self.assertAlmostEqual(drift['reverse_auto_over_forward_auto'], .25)
        self.assertAlmostEqual(drift['auto_p50_geomean_ms_by_order']['baseline-model'], .04)
        self.assertAlmostEqual(drift['auto_p50_geomean_ms_by_order']['model-baseline'], .01)
        self.assertAlmostEqual(reference['summary']['archived_external_over_current_model_geomean'], 2.)
        self.assertAlmostEqual(reference['summary']['paired_speedup_over_current_auto_geomean'], 2.)
        self.assertNotAlmostEqual(reference['summary']['archived_external_over_current_model_geomean'],
                                  reference['original_scope_mean'] * 2.)

    def test_archive_entire_matrix_is_audited_not_just_joined_qkv_subset(self):
        for mutation in ('missing_other_op', 'missing_other_launch', 'geometry', 'actual_backward_total', 'stored_ratio'):
            rows = copy.deepcopy(self.archived_rows)
            index = next(i for i, row in enumerate(rows) if row['operator'] == 'oproj_backward')
            if mutation == 'missing_other_op':
                rows.pop(index)
            elif mutation == 'missing_other_launch':
                rows[index]['eager_p50_ms'] = None
            elif mutation == 'geometry':
                rows[index]['w_mnk'][0] += 1
            elif mutation == 'actual_backward_total':
                rows[index]['graph_total_p50_ms'] *= 2
            else:
                rows[index]['graph_speedup_over_best_external'] *= 2
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.join(rows=rows)

    def test_archive_requires_formal_metadata_sources_and_winner_configs(self):
        for mutation in (lambda m: m.update(schema='other'), lambda m: m.update(diagnostic=True),
                         lambda m: m['measurement_reports'][0].update(complete=False),
                         lambda m: m.update(baseline_winners=48), lambda m: m.update(input_sha256={}),
                         lambda m: m['input_sha256'].update({'formal.json': 'bad'}),
                         lambda m: m['input_sha256'].pop('formal.json')):
            metadata = copy.deepcopy(self.metadata)
            mutation(metadata)
            with self.assertRaises(ValueError):
                self.join(metadata=metadata)
        for field in ('graph_best_external_config', 'eager_config'):
            rows = copy.deepcopy(self.archived_rows)
            row = next(row for row in rows if row['id'] in reporter.registry_for('qkv_forward'))
            row.pop(field)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'configuration is missing'):
                self.join(rows=rows)

    def test_reference_requires_full_balanced_qkv_eager_and_graph(self):
        invalid = []
        other = copy.deepcopy(self.report)
        other['scope'] = reporter.SCOPE
        invalid.append(other)
        single = copy.deepcopy(self.report)
        single['balance_orders'] = False
        invalid.append(single)
        subset = copy.deepcopy(self.report)
        subset['rows'].pop()
        invalid.append(subset)
        graph_only = copy.deepcopy(self.report)
        for row in graph_only['rows']:
            row['eager_speedup'] = None
        invalid.append(graph_only)
        for report in invalid:
            with self.assertRaises(ValueError):
                self.join(report=report)

    def test_unchanged_and_regressions_retained_and_full_configs_preserved(self):
        documents = copy.deepcopy(self.documents)
        for _, document in documents:
            for record in document['cases'][0]['records']:
                record['configs']['model']['comm_ctas'] = record['configs']['baseline']['comm_ctas']
                record['kernel_configuration_changed'] = False
                for pair in record['rounds']:
                    pair['model']['config'] = copy.deepcopy(record['configs']['model'])
                    pair['baseline']['timing'] = copy.deepcopy(pair['model']['timing'])
                    pair['speedup'] = 1.
                record['paired_summary'] = paired_summary(record['rounds'])
            for record in document['cases'][1]['records']:
                for pair in record['rounds']:
                    pair['baseline']['timing'] = timing(document['cases'][1]['case'], 5.)
                    pair['speedup'] = .5
                record['paired_summary'] = paired_summary(record['rounds'])
        report = reporter.build_report(documents, balance_orders=True, require_full=True)
        result = self.join(report=report)
        boundaries = result['reference_comparison']['boundaries']
        self.assertEqual(len(boundaries), 48)
        self.assertFalse(boundaries[0]['kernel_configuration_changed'])
        self.assertAlmostEqual(boundaries[0]['paired_speedup_over_current_auto'], 1.)
        self.assertAlmostEqual(boundaries[2]['paired_speedup_over_current_auto'], .5)
        archived = next(row for row in self.archived_rows if row['id'] == boundaries[0]['id'])
        self.assertEqual(boundaries[0]['original_fuse_config'], archived['eager_config'])
        self.assertEqual(boundaries[0]['archived_external_config'], archived['eager_best_external_config'])
        self.assertEqual(result['rows'][0]['eager_reference_comparison']['archived_external_config'],
                         boundaries[0]['archived_external_config'])

    def test_markdown_separates_paired_ab_archived_reference_and_non_clock_drift(self):
        text = reporter.markdown(self.join())
        self.assertIn('归档外部基线参考（非新实测）', text)
        self.assertIn('不是时钟测量', text)
        self.assertIn('共六个 p50 的几何平均', text)
        self.assertIn('没有重新实测', text)
        self.assertIn('GM=2.000000000×', text)
        self.assertNotIn('未包含外部基线比较。', text)
        self.assertIn('未包含外部基线比较。', reporter.markdown(self.report))

    def test_cli_reference_hashes_companion_and_only_existing_three_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            paths = []
            for name, document in self.documents:
                path = directory / name
                path.write_text(json.dumps(document))
                paths.append(path)
            comparison = directory / 'comparison_summary.json'
            metadata = directory / 'comparison_metadata.json'
            comparison.write_text(json.dumps(self.archived_rows))
            metadata.write_text(json.dumps(self.metadata))
            originals = {str(path): path.read_bytes() for path in [*paths, comparison, metadata]}
            output = directory / 'report'
            reporter.main([*map(str, paths), '--balance-orders', '--reference-comparison', str(comparison),
                           '--output', str(output)])
            result = json.loads((output / 'policy_summary.json').read_text())
            self.assertEqual(len(result['input_sha256']), 2)
            self.assertEqual(result['reference_comparison']['input_sha256'],
                             {str(path): hashlib.sha256(originals[str(path)]).hexdigest()
                              for path in (comparison, metadata)})
            self.assertEqual(originals, {path: Path(path).read_bytes() for path in originals})
            self.assertEqual({path.name for path in output.iterdir()},
                             {'policy_summary.json', 'policy_summary.csv', 'policy_summary.md'})
            with (output / 'policy_summary.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            config = json.loads(rows[0]['eager_reference_comparison'])['archived_external_config']
            self.assertEqual(config['gemm_plan']['algo_id'], 7)
            with self.assertRaises(FileExistsError):
                reporter.main([*map(str, paths), '--balance-orders', '--reference-comparison', str(comparison),
                               '--output', str(output)])

    def test_reference_cli_requires_companion_and_protects_archive_from_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            paths = []
            for name, document in self.documents:
                path = directory / name
                path.write_text(json.dumps(document))
                paths.append(path)
            reference = directory / 'comparison_summary.json'
            reference.write_text(json.dumps(self.archived_rows))
            with self.assertRaises(FileNotFoundError):
                reporter.main([*map(str, paths), '--balance-orders', '--reference-comparison', str(reference),
                               '--output', str(directory / 'output')])
            report = self.join()
            target = directory / 'policy_summary.json'
            report['reference_comparison']['input_sha256'] = {str(target): '0' * 64}
            with self.assertRaisesRegex(ValueError, 'must not replace an input'):
                reporter.write_report(report, directory, overwrite=True)


if __name__ == '__main__':
    unittest.main()
