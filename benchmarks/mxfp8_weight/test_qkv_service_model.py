"""CPU-only contracts for QKV primitive raw data and integer work features."""

import copy
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import unittest
from unittest import mock

import qkv_service_model as model


def fixture(cp=8, c=12, m=16384, n=2048, q=16, kv=8):
    k = (q + 2 * kv) * 128
    case = dict(id='qkv_backward/ignored_label', operator='qkv', model='ignored_label',
                m=m, hidden=n, q_heads=q, kv_heads=kv, head_dim=128, cp=cp,
                batch=1, global_seq=m * cp, b_mnk=[m, n, k], layout='causal_paired')
    config = dict(tile_m=128, tile_n=256, tile_k=64, cluster_m=2,
                  comm_ctas=c, requested_comm_ctas=c, sm_count=132, epoch=1,
                  ready_elements=(m // 128) * (q + 2 * kv) * 32,
                  weight_block_size=32, payload='e4m3', scale='e8m0',
                  gemm_input='bf16', gemm_accumulator='fp32', weight_axis='original_forward_K',
                  route_layout=case['layout'], forward_or_data_mnk=[m, n, k],
                  raster='n', swizzle=1, copy_schedule='native_qkv_peer_head_route',
                  compute_scheduler='matched_tile_subgrid_or_fullgrid_bare_or_preloaded_ready',
                  primitive_scope=model.SCOPE)
    resources = {}
    work_ctas = model.ceil_div(model.ceil_div(m, 128), 2) * model.ceil_div(n, 256) * 2
    for name in model.PRIMITIVES:
        is_copy = name in model.COPY_PRIMITIVES
        budget = 132 if name.endswith('fullgrid') else 132 - c
        resources[name] = dict(selected_gemm_stages=4, ready_block_m=128,
                               ready_flag_stride=32, packed_heads=q + 2 * kv,
                               k_tiles_per_head=2, threads_per_cta=384,
                               primitive_grid_x=c if is_copy else min(budget, work_ctas),
                               primitive_launch_cluster_m=1 if name == 'copy' else 2,
                               primitive_dynamic_smem_bytes=196704 if name == 'copy' else 214016,
                               primitive_registers_per_thread=38 if is_copy else 168,
                               copy_use_tma=int(is_copy), copy_slots=12 if is_copy else 0,
                               copy_fields_apply=is_copy)
    return case, config, resources


def calibration(cp=8, requests=(12,), selected=None, **geometry):
    selected = list(model.PRIMITIVES) if selected is None else list(selected)
    document = dict(schema=model.SCHEMA, complete=True, operator='qkv_backward',
                    phase='data', weight_mode='deferred', production_policy_written=False,
                    milestone_evidence=False, primitive_ids=dict(model.PRIMITIVES),
                    scope='independent_services_' + model.SCOPE,
                    readiness='int32_epoch_preloaded_after_reset_into_actual_strided_flag_prefix',
                    preparation='bare_and_ready_compute_share_identical_preparation_each_sample',
                    staging='peer_arena_data_bound_by_Arguments_not_unbound_tensors_staging',
                    args=dict(cp=cp, operator='qkv_backward', launch='graph', warmup=10,
                              iterations=50, comm_ctas=list(requests), primitives=selected),
                    sources={'cpu_fixture_not_a_gpu_result': 'a' * 64},
                    gpu_clock_snapshot='CPU fixture; no measured GPU clock',
                    devices=[dict(rank=rank, cc=[9, 0], sm_count=132) for rank in range(cp)],
                    timing=dict(sample_statistic='sample_wise_max_across_ranks',
                                included='one_existing_BF16_compute_or_route_primitive',
                                excluded=['DQ', 'W', 'finalize', 'warmup', 'other_primitive',
                                          'allocation_IPC_capture_references',
                                          'staging_preload_control_reset_CPU_barrier',
                                          'integer_ready_prefix_restore']), samples=[])
    for request in requests:
        case, config, resources = fixture(cp=cp, c=request or 4, **geometry)
        config['requested_comm_ctas'] = request
        primitives = {}
        feature = model.features(case, config, 'copy', resources['copy'])
        for name in selected:
            is_copy = name in model.COPY_PRIMITIVES
            # A slightly faster preloaded arm is legal noise, not negative wait.
            offset = 100. if is_copy else 80. if 'bare' in name else 79.
            ranks = [[offset + index * .125 + rank * .5 for index in range(50)]
                     for rank in range(cp)]
            values = [max(rank[index] for rank in ranks) for index in range(50)]
            p50 = statistics.median(values)
            flops = 0 if is_copy else feature['flops_per_gpu']
            timing = dict(p50_us=p50, p95_us=values[46] + .55 * (values[47] - values[46]),
                          mean_us=statistics.fmean(values), samples_us=values, rank_samples_us=ranks,
                          flops_per_gpu=flops, tflops_per_gpu=flops / p50 / 1e6)
            primitives[name] = dict(primitive_id=model.PRIMITIVES[name], resources=resources[name],
                                    scope=model.SCOPE, included='copy_only' if is_copy else 'BF16_GEMM_only',
                                    ready_prepublication_bytes=0 if is_copy else feature['ready_prefix_bytes'],
                                    ready_epoch=None if is_copy else 1,
                                    ready_prepublication=('not_used_by_copy' if is_copy else
                                                         'same_int32_epoch_prefix_for_bare_and_preloaded'),
                                    timing=timing, correctness=dict(max_abs=0., relative_rmse=0.,
                                                                  all_ranks_finite=True))
        work = {key: feature[key] for key in ('m', 'n', 'k', 'cp', 'flops_per_gpu',
                'remote_payload_bytes_per_gpu', 'staging_write_bytes_per_gpu',
                'compulsory_gemm_bytes_per_gpu')}
        document['samples'].append(dict(case=case, config=config, primitives=primitives, work=work))
    document['expected_samples'] = len(document['samples'])
    document['expected_primitive_measurements'] = len(document['samples']) * len(selected)
    return document


FIT_COEFFICIENTS = {name: ((2.7, 17.) if name == 'copy' else
                          (2.1, 23.) if name == 'copy_fused_reservation' else
                          (172000., 3.5)) for name in model.PRIMITIVES}


def fitting_calibration():
    """Small synthetic CPU measurements, deliberately not benchmark evidence."""
    document = calibration(requests=(4, 12, 32))
    for geometry in (dict(n=5120, q=24), dict(n=3072, q=48)):
        document['samples'].extend(calibration(requests=(4, 12, 32), **geometry)['samples'])
    document['expected_samples'] = len(document['samples'])
    document['expected_primitive_measurements'] = len(document['samples']) * len(model.PRIMITIVES)
    for sample in document['samples']:
        for name, primitive in sample['primitives'].items():
            feature = model.features(sample['case'], sample['config'], name, primitive['resources'])
            first, second = FIT_COEFFICIENTS[name]
            if name in model.COPY_PRIMITIVES:
                service = math.hypot(first * feature['remote_payload_bytes_per_gpu'] / 2**20,
                                     second * feature['copy_slot_task_depth'])
            else:
                service = feature['compute_cluster_waves'] * (
                    first * feature['padded_flops_per_output_tile'] / 1e12 + second)
            value = 5. + service
            timing = primitive['timing']
            timing.update(p50_us=value, p95_us=value, mean_us=value,
                          samples_us=[value] * 50, rank_samples_us=[[value] * 50 for _ in range(8)],
                          tflops_per_gpu=timing['flops_per_gpu'] / value / 1e6)
    return document


class QkvServiceModelTest(unittest.TestCase):
    def test_import_is_standard_library_only(self):
        code = ('import sys; import qkv_service_model; '
                'assert not ({"torch", "numpy", "scipy", "cuda"} & set(sys.modules))')
        subprocess.run([sys.executable, '-S', '-c', code], cwd=Path(model.__file__).parent, check=True)

    def test_task_payload_is_two_chunks_in_one_slot_not_two_tasks(self):
        case, config, resources = fixture()
        feature = model.features(case, config, 'copy', resources['copy'])
        self.assertEqual(feature['copy_tasks'], 128 * 32)
        self.assertEqual((feature['copy_stage_bytes'], feature['copy_task_bytes']), (16384, 32768))
        self.assertEqual(feature['copy_tasks'] * feature['copy_task_bytes'], 2 * 16384 * 4096)
        self.assertEqual(feature['remote_payload_bytes_per_gpu'], 117440512)
        self.assertEqual(feature['copy_tma_loads'], 8192)
        self.assertEqual(feature['copy_tma_stores'], 8192)
        self.assertEqual((feature['copy_slot_task_depth'], feature['copy_slot_chunk_depth']), (29, 58))
        self.assertEqual(feature['native_resources']['primitive_dynamic_smem_bytes'], 196704)
        self.assertNotIn('compute_cluster_waves', feature)

    def test_slot_depth_matches_independent_integer_task_assignment(self):
        for m in (128, 640, 16384):
            for cp in (4, 8):
                for c in (4, 12, 32):
                    case, config, resources = fixture(cp=cp, c=c, m=m)
                    feature = model.features(case, config, 'copy', resources['copy'])
                    tasks = feature['copy_tasks']
                    lengths = [len(range(s * c + j, tasks, 12 * c))
                               for s in range(12) for j in range(c)]
                    self.assertEqual(sum(lengths), tasks)
                    self.assertEqual(max(lengths), feature['copy_slot_task_depth'])
                    self.assertEqual(max(len(range(j, tasks, c)) for j in range(c)),
                                     feature['copy_max_tasks_per_cta'])
        case, config, resources = fixture(m=128, c=4)
        feature = model.features(case, config, 'copy', resources['copy'])
        self.assertEqual(feature['copy_slot_chunk_depth'], 2)
        self.assertEqual(model.ceil_div(2 * feature['copy_tasks'], 12 * 4), 2)
        # 64 tasks / 48 slots: max slot has two tasks, hence four chunks, not three.
        case, config, resources = fixture(m=256, c=4)
        feature = model.features(case, config, 'copy', resources['copy'])
        self.assertEqual((feature['copy_slot_chunk_depth'], model.ceil_div(128, 48)), (4, 3))

    def test_cluster_integer_waves_and_small_grid_truncation(self):
        case, config, resources = fixture(m=640, n=16384, c=16)
        feature = model.features(case, config, 'compute_bare_subgrid', resources['compute_bare_subgrid'])
        self.assertEqual((feature['cluster_work_tiles'], feature['compute_grid_clusters'],
                          feature['compute_cluster_waves']), (192, 58, 4))
        self.assertEqual(model.ceil_div(5 * 64, 116), 3)  # Wrong independent-CTA shortcut.
        case, config, resources = fixture(m=128, n=256)
        for name in ('compute_bare_subgrid', 'compute_bare_fullgrid'):
            feature = model.features(case, config, name, resources[name])
            self.assertEqual((feature['compute_grid_ctas'], feature['compute_cluster_waves']), (2, 1))

    def test_flops_heads_and_ready_stride_are_explicit(self):
        case, config, resources = fixture()
        feature = model.features(case, config, 'compute_bare_subgrid', resources['compute_bare_subgrid'])
        self.assertEqual((feature['heads_per_output_tile'], feature['k_tiles_per_head']), (32, 2))
        self.assertEqual(feature['padded_flops_per_output_tile'], 32 * 2 * 128 * 256 * 128)
        self.assertEqual(feature['padded_flops_per_head_per_output_tile'], 2 * feature['padded_flops_per_k_tile'])
        self.assertEqual(feature['ready_prefix_bytes'], 128 * 32 * 32 * 4)
        self.assertEqual(feature['native_resources']['primitive_registers_per_thread'], 168)
        renamed = dict(case, id='arbitrary', model='not_a_training_lookup')
        self.assertEqual(feature, model.features(renamed, config, 'compute_bare_subgrid',
                                                resources['compute_bare_subgrid']))

    def test_domain_and_stale_native_metadata_are_rejected(self):
        case, config, resources = fixture()
        for change in (dict(m=129), dict(hidden=2050), dict(cp=2), dict(q_heads=17),
                       dict(kv_heads=7), dict(head_dim=64), dict(batch=2),
                       dict(layout='unknown'), dict(global_seq=1), dict(b_mnk=[1, 2, 3])):
            with self.subTest(case=change), self.assertRaises(ValueError):
                model.features(dict(case, **change), config, 'copy', resources['copy'])
        for change in (dict(tile_n=320), dict(tile_k=128), dict(cluster_m=1),
                       dict(comm_ctas=13), dict(comm_ctas=132), dict(requested_comm_ctas=16),
                       dict(gemm_input='fp8'), dict(swizzle=2), dict(epoch=0),
                       dict(epoch=2**32), dict(ready_elements=1), dict(route_layout='contiguous')):
            with self.subTest(config=change), self.assertRaises(ValueError):
                model.features(case, dict(config, **change), 'copy', resources['copy'])

    def test_resource_contract_rejects_scalar_copy_and_wrong_compute_grid(self):
        case, config, resources = fixture()
        shared_bad = (dict(selected_gemm_stages=3), dict(ready_block_m=64),
                      dict(ready_flag_stride=128), dict(packed_heads=16), dict(k_tiles_per_head=1),
                      dict(threads_per_cta=256), dict(primitive_dynamic_smem_bytes=0),
                      dict(primitive_registers_per_thread=0), dict(primitive_registers_per_thread=256))
        for name in ('copy', 'compute_bare_subgrid'):
            extra = ((dict(copy_use_tma=0), dict(copy_slots=6), dict(copy_fields_apply=False),
                      dict(primitive_launch_cluster_m=2), dict(primitive_grid_x=10),
                      dict(primitive_dynamic_smem_bytes=196608)) if name == 'copy' else
                     (dict(copy_use_tma=1), dict(copy_slots=12), dict(copy_fields_apply=True),
                      dict(primitive_launch_cluster_m=1), dict(primitive_grid_x=132),
                      dict(primitive_dynamic_smem_bytes=196704)))
            for change in shared_bad + extra:
                with self.subTest(name=name, change=change), self.assertRaises(ValueError):
                    model.features(case, config, name, dict(resources[name], **change))

    def test_complete_raw_data_and_signed_adapter_overhead(self):
        document = calibration(requests=(0, 4, 12))
        original = copy.deepcopy(document)
        rows = model.calibration_rows(document)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]['adapter_overhead_us'], dict(subgrid=-1., fullgrid=-1.))
        self.assertEqual(document, original)
        self.assertEqual(rows[0]['primitives']['copy']['timing']['tflops_per_gpu'], 0.)
        self.assertNotIn('wait', repr(rows[0]['adapter_overhead_us']))

    def test_declared_primitive_subset_is_complete_but_cannot_fake_a_pair(self):
        document = calibration(selected=('compute_bare_subgrid', 'copy'))
        self.assertEqual(model.calibration_rows(document)[0]['adapter_overhead_us'], {})
        del document['samples'][0]['primitives']['copy']
        with self.assertRaisesRegex(ValueError, 'missing'):
            model.calibration_rows(document)

    def test_longer_formal_runs_are_accepted_and_all_error_metrics_are_finite(self):
        document = calibration(cp=4)
        document['args'].update(warmup=12, iterations=100)
        for primitive in document['samples'][0]['primitives'].values():
            timing = primitive['timing']
            timing['samples_us'] *= 2
            for rank in timing['rank_samples_us']:
                rank *= 2
            timing['p95_us'] = statistics.quantiles(timing['samples_us'], n=100, method='inclusive')[94]
        self.assertEqual(len(model.calibration_rows(document)), 1)
        for key in ('max_abs', 'relative_rmse'):
            for value in (-1., math.nan, math.inf, True):
                broken = copy.deepcopy(document)
                broken['samples'][0]['primitives']['compute_bare_subgrid']['correctness'][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    model.calibration_rows(broken)

    def test_partial_short_or_wrong_scope_data_are_rejected(self):
        for change in ('partial', 'sample_count', 'primitive_count', 'warmup', 'iterations',
                       'eager', 'rank_stat', 'missing_finalize_exclusion', 'phase', 'preparation',
                       'sources', 'clocks', 'devices', 'duplicate', 'missing_candidate'):
            document = calibration(requests=(4, 12))
            if change == 'partial': document['complete'] = False
            elif change == 'sample_count': document['samples'].pop()
            elif change == 'primitive_count': document['expected_primitive_measurements'] -= 1
            elif change == 'warmup': document['args']['warmup'] = 2
            elif change == 'iterations': document['args']['iterations'] = 7
            elif change == 'eager': document['args']['launch'] = 'eager'
            elif change == 'rank_stat': document['timing']['sample_statistic'] = 'max_of_rank_medians'
            elif change == 'missing_finalize_exclusion': document['timing']['excluded'].remove('finalize')
            elif change == 'phase': document['phase'] = 'total'
            elif change == 'preparation': document['preparation'] = 'different'
            elif change == 'sources': document['sources'] = {}
            elif change == 'clocks': document['gpu_clock_snapshot'] = ''
            elif change == 'devices': document['devices'][0]['sm_count'] = 120
            elif change == 'duplicate': document['samples'][1] = copy.deepcopy(document['samples'][0])
            else:
                document['samples'].pop()
                document['expected_samples'] = 1
                document['expected_primitive_measurements'] = len(model.PRIMITIVES)
            with self.subTest(change=change), self.assertRaises(ValueError):
                model.calibration_rows(document)

    def test_raw_rankmax_percentiles_throughput_and_correctness_are_checked(self):
        for change in ('rankmax', 'rank_count', 'nan', 'negative', 'bool', 'p50_us',
                       'p95_us', 'mean_us', 'tflops_per_gpu', 'flops_per_gpu',
                       'payload', 'finite', 'rmse', 'exact_copy', 'epoch', 'ready_size'):
            document = calibration()
            sample = document['samples'][0]
            primitive = sample['primitives']['compute_bare_subgrid']
            timing = primitive['timing']
            if change == 'rankmax': timing['rank_samples_us'][0][0] *= 2
            elif change == 'rank_count': timing['rank_samples_us'].pop()
            elif change == 'nan': timing['samples_us'][0] = math.nan
            elif change == 'negative': timing['rank_samples_us'][0][0] = -1.
            elif change == 'bool': timing['rank_samples_us'][0][0] = True
            elif change in timing: timing[change] *= 2
            elif change == 'payload': sample['work']['remote_payload_bytes_per_gpu'] += 1
            elif change == 'finite': primitive['correctness']['all_ranks_finite'] = False
            elif change == 'rmse': primitive['correctness']['relative_rmse'] = .006
            elif change == 'exact_copy': sample['primitives']['copy']['correctness']['max_abs'] = 1e-10
            elif change == 'epoch': primitive['ready_epoch'] = 0x01010101
            elif change == 'ready_size': primitive['ready_prepublication_bytes'] *= 4
            with self.subTest(change=change), self.assertRaises(ValueError):
                model.calibration_rows(document)

    def test_bf16_tolerance_and_native_register_counts_are_not_fabricated(self):
        document = calibration()
        primitives = document['samples'][0]['primitives']
        primitives['compute_bare_subgrid']['correctness'].update(max_abs=2., relative_rmse=.005)
        for name in ('compute_ready_preloaded_subgrid', 'compute_ready_preloaded_fullgrid'):
            primitives[name]['resources']['primitive_registers_per_thread'] = 172
        row = model.calibration_rows(document)[0]
        self.assertEqual(row['primitives']['compute_ready_preloaded_subgrid']['features']
                         ['native_resources']['primitive_registers_per_thread'], 172)
        primitives['compute_bare_fullgrid']['resources']['primitive_registers_per_thread'] = 160
        with self.assertRaisesRegex(ValueError, 'inconsistent resource'):
            model.calibration_rows(document)

    def test_resource_changes_across_points_and_adapter_smem_mismatch_fail(self):
        for change in ('registers', 'adapter_smem'):
            document = calibration(requests=(4, 12))
            if change == 'registers':
                document['samples'][1]['primitives']['copy']['resources']['primitive_registers_per_thread'] = 39
            else:
                document['samples'][0]['primitives']['compute_ready_preloaded_subgrid']['resources']\
                    ['primitive_dynamic_smem_bytes'] = 214144
            with self.subTest(change=change), self.assertRaises(ValueError):
                model.calibration_rows(document)

    def test_copy_launch_variants_remain_separate_and_legacy_raw_is_accepted(self):
        document = calibration()
        row = model.calibration_rows(document)[0]
        original = row['primitives']['copy']['features']
        reserved = row['primitives']['copy_fused_reservation']['features']
        self.assertEqual(original['copy_launch_variant'], 'standalone_cluster1')
        self.assertEqual(reserved['copy_launch_variant'], 'fused_cluster2_reservation')
        self.assertEqual(original['copy_tasks'], reserved['copy_tasks'])
        self.assertEqual(reserved['native_resources']['primitive_dynamic_smem_bytes'], 214016)
        self.assertEqual(reserved['native_resources']['primitive_launch_cluster_m'], 2)
        for key, value in (('primitive_dynamic_smem_bytes', 196704), ('primitive_launch_cluster_m', 1)):
            broken = copy.deepcopy(document)
            broken['samples'][0]['primitives']['copy_fused_reservation']['resources'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                model.calibration_rows(broken)
        legacy = calibration(selected=model.LEGACY_PRIMITIVES)
        legacy['primitive_ids'] = dict(model.LEGACY_PRIMITIVES)
        self.assertEqual(len(model.calibration_rows(legacy)[0]['primitives']), 5)
        document['primitive_ids'] = dict(model.LEGACY_PRIMITIVES)
        with self.assertRaisesRegex(ValueError, 'lists'):
            model.calibration_rows(document)

    def test_fit_requires_measured_primitive_and_three_actual_geometries(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            model.fit_calibration(calibration(), 'copy_combined')
        with self.assertRaisesRegex(ValueError, 'not measured'):
            model.fit_calibration(calibration(selected=('copy',)), 'copy_fused_reservation')
        # More CTA points, labels or sequence lengths do not create new (N,K).
        document = calibration(requests=(4, 12, 32))
        for row in document['samples']:
            row['case']['model'] = str(row['config']['comm_ctas'])
        extra = calibration(requests=(4, 12, 32), m=32768)
        document['samples'].extend(extra['samples'])
        document['expected_samples'] *= 2
        document['expected_primitive_measurements'] *= 2
        with self.assertRaisesRegex(ValueError, 'three distinct actual'):
            model.fit_calibration(document, 'copy')

    def test_fit_does_not_bypass_raw_scope_or_variant_validation(self):
        document = fitting_calibration()
        document['complete'] = False
        with self.assertRaisesRegex(ValueError, 'complete'):
            model.fit_calibration(document, 'copy')
        document['complete'] = True
        sample = document['samples'][0]
        sample['primitives']['copy'] = copy.deepcopy(sample['primitives']['copy_fused_reservation'])
        with self.assertRaises(ValueError):
            model.fit_calibration(document, 'copy')


class QkvServiceFitTest(unittest.TestCase):
    def setUp(self):
        try:
            import numpy
            import scipy.optimize
        except ImportError:
            self.skipTest('Optional NumPy/SciPy fitting packages are not installed')

    def test_each_primitive_recovers_coefficients_and_heldout_errors(self):
        document = fitting_calibration()
        original = copy.deepcopy(document)
        for name in model.PRIMITIVES:
            with self.subTest(primitive=name):
                result = model.fit_calibration(document, name)
                self.assertEqual(result['model_status'], 'diagnostic_only')
                self.assertFalse(result['production_policy_written'])
                self.assertFalse(result['milestone_evidence'])
                self.assertEqual(result['primitive'], name)
                self.assertEqual(result['training']['points'], 9)
                self.assertEqual(result['training']['geometries'], 3)
                self.assertLess(result['training']['errors']['max_absolute_percentage_error'], 1e-5)
                logo = result['leave_one_geometry_out']
                self.assertEqual(logo['grouping'], 'actual_N_K_not_model_name')
                self.assertEqual([fold['held_out_nk'] for fold in logo['folds']],
                                 [[2048, 4096], [3072, 8192], [5120, 5120]])
                self.assertLess(logo['errors']['max_absolute_percentage_error'], 1e-4)
                for fitted in [result] + logo['folds']:
                    for actual, expected in zip(fitted['coefficients'].values(), FIT_COEFFICIENTS[name]):
                        self.assertTrue(math.isclose(actual, expected, rel_tol=1e-5))
                self.assertEqual(result['fitting']['fixed_intercept_us'], 5.)
                self.assertFalse(result['fitting']['intercept_is_measured_launch'])
                self.assertFalse(result['fitting']['fused_winners_used'])
                json.dumps(result, allow_nan=False)
        self.assertEqual(document, original)

    def test_other_copy_variant_and_model_labels_do_not_enter_the_fit(self):
        document = fitting_calibration()
        original = model.fit_calibration(document, 'copy_fused_reservation')
        for sample in document['samples']:
            sample['case'].update(id='renamed', model='never_a_feature')
            timing = sample['primitives']['copy']['timing']
            for key in ('p50_us', 'p95_us', 'mean_us'):
                timing[key] *= 10
            timing['samples_us'] = [value * 10 for value in timing['samples_us']]
            timing['rank_samples_us'] = [[value * 10 for value in rank] for rank in timing['rank_samples_us']]
        self.assertEqual(model.fit_calibration(document, 'copy_fused_reservation'), original)

    def test_log_objective_initial_values_and_solver_options_are_frozen(self):
        import scipy.optimize
        document = fitting_calibration()
        solver = scipy.optimize.least_squares
        for name, initial in [('compute_bare_subgrid', [180000., 4.]), ('copy', [3.4, 30.])]:
            calls = []

            def record(function, start, **kwargs):
                calls.append((list(start), kwargs, list(function(start))))
                return solver(function, start, **kwargs)

            with mock.patch('scipy.optimize.least_squares', side_effect=record):
                model.fit_calibration(document, name)
            self.assertEqual(len(calls), 4)  # Full fit plus all three LOGO fits.
            for start, options, _ in calls:
                self.assertEqual(start, initial)
                self.assertEqual(options, dict(bounds=(0, math.inf), x_scale='jac'))
            first = model.calibration_rows(document)[0]['primitives'][name]
            feature = first['features']
            prediction = (5 + math.hypot(initial[0] * feature['remote_payload_bytes_per_gpu'] / 2**20,
                                         initial[1] * feature['copy_slot_task_depth']) if name == 'copy' else
                          5 + feature['compute_cluster_waves'] * (
                              initial[0] * feature['padded_flops_per_output_tile'] / 1e12 + initial[1]))
            self.assertAlmostEqual(calls[0][2][0], math.log(prediction / first['timing']['p50_us']))

    def test_nonconverged_solver_is_rejected(self):
        failure = mock.Mock(success=False, message='synthetic nonconvergence')
        with mock.patch('scipy.optimize.least_squares', return_value=failure):
            with self.assertRaisesRegex(ValueError, 'did not converge'):
                model.fit_calibration(fitting_calibration(), 'copy')


if __name__ == '__main__':
    unittest.main()
