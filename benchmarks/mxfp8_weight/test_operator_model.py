"""CPU-only checks for physical features, risk gating, and reproducible fits."""

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import operator_model as model


def fixture(cp=4, c=16, m=32768, n=2048, k=2048):
    case = dict(m=m, n=n, k=k, cp=cp, global_seq=m * cp, layout='causal_paired',
                direction='a2a_gemm', model='ignored_business_label', batch=1)
    config = dict(comm_ctas=c, sm_count=132, tile_m=128, tile_n=256, tile_k=64,
                  cluster_m=2, route_layout='causal_paired', gemm_input='bf16',
                  weight_block_size=32, copy_schedule='fused_frontier_window',
                  compute_scheduler='stock_persistent_same_tile_and_SM_budget_no_ready_waits')
    return case, config


def fitted_model(cp=4):
    return dict(world_size=cp, sm_count=132, launch_prior_us=5., minimum_gain=.10,
                domain=dict(tile_families=[[128, 256, 64, 2]]),
                coefficients=dict(a=162853.3940924, e=0, b=3.20761773318, t=4.83655949484))


def calibration():
    coefficients = dict(a=180000., e=4., b=3.4, t=6.)
    samples = []
    for n, k in ((2048, 2048), (4096, 4096), (3072, 6144)):
        for c in (4, 8, 16, 24):
            case, config = fixture(c=c, n=n, k=k)
            features = model.features(case, config)
            predicted = model.service_times(features, coefficients)
            primitives = {}
            for name, key in zip(model.PRIMITIVES, ('compute_us', 'copy_us')):
                value = predicted[key]
                primitives[name] = dict(correctness=dict(all_ranks_finite=True, max_abs=0., relative_rmse=0.),
                                        timing=dict(p50_us=value, samples_us=[value] * 50,
                                                    rank_samples_us=[[value] * 50 for _ in range(case['cp'])]))
            samples.append(dict(case=case, config=config, primitives=primitives,
                                work=dict(remote_payload_bytes_per_gpu=features['remote_payload_bytes'])))
    return dict(schema='mxfp8-oproj-independent-primitives-v1', complete=True,
                expected_samples=len(samples), samples=samples, devices=[], sources={'cpu_fixture': 'source-hash'},
                cuda_visible_devices='fixture-only', policy_environment={},
                args=dict(cp=4, launch='graph', warmup=10, iterations=50),
                timing=dict(sample_statistic='sample_wise_max_across_ranks'))


class OperatorModelTest(unittest.TestCase):
    def test_import_and_help_need_no_numerical_or_cuda_packages(self):
        code = ('import sys; import operator_model; '
                'assert not ({"torch", "numpy", "scipy", "cuda"} & set(sys.modules))')
        subprocess.run([sys.executable, '-S', '-c', code], cwd=Path(model.__file__).parent, check=True)
        result = subprocess.run([sys.executable, '-S', model.__file__, '--help'],
                                capture_output=True, text=True, check=True)
        self.assertIn('--calibration', result.stdout)

    def test_bulk_task_count_includes_local_peer_and_partial_chunks(self):
        case, config = fixture()
        feature = model.features(case, config, 16)
        self.assertEqual((feature['row_bytes'], feature['comm_rows'], feature['chunks_per_ready_tile']),
                         (1024, 48, 3))
        self.assertEqual(feature['copy_tasks'], 256 * 4 * 3)
        self.assertEqual(feature['task_waves'], 48)
        self.assertEqual(feature['remote_payload_bytes'], 2 * 32768 * 2048 * 3 // 4)
        self.assertEqual((feature['waves'], feature['m_window'], feature['window_batches']), (18, 15, 18))

    def test_cluster_wave_padding_is_not_independent_cta_rounding(self):
        case, config = fixture(c=16, m=640, n=16384)
        case['layout'] = config['route_layout'] = 'contiguous'
        # Five M tiles require three cluster tiles, not 2.5 divisible workers.
        feature = model.features(case, config)
        self.assertEqual(feature['waves'], 4)
        self.assertEqual(model.ceil_div(5 * 64, 116), 3)

    def test_ready_window_supports_multiple_compute_tiles_and_clamp(self):
        case, config = fixture(m=256)
        config['tile_m'] = 64
        feature = model.features(case, config)
        self.assertEqual(feature['m_window'], 2)
        self.assertEqual(feature['window_batches'], 1)
        config['comm_m_window'] = 1
        with self.assertRaisesRegex(ValueError, 'm_window'):
            model.features(case, config)

    def test_bad_bulk_inputs_and_stale_configs_are_rejected(self):
        case, config = fixture()
        for updates in (dict(m=129), dict(m=128), dict(k=2050), dict(k=131072),
                        dict(layout='unknown'), dict(batch=2), dict(source_row_begin=128),
                        dict(seq_local=65536), dict(direction='gemm_a2a')):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                model.features(dict(case, **updates), config)
        for updates in (dict(comm_ctas=3), dict(comm_ctas=132), dict(tile_m=96),
                        dict(comm_m_window=2), dict(route_layout='contiguous'),
                        dict(gemm_input='fp8'), dict(copy_schedule='different')):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                model.features(case, dict(config, **updates))
        with self.assertRaisesRegex(ValueError, 'Resolve'):
            model.features(case, config, 24)

    def test_each_peer_k_must_align_to_actual_native_tile_k(self):
        case, config = fixture(k=2112)
        # Divisible by CP and 16-byte copy rows, but peer K=528 is not tile64-aligned.
        with self.assertRaisesRegex(ValueError, 'per-peer K'):
            model.features(case, config)
        case['k'] = 2304
        self.assertEqual(model.features(case, config)['row_bytes'], 1152)
        # Peer K=576 aligns to tile64 but not tile128: use the resolved Tk.
        with self.assertRaisesRegex(ValueError, 'per-peer K'):
            model.features(case, dict(config, tile_k=128))

    def test_predict_uses_explicit_calibration_and_ignores_model_name(self):
        case, config = fixture()
        result = model.predict(case, config, fitted_model())
        renamed = dict(case, model='not_the_training_model', id='unseen')
        self.assertEqual(result, model.predict(renamed, config, fitted_model()))
        for wrong in (dict(case, cp=8),):
            with self.assertRaises(ValueError):
                model.predict(wrong, config, fitted_model())
        with self.assertRaises(ValueError):
            model.predict(case, dict(config, tile_n=320), fitted_model())
        with self.assertRaises(ValueError):
            model.predict(case, dict(config, sm_count=130), fitted_model())

    def test_services_are_finite_and_score_excludes_dq(self):
        case, config = fixture()
        feature = model.features(case, config)
        result = model.service_times(feature, fitted_model()['coefficients'])
        a, b = result['compute_us'], result['copy_us']
        self.assertAlmostEqual(result['candidate_score_us'], max(a, b) + min(a, b) / 18)
        for value in (-1, math.inf, math.nan):
            with self.subTest(value=value), self.assertRaises(ValueError):
                model.service_times(feature, dict(fitted_model()['coefficients'], a=value))

    def test_risk_gate_is_ten_percent_latency_reduction_and_ties_keep_auto(self):
        case, baseline = fixture(c=4)
        candidate = dict(baseline, comm_ctas=16)
        for score, switched in ((91., False), (90., True), (89., True), (100., False)):
            def prediction(case, config, calibration):
                return {'candidate_score_us': 100. if config['comm_ctas'] == 4 else score}
            with self.subTest(score=score), patch.object(model, 'predict', side_effect=prediction):
                result = model.guarded_select(case, baseline, [candidate], fitted_model())
                self.assertEqual(result['switched'], switched)
                self.assertEqual(result['selected_config']['comm_ctas'], 16 if switched else 4)
        self.assertEqual(model.guarded_select(case, baseline, [], fitted_model())['selected_config'], baseline)
        for threshold in (-.1, 1, math.nan):
            with self.assertRaises(ValueError):
                model.guarded_select(case, baseline, [], fitted_model(), threshold)
        with self.assertRaises(ValueError):
            model.guarded_select(case, baseline, [dict(candidate, comm_ctas=20)], fitted_model())

    def test_calibration_validation_rejects_short_failed_or_corrupt_data(self):
        document = calibration()
        self.assertEqual(len(model.calibration_rows(document)), 12)
        for change in ('incomplete', 'short', 'failed', 'payload', 'rankmax', 'median', 'duplicate', 'scheduler'):
            broken = copy.deepcopy(document)
            first = broken['samples'][0]
            primitive = first['primitives']['copy']
            if change == 'incomplete':
                broken['complete'] = False
            elif change == 'short':
                broken['args']['iterations'] = 7
            elif change == 'failed':
                primitive['correctness']['max_abs'] = 1
            elif change == 'payload':
                first['work']['remote_payload_bytes_per_gpu'] += 1
            elif change == 'rankmax':
                primitive['timing']['rank_samples_us'][0][0] *= 2
            elif change == 'median':
                primitive['timing']['p50_us'] *= 2
            elif change == 'duplicate':
                broken['samples'][1] = first
            else:
                first['config']['compute_scheduler'] = 'fullgrid'
            with self.subTest(change=change), self.assertRaises(ValueError):
                model.calibration_rows(broken)

    def test_compute_accepts_bf16_tolerance_but_copy_stays_exact(self):
        document = calibration()
        correct = document['samples'][0]['primitives']['compute_subgrid']['correctness']
        correct.update(max_abs=1., relative_rmse=0.005)
        self.assertEqual(len(model.calibration_rows(document)), 12)
        correct['relative_rmse'] = 0.005000001
        with self.assertRaisesRegex(ValueError, 'tolerance'):
            model.calibration_rows(document)
        correct['relative_rmse'] = 0.004
        document['samples'][0]['primitives']['copy']['correctness']['max_abs'] = 1e-10
        with self.assertRaisesRegex(ValueError, 'exact copy'):
            model.calibration_rows(document)

    def test_all_correctness_errors_are_finite_nonnegative_and_all_rank_gated(self):
        for primitive in model.PRIMITIVES:
            for key in ('max_abs', 'relative_rmse'):
                for value in (-1., math.nan, math.inf):
                    document = calibration()
                    document['samples'][0]['primitives'][primitive]['correctness'][key] = value
                    with self.subTest(primitive=primitive, key=key, value=value), self.assertRaises(ValueError):
                        model.calibration_rows(document)
            document = calibration()
            document['samples'][0]['primitives'][primitive]['correctness']['all_ranks_finite'] = False
            with self.subTest(primitive=primitive), self.assertRaises(ValueError):
                model.calibration_rows(document)

    @unittest.skipUnless(importlib.util.find_spec('scipy'), 'SciPy is optional outside fitting')
    def test_fit_logo_and_single_output_preserve_provenance(self):
        with tempfile.TemporaryDirectory(prefix='mxfp8-model-cpu-') as temporary:
            root = Path(temporary)
            source, output = root / 'calibration.json', root / 'model.json'
            source.write_text(json.dumps(calibration()))
            model.main(['--calibration', str(source), '--output', str(output)])
            document = json.loads(output.read_text())
            frozen = document['models'][0]
            self.assertEqual(frozen['training']['points'], 12)
            self.assertEqual(frozen['training']['geometries'], 3)
            self.assertEqual(len(frozen['leave_one_geometry_out']['folds']), 3)
            self.assertEqual(frozen['calibration_sha256'], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(frozen['source_hashes'], {'cpu_fixture': 'source-hash'})
            self.assertFalse(document['fitting']['fused_winners_used'])
            self.assertFalse(document['fitting']['intercept_is_measured_launch'])
            self.assertEqual(document['fitting']['fixed_intercept_us'], 5.)
            for key, expected in dict(a=180000., e=4., b=3.4, t=6.).items():
                self.assertAlmostEqual(frozen['coefficients'][key], expected)
            for fold in frozen['leave_one_geometry_out']['folds']:
                self.assertEqual((fold['train_points'], fold['test_points']), (8, 4))
                self.assertLess(fold['errors']['copy']['mape_percent'], 1e-6)
            with self.assertRaises(FileExistsError):
                model.main(['--calibration', str(source), '--output', str(output)])
            with self.assertRaises(ValueError):
                model.main(['--calibration', str(source), '--output', str(source), '--overwrite'])
            self.assertEqual({p.name for p in root.iterdir()}, {'calibration.json', 'model.json'})


if __name__ == '__main__':
    unittest.main()
