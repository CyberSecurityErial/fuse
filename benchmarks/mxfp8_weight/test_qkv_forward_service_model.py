"""CPU contracts for fixed-tile QKV-F role fitting and explicit selection."""

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
from unittest import mock

import qkv_forward_service_model as model


HAS_SCIPY = importlib.util.find_spec('scipy') is not None and importlib.util.find_spec('numpy') is not None
COEFFICIENTS = dict(compute_gflop_sm_us=170., route_slot_task_us=6.)


def calibration():
    """Synthetic role observations, not GPU performance evidence."""
    requests, lengths = [4, 8, 12, 16, 24], [131072, 524288]
    document = dict(schema=model.CALIBRATION_SCHEMA, complete=True, profiling_only=True,
                    reported_duration_unit='us',
                    clock_origin='independent per rank; cross-rank absolute subtraction is invalid',
                    selected_case_ids=[], samples=[],
                    args=dict(cp=4, operators='qkv_forward', launch='graph', weight_mode='immediate',
                              oproj_comm_model=None, aggregate_only=True, model='cpu_0,cpu_1,cpu_2',
                              comm_ctas=requests, seqs=','.join(map(str, lengths)), max_cases=30,
                              library='/not_used/build-profile/library.so'),
                    metadata=dict(kind='diagnostic_not_formal_performance', cc=[9, 0],
                                  native_device_name='CPU fixture; no measured GPU',
                                  cuda_visible_devices='0,1,2,3', launch='graph', weight_mode='immediate',
                                  requested_comm_ctas=requests, library_sha256='b' * 64,
                                  sources={name: 'a' * 64 for name in model.REQUIRED_SOURCES},
                                  policy_environment=dict(FUSE_QKV_GEMM_POLICY='m128n256'),
                                  fixed_tile_single_factor_experiment=False))
    document['metadata']['sources']['build-profile/library.so'] = 'b' * 64
    for index, (q, k) in enumerate(((16, 2048), (48, 3072), (128, 16384))):
        for length in lengths:
            m, n = length // 4, (q + 16) * 128
            case = dict(id=f'gemm_a2a/cpu_{index}/s{length}/cp4', model=f'cpu_{index}',
                        direction='gemm_a2a', cp=4, batch=1, layout='rank_major', global_seq=length,
                        m=m, n=n, k=k, hidden=k, q_heads=q, kv_heads=8, head_dim=128,
                        visible_devices='0,1,2,3', registry='benchmarks/QKVproj+a2a/qkv_shape_bench.py')
            document['selected_case_ids'].append(case['id'])
            for c in requests:
                ready = m // 128 * (n // 256) * 32
                config = dict(zip(model.TILE_KEYS, model.TILE))
                config.update(sm_count=132, comm_ctas=c, requested_comm_ctas=c, world_size=4,
                              raster='n', swizzle=1, route_layout='rank_major', epoch=1,
                              gemm_input='bf16', gemm_accumulator='fp32', weight_block_size=32,
                              payload='e4m3', scale='e8m0', weight_axis='original_forward_K',
                              alpha=1, beta=0, weight_mnk=None, forward_or_data_mnk=[m, n, k],
                              weight_workspace_bytes=2*n*k, ready_elements=ready,
                              peer_arena_bytes=2*m*n + 4*ready + 512,
                              gemm_policy_request='auto', policy_enum=0)
                times = model.service_times(model.features(case, config), COEFFICIENTS)
                compute, route = times['compute_us'], times['role_envelope_us']
                ranks = [dict(rank=rank, origin_ns=10**12 + rank * 10**9, config=copy.deepcopy(config),
                              observed_ctas=132, compute_ctas=132-c, route_ctas=c,
                              compute_role_us=compute * factor, route_role_us=route * factor,
                              overlap_us=min(compute, route) * factor,
                              diagnostic_boundary_us=999999.)
                         for rank, factor in enumerate((.97, .98, .99, 1.))]
                correct = {name: dict(max_abs=0., relative_rmse=0., all_ranks_finite=True)
                           for name in ('weight_dequant', 'output', 'local_gemm')}
                document['samples'].append(dict(case=copy.deepcopy(case), operator='qkv_forward',
                                                requested_comm_ctas=c, ranks=ranks, correctness=correct))
    document['expected_samples'] = len(document['samples'])
    return document


def explicit_model():
    return dict(world_size=4, sm_count=132, minimum_gain=0, coefficients=dict(COEFFICIENTS),
                domain=dict(min_m=32768, max_m=131072, min_n=4096, max_n=18432,
                            min_k=2048, max_k=16384, m_multiple=256, n_multiple=256,
                            k_multiple=64, head_dim=128, tile_families=[list(model.TILE)],
                            comm_ctas=list(model.COMM_CANDIDATES)))


class QkvForwardServiceModelTest(unittest.TestCase):
    def test_import_and_prediction_need_only_standard_library(self):
        code = ('import sys; import qkv_forward_service_model; '
                'assert not ({"torch", "triton", "numpy", "scipy", "cuda"} & set(sys.modules))')
        subprocess.run([sys.executable, '-S', '-c', code], cwd=Path(model.__file__).parent, check=True)

    def test_strict_raw_rows_are_rank_maxima_not_sum_or_median(self):
        document = calibration()
        before = copy.deepcopy(document)
        rows = model.calibration_rows(document)
        self.assertEqual(len(rows), 30)
        for row, sample in zip(rows, document['samples']):
            self.assertEqual(row['compute_us'], sample['ranks'][3]['compute_role_us'])
            self.assertEqual(row['route_envelope_us'], sample['ranks'][3]['route_role_us'])
        self.assertEqual(before, document)

    def test_pure_features_match_bf16_task_payload(self):
        sample = calibration()['samples'][0]
        feature = model.features(sample['case'], sample['ranks'][0]['config'])
        self.assertEqual(feature['route_tasks'] * 16384, 2 * feature['m'] * feature['n'])
        self.assertEqual(feature['work_gflop'], 2 * feature['m'] * feature['n'] * feature['k'] / 1e9)
        self.assertEqual(feature['compute_ctas'], 128)
        self.assertNotIn('waves', feature)

    def test_validation_rejects_partial_duplicate_scope_and_source_drift(self):
        mutations = [
            lambda d: d.update(complete=False),
            lambda d: d['samples'].pop(),
            lambda d: d['samples'].__setitem__(1, copy.deepcopy(d['samples'][0])),
            lambda d: d['args'].update(operators='qkv_backward'),
            lambda d: d['metadata'].update(launch='eager'),
            lambda d: d['metadata']['policy_environment'].update(FUSE_QKV_GEMM_POLICY=None),
            lambda d: d['metadata']['sources'].pop(model.REQUIRED_SOURCES[0]),
            lambda d: d['metadata'].update(library_sha256='c' * 64),
            lambda d: d['metadata']['sources'].update(invalid='not_a_hash'),
            lambda d: d['args'].update(model='cpu_0,cpu_1'),
            lambda d: d['samples'][0]['case'].update(id='gemm_a2a/wrong/s131072/cp4'),
            lambda d: d['samples'][0]['ranks'].pop(),
            lambda d: d['samples'][0]['ranks'][0].update(rank=2),
            lambda d: d['samples'][0]['ranks'][0]['config'].update(tile_n=320),
            lambda d: d['samples'][0]['ranks'][0]['config'].update(comm_ctas=8),
            lambda d: d['samples'][0]['ranks'][0]['config'].update(ready_elements=1),
            lambda d: d['samples'][0]['ranks'][0].update(compute_role_us=float('nan')),
            lambda d: d['samples'][0]['ranks'][0].update(overlap_us=1e12),
            lambda d: d['samples'][0]['correctness']['output'].update(all_ranks_finite=False),
            lambda d: d['samples'][0]['correctness']['local_gemm'].update(relative_rmse=.006),
            lambda d: d['samples'][0]['correctness']['weight_dequant'].update(max_abs=.001),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                document = calibration()
                mutate(document)
                with self.assertRaises(ValueError):
                    model.calibration_rows(document)

    def test_complete_forward_latency_is_not_consumed(self):
        document = calibration()
        expected = model.calibration_rows(document)
        for sample in document['samples']:
            sample['fused_winner'] = 'intentionally_not_an_input'
            for rank in sample['ranks']:
                for key in ('diagnostic_boundary_us', 'role_kernel_us', 'dq_us', 'finalize_us'):
                    rank[key] = 'not_a_fitting_input'
        self.assertEqual(model.calibration_rows(document), expected)

    def test_prediction_checks_physical_domain_not_model_label(self):
        sample = calibration()['samples'][0]
        case, config = sample['case'], sample['ranks'][0]['config']
        baseline = model.select(case, config, explicit_model())
        case['model'] = 'unseen_business_label'
        case['global_seq'] *= 2
        case['m'] *= 2
        self.assertEqual(model.select(case, config, explicit_model())['comm_ctas'], baseline['comm_ctas'])
        for key, value in (('cp', 8), ('m', 16384), ('k', 16385), ('n', 18560),
                           ('qkv_peer_interleaved', True)):
            bad = dict(case, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                model.predict(bad, config, explicit_model())
        for c in (0, 2, 3, 26, True):
            with self.subTest(c=c), self.assertRaises(ValueError):
                model.predict(case, config, explicit_model(), c)

    def test_no_gain_gate_and_exact_tie_retains_baseline(self):
        sample = calibration()['samples'][0]
        config = dict(sample['ranks'][0]['config'], comm_ctas=12)
        with mock.patch.object(model, 'predict', return_value={'role_envelope_us': 1.}):
            selected = model.select(sample['case'], config, explicit_model())
        self.assertEqual(selected['comm_ctas'], 12)
        self.assertFalse(selected['switched'])
        invalid = explicit_model()
        invalid['minimum_gain'] = .1
        with self.assertRaises(ValueError):
            model.validate_model(invalid)

    def test_errors_keep_compute_residual_and_censored_subset_separate(self):
        rows = model.calibration_rows(calibration())
        wrong = dict(COEFFICIENTS, compute_gflop_sm_us=100.)
        errors = model.error_summary(model.relative_errors(rows, wrong))
        self.assertGreater(errors['compute']['mape_percent'], 40)
        self.assertAlmostEqual(errors['conditional_route_envelope']['mape_percent'], 0.)
        self.assertLess(errors['conditional_route_communication_exposed']['points'], len(rows))
        self.assertGreater(errors['joint_role_envelope']['mape_percent'], 0)

    @unittest.skipUnless(HAS_SCIPY, 'Fitting optionally requires NumPy/SciPy')
    def test_fit_recovers_services_and_rejects_unidentifiable_route(self):
        rows = model.calibration_rows(calibration())
        fitted = model.fit_services(rows)
        for key, value in COEFFICIENTS.items():
            self.assertAlmostEqual(fitted[key], value, places=7)
        for row in rows:
            row['route_envelope_us'] = row['compute_us']
            row['route_uncovered_us'] = 0.
        with self.assertRaisesRegex(ValueError, 'unidentifiable'):
            model.fit_services(rows)

    @unittest.skipUnless(HAS_SCIPY, 'Fitting optionally requires NumPy/SciPy')
    def test_artifact_has_exact_native_contract_logo_and_length_drift(self):
        with tempfile.TemporaryDirectory(prefix='qkv-forward-model-test-') as directory:
            source, output = Path(directory) / 'raw.json', Path(directory) / 'model.json'
            raw = (json.dumps(calibration()) + '\n').encode()
            source.write_bytes(raw)
            model.main(['--calibration', str(source), '--output', str(output)])
            artifact = json.loads(output.read_text())
            fitted = model.load_model(output)
            self.assertEqual(artifact['schema'], model.SCHEMA)
            self.assertTrue(artifact['frozen'])
            self.assertFalse(artifact['production_policy_written'])
            self.assertFalse(artifact['milestone_evidence'])
            self.assertEqual(fitted['domain'], explicit_model()['domain'])
            self.assertEqual(fitted['calibration_sha256'], hashlib.sha256(raw).hexdigest())
            self.assertEqual(len(fitted['leave_one_geometry_out']['folds']), 3)
            self.assertEqual(fitted['leave_one_geometry_out']['errors']['compute']['points'], 30)
            self.assertEqual(len(fitted['length_drift']['cross_length_validation']), 2)
            self.assertIsNone(fitted['gpu_clock_snapshot'])
            self.assertFalse(fitted['fitting']['complete_forward_times_used'])
            self.assertFalse(fitted['fitting']['measured_winners_used'])
            prior = output.read_bytes()
            with self.assertRaises(FileExistsError):
                model.main(['--calibration', str(source), '--output', str(output)])
            self.assertEqual(output.read_bytes(), prior)
            with self.assertRaises(FileExistsError):
                model.main(['--calibration', str(source), '--output', str(source)])


if __name__ == '__main__':
    unittest.main()
