"""CPU negative tests for the sample/finite gates used by the full audit."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import backward_report as report
from backward_matrix import ROOT, full_matrix
from backward_report import correctness_errors, timing_errors


class TimingAuditTest(unittest.TestCase):
    def sample(self):
        return dict(samples_us=[2.]*50, rank_samples_us=[[1.]*50, [2.]*50],
                    p50_us=2., p95_us=2., mean_us=2.)

    def test_valid(self):
        self.assertEqual(timing_errors(self.sample(), 50, 2), [])

    def test_bad_rank_sample_even_if_hidden_by_max(self):
        sample = self.sample()
        sample['rank_samples_us'][0][0] = float('nan')
        self.assertTrue(timing_errors(sample, 50, 2))

    def test_missing_rank_or_sample(self):
        sample = self.sample()
        sample['rank_samples_us'][0].pop()
        self.assertTrue(timing_errors(sample, 50, 2))

    def test_invented_percentile(self):
        sample = self.sample()
        sample['p95_us'] = 3.
        self.assertTrue(timing_errors(sample, 50, 2))

    def test_rank_mean_is_not_rank_max(self):
        sample = self.sample()
        sample.update(samples_us=[1.5]*50, p50_us=1.5, p95_us=1.5, mean_us=1.5)
        self.assertTrue(timing_errors(sample, 50, 2))

    def test_nested_correctness(self):
        valid = dict(full=dict(wgrad=dict(max_abs=.0001, relative_rmse=.00001)))
        self.assertFalse(correctness_errors(valid))
        for field in ('max_abs', 'relative_rmse'):
            corrupt = copy.deepcopy(valid)
            corrupt['full']['wgrad'][field] = float('nan')
            self.assertTrue(correctness_errors(corrupt))


class CompleteAuditTest(unittest.TestCase):
    """Synthetic fixture, never a performance result: audit one real geometry."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='mxfp8-audit-test-')
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.case = full_matrix()[0]
        self.cid = self.case['id']
        self.keys = {(self.cid, b, l, m, 'fp32') for b in ('teub','cublaslt_nccl')
                     for l in ('eager','graph') for m in ('immediate','deferred')}
        self.patches = [patch.object(report, 'full_matrix', return_value=[self.case]),
                        patch.object(report, 'expected_keys', return_value=self.keys)]
        for p in self.patches:
            p.start(); self.addCleanup(p.stop)
        sources = ['benchmarks/mxfp8_weight/backward_bench.py',
                   'benchmarks/mxfp8_weight/backward_runtime.py',
                   'benchmarks/mxfp8_weight/backward_gemm.py',
                   'benchmarks/mxfp8_weight/backward_gemm.cu',
                   'benchmarks/mxfp8_weight/backward_matrix.py',
                   'benchmarks/backward/backward_shape_bench.py',
                   'build-mxfp8-bench/libmxfp8_backward.so']
        # This is an auditor fixture, not GPU evidence. A clean checkout must
        # run CPU tests without building a CUDA shared library first.
        source_root = self.directory / 'synthetic_sources'
        for source in sources:
            target = source_root / source
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b'synthetic audit fixture\n')
        source_patch = patch.object(report, 'ROOT', source_root)
        source_patch.start()
        self.addCleanup(source_patch.stop)
        gate = dict(max_abs=0., relative_rmse=0.)
        def timing(count, value=2.):
            return dict(samples_us=[value]*count, rank_samples_us=[[value]*count for _ in range(4)],
                        p50_us=value, p95_us=value, mean_us=value)
        def raw(phase, profile, backend='cublaslt_nccl'):
            return dict(complete=True, args=dict(grad_dtypes='fp32', validation=False,
                        sweep_warmup=3, sweep_iters=12, phase=phase, warmup=10, iterations=50,
                        candidates=32, backends=backend),
                        schema='mxfp8-backward-v1', semantic='offline_mxfp8_original_weight_axis_runtime_dq_bf16_gemm',
                        launch_policy={self.cid:['eager','graph']},
                        sources={s:report.digest(source_root/s) for s in sources},
                        environment=dict(CUDA_VISIBLE_DEVICES=self.case['visible_devices'],
                                         NCCL_IB_DISABLE='1', NCCL_GRAPH_REGISTER='1', NCCL_LOCAL_REGISTER='1',
                                         **report.PROFILES[profile]),
                        devices=[dict(cc=[9,0]) for _ in range(4)], cases=[])
        for profile in report.PROFILES:
            r = raw('sweep', profile)
            r['cases'] = [dict(case=self.case, records=[], search=[dict(backend='cublaslt_nccl',
                launch=l, sms=0, data=timing(12, 1. if profile=='auto' else 2.),
                correctness=dict(route=gate,dgrad=gate)) for l in ('eager','graph')])]
            self.write(f'cublaslt_nccl_sweep_{profile}_cp4.json', r)
        self.write('nccl_policy_cp4.json', [dict(id=self.cid, launch=l, profile='auto') for l in ('eager','graph')])
        for backend in ('cublaslt_nccl', 'teub'):
            r = raw('formal', 'auto', backend)
            rows = []
            for launch in ('eager','graph'):
                for mode in ('immediate','deferred'):
                    beta = int(mode == 'deferred')
                    rows.append(dict(backend=backend, launch=launch, weight_mode=mode, grad_dtype='fp32',
                        beta=beta, sms=4 if backend=='teub' else 0,
                        data=timing(50), weight=timing(50), total=timing(50), isolated_sum_samples_us=[4.]*50,
                        correctness=dict(data=dict(route=gate,dgrad=gate), weight=gate,
                            full=dict(route=gate,dgrad=gate,wgrad=gate), nonzero_beta1_twice=[gate,gate]),
                        b_plan=dict(mnk=self.case['b_mnk'],output_dtype='torch.bfloat16',ta=False,tb=False),
                        w_plan=dict(mnk=self.case['w_mnk'],output_dtype='torch.float32',ta=True,tb=False,tune_beta=beta)))
            search = [dict(backend='teub',launch=l,sms=s,data=timing(12,float(s)),
                            correctness=dict(route=gate,dgrad=gate)) for l in ('eager','graph') for s in (4,8,16)] if backend=='teub' else []
            r['cases'] = [dict(case=self.case, records=rows, search=search)]
            self.write(f'{backend}_formal_auto_cp4.json', r)

    def write(self, name, value):
        (self.directory/name).write_text(json.dumps(value))

    def mutate(self, name, fn):
        path = self.directory/name
        value = json.loads(path.read_text())
        fn(value)
        self.write(name, value)

    def test_positive_complete_fixture(self):
        result, rows = report.audit(self.directory)
        self.assertTrue(result['complete'], result['errors'])
        self.assertEqual(len(rows), 8)

    def test_missing_row(self):
        self.mutate('teub_formal_auto_cp4.json', lambda r:r['cases'][0]['records'].pop())
        result, _ = report.audit(self.directory)
        self.assertFalse(result['complete'])
        self.assertEqual(len(result['missing']), 1)

    def test_changed_source(self):
        self.mutate('teub_formal_auto_cp4.json', lambda r:r['sources'].update({'benchmarks/mxfp8_weight/backward_bench.py':'wrong'}))
        self.assertTrue(report.audit(self.directory)[0]['errors'])

    def test_wrong_dtype(self):
        self.mutate('teub_formal_auto_cp4.json', lambda r:r['cases'][0]['records'][0]['w_plan'].update(output_dtype='torch.bfloat16'))
        self.assertTrue(report.audit(self.directory)[0]['errors'])

    def test_not_teub_winner(self):
        self.mutate('teub_formal_auto_cp4.json', lambda r:r['cases'][0]['records'][0].update(sms=16))
        self.assertTrue(report.audit(self.directory)[0]['errors'])

    def test_not_nccl_winner(self):
        self.mutate('nccl_policy_cp4.json', lambda r:r[0].update(profile='ch24_chunk128_ll64'))
        self.assertTrue(report.audit(self.directory)[0]['errors'])

    def test_wrong_matrix_geometry(self):
        self.mutate('teub_formal_auto_cp4.json', lambda r:r['cases'][0]['case'].update(global_seq=999))
        self.assertTrue(report.audit(self.directory)[0]['errors'])

    def test_wrong_actual_formal_nccl_environment(self):
        self.mutate('cublaslt_nccl_formal_auto_cp4.json', lambda r:r['environment'].update(NCCL_MIN_P2P_NCHANNELS='99'))
        self.assertTrue(report.audit(self.directory)[0]['errors'])

    def test_sweep_cannot_impersonate_formal(self):
        self.mutate('cublaslt_nccl_formal_auto_cp4.json', lambda r:r['args'].update(phase='sweep'))
        self.assertTrue(report.audit(self.directory)[0]['errors'])

    def test_duplicate_search_candidate(self):
        self.mutate('teub_formal_auto_cp4.json', lambda r:r['cases'][0]['search'].append(r['cases'][0]['search'][0]))
        self.assertTrue(report.audit(self.directory)[0]['errors'])


if __name__ == '__main__':
    unittest.main()
