"""CPU contracts for batching isolation, cache geometry and resume."""
import argparse
import copy
import json
import os
from pathlib import Path
import tempfile
import sys
import types
import unittest
from unittest import mock

import sm103_batch as batch


class BatchContracts(unittest.TestCase):
    def job(self, backend='te_ub', launch='graph'):
        args = argparse.Namespace(directions='oproj', models='production_qwen_dense',
            seqs=(16384,), cps=(8,), devices='0,1,2,3,4,5,6,7', sm_count=148,
            stage='sweep', results=Path('/tmp/unused-batch-test'),
            library=Path('/tmp/unused.so'), python='python3', te_root=None)
        return batch.bench.make_job(args, next(batch.bench.cases(args)), backend, launch,
                                    batch.bench.initial_configs(backend)[0])

    def test_pack_and_sm_candidates_share_process(self):
        a = self.job()
        b = copy.deepcopy(a)
        b['config'].update(pack_block=256, comm_sm=12)
        self.assertEqual(len(batch.batches([a, b])), 1)

    def test_static_settings_split_processes(self):
        a = self.job('cublaslt_nccl')
        for field, value in (('env', a['env'] | {'NCCL_P2P_NVL_CHUNKSIZE': '524288'}),
                             ('launch', 'eager'), ('backend', 'te_ub'),
                             ('case', a['case'] | {'cp': 4}),
                             ('case', a['case'] | {'direction': 'qkv'}),
                             ('env', a['env'] | {'FUSE_SM103_OPROJ_LAYOUT': 'causal_dual_chunk_v1'}),
                             ('config', a['config'] | {'high_priority': True})):
            with self.subTest(field=field, value=value):
                self.assertEqual(len(batch.batches([a, a | {field: value}])), 2)

    def test_flags_preserve_original_worker_arguments(self):
        flags = batch.worker_flags(self.job())
        self.assertEqual(flags[:2], ['--global-seq', '16384'])
        self.assertIn('--check', flags)
        self.assertIn('--cuda-graph', flags)
        self.assertEqual(flags[flags.index('--iters')+1], '50')

    def test_batch_uses_shared_layout_adapter_before_gpu_initialization(self):
        with tempfile.TemporaryDirectory() as folder:
            job = self.job()
            job['env']['FUSE_SM103_OPROJ_LAYOUT'] = 'causal_dual_chunk_v1'
            manifest = Path(folder)/'batch.json'
            batch.atomic_json(manifest, [job])
            metadata = {'oproj_layout': 'causal_dual_chunk_v1'}
            adapter = mock.Mock(side_effect=RuntimeError('adapter reached'))
            preflight = types.SimpleNamespace(
                main=lambda: print(json.dumps(metadata)),
                WORKERS={('oproj', 'te_ub'): 'sm90/a2a+Oproj/te_userbuffers_oproj.py'},
                load_boundary=adapter)
            torch = types.ModuleType('torch')
            torch.distributed = types.ModuleType('torch.distributed')
            fake_modules = {'torch': torch, 'torch.distributed': torch.distributed,
                            'transformer_engine_torch': types.ModuleType('transformer_engine_torch'),
                            'measurement': types.ModuleType('measurement')}
            with (mock.patch.object(batch, 'module', return_value=preflight),
                  mock.patch.object(sys, 'argv', []),
                  mock.patch.dict(os.environ, job['env'], clear=True),
                  mock.patch.dict(sys.modules, fake_modules)):
                with self.assertRaisesRegex(RuntimeError, 'adapter reached'):
                    batch.worker(manifest)
                adapter.assert_called_once_with('oproj', 'te_ub', 'causal_dual_chunk_v1')
                metadata['oproj_layout'] = 'legacy'
                adapter.reset_mock()
                with self.assertRaisesRegex(ValueError, 'layout differs from manifest'):
                    batch.worker(manifest)
                adapter.assert_not_called()

    def test_completed_resume_does_no_gpu_work(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            job = self.job() | {'output': str(path/'result.json')}
            batch.atomic_json(job['output'], {})
            batch.atomic_json(path/'plan.json', {'jobs': [job]})
            with mock.patch.object(batch.bench, 'read_measurement'), mock.patch.object(batch.bench, 'observe_devices') as query:
                batch.execute(path/'plan.json', 1)
                query.assert_not_called()

    def test_alias_requires_identical_measurement_flags(self):
        a = self.job()
        b = copy.deepcopy(a)
        b['case']['model'] = 'another-label'
        b['output'] = '/tmp/alias.json'
        b['command'][b['command'].index('--json-out')+1] = b['output']
        self.assertEqual(batch.measurement_key(a), batch.measurement_key(b))
        b['command'][b['command'].index('--hidden')+1] = '4096'
        self.assertNotEqual(batch.measurement_key(a), batch.measurement_key(b))

    def test_import_rejects_fingerprint_change(self):
        with tempfile.TemporaryDirectory() as folder:
            target, source = Path(folder)/'target.json', Path(folder)/'source.json'
            batch.atomic_json(target, {'fingerprint': 'new', 'stage': 'sweep', 'jobs': []})
            batch.atomic_json(source, {'fingerprint': 'old', 'stage': 'sweep', 'jobs': []})
            with self.assertRaisesRegex(ValueError, 'differs'):
                batch.import_results(target, source)

    def test_statistical_retry_is_bounded_and_preserves_failed_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            done = self.job() | {'output': str(path/'done.json')}
            pending = self.job() | {'output': str(path/'pending.json')}
            pending['case'] = pending['case'] | {'cp': 2}
            failure = {'error': 'measurement drift exceeds 5% in all 3 rounds: [.1,.1,.1]',
                       'measurement_records': [{'converged_all_ranks': True,
                           'measurement_rounds': [{'half_p50_relative_drift': .1}] * 3}]}
            def read(output, job):
                if job['output'] == pending['output']:
                    raise FileNotFoundError(output)
            with mock.patch.object(batch.bench, 'read_measurement', side_effect=read):
                for attempt in range(3):
                    for rank in range(2):
                        batch.atomic_json(path/f'pending.failed-rank{rank}.json', failure)
                    result = batch.prepare_noise_retry([done, pending], 0)
                    self.assertEqual(result, [pending] if attempt < 2 else None)
            self.assertEqual(len(json.loads((path/'pending.noise-retries.json').read_text())), 2)
            self.assertEqual(len(list(path.glob('*.failed-rank*.noise-*.json'))), 4)

    def test_non_statistical_or_missing_rank_failure_never_retries(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            job = self.job() | {'output': str(path/'bad.json')}
            with mock.patch.object(batch.bench, 'read_measurement', side_effect=FileNotFoundError):
                for error in ('CUDA out of memory', 'route correctness failed', 'NCCL error'):
                    for rank in range(job['case']['cp']):
                        batch.atomic_json(path/f'bad.failed-rank{rank}.json', {'error': error})
                    self.assertIsNone(batch.prepare_noise_retry([job], 0))
                self.assertFalse((path/'bad.noise-retries.json').exists())

    def test_unanimous_finite_warmup_failure_can_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            job = self.job() | {'output': str(path/'warm.json')}
            failure = {'error': 'warmup did not converge within 5s: last windows [0.07,0.05,0.05]'}
            with mock.patch.object(batch.bench, 'read_measurement', side_effect=FileNotFoundError):
                for rank in range(job['case']['cp']-1):
                    batch.atomic_json(path/f'warm.failed-rank{rank}.json', failure)
                self.assertIsNone(batch.prepare_noise_retry([job], 0))
                batch.atomic_json(path/f'warm.failed-rank{job["case"]["cp"]-1}.json', failure)
                self.assertEqual(batch.prepare_noise_retry([job], 0), [job])

    def test_candidate_watchdog_stops_only_owned_process(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(TimeoutError, 'No candidate progress'):
                batch.run_batch([sys.executable, '-c', 'import time; time.sleep(10)'],
                                dict(__import__('os').environ), Path(folder)/'batch.json', .05)


if __name__ == '__main__':
    unittest.main()
