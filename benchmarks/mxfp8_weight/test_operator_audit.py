"""CPU-only source snapshot checks and formal sample audit regressions."""
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import audit_operators as audit_module
from audit_operators import audit, parse_artifacts


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def case_fixture(backward=False):
    case = dict(id='backward' if backward else 'forward', cp=2, m=8, n=16, k=32)
    if backward:
        case.update(b_mnk=[8, 16, 32], w_mnk=[16, 32, 8])
    return case


def records_fixture(case):
    backward = 'b_mnk' in case
    records = []
    for backend in ('fuse', 'pure_cublas'):
        for launch in ('eager', 'graph'):
            for mode in (('immediate', 'deferred') if backward else (None,)):
                record = dict(backend=backend, launch=launch, weight_mode=mode,
                              grad_dtype='fp32' if backward else None,
                              beta=int(mode == 'deferred') if backward else None,
                              warmup=10, iterations=50, config=dict(tile_m=128),
                              correctness=dict(output=dict(all_ranks_finite=True)))
                for phase in (('data', 'weight', 'total') if backward else ('forward',)):
                    rank_samples = np.asarray([[1. + rank + i for i in range(50)]
                                               for rank in range(case['cp'])])
                    samples = rank_samples.max(axis=0)
                    mnk = case['b_mnk'] if backward else [case['m'], case['n'], case['k']]
                    flops = (4 if phase == 'total' else 2) * math.prod(mnk)
                    record[phase] = dict(rank_samples_us=rank_samples.tolist(),
                                         samples_us=samples.tolist(), p50_us=float(np.median(samples)),
                                         p95_us=float(np.percentile(samples, 95)),
                                         mean_us=float(np.mean(samples)), flops_per_gpu=flops,
                                         tflops_per_gpu=flops / float(np.median(samples)) / 1e6)
                records.append(record)
    return records


class OperatorAuditTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = 'bench/source.py'
        self.binary = 'build/liboperator.so'
        for relative, data in [(self.source, b'original source\n'),
                               ('CMakeLists.txt', b'original build recipe\n'),
                               (self.binary, b'\x7fELForiginal binary')]:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.git('init', '-q')
        self.git('add', self.source, 'CMakeLists.txt')
        self.git('-c', 'user.name=Audit Test', '-c', 'user.email=audit@example.invalid',
                 '-c', 'commit.gpgsign=false', 'commit', '-qm', 'fixture')
        self.revision = self.git('rev-parse', 'HEAD').strip()
        self.forward, self.backward = case_fixture(), case_fixture(True)
        self.report = dict(schema='mxfp8-production-operators-v1', complete=True,
                           sources={source: sha256((self.root / source).read_bytes())
                                    for source in (self.source, 'CMakeLists.txt', self.binary)},
                           cases=[dict(case=copy.deepcopy(case), records=records_fixture(case))
                                  for case in (self.forward, self.backward)])
        self.report_path = self.root / 'report.json'
        for name, value in [('ROOT', self.root), ('full_matrix', lambda: [self.forward]),
                            ('backward_matrix', lambda: [self.backward])]:
            patcher = patch.object(audit_module, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def git(self, *args):
        return subprocess.run(['git', '-C', str(self.root), *args], check=True,
                              capture_output=True, text=True).stdout

    def run_audit(self, **kwargs):
        self.report_path.write_text(json.dumps(self.report))
        return audit([self.report_path], **kwargs)

    def assert_error(self, result, text):
        self.assertFalse(result['complete'])
        self.assertTrue(any(text in error for error in result['errors']), result['errors'])

    def snapshot(self):
        path = self.root / 'snapshot.so'
        path.write_bytes((self.root / self.binary).read_bytes())
        return path

    def test_default_strict_worktree_passes(self):
        result = self.run_audit(require_full=True)
        self.assertTrue(result['complete'])
        self.assertEqual((result['records'], result['missing_records']), (12, 0))
        self.assertIsNone(result['source_revision'])
        self.assertEqual(result['artifacts'], {})

    def test_revision_passes_before_and_after_worktree_source_edits(self):
        self.assertTrue(self.run_audit(source_revision=self.revision)['complete'])
        for source in (self.source, 'CMakeLists.txt'):
            (self.root / source).write_text('changed worktree\n')
        result = self.run_audit()
        self.assert_error(result, f'changed source {self.source}')
        self.assert_error(result, 'changed source CMakeLists.txt')
        result = self.run_audit(source_revision='HEAD')
        self.assertTrue(result['complete'])
        self.assertEqual(result['source_revision'], self.revision)

    def test_revision_reads_deleted_worktree_source(self):
        (self.root / self.source).unlink()
        self.assert_error(self.run_audit(), f'cannot verify source {self.source}')
        self.assertTrue(self.run_audit(source_revision=self.revision)['complete'])

    def test_wrong_revision_hash_is_not_replaced_by_matching_worktree(self):
        changed = b'changed worktree\n'
        (self.root / self.source).write_bytes(changed)
        self.report['sources'][self.source] = sha256(changed)
        self.assertTrue(self.run_audit()['complete'])
        self.assert_error(self.run_audit(source_revision=self.revision), f'changed source {self.source}')

    def test_unknown_revision_does_not_fallback(self):
        with self.assertRaises(ValueError):
            self.run_audit(source_revision='nonexistent-revision')

    def test_unknown_revision_source_does_not_fallback_to_worktree(self):
        source = 'bench/new_source.py'
        (self.root / source).write_bytes(b'not committed')
        self.report['sources'][source] = sha256(b'not committed')
        self.assertTrue(self.run_audit()['complete'])
        self.assert_error(self.run_audit(source_revision=self.revision), f'cannot verify source {source}')

    def test_missing_source_returns_explicit_audit_error(self):
        self.report['sources']['bench/missing.py'] = sha256(b'anything')
        self.assert_error(self.run_audit(), 'cannot verify source bench/missing.py')

    def test_unmapped_binary_stays_current_in_revision_mode(self):
        (self.root / self.binary).write_bytes(b'\x7fELFnew binary')
        self.assert_error(self.run_audit(source_revision=self.revision), f'changed source {self.binary}')

    def test_explicit_snapshot_works_in_default_and_revision_modes(self):
        snapshot = self.snapshot()
        (self.root / self.binary).unlink()
        for revision in (None, self.revision):
            result = self.run_audit(source_revision=revision, artifacts={self.binary: snapshot})
            self.assertTrue(result['complete'])
            self.assertEqual(result['artifacts'], {self.binary: str(snapshot)})

    def test_wrong_missing_or_nonbinary_snapshot_rejected(self):
        snapshot = self.root / 'wrong.so'
        for contents, message in [(b'\x7fELFwrong binary', 'changed source'),
                                  (b'plain text', 'not an ELF')]:
            with self.subTest(contents=contents):
                snapshot.write_bytes(contents)
                self.assert_error(self.run_audit(artifacts={self.binary: snapshot}), message)
        snapshot.unlink()
        self.assert_error(self.run_audit(artifacts={self.binary: snapshot}), 'cannot verify source')

    def test_source_mapping_is_rejected_even_if_snapshot_matches(self):
        for source in (self.source, 'CMakeLists.txt'):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, 'cannot replace source'):
                self.run_audit(source_revision=self.revision, artifacts={source: self.root / source})

    def test_unknown_artifact_mapping_is_rejected(self):
        self.assert_error(self.run_audit(artifacts={'build/unknown.so': self.snapshot()}),
                          'unknown artifact mapping: build/unknown.so')

    def test_duplicate_malformed_and_source_cli_mappings_rejected(self):
        for entries in [('missing-equals',), ('=snapshot.so',), ('build/a.so=',),
                        ('build/a.so=a', 'build/a.so=b'), (f'{self.source}=snapshot.so',)]:
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                parse_artifacts(entries)

    def test_multiple_binary_mappings_are_independent(self):
        second = 'build/libsecond.a'
        second_snapshot = self.root / 'second.a'
        second_snapshot.write_bytes(b'!<arch>\noriginal archive')
        self.report['sources'][second] = sha256(second_snapshot.read_bytes())
        artifacts = parse_artifacts([f'{self.binary}={self.snapshot()}', f'{second}={second_snapshot}'])
        self.assertTrue(self.run_audit(source_revision=self.revision, artifacts=artifacts)['complete'])
        second_snapshot.write_bytes(b'!<arch>\nwrong archive')
        self.assert_error(self.run_audit(artifacts=artifacts), f'changed source {second}')

    def test_non_repository_source_paths_rejected(self):
        for source in (str(self.root / self.source), '../source.py'):
            with self.subTest(source=source):
                self.report['sources'] = {source: sha256(b'anything')}
                self.assert_error(self.run_audit(), 'must be repository-relative')

    def test_full_coverage_is_still_required(self):
        self.report['cases'][0]['records'].pop()
        self.assert_error(self.run_audit(require_full=True, source_revision=self.revision),
                          '1 missing formal records')

    def test_incomplete_runner_is_still_rejected(self):
        self.report['complete'] = False
        self.assert_error(self.run_audit(require_full=True, source_revision=self.revision), 'incomplete runner')

    def test_wrong_rank_count_zero_and_nonfinite_samples_rejected(self):
        original = copy.deepcopy(self.report)
        for samples in ([[1.] * 50], [[0.] * 50] * 2, [[float('nan')] * 50] * 2):
            with self.subTest(samples=samples[0][0]):
                self.report = copy.deepcopy(original)
                self.report['cases'][0]['records'][0]['forward']['rank_samples_us'] = samples
                self.assert_error(self.run_audit(source_revision=self.revision), 'invalid sample vectors')

    def test_sample_wise_rank_max_is_still_checked(self):
        self.report['cases'][0]['records'][0]['forward']['samples_us'][0] = 1.
        self.assert_error(self.run_audit(source_revision=self.revision), 'not sample-wise rank-max')

    def test_timing_and_throughput_summaries_still_checked(self):
        original = copy.deepcopy(self.report)
        for field in ('p50_us', 'p95_us', 'mean_us', 'flops_per_gpu', 'tflops_per_gpu'):
            with self.subTest(field=field):
                self.report = copy.deepcopy(original)
                self.report['cases'][0]['records'][0]['forward'][field] *= 2
                result = self.run_audit(source_revision=self.revision)
                self.assert_error(result, 'mismatch')

    def test_formal_iterations_fp32_beta_and_correctness_still_checked(self):
        original = copy.deepcopy(self.report)
        for field, value, message in [('warmup', 1, 'not formal 10+50'),
                                       ('iterations', 10, 'not formal 10+50'),
                                       ('grad_dtype', 'bf16', 'FP32/beta mismatch'),
                                       ('beta', 1, 'FP32/beta mismatch'),
                                       ('correctness', {'all_ranks_finite': False}, 'nonfinite correctness')]:
            with self.subTest(field=field):
                self.report = copy.deepcopy(original)
                self.report['cases'][1]['records'][0][field] = value
                self.assert_error(self.run_audit(source_revision=self.revision), message)

    def test_registry_and_duplicate_checks_still_apply(self):
        self.report['cases'][0]['case']['m'] += 1
        self.assert_error(self.run_audit(), 'registry mismatch')
        self.report['cases'] = [copy.deepcopy(self.report['cases'][1])] * 2
        self.assert_error(self.run_audit(), 'duplicate case')
        self.assert_error(self.run_audit(), 'duplicate/unknown record')


if __name__ == '__main__':
    unittest.main()
