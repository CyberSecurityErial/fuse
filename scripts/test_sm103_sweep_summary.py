import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

import summarize_sm103_sweep as summary


class SweepSummaryTests(unittest.TestCase):
    def fixture(self, folder):
        args = argparse.Namespace(directions='qkv', models='production_qwen_dense',
            seqs=(1024,), cps=(4,), devices='0,1,2,3,4,5,6,7', stage='sweep',
            results=Path('/remote/experiment'), library=Path('/remote/library.so'),
            python='python3', sm_count=148)
        case = next(summary.bench.cases(args))
        jobs = [summary.bench.make_job(args, case, 'cublaslt_nccl', 'eager', config)
                for config in summary.bench.initial_configs('cublaslt_nccl')[:2]]
        plan = dict(stage='sweep', precision='bf16', fingerprint='test', jobs=jobs)
        path = Path(folder) / 'results'
        path.mkdir()
        (path / 'sweep_plan.json').write_text(json.dumps(plan))
        expected = {(job['group'], summary.bench.digest(job['config'])): case for job in jobs}
        return path, plan, expected

    def test_full_matrix_inventory(self):
        expected = summary.expected_candidates()
        self.assertEqual(len(expected), 20736)
        self.assertEqual(len({key[0] for key in expected}), 768)

    def test_selection_remaps_paths_and_marks_sweep_only(self):
        with tempfile.TemporaryDirectory() as folder:
            path, plan, expected = self.fixture(folder)
            out = Path(folder) / 'summary'
            with mock.patch.object(summary, 'expected_candidates', return_value=expected), \
                 mock.patch.object(summary.bench, 'read_measurement', side_effect=[
                     dict(p50_ms=2, p95_ms=3), dict(p50_ms=1, p95_ms=4)]) as read:
                report = summary.summarize(path, out)
            self.assertEqual(read.call_count, 2)
            self.assertEqual(read.call_args.args[0].parent, path.resolve() / 'sweep')
            self.assertFalse(report['independently_remeasured'])
            self.assertFalse(report['globally_optimal'])
            self.assertEqual(report['winner_groups'], 1)
            self.assertIn('sweep', (out / 'summary.csv').read_text())

    def test_incomplete_or_invalid_measurements_publish_nothing(self):
        for error in (FileNotFoundError('rank missing'), ValueError('incorrect samples')):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as folder:
                path, plan, expected = self.fixture(folder)
                out = Path(folder) / 'summary'
                with mock.patch.object(summary, 'expected_candidates', return_value=expected), \
                     mock.patch.object(summary.bench, 'read_measurement', side_effect=error):
                    with self.assertRaises(type(error)):
                        summary.summarize(path, out)
                self.assertFalse(out.exists())

    def test_missing_plan_candidate_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path, plan, expected = self.fixture(folder)
            plan['jobs'].pop()
            (path / 'sweep_plan.json').write_text(json.dumps(plan))
            with mock.patch.object(summary, 'expected_candidates', return_value=expected):
                with self.assertRaisesRegex(ValueError, 'complete historical'):
                    summary.summarize(path, Path(folder) / 'summary')


class PartialSweepSummaryTests(unittest.TestCase):
    def fixture(self, folder, seqs=(131072,)):
        args = argparse.Namespace(directions='qkv', models='production_qwen_dense',
            seqs=seqs, cps=(8,), devices='0,1,2,3,4,5,6,7', stage='sweep',
            results=Path('/remote/experiment'), library=Path('/remote/library.so'),
            python='python3', sm_count=148)
        jobs = [summary.bench.make_job(args, case, backend, 'eager', config)
                for case in summary.bench.cases(args)
                for backend in ('cublaslt_nccl', 'te_ub')
                for config in summary.bench.initial_configs(backend)]
        plan = dict(stage='sweep', precision='bf16', fingerprint='source-v7', jobs=jobs)
        path = Path(folder) / 'results'
        path.mkdir()
        (path / 'sweep_plan.json').write_text(json.dumps(plan))
        expected = {(job['group'], summary.bench.digest(job['config'])): job['case']
                    for job in jobs}
        return path, plan, expected

    def consume(self, path, out, **overrides):
        options = dict(expected_fingerprint='source-v7', seqs=(131072,),
            directions=('qkv',), models=('production_qwen_dense',), cps=(8,), launches=('eager',))
        options.update(overrides)
        with contextlib.redirect_stdout(io.StringIO()):
            return summary.summarize_partial(path, out, **options)

    @staticmethod
    def stats(path, job):
        return dict(p50_ms=1 if job['backend'] == 'te_ub' else 2, p95_ms=3)

    def test_complete_real_config_sets_select_both_then_strongest(self):
        with tempfile.TemporaryDirectory() as folder:
            path, plan, expected = self.fixture(folder)
            out = Path(folder) / 'summary'
            with mock.patch.object(summary, 'expected_candidates', return_value=expected), \
                 mock.patch.object(summary.bench, 'read_measurement', side_effect=self.stats) as read:
                report = self.consume(path, out)
            self.assertEqual(read.call_count, 48 + 6)
            self.assertEqual(report['winner_groups'], 2)
            self.assertEqual(report['strongest_baselines'], 1)
            self.assertEqual(report['pending_groups'], 0)
            self.assertTrue(report['partial'])
            self.assertFalse(report['full_stage_achieved'])
            self.assertFalse(report['independently_remeasured'])
            strongest = json.loads((out / 'strongest.json').read_text())
            self.assertEqual(strongest[0]['backend'], 'te_ub')
            self.assertTrue(strongest[0]['node2_same_boundary_validation_required'])
            self.assertEqual(set(strongest[0]['comparison_groups']), {'te_ub', 'cublaslt_nccl'})
            self.assertEqual(read.call_args.args[0].parent, path.resolve() / 'sweep')

    def test_one_missing_candidate_prevents_group_winner_and_strongest(self):
        with tempfile.TemporaryDirectory() as folder:
            path, plan, expected = self.fixture(folder)
            out = Path(folder) / 'summary'
            missing = plan['jobs'][0]['output']

            def read(raw, job):
                if job['output'] == missing:
                    raise FileNotFoundError('rank7 missing')
                return self.stats(raw, job)

            with mock.patch.object(summary, 'expected_candidates', return_value=expected), \
                 mock.patch.object(summary.bench, 'read_measurement', side_effect=read):
                report = self.consume(path, out)
            self.assertEqual(report['winner_groups'], 1)
            self.assertEqual(report['validated_logical_candidates'], 53)
            self.assertEqual(report['strongest_baselines'], 0)
            self.assertEqual(report['pending_comparisons'], 1)
            winners = json.loads((out / 'winners.json').read_text())
            self.assertEqual([row['backend'] for row in winners], ['te_ub'])
            groups = json.loads((out / 'groups.json').read_text())
            pending = next(row for row in groups if row['status'] == 'pending')
            self.assertEqual(pending['expected_candidates'], 48)
            self.assertEqual(pending['validated_candidates'], 47)
            self.assertNotIn('winner_group', pending)
            self.assertEqual(pending['unavailable_candidates'][0]['reason'], 'missing_raw_or_rank')

    def test_invalid_measurement_is_explicit_pending_not_best_so_far(self):
        with tempfile.TemporaryDirectory() as folder:
            path, plan, expected = self.fixture(folder)
            out = Path(folder) / 'summary'
            with mock.patch.object(summary, 'expected_candidates', return_value=expected), \
                 mock.patch.object(summary.bench, 'read_measurement', side_effect=ValueError('unstable samples')):
                report = self.consume(path, out)
            self.assertEqual(report['winner_groups'], 0)
            self.assertEqual(report['pending_groups'], 2)
            self.assertEqual(json.loads((out / 'winners.json').read_text()), [])
            self.assertEqual(json.loads((out / 'strongest.json').read_text()), [])
            groups = json.loads((out / 'groups.json').read_text())
            self.assertEqual(groups[0]['unavailable_candidates'][0]['reason'], 'invalid_measurement')

    def test_duplicate_replacement_or_truncated_plan_cannot_redefine_candidates(self):
        for mutation in ('duplicate', 'replacement', 'truncated'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                path, plan, expected = self.fixture(folder)
                if mutation == 'duplicate':
                    plan['jobs'].append(plan['jobs'][0])
                elif mutation == 'replacement':
                    plan['jobs'][1] = plan['jobs'][0]
                else:
                    plan['jobs'].pop()
                (path / 'sweep_plan.json').write_text(json.dumps(plan))
                out = Path(folder) / 'summary'
                with mock.patch.object(summary, 'expected_candidates', return_value=expected), \
                     mock.patch.object(summary.bench, 'read_measurement') as read:
                    with self.assertRaisesRegex(ValueError, 'complete historical'):
                        self.consume(path, out)
                read.assert_not_called()
                self.assertFalse(out.exists())

    def test_fingerprint_is_required_and_must_match_before_reading_results(self):
        for fingerprint in ('', None, 'different-source'):
            with self.subTest(fingerprint=fingerprint), tempfile.TemporaryDirectory() as folder:
                path, plan, expected = self.fixture(folder)
                out = Path(folder) / 'summary'
                with mock.patch.object(summary, 'expected_candidates', return_value=expected), \
                     mock.patch.object(summary.bench, 'read_measurement') as read:
                    with self.assertRaisesRegex(ValueError, 'fingerprint'):
                        self.consume(path, out, expected_fingerprint=fingerprint)
                read.assert_not_called()
                self.assertFalse(out.exists())

    def test_explicit_filters_only_read_requested_original_groups(self):
        with tempfile.TemporaryDirectory() as folder:
            path, plan, expected = self.fixture(folder, seqs=(1024, 131072))
            out = Path(folder) / 'summary'
            with mock.patch.object(summary, 'expected_candidates', return_value=expected), \
                 mock.patch.object(summary.bench, 'read_measurement', side_effect=self.stats) as read:
                report = self.consume(path, out)
            self.assertEqual(read.call_count, 54)
            self.assertEqual({call.args[1]['case']['seq'] for call in read.call_args_list}, {131072})
            self.assertEqual(report['filters']['seq'], [131072])
            self.assertEqual(report['selected_groups'], 2)

    def test_unknown_empty_or_unmatched_filters_are_not_silently_dropped(self):
        invalid = [dict(seqs=()), dict(seqs=(65536, 131072)), dict(cps=(4,)),
                   dict(launches=('graph',)), dict(models=('unknown',)),
                   dict(directions=('qkv', 'oproj'))]
        for options in invalid:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as folder:
                path, plan, expected = self.fixture(folder)
                out = Path(folder) / 'summary'
                with mock.patch.object(summary, 'expected_candidates', return_value=expected), \
                     mock.patch.object(summary.bench, 'read_measurement') as read:
                    with self.assertRaises(ValueError):
                        self.consume(path, out, **options)
                read.assert_not_called()
                self.assertFalse(out.exists())

    def test_equal_p50_uses_existing_config_tie_break_not_p95(self):
        with tempfile.TemporaryDirectory() as folder:
            path, plan, expected = self.fixture(folder)
            out = Path(folder) / 'summary'
            with mock.patch.object(summary, 'expected_candidates', return_value=expected), \
                 mock.patch.object(summary.bench, 'read_measurement',
                     side_effect=lambda path, job: dict(p50_ms=1, p95_ms=job['config'].get('comm_sm', 100))):
                self.consume(path, out)
            winners = json.loads((out / 'winners.json').read_text())
            for winner in winners:
                configs = [json.dumps(job['config'], sort_keys=True) for job in plan['jobs']
                           if job['group'] == winner['group']]
                self.assertEqual(winner['config_json'], min(configs))

    def test_original_plan_geometry_sampling_and_output_contract_remain_enforced(self):
        for mutation in ('geometry', 'sampling', 'protocol', 'output'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                path, plan, expected = self.fixture(folder)
                if mutation == 'geometry':
                    plan['jobs'][0]['case'] = plan['jobs'][0]['case'] | {'hidden': 1024}
                elif mutation == 'sampling':
                    plan['jobs'][0]['warmup'] = 2
                elif mutation == 'protocol':
                    plan['jobs'][0]['env']['FUSE_SM103_MEASUREMENT'] = 'v1'
                else:
                    plan['jobs'][0]['output'] = '/remote/not-sweep/result.json'
                (path / 'sweep_plan.json').write_text(json.dumps(plan))
                with mock.patch.object(summary, 'expected_candidates', return_value=expected):
                    with self.assertRaises(ValueError):
                        self.consume(path, Path(folder) / 'summary')

    def test_real_reader_missing_rank_and_invalid_input_cannot_certify_group(self):
        with tempfile.TemporaryDirectory() as folder:
            path, plan, expected = self.fixture(folder)
            (path / 'sweep').mkdir()
            for job in plan['jobs']:
                raw = path / 'sweep' / Path(job['output']).name
                raw.write_text(json.dumps(dict(samples_ms=[1.] * 50,
                    correctness={'route_mismatches': 0}, world_size=8, launch='eager')))
                for rank in range(8):
                    metadata = dict(device={'compute_capability': '10.3'},
                        input_statistics=[{'tensors': {'activation': {
                            'nonzero_fraction': 1, 'sample_std': .1}}}],
                        measurement_records=[dict(converged_all_ranks=True, initial_warmup=10,
                            iterations=50, additional_warmup_cuda_ms=100,
                            minimum_warmup_cuda_ms=100, sample_half_p50_relative_drift=0)])
                    if job['backend'] == 'te_ub' and rank == 7:
                        metadata['input_statistics'] = []
                    if job['backend'] == 'cublaslt_nccl' and rank == 7:
                        continue
                    raw.with_suffix(f'.rank{rank}.json').write_text(json.dumps(metadata))
            out = Path(folder) / 'summary'
            with mock.patch.object(summary, 'expected_candidates', return_value=expected):
                report = self.consume(path, out)
            self.assertEqual(report['winner_groups'], 0)
            groups = json.loads((out / 'groups.json').read_text())
            self.assertEqual({row['unavailable_candidates'][0]['reason'] for row in groups},
                             {'missing_raw_or_rank', 'invalid_measurement'})

    def test_output_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path, plan, expected = self.fixture(folder)
            out = Path(folder) / 'summary'
            out.mkdir()
            with mock.patch.object(summary, 'expected_candidates', return_value=expected), \
                 mock.patch.object(summary.bench, 'read_measurement', side_effect=self.stats):
                with self.assertRaises(FileExistsError):
                    self.consume(path, out)


class SnapshotSweepTests(unittest.TestCase):
    fixture = PartialSweepSummaryTests.fixture

    def measurements(self, path, plan):
        (path / 'sweep').mkdir()
        for job in plan['jobs']:
            raw = path / 'sweep' / Path(job['output']).name
            raw.write_text(json.dumps(dict(samples_ms=[1.] * 50,
                correctness={'route_mismatches': 0}, world_size=8, launch='eager')))
            for rank in range(8):
                raw.with_suffix(f'.rank{rank}.json').write_text(json.dumps(dict(
                    device={'compute_capability': '10.3'},
                    input_statistics=[{'tensors': {'activation': {
                        'nonzero_fraction': 1, 'sample_std': .1}}}],
                    measurement_records=[dict(converged_all_ranks=True, initial_warmup=10,
                        iterations=50, additional_warmup_cuda_ms=100,
                        minimum_warmup_cuda_ms=100, sample_half_p50_relative_drift=0)])))

    def snapshot(self, path, out, **overrides):
        options = dict(expected_fingerprint='source-v7', seqs=(131072,),
            directions=('qkv',), models=('production_qwen_dense',), cps=(8,), launches=('eager',))
        options.update(overrides)
        with contextlib.redirect_stdout(io.StringIO()):
            return summary.snapshot_partial(path, out, **options)

    def test_complete_archive_hashes_exact_bytes_and_local_consumer_accepts(self):
        with tempfile.TemporaryDirectory() as folder:
            path, plan, expected = self.fixture(folder)
            self.measurements(path, plan)
            (path / 'profile.json').write_text('not a measurement')
            original = {str(file.relative_to(path)): file.read_bytes() for file in path.rglob('*') if file.is_file()}
            out = Path(folder) / 'snapshot'
            with mock.patch.object(summary, 'expected_candidates', return_value=expected) as inventory:
                receipt = self.snapshot(path, out)
            self.assertEqual(inventory.call_count, 1)
            self.assertEqual(receipt['complete_groups'], 2)
            self.assertEqual(receipt['archived_candidates'], 54)
            self.assertEqual(receipt['archived_measurement_files'], 486)
            archive = Path(receipt['archive'])
            self.assertEqual(receipt['archive_sha256'], hashlib.sha256(archive.read_bytes()).hexdigest())
            with tarfile.open(archive, 'r:gz') as tar:
                self.assertTrue(all(item.isfile() for item in tar.getmembers()))
                contents = {item.name: tar.extractfile(item).read() for item in tar.getmembers()}
            manifest = json.loads(contents['manifest.json'])
            self.assertEqual(set(manifest['files']), set(contents)-{'manifest.json'})
            for name, evidence in manifest['files'].items():
                self.assertEqual(str(summary.safe_relative_path(name)), name)
                self.assertEqual(evidence['size_bytes'], len(contents[name]))
                self.assertEqual(evidence['sha256'], hashlib.sha256(contents[name]).hexdigest())
            self.assertNotIn('results/profile.json', contents)
            self.assertEqual(contents['results/sweep_plan.json'], original['sweep_plan.json'])
            fetched = Path(folder) / 'fetched'
            for name, content in contents.items():
                if name.startswith('results/'):
                    target = fetched / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
            with mock.patch.object(summary, 'expected_candidates', return_value=expected), \
                 contextlib.redirect_stdout(io.StringIO()):
                report = summary.summarize_partial(fetched / 'results', Path(folder) / 'summary',
                    expected_fingerprint='source-v7', seqs=(131072,), directions=('qkv',), cps=(8,))
            self.assertEqual(report['strongest_baselines'], 1)
            self.assertFalse(report['full_stage_achieved'])
            self.assertEqual(original, {str(file.relative_to(path)): file.read_bytes()
                                       for file in path.rglob('*') if file.is_file()})

    def test_incomplete_group_archives_none_of_its_candidate_files(self):
        with tempfile.TemporaryDirectory() as folder:
            path, plan, expected = self.fixture(folder)
            self.measurements(path, plan)
            raw = path / 'sweep' / Path(plan['jobs'][0]['output']).name
            raw.with_suffix('.rank7.json').unlink()
            with mock.patch.object(summary, 'expected_candidates', return_value=expected):
                receipt = self.snapshot(path, Path(folder) / 'snapshot')
            self.assertEqual(receipt['archived_candidates'], 6)
            self.assertEqual(receipt['archived_measurement_files'], 54)
            self.assertEqual(receipt['complete_groups'], 1)
            self.assertEqual(receipt['pending_groups'], 1)
            self.assertEqual(receipt['strongest_baselines'], 0)
            with tarfile.open(receipt['archive'], 'r:gz') as tar:
                names = tar.getnames()
                groups = json.load(tar.extractfile('provenance/groups.json'))
            self.assertFalse(any('cublaslt_nccl' in name for name in names))
            self.assertEqual({row['status'] for row in groups}, {'complete', 'pending'})

    def test_symlink_file_or_parent_is_rejected_without_outside_read(self):
        for target_kind in ('file', 'directory'):
            with self.subTest(target_kind=target_kind), tempfile.TemporaryDirectory() as folder:
                path, plan, expected = self.fixture(folder)
                outside = Path(folder) / 'outside'
                outside.mkdir()
                secret = outside / Path(plan['jobs'][0]['output']).name
                secret.write_text('private unrelated content')
                if target_kind == 'directory':
                    (path / 'sweep').symlink_to(outside, target_is_directory=True)
                else:
                    (path / 'sweep').mkdir()
                    (path / 'sweep' / secret.name).symlink_to(secret)
                out = Path(folder) / 'snapshot'
                with mock.patch.object(summary, 'expected_candidates', return_value=expected), \
                     mock.patch.object(summary.bench, 'read_measurement') as reader:
                    with self.assertRaises(summary.SnapshotError):
                        self.snapshot(path, out)
                reader.assert_not_called()
                self.assertFalse(out.exists())

    def test_fifo_is_rejected_without_blocking(self):
        with tempfile.TemporaryDirectory() as folder:
            path, plan, expected = self.fixture(folder)
            (path / 'sweep').mkdir()
            os.mkfifo(path / 'sweep' / Path(plan['jobs'][0]['output']).name)
            with mock.patch.object(summary, 'expected_candidates', return_value=expected):
                with self.assertRaisesRegex(summary.SnapshotError, 'regular'):
                    self.snapshot(path, Path(folder) / 'snapshot')

    def test_changing_file_during_validation_invalidates_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            path, plan, expected = self.fixture(folder)
            self.measurements(path, plan)
            real_reader = summary.bench.read_measurement

            def changing_reader(captured, job):
                stats = real_reader(captured, job)
                raw = path / 'sweep' / Path(job['output']).name
                raw.write_bytes(raw.read_bytes() + b'\n')
                return stats

            out = Path(folder) / 'snapshot'
            with mock.patch.object(summary, 'expected_candidates', return_value=expected), \
                 mock.patch.object(summary.bench, 'read_measurement', side_effect=changing_reader):
                with self.assertRaisesRegex(summary.SnapshotError, 'changed'):
                    self.snapshot(path, out)
            self.assertFalse(out.exists())

    def test_changing_file_during_archive_prevents_final_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            path, plan, expected = self.fixture(folder)
            self.measurements(path, plan)
            raw = path / 'sweep' / Path(plan['jobs'][0]['output']).name
            real_addfile = tarfile.TarFile.addfile
            changed = False

            def changing_addfile(archive, member, stream=None):
                nonlocal changed
                real_addfile(archive, member, stream)
                if not changed:
                    raw.write_bytes(raw.read_bytes() + b'\n')
                    changed = True

            out = Path(folder) / 'snapshot'
            with mock.patch.object(summary, 'expected_candidates', return_value=expected), \
                 mock.patch.object(tarfile.TarFile, 'addfile', new=changing_addfile):
                with self.assertRaisesRegex(summary.SnapshotError, 'changed'):
                    self.snapshot(path, out)
            self.assertFalse((out / 'receipt.json').exists())
            self.assertFalse((out / 'snapshot.tar.gz').exists())
            self.assertTrue((out / 'snapshot.tar.gz.pending').exists())

    def test_invalid_member_paths_are_rejected(self):
        for name in ('/absolute', '../escape', 'a/../escape', 'a//b', './a',
                     'a/./b', '', 'a\\b', 'nul\x00byte'):
            with self.subTest(name=name), self.assertRaises(summary.SnapshotError):
                summary.safe_relative_path(name)

    def test_wrong_fingerprint_or_source_output_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path, plan, expected = self.fixture(folder)
            with mock.patch.object(summary, 'expected_candidates', return_value=expected):
                with self.assertRaisesRegex(ValueError, 'fingerprint'):
                    self.snapshot(path, Path(folder) / 'snapshot', expected_fingerprint='wrong')
                with self.assertRaisesRegex(summary.SnapshotError, 'outside'):
                    self.snapshot(path, path / 'snapshot')
            self.assertFalse((Path(folder) / 'snapshot').exists())


if __name__ == '__main__':
    unittest.main()
