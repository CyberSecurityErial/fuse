import copy
import json
import shutil
import unittest

from summarize_sm103_gemm_gaps import summarize, comparison, read_observation
import test_sm103_fused_summary as fixtures


class GapAccounting(unittest.TestCase):
    def data(self):
        rows = [f'profile_gemm_gap_sample,component=compute_reference,rank=0,instrumented={mode},sample={i},event_ms=1'
                for mode in (0, 1) for i in range(50)]
        for cta, end, gap in ((0, 1000, 100), (1, 1200, -20)):
            rows.append(f'profile_gemm_gap,component=compute_reference,rank=0,cta={cta},tiles=2,'
                        f'first_input_ns=100,last_completion_ns={end},service_ns={end-100-gap},'
                        f'signed_gap_ns={gap},positive_gap_ns={max(gap,0)},span_ns={end-100},trace_event_ms=1')
        return rows

    def test_critical_chain_not_parallel_sum(self):
        result = summarize('\n'.join(self.data()))
        self.assertEqual(result['last_completion_cta'], 1)
        self.assertEqual(result['observed_chain_gap_fraction'], -20 / 1e6)
        self.assertEqual(result['tile_count'], 4)
        self.assertIsNone(result['causal_throughput_loss_share'])

    def test_reject_missing_samples_and_duplicate_ctas(self):
        rows = self.data()
        for bad in (rows[1:], rows + [rows[-1]]):
            with self.assertRaises(ValueError):
                summarize('\n'.join(bad))

    def test_reject_nonclosing_timeline(self):
        rows = self.data()
        rows[-1] = rows[-1].replace('span_ns=1100', 'span_ns=1099')
        with self.assertRaises(ValueError):
            summarize('\n'.join(rows))


class GapEvidenceJoin(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.FusedSummaryTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.make_fixture(profile=True, profile_detail='full', seq_local=32768,
            schedule=dict(oproj_raster='along_n', max_swizzle_size=4), comm_layout='rows')
        job = json.loads((self.fixture.control / 'job.json').read_text())
        job.update(oproj_gap_probe=True, directions='oproj')
        for path in (self.fixture.root / 'job.json', self.fixture.control / 'job.json'):
            self.fixture.write_json(path, job)
        text = self.fixture.log.read_text().replace('PASS:', '\n'.join(GapAccounting().data()) + '\nPASS:')
        self.fixture.log.write_text(text)
        self.fixture.repack()
        (self.fixture.root / 'gaps.json').write_text('{}')
        self.identity, _ = read_observation(self.fixture.root)
        self.table = dict(rows=[dict(model='fixture', cp=4, seq=131072, m=32768, n=1024, k=1024,
            compute_sms=140, control=False, a_ms=2., b_ms=1., retained_pct=50.,
            profile_identity=self.identity)])

    def test_complete_identity_joins_and_edited_summary_is_ignored(self):
        (self.fixture.root / 'gaps.json').write_text('{"plain_p50_ms":999999}')
        result = comparison(self.table, self.fixture.root.parent)
        self.assertEqual(result['rows'][0]['join_status'], 'matched')
        self.assertEqual(result['rows'][0]['plain_eager_ms'], 1.)
        self.assertEqual(result['rows'][0]['profile_identity'], self.identity)

    def test_old_or_incomplete_identity_is_not_guessed(self):
        for missing in ('profile_identity', 'raster', 'binary_sha256', 'effective_swizzle_size'):
            table = copy.deepcopy(self.table)
            if missing == 'profile_identity':
                table['rows'][0].pop(missing)
            else:
                table['rows'][0]['profile_identity'].pop(missing)
            self.assertEqual(comparison(table, self.fixture.root.parent)['rows'][0]['join_status'],
                             'unjoined_missing_identity')

    def test_same_mnk_and_budget_do_not_alias_config_or_identity(self):
        for field, value in (('tile', 'm128n256'), ('raster', 'along_m'),
                             ('effective_swizzle_size', '8'), ('oproj_comm_layout', 'columns'),
                             ('node', '09'), ('source_id', 'different'), ('build_inputs', 'different'),
                             ('binary_sha256', 'different'), ('environment_fingerprint', 'different'),
                             ('physical_gpu_uuid', 'different')):
            table = copy.deepcopy(self.table)
            table['rows'][0]['profile_identity'][field] = value
            with self.subTest(field=field):
                self.assertEqual(comparison(table, self.fixture.root.parent)['rows'][0]['join_status'],
                                 'unjoined_identity_mismatch')
        table = copy.deepcopy(self.table)
        table['rows'][0]['profile_identity']['cp'] = 8
        with self.assertRaisesRegex(ValueError, 'geometry or budget mismatch'):
            comparison(table, self.fixture.root.parent)

    def test_duplicate_profile_identity_does_not_silently_replace(self):
        shutil.copytree(self.fixture.root, self.fixture.root.with_name('duplicate'))
        with self.assertRaisesRegex(ValueError, 'Duplicate gap profile identity'):
            comparison(self.table, self.fixture.root.parent)

    def test_modified_archive_or_extracted_log_is_rejected(self):
        self.fixture.log.write_text(self.fixture.log.read_text() + '\nmodified\n')
        with self.assertRaisesRegex(ValueError, 'Extracted evidence differs'):
            comparison(self.table, self.fixture.root.parent)
        self.fixture.repack()
        archive = self.fixture.root / 'artifacts.tar.gz'
        archive.write_bytes(archive.read_bytes() + b'modified')
        with self.assertRaisesRegex(ValueError, 'Artifact SHA256 mismatch'):
            comparison(self.table, self.fixture.root.parent)


if __name__ == '__main__':
    unittest.main()
