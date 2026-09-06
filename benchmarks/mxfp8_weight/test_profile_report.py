"""CPU-only checks for diagnostic timing semantics, never GPU performance."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from profile_report import build_profile, span_us, write_profile


def fixture(operator='qkv_backward'):
    ctas = []
    for index, (start, role, end, grid) in enumerate([
            (2000, 4000, 4700, 4100), (2010, 3900, 4650, 4120),
            (2020, 3800, 4500, 4110), (2030, 3950, 4550, 4130)]):
        ctas.append(dict(start=start, end=end, active_start=2100 if index == 2 else 0,
                         role_done=role, grid_sync_done=grid,
                         fence_done=4200 if index == 0 else 0,
                         publish_done=4300 if index == 0 else 0,
                         source_ready=[4500, 4600, 0, 0, 0, 0, 0, 0] if index == 0 else [0] * 8))
    if operator == 'oproj_forward':
        for cta in ctas:
            for field in ('role_done', 'grid_sync_done', 'fence_done', 'publish_done'):
                cta[field] = 0
            cta['source_ready'] = [0] * 8
    markers = dict(dq_start=1000, dq_end=1500)
    if operator.endswith('backward'):
        markers.update(w_start=5000, w_end=7000)
    return [dict(rank=rank, config=dict(world_size=2, sm_count=4, comm_ctas=2,
                                       requested_comm_ctas=0, tile_m=128, tile_n=128),
                 ctas=copy.deepcopy(ctas), markers=copy.deepcopy(markers)) for rank in (0, 1)]


def shift_rank(item, offset):
    for cta in item['ctas']:
        for field, value in cta.items():
            if field == 'source_ready':
                cta[field] = [stamp + offset if stamp else 0 for stamp in value]
            elif value:
                cta[field] += offset
    for field, value in item['markers'].items():
        if value:
            item['markers'][field] += offset


class ProfileReportTest(unittest.TestCase):
    def test_four_operators(self):
        for operator in ('qkv_forward', 'oproj_forward', 'qkv_backward', 'oproj_backward'):
            with self.subTest(operator=operator):
                result = build_profile(operator, fixture(operator))
                self.assertTrue(result['summary']['profiling_only'])
                self.assertEqual(result['summary']['operator'], operator)
                self.assertEqual(len(result['summary']['ranks']), 2)

    def test_dq_compute_route_overlap_and_w(self):
        row = build_profile('qkv_backward', fixture())['summary']['ranks'][0]
        self.assertEqual(row['dq_us'], 0.5)
        self.assertEqual(row['dq_to_role_kernel_us'], 0.5)
        self.assertEqual(row['role_kernel_us'], 2.7)
        self.assertEqual(row['compute_role_us'], 1.93)
        self.assertEqual(row['route_role_us'], 2)
        self.assertEqual(row['overlap_us'], 1.93)
        self.assertEqual(row['b_to_w_marker_us'], 0.3)
        self.assertEqual(row['wgrad_marker_us'], 2)
        self.assertEqual(row['diagnostic_boundary_us'], 6)

    def test_parallel_source_waits_not_added(self):
        result = build_profile('qkv_backward', fixture())
        row = result['summary']['ranks'][0]
        self.assertEqual(row['source_wait_us'], [0.2, 0.3])
        self.assertEqual(row['exposed_source_wait_us'], 0.3)
        self.assertNotEqual(row['exposed_source_wait_us'], sum(row['source_wait_us']))
        self.assertAlmostEqual(row['finalize_us'], row['system_fence_us'] + row['publish_us'] +
                               row['exposed_source_wait_us'] + row['kernel_retire_us'])
        waits = [event for event in result['trace']['traceEvents']
                 if event['pid'] == 0 and event['name'].startswith('wait for source')]
        self.assertEqual([event['ts'] for event in waits], [3.3, 3.3])

    def test_first_ready_wait_observed_only(self):
        row = build_profile('qkv_backward', fixture())['summary']['ranks'][0]
        waits = row['first_ready_wait']
        self.assertEqual((waits['eligible_compute_ctas'], waits['observed_compute_ctas'],
                          waits['unobserved_compute_ctas']), (2, 1, 1))
        self.assertEqual(waits['maximum_us'], 0.08)
        rows = fixture()
        rows[0]['ctas'][2]['active_start'] = 0
        empty = build_profile('qkv_backward', rows)['summary']['ranks'][0]['first_ready_wait']
        self.assertIsNone(empty['mean_us'])
        self.assertEqual(empty['observed_compute_ctas'], 0)

    def test_producer_does_not_invent_ready_wait(self):
        result = build_profile('oproj_backward', fixture('oproj_backward'))
        self.assertIsNone(result['summary']['ranks'][0]['first_ready_wait'])
        self.assertFalse(any(event.get('cat') == 'ready wait' for event in result['trace']['traceEvents']))

    def test_oproj_forward_uses_end_without_finalize(self):
        result = build_profile('oproj_forward', fixture('oproj_forward'))
        row = result['summary']['ranks'][0]
        self.assertEqual(row['route_role_us'], 2.7)
        self.assertEqual(row['compute_role_us'], 2.53)
        self.assertIsNone(row['finalize_us'])
        self.assertIsNone(row['grid_sync_us'])
        self.assertIsNone(row['exposed_source_wait_us'])
        self.assertIsNone(row['wgrad_marker_us'])
        self.assertFalse(any(event.get('cat') in ('finalize', 'CTA finalize', 'weight gradient')
                             for event in result['trace']['traceEvents']))

    def test_oproj_forward_rejects_wrong_finalize_schema(self):
        rows = fixture('oproj_forward')
        rows[0]['ctas'][0]['role_done'] = 4000
        with self.assertRaisesRegex(ValueError, 'OProj F has no'):
            build_profile('oproj_forward', rows)

    def test_disjoint_envelopes_have_zero_overlap(self):
        rows = fixture('oproj_forward')
        for row in rows:
            for cta in row['ctas'][:2]:
                cta['end'] = 2500
            for cta in row['ctas'][2:]:
                cta['start'] = 3000
                cta['active_start'] = 3200
        result = build_profile('oproj_forward', rows)
        self.assertEqual(result['summary']['ranks'][0]['overlap_us'], 0)
        self.assertFalse(any(event.get('cat') == 'overlap' for event in result['trace']['traceEvents']))

    def test_independent_clock_offset_does_not_change_durations_or_trace(self):
        rows = fixture()
        before = build_profile('qkv_backward', rows)
        shift_rank(rows[0], 10**15)
        shift_rank(rows[1], 10**18)
        after = build_profile('qkv_backward', rows)
        self.assertEqual(before['trace']['traceEvents'], after['trace']['traceEvents'])
        for previous, current in zip(before['summary']['ranks'], after['summary']['ranks']):
            previous.pop('origin_ns')
            current.pop('origin_ns')
            self.assertEqual(previous, current)

    def test_reversed_rank_order_is_stable(self):
        rows = fixture()
        self.assertEqual(build_profile('qkv_backward', rows), build_profile('qkv_backward', list(reversed(rows))))

    def test_missing_or_duplicate_rank_rejected(self):
        for rows in (fixture()[:1], [fixture()[0], fixture()[0]]):
            with self.assertRaisesRegex(ValueError, 'missing or duplicate rank'):
                build_profile('qkv_backward', rows)

    def test_missing_cta_rejected(self):
        rows = fixture()
        rows[0]['ctas'].pop()
        with self.assertRaisesRegex(ValueError, 'physical CTA records'):
            build_profile('qkv_backward', rows)

    def test_invalid_counts_rejected(self):
        for field, value in [('comm_ctas', 0), ('comm_ctas', 4), ('sm_count', 5), ('world_size', 3)]:
            rows = fixture()
            rows[1]['config'][field] = value
            with self.assertRaises(ValueError):
                build_profile('qkv_backward', rows)

    def test_missing_cta_stamp_rejected(self):
        rows = fixture()
        rows[0]['ctas'][2]['start'] = 0
        with self.assertRaisesRegex(ValueError, 'missing or nonpositive CTA'):
            build_profile('qkv_backward', rows)

    def test_negative_or_zero_lifetime_rejected(self):
        for end in (1000, 2000):
            rows = fixture()
            rows[0]['ctas'][0]['end'] = end
            with self.assertRaisesRegex(ValueError, 'nonpositive CTA lifetime'):
                build_profile('qkv_backward', rows)
        with self.assertRaisesRegex(ValueError, 'negative timer span'):
            span_us(2, 1)

    def test_noninteger_and_negative_stamp_rejected(self):
        for stamp in (1.2, float('nan'), -1, True):
            rows = fixture()
            rows[0]['ctas'][2]['active_start'] = stamp
            with self.assertRaises(ValueError):
                build_profile('qkv_backward', rows)

    def test_role_and_ready_outside_lifetime_rejected(self):
        for field, value in [('role_done', 4800), ('role_done', 1900), ('active_start', 1800),
                             ('active_start', 4100)]:
            rows = fixture()
            rows[0]['ctas'][0][field] = value
            with self.assertRaises(ValueError):
                build_profile('qkv_backward', rows)

    def test_finalize_monotonicity_rejected(self):
        for field, value in [('grid_sync_done', 3900), ('fence_done', 4000), ('publish_done', 4100)]:
            rows = fixture()
            rows[0]['ctas'][0][field] = value
            with self.assertRaises(ValueError):
                build_profile('qkv_backward', rows)
        rows = fixture()
        rows[0]['ctas'][2]['grid_sync_done'] = 3900
        with self.assertRaisesRegex(ValueError, 'last local role'):
            build_profile('qkv_backward', rows)

    def test_source_wait_stamps_rejected(self):
        for source, stamp in [(0, 0), (0, 4200), (1, 4800), (2, 4500)]:
            rows = fixture()
            rows[0]['ctas'][0]['source_ready'][source] = stamp
            with self.assertRaises(ValueError):
                build_profile('qkv_backward', rows)
        rows = fixture()
        rows[0]['ctas'][0]['source_ready'] = [4500, 4600]
        with self.assertRaisesRegex(ValueError, 'eight entries'):
            build_profile('qkv_backward', rows)

    def test_nonroot_finalize_stamps_rejected(self):
        rows = fixture()
        rows[0]['ctas'][2]['publish_done'] = 4300
        with self.assertRaisesRegex(ValueError, 'only CTA0'):
            build_profile('qkv_backward', rows)

    def test_markers_must_bracket_correct_stages(self):
        for field, value in [('dq_end', 2100), ('dq_start', 1600), ('w_start', 4600), ('w_end', 4900)]:
            rows = fixture()
            rows[0]['markers'][field] = value
            with self.assertRaises(ValueError):
                build_profile('qkv_backward', rows)
        rows = fixture('qkv_forward')
        rows[0]['markers']['w_start'] = 5000
        with self.assertRaisesRegex(ValueError, 'forward cannot contain W'):
            build_profile('qkv_forward', rows)

    def test_no_raw_cta_dump_and_bounded_per_cta_trace(self):
        result = build_profile('qkv_backward', fixture())
        self.assertNotIn('ctas', result['summary']['ranks'][0])
        self.assertEqual(result['trace']['metadata']['rank_configs'][0]['config']['comm_ctas'], 2)
        self.assertLessEqual(len(result['trace']['traceEvents']), 2 * (4 * 3 + 2 + 20))
        for event in result['trace']['traceEvents']:
            if event['ph'] == 'X':
                self.assertGreaterEqual(event['ts'], 0)
                self.assertGreater(event['dur'], 0)
            if event['tid'] >= 1000:
                self.assertLess(event['tid'], 1004)

    def test_write_exactly_two_valid_json_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = write_profile('qkv_backward', fixture(), Path(directory) / 'qkv', dict(profiling_build=True))
            self.assertEqual(sorted(path.name for path in Path(directory).iterdir()),
                             ['qkv_perfetto.json', 'qkv_summary.json'])
            for path in paths.values():
                json.loads(path.read_text())
            summary = json.loads(paths['summary'].read_text())
            self.assertTrue(summary['profiling_only'])
            self.assertTrue(summary['metadata']['profiling_build'])

    def test_invalid_input_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                write_profile('unknown', fixture(), Path(directory) / 'bad')
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
