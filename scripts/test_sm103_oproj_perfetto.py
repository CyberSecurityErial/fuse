"""Synthetic OProj timeline tests; no cluster or GPU operations."""
import unittest
from export_sm103_oproj_perfetto import make_trace, tile_consumption, cta_time_accounting


class OprojTraceTests(unittest.TestCase):
    def test_cta_accounting_closes_with_overlap_and_other_cta_tail(self):
        ctas = {(0, 0): dict(start=100, end=150),
                (0, 1): dict(start=110, end=200),
                (0, 2): dict(start=112, end=220)}
        pipes = {(0, 0): dict(cta=1, mma_begin=115, acc_wait_end=150),
                 (0, 1): dict(cta=1, mma_begin=130, acc_wait_end=180)}
        stages = {(0, 0, 0): dict(wait_end=120), (0, 1, 0): dict(wait_end=140)}
        gpu, = cta_time_accounting(ctas, pipes, stages, 1)
        p, = gpu['sampled_ctas']
        self.assertEqual((p['startup_ns'], p['observed_service_sum_ns'],
                          p['signed_gap_sum_ns'], p['drain_ns']), (20, 70, -10, 20))
        self.assertEqual(p['end_ns']+p['other_cta_tail_ns'], gpu['kernel_span_ns'])
        self.assertEqual(gpu['last_cta'], 2)
        self.assertEqual(p['closure_error_ns'], 0)

    def test_tile_consumption_is_observed_span_not_wait_subtraction(self):
        p = dict(cta=1, k_tiles=1, mma_begin=110, tmem_acquired=112,
                 mma_return=140, acc_wait_end=145, epi_return=155,
                 ready_begin0=120, ready_end0=125)
        s = dict(wait_begin=115, wait_end=130, issue_end=139)
        row, = tile_consumption({(0, 0): p}, {(0, 0, 0): s},
            m=100, n=200, k=64, tile_m=128, tile_n=256, world=1)
        self.assertEqual(row['flops'], 2*100*200*64)
        self.assertEqual(row['observed_service_us'], .015)
        self.assertEqual(row['ready_poll_sum_us'], .005)
        self.assertAlmostEqual(row['effective_tile_tflops'], 2*100*200*64/15/1000)
        self.assertIsNone(row['start_interval_us'])

    def test_inter_tile_identity_preserves_negative_overlap(self):
        p = dict(cta=1, k_tiles=1, mma_begin=110, tmem_acquired=112,
                 mma_return=140, acc_wait_end=145, epi_return=155,
                 ready_begin0=120, ready_end0=125)
        s = dict(wait_begin=115, wait_end=130, issue_end=139)
        # Next tile begins before prior completion is observed; do not turn
        # overlap into zero idle or subtract unrelated marginal medians.
        q = {key: value+10 if key != 'cta' and key != 'k_tiles' else value
             for key, value in p.items()}
        t = {key: value+10 for key, value in s.items()}
        rows = tile_consumption({(0, 0): p, (0, 1): q},
            {(0, 0, 0): s, (0, 1, 0): t}, m=128, n=512, k=64,
            tile_m=128, tile_n=256, world=1)
        r = rows[1]
        self.assertEqual(r['completion_to_next_input_us'], -.005)
        self.assertAlmostEqual(r['first_input_interval_us'],
            r['previous_observed_service_us']+r['completion_to_next_input_us'])

    def fixture(self):
        job = dict(world=1, comm_sm=1, global_seq=128, hidden=256,
                   oproj_policy_list='m128n256k64e32')
        lines = ['device,rank=0,sms=4', 'profile_host,A2A_GEMM,rank=0',
                 'profile_cta,A2A_GEMM,rank=0,cta=0,start=100,end=160,active_start=0',
                 'profile_cta,A2A_GEMM,rank=0,cta=1,start=90,end=200,active_start=126',
                 'profile_peer,rank=0,index=0,comm_cta=0,comm_slot=0,comm_valid=1,'
                 'task_begin=101,input_ready=105,g2s_issue=110,g2s_done=115,'
                 's2g_issue=116,s2g_done=120,publish_issue=121,release=125,'
                 'source_rank=0,task_id=0,row_chunk=0,copy_rows=128,copy_path=1,'
                 'valid=1,m=0,n=0,batch=0,acquire0=126',
                 'candidate_verified,A2A_GEMM,full_numeric=1,full_route=1,'
                 'payload_generations=2,performance_accepted=0',
                 'PASS: selected BF16 boundaries and diagnostic timelines']
        for phase in ('instrumented', 'host_stages'):
            for kind, field in [('correctness', 'mismatches'), ('route', 'bitwise_mismatches')]:
                lines.append(f'{kind},A2A_GEMM,profile_phase={phase},rank=0,'
                             f'nonfinite=0,{field}=0,checked=128,elements=128')
        return job, lines

    def test_phases_and_metadata(self):
        job, lines = self.fixture()
        result = make_trace(lines, job)
        names = {e['name'] for e in result['traceEvents']}
        self.assertIn('remote G2S', names)
        self.assertIn('local S2G (destination complete)', names)
        self.assertFalse(result['metadata']['performance_accepted'])
        self.assertTrue(all(e.get('dur', 0) >= 0 for e in result['traceEvents']))

    def test_missing_or_duplicate_rejected(self):
        job, lines = self.fixture()
        for bad in (lines + [lines[2]], [l for l in lines if not l.startswith('profile_peer,')],
                    [l for l in lines if not l.startswith('route,')]):
            with self.assertRaises(AssertionError):
                make_trace(bad, job)

    def test_column_chunk_does_not_claim_to_be_a_row_chunk(self):
        job, lines = self.fixture()
        result = make_trace([l.replace('copy_path=1', 'copy_path=3') for l in lines], job)
        phases = [e for e in result['traceEvents'] if e['name'] == 'remote G2S']
        self.assertEqual(len(phases), 1)
        self.assertEqual(phases[0]['args']['column_chunk'], 0)
        self.assertEqual(phases[0]['args']['comm_layout'], 'columns')
        self.assertNotIn('row_chunk', phases[0]['args'])

    def test_single_policy_flag_and_reject_multiple(self):
        job, lines = self.fixture()
        expected = make_trace(lines, job)['traceEvents']
        job['oproj_policy'] = job.pop('oproj_policy_list')
        job['oproj_policy_list'] = None
        self.assertEqual(make_trace(lines, job)['traceEvents'], expected)
        job['oproj_policy_list'] = 'm128n256k64e32,m128n128'
        with self.assertRaises(AssertionError):
            make_trace(lines, job)

    def test_racing_acquire_is_not_negative_duration(self):
        job, lines = self.fixture()
        result = make_trace([l.replace('acquire0=126', 'acquire0=123') for l in lines], job)
        self.assertTrue(any(e['ph'] == 'i' for e in result['traceEvents']))

    def test_invalid_phase_order_rejected(self):
        job, lines = self.fixture()
        with self.assertRaises(AssertionError):
            make_trace([l.replace('g2s_done=115', 'g2s_done=109') for l in lines], job)

    def pipeline_fixture(self):
        job, lines = self.fixture()
        job['oproj_pipeline_probe'] = True
        lines += [
            'profile_oproj_pipeline,rank=0,index=0,cta=1,mma_begin=110,tmem_acquired=112,'
            'mma_return=140,epi_begin=120,acc_wait_begin=121,acc_wait_end=145,'
            'tmem_release_begin=150,tmem_release_end=151,epi_return=155,k_tiles=1,'
            'ready_begin0=120,ready_end0=125,ready_joined0=126,cache_hit0=0',
            'profile_oproj_stage,rank=0,index=0,k=0,wait_begin=115,wait_end=130,issue_end=139']
        return job, lines

    def test_pipeline_roles_grouped_under_own_cta(self):
        job, lines = self.pipeline_fixture()
        trace = make_trace(lines, job)
        events = trace['traceEvents']
        wait = next(e for e in events if e['name'] == 'MMA input-stage wait')
        self.assertEqual(wait['tid'], 100 + 16 + 2)
        ready = next(e for e in events if e['name'] == 'ready check / polling')
        self.assertEqual(ready['args']['check_ns'], 5)
        self.assertEqual(ready['args']['check_begin_minus_release_ns'], -5)
        self.assertTrue(trace['metadata']['pipeline_probe'])
        self.assertEqual(trace['metadata']['pipeline_summary']['ready_check_p50_us'], .005)
        self.assertFalse(any('GEMM tile (' in e['name'] for e in events))

    def test_pipeline_missing_duplicate_or_unordered_rejected(self):
        job, lines = self.pipeline_fixture()
        for bad in (lines[:-1], lines + [lines[-1]],
                    [l.replace('wait_end=130', 'wait_end=114') for l in lines],
                    [l.replace('mma_return=140', 'mma_return=146') for l in lines]):
            with self.assertRaises(AssertionError):
                make_trace(bad, job)

    def test_absolute_clock_precision_preserved(self):
        job, lines = self.pipeline_fixture()
        from export_sm103_oproj_perfetto import parse
        shifted = []
        clocks = {'start', 'end', 'active_start', 'release', 'task_begin', 'input_ready',
                  'g2s_issue', 'g2s_done', 's2g_issue', 's2g_done', 'publish_issue', 'acquire0',
                  'mma_begin', 'tmem_acquired', 'mma_return', 'epi_begin', 'acc_wait_begin',
                  'acc_wait_end', 'tmem_release_begin', 'tmem_release_end', 'epi_return',
                  'ready_begin0', 'ready_end0', 'ready_joined0', 'wait_begin', 'wait_end', 'issue_end'}
        for line in lines:
            for key, value in parse(line).items():
                if key in clocks and int(value) > 0:
                    line = line.replace(f'{key}={value}', f'{key}={int(value) + 1800000000000000000}')
            shifted.append(line)
        summary = make_trace(shifted, job)['metadata']['pipeline_summary']
        self.assertEqual(summary['ready_check_p50_us'], .005)


if __name__ == '__main__':
    unittest.main()
