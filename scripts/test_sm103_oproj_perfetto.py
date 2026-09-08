"""Synthetic OProj timeline tests; no cluster or GPU operations."""
import unittest
from export_sm103_oproj_perfetto import make_trace


class OprojTraceTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
