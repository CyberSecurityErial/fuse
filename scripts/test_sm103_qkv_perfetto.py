"""CPU export contracts; real GPU ordering is checked by the profiling harness."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from export_sm103_qkv_perfetto import export, group_role_tracks


class QkvPerfettoTests(unittest.TestCase):
    def test_details_follow_their_own_role(self):
        tids = [100, 101, 102] + list(range(1000, 1016))
        events = [dict(ph='M', name='thread_name', pid=0, tid=tid, args={}) for tid in tids]
        group_role_tracks(events, 2)
        ordered = sorted((e for e in events if e['name'] == 'thread_name'), key=lambda e:e['tid'])
        self.assertEqual([e['tid'] for e in ordered], list(range(100,119)))
        self.assertIn('QKV route role', ordered[0]['args']['name'])
        self.assertIn('CTA 000 / 8 warp 7', ordered[8]['args']['name'])
        self.assertIn('CTA 001 / 0 QKV route role', ordered[9]['args']['name'])
        self.assertIn('GEMM role', ordered[18]['args']['name'])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.output = self.run / 'trace.json'
        control = self.run / 'artifacts-attempt1/control'
        control.mkdir(parents=True)
        archive = b'fixture archive'
        (self.run / 'artifacts.tar.gz').write_bytes(archive)
        (self.run / 'fetched.json').write_text(json.dumps(dict(state='succeeded',
            exit_code=0, attempt=1, artifact_sha256=hashlib.sha256(archive).hexdigest())))
        (control / 'job.json').write_text(json.dumps(dict(profile=True, directions='qkv',
            mpi=False, world=1, comm_sm=1, global_seq=64, q_heads=1, kv_heads=1,
            qkv_policy_list='m128n256k64e32', profile_detail='full', run_id='fixture',
            source_id='fixture', node='09')))
        self.lines = ['PASS: and diagnostic timelines', 'candidate_verified,GEMM_A2A',
                      'profile_host,GEMM_A2A,rank=0']
        for cta in range(148):
            self.lines.append(f'profile_cta,GEMM_A2A,rank=0,cta={cta},start=100,end=1000,'
                'role_done=700,grid_sync_done=800,fence_done=810,publish_done=820,source_ready0=850')
        for task in range(11):
            drain = task >= 3
            row = dict(rank=0, task=task, drain=int(drain), cta=0,
                warp=task-3 if drain else task, begin=500 if drain else 200,
                ready=210, g2s_begin=220, g2s_done=230, s2g_begin=240,
                s2g_read_done=550 if drain else 250, row=0, column=task*128,
                rows=64, columns=128, peer=0, segment=task if not drain else 0)
            self.lines.append('profile_qkv_route,' + ','.join(f'{k}={v}' for k,v in row.items()))
        self.log = control / 'attempt1.log'

    def render(self):
        self.log.write_text('\n'.join(self.lines))
        with contextlib.redirect_stdout(io.StringIO()):
            export(self.run, self.output)

    def test_complete_tiles_and_drains(self):
        self.render()
        trace = json.loads(self.output.read_text())
        self.assertEqual(trace['metadata']['route_records_per_rank'], [11])
        waits = [e for e in trace['traceEvents'] if e['name'] == 'ready wait']
        self.assertEqual([e['args']['segment'] for e in waits], list('QKV'))
        self.assertEqual(waits[2]['args']['producer_n_first'], 1)
        drains = [e for e in trace['traceEvents'] if e['name'] == 'all peer writes drain']
        self.assertEqual(len(drains), 8)
        self.assertTrue(all(e.get('dur', 0) >= 0 for e in trace['traceEvents']))

    def test_missing_drain_rejected_without_partial_delivery(self):
        self.lines.pop()
        with self.assertRaises(AssertionError): self.render()
        self.assertFalse(self.output.exists())

    def test_producer_order_with_skipped_candidate_slots(self):
        self.lines.insert(0, 'profile_qkv_order,rank=0,version=producer_ready_v1,slots=8,copies=3,comm_sm=1')
        for i in range(len(self.lines)):
            if not self.lines[i].startswith('profile_qkv_route,'): continue
            row = dict(part.split('=',1) for part in self.lines[i].split(',')[1:])
            old = int(row['task'])
            row['task'] = str([1,4,6][old] if old < 3 else 8+old-3)
            row['warp'] = str(int(row['task']) % 8)
            self.lines[i] = 'profile_qkv_route,' + ','.join(f'{k}={v}' for k,v in row.items())
        self.render()
        trace=json.loads(self.output.read_text())
        self.assertEqual(trace['metadata']['route_records_per_rank'],[11])
        self.assertEqual(trace['metadata']['route_order']['0']['slots'],'8')

    def test_unordered_phase_rejected(self):
        self.lines[-11] = self.lines[-11].replace('ready=210', 'ready=260')
        with self.assertRaises(AssertionError): self.render()
        self.assertFalse(self.output.exists())

    def test_existing_delivery_preserved(self):
        self.output.write_text('existing')
        with self.assertRaises(FileExistsError): self.render()
        self.assertEqual(self.output.read_text(), 'existing')


if __name__ == '__main__':
    unittest.main()
