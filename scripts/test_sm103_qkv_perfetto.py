"""CPU export contracts; real GPU ordering is checked by the profiling harness."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from export_sm103_qkv_perfetto import (
    export, group_role_tracks, annotate_trace, annotate_transfer_events,
    append_mxfp8_events, MXFP8_PUBLICATION_PHASES, MXFP8_NO_SC_PUBLICATION_PHASES,
    resolve_route_schedule,
)


class QkvPerfettoTests(unittest.TestCase):
    def mxfp8_fixture(self, phases=None):
        job = dict(world=1, comm_sm=1, global_seq=128, q_heads=1, kv_heads=1, head_dim=128, hidden=128)
        records = {(0,c):dict(start=100,role_done=1000) for c in range(148)}
        lines = []
        for index in range(48):
            row = dict(rank=0,index=index,cta=0,warp=index%8,panel=index//32,groups=32,
                begin=200,quant_done=250,end=270,release=260 if index in (31,47) else 0)
            if phases is not None: row.update(phases)
            lines.append('profile_mxfp8_quant,'+','.join(f'{k}={v}' for k,v in row.items()))
        lines.append('profile_mxfp8_wait,rank=0,index=0,cta=1,warp=3,m=0,panel=0,begin=150,end=300')
        self.log.write_text('\n'.join(lines))
        return job, records, lines

    def test_mxfp8_quantization_coverage_and_role_layout(self):
        job, records, lines = self.mxfp8_fixture()
        events = []
        result = append_mxfp8_events(events,self.log,job,{0:100},records)
        self.assertEqual(result['quant_chunks'],48)
        self.assertEqual(result['panel_releases'],2)
        self.assertEqual(result['publication_subphase_chunks'],0)
        self.assertEqual(result['publication_protocol_chunks'],{'legacy_unsplit':48})
        self.assertTrue(any(e['name']=='W quantize BF16 -> MXFP8' for e in events))
        self.assertEqual(sum(e['name']=='W fence + arrival counter + warp join' for e in events),48)
        self.assertFalse(any(e['name'] in dict(MXFP8_PUBLICATION_PHASES).values() for e in events))
        names = [e for e in events if e['ph']=='M' and e['name']=='thread_name']
        self.assertTrue(all(100<e['tid']<132 for e in names if 'quantization' in e['args']['name']))
        self.log.write_text('\n'.join(lines[1:]))
        with self.assertRaises(AssertionError):
            append_mxfp8_events([],self.log,job,{0:100},records)

    def assert_mxfp8_subspans(self, events, parent_name, phases, protocol):
        parents = [e for e in events if e['name']==parent_name]
        self.assertEqual(len(parents),48)
        phase_names = [name for _,name in phases]
        for parent in parents:
            self.assertEqual(parent['args']['publication_protocol'],protocol)
            children = [e for e in events if e['name'] in phase_names and e['args']==parent['args']]
            self.assertEqual([e['name'] for e in children],phase_names)
            self.assertTrue(all((e['pid'],e['tid'])==(parent['pid'],parent['tid']) for e in children))
            self.assertAlmostEqual(children[0]['ts'],parent['ts'])
            self.assertAlmostEqual(sum(e['dur'] for e in children),parent['dur'])
            for left,right in zip(children,children[1:]):
                self.assertAlmostEqual(left['ts']+left['dur'],right['ts'])
            self.assertAlmostEqual(children[-1]['ts']+children[-1]['dur'],parent['ts']+parent['dur'])

    def test_oproj_reuses_quant_protocol_with_oproj_weight_geometry(self):
        job, records, _ = self.mxfp8_fixture()
        # Same 384x128 W as the fixture, now described by OProj H x HqD.
        # KV heads do not contribute to OProj's weight extent.
        job.update(fused_direction='oproj', hidden=384, q_heads=1, kv_heads=9)
        result = append_mxfp8_events([], self.log, job, {0:100}, records)
        self.assertEqual(result['quant_chunks'], 48)
        self.assertEqual(result['panel_releases'], 2)

    def test_mxfp8_publication_subspans_cover_parent_on_same_track(self):
        job, records, _ = self.mxfp8_fixture(dict(fence_done=253,warp_join_done=255,arrival_done=258))
        events = []
        result = append_mxfp8_events(events,self.log,job,{0:100},records)
        self.assertEqual(result['publication_subphase_chunks'],48)
        self.assertEqual(result['publication_protocol_chunks'],{'sc_fence_acq_rel_v1':48})
        self.assert_mxfp8_subspans(events,'W fence + arrival counter + warp join',
                                  MXFP8_PUBLICATION_PHASES,'sc_fence_acq_rel_v1')

    def test_mxfp8_no_sc_publication_has_three_real_subspans(self):
        job, records, _ = self.mxfp8_fixture(dict(warp_join_done=255,arrival_done=258))
        events = []
        result = append_mxfp8_events(events,self.log,job,{0:100},records)
        self.assertEqual(result['publication_subphase_chunks'],48)
        self.assertEqual(result['publication_protocol_chunks'],{'warp_join_acq_rel_v2':48})
        self.assert_mxfp8_subspans(events,'W arrival counter + warp join',
                                  MXFP8_NO_SC_PUBLICATION_PHASES,'warp_join_acq_rel_v2')
        self.assertFalse(any(e['name'] in ('W device fence','W fence + arrival counter + warp join')
                             for e in events))

    def aggregate_mxfp8_fixture(self, hidden=128, comm=1):
        job, records, _ = self.mxfp8_fixture()
        job.update(hidden=hidden, comm_sm=comm)
        steps, workers = hidden // 4, comm * 8
        rows, owners = [], {}
        for index in range(steps + steps // 2):
            owner, panel = index % workers, index // steps
            row = dict(rank=0,index=index,cta=owner%comm,warp=owner//comm,
                panel=panel,groups=32,begin=200,quant_done=250,end=252,release=0,
                arrival_chunks=0,warp_join_done=0,arrival_done=0)
            rows.append(row)
            owners.setdefault((panel,owner),[]).append(row)
        for owned in owners.values():
            owned[-1].update(arrival_chunks=len(owned),warp_join_done=255,
                             arrival_done=258,end=270)
        for index in (steps-1,len(rows)-1):
            rows[index]['release'] = 260
        self.write_aggregate_mxfp8(rows, comm)
        return job, records, rows

    def write_aggregate_mxfp8(self, rows, comm=1):
        lines = ['precision,mxfp8,publication_protocol=warp_panel_acq_rel_v3']
        lines.extend('profile_mxfp8_quant,'+','.join(f'{k}={v}' for k,v in row.items())
                     for row in rows)
        lines.append(f'profile_mxfp8_wait,rank=0,index=0,cta={comm},warp=3,m=0,panel=0,begin=150,end=300')
        self.log.write_text('\n'.join(lines))

    def test_mxfp8_aggregate_only_actual_publications_have_atomic_spans(self):
        for hidden,comm,contributions in ((128,1,16),(384,1,16),(128,16,48)):
            with self.subTest(hidden=hidden,comm=comm):
                job,records,rows = self.aggregate_mxfp8_fixture(hidden,comm)
                events = []
                result = append_mxfp8_events(events,self.log,job,{0:100},records)
                self.assertEqual(result['quant_chunks'],len(rows))
                self.assertEqual(result['panel_releases'],2)
                self.assertEqual(result['aggregate_publications'],contributions)
                self.assertEqual(result['publication_subphase_chunks'],contributions)
                self.assertEqual(result['publication_protocol_chunks'],{'warp_panel_acq_rel_v3':len(rows)})
                self.assertEqual(sum(e['name']=='W local bookkeeping (no publication)' for e in events),
                                 len(rows)-contributions)
                for _,name in MXFP8_NO_SC_PUBLICATION_PHASES:
                    spans = [e for e in events if e['name']==name]
                    self.assertEqual(len(spans),contributions)
                    self.assertTrue(all(e['args']['arrival_chunks']>0 for e in spans))
                self.assertFalse(any(e['name']=='W device fence' for e in events))

    def test_mxfp8_aggregate_nonpublication_cannot_forge_atomic_or_ready(self):
        for field in ('warp_join_done','arrival_done','release'):
            with self.subTest(field=field):
                job,records,rows = self.aggregate_mxfp8_fixture()
                rows[0][field] = 251
                self.write_aggregate_mxfp8(rows)
                with self.assertRaisesRegex(AssertionError,'Non-contributing chunk'):
                    append_mxfp8_events([],self.log,job,{0:100},records)

    def test_mxfp8_aggregate_requires_final_owned_chunk_and_exact_delta(self):
        for change,reason in (('early','final chunk'),('count','valid chunks'),
                              ('duplicate','Duplicate worker-panel'),('missing','final chunk')):
            with self.subTest(change=change):
                job,records,rows = self.aggregate_mxfp8_fixture()
                last,first = rows[24],rows[0]  # Warp 0 owns panel-0 chunks 0,8,16,24.
                if change=='count':
                    last['arrival_chunks'] += 1
                else:
                    if change in ('early','duplicate'):
                        first.update(arrival_chunks=4,warp_join_done=255,arrival_done=258,end=270)
                    if change in ('early','missing'):
                        last.update(arrival_chunks=0,warp_join_done=0,arrival_done=0,end=252)
                self.write_aggregate_mxfp8(rows)
                with self.assertRaisesRegex(AssertionError,reason):
                    append_mxfp8_events([],self.log,job,{0:100},records)

    def test_mxfp8_aggregate_rejects_missing_fields_or_mixed_protocol(self):
        for field in ('warp_join_done','arrival_done','arrival_chunks'):
            with self.subTest(field=field):
                job,records,rows = self.aggregate_mxfp8_fixture()
                del rows[24][field]
                self.write_aggregate_mxfp8(rows)
                with self.assertRaises(AssertionError):
                    append_mxfp8_events([],self.log,job,{0:100},records)

    def test_mxfp8_aggregate_memory_estimate_uses_72_byte_records(self):
        from l20d import fused_device_memory
        job = dict(world=4,seq_local=128,hidden=128,q_heads=4,kv_heads=4,
                   head_dim=128,profile=True,directions='qkv')
        plain = fused_device_memory(job)
        mxfp8 = fused_device_memory(job | {'mxfp8':True})
        self.assertEqual(mxfp8['profile_bytes']-plain['profile_bytes'],6*32*72+6*24)

    def test_mxfp8_unordered_publication_subspans_rejected(self):
        for phases in (dict(fence_done=249,warp_join_done=255,arrival_done=258),
                       dict(fence_done=256,warp_join_done=255,arrival_done=258),
                       dict(fence_done=253,warp_join_done=259,arrival_done=258),
                       dict(fence_done=253,warp_join_done=255,arrival_done=271),
                       dict(warp_join_done=249,arrival_done=258),
                       dict(warp_join_done=259,arrival_done=258),
                       dict(warp_join_done=255,arrival_done=271)):
            with self.subTest(phases=phases):
                job, records, _ = self.mxfp8_fixture(phases)
                with self.assertRaisesRegex(AssertionError,'Unordered W publication'):
                    append_mxfp8_events([],self.log,job,{0:100},records)

    def test_mxfp8_partial_publication_timestamps_rejected(self):
        for phases in (dict(fence_done=253), dict(warp_join_done=255), dict(arrival_done=258),
                       dict(fence_done=253,warp_join_done=255), dict(fence_done=253,arrival_done=258)):
            with self.subTest(phases=phases):
                job, records, _ = self.mxfp8_fixture(phases)
                with self.assertRaisesRegex(AssertionError,'Incomplete W publication'):
                    append_mxfp8_events([],self.log,job,{0:100},records)

    def test_mxfp8_ready_stamp_cannot_precede_arrival_in_either_protocol(self):
        for phases in (dict(fence_done=253,warp_join_done=255,arrival_done=261),
                       dict(warp_join_done=255,arrival_done=261)):
            with self.subTest(phases=phases):
                job, records, _ = self.mxfp8_fixture(phases)
                with self.assertRaisesRegex(AssertionError,'W ready stamp precedes arrival atomic'):
                    append_mxfp8_events([],self.log,job,{0:100},records)

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

    def comm_warp_fixture(self, handoff=False):
        job_path = self.log.parent / 'job.json'
        job = json.loads(job_path.read_text())
        job.update(mxfp8=True, mxfp8_weight_preparation='comm_warp', comm_sm=2,
                   global_seq=256, hidden=128, head_dim=128)
        job_path.write_text(json.dumps(job))
        self.lines = [line for line in self.lines if not line.startswith('profile_qkv_route,')]
        route_warps = 8 if handoff else 4
        workers = 2 * route_warps
        self.lines.append('profile_qkv_order,rank=0,version=producer_ready_v1,slots=16,'
                          f'copies=12,comm_sm=2,route_warps={route_warps}' +
                          (',weight_schedule=warp_then_route_v1' if handoff else ''))
        # Two waves with padding holes: copy owner uses task % (2 CTAs * 4 warps),
        # while the eight drains start at slots, not the twelve emitted copies.
        tasks = [0,1,2,4,5,6,8,9,10,12,13,14]
        for index, task in enumerate(tasks + list(range(16,16+workers))):
            drain = task >= 16
            owner = (task - 16 if drain else task) % workers
            begin = 500 if drain else 200 + task // 8 * 100
            row = dict(rank=0, task=task, drain=int(drain), cta=owner%2, warp=owner//2,
                       begin=begin, ready=begin+10, g2s_begin=begin+20, g2s_done=begin+30,
                       s2g_begin=begin+40, s2g_read_done=begin+50, row=index//3*64,
                       column=index%3*128, rows=64, columns=128, peer=0, segment=index%3)
            self.lines.append('profile_qkv_route,' + ','.join(f'{k}={v}' for k,v in row.items()))
        _, _, quant_lines = self.mxfp8_fixture(dict(warp_join_done=255,arrival_done=258))
        for line in quant_lines:
            row = dict(part.split('=',1) for part in line.split(',')[1:])
            if line.startswith('profile_mxfp8_quant,'):
                index = int(row['index'])
                row.update(cta=index%2, warp=4+(index//2)%4)
            else:
                row['cta'] = 2
            self.lines.append(line.split(',',1)[0] + ',' + ','.join(f'{k}={v}' for k,v in row.items()))

    def test_comm_warp_complete_copies_drains_and_physical_tracks(self):
        self.comm_warp_fixture()
        self.render()
        trace = json.loads(self.output.read_text())
        self.assertEqual(trace['metadata']['route_warps'],4)
        self.assertEqual(trace['metadata']['weight_schedule'],'fixed_split_legacy_v1')
        self.assertEqual(trace['metadata']['route_records_per_rank'],[20])
        self.assertEqual(trace['metadata']['mxfp8']['quant_chunks'],48)
        self.assertEqual(sum(e['name']=='all peer writes drain' for e in trace['traceEvents']),8)
        names = [e['args']['name'] for e in trace['traceEvents']
                 if e['ph']=='M' and e['name']=='thread_name']
        routes = [name for name in names if 'ready | G2S | S2G | drain' in name]
        self.assertEqual(len(routes),8)
        self.assertTrue(all(any(f'warp {w}:' in name for w in range(4)) for name in routes))
        quant = [name for name in names if 'weight quantization + publication' in name]
        self.assertEqual(len(quant),8)
        self.assertTrue(all(any(f'warp {w}:' in name for w in range(4,8)) for name in quant))
        self.assertTrue(any('MXFP8 route + quant role' in name for name in names))

    def test_comm_warp_missing_drain_rejected(self):
        self.comm_warp_fixture()
        self.lines = [line for line in self.lines if not line.startswith('profile_qkv_route,rank=0,task=23,')]
        with self.assertRaisesRegex(AssertionError,'Incomplete route records'):
            self.render()
        self.assertFalse(self.output.exists())

    def test_comm_warp_wrong_quant_warp_rejected(self):
        self.comm_warp_fixture()
        self.lines = [line.replace('warp=4,','warp=0,') if line.startswith('profile_mxfp8_quant,')
                      else line for line in self.lines]
        with self.assertRaisesRegex(AssertionError,'outside dedicated quant warps'):
            self.render()
        self.assertFalse(self.output.exists())

    def test_route_warp_metadata_fallback_and_consistency(self):
        job = dict(world=1,comm_sm=2,mxfp8=True)
        legacy = {0:dict(comm_sm='2')}
        split = {0:dict(comm_sm='2',route_warps='4')}
        self.assertEqual(resolve_route_schedule(job,legacy),(8,None))
        self.assertEqual(resolve_route_schedule(job,split),(4,'fixed_split_legacy_v1'))
        job['mxfp8_weight_preparation'] = 'comm_warp'
        with self.assertRaisesRegex(AssertionError,'Missing comm_warp route schedule metadata'):
            resolve_route_schedule(job,{})
        self.assertEqual(resolve_route_schedule(job,split),(4,'fixed_split_legacy_v1'))
        with self.assertRaisesRegex(AssertionError,'preparation/route_warps mismatch'):
            resolve_route_schedule(job,legacy)
        handoff = {0:dict(comm_sm='2',route_warps='8',weight_schedule='warp_then_route_v1')}
        self.assertEqual(resolve_route_schedule(job,handoff),(8,'warp_then_route_v1'))
        with self.assertRaisesRegex(AssertionError,'Invalid quant-to-route handoff metadata'):
            resolve_route_schedule(job,{0:dict(handoff[0],route_warps='4')})
        job['world'] = 2
        with self.assertRaisesRegex(AssertionError,'Inconsistent rank route_warps'):
            resolve_route_schedule(job,{0:split[0],1:legacy[0]})
        with self.assertRaisesRegex(AssertionError,'Inconsistent rank weight_schedule'):
            resolve_route_schedule(job,{0:handoff[0],1:legacy[0]})

    def test_quant_to_route_handoff_uses_all_eight_route_warps(self):
        self.comm_warp_fixture(handoff=True)
        self.render()
        trace = json.loads(self.output.read_text())
        self.assertEqual(trace['metadata']['route_warps'],8)
        self.assertEqual(trace['metadata']['weight_schedule'],'warp_then_route_v1')
        self.assertTrue(trace['metadata']['weight_handoff_validated'])
        self.assertEqual(trace['metadata']['route_records_per_rank'],[28])
        self.assertEqual(sum(e['name']=='all peer writes drain' for e in trace['traceEvents']),16)
        names = [e['args']['name'] for e in trace['traceEvents']
                 if e['ph']=='M' and e['name']=='thread_name']
        self.assertEqual(sum('ready | G2S | S2G | drain' in name for name in names),16)
        self.assertEqual(sum('weight quantization + publication' in name for name in names),8)
        self.assertTrue(any('MXFP8 route + quant role' in name for name in names))

    def test_quant_to_route_handoff_requires_all_eight_drains(self):
        self.comm_warp_fixture(handoff=True)
        self.lines = [line for line in self.lines if not line.startswith('profile_qkv_route,rank=0,task=31,')]
        with self.assertRaisesRegex(AssertionError,'Incomplete route records'):
            self.render()
        self.assertFalse(self.output.exists())

    def test_quant_to_route_handoff_rejects_route_before_last_quant(self):
        self.comm_warp_fixture(handoff=True)
        self.lines = [line.replace('begin=300,','begin=260,')
                      if line.startswith('profile_qkv_route,rank=0,task=8,') else line
                      for line in self.lines]
        with self.assertRaisesRegex(AssertionError,'Route precedes quantization handoff'):
            self.render()
        self.assertFalse(self.output.exists())

    def test_quant_to_route_handoff_rejects_later_return_to_quantization(self):
        self.comm_warp_fixture(handoff=True)
        for index, line in enumerate(self.lines):
            if line.startswith('profile_mxfp8_quant,rank=0,index=0,'):
                self.lines[index] = line.replace('begin=200,','begin=400,').replace(
                    'quant_done=250,','quant_done=410,').replace('end=270,','end=430,').replace(
                    'warp_join_done=255,','warp_join_done=415,').replace('arrival_done=258','arrival_done=418')
        with self.assertRaisesRegex(AssertionError,'Route precedes quantization handoff'):
            self.render()
        self.assertFalse(self.output.exists())

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
        copies = [e for e in trace['traceEvents'] if e['name'] in
                  ('local G2S', 'peer S2G (SMEM read complete)')]
        self.assertEqual(len(copies), 6)
        self.assertTrue(all(e['args']['bytes'] == 16384 for e in copies))
        self.assertTrue(all(e['args']['src_gpu'] == e['args']['dst_gpu'] == 0 for e in copies))

    def test_default_detail_uses_resolved_cta_records(self):
        job_path = self.log.parent / 'job.json'
        job = json.loads(job_path.read_text())
        job['profile_detail'] = None
        job_path.write_text(json.dumps(job))
        self.lines = [line + ',profile_detail=full' if line.startswith('profile_cta,')
                      else line for line in self.lines]
        self.render()
        trace = json.loads(self.output.read_text())
        self.assertIsNone(trace['metadata']['profile_detail_requested'])
        self.assertEqual(trace['metadata']['profile_detail'], 'full')
        self.assertEqual(trace['metadata']['route_records_per_rank'], [11])
        self.assertEqual(sum(e['name'] == 'thread_name' and 'ready | G2S | S2G | drain' in e['args']['name']
                             for e in trace['traceEvents'] if e['ph'] == 'M'), 8)

    def test_conflicting_resolved_detail_rejected(self):
        self.lines = [line + ',profile_detail=cta' if line.startswith('profile_cta,')
                      else line for line in self.lines]
        with self.assertRaisesRegex(AssertionError, 'profile detail mismatch'):
            self.render()
        self.assertFalse(self.output.exists())

    def test_inconsistent_cta_detail_rejected(self):
        self.lines[3] += ',profile_detail=full'
        with self.assertRaisesRegex(AssertionError, 'Inconsistent CTA profile detail'):
            self.render()
        self.assertFalse(self.output.exists())

    def test_endpoint_annotation_preserves_sample_and_tracks(self):
        events = [dict(ph='X', pid=2, tid=109, ts=i, dur=0.25, name=name, args=args)
                  for i, (name, args) in enumerate([
                      ('ready wait', dict(task=7, peer=5, rows=64, columns=128)),
                      ('local G2S', dict(task=7)),
                      ('peer S2G (SMEM read complete)', dict(task=7))])]
        original = [{k:v for k,v in e.items() if k != 'args'} for e in events]
        self.assertEqual(annotate_transfer_events(events, 8), 2)
        self.assertEqual(events[1]['args']['dst_gpu'], 2)
        self.assertEqual(events[2]['args']['src_gpu'], 2)
        self.assertEqual(events[2]['args']['dst_gpu'], 5)
        self.assertEqual(events[1]['args']['route_peer'], 5)
        self.assertEqual(events[2]['args']['bytes'], 16384)
        self.assertEqual(annotate_transfer_events(events, 8), 2)
        self.assertEqual(original, [{k:v for k,v in e.items() if k != 'args'} for e in events])

    def test_invalid_annotation_preserves_existing_file(self):
        payload = dict(metadata=dict(schema='fuse_sm103_qkv_perfetto_v2', config=dict(world=8)),
                       traceEvents=[dict(name='ready wait', pid=0,
                           args=dict(task=1, peer=7, rows=64, columns=128))])
        original = json.dumps(payload)
        self.output.write_text(original)
        with self.assertRaises(AssertionError): annotate_trace(self.output)
        self.assertEqual(self.output.read_text(), original)
        self.assertFalse(self.output.with_suffix('.endpoints-tmp').exists())

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
