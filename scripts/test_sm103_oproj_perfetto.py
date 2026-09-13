"""Synthetic OProj timeline tests; no cluster or GPU operations."""
import unittest
from export_sm103_oproj_perfetto import (make_trace, tile_consumption, cta_time_accounting,
                                       overlap_progress, norm_worker_metadata, norm_phase_metadata, cta_physical_end,
                                       profile_comm_sm)


class OprojTraceTests(unittest.TestCase):
    def test_single_candidate_list_overrides_default_comm_budget(self):
        self.assertEqual(profile_comm_sm(dict(comm_sm=8,comm_sm_list='48')),48)
        self.assertEqual(profile_comm_sm(dict(comm_sm=32,comm_sm_list=None)),32)
        with self.assertRaises(AssertionError):profile_comm_sm(dict(comm_sm=8,comm_sm_list='32,48'))

    def test_norm_phase_sums_are_bounded_and_old_captures_stay_unknown(self):
        self.assertEqual(norm_phase_metadata({},100),{})
        r=dict(norm_load_ns=50,norm_reduce_ns=10,norm_store_ns=30)
        m=norm_phase_metadata(r,100)
        self.assertEqual(m['other_row_work_us'],.010)
        self.assertEqual(m['inverse_rms_reduce_us'],.010)
        with self.assertRaises(AssertionError):norm_phase_metadata(r,89)
        with self.assertRaises(AssertionError):norm_phase_metadata(r|{'norm_load_ns':-1},100)

    def test_norm_worker_sums_do_not_become_fake_consecutive_spans(self):
        row = dict(start=100, end=200, norm_worker_begin=205, norm_worker_end=300,
                   norm_wait_ns=20, norm_work_ns=60, norm_rows=16, norm_threads=256)
        self.assertEqual(cta_physical_end(row), 300)
        attrs = norm_worker_metadata(row)
        self.assertEqual(attrs['ready_wait_us'], .020)
        self.assertEqual(attrs['row_work_us'], .060)
        self.assertEqual(attrs['claim_and_other_us'], .015)
        for changes in ({'norm_wait_ns': 36}, {'norm_rows': 17}, {'norm_threads': 64}):
            with self.assertRaises(AssertionError):
                norm_worker_metadata(row | changes)

    def test_assist_accounting_extends_compute_without_changing_gemm_end(self):
        ctas = {(0,0): dict(start=100,end=250),
                (0,1): dict(start=110,end=200,norm_worker_begin=205,norm_worker_end=300,
                           norm_wait_ns=20,norm_work_ns=60,norm_rows=16,norm_threads=256)}
        pipes = {(0,0):dict(cta=1,mma_begin=120,acc_wait_end=180)}
        stages = {(0,0,0):dict(wait_end=130)}
        result = cta_time_accounting(ctas,pipes,stages,1)[0]
        self.assertEqual(result['kernel_span_ns'],200)
        self.assertEqual(result['compute_end_ns'],100)
        self.assertEqual(result['sampled_ctas'][0]['closure_error_ns'],0)

    def test_overlap_counters_weight_tails_and_preserve_integer_clocks(self):
        offset = 1800000000000000000
        peers = {(0,0):dict(comm_valid=1,release=offset+10),
                 (0,1):dict(comm_valid=1,release=offset+30)}
        pipes = {(0,0):dict(first_input=offset+12,acc_wait_end=offset+20),
                 (0,1):dict(first_input=offset+31,acc_wait_end=offset+50)}
        events, summary = overlap_progress(0,offset,offset+60,peers,pipes,
            m=192,n=256,bm=128,bn=256,world=1)
        self.assertAlmostEqual(summary['gemm_completed_at_a_ready_pct'], 100*128/192)
        self.assertEqual(summary['gemm_tail_after_a_ready_us'], .020)
        series = {}
        for e in events: series.setdefault(e['name'],[]).append(e)
        self.assertEqual(len(series),3)
        for points in series.values():
            values=[e['args']['percent'] for e in points]
            self.assertEqual(values,sorted(values))
            self.assertEqual((values[0],values[-1]),(0.,100.))
        done=series['02 GEMM completed (observed %)']
        self.assertEqual(done[1]['ts'],.020)
        with self.assertRaisesRegex(AssertionError,'every GEMM tile'):
            overlap_progress(0,offset,offset+60,peers,{(0,0):pipes[0,0]},
                m=192,n=256,bm=128,bn=256,world=1)

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

    def test_postnorm_tail_is_not_counted_as_gemm_service(self):
        ctas = {(0, 0): dict(start=100, end=150, postnorm_ready=210, postnorm_end=260),
                (0, 1): dict(start=110, end=200, postnorm_ready=215, postnorm_end=270)}
        pipes = {(0, 0): dict(cta=1, mma_begin=115, acc_wait_end=180)}
        stages = {(0, 0, 0): dict(wait_end=120)}
        gpu, = cta_time_accounting(ctas, pipes, stages, 1)
        p, = gpu['sampled_ctas']
        self.assertEqual(gpu['last_role'], 'postnorm')
        self.assertEqual(gpu['compute_end_ns'], 100)
        self.assertEqual(gpu['kernel_span_ns'], 170)
        self.assertEqual(p['observed_service_sum_ns'], 60)
        self.assertEqual(p['postnorm_wait_ns'], 15)
        self.assertEqual(p['postnorm_service_ns'], 55)
        self.assertEqual(p['closure_error_ns'], 0)

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

    def test_epilogue_return_scope_is_explicit_and_legacy_is_not_relabelled(self):
        job,lines=self.fixture()
        self.assertEqual(make_trace(lines,job)['metadata']['epilogue_return_scope'],'legacy_unspecified')
        marker='profile_oproj_epilogue_scope,rank=0,return_scope=whole_collective_v2'
        self.assertEqual(make_trace(lines+[marker],job)['metadata']['epilogue_return_scope'],
                         'whole_collective_v2')
        for extra in ([marker,marker],[marker.replace('rank=0','rank=1')],
                      [marker.replace('whole_collective_v2','unknown')]):
            with self.assertRaises(AssertionError):make_trace(lines+extra,job)

    def test_mxfp8_keeps_roles_and_does_not_claim_bare_tma_latency(self):
        job, lines = self.fixture()
        job.update(mxfp8=True, fused_direction='oproj')
        lines = [line for line in lines if 'profile_phase=host_stages' not in line]
        result = make_trace(lines, job)
        names = {e['name'] for e in result['traceEvents']}
        self.assertIn('remote G2S + SFA repack + W progress (completion observed)', names)
        self.assertIn('local S2G + W progress (destination complete)', names)
        self.assertFalse(result['metadata']['performance_accepted'])
        with self.assertRaises(AssertionError):
            make_trace([line for line in lines if not line.startswith('route,')], job)

    def test_compact_presentation_keeps_exact_roles_and_handoffs(self):
        job, lines = self.fixture()
        full = make_trace(lines, job)
        compact = make_trace(lines, job, omit_comm_details=True)
        def handoffs(trace):
            return [e for e in trace['traceEvents'] if e['name'].startswith('release -> acquire')]
        self.assertEqual(handoffs(full), handoffs(compact))
        names = {e['name'] for e in compact['traceEvents']}
        self.assertIn('GEMM role (includes later peer waits)', names)
        self.assertIn('remote A2A role (includes setup/waits)', names)
        self.assertNotIn('remote G2S', names)
        self.assertNotIn('ready atomic (post-publication sample)', names)
        self.assertEqual(compact['metadata']['presentation'], 'roles_and_handoffs')
        with self.assertRaises(AssertionError):
            make_trace([line for line in lines if not line.startswith('profile_peer,')], job, True)

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

    def mxfp8_pipeline_fixture(self):
        job, lines = self.pipeline_fixture()
        job.update(mxfp8=True, fused_direction='oproj', max_swizzle_size=1, q_heads=1, head_dim=128)
        lines = [l for l in lines if 'profile_phase=host_stages' not in l]
        lines = [l.replace('tmem_acquired=112', 'tmem_acquired=135') for l in lines]
        for i, line in enumerate(lines):
            if line.startswith('profile_oproj_pipeline,'):
                lines[i] += (',mxfp8=1,stage_detail=1,first_input=130,first_wait_ns=15,'
                    'tmem_wait_begin=132,input_wait_ns=15,input_try_ns=1,initial_try_end=110,'
                    'scale_issue_ns=1,mma_issue_ns=4,load_wait_ns=1,load_try_ns=1,'
                    'load_issue_ns=2,load_stages=1,mma_stages=1')
            elif line.startswith('profile_oproj_stage,'):
                lines[i] += (',scale_end=132,try_begin=130,try_end=131,load_begin=126,'
                    'load_ready=127,load_try_begin=128,load_try_end=129,load_end=130')
        lines.append('profile_mxfp8_wait,rank=0,index=0,cta=1,warp=2,m=0,panel=0,begin=126,end=127')
        return job, lines

    def test_mxfp8_pipeline_keeps_compute_details_when_comm_hidden(self):
        job, lines = self.mxfp8_pipeline_fixture()
        trace = make_trace(lines, job, omit_comm_details=True)
        waits = trace['metadata']['gemm_wait_accounting']['per_cta']
        self.assertEqual(len(waits), 1)
        self.assertEqual(waits[0]['a_ready_wait_ns'], 5)
        self.assertEqual(waits[0]['w_ready_join_fence_ns'], 1)
        self.assertEqual(waits[0]['input_wait_ns']+waits[0]['input_try_ns'], 16)
        self.assertEqual(waits[0]['tmem_slot_wait_ns'], 3)  # Not 135-110!
        names = {e['name']:e for e in trace['traceEvents']}
        self.assertEqual(names['MMA next-input try_wait']['tid'], 100+32+2)
        self.assertEqual(names['Load empty-stage acquire']['tid'], 100+32+5)
        self.assertEqual(names['W ready wait / warp join / proxy fence']['tid'], 100+32+6)
        self.assertNotIn('weight_panel_waits', trace['metadata']['omitted_details'])
        self.assertEqual(trace['metadata']['presentation'], 'roles_handoffs_and_gemm_pipeline')

    def test_mxfp8_pipeline_rejects_missing_aggregate_and_bad_scale(self):
        job, lines = self.mxfp8_pipeline_fixture()
        for bad in ([l.replace('load_stages=1', 'load_stages=0') for l in lines],
                    [l.replace('scale_end=132', 'scale_end=129') for l in lines],
                    [l.replace('input_try_ns=1', 'input_try_ns=1000') for l in lines]):
            with self.assertRaises(AssertionError):
                make_trace(bad, job, True)

    def test_mxfp8_deferred_lookahead_keeps_nonoverlapping_accounting(self):
        for policy in (1, 2, 3):
            job, lines = self.mxfp8_pipeline_fixture()
            changed = []
            for line in lines:
                if line.startswith('profile_oproj_pipeline,'):
                    line += f',deferred_lookahead={policy}'
                    if policy & 1:
                        line = line.replace('scale_issue_ns=1', 'scale_issue_ns=2')
                    if policy & 2:
                        line = line.replace('load_issue_ns=2', 'load_issue_ns=3')
                if line.startswith('profile_oproj_stage,'):
                    if policy & 1:
                        line = line.replace('try_begin=130,try_end=131', 'try_begin=139,try_end=140')
                    if policy & 2:
                        line = line.replace('load_try_begin=128,load_try_end=129', 'load_try_begin=130,load_try_end=131')
                changed.append(line)
            trace = make_trace(changed, job, True)
            events = {e['name']: e for e in trace['traceEvents']}
            scale = events['Scale SMEM -> TMEM submission']
            self.assertAlmostEqual(scale['dur'], .002 if policy & 1 else .001)
            load = events['Load TMA A/B/SFA/SFB submission']
            self.assertAlmostEqual(load['dur'], .003 if policy & 2 else .001)
            # A policy label must agree with timestamps and scalar sums.
            with self.assertRaises(AssertionError):
                make_trace([l.replace(f'deferred_lookahead={policy}', 'deferred_lookahead=0') for l in changed], job, True)


if __name__ == '__main__':
    unittest.main()
