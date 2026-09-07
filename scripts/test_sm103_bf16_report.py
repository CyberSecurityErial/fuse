"""CPU-only contracts for explicit full-history report assembly and raw readers."""
import argparse
import copy
import csv
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

import report_sm103_bf16 as report


def stable(samples=None, pure=True):
    samples = samples or [2.] * 50
    drift = abs(report.statistics.median(samples[:25]) - report.statistics.median(samples[25:])) / report.statistics.median(samples)
    return dict(initial_warmup=10, iterations=50, relative_range_limit=.05,
        minimum_warmup_cuda_ms=100, additional_warmup_cuda_ms=120,
        window_ms_per_call=[4., 4., 4.], warmup_converged=True, converged_all_ranks=True,
        collector='single_gpu_primed_events_v2' if pure else 'primed_events_vector_max_v1',
        measurement_rounds=[dict(samples_ms=samples, half_p50_relative_drift=drift,
                                sample_cadence_warmup_ms=[2.] * 10)],
        selected_round=0, sample_half_p50_relative_drift=drift)


def history():
    return list(report.bench.cases(argparse.Namespace(directions='qkv,oproj', models='',
        seqs=(1024, 4096, 16384, 131072, 262144, 524288), cps=(4, 8), devices='0,1,2,3,4,5,6,7')))


def measurements(cases, placements):
    rows, seen, pure_seen = [], set(), set()
    for case in cases:
        physical = report.physical_key(case)
        if physical in seen:
            continue
        seen.add(physical)
        node = placements[physical]
        for launch in report.LAUNCHES:
            for kind, p50 in [('fused', 2.), ('cublaslt_nccl', 3.), ('te_ub', 4.)]:
                rows.append(dict(kind=kind, node=node, case=case, launch=launch,
                    layout=report.LAYOUTS[case['direction']], p50_ms=p50, p95_ms=p50 + 1,
                    devices=','.join(str(i) for i in range(case['cp'])),
                    physical_devices=[str(i) for i in range(case['cp'])],
                    gpu_uuids=['GPU-' + str(i) for i in range(case['cp'])], samples_ms=[p50] * 50,
                    config=dict(tile='m128n128', comm_ctas=8),
                    provenance=dict(run_id=kind + '-' + launch, source_id='a' * 64,
                                    raw=dict(path='/raw', sha256='b' * 64))))
            key = (node, tuple(case[k] for k in ('m', 'n', 'k')), launch)
            if key not in pure_seen:
                pure_seen.add(key)
                rows.append(dict(kind='pure', node=node, mnk=key[1], launch=launch, p50_ms=1., p95_ms=1.5,
                    provenance=dict(run_id='pure-' + launch, source_id='c' * 64)))
    return rows


class ReportContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='fuse-report-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # A reporting regression must never turn into an external command.
        self.patches = [mock.patch('subprocess.run', side_effect=AssertionError('external command')),
                        mock.patch('subprocess.Popen', side_effect=AssertionError('external command'))]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_stability_recomputes_and_preserves_first_rejected_round(self):
        record = stable()
        noisy = [1.] * 25 + [2.] * 25
        record['measurement_rounds'].insert(0, dict(samples_ms=noisy, half_p50_relative_drift=2/3,
                                                  sample_cadence_warmup_ms=[2.] * 10))
        record['selected_round'] = 1
        result = report.audit_stability(record, [2.] * 50, pure=True)
        self.assertEqual(result['rejected_rounds'], 1)
        record['measurement_rounds'][0]['samples_ms'] = [1.] * 50
        record['measurement_rounds'][0]['half_p50_relative_drift'] = 0.
        with self.assertRaisesRegex(ValueError, 'first stable'):
            report.audit_stability(record, [2.] * 50, pure=True)

    def test_stability_rejects_forged_drift_warmup_and_samples(self):
        mutations = [dict(initial_warmup=2), dict(iterations=6), dict(relative_range_limit=.1),
                     dict(additional_warmup_cuda_ms=99), dict(window_ms_per_call=[1., 2., 3.]),
                     dict(selected_round=1), dict(sample_half_p50_relative_drift=.01),
                     dict(warmup_converged=False), dict(collector='other')]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                report.audit_stability(stable() | mutation, [2.] * 50, pure=True)
        with self.assertRaises(ValueError):
            report.samples_stats([0.] * 50)
        with self.assertRaises(ValueError):
            report.samples_stats([float('nan')] * 50)

    def test_zero_budget_only_for_baseline_reused_process(self):
        self.assertEqual(report.audit_stability(stable(pure=False) | dict(minimum_warmup_cuda_ms=0),
                                               [2.] * 50)['selected_round'], 0)
        with self.assertRaises(ValueError):
            report.audit_stability(stable() | dict(minimum_warmup_cuda_ms=0), [2.] * 50, pure=True)

    def test_full_matrix_counts_aliases_and_units(self):
        cases = history()
        placements = {report.physical_key(c): '09' if c['cp'] == 4 else '0a' for c in cases}
        self.assertEqual(len(cases), 192)
        self.assertEqual(len(placements), 168)
        table, missing, _ = report.join_measurements(cases, placements, measurements(cases, placements))
        self.assertFalse(missing)
        self.assertEqual(len(table), 384)
        cell = table[0]
        self.assertEqual(cell['ratios'], dict(te_ub_x=2., strong_x=1.5, pure_percent_diagnostic=50.))
        self.assertEqual(cell['pflops_per_gpu']['fused'],
                         2 * report.math.prod(cell['case'][k] for k in ('m','n','k')) / 2 / 1e12)
        self.assertNotEqual(cell['ratios']['te_ub_x'], cell['ratios']['strong_x'])

    def test_relocated_geometry_never_pairs_original_node(self):
        cases = history()
        placements = {report.physical_key(c): '09' if c['cp'] == 4 else '0a' for c in cases}
        rows = measurements(cases, placements)
        target = next(c for c in cases if c['cp'] == 4 and c['seq'] == 524288 and c['hidden'] == 16384)
        placements[report.physical_key(target)] = '0a'
        _, missing, excluded = report.join_measurements(cases, placements, rows)
        self.assertTrue(excluded)
        self.assertTrue(any(x['case'] == report.logical_id(target) and x['node'] == '0a' for x in missing))

    def test_mnk_equality_does_not_alias_fused_routes(self):
        a = dict(direction='qkv', seq=1024, cp=8, hidden=2048, q_heads=16, kv_heads=8, head_dim=128)
        b = a | dict(q_heads=24, kv_heads=4)
        self.assertEqual(a['q_heads'] + 2*a['kv_heads'], b['q_heads'] + 2*b['kv_heads'])
        self.assertNotEqual(report.physical_key(a), report.physical_key(b))

    def test_ambiguous_versions_and_graph_relabel_fail(self):
        cases = history()[:1]
        placement = {report.physical_key(cases[0]): '09'}
        rows = measurements(cases, placement)
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            report.join_measurements(cases, placement, rows + [copy.deepcopy(rows[0])])
        rows = [r for r in rows if r['kind'] != 'fused' or r['launch'] != 'graph']
        _, missing, _ = report.join_measurements(cases, placement, rows)
        self.assertEqual(missing[0]['launch'], 'graph')
        self.assertEqual(missing[0]['missing'], ['fused'])

    def test_layout_and_physical_gpu_group_rejected(self):
        case = next(c for c in history() if c['direction'] == 'oproj')
        placement = {report.physical_key(case): '09'}
        for field, value in [('layout', 'legacy'), ('physical_devices', ['1','2','3','4'])]:
            rows = measurements([case], placement)
            rows[1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                report.join_measurements([case], placement, rows)

    def test_missing_writes_only_internal_evidence_and_no_overwrite(self):
        target = self.root / 'missing'
        report.write_report(target, dict(complete=False, missing=[dict(case='missing')]), [])
        self.assertEqual({p.name for p in target.iterdir()}, {'evidence.json'})
        self.assertNotIn('aggregates',json.loads((target/'evidence.json').read_text()))
        with self.assertRaises(ValueError):
            report.write_report(target, dict(complete=False), [])

    def test_full_output_has_192_markdown_rows_and_384_csv_cells(self):
        cases = history()
        placement = {report.physical_key(c): '09' if c['cp'] == 4 else '0a' for c in cases}
        table, _, _ = report.join_measurements(cases, placement, measurements(cases, placement))
        target = self.root / 'full'
        report.write_report(target, dict(complete=True), table)
        self.assertEqual({p.name for p in target.iterdir()}, {'evidence.json', 'table.md', 'table.csv'})
        with (target / 'table.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 384)
        markdown = (target / 'table.md').read_text()
        self.assertEqual(sum(line.startswith(('| 4 |', '| 8 |')) for line in markdown.splitlines()), 192)
        self.assertIn('TE Userbuffers', markdown)
        self.assertIn('单卡', markdown)
        aggregates=json.loads((target/'evidence.json').read_text())['aggregates']
        self.assertEqual(aggregates['launches']['eager']['all']['overall']['logical_count'],192)
        self.assertEqual(aggregates['launches']['graph']['long']['overall']['logical_count'],96)
        self.assertIn('几何平均',markdown)

    def test_aggregate_scopes_launches_and_direction_thresholds(self):
        cases=history()
        placement={report.physical_key(c):'09' if c['cp']==4 else '0a' for c in cases}
        rows=measurements(cases,placement)
        for row in rows:
            if row['kind']=='te_ub' and row['case']['direction']=='qkv':
                row['p50_ms']=8. if row['launch']=='eager' else 16.
        table,_,_=report.join_measurements(cases,placement,rows)
        result=report.aggregate_ratios(table)
        self.assertFalse(result['samples_pooled'])
        self.assertFalse(result['goal_achievement_evaluated'])
        for launch,qkv_gm in [('eager',4.),('graph',8.)]:
            for scope,count,physical in [('all',192,168),('long',96,84)]:
                groups=result['launches'][launch][scope]
                self.assertEqual(groups['overall']['logical_count'],count)
                self.assertEqual(groups['overall']['physical_count'],physical)
                self.assertEqual(groups['qkv']['logical_count'],count//2)
                self.assertAlmostEqual(groups['qkv']['geometric_mean']['te_ub_x'],qkv_gm)
                self.assertAlmostEqual(groups['oproj']['geometric_mean']['te_ub_x'],2.)
                self.assertAlmostEqual(groups['overall']['geometric_mean']['te_ub_x'],(qkv_gm*2.)**.5)
                # OProj is below overall GM, but not below its own direction GM.
                self.assertEqual(groups['overall']['below_direction_geomean']['te_ub_x'],[])

    def test_aggregate_preserves_logical_alias_weights_and_shared_source_samples(self):
        cases=history()
        placement={report.physical_key(c):'09' if c['cp']==4 else '0a' for c in cases}
        target=next(c for c in cases if c['direction']=='oproj' and c['model']=='representative_small'
                    and c['seq']==1024 and c['cp']==4)
        key=report.physical_key(target)
        rows=measurements(cases,placement)
        for row in rows:
            if row['kind'] in report.BACKENDS and row['launch']=='eager' and report.physical_key(row['case'])==key:
                row['p50_ms']=16. if row['kind']=='te_ub' else 12.
        table,_,_=report.join_measurements(cases,placement,rows)
        aliases=[c for c in table if c['launch']=='eager' and report.physical_key(c['case'])==key]
        self.assertEqual(len(aliases),2)
        self.assertIs(aliases[0]['measurements']['te_ub'],aliases[1]['measurements']['te_ub'])
        result=report.aggregate_ratios(table)
        op=result['launches']['eager']['all']['oproj']
        expected=report.math.exp((94*report.math.log(2.)+2*report.math.log(8.))/96)
        physical_weighted=report.math.exp((71*report.math.log(2.)+report.math.log(8.))/72)
        self.assertAlmostEqual(op['geometric_mean']['te_ub_x'],expected)
        self.assertNotAlmostEqual(op['geometric_mean']['te_ub_x'],physical_weighted)
        self.assertAlmostEqual(op['geometric_mean']['strong_x'],
                               report.math.exp((94*report.math.log(1.5)+2*report.math.log(6.))/96))
        self.assertEqual(len(op['below_direction_geomean']['te_ub_x']),94)
        self.assertEqual(len(result['launches']['eager']['all']['overall']['below_direction_geomean']['te_ub_x']),94)
        self.assertEqual(result['launches']['eager']['long']['oproj']['below_direction_geomean']['te_ub_x'],[])
        self.assertEqual(result['launches']['graph']['all']['oproj']['below_direction_geomean']['te_ub_x'],[])

    def test_aggregate_rejects_partial_duplicate_and_invalid_ratios(self):
        cases=history()
        placement={report.physical_key(c):'09' if c['cp']==4 else '0a' for c in cases}
        table,_,_=report.join_measurements(cases,placement,measurements(cases,placement))
        for invalid in (table[:-1],table[:-1]+[table[0]]):
            with self.assertRaises(ValueError):report.aggregate_ratios(invalid)
        for value in (0.,float('nan')):
            bad=copy.deepcopy(table)
            bad[0]['ratios']['te_ub_x']=value
            with self.assertRaises(ValueError):report.aggregate_ratios(bad)
        with self.assertRaises(ValueError):
            report.write_report(self.root/'stale-aggregate',dict(complete=False,aggregates={}),table[:1])
        self.assertFalse((self.root/'stale-aggregate').exists())

    def test_fused_adapter_requires_exact_pool_and_does_not_minimize_p95(self):
        candidates = []
        for direction in ('GEMM_A2A', 'A2A_GEMM'):
            for index, (tile, comm) in enumerate(sorted(report.POOL)):
                candidates.append(dict(direction=direction, component='fused', tile_policy=tile, comm_ctas=comm,
                    candidate=index, performance_accepted=True, launch='eager',
                    layout=report.LAYOUTS['qkv' if direction == 'GEMM_A2A' else 'oproj'],
                    max_swizzle_size=1, effective_swizzle_size=1, raster_requested='heuristic',
                    timing=dict(p50_ms=index+1., p95_ms=100.-index), m=128,n=128,k=128))
        run = dict(source_id='a'*64, build=dict(mpi=True,profile=False,binary_sha256='b'*64), diagnostic_only=False,
            config=dict(process_layout='mpi_one_process_per_gpu'), candidates=candidates,
            geometry=dict(world=4,global_seq=512,hidden=128,q_heads=4,kv_heads=4,head_dim=32),
            node='09',run_id='test',environment_fingerprint='c'*64,
            evidence={'control/attempt1.log':dict(path='/raw',sha256='d'*64)},
            telemetry=dict(devices=[dict(physical_index=str(i),uuid='GPU-'+str(i)) for i in range(4)]))
        source = object.__new__(report.Sources)
        source.fused = {}
        with mock.patch.object(report.fused,'audit_run',return_value=run) as audit:
            rows = source.fusion(dict(path='/example',source_id='a'*64,launch='eager'))
            self.assertEqual(rows[0]['p95_ms'], 100.)
            self.assertEqual(audit.call_count, 1)
            with self.assertRaises(ValueError):
                source.fusion(dict(path='/example',source_id='a'*64,launch='eager'))
        for change in ('missing', 'e64', 'profile', 'wrong_source'):
            bad = copy.deepcopy(run)
            if change == 'missing': bad['candidates'].pop()
            if change == 'e64': bad['candidates'][0]['tile_policy'] = 'm128n256k64e64'
            if change == 'profile': bad['build']['profile'] = True
            if change == 'wrong_source': bad['source_id'] = 'e'*64
            source.fused = {}
            with mock.patch.object(report.fused,'audit_run',return_value=bad), self.assertRaises(ValueError):
                source.fusion(dict(path='/example',source_id='a'*64,launch='eager'))

    def make_envelope(self):
        root = self.root / 'run'
        artifact = root / 'artifacts-attempt1'
        control = artifact / 'control'
        control.mkdir(parents=True)
        source = {'file': 'a'*64}
        job = dict(run_id='test',stage='gemm-probe',experiment='test',node='09',
                   files=source,source_id=report.fused.json_digest(source))
        env = dict(host=report.fused.NODES['09'][1])
        env['fingerprint'] = report.fused.json_digest(env)
        receipt = {k:v for k,v in job.items() if k != 'files'} | dict(state='succeeded',phase='finished',
            exit_code=0,work_exit_code=0,attempt=1,environment_fingerprint=env['fingerprint'])
        for name,data in {'job.json':job,'status.json':receipt,'environment.json':env,
                          'source-installed.json':dict(source_id=job['source_id'],files=source)}.items():
            (control/name).write_text(json.dumps(data))
        with tarfile.open(root/'artifacts.tar.gz','w:gz') as stream:
            for path in sorted(control.iterdir()): stream.add(path,arcname='control/'+path.name)
        receipt['artifact_sha256'] = report.fused.file_digest(root/'artifacts.tar.gz')
        (root/'fetched.json').write_text(json.dumps(receipt))
        return root

    def test_real_archive_receipt_and_tampered_extraction(self):
        root = self.make_envelope()
        run = report.audit_envelope(root)
        self.assertEqual(run['node'], '09')
        (root/'artifacts-attempt1/control/job.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'extraction differs'):
            report.audit_envelope(root)

    def test_envelope_cache_reads_each_run_once(self):
        source = object.__new__(report.Sources)
        source.envelopes = {}
        with mock.patch.object(report,'audit_envelope',return_value={}) as audit:
            source.envelope('/a');source.envelope('/a')
            self.assertEqual(audit.call_count, 1)

    def pure_fixture(self):
        control = self.root/'pure/control'
        control.mkdir(parents=True)
        matrix = dict(schema='sm103_gemm_matrix_v1',shapes=[dict(id='one',m=128,n=128,k=128)])
        (control/'gemm-matrix.json').write_text(json.dumps(matrix))
        plan = dict(requested=256,valid=1,returned=1,precision=16,math_sms=0,beta=0,graph_tuning=0,
                    best_ms=1.,algorithm=5,candidates=[dict(tune_ms=1.)])
        results = [dict(precision='bf16',launch=launch,warmup=10,tune_warmup=10,tune_iterations=50,
            samples_ms=[2.]*50,p50_ms=2.,p95_ms=2.,measurement=stable(),
            tuning=plan|dict(replay=launch),correctness=dict(checked_values=4096,relative_rms=.001,max_abs=.001),
            pflops_per_gpu_p50=128**3/1e12) for launch in report.LAUNCHES]
        tensor = dict(dtype='torch.bfloat16',sample_nonzero_fraction=1.,sample_min=-.01,sample_max=.01,
                      sample_std=.005,shape=[128,128],sample_count=4096)
        geometry = dict(shape=dict(m=128,n=128,k=128),aliases=['one'],state='succeeded',results=results,
            inputs=dict(seed=103,distribution='uniform',activation_magnitude=.125,weight_magnitude=.02,
                        magnitude_meaning='uniform_half_range',activation=tensor,weight=tensor))
        data = dict(schema='sm103_gemm_matrix_v1',state='succeeded',measurement='single_gpu_pure_cublaslt',
            compute='10.3',sms=148,output_dtype='bf16',measured_ranks=1,distributed_boundary_measured=False,
            cublas_classic_measured=False,measurement_protocol='single_gpu_pure_gemm_stable_v2',
            library_sha256='b'*64,matrix_sha256=report.fused.file_digest(control/'gemm-matrix.json'),
            unique_geometries=1,logical_shapes=1,geometries=[geometry])
        contract = dict(measurement=data['measurement'],node='09',physical_devices=['0'],
                        cuda_visible_devices='GPU-0',shapes=matrix['shapes'],library_sha256='b'*64)
        before = dict(node='09',selected=['0'],observations=[dict(devices=[dict(index='0',uuid='GPU-0')])])
        for name,value in [('gemm-probe.json',data),('gemm-probe-contract.json',contract),('gpu-before.json',before)]:
            (control/name).write_text(json.dumps(value))
        run = dict(artifact=control.parent,node='09',run_id='pure',source_id='a'*64,environment_fingerprint='c'*64,
            job=dict(stage='gemm-probe',gemm_matrix_payload=matrix,launches='eager,graph'),
            files={'control/gemm-probe.json':report.evidence(control/'gemm-probe.json')})
        return run, data, control

    def test_pure_raw_reader_keeps_eager_graph_separate(self):
        run,data,control = self.pure_fixture()
        rows = report.audit_pure_run(run)
        self.assertEqual([r['launch'] for r in rows], ['eager','graph'])
        self.assertTrue(all(r['scope']=='single_gpu_diagnostic_not_distributed_maxrank' for r in rows))
        self.assertNotEqual(rows[0]['provenance']['pointer'],rows[1]['provenance']['pointer'])

    def test_pure_rejects_native_zero_inputs_forged_stats_and_retuned_graph(self):
        run,original,control = self.pure_fixture()
        for mutation in ('native','zero','stats','graph_plan','missing_graph','warmup'):
            data = copy.deepcopy(original)
            if mutation=='native': data['schema']='sm103_gemm_comparison_v1'
            if mutation=='zero': data['geometries'][0]['inputs']['activation']['sample_nonzero_fraction']=0.
            if mutation=='stats': data['geometries'][0]['results'][0]['p50_ms']=1.
            if mutation=='graph_plan': data['geometries'][0]['results'][1]['tuning']['algorithm']=8
            if mutation=='missing_graph': data['geometries'][0]['results'].pop()
            if mutation=='warmup': data['geometries'][0]['results'][0]['tune_warmup']=2
            (control/'gemm-probe.json').write_text(json.dumps(data))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                report.audit_pure_run(run)

    def campaign_fixture(self):
        manifest = dict(scope=dict(logical_direction_geometry_rows=192,physical_direction_geometry_rows=168,
            requested_launches=['eager','graph'],historical_sequence_lengths=[1024,4096,16384,131072,262144,524288],
            node_world={'09':4,'0a':8}))
        mp = self.root/'manifest.json'
        mp.write_text(json.dumps(manifest))
        case = next(c for c in history() if c['cp']==4 and c['hidden']==16384 and c['q_heads']==128 and c['seq']==524288)
        geometry = {k:case[{'world':'cp','global_seq':'seq'}.get(k,k)] for k in report.GEOMETRY}
        override = dict(geometry=geometry,original_node='09',destination_node='0a',cross_node_ratios_permitted=False,
                        destination_state={'complete':True})
        op = self.root/'placement.json'
        op.write_text(json.dumps(override))
        campaign = dict(schema=report.SCHEMA,manifest=dict(path='manifest.json',sha256=report.fused.file_digest(mp)),
            placement_overrides=[dict(path='placement.json',sha256=report.fused.file_digest(op))],
            source_sweep=dict(path='source',fingerprint='fingerprint'),fused=[],baselines=[],pure=[])
        cp = self.root/'campaign.json'
        cp.write_text(json.dumps(campaign))
        return cp,campaign,case

    def test_campaign_resolves_exact_paths_and_placement_is_not_completion(self):
        path,campaign,case = self.campaign_fixture()
        _,cases,nodes = report.load_campaign(path)
        self.assertEqual(nodes[report.physical_key(case)],'0a')
        _,missing,_ = report.join_measurements(cases,nodes,[])
        self.assertEqual(len(missing),384)
        self.assertEqual({r['node'] for r in missing if r['case']==report.logical_id(case)},{'0a'})

    def test_campaign_pin_unknown_fields_and_source_version_mixing_rejected(self):
        path,original,_ = self.campaign_fixture()
        for mutation in ('pin','unknown','versions','duplicate'):
            c = copy.deepcopy(original)
            if mutation=='pin': c['manifest']['sha256']='0'*64
            if mutation=='unknown': c['pick_fastest_version']=True
            if mutation in ('versions','duplicate'):
                c['fused']=[dict(path='one',source_id='a'*64,launch='eager'),
                            dict(path='two' if mutation=='versions' else 'one',source_id='b'*64,launch='eager')]
            path.write_text(json.dumps(c))
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                report.load_campaign(path)

    def baseline_fixture(self):
        artifact = self.root/'baseline'
        folder = artifact/'results/baseline-replay'
        folder.mkdir(parents=True)
        case = dict(direction='oproj',model='sample',seq=1024,cp=4,hidden=128,q_heads=4,kv_heads=4,
                    head_dim=32,m=256,n=128,k=128)
        config = dict(channels=8,chunk_kib=128,ll_kib=16,pack_block=512,pack_warps=4,high_priority=False)
        env = dict(CUDA_VISIBLE_DEVICES='0,1,2,3',FUSE_SM103_MEASUREMENT='v2',
                   FUSE_SM103_OPROJ_LAYOUT='causal_dual_chunk_v1',NCCL_MIN_P2P_NCHANNELS='8')
        job = dict(case=case,config=config,env=env,warmup=10,iterations=50,backend='cublaslt_nccl',launch='eager',
                   output='/remote/baseline-replay/result.json',group='group')
        metric = report.bench.METRICS['oproj','cublaslt_nccl']
        data = dict(samples_ms={metric:[2.]*50},correctness=dict(inverse_a2a_reference_mismatches=0,
                    cublaslt_vs_torch_mm_max_abs=0.),world_size=4,cuda_graph=False,
                    oproj_layout='causal_dual_chunk_v1',gemm_shape=dict(m=256,n=128,k=128),
                    results={metric:dict(p50_ms=2.,p95_ms=2.)},environment=env,
                    pack_block=512,pack_warps=4,nccl_high_priority=False)
        path = folder/'result.json'
        path.write_text(json.dumps(data))
        ranks = []
        for rank in range(4):
            record = stable(pure=False)|dict(launch='eager',sample_cadence_warmup_ms=[2.]*10,graph_replay_warmup=0)
            tensor = dict(dtype='torch.bfloat16',nonzero_fraction=1.,sample_min=-.01,sample_max=.01,sample_std=.005)
            md = dict(rank=rank,direction='oproj',oproj_layout='causal_dual_chunk_v1',
                device=dict(sm_count=148,compute_capability='10.3'),cuda_visible_devices=env['CUDA_VISIBLE_DEVICES'],
                cublaslt_tune_launch='eager',measurement_records=[record],
                input_statistics=[dict(seed=2701+rank,distribution='uniform',tensors=dict(activation=tensor,weight=tensor))],
                cublaslt_plans=[dict(valid=1,returned=1)])
            ranks.append(md)
            path.with_suffix(f'.rank{rank}.json').write_text(json.dumps(md))
        run = dict(artifact=artifact,node='09',run_id='baseline',source_id='a'*64,environment_fingerprint='b'*64,
                   files={str(p.relative_to(artifact)):report.evidence(p) for p in folder.iterdir()})
        return run,job,path,data,ranks

    def test_baseline_uses_actual_raw_and_all_rank_stability(self):
        run,job,path,data,ranks = self.baseline_fixture()
        row = report.audit_baseline_measurement(run,job,'baseline-replay')
        self.assertEqual(row['p50_ms'],2.)
        self.assertEqual(len(row['provenance']['ranks']),4)
        self.assertTrue(row['sampling'][0]['maxrank_samples_repeated_on_ranks'])
        self.assertEqual(row['provenance']['raw']['path'],str(path))

    def test_baseline_rejects_configuration_layout_correctness_and_missing_rank(self):
        run,job,path,original,ranks = self.baseline_fixture()
        for mutation in ('config','layout','correctness','missing_route','percentile'):
            data = copy.deepcopy(original)
            if mutation=='config': data['environment']['NCCL_MIN_P2P_NCHANNELS']='32'
            if mutation=='layout': data['oproj_layout']='legacy'
            if mutation=='correctness': data['correctness']['inverse_a2a_reference_mismatches']=1
            if mutation=='missing_route': del data['correctness']['inverse_a2a_reference_mismatches']
            if mutation=='percentile': data['results'][report.bench.METRICS['oproj','cublaslt_nccl']]['p50_ms']=1.
            path.write_text(json.dumps(data))
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                report.audit_baseline_measurement(run,job,'baseline-replay')
        path.write_text(json.dumps(original))
        ranks[3]['measurement_records'][0]['selected_round']=1
        path.with_suffix('.rank3.json').write_text(json.dumps(ranks[3]))
        with self.assertRaises(ValueError):
            report.audit_baseline_measurement(run,job,'baseline-replay')

    def test_no_select_or_implicit_source_discovery(self):
        source = object.__new__(report.Sources)
        with mock.patch.object(source,'envelope') as envelope,self.assertRaises(ValueError):
            source.baseline(dict(path='/source',select=dict(directions=['qkv'])))
        envelope.assert_not_called()

    def mixed_replay_fixture(self):
        source = object.__new__(report.Sources)
        source.plan, source.plan_hash = dict(fingerprint='source'), 'plan-hash'
        source.groups = {f'cp{cp}': [None] for cp in (4,8)}
        jobs, entries, winners = [], [], {}
        for cp in (4,8):
            group = f'cp{cp}'
            provenance = dict(raw=dict(sha256=group), ranks=[dict(raw=dict(sha256=str(i))) for i in range(cp)])
            selected = dict(selection_stage='sweep',status='complete',independently_remeasured=False,
                expected_candidates=1,validated_candidates=1,warmup=10,samples=50,raw_sha256=group,
                rank_sha256={str(i):str(i) for i in range(cp)},p50_ms=2.,p95_ms=3.)
            prior = dict(case=dict(cp=cp),backend='te_ub',launch='eager',config={})
            job = prior | dict(group=group,source_winner=selected,
                               env=dict(CUDA_VISIBLE_DEVICES=','.join('GPU-'+str(i) for i in range(cp))))
            jobs.append(job)
            entries.append(prior | dict(group=group,source=selected))
            winners[group] = (dict(p50_ms=2.,p95_ms=3.),prior,dict(provenance=provenance))
        payload = dict(schema='sm103_baseline_replay_input_v1',source_node='09',source_fingerprint='source',
                       source_plan_sha256='plan-hash',entries=entries)
        plan = dict(stage='baseline-replay',precision='bf16',node='0a',source_id='source-id',environment_fingerprint='env',
                    fingerprint='replay',source_winners=payload,imports_source_measurements=False,
                    communication_search=False,jobs=jobs,oproj_layout='causal_dual_chunk_v1')
        run = dict(artifact=self.root,node='0a',source_id='source-id',environment_fingerprint='env',
            job=dict(stage='baseline-replay',baseline_replay=payload,oproj_layout='causal_dual_chunk_v1',
                     devices=','.join(str(i) for i in range(8))))
        records = {'replay_plan.json':plan,
                   'baseline-replay.json':dict(node='0a',fingerprint='replay',source_fingerprint='source'),
                   'baseline-devices.json':dict(node='0a',physical=[str(i) for i in range(8)],
                                               cuda_visible_devices=','.join('GPU-'+str(i) for i in range(8))),
                   'replay_plan.executor.json':dict(measurement_fingerprint='replay')}
        return source,run,records,winners

    def test_mixed_cp_replay_validates_exact_prefix_even_when_selecting_only_cp8(self):
        source,run,records,winners = self.mixed_replay_fixture()
        def measured(_run,job,_folder):
            return dict(group=job['group'],provenance={})
        for groups in (['cp4','cp8'],['cp8']):
            with mock.patch.object(source,'envelope',return_value=run), \
                    mock.patch.object(source,'winner',side_effect=lambda group:winners[group]), \
                    mock.patch.object(report,'read',side_effect=lambda path:records[path.name]), \
                    mock.patch.object(report,'audit_baseline_measurement',side_effect=measured) as audit:
                rows = source.baseline(dict(path='/mixed',groups=groups))
                self.assertEqual([row['group'] for row in rows],groups)
                self.assertEqual(audit.call_count,2)  # Selection must not bypass CP4 audit.
                for row in rows:
                    cp = int(row['group'][2:])
                    self.assertEqual(row['physical_devices'],[str(i) for i in range(cp)])

    def test_mixed_cp_replay_rejects_wrong_prefix_extra_rank_and_bad_inventory(self):
        for mutation in ('reorder','extra','short','duplicate_uuid','duplicate_physical','wrong_index','undersized'):
            source,run,records,winners = self.mixed_replay_fixture()
            env = records['replay_plan.json']['jobs'][0]['env']
            devices = records['baseline-devices.json']
            if mutation=='reorder': env['CUDA_VISIBLE_DEVICES']='GPU-1,GPU-0,GPU-2,GPU-3'
            if mutation=='extra': env['CUDA_VISIBLE_DEVICES']+=',GPU-4'
            if mutation=='short': env['CUDA_VISIBLE_DEVICES']='GPU-0,GPU-1,GPU-2'
            if mutation=='duplicate_uuid':
                uuids = devices['cuda_visible_devices'].split(',')
                uuids[1] = uuids[0]
                devices['cuda_visible_devices'] = ','.join(uuids)
            if mutation=='duplicate_physical': devices['physical'][1]='0'
            if mutation=='wrong_index': devices['physical'][7]='8'
            if mutation=='undersized': devices['physical'].pop()
            with self.subTest(mutation=mutation),mock.patch.object(source,'envelope',return_value=run), \
                    mock.patch.object(source,'winner',side_effect=lambda group:winners[group]), \
                    mock.patch.object(report,'read',side_effect=lambda path:records[path.name]), \
                    mock.patch.object(report,'audit_baseline_measurement',return_value=dict(provenance={})), \
                    self.assertRaises(ValueError):
                source.baseline(dict(path='/mixed',groups=['cp8']))

    def test_legacy_receipt_node_and_qkv_layout_are_manifest_gated(self):
        for controller, accepted in [(report.LEGACY_QKV_REPLAY_CONTROLLER,True),('modern',False)]:
            source,run,records,winners = self.mixed_replay_fixture()
            run['job']['files']={'scripts/l20d.py':controller}
            del records['baseline-replay.json']['node']
            del records['replay_plan.json']['oproj_layout']
            del run['job']['oproj_layout']
            for job in records['replay_plan.json']['jobs']:
                job['case']['direction']='qkv'  # Same shared source/entry case.
            with mock.patch.object(source,'envelope',return_value=run), \
                    mock.patch.object(source,'winner',side_effect=lambda group:winners[group]), \
                    mock.patch.object(report,'read',side_effect=lambda path:records[path.name]), \
                    mock.patch.object(report,'audit_baseline_measurement',side_effect=lambda *args:dict(provenance={})):
                if accepted:
                    self.assertEqual(len(source.baseline(dict(path='/mixed',groups=['cp8']))),1)
                else:
                    with self.assertRaises(ValueError):
                        source.baseline(dict(path='/mixed',groups=['cp8']))
            # Even the old manifest cannot omit an OProj layout or invent node09.
            if accepted:
                for job in records['replay_plan.json']['jobs']:
                    job['case']['direction']='oproj'
                with mock.patch.object(source,'envelope',return_value=run), \
                        mock.patch.object(report,'read',side_effect=lambda path:records[path.name]), \
                        self.assertRaises(ValueError):
                    source.baseline(dict(path='/mixed',groups=['cp8']))

    def test_only_first_qkv_replay_may_omit_rank_direction(self):
        run,job,path,data,ranks = self.baseline_fixture()
        job['case']['direction']='qkv'
        metric=report.bench.METRICS['qkv','cublaslt_nccl']
        data['samples_ms']={metric:[2.]*50}
        data['results']={metric:dict(p50_ms=2.,p95_ms=2.)}
        data['correctness']=dict(packed_qkv_mismatches=0,cublaslt_vs_torch_mm_max_abs=0.)
        path.write_text(json.dumps(data))
        for rank,md in enumerate(ranks):
            del md['direction']
            md['input_statistics'][0]['seed']=3109+rank
            path.with_suffix(f'.rank{rank}.json').write_text(json.dumps(md))
        run['job']=dict(files={'scripts/l20d.py':report.LEGACY_QKV_REPLAY_CONTROLLER})
        self.assertEqual(report.audit_baseline_measurement(run,job,'baseline-replay')['p50_ms'],2.)
        run['job']['files']['scripts/l20d.py']='modern'
        with self.assertRaises(ValueError):
            report.audit_baseline_measurement(run,job,'baseline-replay')


if __name__ == '__main__':
    unittest.main()
