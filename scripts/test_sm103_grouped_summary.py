"""Grouped sample acceptance: missing/unstable rows cannot become winners."""
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from summarize_sm103_grouped import audit_samples, comparison_rows, catalog_rows, seed_fusion_plan, collect_plan, catalog_markdown, collect_external_plan, external_markdown, grouped_handoff_trace
from summarize_sm103_grouped import grouped_ready_checks, grouped_ready_summary


class GroupedSummaryTests(unittest.TestCase):
    def test_ready_summary_validates_schedule_and_separates_first_wait(self):
        r=dict(schema='grouped-ready-summary-v1',world=1,rank=0,n=256,k=128,
            tile_n=128,tile_k=64,swizzle=2,along_n=1,comm=1,compute=2,
            warmup=10,payload=1,epoch=13,rows=[0,256,0],
            ctas=[[100,9100,11100,0,0,0,0,0],
                  [100,10100,11100,1,6000,6000,6000,1],
                  [100,10100,11100,1,5000,5000,5000,1]])
        self.assertEqual(grouped_ready_checks(r),[1,1])
        s=grouped_ready_summary([r]); self.assertAlmostEqual(s['weighted_wait_fraction'],.55)
        self.assertEqual(s['critical']['critical_wait_fraction'],.6)
        self.assertEqual(s['critical']['critical_later_wait_us'],0)
        r['ctas'][1][3]=2
        with self.assertRaisesRegex(ValueError,'tile schedule'): grouped_ready_summary([r])
        r['ctas'][1][3]=1; r['ctas'][1][4]=11000
        with self.assertRaisesRegex(ValueError,'counters'): grouped_ready_summary([r])

    def test_ready_schedule_covers_partial_bands_and_rasters(self):
        r=dict(n=384,tile_n=128,compute=1,swizzle=2,along_n=1,rows=[257])
        self.assertEqual(grouped_ready_checks(r),[7])
        r['along_n']=0
        self.assertEqual(grouped_ready_checks(r),[6])
        r.update(compute=16,rows=[1])
        self.assertEqual(grouped_ready_checks(r),[1,1,1]+[0]*13)

    def test_external_catalog_requires_exact_fetched_plan(self):
        with tempfile.TemporaryDirectory(prefix='grouped-external-test-') as tmp:
            root=Path(tmp); runs=root/'runs'; run=runs/'run'; run.mkdir(parents=True)
            plan=root/'plan.json'; argument='64,128,32,2,7@128,64,0,1,20,128@dispatch'
            plan.write_text(json.dumps(dict(schema='grouped-external-reference-v1',
                source_run='build',case_count=1,note='Pure GEMM only.',
                batches=[dict(ep=4,models=['model'],entries=[argument])])))
            job=dict(source_run='build',world=4,grouped_external=True,
                     grouped_direction='dispatch',grouped_cases=argument,
                     experiment='grouped-external-full-00')
            (run/'job.json').write_text(json.dumps(job))
            with patch('summarize_sm103_grouped.audit_cases') as audit:
                result=collect_external_plan(plan,runs)
                self.assertEqual(result['coverage'],dict(unmeasured=1))
                audit.assert_not_called()
            (run/'fetched.json').write_text(json.dumps(dict(state='succeeded')))
            config=dict(tile_n=128,tile_k=64,along_n=0,swizzle=1,comm=0,compute=148)
            record=dict(run_id='run',case_index=0,source_id='s',environment_fingerprint='e',
                routing_ids={'dispatch':'r'},rows=[
                    dict(mode=mode,valid=valid,ms=ms,pflops=1/ms,config=config)
                    for mode,valid,ms in [('fused',True,4.),('cutlass_stock',True,2.),
                        ('cutlass_stock',False,.5),('deepgemm_m128',True,3.),
                        ('deepgemm_m256',True,1.5)]])
            with patch('summarize_sm103_grouped.audit_cases',return_value=[record]):
                result=collect_external_plan(plan,runs); row=result['rows'][0]
                self.assertEqual(row['best_backend'],'deepgemm_m256')
                self.assertEqual(row['retention'],1.5/4.)
                self.assertEqual(row['excess_ms'],2.5)
                self.assertEqual(row['reference']['ms'],1.5)
                self.assertEqual(row['reference']['mode'],'deepgemm_m256')
                self.assertEqual(row['reference']['config'],config)
                self.assertNotIn('cutlass',row)
                self.assertNotIn('deepgemm',row)
                self.assertIn('Best GEMM P',external_markdown(result))
                self.assertIn('37.5%',external_markdown(result))
                fill_plan=root/'fill.json'
                fill_plan.write_text(json.dumps(dict(schema='grouped-external-fill-v1',
                    source_run='build',case_count=1,batches=[dict(ep=4,entries=[argument])])))
                # Retests fill holes; they must never cherry-pick an already
                # measured point or silently mix a different environment.
                with self.assertRaisesRegex(ValueError,'previously memory-skipped'):
                    collect_external_plan(plan,runs,fill_plan)
                fill=runs/'fill'; fill.mkdir()
                (fill/'job.json').write_text(json.dumps(job|dict(experiment='grouped-external-fill-00')))
                (fill/'fetched.json').write_text('{}')
                skipped={k:record[k] for k in ('run_id','case_index','source_id','environment_fingerprint')}
                skipped.update(skipped=True,reason='insufficient_memory')
                filled=record|dict(run_id='fill')
                with patch('summarize_sm103_grouped.audit_cases',side_effect=
                           lambda p:[filled if p.name=='fill' else skipped]):
                    result=collect_external_plan(plan,runs,fill_plan)
                    self.assertEqual(result['coverage'],dict(valid=1))
                    self.assertEqual(result['rows'][0]['previous_attempt']['run'],'run')
                    self.assertEqual(result['rows'][0]['run'],'fill')
                    self.assertEqual(result['fill_runs'],['fill'])
                    filled['environment_fingerprint']='different'
                    with self.assertRaisesRegex(ValueError,'Mixed external'):
                        collect_external_plan(plan,runs,fill_plan)
                (run/'job.json').write_text(json.dumps(job|dict(world=8)))
                with self.assertRaises(ValueError): collect_external_plan(plan,runs)
                (run/'job.json').write_text(json.dumps(job))
                duplicate=runs/'duplicate'; duplicate.mkdir()
                (duplicate/'job.json').write_text(json.dumps(job))
                (duplicate/'fetched.json').write_text('{}')
                with self.assertRaises(ValueError): collect_external_plan(plan,runs)

    def test_grouped_handoff_observations_not_e2e_wait(self):
        import copy
        rank=dict(schema='grouped-handoff-v1',rank=0,world=1,n=256,k=128,tile_n=256,
            tile_k=64,comm=1,compute=1,swizzle=1,along_n=1,rows=[128],warmup=10,payload=1,epoch=13,
            roles=[[100,170,300],[100,290,300]],panels=[[110,150,155,0,0,0]],
            tiles=[[120,160,162,180,1,0,0,0,1]])
        trace=grouped_handoff_trace([rank]);s=trace['metadata']['rank_summaries'][0]
        self.assertAlmostEqual(s['ready_wait_p50_us'],.04)
        self.assertAlmostEqual(s['release_to_first_acquire_p50_us'],.01)
        self.assertAlmostEqual(s['ctas'][0]['before_publish_us'],.03)
        self.assertAlmostEqual(s['ctas'][0]['after_publish_us'],.01)
        self.assertTrue(all(e.get('dur',0)>=0 for e in trace['traceEvents']))
        for change in ('missing','backwards','wrong_owner','wrong_coord','prepublication'):
            bad=copy.deepcopy(rank)
            if change=='missing':bad['tiles']=[]
            if change=='backwards':bad['tiles'][0][3]=161
            if change=='wrong_owner':bad['tiles'][0][4]=0
            if change=='wrong_coord':bad['tiles'][0][6]=1
            if change=='prepublication':bad['tiles'][0][1]=140
            with self.assertRaises(ValueError):grouped_handoff_trace([bad])

    def test_markdown_preserves_missing_and_config(self):
        base=dict(models=['model'],ep=4,target_rows=128,direction='dispatch')
        measured=base|dict(status='valid',fused_ms=2.,fused_pflops=.5,
            reference_pflops=1.,retention=.5,reference_method='cublaslt_sequence_tuned',
            config=dict(tile_n=256,tile_k=64,along_n=1,swizzle=4,comm=40,compute=108))
        missing=base|dict(status='insufficient_memory',direction='combine')
        text=catalog_markdown(dict(schema='grouped-bf16-current-v1',note='Measured only.',
            completed_batches=1,total_batches=34,coverage=dict(valid=1,insufficient_memory=1),
            rows=[measured,missing]))
        self.assertIn('2.000000 | 0.500 | 1.000 | 50.0% | Lt | 256/64/N/4/40/108',text)
        self.assertIn('| model | 4 | 128 | '+' | '.join(['—']*6)+' | insufficient_memory |',text)

    def setUp(self):
        self.geometry=dict(h=64,f=96,experts=12,topk=2,tokens_per_rank=3)
        self.counts={(d,p,r):[1,2,3] for d in ('dispatch','combine') for p in (0,1) for r in range(4)}
        self.rows=[]
        for direction in ('dispatch','combine'):
            for mode in ('fused','transport_body','cutlass_matched','cublas_grouped_default','cublaslt_sequence_tuned'):
                for payload in (0,1):
                    for sample in range(50):
                        for rank in range(4):
                            self.rows.append(dict(direction=direction,mode=mode,payload=payload,
                                sample=sample,rank=rank,ep=4,h=64,f=96,total_experts=12,topk=2,
                                tokens_per_rank=3,rank_rows=6,ms=(payload+1)*(rank+1),
                                tile_n=128,tile_k=64,along_n=0,swizzle=1,comm_ctas=20,
                                compute_ctas=128,warmup=138,drift=0,round=0,accepted=1))

    def audit(self, rows=None):
        rows=self.rows if rows is None else rows
        stream=io.StringIO()
        writer=csv.DictWriter(stream,fieldnames=self.rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
        return audit_samples(stream.getvalue(),4,self.geometry,self.counts,False)

    def test_max_rank_and_both_payloads(self):
        results=self.audit()
        self.assertEqual(len(results),10)
        for row in results:
            self.assertTrue(row['valid'])
            self.assertEqual(row['ms'],6.)  # mean(max ranks=4, max ranks=8)

    def test_fused_only_is_not_a_pure_gemm_comparison(self):
        rows=[r for r in self.rows if r['mode']=='fused']
        stream=io.StringIO(); writer=csv.DictWriter(stream,fieldnames=rows[0].keys())
        writer.writeheader();writer.writerows(rows)
        result=audit_samples(stream.getvalue(),4,self.geometry,self.counts,False,fused_only=True)
        self.assertEqual(len(result),2)
        self.assertTrue(all(r['mode']=='fused' and r['ms']==6. for r in result))
        with self.assertRaises(ValueError):
            audit_samples(stream.getvalue(),4,self.geometry,self.counts,False)

    def test_transport_comparison_requires_both_measured_modes(self):
        rows=[r for r in self.rows if r['mode'] in ('fused','transport_body')]
        stream=io.StringIO(); writer=csv.DictWriter(stream,fieldnames=rows[0].keys())
        writer.writeheader();writer.writerows(rows)
        result=audit_samples(stream.getvalue(),4,self.geometry,self.counts,False,
                             fused_only=True,transport_compare=True)
        self.assertEqual(len(result),4)
        self.assertTrue(all(r['valid'] and r['ms']==6. for r in result))
        with self.assertRaises(ValueError):
            audit_samples(stream.getvalue(),4,self.geometry,self.counts,False,fused_only=True)

    def test_fixed_compute_comparison_requires_all_three_modes(self):
        rows=[r for r in self.rows if r['mode'] in ('fused','transport_body','cutlass_matched')]
        stream=io.StringIO(); writer=csv.DictWriter(stream,fieldnames=rows[0].keys())
        writer.writeheader();writer.writerows(rows)
        result=audit_samples(stream.getvalue(),4,self.geometry,self.counts,False,
                             fused_only=True,transport_compare=True,compute_compare=True)
        self.assertEqual(len(result),6)
        self.assertTrue(all(r['valid'] for r in result))
        with self.assertRaises(ValueError):
            audit_samples(stream.getvalue(),4,self.geometry,self.counts,False,
                          fused_only=True,transport_compare=True)
        with self.assertRaises(ValueError):
            audit_samples(stream.getvalue(),4,self.geometry,self.counts,False,
                          fused_only=True,compute_compare=True)
        rows=[r for r in rows if r['mode']!='cutlass_matched']
        stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=rows[0].keys())
        writer.writeheader();writer.writerows(rows)
        with self.assertRaises(ValueError):
            audit_samples(stream.getvalue(),4,self.geometry,self.counts,False,
                          fused_only=True,transport_compare=True,compute_compare=True)

    def test_external_reference_grid_and_full_budget(self):
        base=[r for r in self.rows if r['direction']=='dispatch' and r['mode']=='fused']
        records=list(base)
        for n in (128,256):
            for k in (64,128):
                for along in (0,1):
                    for sw in (1,2,4,8):
                        records += [r|dict(mode='cutlass_stock',tile_n=n,tile_k=k,along_n=along,
                                          swizzle=sw,comm_ctas=0,compute_ctas=148) for r in base]
        for mode in ('deepgemm_m128','deepgemm_m256'):
            records += [r|dict(mode=mode,comm_ctas=0,compute_ctas=148) for r in base]
        def audit(rows):
            out=io.StringIO(); writer=csv.DictWriter(out,fieldnames=self.rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
            return audit_samples(out.getvalue(),4,self.geometry,self.counts,'external',('dispatch',))
        self.assertEqual(len(audit(records)),35)
        with self.assertRaises(ValueError):
            audit([r for r in records if r['mode']!='deepgemm_m256'])
        with self.assertRaises(ValueError):
            audit([r|dict(compute_ctas=128) if r['mode']=='deepgemm_m128' else r for r in records])

    def test_missing_rank_or_payload_rejected(self):
        with self.assertRaises(ValueError): self.audit(self.rows[1:])
        with self.assertRaises(ValueError): self.audit([r for r in self.rows if r['payload']==0])

    def test_wrong_rows_and_drift_rejected(self):
        self.rows[0]['rank_rows']=7
        with self.assertRaises(ValueError): self.audit()
        self.rows[0]['rank_rows']=6; self.rows[0]['drift']=.2
        with self.assertRaises(ValueError): self.audit()

    def test_rejected_stable_round_not_a_faster_retry(self):
        extra=[r|dict(round=1) for r in self.rows if r['mode']=='fused' and r['payload']==0]
        with self.assertRaises(ValueError): self.audit(self.rows+extra)

    def test_missing_comparison_mode_rejected(self):
        with self.assertRaises(ValueError):
            self.audit([r for r in self.rows if r['mode']!='cublaslt_sequence_tuned'])

    def test_comparison_uses_strong_reference_not_fastest_repeat(self):
        def record(stamp,comm,fused,reference):
            config=dict(tile_n=128,tile_k=64,along_n=0,swizzle=1,comm=comm,compute=148-comm)
            return dict(run_id=stamp,case_index=0,source_id='source',environment_fingerprint='env',
                routing_ids={'dispatch':'route'},search=False,world=8,
                geometry=dict(h=64,f=96,experts=16,topk=2,target_rows=128),
                rows=[dict(mode=mode,direction='dispatch',config=config,valid=True,ms=ms,pflops=1/ms)
                      for mode,ms in [('fused',fused),('cublaslt_sequence_tuned',reference)]])
        first=record('01',20,2.,1.)
        repeat=record('02',20,3.,1.1)
        candidate=record('03',32,2.5,.9)
        rows=comparison_rows([first,repeat,candidate])
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['fused_run'],'03')
        self.assertEqual(rows[0]['reference_ms'],.9)
        self.assertAlmostEqual(rows[0]['retention'],.36)
        # Different native source or routing is never silently pooled.
        self.assertEqual(len(comparison_rows([first,candidate|dict(source_id='different')])),2)
        self.assertEqual(len(comparison_rows([first,candidate|dict(routing_ids={'dispatch':'other'})])),2)
        candidate['rows'][0]['valid']=False
        self.assertEqual(comparison_rows([first,candidate])[0]['fused_run'],'01')

    def test_catalog_preserves_missing_and_directional_oom(self):
        skip=dict(run_id='01',case_index=0,source_id='s',environment_fingerprint='e',
            search=False,skipped=True,reason='insufficient_memory',world=8,directions=['combine'],
            geometry=dict(h=4096,f=1536,experts=128,topk=8,target_rows=8192))
        rows=catalog_rows([skip])
        self.assertEqual(len(rows),17*15*2*2)
        missing=[r for r in rows if r['status']=='insufficient_memory']
        self.assertEqual(len(missing),1)
        self.assertEqual(missing[0]['direction'],'combine')
        self.assertEqual(missing[0]['models'],['qwen3_235b'])
        self.assertEqual(sum(r['status']=='not_measured' for r in rows),len(rows)-1)
        self.assertTrue(all(r['status']=='not_measured' for r in catalog_rows([skip|dict(search=True)])))
        with self.assertRaises(ValueError): catalog_rows([skip,skip|dict(source_id='different')])

    def test_full_plan_records_ep_and_token_fallbacks(self):
        # Loading the catalog also installs the benchmark catalog import path.
        catalog_rows([])
        from grouped_shapes import MODELS
        searches=[]; budgets=[]
        config=dict(tile_n=256,tile_k=64,along_n=1,swizzle=4,comm=0,compute=128)
        for i,profile in enumerate(dict.fromkeys(MODELS.values())):
            g=dict(zip(('h','f','experts','topk'),profile))|dict(target_rows=128)
            searches.append(dict(search=True,run_id=f'search-{i}',case_index=i,world=8,geometry=g,
                selected={d:dict(valid=True,config=config) for d in ('dispatch','combine')}))
            for d in ('dispatch','combine'):
                budgets.append(g|dict(direction=d,ep=8,config=config|dict(comm=40,compute=108),
                                      fused_run='budget',fused_case=i))
        plan=seed_fusion_plan(searches,budgets)
        self.assertEqual(len(plan),34)
        self.assertEqual(sum(len(b['entries']) for b in plan),1020)
        for b in plan:
            for r in b['entries']:
                self.assertEqual(r['status'],'seed_not_measured')
                self.assertEqual(r['gemm_seed']['ep'],8)
                self.assertEqual(r['gemm_seed']['target_rows'],128)
                self.assertEqual(r['policy'],'256,64,1,4,40,108')
                self.assertEqual(tuple(map(int,r['case'].split(',')[:4])),b['profile'])
        anchor=searches[0]|dict(run_id='new-anchor',geometry=searches[0]['geometry']|dict(target_rows=1024))
        anchor['selected']={d:dict(valid=True,config=config|dict(tile_n=128))
                            for d in ('dispatch','combine')}
        refined=seed_fusion_plan(searches+[anchor],budgets,eps=(8,),targets=(128,1024))
        self.assertEqual(refined[0]['entries'][0]['gemm_seed']['target_rows'],128)
        self.assertEqual(refined[0]['entries'][2]['gemm_seed']['target_rows'],1024)
        self.assertTrue(refined[0]['entries'][2]['policy'].startswith('128,'))
        with self.assertRaises(ValueError): seed_fusion_plan(searches[:-1],budgets)

    def test_plan_collection_requires_fetched_matching_batch(self):
        with tempfile.TemporaryDirectory(prefix='grouped-plan-test-') as tmp:
            root=Path(tmp); runs=root/'runs'; run=runs/'run'; run.mkdir(parents=True)
            plan=root/'plan.json'
            plan.write_text(json.dumps(dict(schema='grouped-bf16-full-seed-plan-v1',source_id='s',
                batches=[dict(ep=4,entries=[dict(argument='case@policy@dispatch')])])) )
            job=dict(run_id='run',source_id='s',world=4,experiment='grouped-full-seed-00-ep4',
                     grouped_cases='case@policy@dispatch',grouped_measure=True)
            (run/'job.json').write_text(json.dumps(job))
            with patch('summarize_sm103_grouped.audit_cases') as audit:
                self.assertEqual(collect_plan(plan,runs)['completed_batches'],0)
                audit.assert_not_called()
            (run/'fetched.json').write_text(json.dumps(dict(state='succeeded')))
            skip=dict(run_id='run',case_index=0,source_id='s',environment_fingerprint='e',
                search=False,skipped=True,reason='insufficient_memory',world=4,directions=['dispatch'],
                geometry=dict(h=4096,f=1536,experts=128,topk=8,target_rows=8192))
            with patch('summarize_sm103_grouped.audit_cases',return_value=[skip]):
                result=collect_plan(plan,runs)
                self.assertEqual(result['completed_batches'],1)
                self.assertEqual(result['coverage']['insufficient_memory'],1)
                with patch('summarize_sm103_grouped.audit_cases',side_effect=[
                        [skip],[skip|dict(run_id='refine')]]):
                    refined=collect_plan(plan,runs,[root/'refine'])
                self.assertEqual(refined['completed_batches'],1)
                self.assertEqual(refined['refinement_runs'],1)
                self.assertIn('bounded refinements',refined['note'])
                for bad in (skip,skip|dict(run_id='refine',source_id='different'),
                            skip|dict(run_id='refine',search=True),
                            skip|dict(run_id='refine',geometry=skip['geometry']|dict(target_rows=17))):
                    with patch('summarize_sm103_grouped.audit_cases',side_effect=[[skip],[bad]]):
                        with self.assertRaises(ValueError): collect_plan(plan,runs,[root/'refine'])
                (run/'job.json').write_text(json.dumps(job|dict(grouped_cases='different')))
                with self.assertRaises(ValueError): collect_plan(plan,runs)


if __name__=='__main__':
    unittest.main()
