import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import hashlib
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks/sm103/backward'))
import fused_baseline
spec=importlib.util.spec_from_file_location('backward_l20d',ROOT/'scripts/l20d.py')
l20d=importlib.util.module_from_spec(spec)
spec.loader.exec_module(l20d)


class BackwardBaselineTests(unittest.TestCase):
    def test_fused_replay_pairs_explicit_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory); search=output/'gemm-search';search.mkdir()
            c=next(r for r in fused_baseline.catalog.cases() if r['model']=='qwen25_72b' and
                   r['direction']=='oproj' and r['phase']=='dgrad' and r['seq']==131072 and r['cp']==8)
            winner=dict(tile_n=256,tile_k=64,epilogue_n=64,swizzle=8,along_m=True)
            row=dict(c,search=dict(status='passed',winner=winner,evidence='fixture'))
            (search/'summary.json').write_text(json.dumps(dict(rows=[row])))
            physical=dict(id='case',cp=8,m=c['m'],hidden=c['output_width'],q_heads=c['q_heads'],
                          kv_heads=8,head_dim=c['head_dim'],direction='oproj',aliases=[c['id']])
            (search/'plan.json').write_text(json.dumps(dict(cases=[physical])))
            with patch.object(fused_baseline,'run') as run:
                fused_baseline.fused_replay(output)
            plan=run.call_args.args[1];a,b=plan['batches'][0]['cases']
            self.assertEqual(plan['comm_ctas'],16)
            self.assertEqual((a['tile_n'],b['tile_n']),(128,256))
            for key in ('m','hidden','q_heads','head_dim','cp'): self.assertEqual(a[key],b[key])
            self.assertTrue(b['along_m'])
            self.assertEqual(b['swizzle'],8)

    def test_fused_backward_ready_follows_selected_tile(self):
        source=(ROOT/'csrc/operators/sm103/api/backward.cuh').read_text()
        self.assertIn('QkvGqaPackCommT<OprojBackwardKernelParams,128,TileN>',source)
        self.assertIn('ceil_div(normalized.gemm.n, BackwardTypes::kTileN)',source)
        self.assertIn('args.gemm.scheduler.max_swizzle_size=normalized.gemm.max_swizzle_size',source)
        self.assertIn('p.gemm_policy==BackwardGemmPolicy::kM128N256?256:128',source)

    def test_gemm_report_multiline_records_and_stable_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory); target=output/'gemm-search'; target.mkdir()
            c=fused_baseline.catalog.cases()[0]
            (target/'plan.json').write_text(json.dumps(dict(rows=[c])))
            (target/'cp4-00.json').write_text(json.dumps([dict(id='case',aliases=[c['id']])]))
            (target/'cp4-00.receipt.json').write_text(json.dumps(dict(run_id='test',exit_code=0)))
            control=output/'test/artifacts-attempt1/control'; control.mkdir(parents=True)
            (output/'test/fetched.json').write_text('{"attempt":1}')
            candidates=[]
            for i in range(48):
                samples=[1.0]*50 if i==0 else ([0.5]*25+[0.8]*25 if i==1 else [0.9]*50)
                candidates.append(dict(tile_n=128,tile_k=64,epilogue_n=0,along_m=False,swizzle=1,
                    compute_only=dict(samples_ms=samples,p50_ms=fused_baseline.statistics.median(samples))))
            f=control/'backward-case.json.gemm-sweep.jsonl'
            f.write_text('\n'.join(json.dumps(c,indent=2) for c in candidates))
            with patch.object(fused_baseline,'LOCAL',output): fused_baseline.gemm_search_report(output)
            r=json.loads((target/'summary.json').read_text())['rows'][0]['search']
            self.assertEqual(r['status'],'passed')
            self.assertEqual(r['winner']['compute_only']['p50_ms'],.9)
            self.assertTrue((target/'README.md').exists())

    def test_gemm_search_large_model_scope_and_isolation(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            manifest=fused_baseline.plan(output)
            with patch.object(fused_baseline.subprocess,'run',return_value=SimpleNamespace(returncode=1,stdout='',stderr='fixture')) as submit:
                with self.assertRaises(RuntimeError): fused_baseline.gemm_search(output,manifest)
            plan=json.loads((output/'gemm-search/plan.json').read_text())
            self.assertIn('llama31_70b',plan['models'])
            self.assertIn('deepseek_v3',plan['models'])
            self.assertTrue(all(r['seq']>=131072 for r in plan['rows']))
            self.assertTrue(any(r['route_issue'] for r in plan['rows']))
            self.assertEqual(plan['candidates'],48)
            self.assertEqual(plan['reserved_ctas'],16)
            self.assertIn('--backward-gemm-sweep',submit.call_args.args[0])
            self.assertNotIn('--auto-oproj-comm',submit.call_args.args[0])

    def test_retry_only_unstable_and_once(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            plan=fused_baseline.plan(output)
            case=next(c for c in plan['cases'] if not c['route_issue'])
            (output/'summary.json').write_text(json.dumps({'rows':[dict(case,status='unstable')]}))
            info=fused_baseline.retry_plan(output,plan)
            self.assertEqual(info['logical'],1)
            self.assertEqual(info['unique'],1)
            self.assertEqual(info['batches'],1)
            with self.assertRaises(ValueError): fused_baseline.retry_plan(output,plan)

    def test_independent_phases_keep_first_stable_not_fastest(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            case=fused_baseline.catalog.cases()[0]
            batches=[]
            for i,(b,w) in enumerate((([1.0]*50,[1.0]*25+[1.3]*25),([.8]*50,[1.2]*50))):
                rid='run'+str(i); folder=output/rid; control=folder/'artifacts-attempt1/control'
                control.mkdir(parents=True)
                (folder/'artifacts.tar.gz').write_bytes(b'fixture')
                (folder/'fetched.json').write_text(json.dumps(dict(attempt=1,artifact_sha256=hashlib.sha256(b'fixture').hexdigest())))
                (output/(rid+'.run.json')).write_text(json.dumps(dict(run_id=rid,exit_code=0)))
                data=dict(correctness='two_payload_full_gemm_and_exact_route',world_size=case['cp'],
                          b_mnk=[case[k] for k in ('m','n','k')],launch='graph',warmup=10,
                          weight_accumulation_beta=0,data_phase=dict(samples_ms=b),weight_phase=dict(samples_ms=w))
                (control/'backward-case.json').write_text(json.dumps(data))
                batches.append(dict(name=rid,cases=[dict(id='case',aliases=[case['id']])]))
            pure=output/'pure.json'; pure.write_text('{"results":{}}')
            with patch.object(fused_baseline,'LOCAL',output):
                fused_baseline.summarize(output,dict(cases=[case],batches=batches[:1]),pure)
                row=json.loads((output/'summary.json').read_text())['rows'][0]
                self.assertEqual(row['b_status'],'passed')
                self.assertIsNotNone(row['b_pflops'])
                self.assertIsNone(row['w_pflops'])
                fused_baseline.summarize(output,dict(cases=[case],batches=batches),pure)
                row=json.loads((output/'summary.json').read_text())['rows'][0]
                self.assertEqual(row['b_p50_ms'],1.0)
                self.assertEqual(row['w_p50_ms'],1.2)
                self.assertEqual(row['b_run_id'],'run0')
                self.assertEqual(row['w_run_id'],'run1')
                self.assertEqual(row['status'],'passed')
                self.assertEqual(row['half_drift'],0)

    def test_full_plan_retains_unsupported_and_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            plan=fused_baseline.plan(Path(directory))
            self.assertEqual(len(plan['cases']),540)
            supported={r['id'] for r in plan['cases'] if not r['route_issue']}
            measured=[alias for b in plan['batches'] for c in b['cases'] for alias in c['aliases']]
            self.assertEqual(set(measured),supported)
            self.assertEqual(len(measured),len(set(measured)))
            self.assertEqual(plan['warmup'],10)
            self.assertEqual(plan['samples'],50)
            self.assertTrue(all(len(b['cases'])<=32 for b in plan['batches']))

    def test_backward_harness_isolation_and_hash(self):
        job=dict(backward=True,mpi=True,files={'benchmarks/sm103/backward/validation.cuh':'a'})
        self.assertEqual(l20d.fused_binary(job).name,'backward_mpi_bench')
        self.assertIn('backward',str(l20d.fused_build_dir(job)))
        changed=job|dict(files={'benchmarks/sm103/backward/validation.cuh':'b'})
        self.assertNotEqual(l20d.fused_build_inputs(job),l20d.fused_build_inputs(changed))
        self.assertNotEqual(l20d.fused_build_dir(job),l20d.fused_build_dir(job|dict(profile=True)))
        self.assertTrue(str(l20d.fused_build_dir(job|dict(profile=True))).endswith('backward-profile'))

    def test_reverse_no_forward_autotune(self):
        source=(ROOT/'csrc/operators/sm103/api/backward.cuh').read_text()
        self.assertIn('cutlass::layout::ColumnMajor, cutlass::layout::RowMajor, Bf16',source)
        self.assertNotIn('recommended_a2a_gemm_comm_ctas',source)
        self.assertNotIn('recommended_gemm_a2a_comm_ctas',source)
        self.assertNotIn('detail::RoleTelemetryKernel<Production>',source)
        self.assertIn('GemmA2ARoleTelemetryKernel<BackwardReadyGemm, QkvBackwardPushComm, true, false>',source)
        self.assertIn('GemmA2ARoleTelemetryKernel<typename BackwardTypes::OutputGemm, OprojBackwardHeadComm, true>',source)


if __name__=='__main__': unittest.main()
