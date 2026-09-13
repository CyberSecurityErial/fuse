"""Host geometry/queue contracts for MXFP8 OProj, not CUDA correctness tests."""

import os
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import tarfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def function(text, signature):
    start = text.index(signature)
    # A profiling-only default argument may contain {}; skip the complete
    # parameter list before looking for the function body.
    cursor = start
    if '(' in signature:
        cursor = text.index('(', start) + 1
        depth = 1
        while depth:
            depth += (text[cursor] == '(') - (text[cursor] == ')')
            cursor += 1
    opening = text.index('{', cursor)
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


class Mxfp8OprojContracts(unittest.TestCase):
    def test_window_swizzle_reports_minor_extent_clipping(self):
        from summarize_sm103_mxfp8_fused import window_effective_swizzle
        config = dict(effective_swizzle_size='8', raster='along_m',
                      oproj_m_window_tiles=64, oproj_n_group_tiles=4)
        self.assertEqual(window_effective_swizzle(config), 4)
        self.assertEqual(window_effective_swizzle(dict(config, oproj_n_group_tiles=8)), 8)
        self.assertEqual(window_effective_swizzle(dict(config, raster='along_n')), 8)
        self.assertEqual(window_effective_swizzle(dict(config, oproj_m_window_tiles=2,
                                                       raster='along_n')), 2)
        for h, p in ((0, 0), (64, 0), (3, 4), (64, 3)):
            self.assertEqual(window_effective_swizzle(dict(config,
                oproj_m_window_tiles=h, oproj_n_group_tiles=p)), 8)
        self.assertEqual(config['effective_swizzle_size'], '8')

    def test_overlap_plan_changes_only_raster_and_preserves_aliases(self):
        import bench_sm103_mxfp8_oproj as bench
        from unittest.mock import patch
        rows=bench.matrix()
        controls=dict(rows=[dict(id=r['id'],gemm=bench.gemm_parameters(
            'm128n256k128e32s0sw4'+('M' if i%2 else 'N')),
            manual=dict(configuration=dict(comm_sm='64')))
            for i,r in enumerate(rows) if not r['status'].startswith('unsupported')])
        with patch.object(bench,'execute_job',side_effect=AssertionError('No GPU in host tests')):
            plan=bench.overlap_plan(rows,controls)
        self.assertEqual(len(plan),75)
        for task in plan:
            self.assertEqual(task['comm_ctas'],64)
            self.assertEqual(task['models'],next(r for r in rows if r['id']==task['id'])['models'])
            old=task['variants'][0]['gemm']
            self.assertEqual(len(task['variants']),2)
            other='along_m' if old['raster']=='along_n' else 'along_n'
            self.assertEqual(task['variants'][1]['gemm'],dict(old,raster=other))
        controls['rows'][0]['manual']['configuration']['comm_sm']='0'
        with self.assertRaisesRegex(ValueError,'explicit budget'):
            bench.overlap_plan(rows,controls)

    def test_overlap_audit_checks_budget_swizzle_and_source(self):
        import bench_sm103_mxfp8_oproj as bench
        from unittest.mock import patch
        row=dict(comm_ctas=64,m=16384,n=7168,k=16384,world=8,global_seq=131072)
        gemm=bench.gemm_parameters('m128n256k128e32s0sw4N')
        base=dict(row,source_id='frozen',epilogue_n=32,binary_sha256='binary',p50_ms=2.,
            configuration=dict(comm_sm='64',raster='along_n',tile_m='128',tile_n='256',tile_k='128',
                max_swizzle_size='4',scheduled_compute_ctas='84'))
        with patch.object(bench,'audit_run',return_value=base):
            result=bench.overlap_measurement('run',row,gemm,'frozen')
            self.assertEqual(result['serial_reference_saving_ms'],2.)
            with self.assertRaisesRegex(ValueError,'source/geometry'):
                bench.overlap_measurement('run',row,gemm,'different')
            base['configuration']['max_swizzle_size']='8'
            with self.assertRaisesRegex(ValueError,'configuration mismatch'):
                bench.overlap_measurement('run',row,gemm,'frozen')

    def test_overlap_table_does_not_fabricate_along_n_gains(self):
        import bench_sm103_mxfp8_oproj as bench
        row=dict(models=['one','alias'],world=8,global_seq=131072,comm_ctas=64,
            gemm=bench.gemm_parameters('m128n256k128e32s0sw4N'),measurements={})
        text=bench.overlap_markdown(dict(rows=[row]))
        self.assertIn('| one |',text)
        self.assertIn('| alias |',text)
        self.assertNotIn('0.000',text)
        self.assertIn('do not directly measure concurrent overlap',text)

    def test_overlap_table_distinguishes_same_budget_and_full_device_references(self):
        import bench_sm103_mxfp8_oproj as bench
        row=dict(models=['model'],world=8,global_seq=131072,comm_ctas=64,
            gemm=bench.gemm_parameters('m128n256k128e32s0sw4N'),
            pure_cublaslt=dict(status='passed',pflops=2.5),measurements=dict(current=dict(results=dict(
                fused=dict(pflops_per_rank=1.,p50_ms=2.),
                compute_reference=dict(pflops_per_rank=2.),producer_reference=dict(p50_ms=1.)))))
        text=bench.overlap_markdown(dict(rows=[row]))
        self.assertIn('full-148-SM',text)
        self.assertIn('| 2.500 | 50.0% | 40.0% |',text)
        row['pure_cublaslt']['status']='memory_skip'
        self.assertNotIn('40.0%',bench.overlap_markdown(dict(rows=[row])))

    def test_window_table_selects_measured_best_and_excludes_partial_rows_from_means(self):
        import bench_sm103_mxfp8_oproj as bench
        def sample(ms):
            return dict(run_id='test-only', results=dict(
                fused=dict(p50_ms=ms, pflops_per_rank=2/ms),
                compute_reference=dict(p50_ms=.5, pflops_per_rank=4.)))
        row = dict(id='one', models=['one','alias'], world=8, global_seq=131072,
            status='passed', baseline=sample(2), pure_cublaslt=dict(status='passed', pflops=5.),
            measurements={'20':sample(1.2), '32':sample(1), '48':sample(1.1)})
        partial = dict(row, id='partial', models=['partial'], status='running',
            measurements={'20':sample(.1)})
        missing = dict(row, id='missing', models=['missing'], status='pending',
            baseline=None, measurements={}, pure_cublaslt=dict(status='memory_skip'))
        text = bench.window_markdown(dict(rows=[row, partial, missing], budgets=[20,32,48]))
        self.assertIn('| 1.000 | 2.000 | +100.00% | 32 | 4.000 | 5.000 | 50.0% | 40.0% | 3/3 |', text)
        self.assertIn('| missing | 128K | 8 | — | — | — | — | — | — | — | — | 0/3 |', text)
        self.assertIn('Fully covered: 1/3 physical points.', text)
        self.assertIn('F=2.000 PFLOPS; F/C=50.0%.', text)
        self.assertIn('not a same-binary ablation', text)
        self.assertIn('| alias |', text)

    def test_lookahead_adapter_preserves_real_wait_and_current_stage_release(self):
        source = (ROOT/'csrc/operators/sm103/detail/cutlass_pipeline.cuh').read_text()
        adapter = function(source, 'struct Mxfp8OprojMainloop')
        load = function(adapter, 'auto load(')
        mma = function(adapter, 'auto mma(')
        self.assertIn('return Base::load(', load)
        self.assertIn('return Base::mma(', mma)
        self.assertLess(load.index('pipeline.producer_acquire(state, token)'), load.index('copy(this->observed_tma_load_a_'))
        self.assertLess(load.index('copy(this->observed_tma_load_sfb_'), load.rindex('pipeline.producer_try_acquire(state)'))
        self.assertLess(mma.index('pipeline.consumer_wait(state, token)'), mma.index('copy(copyA'))
        self.assertLess(mma.index('copy(copyB'), mma.index('acc_pipeline.producer_acquire(acc_state)'))
        self.assertLess(mma.index('cute::gemm('), mma.index('pipeline.consumer_release(current)'))
        self.assertLess(mma.index('pipeline.consumer_release(current)'), mma.rindex('pipeline.consumer_try_wait(state'))
        self.assertNotIn('__syncthreads', adapter)
        self.assertNotIn('fence', load + mma)
        self.assertIn('step(cute::true_type{})', mma)
        self.assertIn('step(cute::false_type{})', mma)

    def test_finalize_does_not_accept_partial_coverage(self):
        import bench_sm103_mxfp8_oproj as bench
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)
            for name, data in [('acceptance-current',dict(rows=[])),('manual-current',{}),
                               ('model-current',{}),('pure-current',{})]:
                (directory/(name+'.json')).write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'full runnable matrix'):
                bench.finalize(directory,bench.matrix())

    def test_final_table_keeps_missing_values_and_model_aliases(self):
        import bench_sm103_mxfp8_oproj as bench
        rows=bench.matrix()
        table=bench.acceptance_markdown(rows,dict(rows=[]))
        self.assertNotIn('0.000',table)
        self.assertIn('llama31_70b',table)
        self.assertIn('qwen25_72b',table)
        self.assertEqual(len([l for l in table.splitlines() if l.startswith('| ')]),
                         1+sum(len(r['models']) for r in rows))
        missing=next(r for r in rows if r['status'].startswith('unsupported'))
        pure=dict(rows=[dict(id=f"m{missing['m']}n{missing['n']}k{missing['k']}",status='passed',pflops=2.)])
        table=bench.acceptance_markdown(rows,dict(rows=[]),pure)
        line=next(l for l in table.splitlines() if l.startswith(
            f"| qwen25_7b | {missing['global_seq']//1024}K | 8 |"))
        self.assertNotIn('2.000',line)

    def test_service_export_reaudits_receipts_and_rejects_fused_times(self):
        import copy
        import bench_sm103_mxfp8_oproj as bench
        g=bench.gemm_parameters('m128n256k128e32s0sw2N')
        anchor=dict(id='anchor',m=16384,n=4096,k=8192,world=8,gemm=g)
        samples=[]
        for c in (32,64,96):
            services={name:dict(run_id='test',candidate_id=c,component=name,
                m=16384,n=4096,k=8192,world=8,global_seq=131072,epilogue_n=32,
                p50_ms=.5,source_id='source',binary_sha256='binary',environment_fingerprint='env',
                configuration=dict(comm_sm=str(c),tile_m='128',tile_n='256',tile_k='128',
                    max_swizzle_size='2',raster='along_n',effective_swizzle_size='2',dynamic_smem='220160'))
                for name in bench.SERVICE_NAMES}
            samples.append(dict(comm_ctas=c,services=services))
        row=dict(anchor,reference_m=16384,samples=samples); row.pop('m')
        report=dict(rows=[row]); calls=[]
        def audit(folder,c,name):
            calls.append((c,name)); return samples[c//32-1]['services'][name]
        points,identity=bench.calibration_points(report,[anchor],audit)
        self.assertEqual(len(calls),9); self.assertEqual(len(points),3)
        self.assertEqual(points[0]['compute_us'],500)
        self.assertEqual(points[0]['compute_ctas'],116)
        bad=copy.deepcopy(report); bad['rows'][0]['samples'][0]['services']['fused']={}
        with self.assertRaisesRegex(ValueError,'Only independent'):
            bench.calibration_points(bad,[anchor],audit)
        bad=copy.deepcopy(report); bad['rows'][0]['samples'][0]['services']['compute_reference']['p50_ms']=.4
        with self.assertRaisesRegex(ValueError,'differs from original'):
            bench.calibration_points(bad,[anchor],audit)

    def test_acceptance_keeps_preselected_manual_and_real_auto(self):
        import bench_sm103_mxfp8_oproj as bench
        row=next(r for r in bench.matrix() if not r['status'].startswith('unsupported'))
        row=dict(row,gemm=bench.gemm_parameters('m128n256k128e32s0sw8M'),requires_cross_peer_k_tile=False)
        manual=dict(gemm=row['gemm'],winner=dict(configuration=dict(comm_sm='48')))
        command=bench.acceptance_command(row,manual,'/home/work/workspace_wct','frozen')
        self.assertEqual(command[command.index('--comm-sm-list')+1],'48')
        self.assertIn('--auto-mxfp8-comm',command)
        self.assertEqual(command[command.index('--source-run')+1],'frozen')
        self.assertNotIn('--calibrate',command)
        self.assertNotIn('--quick',command)

    def test_actual_selector_matches_resources_and_preserves_overrides(self):
        compiler=shutil.which('c++')
        if not compiler: self.skipTest('C++ compiler unavailable')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'cutlass').mkdir()
            (root/'cutlass/cutlass.h').write_text('#pragma once\n#define CUTLASS_HOST_DEVICE\n#define CUTLASS_HOST\n')
            (root/'cutlass/fast_math.h').write_text('''#pragma once
#include <cstdint>
namespace cutlass { struct FastDivmodU64 {
  uint64_t divisor=1;
  FastDivmodU64()=default;
  explicit FastDivmodU64(uint64_t d):divisor(d) {}
  uint64_t divide(uint64_t x) const { return x/divisor; }
}; }
''')
            source=root/'selector.cpp'; binary=root/'selector'
            source.write_text('#include <cassert>\n#include "' + str(ROOT/'csrc/operators/sm103/detail/autotune.cuh') + '"\n' + r'''
int main() {
  using namespace fuse::detail;
  Mxfp8OprojTuningRequest r;
  r.m=16384; r.n=4096; r.k=8192; r.world=8; r.dynamic_smem_bytes=196608;
  Mxfp8OprojCalibrationPoint p[3];
  for (int i=0;i<3;++i) {
    p[i].n=r.n; p[i].k=r.k; p[i].world=r.world; p[i].raster=r.raster;
    p[i].reference_m=r.m; p[i].dynamic_smem_bytes=r.dynamic_smem_bytes;
    p[i].comm_ctas=32*(i+1); p[i].compute_ctas=148-p[i].comm_ctas;
  }
  p[0].compute_us=100; p[0].copy_us=150; p[0].producer_us=170;
  p[1].compute_us=120; p[1].copy_us=80; p[1].producer_us=100;
  p[2].compute_us=160; p[2].copy_us=60; p[2].producer_us=80;
  auto plan=select_mxfp8_oproj_plan(r,p,3);
  assert(plan.status==Mxfp8OprojTuningStatus::Success && plan.comm_ctas==64);
  // A small producer deficit costs less than taking extra SMs from GEMM.
  // Feed feasibility is diagnostic, not an additional hard constraint.
  p[0].producer_us=101;
  assert(select_mxfp8_oproj_plan(r,p,3).comm_ctas==32);
  assert(!select_mxfp8_oproj_plan(r,p,3).feed_feasible);
  p[0].producer_us=170;
  r.m*=4; assert(select_mxfp8_oproj_plan(r,p,3).comm_ctas==64);
  r.comm_ctas=32; assert(select_mxfp8_oproj_plan(r,p,3).comm_ctas==32);
  r.comm_ctas=64; assert(select_mxfp8_oproj_plan(r,p,1).status==Mxfp8OprojTuningStatus::UnsupportedCalibration);
  r.comm_ctas=0; r.raster=0;
  assert(select_mxfp8_oproj_plan(r,p,3).status==Mxfp8OprojTuningStatus::UnsupportedCalibration);
  r.raster=1; r.dynamic_smem_bytes+=256;
  assert(select_mxfp8_oproj_plan(r,p,3).status==Mxfp8OprojTuningStatus::UnsupportedCalibration);
  assert(select_mxfp8_oproj_plan(r,nullptr,0).status==Mxfp8OprojTuningStatus::UnsupportedCalibration);
}
''')
            subprocess.run([compiler,'-std=c++17','-fsanitize=undefined','-I'+str(root),str(source),'-o',str(binary)],check=True)
            subprocess.run([str(binary)],check=True)

    def test_offline_service_balance_has_no_fused_winner_input(self):
        compiler = shutil.which('c++')
        if not compiler: self.skipTest('C++ compiler unavailable')
        header = (ROOT/'csrc/operators/sm103/detail/performance_model.cuh').read_text()
        model = header[header.index('struct Mxfp8OprojServices'):header.index('enum class OProjModelRaster')]
        with tempfile.TemporaryDirectory() as folder:
            src, binary = Path(folder)/'model.cpp', Path(folder)/'model'
            src.write_text('#include <algorithm>\n#include <cmath>\n#include <cstdint>\n#include <cassert>\n' + model + r'''
int main() {
  Mxfp8OprojServices s{16384,148,64,84,100,80,130};
  auto a=score_mxfp8_oproj_bulk(16384,s); assert(a.valid && a.score_us==130);
  auto b=score_mxfp8_oproj_bulk(65536,s);
  assert(b.valid && b.compute_finish_us==400 && b.producer_finish_us==370);
  // A mixed producer can be faster than standalone copy. Do not invent Q=P-R.
  s.producer_us=60; b=score_mxfp8_oproj_bulk(32768,s);
  assert(b.valid && b.producer_finish_us==140);
  assert(!score_mxfp8_oproj_bulk(8192,s).valid);
  assert(!score_mxfp8_oproj_bulk(65664,s).valid);
  s.compute_ctas=148; assert(!score_mxfp8_oproj_bulk(16384,s).valid);
  s.compute_ctas=84; s.copy_us=NAN; assert(!score_mxfp8_oproj_bulk(16384,s).valid);
}
''')
            subprocess.run([compiler,'-std=c++17','-fsanitize=undefined',str(src),'-o',str(binary)],check=True)
            subprocess.run([str(binary)],check=True)

    def test_oproj_services_keep_preparation_outside_compute(self):
        api=(ROOT/'csrc/operators/sm103/api/forward_mxfp8.cuh').read_text()
        body=function(api,'cudaError_t launch_mxfp8_oproj(')
        self.assertIn('using PureGemm = typename Binding::Types::PureGemm',body)
        self.assertIn('CopyReferenceKernel<Comm, Kernel, true>',body)
        self.assertIn('if constexpr (Operation == Mxfp8OprojOperation::kCopy) comm.weights.source = nullptr',body)
        harness=(ROOT/'benchmarks/sm103/fused_bf16.cu').read_text()
        validation=function(harness,'void validate(')
        self.assertIn('launch_a2a_gemm_mxfp8_compute_reference',validation)
        self.assertNotIn('launch_a2a_gemm_mxfp8_producer_reference',validation)

    def test_manual_plan_keeps_pure_winner_and_all_targets(self):
        import bench_sm103_mxfp8_oproj as bench
        rows = bench.matrix()
        # K256 may straddle a K128-aligned peer shard: explicitly flag the
        # integration requirement rather than replace the winner with K128.
        row = next(r for r in rows if 'qwen25_7b' in r['models'] and r['world'] == 4)
        result = dict(schema='sm103_mxfp8_cutlass_search_v1', partial=False, pending=[],
            run_id='test', artifact_sha256='0'*64, rows=[dict(m=row['m'],n=row['n'],k=row['k'],
                status='passed',winner=dict(config='m128n128k256e32s2sw8N'))])
        unsupported = next(r for r in rows if r['status'].startswith('unsupported'))
        plan = bench.manual_plan([row,unsupported], result)
        first = plan['rows'][0]
        self.assertEqual(first['gemm']['tile_n'],128)
        self.assertEqual(first['gemm']['tile_k'],256)
        self.assertEqual(first['gemm']['max_swizzle_size'],8)
        self.assertTrue(first['requires_cross_peer_k_tile'])
        self.assertEqual(first['tuning_status'],'comm_budget_first')
        self.assertNotIn('candidates',first)
        self.assertEqual([c['comm_ctas'] for c in first['proposed_candidates']],[8,16,32,48,64])
        self.assertTrue(all(set(c) == {'comm_ctas'} for c in first['proposed_candidates']))
        self.assertEqual(plan['rows'][1]['tuning_status'],'unsupported_head_partition')
        with self.assertRaises(ValueError): bench.comm_budget_command(first, '/work')
        first['gemm'] = bench.gemm_parameters('m128n256k128e64s0sw4M')
        first['requires_cross_peer_k_tile'] = False
        command = bench.comm_budget_command(first, '/work')
        self.assertEqual(command[command.index('--comm-sm-list')+1], '8,16,32,48,64')
        self.assertEqual(command[command.index('--mxfp8-epilogue-n')+1], '64')
        self.assertEqual(command[command.index('--oproj-raster')+1], 'along_m')
        self.assertEqual(command[command.index('--max-swizzle-size')+1], '4')
        result['partial'] = True
        with self.assertRaises(ValueError): bench.manual_plan([row], result)

    def test_baseline_matrix_aliases_and_missing_rows(self):
        import bench_sm103_mxfp8_oproj as bench
        rows = bench.matrix()
        keys = [(r['m'], r['n'], r['k'], r['world'], r['head_dim']) for r in rows]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(any({'llama31_70b','qwen25_72b'} <= set(r['models']) for r in rows))
        self.assertTrue(all(r['global_seq'] >= 131072 for r in rows))
        unsupported = [r for r in rows if r['status'].startswith('unsupported')]
        self.assertEqual(len(unsupported), 3)
        self.assertTrue(all(r['world'] == 8 and 'qwen25_7b' in r['models'] for r in unsupported))
        self.assertNotIn('0.000', bench.render(rows))

    def test_pure_reference_contract(self):
        import bench_sm103_mxfp8_oproj as bench
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            control = root/'control'
            control.mkdir()
            shape = dict(id='test', m=128, n=256, k=128)
            job = dict(run_id='test', gemm_precision='mxfp8', gemm_matrix_payload=dict(shapes=[shape]))
            contract=dict(measurement='single_gpu_pure_cublaslt',math_sms=0,precision='mxfp8',
                sm_budget_enforcement='full_device_no_restriction',measured_ranks=1,shapes=[shape])
            plan=dict(precision=32,math_sms=0,graph_tuning=1,beta=0,transpose_x=0,transpose_w=0,valid=1)
            lines = ['config,pure_mxfp8,launch=graph,warmup=10,samples=50,sm_budget=full_device,'
                     'compute_ctas=0,device_sms=148,includes_quantization=0,includes_communication=0,'
                     'output=bf16,group_k=32,scale=ue8m0']
            lines.append('plan,pure_mxfp8,id=test,config='+json.dumps(plan))
            for generation in (0,1):
                for operand, count in ((0,16384),(1,32768)):
                    lines.append(f'input,pure_mxfp8,id=test,generation={generation},operand={operand},'
                                 f'count={count},rms=0.1,min=-0.1,max=0.1')
            for generation, phase in ((0,'pre'),(0,'post'),(1,'pre')):
                lines.append(f'correctness,pure_mxfp8,id=test,generation={generation},phase={phase},'
                             'checked=32768,mismatches=0,nonfinite=0')
            lines.append('samples,pure_mxfp8,id=test,round=0,ms='+json.dumps([1.]*50))
            lines.append('RESULT '+json.dumps(dict(shape, status='passed',p50_ms=1.,pflops=8388608/1e12,plan=plan)))
            def save():
                (control/'job.json').write_text(json.dumps(job))
                (control/'gemm-probe-contract.json').write_text(json.dumps(contract))
                (control/'attempt1.log').write_text('\n'.join(lines))
                with tarfile.open(root/'artifacts.tar.gz','w:gz') as archive:
                    archive.add(control,arcname='control')
                digest=hashlib.sha256((root/'artifacts.tar.gz').read_bytes()).hexdigest()
                (root/'fetched.json').write_text(json.dumps(dict(state='succeeded',exit_code=0,artifact_sha256=digest)))
            save()
            self.assertEqual(bench.audit_pure(root)['rows'][0]['p50_ms'],1.)
            lines[0]=lines[0].replace(',compute_ctas=0','')
            save()
            self.assertEqual(bench.audit_pure(root)['rows'][0]['p50_ms'],1.)
            contract['math_sms']=132
            save()
            with self.assertRaises(Exception): bench.audit_pure(root)
            contract['math_sms']=0
            lines[0]=lines[0].replace('device_sms=148','device_sms=132')
            save()
            with self.assertRaises(Exception): bench.audit_pure(root)

    def test_actual_geometry_and_queue_ownership(self):
        compiler = shlex.split(os.environ.get('CXX', 'c++'))
        if not compiler or not shutil.which(compiler[0]):
            self.skipTest('host C++ compiler required')
        gemm = (ROOT / 'csrc/operators/sm103/detail/gemm.cuh').read_text()
        comm = (ROOT / 'csrc/operators/sm103/detail/a2a_gemm.cuh').read_text()
        comm = comm[comm.index('struct Mxfp8A2ALhsInputComm'):]
        order = (ROOT / 'csrc/operators/sm103/detail/producer_consumer.cuh').read_text()
        order = '\n'.join(function(order, 'struct ' + name) + ';' for name in
            ('NBandSwizzle', 'ProducerTileOrder', 'OprojTileOrder', 'A2AInputTileOrder'))
        helpers = '\n'.join(function(gemm, name) for name in (
            '__host__ __device__ constexpr int64_t a_row_stride',
            '__host__ __device__ constexpr int64_t b_row_stride',
            '__host__ __device__ constexpr int64_t d_row_stride',
            'inline bool supported_problem', 'inline bool supported_mxfp8_problem'))
        geometry = function(comm, 'static bool supported_geometry')
        program = r'''
#include "fuse/layout/gemm.h"
#include "fuse/layout/ulysses.h"
#include <algorithm>
#include <climits>
#include <cstdint>
#include <stdexcept>
#include <vector>
#define __host__
#define __device__
#define CUTLASS_HOST_DEVICE
#define CUTLASS_HOST
namespace cutlass { struct FastDivmodU64 {
  uint64_t divisor=1;
  FastDivmodU64()=default;
  explicit FastDivmodU64(uint64_t d):divisor(d) {}
  uint64_t divide(uint64_t x) const { return x/divisor; }
}; }
using namespace fuse;
constexpr int kAlignment = 8, kMaxWorldSize = 8;
constexpr int kReadyBlockM = 128, kTileK = 128;
void check(bool ok) { if (!ok) throw std::runtime_error("contract failure"); }
''' + helpers + '\n' + geometry + '\n' + order + r'''
int main() {
  for (int world : {1,2,4,8}) for (int peer_k : {128,384,512,2048,4096,32768}) {
    GemmProblem g; g.m=512; g.n=1024; g.k=peer_k*world;
    UlyssesRoute r; r.world_size=world; r.rank=world-1; r.seq_local=512;
    r.global_seq=512*world; r.local_heads=peer_k/128; r.q_heads=r.local_heads*world;
    r.head_dim=128; r.kind=RouteKind::kHeadToSequence; r.direction=RouteDirection::kInverse;
    check(supported_geometry(g,r));
    r.causal_load_balanced=true; check(supported_geometry(g,r));
    r.seq_local=128; r.global_seq=128*world; g.m=128;
    check(!supported_geometry(g,r)); // causal jump bisects a ready block
    r.causal_load_balanced=false; check(supported_geometry(g,r));
    g.stride_a.row=g.k+8; check(!supported_geometry(g,r)); g.stride_a.row=-1;
    r.packed_row_granularity=128; check(!supported_geometry(g,r));
  }
  // Execute the production queue for both rasters, padding, swizzles and
  // budgets. Parallel workers may complete out of order, but must cover each
  // complete-shard chunk exactly once. No timing/throughput is simulated.
  for (bool along_n : {false,true}) for (int log : {0,1,2,3})
  for (int mt : {1,3,17,33}) for (int nt : {1,3,16})
  for (int world : {4,8}) for (int comm : {8,16,32,48,64,96}) for (int chunks : {1,4,8,16}) {
    const int width=1<<log, pm=(mt+width-1)/width*width, pn=(nt+width-1)/width*width;
    A2AInputTileOrder q; q.m_tiles=mt; q.n_tiles=along_n?pn:nt;
    q.along_n=along_n; q.log_swizzle=log; q.compute_ctas=std::min(148-comm,pm*pn);
    q.group_stride=uint64_t(along_n?pn:pm)*width;
    q.ready_group_m_tiles=std::max(1,comm*4/chunks);
    const int count=mt*world*chunks;
    std::vector<int> seen(count), arrivals(mt*world);
    for(int worker=comm*4-1;worker>=0;--worker) {
      for(int task=worker;task<count;task+=comm*4) {
        auto t=q.decode(task,world,chunks);
        check(t.m>=0 && t.m<mt && t.peer>=0 && t.peer<world && t.chunk>=0 && t.chunk<chunks);
        int unit=t.m*world+t.peer;
        check(++seen[unit*chunks+t.chunk]==1);
        check(++arrivals[unit]<=chunks);
      }
    }
    for(int n:seen) check(n==1);
    for(int n:arrivals) check(n==chunks);
  }
}
'''
        with tempfile.TemporaryDirectory() as folder:
            binary = str(Path(folder) / 'contracts')
            build = subprocess.run(
                [*compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                 '-fsanitize=undefined', '-fno-sanitize-recover=all',
                 '-I', str(ROOT / 'include'), '-x', 'c++', '-', '-o', binary],
                input=program, text=True, capture_output=True, timeout=60)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([binary], capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stderr)

    def test_scale_word_copy_matches_scalar_layout(self):
        compiler = shlex.split(os.environ.get('CXX', 'c++'))
        if not compiler or not shutil.which(compiler[0]):
            self.skipTest('host C++ compiler required')
        source = (ROOT / 'csrc/operators/sm103/detail/a2a_gemm.cuh').read_text()
        begin = source.index('      const int words_per_row = peer_k / 128;')
        loop = source[begin:source.index('      // Wait for the G2S path', begin)]
        # Independent scalar layout oracle: CUTLASS K-major SfAtom has
        # ((32,4),(32,4)) shape and ((16,4),(0,1)) byte strides.
        program = r'''
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>
struct Coord { int r,k; };
namespace cute { Coord make_coord(int r,int k,int) { return {r,k}; } }
struct Layout {
  int k;
  int operator()(Coord c) const {
    return (c.r/128)*(k/128)*512+(c.k/128)*512+
        (c.r%32)*16+((c.r%128)/32)*4+(c.k%128)/32;
  }
};
struct Args {
  int comm_rows;
  Layout source_scales, destination_scales;
  struct { uint8_t* sfa; } workspace;
  struct { const uint8_t* scales; } activation[8];
};
void transfer(Args a,int peer_k,int source_row,int row,int peer,int lane) {
  struct { int peer; } t{peer};
''' + loop + r'''
}
int main() {
  for(int stage_bytes:{24*1024,48*1024})
  for(int peer_k:{128,384,1024,2048,8192,32768}) for(int world:{2,4,8}) {
    if(peer_k>stage_bytes) continue;
    int rows=128;
    while(rows*peer_k>stage_bytes) rows/=2;
    Layout src{peer_k},dst{peer_k*world};
    std::vector<uint32_t> input(3*peer_k), actual(2*peer_k*world), expected(actual.size());
    for(size_t i=0;i<input.size();++i) input[i]=uint32_t(i*2654435761u);
    auto* bytes=reinterpret_cast<const uint8_t*>(input.data());
    for(int chunk=0;chunk<128;chunk+=rows) for(int peer=0;peer<world;++peer) {
      std::fill(actual.begin(),actual.end(),0xa5a5a5a5u); expected=actual;
      int source_row=256+chunk,row=128+chunk;
      auto* out=reinterpret_cast<uint8_t*>(expected.data());
      for(int r=0;r<rows;++r) for(int k=0;k<peer_k;k+=32)
        out[dst({row+r,peer*peer_k+k})]=bytes[src({source_row+r,k})];
      Args a{rows,src,dst,{reinterpret_cast<uint8_t*>(actual.data())},{}};
      a.activation[peer].scales=bytes;
      for(int lane=0;lane<32;++lane) transfer(a,peer_k,source_row,row,peer,lane);
      if(actual!=expected) throw std::runtime_error("SFA word copy changed bytes or ownership");
    }
  }
}
'''
        with tempfile.TemporaryDirectory() as folder:
            binary = str(Path(folder) / 'scale-copy')
            build = subprocess.run([*compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                '-fsanitize=address,undefined', '-fno-sanitize-recover=all', '-x', 'c++', '-', '-o', binary],
                input=program, text=True, capture_output=True, timeout=60)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([binary], capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stderr)

    def test_precision_and_publication_boundaries(self):
        source = (ROOT / 'csrc/operators/sm103/detail/a2a_gemm.cuh').read_text()
        source = source[source.index('struct Mxfp8A2ALhsInputComm'):]
        self.assertIn('Mxfp8WeightProducer weights(', source)
        self.assertNotIn('quantize_mxfp8_chunk(', source)  # shared producer, no copied quantizer
        self.assertIn('a.source_scales(cute::make_coord(source_row + r, k, 0))', source)
        self.assertIn('a.destination_scales(cute::make_coord(row + r, t.peer * peer_k + k, 0))', source)
        publish = source.index('ready.fetch_add(1, cuda::memory_order_acq_rel)')
        self.assertLess(source.index('detail::tma_store_wait_all()'), publish)
        self.assertLess(source.index('a.activation[peer].scales + src'), publish)
        quant = function(source, 'if (warp >= kA2ALhsBulkSlots)')
        self.assertIn('weights.drain();', quant)
        self.assertIn('return;', quant)
        self.assertNotIn('weights.progress()', source)
        self.assertNotIn('weights.drain()', source[source.index('auto* stage ='):])
        self.assertNotIn('__threadfence(', source)
        api = (ROOT / 'csrc/operators/sm103/api/forward_mxfp8.cuh').read_text()
        start = api.index('cudaError_t launch_mxfp8_oproj(')
        launch = function(api[start:], 'cudaError_t launch_mxfp8_oproj(')
        self.assertNotIn('quantize_gemm_a2a_mxfp8_activation', launch)
        self.assertNotIn('resolve_oproj_communication', launch)
        self.assertNotIn('resolve_mxfp8_qkv_communication', launch)
        header = (ROOT / 'include/fuse/operators/primitives/a2a_gemm_mxfp8.h').read_text()
        self.assertNotIn('#include "fuse/operators/primitives/gemm_a2a', header)

    def test_weight_cohort_executes_dense_exclusive_worker_mapping(self):
        compiler = shutil.which('c++')
        if not compiler: self.skipTest('host C++ compiler required')
        source = (ROOT/'csrc/operators/sm103/detail/a2a_gemm.cuh').read_text()
        source = source[source.index('struct Mxfp8A2ALhsInputComm'):]
        constants = source[source.index('  static constexpr int kReadyBlockM'):source.index('  static constexpr size_t')]
        body = function(source, 'CUTLASS_DEVICE void operator()')
        prologue = body[body.index('{')+1:body.index('    auto* stage =')]
        program = r'''
#include <cassert>
#include <vector>
struct { int x; } threadIdx;
struct Params { struct { int route; } params; int weights=0, producer_order=0; };
struct Mxfp8WeightProducer {
  static inline std::vector<int> seen;
  static inline int workers=0, drains=0;
  struct Swizzle {};
  Mxfp8WeightProducer(int,int,Swizzle,int id,int count) {
    assert(count==workers && id>=0 && id<count);
    assert(++seen[id]==1);
  }
  void drain() { ++drains; }
};
''' + constants + '\nvoid run(const Params& a, int comm_id, int comm_ctas) {\n' + prologue + r'''
}
int main() {
  for (int comm : {1,3,8,16,32,48,64,96,147}) {
    auto& seen=Mxfp8WeightProducer::seen;
    Mxfp8WeightProducer::workers=comm*kWeightQuantWarps;
    Mxfp8WeightProducer::drains=0; seen.assign(comm*kWeightQuantWarps,0);
    for(int c=comm-1;c>=0;--c) for(int warp=0;warp<kMinThreads/32;++warp) {
      threadIdx.x=warp*32;
      int before=Mxfp8WeightProducer::drains;
      run(Params{},c,comm);
      assert(Mxfp8WeightProducer::drains-before==(warp>=kA2ALhsBulkSlots));
    }
    for(int n:seen) assert(n==1);
  }
}
'''
        with tempfile.TemporaryDirectory() as folder:
            binary=str(Path(folder)/'cohort')
            build=subprocess.run([compiler,'-std=c++17','-fsanitize=undefined',
                '-x','c++','-','-o',binary],input=program,text=True,capture_output=True)
            self.assertEqual(build.returncode,0,build.stderr)
            subprocess.run([binary],check=True,timeout=30)

    def test_auto_rejects_stale_producer_services_but_keeps_explicit_budget(self):
        source=(ROOT/'csrc/operators/sm103/api/policy.cuh').read_text()
        body=function(source,'inline cudaError_t resolve_mxfp8_oproj_communication(')
        self.assertIn('Mxfp8A2ALhsInputComm::kWeightProducerRevision) return cudaErrorNotSupported;',body)
        self.assertLess(body.index('if (params.projection.num_comm_ctas > 0) return cudaSuccess;'),
                        body.index('kMxfp8OprojCalibrationProducerRevision'))
        self.assertLess(body.index('kMxfp8OprojCalibrationProducerRevision'),body.index('device_info('))


if __name__ == '__main__':
    unittest.main()
