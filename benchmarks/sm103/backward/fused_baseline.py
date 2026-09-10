#!/usr/bin/env python3
"""Full BF16 reverse-route baseline through the standard Mac/mc/screen runner.

One MPI process group handles a batch of shapes; identical geometry+route aliases
are measured once. Unsupported routing stays in the full table, never replaced
by a forward or standalone GEMM number. B and W are measured separately, so their
sum is not labelled a directly measured immediate B+W latency.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import shutil
import sys

import backward_shape_bench as catalog

REPO = Path(__file__).resolve().parents[3]
LOCAL = REPO.parent / 'fuse_midfile' / 'l20d'


def plan(output):
    rows = [r for r in catalog.cases() if r['phase'] == 'dgrad']
    groups = {}
    for row in rows:
        if row['route_issue']:
            continue
        key = (row['direction'], row['cp'], row['seq'], row['input_width'],
               row['output_width'], row['q_heads'], row['kv_heads'], row['head_dim'])
        groups.setdefault(key, []).append(row)
    batches = []
    for cp in catalog.CPS:
        cases = []
        for key, aliases in groups.items():
            row = aliases[0]
            if row['cp'] != cp:
                continue
            cases.append(dict(id=row['id'].replace('.dgrad',''), direction=row['direction'],
                m=row['seq']//cp, hidden=row['input_width'] if row['direction']=='qkv' else row['output_width'],
                q_heads=row['q_heads'], kv_heads=row['kv_heads'] or cp, head_dim=row['head_dim'],
                aliases=[r['id'] for r in aliases]))
        # Small-to-large reduces time to first results; bounded batches preserve
        # earlier results if a later geometry reports a real correctness failure.
        cases.sort(key=lambda r:(r['m'],r['direction'],r['hidden']))
        for start in range(0,len(cases),32):
            name=f'cp{cp}-{start//32+1:02d}'
            batch=cases[start:start+32]
            (output/(name+'.json')).write_text(json.dumps(batch,indent=2)+'\n')
            batches.append(dict(name=name,cp=cp,cases=batch))
    result=dict(schema='sm103_fused_backward_plan_v1',cases=rows,batches=batches,
                comm_ctas=16,tile=[128,128,64],launch='graph',warmup=10,samples=50,
                causal=True,weight_beta=0,includes_cp_gradient_reduction=False)
    (output/'plan.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def retry_plan(output, manifest):
    """One explicitly requested remeasurement; never retry stable/OOM rows."""
    if any(b['name'].startswith('retry1-') for b in manifest['batches']):
        raise ValueError('This retry round is already planned; run it instead')
    summary=json.loads((output/'summary.json').read_text())
    targets={(r['model'],r['projection'],r['seq'],r['cp']) for r in summary['rows'] if r['status']=='unstable'}
    ids={r['id'] for r in manifest['cases'] if (r['model'],r['projection'],r['seq'],r['cp']) in targets}
    added=[]
    for cp in catalog.CPS:
        selected=[c for b in manifest['batches'] if b['cp']==cp for c in b['cases'] if ids.intersection(c['aliases'])]
        for start in range(0,len(selected),32):
            name=f'retry1-cp{cp}-{start//32+1:02d}'
            batch=selected[start:start+32]
            (output/(name+'.json')).write_text(json.dumps(batch,indent=2)+'\n')
            added.append(dict(name=name,cp=cp,cases=batch))
    manifest['batches'].extend(added)
    manifest['retry_policy']='one_requested_round; retain_first_stable_per_phase_not_fastest'
    (output/'plan.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return dict(logical=len(ids),unique=sum(len(b['cases']) for b in added),batches=len(added))


def run(output, manifest):
    for batch in manifest['batches']:
        receipt=output/(batch['name']+'.run.json')
        if receipt.exists():
            previous=json.loads(receipt.read_text())
            if previous['exit_code']==0:
                continue
            raise RuntimeError('Previous failed batch requires diagnosis: '+batch['name'])
        command=[sys.executable,str(REPO/'scripts/l20d.py'),'run','fused-smoke',
            '--node','09','--backward','--mpi','--world',str(batch['cp']),
            '--fused-direction','qkv','--fused-launch','graph','--causal','--comm-sm','16',
            '--backward-matrix',str(output/(batch['name']+'.json')),
            '--experiment','v19-backward-'+batch['name'],'--timeout','3600']
        print(f'BATCH {batch["name"]} cases={len(batch["cases"])}',flush=True)
        proc=subprocess.Popen(command,cwd=REPO,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        run_id=None
        with (output/(batch['name']+'.log')).open('w') as log:
            for line in proc.stdout:
                log.write(line); log.flush()
                match=re.search(r'"run_id": "([0-9]{8}-[0-9]{6}-[a-f0-9]+)"',line)
                if match: run_id=match.group(1)
                if '"state"' in line or line.startswith(('BACKWARD','FETCHED')):
                    print(line.rstrip(),flush=True)
        code=proc.wait()
        receipt.write_text(json.dumps(dict(run_id=run_id,exit_code=code))+'\n')
        if code:
            raise RuntimeError('Backward batch failed; inspect '+str(receipt))
        print('DONE '+batch['name'],flush=True)


def summarize(output, manifest, pure):
    pure=json.loads(Path(pure).read_text())['results']
    found={}
    for batch in manifest['batches']:
        receipt=output/(batch['name']+'.run.json')
        if not receipt.exists(): continue
        run_id=json.loads(receipt.read_text())['run_id']
        folder=LOCAL/run_id
        fetched=json.loads((folder/'fetched.json').read_text())
        if hashlib.sha256((folder/'artifacts.tar.gz').read_bytes()).hexdigest()!=fetched['artifact_sha256']:
            raise ValueError('Artifact hash mismatch')
        control=folder/f'artifacts-attempt{fetched["attempt"]}'/'control'
        for case in batch['cases']:
            path=control/('backward-'+case['id']+'.json')
            if not path.exists(): continue
            data=json.loads(path.read_text())
            for alias in case['aliases']: found.setdefault(alias,[]).append((data,path,run_id))
    rows=[]
    lines=['# SM103 BF16 融合反向基线','',
        'Graph 10+50；随机 BF16、FP32 累加、beta=0、因果双块路由；固定16通信CTA、128×128×64 tile。',
        'B=完整融合数据梯度；W=本地权重梯度，不含跨CP归约。纯GEMM为满SM cuBLASLt。',
        '吞吐单位 PFLOPS/GPU；B/W独立判断漂移，分别保留首个稳定轮（不是最快轮）；每阶段证据见CSV。',
        '不支持和显存不足保留空白；某阶段漂移超过5%只隐藏该阶段，不连带隐藏另一阶段。','',
        '| 模型/投影 | S | CP | B融合 | B纯GEMM | B/纯 | W本地 | W纯GEMM | 状态 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for c in manifest['cases']:
        row=dict(model=c['model'],projection=c['projection'],seq=c['seq'],cp=c['cp'],
                 status=c['route_issue'] or 'pending',b_pflops=None,w_pflops=None)
        bp=pure.get(c['id'],{}).get('pflops')
        wp=pure.get(c['id'].replace('.dgrad.','.wgrad.'),{}).get('pflops')
        row.update(b_pure_pflops=bp,w_pure_pflops=wp)
        for data,path,run_id in found.get(c['id'],[]):
            row.update(evidence=str(path),run_id=run_id)
            if 'status' in data:
                if row['status']=='pending': row['status']=data['status']
            else:
                if (data['correctness']!='two_payload_full_gemm_and_exact_route' or
                    data['world_size']!=c['cp'] or data['b_mnk']!=[c['m'],c['n'],c['k']] or
                    data['launch']!='graph' or data['warmup']<10 or data['weight_accumulation_beta']!=0):
                    raise ValueError('Wrong backward result contract: '+str(path))
                for phase,key in (('data_phase','b'),('weight_phase','w')):
                    samples=data[phase]['samples_ms']
                    if len(samples)!=50 or not all(math.isfinite(x) and x>0 for x in samples):
                        raise ValueError('Incomplete backward samples')
                    median=statistics.median(samples)
                    d=abs(statistics.median(samples[:25])-statistics.median(samples[25:]))/median
                    if row.get(key+'_status')!='passed':
                        row[key+'_status']='passed' if d<=.05 else 'unstable'
                        row[key+'_half_drift']=d
                        row[key+'_evidence']=str(path)
                        row[key+'_run_id']=run_id
                        if d<=.05:
                            row[key+'_p50_ms']=median
                            row[key+'_pflops']=2*c['m']*c['n']*c['k']/median/1e12
                row['half_drift']=max(row[k+'_half_drift'] for k in ('b','w'))
                row['status']='passed' if all(row.get(k+'_status')=='passed' for k in ('b','w')) else 'unstable'
        fmt=lambda x:'—' if x is None else f'{x:.3f}'
        ratio=row['b_pflops']/bp if row['b_pflops'] and bp else None
        row['b_to_fullsm_gemm']=ratio
        lines.append(f'| {c["model"]}/{c["projection"]} | {c["seq"]//1024}K | {c["cp"]} | {fmt(row["b_pflops"])} | {fmt(bp)} | '+
                     ('—' if ratio is None else f'{ratio:.1%}')+f' | {fmt(row["w_pflops"])} | {fmt(wp)} | '+
                     (f'B:{row["b_status"]}, W:{row["w_status"]}' if 'b_status' in row else row['status'])+' |')
        rows.append(row)
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (output/'fused.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader();writer.writerows(rows)
    summary=dict(total=len(rows),states=dict(Counter(r['status'] for r in rows)),rows=rows)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return {k:v for k,v in summary.items() if k!='rows'}


def gemm_search(output, manifest, retry_unstable=False):
    """Large-model dgrad NN search, isolated from the fused baseline report.

    Keep CP in the dedup key: a four-rank maximum is not an eight-rank maximum.
    Special routes are retained as unsupported, never synthesized to run GEMM.
    Weight gradients are not part of the profiled data-gradient bottleneck.
    """
    models=set(catalog.projections.SOURCES)-{'kimi_linear_48b'}
    models.update(('llama31_70b','llama31_405b','kimi_k3_kda','production_qwen_dense'))
    rows=[r for r in manifest['cases'] if r['model'] in models and r['seq']>=131072]
    selected={r['id'] for r in rows if not r['route_issue']}
    target=output/'gemm-search'
    target.mkdir(exist_ok=True)
    retry_ids=None
    if retry_unstable:
        retry_plan=target/'retry1-plan.json'
        if retry_plan.exists():
            retry_ids=set(json.loads(retry_plan.read_text())['ids'])
        else:
            summary=json.loads((target/'summary.json').read_text())
            retry_ids={r['id'] for r in summary['rows'] if r['search']['status']=='unstable'}
            retry_plan.write_text(json.dumps(dict(ids=sorted(retry_ids),rounds=1))+'\n')
    unique={}
    for batch in manifest['batches']:
        for c in batch['cases']:
            aliases=sorted(selected.intersection(c['aliases']))
            if not aliases: continue
            key=(batch['cp'],c['direction'],c['m'],c['hidden'],c['q_heads'],c['kv_heads'],c['head_dim'])
            unique.setdefault(key,(batch['cp'],dict(c,aliases=aliases)))
    plan=dict(scope='large_models_dgrad_NN_only',models=sorted(models),rows=rows,
              cases=[dict(c,cp=cp) for cp,c in unique.values()],candidates=48,
              warmup=10,samples=50,reserved_ctas=16)
    (target/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    batches=[]
    for cp in catalog.CPS:
        cases=[c for p,c in unique.values() if p==cp]
        if retry_ids is not None: cases=[c for c in cases if retry_ids.intersection(c['aliases'])]
        cases.sort(key=lambda c:(c['m'],c['hidden']))
        for start in range(0,len(cases),8): batches.append((cp,cases[start:start+8]))
    print(f'GEMM SEARCH logical={len(rows)} supported={len(selected)} physical={len(unique)} batches={len(batches)} candidates=48',flush=True)
    for index,(cp,cases) in enumerate(batches):
        name=f'cp{cp}-'+('retry1-' if retry_unstable else '')+f'{index:02d}'
        matrix=target/(name+'.json'); receipt=target/(name+'.receipt.json')
        matrix.write_text(json.dumps(cases)+'\n')
        if receipt.exists():
            if json.loads(receipt.read_text())['exit_code']==0: continue
            raise RuntimeError('Diagnose failed search batch: '+str(receipt))
        command=[sys.executable,str(REPO/'scripts/l20d.py'),'run','fused-smoke',
            '--node','09','--backward','--mpi','--profile','--backward-gemm-sweep',
            '--world',str(cp),'--fused-direction','qkv','--fused-launch','graph',
            '--causal','--comm-sm','16','--backward-matrix',str(matrix),
            '--experiment','v19-gemm-search-'+name,'--timeout','3600']
        print(f'GEMM BATCH {index+1}/{len(batches)} {name} shapes={len(cases)}',flush=True)
        result=subprocess.run(command,cwd=REPO,capture_output=True,text=True)
        (target/(name+'.log')).write_text(result.stdout+result.stderr)
        match=re.search(r'"run_id": "([0-9]{8}-[0-9]{6}-[a-f0-9]+)"',result.stdout)
        rid=match.group(1) if match else None
        receipt.write_text(json.dumps(dict(run_id=rid,exit_code=result.returncode))+'\n')
        if result.returncode or not rid: raise RuntimeError('Search failed: '+str(target/(name+'.log')))
        gemm_search_report(output)


def gemm_search_report(output):
    target=output/'gemm-search'
    plan=json.loads((target/'plan.json').read_text())
    found={}
    for receipt in sorted(target.glob('cp*.receipt.json')):
        r=json.loads(receipt.read_text())
        if r['exit_code']: continue
        folder=LOCAL/r['run_id']; fetched=json.loads((folder/'fetched.json').read_text())
        control=folder/f'artifacts-attempt{fetched["attempt"]}'/'control'
        for c in json.loads(receipt.with_name(receipt.name.replace('.receipt.json','.json')).read_text()):
            f=control/('backward-'+c['id']+'.json.gemm-sweep.jsonl')
            if not f.exists():
                state=json.loads((control/('backward-'+c['id']+'.json')).read_text()).get('status','missing')
                row=dict(status=state)
            else:
                # The shared C++ summary writer pretty-prints each object;
                # parse the object stream, not individual physical lines.
                payload=f.read_text(); decoder=json.JSONDecoder(); candidates=[]
                while payload.strip():
                    payload=payload.lstrip()
                    value,end=decoder.raw_decode(payload)
                    candidates.append(value); payload=payload[end:]
                if len(candidates)!=48: raise ValueError('Incomplete sweep: '+str(f))
                for v in candidates:
                    samples=v['compute_only']['samples_ms']; a=statistics.median(samples[:25]); b=statistics.median(samples[25:])
                    v['half_drift']=abs(a-b)/max(min(a,b),1e-12)
                stable=[v for v in candidates if v['half_drift']<=.05]
                base=candidates[0]
                best=min(stable,key=lambda v:v['compute_only']['p50_ms']) if stable else None
                row=dict(status='passed' if best and base['half_drift']<=.05 else 'unstable',
                         baseline=base,winner=best,evidence=str(f))
                if row['status']=='passed':
                    gain=base['compute_only']['p50_ms']/best['compute_only']['p50_ms']-1
                    print(f'GEMM DONE {c["id"]} {gain:+.1%} N{best["tile_n"]} K{best["tile_k"]} E{best["epilogue_n"]} sw{best["swizzle"]} along={"M" if best["along_m"] else "N"}',flush=True)
            for alias in c['aliases']:
                # Keep the first stable paired round, not the fastest across
                # rounds. Failed/unstable retries cannot erase valid evidence.
                if alias not in found or (found[alias]['status']!='passed' and row['status']=='passed'):
                    found[alias]=row
    rows=[dict(c,search=found.get(c['id'],dict(status=c['route_issue'] or 'pending'))) for c in plan['rows']]
    (target/'summary.json').write_text(json.dumps(dict(rows=rows),indent=2)+'\n')
    lines=['# Large-model backward GEMM search','',
        'Only dgrad NN GEMM; fixed 132 compute CTAs / reserved16, no communication in timed kernels.',
        'Graph10+50 per candidate; complete numerical check for each candidate. Production overlap/defaults unchanged.',
        'Winner means minimum observed stable p50 in 48 candidates, not globally optimal or independently remeasured.',
        'Half-median drift >5% is excluded. Both baseline and winner must be stable to report a gain.',
        'If explicitly retried, retain the first stable paired round; no cross-round cherry-picking.',
        'Special routes and insufficient memory are explicit gaps, not successful coverage. PFLOPS per GPU.', '',
        '| Model / projection | S | CP | Original | Winner | Gain | N/K/E / along / swizzle | Status |',
        '|---|---:|---:|---:|---:|---:|---|---|']
    for c in rows:
        s=c['search']; a=b=g=config='—'
        if s['status']=='passed':
            base=s['baseline']['compute_only']['p50_ms']; best=s['winner']; t=best['compute_only']['p50_ms']
            flops=2*c['m']*c['n']*c['k']/1e12
            a=f'{flops/base:.3f}'; b=f'{flops/t:.3f}'; g=f'{base/t-1:+.1%}'
            config=f'{best["tile_n"]}/{best["tile_k"]}/{best["epilogue_n"]} / {"M" if best["along_m"] else "N"} / {best["swizzle"]}'
        lines.append(f'| {c["model"]} / {c["projection"]} | {c["seq"]//1024}K | {c["cp"]} | {a} | {b} | {g} | {config} | {s["status"]} |')
    (target/'README.md').write_text('\n'.join(lines)+'\n')
    print('GEMM STATUS '+str(dict(Counter(r['search']['status'] for r in rows))),flush=True)


def fused_replay(output, retry_unstable=False):
    """Paired original/winner fused B measurements; no new tuning search."""
    target=output/'gemm-fused'
    target.mkdir(exist_ok=True)
    path=target/'plan.json'
    if path.exists():
        manifest=json.loads(path.read_text())
    else:
        search=json.loads((output/'gemm-search/summary.json').read_text())['rows']
        stable={r['id']:r for r in search if r['search']['status']=='passed'}
        source=json.loads((output/'gemm-search/plan.json').read_text())
        pairs=[]; batches=[]
        for cp in catalog.CPS:
            cases=[]
            for c in source['cases']:
                aliases=[a for a in c['aliases'] if a in stable]
                if c['cp']!=cp or not aliases: continue
                winner=stable[aliases[0]]['search']['winner']
                if winner['tile_k']!=64 or winner['tile_n']!=256 or winner['epilogue_n'] not in (0,64):
                    raise ValueError('Winner is not a registered fused binding')
                pair=dict(id=c['id'],cp=cp,aliases=aliases,winner=winner,
                          gemm_evidence=stable[aliases[0]]['search']['evidence'])
                pairs.append(pair)
                cases.extend([dict(c,id=c['id']+'.before',tile_n=128,epilogue_n=0,swizzle=1,along_m=False),
                    dict(c,id=c['id']+'.after',**{k:winner[k] for k in ('tile_n','epilogue_n','swizzle','along_m')})])
            for start in range(0,len(cases),8):
                name=f'cp{cp}-fused-{start//8:02d}'; batch=cases[start:start+8]
                (target/(name+'.json')).write_text(json.dumps(batch,indent=2)+'\n')
                batches.append(dict(name=name,cp=cp,cases=batch))
        manifest=dict(pairs=pairs,batches=batches,rows=search,comm_ctas=16,warmup=10,samples=50,
                      boundary='fused_data_gradient_B_not_B_plus_W',selection='per_shape_offline_GEMM_winner')
        path.write_text(json.dumps(manifest,indent=2)+'\n')
    if retry_unstable and not any('retry1' in b['name'] for b in manifest['batches']):
        rows=json.loads((target/'summary.json').read_text())['rows']
        ids={r['id'] for r in rows if r['fused_comparison']['status']=='unstable'}
        physical={p['id'] for p in manifest['pairs'] if ids.intersection(p['aliases'])}
        added=[]
        for cp in catalog.CPS:
            cases=[c for b in manifest['batches'] if b['cp']==cp for c in b['cases']
                   if c['id'].rsplit('.',1)[0] in physical]
            for start in range(0,len(cases),8):
                name=f'cp{cp}-retry1-fused-{start//8:02d}'; batch=cases[start:start+8]
                (target/(name+'.json')).write_text(json.dumps(batch,indent=2)+'\n')
                added.append(dict(name=name,cp=cp,cases=batch))
        manifest['batches'].extend(added)
        path.write_text(json.dumps(manifest,indent=2)+'\n')
    run(target,manifest)
    fused_replay_report(output)


def fused_replay_report(output):
    target=output/'gemm-fused'; manifest=json.loads((target/'plan.json').read_text()); found={}
    for b in manifest['batches']:
        f=target/(b['name']+'.run.json')
        if not f.exists(): continue
        r=json.loads(f.read_text())
        if r['exit_code']: continue
        folder=LOCAL/r['run_id']; fetched=json.loads((folder/'fetched.json').read_text())
        control=folder/f'artifacts-attempt{fetched["attempt"]}'/'control'
        for c in b['cases']:
            p=control/('backward-'+c['id']+'.json'); d=json.loads(p.read_text())
            if 'status' not in d and (
                d['kernel_traits']['tile_n']!=c['tile_n'] or d['kernel_traits']['tile_k']!=64 or
                d['gemm_epilogue_n']!=c['epilogue_n'] or d['gemm_swizzle']!=c['swizzle'] or
                d['gemm_along_m']!=c['along_m'] or d['comm_ctas']!=16 or d['world_size']!=b['cp']):
                raise ValueError('Measured fused configuration differs from replay plan: '+c['id'])
            found.setdefault(b['name'],{})[c['id']]=dict(data=d,evidence=str(p))
    paired={}
    for p in manifest['pairs']:
        result=dict(status='pending',winner=p['winner'])
        for batch in found.values():
            a,b=batch.get(p['id']+'.before'),batch.get(p['id']+'.after')
            if not a or not b: continue
            candidate=dict(winner=p['winner'],before=a,after=b)
            if any('status' in v['data'] for v in (a,b)):
                candidate['status']='skipped_memory_precheck'
            else:
                for v in (a,b):
                    d=v['data']
                    if d['correctness']!='two_payload_full_gemm_and_exact_route' or d['warmup']<10 or d['launch']!='graph':
                        raise ValueError('Invalid fused measurement contract')
                    samples=d['data_phase']['samples_ms']
                    if len(samples)!=50: raise ValueError('Expected 50 samples')
                    first=statistics.median(samples[:25]);last=statistics.median(samples[25:])
                    v['p50_ms']=statistics.median(samples);v['half_drift']=abs(first-last)/min(first,last)
                candidate['status']='passed' if max(a['half_drift'],b['half_drift'])<=.05 else 'unstable'
                candidate['gain']=a['p50_ms']/b['p50_ms']-1
            if result['status']=='pending' or candidate['status']=='passed': result=candidate
            if result['status']=='passed': break
        for alias in p['aliases']: paired[alias]=result
    rows=[];lines=['# Backward fused GEMM-winner A/B','',
        'Same-round Graph10+50, fixed comm16, full numerical/routing checks on two payloads.',
        'Only B data-gradient fusion is compared; W unchanged. Per-shape GEMM winner is explicit, not runtime autotune.',
        'A/B drift above5% is diagnostic, not a stable performance claim. Missing/unstable GEMM-search points keep the old configuration and are not re-launched.', '',
        'Retain the first stable paired round; never pair an old baseline with a new winner measurement.', '',
        '| Model / projection | S | CP | Before PFLOPS | After PFLOPS | Change | Status |',
        '|---|---:|---:|---:|---:|---:|---|']
    for c in manifest['rows']:
        s=paired.get(c['id'],dict(status='unchanged_'+c['search']['status']))
        r=dict(c,fused_comparison=s);rows.append(r);a=b=g='—'
        if s['status']=='passed':
            flops=2*c['m']*c['n']*c['k']/1e12
            a=f'{flops/s["before"]["p50_ms"]:.3f}';b=f'{flops/s["after"]["p50_ms"]:.3f}';g=f'{s["gain"]:+.1%}'
            print(f'FUSED {c["id"]} {a}->{b} PFLOPS {g}',flush=True)
        lines.append(f'| {c["model"]} / {c["projection"]} | {c["seq"]//1024}K | {c["cp"]} | {a} | {b} | {g} | {s["status"]} |')
    (target/'summary.json').write_text(json.dumps(dict(rows=rows),indent=2)+'\n')
    (target/'README.md').write_text('\n'.join(lines)+'\n')
    print('FUSED STATUS '+str(dict(Counter(r['fused_comparison']['status'] for r in rows))),flush=True)


def profile_cases(output, manifest, case_ids):
    """Explicit bounded diagnostics, kept separate from the formal table."""
    candidates={c['id']:(b['cp'],c) for b in manifest['batches'] for c in b['cases']}
    if not case_ids or len(set(case_ids))!=len(case_ids) or any(i not in candidates for i in case_ids):
        raise ValueError('Select unique existing physical case IDs')
    target=output/'profile'
    target.mkdir(exist_ok=True)
    for ident in case_ids:
        cp,c=candidates[ident]
        receipt=target/(ident+'.receipt.json')
        if receipt.exists():
            raise ValueError('Profile already exists: '+str(receipt))
        command=[sys.executable,str(REPO/'scripts/l20d.py'),'run','fused-smoke',
            '--node','09','--backward','--mpi','--profile','--world',str(cp),
            '--fused-direction',c['direction'],'--fused-launch','graph','--causal','--comm-sm','16',
            '--seq-local',str(c['m']),'--hidden',str(c['hidden']),
            '--q-heads',str(c['q_heads']),'--kv-heads',str(c['kv_heads']),'--head-dim',str(c['head_dim']),
            '--experiment','v19-profile-'+ident,'--timeout','600']
        print('PROFILE '+ident,flush=True)
        result=subprocess.run(command,cwd=REPO,capture_output=True,text=True)
        (target/(ident+'.log')).write_text(result.stdout+result.stderr)
        match=re.search(r'"run_id": "([0-9]{8}-[0-9]{6}-[a-f0-9]+)"',result.stdout)
        if result.returncode or not match:
            raise RuntimeError('Profile failed; inspect '+str(target/(ident+'.log')))
        rid=match.group(1)
        fetched=json.loads((LOCAL/rid/'fetched.json').read_text())
        control=LOCAL/rid/f'artifacts-attempt{fetched["attempt"]}'/'control'
        trace=control/'backward-perfetto.json'
        parsed=json.loads(trace.read_text())
        if not parsed.get('traceEvents'): raise ValueError('Empty profile')
        shutil.copy2(trace,target/(ident+'.json'))
        shutil.copy2(control/'backward-perfetto.json.gemm.json',target/(ident+'.gemm.json'))
        receipt.write_text(json.dumps(dict(run_id=rid,case=c,cp=cp,control=str(control),
            trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest()),indent=2)+'\n')
        print('PROFILE DONE '+ident+' '+str(target/(ident+'.json')),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('plan','retry-plan','run','report','profile','gemm-search','gemm-report','gemm-retry','fused-replay','fused-report','fused-retry'))
    parser.add_argument('--case-ids',help='comma-separated existing physical IDs for profiling')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pure',type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.action=='plan':
        if (args.output/'plan.json').exists(): raise ValueError('Plan already exists')
        manifest=plan(args.output)
        print(json.dumps(dict(logical=len(manifest['cases']),batches=len(manifest['batches']),
              unique=sum(len(b['cases']) for b in manifest['batches']),
              unsupported=sum(bool(r['route_issue']) for r in manifest['cases']))))
    elif args.action=='retry-plan': print(json.dumps(retry_plan(args.output,json.loads((args.output/'plan.json').read_text()))))
    elif args.action=='run': run(args.output,json.loads((args.output/'plan.json').read_text()))
    elif args.action=='gemm-search': gemm_search(args.output,json.loads((args.output/'plan.json').read_text()))
    elif args.action=='gemm-retry': gemm_search(args.output,json.loads((args.output/'plan.json').read_text()),True)
    elif args.action=='gemm-report': gemm_search_report(args.output)
    elif args.action=='fused-replay': fused_replay(args.output)
    elif args.action=='fused-retry': fused_replay(args.output,True)
    elif args.action=='fused-report': fused_replay_report(args.output)
    elif args.action=='profile': profile_cases(args.output,json.loads((args.output/'plan.json').read_text()),
                                             (args.case_ids or '').split(','))
    else: print(json.dumps(summarize(args.output,json.loads((args.output/'plan.json').read_text()),args.pure)))
