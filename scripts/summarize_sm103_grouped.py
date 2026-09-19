#!/usr/bin/env python3
"""Audit fetched grouped measurements; compare complete workloads, not rank averages.

Prints compact JSON to stdout. No GPU/cloud access or implicit result overwrite.
Search winners use the mean of both payload p50s and only stable verified data.
"""
import argparse
import bisect
from collections import Counter, defaultdict
import csv
import io
import json
import math
from pathlib import Path
import re
import statistics
import sys
import tarfile

from l20d import grouped_build_inputs, grouped_case_geometry, grouped_policy, grouped_batch_cases
from summarize_sm103_fused import (require, digest, file_digest, read_bytes,
    json_bytes, safe_member)


def read_run(directory, case_index=None):
    root = Path(directory).resolve()
    if (root / 'control').is_dir():
        root = root.parent
    receipt = json_bytes(read_bytes(root / 'fetched.json', root))
    require(receipt['state'] == 'succeeded' and receipt['phase'] == 'finished' and
            receipt['exit_code'] == receipt['work_exit_code'] == 0 and
            not receipt.get('collection_error'), 'Run did not finish successfully')
    require(receipt['stage'] == 'grouped-ep', 'Not a grouped EP run')
    archive = root / 'artifacts.tar.gz'
    require(file_digest(archive) == receipt['artifact_sha256'], 'Archive hash mismatch')
    attempt = int(receipt['attempt'])
    folder = root / f'artifacts-attempt{attempt}'
    if case_index is not None:
        require(isinstance(case_index,int) and 0 <= case_index < 64, 'Invalid batch case index')
    csv_name = f'grouped-case-{case_index:04d}.csv' if case_index is not None else 'grouped-samples.csv'
    required = {'job.json', 'source-installed.json', 'environment.json',
                'grouped-contract.json', f'attempt{attempt}.log'}
    data, seen = {}, set()
    with tarfile.open(archive, 'r:gz') as stream:
        for member in stream:
            name = safe_member(member.name)
            require(name not in seen and (member.isdir() or member.isfile()), 'Invalid archive member')
            seen.add(name)
            if name not in {'control/' + item for item in required | {csv_name}}:
                continue
            require(member.isfile() and member.size <= 64 << 20, 'Oversized evidence')
            actual = read_bytes(folder / name, root)
            require(actual == stream.extractfile(member).read(), 'Extracted evidence changed')
            data[name.split('/', 1)[1]] = actual
    require(required <= set(data), 'Missing grouped evidence')
    job = json_bytes(data['job.json'])
    installed = json_bytes(data['source-installed.json'])
    contract = json_bytes(data['grouped-contract.json'])
    env = json_bytes(data['environment.json'])
    for field in ('run_id', 'source_id', 'node'):
        require(job[field] == receipt[field], f'Receipt mismatch: {field}')
    require(installed['source_id'] == job['source_id'] and installed['files'] == job['files'],
            'Installed source differs from job')
    require(digest(json.dumps(job['files'], sort_keys=True).encode()) == job['source_id'],
            'Source manifest hash mismatch')
    require(contract['build']['inputs'] == grouped_build_inputs(job), 'Build input mismatch')
    require(contract['build']['environment_fingerprint'] == env['fingerprint'] ==
            receipt['environment_fingerprint'], 'Environment mismatch')
    require(re.fullmatch('[0-9a-f]{64}', contract['build']['binary_sha256']) is not None,
            'Missing binary attestation')
    require(job.get('grouped_measure') and contract['performance_measured'], 'Not a timed run')
    batch = grouped_batch_cases(job)
    if batch:
        require(case_index is not None and case_index < len(batch), 'Select a valid batch case index')
        require(len(batch)==len(contract['cases']), 'Batch manifest length mismatch')
        for expected,recorded in zip(batch,contract['cases']):
            require(all(expected[k]==recorded[k] for k in ('index','case','policy','direction')),
                    'Batch manifest mismatch')
        case = batch[case_index]
        text = data[f'attempt{attempt}.log'].decode()
        markers = list(re.finditer(r'^CASE grouped index=(\d+) state=(\w+)(.*)$',text,re.M))
        selected = [m for m in markers if int(m[1])==case_index]
        require(selected, 'Missing batch case state')
        status = selected[-1][2]
        if status=='skipped':
            reason = dict(re.findall(r'(\w+)=([^ ]+)',selected[-1][3])).get('reason')
            require(reason in ('insufficient_memory','cuda_oom'), 'Unexpected skip reason')
            data['_skip_reason'] = reason
        else:
            require([m[2] for m in selected]==['running','completed'], 'Incomplete/repeated batch case')
            data[f'attempt{attempt}.log'] = text[selected[0].end():selected[-1].start()].encode()
        job = job | dict(grouped_cases=None,grouped_case=case['case'],grouped_policy=case['policy'],
                         grouped_direction=case['direction'],grouped_buffer_rows=case.get('buffer_rows',0),
                         grouped_tail_balance=case.get('tail_balance',False))
        contract = contract | dict(geometry=case['geometry'])
        receipt = receipt | dict(case_index=case_index)
    else:
        require(case_index is None, 'Case index provided for a single-case run')
    if '_skip_reason' not in data:
        require(csv_name in data, 'Missing sample CSV')
        data['grouped-samples.csv'] = data[csv_name]
    geometry = grouped_case_geometry(job)
    # Memory-estimator margins may evolve without changing an old workload.
    # Audit the exact logical geometry, not today's incidental allocation note.
    for key in ('h','f','experts','topk','target_rows','tokens_per_rank','expert_row_capacity'):
        require(geometry[key] == contract['geometry'][key], f'Geometry mismatch: {key}')
    return root, receipt, job, geometry, data


def audit_samples(text, world, geometry, expert_rows, search, directions=('dispatch','combine'),fused_only=False,transport_compare=False,compute_compare=False):
    grouped = defaultdict(list)
    for row in csv.DictReader(io.StringIO(text)):
        config = tuple(int(row[k]) for k in
                       ('tile_n', 'tile_k', 'along_n', 'swizzle', 'comm_ctas', 'compute_ctas'))
        key = row['direction'], row['mode'], config, int(row['payload']), int(row['round'])
        require(row['direction'] in directions and int(row['payload']) in (0, 1),
                'Unexpected direction/payload')
        require(int(row['ep']) == world and int(row['h']) == geometry['h'] and
                int(row['f']) == geometry['f'] and int(row['total_experts']) == geometry['experts'] and
                int(row['topk']) == geometry['topk'] and int(row['tokens_per_rank']) == geometry['tokens_per_rank'],
                'Raw sample workload mismatch')
        grouped[key].append(row)
    modes = defaultdict(list)
    for (direction, mode, config, payload, round_id), rows in grouped.items():
        pairs = {(int(r['sample']), int(r['rank'])) for r in rows}
        require(len(rows) == 50 * world and pairs == {(s, r) for s in range(50) for r in range(world)},
                'Missing/duplicate rank samples')
        # The complete (sample,rank) set was checked above. Aggregate each row
        # once instead of rescanning all ranks' CSV rows for every sample.
        maximum = [0.0] * 50
        expected_rows = [sum(expert_rows[direction, payload, rank]) for rank in range(world)]
        for row in rows:
            sample, rank = int(row['sample']), int(row['rank'])
            require(int(row['rank_rows']) == expected_rows[rank], 'Row count mismatch')
            ms = float(row['ms'])
            require(math.isfinite(ms) and ms > 0 and int(row['warmup']) >= 10, 'Invalid sample')
            maximum[sample] = max(maximum[sample], ms)
        first, last = statistics.median(maximum[:25]), statistics.median(maximum[25:])
        drift = abs(first-last) / max(first, last)
        require(all(abs(float(r['drift']) - drift) <= 1e-6 for r in rows), 'Incorrect drift')
        flags = {int(r['accepted']) for r in rows}
        require(len(flags) == 1 and flags <= {0, 1}, 'Inconsistent acceptance')
        accepted = flags == {1}
        require(accepted == (drift <= .05), 'Stable round rejected or drifting round accepted')
        modes[direction, mode, config, payload].append(dict(round=round_id, accepted=accepted,
            p50=statistics.median(maximum), drift=drift))
    results = defaultdict(dict)
    for (direction, mode, config, payload), rounds in modes.items():
        rounds.sort(key=lambda r: r['round'])
        require(1 <= len(rounds) <= 3 and [r['round'] for r in rounds] == list(range(len(rounds))),
                'Invalid retry sequence')
        require(not any(r['accepted'] for r in rounds[:-1]), 'Retried an accepted stable round')
        require(rounds[-1]['accepted'] or len(rounds) == 3, 'Stopped retrying early')
        results[direction, mode, config][payload] = rounds
    external = search == 'external'
    require(not transport_compare or (fused_only and not search),'Invalid transport comparison mode')
    require(not compute_compare or transport_compare,'Invalid compute comparison mode')
    expected_modes = {'fused','transport_body'} if transport_compare else {'fused'} if fused_only else {'cutlass_stock','deepgemm_m128','deepgemm_m256','fused'} if external else {'cutlass_search'} if search else {
        'fused', 'transport_body', 'cutlass_matched', 'cublas_grouped_default', 'cublaslt_sequence_tuned'}
    if compute_compare:
        expected_modes.add('cutlass_matched')
    require({key[1] for key in results} == expected_modes, 'Missing or unexpected modes')
    if search:
        for direction in directions:
            configs = {key[2] for key in results if key[0] == direction and
                       (not external or key[1]=='cutlass_stock')}
            require(len(configs) == 32 and {c[:4] for c in configs} ==
                {(n,k,a,s) for n in (128,256) for k in (64,128) for a in (0,1) for s in (1,2,4,8)},
                'Incomplete compute grid')
            require(len({c[5] for c in configs}) == 1 and all(c[4] == 0 for c in configs),
                    'Compute search changed CTA budget')
            if external:
                require(len([key for key in results if key[0]==direction])==35,
                        'Incomplete external grouped comparison')
                require(all(key[2][4:]==(0,148) for key in results if key[0]==direction and
                            key[1]!='fused'), 'External baseline must use full SM budget')
    else:
        require(len(results) == len(expected_modes)*len(directions), 'Incomplete comparison')
        if compute_compare:
            for direction in directions:
                require(len({key[2] for key in results if key[0]==direction})==1,
                        'Fixed compute comparison changed GEMM/CTA configuration')
    output = []
    for (direction, mode, config), payloads in results.items():
        require(set(payloads) == {0,1}, 'Missing second payload')
        valid = all(payloads[p][-1]['accepted'] for p in (0,1))
        ms = statistics.mean(payloads[p][-1]['p50'] for p in (0,1)) if valid else None
        n,k = (2*geometry['f'],geometry['h']) if direction == 'dispatch' else (geometry['h'],geometry['f'])
        flops = statistics.mean(sum(sum(expert_rows[direction,p,r]) for r in range(world))*2*n*k for p in (0,1))
        output.append(dict(direction=direction, mode=mode, config=dict(zip(
            ('tile_n','tile_k','along_n','swizzle','comm','compute'),config)), valid=valid,
            ms=ms, pflops=flops/(world*ms*1e12) if valid else None, payloads=payloads))
    return output


def audit_run(directory, case_index=None):
    root, receipt, job, geometry, data = read_run(directory,case_index)
    world = int(job['world'])
    if '_skip_reason' in data:
        return dict(run_id=receipt['run_id'],case_index=case_index,source_id=receipt['source_id'],
                    environment_fingerprint=receipt['environment_fingerprint'],
                    directions=['dispatch','combine'] if job.get('grouped_direction') in (None,'both') else [job['grouped_direction']],
                    geometry=geometry,world=world,search=bool(job.get('grouped_gemm_search')),
                    skipped=True,reason=data['_skip_reason'])
    direction = job.get('grouped_direction') or 'both'
    directions = ('dispatch','combine') if direction=='both' else (direction,)
    lines = data[f'attempt{receipt["attempt"]}.log'].decode().splitlines()
    counts, checks, algorithms, routes = {}, set(), [], []
    for line in lines:
        if line.startswith('ROWS grouped '):
            fields = dict(re.findall(r'(\w+)=([^ ]+)', line))
            key = fields['direction'], int(fields['payload']), int(fields['rank'])
            require(key not in counts, 'Repeated row metadata')
            counts[key] = list(map(int, fields['values'].split(',')))
            require(re.fullmatch(r'[0-9a-fA-F]+', fields['route_fnv64']) is not None,
                    'Missing routing fingerprint')
            routes.append((key, fields['route_seed'], fields['route_fnv64'], counts[key]))
            require(len(counts[key]) == geometry['experts']//world and
                    all(0 <= n <= geometry['expert_row_capacity'] for n in counts[key]), 'Invalid expert rows')
        elif line.startswith('CHECK grouped-ep ') and line.endswith(' passed'):
            fields = dict(re.findall(r'(\w+)=([^ ]+)', line))
            checks.add((fields['direction'],int(fields['replay'])))
        elif line.startswith('ALGORITHM grouped '):
            fields, info = line.split(' info=',1)
            record = dict(re.findall(r'(\w+)=([^ ]+)', fields))
            record['info'] = json_bytes(info.encode())
            algorithms.append(record)
    require(checks == {(d,p) for d in directions for p in (0,1)}, 'Missing post-checks')
    require(set(counts) == {(d,p,r) for d in directions for p in (0,1) for r in range(world)},
            'Missing expert distribution')
    search = 'external' if job.get('grouped_external') else bool(job.get('grouped_gemm_search'))
    if search == 'external':
        build=json_bytes(data['grouped-contract.json'])['build']
        require(build.get('external',{}).get('upstream')=='78b69000794d0937b47ae3387eff7663410264d1',
                'Missing pinned external dependency receipt')
    if not search and not job.get('grouped_fused_only'):
        expected = {(r,m,n,k) for d in directions for r in range(world)
            for p in (0,1) for m in counts[d,p,r] if m
            for n,k in ([ (2*geometry['f'],geometry['h']) ] if d=='dispatch' else [(geometry['h'],geometry['f'])])}
        actual = {(int(a['rank']),int(a['m']),int(a['n']),int(a['k'])) for a in algorithms}
        require(actual == expected, 'Incomplete Lt tuning configuration')
        for a in algorithms:
            info = a['info']
            require(info['precision']==16 and info['math_sms']==0 and info['graph_tuning']==1 and
                    info['requested']==32 and info['valid']>0 and info['beta']==0, 'Wrong Lt baseline contract')
    rows = audit_samples(data['grouped-samples.csv'].decode(), world, geometry, counts, search,directions,
                         job.get('grouped_fused_only',False),job.get('grouped_transport_compare',False),
                         job.get('grouped_compute_compare',False))
    if job.get('grouped_fused_only'):
        requested=job.get('grouped_buffer_rows',0)
        actual=min(requested,geometry['expert_row_capacity']) if requested else geometry['expert_row_capacity']
        require(all(int(r['buffer_rows'])==actual for r in csv.DictReader(io.StringIO(data['grouped-samples.csv'].decode()))),
                'Measured buffer capacity mismatch')
        for r in csv.DictReader(io.StringIO(data['grouped-samples.csv'].decode())):
            for column,key in (('tail_balance','grouped_tail_balance'),('hot_half','grouped_hot_half')):
                require(int(r.get(column,0))==int(bool(job.get(key))), 'Measured tail/routing option mismatch')
    policy = grouped_policy(job) or dict(tile_n=128,tile_k=64,along_n=0,swizzle=1,comm=20,compute=128)
    # Old archives predate the optional column and mean swapAB=off. Never
    # compare a swapped candidate under an indistinguishable six-field key.
    for sample in csv.DictReader(io.StringIO(data['grouped-samples.csv'].decode())):
        require(int(sample.get('swap_ab',0))==int(policy.get('swap_ab',False)),
                'Measured swapAB differs from request')
        require(int(sample.get('trim_swap_tokens',1))==int(policy.get('trim_swap_tokens',True)),
                'Measured swapAB tail trimming differs from request')
        require(int(sample.get('dispatch_copy',0))==int(policy.get('dispatch_copy')=='tma'),
                'Measured Dispatch copy method differs from request')
    if policy.get('swap_ab'):
        for row in rows: row['config']['swap_ab'] = True
    if policy.get('trim_swap_tokens') is False:
        for row in rows: row['config']['trim_swap_tokens'] = False
    if policy.get('dispatch_copy') == 'tma':
        for row in rows: row['config']['dispatch_copy'] = 'tma'
    for row in rows:
        if search == 'external':
            require(row['config']==policy if row['mode']=='fused' else
                    row['config']['compute']==148 and row['config']['comm']==0,
                    'External/fusion budget mismatch')
            continue
        require(row['config']['compute']==policy['compute'], 'Measured compute budget differs from request')
        if not search:
            require(row['config']==policy, 'Measured fusion configuration differs from request')
    selected = {}
    for direction in directions:
        candidates = [r for r in rows if r['direction']==direction and r['valid'] and
            r['mode'] in ({'cutlass_stock','deepgemm_m128','deepgemm_m256'} if search=='external' else
                         {'cutlass_search'} if search else {'cublas_grouped_default','cublaslt_sequence_tuned'})]
        selected[direction] = min(candidates,key=lambda r:r['ms']) if candidates else None
    return dict(run_id=receipt['run_id'], case_index=case_index, source_id=receipt['source_id'], artifacts=str(root),
                environment_fingerprint=receipt['environment_fingerprint'],
                routing_ids={d:digest(json.dumps(sorted(r for r in routes if r[0][0]==d),
                                              sort_keys=True).encode()) for d in directions},
                geometry=geometry, world=world, search=search, selected=selected, rows=rows, algorithms=algorithms,
                buffer_rows=job.get('grouped_buffer_rows',0),fused_only=bool(job.get('grouped_fused_only')),
                tail_balance=bool(job.get('grouped_tail_balance')),hot_half=bool(job.get('grouped_hot_half')))


def audit_cases(directory):
    """Enumerate from the local job; audit_run verifies it against the archive."""
    root=Path(directory)
    if (root/'control').is_dir():
        root=root.parent
    job=json.loads((root/'job.json').read_text())
    batch=grouped_batch_cases(job)
    return [audit_run(root,i) for i in range(len(batch))] if batch else [audit_run(root)]


def comparison_rows(records):
    """Bounded-candidate best, using a whole pair of payloads for every result.

    Never combine environments, native snapshots or routing. A repeated identical
    configuration uses the latest run, not the fastest repeat. The library
    reference is the strongest valid method over these compatible trials.
    Search-only records are not fused measurements and cannot enter this table.
    """
    groups=defaultdict(dict)
    for record in records:
        if record.get('skipped') or record['search'] or record.get('fused_only'):
            continue
        g=record['geometry']
        identity=tuple(g[k] for k in ('h','f','experts','topk','target_rows'))
        common=(identity,record['world'],record['source_id'],record['environment_fingerprint'])
        for row in record['rows']:
            key=common+(record['routing_ids'][row['direction']],row['direction'])
            config=tuple(sorted(row['config'].items()))
            candidate=(row['mode'],config)
            stamp=(record['run_id'],record.get('case_index') or 0)
            previous=groups[key].get(candidate)
            if previous is None or stamp>previous[0]:
                groups[key][candidate]=(stamp,row,record)
    output=[]
    for key,candidates in sorted(groups.items()):
        values=list(candidates.values())
        fused=[v for v in values if v[1]['mode']=='fused' and v[1]['valid']]
        refs=[v for v in values if v[1]['mode'] in
              ('cublas_grouped_default','cublaslt_sequence_tuned') and v[1]['valid']]
        if not fused or not refs:
            continue
        _,f,fr=min(fused,key=lambda v:v[1]['ms'])
        _,b,br=min(refs,key=lambda v:v[1]['ms'])
        def component(mode):
            matches=[v[1] for v in values if v[1]['mode']==mode and v[1]['valid'] and
                     v[2]['run_id']==fr['run_id'] and v[2].get('case_index')==fr.get('case_index')]
            require(len(matches)<=1,'Ambiguous matched component')
            return matches[0] if matches else None
        own=component('cutlass_matched'); transport=component('transport_body')
        output.append(dict(h=key[0][0],f=key[0][1],experts=key[0][2],topk=key[0][3],
            target_rows=key[0][4],ep=key[1],source_id=key[2],direction=key[-1],
            environment_fingerprint=key[3],routing_id=key[4],
            fused_ms=f['ms'],fused_pflops=f['pflops'],reference_ms=b['ms'],
            reference_pflops=b['pflops'],retention=b['ms']/f['ms'],config=f['config'],
            reference_method=b['mode'],fused_run=fr['run_id'],fused_case=fr.get('case_index'),
            reference_run=br['run_id'],reference_case=br.get('case_index'),
            standalone_cutlass_ms=own['ms'] if own else None,
            standalone_cutlass_pflops=own['pflops'] if own else None,
            fused_over_cutlass=own['ms']/f['ms'] if own else None,
            transport_body_ms=transport['ms'] if transport else None))
    return output


def catalog_rows(records):
    """Complete requested matrix; absent results remain explicitly absent.

    This report accepts one native snapshot/environment. It never chooses a
    faster source version, routing distribution, or node for the caller.
    """
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks/sm103'))
    from grouped_shapes import MODELS, TOKEN_COUNTS
    identities={(r['source_id'],r.get('environment_fingerprint')) for r in records}
    require(len(identities)<=1,'Catalog requires one source/environment selection')
    measurements={}
    for row in comparison_rows(records):
        key=tuple(row[k] for k in ('h','f','experts','topk','target_rows','ep','direction'))
        require(key not in measurements,'Catalog contains different routes for the same geometry')
        measurements[key]=row
    aliases=defaultdict(list)
    for model,profile in MODELS.items(): aliases[profile].append(model)
    attempted=defaultdict(list)
    for r in records:
        if r.get('search'): continue
        g=r['geometry']
        directions=r.get('directions') or sorted({x['direction'] for x in r.get('rows',[])})
        for d in directions:
            key=tuple(g[k] for k in ('h','f','experts','topk','target_rows'))+(r['world'],d)
            attempted[key].append(r)
    output=[]
    for (h,f,e,top),models in aliases.items():
        for ep in (4,8):
            for target in TOKEN_COUNTS:
                for direction in ('dispatch','combine'):
                    key=(h,f,e,top,target,ep,direction)
                    row=dict(models=models,h=h,f=f,experts=e,topk=top,ep=ep,
                        target_rows=target,tokens_per_rank=(e*target+ep*top-1)//(ep*top),
                        direction=direction,status='not_measured')
                    if key in measurements:
                        row.update(measurements[key]); row['status']='valid'
                    elif key in attempted:
                        attempts=attempted[key]
                        row['status']='insufficient_memory' if all(r.get('skipped') for r in attempts) else 'invalid_measurement'
                        row['attempts']=[dict(run=r['run_id'],case=r.get('case_index'),reason=r.get('reason')) for r in attempts]
                    output.append(row)
    return output


def seed_fusion_plan(searches, budgets, eps=(4,8), targets=None):
    """Offline measurement plan, NOT a production heuristic or a result table.

    Reuse measured GEMM/CTA configurations to obtain initial full coverage.
    Prefer the same EP and nearest measured token scale; record every fallback.
    Each generated workload still measures its actual pure-library reference,
    matched CUTLASS and fusion. Untested shapes are never labelled tuned by this
    function. No model names enter a CUDA policy or numerical performance fit.
    """
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks/sm103'))
    from grouped_shapes import MODELS, TOKEN_COUNTS
    targets=TOKEN_COUNTS if targets is None else targets
    seeds=defaultdict(list)
    for r in searches:
        if r.get('skipped') or not r['search']: continue
        g=r['geometry']; profile=tuple(g[k] for k in ('h','f','experts','topk'))
        for direction,winner in r['selected'].items():
            if winner is not None and winner['valid']:
                seeds[profile+(direction,)].append((r,winner))
    budget_groups=defaultdict(list)
    for row in budgets:
        budget_groups[tuple(row[k] for k in ('h','f','experts','topk','direction'))].append(row)
    batches=[]
    for ep in eps:
        require(ep in (4,8),'Invalid plan EP')
        for profile in dict.fromkeys(MODELS.values()):
            entries=[]
            for target in targets:
                require(target>0,'Invalid plan target rows')
                for direction in ('dispatch','combine'):
                    key=profile+(direction,)
                    require(seeds[key] and budget_groups[key],f'Missing measured seed for {key}')
                    newest=sorted(seeds[key],key=lambda x:x[0]['run_id'],reverse=True)
                    r,winner=min(newest,key=lambda x:(x[0]['world']!=ep,
                        abs(math.log2(x[0]['geometry']['target_rows']/target))))
                    b=min(budget_groups[key],key=lambda x:(x['ep']!=ep,
                        abs(math.log2(x['target_rows']/target))))
                    c=winner['config']|dict(comm=b['config']['comm'],compute=b['config']['compute'])
                    case=','.join(map(str,profile+(target,)))
                    policy=','.join(str(c[k]) for k in ('tile_n','tile_k','along_n','swizzle','comm','compute'))
                    entries.append(dict(case=case,policy=policy,direction=direction,
                        argument=f'{case}@{policy}@{direction}',status='seed_not_measured',
                        gemm_seed=dict(run=r['run_id'],case=r.get('case_index'),
                            ep=r['world'],target_rows=r['geometry']['target_rows']),
                        cta_seed=dict(run=b['fused_run'],case=b['fused_case'],
                            ep=b['ep'],target_rows=b['target_rows'])))
            require(len(entries)<=64,'Split plan into at most 64 cases per native batch')
            batches.append(dict(ep=ep,profile=profile,entries=entries))
    return batches


def collect_plan(plan_path, runs_root, refinements=()):
    """Read-only refresh of the frozen full-matrix plan from fetched evidence.

    Running/unfetched jobs never contribute measurements. A matching completed
    job must reproduce the exact frozen batch, then pass the ordinary archive,
    native build, routing, correctness and sample audit. Explicit refinement
    runs may add measured candidates, not coverage obligations or new shapes.
    No GPU/cloud access.
    """
    plan_path=Path(plan_path)
    plan=json.loads(plan_path.read_text())
    require(plan['schema']=='grouped-bf16-full-seed-plan-v1','Unexpected plan schema')
    records=[]; runs=[]; completed=set()
    for directory in sorted(Path(runs_root).glob('*')):
        job_file=directory/'job.json'
        receipt_file=directory/'fetched.json'
        if not job_file.is_file() or not receipt_file.is_file(): continue
        job=json.loads(job_file.read_text())
        match=re.fullmatch(r'grouped-full-seed-(\d{2})-ep([48])',job.get('experiment',''))
        if match is None: continue
        index,ep=map(int,match.groups())
        require(index<len(plan['batches']),'Batch outside frozen plan')
        batch=plan['batches'][index]
        require(job['source_id']==plan['source_id'] and job['world']==ep==batch['ep'],
                'Full-matrix source or EP differs from plan')
        require(job['grouped_cases']==';'.join(e['argument'] for e in batch['entries']) and
                not job.get('grouped_gemm_search') and job.get('grouped_measure'),
                'Full-matrix candidate list differs from plan')
        receipt=json.loads(receipt_file.read_text())
        if receipt.get('state')!='succeeded':
            runs.append(dict(batch=index,run=job['run_id'],state=receipt.get('state')))
            continue
        checked=audit_cases(directory)
        require(len(checked)==len(batch['entries']),'Incomplete audited batch')
        records.extend(checked); completed.add(index)
        runs.append(dict(batch=index,run=job['run_id'],state='audited',cases=len(checked)))
    rows=catalog_rows(records)
    key_fields=('h','f','experts','topk','target_rows')
    allowed={tuple(r[k] for k in key_fields)+(r['ep'],) for r in rows}
    seen={r['run'] for r in runs}
    for directory in refinements:
        checked=audit_cases(directory)
        require(checked,'Empty refinement run')
        run_ids={r['run_id'] for r in checked}
        require(len(run_ids)==1 and not run_ids & seen,'Duplicate refinement run')
        for r in checked:
            require(r['source_id']==plan['source_id'] and not r['search'],
                    'Refinement must measure fusion from the same source')
            require(tuple(r['geometry'][k] for k in key_fields)+(r['world'],) in allowed,
                    'Refinement outside the requested workload catalog')
        records.extend(checked); seen.update(run_ids)
        runs.append(dict(run=checked[0]['run_id'],state='audited',cases=len(checked),
                         kind='bounded_refinement'))
    if refinements:
        # catalog_rows also requires one environment and identical routes before
        # pooling candidates. Repeated configurations use latest, not fastest.
        rows=catalog_rows(records)
    selection_note=('Best measured configurations from initial coverage and explicitly supplied '
        'bounded refinements; not exhaustive per-load tuning or a global optimum. ' if refinements else
        'Initial measured seed configurations, not per-load tuned winners or a global optimum. ')
    return dict(schema='grouped-bf16-current-v1',source_id=plan['source_id'],plan=plan_path.name,
        state='initial_matrix_attempted' if len(completed)==len(plan['batches']) else 'initial_full_matrix_in_progress',
        completed_batches=len(completed),total_batches=len(plan['batches']),runs=runs,
        refinement_runs=len(refinements),
        coverage=dict(Counter(r['status'] for r in rows)),
        note=selection_note+'Both payloads averaged; max-rank Graph 10+50; useful PFLOPS per GPU. '
             'Reference is the faster valid complete native cuBLAS grouped or tuned cuBLASLt Graph sequence, '
             'not a theoretical hardware ceiling. Standalone CUTLASS uses the selected reduced compute budget; '
             'standalone components are diagnostics, not additive measured critical-path fractions.',rows=rows)


def collect_external_plan(plan_path, runs_root, fill_plan_path=None):
    """Full-library reference matrix, separate from the historical cuBLAS table.

    Native DeepGEMM candidates are not claims about its Python default heuristic.
    Compare each library's best VALID dual-payload configuration with the SAME
    run's unchanged fusion configuration. Never select a faster measurement retry.
    """
    plan=json.loads(Path(plan_path).read_text())
    require(plan['schema']=='grouped-external-reference-v1','Wrong external plan')
    found={}
    for path in sorted(Path(runs_root).glob('*/job.json')):
        job=json.loads(path.read_text())
        match=re.fullmatch(r'grouped-external-full-(\d+)',job.get('experiment',''))
        if not match or not (path.parent/'fetched.json').is_file(): continue
        index=int(match[1]); require(index<len(plan['batches']),'Unexpected external batch')
        batch=plan['batches'][index]
        require(job.get('source_run')==plan['source_run'] and job.get('grouped_external') and
                job.get('grouped_direction')=='dispatch' and job['world']==batch['ep'] and
                job['grouped_cases']==';'.join(batch['entries']),'External plan drift')
        require(index not in found,'Repeated external batch: select its run explicitly')
        found[index]=audit_cases(path.parent)
    replacements={}; fill_runs=[]
    if fill_plan_path is not None:
        fill=json.loads(Path(fill_plan_path).read_text())
        require(fill['schema']=='grouped-external-fill-v1' and
                fill['source_run']==plan['source_run'], 'Wrong external fill plan')
        targets={(b['ep'],entry):(i,j) for i,b in enumerate(plan['batches'])
                 for j,entry in enumerate(b['entries'])}
        planned=set(); completed=set()
        for b in fill['batches']:
            for entry in b['entries']:
                key=(b['ep'],entry)
                require(key in targets and key not in planned,'Unexpected or repeated fill target')
                planned.add(key)
                i,j=targets[key]
                require(i in found and found[i][j].get('skipped'),
                        'Fill may only replace a previously memory-skipped point')
        require(len(planned)==fill['case_count'],'Incomplete fill plan')
        for path in sorted(Path(runs_root).glob('*/job.json')):
            job=json.loads(path.read_text())
            match=re.fullmatch(r'grouped-external-fill-(\d+)',job.get('experiment',''))
            if not match or not (path.parent/'fetched.json').is_file(): continue
            index=int(match[1])
            require(index<len(fill['batches']) and index not in completed,'Unexpected or repeated fill batch')
            batch=fill['batches'][index]
            require(job.get('source_run')==plan['source_run'] and job.get('grouped_external') and
                    job.get('grouped_direction')=='dispatch' and job['world']==batch['ep'] and
                    job['grouped_cases']==';'.join(batch['entries']),'External fill plan drift')
            records=audit_cases(path.parent)
            require(len(records)==len(batch['entries']),'Missing fill cases')
            for entry,record in zip(batch['entries'],records):
                replacements[targets[(batch['ep'],entry)]]=record
            completed.add(index); fill_runs.append(path.parent.name)
    output=[]; fingerprints=set()
    for index,batch in enumerate(plan['batches']):
        records=found.get(index)
        if records: require(len(records)==len(batch['entries']),'Missing external cases')
        for i,arg in enumerate(batch['entries']):
            geom,policy,_=arg.split('@')
            h,f,e,top,m=map(int,geom.split(','))
            row=dict(models=batch['models'],ep=batch['ep'],h=h,f=f,experts=e,topk=top,
                     target_rows=m,fusion_policy=policy,status='unmeasured')
            if records:
                original=records[i]
                fingerprints.add((original['source_id'],original['environment_fingerprint']))
                r=replacements.get((index,i),original)
                if r is not original:
                    row['previous_attempt']=dict(run=original['run_id'],case_index=i,
                                                 reason=original['reason'])
                row.update(run=r['run_id'],case_index=r['case_index'],source_id=r['source_id'],
                           environment_fingerprint=r['environment_fingerprint'])
                fingerprints.add((r['source_id'],r['environment_fingerprint']))
                if r.get('skipped'):
                    row['status']='insufficient_memory'
                    row['skip_reason']=r['reason']
                else:
                    row['routing_id']=r['routing_ids']['dispatch']
                    valid=[x for x in r['rows'] if x['valid']]
                    groups=[ [x for x in valid if x['mode']==mode] for mode in ('fused','cutlass_stock')]
                    groups.append([x for x in valid if x['mode'].startswith('deepgemm_')])
                    row['valid_candidates']=len(valid)-len(groups[0])
                    row['status']='unstable'
                    if all(groups):
                        fused,cutlass,deepgemm=[min(g,key=lambda x:x['ms']) for g in groups]
                        best=min((cutlass,deepgemm),key=lambda x:x['ms'])
                        # Complete rounds/rank samples remain in the hash-verified
                        # run archive; keep the unique table compact, not a sweep log.
                        compact=lambda x:{k:x[k] for k in ('mode','config','ms','pflops')}
                        row.update(status='valid',fused=compact(fused),reference=compact(best),
                                   best_backend=best['mode'],retention=best['ms']/fused['ms'],
                                   excess_ms=fused['ms']-best['ms'])
            output.append(row)
    require(len(fingerprints)<=1,'Mixed external source/environment')
    require(len(output)==plan['case_count'],'Incomplete external matrix')
    return dict(schema='grouped-external-reference-results-v2',note=plan['note'],
                completed_batches=len(found),total_batches=len(plan['batches']),
                coverage=dict(Counter(r['status'] for r in output)),rows=output,
                fill_runs=fill_runs)


def external_markdown(table):
    lines=['# Dispatch + BF16 Grouped GEMM — external compute references','',table['note'],'',
           f"Batches: {table['completed_batches']}/{table['total_batches']}; coverage: {table['coverage']}.",'',
           f"Audited memory-skip retest batches: {len(table.get('fill_runs', []))}.", '',
           'P = useful PFLOPS/GPU. M = target token rows per expert. Reference kernels use full148SM; '
           'fusion retains the frozen measured comm/compute budget. Pure GEMM excludes dispatch/layout preparation; '
           'the excess time is NOT an additive measurement of communication time. '
           'DeepGEMM uses unmodified pinned kernels through a native adapter, dynamic compiled dimensions, '
           'masked layout, M128/M256 N128 K64, swapAB, cluster2. '
           'Stock CUTLASS uses M128 N128/256 K64/128, both rasters, swizzle1/2/4/8, cluster1. '
           'No claim of exhaustive library optimum. — = unavailable, never a zero.', '',
           'The table retains only max(CUTLASS, DeepGEMM) throughput per point; '
           'its exact winning configuration remains in JSON. Raw candidate evidence stays in the run archives.', '',
           '| Models | EP | M | Fused P | Best GEMM P | Fused/best | Excess ms | Best | Status |',
           '|---|---:|---:|---:|---:|---:|---:|---|---|']
    for r in table['rows']:
        cells=[', '.join(r['models']),str(r['ep']),str(r['target_rows'])]
        if r['status']=='valid':
            cells += [f"{r[k]['pflops']:.3f}" for k in ('fused','reference')]
            cells += [f"{r['retention']*100:.1f}%",f"{r['excess_ms']:.6f}",r['best_backend']]
        else: cells += ['—']*5
        lines.append('| '+' | '.join(cells+[r['status']])+' |')
    return '\n'.join(lines)+'\n'


def grouped_handoff_trace(ranks):
    """Audit every tile/rank; show longest local-role envelope, all rank summaries.

    These are producer-warp observations, NOT Tensor Core idle time. Publication
    stamps bracket the store instruction, not remote/global visibility latency.
    No cross-GPU timestamp subtraction and no sum of CTA waits as E2E loss.
    """
    require(ranks and sorted(r['rank'] for r in ranks)==list(range(ranks[0]['world'])),
            'Missing grouped profile rank')
    events=[]; summaries=[]; normalized=[]
    percentile=lambda xs,q: sorted(xs)[int((len(xs)-1)*q)] if xs else 0
    for r in sorted(ranks,key=lambda r:r['rank']):
        require(r['schema']=='grouped-handoff-v1' and r['warmup']==10 and
                r['payload']==1 and r['epoch']==13,'Wrong grouped profile protocol')
        comm,compute=r['comm'],r['compute']; roles=r['roles']
        require(len(roles)==comm+compute,'Incomplete CTA coverage')
        require(all(0<a<=b<=c for a,b,c in roles),'Invalid role timestamps')
        origin=min(x[0] for x in roles)
        nt=(r['n']+r['tile_n']-1)//r['tile_n']
        mt=[(m+127)//128 for m in r['rows']]
        require(len(r['panels'])==sum(mt) and len(r['tiles'])==sum(mt)*nt,'Missing panel/tile records')
        # This probe records the final publisher's copy. The selected cases
        # have one producer per panel; do not mislabel split-panel captures.
        require(sum(mt)>=comm,'Split-panel diagnostics need per-producer records')
        panels={}; offset=0
        for e,count in enumerate(mt):
            for m in range(count):
                a,b,c,owner,expert,row=r['panels'][offset+m]
                require((expert,row)==(e,m) and owner==(offset+m)%comm,'Wrong publication ownership')
                require(roles[owner][0]<=a<=b<=c<=roles[owner][1] and a>0,'Invalid publication span')
                panels[e,m]=(a,b,c,owner)
            offset+=count
        waits=[]; publication_gaps=[]; first={}; seen=set(); by_cta=[[] for _ in roles]
        ready_intervals=[]; checks=[]; first_load=[]; offset=0
        for idx,t in enumerate(r['tiles']):
            begin,observed,load,end,cta,e,m,n,polled=t
            require((e,m) in panels and 0<=n<nt and (e,m,n) not in seen,'Invalid or repeated consumer tile')
            seen.add((e,m,n))
            panel=sum(mt[:e])+m
            require(idx==panel*nt+n,'Wrong physical record index')
            major_count=nt if r['along_n'] else mt[e]
            minor_count=mt[e] if r['along_n'] else nt
            minor=m if r['along_n'] else n; major=n if r['along_n'] else m
            band=minor//r['swizzle']*r['swizzle']; width=min(r['swizzle'],minor_count-band)
            logical=sum(mt[:e])*nt+band*major_count+major*width+minor-band
            require(cta==comm+logical%compute,'Wrong compute task ownership')
            require(roles[cta][0]<=begin<=observed<=load<=end<=roles[cta][1] and begin>0,
                    'Invalid consumer timestamps')
            pa,rb,re,owner=panels[e,m]
            require(observed>=rb and polled in (0,1),'Acquire preceded publication or invalid polling flag')
            require(polled or observed==begin,'Cached-ready record claims a polling interval')
            first[e,m]=min(first.get((e,m),observed),observed)
            wait=observed-begin; pre=max(0,min(observed,rb)-begin)
            waits.append(wait);first_load.append(end-load)
            by_cta[cta].append((begin,end,wait,pre))
            checks.append((begin,wait))
            if re<begin: ready_intervals.append((re,begin))
        for key,acquire in first.items(): publication_gaps.append(acquire-panels[key][1])
        starts=sorted(a for a,b in ready_intervals); ends=sorted(b for a,b in ready_intervals)
        # Existence of a ready future load is only a scheduling opportunity:
        # its accumulator/TMA pipeline resources may not be available yet.
        exposed=[(b,w) for b,w in checks if w>=1000]
        opportunities=sum(bisect.bisect_right(starts,b)>bisect.bisect_right(ends,b) for b,w in exposed)
        cta_stats=[]
        for c in range(comm,comm+compute):
            work=sorted(by_cta[c]); require(all(x[1]<=y[0] for x,y in zip(work,work[1:])),
                                         'Overlapping first load calls on one CTA')
            span=roles[c][1]-roles[c][0]; wait=sum(x[2] for x in work); pre=sum(x[3] for x in work)
            require(wait<=span,'Wait sum exceeds role')
            cta_stats.append(dict(cta=c,tiles=len(work),role_us=span/1000,ready_wait_us=wait/1000,
                before_publish_us=pre/1000,after_publish_us=(wait-pre)/1000,wait_fraction=wait/span if span else 0))
        summary=dict(rank=r['rank'],kernel_us=(max(x[2] for x in roles)-origin)/1000,
            comm_done_us=(max(x[1] for x in roles[:comm])-origin)/1000,
            compute_done_us=(max(x[1] for x in roles[comm:])-origin)/1000,
            panels=len(panels),tiles=len(seen),ready_wait_p50_us=percentile(waits,.5)/1000,
            ready_wait_p95_us=percentile(waits,.95)/1000,ready_wait_max_us=max(waits)/1000,
            release_to_first_acquire_p50_us=percentile(publication_gaps,.5)/1000,
            release_to_first_acquire_p95_us=percentile(publication_gaps,.95)/1000,
            first_load_call_p50_us=percentile(first_load,.5)/1000,
            first_load_call_p95_us=percentile(first_load,.95)/1000,
            waits_ge_1us=len(exposed),waits_with_other_ready_unvisited_tile=opportunities,
            ctas=cta_stats)
        summaries.append(summary);normalized.append((r,origin,panels))
    selected=max(summaries,key=lambda x:max(x['comm_done_us'],x['compute_done_us']))['rank']
    r,origin,panels=next(x for x in normalized if x[0]['rank']==selected)
    def span(name,tid,a,b,args):
        events.append(dict(name=name,ph='X',pid=selected,tid=tid,ts=(a-origin)/1000,dur=(b-a)/1000,args=args))
    for c,(a,b,end) in enumerate(r['roles']):
        for lane,label in ((0,'role'),(1,'handoff'),(2,'first load call')):
            events.append(dict(name='thread_name',ph='M',pid=selected,tid=c*3+lane,
                               args=dict(name=f'CTA {c:03d} / {label}')))
            events.append(dict(name='thread_sort_index',ph='M',pid=selected,tid=c*3+lane,args=dict(sort_index=c*3+lane)))
        span('Dispatch role' if c<r['comm'] else 'GEMM role',c*3,a,b,{})
        span('grid join + finalize',c*3,b,end,{})
    for (e,m),(a,b,end,c) in panels.items():
        span('gather + original CTA join',c*3+1,a,b,dict(expert=e,m=m))
        span('ready publication',c*3+1,b,end,dict(expert=e,m=m))
    for a,b,l,end,c,e,m,n,polled in r['tiles']:
        _,rb,re,producer=panels[e,m]
        args=dict(expert=e,m=m,n=n,producer_cta=producer,polled=bool(polled),
                  release_begin_us=(rb-origin)/1000,release_end_us=(re-origin)/1000,
                  acquire_minus_release_begin_us=(b-rb)/1000)
        span('ready check / wait' if polled else 'ready reused',c*3+1,a,b,args)
        span('warp join + proxy fence + bookkeeping',c*3+1,b,l,dict(expert=e,m=m,n=n))
        span('first Base::load call (includes pipeline waits)',c*3+2,l,end,dict(expert=e,m=m,n=n))
    return dict(traceEvents=events,displayTimeUnit='ms',metadata=dict(
        schema='grouped-handoff-perfetto-v1',detail_rank=selected,rank_summaries=summaries,
        config={k:r[k] for k in ('world','n','k','tile_n','tile_k','comm','compute','swizzle','along_n','rows')},
        note='All ranks validated; detailed tracks show rank with longest local-role envelope (before cross-rank finalize). Per-GPU globaltimer origins. '
             'First load is not MMA execution; ready waits can overlap earlier MMA. '
             'Publication stamps bracket instruction issue/completion, not visibility. '
             'Ready future tasks indicate opportunity, not proven reclaimable E2E time.'))


def export_grouped_handoff(run):
    root=Path(run); receipt=json.loads((root/'fetched.json').read_text())
    require(receipt['state']=='succeeded' and receipt['exit_code']==receipt['work_exit_code']==0,
            'Profile run did not succeed')
    archive=root/'artifacts.tar.gz'; require(file_digest(archive)==receipt['artifact_sha256'],'Profile archive mismatch')
    folder=root/f"artifacts-attempt{receipt['attempt']}"/'control'
    names=['job.json','grouped-contract.json',f"attempt{receipt['attempt']}.log"]
    job=json.loads((folder/'job.json').read_text())
    require(job.get('grouped_profile') and not job.get('grouped_measure') and job['grouped_direction']=='dispatch',
            'Not isolated Dispatch profiling')
    names += [f'grouped-handoff-rank-{r}.json' for r in range(job['world'])]
    with tarfile.open(archive,'r:gz') as stream:
        for name in names:
            member=stream.getmember('control/'+name)
            require(member.isfile() and member.size<64<<20,'Invalid profile member')
            require((folder/name).read_bytes()==stream.extractfile(member).read(),'Profile extraction changed')
    contract=json.loads((folder/'grouped-contract.json').read_text())
    require(contract['profiling'] and contract['build']['profiling'],'Profile build was not recorded')
    require('PROFILE grouped: globaltimer' in (folder/names[2]).read_text(),'Missing final profile validation')
    trace=grouped_handoff_trace([json.loads((folder/n).read_text()) for n in names[3:]])
    trace['metadata']['evidence']=dict(run=root.name,source_id=receipt['source_id'],
        binary_sha256=contract['build']['binary_sha256'],archive_sha256=receipt['artifact_sha256'],
        physical_devices=contract['physical_devices'])
    return trace


def grouped_ready_checks(rank):
    """Replay the existing static tile order; no timing assumptions or tuning."""
    nt=(rank['n']+rank['tile_n']-1)//rank['tile_n']
    workers=rank['compute']; sw=rank['swizzle']; along=rank['along_n']
    last=[-1]*workers; checks=[0]*workers; prefix=0
    for rows in rank['rows']:
        mt=(rows+127)//128
        major_count,minor_count=(nt,mt) if along else (mt,nt)
        for local in range(mt*nt):
            begin=local//(major_count*sw)*sw
            width=min(sw,minor_count-begin)
            offset=local-begin*major_count
            major,minor=offset//width,begin+offset%width
            panel=prefix+(minor if along else major)
            worker=(prefix*nt+local)%workers
            if panel!=last[worker]:
                checks[worker]+=1; last[worker]=panel
        prefix+=mt
    return checks


def grouped_ready_summary(ranks):
    """Polling intervals of the load warp, NOT Tensor Core idle or E2E loss."""
    require(bool(ranks),'Missing ready summaries')
    world=ranks[0]['world']
    require(len(ranks)==world and {r['rank'] for r in ranks}==set(range(world)),
            'Missing/duplicate ready-summary rank')
    fields=('world','n','k','tile_n','tile_k','swizzle','along_n','comm','compute')
    summaries=[]
    for r in sorted(ranks,key=lambda r:r['rank']):
        require(r['schema']=='grouped-ready-summary-v1' and
                (r['warmup'],r['payload'],r['epoch'])==(10,1,13),'Invalid ready protocol')
        require(all(r[k]==ranks[0][k] for k in fields),'Rank configuration mismatch')
        require(r['comm']>0 and r['compute']>0 and r['comm']+r['compute']<=148 and
                r['tile_n'] in (128,256) and r['swizzle'] in (1,2,4,8) and
                r['along_n'] in (0,1) and r['n']>0 and r['k']>0 and
                all(isinstance(m,int) and m>=0 for m in r['rows']), 'Invalid ready geometry')
        require(len(r['ctas'])==r['comm']+r['compute'],'Missing CTA summaries')
        expected=grouped_ready_checks(r)
        active=[]
        for c,a in enumerate(r['ctas']):
            require(len(a)==8 and all(isinstance(v,int) and v>=0 for v in a), 'Invalid CTA record')
            begin,end,joined,checks,wait,maximum,first,longs=a
            require(0<begin<=end<=joined,'Invalid role timestamps')
            require(checks==(0 if c<r['comm'] else expected[c-r['comm']]),
                    'Ready count differs from tile schedule')
            require(first<=maximum<=wait<=end-begin and longs<=checks and longs*1000<=wait,
                    'Invalid ready counters')
            if not checks:
                require(wait==maximum==first==longs==0,'Waits on inactive CTA')
            else:
                active.append(dict(cta=c,role_ns=end-begin,end_ns=end,wait_ns=wait,
                    checks=checks,max_wait_ns=maximum,first_wait_ns=first,long_checks=longs))
        require(bool(active),'No active compute CTA')
        critical=max(active,key=lambda a:a['end_ns'])
        origin=min(a[0] for a in r['ctas'])
        sums={k:sum(a[k] for a in active) for k in ('role_ns','wait_ns','checks','first_wait_ns','long_checks')}
        summaries.append(dict(rank=r['rank'],active_compute_ctas=len(active),
            local_role_us=(max(a[1] for a in r['ctas'])-origin)/1000,
            finalize_done_us=(max(a[2] for a in r['ctas'])-origin)/1000,
            grid_finalize_us=(max(a[2] for a in r['ctas'])-max(a[1] for a in r['ctas']))/1000,
            comm_done_us=(max(a[1] for a in r['ctas'][:r['comm']])-origin)/1000,
            compute_done_us=(max(a['end_ns'] for a in active)-origin)/1000,
            critical_cta=critical['cta'],critical_role_us=critical['role_ns']/1000,
            critical_wait_us=critical['wait_ns']/1000,
            critical_wait_fraction=critical['wait_ns']/critical['role_ns'],
            critical_first_wait_us=critical['first_wait_ns']/1000,
            critical_later_wait_us=(critical['wait_ns']-critical['first_wait_ns'])/1000,
            max_wait_us=max(a['max_wait_ns'] for a in active)/1000,
            weighted_wait_fraction=sums['wait_ns']/sums['role_ns'],
            later_wait_fraction=(sums['wait_ns']-sums['first_wait_ns'])/sums['role_ns'],
            long_check_fraction=sums['long_checks']/sums['checks'],checks=sums['checks'],
            total_role_ns=sums['role_ns'],total_wait_ns=sums['wait_ns'],
            total_rows=sum(r['rows']),panels=sum((m+127)//128 for m in r['rows'])))
    critical=max(summaries,key=lambda r:r['local_role_us'])
    return dict(config={k:ranks[0][k] for k in fields},ranks=summaries,
                critical=critical,weighted_wait_fraction=sum(r['total_wait_ns'] for r in summaries)/
                sum(r['total_role_ns'] for r in summaries))


def export_grouped_ready(run):
    """Audit one untimed batch, retaining explicit memory skips."""
    root=Path(run); receipt=json.loads((root/'fetched.json').read_text())
    require(receipt['state']=='succeeded' and receipt['exit_code']==receipt['work_exit_code']==0,
            'Ready-summary run did not succeed')
    archive=root/'artifacts.tar.gz'
    require(file_digest(archive)==receipt['artifact_sha256'],'Ready archive mismatch')
    with tarfile.open(archive,'r:gz') as stream:
        def read(name):
            m=stream.getmember('control/'+name)
            require(m.isfile() and m.size<64<<20,'Invalid ready member')
            return stream.extractfile(m).read()
        job=json.loads(read('job.json')); contract=json.loads(read('grouped-contract.json'))
        installed=json.loads(read('source-installed.json')); env=json.loads(read('environment.json'))
        require(all(job[k]==receipt[k] for k in ('run_id','source_id','node')) and
                installed['source_id']==job['source_id'] and installed['files']==job['files'],
                'Ready source provenance mismatch')
        require(contract['build']['inputs']==grouped_build_inputs(job) and
                contract['build']['environment_fingerprint']==env['fingerprint']==receipt['environment_fingerprint'],
                'Ready build/environment mismatch')
        log=read(f"attempt{receipt['attempt']}.log").decode()
        require(job.get('grouped_ready_summary') and job.get('grouped_profile') and
                not job.get('grouped_measure') and contract['profiling'] and contract['build']['profiling'],
                'Not a ready-summary diagnostic')
        require('PASSED grouped-batch:' in log,'Missing batch completion')
        cases=grouped_batch_cases(job); require(bool(cases),'Expected summary batch')
        out=[]
        for case in cases:
            i=case['index']; entry=dict(case_index=i,case=case['case'],policy=case['policy'],
                tail_balance=case['tail_balance'],buffer_rows=case['buffer_rows'])
            states=re.findall(rf'^CASE grouped index={i} state=(completed|skipped)(?: reason=(\w+))?',log,re.M)
            require(len(states)==1,'Missing/duplicate case result')
            status,reason=states[0]
            if status=='skipped':
                require(reason in ('insufficient_memory','cuda_oom'),'Unexpected skipped case')
                out.append(entry|dict(status='memory_skipped',reason=reason)); continue
            ranks=[json.loads(read(f'grouped-ready-{i:04d}-rank-{r}.json')) for r in range(job['world'])]
            g=case['geometry']; p=grouped_policy(job|dict(grouped_policy=case['policy']))
            require(all(r['world']==job['world'] and r['n']==2*g['f'] and r['k']==g['h'] and
                        all(r[k]==p[k] for k in p) for r in ranks),'Summary differs from requested case')
            require(all(len(r['rows'])==g['experts']//job['world'] for r in ranks) and
                    sum(sum(r['rows']) for r in ranks)==job['world']*g['tokens_per_rank']*g['topk'],
                    'Ready routed row count mismatch')
            out.append(entry|dict(status='valid',summary=grouped_ready_summary(ranks)))
        require(log.count('PROFILE grouped: globaltimer')==sum(r['status']=='valid' for r in out),
                'Missing profile correctness completion')
    return dict(run=root.name,source_id=receipt['source_id'],binary_sha256=contract['build']['binary_sha256'],
                physical_devices=contract['physical_devices'],rows=out)


def collect_ready_plan(plan_path,runs_root):
    plan=json.loads(Path(plan_path).read_text()); require(plan['schema']=='grouped-ready-plan-v1','Invalid ready plan')
    jobs={}
    for path in Path(runs_root).glob('*/job.json'):
        # The experiment prefix narrows discovery without touching cloud state.
        if 'grouped-ready-full-' not in path.read_text(): continue
        job=json.loads(path.read_text()); name=job.get('experiment','')
        if name.startswith('grouped-ready-full-'):
            index=int(name.rsplit('-',1)[1])
            require(0<=index<len(plan['batches']),'Unexpected ready batch')
            if job['source_run']!=plan['batches'][index].get('source_run',plan['source_run']): continue
            require(index not in jobs,'Repeated ready batch')
            jobs[index]=(path.parent,job)
    rows=[]; evidence=[]; completed=0
    for i,b in enumerate(plan['batches']):
        measured=None
        if i in jobs:
            root,job=jobs[i]
            require(job['source_run']==b.get('source_run',plan['source_run']) and job['world']==b['ep'] and
                    job['grouped_cases']==';'.join(b['entries']), 'Changed ready batch plan')
            if (root/'fetched.json').exists():
                measured=export_grouped_ready(root); completed+=1
                evidence.append({k:v for k,v in measured.items() if k!='rows'})
        for j,entry in enumerate(b['entries']):
            case,policy,direction=entry.split('@'); g=list(map(int,case.split(',')))
            row=dict(zip(('h','f','experts','topk','target_rows'),g))
            row.update(models=plan['models'][','.join(map(str,g[:4]))],ep=b['ep'],policy=policy,
                       status='pending',batch=i,case_index=j)
            if measured:
                r=measured['rows'][j]
                require(r['case']==case and r['policy']==policy,'Changed ready result')
                row.update(status=r['status'],run=measured['run'])
                if r['status']=='valid':
                    s=r['summary']; row.update(s['critical'])
                    row['all_rank_wait_fraction']=s['weighted_wait_fraction']
                    row['wait_class']='high' if row['critical_wait_fraction']>=.5 else 'medium' if row['critical_wait_fraction']>=.2 else 'low'
            rows.append(row)
    require(len(rows)==plan['case_count'],'Ready plan coverage mismatch')
    return dict(schema='grouped-ready-full-v1',note=plan['note']+
        ' Wait fraction = load-warp ready check/poll time divided by compute CTA role time, not Tensor Core idle. '
        'Critical CTA is the last compute CTA on the rank with longest local role envelope. '
        'High >=50%, medium 20-50%, low <20%. First wait is separated; raw all-rank counters remain in artifacts.',
        completed_batches=completed,total_batches=len(plan['batches']),coverage=dict(Counter(r['status'] for r in rows)),
        wait_classes=dict(Counter(r['wait_class'] for r in rows if r['status']=='valid')),evidence=evidence,rows=rows)


def ready_markdown(table):
    lines=['# Dispatch + BF16 Grouped GEMM — ready 等待轻量统计','',table['note'],'',
        f"完成批次 {table['completed_batches']}/{table['total_batches']}；覆盖 {table['coverage']}；等待分类 {table['wait_classes']}。",'',
        'M = 每个 expert 的目标 token 行数。关键 CTA = 最长本地角色所在 rank 上最后完成的计算 CTA。',
        '等待占比不能视作端到端吞吐损失；加载 warp 等 ready 时，之前提交的 MMA 可能仍在执行。',
        '策略 N/K/AlongN/swizzle/通信CTA/计算CTA；不改变正式性能表。','',
        '## 高等待点（≥50%）','',
        '| 模型 | EP | M | 等待占比范围 |','|---|---:|---|---:|']
    groups=defaultdict(list)
    for r in table['rows']:
        if r['status']=='valid' and r['wait_class']=='high': groups[tuple(r['models']),r['ep']].append(r)
    for (models,ep),rows in groups.items():
        waits=[100*r['critical_wait_fraction'] for r in rows]
        lines.append('| '+', '.join(models)+f' | {ep} | '+', '.join(str(r['target_rows']) for r in rows)+
                     f' | {min(waits):.1f}–{max(waits):.1f}% |')
    lines+=['','## 完整矩阵（含缺失点）','',
        '| 模型 | EP | M | 策略 | 等待占比 | 全rank加权等待 | 首等 μs | 后续等 μs | 计算角色 μs | 通信完成 μs | 计算完成 μs | Join/finalize μs | 状态 |',
        '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in table['rows']:
        cells=[', '.join(r['models']),str(r['ep']),str(r['target_rows']),r['policy']]
        if r['status']=='valid':
            cells += [f"{100*r['critical_wait_fraction']:.1f}%",f"{100*r['all_rank_wait_fraction']:.1f}%"]
            cells += [f"{r[k]:.2f}" for k in ('critical_first_wait_us','critical_later_wait_us','critical_role_us','comm_done_us','compute_done_us','grid_finalize_us')]
            cells += [r['wait_class']]
        else: cells+=['—']*8+[r['status']]
        lines.append('| '+' | '.join(cells)+' |')
    return '\n'.join(lines)+'\n'


def catalog_markdown(table):
    """Readable view of the same catalog; never fill missing timings from seeds."""
    require(table['schema']=='grouped-bf16-current-v1','Unexpected table schema')
    lines=['# BF16 Grouped GEMM — current measured results', '', table['note'], '',
        f"Completed batches: {table['completed_batches']}/{table['total_batches']}. "
        f"Coverage: {json.dumps(table['coverage'],sort_keys=True)}.", '',
        'M is target rows per expert; exact routed row counts remain in the raw evidence. '
        'Times are milliseconds, throughput is useful PFLOPS/GPU. '
        'Config = N/K/Along/swizzle/communication CTAs/compute CTAs (tile M=128). '
        'Reference: Lt = tuned cuBLASLt Graph sequence; Grouped = native cuBLAS grouped. '
        'The JSON catalog retains source, routing and reference-run provenance.', '']
    for direction in ('dispatch','combine'):
        lines += [f'## {direction.capitalize()}', '',
            '| Models | EP | M | Fused ms | Fused P | Library P | Retained | Reference | Config | Status |',
            '|---|---:|---:|---:|---:|---:|---:|---|---|---|']
        for r in table['rows']:
            if r['direction']!=direction: continue
            cells=[', '.join(r['models']),str(r['ep']),str(r['target_rows'])]
            if r['status']=='valid':
                c=r['config']
                policy='/'.join(map(str,(c['tile_n'],c['tile_k'],
                    'N' if c['along_n'] else 'M',c['swizzle'],c['comm'],c['compute'])))
                method={'cublaslt_sequence_tuned':'Lt','cublas_grouped_default':'Grouped'}[r['reference_method']]
                cells += [f"{r['fused_ms']:.6f}",f"{r['fused_pflops']:.3f}",
                    f"{r['reference_pflops']:.3f}",f"{100*r['retention']:.1f}%",method,policy]
            else:
                cells += ['—']*6
            cells.append(r['status'])
            lines.append('| '+' | '.join(cells)+' |')
        lines.append('')
    return '\n'.join(lines)


def component_headroom(base, records):
    """Refresh F/G/R diagnostics without changing the frozen strong reference.

    Repeated measurements of the same workload use the latest complete run, not
    the fastest run.  F, G and R therefore always come from one run, source,
    route and explicit policy.  The strong full-SM library reference remains the
    audited reference in ``base``; absent references stay absent.
    """
    require(base.get('schema') is not None and isinstance(base.get('rows'), list),
            'Invalid base headroom table')
    expected={tuple(r[k] for k in ('h','f','experts','topk','target_rows','ep')):r
              for r in base['rows'] if r.get('status')=='valid'}
    require(len(expected)==sum(r.get('status')=='valid' for r in base['rows']),
            'Duplicate valid workload in base table')
    measured={}
    for record in records:
        require(not record.get('skipped') and record.get('fused_only') and
                not record.get('search'), 'Expected a completed component run')
        g=record['geometry']
        key=tuple(g[k] for k in ('h','f','experts','topk','target_rows'))+(record['world'],)
        require(key in expected, 'Component workload outside base table')
        stamp=(record['run_id'], record.get('case_index') or 0)
        if key not in measured or stamp>measured[key][0]: measured[key]=(stamp,record)
    require(set(measured)==set(expected), 'Incomplete component matrix')
    identities={(r['source_id'],r['environment_fingerprint']) for _,r in measured.values()}
    require(len(identities)==1, 'Mixed component source or environment')
    rows=[]
    for original in base['rows']:
        row={k:original[k] for k in ('models','ep','h','f','experts','topk','target_rows')}
        if original.get('status')!='valid':
            row.update(status=original.get('status','not_measured'),missing=original.get('missing',[]))
            rows.append(row); continue
        key=tuple(original[k] for k in ('h','f','experts','topk','target_rows','ep'))
        _,record=measured[key]
        require(record['routing_ids']['dispatch']==original['routing_id'],
                'Component routing differs from base reference')
        modes={r['mode']:r for r in record['rows'] if r['direction']=='dispatch'}
        require(set(modes)=={'fused','transport_body','cutlass_matched'} and
                all(r['valid'] for r in modes.values()), 'Incomplete or unstable F/G/R')
        require(len({tuple(sorted(r['config'].items())) for r in modes.values()})==1,
                'F/G/R configuration mismatch')
        fused,gemm,transport=(modes[m] for m in ('fused','cutlass_matched','transport_body'))
        ideal=max(gemm['ms'],transport['ms'])
        reference=original.get('reference')
        row.update(status='valid' if reference else 'reference_missing',
            config=fused['config'],routing_id=record['routing_ids']['dispatch'],
            source_id=record['source_id'],environment_fingerprint=record['environment_fingerprint'],
            measurement=[record['run_id'],record.get('case_index')],
            old_fused_ms=original['fused_ms'],fused_ms=fused['ms'],fused_pflops=fused['pflops'],
            fused_improvement=original['fused_ms']/fused['ms']-1,
            gemm_ms=gemm['ms'],gemm_pflops=gemm['pflops'],
            transport_ms=transport['ms'],ideal_ms=ideal,
            fusion_uplift_raw=fused['ms']/ideal-1,
            fusion_uplift_required=max(0.0,fused['ms']/ideal-1),
            limiting_component='gemm' if gemm['ms']>=transport['ms'] else 'transport',
            reference=reference)
        if reference:
            row.update(actual_retention=reference['ms']/fused['ms'],
                ideal_retention=reference['ms']/ideal,
                gemm_uplift_raw=gemm['ms']/reference['ms']-1,
                gemm_uplift_required=max(0.0,gemm['ms']/reference['ms']-1))
        else:
            row.update(actual_retention=None,ideal_retention=None,
                gemm_uplift_raw=None,gemm_uplift_required=None)
        rows.append(row)

    def geo(values):
        values=list(values); require(values,'Empty summary cohort')
        return math.exp(statistics.mean(math.log(v) for v in values))
    summary=[]
    for ep in (4,8):
        for load in ('all','small','large'):
            selected=[r for r in rows if r.get('fused_ms') and r['ep']==ep and
                (load=='all' or (r['target_rows']<192)==(load=='small'))]
            referenced=[r for r in selected if r['reference'] is not None]
            summary.append(dict(ep=ep,load=load,measured=len(selected),referenced=len(referenced),
                fused_improvement=geo(1+r['fused_improvement'] for r in selected)-1,
                fusion_uplift_required=geo(1+r['fusion_uplift_required'] for r in selected)-1,
                actual_retention=geo(r['actual_retention'] for r in referenced),
                ideal_retention=geo(r['ideal_retention'] for r in referenced),
                gemm_uplift_required=geo(1+r['gemm_uplift_required'] for r in referenced)-1,
                gemm_limiting=sum(r['limiting_component']=='gemm' for r in selected)))
    source_id,environment_fingerprint=next(iter(identities))
    return dict(schema='grouped-bf16-component-headroom-v1',source_id=source_id,
        environment_fingerprint=environment_fingerprint,
        note=('F/G/R are same-run Graph 10+50 dual-payload max-rank measurements. '
              'Ideal=max(G,R) is diagnostic, not a guaranteed attainable time. '
              'The strong full-SM CUTLASS/DeepGEMM reference is preserved from the base table; '
              'two memory-limited points have no strong reference. Repeated identical workloads '
              'use the latest complete run, never the fastest repeat.'),summary=summary,rows=rows)


def component_headroom_markdown(table):
    require(table['schema']=='grouped-bf16-component-headroom-v1','Unexpected headroom schema')
    lines=['# BF16 Dispatch + Grouped GEMM — component headroom','',table['note'],'',
        '## Summary','',
        '| EP | Load | Points | Old→new | Actual retention | Ideal retention | Fusion uplift needed | GEMM uplift needed | GEMM-limited |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in table['summary']:
        lines.append(f"| {s['ep']} | {s['load']} | {s['measured']} | {100*s['fused_improvement']:+.2f}% | "
            f"{100*s['actual_retention']:.1f}% | {100*s['ideal_retention']:.1f}% | "
            f"{100*s['fusion_uplift_required']:.1f}% | {100*s['gemm_uplift_required']:.1f}% | "
            f"{s['gemm_limiting']}/{s['measured']} |")
    lines += ['', '## Full matrix', '',
        '| Models | EP | M | Old F ms | New F ms | Old→new | G ms | R ms | Strong ms | Actual | Ideal | Fusion need | GEMM need | Limit | Config | Status |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|']
    for r in table['rows']:
        cells=[', '.join(r['models']),str(r['ep']),str(r['target_rows'])]
        if r.get('fused_ms'):
            ref=r['reference']; c=r['config']
            cells += [f"{r['old_fused_ms']:.6f}",f"{r['fused_ms']:.6f}",
                f"{100*r['fused_improvement']:+.2f}%",f"{r['gemm_ms']:.6f}",
                f"{r['transport_ms']:.6f}",f"{ref['ms']:.6f}" if ref else '—',
                f"{100*r['actual_retention']:.1f}%" if ref else '—',
                f"{100*r['ideal_retention']:.1f}%" if ref else '—',
                f"{100*r['fusion_uplift_required']:.1f}%",
                f"{100*r['gemm_uplift_required']:.1f}%" if ref else '—',r['limiting_component'],
                '/'.join(str(c[k]) for k in ('tile_n','tile_k','along_n','swizzle','comm','compute'))]
        else: cells += ['—']*12
        cells.append(r['status']); lines.append('| '+' | '.join(cells)+' |')
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='*')
    parser.add_argument('--plan', type=Path, help='refresh the complete frozen plan from local fetched runs')
    parser.add_argument('--runs-root', type=Path)
    parser.add_argument('--fill-plan', type=Path, help='explicit frozen memory-skip retest plan for external references')
    parser.add_argument('--refinement-run', type=Path, action='append', default=[],
                        help='add an audited same-source fusion candidate run to --plan (repeatable)')
    parser.add_argument('--markdown', action='store_true', help='render --plan as the full readable table')
    parser.add_argument('--case-index', type=int)
    parser.add_argument('--all-cases', action='store_true')
    parser.add_argument('--handoff-profile', action='store_true', help='audit one diagnostic run and emit Perfetto JSON')
    parser.add_argument('--ready-summary', action='store_true', help='audit lightweight ready-wait batches')
    parser.add_argument('--ready-plan',type=Path,help='summarize a frozen lightweight diagnostic matrix')
    parser.add_argument('--output',type=Path,help='ready-plan report stem, writes generated .json and .md')
    parser.add_argument('--headroom-base',type=Path,
                        help='refresh a frozen headroom table from component runs under --runs-root')
    parser.add_argument('--component-prefix',default='grouped-cohort-components-final-',
                        help='experiment prefix selected by --headroom-base')
    parser.add_argument('--comparison', action='store_true', help='compact fused/library comparison JSON')
    parser.add_argument('--catalog', action='store_true', help='full requested matrix, including missing points')
    args = parser.parse_args()
    if args.headroom_base:
        if not args.runs_root or not args.output or args.runs or args.plan or args.ready_plan:
            parser.error('--headroom-base needs --runs-root and --output')
        records=[]
        for path in sorted(args.runs_root.glob('*/job.json')):
            job=json.loads(path.read_text())
            if job.get('experiment','').startswith(args.component_prefix) and (path.parent/'fetched.json').is_file():
                records.extend(audit_cases(path.parent))
        table=component_headroom(json.loads(args.headroom_base.read_text()),records)
        args.output.with_suffix('.json').write_text(json.dumps(table,separators=(',',':'))+'\n')
        args.output.with_suffix('.md').write_text(component_headroom_markdown(table))
        print(f"Component headroom: {len([r for r in table['rows'] if r.get('fused_ms')])} measured; "
              f"{args.output.with_suffix('.md')}")
        sys.exit(0)
    if args.ready_plan:
        if not args.runs_root or not args.output or args.runs or args.plan:
            parser.error('--ready-plan needs --runs-root and --output')
        table=collect_ready_plan(args.ready_plan,args.runs_root)
        args.output.with_suffix('.json').write_text(json.dumps(table,separators=(',',':'))+'\n')
        args.output.with_suffix('.md').write_text(ready_markdown(table))
        print(f"Ready summary: batches {table['completed_batches']}/{table['total_batches']}; "
              f"coverage {table['coverage']}; classes {table['wait_classes']}; {args.output.with_suffix('.md')}")
        sys.exit(0)
    if args.ready_summary:
        if not args.runs or args.plan or args.handoff_profile:
            parser.error('--ready-summary requires run directories')
        print(json.dumps([export_grouped_ready(r) for r in args.runs],separators=(',',':'),allow_nan=False))
        sys.exit(0)
    if args.handoff_profile:
        if len(args.runs)!=1 or args.plan or args.runs_root or args.all_cases or args.case_index is not None:
            parser.error('--handoff-profile requires exactly one run')
        print(json.dumps(export_grouped_handoff(args.runs[0]),separators=(',',':'),allow_nan=False))
        sys.exit(0)
    if args.plan is not None:
        if args.runs or args.runs_root is None or args.case_index is not None or args.all_cases or args.comparison or args.catalog:
            parser.error('--plan requires --runs-root, without direct-run options')
        external=json.loads(args.plan.read_text()).get('schema')=='grouped-external-reference-v1'
        if external and args.refinement_run: parser.error('External plan does not accept fusion refinements')
        if args.fill_plan and not external: parser.error('--fill-plan requires an external plan')
        table=collect_external_plan(args.plan,args.runs_root,args.fill_plan) if external else collect_plan(args.plan,args.runs_root,args.refinement_run)
        print((external_markdown(table) if external else catalog_markdown(table)) if args.markdown else json.dumps(table,indent=2,allow_nan=False))
        sys.exit(0)
    if args.markdown or args.refinement_run or args.fill_plan:
        parser.error('--markdown/--refinement-run require --plan and --runs-root')
    if not args.runs or args.runs_root is not None:
        parser.error('provide run directories, or --plan with --runs-root')
    if args.all_cases and args.case_index is not None:
        parser.error('--all-cases and --case-index are mutually exclusive')
    if args.catalog and args.comparison:
        parser.error('--catalog and --comparison are mutually exclusive')
    records=[r for path in args.runs for r in
             (audit_cases(path) if args.all_cases else [audit_run(path,args.case_index)])]
    print(json.dumps(catalog_rows(records) if args.catalog else comparison_rows(records) if args.comparison else records,
                     indent=2, allow_nan=False))
