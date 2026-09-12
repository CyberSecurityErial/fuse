"""Explicit MXFP8 OProj baseline matrix, using the ordinary L20D controller.

No parameter search or runtime policy is hidden here. Identical physical
geometries run once; model aliases remain visible in the final table. Successful
runs are resumed from their receipt and audited before publishing throughput.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tarfile

import l20d
from summarize_sm103_mxfp8_fused import audit_run
import summarize_sm103_fused as evidence
from summarize_sm103_mxfp8_search import fields

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path('/Users/admin/workspace/fuse_midfile/l20d')
SERVICE_NAMES = ('compute_reference', 'copy_reference', 'producer_reference')


def comm_budget_command(row, workspace):
    """One process, fixed GEMM, five budgets; never substitute another tile."""
    g = row['gemm']
    if (g['tile_m'], g['tile_n'], g['tile_k'], g['stages']) != (128,256,128,0):
        raise ValueError('Pure winner needs native fusion integration: ' + row['id'])
    if row['requires_cross_peer_k_tile']:
        raise ValueError('Pure winner crosses an A ready boundary: ' + row['id'])
    return [sys.executable, str(ROOT / 'scripts/l20d.py'), 'run', 'fused-smoke',
        '--node', '09', '--workspace', workspace, '--mpi', '--mxfp8',
        '--fused-direction', 'oproj', '--directions', 'oproj', '--fused-launch', 'graph',
        '--input-generator', 'gpu_philox', '--oproj-policy-list', 'm128n256',
        '--qkv-policy-list', 'm128n256', '--mxfp8-weight-preparation', 'comm',
        '--mxfp8-epilogue-n', str(g['epilogue_n']), '--oproj-raster', g['raster'],
        '--max-swizzle-size', str(g['max_swizzle_size']), '--oproj-comm-layout', 'rows',
        '--comm-sm-list', ','.join(str(c['comm_ctas']) for c in row['proposed_candidates']),
        '--causal', '--world', str(row['world']), '--global-seq', str(row['global_seq']),
        '--hidden', str(row['hidden']), '--q-heads', str(row['q_heads']),
        '--kv-heads', '8', '--head-dim', str(row['head_dim']),
        '--experiment', 'v21-oproj-comm-budget', '--timeout', '600']


def gemm_parameters(name):
    """An explicit pure winner is input, never silently narrowed to the baseline."""
    match = re.fullmatch(r'm128n(128|256)k(128|256)e(32|64)s(0|2|3)sw(1|2|4|8)(M|N)', name)
    if not match:
        raise ValueError('Unregistered pure CUTLASS configuration: ' + name)
    n, k, e, stages, swizzle = map(int, match.groups()[:5])
    if (stages == 3 and k != 128) or (stages == 2 and n == k == 256):
        raise ValueError('Configuration outside the compiled CUTLASS search family')
    return dict(tile_m=128, tile_n=n, tile_k=k, epilogue_n=e, stages=stages,
                max_swizzle_size=swizzle, raster='along_m' if match[6] == 'M' else 'along_n')


def service_plan(rows, pure):
    """Only pure GEMM layouts define anchors; fused/manual timings are not read.

    Measure at S128K, holding S256K/S512K out. Three independently measured
    budgets avoid assuming a linear compute-SM curve or fitting intermediate
    communication winners. Quantization size is constant as sequence grows.
    """
    anchors = {}
    for row in manual_plan(rows, pure)['rows']:
        if row['tuning_status'] != 'comm_budget_first':
            continue
        g = row['gemm']
        key = (row['n'],row['k'],row['world'],g['epilogue_n'],g['raster'],g['max_swizzle_size'])
        if key not in anchors:
            anchor = dict(row, global_seq=131072, m=131072//row['world'],
                          proposed_candidates=[dict(comm_ctas=c) for c in (32,64,96)])
            anchor['id'] = f"n{key[0]}k{key[1]}cp{key[2]}e{key[3]}{key[4]}sw{key[5]}"
            anchors[key] = anchor
    return list(anchors.values())


def execute_job(command):
    process = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
    matches = re.findall(r'"run_id": "([0-9]{8}-[0-9]{6}-[a-f0-9]+)"', process.stdout)
    if process.returncode or not matches:
        raise RuntimeError(process.stdout[-1200:])
    return Path('/Users/admin/workspace/fuse_midfile/l20d') / matches[0]


def overlap_plan(rows, controls):
    """Measure current services, then test the front-loaded-A hypothesis.

    AlongM visits all M in its first N band. The opposite raster is one controlled
    candidate, not a universal replacement: retain tile, swizzle and the budget.
    The native communication order is derived from each actual GEMM traversal.
    Already-AlongN cases also test AlongM; do not assume either raster is optimal.
    This is an offline diagnostic, not an Auto training table.
    """
    by_id = {r['id']: r for r in controls['rows']}
    if len(by_id) != len(controls['rows']):
        raise ValueError('Duplicate overlap control')
    tasks = []
    for row in rows:
        if row['status'].startswith('unsupported'):
            continue
        control = by_id[row['id']]
        g = control['gemm']
        c = int(control['manual']['configuration']['comm_sm'])
        if not 0 < c < 148 or g['raster'] not in ('along_m', 'along_n'):
            raise ValueError('Overlap audit needs an explicit budget and raster')
        other = 'along_n' if g['raster'] == 'along_m' else 'along_m'
        variants = [dict(name='current', gemm=g), dict(name=other, gemm=dict(g, raster=other))]
        geometry = {k:row[k] for k in ('id','models','hidden','q_heads','head_dim',
                                       'world','global_seq','m','n','k')}
        tasks.append(dict(geometry, status='pending', comm_ctas=c, variants=variants))
    return tasks


def overlap_measurement(folder, row, gemm, source_id, candidate_id=1):
    """Reaudit the actual four components, including reused local evidence."""
    results = {name: audit_run(folder, candidate_id, name)
               for name in ('fused',) + SERVICE_NAMES}
    for result in results.values():
        conf = result['configuration']
        expected = dict(comm_sm=row['comm_ctas'], raster=gemm['raster'],
            tile_m=gemm['tile_m'], tile_n=gemm['tile_n'], tile_k=gemm['tile_k'],
            max_swizzle_size=gemm['max_swizzle_size'], scheduled_compute_ctas=148-row['comm_ctas'])
        if any(str(conf.get(k)) != str(v) for k, v in expected.items()):
            raise ValueError('Overlap component configuration mismatch')
        if (result['source_id'] != source_id or result['epilogue_n'] != gemm['epilogue_n']
                or any(result[k] != row[k] for k in ('m','n','k','world','global_seq'))):
            raise ValueError('Overlap component source/geometry mismatch')
    if len({r['binary_sha256'] for r in results.values()}) != 1:
        raise ValueError('Overlap components use different binaries')
    f, c, p = (results[k]['p50_ms'] for k in ('fused','compute_reference','producer_reference'))
    return dict(run_id=Path(folder).name, candidate_id=candidate_id, results=results,
        fused_over_compute=c/f, serial_reference_saving_ms=c+p-f,
        interpretation='C+P-F is a separate-service comparison, NOT measured overlap time')


def overlap_markdown(report):
    lines = ['# MXFP8 OProj traversal / service comparison', '',
        'Graph 10+50, per-rank PFLOPS. C uses the same compute CTA budget and resource floor.',
        'F includes A2A + W quantization + GEMM. P is independent A+W production.',
        'Separate-service timings do not directly measure concurrent overlap or Tensor Core utilization.',
        'Only raster changes in a paired comparison; native communication scheduling follows it.', '',
        'cuBLASLt is the separately audited full-148-SM pure MXFP8 reference, without quantization or communication; not remeasured with this direction sweep.', '',
        '| Model | S | CP | Comm | Original layout | AlongM F | AlongN F | N versus M | C(M) | C(N) | P(M) ms | P(N) ms | Pure cuBLASLt | N / C(N) | N / cuBLASLt |',
        '|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in report['rows']:
        values = row.get('measurements', {})
        g = row.get('gemm')
        directions = {(g['raster'] if name == 'current' else name):sample
                      for name,sample in values.items()}
        old, new = directions.get('along_m'), directions.get('along_n')
        def metric(sample, component, field):
            return f"{sample['results'][component][field]:.3f}" if sample else '—'
        gain = (f"{100*(old['results']['fused']['p50_ms']/new['results']['fused']['p50_ms']-1):+.2f}%"
                if old and new else '—')
        pure = row.get('pure_cublaslt')
        lt = f"{pure['pflops']:.3f}" if pure and pure['status'] == 'passed' else '—'
        own_ratio = (f"{100*new['results']['fused']['pflops_per_rank']/new['results']['compute_reference']['pflops_per_rank']:.1f}%"
                     if new else '—')
        lt_ratio = (f"{100*new['results']['fused']['pflops_per_rank']/pure['pflops']:.1f}%"
                    if new and lt != '—' else '—')
        layout = f"{g['raster']}/sw{g['max_swizzle_size']}" if g else 'unsupported'
        for model in row['models']:
            lines.append(f"| {model} | {row['global_seq']//1024}K | {row['world']} | {row.get('comm_ctas','—')} | {layout} | "
                + ' | '.join([metric(old,'fused','pflops_per_rank'), metric(new,'fused','pflops_per_rank'), gain,
                    metric(old,'compute_reference','pflops_per_rank'), metric(new,'compute_reference','pflops_per_rank'),
                    metric(old,'producer_reference','p50_ms'), metric(new,'producer_reference','p50_ms'),
                    lt, own_ratio, lt_ratio]) + ' |')
    summary = report.get('summary')
    if summary:
        lines += ['', '## Audited result', '',
            f"{summary['physical_points']} physical points / {summary['logical_rows']} model rows; "
            f"{summary['configurations']} configurations, {summary['component_measurements']} F/C/R/P measurements. "
            'Original receipts reaudited; one binary/environment.', '',
            f"AlongN wins {summary['along_n_wins']}/{summary['physical_points']} at these fixed settings: "
            f"geometric mean +{100*(summary['n_over_m_gmean']-1):.2f}% versus AlongM, "
            f"range +{100*(summary['n_over_m_min']-1):.2f}% to +{100*(summary['n_over_m_max']-1):.2f}%.", '',
            f"Selecting the better direction gives geometric mean +{100*(summary['best_over_original_gmean']-1):.2f}% "
            f"versus the original layout; {summary['original_along_n']} already-AlongN points are not new improvements. "
            'Physical geometries weighted equally; aliases are not double-counted.', '',
            'This does not establish a universal AlongN rule. Defaults/Auto are unchanged. '
            'Only six points have paired detailed overlap traces; the others have formal F/C/R/P comparisons. '
            'See [six-profile timing analysis](overlap-profile-current.md).']
    return '\n'.join(lines)+'\n'


def window_markdown(report):
    """Render audited window/budget measurements; never fill an unmeasured point."""
    budgets = tuple(str(c) for c in report.get('budgets', (20, 32, 48)))
    h, p = report.get('m_window_tiles', 64), report.get('n_group_tiles', 4)
    lines = ['# MXFP8 OProj — bounded-window / communication-budget comparison', '',
        'Graph 10+50; per-rank PFLOPS. F includes A2A, BF16 W quantization and MXFP8 GEMM.',
        f'New measurements use H={h}/P={p}; communication CTA candidates: {", ".join(budgets)}. '
        'Best means fastest measured F in these candidates, not global optimum or Auto.',
        'Baseline is the historical best recorded for each physical point; each actual run/source/configuration '
        'is retained under baseline in [the canonical JSON](window-full-current.json). '
        'This is not a same-binary ablation and cannot attribute the gain to the window alone.',
        'C uses the selected candidate\'s same compute budget, collective, window and resource floor. '
        'Pure cuBLASLt uses the full 148-SM device, without A2A or weight quantization.',
        'Partially covered rows show best-so-far; missing measurements stay blank. '
        'Geometric means include only completely covered physical points; model aliases are not counted twice.', '',
        '| Model | S | CP | Baseline F | New F | Gain | Best comm | New C | Pure cuBLASLt | F / C | F / cuBLASLt | Candidates |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    completed, paired, lt_ratios = [], [], []
    fmt = lambda x: f'{x:.3f}' if x is not None else '—'
    ratio = lambda x, y: f'{100*x/y:.1f}%' if x is not None and y else '—'
    for row in report['rows']:
        samples = {c: row.get('measurements', {}).get(c) for c in budgets}
        samples = {c: s for c, s in samples.items() if s and s.get('results')}
        best_comm = min(samples, key=lambda c: samples[c]['results']['fused']['p50_ms']) if samples else None
        best = samples[best_comm]['results'] if best_comm is not None else {}
        old = (row.get('baseline') or {}).get('results', {}).get('fused')
        b = old['pflops_per_rank'] if old else None
        f = best['fused']['pflops_per_rank'] if best else None
        c = best['compute_reference']['pflops_per_rank'] if best else None
        pure = row.get('pure_cublaslt') or {}
        lt = pure.get('pflops') if pure.get('status') == 'passed' else None
        speedup = old['p50_ms']/best['fused']['p50_ms'] if old and best else None
        gain = f'{100*(speedup-1):+.2f}%' if speedup is not None else '—'
        coverage = f'{len(samples)}/{len(budgets)}'
        if row.get('status') == 'failed': coverage += ' failed'
        for model in row['models']:
            lines.append(f'| {model} | {row["global_seq"]//1024}K | {row["world"]} | '
                + ' | '.join((fmt(b), fmt(f), gain, best_comm or '—', fmt(c), fmt(lt),
                              ratio(f, c), ratio(f, lt), coverage)) + ' |')
        if len(samples) == len(budgets) and row.get('status') == 'passed':
            completed.append((f, f/c))
            if speedup is not None: paired.append(speedup)
            if lt: lt_ratios.append(f/lt)
    lines += ['', f'Fully covered: {len(completed)}/{len(report["rows"])} physical points.']
    if completed:
        gm = lambda values: math.exp(sum(math.log(v) for v in values)/len(values))
        lines += [f'Completed-point geometric means: F={gm([v[0] for v in completed]):.3f} PFLOPS; '
                  f'F/C={100*gm([v[1] for v in completed]):.1f}%.']
        if paired:
            lines += [f'Versus historical baseline: {100*(gm(paired)-1):+.2f}% '
                      f'over {len(paired)} fully covered paired points.']
        if lt_ratios:
            lines += [f'F/full-device cuBLASLt: {100*gm(lt_ratios):.1f}% '
                      f'over {len(lt_ratios)} fully covered paired points.']
    return '\n'.join(lines) + '\n'


def run_overlap_audit(directory, rows, workspace, case_ids, source_run):
    if not source_run:
        raise ValueError('Overlap audit requires a frozen --source-run')
    if case_ids:
        rows = [r for r in rows if r['id'] in case_ids]
    controls = json.loads((directory/'acceptance-current.json').read_text())
    plan = overlap_plan(rows, controls)
    source_id = json.loads((RUNS/source_run/'job.json').read_text())['source_id']
    target = directory/'overlap-current.json'
    report = json.loads(target.read_text()) if target.exists() else dict(
        schema='sm103_mxfp8_oproj_overlap_v1', source_run=source_run, source_id=source_id,
        plan_sha256=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest(), rows=[
            dict(r, gemm=r['variants'][0]['gemm'], measurements={}) for r in plan] + [
            dict(r) for r in rows if r['status'].startswith('unsupported')])
    if (report['source_id'] != source_id or report['plan_sha256'] !=
            hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()):
        raise ValueError('Overlap source or frozen controls changed')
    if case_ids and not set(case_ids) <= {r['id'] for r in plan}:
        raise ValueError('Unknown/unsupported overlap cases')
    tasks = []
    for row in plan:
        for variant in row['variants']:
            task = dict(row, gemm=variant['gemm'], requires_cross_peer_k_tile=
                (row['k']//row['world']) % variant['gemm']['tile_k'] != 0,
                proposed_candidates=[dict(comm_ctas=row['comm_ctas'])])
            command = comm_budget_command(task,workspace) + ['--calibrate','--source-run',source_run]
            command[command.index('--experiment')+1] = 'v21-oproj-overlap-'+variant['name']
            if not case_ids or row['id'] in case_ids:
                tasks.append((row,variant,command))
    l20d.configure_workspace(workspace)
    l20d.write_json(target,report)
    by_id = {r['id']:r for r in report['rows']}
    total = sum(len(r['variants']) for r in plan)
    for row, variant, command in tasks:
        entry = by_id[row['id']]
        previous = entry['measurements'].get(variant['name'])
        if previous:
            if overlap_measurement(RUNS/previous['run_id'],row,variant['gemm'],source_id,
                                   previous['candidate_id']) != previous:
                raise ValueError('Overlap resume evidence changed')
            continue
        done = sum(len(r.get('measurements',{})) for r in report['rows'])
        print(f"RUN overlap {done+1}/{total} {row['id']} {variant['name']} c{row['comm_ctas']}",flush=True)
        folder = execute_job(command)
        measured = overlap_measurement(folder,row,variant['gemm'],source_id)
        identities = {(r['binary_sha256'],r['environment_fingerprint'])
                      for s in [measured,*entry['measurements'].values()]
                      for r in s['results'].values()}
        if len(identities) != 1:
            raise ValueError('Traversal comparison must use one binary/environment')
        entry['measurements'][variant['name']] = measured
        entry['status'] = 'passed' if len(entry['measurements']) == len(row['variants']) else 'partial'
        report['complete'] = all(len(by_id[r['id']]['measurements']) == len(r['variants']) for r in plan)
        l20d.write_json(target,report)
        (directory/'overlap-current.md').write_text(overlap_markdown(report))
        result = measured['results']
        suffix = ''
        other = row['variants'][1]['name']
        if all(k in entry['measurements'] for k in ('current',other)):
            a,b = (entry['measurements'][k]['results']['fused']['p50_ms'] for k in ('current',other))
            suffix = f' {other} gain={100*(a/b-1):+.2f}%'
        print(f"DONE overlap {done+1}/{total} {row['id']} {variant['name']} "
              f"F={result['fused']['pflops_per_rank']:.3f}P C={result['compute_reference']['pflops_per_rank']:.3f}P "
              f"F/C={100*measured['fused_over_compute']:.1f}%"+suffix,flush=True)


def run_service_calibration(directory, rows, workspace, source_run=None):
    pure = json.loads((directory/'gemm-search-current.json').read_text())
    plan = service_plan(rows, pure)
    target = directory/'calibration-current.json'
    report = json.loads(target.read_text()) if target.exists() else dict(
        schema='sm103_mxfp8_oproj_services_v1', pure_run_id=pure['run_id'], rows=[])
    if report['pure_run_id'] != pure['run_id']:
        raise ValueError('Service table belongs to another pure search')
    complete = {r['id'] for r in report['rows']}
    for row in plan:
        if row['id'] in complete:
            continue
        command = comm_budget_command(row, workspace) + ['--calibrate']
        if source_run:
            command += ['--source-run',source_run]
        command[command.index('--experiment')+1] = 'v21-oproj-services'
        print('RUN services ' + row['id'], flush=True)
        folder = execute_job(command)
        samples = []
        for i, budget in enumerate(row['proposed_candidates'], 1):
            # Deliberately omit fused measurements from the model input table.
            services = {name:audit_run(folder,i,name) for name in
                        ('compute_reference','copy_reference','producer_reference')}
            samples.append(dict(comm_ctas=budget['comm_ctas'], services=services))
        identities={(s['binary_sha256'],s['environment_fingerprint'])
                    for r in report['rows'] for p in r['samples'] for s in p['services'].values()}
        identities.update((s['binary_sha256'],s['environment_fingerprint'])
                          for p in samples for s in p['services'].values())
        if len(identities) != 1:
            raise ValueError('Service table requires one binary/environment identity')
        report['rows'].append(dict(id=row['id'],reference_m=row['m'],n=row['n'],k=row['k'],
            world=row['world'],gemm=row['gemm'],samples=samples))
        l20d.write_json(target,report)
        print(f"DONE services {len(report['rows'])}/{len(plan)} {row['id']}",flush=True)


def calibration_points(report, plan, audit=audit_run):
    """Export physical services only, rechecking original receipts, not winners."""
    expected = {r['id']: r for r in plan}
    if len(expected) != len(plan) or {r['id'] for r in report['rows']} != set(expected):
        raise ValueError('Incomplete/duplicate service anchor coverage')
    if len(report['rows']) != len(plan):
        raise ValueError('Duplicate calibration anchors')
    points, identities, keys = [], set(), set()
    for row in report['rows']:
        anchor, g = expected[row['id']], expected[row['id']]['gemm']
        for key in ('n','k','world','gemm'):
            if row[key] != anchor[key]:
                raise ValueError('Service geometry/layout mismatch: ' + key)
        if row['reference_m'] != anchor['m']:
            raise ValueError('Service reference M mismatch')
        if sorted(s['comm_ctas'] for s in row['samples']) != [32,64,96]:
            raise ValueError('Expected three independent measured budgets')
        for sample in row['samples']:
            c, services = sample['comm_ctas'], sample['services']
            if set(services) != set(SERVICE_NAMES):
                raise ValueError('Only independent C/R/P services may calibrate Auto')
            for name, s in services.items():
                actual = audit(RUNS/s['run_id'],s['candidate_id'],name)
                if actual != s:
                    raise ValueError('Service summary differs from original evidence')
                config = s['configuration']
                if (s['m'],s['n'],s['k'],s['world'],s['global_seq']) != (
                        anchor['m'],anchor['n'],anchor['k'],anchor['world'],131072):
                    raise ValueError('Independent service must use the S128K anchor')
                if s['component'] != name or int(config['comm_sm']) != c:
                    raise ValueError('Service component/budget mismatch')
                for key in ('tile_m','tile_n','tile_k','max_swizzle_size'):
                    if int(config[key]) != g[key]:
                        raise ValueError('Service GEMM mismatch: ' + key)
                if config['raster'] != g['raster'] or s['epilogue_n'] != g['epilogue_n']:
                    raise ValueError('Service raster/epilogue mismatch')
                if not math.isfinite(s['p50_ms']) or s['p50_ms'] <= 0:
                    raise ValueError('Invalid service time')
                identities.add((s['source_id'],s['binary_sha256'],s['environment_fingerprint']))
            cs = services['compute_reference']['configuration']
            for s in services.values():
                for key in ('dynamic_smem','effective_swizzle_size'):
                    if s['configuration'][key] != cs[key]:
                        raise ValueError('Service physical resources mismatch')
            point = dict(n=row['n'],k=row['k'],world=row['world'],sm_count=148,
                tile_m=g['tile_m'],tile_n=g['tile_n'],tile_k=g['tile_k'],
                epilogue_n=g['epilogue_n'],stage_policy=g['stages'],
                raster=int(g['raster']=='along_n'),swizzle=int(cs['effective_swizzle_size']),
                dynamic_smem_bytes=int(cs['dynamic_smem']),comm_ctas=c,compute_ctas=148-c,
                reference_m=row['reference_m'],
                **{key:services[name]['p50_ms']*1000 for key,name in
                   zip(('compute_us','copy_us','producer_us'),SERVICE_NAMES)})
            key = tuple(point[k] for k in point if k not in ('compute_us','copy_us','producer_us'))
            if key in keys:
                raise ValueError('Duplicate physical calibration key')
            keys.add(key); points.append(point)
    if len(identities) != 1:
        raise ValueError('Calibration requires one source/binary/environment')
    return sorted(points,key=lambda p:tuple(p.values())[:15]), next(iter(identities))


def export_calibration(directory, rows):
    pure = json.loads((directory/'gemm-search-current.json').read_text())
    report = json.loads((directory/'calibration-current.json').read_text())
    if report['pure_run_id'] != pure['run_id']:
        raise ValueError('Calibration belongs to another pure search')
    points, identity = calibration_points(report,service_plan(rows,pure))
    payload = dict(policy='minmax_bulk_v1',identity=identity,points=points)
    digest = hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:16]
    version = 'oproj_services_' + digest
    # Field order is the native aggregate declaration; no measured fused time
    # or selected manual budget is embedded. The generated fragment is reviewed
    # and applied to model_calibration.cuh, never silently modifying native code.
    lines = ['// S128K independent C/R/P; S256K/S512K are held out.',
             '// Source: '+identity[0],
             f'inline constexpr const char* kMxfp8OprojCalibrationVersion = "{version}";',
             f'inline constexpr std::array<Mxfp8OprojCalibrationPoint, {len(points)}> kMxfp8OprojCalibrationPoints{{{{']
    for point in points:
        lines.append('  {'+', '.join(format(v,'.17g') if isinstance(v,float) else str(v)
                                    for v in point.values())+'},')
    lines.append('}};')
    (directory/'calibration-current.inc').write_text('\n'.join(lines)+'\n')
    l20d.write_json(directory/'model-current.json',dict(version=version,**payload))
    print(f'EXPORTED {len(points)} independently audited services: {version}',flush=True)


def manual_plan(rows, pure):
    """Stage two freezes GEMM and varies only producer settings.

    First vary only the CTA budget: this is the cheapest controlled test of
    under-supply after integrating a faster GEMM. Queue group width remains
    derived from the actual compute budget and copy slots, not independently
    searched. A larger budget accelerating fusion supports a producer-service
    bottleneck; it does not distinguish A transfer from W quantization. Inspect
    those separately only if this bounded sweep does not close the gap.
    """
    if pure.get('schema') != 'sm103_mxfp8_cutlass_search_v1' or pure.get('partial') or pure.get('pending'):
        raise ValueError('Complete audited pure search required before fused tuning')
    by_shape = {(r['m'],r['n'],r['k']): r for r in pure['rows'] if r['status'] == 'passed'}
    if len(by_shape) != sum(r['status'] == 'passed' for r in pure['rows']):
        raise ValueError('Duplicate pure geometry')
    tasks = []
    for row in rows:
        if row['status'].startswith('unsupported'):
            tasks.append(dict(row, tuning_status='unsupported_head_partition'))
            continue
        key = row['m'], row['n'], row['k']
        winner = by_shape.get(key)
        if winner is None:
            skipped = any(r['id'] == f'm{key[0]}n{key[1]}k{key[2]}' and r['status'] == 'memory_skip'
                          for r in pure['rows'])
            if not skipped:
                raise ValueError('Missing pure winner: ' + row['id'])
            tasks.append(dict(row, tuning_status='pure_memory_skip'))
            continue
        params = gemm_parameters(winner['winner']['config'])
        tasks.append(dict(id=row['id'], models=row['models'],
            **{k:row[k] for k in ('m','n','k','hidden','q_heads','head_dim','world','global_seq')},
            tuning_status='comm_budget_first', gemm=params, pure_winner=winner['winner'],
            requires_cross_peer_k_tile=(row['k']//row['world']) % params['tile_k'] != 0,
            proposed_candidates=[dict(comm_ctas=c) for c in (8,16,32,48,64)]))
    return dict(schema='sm103_mxfp8_oproj_manual_plan_v1',
        pure_run_id=pure['run_id'], pure_artifact_sha256=pure['artifact_sha256'],
        scope='fixed pure GEMM; evidence-guided bounded producer experiments', rows=tasks)


def matrix():
    spec = importlib.util.spec_from_file_location('oproj_bench_catalog', ROOT / 'benchmarks/sm103/bench.py')
    catalog = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(catalog)
    groups = {}
    for name, model in catalog.load_shapes('oproj').MODELS.items():
        for cp in (4, 8):
            for seq in (131072, 262144, 524288):
                key = model.hidden, model.q_heads, model.head_dim, cp, seq
                if key not in groups:
                    groups[key] = dict(id=f'h{key[0]}q{key[1]}d{key[2]}cp{cp}s{seq}',
                        models=[], hidden=key[0], q_heads=key[1], head_dim=key[2], world=cp,
                        global_seq=seq, m=seq//cp, n=key[0], k=key[1]*key[2],
                        status='pending' if key[1] % cp == 0 else 'unsupported_head_partition')
                groups[key]['models'].append(name)
    # Establish the large-model baseline first, then complete the catalog.
    return sorted(groups.values(), key=lambda r: (r['hidden'] < 4096, r['global_seq'], r['world'], -r['hidden']))


def audit_pure(folder):
    """Full-SM cuBLASLt only; retain the first stable 50-sample round."""
    folder = Path(folder)
    fetched = json.loads((folder / 'fetched.json').read_text())
    digest = hashlib.sha256((folder / 'artifacts.tar.gz').read_bytes()).hexdigest()
    evidence.require(fetched.get('state') == 'succeeded' and fetched.get('exit_code') == 0
                     and fetched.get('artifact_sha256') == digest, 'Pure reference receipt/digest failed')
    with tarfile.open(folder / 'artifacts.tar.gz') as archive:
        job = json.load(archive.extractfile('control/job.json'))
        text = archive.extractfile('control/attempt1.log').read().decode()
        contract = json.load(archive.extractfile('control/gemm-probe-contract.json'))
    evidence.require(job.get('gemm_precision') == 'mxfp8' and not job.get('mxfp8_gemm_search'),
                     'Expected pure cuBLASLt MXFP8')
    configs = [fields(line) for line in text.splitlines() if line.startswith('config,pure_mxfp8,')]
    evidence.require(len(configs) == 1, 'Missing pure config')
    config = configs[0]
    for key, value in dict(launch='graph', warmup='10', samples='50', sm_budget='full_device',
            device_sms='148', includes_quantization='0', includes_communication='0',
            output='bf16', group_k='32', scale='ue8m0').items():
        evidence.require(config.get(key) == value, 'Pure config mismatch: ' + key)
    # The original cuBLASLt-only harness predates the optional CUTLASS-budget
    # field. Missing compute_ctas is not proof of unrestricted SMs: require
    # both the controller contract and EACH actual cuBLASLt plan's math_sms=0.
    evidence.require(config.get('compute_ctas') in (None,'0') and
                     job.get('gemm_sm_budget') in (None,0) and job.get('cublaslt_sm_target') in (None,0),
                     'Pure reference has a requested compute budget')
    for key, value in dict(measurement='single_gpu_pure_cublaslt',math_sms=0,
            precision='mxfp8',sm_budget_enforcement='full_device_no_restriction',measured_ranks=1).items():
        evidence.require(contract.get(key)==value,'Pure controller contract mismatch: '+key)
    evidence.require(contract.get('shapes') == job['gemm_matrix_payload']['shapes'],
                     'Pure contract geometry mismatch')
    shapes = {r['id']: r for r in job['gemm_matrix_payload']['shapes']}
    checks, samples, inputs, output, plans = {}, {}, {}, {}, {}
    for line in text.splitlines():
        if line.startswith('plan,pure_mxfp8,'):
            prefix, encoded = line.split(',config=',1)
            name, plan = fields(prefix)['id'], json.loads(encoded)
            evidence.require(name in shapes and name not in plans,'Duplicate/unknown pure plan')
            for key,value in dict(precision=32,math_sms=0,graph_tuning=1,beta=0,
                                  transpose_x=0,transpose_w=0).items():
                evidence.require(plan.get(key)==value,'Pure executed plan mismatch: '+key)
            evidence.require(plan.get('valid',0)>0,'No valid cuBLASLt algorithm')
            plans[name]=plan
        elif line.startswith('correctness,pure_mxfp8,'):
            f = fields(line)
            shape = shapes[f['id']]
            evidence.require(int(f['checked']) == shape['m']*shape['n'] and
                             int(f['mismatches']) == int(f['nonfinite']) == 0, 'Pure numerical check failed')
            checks.setdefault(f['id'], set()).add((int(f['generation']), f['phase']))
        elif line.startswith('input,pure_mxfp8,'):
            f = fields(line)
            shape = shapes[f['id']]
            count = shape['k'] * shape['n' if f['operand'] == '1' else 'm']
            evidence.require(int(f['count']) == count and float(f['rms']) > 0
                             and float(f['min']) < 0 < float(f['max']), 'Invalid random pure input')
            inputs.setdefault(f['id'], set()).add((int(f['generation']), int(f['operand'])))
        elif line.startswith('samples,pure_mxfp8,'):
            prefix, values = line.split(',ms=', 1)
            f, values = fields(prefix), json.loads(values)
            evidence.require(len(values) == 50 and all(math.isfinite(v) and v > 0 for v in values),
                             'Missing pure samples')
            if evidence.drift(values) <= .05 and f['id'] not in samples:
                samples[f['id']] = values
        elif line.startswith('RESULT '):
            r = json.loads(line[7:])
            name = r['id']
            evidence.require(name in shapes and name not in output, 'Unknown/duplicate pure result')
            if r['status'] == 'passed':
                evidence.require(checks.get(name) == {(0, 'pre'), (0, 'post'), (1, 'pre')} and
                                 inputs.get(name) == {(0,0),(0,1),(1,0),(1,1)} and name in samples,
                                 'Incomplete pure payload checks')
                evidence.require(all(r[k] == shapes[name][k] for k in ('m','n','k')), 'Pure geometry mismatch')
                evidence.require(name in plans and r.get('plan')==plans[name],'Pure selected plan mismatch')
                evidence.close(r['p50_ms'], evidence.percentile(samples[name], .5), 'Pure p50')
                evidence.close(r['pflops'], 2*r['m']*r['n']*r['k']/r['p50_ms']/1e12, 'Pure PFLOPS')
                r['raw_samples_ms'] = samples[name]
            else:
                evidence.require(r['status'] in ('memory_skip', 'unstable'), 'Failed pure reference')
            output[name] = r
    evidence.require(set(output) == set(shapes), 'Incomplete pure matrix')
    return dict(run_id=job['run_id'], artifact_sha256=digest, config=config,
                contract=contract,rows=list(output.values()))


def run_pure_reference(directory, rows, workspace, source_run=None):
    target = directory/'pure-current.json'
    if target.exists():
        previous = json.loads(target.read_text())
        if previous != audit_pure(RUNS/previous['run_id']):
            raise ValueError('Pure summary differs from original receipt')
    else:
        shapes = sorted({(r['m'],r['n'],r['k']) for r in rows
                         if not r['status'].startswith('unsupported')})
        l20d.write_json(directory/'gemm-matrix.json',dict(schema='sm103_gemm_matrix_v1',shapes=[
            dict(id=f'm{m}n{n}k{k}',m=m,n=n,k=k) for m,n,k in shapes]))
        command = [sys.executable,str(ROOT/'scripts/l20d.py'),'run','gemm-probe',
            '--node','09','--workspace',workspace,'--directions','oproj','--launches','graph',
            '--gemm-precision','mxfp8','--gemm-matrix',str(directory/'gemm-matrix.json'),
            '--gemm-candidates','32','--timeout','1800']
        if source_run:
            command += ['--source-run',source_run]
        print('RUN full-148-SM cuBLASLt '+str(len(shapes))+' MNK',flush=True)
        folder = execute_job(command)
        previous = audit_pure(folder)
        l20d.write_json(target,previous)
    acceptance = directory/'acceptance-current.json'
    if acceptance.exists():
        (directory/'acceptance-current.md').write_text(acceptance_markdown(
            rows,json.loads(acceptance.read_text()),previous))
    print('DONE pure cuBLASLt '+str(sum(r['status']=='passed' for r in previous['rows']))+
          '/'+str(len(previous['rows'])),flush=True)
    if acceptance.exists():
        completed = {r['id'] for r in json.loads(acceptance.read_text())['rows']}
        targets = {r['id'] for r in rows if not r['status'].startswith('unsupported')}
        if completed == targets:
            finalize(directory,rows)


def render(rows, pure=None):
    lines = ['# MXFP8 A2A + OProj baseline', '',
        'Per-rank PFLOPS; Graph 10+50, two random payloads and full numeric/route checks.',
        'Input: prequantized A + SFA; timed: A2A + BF16 weight quantization + GEMM to BF16.',
        'Fixed c16, M128/N256/K128, E32, AlongN/sw1, rows delivery; not tuned.', '',
        '| Model | S | CP | Fused PFLOPS | Pure cuBLASLt | Fused/Pure | Status |',
        '|---|---:|---:|---:|---:|---:|---|']
    references = {r['id']: r for r in (pure or {}).get('rows', [])}
    for row in rows:
        result = row.get('result')
        value = f"{result['pflops_per_rank']:.3f}" if result else '—'
        ref = references.get(f"m{row['m']}n{row['n']}k{row['k']}", {})
        p = f"{ref['pflops']:.3f}" if ref.get('status') == 'passed' else '—'
        ratio = f"{100*result['pflops_per_rank']/ref['pflops']:.1f}%" if result and p != '—' else '—'
        for name in row['models']:
            lines.append(f"| {name} | {row['global_seq']//1024}K | {row['world']} | {value} | {p} | {ratio} | {row['status']} |")
    return '\n'.join(lines) + '\n'


def save(directory, rows):
    directory.mkdir(parents=True, exist_ok=True)
    l20d.write_json(directory / 'baseline-current.json', dict(schema='sm103_mxfp8_oproj_baseline_v1', rows=rows))
    pure = directory / 'pure-current.json'
    (directory / 'baseline-current.md').write_text(render(rows, json.loads(pure.read_text()) if pure.exists() else None))


def run_comm_search(directory, rows, workspace, case_ids, budgets=None, source_run=None):
    pure = json.loads((directory / 'gemm-search-current.json').read_text())
    plan = manual_plan(rows, pure)
    l20d.write_json(directory / 'manual-plan-current.json', plan)
    tasks = [r for r in plan['rows'] if r['tuning_status'] == 'comm_budget_first'
             and (not case_ids or r['id'] in case_ids)]
    if budgets is not None:
        if not budgets or len(set(budgets)) != len(budgets) or any(c <= 0 or c >= 148 for c in budgets):
            raise ValueError('Distinct explicit CTA budgets in 1..147 required')
        for row in tasks:
            row['proposed_candidates'] = [dict(comm_ctas=c) for c in budgets]
    if case_ids and set(case_ids) != {r['id'] for r in tasks}:
        raise ValueError('Requested cases must have verified pure winners')
    # Check every selected native configuration before any remote submission.
    commands = [comm_budget_command(r, workspace) for r in tasks]
    target = directory / 'manual-current.json'
    report = json.loads(target.read_text()) if target.exists() else dict(
        schema='sm103_mxfp8_oproj_manual_v1', pure_run_id=pure['run_id'], rows=[])
    if report['pure_run_id'] != pure['run_id']:
        raise ValueError('Manual results belong to a different pure search')
    done = {r['id']:r for r in report['rows']}
    for row, command in zip(tasks, commands):
        previous = done.get(row['id'])
        if previous and previous['gemm'] != row['gemm']:
            raise ValueError('Cannot retune GEMM during the producer search')
        existing = previous['candidates'] if previous else []
        measured = {int(c['configuration']['comm_sm']) for c in existing}
        missing = [c for c in row['proposed_candidates'] if c['comm_ctas'] not in measured]
        if not missing:
            continue
        row = dict(row, proposed_candidates=missing)
        command = comm_budget_command(row, workspace)
        if source_run:
            command += ['--source-run', source_run]
        print('RUN comm-budget ' + row['id'] + ' c=' + ','.join(str(c['comm_ctas']) for c in missing), flush=True)
        folder = execute_job(command)
        candidates = [audit_run(folder, i+1) for i in range(len(row['proposed_candidates']))]
        candidates = existing + candidates
        if len({r['binary_sha256'] for r in candidates}) != 1:
            raise ValueError('Budget comparison must use one binary')
        winner = min(candidates, key=lambda r:r['p50_ms'])
        entry = dict(id=row['id'], gemm=row['gemm'],
            models=row['models'], winner=winner, candidates=candidates,
            selection='best in tested CTA budgets; not independently remeasured')
        if previous:
            report['rows'][report['rows'].index(previous)] = entry
        else:
            report['rows'].append(entry)
        l20d.write_json(target, report)
        values = ' '.join(f"c{r['configuration']['comm_sm']}={r['pflops_per_rank']:.3f}P"
                          for r in candidates)
        print(f"DONE {row['id']} {values}", flush=True)


def acceptance_command(row, manual, workspace, source_run=None):
    if manual['gemm'] != row['gemm']:
        raise ValueError('Acceptance may not change the fixed pure GEMM')
    budget = int(manual['winner']['configuration']['comm_sm'])
    command = comm_budget_command(dict(row,proposed_candidates=[dict(comm_ctas=budget)]),workspace)
    # Controller appends the unresolved zero candidate after this fixed control.
    # Even if Auto chooses the same budget, measure it independently: no aliasing.
    command += ['--auto-mxfp8-comm']
    if source_run:
        command += ['--source-run',source_run]
    command[command.index('--experiment')+1] = 'v21-oproj-auto-acceptance'
    return command


def acceptance_markdown(rows, report, pure=None):
    measured = {r['id']:r for r in report['rows']}
    references = {r['id']:r for r in (pure or {}).get('rows',[])}
    lines = ['# MXFP8 A2A + OProj — manual / Auto', '',
        'Per-rank PFLOPS; Graph 10+50. A is prequantized; W quantization, A2A and GEMM are timed.',
        'Manual = independently remeasured historical winning configuration, not a new search.',
        'Auto and manual use the same job/binary. cuBLASLt has no SM-budget restriction (148-SM device), no communication/quantization.', '',
        '| Model | S | CP | Manual | Auto | Pure cuBLASLt | Auto/manual | Auto/pure | comm manual→Auto | GEMM |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|---|']
    for row in rows:
        r = measured.get(row['id'],{})
        a, b = r.get('auto'), r.get('manual')
        ref = references.get(f"m{row['m']}n{row['n']}k{row['k']}",{})
        p = ref.get('pflops') if ref.get('status') == 'passed' and not row['status'].startswith('unsupported') else None
        fmt = lambda v: f'{v:.3f}' if v is not None else '—'
        ratio = lambda x,y: f'{100*x/y:.1f}%' if x is not None and y else '—'
        av, bv = (a['pflops_per_rank'] if a else None),(b['pflops_per_rank'] if b else None)
        budget = f"{b['configuration']['comm_sm']}→{a['configuration']['comm_sm']}" if a and b else '—'
        g = r.get('gemm')
        config = (f"128×256×128 E{g['epilogue_n']} / {g['raster']} sw{g['max_swizzle_size']}"
                  if g else ('unsupported' if row['status'].startswith('unsupported') else 'pending'))
        for name in row['models']:
            lines.append(f"| {name} | {row['global_seq']//1024}K | {row['world']} | {fmt(bv)} | {fmt(av)} | {fmt(p)} | {ratio(av,bv)} | {ratio(av,p)} | {budget} | {config} |")
    return '\n'.join(lines)+'\n'


def run_acceptance(directory, rows, workspace, case_ids, source_run=None):
    pure = json.loads((directory/'gemm-search-current.json').read_text())
    manual = json.loads((directory/'manual-current.json').read_text())
    model = json.loads((directory/'model-current.json').read_text())
    if manual['pure_run_id'] != pure['run_id']:
        raise ValueError('Manual and Auto require the same pure inputs')
    plan = manual_plan(rows,pure)
    tasks = [r for r in plan['rows'] if r['tuning_status']=='comm_budget_first'
             and (not case_ids or r['id'] in case_ids)]
    if case_ids and {r['id'] for r in tasks} != set(case_ids):
        raise ValueError('Unknown acceptance cases')
    controls = {r['id']:r for r in manual['rows']}
    commands = [acceptance_command(r,controls[r['id']],workspace,source_run) for r in tasks]
    target = directory/'acceptance-current.json'
    report = json.loads(target.read_text()) if target.exists() else dict(
        schema='sm103_mxfp8_oproj_acceptance_v1',pure_run_id=pure['run_id'],
        model_version=model['version'],rows=[])
    if (report['pure_run_id'],report['model_version']) != (pure['run_id'],model['version']):
        raise ValueError('Acceptance must not mix model/input versions')
    done = {r['id'] for r in report['rows']}
    for row, command in zip(tasks,commands):
        if row['id'] in done:
            continue
        print('RUN acceptance '+row['id'],flush=True)
        folder = execute_job(command)
        control, auto = audit_run(folder,1), audit_run(folder,2)
        if auto['communication_selection'].get('model_version') != model['version']:
            raise ValueError('GPU launch used a different compiled model')
        if control['communication_selection']['mode'] != 'explicit':
            raise ValueError('Manual confirmation must remain explicit')
        for key in ('binary_sha256','environment_fingerprint','run_id'):
            if control[key] != auto[key]:
                raise ValueError('Unpaired acceptance comparison')
        identities = {(r['auto']['binary_sha256'],r['auto']['environment_fingerprint']) for r in report['rows']}
        identities.add((auto['binary_sha256'],auto['environment_fingerprint']))
        if len(identities) != 1:
            raise ValueError('Acceptance table requires one binary/environment')
        report['rows'].append(dict(id=row['id'],models=row['models'],gemm=row['gemm'],
            manual=control,auto=auto,manual_selection_run=controls[row['id']]['winner']['run_id'],
            auto_over_manual=auto['pflops_per_rank']/control['pflops_per_rank'],
            held_out_sequence=row['global_seq']>131072))
        l20d.write_json(target,report)
        reference = directory/'pure-current.json'
        (directory/'acceptance-current.md').write_text(acceptance_markdown(rows,report,
            json.loads(reference.read_text()) if reference.exists() else None))
        print(f"DONE acceptance {len(report['rows'])}/75 {row['id']} manual={control['pflops_per_rank']:.3f}P "
              f"auto={auto['pflops_per_rank']:.3f}P ratio={report['rows'][-1]['auto_over_manual']:.2%} "
              f"c={control['configuration']['comm_sm']}→{auto['configuration']['comm_sm']}",flush=True)


def finalize(directory, rows):
    """Reaudit complete results, preserve a compact manual best, no GPU work."""
    report = json.loads((directory/'acceptance-current.json').read_text())
    selected = json.loads((directory/'manual-current.json').read_text())
    model = json.loads((directory/'model-current.json').read_text())
    pure = json.loads((directory/'pure-current.json').read_text())
    targets = {r['id']:r for r in rows if not r['status'].startswith('unsupported')}
    actual = {r['id']:r for r in report['rows']}
    if len(actual) != len(report['rows']) or set(actual) != set(targets):
        raise ValueError('Final report requires the full runnable matrix')
    if report['model_version'] != model['version'] or selected['pure_run_id'] != report['pure_run_id']:
        raise ValueError('Final report mixes model/pure-input versions')
    if pure != audit_pure(RUNS/pure['run_id']):
        raise ValueError('Pure reference differs from original evidence')
    pure_keys = {(r['m'],r['n'],r['k']) for r in pure['rows']}
    if pure_keys != {(r['m'],r['n'],r['k']) for r in targets.values()}:
        raise ValueError('Pure reference does not cover the full matrix')
    if any(r['status'] not in ('passed','memory_skip') for r in pure['rows']):
        raise ValueError('Unstable pure references need remeasurement')
    historical = {r['id']:r for r in selected['rows']}
    identities, confirmed = set(), []
    for key, row in actual.items():
        control, auto = row['manual'], row['auto']
        for value in (control,auto):
            if value != audit_run(RUNS/value['run_id'],value['candidate_id']):
                raise ValueError('Acceptance differs from original evidence: '+key)
            identities.add((value['source_id'],value['binary_sha256'],value['environment_fingerprint']))
        prior = historical[key]
        if row['gemm'] != prior['gemm'] or control['run_id'] != auto['run_id']:
            raise ValueError('GEMM input changed or comparison is unpaired')
        if control['configuration']['comm_sm'] != prior['winner']['configuration']['comm_sm']:
            raise ValueError('Manual configuration was reselected during acceptance')
        if auto['communication_selection'].get('model_version') != model['version']:
            raise ValueError('Auto did not use this compiled model')
        evidence.close(row['auto_over_manual'],auto['pflops_per_rank']/control['pflops_per_rank'],
                       'Auto/manual ratio')
        confirmed.append(dict(id=key,models=targets[key]['models'],gemm=row['gemm'],
            selection_run=prior['winner']['run_id'],confirmation=control))
    if len(identities) != 1:
        raise ValueError('Final table requires one fused source/binary/environment')
    # Check the current local native code, not only the historical green build.
    job = json.loads((RUNS/report['rows'][0]['auto']['run_id']/'job.json').read_text())
    native = {p:d for p,d in job['files'].items()
              if Path(p).suffix in ('.h','.hpp','.cuh','.cu','.cpp','.cmake') or Path(p).name=='CMakeLists.txt'}
    for path, digest in native.items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != digest:
            raise ValueError('Local native code differs from tested snapshot: '+path)
    stats = {}
    for name, subset in [('all',report['rows']),
                         ('anchor_sequence',[r for r in report['rows'] if not r['held_out_sequence']]),
                         ('held_out_sequences',[r for r in report['rows'] if r['held_out_sequence']])]:
        ratios = [r['auto_over_manual'] for r in subset]
        stats[name] = dict(count=len(ratios),geomean=math.exp(sum(map(math.log,ratios))/len(ratios)),
                           minimum=min(ratios))
    l20d.write_json(directory/'manual-best-current.json',dict(
        schema='sm103_mxfp8_oproj_manual_best_v1',rows=confirmed,
        scope='independent confirmation of bounded search winners, not a global optimum'))
    l20d.write_json(directory/'completion-current.json',dict(
        model_version=model['version'],coverage=dict(physical_passed=len(actual),
            logical_total=sum(len(r['models']) for r in rows),
            unsupported_physical=len(rows)-len(actual),pure_passed=sum(r['status']=='passed' for r in pure['rows'])),
        auto_over_manual=stats,identity=next(iter(identities)),native_files_verified=len(native),
        pure_run_id=pure['run_id'],limitations=[
            'Model calibrated at exact N/K/CP/GEMM layouts; only M0..4*M0 is extrapolated.',
            'No unseen N/K guarantee; unmeasured domains require an explicit manual CTA budget.',
            'No concurrent-interference or exact per-panel publication-latency prediction.',
            'No MXFP8 OProj role-profiler adapter or special KDA/MLA routing expansion.']))
    (directory/'acceptance-current.md').write_text(acceptance_markdown(rows,report,pure))
    print('FINAL '+json.dumps(stats),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workspace', default='/home/work/workspace_wct')
    p.add_argument('--run', action='store_true', help='Submit pending physical cases; otherwise create a plan only')
    p.add_argument('--comm-search', action='store_true', help='Fixed pure winners, five explicit CTA budgets')
    p.add_argument('--case-id', action='append', default=[], help='Restrict the comm-budget experiment')
    p.add_argument('--comm-budgets', help='Explicit comma-separated producer budgets; missing candidates only')
    p.add_argument('--source-run', help='Pin native source while developing the next stage locally')
    p.add_argument('--service-calibration', action='store_true', help='Independent C/A/A+W services at S128K only')
    p.add_argument('--export-calibration', action='store_true', help='Audit independent services and export a native fragment')
    p.add_argument('--acceptance', action='store_true', help='Paired historical manual configuration and actual Auto=0')
    p.add_argument('--pure-reference', action='store_true', help='Full-148-SM pure cuBLASLt, no communication/quantization')
    p.add_argument('--finalize', action='store_true', help='Reaudit complete results and write final result-only tables')
    p.add_argument('--overlap-audit', action='store_true', help='F/C/R/P for current layouts and controlled opposite-raster pairs')
    args = p.parse_args()
    target = args.output / 'baseline-current.json'
    rows = json.loads(target.read_text())['rows'] if target.exists() else matrix()
    if sum((args.run,args.comm_search,args.service_calibration,args.export_calibration,args.acceptance,args.pure_reference,args.finalize,args.overlap_audit))>1:
        p.error('Choose one stage')
    if args.overlap_audit:
        run_overlap_audit(args.output,rows,args.workspace,args.case_id,args.source_run)
        return
    if args.finalize:
        finalize(args.output,rows)
        return
    if args.pure_reference:
        run_pure_reference(args.output,rows,args.workspace,args.source_run)
        return
    if args.export_calibration:
        export_calibration(args.output,rows)
        return
    if args.acceptance:
        run_acceptance(args.output,rows,args.workspace,args.case_id,args.source_run)
        return
    if args.service_calibration:
        if args.run or args.comm_search:
            p.error('Service calibration is a separate stage')
        run_service_calibration(args.output,rows,args.workspace,args.source_run)
        return
    if args.comm_search:
        if args.run:
            p.error('--run and --comm-search are separate stages')
        budgets = list(map(int,args.comm_budgets.split(','))) if args.comm_budgets else None
        run_comm_search(args.output, rows, args.workspace, args.case_id, budgets, args.source_run)
        return
    # The pure reference needs only MNK, not a duplicate process for each CP/S alias.
    shapes = {(r['m'], r['n'], r['k']) for r in rows if not r['status'].startswith('unsupported')}
    args.output.mkdir(parents=True, exist_ok=True)
    l20d.write_json(args.output / 'gemm-matrix.json', dict(schema='sm103_gemm_matrix_v1', shapes=[
        dict(id=f'm{m}n{n}k{k}', m=m, n=n, k=k) for m, n, k in sorted(shapes)]))
    save(args.output, rows)
    if not args.run:
        print(f"PLAN physical={len(rows)} runnable={sum(r['status']=='pending' for r in rows)} pure={len(shapes)}", flush=True)
        return
    for index, row in enumerate(rows, 1):
        if row['status'] != 'pending':
            continue
        argv = [sys.executable, str(ROOT / 'scripts/l20d.py'), 'run', 'fused-smoke',
            '--node', '09', '--workspace', args.workspace, '--mpi', '--mxfp8',
            '--fused-direction', 'oproj', '--directions', 'oproj', '--fused-launch', 'graph',
            '--input-generator', 'gpu_philox', '--oproj-policy-list', 'm128n256',
            '--qkv-policy-list', 'm128n256', '--mxfp8-weight-preparation', 'comm',
            '--mxfp8-epilogue-n', '32', '--oproj-raster', 'along_n', '--max-swizzle-size', '1',
            '--oproj-comm-layout', 'rows', '--comm-sm', '16', '--causal',
            '--world', str(row['world']), '--global-seq', str(row['global_seq']),
            '--hidden', str(row['hidden']), '--q-heads', str(row['q_heads']),
            '--kv-heads', '8', '--head-dim', str(row['head_dim']),
            '--experiment', 'v21-oproj-full', '--timeout', '180']
        print(f"RUN {index}/{len(rows)} {row['id']}", flush=True)
        completed = subprocess.run(argv, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        matches = re.findall(r'"run_id": "([0-9]{8}-[0-9]{6}-[a-f0-9]+)"', completed.stdout)
        if not matches:
            raise RuntimeError(completed.stdout[-1200:])
        row['run_id'] = matches[0]
        folder = Path('/Users/admin/workspace/fuse_midfile/l20d') / row['run_id']
        if completed.returncode:
            row['status'] = 'failed'
            row['error'] = completed.stdout[-1200:]
            save(args.output, rows)
            # A failure is not silently treated as OOM. Diagnose before proceeding.
            raise RuntimeError(f"{row['run_id']}: {row['error']}")
        row['result'] = audit_run(folder)
        row['status'] = 'passed'
        save(args.output, rows)
        print(f"DONE {index}/{len(rows)} {row['id']} {row['result']['pflops_per_rank']:.3f} PFLOPS "
              f"p50={row['result']['p50_ms']:.4f}ms", flush=True)


if __name__ == '__main__':
    main()
