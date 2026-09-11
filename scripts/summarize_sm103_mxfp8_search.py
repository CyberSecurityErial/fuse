"""Audit fetched pure-CUTLASS grid/neighbor results; never run GPU work."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import tarfile

BASELINE = 'm128n256k128e64s0sw1M'


def fields(line):
    return dict(re.findall(r'(\w+)=([^,\n]+)', line))


def audit(folder, allow_timeout_partial=False, allow_boundary_partial=False):
    folder = Path(folder).resolve()
    fetched = json.loads((folder / 'fetched.json').read_text())
    directories = sorted(folder.glob('artifacts-attempt*/control'))
    if len(directories) != 1:
        raise ValueError('Expected exactly one fetched attempt')
    control = directories[0]
    status = json.loads((control / 'status.json').read_text())
    timed_out = (allow_timeout_partial and fetched.get('state') == 'failed' and
                 'TimeoutError: Task timeout; terminating only its own process group' in fetched.get('error', ''))
    boundary_stop = (allow_boundary_partial and fetched.get('state') == 'failed' and
                     fetched.get('exit_code') == 1 and not fetched.get('error'))
    if not (timed_out or boundary_stop) and (fetched.get('state') != 'succeeded' or fetched.get('exit_code') != 0 or status.get('work_exit_code') != 0):
        raise ValueError('Search did not complete successfully')
    artifact_hash = hashlib.sha256((folder / 'artifacts.tar.gz').read_bytes()).hexdigest()
    if fetched.get('artifact_sha256') != artifact_hash:
        raise ValueError('Artifact digest mismatch')
    with tarfile.open(folder / 'artifacts.tar.gz', 'r:gz') as archive:
        for name in ('status.json', 'job.json', 'mxfp8-search-build.json', 'attempt1.log'):
            member = archive.extractfile('control/' + name)
            if member is None or member.read() != (control / name).read_bytes():
                raise ValueError('Extracted evidence differs from the verified artifact')
    job = json.loads((control / 'job.json').read_text())
    if not job.get('mxfp8_gemm_search') or job.get('gemm_sm_budget') != 132:
        raise ValueError('This report requires explicit 132-CTA pure MXFP8 search')
    receipt = json.loads((control / 'mxfp8-search-build.json').read_text())
    matrix = {row['id']: row for row in job['gemm_matrix_payload']['shapes']}
    current = None
    checks, samples, candidates, skips, passes = {}, {}, {}, {}, {}
    log = (control / 'attempt1.log').read_text()
    lines = log.splitlines()
    if (timed_out or allow_boundary_partial) and not log.endswith('\n'):
        lines = lines[:-1]  # Never parse a partially flushed final record.
    for line in lines:
        if line.startswith(('RUN cutlass_mxfp8,', 'verify_candidate,cutlass_mxfp8,')):
            f = fields(line)
            current = (f['id'], f.get('candidate', f.get('config')))
            if line.startswith('RUN '):
                passes[current] = int(f.get('pass', 0))
        elif line.startswith('correctness,pure_mxfp8,'):
            f = fields(line)
            if current is None or current[0] != f['id']:
                raise ValueError('Unattributed numerical check')
            shape = matrix[f['id']]
            if int(f['checked']) != shape['m'] * shape['n'] or int(f['mismatches']) or int(f['nonfinite']):
                raise ValueError('Incomplete or failed numerical check')
            checks.setdefault(current, set()).add((int(f['generation']), f['phase']))
        elif line.startswith('samples,cutlass_mxfp8,'):
            prefix, raw = line.split(',ms=', 1)
            f, values = fields(prefix), json.loads(raw)
            key = (f['id'], f['config'])
            if len(values) != 50 or not all(math.isfinite(v) and v > 0 for v in values):
                raise ValueError('Invalid Graph samples')
            drift = abs(statistics.median(values[25:]) / statistics.median(values[:25]) - 1)
            # Keep first stable round, not minimum time across noise retries.
            if drift <= .05 and key not in samples:
                samples[key] = (statistics.median(values), sorted(values)[47], drift, int(f['round']))
        elif line.startswith('RESULT '):
            row = json.loads(line[7:])
            if row['status'] == 'memory_skip':
                skips[row['id']] = row
                continue
            key = (row['id'], row['config'])
            if key in candidates or row.get('backend') != 'cutlass_mxfp8':
                raise ValueError('Duplicate or wrong backend result')
            if checks.get(key) != {(0, 'pre'), (0, 'post'), (1, 'post')} or key not in samples:
                raise ValueError('Missing dual-payload validation or stable samples')
            ms, p95, drift, accepted_round = samples[key]
            shape = matrix[row['id']]
            if any(row[k] != shape[k] for k in ('m', 'n', 'k')) or row['compute_ctas'] != 132:
                raise ValueError('Result geometry/budget mismatch')
            for reported, measured in ((row['p50_ms'], ms), (row['p95_ms'], p95), (row['drift'], drift)):
                if not math.isclose(reported, measured, rel_tol=2e-5, abs_tol=2e-7):
                    raise ValueError('Reported timing does not match samples')
            row.update(p50_ms=ms, p95_ms=p95, drift=drift, accepted_round=accepted_round,
                       pflops=2 * shape['m'] * shape['n'] * shape['k'] / ms * 1e-12,
                       search_pass=passes[key])
            candidates[key] = row
    rows, pending = [], []
    for name, shape in matrix.items():
        group = [v for (sid, _), v in candidates.items() if sid == name]
        if not group:
            if name not in skips:
                if timed_out or allow_boundary_partial:
                    pending.append(shape)
                    continue
                raise ValueError(f'Missing matrix point: {name}')
            rows.append(skips[name])
            continue
        baseline = candidates.get((name, BASELINE))
        winner = min(group, key=lambda row: row['p50_ms'])
        grid_best = min((r for r in group if r['search_pass'] == 0), key=lambda r: r['p50_ms'])
        rows.append(dict(**shape, status='passed', candidates=len(group), baseline=baseline,
                         winner=winner, grid_best=grid_best,
                         gain_vs_grid=grid_best['p50_ms'] / winner['p50_ms'] - 1,
                         gain=(baseline['p50_ms'] / winner['p50_ms'] - 1) if baseline else None))
    gains = [r['gain'] for r in rows if r.get('gain') is not None]
    return dict(schema='sm103_mxfp8_cutlass_search_v1', run_id=job['run_id'],
                source_id=job.get('source_id'),
                compute_ctas=132, includes_quantization=False, includes_communication=False,
                collector='single_gpu_CUDAEvent_around_host_GraphLaunch', warmup=10, samples=50,
                artifact_sha256=artifact_hash, raw_evidence=str(control), build=receipt,
                geometric_mean_gain=math.exp(sum(math.log1p(g) for g in gains)/len(gains))-1 if gains else None,
                rows=rows, pending=pending, partial=bool(timed_out or allow_boundary_partial))


def merge(first, remaining):
    """Resume only unfinished matrices, never cherry-pick across two runs."""
    for key in ('build', 'compute_ctas', 'collector', 'warmup', 'samples'):
        if first[key] != remaining[key]:
            raise ValueError('Cannot merge different builds or measurement contracts')
    wanted = {row['id'] for row in first['pending']}
    received = {row['id'] for row in remaining['rows']}
    if wanted != received or remaining['pending'] or remaining['partial']:
        raise ValueError('Continuation must account for exactly the unfinished matrices')
    if received & {row['id'] for row in first['rows']}:
        raise ValueError('Do not compare or select across repeated matrix runs')
    result = dict(first)
    result['rows'] = [dict(row, source_run_id=part['run_id'])
                      for part in (first, remaining) for row in part['rows']]
    result['runs'] = [{key: part.get(key) for key in
                       ('run_id', 'source_id', 'artifact_sha256', 'raw_evidence', 'partial')}
                      for part in (first, remaining)]
    for key in ('run_id', 'source_id', 'artifact_sha256', 'raw_evidence'):
        result.pop(key, None)
    result.update(pending=[], partial=False)
    gains = [row['gain'] for row in result['rows'] if row.get('gain') is not None]
    result['geometric_mean_gain'] = math.exp(sum(math.log1p(g) for g in gains)/len(gains))-1 if gains else None
    return result


def render(report):
    lines = ['# MXFP8 QKV：独立 GEMM 网格＋邻域搜索', '',
        '单 GPU、132 个计算 CTA，原配置与优胜配置同轮 Graph 10+50；双随机 payload 全量数值校验。',
        '不含量化、通信与 overlap；候选范围内最优，不代表融合收益或全局最优。', '',
        '| 矩阵 ID | M×N×K | 原配置 PFLOPS | 最优 PFLOPS | 提升 | 最优参数 | 合格候选 |',
        '|---|---|---:|---:|---:|---|---:|']
    for row in report['rows']:
        if row['status'] != 'passed':
            lines.append(f"| {row['id']} | — | — | — | 显存跳过 | — | 0 |")
            continue
        b, w = row['baseline'], row['winner']
        old = f"{b['pflops']:.3f}" if b else '—'
        gain = f"{row['gain']:+.2%}" if b else '原配置不稳定'
        lines.append(f"| {row['id']} | {row['m']}×{row['n']}×{row['k']} | {old} | {w['pflops']:.3f} | {gain} | {w['config']} | {row['candidates']} |")
    lines += ['', '参数：m/n/k 为 tile，e 为 epilogue N，s0 为 CUTLASS 自动 stage 数，sw 为 swizzle，M/N 为 raster。',
              f"可配对点几何平均提升：{report['geometric_mean_gain']:+.2%}。", '']
    for run in report.get('runs', [report]):
        lines.append(f"运行：`{run['run_id']}`；原始证据：`{run['raw_evidence']}`。")
    lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_folder', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--boundary-partial', action='store_true',
                        help='First run was intentionally stopped at a matrix boundary; accept only fully verified matrices')
    parser.add_argument('--update', action='store_true', help='Emit an explicit patch replacing the existing canonical reports')
    args = parser.parse_args()
    report = audit(args.run_folder[0], allow_boundary_partial=args.boundary_partial)
    for folder in args.run_folder[1:]:
        report = merge(report, audit(folder))
    # Emit a patch so the caller can apply the two small canonical reports;
    # raw candidate samples remain only in the fetched, hashed artifact.
    print('*** Begin Patch')
    for name, contents in [('summary.json', json.dumps(report, indent=2)), ('README.md', render(report))]:
        path = args.output / name
        if path.exists():
            if not args.update:
                raise ValueError('Output already exists; pass --update to replace canonical reports')
            print(f'*** Update File: {path}\n@@')
            for line in path.read_text().splitlines():
                print('-' + line)
        else:
            print(f'*** Add File: {path}')
        for line in contents.splitlines():
            print('+' + line)
    print('*** End Patch')


if __name__ == '__main__':
    main()
