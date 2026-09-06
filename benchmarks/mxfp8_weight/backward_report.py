"""Strict source/sample/selection/coverage audit and full FP32 backward table."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import geometric_mean

from backward_matrix import ROOT, expected_keys, full_matrix
from run_suite import PROFILES, write


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values, q):
    xs = sorted(values)
    position = (len(xs)-1)*q
    lo, hi = math.floor(position), math.ceil(position)
    return xs[lo]*(hi-position) + xs[hi]*(position-lo) if lo != hi else xs[lo]


def timing_errors(t, count, cp):
    errors = []
    values, ranks = t['samples_us'], t['rank_samples_us']
    if len(values) != count or len(ranks) != cp or any(len(r) != count for r in ranks):
        return ['wrong number of samples/ranks']
    if not all(math.isfinite(v) and v > 0 for r in ranks for v in r):
        errors.append('nonfinite/nonpositive rank sample')
    if values != [max(x) for x in zip(*ranks)]:
        errors.append('not samplewise rank-max')
    for key, want in (('p50_us', percentile(values, .5)), ('p95_us', percentile(values, .95)),
                      ('mean_us', sum(values)/count)):
        if not math.isclose(t[key], want, rel_tol=1e-10):
            errors.append(f'wrong {key}')
    return errors


def correctness_errors(node):
    errors = []
    if isinstance(node, dict):
        if 'relative_rmse' in node:
            if (not math.isfinite(node['max_abs']) or not math.isfinite(node['relative_rmse']) or
                    node['relative_rmse'] > .005 or node['max_abs'] < 0):
                errors.append('invalid correctness metric')
        else:
            for child in node.values():
                errors += correctness_errors(child)
    elif isinstance(node, list):
        for child in node:
            errors += correctness_errors(child)
    return errors


def audit(directory):
    matrix = {c['id']:c for c in full_matrix()}
    expected = expected_keys()
    errors, seen, rows = [], set(), []
    raw = {}
    # Glob selects known result filenames, not logs/manifests or synthetic data.
    files = sorted(directory.glob('*_formal_*_cp[48].json')) + sorted(directory.glob('*_sweep_*_cp[48].json'))
    for path in files:
        report = json.loads(path.read_text())
        raw[path.name] = report
        args = report['args']
        filename_spec = [(backend, phase, profile, cp)
                         for backend in ('cublaslt_nccl','teub') for phase in ('formal','sweep')
                         for profile in PROFILES for cp in (4,8)
                         if path.name == f'{backend}_{phase}_{profile}_cp{cp}.json']
        if len(filename_spec) != 1:
            errors.append(f'{path.name}: unrecognized raw filename'); continue
        backend, phase, profile, file_cp = filename_spec[0]
        if args['phase'] != phase or args.get('backends') != backend:
            errors.append(f'{path.name}: filename disagrees with execution arguments')
        if args.get('candidates') != 32:
            errors.append(f'{path.name}: wrong GEMM search budget')
        if report.get('schema') != 'mxfp8-backward-v1' or report.get('semantic') != 'offline_mxfp8_original_weight_axis_runtime_dq_bf16_gemm':
            errors.append(f'{path.name}: wrong schema/weight semantic')
        expected_env = dict(NCCL_IB_DISABLE='1', NCCL_GRAPH_REGISTER='1', NCCL_LOCAL_REGISTER='1', **PROFILES[profile])
        got_env = {k:v for k,v in report['environment'].items() if k.startswith('NCCL_')}
        if got_env != expected_env:
            errors.append(f'{path.name}: actual NCCL configuration differs from declared profile')
        if not report['complete']:
            errors.append(f'{path.name}: incomplete job')
        required_sources = {'benchmarks/mxfp8_weight/backward_bench.py',
                            'benchmarks/mxfp8_weight/backward_runtime.py',
                            'benchmarks/mxfp8_weight/backward_gemm.py',
                            'benchmarks/mxfp8_weight/backward_gemm.cu',
                            'benchmarks/mxfp8_weight/backward_matrix.py',
                            'benchmarks/backward/backward_shape_bench.py',
                            'build-mxfp8-bench/libmxfp8_backward.so'}
        if not required_sources <= set(report['sources']):
            errors.append(f'{path.name}: missing provenance')
        for source, sha in report['sources'].items():
            target = ROOT / source
            if not target.is_file() or digest(target) != sha:
                errors.append(f'{path.name}: changed/missing source {source}')
        if args['grad_dtypes'] != 'fp32' or args['validation']:
            errors.append(f'{path.name}: not formal FP32 matrix')
        if (args['sweep_warmup'], args['sweep_iters']) != (3, 12):
            errors.append(f'{path.name}: wrong search protocol')
        if args['phase'] == 'formal' and (args['warmup'], args['iterations']) != (10, 50):
            errors.append(f'{path.name}: wrong formal protocol')
        for item in report['cases']:
            c = item['case']; cid = c['id']
            if matrix.get(cid) != c:
                errors.append(f'{cid}: not exact backward registry')
                continue
            if c['cp'] != file_cp:
                errors.append(f'{cid}: wrong process-group CP')
            if report['environment'].get('CUDA_VISIBLE_DEVICES') != c['visible_devices']:
                errors.append(f'{cid}: wrong physical GPU set')
            if len(report['devices']) != c['cp'] or any(d['cc'] != [9,0] for d in report['devices']):
                errors.append(f'{cid}: wrong device metadata')
            for sample in item['search']:
                errors += [f'{cid}/search: {e}' for e in timing_errors(sample['data'], 12, c['cp'])]
                errors += correctness_errors(sample['correctness'])
            if backend == 'cublaslt_nccl' and phase == 'sweep':
                if item['records'] or len(item['search']) != 2 or {(s['launch'],s['sms']) for s in item['search']} != {('eager',0),('graph',0)}:
                    errors.append(f'{cid}: wrong NCCL short-sweep scope')
            if backend == 'cublaslt_nccl' and phase == 'formal':
                launches = report.get('launch_policy', {}).get(cid, [])
                actual_keys = [(r['launch'],r['weight_mode']) for r in item['records']]
                want_keys = {(l,m) for l in launches for m in ('immediate','deferred')}
                if not launches or len(actual_keys) != len(want_keys) or set(actual_keys) != want_keys or item['search']:
                    errors.append(f'{cid}: formal scope differs from selected launch policy')
            for row in item['records']:
                key = (cid, row['backend'], row['launch'], row['weight_mode'], row['grad_dtype'])
                if key not in expected or key in seen:
                    errors.append(f'duplicate/unexpected: {key}')
                seen.add(key)
                if row['backend'] != backend or phase != 'formal':
                    errors.append(f'formal row has wrong source phase/backend: {key}')
                if row['beta'] != int(row['weight_mode'] == 'deferred'):
                    errors.append(f'wrong beta: {key}')
                for stage in ('data','weight','total'):
                    errors += [f'{key}/{stage}: {e}' for e in timing_errors(row[stage], 50, c['cp'])]
                gates = row.get('correctness', {})
                if set(gates) != {'data','weight','full','nonzero_beta1_twice'} or len(gates['nonzero_beta1_twice']) != 2:
                    errors.append(f'missing correctness gates: {key}')
                errors += correctness_errors(gates)
                fp32_gates = [gates['weight'], gates['full']['wgrad'], *gates['nonzero_beta1_twice']]
                if any(g['relative_rmse'] > .0001 for g in fp32_gates):
                    errors.append(f'FP32 wgrad accuracy exceeded 1e-4: {key}')
                if c['operator'] == 'qkv' and any(gates[s]['route']['max_abs'] != 0 for s in ('data','full')):
                    errors.append(f'inexact QKV route: {key}')
                if row['b_plan']['mnk'] != c['b_mnk'] or row['w_plan']['mnk'] != c['w_mnk']:
                    errors.append(f'wrong GEMM shapes: {key}')
                if row['b_plan']['output_dtype'] != 'torch.bfloat16' or row['w_plan']['output_dtype'] != 'torch.float32':
                    errors.append(f'wrong dtype: {key}')
                if row['b_plan']['ta'] or row['b_plan']['tb'] or not row['w_plan']['ta'] or row['w_plan']['tb']:
                    errors.append(f'wrong transpose: {key}')
                if row['w_plan']['tune_beta'] != row['beta']:
                    errors.append(f'tuned wrong beta: {key}')
                if row['isolated_sum_samples_us'] != [a+b for a,b in zip(row['data']['samples_us'], row['weight']['samples_us'])]:
                    errors.append(f'wrong isolated sum: {key}')
                if row['backend'] == 'teub':
                    eligible = [r for r in item['search'] if r['launch'] == row['launch']]
                    if len(eligible) != 3 or {r['sms'] for r in eligible} != {4,8,16}:
                        errors.append(f'incomplete TEUB candidates: {key}')
                    elif min(eligible, key=lambda r:r['data']['p50_us'])['sms'] != row['sms']:
                        errors.append(f'not short-sweep TEUB winner: {key}')
                rows.append(dict(id=cid, operator=c['operator'], model=c['model'], global_seq=c['global_seq'],
                    cp=c['cp'], b_mnk='x'.join(map(str,c['b_mnk'])), w_mnk='x'.join(map(str,c['w_mnk'])),
                    backend=row['backend'], launch=row['launch'], weight_mode=row['weight_mode'],
                    grad_dtype=row['grad_dtype'], b_us=row['data']['p50_us'], w_us=row['weight']['p50_us'],
                    total_us=row['total']['p50_us'], total_p95_us=row['total']['p95_us'],
                    isolated_sum_us=percentile(row['isolated_sum_samples_us'], .5),
                    total_tflops_per_gpu=4*math.prod(c['b_mnk'])/row['total']['p50_us']/1e6,
                    sms=row['sms'], source=path.name))
    # Independently reconstruct external NCCL selection; never trust policy alone.
    for cp in (4,8):
        policy_path = directory / f'nccl_policy_cp{cp}.json'
        if not policy_path.exists():
            continue
        policy_rows = json.loads(policy_path.read_text())
        policy = {(r['id'],r['launch']):r['profile'] for r in policy_rows}
        wanted = {(c['id'],l) for c in matrix.values() if c['cp']==cp for l in ('eager','graph')}
        if set(policy) != wanted or len(policy_rows) != len(wanted):
            errors.append(f'CP{cp}: incomplete/duplicate policy')
        candidates = {}
        for profile in PROFILES:
            report = raw.get(f'cublaslt_nccl_sweep_{profile}_cp{cp}.json')
            if report is None:
                errors.append(f'CP{cp}: missing {profile} search'); continue
            for k,v in PROFILES[profile].items():
                if report['environment'].get(k) != v:
                    errors.append(f'CP{cp}/{profile}: wrong NCCL env')
            for item in report['cases']:
                for sample in item['search']:
                    candidates.setdefault((item['case']['id'], sample['launch']), []).append((sample['data']['p50_us'], profile))
        for key, profile in policy.items():
            values = candidates.get(key, [])
            if len(values) != len(PROFILES) or min(values)[1] != profile:
                errors.append(f'not independent NCCL winner: {key}')
            for row in rows:
                if (row['id'],row['launch']) == key and row['backend'] == 'cublaslt_nccl':
                    if row['source'] != f'cublaslt_nccl_formal_{profile}_cp{cp}.json':
                        errors.append(f'formal row not selected profile: {key}')
    for row in rows:
        if row['backend'] == 'cublaslt_nccl' and not (directory / f"nccl_policy_cp{row['cp']}.json").exists():
            errors.append(f"missing NCCL policy CP{row['cp']}")
    missing = sorted(expected-seen)
    result = dict(expected_settings=192, expected_rows=1536, collected_rows=len(seen & expected),
                  complete=not errors and not missing, errors=errors, missing=missing,
                  raw_files={p.name:digest(p) for p in files})
    return result, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--require-full', action='store_true')
    args = parser.parse_args()
    result, rows = audit(args.directory)
    write(args.directory / 'coverage.json', result)
    if rows:
        with (args.directory / 'coverage.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    print(json.dumps({k:v for k,v in result.items() if k not in ('missing','raw_files')}, indent=2))
    if result['errors'] or (args.require_full and not result['complete']):
        raise SystemExit(1)
    lines = ['# MXFP8 权重反向：FP32 main_grad 全量表', '',
        f"覆盖 {result['collected_rows']}/1536 条；完整审计：{result['complete']}。", '',
        '权重离线 MXFP8；每次 B/full 内反量化 BF16；BF16 GEMM 输入与通信，FP32 dW/main_grad。',
        '与旧 SM90 对齐 shape/CP/seqlen/布局/计时边界，但旧 dW 是 BF16，不能据此声称量化收益。',
        'TEUB = TE Userbuffers P2P + cuBLASLt；不是原生 MXFP8 GEMM，也不是新的 Fuse 算子。',
        '正式 10 warmup + 50 样本，逐样本 rank-max；单位 µs。离线量化、调参和校验不计时。',
        'B/W 分别实测；全程为同流 B→W 独立实测，不是两段 p50 相加。分离两段的样本和见 CSV。',
        'deferred 全程列仅表示 beta=1 顺序基线，不代表实现了 ZeroBubble 调度收益。',
        '设备沿用用户指定 H200 卡组；保留 JSON 中原始运行时名称和 CC9.0/132SM 属性。',
        'Eager 可能包含主机提交间隙；Graph 为单操作 replay。算法元数据为 rank0，各 rank 各自调优。', '']
    index = {(r['id'],r['launch'],r['weight_mode'],r['backend']):r for r in rows}
    statistics = {}
    for mode in ('immediate','deferred'):
        for launch in ('eager','graph'):
            lines += [f'## {mode} / {launch}', '',
                '| 算子/模型 | S | CP | B MNK | W MNK | NCCL B/W/全程 | TEUB B/W/全程 | 全程加速 |',
                '|---|---:|---:|---|---|---:|---:|---:|']
            speedups = []
            for c in full_matrix():
                a = index.get((c['id'],launch,mode,'cublaslt_nccl'))
                b = index.get((c['id'],launch,mode,'teub'))
                if a is None or b is None:
                    continue
                speedup = a['total_us']/b['total_us']; speedups.append(speedup)
                fmt = lambda r:'/'.join(f'{r[k]:.2f}' for k in ('b_us','w_us','total_us'))
                lines.append(f"| {c['operator']}/{c['model']} | {c['global_seq']} | {c['cp']} | {a['b_mnk']} | {a['w_mnk']} | {fmt(a)} | {fmt(b)} | {speedup:.3f}× |")
            lines.append('')
            if speedups:
                statistics[f'{mode}/{launch}'] = dict(pairs=len(speedups), teub_faster=sum(x>1 for x in speedups),
                                                       geomean=geometric_mean(speedups))
    (args.directory / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(args.directory / 'statistics.json', statistics)


if __name__ == '__main__':
    main()
