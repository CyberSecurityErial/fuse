#!/usr/bin/env python3
"""Merge fetched NN/TN probes into one complete backward coverage table.

Only accepted pure GEMM measurements are populated. A pure GEMM result never
fills the distributed/fused backward column. Raw samples stay in the original
artifact; this table retains the winner config and a link to that evidence.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import tarfile


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_run(directory):
    directory = Path(directory)
    receipt = json.loads((directory / 'fetched.json').read_text())
    require(receipt['stage'] == 'gemm-probe' and receipt['state'] in ('succeeded', 'failed'),
            'Expected a terminal fetched GEMM probe')
    archive = directory / 'artifacts.tar.gz'
    require(hashlib.sha256(archive.read_bytes()).hexdigest() == receipt['artifact_sha256'],
            'Artifact checksum mismatch')
    with tarfile.open(archive) as tar:
        def read(name):
            members = [m for m in tar.getmembers() if m.name.endswith('/control/' + name)
                       or m.name == 'control/' + name]
            require(len(members) == 1 and members[0].isfile(), 'Missing/duplicate ' + name)
            return json.load(tar.extractfile(members[0]))
        result, contract = read('gemm-probe.json'), read('gemm-probe-contract.json')
    require(result['measurement'] == 'single_gpu_pure_cublaslt' and
            result['measured_ranks'] == 1 and not result['distributed_boundary_measured'],
            'Wrong benchmark boundary')
    require(contract['library_sha256'] == result['library_sha256'] and contract['math_sms'] == 0,
            'Wrong library or restricted-SM measurement')
    layout = result['operand_layout']
    require(layout in ('nn', 'tn') and contract['operand_layout'] == layout and
            not result['transpose_materialized'], 'Wrong operand layout')
    output = []
    for geometry in result['geometries']:
        if geometry['state'] != 'succeeded':
            continue
        require(len(geometry['results']) == 1, 'Expected only one BF16 Graph result')
        row = geometry['results'][0]
        samples, tune = row['samples_ms'], row['tuning']
        require(row['precision'] == 'bf16' and row['launch'] == 'graph' and
                row['warmup'] >= 10 and len(samples) >= 50 and row['tune_warmup'] >= 10 and
                row['tune_iterations'] >= 50, 'Wrong sampling contract')
        require(all(math.isfinite(x) and x > 0 for x in samples), 'Invalid samples')
        require(tune['transpose_x'] == int(layout == 'tn') and tune['transpose_w'] == 1 and
                tune['beta'] == 0 and tune['graph_tuning'] and tune['math_sms'] == 0,
                'Native plan disagrees with backward contract')
        evidence = row['measurement']
        drift = abs(statistics.median(samples[:len(samples)//2]) -
                    statistics.median(samples[len(samples)//2:])) / statistics.median(samples)
        require(evidence['warmup_converged'] and evidence['additional_warmup_cuda_ms'] >= 100 and
                drift <= .05, 'Unstable formal measurement')
        m, n, k = (geometry['shape'][key] for key in ('m', 'n', 'k'))
        correctness = row['correctness']
        require(math.isfinite(correctness['relative_rms']) and correctness['relative_rms'] <= .02 and
                math.isfinite(correctness['max_abs']) and correctness['checked_values'] == min(m,64)*min(n,64),
                'Missing/invalid numerical check')
        p50 = statistics.median(samples)
        require(math.isclose(p50, row['p50_ms'], rel_tol=1e-9), 'p50 disagrees with samples')
        winner = min(tune['candidates'], key=lambda c: c['tune_ms'])
        require(winner['algorithm'] == tune['algorithm'], 'Winner disagrees with tuning evidence')
        for alias in geometry['aliases']:
            output.append(dict(id=alias, m=m, n=n, k=k, operand_layout=layout,
                p50_ms=p50, p95_ms=row['p95_ms'], pflops=2*m*n*k/p50/1e12,
                algorithm=winner['algorithm'], tile=winner['tile'], split_k=winner['split_k'],
                candidate_index=winner['index'], workspace_bytes=winner['workspace_bytes'],
                half_drift=drift, sms=result['sms'], device=result['device'], node=receipt['node'],
                run_id=receipt['run_id'], source_id=receipt['source_id'],
                library_sha256=result['library_sha256'], environment=receipt['environment_fingerprint'],
                evidence=str(directory / f'artifacts-attempt{receipt["attempt"]}' / 'control/gemm-probe.json')))
    return output


def summarize(manifest, directories, output):
    manifest = json.loads(Path(manifest).read_text())
    require(manifest['schema'] == 'sm103_backward_catalog_v1', 'Wrong catalog')
    expected = {row['id']: row for row in manifest['cases']}
    measured = {}
    for directory in directories:
        for row in read_run(directory):
            require(row['id'] in expected and row['id'] not in measured, 'Unknown/duplicate case')
            case = expected[row['id']]
            require(all(row[key] == case[key] for key in ('m','n','k','operand_layout')), 'Wrong geometry')
            measured[row['id']] = row
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    csv_file = output / 'pure-gemm.csv'
    fields = ('model','projection','direction','phase','seq','cp','m','n','k','operand_layout',
              'pure_gemm_state','p50_ms','p95_ms','pflops','algorithm','tile','split_k',
              'candidate_index','workspace_bytes','half_drift','sms','node','fused_state','route_issue','run_id','evidence')
    rows = [case | measured.get(case['id'], {}) |
            {'pure_gemm_state': 'passed' if case['id'] in measured else 'pending'} for case in expected.values()]
    with csv_file.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(schema='sm103_backward_pure_summary_v1', total=len(expected), passed=len(measured),
        pending=len(expected)-len(measured), fused_measured=0,
        by_phase=dict(Counter(expected[key]['phase'] for key in measured)),
        unique_measured_geometries=len({(r['operand_layout'],r['m'],r['n'],r['k']) for r in measured.values()}),
        sampling='Graph 10+50, random BF16, FP32 accumulation, BF16 output, beta=0',
        boundary='single GPU, full SM cuBLASLt; excludes communication and CP weight-gradient reduction',
        globally_optimal=False, results=measured)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    lines = ['# SM103 BF16 反向纯 GEMM 基线', '',
        f'覆盖：{len(measured)}/{len(expected)}；融合反向尚未测量。单位为单卡 PFLOPS。', '',
        'Graph 10+50；随机 BF16，FP32 累加、BF16 输出，beta=0。满 SM cuBLASLt，',
        '不含通信、转置物化或 CP 的 dW 归约。最多 32 个启发式候选中选优，不宣称全局最优。', '',
        '| 模型 | 投影 | 序列 | CP | dX（NN） | dW（TN） | 融合反向 |',
        '|---|---|---:|---:|---:|---:|---|']
    grouped = {}
    for row in rows:
        key = (row['model'], row['projection'], row['seq'], row['cp'])
        grouped.setdefault(key, {})[row['phase']] = row
    for (model, projection, seq, cp), phases in grouped.items():
        values = [f'{phases[p]["pflops"]:.3f}' if 'pflops' in phases[p] else '—' for p in ('dgrad','wgrad')]
        lines.append(f'| {model} | {projection} | {seq//1024}K | {cp} | {values[0]} | {values[1]} | 未测 |')
    lines += ['', '逐点 MNK、算法、tile、split-K、workspace 和原始证据路径见 [CSV](pure-gemm.csv)。']
    (output / 'README.md').write_text('\n'.join(lines) + '\n')
    return {k:v for k,v in summary.items() if k != 'results'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.manifest, args.runs, args.output), ensure_ascii=False))
