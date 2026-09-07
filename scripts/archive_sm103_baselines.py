#!/usr/bin/env python3
"""Compact audited comparison evidence into one updatable baseline CSV.

Consumes an already audited report, not arbitrary raw measurements. Does not
delete input archives or claim independent winner remeasurement.
"""
import argparse
import csv
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.evidence.read_text())
    assert report['schema'] == 'sm103_bf16_report_v1' and report['complete']
    rows = []
    for cell in report['measurements']:
        for backend in ('cublaslt_nccl', 'te_ub', 'pure'):
            value = cell['measurements'][backend]
            provenance = value['provenance']
            case = cell['case']
            config = dict(value.get('config', {}))
            if backend == 'pure':
                raw = json.loads(Path(provenance['raw']['path']).read_text())
                geometry = next(g for g in raw['geometries'] if
                                all(g['shape'][k] == case[k] for k in ('m', 'n', 'k')))
                result = next(r for r in geometry['results'] if
                              r['precision'] == 'bf16' and r['launch'] == cell['launch'])
                tuning = result['tuning']
                config = {k: v for k, v in tuning.items() if k != 'candidates'}
                matches = [c for c in tuning['candidates'] if
                           c['algorithm'] == tuning['algorithm'] and
                           math.isclose(c['tune_ms'], tuning['best_ms'], rel_tol=1e-5)]
                config['selected_candidate_records'] = matches
                config['library_sha256'] = provenance['library_sha256']
                config['exact_serialized_algorithm_available'] = False
            p50, p95 = float(value['p50_ms']), float(value['p95_ms'])
            assert math.isfinite(p50) and p50 > 0 and math.isfinite(p95) and p95 >= p50
            rows.append(dict(
                **case, backend='cublaslt_gemm' if backend == 'pure' else backend,
                precision='bf16', node=cell['node'], launch=cell['launch'],
                layout=value.get('layout', 'pure_gemm'),
                timing_scope=value.get('scope', 'distributed_maxrank'),
                environment=provenance['environment_fingerprint'],
                p50_ms=p50, p95_ms=p95,
                pflops_per_gpu=2*case['m']*case['n']*case['k']/(p50*1e12),
                config_json=json.dumps(config, sort_keys=True),
                run_id=provenance['run_id'], source_id=provenance['source_id'],
                raw_sha256=provenance['raw']['sha256'],
                raw_pointer=json.dumps(provenance.get('pointer', ''), sort_keys=True),
                warmup=10, samples=50, selection='best_observed_not_independent_retest'))
    fields = list(rows[0])
    key_fields = [k for k in fields if k not in (
        'p50_ms', 'p95_ms', 'pflops_per_gpu', 'config_json', 'run_id',
        'source_id', 'raw_sha256', 'raw_pointer', 'selection')]
    if args.output.exists():
        with args.output.open() as stream:
            reader = csv.DictReader(stream)
            assert reader.fieldnames == fields, 'Existing archive schema differs'
            rows.extend(reader)
    winners = {}
    for row in rows:
        key = tuple(str(row[k]) for k in key_fields)
        if key not in winners or float(row['p50_ms']) < float(winners[key]['p50_ms']):
            winners[key] = row
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.csv.tmp')
    with temporary.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(winners[k] for k in sorted(winners))
    temporary.replace(args.output)
    print(json.dumps({'rows': len(winners), 'table': str(args.output)}))


if __name__ == '__main__':
    main()
