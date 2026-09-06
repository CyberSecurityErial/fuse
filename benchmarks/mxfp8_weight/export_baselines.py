"""Publish compact, provenance-preserving best-tested 2F2B baseline tables.

Input is the completed/audited full_v2 and backward_full_v1 directories, never
pilot data. Raw timings are not relabelled as measurements of production Fuse.
Large candidate/sample vectors remain in the original run archive; hashes and
selected configs/algorithm metadata are preserved in this portable summary.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from matrix import full_matrix
from backward_matrix import full_matrix as backward_matrix


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(value):
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()
                if k not in ('samples_us', 'rank_samples_us', 'isolated_sum_samples_us', 'candidates')}
    if isinstance(value, list):
        return [compact(v) for v in value]
    return value


def write_tables(output, name, rows):
    from comparison_report import config_label, matching_config, mnk_label, number, seq_label, shape_order

    # Preserve the archival JSON representation and its original provenance.
    (output / f'{name}.json').write_text(json.dumps(rows, indent=2) + '\n')
    rows = sorted(rows, key=lambda row: shape_order(row['case']['id']))
    columns = ['id', 'model', 'global_seq', 'cp', 'backend', 'launch', 'weight_mode',
               'mnk', 'b_mnk', 'w_mnk', 'p50_ms', 'p95_ms', 'b_ms', 'w_ms',
               'tflops_per_gpu', 'p50_us', 'p95_us', 'b_us', 'w_us',
               'config', 'gemm_plan', 'source']
    with (output / f'{name}.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            case, record = row['case'], row['record']
            timing = record.get('total', record)
            backward = 'total' in record
            mnk = None if backward else [case['m'], case['n'], case['k']]
            flops = (4 * case['b_mnk'][0] * case['b_mnk'][1] * case['b_mnk'][2] if backward
                     else 2 * case['m'] * case['n'] * case['k'])
            writer.writerow(dict(
                id=case['id'], model=case['model'], global_seq=case['global_seq'],
                cp=case['cp'], backend=record['backend'], launch=record['launch'],
                weight_mode=record.get('weight_mode', ''),
                mnk=json.dumps(mnk) if mnk else '', b_mnk=json.dumps(case['b_mnk']) if backward else '',
                w_mnk=json.dumps(case['w_mnk']) if backward else '',
                p50_ms=timing['p50_us'] / 1000, p95_ms=timing['p95_us'] / 1000,
                b_ms=record['data']['p50_us'] / 1000 if backward else '',
                w_ms=record['weight']['p50_us'] / 1000 if backward else '',
                tflops_per_gpu=flops / (timing['p50_us'] * 1e6),
                p50_us=timing['p50_us'], p95_us=timing['p95_us'],
                b_us=record.get('data', {}).get('p50_us', ''),
                w_us=record.get('weight', {}).get('p50_us', ''),
                config=json.dumps(row['config'], sort_keys=True),
                gemm_plan=json.dumps(record.get('gemm_plan', {
                    'B': record.get('b_plan'), 'W': record.get('w_plan')}), sort_keys=True),
                source=row['source']))
    lines = [f'# MXFP8-weight {name} Benchmark', '', '## 口径', '',
             'TE Userbuffers / cuBLASLt+NCCL 的有限候选最佳基线，不是新 Fuse 算子的计时。', '',
             '沿用 BF16/FP8 表格格式：p50 单位 ms、吞吐 TFLOPS/GPU；S 为展平后的 token 数，M=S/CP。', '',
             '每个样本取跨 rank 最大时延，再报告 p50/p95；JSON/CSV 保留匹配配置、GEMM 算法与原始来源。', '',
             '反向使用 FP32 main_grad；总时间是实际顺序 B→W 计时，不是 B/W 中位数相加。', '']
    if name == 'forward_best':
        for direction in ('gemm_a2a', 'a2a_gemm'):
            for backend in ('cublaslt_nccl', 'teub'):
                for launch in ('eager', 'graph'):
                    selected = [r for r in rows if r['case']['direction'] == direction
                                and r['record']['backend'] == backend and r['record']['launch'] == launch]
                    if not selected:
                        continue
                    lines += [f'## {direction} · {backend} · {launch}', '',
                              '| CP | 模型 | S | GEMM M×N×K | p50 ms | TFLOPS/GPU | 配置 |',
                              '|---:|---|---:|---|---:|---:|---|']
                    for row in selected:
                        c, r = row['case'], row['record']
                        flops = 2 * c['m'] * c['n'] * c['k']
                        lines.append(f"| {c['cp']} | {c['model']} | {seq_label(c['global_seq'])} | "
                                     f"{mnk_label([c['m'], c['n'], c['k']])} | {r['p50_us'] / 1000:.4f} | "
                                     f"{flops / (r['p50_us'] * 1e6):.1f} | {config_label(matching_config(row))} |")
                    lines.append('')
    else:
        for operator in ('qkv', 'oproj'):
            for backend in ('cublaslt_nccl', 'teub'):
                for mode in ('immediate', 'deferred'):
                    grouped = {}
                    for row in rows:
                        c, r = row['case'], row['record']
                        if c['operator'] == operator and r['backend'] == backend and r['weight_mode'] == mode:
                            grouped.setdefault(c['id'], {})[r['launch']] = row
                    if not grouped:
                        continue
                    label = '普通同流 B→W，beta=0' if mode == 'immediate' else 'B/W 分离，main_grad 累加，beta=1'
                    lines += [f'## {operator} backward · {backend} · {label}', '',
                              '| CP | 模型 | S | B GEMM M×N×K | W GEMM M×N×K | Eager B / W / 总 ms | Eager TFLOPS/GPU | Graph B / W / 总 ms | Graph TFLOPS/GPU | 配置 |',
                              '|---:|---|---:|---|---|---:|---:|---:|---:|---|']
                    for pair in grouped.values():
                        c = next(iter(pair.values()))['case']
                        values = [str(c['cp']), c['model'], seq_label(c['global_seq']),
                                  mnk_label(c['b_mnk']), mnk_label(c['w_mnk'])]
                        flops = 4 * c['b_mnk'][0] * c['b_mnk'][1] * c['b_mnk'][2]
                        for launch in ('eager', 'graph'):
                            if launch not in pair:
                                values += ['—', '—']
                                continue
                            r = pair[launch]['record']
                            values += [' / '.join(number(r[p]['p50_us'] / 1000) for p in ('data', 'weight', 'total')),
                                       number(flops / (r['total']['p50_us'] * 1e6), 1)]
                        values.append(' ; '.join(f'{launch}: {config_label(matching_config(row))}' for launch, row in pair.items()))
                        lines.append('| ' + ' | '.join(values) + ' |')
                    lines.append('')
    (output / f'{name}.md').write_text('\n'.join(lines).rstrip() + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    archive, output = args.archive.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    forward, backward, provenance, parsed = [], [], {}, {}

    def read(path):
        if path not in parsed:
            provenance[str(path.relative_to(archive))] = digest(path)
            parsed[path] = json.loads(path.read_text())
        return parsed[path]

    for directory, count in [('full_v2', 768), ('backward_full_v1', 1536)]:
        audit = read(archive / directory / 'coverage.json')
        assert audit['complete'] and not audit['errors'], directory
    fmatrix = {c['id']: c for c in full_matrix()}
    bmatrix = {c['id']: c for c in backward_matrix()}
    for cp in (4, 8):
        combined = read(archive / 'full_v2' / f'combined_cp{cp}.json')
        for item in combined['cases']:
            case = item['case']
            assert case == fmatrix[case['id']]
            for record in item['best_tested']:
                source = archive / 'full_v2' / Path(record['source_run']).name
                raw = read(source)
                source_hash = provenance[str(source.relative_to(archive))]
                for path, expected in combined['source_reports'].items():
                    if Path(path).name == source.name:
                        assert source_hash == expected
                        break
                else:
                    raise AssertionError(f'untracked source {source}')
                stored = compact(record)
                stored.pop('source_run', None)
                forward.append(dict(case=case, record=stored,
                                    config=dict(operator=record['config'], nccl_environment=raw['nccl_environment']),
                                    source=str(source.relative_to(archive)), source_sha256=source_hash))
    cache = {}
    with (archive / 'backward_full_v1/coverage.csv').open() as f:
        for selected in csv.DictReader(f):
            source = archive / 'backward_full_v1' / Path(selected['source']).name
            if source not in cache:
                raw = read(source)
                cache[source] = raw, {item['case']['id']: item for item in raw['cases']}
            raw, lookup = cache[source]
            item = lookup[selected['id']]
            assert item['case'] == bmatrix[selected['id']]
            matches = [r for r in item['records'] if
                       all(str(r[k]) == selected[k] for k in ('backend', 'launch', 'weight_mode', 'grad_dtype'))]
            assert len(matches) == 1
            r = matches[0]
            assert abs(r['total']['p50_us'] - float(selected['total_us'])) < 1e-6
            backward.append(dict(case=item['case'], record=compact(r),
                                 config=dict(sms=r['sms'], environment=raw['environment'],
                                             fixed=dict(pack_block=512, pack_warps=4, push=True,
                                                        use_ce=False, max_send_streams=3)),
                                 source=str(source.relative_to(archive)),
                                 source_sha256=provenance[str(source.relative_to(archive))]))
    for rows, expected in [(forward, 768), (backward, 1536)]:
        keys = [(r['case']['id'], r['record']['backend'], r['record']['launch'],
                 r['record'].get('weight_mode')) for r in rows]
        assert len(keys) == len(set(keys)) == expected
        rows.sort(key=lambda r: (r['case']['id'], r['record']['backend'],
                                 r['record']['launch'], r['record'].get('weight_mode', '')))
    write_tables(output, 'forward_best', forward)
    write_tables(output, 'backward_best', backward)
    meta = dict(schema='mxfp8-weight-published-baselines-v1', forward_rows=768, backward_rows=1536,
                settings=384, original_archive=str(archive), source_reports=provenance,
                devices=combined['devices'], quantization=combined['quantization'],
                original_git_head=combined['git_head'],
                software={k: combined[k] for k in ('torch', 'cuda', 'transformer_engine', 'triton')},
                forward_benchmark_sources=combined['sources'],
                backward_benchmark_sources=raw['sources'],
                gemm='BF16 operands / FP32 accumulation', communication='BF16', main_grad='FP32',
                timing='10 warmups / 50 independent formal samples / sample-wise rank-max p50,p95',
                timed='runtime dequant + projection + route; backward W contains no weight dequant',
                algorithm_metadata_rank=0, algorithm_selection='independent on each rank',
                limitation='finite candidate winners, not global optima; imported baseline results, not Fuse timings',
                exported_files={p.name: digest(p) for p in sorted(output.glob('*_best.*'))})
    (output/'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(f'Published {len(forward)} forward + {len(backward)} backward rows to {output}')


if __name__ == '__main__':
    main()
