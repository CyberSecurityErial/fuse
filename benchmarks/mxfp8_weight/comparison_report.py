"""Compare matched 2F2B baselines, production Fuse and pure classic cuBLAS.

The published baseline winner is selected independently for each case, launch
and backward mode. Backward selects by the measured sequential total, then
retains that backend's B/W measurements and configuration. Pure GEMM excludes
communication and runtime dequantization: its rate is a compute-only reference,
not an end-to-end competitor or a hardware utilization measurement.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINES = ('cublaslt_nccl', 'teub')
MEASURED = ('fuse', 'pure_cublas')
SCHEMA = 'mxfp8-production-operators-v1'


def positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{label}: expected a finite positive number, got {value!r}')
    return value


def operator_of(case):
    if 'direction' in case:
        mapping = {'gemm_a2a': 'qkv_forward', 'a2a_gemm': 'oproj_forward'}
        operator = mapping.get(case['direction'])
    else:
        mapping = {'qkv': 'qkv_backward', 'oproj': 'oproj_backward',
                   'qkv_backward': 'qkv_backward', 'oproj_backward': 'oproj_backward'}
        operator = mapping.get(case.get('operator'))
    if operator is None:
        raise ValueError(f'unknown operator in case {case.get("id")}')
    return operator


def geometry(case):
    """Match physical geometry and layout, not merely a human-readable ID."""
    operator = operator_of(case)
    for field in ('global_seq', 'cp', 'm'):
        value = positive(case[field], field)
        if not isinstance(value, int):
            raise ValueError(f'{field} must be an integer')
    if case.get('batch', 1) != 1 or case['global_seq'] != case['m'] * case['cp']:
        raise ValueError('expected flattened T = local M * CP, with no separate batch')
    if operator.endswith('forward'):
        shapes = {'forward': [case['m'], case['n'], case['k']]}
    else:
        shapes = {'data': list(case['b_mnk']), 'weight': list(case['w_mnk'])}
        b, w = shapes['data'], shapes['weight']
        if len(b) != 3 or len(w) != 3 or b != [case['m'], w[1], w[0]] or w[2] != case['m']:
            raise ValueError('inconsistent backward B/W MNK')
    for mnk in shapes.values():
        for value in mnk:
            positive(value, 'MNK')
            if not isinstance(value, int):
                raise ValueError('MNK must be integers')
    return dict(operator=operator, model=case['model'], T=case['global_seq'],
                CP=case['cp'], M=case['m'], shapes=shapes,
                layout=case.get('layout'), visible_devices=case.get('visible_devices'),
                hidden=case.get('hidden'), q_heads=case.get('q_heads'),
                kv_heads=case.get('kv_heads'), head_dim=case.get('head_dim'))


def record_key(case, record):
    operator = operator_of(case)
    launch, mode = record['launch'], record.get('weight_mode')
    if launch not in ('eager', 'graph'):
        raise ValueError(f'unknown launch {launch!r}')
    if operator.endswith('backward'):
        if mode not in ('immediate', 'deferred'):
            raise ValueError(f'unknown backward weight mode {mode!r}')
        if record.get('grad_dtype') != 'fp32':
            raise ValueError('backward comparison requires FP32 main_grad')
        if record.get('beta') != int(mode == 'deferred'):
            raise ValueError('backward beta must match immediate=0 / deferred=1')
    elif mode not in (None, ''):
        raise ValueError('forward has no weight-gradient mode')
    return case['id'], operator, launch, mode or None


def timings(case, record, published=False):
    if operator_of(case).endswith('forward'):
        result = {'forward': record if published else record['forward']}
    else:
        # Never synthesize total from isolated B/W medians.
        result = {phase: record[phase] for phase in ('data', 'weight', 'total')}
    for phase, timing in result.items():
        median = positive(timing['p50_us'], f'{phase}.p50_us')
        if 'p95_us' in timing and positive(timing['p95_us'], f'{phase}.p95_us') < median:
            raise ValueError(f'{phase}: p95 must be at least p50')
    return result


def matching_config(row):
    """Keep the winning transport knobs and GEMM plans without raw samples."""
    config = dict(row['config'])
    record = row['record']
    for field in ('gemm_plan', 'b_plan', 'w_plan'):
        if field in record:
            config[field] = {k: v for k, v in record[field].items() if k != 'candidates'}
    return config


def select_baselines(forward, backward):
    winners, seen, shapes = {}, set(), {}
    for rows, suffix in ((forward, 'forward'), (backward, 'backward')):
        for row in rows:
            case, record = row['case'], row['record']
            key = record_key(case, record)
            if not key[1].endswith(suffix) or record['backend'] not in BASELINES:
                raise ValueError(f'invalid published baseline {key}')
            signature = geometry(case)
            if case['id'] in shapes and shapes[case['id']] != signature:
                raise ValueError(f'conflicting baseline geometry: {case["id"]}')
            shapes[case['id']] = signature
            identity = key + (record['backend'],)
            if identity in seen:
                raise ValueError(f'duplicate baseline row: {identity}')
            seen.add(identity)
            phase = 'total' if suffix == 'backward' else 'forward'
            median = timings(case, record, published=True)[phase]['p50_us']
            candidate = (median, record['backend'])
            previous = winners.get(key)
            if previous is None or candidate < previous[0]:
                winners[key] = candidate, row
    return {key: value[1] for key, value in winners.items()}


def collect_measurements(reports, winners, allow_diagnostic=False):
    measured = {}
    for source, report in reports:
        if report.get('schema') != SCHEMA or not isinstance(report.get('complete'), bool):
            raise ValueError(f'unsupported measurement report: {source}')
        for item in report['cases']:
            case = item['case']
            if item['operator'] != operator_of(case):
                raise ValueError(f'operator differs from registry: {case["id"]}')
            for record in item['records']:
                key = record_key(case, record)
                backend = record['backend']
                if backend not in MEASURED or key not in winners:
                    raise ValueError(f'unmatched measured row: {key}, {backend}')
                if geometry(case) != geometry(winners[key]['case']):
                    raise ValueError(f'measured geometry/layout differs: {key}')
                identity = key + (backend,)
                if identity in measured:
                    raise ValueError(f'duplicate measured row: {identity}')
                timings(case, record)
                if not allow_diagnostic and (record.get('warmup') != 10 or record.get('iterations') != 50):
                    raise ValueError(f'formal comparison requires 10 warmups / 50 samples: {identity}')
                if not isinstance(record.get('config'), dict) or not record.get('correctness'):
                    raise ValueError(f'missing measured config/correctness: {identity}')
                measured[identity] = dict(record=record, config=record['config'], source=source)
    return measured


def phase_flops(shapes, phase):
    if phase == 'total':
        return sum(2 * math.prod(shapes[p]) for p in ('data', 'weight'))
    return 2 * math.prod(shapes[phase])


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['operator'], row['CP'], row['launch'], row['weight_mode'], row['phase'])].append(row)
    result = []
    for key, members in sorted(groups.items()):
        paired = [r for r in members if r['baseline_over_fuse'] is not None]
        pure = [r for r in members if r['fuse_over_pure_rate_pct'] is not None]
        geometric = lambda values: math.exp(sum(map(math.log, values)) / len(values)) if values else None
        result.append(dict(operator=key[0], CP=key[1], launch=key[2], weight_mode=key[3], phase=key[4],
                           baseline_rows=len(members), matched_fuse_rows=len(paired),
                           fuse_wins=sum(r['baseline_over_fuse'] > 1 for r in paired),
                           baseline_over_fuse_geomean=geometric([r['baseline_over_fuse'] for r in paired]),
                           matched_pure_rows=len(pure),
                           fuse_over_pure_rate_pct_geomean=geometric([r['fuse_over_pure_rate_pct'] for r in pure])))
    return result


def build_report(forward, backward, reports=(), allow_diagnostic=False):
    winners = select_baselines(forward, backward)
    measured = collect_measurements(reports, winners, allow_diagnostic)
    rows = []
    for key, baseline in sorted(winners.items()):
        case, record = baseline['case'], baseline['record']
        shape = geometry(case)
        for phase, timing in timings(case, record, published=True).items():
            flops = phase_flops(shape['shapes'], phase)
            mnk = shape['shapes'] if phase == 'total' else shape['shapes'][phase]
            row = dict(id=key[0], operator=key[1], model=shape['model'], T=shape['T'], CP=shape['CP'],
                       M=shape['M'], mnk=mnk, launch=key[2], weight_mode=key[3],
                       grad_dtype=record.get('grad_dtype'), beta=record.get('beta'), phase=phase,
                       flops_per_gpu=flops, best_backend=record['backend'], best_p50_us=timing['p50_us'],
                       best_p95_us=timing.get('p95_us'),
                       best_tflops_per_gpu=flops / (timing['p50_us'] * 1e6),
                       best_config=matching_config(baseline), baseline_source=baseline.get('source'))
            for backend in MEASURED:
                item = measured.get(key + (backend,))
                median = timings(case, item['record'])[phase]['p50_us'] if item else None
                row[backend + '_p50_us'] = median
                row[backend + '_p95_us'] = timings(case, item['record'])[phase].get('p95_us') if item else None
                row[backend + '_tflops_per_gpu'] = flops / (median * 1e6) if median is not None else None
                row[backend + '_config'] = matching_config(item) if item else None
                row[backend + '_source'] = item['source'] if item else None
            row['baseline_over_fuse'] = timing['p50_us'] / row['fuse_p50_us'] if row['fuse_p50_us'] else None
            row['fuse_over_pure_rate_pct'] = (100 * row['fuse_tflops_per_gpu'] / row['pure_cublas_tflops_per_gpu']
                                               if row['fuse_p50_us'] and row['pure_cublas_p50_us'] else None)
            rows.append(row)
    return dict(schema='mxfp8-matched-comparison-v1', diagnostic=allow_diagnostic,
                baseline_winners=len(winners), phase_rows=len(rows),
                measured_records=len(measured), missing_fuse_records=len(winners) - sum(k[-1] == 'fuse' for k in measured),
                missing_pure_cublas_records=len(winners) - sum(k[-1] == 'pure_cublas' for k in measured),
                measurement_reports=[dict(source=source, complete=report['complete']) for source, report in reports],
                notes=[
                    'Historical published baseline winners, not same-run quantization-only A/B results.',
                    'Backward winner is selected by actual sequential total; its B/W timings keep the same backend/config.',
                    'Deferred total times B then W with beta=1, not a ZeroBubble scheduler gain.',
                    'Effective TFLOPS/GPU = local GEMM FLOPs / measured phase latency; total counts both GEMMs.',
                    'Pure classic cuBLAS excludes communication and runtime DQ; its compute-rate ratio is not hardware utilization.',
                    'Null means unmeasured; summaries include only matched case/launch/mode/phase rows.',
                ], rows=rows, summary=summarize(rows))


def cell(value):
    if value is None:
        return '—'
    if isinstance(value, float):
        return f'{value:.3f}'
    if isinstance(value, (dict, list)):
        value = json.dumps(value, sort_keys=True, separators=(',', ':'))
    return str(value).replace('|', '\\|').replace('\n', ' ')


def number(value, digits=4, suffix=''):
    return '—' if value is None else f'{value:.{digits}f}{suffix}'


def mnk_label(mnk):
    return chr(96) + '×'.join(map(str, mnk)) + chr(96)


def seq_label(tokens):
    # Preserve the old S column spelling; S is the flattened token count.
    return f'{tokens // 1024}K' if tokens % 1024 == 0 else str(tokens)


@lru_cache(maxsize=1)
def registry_order():
    from matrix import full_matrix
    from backward_matrix import full_matrix as backward_matrix
    return {case['id']: index for index, case in enumerate(full_matrix() + backward_matrix())}


def shape_order(identifier):
    return registry_order().get(identifier, len(registry_order())), identifier


def config_label(config):
    """Readable knob names; JSON/CSV retain the exact nested configuration."""
    if not config:
        return '—'
    knobs = dict(config.get('fixed', {}), **config.get('operator', config))
    if all(key in config for key in ('comm_ctas', 'raster', 'swizzle', 'tile_m', 'tile_n', 'cluster_m')):
        parts = [f"c{config['comm_ctas']}/r{config['raster'].upper()}/s{config['swizzle']}/"
                 f"M{config['tile_m']}N{config['tile_n']}C{config['cluster_m']}"]
        if config.get('weight_gemm'):
            w = config['weight_gemm']
            parts.append(f"W:M{w['tile_m']}N{w['tile_n']}C{w['cluster_m']}/fp32/beta{knobs['beta']}")
    else:
        parts = [str(knobs['schedule'])] if 'schedule' in knobs else []
        parts += [f'{label}{knobs[key]}' for key, label in
                  (('sms', 'sms'), ('chunks', 'chunks'), ('streams', 'streams'),
                   ('pack_block', 'pb'), ('pack_warps', 'pw'), ('beta', 'beta')) if key in knobs]
        if 'push' in knobs:
            parts.append('push' if knobs['push'] else 'pull')
        if 'use_ce' in knobs:
            parts.append('ce' if knobs['use_ce'] else 'sm')
    environment = config.get('nccl_environment', config.get('environment', {}))
    for key, label, divisor in (('NCCL_MAX_P2P_NCHANNELS', 'ch', 1),
                                ('NCCL_P2P_NVL_CHUNKSIZE', 'chunk', 1024),
                                ('NCCL_P2P_LL_THRESHOLD', 'll', 1024)):
        if environment.get(key) is not None:
            parts.append(f'{label}{int(environment[key]) // divisor}')
    for key, label in (('gemm_plan', 'algo'), ('b_plan', 'B_algo'), ('w_plan', 'W_algo')):
        if isinstance(config.get(key), dict) and 'algo_id' in config[key]:
            parts.append(f'{label}={config[key]["algo_id"]}')
    if not parts:
        parts = [f'{key}={value}' for key, value in config.items() if not isinstance(value, (dict, list))]
    return cell('/'.join(parts))


def wide_rows(report):
    """Old comparison_summary layout: one shape/mode, Eager and Graph columns."""
    grouped = defaultdict(dict)
    for row in report['rows']:
        grouped[(row['id'], row['weight_mode'])][(row['launch'], row['phase'])] = row
    result = []
    for (identifier, mode), phases in sorted(grouped.items(), key=lambda pair: (shape_order(pair[0][0]), pair[0][1] or '')):
        first = next(iter(phases.values()))
        backward = first['operator'].endswith('backward')
        main_phase = 'total' if backward else 'forward'
        row = dict(id=identifier, operator=first['operator'], model=first['model'],
                   global_seq=first['T'], cp=first['CP'], m=first['M'], weight_mode=mode,
                   grad_dtype=first['grad_dtype'], beta=first['beta'])
        shapes = next(p['mnk'] for p in phases.values() if p['phase'] == main_phase)
        if backward:
            row.update(b_mnk=shapes['data'], w_mnk=shapes['weight'])
        else:
            row.update(n=shapes[1], k=shapes[2])
        for launch in ('eager', 'graph'):
            selected = phases.get((launch, main_phase))
            row[f'{launch}_best_external_backend'] = selected['best_backend'] if selected else None
            for backend, prefix in (('best', f'{launch}_best_external'), ('fuse', launch),
                                    ('pure_cublas', f'{launch}_cublas')):
                for percentile in ('p50', 'p95'):
                    value = selected.get(f'{backend}_{percentile}_us') if selected else None
                    row[f'{prefix}_{percentile}_ms'] = value / 1000 if value is not None else None
                row[f'{prefix}_p50_tflops_per_gpu'] = selected[f'{backend}_tflops_per_gpu'] if selected else None
                row[f'{prefix}_config'] = selected[f'{backend}_config'] if selected else None
                if backward:
                    for phase, label in (('data', 'b'), ('weight', 'w'), ('total', 'total')):
                        timing = phases.get((launch, phase))
                        value = timing[f'{backend}_p50_us'] if timing else None
                        row[f'{prefix}_{label}_p50_ms'] = value / 1000 if value is not None else None
                        row[f'{prefix}_{label}_p50_tflops_per_gpu'] = timing[f'{backend}_tflops_per_gpu'] if timing else None
            row[f'{launch}_speedup_over_best_external'] = selected['baseline_over_fuse'] if selected else None
            row[f'{launch}_throughput_as_cublas_percent'] = selected['fuse_over_pure_rate_pct'] if selected else None
        result.append(row)
    return result


def write_rows(output, name, rows):
    (output / f'{name}.json').write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n')
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (output / f'{name}.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def triplet(row, prefix):
    return ' / '.join(number(row[f'{prefix}_{phase}_p50_ms']) for phase in ('b', 'w', 'total'))


def comparison_markdown(report, rows):
    lines = ['# SM90 MXFP8-weight 四算子 Benchmark', '', '## 口径', '',
             '沿用 BF16/FP8 的逐 shape、Eager/Graph 并列表和 ms 单位。S 表示展平后的总 token 数 T，M=T/CP。', '',
             '基线和 Fuse 均包含运行时 MXFP8→BF16 反量化与通信；经典 cuBLAS 仅测纯 BF16 GEMM。', '',
             '反向 main_grad 为 FP32；普通模式 beta=0，分离模式 beta=1。总时间是实际顺序 B→W 计时，不是两个中位数相加。', '',
             '最佳外部基线逐 shape、逐启动方式、逐模式选择。反向按总时间选后端，B/W 保留同一后端及配置。', '',
             '历史基线不冒充同轮 A/B；— 表示未测。吞吐占比不等于硬件利用率，分离模式总时间不表示 ZeroBubble 调度收益。', '',
             '## 汇总', '',
             '| 算子 | CP | 启动 | 模式 | 阶段 | 已测 / 基线点 | Fuse 获胜点 | 相对最佳外部基线几何平均 | 经典cuBLAS吞吐占比几何平均 |',
             '|---|---:|---|---|---|---:|---:|---:|---:|']
    if report['diagnostic']:
        lines[2:2] = ['**诊断表：允许非正式样本数，不得作为全量正式 benchmark 结果。**', '']
    for row in report['summary']:
        if row['phase'] not in ('forward', 'total'):
            continue
        values = [row[k] for k in ('operator', 'CP', 'launch', 'weight_mode', 'phase')]
        values += [f"{row['matched_fuse_rows']}/{row['baseline_rows']}", row['fuse_wins'],
                   number(row['baseline_over_fuse_geomean'], 3, '×'),
                   number(row['fuse_over_pure_rate_pct_geomean'], 1, '%')]
        lines.append('| ' + ' | '.join(map(cell, values)) + ' |')
    for operator in ('qkv_forward', 'oproj_forward', 'qkv_backward', 'oproj_backward'):
        backward = operator.endswith('backward')
        for mode in (('immediate', 'deferred') if backward else (None,)):
            members = [r for r in rows if r['operator'] == operator and r['weight_mode'] == mode]
            if not members:
                continue
            label = '' if mode is None else ('普通同流 B→W，beta=0' if mode == 'immediate' else 'B/W 分离，main_grad 累加，beta=1')
            lines += ['', f'## {operator} {label}'.rstrip(), '']
            if backward:
                lines += ['| CP | 模型 | S | B GEMM M×N×K | W GEMM M×N×K | Eager 最佳基线 B / W / 总 ms · TFLOPS | Graph 最佳基线 B / W / 总 ms · TFLOPS | Eager B / W / 总 ms | Eager TFLOPS / 外部加速 / cuBLAS占比 | Graph B / W / 总 ms | Graph TFLOPS / 外部加速 / cuBLAS占比 | 经典cuBLAS Eager B / W / 总 ms · TFLOPS | 经典cuBLAS Graph B / W / 总 ms · TFLOPS | 配置 |',
                          '|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
            else:
                lines += ['| CP | 模型 | S | GEMM M×N×K | Eager p50/p95 ms / TFLOPS | Graph p50/p95 ms / TFLOPS | 最佳外部基线 Eager ms / TFLOPS | 最佳外部基线 Graph ms / TFLOPS | 经典cuBLAS Eager ms / TFLOPS | 经典cuBLAS Graph ms / TFLOPS | Eager/Graph vs cuBLAS | Eager / external | Graph / external | 配置 |',
                          '|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
            for row in members:
                values = [row['cp'], row['model'], seq_label(row['global_seq'])]
                if backward:
                    values += [mnk_label(row['b_mnk']), mnk_label(row['w_mnk'])]
                    for launch in ('eager', 'graph'):
                        prefix = f'{launch}_best_external'
                        values.append(f"{cell(row[prefix + '_backend'])}: {triplet(row, prefix)} · {number(row[prefix + '_p50_tflops_per_gpu'], 1)}")
                    for launch in ('eager', 'graph'):
                        values += [triplet(row, launch),
                                   f"{number(row[launch + '_p50_tflops_per_gpu'], 1)} / {number(row[launch + '_speedup_over_best_external'], 3, '×')} / {number(row[launch + '_throughput_as_cublas_percent'], 1, '%')}"]
                    for launch in ('eager', 'graph'):
                        prefix = f'{launch}_cublas'
                        values.append(f"{triplet(row, prefix)} · {number(row[prefix + '_p50_tflops_per_gpu'], 1)}")
                else:
                    values.append(mnk_label([row['m'], row['n'], row['k']]))
                    for launch in ('eager', 'graph'):
                        values.append(f"{number(row[launch + '_p50_ms'])}/{number(row[launch + '_p95_ms'])} / {number(row[launch + '_p50_tflops_per_gpu'], 1)}")
                    for suffix in ('best_external', 'cublas'):
                        for launch in ('eager', 'graph'):
                            prefix = f'{launch}_{suffix}'
                            backend = f"{cell(row[prefix + '_backend'])}: " if suffix == 'best_external' else ''
                            values.append(f"{backend}{number(row[prefix + '_p50_ms'])} / {number(row[prefix + '_p50_tflops_per_gpu'], 1)}")
                    values.append('/'.join(number(row[l + '_throughput_as_cublas_percent'], 1, '%') for l in ('eager', 'graph')))
                    values += [number(row[l + '_speedup_over_best_external'], 3, '×') for l in ('eager', 'graph')]
                values.append(' ; '.join(f'{launch} Fuse: {config_label(row[launch + "_config"])}; baseline: {config_label(row[launch + "_best_external_config"])}' for launch in ('eager', 'graph')))
                lines.append('| ' + ' | '.join(map(cell, values)) + ' |')
    return '\n'.join(lines) + '\n'


def write_report(report, output, phase_details=False):
    output.mkdir(parents=True, exist_ok=True)
    rows = wide_rows(report)
    write_rows(output, 'comparison_summary', rows)
    if phase_details:
        write_rows(output, 'comparison_phases', report['rows'])
    metadata = {key: value for key, value in report.items() if key != 'rows'}
    (output / 'comparison_metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    (output / 'comparison_summary.md').write_text(comparison_markdown(report, rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--published', type=Path, default=ROOT / 'results/mxfp8_weight/published')
    parser.add_argument('--measurements', type=Path, nargs='*', default=[])
    parser.add_argument('--allow-diagnostic', action='store_true',
                        help='allow non-10/50 samples and mark the report as diagnostic')
    parser.add_argument('--phase-details', action='store_true',
                        help='also export redundant normalized phase tables (off by default)')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/mxfp8_weight/comparison')
    args = parser.parse_args()
    paths = [args.published / 'forward_best.json', args.published / 'backward_best.json', *args.measurements]
    loaded = [json.loads(path.read_text()) for path in paths]
    report = build_report(loaded[0], loaded[1], [(str(p), r) for p, r in zip(args.measurements, loaded[2:])],
                          allow_diagnostic=args.allow_diagnostic)
    report['input_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    write_report(report, args.output, phase_details=args.phase_details)
    print(f"Wrote {report['baseline_winners']} matched baseline winners / {report['phase_rows']} phase rows to {args.output}")


if __name__ == '__main__':
    main()
