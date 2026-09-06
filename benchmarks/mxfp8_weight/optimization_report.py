"""Compact optimization progress from the existing wide comparison summary.

Only forward and actual backward-total boundaries count. Eager/Graph and
immediate/deferred remain separate measurements. Coverage comes from the
original registries; model names are never restated here. Milestones require
complete coverage plus the companion non-diagnostic comparison metadata.
The optional pure-cuBLAS section is a measured no-A2A/no-DQ GEMM reference,
not a hardware bound or a replacement for the communication baseline.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median

from matrix import ROOT, full_matrix
from backward_matrix import full_matrix as backward_matrix

LONG_TOKENS = (131072, 262144, 524288)
TARGETS = (1.2, 1.3)
REFERENCE_TARGETS = (1.2, 1.3, 1.4, 1.5)


def geometric_mean(values):
    return math.exp(math.fsum(math.log(value) for value in values) / len(values)) if values else None


def positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{label}: expected finite positive timing/ratio')
    return value


def expected_rows():
    result = {}
    for case in full_matrix() + backward_matrix():
        forward = 'direction' in case
        operator = ({'gemm_a2a': 'qkv_forward', 'a2a_gemm': 'oproj_forward'}[case['direction']]
                    if forward else case['operator'] + '_backward')
        for mode in ((None,) if forward else ('immediate', 'deferred')):
            result[(case['id'], mode)] = dict(case=case, operator=operator)
    return result


def extract_boundaries(rows, expected):
    if not isinstance(rows, list):
        raise ValueError('comparison_summary.json must contain wide rows, not a raw runner report')
    seen, records = set(), {}
    for row in rows:
        key = row['id'], row['weight_mode']
        if key not in expected:
            raise ValueError(f'unexpected shape/mode: {key}')
        if key in seen:
            raise ValueError(f'duplicate wide row: {key}')
        seen.add(key)
        contract = expected[key]
        case, operator = contract['case'], contract['operator']
        if row['operator'] != operator:
            raise ValueError(f'operator differs from registry: {key}')
        for field in ('model', 'global_seq', 'cp', 'm'):
            if row[field] != case[field]:
                raise ValueError(f'{field} differs from registry: {key}')
        backward = operator.endswith('backward')
        fields = ('b_mnk', 'w_mnk') if backward else ('n', 'k')
        if any(row[field] != case[field] for field in fields):
            raise ValueError(f'MNK differs from registry: {key}')
        if backward and (row.get('grad_dtype') != 'fp32' or row.get('beta') != int(key[1] == 'deferred')):
            raise ValueError(f'backward needs FP32 main_grad and matching beta: {key}')
        for launch in ('eager', 'graph'):
            base, fuse = row.get(f'{launch}_best_external_p50_ms'), row.get(f'{launch}_p50_ms')
            for value, label in ((base, 'baseline'), (fuse, 'Fuse')):
                if value is not None:
                    positive(value, f'{key}/{launch}/{label}')
            if base is None or fuse is None:
                continue
            if row.get(f'{launch}_best_external_backend') not in ('teub', 'cublaslt_nccl'):
                raise ValueError(f'unknown external baseline: {key}/{launch}')
            if backward:
                # The wide main p50 must refer to ACTUAL sequential total,
                # not an accidentally substituted isolated B or B+W median.
                for prefix, value in ((launch, fuse), (f'{launch}_best_external', base)):
                    total = positive(row.get(f'{prefix}_total_p50_ms'), f'{key}/{prefix}/actual total')
                    if not math.isclose(value, total, rel_tol=1e-9, abs_tol=1e-12):
                        raise ValueError(f'backward main p50 differs from actual total: {key}/{prefix}')
            ratio = positive(base / fuse, f'{key}/{launch}/speedup')
            recorded = row.get(f'{launch}_speedup_over_best_external')
            if recorded is not None and not math.isclose(positive(recorded, 'recorded speedup'), ratio,
                                                         rel_tol=1e-6, abs_tol=1e-9):
                raise ValueError(f'stored speedup disagrees with timings: {key}/{launch}')
            p95 = row.get(f'{launch}_p95_ms')
            tail_ratio = (p95 / fuse if not isinstance(p95, bool) and isinstance(p95, (int, float))
                          and math.isfinite(p95) and p95 >= fuse else None)
            if tail_ratio is not None and not math.isfinite(tail_ratio):
                tail_ratio = None
            records[key + (launch,)] = dict(id=key[0], operator=operator, cp=case['cp'],
                                           global_seq=case['global_seq'], launch=launch,
                                           weight_mode=key[1], speedup=ratio, p95_over_p50=tail_ratio,
                                           baseline_ms=base, fuse_ms=fuse)
    expected_boundaries = len(expected) * 2
    coverage = dict(expected_settings=len({key[0] for key in expected}),
                    observed_settings=len({key[0] for key in seen}),
                    expected_wide_rows=len(expected), observed_wide_rows=len(seen),
                    expected_boundary_records=expected_boundaries, measured_boundary_records=len(records),
                    missing_wide_rows=len(expected) - len(seen),
                    missing_boundary_records=expected_boundaries - len(records),
                    complete=len(seen) == len(expected) and len(records) == expected_boundaries)
    return records, coverage


def statistics(values, expected_count):
    return dict(measured=len(values), expected=expected_count,
                wins=sum(value > 1 for value in values), geomean=geometric_mean(values),
                minimum=min(values) if values else None, maximum=max(values) if values else None)


def tail_statistics(rows):
    ratios = [row['p95_over_p50'] for row in rows if row.get('p95_over_p50') is not None]
    return dict(p95_p50_measured=len(ratios), p95_p50_median=median(ratios) if ratios else None,
                p95_p50_maximum=max(ratios) if ratios else None)


def expected_boundaries(expected, long_only):
    return {key + (launch,): dict(operator=item['operator'], cp=item['case']['cp'],
                                 launch=launch, weight_mode=key[1])
            for key, item in expected.items()
            if not long_only or item['case']['global_seq'] in LONG_TOKENS
            for launch in ('eager', 'graph')}


def aggregate(records, expected):
    """Compare equal boundary weights with equal operator weights explicitly."""
    matched = {key: records[key] for key in expected if key in records}
    boundary = statistics([row['speedup'] for row in matched.values()], len(expected))
    boundary.update(tail_statistics(matched.values()))
    required = defaultdict(set)
    for (identifier, mode, launch), row in expected.items():
        required[(row['operator'], identifier, launch)].add(mode)
    observed = defaultdict(dict)
    for (identifier, mode, launch), row in matched.items():
        observed[(row['operator'], identifier, launch)][mode] = row['speedup']
    per_operator, expected_operator = defaultdict(list), defaultdict(int)
    paired = 0
    for key, modes in required.items():
        expected_operator[key[0]] += 1
        # Pair both backward modes before giving this shape/launch one vote.
        if set(observed[key]) == modes:
            per_operator[key[0]].append(geometric_mean(list(observed[key].values())))
            paired += 1
    operators = {operator: statistics(per_operator[operator], count)
                 for operator, count in sorted(expected_operator.items())}
    means = [row['geomean'] for row in operators.values() if row['geomean'] is not None]
    balanced = dict(geomean=geometric_mean(means) if len(means) == len(operators) else None,
                    expected_operators=len(operators), measured_operators=len(means),
                    expected_case_launches=len(required), mode_paired_case_launches=paired,
                    complete=paired == len(required), operators=operators)
    return dict(boundary_equal=boundary, operator_equal=balanced)


def grouped_boundaries(records, expected):
    grouped, required = defaultdict(list), defaultdict(int)
    for key, row in expected.items():
        group = row['operator'], row['cp'], row['launch'], row['weight_mode']
        required[group] += 1
        if key in records:
            grouped[group].append(records[key])
    for key, count in sorted(required.items()):
        yield dict(operator=key[0], cp=key[1], launch=key[2], weight_mode=key[3]), grouped[key], count


def extract_pure_references(rows):
    """Use forward p50 or the measured B-then-W total, never isolated B+W."""
    references = {}
    for row in rows:
        backward = row['operator'].endswith('backward')
        for launch in ('eager', 'graph'):
            prefix = f'{launch}_cublas'
            main = row.get(f'{prefix}_p50_ms')
            total = row.get(f'{prefix}_total_p50_ms') if backward else main
            for value in (main, total):
                if value is not None:
                    positive(value, f'{row["id"]}/{prefix}/pure GEMM')
            if total is None:
                continue
            if backward and main is not None and not math.isclose(main, total, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f'pure cuBLAS main p50 differs from actual total: {row["id"]}/{launch}')
            config = row.get(f'{prefix}_config')
            if config is not None:
                if (not isinstance(config, dict) or config.get('dq_included') is not False or
                        config.get('communication_included') is not False):
                    raise ValueError('pure cuBLAS reference must exclude both DQ and communication')
                data = config.get('data', {})
                mnk = row['b_mnk'] if backward else [row['m'], row['n'], row['k']]
                if (not isinstance(data, dict) or data.get('mnk') != mnk or
                        data.get('output_dtype') != 'torch.bfloat16'):
                    raise ValueError('pure cuBLAS data GEMM differs from the matched boundary')
                if backward:
                    weight = config.get('weight', {})
                    if (not isinstance(weight, dict) or weight.get('mnk') != row['w_mnk'] or weight.get('beta') != row['beta'] or
                            weight.get('output_dtype') != 'torch.float32'):
                        raise ValueError('pure cuBLAS W GEMM needs matching MNK, FP32 main_grad and beta')
            references[(row['id'], row['weight_mode'], launch)] = total
    return references


def reference_ratios(base_over_pure, fuse_over_pure):
    return dict(base_over_pure_geomean=base_over_pure,
                pure_over_base_geomean_percent=positive(100 / base_over_pure, 'pure/base percent')
                if base_over_pure is not None else None,
                fuse_over_pure_geomean=fuse_over_pure)


def reference_counts(rows, expected):
    return dict(measured=len(rows), expected=expected, missing_matched_reference_boundaries=expected - len(rows),
                targets_exceeding_point_reference={str(target): sum(target > row['base_over_pure'] for row in rows)
                                                   for target in REFERENCE_TARGETS},
                fuse_faster_than_pure=sum(row['fuse_over_pure'] < 1 for row in rows))


def pure_reference_summary(rows, records, expected):
    """Hypothetical free DQ/A2A relative to measured GEMMs, not a hard ceiling.

    All three timings must exist for a matched boundary. Missing references do
    not alter the separate Fuse-vs-external milestone gate. Backward mode
    pairing and equal-operator weighting reuse the original aggregation.
    """
    references = extract_pure_references(rows)
    matched = {key: dict(row, base_over_pure=positive(row['baseline_ms'] / references[key], 'base/pure'),
                        fuse_over_pure=positive(row['fuse_ms'] / references[key], 'Fuse/pure'))
               for key, row in records.items() if key in references}
    scopes = {}
    for name, long_only in (('all', False), ('long', True)):
        boundaries = expected_boundaries(expected, long_only)
        subset = {key: row for key, row in matched.items() if key in boundaries}
        metrics = {metric: aggregate({key: dict(row, speedup=row[metric]) for key, row in subset.items()}, boundaries)
                   for metric in ('base_over_pure', 'fuse_over_pure')}
        totals = {}
        for weighting in ('boundary_equal', 'operator_equal'):
            base, fuse = (metrics[metric][weighting] for metric in ('base_over_pure', 'fuse_over_pure'))
            totals[weighting] = reference_ratios(base['geomean'], fuse['geomean'])
            if weighting == 'boundary_equal':
                totals[weighting].update(reference_counts(list(subset.values()), len(boundaries)))
            else:
                totals[weighting].update({key: value for key, value in base.items()
                                          if key not in ('geomean', 'operators')})
                totals[weighting]['operators'] = {
                    operator: dict(measured=item['measured'], expected=item['expected'],
                                   **reference_ratios(item['geomean'], fuse['operators'][operator]['geomean']))
                    for operator, item in base['operators'].items()}
        groups = [dict(group, **reference_counts(members, count),
                       **reference_ratios(geometric_mean([row['base_over_pure'] for row in members]),
                                          geometric_mean([row['fuse_over_pure'] for row in members])))
                  for group, members, count in grouped_boundaries(subset, boundaries)]
        scopes[name] = dict(aggregate=totals, groups=groups)
    count = len(expected) * 2
    return dict(reference_kind='measured_pure_cublas_no_DQ_no_A2A', is_hardware_hard_limit=False,
                affects_milestone_gate=False, comparison_targets=list(REFERENCE_TARGETS),
                coverage=dict(expected_boundary_records=count, measured_pure_records=len(references),
                              missing_pure_records=count - len(references), matched_boundary_records=len(matched),
                              missing_matched_reference_records=count - len(matched), complete=len(matched) == count),
                scopes=scopes, definitions=[
                    '实测纯 cuBLAS GEMM 参考≠硬件硬上限；假设通信和 DQ 都免费，不宣称只消除通信后的性能。',
                    'DQ 尚无全量独立分项，不能从此参考推导保留 DQ 的上限；前向用 GEMM p50，反向用实际 B→W total p50，绝不相加 B/W median。',
                    'base/pure 是外部基线相对实测纯 GEMM 的参考倍数；pure/base% 是对应时延比例，按同一几何平均权重计算。',
                    'Fuse/pure 是当前 Fuse 到实测 GEMM 的剩余参考空间；小于 1 表示 Fuse 已更快，原值保留、不截断，也不是物理上限被突破。',
                    '各目标超过逐点参考数统计 target > base/pure；不是该目标不可实现的硬件证明。缺失参考不填补、不影响独立的优化里程碑判定。',
                    '参考表保留 1.4/1.5× 作为假设对照，不是当前硬目标；正式目标仅 1.2× 第一验收、1.3× 冲刺。',
                ])


def formal_gate(metadata, coverage):
    reasons = []
    if not coverage['complete']:
        reasons.append('完整矩阵仍有缺失的 shape/mode/launch 或 Fuse/外部基线时延')
    if metadata is None:
        reasons.append('缺少 comparison_metadata.json，无法确认正式计时口径')
    elif metadata.get('schema') != 'mxfp8-matched-comparison-v1':
        reasons.append('comparison metadata schema 不匹配')
    else:
        if metadata.get('diagnostic') is not False:
            reasons.append('输入是 diagnostic，或未明确标记为非 diagnostic')
        sources = metadata.get('measurement_reports', [])
        if not sources or any(source.get('complete') is not True for source in sources):
            reasons.append('测量来源缺失或尚未 complete')
        if metadata.get('missing_fuse_records') != 0:
            reasons.append('comparison metadata 未确认 Fuse 边界完整')
        if metadata.get('baseline_winners') != coverage['expected_boundary_records']:
            reasons.append('comparison metadata 的基线覆盖数不匹配')
    return not reasons, reasons


def build_summary(rows, metadata=None):
    expected = expected_rows()
    records, coverage = extract_boundaries(rows, expected)
    eligible, reasons = formal_gate(metadata, coverage)
    scopes = {}
    for name, long_only in (('all', False), ('long', True)):
        boundaries = expected_boundaries(expected, long_only)
        groups = [dict(group, **statistics([row['speedup'] for row in members], count),
                       **tail_statistics(members)) for group, members, count in grouped_boundaries(records, boundaries)]
        totals = aggregate(records, boundaries)
        milestones = {weighting: {str(target): bool(eligible and value['geomean'] is not None and
                                                    value['geomean'] >= target) for target in TARGETS}
                      for weighting, value in totals.items()}
        scopes[name] = dict(groups=groups, aggregate=totals, milestones=milestones)
    return dict(schema='mxfp8-optimization-summary-v1', coverage=coverage,
                milestone_eligible=eligible, ineligible_reasons=reasons,
                milestone_targets=list(TARGETS),
                long_global_tokens=list(LONG_TOKENS), scopes=scopes,
                pure_gemm_reference=pure_reference_summary(rows, records, expected),
                definitions=[
                    '加速比由匹配的 best_external_p50_ms / Fuse_p50_ms 重算；只统计 F 和实际 backward total。',
                    'boundary_equal：每个 shape×launch×mode 的完整边界等权，因此两个反向算子各有两种 mode。',
                    'operator_equal：反向两个 mode 先在同一 shape×launch 内几何平均，再逐算子几何平均，最后四算子等权。',
                    '全部和长序列分别报告；统计中的缺项不填零，不借其他 launch/mode 补齐。',
                    '里程碑按 scope×weighting 分别判定，不表示所有点均达到该倍数；完整正式矩阵不足时绝不标 achieved。',
                    '当前目标为 1.2× 第一验收、1.3× 冲刺；1.4/1.5× 不再作为硬指标。',
                    '此摘要继承历史外部基线的比较边界，不宣称同轮量化-only A/B，也不使用纯 cuBLAS 代替通信基线。',
                    'Fuse p95/p50 中位/最高值只统计提供了有效 p95 的边界，不要求补测；微小收益需要重复配对 A/B，不能靠一轮最小差值判优。',
                ])


def number(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}×'


def percentage(value):
    return '—' if value is None else f'{value:.2f}%'


def pure_reference_markdown(reference):
    targets = reference.get('comparison_targets', REFERENCE_TARGETS)
    lines = ['', '## 无通信且 DQ 免费：实测纯 GEMM 参考', '']
    lines += [definition + '\n' for definition in reference['definitions']]
    coverage = reference['coverage']
    lines += [f"纯 GEMM 已测 {coverage['measured_pure_records']}/{coverage['expected_boundary_records']}，"
              f"缺参考 {coverage['missing_pure_records']}；与基线/Fuse 完整匹配 "
              f"{coverage['matched_boundary_records']}/{coverage['expected_boundary_records']}。"
              + ('参考覆盖完整。' if coverage['complete'] else '参考覆盖不完整，不代表全量上限。'), '',
              '| 范围 | 权重 | base/pure 参考倍数 | pure/base 时延比例 | Fuse/pure 剩余参考空间 |',
              '|---|---|---:|---:|---:|']
    for scope, item in reference['scopes'].items():
        for weighting, row in item['aggregate'].items():
            lines.append(f"| {scope} | {weighting} | {number(row['base_over_pure_geomean'])} | "
                         f"{percentage(row['pure_over_base_geomean_percent'])} | {number(row['fuse_over_pure_geomean'])} |")
    lines += ['', '| 范围 | 已匹配 / 应匹配边界 | ' +
              ' | '.join(f'{target}× 超过逐点参考' for target in targets) + ' | Fuse 已快于 pure |',
              '|---|---:|' + '---:|' * (len(targets) + 1)]
    for scope, item in reference['scopes'].items():
        row = item['aggregate']['boundary_equal']
        counts = ' | '.join(str(row['targets_exceeding_point_reference'][str(target)]) for target in targets)
        lines.append(f"| {scope} | {row['measured']}/{row['expected']} | {counts} | {row['fuse_faster_than_pure']} |")
    for scope, item in reference['scopes'].items():
        lines += ['', f'### {scope}：纯 GEMM 参考分组', '',
                  '| 算子 | CP | 启动 | 模式 | 匹配 / 应有 | base/pure | pure/base | Fuse/pure | 假设超过参考点数 ' +
                  '/'.join(map(str, targets)) + '× | Fuse 快于 pure |',
                  '|---|---:|---|---|---:|---:|---:|---:|---|---:|']
        for row in item['groups']:
            counts = '/'.join(str(row['targets_exceeding_point_reference'][str(target)]) for target in targets)
            lines.append(f"| {row['operator']} | {row['cp']} | {row['launch']} | {row['weight_mode'] or '—'} | "
                         f"{row['measured']}/{row['expected']} | {number(row['base_over_pure_geomean'])} | "
                         f"{percentage(row['pure_over_base_geomean_percent'])} | {number(row['fuse_over_pure_geomean'])} | "
                         f"{counts} | {row['fuse_faster_than_pure']} |")
    return lines


def markdown(summary):
    lines = ['# SM90 MXFP8 四算子优化汇总', '', '## 口径', '']
    lines += [definition + '\n' for definition in summary['definitions']]
    coverage = summary['coverage']
    lines += [f"覆盖：{coverage['observed_settings']}/{coverage['expected_settings']} settings，"
              f"{coverage['observed_wide_rows']}/{coverage['expected_wide_rows']} 宽表行，"
              f"{coverage['measured_boundary_records']}/{coverage['expected_boundary_records']} 完整边界。", '']
    if not summary['milestone_eligible']:
        lines += ['里程碑不可验收：' + '；'.join(summary['ineligible_reasons']) + '。', '']
    lines += ['## 汇总', '',
              '| 范围 | 已测 / 应测边界 | 全边界等权几何平均 | 四算子等权几何平均 | 获胜边界 | 最低 / 最高 | Fuse p95/p50 中位 / 最高 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for scope, item in summary['scopes'].items():
        boundary, balanced = item['aggregate']['boundary_equal'], item['aggregate']['operator_equal']
        lines.append(f"| {scope} | {boundary['measured']}/{boundary['expected']} | {number(boundary['geomean'])} | "
                     f"{number(balanced['geomean'])} | {boundary['wins']}/{boundary['measured']} | "
                     f"{number(boundary['minimum'])} / {number(boundary['maximum'])} | "
                     f"{number(boundary['p95_p50_median'], 4)} / {number(boundary['p95_p50_maximum'], 4)} |")
    targets = summary.get('milestone_targets', list(next(iter(summary['scopes'].values()))['milestones']['boundary_equal']))
    lines += ['', '## 里程碑', '', '| 范围 | 权重 | ' + ' | '.join(f'{target}×' for target in targets) + ' |',
              '|---|---|' + '---|' * len(targets)]
    for scope, item in summary['scopes'].items():
        for weighting, milestones in item['milestones'].items():
            labels = [('已达到' if milestones[str(target)] else '未达到') if summary['milestone_eligible'] else '不可验收'
                      for target in targets]
            lines.append(f'| {scope} | {weighting} | ' + ' | '.join(labels) + ' |')
    for scope, item in summary['scopes'].items():
        lines += ['', f'## {scope}：算子 / CP / 启动 / 模式', '',
                  '| 算子 | CP | 启动 | 模式 | 已测 / 应测 | 几何平均 | 获胜点 | 最低 | 最高 | Fuse p95/p50 中位 / 最高 |',
                  '|---|---:|---|---|---:|---:|---:|---:|---:|---:|']
        for row in item['groups']:
            lines.append(f"| {row['operator']} | {row['cp']} | {row['launch']} | {row['weight_mode'] or '—'} | "
                         f"{row['measured']}/{row['expected']} | {number(row['geomean'])} | "
                         f"{row['wins']}/{row['measured']} | {number(row['minimum'])} | {number(row['maximum'])} | "
                         f"{number(row['p95_p50_median'], 4)} / {number(row['p95_p50_maximum'], 4)} |")
    if summary.get('pure_gemm_reference') is not None:
        lines += pure_reference_markdown(summary['pure_gemm_reference'])
    return '\n'.join(lines) + '\n'


def write_summary(summary, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / 'optimization_summary.json').write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    (output / 'optimization_summary.md').write_text(markdown(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path,
                        default=ROOT / 'results/mxfp8_weight/comparison/comparison_summary.json')
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    metadata_path = args.metadata or args.comparison.with_name('comparison_metadata.json')
    rows = json.loads(args.comparison.read_text())
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
    summary = build_summary(rows, metadata)
    paths = [args.comparison] + ([metadata_path] if metadata is not None else [])
    summary['input_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    output = args.output or args.comparison.parent
    write_summary(summary, output)
    print(f"Wrote compact optimization summary to {output}; milestone_eligible={summary['milestone_eligible']}")


if __name__ == '__main__':
    main()
