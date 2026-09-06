#!/usr/bin/env python3
"""Strict, compact reporting of frozen forward policy A/B (never 2F2B totals).

Read-only inputs retain every registry row, including repeated physical shapes
and unchanged/fallback configurations. Three arm p50s are geometrically averaged;
--balance-orders combines six (three per order). These are not pooled-sample
medians. No timing-based policy selection is done.
"""

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from comparison_report import ROOT, cell, mnk_label, number, positive, seq_label
from matrix import full_matrix
from operator_policy_ab import ARMS, ITERATIONS, ROUNDS, WARMUP, arm_sequence, paired_summary
from operator_sweep import validate_config


LAUNCHES = ('eager', 'graph')
LONG_TOKENS = 128 * 1024
SCOPE = 'OProj_F_only_not_2F2B_total'
QKV_SCOPE = 'QKV_F_only_not_2F2B_total'
KERNEL_KEYS = ('comm_ctas', 'tile_m', 'tile_n', 'tile_k', 'cluster_m', 'policy_enum')
SEMANTIC = 'offline_mxfp8_weight_runtime_dq_bf16_tensor_core_gemm'
CONTRACTS = {
    'oproj_forward': dict(input_schema='mxfp8-frozen-policy-ab-v1',
                          summary_schema='mxfp8-frozen-policy-summary-v1',
                          scope=SCOPE, model_key='oproj_comm_model', direction='a2a_gemm', label='OProj F'),
    'qkv_forward': dict(input_schema='mxfp8-qkv-forward-frozen-policy-ab-v1',
                        summary_schema='mxfp8-qkv-forward-frozen-policy-summary-v1',
                        scope=QKV_SCOPE, model_key='qkv_forward_comm_model', direction='gemm_a2a', label='QKV F'),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal_number(recorded, expected, label):
    positive(recorded, label)
    require(math.isclose(recorded, expected, rel_tol=1e-10, abs_tol=0), f'{label}: recorded value mismatch')


def geometric(values):
    return math.exp(statistics.mean(map(math.log, values))) if values else None


def report_operator(documents):
    operators = {operator for _, document in documents for operator, contract in CONTRACTS.items()
                 if document.get('schema') == contract['input_schema']}
    require(len(operators) == 1, 'Inputs must contain one supported forward operator/schema, never a mixed scope')
    return operators.pop()


def registry_for(operator):
    return {case['id']: case for case in full_matrix()
            if case['direction'] == CONTRACTS[operator]['direction'] and
            (operator != 'qkv_forward' or case['cp'] == 4 and case['global_seq'] >= LONG_TOKENS)}


def check_correctness(checks, operator='oproj_forward'):
    routed = 'local_gemm' if operator == 'qkv_forward' else 'route'
    require(isinstance(checks, dict) and set(checks) == {'weight_dequant', routed, 'output'},
            f'Missing forward output/{routed}/dequant correctness')
    for name, metrics in checks.items():
        require(isinstance(metrics, dict), f'{name}: correctness metrics must be an object')
        require(metrics.get('all_ranks_finite') is True, f'{name}: nonfinite correctness')
        for field in ('max_abs', 'relative_rmse'):
            value = metrics[field]
            require(type(value) in (float, int) and math.isfinite(value) and value >= 0,
                    f'{name}: invalid {field}')
        require(metrics['relative_rmse'] <= .005, f'{name}: BF16 correctness tolerance exceeded')
        if name in ('weight_dequant', 'route'):
            require(metrics['max_abs'] == 0 and metrics['relative_rmse'] == 0,
                    f'{name}: exact correctness required')


def check_timing(timing, case):
    ranks, samples = timing['rank_samples_us'], timing['samples_us']
    require(len(ranks) == case['cp'] and all(len(row) == ITERATIONS for row in ranks),
            'Per-rank vectors must be CP x 50')
    require(len(samples) == ITERATIONS, 'Exactly 50 rank-max samples required')
    for row in ranks:
        for value in row:
            positive(value, 'rank sample')
    for value in samples:
        positive(value, 'rank-max sample')
    require(samples == [max(row[index] for row in ranks) for index in range(ITERATIONS)],
            'Samples are not sample-wise rank maxima')
    ordered = sorted(samples)
    position = .95 * (len(ordered) - 1)
    low = math.floor(position)
    p95 = ordered[low] + (ordered[low + 1] - ordered[low]) * (position - low)
    median = statistics.median(samples)
    for field, expected in (('p50_us', median), ('p95_us', p95), ('mean_us', statistics.mean(samples))):
        equal_number(timing[field], expected, field)
    flops = 2 * case['m'] * case['n'] * case['k']
    require(type(timing['flops_per_gpu']) is int and timing['flops_per_gpu'] == flops, 'Incorrect GEMM FLOPs')
    equal_number(timing['tflops_per_gpu'], flops / median / 1e6, 'TFLOPS/GPU')


def check_config(config, arm, model, case, operator='oproj_forward'):
    validate_config(0, config)
    for key in (*KERNEL_KEYS, 'sm_count'):
        require(type(config[key]) is int, f'Invalid native {key}')
        valid = config[key] == 0 if operator == 'qkv_forward' and key == 'policy_enum' else config[key] > 0
        require(valid, f'Invalid native {key}')
    for key in ('requested_comm_ctas', 'weight_block_size', 'swizzle', 'epoch'):
        require(type(config[key]) is int, f'Native {key} must be an integer')
    expected = dict(weight_block_size=32, payload='e4m3', scale='e8m0', weight_axis='original_forward_K',
                    gemm_input='bf16', gemm_accumulator='fp32', requested_comm_ctas=0,
                    sm_count=model['sm_count'], raster='n', swizzle=1, gemm_policy_request='auto',
                    route_layout=case['layout'], forward_or_data_mnk=[case['m'], case['n'], case['k']],
                    weight_mnk=None, alpha=1, epoch=1)
    require(all(config.get(key) == value for key, value in expected.items()), 'Config precision/shape/arm contract')
    expected_dispatch = ('existing_bf16_production_auto' if arm == 'baseline'
                         else 'calibrated_model_or_domain_fallback')
    require(config['dispatch'] == expected_dispatch, 'Arm dispatch label mismatch')
    metadata_key = CONTRACTS[operator]['model_key']
    require((metadata_key not in config if arm == 'baseline' else config.get(metadata_key) == model),
            'Baseline must disable model; model arm must retain the frozen calibration')
    if operator == 'qkv_forward':
        require('oproj_comm_model' not in config, 'QKV arm must not carry an OProj model')
        # QKV's test bridge reports zero as a placeholder, not an OProj enum.
        # Validate its actual BF16 tile traits, retaining unsupported-model
        # families as genuine fallback records rather than dropping them.
        tile_n = config['tile_n']
        require(tile_n in (64, 128, 160, 192, 256, 320) and
                (config['tile_m'], config['tile_k'], config['cluster_m']) ==
                (128, 64, 2 if tile_n >= 256 else 1), 'Native QKV tile/cluster mismatch')
        return
    tiles = {1: (64, 128, 64, 1), 2: (128, 128, 64, 1), 3: (128, 160, 64, 1),
             4: (128, 256, 64, 2), 5: (128, 320, 64, 2)}
    require(tuple(config[key] for key in ('tile_m', 'tile_n', 'tile_k', 'cluster_m')) ==
            tiles.get(config['policy_enum']), 'Native policy enum/tile mismatch')


def launch_summary(record, case, model, order=ARMS, explicit_order=False, operator='oproj_forward'):
    require(record.get('launch') in LAUNCHES and record.get('phase') == 'forward' and
            record.get('scope') == CONTRACTS[operator]['scope'],
            f'Only {CONTRACTS[operator]["label"]} eager/graph records are supported')
    require(record.get('warmup') == WARMUP and record.get('iterations') == ITERATIONS, 'Formal 10+50 required')
    configs, rounds = record['configs'], record['rounds']
    # Reports written before --arm-order only used baseline -> model. New
    # reports must carry explicit capture and per-pair execution sequences.
    expected_order = list(order)
    fallback = None if explicit_order else expected_order
    require(record.get('capture_order', fallback) == expected_order, 'Capture order differs from --arm-order')
    require(set(configs) == set(ARMS) and set(record['correctness_after_capture']) == set(ARMS), 'Missing A/B arm')
    for arm in ARMS:
        check_config(configs[arm], arm, model, case, operator)
        check_correctness(record['correctness_after_capture'][arm], operator)
    changed = any(configs['baseline'][key] != configs['model'][key] for key in KERNEL_KEYS)
    require(type(record['kernel_configuration_changed']) is bool and record['kernel_configuration_changed'] == changed,
            'Recorded changed/fallback classification is incorrect')
    require(all(type(row['round']) is int for row in rounds) and
            [row['round'] for row in rounds] == list(range(1, ROUNDS + 1)), 'Three ordered A/B rounds required')
    for pair in rounds:
        require(pair.get('execution_order', fallback) == expected_order, 'Pair execution order differs from --arm-order')
        for arm in ARMS:
            value = pair[arm]
            require(value.get('arm') == arm and value.get('config') == configs[arm], 'Arm/config changed between rounds')
            check_timing(value['timing'], case)
            if pair['round'] == ROUNDS or 'correctness_after_formal_samples' in value:
                check_correctness(value['correctness_after_formal_samples'], operator)
        equal_number(pair['speedup'], pair['baseline']['timing']['p50_us'] / pair['model']['timing']['p50_us'],
                     'paired speedup')
    summary = paired_summary(rounds)
    recorded = record['paired_summary']
    require(recorded['definition'] == summary['definition'] and len(recorded['paired_speedups']) == ROUNDS,
            'Paired summary definition/count mismatch')
    for actual, expected in zip(recorded['paired_speedups'], summary['paired_speedups']):
        equal_number(actual, expected, 'paired summary ratio')
    for field in ('geometric_mean_speedup', 'min_speedup', 'max_speedup'):
        equal_number(recorded[field], summary[field], field)
    result = dict(speedup=summary['geometric_mean_speedup'], paired_speedups=summary['paired_speedups'], changed=changed)
    for arm in ARMS:
        result[arm + '_p50_geomean_ms'] = geometric([pair[arm]['timing']['p50_us'] for pair in rounds]) / 1000
        result[arm + '_config'] = {key: configs[arm][key] for key in (*KERNEL_KEYS, 'sm_count', 'raster', 'swizzle')}
    equal_number(result['speedup'], result['baseline_p50_geomean_ms'] / result['model_p50_geomean_ms'],
                 'Ratio of geometric arm p50s')
    return result


def check_qkv_model(model):
    domain = dict(min_m=32768, max_m=131072, min_n=4096, max_n=18432,
                  min_k=2048, max_k=16384, m_multiple=256, n_multiple=256,
                  k_multiple=64, head_dim=128, tile_families=[[128, 256, 64, 2]],
                  comm_ctas=list(range(4, 25, 2)))
    actual = model.get('domain')
    require(model.get('operator') == 'qkv_forward' and actual == domain, 'Frozen QKV domain/operator mismatch')
    require(all(type(actual[key]) is int for key in domain if key not in ('tile_families', 'comm_ctas')) and
            all(type(value) is int for value in actual['comm_ctas']) and
            all(type(value) is int for tile in actual['tile_families'] for value in tile),
            'Frozen QKV physical-domain values must be integers')
    coefficients = model.get('coefficients')
    require(isinstance(coefficients, dict) and set(coefficients) == {'compute_gflop_sm_us', 'route_slot_task_us'},
            'Frozen QKV service coefficients required')
    for value in coefficients.values():
        positive(value, 'Frozen QKV service coefficient')
    margin = model.get('minimum_gain')
    require(type(margin) in (int, float) and math.isfinite(margin) and 0 <= margin < 1,
            'Invalid frozen QKV margin')
    require(model.get('auto_reference') == 'explicit_calibrated_model_or_domain_fallback_not_legacy_auto' and
            model.get('scope') == 'qkv_forward_only_manual_comm_ctas_take_precedence',
            'Frozen QKV model boundary mismatch')


def provenance(source, document, cp, operator='oproj_forward'):
    hashes = document['sources']
    require(isinstance(hashes, dict) and hashes and all(isinstance(key, str) and
            isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) for key, value in hashes.items()),
            'Complete recorded source SHA256 map required')
    model = document[CONTRACTS[operator]['model_key']]
    qkv = operator == 'qkv_forward'
    schema = 'mxfp8-qkv-forward-service-model-v1' if qkv else 'mxfp8-oproj-service-model-v1'
    require(model.get('schema') == schema and model.get('world_size') == cp and
            model.get('sm_count') == 132 and hashes.get(model['source']) == model['sha256'],
            'Frozen model CP/SM/source fingerprint mismatch')
    if qkv:
        require(cp == 4 and type(model['world_size']) is int and type(model['sm_count']) is int,
                'Frozen QKV calibration is CP4/SM132 only')
        check_qkv_model(model)
    else:
        for key in ('a', 'e', 'b', 't'):
            value = model['coefficients'][key]
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0 and (key == 'e' or value > 0),
                    'Invalid frozen service coefficients')
        require(type(model['minimum_gain']) in (int, float) and 0 <= model['minimum_gain'] < 1 and
                type(model['launch_prior_us']) in (int, float) and math.isfinite(model['launch_prior_us']) and
                model['launch_prior_us'] >= 0, 'Invalid frozen margin/prior')
        require(model['tile_families'] == [[128, 256, 64, 2], [128, 320, 64, 2]], 'Frozen tile domain mismatch')
    library = Path(document['args']['library'])
    key = str(library.relative_to(ROOT)) if library.is_relative_to(ROOT) else str(library)
    require(key in hashes and 'benchmarks/mxfp8_weight/operator_policy_ab.py' in hashes,
            'Missing native library or A/B runner fingerprint')
    devices = document['devices']
    require(len(devices) == cp and sorted(device['rank'] for device in devices) == list(range(cp)) and
            all(device['cc'] == [9, 0] and device['sm_count'] == 132 for device in devices), 'Device/CP domain mismatch')
    require(isinstance(document['gpu_clock_snapshot'], str) and document['gpu_clock_snapshot'], 'Missing clock snapshot')
    return dict(source=source, recorded_sources=copy.deepcopy(hashes), model=copy.deepcopy(model), library_source=key,
                arm_order=document['args'].get('arm_order', 'baseline-model'),
                devices=copy.deepcopy(devices), gpu_clock_snapshot=document['gpu_clock_snapshot'],
                policy_environment=copy.deepcopy(document['policy_environment']),
                torch=document['torch'], cuda=document['cuda'])


def statistics_for(points):
    ratios = [point['speedup'] for point in points]
    worst = min(points, key=lambda point: point['speedup']) if points else None
    return dict(records=len(points), geometric_mean_speedup=geometric(ratios),
                wins=sum(value > 1 for value in ratios), losses=sum(value < 1 for value in ratios),
                ties=sum(value == 1 for value in ratios),
                minimum_speedup=min(ratios) if ratios else None, maximum_speedup=max(ratios) if ratios else None,
                maximum_latency_regression_percent=max(0., (1 / min(ratios) - 1) * 100) if ratios else None,
                latency_regressions_over_3pct=sum(1 / value > 1.03 for value in ratios),
                worst_record=copy.deepcopy(worst))


def summarize(rows):
    points = [dict(id=row['id'], cp=row['cp'], launch=launch, global_seq=row['global_seq'],
                   changed=row[launch + '_changed'], speedup=row[launch + '_speedup'])
              for row in rows for launch in LAUNCHES if row.get(launch + '_speedup') is not None]
    scopes = {}
    for scope in ('all', 'long'):
        selected = [point for point in points if scope == 'all' or point['global_seq'] >= LONG_TOKENS]
        result = dict(aggregate=statistics_for(selected))
        for field, values in (('cp', sorted({point['cp'] for point in points})), ('launch', LAUNCHES),
                              ('changed', (False, True))):
            result['by_' + field] = [dict({field: value}, **statistics_for([p for p in selected if p[field] == value]))
                                    for value in values]
        result['groups'] = [dict(cp=cp, launch=launch, change=change,
                                 **statistics_for([p for p in selected if p['cp'] == cp and p['launch'] == launch and
                                                   (change == 'all' or p['changed'] == (change == 'changed'))]))
                            for cp in sorted({point['cp'] for point in points}) for launch in LAUNCHES
                            for change in ('all', 'changed', 'unchanged')]
        scopes[scope] = result
    return scopes


def build_report(documents, require_full=False, balance_orders=False):
    documents = list(documents)
    if balance_orders:
        return build_balanced_report(documents, require_full)
    operator = report_operator(documents)
    contract = CONTRACTS[operator]
    registry = registry_for(operator)
    rows, inputs, seen = [], [], set()
    for source, document in documents:
        require(document.get('schema') == contract['input_schema'] and document.get('complete') is True,
                f'{source}: complete frozen A/B report required')
        require(document.get('scope') == contract['scope'] and document.get('formal_ab_protocol') is True and
                document.get('production_policy_written') is False and document.get('milestone_evidence') is False,
                f'Formal {contract["label"]}-only/no-milestone boundary required')
        require(not document.get('diagnostic_trace', False) and
                not document.get('args', {}).get('diagnostic_trace', False),
                'Externally traced diagnostic timings cannot enter a formal report')
        require(document.get('semantic') == SEMANTIC and document.get('token_semantic') ==
                'T_is_flattened_tokens_no_training_batch', 'Weight/token semantic mismatch')
        require(document.get('selection') == 'one_explicit_frozen_model_no_search_no_winner_selection' and
                document['arms'] == dict(baseline='manual_CTA_zero_native_model_disabled_unchanged_auto',
                                        model='manual_CTA_zero_explicit_model_native_selector_or_domain_fallback'),
                'Unchanged-auto/frozen-model arm contract required')
        args, cases = document['args'], document['cases']
        order = arm_sequence(args.get('arm_order', 'baseline-model'))
        explicit_order = 'arm_order' in args
        require(operator != 'qkv_forward' or explicit_order, 'QKV reports require an explicit arm order')
        timing = document['timing']
        require(all(timing.get(key) == value for key, value in dict(rounds=ROUNDS, warmup_per_arm=WARMUP,
                    samples_per_arm=ITERATIONS, sample_statistic='sample_wise_max_across_ranks',
                    clocks='CUDA_events_one_operation_per_sample',
                    ab_order='_'.join(order) + '_repeated_three_times_per_case_and_launch').items()), 'Formal timing protocol mismatch')
        require(timing.get('capture_order', None if explicit_order else list(order)) == list(order),
                'Declared capture order differs from --arm-order')
        cp, launches = args['cp'], args['launches'].split(',')
        require(type(cp) is int and cp in ((4,) if operator == 'qkv_forward' else (4, 8)) and
                args['operators'] == operator and args['backends'] == 'fuse',
                f'Unsupported CP/operator/backend for {contract["label"]}')
        require(launches and len(set(launches)) == len(launches) and set(launches) <= set(LAUNCHES), 'Invalid launch manifest')
        require(cases and [item['case']['id'] for item in cases] == document['selected_case_ids'], 'Incomplete/changed case manifest')
        inputs.append(provenance(source, document, cp, operator))
        model = document[contract['model_key']]
        for item in cases:
            case, records = item['case'], item['records']
            require(case == registry.get(case['id']) and case['cp'] == cp, 'Case differs from original registry')
            require(case['id'] not in seen, 'Duplicate registry case across inputs')
            seen.add(case['id'])
            require(item['operator'] == operator and [record['launch'] for record in records] == launches,
                    'Missing/duplicate/mismatched launch record')
            row = {key: case[key] for key in ('id', 'model', 'global_seq', 'cp', 'm', 'n', 'k')}
            row.update(source=source, operator=operator, arm_order='-'.join(order))
            for launch in LAUNCHES:
                row.update({launch + '_' + key: None for key in
                            ('baseline_p50_geomean_ms', 'model_p50_geomean_ms', 'speedup',
                             'baseline_config', 'model_config', 'paired_speedups', 'changed')})
            first_configs = records[0]['configs']
            for record in records:
                require(record['configs'] == first_configs, 'Native configuration changed across launch modes')
                measured = launch_summary(record, case, model, order, explicit_order, operator)
                row.update({record['launch'] + '_' + key: value for key, value in measured.items()})
            rows.append(row)
    require(rows, 'No complete A/B records')
    rows.sort(key=lambda row: list(registry).index(row['id']))
    coverage = []
    for cp in sorted({row['cp'] for row in rows}):
        expected = {key for key, case in registry.items() if case['cp'] == cp}
        members = [row for row in rows if row['cp'] == cp]
        records = sum(row.get(launch + '_speedup') is not None for row in members for launch in LAUNCHES)
        complete = {row['id'] for row in members} == expected and records == 2 * len(expected)
        require(not require_full or complete, f'CP{cp}: full original registry and Eager/Graph coverage required')
        coverage.append(dict(cp=cp, settings=len(members), expected_settings=len(expected),
                             records=records, expected_records=2 * len(expected), complete=complete))
    result = dict(schema=contract['summary_schema'], scope=contract['scope'], milestone_evidence=False,
                balance_orders=False,
                require_full=require_full, coverage=coverage, inputs=inputs, rows=rows, summary=summarize(rows),
                aggregation='Equal weight per original registry setting x launch; keep repeated geometry and fallback rows',
                latency_statistic='Geometric mean of three arm p50s, not a pooled median; ratio equals paired speedup GM',
                long_tokens_minimum=LONG_TOKENS,
                provenance_note='Input JSON bytes are hashed; recorded source/library/model fingerprints are retained, not rehashed against the current worktree',
                win_loss_note='Win/loss uses speedup >/< 1, not statistical significance; regression percent = (model/baseline - 1)*100')
    if operator == 'qkv_forward':
        result.update(baseline_reference='unchanged_native_auto_not_external_baseline',
                      external_baseline_comparison_included=False)
    return result


def build_balanced_report(documents, require_full=False):
    """Validate each complete run once, then combine equal-weight order pairs.

    Never use measured winners, replace original rows, or pool raw samples.
    Normal build_report() still rejects every duplicate setting across inputs.
    """
    orders = ('baseline-model', 'model-baseline')
    operator = report_operator(documents)
    qkv = operator == 'qkv_forward'
    parts, source_inputs, config_maps = {}, [], {}
    native_reference, runtime_reference = None, None
    for source, document in documents:
        part = build_report([(source, document)], require_full=require_full if qkv else True)
        metadata = part['inputs'][0]
        cp, order = part['coverage'][0]['cp'], metadata['arm_order']
        key = cp, order
        require(key not in parts, f'CP{cp}: duplicate {order} full run; exactly one run per order is required')
        hashes = metadata['recorded_sources']
        native = (hashes if qkv else {path: value for path, value in hashes.items()
                  if Path(path).suffix in ('.cu', '.cuh', '.h') or
                  path in (metadata['library_source'], metadata['model']['source'])})
        require('csrc/operators/ulysses_sm90.cu' in native and
                any(path.endswith('.cuh') for path in native) and any(path.endswith('.h') for path in native),
                'Balanced orders require recorded native .cu/.cuh/.h fingerprints')
        runtime = (metadata['torch'], metadata['cuda'], metadata['policy_environment'])
        if native_reference is None:
            native_reference, runtime_reference = native, runtime
        require(native == native_reference, 'Cannot balance different native/library/model versions or source sets')
        require(runtime == runtime_reference, 'Cannot balance different Torch/CUDA/policy environments')
        parts[key] = part
        config_maps[key] = {item['case']['id']: item['records'][0]['configs'] for item in document['cases']}
        source_inputs.append(metadata)
    cps = sorted({cp for cp, _ in parts})
    require(cps, 'No complete order pairs')
    require(set(parts) == {(cp, order) for cp in cps for order in orders},
            'Each CP requires exactly one complete baseline-model and one complete model-baseline report')
    rows, coverage = [], []
    order_fields = ['source'] + [launch + '_' + field for launch in LAUNCHES for field in
                                 ('baseline_p50_geomean_ms', 'model_p50_geomean_ms', 'speedup', 'paired_speedups')]
    for cp in cps:
        first, second = (parts[cp, order] for order in orders)
        require(first['inputs'][0]['model'] == second['inputs'][0]['model'], 'Frozen model differs between orders')
        require(first['inputs'][0]['devices'] == second['inputs'][0]['devices'], 'GPU/card group differs between orders')
        points = [{(row['id'], launch) for row in part['rows'] for launch in LAUNCHES
                   if row.get(launch + '_speedup') is not None} for part in (first, second)]
        require(points[0] == points[1], 'Balanced orders must cover the same complete setting/launch subset')
        require(config_maps[cp, orders[0]] == config_maps[cp, orders[1]], 'Actual native configuration differs between orders')
        indexed = {order: {row['id']: row for row in parts[cp, order]['rows']} for order in orders}
        for original in first['rows']:
            by_order = {order: indexed[order][original['id']] for order in orders}
            row = copy.deepcopy(original)
            row.update(arm_order='balanced_equal_orders', source=[by_order[order]['source'] for order in orders],
                       by_order={order: {key: measured[key] for key in order_fields}
                                 for order, measured in by_order.items()})
            for launch in LAUNCHES:
                if original.get(launch + '_speedup') is None:
                    continue
                for arm in ARMS:
                    field = launch + '_' + arm + '_p50_geomean_ms'
                    row[field] = geometric([by_order[order][field] for order in orders])
                ratios = [ratio for order in orders for ratio in by_order[order][launch + '_paired_speedups']]
                row[launch + '_paired_speedups'] = ratios
                row[launch + '_speedup'] = geometric(ratios)
                equal_number(row[launch + '_speedup'], row[launch + '_baseline_p50_geomean_ms'] /
                             row[launch + '_model_p50_geomean_ms'], 'Balanced six-pair/arm ratio')
            rows.append(row)
        coverage.append(dict(first['coverage'][0], orders=list(orders), rounds_per_order=ROUNDS,
                             paired_rounds_per_setting_launch=2 * ROUNDS,
                             original_order_records=2 * first['coverage'][0]['records']))
        if qkv:
            coverage[-1]['paired_subset_complete'] = True
    registry = {case['id']: index for index, case in enumerate(full_matrix())}
    rows.sort(key=lambda row: registry[row['id']])
    # Preserve provenance differences instead of treating a runner revision as
    # a native/model revision. All original maps remain attached to their input.
    sources = sorted({path for metadata in source_inputs for path in metadata['recorded_sources']})
    differences = {path: {metadata['source']: metadata['recorded_sources'].get(path) for metadata in source_inputs}
                   for path in sources if len({metadata['recorded_sources'].get(path) for metadata in source_inputs}) > 1}
    require(all(Path(path).suffix == '.py' for path in differences),
            'Only recorded Python runner/helper revision differences may vary between balanced inputs')
    result = {key: copy.deepcopy(value) for key, value in next(iter(parts.values())).items()
              if key not in ('rows', 'inputs', 'summary', 'coverage')}
    result.update(balance_orders=True, require_full=require_full if qkv else True,
                  coverage=coverage, inputs=source_inputs, rows=rows,
                  summary=summarize(rows), summary_by_order={order: summarize([row for cp in cps
                      for row in parts[cp, order]['rows']]) for order in orders},
                  recorded_source_differences=differences,
                  aggregation='Equal weight per registry setting x launch, with equal baseline-model/model-baseline order weight',
                  latency_statistic='Geometric mean of six arm p50s (three per order), not a pooled median; speedup is six paired ratios GM',
                  balance_note='Exactly one full run per CP/order; unchanged configurations and regressions remain; no selection from repeated controls')
    if qkv:
        result['balance_note'] = ('Exactly one complete run per CP4/order with the same long-registry setting/launch '
                                  'subset and identical source hashes; --require-full additionally requires all 24 '
                                  'settings x Eager/Graph; unchanged configurations and regressions remain')
    return result


def add_reference_comparison(report, rows, metadata):
    """Join a complete balanced QKV scope to archived, not newly timed, baselines."""
    from optimization_report import expected_rows, extract_boundaries, formal_gate

    require(report['scope'] == QKV_SCOPE and report.get('balance_orders') is True,
            'Reference comparison requires balanced QKV F, not OProj or a single order')
    registry = registry_for('qkv_forward')
    expected_points = {(identifier, launch) for identifier in registry for launch in LAUNCHES}
    points = {(row['id'], launch) for row in report['rows'] for launch in LAUNCHES
              if row.get(launch + '_speedup') is not None}
    require(points == expected_points and len(report['rows']) == len(registry),
            'Reference comparison requires all CP4 long QKV settings x Eager/Graph')
    records, coverage = extract_boundaries(rows, expected_rows())
    valid, reasons = formal_gate(metadata, coverage)
    require(valid, 'Archived full comparison failed formal/coverage audit: ' + '; '.join(reasons))
    hashes = metadata.get('input_sha256')
    require(isinstance(hashes, dict) and hashes and all(isinstance(path, str) and isinstance(sha, str) and
            re.fullmatch('[0-9a-f]{64}', sha) for path, sha in hashes.items()),
            'Archived comparison metadata needs recorded input SHA256 fingerprints')
    require(all(source['source'] in hashes for source in metadata['measurement_reports']),
            'Archived measurement source is missing its recorded SHA256')
    original = {(identifier, launch): records[identifier, None, launch] for identifier, launch in expected_points}
    threshold = geometric([row['speedup'] for row in original.values()])
    indexed = {(row['id'], row['weight_mode']): row for row in rows}
    result = copy.deepcopy(report)
    boundaries = []
    for row in result['rows']:
        old = indexed[row['id'], None]
        for launch in LAUNCHES:
            reference = original[row['id'], launch]
            external_config = old.get(launch + '_best_external_config')
            require(isinstance(external_config, dict) and external_config,
                    'Archived QKV external winner configuration is missing')
            original_config = old.get(launch + '_config')
            require(isinstance(original_config, dict) and original_config,
                    'Archived QKV Fuse configuration is missing')
            auto, model = (positive(row[launch + '_' + arm + '_p50_geomean_ms'], arm + ' six-p50 GM')
                           for arm in ARMS)
            external, old_fuse = reference['baseline_ms'], reference['fuse_ms']
            by_order = {order: positive(row['by_order'][order][launch + '_baseline_p50_geomean_ms'],
                                       'order auto p50 GM') for order in ('baseline-model', 'model-baseline')}
            joined = dict(original_fuse_p50_ms=old_fuse,
                          original_fuse_config=copy.deepcopy(original_config),
                          archived_external_backend=old[launch + '_best_external_backend'],
                          archived_external_p50_ms=external,
                          archived_external_config=copy.deepcopy(external_config),
                          current_auto_six_p50_geomean_ms=auto,
                          current_model_six_p50_geomean_ms=model,
                          original_external_over_fuse=reference['speedup'],
                          archived_external_over_current_model=external / model,
                          archived_external_over_current_auto=external / auto,
                          original_below_scope_mean=reference['speedup'] < threshold,
                          current_model_reaches_original_scope_mean=external / model >= threshold,
                          paired_speedup_over_current_auto=row[launch + '_speedup'],
                          kernel_configuration_changed=row[launch + '_changed'],
                          auto_latency_drift=dict(current_auto_over_original_fuse=auto / old_fuse,
                              reverse_auto_over_forward_auto=by_order['model-baseline'] / by_order['baseline-model'],
                              auto_p50_geomean_ms_by_order=by_order))
            row[launch + '_reference_comparison'] = joined
            boundaries.append(dict(id=row['id'], model=row['model'], cp=row['cp'],
                                   global_seq=row['global_seq'], launch=launch, **joined))

    def statistics(members):
        return dict(records=len(members),
                    original_external_over_fuse_geomean=geometric([r['original_external_over_fuse'] for r in members]),
                    archived_external_over_current_model_geomean=geometric(
                        [r['archived_external_over_current_model'] for r in members]),
                    archived_external_over_current_auto_geomean=geometric(
                        [r['archived_external_over_current_auto'] for r in members]),
                    paired_speedup_over_current_auto_geomean=geometric(
                        [r['paired_speedup_over_current_auto'] for r in members]),
                    original_below_scope_mean=sum(r['original_below_scope_mean'] for r in members),
                    current_model_reaches_original_scope_mean=sum(r['current_model_reaches_original_scope_mean'] for r in members),
                    original_bad_reaching_scope_mean=sum(r['original_below_scope_mean'] and
                        r['current_model_reaches_original_scope_mean'] for r in members),
                    current_model_below_original_scope_mean=sum(not r['current_model_reaches_original_scope_mean'] for r in members),
                    current_auto_over_original_fuse_geomean=geometric(
                        [r['auto_latency_drift']['current_auto_over_original_fuse'] for r in members]))

    result['reference_comparison'] = dict(
        schema='mxfp8-qkv-forward-archived-reference-join-v1', scope=QKV_SCOPE,
        archived_coverage=coverage, original_scope_mean=threshold,
        threshold_definition='GM of archived external/Fuse ratios over the full CP4 long QKV setting x launch scope; never fitted to current winners',
        summary=statistics(boundaries), by_launch=[dict(launch=launch, **statistics(
            [r for r in boundaries if r['launch'] == launch])) for launch in LAUNCHES],
        boundaries=boundaries, recorded_archive_input_sha256=copy.deepcopy(hashes),
        archived_measurement_reports=copy.deepcopy(metadata['measurement_reports']),
        archive_fingerprint_note='Reference wide JSON and companion metadata bytes are hashed by the CLI; upstream recorded hashes are retained, not rehashed here',
        timing_note='Archived values are their original p50s; current arm latency is the GM of six p50s, not a 50-sample median or pooled median',
        comparison_note='External timings are archived cross-run references, not new paired external measurements; only current auto/model is paired A/B',
        drift_note='Auto latency ratios are raw timing observations, not clock measurements, frequency-normalized latencies, or a correction applied to the results',
        milestone_evidence=False, hardware_upper_bound=False)
    result['external_baseline_comparison_included'] = True
    return result


def reference_markdown(reference):
    mean = reference['original_scope_mean']
    lines = ['', '## 归档外部基线参考（非新实测）', '',
             '本节与前面的 paired auto/model A/B 分开：外部 TEUB/cuBLASLt 沿用归档值，没有重新实测。'
             '原 Fuse/外部为原 p50，本次 auto/model 为正反顺序各三次、共六个 p50 的几何平均，不能混称 50-sample median。', '',
             f'固定筛选线来自原 CP4 QKV 长序列全部边界的 GM={number(mean, 9, "×")}；'
             '“原弱点”指原 external/Fuse 低于这条线，不按本次赢家重选。所有不变和退化点均保留。', '',
             '| 范围 | 边界数 | 原 external/Fuse GM | 归档 external/本次 model GM | 本次 auto/model GM | 原弱点 | 原弱点达到原 GM | 当前低于原 GM |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for group in [dict(launch='all', **reference['summary']), *reference['by_launch']]:
        values = [group['launch'], group['records'], number(group['original_external_over_fuse_geomean'], 4, '×'),
                  number(group['archived_external_over_current_model_geomean'], 4, '×'),
                  number(group['paired_speedup_over_current_auto_geomean'], 4, '×'), group['original_below_scope_mean'],
                  group['original_bad_reaching_scope_mean'], group['current_model_below_original_scope_mean']]
        lines.append('| ' + ' | '.join(map(cell, values)) + ' |')
    lines += ['', 'auto 漂移列依次为“本次 auto/原 Fuse”和“反序 auto/正序 auto”的原始时延比，'
              '不是时钟测量，不用它修正任何时延或加速比。归档 external 的完整 winner 配置保留在 JSON/CSV。'
              '这些跨批次参考不构成四算子 milestone 或硬件上限。', '',
              '| 模型 | S | 启动 | 原 Fuse ms | 归档外部 / ms | 本次 auto / model ms | 原 external/Fuse | 归档 external/model | 原弱点 / 达到原 GM | auto 漂移比 |',
              '|---|---:|---|---:|---|---:|---:|---:|---|---|']
    for row in reference['boundaries']:
        values = [row['model'], seq_label(row['global_seq']), row['launch'],
                  number(row['original_fuse_p50_ms']),
                  row['archived_external_backend'] + ' / ' + number(row['archived_external_p50_ms']),
                  number(row['current_auto_six_p50_geomean_ms']) + ' / ' + number(row['current_model_six_p50_geomean_ms']),
                  number(row['original_external_over_fuse'], 4, '×'),
                  number(row['archived_external_over_current_model'], 4, '×'),
                  ('是' if row['original_below_scope_mean'] else '否') + ' / ' +
                  ('是' if row['current_model_reaches_original_scope_mean'] else '否'),
                  number(row['auto_latency_drift']['current_auto_over_original_fuse'], 4, '×') + ' / ' +
                  number(row['auto_latency_drift']['reverse_auto_over_forward_auto'], 4, '×')]
        lines.append('| ' + ' | '.join(map(cell, values)) + ' |')
    return lines


def markdown(report):
    label = 'QKV F' if report['scope'] == QKV_SCOPE else 'OProj F'
    latency = ('正序和反序各 3 轮、各 10 warmup + 50 sample-wise rank-max，每顺序等权。表中时间是六个 p50 的几何平均，不是 pooled median；加速是六个配对比值的几何平均。'
               if report.get('balance_orders') else
               '每臂 3 轮，各 10 warmup + 50 sample-wise rank-max。表中时间是三个 p50 的几何平均，不是 pooled median；加速是三轮配对比值的几何平均。')
    lines = [f'# SM90 MXFP8-weight {label} 冻结策略 A/B', '', '## 口径', '',
             '沿用 S、M×N×K、Eager/Graph 并列表及 ms 单位；S 是展平 token 数，M=S/CP。', '',
             '基线是模型关闭、manual CTA=0 的 unchanged auto；另一臂运行显式冻结模型，未切换/回退点也保留。', '',
             '输入执行顺序：' + ', '.join(sorted({row['arm_order'] for row in report['rows']})) +
             '；capture 和每轮计时按声明顺序，比值方向始终是 baseline/model。', '',
             latency, '',
             '汇总对原 registry 的 setting×launch 等权，重复几何不去重；long 指 S≥128K。最大时延退化=(model/baseline−1)×100%，无退化记 0。', '',
             f'均包含软件 MXFP8 权重反量化、BF16 A2A 与 BF16 GEMM；只测 {label}，不构成四算子 2F2B milestone。胜负不表示统计显著性。', '',
             '## 汇总', '', '| 范围 | CP | 启动 | 配置变化 | 点数 | 胜/负/平 | 加速 GM | 最大时延退化 |',
             '|---|---:|---|---|---:|---:|---:|---:|']
    if report['scope'] == QKV_SCOPE:
        position = lines.index('## 汇总')
        lines[position:position] = [
            '本表加速比只相对模型关闭的原生 auto，不是相对 TEUB/cuBLASLt 或纯 GEMM 的外部基线加速；' +
            ('归档外部基线参考单独列于文末。' if 'reference_comparison' in report else '未包含外部基线比较。'), '',
            'QKV 范围为 CP4、原 gemm_a2a registry 的 128K/256K/512K，共 24 settings、48 Eager/Graph 边界；'
            '若只输入 subset，coverage 保留其缺项，双顺序合并不把 subset 冒充全量。', '']
    for scope, summary in report['summary'].items():
        for row in [dict(cp='all', launch='all', change='all', **summary['aggregate']), *summary['groups']]:
            values = [scope, row['cp'], row['launch'], row['change'], row['records'],
                      f"{row['wins']}/{row['losses']}/{row['ties']}", number(row['geometric_mean_speedup'], 3, '×'),
                      number(row['maximum_latency_regression_percent'], 2, '%')]
            lines.append('| ' + ' | '.join(map(cell, values)) + ' |')
    if report.get('balance_orders'):
        lines += ['', '## 原始顺序分项（不选赢家）', '', '| 执行顺序 | 范围 | CP | 启动 | 点数 | 加速 GM | 最大时延退化 |',
                  '|---|---|---:|---|---:|---:|---:|']
        for order, scopes in report['summary_by_order'].items():
            for scope, summary in scopes.items():
                for row in summary['groups']:
                    if row['change'] == 'all':
                        values = [order, scope, row['cp'], row['launch'], row['records'],
                                  number(row['geometric_mean_speedup'], 3, '×'),
                                  number(row['maximum_latency_regression_percent'], 2, '%')]
                        lines.append('| ' + ' | '.join(map(cell, values)) + ' |')
    lines += ['', '## 逐 shape', '',
              '| CP | 模型 | S | GEMM M×N×K | Eager baseline / model ms | Eager 加速 | Graph baseline / model ms | Graph 加速 | 实际配置 baseline → model |',
              '|---:|---|---:|---|---:|---:|---:|---:|---|']
    for row in report['rows']:
        values = [row['cp'], row['model'], seq_label(row['global_seq']), mnk_label([row['m'], row['n'], row['k']])]
        configs = []
        for launch in LAUNCHES:
            values += [' / '.join(number(row.get(launch + '_' + arm + '_p50_geomean_ms')) for arm in ARMS),
                       number(row.get(launch + '_speedup'), 3, '×')]
            labels = []
            for arm in ARMS:
                config = row.get(launch + '_' + arm + '_config')
                labels.append('—' if config is None else
                              f"c{config['comm_ctas']}/M{config['tile_m']}N{config['tile_n']}K{config['tile_k']}C{config['cluster_m']}")
            configs.append(launch + ': ' + ' → '.join(labels))
        values.append('; '.join(configs))
        lines.append('| ' + ' | '.join(map(cell, values)) + ' |')
    if 'reference_comparison' in report:
        lines += reference_markdown(report['reference_comparison'])
    return '\n'.join(lines) + '\n'


def write_report(report, output, overwrite=False):
    output = Path(output)
    paths = [output / ('policy_summary.' + suffix) for suffix in ('json', 'csv', 'md')]
    inputs = {Path(item['source']).resolve() for item in report['inputs']}
    inputs.update(Path(path).resolve() for path in report.get('reference_comparison', {}).get('input_sha256', {}))
    require(not any(path.resolve() in inputs for path in paths),
            'Output must not replace an input report')
    if not overwrite and any(path.exists() for path in paths):
        raise FileExistsError('Refusing to replace policy_summary outputs; pass --overwrite to rebuild')
    output.mkdir(parents=True, exist_ok=True)
    mode = 'w' if overwrite else 'x'
    with paths[0].open(mode) as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    columns = list(dict.fromkeys(key for row in report['rows'] for key in row))
    with paths[1].open(mode) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        for row in report['rows']:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})
    with paths[2].open(mode) as stream:
        stream.write(markdown(report))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', type=Path, nargs='+')
    parser.add_argument('--require-full', action='store_true')
    parser.add_argument('--balance-orders', action='store_true',
                        help='equally combine two complete orders; OProj requires full CP registry, QKV requires matching subsets')
    parser.add_argument('--reference-comparison', type=Path,
                        help='optional archived comparison_summary.json plus sibling comparison_metadata.json; full balanced QKV only')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/mxfp8_weight/policy_ab/summary')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    paths = [path.resolve() for path in args.paths]
    require(len(paths) == len(set(paths)), 'Duplicate input path')
    raw = {str(path): path.read_bytes() for path in paths}
    report = build_report([(path, json.loads(data)) for path, data in raw.items()], args.require_full, args.balance_orders)
    if args.reference_comparison is not None:
        reference = args.reference_comparison.resolve()
        metadata = reference.with_name('comparison_metadata.json')
        archived = {str(path): path.read_bytes() for path in (reference, metadata)}
        report = add_reference_comparison(report, json.loads(archived[str(reference)]), json.loads(archived[str(metadata)]))
        report['reference_comparison']['input_sha256'] = {
            path: hashlib.sha256(data).hexdigest() for path, data in archived.items()}
    report['input_sha256'] = {path: hashlib.sha256(data).hexdigest() for path, data in raw.items()}
    report['reporter_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_report(report, args.output, args.overwrite)
    label = 'QKV F' if report['scope'] == QKV_SCOPE else 'OProj F'
    print(f'Wrote {len(report["rows"])} {label} settings to {args.output}; not 2F2B total')


if __name__ == '__main__':
    main()
