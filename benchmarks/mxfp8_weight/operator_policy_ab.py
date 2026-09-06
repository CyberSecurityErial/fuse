#!/usr/bin/env python3
"""Interleave unchanged forward auto and one frozen explicit model, without search.

One OperatorCase owns allocation, IPC and references for both arms and launch
modes. Each arm is captured once; all three formal pairs include 10 fresh
warmups and 50 sample-wise rank-max timings. Unchanged/fallback cases remain
in the matrix. Measure QKV F or OProj F separately, never a 2F2B milestone.
"""

import argparse
import copy
import ctypes as ct
from datetime import timedelta
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import operator_bench as bench
from operator_sweep import positive_ratio, validate_config


ARMS = ('baseline', 'model')
ROUNDS, WARMUP, ITERATIONS = 3, 10, 50


def arm_sequence(value):
    """Execution order is independent of the baseline/model ratio direction."""
    if value not in ('baseline-model', 'model-baseline'):
        raise ValueError('Arm order must be baseline-model or model-baseline')
    return tuple(value.split('-'))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cp', type=int, choices=(4, 8), required=True)
    models = parser.add_mutually_exclusive_group(required=True)
    models.add_argument('--oproj-comm-model', type=Path)
    models.add_argument('--qkv-comm-model', type=Path)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--list-only', action='store_true')
    parser.add_argument('--diagnostic-trace', action='store_true',
                        help='mark externally traced runs non-formal; preserve the same A/B execution sequence')
    parser.add_argument('--model', default='production_qwen_dense')
    parser.add_argument('--seqs', default='131072')
    parser.add_argument('--case-ids', type=Path)
    parser.add_argument('--launches', default='eager,graph')
    parser.add_argument('--arm-order', choices=('baseline-model', 'model-baseline'), default='baseline-model',
                        help='capture and execute both arms in this order in every round; ratio stays baseline/model')
    parser.add_argument('--library', type=Path,
                        default=bench.ROOT / 'build-mxfp8/libfuse_mxfp8_torch_bridge.so')
    parser.add_argument('--output', type=Path,
                        default=bench.ROOT / 'results/mxfp8_weight/policy_ab/current.json')
    args = parser.parse_args(argv)
    if args.qkv_comm_model is not None and args.cp != 4:
        parser.error('--qkv-comm-model is calibrated only for CP4')
    launches = args.launches.split(',')
    if len(set(launches)) != len(launches) or not set(launches) <= {'eager', 'graph'}:
        parser.error('--launches must contain unique eager/graph entries')
    args.operators = 'qkv_forward' if args.qkv_comm_model else 'oproj_forward'
    args.backends = 'fuse'
    return args


def service_prefix(model):
    return 'qkv_forward' if model.get('operator') == 'qkv_forward' else 'oproj'


def measurement_contract(args):
    """External tracing must not turn diagnostic timings into formal evidence."""
    if args.diagnostic_trace:
        return dict(formal_ab_protocol=False, diagnostic_trace=True,
                    diagnostic_reason='external_profiler_same_execution_sequence_not_formal_performance')
    return dict(formal_ab_protocol=True)


def service_metadata_key(model):
    return service_prefix(model) + '_comm_model'


def set_arm(library, runtime, model, arm):
    """Restore the real host dispatch, never replace model selection with a CTA."""
    if arm not in ARMS:
        raise ValueError(f'Unknown frozen-policy arm: {arm}')
    manual = library.fuse_mxfp8_test_set_comm_ctas
    manual.argtypes, manual.restype = [ct.c_int32], ct.c_int
    runtime.check(manual(0))
    prefix = service_prefix(model)
    qkv = prefix == 'qkv_forward'
    setter = getattr(library, f'fuse_mxfp8_test_set_{prefix}_comm_model')
    setter.argtypes = [ct.c_int32, ct.c_int32, ct.POINTER(ct.c_double)]
    setter.restype = ct.c_int
    if arm == 'model':
        value_fn = bench.qkv_forward_comm_model_values if qkv else bench.oproj_comm_model_values
        values = (ct.c_double * (3 if qkv else 6))(*value_fn(model))
        runtime.check(setter(model['world_size'], model['sm_count'], values))
        setattr(library, '_fuse_' + service_metadata_key(model), copy.deepcopy(model))
        reader = bench.read_qkv_forward_comm_model if qkv else bench.read_oproj_comm_model
        reader(library, runtime)
    else:
        runtime.check(setter(0, 0, None))
        vars(library).pop('_fuse_' + service_metadata_key(model), None)
        query = getattr(library, f'fuse_mxfp8_test_get_{prefix}_comm_model')
        query.argtypes, query.restype = [ct.POINTER(ct.c_double)], ct.c_int
        values = (ct.c_double * (5 if qkv else 8))()
        runtime.check(query(values))
        zeros = 4 if qkv else 6
        if list(values)[:zeros] != [0.0] * zeros or not all(map(math.isfinite, values)):
            raise RuntimeError('Baseline did not disable the native forward service model')


def activate(state, model, arm, dist):
    set_arm(state.library, state.runtime, model, arm)
    # refresh_config() updates fields in place; do not carry model metadata
    # from the previous arm into the unchanged-auto baseline.
    metadata_key = service_metadata_key(model)
    state.config.pop(metadata_key, None)
    state.refresh_config(requested_comm_ctas=0)
    config = copy.deepcopy(state.config)
    validate_config(0, config)
    expected = ('existing_bf16_production_auto' if arm == 'baseline'
                else 'calibrated_model_or_domain_fallback')
    if config['dispatch'] != expected or (metadata_key in config) != (arm == 'model'):
        raise RuntimeError('Arm label/model metadata does not match native dispatch')
    ranks = [None] * state.world
    dist.all_gather_object(ranks, config)
    if any(other != config for other in ranks):
        raise RuntimeError('Ranks selected different frozen-policy configurations')
    return config


def paired_summary(rounds):
    if [row['round'] for row in rounds] != list(range(1, ROUNDS + 1)):
        raise ValueError('Exactly three ordered, complete formal pairs are required')
    ratios = []
    for row in rounds:
        if [row[arm]['arm'] for arm in ARMS] != list(ARMS):
            raise ValueError('Formal pairs must preserve baseline/model arm labels')
        ratios.append(positive_ratio(row['baseline']['timing']['p50_us'],
                                     row['model']['timing']['p50_us']))
    return dict(paired_speedups=ratios,
                geometric_mean_speedup=math.exp(sum(map(math.log, ratios)) / len(ratios)),
                min_speedup=min(ratios), max_speedup=max(ratios),
                definition='paired_baseline_p50_divided_by_frozen_model_p50')


def measure_launch(state, args, model, launch, runtime, dist):
    qkv = service_prefix(model) == 'qkv_forward'
    if (state.op != (0 if qkv else 1) or state.name != ('qkv_forward' if qkv else 'oproj_forward') or
            launch not in ('eager', 'graph')):
        raise ValueError('Frozen-policy A/B requires the matching forward operator, eager or graph')
    configs, invocations, checks = {}, {}, {}
    order = arm_sequence(getattr(args, 'arm_order', 'baseline-model'))
    result = dict(launch=launch, phase='forward', rounds=[], warmup=WARMUP,
                  iterations=ITERATIONS, scope='QKV_F_only_not_2F2B_total' if qkv else
                  'OProj_F_only_not_2F2B_total', capture_order=[])
    try:
        for arm in order:
            configs[arm] = activate(state, model, arm, dist)
            invocation = runtime.Invocation(state.launch('forward', None), launch, 2,
                                            state.prepare('forward', None), state.stream)
            invocations[arm] = invocation
            if activate(state, model, arm, dist) != configs[arm]:
                raise RuntimeError('Native configuration changed during capture')
            invocation.once(state.prepare('forward', None, poison=True))
            checks[arm] = state.verify('forward', None)
            result['capture_order'].append(arm)
        result['configs'] = configs
        result['correctness_after_capture'] = checks
        # Compare actual kernel choices, not labels: fallback and unchanged
        # configurations are still timed, including Eager host model overhead.
        kernel_keys = ('comm_ctas', 'tile_m', 'tile_n', 'tile_k', 'cluster_m', 'policy_enum')
        result['kernel_configuration_changed'] = any(
            configs['baseline'][key] != configs['model'][key] for key in kernel_keys)
        for round_index in range(1, ROUNDS + 1):
            pair = dict(round=round_index, execution_order=[])
            for arm in order:
                if activate(state, model, arm, dist) != configs[arm]:
                    raise RuntimeError('Current native configuration differs from the captured arm')
                invocation = invocations[arm]
                for _ in range(WARMUP):
                    invocation.once()
                record = dict(arm=arm, config=copy.deepcopy(configs[arm]),
                              timing=invocation.measure(ITERATIONS, state.flops('forward')))
                if round_index == ROUNDS:
                    record['correctness_after_formal_samples'] = state.verify('forward', None)
                pair[arm] = record
                pair['execution_order'].append(arm)
            pair['speedup'] = positive_ratio(pair['baseline']['timing']['p50_us'],
                                             pair['model']['timing']['p50_us'])
            result['rounds'].append(pair)
            if state.rank == 0:
                print(f'  {launch} round {round_index}: baseline/model {pair["speedup"]:.3f}x', flush=True)
        result['paired_summary'] = paired_summary(result['rounds'])
        return result
    finally:
        invocations.clear()
        set_arm(state.library, state.runtime, model, 'baseline')
        state.config.pop(service_metadata_key(model), None)


def main(argv=None):
    args = arguments(argv)
    order = arm_sequence(args.arm_order)
    cases = bench.select_cases(args, args.cp)
    if args.list_only:
        print(json.dumps(dict(settings=len(cases), cases=cases), indent=2))
        return
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}; choose a new output')
    if int(os.environ.get('WORLD_SIZE', 1)) != args.cp:
        raise RuntimeError('Use torchrun with one process per GPU and matching --cp')
    bench.initialize_runtime()
    torch, dist, runtime = bench.torch, bench.dist, bench.rt
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    dist.init_process_group('gloo', timeout=timedelta(minutes=30))
    rank = dist.get_rank()
    try:
        props = torch.cuda.get_device_properties(local_rank)
        devices = [None] * args.cp
        dist.all_gather_object(devices, dict(rank=rank, local_rank=local_rank, name=props.name,
                                           cc=[props.major, props.minor], sm_count=props.multi_processor_count,
                                           total_memory_bytes=props.total_memory,
                                           cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
        if any(item['cc'] != [9, 0] for item in devices):
            raise RuntimeError('Frozen-policy A/B targets SM90')
        library, cuda = ct.CDLL(str(args.library)), runtime.CudaRuntime()
        library.fuse_mxfp8_test_profiling_enabled.restype = ct.c_int
        if library.fuse_mxfp8_test_profiling_enabled() != 0:
            raise RuntimeError('Formal policy A/B refuses a profiling-enabled library')
        fingerprints, clock_snapshot = [None], [None]
        if rank == 0:
            fingerprints[0] = bench.source_hashes(args)
            for path in (Path(__file__), Path(__file__).with_name('operator_sweep.py')):
                fingerprints[0][str(path.relative_to(bench.ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            clock_snapshot[0] = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=index,uuid,clocks.max.sm,clocks.current.sm,clocks.max.memory,power.limit',
                 '--format=csv'], text=True)
        dist.broadcast_object_list(fingerprints, src=0)
        dist.broadcast_object_list(clock_snapshot, src=0)
        qkv = args.qkv_comm_model is not None
        configure = bench.configure_qkv_forward_comm_model if qkv else bench.configure_oproj_comm_model
        model = configure(args.qkv_comm_model if qkv else args.oproj_comm_model,
                          args.cp, props.multi_processor_count, library, cuda, fingerprints[0], workers=dist)
        set_arm(library, cuda, model, 'baseline')
        report = dict(schema='mxfp8-qkv-forward-frozen-policy-ab-v1' if qkv else 'mxfp8-frozen-policy-ab-v1',
                      scope='QKV_F_only_not_2F2B_total' if qkv else 'OProj_F_only_not_2F2B_total',
                      production_policy_written=False, milestone_evidence=False,
                      **measurement_contract(args),
                      selection='one_explicit_frozen_model_no_search_no_winner_selection',
                      arms=dict(baseline='manual_CTA_zero_native_model_disabled_unchanged_auto',
                                model='manual_CTA_zero_explicit_model_native_selector_or_domain_fallback'),
                      args={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                      devices=devices, gpu_clock_snapshot=clock_snapshot[0], sources=fingerprints[0],
                      **{service_metadata_key(model): model}, torch=torch.__version__, cuda=torch.version.cuda,
                      policy_environment={key: os.environ.get(key) for key in bench.POLICY_ENV_VARIABLES},
                      semantic='offline_mxfp8_weight_runtime_dq_bf16_tensor_core_gemm',
                      token_semantic='T_is_flattened_tokens_no_training_batch',
                      timing=dict(rounds=ROUNDS, warmup_per_arm=WARMUP, samples_per_arm=ITERATIONS,
                                  sample_statistic='sample_wise_max_across_ranks',
                                  clocks='CUDA_events_one_operation_per_sample',
                                  ab_order='_'.join(order) + '_repeated_three_times_per_case_and_launch',
                                  capture_order=list(order),
                                  included=['software_weight_dequant', 'BF16_A2A', 'BF16_GEMM'],
                                  excluded=['offline_quantization', 'allocation_IPC_references_correctness',
                                            'graph_capture_JIT', 'ready_done_reset_CPU_Gloo_barrier',
                                            'arm_setters_model_metadata_checks', 'warmup'],
                                  eager='real_native_host_selector_executes_on_every_call_not_manual_CTA_replacement',
                                  graph='one_capture_per_arm_then_reuse_capture_across_three_pairs'),
                      selected_case_ids=[case['id'] for case in cases], cases=[], complete=False)
        started = time.monotonic()
        if rank == 0:
            bench.checkpoint(args.output, report)
        for case in cases:
            if rank == 0:
                print(f'[{len(report["cases"]) + 1}/{len(cases)}] {case["id"]}: frozen policy A/B', flush=True)
            state = bench.OperatorCase(case, library, cuda)
            try:
                record = dict(case=case, operator=state.name, records=[])
                for launch in args.launches.split(','):
                    record['records'].append(measure_launch(state, args, model, launch, runtime, dist))
                report['cases'].append(record)
                report['wall_seconds'] = time.monotonic() - started
                if rank == 0:
                    bench.checkpoint(args.output, report)
            finally:
                state.close()
                del state
                gc.collect()
                torch.cuda.empty_cache()
        report['complete'] = len(report['cases']) == len(cases)
        report['wall_seconds'] = time.monotonic() - started
        if rank == 0:
            bench.checkpoint(args.output, report)
            label = 'QKV F' if qkv else 'OProj F'
            print(f'Completed {len(cases)} {label} settings (not 2F2B total): {args.output}', flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
