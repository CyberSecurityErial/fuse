#!/usr/bin/env python3
"""Measure communication-CTA, QKV tile or WGrad candidates, never a production policy.

Communication sweeps time F or independent B without W. WGrad sweeps time
only W, with valid inputs prepared once by an untimed B. Neither establishes
a 2F2B-total speedup or milestone; use the full operator benchmark for that.
"""

import argparse
import copy
import ctypes as ct
from datetime import timedelta
import gc
import hashlib
import math
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import operator_bench as bench


QKV_TILES = {1: (64, 1), 2: (128, 1), 3: (160, 1),
             4: (192, 1), 5: (256, 2), 6: (320, 2)}


def qkv_tile_candidates(value):
    # Candidate IDs are benchmark-local, not a native policy ABI.
    try:
        values = [int(item) for item in value.split(',')]
    except ValueError as error:
        raise argparse.ArgumentTypeError('QKV tile IDs must be integers') from error
    if (len(values) < 2 or len(set(values)) != len(values) or 0 not in values or
            not set(values) <= {0, *QKV_TILES}):
        raise argparse.ArgumentTypeError('Include auto (0) and unique QKV tile IDs from 1..6')
    return [0] + [item for item in values if item]


def set_qkv_tile(requested):
    if requested == 0:
        os.environ.pop('FUSE_QKV_GEMM_POLICY', None)
    elif requested in QKV_TILES:
        os.environ['FUSE_QKV_GEMM_POLICY'] = f'm128n{QKV_TILES[requested][0]}'
    else:
        raise ValueError(f'Unknown QKV tile ID: {requested}')
    return 0


def comm_candidates(value):
    try:
        values = [int(item) for item in value.split(',')]
    except ValueError as error:
        raise argparse.ArgumentTypeError('Communication CTAs must be comma-separated integers') from error
    if len(values) < 2 or len(set(values)) != len(values) or 0 not in values:
        raise argparse.ArgumentTypeError('Include auto (0) and at least one unique manual candidate')
    if any(item < 0 or item % 2 for item in values):
        raise argparse.ArgumentTypeError('Communication CTAs must be zero or positive even integers')
    # Always establish the unchanged production-auto reference first.
    return [0] + [item for item in values if item]


def wgrad_candidates(value):
    try:
        values = [int(item) for item in value.split(',')]
    except ValueError as error:
        raise argparse.ArgumentTypeError('WGrad policies must be comma-separated integers') from error
    if len(values) < 2 or len(set(values)) != len(values) or 0 not in values or not set(values) <= set(range(7)):
        raise argparse.ArgumentTypeError('Include auto (0) and unique manual WGrad policies from 1,2,3,4,5,6')
    return [0] + [item for item in values if item]


def request_key(kind):
    if kind == 'qkv_tile':
        return 'requested_qkv_tile'
    return 'requested_wgrad_policy' if kind == 'wgrad' else 'requested_comm_ctas'


def actual_candidate(config, kind):
    if kind == 'qkv_tile':
        return tuple(config[key] for key in ('tile_m', 'tile_n', 'tile_k', 'cluster_m'))
    return config['weight_gemm']['policy_enum'] if kind == 'wgrad' else config['comm_ctas']


def override_setter(library, kind):
    if kind == 'qkv_tile':
        return set_qkv_tile
    return (library.fuse_mxfp8_test_set_wgrad_policy if kind == 'wgrad'
            else library.fuse_mxfp8_test_set_comm_ctas)


def validate_config(requested, config, kind='comm'):
    if kind == 'qkv_tile':
        if requested not in {0, *QKV_TILES}:
            raise ValueError('Unknown QKV tile request')
        validate_config(config['fixed_comm_ctas'], config, 'comm')
        if config['requested_qkv_tile'] != requested:
            raise ValueError('QKV tile request metadata mismatch')
        tile = actual_candidate(config, kind)
        if requested and tile != (128, QKV_TILES[requested][0], 64, QKV_TILES[requested][1]):
            raise ValueError(f'QKV tile request {requested} silently changed to {tile}')
        if tile[0] != 128 or tile[2] != 64 or (tile[1], tile[3]) not in QKV_TILES.values():
            raise ValueError(f'Unexpected native QKV tile: {tile}')
        return
    if kind == 'wgrad':
        weight = config['weight_gemm']
        actual = weight['policy_enum']
        if actual not in (1, 2, 3, 4, 5, 6) or actual != (requested or 1):
            raise ValueError(f'Invalid or silently changed WGrad policy: {weight}')
        if weight['requested_policy'] != requested or weight['output_dtype'] != 'fp32':
            raise ValueError('WGrad request/output dtype metadata does not match the timed path')
        if any(weight[key] <= 0 for key in ('tile_m', 'tile_n', 'tile_k', 'cluster_m',
                                           'stages', 'dynamic_smem_bytes', 'registers_per_thread')):
            raise ValueError(f'Invalid native WGrad tile/resource metadata: {weight}')
        return
    actual, sm, cluster = (config[key] for key in ('comm_ctas', 'sm_count', 'cluster_m'))
    if not 0 < actual < sm or cluster < 1:
        raise ValueError(f'Invalid selected communication/SM/cluster counts: {config}')
    if requested and actual != requested:
        raise ValueError(f'Manual communication request {requested} silently changed to {actual}')
    if actual % cluster or (sm - actual) % cluster:
        raise ValueError(f'Communication split is incompatible with GEMM cluster: {config}')
    if config['requested_comm_ctas'] != requested:
        raise ValueError('Requested communication metadata does not match the bridge override')


def positive_ratio(numerator, denominator):
    if not all(math.isfinite(value) and value > 0 for value in (numerator, denominator)):
        raise ValueError('A/B durations must be positive and finite')
    return numerator / denominator


def shortlist(search, count=2, kind='comm'):
    """Exclude auto's actual kernel configuration and deduplicate candidates."""
    key = request_key(kind)
    autos = [record for record in search if record[key] == 0]
    if len(autos) != 1:
        raise ValueError('Search must contain exactly one auto measurement')
    auto_actual = actual_candidate(autos[0]['config'], kind)
    selected, actuals = [], {auto_actual}
    for record in sorted(search, key=lambda row: (row['timing']['p50_us'], row[key])):
        actual = actual_candidate(record['config'], kind)
        if actual in actuals:
            continue
        selected.append(record[key])
        actuals.add(actual)
        if len(selected) == count:
            break
    if not selected:
        raise ValueError('No measured manual candidate differs from the actual auto configuration')
    return selected


def interleaved_schedule(candidates, rounds, order='auto-candidate'):
    if order not in ('auto-candidate', 'candidate-auto'):
        raise ValueError('Unknown paired execution order')
    return [(round_index, requested) for round_index in range(1, rounds + 1)
            for candidate in candidates for requested in
            ((0, candidate) if order == 'auto-candidate' else (candidate, 0))]


def summarize_round(round_index, measurements, kind='comm', order='auto-candidate'):
    if order not in ('auto-candidate', 'candidate-auto'):
        raise ValueError('Unknown paired execution order')
    if not measurements or len(measurements) % 2:
        raise ValueError('An A/B round must have complete auto/candidate pairs')
    pairs = []
    key = request_key(kind)
    for index in range(0, len(measurements), 2):
        pair = measurements[index:index + 2]
        auto, candidate = pair if order == 'auto-candidate' else pair[::-1]
        if auto[key] != 0 or candidate[key] == 0:
            raise ValueError('A/B measurements must remain in auto/candidate order')
        pairs.append(dict(auto=auto, candidate=candidate,
                          speedup=positive_ratio(auto['timing']['p50_us'], candidate['timing']['p50_us'])))
    first = pairs[0]['auto']['timing']['p50_us']
    self_ratios = [positive_ratio(first, pair['auto']['timing']['p50_us']) for pair in pairs]
    result = dict(round=round_index, pairs=pairs, auto_self_ratios=self_ratios,
                  auto_self_ratio_definition='first_auto_p50 / each_auto_p50_within_this_round')
    if order == 'candidate-auto':
        result['execution_order'] = [row[key] for row in measurements]
    return result


def summarize_candidates(rounds, kind='comm'):
    grouped = {}
    key = request_key(kind)
    for result in rounds:
        for pair in result['pairs']:
            requested = pair['candidate'][key]
            grouped.setdefault(requested, []).append(pair['speedup'])
    return [dict({key: requested}, rounds=len(ratios),
                 paired_speedups=ratios,
                 geometric_mean_speedup=math.exp(sum(map(math.log, ratios)) / len(ratios)),
                 min_speedup=min(ratios), max_speedup=max(ratios))
            for requested, ratios in grouped.items()]


def measurement_contract(args):
    """Preserve the execution protocol while isolating externally traced data."""
    formal = args.warmup == 10 and args.iterations == 50 and args.rounds >= 3
    traced = getattr(args, 'diagnostic_trace', False)
    result = dict(formal_ab_protocol=formal and not traced,
                  ab_kind='formal_paired_ab' if formal and not traced else 'diagnostic_paired_ab')
    if traced:
        result.update(diagnostic_trace=True,
                      diagnostic_reason='external_profiler_same_execution_sequence_not_formal_performance')
    return result


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cp', type=int, choices=(4, 8), required=True)
    parser.add_argument('--sweep-kind', choices=('comm', 'wgrad', 'qkv_tile'), default='comm')
    parser.add_argument('--operators', help='default: four operators for comm, two backward operators for WGrad')
    parser.add_argument('--model', default='production_qwen_dense')
    parser.add_argument('--seqs', default='131072')
    parser.add_argument('--case-ids', type=Path)
    parser.add_argument('--comm-ctas', type=comm_candidates, default='0,4,8,12,16,24,32')
    parser.add_argument('--wgrad-policies', type=wgrad_candidates, default='0,2,3,4')
    parser.add_argument('--qkv-tiles', type=qkv_tile_candidates, default='0,2,4,6',
                        help='0:auto; 1..6:N64/128/160/192/256/320, M128 K64')
    parser.add_argument('--fixed-comm-ctas', type=int,
                        help='required for qkv_tile: keep the same positive even communication budget')
    parser.add_argument('--oproj-comm-model', type=Path,
                        help='not supported in sweeps; use the formal benchmark or profile runner')
    parser.add_argument('--weight-modes', default='immediate,deferred', help='WGrad beta0/beta1 modes')
    parser.add_argument('--backward-phase', choices=('data', 'total'), default='data',
                        help='comm sweep: independent B or actual B then W for each weight mode')
    parser.add_argument('--pair-order', choices=('auto-candidate', 'candidate-auto'), default='auto-candidate')
    parser.add_argument('--launch', choices=('graph', 'eager'), default='graph')
    parser.add_argument('--search-warmup', type=int, default=2)
    parser.add_argument('--search-iterations', type=int, default=7)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--diagnostic-trace', action='store_true',
                        help='external profiler: keep execution counts/order, add NVTX arm ranges, mark output non-formal')
    parser.add_argument('--library', type=Path,
                        default=bench.ROOT / 'build-mxfp8/libfuse_mxfp8_torch_bridge.so')
    parser.add_argument('--output', type=Path,
                        default=bench.ROOT / 'results/mxfp8_weight/sweep/current.json')
    args = parser.parse_args(argv)
    if args.oproj_comm_model is not None:
        parser.error('Sweeps require the unchanged auto baseline; --oproj-comm-model is only for bench/profile')
    if args.operators is None:
        args.operators = ('qkv_forward' if args.sweep_kind == 'qkv_tile' else
                          ','.join(bench.OPERATORS[2:] if args.sweep_kind == 'wgrad' else bench.OPERATORS))
    if args.backward_phase == 'total' and (args.sweep_kind != 'comm' or
            not set(args.operators.split(',')) <= set(bench.OPERATORS[2:])):
        parser.error('--backward-phase total requires a comm sweep of backward operators only')
    if args.sweep_kind == 'qkv_tile':
        if (args.operators != 'qkv_forward' or args.fixed_comm_ctas is None or
                args.fixed_comm_ctas <= 0 or args.fixed_comm_ctas % 2):
            parser.error('QKV tile sweep requires only qkv_forward and positive even --fixed-comm-ctas')
    elif args.fixed_comm_ctas is not None:
        parser.error('--fixed-comm-ctas applies only to qkv_tile sweeps')
    if args.sweep_kind == 'wgrad' and not set(args.operators.split(',')) <= set(bench.OPERATORS[2:]):
        parser.error('WGrad sweep supports only qkv_backward and oproj_backward')
    modes = args.weight_modes.split(',')
    if len(set(modes)) != len(modes) or not set(modes) <= {'immediate', 'deferred'}:
        parser.error('--weight-modes must contain unique immediate/deferred modes')
    if min(args.search_warmup, args.search_iterations, args.rounds, args.warmup, args.iterations) < 1:
        parser.error('All warmup, iteration and round counts must be positive')
    return args


def activate(state, requested, kind='comm', fixed_comm_ctas=None):
    state.runtime.check(override_setter(state.library, kind)(requested))
    comm_request = fixed_comm_ctas if kind == 'qkv_tile' else (requested if kind == 'comm' else 0)
    if kind == 'qkv_tile':
        state.runtime.check(state.library.fuse_mxfp8_test_set_comm_ctas(comm_request))
    state.refresh_config(requested_comm_ctas=comm_request)
    if kind == 'qkv_tile':
        state.config.update(requested_qkv_tile=requested, fixed_comm_ctas=fixed_comm_ctas,
                            gemm_policy_request=os.environ.get('FUSE_QKV_GEMM_POLICY', 'auto'),
                            dispatch='fixed_comm_native_qkv_tile_selection')
    if kind == 'wgrad':
        state.config['weight_gemm']['requested_policy'] = requested
    config = copy.deepcopy(state.config)
    validate_config(requested, config, kind)
    return config


def prepare_wgrad_inputs(state, runtime):
    """One untimed, independently checked B; preserve its dQKV staging for W."""
    if state.op < 2:
        raise ValueError('WGrad requires a backward operator')
    with runtime.torch.cuda.stream(state.stream):
        runtime.aligned_prepare(state.prepare('data', 'deferred', poison=True), state.stream)
        state.launch('data', 'deferred')()
        state.stream.synchronize()
        return state.verify('data', 'deferred')


def verify_weight_update(state, invocation, mode, phase='weight'):
    if mode == 'deferred':
        invocation.once(state.prepare(phase, mode))
        invocation.once(state.arena.reset_control)
        return {'nonzero_beta1_twice': state.verify(phase, mode, accumulated=2)}

    def prepare_nonzero():
        state.prepare(phase, mode)()
        state.tensors['dw'].fill_(0.125)

    invocation.once(prepare_nonzero)
    return {'nonzero_beta0_overwrite': state.verify(phase, mode)}


def sweep_case(state, args, runtime, dist, mode=None):
    if getattr(args, 'sweep_kind', 'comm') != 'qkv_tile':
        return _sweep_case(state, args, runtime, dist, mode)
    if state.op != 0 or mode is not None:
        raise ValueError('QKV tile sweep supports only QKV forward')
    original = os.environ.get('FUSE_QKV_GEMM_POLICY')
    try:
        return _sweep_case(state, args, runtime, dist, mode)
    finally:
        if original is None:
            os.environ.pop('FUSE_QKV_GEMM_POLICY', None)
        else:
            os.environ['FUSE_QKV_GEMM_POLICY'] = original
        state.runtime.check(state.library.fuse_mxfp8_test_set_comm_ctas(0))


def _sweep_case(state, args, runtime, dist, mode=None):
    kind = getattr(args, 'sweep_kind', 'comm')
    total = kind == 'comm' and getattr(args, 'backward_phase', 'data') == 'total'
    order = getattr(args, 'pair_order', 'auto-candidate')
    if kind == 'wgrad':
        if state.op < 2 or mode not in ('immediate', 'deferred'):
            raise ValueError('WGrad requires a backward operator and an explicit beta0/beta1 mode')
        phase, candidates = 'weight', args.wgrad_policies
    elif total:
        if state.op < 2 or mode not in ('immediate', 'deferred'):
            raise ValueError('Complete backward requires a backward operator and explicit weight mode')
        phase, candidates = 'total', args.comm_ctas
    else:
        phase, mode = ('forward', None) if state.op < 2 else ('data', 'deferred')
        candidates = args.qkv_tiles if kind == 'qkv_tile' else args.comm_ctas
    def activate_candidate(requested):
        return activate(state, requested, kind, getattr(args, 'fixed_comm_ctas', None))
    key = request_key(kind)
    configs = {}
    for requested in candidates:
        config = activate_candidate(requested)
        rank_configs = [None] * state.world
        dist.all_gather_object(rank_configs, config)
        if any(other != config for other in rank_configs):
            raise RuntimeError('Ranks selected different communication/GEMM configurations')
        configs[requested] = config
    result = dict(case=state.case, operator=state.name, phase=phase, weight_mode=mode,
                  sweep_kind=kind, beta=int(mode == 'deferred'),
                  scope='W_only_no_B_DQ_communication_or_2F2B_total' if kind == 'wgrad'
                        else 'F_or_independent_B_only_no_W_or_2F2B_total',
                  tile_selection='explicit_native_WGrad_policy' if kind == 'wgrad'
                                 else 'existing_auto_selector_may_change_tile_when_comm_ctas_changes',
                  fixed_tile_single_factor_experiment=False,
                  search=[], rounds=[])
    if total:
        result.update(scope='actual_B_then_W_not_2F2B_total_or_scheduler',
                      total_semantic='actual_B_then_W_not_sum_of_isolated_medians')
    if kind == 'qkv_tile':
        result.update(scope='QKV_F_only_fixed_comm_tile_experiment_not_2F2B_total',
                      tile_selection='native_environment_tile_override_with_fixed_comm_budget',
                      baseline='auto_tile_at_fixed_comm_not_unmodified_production_auto',
                      fixed_comm_ctas=args.fixed_comm_ctas)
    invocations = {}
    try:
        # One allocation/reference/IPC setup per case. Captures stay outside
        # timing and are reused for the shortlisted formal A/B measurements.
        for requested in candidates:
            if activate_candidate(requested) != configs[requested]:
                raise RuntimeError('Configuration changed after candidate validation')
            call = state.launch_boundary(phase, mode) if total else state.launch(phase, mode)
            invocation = runtime.Invocation(call, args.launch, args.search_warmup,
                                            state.prepare(phase, mode), state.stream)
            invocation.once(state.prepare(phase, mode, poison=True))
            timing = invocation.measure(args.search_iterations, state.flops(phase))
            correctness = state.verify(phase, mode)
            if kind == 'wgrad' or total:
                correctness.update(verify_weight_update(state, invocation, mode, phase))
            result['search'].append(dict({key: requested}, config=configs[requested],
                                         timing=timing, correctness=correctness,
                                         kind='diagnostic_search_not_formal_performance'))
            invocations[requested] = invocation
            if state.rank == 0:
                print(f'  {kind}/{mode} search request={requested} actual={actual_candidate(configs[requested], kind)}: '
                      f'{timing["p50_us"]:.3f} us (diagnostic)', flush=True)
        selected = shortlist(result['search'], kind=kind)
        result['selected_' + key] = selected
        result['selection'] = 'two_fastest_distinct_actual_configurations_excluding_auto_or_fewer_if_unavailable'
        for requested in list(invocations):
            if requested not in [0, *selected]:
                del invocations[requested]
        measurements = []
        for round_index, requested in interleaved_schedule(selected, args.rounds, order):
            if activate_candidate(requested) != configs[requested]:
                raise RuntimeError('Current configuration differs from the captured configuration')
            invocation = invocations[requested]
            traced = getattr(args, 'diagnostic_trace', False)
            if traced:
                bench.torch.cuda.nvtx.range_push(
                    f'sweep/{state.case["id"]}/{phase}/{mode}/round{round_index}/request{requested}')
            try:
                for _ in range(args.warmup):
                    invocation.once()
                timing = invocation.measure(args.iterations, state.flops(phase))
            finally:
                if traced:
                    bench.torch.cuda.nvtx.range_pop()
            record = dict({key: requested}, config=configs[requested], timing=timing)
            if round_index == args.rounds:
                record['correctness_after_formal_samples'] = state.verify(phase, mode)
            measurements.append(record)
            if len(measurements) == 2 * len(selected):
                result['rounds'].append(summarize_round(round_index, measurements, kind, order))
                if state.rank == 0:
                    pairs = result['rounds'][-1]['pairs']
                    ratios = ', '.join(f'{kind}{actual_candidate(pair["candidate"]["config"], kind)}: '
                                       f'{pair["speedup"]:.3f}x' for pair in pairs)
                    print(f'  A/B round {round_index}: {ratios}', flush=True)
                measurements = []
        result['paired_summary'] = summarize_candidates(result['rounds'], kind)
        return result
    finally:
        invocations.clear()
        state.runtime.check(override_setter(state.library, kind)(0))


def main():
    args = arguments()
    if int(os.environ.get('WORLD_SIZE', 1)) != args.cp:
        raise RuntimeError('Use torchrun with one process per GPU and matching --cp')
    cases = bench.select_cases(SimpleNamespace(**vars(args), full=False), args.cp)
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}; choose a new output')
    bench.initialize_runtime()
    torch, dist, runtime = bench.torch, bench.dist, bench.rt
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    dist.init_process_group('gloo', timeout=timedelta(minutes=30))
    rank = dist.get_rank()
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        device = dict(rank=rank, name=props.name, cc=[props.major, props.minor],
                      sm_count=props.multi_processor_count, total_memory_bytes=props.total_memory,
                      cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
        devices = [None] * args.cp
        dist.all_gather_object(devices, device)
        if any(item['cc'] != [9, 0] for item in devices):
            raise RuntimeError('This sweep targets SM90')
        if args.sweep_kind == 'comm' and max(args.comm_ctas) >= min(item['sm_count'] for item in devices):
            raise ValueError('Every manual communication candidate must be smaller than each GPU SM count')
        if args.sweep_kind == 'qkv_tile' and args.fixed_comm_ctas >= min(item['sm_count'] for item in devices):
            raise ValueError('Fixed communication budget must leave compute SMs on every GPU')
        library = ct.CDLL(str(args.library))
        library.fuse_mxfp8_test_profiling_enabled.restype = ct.c_int
        if library.fuse_mxfp8_test_profiling_enabled() != 0:
            raise RuntimeError('Performance sweep refuses a profiling-enabled library')
        library.fuse_mxfp8_test_set_comm_ctas.argtypes = [ct.c_int32]
        library.fuse_mxfp8_test_set_comm_ctas.restype = ct.c_int
        cuda = runtime.CudaRuntime()
        cuda.check(library.fuse_mxfp8_test_set_comm_ctas(0))
        if args.sweep_kind == 'wgrad':
            library.fuse_mxfp8_test_set_wgrad_policy.argtypes = [ct.c_int32]
            library.fuse_mxfp8_test_set_wgrad_policy.restype = ct.c_int
            cuda.check(library.fuse_mxfp8_test_set_wgrad_policy(0))
        fingerprints = [None]
        clock_snapshot = [None]
        if rank == 0:
            fingerprints[0] = bench.source_hashes(SimpleNamespace(library=args.library, backends='fuse'))
            fingerprints[0][str(Path(__file__).relative_to(bench.ROOT))] = hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest()
            clock_snapshot[0] = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=index,uuid,clocks.max.sm,clocks.current.sm,clocks.max.memory,power.limit',
                 '--format=csv'], text=True)
        dist.broadcast_object_list(fingerprints, src=0)
        dist.broadcast_object_list(clock_snapshot, src=0)
        wgrad = args.sweep_kind == 'wgrad'
        total = args.sweep_kind == 'comm' and args.backward_phase == 'total'
        report = dict(schema='mxfp8-wgrad-policy-sweep-v1' if wgrad else 'mxfp8-communication-cta-sweep-v1',
                      sweep_kind=args.sweep_kind,
                      scope='W_only_no_B_DQ_communication_or_2F2B_total' if wgrad
                            else 'F_or_independent_B_only_no_W_or_2F2B_total',
                      production_policy_written=False, milestone_evidence=False,
                      **measurement_contract(args),
                      args={key: str(value) if isinstance(value, Path) else value
                            for key, value in vars(args).items()},
                      devices=devices, sources=fingerprints[0], gpu_clock_snapshot=clock_snapshot[0],
                      torch=torch.__version__, cuda=torch.version.cuda,
                      policy_environment={key: os.environ.get(key) for key in bench.POLICY_ENV_VARIABLES},
                      semantic='BF16_inputs_FP32_WGrad_beta0_or_beta1' if wgrad
                               else 'offline_mxfp8_weight_runtime_dq_bf16_tensor_core_gemm',
                      token_semantic='T_is_flattened_tokens_no_training_batch',
                      timing=dict(sample_statistic='sample_wise_max_across_ranks',
                                  clocks='CUDA_events_one_operation_per_sample',
                                  included=['BF16_input_FP32_output_WGrad_GEMM'] if wgrad
                                           else ['software_weight_dequant', 'BF16_GEMM', 'BF16_A2A'],
                                  excluded=(['B_input_preparation', 'software_weight_dequant', 'BF16_A2A',
                                             'main_grad_initialization_for_each_sample'] if wgrad else ['W']) +
                                           ['offline_weight_quantization', 'references_and_correctness',
                                            'allocation_IPC_bootstrap_capture_and_JIT',
                                            'ready_done_reset_and_CPU_Gloo_barrier', 'warmup'],
                                  graph='reuse_candidate_capture_with_fresh_replay_warmup_for_each_AB_arm',
                                  search='diagnostic_only_never_formal_or_production_dispatch',
                                  ab_order='auto_candidate1_auto_candidate2_per_round'),
                      selected_case_ids=[case['id'] for case in cases], cases=[], complete=False)
        if args.pair_order == 'candidate-auto':
            report['timing']['ab_order'] = 'candidate1_auto_candidate2_auto_per_round'
        if total:
            report.update(schema='mxfp8-backward-complete-cta-sweep-v1',
                          scope='actual_B_then_W_not_2F2B_total_or_scheduler')
            report['timing']['included'].append('FP32_WGrad_GEMM_beta0_or_beta1')
            report['timing']['excluded'].remove('W')
            report['timing']['excluded'].append('main_grad_initialization_for_each_sample')
        if args.sweep_kind == 'qkv_tile':
            report.update(schema='mxfp8-qkv-fixed-comm-tile-sweep-v1',
                          scope='QKV_F_only_fixed_comm_tile_experiment_not_2F2B_total',
                          baseline='auto_tile_at_fixed_comm_not_unmodified_production_auto')
        started = time.monotonic()
        if rank == 0:
            bench.checkpoint(args.output, report)
        for case in cases:
            if rank == 0:
                print(f'[{len(report["cases"]) + 1}/{len(cases)}] {case["id"]}: {args.sweep_kind} sweep', flush=True)
            state = bench.OperatorCase(case, library, cuda)
            try:
                if wgrad or total:
                    record = dict(case=case, operator=state.name, records=[])
                    if wgrad:
                        record['input_preparation_correctness'] = prepare_wgrad_inputs(state, runtime)
                    for mode in args.weight_modes.split(','):
                        record['records'].append(sweep_case(state, args, runtime, dist, mode))
                    report['cases'].append(record)
                else:
                    report['cases'].append(sweep_case(state, args, runtime, dist))
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
            print(f'Completed {len(cases)} {args.sweep_kind} sweeps (not 2F2B total): {args.output}', flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
