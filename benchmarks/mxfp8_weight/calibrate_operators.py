"""Measure matched OProj/QKV primitives for a generic cost model.

One state and IPC setup per registry geometry; one graph per primitive and
CTA request. These are independent service times, not overlapping role spans
or a 2F2B speedup claim. No policy or per-shape winner table is generated.
"""
import argparse
import copy
import ctypes as ct
from datetime import timedelta
import gc
import hashlib
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import operator_bench as bench
from profile_operators import comm_candidates, validate_sample_limit

PRIMITIVES = {'compute_subgrid': 0, 'copy': 1, 'compute_fullgrid': 2}
QKV_PRIMITIVES = {'compute_bare_subgrid': 0, 'copy': 1,
                  'compute_ready_preloaded_subgrid': 2, 'compute_bare_fullgrid': 3,
                  'compute_ready_preloaded_fullgrid': 4, 'copy_fused_reservation': 5}
QKV_FORWARD_PRIMITIVES = {'producer_signaling_subgrid': 0, 'producer_no_signal_subgrid': 1,
                          'producer_signaling_fullgrid': 2, 'producer_no_signal_fullgrid': 3,
                          'copy_fused_reservation': 4}
COPY_PRIMITIVES = ('copy', 'copy_fused_reservation')
QKV_RESOURCE_KEYS = ('selected_gemm_stages', 'primitive_dynamic_smem_bytes',
                     'primitive_registers_per_thread', 'primitive_grid_x',
                     'primitive_launch_cluster_m', 'threads_per_cta', 'copy_use_tma',
                     'copy_slots', 'ready_block_m', 'ready_flag_stride',
                     'packed_heads', 'k_tiles_per_head')
QKV_FORWARD_RESOURCE_KEYS = ('tile_m', 'tile_n', 'tile_k', 'selected_cluster_m',
                             'selected_gemm_stages', 'primitive_dynamic_smem_bytes',
                             'primitive_registers_per_thread', 'primitive_grid_x',
                             'primitive_launch_cluster_m', 'threads_per_cta',
                             'ready_flag_stride', 'ready_m_tiles', 'ready_n_tiles',
                             'copy_use_tma', 'copy_use_tma_store', 'copy_slots')


def primitive_ids(operator):
    if operator == 'oproj_forward':
        return PRIMITIVES
    if operator == 'qkv_backward':
        return QKV_PRIMITIVES
    if operator == 'qkv_forward':
        return QKV_FORWARD_PRIMITIVES
    raise ValueError(f'Unsupported calibration operator: {operator}')


def geometry_work(case, operator='oproj_forward'):
    primitive_ids(operator)
    heads = case['q_heads'] + (2 * case['kv_heads'] if operator.startswith('qkv_') else 0)
    m, n, k, world = case['m'], case['hidden'], heads * case['head_dim'], case['cp']
    if operator == 'qkv_forward':
        n, k = k, n
    if min(m, n, k) <= 0 or world not in (4, 8) or m * world != case['global_seq']:
        raise ValueError('Invalid flattened-token calibration geometry')
    if operator == 'qkv_backward' and case.get('b_mnk', [m, n, k]) != [m, n, k]:
        raise ValueError('QKV backward registry MNK disagrees with packed-head geometry')
    if operator == 'qkv_forward' and (case['n'], case['k']) != (n, k):
        raise ValueError('QKV forward registry MNK disagrees with packed-head geometry')
    routed_width = n if operator == 'qkv_forward' else k
    return dict(m=m, n=n, k=k, cp=world, flops_per_gpu=2 * m * n * k,
                remote_payload_bytes_per_gpu=2 * m * routed_width * (world - 1) // world,
                staging_write_bytes_per_gpu=2 * m * routed_width,
                compulsory_gemm_bytes_per_gpu=2 * (m * k + n * k + m * n),
                byte_definition='logical payload/compulsory lower bounds; not measured DRAM transactions')


def reference_prefix(operator):
    return {'oproj_forward': 'oproj', 'qkv_backward': 'qkv', 'qkv_forward': 'qkv_forward'}[operator]


def forward_ready_bytes(state, resources):
    size = resources['ready_m_tiles'] * resources['ready_n_tiles'] * resources['ready_flag_stride'] * 4
    if not 0 < size <= state.arena.ready_bytes:
        raise ValueError('Actual forward producer flags exceed the allocated IPC arena')
    return size


def allocate_forward_buffers(state, configurations, torch):
    """One reusable verification set per shape, never in a timed invocation."""
    all_resources = [r for _, _, primitives in configurations for r in primitives.values()]
    sizes = {forward_ready_bytes(state, r) for r in all_resources}
    strides = {r['ready_flag_stride'] for r in all_resources}
    if len(sizes) != 1 or len(strides) != 1:
        raise ValueError('Forward calibration must retain one tile/ready layout across CTA candidates')
    size, stride = sizes.pop(), strides.pop()
    with torch.cuda.stream(state.stream):
        zeros = torch.zeros(size // 4, dtype=torch.int32, device=state.device)
        signaled = zeros.clone()
        signaled[::stride] = ct.c_int32(state.argument.geometry[10]).value
        buffers = dict(ready_bytes=size, ready_zero=zeros, ready_signaled=signaled,
                       ready_observed=torch.empty_like(zeros))
        if any(name in COPY_PRIMITIVES for _, _, primitives in configurations for name in primitives):
            # Copy correctness must be bit-exact without assuming two differently
            # tiled reference GEMMs round identically. Build a deterministic BF16
            # producer payload and its rank-major Q/K/V destination once.
            payload = state.values((state.m, state.width), 701 + state.rank)
            expected = torch.empty_like(state.reference['out'])
            for peer in range(state.world):
                source = payload if peer == state.rank else state.values((state.m, state.width), 701 + peer)
                base = 0
                for offset, width in ((0, state.q // state.world),
                                      (state.q, state.kv // state.world),
                                      (state.q + state.kv, state.kv // state.world)):
                    first = base + peer * state.m * width
                    expected[first:first + state.m * width].copy_(
                        source[:, offset + state.rank * width:offset + (state.rank + 1) * width].reshape(-1))
                    base += state.m * state.world * width
            buffers.update(copy_payload=payload, copy_expected=expected)
    return buffers


def required_ready_bytes(state, resources):
    rows = (state.m + resources['ready_block_m'] - 1) // resources['ready_block_m']
    size = rows * resources['packed_heads'] * resources['ready_flag_stride'] * 4
    if not 0 < size <= state.arena.ready_bytes:
        raise ValueError('Actual QKV ready-flag prefix exceeds the allocated IPC arena')
    return size


def allocate_qkv_buffers(state, configurations, torch):
    """One int32 allocation per shape, sized to actual native ready prefixes."""
    required = [required_ready_bytes(state, resources) for _, _, primitives in configurations
                for name, resources in primitives.items() if name not in COPY_PRIMITIVES]
    if not required:
        return None
    epoch = int(state.argument.geometry[10]) & 0xffffffff
    if epoch == 0:
        raise ValueError('QKV ready prepublication needs a nonzero epoch')
    with torch.cuda.stream(state.stream):
        ready = torch.full((max(required) // 4,), ct.c_int32(epoch).value,
                           dtype=torch.int32, device=state.device)
    return dict(ready=ready, bytes=max(required), epoch=epoch)


def prepare_primitive(state, primitive, poison=False, operator='oproj_forward',
                      buffers=None, resources=None):
    """Prepare outside timing; QKV compute arms preload identically each sample."""
    if primitive not in primitive_ids(operator):
        raise ValueError(f'Unknown {operator} primitive: {primitive}')
    if operator == 'qkv_forward':
        reset = state.prepare('forward', None)

        def prepare_forward():
            reset()
            if primitive in COPY_PRIMITIVES:
                if poison:
                    # The route never modifies its source. Initialize before
                    # warmup/verification, not with another multi-GB copy per
                    # sample. A preceding producer may have overwritten it.
                    state.tensors['staging'].copy_(buffers['copy_payload'])
                    state.scratch.fill_(float('nan'))
                    state.arena.write_data(state.scratch)
            else:
                state.tensors['workspace'].copy_(state.effective)
                if poison:
                    state.tensors['staging'].fill_(float('nan'))

        return prepare_forward
    if operator == 'qkv_backward':
        reset = state.prepare('data', 'deferred')
        ready_bytes = required_ready_bytes(state, resources) if primitive not in COPY_PRIMITIVES else 0
        if ready_bytes and (buffers is None or buffers['bytes'] < ready_bytes):
            raise ValueError('Missing or undersized preallocated QKV integer-epoch buffer')
        if ready_bytes and buffers['epoch'] != (int(state.argument.geometry[10]) & 0xffffffff):
            raise ValueError('Preallocated ready buffer contains a different epoch')

        def prepare_qkv():
            reset()
            if primitive in COPY_PRIMITIVES:
                if poison:
                    state.scratch.fill_(float('nan'))
                    state.arena.write_data(state.scratch)
            else:
                # Both bare and ready-preloaded paths get identical preparation
                # on every sample. The real B input is arena.data, not t.staging.
                state.tensors['workspace'].copy_(state.effective)
                state.arena.write_data(state.reference['route'])
                state.runtime.copy(state.arena.pointer('ready'), buffers['ready'].data_ptr(),
                                   ready_bytes, state.stream.cuda_stream)
                if poison:
                    state.tensors['out'].fill_(float('nan'))

        return prepare_qkv
    reset = state.prepare('forward', None)

    def prepare():
        reset()
        if poison:
            state.tensors['workspace'].copy_(state.effective)
            if primitive == 'copy':
                state.tensors['staging'].fill_(float('nan'))
            else:
                state.tensors['staging'].copy_(state.reference['route'])
                state.tensors['out'].fill_(float('nan'))

    return prepare


def check_forward_primitive(state, primitive, runtime, buffers):
    if primitive in COPY_PRIMITIVES:
        state.arena.read_data(state.scratch)
        result = dict(route=runtime.check_error(state.scratch, buffers['copy_expected'], exact=True))
    else:
        result = dict(local_gemm=runtime.check_error(state.tensors['staging'], state.reference['local']))
    state.runtime.copy(buffers['ready_observed'].data_ptr(), state.arena.pointer('ready'),
                       buffers['ready_bytes'], state.stream.cuda_stream)
    signaling = primitive.startswith('producer_signaling_')
    result['ready_flags'] = runtime.check_error(
        buffers['ready_observed'], buffers['ready_signaled' if signaling else 'ready_zero'], exact=True)
    return result


def sample_primitive(state, primitive, args, runtime, resources=None, buffers=None):
    operator = getattr(args, 'operator', 'oproj_forward')
    qkv = operator == 'qkv_backward'
    forward = operator == 'qkv_forward'
    is_copy = primitive in COPY_PRIMITIVES
    primitive_id = primitive_ids(operator)[primitive]
    entry = getattr(state.library, f'fuse_mxfp8_test_{reference_prefix(operator)}_reference')

    def call():
        state.runtime.check(entry(ct.byref(state.argument), primitive_id))

    prepare = prepare_primitive(state, primitive, operator=operator, buffers=buffers, resources=resources)
    poison = prepare_primitive(state, primitive, poison=True, operator=operator,
                               buffers=buffers, resources=resources)

    with bench.torch.cuda.stream(state.stream):
        if (qkv or forward) and is_copy:
            state.stream.synchronize()
            bench.dist.barrier()
        runtime.aligned_prepare(poison, state.stream)
        invocation = runtime.Invocation(call, args.launch, args.warmup, prepare, state.stream)
        if (qkv or forward) and is_copy:
            # Constructor warmups can write this rank's arena from other GPUs.
            state.stream.synchronize()
            bench.dist.barrier()
        invocation.once(poison)
        if forward:
            # once() waits only for this stream; peer writers must also finish
            # before destination validation. measure() later gathers all ranks.
            if is_copy:
                bench.dist.barrier()
            poison_checked = check_forward_primitive(state, primitive, runtime, buffers)
        timing = invocation.measure(args.iterations, 0 if is_copy
                                    else state.flops('data' if qkv else 'forward'))
        if forward:
            checked = check_forward_primitive(state, primitive, runtime, buffers)
        elif is_copy:
            if qkv:
                # measure() all-gathers after every rank's final stop event;
                # only now may we read a destination written by other ranks.
                state.arena.read_data(state.scratch)
            checked = runtime.check_error(state.scratch.view_as(state.reference['route']) if qkv
                                          else state.tensors['staging'],
                                          state.reference['route'], exact=True)
        else:
            checked = runtime.check_error(state.tensors['out'], state.reference['out'])
        del invocation
    # check_error already reduces finite/error metrics across every rank.
    result = dict(timing=timing, correctness=checked)
    if forward:
        result.update(primitive_id=primitive_id, resources=resources,
                      correctness_after_poison=poison_checked,
                      included='copy_only' if is_copy else 'BF16_producer_only',
                      ready_behavior='publish_epoch' if primitive.startswith('producer_signaling_')
                      else 'remain_zero',
                      scope='no_DQ_no_other_role_no_finalize_not_F_total')
    elif qkv:
        result.update(primitive_id=primitive_id, resources=resources,
                      included='copy_only' if is_copy else 'BF16_GEMM_only',
                      ready_prepublication_bytes=0 if is_copy else required_ready_bytes(state, resources),
                      ready_epoch=None if is_copy else buffers['epoch'],
                      ready_prepublication=('not_used_by_copy' if is_copy
                                            else 'same_int32_epoch_prefix_for_bare_and_preloaded'),
                      scope='no_DQ_no_W_no_finalize_not_B_total')
    return result


def forward_resources(state, primitive, config):
    output = (ct.c_int32 * len(QKV_FORWARD_RESOURCE_KEYS))()
    state.runtime.check(state.library.fuse_mxfp8_test_qkv_forward_reference_config(
        ct.byref(state.argument), QKV_FORWARD_PRIMITIVES[primitive], output))
    resource = dict(zip(QKV_FORWARD_RESOURCE_KEYS, output))
    if any(value < 0 for value in output) or any(resource[key] <= 0 for key in QKV_FORWARD_RESOURCE_KEYS[:13]):
        raise RuntimeError('Invalid native QKV forward resource metadata')
    if any(resource[key] != config[key] for key in ('tile_m', 'tile_n', 'tile_k')):
        raise RuntimeError('Forward reference silently changed the selected GEMM tile')
    if (resource['selected_cluster_m'] != config['cluster_m'] or
            resource['primitive_launch_cluster_m'] != config['cluster_m'] or
            resource['threads_per_cta'] % 32 or
            resource['primitive_grid_x'] % resource['primitive_launch_cluster_m']):
        raise RuntimeError('Forward reference has inconsistent cluster/thread resources')
    copy_only = primitive in COPY_PRIMITIVES
    compute_budget = config['sm_count'] - (0 if primitive.endswith('_fullgrid') else config['comm_ctas'])
    # The scheduler caps a short problem's grid at its cluster-padded output
    # tiles. The SM budget and the actual launch grid are distinct quantities.
    cluster_rows = config['tile_m'] * config['cluster_m']
    padded_tiles = ((state.m + cluster_rows - 1) // cluster_rows * config['cluster_m'] *
                    ((state.width + config['tile_n'] - 1) // config['tile_n']))
    grid = config['comm_ctas'] if copy_only else min(compute_budget, padded_tiles)
    if resource['primitive_grid_x'] != grid:
        raise RuntimeError('Forward reference silently changed its requested SM budget')
    if (resource['ready_m_tiles'] != (state.m + config['tile_m'] - 1) // config['tile_m'] or
            resource['ready_n_tiles'] != (state.width + config['tile_n'] - 1) // config['tile_n']):
        raise RuntimeError('Forward reference ready layout disagrees with the output tile grid')
    if copy_only and (resource['copy_use_tma'] != 1 or resource['copy_use_tma_store'] != 1 or
                      resource['copy_slots'] != 12):
        raise RuntimeError('Forward copy calibration requires the actual twelve-slot TMA load/store route')
    forward_ready_bytes(state, resource)
    resource['copy_fields_apply'] = copy_only
    resource['compute_sm_budget'] = 0 if copy_only else compute_budget
    return resource


def candidate_configurations(state, args, dist):
    """Metadata-only prepass also bounds the one reusable ready allocation."""
    configurations = []
    for requested in args.comm_ctas:
        state.runtime.check(state.library.fuse_mxfp8_test_set_comm_ctas(requested))
        state.refresh_config(requested_comm_ctas=requested)
        config = copy.deepcopy(state.config)
        resources = {}
        if args.operator == 'oproj_forward':
            window = (ct.c_int32 * 1)()
            state.runtime.check(state.library.fuse_mxfp8_test_oproj_reference_config(
                ct.byref(state.argument), window))
            if window[0] < 1:
                raise RuntimeError('Invalid native communication frontier window')
            config.update(comm_m_window=window[0], copy_schedule='fused_frontier_window',
                          compute_scheduler='stock_persistent_same_tile_and_SM_budget_no_ready_waits')
        elif args.operator == 'qkv_forward':
            for primitive in args.primitives:
                resources[primitive] = forward_resources(state, primitive, config)
            config.update(copy_schedule='native_rank_major_QKV_route_fused_reservation',
                          compute_scheduler='same_monolithic_scheduler_signaling_entry_ready_pointer_toggle',
                          primitive_scope='no_DQ_no_other_role_no_finalize_not_F_total')
        else:
            for primitive in args.primitives:
                output = (ct.c_int32 * len(QKV_RESOURCE_KEYS))()
                state.runtime.check(state.library.fuse_mxfp8_test_qkv_reference_config(
                    ct.byref(state.argument), QKV_PRIMITIVES[primitive], output))
                resource = dict(zip(QKV_RESOURCE_KEYS, output))
                if any(value < 0 for value in output) or any(resource[key] <= 0 for key in (
                        'selected_gemm_stages', 'primitive_dynamic_smem_bytes',
                        'primitive_registers_per_thread', 'primitive_grid_x',
                        'primitive_launch_cluster_m', 'threads_per_cta', 'ready_block_m',
                        'ready_flag_stride', 'packed_heads', 'k_tiles_per_head')):
                    raise RuntimeError('Invalid native QKV primitive resource metadata')
                if (resource['copy_use_tma'] not in (0, 1) or
                        resource['primitive_grid_x'] % resource['primitive_launch_cluster_m'] or
                        resource['threads_per_cta'] % 32 or
                        resource['packed_heads'] != state.case['q_heads'] + 2 * state.case['kv_heads']):
                    raise RuntimeError('Inconsistent native QKV primitive geometry')
                if primitive == 'copy' and resource['primitive_launch_cluster_m'] != 1:
                    raise RuntimeError('Copy launch cluster must be independent of the selected GEMM cluster')
                if (primitive == 'copy_fused_reservation' and
                        resource['primitive_launch_cluster_m'] != config['cluster_m']):
                    raise RuntimeError('Fused-reservation copy must retain the selected GEMM launch cluster')
                if primitive in COPY_PRIMITIVES and (resource['copy_use_tma'] != 1 or resource['copy_slots'] <= 0):
                    raise RuntimeError('QKV copy calibration requires the native TMA route with active slots')
                required_ready_bytes(state, resource)
                resource['copy_fields_apply'] = primitive in COPY_PRIMITIVES
                resources[primitive] = resource
            config.update(copy_schedule='native_qkv_peer_head_route',
                          compute_scheduler='matched_tile_subgrid_or_fullgrid_bare_or_preloaded_ready',
                          primitive_scope='no_DQ_no_W_no_finalize_not_B_total')
        if requested and config['comm_ctas'] != requested:
            raise RuntimeError('Communication request silently changed')
        ranks = [None] * state.world
        dist.all_gather_object(ranks, (config, resources))
        if any(other != (config, resources) for other in ranks):
            raise RuntimeError('Ranks selected different primitive configurations/resources')
        configurations.append((requested, config, resources))
    return configurations


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cp', type=int, choices=(4, 8), required=True)
    parser.add_argument('--operator', choices=('oproj_forward', 'qkv_backward', 'qkv_forward'), default='oproj_forward')
    parser.add_argument('--primitives', help='comma-separated subset; defaults to all primitives of this operator')
    parser.add_argument('--model', default='production_qwen_dense')
    parser.add_argument('--seqs', default='131072')
    parser.add_argument('--case-ids', type=Path)
    parser.add_argument('--comm-ctas', type=comm_candidates, default='4,8,16,24,32')
    parser.add_argument('--max-cases', type=int, default=20, help='maximum shape × CTA requests')
    parser.add_argument('--launch', choices=('eager', 'graph'), default='graph')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--library', type=Path,
                        default=bench.ROOT / 'build-mxfp8/libfuse_mxfp8_torch_bridge.so')
    parser.add_argument('--output', type=Path,
                        default=bench.ROOT / 'results/mxfp8_weight/calibration/current.json')
    args = parser.parse_args(argv)
    if min(args.warmup, args.iterations, args.max_cases) < 1:
        parser.error('Warmup, sample and case counts must be positive')
    available = primitive_ids(args.operator)
    selected = list(available) if args.primitives is None else args.primitives.split(',')
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(available):
        parser.error(f'Use unique --primitives from {",".join(available)}')
    args.primitives = selected
    if args.operator.startswith('qkv_') and (args.warmup < 10 or args.iterations < 50):
        parser.error('QKV service calibration requires at least 10 warmups and 50 samples')
    return args


def main():
    args = arguments()
    if int(os.environ.get('WORLD_SIZE', 1)) != args.cp:
        raise RuntimeError('Use torchrun with matching --cp')
    cases = bench.select_cases(SimpleNamespace(**vars(args), full=False, operators=args.operator), args.cp)
    count = validate_sample_limit(cases, args.comm_ctas, args.max_cases)
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    bench.initialize_runtime()
    torch, dist, runtime = bench.torch, bench.dist, bench.rt
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    dist.init_process_group('gloo', timeout=timedelta(minutes=30))
    rank = dist.get_rank()
    try:
        library = ct.CDLL(str(args.library))
        library.fuse_mxfp8_test_profiling_enabled.restype = ct.c_int
        if library.fuse_mxfp8_test_profiling_enabled() != 0:
            raise RuntimeError('Primitive measurements require profiling OFF')
        from test_operators import Arguments
        prefix = reference_prefix(args.operator)
        entry = getattr(library, f'fuse_mxfp8_test_{prefix}_reference')
        entry.argtypes, entry.restype = [ct.POINTER(Arguments), ct.c_int32], ct.c_int
        query = getattr(library, f'fuse_mxfp8_test_{prefix}_reference_config')
        query.argtypes = ([ct.POINTER(Arguments), ct.c_int32, ct.POINTER(ct.c_int32)]
                          if args.operator.startswith('qkv_')
                          else [ct.POINTER(Arguments), ct.POINTER(ct.c_int32)])
        query.restype = ct.c_int
        library.fuse_mxfp8_test_set_comm_ctas.argtypes = [ct.c_int32]
        library.fuse_mxfp8_test_set_comm_ctas.restype = ct.c_int
        cuda = runtime.CudaRuntime()
        cuda.check(library.fuse_mxfp8_test_set_comm_ctas(0))
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        device = dict(rank=rank, name=props.name, cc=[props.major, props.minor],
                      sm_count=props.multi_processor_count, total_memory_bytes=props.total_memory)
        devices = [None] * args.cp
        dist.all_gather_object(devices, device)
        if any(item['cc'] != [9, 0] for item in devices):
            raise RuntimeError('This calibration targets SM90')
        if max(args.comm_ctas) >= min(item['sm_count'] for item in devices):
            raise ValueError('Communication candidates must be smaller than the SM count')
        sources, clocks = [None], [None]
        if rank == 0:
            sources[0] = bench.source_hashes(SimpleNamespace(library=args.library, backends='fuse'))
            for name in ('calibrate_operators.py', 'operator_reference.cuh', 'profile_operators.py'):
                path = Path(__file__).with_name(name)
                sources[0][str(path.relative_to(bench.ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            clocks[0] = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=index,uuid,clocks.max.sm,clocks.current.sm,clocks.max.memory,power.limit',
                 '--format=csv'], text=True)
        dist.broadcast_object_list(sources, src=0)
        dist.broadcast_object_list(clocks, src=0)
        qkv = args.operator == 'qkv_backward'
        forward = args.operator == 'qkv_forward'
        schema = ('mxfp8-qkv-forward-independent-primitives-v1' if forward else
                  'mxfp8-qkv-backward-independent-primitives-v1' if qkv else
                  'mxfp8-oproj-independent-primitives-v1')
        report = dict(schema=schema, complete=False,
                      production_policy_written=False, milestone_evidence=False, devices=devices,
                      sources=sources[0], gpu_clock_snapshot=clocks[0],
                      cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                      policy_environment={key: os.environ.get(key) for key in bench.POLICY_ENV_VARIABLES},
                      args={key: str(value) if isinstance(value, Path) else value
                            for key, value in vars(args).items()},
                      timing=dict(sample_statistic='sample_wise_max_across_ranks',
                                  included='one_existing_BF16_compute_or_route_primitive',
                                  excluded=['DQ', 'other_primitive', 'allocation_IPC_capture_references',
                                            'staging_preload_control_reset_CPU_barrier', 'warmup']),
                      expected_samples=count, samples=[])
        if forward:
            report.update(operator='qkv_forward', phase='forward', weight_mode=None,
                          scope='independent_services_no_DQ_no_other_role_no_finalize_not_F_total',
                          primitive_ids=QKV_FORWARD_PRIMITIVES,
                          expected_primitive_measurements=count * len(args.primitives),
                          readiness='output_tile_epochs_zeroed_before_each_producer',
                          preparation='producer_variants_share_identical_predecoded_weight_and_control_reset',
                          copy_preparation='read_only_payload_initialized_before_warmup_and_poison_not_each_sample',
                          staging='producer_output_or_deterministic_copy_input_in_bound_local_staging')
            report['timing']['excluded'].append('finalize')
        elif qkv:
            report.update(operator='qkv_backward', phase='data', weight_mode='deferred',
                          scope='independent_services_no_DQ_no_W_no_finalize_not_B_total',
                          primitive_ids=QKV_PRIMITIVES,
                          expected_primitive_measurements=count * len(args.primitives),
                          readiness='int32_epoch_preloaded_after_reset_into_actual_strided_flag_prefix',
                          preparation='bare_and_ready_compute_share_identical_preparation_each_sample',
                          staging='peer_arena_data_bound_by_Arguments_not_unbound_tensors_staging')
            report['timing']['excluded'].extend(['W', 'finalize', 'integer_ready_prefix_restore'])
        if rank == 0:
            bench.checkpoint(args.output, report)
        for case in cases:
            state = bench.OperatorCase(case, library, cuda)
            buffers = None
            try:
                if qkv:
                    state.argument.beta = 1
                    state.argument.geometry[11], state.argument.geometry[12] = 1, 0
                configurations = candidate_configurations(state, args, dist)
                buffers = (allocate_forward_buffers(state, configurations, torch) if forward else
                           allocate_qkv_buffers(state, configurations, torch) if qkv else None)
                for requested, config, resources in configurations:
                    cuda.check(library.fuse_mxfp8_test_set_comm_ctas(requested))
                    state.refresh_config(requested_comm_ctas=requested)
                    if any(state.config[key] != config[key] for key in state.config):
                        raise RuntimeError('Native config changed since primitive metadata prepass')
                    sample = dict(case=case, config=config, work=geometry_work(case, args.operator), primitives={})
                    for primitive in args.primitives:
                        sample['primitives'][primitive] = sample_primitive(
                            state, primitive, args, runtime, resources.get(primitive), buffers)
                    report['samples'].append(sample)
                    if rank == 0:
                        bench.checkpoint(args.output, report)
                        values = ', '.join(f'{key}={value["timing"]["p50_us"]:.2f} us'
                                           for key, value in sample['primitives'].items())
                        print(f'{case["id"]} / c{config["comm_ctas"]}: {values}; correctness PASS', flush=True)
            finally:
                cuda.check(library.fuse_mxfp8_test_set_comm_ctas(0))
                state.close()
                buffers = None
                del state
                gc.collect()
                torch.cuda.empty_cache()
        if rank == 0:
            report['complete'] = len(report['samples']) == count
            bench.checkpoint(args.output, report)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
