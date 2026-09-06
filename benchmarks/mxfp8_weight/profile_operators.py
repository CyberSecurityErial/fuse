"""Bounded diagnostic CTA traces for MXFP8-weight 2F2B, never formal timings.

Reuse the formal runner's exact matrix, CUDA IPC arena and independent
references. Reuse each case's allocations across CTA candidates. Aggregate
mode retains only compact rank summaries, not traces or per-tile raw dumps.
The native bridge must be built with FUSE_ENABLE_PROFILING=ON.
"""
import argparse
import ctypes as ct
from datetime import timedelta
import gc
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import operator_bench as bench
from profile_report import DESCRIPTIONS, rank_profile, write_profile

ROOT = Path(__file__).resolve().parents[2]


class CtaTimeline(ct.Structure):
    _fields_ = [(field, ct.c_uint64) for field in
                ('start', 'end', 'active_start', 'role_done', 'grid_sync_done',
                 'fence_done', 'publish_done')] + [('source_ready', ct.c_uint64 * 8)]


def decode_timeline(tensor, count):
    raw = tensor.cpu().numpy().tobytes()
    entries = (CtaTimeline * count).from_buffer_copy(raw)
    return [{name: list(getattr(entry, name)) if name == 'source_ready' else getattr(entry, name)
             for name, _ in CtaTimeline._fields_} for entry in entries]


def qkv_backward_grid_ctas(config):
    """Static SM90 grid, not the allocation capacity or a count of nonzero rows.

    Match MonolithicGemm + CUTLASS get_grid_shape for the existing AlongN,
    swizzle1, L1, cluster (1|2,1,1) path. Both clusters divide the SM90
    scheduler's 18-SM GPC bound. Reject unsupported layouts instead of
    guessing a grid from missing timestamps.
    """
    m, n, _ = config['forward_or_data_mnk']
    cm, sm, comm = (config[key] for key in ('cluster_m', 'sm_count', 'comm_ctas'))
    tm, tn = config['tile_m'], config['tile_n']
    if (cm not in (1, 2) or config['raster'] != 'n' or config['swizzle'] != 1 or
            min(m, n, tm, tn, comm) <= 0 or not comm < sm or
            comm % cm or (sm - comm) % cm):
        raise ValueError('Unsupported QKV backward diagnostic grid geometry')
    m_tiles, n_tiles = (m + tm - 1) // tm, (n + tn - 1) // tn
    padded_tiles = ((m_tiles + cm - 1) // cm) * cm * n_tiles
    return comm + min(sm - comm, padded_tiles)


def qkv_backward_timeline(config, ctas):
    count = qkv_backward_grid_ctas(config)
    if len(ctas) != config['sm_count']:
        raise ValueError('QKV backward timeline capacity differs from SM count')
    if any(any(value) if isinstance(value, list) else value
           for cta in ctas[count:] for value in cta.values()):
        raise ValueError('Unexpected timestamps beyond the predicted native grid')
    return dict(config, grid_ctas=count), ctas[:count]


def comm_candidates(value):
    try:
        candidates = [int(item) for item in value.split(',')]
    except ValueError as error:
        raise argparse.ArgumentTypeError('Communication CTAs must be comma-separated integers') from error
    if len(set(candidates)) != len(candidates) or any(item < 0 or item % 2 for item in candidates):
        raise argparse.ArgumentTypeError('Use unique zero or positive even communication CTA counts')
    return candidates


def validate_sample_limit(cases, candidates, limit):
    count = len(cases) * len(candidates)
    if not 1 <= count <= limit:
        raise ValueError(f'Profile sample limit exceeded: {len(cases)} shapes × '
                         f'{len(candidates)} CTA candidates = {count}; --max-cases={limit}')
    return count


def profile_prefix(output, case_id, requested, multiple):
    name = case_id.replace('/', '_')
    return output / (f'{name}_c{requested}' if multiple else name)


def calibration_record(case, operator, requested, ranks, correctness):
    world = case['cp']
    if len(ranks) != world or sorted(row['rank'] for row in ranks) != list(range(world)):
        raise ValueError('Calibration summary must contain every rank exactly once')
    for row in ranks:
        config = row['config']
        if config['world_size'] != world or config['requested_comm_ctas'] != requested:
            raise ValueError('Calibration summary configuration disagrees with its request')
        if requested and config['comm_ctas'] != requested:
            raise ValueError('Manual communication CTA request was silently changed')
        if 'ctas' in row or 'traceEvents' in row:
            raise ValueError('Aggregate calibration must not retain raw CTA arrays or Perfetto events')
    return dict(case=case, operator=operator, requested_comm_ctas=requested,
                ranks=sorted(ranks, key=lambda row: row['rank']), correctness=correctness)


def sample_profile(state, args, requested, timeline, markers):
    state.runtime.check(state.library.fuse_mxfp8_test_set_comm_ctas(requested))
    state.refresh_config(requested_comm_ctas=requested)
    count = state.config['sm_count']
    if timeline.numel() != count * ct.sizeof(CtaTimeline):
        raise ValueError('Timeline allocation no longer matches selected SM count')
    if requested and state.config['comm_ctas'] != requested:
        raise ValueError('Manual communication CTA request was silently changed')
    mode = args.weight_mode if state.op >= 2 else None
    phase = 'total' if state.op >= 2 else 'forward'
    state.argument.beta = int(mode == 'deferred')
    with torch.cuda.stream(state.stream):
        reset = state.prepare(phase, mode, poison=True)

        def prepare():
            reset()
            timeline.zero_()
            markers.zero_()

        def call():
            state.runtime.check(state.library.fuse_mxfp8_test_profile(
                ct.byref(state.argument), timeline.data_ptr(), count, markers.data_ptr()))

        # Refresh the request before capture; each candidate gets its own
        # graph, while tensors, references and CUDA IPC mappings stay shared.
        invocation = runtime.Invocation(call, args.launch, 2, prepare, state.stream)
        # Invocation already warms the selected launch mode twice. Extra
        # warmups reuse that capture; they do not create extra trace files.
        for _ in range(getattr(args, 'warmup', 2) - 2):
            invocation.once()
        invocation.once()
        correctness = state.verify(phase, mode)
        marker_values = markers.cpu().tolist()
        local = dict(rank=state.rank,
                     config=dict(state.config, world_size=args.cp, beta=int(mode == 'deferred')),
                     ctas=decode_timeline(timeline, count),
                     markers=dict(zip(('dq_start', 'dq_end', 'w_start', 'w_end'), marker_values)))
        if state.op == 2:
            local['config'], local['ctas'] = qkv_backward_timeline(local['config'], local['ctas'])
        del invocation
    if args.aggregate_only:
        # Validate and reduce on each rank before the CPU collective. The
        # shared reporter defines all spans; its transient trace is discarded.
        local = rank_profile(state.name, local)[0]
    ranks = [None] * args.cp
    dist.all_gather_object(ranks, local)
    return ranks, correctness


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cp', type=int, choices=(4, 8), required=True)
    parser.add_argument('--operators', default=','.join(bench.OPERATORS))
    parser.add_argument('--model', default='production_qwen_dense')
    parser.add_argument('--seqs', default='131072')
    parser.add_argument('--case-ids', type=Path)
    parser.add_argument('--launch', choices=('eager', 'graph'), default='graph')
    parser.add_argument('--warmup', type=int, default=2,
                        help='launch-mode warmups including the existing two; Graph also has two pre-capture eager setup calls')
    parser.add_argument('--comm-ctas', type=comm_candidates, default='0',
                        help='test-only even CTA requests; zero uses auto or the explicitly supplied model')
    parser.add_argument('--oproj-comm-model', type=Path,
                        help='explicit frozen OProj service model; omitted keeps the existing baseline')
    parser.add_argument('--qkv-comm-model', type=Path,
                        help='explicit frozen QKV forward service model; omitted keeps the existing baseline')
    parser.add_argument('--weight-mode', choices=('immediate', 'deferred'), default='immediate')
    parser.add_argument('--max-cases', type=int, default=4,
                        help='bound on shapes × CTA candidates, including aggregate-only samples')
    parser.add_argument('--aggregate-only', action='store_true',
                        help='write only calibration_summary.json, never Perfetto or raw CTA arrays')
    parser.add_argument('--library', type=Path,
                        default=ROOT / 'build-mxfp8-profile/libfuse_mxfp8_torch_bridge.so')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/mxfp8_weight/profile/current')
    args = parser.parse_args(argv)
    if args.max_cases < 1:
        parser.error('--max-cases must be positive')
    if args.warmup < 2:
        parser.error('--warmup must be at least 2; setup/capture warmups are retained')
    return args


def main():
    args = arguments()
    if int(os.environ.get('WORLD_SIZE', 1)) != args.cp:
        raise RuntimeError('Use torchrun with one process per GPU and matching --cp')
    cases = bench.select_cases(SimpleNamespace(**vars(args), full=False), args.cp)
    sample_count = validate_sample_limit(cases, args.comm_ctas, args.max_cases)
    aggregate_path = args.output / 'calibration_summary.json'
    if args.aggregate_only and aggregate_path.exists():
        raise FileExistsError(f'Refusing to overwrite {aggregate_path}; choose a new output directory')
    bench.initialize_runtime()
    global torch, dist, runtime
    torch, dist, runtime = bench.torch, bench.dist, bench.rt
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    dist.init_process_group('gloo', timeout=timedelta(minutes=30))
    rank = dist.get_rank()
    library = ct.CDLL(str(args.library))
    library.fuse_mxfp8_test_profiling_enabled.restype = ct.c_int
    if library.fuse_mxfp8_test_profiling_enabled() != 1:
        raise RuntimeError('Diagnostic runner requires a profiling build')
    library.fuse_mxfp8_test_set_comm_ctas.argtypes = [ct.c_int32]
    library.fuse_mxfp8_test_set_comm_ctas.restype = ct.c_int
    library.fuse_mxfp8_test_timeline_bytes.restype = ct.c_uint64
    if library.fuse_mxfp8_test_timeline_bytes() != ct.sizeof(CtaTimeline):
        raise RuntimeError('Native/Python timeline ABI mismatch')
    from test_operators import Arguments
    library.fuse_mxfp8_test_profile.argtypes = [ct.POINTER(Arguments), ct.c_uint64, ct.c_int32, ct.c_uint64]
    library.fuse_mxfp8_test_profile.restype = ct.c_int
    rt = runtime.CudaRuntime()
    rt.check(library.fuse_mxfp8_test_set_comm_ctas(0))
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if max(args.comm_ctas) >= props.multi_processor_count:
        raise ValueError('Every communication CTA request must be smaller than the GPU SM count')
    fingerprints = [None]
    if rank == 0:
        fingerprints[0] = bench.source_hashes(SimpleNamespace(library=args.library, backends='fuse',
                                                            oproj_comm_model=args.oproj_comm_model,
                                                            qkv_comm_model=args.qkv_comm_model))
        for name in ('profile_operators.py', 'operator_profile.cuh', 'profile_report.py'):
            path = Path(__file__).with_name(name)
            fingerprints[0][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    dist.broadcast_object_list(fingerprints, src=0)
    sources = fingerprints[0]
    model = bench.configure_oproj_comm_model(args.oproj_comm_model, args.cp, props.multi_processor_count,
                                             library, rt, sources, workers=dist)
    qkv_model = bench.configure_qkv_forward_comm_model(args.qkv_comm_model, args.cp, props.multi_processor_count,
                                                      library, rt, sources, workers=dist)
    library_key = str(args.library.relative_to(ROOT)) if args.library.is_relative_to(ROOT) else str(args.library)
    metadata = dict(kind='diagnostic_not_formal_performance',
                    library_sha256=sources[library_key],
                    sources=sources,
                    policy_environment={key: os.environ.get(key) for key in bench.POLICY_ENV_VARIABLES},
                    native_device_name=props.name, cc=[props.major, props.minor],
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    launch=args.launch,
                    launch_mode_warmup=args.warmup,
                    warmup_scope='two_setup_warmups_then_extra_captured_replays; Graph_also_has_two_pre_capture_eager_calls; not_formal_10_plus_50_history',
                    requested_comm_ctas=args.comm_ctas[0] if len(args.comm_ctas) == 1 else args.comm_ctas,
                    weight_mode=args.weight_mode,
                    tile_selection='existing_auto_selector_may_change_tile_when_comm_ctas_changes',
                    fixed_tile_single_factor_experiment=False,
                    note='CTA envelopes only; no per-tile raw dump. Marker spans include marker overhead.')
    if model is not None:
        metadata['oproj_comm_model'] = model
    if qkv_model is not None:
        metadata['qkv_forward_comm_model'] = qkv_model
    report = dict(schema='mxfp8-route-compute-calibration-v1', profiling_only=True,
                  metadata=metadata, descriptions=DESCRIPTIONS,
                  clock_origin='independent per rank; cross-rank absolute subtraction is invalid',
                  reported_duration_unit='us',
                  args={key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()},
                  selected_case_ids=[case['id'] for case in cases], expected_samples=sample_count,
                  samples=[], complete=False)
    try:
        if args.aggregate_only and rank == 0:
            bench.checkpoint(aggregate_path, report)
        for case in cases:
            state = bench.OperatorCase(case, library, rt)
            try:
                count = state.config['sm_count']
                with torch.cuda.stream(state.stream):
                    timeline = torch.empty(count * ct.sizeof(CtaTimeline), dtype=torch.uint8, device=state.device)
                    markers = torch.empty(4, dtype=torch.uint64, device=state.device)
                for requested in args.comm_ctas:
                    ranks, correctness = sample_profile(state, args, requested, timeline, markers)
                    if args.aggregate_only:
                        report['samples'].append(calibration_record(case, state.name, requested, ranks, correctness))
                        if rank == 0:
                            bench.checkpoint(aggregate_path, report)
                    elif rank == 0:
                        prefix = profile_prefix(args.output, case['id'], requested, len(args.comm_ctas) > 1)
                        write_profile(state.name, ranks, prefix,
                                      metadata=dict(metadata, case=case, correctness=correctness,
                                                    requested_comm_ctas=requested))
                    if rank == 0:
                        destination = aggregate_path if args.aggregate_only else prefix
                        print(f'{case["id"]} / request c{requested}: diagnostic PASS → {destination}', flush=True)
                del timeline, markers
            finally:
                rt.check(library.fuse_mxfp8_test_set_comm_ctas(0))
                state.close()
                del state
                gc.collect()
                torch.cuda.empty_cache()
        if args.aggregate_only and rank == 0:
            report['complete'] = len(report['samples']) == sample_count
            bench.checkpoint(aggregate_path, report)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
