"""Compact SM90 2F2B role summaries and Perfetto traces, diagnostic only.

Input to ``build_profile(operator, ranks, metadata=None)`` is one dict per GPU:
``rank``; ``config`` with world_size, sm_count and resolved comm_ctas; ``ctas``
in physical CTA order with every A2AGemmCtaTimeline field; and ``markers`` with
dq_start/dq_end and, for backward only, w_start/w_end. All timestamps are integer
%globaltimer nanoseconds on THAT GPU. source_ready always has eight entries.

``write_profile(..., output_prefix, metadata=None)`` retains only
<prefix>_summary.json and <prefix>_perfetto.json, not another full CTA dump.
The marker/envelope/finalize meanings follow backward_mpi_bench.cu. OProj F uses
the existing simpler RoleTelemetryKernel: end closes its role, without invented
grid/finalize stamps. No phase is described as isolated Tensor Core/NVLink time.
"""
import json
from pathlib import Path
import statistics

OPERATORS = ('qkv_forward', 'oproj_forward', 'qkv_backward', 'oproj_backward')
CTA_FIELDS = ('start', 'end', 'active_start', 'role_done', 'grid_sync_done',
              'fence_done', 'publish_done')
DESCRIPTIONS = {
    'DQ': 'MXFP8-to-BF16 weight conversion bounded by two tiny marker kernels; includes marker/launch gaps.',
    'compute': 'Persistent compute-role envelope; includes scheduling, epilogue and any ready waits, not isolated Tensor Core time.',
    'route': 'Persistent route-role envelope; includes queueing and ready waits, not bare NVLink transfer time.',
    'overlap': 'Intersection of the compute and route envelopes, not a sum of CTA durations.',
    'first_ready': 'CTA entry to first observed ready acquire, including startup work; not pure or cumulative ready wait. Absent observations do not mean zero wait.',
    'source_wait': 'Parallel waits share a start; exposed wait is their maximum, never their sum.',
    'W': 'Separate BF16-input/FP32-main_grad GEMM bounded by marker kernels, not a formal W timing sample.',
    'clock': 'Each GPU has its own origin. Equal horizontal positions on different ranks do not imply synchronized clocks.',
}


def integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{label}: expected integer >= {minimum}, got {value!r}')
    return value


def span_us(begin, end):
    if end < begin:
        raise ValueError(f'negative timer span: {begin} -> {end}')
    return (end - begin) / 1000.0


def validate_rank(operator, item):
    rank = integer(item['rank'], 'rank')
    config = item['config']
    world = integer(config['world_size'], 'world_size', 2)
    sm_count = integer(config['sm_count'], 'sm_count', 2)
    comm_ctas = integer(config['comm_ctas'], 'comm_ctas', 1)
    if world > 8 or rank >= world or comm_ctas >= sm_count:
        raise ValueError(f'rank {rank}: invalid world/SM/communication CTA count')
    ctas = item['ctas']
    grid_ctas = integer(config.get('grid_ctas', sm_count), 'grid_ctas', 2)
    if not comm_ctas < grid_ctas <= sm_count or len(ctas) != grid_ctas:
        raise ValueError(f'rank {rank}: expected {grid_ctas} physical CTA records, got {len(ctas)}')
    simple = operator == 'oproj_forward'
    for index, cta in enumerate(ctas):
        label = f'rank {rank} CTA {index}'
        for field in CTA_FIELDS:
            integer(cta[field], f'{label}.{field}')
        sources = cta['source_ready']
        if len(sources) != 8:
            raise ValueError(f'{label}: source_ready must have eight entries')
        for value in sources:
            integer(value, f'{label}.source_ready')
        if cta['start'] == 0 or cta['end'] <= cta['start']:
            raise ValueError(f'{label}: missing or nonpositive CTA lifetime')
        end = cta['end'] if simple else cta['role_done']
        if not cta['start'] <= end <= cta['end']:
            raise ValueError(f'{label}: role_done outside CTA lifetime')
        if cta['active_start'] and not cta['start'] <= cta['active_start'] <= end:
            raise ValueError(f'{label}: first ready observation outside local role')
        if simple:
            if any(cta[k] for k in ('role_done', 'grid_sync_done', 'fence_done', 'publish_done')) or any(sources):
                raise ValueError(f'{label}: OProj F has no role_done/finalize stamps')
        else:
            if not end <= cta['grid_sync_done'] <= cta['end']:
                raise ValueError(f'{label}: invalid cooperative grid-sync stamp')
            if index != 0 and (cta['fence_done'] or cta['publish_done'] or any(sources)):
                raise ValueError(f'{label}: only CTA0 records shared finalize stamps')
    first = min(cta['start'] for cta in ctas)
    last = max(cta['end'] for cta in ctas)
    if not simple:
        roles_done = max(cta['role_done'] for cta in ctas)
        if any(cta['grid_sync_done'] < roles_done for cta in ctas):
            raise ValueError(f'rank {rank}: grid sync precedes the last local role')
        root = ctas[0]
        if not root['grid_sync_done'] <= root['fence_done'] <= root['publish_done']:
            raise ValueError(f'rank {rank}: nonmonotonic finalize stamps')
        if any(not root['publish_done'] <= value <= root['end'] for value in root['source_ready'][:world]):
            raise ValueError(f'rank {rank}: missing or nonmonotonic source-ready stamp')
        if any(root['source_ready'][world:]):
            raise ValueError(f'rank {rank}: nonzero source-ready stamp outside active CP ranks')
    markers = item['markers']
    dq_start = integer(markers['dq_start'], f'rank {rank}.dq_start', 1)
    dq_end = integer(markers['dq_end'], f'rank {rank}.dq_end', 1)
    if not dq_start < dq_end <= first:
        raise ValueError(f'rank {rank}: expected DQ markers before the F/B kernel')
    if operator.endswith('backward'):
        w_start = integer(markers['w_start'], f'rank {rank}.w_start', 1)
        w_end = integer(markers['w_end'], f'rank {rank}.w_end', 1)
        if not last <= w_start < w_end:
            raise ValueError(f'rank {rank}: expected W markers after B completes')
    elif markers.get('w_start', 0) != 0 or markers.get('w_end', 0) != 0:
        raise ValueError(f'rank {rank}: forward cannot contain W markers')
    return first, last


def summarize_waits(values, eligible):
    return dict(eligible_compute_ctas=eligible, observed_compute_ctas=len(values),
                unobserved_compute_ctas=eligible - len(values),
                minimum_us=min(values) if values else None,
                median_us=statistics.median(values) if values else None,
                mean_us=statistics.mean(values) if values else None,
                maximum_us=max(values) if values else None)


def rank_profile(operator, item):
    first, last = validate_rank(operator, item)
    rank, config, ctas, markers = item['rank'], item['config'], item['ctas'], item['markers']
    comm, world = config['comm_ctas'], config['world_size']
    simple, backward = operator == 'oproj_forward', operator.endswith('backward')
    ready_consumer = operator in ('oproj_forward', 'qkv_backward')
    role_end = lambda cta: cta['end'] if simple else cta['role_done']
    route_start = min(cta['start'] for cta in ctas[:comm])
    route_end = max(role_end(cta) for cta in ctas[:comm])
    compute_start = min(cta['start'] for cta in ctas[comm:])
    compute_end = max(role_end(cta) for cta in ctas[comm:])
    overlap_start, overlap_end = max(route_start, compute_start), min(route_end, compute_end)
    overlap = max(0, overlap_end - overlap_start) / 1000.0
    origin = markers['dq_start']
    boundary_end = markers['w_end'] if backward else last
    waits = [span_us(cta['start'], cta['active_start']) for cta in ctas[comm:]
             if ready_consumer and cta['active_start']]
    summary = dict(rank=rank, config=dict(config), origin_ns=origin,
                   observed_ctas=len(ctas), route_ctas=comm, compute_ctas=len(ctas) - comm,
                   dq_us=span_us(markers['dq_start'], markers['dq_end']),
                   dq_to_role_kernel_us=span_us(markers['dq_end'], first),
                   role_kernel_us=span_us(first, last),
                   compute_role_us=span_us(compute_start, compute_end),
                   route_role_us=span_us(route_start, route_end), overlap_us=overlap,
                   first_ready_wait=summarize_waits(waits, len(ctas) - comm) if ready_consumer else None,
                   grid_sync_us=None, system_fence_us=None, publish_us=None,
                   source_wait_us=None, exposed_source_wait_us=None, kernel_retire_us=None,
                   finalize_us=None, local_roles_to_kernel_complete_us=None,
                   b_to_w_marker_us=span_us(last, markers['w_start']) if backward else None,
                   wgrad_marker_us=span_us(markers['w_start'], markers['w_end']) if backward else None,
                   diagnostic_boundary_us=span_us(origin, boundary_end))
    events = [dict(name='process_name', ph='M', pid=rank, tid=0,
                   args=dict(name=f'GPU rank {rank} (independent clock origin)'))]

    def emit(tid, name, category, begin, end):
        if begin < origin:
            raise ValueError(f'rank {rank}: event precedes its own clock origin')
        duration = span_us(begin, end)
        if duration > 0:
            events.append(dict(name=name, cat=category, ph='X', pid=rank, tid=tid,
                               ts=span_us(origin, begin), dur=duration))

    phase = 'B' if backward else 'F'
    emit(90, f'DQ -> {phase}' + (' -> W' if backward else '') + ' diagnostic boundary',
         'diagnostic boundary', origin, boundary_end)
    emit(91, 'MXFP8 -> BF16 weight DQ (marker bounded)', 'weight conversion',
         markers['dq_start'], markers['dq_end'])
    emit(92, f'DQ end marker -> {phase} kernel entry', 'handoff', markers['dq_end'], first)
    emit(100, f'{phase} boundary: BF16 projection + route', 'operator boundary', first, last)
    emit(110, 'compute envelope (includes scheduling and ready waits)', 'compute', compute_start, compute_end)
    emit(111, 'route envelope (includes queueing and ready waits)', 'communication', route_start, route_end)
    if overlap_end > overlap_start:
        emit(112, 'compute / route overlap', 'overlap', overlap_start, overlap_end)
    if not simple:
        root = ctas[0]
        roles_done = max(cta['role_done'] for cta in ctas)
        sources_done = max(root['source_ready'][:world])
        source_waits = [span_us(root['publish_done'], stamp) for stamp in root['source_ready'][:world]]
        summary.update(grid_sync_us=span_us(roles_done, root['grid_sync_done']),
                       system_fence_us=span_us(root['grid_sync_done'], root['fence_done']),
                       publish_us=span_us(root['fence_done'], root['publish_done']),
                       source_wait_us=source_waits, exposed_source_wait_us=max(source_waits),
                       kernel_retire_us=span_us(sources_done, last),
                       finalize_us=span_us(root['grid_sync_done'], last),
                       local_roles_to_kernel_complete_us=span_us(roles_done, last))
        emit(120, 'local roles -> cooperative grid sync', 'finalize', roles_done, root['grid_sync_done'])
        emit(121, 'system fence: make routed writes visible', 'finalize', root['grid_sync_done'], root['fence_done'])
        emit(122, 'publish this source completion epoch', 'finalize', root['fence_done'], root['publish_done'])
        for source in range(world):
            emit(130 + source, f'wait for source {source} completion epoch', 'finalize',
                 root['publish_done'], root['source_ready'][source])
        emit(140, 'kernel retire after every source is visible', 'finalize', sources_done, last)
    if backward:
        emit(150, 'B kernel completion -> WGrad marker', 'handoff', last, markers['w_start'])
        emit(160, 'WGrad GEMM: BF16 inputs / FP32 main_grad (marker bounded)',
             'weight gradient', markers['w_start'], markers['w_end'])
    for index, cta in enumerate(ctas):
        end = role_end(cta)
        if index < comm:
            emit(1000 + index, f'route CTA {index}: communication + ready waits',
                 'communication CTA', cta['start'], end)
        elif ready_consumer and cta['active_start']:
            emit(1000 + index, f'compute CTA {index}: first-ready wait',
                 'ready wait', cta['start'], cta['active_start'])
            emit(1000 + index, f'compute CTA {index}: after first acquire (includes later waits)',
                 'compute CTA', cta['active_start'], end)
        else:
            emit(1000 + index, f'compute CTA {index}: persistent compute role',
                 'compute CTA', cta['start'], end)
        if not simple:
            emit(1000 + index, f'CTA {index}: local role done -> CTA exit',
                 'CTA finalize', end, cta['end'])
    return summary, events


def build_profile(operator, ranks, metadata=None):
    """Return separate compact summary and Perfetto objects; write nothing."""
    if operator not in OPERATORS:
        raise ValueError(f'unsupported operator {operator!r}')
    if not ranks:
        raise ValueError('profile must contain every participating rank')
    world = integer(ranks[0]['config']['world_size'], 'world_size', 2)
    identifiers = [integer(item['rank'], 'rank') for item in ranks]
    if len(ranks) != world or sorted(identifiers) != list(range(world)):
        raise ValueError('missing or duplicate rank records')
    summaries, events = [], []
    for item in sorted(ranks, key=lambda item: item['rank']):
        if item['config']['world_size'] != world:
            raise ValueError('inconsistent world_size across ranks')
        summary, rank_events = rank_profile(operator, item)
        summaries.append(summary)
        events.extend(rank_events)
    common = dict(schema='mxfp8-role-profile-v1', profiling_only=True, operator=operator,
                  world_size=world, clock_origin='independent per rank; cross-rank absolute subtraction is invalid',
                  input_timestamp_unit='ns', reported_duration_unit='us', metadata=dict(metadata or {}))
    summary = dict(common, descriptions=DESCRIPTIONS, ranks=summaries)
    trace_metadata = dict(common, rank_configs=[dict(rank=row['rank'], config=row['config']) for row in summaries])
    trace = dict(displayTimeUnit='ns', metadata=trace_metadata, stripe_descriptions=DESCRIPTIONS, traceEvents=events)
    return dict(summary=summary, trace=trace)


def write_profile(operator, ranks, output_prefix, metadata=None):
    """Write exactly two artifacts; return their paths for the caller's log."""
    profile = build_profile(operator, ranks, metadata)
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = dict(summary=prefix.with_name(prefix.name + '_summary.json'),
                 trace=prefix.with_name(prefix.name + '_perfetto.json'))
    for key, path in paths.items():
        path.write_text(json.dumps(profile[key], ensure_ascii=False, allow_nan=False,
                                   indent=2 if key == 'summary' else None) + '\n')
    return paths
