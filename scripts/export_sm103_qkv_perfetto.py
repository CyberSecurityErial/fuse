#!/usr/bin/env python3
"""Export existing SM103 CTA telemetry as a protocol-aligned Perfetto JSON.

No cross-GPU globaltimer subtraction. This is a diagnostic trace, not a
Tensor Core utilization measurement or an MPI performance sample.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path


TRANSFER_ENDPOINTS = dict(
    version='gpu_endpoints_v1',
    gpu_ids='Logical ranks matching trace GPU process IDs, not PCI device ordinals',
    local_G2S='src_gpu GMEM -> same GPU communication SMEM; route_peer is the eventual destination',
    peer_S2G='src_gpu communication SMEM -> dst_gpu GMEM; src_gpu == dst_gpu is a local store',
    completion='G2S ends at mbarrier completion; S2G ends at source SMEM read completion, NOT destination arrival',
    bytes='BF16 payload bytes per copy, not measured NVLink wire bytes')

# Preserve the old four-stage trace layout; current publication omits the SC fence.
MXFP8_PUBLICATION_PHASES = (
    ('fence_done', 'W device fence'),
    ('warp_join_done', 'W first warp join'),
    ('arrival_done', 'W arrival counter (address + atomic return)'),
    ('end', 'W ready publication + final warp join'),
)
MXFP8_NO_SC_PUBLICATION_PHASES = MXFP8_PUBLICATION_PHASES[1:]


def transfer_args(rank, peer, task, rows, columns, phase):
    assert phase in ('g2s', 's2g') and rows > 0 and columns > 0
    return dict(task=task, src_gpu=rank, dst_gpu=rank if phase == 'g2s' else peer,
                route_peer=peer, bytes=rows * columns * 2)


def annotate_transfer_events(events, world):
    """Join the exporter's ready/G2S/S2G triplets; never infer receive times."""
    pending, count = {}, 0
    for event in events:
        name = event['name']
        if name == 'ready wait':
            args = event['args']
            key = event['pid'], args['task']
            assert key not in pending
            assert 0 <= key[0] < world and 0 <= args['peer'] < world
            pending[key] = (args, False)
        elif name in ('local G2S', 'peer S2G (SMEM read complete)'):
            key = event['pid'], event['args']['task']
            args, seen_g2s = pending[key]
            phase = 'g2s' if name == 'local G2S' else 's2g'
            assert seen_g2s == (phase == 's2g'), 'Missing/duplicate transfer phase'
            event['args'].update(transfer_args(key[0], args['peer'], key[1],
                                             args['rows'], args['columns'], phase))
            if phase == 's2g': del pending[key]
            else: pending[key] = (args, True)
            count += 1
    assert not pending, 'Incomplete ready/G2S/S2G triplet'
    return count


def rewrite_trace(path, payload, suffix):
    temporary = path.with_suffix(suffix)
    created = False
    try:
        with temporary.open('x') as stream:
            created = True
            json.dump(payload, stream, separators=(',', ':'))
        temporary.replace(path)
    finally:
        if created and temporary.exists(): temporary.unlink()


def annotate_trace(path):
    with path.open() as stream:
        payload = json.load(stream)
    assert payload['metadata']['schema'] == 'fuse_sm103_qkv_perfetto_v2'
    count = annotate_transfer_events(payload['traceEvents'], int(payload['metadata']['config']['world']))
    payload['metadata']['transfer_endpoints'] = TRANSFER_ENDPOINTS
    rewrite_trace(path, payload, '.endpoints-tmp')
    print(f'Annotated {path.name}: {count} transfer spans; timestamps/tracks unchanged')


def grouped_tid(tid, comm, mxfp8=False):
    """Reserve nine adjacent tracks per route CTA: role, then eight warps."""
    if mxfp8 and tid >= 1000:
        cta, warp = divmod(tid - 1000, 8)
        return 100 + cta * 32 + 1 + warp * 3
    if mxfp8 and tid >= 100:
        return 100 + (tid - 100) * 32
    if tid >= 1000:
        cta, warp = divmod(tid - 1000, 8)
        assert cta < comm
        return 100 + cta * 9 + 1 + warp
    if tid >= 100:
        cta = tid - 100
        return 100 + (cta * 9 if cta < comm else comm * 9 + cta - comm)
    return tid


def group_role_tracks(events, comm, mxfp8=False, route_warps=8, weight_schedule=None):
    ordering = []
    for event in events:
        if 'tid' not in event:
            continue
        old_tid = event['tid']
        event['tid'] = grouped_tid(old_tid, comm, mxfp8)
        if event['ph'] == 'M' and event['name'] == 'thread_name':
            if old_tid >= 1000:
                cta, warp = divmod(old_tid - 1000, 8)
                event['args']['name'] = f'CTA {cta:03d} / {warp+1} warp {warp}: ready | G2S | S2G | drain'
            elif old_tid >= 100:
                cta = old_tid - 100
                role = ('MXFP8 route + quant role' if mxfp8 and
                        (route_warps == 4 or weight_schedule == 'warp_then_route_v1')
                        else 'QKV route role') if cta < comm else 'GEMM role'
                event['args']['name'] = f'CTA {cta:03d} / 0 {role}'
            ordering.append(dict(ph='M', name='thread_sort_index',
                pid=event['pid'], tid=event['tid'], args={'sort_index': event['tid']}))
    events.extend(ordering)


def resolve_route_schedule(job, orders):
    """Do not reinterpret historical fixed 4+4 captures as one-way handoff."""
    preparation = job.get('mxfp8_weight_preparation')
    assert preparation in (None, 'comm', 'all', 'comm_warp'), 'Unknown weight preparation'
    if orders:
        assert set(orders) == set(range(int(job['world']))), 'Missing rank route order'
        widths = {int(row.get('route_warps', 8)) for row in orders.values()}
        assert len(widths) == 1, 'Inconsistent rank route_warps'
        route_warps = widths.pop()
        assert all(int(row['comm_sm']) == int(job['comm_sm']) for row in orders.values()), 'Route CTA count mismatch'
        schedules = {row.get('weight_schedule') for row in orders.values()}
        assert len(schedules) == 1, 'Inconsistent rank weight_schedule'
        schedule = schedules.pop()
    else:
        assert preparation != 'comm_warp', 'Missing comm_warp route schedule metadata'
        route_warps, schedule = 8, None
    assert route_warps in (4, 8), 'Invalid route_warps'
    assert schedule in (None, 'warp_then_route_v1'), 'Unknown weight_schedule'
    if schedule == 'warp_then_route_v1':
        assert job.get('mxfp8') and preparation in (None, 'comm_warp') and route_warps == 8, 'Invalid quant-to-route handoff metadata'
    elif preparation is not None:
        assert route_warps == (4 if preparation == 'comm_warp' else 8), 'Weight preparation/route_warps mismatch'
    assert route_warps != 4 or job.get('mxfp8'), 'Split route warps require MXFP8'
    return route_warps, 'fixed_split_legacy_v1' if route_warps == 4 else schedule


def append_mxfp8_events(events, log_path, job, origins, records, route_warps=8,
                       weight_schedule=None, quant_ends=None):
    """Observed quantization chunks and actual new-panel acquires, no inferred MMA spans."""
    world, comm = int(job['world']), int(job['comm_sm'])
    n = (int(job['q_heads']) + 2 * int(job['kv_heads'])) * int(job['head_dim'])
    k, m = int(job['hidden']), int(job['global_seq']) // world
    if job.get('fused_direction') == 'oproj':
        n, k = int(job['hidden']), int(job['q_heads']) * int(job['head_dim'])
    panels, steps = (n + 255) // 256, 256 * (k // 32) // 32
    seen, releases, waits, tracks = set(), {}, set(), set()
    publication_subphase_chunks = 0
    publication_protocol_chunks = {}
    declared_protocols, worker_panels, panel_contributions = set(), {}, {}
    sums = [0] * world
    def track(rank, tid, name):
        if (rank, tid) not in tracks:
            tracks.add((rank, tid))
            events.extend([dict(ph='M', name='thread_name', pid=rank, tid=tid, args=dict(name=name)),
                dict(ph='M', name='thread_sort_index', pid=rank, tid=tid, args=dict(sort_index=tid))])
    def span(rank, tid, name, begin, end, attrs):
        events.append(dict(ph='X', name=name, cat='fuse.mxfp8', pid=rank, tid=tid,
            ts=(begin-origins[rank])/1000, dur=(end-begin)/1000, args=attrs))
    with log_path.open() as stream:
        for line in stream:
            if line.startswith('precision,mxfp8,'):
                precision = dict(part.split('=', 1) for part in line.strip().split(',')[2:])
                if 'publication_protocol' in precision:
                    declared_protocols.add(precision['publication_protocol'])
                continue
            if not line.startswith(('profile_mxfp8_quant,', 'profile_mxfp8_wait,')): continue
            r = {key:int(value) for key,value in (p.split('=',1) for p in line.strip().split(',')[1:])}
            rank, cta, warp, index = (r[key] for key in ('rank','cta','warp','index'))
            assert 0 <= rank < world and 0 <= cta < 148 and 0 <= warp < 8
            role = records[rank, cta]
            assert int(role['start']) <= r['begin'] <= r['end'] <= int(role['role_done'])
            if line.startswith('profile_mxfp8_quant,'):
                if route_warps == 4 or weight_schedule == 'warp_then_route_v1':
                    assert cta < comm and 4 <= warp < 8, 'Quant record outside dedicated quant warps'
                if quant_ends is not None:
                    key = rank, cta, warp
                    quant_ends[key] = max(quant_ends.get(key, 0), r['end'])
                assert (rank,index) not in seen
                seen.add((rank,index))
                panel, step = divmod(index,steps)
                assert panel == r['panel'] and 0 <= panel < panels
                rows = min(256, ((n+127)//128)*128 - panel*256)
                assert step < rows*(k//32)//32 and r['groups'] == 32
                assert r['begin'] <= r['quant_done'] <= r['end']
                phases, protocol = (), 'legacy_unsplit'
                publication_name = 'W fence + arrival counter + warp join'
                phase_fields = [field for field,_ in MXFP8_PUBLICATION_PHASES[:-1]]
                if 'arrival_chunks' in r:
                    protocol = 'warp_panel_acq_rel_v3'
                    delta = r['arrival_chunks']
                    assert delta >= 0 and 'fence_done' not in r, 'Invalid aggregate publication record'
                    assert 'warp_join_done' in r and 'arrival_done' in r, 'Incomplete W publication timestamps'
                    worker = worker_panels.setdefault((rank, cta, warp, panel),
                        dict(chunks=0, last_step=-1, contribution_step=None, delta=0))
                    worker['chunks'] += 1
                    worker['last_step'] = max(worker['last_step'], step)
                    if delta:
                        assert worker['contribution_step'] is None, 'Duplicate worker-panel contribution'
                        worker.update(contribution_step=step, delta=delta)
                        panel_contributions[rank, panel] = panel_contributions.get((rank, panel), 0) + delta
                        phases = MXFP8_NO_SC_PUBLICATION_PHASES
                        publication_name = 'W arrival counter + warp join'
                    else:
                        assert r['warp_join_done'] == r['arrival_done'] == r['release'] == 0, 'Non-contributing chunk has publication timestamps'
                        publication_name = 'W local bookkeeping (no publication)'
                elif any(field in r for field in phase_fields):
                    assert 'warp_join_done' in r and 'arrival_done' in r, 'Incomplete W publication timestamps'
                    if 'fence_done' in r:
                        phases, protocol = MXFP8_PUBLICATION_PHASES, 'sc_fence_acq_rel_v1'
                    else:
                        phases, protocol = MXFP8_NO_SC_PUBLICATION_PHASES, 'warp_join_acq_rel_v2'
                        publication_name = 'W arrival counter + warp join'
                if phases:
                    boundaries = [r['quant_done']] + [r[field] for field,_ in phases]
                    assert all(a <= b for a,b in zip(boundaries,boundaries[1:])), 'Unordered W publication timestamps'
                publication_protocol_chunks[protocol] = publication_protocol_chunks.get(protocol, 0) + 1
                tid = 100 + cta*32 + 2 + warp*3
                track(rank,tid,f'CTA {cta:03d} / warp {warp}: weight quantization + publication')
                attrs = dict(panel=panel, chunk=step, n_begin=panel*256,
                    group_begin=step*32, groups=r['groups'], group_k=32, values=r['groups']*32,
                    input='BF16 master W', output='MXFP8 E4M3 + UE8M0', cta=cta, warp=warp,
                    publication_protocol=protocol)
                if 'arrival_chunks' in r:
                    attrs['arrival_chunks'] = r['arrival_chunks']
                span(rank,tid,'W quantize BF16 -> MXFP8',r['begin'],r['quant_done'],attrs)
                span(rank,tid,publication_name,r['quant_done'],r['end'],attrs)
                if phases:
                    # Same tid and enclosing interval: Perfetto nests these directly
                    # below this warp's compound publication span, without new rows.
                    # Missing fence_done identifies the no-SC protocol, not a zero-time fence.
                    for (field,name),begin in zip(phases,boundaries):
                        span(rank,tid,name,begin,r[field],attrs)
                    publication_subphase_chunks += 1
                sums[rank] += r['quant_done']-r['begin']
                if r['release']:
                    assert r['quant_done'] <= r['release'] <= r['end'] and (rank,panel) not in releases
                    if 'arrival_done' in r:
                        assert r['arrival_done'] <= r['release'], 'W ready stamp precedes arrival atomic'
                    releases[rank,panel] = r['release']
                    events.append(dict(ph='i',s='t',name='W panel ready published (post-store stamp)',
                        pid=rank,tid=tid,ts=(r['release']-origins[rank])/1000,args=attrs))
            else:
                assert (rank,index) not in waits and comm <= cta < 148
                waits.add((rank,index))
                assert 0 <= r['m'] < (m+127)//128 and index == r['m']*panels+r['panel']
                assert 0 <= r['panel'] < panels
                tid = 100 + cta*32 + 1
                track(rank,tid,f'CTA {cta:03d} / GEMM producer: weight panel acquire')
                span(rank,tid,'GEMM waits W panel + warp join + proxy fence',r['begin'],r['end'],
                    dict(m_tile=r['m'],n_panel=r['panel'],cta=cta,warp=warp,
                         cached_panel_uses='not rechecked; no synthetic wait events'))
    expected_steps = ((n+127)//128)*128*(k//32)//32
    assert len(seen) == world*expected_steps, ('Incomplete weight quantization',len(seen),world*expected_steps)
    assert len(releases) == world*panels and waits, 'Missing panel releases or GEMM acquires'
    if declared_protocols:
        assert len(declared_protocols) == 1 and set(publication_protocol_chunks) == declared_protocols, 'Declared/recorded publication protocol mismatch'
    if worker_panels:
        assert publication_protocol_chunks == {'warp_panel_acq_rel_v3':len(seen)}, 'Mixed aggregate and legacy publication records'
        for worker in worker_panels.values():
            assert worker['contribution_step'] == worker['last_step'], 'Contribution is not the worker-panel final chunk'
            assert worker['delta'] == worker['chunks'], 'Worker-panel contribution does not cover its valid chunks'
        for rank in range(world):
            for panel in range(panels):
                rows = min(256, ((n+127)//128)*128 - panel*256)
                assert panel_contributions.get((rank,panel),0) == rows*(k//32)//32, 'Incorrect panel contribution total'
    return dict(quant_chunks=len(seen), panel_releases=len(releases), observed_acquires=len(waits),
        publication_subphase_chunks=publication_subphase_chunks,
        publication_protocol_chunks=publication_protocol_chunks,
        aggregate_publications=len(worker_panels),
        quant_warp_sum_us=[v/1000 for v in sums],
        interpretation='Warp-time sums overlap, not critical-path latency. Release stamp is after the store; '
                       'a consumer can observe ready before this post-store stamp. Activation is already MXFP8.')


def reorder_trace(path):
    """Change presentation only; retain all sample values and provenance."""
    with path.open() as stream:
        payload = json.load(stream)
    if payload['metadata'].get('track_layout') == 'role_then_own_warps_v1':
        print(f'Already grouped: {path.name}')
        return
    assert payload['metadata']['schema'] == 'fuse_sm103_qkv_perfetto_v2'
    events = payload['traceEvents']
    count = len(events)
    group_role_tracks(events, int(payload['metadata']['config']['comm_sm']))
    payload['metadata']['track_layout'] = 'role_then_own_warps_v1'
    rewrite_trace(path, payload, '.layout-tmp')
    print(f'Grouped {path.name}: {count} existing events preserved')


def export(run, output):
    receipt = json.loads((run / 'fetched.json').read_text())
    assert receipt['state'] == 'succeeded' and receipt['exit_code'] == 0
    digest = hashlib.sha256()
    with (run / 'artifacts.tar.gz').open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    assert digest.hexdigest() == receipt['artifact_sha256']
    control = run / f"artifacts-attempt{receipt['attempt']}" / 'control'
    job = json.loads((control / 'job.json').read_text())
    assert job['profile'] and job['directions'] == 'qkv' and not job['mpi']
    log_path = control / 'attempt1.log'
    with log_path.open() as stream:
        selected = [line for line in stream if line.startswith(
            ('profile_cta,', 'profile_host,', 'profile_qkv_order,', 'candidate_verified,', 'PASS:'))]
    assert any('and diagnostic timelines' in line for line in selected)
    assert any('candidate_verified,GEMM_A2A' in line for line in selected)
    records, hosts, orders = {}, {}, {}
    for line in selected:
        if line.startswith('profile_qkv_order,'):
            row = dict(part.split('=',1) for part in line.strip().split(',')[1:])
            assert int(row['rank']) not in orders, 'Duplicate rank route order'
            orders[int(row['rank'])] = row
            continue
        if not line.startswith(('profile_cta,GEMM_A2A,', 'profile_host,GEMM_A2A,')):
            continue
        pieces = line.strip().split(',')
        row = dict(p.split('=', 1) for p in pieces[2:] if '=' in p)
        rank = int(row['rank'])
        if pieces[0] == 'profile_host':
            hosts[rank] = row
        else:
            key = rank, int(row['cta'])
            assert key not in records, 'Duplicate CTA sample'
            records[key] = row
    world, comm = int(job['world']), int(job['comm_sm'])
    route_warps, weight_schedule = resolve_route_schedule(job, orders)
    assert set(hosts) == set(range(world))
    # An omitted CLI option is stored as None, but the benchmark resolves it
    # before capture. Use the recorded detail, not the unresolved request.
    requested_detail = job.get('profile_detail')
    recorded_details = {r.get('profile_detail') for r in records.values()}
    if recorded_details == {None}:
        profile_detail = requested_detail  # Legacy logs lack the resolved field.
    else:
        assert len(recorded_details) == 1 and None not in recorded_details, 'Inconsistent CTA profile detail'
        profile_detail = next(iter(recorded_details))
        assert requested_detail in (None, profile_detail), 'Requested/recorded profile detail mismatch'
    assert profile_detail in ('full', 'cta'), 'Missing or invalid resolved profile detail'
    events = []
    def meta(rank, tid, name):
        events.append(dict(ph='M', name='thread_name', pid=rank, tid=tid, args=dict(name=name)))
    origins, ends = {}, {}
    for rank in range(world):
        ctas = {cta: row for (r, cta), row in records.items() if r == rank}
        assert set(ctas) == set(range(148)), 'Missing physical CTA'
        origin = min(int(r['start']) for r in ctas.values())
        end = max(int(r['end']) for r in ctas.values())
        origins[rank], ends[rank] = origin, end
        events.append(dict(ph='M', name='process_name', pid=rank, args=dict(name=f'GPU rank {rank} (independent time origin)')))
        def span(tid, name, start, stop, **args):
            start, stop = int(start), int(stop)
            assert origin <= start <= stop <= end, (rank, tid, name, start, stop)
            events.append(dict(ph='X', name=name, cat='fuse.qkv', pid=rank, tid=tid,
                               ts=(start-origin)/1000, dur=(stop-start)/1000, args=args))
        meta(rank, 0, 'QKV fused boundary / local envelopes')
        span(0, 'W quantization + GEMM -> A2A diagnostic boundary' if job.get('mxfp8')
             else 'GEMM -> A2A diagnostic boundary', origin, end)
        envelopes = []
        for label, selected in [('compute envelope', [r for c,r in ctas.items() if c>=comm]),
                                ('route envelope (includes ready wait)', [r for c,r in ctas.items() if c<comm])]:
            first = min(int(r['start']) for r in selected)
            last = max(int(r['role_done']) for r in selected)
            tid = len(envelopes)+1
            meta(rank, tid, label); span(tid, label, first, last)
            envelopes.append((first,last))
        first, last = max(x[0] for x in envelopes), min(x[1] for x in envelopes)
        meta(rank, 3, 'Local role overlap (not pure transfer time)')
        if last>=first: span(3, 'compute / route overlap', first, last)
        local_done = max(int(r['role_done']) for r in ctas.values())
        meta(rank, 4, 'Exposed finalization tail')
        span(4, 'all local roles done -> kernel complete', local_done, end)
        zero = ctas[0]
        meta(rank, 5, 'CTA0 finalize phases')
        span(5, 'local roles -> grid sync', local_done, zero['grid_sync_done'])
        span(5, 'fence.sc.sys', zero['grid_sync_done'], zero['fence_done'])
        span(5, 'publish source-complete epochs', zero['fence_done'], zero['publish_done'])
        for peer in range(world):
            tid = 10+peer; meta(rank, tid, f'wait source {peer} epoch')
            span(tid, f'wait source {peer} epoch', zero['publish_done'], zero[f'source_ready{peer}'])
        span(5, 'kernel retire', max(int(zero[f'source_ready{peer}']) for peer in range(world)), end)
        for cta, row in sorted(ctas.items()):
            tid = 100+cta
            label = ('MXFP8 route + quant role' if route_warps == 4 or weight_schedule == 'warp_then_route_v1'
                     else 'QKV route role') if cta < comm else 'GEMM role'
            meta(rank, tid, f'{label} CTA {cta}')
            assert int(row['start']) <= int(row['role_done']) <= int(row['grid_sync_done']) <= int(row['end'])
            span(tid, label, row['start'], row['role_done'])
            span(tid, 'grid barrier / finalize', row['role_done'], row['end'])
        if profile_detail == 'full':
            for cta in range(comm):
                for warp in range(route_warps):
                    meta(rank, 1000 + cta * 8 + warp, f'Route CTA {cta} / warp {warp}: tile phases')
    payload = dict(traceEvents=events, displayTimeUnit='ns', metadata=dict(
        schema='fuse_sm103_qkv_perfetto_v2', diagnostic_only=True, performance_accepted=False,
        profile_detail_requested=requested_detail, profile_detail=profile_detail,
        protocol='PROFILE_PROTOCOL.md: GEMM -> A2A', run_id=job['run_id'], source_id=job['source_id'],
        artifact_sha256=receipt['artifact_sha256'], node=job['node'],
        config={k:job.get(k) for k in ['world','global_seq','hidden','q_heads','kv_heads','head_dim',
                'comm_sm','qkv_policy_list','max_swizzle_size','qkv_raster','host_launch','input_generator',
                'mxfp8_weight_preparation']},
        route_warps=route_warps, weight_schedule=weight_schedule,
        clock='per-GPU globaltimer ns converted to trace microseconds; independent rank origins',
        scope='single-process diagnostic; host enqueue skew can affect cross-rank finalize',
        route_semantics='ready wait includes warp join; G2S ends at mbarrier completion; '
            'S2G ends at SMEM read completion, NOT peer write completion; '
            'final per-warp drain waits all destination writes; gaps include setup and telemetry writes',
        profile_host=hosts, route_order=orders, track_layout='role_then_own_warps_v1',
        transfer_endpoints=TRANSFER_ENDPOINTS))
    group_role_tracks(events, comm, bool(job.get('mxfp8')), route_warps, weight_schedule)
    quant_ends = {}
    if job.get('mxfp8'):
        payload['metadata']['mxfp8'] = append_mxfp8_events(
            events, log_path, job, origins, records, route_warps, weight_schedule, quant_ends)
        payload['metadata']['track_layout'] = 'role_then_own_warp_route_and_quantization_v1'
    output.parent.mkdir(parents=True, exist_ok=True)
    # Stream tile events: largest traces have over a million spans. Do not
    # retain a second full trace/log object in RAM during export.
    counts = [0] * world
    copies, drains = [0] * world, [0] * world
    last_task = [-1] * world
    tasks = ((int(job['global_seq']) // world + 63) // 64) * (int(job['q_heads']) + 2 * int(job['kv_heads']))
    if orders:
        assert set(orders) == set(range(world))
        assert all(int(r['copies']) == tasks for r in orders.values())
    route_workers = comm * route_warps
    previous = [[0] * route_workers for _ in range(world)]
    emitted = len(events)
    tile = re.fullmatch(r'm(\d+)n(\d+).*', job['qkv_policy_list'])
    assert tile, 'Detailed export requires an explicit GEMM tile'
    tile_m, tile_n = map(int, tile.groups())
    created = False
    try:
        with output.open('x') as stream:
            created = True
            stream.write('{"traceEvents":[')
            for index, event in enumerate(events):
                if index: stream.write(',')
                json.dump(event, stream, separators=(',', ':'))
            with log_path.open() as lines:
                for line in lines:
                    if not line.startswith('profile_qkv_route,'): continue
                    r = {k:int(v) for k,v in (part.split('=',1) for part in line.strip().split(',')[1:])}
                    rank, task, cta, warp = (r[k] for k in ('rank','task','cta','warp'))
                    assert 0 <= rank < world and task > last_task[rank]
                    if not orders: assert task == counts[rank]
                    last_task[rank] = task
                    assert 0 <= cta < comm and 0 <= warp < route_warps
                    slots = int(orders[rank]['slots']) if orders else tasks
                    owner = (task - slots if r['drain'] else task) % route_workers
                    assert (cta, warp) == (owner % comm, owner // comm)
                    assert bool(r['drain']) == (task >= slots)
                    assert task < slots + route_workers
                    counts[rank] += 1
                    assert int(records[rank,cta]['start']) <= r['begin'] <= r['s2g_read_done'] <= int(records[rank,cta]['role_done'])
                    assert previous[rank][owner] <= r['begin']
                    previous[rank][owner] = r['s2g_read_done']
                    if weight_schedule == 'warp_then_route_v1':
                        # Same physical warp only: other warps intentionally overlap.
                        # Checking the last quant end against every route begin also
                        # rejects an illegal return to quantization after routing.
                        assert quant_ends.get((rank, cta, warp), 0) <= r['begin'], 'Route precedes quantization handoff'
                    if r['drain']:
                        drains[rank] += 1
                        phases = [('all peer writes drain', r['begin'], r['s2g_read_done'], {})]
                    else:
                        copies[rank] += 1
                        times = [r[k] for k in ('begin','ready','g2s_begin','g2s_done','s2g_begin','s2g_read_done')]
                        assert times == sorted(times)
                        attrs = {k:r[k] for k in ('task','row','column','rows','columns','peer')}
                        attrs.update(segment='QKV'[r['segment']], producer_m=r['row']//tile_m,
                                     producer_n_first=r['column']//tile_n,
                                     producer_n_last=(r['column']+r['columns']-1)//tile_n)
                        phases = [('ready wait', r['begin'], r['ready'], attrs),
                                  ('local G2S', r['g2s_begin'], r['g2s_done'],
                                   transfer_args(rank, r['peer'], task, r['rows'], r['columns'], 'g2s')),
                                  ('peer S2G (SMEM read complete)', r['s2g_begin'], r['s2g_read_done'],
                                   transfer_args(rank, r['peer'], task, r['rows'], r['columns'], 's2g'))]
                        if job.get('qkv_postprocess') and r['segment'] < 2:
                            assert r['g2s_done'] <= r['post_begin'] <= r['post_math_done'] <= r['post_end'] <= r['s2g_begin']
                            mode = 'Q/K RMSNorm + RoPE' if job['qkv_postprocess'] == 'qknorm_rope' else 'Q/K RoPE'
                            phases[2:2] = [(mode + ' arithmetic + SMEM writes', r['post_begin'], r['post_math_done'], attrs),
                                           ('postprocess SMEM proxy publication', r['post_math_done'], r['post_end'], attrs)]
                        else:
                            assert not any(r.get(k, 0) for k in ('post_begin', 'post_math_done', 'post_end'))
                    for name, start, stop, attrs in phases:
                        stream.write(',')
                        json.dump(dict(ph='X', name=name, pid=rank, tid=grouped_tid(1000+cta*8+warp, comm, bool(job.get('mxfp8'))),
                            ts=(start-origins[rank])/1000, dur=(stop-start)/1000, args=attrs),
                            stream, separators=(',', ':'))
                        emitted += 1
            expected = tasks + route_workers if profile_detail == 'full' else 0
            assert counts == [expected] * world, ('Incomplete route records', counts, expected)
            assert copies == [tasks if profile_detail == 'full' else 0] * world, 'Incomplete route copies'
            assert drains == [route_workers if profile_detail == 'full' else 0] * world, 'Incomplete route drains'
            stream.write('],"displayTimeUnit":"ns","metadata":')
            payload['metadata']['route_records_per_rank'] = counts
            if weight_schedule == 'warp_then_route_v1':
                payload['metadata']['weight_handoff_validated'] = profile_detail == 'full'
            json.dump(payload['metadata'], stream, separators=(',', ':'))
            stream.write('}')
    except Exception:
        # Only remove the incomplete file created by this export, never an
        # existing delivery (open('x') refuses those).
        if created:
            output.unlink()
        raise
    print(json.dumps(dict(file=str(output), ranks=world, ctas=len(records), events=emitted)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, nargs='?')
    parser.add_argument('output', type=Path, nargs='?')
    parser.add_argument('--reorder-trace', type=Path,
                        help='Regroup an existing JSON in place; no GPU work')
    parser.add_argument('--annotate-trace', type=Path,
                        help='Add GPU endpoints/bytes to an existing JSON in place; no GPU work')
    args = parser.parse_args()
    if args.annotate_trace:
        if args.run or args.output or args.reorder_trace: parser.error('Do not mix annotation and other operations')
        annotate_trace(args.annotate_trace)
    elif args.reorder_trace:
        if args.run or args.output: parser.error('Do not mix export and reorder arguments')
        reorder_trace(args.reorder_trace)
    else:
        if not args.run or not args.output: parser.error('Export requires run and output')
        export(args.run, args.output)
