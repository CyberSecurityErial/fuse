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


def grouped_tid(tid, comm):
    """Reserve nine adjacent tracks per route CTA: role, then eight warps."""
    if tid >= 1000:
        cta, warp = divmod(tid - 1000, 8)
        assert cta < comm
        return 100 + cta * 9 + 1 + warp
    if tid >= 100:
        cta = tid - 100
        return 100 + (cta * 9 if cta < comm else comm * 9 + cta - comm)
    return tid


def group_role_tracks(events, comm):
    ordering = []
    for event in events:
        if 'tid' not in event:
            continue
        old_tid = event['tid']
        event['tid'] = grouped_tid(old_tid, comm)
        if event['ph'] == 'M' and event['name'] == 'thread_name':
            if old_tid >= 1000:
                cta, warp = divmod(old_tid - 1000, 8)
                event['args']['name'] = f'CTA {cta:03d} / {warp+1} warp {warp}: ready | G2S | S2G | drain'
            elif old_tid >= 100:
                cta = old_tid - 100
                role = 'QKV route role' if cta < comm else 'GEMM role'
                event['args']['name'] = f'CTA {cta:03d} / 0 {role}'
            ordering.append(dict(ph='M', name='thread_sort_index',
                pid=event['pid'], tid=event['tid'], args={'sort_index': event['tid']}))
    events.extend(ordering)


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
    temporary = path.with_suffix('.layout-tmp')
    created = False
    try:
        with temporary.open('x') as stream:
            created = True
            json.dump(payload, stream, separators=(',', ':'))
        temporary.replace(path)
    finally:
        if created and temporary.exists():
            temporary.unlink()
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
    assert set(hosts) == set(range(world))
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
        span(0, 'GEMM -> A2A diagnostic boundary', origin, end)
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
            label = 'QKV route role' if cta<comm else 'GEMM role'
            meta(rank, tid, f'{label} CTA {cta}')
            assert int(row['start']) <= int(row['role_done']) <= int(row['grid_sync_done']) <= int(row['end'])
            span(tid, label, row['start'], row['role_done'])
            span(tid, 'grid barrier / finalize', row['role_done'], row['end'])
        if job['profile_detail'] == 'full':
            for cta in range(comm):
                for warp in range(8):
                    meta(rank, 1000 + cta * 8 + warp, f'Route CTA {cta} / warp {warp}: tile phases')
    payload = dict(traceEvents=events, displayTimeUnit='ns', metadata=dict(
        schema='fuse_sm103_qkv_perfetto_v2', diagnostic_only=True, performance_accepted=False,
        protocol='PROFILE_PROTOCOL.md: GEMM -> A2A', run_id=job['run_id'], source_id=job['source_id'],
        artifact_sha256=receipt['artifact_sha256'], node=job['node'],
        config={k:job.get(k) for k in ['world','global_seq','hidden','q_heads','kv_heads','head_dim',
                'comm_sm','qkv_policy_list','max_swizzle_size','qkv_raster','host_launch','input_generator']},
        clock='per-GPU globaltimer ns converted to trace microseconds; independent rank origins',
        scope='single-process diagnostic; host enqueue skew can affect cross-rank finalize',
        route_semantics='ready wait includes warp join; G2S ends at mbarrier completion; '
            'S2G ends at SMEM read completion, NOT peer write completion; '
            'final per-warp drain waits all destination writes; gaps include setup and telemetry writes',
        profile_host=hosts, route_order=orders, track_layout='role_then_own_warps_v1'))
    group_role_tracks(events, comm)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Stream tile events: largest traces have over a million spans. Do not
    # retain a second full trace/log object in RAM during export.
    counts = [0] * world
    last_task = [-1] * world
    tasks = ((int(job['global_seq']) // world + 63) // 64) * (int(job['q_heads']) + 2 * int(job['kv_heads']))
    if orders:
        assert set(orders) == set(range(world))
        assert all(int(r['copies']) == tasks for r in orders.values())
    previous = [[0] * (comm * 8) for _ in range(world)]
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
                    assert 0 <= cta < comm and 0 <= warp < 8
                    slots = int(orders[rank]['slots']) if orders else tasks
                    owner = (task - slots if r['drain'] else task) % (comm * 8)
                    assert (cta, warp) == (owner % comm, owner // comm)
                    assert bool(r['drain']) == (task >= slots)
                    assert task < slots + comm * 8
                    counts[rank] += 1
                    assert int(records[rank,cta]['start']) <= r['begin'] <= r['s2g_read_done'] <= int(records[rank,cta]['role_done'])
                    assert previous[rank][owner] <= r['begin']
                    previous[rank][owner] = r['s2g_read_done']
                    if r['drain']:
                        phases = [('all peer writes drain', r['begin'], r['s2g_read_done'], {})]
                    else:
                        times = [r[k] for k in ('begin','ready','g2s_begin','g2s_done','s2g_begin','s2g_read_done')]
                        assert times == sorted(times)
                        attrs = {k:r[k] for k in ('task','row','column','rows','columns','peer')}
                        attrs.update(segment='QKV'[r['segment']], producer_m=r['row']//tile_m,
                                     producer_n_first=r['column']//tile_n,
                                     producer_n_last=(r['column']+r['columns']-1)//tile_n)
                        phases = [('ready wait', r['begin'], r['ready'], attrs),
                                  ('local G2S', r['g2s_begin'], r['g2s_done'], {'task':task}),
                                  ('peer S2G (SMEM read complete)', r['s2g_begin'], r['s2g_read_done'], {'task':task})]
                    for name, start, stop, attrs in phases:
                        stream.write(',')
                        json.dump(dict(ph='X', name=name, pid=rank, tid=grouped_tid(1000+cta*8+warp, comm),
                            ts=(start-origins[rank])/1000, dur=(stop-start)/1000, args=attrs),
                            stream, separators=(',', ':'))
                        emitted += 1
            expected = tasks + comm * 8 if job['profile_detail'] == 'full' else 0
            assert counts == [expected] * world, ('Incomplete route records', counts, expected)
            stream.write('],"displayTimeUnit":"ns","metadata":')
            payload['metadata']['route_records_per_rank'] = counts
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
    args = parser.parse_args()
    if args.reorder_trace:
        if args.run or args.output: parser.error('Do not mix export and reorder arguments')
        reorder_trace(args.reorder_trace)
    else:
        if not args.run or not args.output: parser.error('Export requires run and output')
        export(args.run, args.output)
