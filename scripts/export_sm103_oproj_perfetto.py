#!/usr/bin/env python3
"""Export SM103 OProj diagnostic telemetry; no kernel changes or GPU launches.

Each GPU has its own time origin. Communication details describe only the final
publishing chunk of a ready region, not every copy issued by the communication
CTA. Release is sampled after publication: acquire can precede that sample.
"""
import argparse
import csv
import hashlib
import json
import tarfile
from collections import defaultdict
from statistics import median
from pathlib import Path


def parse(line):
    return dict(p.split('=', 1) for p in line.strip().split(',')[1:] if '=' in p)


def pipeline_summary(pipelines, stages, peers, n_tiles, world):
    """Use integer globaltimer differences before converting ns to us.

    Absolute globaltimer stamps can exceed float's exact integer range. Never
    cast those operands to float first: it corrupts the shortest wait spans.
    """
    if not pipelines:
        return None
    checks, ages, already_ready = [], [], 0
    per_cta = defaultdict(lambda: defaultdict(int))
    for (rank, index), p in pipelines.items():
        totals = per_cta[rank, p['cta']]
        totals['tiles'] += 1
        totals['mma_loop_ns'] += p['mma_return'] - p['tmem_acquired']
        totals['tmem_slot_wait_ns'] += p['tmem_acquired'] - p['mma_begin']
        for peer in range(world):
            release = peers[rank, index//n_tiles * world + peer]['release']
            checks.append(p[f'ready_end{peer}'] - p[f'ready_begin{peer}'])
            ages.append(peers[rank, index][f'acquire{peer}'] - release)
            already_ready += p[f'ready_begin{peer}'] >= release
            totals['ready_check_ns'] += checks[-1]
        for k in range(p['k_tiles']):
            s = stages[rank, index, k]
            totals['mma_input_wait_ns'] += s['wait_end'] - s['wait_begin']
    return dict(sampled_tiles=len(pipelines), sampled_k_stages=len(stages),
        ready_checks=len(checks), ready_before_check=already_ready,
        release_to_acquire_p50_us=median(ages)/1000,
        ready_check_p50_us=median(checks)/1000, ready_check_max_us=max(checks)/1000,
        per_cta=[dict(rank=r, cta=c, **totals) for (r, c), totals in sorted(per_cta.items())],
        scope='selected workers only; polling includes check overhead; input waits may overlap prior async MMA')


def tile_consumption(pipelines, stages, *, m, n, k, tile_m, tile_n, world):
    """Observed per-tile service spans, never async-wait subtraction.

    wait_end(K0) precedes the first MMA submission (not exact issue time).
    acc_wait_end observes completion, possibly late. Their span includes later
    input starvation; FLOP/span is effective tile throughput, not tensor busy.
    Ready polling runs on another warp and must not be subtracted from it.
    """
    n_tiles = (n + tile_n - 1) // tile_n
    rows = []
    previous = {}
    for (rank, index), p in sorted(pipelines.items(), key=lambda item: (
            item[0][0], item[1]['cta'], item[1]['mma_begin'])):
        ss = [stages[rank, index, kk] for kk in range(p['k_tiles'])]
        mi, ni = divmod(index, n_tiles)
        flops = 2 * min(tile_m, m-mi*tile_m) * min(tile_n, n-ni*tile_n) * k
        span = p['acc_wait_end'] - ss[0]['wait_end']
        assert span > 0 and flops > 0
        worker = rank, p['cta']
        prev = previous.get(worker)
        rows.append(dict(rank=rank, cta=p['cta'], m_tile=mi, n_tile=ni,
            flops=flops, effective_tile_tflops=flops/span/1000,
            first_input_wait_us=(ss[0]['wait_end']-ss[0]['wait_begin'])/1000,
            later_input_wait_sum_us=sum(s['wait_end']-s['wait_begin'] for s in ss[1:])/1000,
            ready_poll_sum_us=sum(p[f'ready_end{peer}']-p[f'ready_begin{peer}']
                                  for peer in range(world))/1000,
            tmem_slot_wait_us=(p['tmem_acquired']-p['mma_begin'])/1000,
            observed_service_us=span/1000,
            completion_tail_us=(p['acc_wait_end']-p['mma_return'])/1000,
            epilogue_after_observed_completion_us=(p['epi_return']-p['acc_wait_end'])/1000,
            start_interval_us=None if prev is None else (p['mma_begin']-prev['mma_begin'])/1000,
            first_input_interval_us=None if prev is None else
                (ss[0]['wait_end']-prev['first_input'])/1000,
            previous_observed_service_us=None if prev is None else prev['service_ns']/1000,
            # Signed, not clamped: next tile can start before the epilogue
            # observes previous completion. A positive gap is only an observed
            # inter-tile gap, not a direct Tensor Core idle measurement.
            completion_to_next_input_us=None if prev is None else
                (ss[0]['wait_end']-prev['completion'])/1000))
        previous[worker] = dict(mma_begin=p['mma_begin'], first_input=ss[0]['wait_end'],
                                completion=p['acc_wait_end'], service_ns=span)
    return rows


def cta_time_accounting(ctas, pipelines, stages, comm_ctas):
    """Telescope observed timestamps, without adding overlapping warp spans.

    This closes a sampled CTA and then its GPU's observed kernel envelope.
    It does NOT decompose an uninstrumented Graph baseline or measure tensor
    busy. In particular other-CTA tail is not a wait executed by this CTA.
    """
    result = []
    for rank in sorted({r for r, _ in ctas}):
        local = {c: p for (r, c), p in ctas.items() if r == rank}
        origin = min(p['start'] for p in local.values())
        last = max(local, key=lambda c: local[c]['end'])
        kernel_end = local[last]['end']
        row = dict(rank=rank, kernel_span_ns=kernel_end-origin, last_cta=last,
            last_role='comm' if last < comm_ctas else 'compute',
            comm_end_ns=max(p['end'] for c, p in local.items() if c < comm_ctas)-origin,
            compute_end_ns=max(p['end'] for c, p in local.items() if c >= comm_ctas)-origin,
            sampled_ctas=[])
        for cta in sorted({p['cta'] for (r, _), p in pipelines.items() if r == rank}):
            tiles = sorted(((i, p) for (r, i), p in pipelines.items()
                            if r == rank and p['cta'] == cta), key=lambda x: x[1]['mma_begin'])
            starts = [stages[rank, i, 0]['wait_end'] for i, _ in tiles]
            ends = [p['acc_wait_end'] for _, p in tiles]
            parts = dict(startup_ns=starts[0]-origin,
                observed_service_sum_ns=sum(b-a for a, b in zip(starts, ends)),
                signed_gap_sum_ns=sum(a-b for a, b in zip(starts[1:], ends[:-1])),
                drain_ns=local[cta]['end']-ends[-1])
            cta_end = local[cta]['end']-origin
            assert sum(parts.values()) == cta_end, 'CTA accounting does not close'
            other_tail = kernel_end-local[cta]['end']
            assert cta_end+other_tail == row['kernel_span_ns']
            row['sampled_ctas'].append(dict(cta=cta, tiles=len(tiles), **parts,
                end_ns=cta_end, other_cta_tail_ns=other_tail, closure_error_ns=0))
        result.append(row)
    return result


def make_trace(lines, job):
    ctas, peers, hosts, devices, validations = {}, {}, {}, {}, set()
    pipelines, stages = {}, {}
    verified = passed = False
    for line in lines:
        row = parse(line)
        if line.startswith('device,'):
            devices[int(row['rank'])] = int(row['sms'])
        elif line.startswith('profile_cta,A2A_GEMM,'):
            key = int(row['rank']), int(row['cta'])
            assert key not in ctas, 'Duplicate CTA'
            ctas[key] = {k: int(row[k]) for k in ('start', 'end', 'active_start')}
        elif line.startswith('profile_oproj_pipeline,'):
            key = int(row['rank']), int(row['index'])
            assert key not in pipelines, 'Duplicate pipeline tile'
            pipelines[key] = {k: int(v) for k, v in row.items()}
        elif line.startswith('profile_oproj_stage,'):
            key = int(row['rank']), int(row['index']), int(row['k'])
            assert key not in stages, 'Duplicate MMA stage'
            stages[key] = {k: int(v) for k, v in row.items()}
        elif line.startswith('profile_peer,'):
            key = int(row['rank']), int(row['index'])
            assert key not in peers, 'Duplicate peer record'
            peers[key] = {k: int(v) for k, v in row.items()
                          if k not in ('profile_detail', 'host_launch')}
        elif line.startswith('profile_host,A2A_GEMM,'):
            rank = int(row['rank'])
            assert rank not in hosts
            hosts[rank] = row
        elif line.startswith(('correctness,A2A_GEMM,', 'route,A2A_GEMM,')):
            phase = row.get('profile_phase')
            if phase in ('instrumented', 'host_stages'):
                assert int(row['nonfinite']) == 0
                field = 'bitwise_mismatches' if line.startswith('route,') else 'mismatches'
                assert int(row[field]) == 0 and int(row['checked']) == int(row['elements']) > 0
                validations.add((phase, line.split(',')[0], int(row['rank'])))
        elif line.startswith('candidate_verified,A2A_GEMM,'):
            assert row['full_numeric'] == row['full_route'] == '1'
            assert row['payload_generations'] == '2' and row['performance_accepted'] == '0'
            verified = True
        elif line.startswith('PASS:') and 'diagnostic timelines' in line:
            passed = True
    world, comm = job['world'], job['comm_sm']
    m = job['global_seq'] // world
    import re
    policy = job.get('oproj_policy_list') or job.get('oproj_policy')
    assert isinstance(policy, str) and ',' not in policy, 'One explicit tile required'
    tile = re.fullmatch(r'm(\d+)n(\d+).*', policy)
    assert tile, 'Explicit tile required'
    bm, bn = map(int, tile.groups())
    mt, nt = (m + bm - 1) // bm, (job['hidden'] + bn - 1) // bn
    capacity = max(mt * nt, mt * world)
    assert verified and passed and set(hosts) == set(devices) == set(range(world))
    assert validations == {(p, k, r) for p in ('instrumented', 'host_stages')
                           for k in ('correctness', 'route') for r in range(world)}
    assert set(peers) == {(r, i) for r in range(world) for i in range(capacity)}
    if job.get('oproj_pipeline_probe'):
        assert pipelines and {r for r, _ in pipelines} == {0}, 'Missing GPU0 pipeline probe'
        assert set(stages) == {(r, i, k) for (r, i), p in pipelines.items() for k in range(p['k_tiles'])}
        compute = min(mt * nt, devices[0] - comm)
        workers = {w for w in (0, 1, compute - min(job.get('max_swizzle_size', 1), compute))
                   if 0 <= w < compute}
        assert {p['cta'] - comm for p in pipelines.values()} == workers, 'Incomplete sampled workers'
        for worker in workers:
            assert sum(p['cta'] == worker + comm for p in pipelines.values()) == (
                mt * nt - 1 - worker) // compute + 1, 'Incomplete persistent tiles'
    events = []
    for rank in range(world):
        selected = {c: row for (r, c), row in ctas.items() if r == rank}
        count = comm + min(mt * nt, devices[rank] - comm)
        assert set(selected) == set(range(count))
        # The private probe still validates all eight GPUs, but its delivered
        # detail is GPU0 only. Do not replicate thousands of unrelated tracks.
        if pipelines and rank != 0:
            continue
        origin = min(r['start'] for r in selected.values())
        end = max(r['end'] for r in selected.values())
        events.append(dict(ph='M', name='process_name', pid=rank,
                           args=dict(name=f'GPU {rank} (independent origin)')))

        def track(tid, name):
            events.extend([dict(ph='M', name='thread_name', pid=rank, tid=tid, args=dict(name=name)),
                           dict(ph='M', name='thread_sort_index', pid=rank, tid=tid, args=dict(sort_index=tid))])

        def span(tid, name, begin, finish, **attrs):
            assert origin <= begin <= finish <= end, (rank, name, begin, finish, origin, end)
            events.append(dict(ph='X', name=name, cat='fuse.oproj', pid=rank, tid=tid,
                               ts=(begin-origin)/1000, dur=(finish-begin)/1000, args=attrs))

        track(0, 'A2A -> GEMM diagnostic boundary')
        span(0, 'A2A -> GEMM', origin, end)
        slots = {(p['comm_cta'], p['comm_slot']) for (r, i), p in peers.items()
                 if r == rank and i < mt * world}
        for cta, row in sorted(selected.items()):
            tid = 100 + cta * 16
            track(tid, f'{"Communication" if cta < comm else "GEMM"} CTA {cta}')
            if cta < comm:
                span(tid, 'remote A2A role (includes setup/waits)', row['start'], row['end'])
                for c, slot in sorted(slots):
                    if c == cta:
                        assert -1 <= slot < 8
                        track(tid + slot + 2, f'CTA {cta} slot {slot}: final-publisher chunk phases')
            else:
                span(tid, 'startup -> first ready observation (not spin time)', row['start'], row['active_start'])
                span(tid, 'GEMM role (includes later peer waits)', row['active_start'], row['end'])
                if any(r == rank and p['cta'] == cta for (r, _), p in pipelines.items()):
                    track(tid + 1, f'CTA {cta}: load warp / ready checks')
                    track(tid + 2, f'CTA {cta}: MMA warp / input stages')
                    track(tid + 3, f'CTA {cta}: MMA tile / TMEM backpressure')
                    track(tid + 4, f'CTA {cta}: epilogue / observed completion')
        for index in range(mt * world):
            p = peers[rank, index]
            assert p['comm_valid'] and 0 <= p['comm_cta'] < comm
            assert 0 <= p['source_rank'] < world
            assert p['task_begin'] <= p['input_ready'] <= p['publish_issue'] <= p['release']
            tid = 100 + p['comm_cta'] * 16 + p['comm_slot'] + 2
            attrs = dict(ready_m=index//world, ready_peer=index%world,
                         src_gpu=p['source_rank'], dst_gpu=rank, task=p['task_id'],
                         row_chunk=p['row_chunk'], copy_rows=p['copy_rows'],
                         final_publisher_only=True, copy_path=p['copy_path'])
            if p['copy_path'] == 3:
                attrs['column_chunk'] = attrs.pop('row_chunk')
                attrs['comm_layout'] = 'columns'
            span(tid, 'task setup / input-ready wait', p['task_begin'], p['input_ready'], **attrs)
            if p['g2s_issue']:
                span(tid, 'remote G2S' if p['copy_path'] else 'vector copy', p['g2s_issue'], p['g2s_done'], **attrs)
            if p['s2g_issue']:
                local_attrs = dict(attrs, src_gpu=rank, remote_source_gpu=p['source_rank'])
                span(tid, 'local S2G (destination complete)', p['s2g_issue'], p['s2g_done'], **local_attrs)
            span(tid, 'ready atomic (post-publication sample)', p['publish_issue'], p['release'], **attrs)
        for index in range(mt * nt):
            p = peers[rank, index]
            assert p['valid'] and (p['m'], p['n'], p['batch']) == (index//nt, index%nt, 0)
            if pipelines:
                continue
            tid = 10000 + index
            track(tid, f'GEMM tile ({p["m"]},{p["n"]}): ready handoff, not GEMM duration')
            for peer in range(world):
                acquire = p[f'acquire{peer}']
                release = peers[rank, p['m'] * world + peer]['release']
                attrs = dict(m_tile=p['m'], n_tile=p['n'], ready_peer=peer,
                             acquire_minus_release_ns=acquire-release,
                             note='release timestamp is sampled after atomic; negative delta is permitted')
                assert origin <= acquire <= end
                if acquire >= release:
                    span(tid, 'release -> acquire (may include preceding GEMM)', release, acquire, **attrs)
                else:
                    events.append(dict(ph='i', s='t', name='acquire before post-release timestamp',
                                       pid=rank, tid=tid, ts=(acquire-origin)/1000, args=attrs))
        for (r, index), p in sorted(pipelines.items()):
            if r != rank:
                continue
            assert 0 <= index < mt * nt and comm <= p['cta'] < count
            tid = 100 + p['cta'] * 16
            attrs = dict(m_tile=index//nt, n_tile=index%nt, cta=p['cta'],
                         clock='GPU-local globaltimer', diagnostic_only=True)
            assert p['mma_begin'] <= p['tmem_acquired'] <= p['mma_return'] <= p['acc_wait_end']
            assert p['epi_begin'] <= p['acc_wait_begin'] <= p['acc_wait_end']
            assert p['acc_wait_end'] <= p['tmem_release_begin'] <= p['tmem_release_end'] <= p['epi_return']
            span(tid+3, 'MMA wait: reusable TMEM slot', p['mma_begin'], p['tmem_acquired'], **attrs)
            span(tid+3, 'MMA submission loop (includes input waits; not tensor busy)',
                 p['tmem_acquired'], p['mma_return'], **attrs)
            span(tid+4, 'epilogue setup', p['epi_begin'], p['acc_wait_begin'], **attrs)
            span(tid+4, 'epilogue wait: MMA completion observed', p['acc_wait_begin'], p['acc_wait_end'], **attrs)
            span(tid+4, 'epilogue: TMEM reads / subtile work', p['acc_wait_end'], p['tmem_release_begin'], **attrs)
            span(tid+4, 'epilogue: sampled lane TMEM release', p['tmem_release_begin'], p['tmem_release_end'], **attrs)
            span(tid+4, 'epilogue: remaining work (stores may remain in flight)',
                 p['tmem_release_end'], p['epi_return'], **attrs)
            for peer in range(world):
                begin, finish, joined = (p[f'{name}{peer}'] for name in ('ready_begin', 'ready_end', 'ready_joined'))
                assert begin <= finish <= joined
                release = peers[rank, index//nt * world + peer]['release']
                ready_attrs = dict(attrs, peer=peer, release_us=(release-origin)/1000,
                                   check_begin_minus_release_ns=begin-release,
                                   check_ns=finish-begin, cached=bool(p[f'cache_hit{peer}']))
                span(tid+1, 'cached ready' if p[f'cache_hit{peer}'] else 'ready check / polling',
                     begin, finish, **ready_attrs)
                span(tid+1, 'load warp join / proxy fence', finish, joined, **ready_attrs)
            for k in range(p['k_tiles']):
                s = stages[rank, index, k]
                assert s['wait_begin'] <= s['wait_end'] <= s['issue_end']
                stage_attrs = dict(attrs, k_tile=k, peer=k*world//p['k_tiles'])
                span(tid+2, 'MMA input-stage wait', s['wait_begin'], s['wait_end'], **stage_attrs)
                span(tid+2, 'MMA issue / stage release (asynchronous)', s['wait_end'], s['issue_end'], **stage_attrs)
    return dict(traceEvents=events, displayTimeUnit='ns', metadata=dict(
        schema='fuse_sm103_oproj_perfetto_v1', diagnostic_only=True, performance_accepted=False,
        pipeline_probe=bool(pipelines),
        cta_time_accounting=cta_time_accounting(ctas, pipelines, stages, comm),
        tile_consumption=tile_consumption(pipelines, stages, m=m, n=job['hidden'],
            k=job.get('q_heads', 0)*job.get('head_dim', 0),
            tile_m=int(tile[1]), tile_n=int(tile[2]), world=world) if pipelines
            and job.get('q_heads') and job.get('head_dim') else [],
        pipeline_summary=pipeline_summary(pipelines, stages, peers, nt, world),
        completion_semantics='epilogue observes existing MMA barrier; observation is an upper bound, not exact hardware completion',
        config=job, profile_host=hosts,
        clock='GPU-local globaltimer ns converted to microseconds; never subtract across ranks',
        scope='single-process diagnostic; not MPI Graph performance; communication records only final publishing chunks'))


def export(run, output, tiles_csv=None):
    receipt = json.loads((run / 'fetched.json').read_text())
    assert receipt['state'] == 'succeeded' and receipt['exit_code'] == 0
    digest = hashlib.sha256()
    with (run / 'artifacts.tar.gz').open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    assert digest.hexdigest() == receipt['artifact_sha256']
    control = run / f'artifacts-attempt{receipt["attempt"]}' / 'control'
    # Verify extracted inputs against the receipt-bound archive, not just the
    # archive file itself. A locally edited log must not become a new trace.
    with tarfile.open(run / 'artifacts.tar.gz') as archive:
        for name in ('job.json', f'attempt{receipt["attempt"]}.log'):
            archived = archive.extractfile('control/' + name)
            assert archived is not None
            with (control / name).open('rb') as local:
                while True:
                    chunk = archived.read(1024 * 1024)
                    assert local.read(len(chunk)) == chunk, 'Extracted evidence differs from archive'
                    if not chunk:
                        assert not local.read(1)
                        break
    job = json.loads((control / 'job.json').read_text())
    assert job['profile'] and job['directions'] == 'oproj' and not job['mpi']
    # The controller serializes an omitted --profile-detail as null; both its
    # argv builder and the harness resolve that default to full peer telemetry.
    assert (job.get('profile_detail') or 'full') == 'full'
    with (control / f'attempt{receipt["attempt"]}.log').open() as stream:
        trace = make_trace(stream, job)
    trace['metadata']['config'] = {k: job.get(k) for k in (
        'run_id', 'source_id', 'node', 'global_seq', 'world', 'hidden', 'q_heads',
        'kv_heads', 'head_dim', 'comm_sm', 'oproj_policy', 'oproj_policy_list', 'max_swizzle_size',
        'oproj_raster', 'oproj_comm_layout', 'oproj_pipeline_probe', 'host_launch', 'input_generator')}
    trace['metadata']['artifact_sha256'] = receipt['artifact_sha256']
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(trace, stream, separators=(',', ':'))
    if tiles_csv is not None:
        rows = trace['metadata']['tile_consumption']
        assert rows, 'Tile CSV requires the OProj pipeline probe'
        tiles_csv.parent.mkdir(parents=True, exist_ok=True)
        with tiles_csv.open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(dict(file=str(output), events=len(trace['traceEvents']))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--tiles-csv', type=Path,
                        help='selected-worker observed tile spans; not pure tensor execution time')
    args = parser.parse_args()
    export(args.run, args.output, args.tiles_csv)
