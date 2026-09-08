#!/usr/bin/env python3
"""Export SM103 OProj diagnostic telemetry; no kernel changes or GPU launches.

Each GPU has its own time origin. Legacy communication details describe only
the final publishing chunk; aligned plans describe the single owner's complete
ready-region transfer, not each internal copy. Release is sampled after
publication: acquire can precede that sample.
"""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from l20d import fused_policy_alignment, fused_policy_tile


def parse(line):
    return dict(p.split('=', 1) for p in line.strip().split(',')[1:] if '=' in p)


def make_trace(lines, job):
    ctas, peers, hosts, devices, validations = {}, {}, {}, {}, set()
    candidates = []
    verified = passed = False
    for line in lines:
        row = parse(line)
        if line.startswith('device,'):
            devices[int(row['rank'])] = int(row['sms'])
        elif line.startswith('profile_cta,A2A_GEMM,'):
            key = int(row['rank']), int(row['cta'])
            assert key not in ctas, 'Duplicate CTA'
            ctas[key] = {k: int(row[k]) for k in ('start', 'end', 'active_start')}
        elif line.startswith('profile_peer,'):
            key = int(row['rank']), int(row['index'])
            assert key not in peers, 'Duplicate peer record'
            peers[key] = {k: int(v) for k, v in row.items()
                          if k not in ('profile_detail', 'host_launch')}
        elif line.startswith('profile_host,A2A_GEMM,'):
            rank = int(row['rank'])
            assert rank not in hosts
            hosts[rank] = row
        elif line.startswith('candidate,A2A_GEMM,'):
            candidates.append(row)
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
    policy = job.get('oproj_policy_list') or job.get('oproj_policy')
    assert isinstance(policy, str) and ',' not in policy, 'One explicit tile required'
    assert policy != 'auto', 'Explicit tile required'
    bm, bn, bk = fused_policy_tile(policy)
    peer_k = job.get('q_heads', 0) * job.get('head_dim', 0) // world
    alignment = fused_policy_alignment(policy, peer_k)
    slices = alignment.get('ready_slices', 1)
    if alignment:
        assert {int(row['rank']) for row in candidates} == set(range(world)), 'Missing joint-policy metadata'
        for row in candidates:
            assert row['tile'] == policy and tuple(int(row[f'tile_{axis}']) for axis in 'mnk') == (bm, bn, bk)
            assert all(row.get(key) == str(value) for key, value in alignment.items()), 'Invalid joint-policy metadata'
    mt, nt = (m + bm - 1) // bm, (job['hidden'] + bn - 1) // bn
    capacity = max(mt * nt, mt * world) * slices
    assert verified and passed and set(hosts) == set(devices) == set(range(world))
    assert validations == {(p, k, r) for p in ('instrumented', 'host_stages')
                           for k in ('correctness', 'route') for r in range(world)}
    assert set(peers) == {(r, i) for r in range(world) for i in range(capacity)}
    if alignment:
        for (_, index), row in peers.items():
            slice_id = index % slices
            assert (row.get('ready_slice'), row.get('ready_slices'), row.get('k_begin_in_peer'), row.get('k_end_in_peer')) == (
                slice_id, slices, slice_id * alignment['ready_k'], (slice_id + 1) * alignment['ready_k']), 'Invalid ready slice metadata'
    events = []
    for rank in range(world):
        selected = {c: row for (r, c), row in ctas.items() if r == rank}
        count = comm + min(mt * nt, devices[rank] - comm)
        assert set(selected) == set(range(count))
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
                 if r == rank and i < mt * world * slices}
        for cta, row in sorted(selected.items()):
            tid = 100 + cta * 16
            track(tid, f'{"Communication" if cta < comm else "GEMM"} CTA {cta}')
            if cta < comm:
                span(tid, 'remote A2A role (includes setup/waits)', row['start'], row['end'])
                for c, slot in sorted(slots):
                    if c == cta:
                        assert -1 <= slot < 8
                        scope = 'ready-region transfer' if alignment else 'final-publisher chunk'
                        track(tid + slot + 2, f'CTA {cta} slot {slot}: {scope} phases')
            else:
                span(tid, 'first ready wait', row['start'], row['active_start'])
                span(tid, 'GEMM role (includes later peer waits)', row['active_start'], row['end'])
        for index in range(mt * world * slices):
            p = peers[rank, index]
            assert p['comm_valid'] and 0 <= p['comm_cta'] < comm
            assert 0 <= p['source_rank'] < world
            assert p['task_begin'] <= p['input_ready'] <= p['publish_issue'] <= p['release']
            tid = 100 + p['comm_cta'] * 16 + p['comm_slot'] + 2
            ready_index, slice_id = divmod(index, slices)
            attrs = dict(ready_m=ready_index//world, ready_peer=ready_index%world,
                         src_gpu=p['source_rank'], dst_gpu=rank, task=p['task_id'],
                         row_chunk=p['row_chunk'], copy_rows=p['copy_rows'],
                         final_publisher_only=not bool(alignment), copy_path=p['copy_path'])
            if alignment:
                attrs.update(ready_slice=slice_id, k_begin_in_peer=p['k_begin_in_peer'],
                             k_end_in_peer=p['k_end_in_peer'])
            span(tid, 'task setup / input-ready wait', p['task_begin'], p['input_ready'], **attrs)
            if p['copy_path'] == 3:
                assert alignment and p['g2s_issue'] <= p['g2s_done'] <= p['s2g_issue'] <= p['s2g_done'] <= p['publish_issue']
                # Row-owner can issue several subrectangles. The record has
                # first-G2S and final-S2G endpoints, not separate accumulated
                # G2S/S2G durations; never mislabel that interval as pure S2G.
                attrs['phase_scope'] = 'first G2S issue through final destination S2G completion'
                span(tid, 'rectangular G2S + S2G pipeline', p['g2s_issue'], p['s2g_done'], **attrs)
            elif p['g2s_issue']:
                span(tid, 'remote G2S' if p['copy_path'] else 'vector copy', p['g2s_issue'], p['g2s_done'], **attrs)
            if p['s2g_issue'] and p['copy_path'] != 3:
                local_attrs = dict(attrs, src_gpu=rank, remote_source_gpu=p['source_rank'])
                span(tid, 'local S2G (destination complete)', p['s2g_issue'], p['s2g_done'], **local_attrs)
            span(tid, 'ready atomic (post-publication sample)', p['publish_issue'], p['release'], **attrs)
        for index in range(mt * nt * slices):
            p = peers[rank, index]
            tile_index, slice_id = divmod(index, slices)
            assert p['valid'] and (p['m'], p['n'], p['batch']) == (tile_index//nt, tile_index%nt, 0)
            tid = 10000 + index
            suffix = f' K slice {slice_id}' if alignment else ''
            track(tid, f'GEMM tile ({p["m"]},{p["n"]}){suffix}: ready handoff, not GEMM duration')
            for peer in range(world):
                acquire = p[f'acquire{peer}']
                release = peers[rank, (p['m'] * world + peer) * slices + slice_id]['release']
                attrs = dict(m_tile=p['m'], n_tile=p['n'], ready_peer=peer,
                             acquire_minus_release_ns=acquire-release,
                             note='release timestamp is sampled after atomic; negative delta is permitted')
                if alignment:
                    attrs.update(ready_slice=slice_id,
                        k_begin=peer * peer_k + slice_id * alignment['ready_k'],
                        k_end=peer * peer_k + (slice_id + 1) * alignment['ready_k'])
                assert origin <= acquire <= end
                if acquire >= release:
                    span(tid, 'release -> acquire (may include preceding GEMM)', release, acquire, **attrs)
                else:
                    events.append(dict(ph='i', s='t', name='acquire before post-release timestamp',
                                       pid=rank, tid=tid, ts=(acquire-origin)/1000, args=attrs))
    return dict(traceEvents=events, displayTimeUnit='ns', metadata=dict(
        schema='fuse_sm103_oproj_perfetto_v1', diagnostic_only=True, performance_accepted=False,
        config=job, profile_host=hosts, joint_policy=alignment,
        clock='GPU-local globaltimer ns converted to microseconds; never subtract across ranks',
        scope='single-process diagnostic; not MPI Graph performance; ' +
              ('communication records cover each single-writer ready region' if alignment else
               'communication records only final publishing chunks')))


def export(run, output):
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
    assert job['profile_detail'] == 'full'
    with (control / f'attempt{receipt["attempt"]}.log').open() as stream:
        trace = make_trace(stream, job)
    trace['metadata']['config'] = {k: job.get(k) for k in (
        'run_id', 'source_id', 'node', 'global_seq', 'world', 'hidden', 'q_heads',
        'kv_heads', 'head_dim', 'comm_sm', 'oproj_policy', 'oproj_policy_list', 'max_swizzle_size',
        'oproj_raster', 'host_launch', 'input_generator')}
    trace['metadata']['artifact_sha256'] = receipt['artifact_sha256']
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(trace, stream, separators=(',', ':'))
    print(json.dumps(dict(file=str(output), events=len(trace['traceEvents']))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    export(args.run, args.output)
