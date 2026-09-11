"""Direct MXFP8 primitive-service calibration; never fit fused performance.

Input is the versioned, rank-local service JSONL sidecar in an audited L20D
run. Each stage has ONE complete instrumented epoch. Its many tile/warp
observations are correlated samples, not 50 independent profile repetitions.
The separate 10+50 eager event timings include host API/launch preparation;
they diagnose launch-inclusive perturbation and never correct GPU services.
"""

import argparse
from collections import defaultdict
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile

import l20d
import summarize_sm103_fused as evidence


SCHEMA = 'sm103_mxfp8_services_v1'
STAGES = ('C_allready', 'C_delay1', 'C_delay2', 'Q', 'R', 'QR', 'QR_phase1')
KINDS = ('config', 'cta', 'tile', 'panel', 'quant', 'route')
PHYSICAL_FIELDS = ('m', 'n', 'k', 'world', 'sm_count', 'capability', 'comm_ctas',
                   'compute_ctas', 'tile_m', 'tile_n', 'tile_k', 'epilogue_n', 'stages',
                   'cluster_ctas', 'raster', 'resolved_swizzle', 'max_swizzle_size',
                   'dynamic_smem_bytes', 'q_heads', 'kv_heads', 'head_dim',
                   'weight_preparation', 'rank_swizzle')
SERVICE_FIELDS = ('startup_us', 'tile_first_us', 'tile_cycle_us', 'tile_latency_us',
                  'quant_us', 'quant_g2s_us', 'quant_s2g_us', 'quant_publish_us',
                  'g2s_us', 'g2s_mixed_us', 's2g_us', 's2g_mixed_us', 'issue_us',
                  'join_us', 'ready_poll_us', 'drain_us',
                  'quant_g2s_publish_us', 'quant_s2g_publish_us',
                  'g2s_publish_us', 's2g_publish_us')
require = evidence.require


def integer(row, field, minimum=0):
    value = row.get(field)
    require(type(value) is int and value >= minimum, f'Invalid integer {field}')
    return value


def interval(row, begin, end):
    """Only differences within one rank/epoch record are legal."""
    require(integer(row, 'rank') == 0 and integer(row, 'iteration') == 0,
            'Only the declared GPU0 capture is supported; do not subtract ranks/epochs')
    first, last = integer(row, begin, 1), integer(row, end, 1)
    require(last >= first, f'Unordered local timestamps {begin}/{end}')
    return last - first


def statistics(values):
    require(values and all(math.isfinite(v) and v >= 0 for v in values), 'Missing/nonfinite observations')
    total = math.fsum(values)
    return dict(count=len(values), sum_ns=total, mean_ns=total / len(values), min_ns=min(values),
                max_ns=max(values), p50_ns=evidence.percentile(values, .5),
                p95_ns=evidence.percentile(values, .95))


def physical_key(config):
    require(all(field in config for field in PHYSICAL_FIELDS), 'Incomplete physical key')
    key = {field: config[field] for field in PHYSICAL_FIELDS}
    for field in set(PHYSICAL_FIELDS) - {'raster', 'weight_preparation', 'rank_swizzle'}:
        integer(key, field, 1)
    integer(key, 'raster')
    require(key['sm_count'] == 148 and key['capability'] == 103 and key['world'] in (4, 8)
            and key['tile_m'] == 128 and key['tile_n'] == 256 and key['tile_k'] == 128
            and key['epilogue_n'] in (32, 64) and key['cluster_ctas'] == 1
            and key['m'] % 128 == key['n'] % 256 == key['k'] % 128 == 0
            and key['raster'] in (0, 1)
            and key['resolved_swizzle'] in (1, 2, 4, 8)
            and key['max_swizzle_size'] in (1, 2, 4, 8)
            and key['resolved_swizzle'] <= key['max_swizzle_size']
            and key['weight_preparation'] == 'comm' and key['rank_swizzle'] == 'off',
            'Unregistered MXFP8 service geometry/schedule')
    require(key['head_dim'] == 128 and key['q_heads'] % key['world'] == key['kv_heads'] % key['world'] == 0
            and key['q_heads'] % key['kv_heads'] == 0
            and key['n'] == (key['q_heads'] + 2 * key['kv_heads']) * key['head_dim'],
            'Incompatible QKV routing geometry')
    mt, nt, sw = key['m'] // 128, key['n'] // 256, key['resolved_swizzle']
    require(mt % sw == nt % sw == 0, 'First calibration requires unpadded service geometry')
    require(0 < key['comm_ctas'] < 148
            and key['compute_ctas'] == min(mt * nt, 148 - key['comm_ctas']), 'Compute budget mismatch')
    return key


def read_capture(path=None, *, data=None):
    """Parse one stable schema, without compatibility guesses for old logs."""
    stages = {}
    with io.StringIO(data.decode() if data is not None else Path(path).read_text()) as stream:
        for number, text in enumerate(stream, 1):
            row = evidence.json_bytes(text.encode())
            require(isinstance(row, dict) and row.get('schema') == SCHEMA
                    and row.get('kind') in KINDS and row.get('stage') in STAGES,
                    f'Unknown service record/schema at line {number}')
            require(integer(row, 'rank') == 0 and integer(row, 'iteration') == 0,
                    'Unexpected capture rank/epoch')
            stage = stages.setdefault(row['stage'], {kind: [] for kind in KINDS})
            stage[row['kind']].append(row)
    require({'C_allready', 'Q', 'R', 'QR', 'QR_phase1'} <= set(stages), 'Missing service stage')
    common = None
    gpu_uuid = None
    for name, rows in stages.items():
        require(len(rows['config']) == 1, 'Missing/duplicate stage config')
        config = rows['config'][0]
        key = physical_key(config)
        require(common is None or common == key, 'Physical key changed between service stages')
        common = key
        uuid = config.get('gpu_uuid')
        require(isinstance(uuid, str) and re.fullmatch(
            r'GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', uuid),
            'Missing physical capture GPU UUID')
        require(gpu_uuid is None or gpu_uuid == uuid, 'Capture GPU changed between stages')
        gpu_uuid = uuid
        require(config.get('clock') == 'globaltimer' and config.get('clock_unit') == 'ns'
                and config.get('event_boundary') == 'host_api_inclusive_eager_cuda_event'
                and integer(config, 'captured_rank') == 0 and integer(config, 'capture_epochs') == 1
                and integer(config, 'warmup') >= 10 and integer(config, 'samples') == 50
                and integer(config, 'validation_generations') == 2,
                'Service sampling/clock/validation contract mismatch')
        require(integer(config, 'quant_phase_steps') == (1 if name == 'QR_phase1' else 0),
                'Unregistered phase-excitation protocol')
        for field in ('control_samples_ms', 'instrumented_samples_ms'):
            values = config.get(field)
            require(isinstance(values, list) and len(values) == 50
                    and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in values),
                    'Missing 50 positive perturbation samples')
    return common, stages


def indexed(rows):
    result = {}
    for row in rows:
        index = integer(row, 'index')
        require(index not in result, 'Duplicate raw record index')
        result[index] = row
    return result


def reduce_bulk_services(key, captures):
    """Whole local services, not per-instruction latencies or fused samples.

    Start at the earliest CTA entering its actual work, after diagnostic reset.
    R/QR end after every warp's remote drain and (for QR) owned quant work,
    BEFORE cross-rank finalization. Q cannot use thread0's CTA exit as the
    completion of the other seven warps. All endpoints belong to this GPU and
    this capture. These spans are observations, not bounds on fused contention.
    """
    result = dict(reference_m=key['m'], reference_n=key['n'], reference_k=key['k'],
        boundary='local_service_after_setup_before_cross_rank_finalize',
        extrapolation='fixed_weight_work_plus_incremental_output_at_R_effective_rate',
        measured_global_seq=key['m'] * key['world'])
    for stage, field in (('C_allready', 'compute_us'), ('Q', 'quant_us'),
                         ('R', 'route_us'), ('QR', 'quant_route_us')):
        rows = captures[stage]
        start = min(integer(row, 'setup_done', 1) for row in rows['cta'])
        if stage == 'C_allready':
            finish = max(integer(row, 'end', 1) for row in rows['cta'])
        else:
            endpoints = [integer(row, 'end', 1) for row in rows['quant']]
            # Final drain records follow the valid output copy slots.
            tasks = key['m'] // 128 * (key['n'] // 256) * 4
            endpoints.extend(integer(row, 's2g_read_done', 1) for row in rows['route']
                             if integer(row, 'index') >= tasks)
            require(endpoints, 'Missing local bulk service completion')
            finish = max(endpoints)
        require(finish > start, 'Nonpositive local bulk service')
        result[field] = (finish - start) / 1000
    return result


def reduce_capture(key, captures):
    """Every coefficient is an arithmetic mean of explicitly defined intervals.

    No fused times, winners, correction factors or adjustable quantiles enter
    this function. Zero issue/join means INCLUDED in occupied-slot intervals.
    When 8*c >= K/4, every worker owns at most one chunk in each panel: publication
    is included in each complete progress service and its separate charge is 0.
    """
    values = defaultdict(list)
    diagnostics, missing = {}, []
    comm, compute, panels = key['comm_ctas'], key['compute_ctas'], key['n'] // 256
    workers, steps = 8 * comm, key['k'] // 4
    always_publish = workers >= steps
    for name, rows in captures.items():
        config = rows['config'][0]
        require(physical_key(config) == key, 'Service stage uses another physical key')
        ctas = indexed(rows['cta'])
        expected_ctas = comm + compute if name.startswith('C_') else comm
        require(set(ctas) == set(range(expected_ctas)), 'Missing CTA lifecycle coverage')
        for row in ctas.values():
            interval(row, 'begin', 'setup_done')
            interval(row, 'setup_done', 'end')
        origin = min(row['begin'] for row in ctas.values())
        setup = max(row['setup_done'] for row in ctas.values())
        if name == 'C_allready':
            values['startup_us'].append(setup - origin)
        stage_diagnostics = dict(capture_epochs=1, independent_capture_repetitions=1,
            event_boundary='eager_cuda_event_including_host_API_and_enqueue',
            event_timings_used_for_service_coefficients=False,
            control_p50_ms=evidence.percentile(config['control_samples_ms'], .5),
            instrumented_p50_ms=evidence.percentile(config['instrumented_samples_ms'], .5),
            control_half_drift=evidence.drift(config['control_samples_ms']),
            instrumented_half_drift=evidence.drift(config['instrumented_samples_ms']))
        stage_diagnostics['instrumentation_ratio'] = (
            stage_diagnostics['instrumented_p50_ms'] / stage_diagnostics['control_p50_ms'])
        diagnostics[name] = stage_diagnostics
        if name.startswith('C_'):
            tiles = indexed(rows['tile'])
            require(set(tiles) == set(range(key['m'] // 128 * panels)), 'Incomplete compute tile coverage')
            by_cta = defaultdict(list)
            for index, row in tiles.items():
                require(integer(row, 'm') * panels + integer(row, 'n') == index
                        and comm <= integer(row, 'cta') < comm + compute, 'Invalid compute tile owner/coordinate')
                interval(row, 'first_load', 'load_return')
                interval(row, 'store_begin', 'ready_after')
                require(row['first_load'] >= ctas[row['cta']]['setup_done']
                        and row['ready_after'] >= row['first_load'], 'Tile escaped its compute lifecycle')
                by_cta[row['cta']].append(row)
            for owned in by_cta.values():
                owned.sort(key=lambda row: row['first_load'])
                if name == 'C_allready':
                    require(owned[0]['ready_after'] >= setup, 'First output precedes common setup')
                    values['tile_first_us'].append(owned[0]['ready_after'] - setup)
                    for first, second in zip(owned, owned[1:]):
                        require(second['ready_after'] >= first['ready_after'], 'Output order changed within CTA')
                        values['tile_cycle_us'].append(second['ready_after'] - first['ready_after'])
            if name != 'C_allready':
                delayed = integer(config, 'delayed_panel')
                releases = indexed(rows['panel'])
                require(delayed < panels and delayed in releases, 'Missing controlled delayed panel')
                release = releases[delayed]
                interval(release, 'release_begin', 'release')
                require(integer(config, 'delay_ns', 1) > 0, 'Missing controlled release delay')
                lower, upper = [], []
                for owned in by_cta.values():
                    for previous, row in zip(owned, owned[1:]):
                        if row['n'] != delayed or not row.get('wait_begin'):
                            continue
                        interval(row, 'wait_begin', 'wait_end')
                        # Producer pre/post stamps bound publication. Do not
                        # reject a racing acquire merely before the POST stamp.
                        if not (previous['ready_after'] <= release['release_begin']
                                and row['wait_begin'] < release['release_begin'] <= row['wait_end']):
                            continue
                        require(row['ready_after'] >= release['release'], 'Output precedes controlled release')
                        lower.append(row['ready_after'] - release['release'])
                        upper.append(row['ready_after'] - release['release_begin'])
                if not upper:
                    missing.append(name + ': no noninitial, genuinely blocked recovery tile')
                else:
                    values['tile_latency_us'].extend(upper)
                    stage_diagnostics['recovery_lower'] = statistics(lower)
                    stage_diagnostics['recovery_upper'] = statistics(upper)
            continue

        quant = indexed(rows['quant'])
        quant_by_worker = defaultdict(list)
        if name in ('Q', 'QR', 'QR_phase1'):
            require(set(quant) == set(range(panels * steps)), 'Incomplete quant chunk coverage')
            arrivals, published = defaultdict(int), defaultdict(int)
            for index, row in quant.items():
                owner = integer(row, 'warp') * comm + integer(row, 'cta')
                require(row['warp'] < 8 and row['cta'] < comm and owner == index % workers
                        and integer(row, 'panel') == index // steps and integer(row, 'groups') == 32,
                        'Quant queue coordinate/owner mismatch')
                interval(row, 'begin', 'quant_done')
                interval(row, 'quant_done', 'end')
                contribution = integer(row, 'arrival_chunks')
                require(bool(contribution) == (index % steps + workers >= steps),
                        'Publication does not match final owned chunk')
                if always_publish:
                    require(contribution == 1, 'Every-chunk publication proof failed')
                if contribution:
                    panel_step = index % steps
                    require(contribution == panel_step // workers + 1,
                            'Aggregated publication differs from owned panel chunks')
                    interval(row, 'quant_done', 'warp_join_done')
                    interval(row, 'warp_join_done', 'arrival_done')
                    interval(row, 'arrival_done', 'end')
                    arrivals[row['panel']] += contribution
                    if row.get('release'):
                        require(row['quant_done'] <= row['release'] <= row['end'], 'Invalid ready publication timestamp')
                        published[row['panel']] += 1
                else:
                    require(not row.get('release') and not row.get('warp_join_done')
                            and not row.get('arrival_done'), 'Nonpublishing chunk has publication timestamps')
                quant_by_worker[owner].append(row)
            require(all(arrivals[p] == steps and published[p] == 1 for p in range(panels)),
                    'Incomplete full-panel publication')
            for owned in quant_by_worker.values():
                owned.sort(key=lambda row: row['index'])
                if name == 'Q':
                    # Q has no final CTA barrier: thread0's CTA.end is NOT an
                    # upper bound for all eight independently draining warps.
                    # This first queue prefix is observable but is not silently
                    # folded into the common C-grid startup coefficient.
                    first = owned[0]
                    stage_diagnostics.setdefault('first_chunk_prefix_ns', []).append(
                        first['begin'] - ctas[first['cta']]['begin'])
                for first, second in zip(owned, owned[1:]):
                    require(first['end'] <= second['begin'], 'Overlapping sequential warp quantization')
                    if name == 'Q':
                        finish = second['end'] if always_publish else second['quant_done']
                        values['quant_us'].append(finish - first['end'])
                if name == 'Q' and not always_publish:
                    for row in owned:
                        if row['arrival_chunks']:
                            values['quant_publish_us'].append(row['end'] - row['quant_done'])
            stage_diagnostics['quant_chunks'] = len(quant)
            stage_diagnostics['quant_publications'] = sum(bool(row['arrival_chunks']) for row in quant.values())
        else:
            require(not quant, 'Quant records in route-only stage')

        if name == 'Q':
            require(not rows['route'], 'Route records in quant-only stage')
            continue
        routes, drains, by_worker = indexed(rows['route']), [], defaultdict(list)
        bulk_tasks = key['m'] // 128 * panels * 4
        coordinates = set()
        for index, row in routes.items():
            owner = integer(row, 'warp') * comm + integer(row, 'cta')
            require(row['warp'] < 8 and row['cta'] < comm,
                    'Route queue owner mismatch')
            if index >= bulk_tasks:
                require(index < bulk_tasks + workers, 'Unexpected route drain index')
                # Drain indices are bulk_tasks + worker, not a strided task.
                require(owner == index - bulk_tasks, 'Drain owner mismatch')
                drains.append(row)
                continue
            require(owner == index % workers, 'Route queue owner mismatch')
            require(integer(row, 'row') % 64 == 0 and row['row'] < key['m']
                    and integer(row, 'column') % 128 == 0 and row['column'] < key['n']
                    and integer(row, 'rows') == 64 and integer(row, 'columns') == 128,
                    'Invalid route rectangle')
            coordinate = row['row'], row['column']
            require(coordinate not in coordinates, 'Duplicate route rectangle')
            coordinates.add(coordinate)
            for first, last in (('begin', 'ready'), ('ready', 'g2s_begin'), ('g2s_begin', 'g2s_done'),
                                ('g2s_done', 's2g_begin'), ('s2g_begin', 's2g_read_done'), ('s2g_read_done', 'copy_end')):
                interval(row, first, last)
            by_worker[owner].append(row)
        require(len(coordinates) == bulk_tasks and len(drains) == workers, 'Incomplete route/drain coverage')
        for owned in by_worker.values():
            owned.sort(key=lambda row: row['index'])
            if name == 'R':
                first = owned[0]
                stage_diagnostics.setdefault('first_route_prefix_ns', []).append(
                    first['ready'] - ctas[first['cta']]['begin'])
            for first, second in zip(owned, owned[1:]):
                require(first['copy_end'] <= second['begin'], 'Overlapping route tasks in one warp')
                if name == 'R':
                    values['ready_poll_us'].append(second['ready'] - first['copy_end'])
            if name == 'R':
                for row in owned:
                    values['g2s_us'].append(row['s2g_begin'] - row['ready'])
                    values['s2g_us'].append(row['copy_end'] - row['s2g_begin'])
        if name == 'R':
            for row in drains:
                values['drain_us'].append(interval(row, 'begin', 's2g_read_done'))
        else:
            # The two branches share the SAME slot origin. Mixed envelopes
            # already include issue, address calculation, fence and warp join.
            # Standalone/body-only duration must not be used as that origin.
            # A publishing chunk has its OWN measured completion and occupied
            # slot: Q-only publication is not a transferable DMA-context cost.
            # In particular, the model must not add Q publication to these
            # complete publishing intervals a second time.
            for owner, owned in by_worker.items():
                chunks = quant_by_worker[owner]
                position = 0
                for row in owned:
                    for label, begin, end in (('g2s', row['ready'], row['s2g_begin']),
                                              ('s2g', row['s2g_begin'], row['copy_end'])):
                        while position < len(chunks) and chunks[position]['end'] <= begin:
                            position += 1
                        if position == len(chunks) or chunks[position]['begin'] >= end:
                            continue
                        chunk = chunks[position]
                        require(begin <= chunk['begin'] <= chunk['end'] <= end, 'Quantization crosses occupied slot')
                        position += 1
                        if chunk['arrival_chunks']:
                            values['quant_' + label + '_publish_us'].append(chunk['end'] - begin)
                            values[label + '_publish_us'].append(end - begin)
                            if not always_publish:
                                continue
                        # In the provable every-chunk domain, old mixed fields
                        # also mean complete publishing intervals. Preserve
                        # that accounting without inventing nonpublish samples.
                        finish = chunk['end'] if always_publish else chunk['quant_done']
                        values['quant_' + label + '_us'].append(finish - begin)
                        values[label + '_mixed_us'].append(end - begin)

    for name in ('C_delay1', 'C_delay2'):
        if name not in captures:
            missing.append(name + ': controlled noninitial-panel recovery is unavailable')
    for stage_diagnostics in diagnostics.values():
        for field in ('first_chunk_prefix_ns', 'first_route_prefix_ns'):
            if field in stage_diagnostics:
                stage_diagnostics[field] = statistics(stage_diagnostics[field])
    for field in SERVICE_FIELDS:
        if field in ('issue_us', 'join_us') or (always_publish and field == 'quant_publish_us'):
            continue
        if not values[field]:
            missing.append('unmeasurable service: ' + field)
    stats = {field: statistics(samples) for field, samples in values.items() if samples}
    services = {field: stats[field]['mean_ns'] / 1000 for field in SERVICE_FIELDS if field in stats}
    services.update(issue_us=0.0, join_us=0.0)
    if always_publish:
        services['quant_publish_us'] = 0.0
    return dict(status='unavailable' if missing else 'validation_pending', unavailable_reasons=missing,
        physical_key=key, services=services, statistics=stats, stages=diagnostics,
        bulk_services=reduce_bulk_services(key, captures),
        publication_accounting='every_chunk_included' if always_publish else 'sparse_separate',
        boundary_accounting=dict(issue_us='included_in_slots', join_us='included_in_slots',
            tile_latency_us='measured_upper_bound_from_pre_release_stamp',
            quant_publish_us='included_in_every_chunk' if always_publish else 'Q_only_publishing_chunks',
            quant_g2s_publish_us='G2S_slot_origin_to_quant_end_including_publication',
            quant_s2g_publish_us='S2G_slot_origin_to_quant_end_including_publication',
            g2s_publish_us='G2S_slot_origin_to_s2g_begin_including_publication',
            s2g_publish_us='S2G_slot_origin_to_copy_end_including_publication',
            first_worker_queue_prefix='reported_separately_not_fitted_into_C_startup'),
        calibration_domain=dict(global_seq_min=131072, global_seq_max=524288,
            calibration_global_seq=131072, holdout_global_sequences=[262144, 524288],
            holdout_status='pending', k_interpolation_group=0),
        uncertainty='One capture per stage; within-epoch tiles/warps are correlated. '
                    'No independent-repetition confidence interval or fused-performance fit is claimed.')


def score_bulk_point(point, m):
    """Host arithmetic reference for the coarse OFFLINE policy, not acceptance.

    C uses directly observed first/cycle tile services, never C/waves. For P,
    fixed W work is already in the measured QR anchor. Only additional output
    volume grows at R's measured whole-service effective rate. R is not a TMA
    latency, and QR-R is not called quantization time (it can be negative due
    to pacing). No per-publication timing, fused sample or winner is consumed.
    This long-M extrapolation remains an approximation to be tested, not a
    promise that the standalone probes reproduce fused resource contention.
    """
    key = physical_key(point['physical_key'])
    bulk, services = point['bulk_services'], point['services']
    require(type(m) is int and key['m'] <= m <= 4 * key['m'] and m % key['tile_m'] == 0,
            'Bulk model is limited to the predeclared long-sequence domain')
    require((bulk['reference_m'], bulk['reference_n'], bulk['reference_k']) ==
            (key['m'], key['n'], key['k']), 'Bulk service reference mismatch')
    require(bulk['boundary'] == 'local_service_after_setup_before_cross_rank_finalize',
            'Bulk service boundary mismatch')
    for name in ('route_us', 'quant_route_us'):
        require(math.isfinite(bulk[name]) and bulk[name] > 0, 'Invalid bulk service')
    for name in ('tile_first_us', 'tile_cycle_us'):
        require(math.isfinite(services[name]) and services[name] > 0, 'Invalid direct compute service')
    require(math.isfinite(services['startup_us']) and services['startup_us'] >= 0, 'Invalid startup')
    tiles = (m // key['tile_m']) * (key['n'] // key['tile_n'])
    compute = min(tiles, key['sm_count'] - key['comm_ctas'])
    waves = (tiles + compute - 1) // compute
    c = services['tile_first_us'] + (waves - 1) * services['tile_cycle_us']
    p = bulk['quant_route_us'] + (m - key['m']) / key['m'] * bulk['route_us']
    return dict(m=m, comm_ctas=key['comm_ctas'], compute_ctas=compute, waves=waves,
        compute_us=c, production_us=p, score_us=services['startup_us'] + max(c, p),
        measurement_role='offline_prediction_not_fused_measurement')


def read_probe_sidecar(directory, attempt):
    """Read verified-archive payload without requiring a redundant extraction."""
    directory = Path(directory).resolve()
    name = 'control/services-rank-0.jsonl'
    path = directory / f'artifacts-attempt{attempt}' / name
    archive_path = directory / 'artifacts.tar.gz'
    require(archive_path.is_file() and not archive_path.is_symlink(), 'Unsafe or missing probe archive')
    limit = 1024 * 1024 * 1024
    # Large CP4 full-K32 captures exceed 256 MiB. Keep a finite bound and all
    # schema/coverage checks; do not sample away records to fit the file limit.
    actual = evidence.read_bytes(path, directory, limit=limit) if path.exists() else None
    identity = evidence.file_identity(archive_path.stat())
    with tarfile.open(archive_path, 'r:gz') as archive:
        entries = [member for member in archive if member.name == name]
        require(len(entries) == 1 and entries[0].isfile(), 'Missing/duplicate service sidecar in artifact')
        require(0 < entries[0].size <= limit, 'Service sidecar exceeds size limit')
        archived = archive.extractfile(entries[0]).read(limit + 1)
        require(len(archived) == entries[0].size, 'Incomplete service sidecar in artifact')
        require(actual is None or actual == archived, 'Service sidecar differs from audited artifact')
    require(identity == evidence.file_identity(archive_path.stat()), 'Artifact changed while reading sidecar')
    return archived, path


def audit_probe_run(directory):
    directory = Path(directory).resolve()
    requested = evidence.json_bytes(evidence.read_bytes(directory / 'job.json', directory))
    require(requested.get('mxfp8_service_probe') and requested.get('mxfp8') and requested.get('profile')
            and not requested.get('mpi'), 'Not an independent MXFP8 service-probe run')
    original = str(l20d.WORKSPACE)
    try:
        l20d.configure_workspace(requested['workspace'])
        job, receipts, data, receipts_evidence = evidence.read_receipts(directory)
    finally:
        l20d.configure_workspace(original)
    attempt = receipts['status.json']['attempt']
    actual, path = read_probe_sidecar(directory, attempt)
    key, captures = read_capture(data=actual)
    geometry = l20d.fused_geometry(job)
    comm, _, _ = l20d.fused_candidates(job)
    for field, expected in dict(world=job['world'], m=geometry['seq_local'], n=geometry['projection_width'],
            k=geometry['hidden'], q_heads=geometry['q_heads'], kv_heads=geometry['kv_heads'],
            head_dim=geometry['head_dim'], comm_ctas=comm[0],
            raster=int(job.get('qkv_raster') == 'along_n'), max_swizzle_size=job.get('max_swizzle_size', 8),
            epilogue_n=job.get('mxfp8_epilogue_n') or 64).items():
        require(key[field] == expected, 'Probe job/physical key mismatch: ' + field)
    require(not job.get('qkv_rank_swizzle') and (job.get('mxfp8_weight_preparation') or 'comm') == 'comm',
            'Service schedule differs from original communication-warps protocol')
    require(key['m'] * key['world'] == 131072,
            'Calibration anchor differs from the predeclared 128K protocol')
    telemetry = evidence.audit_telemetry(data['gpu-telemetry.csv'], receipts['gpu-before.json'], job)
    require(telemetry['devices'][0]['uuid'].lower() == captures['C_allready']['config'][0]['gpu_uuid'].lower(),
            'Captured CUDA GPU UUID differs from physical launch receipt')
    verified, unavailable = {}, {}
    for line in data[f'attempt{attempt}.log'].decode().splitlines():
        if line.startswith('service_stage_verified,'):
            fields = dict(part.split('=', 1) for part in line.split(',')[1:])
            stage = fields.get('stage')
            require(stage in captures and stage not in verified, 'Duplicate/unknown stage acceptance')
            expected_numeric = int(stage != 'R')
            expected_route = int(stage in ('R', 'QR', 'QR_phase1'))
            require(fields.get('payload_generations') == '2' and fields.get('full_numeric') == str(expected_numeric)
                    and fields.get('full_route') == str(expected_route) and fields.get('performance_accepted') == '0',
                    'Service stage missing independent validation')
            verified[stage] = fields
        elif line.startswith('service_unavailable,'):
            fields = dict(part.split('=', 1) for part in line.split(',')[1:])
            stage = fields.get('stage')
            require(stage in ('C_delay1', 'C_delay2') and stage not in captures and stage not in unavailable
                    and fields.get('reason') == 'all_panels_in_initial_compute_wave',
                    'Invalid unavailable-service declaration')
            unavailable[stage] = fields['reason']
    require(set(verified) == set(captures), 'Service stages not all validated')
    require(set(captures) | set(unavailable) == set(STAGES), 'Unexplained missing service stage')
    result = reduce_capture(key, captures)
    result['provenance'] = dict(run_id=job['run_id'], source_id=job['source_id'],
        build_inputs=receipts['fused-build.json']['build_inputs'],
        binary_sha256=receipts['fused-build.json']['binary_sha256'],
        environment_fingerprint=receipts['environment.json']['fingerprint'],
        artifact_sha256=receipts_evidence['artifacts.tar.gz']['sha256'],
        sidecar_sha256=hashlib.sha256(actual).hexdigest(), sidecar=str(path),
        source_manifest_path=str(directory / f'artifacts-attempt{attempt}/control/source-installed.json'),
        telemetry=telemetry, unavailable_stages=unavailable)
    return result


def export_cpp(points):
    """Return only the two MXFP8 declarations; never edit the BF16 table.

    Unavailable anchors remain in JSON but cannot become C++ constants. The
    development table is explicitly provisional until both declared holdouts
    are checked. All present observations are retained, never winner filtered.
    """
    require(points and all(p['status'] in ('validation_pending', 'unavailable') for p in points),
            'Unknown/empty calibration point state')
    available = [point for point in points if point['status'] == 'validation_pending']
    digest = hashlib.sha256(json.dumps(dict(points=points, model='bulk_v1'),
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    lines = ['// Generated by scripts/plan_sm103_mxfp8.py; direct intervals, no F fitting.',
             '// VALIDATION PENDING: S128K calibration; S256K and S512K are holdouts.',
             '// Whole local R/QR services; fixed W plus incremental output, not DMA latency.',
             f'inline constexpr char kMxfp8QkvCalibrationVersion[] = "validation_pending_{digest}";',
             f'inline constexpr std::array<Mxfp8QkvCalibrationPoint, {len(available)}> kMxfp8QkvCalibrationPoints = [] {{',
             f'  std::array<Mxfp8QkvCalibrationPoint, {len(available)}> points{{}};']
    fields = ('world', 'sm_count', 'capability', 'tile_m', 'tile_n', 'tile_k', 'epilogue_n', 'stages',
              'cluster_ctas', 'raster', 'comm_ctas', 'compute_ctas', 'k', 'dynamic_smem_bytes',
              'q_heads', 'kv_heads', 'head_dim')
    seen, provenance = set(), None
    for point in points:
        key = physical_key(point['physical_key'])
        signature = tuple(key[field] for field in PHYSICAL_FIELDS)
        require(signature not in seen, 'Duplicate physical service anchor')
        seen.add(signature)
        require(key['m'] * key['world'] == 131072, 'Header anchor is not the declared calibration sequence')
        source = point['provenance']
        identity = source['binary_sha256'], source['environment_fingerprint']
        require(provenance is None or provenance == identity, 'Cannot merge different probe binaries/environments')
        provenance = identity
    for i, point in enumerate(available):
        key, source = point['physical_key'], point['provenance']
        require(set(point['services']) == set(SERVICE_FIELDS), 'Incomplete service coefficients')
        require(point['calibration_domain'] == dict(global_seq_min=131072, global_seq_max=524288,
            calibration_global_seq=131072, holdout_global_sequences=[262144, 524288],
            holdout_status='pending', k_interpolation_group=0), 'Changed predeclared holdout domain')
        lines.extend([f'  // {source["run_id"]}; {point["publication_accounting"]}', '  {', f'    auto& p = points[{i}];'])
        lines.extend(f'    p.{field} = {key[field]};' for field in fields)
        lines.extend([f'    p.swizzle = {key["resolved_swizzle"]};',
                      f'    p.m_min = {131072 // key["world"]}; p.m_max = {524288 // key["world"]};',
                      f'    p.n_min = {key["n"]}; p.n_max = {key["n"]};', '    p.k_interpolation_group = 0;'])
        exported_fields = ('startup_us', 'tile_first_us', 'tile_cycle_us')
        for field in exported_fields:
            value = point['services'][field]
            require(math.isfinite(value) and value >= 0, 'Invalid service coefficient')
            lines.append(f'    p.services.{field} = {value:.17g};')
        # All observations remain in JSON; only decision inputs enter C++.
        score_bulk_point(point, key['m'])
        lines.append(f'    p.bulk.reference_m = {key["m"]};')
        for field in ('route_us', 'quant_route_us'):
            lines.append(f'    p.bulk.{field} = {point["bulk_services"][field]:.17g};')
        lines.append('  }')
    lines.extend(['  return points;', '}();', ''])
    return '\n'.join(lines)


def write_outputs(points, output_directory):
    # Fully audit and render before replacing either maintained output. Raw
    # sidecars remain in the fetched evidence; this is not a second raw archive.
    header = export_cpp(points)
    state = 'validation_pending' if any(p['status'] == 'validation_pending' for p in points) else 'unavailable'
    data = json.dumps(dict(schema=SCHEMA, status=state,
        decision_model='bulk_v1', points=points), indent=2, allow_nan=False) + '\n'
    output_directory.mkdir(parents=True, exist_ok=True)
    texts = {'calibration-current.json': data, 'calibration-current.inc': header}
    paths = [output_directory / name for name in texts]
    require(all(not path.is_symlink() for path in paths), 'Refusing symlink calibration output')
    with tempfile.TemporaryDirectory(dir=output_directory, prefix='.calibration-') as temporary:
        staged = Path(temporary)
        previous = {path: path.read_bytes() if path.exists() else None for path in paths}
        for name, text in texts.items():
            with (staged / name).open('w') as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
        replaced = []
        try:
            for path in paths:
                os.replace(staged / path.name, path)
                replaced.append(path)
        except OSError:
            for path in reversed(replaced):
                if previous[path] is None:
                    path.unlink()
                else:
                    restore = staged / ('restore-' + path.name)
                    restore.write_bytes(previous[path])
                    os.replace(restore, path)
            raise
    return paths


def score_points(points, global_sequences=(131072, 262144, 524288), *, compiler=None):
    """Run the production C++ selector against the audited, caller-owned table.

    This is a host prediction, not a GPU benchmark. No Python copy of the event
    model, coefficient correction or measured-F input participates. Completion
    timestamps (compute/Q/copy) overlap; they are NOT additive latency parts.
    Temporary source, macro-only CUTLASS stub and binary are always removed.
    """
    require(all(re.fullmatch(r'[A-Za-z0-9_.-]+', point.get('provenance', {}).get('run_id', ''))
                and point.get('publication_accounting') in ('every_chunk_included', 'sparse_separate')
                for point in points), 'Unsafe/noncanonical generated calibration comment')
    declarations = export_cpp(points)  # Same provenance/domain checks as header export.
    require(global_sequences and all(type(s) is int and s in (131072, 262144, 524288)
                                    for s in global_sequences), 'Only declared calibration/holdout sequences are scored')
    compiler = compiler or shutil.which('c++') or shutil.which('clang++') or shutil.which('g++')
    require(compiler is not None, 'A host C++17 compiler is required for the production score bridge')
    root = Path(__file__).resolve().parents[1]
    group_fields = tuple(field for field in PHYSICAL_FIELDS if field not in ('m', 'comm_ctas', 'compute_ctas'))
    groups = {}
    for point in points:
        key = point['physical_key']
        signature = tuple(key[field] for field in group_fields)
        group = groups.setdefault(signature, dict(physical_key={field: key[field] for field in group_fields},
                                                   comm_ctas=[]))
        group['comm_ctas'].append(key['comm_ctas'])
    requests, rows = [], []
    request_fields = ('m', 'n', 'k', 'world', 'sm_count', 'capability', 'tile_m', 'tile_n', 'tile_k',
                      'epilogue_n', 'stages', 'cluster_ctas', 'raster', 'max_swizzle_size', 'swizzle',
                      'comm_ctas', 'dynamic_smem_bytes', 'q_heads', 'kv_heads', 'head_dim')
    for group in groups.values():
        key = group['physical_key']
        for sequence in sorted(set(global_sequences)):
            row = dict(physical_key=key, global_seq=sequence, m=sequence // key['world'], candidates=[])
            rows.append(row)
            for comm in [*sorted(set(group['comm_ctas'])), 0]:
                request = dict(key, m=row['m'], comm_ctas=comm, swizzle=key['resolved_swizzle'])
                requests.append((len(rows) - 1, comm, ' '.join(str(request[field]) for field in request_fields)))
    fields = '\n'.join(f'    std::cin >> request.{field};' for field in request_fields)
    program = r'''
#include "csrc/operators/sm103/detail/autotune.cuh"
#include <chrono>
#include <iomanip>
#include <iostream>
namespace probe {
using fuse::detail::Mxfp8QkvCalibrationPoint;
''' + declarations + r'''
}
const char* status_name(fuse::detail::Mxfp8QkvTuningStatus value) {
  using S = fuse::detail::Mxfp8QkvTuningStatus;
  switch (value) {
    case S::Success: return "Success";
    case S::InvalidInput: return "InvalidInput";
    case S::UnsupportedCalibration: return "UnsupportedCalibration";
    case S::ModelFailure: return "ModelFailure";
  }
  return "Unknown";
}
int main() {
  std::cout << std::setprecision(17);
  size_t count = 0;
  if (!(std::cin >> count)) return 2;
  for (size_t index = 0; index < count; ++index) {
    fuse::detail::Mxfp8QkvTuningRequest request{};
''' + fields + r'''
    if (!std::cin) return 3;
    const auto begin = std::chrono::steady_clock::now();
    const auto result = fuse::detail::select_mxfp8_qkv_plan(request,
        probe::kMxfp8QkvCalibrationPoints.data(), probe::kMxfp8QkvCalibrationPoints.size());
    const auto query_us = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - begin).count();
    auto p = result.prediction;
    if (result.status != fuse::detail::Mxfp8QkvTuningStatus::Success) {
      // Failure can leave a nonfinite partial time. Emit valid JSON; Python
      // exposes missing predictions as null while retaining status/event counts.
      p.score_us = p.compute_finish_us = p.copy_finish_us = 0;
    }
    std::cout << "{\"index\":" << index << ",\"requested_comm_ctas\":" << request.comm_ctas
        << ",\"status\":\"" << status_name(result.status) << "\",\"model_status\":" << int(p.status)
        << ",\"selected_comm_ctas\":" << result.comm_ctas << ",\"compute_ctas\":" << result.compute_ctas
        << ",\"predicted_us\":" << p.score_us << ",\"compute_finish_us\":" << p.compute_finish_us
        << ",\"quant_finish_us\":null,\"copy_finish_us\":" << p.copy_finish_us
        << ",\"first_ready_us\":null,\"first_output_us\":null,\"critical_worker_feed_wait_us\":null"
        << ",\"exposed_feed_us\":null,\"output_tail_us\":null,\"events\":0"
        << ",\"scheduled_work_tiles\":" << p.scheduled_work_tiles << ",\"whole_service_model\":true"
        << ",\"host_query_us\":" << query_us << "}\n";
  }
}
'''
    with tempfile.TemporaryDirectory(prefix='fuse-mxfp8-score-') as temporary:
        directory = Path(temporary)
        (directory / 'cutlass').mkdir()
        (directory / 'cutlass/cutlass.h').write_text('#pragma once\n#define CUTLASS_HOST_DEVICE\n')
        source, binary = directory / 'score.cc', directory / 'score'
        source.write_text(program)
        built = subprocess.run([str(compiler), '-std=c++17', '-O2', '-I', str(directory),
            '-I', str(root), '-I', str(root / 'include'), str(source), '-o', str(binary)],
            capture_output=True, text=True, timeout=120)
        require(built.returncode == 0, 'Host score bridge compilation failed: ' + built.stderr[-4000:])
        execution = subprocess.run([str(binary)], input=str(len(requests)) + '\n' +
            '\n'.join(request[2] for request in requests) + '\n', capture_output=True, text=True, timeout=120)
        require(execution.returncode == 0, 'Host score bridge failed: ' + execution.stderr[-4000:])
    predictions = [evidence.json_bytes(line.encode()) for line in execution.stdout.splitlines()]
    require(len(predictions) == len(requests), 'Incomplete host score bridge output')
    for index, (prediction, (row_index, comm, _)) in enumerate(zip(predictions, requests)):
        require(prediction.pop('index') == index and prediction['requested_comm_ctas'] == comm,
                'Host score bridge request/output mismatch')
        if prediction['status'] != 'Success':
            for field in tuple(prediction):
                if field.endswith('_us') and field != 'host_query_us': prediction[field] = None
            prediction['selected_comm_ctas'] = None
            prediction['compute_ctas'] = None
        elif prediction['whole_service_model']:
            # The bulk score does not predict individual readiness or feed
            # stalls. Null is unknown; zero would falsely claim they vanished.
            for field in ('quant_finish_us', 'first_ready_us', 'first_output_us',
                          'critical_worker_feed_wait_us', 'exposed_feed_us', 'output_tail_us'):
                prediction[field] = None
        if comm: rows[row_index]['candidates'].append(prediction)
        else: rows[row_index]['auto'] = prediction
    return dict(schema='sm103_mxfp8_host_prediction_v1', source='production_cpp_select_mxfp8_qkv_plan',
        calibration_sha256=hashlib.sha256(declarations.encode()).hexdigest(),
        performance_accepted=False, coefficient_fit=False,
        note='Host predictions only. Compute and role completion estimates overlap, not additive parts. '
             'For whole_service_model, copy_finish includes quantization+route and unmodeled stages are null. '
             'S256K/S512K remain holdouts; predictions do not establish GPU validation.', rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--score', action='store_true', help='Run the actual C++ selector for 128K/256K/512K predictions')
    args = parser.parse_args()
    points = [audit_probe_run(path) for path in args.runs]
    paths = write_outputs(points, args.output)
    if args.score:
        prediction_path = args.output / 'predictions-current.json'
        require(not prediction_path.is_symlink(), 'Refusing symlink prediction output')
        prediction_path.write_text(json.dumps(score_points(points), indent=2, allow_nan=False) + '\n')
        paths.append(prediction_path)
    print(f'{len(points)} audited service anchors; {sum(p["status"] == "unavailable" for p in points)} unavailable')
    for path in paths:
        print(path)


if __name__ == '__main__':
    main()
