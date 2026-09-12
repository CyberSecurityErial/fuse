#!/usr/bin/env python3
"""Audit fetched fused-smoke evidence and export a small, local-only dataset.

Usage: python3 scripts/summarize_sm103_fused.py --artifacts <run-or-attempt-dir>
       [<another-dir> ...] --output <new-directory>

The archive's status is captured before upload (running/collect); fetched.json
is the terminal receipt. Both are checked, including the archive and extracted
evidence bytes. Binary hashes are remote build attestations, not local binaries.
Fused and opt-in C/R components have distinct acceptance/validation domains.
The C/R schema follows fused_bf16's actual --calibrate implementation.
"""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
import tarfile
import tempfile

from l20d import NODES, fused_binary, fused_build_inputs, fused_candidates, fused_devices, fused_geometry, fused_policy_tile


SCHEMA = 'sm103_fused_verified_v1'
DIRECTIONS = ('GEMM_A2A', 'A2A_GEMM')
COMPONENTS = ('fused', 'compute_reference', 'copy_reference')
COLLECTORS = ('per_epoch_rank_events_v2', 'per_epoch_rank_events_v3_eventsync')
SHA256 = re.compile(r'[0-9a-f]{64}')
CONTROL_FILES = ('job.json', 'status.json', 'source-installed.json', 'environment.json',
                 'fused-build.json', 'gpu-before.json', 'gpu-telemetry.csv', 'attempt{attempt}.log')
KINDS = {'config', 'device', 'input', 'candidate', 'component_resources', 'correctness', 'route', 'warmup',
         'sample', 'summary', 'candidate_verified', 'validation_self_test', 'validation_oracle',
         'profile_host', 'host_stage', 'input_oracle', 'epilogue_resources', 'epilogue_sample', 'epilogue_cta',
         'graph_prepare', 'auto_comm', 'quant_validation', 'producer_validation'}
GRAPH_EPOCH_MODE = 'recapture_update_v1'
GRAPH_PREPARE_FIELDS = {'kind', 'line', 'label', 'candidate', 'comm_sm', 'tile', 'generation',
    'component', 'rank', 'launch', 'graph_epoch_mode', 'calls', 'first_epoch', 'last_epoch',
    'wall_s', 'includes', 'gpu_sample_time'}
EPILOGUE_MODES = ('production', 'role_telemetry', 'epilogue_telemetry')
EPILOGUE_PHASES = tuple('epilogue_' + mode for mode in EPILOGUE_MODES) + ('epilogue_record',)
# v21 predates an explicit role timestamp join record. Only this immutable
# harness may omit it; future/missing-source logs cannot fall back silently.
LEGACY_EPILOGUE_JOIN_HARNESS_SHA256 = {'59642de984f26f6b88b3d658f1423aca44d15a77c03cba02434e0c41de5464c4'}
EPILOGUE_FIELDS = {
    'epilogue_resources': ('rank', 'diagnostic_kind', 'schema', 'clock', 'clock_unit', 'store_interval',
        'drain_interval', 'record_bytes', 'regs', 'local_bytes', 'static_smem', 'dynamic_smem',
        'max_threads', 'cluster_ctas', 'tile_m', 'tile_n', 'tile_k', 'performance_accepted'),
    'epilogue_sample': ('rank', 'sample', 'diagnostic_kind', 'epoch', 'warmup', 'samples', 'host_launch',
        'launch', 'process_layout', 'performance_accepted', 'final_sample_poisoned', 'event_ms'),
    'epilogue_cta': ('rank', 'cta', 'epoch', 'schema', 'clock', 'clock_unit', 'tile_count', 'first_m_tile',
        'first_n_tile', 'first_batch', 'store_ns_sum', 'store_ns_max', 'drain_ns_sum', 'drain_ns_max',
        'first_store_begin', 'first_store_end', 'first_drain_end', 'first_ready_after', 'last_ready_after',
        'cta_start', 'cta_role_done', 'cta_end', 'performance_accepted'),
}
HOST_STAGES = ('policy_device_validation', 'communication_prepare', 'arguments',
               'implement_workspace', 'lower_parameters', 'launch_setup', 'cuda_enqueue')
SCHEDULE_REQUEST_FIELDS = ('max_swizzle_size', 'qkv_raster', 'oproj_raster')
SCHEDULE_CONFIG_FIELDS = SCHEDULE_REQUEST_FIELDS + ('qkv_effective_raster', 'oproj_effective_raster')
SCHEDULE_ROW_FIELDS = ('raster', 'max_swizzle_size', 'effective_swizzle_size',
                       'padded_m_tiles', 'padded_n_tiles', 'scheduled_compute_ctas')
# First frozen harness with explicit scheduling records. The manifest also
# prevents a stripped new default log from masquerading as a legacy log.
# Explicit job/log fields activate the contract for later source versions too.
SCHEDULE_HARNESS_SHA256 = {'766bb5da2a2b20319a744c74af06495a092ffe3eeaac231ebb467a520693e064'}
OPROJ_PROBE_FIELDS = {
    'profile_oproj_pipeline': ('rank', 'index', 'cta', 'mma_begin', 'tmem_acquired', 'mma_return',
        'epi_begin', 'acc_wait_begin', 'acc_wait_end', 'tmem_release_begin', 'tmem_release_end',
        'epi_return', 'k_tiles'),
    'profile_oproj_stage': ('rank', 'index', 'k', 'wait_begin', 'wait_end', 'issue_end'),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_digest(value):
    return digest(json.dumps(value, sort_keys=True).encode())


def finite(value, label, minimum=None):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{label}: expected a finite number') from error
    require(not isinstance(value, bool) and math.isfinite(result), f'{label}: nonfinite number')
    require(minimum is None or result >= minimum, f'{label}: below {minimum}')
    return result


def integer(value, label, minimum=0):
    require(not isinstance(value, bool) and re.fullmatch(r'-?\d+', str(value)) is not None,
            f'{label}: expected integer')
    result = int(value)
    require(result >= minimum, f'{label}: below {minimum}')
    return result


def number(row, field, minimum=None):
    require(field in row, f'line {row.get("line")}: missing {field}')
    return finite(row[field], field, minimum)


def count(row, field, minimum=0):
    require(field in row, f'line {row.get("line")}: missing {field}')
    return integer(row[field], field, minimum)


def close(actual, expected, label):
    # Logs contain nine significant digits. Allow formatting error, not a
    # measurement tolerance or rounding away an invalid sample.
    require(math.isclose(actual, expected, rel_tol=2e-7, abs_tol=2e-9),
            f'{label}: {actual} != recomputed {expected}')


def file_identity(status):
    # Reading can legitimately update atime; it is not a content mutation.
    return status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns


def read_bytes(path, root, limit=64 * 1024 * 1024):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root),
            f'Unsafe or missing evidence: {path}')
    before = path.stat()
    require(before.st_size <= limit, f'Evidence exceeds size limit: {path}')
    data = path.read_bytes()
    require(file_identity(before) == file_identity(path.stat()) and len(data) == before.st_size,
            f'Evidence changed: {path}')
    return data


def file_digest(path):
    before = path.stat()
    require(path.is_file() and not path.is_symlink(), f'Unsafe archive: {path}')
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    require(file_identity(before) == file_identity(path.stat()), f'Archive changed: {path}')
    return value.hexdigest()


def json_bytes(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f'Nonfinite JSON value: {value}')
    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def safe_member(name):
    path = PurePosixPath(name)
    require(name and not path.is_absolute() and '..' not in path.parts and '\\' not in name,
            f'Unsafe archive member: {name}')
    return path.as_posix()


def read_receipts(directory):
    directory = Path(directory).absolute()
    require(not directory.is_symlink(), f'Symlink input directory: {directory}')
    directory = directory.resolve()
    root = directory.parent if (directory / 'control').is_dir() else directory
    fetched_data = read_bytes(root / 'fetched.json', root)
    fetched = json_bytes(fetched_data)
    attempt = integer(fetched.get('attempt'), 'attempt', 1)
    artifact_dir = root / f'artifacts-attempt{attempt}'
    require(directory in (root, artifact_dir), 'Input attempt differs from fetched receipt')
    require(fetched.get('state') == 'succeeded' and fetched.get('phase') == 'finished' and
            fetched.get('exit_code') == 0 and fetched.get('work_exit_code') == 0 and
            not fetched.get('collection_error') and not fetched.get('error'),
            'A successful terminal fetched receipt is required')
    archive = root / 'artifacts.tar.gz'
    require(file_digest(archive) == fetched.get('artifact_sha256'), 'Artifact SHA256 mismatch')
    files = {f'control/{name.format(attempt=attempt)}' for name in CONTROL_FILES}
    data, evidence, seen = {}, {}, set()
    archive_stat = archive.stat()
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            name = safe_member(member.name)
            require(name not in seen and (member.isdir() or member.isfile()),
                    f'Duplicate/link/special archive member: {name}')
            seen.add(name)
            mpi_member = re.fullmatch(
                rf'control/mpi-(?:runtime-attempt{attempt}\.json|logs-attempt{attempt}\.json|'
                rf'attempt{attempt}-rank-[0-7]\.(?:stdout|stderr)\.log)', name)
            if name not in files and not mpi_member:
                continue
            require(member.isfile() and member.size <= 64 * 1024 * 1024,
                    f'Invalid evidence member: {name}')
            archived = tar.extractfile(member).read()
            path = artifact_dir / name
            actual = read_bytes(path, root)
            require(actual == archived, f'Extracted evidence differs from archive: {name}')
            data[name.removeprefix('control/')] = actual
            evidence[name] = {'path': str(path), 'sha256': digest(actual), 'bytes': len(actual)}
    require(file_identity(archive.stat()) == file_identity(archive_stat), 'Archive changed during verification')
    require(files <= seen, f'Missing archive evidence: {sorted(files - seen)}')
    records = {name: json_bytes(value) for name, value in data.items() if name.endswith('.json')}
    job, status = records['job.json'], records['status.json']
    require(job.get('stage') == 'fused-smoke', 'Only fused-smoke artifacts are supported')
    for key in ('run_id', 'node', 'stage', 'experiment', 'source_id'):
        require(job.get(key) == fetched.get(key) == status.get(key), f'Receipt mismatch: {key}')
    require(status.get('attempt') == attempt and status.get('exit_code') == 0 and
            status.get('work_exit_code') == 0 and not status.get('error'), 'Invalid archived status')
    require((status.get('state'), status.get('phase')) in
            {('running', 'collect'), ('succeeded', 'finished')}, 'Unexpected archived status phase')
    installed = records['source-installed.json']
    require(isinstance(job.get('files'), dict) and job['files'], 'Missing source manifest')
    for name, value in job['files'].items():
        safe_member(name)
        require(isinstance(value, str) and (SHA256.fullmatch(value) or value.startswith('link:')),
                f'Invalid source digest: {name}')
    require(job['source_id'] == json_digest(job['files']) and
            installed == {'source_id': job['source_id'], 'files': job['files']},
            'Installed source manifest mismatch')
    if (root / 'job.json').exists():
        require(json_bytes(read_bytes(root / 'job.json', root)) == job, 'Local/archived job mismatch')
    environment = records['environment.json']
    require(job['node'] in NODES and environment.get('host') == NODES[job['node']][1],
            'Environment host does not match selected node')
    env_id = environment.get('fingerprint')
    require(env_id == json_digest({k: v for k, v in environment.items() if k != 'fingerprint'}) and
            env_id == fetched.get('environment_fingerprint') == status.get('environment_fingerprint'),
            'Environment fingerprint mismatch')
    build = records['fused-build.json']
    expected_binary = str(fused_binary(job))
    require(build.get('qkv_rank_swizzle', 'off') ==
            ('rank_n_band_v1' if job.get('qkv_rank_swizzle') else 'off'), 'Rank swizzle build mismatch')
    require(build.get('node') == job['node'] and build.get('profile') == bool(job.get('profile')) and
            build.get('binary') == expected_binary and SHA256.fullmatch(build.get('binary_sha256', '')) and
            build.get('build_inputs') == fused_build_inputs(job) and
            build.get('environment_fingerprint') == env_id, 'Build receipt mismatch')
    require(bool(build.get('mpi')) == bool(job.get('mpi')), 'MPI build identity mismatch')
    if job.get('mpi'):
        require(not job.get('profile') and environment.get('mpi_toolchain') and
                build.get('mpi_toolchain') == environment['mpi_toolchain'], 'MPI toolchain receipt mismatch')
        audit_mpi_receipts(job, records, data, attempt)
    else:
        require(not any(name.startswith('mpi-') for name in data), 'MPI evidence in a non-MPI run')
    evidence['fetched.json'] = {'path': str(root / 'fetched.json'), 'sha256': digest(fetched_data)}
    evidence['artifacts.tar.gz'] = {'path': str(archive), 'sha256': fetched['artifact_sha256']}
    return job, records, data, evidence


def audit_mpi_receipts(job, records, data, attempt):
    """Audit each original rank stream; merged bytes are not global time order."""
    runtime = records.get(f'mpi-runtime-attempt{attempt}.json', {})
    manifest = records.get(f'mpi-logs-attempt{attempt}.json', {})
    launch = fused_launch(job)
    collector = 'mpi_graph_rank_events_v1' if launch == 'graph' else 'mpi_rank_events_v1'
    for key, value in dict(schema='sm103_mpi_runtime_v1', node=job['node'], world=job['world'],
                           process_layout='mpi_one_process_per_gpu', host_launch='mpi_process',
                           launch=launch, collector=collector,
                           boundary=f'mpi_{launch}_maxrank_cudaevent').items():
        require(runtime.get(key) == value, f'MPI runtime contract mismatch: {key}')
    require(runtime.get('graph_epoch_mode') == (GRAPH_EPOCH_MODE if launch == 'graph' else None),
            'MPI runtime Graph epoch contract mismatch')
    require(runtime.get('overrides') == records['environment.json']['mpi_toolchain']['overrides'] and
            runtime['overrides'].get('UCX_TLS') == 'sm,self', 'MPI transport/compiler mismatch')
    argv = runtime.get('argv', [])
    prefix = '/root/workspace_wct/toolchain/mpich-5.0.1.post1'
    require(isinstance(argv, list) and argv[:5] ==
            [prefix + '/bin/mpiexec', '-launcher', 'fork', '-n', str(job['world'])] and
            records['fused-build.json']['binary'] in argv, 'MPI launcher identity mismatch')
    launch_flags = [index for index, value in enumerate(argv) if value == '--launch']
    require((not launch_flags and launch == 'eager') or
            (len(launch_flags) == 1 and launch_flags[0] + 1 < len(argv) and
             argv[launch_flags[0] + 1] == launch), 'MPI launcher launch flag mismatch')
    merged_name = f'attempt{attempt}.log'
    merged = data[merged_name]
    require(manifest.get('schema') == 'sm103_mpi_rank_logs_v1' and manifest.get('complete') is True and
            manifest.get('collector') == collector and
            manifest.get('ordering') == 'rank_then_stream_not_global_chronological' and
            manifest.get('merged_log') == merged_name and manifest.get('merged_sha256') == digest(merged),
            'MPI merged-log receipt mismatch')
    streams = manifest.get('ranks', [])
    require(isinstance(streams, list) and len(streams) == job['world'] * 2, 'Missing MPI rank streams')
    expected = [(rank, stream) for rank in range(job['world']) for stream in ('stdout', 'stderr')]
    require([(row.get('rank'), row.get('stream')) for row in streams] == expected,
            'MPI stream rank/order mismatch')
    previous_end = 0
    for row in streams:
        name = f'mpi-attempt{attempt}-rank-{row["rank"]}.{row["stream"]}.log'
        require(row.get('path') == name and row.get('present') is True and row.get('rank_started') is True and
                name in data, 'MPI rank stream missing or unexpected')
        original = data[name]
        begin, end = integer(row.get('merged_begin'), 'merged_begin'), integer(row.get('merged_end'), 'merged_end')
        require(previous_end <= begin <= end <= len(merged) and merged[begin:end] == original and
                row.get('sha256') == digest(original) and row.get('bytes') == len(original),
                'MPI original rank bytes/hash/merged range mismatch')
        outside, diagnostics = parse_log(merged[previous_end:begin].decode('utf-8'), completion='none')
        require(not outside and not diagnostics, 'MPI merged gap contains unowned harness evidence')
        previous_end = end
        if row['stream'] == 'stdout':
            parsed, _ = parse_log(original.decode('utf-8'), completion='last' if row['rank'] == 0 else 'none')
            devices = [entry for entry in parsed if entry['kind'] == 'device']
            require(len(devices) == 1 and count(devices[0], 'rank') == row['rank'],
                    'MPI rank startup belongs to a different process')
            configs = [entry for entry in parsed if entry['kind'] == 'config']
            require(len(configs) == (1 if row['rank'] == 0 else 0), 'MPI root config count mismatch')
            native_generations = defaultdict(list)
            for entry in parsed:
                native = entry['kind'] in ('device', 'input', 'candidate', 'component_resources', 'graph_prepare', 'auto_comm') or (
                    entry['kind'] == 'input_oracle' and 'rank' in entry)
                if native:
                    require(count(entry, 'rank') == row['rank'], 'MPI native record belongs to another rank')
                    if entry['kind'] in ('candidate', 'component_resources', 'graph_prepare'):
                        key = entry['kind'], entry.get('candidate'), entry.get('component', 'fused')
                        native_generations[key].append(count(entry, 'generation'))
                else:
                    require(row['rank'] == 0, 'MPI nonroot emitted root-only harness evidence')
            require(all(generations == [0, 1] for generations in native_generations.values()),
                    'MPI native payload generations are out of local order')
        else:
            parsed, diagnostics = parse_log(original.decode('utf-8'), completion='none')
            require(not parsed and not diagnostics, 'MPI stderr contains harness evidence')
    outside, diagnostics = parse_log(merged[previous_end:].decode('utf-8'), completion='none')
    require(not outside and not diagnostics, 'MPI merged tail contains unowned harness evidence')
    required_names = {f'mpi-runtime-attempt{attempt}.json', f'mpi-logs-attempt{attempt}.json'} | {
        f'mpi-attempt{attempt}-rank-{rank}.{stream}.log' for rank, stream in expected}
    require({name for name in data if name.startswith('mpi-')} == required_names,
            'MPI evidence contains unexpected rank streams')


def audit_oproj_probe_record(parts, world, line_number):
    """Validate compact probe records; full tile coverage belongs to the trace exporter."""
    kind, values = parts[0], {}
    for part in parts[1:]:
        require('=' in part, f'Malformed OProj probe field at line {line_number}')
        key, value = part.split('=', 1)
        require(key not in values, f'Duplicate OProj probe field at line {line_number}')
        values[key] = integer(value, f'OProj probe {key}')
    fields = set(OPROJ_PROBE_FIELDS[kind])
    if kind == 'profile_oproj_pipeline':
        fields.update(f'{name}{peer}' for peer in range(world)
                      for name in ('ready_begin', 'ready_end', 'ready_joined', 'cache_hit'))
    require(set(values) == fields and values['rank'] == 0,
            f'OProj probe field/rank mismatch at line {line_number}')
    chains = [('wait_begin', 'wait_end', 'issue_end')]
    if kind == 'profile_oproj_pipeline':
        require(values['k_tiles'] > 0 and all(values[f'cache_hit{peer}'] in (0, 1) for peer in range(world)),
                f'Invalid OProj probe K count/cache hit at line {line_number}')
        chains = [('mma_begin', 'tmem_acquired', 'mma_return'),
                  ('epi_begin', 'acc_wait_begin', 'acc_wait_end', 'tmem_release_begin',
                   'tmem_release_end', 'epi_return')]
        chains += [tuple(f'{name}{peer}' for name in ('ready_begin', 'ready_end', 'ready_joined'))
                   for peer in range(world)]
    for chain in chains:
        times = [values[field] for field in chain]
        require(0 < times[0] and times == sorted(times) and times[-1] < 2**64,
                f'Invalid OProj probe phase order at line {line_number}')


def parse_log(text, *, completion='last', components=COMPONENTS):
    rows, diagnostics = [], Counter()
    profile_detail = None  # Old logs predate explicit full/CTA-only selection.
    profile_world = 0
    lines = text.splitlines()
    require(completion in ('last', 'any', 'none'), 'Unknown log completion contract')
    passes = [index for index, line in enumerate(lines)
              if line.startswith(('PASS: both BF16 boundaries, complete routes, changed payloads',
                                  'PASS: selected BF16 boundaries, complete routes, changed payloads'))]
    require((completion == 'none' and not passes) or
            (completion == 'any' and len(passes) == 1) or
            (completion == 'last' and passes == [len(lines) - 1]), 'Missing final harness PASS')
    for line_number, line in enumerate(lines, 1):
        parts = line.split(',')
        kind = parts[0]
        require(not (kind == 'stage_time' and ',status=failed' in line), 'Failed harness stage in log')
        require(not kind.startswith('epilogue_') or kind in EPILOGUE_FIELDS,
                f'Unknown epilogue record at line {line_number}')
        require(not kind.startswith('graph_') or kind == 'graph_prepare',
                f'Unknown Graph record at line {line_number}')
        if kind in OPROJ_PROBE_FIELDS:
            # These two compact emitters omit profile_detail intentionally;
            # they are full-only records, not a blanket exception for profiles.
            require(profile_detail == 'full' and profile_world > 0,
                    f'OProj probe requires explicit full profile at line {line_number}')
            audit_oproj_probe_record(parts, profile_world, line_number)
        elif kind.startswith('profile') or kind == 'host_stage':
            detail_fields = [part for part in parts[1:] if part.startswith('profile_detail=')]
            require(detail_fields == ([] if profile_detail is None else ['profile_detail=' + profile_detail]),
                    f'Per-record profile detail mismatch at line {line_number}')
        if kind.startswith('profile'):
            diagnostics[kind] += 1
            if kind != 'profile_host':
                continue
        elif kind == 'host_stage':
            diagnostics[kind] += 1
        elif kind in EPILOGUE_FIELDS:
            diagnostics[kind] += 1
        if kind not in KINDS:
            continue
        row = {'kind': kind, 'line': line_number}
        for part in parts[1:]:
            if '=' not in part:
                require('label' not in row, f'Unexpected positional field at line {line_number}')
                row['label'] = part
                continue
            key, value = part.split('=', 1)
            # The harness's host_stage kind describes the diagnostic kernel,
            # whereas our kind identifies the record prefix.
            if (kind == 'host_stage' or kind in EPILOGUE_FIELDS) and key == 'kind':
                key = 'diagnostic_kind'
            require(key not in row, f'Duplicate field {key} at line {line_number}')
            require(value.lower() not in ('nan', 'inf', '+inf', '-inf', 'infinity', '-infinity'),
                    f'Nonfinite field at line {line_number}')
            row[key] = value
        require(row.get('component', 'fused') in components, 'Unsupported measurement component')
        require(kind != 'quant_validation' or row.get('component') == 'quantize_reference',
                'Quantization validation requires its explicit reference boundary')
        require(kind != 'producer_validation' or row.get('component') == 'producer_reference',
                'Joint production validation requires its explicit reference boundary')
        if kind == 'config':
            profile_detail = row.get('profile_detail')
            profile_world = count(row, 'world', 1)
        rows.append(row)
    return rows, dict(diagnostics)


def percentile(values, quantile):
    values = sorted(values)
    position = quantile * (len(values) - 1)
    lo = int(position)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (position - lo)


def drift(values):
    middle = len(values) // 2
    return abs(percentile(values[:middle], .5) - percentile(values[middle:], .5)) / percentile(values, .5)


def audit_samples(rows, world, expected_count, label):
    require(len(rows) == expected_count, f'{label}: requires {expected_count} raw samples')
    require([count(row, 'index') for row in rows] == list(range(expected_count)),
            f'{label}: missing/duplicate/out-of-order sample indices')
    samples, ranks, epochs = [], [], []
    expected_rank_keys = {f'rank{rank}_ms' for rank in range(world)}
    for row in rows:
        require({key for key in row if re.fullmatch(r'rank\d+_ms', key)} == expected_rank_keys,
                f'{label}: missing/extra rank timing')
        # CUDA event results are float32. Nine significant digits round-trip
        # those values exactly; restore them before the C++ double percentile
        # and small-difference drift calculations.
        times = [struct.unpack('<f', struct.pack('<f', number(row, f'rank{rank}_ms', 0)))[0]
                 for rank in range(world)]
        require(min(times) > 0, f'{label}: nonpositive rank timing')
        value = max(times)
        logged_max = struct.unpack('<f', struct.pack('<f', number(row, 'maxrank_ms', 0)))[0]
        require(logged_max == value, f'{label}: maxrank differs from rank samples')
        epochs.append(count(row, 'epoch', 1))
        samples.append(value)
        ranks.append(times)
    require(all(b == a + 1 for a, b in zip(epochs, epochs[1:])), f'{label}: nonconsecutive epochs')
    return {'maxrank_ms': samples, 'rank_ms': ranks, 'epoch_first': epochs[0],
            'epoch_last': epochs[-1], 'lines': [row['line'] for row in rows]}


def fused_launch(job):
    launch = job.get('fused_launch', 'eager')
    require(launch in ('eager', 'graph'), 'Unknown fused launch mode')
    require(launch != 'graph' or (job.get('mpi') and not job.get('profile')),
            'Graph requires MPI without profiling')
    return launch


def audit_quick_timing(rows, world, mpi, launch):
    summaries = [r for r in rows if r['kind'] == 'summary']
    require(len(summaries) == 1, 'Quick mode requires one summary')
    summary = summaries[0]
    require(summary.get('sampling_mode') == 'quick_1_5' and
            summary.get('verification') == 'pending' and count(summary, 'formal_eligible') == 0 and
            count(summary, 'warmup') == 1 and count(summary, 'samples') == 5 and
            count(summary, 'additional_warmup_calls') == count(summary, 'sample_cadence_warmup') == 0 and
            count(summary, 'minimum_warmup_cuda_ms') == count(summary, 'converged_all_ranks') == 0 and
            count(summary, 'selected_round') == 0,
            'Invalid 1+5 screening contract')
    expected_collector = ('mpi_graph_rank_events_v1' if launch == 'graph' else 'mpi_rank_events_v1') if mpi else 'per_epoch_rank_events_v3_eventsync'
    require(summary.get('collector') == expected_collector and
            summary.get('boundary') == (f'mpi_{launch}_maxrank_cudaevent' if mpi else
                                       'single_process_eager_maxrank_cudaevent'), 'Quick boundary mismatch')
    require((summary.get('launch') == 'graph' and summary.get('graph_epoch_mode') == GRAPH_EPOCH_MODE)
            if launch == 'graph' else 'graph_epoch_mode' not in summary, 'Quick Graph mode mismatch')
    warm = [r for r in rows if r['kind'] == 'warmup']
    samples = [r for r in rows if r['kind'] == 'sample']
    require(all(r.get('phase') == 'initial' and count(r, 'round', -1) == -1 for r in warm) and
            all(r.get('phase') == 'measurement' and count(r, 'round') == 0 for r in samples),
            'Quick mode has extra warmup/measurement phases')
    initial = audit_samples(warm, world, 1, 'quick warmup')
    values = audit_samples(samples, world, 5, 'quick samples')
    require(values['epoch_first'] == initial['epoch_last'] + 1 and
            initial['lines'][-1] < values['lines'][0] < values['lines'][-1] < summary['line'],
            'Quick epochs/records are not consecutive')
    close(number(summary, 'warmup_p50_ms'), initial['maxrank_ms'][0], 'quick warmup p50')
    close(number(summary, 'warmup_p95_ms'), initial['maxrank_ms'][0], 'quick warmup p95')
    p50, p95, half = percentile(values['maxrank_ms'], .5), percentile(values['maxrank_ms'], .95), drift(values['maxrank_ms'])
    for key, value in (('p50_ms', p50), ('p95_ms', p95), ('half_drift', half)):
        close(number(summary, key), value, key)
    return dict(p50_ms=p50, p95_ms=p95, half_drift=half, selected_round=0,
                collector=expected_collector, boundary=summary['boundary'], warmup_calls=1,
                minimum_warmup_cuda_ms=0, sampling_mode='quick_1_5', formal_eligible=False,
                rounds=[values | dict(round=0, half_drift=half, accepted=True)], summary_line=summary['line'])


def audit_timing(rows, world, mpi=False, launch='eager', quick=False):
    if quick:
        return audit_quick_timing(rows, world, mpi, launch)
    summaries = [r for r in rows if r['kind'] == 'summary']
    require(len(summaries) == 1, 'Requires exactly one pending timing summary per candidate')
    summary = summaries[0]
    require(summary.get('verification') == 'pending' and count(summary, 'samples') == 50 and
            count(summary, 'warmup') >= 10 and count(summary, 'sample_cadence_warmup') >= 10 and
            count(summary, 'converged_all_ranks') == 1 and count(summary, 'stable_5pct') == 1 and
            summary.get('collector') in (('mpi_graph_rank_events_v1' if launch == 'graph' else
                                         'mpi_rank_events_v1',) if mpi else COLLECTORS) and
            summary.get('boundary') == (f'mpi_{launch}_maxrank_cudaevent' if mpi else
                                         'single_process_eager_maxrank_cudaevent'),
            'Unsupported or incomplete sampling contract')
    require((summary.get('launch') == 'graph' and summary.get('graph_epoch_mode') == GRAPH_EPOCH_MODE)
            if launch == 'graph' else ('graph_epoch_mode' not in summary and
                                      summary.get('launch', 'eager') == 'eager'),
            'Sampling Graph epoch/launch mismatch')
    minimum_ms = number(summary, 'minimum_warmup_cuda_ms', 100)
    selected = count(summary, 'selected_round')
    require(selected <= 2, 'Sampling retry budget exceeded')
    warm = [r for r in rows if r['kind'] == 'warmup']
    initial = audit_samples([r for r in warm if r.get('phase') == 'initial'], world,
                            count(summary, 'warmup'), 'initial warmup')
    cadence = audit_samples([r for r in warm if r.get('phase') == 'sample_cadence'], world,
                            count(summary, 'sample_cadence_warmup'), 'cadence warmup')
    close(number(summary, 'warmup_p50_ms'), percentile(initial['maxrank_ms'], .5), 'warmup p50')
    close(number(summary, 'warmup_p95_ms'), percentile(initial['maxrank_ms'], .95), 'warmup p95')
    convergence = [r for r in warm if r.get('phase') == 'convergence']
    require(len(warm) == len(initial['maxrank_ms']) + len(cadence['maxrank_ms']) + len(convergence),
            'Unknown warmup phase')
    window_counts, final_epochs = set(), set()
    for rank in range(world):
        windows = [r for r in convergence if count(r, 'rank') == rank]
        require(len(windows) >= 3 and [count(r, 'window') for r in windows] == list(range(len(windows))),
                'Incomplete convergence windows')
        elapsed, calls, averages = 0.0, 0, []
        for window in windows:
            call_count = count(window, 'calls', 1)
            average = number(window, 'ms_per_call', 0)
            require(average > 0, 'Nonpositive convergence time')
            elapsed += call_count * average
            calls += call_count
            averages.append(average)
            close(number(window, 'accumulated_cuda_ms'), elapsed, 'convergence accumulated time')
            require(count(window, 'epoch') == initial['epoch_last'] + calls, 'Convergence epoch mismatch')
        require(elapsed >= minimum_ms - 1e-5 and count(windows[-1], 'ready') == 1 and
                (max(averages[-3:]) - min(averages[-3:])) / percentile(averages[-3:], .5) <= .05,
                'Warmup has not converged on every rank')
        window_counts.add(tuple(count(r, 'calls') for r in windows))
        final_epochs.add(count(windows[-1], 'epoch'))
        require(calls == count(summary, 'additional_warmup_calls'), 'Warmup call count mismatch')
    require(len(convergence) == world * len(next(iter(window_counts))) and len(window_counts) == 1 and
            len(final_epochs) == 1 and cadence['epoch_first'] == next(iter(final_epochs)) + 1,
            'Convergence rank coverage/cadence mismatch')
    sample_rows = [r for r in rows if r['kind'] == 'sample']
    require({count(r, 'round') for r in sample_rows} == set(range(selected + 1)), 'Unexpected sample rounds')
    rounds, last_epoch = [], cadence['epoch_last']
    for round_id in range(selected + 1):
        records = [r for r in sample_rows if count(r, 'round') == round_id]
        values = audit_samples([r for r in records if r.get('phase') == 'measurement'], world, 50,
                               f'round {round_id}')
        start = [r for r in records if r.get('state') == 'collecting']
        end = [r for r in records if r.get('state') == 'complete']
        require(len(records) == 52 and len(start) == len(end) == 1 and count(start[0], 'count') == 50,
                'Missing/duplicate round start/end')
        require(start[0]['line'] < min(values['lines']) <= max(values['lines']) < end[0]['line'] < summary['line'],
                'Round events/summary are out of order')
        require(values['epoch_first'] == last_epoch + 1, 'Nonconsecutive round epochs')
        last_epoch = values['epoch_last']
        actual_drift = drift(values['maxrank_ms'])
        close(number(end[0], 'half_drift'), actual_drift, 'round drift')
        stable = actual_drift <= .05
        require(count(end[0], 'stable_5pct') == int(stable) and stable == (round_id == selected),
                'Selected round is not the first stable round')
        rounds.append(values | {'round': round_id, 'half_drift': actual_drift, 'accepted': stable})
    chosen = rounds[-1]['maxrank_ms']
    p50, p95, half_drift = percentile(chosen, .5), percentile(chosen, .95), drift(chosen)
    for key, value in (('p50_ms', p50), ('p95_ms', p95), ('half_drift', half_drift)):
        close(number(summary, key), value, key)
    number(summary, 'warmup_wall_s', 0)
    return {'p50_ms': p50, 'p95_ms': p95, 'half_drift': half_drift, 'selected_round': selected,
            'collector': summary['collector'], 'boundary': summary['boundary'],
            'warmup_calls': count(summary, 'warmup') + count(summary, 'additional_warmup_calls') +
                            count(summary, 'sample_cadence_warmup'),
            'minimum_warmup_cuda_ms': minimum_ms, 'rounds': rounds, 'summary_line': summary['line']}


def bf16_round(value):
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    rounded = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack('<f', struct.pack('<I', rounded))[0]


def audit_graph_preparation(records, timing, checks, domains, world):
    """One prepare/one actual launch; capture/update is host work, not an epoch."""
    rows = [row for row in records if row['kind'] == 'graph_prepare']
    require(Counter((count(row, 'generation'), count(row, 'rank')) for row in rows) ==
            Counter((generation, rank) for generation in (0, 1) for rank in range(world)),
            'Missing/duplicate Graph preparation rank/generation')
    for row in rows:
        require(set(row) == GRAPH_PREPARE_FIELDS and row['launch'] == 'graph' and
                row['graph_epoch_mode'] == GRAPH_EPOCH_MODE and count(row, 'gpu_sample_time') == 0 and
                row['includes'] == 'capture_inspect_instantiate_initial_upload_sync_update',
                'Graph preparation schema/untimed contract mismatch')
        number(row, 'wall_s', 0)
        generation, rank = count(row, 'generation'), count(row, 'rank')
        first, last, calls = (count(row, key, 1) for key in ('first_epoch', 'last_epoch', 'calls'))
        require(last < 2**32 and last - first + 1 == calls, 'Graph preparation epoch/count mismatch')
        expected_calls = 1 if generation == 1 or timing is None else (
            1 + timing['warmup_calls'] + sum(len(value['maxrank_ms']) for value in timing['rounds']))
        require(calls == expected_calls, 'Graph preparations differ from actual launch count')
        if generation == 0 and timing is not None:
            require(first == timing['rounds'][0]['epoch_first'] - timing['warmup_calls'] - 1 and
                    last == timing['rounds'][-1]['epoch_last'], 'Graph preparation/sample epochs disagree')
        require(checks['candidate', generation, 'pre', rank]['line'] < row['line'],
                'Graph preparation precedes native candidate binding')
        # Validations are root-owned; rank-concatenated logs are not a global
        # clock. Only rank 0's preparation can be compared with these lines.
        if rank == 0:
            phase = 'post' if generation == 0 and timing is not None else 'pre'
            require(all(checks[kind, generation, phase, peer]['line'] < row['line']
                        for kind in domains for peer in range(world)),
                    'Graph preparation report precedes full final validation')
            final = next(entry for entry in records if entry['kind'] == 'candidate_verified')
            require(row['line'] < final['line'], 'Graph preparation report follows final acceptance')
    by_generation = []
    for generation in (0, 1):
        selected = [row for row in rows if count(row, 'generation') == generation]
        require(len({(row['calls'], row['first_epoch'], row['last_epoch']) for row in selected}) == 1,
                'Graph preparation rank epochs/counts disagree')
        by_generation.append({'generation': generation,
            **{key: count(selected[0], key, 1) for key in ('calls', 'first_epoch', 'last_epoch')},
            'gpu_sample_time': False,
            'rank_host_preparation': [{'rank': count(row, 'rank'), 'wall_s': number(row, 'wall_s', 0),
                                       'line': row['line']} for row in selected]})
    for rank in range(world):
        local = [row for row in rows if count(row, 'rank') == rank]
        require([count(row, 'generation') for row in local] == [0, 1],
                'Graph preparation payload generations are out of rank-local order')
    return by_generation


def audit_graph_epoch_continuity(results):
    # F counters continue across candidates and payload generations, separately
    # for each direction. C/R flags/counters reset for every candidate/payload.
    for direction in DIRECTIONS:
        fused = [row for row in results if row['direction'] == direction and row['component'] == 'fused']
        committed = 0
        for generation in (0, 1):
            for row in fused:
                preparation = row['graph_preparation'][generation]
                require(preparation['first_epoch'] == committed + 1, 'Graph fused epochs are not contiguous')
                committed = preparation['last_epoch']
    for row in results:
        if row['component'] != 'fused':
            require(all(value['first_epoch'] == 1 for value in row['graph_preparation']),
                    'Graph reference epoch baseline was not reset')


def audit_inputs(rows, config, shape):
    generator, world = config['input_generator'], shape['world']
    seed = count(config, 'seed')
    expected = {}
    direction = config.get('fused_direction', 'both')
    def selected(label):
        return direction == 'both' or label.lower().startswith(direction + '-')
    for label, count_value, offset in (
        ('QKV-weight', shape['hidden'] * shape['projection_width'], 11),
        ('OProj-weight', shape['hidden'] * shape['q_width'], 17)):
        if not selected(label):
            continue
        for rank in (range(world) if generator == 'gpu_philox' else (-1,)):
            expected[label, -1, rank] = (count_value, (seed + offset) % 2**32, .02)
    for generation in range(2):
        for rank in range(world):
            for label, width, offset in (('QKV-activation', shape['hidden'], 1000),
                                          ('OProj-activation', shape['q_width'], 2000)):
                if not selected(label):
                    continue
                expected[label, generation, rank] = (shape['seq_local'] * width,
                    (seed + generation * 100003 + rank * 101 + offset) % 2**32, .125)
    actual = {}
    for row in rows:
        if row['kind'] != 'input':
            continue
        if row.get('label') in ('QKV-repeat', 'OProj-repeat'):
            require(count(config, 'cpu_oracle') == 1, 'Unexpected repeat-input diagnostic')
            require(selected(row['label']), 'Repeat diagnostic for unselected direction')
            continue
        key = row.get('label'), int(row.get('generation', -1)), int(row.get('rank', -1))
        require(key in expected and key not in actual, f'Unknown/duplicate input statistics: {key}')
        elements, expected_seed, magnitude = expected[key]
        require(count(row, 'count') == elements and count(row, 'seed') == expected_seed and
                row.get('distribution') == 'uniform' and row.get('generator', 'cpu_mt19937') == generator,
                f'Input count/seed/distribution mismatch: {key}')
        close(number(row, 'lower'), -magnitude, 'input lower bound')
        close(number(row, 'upper'), magnitude, 'input upper bound')
        minimum, maximum = number(row, 'min'), number(row, 'max')
        bound = bf16_round(magnitude)
        require(-bound - 1e-9 <= minimum <= maximum <= bound + 1e-9 and minimum < maximum,
                'Input range exceeds BF16-rounded distribution bounds')
        mean, rms, std = number(row, 'mean'), number(row, 'rms', 0), number(row, 'std', 0)
        nonzero = number(row, 'nonzero_fraction', 0)
        require(minimum <= mean <= maximum and rms > 0 and std > 0 and 0 < nonzero <= 1,
                'Degenerate/nonrandom input statistics')
        close(std * std, max(0.0, rms * rms - mean * mean), 'input variance')
        if generator == 'gpu_philox':
            require(count(row, 'finite') == elements and row.get('algorithm') == 'curand_philox4x32_10' and
                    row.get('mapping') == 'thread_subsequence_v1' and count(row, 'offset') == 0 and
                    count(row, 'blocks', 1) == 256 and count(row, 'threads', 1) == 256,
                    'Incomplete Philox input evidence')
        actual[key] = dict(row)
    require(set(actual) == set(expected), 'Missing rank/generation input statistics')
    oracles = [r for r in rows if r['kind'] == 'input_oracle']
    if generator == 'gpu_philox' and count(config, 'cpu_oracle'):
        expected_checks = {'cross_rank_weights': 1, 'same_seed_repeat': 1}
        if direction != 'qkv':
            expected_checks['oproj_reference_gather'] = 2 * world
        require(Counter(r.get('check') for r in oracles) == expected_checks,
                'Missing Philox CPU-oracle diagnostics')
        gathers = []
        for row in oracles:
            require(row.get('generator') == 'gpu_philox', 'Input oracle generator mismatch')
            if row['check'] == 'oproj_reference_gather':
                require(count(row, 'elements') == shape['seq_local'] * shape['q_width'] and
                        count(row, 'full_cpu_bitwise_match') == 1, 'Input gather oracle failed')
                gathers.append(count(row, 'rank'))
            else:
                require(count(row, 'full_bitwise_match') == 1, 'Input repeat/weight oracle failed')
        require(Counter(gathers) == ({rank: 2 for rank in range(world)} if direction != 'qkv' else {}),
                'Missing gather-oracle rank')
    else:
        require(not oracles, 'Unexpected input oracle diagnostics')
    return list(actual.values())


def audit_telemetry(data, before, job):
    devices = fused_devices(job)
    require(before.get('node') == job['node'] and before.get('selected') == devices,
            'GPU device mapping mismatch')
    observations = before.get('observations', [])
    require(len(observations) >= 1, 'Missing GPU launch observations')
    identities = {}
    for observation in observations:
        measured = observation.get('devices', [])
        require([r.get('index') for r in measured] == devices, 'Incomplete launch GPU identities')
        for row in measured:
            index, uuid = row['index'], row.get('uuid')
            require(isinstance(uuid, str) and uuid.startswith('GPU-'), 'Invalid physical GPU UUID')
            require(identities.setdefault(index, uuid) == uuid, 'Physical GPU identity changed')
    measurements = defaultdict(list)
    reader = csv.DictReader(io.StringIO(data.decode()))
    require(reader.fieldnames is not None, 'Missing telemetry header')
    reader.fieldnames = [field.strip() for field in reader.fieldnames]
    for row in reader:
        row = {key: value.strip() for key, value in row.items()}
        index = row.get('index')
        require(index in devices, 'Unexpected GPU in telemetry')
        timestamp = datetime.strptime(row['timestamp'], '%Y/%m/%d %H:%M:%S.%f')
        numbers = {}
        for key, field in (('sm_mhz', 'clocks.current.sm [MHz]'),
                           ('memory_mhz', 'clocks.current.memory [MHz]'), ('power_w', 'power.draw [W]')):
            numbers[key] = finite(row[field].split()[0], field, 0)
        prior = measurements[index]
        require(not prior or timestamp >= prior[-1][0], 'GPU telemetry time moved backward')
        prior.append((timestamp, numbers))
    require(set(measurements) == set(devices) and all(len(v) >= 2 for v in measurements.values()),
            'Missing continuous GPU telemetry')
    return {'scope': 'whole_subprocess_not_per_candidate', 'devices': [
        {'physical_index': index, 'uuid': identities[index], 'samples': len(measurements[index]),
         **{key: {'min': min(r[1][key] for r in measurements[index]),
                  'max': max(r[1][key] for r in measurements[index])}
            for key in ('sm_mhz', 'memory_mhz', 'power_w')}} for index in devices]}


def audit_validation(row, elements):
    require(row.get('validator') == 'gpu_full' and count(row, 'elements') == elements and
            count(row, 'checked') == elements and count(row, 'nonfinite') == 0,
            'Incomplete full-element validation')
    if row['kind'] == 'correctness':
        require(count(row, 'mismatches') == 0, 'Numerical mismatch')
        number(row, 'max_abs', 0)
        number(row, 'relative_l2', 0)
        close(number(row, 'atol'), .01, 'numeric atol')
        close(number(row, 'rtol'), .01, 'numeric rtol')
    else:
        require(count(row, 'bitwise_mismatches') == 0, 'Route mismatch')


def audit_host_stages(rows, validations, world, host_launch):
    """Verify diagnostic 10+50 records without admitting their events as samples."""
    expected = Counter((direction, sample, rank) for direction in DIRECTIONS
                       for sample in range(50) for rank in range(world))
    require(Counter((r.get('label'), count(r, 'sample'), count(r, 'rank')) for r in rows) == expected,
            'Missing/duplicate host-stage sample/rank')
    duration_fields = tuple(name + '_us' for name in HOST_STAGES) + (
        'api_us', 'api_prefix_us', 'library_return_us', 'api_suffix_us',
        'local_descriptor_us', 'peer_descriptors_us', 'event_us')
    for row in rows:
        require(row.get('profile_schema') == 'host_stages_v1' and row.get('profile_phase') == 'host_stages'
                and row.get('diagnostic_kind') == 'production_kernel_diagnostic'
                and row.get('host_launch') == host_launch and row.get('clock') == 'steady_clock'
                and row.get('api_boundary') == 'after_start_event_before_end_event'
                and row.get('stage_begin') == 'policy_entry'
                and row.get('descriptor_timing') == 'subset_of_communication_prepare'
                and not any(key in row for key in ('candidate', 'generation', 'component')),
                'Invalid host-stage diagnostic contract')
        require(count(row, 'performance_accepted') == count(row, 'clock_overhead_subtracted') ==
                count(row, 'status') == 0 and count(row, 'stage_mask') == 255 and count(row, 'complete') == 1
                and count(row, 'warmup') >= 10 and count(row, 'samples') == 50,
                'Incomplete/accepted host-stage diagnostic')
        for field in duration_fields:
            number(row, field, 0)
        count(row, 'local_descriptor_count')
        count(row, 'peer_descriptor_count')
        total = sum(number(row, name + '_us') for name in HOST_STAGES) + sum(
            number(row, field) for field in ('api_prefix_us', 'library_return_us', 'api_suffix_us'))
        close(number(row, 'api_us'), total, 'Host-stage API duration sum')
        descriptor_time = number(row, 'local_descriptor_us') + number(row, 'peer_descriptors_us')
        require(descriptor_time <= number(row, 'communication_prepare_us') + 2e-7 * max(1, total),
                'Descriptor timing is not a communication-prepare subset')
    groups = []
    for direction in DIRECTIONS:
        selected = [r for r in rows if r['label'] == direction]
        instrumented = [r['line'] for r in validations if r.get('label') == direction
                        and r.get('profile_phase') == 'instrumented']
        production = [r['line'] for r in validations if r.get('label') == direction
                      and r.get('profile_phase') == 'host_stages']
        require(max(instrumented) < min(r['line'] for r in selected) and
                max(r['line'] for r in selected) < min(production),
                'Host-stage diagnostics must follow instrumented validation and precede production validation')
        previous_epoch = None
        for sample in range(50):
            ranks = [r for r in selected if count(r, 'sample') == sample]
            epochs = {count(r, 'epoch', 1) for r in ranks}
            require(len(epochs) == 1, 'Host-stage rank epochs differ')
            epoch = next(iter(epochs))
            require(previous_epoch is None or epoch == previous_epoch + 1, 'Host-stage epochs are not consecutive')
            previous_epoch = epoch
            begins = [number(r, 'api_begin_us', 0) for r in ranks]
            ends = [number(r, 'api_end_us', 0) for r in ranks]
            for row, begin, end in zip(ranks, begins, ends):
                # Subtraction of separately rounded timestamps needs an
                # absolute formatting allowance proportional to their scale.
                tolerance = 2e-8 * max(1, *ends)
                require(end >= begin and math.isclose(end - begin, number(row, 'api_us'),
                        rel_tol=2e-7, abs_tol=tolerance), 'Host-stage API begin/end mismatch')
                for field, value in (('api_begin_skew_us', max(begins) - min(begins)),
                                     ('api_end_skew_us', max(ends) - min(ends))):
                    require(math.isclose(number(row, field, 0), value, rel_tol=2e-7, abs_tol=tolerance),
                            'Host-stage rank skew mismatch')
                require(number(row, 'all_enqueued_us', 0) + tolerance >= max(ends),
                        'Host-stage enqueue completion precedes API return')
        for rank in range(world):
            rank_rows = [r for r in selected if count(r, 'rank') == rank]
            groups.append({'direction': direction, 'rank': rank, 'samples': 50,
                           'p50_us': {field: percentile([number(r, field) for r in rank_rows], .5)
                                      for field in duration_fields}})
    return {'schema': 'host_stages_v1', 'kind': 'production_kernel_diagnostic',
            'performance_accepted': False, 'groups': groups}


def audit_epilogue(rows, validations, host_stages, shape, sm_counts, comm, host_launch, *, legacy_join=False):
    """Bounded single-process diagnostics, never accepted benchmark samples."""
    world = shape['world']
    for row in rows:
        fields = set(EPILOGUE_FIELDS[row['kind']]) | {'kind', 'line'}
        if row['kind'] == 'epilogue_resources' and not legacy_join:
            fields.add('role_timestamp_join')
        require(set(row) == fields,
                'Missing/unknown epilogue diagnostic fields')
        require(count(row, 'rank') < world and count(row, 'performance_accepted') == 0,
                'Invalid/accepted epilogue diagnostic rank')
        if row['kind'] != 'epilogue_sample':
            require(row.get('schema') == 'qkv_epilogue_cta_v1' and row.get('clock') == 'globaltimer' and
                    row.get('clock_unit') == 'ns', 'Epilogue clock/schema mismatch')
    resources = [row for row in rows if row['kind'] == 'epilogue_resources']
    samples = [row for row in rows if row['kind'] == 'epilogue_sample']
    ctas = [row for row in rows if row['kind'] == 'epilogue_cta']
    require(Counter((count(row, 'rank'), row.get('diagnostic_kind')) for row in resources) ==
            Counter((rank, mode) for rank in range(world) for mode in EPILOGUE_MODES),
            'Missing/duplicate epilogue resource rank/mode')
    for row in resources:
        if not legacy_join:
            expected_join = {'production': 'none', 'role_telemetry': 'legacy_bar_sync',
                             'epilogue_telemetry': 'cta_popc256_dependency'}[row['diagnostic_kind']]
            require(row.get('role_timestamp_join') == expected_join, 'Epilogue role timestamp join mismatch')
        require(row['store_interval'] == 'issuing_lane_base_call_including_accumulator_wait' and
                row['drain_interval'] == 'issuing_warp_global_wait_and_warp_join' and
                count(row, 'record_bytes') == 96 and count(row, 'regs', 1) > 0 and
                count(row, 'max_threads', 256) >= 256 and count(row, 'cluster_ctas') == 1 and
                (count(row, 'tile_m'), count(row, 'tile_n'), count(row, 'tile_k')) == (128, 256, 64),
                'Epilogue resource geometry/semantics mismatch')
        count(row, 'local_bytes')  # Nonzero spills are reported, not hidden or rejected.
        count(row, 'static_smem')
        count(row, 'dynamic_smem', 1)
    for rank in range(world):
        rank_resources = [row for row in resources if count(row, 'rank') == rank]
        require(len({row['dynamic_smem'] for row in rank_resources}) == 1,
                'Epilogue modes changed dynamic shared-memory reservation')
    order = [(sample, mode, rank) for sample in range(50)
             for mode in (EPILOGUE_MODES if sample % 2 == 0 else EPILOGUE_MODES[::-1])
             for rank in range(world)]
    require([(count(row, 'sample'), row['diagnostic_kind'], count(row, 'rank')) for row in samples] == order,
            'Missing/duplicate/out-of-order epilogue sample rank/mode')
    qkv_host = [row for row in host_stages if row['label'] == DIRECTIONS[0]]
    require(qkv_host, 'Epilogue diagnostics require preceding QKV host-stage evidence')
    initial_epoch = max(count(row, 'epoch') for row in qkv_host) + 31  # 10 warmups for each of 3 modes.
    require(initial_epoch + 150 <= 2**32 - 1, 'Epilogue epoch exceeds uint32')
    for position, row in enumerate(samples):
        require(count(row, 'epoch', 1) == initial_epoch + position // world and
                count(row, 'warmup') == 10 and count(row, 'samples') == 50 and
                count(row, 'final_sample_poisoned') == int(count(row, 'sample') == 49) and
                row['host_launch'] == host_launch and row['launch'] == 'eager' and
                row['process_layout'] == 'single_process' and number(row, 'event_ms', 0) > 0,
                'Epilogue epoch/sampling/execution contract mismatch')
    phase_rows = {phase: [row for row in validations if row.get('label') == DIRECTIONS[0] and
                         row.get('profile_phase') == phase] for phase in EPILOGUE_PHASES}
    prior = [row['line'] for row in validations if row.get('label') == DIRECTIONS[0] and
             row.get('profile_phase') == 'host_stages']
    require(prior and max(prior) < min(row['line'] for row in resources),
            'Epilogue resources must follow QKV host-stage validation')
    # Samples are buffered and printed only after all four final validations.
    # Reconstruct their execution from sample/mode/epoch, not print timestamps.
    previous = max(row['line'] for row in resources)
    for phase in tuple('epilogue_' + mode for mode in EPILOGUE_MODES[::-1]) + ('epilogue_record',):
        phase_lines = [row['line'] for row in phase_rows[phase]]
        require(phase_lines and previous < min(phase_lines), 'Epilogue final validation phase order mismatch')
        previous = max(phase_lines)
    require(previous < min(row['line'] for row in samples), 'Epilogue buffered samples precede final validation')
    m_tiles = (shape['seq_local'] + 127) // 128
    n_tiles = (shape['projection_width'] + 255) // 256
    total_tiles = m_tiles * n_tiles
    compute = [min(total_tiles, sm - comm) for sm in sm_counts]
    require(Counter((count(row, 'rank'), count(row, 'cta')) for row in ctas) ==
            Counter((rank, cta) for rank in range(world) for cta in range(comm, comm + compute[rank])),
            'Missing/duplicate/out-of-grid epilogue CTA record')
    require(max(row['line'] for row in samples) < min(row['line'] for row in ctas),
            'Epilogue CTA records precede buffered samples')
    time_fields = ('cta_start', 'first_store_begin', 'first_store_end', 'first_drain_end',
                   'first_ready_after', 'last_ready_after', 'cta_role_done', 'cta_end')
    for row in ctas:
        rank, cta = count(row, 'rank'), count(row, 'cta')
        expected_tiles = (total_tiles - (cta - comm) + compute[rank] - 1) // compute[rank]
        require(count(row, 'tile_count', 1) == expected_tiles and count(row, 'epoch') == initial_epoch + 150 and
                count(row, 'first_m_tile') < m_tiles and count(row, 'first_n_tile') < n_tiles and
                count(row, 'first_batch') == 0, 'Epilogue tile count/epoch/coordinate mismatch')
        times = [count(row, field, 1) for field in time_fields]
        require(times == sorted(times) and times[-1] < 2**64, 'Unordered/out-of-range epilogue CTA clock')
        store_sum, drain_sum = count(row, 'store_ns_sum'), count(row, 'drain_ns_sum')
        store_max, drain_max = count(row, 'store_ns_max'), count(row, 'drain_ns_max')
        span = count(row, 'cta_role_done') - count(row, 'cta_start')
        require(span > 0 and store_sum <= span and drain_sum <= span - store_sum and
                count(row, 'first_store_end') - count(row, 'first_store_begin') <= store_max <= store_sum and
                count(row, 'first_drain_end') - count(row, 'first_store_end') <= drain_max <= drain_sum and
                (store_sum + expected_tiles - 1) // expected_tiles <= store_max and
                (drain_sum + expected_tiles - 1) // expected_tiles <= drain_max,
                'Epilogue duration sums/maxima disagree with per-CTA window')

    def stats(values):
        return {'p50': percentile(values, .5), 'p95': percentile(values, .95), 'max': max(values)}

    mode_results = []
    for mode in EPILOGUE_MODES:
        selected = [row for row in samples if row['diagnostic_kind'] == mode]
        maximum = [max(number(row, 'event_ms') for row in selected if count(row, 'sample') == sample)
                   for sample in range(50)]
        mode_results.append({'mode': mode, 'samples': 50, 'maxrank_event_ms': stats(maximum),
            'per_rank_event_ms': [{'rank': rank, **stats([number(row, 'event_ms') for row in selected
                                                       if count(row, 'rank') == rank])} for rank in range(world)],
            'sample_lines': [row['line'] for row in selected]})
    rank_results = []
    for rank in range(world):
        selected = [row for row in ctas if count(row, 'rank') == rank]
        require(sum(count(row, 'tile_count') for row in selected) == total_tiles, 'Incomplete epilogue tile coverage')
        first_start = min(count(row, 'cta_start') for row in selected)
        rank_results.append({'rank': rank, 'compute_ctas': compute[rank], 'tiles': total_tiles,
            'record_epoch': initial_epoch + 150,
            'first_ready_from_first_compute_cta_ns': min(count(row, 'first_ready_after') for row in selected) - first_start,
            'compute_role_span_ns': max(count(row, 'cta_role_done') for row in selected) - first_start,
            'per_cta_ns': {field: stats([count(row, field) for row in selected]) for field in
                           ('store_ns_sum', 'store_ns_max', 'drain_ns_sum', 'drain_ns_max')},
            'drain_fraction_of_own_cta_role': stats([count(row, 'drain_ns_sum') /
                (count(row, 'cta_role_done') - count(row, 'cta_start')) for row in selected]),
            'record_lines': [row['line'] for row in selected]})
    return {'schema': 'qkv_epilogue_cta_v1', 'performance_accepted': False,
            'role_timestamp_join_schema': 'legacy_unlogged' if legacy_join else 'explicit_v1',
            'scope': 'single_process_eager_diagnostic_not_mpi_performance',
            'clock': 'globaltimer', 'clock_unit': 'ns', 'durations_summed_across_ctas': False,
            'base_store_includes_accumulator_wait': True, 'resources': resources,
            'modes': mode_results, 'ranks': rank_results}


def audit_schedule_config(job, config, rows):
    """Old fixed scheduling is explicit provenance, never a missing-field fallback."""
    # Legacy component_resources already had raster and scheduled_compute_ctas;
    # neither field alone identifies the new record contract on that row kind.
    explicit = (any(key in job for key in SCHEDULE_REQUEST_FIELDS) or
                any(key in config for key in SCHEDULE_CONFIG_FIELDS) or
                job.get('files', {}).get('benchmarks/sm103/fused_bf16.cu') in SCHEDULE_HARNESS_SHA256 or
                any(any(key in row for key in (SCHEDULE_ROW_FIELDS if row['kind'] == 'candidate'
                                              else SCHEDULE_ROW_FIELDS[1:5]))
                    for row in rows if row['kind'] in ('candidate', 'component_resources')))
    maximum = job.get('max_swizzle_size', 1)
    require(type(maximum) is int and maximum in (1, 2, 4, 8), 'Invalid job maximum swizzle size')
    requested = {direction: job.get(key, 'heuristic')
                 for direction, key in zip(DIRECTIONS, ('qkv_raster', 'oproj_raster'))}
    require(all(value in ('heuristic', 'along_m', 'along_n') for value in requested.values()),
            'Invalid job raster request')
    effective = {direction: ('along_m' if direction == DIRECTIONS[0] else 'along_n')
                 if value == 'heuristic' else value for direction, value in requested.items()}
    if explicit:
        require(all(key in config for key in SCHEDULE_CONFIG_FIELDS), 'Missing explicit scheduling config')
        require(count(config, 'max_swizzle_size', 1) == maximum and
                all(config[key] == requested[direction] and config[key.replace('_raster', '_effective_raster')] ==
                    effective[direction] for direction, key in zip(DIRECTIONS, ('qkv_raster', 'oproj_raster'))),
                'Job/config scheduling mismatch')
    return {'schema': 'explicit_v1' if explicit else 'legacy_fixed_v1', 'max_swizzle_size': maximum,
            'requested_rasters': requested, 'effective_rasters': effective}


def resolved_schedule(scheduling, direction, m, n, tile_n):
    # Independent audit of pinned CUTLASS 57e3cfb's cluster-1 static scheduler:
    # tile_scheduler_params.h:get_log_swizzle_size/initialize. The requested
    # maximum is not necessarily the effective swizzle on a small tile grid.
    m_tiles, n_tiles = (m + 127) // 128, (n + tile_n - 1) // tile_n
    minimum, maximum = min(m_tiles, n_tiles), scheduling['max_swizzle_size']
    effective = (8 if maximum >= 8 and minimum >= 6 else
                 4 if maximum >= 4 and minimum >= 3 else
                 2 if maximum >= 2 and minimum >= 2 else 1)
    padded_m = ((m_tiles + effective - 1) // effective) * effective
    padded_n = ((n_tiles + effective - 1) // effective) * effective
    require(padded_m * padded_n <= 2**31 - 1, 'Scheduled tile grid exceeds int32')
    return {'schedule_schema': scheduling['schema'], 'raster_requested': scheduling['requested_rasters'][direction],
            'raster': scheduling['effective_rasters'][direction], 'max_swizzle_size': maximum,
            'effective_swizzle_size': effective, 'swizzle': effective,
            'padded_m_tiles': padded_m, 'padded_n_tiles': padded_n,
            'has_padding': (padded_m != m_tiles or padded_n != n_tiles),
            'scheduled_work_tiles_derived': padded_m * padded_n}


def audit_schedule_row(row, schedule, compute_ctas):
    if schedule['schedule_schema'] != 'explicit_v1':
        return  # Legacy component raster/CTA counts have their existing checks.
    require(all(key in row for key in SCHEDULE_ROW_FIELDS), 'Missing explicit scheduling resource fields')
    require(row['raster'] == schedule['raster'] and
            all(count(row, field, 1) == schedule[field] for field in SCHEDULE_ROW_FIELDS[1:5]) and
            count(row, 'scheduled_compute_ctas') == compute_ctas,
            'Scheduling resources disagree with request/geometry/budget')


def audit_auto_comm(rows, job, config, expected, devices):
    """Resolve automatic budgets from rank-owned evidence, not a default CTA count.

    Selection precedes the first launch and both payloads. Existing candidate,
    resource, reference and numerical checks then use this actual positive
    budget; zero denotes only the API request, never zero communication work.
    """
    enabled = job.get('auto_oproj_comm', False)
    require(type(enabled) is bool and
            integer(config.get('auto_oproj_comm', 0), 'auto_oproj_comm') == int(enabled),
            'Automatic OProj communication job/config mismatch')
    records = [r for r in rows if r['kind'] == 'auto_comm']
    if not enabled:
        require(not records, 'Unexpected automatic communication records')
        return expected, {}
    require(job.get('fused_direction') == 'oproj' and fused_launch(job) == 'graph' and
            job.get('causal') and job.get('oproj_comm_layout', 'rows') == 'rows' and
            not job.get('profile') and not job.get('compute_only') and
            count(config, 'comm_sm') == 0 and all(count(d, 'sms') == 148 for d in devices),
            'Unsupported automatic communication measurement scope')
    required_fields = {'kind', 'line', 'label', 'candidate', 'comm_sm', 'tile', 'rank',
                       'query_us', 'repeat_query_us', 'launch_comm'}
    groups = defaultdict(dict)
    for row in records:
        require(set(row) == required_fields, 'Automatic communication record fields mismatch')
        index, rank, comm = count(row, 'candidate', 1), count(row, 'rank'), count(row, 'comm_sm', 1)
        require(index <= len(expected) and rank < len(devices) and rank not in groups[index],
                'Duplicate/unknown automatic communication candidate/rank')
        direction, requested, tile = expected[index - 1]
        require(direction == row['label'] == 'A2A_GEMM' and requested == 0 and row['tile'] == tile and
                comm in (8, 16, 24, 32, 48) and count(row, 'launch_comm') == 0,
                'Automatic communication request/resolved configuration mismatch')
        number(row, 'query_us', 0)
        number(row, 'repeat_query_us', 0)
        bindings = [r for r in rows if r['kind'] == 'candidate' and
                    count(r, 'candidate', 1) == index and count(r, 'rank') == rank]
        require(bindings and row['line'] < min(r['line'] for r in bindings),
                'Automatic communication selection must precede native binding')
        groups[index][rank] = row
    require(set(groups) == set(range(1, len(expected) + 1)), 'Missing automatic communication candidate')
    resolved, metadata = [], {}
    for index, (direction, _, tile) in enumerate(expected, 1):
        ranks = groups[index]
        require(set(ranks) == set(range(len(devices))), 'Missing automatic communication rank')
        budgets = {count(row, 'comm_sm') for row in ranks.values()}
        require(len(budgets) == 1, 'Automatic communication ranks disagree on budget')
        comm = budgets.pop()
        resolved.append((direction, comm, tile))
        metadata[index] = dict(mode='runtime_model', requested_comm_ctas=0, resolved_comm_ctas=comm,
            launch_comm_ctas=0, rank_queries=[ranks[rank] for rank in range(len(devices))])
    return resolved, metadata


def audit_log(text, job):
    rows, profile_records = parse_log(text, completion='any' if job.get('mpi') else 'last')
    configs = [r for r in rows if r['kind'] == 'config']
    require(len(configs) == 1, 'Requires exactly one harness config')
    config = configs[0]
    rank_swizzle = 'rank_n_band_v1' if job.get('qkv_rank_swizzle') else 'off'
    require(config.get('qkv_rank_swizzle', 'off') == rank_swizzle,
            'QKV rank swizzle job/config mismatch')
    oproj_comm_layout = job.get('oproj_comm_layout', 'rows')
    require(oproj_comm_layout in ('rows', 'columns'), 'Unknown OProj communication layout')
    explicit_comm_layout = 'oproj_comm_layout' in job or any('oproj_comm_layout' in row for row in rows)
    require(config.get('oproj_comm_layout', None if explicit_comm_layout else 'rows') == oproj_comm_layout,
            'OProj communication layout job/config mismatch')
    launch = fused_launch(job)
    graph = launch == 'graph'
    require(not graph or (config.get('launch') == 'graph' and
            config.get('graph_epoch_mode') == GRAPH_EPOCH_MODE and
            config.get('collector') == 'mpi_graph_rank_events_v1'), 'Graph config contract mismatch')
    for row in rows:
        if 'launch' in row:
            require(row['launch'] == launch, 'Per-record launch mismatch')
        if 'graph_epoch_mode' in row:
            require(graph and row['graph_epoch_mode'] == GRAPH_EPOCH_MODE, 'Unexpected Graph epoch mode')
    require(graph or (not any(row['kind'] == 'graph_prepare' for row in rows) and
                      config.get('collector') != 'mpi_graph_rank_events_v1'), 'Unexpected Graph preparation')
    scheduling = audit_schedule_config(job, config, rows)
    epilogue_probe = job.get('qkv_epilogue_probe', False)
    require(type(epilogue_probe) is bool, 'QKV epilogue job option must be boolean')
    require(('qkv_epilogue_probe' in config or 'qkv_epilogue_probe' not in job) and
            integer(config.get('qkv_epilogue_probe', 0), 'qkv_epilogue_probe') == int(epilogue_probe),
            'QKV epilogue job/config mismatch')
    require(epilogue_probe or not any(row['kind'] in EPILOGUE_FIELDS for row in rows),
            'Unexpected QKV epilogue diagnostics without explicit option')
    profile_schema = config.get('profile_schema', 'legacy_single_validation' if job.get('profile') else 'none')
    require(profile_schema in (('legacy_single_validation', 'host_stages_v1') if job.get('profile') else ('none',)),
            'Unsupported profile schema')
    profile_detail = config.get('profile_detail', 'full' if job.get('profile') else 'none')
    require(profile_detail == ((job.get('profile_detail') or 'full') if job.get('profile') else 'none') and
            profile_detail in ('full', 'cta', 'none'), 'Profile detail config mismatch')
    if profile_detail == 'cta':
        require(profile_schema == 'host_stages_v1' and profile_records.get('profile_peer', 0) == 0,
                'CTA-only diagnostics must not contain peer traces')
    elif profile_detail == 'full' and 'profile_detail' in config:
        require(profile_records.get('profile_peer', 0) > 0, 'Full diagnostics require peer traces')
    oproj_probe = job.get('oproj_pipeline_probe', False)
    require(type(oproj_probe) is bool, 'OProj pipeline probe job option must be boolean')
    probe_records = {kind for kind in OPROJ_PROBE_FIELDS if profile_records.get(kind, 0)}
    require(not probe_records or oproj_probe, 'Unexpected OProj pipeline probe records')
    require(not oproj_probe or (job.get('profile') and profile_detail == 'full' and
            job.get('directions') == 'oproj' and not job.get('mpi') and not epilogue_probe and
            probe_records == set(OPROJ_PROBE_FIELDS)), 'Unsupported/incomplete OProj pipeline probe')
    shape = fused_geometry(job)
    require(config.get('fused_direction', 'both') == job.get('fused_direction', 'both'),
            'Fused direction mismatch')
    for key in ('world', 'global_seq', 'seq_local', 'hidden', 'q_heads', 'kv_heads', 'head_dim', 'timeout_seconds'):
        require(count(config, key, 1) == shape[key], f'Job/config mismatch: {key}')
    diagnostic = bool(job.get('validation_self_test'))
    calibrate = bool(job.get('calibrate'))
    require(integer(config.get('calibrate', 0), 'calibrate') == int(calibrate) and
            not (calibrate and (diagnostic or job.get('profile'))), 'Calibration config mismatch')
    compute_only = bool(job.get('compute_only'))
    require(not compute_only or (calibrate and job.get('fused_direction') == 'oproj'),
            'Compute-only requires explicit OProj calibration')
    components = ('compute_reference',) if compute_only else COMPONENTS if calibrate else ('fused',)
    for key, expected in (('causal', bool(job.get('causal'))), ('profile', bool(job.get('profile'))),
                          ('validation_self_test', diagnostic),
                          ('cpu_oracle', diagnostic or bool(job.get('cpu_oracle')))):
        require(count(config, key) == int(expected), f'Job/config mismatch: {key}')
    require(config.get('input_generator', 'cpu_mt19937') == job.get('input_generator', 'cpu_mt19937'),
            'Input generator mismatch')
    config.setdefault('input_generator', 'cpu_mt19937')
    host_launch = config.get('host_launch', 'sequential')
    if job.get('mpi'):
        require(host_launch == 'mpi_process' and config.get('process_layout') == 'mpi_one_process_per_gpu' and
                config.get('launch') == launch and job.get('host_launch', 'sequential') == 'sequential',
                'MPI process/launch contract mismatch')
    else:
        require(host_launch in ('sequential', 'per_gpu_thread') and host_launch == job.get('host_launch', 'sequential'),
                'Host launch contract mismatch')
    quick = bool(job.get('quick'))
    require(config.get('sampling_mode', 'formal_10_50') == ('quick_1_5' if quick else 'formal_10_50'),
            'Sampling mode differs from job')
    require(count(config, 'seed') == job.get('seed', 20260906) and
            (count(config, 'warmup') == 1 if quick else count(config, 'warmup') >= 10) and
            count(config, 'samples') == (5 if quick else 50), 'Seed/warmup/sample config mismatch')
    comm, qkv, oproj = fused_candidates(job)
    require(not epilogue_probe or (job.get('profile') and profile_detail == 'cta' and
            profile_schema == 'host_stages_v1' and not job.get('mpi') and not diagnostic and not calibrate and
            scheduling['schema'] == 'explicit_v1' and qkv == ['m128n256k64e32'] and len(comm) == len(oproj) == 1),
            'Unsupported QKV epilogue diagnostic scope')
    expected = [(direction, c, tile) for direction, policies in zip(DIRECTIONS, (qkv, oproj))
                if job.get('fused_direction', 'both') in ('both', 'qkv' if direction == DIRECTIONS[0] else 'oproj')
                for tile in policies for c in comm]
    require(count(config, 'candidates') == len(expected), 'Candidate plan count mismatch')
    require(count(config, 'comm_sm') == comm[0], 'Resolved default communication CTA mismatch')
    inputs = audit_inputs(rows, config, shape)
    devices = [r for r in rows if r['kind'] == 'device']
    require([count(r, 'rank') for r in devices] == list(range(shape['world'])), 'Missing/duplicate device rank')
    require(all(r.get('runtime_cc') == '10.3' and count(r, 'sms', 1) > max(comm) for r in devices),
            'Invalid runtime architecture or communication budget')
    sm_counts = [count(r, 'sms') for r in devices]
    expected, auto_comm = audit_auto_comm(rows, job, config, expected, devices)
    grouped = defaultdict(list)
    profile_validation = []
    for row in rows:
        if row['kind'] not in {'candidate', 'component_resources', 'correctness', 'route', 'warmup', 'sample', 'summary',
                               'candidate_verified', 'validation_self_test', 'validation_oracle', 'graph_prepare'}:
            continue
        if 'host_launch' in row:
            require(row['host_launch'] == host_launch, 'Per-record host launch mismatch')
        component = row.get('component', 'fused')
        require(component in components or (compute_only and row['kind'] == 'candidate' and component == 'fused'),
                'Component not enabled by job/config')
        if 'candidate' not in row:
            require(bool(job.get('profile')) and row['kind'] in ('correctness', 'route', 'validation_oracle'),
                    'Unscoped measurement/validation record')
            if row['kind'] == 'validation_oracle':
                require(count(config, 'cpu_oracle') and count(row, 'full_cpu_match') == 1,
                        'Invalid profile CPU oracle')
                profile_validation.append(row)
                continue
            direction, rank = row.get('label'), count(row, 'rank')
            require(direction in DIRECTIONS and rank < shape['world'], 'Unknown profile validation rank/direction')
            width = shape['projection_width'] if direction == 'GEMM_A2A' else (
                shape['hidden'] if row['kind'] == 'correctness' else shape['q_width'])
            audit_validation(row, shape['seq_local'] * width)
            profile_validation.append(row)
            continue
        index = count(row, 'candidate', 1)
        require('profile_phase' not in row, 'Profile validation cannot be a production candidate record')
        require(index <= len(expected), 'Unknown candidate')
        direction, c, tile = expected[index - 1]
        require((row.get('label'), count(row, 'comm_sm'), row.get('tile')) == (direction, c, tile),
                'Candidate config/direction mismatch')
        if direction == 'A2A_GEMM' and row['kind'] in ('candidate', 'component_resources'):
            require(row.get('oproj_comm_layout', None if explicit_comm_layout else 'rows') == oproj_comm_layout,
                    'OProj communication layout resource mismatch')
        if row['kind'] == 'candidate':
            rank = count(row, 'rank')
            require(rank < shape['world'], 'Unexpected scheduled candidate rank')
            _, tile_n, _ = fused_policy_tile(tile)
            width = shape['projection_width'] if direction == DIRECTIONS[0] else shape['hidden']
            schedule = resolved_schedule(scheduling, direction, shape['seq_local'], width, tile_n)
            require(not job.get('profile') or not schedule['has_padding'],
                    'Profiling does not support swizzle-padded tile grids')
            audit_schedule_row(row, schedule, min(schedule['scheduled_work_tiles_derived'], sm_counts[rank] - c))
        if row['kind'] == 'candidate' and 'generation' not in row:
            require(bool(job.get('profile')), 'Unexpected unscoped candidate query')
            continue
        require(row['kind'] != 'candidate' or component == 'fused', 'Resolved candidate resources must be shared production records')
        grouped[index, component].append(row)
    require(set(grouped) == {(index, component) for index in range(1, len(expected) + 1)
                            for component in (('fused',) + components if compute_only else components)},
            'Missing candidate/component group')
    require(not profile_records or bool(job.get('profile')), 'Unexpected profiling diagnostics')
    results = []
    for index, component in ((i, comp) for i in range(1, len(expected) + 1) for comp in components):
        direction, c, tile = expected[index - 1]
        tile_m, tile_n, tile_k = fused_policy_tile(tile)
        records = grouped[index, component]
        if component != 'fused':
            records = records + [r for r in grouped[index, 'fused'] if r['kind'] == 'candidate']
        qkv_direction = direction == DIRECTIONS[0]
        m, n, k = shape['seq_local'], shape['projection_width'] if qkv_direction else shape['hidden'], \
                  shape['hidden'] if qkv_direction else shape['q_width']
        final = [r for r in records if r['kind'] == 'candidate_verified']
        require(len(final) == 1 and count(final[0], 'payload_generations') == 2 and
                count(final[0], 'full_numeric') == int(component != 'copy_reference') and
                count(final[0], 'full_route') == int(component != 'compute_reference') and
                count(final[0], 'performance_accepted') == int(not diagnostic),
                'Candidate incomplete or performance acceptance mismatch')
        require(not graph or (final[0].get('launch') == 'graph' and
                final[0].get('graph_epoch_mode') == GRAPH_EPOCH_MODE), 'Graph final acceptance contract mismatch')
        checks = {}
        for row in records:
            if row['kind'] not in ('candidate', 'correctness', 'route'):
                continue
            generation, rank = count(row, 'generation'), count(row, 'rank')
            require(generation in (0, 1) and rank < shape['world'], 'Unexpected validation rank/generation')
            phase = row.get('validation_phase', 'pre')
            require(phase in ('pre', 'post'), 'Unknown validation phase')
            key = row['kind'], generation, phase, rank
            require(key not in checks, 'Duplicate per-rank candidate validation')
            checks[key] = row
            if row['kind'] == 'candidate':
                require(row.get('state') == 'resolved' and count(row, 'tile_m') == tile_m and
                        count(row, 'tile_n') == tile_n and
                        count(row, 'tile_k') == tile_k and count(row, 'threads') == 256 and
                        count(row, 'dynamic_smem', 1) > 0, 'Resolved tile/resources mismatch')
            else:
                elements = m * (n if row['kind'] == 'correctness' or qkv_direction else k)
                audit_validation(row, elements)
        domains = ('correctness', 'route') if component == 'fused' else (
            ('correctness',) if component == 'compute_reference' else ('route',))
        required = {('candidate', gen, 'pre', rank) for gen in (0, 1) for rank in range(shape['world'])}
        required.update((kind, gen, phase, rank) for kind in domains for gen in (0, 1)
                        for phase in (('pre', 'post') if (component != 'fused' or graph) and gen == 0
                                      and not diagnostic else ('pre',))
                        for rank in range(shape['world']))
        require(set(checks) == required, 'Missing rank/payload numerical or route validation')
        cpu_oracles = [r for r in records if r['kind'] == 'validation_oracle']
        if count(config, 'cpu_oracle'):
            oracle_keys = [(count(r, 'generation'), r.get('validation_phase', 'pre')) for r in cpu_oracles]
            expected_oracles = {(gen, phase) for kind, gen, phase, rank in required if kind != 'candidate'}
            require(len(oracle_keys) == len(expected_oracles) and set(oracle_keys) == expected_oracles and
                    all(count(r, 'full_cpu_match') == 1 for r in cpu_oracles),
                    'Missing/failed component CPU oracle confirmation')
        else:
            require(not cpu_oracles, 'Unexpected component CPU oracle confirmation')
        # Numerical/route records are gathered and printed by root. Native
        # resolved-candidate records come from their owning rank; concatenating
        # rank streams must not invent a cross-process chronological comparison.
        ordered_checks = [r for r in checks.values() if not job.get('mpi') or
                          r['kind'] != 'candidate' or count(r, 'rank') == 0]
        require(final[0]['line'] > max(r['line'] for r in ordered_checks), 'Premature candidate acceptance')
        timings = [r for r in records if r['kind'] in ('warmup', 'sample', 'summary')]
        require(all(count(r, 'generation') == 0 for r in timings), 'Unexpected timed payload generation')
        require(not diagnostic or not timings, 'Diagnostic-only run contains performance records')
        timing = None if diagnostic else audit_timing(timings, shape['world'], mpi=bool(job.get('mpi')), launch=launch, quick=quick)
        if timing:
            first_warmup_line = min(r['line'] for r in timings if r['kind'] == 'warmup')
            require(all(checks[kind, 0, 'pre', rank]['line'] < first_warmup_line
                        for kind in domains for rank in range(shape['world'])),
                    'Pre-validation does not precede measurement')
            require(all(row['line'] > timing['summary_line'] for row in ordered_checks
                        if count(row, 'generation') == 1) and
                    final[0]['line'] > timing['summary_line'],
                    'Second payload/final acceptance does not follow measurement')
        if (component != 'fused' or graph) and timing:
            require(all(checks[kind, 0, 'post', rank]['line'] > timing['summary_line']
                        for kind in domains for rank in range(shape['world'])),
                    'Reference/Graph post-validation does not follow measurement')
        graph_preparation = audit_graph_preparation(records, timing, checks, domains, shape['world']) if graph else None
        resolved = [checks['candidate', 0, 'pre', rank] for rank in range(shape['world'])]
        for rank, first in enumerate(resolved):
            second = checks['candidate', 1, 'pre', rank]
            require(all(first[field] == second[field] for field in ('tile_m', 'tile_n', 'tile_k', 'threads', 'dynamic_smem')),
                    'Resources changed between payload generations')
        schedule = resolved_schedule(scheduling, direction, m, n, tile_n)
        work_tiles = ((m + tile_m - 1) // tile_m) * ((n + tile_n - 1) // tile_n)
        scheduled_work_tiles = schedule['scheduled_work_tiles_derived']
        compute_ctas = [min(scheduled_work_tiles, sm - c) for sm in sm_counts]
        component_resources = [r for r in records if r['kind'] == 'component_resources']
        require(bool(component_resources) == calibrate, 'Component resource contract mismatch')
        resource_keys = set()
        for row in component_resources:
            generation, rank = count(row, 'generation'), count(row, 'rank')
            require(generation in (0, 1) and rank < shape['world'] and (generation, rank) not in resource_keys,
                    'Duplicate/invalid component resource record')
            resource_keys.add((generation, rank))
            production = resolved[rank]
            audit_schedule_row(row, schedule, 0 if component == 'copy_reference' else compute_ctas[rank])
            require(all(row[field] == production[field] for field in ('tile_m', 'tile_n', 'tile_k')) and
                    row.get('raster') == schedule['raster'] and
                    count(row, 'compute_budget') == sm_counts[rank] - c and
                    count(row, 'scheduled_compute_ctas') == (0 if component == 'copy_reference' else compute_ctas[rank]) and
                    count(row, 'scheduled_comm_ctas') == (0 if component == 'compute_reference' else c) and
                    count(row, 'production_threads') == count(production, 'threads') and
                    count(row, 'production_dynamic_smem') == count(production, 'dynamic_smem') and
                    row.get('reference_resources') == ('not_applicable' if component == 'fused' else 'unknown'),
                    'Component resources disagree with production geometry/budget')
        require(not calibrate or resource_keys == {(gen, rank) for gen in (0, 1) for rank in range(shape['world'])},
                'Missing component resource rank/generation')
        payload = 2 * m * (n if qkv_direction else k)
        results.append({'candidate': index, 'direction': direction, 'component': component,
            'communication_selection': auto_comm.get(index, {'mode': 'explicit'}),
            'sampling_mode': 'quick_1_5' if quick else 'formal_10_50',
            'formal_eligible': not quick and not diagnostic,
            'measurement_role': 'production' if component == 'fused' else 'calibration',
            'qkv_rank_swizzle': rank_swizzle if qkv_direction else 'off',
            'oproj_comm_layout': 'not_applicable' if qkv_direction else oproj_comm_layout,
            'performance_accepted': not diagnostic, 'host_launch': host_launch,
            'collector': None if diagnostic else timing['collector'],
            'launch': launch, 'graph_epoch_mode': GRAPH_EPOCH_MODE if graph else None,
            'graph_preparation': graph_preparation, 'precision': 'bf16_accfp32_bf16',
            'layout': 'qkv_source_rank_major_v1' if qkv_direction else
                      ('causal_dual_chunk_v1' if job.get('causal') else 'sequence_rank_major_v1'),
            'world': shape['world'], 'global_seq': shape['global_seq'], 'seq_local': m,
            **{key: shape[key] for key in ('hidden', 'q_heads', 'kv_heads', 'head_dim')},
            'm': m, 'n': n, 'k': k, 'comm_ctas': c, 'tile_policy': tile,
            'tile_m': tile_m, 'tile_n': tile_n, 'tile_k': tile_k, 'cluster_ctas': 1,
            **schedule,
            'compute_ctas_derived': [0] * shape['world'] if component == 'copy_reference' else compute_ctas,
            'production_compute_ctas_derived': compute_ctas,
            'production_waves_derived': [math.ceil(scheduled_work_tiles / value) for value in compute_ctas],
            'sm_counts': sm_counts, 'sm_count': sm_counts[0] if len(set(sm_counts)) == 1 else None,
            'problem_gemm_flops': 2 * m * n * k, 'work_tiles_derived': work_tiles,
            'problem_route_payload_bytes': payload,
            'problem_remote_payload_bytes': payload * (shape['world'] - 1) // shape['world'],
            'executed_gemm_flops': 0 if component == 'copy_reference' else 2 * m * n * k,
            'executed_route_payload_bytes': 0 if component == 'compute_reference' else payload,
            'production_resources': resolved, 'component_resources': component_resources, 'timing': timing,
            'validation_lines': [r['line'] for r in checks.values()], 'acceptance_line': final[0]['line']})
    if graph:
        audit_graph_epoch_continuity(results)
    if diagnostic:
        faults = [r for r in rows if r['kind'] == 'validation_self_test']
        m, q, kv = shape['seq_local'], shape['q_width'], shape['kv_width']
        q_area, kv_area, output_area = m * q, m * kv, m * shape['hidden']
        expected_faults = [('GEMM_A2A', 'local_finite_error', 0),
                           ('GEMM_A2A', 'local_nan', q_area + 2 * kv_area - 1),
                           ('GEMM_A2A', 'route_nan', q_area + 2 * kv_area - 1)]
        expected_faults += [('GEMM_A2A', 'route_segment_boundary', index) for index in
                            (0, q_area - 1, q_area, q_area + kv_area - 1,
                             q_area + kv_area, q_area + 2 * kv_area - 1)]
        expected_faults += [('A2A_GEMM', 'staging_tail_bit', q_area - 1),
                            ('A2A_GEMM', 'staging_nan', 0),
                            ('A2A_GEMM', 'output_tail_finite_error', output_area - 1),
                            ('A2A_GEMM', 'output_nan', 0)]
        first = {direction: next(i for i, item in enumerate(expected, 1) if item[0] == direction)
                 for direction in DIRECTIONS}
        require(Counter((r.get('label'), r.get('fault'), count(r, 'index')) for r in faults) == Counter(expected_faults)
                and all(count(r, 'rank') == count(r, 'generation') == 0 and count(r, 'candidate') == first[r['label']]
                        and count(r, 'gpu_cpu_detected') == count(r, 'restored_full_clean') == count(r, 'diagnostic_only') == 1
                        for r in faults), 'Incomplete checker self-test diagnostics')
    else:
        require(not any(r['kind'] == 'validation_self_test' for r in rows), 'Unexpected checker self-test')
    profile_hosts = [r for r in rows if r['kind'] == 'profile_host']
    if job.get('profile'):
        require(Counter((r.get('label'), count(r, 'rank')) for r in profile_hosts) ==
                Counter((direction, rank) for direction in DIRECTIONS for rank in range(shape['world'])) and
                profile_records.get('profile_cta', 0) > 0, 'Missing profile host/CTA evidence')
        for row in profile_hosts:
            require(row.get('host_launch', 'sequential') == host_launch and count(row, 'instrumented_warmup') >= 10,
                    'Profile launch/warmup mismatch')
            for field in ('production_launch_us', 'instrumented_launch_us', 'production_event_us', 'instrumented_event_us'):
                number(row, field, 0)
        phases = ('instrumented', 'host_stages') if profile_schema == 'host_stages_v1' else (None,)
        phase_directions = [(phase, direction) for phase in phases for direction in DIRECTIONS]
        if epilogue_probe:
            phase_directions += [(phase, DIRECTIONS[0]) for phase in EPILOGUE_PHASES]
        require(Counter((r.get('profile_phase'), r['kind'], r.get('label'), count(r, 'rank')) for r in profile_validation
                        if r['kind'] != 'validation_oracle') == Counter(
                            (phase, kind, direction, rank) for phase, direction in phase_directions
                            for kind in ('correctness', 'route') for rank in range(shape['world'])),
                'Missing profile final numerical/route validation')
        oracle_directions = [(r.get('profile_phase'), r.get('label')) for r in profile_validation
                             if r['kind'] == 'validation_oracle']
        require(Counter(oracle_directions) == (Counter(phase_directions)
                                               if count(config, 'cpu_oracle') else Counter()),
                'Missing profile CPU oracle confirmation')
    else:
        require(not profile_hosts and not profile_validation, 'Unexpected profile evidence')
    host_stages = [r for r in rows if r['kind'] == 'host_stage']
    if profile_schema == 'host_stages_v1':
        host_stage_diagnostics = audit_host_stages(host_stages, profile_validation, shape['world'], host_launch)
    else:
        require(not host_stages, 'Host-stage evidence requires explicit profile schema')
        host_stage_diagnostics = None
    epilogue_rows = [r for r in rows if r['kind'] in EPILOGUE_FIELDS]
    if epilogue_probe:
        epilogue_diagnostics = audit_epilogue(epilogue_rows, profile_validation, host_stages,
            shape, sm_counts, comm[0], host_launch,
            legacy_join=job.get('files', {}).get('benchmarks/sm103/fused_bf16.cu') in
                        LEGACY_EPILOGUE_JOIN_HARNESS_SHA256)
    else:
        epilogue_diagnostics = None
    return {'config': config, 'geometry': shape, 'input_statistics': inputs, 'devices': devices,
            'scheduling': scheduling,
            'schema_defaults': ({'host_launch': 'sequential'} if 'host_launch' not in config else {}) |
                ({'component': 'fused'} if not any('component' in r for r in rows) else {}) |
                ({'oproj_comm_layout': 'rows'} if not explicit_comm_layout else {}),
            'diagnostic_only': diagnostic, 'profile_diagnostic_records': profile_records,
            'profile_schema': profile_schema, 'profile_detail': profile_detail,
            'host_stage_diagnostics': host_stage_diagnostics,
            'epilogue_diagnostics': epilogue_diagnostics,
            'profile_validation_records': len(profile_validation), 'candidates': results}


def audit_run(directory):
    job, receipts, data, evidence = read_receipts(directory)
    attempt = receipts['status.json']['attempt']
    result = audit_log(data[f'attempt{attempt}.log'].decode(), job)
    result.update(run_id=job['run_id'], experiment=job['experiment'], node=job['node'],
                  source_id=job['source_id'], environment_fingerprint=receipts['environment.json']['fingerprint'],
                  build=receipts['fused-build.json'], evidence=evidence,
                  telemetry=audit_telemetry(data['gpu-telemetry.csv'], receipts['gpu-before.json'], job))
    return result


def summarize(directories, output):
    output = Path(output).absolute()
    require(not output.exists() and not output.is_symlink(), f'Output already exists: {output}')
    runs = [audit_run(directory) for directory in directories]
    require(runs and len({run['run_id'] for run in runs}) == len(runs), 'Empty/duplicate input runs')
    report = {'schema': SCHEMA, 'globally_optimal': False, 'model_fitted': False,
              'independent_reference_components_present': any(c['component'] != 'fused' for run in runs for c in run['candidates']),
              'performance_rows': sum(len(run['candidates']) for run in runs if not run['diagnostic_only']),
              'diagnostic_rows': sum(len(run['candidates']) for run in runs if run['diagnostic_only']),
              'runs': runs}
    columns = ('run_id', 'node', 'source_id', 'binary_sha256', 'environment_fingerprint', 'direction', 'component', 'measurement_role',
               'sampling_mode', 'formal_eligible',
               'world', 'global_seq', 'seq_local', 'hidden', 'q_heads', 'kv_heads', 'head_dim',
               'm', 'n', 'k', 'layout', 'host_launch', 'launch', 'graph_epoch_mode', 'collector', 'precision', 'sm_count',
               'candidate', 'comm_ctas', 'tile_policy', 'tile_m', 'tile_n', 'tile_k',
               'schedule_schema', 'raster_requested', 'raster', 'max_swizzle_size', 'effective_swizzle_size',
               'swizzle', 'qkv_rank_swizzle', 'oproj_comm_layout', 'padded_m_tiles', 'padded_n_tiles', 'has_padding', 'scheduled_work_tiles_derived',
               'problem_gemm_flops', 'problem_route_payload_bytes', 'problem_remote_payload_bytes',
               'executed_gemm_flops', 'executed_route_payload_bytes', 'p50_ms', 'p95_ms', 'half_drift')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.fused-summary-', dir=output.parent) as temporary:
        staged = Path(temporary) / 'complete'
        staged.mkdir()
        (staged / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        with (staged / 'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for run in runs:
                for candidate in run['candidates']:
                    if not candidate['performance_accepted']:
                        continue
                    values = run | candidate | candidate['timing'] | {'binary_sha256': run['build']['binary_sha256']}
                    writer.writerow({key: values[key] for key in columns})
        require(not output.exists(), 'Output appeared during audit; refusing overwrite')
        staged.rename(output)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = summarize(args.artifacts, args.output)
    except (ValueError, OSError, tarfile.TarError, KeyError, OverflowError) as error:
        parser.exit(1, f'fused summary audit failed: {error}\n')
    print(json.dumps({'schema': SCHEMA, 'runs': len(result['runs']),
                      'performance_rows': result['performance_rows'], 'diagnostic_rows': result['diagnostic_rows'],
                      'summary': str(args.output / 'summary.json')}))


if __name__ == '__main__':
    main()
