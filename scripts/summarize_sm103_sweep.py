#!/usr/bin/env python3
"""Offline sweep selection, with opt-in consumption of complete partial groups."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import itertools
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile

from sm103_batch import bench


def expected_candidates():
    args = argparse.Namespace(directions='qkv,oproj', models='',
        seqs=(1024, 4096, 16384, 131072, 262144, 524288), cps=(4, 8),
        devices='0,1,2,3,4,5,6,7')
    expected = {}
    for case, backend, launch in itertools.product(
            bench.cases(args), ('cublaslt_nccl', 'te_ub'), ('eager', 'graph')):
        group = bench.group_key(case, backend, launch)
        for config in bench.initial_configs(backend):
            expected[(group, bench.digest(config))] = case
    return expected


def load_plan(results, expected_fingerprint=None, *, raw_plan=None):
    """Keep the original full plan authoritative, even for a partial snapshot."""
    if raw_plan is None:
        raw_plan = (results / 'sweep_plan.json').read_bytes()
    plan = json.loads(raw_plan)
    if plan['stage'] != 'sweep' or plan['precision'] != 'bf16':
        raise ValueError('expected BF16 sweep plan')
    if expected_fingerprint is not None and plan['fingerprint'] != expected_fingerprint:
        raise ValueError('measurement fingerprint differs from the expected source plan')
    expected = expected_candidates()
    actual = Counter((j['group'], bench.digest(j['config'])) for j in plan['jobs'])
    if actual != Counter({key: 1 for key in expected}):
        raise ValueError('sweep plan does not cover the complete historical candidate matrix exactly')
    outputs = set()
    for job in plan['jobs']:
        key = (job['group'], bench.digest(job['config']))
        if job['case'] != expected[key] or job['warmup'] < 10 or job['iterations'] < 50:
            raise ValueError(f'incorrect geometry or sampling contract: {job["group"]}')
        if job['group'] != bench.group_key(job['case'], job['backend'], job['launch']):
            raise ValueError('group and backend/launch disagree')
        if job['env'].get('FUSE_SM103_MEASUREMENT') != 'v2':
            raise ValueError('missing random-input/warmup protocol')
        remote = Path(job['output'])
        if remote.parent.name != 'sweep' or remote.name in outputs:
            raise ValueError('invalid or duplicate output path')
        outputs.add(remote.name)
    return plan, raw_plan


def measurement_row(results, plan, job, *, stats=None):
    # Artifacts retain remote absolute paths in the plan. Remap only the
    # known sweep leaf; never rewrite plans or raw measurement files.
    remote = Path(job['output'])
    path = results / 'sweep' / remote.name
    if stats is None:
        stats = bench.read_measurement(path, job)
    return dict(group=job['group'], **job['case'], backend=job['backend'],
        launch=job['launch'], precision='bf16', selection_stage='sweep',
        independently_remeasured=False, **stats,
        config_json=json.dumps(job['config'], sort_keys=True),
        samples=job['iterations'], warmup=job['warmup'],
        raw=str(path), remote_raw=str(remote), fingerprint=plan['fingerprint'])


def summarize(results, destination):
    results, destination = Path(results).resolve(), Path(destination).resolve()
    plan, raw_plan = load_plan(results)
    winners = {}
    for job in plan['jobs']:
        row = measurement_row(results, plan, job)
        # Deterministic tie break; p95 is reported, not silently optimized.
        previous = winners.get(job['group'])
        if previous is None or (row['p50_ms'], row['config_json']) < (
                previous['p50_ms'], previous['config_json']):
            winners[job['group']] = row
    rows = [winners[key] for key in sorted(winners)]
    report = dict(schema='sm103_full_sweep_summary_v1', precision='bf16',
        selection_stage='sweep', independently_remeasured=False,
        globally_optimal=False, fingerprint=plan['fingerprint'],
        plan_sha256=hashlib.sha256(raw_plan).hexdigest(),
        validated_logical_candidates=len(plan['jobs']), winner_groups=len(rows),
        results=str(results),
        scope='Historical defaults; S1024/4096/16384/131072/262144/524288; CP4/8; '
              'QKV/OProj; cuBLASLt+NCCL/TE UB; eager/graph',
        limitation='Best observed p50 within the sweep grid; no refine or independent '
                   'formal remeasurement, per user scope change.')
    # Validate every candidate before publishing anything; refuse overwrite.
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / 'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (destination / 'coverage.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


def inspect_partial(results, *, expected_fingerprint, seqs,
                    directions=('qkv', 'oproj'), models=(), cps=(4, 8),
                    launches=('eager',), raw_plan=None, read_row=measurement_row):
    """Shared inspection for local consumption and a read-only source snapshot."""
    if not expected_fingerprint:
        raise ValueError('partial consumption requires an expected measurement fingerprint')
    if not seqs:
        raise ValueError('partial consumption requires explicit sequence lengths')
    results = Path(results).resolve()
    plan, raw_plan = load_plan(results, expected_fingerprint, raw_plan=raw_plan)
    filters = dict(direction=tuple(directions), model=tuple(models), seq=tuple(seqs),
                   cp=tuple(cps), launch=tuple(launches))
    if any(not filters[key] for key in ('direction', 'cp', 'launch')):
        raise ValueError('direction, CP and launch filters cannot be empty')

    def value(job, key):
        return job['launch'] if key == 'launch' else job['case'][key]

    selected = [job for job in plan['jobs'] if all(
        not wanted or value(job, key) in wanted for key, wanted in filters.items())]
    for key, wanted in filters.items():
        missing = set(wanted) - {value(job, key) for job in selected}
        if missing:
            raise ValueError(f'{key} filter values have no selected cases in the original plan: {sorted(missing)}')
    if not selected:
        raise ValueError('filters select no groups')
    jobs_by_group = defaultdict(list)
    for job in selected:
        jobs_by_group[job['group']].append(job)
    winners, groups = {}, []
    validated = 0
    for group, jobs in sorted(jobs_by_group.items()):
        rows, unavailable = [], []
        for job in jobs:
            try:
                rows.append(read_row(results, plan, job))
            except FileNotFoundError as error:
                unavailable.append(dict(config_digest=bench.digest(job['config']),
                    reason='missing_raw_or_rank', detail=str(error)))
            except (ValueError, KeyError, TypeError) as error:
                unavailable.append(dict(config_digest=bench.digest(job['config']),
                    reason='invalid_measurement', detail=f'{type(error).__name__}: {error}'))
        validated += len(rows)
        state = dict(group=group, **jobs[0]['case'], backend=jobs[0]['backend'],
            launch=jobs[0]['launch'], expected_candidates=len(jobs),
            validated_candidates=len(rows), status='pending' if unavailable else 'complete')
        if unavailable:
            # No "best so far" is published under a winner label.
            state['unavailable_candidates'] = unavailable
        else:
            winner = min(rows, key=lambda row: (row['p50_ms'], row['config_json']))
            winners[group] = winner
            state['winner_group'] = group
        groups.append(state)

    cases = {}
    for job in selected:
        key = (job['case']['direction'], job['case']['model'], job['case']['seq'],
               job['case']['cp'], job['launch'])
        cases[key] = job
    strongest, pending_comparisons = [], []
    for _, job in sorted(cases.items()):
        required = {backend: bench.group_key(job['case'], backend, job['launch'])
                    for backend in ('cublaslt_nccl', 'te_ub')}
        missing = [group for group in required.values() if group not in winners]
        if missing:
            pending_comparisons.append(dict(**job['case'], launch=job['launch'],
                status='pending', missing_complete_groups=missing))
            continue
        row = min((winners[group] for group in required.values()),
                  key=lambda item: (item['p50_ms'], item['backend'], item['config_json']))
        strongest.append(row | {'comparison_groups': required,
            'comparison_kind': 'cross_node_initial_reference',
            'node2_same_boundary_validation_required': True})
    report = dict(schema='sm103_partial_sweep_summary_v1', precision='bf16',
        selection_stage='sweep', independently_remeasured=False, globally_optimal=False,
        partial=True, full_stage_achieved=False, fingerprint=plan['fingerprint'],
        expected_fingerprint=expected_fingerprint,
        plan_sha256=hashlib.sha256(raw_plan).hexdigest(), results=str(results),
        filters={key: list(wanted) for key, wanted in filters.items()},
        selected_logical_candidates=len(selected), validated_logical_candidates=validated,
        selected_groups=len(groups), winner_groups=len(winners),
        pending_groups=len(groups)-len(winners), strongest_baselines=len(strongest),
        pending_comparisons=len(pending_comparisons),
        limitation='Only complete selected sweep groups yield winners; partial coverage is '
                   'not a stage achievement. No independent formal remeasurement or '
                   'same-node/layout/timing-boundary validation is implied.')
    payloads = {'coverage.json': report, 'groups.json': groups,
        'winners.json': [winners[key] for key in sorted(winners)],
        'strongest.json': strongest, 'pending_comparisons.json': pending_comparisons}
    return payloads, selected


def summarize_partial(results, destination, **filters):
    """Consume complete groups only; never certify a partial evaluation matrix.

    The caller must verify the fetched snapshot/archive before calling this
    function. Its sweep_plan.json must be the unmodified original FULL plan;
    selected raw/rank files may be absent. The required fingerprint binds the
    plan to the caller's expected measurement source, not to today's checkout.
    It cannot by itself authenticate files mixed in from another artifact.
    """
    payloads, _ = inspect_partial(results, **filters)
    # Publish only after the original plan, filters and all selected candidates
    # have been inspected. Never overwrite a previous consumption snapshot.
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        (destination / name).write_text(json.dumps(payload, indent=2) + '\n')
    report = payloads['coverage.json']
    print(json.dumps(report, indent=2))
    return report


class SnapshotError(RuntimeError):
    """Unsafe or changing source files invalidate the entire snapshot attempt."""


class CapturedMeasurementPath:
    """The reader's read_text/with_suffix interface over safely captured bytes.

    Validation must read the exact bytes that will be archived, not reopen live
    paths between the no-symlink capture and the final source stability check.
    """
    def __init__(self, path, files):
        self.path, self.files = path, files

    def read_text(self):
        return self.files[str(self.path)][0].decode('utf-8')

    def with_suffix(self, suffix):
        return CapturedMeasurementPath(self.path.with_suffix(suffix), self.files)

    def __str__(self):
        return str(self.path)


def safe_relative_path(name):
    path = PurePosixPath(name)
    if (path.is_absolute() or not path.parts or any(part in ('', '.', '..') for part in path.parts)
            or '\\' in name or '\x00' in name or str(path) != name):
        raise SnapshotError(f'unsafe snapshot member path: {name!r}')
    return path


def stable_file(results, relative):
    """Read regular files below results, without following symlinks or locking.

    Opening each component relative to its directory fd prevents a symlink
    replacement from redirecting a read outside the selected source tree.
    """
    parts = safe_relative_path(relative).parts
    directory_fd = os.open(results, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in parts[:-1]:
            next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                              dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                     dir_fd=directory_fd)
        with os.fdopen(fd, 'rb') as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise SnapshotError(f'not a regular snapshot source file: {relative}')
            content = stream.read()
            after = os.fstat(stream.fileno())
            current = os.stat(parts[-1], dir_fd=directory_fd, follow_symlinks=False)
        def version(info):
            return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
                    info.st_mtime_ns, info.st_ctime_ns)
        if version(before) != version(after) or version(after) != version(current):
            raise SnapshotError(f'source changed while reading: {relative}')
        return content, version(current)
    except FileNotFoundError:
        raise
    except OSError as error:
        raise SnapshotError(f'cannot safely read snapshot source {relative}: {error}') from error
    finally:
        os.close(directory_fd)


def snapshot_partial(results, destination, **filters):
    """Archive only completed selected groups; never mutate or lock source files.

    Uses the existing measurement reader from the node's installed source tree.
    Run this uploaded script with python -B and PYTHONPATH=<source>/scripts;
    imports then require neither a source sync nor source __pycache__ writes.
    No GPU, subprocess, cloud transfer or workload lock is used.
    """
    results = Path(results).resolve(strict=True)
    destination = Path(destination).resolve()
    if destination == results or results in destination.parents:
        raise SnapshotError('snapshot output must be outside the source results tree')
    if destination.exists():
        raise FileExistsError(f'snapshot output already exists: {destination}')
    raw_plan, plan_version = stable_file(results, 'sweep_plan.json')
    captured = {}

    def snapshot_row(source, plan, job):
        raw = PurePosixPath('sweep') / Path(job['output']).name
        paths = [str(raw)] + [str(raw.with_suffix(f'.rank{rank}.json'))
                             for rank in range(job['case']['cp'])]
        candidate = {name: stable_file(source, name) for name in paths}
        stats = bench.read_measurement(CapturedMeasurementPath(raw, candidate), job)
        row = measurement_row(source, plan, job, stats=stats)
        for name, original in candidate.items():
            if stable_file(source, name) != original:
                raise SnapshotError(f'source changed during measurement validation: {name}')
        captured[job['output']] = candidate
        return row

    payloads, selected = inspect_partial(results, **filters, raw_plan=raw_plan,
                                         read_row=snapshot_row)
    complete = {row['group'] for row in payloads['winners.json']}
    selected_complete = [job for job in selected if job['group'] in complete]
    sources = {'sweep_plan.json': (raw_plan, plan_version)}
    for job in selected_complete:
        sources.update(captured[job['output']])

    def verify_unchanged():
        for name, original in sources.items():
            if stable_file(results, name) != original:
                raise SnapshotError(f'source changed before snapshot publication: {name}')

    verify_unchanged()
    members = {'results/' + name: content for name, (content, _) in sources.items()}
    for name, payload in payloads.items():
        members['provenance/' + name] = (json.dumps(payload, indent=2) + '\n').encode()
    for name in members:
        safe_relative_path(name)
    coverage = payloads['coverage.json']
    manifest = dict(schema='sm103_partial_sweep_snapshot_v1',
        fingerprint=coverage['fingerprint'], plan_sha256=coverage['plan_sha256'],
        source_results=str(results), filters=coverage['filters'],
        complete_groups=sorted(complete), pending_groups=coverage['pending_groups'],
        archived_candidates=len(selected_complete),
        files={name: dict(size_bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
               for name, content in sorted(members.items())})
    members['manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    destination.mkdir(parents=True, exist_ok=False)
    pending_archive = destination / 'snapshot.tar.gz.pending'
    with pending_archive.open('xb') as output:
        with tarfile.open(fileobj=output, mode='w:gz') as archive:
            for name, content in sorted(members.items()):
                member = tarfile.TarInfo(str(safe_relative_path(name)))
                member.size, member.mode, member.mtime = len(content), 0o644, 0
                archive.addfile(member, io.BytesIO(content))
    # Never publish a receipt/final archive when a selected source changed while
    # the archive was being built. An incomplete file remains diagnostic-only.
    verify_unchanged()
    archive_path = destination / 'snapshot.tar.gz'
    pending_archive.rename(archive_path)
    digest = hashlib.sha256()
    with archive_path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    receipt = dict(schema='sm103_partial_sweep_snapshot_receipt_v1',
        archive=str(archive_path), archive_sha256=digest.hexdigest(),
        archive_bytes=archive_path.stat().st_size, fingerprint=coverage['fingerprint'],
        plan_sha256=coverage['plan_sha256'], complete_groups=len(complete),
        pending_groups=coverage['pending_groups'], archived_candidates=len(selected_complete),
        archived_measurement_files=len(sources)-1, strongest_baselines=coverage['strongest_baselines'])
    (destination / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, separators=(',', ':')))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True, type=Path,
                        help='Fetched artifact results directory containing sweep_plan.json')
    parser.add_argument('--output', required=True, type=Path,
                        help='New local summary directory; existing directory is never overwritten')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--partial', action='store_true',
                      help='Consume complete selected groups; never report full-stage completion')
    mode.add_argument('--snapshot', action='store_true',
                      help='Archive complete selected groups read-only; no GPU, cloud upload or source sync')
    parser.add_argument('--expected-fingerprint',
                        help='Required for --partial; copied from the verified source receipt/plan')
    parser.add_argument('--directions', help='Comma-separated plan directions; partial default: qkv,oproj')
    parser.add_argument('--models', help='Comma-separated original model labels; partial default: all')
    parser.add_argument('--seqs', type=bench.ints, help='Explicit global sequence lengths; required for --partial')
    parser.add_argument('--cps', type=bench.ints, help='Comma-separated CP sizes; partial default: 4,8')
    parser.add_argument('--launches', help='Comma-separated launch modes; partial default: eager')
    args = parser.parse_args()
    if args.partial or args.snapshot:
        if not args.expected_fingerprint or not args.seqs:
            parser.error('--partial/--snapshot requires --expected-fingerprint and explicit --seqs')
        operation = snapshot_partial if args.snapshot else summarize_partial
        operation(args.results, args.output,
            expected_fingerprint=args.expected_fingerprint, seqs=args.seqs,
            directions=tuple((args.directions or 'qkv,oproj').split(',')),
            models=tuple(args.models.split(',')) if args.models else (),
            cps=args.cps or (4, 8), launches=tuple((args.launches or 'eager').split(',')))
    else:
        if any(getattr(args, name) is not None for name in (
                'expected_fingerprint', 'directions', 'models', 'seqs', 'cps', 'launches')):
            parser.error('filters and --expected-fingerprint require --partial/--snapshot; full audit is unfiltered')
        summarize(args.results, args.output)
