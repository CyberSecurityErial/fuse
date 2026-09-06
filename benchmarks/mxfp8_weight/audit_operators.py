"""Audit full production Fuse/classic cuBLAS coverage and raw rank-max samples."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

import numpy as np

from matrix import ROOT, full_matrix
from backward_matrix import full_matrix as backward_matrix


def git_output(*args):
    result = subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True)
    if result.returncode:
        raise ValueError(result.stderr.decode(errors='replace').strip())
    return result.stdout


def is_build_artifact(source):
    """Only explicit compiled-file names may use a binary snapshot mapping."""
    path = Path(source)
    return (path.suffix in ('.a', '.o', '.cubin', '.fatbin') or
            re.search(r'\.so(?:\.\d+)*$', path.name) is not None)


def parse_artifacts(entries):
    artifacts = {}
    for entry in entries:
        source, separator, actual = entry.partition('=')
        if not separator or not source or not actual:
            raise ValueError('--artifact requires RECORDED_PATH=ACTUAL_PATH')
        if source in artifacts:
            raise ValueError(f'duplicate artifact mapping: {source}')
        if not is_build_artifact(source):
            raise ValueError(f'artifact mapping cannot replace source: {source}')
        artifacts[source] = Path(actual).resolve()
    return artifacts


def recorded_hash(source, revision, artifacts):
    path = Path(source)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('recorded source must be repository-relative without ..')
    if is_build_artifact(source):
        data = artifacts.get(source, ROOT / source).read_bytes()
        if source in artifacts and not data.startswith((b'\x7fELF', b'!<arch>\n', b'\x50\xed\x55\xba')):
            raise ValueError('mapped artifact is not an ELF, archive or CUDA fatbinary')
    elif revision is not None:
        # A missing revision source is an error, never a worktree fallback.
        data = git_output('show', f'{revision}:{source}')
    else:
        data = (ROOT / source).read_bytes()
    return hashlib.sha256(data).hexdigest()


def audit(paths, require_full=False, source_revision=None, artifacts=None):
    """Check revision sources and explicitly mapped binaries, plus formal samples.

    Artifact keys must occur verbatim in an input report's ``sources``. Relative
    snapshot paths are relative to the caller's cwd; unmapped artifacts always
    use the current repository file, even when source_revision is specified.
    """
    artifacts = parse_artifacts(f'{source}={actual}' for source, actual in (artifacts or {}).items())
    revision = (git_output('rev-parse', '--verify', '--end-of-options',
                           f'{source_revision}^{{commit}}').decode().strip()
                if source_revision is not None else None)
    expected_cases = {case['id']: case for case in full_matrix() + backward_matrix()}
    expected = set()
    for case in expected_cases.values():
        modes = ('immediate', 'deferred') if 'b_mnk' in case else (None,)
        expected.update((case['id'], backend, launch, mode)
                        for backend in ('fuse', 'pure_cublas')
                        for launch in ('eager', 'graph') for mode in modes)
    seen, cases, errors = set(), set(), []
    source_hashes, source_errors = {}, {}
    recorded_sources = set()
    for path in paths:
        report = json.loads(path.read_text())
        if report['schema'] != 'mxfp8-production-operators-v1':
            errors.append(f'{path}: schema')
            continue
        if require_full and not report['complete']:
            errors.append(f'{path}: incomplete runner')
        for source, expected_hash in report['sources'].items():
            recorded_sources.add(source)
            if source not in source_hashes and source not in source_errors:
                try:
                    source_hashes[source] = recorded_hash(source, revision, artifacts)
                except (OSError, ValueError) as exc:
                    source_errors[source] = str(exc)
            if source in source_errors:
                errors.append(f'{path}: cannot verify source {source}: {source_errors[source]}')
                continue
            if source_hashes[source] != expected_hash:
                errors.append(f'{path}: changed source {source}')
        for item in report['cases']:
            case = item['case']
            if case != expected_cases.get(case['id']):
                errors.append(f'{case["id"]}: registry mismatch')
                continue
            if case['id'] in cases:
                errors.append(f'{case["id"]}: duplicate case')
            cases.add(case['id'])
            for record in item['records']:
                key = (case['id'], record['backend'], record['launch'], record['weight_mode'])
                if key in seen or key not in expected:
                    errors.append(f'{key}: duplicate/unknown record')
                seen.add(key)
                if record['warmup'] != 10 or record['iterations'] != 50:
                    errors.append(f'{key}: not formal 10+50')
                backward = 'b_mnk' in case
                if backward and (record['grad_dtype'] != 'fp32' or
                                 record['beta'] != int(record['weight_mode'] == 'deferred')):
                    errors.append(f'{key}: FP32/beta mismatch')
                for phase in (('data', 'weight', 'total') if backward else ('forward',)):
                    timing = record[phase]
                    rank_samples = np.asarray(timing['rank_samples_us'])
                    if (rank_samples.shape != (case['cp'], 50) or
                            not np.isfinite(rank_samples).all() or (rank_samples <= 0).any()):
                        errors.append(f'{key}/{phase}: invalid sample vectors')
                        continue
                    samples = rank_samples.max(axis=0)
                    if not np.array_equal(samples, timing['samples_us']):
                        errors.append(f'{key}/{phase}: not sample-wise rank-max')
                    for field, value in [('p50_us', np.median(samples)),
                                         ('p95_us', np.percentile(samples, 95)),
                                         ('mean_us', np.mean(samples))]:
                        if not math.isclose(timing[field], value, rel_tol=1e-10):
                            errors.append(f'{key}/{phase}: {field} mismatch')
                    mnk = case['b_mnk'] if backward else [case['m'], case['n'], case['k']]
                    flops = (4 if phase == 'total' else 2) * math.prod(mnk)
                    if (timing['flops_per_gpu'] != flops or
                            not math.isclose(timing['tflops_per_gpu'], flops / np.median(samples) / 1e6,
                                             rel_tol=1e-10)):
                        errors.append(f'{key}/{phase}: FLOPs/throughput mismatch')
                def check_metrics(value):
                    if isinstance(value, dict):
                        if 'all_ranks_finite' in value and not value['all_ranks_finite']:
                            errors.append(f'{key}: nonfinite correctness result')
                        for child in value.values():
                            check_metrics(child)
                if not record['correctness'] or not record['config']:
                    errors.append(f'{key}: missing correctness/config')
                check_metrics(record['correctness'])
    missing = expected - seen
    errors.extend(f'unknown artifact mapping: {source}'
                  for source in sorted(set(artifacts) - recorded_sources))
    if require_full and missing:
        errors.append(f'{len(missing)} missing formal records')
    return dict(complete=not missing and not errors, settings=len(cases), expected_settings=384,
                records=len(seen), expected_records=len(expected), errors=errors,
                missing_records=len(missing),
                source_revision=revision,
                artifacts={source: str(actual) for source, actual in artifacts.items()},
                sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+', type=Path)
    parser.add_argument('--require-full', action='store_true')
    parser.add_argument('--source-revision', metavar='REV',
                        help='verify source bytes with git show REV:path; binaries stay current unless mapped')
    parser.add_argument('--artifact', action='append', default=[], metavar='RECORDED_PATH=ACTUAL_PATH',
                        help='map one recorded compiled binary to its snapshot (repeatable; never source files)')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        artifacts = parse_artifacts(args.artifact)
        result = audit(args.paths, args.require_full, args.source_revision, artifacts)
    except ValueError as exc:
        parser.error(str(exc))
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    print(text)
    if result['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
