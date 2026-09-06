"""Serialized, resumable finite-policy search and independent formal sampling."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from backward_matrix import ROOT, full_matrix
from run_suite import PROFILES, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'results/mxfp8_weight/backward_full_v1')
    p.add_argument('--cps', default='4,8')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    matrix = full_matrix()
    cps = [int(c) for c in args.cps.split(',')]
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'suite.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(args.output / 'matrix.json', matrix)
        if args.dry_run:
            print(json.dumps(dict(settings=192, fp32_rows=1536, cps=cps, profiles=PROFILES,
                                  formal='10 warmups/50 samples; B,W,sequential total',
                                  selection='minimum short-sweep B p50 per shape and launch'), indent=2))
            return
        start = time.monotonic()
        jobs = []

        def job(cp, profile, backend, phase, policy=None):
            tag = f'{backend}_{phase}_{profile}_cp{cp}'
            output = args.output / f'{tag}.json'
            env = {k:v for k,v in os.environ.items() if not k.startswith('NCCL_')}
            env.update(CUDA_VISIBLE_DEVICES=next(c['visible_devices'] for c in matrix if c['cp'] == cp),
                       OMP_NUM_THREADS='1', NCCL_IB_DISABLE='1', NCCL_GRAPH_REGISTER='1', NCCL_LOCAL_REGISTER='1')
            env.update(PROFILES[profile])
            cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                   f'--nproc-per-node={cp}', str(Path(__file__).with_name('backward_bench.py')),
                   '--full', '--resume', '--backends', backend, '--phase', phase,
                   '--grad-dtypes', 'fp32', '--output', str(output)]
            if policy is not None:
                path = args.output / f'{tag}.launches.json'
                ids = args.output / f'{tag}.ids.json'
                write(path, policy); write(ids, sorted(policy))
                cmd += ['--launch-policy', str(path), '--case-ids', str(ids), '--skip-sweep']
            print(f'JOB {tag}; log={output.with_suffix(".log")}', flush=True)
            begin = time.monotonic()
            with output.with_suffix('.log').open('a') as log:
                subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            report = json.loads(output.read_text())
            if not report['complete']:
                raise RuntimeError(f'incomplete job: {output}')
            jobs.append(dict(tag=tag, command=cmd, seconds=time.monotonic()-begin,
                             output=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest()))
            write(args.output / 'jobs.json', dict(jobs=jobs, wall_seconds=time.monotonic()-start))
            return report

        for cp in cps:
            surveys = {profile:job(cp, profile, 'cublaslt_nccl', 'sweep') for profile in PROFILES}
            candidates = {}
            for profile, report in surveys.items():
                for item in report['cases']:
                    for row in item['search']:
                        key = (item['case']['id'], row['launch'])
                        candidates.setdefault(key, []).append((row['data']['p50_us'], profile))
            expected = {(c['id'], l) for c in matrix if c['cp'] == cp for l in ('eager', 'graph')}
            if set(candidates) != expected or any(len(v) != len(PROFILES) for v in candidates.values()):
                raise RuntimeError('incomplete NCCL search coverage')
            policy = {key:min(v)[1] for key,v in candidates.items()}
            write(args.output / f'nccl_policy_cp{cp}.json',
                  [dict(id=k[0], launch=k[1], profile=v) for k,v in sorted(policy.items())])
            for profile in PROFILES:
                selected = {}
                for (case_id, launch), winner in sorted(policy.items()):
                    if winner == profile:
                        selected.setdefault(case_id, []).append(launch)
                if selected:
                    job(cp, profile, 'cublaslt_nccl', 'formal', selected)
            job(cp, 'auto', 'teub', 'formal')
        command = [sys.executable, str(Path(__file__).with_name('backward_report.py')), str(args.output)]
        if set(cps) == {4,8}:
            command += ['--require-full']
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
