"""Run the full production Fuse + classic cuBLAS matrix without retuning baselines."""
import argparse
import fcntl
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEVICES = {4: '0,2,4,5', 8: '0,1,2,3,4,5,6,7'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/mxfp8_weight/operators/full_v1')
    parser.add_argument('--cps', default='4,8')
    parser.add_argument('--nvcc', default='/usr/local/cuda/bin/nvcc')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--skip-build', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--oproj-comm-model', type=Path,
                        help='explicit frozen OProj calibration; omitted keeps the existing default')
    args = parser.parse_args()
    cps = [int(cp) for cp in args.cps.split(',')]
    if len(cps) != len(set(cps)) or not set(cps) <= set(DEVICES):
        raise ValueError('--cps must be a unique subset of 4,8')
    args.output = args.output.resolve()
    if args.oproj_comm_model is not None:
        args.oproj_comm_model = args.oproj_comm_model.resolve(strict=True)
    build = ROOT / 'build-mxfp8-bench'
    jobs = []
    if not args.skip_build:
        jobs.append(('build', [args.nvcc, '-shared', '-Xcompiler', '-fPIC', '-O3', '-lineinfo',
                              str(HERE / 'classic_gemm.cu'), '-lcublas',
                              '-o', str(build / 'libmxfp8_classic_gemm.so')], {}))
    for cp in cps:
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                   f'--nproc-per-node={cp}', str(HERE / 'operator_bench.py'),
                   '--full', '--cp', str(cp), '--warmup', '10', '--iterations', '50',
                   '--output', str(args.output / f'cp{cp}.json')]
        if args.resume:
            command.append('--resume')
        if args.oproj_comm_model is not None:
            command.extend(['--oproj-comm-model', str(args.oproj_comm_model)])
        jobs.append((f'cp{cp}', command, {'CUDA_VISIBLE_DEVICES': DEVICES[cp]}))
    if args.dry_run:
        for name, command, environment in jobs:
            print(f'{name}: {environment} {shlex.join(command)}')
        return
    build.mkdir(exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    with (build / 'operator_suite.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name, command, environment in jobs:
            print(f'{name}: {shlex.join(command)}', flush=True)
            with (args.output / f'{name}.log').open('a' if args.resume else 'w') as log:
                run = subprocess.run(command, cwd=ROOT,
                                     env=dict(os.environ, OMP_NUM_THREADS='1', **environment),
                                     stdout=log, stderr=subprocess.STDOUT)
            if run.returncode:
                print((args.output / f'{name}.log').read_text()[-6000:])
                raise SystemExit(run.returncode)
            print(f'{name}: complete', flush=True)


if __name__ == '__main__':
    main()
