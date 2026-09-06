"""Build and reproduce both full baseline suites, serialized with low overhead."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'results/mxfp8_weight/reproduce')
    p.add_argument('--nvcc', default='nvcc')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--build-only', action='store_true')
    args = p.parse_args()
    build = ROOT/'build-mxfp8-bench'
    commands = [
        [args.nvcc, '-shared', '-Xcompiler', '-fPIC', '-O3', '-lineinfo',
         str(ROOT/'csrc/baselines/cublaslt_runner.cu'), '-lcublasLt', '-lcublas',
         '-o', str(build/'libfuse_cublaslt_runner.so')],
        [args.nvcc, '-shared', '-Xcompiler', '-fPIC', '-O3', '-lineinfo',
         str(Path(__file__).with_name('backward_gemm.cu')), '-lcublasLt',
         '-o', str(build/'libmxfp8_backward.so')],
    ]
    if not args.build_only:
        for suite, directory in [('run_suite.py', 'full_v2'), ('backward_suite.py', 'backward_full_v1')]:
            commands.append([sys.executable, str(Path(__file__).with_name(suite)),
                             '--output', str(args.output/directory)])
        commands += [[sys.executable, str(Path(__file__).with_name('report.py')), str(args.output/'full_v2')],
                     [sys.executable, str(Path(__file__).with_name('backward_report.py')),
                      str(args.output/'backward_full_v1')],
                     [sys.executable, str(Path(__file__).with_name('export_baselines.py')),
                      '--archive', str(args.output), '--output', str(args.output/'published')]]
    if args.dry_run:
        import shlex
        for command in commands:
            print(shlex.join(command))
        return
    import fcntl
    build.mkdir(exist_ok=True)
    with (build/'baseline_suite.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for command in commands:
            subprocess.run(command, cwd=ROOT, check=True, env=dict(os.environ, OMP_NUM_THREADS='1'))


if __name__ == '__main__':
    main()
