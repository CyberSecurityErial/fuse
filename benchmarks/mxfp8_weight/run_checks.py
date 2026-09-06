"""Run correctness checks serially and preserve logs; never publish them as perf."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scope', choices=['all', 'cpu', 'operators', 'legacy'], default='all')
    p.add_argument('--build-dir', type=Path, default=ROOT/'build-mxfp8')
    p.add_argument('--output', type=Path, default=ROOT/'results/mxfp8_weight/validation')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = []
    if args.scope in ('all', 'cpu'):
        jobs.append(('cpu', [sys.executable, '-m', 'unittest', '-v', 'test_matrix',
                             'test_backward_matrix', 'test_backward_report',
                             'test_backward_sources', 'test_published',
                             'test_comparison_report', 'test_operator_matrix',
                             'test_profile_report', 'test_optimization_report',
                             'test_operator_audit', 'test_operator_sweep',
                             'test_profile_operators', 'test_calibrate_operators', 'test_qkv_service_model',
                             'test_qkv_forward_service_model',
                             'test_operator_model', 'test_operator_comm_model',
                             'test_operator_policy_ab', 'test_operator_policy_report'], HERE))
    if args.scope in ('all', 'operators'):
        jobs.append(('operators', [sys.executable, str(HERE/'test_operators.py'),
                                    '--library', str(args.build_dir/'libfuse_mxfp8_torch_bridge.so'),
                                    '--output', str(args.output/'operator_correctness.json')], ROOT))
    if args.scope in ('all', 'legacy'):
        for name in ('backward_smoke', 'fp8_smoke', 'qkvproj_a2a_smoke', 'fuse_smoke'):
            jobs.append((name, [str(args.build_dir/name)], ROOT))
    results = []
    for name, command, cwd in jobs:
        start = time.monotonic()
        log = args.output / f'{name}.log'
        with log.open('w') as f:
            run = subprocess.run(command, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
        result = dict(name=name, command=command, returncode=run.returncode,
                      wall_seconds=time.monotonic()-start, log=log.name,
                      log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
        if name not in ('cpu', 'operators'):
            result['executable_sha256'] = hashlib.sha256(Path(command[0]).read_bytes()).hexdigest()
        results.append(result)
        report = dict(kind='correctness_not_performance', scope=args.scope,
                      complete=len(results) == len(jobs),
                      passed=all(r['returncode'] == 0 for r in results), checks=results)
        (args.output / f'{args.scope}_checks.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f'{name}: {"PASS" if run.returncode == 0 else "FAIL"}; log={log}', flush=True)
        if run.returncode:
            print(log.read_text()[-6000:])
            raise SystemExit(run.returncode)


if __name__ == '__main__':
    main()
