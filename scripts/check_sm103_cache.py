#!/usr/bin/env python3
"""Verify persistent GEMM winners across fresh PGs and NCCL configurations."""
import argparse
import json
from pathlib import Path
import sm103_batch as batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True)
    args = p.parse_args()
    spec = argparse.Namespace(directions='qkv,oproj', models='production_qwen_dense',
        seqs=(16384,), cps=(8,), devices='0,1,2,3,4,5,6,7', sm_count=148,
        stage='smoke', results=args.results, library=batch.ROOT/'build/sm103/libfuse_cublaslt_runner.so',
        python='/root/workspace_wct/bench-env/bin/python', te_root=None)
    fingerprint = batch.bench.fingerprint(spec)
    spec.cache_namespace = fingerprint + '-check-' + batch.bench.digest(str(args.results))
    for phase in ('cold','warm'):
        spec.stage = phase
        jobs = []
        for case in batch.bench.cases(spec):
            backend = 'cublaslt_nccl' if case['direction'] == 'qkv' else 'te_ub'
            config = batch.bench.initial_configs(backend, smoke=True)[0]
            if phase == 'warm' and backend == 'cublaslt_nccl':
                config = config | {'channels': 8}
            for launch in ('eager','graph'):
                jobs.append(batch.bench.make_job(spec, case, backend, launch, config))
        plan = args.results/f'{phase}_plan.json'
        batch.atomic_json(plan, {'stage': phase, 'fingerprint': fingerprint, 'jobs': jobs})
        batch.execute(plan, 180)
        for job in jobs:
            for rank in range(8):
                meta = json.loads(Path(job['output']).with_suffix(f'.rank{rank}.json').read_text())
                if len(meta['measurement_records']) != 1:
                    raise RuntimeError('Cache check unexpectedly timed an unrequested boundary')
                plans = meta['cublaslt_plans']
                if not plans or any(p['disk_cache_hit'] != (phase == 'warm') for p in plans):
                    raise RuntimeError(f'Unexpected {phase} cache status: {plans}')
        print(f'DONE cache {phase}: 4 configurations x 8 ranks, validated', flush=True)


if __name__ == '__main__':
    main()
