#!/usr/bin/env python3
"""Small + largest default geometry, CP4/8, both boundaries/backends/launches."""
import argparse
from pathlib import Path
import sm103_batch as batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--directions', default='qkv,oproj')
    args = p.parse_args()
    spec = argparse.Namespace(directions=args.directions, models='llama31_405b',
        seqs=(1024,524288), cps=(4,8), devices='0,1,2,3,4,5,6,7', sm_count=148,
        stage='smoke', results=args.results, library=batch.ROOT/'build/sm103/libfuse_cublaslt_runner.so',
        python='/root/workspace_wct/bench-env/bin/python', te_root=None)
    spec.cache_namespace = batch.bench.fingerprint(spec)
    jobs = []
    for case in batch.bench.cases(spec):
        for backend in ('te_ub','cublaslt_nccl'):
            for launch in ('eager','graph'):
                jobs.append(batch.bench.make_job(spec, case, backend, launch,
                                                batch.bench.initial_configs(backend, smoke=True)[0]))
    plan = args.results/'smoke_plan.json'
    batch.atomic_json(plan, {'stage': 'smoke', 'fingerprint': batch.bench.fingerprint(spec), 'jobs': jobs})
    batch.execute(plan, 300)


if __name__ == '__main__':
    main()
