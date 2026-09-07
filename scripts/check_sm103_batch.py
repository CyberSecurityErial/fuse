#!/usr/bin/env python3
"""Same-work A/B: four CP8 S16384 candidates, serial vs persistent processes."""
import argparse
import json
from pathlib import Path
import time

import sm103_batch as batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True)
    args = p.parse_args()
    spec = argparse.Namespace(directions='oproj', models='production_qwen_dense',
        seqs=(16384,), cps=(8,), devices='0,1,2,3,4,5,6,7', sm_count=148,
        stage='sweep', results=args.results, library=batch.ROOT/'build/sm103/libfuse_cublaslt_runner.so',
        python='/root/workspace_wct/bench-env/bin/python', te_root=None)
    case = next(batch.bench.cases(spec))
    spec.cache_namespace = batch.bench.fingerprint(spec)
    records = {}
    for mode in ('serial', 'batch'):
        spec.results = args.results / mode
        jobs = []
        for backend in ('te_ub', 'cublaslt_nccl'):
            config = batch.bench.initial_configs(backend)[0]
            for block in (256, 512):
                jobs.append(batch.bench.make_job(spec, case, backend, 'graph', config | {'pack_block': block}))
        plan = spec.results / 'sweep_plan.json'
        batch.atomic_json(plan, {'stage': 'sweep', 'fingerprint': batch.bench.fingerprint(spec), 'jobs': jobs})
        start = time.monotonic()
        batch.execute(plan, 180, serial=mode == 'serial')
        records[mode] = {'wall_s': time.monotonic()-start, 'results': [
            batch.bench.read_measurement(Path(job['output']), job) for job in jobs]}
        batch.atomic_json(args.results/'comparison.json', records)
    records['speedup'] = records['serial']['wall_s'] / records['batch']['wall_s']
    batch.atomic_json(args.results/'comparison.json', records)
    print('BATCH_AB ' + json.dumps(records), flush=True)


if __name__ == '__main__':
    main()
