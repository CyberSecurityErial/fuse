#!/usr/bin/env python3
"""Execution-only batching for the unchanged SM103 measurement/search contract.

Reuses SM90 run_initialized and its cuBLASLt caches. Process-static environment,
launch mode, direction, CP and NCCL stream priority never change within a batch.
Executor hashes are saved separately; old validated measurements remain usable.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


bench = module("sm103_contract", ROOT / "benchmarks/sm103/bench.py")


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def batch_key(job):
    # GEMM caches omit launch mode; never share a process across eager/graph.
    return json.dumps([job["case"]["direction"], job["backend"], job["launch"],
                       job["case"]["cp"], job["config"].get("high_priority"),
                       job["env"]], sort_keys=True)


def batches(jobs):
    grouped = {}
    for job in jobs:
        grouped.setdefault(batch_key(job), []).append(job)
    return list(grouped.values())


def worker_flags(job):
    cmd = job["command"]
    return cmd[cmd.index("--expected-sms") + 2:]


def measurement_key(job):
    flags = worker_flags(job).copy()
    flags[flags.index("--json-out") + 1] = "<output>"
    return json.dumps([batch_key(job), flags], sort_keys=True)


def copy_measurement(source, destination, provenance):
    """Keep original raw bytes and explicitly record equivalent-geometry reuse."""
    src, dst = Path(source['output']), Path(destination['output'])
    if measurement_key(source) != measurement_key(destination):
        raise ValueError('cannot reuse different measurement contracts')
    bench.read_measurement(src, source)
    if dst.exists():
        bench.read_measurement(dst, destination)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    for rank in range(destination['case']['cp']):
        shutil.copy2(src.with_suffix(f'.rank{rank}.json'), dst.with_suffix(f'.rank{rank}.json'))
    if src.with_suffix('.gpu-before.json').exists():
        shutil.copy2(src.with_suffix('.gpu-before.json'), dst.with_suffix('.gpu-before.json'))
    atomic_json(dst.with_suffix('.reuse.json'), {'source': str(src), **provenance})
    shutil.copy2(src, dst)
    bench.read_measurement(dst, destination)


def import_results(plan_path, source_plan_path):
    target = json.loads(Path(plan_path).read_text())
    source = json.loads(Path(source_plan_path).read_text())
    if target['fingerprint'] != source['fingerprint'] or target['stage'] != source['stage']:
        raise ValueError('source/build/environment or stage differs; no results imported')
    lookup = {measurement_key(j): j for j in source['jobs']}
    count = 0
    for job in target['jobs']:
        previous = lookup.get(measurement_key(job))
        if previous is None or Path(job['output']).exists():
            continue
        try:
            bench.read_measurement(Path(previous['output']), previous)
        except (OSError, KeyError, ValueError):
            continue
        copy_measurement(previous, job, {'source_plan': str(source_plan_path),
                                         'fingerprint': source['fingerprint']})
        count += 1
    print(f'IMPORTED {count} validated measurements from {source_plan_path}', flush=True)


def worker(manifest):
    process_started = time.monotonic()
    jobs = json.loads(Path(manifest).read_text())
    first = jobs[0]
    if any(batch_key(job) != batch_key(first) for job in jobs):
        raise ValueError("mixed process-static batch")
    if any(os.environ.get(k) != v for k, v in first["env"].items()):
        raise ValueError("worker environment differs from manifest")
    preflight = module("sm103_preflight", ROOT / "benchmarks/sm103/worker.py")
    sys.argv = ["preflight", "--direction", first["case"]["direction"],
                "--backend", first["backend"], "--expected-sms",
                first["command"][first["command"].index("--expected-sms") + 1],
                "--preflight-only"]
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        preflight.main()
    metadata = json.loads(captured.getvalue())
    if metadata['oproj_layout'] != first['env'].get('FUSE_SM103_OPROJ_LAYOUT', 'legacy'):
        raise ValueError('worker OProj layout differs from manifest')
    import measurement
    import torch
    import torch.distributed as dist
    import transformer_engine_torch as tex
    path = ROOT / "benchmarks" / preflight.WORKERS[first["case"]["direction"], first["backend"]]
    old = preflight.load_boundary(first['case']['direction'], first['backend'],
                                  metadata['oproj_layout'])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    sys.argv = [str(path), *worker_flags(first)]
    initial_args = old.parse_args()
    if first["backend"] == "te_ub":
        dist.init_process_group("nccl", device_id=device)
        helper = tex.CommOverlapHelper(dist.group.WORLD, dist.group.WORLD)
        close = old.close_plans
    else:
        old.initialize_process_group(initial_args, device)
        close = old.close_cublaslt_runners
    rank, world = dist.get_rank(), dist.get_world_size()
    startup_s = time.monotonic() - process_started
    previous_geometry = None
    best = {}
    for index, job in enumerate(jobs, 1):
        # Bound cached workspace memory across the full shape matrix. Keep plans
        # through all transport/pack candidates of the same GEMM geometry.
        geometry = tuple(job["case"][k] for k in ("m", "n", "k"))
        if geometry != previous_geometry:
            close()
        previous_geometry = geometry
        sys.argv = [str(path), *worker_flags(job)]
        args = old.parse_args()
        measurement.reset()
        output = Path(job["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        if rank == 0:
            atomic_json(Path(manifest).with_suffix(".progress.json"), {"candidate": index})
            observation = bench.observe_devices(job["env"]["CUDA_VISIBLE_DEVICES"], samples=1, check_idle=False)
            atomic_json(output.with_suffix(".gpu-before.json"), observation)
            print(f"RUN {index}/{len(jobs)} {job['group']} {json.dumps(job['config'], sort_keys=True)}", flush=True)
        dist.barrier()
        started = time.monotonic()
        # Each candidate has its own original worker log; screen gets just progress.
        with output.with_suffix(f".worker-rank{rank}.log").open("w") as log:
            with contextlib.redirect_stdout(log):
                try:
                    if job["backend"] == "te_ub":
                        old.run_initialized(args, rank, world, device, helper)
                    else:
                        old.run_initialized(args, device)
                except Exception as error:
                    # Keep raw unstable samples without printing 8 copies of
                    # large arrays into the terminal. Never certify this rank.
                    atomic_json(output.with_suffix(f'.failed-rank{rank}.json'), metadata | {
                        'error': str(error), 'measurement_records': measurement.RECORDS,
                        'input_statistics': measurement.INPUTS})
                    raise
        elapsed = time.monotonic() - started
        preflight.record_result_layout(output, metadata)
        atomic_json(output.with_suffix(f".rank{rank}.json"), metadata | {
            "executor": "sm103_batch_v1", "batch_manifest": str(manifest),
            "batch_index": index, "startup_s": startup_s, "candidate_wall_s": elapsed,
            "measurement_records": measurement.RECORDS, "input_statistics": measurement.INPUTS})
        rank_path = output.with_suffix(f".rank{rank}.json")
        rank_data = json.loads(rank_path.read_text())
        rank_data['cublaslt_plans'] = [plan.info for plan in getattr(old, '_PLAN_CACHE',
                                        getattr(old, '_CUBLASLT_RUNNERS', {})).values()]
        atomic_json(rank_path, rank_data)
        dist.barrier()
        if rank == 0:
            stats = bench.read_measurement(output, job)
            best[job["group"]] = min(stats["p50_ms"], best.get(job["group"], float("inf")))
            print(f"DONE {index}/{len(jobs)} {job['group']} p50={stats['p50_ms']:.6f}ms "
                  f"p95={stats['p95_ms']:.6f}ms best={best[job['group']]:.6f}ms "
                  f"candidate_wall={elapsed:.3f}s startup={startup_s:.3f}s", flush=True)
    close()
    dist.destroy_process_group()


@contextlib.contextmanager
def telemetry(devices, path):
    # Read-only monitor, one process per batch, never changes power/clock settings.
    with Path(path).open('w') as log:
        process = subprocess.Popen(['nvidia-smi', f'--id={devices}',
            '--query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu',
            '--format=csv', '-lms', '200'], stdout=log, stderr=subprocess.STDOUT)
        try:
            yield
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if Path(path).stat().st_size == 0 or 'timestamp' not in Path(path).read_text()[:500]:
            raise RuntimeError(f'GPU telemetry was not collected: {path}')


def run_batch(argv, env, manifest, timeout):
    """Watch candidate progress, not a scaled whole-batch timeout."""
    progress = Path(manifest).with_suffix(".progress.json")
    last_progress = None
    deadline = time.monotonic() + timeout
    with subprocess.Popen(argv, cwd=ROOT, env=env, start_new_session=True) as process:
        try:
            while True:
                try:
                    rc = process.wait(timeout=min(1, timeout))
                    if rc:
                        raise subprocess.CalledProcessError(rc, argv)
                    return
                except subprocess.TimeoutExpired:
                    token = progress.read_text() if progress.exists() else None
                    if token != last_progress:
                        last_progress = token
                        deadline = time.monotonic() + timeout
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"No candidate progress for {timeout}s: {manifest}")
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise


def prepare_noise_retry(group, attempt_started_ns):
    """Only unanimous, newly recorded sample-drift failures may be retried.

    A persisted budget permits at most two fresh-process retries per candidate.
    Correctness, OOM, transport, timeout and incomplete-rank failures never qualify.
    """
    remaining = []
    for job in group:
        try:
            bench.read_measurement(Path(job['output']), job)
        except FileNotFoundError:
            remaining.append(job)
    if not remaining:
        return None
    job = remaining[0]
    output = Path(job['output'])
    failed = [output.with_suffix(f'.failed-rank{rank}.json') for rank in range(job['case']['cp'])]
    kinds = set()
    for path in failed:
        if not path.exists() or path.stat().st_mtime_ns < attempt_started_ns:
            return None
        data = json.loads(path.read_text())
        records = data.get('measurement_records', [])
        error = data.get('error', '')
        warmup_prefix = 'warmup did not converge within 5s: last windows '
        if error.startswith(warmup_prefix):
            try:
                windows = ast.literal_eval(error[len(warmup_prefix):])
                if len(windows) != 3 or not all(math.isfinite(x) and x > 0 for x in windows):
                    return None
            except (SyntaxError, ValueError, TypeError):
                return None
            kinds.add('warmup_convergence')
        elif error.startswith('measurement drift exceeds 5% in all 3 rounds:'):
            if not records or not records[-1].get('converged_all_ranks'):
                return None
            rounds = records[-1].get('measurement_rounds', [])
            if len(rounds) != 3 or not all(math.isfinite(r['half_p50_relative_drift'])
                                           and r['half_p50_relative_drift'] > .05 for r in rounds):
                return None
            kinds.add('sample_drift')
        else:
            return None
    if len(kinds) != 1:
        return None
    journal = output.with_suffix('.noise-retries.json')
    history = json.loads(journal.read_text()) if journal.exists() else []
    if len(history) >= 2:
        return None
    stamp = time.time_ns()
    archives = [p.with_name(p.stem + f'.noise-{stamp}.json') for p in failed]
    history.append({'reason': 'all ranks rejected ' + next(iter(kinds)),
                    'fresh_process_retry': len(history)+1, 'failed_samples': list(map(str, archives))})
    atomic_json(journal, history)
    for src, dst in zip(failed, archives):
        src.rename(dst)
    print(f"RETRY statistical-noise {len(history)}/2 {job['group']} "
          f"remaining={len(remaining)}; thresholds unchanged; raw failure records preserved", flush=True)
    return remaining


def run_batch_with_noise_recovery(argv, env, manifest, timeout, group):
    current = group
    current_argv = argv
    current_manifest = manifest
    while True:
        attempt_started_ns = time.time_ns()
        try:
            run_batch(current_argv, env, current_manifest, timeout)
            return
        except subprocess.CalledProcessError:
            remaining = prepare_noise_retry(current, attempt_started_ns)
            if remaining is None:
                raise
            current = remaining
            current_manifest = manifest.with_name(manifest.stem + f'.retry-{time.time_ns()}.json')
            atomic_json(current_manifest, current)
            current_argv = argv[:-1] + [str(current_manifest)]


def execute(plan_path, timeout, serial=False):
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text())
    pending = []
    reused = 0
    for job in plan["jobs"]:
        output = Path(job["output"])
        if output.exists():
            try:
                bench.read_measurement(output, job)
            except FileNotFoundError:
                # Interrupted output is preserved, never accepted as complete.
                stamp = time.time_ns()
                for path in [output, *output.parent.glob(output.stem + ".rank*.json")]:
                    path.rename(path.with_name(path.name + f".interrupted-{stamp}"))
            else:
                reused += 1
                continue
        pending.append(job)
    if not pending:
        print(f"All {reused} results validated; no execution", flush=True)
        return
    canonical = {}
    aliases = {}
    pending_outputs = {j['output'] for j in pending}
    completed = {measurement_key(j): j for j in plan['jobs'] if j['output'] not in pending_outputs}
    for job in pending:
        key = measurement_key(job)
        if key in completed:
            copy_measurement(completed[key], job, {'reason': 'validated equivalent geometry on resume',
                                                   'fingerprint': plan['fingerprint']})
            reused += 1
            continue
        if key in canonical:
            aliases.setdefault(canonical[key]['output'], []).append(job)
        else:
            canonical[key] = job
    pending = list(canonical.values())
    inherited = [key for key in os.environ if key.startswith("NCCL_")]
    if inherited:
        raise ValueError(f"unset inherited NCCL overrides: {inherited}")
    devices = ",".join(dict.fromkeys(d for j in pending for d in j["env"]["CUDA_VISIBLE_DEVICES"].split(",")))
    atomic_json(plan_path.with_suffix(".gpu-before.json"), bench.observe_devices(devices))
    groups = [[job] for job in pending] if serial else batches(pending)
    receipt = {"executor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "measurement_fingerprint": plan["fingerprint"], "reused": reused,
               "pending": len(pending), "process_groups": len(groups), "serial": serial,
               "geometry_aliases": sum(map(len, aliases.values())),
               "batches": []}
    print(f"EXECUTOR reused={reused} pending={len(pending)} process_groups={len(groups)}", flush=True)
    for index, group in enumerate(groups, 1):
        manifest = plan_path.parent / "batches" / f"{plan['stage']}-{time.time_ns()}.json"
        atomic_json(manifest, group)
        first = group[0]
        if first['env'].get('FUSE_CUBLASLT_CACHE_DIR'):
            Path(first['env']['FUSE_CUBLASLT_CACHE_DIR']).mkdir(parents=True, exist_ok=True)
        argv = first["command"] if serial else [
            first["command"][0], "-m", "torch.distributed.run", "--standalone",
            f"--nproc-per-node={first['case']['cp']}", str(Path(__file__).resolve()),
            "--worker", str(manifest)]
        started = time.monotonic()
        print(f"BATCH {index}/{len(groups)} candidates={len(group)} {first['group']}", flush=True)
        try:
            with telemetry(first['env']['CUDA_VISIBLE_DEVICES'], manifest.with_suffix('.gpu.csv')):
                if serial:
                    bench.run_job(argv, os.environ | first["env"], sys.stdout, timeout)
                else:
                    run_batch_with_noise_recovery(argv, os.environ | first["env"], manifest, timeout, group)
            for job in group:
                bench.read_measurement(Path(job["output"]), job)
                for alias in aliases.get(job['output'], []):
                    copy_measurement(job, alias, {'reason': 'identical geometry and measurement arguments',
                                                 'fingerprint': plan['fingerprint']})
        finally:
            receipt["batches"].append({"manifest": str(manifest), "candidates": len(group),
                                       "wall_s": time.monotonic() - started})
            atomic_json(plan_path.with_suffix(".executor.json"), receipt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--import-plan", type=Path)
    parser.add_argument("--job-timeout", type=float, default=600)
    args, rest = parser.parse_known_args()
    if args.worker:
        worker(args.worker)
    elif args.plan:
        if args.import_plan:
            import_results(args.plan, args.import_plan)
        execute(args.plan, args.job_timeout, args.serial)
    else:
        rest = [arg for arg in rest if arg != "--execute"]
        sys.argv = [str(ROOT / "benchmarks/sm103/bench.py"), *rest,
                    "--job-timeout", str(args.job_timeout)]
        bench.main()
        stage = rest[rest.index("--stage") + 1]
        if stage != "summary":
            results = Path(rest[rest.index("--results") + 1])
            if args.import_plan:
                import_results(results / f"{stage}_plan.json", args.import_plan)
            execute(results / f"{stage}_plan.json", args.job_timeout, args.serial)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, bench.handle_termination)
    main()
