#!/usr/bin/env python3
"""SM103 BF16 baseline plans, staged tuning and top-3 formal remeasurement.

Planning needs only Python's standard library. GPU execution is explicit.
"""
from __future__ import annotations

import argparse
from collections import deque
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
_projection_spec = importlib.util.spec_from_file_location(
    'sm103_projection_shapes', HERE / 'projection_shapes.py')
_projection_shapes = importlib.util.module_from_spec(_projection_spec)
_projection_spec.loader.exec_module(_projection_shapes)
BOUNDARY_MODELS = _projection_shapes.BOUNDARY_MODELS
DIRECTORIES = {"qkv": "sm90/QKVproj+a2a", "oproj": "sm90/a2a+Oproj"}
METRICS = {
    ("qkv", "cublaslt_nccl"): "cublaslt_packed_qkv_gemm_a2a",
    ("oproj", "cublaslt_nccl"): "cublaslt_nccl_oproj_boundary",
    ("qkv", "te_ub"): "te_userbuffers_qkv_boundary",
    ("oproj", "te_ub"): "te_userbuffers_oproj_boundary",
}
OPROJ_LAYOUTS = ('legacy', 'causal_dual_chunk_v1')
# Production sweeps exclude 1K/4K. Explicit small smoke/regression inputs and
# historical result readers remain supported, but are not production cases.
PRODUCTION_SEQUENCES = (16384, 131072, 262144, 524288)


def load_shapes(direction):
    filename = "qkv_shape_bench.py" if direction == "qkv" else "oproj_shape_bench.py"
    spec = importlib.util.spec_from_file_location(
        f"sm103_{direction}_shapes", ROOT / "benchmarks" / DIRECTORIES[direction] / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Extend SM103 only; preserve the historical default matrix and SM90 files.
    for name, (hidden, q_heads, kv_heads, head_dim) in BOUNDARY_MODELS.items():
        fields = ((hidden, q_heads, kv_heads, head_dim) if direction == 'qkv'
                  else (hidden, q_heads, head_dim))
        module.MODELS[name] = module.Model(
            name, 'model', *fields, 'Pinned projection geometry; see projection_shapes.py')
    return module


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def ints(value):
    values = tuple(int(x) for x in value.split(","))
    if not values or any(x <= 0 for x in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def cases(args):
    for direction in args.directions.split(","):
        shapes = load_shapes(direction)
        models = args.models.split(",") if args.models else shapes.DEFAULT_MODELS
        for name, seq, cp in itertools.product(models, args.seqs, args.cps):
            model = shapes.MODELS[name]
            if seq % (2*cp) or model.q_heads % cp or (
                    direction == "qkv" and model.kv_heads % cp):
                raise ValueError(f"invalid causal Ulysses geometry: {direction}/{name}/{seq}/{cp}")
            if cp > len(args.devices.split(",")):
                raise ValueError(f"CP{cp} exceeds supplied device list")
            n = model.qkv_width if direction == "qkv" else model.hidden
            k = model.hidden if direction == "qkv" else model.attention_width
            yield {"direction": direction, "model": name, "seq": seq, "cp": cp,
                   "hidden": model.hidden, "q_heads": model.q_heads,
                   "kv_heads": (BOUNDARY_MODELS[name][2] if name in BOUNDARY_MODELS
                                else getattr(model, "kv_heads", 8)), "head_dim": model.head_dim,
                   "m": seq//cp, "n": n, "k": k}


def initial_configs(backend, smoke=False):
    if backend == "cublaslt_nccl":
        tuples = [(16, 256, 64)] if smoke else itertools.product(
            (8, 16, 24, 32), (128, 256, 512, 1024), (16, 64, 128))
        return [{"channels": c, "chunk_kib": k, "ll_kib": l,
                 "pack_block": 512, "pack_warps": 4, "high_priority": False}
                for c, k, l in tuples]
    return [{"comm_sm": sm, "streams": 1, "push": True, "use_ce": False,
             "pack_block": 512, "pack_warps": 4, "reverse": False,
             "local_first": False} for sm in ((4,) if smoke else (4,8,12,16,20,24))]


def refine(config, backend, direction):
    if backend == "cublaslt_nccl":
        return [config | {"pack_block": b, "pack_warps": w, "high_priority": p}
                for b, w, p in itertools.product((128,256,512,1024), (4,8), (False,True))]
    changes = [{}, {"streams": 3}, {"push": False}, {"use_ce": True},
               {"pack_block": 256}, {"pack_block": 1024}, {"pack_warps": 8},
               {"reverse": True}]
    if direction == "qkv":
        changes.append({"local_first": True})
    return [config | change for change in changes]


def group_key(case, backend, launch):
    return f"{case['direction']}_{case['model']}_s{case['seq']}_cp{case['cp']}_{backend}_{launch}"


def read_measurement(path, job):
    data = json.loads(path.read_text())
    layout = job.get('env', {}).get('FUSE_SM103_OPROJ_LAYOUT', 'legacy')
    if layout not in OPROJ_LAYOUTS:
        raise ValueError(f'unknown expected OProj layout: {layout!r}')
    if job['case']['direction'] == 'oproj' and data.get('oproj_layout', 'legacy') != layout:
        raise ValueError(f'OProj result layout differs from the plan: {path}')
    metric = METRICS[job["case"]["direction"], job["backend"]]
    samples = data["samples_ms"]
    if isinstance(samples, dict):
        samples = samples[metric]
    if len(samples) != job["iterations"] or any(not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError(f"invalid samples: {path}")
    if not data.get("correctness"):
        raise ValueError(f"missing correctness validation: {path}")
    for name, value in data["correctness"].items():
        values = value if isinstance(value, list) else [value]
        if any(not math.isfinite(x) or ("mismatch" in name and x != 0) for x in values):
            raise ValueError(f"failed correctness validation: {path}: {name}")
    if data["world_size"] != job["case"]["cp"]:
        raise ValueError(f"wrong world size: {path}")
    launch = data.get("launch", "graph" if data.get("cuda_graph") else "eager")
    if launch != job["launch"]:
        raise ValueError(f"wrong launch mode: {path}")
    for rank in range(job["case"]["cp"]):
        metadata = json.loads(path.with_suffix(f".rank{rank}.json").read_text())
        if metadata["device"]["compute_capability"] != "10.3":
            raise ValueError(f"non-SM103 result: {path}")
        if job['case']['direction'] == 'oproj' and metadata.get('oproj_layout', 'legacy') != layout:
            raise ValueError(f'OProj rank {rank} layout differs from the plan: {path}')
        if job.get('env', {}).get('FUSE_SM103_MEASUREMENT') == 'v2':
            inputs = metadata.get('input_statistics', [])
            records = metadata.get('measurement_records', [])
            if not inputs or not records:
                raise ValueError(f'missing random-input or measured-warmup evidence: {path}')
            for item in inputs:
                for tensor in item['tensors'].values():
                    if not 0 < tensor['nonzero_fraction'] <= 1 or not math.isfinite(tensor['sample_std']):
                        raise ValueError(f'invalid random-input evidence: {path}')
            for item in records:
                if (not item['converged_all_ranks'] or item['initial_warmup'] < 10
                        or item['iterations'] < 50
                        or item['additional_warmup_cuda_ms'] < item['minimum_warmup_cuda_ms']
                        or item.get('sample_half_p50_relative_drift', 1) > .05):
                    raise ValueError(f'insufficient warmup/samples: {path}')
    # Recompute statistics from per-sample max-rank measurements. No historical
    # aggregate or per-rank percentile is accepted as a measurement.
    ordered = sorted(samples)
    def percentile(q):
        p = (len(ordered)-1)*q
        lo, hi = math.floor(p), math.ceil(p)
        return ordered[lo] + (ordered[hi]-ordered[lo])*(p-lo)
    return {"p50_ms": percentile(.5), "p95_ms": percentile(.95)}


def prior_winners(args, group, stages, fingerprint):
    found = {}
    for stage in stages:
        plan = json.loads((args.results / f"{stage}_plan.json").read_text())
        if plan["fingerprint"] != fingerprint:
            raise ValueError("source or build library changed since tuning; use a fresh result directory")
        for job in plan["jobs"]:
            if job["group"] != group:
                continue
            path = Path(job["output"])
            if not path.exists():
                raise ValueError(f"incomplete {stage} stage: missing {path}")
            value = read_measurement(path, job)["p50_ms"]
            found[digest(job["config"])] = (value, job["config"])
    if not found:
        raise ValueError(f"no tuning data for {group}")
    return [config for _, config in sorted(found.values(), key=lambda item: item[0])[:3]]


def make_job(args, case, backend, launch, config):
    layout = getattr(args, 'oproj_layout', 'legacy')
    if layout not in OPROJ_LAYOUTS:
        raise ValueError(f'unknown OProj layout: {layout!r}')
    group = group_key(case, backend, launch)
    output = (args.results / args.stage / f"{group}_{digest(config)}.json").resolve()
    warmup, iterations = (10,50)
    flags = ["--global-seq", str(case["seq"]), "--hidden", str(case["hidden"]),
             "--q-heads", str(case["q_heads"]), "--head-dim", str(case["head_dim"]),
             "--batch", "1", "--warmup", str(warmup), "--iters", str(iterations),
             "--check", "--cuda-graph" if launch == "graph" else "--no-cuda-graph",
             "--pack-block", str(config["pack_block"]), "--pack-warps", str(config["pack_warps"]),
             "--cublaslt-library", str(args.library.resolve()), "--json-out", str(output)]
    if case["direction"] == "qkv":
        flags += ["--kv-heads", str(case["kv_heads"])]
    env = {"CUDA_VISIBLE_DEVICES": ",".join(args.devices.split(",")[:case["cp"]]),
           "OMP_NUM_THREADS": "1", "FUSE_SM103_MEASUREMENT": "v2",
           "FUSE_SM103_OPROJ_LAYOUT": layout if case['direction'] == 'oproj' else 'legacy',
           "FUSE_CUBLASLT_TUNE_GRAPH": "1" if launch == "graph" else "0"}
    if getattr(args, 'cache_namespace', None):
        env['FUSE_CUBLASLT_CACHE_DIR'] = str(ROOT / 'results/sm103/.gemm-cache' / args.cache_namespace)
    if backend == "cublaslt_nccl":
        flags += ["--mode", "qkv_gemm_a2a" if case["direction"] == "qkv" else "oproj_a2a_gemm",
                  "--metric-profile", "boundary", "--no-include-source", "--no-include-te",
                  "--cublaslt-tune-warmup", "10", "--cublaslt-tune-iters", "50",
                  "--cublaslt-workspace-mib", "64",
                  "--nccl-high-priority" if config["high_priority"] else "--no-nccl-high-priority"]
        env |= {"NCCL_MIN_P2P_NCHANNELS": str(config["channels"]),
                "NCCL_MAX_P2P_NCHANNELS": str(config["channels"]),
                "NCCL_P2P_NVL_CHUNKSIZE": str(config["chunk_kib"]*1024),
                "NCCL_P2P_LL_THRESHOLD": str(config["ll_kib"]*1024),
                "NCCL_IB_DISABLE": "1", "NCCL_GRAPH_REGISTER": "1", "NCCL_LOCAL_REGISTER": "1"}
    else:
        if not 0 < config["comm_sm"] < args.sm_count:
            raise ValueError("communication SM reservation must be below device SM count")
        if config["comm_sm"] > 32:
            raise ValueError("native TE P2P communication SM budget exceeds UB_MAX_SM=32")
        flags += ["--num-comm-sm", str(config["comm_sm"]), "--math-sm", str(args.sm_count-config["comm_sm"]),
                  "--num-streams", str(config["streams"]),
                  "--parallel-sends" if config["streams"] > 1 else "--no-parallel-sends",
                  "--tune-warmup", "10", "--tune-iters", "50", "--workspace-mib", "64"]
        for key in ("push", "use_ce", "reverse", "local_first"):
            if key == "local_first" and case["direction"] != "qkv":
                continue
            flags += ["--" + ("" if config[key] else "no-") + key.replace("_", "-")]
    command = [args.python, "-m", "torch.distributed.run", "--standalone",
               f"--nproc-per-node={case['cp']}", str(HERE / "worker.py"),
               "--direction", case["direction"], "--backend", backend,
               "--expected-sms", str(args.sm_count), *flags]
    return {"group": group, "case": case, "backend": backend, "launch": launch,
            "config": config, "warmup": warmup, "iterations": iterations,
            "env": env, "command": command, "output": str(output)}


def fingerprint(args):
    files = list(HERE.glob("*.py")) + [ROOT / "csrc/baselines/cublaslt_runner.cu"]
    for folder in DIRECTORIES.values():
        files += list((ROOT / "benchmarks" / folder).glob("*.py"))
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(files)}
    hashes["library"] = hashlib.sha256(args.library.read_bytes()).hexdigest() if args.library.exists() else "not-built"
    inputs = {"files": hashes, "sm_count": args.sm_count, "devices": args.devices,
              "python": args.python, "te_root": str(args.te_root),
              "oproj_layout": getattr(args, 'oproj_layout', 'legacy')}
    if os.environ.get("FUSE_ENV_FINGERPRINT"):
        inputs["environment"] = os.environ["FUSE_ENV_FINGERPRINT"]
    return digest(inputs)


def observe_devices(visible_devices, samples=3, check_idle=True):
    observations = []
    for index in range(samples):
        result = subprocess.run(
            ["nvidia-smi", f"--id={visible_devices}",
             "--query-gpu=index,uuid,utilization.gpu,memory.free,memory.used,clocks.sm,clocks.mem,power.draw",
             "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
        rows = list(csv.reader(result.stdout.splitlines()))
        if len(rows) != len(visible_devices.split(",")):
            raise ValueError("GPU observation did not cover every selected device")
        for row in rows:
            if check_idle and float(row[2]) > 5:
                raise ValueError(f"GPU {row[0]} is computing ({row[2]}%); no processes were stopped")
            if float(row[3]) < 2048:
                raise ValueError(f"GPU {row[0]} has less than 2 GiB free")
        observations.append(rows)
        if index < samples - 1:
            time.sleep(1)
    return observations


def run_job(command, env, log, timeout):
    # Give only this torchrun and its children a new process group, so timeout
    # cleanup cannot signal resident models or another benchmark session.
    with subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                          stderr=subprocess.STDOUT, start_new_session=True) as process:
        try:
            returncode = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            # torchrun can exit before a stuck rank. Check and kill the owned
            # group even when its leader was already reaped by wait().
            try:
                os.killpg(process.pid, 0)
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)


def report_job_failure(job, error, timeout):
    output = Path(job["output"])
    log_path = output.with_suffix(".log")
    failure = {"group": job["group"], "config": job["config"],
               "command": job["command"], "log": str(log_path),
               "error": str(error), "job_timeout_seconds": timeout,
               "returncode": getattr(error, "returncode", None),
               "timed_out": isinstance(error, subprocess.TimeoutExpired)}
    output.with_suffix(".failed.json").write_text(json.dumps(failure, indent=2) + "\n")
    print(f"FAILED {job['group']}: {error}\nJob log: {log_path}", file=sys.stderr, flush=True)
    if log_path.exists():
        with log_path.open(errors="replace") as log:
            tail = "".join(deque(log, maxlen=30))
        if tail:
            print(tail, end="" if tail.endswith("\n") else "\n", file=sys.stderr, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--precision", choices=("bf16",), default="bf16",
                   help="only BF16 is implemented here; FP8/FP4 references: PRECISIONS.md")
    p.add_argument("--stage", choices=("smoke","sweep","refine","formal","summary"), default="smoke")
    p.add_argument("--execute", action="store_true", help="otherwise only write a reviewable plan")
    p.add_argument("--directions", default="qkv,oproj")
    p.add_argument('--oproj-layout', choices=OPROJ_LAYOUTS, default='legacy',
                   help='explicit OProj input routing; canonical mode keeps the complete timed boundary')
    p.add_argument("--backends", default="cublaslt_nccl,te_ub")
    p.add_argument("--launches", default="eager,graph",
                   help="comma-separated launch modes; use graph for a shorter first pass")
    p.add_argument("--job-timeout", type=float, default=600,
                   help="maximum seconds per torchrun job (default: 600)")
    p.add_argument("--models", default="")
    p.add_argument("--seqs", type=ints, default=PRODUCTION_SEQUENCES,
                   help="production default: 16K,128K,256K,512K; small sizes are explicit diagnostics")
    p.add_argument("--cps", type=ints, default=(4,8))
    p.add_argument("--smoke-cp", type=int, choices=(2,4,8), default=8,
                   help="world size for the smoke stage; final SM103 validation defaults to CP8")
    p.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    p.add_argument("--sm-count", type=int, default=148)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--te-root", type=Path)
    p.add_argument("--library", type=Path, default=ROOT / "build/sm103/libfuse_cublaslt_runner.so")
    p.add_argument("--results", type=Path, default=ROOT / "results/sm103/baselines")
    args = p.parse_args()
    if len(set(args.devices.split(","))) != len(args.devices.split(",")):
        p.error("devices must be unique")
    if not set(args.directions.split(",")) <= set(DIRECTORIES):
        p.error("directions must be qkv,oproj")
    if not set(args.backends.split(",")) <= {"cublaslt_nccl", "te_ub"}:
        p.error("unknown baseline")
    launches = args.launches.split(",")
    if len(set(launches)) != len(launches) or not set(launches) <= {"eager", "graph"}:
        p.error("launches must contain eager, graph, or eager,graph without duplicates")
    if not math.isfinite(args.job_timeout) or args.job_timeout <= 0:
        p.error("job-timeout must be a finite positive number")
    if args.stage == "smoke":
        args.models, args.seqs, args.cps = "production_qwen_dense", (1024,), (args.smoke_cp,)
    fp = fingerprint(args)
    args.cache_namespace = fp
    args.results.mkdir(parents=True, exist_ok=True)
    if args.stage == "summary":
        plan = json.loads((args.results / "formal_plan.json").read_text())
        if plan["fingerprint"] != fp:
            raise ValueError("formal plan belongs to a different source/build/environment")
        rows = []
        for job in plan["jobs"]:
            stats = read_measurement(Path(job["output"]), job)
            rows.append({"group": job["group"], **stats, "raw": job["output"],
                         "tflops_per_gpu_p50": 2*job["case"]["m"]*job["case"]["n"]*job["case"]["k"]/stats["p50_ms"]/1e9})
        winners = {}
        for row in rows:
            if row["p50_ms"] < winners.get(row["group"], {"p50_ms": math.inf})["p50_ms"]:
                winners[row["group"]] = row
        with (args.results / "summary.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=("group","p50_ms","p95_ms","raw","tflops_per_gpu_p50"))
            writer.writeheader(); writer.writerows(winners.values())
        print(f"Wrote {len(winners)} baseline winners; no fused results exist yet")
        return
    jobs = []
    for case, backend, launch in itertools.product(cases(args), args.backends.split(","), launches):
        group = group_key(case, backend, launch)
        if args.stage in ("smoke","sweep"):
            configs = initial_configs(backend, args.stage == "smoke")
        elif args.stage == "refine":
            configs = [c for winner in prior_winners(args, group, ("sweep",), fp)
                       for c in refine(winner, backend, case["direction"])]
        else:
            configs = prior_winners(args, group, ("sweep","refine"), fp)
        for config in {digest(c): c for c in configs}.values():
            jobs.append(make_job(args, case, backend, launch, config))
    plan = {"schema": "sm103_baseline_v1", "fingerprint": fp, "precision": args.precision,
            "stage": args.stage, "sm_count": args.sm_count, "jobs": jobs}
    plan_path = args.results / f"{args.stage}_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("existing plan differs; choose a new --results directory")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"{len(jobs)} jobs planned: {plan_path}", flush=True)
    if not args.execute:
        return
    if not args.library.is_file():
        raise ValueError("build the SM103 cuBLASLt library first")
    pending = []
    for job in jobs:
        output = Path(job["output"])
        if output.exists():
            read_measurement(output, job)
            continue
        pending.append(job)
    if not pending:
        print("All planned jobs already have validated results", flush=True)
        return
    base_env = os.environ.copy()
    # NCCL overrides affect process-group initialization; reject them once per
    # execution stage, before GPU observation or any torchrun is started.
    inherited = [key for key in base_env if key.startswith("NCCL_")]
    if inherited:
        raise ValueError(f"unset inherited NCCL overrides before tuning: {inherited}")
    devices = ",".join(dict.fromkeys(device for job in pending
                                   for device in job["env"]["CUDA_VISIBLE_DEVICES"].split(",")))
    observations = observe_devices(devices)
    (args.results / f"{args.stage}.gpu-before.json").write_text(json.dumps(observations, indent=2) + "\n")
    for index, job in enumerate(pending, 1):
        output = Path(job["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        env = base_env.copy()
        env.update(job["env"])
        if env.get('FUSE_CUBLASLT_CACHE_DIR'):
            Path(env['FUSE_CUBLASLT_CACHE_DIR']).mkdir(parents=True, exist_ok=True)
        if args.te_root:
            for key in ("PYTHONPATH", "LD_LIBRARY_PATH"):
                env[key] = str(args.te_root.resolve()) + os.pathsep + env.get(key, "")
        print(f"RUN {index}/{len(pending)} {job['group']} {digest(job['config'])}", flush=True)
        # A utilization sample here may still include the preceding job from
        # this stage. Record it, but only enforce the current free-memory check.
        observations = observe_devices(job["env"]["CUDA_VISIBLE_DEVICES"], samples=1, check_idle=False)
        output.with_suffix(".gpu-before.json").write_text(json.dumps(observations, indent=2) + "\n")
        try:
            with output.with_suffix(".log").open("w") as log:
                run_job(job["command"], env, log, args.job_timeout)
            read_measurement(output, job)
        except (subprocess.SubprocessError, OSError, ValueError, KeyError) as error:
            report_job_failure(job, error, args.job_timeout)
            raise SystemExit(1) from error
        print(f"DONE {index}/{len(pending)} {job['group']}", flush=True)


def handle_termination(signum, _frame):
    # The outer runner terminates the bench process group; each torchrun has
    # its own session and must be cleaned up through run_job's exception path.
    raise KeyboardInterrupt(f"received signal {signum}")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, handle_termination)
    main()
