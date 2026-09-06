"""Resumable finite tuning + formal full-matrix run, serialized GPU jobs.

NCCL environment candidates run in separate processes (no assumption that NCCL
re-reads process-global environment when a communicator is recreated). Shapes
share each process group. Formal selection is fixed by disjoint short samples.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from matrix import ROOT, full_matrix

PROFILES = {
    "auto": {},
    "ch24_chunk128_ll64": {"NCCL_MIN_P2P_NCHANNELS": "24", "NCCL_MAX_P2P_NCHANNELS": "24",
                          "NCCL_P2P_NVL_CHUNKSIZE": "131072", "NCCL_P2P_LL_THRESHOLD": "65536"},
    "ch32_chunk512_ll64": {"NCCL_MIN_P2P_NCHANNELS": "32", "NCCL_MAX_P2P_NCHANNELS": "32",
                          "NCCL_P2P_NVL_CHUNKSIZE": "524288", "NCCL_P2P_LL_THRESHOLD": "65536"},
}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/mxfp8_weight/full_v2")
    parser.add_argument("--cps", default="4,8")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    matrix = full_matrix()
    write(args.output / "matrix.json", matrix)
    cps = [int(x) for x in args.cps.split(",")]
    if args.dry_run:
        print(json.dumps(dict(settings=len(matrix), expected_rows=768, cps=cps,
                              nccl_candidates=PROFILES), indent=2))
        return
    started = time.monotonic()
    jobs = []

    def job(cp, profile, backend, phase, ids=None):
        tag = f"{backend}_{phase}_{profile}_cp{cp}"
        output = args.output / f"{tag}.json"
        env = {k: v for k, v in os.environ.items() if not k.startswith("NCCL_")}
        env.update(CUDA_VISIBLE_DEVICES=next(c["visible_devices"] for c in matrix if c["cp"] == cp),
                   OMP_NUM_THREADS="1", NCCL_IB_DISABLE="1", NCCL_GRAPH_REGISTER="1", NCCL_LOCAL_REGISTER="1")
        env.update(PROFILES[profile])
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                   f"--nproc-per-node={cp}", str(Path(__file__).with_name("bench.py")),
                   "--full", "--resume", "--backends", backend, "--output", str(output)]
        if phase == "sweep":
            command += ["--warmup", "3", "--iterations", "12"]
        if ids is not None:
            manifest = args.output / f"{tag}.ids.json"
            write(manifest, sorted(ids))
            command += ["--case-ids", str(manifest)]
        print(f"JOB {tag}; log={output.with_suffix('.log')}", flush=True)
        begin = time.monotonic()
        with output.with_suffix(".log").open("a") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        jobs.append(dict(tag=tag, command=command, seconds=time.monotonic()-begin,
                         output=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest()))
        write(args.output / "jobs.json", dict(jobs=jobs, wall_seconds=time.monotonic()-started))
        return output, json.loads(output.read_text())

    for cp in cps:
        # Both directions / all model labels remain in every selection sweep.
        surveys = {profile: job(cp, profile, "cublaslt_nccl", "sweep") for profile in PROFILES}
        by_key = {}
        for profile, (path, report) in surveys.items():
            for item in report["cases"]:
                for record in item["best_tested"]:
                    key = (item["case"]["id"], record["launch"])
                    by_key.setdefault(key, []).append((record["p50_us"], profile))
        policy = {key: min(values)[1] for key, values in by_key.items()}
        write(args.output / f"nccl_policy_cp{cp}.json", [dict(id=k[0], launch=k[1], profile=v)
                                                       for k, v in sorted(policy.items())])
        formal = {}
        for profile in PROFILES:
            ids = {key[0] for key, winner in policy.items() if winner == profile}
            if ids:
                formal[profile] = job(cp, profile, "cublaslt_nccl", "formal", ids)
        te_path, te = job(cp, "auto", "teub", "formal")
        combined = copy.deepcopy(te)
        combined["schema"] = "fuse-mxfp8-weight-suite-v2"
        combined["nccl_environment"] = "per-row source_run; selection fixed by independent sweep"
        combined["source_reports"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p, _ in [*formal.values(), *surveys.values(), (te_path, te)]}
        lookup = {(item["case"]["id"], row["launch"], profile): (row, path)
                  for profile, (path, report) in formal.items() for item in report["cases"]
                  for row in item["best_tested"]}
        for item in combined["cases"]:
            for row in item["records"] + item["best_tested"]:
                row["source_run"] = str(te_path)
            for launch in ("eager", "graph"):
                profile = policy[item["case"]["id"], launch]
                row, path = lookup[item["case"]["id"], launch, profile]
                selected = dict(row, source_run=str(path), nccl_profile=profile)
                item["records"].append(selected)
                item["best_tested"].append(selected)
        write(args.output / f"combined_cp{cp}.json", combined)
    files = [str(args.output / f"combined_cp{cp}.json") for cp in cps]
    command = [sys.executable, str(Path(__file__).with_name("audit.py")), *files,
               "--output", str(args.output / "coverage.json")]
    if set(cps) == {4, 8}:
        command.append("--require-full")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
