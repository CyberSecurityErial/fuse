#!/usr/bin/env python3
"""Measure the old/new GPU observation overhead, without launching GPU work.

This is not a kernel or end-to-end benchmark speedup measurement. Both variants
only run the same read-only nvidia-smi query and the historical one-second waits.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


FIELDS = (
    "index", "uuid", "utilization.gpu", "memory.free", "memory.used",
    "clocks.sm", "clocks.mem", "power.draw",
)
SCOPE = (
    "Read-only GPU observation and forced-wait overhead only; no GPU kernel, "
    "torchrun, compilation, transfer, or full benchmark speedup is measured. "
    "GPU utilization and free memory are recorded, not used to stop this experiment."
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, document):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    os.replace(temporary, path)


def observe(command, timeout, phase, job):
    started_at = utc_now()
    start = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    snapshot = {
        "phase": phase, "job": job, "started_at": started_at,
        "wall_seconds": time.perf_counter() - start,
        "returncode": completed.returncode,
        "stdout": completed.stdout, "stderr": completed.stderr,
        "gpus": [dict(zip(FIELDS, row)) for row in csv.reader(completed.stdout.splitlines())],
    }
    return snapshot


def run_variant(record, command, jobs, timeout):
    start = time.perf_counter()
    record["started_at"] = utc_now()
    groups = [("job", job, 3) for job in range(1, jobs + 1)] if record["variant"] == "old" else (
        [("stage", None, 3)] + [("job", job, 1) for job in range(1, jobs + 1)])
    try:
        for phase, job, samples in groups:
            for index in range(samples):
                snapshot = observe(command, timeout, phase, job)
                record["snapshots"].append(snapshot)
                record["query_count"] += 1
                if snapshot["returncode"]:
                    raise RuntimeError(
                        f"nvidia-smi exited {snapshot['returncode']}: {snapshot['stderr'].strip()}")
                if index + 1 < samples:
                    sleep_start = time.perf_counter()
                    time.sleep(1)
                    record["forced_sleep_seconds"] += 1
                    record["observed_sleep_seconds"] += time.perf_counter() - sleep_start
    finally:
        record["wall_seconds"] = time.perf_counter() - start
        record["finished_at"] = utc_now()
        record["query_wall_seconds"] = sum(item["wall_seconds"] for item in record["snapshots"])


def summarize(rounds):
    old = [item for item in rounds if item["variant"] == "old"]
    new = [item for item in rounds if item["variant"] == "new"]
    old_median = statistics.median(item["wall_seconds"] for item in old)
    new_median = statistics.median(item["wall_seconds"] for item in new)
    pairs = [
        {"repeat": old_item["repeat"],
         "old_seconds": old_item["wall_seconds"], "new_seconds": new_item["wall_seconds"],
         "speedup": old_item["wall_seconds"] / new_item["wall_seconds"],
         "seconds_saved": old_item["wall_seconds"] - new_item["wall_seconds"]}
        for old_item, new_item in zip(old, new)
    ]
    return {
        "scope": SCOPE,
        "median_old_seconds": old_median,
        "median_new_seconds": new_median,
        "median_speedup": old_median / new_median,
        "median_seconds_saved": old_median - new_median,
        "median_paired_speedup": statistics.median(item["speedup"] for item in pairs),
        "median_paired_seconds_saved": statistics.median(item["seconds_saved"] for item in pairs),
        "paired_results": pairs,
        "total_query_count": {"old": sum(item["query_count"] for item in old),
                              "new": sum(item["query_count"] for item in new)},
        "total_forced_sleep_seconds": {
            "old": sum(item["forced_sleep_seconds"] for item in old),
            "new": sum(item["forced_sleep_seconds"] for item in new)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--query-timeout", type=float, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.jobs < 1 or args.repeats < 1:
        parser.error("jobs and repeats must be positive")
    if not 0 < args.query_timeout < float("inf"):
        parser.error("query-timeout must be finite and positive")
    devices = [device.strip() for device in args.devices.split(",")]
    if not all(devices) or len(set(devices)) != len(devices):
        parser.error("devices must be nonempty and unique")
    if args.output.exists():
        parser.error("output already exists; choose a new result path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = ["nvidia-smi", f"--id={','.join(devices)}",
               f"--query-gpu={','.join(FIELDS)}", "--format=csv,noheader,nounits"]
    document = {
        "schema": "fuse_l20d_workflow_overhead_v1", "status": "running", "scope": SCOPE,
        "started_at": utc_now(), "hostname": platform.node(), "python": sys.version,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "jobs": args.jobs, "repeats": args.repeats, "devices": devices,
        "command": command, "query_timeout_seconds": args.query_timeout,
        "order": "alternating paired AB, BA, AB; A=old, B=new",
        "expected_per_round": {
            "old": {"query_count": 3 * args.jobs, "forced_sleep_seconds": 2 * args.jobs},
            "new": {"query_count": 3 + args.jobs, "forced_sleep_seconds": 2}},
        "rounds": [],
    }
    write_json(args.output, document)
    try:
        for repeat in range(1, args.repeats + 1):
            order = ("old", "new") if repeat % 2 else ("new", "old")
            for variant in order:
                record = {"repeat": repeat, "variant": variant, "query_count": 0,
                          "forced_sleep_seconds": 0, "observed_sleep_seconds": 0,
                          "snapshots": []}
                document["rounds"].append(record)
                print(f"START repeat={repeat}/{args.repeats} variant={variant} jobs={args.jobs}", flush=True)
                run_variant(record, command, args.jobs, args.query_timeout)
                write_json(args.output, document)
                print(f"DONE repeat={repeat}/{args.repeats} variant={variant} "
                      f"wall={record['wall_seconds']:.3f}s queries={record['query_count']} "
                      f"forced_sleep={record['forced_sleep_seconds']}s", flush=True)
        document["summary"] = summarize(document["rounds"])
        document["status"] = "complete"
    except (Exception, KeyboardInterrupt) as error:
        document["status"] = "failed"
        document["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        document["finished_at"] = utc_now()
        write_json(args.output, document)
    summary = document["summary"]
    print(f"OBSERVATION-OVERHEAD old={summary['median_old_seconds']:.3f}s "
          f"new={summary['median_new_seconds']:.3f}s speedup={summary['median_speedup']:.2f}x "
          f"saved={summary['median_seconds_saved']:.3f}s output={args.output}; "
          "not a kernel or end-to-end benchmark speedup", flush=True)


if __name__ == "__main__":
    main()
