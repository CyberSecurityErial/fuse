#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def percentile(samples: list[float], q: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = [json.loads(path.read_text()) for path in args.runs]
    immutable_keys = (
        "mode",
        "model_shape",
        "gemm_shapes",
        "world_size",
        "warmup",
        "iterations",
        "dtype",
        "timing",
        "scope",
        "include_te",
        "include_source",
        "cuda_graph",
        "pack_backend",
        "pack_block",
        "pack_warps",
        "nccl_high_priority",
        "software",
        "devices",
        "environment",
        "implementations",
    )
    reference = {key: runs[0][key] for key in immutable_keys}
    for index, run in enumerate(runs[1:], start=2):
        candidate = {key: run[key] for key in immutable_keys}
        if candidate != reference:
            raise ValueError(f"run {index} does not match the frozen configuration")
    if any(any(value != 0 for value in run["correctness"].values()) for run in runs):
        raise ValueError("a run contains correctness mismatches")

    metric_names = list(runs[0]["samples_ms"])
    if any(list(run["samples_ms"]) != metric_names for run in runs[1:]):
        raise ValueError("metric sets or ordering differ")
    metrics: dict[str, object] = {}
    for name in metric_names:
        per_run = [run["results"][name] for run in runs]
        pooled = [sample for run in runs for sample in run["samples_ms"][name]]
        metrics[name] = {
            "median_of_run_mean_ms": statistics.median(
                entry["mean_ms"] for entry in per_run
            ),
            "median_of_run_p50_ms": statistics.median(
                entry["p50_ms"] for entry in per_run
            ),
            "median_of_run_p95_ms": statistics.median(
                entry["p95_ms"] for entry in per_run
            ),
            "pooled_mean_ms": statistics.fmean(pooled),
            "pooled_p50_ms": percentile(pooled, 0.50),
            "pooled_p95_ms": percentile(pooled, 0.95),
            "pooled_min_ms": min(pooled),
            "pooled_max_ms": max(pooled),
            "run_mean_ms": [entry["mean_ms"] for entry in per_run],
            "run_p50_ms": [entry["p50_ms"] for entry in per_run],
            "run_p95_ms": [entry["p95_ms"] for entry in per_run],
        }

    encoded_config = json.dumps(reference, sort_keys=True, separators=(",", ":"))
    output = {
        "schema": "frozen_te_cublas_nccl_baseline_v1",
        "source_files": [str(path) for path in args.runs],
        "independent_runs": len(runs),
        "samples_per_metric": sum(run["iterations"] for run in runs),
        "configuration_sha256": hashlib.sha256(encoded_config.encode()).hexdigest(),
        "configuration": reference,
        "correctness": runs[0]["correctness"],
        "metrics": metrics,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
