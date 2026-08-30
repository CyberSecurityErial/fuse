#!/usr/bin/env python3
"""Pure-E4M3 A2A->OProj Eager/Graph matrix.

This reuses the established OProj model/shape registry but writes an isolated
V12 result tree.  It never reads or overwrites the BF16 launch manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OPROJ_DIR = ROOT / "benchmarks" / "a2a+Oproj"
sys.path.insert(0, str(OPROJ_DIR))
import oproj_shape_bench as shapes  # noqa: E402


def csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item for item in value.split(",") if item)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench", type=Path, default=ROOT / "build" / "fuse_bench"
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--models", default=",".join(shapes.DEFAULT_MODELS))
    parser.add_argument("--seqs", type=csv_ints, default=shapes.SEQUENCES)
    parser.add_argument("--cps", type=csv_ints, default=shapes.CONTEXT_PARALLEL)
    parser.add_argument(
        "--launches", type=csv_strings, default=("eager", "graph")
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def selected_models(value: str):
    names = tuple(item for item in value.split(",") if item)
    unknown = sorted(set(names) - shapes.MODELS.keys())
    if unknown:
        raise ValueError(f"unknown models: {unknown}")
    return tuple(shapes.MODELS[name] for name in names)


def cases(args: argparse.Namespace):
    for model in selected_models(args.models):
        for seq in args.seqs:
            for cp in args.cps:
                if cp in shapes.VISIBLE_DEVICES and seq % cp == 0 \
                        and model.q_heads % cp == 0:
                    yield model, seq, cp


def key(model, seq: int, cp: int) -> str:
    return f"{model.name}_s{seq}_cp{cp}"


def path(args, model, seq: int, cp: int, launch: str) -> Path:
    return args.results / "formal" / launch / (
        f"{key(model, seq, cp)}_{args.warmup}w{args.iterations}i.json"
    )


def valid(args, output: Path, model, seq: int, cp: int, launch: str) -> bool:
    if not output.exists():
        return False
    try:
        data = json.loads(output.read_text())
        fused = data["results"]["fused"]
        return (
            data["mode"] == "a2a_gemm_fp8"
            and data.get("dtype") == "e4m3xe4m3_fp32acc_e4m3out"
            and data.get("fp8_pipeline") in {"pingpong", "cooperative"}
            and data["world_size"] == cp
            and data["warmup"] == args.warmup
            and data["iterations"] == args.iterations
            and data["fused_cuda_graph"] == (launch == "graph")
            and data.get("cuda_graph_preuploaded") == (launch == "graph")
            and data["shape"] == {
                "m": seq // cp,
                "n": model.hidden,
                "k": model.attention_width,
                "l": 1,
            }
            and isinstance(data.get("comm_ctas"), int)
            and data["comm_ctas"] > 0
            and data.get("correctness", {}).get("exact_mismatches") == 0
            and all(
                isinstance(fused[name], (int, float))
                and math.isfinite(fused[name])
                and fused[name] > 0
                for name in ("p50_ms", "p95_ms")
            )
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def command(args, output: Path, model, seq: int, cp: int, launch: str):
    cmd = [
        str(args.bench),
        "--mode", "a2a_gemm_fp8",
        "--m", str(seq // cp),
        "--n", str(model.hidden),
        "--k", str(model.attention_width),
        "--batch", "1",
        "--q-heads", str(model.q_heads),
        "--head-dim", str(model.head_dim),
        "--comm-ctas", "0",
        "--lhs-policy", "m128n128",
        "--raster", "n",
        "--swizzle", "1",
        "--warmup", str(args.warmup),
        "--iterations", str(args.iterations),
        "--json-out", str(output.resolve()),
    ]
    if launch == "graph":
        cmd.append("--cuda-graph")
    return cmd


def run_formal(args: argparse.Namespace) -> None:
    invalid = sorted(set(args.launches) - {"eager", "graph"})
    if invalid:
        raise ValueError(f"invalid launches: {invalid}")
    for model, seq, cp in cases(args):
        for launch in args.launches:
            output = path(args, model, seq, cp, launch)
            if args.resume and valid(args, output, model, seq, cp, launch):
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = shapes.VISIBLE_DEVICES[cp]
            cmd = command(args, output, model, seq, cp, launch)
            print("RUN", " ".join(cmd), flush=True)
            completed = subprocess.run(
                cmd, cwd=ROOT, env=env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            if completed.returncode != 0:
                print(completed.stdout, end="")
                print(completed.stderr, end="")
                output.unlink(missing_ok=True)
                completed.check_returncode()
            if not valid(args, output, model, seq, cp, launch):
                output.unlink(missing_ok=True)
                raise RuntimeError(f"invalid FP8 OProj result: {output}")


def write_summary(args: argparse.Namespace) -> None:
    rows = []
    for model, seq, cp in cases(args):
        eager = json.loads(path(args, model, seq, cp, "eager").read_text())
        graph = json.loads(path(args, model, seq, cp, "graph").read_text())
        if eager["shape"] != graph["shape"] or \
                eager["comm_ctas"] != graph["comm_ctas"] or \
                eager.get("fp8_pipeline") != graph.get("fp8_pipeline"):
            raise ValueError(f"Eager/Graph mismatch: {key(model, seq, cp)}")
        shape = eager["shape"]
        flops = 2.0 * shape["m"] * shape["n"] * shape["k"]
        e = eager["results"]["fused"]
        g = graph["results"]["fused"]
        rows.append({
            "model": model.name,
            "suite": model.suite,
            "aliases": model.aliases,
            "precision": "fp8_e4m3",
            "fp8_pipeline": eager.get("fp8_pipeline"),
            "global_seq": seq,
            "cp": cp,
            "m": shape["m"], "n": shape["n"], "k": shape["k"],
            "q_heads": model.q_heads, "head_dim": model.head_dim,
            "requested_comm_ctas": 0,
            "comm_ctas": eager["comm_ctas"],
            "policy": "m128n128",
            "warmup": args.warmup, "iterations": args.iterations,
            "process_model": "single_process_multi_gpu",
            "timing": "per_sample_max_rank_cuda_event",
            "eager_p50_ms": e["p50_ms"], "eager_p95_ms": e["p95_ms"],
            "eager_p50_tflops_per_gpu": flops / e["p50_ms"] / 1.0e9,
            "graph_p50_ms": g["p50_ms"], "graph_p95_ms": g["p95_ms"],
            "graph_p50_tflops_per_gpu": flops / g["p50_ms"] / 1.0e9,
            "exact_mismatches_eager": 0,
            "exact_mismatches_graph": 0,
        })
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with (args.results / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    run_formal(args)
    write_summary(args)


if __name__ == "__main__":
    main()
