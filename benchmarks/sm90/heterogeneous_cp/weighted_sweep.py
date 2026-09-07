#!/usr/bin/env python3
"""Sweep weighted CP over CP degree, slow-rank count, and resource ratios."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time


RESULT_RE = re.compile(r"^RESULT (?P<body>.*)$")
SPEEDUP_RE = re.compile(r"^SPEEDUP (?P<body>.*)$")
PLAN_RE = re.compile(r"^PLAN (?P<body>.*)$")
PLAN_RANK_RE = re.compile(r"^PLAN_RANK (?P<body>.*)$")


def csv_ints(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value]


def csv_floats(text: str) -> list[float]:
    return [float(value) for value in text.split(",") if value]


def fields(body: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in body.split():
        key, value = item.split("=", 1)
        result[key] = value
    return result


def parse_output(output: str) -> dict[str, object]:
    policies: dict[str, dict[str, object]] = {}
    speedups: dict[str, dict[str, object]] = {}
    plans: dict[str, dict[str, object]] = {}
    for line in output.splitlines():
        result_match = RESULT_RE.match(line)
        if result_match:
            raw = fields(result_match.group("body"))
            key = f"{raw['op']}:{raw['policy']}"
            policies[key] = {
                "implementation": raw["implementation"],
                "mean_ms": float(raw["mean_ms"]),
                "p50_ms": float(raw["p50_ms"]),
                "p95_ms": float(raw["p95_ms"]),
                "rows": csv_ints(raw["rows"]),
                "rank_p50_ms": csv_floats(raw["rank_p50"]),
            }
        speedup_match = SPEEDUP_RE.match(line)
        if speedup_match:
            raw = fields(speedup_match.group("body"))
            speedups[raw["op"]] = {
                "p50": float(raw["value"]),
                "comm_uniform": csv_ints(raw["comm_uniform"]),
                "comm_weighted": csv_ints(raw["comm_weighted"]),
            }
        plan_match = PLAN_RE.match(line)
        if plan_match:
            raw = fields(plan_match.group("body"))
            plans[raw["op"]] = {
                "weighted": bool(int(raw["weighted"])),
                "predicted_speedup": float(raw["predicted_speedup"]),
                "uniform_us": float(raw["uniform_us"]),
                "weighted_us": float(raw["weighted_us"]),
                "redistributed_rows": int(raw["redistributed_rows"]),
                "equivalent_alpha": float(raw["equivalent_alpha"]),
                "uniform_bottleneck": int(raw["uniform_bottleneck"]),
                "weighted_bottleneck": int(raw["weighted_bottleneck"]),
                "rank": [],
            }
        rank_match = PLAN_RANK_RE.match(line)
        if rank_match:
            raw = fields(rank_match.group("body"))
            plans.setdefault(raw["op"], {"rank": []})["rank"].append(
                {
                    "rank": int(raw["rank"]),
                    "rows": int(raw["rows"]),
                    "begin": int(raw["begin"]),
                    "comm": int(raw["comm"]),
                    "tile": raw["tile"],
                    "cluster_m": int(raw["cluster_m"]),
                    "waves": int(raw["waves"]),
                    "compute_us": float(raw["compute_us"]),
                    "route_us": float(raw["route_us"]),
                    "critical_us": float(raw["critical_us"]),
                }
            )
    if not policies or not speedups:
        raise RuntimeError(f"benchmark output was incomplete:\n{output}")
    return {"policies": policies, "speedups": speedups, "plans": plans}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fast-devices", default="0,2,4,5,7")
    parser.add_argument("--slow-devices", default="1,3,6")
    parser.add_argument("--worlds", default="2,4,6,8")
    parser.add_argument("--m-values", default="2048,16384")
    parser.add_argument("--alphas", default="0,.25,.5,.75,1")
    parser.add_argument("--slow-ratio", type=float, default=1500.0 / 1980.0)
    parser.add_argument("--slow-hbm-ratio", type=float, default=1.0)
    parser.add_argument("--slow-nvlink-ratio", type=float, default=1.0)
    parser.add_argument("--nvlink-bidir-gbps", type=float, default=900.0)
    parser.add_argument("--q-heads", type=int, default=24)
    parser.add_argument("--kv-heads", type=int, default=24)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--row-quantum", type=int, default=256)
    parser.add_argument("--auto-plan", action="store_true")
    parser.add_argument("--allow-long-qkv", action="store_true")
    parser.add_argument("--allow-power-limited", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--oproj-fast-comm", type=int, default=8)
    parser.add_argument("--oproj-slow-comm", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    fast_devices = csv_ints(args.fast_devices)
    slow_devices = csv_ints(args.slow_devices)
    worlds = csv_ints(args.worlds)
    m_values = csv_ints(args.m_values)
    alphas: list[float | None] = (
        [None] if args.auto_plan else csv_floats(args.alphas)
    )
    records: list[dict[str, object]] = []
    started = time.time()

    cases: list[tuple[int, int, list[int], list[int]]] = []
    for world in worlds:
        minimum_slow = max(0, world - len(fast_devices))
        maximum_slow = min(world, len(slow_devices))
        for slow_count in range(minimum_slow, maximum_slow + 1):
            fast_count = world - slow_count
            devices = fast_devices[:fast_count] + slow_devices[:slow_count]
            slow_ranks = list(range(fast_count, world))
            cases.append((world, slow_count, devices, slow_ranks))

    total = len(cases) * len(m_values) * len(alphas)
    completed = 0
    for world, slow_count, devices, slow_ranks in cases:
        for local_rows in m_values:
            for alpha in alphas:
                command = [
                    str(args.binary),
                    "--world", str(world),
                    "--m", str(local_rows),
                    "--slow-ranks",
                    ",".join(str(rank) for rank in slow_ranks) or "none",
                    "--slow-ratio", str(args.slow_ratio),
                    "--slow-hbm-ratio", str(args.slow_hbm_ratio),
                    "--slow-nvlink-ratio", str(args.slow_nvlink_ratio),
                    "--nvlink-bidir-gbps", str(args.nvlink_bidir_gbps),
                    "--q-heads", str(args.q_heads),
                    "--kv-heads", str(args.kv_heads),
                    "--head-dim", str(args.head_dim),
                    "--hidden", str(args.hidden),
                    "--row-quantum", str(args.row_quantum),
                    "--warmup", str(args.warmup),
                    "--iterations", str(args.iterations),
                    "--oproj-fast-comm", str(args.oproj_fast_comm),
                    "--oproj-slow-comm", str(args.oproj_slow_comm),
                ]
                if args.auto_plan:
                    command.append("--auto-plan")
                else:
                    command.extend(["--alpha", str(alpha)])
                if args.allow_long_qkv:
                    command.append("--allow-long-qkv")
                if args.allow_power_limited:
                    command.append("--allow-power-limited")
                if not args.check:
                    command.append("--no-check")
                environment = os.environ.copy()
                environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                environment["CUDA_VISIBLE_DEVICES"] = ",".join(
                    str(device) for device in devices
                )
                run = subprocess.run(
                    command,
                    env=environment,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                    check=False,
                )
                if run.returncode != 0:
                    raise RuntimeError(
                        f"case failed ({run.returncode}): {' '.join(command)}\n"
                        f"{run.stdout}"
                    )
                record: dict[str, object] = {
                    "world": world,
                    "slow_count": slow_count,
                    "physical_devices": devices,
                    "logical_slow_ranks": slow_ranks,
                    "slow_ratio": args.slow_ratio,
                    "slow_hbm_ratio": args.slow_hbm_ratio,
                    "slow_nvlink_ratio": args.slow_nvlink_ratio,
                    "baseline_nvlink_bidirectional_gbps":
                        args.nvlink_bidir_gbps,
                    "q_heads": args.q_heads,
                    "kv_heads": args.kv_heads,
                    "head_dim": args.head_dim,
                    "hidden": args.hidden,
                    "qkv_n": (args.q_heads + 2 * args.kv_heads)
                        * args.head_dim,
                    "oproj_k": args.q_heads * args.head_dim,
                    "local_rows": local_rows,
                    "row_quantum": args.row_quantum,
                    "alpha": alpha,
                    "auto_plan": args.auto_plan,
                    "allow_long_qkv": args.allow_long_qkv,
                    "allow_power_limited": args.allow_power_limited,
                    "correctness": (
                        "exact_bf16_output" if args.check else "not_run"
                    ),
                    "warmup": args.warmup,
                    "iterations": args.iterations,
                    "oproj_fast_comm": args.oproj_fast_comm,
                    "oproj_slow_comm": args.oproj_slow_comm,
                    **parse_output(run.stdout),
                }
                records.append(record)
                completed += 1
                atomic_json(
                    args.output,
                    {
                        "schema": "fuse_heterogeneous_cp_weighted_sweep_v1",
                        "complete": completed == total,
                        "completed": completed,
                        "total": total,
                        "elapsed_seconds": time.time() - started,
                        "records": records,
                    },
                )
                qkv = record["speedups"]["qkv"]["p50"]
                oproj = record["speedups"]["oproj"]["p50"]
                setting = "auto" if alpha is None else f"alpha={alpha:.2f}"
                print(
                    f"[{completed:3d}/{total}] CP{world} x={slow_count} "
                    f"M={local_rows} {setting} "
                    f"QKV={qkv:.4f}x OProj={oproj:.4f}x",
                    flush=True,
                )


if __name__ == "__main__":
    main()
