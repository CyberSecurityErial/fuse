#!/usr/bin/env python3
"""Tune and formalize the TE Userbuffers A2A->O-projection baseline."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/chen/miniforge3/envs/mmunlearner/bin/python")
TE_ROOT = Path("/home/chen/workspace/source_code/TransformerEngine")
MODELS = {
    "small": (4096, 32, 128),
    "medium": (5120, 40, 128),
    "large": (7168, 128, 128),
}
DEVICES = {4: "0,2,4,5", 8: "0,1,2,3,4,5,6,7"}


@dataclass(frozen=True)
class Config:
    comm_sm: int
    streams: int = 1
    push: bool = True
    use_ce: bool = False
    pack_block: int = 512
    pack_warps: int = 4
    reverse: bool = False

    @property
    def math_sm(self) -> int:
        return 132 - self.comm_sm

    @property
    def tag(self) -> str:
        direction = "push" if self.push else "pull"
        ce = "ce" if self.use_ce else "sm"
        reverse = "rev" if self.reverse else "fwd"
        return (
            f"c{self.comm_sm}_m{self.math_sm}_s{self.streams}_{direction}_{ce}_"
            f"b{self.pack_block}_w{self.pack_warps}_{reverse}"
        )


def csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("sweep", "formal", "summary", "all"), default="all")
    parser.add_argument("--models", default="small,medium,large")
    parser.add_argument("--seqs", type=csv_ints, default=(1024, 4096, 16384))
    parser.add_argument("--cps", type=csv_ints, default=(4, 8))
    parser.add_argument("--comm-sm", type=csv_ints, default=(4, 8, 12, 16, 20, 24))
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "te_userbuffers_shape_bench")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sweep-warmup", type=int, default=5)
    parser.add_argument("--sweep-iters", type=int, default=15)
    parser.add_argument("--formal-warmup", type=int, default=10)
    parser.add_argument("--formal-iters", type=int, default=50)
    return parser.parse_args()


def cases(args: argparse.Namespace):
    for name in (item for item in args.models.split(",") if item):
        hidden, q_heads, head_dim = MODELS[name]
        for seq in args.seqs:
            for cp in args.cps:
                if seq % cp == 0 and q_heads % cp == 0:
                    yield name, hidden, q_heads, head_dim, seq, cp


def base_key(name: str, seq: int, cp: int) -> str:
    return f"{name}_s{seq}_cp{cp}"


def structural_candidates(comm: int) -> tuple[Config, ...]:
    return (
        Config(comm, streams=3),
        Config(comm, push=False),
        Config(comm, use_ce=True),
        Config(comm, pack_block=256),
        Config(comm, pack_block=1024),
        Config(comm, pack_warps=8),
        Config(comm, reverse=True),
    )


def command(
    *, hidden: int, q_heads: int, head_dim: int, seq: int, cp: int,
    config: Config, warmup: int, iters: int, output: Path,
    tune_warmup: int, tune_iters: int,
) -> list[str]:
    result = [
        str(PYTHON), "-m", "torch.distributed.run", "--standalone",
        f"--nproc-per-node={cp}", str(ROOT / "benchmarks" / "te_userbuffers_oproj.py"),
        "--global-seq", str(seq), "--hidden", str(hidden),
        "--q-heads", str(q_heads), "--head-dim", str(head_dim),
        "--batch", "1", "--warmup", str(warmup), "--iters", str(iters),
        "--num-comm-sm", str(config.comm_sm), "--math-sm", str(config.math_sm),
        "--num-streams", str(config.streams),
        "--pack-block", str(config.pack_block), "--pack-warps", str(config.pack_warps),
        "--tune-warmup", str(tune_warmup), "--tune-iters", str(tune_iters),
        "--workspace-mib", "64", "--cuda-graph", "--json-out", str(output),
        "--parallel-sends" if config.streams > 1 else "--no-parallel-sends",
        "--push" if config.push else "--no-push",
        "--use-ce" if config.use_ce else "--no-use-ce",
    ]
    if config.reverse:
        result.append("--reverse")
    return result


def environment(cp: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = DEVICES[cp]
    env["PYTHONPATH"] = str(TE_ROOT)
    env["LD_LIBRARY_PATH"] = f"{TE_ROOT}:/usr/local/cuda/lib64"
    return env


def run_one(cmd: list[str], env: dict[str, str], output: Path, resume: bool) -> None:
    if resume and output.exists():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", output, flush=True)
    completed = subprocess.run(
        cmd, cwd=ROOT, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if completed.returncode:
        print(completed.stdout[-12000:], flush=True)
        completed.check_returncode()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def valid_result(path: Path) -> bool:
    if not path.exists():
        return False
    data = load(path)
    check = data["correctness"]
    return (
        check["self_pack_mismatches"] == 0
        and check["remote_recv_mismatches"] == 0
        and check.get("post_graph_max_abs", 1.0) <= 0.01
    )


def best_sweep(results: Path, key: str) -> tuple[Config, Path]:
    records: list[tuple[float, Config, Path]] = []
    for path in (results / "sweep" / key).glob("*.json"):
        if not valid_result(path):
            continue
        data = load(path)
        cfg = data["config"]
        config = Config(
            cfg["num_comm_sm"], cfg["num_streams"], cfg["push"], cfg["use_ce"],
            cfg["pack_block"], cfg["pack_warps"], cfg["reverse"],
        )
        p50 = data["results"]["te_userbuffers_oproj_boundary"]["p50_ms"]
        records.append((p50, config, path))
    if not records:
        raise RuntimeError(f"no valid sweep result for {key}")
    _, config, path = min(records, key=lambda item: item[0])
    return config, path


def sweep(args: argparse.Namespace) -> None:
    for name, hidden, q_heads, head_dim, seq, cp in cases(args):
        key = base_key(name, seq, cp)
        base_records = []
        for comm in args.comm_sm:
            config = Config(comm)
            output = args.results / "sweep" / key / f"{config.tag}.json"
            run_one(
                command(
                    hidden=hidden, q_heads=q_heads, head_dim=head_dim, seq=seq, cp=cp,
                    config=config, warmup=args.sweep_warmup, iters=args.sweep_iters,
                    output=output, tune_warmup=5, tune_iters=15,
                ),
                environment(cp), output, args.resume,
            )
            if valid_result(output):
                p50 = load(output)["results"]["te_userbuffers_oproj_boundary"]["p50_ms"]
                base_records.append((p50, comm))
        if not base_records:
            raise RuntimeError(f"no valid communication sweep for {key}")
        best_comm = min(base_records)[1]
        for config in structural_candidates(best_comm):
            output = args.results / "sweep" / key / f"{config.tag}.json"
            run_one(
                command(
                    hidden=hidden, q_heads=q_heads, head_dim=head_dim, seq=seq, cp=cp,
                    config=config, warmup=args.sweep_warmup, iters=args.sweep_iters,
                    output=output, tune_warmup=5, tune_iters=15,
                ),
                environment(cp), output, args.resume,
            )


def formal(args: argparse.Namespace) -> None:
    for name, hidden, q_heads, head_dim, seq, cp in cases(args):
        key = base_key(name, seq, cp)
        config, _ = best_sweep(args.results, key)
        output = args.results / "formal" / f"{key}_{config.tag}.json"
        run_one(
            command(
                hidden=hidden, q_heads=q_heads, head_dim=head_dim, seq=seq, cp=cp,
                config=config, warmup=args.formal_warmup, iters=args.formal_iters,
                output=output, tune_warmup=10, tune_iters=30,
            ),
            environment(cp), output, args.resume,
        )


def summarize(args: argparse.Namespace) -> None:
    rows = []
    for name, hidden, q_heads, head_dim, seq, cp in cases(args):
        key = base_key(name, seq, cp)
        config, _ = best_sweep(args.results, key)
        formal_paths = list((args.results / "formal").glob(f"{key}_*.json"))
        if not formal_paths:
            continue
        formal_path = formal_paths[0]
        data = load(formal_path)
        stats = data["results"]["te_userbuffers_oproj_boundary"]
        m, n, k = (data["gemm_shape"][dim] for dim in ("m", "n", "k"))
        p50_tflops = 2.0 * m * n * k / stats["p50_ms"] / 1.0e9
        rows.append({
            "model": name, "global_seq": seq, "cp": cp, "m": m, "n": n, "k": k,
            **asdict(config), "math_sm": config.math_sm,
            "p50_ms": stats["p50_ms"], "p95_ms": stats["p95_ms"],
            "p50_tflops_per_gpu": p50_tflops,
        })
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    if rows:
        with (args.results / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=rows[0].keys(), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
    print(f"SUMMARY {len(rows)} formal cases", flush=True)


def main() -> None:
    args = parse_args()
    if args.phase in ("sweep", "all"):
        sweep(args)
    if args.phase in ("formal", "all"):
        formal(args)
    if args.phase in ("summary", "all"):
        summarize(args)


if __name__ == "__main__":
    main()
