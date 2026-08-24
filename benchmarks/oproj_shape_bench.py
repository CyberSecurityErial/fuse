#!/usr/bin/env python3
"""Reproducible shape and baseline sweep for inverse A2A -> dense GEMM."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "oproj_shape_bench"
DEFAULT_TORCHRUN = Path("/home/chen/miniforge3/envs/mmunlearner/bin/torchrun")


@dataclass(frozen=True)
class Model:
    name: str
    hidden: int
    q_heads: int
    aliases: str


MODELS = {
    model.name: model
    for model in (
        Model("qwen_dense_2k", 2048, 16, "current production Qwen dense"),
        Model("qwen25_7b", 3584, 28, "Qwen2.5-7B; same O-proj as Gemma2-9B"),
        Model("llama3_8b", 4096, 32, "Llama3-8B; same O-proj as Mistral/Mixtral"),
        Model("qwen25_32b", 5120, 40, "Qwen2.5-32B"),
        Model("llama31_70b", 8192, 64, "Llama3.1-70B; same O-proj as Qwen-72B"),
    )
}
SEQUENCES = (1024, 2048, 4096, 8192, 16384, 32768, 65536)
CONTEXT_PARALLEL = (4, 8)
COMM_CANDIDATES = (4, 8, 12, 14, 16, 20, 24, 32)
VISIBLE_DEVICES = {4: "0,2,4,5", 8: "0,1,2,3,4,5,6,7"}


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        action="append",
        choices=("fuse-sweep", "fuse-formal", "external-sweep", "external-formal", "aggregate"),
        required=True,
    )
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--seqs", type=parse_csv_ints, default=SEQUENCES)
    parser.add_argument("--cps", type=parse_csv_ints, default=CONTEXT_PARALLEL)
    parser.add_argument("--comm-ctas", type=parse_csv_ints, default=COMM_CANDIDATES)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sweep-warmup", type=int, default=3)
    parser.add_argument("--sweep-iters", type=int, default=12)
    parser.add_argument("--formal-warmup", type=int, default=10)
    parser.add_argument("--formal-iters", type=int, default=50)
    parser.add_argument(
        "--torchrun",
        type=Path,
        default=Path(os.environ.get("FUSE_TE_TORCHRUN", DEFAULT_TORCHRUN)),
    )
    return parser.parse_args()


def load(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def run(command: list[str], *, env: dict[str, str], output: Path, resume: bool) -> None:
    if resume and output.exists():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        print(completed.stdout, end="")
        print(completed.stderr, end="")
        completed.check_returncode()


def selected_models(value: str) -> list[Model]:
    names = [name for name in value.split(",") if name]
    unknown = sorted(set(names) - MODELS.keys())
    if unknown:
        raise ValueError(f"unknown models: {', '.join(unknown)}")
    return [MODELS[name] for name in names]


def key(model: Model, seq: int, cp: int) -> str:
    return f"{model.name}_s{seq}_cp{cp}"


def cases(args: argparse.Namespace):
    for model in selected_models(args.models):
        if model.q_heads * 128 != model.hidden:
            raise ValueError(f"{model.name}: hidden must equal q_heads * 128")
        for seq in args.seqs:
            for cp in args.cps:
                if cp not in VISIBLE_DEVICES or seq % cp != 0:
                    continue
                yield model, seq, cp


def fuse_command(
    model: Model,
    seq: int,
    cp: int,
    comm: int,
    warmup: int,
    iters: int,
    output: Path,
) -> list[str]:
    return [
        str(ROOT / "build" / "fuse_bench"),
        "--mode", "a2a_gemm_lhs",
        "--m", str(seq // cp),
        "--n", str(model.hidden),
        "--k", str(model.hidden),
        "--batch", "1",
        "--q-heads", str(model.q_heads),
        "--head-dim", "128",
        "--comm-ctas", str(comm),
        "--lhs-policy", "auto",
        "--raster", "n",
        "--swizzle", "1",
        "--warmup", str(warmup),
        "--iterations", str(iters),
        "--json-out", str(output),
    ]


def gpu_env(cp: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = VISIBLE_DEVICES[cp]
    return env


def run_fuse_sweep(args: argparse.Namespace) -> None:
    for model, seq, cp in cases(args):
        directory = args.results / "fuse_sweep" / key(model, seq, cp)
        for comm in args.comm_ctas:
            output = directory / f"comm{comm}_{args.sweep_warmup}w{args.sweep_iters}i.json"
            run(
                fuse_command(model, seq, cp, comm, args.sweep_warmup, args.sweep_iters, output),
                env=gpu_env(cp), output=output, resume=args.resume,
            )


def best_fuse_sweep(args: argparse.Namespace, model: Model, seq: int, cp: int) -> dict:
    files = sorted((args.results / "fuse_sweep" / key(model, seq, cp)).glob(
        f"*_{args.sweep_warmup}w{args.sweep_iters}i.json"
    ))
    if not files:
        raise FileNotFoundError(f"missing fuse sweep for {key(model, seq, cp)}")
    return min((load(path) for path in files), key=lambda item: item["results"]["fused"]["mean_ms"])


def run_fuse_formal(args: argparse.Namespace) -> None:
    for model, seq, cp in cases(args):
        winner = best_fuse_sweep(args, model, seq, cp)
        comm = int(winner["comm_ctas"])
        output = args.results / "fuse_formal" / f"{key(model, seq, cp)}_{args.formal_warmup}w{args.formal_iters}i.json"
        run(
            fuse_command(model, seq, cp, comm, args.formal_warmup, args.formal_iters, output),
            env=gpu_env(cp), output=output, resume=args.resume,
        )


def nccl_configs() -> list[tuple[int, int, int]]:
    return [
        (8, 512, 16), (16, 512, 16),
        (32, 128, 16), (32, 128, 64), (32, 128, 128),
        (32, 256, 16), (32, 256, 64), (32, 256, 128),
        (32, 512, 16), (32, 1024, 16), (32, 1024, 64), (32, 1024, 128),
    ]


def tuned_bucket_config(model: Model, seq: int, cp: int) -> tuple[int, int, int]:
    payload = (seq // cp) * model.hidden * 2
    if payload < 8 * 1024 * 1024:
        return 32, 1024, 16
    if payload < 64 * 1024 * 1024:
        return 32, 256, 64
    return 32, 128, 128


def external_command(
    args: argparse.Namespace,
    model: Model,
    seq: int,
    cp: int,
    warmup: int,
    iters: int,
    output: Path,
) -> list[str]:
    return [
        str(args.torchrun), "--standalone", f"--nproc-per-node={cp}",
        str(ROOT / "benchmarks" / "te_nccl_baseline.py"),
        "--mode", "oproj_a2a_gemm",
        "--global-seq", str(seq),
        "--hidden", str(model.hidden),
        "--batch", "1",
        "--q-heads", str(model.q_heads),
        "--head-dim", "128",
        "--warmup", str(warmup),
        "--iters", str(iters),
        "--cuda-graph", "--nccl-high-priority",
        "--json-out", str(output),
    ]


def nccl_env(cp: int, config: tuple[int, int, int]) -> dict[str, str]:
    channels, chunk_kib, ll_kib = config
    env = gpu_env(cp)
    env.update({
        "NCCL_MIN_P2P_NCHANNELS": str(channels),
        "NCCL_MAX_P2P_NCHANNELS": str(channels),
        "NCCL_P2P_NVL_CHUNKSIZE": str(chunk_kib * 1024),
        "NCCL_P2P_LL_THRESHOLD": str(ll_kib * 1024),
        "NCCL_IB_DISABLE": "1",
    })
    return env


def run_external_sweep(args: argparse.Namespace) -> None:
    for model, seq, cp in cases(args):
        directory = args.results / "external_sweep" / key(model, seq, cp)
        for config in nccl_configs():
            channels, chunk, ll = config
            output = directory / f"ch{channels}_chunk{chunk}_ll{ll}_{args.sweep_warmup}w{args.sweep_iters}i.json"
            run(
                external_command(args, model, seq, cp, args.sweep_warmup, args.sweep_iters, output),
                env=nccl_env(cp, config), output=output, resume=args.resume,
            )


def best_external_configs(args: argparse.Namespace, model: Model, seq: int, cp: int) -> dict[str, tuple[int, int, int]]:
    directory = args.results / "external_sweep" / key(model, seq, cp)
    files = sorted(directory.glob(f"*_{args.sweep_warmup}w{args.sweep_iters}i.json"))
    if not files:
        config = tuned_bucket_config(model, seq, cp)
        return {"te_nccl_oproj_boundary": config, "cublas_nccl_oproj_boundary": config}
    winners = {}
    for metric in ("te_nccl_oproj_boundary", "cublas_nccl_oproj_boundary"):
        candidates = []
        for path in files:
            result = load(path)
            if metric not in result["results"]:
                continue
            env = result["environment"]
            config = (
                int(env["NCCL_MIN_P2P_NCHANNELS"]),
                int(env["NCCL_P2P_NVL_CHUNKSIZE"]) // 1024,
                int(env["NCCL_P2P_LL_THRESHOLD"]) // 1024,
            )
            candidates.append((result["results"][metric]["mean_ms"], config))
        winners[metric] = min(candidates)[1]
    return winners


def run_external_formal(args: argparse.Namespace) -> None:
    for model, seq, cp in cases(args):
        configs = best_external_configs(args, model, seq, cp)
        unique_configs = sorted(set(configs.values()))
        for index, config in enumerate(unique_configs):
            label = "shared" if len(unique_configs) == 1 else f"winner{index}"
            channels, chunk, ll = config
            output = args.results / "external_formal" / (
                f"{key(model, seq, cp)}_{label}_ch{channels}_chunk{chunk}_ll{ll}_"
                f"{args.formal_warmup}w{args.formal_iters}i.json"
            )
            run(
                external_command(args, model, seq, cp, args.formal_warmup, args.formal_iters, output),
                env=nccl_env(cp, config), output=output, resume=args.resume,
            )


def metric_min(files: list[Path], metric: str) -> dict | None:
    candidates = [load(path) for path in files]
    candidates = [item for item in candidates if metric in item["results"]]
    return min(candidates, key=lambda item: item["results"][metric]["mean_ms"]) if candidates else None


def aggregate(args: argparse.Namespace) -> None:
    rows = []
    for model, seq, cp in cases(args):
        fuse_files = sorted((args.results / "fuse_formal").glob(f"{key(model, seq, cp)}_*.json"))
        if not fuse_files:
            continue
        fused = load(fuse_files[-1])
        external_files = sorted((args.results / "external_formal").glob(f"{key(model, seq, cp)}_*.json"))
        te = metric_min(external_files, "te_nccl_oproj_boundary")
        cublas = metric_min(external_files, "cublas_nccl_oproj_boundary")
        fused_ms = fused["results"]["fused"]["mean_ms"]
        pure = fused["results"]["cublaslt_autotuned"]
        row = {
            "model": model.name, "aliases": model.aliases, "cp": cp,
            "global_seq": seq, "m": seq // cp, "n": model.hidden, "k": model.hidden,
            "comm_ctas": fused["comm_ctas"], "policy": fused["lhs_policy"]["name"],
            "tile_m": fused["lhs_policy"]["tile_m"], "tile_n": fused["lhs_policy"]["tile_n"],
            "waves": fused["lhs_policy"]["waves"], "last_wave_ctas": fused["lhs_policy"]["last_wave_ctas"],
            "fused_ms": fused_ms, "fused_tflops": fused["results"]["fused"]["tflops_per_gpu"],
            "pure_cublaslt_ms": pure["mean_ms"],
            "fused_as_pure_percent": 100.0 * pure["mean_ms"] / fused_ms,
            "te_nccl_ms": te["results"]["te_nccl_oproj_boundary"]["mean_ms"] if te else None,
            "cublas_nccl_ms": cublas["results"]["cublas_nccl_oproj_boundary"]["mean_ms"] if cublas else None,
        }
        if row["te_nccl_ms"]:
            row["speedup_vs_te_nccl"] = row["te_nccl_ms"] / fused_ms
        if row["cublas_nccl_ms"]:
            row["speedup_vs_cublas_nccl"] = row["cublas_nccl_ms"] / fused_ms
        rows.append(row)
    args.results.mkdir(parents=True, exist_ok=True)
    with (args.results / "summary.json").open("w") as handle:
        json.dump(rows, handle, indent=2)
    if rows:
        fields = sorted({field for row in rows for field in row})
        with (args.results / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    args = parse_args()
    for phase in args.phase:
        {
            "fuse-sweep": run_fuse_sweep,
            "fuse-formal": run_fuse_formal,
            "external-sweep": run_external_sweep,
            "external-formal": run_external_formal,
            "aggregate": aggregate,
        }[phase](args)


if __name__ == "__main__":
    main()
