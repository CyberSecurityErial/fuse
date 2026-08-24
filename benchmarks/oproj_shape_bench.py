#!/usr/bin/env python3
"""Reproducible strong-baseline matrix for inverse A2A -> dense GEMM."""

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
    head_dim: int
    aliases: str

    @property
    def attention_width(self) -> int:
        return self.q_heads * self.head_dim


MODELS = {
    model.name: model
    for model in (
        Model("qwen_dense_2k", 2048, 16, 128, "current production Qwen dense"),
        Model("qwen25_7b", 3584, 28, 128, "Qwen2.5-7B"),
        Model("gemma2_9b", 3584, 16, 256, "Gemma2-9B"),
        Model("llama3_8b", 4096, 32, 128, "Llama3-8B; same O-proj as Mistral/Mixtral"),
        Model("qwen25_32b", 5120, 40, 128, "Qwen2.5-32B"),
        Model("llama31_70b", 8192, 64, 128, "Llama3.1-70B; same O-proj as Qwen-72B"),
        Model("representative_small", 4096, 32, 128, "single-digit-B median-like O-proj"),
        Model("representative_medium", 5120, 40, 128, "tens-of-B median-like O-proj"),
        Model("representative_large", 7168, 128, 128, "hundreds-of-B median-like O-proj"),
    )
}
# Three model-width representatives crossed with three independent per-rank
# token counts.  No entry identifies an exact model or an exact-shape policy.
DEFAULT_MODELS = (
    "representative_small",
    "representative_medium",
    "representative_large",
)
SEQUENCES = (1024, 4096, 16384)
CONTEXT_PARALLEL = (4,)
VISIBLE_DEVICES = {
    4: os.environ.get("FUSE_CP4_DEVICES", "0,2,4,5"),
    8: os.environ.get("FUSE_CP8_DEVICES", "0,1,2,3,4,5,6,7"),
}
NCCL_CHANNELS = (8, 16, 24, 32)
NCCL_CHUNK_KIB = (128, 256, 512, 1024)
NCCL_LL_KIB = (16, 64, 128)
EXECUTION_MODES = (
    (True, True),
    (True, False),
    (False, True),
    (False, False),
)


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        action="append",
        choices=(
            "baseline-nccl-sweep",
            "baseline-mode-sweep",
            "baseline-formal",
            "baseline-aggregate",
            "shape-table",
            "fuse-formal",
        ),
        required=True,
    )
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seqs", type=parse_csv_ints, default=SEQUENCES)
    parser.add_argument("--cps", type=parse_csv_ints, default=CONTEXT_PARALLEL)
    parser.add_argument("--nccl-channels", type=parse_csv_ints, default=NCCL_CHANNELS)
    parser.add_argument("--nccl-chunk-kib", type=parse_csv_ints, default=NCCL_CHUNK_KIB)
    parser.add_argument("--nccl-ll-kib", type=parse_csv_ints, default=NCCL_LL_KIB)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sweep-warmup", type=int, default=3)
    parser.add_argument("--sweep-iters", type=int, default=12)
    parser.add_argument("--formal-warmup", type=int, default=10)
    parser.add_argument("--formal-iters", type=int, default=50)
    parser.add_argument("--cublaslt-tune-warmup", type=int, default=5)
    parser.add_argument("--cublaslt-tune-iters", type=int, default=30)
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


def run_matrix(
    command: list[str],
    *,
    env: dict[str, str],
    entries: list[dict[str, object]],
    manifest: Path,
    resume: bool,
) -> None:
    pending = [entry for entry in entries if not (resume and Path(entry["json_out"]).exists())]
    if not pending:
        return
    manifest.parent.mkdir(parents=True, exist_ok=True)
    rendezvous_files = []
    for index, entry in enumerate(pending):
        rendezvous = Path(f"/tmp/fuse_oproj_rendezvous_{os.getpid()}_{index}")
        rendezvous.unlink(missing_ok=True)
        entry["rendezvous_file"] = str(rendezvous)
        rendezvous_files.append(rendezvous)
    manifest.write_text(json.dumps(pending, indent=2) + "\n")
    command = command + ["--matrix-manifest", str(manifest)]
    print(f"RUN-MATRIX {manifest} ({len(pending)} configurations)", flush=True)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    for rendezvous in rendezvous_files:
        rendezvous.unlink(missing_ok=True)
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
        for seq in args.seqs:
            for cp in args.cps:
                if (cp not in VISIBLE_DEVICES or seq % cp != 0 or
                        model.q_heads % cp != 0):
                    continue
                yield model, seq, cp


def fuse_command(
    model: Model,
    seq: int,
    cp: int,
    warmup: int,
    iters: int,
    output: Path,
) -> list[str]:
    return [
        str(ROOT / "build" / "fuse_bench"),
        "--mode", "a2a_gemm_lhs",
        "--m", str(seq // cp),
        "--n", str(model.hidden),
        "--k", str(model.attention_width),
        "--batch", "1",
        "--q-heads", str(model.q_heads),
        "--head-dim", str(model.head_dim),
        # Zero delegates CTA count to the production runtime heuristic.  The
        # benchmark never selects a per-shape winner from a manual sweep.
        "--comm-ctas", "0",
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


def run_fuse_formal(args: argparse.Namespace) -> None:
    for model, seq, cp in cases(args):
        output = args.results / "fuse_formal" / f"{key(model, seq, cp)}_{args.formal_warmup}w{args.formal_iters}i.json"
        run(
            fuse_command(model, seq, cp, args.formal_warmup, args.formal_iters, output),
            env=gpu_env(cp), output=output, resume=args.resume,
        )


def nccl_configs(args: argparse.Namespace) -> list[tuple[int, int, int]]:
    return [
        (channels, chunk, ll)
        for channels in args.nccl_channels
        for chunk in args.nccl_chunk_kib
        for ll in args.nccl_ll_kib
    ]


def external_matrix_command(
    args: argparse.Namespace,
    model: Model,
    seq: int,
    cp: int,
    warmup: int,
    iters: int,
) -> list[str]:
    return [
        str(args.torchrun), "--standalone", f"--nproc-per-node={cp}",
        str(ROOT / "benchmarks" / "te_nccl_baseline.py"),
        "--mode", "oproj_a2a_gemm",
        "--global-seq", str(seq),
        "--hidden", str(model.hidden),
        "--batch", "1",
        "--q-heads", str(model.q_heads),
        "--head-dim", str(model.head_dim),
        "--warmup", str(warmup), "--iters", str(iters),
        "--cublaslt-tune-warmup", str(args.cublaslt_tune_warmup),
        "--cublaslt-tune-iters", str(args.cublaslt_tune_iters),
    ]


def nccl_environment_values(config: tuple[int, int, int]) -> dict[str, str]:
    channels, chunk_kib, ll_kib = config
    return {
        "NCCL_MIN_P2P_NCHANNELS": str(channels),
        "NCCL_MAX_P2P_NCHANNELS": str(channels),
        "NCCL_P2P_NVL_CHUNKSIZE": str(chunk_kib * 1024),
        "NCCL_P2P_LL_THRESHOLD": str(ll_kib * 1024),
        "NCCL_IB_DISABLE": "1",
        "NCCL_GRAPH_REGISTER": "1",
        "NCCL_LOCAL_REGISTER": "1",
    }


def matrix_entry(
    output: Path,
    config: tuple[int, int, int],
    cuda_graph: bool,
    high_priority: bool,
) -> dict[str, object]:
    return {
        "json_out": str(output.resolve()),
        "environment": nccl_environment_values(config),
        "cuda_graph": cuda_graph,
        "nccl_high_priority": high_priority,
    }


def spec_label(config: tuple[int, int, int], cuda_graph: bool, high_priority: bool) -> str:
    channels, chunk, ll = config
    return (
        f"ch{channels}_chunk{chunk}_ll{ll}_"
        f"{'graph' if cuda_graph else 'eager'}_"
        f"{'hp' if high_priority else 'normal'}"
    )


def run_baseline_nccl_sweep(args: argparse.Namespace) -> None:
    for model, seq, cp in cases(args):
        directory = args.results / "baseline_nccl_sweep" / key(model, seq, cp)
        entries = []
        for config in nccl_configs(args):
            output = directory / (
                f"{spec_label(config, True, True)}_"
                f"{args.sweep_warmup}w{args.sweep_iters}i.json"
            )
            entries.append(matrix_entry(output, config, True, True))
        run_matrix(
            external_matrix_command(
                args, model, seq, cp, args.sweep_warmup, args.sweep_iters,
            ),
            env=gpu_env(cp), entries=entries,
            manifest=args.results / "manifests" / "baseline_nccl_sweep" / f"{key(model, seq, cp)}.json",
            resume=args.resume,
        )


BOUNDARY_METRICS = ("te_nccl_oproj_boundary", "cublaslt_nccl_oproj_boundary")


def result_spec(result: dict) -> tuple[tuple[int, int, int], bool, bool]:
    env = result["environment"]
    return (
        (
            int(env["NCCL_MIN_P2P_NCHANNELS"]),
            int(env["NCCL_P2P_NVL_CHUNKSIZE"]) // 1024,
            int(env["NCCL_P2P_LL_THRESHOLD"]) // 1024,
        ),
        bool(result["cuda_graph"]),
        bool(result["nccl_high_priority"]),
    )


def best_nccl_specs(args: argparse.Namespace, model: Model, seq: int, cp: int):
    directory = args.results / "baseline_nccl_sweep" / key(model, seq, cp)
    files = sorted(directory.glob(f"*_{args.sweep_warmup}w{args.sweep_iters}i.json"))
    if not files:
        raise FileNotFoundError(f"missing NCCL sweep for {key(model, seq, cp)}")
    winners = {}
    for metric in BOUNDARY_METRICS:
        winner = min(
            (load(path) for path in files),
            key=lambda item: item["results"][metric]["mean_ms"],
        )
        winners[metric] = result_spec(winner)
    return winners


def run_baseline_mode_sweep(args: argparse.Namespace) -> None:
    for model, seq, cp in cases(args):
        configs = {spec[0] for spec in best_nccl_specs(args, model, seq, cp).values()}
        directory = args.results / "baseline_mode_sweep" / key(model, seq, cp)
        entries = []
        for config in sorted(configs):
            for cuda_graph, high_priority in EXECUTION_MODES:
                output = directory / (
                    f"{spec_label(config, cuda_graph, high_priority)}_"
                    f"{args.sweep_warmup}w{args.sweep_iters}i.json"
                )
                entries.append(matrix_entry(output, config, cuda_graph, high_priority))
        run_matrix(
            external_matrix_command(
                args, model, seq, cp, args.sweep_warmup, args.sweep_iters,
            ),
            env=gpu_env(cp), entries=entries,
            manifest=args.results / "manifests" / "baseline_mode_sweep" / f"{key(model, seq, cp)}.json",
            resume=args.resume,
        )


def best_mode_specs(args: argparse.Namespace, model: Model, seq: int, cp: int):
    directory = args.results / "baseline_mode_sweep" / key(model, seq, cp)
    files = sorted(directory.glob(f"*_{args.sweep_warmup}w{args.sweep_iters}i.json"))
    if not files:
        raise FileNotFoundError(f"missing execution-mode sweep for {key(model, seq, cp)}")
    winners = {}
    for metric in BOUNDARY_METRICS:
        winner = min(
            (load(path) for path in files),
            key=lambda item: item["results"][metric]["mean_ms"],
        )
        winners[metric] = result_spec(winner)
    return winners


def run_baseline_formal(args: argparse.Namespace) -> None:
    for model, seq, cp in cases(args):
        specs = sorted(set(best_mode_specs(args, model, seq, cp).values()))
        entries = []
        for config, cuda_graph, high_priority in specs:
            output = args.results / "baseline_formal" / (
                f"{key(model, seq, cp)}_{spec_label(config, cuda_graph, high_priority)}_"
                f"{args.formal_warmup}w{args.formal_iters}i.json"
            )
            entries.append(matrix_entry(output, config, cuda_graph, high_priority))
        run_matrix(
            external_matrix_command(
                args, model, seq, cp, args.formal_warmup, args.formal_iters,
            ),
            env=gpu_env(cp), entries=entries,
            manifest=args.results / "manifests" / "baseline_formal" / f"{key(model, seq, cp)}.json",
            resume=args.resume,
        )


def metric_min(files: list[Path], metric: str) -> dict | None:
    candidates = [load(path) for path in files]
    candidates = [item for item in candidates if metric in item["results"]]
    return min(candidates, key=lambda item: item["results"][metric]["mean_ms"]) if candidates else None


def baseline_aggregate(args: argparse.Namespace) -> None:
    rows = []
    for model, seq, cp in cases(args):
        formal = sorted((args.results / "baseline_formal").glob(f"{key(model, seq, cp)}_*.json"))
        if not formal:
            raise FileNotFoundError(f"missing formal baseline for {key(model, seq, cp)}")
        metrics = {
            name: metric_min(formal, name)
            for name in (
                "te_oproj_gemm",
                "cublaslt_oproj_gemm",
                "te_nccl_inverse_a2a",
                "te_nccl_oproj_boundary",
                "cublaslt_nccl_oproj_boundary",
            )
        }
        row = {
            "model": model.name, "aliases": model.aliases, "cp": cp,
            "global_seq": seq, "m": seq // cp, "n": model.hidden,
            "k": model.attention_width,
        }
        for name, result in metrics.items():
            if result is None:
                continue
            stats = result["results"][name]
            prefix = name.removesuffix("_oproj_boundary").removesuffix("_oproj_gemm")
            row[f"{prefix}_mean_ms"] = stats["mean_ms"]
            row[f"{prefix}_p50_ms"] = stats["p50_ms"]
            row[f"{prefix}_p95_ms"] = stats["p95_ms"]
            if "tflops_per_gpu" in stats:
                row[f"{prefix}_tflops"] = stats["tflops_per_gpu"]
            if "nccl" in name:
                config, graph, high_priority = result_spec(result)
                row[f"{prefix}_nccl_channels"] = config[0]
                row[f"{prefix}_nccl_chunk_kib"] = config[1]
                row[f"{prefix}_nccl_ll_kib"] = config[2]
                row[f"{prefix}_cuda_graph"] = graph
                row[f"{prefix}_high_priority"] = high_priority
            if name == "cublaslt_oproj_gemm":
                row["cublaslt_plans"] = result["cublaslt_plans"]
        rows.append(row)
    args.results.mkdir(parents=True, exist_ok=True)
    with (args.results / "baseline_summary.json").open("w") as handle:
        json.dump(rows, handle, indent=2)
    if rows:
        fields = sorted({field for row in rows for field in row})
        with (args.results / "baseline_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def write_shape_table(args: argparse.Namespace) -> None:
    rows = []
    for model, seq, cp in cases(args):
        m = seq // cp
        n = model.hidden
        k = model.attention_width
        rows.append({
            "model": model.name,
            "aliases": model.aliases,
            "global_seq": seq,
            "cp": cp,
            "m": m,
            "n": n,
            "k": k,
            "q_heads": model.q_heads,
            "head_dim": model.head_dim,
            "a2a_input": f"[1,{seq},{model.q_heads // cp},{model.head_dim}]",
            "gemm_input": f"[{m},{k}]",
            "gemm_weight": f"[{n},{k}]",
            "gemm_output": f"[{m},{n}]",
            "payload_mib_per_gpu": m * k * 2 / (1 << 20),
            "gemm_gflop_per_gpu": 2 * m * n * k / 1.0e9,
        })
    args.results.mkdir(parents=True, exist_ok=True)
    with (args.results / "shape_matrix.json").open("w") as handle:
        json.dump(rows, handle, indent=2)
    fields = list(rows[0])
    with (args.results / "shape_matrix.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (args.results / "shape_matrix.md").open("w") as handle:
        handle.write(
            "| Model | S | CP | M | N | K | A2A input/rank | GEMM | Payload MiB | GFLOP/GPU |\n"
            "|---|---:|---:|---:|---:|---:|---|---|---:|---:|\n"
        )
        for row in rows:
            handle.write(
                f"| {row['model']} | {row['global_seq']} | {row['cp']} | "
                f"{row['m']} | {row['n']} | {row['k']} | {row['a2a_input']} | "
                f"{row['gemm_input']} × {row['gemm_weight']}^T | "
                f"{row['payload_mib_per_gpu']:.2f} | {row['gemm_gflop_per_gpu']:.3f} |\n"
            )


def main() -> None:
    args = parse_args()
    for phase in args.phase:
        {
            "baseline-nccl-sweep": run_baseline_nccl_sweep,
            "baseline-mode-sweep": run_baseline_mode_sweep,
            "baseline-formal": run_baseline_formal,
            "baseline-aggregate": baseline_aggregate,
            "shape-table": write_shape_table,
            "fuse-formal": run_fuse_formal,
        }[phase](args)


if __name__ == "__main__":
    main()
