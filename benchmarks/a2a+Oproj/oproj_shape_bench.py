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


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "results" / "a2a-Oproj" / "oproj_mixed_shape_bench"
DEFAULT_TORCHRUN = Path("/home/chen/miniforge3/envs/mmunlearner/bin/torchrun")
V3_GOLDEN = (
    ROOT / "results" / "a2a-Oproj" / "oproj_v3_manual_comm_bench"
    / "summary.json"
)


@dataclass(frozen=True)
class Model:
    name: str
    suite: str
    hidden: int
    q_heads: int
    head_dim: int
    aliases: str
    max_context: int | None = None

    @property
    def attention_width(self) -> int:
        return self.q_heads * self.head_dim


MODELS = {
    model.name: model
    for model in (
        # Preserve the original three Golden stress geometries unchanged.
        Model("representative_small", "artificial", 4096, 32, 128,
              "single-digit-B median-like O-proj"),
        Model("representative_medium", "artificial", 5120, 40, 128,
              "tens-of-B median-like O-proj"),
        Model("representative_large", "artificial", 7168, 128, 128,
              "wide-K stress shape; not one specific GQA checkpoint"),
        # Real model geometries used by the mixed benchmark.
        Model("production_qwen_dense", "model", 2048, 16, 128,
              "current production Qwen dense"),
        Model("llama3_8b", "model", 4096, 32, 128,
              "Llama 3/3.1 8B"),
        Model("qwen25_14b_32b", "model", 5120, 40, 128,
              "Qwen2.5 14B/32B"),
        Model("llama31_405b", "model", 16384, 128, 128,
              "Llama 3.1 405B"),
        Model("nanbeige42_3b", "model", 3072, 48, 128,
              "Nanbeige4.2-3B", 262144),
        # Additional opt-in real shapes retained for targeted runs.
        Model("qwen25_7b", "model", 3584, 28, 128, "Qwen2.5 7B"),
        Model("gemma2_9b", "model", 3584, 16, 256, "Gemma 2 9B"),
        Model("llama31_70b", "model", 8192, 64, 128,
              "Llama 3.1 70B; same O-proj as Qwen-72B"),
    )
}
MODEL_ALIASES = {
    "qwen_dense_2k": "production_qwen_dense",
    "qwen25_32b": "qwen25_14b_32b",
}
DEFAULT_MODELS = (
    "representative_small",
    "representative_medium",
    "representative_large",
    "production_qwen_dense",
    "llama3_8b",
    "qwen25_14b_32b",
    "llama31_405b",
    "nanbeige42_3b",
)
SEQUENCES = (1024, 4096, 16384, 131072, 262144, 524288)
CONTEXT_PARALLEL = (4, 8)
VISIBLE_DEVICES = {
    4: os.environ.get("FUSE_CP4_DEVICES", "0,2,4,5"),
    8: os.environ.get("FUSE_CP8_DEVICES", "0,1,2,3,4,5,6,7"),
}
NCCL_CHANNELS = (8, 16, 24, 32)
NCCL_CHUNK_KIB = (128, 256, 512, 1024)
NCCL_LL_KIB = (16, 64, 128)
PACK_BLOCKS = (128, 256, 512, 1024)
PACK_WARPS = (4, 8)
EXECUTION_MODES = (
    (True, True),
    (True, False),
    (False, True),
    (False, False),
)
FUSE_COMM_CTAS = (2, 4, 6, 8, 10, 12, 14, 16, 20, 24)


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item for item in value.split(",") if item)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        action="append",
        choices=(
            "baseline-nccl-sweep",
            "baseline-mode-sweep",
            "baseline-pack-sweep",
            "baseline-formal",
            "baseline-aggregate",
            "shape-table",
            "fuse-formal",
            "fuse-aggregate",
            "fuse-comm-sweep",
            "fuse-comm-refine",
            "fuse-comm-formal",
            "fuse-calibrated-aggregate",
            "fuse-launch-formal",
            "fuse-launch-aggregate",
            "comparison-table",
        ),
        required=True,
    )
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seqs", type=parse_csv_ints, default=SEQUENCES)
    parser.add_argument("--cps", type=parse_csv_ints, default=CONTEXT_PARALLEL)
    parser.add_argument("--nccl-channels", type=parse_csv_ints, default=NCCL_CHANNELS)
    parser.add_argument("--nccl-chunk-kib", type=parse_csv_ints, default=NCCL_CHUNK_KIB)
    parser.add_argument("--nccl-ll-kib", type=parse_csv_ints, default=NCCL_LL_KIB)
    parser.add_argument("--nccl-shortlist", type=int, default=3)
    parser.add_argument("--pack-blocks", type=parse_csv_ints, default=PACK_BLOCKS)
    parser.add_argument("--pack-warps", type=parse_csv_ints, default=PACK_WARPS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--te-userbuffers-summary",
        type=Path,
        default=ROOT / "results" / "a2a-Oproj"
        / "te_userbuffers_mixed_shape_bench" / "summary.json",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sweep-warmup", type=int, default=3)
    parser.add_argument("--sweep-iters", type=int, default=12)
    parser.add_argument("--formal-warmup", type=int, default=10)
    parser.add_argument("--formal-iters", type=int, default=50)
    parser.add_argument(
        "--fuse-comm-ctas", type=parse_csv_ints, default=FUSE_COMM_CTAS
    )
    parser.add_argument("--fuse-comm-shortlist", type=int, default=3)
    parser.add_argument("--cublaslt-tune-warmup", type=int, default=5)
    parser.add_argument("--cublaslt-tune-iters", type=int, default=30)
    parser.add_argument(
        "--torchrun",
        type=Path,
        default=Path(os.environ.get("FUSE_TE_TORCHRUN", DEFAULT_TORCHRUN)),
    )
    parser.add_argument(
        "--fuse-launch-config",
        type=Path,
        default=Path(__file__).with_name("fuse_launch_config.csv"),
    )
    parser.add_argument(
        "--fuse-launches",
        type=parse_csv_strings,
        choices=None,
        default=("eager", "graph"),
        help="comma-separated formal launch modes: eager,graph",
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


def external_result_matches(
    path: Path,
    entry: dict[str, object],
    *,
    cp: int,
    warmup: int,
    iters: int,
) -> bool:
    if not path.exists():
        return False
    try:
        result = load(path)
        arguments = entry["arguments"]
        shape = result["model_shape"]
        environment = result["environment"]
        return (
            result.get("benchmark_schema") == "oproj_inverse_a2a_v1"
            and result["world_size"] == cp
            and result["warmup"] == warmup
            and result["iterations"] == iters
            and result["cuda_graph"] == entry["cuda_graph"]
            and result["nccl_high_priority"] == entry["nccl_high_priority"]
            and result["metric_profile"] == arguments["metric_profile"]
            and result.get("pack_block", 1024) == arguments.get("pack_block", 1024)
            and result.get("pack_warps", 4) == arguments.get("pack_warps", 4)
            and shape["global_seq"] == arguments["global_seq"]
            and shape["hidden"] == arguments["hidden"]
            and shape["q_heads"] == arguments["q_heads"]
            and shape["head_dim"] == arguments["head_dim"]
            and all(
                environment[name] == value
                for name, value in entry["environment"].items()
            )
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def run_matrix(
    command: list[str],
    *,
    env: dict[str, str],
    entries: list[dict[str, object]],
    manifest: Path,
    resume: bool,
    cp: int,
    warmup: int,
    iters: int,
) -> None:
    pending = [
        entry
        for entry in entries
        if not (
            resume
            and external_result_matches(
                Path(entry["json_out"]), entry, cp=cp, warmup=warmup, iters=iters
            )
        )
    ]
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
    invalid = [
        Path(entry["json_out"])
        for entry in pending
        if not external_result_matches(
            Path(entry["json_out"]), entry, cp=cp, warmup=warmup, iters=iters
        )
    ]
    if invalid:
        raise RuntimeError(f"invalid or missing benchmark outputs: {invalid[:4]}")


def selected_models(value: str) -> list[Model]:
    names = [name for name in value.split(",") if name]
    unknown = sorted(set(names) - MODELS.keys() - MODEL_ALIASES.keys())
    if unknown:
        raise ValueError(f"unknown models: {', '.join(unknown)}")
    return [MODELS[MODEL_ALIASES.get(name, name)] for name in names]


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
    comm_ctas: int = 0,
    lhs_policy: str = "auto",
    cuda_graph: bool = False,
) -> list[str]:
    command = [
        str(ROOT / "build" / "fuse_bench"),
        "--mode", "a2a_gemm_lhs",
        "--m", str(seq // cp),
        "--n", str(model.hidden),
        "--k", str(model.attention_width),
        "--batch", "1",
        "--q-heads", str(model.q_heads),
        "--head-dim", str(model.head_dim),
        "--comm-ctas", str(comm_ctas),
        "--lhs-policy", lhs_policy,
        "--raster", "n",
        "--swizzle", "1",
        "--warmup", str(warmup),
        "--iterations", str(iters),
        "--json-out", str(output),
    ]
    if cuda_graph:
        command.append("--cuda-graph")
    return command


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


def fuse_comm_path(
    args: argparse.Namespace,
    phase: str,
    model: Model,
    seq: int,
    cp: int,
    comm_ctas: int,
    warmup: int,
    iters: int,
) -> Path:
    return args.results / phase / (
        f"{key(model, seq, cp)}_comm{comm_ctas}_{warmup}w{iters}i.json"
    )


def run_fuse_comm_cases(
    args: argparse.Namespace,
    *,
    phase: str,
    candidates,
    warmup: int,
    iters: int,
) -> None:
    for model, seq, cp in cases(args):
        for comm_ctas in candidates(model, seq, cp):
            output = fuse_comm_path(
                args, phase, model, seq, cp, comm_ctas, warmup, iters
            )
            run(
                fuse_command(
                    model, seq, cp, warmup, iters, output, comm_ctas
                ),
                env=gpu_env(cp), output=output, resume=args.resume,
            )


def fuse_comm_sweep_results(
    args: argparse.Namespace, model: Model, seq: int, cp: int
) -> list[tuple[int, dict]]:
    results = []
    for comm_ctas in args.fuse_comm_ctas:
        path = fuse_comm_path(
            args, "fuse_comm_sweep", model, seq, cp, comm_ctas,
            args.sweep_warmup, args.sweep_iters,
        )
        if not path.exists():
            raise FileNotFoundError(path)
        results.append((comm_ctas, load(path)))
    return results


def fuse_refine_candidates(
    args: argparse.Namespace, model: Model, seq: int, cp: int
) -> tuple[int, ...]:
    winner = min(
        fuse_comm_sweep_results(args, model, seq, cp),
        key=lambda item: item[1]["results"]["fused"]["mean_ms"],
    )[0]
    coarse = set(args.fuse_comm_ctas)
    return tuple(
        value for value in (winner - 1, winner, winner + 1)
        if value > 0 and value not in coarse
    )


def fuse_all_short_results(
    args: argparse.Namespace, model: Model, seq: int, cp: int
) -> list[tuple[int, dict]]:
    results = fuse_comm_sweep_results(args, model, seq, cp)
    for comm_ctas in fuse_refine_candidates(args, model, seq, cp):
        path = fuse_comm_path(
            args, "fuse_comm_refine", model, seq, cp, comm_ctas,
            args.sweep_warmup, args.sweep_iters,
        )
        if not path.exists():
            raise FileNotFoundError(path)
        results.append((comm_ctas, load(path)))
    return results


def fuse_formal_candidates(
    args: argparse.Namespace, model: Model, seq: int, cp: int
) -> tuple[int, ...]:
    ranked = sorted(
        fuse_all_short_results(args, model, seq, cp),
        key=lambda item: item[1]["results"]["fused"]["mean_ms"],
    )
    return tuple(
        comm_ctas for comm_ctas, _ in ranked[:args.fuse_comm_shortlist]
    )


def run_fuse_comm_sweep(args: argparse.Namespace) -> None:
    run_fuse_comm_cases(
        args, phase="fuse_comm_sweep",
        candidates=lambda _model, _seq, _cp: args.fuse_comm_ctas,
        warmup=args.sweep_warmup, iters=args.sweep_iters,
    )


def run_fuse_comm_refine(args: argparse.Namespace) -> None:
    run_fuse_comm_cases(
        args, phase="fuse_comm_refine",
        candidates=lambda model, seq, cp: fuse_refine_candidates(
            args, model, seq, cp
        ),
        warmup=args.sweep_warmup, iters=args.sweep_iters,
    )


def run_fuse_comm_formal(args: argparse.Namespace) -> None:
    run_fuse_comm_cases(
        args, phase="fuse_comm_formal",
        candidates=lambda model, seq, cp: fuse_formal_candidates(
            args, model, seq, cp
        ),
        warmup=args.formal_warmup, iters=args.formal_iters,
    )


def calibrated_fuse_row(
    model: Model,
    seq: int,
    cp: int,
    data: dict,
    result_source: str,
) -> dict:
    shape = data["shape"]
    flops = 2.0 * shape["m"] * shape["n"] * shape["k"]
    fused = data["results"]["fused"]
    cublas = data["results"]["cublas"]
    cublaslt = data["results"]["cublaslt_autotuned"]
    pure_best_p50 = min(cublas["p50_ms"], cublaslt["p50_ms"])
    policy = data["lhs_policy"]

    def p50_tflops(name: str) -> float:
        return flops / data["results"][name]["p50_ms"] / 1.0e9

    return {
        "model": model.name,
        "suite": model.suite,
        "aliases": model.aliases,
        "global_seq": seq,
        "cp": cp,
        "m": shape["m"],
        "n": shape["n"],
        "k": shape["k"],
        "q_heads": model.q_heads,
        "head_dim": model.head_dim,
        "native_max_context": model.max_context,
        "beyond_native_context": (
            model.max_context is not None and seq > model.max_context
        ),
        "result_source": result_source,
        "comm_ctas": data["comm_ctas"],
        "policy": policy["name"],
        "tile_m": policy["tile_m"],
        "tile_n": policy["tile_n"],
        "tile_k": policy["tile_k"],
        "cluster_m": policy["cluster_m"],
        "compute_clusters": policy["compute_clusters"],
        "n_tiles": policy["n_tiles"],
        "waves": policy["waves"],
        "last_wave_clusters": policy["last_wave_clusters"],
        "last_wave_ctas": policy["last_wave_ctas"],
        "frontier_aligned": policy["frontier_aligned"],
        "full_last_wave": policy["full_last_wave"],
        "fused_mean_ms": fused["mean_ms"],
        "fused_p50_ms": fused["p50_ms"],
        "fused_p95_ms": fused["p95_ms"],
        "fused_p50_tflops_per_gpu": p50_tflops("fused"),
        "cublas_p50_ms": cublas["p50_ms"],
        "cublas_p50_tflops_per_gpu": p50_tflops("cublas"),
        "cublaslt_p50_ms": cublaslt["p50_ms"],
        "cublaslt_p50_tflops_per_gpu": p50_tflops("cublaslt_autotuned"),
        "fused_throughput_as_cublas_percent": (
            100.0 * cublas["p50_ms"] / fused["p50_ms"]
        ),
        "fused_throughput_as_best_pure_percent": (
            100.0 * pure_best_p50 / fused["p50_ms"]
        ),
        "same_policy_p50_ms": data["results"]["same_policy_cutlass"]["p50_ms"],
        "compute_subgrid_p50_ms": (
            data["results"]["compute_subgrid_cutlass"]["p50_ms"]
        ),
        "route_p50_ms": data["results"]["inverse_a2a_route"]["p50_ms"],
        "sequential_p50_ms": (
            data["results"]["same_policy_sequential"]["p50_ms"]
        ),
        "overlap_ratio": data["overlap_ratio"],
        "exact_mismatches": data.get("correctness", {}).get(
            "exact_mismatches", data.get("exact_mismatches", 0)
        ),
    }


def best_calibrated_formal(
    args: argparse.Namespace, model: Model, seq: int, cp: int
) -> tuple[int, dict]:
    candidates = []
    for comm_ctas in fuse_formal_candidates(args, model, seq, cp):
        path = fuse_comm_path(
            args, "fuse_comm_formal", model, seq, cp, comm_ctas,
            args.formal_warmup, args.formal_iters,
        )
        if not path.exists():
            raise FileNotFoundError(path)
        candidates.append((comm_ctas, load(path)))
    return min(
        candidates,
        key=lambda item: item[1]["results"]["fused"]["p50_ms"],
    )


def v3_golden_rows() -> dict[tuple[str, int, int], dict]:
    data = load(V3_GOLDEN)
    return {
        (row["model"], row["global_seq"], row["cp"]): row
        for row in data["cases"]
    }


def normalize_v3_row(
    source: dict, model: Model, *, result_source: str
) -> dict:
    row = dict(source)
    row.update({
        "model": model.name,
        "suite": model.suite,
        "aliases": model.aliases,
        "q_heads": model.q_heads,
        "head_dim": model.head_dim,
        "native_max_context": model.max_context,
        "beyond_native_context": (
            model.max_context is not None
            and row["global_seq"] > model.max_context
        ),
        "result_source": result_source,
        "fused_throughput_as_cublas_percent": (
            source["v3_throughput_as_cublas_percent"]
        ),
        "fused_throughput_as_best_pure_percent": (
            source["v3_throughput_as_cublas_percent"]
        ),
    })
    return row


def write_rows(rows: list[dict], stem: Path) -> None:
    with stem.with_suffix(".json").open("w") as handle:
        json.dump(rows, handle, indent=2)
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fuse_calibrated_aggregate(args: argparse.Namespace) -> None:
    golden = v3_golden_rows()
    geometry_aliases = {
        "llama3_8b": "representative_small",
        "qwen25_14b_32b": "representative_medium",
    }
    new_models = {
        "production_qwen_dense", "nanbeige42_3b", "llama31_405b"
    }
    rows = []
    for model in (MODELS[name] for name in DEFAULT_MODELS):
        for seq in SEQUENCES:
            for cp in CONTEXT_PARALLEL:
                if model.name.startswith("representative_"):
                    source = golden[(model.name, seq, cp)]
                    row = normalize_v3_row(
                        source, model, result_source="v3_golden"
                    )
                elif model.name in geometry_aliases:
                    source = golden[(geometry_aliases[model.name], seq, cp)]
                    row = normalize_v3_row(
                        source, model,
                        result_source="v3_golden_same_geometry",
                    )
                elif model.name in new_models:
                    _, data = best_calibrated_formal(args, model, seq, cp)
                    row = calibrated_fuse_row(
                        model, seq, cp, data,
                        result_source="new_geometry_manual_comm_ctas",
                    )
                else:
                    raise AssertionError(model.name)
                rows.append(row)
    if len(rows) != 96:
        raise AssertionError(f"expected 96 calibrated rows, got {len(rows)}")
    write_rows(rows, args.results / "fused_calibrated_summary")


def fuse_launch_configs(args: argparse.Namespace) -> dict[str, dict]:
    path = args.fuse_launch_config
    if not path.is_absolute():
        path = ROOT / path
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    configs = {}
    for row in rows:
        row["comm_ctas"] = int(row["comm_ctas"])
        row["swizzle"] = int(row["swizzle"])
        if row["case"] in configs:
            raise ValueError(f"duplicate launch config: {row['case']}")
        configs[row["case"]] = row
    expected = {key(model, seq, cp) for model, seq, cp in cases(args)}
    missing = sorted(expected - configs.keys())
    if missing:
        raise KeyError(f"launch config is missing cases: {missing[:4]}")
    return configs


def fuse_launch_path(
    args: argparse.Namespace,
    model: Model,
    seq: int,
    cp: int,
    launch: str,
) -> Path:
    return args.results / "fuse_launch_formal" / launch / (
        f"{key(model, seq, cp)}_"
        f"{args.formal_warmup}w{args.formal_iters}i.json"
    )


def cli_policy(policy: str) -> str:
    return {
        "m128n256_cluster_m2": "m128n256c2",
        "m128n320_cluster_m2": "m128n320c2",
    }.get(policy, policy)


def fuse_launch_result_matches(
    path: Path,
    *,
    model: Model,
    seq: int,
    cp: int,
    launch: str,
    config: dict,
    warmup: int,
    iters: int,
) -> bool:
    if not path.exists():
        return False
    try:
        result = load(path)
        return (
            result["mode"] == "a2a_gemm_lhs"
            and result["world_size"] == cp
            and result["warmup"] == warmup
            and result["iterations"] == iters
            and result["fused_cuda_graph"] == (launch == "graph")
            and result.get("cuda_graph_preuploaded", False)
            == (launch == "graph")
            and result["shape"] == {
                "m": seq // cp,
                "n": model.hidden,
                "k": model.attention_width,
                "l": 1,
            }
            and result["comm_ctas"] == config["comm_ctas"]
            and result["lhs_policy"]["name"] == config["policy"]
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def run_fuse_launch_formal(args: argparse.Namespace) -> None:
    invalid = sorted(set(args.fuse_launches) - {"eager", "graph"})
    if invalid:
        raise ValueError(f"invalid --fuse-launches values: {invalid}")
    configs = fuse_launch_configs(args)
    for model, seq, cp in cases(args):
        config = configs[key(model, seq, cp)]
        for launch in args.fuse_launches:
            output = fuse_launch_path(args, model, seq, cp, launch)
            if args.resume and fuse_launch_result_matches(
                output,
                model=model,
                seq=seq,
                cp=cp,
                launch=launch,
                config=config,
                warmup=args.formal_warmup,
                iters=args.formal_iters,
            ):
                continue
            run(
                fuse_command(
                    model,
                    seq,
                    cp,
                    args.formal_warmup,
                    args.formal_iters,
                    output,
                    config["comm_ctas"],
                    cli_policy(config["policy"]),
                    launch == "graph",
                ),
                env=gpu_env(cp),
                output=output,
                resume=False,
            )
            if not fuse_launch_result_matches(
                output,
                model=model,
                seq=seq,
                cp=cp,
                launch=launch,
                config=config,
                warmup=args.formal_warmup,
                iters=args.formal_iters,
            ):
                raise RuntimeError(f"invalid OProj launch result: {output}")


def fuse_launch_aggregate(args: argparse.Namespace) -> None:
    configs = fuse_launch_configs(args)
    case_list = list(cases(args))
    use_formal_eager = all(
        fuse_launch_result_matches(
            fuse_launch_path(args, model, seq, cp, "eager"),
            model=model,
            seq=seq,
            cp=cp,
            launch="eager",
            config=configs[key(model, seq, cp)],
            warmup=args.formal_warmup,
            iters=args.formal_iters,
        )
        for model, seq, cp in case_list
    )
    eager_by_key = {}
    if not use_formal_eager:
        calibrated = load(args.results / "fused_calibrated_summary.json")
        eager_by_key = {
            (row["model"], row["global_seq"], row["cp"]): row
            for row in calibrated
        }
        if len(eager_by_key) != len(calibrated):
            raise ValueError("duplicate eager rows in fused_calibrated_summary.json")
    rows = []
    for model, seq, cp in case_list:
        config = configs[key(model, seq, cp)]
        graph_path = fuse_launch_path(args, model, seq, cp, "graph")
        if not fuse_launch_result_matches(
            graph_path,
            model=model,
            seq=seq,
            cp=cp,
            launch="graph",
            config=config,
            warmup=args.formal_warmup,
            iters=args.formal_iters,
        ):
            raise ValueError(f"invalid Graph result for {key(model, seq, cp)}")
        graph = load(graph_path)
        shape = graph["shape"]
        if use_formal_eager:
            eager = load(fuse_launch_path(args, model, seq, cp, "eager"))
            expected_shape = eager["shape"]
            eager_p50 = eager["results"]["fused"]["p50_ms"]
            eager_p95 = eager["results"]["fused"]["p95_ms"]
            pure_p50 = min(
                eager["results"]["cublas"]["p50_ms"],
                eager["results"]["cublaslt_autotuned"]["p50_ms"],
                eager["results"]["same_policy_cutlass"]["p50_ms"],
            )
            eager_mismatches = eager.get("correctness", {}).get(
                "exact_mismatches", eager.get("exact_mismatches", 0)
            )
            eager_source = "formal_10w50_eager"
            eager_policy = eager["lhs_policy"]["name"]
        else:
            eager = eager_by_key[(model.name, seq, cp)]
            expected_shape = {
                "m": eager["m"],
                "n": eager["n"],
                "k": eager["k"],
                "l": 1,
            }
            eager_p50 = eager["fused_p50_ms"]
            eager_p95 = eager["fused_p95_ms"]
            pure_p50 = (
                eager["fused_throughput_as_best_pure_percent"]
                * eager_p50
                / 100.0
            )
            eager_mismatches = eager.get("exact_mismatches", 0)
            eager_source = eager["result_source"]
            eager_policy = eager["policy"]
        if shape != expected_shape:
            raise ValueError(f"eager/graph shape mismatch for {key(model, seq, cp)}")
        if (
            eager["comm_ctas"] != config["comm_ctas"]
            or eager_policy != config["policy"]
        ):
            raise ValueError(
                f"eager/config mismatch for {key(model, seq, cp)}"
            )
        flops = 2.0 * shape["m"] * shape["n"] * shape["k"]
        graph_p50 = graph["results"]["fused"]["p50_ms"]
        rows.append({
            "model": model.name,
            "suite": model.suite,
            "aliases": model.aliases,
            "global_seq": seq,
            "cp": cp,
            "m": shape["m"],
            "n": shape["n"],
            "k": shape["k"],
            "q_heads": model.q_heads,
            "head_dim": model.head_dim,
            "comm_ctas": config["comm_ctas"],
            "policy": config["policy"],
            "raster": config["raster"],
            "swizzle": config["swizzle"],
            "warmup": args.formal_warmup,
            "iterations": args.formal_iters,
            "process_model": "single_process_multi_gpu",
            "timing": "per_sample_max_rank_cuda_event",
            "eager_result_source": eager_source,
            "graph_result_source": "formal_10w50_cuda_graph_preuploaded",
            "graph_setup_timed": False,
            "graph_epoch_mode": graph["cuda_graph_epoch_mode"],
            "eager_p50_ms": eager_p50,
            "eager_p95_ms": eager_p95,
            "eager_p50_tflops_per_gpu": (
                flops / eager_p50 / 1.0e9
            ),
            "graph_p50_ms": graph_p50,
            "graph_p95_ms": graph["results"]["fused"]["p95_ms"],
            "graph_p50_tflops_per_gpu": (
                flops / graph_p50 / 1.0e9
            ),
            "best_pure_gemm_p50_ms": pure_p50,
            "eager_throughput_as_best_pure_percent": (
                100.0 * pure_p50 / eager_p50
            ),
            "graph_throughput_as_best_pure_percent": (
                100.0 * pure_p50 / graph_p50
            ),
            "exact_mismatches_eager": eager_mismatches,
            "exact_mismatches_graph": graph.get("correctness", {}).get(
                "exact_mismatches", graph.get("exact_mismatches", 0)
            ),
        })
    write_rows(rows, args.results / "fused_launch_summary")


def fuse_aggregate(args: argparse.Namespace) -> None:
    rows = []
    for model, seq, cp in cases(args):
        formal = args.results / "fuse_formal" / (
            f"{key(model, seq, cp)}_{args.formal_warmup}w{args.formal_iters}i.json"
        )
        if not formal.exists():
            raise FileNotFoundError(
                f"missing fused formal result for {key(model, seq, cp)}"
            )
        data = load(formal)
        shape = data["shape"]
        flops = 2.0 * shape["m"] * shape["n"] * shape["k"]

        def p50_tflops(name: str) -> float:
            return flops / data["results"][name]["p50_ms"] / 1.0e9

        fused = data["results"]["fused"]
        cublas = data["results"]["cublas"]
        cublaslt = data["results"]["cublaslt_autotuned"]
        pure_best_p50 = min(cublas["p50_ms"], cublaslt["p50_ms"])
        policy = data["lhs_policy"]
        rows.append({
            "model": model.name,
            "suite": model.suite,
            "aliases": model.aliases,
            "global_seq": seq,
            "cp": cp,
            "m": shape["m"],
            "n": shape["n"],
            "k": shape["k"],
            "q_heads": model.q_heads,
            "head_dim": model.head_dim,
            "native_max_context": model.max_context,
            "beyond_native_context": (
                model.max_context is not None and seq > model.max_context
            ),
            "comm_ctas": data["comm_ctas"],
            "policy": policy["name"],
            "tile_m": policy["tile_m"],
            "tile_n": policy["tile_n"],
            "tile_k": policy["tile_k"],
            "cluster_m": policy["cluster_m"],
            "compute_clusters": policy["compute_clusters"],
            "n_tiles": policy["n_tiles"],
            "waves": policy["waves"],
            "last_wave_clusters": policy["last_wave_clusters"],
            "last_wave_ctas": policy["last_wave_ctas"],
            "frontier_aligned": policy["frontier_aligned"],
            "full_last_wave": policy["full_last_wave"],
            "fused_mean_ms": fused["mean_ms"],
            "fused_p50_ms": fused["p50_ms"],
            "fused_p95_ms": fused["p95_ms"],
            "fused_p50_tflops_per_gpu": p50_tflops("fused"),
            "cublas_p50_ms": cublas["p50_ms"],
            "cublas_p50_tflops_per_gpu": p50_tflops("cublas"),
            "cublaslt_p50_ms": cublaslt["p50_ms"],
            "cublaslt_p50_tflops_per_gpu": p50_tflops("cublaslt_autotuned"),
            "fused_throughput_as_cublas_percent":
                100.0 * cublas["p50_ms"] / fused["p50_ms"],
            "fused_throughput_as_best_pure_percent":
                100.0 * pure_best_p50 / fused["p50_ms"],
            "same_policy_p50_ms":
                data["results"]["same_policy_cutlass"]["p50_ms"],
            "compute_subgrid_p50_ms":
                data["results"]["compute_subgrid_cutlass"]["p50_ms"],
            "route_p50_ms": data["results"]["inverse_a2a_route"]["p50_ms"],
            "sequential_p50_ms":
                data["results"]["same_policy_sequential"]["p50_ms"],
            "overlap_ratio": data["overlap_ratio"],
        })
    args.results.mkdir(parents=True, exist_ok=True)
    with (args.results / "fused_summary.json").open("w") as handle:
        json.dump(rows, handle, indent=2)
    fields = list(rows[0])
    with (args.results / "fused_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def nccl_configs(args: argparse.Namespace) -> list[tuple[int, int, int]]:
    return [
        (channels, chunk, ll)
        for channels in args.nccl_channels
        for chunk in args.nccl_chunk_kib
        for ll in args.nccl_ll_kib
    ]


def external_matrix_command(
    args: argparse.Namespace, cp: int, warmup: int, iters: int
) -> list[str]:
    command = [
        str(args.torchrun), "--standalone", f"--nproc-per-node={cp}",
        str(Path(__file__).with_name("te_nccl_baseline.py")),
        "--mode", "oproj_a2a_gemm",
        "--global-seq", "1024",
        "--hidden", "4096",
        "--batch", "1",
        "--q-heads", "32",
        "--head-dim", "128",
        "--warmup", str(warmup), "--iters", str(iters),
        "--cublaslt-tune-warmup", str(args.cublaslt_tune_warmup),
        "--cublaslt-tune-iters", str(args.cublaslt_tune_iters),
    ]
    return command


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
    model: Model,
    seq: int,
    cp: int,
    config: tuple[int, int, int],
    cuda_graph: bool,
    high_priority: bool,
    metric_profile: str,
    pack_block: int = 1024,
    pack_warps: int = 4,
) -> dict[str, object]:
    return {
        "json_out": str(output.resolve()),
        "environment": nccl_environment_values(config),
        "cuda_graph": cuda_graph,
        "nccl_high_priority": high_priority,
        "arguments": {
            "global_seq": seq,
            "hidden": model.hidden,
            "batch": 1,
            "q_heads": model.q_heads,
            "head_dim": model.head_dim,
            "metric_profile": metric_profile,
            "pack_block": pack_block,
            "pack_warps": pack_warps,
            "check": (
                seq * (model.q_heads // cp) * model.head_dim < (1 << 30)
            ),
        },
    }


def spec_label(config: tuple[int, int, int], cuda_graph: bool, high_priority: bool) -> str:
    channels, chunk, ll = config
    return (
        f"ch{channels}_chunk{chunk}_ll{ll}_"
        f"{'graph' if cuda_graph else 'eager'}_"
        f"{'hp' if high_priority else 'normal'}"
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


def best_specs(files: list[Path]):
    if not files or any(not path.exists() for path in files):
        raise FileNotFoundError("missing expected sweep result")
    winners: dict[str, tuple[tuple[int, int, int], bool, bool]] = {}
    for metric in BOUNDARY_METRICS:
        winner = min(
            (load(path) for path in files),
            key=lambda item: item["results"][metric]["mean_ms"],
        )
        winners[metric] = result_spec(winner)
    return winners


def nccl_sweep_path(
    args: argparse.Namespace,
    model: Model,
    seq: int,
    cp: int,
    config: tuple[int, int, int],
) -> Path:
    return args.results / "baseline_nccl_sweep" / key(model, seq, cp) / (
        f"{spec_label(config, True, True)}_"
        f"{args.sweep_warmup}w{args.sweep_iters}i.json"
    )


def mode_sweep_path(
    args: argparse.Namespace,
    model: Model,
    seq: int,
    cp: int,
    spec: tuple[tuple[int, int, int], bool, bool],
) -> Path:
    config, graph, priority = spec
    return args.results / "baseline_mode_sweep" / key(model, seq, cp) / (
        f"{spec_label(config, graph, priority)}_"
        f"{args.sweep_warmup}w{args.sweep_iters}i.json"
    )


def formal_path(
    args: argparse.Namespace,
    model: Model,
    seq: int,
    cp: int,
    spec: tuple[tuple[int, int, int], bool, bool, int, int],
) -> Path:
    return args.results / "baseline_formal" / (
        f"{key(model, seq, cp)}_{pack_spec_label(spec)}_"
        f"{args.formal_warmup}w{args.formal_iters}i.json"
    )


def shortlisted_nccl_configs(
    args: argparse.Namespace, model: Model, seq: int, cp: int
) -> set[tuple[int, int, int]]:
    paths = [nccl_sweep_path(args, model, seq, cp, config) for config in nccl_configs(args)]
    if any(not path.exists() for path in paths):
        raise FileNotFoundError("missing expected NCCL sweep result")
    results = [load(path) for path in paths]
    ranked = sorted(
        results, key=lambda item: item["results"]["te_nccl_inverse_a2a"]["mean_ms"]
    )
    return {
        result_spec(item)[0] for item in ranked[: args.nccl_shortlist]
    }


def best_mode_specs(args: argparse.Namespace, model: Model, seq: int, cp: int):
    paths = [
        mode_sweep_path(args, model, seq, cp, (config, graph, priority))
        for config in sorted(shortlisted_nccl_configs(args, model, seq, cp))
        for graph, priority in EXECUTION_MODES
    ]
    return best_specs(paths)


def pack_spec_label(
    spec: tuple[tuple[int, int, int], bool, bool, int, int]
) -> str:
    config, graph, priority, pack_block, pack_warps = spec
    return (
        f"{spec_label(config, graph, priority)}_"
        f"pb{pack_block}_pw{pack_warps}"
    )


def pack_sweep_path(
    args: argparse.Namespace,
    model: Model,
    seq: int,
    cp: int,
    spec: tuple[tuple[int, int, int], bool, bool, int, int],
) -> Path:
    return args.results / "baseline_pack_sweep" / key(model, seq, cp) / (
        f"{pack_spec_label(spec)}_{args.sweep_warmup}w{args.sweep_iters}i.json"
    )


def pack_candidates(
    args: argparse.Namespace, model: Model, seq: int, cp: int
) -> list[tuple[tuple[int, int, int], bool, bool, int, int]]:
    mode_specs = sorted(set(best_mode_specs(args, model, seq, cp).values()))
    return [
        (config, graph, priority, pack_block, pack_warps)
        for config, graph, priority in mode_specs
        for pack_block in args.pack_blocks
        for pack_warps in args.pack_warps
    ]


def best_pack_specs(args: argparse.Namespace, model: Model, seq: int, cp: int):
    candidates = pack_candidates(args, model, seq, cp)
    paths = [pack_sweep_path(args, model, seq, cp, spec) for spec in candidates]
    if not paths or any(not path.exists() for path in paths):
        raise FileNotFoundError("missing expected pack sweep result")
    results = [load(path) for path in paths]
    winners = {}
    for metric in BOUNDARY_METRICS:
        winner = min(
            results, key=lambda item: item["results"][metric]["mean_ms"]
        )
        config, graph, priority = result_spec(winner)
        winners[metric] = (
            config,
            graph,
            priority,
            int(winner["pack_block"]),
            int(winner["pack_warps"]),
        )
    return winners


def run_grouped_external(
    args: argparse.Namespace,
    *,
    phase: str,
    groups: dict[tuple[int, tuple[int, int, int]], list[dict[str, object]]],
    warmup: int,
    iters: int,
) -> None:
    for (cp, config), entries in sorted(groups.items()):
        environment = gpu_env(cp)
        environment.update(nccl_environment_values(config))
        config_name = f"ch{config[0]}_chunk{config[1]}_ll{config[2]}"
        run_matrix(
            external_matrix_command(args, cp, warmup, iters),
            env=environment,
            entries=entries,
            manifest=args.results / "manifests" / phase / f"cp{cp}_{config_name}.json",
            resume=args.resume,
            cp=cp,
            warmup=warmup,
            iters=iters,
        )


def run_baseline_nccl_sweep(args: argparse.Namespace) -> None:
    groups: dict[tuple[int, tuple[int, int, int]], list[dict[str, object]]] = {}
    for model, seq, cp in cases(args):
        for config in nccl_configs(args):
            output = nccl_sweep_path(args, model, seq, cp, config)
            groups.setdefault((cp, config), []).append(
                matrix_entry(output, model, seq, cp, config, True, True, "route")
            )
    run_grouped_external(
        args,
        phase="baseline_nccl_sweep",
        groups=groups,
        warmup=args.sweep_warmup,
        iters=args.sweep_iters,
    )


def run_baseline_mode_sweep(args: argparse.Namespace) -> None:
    groups: dict[tuple[int, tuple[int, int, int]], list[dict[str, object]]] = {}
    for model, seq, cp in cases(args):
        for config in sorted(shortlisted_nccl_configs(args, model, seq, cp)):
            for graph, priority in EXECUTION_MODES:
                spec = (config, graph, priority)
                output = mode_sweep_path(args, model, seq, cp, spec)
                groups.setdefault((cp, config), []).append(
                    matrix_entry(
                        output, model, seq, cp, config, graph, priority, "boundary"
                    )
                )
    run_grouped_external(
        args,
        phase="baseline_mode_sweep",
        groups=groups,
        warmup=args.sweep_warmup,
        iters=args.sweep_iters,
    )


def run_baseline_pack_sweep(args: argparse.Namespace) -> None:
    groups: dict[tuple[int, tuple[int, int, int]], list[dict[str, object]]] = {}
    for model, seq, cp in cases(args):
        for spec in pack_candidates(args, model, seq, cp):
            config, graph, priority, pack_block, pack_warps = spec
            output = pack_sweep_path(args, model, seq, cp, spec)
            groups.setdefault((cp, config), []).append(
                matrix_entry(
                    output,
                    model,
                    seq,
                    cp,
                    config,
                    graph,
                    priority,
                    "boundary",
                    pack_block,
                    pack_warps,
                )
            )
    run_grouped_external(
        args,
        phase="baseline_pack_sweep",
        groups=groups,
        warmup=args.sweep_warmup,
        iters=args.sweep_iters,
    )


def run_baseline_formal(args: argparse.Namespace) -> None:
    groups: dict[tuple[int, tuple[int, int, int]], list[dict[str, object]]] = {}
    for model, seq, cp in cases(args):
        specs = sorted(set(best_pack_specs(args, model, seq, cp).values()))
        for config, cuda_graph, high_priority, pack_block, pack_warps in specs:
            spec = (
                config, cuda_graph, high_priority, pack_block, pack_warps
            )
            output = formal_path(args, model, seq, cp, spec)
            groups.setdefault((cp, config), []).append(
                matrix_entry(
                    output,
                    model,
                    seq,
                    cp,
                    config,
                    cuda_graph,
                    high_priority,
                    "full",
                    pack_block,
                    pack_warps,
                )
            )
    run_grouped_external(
        args,
        phase="baseline_formal",
        groups=groups,
        warmup=args.formal_warmup,
        iters=args.formal_iters,
    )


def formal_result(
    args: argparse.Namespace,
    model: Model,
    seq: int,
    cp: int,
    metric: str,
) -> dict:
    spec = best_pack_specs(args, model, seq, cp)[metric]
    return load(formal_path(args, model, seq, cp, spec))


def baseline_aggregate(args: argparse.Namespace) -> None:
    rows = []
    for model, seq, cp in cases(args):
        te = formal_result(args, model, seq, cp, BOUNDARY_METRICS[0])
        lt = formal_result(args, model, seq, cp, BOUNDARY_METRICS[1])
        te_boundary = te["results"][BOUNDARY_METRICS[0]]
        lt_boundary = lt["results"][BOUNDARY_METRICS[1]]
        row = {
            "model": model.name, "suite": model.suite,
            "aliases": model.aliases, "cp": cp,
            "global_seq": seq, "m": seq // cp, "n": model.hidden,
            "k": model.attention_width, "q_heads": model.q_heads,
            "head_dim": model.head_dim,
            "native_max_context": model.max_context,
            "beyond_native_context": (
                model.max_context is not None and seq > model.max_context
            ),
            "te_gemm_p50_ms": te["results"]["te_oproj_gemm"]["p50_ms"],
            "cublaslt_gemm_p50_ms": lt["results"]["cublaslt_oproj_gemm"]["p50_ms"],
            "te_route_p50_ms": te["results"]["te_nccl_inverse_a2a"]["p50_ms"],
            "cublaslt_route_p50_ms": lt["results"]["te_nccl_inverse_a2a"]["p50_ms"],
            "te_boundary_p50_ms": te_boundary["p50_ms"],
            "cublaslt_boundary_p50_ms": lt_boundary["p50_ms"],
            "best_separated": (
                "TE+NCCL"
                if te_boundary["p50_ms"] <= lt_boundary["p50_ms"]
                else "cuBLASLt+NCCL"
            ),
            "best_separated_p50_ms": min(
                te_boundary["p50_ms"], lt_boundary["p50_ms"]
            ),
            "te_nccl_config": pack_spec_label((
                *result_spec(te), int(te["pack_block"]), int(te["pack_warps"])
            )),
            "cublaslt_nccl_config": pack_spec_label((
                *result_spec(lt), int(lt["pack_block"]), int(lt["pack_warps"])
            )),
            "cublaslt_plans": lt["cublaslt_plans"],
        }
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
            "suite": model.suite,
            "aliases": model.aliases,
            "global_seq": seq,
            "cp": cp,
            "m": m,
            "n": n,
            "k": k,
            "q_heads": model.q_heads,
            "head_dim": model.head_dim,
            "q_heads_per_rank": model.q_heads // cp,
            "native_max_context": model.max_context,
            "beyond_native_context": (
                model.max_context is not None and seq > model.max_context
            ),
            "a2a_input": f"[1,{seq},{model.q_heads // cp},{model.head_dim}]",
            "gemm_input": f"[{m},{k}]",
            "gemm_weight": f"[{n},{k}]",
            "gemm_output": f"[{m},{n}]",
            "payload_mib_per_gpu": m * k * 2 / (1 << 20),
            "remote_payload_mib_per_gpu": (
                m * k * 2 * (cp - 1) / cp / (1 << 20)
            ),
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
            "| Suite | Model | S | CP | MxNxK | Hq/D | Context | Remote MiB/GPU | GFLOP/GPU |\n"
            "|---|---|---:|---:|---|---|---|---:|---:|\n"
        )
        for row in rows:
            context = "stress>native" if row["beyond_native_context"] else "native/unspecified"
            handle.write(
                f"| {row['suite']} | {row['model']} | {row['global_seq']} | "
                f"{row['cp']} | {row['m']}x{row['n']}x{row['k']} | "
                f"{row['q_heads']}/{row['head_dim']} | {context} | "
                f"{row['remote_payload_mib_per_gpu']:.2f} | "
                f"{row['gemm_gflop_per_gpu']:.3f} |\n"
            )


def comparison_table(args: argparse.Namespace) -> None:
    launch_summary = args.results / "fused_launch_summary.json"
    if not launch_summary.exists():
        raise FileNotFoundError(
            "missing fused_launch_summary.json; run "
            "--phase fuse-launch-aggregate first"
        )
    fused = load(launch_summary)
    baseline = load(args.results / "baseline_summary.json")
    userbuffers = (
        load(args.te_userbuffers_summary)
        if args.te_userbuffers_summary.exists()
        else []
    )
    baseline_by_key = {
        (row["model"], row["global_seq"], row["cp"]): row for row in baseline
    }
    userbuffers_by_key = {
        (row["model"], row["global_seq"], row["cp"]): row
        for row in userbuffers
    }
    rows = []
    for row in fused:
        other = baseline_by_key[(row["model"], row["global_seq"], row["cp"])]
        merged = dict(row)
        merged.update({
            "best_separated": other["best_separated"],
            "best_separated_p50_ms": other["best_separated_p50_ms"],
            "eager_speedup_over_best_separated": (
                other["best_separated_p50_ms"] / row["eager_p50_ms"]
            ),
            "graph_speedup_over_best_separated": (
                other["best_separated_p50_ms"] / row["graph_p50_ms"]
            ),
            "te_boundary_p50_ms": other["te_boundary_p50_ms"],
            "cublaslt_boundary_p50_ms": other["cublaslt_boundary_p50_ms"],
        })
        ub = userbuffers_by_key.get(
            (row["model"], row["global_seq"], row["cp"])
        )
        if ub is not None:
            ub_ms = ub["p50_ms"]
            best_external_ms = min(other["best_separated_p50_ms"], ub_ms)
            merged.update({
                "te_userbuffers_p50_ms": ub_ms,
                "te_userbuffers_p95_ms": ub["p95_ms"],
                "te_userbuffers_launch": ub["launch"],
                "te_userbuffers_timing": ub["timing"],
                "te_userbuffers_timed_boundary": ub["timed_boundary"],
                "eager_speedup_over_te_userbuffers": (
                    ub_ms / row["eager_p50_ms"]
                ),
                "graph_speedup_over_te_userbuffers": (
                    ub_ms / row["graph_p50_ms"]
                ),
                "best_external_p50_ms": best_external_ms,
                "eager_speedup_over_best_external": (
                    best_external_ms / row["eager_p50_ms"]
                ),
                "graph_speedup_over_best_external": (
                    best_external_ms / row["graph_p50_ms"]
                ),
            })
        rows.append(merged)
    with (args.results / "comparison_summary.json").open("w") as handle:
        json.dump(rows, handle, indent=2)
    fields = sorted({field for row in rows for field in row})
    with (args.results / "comparison_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (args.results / "comparison_summary.md").open("w") as handle:
        handle.write(
            "| Suite | Model | S | CP | MxNxK | Eager p50/p95 / TFLOPS | "
            "Graph p50/p95 / TFLOPS | Best separated ms | TE-UB p50/p95 | "
            "Eager / external | Graph / external | Config |\n"
            "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|\n"
        )
        for row in rows:
            handle.write(
                f"| {row['suite']} | {row['model']} | {row['global_seq']} | "
                f"{row['cp']} | {row['m']}x{row['n']}x{row['k']} | "
                f"{row['eager_p50_ms']:.4f}/{row['eager_p95_ms']:.4f} / "
                f"{row['eager_p50_tflops_per_gpu']:.1f} | "
                f"{row['graph_p50_ms']:.4f}/{row['graph_p95_ms']:.4f} / "
                f"{row['graph_p50_tflops_per_gpu']:.1f} | "
                f"{row['best_separated_p50_ms']:.4f} | "
                f"{row.get('te_userbuffers_p50_ms', float('nan')):.4f}/"
                f"{row.get('te_userbuffers_p95_ms', float('nan')):.4f} | "
                f"{row.get('eager_speedup_over_best_external', row['eager_speedup_over_best_separated']):.3f}x | "
                f"{row.get('graph_speedup_over_best_external', row['graph_speedup_over_best_separated']):.3f}x | "
                f"c{row['comm_ctas']}/{row['policy']}/r{row['raster'].upper()}/s{row['swizzle']} |\n"
            )


def main() -> None:
    args = parse_args()
    for phase in args.phase:
        {
            "baseline-nccl-sweep": run_baseline_nccl_sweep,
            "baseline-mode-sweep": run_baseline_mode_sweep,
            "baseline-pack-sweep": run_baseline_pack_sweep,
            "baseline-formal": run_baseline_formal,
            "baseline-aggregate": baseline_aggregate,
            "shape-table": write_shape_table,
            "fuse-formal": run_fuse_formal,
            "fuse-aggregate": fuse_aggregate,
            "fuse-comm-sweep": run_fuse_comm_sweep,
            "fuse-comm-refine": run_fuse_comm_refine,
            "fuse-comm-formal": run_fuse_comm_formal,
            "fuse-calibrated-aggregate": fuse_calibrated_aggregate,
            "fuse-launch-formal": run_fuse_launch_formal,
            "fuse-launch-aggregate": fuse_launch_aggregate,
            "comparison-table": comparison_table,
        }[phase](args)


if __name__ == "__main__":
    main()
