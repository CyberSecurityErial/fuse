#!/usr/bin/env python3
"""Reproducible QKV projection + Ulysses A2A benchmark matrix."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "results" / "QKVproj-a2a" / "qkv_shape_bench"
DEFAULT_TORCHRUN = Path("/home/chen/miniforge3/envs/mmunlearner/bin/torchrun")


@dataclass(frozen=True)
class Model:
    name: str
    suite: str
    hidden: int
    q_heads: int
    kv_heads: int
    head_dim: int
    aliases: str
    max_context: int | None = None

    @property
    def qkv_width(self) -> int:
        return (self.q_heads + 2 * self.kv_heads) * self.head_dim


# The artificial suite retains the three OProj MNK stress points while giving
# them valid GQA head geometry.  The model suite uses real open-source shapes.
MODELS = {
    model.name: model
    for model in (
        Model(
            "artificial_small", "artificial", 4096, 16, 8, 128,
            "OProj-matched Mx4096x4096 stress shape",
        ),
        Model(
            "artificial_medium", "artificial", 5120, 24, 8, 128,
            "OProj-matched Mx5120x5120 stress shape",
        ),
        Model(
            "artificial_large", "artificial", 16384, 40, 8, 128,
            "OProj-matched Mx7168x16384 stress shape",
        ),
        Model(
            "production_qwen_dense", "model", 2048, 16, 8, 128,
            "current production Qwen dense",
        ),
        Model(
            "nanbeige42_3b", "model", 3072, 48, 8, 128,
            "Nanbeige/Nanbeige4.2-3B", 262144,
        ),
        Model(
            "llama3_8b", "model", 4096, 32, 8, 128,
            "Llama 3/3.1 8B",
        ),
        Model(
            "qwen25_14b_32b", "model", 5120, 40, 8, 128,
            "Qwen2.5 14B/32B; Llama 4 uses the same QKV shape",
        ),
        Model(
            "llama31_405b", "model", 16384, 128, 8, 128,
            "Llama 3.1 405B",
        ),
    )
}
DEFAULT_MODELS = tuple(MODELS)
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


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        action="append",
        choices=(
            "shape-table",
            "baseline-nccl-sweep",
            "baseline-mode-sweep",
            "baseline-formal",
            "baseline-aggregate",
            "fuse-formal",
            "fuse-aggregate",
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
        default=ROOT / "results" / "QKVproj-a2a"
        / "te_userbuffers_shape_bench" / "summary.json",
    )
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


def dump_rows(rows: list[dict[str, object]], stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    with stem.with_suffix(".json").open("w") as handle:
        json.dump(rows, handle, indent=2)
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


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
            result.get("benchmark_schema") == "qkv_full_qkv_packed_v1"
            and result["world_size"] == cp
            and result["warmup"] == warmup
            and result["iterations"] == iters
            and result["cuda_graph"] == entry["cuda_graph"]
            and result["nccl_high_priority"] == entry["nccl_high_priority"]
            and result["include_source"] is False
            and result["include_qk"] is False
            and result["metric_profile"] == arguments["metric_profile"]
            and result["pack_block"] == arguments["pack_block"]
            and result["pack_warps"] == arguments["pack_warps"]
            and shape["global_seq"] == arguments["global_seq"]
            and shape["hidden"] == arguments["hidden"]
            and shape["q_heads"] == arguments["q_heads"]
            and shape["kv_heads"] == arguments["kv_heads"]
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
        rendezvous = Path(f"/tmp/fuse_qkv_rendezvous_{os.getpid()}_{index}")
        rendezvous.unlink(missing_ok=True)
        entry["rendezvous_file"] = str(rendezvous)
        rendezvous_files.append(rendezvous)
    manifest.write_text(json.dumps(pending, indent=2) + "\n")
    print(f"RUN-MATRIX {manifest} ({len(pending)} configurations)", flush=True)
    completed = subprocess.run(
        command + ["--matrix-manifest", str(manifest)],
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
                if (
                    cp not in VISIBLE_DEVICES
                    or seq % cp
                    or model.q_heads % cp
                    or model.kv_heads % cp
                    or model.q_heads % model.kv_heads
                ):
                    continue
                yield model, seq, cp


def gpu_env(cp: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = VISIBLE_DEVICES[cp]
    return env


def fuse_command(
    model: Model,
    seq: int,
    cp: int,
    warmup: int,
    iters: int,
    output: Path,
) -> list[str]:
    return [
        str(ROOT / "build" / "qkvproj_a2a_bench"),
        "--mode", "qkv_gemm_a2a",
        "--m", str(seq // cp),
        "--k", str(model.hidden),
        "--batch", "1",
        "--q-heads", str(model.q_heads),
        "--kv-heads", str(model.kv_heads),
        "--head-dim", str(model.head_dim),
        "--comm-ctas", "0",
        "--raster", "n",
        "--swizzle", "1",
        "--warmup", str(warmup),
        "--iterations", str(iters),
        "--json-out", str(output),
    ]


def external_command(
    args: argparse.Namespace, cp: int, warmup: int, iters: int
) -> list[str]:
    command = [
        str(args.torchrun), "--standalone", f"--nproc-per-node={cp}",
        str(Path(__file__).with_name("te_nccl_baseline.py")),
        "--mode", "qkv_gemm_a2a",
        "--global-seq", "1024",
        "--hidden", "4096",
        "--batch", "1",
        "--q-heads", "16",
        "--kv-heads", "8",
        "--head-dim", "128",
        "--warmup", str(warmup),
        "--iters", str(iters),
        "--cublaslt-tune-warmup", str(args.cublaslt_tune_warmup),
        "--cublaslt-tune-iters", str(args.cublaslt_tune_iters),
        "--no-include-source",
        "--no-include-qk",
    ]
    return command


def nccl_configs(args: argparse.Namespace) -> list[tuple[int, int, int]]:
    return [
        (channels, chunk, ll)
        for channels in args.nccl_channels
        for chunk in args.nccl_chunk_kib
        for ll in args.nccl_ll_kib
    ]


def nccl_environment(config: tuple[int, int, int]) -> dict[str, str]:
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
    pack_block: int,
    pack_warps: int,
    metric_profile: str,
) -> dict[str, object]:
    return {
        "json_out": str(output.resolve()),
        "environment": nccl_environment(config),
        "cuda_graph": cuda_graph,
        "nccl_high_priority": high_priority,
        "arguments": {
            "global_seq": seq,
            "hidden": model.hidden,
            "batch": 1,
            "q_heads": model.q_heads,
            "kv_heads": model.kv_heads,
            "head_dim": model.head_dim,
            "pack_backend": "triton",
            "pack_block": pack_block,
            "pack_warps": pack_warps,
            "metric_profile": metric_profile,
            "check": (seq // cp) * model.qkv_width < (1 << 30),
        },
    }


def spec_label(
    config: tuple[int, int, int],
    graph: bool,
    priority: bool,
    pack_block: int,
    pack_warps: int,
) -> str:
    channels, chunk, ll = config
    return (
        f"ch{channels}_chunk{chunk}_ll{ll}_"
        f"{'graph' if graph else 'eager'}_"
        f"{'hp' if priority else 'normal'}_pb{pack_block}_pw{pack_warps}"
    )


BOUNDARY_METRICS = (
    "te_packed_qkv_gemm_a2a",
    "cublas_packed_qkv_gemm_a2a",
    "cublaslt_packed_qkv_gemm_a2a",
)


def result_spec(
    result: dict,
) -> tuple[tuple[int, int, int], bool, bool, int, int]:
    env = result["environment"]
    return (
        (
            int(env["NCCL_MIN_P2P_NCHANNELS"]),
            int(env["NCCL_P2P_NVL_CHUNKSIZE"]) // 1024,
            int(env["NCCL_P2P_LL_THRESHOLD"]) // 1024,
        ),
        bool(result["cuda_graph"]),
        bool(result["nccl_high_priority"]),
        int(result["pack_block"]),
        int(result["pack_warps"]),
    )


def best_specs(
    files: list[Path],
) -> dict[str, tuple[tuple[int, int, int], bool, bool, int, int]]:
    if not files or any(not path.exists() for path in files):
        raise FileNotFoundError("missing expected sweep result")
    winners = {}
    for metric in BOUNDARY_METRICS:
        candidates = [load(path) for path in files]
        candidates = [item for item in candidates if metric in item["results"]]
        winners[metric] = result_spec(
            min(candidates, key=lambda item: item["results"][metric]["mean_ms"])
        )
    return winners


def nccl_sweep_path(
    args: argparse.Namespace,
    model: Model,
    seq: int,
    cp: int,
    config: tuple[int, int, int],
) -> Path:
    return args.results / "baseline_nccl_sweep" / key(model, seq, cp) / (
        f"{spec_label(config, True, True, 1024, 4)}_"
        f"{args.sweep_warmup}w{args.sweep_iters}i.json"
    )


def mode_sweep_path(
    args: argparse.Namespace,
    model: Model,
    seq: int,
    cp: int,
    spec: tuple[tuple[int, int, int], bool, bool, int, int],
) -> Path:
    config, graph, priority, pack_block, pack_warps = spec
    return args.results / "baseline_mode_sweep" / key(model, seq, cp) / (
        f"{spec_label(config, graph, priority, pack_block, pack_warps)}_"
        f"{args.sweep_warmup}w{args.sweep_iters}i.json"
    )


def formal_path(
    args: argparse.Namespace,
    model: Model,
    seq: int,
    cp: int,
    spec: tuple[tuple[int, int, int], bool, bool, int, int],
) -> Path:
    config, graph, priority, pack_block, pack_warps = spec
    return args.results / "baseline_formal" / (
        f"{key(model, seq, cp)}_"
        f"{spec_label(config, graph, priority, pack_block, pack_warps)}_"
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
        results, key=lambda item: item["results"]["packed_qkv_a2a"]["mean_ms"]
    )
    return {
        result_spec(item)[0]
        for item in ranked[: args.nccl_shortlist]
    }


def best_mode_specs(args: argparse.Namespace, model: Model, seq: int, cp: int):
    paths = [
        mode_sweep_path(
            args,
            model,
            seq,
            cp,
            (config, graph, priority, pack_block, pack_warps),
        )
        for config in sorted(shortlisted_nccl_configs(args, model, seq, cp))
        for graph, priority in EXECUTION_MODES
        for pack_block in args.pack_blocks
        for pack_warps in args.pack_warps
    ]
    return best_specs(paths)


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
        environment.update(nccl_environment(config))
        config_name = f"ch{config[0]}_chunk{config[1]}_ll{config[2]}"
        run_matrix(
            external_command(args, cp, warmup, iters),
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
                matrix_entry(
                    output, model, seq, cp, config, True, True, 1024, 4, "route"
                )
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
                for pack_block in args.pack_blocks:
                    for pack_warps in args.pack_warps:
                        spec = (
                            config,
                            graph,
                            priority,
                            pack_block,
                            pack_warps,
                        )
                        output = mode_sweep_path(args, model, seq, cp, spec)
                        groups.setdefault((cp, config), []).append(
                            matrix_entry(
                                output,
                                model,
                                seq,
                                cp,
                                config,
                                graph,
                                priority,
                                pack_block,
                                pack_warps,
                                "boundary",
                            )
                        )
    run_grouped_external(
        args,
        phase="baseline_mode_sweep",
        groups=groups,
        warmup=args.sweep_warmup,
        iters=args.sweep_iters,
    )


def run_baseline_formal(args: argparse.Namespace) -> None:
    groups: dict[tuple[int, tuple[int, int, int]], list[dict[str, object]]] = {}
    for model, seq, cp in cases(args):
        specs = sorted(set(best_mode_specs(args, model, seq, cp).values()))
        for config, graph, priority, pack_block, pack_warps in specs:
            spec = (config, graph, priority, pack_block, pack_warps)
            output = formal_path(args, model, seq, cp, spec)
            groups.setdefault((cp, config), []).append(
                matrix_entry(
                    output,
                    model,
                    seq,
                    cp,
                    config,
                    graph,
                    priority,
                    pack_block,
                    pack_warps,
                    "full",
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
    spec = best_mode_specs(args, model, seq, cp)[metric]
    return load(formal_path(args, model, seq, cp, spec))


def baseline_aggregate(args: argparse.Namespace) -> None:
    rows = []
    for model, seq, cp in cases(args):
        formal = {
            metric: formal_result(args, model, seq, cp, metric)
            for metric in BOUNDARY_METRICS
        }
        boundaries = {
            metric: result["results"][metric]
            for metric, result in formal.items()
        }
        best_metric = min(
            BOUNDARY_METRICS,
            key=lambda metric: boundaries[metric]["p50_ms"],
        )
        te = formal["te_packed_qkv_gemm_a2a"]
        cublas = formal["cublas_packed_qkv_gemm_a2a"]
        lt = formal["cublaslt_packed_qkv_gemm_a2a"]
        row = {
            "model": model.name,
            "suite": model.suite,
            "aliases": model.aliases,
            "global_seq": seq,
            "cp": cp,
            "m": seq // cp,
            "n": model.qkv_width,
            "k": model.hidden,
            "q_heads": model.q_heads,
            "kv_heads": model.kv_heads,
            "head_dim": model.head_dim,
            "max_context": model.max_context,
            "within_native_context": (
                model.max_context is None or seq <= model.max_context
            ),
            "te_gemm_p50_ms": te["results"]["te_qkv_gemm"]["p50_ms"],
            "cublas_p50_ms": cublas["results"]["cublas_qkv_gemm"]["p50_ms"],
            "cublaslt_p50_ms": lt["results"]["cublaslt_qkv_gemm"]["p50_ms"],
            "te_qkv_route_p50_ms": te["results"]["packed_qkv_a2a"]["p50_ms"],
            "cublas_qkv_route_p50_ms": cublas["results"]["packed_qkv_a2a"]["p50_ms"],
            "cublaslt_qkv_route_p50_ms": lt["results"]["packed_qkv_a2a"]["p50_ms"],
            "te_boundary_p50_ms": boundaries["te_packed_qkv_gemm_a2a"]["p50_ms"],
            "cublas_boundary_p50_ms": boundaries["cublas_packed_qkv_gemm_a2a"]["p50_ms"],
            "cublaslt_boundary_p50_ms": boundaries["cublaslt_packed_qkv_gemm_a2a"]["p50_ms"],
            "best_separated": {
                "te_packed_qkv_gemm_a2a": "TE+NCCL",
                "cublas_packed_qkv_gemm_a2a": "cuBLAS+NCCL",
                "cublaslt_packed_qkv_gemm_a2a": "cuBLASLt+NCCL",
            }[best_metric],
            "best_separated_p50_ms": boundaries[best_metric]["p50_ms"],
            "te_nccl_config": spec_label(*result_spec(te)),
            "cublas_nccl_config": spec_label(*result_spec(cublas)),
            "cublaslt_nccl_config": spec_label(*result_spec(lt)),
            "cublaslt_plans": lt["cublaslt_plans"],
        }
        rows.append(row)
    dump_rows(rows, args.results / "baseline_summary")


def run_fuse_formal(args: argparse.Namespace) -> None:
    for model, seq, cp in cases(args):
        output = args.results / "fuse_formal" / (
            f"{key(model, seq, cp)}_{args.formal_warmup}w{args.formal_iters}i.json"
        )
        run(
            fuse_command(model, seq, cp, args.formal_warmup, args.formal_iters, output),
            env=gpu_env(cp),
            output=output,
            resume=args.resume,
        )


def fuse_aggregate(args: argparse.Namespace) -> None:
    rows = []
    for model, seq, cp in cases(args):
        path = args.results / "fuse_formal" / (
            f"{key(model, seq, cp)}_{args.formal_warmup}w{args.formal_iters}i.json"
        )
        data = load(path)
        shape = data["shape"]
        flops = 2.0 * shape["m"] * shape["n"] * shape["k"]
        fused = data["results"]["fused"]
        pure_names = ("cublas", "cublaslt_autotuned", "same_policy_cutlass")
        pure_best = min(data["results"][name]["p50_ms"] for name in pure_names)
        traits = data["kernel_traits"]
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
            "kv_heads": model.kv_heads,
            "head_dim": model.head_dim,
            "max_context": model.max_context,
            "within_native_context": (
                model.max_context is None or seq <= model.max_context
            ),
            "comm_ctas": data["comm_ctas"],
            "tile_m": traits["tile_m"],
            "tile_n": traits["tile_n"],
            "tile_k": traits["tile_k"],
            "fused_p50_ms": fused["p50_ms"],
            "fused_p95_ms": fused["p95_ms"],
            "fused_p50_tflops_per_gpu": flops / fused["p50_ms"] / 1.0e9,
            "fused_throughput_as_best_pure_percent": 100.0 * pure_best / fused["p50_ms"],
            "cublas_p50_ms": data["results"]["cublas"]["p50_ms"],
            "cublaslt_p50_ms": data["results"]["cublaslt_autotuned"]["p50_ms"],
            "cutlass_p50_ms": data["results"]["same_policy_cutlass"]["p50_ms"],
            "compute_subgrid_p50_ms": data["results"]["compute_subgrid_cutlass"]["p50_ms"],
            "qkv_route_p50_ms": data["results"]["qkv_a2a_route"]["p50_ms"],
            "sequential_p50_ms": data["results"]["sequential"]["p50_ms"],
            "overlap_ratio": data["overlap_ratio"],
        })
    dump_rows(rows, args.results / "fused_summary")


def write_shape_table(args: argparse.Namespace) -> None:
    rows = []
    for model, seq, cp in cases(args):
        m = seq // cp
        rows.append({
            "model": model.name,
            "suite": model.suite,
            "aliases": model.aliases,
            "global_seq": seq,
            "cp": cp,
            "m": m,
            "n": model.qkv_width,
            "k": model.hidden,
            "q_heads": model.q_heads,
            "kv_heads": model.kv_heads,
            "head_dim": model.head_dim,
            "max_context": model.max_context,
            "within_native_context": (
                model.max_context is None or seq <= model.max_context
            ),
            "q_heads_per_rank": model.q_heads // cp,
            "kv_heads_per_rank": model.kv_heads // cp,
            "gemm_input": f"[{m},{model.hidden}]",
            "gemm_weight": f"[{model.qkv_width},{model.hidden}]",
            "gemm_output": f"[{m},{model.qkv_width}]",
            "route_output": (
                f"Q[1,{seq},{model.q_heads // cp},{model.head_dim}] + "
                f"K[1,{seq},{model.kv_heads // cp},{model.head_dim}] + "
                f"V[1,{seq},{model.kv_heads // cp},{model.head_dim}]"
            ),
            "qkv_output_mib_per_gpu": m * model.qkv_width * 2 / (1 << 20),
            "remote_payload_mib_per_gpu": (
                m * model.qkv_width * 2 * (cp - 1) / cp / (1 << 20)
            ),
            "gemm_gflop_per_gpu": 2 * m * model.qkv_width * model.hidden / 1.0e9,
        })
    dump_rows(rows, args.results / "shape_matrix")
    with (args.results / "shape_matrix.md").open("w") as handle:
        handle.write(
            "| Suite | Model | S | CP | MxNxK | Hq/Hkv/D | QKV MiB/GPU | Remote MiB/GPU | GFLOP/GPU |\n"
            "|---|---|---:|---:|---|---|---:|---:|---:|\n"
        )
        for row in rows:
            handle.write(
                f"| {row['suite']} | {row['model']} | {row['global_seq']} | "
                f"{row['cp']} | {row['m']}x{row['n']}x{row['k']} | "
                f"{row['q_heads']}/{row['kv_heads']}/{row['head_dim']} | "
                f"{row['qkv_output_mib_per_gpu']:.2f} | "
                f"{row['remote_payload_mib_per_gpu']:.2f} | "
                f"{row['gemm_gflop_per_gpu']:.3f} |\n"
            )


def comparison_table(args: argparse.Namespace) -> None:
    fused = load(args.results / "fused_summary.json")
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
        key_value = (row["model"], row["global_seq"], row["cp"])
        other = baseline_by_key[key_value]
        merged = dict(row)
        merged.update({
            "best_separated": other["best_separated"],
            "best_separated_p50_ms": other["best_separated_p50_ms"],
            "speedup_over_best_separated": (
                other["best_separated_p50_ms"] / row["fused_p50_ms"]
            ),
            "te_boundary_p50_ms": other["te_boundary_p50_ms"],
            "cublas_boundary_p50_ms": other["cublas_boundary_p50_ms"],
            "cublaslt_boundary_p50_ms": other["cublaslt_boundary_p50_ms"],
        })
        ub = userbuffers_by_key.get(key_value)
        if ub is not None:
            ub_ms = ub["p50_ms"]
            best_external_ms = min(other["best_separated_p50_ms"], ub_ms)
            merged.update({
                "te_userbuffers_p50_ms": ub_ms,
                "speedup_over_te_userbuffers": ub_ms / row["fused_p50_ms"],
                "best_external_p50_ms": best_external_ms,
                "speedup_over_best_external": (
                    best_external_ms / row["fused_p50_ms"]
                ),
            })
        rows.append(merged)
    dump_rows(rows, args.results / "comparison_summary")
    with (args.results / "comparison_summary.md").open("w") as handle:
        handle.write(
            "| Suite | Model | S | CP | MxNxK | Fused ms / TFLOPS | "
            "Best separated ms | TE-UB ms | Best speedup | Pure GEMM % |\n"
            "|---|---|---:|---:|---|---:|---:|---:|---:|---:|\n"
        )
        for row in rows:
            handle.write(
                f"| {row['suite']} | {row['model']} | {row['global_seq']} | "
                f"{row['cp']} | {row['m']}x{row['n']}x{row['k']} | "
                f"{row['fused_p50_ms']:.4f} / "
                f"{row['fused_p50_tflops_per_gpu']:.1f} | "
                f"{row['best_separated_p50_ms']:.4f} | "
                f"{row.get('te_userbuffers_p50_ms', float('nan')):.4f} | "
                f"{row.get('speedup_over_best_external', row['speedup_over_best_separated']):.3f}x | "
                f"{row['fused_throughput_as_best_pure_percent']:.1f}% |\n"
            )


def main() -> None:
    args = parse_args()
    for phase in args.phase:
        {
            "shape-table": write_shape_table,
            "baseline-nccl-sweep": run_baseline_nccl_sweep,
            "baseline-mode-sweep": run_baseline_mode_sweep,
            "baseline-formal": run_baseline_formal,
            "baseline-aggregate": baseline_aggregate,
            "fuse-formal": run_fuse_formal,
            "fuse-aggregate": fuse_aggregate,
            "comparison-table": comparison_table,
        }[phase](args)


if __name__ == "__main__":
    main()
