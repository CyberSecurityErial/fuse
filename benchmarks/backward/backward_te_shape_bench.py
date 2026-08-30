#!/usr/bin/env python3
"""Search and formalize TE GEMM + NCCL baselines for v10 backward.

Every NCCL tuple is run in a fresh torchrun process because NCCL caches these
environment variables when the communicator is created.  The search is
two-stage: a full NCCL/priority sweep of the B phase, then a joint pack/mode
sweep over the per-case top-K communication configurations.  Formal winners
are always rerun with 10 warmups and 50 measured samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from backward_shape_bench import (
    CONTEXT_PARALLEL,
    OPROJ_MODELS,
    QKV_MODELS,
    SEQUENCES,
    VISIBLE_DEVICES,
    Model,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "results" / "backward" / "v10_te_nccl"
NCCL_CHANNELS = (8, 16, 24, 32)
NCCL_CHUNK_KIB = (128, 256, 512, 1024)
NCCL_LL_KIB = (16, 64, 128)
PACK_BLOCKS = (128, 256, 512, 1024)
PACK_WARPS = (4, 8)


@dataclass(frozen=True, order=True)
class Spec:
    channels: int
    chunk_kib: int
    ll_kib: int
    high_priority: bool
    launch: str
    pack_block: int
    pack_warps: int


def csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item for item in value.split(",") if item)


def csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        action="append",
        choices=(
            "nccl-sweep",
            "pack-mode-sweep",
            "formal",
            "correctness",
            "aggregate",
        ),
        required=True,
    )
    parser.add_argument("--operators", type=csv_strings, default=("qkv", "oproj"))
    parser.add_argument("--qkv-models", type=csv_strings, default=tuple(QKV_MODELS))
    parser.add_argument("--oproj-models", type=csv_strings, default=tuple(OPROJ_MODELS))
    parser.add_argument("--seqs", type=csv_ints, default=SEQUENCES)
    parser.add_argument("--cps", type=csv_ints, default=CONTEXT_PARALLEL)
    parser.add_argument("--launches", type=csv_strings, default=("eager", "graph"))
    parser.add_argument(
        "--weight-modes",
        type=csv_strings,
        default=("immediate", "deferred"),
    )
    parser.add_argument("--nccl-channels", type=csv_ints, default=NCCL_CHANNELS)
    parser.add_argument("--nccl-chunk-kib", type=csv_ints, default=NCCL_CHUNK_KIB)
    parser.add_argument("--nccl-ll-kib", type=csv_ints, default=NCCL_LL_KIB)
    parser.add_argument("--nccl-shortlist", type=int, default=3)
    parser.add_argument("--pack-blocks", type=csv_ints, default=PACK_BLOCKS)
    parser.add_argument("--pack-warps", type=csv_ints, default=PACK_WARPS)
    parser.add_argument("--sweep-warmup", type=int, default=3)
    parser.add_argument("--sweep-iters", type=int, default=12)
    parser.add_argument("--formal-warmup", type=int, default=10)
    parser.add_argument("--formal-iters", type=int, default=50)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(
            os.environ.get(
                "FUSE_TE_PYTHON",
                "/home/chen/miniforge3/envs/mmunlearner/bin/python",
            )
        ),
    )
    parser.add_argument(
        "--torchrun",
        type=Path,
        default=Path(
            os.environ.get(
                "FUSE_TE_TORCHRUN",
                "/home/chen/miniforge3/envs/mmunlearner/bin/torchrun",
            )
        ),
    )
    parser.add_argument(
        "--te-source-root",
        type=Path,
        default=Path(
            os.environ.get(
                "FUSE_TE_SOURCE_ROOT",
                "/home/chen/workspace/source_code/TransformerEngine",
            )
        ),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(__file__).with_name("backward_te_nccl_baseline.py"),
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--publish-results",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="write compact comparison files into each operator result directory",
    )
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    return parser.parse_args()


def models_for(operator: str) -> dict[str, Model]:
    if operator == "qkv":
        return QKV_MODELS
    if operator == "oproj":
        return OPROJ_MODELS
    raise ValueError(f"unknown operator {operator!r}")


def selected_models(args: argparse.Namespace, operator: str) -> list[Model]:
    registry = models_for(operator)
    names = args.qkv_models if operator == "qkv" else args.oproj_models
    missing = sorted(set(names) - set(registry))
    if missing:
        raise ValueError(f"unknown {operator} models: {missing}")
    return [registry[name] for name in names]


def cases(args: argparse.Namespace):
    for operator in args.operators:
        for model in selected_models(args, operator):
            for sequence in args.seqs:
                for cp in args.cps:
                    if (
                        sequence % cp == 0
                        and model.q_heads % cp == 0
                        and (operator != "qkv" or model.kv_heads % cp == 0)
                    ):
                        yield operator, model, sequence, cp


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


def case_key(operator: str, model: Model, sequence: int, cp: int) -> str:
    return f"{operator}_{model.name}_s{sequence}_cp{cp}"


def spec_label(spec: Spec) -> str:
    return (
        f"ch{spec.channels}_chunk{spec.chunk_kib}_ll{spec.ll_kib}_"
        f"{'hp' if spec.high_priority else 'normal'}_{spec.launch}_"
        f"pb{spec.pack_block}_pw{spec.pack_warps}"
    )


def sweep_path(
    args: argparse.Namespace,
    operator: str,
    model: Model,
    sequence: int,
    cp: int,
    spec: Spec,
) -> Path:
    return (
        args.results
        / "nccl_sweep"
        / case_key(operator, model, sequence, cp)
        / f"{spec_label(spec)}_{args.sweep_warmup}w{args.sweep_iters}i.json"
    )


def pack_path(
    args: argparse.Namespace,
    operator: str,
    model: Model,
    sequence: int,
    cp: int,
    mode: str,
    spec: Spec,
) -> Path:
    return (
        args.results
        / "pack_mode_sweep"
        / case_key(operator, model, sequence, cp)
        / f"{mode}_{spec_label(spec)}_{args.sweep_warmup}w{args.sweep_iters}i.json"
    )


def formal_path(
    args: argparse.Namespace,
    operator: str,
    model: Model,
    sequence: int,
    cp: int,
    mode: str,
    spec: Spec,
) -> Path:
    return (
        args.results
        / "formal"
        / operator
        / (
            f"{model.name}_s{sequence}_cp{cp}_{mode}_{spec_label(spec)}_"
            f"{args.formal_warmup}w{args.formal_iters}i.json"
        )
    )


def correctness_path(
    args: argparse.Namespace,
    operator: str,
    model: Model,
    cp: int,
    spec: Spec,
) -> Path:
    return (
        args.results
        / "correctness"
        / operator
        / f"{model.name}_s1024_cp{cp}_{spec_label(spec)}_1w1i.json"
    )


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def result_spec(payload: dict[str, object]) -> Spec:
    environment = payload["environment"]
    route = payload["route"]
    assert isinstance(environment, dict) and isinstance(route, dict)
    return Spec(
        int(environment["NCCL_MIN_P2P_NCHANNELS"]),
        int(environment["NCCL_P2P_NVL_CHUNKSIZE"]) // 1024,
        int(environment["NCCL_P2P_LL_THRESHOLD"]) // 1024,
        bool(payload["nccl_high_priority"]),
        str(payload["launch"]),
        int(route["pack_block"]),
        int(route["pack_warps"]),
    )


def finite_samples(value: object, count: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == count
        and all(
            isinstance(item, (int, float))
            and math.isfinite(float(item))
            and float(item) > 0.0
            for item in value
        )
    )


def correctness_passes(value: object) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    for name, result in value.items():
        if not isinstance(result, (int, float)) or not math.isfinite(float(result)):
            return False
        if "mismatches" in name:
            if int(result) != 0:
                return False
        elif float(result) > 0.25:
            return False
    return True


def samples_match_scope(
    results: dict[str, object], phase_scope: str, weight_mode: str, count: int
) -> bool:
    if phase_scope == "data":
        return finite_samples(results.get("data_samples_ms"), count)
    if phase_scope != "full":
        return False
    total = results.get("samples_ms")
    if not finite_samples(total, count):
        return False
    if weight_mode == "immediate":
        return True
    if weight_mode != "deferred":
        return False
    data_samples = results.get("data_samples_ms")
    weight_samples = results.get("weight_samples_ms")
    if not (
        finite_samples(data_samples, count)
        and finite_samples(weight_samples, count)
    ):
        return False
    assert isinstance(total, list)
    assert isinstance(data_samples, list)
    assert isinstance(weight_samples, list)
    return all(
        math.isclose(
            float(combined),
            float(data_value) + float(weight_value),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        for combined, data_value, weight_value in zip(
            total, data_samples, weight_samples
        )
    )


def result_matches(
    path: Path,
    entry: dict[str, object],
    *,
    cp: int,
    warmup: int,
    iterations: int,
) -> bool:
    if not path.exists():
        return False
    try:
        payload = load(path)
        arguments = entry["arguments"]
        environment = payload["environment"]
        route = payload["route"]
        results = payload["results"]
        assert isinstance(arguments, dict)
        assert isinstance(environment, dict)
        assert isinstance(route, dict)
        assert isinstance(results, dict)
        phase_scope = str(arguments["phase_scope"])
        correctness = payload.get("correctness", {})
        check = bool(arguments["check"])
        return (
            payload.get("schema") == "v10_backward_te_nccl_v2"
            and payload.get("numerical_contract")
            == "complete_route_then_single_full_gemm"
            and payload.get("determinism_contract")
            == "repeat_route_dgrad_wgrad_exact_at_s1k"
            and (
                payload.get("operator") != "qkv"
                or payload.get("qkv_pack_kernel") == "branch_masked_v1"
            )
            and payload.get("operator") == arguments["operator"]
            and payload.get("model_name") == arguments["model_name"]
            and payload.get("weight_mode") == arguments["weight_mode"]
            and payload.get("phase_scope") == phase_scope
            and payload.get("launch") == ("graph" if entry["cuda_graph"] else "eager")
            and payload.get("global_seq") == arguments["global_seq"]
            and payload.get("hidden") == arguments["hidden"]
            and payload.get("batch") == arguments["batch"]
            and payload.get("q_heads") == arguments["q_heads"]
            and payload.get("kv_heads") == arguments["kv_heads"]
            and payload.get("head_dim") == arguments["head_dim"]
            and payload.get("world_size") == cp
            and payload.get("warmup") == warmup
            and payload.get("iterations") == iterations
            and payload.get("nccl_high_priority") == entry["nccl_high_priority"]
            and route.get("pack_block") == arguments["pack_block"]
            and route.get("pack_warps") == arguments["pack_warps"]
            and route.get("causal_load_balanced") is True
            and environment.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
            and all(
                environment.get(name) == value
                for name, value in entry["environment"].items()
                if name in environment
            )
            and samples_match_scope(
                results,
                phase_scope,
                str(arguments["weight_mode"]),
                iterations,
            )
            and (not check or correctness_passes(correctness))
        )
    except (AssertionError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def matrix_entry(
    output: Path,
    operator: str,
    model: Model,
    sequence: int,
    cp: int,
    mode: str,
    spec: Spec,
    *,
    phase_scope: str,
    check: bool,
) -> dict[str, object]:
    config = (spec.channels, spec.chunk_kib, spec.ll_kib)
    return {
        "json_out": str(output.resolve()),
        "environment": nccl_environment(config),
        "cuda_graph": spec.launch == "graph",
        "nccl_high_priority": spec.high_priority,
        "arguments": {
            "operator": operator,
            "model_name": model.name,
            "global_seq": sequence,
            "hidden": model.hidden,
            "batch": 1,
            "q_heads": model.q_heads,
            "kv_heads": model.kv_heads,
            "head_dim": model.head_dim,
            "weight_mode": mode,
            "phase_scope": phase_scope,
            "pack_block": spec.pack_block,
            "pack_warps": spec.pack_warps,
            "causal_load_balanced": True,
            "check": check,
        },
    }


def base_command(args: argparse.Namespace, cp: int, warmup: int, iters: int) -> list[str]:
    return [
        str(args.torchrun),
        "--standalone",
        f"--nproc_per_node={cp}",
        str(args.baseline),
        "--operator",
        "qkv",
        "--global-seq",
        "1024",
        "--hidden",
        "4096",
        "--q-heads",
        "16",
        "--kv-heads",
        "8",
        "--head-dim",
        "128",
        "--weight-mode",
        "deferred",
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
    ]


def process_environment(
    args: argparse.Namespace,
    cp: int,
    config: tuple[int, int, int],
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(nccl_environment(config))
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = VISIBLE_DEVICES[cp]
    environment["OMP_NUM_THREADS"] = "1"
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    source_root = str(args.te_source_root.resolve())
    old_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        source_root if not old_pythonpath else f"{source_root}:{old_pythonpath}"
    )
    old_library_path = environment.get("LD_LIBRARY_PATH", "")
    environment["LD_LIBRARY_PATH"] = (
        source_root
        if not old_library_path
        else f"{source_root}:{old_library_path}"
    )
    return environment


def run_matrix(
    args: argparse.Namespace,
    *,
    cp: int,
    config: tuple[int, int, int],
    entries: list[dict[str, object]],
    phase: str,
    warmup: int,
    iterations: int,
) -> None:
    pending = [
        entry
        for entry in entries
        if not (
            args.resume
            and result_matches(
                Path(str(entry["json_out"])),
                entry,
                cp=cp,
                warmup=warmup,
                iterations=iterations,
            )
        )
    ]
    if not pending:
        return
    manifest = (
        args.results
        / "manifests"
        / phase
        / f"cp{cp}_ch{config[0]}_chunk{config[1]}_ll{config[2]}.json"
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    rendezvous_files: list[Path] = []
    for index, entry in enumerate(pending):
        rendezvous = Path(
            f"/tmp/fuse_backward_te_{os.getpid()}_{phase}_{cp}_{index}"
        )
        rendezvous.unlink(missing_ok=True)
        entry["rendezvous_file"] = str(rendezvous)
        rendezvous_files.append(rendezvous)
    manifest.write_text(json.dumps(pending, indent=2) + "\n")
    print(f"RUN-MATRIX {phase} CP{cp} {config}: {len(pending)} entries", flush=True)
    try:
        completed = subprocess.run(
            base_command(args, cp, warmup, iterations)
            + ["--matrix-manifest", str(manifest)],
            cwd=ROOT,
            env=process_environment(args, cp, config),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.timeout_seconds,
        )
    except BaseException:
        for entry in pending:
            Path(str(entry["json_out"])).unlink(missing_ok=True)
        raise
    finally:
        for rendezvous in rendezvous_files:
            rendezvous.unlink(missing_ok=True)
    if completed.returncode != 0:
        print(completed.stdout, end="")
        print(completed.stderr, end="")
        for entry in pending:
            Path(str(entry["json_out"])).unlink(missing_ok=True)
        completed.check_returncode()
    invalid = [
        Path(str(entry["json_out"]))
        for entry in pending
        if not result_matches(
            Path(str(entry["json_out"])),
            entry,
            cp=cp,
            warmup=warmup,
            iterations=iterations,
        )
    ]
    if invalid:
        for path in invalid:
            path.unlink(missing_ok=True)
        raise RuntimeError(f"matrix produced invalid results: {invalid[:4]}")


def run_grouped(
    args: argparse.Namespace,
    phase: str,
    groups: dict[tuple[int, tuple[int, int, int]], list[dict[str, object]]],
    warmup: int,
    iterations: int,
) -> None:
    for (cp, config), entries in sorted(groups.items()):
        run_matrix(
            args,
            cp=cp,
            config=config,
            entries=entries,
            phase=phase,
            warmup=warmup,
            iterations=iterations,
        )


def run_nccl_sweep(args: argparse.Namespace) -> None:
    groups: dict[tuple[int, tuple[int, int, int]], list[dict[str, object]]] = {}
    for operator, model, sequence, cp in cases(args):
        for config in nccl_configs(args):
            for launch in args.launches:
                for high_priority in (False, True):
                    spec = Spec(*config, high_priority, launch, 512, 4)
                    output = sweep_path(args, operator, model, sequence, cp, spec)
                    groups.setdefault((cp, config), []).append(
                        matrix_entry(
                            output,
                            operator,
                            model,
                            sequence,
                            cp,
                            "deferred",
                            spec,
                            phase_scope="data",
                            check=False,
                        )
                    )
    run_grouped(
        args,
        "nccl_sweep",
        groups,
        args.sweep_warmup,
        args.sweep_iters,
    )


def shortlisted_specs(
    args: argparse.Namespace,
    operator: str,
    model: Model,
    sequence: int,
    cp: int,
    launch: str,
) -> list[Spec]:
    candidates: list[dict[str, object]] = []
    for config in nccl_configs(args):
        for high_priority in (False, True):
            spec = Spec(*config, high_priority, launch, 512, 4)
            path = sweep_path(args, operator, model, sequence, cp, spec)
            if not path.exists():
                raise FileNotFoundError(path)
            candidates.append(load(path))
    candidates.sort(key=lambda item: float(item["results"]["data"]["p50_ms"]))
    return [result_spec(item) for item in candidates[: args.nccl_shortlist]]


def run_pack_mode_sweep(args: argparse.Namespace) -> None:
    groups: dict[tuple[int, tuple[int, int, int]], list[dict[str, object]]] = {}
    for operator, model, sequence, cp in cases(args):
        for launch in args.launches:
            for base in shortlisted_specs(
                args, operator, model, sequence, cp, launch
            ):
                config = (base.channels, base.chunk_kib, base.ll_kib)
                for block in args.pack_blocks:
                    for warps in args.pack_warps:
                        spec = Spec(
                            *config,
                            base.high_priority,
                            launch,
                            block,
                            warps,
                        )
                        for mode in args.weight_modes:
                            output = pack_path(
                                args,
                                operator,
                                model,
                                sequence,
                                cp,
                                mode,
                                spec,
                            )
                            groups.setdefault((cp, config), []).append(
                                matrix_entry(
                                    output,
                                    operator,
                                    model,
                                    sequence,
                                    cp,
                                    mode,
                                    spec,
                                    phase_scope="full",
                                    check=False,
                                )
                            )
    run_grouped(
        args,
        "pack_mode_sweep",
        groups,
        args.sweep_warmup,
        args.sweep_iters,
    )


def best_full_spec(
    args: argparse.Namespace,
    operator: str,
    model: Model,
    sequence: int,
    cp: int,
    launch: str,
    mode: str,
) -> Spec:
    paths: list[Path] = []
    for base in shortlisted_specs(args, operator, model, sequence, cp, launch):
        for block in args.pack_blocks:
            for warps in args.pack_warps:
                spec = Spec(
                    base.channels,
                    base.chunk_kib,
                    base.ll_kib,
                    base.high_priority,
                    launch,
                    block,
                    warps,
                )
                paths.append(
                    pack_path(args, operator, model, sequence, cp, mode, spec)
                )
    if any(not path.exists() for path in paths):
        raise FileNotFoundError("missing pack/mode sweep result")
    candidates = [load(path) for path in paths]
    winner = min(
        candidates,
        key=lambda item: float(item["results"]["total"]["p50_ms"]),
    )
    return result_spec(winner)


def run_formal(args: argparse.Namespace) -> None:
    groups: dict[tuple[int, tuple[int, int, int]], list[dict[str, object]]] = {}
    for operator, model, sequence, cp in cases(args):
        for launch in args.launches:
            for mode in args.weight_modes:
                spec = best_full_spec(
                    args, operator, model, sequence, cp, launch, mode
                )
                config = (spec.channels, spec.chunk_kib, spec.ll_kib)
                output = formal_path(
                    args,
                    operator,
                    model,
                    sequence,
                    cp,
                    mode,
                    spec,
                )
                groups.setdefault((cp, config), []).append(
                    matrix_entry(
                        output,
                        operator,
                        model,
                        sequence,
                        cp,
                        mode,
                        spec,
                        phase_scope="full",
                        check=False,
                    )
                )
    run_grouped(
        args,
        "formal",
        groups,
        args.formal_warmup,
        args.formal_iters,
    )


def run_correctness(args: argparse.Namespace) -> None:
    # A correctness run performs extra reference GEMMs and allocations.  Keep
    # each geometry in a fresh torchrun process so none of that state can
    # contaminate a later formal sample or another shape.
    for operator, model, sequence, cp in cases(args):
        if sequence != 1024:
            continue
        spec = best_full_spec(
            args,
            operator,
            model,
            sequence,
            cp,
            "eager",
            "deferred",
        )
        config = (spec.channels, spec.chunk_kib, spec.ll_kib)
        output = correctness_path(args, operator, model, cp, spec)
        entry = matrix_entry(
            output,
            operator,
            model,
            sequence,
            cp,
            "deferred",
            spec,
            phase_scope="full",
            check=True,
        )
        run_matrix(
            args,
            cp=cp,
            config=config,
            entries=[entry],
            phase=f"correctness_{case_key(operator, model, sequence, cp)}",
            warmup=1,
            iterations=1,
        )


def dump_rows(rows: list[dict[str, object]], stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def sequence_label(value: int) -> str:
    return f"{value // 1024}K" if value % 1024 == 0 else str(value)


def fused_summary_path(operator: str) -> Path:
    if operator == "qkv":
        return (
            ROOT
            / "results/QKVproj-backward/qkv_backward_shape_bench/"
            "qkv_backward_summary.json"
        )
    return (
        ROOT
        / "results/Oproj-backward/oproj_backward_shape_bench/"
        "oproj_backward_summary.json"
    )


def operator_result_dir(operator: str) -> Path:
    return fused_summary_path(operator).parent


def aggregate_operator(args: argparse.Namespace, operator: str) -> None:
    fused_rows = load(fused_summary_path(operator))
    assert isinstance(fused_rows, list)
    fused_by_key = {
        (str(row["model"]), int(row["global_seq"]), int(row["cp"])): row
        for row in fused_rows
    }
    rows: list[dict[str, object]] = []
    for _, model, sequence, cp in cases(args):
        if _ != operator:
            continue
        key = (model.name, sequence, cp)
        if key not in fused_by_key:
            raise KeyError(f"missing fused row {key}")
        fused = fused_by_key[key]
        row: dict[str, object] = {
            "operator": operator,
            "model": model.name,
            "suite": model.suite,
            "aliases": model.aliases,
            "global_seq": sequence,
            "cp": cp,
            "local_tokens": sequence // cp,
            "hidden": model.hidden,
            "projection_width": (
                model.qkv_width if operator == "qkv" else model.attention_width
            ),
            "q_heads": model.q_heads,
            "kv_heads": model.kv_heads,
            "head_dim": model.head_dim,
            "warmup": args.formal_warmup,
            "iterations": args.formal_iters,
            "timing": "per-sample max-rank CUDA-event critical path",
        }
        for launch in args.launches:
            for mode in args.weight_modes:
                spec = best_full_spec(
                    args, operator, model, sequence, cp, launch, mode
                )
                payload = load(
                    formal_path(
                        args,
                        operator,
                        model,
                        sequence,
                        cp,
                        mode,
                        spec,
                    )
                )
                software = payload.get("software") or {}
                devices = payload.get("devices") or []
                row["qkv_pack_kernel"] = payload["qkv_pack_kernel"]
                row["te_nccl_schema"] = payload["schema"]
                row["te_nccl_numerical_contract"] = payload[
                    "numerical_contract"
                ]
                row["te_nccl_determinism_contract"] = payload[
                    "determinism_contract"
                ]
                row["transformer_engine_version"] = software.get(
                    "transformer_engine"
                )
                row["transformer_engine_path"] = software.get(
                    "transformer_engine_path"
                )
                row["torch_version"] = software.get("torch")
                row["torch_cuda_version"] = software.get("torch_cuda")
                row["nccl_version"] = software.get("nccl")
                row["device_names"] = "|".join(
                    str(device.get("name"))
                    for device in devices
                    if isinstance(device, dict)
                )
                row["device_sm_counts"] = "|".join(
                    str(device.get("sm_count"))
                    for device in devices
                    if isinstance(device, dict)
                )
                row["cublas_workspace_config"] = payload["environment"].get(
                    "CUBLAS_WORKSPACE_CONFIG"
                )
                correctness_spec = best_full_spec(
                    args,
                    operator,
                    model,
                    1024,
                    cp,
                    "eager",
                    "deferred",
                )
                correctness_payload = load(
                    correctness_path(
                        args,
                        operator,
                        model,
                        cp,
                        correctness_spec,
                    )
                )
                if not correctness_passes(correctness_payload.get("correctness")):
                    raise ValueError(
                        f"missing/failed TE correctness for {operator}/{model.name}/CP{cp}"
                    )
                shape = payload["shape"]
                expected_b = [fused["b_m"], fused["b_n"], fused["b_k"]]
                expected_w = [fused["w_m"], fused["w_n"], fused["w_k"]]
                if shape["b_mnk"] != expected_b or shape["w_mnk"] != expected_w:
                    raise ValueError(f"TE/fused backward MNK mismatch for {key}")
                prefix = f"{launch}_{mode}"
                total = payload["results"]["total"]
                data_stats = payload["results"].get("data", {})
                weight_stats = payload["results"].get("weight", {})
                fused_ms = float(fused[f"{prefix}_total_p50_ms"])
                te_ms = float(total["p50_ms"])
                beta = 1 if mode == "deferred" else 0
                cublas_ms = float(fused[f"cublas_total_beta{beta}_p50_ms"])
                forward_tflops = float(
                    fused[f"{prefix}_forward_p50_tflops_per_gpu"]
                )
                te_tflops = float(total["p50_tflops_per_gpu"])
                row[f"{prefix}_te_nccl_p50_ms"] = te_ms
                row[f"{prefix}_te_nccl_p95_ms"] = float(total["p95_ms"])
                row[f"{prefix}_te_nccl_p50_tflops_per_gpu"] = float(
                    total["p50_tflops_per_gpu"]
                )
                row[f"{prefix}_te_nccl_data_p50_ms"] = data_stats.get(
                    "p50_ms"
                )
                row[f"{prefix}_te_nccl_weight_p50_ms"] = weight_stats.get(
                    "p50_ms"
                )
                row[f"{prefix}_fused_p50_ms"] = fused_ms
                row[f"{prefix}_fused_p95_ms"] = float(
                    fused[f"{prefix}_total_p95_ms"]
                )
                row[f"{prefix}_fused_p50_tflops_per_gpu"] = float(
                    fused[f"{prefix}_total_p50_tflops_per_gpu"]
                )
                row[f"{prefix}_fused_speedup_over_te_nccl"] = te_ms / fused_ms
                row[f"{prefix}_forward_p50_tflops_per_gpu"] = forward_tflops
                row[f"{prefix}_cublas_p50_ms"] = cublas_ms
                row[f"{prefix}_cublas_p50_tflops_per_gpu"] = float(
                    fused[f"cublas_total_beta{beta}_p50_tflops_per_gpu"]
                )
                row[f"{prefix}_fused_throughput_as_forward_percent"] = float(
                    fused[f"{prefix}_throughput_as_forward_percent"]
                )
                row[f"{prefix}_fused_throughput_as_cublas_percent"] = float(
                    fused[f"{prefix}_throughput_as_cublas_percent"]
                )
                row[f"{prefix}_te_nccl_throughput_as_forward_percent"] = (
                    100.0 * te_tflops / forward_tflops
                )
                row[f"{prefix}_te_nccl_as_cublas_percent"] = (
                    100.0 * cublas_ms / te_ms
                )
                row[f"{prefix}_te_nccl_config"] = spec_label(spec)
                row[f"{prefix}_te_nccl_beta"] = payload[
                    "weight_accumulation_beta"
                ]
        rows.append(row)
    rows.sort(key=lambda item: (int(item["cp"]), str(item["model"]), int(item["global_seq"])))
    stem = (
        operator_result_dir(operator) / f"{operator}_backward_te_nccl_comparison"
        if args.publish_results
        else args.results / "summary" / f"{operator}_te_nccl_comparison"
    )
    dump_rows(rows, stem)
    lines = [
        f"# {operator.upper()} backward：fused 与 TE+NCCL",
        "",
        "比值为 `TE+NCCL p50 / fused p50`；大于 1 表示 fused 更快。",
        "",
        "| CP | 模型 | S | 模式 | fused Eager / TE / 加速 | fused Graph / TE / 加速 |",
        "|---:|---|---:|---|---:|---:|",
    ]
    for row in rows:
        for mode in args.weight_modes:
            eager = f"eager_{mode}"
            graph = f"graph_{mode}"
            lines.append(
                f"| {row['cp']} | {row['model']} | "
                f"{sequence_label(int(row['global_seq']))} | {mode} | "
                f"{row[f'{eager}_fused_p50_ms']:.4f} / "
                f"{row[f'{eager}_te_nccl_p50_ms']:.4f} / "
                f"{row[f'{eager}_fused_speedup_over_te_nccl']:.3f}x | "
                f"{row[f'{graph}_fused_p50_ms']:.4f} / "
                f"{row[f'{graph}_te_nccl_p50_ms']:.4f} / "
                f"{row[f'{graph}_fused_speedup_over_te_nccl']:.3f}x |"
            )
    for mode in args.weight_modes:
        eager_speedups = [
            float(row[f"eager_{mode}_fused_speedup_over_te_nccl"])
            for row in rows
        ]
        graph_speedups = [
            float(row[f"graph_{mode}_fused_speedup_over_te_nccl"])
            for row in rows
        ]
        eager_forward = [
            float(row[f"eager_{mode}_te_nccl_throughput_as_forward_percent"])
            for row in rows
        ]
        graph_forward = [
            float(row[f"graph_{mode}_te_nccl_throughput_as_forward_percent"])
            for row in rows
        ]
        eager_cublas = [
            float(row[f"eager_{mode}_te_nccl_as_cublas_percent"])
            for row in rows
        ]
        graph_cublas = [
            float(row[f"graph_{mode}_te_nccl_as_cublas_percent"])
            for row in rows
        ]
        lines.extend((
            "",
            f"{mode}：fused 对 TE+NCCL 的 Eager 胜点 "
            f"{sum(value > 1 for value in eager_speedups)}/{len(eager_speedups)}，"
            f"几何平均 {geometric_mean(eager_speedups):.3f}x；Graph 胜点 "
            f"{sum(value > 1 for value in graph_speedups)}/{len(graph_speedups)}，"
            f"几何平均 {geometric_mean(graph_speedups):.3f}x。",
            f"TE+NCCL 吞吐相对前向的中位数为 "
            f"{median(eager_forward):.1f}%/{median(graph_forward):.1f}% "
            f"(Eager/Graph)，相对经典 cuBLAS 纯 B+W 为 "
            f"{median(eager_cublas):.1f}%/{median(graph_cublas):.1f}%。",
        ))
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n")


def run_aggregate(args: argparse.Namespace) -> None:
    for operator in args.operators:
        aggregate_operator(args, operator)


def validate_args(args: argparse.Namespace) -> None:
    if any(value not in {"qkv", "oproj"} for value in args.operators):
        raise ValueError("--operators accepts qkv,oproj")
    if any(value not in {"eager", "graph"} for value in args.launches):
        raise ValueError("--launches accepts eager,graph")
    if any(value not in {"immediate", "deferred"} for value in args.weight_modes):
        raise ValueError("--weight-modes accepts immediate,deferred")
    if args.nccl_shortlist <= 0:
        raise ValueError("--nccl-shortlist must be positive")
    if not args.baseline.is_file():
        raise FileNotFoundError(args.baseline)
    if not args.torchrun.is_file() or not args.python.is_file():
        raise FileNotFoundError("TE Python/torchrun executable is missing")


def main() -> None:
    args = parse_args()
    validate_args(args)
    runners = {
        "nccl-sweep": run_nccl_sweep,
        "pack-mode-sweep": run_pack_mode_sweep,
        "formal": run_formal,
        "correctness": run_correctness,
        "aggregate": run_aggregate,
    }
    for phase in args.phase:
        runners[phase](args)


if __name__ == "__main__":
    main()
