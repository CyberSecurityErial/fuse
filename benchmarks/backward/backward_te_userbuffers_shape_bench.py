#!/usr/bin/env python3
"""Lightweight tuning and formal driver for the v10 backward TE-UB baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class Config:
    comm_sm: int
    streams: int = 3
    push: bool = True
    use_ce: bool = False
    pack_block: int = 512
    pack_warps: int = 4
    reverse: bool = False
    local_first: bool = True

    @property
    def math_sm(self) -> int:
        # The formal single-GEMM boundary does not overlap GEMM with UB
        # communication, so reserving math SMs would only weaken the baseline.
        # Zero asks cuBLASLt to use the full device.
        return 0

    @property
    def tag(self) -> str:
        return (
            f"c{self.comm_sm}_mall_s{self.streams}_"
            f"{'push' if self.push else 'pull'}_"
            f"{'ce' if self.use_ce else 'sm'}_"
            f"b{self.pack_block}_w{self.pack_warps}_"
            f"{'rev' if self.reverse else 'fwd'}_"
            f"{'localfirst' if self.local_first else 'remotefirst'}"
        )


def csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item for item in value.split(",") if item)


def csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("sweep", "refine", "formal", "summary", "all"),
        default="all",
    )
    parser.add_argument("--operators", type=csv_strings, default=("qkv", "oproj"))
    parser.add_argument("--qkv-models", type=csv_strings, default=tuple(QKV_MODELS))
    parser.add_argument("--oproj-models", type=csv_strings, default=tuple(OPROJ_MODELS))
    parser.add_argument("--seqs", type=csv_ints, default=SEQUENCES)
    parser.add_argument("--cps", type=csv_ints, default=CONTEXT_PARALLEL)
    parser.add_argument("--launches", type=csv_strings, default=("eager", "graph"))
    parser.add_argument("--weight-modes", type=csv_strings, default=("immediate", "deferred"))
    # TE-UB is the strong baseline, but the search intentionally stays small:
    # communication reservation first, then one structural neighborhood.
    parser.add_argument(
        "--comm-sm", type=csv_ints, default=(4, 8, 12, 16, 20, 24)
    )
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "backward" / "v10_te_userbuffers")
    parser.add_argument("--python", type=Path, default=Path("/home/chen/miniforge3/envs/mmunlearner/bin/python"))
    parser.add_argument("--te-source-root", type=Path, default=Path("/tmp/TransformerEngine-v10-clean"))
    parser.add_argument("--cublaslt-library", type=Path, default=ROOT / "build" / "libfuse_cublaslt_runner.so")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--publish-results",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="also write compact machine-readable summaries into each operator result directory",
    )
    parser.add_argument("--sweep-warmup", type=int, default=2)
    parser.add_argument("--sweep-iters", type=int, default=8)
    parser.add_argument("--formal-warmup", type=int, default=10)
    parser.add_argument("--formal-iters", type=int, default=50)
    parser.add_argument("--formal-shortlist", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    return parser.parse_args()


def registry(operator: str) -> dict[str, Model]:
    if operator == "qkv":
        return QKV_MODELS
    if operator == "oproj":
        return OPROJ_MODELS
    raise ValueError(f"unknown operator {operator}")


def cases(args: argparse.Namespace):
    for operator in args.operators:
        models = registry(operator)
        selected = args.qkv_models if operator == "qkv" else args.oproj_models
        missing = set(selected) - set(models)
        if missing:
            raise ValueError(f"unknown {operator} models: {sorted(missing)}")
        for name in selected:
            model = models[name]
            for sequence in args.seqs:
                for cp in args.cps:
                    if sequence % cp or model.q_heads % cp:
                        continue
                    if operator == "qkv" and model.kv_heads % cp:
                        continue
                    yield operator, model, sequence, cp


def case_key(operator: str, model: Model, sequence: int, cp: int, launch: str) -> str:
    return f"{operator}_{model.name}_s{sequence}_cp{cp}_{launch}"


def structural_candidates(comm: int) -> tuple[Config, ...]:
    # One factor at a time around the best communication reservation.  This
    # keeps tuning cheap while covering the knobs that materially change UB.
    return (
        Config(comm),
        Config(comm, streams=1),
        Config(comm, streams=2),
        Config(comm, push=False),
        Config(comm, use_ce=True),
        Config(comm, pack_block=256),
        Config(comm, pack_block=1024),
        Config(comm, pack_warps=8),
        Config(comm, reverse=True),
    )


def matrix_entry(
    args: argparse.Namespace,
    *,
    operator: str,
    model: Model,
    sequence: int,
    config: Config,
    launch: str,
    mode: str,
    scope: str,
    warmup: int,
    iterations: int,
    output: Path,
    check: bool,
) -> dict[str, object]:
    return {
        "json_out": str(output.resolve()),
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
            "phase_scope": scope,
            "data_gemm_mode": "single",
            "warmup": warmup,
            "iters": iterations,
            "num_comm_sm": config.comm_sm,
            "math_sm": config.math_sm,
            "num_streams": config.streams,
            "parallel_sends": config.streams > 1,
            "push": config.push,
            "use_ce": config.use_ce,
            "reverse": config.reverse,
            "local_first": config.local_first,
            "pack_block": config.pack_block,
            "pack_warps": config.pack_warps,
            "tune_warmup": 5 if scope == "data" else 10,
            "tune_iters": 15 if scope == "data" else 30,
            "workspace_mib": 64,
            "cuda_graph": launch == "graph",
            "causal_load_balanced": True,
            "check": check,
            "cublaslt_library": str(args.cublaslt_library.resolve()),
        },
    }


def environment(args: argparse.Namespace, cp: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = VISIBLE_DEVICES[cp]
    env["OMP_NUM_THREADS"] = "1"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    root = str(args.te_source_root.resolve())
    env["PYTHONPATH"] = root
    env["LD_LIBRARY_PATH"] = f"{root}:/usr/local/cuda/lib64"
    return env


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def correctness_passes(data: dict[str, object]) -> bool:
    check = data.get("correctness", {})
    if not isinstance(check, dict) or not check:
        return False
    if int(check.get("route_mismatches", 0)) != 0:
        return False
    return all(float(value) <= 0.25 for name, value in check.items() if name != "route_mismatches")


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


def samples_match_scope(data: dict[str, object], count: int) -> bool:
    results = data.get("results")
    if not isinstance(results, dict):
        return False
    scope = data.get("phase_scope")
    mode = data.get("weight_mode")
    if scope in ("comm", "data"):
        return finite_samples(results.get("data_samples_ms"), count)
    if scope != "full":
        return False
    total = results.get("samples_ms")
    if not finite_samples(total, count):
        return False
    if mode == "immediate":
        return True
    if mode != "deferred":
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


def valid_result(
    path: Path,
    *,
    checked: bool | None = None,
    expected_scope: str | None = None,
) -> bool:
    if not path.exists():
        return False
    try:
        data = load(path)
        if (
            data.get("schema") != "v10_backward_te_userbuffers_v3"
            or data.get("mode") != "te_userbuffers_adapted_backward"
            or data.get("numerical_contract")
            != "complete_route_then_single_full_gemm"
            or data.get("ub_object_lifecycle") != "recreate_per_entry"
            or data.get("ub_unpack_kernel") != "branch_masked_v1"
        ):
            return False
        if data.get("data_gemm_mode") != "single":
            return False
        software = data.get("software")
        devices = data.get("devices")
        environment = data.get("environment")
        if (
            not isinstance(software, dict)
            or not software.get("transformer_engine")
            or not software.get("torch")
            or not isinstance(devices, list)
            or len(devices) != int(data.get("world_size", 0))
            or not isinstance(environment, dict)
            or environment.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        ):
            return False
        if (
            data.get("operator") == "qkv"
            and data.get("qkv_pack_kernel") != "branch_masked_v1"
        ):
            return False
        if expected_scope is not None and data.get("phase_scope") != expected_scope:
            return False
        if (
            data.get("phase_scope") == "full"
            and data.get("cublaslt_plan_lifecycle")
            != "reuse_adjacent_same_shape_v1"
        ):
            return False
        config = data.get("config", {})
        if not isinstance(config, dict) or config.get("data_gemm_mode") != "single":
            return False
        iterations = int(config.get("iters", 0))
        if iterations <= 0 or not samples_match_scope(data, iterations):
            return False
        if checked is True and not correctness_passes(data):
            return False
        return True
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def result_matches_entry(path: Path, entry: dict[str, object], cp: int) -> bool:
    arguments = entry["arguments"]
    assert isinstance(arguments, dict)
    checked = bool(arguments["check"])
    if not valid_result(
        path,
        checked=checked,
        expected_scope=str(arguments["phase_scope"]),
    ):
        return False
    data = load(path)
    config = data["config"]
    assert isinstance(config, dict)
    keys = (
        "operator", "model_name", "global_seq", "hidden", "batch",
        "q_heads", "kv_heads", "head_dim", "weight_mode", "phase_scope",
        "data_gemm_mode", "warmup", "iters", "num_comm_sm", "math_sm",
        "num_streams", "parallel_sends", "push", "use_ce", "reverse",
        "local_first", "pack_block", "pack_warps", "cuda_graph",
        "causal_load_balanced",
    )
    return (
        int(data.get("world_size", 0)) == cp
        and all(config.get(key) == arguments.get(key) for key in keys)
        and data.get("operator") == arguments["operator"]
        and data.get("launch") == ("graph" if arguments["cuda_graph"] else "eager")
        and data.get("weight_mode") == arguments["weight_mode"]
        and (
            arguments["phase_scope"] != "full"
            or int(data.get("weight_accumulation_beta", -1))
            == (1 if arguments["weight_mode"] == "deferred" else 0)
        )
    )


def run_matrix(
    args: argparse.Namespace,
    *,
    cp: int,
    entries: list[dict[str, object]],
    manifest: Path,
) -> None:
    pending = []
    for entry in entries:
        output = Path(str(entry["json_out"]))
        checked = bool(entry["arguments"]["check"])
        if args.resume and result_matches_entry(output, entry, cp):
            continue
        output.unlink(missing_ok=True)
        pending.append(entry)
    if not pending:
        return
    # TE Userbuffers owns process-global communicator state.  A process may
    # safely reuse one exact UB configuration across shapes, launch modes, and
    # weight modes, but changing that configuration after initialization is
    # unsupported and has reproduced stale device state.  Callers therefore
    # group entries by operator, CP, and Config and launch one fresh process
    # for every group.
    state_keys = {
        (
            str(entry["arguments"]["operator"]),
            int(entry["arguments"]["num_comm_sm"]),
            int(entry["arguments"]["num_streams"]),
            bool(entry["arguments"]["push"]),
            bool(entry["arguments"]["use_ce"]),
            bool(entry["arguments"]["parallel_sends"]),
        )
        for entry in pending
    }
    if len(state_keys) != 1:
        raise ValueError(f"mixed Userbuffers process state in {manifest}: {state_keys}")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_suffix(manifest.suffix + ".tmp")
    temporary.write_text(json.dumps(pending, indent=2) + "\n")
    temporary.replace(manifest)
    print(f"RUN-MATRIX {manifest} ({len(pending)} entries, one fresh process)", flush=True)
    command = [
        str(args.python), "-m", "torch.distributed.run", "--standalone",
        f"--nproc-per-node={cp}",
        str(Path(__file__).with_name("backward_te_userbuffers.py")),
        "--operator", "qkv", "--global-seq", "1024", "--hidden", "4096",
        "--q-heads", "16", "--kv-heads", "8", "--head-dim", "128",
        "--weight-mode", "deferred", "--matrix-manifest", str(manifest),
    ]
    try:
        completed = subprocess.run(
            command, cwd=ROOT, env=environment(args, cp), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=args.timeout_seconds,
        )
    except BaseException:
        for entry in pending:
            Path(str(entry["json_out"])).unlink(missing_ok=True)
        raise
    if completed.returncode:
        print(completed.stdout[-16000:], flush=True)
        for entry in pending:
            Path(str(entry["json_out"])).unlink(missing_ok=True)
        completed.check_returncode()
    invalid = [
        Path(str(entry["json_out"]))
        for entry in pending
        if not result_matches_entry(Path(str(entry["json_out"])), entry, cp)
    ]
    if invalid:
        for output in invalid:
            output.unlink(missing_ok=True)
        raise RuntimeError(f"invalid TE-UB results: {invalid[:4]}")


def run_grouped(
    args: argparse.Namespace,
    *,
    phase: str,
    groups: dict[tuple[str, int, Config], list[dict[str, object]]],
) -> None:
    # pack size/warps and peer order change only Triton work issued through an
    # already configured communicator.  Merge those variants into the same
    # process; keep the four fields that actually construct/configure UB as
    # hard process boundaries.
    process_groups: dict[
        tuple[str, int, int, int, bool, bool], list[dict[str, object]]
    ] = {}
    for (operator, cp, config), entries in groups.items():
        key = (
            operator,
            cp,
            config.comm_sm,
            config.streams,
            config.push,
            config.use_ce,
        )
        process_groups.setdefault(key, []).extend(entries)
    for (operator, cp, comm_sm, streams, push, use_ce), entries in sorted(
        process_groups.items(), key=lambda item: item[0]
    ):
        if phase == "formal":
            # Keep Eager/Graph and immediate/deferred entries for one GEMM
            # shape adjacent.  The child process can then reuse exactly the
            # same cuBLASLt algorithm plan while still recreating every UB
            # object and every measured buffer.
            entries = sorted(
                entries,
                key=lambda entry: (
                    int(entry["arguments"]["global_seq"]),
                    int(entry["arguments"]["batch"]),
                    int(entry["arguments"]["hidden"]),
                    int(entry["arguments"]["q_heads"]),
                    int(entry["arguments"]["kv_heads"]),
                    int(entry["arguments"]["head_dim"]),
                    bool(entry["arguments"]["cuda_graph"]),
                    str(entry["arguments"]["weight_mode"]),
                ),
            )
        process_tag = (
            f"c{comm_sm}_s{streams}_"
            f"{'push' if push else 'pull'}_{'ce' if use_ce else 'sm'}"
        )
        run_matrix(
            args,
            cp=cp,
            entries=entries,
            manifest=(
                args.results / "manifests" / phase /
                f"{operator}_cp{cp}_{process_tag}.json"
            ),
        )


def config_from_result(data: dict[str, object]) -> Config:
    cfg = data["config"]
    return Config(
        int(cfg["num_comm_sm"]), int(cfg["num_streams"]), bool(cfg["push"]),
        bool(cfg["use_ce"]), int(cfg["pack_block"]), int(cfg["pack_warps"]),
        bool(cfg["reverse"]), bool(cfg["local_first"]),
    )


def sweep_path(args: argparse.Namespace, key: str, config: Config) -> Path:
    return args.results / "sweep" / key / f"{config.tag}.json"


def ranked_sweep(args: argparse.Namespace, key: str) -> list[tuple[float, Config, Path]]:
    records = []
    for path in (args.results / "sweep" / key).glob("*.json"):
        if not valid_result(path, expected_scope="comm"):
            continue
        data = load(path)
        records.append((float(data["results"]["data"]["p50_ms"]), config_from_result(data), path))
    if not records:
        raise RuntimeError(f"no valid TE-UB sweep for {key}")
    return sorted(records, key=lambda item: item[0])


def best_existing_config(
    args: argparse.Namespace, key: str, candidates: tuple[Config, ...]
) -> Config:
    scored: list[tuple[float, Config]] = []
    for config in candidates:
        path = sweep_path(args, key, config)
        if not valid_result(path, expected_scope="comm"):
            raise RuntimeError(f"missing valid TE-UB sweep result: {path}")
        scored.append((float(load(path)["results"]["data"]["p50_ms"]), config))
    return min(scored, key=lambda item: item[0])[1]


def sweep(args: argparse.Namespace) -> None:
    communication_groups: dict[
        tuple[str, int, Config], list[dict[str, object]]
    ] = {}
    for operator, model, sequence, cp in cases(args):
        for launch in args.launches:
            for comm in args.comm_sm:
                config = Config(comm)
                key = case_key(operator, model, sequence, cp, launch)
                communication_groups.setdefault((operator, cp, config), []).append(matrix_entry(
                    args, operator=operator, model=model, sequence=sequence,
                    config=config, launch=launch, mode="deferred", scope="comm",
                    warmup=args.sweep_warmup, iterations=args.sweep_iters,
                    output=sweep_path(args, key, config), check=False,
                ))
    run_grouped(args, phase="sweep_comm", groups=communication_groups)

    structure_groups: dict[
        tuple[str, int, Config], list[dict[str, object]]
    ] = {}
    for operator, model, sequence, cp in cases(args):
        for launch in args.launches:
            key = case_key(operator, model, sequence, cp, launch)
            # Choose the structural neighborhood from the communication-only
            # base points.  ranked_sweep() also contains previously measured
            # stream/transport variants; using it here makes an incremental
            # resume order-dependent when a new comm-SM candidate is added.
            best_comm = best_existing_config(
                args, key, tuple(Config(comm) for comm in args.comm_sm)
            ).comm_sm
            for config in structural_candidates(best_comm):
                structure_groups.setdefault((operator, cp, config), []).append(matrix_entry(
                    args, operator=operator, model=model, sequence=sequence,
                    config=config, launch=launch, mode="deferred", scope="comm",
                    warmup=args.sweep_warmup, iterations=args.sweep_iters,
                    output=sweep_path(args, key, config), check=False,
                ))
    run_grouped(args, phase="sweep_structure", groups=structure_groups)


def refine(args: argparse.Namespace) -> None:
    """Test a compact joint neighborhood after the one-factor sweep.

    Full Cartesian tuning would require 144 structural configurations for
    every shape and launch mode.  Instead, choose each axis from measurements
    that already exist, combine those winners, and retest the four transport
    pairs.  This catches the important stream/pack/transport interaction while
    keeping the strong-baseline search finite and explainable.
    """

    groups: dict[tuple[str, int, Config], list[dict[str, object]]] = {}
    for operator, model, sequence, cp in cases(args):
        for launch in args.launches:
            key = case_key(operator, model, sequence, cp, launch)
            base = best_existing_config(
                args, key, tuple(Config(comm) for comm in args.comm_sm)
            )
            comm = base.comm_sm
            stream = best_existing_config(
                args,
                key,
                tuple(Config(comm, streams=value) for value in (1, 2, 3)),
            ).streams
            transport = best_existing_config(
                args,
                key,
                (
                    Config(comm),
                    Config(comm, push=False),
                    Config(comm, use_ce=True),
                ),
            )
            block = best_existing_config(
                args,
                key,
                tuple(Config(comm, pack_block=value) for value in (256, 512, 1024)),
            ).pack_block
            warps = best_existing_config(
                args,
                key,
                (Config(comm), Config(comm, pack_warps=8)),
            ).pack_warps
            reverse = best_existing_config(
                args,
                key,
                (Config(comm), Config(comm, reverse=True)),
            ).reverse
            for push in (False, True):
                for use_ce in (False, True):
                    config = Config(
                        comm,
                        streams=stream,
                        push=push,
                        use_ce=use_ce,
                        pack_block=block,
                        pack_warps=warps,
                        reverse=reverse,
                        local_first=transport.local_first,
                    )
                    groups.setdefault((operator, cp, config), []).append(
                        matrix_entry(
                            args,
                            operator=operator,
                            model=model,
                            sequence=sequence,
                            config=config,
                            launch=launch,
                            mode="deferred",
                            scope="comm",
                            warmup=args.sweep_warmup,
                            iterations=args.sweep_iters,
                            output=sweep_path(args, key, config),
                            check=False,
                        )
                    )
    run_grouped(args, phase="sweep_refine", groups=groups)


def formal(args: argparse.Namespace) -> None:
    groups: dict[tuple[str, int, Config], list[dict[str, object]]] = {}
    for operator, model, sequence, cp in cases(args):
        for launch in args.launches:
            key = case_key(operator, model, sequence, cp, launch)
            finalists = ranked_sweep(args, key)[:args.formal_shortlist]
            for mode in args.weight_modes:
                for _, config, _ in finalists:
                    output = args.results / "formal" / key / f"{mode}_{config.tag}.json"
                    groups.setdefault((operator, cp, config), []).append(matrix_entry(
                        args, operator=operator, model=model, sequence=sequence,
                        config=config, launch=launch, mode=mode, scope="full",
                        warmup=args.formal_warmup, iterations=args.formal_iters,
                        output=output, check=sequence == 1024,
                    ))
    run_grouped(args, phase="formal", groups=groups)


def summary(args: argparse.Namespace) -> None:
    rows: list[dict[str, object]] = []
    for operator, model, sequence, cp in cases(args):
        for launch in args.launches:
            key = case_key(operator, model, sequence, cp, launch)
            for mode in args.weight_modes:
                candidates = [
                    path for path in (args.results / "formal" / key).glob(f"{mode}_*.json")
                    if valid_result(
                        path,
                        checked=sequence == 1024,
                        expected_scope="full",
                    )
                ]
                if not candidates:
                    raise RuntimeError(f"no formal TE-UB result for {key}/{mode}")
                path = min(candidates, key=lambda p: float(load(p)["results"]["total"]["p50_ms"]))
                data = load(path)
                config = config_from_result(data)
                stats = data["results"]["total"]
                data_stats = data["results"].get("data", {})
                weight_stats = data["results"].get("weight", {})
                correctness = data.get("correctness", {})
                run_config = data.get("config", {})
                plan = data.get("cublaslt_plan") or {}
                software = data.get("software") or {}
                devices = data.get("devices") or []
                environment = data.get("environment") or {}
                formal_warmup = int(run_config.get("warmup", args.formal_warmup))
                formal_iters = int(run_config.get("iters", args.formal_iters))
                b_mnk = data["shape"]["b_mnk"]
                w_mnk = data["shape"]["w_mnk"]
                local_tokens = sequence // cp
                projection_width = (
                    model.qkv_width if operator == "qkv"
                    else model.attention_width
                )
                expected_b_mnk = (
                    [local_tokens, model.hidden, projection_width]
                    if operator == "qkv"
                    else [local_tokens, projection_width, model.hidden]
                )
                expected_w_mnk = (
                    [projection_width, model.hidden, local_tokens]
                    if operator == "qkv"
                    else [model.hidden, projection_width, local_tokens]
                )
                if b_mnk != expected_b_mnk or w_mnk != expected_w_mnk:
                    raise ValueError(
                        f"TE-UB backward MNK mismatch for "
                        f"{operator}/{model.name}/S{sequence}/CP{cp}: "
                        f"B={b_mnk}, W={w_mnk}"
                    )
                if data.get("head_geometry") != {
                    "q_heads": model.q_heads,
                    "kv_heads": model.kv_heads,
                    "head_dim": model.head_dim,
                }:
                    raise ValueError(
                        f"TE-UB head geometry mismatch for "
                        f"{operator}/{model.name}/S{sequence}/CP{cp}"
                    )
                rows.append({
                    "operator": operator,
                    "model": model.name,
                    "suite": model.suite,
                    "aliases": model.aliases,
                    "global_seq": sequence,
                    "cp": cp,
                    "launch": launch,
                    "weight_mode": mode,
                    "data_gemm_mode": data.get("data_gemm_mode", "single"),
                    "te_userbuffers_schema": data["schema"],
                    "numerical_contract": data["numerical_contract"],
                    "ub_object_lifecycle": data["ub_object_lifecycle"],
                    "cublaslt_plan_lifecycle": data[
                        "cublaslt_plan_lifecycle"
                    ],
                    "qkv_pack_kernel": data["qkv_pack_kernel"],
                    "ub_unpack_kernel": data["ub_unpack_kernel"],
                    "weight_accumulation_beta": data["weight_accumulation_beta"],
                    "b_m": b_mnk[0], "b_n": b_mnk[1], "b_k": b_mnk[2],
                    "w_m": w_mnk[0], "w_n": w_mnk[1], "w_k": w_mnk[2],
                    "q_heads": model.q_heads,
                    "kv_heads": model.kv_heads,
                    "head_dim": model.head_dim,
                    "result_source": (
                        "adapted_te_userbuffers_backward_formal_"
                        f"{formal_warmup}w{formal_iters}i"
                    ),
                    "timing": data["timing"],
                    "timed_boundary": data["timed_boundary"],
                    "rank_reduction": "MAX",
                    "torch_version": software.get("torch"),
                    "transformer_engine_version": software.get(
                        "transformer_engine"
                    ),
                    "transformer_engine_path": software.get(
                        "transformer_engine_path"
                    ),
                    "torch_cuda_version": software.get("torch_cuda"),
                    "nccl_version": software.get("nccl"),
                    "device_names": "|".join(
                        str(device.get("name"))
                        for device in devices
                        if isinstance(device, dict)
                    ),
                    "device_sm_counts": "|".join(
                        str(device.get("sm_count"))
                        for device in devices
                        if isinstance(device, dict)
                    ),
                    "cublas_workspace_config": environment.get(
                        "CUBLAS_WORKSPACE_CONFIG"
                    ),
                    **asdict(config),
                    "math_sm": config.math_sm,
                    "p50_ms": stats["p50_ms"],
                    "p95_ms": stats["p95_ms"],
                    "p50_tflops_per_gpu": stats["p50_tflops_per_gpu"],
                    "data_p50_ms": data_stats.get("p50_ms"),
                    "data_p95_ms": data_stats.get("p95_ms"),
                    "weight_p50_ms": weight_stats.get("p50_ms"),
                    "weight_p95_ms": weight_stats.get("p95_ms"),
                    "cublaslt_algo_id": plan.get("algo_id"),
                    "cublaslt_tile_id": plan.get("tile_id"),
                    "cublaslt_stages_id": plan.get("stages_id"),
                    "cublaslt_workspace_bytes": plan.get("workspace_bytes"),
                    "cublaslt_tune_ms": plan.get("tune_ms"),
                    "cublaslt_waves": plan.get("waves"),
                    "cublaslt_plan_cache_hit": data.get("cublaslt_plan_cache_hit"),
                    "correctness_scope": "exact" if sequence == 1024 else "same_geometry_s1k",
                    "route_mismatches": correctness.get("route_mismatches"),
                    "route_max_abs": correctness.get("route_max_abs"),
                    "dgrad_max_abs": correctness.get("dgrad_max_abs"),
                    "wgrad_max_abs": correctness.get("wgrad_max_abs"),
                    "main_grad_beta1_once_max_abs": correctness.get(
                        "main_grad_beta1_once_max_abs"
                    ),
                    "main_grad_beta1_twice_max_abs": correctness.get(
                        "main_grad_beta1_twice_max_abs"
                    ),
                    "deterministic_route_mismatches": correctness.get(
                        "deterministic_route_mismatches"
                    ),
                    "deterministic_dgrad_mismatches": correctness.get(
                        "deterministic_dgrad_mismatches"
                    ),
                    "deterministic_wgrad_mismatches": correctness.get(
                        "deterministic_wgrad_mismatches"
                    ),
                })
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (args.results / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    if args.publish_results:
        for operator in args.operators:
            selected_rows = [row for row in rows if row["operator"] == operator]
            dump_rows(
                selected_rows,
                operator_result_dir(operator)
                / f"{operator}_backward_te_userbuffers_summary",
            )
    write_comparisons(args, rows)
    print(f"SUMMARY {len(rows)} TE-UB backward rows", flush=True)


def fused_summary_path(operator: str) -> Path:
    if operator == "qkv":
        return (
            ROOT / "results/QKVproj-backward/qkv_backward_shape_bench/"
            "qkv_backward_summary.json"
        )
    return (
        ROOT / "results/Oproj-backward/oproj_backward_shape_bench/"
        "oproj_backward_summary.json"
    )


def operator_result_dir(operator: str) -> Path:
    return fused_summary_path(operator).parent


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


def write_comparisons(
    args: argparse.Namespace, teub_rows: list[dict[str, object]]
) -> None:
    by_key = {
        (
            str(row["operator"]), str(row["model"]), int(row["global_seq"]),
            int(row["cp"]), str(row["launch"]), str(row["weight_mode"]),
        ): row
        for row in teub_rows
    }
    for operator in args.operators:
        fused_payload = load(fused_summary_path(operator))
        if not isinstance(fused_payload, list):
            raise TypeError(f"invalid fused summary for {operator}")
        selected = {
            (model.name, sequence, cp)
            for candidate_operator, model, sequence, cp in cases(args)
            if candidate_operator == operator
        }
        comparison: list[dict[str, object]] = []
        for fused in fused_payload:
            model = str(fused["model"])
            sequence = int(fused["global_seq"])
            cp = int(fused["cp"])
            if (model, sequence, cp) not in selected:
                continue
            row: dict[str, object] = {
                "operator": operator,
                "model": model,
                "suite": fused["suite"],
                "aliases": fused["aliases"],
                "global_seq": sequence,
                "cp": cp,
                "local_tokens": fused["local_tokens"],
                "hidden": fused["hidden"],
                "projection_width": fused["projection_width"],
                "b_m": fused["b_m"], "b_n": fused["b_n"], "b_k": fused["b_k"],
                "w_m": fused["w_m"], "w_n": fused["w_n"], "w_k": fused["w_k"],
                "q_heads": fused["q_heads"],
                "kv_heads": fused["kv_heads"],
                "head_dim": fused["head_dim"],
                "warmup": args.formal_warmup,
                "iterations": args.formal_iters,
                "timing": "per-sample max-rank CUDA-event critical path",
            }
            for launch in args.launches:
                for mode in args.weight_modes:
                    key = (operator, model, sequence, cp, launch, mode)
                    if key not in by_key:
                        raise KeyError(f"missing TE-UB result {key}")
                    teub = by_key[key]
                    if any(
                        int(teub[name]) != int(fused[name])
                        for name in (
                            "b_m", "b_n", "b_k", "w_m", "w_n", "w_k"
                        )
                    ):
                        raise ValueError(
                            f"fused/TE-UB backward MNK mismatch for {key}"
                        )
                    prefix = f"{launch}_{mode}"
                    fused_ms = float(fused[f"{prefix}_total_p50_ms"])
                    teub_ms = float(teub["p50_ms"])
                    beta = 1 if mode == "deferred" else 0
                    cublas_ms = float(fused[f"cublas_total_beta{beta}_p50_ms"])
                    forward_tflops = float(
                        fused[f"{prefix}_forward_p50_tflops_per_gpu"]
                    )
                    teub_tflops = float(teub["p50_tflops_per_gpu"])
                    row["te_userbuffers_schema"] = teub[
                        "te_userbuffers_schema"
                    ]
                    row["te_userbuffers_numerical_contract"] = teub[
                        "numerical_contract"
                    ]
                    row["transformer_engine_version"] = teub[
                        "transformer_engine_version"
                    ]
                    row["transformer_engine_path"] = teub[
                        "transformer_engine_path"
                    ]
                    row["torch_version"] = teub["torch_version"]
                    row["torch_cuda_version"] = teub["torch_cuda_version"]
                    row["nccl_version"] = teub["nccl_version"]
                    row["device_names"] = teub["device_names"]
                    row["device_sm_counts"] = teub["device_sm_counts"]
                    row["cublas_workspace_config"] = teub[
                        "cublas_workspace_config"
                    ]
                    row[f"{prefix}_fused_p50_ms"] = fused_ms
                    row[f"{prefix}_fused_p95_ms"] = float(
                        fused[f"{prefix}_total_p95_ms"]
                    )
                    row[f"{prefix}_fused_p50_tflops_per_gpu"] = float(
                        fused[f"{prefix}_total_p50_tflops_per_gpu"]
                    )
                    row[f"{prefix}_te_userbuffers_p50_ms"] = teub_ms
                    row[f"{prefix}_te_userbuffers_p95_ms"] = float(teub["p95_ms"])
                    row[f"{prefix}_te_userbuffers_p50_tflops_per_gpu"] = float(
                        teub["p50_tflops_per_gpu"]
                    )
                    row[f"{prefix}_te_userbuffers_data_p50_ms"] = teub[
                        "data_p50_ms"
                    ]
                    row[f"{prefix}_te_userbuffers_weight_p50_ms"] = teub[
                        "weight_p50_ms"
                    ]
                    row[f"{prefix}_forward_p50_tflops_per_gpu"] = forward_tflops
                    row[f"{prefix}_cublas_p50_ms"] = cublas_ms
                    row[f"{prefix}_cublas_p50_tflops_per_gpu"] = float(
                        fused[f"cublas_total_beta{beta}_p50_tflops_per_gpu"]
                    )
                    row[f"{prefix}_fused_speedup_over_te_userbuffers"] = (
                        teub_ms / fused_ms
                    )
                    row[f"{prefix}_fused_throughput_as_forward_percent"] = float(
                        fused[f"{prefix}_throughput_as_forward_percent"]
                    )
                    row[f"{prefix}_fused_throughput_as_cublas_percent"] = float(
                        fused[f"{prefix}_throughput_as_cublas_percent"]
                    )
                    row[f"{prefix}_te_userbuffers_throughput_as_forward_percent"] = (
                        100.0 * teub_tflops / forward_tflops
                    )
                    row[f"{prefix}_te_userbuffers_as_cublas_percent"] = (
                        100.0 * cublas_ms / teub_ms
                    )
                    row[f"{prefix}_te_userbuffers_config"] = (
                        f"c{teub['comm_sm']}/s{teub['streams']}/"
                        f"{'push' if teub['push'] else 'pull'}/"
                        f"{'ce' if teub['use_ce'] else 'sm'}/"
                        f"b{teub['pack_block']}/w{teub['pack_warps']}/"
                        f"{'rev' if teub['reverse'] else 'fwd'}"
                    )
                    row[f"{prefix}_te_userbuffers_numerical_contract"] = teub[
                        "numerical_contract"
                    ]
                    row[f"{prefix}_te_userbuffers_unpack_kernel"] = teub[
                        "ub_unpack_kernel"
                    ]
                    row[f"{prefix}_weight_accumulation_beta"] = beta
            comparison.append(row)
        comparison.sort(
            key=lambda item: (
                int(item["cp"]), str(item["model"]), int(item["global_seq"])
            )
        )
        stem = (
            operator_result_dir(operator)
            / f"{operator}_backward_te_userbuffers_comparison"
            if args.publish_results
            else args.results / "summary" / f"{operator}_te_userbuffers_comparison"
        )
        dump_rows(comparison, stem)
        lines = [
            f"# {operator.upper()} backward：fused 与 TE Userbuffers",
            "",
            "比值为 `TE Userbuffers p50 / fused p50`；大于 1 表示 fused 更快。",
            "TE Userbuffers 正式列先完成完整路由，再执行一次完整 BF16 GEMM，"
            "避免把分段 beta=1 的累加顺序误差当作强基线收益。",
            "",
        ]
        for mode in args.weight_modes:
            lines.extend((
                f"## {'ZeroBubble B/W 分离' if mode == 'deferred' else '普通 B→W'}",
                "",
                "| CP | 模型 | S | fused Eager / TEUB / 加速 | "
                "fused Graph / TEUB / 加速 |",
                "|---:|---|---:|---:|---:|",
            ))
            for row in comparison:
                eager = f"eager_{mode}"
                graph = f"graph_{mode}"
                lines.append(
                    f"| {row['cp']} | {row['model']} | "
                    f"{sequence_label(int(row['global_seq']))} | "
                    f"{row[f'{eager}_fused_p50_ms']:.4f} / "
                    f"{row[f'{eager}_te_userbuffers_p50_ms']:.4f} / "
                    f"{row[f'{eager}_fused_speedup_over_te_userbuffers']:.3f}x | "
                    f"{row[f'{graph}_fused_p50_ms']:.4f} / "
                    f"{row[f'{graph}_te_userbuffers_p50_ms']:.4f} / "
                    f"{row[f'{graph}_fused_speedup_over_te_userbuffers']:.3f}x |"
                )
            eager_values = [
                float(row[f"eager_{mode}_fused_speedup_over_te_userbuffers"])
                for row in comparison
            ]
            graph_values = [
                float(row[f"graph_{mode}_fused_speedup_over_te_userbuffers"])
                for row in comparison
            ]
            eager_forward = [
                float(
                    row[
                        f"eager_{mode}_te_userbuffers_throughput_as_forward_percent"
                    ]
                )
                for row in comparison
            ]
            graph_forward = [
                float(
                    row[
                        f"graph_{mode}_te_userbuffers_throughput_as_forward_percent"
                    ]
                )
                for row in comparison
            ]
            eager_cublas = [
                float(row[f"eager_{mode}_te_userbuffers_as_cublas_percent"])
                for row in comparison
            ]
            graph_cublas = [
                float(row[f"graph_{mode}_te_userbuffers_as_cublas_percent"])
                for row in comparison
            ]
            lines.extend((
                "",
                f"Eager：{sum(value > 1 for value in eager_values)}/{len(eager_values)} "
                f"胜，几何平均 {geometric_mean(eager_values):.3f}x；"
                f"Graph：{sum(value > 1 for value in graph_values)}/{len(graph_values)} "
                f"胜，几何平均 {geometric_mean(graph_values):.3f}x。",
                f"TE Userbuffers 吞吐相对同 shape 前向的中位数："
                f"Eager {median(eager_forward):.1f}%，Graph {median(graph_forward):.1f}%；"
                f"相对经典 cuBLAS 纯 B+W 的中位数："
                f"Eager {median(eager_cublas):.1f}%，Graph {median(graph_cublas):.1f}%。",
                "",
            ))
        stem.with_suffix(".md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.phase in ("sweep", "all"):
        sweep(args)
    if args.phase in ("refine", "all"):
        refine(args)
    if args.phase in ("formal", "all"):
        formal(args)
    if args.phase in ("summary", "all"):
        summary(args)


if __name__ == "__main__":
    main()
