#!/usr/bin/env python3
"""Tune the adapted TE Userbuffers QKV-projection -> A2A baseline."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from qkv_shape_bench import (
    CONTEXT_PARALLEL,
    DEFAULT_MODELS,
    MODELS,
    SEQUENCES,
    VISIBLE_DEVICES,
)


ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/home/chen/miniforge3/envs/mmunlearner/bin/python")
TE_ROOT = Path("/home/chen/workspace/source_code/TransformerEngine")


@dataclass(frozen=True)
class Config:
    comm_sm: int
    streams: int = 1
    push: bool = True
    use_ce: bool = False
    pack_block: int = 512
    pack_warps: int = 4
    reverse: bool = False
    local_first: bool = False

    @property
    def math_sm(self) -> int:
        return 132 - self.comm_sm

    @property
    def tag(self) -> str:
        return (
            f"c{self.comm_sm}_m{self.math_sm}_s{self.streams}_"
            f"{'push' if self.push else 'pull'}_"
            f"{'ce' if self.use_ce else 'sm'}_"
            f"b{self.pack_block}_w{self.pack_warps}_"
            f"{'rev' if self.reverse else 'fwd'}_"
            f"{'localfirst' if self.local_first else 'remotefirst'}"
        )


def csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("sweep", "formal", "summary", "all"), default="all")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seqs", type=csv_ints, default=SEQUENCES)
    parser.add_argument("--cps", type=csv_ints, default=CONTEXT_PARALLEL)
    parser.add_argument("--comm-sm", type=csv_ints, default=(4, 8, 12, 16, 20, 24))
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "results" / "QKVproj-a2a" / "te_userbuffers_shape_bench",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sweep-warmup", type=int, default=1)
    parser.add_argument("--sweep-iters", type=int, default=3)
    parser.add_argument("--formal-warmup", type=int, default=10)
    parser.add_argument("--formal-iters", type=int, default=50)
    parser.add_argument("--formal-shortlist", type=int, default=3)
    return parser.parse_args()


def cases(args: argparse.Namespace):
    for name in (item for item in args.models.split(",") if item):
        if name not in MODELS:
            raise ValueError(f"unknown model: {name}")
        model = MODELS[name]
        for seq in args.seqs:
            for cp in args.cps:
                if (
                    seq % cp == 0
                    and model.q_heads % cp == 0
                    and model.kv_heads % cp == 0
                ):
                    yield name, model, seq, cp


def case_key(name: str, seq: int, cp: int) -> str:
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
        Config(comm, local_first=True),
    )


def matrix_entry(
    *, hidden: int, q_heads: int, kv_heads: int, head_dim: int,
    seq: int, config: Config, warmup: int, iters: int,
    output: Path, tune_warmup: int, tune_iters: int, check: bool,
) -> dict[str, object]:
    return {
        "json_out": str(output),
        "arguments": {
            "global_seq": seq,
            "hidden": hidden,
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "batch": 1,
            "warmup": warmup,
            "iters": iters,
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
            "tune_warmup": tune_warmup,
            "tune_iters": tune_iters,
            "workspace_mib": 64,
            "cuda_graph": True,
            "check": check,
        },
    }


def environment(cp: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = VISIBLE_DEVICES[cp]
    env["PYTHONPATH"] = str(TE_ROOT)
    env["LD_LIBRARY_PATH"] = f"{TE_ROOT}:/usr/local/cuda/lib64"
    return env


def run_matrix(
    *, cp: int, entries: list[dict[str, object]], manifest: Path, resume: bool,
) -> None:
    # A torchrun cold start initializes NCCL, TE Userbuffers, and cuBLASLt.
    # Keep every candidate for one shape in the same process group; the
    # single-case runner caches the matching cuBLASLt plan across entries.
    pending = []
    for entry in entries:
        output = Path(str(entry["json_out"]))
        if resume and valid_result(output):
            continue
        pending.append(entry)
    if not pending:
        return

    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_suffix(manifest.suffix + ".tmp")
    temporary.write_text(json.dumps(pending, indent=2) + "\n")
    temporary.replace(manifest)
    command = [
        str(PYTHON), "-m", "torch.distributed.run", "--standalone",
        f"--nproc-per-node={cp}",
        str(Path(__file__).with_name("te_userbuffers_qkv.py")),
        "--matrix-manifest", str(manifest),
    ]
    print(f"RUN-MATRIX {manifest} ({len(pending)} configurations)", flush=True)
    completed = subprocess.run(
        command, cwd=ROOT, env=environment(cp), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if completed.returncode:
        print(completed.stdout[-12000:], flush=True)
        completed.check_returncode()
    invalid = [
        str(entry["json_out"])
        for entry in pending
        if not valid_result(Path(str(entry["json_out"])))
    ]
    if invalid:
        raise RuntimeError(f"matrix run did not produce valid results: {invalid}")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def valid_result(path: Path) -> bool:
    if not path.exists():
        return False
    data = load(path)
    check = data.get("correctness", {})
    if data.get("mode") != "te_userbuffers_adapted_qkv_a2a":
        return False
    if not data.get("config", {}).get("check", False):
        return True
    return (
        check.get("max_abs", 1.0) <= 0.01
        and check.get("remote_recv_mismatches", -1) == 0
        and check.get("post_graph_max_abs", 1.0) <= 0.01
    )


def config_from_result(data: dict) -> Config:
    cfg = data["config"]
    return Config(
        cfg["num_comm_sm"], cfg["num_streams"], cfg["push"], cfg["use_ce"],
        cfg["pack_block"], cfg["pack_warps"], cfg["reverse"], cfg["local_first"],
    )


def ranked_sweep(results: Path, key: str) -> list[tuple[float, Config, Path]]:
    records: list[tuple[float, Config, Path]] = []
    for path in (results / "sweep" / key).glob("*.json"):
        if not valid_result(path):
            continue
        data = load(path)
        p50 = data["results"]["te_userbuffers_qkv_boundary"]["p50_ms"]
        records.append((p50, config_from_result(data), path))
    if not records:
        raise RuntimeError(f"no valid sweep result for {key}")
    return sorted(records, key=lambda item: item[0])


def should_check(model, seq: int, cp: int) -> bool:
    del model, cp
    # One exact CP4 and CP8 conformance run per head geometry is sufficient;
    # repeating the same layout proof at long S only materializes multi-GiB
    # all-gather references and does not exercise a new routing rule.
    return seq == min(SEQUENCES)


def sweep(args: argparse.Namespace) -> None:
    for name, model, seq, cp in cases(args):
        key = case_key(name, seq, cp)
        check = should_check(model, seq, cp)
        base_entries = []
        for comm in args.comm_sm:
            config = Config(comm)
            output = args.results / "sweep" / key / f"{config.tag}.json"
            base_entries.append(matrix_entry(
                hidden=model.hidden, q_heads=model.q_heads,
                kv_heads=model.kv_heads, head_dim=model.head_dim,
                seq=seq, config=config,
                warmup=args.sweep_warmup, iters=args.sweep_iters,
                output=output, tune_warmup=5, tune_iters=15, check=check,
            ))
        run_matrix(
            cp=cp,
            entries=base_entries,
            manifest=args.results / "manifests" / "sweep" / f"{key}_comm.json",
            resume=args.resume,
        )

        comm_records = []
        for comm in args.comm_sm:
            config = Config(comm)
            output = args.results / "sweep" / key / f"{config.tag}.json"
            if valid_result(output):
                p50 = load(output)["results"]["te_userbuffers_qkv_boundary"]["p50_ms"]
                comm_records.append((p50, comm))
        if not comm_records:
            raise RuntimeError(f"no valid communication sweep for {key}")
        best_comm = min(comm_records)[1]
        structural_entries = []
        for config in structural_candidates(best_comm):
            output = args.results / "sweep" / key / f"{config.tag}.json"
            structural_entries.append(matrix_entry(
                hidden=model.hidden, q_heads=model.q_heads,
                kv_heads=model.kv_heads, head_dim=model.head_dim,
                seq=seq, config=config,
                warmup=args.sweep_warmup, iters=args.sweep_iters,
                output=output, tune_warmup=5, tune_iters=15, check=check,
            ))
        run_matrix(
            cp=cp,
            entries=structural_entries,
            manifest=args.results / "manifests" / "sweep" / f"{key}_structure.json",
            resume=args.resume,
        )


def formal(args: argparse.Namespace) -> None:
    for name, model, seq, cp in cases(args):
        key = case_key(name, seq, cp)
        entries = []
        for _, config, _ in ranked_sweep(args.results, key)[: args.formal_shortlist]:
            output = args.results / "formal" / f"{key}_{config.tag}.json"
            entries.append(matrix_entry(
                hidden=model.hidden, q_heads=model.q_heads,
                kv_heads=model.kv_heads, head_dim=model.head_dim,
                seq=seq, config=config,
                warmup=args.formal_warmup, iters=args.formal_iters,
                output=output, tune_warmup=10, tune_iters=30,
                check=should_check(model, seq, cp),
            ))
        run_matrix(
            cp=cp,
            entries=entries,
            manifest=args.results / "manifests" / "formal" / f"{key}.json",
            resume=args.resume,
        )


def summarize_results(args: argparse.Namespace) -> None:
    rows = []
    for name, model, seq, cp in cases(args):
        key = case_key(name, seq, cp)
        candidates = [
            path for path in (args.results / "formal").glob(f"{key}_*.json")
            if valid_result(path)
        ]
        if not candidates:
            raise RuntimeError(f"no valid formal result for {key}")
        path = min(
            candidates,
            key=lambda item: load(item)["results"]["te_userbuffers_qkv_boundary"]["p50_ms"],
        )
        data = load(path)
        config = config_from_result(data)
        stats = data["results"]["te_userbuffers_qkv_boundary"]
        correctness = data.get("correctness", {})
        checked = bool(data.get("config", {}).get("check", False))
        m, n, k = (data["gemm_shape"][dim] for dim in ("m", "n", "k"))
        rows.append({
            "model": name, "suite": model.suite, "aliases": model.aliases,
            "global_seq": seq, "cp": cp, "m": m, "n": n, "k": k,
            "q_heads": model.q_heads, "kv_heads": model.kv_heads,
            "head_dim": model.head_dim, "max_context": model.max_context,
            "within_native_context": model.max_context is None or seq <= model.max_context,
            "result_source": "adapted_te_userbuffers_formal_10w50i",
            **asdict(config), "math_sm": config.math_sm,
            "p50_ms": stats["p50_ms"], "p95_ms": stats["p95_ms"],
            "p50_tflops_per_gpu": 2.0 * m * n * k / stats["p50_ms"] / 1.0e9,
            "correctness_scope": "exact" if checked else "same_head_geometry_s1k",
            "max_abs": correctness.get("max_abs"),
            "remote_recv_mismatches": correctness.get("remote_recv_mismatches"),
            "post_graph_max_abs": correctness.get("post_graph_max_abs"),
        })
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (args.results / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
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
        summarize_results(args)


if __name__ == "__main__":
    main()
