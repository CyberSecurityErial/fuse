#!/usr/bin/env python3
"""Formal QKV/OProj backward matrices for the v10 fused operators."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "results" / "backward" / "v10_shape_bench"
SEQUENCES = (1024, 4096, 16384, 131072, 262144, 524288)
CONTEXT_PARALLEL = (4, 8)
VISIBLE_DEVICES = {
    4: os.environ.get("FUSE_CP4_DEVICES", "0,2,4,5"),
    8: os.environ.get("FUSE_CP8_DEVICES", "0,1,2,3,4,5,6,7"),
}


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

    @property
    def attention_width(self) -> int:
        return self.q_heads * self.head_dim


QKV_MODELS = {
    model.name: model
    for model in (
        Model("artificial_small", "artificial", 4096, 16, 8, 128,
              "Mx4096x4096 QKV stress shape"),
        Model("artificial_medium", "artificial", 5120, 24, 8, 128,
              "Mx5120x5120 QKV stress shape"),
        Model("artificial_large", "artificial", 16384, 40, 8, 128,
              "Mx7168x16384 QKV stress shape"),
        Model("production_qwen_dense", "model", 2048, 16, 8, 128,
              "current production Qwen dense"),
        Model("nanbeige42_3b", "model", 3072, 48, 8, 128,
              "Nanbeige4.2-3B", 262144),
        Model("llama3_8b", "model", 4096, 32, 8, 128,
              "Llama 3/3.1 8B"),
        Model("qwen25_14b_32b", "model", 5120, 40, 8, 128,
              "Qwen2.5 14B/32B"),
        Model("llama31_405b", "model", 16384, 128, 8, 128,
              "Llama 3.1 405B"),
    )
}

OPROJ_MODELS = {
    model.name: model
    for model in (
        Model("representative_small", "artificial", 4096, 32, 8, 128,
              "Mx4096x4096 OProj stress shape"),
        Model("representative_medium", "artificial", 5120, 40, 8, 128,
              "Mx5120x5120 OProj stress shape"),
        Model("representative_large", "artificial", 7168, 128, 8, 128,
              "Mx7168x16384 OProj stress shape"),
        Model("production_qwen_dense", "model", 2048, 16, 8, 128,
              "current production Qwen dense"),
        Model("llama3_8b", "model", 4096, 32, 8, 128,
              "Llama 3/3.1 8B"),
        Model("qwen25_14b_32b", "model", 5120, 40, 8, 128,
              "Qwen2.5 14B/32B"),
        Model("llama31_405b", "model", 16384, 128, 8, 128,
              "Llama 3.1 405B"),
        Model("nanbeige42_3b", "model", 3072, 48, 8, 128,
              "Nanbeige4.2-3B", 262144),
    )
}


def csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item for item in value.split(",") if item)


def csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", action="append", choices=("formal", "cublas", "aggregate"),
        required=True,
    )
    parser.add_argument(
        "--operators", type=csv_strings, default=("qkv", "oproj"),
    )
    parser.add_argument(
        "--launches", type=csv_strings, default=("eager", "graph"),
    )
    parser.add_argument(
        "--weight-modes", type=csv_strings, default=("deferred",),
        help="deferred records B and W separately; immediate records same-stream B->W",
    )
    parser.add_argument("--seqs", type=csv_ints, default=SEQUENCES)
    parser.add_argument("--cps", type=csv_ints, default=CONTEXT_PARALLEL)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--mpirun", type=Path,
        default=Path(os.environ.get("FUSE_MPIRUN", "/usr/bin/mpirun")),
    )
    parser.add_argument(
        "--bench", type=Path,
        default=ROOT / "build-v10-release" / "backward_mpi_bench",
    )
    parser.add_argument(
        "--cublas-bench", type=Path,
        default=ROOT / "build-v10-release" / "backward_cublas_mpi_bench",
    )
    parser.add_argument("--mpi-bind", default="none")
    parser.add_argument(
        "--batch-modes", action=argparse.BooleanOptionalAction, default=True,
        help="reuse one MPI/CUDA process for eager/graph and deferred/immediate",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    return parser.parse_args()


def models_for(operator: str) -> dict[str, Model]:
    if operator == "qkv":
        return QKV_MODELS
    if operator == "oproj":
        return OPROJ_MODELS
    raise ValueError(f"unknown operator: {operator}")


def raw_path(
    root: Path,
    operator: str,
    model: Model,
    sequence: int,
    cp: int,
    launch: str,
    weight_mode: str,
) -> Path:
    return (
        root / "raw" / operator
        / f"{model.name}_s{sequence}_cp{cp}_{launch}_{weight_mode}.json"
    )


def expected_mode(operator: str) -> str:
    return (
        "qkv_backward_a2a_dgrad_mpi"
        if operator == "qkv"
        else "oproj_backward_dgrad_a2a_mpi"
    )


def cublas_raw_path(
    root: Path,
    operator: str,
    model: Model,
    sequence: int,
    cp: int,
) -> Path:
    return (
        root / "raw_cublas" / operator
        / f"{model.name}_s{sequence}_cp{cp}.json"
    )


def weight_beta_for_mode(weight_mode: str) -> int:
    return 1 if weight_mode == "deferred" else 0


def valid_summary(summary: object, iterations: int, required: bool) -> bool:
    if not isinstance(summary, dict):
        return False
    samples = summary.get("samples_ms")
    if not isinstance(samples, list):
        return False
    if required and len(samples) != iterations:
        return False
    if not required and samples:
        return False
    return all(
        isinstance(value, (int, float)) and math.isfinite(value) and value > 0
        for value in samples
    )


def matches(
    path: Path,
    operator: str,
    model: Model,
    sequence: int,
    cp: int,
    launch: str,
    weight_mode: str,
    warmup: int,
    iterations: int,
) -> bool:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    projection = model.qkv_width if operator == "qkv" else model.attention_width
    shape = payload.get("shape", {})
    heads = payload.get("head_geometry", {})
    expected_b = (
        [sequence // cp, model.hidden, projection]
        if operator == "qkv"
        else [sequence // cp, projection, model.hidden]
    )
    expected_w = (
        [projection, model.hidden, sequence // cp]
        if operator == "qkv"
        else [model.hidden, projection, sequence // cp]
    )
    common = (
        payload.get("mode") == expected_mode(operator)
        and payload.get("launch") == launch
        and payload.get("weight_mode") == weight_mode
        and payload.get("backward_schema") == "v10_b_w_split_v1"
        and payload.get("backward_policy_model")
        == "route_task_wave_compute_residency_v3"
        and payload.get("launch_plan_cache") == "per_process_v1"
        and payload.get("zero_bubble_contract") == (
            "separate_B_and_W_with_operand_lease"
            if weight_mode == "deferred"
            else "same_stream_B_then_W"
        )
        and payload.get("weight_accumulation_beta")
        == weight_beta_for_mode(weight_mode)
        and payload.get("profiling_build") is False
        and payload.get("role_profile_requested") is False
        and payload.get("world_size") == cp
        and payload.get("warmup") == warmup
        and payload.get("iterations") == iterations
        and payload.get("requested_comm_ctas") == 0
        and payload.get("requested_gemm_policy") == "auto"
        and shape.get("local_tokens") == sequence // cp
        and shape.get("hidden") == model.hidden
        and shape.get("projection_width") == projection
        and payload.get("b_mnk") == expected_b
        and payload.get("w_mnk") == expected_w
        and heads.get("batch") == 1
        and heads.get("q_heads") == model.q_heads
        and heads.get("kv_heads") == model.kv_heads
        and heads.get("head_dim") == model.head_dim
        and isinstance(payload.get("comm_ctas"), int)
        and payload.get("comm_ctas", 0) > 0
    )
    if not common:
        return False
    if sequence == 1024 and payload.get("correctness") == "not_run":
        return False
    deferred = weight_mode == "deferred"
    return (
        valid_summary(payload.get("data_phase"), iterations, deferred)
        and valid_summary(payload.get("weight_phase"), iterations, deferred)
        and valid_summary(payload.get("total"), iterations, True)
    )


def cublas_matches(
    path: Path,
    operator: str,
    model: Model,
    sequence: int,
    cp: int,
    warmup: int,
    iterations: int,
) -> bool:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    projection = model.qkv_width if operator == "qkv" else model.attention_width
    shape = payload.get("shape", {})
    expected_b = (
        [sequence // cp, model.hidden, projection]
        if operator == "qkv"
        else [sequence // cp, projection, model.hidden]
    )
    expected_w = (
        [projection, model.hidden, sequence // cp]
        if operator == "qkv"
        else [model.hidden, projection, sequence // cp]
    )
    return (
        payload.get("mode") == "classic_cublas_backward_mpi"
        and payload.get("schema") == "v10_backward_cublas_v2"
        and payload.get("operator") == operator
        and payload.get("world_size") == cp
        and payload.get("warmup") == warmup
        and payload.get("iterations") == iterations
        and payload.get("timing") == "per-sample max-rank CUDA event"
        and shape.get("local_tokens") == sequence // cp
        and shape.get("hidden") == model.hidden
        and shape.get("projection_width") == projection
        and payload.get("b_mnk") == expected_b
        and payload.get("w_mnk") == expected_w
        and valid_summary(payload.get("b_gemm"), iterations, True)
        and valid_summary(payload.get("w_gemm_beta0"), iterations, True)
        and valid_summary(payload.get("w_gemm_beta1"), iterations, True)
        and valid_summary(payload.get("total_beta0"), iterations, True)
        and valid_summary(payload.get("total_beta1"), iterations, True)
    )


def cublas_formal(args: argparse.Namespace) -> None:
    cases: list[tuple[str, Model, int, int]] = []
    for operator in args.operators:
        for model in models_for(operator).values():
            for sequence in args.seqs:
                for cp in args.cps:
                    if sequence % cp == 0:
                        cases.append((operator, model, sequence, cp))
    for index, (operator, model, sequence, cp) in enumerate(cases, 1):
        output = cublas_raw_path(args.results, operator, model, sequence, cp)
        if args.resume and output.exists() and cublas_matches(
            output, operator, model, sequence, cp,
            args.warmup, args.iterations,
        ):
            print(f"[{index}/{len(cases)}] RESUME cuBLAS {output.name}", flush=True)
            continue
        output.unlink(missing_ok=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(args.mpirun), "-np", str(cp), "--bind-to", args.mpi_bind,
            str(args.cublas_bench),
            "--operator", operator,
            "--m", str(sequence // cp),
            "--hidden", str(model.hidden),
            "--q-heads", str(model.q_heads),
            "--kv-heads", str(model.kv_heads),
            "--head-dim", str(model.head_dim),
            "--warmup", str(args.warmup),
            "--iterations", str(args.iterations),
            "--json-out", str(output),
        ]
        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = VISIBLE_DEVICES[cp]
        print(
            f"[{index}/{len(cases)}] RUN cuBLAS {operator} {model.name} "
            f"S={sequence} CP={cp}",
            flush=True,
        )
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=args.timeout_seconds,
            )
        except BaseException:
            output.unlink(missing_ok=True)
            raise
        if completed.returncode != 0:
            output.unlink(missing_ok=True)
            print(completed.stdout, end="")
            print(completed.stderr, end="")
            raise RuntimeError(f"cuBLAS benchmark failed: {' '.join(command)}")
        if not cublas_matches(
            output, operator, model, sequence, cp,
            args.warmup, args.iterations,
        ):
            output.unlink(missing_ok=True)
            raise RuntimeError(f"invalid cuBLAS benchmark output: {output}")


def formal(args: argparse.Namespace) -> None:
    if (
        args.batch_modes
        and set(args.launches) == {"eager", "graph"}
        and len(args.launches) == 2
        and set(args.weight_modes) == {"deferred", "immediate"}
        and len(args.weight_modes) == 2
    ):
        formal_batched(args)
        return
    cases: list[tuple[str, Model, int, int, str, str]] = []
    for operator in args.operators:
        for model in models_for(operator).values():
            for sequence in args.seqs:
                for cp in args.cps:
                    if sequence % cp:
                        continue
                    for launch in args.launches:
                        for weight_mode in args.weight_modes:
                            cases.append(
                                (operator, model, sequence, cp, launch, weight_mode)
                            )
    for index, (operator, model, sequence, cp, launch, weight_mode) in enumerate(
        cases, 1
    ):
        output = raw_path(
            args.results, operator, model, sequence, cp, launch, weight_mode
        )
        if args.resume and output.exists() and matches(
            output, operator, model, sequence, cp, launch, weight_mode,
            args.warmup, args.iterations,
        ):
            print(f"[{index}/{len(cases)}] RESUME {output.name}", flush=True)
            continue
        output.unlink(missing_ok=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(args.mpirun), "-np", str(cp), "--bind-to", args.mpi_bind,
            str(args.bench),
            "--operator", operator,
            "--weight-mode", weight_mode,
            "--launch", launch,
            "--m", str(sequence // cp),
            "--hidden", str(model.hidden),
            "--batch", "1",
            "--q-heads", str(model.q_heads),
            "--kv-heads", str(model.kv_heads),
            "--head-dim", str(model.head_dim),
            "--comm-ctas", "0",
            "--weight-beta", str(weight_beta_for_mode(weight_mode)),
            "--gemm-policy", "auto",
            "--warmup", str(args.warmup),
            "--iterations", str(args.iterations),
            "--causal-load-balanced",
            "--check" if sequence == 1024 else "--no-check",
            "--json-out", str(output),
        ]
        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = VISIBLE_DEVICES[cp]
        print(
            f"[{index}/{len(cases)}] RUN {operator} {model.name} "
            f"S={sequence} CP={cp} {launch} {weight_mode}",
            flush=True,
        )
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=args.timeout_seconds,
            )
        except BaseException:
            output.unlink(missing_ok=True)
            raise
        if completed.returncode != 0:
            output.unlink(missing_ok=True)
            print(completed.stdout, end="")
            print(completed.stderr, end="")
            raise RuntimeError(f"benchmark failed: {' '.join(command)}")
        if not matches(
            output, operator, model, sequence, cp, launch, weight_mode,
            args.warmup, args.iterations,
        ):
            output.unlink(missing_ok=True)
            raise RuntimeError(f"invalid benchmark output: {output}")


def formal_batched(args: argparse.Namespace) -> None:
    cases: list[tuple[str, Model, int, int]] = []
    for operator in args.operators:
        for model in models_for(operator).values():
            for sequence in args.seqs:
                for cp in args.cps:
                    if sequence % cp == 0:
                        cases.append((operator, model, sequence, cp))
    variants = tuple(
        (launch, weight_mode)
        for launch in ("eager", "graph")
        for weight_mode in ("deferred", "immediate")
    )
    for index, (operator, model, sequence, cp) in enumerate(cases, 1):
        outputs = [
            raw_path(
                args.results, operator, model, sequence, cp, launch,
                weight_mode,
            )
            for launch, weight_mode in variants
        ]
        if args.resume and all(
            output.exists()
            and matches(
                output, operator, model, sequence, cp, launch, weight_mode,
                args.warmup, args.iterations,
            )
            for output, (launch, weight_mode) in zip(outputs, variants)
        ):
            print(
                f"[{index}/{len(cases)}] RESUME {operator} {model.name} "
                f"S={sequence} CP={cp}",
                flush=True,
            )
            continue
        for output in outputs:
            output.unlink(missing_ok=True)
            output.parent.mkdir(parents=True, exist_ok=True)
        prefix = outputs[0].parent / f"{model.name}_s{sequence}_cp{cp}"
        command = [
            str(args.mpirun), "-np", str(cp), "--bind-to", args.mpi_bind,
            str(args.bench),
            "--operator", operator,
            "--run-all-modes",
            "--json-prefix", str(prefix),
            "--m", str(sequence // cp),
            "--hidden", str(model.hidden),
            "--batch", "1",
            "--q-heads", str(model.q_heads),
            "--kv-heads", str(model.kv_heads),
            "--head-dim", str(model.head_dim),
            "--comm-ctas", "0",
            "--gemm-policy", "auto",
            "--warmup", str(args.warmup),
            "--iterations", str(args.iterations),
            "--causal-load-balanced",
            "--check" if sequence == 1024 else "--no-check",
        ]
        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = VISIBLE_DEVICES[cp]
        print(
            f"[{index}/{len(cases)}] RUN {operator} {model.name} "
            f"S={sequence} CP={cp} all-modes",
            flush=True,
        )
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=args.timeout_seconds,
            )
        except BaseException:
            for output in outputs:
                output.unlink(missing_ok=True)
            raise
        if completed.returncode != 0:
            for output in outputs:
                output.unlink(missing_ok=True)
            print(completed.stdout, end="")
            print(completed.stderr, end="")
            raise RuntimeError(f"benchmark failed: {' '.join(command)}")
        for output, (launch, weight_mode) in zip(outputs, variants):
            if not matches(
                output, operator, model, sequence, cp, launch, weight_mode,
                args.warmup, args.iterations,
            ):
                for candidate in outputs:
                    candidate.unlink(missing_ok=True)
                raise RuntimeError(f"invalid benchmark output: {output}")


def load_forward_rows(operator: str) -> dict[tuple[str, int, int], dict]:
    path = (
        ROOT / "results" / "QKVproj-a2a" / "qkv_shape_bench"
        / "fused_summary.json"
        if operator == "qkv"
        else ROOT / "results" / "a2a-Oproj" / "oproj_mixed_shape_bench"
        / "comparison_summary.json"
    )
    if not path.exists():
        return {}
    rows = json.loads(path.read_text())
    return {(row["model"], row["global_seq"], row["cp"]): row for row in rows}


def phase_fields(prefix: str, payload: dict, phase_flops: float) -> dict:
    summary = payload[prefix]
    p50 = float(summary["p50_ms"])
    p95 = float(summary["p95_ms"])
    return {
        f"{prefix}_p50_ms": p50,
        f"{prefix}_p95_ms": p95,
        f"{prefix}_p50_tflops_per_gpu": phase_flops / p50 / 1.0e9,
    }


def dump_rows(rows: list[dict], stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    temporary = stem.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(rows, indent=2) + "\n")
    os.replace(temporary, stem.with_suffix(".json"))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = stem.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, stem.with_suffix(".csv"))


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def sequence_label(sequence: int) -> str:
    return f"{sequence // 1024}K"


def write_markdown(rows: list[dict], stem: Path, operator: str) -> None:
    title = "QKV projection backward" if operator == "qkv" else "OProj backward"
    lines = [f"# {title} formal results", ""]
    lines.extend([
        "## 汇总",
        "",
        "吞吐按 B 与 W 两次 GEMM 的总 FLOPs 计算；“前向占比”是同一 shape、同一启动方式下的反向总吞吐除以前向融合吞吐。",
        "",
        "| 模式 | 启动 | 几何平均 / 中位 / 最低前向占比 | 达到前向95%的点 | 经典cuBLAS吞吐占比中位 | 989T MFU中位 |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for weight_mode in ("deferred", "immediate"):
        for launch in ("eager", "graph"):
            prefix = f"{launch}_{weight_mode}"
            ratios = [
                float(row[f"{prefix}_throughput_as_forward_percent"])
                for row in rows
                if f"{prefix}_throughput_as_forward_percent" in row
            ]
            mfus = [float(row[f"{prefix}_mfu_percent_of_989t"]) for row in rows]
            cublas_ratios = [
                float(row[f"{prefix}_throughput_as_cublas_percent"])
                for row in rows
            ]
            lines.append(
                f"| {'ZeroBubble B/W分离，beta=1' if weight_mode == 'deferred' else '普通同流 B→W，beta=0'} "
                f"| {'Eager' if launch == 'eager' else 'CUDA Graph'} "
                f"| {geometric_mean(ratios):.1f}% / {statistics.median(ratios):.1f}% / {min(ratios):.1f}% "
                f"| {sum(value >= 95.0 for value in ratios)}/{len(ratios)} "
                f"| {statistics.median(cublas_ratios):.1f}% "
                f"| {statistics.median(mfus):.1f}% |"
            )

    lines.extend([
        "",
        "## ZeroBubble：B/W 分离，main_grad 累加",
        "",
        "| CP | 模型 | S | B GEMM M×N×K | W GEMM M×N×K | 经典cuBLAS B / W(beta=1) / 总 ms · TFLOPS | Eager B / W / 总 ms | Eager TFLOPS / 前向 / cuBLAS占比 | Graph B / W / 总 ms | Graph TFLOPS / 前向 / cuBLAS占比 |",
        "|---:|---|---:|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        b_shape = f"{row['b_m']}×{row['b_n']}×{row['b_k']}"
        w_shape = f"{row['w_m']}×{row['w_n']}×{row['w_k']}"
        eager = "eager_deferred"
        graph = "graph_deferred"
        lines.append(
            f"| {row['cp']} | {row['model']} | {sequence_label(int(row['global_seq']))} "
            f"| `{b_shape}` | `{w_shape}` "
            f"| {row['cublas_b_p50_ms']:.4f} / {row['cublas_w_beta1_p50_ms']:.4f} / {row['cublas_total_beta1_p50_ms']:.4f} · {row['cublas_total_beta1_p50_tflops_per_gpu']:.1f} "
            f"| {row[f'{eager}_data_phase_p50_ms']:.4f} / {row[f'{eager}_weight_phase_p50_ms']:.4f} / {row[f'{eager}_total_p50_ms']:.4f} "
            f"| {row[f'{eager}_total_p50_tflops_per_gpu']:.1f} / {row[f'{eager}_throughput_as_forward_percent']:.1f}% / {row[f'{eager}_throughput_as_cublas_percent']:.1f}% "
            f"| {row[f'{graph}_data_phase_p50_ms']:.4f} / {row[f'{graph}_weight_phase_p50_ms']:.4f} / {row[f'{graph}_total_p50_ms']:.4f} "
            f"| {row[f'{graph}_total_p50_tflops_per_gpu']:.1f} / {row[f'{graph}_throughput_as_forward_percent']:.1f}% / {row[f'{graph}_throughput_as_cublas_percent']:.1f}% |"
        )

    lines.extend([
        "",
        "## 普通反向：同一 stream 内 B→W",
        "",
        "| CP | 模型 | S | B GEMM M×N×K | W GEMM M×N×K | 经典cuBLAS总(beta=0) ms / TFLOPS | Eager 总 ms / TFLOPS / 前向 / cuBLAS占比 | Graph 总 ms / TFLOPS / 前向 / cuBLAS占比 |",
        "|---:|---|---:|---|---|---:|---:|---:|",
    ])
    for row in rows:
        b_shape = f"{row['b_m']}×{row['b_n']}×{row['b_k']}"
        w_shape = f"{row['w_m']}×{row['w_n']}×{row['w_k']}"
        eager = "eager_immediate"
        graph = "graph_immediate"
        lines.append(
            f"| {row['cp']} | {row['model']} | {sequence_label(int(row['global_seq']))} "
            f"| `{b_shape}` | `{w_shape}` "
            f"| {row['cublas_total_beta0_p50_ms']:.4f} / {row['cublas_total_beta0_p50_tflops_per_gpu']:.1f} "
            f"| {row[f'{eager}_total_p50_ms']:.4f} / {row[f'{eager}_total_p50_tflops_per_gpu']:.1f} / {row[f'{eager}_throughput_as_forward_percent']:.1f}% / {row[f'{eager}_throughput_as_cublas_percent']:.1f}% "
            f"| {row[f'{graph}_total_p50_ms']:.4f} / {row[f'{graph}_total_p50_tflops_per_gpu']:.1f} / {row[f'{graph}_throughput_as_forward_percent']:.1f}% / {row[f'{graph}_throughput_as_cublas_percent']:.1f}% |"
        )
    temporary = stem.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines) + "\n")
    os.replace(temporary, stem.with_suffix(".md"))


def aggregate_operator(args: argparse.Namespace, operator: str) -> None:
    forward = load_forward_rows(operator)
    rows: list[dict] = []
    for model in models_for(operator).values():
        projection = model.qkv_width if operator == "qkv" else model.attention_width
        for sequence in args.seqs:
            for cp in args.cps:
                if sequence % cp:
                    continue
                m = sequence // cp
                phase_flops = 2.0 * m * model.hidden * projection
                row: dict[str, object] = {
                    "operator": operator,
                    "model": model.name,
                    "suite": model.suite,
                    "aliases": model.aliases,
                    "global_seq": sequence,
                    "cp": cp,
                    "local_tokens": m,
                    "hidden": model.hidden,
                    "projection_width": projection,
                    "q_heads": model.q_heads,
                    "kv_heads": model.kv_heads,
                    "head_dim": model.head_dim,
                    "max_context": model.max_context,
                    "within_native_context": (
                        model.max_context is None or sequence <= model.max_context
                    ),
                    "b_m": m,
                    "b_n": model.hidden if operator == "qkv" else projection,
                    "b_k": projection if operator == "qkv" else model.hidden,
                    "w_m": projection if operator == "qkv" else model.hidden,
                    "w_n": model.hidden if operator == "qkv" else projection,
                    "w_k": m,
                    "warmup": args.warmup,
                    "iterations": args.iterations,
                    "timing": "per-sample max-rank CUDA event",
                    "process_model": "MPI one process per GPU",
                }
                cublas_path = cublas_raw_path(
                    args.results, operator, model, sequence, cp
                )
                if not cublas_matches(
                    cublas_path, operator, model, sequence, cp,
                    args.warmup, args.iterations,
                ):
                    raise RuntimeError(
                        f"missing/invalid cuBLAS result: {cublas_path}"
                    )
                cublas_payload = json.loads(cublas_path.read_text())
                cublas_b = float(cublas_payload["b_gemm"]["p50_ms"])
                cublas_w_beta0 = float(
                    cublas_payload["w_gemm_beta0"]["p50_ms"]
                )
                cublas_w_beta1 = float(
                    cublas_payload["w_gemm_beta1"]["p50_ms"]
                )
                cublas_total_beta0 = float(
                    cublas_payload["total_beta0"]["p50_ms"]
                )
                cublas_total_beta1 = float(
                    cublas_payload["total_beta1"]["p50_ms"]
                )
                row["cublas_b_p50_ms"] = cublas_b
                row["cublas_w_beta0_p50_ms"] = cublas_w_beta0
                row["cublas_w_beta1_p50_ms"] = cublas_w_beta1
                row["cublas_total_beta0_p50_ms"] = cublas_total_beta0
                row["cublas_total_beta1_p50_ms"] = cublas_total_beta1
                row["cublas_total_beta0_p50_tflops_per_gpu"] = (
                    2.0 * phase_flops / cublas_total_beta0 / 1.0e9
                )
                row["cublas_total_beta1_p50_tflops_per_gpu"] = (
                    2.0 * phase_flops / cublas_total_beta1 / 1.0e9
                )
                for launch in args.launches:
                    for weight_mode in args.weight_modes:
                        path = raw_path(
                            args.results, operator, model, sequence, cp,
                            launch, weight_mode,
                        )
                        if not matches(
                            path, operator, model, sequence, cp, launch,
                            weight_mode, args.warmup, args.iterations,
                        ):
                            raise RuntimeError(f"missing/invalid result: {path}")
                        payload = json.loads(path.read_text())
                        key = f"{launch}_{weight_mode}"
                        row[f"{key}_requested_comm_ctas"] = payload[
                            "requested_comm_ctas"
                        ]
                        row[f"{key}_weight_accumulation_beta"] = payload[
                            "weight_accumulation_beta"
                        ]
                        row[f"{key}_comm_ctas"] = payload["comm_ctas"]
                        row[f"{key}_gemm_policy"] = payload["gemm_policy"]
                        traits = payload["kernel_traits"]
                        row[f"{key}_tile_m"] = traits["tile_m"]
                        row[f"{key}_tile_n"] = traits["tile_n"]
                        if weight_mode == "deferred":
                            for phase in ("data_phase", "weight_phase"):
                                for name, value in phase_fields(
                                    phase, payload, phase_flops
                                ).items():
                                    row[f"{key}_{name}"] = value
                        total = payload["total"]
                        total_p50 = float(total["p50_ms"])
                        total_tflops = 2.0 * phase_flops / total_p50 / 1.0e9
                        row[f"{key}_total_p50_ms"] = total_p50
                        row[f"{key}_total_p95_ms"] = float(total["p95_ms"])
                        row[f"{key}_total_p50_tflops_per_gpu"] = total_tflops
                        row[f"{key}_mfu_percent_of_989t"] = (
                            100.0 * total_tflops / 989.0
                        )
                        row[f"{key}_throughput_as_cublas_percent"] = (
                            100.0 * (
                                cublas_total_beta1
                                if weight_mode == "deferred"
                                else cublas_total_beta0
                            ) / total_p50
                        )
                        forward_row = forward.get((model.name, sequence, cp))
                        if forward_row is not None:
                            forward_tflops = float(
                                forward_row[f"{launch}_p50_tflops_per_gpu"]
                            )
                            row[f"{key}_forward_p50_tflops_per_gpu"] = (
                                forward_tflops
                            )
                            row[f"{key}_throughput_as_forward_percent"] = (
                                100.0 * total_tflops / forward_tflops
                            )
                rows.append(row)
    stem = args.results / f"{operator}_backward_summary"
    dump_rows(rows, stem)
    write_markdown(rows, stem, operator)


def aggregate(args: argparse.Namespace) -> None:
    for operator in args.operators:
        aggregate_operator(args, operator)


def main() -> None:
    args = parse_args()
    if any(item not in {"qkv", "oproj"} for item in args.operators):
        raise SystemExit("--operators supports qkv,oproj")
    if any(item not in {"eager", "graph"} for item in args.launches):
        raise SystemExit("--launches supports eager,graph")
    if any(item not in {"deferred", "immediate"} for item in args.weight_modes):
        raise SystemExit("--weight-modes supports deferred,immediate")
    if "formal" in args.phase:
        formal(args)
    if "cublas" in args.phase:
        cublas_formal(args)
    if "aggregate" in args.phase:
        aggregate(args)


if __name__ == "__main__":
    main()
