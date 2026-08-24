#!/usr/bin/env python3
"""Flux baselines for the two Ulysses communication directions.

The A2A->GEMM case is shape/layout equivalent to the dense fuse kernel.  Flux
does not provide batched PV->A2A, so the reverse case is deliberately reported
as a dense projection analog with the same aggregate GEMM FLOPs and A2A bytes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Callable

import flux
import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "a2a_gemm",
            "gemm_a2a_dense_analog",
            "fp8_a2a_gemm_capability",
            "fp8_qkv_gemm_a2a_capability",
        ),
        required=True,
    )
    parser.add_argument("--gemm-m", dest="m", type=int, required=True)
    parser.add_argument("--gemm-n", dest="n", type=int, required=True)
    parser.add_argument("--gemm-k", dest="k", type=int, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-comm-sm", type=int, required=True)
    parser.add_argument("--sm-margin", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def percentile(ordered: list[float], q: float) -> float:
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    weight = position - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def summarize(samples: list[float], flops: int, world: int) -> dict[str, float]:
    ordered = sorted(samples)
    mean = sum(samples) / len(samples)
    tflops = flops / mean / 1.0e9
    return {
        "mean_ms": mean,
        "p50_ms": percentile(ordered, 0.50),
        "p95_ms": percentile(ordered, 0.95),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "tflops_per_gpu": tflops,
        "aggregate_tflops": tflops * world,
    }


@torch.no_grad()
def timed_critical(
    fn: Callable[[], object], warmup: int, iters: int, device: torch.device
) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    dist.barrier()

    samples: list[float] = []
    for _ in range(iters):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        critical = torch.tensor(start.elapsed_time(stop), dtype=torch.float64, device=device)
        dist.all_reduce(critical, op=dist.ReduceOp.MAX)
        samples.append(float(critical.item()))
    return samples


@torch.no_grad()
def error_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    actual = actual.reshape(expected.shape)
    diff = actual.float() - expected.float()
    max_abs = diff.abs().max()
    max_rel = (diff.abs() / expected.float().abs().clamp_min(1.0e-6)).max()
    diff_l2 = diff.square().sum(dtype=torch.float64)
    ref_l2 = expected.float().square().sum(dtype=torch.float64)
    exact = torch.tensor(int(torch.equal(actual, expected)), dtype=torch.int32, device=actual.device)
    dist.all_reduce(max_abs, op=dist.ReduceOp.MAX)
    dist.all_reduce(max_rel, op=dist.ReduceOp.MAX)
    dist.all_reduce(diff_l2, op=dist.ReduceOp.SUM)
    dist.all_reduce(ref_l2, op=dist.ReduceOp.SUM)
    dist.all_reduce(exact, op=dist.ReduceOp.MIN)
    bitwise = bool(exact.item())
    max_abs_value = float(max_abs.item())
    rel_l2_value = float(torch.sqrt(diff_l2 / ref_l2.clamp_min(1.0e-30)).item())
    return {
        "passed": bitwise or (max_abs_value <= 2.0e-2 and rel_l2_value <= 1.0e-3),
        "bitwise": bitwise,
        "max_abs": max_abs_value,
        "max_rel": float(max_rel.item()),
        "rel_l2": rel_l2_value,
    }


def make_gemm_only() -> object:
    return flux.GemmOnly(
        input_dtype=torch.bfloat16,
        weight_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        transpose_weight=False,
        use_fp8_gemm=False,
    )


@torch.no_grad()
def fp8_capability(
    args: argparse.Namespace, world: int
) -> tuple[dict[str, object], dict[str, list[float]]]:
    """Probe Flux's exact FP8 Ulysses fused operator, not a substitute op."""
    if args.m % args.batch:
        raise ValueError("M must be divisible by batch")
    seq_local = args.m // args.batch
    seq_global = seq_local * world
    dtype = torch.float8_e4m3fn

    try:
        if args.mode == "fp8_a2a_gemm_capability":
            if args.k % args.head_dim or args.k // args.head_dim % world:
                raise ValueError("K/head_dim must be integral and head count divisible by CP")
            heads = args.k // args.head_dim
            flux.AllToAllTransposeGemm(
                dist.group.WORLD,
                1,
                world,
                args.batch,
                heads,
                seq_global,
                args.head_dim,
                dtype,
                output_dtype=torch.bfloat16,
                a2a_only=False,
            )
            operator = "AllToAllTransposeGemm"
            direction = "A2A_to_projection"
        else:
            flux.GemmAllToAllTranspose(
                dist.group.WORLD,
                1,
                world,
                args.batch,
                seq_global,
                args.k,
                args.head_dim,
                args.n,
                dtype,
                output_dtype=torch.bfloat16,
                transpose_weight=False,
                gqa=8,
                comm_op=flux.PreAttnAllToAllCommOp.QKVPackA2A,
            )
            operator = "GemmAllToAllTranspose"
            direction = "projection_to_QKV_GQA_A2A"
    except (RuntimeError, ValueError) as error:
        operator = (
            "AllToAllTransposeGemm"
            if args.mode == "fp8_a2a_gemm_capability"
            else "GemmAllToAllTranspose"
        )
        direction = (
            "A2A_to_projection"
            if args.mode == "fp8_a2a_gemm_capability"
            else "projection_to_QKV_GQA_A2A"
        )
        reasons: list[object] = [None] * world
        dist.all_gather_object(reasons, str(error))
        return (
            {
                "comparison_scope": "exact_Flux_fused_operator_FP8_capability",
                "direction": direction,
                "operator": operator,
                "supported": False,
                "reason": reasons[0],
                "all_ranks_same_reason": len(set(reasons)) == 1,
            },
            {},
        )

    return (
        {
            "comparison_scope": "exact_Flux_fused_operator_FP8_capability",
            "direction": direction,
            "operator": operator,
            "supported": True,
        },
        {},
    )


@torch.no_grad()
def a2a_gemm(
    args: argparse.Namespace, rank: int, world: int, device: torch.device
) -> tuple[dict[str, object], dict[str, list[float]]]:
    if args.m % args.batch:
        raise ValueError("M must be divisible by batch")
    if args.k % args.head_dim or args.k // args.head_dim % world:
        raise ValueError("K/head_dim must be integral and head count divisible by CP")
    seq_local = args.m // args.batch
    seq_global = seq_local * world
    heads = args.k // args.head_dim
    local_heads = heads // world

    generator = torch.Generator(device=device).manual_seed(7001 + rank)
    peer_input = torch.empty(
        (args.batch, local_heads, seq_global, args.head_dim),
        dtype=torch.bfloat16,
        device=device,
    ).uniform_(-0.1, 0.1, generator=generator)
    weight = torch.empty((args.n, args.k), dtype=torch.bfloat16, device=device).uniform_(
        -0.1, 0.1, generator=generator
    )
    dist.broadcast(weight, src=0)

    recv = torch.empty(
        (world, args.batch, seq_local, local_heads, args.head_dim),
        dtype=torch.bfloat16,
        device=device,
    )

    def nccl_route() -> torch.Tensor:
        send = (
            peer_input.view(args.batch, local_heads, world, seq_local, args.head_dim)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )
        dist.all_to_all_single(recv, send)
        return recv.permute(1, 2, 0, 3, 4).contiguous().view(args.m, args.k)

    staging = nccl_route()
    reference = staging @ weight.t()
    flux_output = torch.empty((args.m, args.n), dtype=torch.bfloat16, device=device)
    op = flux.AllToAllTransposeGemm(
        dist.group.WORLD,
        1,
        world,
        args.batch,
        heads,
        seq_global,
        args.head_dim,
        torch.bfloat16,
        output_dtype=torch.bfloat16,
        a2a_only=False,
    )
    option = flux.AllToAllOption()

    def fused() -> torch.Tensor:
        return op.forward(
            peer_input,
            weight,
            output=flux_output,
            all_to_all_option=option,
            num_comm_sm=args.num_comm_sm,
            sm_margin=args.sm_margin,
        )

    actual = fused().reshape(reference.shape)
    correctness = error_stats(actual, reference)
    gemm_output = torch.empty_like(reference)
    gemm_only = make_gemm_only()

    def flux_gemm() -> torch.Tensor:
        return gemm_only.forward(staging, weight, output_buf=gemm_output)

    samples = {
        "flux_a2a_gemm": timed_critical(fused, args.warmup, args.iters, device),
        "flux_gemm_only": timed_critical(flux_gemm, args.warmup, args.iters, device),
    }
    metadata: dict[str, object] = {
        "comparison_scope": "shape_and_layout_equivalent_dense_A2A_to_GEMM",
        "route": "[B,H_local,S_global,D] -> [B,S_local,H_global,D]",
        "correctness": correctness,
    }
    return metadata, samples


@torch.no_grad()
def gemm_a2a_dense_analog(
    args: argparse.Namespace, rank: int, world: int, device: torch.device
) -> tuple[dict[str, object], dict[str, list[float]]]:
    if args.m % args.batch:
        raise ValueError("M must be divisible by batch")
    if args.n % args.head_dim or args.n // args.head_dim % world:
        raise ValueError("N/head_dim must be integral and head count divisible by CP")
    seq_local = args.m // args.batch
    seq_global = seq_local * world
    heads = args.n // args.head_dim
    local_heads = heads // world

    generator = torch.Generator(device=device).manual_seed(8001 + rank)
    gemm_input = torch.empty(
        (args.batch, seq_local, args.k), dtype=torch.bfloat16, device=device
    ).uniform_(-0.1, 0.1, generator=generator)
    weight = torch.empty((args.n, args.k), dtype=torch.bfloat16, device=device).uniform_(
        -0.1, 0.1, generator=generator
    )
    dist.broadcast(weight, src=0)

    local_d = gemm_input @ weight.t()
    recv = torch.empty(
        (world, args.batch, seq_local, local_heads, args.head_dim),
        dtype=torch.bfloat16,
        device=device,
    )

    def nccl_route(tensor: torch.Tensor = local_d) -> torch.Tensor:
        send = (
            tensor.view(args.batch, seq_local, world, local_heads, args.head_dim)
            .permute(2, 0, 1, 3, 4)
            .contiguous()
        )
        dist.all_to_all_single(recv, send)
        return (
            recv.permute(1, 3, 0, 2, 4)
            .contiguous()
            .view(args.batch, local_heads, seq_global, args.head_dim)
        )

    reference = nccl_route()
    flux_output = torch.empty_like(reference)
    op = flux.GemmAllToAllTranspose(
        dist.group.WORLD,
        1,
        world,
        args.batch,
        seq_global,
        args.k,
        args.head_dim,
        args.n,
        torch.bfloat16,
        output_dtype=torch.bfloat16,
        transpose_weight=False,
        gqa=0,
        comm_op=flux.PreAttnAllToAllCommOp.A2ATranspose,
    )

    def fused() -> torch.Tensor:
        return op.forward(
            gemm_input,
            weight,
            outputs=[flux_output],
            num_comm_sm=args.num_comm_sm,
            sm_margin=args.sm_margin,
        )[0]

    actual = fused()
    correctness = error_stats(actual, reference)
    gemm_output = torch.empty((args.m, args.n), dtype=torch.bfloat16, device=device)
    gemm_only = make_gemm_only()

    def flux_gemm() -> torch.Tensor:
        return gemm_only.forward(
            gemm_input.reshape(args.m, args.k), weight, output_buf=gemm_output
        )

    samples = {
        "flux_gemm_a2a_dense_analog": timed_critical(
            fused, args.warmup, args.iters, device
        ),
        "flux_gemm_only": timed_critical(flux_gemm, args.warmup, args.iters, device),
    }
    metadata: dict[str, object] = {
        "comparison_scope": (
            "same_aggregate_FLOPs_and_output_A2A_bytes_but_dense_projection;"
            "not_equivalent_to_strided_batched_PV"
        ),
        "route": "[B,S_local,H_global,D] -> [B,H_local,S_global,D]",
        "correctness": correctness,
    }
    return metadata, samples


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    flux.init_flux_shm(dist.group.WORLD)
    torch.cuda.synchronize(device)

    if args.mode == "a2a_gemm":
        metadata, samples = a2a_gemm(args, rank, world, device)
    elif args.mode == "gemm_a2a_dense_analog":
        metadata, samples = gemm_a2a_dense_analog(args, rank, world, device)
    else:
        metadata, samples = fp8_capability(args, world)
    flops = 2 * args.m * args.n * args.k
    result = {
        "mode": args.mode,
        "shape": {"m": args.m, "n": args.n, "k": args.k, "l": 1},
        "world_size": world,
        "dtype": (
            "float8_e4m3fn_to_bfloat16"
            if args.mode.startswith("fp8_")
            else "bfloat16"
        ),
        "num_comm_sm": args.num_comm_sm,
        "sm_margin": args.sm_margin,
        "timing": (
            "not_applicable_capability_probe"
            if args.mode.startswith("fp8_")
            else "maximum rank CUDA-event critical path"
        ),
        "flux_version": flux.__version__,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(device),
        **metadata,
        "results": {name: summarize(values, flops, world) for name, values in samples.items()},
    }
    if rank == 0:
        print(json.dumps(result, indent=2), flush=True)
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
