#!/usr/bin/env python3
"""Transformer Engine GEMM + NCCL baseline for fused projection backward.

The measured boundary exactly matches the production backward APIs:

* QKV B: Head->Sequence A2A, packed [Q,K,V], then TE DGrad GEMM.
* OProj B: TE DGrad GEMM, then Sequence->Head A2A.
* W: TE WGrad GEMM, either adjacent (ordinary) or measured separately with
  beta=1 accumulation (ZeroBubble).

Packing and unpacking are explicit Triton kernels and therefore remain inside
the CUDA-event interval.  The script uses TE's ``general_gemm`` with the same
NN/NT layouts used by ``transformer_engine.pytorch.Linear.backward``.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist
import transformer_engine
import triton
import triton.language as tl
from transformer_engine.pytorch.cpp_extensions import general_gemm


@triton.jit
def _qkv_inverse_pack_kernel(
    grad_q,
    grad_k,
    grad_v,
    packed,
    elements: tl.constexpr,
    local_tokens: tl.constexpr,
    sequence_local: tl.constexpr,
    world: tl.constexpr,
    q_local_width: tl.constexpr,
    kv_local_width: tl.constexpr,
    peer_width: tl.constexpr,
    causal: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offsets < elements
    within = offsets % peer_width
    row = (offsets // peer_width) % local_tokens
    owner = offsets // (peer_width * local_tokens)
    batch_index = row // sequence_local
    row_in_batch = row % sequence_local
    if causal:
        chunk_rows = sequence_local // 2
        chunk = tl.where(row_in_batch < chunk_rows, owner, 2 * world - owner - 1)
        global_row = (
            batch_index * sequence_local * world
            + chunk * chunk_rows
            + row_in_batch % chunk_rows
        )
    else:
        global_row = (
            batch_index * sequence_local * world
            + owner * sequence_local
            + row_in_batch
        )
    is_q = within < q_local_width
    is_k = (within >= q_local_width) & (within < q_local_width + kv_local_width)
    is_v = ~is_q & ~is_k
    q_source = global_row * q_local_width + within
    kv_column = tl.where(
        is_k,
        within - q_local_width,
        within - q_local_width - kv_local_width,
    )
    kv_source = global_row * kv_local_width + kv_column
    # Triton evaluates both sides of tl.where.  Branch-specific masks are
    # required: using one selected address for three unconditional loads lets
    # Q lanes read past the smaller K/V allocations at long sequence lengths.
    q_value = tl.load(grad_q + q_source, mask=mask & is_q, other=0.0)
    k_value = tl.load(grad_k + kv_source, mask=mask & is_k, other=0.0)
    v_value = tl.load(grad_v + kv_source, mask=mask & is_v, other=0.0)
    value = tl.where(is_q, q_value, tl.where(is_k, k_value, v_value))
    tl.store(packed + offsets, value, mask=mask)


@triton.jit
def _qkv_inverse_unpack_kernel(
    received,
    dqkv,
    elements: tl.constexpr,
    local_tokens: tl.constexpr,
    world: tl.constexpr,
    q_local_width: tl.constexpr,
    kv_local_width: tl.constexpr,
    peer_width: tl.constexpr,
    q_width: tl.constexpr,
    kv_width: tl.constexpr,
    qkv_width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offsets < elements
    row = offsets // qkv_width
    column = offsets % qkv_width
    is_q = column < q_width
    is_k = (column >= q_width) & (column < q_width + kv_width)
    segment_column = tl.where(
        is_q,
        column,
        tl.where(is_k, column - q_width, column - q_width - kv_width),
    )
    local_width = tl.where(is_q, q_local_width, kv_local_width)
    source_rank = segment_column // local_width
    source_column = segment_column % local_width
    source_base = tl.where(
        is_q,
        0,
        tl.where(is_k, q_local_width, q_local_width + kv_local_width),
    )
    source = (
        source_rank * local_tokens * peer_width
        + row * peer_width
        + source_base
        + source_column
    )
    tl.store(dqkv + offsets, tl.load(received + source, mask=mask), mask=mask)


@triton.jit
def _oproj_route_pack_kernel(
    local_da,
    packed,
    elements: tl.constexpr,
    local_tokens: tl.constexpr,
    attention_width: tl.constexpr,
    local_width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offsets < elements
    column = offsets % local_width
    row = (offsets // local_width) % local_tokens
    destination = offsets // (local_width * local_tokens)
    source = row * attention_width + destination * local_width + column
    tl.store(packed + offsets, tl.load(local_da + source, mask=mask), mask=mask)


@triton.jit
def _oproj_route_unpack_kernel(
    received,
    output,
    elements: tl.constexpr,
    local_tokens: tl.constexpr,
    sequence_local: tl.constexpr,
    world: tl.constexpr,
    local_width: tl.constexpr,
    causal: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offsets < elements
    column = offsets % local_width
    row = (offsets // local_width) % local_tokens
    source_rank = offsets // (local_width * local_tokens)
    batch_index = row // sequence_local
    row_in_batch = row % sequence_local
    if causal:
        chunk_rows = sequence_local // 2
        chunk = tl.where(
            row_in_batch < chunk_rows,
            source_rank,
            2 * world - source_rank - 1,
        )
        global_row = (
            batch_index * sequence_local * world
            + chunk * chunk_rows
            + row_in_batch % chunk_rows
        )
    else:
        global_row = (
            batch_index * sequence_local * world
            + source_rank * sequence_local
            + row_in_batch
        )
    destination = global_row * local_width + column
    tl.store(output + destination, tl.load(received + offsets, mask=mask), mask=mask)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", choices=("qkv", "oproj"), required=True)
    parser.add_argument("--model-name", default="manual")
    parser.add_argument("--global-seq", type=int, required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--q-heads", type=int, required=True)
    parser.add_argument("--kv-heads", type=int, default=0)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--weight-mode", choices=("immediate", "deferred"), required=True)
    parser.add_argument(
        "--phase-scope",
        choices=("data", "full"),
        default="full",
        help="data times only the communication+DGrad B phase during tuning",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--pack-block", type=int, choices=(128, 256, 512, 1024), default=512)
    parser.add_argument("--pack-warps", type=int, choices=(4, 8), default=4)
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--causal-load-balanced", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nccl-high-priority", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--matrix-manifest", type=Path)
    return parser.parse_args()


def percentile(ordered: list[float], q: float) -> float:
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def summarize(samples: list[float], flops: int) -> dict[str, float]:
    ordered = sorted(samples)
    p50 = percentile(ordered, 0.50)
    p95 = percentile(ordered, 0.95)
    return {
        "mean_ms": sum(samples) / len(samples),
        "p50_ms": p50,
        "p95_ms": p95,
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "p50_tflops_per_gpu": flops / p50 / 1.0e9,
    }


def timed_critical(
    fn: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
    device: torch.device,
    cuda_graph: bool,
) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    dist.barrier()

    timed_fn = fn
    graph = None
    if cuda_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        torch.cuda.synchronize(device)
        dist.barrier()
        timed_fn = graph.replay
        for _ in range(warmup):
            timed_fn()
        torch.cuda.synchronize(device)
        dist.barrier()

    samples: list[float] = []
    for _ in range(iterations):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        timed_fn()
        stop.record()
        stop.synchronize()
        rank_ms = torch.tensor(start.elapsed_time(stop), dtype=torch.float64, device=device)
        dist.all_reduce(rank_ms, op=dist.ReduceOp.MAX)
        samples.append(float(rank_ms.item()))
    return samples


def deterministic(shape: tuple[int, ...], seed: int, device: torch.device) -> torch.Tensor:
    # Do not materialize an int64 arange: the largest formal 405B case has
    # billions of elements and that temporary alone would consume tens of
    # GiB.  A per-device generator produces the same deterministic BF16 input
    # with no storage beyond the tensor being initialized.
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.empty(shape, dtype=torch.bfloat16, device=device).uniform_(
        -0.09375,
        0.09375,
        generator=generator,
    )


def launch_te_dgrad(weight: torch.Tensor, grad: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    general_gemm(
        weight,
        grad,
        out_dtype=torch.bfloat16,
        layout="NN",
        out=output,
        grad=True,
    )
    return output


def launch_te_wgrad(
    saved_input: torch.Tensor,
    grad: torch.Tensor,
    output: torch.Tensor,
    *,
    beta: int,
) -> torch.Tensor:
    general_gemm(
        saved_input,
        grad,
        out_dtype=torch.bfloat16,
        layout="NT",
        out=output,
        grad=True,
        accumulate=beta == 1,
        beta=float(beta),
    )
    return output


def global_rows(
    *,
    rank: int,
    local_tokens: int,
    batch: int,
    world: int,
    causal: bool,
    device: torch.device,
) -> torch.Tensor:
    sequence_local = local_tokens // batch
    result: list[int] = []
    for row in range(local_tokens):
        batch_index, local_row = divmod(row, sequence_local)
        if causal:
            chunk_rows = sequence_local // 2
            chunk = rank if local_row < chunk_rows else 2 * world - rank - 1
            index = (
                batch_index * sequence_local * world
                + chunk * chunk_rows
                + local_row % chunk_rows
            )
        else:
            index = (
                batch_index * sequence_local * world
                + rank * sequence_local
                + local_row
            )
        result.append(index)
    return torch.tensor(result, dtype=torch.int64, device=device)


def run_qkv(
    args: argparse.Namespace,
    rank: int,
    world: int,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, object]]:
    local_tokens = args.batch * (args.global_seq // world)
    sequence_local = args.global_seq // world
    q_local_width = args.q_heads // world * args.head_dim
    kv_local_width = args.kv_heads // world * args.head_dim
    q_width = args.q_heads * args.head_dim
    kv_width = args.kv_heads * args.head_dim
    width = q_width + 2 * kv_width
    peer_width = q_local_width + 2 * kv_local_width
    grad_q = deterministic((args.batch * args.global_seq, q_local_width), 101 + rank, device)
    grad_k = deterministic((args.batch * args.global_seq, kv_local_width), 201 + rank, device)
    grad_v = deterministic((args.batch * args.global_seq, kv_local_width), 301 + rank, device)
    saved_input = deterministic((local_tokens, args.hidden), 401 + rank, device)
    weight = deterministic((width, args.hidden), 501, device)
    send = torch.empty((world, local_tokens, peer_width), dtype=torch.bfloat16, device=device)
    receive = torch.empty_like(send)
    dqkv = torch.empty((local_tokens, width), dtype=torch.bfloat16, device=device)
    grad_input = torch.empty((local_tokens, args.hidden), dtype=torch.bfloat16, device=device)
    grad_weight = torch.zeros_like(weight)
    pack_elements = send.numel()
    unpack_elements = dqkv.numel()

    def route() -> torch.Tensor:
        _qkv_inverse_pack_kernel[(triton.cdiv(pack_elements, args.pack_block),)](
            grad_q,
            grad_k,
            grad_v,
            send,
            elements=pack_elements,
            local_tokens=local_tokens,
            sequence_local=sequence_local,
            world=world,
            q_local_width=q_local_width,
            kv_local_width=kv_local_width,
            peer_width=peer_width,
            causal=args.causal_load_balanced,
            BLOCK=args.pack_block,
            num_warps=args.pack_warps,
        )
        dist.all_to_all_single(receive, send)
        _qkv_inverse_unpack_kernel[(triton.cdiv(unpack_elements, args.pack_block),)](
            receive,
            dqkv,
            elements=unpack_elements,
            local_tokens=local_tokens,
            world=world,
            q_local_width=q_local_width,
            kv_local_width=kv_local_width,
            peer_width=peer_width,
            q_width=q_width,
            kv_width=kv_width,
            qkv_width=width,
            BLOCK=args.pack_block,
            num_warps=args.pack_warps,
        )
        return dqkv

    def data_phase() -> torch.Tensor:
        route()
        return launch_te_dgrad(weight, dqkv, grad_input)

    beta = 0 if args.weight_mode == "immediate" else 1

    def weight_phase() -> torch.Tensor:
        return launch_te_wgrad(saved_input, dqkv, grad_weight, beta=beta)

    def immediate() -> torch.Tensor:
        data_phase()
        return weight_phase()

    route()
    torch.cuda.synchronize(device)
    correctness: dict[str, object] = {}
    if args.check:
        dqkv_first = dqkv.clone()
        route()
        torch.cuda.synchronize(device)
        deterministic_route_mismatches = torch.count_nonzero(
            dqkv != dqkv_first
        ).to(torch.int64)
        # Recreate peer inputs from their documented seeds instead of adding
        # a second family of collectives to a matrix process group.  This
        # keeps correctness independent of the measured A2A and avoids
        # retaining ProcessGroup work/tensor lifetimes across later cases.
        peer_q = [
            deterministic(grad_q.shape, 101 + peer, device)
            for peer in range(world)
        ]
        peer_k = [
            deterministic(grad_k.shape, 201 + peer, device)
            for peer in range(world)
        ]
        peer_v = [
            deterministic(grad_v.shape, 301 + peer, device)
            for peer in range(world)
        ]
        rows = global_rows(
            rank=rank,
            local_tokens=local_tokens,
            batch=args.batch,
            world=world,
            causal=args.causal_load_balanced,
            device=device,
        )
        expected = torch.cat(
            [tensor.index_select(0, rows) for tensor in peer_q]
            + [tensor.index_select(0, rows) for tensor in peer_k]
            + [tensor.index_select(0, rows) for tensor in peer_v],
            dim=1,
        )
        route_errors = torch.count_nonzero(dqkv != expected).to(torch.int64)
        launch_te_dgrad(weight, dqkv, grad_input)
        grad_input_first = grad_input.clone()
        launch_te_dgrad(weight, dqkv, grad_input)
        deterministic_dgrad_mismatches = torch.count_nonzero(
            grad_input != grad_input_first
        ).to(torch.int64)
        launch_te_wgrad(saved_input, dqkv, grad_weight, beta=0)
        grad_weight_first = grad_weight.clone()
        launch_te_wgrad(saved_input, dqkv, grad_weight, beta=0)
        deterministic_wgrad_mismatches = torch.count_nonzero(
            grad_weight != grad_weight_first
        ).to(torch.int64)
        torch.cuda.synchronize(device)
        dx_error = (grad_input.float() - (dqkv @ weight).float()).abs().max()
        reference_dw = dqkv.T @ saved_input
        dw_error = (grad_weight.float() - reference_dw.float()).abs().max()

        # ZeroBubble accumulates a deferred WGrad directly into persistent
        # main_grad.  Check the actual beta=1 contract from a non-zero value,
        # twice, rather than inferring it from the launch arguments.
        initial_main_grad = deterministic(weight.shape, 901 + rank, device)
        grad_weight.copy_(initial_main_grad)
        launch_te_wgrad(saved_input, dqkv, grad_weight, beta=1)
        torch.cuda.synchronize(device)
        beta1_once_error = (
            grad_weight.float()
            - (initial_main_grad.float() + reference_dw.float())
        ).abs().max()
        launch_te_wgrad(saved_input, dqkv, grad_weight, beta=1)
        torch.cuda.synchronize(device)
        beta1_twice_error = (
            grad_weight.float()
            - (initial_main_grad.float() + 2.0 * reference_dw.float())
        ).abs().max()
        vector = torch.stack(
            (
                route_errors.float(),
                dx_error,
                dw_error,
                beta1_once_error,
                beta1_twice_error,
                deterministic_route_mismatches.float(),
                deterministic_dgrad_mismatches.float(),
                deterministic_wgrad_mismatches.float(),
            )
        )
        dist.all_reduce(vector, op=dist.ReduceOp.MAX)
        correctness = {
            "route_mismatches": int(vector[0].item()),
            "dgrad_max_abs": float(vector[1].item()),
            "wgrad_max_abs": float(vector[2].item()),
            "main_grad_beta1_once_max_abs": float(vector[3].item()),
            "main_grad_beta1_twice_max_abs": float(vector[4].item()),
            "deterministic_route_mismatches": int(vector[5].item()),
            "deterministic_dgrad_mismatches": int(vector[6].item()),
            "deterministic_wgrad_mismatches": int(vector[7].item()),
        }
        if any(
            correctness[name]
            for name in (
                "route_mismatches",
                "deterministic_route_mismatches",
                "deterministic_dgrad_mismatches",
                "deterministic_wgrad_mismatches",
            )
        ) or max(
            float(value)
            for name, value in correctness.items()
            if "mismatches" not in name
        ) > 0.25:
            raise RuntimeError(f"QKV TE baseline correctness failed: {correctness}")
        grad_weight.zero_()

    phase_flops = 2 * local_tokens * args.hidden * width
    if args.phase_scope == "data":
        data_samples = timed_critical(
            data_phase,
            warmup=args.warmup,
            iterations=args.iters,
            device=device,
            cuda_graph=args.cuda_graph,
        )
        metrics = {
            "data": summarize(data_samples, phase_flops),
            "data_samples_ms": data_samples,
        }
    elif args.weight_mode == "immediate":
        samples = timed_critical(
            immediate,
            warmup=args.warmup,
            iterations=args.iters,
            device=device,
            cuda_graph=args.cuda_graph,
        )
        metrics = {
            "total": summarize(samples, 2 * phase_flops),
            "samples_ms": samples,
        }
    else:
        data_samples = timed_critical(
            data_phase,
            warmup=args.warmup,
            iterations=args.iters,
            device=device,
            cuda_graph=args.cuda_graph,
        )
        weight_samples = timed_critical(
            weight_phase,
            warmup=args.warmup,
            iterations=args.iters,
            device=device,
            cuda_graph=args.cuda_graph,
        )
        total_samples = [a + b for a, b in zip(data_samples, weight_samples)]
        metrics = {
            "data": summarize(data_samples, phase_flops),
            "weight": summarize(weight_samples, phase_flops),
            "total": summarize(total_samples, 2 * phase_flops),
            "data_samples_ms": data_samples,
            "weight_samples_ms": weight_samples,
            "samples_ms": total_samples,
        }
    shape = {
        "b_mnk": [local_tokens, args.hidden, width],
        "w_mnk": [width, args.hidden, local_tokens],
    }
    return metrics, {"correctness": correctness, "shape": shape}


def run_oproj(
    args: argparse.Namespace,
    rank: int,
    world: int,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, object]]:
    local_tokens = args.batch * (args.global_seq // world)
    sequence_local = args.global_seq // world
    width = args.q_heads * args.head_dim
    local_width = width // world
    grad_output = deterministic((local_tokens, args.hidden), 601 + rank, device)
    saved_attention = deterministic((local_tokens, width), 701 + rank, device)
    weight = deterministic((args.hidden, width), 801, device)
    local_da = torch.empty_like(saved_attention)
    send = torch.empty((world, local_tokens, local_width), dtype=torch.bfloat16, device=device)
    receive = torch.empty_like(send)
    peer_da = torch.empty(
        (args.batch * args.global_seq, local_width),
        dtype=torch.bfloat16,
        device=device,
    )
    grad_weight = torch.zeros_like(weight)
    route_elements = send.numel()

    def route() -> torch.Tensor:
        _oproj_route_pack_kernel[(triton.cdiv(route_elements, args.pack_block),)](
            local_da,
            send,
            elements=route_elements,
            local_tokens=local_tokens,
            attention_width=width,
            local_width=local_width,
            BLOCK=args.pack_block,
            num_warps=args.pack_warps,
        )
        dist.all_to_all_single(receive, send)
        _oproj_route_unpack_kernel[(triton.cdiv(route_elements, args.pack_block),)](
            receive,
            peer_da,
            elements=route_elements,
            local_tokens=local_tokens,
            sequence_local=sequence_local,
            world=world,
            local_width=local_width,
            causal=args.causal_load_balanced,
            BLOCK=args.pack_block,
            num_warps=args.pack_warps,
        )
        return peer_da

    def data_phase() -> torch.Tensor:
        launch_te_dgrad(weight, grad_output, local_da)
        return route()

    beta = 0 if args.weight_mode == "immediate" else 1

    def weight_phase() -> torch.Tensor:
        return launch_te_wgrad(saved_attention, grad_output, grad_weight, beta=beta)

    def immediate() -> torch.Tensor:
        data_phase()
        return weight_phase()

    data_phase()
    torch.cuda.synchronize(device)
    correctness: dict[str, object] = {}
    if args.check:
        local_da_first = local_da.clone()
        peer_da_first = peer_da.clone()
        data_phase()
        torch.cuda.synchronize(device)
        deterministic_dgrad_mismatches = torch.count_nonzero(
            local_da != local_da_first
        ).to(torch.int64)
        deterministic_route_mismatches = torch.count_nonzero(
            peer_da != peer_da_first
        ).to(torch.int64)
        reference_da = grad_output @ weight
        local_error = (local_da.float() - reference_da.float()).abs().max()
        peer_local_da = [
            deterministic(
                (local_tokens, args.hidden),
                601 + source,
                device,
            )
            @ weight
            for source in range(world)
        ]
        expected = torch.empty_like(peer_da)
        for source in range(world):
            rows = global_rows(
                rank=source,
                local_tokens=local_tokens,
                batch=args.batch,
                world=world,
                causal=args.causal_load_balanced,
                device=device,
            )
            expected.index_copy_(
                0,
                rows,
                peer_local_da[source][
                    :, rank * local_width : (rank + 1) * local_width
                ],
            )
        route_error = (peer_da.float() - expected.float()).abs().max()
        launch_te_wgrad(saved_attention, grad_output, grad_weight, beta=0)
        grad_weight_first = grad_weight.clone()
        launch_te_wgrad(saved_attention, grad_output, grad_weight, beta=0)
        deterministic_wgrad_mismatches = torch.count_nonzero(
            grad_weight != grad_weight_first
        ).to(torch.int64)
        torch.cuda.synchronize(device)
        reference_dw = grad_output.T @ saved_attention
        dw_error = (grad_weight.float() - reference_dw.float()).abs().max()

        initial_main_grad = deterministic(weight.shape, 1001 + rank, device)
        grad_weight.copy_(initial_main_grad)
        launch_te_wgrad(saved_attention, grad_output, grad_weight, beta=1)
        torch.cuda.synchronize(device)
        beta1_once_error = (
            grad_weight.float()
            - (initial_main_grad.float() + reference_dw.float())
        ).abs().max()
        launch_te_wgrad(saved_attention, grad_output, grad_weight, beta=1)
        torch.cuda.synchronize(device)
        beta1_twice_error = (
            grad_weight.float()
            - (initial_main_grad.float() + 2.0 * reference_dw.float())
        ).abs().max()
        vector = torch.stack(
            (
                local_error,
                route_error,
                dw_error,
                beta1_once_error,
                beta1_twice_error,
                deterministic_dgrad_mismatches.float(),
                deterministic_route_mismatches.float(),
                deterministic_wgrad_mismatches.float(),
            )
        )
        dist.all_reduce(vector, op=dist.ReduceOp.MAX)
        correctness = {
            "local_dgrad_max_abs": float(vector[0].item()),
            "route_max_abs": float(vector[1].item()),
            "wgrad_max_abs": float(vector[2].item()),
            "main_grad_beta1_once_max_abs": float(vector[3].item()),
            "main_grad_beta1_twice_max_abs": float(vector[4].item()),
            "deterministic_dgrad_mismatches": int(vector[5].item()),
            "deterministic_route_mismatches": int(vector[6].item()),
            "deterministic_wgrad_mismatches": int(vector[7].item()),
        }
        if any(
            int(value) != 0
            for name, value in correctness.items()
            if "mismatches" in name
        ) or max(
            float(value)
            for name, value in correctness.items()
            if "mismatches" not in name
        ) > 0.25:
            raise RuntimeError(f"OProj TE baseline correctness failed: {correctness}")
        grad_weight.zero_()

    phase_flops = 2 * local_tokens * args.hidden * width
    if args.phase_scope == "data":
        data_samples = timed_critical(
            data_phase,
            warmup=args.warmup,
            iterations=args.iters,
            device=device,
            cuda_graph=args.cuda_graph,
        )
        metrics = {
            "data": summarize(data_samples, phase_flops),
            "data_samples_ms": data_samples,
        }
    elif args.weight_mode == "immediate":
        samples = timed_critical(
            immediate,
            warmup=args.warmup,
            iterations=args.iters,
            device=device,
            cuda_graph=args.cuda_graph,
        )
        metrics = {
            "total": summarize(samples, 2 * phase_flops),
            "samples_ms": samples,
        }
    else:
        data_samples = timed_critical(
            data_phase,
            warmup=args.warmup,
            iterations=args.iters,
            device=device,
            cuda_graph=args.cuda_graph,
        )
        weight_samples = timed_critical(
            weight_phase,
            warmup=args.warmup,
            iterations=args.iters,
            device=device,
            cuda_graph=args.cuda_graph,
        )
        total_samples = [a + b for a, b in zip(data_samples, weight_samples)]
        metrics = {
            "data": summarize(data_samples, phase_flops),
            "weight": summarize(weight_samples, phase_flops),
            "total": summarize(total_samples, 2 * phase_flops),
            "data_samples_ms": data_samples,
            "weight_samples_ms": weight_samples,
            "samples_ms": total_samples,
        }
    shape = {
        "b_mnk": [local_tokens, width, args.hidden],
        "w_mnk": [args.hidden, width, local_tokens],
    }
    return metrics, {"correctness": correctness, "shape": shape}


def validate(args: argparse.Namespace, world: int) -> None:
    if args.global_seq % world:
        raise ValueError("global sequence must be divisible by CP")
    if args.q_heads % world:
        raise ValueError("Q heads must be divisible by CP")
    if args.operator == "qkv":
        if args.kv_heads <= 0 or args.kv_heads % world:
            raise ValueError("KV heads must be positive and divisible by CP")
        if args.q_heads % args.kv_heads:
            raise ValueError("Q heads must be divisible by KV heads")
    sequence_local = args.global_seq // world
    if args.causal_load_balanced and sequence_local % 2:
        raise ValueError("causal load-balanced layout requires even local sequence")
    if min(args.global_seq, args.hidden, args.batch, args.head_dim, args.warmup, args.iters) <= 0:
        raise ValueError("shape and sampling arguments must be positive")
    if args.phase_scope == "data" and args.weight_mode != "deferred":
        raise ValueError("data-only tuning requires deferred mode")


def initialize_process_group(
    args: argparse.Namespace,
    device: torch.device,
    *,
    init_method: str | None = None,
) -> None:
    options = dist.ProcessGroupNCCL.Options()
    options.is_high_priority_stream = args.nccl_high_priority
    init_arguments: dict[str, object] = {}
    if init_method is not None:
        init_arguments.update(
            {
                "init_method": init_method,
                "rank": int(os.environ["RANK"]),
                "world_size": int(os.environ["WORLD_SIZE"]),
            }
        )
    dist.init_process_group(
        "nccl",
        device_id=device,
        pg_options=options,
        **init_arguments,
    )


def run_initialized(args: argparse.Namespace, device: torch.device) -> None:
    rank, world = dist.get_rank(), dist.get_world_size()
    validate(args, world)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    runner = run_qkv if args.operator == "qkv" else run_oproj
    metrics, extra = runner(args, rank, world, device)
    properties = torch.cuda.get_device_properties(device)
    local_device = {
        "rank": rank,
        "name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "sm_count": properties.multi_processor_count,
    }
    devices: list[dict[str, object] | None] = [None] * world
    dist.all_gather_object(devices, local_device)
    result = {
        "schema": "v10_backward_te_nccl_v2",
        "numerical_contract": "complete_route_then_single_full_gemm",
        "determinism_contract": "repeat_route_dgrad_wgrad_exact_at_s1k",
        "model_name": args.model_name,
        "operator": args.operator,
        "qkv_pack_kernel": (
            "branch_masked_v1" if args.operator == "qkv" else "not_applicable"
        ),
        "weight_mode": args.weight_mode,
        "phase_scope": args.phase_scope,
        "weight_accumulation_beta": 0 if args.weight_mode == "immediate" else 1,
        "launch": "graph" if args.cuda_graph else "eager",
        "global_seq": args.global_seq,
        "local_tokens": args.batch * (args.global_seq // world),
        "hidden": args.hidden,
        "batch": args.batch,
        "q_heads": args.q_heads,
        "kv_heads": args.kv_heads,
        "head_dim": args.head_dim,
        "world_size": world,
        "warmup": args.warmup,
        "iterations": args.iters,
        "timing": "per-sample max-rank CUDA-event critical path",
        "boundary": (
            "HeadToSequence A2A -> TE DGrad; TE WGrad"
            if args.operator == "qkv"
            else "TE DGrad -> SequenceToHead A2A; TE WGrad"
        ),
        "te_gemm": {
            "dgrad": "general_gemm layout=NN grad=True",
            "wgrad": "general_gemm layout=NT grad=True",
        },
        "route": {
            "collective": "torch.distributed.all_to_all_single / NCCL",
            "pack": "Triton",
            "pack_block": args.pack_block,
            "pack_warps": args.pack_warps,
            "causal_load_balanced": args.causal_load_balanced,
        },
        "nccl_high_priority": args.nccl_high_priority,
        "environment": {
            name: os.environ.get(name)
            for name in (
                "NCCL_MIN_P2P_NCHANNELS",
                "NCCL_MAX_P2P_NCHANNELS",
                "NCCL_P2P_NVL_CHUNKSIZE",
                "NCCL_P2P_LL_THRESHOLD",
                "NCCL_GRAPH_REGISTER",
                "NCCL_LOCAL_REGISTER",
                "CUBLAS_WORKSPACE_CONFIG",
            )
        },
        "software": {
            "torch": torch.__version__,
            "transformer_engine": transformer_engine.__version__,
            "transformer_engine_path": str(Path(transformer_engine.__file__).resolve()),
            "torch_cuda": torch.version.cuda,
            "nccl": ".".join(str(value) for value in torch.cuda.nccl.version()),
        },
        "devices": devices,
        **extra,
        "results": metrics,
    }
    if rank == 0:
        total = metrics["total"] if "total" in metrics else metrics["data"]
        print(
            f"TE+NCCL {args.operator} {args.weight_mode} "
            f"{'Graph' if args.cuda_graph else 'Eager'} "
            f"p50={total['p50_ms']:.6f} ms "
            f"TFLOPS/GPU={total['p50_tflops_per_gpu']:.1f}",
            flush=True,
        )
        print(f"correctness={extra['correctness']}", flush=True)
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.json_out.with_suffix(args.json_out.suffix + ".tmp")
            temporary.write_text(json.dumps(result, indent=2) + "\n")
            temporary.replace(args.json_out)
    torch.cuda.synchronize(device)
    del result, metrics, extra
    gc.collect()
    torch.cuda.empty_cache()


def run_one_process_group(
    args: argparse.Namespace,
    device: torch.device,
    *,
    init_method: str | None = None,
) -> None:
    initialize_process_group(args, device, init_method=init_method)
    run_initialized(args, device)
    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if args.matrix_manifest is None:
        run_one_process_group(args, device)
        return

    entries = json.loads(args.matrix_manifest.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError("matrix manifest must be a non-empty JSON list")
    for high_priority in (False, True):
        priority_entries = [
            entry
            for entry in entries
            if bool(entry["nccl_high_priority"]) == high_priority
        ]
        if not priority_entries:
            continue
        args.nccl_high_priority = high_priority
        initialize_process_group(
            args,
            device,
            init_method=f"file://{priority_entries[0]['rendezvous_file']}",
        )
        for entry in priority_entries:
            for name, value in entry.get("arguments", {}).items():
                setattr(args, name, value)
            for name, value in entry["environment"].items():
                if os.environ.get(name) != str(value):
                    raise RuntimeError(
                        f"matrix entry attempted to change process-static {name}: "
                        f"{os.environ.get(name)!r} -> {value!r}"
                    )
            args.cuda_graph = bool(entry["cuda_graph"])
            args.json_out = Path(entry["json_out"])
            run_initialized(args, device)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
