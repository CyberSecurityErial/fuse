#!/usr/bin/env python3
"""Adapted Transformer Engine Userbuffers baseline for projection backward.

This is an operator-boundary baseline, not a framework approximation:

* Formal QKV B packs planar dQ/dK/dV, routes the complete matrix with
  Userbuffers, then executes one full-K DGrad GEMM.
* Formal OProj B executes one full-width DGrad GEMM, routes its head slabs with
  Userbuffers, then restores the final sequence layout.
* A peer-by-peer overlap mode remains available only for diagnosis because its
  repeated BF16 beta=1 accumulation is not the formal numerical contract.
* WGrad uses Transformer Engine's BF16 GEMM and is either adjacent to B or
  measured separately with beta=1 for the ZeroBubble contract.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist
import transformer_engine  # noqa: F401: load TE global symbols before tex
import transformer_engine_torch as tex
import triton
import triton.language as tl


THIS_DIR = Path(__file__).resolve().parent
QKV_FORWARD_DIR = THIS_DIR.parent / "QKVproj+a2a"
sys.path.insert(0, str(QKV_FORWARD_DIR))
from te_nccl_baseline import CublasLtRunner  # noqa: E402
from backward_te_nccl_baseline import (  # noqa: E402
    _oproj_route_pack_kernel,
    _qkv_inverse_pack_kernel,
    deterministic,
    global_rows,
    launch_te_dgrad,
    launch_te_wgrad,
)


@triton.jit
def _qkv_ub_unpack_kernel(
    send,
    recv,
    dqkv,
    elements: tl.constexpr,
    local_tokens: tl.constexpr,
    world: tl.constexpr,
    rank: tl.constexpr,
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
    is_local = source_rank == rank
    # Triton evaluates both operands of tl.where.  Mask the two loads
    # separately so a local element reads only ``send`` and a remote element
    # reads only ``recv``.  The previous form doubled unpack traffic even
    # though it selected the correct value afterwards.
    local_value = tl.load(send + source, mask=mask & is_local, other=0.0)
    remote_value = tl.load(recv + source, mask=mask & ~is_local, other=0.0)
    value = tl.where(is_local, local_value, remote_value)
    tl.store(dqkv + offsets, value, mask=mask)


@triton.jit
def _oproj_ub_unpack_kernel(
    send,
    recv,
    output,
    elements: tl.constexpr,
    local_tokens: tl.constexpr,
    sequence_local: tl.constexpr,
    world: tl.constexpr,
    rank: tl.constexpr,
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
    source = source_rank * local_tokens * local_width + row * local_width + column
    is_local = source_rank == rank
    local_value = tl.load(send + source, mask=mask & is_local, other=0.0)
    remote_value = tl.load(recv + source, mask=mask & ~is_local, other=0.0)
    value = tl.where(is_local, local_value, remote_value)
    tl.store(output + global_row * local_width + column, value, mask=mask)


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
        choices=("comm", "data", "full"),
        default="full",
    )
    parser.add_argument(
        "--data-gemm-mode",
        choices=("overlap", "single"),
        default="single",
        help=(
            "single is the formal, single-GEMM boundary; overlap is a "
            "diagnostic that accumulates one BF16 GEMM per arriving peer slab"
        ),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--num-comm-sm", type=int, default=8)
    parser.add_argument("--num-streams", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--parallel-sends", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-ce", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--push", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reverse", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--local-first", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pack-block", type=int, choices=(128, 256, 512, 1024), default=512)
    parser.add_argument("--pack-warps", type=int, choices=(4, 8), default=4)
    parser.add_argument("--tune-warmup", type=int, default=3)
    parser.add_argument("--tune-iters", type=int, default=12)
    parser.add_argument("--workspace-mib", type=int, default=64)
    parser.add_argument("--math-sm", type=int, default=0)
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--causal-load-balanced", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--matrix-manifest", type=Path)
    parser.add_argument(
        "--cublaslt-library",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "build" / "libfuse_cublaslt_runner.so",
    )
    return parser.parse_args()


def percentile(ordered: list[float], q: float) -> float:
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def summarize(samples: list[float], flops: int) -> dict[str, float]:
    ordered = sorted(samples)
    p50 = percentile(ordered, 0.50)
    return {
        "mean_ms": sum(samples) / len(samples),
        "p50_ms": p50,
        "p95_ms": percentile(ordered, 0.95),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "p50_tflops_per_gpu": flops / p50 / 1.0e9,
    }


@torch.inference_mode()
def timed_critical(
    fn: Callable[[], object],
    warmup: int,
    iterations: int,
    device: torch.device,
    *,
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
        value = torch.tensor(start.elapsed_time(stop), dtype=torch.float64, device=device)
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        samples.append(float(value.item()))
    return samples


def make_plan(
    args: argparse.Namespace,
    source: torch.Tensor,
    weight_nt: torch.Tensor,
    output: torch.Tensor,
) -> CublasLtRunner:
    global _LAST_CUBLASLT_PLAN_KEY, _LAST_CUBLASLT_PLAN
    m, k = source.shape
    n, weight_k = weight_nt.shape
    if weight_k != k or output.shape != (m, n):
        raise ValueError("incompatible cuBLASLt matrix shapes")
    key = (
        source.device.index,
        m,
        n,
        k,
        str(Path(args.cublaslt_library).resolve()),
        args.tune_warmup,
        args.tune_iters,
        args.workspace_mib,
        args.math_sm,
    )
    if _LAST_CUBLASLT_PLAN_KEY == key and _LAST_CUBLASLT_PLAN is not None:
        _LAST_CUBLASLT_PLAN.cache_hit_for_current_entry = True
        return _LAST_CUBLASLT_PLAN
    close_cached_plan()
    _LAST_CUBLASLT_PLAN = CublasLtRunner(
        Path(args.cublaslt_library),
        source,
        weight_nt,
        output,
        tune_warmup=args.tune_warmup,
        tune_iters=args.tune_iters,
        workspace_mib=args.workspace_mib,
        sm_count_target=args.math_sm,
    )
    _LAST_CUBLASLT_PLAN.cache_hit_for_current_entry = False
    _LAST_CUBLASLT_PLAN_KEY = key
    return _LAST_CUBLASLT_PLAN


_LAST_CUBLASLT_PLAN_KEY: tuple[object, ...] | None = None
_LAST_CUBLASLT_PLAN: CublasLtRunner | None = None
_RUNTIME_METADATA: dict[str, object] = {}


def close_cached_plan() -> None:
    global _LAST_CUBLASLT_PLAN_KEY, _LAST_CUBLASLT_PLAN
    if _LAST_CUBLASLT_PLAN is not None:
        _LAST_CUBLASLT_PLAN.close()
    _LAST_CUBLASLT_PLAN_KEY = None
    _LAST_CUBLASLT_PLAN = None


def make_ub(
    args: argparse.Namespace,
    helper: tex.CommOverlapHelper,
    world: int,
    rows: int,
    width: int,
) -> tex.CommOverlapP2P:
    ub = tex.CommOverlapP2P(
        [2 * rows * world, width],
        torch.bfloat16,
        helper,
        world,
        tex.CommOverlapType.AG,
        num_max_streams=args.num_streams,
        comm_cga_size=1,
        gemm_priority=0,
        comm_priority=-1,
        num_comm_sm=args.num_comm_sm,
        set_sm_margin=False,
        atomic_gemm=False,
        use_ce=args.use_ce,
        aggregate=False,
    )
    ub.configure_userbuffers_p2p(args.num_comm_sm, args.use_ce, args.push)
    return ub


def stream_state(
    args: argparse.Namespace,
    ub: tex.CommOverlapP2P,
    world: int,
    device: torch.device,
) -> tuple[list[torch.cuda.ExternalStream], torch.cuda.ExternalStream]:
    active = min(args.num_streams, world - 1) if args.parallel_sends else 1
    sends = [
        torch.cuda.ExternalStream(ub.get_userbuffers_send_stream(i).stream_id, device=device)
        for i in range(active)
    ]
    _, receive = ub.get_communication_stream()
    return sends, torch.cuda.ExternalStream(receive.stream_id, device=device)


def run_qkv(
    args: argparse.Namespace,
    rank: int,
    world: int,
    device: torch.device,
    helper: tex.CommOverlapHelper,
) -> dict[str, object]:
    if args.global_seq % world or args.q_heads % world or args.kv_heads % world:
        raise ValueError("sequence, Q heads, and KV heads must be divisible by CP")
    if args.causal_load_balanced and args.global_seq % (2 * world):
        raise ValueError("causal layout requires sequence divisible by 2*CP")
    m = args.batch * (args.global_seq // world)
    sequence_local = args.global_seq // world
    q_width = args.q_heads * args.head_dim
    kv_width = args.kv_heads * args.head_dim
    width = q_width + 2 * kv_width
    q_local = q_width // world
    kv_local = kv_width // world
    slab_width = q_local + 2 * kv_local

    grad_q = deterministic((args.batch * args.global_seq, q_local), 101 + rank, device)
    grad_k = deterministic((args.batch * args.global_seq, kv_local), 201 + rank, device)
    grad_v = deterministic((args.batch * args.global_seq, kv_local), 301 + rank, device)
    saved_input = deterministic((m, args.hidden), 401 + rank, device)
    weight = deterministic((width, args.hidden), 501, device)
    weight_slabs = []
    for source in range(world):
        q0 = source * q_local
        k0 = q_width + source * kv_local
        v0 = q_width + kv_width + source * kv_local
        rows = torch.cat(
            (weight[q0:q0 + q_local], weight[k0:k0 + kv_local], weight[v0:v0 + kv_local]),
            dim=0,
        )
        weight_slabs.append(rows.T.contiguous())

    ub = make_ub(args, helper, world, m, slab_width)
    storage = ub.get_buffer(False, [2, world, m, slab_width])
    send, recv = storage[0], storage[1]
    send_streams, recv_stream = stream_state(args, ub, world, device)
    dqkv = torch.empty((m, width), dtype=torch.bfloat16, device=device)
    grad_input = torch.empty((m, args.hidden), dtype=torch.bfloat16, device=device)
    grad_weight = torch.zeros_like(weight)
    full_weight_nt = weight.T.contiguous()
    if args.phase_scope == "comm" and args.data_gemm_mode != "single":
        raise ValueError("communication-only tuning requires single-GEMM mode")
    plan = None
    if args.phase_scope != "comm":
        plan = (
            make_plan(args, send[rank], weight_slabs[rank], grad_input)
            if args.data_gemm_mode == "overlap"
            else make_plan(args, dqkv, full_weight_nt, grad_input)
        )

    chunk_bytes = m * slab_width * torch.bfloat16.itemsize
    peer_steps = list(range(1, world))
    if args.reverse:
        peer_steps.reverse()
    remote_sources = [(rank - step) % world for step in peer_steps]
    pack_done = torch.cuda.Event()
    recv_ready = [torch.cuda.Event() for _ in peer_steps]
    send_done = [torch.cuda.Event() for _ in send_streams]
    boundary_start = torch.cuda.Event()
    pack_elements = send.numel()

    def data_phase() -> torch.Tensor:
        main = torch.cuda.current_stream(device)
        boundary_start.record(main)
        recv_stream.wait_event(boundary_start)
        _qkv_inverse_pack_kernel[(triton.cdiv(pack_elements, args.pack_block),)](
            grad_q, grad_k, grad_v, send,
            elements=pack_elements,
            local_tokens=m,
            sequence_local=sequence_local,
            world=world,
            q_local_width=q_local,
            kv_local_width=kv_local,
            peer_width=slab_width,
            causal=args.causal_load_balanced,
            BLOCK=args.pack_block,
            num_warps=args.pack_warps,
        )
        pack_done.record(main)
        for stream in send_streams:
            stream.wait_event(pack_done)
        recv_stream.wait_event(pack_done)
        for index, step in enumerate(peer_steps):
            destination = (rank + step) % world
            ub.userbuffers_p2p_send(
                destination * chunk_bytes,
                (world + rank) * chunk_bytes,
                chunk_bytes,
                destination,
                index % len(send_streams),
            )

        beta = 0.0
        if args.data_gemm_mode == "overlap" and args.local_first:
            assert plan is not None
            plan.accumulate(send[rank], weight_slabs[rank], grad_input, beta=beta)
            beta = 1.0
        for index, step in enumerate(peer_steps):
            source = (rank - step) % world
            ub.userbuffers_p2p_recv(
                rank * chunk_bytes,
                (world + source) * chunk_bytes,
                chunk_bytes,
                source,
            )
            recv_ready[index].record(recv_stream)
            main.wait_event(recv_ready[index])
            if args.data_gemm_mode == "overlap":
                assert plan is not None
                plan.accumulate(recv[source], weight_slabs[source], grad_input, beta=beta)
                beta = 1.0
        if args.data_gemm_mode == "overlap" and not args.local_first:
            assert plan is not None
            plan.accumulate(send[rank], weight_slabs[rank], grad_input, beta=beta)
        for stream, event in zip(send_streams, send_done):
            event.record(stream)
            main.wait_event(event)
        _qkv_ub_unpack_kernel[(triton.cdiv(dqkv.numel(), args.pack_block),)](
            send, recv, dqkv,
            elements=dqkv.numel(),
            local_tokens=m,
            world=world,
            rank=rank,
            q_local_width=q_local,
            kv_local_width=kv_local,
            peer_width=slab_width,
            q_width=q_width,
            kv_width=kv_width,
            qkv_width=width,
            BLOCK=args.pack_block,
            num_warps=args.pack_warps,
        )
        if args.data_gemm_mode == "single" and args.phase_scope != "comm":
            assert plan is not None
            plan.accumulate(dqkv, full_weight_nt, grad_input, beta=0.0)
        return grad_input

    beta = 0 if args.weight_mode == "immediate" else 1

    def weight_phase() -> torch.Tensor:
        return launch_te_wgrad(saved_input, dqkv, grad_weight, beta=beta)

    def full_phase() -> torch.Tensor:
        data_phase()
        return weight_phase()

    data_phase()
    torch.cuda.synchronize(device)
    correctness: dict[str, object] = {}
    if args.check:
        if args.phase_scope == "comm":
            raise ValueError("correctness checks require the full data GEMM")
        dqkv_first = dqkv.clone()
        grad_input_first = grad_input.clone()
        data_phase()
        torch.cuda.synchronize(device)
        deterministic_route_mismatches = torch.count_nonzero(dqkv != dqkv_first).float()
        deterministic_dgrad_mismatches = torch.count_nonzero(
            grad_input != grad_input_first
        ).float()
        peer_q = [deterministic(grad_q.shape, 101 + peer, device) for peer in range(world)]
        peer_k = [deterministic(grad_k.shape, 201 + peer, device) for peer in range(world)]
        peer_v = [deterministic(grad_v.shape, 301 + peer, device) for peer in range(world)]
        rows = global_rows(
            rank=rank, local_tokens=m, batch=args.batch, world=world,
            causal=args.causal_load_balanced, device=device,
        )
        expected_dqkv = torch.cat(
            [item.index_select(0, rows) for item in peer_q]
            + [item.index_select(0, rows) for item in peer_k]
            + [item.index_select(0, rows) for item in peer_v], dim=1,
        )
        route_errors = torch.count_nonzero(dqkv != expected_dqkv).float()
        dx_error = (grad_input.float() - (expected_dqkv @ weight).float()).abs().max()
        launch_te_wgrad(saved_input, dqkv, grad_weight, beta=0)
        first_wgrad = grad_weight.clone()
        launch_te_wgrad(saved_input, dqkv, grad_weight, beta=0)
        deterministic_wgrad_mismatches = torch.count_nonzero(
            grad_weight != first_wgrad
        ).float()
        reference_dw = expected_dqkv.T @ saved_input
        dw_error = (grad_weight.float() - reference_dw.float()).abs().max()
        initial_main_grad = deterministic(weight.shape, 901 + rank, device)
        grad_weight.copy_(initial_main_grad)
        launch_te_wgrad(saved_input, dqkv, grad_weight, beta=1)
        beta1_once_error = (
            grad_weight.float() - (initial_main_grad.float() + reference_dw.float())
        ).abs().max()
        launch_te_wgrad(saved_input, dqkv, grad_weight, beta=1)
        beta1_twice_error = (
            grad_weight.float() - (initial_main_grad.float() + 2.0 * reference_dw.float())
        ).abs().max()
        vector = torch.stack(
            (
                route_errors,
                dx_error,
                dw_error,
                beta1_once_error,
                beta1_twice_error,
                deterministic_route_mismatches,
                deterministic_dgrad_mismatches,
                deterministic_wgrad_mismatches,
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
        mismatch_keys = (
            "route_mismatches",
            "deterministic_route_mismatches",
            "deterministic_dgrad_mismatches",
            "deterministic_wgrad_mismatches",
        )
        if any(correctness[key] for key in mismatch_keys) or max(vector[1:5].tolist()) > 0.25:
            raise RuntimeError(f"QKV TE-UB correctness failed: {correctness}")
        grad_weight.zero_()

    one_gemm_flops = 2 * m * args.hidden * width
    if args.phase_scope in ("comm", "data"):
        data_samples = timed_critical(data_phase, args.warmup, args.iters, device, cuda_graph=args.cuda_graph)
        measured_flops = 0 if args.phase_scope == "comm" else one_gemm_flops
        metrics = {"data": summarize(data_samples, measured_flops), "data_samples_ms": data_samples}
    elif args.weight_mode == "immediate":
        samples = timed_critical(full_phase, args.warmup, args.iters, device, cuda_graph=args.cuda_graph)
        metrics = {"total": summarize(samples, 2 * one_gemm_flops), "samples_ms": samples}
    else:
        data_samples = timed_critical(data_phase, args.warmup, args.iters, device, cuda_graph=args.cuda_graph)
        weight_samples = timed_critical(weight_phase, args.warmup, args.iters, device, cuda_graph=args.cuda_graph)
        samples = [a + b for a, b in zip(data_samples, weight_samples)]
        metrics = {
            "data": summarize(data_samples, one_gemm_flops),
            "weight": summarize(weight_samples, one_gemm_flops),
            "total": summarize(samples, 2 * one_gemm_flops),
            "data_samples_ms": data_samples,
            "weight_samples_ms": weight_samples,
            "samples_ms": samples,
        }
    result = common_result(args, world, "qkv", m, args.hidden, width, metrics, correctness)
    result["cublaslt_plan"] = None if plan is None else plan.info
    result["cublaslt_plan_cache_hit"] = (
        False if plan is None else bool(plan.cache_hit_for_current_entry)
    )
    return result


def run_oproj(
    args: argparse.Namespace,
    rank: int,
    world: int,
    device: torch.device,
    helper: tex.CommOverlapHelper,
) -> dict[str, object]:
    if args.global_seq % world or args.q_heads % world:
        raise ValueError("sequence and Q heads must be divisible by CP")
    if args.causal_load_balanced and args.global_seq % (2 * world):
        raise ValueError("causal layout requires sequence divisible by 2*CP")
    m = args.batch * (args.global_seq // world)
    sequence_local = args.global_seq // world
    width = args.q_heads * args.head_dim
    local_width = width // world
    grad_output = deterministic((m, args.hidden), 601 + rank, device)
    saved_attention = deterministic((m, width), 701 + rank, device)
    weight = deterministic((args.hidden, width), 801, device)
    weight_shards = [
        weight[:, peer * local_width:(peer + 1) * local_width].T.contiguous()
        for peer in range(world)
    ]
    ub = make_ub(args, helper, world, m, local_width)
    storage = ub.get_buffer(False, [2, world, m, local_width])
    send, recv = storage[0], storage[1]
    send_streams, recv_stream = stream_state(args, ub, world, device)
    peer_grad = torch.empty(
        (args.batch * args.global_seq, local_width), dtype=torch.bfloat16, device=device
    )
    grad_weight = torch.zeros_like(weight)
    local_da = torch.empty((m, width), dtype=torch.bfloat16, device=device)
    full_weight_nt = weight.T.contiguous()
    if args.phase_scope == "comm" and args.data_gemm_mode != "single":
        raise ValueError("communication-only tuning requires single-GEMM mode")
    plan = None
    if args.phase_scope != "comm":
        plan = (
            make_plan(args, grad_output, weight_shards[rank], send[rank])
            if args.data_gemm_mode == "overlap"
            else make_plan(args, grad_output, full_weight_nt, local_da)
        )
    else:
        # The communication sweep optimizes only packing, Userbuffers, and
        # unpacking.  Materialize its fixed GEMM input once outside timing.
        local_da.copy_(grad_output @ weight)
    chunk_bytes = m * local_width * torch.bfloat16.itemsize
    peer_steps = list(range(1, world))
    if args.reverse:
        peer_steps.reverse()
    destinations = [(rank + step) % world for step in peer_steps]
    gemm_order = ([rank] + destinations) if args.local_first else (destinations + [rank])
    gemm_done = [torch.cuda.Event() for _ in range(world)]
    recv_ready = [torch.cuda.Event() for _ in peer_steps]
    send_done = [torch.cuda.Event() for _ in send_streams]
    boundary_start = torch.cuda.Event()
    packed_done = torch.cuda.Event()

    def data_phase() -> torch.Tensor:
        main = torch.cuda.current_stream(device)
        boundary_start.record(main)
        recv_stream.wait_event(boundary_start)
        if args.data_gemm_mode == "single":
            if args.phase_scope != "comm":
                assert plan is not None
                plan.accumulate(grad_output, full_weight_nt, local_da, beta=0.0)
            _oproj_route_pack_kernel[(triton.cdiv(send.numel(), args.pack_block),)](
                local_da,
                send,
                elements=send.numel(),
                local_tokens=m,
                attention_width=width,
                local_width=local_width,
                BLOCK=args.pack_block,
                num_warps=args.pack_warps,
            )
            packed_done.record(main)
            for stream in send_streams:
                stream.wait_event(packed_done)
            recv_stream.wait_event(packed_done)
            for index, destination in enumerate(destinations):
                ub.userbuffers_p2p_send(
                    destination * chunk_bytes,
                    (world + rank) * chunk_bytes,
                    chunk_bytes,
                    destination,
                    index % len(send_streams),
                )
        else:
            remote_index = 0
            for destination in gemm_order:
                assert plan is not None
                plan.accumulate(
                    grad_output, weight_shards[destination], send[destination], beta=0.0
                )
                gemm_done[destination].record(main)
                if destination != rank:
                    stream_id = remote_index % len(send_streams)
                    send_streams[stream_id].wait_event(gemm_done[destination])
                    ub.userbuffers_p2p_send(
                        destination * chunk_bytes,
                        (world + rank) * chunk_bytes,
                        chunk_bytes,
                        destination,
                        stream_id,
                    )
                    remote_index += 1
        for index, step in enumerate(peer_steps):
            source = (rank - step) % world
            ub.userbuffers_p2p_recv(
                rank * chunk_bytes,
                (world + source) * chunk_bytes,
                chunk_bytes,
                source,
            )
            recv_ready[index].record(recv_stream)
        for event in recv_ready:
            main.wait_event(event)
        for stream, event in zip(send_streams, send_done):
            event.record(stream)
            main.wait_event(event)
        _oproj_ub_unpack_kernel[(triton.cdiv(send.numel(), args.pack_block),)](
            send, recv, peer_grad,
            elements=send.numel(),
            local_tokens=m,
            sequence_local=sequence_local,
            world=world,
            rank=rank,
            local_width=local_width,
            causal=args.causal_load_balanced,
            BLOCK=args.pack_block,
            num_warps=args.pack_warps,
        )
        return peer_grad

    beta = 0 if args.weight_mode == "immediate" else 1

    def weight_phase() -> torch.Tensor:
        return launch_te_wgrad(saved_attention, grad_output, grad_weight, beta=beta)

    def full_phase() -> torch.Tensor:
        data_phase()
        return weight_phase()

    data_phase()
    torch.cuda.synchronize(device)
    correctness: dict[str, object] = {}
    if args.check:
        if args.phase_scope == "comm":
            raise ValueError("correctness checks require the full data GEMM")
        peer_grad_first = peer_grad.clone()
        data_phase()
        torch.cuda.synchronize(device)
        deterministic_route_mismatches = torch.count_nonzero(
            peer_grad != peer_grad_first
        ).float()
        peer_local = []
        for source in range(world):
            source_grad = deterministic((m, args.hidden), 601 + source, device)
            peer_local.append(source_grad @ weight)
        expected = torch.empty_like(peer_grad)
        for source in range(world):
            rows = global_rows(
                rank=source, local_tokens=m, batch=args.batch, world=world,
                causal=args.causal_load_balanced, device=device,
            )
            expected.index_copy_(
                0, rows,
                peer_local[source][:, rank * local_width:(rank + 1) * local_width],
            )
        route_error = (peer_grad.float() - expected.float()).abs().max()
        launch_te_wgrad(saved_attention, grad_output, grad_weight, beta=0)
        first_wgrad = grad_weight.clone()
        launch_te_wgrad(saved_attention, grad_output, grad_weight, beta=0)
        deterministic_wgrad_mismatches = torch.count_nonzero(
            grad_weight != first_wgrad
        ).float()
        reference_dw = grad_output.T @ saved_attention
        dw_error = (grad_weight.float() - reference_dw.float()).abs().max()
        initial_main_grad = deterministic(weight.shape, 1001 + rank, device)
        grad_weight.copy_(initial_main_grad)
        launch_te_wgrad(saved_attention, grad_output, grad_weight, beta=1)
        beta1_once_error = (
            grad_weight.float() - (initial_main_grad.float() + reference_dw.float())
        ).abs().max()
        launch_te_wgrad(saved_attention, grad_output, grad_weight, beta=1)
        beta1_twice_error = (
            grad_weight.float() - (initial_main_grad.float() + 2.0 * reference_dw.float())
        ).abs().max()
        vector = torch.stack(
            (
                route_error,
                dw_error,
                beta1_once_error,
                beta1_twice_error,
                deterministic_route_mismatches,
                deterministic_wgrad_mismatches,
            )
        )
        dist.all_reduce(vector, op=dist.ReduceOp.MAX)
        correctness = {
            "route_max_abs": float(vector[0].item()),
            "wgrad_max_abs": float(vector[1].item()),
            "main_grad_beta1_once_max_abs": float(vector[2].item()),
            "main_grad_beta1_twice_max_abs": float(vector[3].item()),
            "deterministic_route_mismatches": int(vector[4].item()),
            "deterministic_wgrad_mismatches": int(vector[5].item()),
        }
        if (
            correctness["deterministic_route_mismatches"]
            or correctness["deterministic_wgrad_mismatches"]
            or max(vector[:4].tolist()) > 0.25
        ):
            raise RuntimeError(f"OProj TE-UB correctness failed: {correctness}")
        grad_weight.zero_()

    one_gemm_flops = 2 * m * args.hidden * width
    if args.phase_scope in ("comm", "data"):
        data_samples = timed_critical(data_phase, args.warmup, args.iters, device, cuda_graph=args.cuda_graph)
        measured_flops = 0 if args.phase_scope == "comm" else one_gemm_flops
        metrics = {"data": summarize(data_samples, measured_flops), "data_samples_ms": data_samples}
    elif args.weight_mode == "immediate":
        samples = timed_critical(full_phase, args.warmup, args.iters, device, cuda_graph=args.cuda_graph)
        metrics = {"total": summarize(samples, 2 * one_gemm_flops), "samples_ms": samples}
    else:
        data_samples = timed_critical(data_phase, args.warmup, args.iters, device, cuda_graph=args.cuda_graph)
        weight_samples = timed_critical(weight_phase, args.warmup, args.iters, device, cuda_graph=args.cuda_graph)
        samples = [a + b for a, b in zip(data_samples, weight_samples)]
        metrics = {
            "data": summarize(data_samples, one_gemm_flops),
            "weight": summarize(weight_samples, one_gemm_flops),
            "total": summarize(samples, 2 * one_gemm_flops),
            "data_samples_ms": data_samples,
            "weight_samples_ms": weight_samples,
            "samples_ms": samples,
        }
    result = common_result(args, world, "oproj", m, args.hidden, width, metrics, correctness)
    result["cublaslt_plan"] = None if plan is None else plan.info
    result["cublaslt_plan_cache_hit"] = (
        False if plan is None else bool(plan.cache_hit_for_current_entry)
    )
    return result


def common_result(
    args: argparse.Namespace,
    world: int,
    operator: str,
    m: int,
    hidden: int,
    width: int,
    metrics: dict[str, object],
    correctness: dict[str, object],
) -> dict[str, object]:
    if args.phase_scope == "comm":
        boundary = (
            "pack planar dQ/dK/dV -> Userbuffers A2A -> unpack complete dQKV"
            if operator == "qkv"
            else "pack precomputed dA -> Userbuffers A2A -> unpack final head shard"
        )
    else:
        data_boundary = (
            "pack planar dQ/dK/dV -> Userbuffers A2A -> unpack complete dQKV -> full DGrad GEMM"
            if operator == "qkv"
            else "full DGrad GEMM -> pack dA -> Userbuffers A2A -> unpack final head shard"
        )
        if args.phase_scope == "data":
            boundary = data_boundary
        elif args.weight_mode == "immediate":
            boundary = f"{data_boundary} -> adjacent beta=0 WGrad GEMM"
        else:
            boundary = f"separately timed {data_boundary} and beta=1 WGrad GEMM"
    return {
        "schema": "v10_backward_te_userbuffers_v3",
        "mode": "te_userbuffers_adapted_backward",
        "numerical_contract": (
            "complete_route_then_single_full_gemm"
            if args.data_gemm_mode == "single"
            else (
                "peerwise_overlap_bf16_accumulate_diagnostic"
                if operator == "qkv"
                else "peerwise_disjoint_output_gemm_overlap"
            )
        ),
        "ub_object_lifecycle": "recreate_per_entry",
        "cublaslt_plan_lifecycle": "reuse_adjacent_same_shape_v1",
        "operator": operator,
        "qkv_pack_kernel": (
            "branch_masked_v1" if operator == "qkv" else "not_applicable"
        ),
        "ub_unpack_kernel": "branch_masked_v1",
        "launch": "graph" if args.cuda_graph else "eager",
        "weight_mode": args.weight_mode,
        "weight_accumulation_beta": 1 if args.weight_mode == "deferred" else 0,
        "phase_scope": args.phase_scope,
        "data_gemm_mode": args.data_gemm_mode,
        "timing": "per_sample_local_cuda_event_then_dist_max",
        "timed_boundary": boundary,
        "graph_setup_timed": False,
        "rank_reduction": "MAX",
        "world_size": world,
        "shape": {
            "b_mnk": (
                [m, hidden, width] if operator == "qkv" else [m, width, hidden]
            ),
            "w_mnk": [width if operator == "qkv" else hidden,
                      hidden if operator == "qkv" else width, m],
        },
        "head_geometry": {
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
        },
        "sequence_order": "causal_dual_chunk" if args.causal_load_balanced else "rank_major",
        **_RUNTIME_METADATA,
        "config": vars(args) | {"json_out": str(args.json_out) if args.json_out else None},
        "correctness": correctness,
        "results": metrics,
    }


def write_result(args: argparse.Namespace, result: dict[str, object], rank: int) -> None:
    if rank != 0:
        return
    print(json.dumps(result, indent=2, default=str))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.json_out.with_suffix(args.json_out.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, default=str) + "\n")
        temporary.replace(args.json_out)


def run_one(
    args: argparse.Namespace,
    rank: int,
    world: int,
    device: torch.device,
    helper: tex.CommOverlapHelper,
) -> None:
    if args.operator == "qkv":
        result = run_qkv(args, rank, world, device, helper)
    else:
        result = run_oproj(args, rank, world, device, helper)
    write_result(args, result, rank)
    torch.cuda.synchronize(device)
    dist.barrier()
    del result
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    global _RUNTIME_METADATA
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    properties = torch.cuda.get_device_properties(device)
    local_device = {
        "rank": rank,
        "name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "sm_count": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
    }
    devices: list[dict[str, object] | None] = [None] * world
    dist.all_gather_object(devices, local_device)
    _RUNTIME_METADATA = {
        "environment": {
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "software": {
            "torch": torch.__version__,
            "transformer_engine": transformer_engine.__version__,
            "transformer_engine_path": str(
                Path(transformer_engine.__file__).resolve()
            ),
            "torch_cuda": torch.version.cuda,
            "nccl": ".".join(str(value) for value in torch.cuda.nccl.version()),
        },
        "devices": devices,
    }
    helper = tex.CommOverlapHelper(dist.group.WORLD, dist.group.WORLD)
    try:
        if args.matrix_manifest:
            for entry in json.loads(args.matrix_manifest.read_text()):
                for name, value in entry["arguments"].items():
                    setattr(args, name, value)
                args.json_out = Path(entry["json_out"])
                run_one(args, rank, world, device, helper)
        else:
            run_one(args, rank, world, device, helper)
    finally:
        close_cached_plan()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
