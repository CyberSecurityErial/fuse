#!/usr/bin/env python3
"""Exact TE/cuBLAS + NCCL baselines for the Ulysses fusion boundaries."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist
import transformer_engine
import transformer_engine.pytorch as te
import triton
import triton.language as tl
from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import (
    flash_attn_a2a_communicate,
    get_seq_chunk_ids_for_reordering_before_attn,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("qkv_gemm_a2a", "a2a_gemm", "ulysses_forward"),
        required=True,
    )
    parser.add_argument("--global-seq", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=8192)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=64)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--include-te", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--include-source", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--cuda-graph", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--pack-backend", choices=("torch", "triton"), default="triton")
    parser.add_argument("--pack-block", type=int, choices=(128, 256, 512, 1024), default=1024)
    parser.add_argument("--pack-warps", type=int, choices=(4, 8), default=4)
    parser.add_argument(
        "--nccl-high-priority", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


@triton.jit
def _pack_qk_kernel(
    qkv,
    packed,
    numel: tl.constexpr,
    seq_local: tl.constexpr,
    qkv_width: tl.constexpr,
    q_width: tl.constexpr,
    local_q_heads: tl.constexpr,
    local_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    local_heads = local_q_heads + local_kv_heads
    channel = offsets % head_dim
    logical_head = (offsets // head_dim) % local_heads
    token = (offsets // (head_dim * local_heads)) % seq_local
    peer = offsets // (head_dim * local_heads * seq_local)
    source_head = tl.where(
        logical_head < local_q_heads,
        peer * local_q_heads + logical_head,
        peer * local_kv_heads + logical_head - local_q_heads,
    )
    source_base = tl.where(logical_head < local_q_heads, 0, q_width)
    source = token * qkv_width + source_base + source_head * head_dim + channel
    tl.store(packed + offsets, tl.load(qkv + source, mask=mask), mask=mask)


@triton.jit
def _pack_v_kernel(
    qkv,
    packed,
    numel: tl.constexpr,
    seq_local: tl.constexpr,
    qkv_width: tl.constexpr,
    v_base: tl.constexpr,
    local_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    channel = offsets % head_dim
    logical_head = (offsets // head_dim) % local_kv_heads
    token = (offsets // (head_dim * local_kv_heads)) % seq_local
    peer = offsets // (head_dim * local_kv_heads * seq_local)
    source_head = peer * local_kv_heads + logical_head
    source = token * qkv_width + v_base + source_head * head_dim + channel
    tl.store(packed + offsets, tl.load(qkv + source, mask=mask), mask=mask)


def percentile(ordered: list[float], q: float) -> float:
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def summarize(
    samples: list[float],
    *,
    flops: int | None = None,
    world: int,
    payload_bytes: int | None = None,
) -> dict[str, float]:
    ordered = sorted(samples)
    result = {
        "mean_ms": sum(samples) / len(samples),
        "p50_ms": percentile(ordered, 0.50),
        "p95_ms": percentile(ordered, 0.95),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }
    if flops is not None:
        tflops = flops / result["mean_ms"] / 1.0e9
        result["tflops_per_gpu"] = tflops
        result["aggregate_tflops"] = tflops * world
    if payload_bytes is not None:
        payload_gbps = payload_bytes / result["mean_ms"] / 1.0e6
        result["payload_gbps_per_gpu"] = payload_gbps
        result["remote_gbps_per_gpu"] = payload_gbps * (world - 1) / world
    return result


@torch.inference_mode()
def timed_critical(
    fn: Callable[[], object],
    warmup: int,
    iterations: int,
    device: torch.device,
    *,
    use_cuda_graph: bool,
) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    dist.barrier()

    timed_fn = fn
    graph = None
    if use_cuda_graph:
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
        rank_ms = torch.tensor(
            start.elapsed_time(stop), dtype=torch.float64, device=device
        )
        dist.all_reduce(rank_ms, op=dist.ReduceOp.MAX)
        samples.append(float(rank_ms.item()))
    return samples


class UlyssesRoutes:
    """Preallocated Q/K and V all-to-all layouts for one CP rank."""

    def __init__(
        self,
        *,
        batch: int,
        seq_local: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        world: int,
        device: torch.device,
        pack_backend: str,
        pack_block: int,
        pack_warps: int,
    ) -> None:
        self.batch = batch
        self.seq_local = seq_local
        self.global_seq = seq_local * world
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.world = world
        self.pack_backend = pack_backend
        self.pack_block = pack_block
        self.pack_warps = pack_warps
        self.local_q_heads = q_heads // world
        self.local_kv_heads = kv_heads // world
        self.aliases = self.local_q_heads // self.local_kv_heads
        self.q_width = q_heads * head_dim
        self.kv_width = kv_heads * head_dim
        self.qkv_width = self.q_width + 2 * self.kv_width

        qk_local_heads = self.local_q_heads + self.local_kv_heads
        qk_shape = (world, batch, seq_local, qk_local_heads, head_dim)
        self.qk_send = torch.empty(qk_shape, dtype=torch.bfloat16, device=device)
        self.qk_recv = torch.empty_like(self.qk_send)
        self.qk_output = torch.empty(
            (batch, world, seq_local, qk_local_heads, head_dim),
            dtype=torch.bfloat16,
            device=device,
        )

        v_shape = (world, batch, seq_local, self.local_kv_heads, head_dim)
        self.v_send = torch.empty(v_shape, dtype=torch.bfloat16, device=device)
        self.v_recv = torch.empty_like(self.v_send)
        self.v_kv = torch.empty(
            (batch, self.local_kv_heads, world, seq_local, head_dim),
            dtype=torch.bfloat16,
            device=device,
        )

    def qk(self, qkv: torch.Tensor) -> torch.Tensor:
        qkv = qkv.view(self.batch, self.seq_local, self.qkv_width)
        if self.pack_backend == "triton" and self.batch == 1:
            output_elements = self.qk_send.numel()
            _pack_qk_kernel[(triton.cdiv(output_elements, self.pack_block),)](
                qkv,
                self.qk_send,
                numel=output_elements,
                seq_local=self.seq_local,
                qkv_width=self.qkv_width,
                q_width=self.q_width,
                local_q_heads=self.local_q_heads,
                local_kv_heads=self.local_kv_heads,
                head_dim=self.head_dim,
                BLOCK=self.pack_block,
                num_warps=self.pack_warps,
            )
        else:
            q = qkv[..., : self.q_width].view(
                self.batch,
                self.seq_local,
                self.world,
                self.local_q_heads,
                self.head_dim,
            )
            k = qkv[..., self.q_width : self.q_width + self.kv_width].view(
                self.batch,
                self.seq_local,
                self.world,
                self.local_kv_heads,
                self.head_dim,
            )
            self.qk_send[..., : self.local_q_heads, :].copy_(
                q.permute(2, 0, 1, 3, 4)
            )
            self.qk_send[..., self.local_q_heads :, :].copy_(
                k.permute(2, 0, 1, 3, 4)
            )
        dist.all_to_all_single(self.qk_recv, self.qk_send)
        if self.batch == 1:
            return self.qk_recv.view(
                1,
                self.global_seq,
                self.local_q_heads + self.local_kv_heads,
                self.head_dim,
            )
        self.qk_output.copy_(self.qk_recv.permute(1, 0, 2, 3, 4))
        return self.qk_output.view(
            self.batch,
            self.global_seq,
            self.local_q_heads + self.local_kv_heads,
            self.head_dim,
        )

    def v(self, qkv: torch.Tensor) -> torch.Tensor:
        qkv = qkv.view(self.batch, self.seq_local, self.qkv_width)
        if self.pack_backend == "triton" and self.batch == 1:
            output_elements = self.v_send.numel()
            _pack_v_kernel[(triton.cdiv(output_elements, self.pack_block),)](
                qkv,
                self.v_send,
                numel=output_elements,
                seq_local=self.seq_local,
                qkv_width=self.qkv_width,
                v_base=self.q_width + self.kv_width,
                local_kv_heads=self.local_kv_heads,
                head_dim=self.head_dim,
                BLOCK=self.pack_block,
                num_warps=self.pack_warps,
            )
        else:
            v = qkv[..., self.q_width + self.kv_width :].view(
                self.batch,
                self.seq_local,
                self.world,
                self.local_kv_heads,
                self.head_dim,
            )
            self.v_send.copy_(v.permute(2, 0, 1, 3, 4))
        dist.all_to_all_single(self.v_recv, self.v_send)
        if self.batch == 1 and self.local_kv_heads == 1:
            return self.v_recv.view(1, self.global_seq, self.head_dim).expand(
                self.local_q_heads, self.global_seq, self.head_dim
            )
        self.v_kv.copy_(self.v_recv.permute(1, 3, 0, 2, 4))
        base = self.v_kv.view(
            self.batch * self.local_kv_heads, self.global_seq, self.head_dim
        )
        if self.batch == 1 and self.local_kv_heads == 1:
            return base.expand(self.local_q_heads, self.global_seq, self.head_dim)
        return (
            base.view(
                self.batch,
                self.local_kv_heads,
                1,
                self.global_seq,
                self.head_dim,
            )
            .expand(
                self.batch,
                self.local_kv_heads,
                self.aliases,
                self.global_seq,
                self.head_dim,
            )
            .reshape(self.batch * self.local_q_heads, self.global_seq, self.head_dim)
        )


def validate(args: argparse.Namespace, world: int) -> None:
    positive = (
        args.global_seq,
        args.hidden,
        args.batch,
        args.q_heads,
        args.kv_heads,
        args.head_dim,
        args.warmup,
        args.iters,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("all shape and sampling arguments must be positive")
    if args.global_seq % world:
        raise ValueError("global sequence must be divisible by CP world size")
    if args.q_heads % world or args.kv_heads % world:
        raise ValueError("Q and KV head counts must be divisible by CP world size")
    if args.q_heads % args.kv_heads:
        raise ValueError("Q heads must be divisible by KV heads")


def exact_route_check(
    routes: UlyssesRoutes,
    qkv: torch.Tensor,
    rank: int,
    world: int,
) -> dict[str, int]:
    peers = [torch.empty_like(qkv) for _ in range(world)]
    dist.all_gather(peers, qkv)

    qk_actual = routes.qk(qkv)
    qk_parts: list[torch.Tensor] = []
    kv_begin = rank * routes.local_kv_heads
    kv_end = kv_begin + routes.local_kv_heads
    q_begin = rank * routes.local_q_heads
    q_end = q_begin + routes.local_q_heads
    v_parts: list[torch.Tensor] = []
    for peer in peers:
        peer = peer.view(routes.batch, routes.seq_local, routes.qkv_width)
        q = peer[..., : routes.q_width].view(
            routes.batch,
            routes.seq_local,
            routes.q_heads,
            routes.head_dim,
        )[..., q_begin:q_end, :]
        k = peer[..., routes.q_width : routes.q_width + routes.kv_width].view(
            routes.batch,
            routes.seq_local,
            routes.kv_heads,
            routes.head_dim,
        )[..., kv_begin:kv_end, :]
        v = peer[..., routes.q_width + routes.kv_width :].view(
            routes.batch,
            routes.seq_local,
            routes.kv_heads,
            routes.head_dim,
        )[..., kv_begin:kv_end, :]
        qk_parts.append(torch.cat((q, k), dim=2))
        v_parts.append(v)
    qk_expected = torch.cat(qk_parts, dim=1)

    v_actual = routes.v(qkv)
    v_kv = torch.cat(v_parts, dim=1).permute(0, 2, 1, 3)
    v_expected = (
        v_kv.unsqueeze(2)
        .expand(
            routes.batch,
            routes.local_kv_heads,
            routes.aliases,
            routes.global_seq,
            routes.head_dim,
        )
        .reshape(routes.batch * routes.local_q_heads, routes.global_seq, routes.head_dim)
    )
    qk_mismatch = int(torch.count_nonzero(qk_actual != qk_expected).item())
    v_mismatch = int(torch.count_nonzero(v_actual != v_expected).item())
    counts = torch.tensor([qk_mismatch, v_mismatch], dtype=torch.int64, device=qkv.device)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    return {"qk_mismatches": int(counts[0]), "v_mismatches": int(counts[1])}


def exact_te_source_route_check(
    qkv: torch.Tensor,
    actual: list[torch.Tensor],
    *,
    batch: int,
    seq_local: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    rank: int,
    world: int,
    chunk_ids: torch.Tensor,
) -> int:
    peers = [torch.empty_like(qkv) for _ in range(world)]
    dist.all_gather(peers, qkv)
    q_local, kv_local = q_heads // world, kv_heads // world
    q_begin, kv_begin = rank * q_local, rank * kv_local
    peer_parts: list[list[torch.Tensor]] = [[], [], []]
    q_width, kv_width = q_heads * head_dim, kv_heads * head_dim
    for peer in peers:
        peer = peer.view(batch, seq_local, q_width + 2 * kv_width)
        peer_parts[0].append(
            peer[..., :q_width]
            .view(batch, seq_local, q_heads, head_dim)[..., q_begin : q_begin + q_local, :]
        )
        peer_parts[1].append(
            peer[..., q_width : q_width + kv_width]
            .view(batch, seq_local, kv_heads, head_dim)[..., kv_begin : kv_begin + kv_local, :]
        )
        peer_parts[2].append(
            peer[..., q_width + kv_width :]
            .view(batch, seq_local, kv_heads, head_dim)[..., kv_begin : kv_begin + kv_local, :]
        )

    mismatch = 0
    for parts, tensor in zip(peer_parts, actual):
        raw_receive = torch.stack(parts, dim=0)
        expected = raw_receive.movedim(0, 1).contiguous()
        expected = expected.view(batch, world * 2, -1, expected.shape[-2], head_dim)
        expected = torch.index_select(expected, dim=1, index=chunk_ids)
        expected = expected.view(batch, world * seq_local, expected.shape[-2], head_dim)
        mismatch += int(torch.count_nonzero(tensor != expected).item())
    total = torch.tensor(mismatch, dtype=torch.int64, device=qkv.device)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return int(total)


def expand_gqa_value(
    value: torch.Tensor,
    *,
    batch: int,
    local_q_heads: int,
    local_kv_heads: int,
) -> torch.Tensor:
    """Expose local KV groups to query groups without copying the common case."""
    global_seq, head_dim = value.shape[1], value.shape[-1]
    base = value.permute(0, 2, 1, 3).reshape(
        batch * local_kv_heads, global_seq, head_dim
    )
    aliases = local_q_heads // local_kv_heads
    if batch == 1 and local_kv_heads == 1:
        return base.expand(local_q_heads, global_seq, head_dim)
    return (
        base.view(batch, local_kv_heads, 1, global_seq, head_dim)
        .expand(batch, local_kv_heads, aliases, global_seq, head_dim)
        .reshape(batch * local_q_heads, global_seq, head_dim)
    )


def overlap_summary(
    compute: dict[str, float],
    communication: dict[str, float],
    combined: dict[str, float],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for statistic in ("mean_ms", "p50_ms"):
        compute_ms = compute[statistic]
        communication_ms = communication[statistic]
        combined_ms = combined[statistic]
        hidden_ms = compute_ms + communication_ms - combined_ms
        result[f"{statistic.removesuffix('_ms')}_hidden_ms"] = hidden_ms
        result[f"{statistic.removesuffix('_ms')}_communication_hidden_fraction"] = (
            hidden_ms / min(compute_ms, communication_ms)
        )
        result[f"{statistic.removesuffix('_ms')}_exposed_above_ideal_ms"] = (
            combined_ms - max(compute_ms, communication_ms)
        )
    return result


def run(args: argparse.Namespace, rank: int, world: int, device: torch.device):
    validate(args, world)
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
    seq_local = args.global_seq // world
    qkv_m = args.batch * seq_local
    qkv_n = (args.q_heads + 2 * args.kv_heads) * args.head_dim
    local_q_heads = args.q_heads // world
    pv_l = args.batch * local_q_heads
    generator = torch.Generator(device=device).manual_seed(1701 + rank)

    x = torch.empty(
        (qkv_m, args.hidden), dtype=torch.bfloat16, device=device
    ).uniform_(-0.125, 0.125, generator=generator)
    linear = te.Linear(
        args.hidden,
        qkv_n,
        bias=False,
        params_dtype=torch.bfloat16,
        device=device,
        name="ulysses_qkv_projection",
    ).eval()
    dist.broadcast(linear.weight, src=0)
    qkv_cublas = torch.empty((qkv_m, qkv_n), dtype=torch.bfloat16, device=device)
    routes = UlyssesRoutes(
        batch=args.batch,
        seq_local=seq_local,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        world=world,
        device=device,
        pack_backend=args.pack_backend,
        pack_block=args.pack_block,
        pack_warps=args.pack_warps,
    )
    cp_stream = torch.cuda.Stream(device=device)
    chunk_ids = get_seq_chunk_ids_for_reordering_before_attn(world, device)

    def split_qkv(qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = qkv.view(args.batch, seq_local, qkv_n)
        q_width = args.q_heads * args.head_dim
        kv_width = args.kv_heads * args.head_dim
        q = qkv[..., :q_width].view(
            args.batch, seq_local, args.q_heads, args.head_dim
        )
        k = qkv[..., q_width : q_width + kv_width].view(
            args.batch, seq_local, args.kv_heads, args.head_dim
        )
        v = qkv[..., q_width + kv_width :].view(
            args.batch, seq_local, args.kv_heads, args.head_dim
        )
        return q, k, v

    def te_source_qkv_a2a(qkv: torch.Tensor):
        q, k, v = split_qkv(qkv)
        return flash_attn_a2a_communicate(
            [q, k, v],
            chunk_ids,
            1,
            world,
            dist.group.WORLD,
            cp_stream,
            True,
            qkv_format="bshd",
            a2a_input_names=["q", "k", "v"],
        )

    def cublas_project() -> torch.Tensor:
        return torch.mm(x, linear.weight.t(), out=qkv_cublas)

    def te_project() -> torch.Tensor:
        return linear(x)

    cublas_project()
    torch.cuda.synchronize(device)
    check = exact_route_check(routes, qkv_cublas, rank, world) if args.check else {}
    if args.check and args.include_source:
        check["te_source_qkv_mismatches"] = exact_te_source_route_check(
            qkv_cublas,
            te_source_qkv_a2a(qkv_cublas),
            batch=args.batch,
            seq_local=seq_local,
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            rank=rank,
            world=world,
            chunk_ids=chunk_ids,
        )
    if any(check.values()):
        raise RuntimeError(f"route correctness failed: {check}")

    qkv_flops = 2 * qkv_m * qkv_n * args.hidden
    qk_payload = (
        args.batch
        * seq_local
        * (args.q_heads + args.kv_heads)
        * args.head_dim
        * torch.bfloat16.itemsize
    )
    v_payload = (
        args.batch
        * seq_local
        * args.kv_heads
        * args.head_dim
        * torch.bfloat16.itemsize
    )

    metrics: dict[str, dict[str, object]] = {}

    def measure(
        name: str,
        fn: Callable[[], object],
        *,
        flops: int | None = None,
        payload_bytes: int | None = None,
    ) -> None:
        samples = timed_critical(
            fn,
            args.warmup,
            args.iters,
            device,
            use_cuda_graph=args.cuda_graph,
        )
        metrics[name] = {
            "samples_ms": samples,
            "stats": summarize(
                samples,
                flops=flops,
                world=world,
                payload_bytes=payload_bytes,
            ),
        }

    if args.mode in ("qkv_gemm_a2a", "ulysses_forward"):
        if args.include_te:
            measure("te_qkv_gemm", te_project, flops=qkv_flops)
        measure("cublas_qkv_gemm", cublas_project, flops=qkv_flops)
        measure("nccl_qk_a2a", lambda: routes.qk(qkv_cublas), payload_bytes=qk_payload)

        def te_qkv_qk() -> torch.Tensor:
            return routes.qk(te_project())

        def cublas_qkv_qk() -> torch.Tensor:
            return routes.qk(cublas_project())

        if args.include_te:
            measure("te_qkv_gemm_qk_a2a", te_qkv_qk, flops=qkv_flops)
        measure("cublas_qkv_gemm_qk_a2a", cublas_qkv_qk, flops=qkv_flops)

        if args.include_source:
            measure(
                "te_source_qkv_a2a",
                lambda: te_source_qkv_a2a(qkv_cublas),
                payload_bytes=qk_payload + v_payload,
            )

        def te_source_qkv_gemm_a2a():
            return te_source_qkv_a2a(te_project())

        if args.include_source:
            measure(
                "te_source_qkv_gemm_a2a", te_source_qkv_gemm_a2a, flops=qkv_flops
            )

    if args.mode in ("a2a_gemm", "ulysses_forward"):
        probability = torch.empty(
            (pv_l, args.global_seq, args.global_seq),
            dtype=torch.bfloat16,
            device=device,
        ).uniform_(-0.125, 0.125, generator=generator)
        pv_output = torch.empty(
            (pv_l, args.global_seq, args.head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        value = routes.v(qkv_cublas)

        def cublas_bmm() -> torch.Tensor:
            return torch.bmm(probability, value, out=pv_output)

        def nccl_v_a2a() -> torch.Tensor:
            return routes.v(qkv_cublas)

        def nccl_v_a2a_bmm() -> torch.Tensor:
            routed = routes.v(qkv_cublas)
            return torch.bmm(probability, routed, out=pv_output)

        pv_flops = 2 * pv_l * args.global_seq * args.global_seq * args.head_dim
        measure("nccl_v_a2a", nccl_v_a2a, payload_bytes=v_payload)
        measure("cublas_pv_bmm", cublas_bmm, flops=pv_flops)
        measure("nccl_v_a2a_cublas_pv_bmm", nccl_v_a2a_bmm, flops=pv_flops)

        if args.mode == "ulysses_forward":
            total_flops = qkv_flops + pv_flops

            def te_boundary_pair() -> torch.Tensor:
                qkv = te_project()
                routes.qk(qkv)
                routed_v = routes.v(qkv)
                return torch.bmm(probability, routed_v, out=pv_output)

            def cublas_boundary_pair() -> torch.Tensor:
                qkv = cublas_project()
                routes.qk(qkv)
                routed_v = routes.v(qkv)
                return torch.bmm(probability, routed_v, out=pv_output)

            def te_source_boundary_pair() -> torch.Tensor:
                _, _, value_source = te_source_qkv_a2a(te_project())
                routed_v = expand_gqa_value(
                    value_source,
                    batch=args.batch,
                    local_q_heads=local_q_heads,
                    local_kv_heads=args.kv_heads // world,
                )
                return torch.bmm(probability, routed_v, out=pv_output)

            if args.include_te:
                measure("te_nccl_boundary_pair", te_boundary_pair, flops=total_flops)
            measure(
                "cublas_nccl_boundary_pair", cublas_boundary_pair, flops=total_flops
            )
            if args.include_source:
                measure(
                    "te_source_boundary_pair", te_source_boundary_pair, flops=total_flops
                )

    derived: dict[str, object] = {}
    if args.mode in ("qkv_gemm_a2a", "ulysses_forward"):
        if args.include_te:
            derived["te_qkv_qk_overlap"] = overlap_summary(
                metrics["te_qkv_gemm"]["stats"],
                metrics["nccl_qk_a2a"]["stats"],
                metrics["te_qkv_gemm_qk_a2a"]["stats"],
            )
        derived["cublas_qkv_qk_overlap"] = overlap_summary(
            metrics["cublas_qkv_gemm"]["stats"],
            metrics["nccl_qk_a2a"]["stats"],
            metrics["cublas_qkv_gemm_qk_a2a"]["stats"],
        )
    if args.mode in ("a2a_gemm", "ulysses_forward"):
        derived["nccl_v_cublas_pv_overlap"] = overlap_summary(
            metrics["cublas_pv_bmm"]["stats"],
            metrics["nccl_v_a2a"]["stats"],
            metrics["nccl_v_a2a_cublas_pv_bmm"]["stats"],
        )
    if args.mode == "ulysses_forward":
        prefixes = ("te", "cublas") if args.include_te else ("cublas",)
        for prefix in prefixes:
            separated_mean = (
                metrics[f"{prefix}_qkv_gemm_qk_a2a"]["stats"]["mean_ms"]
                + metrics["nccl_v_a2a_cublas_pv_bmm"]["stats"]["mean_ms"]
            )
            separated_p50 = (
                metrics[f"{prefix}_qkv_gemm_qk_a2a"]["stats"]["p50_ms"]
                + metrics["nccl_v_a2a_cublas_pv_bmm"]["stats"]["p50_ms"]
            )
            pair = metrics[f"{prefix}_nccl_boundary_pair"]["stats"]
            derived[f"{prefix}_joint_boundary_pair"] = {
                "mean_separate_over_joint": separated_mean / pair["mean_ms"],
                "p50_separate_over_joint": separated_p50 / pair["p50_ms"],
            }

    return {
        "mode": args.mode,
        "model_shape": {
            "batch": args.batch,
            "global_seq": args.global_seq,
            "seq_local": seq_local,
            "hidden": args.hidden,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
        },
        "gemm_shapes": {
            "qkv": {"m": qkv_m, "n": qkv_n, "k": args.hidden, "l": 1},
            "pv": {
                "m": args.global_seq,
                "n": args.head_dim,
                "k": args.global_seq,
                "l": pv_l,
            },
        },
        "world_size": world,
        "warmup": args.warmup,
        "iterations": args.iters,
        "dtype": "bfloat16",
        "timing": "per-sample max-rank CUDA-event critical path",
        "scope": "QKV projection + Q/K A2A and V A2A + PV; QK/softmax excluded",
        "include_te": args.include_te,
        "include_source": args.include_source,
        "cuda_graph": args.cuda_graph,
        "pack_backend": args.pack_backend,
        "pack_block": args.pack_block,
        "pack_warps": args.pack_warps,
        "nccl_high_priority": args.nccl_high_priority,
        "software": {
            "torch": torch.__version__,
            "transformer_engine": transformer_engine.__version__,
            "torch_cuda": torch.version.cuda,
            "nccl": ".".join(str(value) for value in torch.cuda.nccl.version()),
        },
        "devices": devices,
        "environment": {
            name: os.environ.get(name)
            for name in (
                "NCCL_MIN_P2P_NCHANNELS",
                "NCCL_MAX_P2P_NCHANNELS",
                "NCCL_P2P_NVL_CHUNKSIZE",
                "NCCL_P2P_LL_THRESHOLD",
                "NCCL_GRAPH_REGISTER",
                "NCCL_LOCAL_REGISTER",
                "NCCL_IB_DISABLE",
            )
        },
        "implementations": {
            "te_qkv": "transformer_engine.pytorch.Linear",
            "cublas_qkv": "torch.mm(out=...), PyTorch CUDA BLAS",
            "pv": "torch.bmm(out=...), PyTorch CUDA BLAS",
            "a2a": "torch.distributed.all_to_all_single / NCCL grouped send-recv",
        },
        "correctness": check,
        "derived": derived,
        "results": {name: entry["stats"] for name, entry in metrics.items()},
        "samples_ms": {name: entry["samples_ms"] for name, entry in metrics.items()},
    }


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    process_group_options = dist.ProcessGroupNCCL.Options()
    process_group_options.is_high_priority_stream = args.nccl_high_priority
    dist.init_process_group(
        "nccl", device_id=device, pg_options=process_group_options
    )
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    result = run(args, rank, world, device)
    if rank == 0:
        for name, values in result["results"].items():
            suffix = ""
            if "tflops_per_gpu" in values:
                suffix += f" TFLOPS/GPU={values['tflops_per_gpu']:.1f}"
            if "payload_gbps_per_gpu" in values:
                suffix += f" payload={values['payload_gbps_per_gpu']:.1f} GB/s"
            print(
                f"{name:31s} mean={values['mean_ms']:.4f} ms "
                f"p50={values['p50_ms']:.4f} p95={values['p95_ms']:.4f}{suffix}",
                flush=True,
            )
        print(f"correctness={result['correctness']}", flush=True)
        print(f"derived={json.dumps(result['derived'], sort_keys=True)}", flush=True)
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
