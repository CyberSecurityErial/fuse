#!/usr/bin/env python3
"""Exact TE/cuBLAS + NCCL baselines for the Ulysses fusion boundaries."""

from __future__ import annotations

import argparse
import ctypes
import gc
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


class CublasLtInfo(ctypes.Structure):
    _fields_ = [
        ("returned", ctypes.c_int),
        ("valid", ctypes.c_int),
        ("algo_id", ctypes.c_int),
        ("tile_id", ctypes.c_int),
        ("stages_id", ctypes.c_int),
        ("split_k", ctypes.c_int),
        ("reduction", ctypes.c_int),
        ("cta_swizzle", ctypes.c_int),
        ("custom", ctypes.c_int),
        ("inner_shape", ctypes.c_int),
        ("cluster_shape", ctypes.c_int),
        ("workspace_bytes", ctypes.c_uint64),
        ("tune_ms", ctypes.c_float),
        ("waves", ctypes.c_float),
    ]


class CublasLtRunner:
    """Locally autotuned BF16 NT cuBLASLt plan, callable on a Torch stream."""

    def __init__(
        self,
        library: Path,
        a: torch.Tensor,
        b_nt: torch.Tensor,
        d: torch.Tensor,
        *,
        tune_warmup: int,
        tune_iters: int,
        workspace_mib: int,
        sm_count_target: int = 0,
    ) -> None:
        self.library = ctypes.CDLL(str(library))
        self.library.fuse_cublaslt_last_error.restype = ctypes.c_char_p
        self.library.fuse_cublaslt_bf16_create.argtypes = [
            ctypes.c_int,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
        ]
        self.library.fuse_cublaslt_bf16_create.restype = ctypes.c_void_p
        self.library.fuse_cublaslt_bf16_create_ex.argtypes = [
            *self.library.fuse_cublaslt_bf16_create.argtypes,
            ctypes.c_int,
        ]
        self.library.fuse_cublaslt_bf16_create_ex.restype = ctypes.c_void_p
        self.library.fuse_cublaslt_bf16_run.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.library.fuse_cublaslt_bf16_run.restype = ctypes.c_int
        self.library.fuse_cublaslt_bf16_run_beta.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_float,
        ]
        self.library.fuse_cublaslt_bf16_run_beta.restype = ctypes.c_int
        self.library.fuse_cublaslt_bf16_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(CublasLtInfo),
        ]
        self.library.fuse_cublaslt_bf16_info.restype = ctypes.c_int
        self.library.fuse_cublaslt_bf16_destroy.argtypes = [ctypes.c_void_p]
        self.library.fuse_cublaslt_bf16_destroy.restype = None
        if a.dtype != torch.bfloat16 or b_nt.dtype != torch.bfloat16 or d.dtype != torch.bfloat16:
            raise TypeError("cuBLASLt runner requires BF16 tensors")
        if not (a.is_contiguous() and b_nt.is_contiguous() and d.is_contiguous()):
            raise ValueError("cuBLASLt runner requires contiguous row-major tensors")
        m, k = a.shape
        n, weight_k = b_nt.shape
        if weight_k != k or d.shape != (m, n):
            raise ValueError("incompatible cuBLASLt matrix shapes")
        stream = torch.cuda.current_stream(a.device).cuda_stream
        create = (
            self.library.fuse_cublaslt_bf16_create_ex
            if sm_count_target > 0
            else self.library.fuse_cublaslt_bf16_create
        )
        create_args = [
            a.device.index,
            m,
            n,
            k,
            a.data_ptr(),
            b_nt.data_ptr(),
            d.data_ptr(),
            stream,
            tune_warmup,
            tune_iters,
            workspace_mib << 20,
        ]
        if sm_count_target > 0:
            create_args.append(sm_count_target)
        self.handle = create(*create_args)
        if not self.handle:
            error = self.library.fuse_cublaslt_last_error().decode()
            raise RuntimeError(f"cuBLASLt autotune failed: {error}")
        raw = CublasLtInfo()
        if not self.library.fuse_cublaslt_bf16_info(self.handle, ctypes.byref(raw)):
            raise RuntimeError("cuBLASLt plan info query failed")
        self.info = {name: getattr(raw, name) for name, _ in raw._fields_}

    def __call__(self, a: torch.Tensor, b_nt: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        stream = torch.cuda.current_stream(a.device).cuda_stream
        if not self.library.fuse_cublaslt_bf16_run(
            self.handle, a.data_ptr(), b_nt.data_ptr(), d.data_ptr(), stream
        ):
            error = self.library.fuse_cublaslt_last_error().decode()
            raise RuntimeError(f"cuBLASLt launch failed: {error}")
        return d

    def accumulate(
        self, a: torch.Tensor, b_nt: torch.Tensor, d: torch.Tensor, *, beta: float
    ) -> torch.Tensor:
        stream = torch.cuda.current_stream(a.device).cuda_stream
        if not self.library.fuse_cublaslt_bf16_run_beta(
            self.handle,
            a.data_ptr(),
            b_nt.data_ptr(),
            d.data_ptr(),
            stream,
            beta,
        ):
            error = self.library.fuse_cublaslt_last_error().decode()
            raise RuntimeError(f"cuBLASLt launch failed: {error}")
        return d

    def close(self) -> None:
        if self.handle:
            self.library.fuse_cublaslt_bf16_destroy(self.handle)
            self.handle = None


_CUBLASLT_RUNNERS: dict[tuple[object, ...], CublasLtRunner] = {}


def cached_cublaslt_runner(
    args: argparse.Namespace,
    a: torch.Tensor,
    b_nt: torch.Tensor,
    d: torch.Tensor,
) -> CublasLtRunner:
    m, k = a.shape
    n = b_nt.shape[0]
    key = (
        a.device.index,
        m,
        n,
        k,
        str(args.cublaslt_library.resolve()),
        args.cublaslt_tune_warmup,
        args.cublaslt_tune_iters,
        args.cublaslt_workspace_mib,
    )
    if key not in _CUBLASLT_RUNNERS:
        _CUBLASLT_RUNNERS[key] = CublasLtRunner(
            args.cublaslt_library,
            a,
            b_nt,
            d,
            tune_warmup=args.cublaslt_tune_warmup,
            tune_iters=args.cublaslt_tune_iters,
            workspace_mib=args.cublaslt_workspace_mib,
        )
    return _CUBLASLT_RUNNERS[key]


def close_cublaslt_runners() -> None:
    for runner in _CUBLASLT_RUNNERS.values():
        runner.close()
    _CUBLASLT_RUNNERS.clear()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "qkv_gemm_a2a",
            "oproj_a2a_gemm",
        ),
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
        "--include-qk", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--metric-profile",
        choices=("route", "boundary", "full"),
        default="full",
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
    parser.add_argument(
        "--matrix-manifest",
        type=Path,
        help="JSON list of NCCL/execution configurations and output paths",
    )
    parser.add_argument(
        "--cublaslt-library",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "build" / "libfuse_cublaslt_runner.so",
    )
    parser.add_argument("--cublaslt-tune-warmup", type=int, default=5)
    parser.add_argument("--cublaslt-tune-iters", type=int, default=30)
    parser.add_argument("--cublaslt-workspace-mib", type=int, default=64)
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
    offsets = (
        tl.program_id(0).to(tl.int64) * BLOCK
        + tl.arange(0, BLOCK).to(tl.int64)
    )
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
    offsets = (
        tl.program_id(0).to(tl.int64) * BLOCK
        + tl.arange(0, BLOCK).to(tl.int64)
    )
    mask = offsets < numel
    channel = offsets % head_dim
    logical_head = (offsets // head_dim) % local_kv_heads
    token = (offsets // (head_dim * local_kv_heads)) % seq_local
    peer = offsets // (head_dim * local_kv_heads * seq_local)
    source_head = peer * local_kv_heads + logical_head
    source = token * qkv_width + v_base + source_head * head_dim + channel
    tl.store(packed + offsets, tl.load(qkv + source, mask=mask), mask=mask)


@triton.jit
def _pack_full_qkv_kernel(
    qkv,
    packed,
    numel: tl.constexpr,
    seq_local: tl.constexpr,
    qkv_width: tl.constexpr,
    q_width: tl.constexpr,
    kv_width: tl.constexpr,
    local_q_heads: tl.constexpr,
    local_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = (
        tl.program_id(0).to(tl.int64) * BLOCK
        + tl.arange(0, BLOCK).to(tl.int64)
    )
    mask = offsets < numel
    q_elements = tl.full((), seq_local * local_q_heads * head_dim, tl.int64)
    kv_elements = tl.full((), seq_local * local_kv_heads * head_dim, tl.int64)
    peer_elements = q_elements + 2 * kv_elements
    peer = offsets // peer_elements
    within = offsets % peer_elements
    is_q = within < q_elements
    is_k = (within >= q_elements) & (within < q_elements + kv_elements)
    segment_element = tl.where(
        is_q,
        within,
        tl.where(is_k, within - q_elements, within - q_elements - kv_elements),
    )
    local_heads = tl.where(is_q, local_q_heads, local_kv_heads)
    channel = segment_element % head_dim
    logical_head = (segment_element // head_dim) % local_heads
    token = segment_element // (head_dim * local_heads)
    source_head = peer * local_heads + logical_head
    source_base = tl.where(is_q, 0, tl.where(is_k, q_width, q_width + kv_width))
    source = token * qkv_width + source_base + source_head * head_dim + channel
    tl.store(packed + offsets, tl.load(qkv + source, mask=mask), mask=mask)


@triton.jit
def _unpack_full_qkv_kernel(
    received,
    output,
    numel: tl.constexpr,
    seq_local: tl.constexpr,
    world: tl.constexpr,
    local_q_heads: tl.constexpr,
    local_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = (
        tl.program_id(0).to(tl.int64) * BLOCK
        + tl.arange(0, BLOCK).to(tl.int64)
    )
    mask = offsets < numel
    global_seq = tl.full((), seq_local * world, tl.int64)
    q_elements = global_seq * local_q_heads * head_dim
    kv_elements = global_seq * local_kv_heads * head_dim
    local_q_per_peer = tl.full(
        (), seq_local * local_q_heads * head_dim, tl.int64
    )
    local_kv_per_peer = tl.full(
        (), seq_local * local_kv_heads * head_dim, tl.int64
    )
    peer_elements = local_q_per_peer + 2 * local_kv_per_peer
    is_q = offsets < q_elements
    is_k = (offsets >= q_elements) & (offsets < q_elements + kv_elements)
    segment_element = tl.where(
        is_q,
        offsets,
        tl.where(is_k, offsets - q_elements, offsets - q_elements - kv_elements),
    )
    local_heads = tl.where(is_q, local_q_heads, local_kv_heads)
    channel = segment_element % head_dim
    logical_head = (segment_element // head_dim) % local_heads
    global_token = segment_element // (head_dim * local_heads)
    source_rank = global_token // seq_local
    local_token = global_token % seq_local
    segment_base = tl.where(
        is_q,
        0,
        tl.where(is_k, local_q_per_peer, local_q_per_peer + local_kv_per_peer),
    )
    source = (
        source_rank * peer_elements
        + segment_base
        + (local_token * local_heads + logical_head) * head_dim
        + channel
    )
    tl.store(output + offsets, tl.load(received + source, mask=mask), mask=mask)


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
        full_elements = batch * seq_local * (
            self.local_q_heads + 2 * self.local_kv_heads
        ) * head_dim
        self.full_send = torch.empty(
            (world, full_elements), dtype=torch.bfloat16, device=device
        )
        self.full_recv = torch.empty_like(self.full_send)
        self.full_output = torch.empty(
            world * full_elements, dtype=torch.bfloat16, device=device
        )

    def full(self, qkv: torch.Tensor) -> torch.Tensor:
        if self.batch != 1:
            raise ValueError("full QKV benchmark route currently requires batch=1")
        qkv = qkv.view(self.seq_local, self.qkv_width)
        if self.pack_backend == "triton":
            output_elements = self.full_send.numel()
            _pack_full_qkv_kernel[(triton.cdiv(output_elements, self.pack_block),)](
                qkv,
                self.full_send,
                numel=output_elements,
                seq_local=self.seq_local,
                qkv_width=self.qkv_width,
                q_width=self.q_width,
                kv_width=self.kv_width,
                local_q_heads=self.local_q_heads,
                local_kv_heads=self.local_kv_heads,
                head_dim=self.head_dim,
                BLOCK=self.pack_block,
                num_warps=self.pack_warps,
            )
        else:
            q, k, v = (
                qkv[:, : self.q_width].view(
                    self.seq_local, self.world, self.local_q_heads, self.head_dim
                ),
                qkv[:, self.q_width : self.q_width + self.kv_width].view(
                    self.seq_local, self.world, self.local_kv_heads, self.head_dim
                ),
                qkv[:, self.q_width + self.kv_width :].view(
                    self.seq_local, self.world, self.local_kv_heads, self.head_dim
                ),
            )
            peer_elements = self.full_send.shape[1]
            q_elements = self.seq_local * self.local_q_heads * self.head_dim
            kv_elements = self.seq_local * self.local_kv_heads * self.head_dim
            flat = self.full_send.view(-1)
            for peer in range(self.world):
                base = peer * peer_elements
                flat[base : base + q_elements].copy_(q[:, peer].reshape(-1))
                flat[base + q_elements : base + q_elements + kv_elements].copy_(
                    k[:, peer].reshape(-1)
                )
                flat[
                    base + q_elements + kv_elements : base + q_elements + 2 * kv_elements
                ].copy_(v[:, peer].reshape(-1))
        dist.all_to_all_single(self.full_recv, self.full_send)
        output_elements = self.full_output.numel()
        _unpack_full_qkv_kernel[(triton.cdiv(output_elements, self.pack_block),)](
            self.full_recv,
            self.full_output,
            numel=output_elements,
            seq_local=self.seq_local,
            world=self.world,
            local_q_heads=self.local_q_heads,
            local_kv_heads=self.local_kv_heads,
            head_dim=self.head_dim,
            BLOCK=self.pack_block,
            num_warps=self.pack_warps,
        )
        return self.full_output

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
        args.head_dim,
        args.warmup,
        args.iters,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("all shape and sampling arguments must be positive")
    if args.global_seq % world:
        raise ValueError("global sequence must be divisible by CP world size")
    if args.q_heads % world:
        raise ValueError("Q head count must be divisible by CP world size")
    if args.mode != "oproj_a2a_gemm":
        if args.kv_heads <= 0:
            raise ValueError("KV head count must be positive")
        if args.kv_heads % world:
            raise ValueError("KV head count must be divisible by CP world size")
        if args.q_heads % args.kv_heads:
            raise ValueError("Q heads must be divisible by KV heads")


def run_oproj(
    args: argparse.Namespace,
    rank: int,
    world: int,
    device: torch.device,
    devices: list[dict[str, object] | None],
) -> dict[str, object]:
    """TE/PyTorch BLAS + NCCL reference for inverse-A2A then O projection."""
    seq_local = args.global_seq // world
    local_heads = args.q_heads // world
    gemm_m = args.batch * seq_local
    gemm_k = args.q_heads * args.head_dim
    gemm_n = args.hidden
    generator = torch.Generator(device=device).manual_seed(2701 + rank)

    attention_output = torch.empty(
        (args.batch, args.global_seq, local_heads, args.head_dim),
        dtype=torch.bfloat16,
        device=device,
    ).uniform_(-0.125, 0.125, generator=generator)
    linear = te.Linear(
        gemm_k,
        gemm_n,
        bias=False,
        params_dtype=torch.bfloat16,
        device=device,
        name="ulysses_output_projection",
    ).eval()
    dist.broadcast(linear.weight, src=0)
    cp_stream = torch.cuda.Stream(device=device)
    chunk_ids = get_seq_chunk_ids_for_reordering_before_attn(world, device)

    def inverse_a2a() -> torch.Tensor:
        return flash_attn_a2a_communicate(
            attention_output,
            chunk_ids,
            1,
            world,
            dist.group.WORLD,
            cp_stream,
            False,
            qkv_format="bshd",
            a2a_input_names=["out"],
        )

    staging = inverse_a2a().reshape(gemm_m, gemm_k)
    cublas_output = torch.empty(
        (gemm_m, gemm_n), dtype=torch.bfloat16, device=device
    )

    def te_project() -> torch.Tensor:
        return linear(staging)

    cublaslt = cached_cublaslt_runner(
        args, staging, linear.weight, cublas_output
    )
    rank_cublaslt_plans: list[dict[str, object] | None] = [None] * world
    dist.all_gather_object(rank_cublaslt_plans, cublaslt.info)

    def cublaslt_project() -> torch.Tensor:
        return cublaslt(staging, linear.weight, cublas_output)

    def te_boundary() -> torch.Tensor:
        routed = inverse_a2a().reshape(gemm_m, gemm_k)
        return linear(routed)

    def cublaslt_boundary() -> torch.Tensor:
        routed = inverse_a2a().reshape(gemm_m, gemm_k)
        return cublaslt(routed, linear.weight, cublas_output)

    check: dict[str, int] = {}
    if args.check:
        peers = [torch.empty_like(attention_output) for _ in range(world)]
        dist.all_gather(peers, attention_output)
        causal_chunks = [2 * peer for peer in range(world)] + [
            2 * world - 2 * peer - 1 for peer in range(world)
        ]
        selected_chunks = causal_chunks[2 * rank : 2 * rank + 2]
        chunk_tokens = args.global_seq // (2 * world)
        expected_parts = [
            peer.view(
                args.batch,
                2 * world,
                chunk_tokens,
                local_heads,
                args.head_dim,
            )[:, selected_chunks]
            for peer in peers
        ]
        expected = torch.stack(expected_parts, dim=3).reshape(
            args.batch * seq_local, args.q_heads, args.head_dim
        )
        mismatch = torch.count_nonzero(
            staging.view(args.batch * seq_local, args.q_heads, args.head_dim)
            != expected
        ).to(torch.int64)
        dist.all_reduce(mismatch, op=dist.ReduceOp.SUM)
        check["inverse_a2a_reference_mismatches"] = int(mismatch.item())
        if mismatch.item():
            raise RuntimeError(f"O-projection route correctness failed: {check}")
        cublaslt_project()
        torch.cuda.synchronize(device)
        reference_output = torch.mm(staging, linear.weight.t())
        max_abs = (cublas_output.float() - reference_output.float()).abs().max()
        dist.all_reduce(max_abs, op=dist.ReduceOp.MAX)
        check["cublaslt_vs_torch_mm_max_abs"] = float(max_abs.item())
        if not torch.isfinite(max_abs):
            raise RuntimeError(f"cuBLASLt produced non-finite output: {check}")

    flops = 2 * gemm_m * gemm_n * gemm_k
    payload_bytes = attention_output.numel() * attention_output.element_size()
    metrics: dict[str, dict[str, object]] = {}

    def measure(
        name: str,
        fn: Callable[[], object],
        *,
        metric_flops: int | None = None,
        metric_payload_bytes: int | None = None,
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
                flops=metric_flops,
                world=world,
                payload_bytes=metric_payload_bytes,
            ),
        }

    if args.include_te:
        measure("te_oproj_gemm", te_project, metric_flops=flops)
    measure("cublaslt_oproj_gemm", cublaslt_project, metric_flops=flops)
    measure(
        "te_nccl_inverse_a2a",
        inverse_a2a,
        metric_payload_bytes=payload_bytes,
    )
    if args.include_te:
        measure("te_nccl_oproj_boundary", te_boundary, metric_flops=flops)
    measure("cublaslt_nccl_oproj_boundary", cublaslt_boundary, metric_flops=flops)

    result = {
        "mode": args.mode,
        "model_shape": {
            "batch": args.batch,
            "global_seq": args.global_seq,
            "seq_local": seq_local,
            "hidden": args.hidden,
            "q_heads": args.q_heads,
            "head_dim": args.head_dim,
        },
        "gemm_shape": {"m": gemm_m, "n": gemm_n, "k": gemm_k, "l": 1},
        "world_size": world,
        "warmup": args.warmup,
        "iterations": args.iters,
        "dtype": "bfloat16",
        "timing": "per-sample max-rank CUDA-event critical path",
        "scope": "TE Ulysses inverse A2A (including reorder) + output projection",
        "include_te": args.include_te,
        "cuda_graph": args.cuda_graph,
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
                "NCCL_IB_DISABLE",
            )
        },
        "devices": devices,
        "software": {
            "torch": torch.__version__,
            "transformer_engine": transformer_engine.__version__,
            "torch_cuda": torch.version.cuda,
            "nccl": ".".join(str(value) for value in torch.cuda.nccl.version()),
        },
        "cublaslt_plans": rank_cublaslt_plans,
        "implementations": {
            "te_oproj": "transformer_engine.pytorch.Linear",
            "cublaslt_oproj": "explicit cuBLASLt, 64 heuristic candidates, locally timed winner",
            "inverse_a2a": "TE flash_attn_a2a_communicate / NCCL",
        },
        "correctness": check,
        "derived": {},
        "results": {name: entry["stats"] for name, entry in metrics.items()},
        "samples_ms": {name: entry["samples_ms"] for name, entry in metrics.items()},
    }
    return result


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


def exact_full_qkv_route_check(
    qkv: torch.Tensor,
    actual: torch.Tensor,
    *,
    batch: int,
    seq_local: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    rank: int,
    world: int,
) -> int:
    peers = [torch.empty_like(qkv) for _ in range(world)]
    dist.all_gather(peers, qkv)
    q_local, kv_local = q_heads // world, kv_heads // world
    q_begin, kv_begin = rank * q_local, rank * kv_local
    q_width, kv_width = q_heads * head_dim, kv_heads * head_dim
    segments: list[torch.Tensor] = []
    for segment, local_heads, begin in (
        (0, q_local, q_begin),
        (1, kv_local, kv_begin),
        (2, kv_local, kv_begin),
    ):
        base = 0 if segment == 0 else q_width + (kv_width if segment == 2 else 0)
        global_heads = q_heads if segment == 0 else kv_heads
        parts = [
            peer.view(batch, seq_local, q_width + 2 * kv_width)[
                ..., base : base + global_heads * head_dim
            ]
            .view(batch, seq_local, global_heads, head_dim)[
                ..., begin : begin + local_heads, :
            ]
            for peer in peers
        ]
        segments.append(torch.cat(parts, dim=1).contiguous().view(-1))
    expected = torch.cat(segments)
    mismatch = torch.count_nonzero(actual.view(-1) != expected).to(torch.int64)
    dist.all_reduce(mismatch, op=dist.ReduceOp.SUM)
    return int(mismatch.item())


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
    if args.mode == "oproj_a2a_gemm":
        return run_oproj(args, rank, world, device, devices)
    seq_local = args.global_seq // world
    qkv_m = args.batch * seq_local
    qkv_n = (args.q_heads + 2 * args.kv_heads) * args.head_dim
    local_q_heads = args.q_heads // world
    generator = torch.Generator(device=device).manual_seed(1701 + rank)

    x = torch.empty(
        (qkv_m, args.hidden), dtype=torch.bfloat16, device=device
    ).uniform_(-0.125, 0.125, generator=generator)
    qkv_cublas = torch.empty((qkv_m, qkv_n), dtype=torch.bfloat16, device=device)
    linear = None
    cublaslt = None
    rank_cublaslt_plans: list[dict[str, object] | None] = []
    if args.metric_profile != "route":
        linear = te.Linear(
            args.hidden,
            qkv_n,
            bias=False,
            params_dtype=torch.bfloat16,
            device=device,
            name="ulysses_qkv_projection",
        ).eval()
        dist.broadcast(linear.weight, src=0)
        cublaslt = cached_cublaslt_runner(
            args, x, linear.weight, qkv_cublas
        )
        rank_cublaslt_plans = [None] * world
        dist.all_gather_object(rank_cublaslt_plans, cublaslt.info)
    else:
        qkv_cublas.uniform_(-0.125, 0.125, generator=generator)
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

    def te_qkv_a2a(qkv: torch.Tensor, sequence_chunks: torch.Tensor):
        q, k, v = split_qkv(qkv)
        return flash_attn_a2a_communicate(
            [q, k, v],
            sequence_chunks,
            1,
            world,
            dist.group.WORLD,
            cp_stream,
            True,
            qkv_format="bshd",
            a2a_input_names=["q", "k", "v"],
        )

    def te_source_qkv_a2a(qkv: torch.Tensor):
        return te_qkv_a2a(qkv, chunk_ids)

    def cublas_project() -> torch.Tensor:
        assert linear is not None
        return torch.mm(x, linear.weight.t(), out=qkv_cublas)

    def cublaslt_project() -> torch.Tensor:
        assert linear is not None and cublaslt is not None
        return cublaslt(x, linear.weight, qkv_cublas)

    def te_project() -> torch.Tensor:
        assert linear is not None
        return linear(x)

    if args.metric_profile != "route":
        cublas_project()
    torch.cuda.synchronize(device)
    check = (
        exact_route_check(routes, qkv_cublas, rank, world)
        if args.check and args.include_qk
        else {}
    )
    if args.check:
        check["packed_qkv_mismatches"] = exact_full_qkv_route_check(
            qkv_cublas,
            routes.full(qkv_cublas),
            batch=args.batch,
            seq_local=seq_local,
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            rank=rank,
            world=world,
        )
        if args.include_source:
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

    if args.mode == "qkv_gemm_a2a":
        if args.metric_profile == "full":
            if args.include_te:
                measure("te_qkv_gemm", te_project, flops=qkv_flops)
            measure("cublas_qkv_gemm", cublas_project, flops=qkv_flops)
            measure("cublaslt_qkv_gemm", cublaslt_project, flops=qkv_flops)
        if args.metric_profile in ("route", "full"):
            measure(
                "packed_qkv_a2a",
                lambda: routes.full(qkv_cublas),
                payload_bytes=qk_payload + v_payload,
            )

        if args.include_qk and args.metric_profile == "full":
            measure(
                "nccl_qk_a2a",
                lambda: routes.qk(qkv_cublas),
                payload_bytes=qk_payload,
            )

            def te_qkv_qk() -> torch.Tensor:
                return routes.qk(te_project())

            def cublas_qkv_qk() -> torch.Tensor:
                return routes.qk(cublas_project())

            def cublaslt_qkv_qk() -> torch.Tensor:
                return routes.qk(cublaslt_project())

            if args.include_te:
                measure("te_qkv_gemm_qk_a2a", te_qkv_qk, flops=qkv_flops)
            measure("cublas_qkv_gemm_qk_a2a", cublas_qkv_qk, flops=qkv_flops)
            measure(
                "cublaslt_qkv_gemm_qk_a2a", cublaslt_qkv_qk, flops=qkv_flops
            )

        if args.include_source and args.metric_profile == "full":
            measure(
                "te_source_qkv_a2a",
                lambda: te_source_qkv_a2a(qkv_cublas),
                payload_bytes=qk_payload + v_payload,
            )

        def te_packed_qkv_gemm_a2a():
            return routes.full(te_project())

        def cublas_packed_qkv_gemm_a2a():
            return routes.full(cublas_project())

        def cublaslt_packed_qkv_gemm_a2a():
            return routes.full(cublaslt_project())

        def te_source_qkv_gemm_a2a():
            return te_source_qkv_a2a(te_project())

        def cublas_source_qkv_gemm_a2a():
            return te_source_qkv_a2a(cublas_project())

        def cublaslt_source_qkv_gemm_a2a():
            return te_source_qkv_a2a(cublaslt_project())

        if args.metric_profile in ("boundary", "full"):
            measure(
                "te_packed_qkv_gemm_a2a",
                te_packed_qkv_gemm_a2a,
                flops=qkv_flops,
            )
            measure(
                "cublas_packed_qkv_gemm_a2a",
                cublas_packed_qkv_gemm_a2a,
                flops=qkv_flops,
            )
            measure(
                "cublaslt_packed_qkv_gemm_a2a",
                cublaslt_packed_qkv_gemm_a2a,
                flops=qkv_flops,
            )
        if args.include_source and args.metric_profile == "full":
            measure(
                "te_source_qkv_gemm_a2a", te_source_qkv_gemm_a2a, flops=qkv_flops
            )
            measure(
                "cublas_source_qkv_gemm_a2a",
                cublas_source_qkv_gemm_a2a,
                flops=qkv_flops,
            )
            measure(
                "cublaslt_source_qkv_gemm_a2a",
                cublaslt_source_qkv_gemm_a2a,
                flops=qkv_flops,
            )

    derived: dict[str, object] = {}
    if (
        args.mode == "qkv_gemm_a2a"
        and args.include_qk
        and args.metric_profile == "full"
    ):
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
        derived["cublaslt_qkv_qk_overlap"] = overlap_summary(
            metrics["cublaslt_qkv_gemm"]["stats"],
            metrics["nccl_qk_a2a"]["stats"],
            metrics["cublaslt_qkv_gemm_qk_a2a"]["stats"],
        )
    return {
        "benchmark_schema": "qkv_full_qkv_packed_v1",
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
        },
        "world_size": world,
        "warmup": args.warmup,
        "iterations": args.iters,
        "dtype": "bfloat16",
        "timing": "per-sample max-rank CUDA-event critical path",
        "scope": (
            "QKV projection followed by full Q/K/V A2A; packed metrics "
            "match the fused segment-major, rank-major output, te_source metrics use "
            "TE causal-load-balanced sequence order"
        ),
        "include_te": args.include_te,
        "include_source": args.include_source,
        "include_qk": args.include_qk,
        "metric_profile": args.metric_profile,
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
        "cublaslt_plans": rank_cublaslt_plans,
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
            "cublaslt_qkv": "explicit locally autotuned cuBLASLt",
            "packed_a2a": "Triton pack + one NCCL A2A + Triton unpack",
            "te_source_a2a": (
                "TE flash_attn_a2a_communicate with causal load balancing"
            ),
        },
        "correctness": check,
        "derived": derived,
        "results": {name: entry["stats"] for name, entry in metrics.items()},
        "samples_ms": {name: entry["samples_ms"] for name, entry in metrics.items()},
    }


def print_and_write_result(
    args: argparse.Namespace,
    result: dict[str, object],
    rank: int,
) -> None:
    if rank != 0:
        return
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
        temporary = args.json_out.with_suffix(args.json_out.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(args.json_out)


def initialize_process_group(
    args: argparse.Namespace,
    device: torch.device,
    *,
    init_method: str | None = None,
) -> None:
    process_group_options = dist.ProcessGroupNCCL.Options()
    process_group_options.is_high_priority_stream = args.nccl_high_priority
    init_kwargs: dict[str, object] = {}
    if init_method is not None:
        init_kwargs.update({
            "init_method": init_method,
            "rank": int(os.environ["RANK"]),
            "world_size": int(os.environ["WORLD_SIZE"]),
        })
    dist.init_process_group(
        "nccl", device_id=device, pg_options=process_group_options, **init_kwargs
    )


def run_initialized(args: argparse.Namespace, device: torch.device) -> None:
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    result = run(args, rank, world, device)
    print_and_write_result(args, result, rank)
    torch.cuda.synchronize(device)
    del result
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
    if not args.matrix_manifest:
        run_one_process_group(args, device)
        close_cublaslt_runners()
        return

    entries = json.loads(args.matrix_manifest.read_text())
    for high_priority in (True, False):
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
    close_cublaslt_runners()


if __name__ == "__main__":
    main()
