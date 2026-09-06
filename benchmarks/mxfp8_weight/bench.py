#!/usr/bin/env python3
"""Offline MXFP8 weights -> timed BF16 dequant + existing Ulysses boundaries.

No production fused operator is introduced. TEUB uses the repository's public
TE P2P adaptation, not an invented native AllToAll TE API. Both backends use
the existing autotuned cuBLASLt BF16 runner. Inverse A2A performs one full-K
GEMM, never a chain of BF16 partial-sum GEMMs with different rounding.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from matrix import ROOT, full_matrix


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list-only", action="store_true")
    p.add_argument("--direction", choices=("both", "gemm_a2a", "a2a_gemm"), default="both")
    p.add_argument("--model", default="production_qwen_dense")
    p.add_argument("--seqs", default="4096")
    p.add_argument("--full", action="store_true")
    p.add_argument("--case-ids", type=Path, help="JSON list of exact legacy case IDs")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--sweep-iters", type=int, default=12)
    p.add_argument("--sweep-warmup", type=int, default=3)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--tune-iters", type=int, default=6)
    p.add_argument("--tune-warmup", type=int, default=2)
    p.add_argument("--ub-sms", default="4,8,16")
    p.add_argument("--launches", default="eager,graph")
    p.add_argument("--backends", default="cublaslt_nccl,teub")
    p.add_argument("--library", type=Path, required=False,
                   default=ROOT / "build-mxfp8-bench/libfuse_cublaslt_runner.so")
    p.add_argument("--output", type=Path, default=ROOT / "results/mxfp8_weight/first_bench.json")
    return p.parse_args()


args = arguments()
matrix = full_matrix()
if args.list_only:
    print(json.dumps(dict(settings=len(matrix), rows=matrix), indent=2))
    sys.exit(0)
if args.warmup < 1 or args.iterations < 1 or args.tune_iters < 1 or args.tune_warmup < 1:
    raise ValueError("Warmup, sampling and tuning counts must be positive")
if not set(args.backends.split(",")) <= {"cublaslt_nccl", "teub"}:
    raise ValueError("Unknown backend")
if not set(args.launches.split(",")) <= {"eager", "graph"}:
    raise ValueError("Unknown execution mode")

# Heavy imports occur once per rank, not once per matrix setting.
import numpy as np
import torch
import torch.distributed as dist
import transformer_engine
import transformer_engine_torch as tex
import triton
import triton.language as tl

from cublaslt import CublasLtRunner


@triton.jit
def dequant_kernel(q, scales, out, E: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(q + i, i < E, 0.0).to(tl.float32)
    exponent = tl.load(scales + i // 32, i < E, 127).to(tl.int32) - 127
    tl.store(out + i, x * tl.exp2(exponent.to(tl.float32)), i < E)


@triton.jit
def pack_qkv(x, out, M: tl.constexpr, Q: tl.constexpr, KV: tl.constexpr,
             P: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    S: tl.constexpr = (Q + 2 * KV) // P
    dest = i // (M * S)
    row = i // S % M
    col = i % S
    src_col = tl.where(col < Q // P, dest * (Q // P) + col,
        tl.where(col < (Q + KV) // P,
                 Q + dest * (KV // P) + col - Q // P,
                 Q + KV + dest * (KV // P) + col - (Q + KV) // P))
    mask = i < M * (Q + 2 * KV)
    tl.store(out + i, tl.load(x + row * (Q + 2 * KV) + src_col, mask, 0), mask)


@triton.jit
def unpack_qkv(send, recv, out, M: tl.constexpr, Q: tl.constexpr,
               KV: tl.constexpr, P: tl.constexpr, R: tl.constexpr,
               UB: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    S: tl.constexpr = (Q + 2 * KV) // P
    isq = i < M * Q
    isk = (i >= M * Q) & (i < M * (Q + KV))
    rel = tl.where(isq, i, tl.where(isk, i - M * Q, i - M * (Q + KV)))
    width = tl.where(isq, Q // P, KV // P)
    base = tl.where(isq, 0, tl.where(isk, Q // P, (Q + KV) // P))
    peer = rel // (M * width)
    src = peer * M * S + (rel // width % M) * S + base + rel % width
    mask = i < M * (Q + 2 * KV)
    if UB:
        a = tl.load(send + src, mask & (peer == R), 0)
        b = tl.load(recv + src, mask & (peer != R), 0)
        val = a + b
    else:
        val = tl.load(recv + src, mask, 0)
    tl.store(out + i, val, mask)


@triton.jit
def pack_inverse(x, out, M: tl.constexpr, KL: tl.constexpr,
                 P: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    dest = i // (M * KL)
    row = i // KL % M
    chunk = tl.where(row < M // 2, 2 * dest, 2 * P - 2 * dest - 1)
    srcrow = chunk * (M // 2) + row % (M // 2)
    mask = i < P * M * KL
    tl.store(out + i, tl.load(x + srcrow * KL + i % KL, mask, 0), mask)


@triton.jit
def unpack_inverse(send, recv, out, M: tl.constexpr, K: tl.constexpr,
                   P: tl.constexpr, R: tl.constexpr, UB: tl.constexpr,
                   BLOCK: tl.constexpr, FIRST: tl.constexpr = 0,
                   ROWS: tl.constexpr = 0):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    i = i + FIRST * K
    peer = i % K // (K // P)
    src = peer * M * (K // P) + i // K * (K // P) + i % (K // P)
    mask = i < (FIRST + ROWS) * K if ROWS else i < M * K
    if UB:
        val = tl.load(send + src, mask & (peer == R), 0) + tl.load(recv + src, mask & (peer != R), 0)
    else:
        val = tl.load(recv + src, mask, 0)
    tl.store(out + i, val, mask)


@triton.jit
def error_partials(a, b, partial, E: tl.constexpr, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    i = block.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(a + i, i < E, 0).to(tl.float32)
    y = tl.load(b + i, i < E, 0).to(tl.float32)
    d = x - y
    tl.store(partial + block * 3, tl.max(tl.abs(d), 0))
    tl.store(partial + block * 3 + 1, tl.sum(d * d, 0))
    tl.store(partial + block * 3 + 2, tl.sum(y * y, 0))


def quantize_offline(weight):
    """FP32 reference quantizer, RNE E4M3, ceil power-of-two no-clipping scale.

    Offline chunking bounds temporary memory for the largest published weights.
    Zero blocks use E8M0 byte 0. Nonfinite inputs are outside this benchmark.
    """
    q = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    scales = torch.empty((weight.shape[0], weight.shape[1] // 32),
                         dtype=torch.uint8, device=weight.device)
    for first in range(0, weight.shape[0], 128):
        x = weight[first:first + 128].float().reshape(-1, weight.shape[1] // 32, 32)
        amax = x.abs().amax(-1)
        exponent = torch.ceil(torch.log2(amax / 448)).clamp(-127, 127)
        s = torch.exp2(exponent)
        v = (x / s.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        q[first:first + 128].copy_(v.reshape(-1, weight.shape[1]))
        scales[first:first + 128].copy_((exponent + 127).to(torch.uint8))
    return q, scales


def reference_dequant(q, s):
    out = torch.empty_like(q, dtype=torch.bfloat16)
    for first in range(0, q.shape[0], 128):
        block = q[first:first + 128].float().reshape(-1, q.shape[1] // 32, 32)
        scale = torch.exp2(s[first:first + 128].float() - 127)
        out[first:first + 128].copy_((block * scale.unsqueeze(-1)).reshape(-1, q.shape[1]))
    return out


class Userbuffers:
    def __init__(self, shape, helper, device, rank, world, sms, streams=3,
                 use_ce=False, push=True, reverse=False):
        self.rank, self.world, self.device = rank, world, device
        self.ub = tex.CommOverlapP2P(
            [2 * world * shape[1], shape[2]], torch.bfloat16, helper, world,
            tex.CommOverlapType.AG, num_max_streams=streams, comm_cga_size=1,
            gemm_priority=0, comm_priority=-1, num_comm_sm=sms,
            set_sm_margin=False, atomic_gemm=False, use_ce=use_ce, aggregate=False)
        self.ub.configure_userbuffers_p2p(sms, use_ce, push)
        storage = self.ub.get_buffer(False, [2, *shape])
        self.send, self.recv = storage[0], storage[1]
        self.bytes = shape[1] * shape[2] * 2
        self.streams = [torch.cuda.ExternalStream(
            self.ub.get_userbuffers_send_stream(i).stream_id, device=device)
            for i in range(min(streams, world - 1))]
        self.rx = torch.cuda.ExternalStream(self.ub.get_communication_stream()[1].stream_id,
                                            device=device)
        self.fork = torch.cuda.Event()
        self.ready = [torch.cuda.Event() for _ in range(world)]
        self.ends = [torch.cuda.Event() for _ in self.streams]
        self.rxend = torch.cuda.Event()
        self.steps = list(range(1, world))
        if reverse:
            self.steps.reverse()
        self.peers = [(rank + i) % world for i in self.steps]

    def begin(self):
        self.fork.record()
        self.rx.wait_event(self.fork)

    def send_peer(self, peer, i, offset=0, count=None):
        self.ready[peer].record()
        sid = i % len(self.streams)
        self.streams[sid].wait_event(self.ready[peer])
        self.ub.userbuffers_p2p_send(peer * self.bytes + offset,
                                     (self.world + self.rank) * self.bytes + offset,
                                     self.bytes if count is None else count, peer, sid)

    def receive(self, offset=0, count=None):
        for step in self.steps:
            peer = (self.rank - step) % self.world
            self.ub.userbuffers_p2p_recv(self.rank * self.bytes + offset,
                (self.world + peer) * self.bytes + offset,
                self.bytes if count is None else count, peer)

    def receive_and_join(self):
        self.receive()
        self.join()

    def join(self):
        self.rxend.record(self.rx)
        main = torch.cuda.current_stream(self.device)
        main.wait_event(self.rxend)
        for stream, event in zip(self.streams, self.ends):
            event.record(stream)
            main.wait_event(event)


def measure(fn, launch, warmup, iters, cpu_group, poison):
    """One operation/replay. No allocation or rank reduction in timed region.

    CPU barriers keep independent samples aligned without injecting GPU NCCL
    barriers into the stream. Gather duration vectors once, then max by sample.
    Events are reused; graph instantiation/upload is paid during warmup.
    """
    for _ in range(max(warmup, 2)):
        fn()
    torch.cuda.synchronize()
    dist.barrier(group=cpu_group)
    graph = None
    if launch == "graph":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        call = graph.replay
        for _ in range(max(warmup, 2)):
            call()
        torch.cuda.synchronize()
    else:
        call = fn
    # Poison after capture and warmup: a graph that accidentally reuses a
    # precomputed BF16 weight or communication result must fail correctness.
    poison()
    torch.cuda.synchronize()
    start, stop = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    # Materialize CUDA event handles outside samples.
    start.record()
    stop.record()
    stop.synchronize()
    samples = []
    for _ in range(iters):
        dist.barrier(group=cpu_group)
        start.record()
        call()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop) * 1000)
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, samples, group=cpu_group)
    values = np.max(np.asarray(gathered), axis=0)
    return dict(p50_us=float(np.median(values)), p95_us=float(np.percentile(values, 95)),
                mean_us=float(values.mean()), samples_us=values.tolist(), rank_samples_us=gathered)


def check_error(actual, expected, group):
    count = triton.cdiv(actual.numel(), 4096)
    partial = torch.empty((count, 3), device=actual.device, dtype=torch.float32)
    error_partials[(count,)](actual, expected, partial, actual.numel(), 4096)
    metric = torch.stack((partial[:, 0].amax(), (partial[:, 1].sum() /
                         partial[:, 2].sum().clamp_min(1e-30)).sqrt()))
    # NCCL MAX must not be used to propagate NaN: a finite value from another
    # rank may win. Reduce an explicit failure flag before reducing metrics.
    invalid = (~torch.isfinite(metric).all()).to(torch.int32)
    dist.all_reduce(invalid, op=dist.ReduceOp.MAX)
    dist.all_reduce(metric, op=dist.ReduceOp.MAX)
    result = dict(max_abs=metric[0].item(), relative_rmse=metric[1].item())
    if invalid.item() or result["relative_rmse"] > 0.005:
        raise RuntimeError(f"BF16 GEMM reference failed: {result}")
    return result


def configurations(case, backend):
    if backend == "cublaslt_nccl":
        return [dict(schedule="full", sms=0, streams=3, pack_block=1024,
                     pack_warps=4, use_ce=False, push=True, reverse=False,
                     local_first=False, math_sms=0, chunks=1)]
    base = dict(schedule="full", streams=3, pack_block=1024, pack_warps=4,
                use_ce=False, push=True, reverse=False, local_first=False,
                math_sms=0, chunks=1)
    configs = [dict(base, sms=int(s)) for s in args.ub_sms.split(",")]
    if case["direction"] == "gemm_a2a":
        configs += [dict(base, schedule="peer_gemm", sms=int(s),
                         math_sms=132-int(s), streams=1) for s in args.ub_sms.split(",")]
    else:
        configs += [dict(base, schedule="row_pipeline", sms=8, chunks=c)
                    for c in (2, 4) if case["m"] % c == 0]
    # Re-test the old shape-specific winner, rather than assuming its policy
    # still wins after adding dequantization. No historical timings are reused.
    path = ROOT / ("results/QKVproj-a2a/te_userbuffers_shape_bench/summary.json"
                   if case["direction"] == "gemm_a2a" else
                   "results/a2a-Oproj/te_userbuffers_mixed_shape_bench/summary.json")
    if path.exists():
        for old in json.loads(path.read_text()):
            if (old["model"], old["global_seq"], old["cp"]) == (case["model"], case["global_seq"], case["cp"]):
                config = dict(base, sms=old["comm_sm"], streams=old["streams"],
                              use_ce=old["use_ce"], push=old["push"],
                              pack_block=old["pack_block"], pack_warps=old["pack_warps"],
                              reverse=old["reverse"], local_first=old.get("local_first", False))
                if case["direction"] == "gemm_a2a":
                    config.update(schedule="peer_gemm", math_sms=old["math_sm"])
                configs.append(config)
                break
    return list({json.dumps(c, sort_keys=True): c for c in configs}.values())


def run_case(case, device, helper, cpu_group):
    rank, world = dist.get_rank(), dist.get_world_size()
    m, n, k = case["m"], case["n"], case["k"]
    qwidth, kvwidth = case["q_heads"] * case["head_dim"], case["kv_heads"] * case["head_dim"]
    forward = case["direction"] == "gemm_a2a"
    slab = n // world if forward else k // world
    generator = torch.Generator(device=device).manual_seed(2701 + rank)
    source = torch.empty((m, k) if forward else (m * world, slab),
                         dtype=torch.bfloat16, device=device).uniform_(-0.125, 0.125, generator=generator)
    weight = torch.empty((n, k), dtype=torch.bfloat16, device=device).uniform_(-0.02, 0.02, generator=generator)
    dist.broadcast(weight, 0)
    q, s = quantize_offline(weight)
    reference_w = reference_dequant(q, s)
    weight_error = check_error_quant(weight, reference_w)
    del weight
    workspace = torch.empty_like(reference_w)

    def dq():
        dequant_kernel[(triton.cdiv(n * k, 2048),)](q, s, workspace, n * k, 2048)

    dq()
    if not torch.equal(workspace, reference_w):
        raise RuntimeError("Triton dequant differs from independent PyTorch BF16 reference")
    peers = [torch.empty_like(source) for _ in range(world)]
    dist.all_gather(peers, source)
    if forward:
        local_w = torch.cat([reference_w[offset + rank * width:offset + (rank + 1) * width]
                             for offset, width in ((0, qwidth // world),
                                 (qwidth, kvwidth // world), (qwidth + kvwidth, kvwidth // world))])
        projections = [torch.mm(x, local_w.t()) for x in peers]
        expected = torch.cat([z[:, offset:offset + width].reshape(-1)
                              for offset, width in ((0, qwidth // world),
                                  (qwidth // world, kvwidth // world),
                                  ((qwidth + kvwidth) // world, kvwidth // world))
                              for z in projections])
        del projections, local_w
    else:
        expected_a = torch.cat([torch.cat((x[rank * m:rank * m + m // 2],
                        x[(2 * world - 2 * rank - 1) * (m // 2):
                          (2 * world - 2 * rank) * (m // 2)]), dim=0) for x in peers], dim=1)
        expected = torch.mm(expected_a, reference_w.t())
    del peers, reference_w
    projected = torch.empty((m, n), dtype=torch.bfloat16, device=device)
    staging = torch.empty((m, k), dtype=torch.bfloat16, device=device) if not forward else source
    if not forward:
        staging.copy_(expected_a)
    output = torch.empty_like(expected)
    plans = {}

    def get_plan(a, b, d, math_sms):
        key = (tuple(a.shape), tuple(b.shape), math_sms)
        if key not in plans:
            plans[key] = CublasLtRunner(args.library, a, b, d,
                tune_warmup=args.tune_warmup, tune_iters=args.tune_iters,
                workspace_mib=64, sm_count_target=math_sms)
        return plans[key]

    # Offline destination-major packing follows the old TEUB QKV baseline.
    # Scale blocks remain along K, so this is only a row permutation.
    packed_q = packed_s = packed_w = None
    if forward and "teub" in args.backends.split(","):
        order = torch.cat([torch.cat([torch.arange(offset + p * width,
                    offset + (p + 1) * width, device=device) for offset, width in
                    ((0, qwidth // world), (qwidth, kvwidth // world),
                     (qwidth + kvwidth, kvwidth // world))]) for p in range(world)])
        packed_q = q.view(torch.uint8).index_select(0, order).view(torch.float8_e4m3fn)
        packed_s = s.index_select(0, order)
        packed_w = workspace.index_select(0, order)
        del order

    def candidate(backend, config, launches, warmup, iters, phase):
        sms, schedule = config["sms"], config["schedule"]
        block, warps = config["pack_block"], config["pack_warps"]
        ub = Userbuffers((world, m, slab), helper, device, rank, world, sms,
                         streams=config["streams"], use_ce=config["use_ce"],
                         push=config["push"], reverse=config["reverse"]) if backend == "teub" else None
        send = ub.send if ub else torch.empty((world, m, slab), dtype=torch.bfloat16, device=device)
        recv = ub.recv if ub else torch.empty_like(send)
        # Restore tuning inputs after another candidate's poison test.
        dq()
        if not forward:
            staging.copy_(expected_a)
        if schedule == "peer_gemm":
            dequant_kernel[(triton.cdiv(n*k, 2048),)](packed_q, packed_s, packed_w, n*k, 2048)
            weights = list(packed_w.view(world, slab, k).unbind(0))
            outputs = list(send.unbind(0))
            plan = get_plan(source, weights[0], outputs[0], config["math_sms"])
            order = ([rank] + ub.peers) if config["local_first"] else (ub.peers + [rank])
        elif schedule == "row_pipeline":
            rows = m // config["chunks"]
            inputs = list(staging.split(rows, dim=0))
            outputs = list(output.split(rows, dim=0))
            ready = [torch.cuda.Event() for _ in inputs]
            plan = get_plan(inputs[0], workspace, outputs[0], config["math_sms"])
        else:
            plan = get_plan(staging, workspace, projected, config["math_sms"])

        def exchange():
            if ub:
                ub.begin()
                for i, peer in enumerate(ub.peers):
                    ub.send_peer(peer, i)
                ub.receive_and_join()
            else:
                dist.all_to_all_single(recv, send)

        def boundary():
            if schedule == "peer_gemm":
                ub.begin()
                dequant_kernel[(triton.cdiv(n*k, 2048),)](packed_q, packed_s, packed_w, n*k, 2048)
                index = 0
                for dest in order:
                    plan(source, weights[dest], outputs[dest])
                    if dest != rank:
                        ub.send_peer(dest, index)
                        index += 1
                ub.receive_and_join()
                unpack_qkv[(triton.cdiv(m*n, block),)](send, recv, output,
                            m, qwidth, kvwidth, world, rank, True, block, num_warps=warps)
            elif schedule == "row_pipeline":
                pack_inverse[(triton.cdiv(m*k, block),)](source, send, m, slab, world, block, num_warps=warps)
                ub.begin()
                # All sends are submitted before receives, preserving the TEUB
                # handshake. Receives for later rows overlap earlier GEMMs.
                for chunk in range(config["chunks"]):
                    for i, peer in enumerate(ub.peers):
                        ub.send_peer(peer, i, chunk * rows * slab * 2, rows * slab * 2)
                dq()
                for chunk in range(config["chunks"]):
                    ub.receive(chunk * rows * slab * 2, rows * slab * 2)
                    ready[chunk].record(ub.rx)
                    torch.cuda.current_stream(device).wait_event(ready[chunk])
                    unpack_inverse[(triton.cdiv(rows*k, block),)](send, recv, staging,
                        m, k, world, rank, True, block, chunk * rows, rows, num_warps=warps)
                    plan(inputs[chunk], workspace, outputs[chunk])
                ub.join()
            elif forward:
                dq()
                plan(source, workspace, projected)
                pack_qkv[(triton.cdiv(m*n, block),)](projected, send, m, qwidth, kvwidth, world, block, num_warps=warps)
                exchange()
                unpack_qkv[(triton.cdiv(m*n, block),)](send, recv, output,
                            m, qwidth, kvwidth, world, rank, ub is not None, block, num_warps=warps)
            else:
                pack_inverse[(triton.cdiv(m*k, block),)](source, send, m, slab, world, block, num_warps=warps)
                exchange()
                unpack_inverse[(triton.cdiv(m*k, block),)](send, recv, staging,
                            m, k, world, rank, ub is not None, block, num_warps=warps)
                dq()
                plan(staging, workspace, output)

        def poison():
            for tensor in (workspace, output, projected, send, recv):
                tensor.fill_(float("nan"))
            if packed_w is not None:
                packed_w.fill_(float("nan"))
            if not forward:
                staging.fill_(float("nan"))

        poison()
        # A fast peer must not send into a buffer whose owner is still
        # poisoning it. This synchronization is validation-only, never timed.
        torch.cuda.synchronize()
        dist.barrier(group=cpu_group)
        boundary()
        torch.cuda.synchronize()
        error = check_error(output, expected, cpu_group)
        if not forward:
            mismatches = (staging != expected_a).sum()
            dist.all_reduce(mismatches)
            if mismatches.item():
                route = check_error(staging, expected_a, cpu_group)
                raise RuntimeError(f"Inverse route not byte-exact: {config}, {route}, "
                                   f"global_mismatches={mismatches.item()}")
        records = []
        for launch in launches:
            result = measure(boundary, launch, warmup, iters, cpu_group, poison)
            result["correctness_after_samples"] = check_error(output, expected, cpu_group)
            result.update(backend=backend, ub_comm_sms=sms, config=config, launch=launch,
                          phase=phase, warmup=warmup, iterations=iters,
                          correctness=error, gemm_plan=plan.info,
                          tflops_per_gpu=2*m*n*k/result["p50_us"]/1e6)
            records.append(result)
            if rank == 0:
                print(f'{case["id"]} {phase} {backend} {schedule} SM{sms} {launch}: '
                      f'{result["p50_us"]:.3f}/{result["p95_us"]:.3f} us', flush=True)
        torch.cuda.synchronize()
        dist.barrier(group=cpu_group)
        return records

    records, sweep = [], []
    for backend in args.backends.split(","):
        configs = configurations(case, backend)
        if len(configs) == 1:
            records += candidate(backend, configs[0], args.launches.split(","),
                                 args.warmup, args.iterations, "formal")
            continue
        for config in configs:
            sweep += candidate(backend, config, args.launches.split(","),
                               args.sweep_warmup, args.sweep_iters, "sweep")
        winners = {}
        for launch in args.launches.split(","):
            winner = min((r for r in sweep if r["backend"] == backend and r["launch"] == launch),
                         key=lambda r: r["p50_us"])
            key = json.dumps(winner["config"], sort_keys=True)
            winners.setdefault(key, []).append(launch)
        for key, launches in winners.items():
            records += candidate(backend, json.loads(key), launches, args.warmup, args.iterations, "formal")
    for plan in plans.values():
        plan.close()
    return dict(case=case, weight_quantization_error=weight_error,
                weight_payload_bytes=n * k, weight_scale_bytes=n * k // 32,
                bf16_workspace_bytes=n * k * 2, records=records, sweep_records=sweep,
                best_tested=[min([r for r in records if r["backend"] == b and r["launch"] == l],
                                key=lambda r: r["p50_us"])
                             for b in args.backends.split(",") for l in args.launches.split(",")])


def check_error_quant(original, recovered):
    # Quantization quality is separate from the implementation correctness gate.
    numerator = denominator = maximum = 0.0
    for first in range(0, original.shape[0], 128):
        a, b = original[first:first + 128].float(), recovered[first:first + 128].float()
        diff = a - b
        numerator += diff.square().sum().item()
        denominator += a.square().sum().item()
        maximum = max(maximum, diff.abs().max().item())
    return dict(max_abs=maximum, relative_rmse=(numerator / max(denominator, 1e-30)) ** 0.5)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.inference_mode()
def main():
    if not args.library.is_file():
        raise FileNotFoundError(f"Build csrc/baselines/cublaslt_runner.cu first: {args.library}")
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=device)
    cpu_group = dist.new_group(backend="gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    helper = tex.CommOverlapHelper(dist.group.WORLD, dist.group.WORLD)
    props = torch.cuda.get_device_properties(device)
    identity = dict(rank=rank, runtime_name=props.name, cc=[props.major, props.minor],
                    sms=props.multi_processor_count, memory_bytes=props.total_memory,
                    uuid=str(getattr(props, "uuid", "unknown")))
    devices = [None] * world
    dist.all_gather_object(devices, identity, group=cpu_group)
    selected = [c for c in matrix if c["cp"] == world and
                (args.direction == "both" or c["direction"] == args.direction) and
                (args.full or (c["model"] == args.model and c["global_seq"] in
                 [int(x) for x in args.seqs.split(",")]))]
    if args.case_ids:
        requested = set(json.loads(args.case_ids.read_text()))
        known = {c["id"] for c in matrix if c["cp"] == world}
        if not requested <= known:
            raise ValueError(f"Unknown/wrong-CP requested cases: {requested - known}")
        selected = [c for c in matrix if c["id"] in requested]
    if not selected:
        raise ValueError("No published cases match requested selection")
    result = dict(schema="fuse-mxfp8-weight-baseline-v2", devices=devices,
                  semantic="offline_mxfp8_weight_dequant_bf16_gemm_bf16_a2a_bf16_output",
                  weight_block_size=32, quantization="ceil_pow2_amax448_e4m3_rne_e8m0",
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
                  tune_warmup=args.tune_warmup, tune_iters=args.tune_iters,
                  sweep_warmup=args.sweep_warmup, sweep_iters=args.sweep_iters,
                  nccl_environment={k: v for k, v in os.environ.items() if k.startswith("NCCL_")},
                  timed="dequant+gemm+pack+communication+unpack; one operation per replay",
                  excluded="offline weight quantization, allocations, JIT, tuning, graph setup, validation",
                  timing="preallocated CUDA events; per-sample CPU barrier; batch sample-wise rank MAX",
                  ub_implementation="repository-style TE Userbuffers P2P adaptation + cuBLASLt",
                  limitation="best independently retested winner within explicit finite candidate set; not exhaustive global optimum",
                  warmup=args.warmup, iterations=args.iterations,
                  torch=torch.__version__, cuda=torch.version.cuda,
                  transformer_engine=transformer_engine.__version__, triton=triton.__version__,
                  git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  sources={str(p): sha(p) for p in [Path(__file__), Path(__file__).with_name("matrix.py"),
                      Path(__file__).with_name("cublaslt.py"),
                      ROOT / "benchmarks/QKVproj+a2a/qkv_shape_bench.py",
                      ROOT / "benchmarks/a2a+Oproj/oproj_shape_bench.py",
                      ROOT / "results/QKVproj-a2a/te_userbuffers_shape_bench/summary.json",
                      ROOT / "results/a2a-Oproj/te_userbuffers_mixed_shape_bench/summary.json",
                      ROOT / "csrc/baselines/cublaslt_runner.cu", args.library]},
                  protocol={k: str(v) for k, v in vars(args).items() if k not in ("resume", "output")},
                  selected_case_count=len(selected), matrix_setting_count=len(matrix), cases=[])
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text())
        for field in ("schema", "sources", "protocol", "devices", "nccl_environment", "cuda_visible_devices"):
            if previous[field] != result[field]:
                raise RuntimeError(f"Cannot resume: {field} changed")
        result = previous
    done = {c["case"]["id"] for c in result["cases"]}
    old_wall = result.get("wall_seconds", 0)
    started = time.monotonic()
    for case in selected:
        if case["id"] in done:
            continue
        if rank == 0:
            print(f'START {case["id"]} ({len(result["cases"])}/{len(selected)})', flush=True)
        result["cases"].append(run_case(case, device, helper, cpu_group))
        result["wall_seconds"] = old_wall + time.monotonic() - started
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, indent=2) + "\n")
            temporary.replace(args.output)
        dist.barrier(group=cpu_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
