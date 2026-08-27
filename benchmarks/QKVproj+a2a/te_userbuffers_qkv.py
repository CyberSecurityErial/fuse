#!/usr/bin/env python3
"""Adapted TE Userbuffers baseline for QKV projection -> Ulysses A2A."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import transformer_engine  # Loads libtransformer_engine.so with global symbols.
import transformer_engine_torch as tex
import triton
import triton.language as tl

from te_nccl_baseline import CublasLtRunner, summarize, timed_critical


@triton.jit
def _unpack_qkv_slabs_kernel(
    send,
    recv,
    output,
    numel: tl.constexpr,
    m: tl.constexpr,
    rank: tl.constexpr,
    q_local_width: tl.constexpr,
    kv_local_width: tl.constexpr,
    slab_width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = (
        tl.program_id(0).to(tl.int64) * BLOCK
        + tl.arange(0, BLOCK).to(tl.int64)
    )
    mask = offsets < numel
    q_elements = tl.full((), (numel // slab_width) * q_local_width, tl.int64)
    kv_elements = tl.full((), (numel // slab_width) * kv_local_width, tl.int64)
    is_q = offsets < q_elements
    is_k = (offsets >= q_elements) & (offsets < q_elements + kv_elements)
    segment_element = tl.where(
        is_q,
        offsets,
        tl.where(is_k, offsets - q_elements, offsets - q_elements - kv_elements),
    )
    segment_width = tl.where(is_q, q_local_width, kv_local_width)
    source_rank = segment_element // (m * segment_width)
    within_source = segment_element % (m * segment_width)
    token = within_source // segment_width
    channel = within_source % segment_width
    segment_base = tl.where(
        is_q,
        0,
        tl.where(is_k, q_local_width, q_local_width + kv_local_width),
    )
    source = source_rank * m * slab_width + token * slab_width + segment_base + channel
    send_value = tl.load(send + source, mask=mask)
    recv_value = tl.load(recv + source, mask=mask)
    tl.store(output + offsets, tl.where(source_rank == rank, send_value, recv_value), mask=mask)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-seq", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--num-comm-sm", type=int, default=8)
    parser.add_argument("--num-streams", type=int, default=3)
    parser.add_argument("--parallel-sends", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-ce", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--push", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reverse", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--local-first", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pack-block", type=int, default=1024)
    parser.add_argument("--pack-warps", type=int, default=4)
    parser.add_argument("--tune-warmup", type=int, default=3)
    parser.add_argument("--tune-iters", type=int, default=12)
    parser.add_argument("--workspace-mib", type=int, default=64)
    parser.add_argument("--math-sm", type=int, default=0)
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--trace-out", type=Path)
    parser.add_argument("--matrix-manifest", type=Path)
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--cublaslt-library",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "build" / "libfuse_cublaslt_runner.so",
    )
    return parser.parse_args()


_PLAN_CACHE: dict[tuple[object, ...], CublasLtRunner] = {}


def cached_plan(
    args: argparse.Namespace,
    source: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
) -> CublasLtRunner:
    m, k = source.shape
    n = weight.shape[0]
    key = (
        source.device.index,
        m,
        n,
        k,
        args.math_sm,
        str(args.cublaslt_library.resolve()),
        args.tune_warmup,
        args.tune_iters,
        args.workspace_mib,
    )
    if key not in _PLAN_CACHE:
        _PLAN_CACHE[key] = CublasLtRunner(
            args.cublaslt_library,
            source,
            weight,
            output,
            tune_warmup=args.tune_warmup,
            tune_iters=args.tune_iters,
            workspace_mib=args.workspace_mib,
            sm_count_target=args.math_sm,
        )
    return _PLAN_CACHE[key]


def close_plans() -> None:
    for plan in _PLAN_CACHE.values():
        plan.close()
    _PLAN_CACHE.clear()


def run(
    args: argparse.Namespace,
    rank: int,
    world: int,
    device: torch.device,
    helper: tex.CommOverlapHelper,
) -> dict[str, object]:

    if args.batch != 1:
        raise ValueError("the QKV Userbuffers baseline currently requires batch=1")
    if args.global_seq % world or args.q_heads % world or args.kv_heads % world:
        raise ValueError("sequence, Q heads, and KV heads must be divisible by CP")

    m = args.global_seq // world
    k = args.hidden
    q_width = args.q_heads * args.head_dim
    kv_width = args.kv_heads * args.head_dim
    n = q_width + 2 * kv_width
    q_local_width = q_width // world
    kv_local_width = kv_width // world
    slab_width = q_local_width + 2 * kv_local_width
    slab_bytes = m * slab_width * torch.bfloat16.itemsize

    generator = torch.Generator(device=device).manual_seed(3109 + rank)
    source = torch.empty((m, k), dtype=torch.bfloat16, device=device).uniform_(
        -0.125, 0.125, generator=generator
    )
    weight = torch.empty((n, k), dtype=torch.bfloat16, device=device).uniform_(
        -0.02, 0.02, generator=generator
    )
    dist.broadcast(weight, src=0)

    weight_slabs = []
    for peer in range(world):
        q0 = peer * q_local_width
        k0 = q_width + peer * kv_local_width
        v0 = q_width + kv_width + peer * kv_local_width
        weight_slabs.append(torch.cat(
            (
                weight[q0 : q0 + q_local_width],
                weight[k0 : k0 + kv_local_width],
                weight[v0 : v0 + kv_local_width],
            ),
            dim=0,
        ).contiguous())

    ub = tex.CommOverlapP2P(
        [2 * args.global_seq, slab_width],
        torch.bfloat16,
        helper,
        world,
        # Manual P2P offsets need a flat, exactly-sized registered buffer.
        # AG mode preserves that layout; RS mode adds TE's internal staging.
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
    storage = ub.get_buffer(False, [2, world, m, slab_width])
    send, recv = storage[0], storage[1]
    output = torch.empty(world * m * slab_width, dtype=torch.bfloat16, device=device)

    _, recv_stream_raw = ub.get_communication_stream()
    active_send_streams = min(args.num_streams, world - 1) if args.parallel_sends else 1
    send_streams = [
        torch.cuda.ExternalStream(ub.get_userbuffers_send_stream(i).stream_id, device=device)
        for i in range(active_send_streams)
    ]
    recv_stream = torch.cuda.ExternalStream(recv_stream_raw.stream_id, device=device)

    plan = cached_plan(args, source, weight_slabs[0], send[0])

    peer_steps = list(range(1, world))
    if args.reverse:
        peer_steps.reverse()
    remote_peers = [(rank + step) % world for step in peer_steps]
    gemm_order = ([rank] + remote_peers) if args.local_first else (remote_peers + [rank])
    # Timing is enabled only for an explicit diagnostic run.  Normal tuning
    # and formal measurements retain the original lightweight event objects.
    trace_enabled = args.trace_out is not None
    gemm_done = [torch.cuda.Event() for _ in range(world)]
    boundary_start = torch.cuda.Event()
    recv_ready = [torch.cuda.Event() for _ in peer_steps]
    send_done = [torch.cuda.Event() for _ in send_streams]
    trace_events = None
    if trace_enabled:
        trace_events = {
            "start": torch.cuda.Event(enable_timing=True, external=True),
            "compute_end": torch.cuda.Event(enable_timing=True, external=True),
            "send_start": [
                torch.cuda.Event(enable_timing=True, external=True)
                for _ in send_streams
            ],
            "send_end": [
                torch.cuda.Event(enable_timing=True, external=True)
                for _ in send_streams
            ],
            "recv_start": torch.cuda.Event(enable_timing=True, external=True),
            "recv_end": torch.cuda.Event(enable_timing=True, external=True),
            "unpack_start": torch.cuda.Event(enable_timing=True, external=True),
            "end": torch.cuda.Event(enable_timing=True, external=True),
        }

    def boundary() -> torch.Tensor:
        main_stream = torch.cuda.current_stream(device)
        # Join TE's external receive stream to the same Graph capture, exactly
        # as the OProj adapter does with its packed-data event.
        boundary_start.record(main_stream)
        if trace_events is not None:
            trace_events["start"].record(main_stream)
        recv_stream.wait_event(boundary_start)
        remote_index = 0
        for destination in gemm_order:
            plan.accumulate(source, weight_slabs[destination], send[destination], beta=0.0)
            gemm_done[destination].record(main_stream)
            if trace_events is not None and destination == gemm_order[-1]:
                trace_events["compute_end"].record(main_stream)
            if destination != rank:
                stream_id = remote_index % active_send_streams
                send_streams[stream_id].wait_event(gemm_done[destination])
                if trace_events is not None and remote_index < active_send_streams:
                    trace_events["send_start"][stream_id].record(send_streams[stream_id])
                ub.userbuffers_p2p_send(
                    destination * slab_bytes,
                    (world + rank) * slab_bytes,
                    slab_bytes,
                    destination,
                    stream_id,
                )
                remote_index += 1

        # Keep the same Userbuffers handshake as the validated OProj adapter:
        # queue every send first, then submit receives in ring-peer order.
        if trace_events is not None:
            trace_events["recv_start"].record(recv_stream)
        for index, step in enumerate(peer_steps):
            source_peer = (rank - step) % world
            ub.userbuffers_p2p_recv(
                rank * slab_bytes,
                (world + source_peer) * slab_bytes,
                slab_bytes,
                source_peer,
            )
            if trace_events is not None and index == len(peer_steps) - 1:
                trace_events["recv_end"].record(recv_stream)
            recv_ready[index].record(recv_stream)

        for event in recv_ready:
            main_stream.wait_event(event)
        for stream_id, (stream, event) in enumerate(zip(send_streams, send_done)):
            if trace_events is not None:
                trace_events["send_end"][stream_id].record(stream)
            event.record(stream)
            main_stream.wait_event(event)

        numel = output.numel()
        if trace_events is not None:
            trace_events["unpack_start"].record(main_stream)
        _unpack_qkv_slabs_kernel[(triton.cdiv(numel, args.pack_block),)](
            send,
            recv,
            output,
            numel=numel,
            m=m,
            rank=rank,
            q_local_width=q_local_width,
            kv_local_width=kv_local_width,
            slab_width=slab_width,
            BLOCK=args.pack_block,
            num_warps=args.pack_warps,
        )
        if trace_events is not None:
            trace_events["end"].record(main_stream)
        return output

    check: dict[str, float | int] = {}
    expected = None
    if args.check:
        boundary()
        torch.cuda.synchronize(device)

        # Mirror the validated OProj adapter's two-layer check: first prove
        # that Userbuffers delivered each peer slab byte-for-byte, then check
        # the final Q/K/V segment layout against an independent projection.
        gathered_send = [torch.empty_like(send) for _ in range(world)]
        dist.all_gather(gathered_send, send)
        recv_mismatches_by_peer = torch.zeros(world, dtype=torch.int64, device=device)
        for peer in range(world):
            if peer != rank:
                recv_mismatches_by_peer[peer] = torch.count_nonzero(
                    recv[peer] != gathered_send[peer][rank]
                )

        source_peers = [torch.empty_like(source) for _ in range(world)]
        dist.all_gather(source_peers, source)
        q0 = rank * q_local_width
        k0 = q_width + rank * kv_local_width
        v0 = q_width + kv_width + rank * kv_local_width
        projected = [torch.mm(peer_source, weight.t()) for peer_source in source_peers]
        expected = torch.cat(
            tuple(item[:, q0 : q0 + q_local_width].reshape(-1) for item in projected)
            + tuple(item[:, k0 : k0 + kv_local_width].reshape(-1) for item in projected)
            + tuple(item[:, v0 : v0 + kv_local_width].reshape(-1) for item in projected)
        )
        max_abs = (output.float() - expected.float()).abs().max()
        dist.all_reduce(max_abs, op=dist.ReduceOp.MAX)
        dist.all_reduce(recv_mismatches_by_peer, op=dist.ReduceOp.SUM)
        recv_mismatches = recv_mismatches_by_peer.sum()
        check = {
            "max_abs": float(max_abs.item()),
            "remote_recv_mismatches": int(recv_mismatches.item()),
            "remote_recv_mismatches_by_peer": recv_mismatches_by_peer.cpu().tolist(),
        }
        if recv_mismatches.item() or max_abs.item() > 0.01:
            raise RuntimeError(f"QKV Userbuffers correctness failed: {check}")
        del gathered_send, source_peers, projected, recv_mismatches_by_peer

    samples = timed_critical(
        boundary, args.warmup, args.iters, device, use_cuda_graph=args.cuda_graph
    )

    trace_summary = None
    if trace_enabled:
        torch.cuda.synchronize(device)

        assert trace_events is not None

        def elapsed_us(event: torch.cuda.Event) -> float:
            return 1000.0 * trace_events["start"].elapsed_time(event)

        compute_end_us = elapsed_us(trace_events["compute_end"])
        send_start_us = min(elapsed_us(event) for event in trace_events["send_start"])
        send_end_us = max(elapsed_us(event) for event in trace_events["send_end"])
        recv_start_us = elapsed_us(trace_events["recv_start"])
        recv_end_us = elapsed_us(trace_events["recv_end"])
        comm_end_us = max(send_end_us, recv_end_us)
        comm_start_us = min(send_start_us, recv_start_us)
        unpack_start_us = elapsed_us(trace_events["unpack_start"])
        total_us = elapsed_us(trace_events["end"])
        overlap_us = max(
            0.0, min(compute_end_us, comm_end_us) - max(0.0, comm_start_us)
        )
        trace_summary = {
            "rank": rank,
            "compute_us": compute_end_us,
            "send_start_us": send_start_us,
            "send_end_us": send_end_us,
            "recv_start_us": recv_start_us,
            "recv_end_us": recv_end_us,
            "communication_start_us": comm_start_us,
            "communication_end_us": comm_end_us,
            "communication_us": comm_end_us - comm_start_us,
            "overlap_us": overlap_us,
            "communication_hidden_fraction": (
                overlap_us / (comm_end_us - comm_start_us)
                if comm_end_us > comm_start_us else 0.0
            ),
            "unpack_start_us": unpack_start_us,
            "unpack_tail_us": max(0.0, total_us - unpack_start_us),
            "total_us": total_us,
        }
        rank_summaries: list[dict[str, object] | None] = [None] * world
        dist.all_gather_object(rank_summaries, trace_summary)
        if rank == 0:
            perfetto_events: list[dict[str, object]] = []
            lane_names = {
                0: "boundary",
                1: "QKV slab GEMMs",
                2: "communication envelope",
                3: "send",
                4: "recv",
                5: "unpack / tail",
            }
            for peer_rank, summary in enumerate(rank_summaries):
                assert summary is not None
                perfetto_events.append({
                    "name": "process_name", "ph": "M", "pid": peer_rank,
                    "tid": 0, "args": {"name": f"TE Userbuffers rank {peer_rank}"},
                })
                for tid, name in lane_names.items():
                    perfetto_events.append({
                        "name": "thread_name", "ph": "M", "pid": peer_rank,
                        "tid": tid, "args": {"name": name},
                    })

                def complete(name: str, category: str, tid: int,
                             start_us: float, end_us: float) -> None:
                    perfetto_events.append({
                        "name": name, "cat": category, "ph": "X",
                        "pid": peer_rank, "tid": tid, "ts": start_us,
                        "dur": max(0.0, end_us - start_us),
                    })

                compute_us = float(summary["compute_us"])
                send_start = float(summary["send_start_us"])
                send_end = float(summary["send_end_us"])
                recv_end = float(summary["recv_end_us"])
                recv_start = float(summary["recv_start_us"])
                comm_start = float(summary["communication_start_us"])
                comm_end = float(summary["communication_end_us"])
                unpack_start = float(summary["unpack_start_us"])
                total_us = float(summary["total_us"])
                complete("TE UB boundary", "boundary", 0, 0.0, total_us)
                complete("8 QKV slab GEMMs", "compute", 1, 0.0, compute_us)
                complete("send + recv envelope", "communication", 2,
                         comm_start, comm_end)
                complete("remote sends", "communication", 3, send_start, send_end)
                complete("remote receives", "communication", 4,
                         recv_start, recv_end)
                complete("unpack / dependency tail", "epilogue", 5,
                         unpack_start, total_us)
            payload = {"displayTimeUnit": "ns", "traceEvents": perfetto_events}
            args.trace_out.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.trace_out.with_suffix(args.trace_out.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, indent=2) + "\n")
            temporary.replace(args.trace_out)
    if args.check and args.cuda_graph:
        torch.cuda.synchronize(device)
        post_graph_max_abs = (output.float() - expected.float()).abs().max()
        dist.all_reduce(post_graph_max_abs, op=dist.ReduceOp.MAX)
        check["post_graph_max_abs"] = float(post_graph_max_abs.item())
        if post_graph_max_abs.item() > 0.01:
            raise RuntimeError(f"QKV Userbuffers graph correctness failed: {check}")

    flops = 2 * m * n * k
    result = {
        "mode": "te_userbuffers_adapted_qkv_a2a",
        "launch": "graph" if args.cuda_graph else "eager",
        "timing": "per_sample_local_cuda_event_then_dist_max",
        "timed_boundary": "all_qkv_slab_gemms+userbuffers_send_recv+unpack",
        "graph_setup_timed": False,
        "rank_reduction": "MAX",
        "world_size": world,
        "gemm_shape": {"m": m, "n": n, "k": k},
        "head_geometry": {
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
        },
        "sequence_order": "rank_major",
        "config": vars(args) | {"json_out": str(args.json_out) if args.json_out else None},
        "cublaslt_plan": plan.info,
        "correctness": check,
        "results": {
            "te_userbuffers_qkv_boundary": summarize(samples, flops=flops, world=world)
        },
        "samples_ms": samples,
    }
    if trace_summary is not None:
        result["trace_summary"] = trace_summary
    dist.barrier()
    return result


def write_result(args: argparse.Namespace, result: dict[str, object], rank: int) -> None:
    if rank != 0:
        return
    print(json.dumps(result, indent=2, default=str))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.json_out.with_suffix(args.json_out.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, default=str) + "\n")
        temporary.replace(args.json_out)


def run_initialized(
    args: argparse.Namespace,
    rank: int,
    world: int,
    device: torch.device,
    helper: tex.CommOverlapHelper,
) -> None:
    result = run(args, rank, world, device, helper)
    write_result(args, result, rank)
    torch.cuda.synchronize(device)
    del result
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    helper = tex.CommOverlapHelper(dist.group.WORLD, dist.group.WORLD)

    if args.matrix_manifest:
        entries = json.loads(args.matrix_manifest.read_text())
        for entry in entries:
            for name, value in entry["arguments"].items():
                setattr(args, name, value)
            args.json_out = Path(entry["json_out"])
            run_initialized(args, rank, world, device, helper)
    else:
        run_initialized(args, rank, world, device, helper)

    close_plans()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
