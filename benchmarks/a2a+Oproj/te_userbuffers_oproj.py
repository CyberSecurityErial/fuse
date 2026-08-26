#!/usr/bin/env python3
"""Adapted TE Userbuffers baseline for Ulysses inverse-A2A -> O projection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import transformer_engine  # Loads libtransformer_engine.so with global symbol visibility.
import transformer_engine_torch as tex
import triton
import triton.language as tl

from te_nccl_baseline import CublasLtRunner, summarize, timed_critical


@triton.jit
def _pack_inverse_a2a_kernel(
    source,
    packed,
    numel: tl.constexpr,
    seq: tl.constexpr,
    seq_local: tl.constexpr,
    chunk_tokens: tl.constexpr,
    k_local: tl.constexpr,
    world: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    k = offsets % k_local
    m = (offsets // k_local) % seq_local
    target = offsets // (k_local * seq_local)
    batch = m // (2 * chunk_tokens)
    local_m = m % (2 * chunk_tokens)
    first = local_m < chunk_tokens
    chunk = tl.where(first, 2 * target, 2 * world - 2 * target - 1)
    in_chunk = tl.where(first, local_m, local_m - chunk_tokens)
    source_token = batch * seq + chunk * chunk_tokens + in_chunk
    tl.store(packed + offsets, tl.load(source + source_token * k_local + k, mask=mask), mask=mask)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-seq", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--q-heads", type=int, default=40)
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
    parser.add_argument("--pack-block", type=int, default=1024)
    parser.add_argument("--pack-warps", type=int, default=4)
    parser.add_argument("--tune-warmup", type=int, default=3)
    parser.add_argument("--tune-iters", type=int, default=12)
    parser.add_argument("--workspace-mib", type=int, default=64)
    parser.add_argument("--math-sm", type=int, default=0)
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--cublaslt-library",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "build" / "libfuse_cublaslt_runner.so",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if args.global_seq % world or args.q_heads % world:
        raise ValueError("global sequence and Q heads must be divisible by CP")
    if args.global_seq % (2 * world):
        raise ValueError("causal Ulysses layout requires global sequence divisible by 2*CP")

    seq_local = args.global_seq // world
    m = args.batch * seq_local
    k = args.q_heads * args.head_dim
    k_local = k // world
    n = args.hidden
    chunk_tokens = args.global_seq // (2 * world)
    chunk_bytes = m * k_local * torch.bfloat16.itemsize

    generator = torch.Generator(device=device).manual_seed(2701 + rank)
    source = torch.empty(
        (args.batch, args.global_seq, k_local), dtype=torch.bfloat16, device=device
    ).uniform_(-0.125, 0.125, generator=generator)
    weight = torch.empty((n, k), dtype=torch.bfloat16, device=device).uniform_(
        -0.02, 0.02, generator=generator
    )
    dist.broadcast(weight, src=0)
    weight_shards = [
        weight[:, peer * k_local : (peer + 1) * k_local].contiguous() for peer in range(world)
    ]
    output = torch.empty((m, n), dtype=torch.bfloat16, device=device)

    helper = tex.CommOverlapHelper(dist.group.WORLD, dist.group.WORLD)
    ub = tex.CommOverlapP2P(
        [2 * args.batch * args.global_seq, k_local],
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
    storage = ub.get_buffer(False, [2, world, m, k_local])
    send, recv = storage[0], storage[1]
    _, recv_stream_raw = ub.get_communication_stream()
    # TE returns generic ATen streams. Retag their external CUDA handles with
    # the torchrun-local device ordinal (important with CUDA_VISIBLE_DEVICES).
    active_send_streams = min(args.num_streams, world - 1) if args.parallel_sends else 1
    send_streams = [
        torch.cuda.ExternalStream(ub.get_userbuffers_send_stream(i).stream_id, device=device)
        for i in range(active_send_streams)
    ]
    recv_stream = torch.cuda.ExternalStream(recv_stream_raw.stream_id, device=device)

    plan = CublasLtRunner(
        args.cublaslt_library,
        recv[0],
        weight_shards[0],
        output,
        tune_warmup=args.tune_warmup,
        tune_iters=args.tune_iters,
        workspace_mib=args.workspace_mib,
        sm_count_target=args.math_sm,
    )

    pack_numel = send.numel()
    peer_steps = list(range(1, world))
    if args.reverse:
        peer_steps.reverse()
    packed_event = torch.cuda.Event()
    send_done_events = [torch.cuda.Event() for _ in send_streams]
    ready_events = [torch.cuda.Event() for _ in peer_steps]

    def boundary() -> torch.Tensor:
        main_stream = torch.cuda.current_stream(device)
        _pack_inverse_a2a_kernel[(triton.cdiv(pack_numel, args.pack_block),)](
            source,
            send,
            numel=pack_numel,
            seq=args.global_seq,
            seq_local=seq_local,
            chunk_tokens=chunk_tokens,
            k_local=k_local,
            world=world,
            BLOCK=args.pack_block,
            num_warps=args.pack_warps,
        )
        packed_event.record(main_stream)
        for send_stream in send_streams:
            send_stream.wait_event(packed_event)
        recv_stream.wait_event(packed_event)

        for index, step in enumerate(peer_steps):
            send_peer = (rank + step) % world
            stream_id = index % active_send_streams
            ub.userbuffers_p2p_send(
                send_peer * chunk_bytes,
                (world + rank) * chunk_bytes,
                chunk_bytes,
                send_peer,
                stream_id,
            )

        plan.accumulate(send[rank], weight_shards[rank], output, beta=0.0)
        for index, step in enumerate(peer_steps):
            recv_peer = (rank - step) % world
            ub.userbuffers_p2p_recv(
                rank * chunk_bytes,
                (world + recv_peer) * chunk_bytes,
                chunk_bytes,
                recv_peer,
            )
            ready_events[index].record(recv_stream)
            main_stream.wait_event(ready_events[index])
            plan.accumulate(recv[recv_peer], weight_shards[recv_peer], output, beta=1.0)
        for send_stream, send_done_event in zip(send_streams, send_done_events):
            send_done_event.record(send_stream)
            main_stream.wait_event(send_done_event)
        return output

    check: dict[str, float | int] = {}
    if args.check:
        boundary()
        torch.cuda.synchronize(device)
        peers = [torch.empty_like(source) for _ in range(world)]
        dist.all_gather(peers, source)
        selected = (2 * rank, 2 * world - 2 * rank - 1)
        # Selecting both causal chunks with one advanced-index kernel exceeds
        # PyTorch's launch geometry at the largest benchmark shapes.  Each
        # chunk is already contiguous, so concatenate the two views directly.
        expected_shards = []
        for peer in peers:
            peer_chunks = peer.view(
                args.batch, 2 * world, chunk_tokens, k_local
            )
            expected_shards.append(torch.cat(
                (
                    peer_chunks[:, selected[0]].reshape(-1, k_local),
                    peer_chunks[:, selected[1]].reshape(-1, k_local),
                ),
                dim=0,
            ))
        expected_a = torch.cat(expected_shards, dim=1)
        expected = torch.mm(expected_a, weight.t())
        max_abs = (output.float() - expected.float()).abs().max()
        expected_self = expected_a[:, rank * k_local : (rank + 1) * k_local]
        mismatches = torch.count_nonzero(send[rank] != expected_self)
        recv_mismatches_by_peer = torch.zeros(world, dtype=torch.int64, device=device)
        for peer in range(world):
            if peer != rank:
                expected_peer = expected_a[:, peer * k_local : (peer + 1) * k_local]
                recv_mismatches_by_peer[peer] = torch.count_nonzero(recv[peer] != expected_peer)
        dist.all_reduce(max_abs, op=dist.ReduceOp.MAX)
        dist.all_reduce(mismatches, op=dist.ReduceOp.SUM)
        dist.all_reduce(recv_mismatches_by_peer, op=dist.ReduceOp.SUM)
        recv_mismatches = recv_mismatches_by_peer.sum()
        check = {
            "max_abs": float(max_abs.item()),
            "self_pack_mismatches": int(mismatches.item()),
            "remote_recv_mismatches": int(recv_mismatches.item()),
            "remote_recv_mismatches_by_peer": recv_mismatches_by_peer.cpu().tolist(),
        }
        if mismatches.item() or recv_mismatches.item() or max_abs.item() > 0.01:
            raise RuntimeError(f"Userbuffers A2A correctness failed: {check}")

    samples = timed_critical(
        boundary, args.warmup, args.iters, device, use_cuda_graph=args.cuda_graph
    )
    if args.check and args.cuda_graph:
        torch.cuda.synchronize(device)
        post_graph_max_abs = (output.float() - expected.float()).abs().max()
        dist.all_reduce(post_graph_max_abs, op=dist.ReduceOp.MAX)
        check["post_graph_max_abs"] = float(post_graph_max_abs.item())
        if post_graph_max_abs.item() > 0.01:
            raise RuntimeError(f"Userbuffers CUDA Graph replay correctness failed: {check}")
    flops = 2 * m * n * k
    result = {
        "mode": "te_userbuffers_adapted_a2a_oproj",
        "world_size": world,
        "gemm_shape": {"m": m, "n": n, "k": k},
        "config": vars(args) | {"json_out": str(args.json_out) if args.json_out else None},
        "cublaslt_plan": plan.info,
        "correctness": check,
        "results": {"te_userbuffers_oproj_boundary": summarize(samples, flops=flops, world=world)},
        "samples_ms": samples,
    }
    if rank == 0:
        print(json.dumps(result, indent=2, default=str))
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(result, indent=2, default=str) + "\n")
    plan.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
