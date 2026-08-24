#!/usr/bin/env python3
"""TE dense Transformer forward benchmark with Ulysses context parallelism."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
import transformer_engine
import transformer_engine.pytorch as te


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-seq", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=8192)
    parser.add_argument("--ffn-hidden", type=int, default=28672)
    parser.add_argument("--q-heads", type=int, default=64)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--layers", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--cp-comm-type", choices=("a2a", "p2p", "all_gather"), default="a2a")
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": sum(values) / len(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(4000)
    torch.cuda.manual_seed(4000)
    if args.global_seq % world != 0:
        raise ValueError("global sequence length must be divisible by CP size")
    if args.q_heads % world != 0 or args.kv_heads % world != 0:
        raise ValueError("Q and KV heads must be divisible by CP size")

    seq_local = args.global_seq // world
    layer = te.TransformerLayer(
        args.hidden,
        args.ffn_hidden,
        args.q_heads,
        num_gqa_groups=args.kv_heads,
        kv_channels=args.head_dim,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        params_dtype=torch.bfloat16,
        self_attn_mask_type="causal",
        bias=False,
        activation="swiglu",
        normalization="RMSNorm",
        fuse_qkv_params=True,
        qkv_weight_interleaved=False,
        attn_input_format="bshd",
        seq_length=args.global_seq,
        micro_batch_size=args.batch,
        device=device,
        name="dense_70b_reused_layer",
    ).eval()
    cp_stream = torch.cuda.Stream(device=device, priority=-1)
    layer.set_context_parallel_group(
        dist.group.WORLD,
        list(range(world)),
        cp_stream,
        args.cp_comm_type,
    )
    generator = torch.Generator(device=device).manual_seed(4100 + rank)
    hidden_states = torch.empty(
        (args.batch, seq_local, args.hidden),
        dtype=torch.bfloat16,
        device=device,
    ).uniform_(-0.01, 0.01, generator=generator)

    def forward() -> torch.Tensor:
        value = hidden_states
        for _ in range(args.layers):
            value = layer(value, self_attn_mask_type="causal")
        return value

    for _ in range(args.warmup):
        forward()
    torch.cuda.synchronize(device)
    dist.barrier()

    timed_forward = forward
    graph = None
    if args.cuda_graph:
        graph = torch.cuda.CUDAGraph()
        dist.barrier()
        with torch.cuda.graph(graph):
            forward()
        torch.cuda.synchronize(device)
        dist.barrier()
        timed_forward = graph.replay
        for _ in range(args.warmup):
            timed_forward()
        torch.cuda.synchronize(device)
        dist.barrier()

    samples: list[float] = []
    for _ in range(args.iters):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        timed_forward()
        stop.record()
        stop.synchronize()
        elapsed = torch.tensor(start.elapsed_time(stop), dtype=torch.float64, device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        samples.append(float(elapsed.item()))

    validation_output = forward()
    nonfinite = torch.count_nonzero(~torch.isfinite(validation_output)).to(
        dtype=torch.int64
    )
    dist.all_reduce(nonfinite, op=dist.ReduceOp.SUM)
    stats = summarize(samples)
    parameter_elements = sum(parameter.numel() for parameter in layer.parameters())
    output = {
        "scope": "TE dense TransformerLayer forward replay; one weight set reused for every layer",
        "model": {
            "layers": args.layers,
            "batch": args.batch,
            "global_seq": args.global_seq,
            "seq_local": seq_local,
            "hidden": args.hidden,
            "ffn_hidden": args.ffn_hidden,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
        },
        "world_size": world,
        "dtype": "bfloat16",
        "cp_comm_type": args.cp_comm_type,
        "cuda_graph": args.cuda_graph,
        "warmup": args.warmup,
        "iterations": args.iters,
        "parameter_elements_per_reused_layer": parameter_elements,
        "software": {
            "torch": torch.__version__,
            "transformer_engine": transformer_engine.__version__,
            "cuda": torch.version.cuda,
            "nccl": ".".join(str(item) for item in torch.cuda.nccl.version()),
        },
        "timing": "per-sample max-rank CUDA-event critical path",
        "nonfinite_count": int(nonfinite.item()),
        "results": {
            "full_forward": stats,
            "per_layer_mean_ms": stats["mean_ms"] / args.layers,
            "samples_ms": samples,
        },
    }
    if rank == 0:
        print(json.dumps(output, indent=2))
        if args.json_out is not None:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(output, indent=2) + "\n")
    # TE keeps auxiliary CP-stream state alive past ProcessGroup teardown on
    # this local build. All measured work and the result file are complete at
    # this barrier; terminate the disposable benchmark workers deterministically.
    dist.barrier()
    torch.cuda.synchronize(device)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
