#!/usr/bin/env python3
"""CP8 changing-payload regression for native P2P copy bounds and completion."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
import transformer_engine.pytorch
import transformer_engine_torch as tex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    if rank == 0:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks/sm103'))
        from bench import observe_devices
        observations = observe_devices(os.environ['CUDA_VISIBLE_DEVICES'])
        args.output.with_suffix('.gpu-before.json').write_text(json.dumps(observations))
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    dist.init_process_group('nccl', device_id=device)
    world = dist.get_world_size()
    assert world == 8
    assert torch.cuda.get_device_capability() == (10, 3)
    helper = tex.CommOverlapHelper(dist.group.WORLD, dist.group.WORLD)
    # 2 MiB + 16 bytes per peer: exercise a non-unrolled tail as well as bulk.
    elements = 1048584
    nbytes = elements * 2
    ub = tex.CommOverlapP2P(
        [2 * world, elements], torch.bfloat16, helper, world, tex.CommOverlapType.AG,
        num_max_streams=1, comm_cga_size=1, gemm_priority=0, comm_priority=-1,
        num_comm_sm=4, set_sm_margin=False, atomic_gemm=False, use_ce=False, aggregate=False)
    storage = ub.get_buffer(False, [2, world, elements])
    send, recv = storage[0], storage[1]
    snapshot = torch.empty_like(recv)
    base = (torch.arange(elements, device=device, dtype=torch.int32) % 17).to(torch.bfloat16)
    tx = torch.cuda.ExternalStream(ub.get_userbuffers_send_stream(0).stream_id, device=device)
    rx = torch.cuda.ExternalStream(ub.get_communication_stream()[1].stream_id, device=device)
    ready, tx_done, rx_done = (torch.cuda.Event() for _ in range(3))

    def boundary():
        main_stream = torch.cuda.current_stream()
        ready.record(main_stream)
        tx.wait_event(ready)
        rx.wait_event(ready)
        for step in range(1, world):
            peer = (rank + step) % world
            ub.userbuffers_p2p_send(peer * nbytes, (world + rank) * nbytes, nbytes, peer, 0)
        for step in range(1, world):
            peer = (rank - step) % world
            ub.userbuffers_p2p_recv(rank * nbytes, (world + peer) * nbytes, nbytes, peer)
        rx_done.record(rx)
        main_stream.wait_event(rx_done)
        # Snapshot immediately when receive declares completion, before waiting
        # on local sends or entering a cross-rank check.
        snapshot.copy_(recv)
        tx_done.record(tx)
        main_stream.wait_event(tx_done)

    results = []
    configs = [(sm, push, False) for sm in range(1, 33) for push in (True, False)]
    configs += [(4, push, True) for push in (True, False)]
    for sm, push, ce in configs:
        ub.configure_userbuffers_p2p(sm, ce, push)
        for launch in ('eager', 'graph'):
            dist.barrier()
            graph = None
            send.zero_()
            boundary()
            torch.cuda.synchronize()
            if launch == 'graph':
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    boundary()
            for iteration in range(3):
                for peer in range(world):
                    send[peer].copy_(base + rank * 8 + peer + iteration * 64)
                # Sentinel exposes an incomplete transfer even when a previous
                # iteration happened to leave the same payload in this buffer.
                recv.fill_(-1)
                torch.cuda.synchronize()
                dist.barrier()  # Every rank finishes clearing before any peer sends.
                if graph is None:
                    boundary()
                else:
                    graph.replay()
                errors = torch.zeros((), device=device, dtype=torch.int64)
                for peer in range(world):
                    if peer != rank:
                        expected = base + peer * 8 + rank + iteration * 64
                        errors += torch.count_nonzero(snapshot[peer] != expected)
                dist.all_reduce(errors)
                if errors.item():
                    raise RuntimeError(f'P2P mismatch: sms={sm} push={push} ce={ce} '
                                       f'launch={launch} iteration={iteration} errors={errors.item()}')
            if graph is not None:
                graph.reset()
            results.append(dict(sms=sm, push=push, use_ce=ce, launch=launch,
                                changing_payload_rounds=3, mismatches=0))
        if rank == 0:
            print(f'PASS sms={sm} push={push} ce={ce}: eager+graph', flush=True)
    torch.cuda.synchronize()
    if rank == 0:
        args.output.write_text(json.dumps(dict(world_size=world, bytes_per_peer=nbytes,
                                              checks=results), indent=2) + '\n')
        print(f'PASSED {len(results)} configurations, each with three changing payloads', flush=True)
    del storage, send, recv, ub
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
