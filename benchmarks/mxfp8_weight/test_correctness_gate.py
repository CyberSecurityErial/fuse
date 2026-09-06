"""Distributed negative tests: one rank's NaN/Inf/error must fail every rank.

Run with the same torchrun/environment as bench.py, no extra arguments.
"""
import os
import torch
import torch.distributed as dist

import bench


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", device_id=device)
    group = dist.new_group(backend="gloo")
    reference = torch.ones(8192, device=device, dtype=torch.bfloat16)
    actual = reference.clone()
    assert bench.check_error(actual, reference, group)["max_abs"] == 0
    for value in (float("nan"), float("inf"), 100.0):
        actual.copy_(reference)
        if dist.get_rank() == 1:
            actual[4097] = value
        failed = False
        try:
            bench.check_error(actual, reference, group)
        except RuntimeError:
            failed = True
        flags = [None] * dist.get_world_size()
        dist.all_gather_object(flags, failed, group=group)
        assert all(flags), (value, flags)
    # Software dequantization against the independent reference, including
    # zero blocks and several ordinary exponent ranges used by the benchmark.
    for magnitude in (0., .001, .02, 1., 128.):
        weight = (torch.randn((128, 256), device=device) * magnitude).bfloat16()
        q, s = bench.quantize_offline(weight)
        expected = bench.reference_dequant(q, s)
        recovered = torch.empty_like(weight)
        bench.dequant_kernel[(bench.triton.cdiv(q.numel(), 2048),)](q, s, recovered, q.numel(), 2048)
        assert torch.equal(recovered, expected)
    dist.barrier(group=group)
    if dist.get_rank() == 0:
        print("PASS: exact, one-rank NaN/Inf/finite corruption, zero/multi-scale dequant", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
