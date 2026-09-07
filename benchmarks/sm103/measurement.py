"""SM103 v2: random-input evidence and bounded, measured steady-state warmup."""
import math
import os
import statistics
import time

RECORDS = []
INPUTS = []
LAST_MEASUREMENT_FINISHED = None
OPROJ_LAYOUTS = ('legacy', 'causal_dual_chunk_v1')


def oproj_layout(value=None):
    value = os.environ.get('FUSE_SM103_OPROJ_LAYOUT', 'legacy') if value is None else value
    if value not in OPROJ_LAYOUTS:
        raise ValueError(f'unknown OProj layout: {value!r}')
    return value


def oproj_chunk_pairs(world, backend, layout='legacy'):
    """Independent CPU oracle: destination rank -> two chronological chunks.

    Legacy NCCL and UB use different permutations. Canonical input is sequence
    ordered on each head-owning peer; neither weights nor heads are permuted.
    This helper is deliberately not called by the timed packing implementation.
    """
    if layout not in OPROJ_LAYOUTS:
        raise ValueError(f'unknown OProj layout: {layout!r}')
    if type(world) is not int or world <= 0:
        raise ValueError('world must be a positive integer')
    if backend not in ('te_ub', 'cublaslt_nccl'):
        raise ValueError(f'unknown OProj backend: {backend!r}')
    if layout == 'causal_dual_chunk_v1':
        return tuple((rank, 2*world-rank-1) for rank in range(world))
    if backend == 'te_ub':
        return tuple((2*rank, 2*world-2*rank-1) for rank in range(world))
    order = tuple(range(0, 2*world, 2)) + tuple(range(2*world-1, 0, -2))
    return tuple(order[2*rank:2*rank+2] for rank in range(world))


def collect_samples(timed_fn, count, device):
    """One boundary per sample; defer statistics communication until the end."""
    import gc
    import torch
    import torch.distributed as dist
    pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
             for _ in range(count)]
    # CUDA events are initialized lazily. Prime every handle outside the timed
    # boundary, including each stop event, to avoid measuring its creation cost.
    for start, stop in pairs:
        start.record()
        stop.record()
    torch.cuda.synchronize(device)
    gc_enabled = gc.isenabled()
    try:
        gc.disable()
        for start, stop in pairs:
            dist.barrier()
            start.record()
            timed_fn()
            stop.record()
            stop.synchronize()
    finally:
        if gc_enabled:
            gc.enable()
    # Vector MAX is the same per-sample max-rank statistic as scalar MAX, with
    # one collective instead of fifty allocations/reductions/host round trips.
    rank_ms = torch.tensor([start.elapsed_time(stop) for start, stop in pairs],
                           dtype=torch.float64, device=device)
    dist.all_reduce(rank_ms, op=dist.ReduceOp.MAX)
    return rank_ms.tolist()


def count_nonzero_bounded(tensor):
    import torch
    result = torch.zeros((), dtype=torch.int64, device=tensor.device)
    rows = max(1, 1048576 // max(1, tensor.numel() // tensor.shape[0]))
    for begin in range(0, tensor.shape[0], rows):
        result += torch.count_nonzero(tensor[begin:begin+rows])
    return result


def count_mismatches(actual, expected):
    import torch
    if actual.shape != expected.shape:
        raise ValueError(f'comparison shape mismatch: {actual.shape} != {expected.shape}')
    result = torch.zeros((), dtype=torch.int64, device=actual.device)
    rows = max(1, 1048576 // max(1, actual.numel() // actual.shape[0]))
    for begin in range(0, actual.shape[0], rows):
        result += torch.count_nonzero(actual[begin:begin+rows] != expected[begin:begin+rows])
    return result


def reset():
    RECORDS.clear()
    INPUTS.clear()


def record_inputs(source, weight, *, seed, label):
    import torch
    stats = {'label': label, 'seed': seed, 'distribution': 'uniform', 'tensors': {}}
    for name, tensor in (('activation', source), ('weight', weight)):
        flat = tensor.detach().reshape(-1)
        sample = flat[::max(1, flat.numel() // 4096)][:4096].float()
        fraction = float(count_nonzero_bounded(flat).item()) / flat.numel()
        if fraction == 0 or not bool(torch.isfinite(sample).all()):
            raise ValueError(f'{label}/{name}: all-zero or nonfinite input')
        stats['tensors'][name] = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
            'nonzero_fraction': fraction, 'sample_min': sample.min().item(),
            'sample_max': sample.max().item(), 'sample_mean': sample.mean().item(),
            'sample_std': sample.std().item()}
    INPUTS.append(stats)


def stable_window(values, tolerance=.05):
    return (len(values) >= 3 and all(math.isfinite(x) and x > 0 for x in values[-3:])
            and (max(values[-3:])-min(values[-3:])) / statistics.median(values[-3:]) <= tolerance)


def max_abs_difference(actual, expected):
    """Bound FP32 comparison temporaries even for multi-GiB BF16 outputs."""
    import torch
    a, b = actual.reshape(-1), expected.reshape(-1)
    error = torch.zeros((), device=a.device)
    for begin in range(0, a.numel(), 1048576):
        end = min(begin+1048576, a.numel())
        error = torch.maximum(error, (a[begin:end].float()-b[begin:end].float()).abs().max())
    return error


def qkv_expected(source, weight, output, rank, world, q_width, kv_width):
    """Broadcast one source tile at a time; never all-gather full activations."""
    import torch
    import torch.distributed as dist
    m = source.shape[0]
    ql, kl = q_width//world, kv_width//world
    expected = torch.empty_like(output)
    tile = torch.empty((min(1024, m), source.shape[1]), dtype=source.dtype, device=source.device)
    for peer in range(world):
        for begin in range(0, m, tile.shape[0]):
            rows = min(tile.shape[0], m-begin)
            block = tile[:rows]
            if rank == peer:
                block.copy_(source[begin:begin+rows])
            dist.broadcast(block, src=peer)
            projected = torch.mm(block, weight.t())
            for base, width, dst_base in ((rank*ql, ql, 0),
                    (q_width+rank*kl, kl, world*m*ql),
                    (q_width+kv_width+rank*kl, kl, world*m*(ql+kl))):
                offset = dst_base+(peer*m+begin)*width
                expected[offset:offset+rows*width].copy_(projected[:, base:base+width].reshape(-1))
    return expected


def qkv_ub_check(source, weight, send, recv, output, rank, world, q_width, kv_width):
    import torch
    import torch.distributed as dist
    reference_recv = torch.empty_like(recv)
    dist.all_to_all_single(reference_recv, send.contiguous())
    counts = torch.zeros(world, dtype=torch.int64, device=source.device)
    for peer in range(world):
        if peer != rank:
            counts[peer] = count_mismatches(recv[peer], reference_recv[peer])
    del reference_recv
    expected = qkv_expected(source, weight, output, rank, world, q_width, kv_width)
    error = max_abs_difference(output, expected)
    dist.all_reduce(error, op=dist.ReduceOp.MAX)
    dist.all_reduce(counts)
    check = {'max_abs': error.item(), 'remote_recv_mismatches': int(counts.sum().item()),
             'remote_recv_mismatches_by_peer': counts.tolist()}
    if not math.isfinite(error.item()) or error.item() > .01 or counts.sum().item():
        raise RuntimeError(f'QKV Userbuffers correctness failed: {check}')
    return check, expected


def inverse_expected(source, rank, world, *, te_order=False):
    import torch
    import torch.distributed as dist
    layout = oproj_layout()
    batch = source.shape[0] if layout == 'causal_dual_chunk_v1' and source.ndim == 3 else 1
    seq, kl = source.numel()//(batch*source.shape[-1]), source.shape[-1]
    # Callers flatten their head dimension before entering this independent route.
    local = source.reshape(batch, seq, kl)
    chunk = seq//(2*world)
    if seq % (2*world):
        raise ValueError('inverse A2A requires sequence divisible by 2*CP')
    pairs = oproj_chunk_pairs(world, 'cublaslt_nccl' if te_order else 'te_ub', layout)
    send = torch.empty((world, batch, seq//world, kl), dtype=source.dtype, device=source.device)
    for peer, (a, b) in enumerate(pairs):
        send[peer, :, :chunk].copy_(local[:, a*chunk:(a+1)*chunk])
        send[peer, :, chunk:].copy_(local[:, b*chunk:(b+1)*chunk])
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send)
    del send
    return recv.permute(1, 2, 0, 3).reshape(batch*seq//world, world*kl)


def oproj_ub_check(source, weight, send, recv, output, rank, world):
    import torch
    import torch.distributed as dist
    expected_a = inverse_expected(source, rank, world)
    kl = source.shape[-1]
    self_errors = count_mismatches(send[rank], expected_a[:, rank*kl:(rank+1)*kl])
    counts = torch.zeros(world, dtype=torch.int64, device=source.device)
    for peer in range(world):
        if peer != rank:
            counts[peer] = count_mismatches(recv[peer], expected_a[:, peer*kl:(peer+1)*kl])
    expected = torch.mm(expected_a, weight.t())
    del expected_a
    error = max_abs_difference(output, expected)
    dist.all_reduce(error, op=dist.ReduceOp.MAX)
    dist.all_reduce(self_errors)
    dist.all_reduce(counts)
    check = {'max_abs': error.item(), 'self_pack_mismatches': int(self_errors.item()),
             'remote_recv_mismatches': int(counts.sum().item()), 'remote_recv_mismatches_by_peer': counts.tolist()}
    if not math.isfinite(error.item()) or error.item() > .01 or self_errors.item() or counts.sum().item():
        raise RuntimeError(f'OProj Userbuffers correctness failed: {check}')
    return check, expected


def qkv_route_check(qkv, actual, seq_local, q_width, kv_width, rank, world):
    import torch
    import torch.distributed as dist
    ql, kl = q_width//world, kv_width//world
    tile = torch.empty((min(1024, seq_local), q_width+2*kv_width), dtype=qkv.dtype, device=qkv.device)
    mismatches = torch.zeros((), dtype=torch.int64, device=qkv.device)
    for peer in range(world):
        for begin in range(0, seq_local, tile.shape[0]):
            rows = min(tile.shape[0], seq_local-begin)
            block = tile[:rows]
            if rank == peer:
                block.copy_(qkv.reshape(seq_local, -1)[begin:begin+rows])
            dist.broadcast(block, src=peer)
            for base, width, dst_base in ((rank*ql, ql, 0),
                    (q_width+rank*kl, kl, world*seq_local*ql),
                    (q_width+kv_width+rank*kl, kl, world*seq_local*(ql+kl))):
                offset = dst_base+(peer*seq_local+begin)*width
                mismatches += count_mismatches(actual.reshape(-1)[offset:offset+rows*width], block[:, base:base+width].reshape(-1))
    dist.all_reduce(mismatches)
    return int(mismatches.item())


def timed_critical(fn, warmup, iterations, device, *, use_cuda_graph):
    global LAST_MEASUREMENT_FINISHED
    import torch
    import torch.distributed as dist
    if warmup < 10 or iterations < 50:
        raise ValueError('SM103 v2 requires at least 10 warmups and 50 samples')
    # Reuse a recently exercised process, not the candidate's own warmup.
    # Long setup/JIT/idle gaps trigger a fresh elapsed-time warmup budget.
    initial_budget_ms = (100 if LAST_MEASUREMENT_FINISHED is None or
                         time.monotonic()-LAST_MEASUREMENT_FINISHED > 2 else 0)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    dist.barrier()
    graph = None
    timed_fn = fn
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

    # CUDA-event elapsed budget, not an arbitrary count of very short kernels.
    # Both convergence and minimum accumulated warmup must hold on every rank.
    warm_started = time.monotonic()
    windows = []
    accumulated_ms = 0.0
    calls = 0
    count = 10
    while True:
        start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(count):
            timed_fn()
        stop.record()
        stop.synchronize()
        elapsed = start.elapsed_time(stop)
        accumulated_ms += elapsed
        calls += count
        windows.append(elapsed / count)
        state = torch.tensor([int(accumulated_ms >= initial_budget_ms and stable_window(windows)),
                              int(time.monotonic()-warm_started >= 5)], device=device)
        ready = state[:1].clone()
        expired = state[1:].clone()
        dist.all_reduce(ready, op=dist.ReduceOp.MIN)
        dist.all_reduce(expired, op=dist.ReduceOp.MAX)
        if ready.item():
            break
        if expired.item():
            raise RuntimeError(f'warmup did not converge within 5s: last windows {windows[-3:]}')
        window_target_ms = 20 if initial_budget_ms else 1
        next_count = torch.tensor(max(10, min(1000, math.ceil(window_target_ms / max(windows[-1], .001)))), device=device)
        # A2A calls must stay in lockstep, even when ranks have different speed.
        dist.all_reduce(next_count, op=dist.ReduceOp.MIN)
        count = int(next_count.item())
    record = {'schema': 'sm103_measurement_v2', 'initial_warmup': warmup,
        'graph_replay_warmup': warmup if use_cuda_graph else 0,
        'additional_warmup_calls': calls, 'additional_warmup_cuda_ms': accumulated_ms,
        'warmup_wall_s': time.monotonic()-warm_started, 'window_ms_per_call': windows,
        'converged_all_ranks': True, 'relative_range_limit': .05,
        'minimum_warmup_cuda_ms': initial_budget_ms, 'iterations': iterations,
        'launch': 'graph' if use_cuda_graph else 'eager',
        'collector': 'primed_events_vector_max_v1'}
    def collect(count):
        return collect_samples(timed_fn, count, device)

    # Match the barrier/event cadence of measurement, not only continuous replay.
    record['sample_cadence_warmup_ms'] = collect(warmup)
    record['measurement_rounds'] = []
    RECORDS.append(record)
    for attempt in range(3):
        samples = collect(iterations)
        first = statistics.median(samples[:iterations//2])
        second = statistics.median(samples[iterations//2:])
        drift = abs(second-first)/statistics.median(samples)
        record['measurement_rounds'].append({'samples_ms': samples, 'half_p50_relative_drift': drift})
        if drift <= .05:
            break
    else:
        drifts = [round(r['half_p50_relative_drift'], 6) for r in record['measurement_rounds']]
        raise RuntimeError(f'measurement drift exceeds 5% in all 3 rounds: {drifts}; raw rounds in failure metadata')
    # Take the FIRST stable round, never the fastest. Earlier rounds remain evidence.
    record['selected_round'] = attempt
    record['first_half_p50_ms'] = first
    record['second_half_p50_ms'] = second
    record['sample_half_p50_relative_drift'] = drift
    LAST_MEASUREMENT_FINISHED = time.monotonic()
    return samples
