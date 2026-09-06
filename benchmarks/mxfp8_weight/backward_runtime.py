"""Frozen, import-safe benchmark helpers; production and completed forward files unchanged.

Routing kernels copied verbatim from backward/backward_te_nccl_baseline.py.
DQ, Userbuffers, measurement and finite gates from the completed forward bench.py.
Copies avoid importing CLI entry points or missing unrelated TE pytorch modules.
"""
import torch
import torch.distributed as dist
import numpy as np
import transformer_engine
import transformer_engine_torch as tex
import triton
import triton.language as tl

@triton.jit
def dequant_kernel(q, scales, out, E: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(q + i, i < E, 0.0).to(tl.float32)
    exponent = tl.load(scales + i // 32, i < E, 127).to(tl.int32) - 127
    tl.store(out + i, x * tl.exp2(exponent.to(tl.float32)), i < E)


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
    limit = 0.0001 if actual.dtype == torch.float32 else 0.005
    if invalid.item() or result["relative_rmse"] > limit:
        raise RuntimeError(f"BF16 GEMM reference failed: {result}")
    return result


@triton.jit
def _qkv_inverse_pack_kernel(
    grad_q,
    grad_k,
    grad_v,
    packed,
    elements: tl.constexpr,
    local_tokens: tl.constexpr,
    sequence_local: tl.constexpr,
    world: tl.constexpr,
    q_local_width: tl.constexpr,
    kv_local_width: tl.constexpr,
    peer_width: tl.constexpr,
    causal: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offsets < elements
    within = offsets % peer_width
    row = (offsets // peer_width) % local_tokens
    owner = offsets // (peer_width * local_tokens)
    batch_index = row // sequence_local
    row_in_batch = row % sequence_local
    if causal:
        chunk_rows = sequence_local // 2
        chunk = tl.where(row_in_batch < chunk_rows, owner, 2 * world - owner - 1)
        global_row = (
            batch_index * sequence_local * world
            + chunk * chunk_rows
            + row_in_batch % chunk_rows
        )
    else:
        global_row = (
            batch_index * sequence_local * world
            + owner * sequence_local
            + row_in_batch
        )
    is_q = within < q_local_width
    is_k = (within >= q_local_width) & (within < q_local_width + kv_local_width)
    is_v = ~is_q & ~is_k
    q_source = global_row * q_local_width + within
    kv_column = tl.where(
        is_k,
        within - q_local_width,
        within - q_local_width - kv_local_width,
    )
    kv_source = global_row * kv_local_width + kv_column
    # Triton evaluates both sides of tl.where.  Branch-specific masks are
    # required: using one selected address for three unconditional loads lets
    # Q lanes read past the smaller K/V allocations at long sequence lengths.
    q_value = tl.load(grad_q + q_source, mask=mask & is_q, other=0.0)
    k_value = tl.load(grad_k + kv_source, mask=mask & is_k, other=0.0)
    v_value = tl.load(grad_v + kv_source, mask=mask & is_v, other=0.0)
    value = tl.where(is_q, q_value, tl.where(is_k, k_value, v_value))
    tl.store(packed + offsets, value, mask=mask)


@triton.jit
def _qkv_inverse_unpack_kernel(
    received,
    dqkv,
    elements: tl.constexpr,
    local_tokens: tl.constexpr,
    world: tl.constexpr,
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
    tl.store(dqkv + offsets, tl.load(received + source, mask=mask), mask=mask)


@triton.jit
def _oproj_route_pack_kernel(
    local_da,
    packed,
    elements: tl.constexpr,
    local_tokens: tl.constexpr,
    attention_width: tl.constexpr,
    local_width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offsets < elements
    column = offsets % local_width
    row = (offsets // local_width) % local_tokens
    destination = offsets // (local_width * local_tokens)
    source = row * attention_width + destination * local_width + column
    tl.store(packed + offsets, tl.load(local_da + source, mask=mask), mask=mask)


@triton.jit
def _oproj_route_unpack_kernel(
    received,
    output,
    elements: tl.constexpr,
    local_tokens: tl.constexpr,
    sequence_local: tl.constexpr,
    world: tl.constexpr,
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
    destination = global_row * local_width + column
    tl.store(output + destination, tl.load(received + offsets, mask=mask), mask=mask)


def deterministic(shape: tuple[int, ...], seed: int, device: torch.device) -> torch.Tensor:
    # Do not materialize an int64 arange: the largest formal 405B case has
    # billions of elements and that temporary alone would consume tens of
    # GiB.  A per-device generator produces the same deterministic BF16 input
    # with no storage beyond the tensor being initialized.
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.empty(shape, dtype=torch.bfloat16, device=device).uniform_(
        -0.09375,
        0.09375,
        generator=generator,
    )


def global_rows(
    *,
    rank: int,
    local_tokens: int,
    batch: int,
    world: int,
    causal: bool,
    device: torch.device,
) -> torch.Tensor:
    sequence_local = local_tokens // batch
    result: list[int] = []
    for row in range(local_tokens):
        batch_index, local_row = divmod(row, sequence_local)
        if causal:
            chunk_rows = sequence_local // 2
            chunk = rank if local_row < chunk_rows else 2 * world - rank - 1
            index = (
                batch_index * sequence_local * world
                + chunk * chunk_rows
                + local_row % chunk_rows
            )
        else:
            index = (
                batch_index * sequence_local * world
                + rank * sequence_local
                + local_row
            )
        result.append(index)
    return torch.tensor(result, dtype=torch.int64, device=device)
