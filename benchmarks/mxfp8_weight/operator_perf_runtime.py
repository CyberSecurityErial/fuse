"""Distributed storage, references and timing for the MXFP8 production operators.

Only cudaMalloc allocations are exported over CUDA IPC. PyTorch's caching
allocator never owns an exported allocation. All CPU collectives use Gloo;
there are no NCCL operations hidden in the measured Fuse path.
"""

import ctypes as ct
import ctypes.util
import math

import numpy as np
import torch
import torch.distributed as dist
import triton
import triton.language as tl


class CudaRuntime:
    def __init__(self):
        name = ctypes.util.find_library('cudart') or 'libcudart.so.12'
        self.lib = ct.CDLL(name)
        signatures = {
            'cudaMalloc': [ct.POINTER(ct.c_void_p), ct.c_size_t],
            'cudaFree': [ct.c_void_p],
            'cudaIpcGetMemHandle': [ct.c_void_p, ct.c_void_p],
            'cudaIpcOpenMemHandle': [ct.POINTER(ct.c_void_p), IpcHandle, ct.c_uint],
            'cudaIpcCloseMemHandle': [ct.c_void_p],
            'cudaMemcpyAsync': [ct.c_void_p, ct.c_void_p, ct.c_size_t,
                                ct.c_int, ct.c_void_p],
            'cudaMemsetAsync': [ct.c_void_p, ct.c_int, ct.c_size_t, ct.c_void_p],
        }
        for function, arguments in signatures.items():
            getattr(self.lib, function).argtypes = arguments
            getattr(self.lib, function).restype = ct.c_int
        self.lib.cudaGetErrorString.argtypes = [ct.c_int]
        self.lib.cudaGetErrorString.restype = ct.c_char_p

    def check(self, status):
        if status:
            message = self.lib.cudaGetErrorString(status).decode()
            raise RuntimeError(f'CUDA runtime error {status}: {message}')

    def copy(self, destination, source, size, stream):
        self.check(self.lib.cudaMemcpyAsync(destination, source, size, 3, stream))

    def zero(self, pointer, size, stream):
        self.check(self.lib.cudaMemsetAsync(pointer, 0, size, stream))


class IpcHandle(ct.Structure):
    _fields_ = [('reserved', ct.c_byte * 64)]


class PeerArena:
    """One allocation per rank, with disjoint data and epoch-control regions."""

    def __init__(self, runtime, data_bytes, ready_elements, stream):
        self.runtime, self.stream = runtime, stream
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        align = lambda size: (size + 255) // 256 * 256
        self.data_bytes = data_bytes
        self.ready_bytes = ready_elements * 4
        self.done_bytes = self.world * 32 * 4
        self.offsets = {'data': 0, 'ready': align(data_bytes),
                        'done': align(data_bytes) + align(self.ready_bytes)}
        self.bytes = self.offsets['done'] + align(self.done_bytes)
        pointer = ct.c_void_p()
        runtime.check(runtime.lib.cudaMalloc(ct.byref(pointer), self.bytes))
        self.local = pointer.value
        self.opened = []
        handle = IpcHandle()
        runtime.check(runtime.lib.cudaIpcGetMemHandle(ct.byref(handle), self.local))
        handles = [None] * self.world
        dist.all_gather_object(handles, bytes(handle))
        self.bases = []
        for rank, encoded in enumerate(handles):
            if rank == self.rank:
                self.bases.append(self.local)
                continue
            remote = ct.c_void_p()
            # cudaIpcMemLazyEnablePeerAccess. CUDA IPC manages peer mappings;
            # do not enable access to private Torch allocations unnecessarily.
            runtime.check(runtime.lib.cudaIpcOpenMemHandle(
                ct.byref(remote), IpcHandle.from_buffer_copy(encoded), 1))
            self.bases.append(remote.value)
            self.opened.append(remote.value)
        self.reset_control()
        stream.synchronize()
        dist.barrier()

    def pointer(self, region, rank=None):
        return self.bases[self.rank if rank is None else rank] + self.offsets[region]

    def reset_control(self):
        stream = self.stream.cuda_stream
        self.runtime.zero(self.pointer('ready'), self.ready_bytes, stream)
        self.runtime.zero(self.pointer('done'), self.done_bytes, stream)

    def write_data(self, source):
        assert source.is_contiguous() and source.numel() * source.element_size() == self.data_bytes
        self.runtime.copy(self.pointer('data'), source.data_ptr(), self.data_bytes,
                          self.stream.cuda_stream)

    def read_data(self, destination):
        assert destination.is_contiguous()
        assert destination.numel() * destination.element_size() == self.data_bytes
        self.runtime.copy(destination.data_ptr(), self.pointer('data'), self.data_bytes,
                          self.stream.cuda_stream)

    def close(self):
        # No peer closes a mapping while another rank can still write it.
        self.stream.synchronize()
        dist.barrier()
        for pointer in self.opened:
            self.runtime.check(self.runtime.lib.cudaIpcCloseMemHandle(pointer))
        dist.barrier()
        self.runtime.check(self.runtime.lib.cudaFree(self.local))
        self.opened.clear()


def deterministic(shape, seed, device):
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.empty(shape, dtype=torch.bfloat16, device=device).uniform_(
        -0.09375, 0.09375, generator=generator)


def quantize_offline(weight):
    """The published baseline's original-weight-axis E4M3/E8M0 recipe."""
    q = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    scales = torch.empty((weight.shape[0], weight.shape[1] // 32),
                         dtype=torch.uint8, device=weight.device)
    for first in range(0, weight.shape[0], 128):
        block = weight[first:first + 128].float().reshape(-1, weight.shape[1] // 32, 32)
        exponent = torch.ceil(torch.log2(block.abs().amax(-1) / 448)).clamp(-127, 127)
        values = (block / torch.exp2(exponent).unsqueeze(-1)).clamp(-448, 448)
        q[first:first + 128].copy_(values.to(torch.float8_e4m3fn).reshape(-1, weight.shape[1]))
        scales[first:first + 128].copy_((exponent + 127).to(torch.uint8))
    return q, scales


def reference_dequant(q, scales):
    effective = torch.empty_like(q, dtype=torch.bfloat16)
    for first in range(0, q.shape[0], 128):
        block = q[first:first + 128].float().reshape(-1, q.shape[1] // 32, 32)
        scale = torch.exp2(scales[first:first + 128].float() - 127)
        effective[first:first + 128].copy_((block * scale.unsqueeze(-1)).reshape(-1, q.shape[1]))
    return effective


def global_rows(rank, local_tokens, world, causal, device):
    local = torch.arange(local_tokens, device=device)
    if not causal:
        return local + rank * local_tokens
    half = local_tokens // 2
    return torch.where(local < half, rank * half + local,
                       (2 * world - rank - 1) * half + local - half)


@triton.jit
def _error_partials(actual, expected, out, ELEMENTS: tl.constexpr, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    index = block.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(actual + index, index < ELEMENTS, 0).to(tl.float32)
    y = tl.load(expected + index, index < ELEMENTS, 0).to(tl.float32)
    delta = x - y
    invalid = (x != x) | (y != y) | (tl.abs(x) == float('inf')) | (tl.abs(y) == float('inf'))
    tl.store(out + block * 4, tl.max(tl.abs(delta), 0))
    tl.store(out + block * 4 + 1, tl.sum(delta * delta, 0))
    tl.store(out + block * 4 + 2, tl.sum(y * y, 0))
    tl.store(out + block * 4 + 3, tl.max(invalid.to(tl.float32), 0))


def check_error(actual, expected, *, exact=False):
    """All-rank finite gate, including ranks whose peers happen to be finite."""
    assert actual.shape == expected.shape and actual.is_contiguous() and expected.is_contiguous()
    count = triton.cdiv(actual.numel(), 4096)
    partial = torch.empty((count, 4), dtype=torch.float32, device=actual.device)
    _error_partials[(count,)](actual, expected, partial, actual.numel(), 4096)
    values = torch.stack((partial[:, 0].amax(),
                          (partial[:, 1].sum() / partial[:, 2].sum().clamp_min(1e-30)).sqrt(),
                          partial[:, 3].amax())).cpu().tolist()
    ranks = [None] * dist.get_world_size()
    dist.all_gather_object(ranks, values)
    finite = all(all(math.isfinite(x) for x in rank) and rank[2] == 0 for rank in ranks)
    result = dict(max_abs=max(rank[0] for rank in ranks),
                  relative_rmse=max(rank[1] for rank in ranks), all_ranks_finite=finite)
    limit = 1e-4 if actual.dtype == torch.float32 else 0.005
    if not finite or result['relative_rmse'] > limit or (exact and result['max_abs'] != 0):
        raise RuntimeError(f'Independent reference failed: {result}; exact={exact}')
    return result


def aligned_prepare(prepare, stream):
    prepare()
    stream.synchronize()
    # Every rank finishes clearing its own flags before any peer may signal.
    dist.barrier()


class Invocation:
    """Warm and optionally capture once; all allocation stays outside samples."""

    def __init__(self, function, launch, warmup, prepare, stream):
        self.stream, self.prepare = stream, prepare
        self.graph = None
        with torch.cuda.stream(stream):
            for _ in range(max(warmup, 2)):
                aligned_prepare(prepare, stream)
                function()
                stream.synchronize()
            if launch == 'graph':
                aligned_prepare(prepare, stream)
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph, stream=stream):
                    function()
                self.call = self.graph.replay
                for _ in range(max(warmup, 2)):
                    aligned_prepare(prepare, stream)
                    self.call()
                    stream.synchronize()
            else:
                self.call = function
        self.events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        for event in self.events:
            event.record(stream)
        self.events[-1].synchronize()

    def once(self, prepare=None):
        with torch.cuda.stream(self.stream):
            aligned_prepare(prepare or self.prepare, self.stream)
            self.call()
            self.stream.synchronize()

    def measure(self, iterations, flops):
        samples = []
        start, stop = self.events
        with torch.cuda.stream(self.stream):
            for _ in range(iterations):
                aligned_prepare(self.prepare, self.stream)
                start.record(self.stream)
                self.call()
                stop.record(self.stream)
                stop.synchronize()
                samples.append(start.elapsed_time(stop) * 1000)
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, samples)
        array = np.asarray(gathered)
        if array.shape != (dist.get_world_size(), iterations) or not np.isfinite(array).all() or (array <= 0).any():
            raise RuntimeError('Invalid per-rank duration vectors')
        values = array.max(axis=0)
        p50 = float(np.median(values))
        return dict(p50_us=p50, p95_us=float(np.percentile(values, 95)),
                    mean_us=float(values.mean()), samples_us=values.tolist(),
                    rank_samples_us=gathered, flops_per_gpu=flops,
                    tflops_per_gpu=flops / p50 / 1e6)
