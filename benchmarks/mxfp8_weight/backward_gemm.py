"""Benchmark-only logical NN/TN/NT BF16 GEMMs with BF16/FP32 output."""
import ctypes as ct
from pathlib import Path
import torch

LIBRARY = Path(__file__).resolve().parents[2] / 'build-mxfp8-bench/libmxfp8_backward.so'


class Gemm:
    def __init__(self, a, b, out, *, ta=False, tb=False, beta=0,
                 candidates=32, warmup=2, iterations=6, library=LIBRARY):
        if any(not t.is_contiguous() or t.device != a.device for t in (a, b, out)):
            raise ValueError('physical tensors must be contiguous on one GPU')
        if not a.is_cuda or a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
            raise ValueError('inputs must be CUDA BF16')
        if out.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError('output must be BF16 or FP32')
        m, k = a.shape[::-1] if ta else a.shape
        kb, n = b.shape[::-1] if tb else b.shape
        if k != kb or out.shape != (m, n):
            raise ValueError('incompatible matrix dimensions')
        self.signature = [(tuple(t.shape), t.dtype, t.device) for t in (a, b, out)]
        self.lib = ct.CDLL(str(library))
        self.lib.mxfp8_bwd_error.restype = ct.c_char_p
        create = self.lib.mxfp8_bwd_create
        create.argtypes = [ct.c_int64]*3 + [ct.c_int]*3 + [ct.c_void_p]*4 + [ct.c_float] + [ct.c_int]*3
        create.restype = ct.c_void_p
        self.lib.mxfp8_bwd_run.argtypes = [ct.c_void_p]*5 + [ct.c_float]
        self.lib.mxfp8_bwd_run.restype = ct.c_int
        self.lib.mxfp8_bwd_destroy.argtypes = [ct.c_void_p]
        self.lib.mxfp8_bwd_destroy.restype = None
        self.lib.mxfp8_bwd_info.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int),
                                          ct.POINTER(ct.c_float), ct.POINTER(ct.c_uint64)]
        self.lib.mxfp8_bwd_info.restype = ct.c_int
        with torch.cuda.device(a.device):
            self.handle = create(m, n, k, ta, tb, out.dtype == torch.float32,
                                 a.data_ptr(), b.data_ptr(), out.data_ptr(),
                                 torch.cuda.current_stream(a.device).cuda_stream,
                                 beta, candidates, warmup, iterations)
        if not self.handle:
            raise RuntimeError(self.lib.mxfp8_bwd_error().decode())
        ints, timing, workspace = (ct.c_int*3)(), ct.c_float(), ct.c_uint64()
        self.lib.mxfp8_bwd_info(self.handle, ints, ct.byref(timing), ct.byref(workspace))
        self.info = dict(mnk=[m, n, k], ta=ta, tb=tb, output_dtype=str(out.dtype),
                         tune_beta=beta, returned=ints[0], valid=ints[1], algo_id=ints[2],
                         tune_us=timing.value, workspace_bytes=workspace.value,
                         candidates=candidates)

    def __call__(self, a, b, out, beta=0):
        if not self.lib.mxfp8_bwd_run(self.handle, a.data_ptr(), b.data_ptr(), out.data_ptr(),
                                    torch.cuda.current_stream(a.device).cuda_stream, beta):
            raise RuntimeError(self.lib.mxfp8_bwd_error().decode())
        return out

    def close(self):
        if self.handle:
            self.lib.mxfp8_bwd_destroy(self.handle)
            self.handle = None
