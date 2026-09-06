"""Preallocated classic cuBLAS GEMM reference, explicitly not cuBLASLt."""
import ctypes as ct
from pathlib import Path

import torch

LIBRARY = Path(__file__).resolve().parents[2] / 'build-mxfp8-bench/libmxfp8_classic_gemm.so'


class ClassicGemm:
    def __init__(self, a, b, out, *, ta=False, tb=False, beta=0.0, library=LIBRARY):
        self.handle = None
        if any(t.ndim != 2 or not t.is_contiguous() or t.device != a.device
               for t in (a, b, out)):
            raise ValueError('physical matrices must be contiguous on one GPU')
        if not a.is_cuda or a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
            raise ValueError('inputs must be CUDA BF16')
        if out.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError('output must be BF16 or FP32')
        m, k = a.shape[::-1] if ta else a.shape
        kb, n = b.shape[::-1] if tb else b.shape
        if k != kb or out.shape != (m, n) or max(m, n, k) >= 2**31:
            raise ValueError('incompatible or out-of-range matrix dimensions')
        self.a, self.b, self.out = a, b, out
        self.beta = float(beta)
        self.lib = ct.CDLL(str(library))
        self.lib.mxfp8_classic_error.restype = ct.c_char_p
        self.lib.mxfp8_classic_create.argtypes = [ct.c_int]*6 + [ct.c_void_p]
        self.lib.mxfp8_classic_create.restype = ct.c_void_p
        self.lib.mxfp8_classic_run.argtypes = [ct.c_void_p]*5 + [ct.c_float]
        self.lib.mxfp8_classic_run.restype = ct.c_int
        self.lib.mxfp8_classic_destroy.argtypes = [ct.c_void_p]
        self.lib.mxfp8_classic_destroy.restype = None
        self.lib.mxfp8_classic_version.argtypes = [ct.c_void_p]
        self.lib.mxfp8_classic_version.restype = ct.c_int
        with torch.cuda.device(a.device):
            self.handle = self.lib.mxfp8_classic_create(
                m, n, k, ta, tb, out.dtype == torch.float32,
                torch.cuda.current_stream(a.device).cuda_stream)
        if not self.handle:
            raise RuntimeError(self.lib.mxfp8_classic_error().decode())
        self.config = dict(backend='classic_cublas', api='cublasGemmEx',
                           version=self.lib.mxfp8_classic_version(self.handle),
                           mnk=[m, n, k], ta=ta, tb=tb, input_dtype='torch.bfloat16',
                           output_dtype=str(out.dtype), beta=self.beta,
                           compute_type='CUBLAS_COMPUTE_32F',
                           algorithm='CUBLAS_GEMM_DEFAULT_TENSOR_OP',
                           workspace_bytes=32 << 20,
                           boundary='GEMM only; no dequantization or A2A')

    def __call__(self):
        if not self.lib.mxfp8_classic_run(
                self.handle, self.a.data_ptr(), self.b.data_ptr(), self.out.data_ptr(),
                torch.cuda.current_stream(self.a.device).cuda_stream, self.beta):
            raise RuntimeError(self.lib.mxfp8_classic_error().decode())
        return self.out

    def close(self):
        if self.handle:
            with torch.cuda.device(self.a.device):
                self.lib.mxfp8_classic_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()
