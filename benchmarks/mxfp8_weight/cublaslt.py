"""Lightweight binding to the unchanged repository cuBLASLt baseline ABI.

Mirrors a2a+Oproj/te_nccl_baseline.py without importing unrelated TE attention
modules. Algorithm enumeration, correctness filtering and timing stay in the
original csrc/baselines/cublaslt_runner.cu implementation.
"""
import ctypes as ct

import torch


class Info(ct.Structure):
    _fields_ = [(name, ct.c_int) for name in (
        "returned", "valid", "algo_id", "tile_id", "stages_id", "split_k",
        "reduction", "cta_swizzle", "custom", "inner_shape", "cluster_shape")]
    _fields_ += [("workspace_bytes", ct.c_uint64), ("tune_ms", ct.c_float),
                 ("waves", ct.c_float)]


class CublasLtRunner:
    def __init__(self, library, a, b, d, *, tune_warmup, tune_iters,
                 workspace_mib, sm_count_target=0):
        if any(t.dtype != torch.bfloat16 or not t.is_contiguous() for t in (a, b, d)):
            raise ValueError("cuBLASLt requires contiguous BF16 tensors")
        m, k = a.shape
        n, wk = b.shape
        if wk != k or d.shape != (m, n):
            raise ValueError("incompatible cuBLASLt matrix shapes")
        self.lib = ct.CDLL(str(library))
        self.lib.fuse_cublaslt_last_error.restype = ct.c_char_p
        create = self.lib.fuse_cublaslt_bf16_create_ex
        create.argtypes = [ct.c_int, ct.c_int64, ct.c_int64, ct.c_int64,
                           ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p,
                           ct.c_int, ct.c_int, ct.c_uint64, ct.c_int]
        create.restype = ct.c_void_p
        self.run = self.lib.fuse_cublaslt_bf16_run
        self.run.argtypes = [ct.c_void_p] * 5
        self.run.restype = ct.c_int
        query = self.lib.fuse_cublaslt_bf16_info
        query.argtypes = [ct.c_void_p, ct.POINTER(Info)]
        query.restype = ct.c_int
        self.lib.fuse_cublaslt_bf16_destroy.argtypes = [ct.c_void_p]
        self.lib.fuse_cublaslt_bf16_destroy.restype = None
        self.handle = create(a.device.index, m, n, k, a.data_ptr(), b.data_ptr(),
                             d.data_ptr(), torch.cuda.current_stream(a.device).cuda_stream,
                             tune_warmup, tune_iters, workspace_mib << 20, sm_count_target)
        if not self.handle:
            raise RuntimeError(self.lib.fuse_cublaslt_last_error().decode())
        info = Info()
        if not query(self.handle, ct.byref(info)):
            self.close()
            raise RuntimeError("cuBLASLt info query failed")
        self.info = {key: getattr(info, key) for key, _ in Info._fields_}

    def __call__(self, a, b, d):
        if not self.run(self.handle, a.data_ptr(), b.data_ptr(), d.data_ptr(),
                        torch.cuda.current_stream(a.device).cuda_stream):
            raise RuntimeError(self.lib.fuse_cublaslt_last_error().decode())
        return d

    def close(self):
        if self.handle:
            self.lib.fuse_cublaslt_bf16_destroy(self.handle)
            self.handle = None
