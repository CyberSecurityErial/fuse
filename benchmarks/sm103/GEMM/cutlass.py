"""Optional pointer-only 1-SM/2-SM BF16 plans; no Torch C++ ABI or search."""
import ctypes as ct
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]


class Library:
    def __init__(self, path=None):
        self.path = Path(path or ROOT / 'build/sm103-cutlass/libfuse_sm103_cutlass_bf16.so').resolve()
        self.lib = ct.CDLL(str(self.path))
        ptr = ct.c_void_p
        self.lib.sm103_cutlass_create_v5.argtypes = [ct.c_int] * 5 + [ct.c_int64] * 3 + [ptr] * 4
        self.lib.sm103_cutlass_create_v5.restype = ptr
        self.lib.sm103_cutlass_run.argtypes = [ptr, ptr]
        self.lib.sm103_cutlass_run.restype = ct.c_int
        self.lib.sm103_cutlass_plan_info.argtypes = [ptr]
        self.lib.sm103_cutlass_plan_info.restype = ct.c_char_p
        self.lib.sm103_cutlass_destroy.argtypes = [ptr]
        self.lib.sm103_cutlass_destroy.restype = None
        self.lib.sm103_cutlass_last_error.argtypes = []
        self.lib.sm103_cutlass_last_error.restype = ct.c_char_p

    def error(self):
        message = self.lib.sm103_cutlass_last_error()
        return message.decode() if message else 'CUTLASS plan failure without an error message'

    def check(self, status):
        if status != 0:
            raise RuntimeError(self.error())


class Plan:
    """Bind fixed CUDA buffers; construct outside timing/capture, close after use.

    x/weight are existing BF16 cublaslt.Operand objects, so check_gemm() works
    unchanged. Their contents may change; their pointers/device/layout may not.
    Keep the plan and tensors alive until every using stream/graph has finished.
    """
    def __init__(self, library, x, weight, output, *, sm_mode=1, max_swizzle_size=1,
                 epilogue_n=32, cluster_m=None, sm_budget=0):
        self.handle = None
        if type(sm_budget) is not int or not 0 <= sm_budget <= 2**31 - 1:
            raise ValueError('sm_budget must be zero (all) or a positive int32')
        if type(sm_mode) is not int or sm_mode not in (1, 2):
            raise ValueError('sm_mode must be 1 or 2')
        if type(max_swizzle_size) is not int or max_swizzle_size not in (1, 2, 4, 8):
            raise ValueError('max_swizzle_size must be 1, 2, 4 or 8')
        if type(epilogue_n) is not int or epilogue_n not in (32, 64):
            raise ValueError('epilogue_n must be 32 or 64')
        if cluster_m is None:
            cluster_m = sm_mode
        if (type(cluster_m) is not int or cluster_m not in (1, 2)
                or (sm_mode == 2 and cluster_m != 2)):
            raise ValueError('cluster_m must be 1 or 2 for 1-SM MMA, and 2 for 2-SM MMA')
        if sm_mode == 1 and cluster_m == 2 and epilogue_n != 32:
            raise ValueError('1-SM cluster2 comparison requires epilogue_n=32')
        if x.precision != 'bf16' or weight.precision != 'bf16' or x.k != weight.k:
            raise ValueError('CUTLASS plans require BF16 operands with identical K')
        shapes = ((x.rows, x.k), (weight.rows, weight.k), (x.rows, weight.rows))
        tensors = (x.data, weight.data, output)
        for tensor, shape in zip(tensors, shapes):
            if (tensor.ndim != 2 or tuple(tensor.shape) != shape or tensor.dtype != torch.bfloat16
                    or not tensor.is_cuda or not tensor.is_contiguous()):
                raise ValueError('CUTLASS buffers must be contiguous 2D CUDA BF16 with matching shape')
            if any(type(dim) is not int or not 1 <= dim <= 2**31 - 1 for dim in shape):
                raise ValueError('CUTLASS dimensions must be positive int32 values')
        if any(tensor.device != output.device for tensor in tensors):
            raise ValueError('CUTLASS buffers must share a device')
        if output.device.index != torch.cuda.current_device():
            raise ValueError('Set the current CUDA device before creating a CUTLASS plan')
        self.library, self.x, self.weight, self.output = library, x, weight, output
        self._buffers = tensors  # Keep the allocations alive independently of Operand.data.
        self._pointers = tuple(tensor.data_ptr() for tensor in tensors)
        self.handle = library.lib.sm103_cutlass_create_v5(
            sm_mode, max_swizzle_size, epilogue_n, cluster_m, sm_budget, x.rows, weight.rows, x.k, *self._pointers,
            torch.cuda.current_stream().cuda_stream)
        if not self.handle:
            raise RuntimeError(library.error())
        try:
            raw = library.lib.sm103_cutlass_plan_info(self.handle)
            if not raw:
                raise RuntimeError(library.error())
            try:
                self.info = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                # Native plan metadata contains only geometry/resource fields.
                # Keep a bounded, escaped prefix, not a bare decoder position.
                excerpt = raw[:1024].decode('utf-8', errors='backslashreplace')
                raise ValueError(f'Invalid CUTLASS plan JSON: raw_bytes={len(raw)}, '
                                 f'raw_prefix={excerpt!r}') from error
            if (self.info.get('schema') != 'sm103_cutlass_bf16_plan_v1'
                    or any(type(self.info.get(field)) is not int
                           for field in ('sm_mode', 'cluster_m', 'cluster_size', 'mma_sm_count'))
                    or self.info.get('sm_mode') != sm_mode
                    or self.info.get('max_swizzle_size') != max_swizzle_size
                    or self.info.get('epilogue_n') != epilogue_n
                    or self.info.get('epilogue_tile') != [128, epilogue_n]
                    or self.info.get('cluster_m') != cluster_m
                    or self.info.get('cluster') != [cluster_m, 1, 1]
                    or self.info.get('cluster_size') != cluster_m
                    or self.info.get('mma_sm_count') != sm_mode
                    or self.info.get('backend') != f'cutlass_{sm_mode}sm'
                    or (self.info.get('m'), self.info.get('n'), self.info.get('k'))
                    != (x.rows, weight.rows, x.k)):
                raise ValueError('CUTLASS plan metadata differs from the requested geometry/mode')
            effective = self.info.get('effective_swizzle_size')
            padded = self.info.get('padded_work_grid_ctas')
            if (type(effective) is not int or effective not in (1, 2, 4, 8)
                    or effective > max_swizzle_size
                    or not isinstance(padded, list) or len(padded) != 3
                    or any(type(value) is not int or value <= 0 for value in padded)
                    or padded[0] % (effective * cluster_m) or padded[1] % effective
                    or padded[0] * padded[1] * padded[2] != self.info.get('padded_work_ctas')):
                raise ValueError('CUTLASS plan lowered swizzle/work-grid metadata is inconsistent')
            grid = self.info.get('grid')
            physical = self.info.get('physical_sm_count')
            budget = self.info.get('requested_sm_budget')
            active_clusters = self.info.get('active_clusters')
            if (not isinstance(grid, list) or len(grid) != 3
                    or any(type(value) is not int or value <= 0 for value in grid)
                    or type(physical) is not int or physical <= 0
                    or type(budget) is not int or not 0 < budget <= physical
                    or budget != (sm_budget or physical)
                    or budget % cluster_m
                    or grid[0] % cluster_m or grid[1:] != [1, 1]
                    or grid[0] != self.info.get('grid_ctas') or grid[0] > budget
                    or (cluster_m > 1 and (type(active_clusters) is not int or active_clusters <= 0
                                          or grid[0] // cluster_m > active_clusters))
                    or (cluster_m == 1 and active_clusters is not None)):
                raise ValueError('CUTLASS plan actual grid/cluster residency is inconsistent')
        except Exception:
            self.close()
            raise

    def run(self):
        if not self.handle:
            raise RuntimeError('CUTLASS plan is closed')
        if tuple(tensor.data_ptr() for tensor in (self.x.data, self.weight.data, self.output)) != self._pointers:
            raise RuntimeError('CUTLASS plan buffers changed; create a new plan outside timing/capture')
        self.library.check(self.library.lib.sm103_cutlass_run(
            self.handle, torch.cuda.current_stream().cuda_stream))
        return self.output

    def close(self):
        if self.handle:
            self.library.lib.sm103_cutlass_destroy(self.handle)
            self.handle = None
