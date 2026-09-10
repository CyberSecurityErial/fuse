"""Pointer-only cuBLASLt ABI for training baseline GEMMs; no Torch C++ ABI."""
import ctypes as ct
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[3]
PRECISIONS = {"bf16": 16, "fp8": 8, "fp4": 4}


class Library:
    def __init__(self, path=None):
        self.path = Path(path or ROOT / "build/sm103/libfuse_sm103_cublaslt.so").resolve()
        self.lib = ct.CDLL(str(self.path))
        ptr, integer, wide = ct.c_void_p, ct.c_int, ct.c_int64
        self.lib.sm103_create.argtypes = [integer,wide,wide,wide] + [ptr]*6 + [integer]*6 + [ct.c_float]
        self.lib.sm103_create.restype = ptr
        self.lib.sm103_run.argtypes = [ptr]*5
        self.lib.sm103_run.restype = integer
        self.lib.sm103_quantize.argtypes = [integer,ptr,ptr,ptr,wide,wide,ptr]
        self.lib.sm103_quantize.restype = integer
        self.lib.sm103_destroy.argtypes = [ptr]
        self.lib.sm103_destroy.restype = None
        self.lib.sm103_plan_info.argtypes = [ptr]
        self.lib.sm103_plan_info.restype = ct.c_char_p
        self.lib.sm103_last_error.restype = ct.c_char_p

    def check(self, value):
        if value != 0:
            raise RuntimeError(self.lib.sm103_last_error().decode())


class SmBudget:
    """Own a process-local green context and its stream; no system configuration."""
    def __init__(self, library, sms):
        self.library, self.handle = library, None
        api = library.lib
        for name, args, result in (
                ('create', [ct.c_int], ct.c_void_p), ('stream', [ct.c_void_p], ct.c_void_p),
                ('info', [ct.c_void_p], ct.c_char_p), ('destroy', [ct.c_void_p], None)):
            function = getattr(api, 'sm103_sm_budget_' + name)
            function.argtypes, function.restype = args, result
        torch.cuda.synchronize()
        self.handle = api.sm103_sm_budget_create(sms)
        if not self.handle:
            raise RuntimeError(api.sm103_last_error().decode())
        try:
            self.info = json.loads(api.sm103_sm_budget_info(self.handle))
            self.stream = torch.cuda.ExternalStream(api.sm103_sm_budget_stream(self.handle), device=0)
        except Exception:
            self.close()
            raise

    def close(self):
        if self.handle:
            self.library.lib.sm103_sm_budget_destroy(self.handle)
            self.handle = None


class Operand:
    def __init__(self, library, source, precision):
        if source.ndim != 2 or source.dtype != torch.bfloat16 or not source.is_cuda:
            raise ValueError("source must be 2D CUDA BF16")
        self.transposed = not source.is_contiguous()
        if self.transposed and (precision != 'bf16' or not source.T.is_contiguous()):
            raise ValueError("only BF16 contiguous or zero-copy transpose operands are supported")
        self.library, self.source, self.precision = library, source, precision
        self.rows, self.k = source.shape
        self.scales = None
        if precision == "bf16":
            self.data = source
        elif precision == "fp8":
            self.data = torch.empty_like(source, dtype=torch.uint8)
            self.scales = torch.empty(1, device=source.device, dtype=torch.float32)
        elif precision == "fp4":
            if self.k % 16:
                raise ValueError("FP4 requires K divisible by 16")
            self.data = torch.empty((self.rows,self.k//2),device=source.device,dtype=torch.uint8)
            size = ((self.rows+127)//128)*((self.k+63)//64)*512
            self.scales = torch.zeros(size,device=source.device,dtype=torch.uint8)
        else:
            raise ValueError(precision)
        self.update()

    def update(self):
        if self.precision != "bf16":
            self.library.check(self.library.lib.sm103_quantize(
                PRECISIONS[self.precision], self.source.data_ptr(), self.data.data_ptr(),
                self.scales.data_ptr(), self.rows, self.k, torch.cuda.current_stream().cuda_stream))

    def decode(self):
        """Independent tensor expression reference; never runs in the timed region."""
        if self.precision == "bf16":
            return self.data.float()
        if self.precision == "fp8":
            return self.data.view(torch.float8_e4m3fn).float()*self.scales
        codes = torch.stack((self.data & 15, self.data >> 4), dim=-1).reshape(self.rows,self.k)
        lut = torch.tensor([0,.5,1,1.5,2,3,4,6],device=self.data.device,dtype=torch.float32)
        values = lut[(codes & 7).long()] * torch.where(codes < 8,1.0,-1.0)
        row = torch.arange(self.rows,device=self.data.device)[:,None]
        block = torch.arange(self.k//16,device=self.data.device)[None,:]
        index = ((row//128)*((self.k+63)//64)+block//4)*512+(row%32)*16+((row%128)//32)*4+block%4
        scales = self.scales.view(torch.float8_e4m3fn).float()[index]
        return (values.reshape(self.rows,-1,16)*scales[:,:,None]).reshape(self.rows,self.k)


class Plan:
    def __init__(self, library, x, weight, output, *, candidates=256, workspace_mib=256,
                 warmup=5, iterations=30, graph=False, math_sms=0, beta=0.0):
        if x.precision != weight.precision or x.k != weight.k:
            raise ValueError("GEMM operand precision/K mismatch")
        if output.shape != (x.rows, weight.rows) or output.dtype != torch.bfloat16:
            raise ValueError("output must be BF16 [M,N]")
        self.library, self.x, self.weight, self.output = library, x, weight, output
        create = library.lib.sm103_create
        extra = []
        if x.transposed or weight.transposed:
            create = library.lib.sm103_create_strided
            create.argtypes = library.lib.sm103_create.argtypes + [ct.c_int, ct.c_int]
            create.restype = ct.c_void_p
            extra = [int(x.transposed), int(weight.transposed)]
        self.handle = create(
            PRECISIONS[x.precision],x.rows,weight.rows,x.k,
            x.data.data_ptr(),weight.data.data_ptr(),output.data_ptr(),
            x.scales.data_ptr() if x.scales is not None else None,
            weight.scales.data_ptr() if weight.scales is not None else None,
            torch.cuda.current_stream().cuda_stream,candidates,workspace_mib,warmup,
            iterations,int(graph),math_sms,ct.c_float(beta),*extra)
        if not self.handle:
            raise RuntimeError(library.lib.sm103_last_error().decode())
        self.info = json.loads(library.lib.sm103_plan_info(self.handle))

    def run(self):
        self.library.check(self.library.lib.sm103_run(self.handle,self.x.data.data_ptr(),
            self.weight.data.data_ptr(),self.output.data_ptr(),torch.cuda.current_stream().cuda_stream))
        return self.output

    def close(self):
        if self.handle:
            self.library.lib.sm103_destroy(self.handle)
            self.handle = None


def check_gemm(plan, tolerance=0.02, *, full=False):
    if full:
        m, n, k = plan.x.rows, plan.weight.rows, plan.x.k
        if (plan.x.precision != 'bf16' or plan.weight.precision != 'bf16'
                or max(m * n, m * k, n * k) > 4_194_304):
            raise ValueError('Full GEMM check requires small BF16 buffers (each at most 4194304 elements)')
    plan.run()
    # The default remains a deterministic 64x64 sample. Explicit small BF16
    # diagnostics cover every row/column, including tails and padded clusters.
    if full:
        rows = torch.arange(plan.x.rows, device=plan.output.device)
        cols = torch.arange(plan.weight.rows, device=plan.output.device)
    else:
        rows = torch.linspace(0,plan.x.rows-1,min(64,plan.x.rows),device=plan.output.device).long()
        cols = torch.linspace(0,plan.weight.rows-1,min(64,plan.weight.rows),device=plan.output.device).long()
    torch.backends.cuda.matmul.allow_tf32 = False
    # Select BF16 rows before conversion, keeping long-K backward checks
    # bounded without materializing either full transposed operand.
    if plan.x.precision == plan.weight.precision == 'bf16':
        ref = plan.x.data[rows].float() @ plan.weight.data[cols].float().T
    else:
        ref = plan.x.decode()[rows] @ plan.weight.decode()[cols].T
    actual = plan.output[rows[:,None],cols].float()
    difference = actual-ref
    relative_rms = (difference.square().mean().sqrt()/ref.square().mean().sqrt().clamp_min(1e-12)).item()
    maximum = difference.abs().max().item()
    if not torch.isfinite(actual).all() or relative_rms > tolerance:
        raise RuntimeError(f"GEMM correctness failure: relative_rms={relative_rms}, max_abs={maximum}")
    result = {"relative_rms":relative_rms,"max_abs":maximum,"checked_values":actual.numel()}
    if full:
        # Evidence comes from the actual checked tensor, not the CLI request.
        result['full_output_checked'] = actual.numel() == plan.x.rows * plan.weight.rows
        if not result['full_output_checked'] or not torch.isfinite(ref).all():
            raise RuntimeError('Full GEMM check did not cover a finite reference for every output')
    return result
