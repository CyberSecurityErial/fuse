#!/usr/bin/env python3
"""One real CUDA/Triton JIT check. Does not certify distributed UB correctness."""
import json
import subprocess
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl
import transformer_engine.pytorch as te
import transformer_engine_torch as tex


@triton.jit
def add_one(x, y, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(y + i, tl.load(x + i, i < N, other=0) + 1, i < N)


def main():
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
    gpu = subprocess.check_output([
        'nvidia-smi', '--id=0', '--query-gpu=utilization.gpu,memory.free',
        '--format=csv,noheader,nounits'], text=True).strip().split(',')
    if float(gpu[0]) > 5 or float(gpu[1]) < 2048:
        raise RuntimeError(f'GPU0 unavailable for JIT check: utilization/free MiB={gpu}')
    p = torch.cuda.get_device_properties(0)
    if (p.major, p.minor) != (10, 3):
        raise RuntimeError(f'Expected SM103 CUDA runtime, got {p.major}.{p.minor}')
    x = torch.arange(1024, device='cuda', dtype=torch.float32)
    y = torch.empty_like(x)
    add_one[(4,)](x, y, 1024, BLOCK=256)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, x + 1, rtol=0, atol=0)
    methods = ('configure_userbuffers_p2p', 'userbuffers_p2p_send',
               'userbuffers_p2p_recv', 'get_userbuffers_send_stream')
    result = dict(environment_imports='passed', triton_jit='passed',
                  gpu_before={'utilization_pct': float(gpu[0]), 'free_mib': float(gpu[1])},
                  runtime_compute=f'{p.major}.{p.minor}', sm_count=p.multi_processor_count,
                  ub_bindings={m: hasattr(tex.CommOverlapP2P, m) for m in methods},
                  ub_distributed_correctness='not tested by doctor',
                  te_module=te.__file__, tex_module=tex.__file__)
    Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
