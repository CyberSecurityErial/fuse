"""Standalone GPU gate: transpose layouts, autograd dX/dW, beta and graph."""
import json
import torch
from backward_gemm import Gemm


def main():
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    records = []
    for m, n, k in ((128, 256, 96), (256, 192, 320)):
        x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16) * .0625
        w = torch.randn((n, k), device='cuda', dtype=torch.bfloat16) * .0625
        dy = torch.randn((m, n), device='cuda', dtype=torch.bfloat16) * .0625
        # Actual autograd from independent FP32 linear with exact BF16 values.
        rx, rw = x.float().requires_grad_(), w.float().requires_grad_()
        torch.nn.functional.linear(rx, rw).backward(dy.float())
        for dtype in (torch.bfloat16, torch.float32):
            out = torch.empty((n, k), device='cuda', dtype=dtype)
            plan = Gemm(dy, x, out, ta=True, beta=1, candidates=16)
            for launch in ('eager', 'graph'):
                initial = torch.randn_like(out) * .01
                out.copy_(initial)
                fn = lambda: plan(dy, x, out, beta=1)
                fn(); torch.cuda.synchronize()
                if launch == 'graph':
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        fn()
                    fn = g.replay
                out.copy_(initial)
                expected = initial.float()
                for _ in range(2):
                    fn()
                    expected = (expected + rw.grad).to(dtype).float()
                    torch.testing.assert_close(out.float(), expected, atol=.004 if dtype == torch.bfloat16 else 2e-6,
                                               rtol=.008 if dtype == torch.bfloat16 else 2e-5)
                plan(dy, x, out, beta=0)
                torch.testing.assert_close(out.float(), rw.grad.to(dtype).float(),
                                           atol=.002 if dtype == torch.bfloat16 else 2e-6,
                                           rtol=.008 if dtype == torch.bfloat16 else 2e-5)
                records.append(dict(stage='wgrad', shape=[m, n, k], dtype=str(dtype),
                                    launch=launch, plan=plan.info, passed=True))
            plan.close()
        dx = torch.empty_like(x)
        plan = Gemm(dy, w, dx)
        plan(dy, w, dx)
        torch.testing.assert_close(dx.float(), rx.grad.to(torch.bfloat16).float(), atol=.002, rtol=.008)
        records.append(dict(stage='dgrad', shape=[m, n, k], plan=plan.info, passed=True))
        plan.close()
    print(json.dumps(dict(passed=True, records=records), indent=2))


if __name__ == '__main__':
    main()
