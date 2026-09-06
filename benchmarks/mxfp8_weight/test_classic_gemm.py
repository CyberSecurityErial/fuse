"""Small non-square correctness gate for the classic cuBLAS reference."""
import itertools
import unittest

import torch

from classic_gemm import ClassicGemm


class ClassicGemmTest(unittest.TestCase):
    def test_layout_output_beta_and_graph(self):
        torch.manual_seed(1709)
        for ta, tb, fp32, beta in itertools.product((False, True), repeat=4):
            with self.subTest(ta=ta, tb=tb, fp32=fp32, beta=beta):
                m, n, k = 80, 112, 144
                a = torch.randn((k, m) if ta else (m, k), device='cuda', dtype=torch.bfloat16)
                b = torch.randn((n, k) if tb else (k, n), device='cuda', dtype=torch.bfloat16)
                out = torch.full((m, n), 0.125, device='cuda',
                                 dtype=torch.float32 if fp32 else torch.bfloat16)
                reference = (a.T if ta else a).float() @ (b.T if tb else b).float()
                gemm = ClassicGemm(a, b, out, ta=ta, tb=tb, beta=float(beta))
                gemm()
                torch.testing.assert_close(out, (reference + beta*0.125).to(out.dtype),
                                           rtol=1e-5 if fp32 else 0.01, atol=1e-4 if fp32 else 0.25)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    gemm()
                out.fill_(0.125)
                for _ in range(2):
                    before = out.float().clone()
                    graph.replay()
                    torch.testing.assert_close(out, (reference + beta*before).to(out.dtype),
                                               rtol=1e-5 if fp32 else 0.01,
                                               atol=2e-4 if fp32 else 0.5)
                del graph
                torch.cuda.synchronize()
                gemm.close()


if __name__ == '__main__':
    unittest.main()
