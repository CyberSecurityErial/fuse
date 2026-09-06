"""CPU-only regression tests for exact published shape coverage."""
import unittest
from collections import Counter
from types import SimpleNamespace

from matrix import full_matrix, registry


class CoverageTest(unittest.TestCase):
    def test_exact_legacy_geometry(self):
        rows = full_matrix()
        self.assertEqual(len(rows), 192)
        self.assertEqual(Counter(r["direction"] for r in rows),
                         {"gemm_a2a": 96, "a2a_gemm": 96})
        for direction in ("gemm_a2a", "a2a_gemm"):
            old = registry(direction)
            cases = list(old.cases(SimpleNamespace(models=",".join(old.DEFAULT_MODELS),
                                                   seqs=old.SEQUENCES,
                                                   cps=old.CONTEXT_PARALLEL)))
            selected = [r for r in rows if r["direction"] == direction]
            self.assertEqual(len(cases), len(selected))
            for row, (model, seq, cp) in zip(selected, cases):
                self.assertEqual((row["model"], row["global_seq"], row["cp"]),
                                 (model.name, seq, cp))
                self.assertEqual(row["m"], seq // cp)
                self.assertEqual(row["n"], model.qkv_width if direction == "gemm_a2a" else model.hidden)
                self.assertEqual(row["k"], model.hidden if direction == "gemm_a2a" else model.attention_width)
                self.assertEqual(row["visible_devices"], old.VISIBLE_DEVICES[cp])
                self.assertEqual(row["batch"], 1)
                self.assertEqual(row["k"] % 32, 0)
                self.assertEqual(row["m"] % 2, 0)

    def test_axes_and_result_count(self):
        rows = full_matrix()
        self.assertEqual({r["global_seq"] for r in rows},
                         {1024, 4096, 16384, 131072, 262144, 524288})
        self.assertEqual({r["cp"] for r in rows}, {4, 8})
        expected = {(r["id"], backend, launch) for r in rows
                    for backend in ("teub", "cublaslt_nccl") for launch in ("eager", "graph")}
        self.assertEqual(len(expected), 768)


if __name__ == "__main__":
    unittest.main()
