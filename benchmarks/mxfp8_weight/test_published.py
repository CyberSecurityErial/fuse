"""CPU-only consistency checks for the checked-in baseline snapshot."""
import hashlib
import json
import math
from pathlib import Path
import unittest

from matrix import full_matrix
from backward_matrix import full_matrix as backward_matrix

DIRECTORY = Path(__file__).resolve().parents[2] / 'results/mxfp8_weight/published'


class PublishedBaselinesTest(unittest.TestCase):
    def test_hashes(self):
        meta = json.loads((DIRECTORY/'metadata.json').read_text())
        for name, expected in meta['exported_files'].items():
            self.assertEqual(hashlib.sha256((DIRECTORY/name).read_bytes()).hexdigest(), expected)

    def test_full_matrix_and_configs(self):
        for name, matrix, modes in [('forward_best', full_matrix(), ['']),
                                     ('backward_best', backward_matrix(), ['immediate', 'deferred'])]:
            expected = {(c['id'], b, l, w) for c in matrix for b in ('teub', 'cublaslt_nccl')
                        for l in ('eager', 'graph') for w in modes}
            seen = set()
            cases = {c['id']: c for c in matrix}
            rows = json.loads((DIRECTORY/f'{name}.json').read_text())
            for row in rows:
                r = row['record']
                key = (row['case']['id'], r['backend'], r['launch'], r.get('weight_mode', ''))
                self.assertNotIn(key, seen)
                seen.add(key)
                self.assertEqual(row['case'], cases[key[0]])
                t = r.get('total', r)
                self.assertTrue(math.isfinite(t['p50_us']) and 0 < t['p50_us'] <= t['p95_us'])
                self.assertTrue(row['config'])
                self.assertEqual(len(row['source_sha256']), 64)
                if name == 'backward_best':
                    self.assertEqual(r['grad_dtype'], 'fp32')
                    self.assertEqual(r['w_plan']['output_dtype'], 'torch.float32')
                    self.assertEqual(r['beta'], int(r['weight_mode'] == 'deferred'))
                else:
                    self.assertTrue(r['gemm_plan'])
            self.assertEqual(seen, expected)


if __name__ == '__main__':
    unittest.main()
