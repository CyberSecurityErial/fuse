import unittest
from backward_matrix import expected_keys, full_matrix, legacy_registry


class BackwardMatrixTest(unittest.TestCase):
    def test_exact_backward_registry(self):
        old = legacy_registry()
        rows = full_matrix()
        self.assertEqual(len(rows), 192)
        for r in rows:
            model = old.models_for(r['operator'])[r['model']]
            width = model.qkv_width if r['operator'] == 'qkv' else model.attention_width
            wn, wk = ((width, model.hidden) if r['operator'] == 'qkv'
                      else (model.hidden, width))
            self.assertEqual(r['b_mnk'], [r['global_seq'] // r['cp'], wk, wn])
            self.assertEqual(r['w_mnk'], [wn, wk, r['global_seq'] // r['cp']])
            self.assertEqual(r['weight_shape'], [wn, wk])
            self.assertEqual(r['visible_devices'], old.VISIBLE_DEVICES[r['cp']])
            self.assertEqual(r['layout'], 'causal_paired')
        for operator in ('qkv', 'oproj'):
            self.assertEqual(sum(r['operator'] == operator for r in rows), 96)
        self.assertEqual(len(expected_keys()), 1536)
        self.assertEqual(len(expected_keys(('bf16', 'fp32'))), 3072)
        self.assertEqual(len(expected_keys(('bf16',))), 1536)

    def test_long_context_is_not_clipped(self):
        self.assertEqual(sum(r['model'] == 'nanbeige42_3b' and
                             r['global_seq'] == 524288 for r in full_matrix()), 4)


if __name__ == '__main__':
    unittest.main()
