"""No GPU/network: projection arithmetic and legacy matrix isolation."""
import unittest
import subprocess
import sys
from pathlib import Path

import bench
from projection_shapes import BOUNDARY_MODELS, SOURCES, matrix, projections, grouped_projections
import test_bench


class ProjectionShapes(unittest.TestCase):
    def test_shared_input_groups(self):
        expected = {'kimi_k3': ('kda_input_packed', 49376, 7168),
                    'kimi_linear_48b': ('kda_input_packed', 12576, 2304),
                    'deepseek_v3': ('mla_input_packed', 2112, 7168),
                    'glm5': ('mla_input_packed', 2624, 6144)}
        for model, (name, n, k) in expected.items():
            rows = {p['id']: p for p in grouped_projections(model)}
            self.assertEqual((rows[name]['n'], rows[name]['k']), (n, k))
            self.assertEqual(sum(p['n'] * p['k'] for p in rows.values()),
                             sum(p['n'] * p['k'] for p in projections(model)))
        small = {p['id']: p for p in grouped_projections('kimi_linear_48b')}
        self.assertIn('kda_f_b', small)
        self.assertIn('kda_g_b', small)
        self.assertEqual(small['mla_input_packed']['n'], 6720)
        k3 = {p['id']: p for p in grouped_projections('kimi_k3')}
        self.assertEqual(k3['mla_input_packed']['n'], 14400)
        self.assertIn('mla_q_b', k3)
        self.assertIn('mla_kv_b', k3)

    def test_dynamic_load_without_sibling_import_path(self):
        source = str(Path(bench.__file__).resolve())
        code = ('import importlib.util; '
                f's=importlib.util.spec_from_file_location("dynamic_bench", {source!r}); '
                'm=importlib.util.module_from_spec(s); s.loader.exec_module(m); '
                'assert "bloom_176b" in m.load_shapes("qkv").MODELS')
        # Isolated mode matches controller loading: no inherited PYTHONPATH.
        result = subprocess.run([sys.executable, '-I', '-c', code],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_native_projection_dimensions(self):
        expected = {
            'bloom_176b': {'qkv_native_packed': (43008, 14336), 'o': (14336, 14336)},
            'deepseek_v3': {'mla_q_a': (1536, 7168), 'mla_q_b': (24576, 1536),
                            'mla_kv_a': (576, 7168), 'mla_kv_b': (32768, 512),
                            'mla_o': (7168, 16384)},
        }
        for model, dimensions in expected.items():
            self.assertEqual({p['id']: (p['n'], p['k']) for p in projections(model)}, dimensions)
        k3 = {p['id']: (p['n'], p['k']) for p in projections('kimi_k3')}
        self.assertEqual(k3['kda_q'], (12288, 7168))
        self.assertEqual(k3['kda_g'], (12288, 7168))
        self.assertEqual(k3['kda_beta'], (96, 7168))
        self.assertEqual(k3['kda_f_b'], (12288, 128))
        self.assertEqual(k3['mla_q_b'], (18432, 1536))
        small = {p['id']: (p['n'], p['k']) for p in projections('kimi_linear_48b')}
        self.assertEqual(small['kda_g_a'], (128, 2304))
        self.assertEqual(small['kda_g_b'], (4096, 128))
        self.assertNotIn('kda_g', small)

    def test_export_and_bounds(self):
        recent = {p['id']: (p['n'], p['k']) for p in projections('qwen35_397b')}
        self.assertEqual(recent['full_q_gate'], (16384, 4096))
        self.assertEqual(recent['gdn_qkv'], (12288, 4096))
        glm = {p['id']: (p['n'], p['k']) for p in projections('glm5')}
        self.assertEqual(glm['mla_kv_b'], (28672, 512))
        for model in SOURCES:
            data = matrix([model])
            self.assertEqual(len(data['shapes']), len(projections(model)) * 8)
            self.assertEqual(len({p['id'] for p in data['shapes']}), len(data['shapes']))
            self.assertTrue(all(set(p) == {'id', 'm', 'n', 'k'} for p in data['shapes']))
        for names, seqs, cps in [([], (65536,), (4,)), (['deepseek_v3'], (3,), (4,)),
                                  (['deepseek_v3'], (65536,), (3,))]:
            with self.assertRaises(ValueError):
                matrix(names, seqs, cps)
        with self.assertRaises(ValueError):
            matrix(list(SOURCES))

    def test_boundary_opt_in_and_cp_constraints(self):
        helper = test_bench.BenchmarkContracts()
        self.assertEqual(len(list(bench.cases(helper.args()))), 192)
        for name in ('qwen25_72b', 'bloom_176b', 'kimi_k3_kda', 'kimi_linear_48b_kda'):
            cases = list(bench.cases(helper.args(models=name, seqs=(65536,), cps=(4, 8))))
            self.assertEqual(len(cases), 4)
            self.assertTrue(all(c['kv_heads'] == BOUNDARY_MODELS[name][2] for c in cases))
        with self.assertRaises(ValueError):
            list(bench.cases(helper.args(models='qwen3_235b', directions='qkv', cps=(8,))))
        self.assertEqual(len(list(bench.cases(helper.args(
            models='qwen3_235b', directions='qkv', cps=(4,), seqs=(65536,))))), 1)
        for direction in ('qkv', 'oproj'):
            loaded = bench.load_shapes(direction)
            self.assertTrue(all(name not in loaded.DEFAULT_MODELS for name in BOUNDARY_MODELS))
            self.assertNotIn('deepseek_v3', loaded.MODELS)


if __name__ == '__main__':
    unittest.main()
