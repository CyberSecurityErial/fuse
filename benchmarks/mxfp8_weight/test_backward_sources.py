"""CPU regression gate for frozen copies of already-validated helper logic."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]


def functions(path):
    return {node.name:ast.dump(node, include_attributes=False)
            for node in ast.parse(path.read_text()).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


class FrozenHelpersTest(unittest.TestCase):
    def test_backward_route_logic_is_legacy_logic(self):
        new = functions(Path(__file__).with_name('backward_runtime.py'))
        old = functions(ROOT/'benchmarks/backward/backward_te_nccl_baseline.py')
        for name in ('_qkv_inverse_pack_kernel', '_qkv_inverse_unpack_kernel',
                     '_oproj_route_pack_kernel', '_oproj_route_unpack_kernel',
                     'deterministic', 'global_rows'):
            with self.subTest(name=name):
                self.assertEqual(new[name], old[name])

    def test_weight_and_measurement_logic_matches_forward(self):
        new = functions(Path(__file__).with_name('backward_runtime.py'))
        old = functions(Path(__file__).with_name('bench.py'))
        for name in ('dequant_kernel', 'quantize_offline', 'reference_dequant',
                     'Userbuffers', 'measure', 'error_partials'):
            with self.subTest(name=name):
                self.assertEqual(new[name], old[name])


if __name__ == '__main__':
    unittest.main()
