"""Exercise the actual nested QKV measurement dispatcher without GPU imports."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock


class MetricSelection(unittest.TestCase):
    def check_selection(self, sm103, profile, include_te, expected):
        source = Path(__file__).resolve().parents[1] / 'benchmarks/sm90/QKVproj+a2a/te_nccl_baseline.py'
        nodes = [node for node in ast.walk(ast.parse(source.read_text()))
                 if isinstance(node, ast.FunctionDef) and node.name == 'measure'
                 and 'cublaslt_packed_qkv_gemm_a2a' in ast.unparse(node)]
        self.assertEqual(len(nodes), 1)
        module = ast.Module(body=[ast.ImportFrom(module='__future__',
                             names=[ast.alias(name='annotations')], level=0), nodes[0]], type_ignores=[])
        timer = mock.Mock(return_value=[.1] * 50)
        scope = dict(os=os, args=SimpleNamespace(metric_profile=profile, include_te=include_te,
                     warmup=10, iters=50, cuda_graph=False), timed_critical=timer,
                     summarize=mock.Mock(return_value={}), metrics={}, device=0, world=4)
        exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), scope)
        env = {'FUSE_SM103_MEASUREMENT': 'v2'} if sm103 else {}
        names = ('te_packed_qkv_gemm_a2a', 'cublas_packed_qkv_gemm_a2a',
                 'cublaslt_packed_qkv_gemm_a2a')
        with mock.patch.dict(os.environ, env, clear=True):
            for name in names:
                scope['measure'](name, lambda: None)
        self.assertEqual(list(scope['metrics']), expected)
        self.assertEqual(timer.call_count, len(expected))
        for call in timer.call_args_list:
            self.assertEqual(call.args[1:3], (10, 50))

    def test_sm103_tuning_measures_only_requested_cublaslt_boundary(self):
        self.check_selection(True, 'boundary', False, ['cublaslt_packed_qkv_gemm_a2a'])

    def test_legacy_and_explicit_comparisons_are_unchanged(self):
        names = ['te_packed_qkv_gemm_a2a', 'cublas_packed_qkv_gemm_a2a',
                 'cublaslt_packed_qkv_gemm_a2a']
        for args in ((False, 'boundary', False), (True, 'full', False), (True, 'boundary', True)):
            self.check_selection(*args, names)


if __name__ == '__main__':
    unittest.main()
