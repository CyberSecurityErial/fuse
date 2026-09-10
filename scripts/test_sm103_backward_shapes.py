"""CPU-only coverage and adjoint geometry checks; never submit GPU jobs."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import json
from unittest import mock

ENTRY = Path(__file__).resolve().parents[1] / 'benchmarks/sm103/backward/backward_shape_bench.py'
spec = importlib.util.spec_from_file_location('sm103_backward_shapes_test', ENTRY)
catalog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(catalog)
report_spec = importlib.util.spec_from_file_location('sm103_backward_report_test', ENTRY.with_name('backward_report.py'))
report = importlib.util.module_from_spec(report_spec)
report_spec.loader.exec_module(report)


class BackwardShapes(unittest.TestCase):
    def test_adjoint_dimensions_and_long_sequence_coverage(self):
        rows = catalog.cases()
        self.assertEqual(len(rows), len(catalog.projection_catalog()) * 5 * 2 * 2)
        self.assertEqual(len({r['id'] for r in rows}), len(rows))
        for row in rows:
            t, i, o = row['seq'] // row['cp'], row['input_width'], row['output_width']
            expected = (t, i, o) if row['phase'] == 'dgrad' else (o, i, t)
            self.assertEqual(tuple(row[k] for k in ('m', 'n', 'k')), expected)
            self.assertEqual(2 * row['m'] * row['n'] * row['k'], 2 * t * i * o)
            self.assertFalse(row['includes_cp_gradient_reduction'])
            self.assertEqual(row['fused_state'], 'not_measured')

    def test_kda_retains_six_projection_group_not_old_qkv(self):
        groups = catalog.projection_catalog()
        kda = next(p for p in groups if p['model'] == 'kimi_k3' and p['projection'] == 'kda_input_packed')
        self.assertEqual((kda['output_width'], kda['input_width']), (49376, 7168))
        self.assertFalse(any(p['model'] == 'kimi_k3_kda' and p['direction'] == 'qkv' for p in groups))
        self.assertTrue(any(p['model'] == 'kimi_k3_kda' and p['direction'] == 'oproj' for p in groups))

    def test_unsupported_routes_remain_in_full_matrix(self):
        rows = catalog.cases((131072,), (8,))
        qwen = next(r for r in rows if r['model'] == 'qwen3_235b' and r['projection'] == 'qkv')
        self.assertEqual(qwen['route_issue'], 'nondivisible_heads')
        self.assertTrue(any(r['route_issue'] == 'segmented_routing_not_implemented' for r in rows))

    def test_export_keeps_aliases_and_deduplicates_globally(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'matrix'
            info = catalog.export(output)
            manifest = json.loads((output / 'manifest.json').read_text())
            seen, ids = set(), set()
            for batch in manifest['batches']:
                rows = json.loads((output / batch['file']).read_text())['shapes']
                self.assertLessEqual(len(rows), 256)
                keys = {(batch['operand_layout'], r['m'], r['n'], r['k']) for r in rows}
                self.assertFalse(seen.intersection(keys))
                seen.update(keys)
                ids.update(r['id'] for r in rows)
            self.assertEqual(len(seen), info['unique_gemms'])
            self.assertEqual(ids, {r['id'] for r in manifest['cases']})
            with self.assertRaises(FileExistsError):
                catalog.export(output)

    def test_report_keeps_missing_points_and_never_claims_fused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            catalog.export(root / 'matrix', (131072,), (4,))
            manifest = root / 'matrix/manifest.json'
            case = json.loads(manifest.read_text())['cases'][0]
            measured = {key: case[key] for key in ('id','m','n','k','operand_layout')}
            measured.update(pflops=1.2, p50_ms=1.0)
            with mock.patch.object(report, 'read_run', return_value=[measured]):
                summary = report.summarize(manifest, ['fake-run'], root / 'table')
                self.assertEqual(summary['passed'], 1)
                self.assertEqual(summary['pending'], summary['total'] - 1)
                self.assertEqual(summary['fused_measured'], 0)
                text = (root / 'table/README.md').read_text()
                self.assertIn('1.200', text)
                self.assertIn('—', text)
                with self.assertRaisesRegex(ValueError, 'duplicate'):
                    report.summarize(manifest, ['fake-run', 'fake-run'], root / 'table')
            with mock.patch.object(report, 'read_run', return_value=[measured | {'operand_layout': 'nt'}]):
                with self.assertRaisesRegex(ValueError, 'geometry'):
                    report.summarize(manifest, ['fake-run'], root / 'table')


if __name__ == '__main__':
    unittest.main()
