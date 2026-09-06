"""CPU coverage guard: production measurements use the old full matrix."""
from types import SimpleNamespace
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from operator_bench import OPERATORS, select_cases


class OperatorMatrixTest(unittest.TestCase):
    def arguments(self, **changes):
        values = dict(operators=','.join(OPERATORS), case_ids=None, cp=None,
                      full=True, model='production_qwen_dense', seqs='1024')
        values.update(changes)
        return SimpleNamespace(**values)

    def test_full_matrix(self):
        rows = select_cases(self.arguments())
        self.assertEqual(len(rows), 384)
        self.assertEqual(len({row['id'] for row in rows}), 384)
        self.assertEqual({r['global_seq'] for r in rows},
                         {1024, 4096, 16384, 131072, 262144, 524288})
        self.assertTrue(all(r['m'] * r['cp'] == r['global_seq'] and r['batch'] == 1 for r in rows))

    def test_cp_groups(self):
        for cp, devices in [(4, '0,2,4,5'), (8, '0,1,2,3,4,5,6,7')]:
            rows = select_cases(self.arguments(cp=cp), cp)
            self.assertEqual(len(rows), 192)
            self.assertEqual({r['visible_devices'] for r in rows}, {devices})

    def test_four_operator_pilot(self):
        self.assertEqual(len(select_cases(self.arguments(full=False, cp=4))), 4)

    def test_multiple_models_share_one_process_group(self):
        rows = select_cases(self.arguments(full=False, cp=8, seqs='131072',
                                          model='llama3_8b,qwen25_14b_32b,nanbeige42_3b'))
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({row['id'] for row in rows}), 12)

    def test_partial_unknown_model_is_not_silently_ignored(self):
        with self.assertRaises(ValueError):
            select_cases(self.arguments(full=False, model='llama3_8b,misspelled_model'))

    def test_no_silent_bad_filter(self):
        with self.assertRaises(ValueError):
            select_cases(self.arguments(cp=4), 8)
        with self.assertRaises(ValueError):
            select_cases(self.arguments(operators='qkv_forward,qkv_forward'))
        with self.assertRaises(ValueError):
            select_cases(self.arguments(full=False, seqs='512'))

    def test_full_launcher_forwards_only_explicit_model(self):
        import run_operator_bench as launcher

        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / 'model.json'
            model.write_text('{}')  # The workers, not dry-run, validate coefficients.
            for explicit in (False, True):
                args = ['run_operator_bench.py', '--dry-run', '--skip-build']
                if explicit:
                    args += ['--oproj-comm-model', str(model)]
                output = io.StringIO()
                with patch('sys.argv', args), redirect_stdout(output), \
                        patch.object(launcher.subprocess, 'run') as run:
                    launcher.main()
                lines = output.getvalue().splitlines()
                self.assertEqual(len(lines), 2)
                for cp, line in zip((4, 8), lines):
                    self.assertIn(f'--full --cp {cp} --warmup 10 --iterations 50', line)
                    self.assertEqual('--oproj-comm-model' in line, explicit)
                    if explicit:
                        self.assertIn(str(model.resolve()), line)
                run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
