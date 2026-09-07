"""Pure cuBLASLt matrix/sampler contracts using CPU CUDA stand-ins only."""
import argparse
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


ENTRY = Path(__file__).resolve().parents[1] / 'benchmarks/sm103/GEMM/cublaslt_bench.py'
spec = importlib.util.spec_from_file_location('sm103_pure_gemm_test_entry', ENTRY)
bench = importlib.util.module_from_spec(spec)
native_stub = types.SimpleNamespace(**{name: mock.Mock() for name in ('Library', 'Operand', 'Plan', 'check_gemm')})
with mock.patch.dict(sys.modules, {'torch': types.ModuleType('torch'), 'cublaslt': native_stub}):
    spec.loader.exec_module(bench)


class FakeCuda:
    def __init__(self):
        self.clock = 0.0
        self.events = []

    def Event(self, enable_timing):
        assert enable_timing
        cuda = self

        class Event:
            records = 0

            def record(self):
                self.timestamp = cuda.clock
                self.records += 1

            def synchronize(self):
                pass

            def elapsed_time(self, other):
                return other.timestamp - self.timestamp

        event = Event()
        self.events.append(event)
        return event

    def synchronize(self):
        pass

    def run(self):
        self.clock += 4.0


class PureGemmContracts(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(prefix='fuse-pure-gemm-test-')
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def payload(self):
        return {'schema': 'sm103_gemm_matrix_v1', 'shapes': [
            {'id': 'qkv-a', 'm': 128, 'n': 192, 'k': 64},
            {'id': 'qkv-alias', 'm': 128, 'n': 192, 'k': 64},
            {'id': 'oproj', 'm': 128, 'n': 64, 'k': 192}]}

    def test_matrix_deduplicates_exact_mnk_without_losing_aliases(self):
        groups = bench.matrix_shapes(self.payload())
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]['aliases'], ['qkv-a', 'qkv-alias'])
        self.assertEqual(groups[1]['shape'], {'m': 128, 'n': 64, 'k': 192})

    def test_matrix_rejects_bad_schema_ids_dimensions_and_extra_fields(self):
        row = self.payload()['shapes'][0]
        bad_rows = [row | {'m': True}, row | {'n': 1.5}, row | {'k': 0}, row | {'m': 2**31},
                    row | {'id': 'bad\nlabel'}, row | {'id': ''}, row | {'id': 'x' * 161},
                    row | {'comm': 8}, {'m': 1, 'n': 2, 'k': 3}]
        for candidate in bad_rows:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                bench.matrix_shapes({'schema': 'sm103_gemm_matrix_v1', 'shapes': [candidate]})
        for payload in ([], {}, self.payload() | {'schema': 'other'}, self.payload() | {'shapes': []},
                        self.payload() | {'shapes': [row, row]}, self.payload() | {'shapes': [row] * 257}):
            with self.assertRaises(ValueError):
                bench.matrix_shapes(payload)

    def test_sampler_primes_all_events_and_times_exactly_one_gemm(self):
        cuda = FakeCuda()
        calls = []

        def run():
            self.assertEqual(len(cuda.events), 100)
            self.assertTrue(all(event.records >= 1 for event in cuda.events))
            calls.append(1)
            cuda.run()

        with mock.patch.object(bench, 'torch', types.SimpleNamespace(cuda=cuda)):
            self.assertEqual(bench.collect_samples(run, 50), [4.0] * 50)
        self.assertEqual(len(calls), 50)
        self.assertTrue(all(event.records == 2 for event in cuda.events))

    def test_measure_requires_100ms_three_windows_and_first_stable_round(self):
        cuda = FakeCuda()
        noisy = [1.0] * 25 + [1.2] * 25
        stable = [2.0] * 50
        with mock.patch.object(bench, 'torch', types.SimpleNamespace(cuda=cuda)), \
                mock.patch.object(bench, 'collect_samples', side_effect=[[4.] * 10, noisy, [4.] * 10, stable]) as collect:
            samples, evidence = bench.measure(cuda.run, 10, 50)
        self.assertEqual(samples, stable)
        self.assertEqual(evidence['additional_warmup_cuda_ms'], 120)
        self.assertEqual(evidence['window_ms_per_call'], [4., 4., 4.])
        self.assertEqual(evidence['selected_round'], 1)
        self.assertEqual(evidence['measurement_rounds'][0]['samples_ms'], noisy)
        self.assertEqual([call.args[1] for call in collect.call_args_list], [10, 50, 10, 50])

    def test_measure_keeps_failed_rounds_and_does_not_retry_forever(self):
        cuda = FakeCuda()
        noisy = [1.] * 25 + [2.] * 25
        with mock.patch.object(bench, 'torch', types.SimpleNamespace(cuda=cuda)), \
                mock.patch.object(bench, 'collect_samples', side_effect=[[4.] * 10, noisy] * 3), \
                self.assertRaises(bench.MeasurementFailure) as failure:
            bench.measure(cuda.run, 10, 50)
        self.assertEqual(len(failure.exception.evidence['measurement_rounds']), 3)
        self.assertNotIn('selected_round', failure.exception.evidence)

    def test_warmup_timeout_cannot_publish_formal_samples(self):
        cuda = FakeCuda()
        with mock.patch.object(bench, 'torch', types.SimpleNamespace(cuda=cuda)), \
                mock.patch.object(bench.time, 'monotonic', side_effect=[0., 6.]), \
                mock.patch.object(bench, 'collect_samples') as collect, \
                self.assertRaises(bench.MeasurementFailure) as failure:
            bench.measure(cuda.run, 10, 50)
        collect.assert_not_called()
        self.assertEqual(failure.exception.evidence['additional_warmup_cuda_ms'], 40)

    def test_one_plan_is_reused_by_eager_graph_with_separate_measurements(self):
        a = argparse.Namespace(matrix_json=Path('matrix.json'), precisions='bf16', launches='eager,graph',
                               candidates=256, workspace_mib=256, tune_warmup=10, tune_iterations=50,
                               warmup=10, iterations=50)
        geometry = dict(bench.matrix_shapes(self.payload())[0], results=[])
        plan = mock.Mock(info={'valid': 8, 'requested': 256})
        graph = mock.Mock()
        torch = mock.Mock()
        torch.cuda.CUDAGraph.return_value = graph
        torch.cuda.graph.return_value = contextlib.nullcontext()
        with mock.patch.object(bench, 'torch', torch), \
                mock.patch.object(bench, 'input_tensor', side_effect=['activation', 'weight']) as inputs, \
                mock.patch.object(bench, 'input_statistics', return_value={'sample_count': 4096}), \
                mock.patch.object(bench, 'Operand'), mock.patch.object(bench, 'Plan', return_value=plan) as create, \
                mock.patch.object(bench, 'check_gemm', return_value={'checked_values': 4096}) as checker, \
                mock.patch.object(bench, 'measure', return_value=([4.] * 50, {'protocol': bench.PROTOCOL})) as measure, \
                contextlib.redirect_stdout(io.StringIO()):
            bench.run_geometry(a, object(), geometry, 1, 2)
        create.assert_called_once()
        self.assertEqual(create.call_args.kwargs['candidates'], 256)
        self.assertEqual((create.call_args.kwargs['warmup'], create.call_args.kwargs['iterations']), (10, 50))
        self.assertEqual([call.args for call in inputs.call_args_list], [((128, 64), .125, True), ((192, 64), .02, True)])
        self.assertEqual(measure.call_args_list[0].args[0], plan.run)
        self.assertEqual(measure.call_args_list[1].args[0], graph.replay)
        self.assertEqual(checker.call_args_list[-1].args[0].run, graph.replay)
        plan.close.assert_called_once()
        self.assertEqual([row['launch'] for row in geometry['results']], ['eager', 'graph'])
        self.assertEqual(geometry['inputs']['distribution'], 'uniform')
        for row in geometry['results']:
            self.assertEqual(row['pflops_per_gpu_p50'], 2 * 128 * 192 * 64 / 4 / 1e12)

    def test_cutlass_comparison_shares_inputs_keeps_both_orders_and_closes_plans(self):
        a = argparse.Namespace(candidates=256, workspace_mib=256, tune_warmup=10,
                               tune_iterations=50, warmup=10, iterations=50, cutlass_swizzle_size=4, cutlass_epilogue_n=64, cutlass_1sm_cluster_m=1, cutlass_full_check=False)
        geometry = dict(bench.matrix_shapes(self.payload())[0], results=[])
        output = mock.Mock()
        plans = [mock.Mock(info={'backend': name}, output=output)
                 for name in ('cublaslt', 'cutlass_1sm', 'cutlass_2sm')]
        create_cutlass = mock.Mock(side_effect=plans[1:])
        torch = mock.Mock()
        torch.empty.return_value = output
        with mock.patch.dict(sys.modules, {'cutlass': types.SimpleNamespace(Plan=create_cutlass)}), \
                mock.patch.object(bench, 'torch', torch), \
                mock.patch.object(bench, 'prepare_inputs', return_value=('source', 'weight')) as inputs, \
                mock.patch.object(bench, 'Operand', side_effect=['x', 'w']), \
                mock.patch.object(bench, 'Plan', return_value=plans[0]) as create_lt, \
                mock.patch.object(bench, 'check_gemm', return_value={'checked_values': 4096}) as check, \
                mock.patch.object(bench, 'measure', return_value=([4.] * 50, {'protocol': bench.PROTOCOL})) as measure, \
                contextlib.redirect_stdout(io.StringIO()):
            bench.run_cutlass_comparison(a, 'lt_library', geometry, 1, 1, 'cutlass_library')
        inputs.assert_called_once_with(geometry, True)
        create_lt.assert_called_once()
        self.assertEqual(create_lt.call_args.args, ('lt_library', 'x', 'w', output))
        self.assertEqual([c.args for c in create_cutlass.call_args_list],
                         [('cutlass_library', 'x', 'w', output)] * 2)
        self.assertEqual([c.kwargs for c in create_cutlass.call_args_list],
                         [{'sm_mode': 1, 'max_swizzle_size': 4, 'epilogue_n': 64, 'cluster_m': 1, 'sm_budget': 0},
                          {'sm_mode': 2, 'max_swizzle_size': 4, 'epilogue_n': 64, 'cluster_m': 2, 'sm_budget': 0}])
        self.assertEqual(geometry['comparison']['cutlass_max_swizzle_size'], 4)
        self.assertEqual(geometry['comparison']['cutlass_epilogue_n'], 64)
        self.assertNotIn('epilogue_n', create_lt.call_args.kwargs)
        self.assertNotIn('max_swizzle_size', create_lt.call_args.kwargs)
        self.assertNotIn('cluster_m', create_lt.call_args.kwargs)
        self.assertEqual([c.args[0] for c in measure.call_args_list],
                         [plan.run for plan in plans + list(reversed(plans))])
        self.assertEqual([r['backend'] for r in geometry['results']],
                         ['cublaslt', 'cutlass_1sm', 'cutlass_2sm', 'cutlass_2sm', 'cutlass_1sm', 'cublaslt'])
        self.assertEqual([r['comparison_block'] for r in geometry['results']], [0] * 3 + [1] * 3)
        self.assertEqual(output.fill_.call_count, 6)
        # Post-checks inspect saved output without another plan invocation.
        for call in check.call_args_list[1::2]:
            self.assertIs(call.args[0].run(), output)
        for plan in plans:
            plan.close.assert_called_once()

    def test_cutlass_comparison_retains_measurement_failure_and_closes_every_plan(self):
        a = argparse.Namespace(candidates=256, workspace_mib=256, tune_warmup=10,
                               tune_iterations=50, warmup=10, iterations=50, cutlass_swizzle_size=1, cutlass_epilogue_n=32, cutlass_1sm_cluster_m=1, cutlass_full_check=False)
        geometry = dict(bench.matrix_shapes(self.payload())[0], results=[])
        plans = [mock.Mock(info={}) for _ in range(3)]
        failure = bench.MeasurementFailure('noisy', {'selected_round': None})
        with mock.patch.dict(sys.modules, {'cutlass': types.SimpleNamespace(Plan=mock.Mock(side_effect=plans[1:]))}), \
                mock.patch.object(bench, 'torch', mock.Mock()), \
                mock.patch.object(bench, 'prepare_inputs', return_value=('x', 'w')), \
                mock.patch.object(bench, 'Operand'), mock.patch.object(bench, 'Plan', return_value=plans[0]), \
                mock.patch.object(bench, 'check_gemm'), mock.patch.object(bench, 'measure', side_effect=failure), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(bench.MeasurementFailure):
            bench.run_cutlass_comparison(a, None, geometry, 1, 1, None)
        self.assertEqual(geometry['results'], [])
        self.assertEqual(geometry['failed_measurement']['backend'], 'cublaslt')
        self.assertIs(geometry['failed_measurement']['measurement'], failure.evidence)
        for plan in plans:
            plan.close.assert_called_once()

    def test_cutlass_mode_rejects_other_precisions_graph_and_implicit_matrix_before_cuda(self):
        for extra in ([], ['--matrix-json', 'unused', '--precisions', 'fp8'],
                      ['--matrix-json', 'unused', '--launches', 'graph']):
            argv = ['bench', '--output', str(self.root / 'out.json'), '--compare-cutlass-library', 'unused', *extra]
            with mock.patch.object(sys, 'argv', argv), mock.patch.object(bench, 'torch') as torch, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                bench.main()
            torch.cuda.get_device_properties.assert_not_called()

    def test_cutlass_swizzle_rejects_invalid_or_noncomparison_use_before_cuda(self):
        for extra in (['--cutlass-swizzle-size', '1'], ['--cutlass-swizzle-size', '4'],
                      ['--compare-cutlass-library', 'unused', '--cutlass-swizzle-size', '3'],
                      ['--compare-cutlass-library', 'unused', '--cutlass-swizzle-size', '16']):
            argv = ['bench', '--output', str(self.root / 'out.json'), '--matrix-json', 'unused',
                    '--launches', 'eager', *extra]
            with mock.patch.object(sys, 'argv', argv), mock.patch.object(bench, 'torch') as torch, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                bench.main()
            torch.cuda.get_device_properties.assert_not_called()

    def test_cutlass_cluster_rejects_invalid_or_noncomparison_use_before_cuda(self):
        for extra in (['--cutlass-1sm-cluster-m', '1'], ['--cutlass-1sm-cluster-m', '2'],
                      ['--compare-cutlass-library', 'unused', '--cutlass-1sm-cluster-m', '3'],
                      ['--compare-cutlass-library', 'unused', '--cutlass-1sm-cluster-m', '2',
                       '--cutlass-epilogue-n', '64']):
            argv = ['bench', '--output', str(self.root / 'out.json'), '--matrix-json', 'unused',
                    '--launches', 'eager', *extra]
            with mock.patch.object(sys, 'argv', argv), mock.patch.object(bench, 'torch') as torch, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                bench.main()
            torch.cuda.get_device_properties.assert_not_called()

    def test_cutlass_cluster_cli_default_and_explicit_forwarding(self):
        matrix = self.root / 'matrix.json'
        matrix.write_text(json.dumps(self.payload()))
        torch, fake_library = self.main_mocks()
        for extra, expected in (([], 1), (['--cutlass-1sm-cluster-m', '2', '--cutlass-full-check'], 2)):
            argv = ['bench', '--output', str(self.root / f'cluster-{expected}.json'),
                    '--matrix-json', str(matrix), '--launches', 'eager',
                    '--compare-cutlass-library', str(fake_library.path), *extra]
            with mock.patch.object(sys, 'argv', argv), mock.patch.object(bench, 'torch', torch), \
                    mock.patch.object(bench, 'Library', return_value=fake_library), \
                    mock.patch.dict(sys.modules, {'cutlass': types.SimpleNamespace(Library=lambda _: fake_library)}), \
                    mock.patch.object(bench, 'run_cutlass_comparison') as run, \
                    contextlib.redirect_stdout(io.StringIO()):
                bench.main()
            self.assertEqual(len(run.call_args_list), 2)
            self.assertTrue(all(call.args[0].cutlass_1sm_cluster_m == expected for call in run.call_args_list))
            self.assertTrue(all(call.args[0].cutlass_epilogue_n == 32 for call in run.call_args_list))
            self.assertTrue(all(call.args[0].cutlass_full_check == (expected == 2) for call in run.call_args_list))

    def test_cluster2_and_full_check_change_only_the_requested_comparison_paths(self):
        a = argparse.Namespace(candidates=256, workspace_mib=256, tune_warmup=10,
                               tune_iterations=50, warmup=10, iterations=50, cutlass_swizzle_size=4,
                               cutlass_epilogue_n=32, cutlass_1sm_cluster_m=2, cutlass_full_check=True)
        geometry = dict(bench.matrix_shapes(self.payload())[0], results=[])
        shared_output = object()
        plans = [mock.Mock(info={}, output=shared_output) for _ in range(3)]
        create_cutlass = mock.Mock(side_effect=plans[1:])
        with mock.patch.dict(sys.modules, {'cutlass': types.SimpleNamespace(Plan=create_cutlass)}), \
                mock.patch.object(bench, 'torch', mock.Mock()), \
                mock.patch.object(bench, 'prepare_inputs', return_value=('x', 'w')), \
                mock.patch.object(bench, 'Operand'), mock.patch.object(bench, 'Plan', return_value=plans[0]) as lt, \
                mock.patch.object(bench, 'check_gemm', return_value={'checked_values': 24576, 'full_output_checked': True}) as check, \
                mock.patch.object(bench, 'measure', return_value=([4.] * 50, {'protocol': bench.PROTOCOL})), \
                contextlib.redirect_stdout(io.StringIO()):
            bench.run_cutlass_comparison(a, None, geometry, 1, 1, None)
        self.assertEqual([c.kwargs for c in create_cutlass.call_args_list],
                         [{'sm_mode': 1, 'max_swizzle_size': 4, 'epilogue_n': 32, 'cluster_m': 2, 'sm_budget': 0},
                          {'sm_mode': 2, 'max_swizzle_size': 4, 'epilogue_n': 32, 'cluster_m': 2, 'sm_budget': 0}])
        self.assertNotIn('cluster_m', lt.call_args.kwargs)
        self.assertNotIn('full', lt.call_args.kwargs)
        self.assertEqual(check.call_count, 12)
        self.assertTrue(all(c.kwargs == {'full': True} for c in check.call_args_list))
        for call, row in zip(check.call_args_list[1::2], geometry['results']):
            backend_index = ('cublaslt', 'cutlass_1sm', 'cutlass_2sm').index(row['backend'])
            self.assertIs(call.args[0].run(), plans[backend_index].output)
            self.assertTrue(row['correctness_post']['full_output_checked'])

    def test_full_check_cli_rejects_large_or_noncomparison_before_cuda(self):
        matrix = self.root / 'matrix.json'
        for shape in ({'m': 4096, 'n': 4096, 'k': 8}, {'m': 8, 'n': 8, 'k': 1048576}):
            matrix.write_text(json.dumps({'schema': 'sm103_gemm_matrix_v1', 'shapes': [dict(id='large', **shape)]}))
            for comparison in ([], ['--compare-cutlass-library', 'unused']):
                argv = ['bench', '--output', str(self.root / 'out.json'), '--matrix-json', str(matrix),
                        '--launches', 'eager', '--cutlass-full-check', *comparison]
                with mock.patch.object(sys, 'argv', argv), mock.patch.object(bench, 'torch') as torch, \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    bench.main()
                torch.cuda.get_device_properties.assert_not_called()

    def test_actual_checker_covers_odd_tail_and_catches_unsampled_nan_on_cpu(self):
        # Execute the real checker with small scalar/list tensor arithmetic.
        # No Torch/CUDA dependency; the fault at [1,1] is outside old linspace
        # samples for 257x264, but must be included by the full mode.
        class Indices(list):
            def __getitem__(self, key):
                return self if isinstance(key, tuple) else super().__getitem__(key)

            def long(self):
                return self

        class Scalar(float):
            def sqrt(self): return Scalar(math.sqrt(self))
            def clamp_min(self, bound): return Scalar(max(self, bound))
            def item(self): return float(self)
            def __truediv__(self, value): return Scalar(float(self) / value)

        class Matrix:
            device = 'cpu'

            def __init__(self, values): self.values = values

            def __getitem__(self, key):
                if isinstance(key, tuple):
                    rows, cols = key
                    return Matrix([[self.values[r][c] for c in cols] for r in rows])
                return Matrix([self.values[r] for r in key])

            @property
            def T(self): return Matrix(list(map(list, zip(*self.values))))

            def __matmul__(self, other):
                cols = list(zip(*other.values))
                return Matrix([[sum(a * b for a, b in zip(row, col)) for col in cols]
                               for row in self.values])

            def __sub__(self, other):
                return Matrix([[a - b for a, b in zip(x, y)] for x, y in zip(self.values, other.values)])

            def map(self, fn): return Matrix([[fn(v) for v in row] for row in self.values])
            def square(self): return self.map(lambda v: v * v)
            def abs(self): return self.map(abs)
            def float(self): return self
            def numel(self): return sum(map(len, self.values))
            def mean(self): return Scalar(sum(map(sum, self.values)) / self.numel())
            def max(self): return Scalar(max(map(max, self.values)))

        torch = types.SimpleNamespace(
            arange=lambda n, device: Indices(range(n)),
            linspace=lambda start, end, count, device: Indices(int(start + (end - start) * i / (count - 1)) for i in range(count)),
            isfinite=lambda tensor: types.SimpleNamespace(all=lambda: all(math.isfinite(v) for row in tensor.values for v in row)),
            backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=True))))
        checker_file = ENTRY.with_name('cublaslt.py')
        node = next(n for n in ast.parse(checker_file.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'check_gemm')
        scope = {'torch': torch}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(checker_file), 'exec'), scope)
        check = scope['check_gemm']
        output = Matrix([[1.] * 264 for _ in range(257)])
        x = types.SimpleNamespace(rows=257, k=1, precision='bf16', decode=lambda: Matrix([[1.]] * 257))
        w = types.SimpleNamespace(rows=264, k=1, precision='bf16', decode=lambda: Matrix([[1.]] * 264))
        run = mock.Mock(return_value=output)
        plan = types.SimpleNamespace(x=x, weight=w, output=output, run=run)
        sampled = check(plan)
        self.assertEqual(sampled, {'relative_rms': 0., 'max_abs': 0., 'checked_values': 4096})
        full = check(plan, full=True)
        self.assertEqual(full['checked_values'], 257 * 264)
        self.assertTrue(full['full_output_checked'])
        output.values[1][1] = float('nan')
        self.assertEqual(check(plan)['checked_values'], 4096)
        with self.assertRaisesRegex(RuntimeError, 'correctness failure'):
            check(plan, full=True)
        self.assertEqual(run.call_count, 4)
        x.rows = 4_194_305
        with self.assertRaisesRegex(ValueError, 'small BF16'):
            check(plan, full=True)
        self.assertEqual(run.call_count, 4)  # Bounds enforced before launch.

    def test_cutlass_swizzle_cli_default_and_explicit_forwarding(self):
        matrix = self.root / 'matrix.json'
        matrix.write_text(json.dumps(self.payload()))
        torch, fake_library = self.main_mocks()
        library = fake_library.path
        for extra, expected in (([], 1), (['--cutlass-swizzle-size', '4'], 4)):
            argv = ['bench', '--output', str(self.root / f'compare-{expected}.json'),
                    '--matrix-json', str(matrix), '--launches', 'eager',
                    '--compare-cutlass-library', str(library), *extra]
            with mock.patch.object(sys, 'argv', argv), mock.patch.object(bench, 'torch', torch), \
                    mock.patch.object(bench, 'Library', return_value=fake_library), \
                    mock.patch.dict(sys.modules, {'cutlass': types.SimpleNamespace(Library=lambda _: fake_library)}), \
                    mock.patch.object(bench, 'run_cutlass_comparison') as run, \
                    contextlib.redirect_stdout(io.StringIO()):
                bench.main()
            self.assertEqual(len(run.call_args_list), 2)
            self.assertTrue(all(call.args[0].cutlass_swizzle_size == expected for call in run.call_args_list))

    def test_cutlass_epilogue_rejects_invalid_or_noncomparison_use_before_cuda(self):
        for extra in (['--cutlass-epilogue-n', '32'], ['--cutlass-epilogue-n', '64'],
                      ['--compare-cutlass-library', 'unused', '--cutlass-epilogue-n', '16'],
                      ['--compare-cutlass-library', 'unused', '--cutlass-epilogue-n', '128']):
            argv = ['bench', '--output', str(self.root / 'out.json'), '--matrix-json', 'unused',
                    '--launches', 'eager', *extra]
            with mock.patch.object(sys, 'argv', argv), mock.patch.object(bench, 'torch') as torch, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                bench.main()
            torch.cuda.get_device_properties.assert_not_called()

    def test_cutlass_epilogue_cli_default_and_explicit_forwarding(self):
        matrix = self.root / 'matrix.json'
        matrix.write_text(json.dumps(self.payload()))
        torch, fake_library = self.main_mocks()
        for extra, expected in (([], 32), (['--cutlass-epilogue-n', '32'], 32),
                                (['--cutlass-epilogue-n', '64'], 64)):
            argv = ['bench', '--output', str(self.root / f'compare-{len(extra)}-{expected}.json'),
                    '--matrix-json', str(matrix), '--launches', 'eager',
                    '--compare-cutlass-library', str(fake_library.path), *extra]
            with mock.patch.object(sys, 'argv', argv), mock.patch.object(bench, 'torch', torch), \
                    mock.patch.object(bench, 'Library', return_value=fake_library), \
                    mock.patch.dict(sys.modules, {'cutlass': types.SimpleNamespace(Library=lambda _: fake_library)}), \
                    mock.patch.object(bench, 'run_cutlass_comparison') as run, \
                    contextlib.redirect_stdout(io.StringIO()):
                bench.main()
            self.assertEqual(len(run.call_args_list), 2)
            self.assertTrue(all(call.args[0].cutlass_epilogue_n == expected for call in run.call_args_list))
            self.assertTrue(all(call.args[0].cutlass_swizzle_size == 1 for call in run.call_args_list))

    def test_cutlass_failed_postcheck_keeps_actual_samples_without_accepting_them(self):
        a = argparse.Namespace(candidates=256, workspace_mib=256, tune_warmup=10,
                               tune_iterations=50, warmup=10, iterations=50, cutlass_swizzle_size=1, cutlass_epilogue_n=32, cutlass_1sm_cluster_m=1, cutlass_full_check=False)
        geometry = dict(bench.matrix_shapes(self.payload())[0], results=[])
        plans = [mock.Mock(info={}) for _ in range(3)]
        samples, evidence = [4.] * 50, {'protocol': bench.PROTOCOL}
        with mock.patch.dict(sys.modules, {'cutlass': types.SimpleNamespace(Plan=mock.Mock(side_effect=plans[1:]))}), \
                mock.patch.object(bench, 'torch', mock.Mock()), \
                mock.patch.object(bench, 'prepare_inputs', return_value=('x', 'w')), \
                mock.patch.object(bench, 'Operand'), mock.patch.object(bench, 'Plan', return_value=plans[0]), \
                mock.patch.object(bench, 'check_gemm', side_effect=[{}, ValueError('post failed')]), \
                mock.patch.object(bench, 'measure', return_value=(samples, evidence)), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'post failed'):
            bench.run_cutlass_comparison(a, None, geometry, 1, 1, None)
        failure = geometry['failed_measurement']
        self.assertEqual(failure['phase'], 'correctness_post')
        self.assertEqual(failure['backend'], 'cublaslt')
        self.assertIs(failure['samples_ms'], samples)
        self.assertIs(failure['measurement'], evidence)
        self.assertFalse(failure['performance_accepted'])
        self.assertEqual(geometry['results'], [])
        for plan in plans:
            plan.close.assert_called_once()

    def main_mocks(self):
        library = self.root / 'libpure.so'
        library.write_bytes(b'fake local library')
        torch = types.SimpleNamespace(__version__='test', version=types.SimpleNamespace(cuda='13'),
            cuda=types.SimpleNamespace(get_device_properties=lambda _: types.SimpleNamespace(
                major=10, minor=3, multi_processor_count=148, name='test device')))
        return torch, types.SimpleNamespace(path=library)

    def test_matrix_cli_reuses_process_and_checkpoints_failure_without_false_completion(self):
        matrix, output = self.root / 'matrix.json', self.root / 'matrix-result.json'
        matrix.write_text(json.dumps(self.payload()))
        torch, lib = self.main_mocks()

        def run(a, library, geometry, index, total):
            self.assertIs(library, lib)
            self.assertEqual((a.precisions, a.candidates, a.tune_warmup, a.tune_iterations), ('bf16', 256, 10, 50))
            self.assertEqual(total, 2)
            if index == 2:
                geometry['failed_measurement'] = {'measurement': {'measurement_rounds': [1, 2, 3]}}
                raise RuntimeError('measured failure')
            geometry['results'].append({'precision': 'bf16', 'launch': 'eager'})

        with mock.patch.object(sys, 'argv', ['probe', '--matrix-json', str(matrix), '--output', str(output)]), \
                mock.patch.object(bench, 'torch', torch), mock.patch.object(bench, 'Library', return_value=lib) as create, \
                mock.patch.object(bench, 'run_geometry', side_effect=run) as execute, \
                self.assertRaisesRegex(RuntimeError, 'measured failure'):
            bench.main()
        result = json.loads(output.read_text())
        create.assert_called_once()
        self.assertEqual(execute.call_count, 2)
        self.assertEqual((result['schema'], result['state']), ('sm103_gemm_matrix_v1', 'failed'))
        self.assertEqual((result['logical_shapes'], result['unique_geometries']), (3, 2))
        self.assertEqual([row['state'] for row in result['geometries']], ['succeeded', 'failed'])
        self.assertEqual(result['geometries'][1]['failed_measurement']['measurement']['measurement_rounds'], [1, 2, 3])
        self.assertFalse(result['distributed_boundary_measured'])
        self.assertFalse(result['cublas_classic_measured'])

    def test_single_cli_preserves_shape_results_layout_and_rejects_existing_output(self):
        output = self.root / 'single.json'
        torch, lib = self.main_mocks()

        def run(a, library, geometry, index, total):
            self.assertIsNone(a.matrix_json)
            self.assertEqual(a.precisions, 'bf16,fp8,fp4')
            geometry.update(inputs={'distribution': 'normal'})
            geometry['results'].append({'launch': 'eager', 'samples_ms': [4.] * 50})

        with mock.patch.object(sys, 'argv', ['probe', '--output', str(output)]), \
                mock.patch.object(bench, 'torch', torch), mock.patch.object(bench, 'Library', return_value=lib), \
                mock.patch.object(bench, 'run_geometry', side_effect=run):
            bench.main()
        result = json.loads(output.read_text())
        self.assertEqual((result['schema'], result['state']), ('sm103_gemm_v1', 'succeeded'))
        self.assertEqual(result['shape'], {'m': 512, 'n': 4096, 'k': 2048})
        self.assertEqual(len(result['results']), 1)
        self.assertNotIn('geometries', result)
        original = output.read_bytes()
        with mock.patch.object(sys, 'argv', ['probe', '--output', str(output)]), \
                mock.patch.object(bench, 'Library', side_effect=AssertionError('No CUDA library call')), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            bench.main()
        self.assertEqual(output.read_bytes(), original)

    def test_cli_rejects_unsafe_measurement_or_matrix_overrides_before_gpu(self):
        matrix = self.root / 'matrix.json'
        matrix.write_text(json.dumps(self.payload()))
        for flags in (['--warmup', '9'], ['--iterations', '49'], ['--tune-warmup', '5'],
                      ['--tune-iterations', '30'], ['--precisions', ''], ['--launches', 'eager,eager'],
                      ['--matrix-json', str(matrix), '--m', '128'],
                      ['--matrix-json', str(matrix), '--precisions', 'fp8']):
            with self.subTest(flags=flags), \
                    mock.patch.object(sys, 'argv', ['probe', '--output', str(self.root / 'unused.json'), *flags]), \
                    mock.patch.object(bench, 'torch', mock.Mock(side_effect=AssertionError('No GPU calls'))), \
                    mock.patch.object(bench, 'Library', side_effect=AssertionError('No CUDA library call')), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                bench.main()
        self.assertFalse((self.root / 'unused.json').exists())

    def test_counter_and_cluster_additions_preserve_default_sampler_function_asts(self):
        names = ('collect_samples', 'measure', 'input_tensor', 'input_statistics',
                 'prepare_inputs', 'run_geometry')
        nodes = {node.name: node for node in ast.parse(ENTRY.read_text()).body
                 if isinstance(node, ast.FunctionDef)}
        # Python 3.12 adds empty type_params to FunctionDef; ignore that
        # parser-only addition so the v23 contract also runs on the GPU host.
        for name in names:
            if getattr(nodes[name], 'type_params', None) == []:
                nodes[name]._fields = tuple(field for field in nodes[name]._fields if field != 'type_params')
        digest = hashlib.sha256('\n'.join(ast.dump(nodes[name], include_attributes=False)
                                          for name in names).encode()).hexdigest()
        # Same six function ASTs as frozen v23/v24. The comparison's explicit
        # plan arguments/check coverage changed; order/lifetime have tests above.
        self.assertEqual(digest, '9ae2e93141fdea8936313d83a671352467d7a722785d6ab6fa96884f9175472e')

    def test_counter_warmup_body_matches_production_sampler_exactly(self):
        nodes = {node.name: node for node in ast.parse(ENTRY.read_text()).body
                 if isinstance(node, ast.FunctionDef)}

        def warmup_body(name):
            body = nodes[name].body
            start = next(i for i, node in enumerate(body) if isinstance(node, ast.For))
            end = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign)
                       and ast.unparse(node.targets[0]) == "evidence['warmup_converged']")
            return [ast.dump(node, include_attributes=False) for node in body[start:end + 1]]

        self.assertEqual(warmup_body('warmup_counters'), warmup_body('measure'))

    def test_counter_warmup_requires_10_calls_100ms_three_windows_without_samples(self):
        cuda = FakeCuda()
        with mock.patch.object(bench, 'torch', types.SimpleNamespace(cuda=cuda)), \
                mock.patch.object(bench, 'measure', side_effect=AssertionError('formal measure')), \
                mock.patch.object(bench, 'collect_samples', side_effect=AssertionError('formal samples')):
            evidence = bench.warmup_counters(cuda.run, 10)
        self.assertEqual(cuda.clock, 160)
        self.assertEqual(evidence['additional_warmup_calls'], 30)
        self.assertEqual(evidence['additional_warmup_cuda_ms'], 120)
        self.assertEqual(evidence['window_ms_per_call'], [4., 4., 4.])
        self.assertTrue(evidence['warmup_converged'])
        self.assertNotIn('measurement_rounds', evidence)

    def test_counter_warmup_invalid_time_and_timeout_keep_evidence(self):
        for invalid in (False, True):
            cuda = FakeCuda()
            run = (lambda: None) if invalid else cuda.run
            with self.subTest(invalid=invalid), \
                    mock.patch.object(bench, 'torch', types.SimpleNamespace(cuda=cuda)), \
                    mock.patch.object(bench.time, 'monotonic', side_effect=[0., 6.]), \
                    mock.patch.object(bench, 'collect_samples', side_effect=AssertionError('formal samples')), \
                    self.assertRaises(bench.MeasurementFailure) as failure:
                bench.warmup_counters(run, 10)
            self.assertNotIn('warmup_converged', failure.exception.evidence)
            self.assertNotIn('measurement_rounds', failure.exception.evidence)

    @contextlib.contextmanager
    def counter_stubs(self, fault=None):
        cuda, calls = FakeCuda(), []
        state = {'in_range': False, 'nan': True, 'checks': 0}
        geometry = dict(bench.matrix_shapes(self.payload())[0])
        a = argparse.Namespace(warmup=10, cutlass_swizzle_size=4, cutlass_epilogue_n=64, cutlass_1sm_cluster_m=1, cutlass_full_check=False)

        def poison(value):
            self.assertNotEqual(value, value)  # NaN, not a constant valid output.
            calls.append('poison')
            state['nan'] = True

        output = types.SimpleNamespace(fill_=poison)

        def run():
            calls.append('run')
            if state['in_range'] and fault == 'nvtx_launch':
                raise RuntimeError(fault)
            if not (state['in_range'] and fault == 'no_op_target'):
                state['nan'] = False
            cuda.run()
            return output

        def push(label):
            self.assertEqual(label, 'fuse_cutlass_1sm_counters')
            self.assertFalse(state['in_range'])
            calls.append('push')
            state['in_range'] = True

        def pop():
            self.assertTrue(state['in_range'])
            calls.append('pop')
            state['in_range'] = False

        def synchronize():
            self.assertFalse(state['in_range'])
            calls.append('sync')

        def check(plan):
            self.assertFalse(state['in_range'])
            calls.append('check')
            state['checks'] += 1
            plan.run()
            if state['nan']:
                raise RuntimeError('unwritten NaN output')
            if fault == ('correctness_pre' if state['checks'] == 1 else 'correctness_post'):
                raise RuntimeError(fault)
            return {'checked_values': 4096, 'relative_rms': 0., 'max_abs': 0.}

        def close():
            calls.append('close')
            if fault == 'cleanup':
                raise RuntimeError(fault)

        cuda.synchronize = synchronize
        cuda.nvtx = types.SimpleNamespace(range_push=push, range_pop=pop)
        info = {'schema': 'sm103_cutlass_bf16_plan_v1', 'backend': 'cutlass_1sm',
                'sm_mode': 1, 'm': 128, 'n': 192, 'k': 64, 'max_swizzle_size': 4,
                'effective_swizzle_size': 1, 'epilogue_n': 64, 'threads': 256,
                'dynamic_smem': 230400, 'grid': [148, 1, 1]}
        plan = types.SimpleNamespace(x='operand_x', weight='operand_w', output=output,
                                     info=info, run=mock.Mock(side_effect=run), close=mock.Mock(side_effect=close))
        create = mock.Mock(return_value=plan, side_effect=RuntimeError('create') if fault == 'create' else None)
        torch = types.SimpleNamespace(cuda=cuda, bfloat16='bf16', empty=lambda *args, **kwargs: output)
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.dict(sys.modules, {'cutlass': types.SimpleNamespace(Plan=create)}))
            stack.enter_context(mock.patch.object(bench, 'torch', torch))
            inputs = stack.enter_context(mock.patch.object(bench, 'prepare_inputs', return_value=('source', 'weight')))
            operands = stack.enter_context(mock.patch.object(bench, 'Operand', side_effect=['operand_x', 'operand_w']))
            stack.enter_context(mock.patch.object(bench, 'Plan', side_effect=AssertionError('no Lt plan/tuning')))
            stack.enter_context(mock.patch.object(bench, 'check_gemm', side_effect=check))
            stack.enter_context(mock.patch.object(bench, 'measure', side_effect=AssertionError('no formal measure')))
            stack.enter_context(mock.patch.object(bench, 'collect_samples', side_effect=AssertionError('no formal samples')))
            if fault == 'warmup':
                stack.enter_context(mock.patch.object(bench, 'warmup_counters',
                    side_effect=bench.MeasurementFailure('warmup', {'additional_warmup_cuda_ms': 40.})))
            stdout = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            yield types.SimpleNamespace(args=a, geometry=geometry, calls=calls, plan=plan, create=create,
                                        output=output, inputs=inputs, operands=operands, stdout=stdout)

    def test_counter_explicit_budget_is_forwarded_and_recorded(self):
        with self.counter_stubs() as test:
            test.args.cutlass_sm_budget = 132
            bench.run_cutlass_counters(test.args, 'lt_lib', test.geometry, 'cutlass_lib')
        self.assertEqual(test.create.call_args.kwargs['sm_budget'], 132)
        self.assertEqual(test.geometry['counter_diagnostic']['physical_sm_budget'], 132)

    def test_counter_range_contains_only_one_warmed_1sm_run_and_postcheck_does_not_rerun(self):
        with self.counter_stubs() as test:
            bench.run_cutlass_counters(test.args, 'lt_lib', test.geometry, 'cutlass_lib')
        test.create.assert_called_once_with('cutlass_lib', 'operand_x', 'operand_w', test.output,
                                           sm_mode=1, max_swizzle_size=4, epilogue_n=64, cluster_m=1, sm_budget=0)
        test.inputs.assert_called_once_with(test.geometry, True)
        self.assertEqual(test.operands.call_args_list, [mock.call('lt_lib', 'source', 'bf16'),
                                                       mock.call('lt_lib', 'weight', 'bf16')])
        begin = test.calls.index('push')
        self.assertEqual(test.calls[begin - 2:], ['poison', 'sync', 'push', 'run', 'pop',
                                                'sync', 'check', 'sync', 'close'])
        self.assertEqual(test.calls.count('push'), 1)
        self.assertEqual(test.calls.count('poison'), 2)
        self.assertEqual(test.plan.run.call_count, 42)  # pre + initial10 + 3x10 + target.
        test.plan.close.assert_called_once()
        diagnostic = test.geometry['counter_diagnostic']
        self.assertEqual(diagnostic['state'], 'succeeded')
        self.assertEqual(diagnostic['plan'], test.plan.info)
        self.assertEqual(diagnostic['range_launches'], 1)
        self.assertFalse(diagnostic['formal_sampling_performed'])
        self.assertFalse(diagnostic['ncu_metrics_verified'])
        self.assertEqual(diagnostic['correctness_post']['checked_values'], 4096)

    def test_counter_cluster2_is_forwarded_without_a_2sm_or_lt_plan(self):
        with self.counter_stubs() as test:
            test.args.cutlass_1sm_cluster_m = 2
            test.args.cutlass_epilogue_n = 32
            bench.run_cutlass_counters(test.args, 'lt_lib', test.geometry, 'cutlass_lib')
        test.create.assert_called_once_with('cutlass_lib', 'operand_x', 'operand_w', test.output,
                                           sm_mode=1, max_swizzle_size=4, epilogue_n=32, cluster_m=2, sm_budget=0)
        self.assertEqual(test.geometry['counter_diagnostic']['cutlass_1sm_cluster_m'], 2)
        for forbidden in ('performance_accepted', 'samples_ms', 'p50', 'pflops', 'results'):
            self.assertNotIn(forbidden, json.dumps(test.geometry))
            self.assertNotIn(forbidden, test.stdout.getvalue())

    def test_counter_failures_keep_phase_close_plan_and_never_enter_formal_sampling(self):
        for fault in ('create', 'correctness_pre', 'warmup', 'nvtx_launch', 'correctness_post', 'cleanup'):
            with self.subTest(fault=fault), self.counter_stubs(fault) as test:
                with self.assertRaisesRegex(RuntimeError, fault):
                    bench.run_cutlass_counters(test.args, 'lt_lib', test.geometry, 'cutlass_lib')
            diagnostic = test.geometry['counter_diagnostic']
            self.assertEqual((diagnostic['state'], diagnostic['phase']), ('failed', fault))
            self.assertEqual(test.plan.close.call_count, int(fault != 'create'))
            if fault in ('create', 'correctness_pre', 'warmup'):
                self.assertNotIn('push', test.calls)
            if fault == 'nvtx_launch':
                self.assertEqual(test.calls[-4:], ['run', 'pop', 'sync', 'close'])
            if fault == 'warmup':
                self.assertEqual(diagnostic['warmup']['additional_warmup_cuda_ms'], 40.)
            self.assertNotIn('performance_accepted', json.dumps(test.geometry))
            self.assertNotIn('COUNTERS DONE', test.stdout.getvalue())

    def test_counter_target_noop_cannot_reuse_the_valid_warmup_output(self):
        with self.counter_stubs('no_op_target') as test, \
                self.assertRaisesRegex(RuntimeError, 'unwritten NaN output'):
            bench.run_cutlass_counters(test.args, 'lt_lib', test.geometry, 'cutlass_lib')
        self.assertEqual(test.geometry['counter_diagnostic']['phase'], 'correctness_post')
        self.assertEqual(test.plan.run.call_count, 42)
        test.plan.close.assert_called_once()

    def test_counter_cli_rejects_missing_library_multiple_mnk_and_wrong_mode_before_cuda(self):
        matrix = self.root / 'matrix.json'
        matrix.write_text(json.dumps(self.payload()))
        base = ['probe', '--output', str(self.root / 'out.json'), '--cutlass-counters']
        for extra in ([], ['--matrix-json', str(matrix), '--launches', 'eager'],
                      ['--compare-cutlass-library', 'unused', '--launches', 'eager'],
                      ['--compare-cutlass-library', 'unused', '--matrix-json', str(matrix), '--launches', 'graph'],
                      ['--compare-cutlass-library', 'unused', '--matrix-json', str(matrix), '--precisions', 'fp8'],
                      ['--compare-cutlass-library', 'unused', '--matrix-json', str(matrix), '--launches', 'eager']):
            with self.subTest(extra=extra), mock.patch.object(sys, 'argv', base + extra), \
                    mock.patch.object(bench, 'torch') as torch, \
                    mock.patch.object(bench, 'Library', side_effect=AssertionError('no CUDA library')), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                bench.main()
            torch.cuda.get_device_properties.assert_not_called()
        self.assertFalse((self.root / 'out.json').exists())

    def test_counter_cli_has_separate_hashed_diagnostic_schema_and_checkpoints_failures(self):
        matrix = self.root / 'matrix.json'
        payload = self.payload()
        payload['shapes'].pop()  # One physical geometry, two preserved aliases.
        matrix.write_text(json.dumps(payload))
        torch, lib = self.main_mocks()
        for failed in (False, True):
            output = self.root / f'counters-{failed}.json'
            argv = ['probe', '--output', str(output), '--matrix-json', str(matrix), '--launches', 'eager',
                    '--compare-cutlass-library', str(lib.path), '--cutlass-counters',
                    '--cutlass-swizzle-size', '4', '--cutlass-epilogue-n', '64']

            def run(a, library, geometry, cutlass_library):
                self.assertEqual((a.cutlass_swizzle_size, a.cutlass_epilogue_n), (4, 64))
                self.assertIs(library, lib)
                self.assertIs(cutlass_library, lib)
                self.assertNotIn('results', geometry)
                geometry['inputs'] = {'seed': 103, 'generator': 'torch_cuda', 'distribution': 'uniform'}
                geometry['counter_diagnostic'] = {'diagnostic_only': True, 'state': 'failed' if failed else 'succeeded'}
                if failed:
                    raise RuntimeError('counter failure')

            with mock.patch.object(sys, 'argv', argv), mock.patch.object(bench, 'torch', torch), \
                    mock.patch.object(bench, 'Library', return_value=lib), \
                    mock.patch.dict(sys.modules, {'cutlass': types.SimpleNamespace(Library=lambda _: lib)}), \
                    mock.patch.object(bench, 'run_cutlass_comparison', side_effect=AssertionError('formal comparison')), \
                    mock.patch.object(bench, 'run_geometry', side_effect=AssertionError('formal Lt')), \
                    mock.patch.object(bench, 'run_cutlass_counters', side_effect=run) as execute:
                with self.assertRaisesRegex(RuntimeError, 'counter failure') if failed else contextlib.nullcontext():
                    bench.main()
            execute.assert_called_once()
            result = json.loads(output.read_text())
            self.assertEqual(result['schema'], 'sm103_cutlass_counters_v1')
            self.assertEqual(result['state'], 'failed' if failed else 'succeeded')
            self.assertEqual(result['measurement_protocol'], 'single_gpu_nvtx_counter_only_v1')
            self.assertEqual(result['nvtx_range'], 'fuse_cutlass_1sm_counters')
            self.assertTrue(result['diagnostic_only'])
            self.assertFalse(result['formal_sampling_performed'])
            self.assertFalse(result['ncu_metrics_verified'])
            self.assertEqual((result['logical_shapes'], result['unique_geometries'], result['measured_ranks']), (2, 1, 1))
            self.assertEqual(result['matrix_sha256'], hashlib.sha256(matrix.read_bytes()).hexdigest())
            self.assertEqual(result['cutlass_library_sha256'], hashlib.sha256(lib.path.read_bytes()).hexdigest())
            self.assertEqual(result['frontend_sha256'], hashlib.sha256(ENTRY.read_bytes()).hexdigest())
            self.assertEqual(result['geometries'][0]['inputs']['seed'], 103)
            self.assertEqual(result['geometries'][0]['state'], result['state'])
            for forbidden in ('performance_accepted', 'samples_ms', 'p50', 'pflops', 'results'):
                self.assertNotIn(forbidden, json.dumps(result))


if __name__ == '__main__':
    unittest.main()
