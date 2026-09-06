"""CUDA-free checks of frozen-model A/B boundaries and capture reuse."""

from contextlib import redirect_stderr, redirect_stdout
import copy
import ctypes as ct
import io
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import operator_bench as bench
import operator_policy_ab as ab


class Call:
    def __init__(self, function):
        self.function, self.calls = function, []

    def __call__(self, *args):
        self.calls.append(args)
        return self.function(*args)


def model_fixture():
    return dict(world_size=4, sm_count=132, coefficients=dict(a=1000., e=0., b=1., t=2.),
                launch_prior_us=5., minimum_gain=.10, sha256='cpu-fixture-not-a-measurement')


def qkv_model_fixture():
    return dict(schema='mxfp8-qkv-forward-service-model-v1', operator='qkv_forward',
                world_size=4, sm_count=132,
                coefficients=dict(compute_gflop_sm_us=123.5, route_slot_task_us=7.25),
                minimum_gain=0., sha256='cpu-qkv-fixture-not-a-measurement')


def check(status):
    if status:
        raise RuntimeError(f'Fake native status: {status}')


class State:
    """Fake native dispatch, not a performance model or GPU measurement."""
    def __init__(self, unchanged=False):
        self.op, self.name, self.rank, self.world = 1, 'oproj_forward', 1, 4
        self.case, self.stream = {'id': 'cpu-only-fixture'}, None
        self.config, self.native, self.manual = {}, [0.] * 6 + [5., .10], 0
        self.events, self.output, self.unchanged = [], None, unchanged
        self.runtime = SimpleNamespace(check=check)
        self.library = SimpleNamespace(
            fuse_mxfp8_test_set_comm_ctas=Call(self.set_manual),
            fuse_mxfp8_test_set_oproj_comm_model=Call(self.set_model),
            fuse_mxfp8_test_get_oproj_comm_model=Call(self.get_model))

    @property
    def arm(self):
        return 'model' if self.native[0] else 'baseline'

    def set_manual(self, value):
        self.manual = value
        return 0

    def set_model(self, cp, sm, values):
        self.native = [cp, sm, *values] if cp else [0.] * 6 + [5., .10]
        return 0

    def get_model(self, output):
        output[:] = self.native
        return 0

    def refresh_config(self, requested_comm_ctas=0):
        if self.manual != 0:
            raise AssertionError('Frozen model was replaced by a manual CTA')
        self.config.update(requested_comm_ctas=requested_comm_ctas,
                           comm_ctas=8 if self.unchanged or self.arm == 'baseline' else 16,
                           tile_m=128, tile_n=256, tile_k=64, cluster_m=2,
                           sm_count=132, policy_enum=4,
                           dispatch='existing_bf16_production_auto')
        if self.arm == 'model':
            self.config['oproj_comm_model'] = bench.read_oproj_comm_model(self.library, self.runtime)
            self.config['dispatch'] = 'calibrated_model_or_domain_fallback'

    def launch(self, phase, mode):
        if (phase, mode) != ('forward', None):
            raise AssertionError('Only the forward operator may be launched')

        def function():
            self.events.append(('native_host_call', self.arm))
            self.output = self.arm
        return function

    def prepare(self, phase, mode, poison=False):
        if (phase, mode) != ('forward', None):
            raise AssertionError('Only forward buffers may be prepared')

        def prepare():
            if poison:
                self.output = 'poison'
        return prepare

    def verify(self, phase, mode):
        if self.output != self.arm or (phase, mode) != ('forward', None):
            raise AssertionError('Verification is not checking the current arm output')
        self.events.append(('verify', self.arm))
        return {'all_ranks_finite': True, 'max_abs': 0.0, 'relative_rmse': 0.0}

    def flops(self, phase):
        return 1000


class QkvState(State):
    """Only the QKV 3/5 ABI is exposed; an OProj call must fail this fixture."""
    def __init__(self, unchanged=False):
        super().__init__(unchanged)
        self.op, self.name, self.native = 0, 'qkv_forward', [0.] * 5
        self.library = SimpleNamespace(
            fuse_mxfp8_test_set_comm_ctas=Call(self.set_manual),
            fuse_mxfp8_test_set_qkv_forward_comm_model=Call(self.set_model),
            fuse_mxfp8_test_get_qkv_forward_comm_model=Call(self.get_model))

    def set_model(self, cp, sm, values):
        if cp:
            if len(values) != 3:
                raise AssertionError('QKV setter requires exactly three doubles')
            self.native = [cp, sm, *values]
        else:
            if sm != 0 or values is not None:
                raise AssertionError('QKV disable must use (0, 0, nullptr)')
            self.native = [0.] * 5
        return 0

    def get_model(self, output):
        if len(output) != 5:
            raise AssertionError('QKV getter requires exactly five doubles')
        output[:] = self.native
        return 0

    def refresh_config(self, requested_comm_ctas=0):
        if self.manual != 0 or requested_comm_ctas != 0:
            raise AssertionError('A/B must use native auto/model, not a manual winner')
        self.config.update(requested_comm_ctas=0,
                           comm_ctas=12 if self.unchanged or self.arm == 'baseline' else 18,
                           tile_m=128, tile_n=256, tile_k=64, cluster_m=2,
                           sm_count=132, policy_enum=4,
                           dispatch='existing_bf16_production_auto')
        if self.arm == 'model':
            self.config['qkv_forward_comm_model'] = bench.read_qkv_forward_comm_model(
                self.library, self.runtime)
            self.config['dispatch'] = 'calibrated_model_or_domain_fallback'


def distributed(state):
    return SimpleNamespace(all_gather_object=lambda output, value:
                           output.__setitem__(slice(None), [copy.deepcopy(value)] * state.world))


def invocation_runtime(state, model_time=5):
    class Invocation:
        def __init__(self, function, launch, warmup, prepare, stream):
            self.function, self.prepare, self.launch = function, prepare, launch
            self.arm = state.arm
            self.warmups = 0
            state.events.append(('create', self.arm, launch, warmup))
            if launch == 'graph':
                function()

        def once(self, prepare=None):
            if state.arm != self.arm:
                raise AssertionError('Wrong native model is installed for the captured arm')
            self.warmups += 1
            (prepare or self.prepare)()
            if self.launch == 'graph':
                state.output = self.arm
            else:
                self.function()

        def measure(self, iterations, flops):
            state.events.append(('measure', state.arm, self.launch, iterations, self.warmups))
            self.warmups = 0
            for _ in range(iterations):
                self.once()
            self.warmups = 0
            duration = 10 if state.arm == 'baseline' else model_time
            return dict(p50_us=duration, samples_us=[duration] * iterations,
                        rank_samples_us=[[duration] * iterations for _ in range(state.world)],
                        flops_per_gpu=flops)

    return SimpleNamespace(Invocation=Invocation)


class FrozenPolicyABTest(unittest.TestCase):
    def test_external_trace_is_not_formal_evidence(self):
        for model_flag in ('--oproj-comm-model', '--qkv-comm-model'):
            argv = ['--cp', '4', model_flag, 'model.json']
            self.assertEqual(ab.measurement_contract(ab.arguments(argv)),
                             dict(formal_ab_protocol=True))
            traced = ab.measurement_contract(ab.arguments([*argv, '--diagnostic-trace']))
            self.assertIs(traced['formal_ab_protocol'], False)
            self.assertIs(traced['diagnostic_trace'], True)
            self.assertEqual((ab.ROUNDS, ab.WARMUP, ab.ITERATIONS), (3, 10, 50))

    def test_cli_has_fixed_formal_protocol_and_no_search_knobs(self):
        args = ab.arguments(['--cp', '4', '--oproj-comm-model', 'model.json', '--full'])
        self.assertEqual(args.operators, 'oproj_forward')
        self.assertEqual(args.launches, 'eager,graph')
        self.assertEqual(args.arm_order, 'baseline-model')
        reverse = ab.arguments(['--cp', '4', '--oproj-comm-model', 'model.json', '--arm-order', 'model-baseline'])
        self.assertEqual(ab.arm_sequence(reverse.arm_order), ('model', 'baseline'))
        self.assertEqual((ab.ROUNDS, ab.WARMUP, ab.ITERATIONS), (3, 10, 50))
        rows = bench.select_cases(args, 4)
        expected = [case for case in bench.forward_matrix()
                    if case['cp'] == 4 and case['direction'] == 'a2a_gemm']
        self.assertEqual(rows, expected)
        for extra in (['--comm-ctas', '16'], ['--iterations', '7'], ['--rounds', '1'],
                      ['--operators', 'qkv_forward'], ['--launches', 'graph,graph'],
                      ['--launches', ''], ['--launches', 'native'], ['--arm-order', 'random']):
            with self.subTest(extra=extra), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                ab.arguments(['--cp', '4', '--oproj-comm-model', 'model.json', *extra])

    def test_model_and_baseline_use_real_native_model_and_manual_zero(self):
        state, model = State(), model_fixture()
        ab.set_arm(state.library, state.runtime, model, 'model')
        self.assertEqual(state.native, [4, 132, 1000., 0., 1., 2., 5., .10])
        self.assertEqual(bench.read_oproj_comm_model(state.library, state.runtime), model)
        ab.set_arm(state.library, state.runtime, model, 'baseline')
        self.assertEqual(state.native[:6], [0.] * 6)
        self.assertNotIn('_fuse_oproj_comm_model', vars(state.library))
        self.assertEqual(state.library.fuse_mxfp8_test_set_comm_ctas.calls, [(0,), (0,)])
        with self.assertRaises(ValueError):
            ab.set_arm(state.library, state.runtime, model, 'winner')

    def test_disable_is_verified_and_native_errors_propagate(self):
        state = State()
        state.native[0] = 4
        state.library.fuse_mxfp8_test_set_oproj_comm_model = Call(lambda *args: 0)
        with self.assertRaises(RuntimeError):
            ab.set_arm(state.library, state.runtime, model_fixture(), 'baseline')
        state.library.fuse_mxfp8_test_set_oproj_comm_model = Call(lambda *args: 1)
        with self.assertRaises(RuntimeError):
            ab.set_arm(state.library, state.runtime, model_fixture(), 'model')

    def test_activation_clears_stale_metadata_and_deep_copies_config(self):
        state, model = State(), model_fixture()
        enabled = ab.activate(state, model, 'model', distributed(state))
        baseline = ab.activate(state, model, 'baseline', distributed(state))
        self.assertNotIn('oproj_comm_model', baseline)
        self.assertEqual(enabled['oproj_comm_model'], model)
        state.config['tile_n'] = 320
        self.assertEqual(baseline['tile_n'], 256)

    def test_rank_config_disagreement_is_an_error(self):
        state = State()
        dist = SimpleNamespace(all_gather_object=lambda output, value:
                               output.__setitem__(slice(None), [value] * 3 + [dict(value, comm_ctas=24)]))
        with self.assertRaises(RuntimeError):
            ab.activate(state, model_fixture(), 'model', dist)

    def test_each_arm_captured_once_and_three_unselected_pairs_keep_vectors(self):
        state = State()
        result = ab.measure_launch(state, None, model_fixture(), 'graph',
                                   invocation_runtime(state), distributed(state))
        self.assertEqual([event[:3] for event in state.events if event[0] == 'create'],
                         [('create', 'baseline', 'graph'), ('create', 'model', 'graph')])
        measured = [event for event in state.events if event[0] == 'measure']
        self.assertEqual([event[1] for event in measured], ['baseline', 'model'] * 3)
        self.assertEqual([event[3] for event in measured], [50] * 6)
        self.assertEqual([event[4] for event in measured], [11, 11, 10, 10, 10, 10])
        self.assertEqual(len([event for event in state.events if event[0] == 'native_host_call']), 2)
        self.assertEqual(len([event for event in state.events if event[0] == 'verify']), 4)
        self.assertEqual(result['paired_summary']['paired_speedups'], [2, 2, 2])
        for pair in result['rounds']:
            for arm in ab.ARMS:
                self.assertEqual(len(pair[arm]['timing']['samples_us']), 50)
                self.assertEqual(len(pair[arm]['timing']['rank_samples_us']), 4)
                self.assertEqual('oproj_comm_model' in pair[arm]['config'], arm == 'model')
        self.assertTrue(result['kernel_configuration_changed'])
        self.assertEqual(result['capture_order'], ['baseline', 'model'])
        self.assertEqual([pair['execution_order'] for pair in result['rounds']], [['baseline', 'model']] * 3)
        self.assertEqual(state.arm, 'baseline')
        self.assertNotIn('oproj_comm_model', state.config)

    def test_eager_keeps_native_selector_calls_and_retains_unchanged_regressions(self):
        state = State(unchanged=True)
        result = ab.measure_launch(state, None, model_fixture(), 'eager',
                                   invocation_runtime(state, model_time=20), distributed(state))
        self.assertFalse(result['kernel_configuration_changed'])
        self.assertEqual(result['paired_summary']['paired_speedups'], [.5, .5, .5])
        self.assertAlmostEqual(result['paired_summary']['geometric_mean_speedup'], .5)
        for arm in ab.ARMS:
            self.assertEqual(state.events.count(('native_host_call', arm)), 1 + 3 * (10 + 50))
        self.assertTrue(all(call == (0,) for call in state.library.fuse_mxfp8_test_set_comm_ctas.calls))

    def test_reverse_order_controls_capture_and_execution_without_inverting_ratio(self):
        for launch in ('eager', 'graph'):
            with self.subTest(launch=launch):
                state = State()
                args = SimpleNamespace(arm_order='model-baseline')
                result = ab.measure_launch(state, args, model_fixture(), launch,
                                           invocation_runtime(state), distributed(state))
                self.assertEqual([event[1] for event in state.events if event[0] == 'create'], ['model', 'baseline'])
                self.assertEqual([event[1] for event in state.events if event[0] == 'measure'], ['model', 'baseline'] * 3)
                self.assertEqual(result['capture_order'], ['model', 'baseline'])
                self.assertEqual([pair['execution_order'] for pair in result['rounds']], [['model', 'baseline']] * 3)
                self.assertEqual(result['paired_summary']['paired_speedups'], [2, 2, 2])
                self.assertAlmostEqual(result['paired_summary']['geometric_mean_speedup'], 2)
                self.assertEqual(state.arm, 'baseline')

    def test_capture_config_change_cannot_be_timed_and_cleanup_disables_model(self):
        state = State()
        runtime = invocation_runtime(state)
        original = runtime.Invocation

        def changing_capture(*args):
            invocation = original(*args)
            state.unchanged = not state.unchanged
            return invocation

        runtime.Invocation = changing_capture
        # Baseline is independent of unchanged; the model capture changes 8 -> 16.
        with self.assertRaises(RuntimeError):
            ab.measure_launch(state, None, model_fixture(), 'graph', runtime, distributed(state))
        self.assertFalse(any(event[0] == 'measure' for event in state.events))
        self.assertEqual(state.arm, 'baseline')

    def test_summary_rejects_missing_pairs_labels_and_nonfinite_times(self):
        rounds = [dict(round=index, baseline=dict(arm='baseline', timing={'p50_us': 10}),
                       model=dict(arm='model', timing={'p50_us': 5})) for index in range(1, 4)]
        self.assertEqual(ab.paired_summary(rounds)['paired_speedups'], [2] * 3)
        with self.assertRaises(ValueError):
            ab.paired_summary(rounds[:2])
        for value in (0, -1, math.inf, math.nan):
            bad = copy.deepcopy(rounds)
            bad[0]['model']['timing']['p50_us'] = value
            with self.assertRaises(ValueError):
                ab.paired_summary(bad)
        rounds[0]['model']['arm'] = 'baseline'
        with self.assertRaises(ValueError):
            ab.paired_summary(rounds)

    def test_existing_output_is_rejected_before_cuda_initialization(self):
        with patch.object(Path, 'exists', return_value=True), \
                patch.object(bench, 'initialize_runtime') as initialize:
            with self.assertRaises(FileExistsError):
                ab.main(['--cp', '4', '--oproj-comm-model', 'model.json'])
        initialize.assert_not_called()

    def test_main_reuses_one_case_for_both_launches_and_refuses_profiling(self):
        for profiling in (0, 1):
            with self.subTest(profiling=profiling):
                state, model = State(), model_fixture()
                state.library.fuse_mxfp8_test_profiling_enabled = Call(lambda: profiling)
                state.close = lambda: state.events.append(('close',))
                dist = distributed(state)
                dist.init_process_group, dist.destroy_process_group = lambda *a, **k: None, lambda: None
                dist.get_rank, dist.broadcast_object_list = lambda: 0, lambda *a, **k: None
                props = SimpleNamespace(name='CPU fixture', major=9, minor=0,
                                        multi_processor_count=132, total_memory=1)
                torch = SimpleNamespace(__version__='mock', version=SimpleNamespace(cuda='mock'),
                                        cuda=SimpleNamespace(set_device=lambda rank: None,
                                                             get_device_properties=lambda rank: props,
                                                             empty_cache=lambda: None),
                                        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace())))
                runtime = invocation_runtime(state)
                runtime.CudaRuntime = lambda: state.runtime
                reports = []
                with patch.dict('os.environ', WORLD_SIZE='4', LOCAL_RANK='0'), \
                        patch.object(Path, 'exists', return_value=False), \
                        patch.object(bench, 'select_cases', return_value=[state.case]), \
                        patch.object(bench, 'initialize_runtime'), \
                        patch.object(bench, 'torch', torch, create=True), \
                        patch.object(bench, 'dist', dist, create=True), \
                        patch.object(bench, 'rt', runtime, create=True), \
                        patch.object(bench, 'source_hashes', return_value={}), \
                        patch.object(bench, 'configure_oproj_comm_model', return_value=model), \
                        patch.object(bench, 'OperatorCase', return_value=state) as create, \
                        patch.object(bench, 'checkpoint', side_effect=lambda path, report:
                                     reports.append(copy.deepcopy(report))), \
                        patch.object(ab.ct, 'CDLL', return_value=state.library), \
                        patch.object(ab.subprocess, 'check_output', return_value='mock clock snapshot'), \
                        redirect_stdout(io.StringIO()):
                    if profiling:
                        with self.assertRaisesRegex(RuntimeError, 'profiling-enabled'):
                            ab.main(['--cp', '4', '--oproj-comm-model', 'model.json'])
                        create.assert_not_called()
                        self.assertEqual(reports, [])
                    else:
                        ab.main(['--cp', '4', '--oproj-comm-model', 'model.json', '--arm-order', 'model-baseline'])
                        create.assert_called_once()
                        self.assertEqual(state.events.count(('close',)), 1)
                        self.assertEqual(sum(event[0] == 'create' for event in state.events), 4)
                        report = reports[-1]
                        self.assertTrue(report['complete'])
                        self.assertEqual(report['schema'], 'mxfp8-frozen-policy-ab-v1')
                        self.assertEqual(report['scope'], 'OProj_F_only_not_2F2B_total')
                        self.assertFalse(report['production_policy_written'])
                        self.assertEqual([row['launch'] for row in report['cases'][0]['records']],
                                         ['eager', 'graph'])
                        self.assertEqual(report['gpu_clock_snapshot'], 'mock clock snapshot')
                        self.assertEqual(report['args']['arm_order'], 'model-baseline')
                        self.assertEqual(report['timing']['capture_order'], ['model', 'baseline'])
                        self.assertEqual(report['timing']['ab_order'],
                                         'model_baseline_repeated_three_times_per_case_and_launch')


class QkvFrozenPolicyABTest(unittest.TestCase):
    def test_cli_selects_qkv_matrix_without_silently_dropping_fallback_shapes(self):
        args = ab.arguments(['--cp', '4', '--qkv-comm-model', 'model.json', '--full'])
        self.assertEqual(args.operators, 'qkv_forward')
        self.assertIsNone(args.oproj_comm_model)
        rows = bench.select_cases(args, 4)
        expected = [case for case in bench.forward_matrix()
                    if case['cp'] == 4 and case['direction'] == 'gemm_a2a']
        self.assertEqual(rows, expected)
        self.assertTrue(any(case['m'] < 32768 for case in rows))
        self.assertTrue(any(case['m'] >= 32768 for case in rows))
        for extra in (['--oproj-comm-model', 'other.json'], ['--comm-ctas', '18'],
                      ['--operators', 'oproj_forward'], ['--backends', 'pure_cublas']):
            with self.subTest(extra=extra), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                ab.arguments(['--cp', '4', '--qkv-comm-model', 'model.json', *extra])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            ab.arguments(['--cp', '4'])

    def test_model_uses_qkv_three_five_abi_and_clears_prior_manual_override(self):
        state, model = QkvState(), qkv_model_fixture()
        state.manual = 24
        ab.set_arm(state.library, state.runtime, model, 'model')
        self.assertEqual(state.manual, 0)
        self.assertEqual(state.native, [4, 132, 123.5, 7.25, 0.])
        setter = state.library.fuse_mxfp8_test_set_qkv_forward_comm_model
        getter = state.library.fuse_mxfp8_test_get_qkv_forward_comm_model
        self.assertEqual(setter.argtypes, [ct.c_int32, ct.c_int32, ct.POINTER(ct.c_double)])
        self.assertEqual(getter.argtypes, [ct.POINTER(ct.c_double)])
        self.assertEqual((setter.restype, getter.restype), (ct.c_int, ct.c_int))
        self.assertEqual(list(setter.calls[0][2]), [123.5, 7.25, 0.])
        self.assertEqual(len(getter.calls[0][0]), 5)
        model['coefficients']['route_slot_task_us'] = 999.
        self.assertEqual(bench.read_qkv_forward_comm_model(state.library, state.runtime), qkv_model_fixture())
        ab.set_arm(state.library, state.runtime, model, 'baseline')
        self.assertEqual(setter.calls[-1], (0, 0, None))
        self.assertEqual(state.native, [0.] * 5)
        self.assertNotIn('_fuse_qkv_forward_comm_model', vars(state.library))
        self.assertEqual(state.library.fuse_mxfp8_test_set_comm_ctas.calls, [(0,), (0,)])

    def test_qkv_native_errors_and_wrong_coefficient_readback_cannot_be_labelled_model(self):
        for failure in ('setter', 'getter', 'coefficient', 'disable'):
            with self.subTest(failure=failure):
                state, model = QkvState(), qkv_model_fixture()
                if failure == 'setter':
                    state.library.fuse_mxfp8_test_set_qkv_forward_comm_model = Call(lambda *args: 1)
                elif failure == 'getter':
                    state.library.fuse_mxfp8_test_get_qkv_forward_comm_model = Call(lambda output: 1)
                elif failure == 'coefficient':
                    def corrupt(output):
                        state.get_model(output)
                        output[3] += 1
                        return 0
                    state.library.fuse_mxfp8_test_get_qkv_forward_comm_model = Call(corrupt)
                else:
                    ab.set_arm(state.library, state.runtime, model, 'model')
                    state.library.fuse_mxfp8_test_set_qkv_forward_comm_model = Call(lambda *args: 0)
                with self.assertRaises(RuntimeError):
                    ab.set_arm(state.library, state.runtime, model,
                               'baseline' if failure == 'disable' else 'model')

    def test_operator_and_model_family_must_match_before_capture(self):
        for state, model in ((State(), qkv_model_fixture()), (QkvState(), model_fixture())):
            with self.subTest(operator=state.name), self.assertRaises(ValueError):
                ab.measure_launch(state, None, model, 'graph', invocation_runtime(state), distributed(state))
            self.assertEqual(state.events, [])

    def test_qkv_metadata_is_deep_copied_and_removed_for_baseline(self):
        state, model = QkvState(), qkv_model_fixture()
        enabled = ab.activate(state, model, 'model', distributed(state))
        state.config['qkv_forward_comm_model']['coefficients']['route_slot_task_us'] = 999.
        self.assertEqual(enabled['qkv_forward_comm_model'], model)
        baseline = ab.activate(state, model, 'baseline', distributed(state))
        self.assertNotIn('qkv_forward_comm_model', baseline)
        self.assertNotIn('oproj_comm_model', enabled)
        self.assertEqual(baseline['dispatch'], 'existing_bf16_production_auto')

    def test_rank_model_disagreement_is_rejected_even_with_the_same_tile_and_ctas(self):
        state = QkvState()

        def disagree(output, config):
            output[:] = [copy.deepcopy(config) for _ in range(state.world)]
            output[-1]['qkv_forward_comm_model']['sha256'] = 'different-calibration'

        with self.assertRaisesRegex(RuntimeError, 'Ranks selected different'):
            ab.activate(state, qkv_model_fixture(), 'model', SimpleNamespace(all_gather_object=disagree))

    def test_qkv_graph_reverse_order_reuses_captures_and_preserves_unchanged_regressions(self):
        state = QkvState(unchanged=True)
        result = ab.measure_launch(state, SimpleNamespace(arm_order='model-baseline'),
                                   qkv_model_fixture(), 'graph', invocation_runtime(state, model_time=20),
                                   distributed(state))
        self.assertEqual(result['scope'], 'QKV_F_only_not_2F2B_total')
        self.assertEqual(result['phase'], 'forward')
        self.assertFalse(result['kernel_configuration_changed'])
        self.assertEqual(result['capture_order'], ['model', 'baseline'])
        self.assertEqual(result['paired_summary']['paired_speedups'], [.5] * 3)
        self.assertEqual(sum(event[0] == 'native_host_call' for event in state.events), 2)
        self.assertEqual(sum(event[0] == 'verify' for event in state.events), 4)
        for pair in result['rounds']:
            self.assertEqual(pair['execution_order'], ['model', 'baseline'])
            for arm in ab.ARMS:
                record = pair[arm]
                self.assertEqual(len(record['timing']['samples_us']), 50)
                self.assertEqual([len(row) for row in record['timing']['rank_samples_us']], [50] * 4)
                self.assertEqual('qkv_forward_comm_model' in record['config'], arm == 'model')
                self.assertNotIn('oproj_comm_model', record['config'])
                self.assertEqual(record['config']['requested_comm_ctas'], 0)
        self.assertEqual(state.arm, 'baseline')
        self.assertNotIn('qkv_forward_comm_model', state.config)

    def test_qkv_main_emits_separate_schema_scope_and_only_configures_the_qkv_model(self):
        state, model = QkvState(), qkv_model_fixture()
        state.library.fuse_mxfp8_test_profiling_enabled = Call(lambda: 0)
        state.close = lambda: state.events.append(('close',))
        dist = distributed(state)
        dist.init_process_group, dist.destroy_process_group = lambda *a, **k: None, lambda: None
        dist.get_rank, dist.broadcast_object_list = lambda: 0, lambda *a, **k: None
        props = SimpleNamespace(name='CPU fixture', major=9, minor=0,
                                multi_processor_count=132, total_memory=1)
        torch = SimpleNamespace(__version__='mock', version=SimpleNamespace(cuda='mock'),
                                cuda=SimpleNamespace(set_device=lambda rank: None,
                                                     get_device_properties=lambda rank: props,
                                                     empty_cache=lambda: None),
                                backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace())))
        runtime = invocation_runtime(state)
        runtime.CudaRuntime = lambda: state.runtime
        reports, sources = [], {'cpu-model.json': 'cpu-sha'}
        with patch.dict('os.environ', WORLD_SIZE='4', LOCAL_RANK='0'), \
                patch.object(Path, 'exists', return_value=False), \
                patch.object(bench, 'select_cases', return_value=[state.case]), \
                patch.object(bench, 'initialize_runtime'), \
                patch.object(bench, 'torch', torch, create=True), \
                patch.object(bench, 'dist', dist, create=True), \
                patch.object(bench, 'rt', runtime, create=True), \
                patch.object(bench, 'source_hashes', return_value=sources), \
                patch.object(bench, 'configure_qkv_forward_comm_model', return_value=model) as configure, \
                patch.object(bench, 'configure_oproj_comm_model') as oproj, \
                patch.object(bench, 'OperatorCase', return_value=state) as create, \
                patch.object(bench, 'checkpoint', side_effect=lambda path, report:
                             reports.append(copy.deepcopy(report))), \
                patch.object(ab.ct, 'CDLL', return_value=state.library), \
                patch.object(ab.subprocess, 'check_output', return_value='mock clock snapshot'), \
                redirect_stdout(io.StringIO()):
            ab.main(['--cp', '4', '--qkv-comm-model', 'model.json', '--arm-order', 'model-baseline'])
        configure.assert_called_once_with(Path('model.json'), 4, 132, state.library,
                                          state.runtime, sources, workers=dist)
        oproj.assert_not_called()
        create.assert_called_once()
        self.assertEqual(state.events.count(('close',)), 1)
        report = reports[-1]
        self.assertTrue(report['complete'])
        self.assertEqual(report['schema'], 'mxfp8-qkv-forward-frozen-policy-ab-v1')
        self.assertEqual(report['scope'], 'QKV_F_only_not_2F2B_total')
        self.assertEqual(report['qkv_forward_comm_model'], model)
        self.assertNotIn('oproj_comm_model', report)
        self.assertFalse(report['production_policy_written'])
        self.assertFalse(report['milestone_evidence'])
        self.assertEqual(report['cases'][0]['operator'], 'qkv_forward')
        self.assertEqual([row['launch'] for row in report['cases'][0]['records']], ['eager', 'graph'])
        self.assertEqual(report['timing']['sample_statistic'], 'sample_wise_max_across_ranks')
        self.assertEqual(report['timing']['included'], ['software_weight_dequant', 'BF16_A2A', 'BF16_GEMM'])
        self.assertEqual(report['sources']['cpu-model.json'], 'cpu-sha')
        self.assertEqual(report['timing']['capture_order'], ['model', 'baseline'])


if __name__ == '__main__':
    unittest.main()
