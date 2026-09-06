"""CUDA-free checks for bounded CTA selection and unbiased A/B bookkeeping."""

import argparse
from contextlib import nullcontext
import math
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from operator_sweep import (arguments, comm_candidates, interleaved_schedule,
                            positive_ratio, prepare_wgrad_inputs, shortlist, summarize_candidates,
                            summarize_round, sweep_case, validate_config, wgrad_candidates)
from operator_sweep import QKV_TILES, measurement_contract, qkv_tile_candidates
from operator_bench import OperatorCase


class CompleteBackwardSweepTest(unittest.TestCase):
    def test_external_trace_keeps_counts_but_cannot_be_formal(self):
        plain = arguments(['--cp', '4'])
        traced = arguments(['--cp', '4', '--diagnostic-trace'])
        self.assertEqual(measurement_contract(plain),
                         dict(formal_ab_protocol=True, ab_kind='formal_paired_ab'))
        for field in ('rounds', 'warmup', 'iterations', 'search_warmup',
                      'search_iterations', 'pair_order', 'launch'):
            self.assertEqual(getattr(plain, field), getattr(traced, field))
        contract = measurement_contract(traced)
        self.assertFalse(contract['formal_ab_protocol'])
        self.assertTrue(contract['diagnostic_trace'])
        self.assertEqual(contract['ab_kind'], 'diagnostic_paired_ab')
        plain.iterations = 7
        self.assertFalse(measurement_contract(plain)['formal_ab_protocol'])

    def test_boundary_composes_deferred_b_then_w(self):
        events = []
        state = SimpleNamespace(launch=lambda phase, mode: lambda: events.append((phase, mode)))
        for mode in ('immediate', 'deferred'):
            call = OperatorCase.launch_boundary(state, 'total', mode)
            self.assertEqual(events, [])
            call()
            self.assertEqual(events, [('total', mode)] if mode == 'immediate'
                             else [('data', mode), ('weight', mode)])
            events.clear()

    def test_reverse_order_preserves_ratio_direction_and_pair_provenance(self):
        schedule = interleaved_schedule([4, 8], 2, 'candidate-auto')
        self.assertEqual(schedule, [(1, 4), (1, 0), (1, 8), (1, 0),
                                    (2, 4), (2, 0), (2, 8), (2, 0)])
        rows = [measurement(4, 4, 5), measurement(0, 12, 10),
                measurement(8, 8, 8), measurement(0, 12, 12)]
        result = summarize_round(1, rows, order='candidate-auto')
        self.assertEqual(result['execution_order'], [4, 0, 8, 0])
        self.assertEqual([p['speedup'] for p in result['pairs']], [2, 1.5])
        self.assertIs(result['pairs'][0]['auto'], rows[1])
        with self.assertRaises(ValueError):
            summarize_round(1, rows)

    def test_complete_sweep_checks_real_weight_updates_for_both_betas(self):
        for mode in ('immediate', 'deferred'):
            with self.subTest(mode=mode):
                state = SimpleNamespace(op=2, name='qkv_backward', world=4, rank=1,
                                        case={'id': 'cpu-fixture'}, stream=None, comm=0, dw=0, output=False)
                events, captures, iterations = [], [], []
                def setter(c):
                    state.comm = c
                    return 0
                state.library = SimpleNamespace(fuse_mxfp8_test_set_comm_ctas=setter)
                state.runtime = SimpleNamespace(check=lambda status: self.assertEqual(status, 0))
                state.refresh_config = lambda requested_comm_ctas: setattr(
                    state, 'config', measurement(requested_comm_ctas, requested_comm_ctas or 12, 1)['config'])
                state.tensors = {'dw': SimpleNamespace(fill_=lambda value: setattr(state, 'dw', value))}
                state.arena = SimpleNamespace(reset_control=lambda: None)
                def prepare(phase, selected_mode, poison=False):
                    self.assertEqual((phase, selected_mode), ('total', mode))
                    def reset():
                        state.dw = .125 if mode == 'deferred' else 0
                        state.output = False
                    return reset
                state.prepare = prepare
                def launch(phase, selected_mode):
                    def call():
                        events.append(phase)
                        if phase in ('data', 'total'):
                            state.output = True
                        if phase in ('weight', 'total'):
                            state.dw = state.dw + 2 if mode == 'deferred' else 2
                    return call
                state.launch = launch
                state.launch_boundary = lambda phase, selected_mode: OperatorCase.launch_boundary(
                    state, phase, selected_mode)
                def verify(phase, selected_mode, accumulated=1):
                    self.assertEqual((phase, selected_mode), ('total', mode))
                    self.assertTrue(state.output)
                    self.assertEqual(state.dw, 2 * accumulated + (.125 if mode == 'deferred' else 0))
                    return {'weight_gradient': {'pass': True}}
                state.verify = verify
                state.flops = lambda phase: 4 if phase == 'total' else 2
                class Invocation:
                    def __init__(self, call, launch, warmup, reset, stream):
                        self.call, self.reset, self.comm = call, reset, state.comm
                        captures.append(self.comm)
                    def once(self, prepare=None):
                        if state.comm != self.comm:
                            raise AssertionError('Captured CTA budget changed')
                        (prepare or self.reset)()
                        self.call()
                    def measure(self, count, flops):
                        if flops != 4:
                            raise AssertionError('Full backward FLOPs missing W')
                        iterations.append(count)
                        for _ in range(count):
                            self.once()
                        return dict(p50_us=10 if not self.comm else 9)
                args = arguments(['--cp', '4', '--operators', 'qkv_backward', '--backward-phase', 'total',
                                  '--comm-ctas', '0,4', '--pair-order', 'candidate-auto'])
                dist = SimpleNamespace(all_gather_object=lambda out, value:
                                       out.__setitem__(slice(None), [value] * 4))
                result = sweep_case(state, args, SimpleNamespace(Invocation=Invocation), dist, mode)
                self.assertEqual(result['phase'], 'total')
                self.assertEqual(result['beta'], int(mode == 'deferred'))
                self.assertEqual(captures, [0, 4])
                self.assertEqual(iterations.count(50), 6)
                for r in result['rounds']:
                    self.assertEqual(r['execution_order'], [4, 0])
                for r in result['search']:
                    key = 'nonzero_beta1_twice' if mode == 'deferred' else 'nonzero_beta0_overwrite'
                    self.assertIn(key, r['correctness'])
                self.assertEqual(events.count('data'), events.count('weight'))
                self.assertEqual(state.comm, 0)


class QkvTileSweepTest(unittest.TestCase):
    def test_parser_and_same_comm_contract(self):
        args = arguments(['--cp', '4', '--sweep-kind', 'qkv_tile', '--fixed-comm-ctas', '4'])
        self.assertEqual(args.operators, 'qkv_forward')
        self.assertEqual(args.qkv_tiles, [0, 2, 4, 6])
        self.assertEqual(qkv_tile_candidates('6,0,2'), [0, 6, 2])
        for value in ('0', '1,2', '0,7', '0,-1', '0,2,2', '0,x'):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                qkv_tile_candidates(value)

    def test_requested_tile_and_fixed_budget_are_checked(self):
        config = dict(requested_qkv_tile=6, fixed_comm_ctas=4, requested_comm_ctas=4,
                      comm_ctas=4, sm_count=132, tile_m=128, tile_n=320, tile_k=64, cluster_m=2)
        validate_config(6, config, 'qkv_tile')
        for changes in (dict(tile_n=256), dict(cluster_m=1), dict(comm_ctas=6),
                        dict(requested_comm_ctas=0), dict(requested_qkv_tile=0)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_config(6, dict(config, **changes), 'qkv_tile')

    def test_captures_are_reused_and_environment_restored_even_on_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail), patch.dict(os.environ, {'FUSE_QKV_GEMM_POLICY': 'original'}):
                captures, samples, checks = [], [], []
                state = SimpleNamespace(op=0, world=4, rank=1, stream=None,
                                        case={'id': 'cpu-only'}, name='qkv_forward', comm=0)
                def setter(value):
                    state.comm = value
                    return 0
                state.runtime = SimpleNamespace(check=lambda status: self.assertEqual(status, 0))
                state.library = SimpleNamespace(fuse_mxfp8_test_set_comm_ctas=setter)
                def refresh(requested_comm_ctas):
                    self.assertEqual(state.comm, 4)
                    value = os.environ.get('FUSE_QKV_GEMM_POLICY', 'm128n256')
                    n = int(value.removeprefix('m128n'))
                    cluster = next(c for width, c in QKV_TILES.values() if width == n)
                    state.config = dict(requested_comm_ctas=requested_comm_ctas, comm_ctas=state.comm,
                                        sm_count=132, tile_m=128, tile_n=n, tile_k=64, cluster_m=cluster)
                    if fail:
                        raise ValueError('injected config failure')
                state.refresh_config = refresh
                state.launch = lambda phase, mode: lambda: None
                state.prepare = lambda phase, mode, poison=False: lambda: None
                state.flops = lambda phase: 1
                state.verify = lambda phase, mode: checks.append((phase, mode)) or {'pass': True}
                class Invocation:
                    def __init__(self, function, launch, warmup, prepare, stream):
                        self.tile = state.config['tile_n']
                        captures.append(self.tile)
                    def once(self, prepare=None):
                        if self.tile != state.config['tile_n'] or state.comm != 4:
                            raise AssertionError('Capture/native configuration changed')
                    def measure(self, iterations, flops):
                        self.once()
                        samples.append((self.tile, iterations))
                        return dict(p50_us={256: 10, 128: 12, 320: 9}[self.tile])
                dist = SimpleNamespace(all_gather_object=lambda out, value:
                                       out.__setitem__(slice(None), [value] * 4))
                args = arguments(['--cp', '4', '--sweep-kind', 'qkv_tile', '--fixed-comm-ctas', '4',
                                  '--qkv-tiles', '0,2,5,6'])
                if fail:
                    with self.assertRaisesRegex(ValueError, 'injected'):
                        sweep_case(state, args, SimpleNamespace(Invocation=Invocation), dist)
                else:
                    result = sweep_case(state, args, SimpleNamespace(Invocation=Invocation), dist)
                    self.assertEqual(captures, [256, 128, 256, 320])
                    self.assertEqual(result['selected_requested_qkv_tile'], [6, 2])
                    self.assertEqual(len([x for x in samples if x[1] == 50]), 12)
                    self.assertTrue(all(x == ('forward', None) for x in checks))
                    self.assertEqual(result['baseline'], 'auto_tile_at_fixed_comm_not_unmodified_production_auto')
                self.assertEqual(state.comm, 0)
                self.assertEqual(os.environ['FUSE_QKV_GEMM_POLICY'], 'original')


def measurement(requested, actual, duration):
    return dict(requested_comm_ctas=requested,
                config=dict(requested_comm_ctas=requested, comm_ctas=actual,
                            sm_count=132, cluster_m=2),
                timing=dict(p50_us=duration))


def weight_measurement(requested, duration):
    actual = requested or 1
    tiles = {1: (128, 256, 64, 2), 2: (128, 128, 64, 2),
             3: (128, 128, 128, 2), 4: (128, 256, 64, 1),
             5: (128, 192, 64, 2), 6: (128, 256, 32, 2)}
    weight = dict(zip(('tile_m', 'tile_n', 'tile_k', 'cluster_m'), tiles[actual]))
    weight.update(policy_enum=actual, requested_policy=requested, output_dtype='fp32',
                  stages=3, dynamic_smem_bytes=215040, registers_per_thread=64)
    return dict(requested_wgrad_policy=requested, config={'weight_gemm': weight},
                timing={'p50_us': duration})


class OperatorSweepTest(unittest.TestCase):
    def test_candidates_include_auto_and_are_unique_even(self):
        self.assertEqual(comm_candidates('8,0,4'), [0, 8, 4])
        for value in ('0', '4,8', '0,4,4', '0,-2', '0,3', '0,x', '0,'):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                comm_candidates(value)

    def test_default_protocol(self):
        args = arguments(['--cp', '4'])
        self.assertEqual(args.comm_ctas, [0, 4, 8, 12, 16, 24, 32])
        self.assertEqual((args.search_warmup, args.search_iterations), (2, 7))
        self.assertEqual((args.warmup, args.iterations, args.rounds), (10, 50, 3))
        self.assertEqual((args.launch, args.seqs), ('graph', '131072'))

    def test_manual_override_must_be_honored(self):
        config = measurement(8, 8, 1)['config']
        validate_config(8, config)
        for changes in (dict(comm_ctas=12), dict(sm_count=8), dict(cluster_m=3),
                        dict(requested_comm_ctas=0)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_config(8, dict(config, **changes))

    def test_shortlist_excludes_auto_actual_and_deduplicates(self):
        search = [measurement(0, 8, 10), measurement(8, 8, 1),
                  measurement(4, 4, 6), measurement(12, 12, 5),
                  measurement(16, 12, 4), measurement(24, 24, 7)]
        self.assertEqual(shortlist(search), [16, 4])

    def test_shortlist_retains_slower_candidates_without_claiming_a_win(self):
        search = [measurement(0, 8, 1), measurement(4, 4, 6), measurement(12, 12, 5)]
        self.assertEqual(shortlist(search), [12, 4])
        self.assertEqual(shortlist(search[:2]), [4])
        with self.assertRaises(ValueError):
            shortlist([measurement(0, 8, 1), measurement(8, 8, 2)])

    def test_interleaved_rounds(self):
        self.assertEqual(interleaved_schedule([12, 4], 2),
                         [(1, 0), (1, 12), (1, 0), (1, 4),
                          (2, 0), (2, 12), (2, 0), (2, 4)])

    def test_speedup_uses_its_own_paired_auto(self):
        records = [measurement(0, 8, 10), measurement(12, 12, 5),
                   measurement(0, 8, 12), measurement(4, 4, 8)]
        result = summarize_round(1, records)
        self.assertEqual([pair['speedup'] for pair in result['pairs']], [2, 1.5])
        self.assertEqual(result['auto_self_ratios'], [1, 10 / 12])
        self.assertIs(result['pairs'][0]['auto'], records[0])
        summary = summarize_candidates([result, result])
        self.assertEqual([row['paired_speedups'] for row in summary], [[2, 2], [1.5, 1.5]])
        self.assertAlmostEqual(summary[0]['geometric_mean_speedup'], 2)

    def test_invalid_ratios_or_pair_order_are_errors(self):
        for duration in (0, -1, math.inf, math.nan):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                positive_ratio(1, duration)
        for records in ([], [measurement(0, 8, 1)],
                        [measurement(4, 4, 1), measurement(0, 8, 2)]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                summarize_round(1, records)

    def test_case_reuses_capture_and_never_launches_weight(self):
        captures, launches, poison_calls, checked = [], [], [], []
        state = SimpleNamespace(op=3, world=4, rank=1, stream=None,
                                case={'id': 'cpu-fixture'}, name='oproj_backward')
        state.runtime = SimpleNamespace(check=lambda status: self.assertEqual(status, 0))

        def setter(requested):
            state.requested = requested
            return 0

        def refresh(requested_comm_ctas):
            self.assertEqual(state.requested, requested_comm_ctas)
            state.config = measurement(requested_comm_ctas, requested_comm_ctas or 8, 1)['config']

        def launch(phase, mode):
            launches.append((phase, mode))
            return lambda: None

        def prepare(phase, mode, poison=False):
            if poison:
                # Record construction once, not the repeated control reset.
                poison_calls.append((phase, mode, state.requested))
            return lambda: None

        state.library = SimpleNamespace(fuse_mxfp8_test_set_comm_ctas=setter)
        state.refresh_config, state.launch, state.prepare = refresh, launch, prepare
        state.flops = lambda phase: 1000
        state.verify = lambda phase, mode: checked.append((phase, mode)) or {'pass': True}

        class Invocation:
            def __init__(self, function, launch, warmup, prepare, stream):
                self.requested = state.requested
                self.prepare = prepare
                captures.append(self.requested)

            def once(self, prepare=None):
                (prepare or self.prepare)()

            def measure(self, iterations, flops):
                if state.requested != self.requested:
                    raise AssertionError('Captured graph configuration was not restored')
                return {'p50_us': 20 / ((self.requested or 8) + 1), 'iterations': iterations}

        dist = SimpleNamespace(all_gather_object=lambda output, value:
                               output.__setitem__(slice(None), [value] * state.world))
        args = SimpleNamespace(comm_ctas=[0, 4, 8, 12], launch='graph', search_warmup=2,
                               search_iterations=7, warmup=10, iterations=50, rounds=3)
        result = sweep_case(state, args, SimpleNamespace(Invocation=Invocation), dist)
        self.assertEqual(captures, [0, 4, 8, 12])
        self.assertEqual(launches, [('data', 'deferred')] * 4)
        self.assertEqual(len(poison_calls), 4)
        self.assertTrue(all(phase == 'data' and mode == 'deferred' for phase, mode in checked))
        self.assertEqual(result['selected_requested_comm_ctas'], [12, 4])
        self.assertEqual(len(result['rounds']), 3)
        self.assertEqual(state.requested, 0)
        self.assertFalse(result['fixed_tile_single_factor_experiment'])

    def test_wgrad_defaults_and_candidate_validation(self):
        args = arguments(['--cp', '8', '--sweep-kind', 'wgrad'])
        self.assertEqual(args.operators, 'qkv_backward,oproj_backward')
        self.assertEqual(args.wgrad_policies, [0, 2, 3, 4])
        self.assertEqual(args.weight_modes, 'immediate,deferred')
        self.assertEqual(wgrad_candidates('4,0,1'), [0, 4, 1])
        self.assertEqual(wgrad_candidates('5,0,6'), [0, 5, 6])
        explicit = arguments(['--cp', '8', '--sweep-kind', 'wgrad', '--wgrad-policies', '0,5,6'])
        self.assertEqual(explicit.wgrad_policies, [0, 5, 6])
        for value in ('0', '1,2', '0,7', '0,-1', '0,2,2', '0,5,5', '0,x'):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                wgrad_candidates(value)

    def test_wgrad_shortlist_excludes_explicit_baseline(self):
        search = [weight_measurement(0, 10), weight_measurement(1, 1),
                  weight_measurement(2, 8), weight_measurement(3, 9), weight_measurement(4, 12)]
        self.assertEqual(shortlist(search, kind='wgrad'), [2, 3])
        for item in search:
            validate_config(item['requested_wgrad_policy'], item['config'], 'wgrad')
        with self.assertRaises(ValueError):
            validate_config(4, search[2]['config'], 'wgrad')

    def test_new_wgrad_candidates_keep_the_auto_baseline_and_native_resources(self):
        search = [weight_measurement(0, 10), weight_measurement(5, 9), weight_measurement(6, 8)]
        self.assertEqual(shortlist(search, kind='wgrad'), [6, 5])
        for item in search:
            validate_config(item['requested_wgrad_policy'], item['config'], 'wgrad')
        for requested, changes in ((0, dict(policy_enum=5, requested_policy=0)),
                                   (5, dict(policy_enum=6)), (5, dict(policy_enum=7)),
                                   (5, dict(registers_per_thread=0))):
            config = {'weight_gemm': dict(search[1]['config']['weight_gemm'], **changes)}
            with self.subTest(requested=requested, changes=changes), self.assertRaises(ValueError):
                validate_config(requested, config, 'wgrad')

    def test_wgrad_pairing_uses_policy_keys(self):
        result = summarize_round(1, [weight_measurement(0, 10), weight_measurement(2, 8),
                                    weight_measurement(0, 12), weight_measurement(3, 10)], 'wgrad')
        summary = summarize_candidates([result], 'wgrad')
        self.assertEqual([row['requested_wgrad_policy'] for row in summary], [2, 3])
        self.assertEqual([row['paired_speedups'] for row in summary], [[1.25], [1.2]])

    def test_wgrad_input_preparation_launches_one_untimed_data_path(self):
        events = []
        state = SimpleNamespace(op=2, stream=SimpleNamespace(synchronize=lambda: events.append('sync')))
        state.prepare = lambda phase, mode, poison: lambda: events.append(('prepare', phase, mode, poison))
        state.launch = lambda phase, mode: lambda: events.append(('launch', phase, mode))
        state.verify = lambda phase, mode: events.append(('verify', phase, mode)) or {'pass': True}
        runtime = SimpleNamespace(torch=SimpleNamespace(cuda=SimpleNamespace(stream=lambda _: nullcontext())),
                                  aligned_prepare=lambda prepare, stream: prepare())
        self.assertEqual(prepare_wgrad_inputs(state, runtime), {'pass': True})
        self.assertEqual(events, [('prepare', 'data', 'deferred', True),
                                  ('launch', 'data', 'deferred'), 'sync', ('verify', 'data', 'deferred')])

    def test_wgrad_reuses_captures_and_checks_nonzero_beta0_beta1(self):
        captures, launches, checks = [], [], []

        class Tensor:
            def fill_(self, value):
                self.value = value

        dw = Tensor()
        state = SimpleNamespace(op=2, world=4, rank=1, stream=None, requested=0,
                                case={'id': 'cpu-fixture'}, name='qkv_backward',
                                tensors={'dw': dw}, staging='valid')
        state.runtime = SimpleNamespace(check=lambda status: self.assertEqual(status, 0))
        state.arena = SimpleNamespace(reset_control=lambda: None)

        def setter(requested):
            state.requested = requested
            return 0

        def refresh(requested_comm_ctas):
            self.assertEqual(requested_comm_ctas, 0)
            state.config = weight_measurement(state.requested, 1)['config']

        def launch(phase, mode):
            self.assertEqual(phase, 'weight')
            launches.append((phase, mode))

            def call():
                self.assertEqual(state.staging, 'valid')
                dw.value = 2 + (dw.value if mode == 'deferred' else 0)
            return call

        def prepare(phase, mode, poison=False):
            self.assertEqual(phase, 'weight')
            return lambda: dw.fill_(0.125 if mode == 'deferred' else 0)

        def verify(phase, mode, accumulated=1):
            self.assertEqual(phase, 'weight')
            self.assertEqual(dw.value, 2 * accumulated + (0.125 if mode == 'deferred' else 0))
            checks.append((mode, accumulated))
            return {'pass': True}

        class Invocation:
            def __init__(self, function, launch, warmup, prepare, stream):
                self.function, self.prepare, self.requested = function, prepare, state.requested
                captures.append(self.requested)

            def once(self, prepare=None):
                (prepare or self.prepare)()
                self.function()

            def measure(self, iterations, flops):
                if state.requested != self.requested:
                    raise AssertionError('Captured WGrad policy was not restored')
                self.once()
                return {'p50_us': 20 / ((self.requested or 1) + 1), 'iterations': iterations}

        state.library = SimpleNamespace(fuse_mxfp8_test_set_wgrad_policy=setter)
        state.refresh_config, state.launch, state.prepare, state.verify = refresh, launch, prepare, verify
        state.flops = lambda phase: 1000
        dist = SimpleNamespace(all_gather_object=lambda output, value:
                               output.__setitem__(slice(None), [value] * state.world))
        args = SimpleNamespace(sweep_kind='wgrad', wgrad_policies=[0, 1, 2, 3, 4, 5, 6],
                               launch='graph', search_warmup=2, search_iterations=7,
                               warmup=10, iterations=50, rounds=3)
        for mode in ('immediate', 'deferred'):
            result = sweep_case(state, args, SimpleNamespace(Invocation=Invocation), dist, mode)
            self.assertEqual(result['phase'], 'weight')
            self.assertEqual(result['selected_requested_wgrad_policy'], [6, 5])
            expected = 'nonzero_beta1_twice' if mode == 'deferred' else 'nonzero_beta0_overwrite'
            self.assertTrue(all(expected in row['correctness'] for row in result['search']))
            self.assertEqual(state.requested, 0)
        self.assertEqual(captures, [0, 1, 2, 3, 4, 5, 6] * 2)
        self.assertEqual(launches, [('weight', 'immediate')] * 7 + [('weight', 'deferred')] * 7)
        self.assertEqual(checks.count(('deferred', 2)), 7)


if __name__ == '__main__':
    unittest.main()
