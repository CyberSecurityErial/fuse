"""CUDA-free checks for bounded, compact profiling calibration artifacts."""

import argparse
from contextlib import nullcontext, redirect_stderr
import copy
import ctypes as ct
import json
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import operator_bench as bench
import profile_operators as profiler
from profile_operators import (arguments, calibration_record, comm_candidates,
                               profile_prefix, validate_sample_limit)
from profile_report import rank_profile
from test_profile_report import fixture, shift_rank


class ProfileOperatorsTest(unittest.TestCase):
    def test_small_qkv_grid_is_derived_from_geometry_not_observations(self):
        config = dict(forward_or_data_mnk=[256, 4096, 4096], cluster_m=1,
                      sm_count=132, comm_ctas=10, tile_m=128, tile_n=128,
                      raster='n', swizzle=1)
        self.assertEqual(profiler.qkv_backward_grid_ctas(config), 74)
        self.assertEqual(profiler.qkv_backward_grid_ctas(
            dict(config, forward_or_data_mnk=[32768, 4096, 4096])), 132)
        self.assertEqual(profiler.qkv_backward_grid_ctas(
            dict(config, forward_or_data_mnk=[128, 512, 4096], cluster_m=2)), 18)
        for changes in (dict(cluster_m=4), dict(raster='m'), dict(swizzle=2),
                        dict(comm_ctas=132), dict(cluster_m=2, comm_ctas=11)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                profiler.qkv_backward_grid_ctas(dict(config, **changes))

    def test_qkv_padding_must_be_zero_and_missing_launched_cta_still_fails(self):
        local = fixture()[0]
        local['config'].update(forward_or_data_mnk=[128, 128, 4096], cluster_m=1,
                               tile_m=128, tile_n=128, raster='n', swizzle=1)
        empty = {key: [0] * 8 if key == 'source_ready' else 0
                 for key in local['ctas'][-1]}
        with self.assertRaises(ValueError):
            profiler.qkv_backward_timeline(local['config'], local['ctas'])
        local['ctas'][-1] = empty
        local['config'], local['ctas'] = profiler.qkv_backward_timeline(
            local['config'], local['ctas'])
        self.assertEqual(local['config']['grid_ctas'], 3)
        self.assertEqual(rank_profile('qkv_backward', local)[0]['observed_ctas'], 3)
        local['ctas'][2] = copy.deepcopy(empty)
        with self.assertRaises(ValueError):
            rank_profile('qkv_backward', local)

    def test_single_cta_cli_remains_compatible(self):
        self.assertEqual(arguments(['--cp', '4']).comm_ctas, [0])
        args = arguments(['--cp', '4', '--comm-ctas', '16'])
        self.assertEqual(args.comm_ctas, [16])
        self.assertFalse(args.aggregate_only)
        self.assertEqual(args.max_cases, 4)
        self.assertEqual(args.warmup, 2)
        self.assertEqual(arguments(['--cp', '4', '--warmup', '10']).warmup, 10)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            arguments(['--cp', '4', '--warmup', '1'])

    def test_multiple_ctas_are_unique_nonnegative_and_even(self):
        self.assertEqual(comm_candidates('0,4,8,16'), [0, 4, 8, 16])
        self.assertEqual(comm_candidates('16,0'), [16, 0])
        for value in ('', '0,', '4,4', '-2', '3', 'nan'):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                comm_candidates(value)

    def test_limit_counts_shapes_times_candidates(self):
        self.assertEqual(validate_sample_limit([{}, {}], [0, 16], 4), 4)
        for shapes, candidates, limit in (([{}, {}], [0, 4, 16], 4), ([], [0], 4)):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                validate_sample_limit(shapes, candidates, limit)

    def test_single_prefix_is_unchanged_multiple_prefixes_cannot_collide(self):
        root, case = Path('/unused'), 'oproj_forward/model/s131072/cp8'
        self.assertEqual(profile_prefix(root, case, 16, False),
                         root / 'oproj_forward_model_s131072_cp8')
        values = [profile_prefix(root, case, candidate, True) for candidate in (0, 4, 8, 16)]
        self.assertEqual(len(set(values)), 4)
        self.assertEqual(values[-1].name, 'oproj_forward_model_s131072_cp8_c16')

    def summaries(self):
        return [rank_profile('qkv_backward', row)[0] for row in fixture()]

    def test_aggregate_keeps_same_rank_semantics_without_raw_trace(self):
        ranks = self.summaries()
        result = calibration_record({'id': 'cpu-fixture', 'cp': 2}, 'qkv_backward', 0,
                                    list(reversed(ranks)), {'pass': True})
        self.assertEqual([row['rank'] for row in result['ranks']], [0, 1])
        self.assertEqual(result['ranks'][0]['route_role_us'], 2)
        self.assertEqual(result['ranks'][0]['wgrad_marker_us'], 2)
        self.assertEqual(result['ranks'][0]['config']['comm_ctas'], 2)
        self.assertEqual(result['requested_comm_ctas'], 0)
        encoded = json.dumps(result)
        self.assertNotIn('traceEvents', encoded)
        self.assertNotIn('"ctas"', encoded)

    def test_missing_rank_wrong_request_or_raw_cta_is_rejected(self):
        ranks = self.summaries()
        for changed in (ranks[:1], [ranks[0], ranks[0]],
                        [dict(ranks[0], ctas=[]), ranks[1]]):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                calibration_record({'cp': 2}, 'qkv_backward', 0, changed, {})
        with self.assertRaises(ValueError):
            calibration_record({'cp': 2}, 'qkv_backward', 16, ranks, {})
        wrong = copy.deepcopy(ranks)
        for row in wrong:
            row['config']['requested_comm_ctas'] = 16
        with self.assertRaises(ValueError):
            calibration_record({'cp': 2}, 'qkv_backward', 16, wrong, {})

    def test_calibration_durations_do_not_subtract_cross_rank_clocks(self):
        original = self.summaries()
        shifted = fixture()
        shift_rank(shifted[0], 10**15)
        shift_rank(shifted[1], 10**18)
        updated = [rank_profile('qkv_backward', row)[0] for row in shifted]
        for before, after in zip(original, updated):
            before.pop('origin_ns')
            after.pop('origin_ns')
            self.assertEqual(before, after)

    def test_checkpoint_retains_only_one_summary_file(self):
        sample = calibration_record({'cp': 2}, 'qkv_backward', 0, self.summaries(), {})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'calibration_summary.json'
            report = dict(samples=[], complete=False)
            bench.checkpoint(path, report)
            report['samples'].append(sample)
            bench.checkpoint(path, report)
            report['complete'] = True
            bench.checkpoint(path, report)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            self.assertTrue(json.loads(path.read_text())['complete'])

    def test_candidate_switch_refreshes_before_capture_and_reuses_buffers(self):
        events = []
        source = fixture('oproj_backward')[0]
        state = SimpleNamespace(op=3, rank=0, name='oproj_backward', stream=None,
                                config=dict(source['config']), argument=SimpleNamespace(beta=0))

        class Argument(ct.Structure):
            _fields_ = [('beta', ct.c_float)]

        class Buffer:
            def __init__(self, elements, values=None):
                self.elements, self.values = elements, values

            def numel(self):
                return self.elements

            def zero_(self):
                events.append(('zero', id(self)))

            def data_ptr(self):
                return id(self)

            def cpu(self):
                return self

            def tolist(self):
                return self.values

        def setter(requested):
            events.append(('setter', requested))
            return 0

        def refresh(requested_comm_ctas):
            events.append(('refresh', requested_comm_ctas))
            state.config['requested_comm_ctas'] = requested_comm_ctas

        def gather(output, local):
            self.assertNotIn('ctas', local)
            output[:] = [dict(copy.deepcopy(local), rank=rank) for rank in range(2)]

        class Invocation:
            def __init__(self, call, launch, warmup, prepare, stream):
                events.append(('capture', state.config['requested_comm_ctas']))
                self.call, self.prepare = call, prepare

            def once(self):
                self.prepare()
                self.call()

        state.argument = Argument()
        state.library = SimpleNamespace(fuse_mxfp8_test_set_comm_ctas=setter,
                                        fuse_mxfp8_test_profile=lambda *unused: 0)
        state.runtime = SimpleNamespace(check=lambda status: self.assertEqual(status, 0))
        state.refresh_config = refresh
        state.prepare = lambda *args, **kwargs: lambda: None
        state.verify = lambda *args: {'pass': True}
        timeline = Buffer(4 * ct.sizeof(profiler.CtaTimeline))
        markers = Buffer(4, [1000, 1500, 5000, 7000])
        args = SimpleNamespace(cp=2, weight_mode='deferred', launch='graph', aggregate_only=True, warmup=10)
        with patch.object(profiler, 'torch', SimpleNamespace(cuda=SimpleNamespace(stream=lambda _: nullcontext())),
                          create=True), \
                patch.object(profiler, 'runtime', SimpleNamespace(Invocation=Invocation), create=True), \
                patch.object(profiler, 'dist', SimpleNamespace(all_gather_object=gather), create=True), \
                patch.object(profiler, 'decode_timeline', return_value=source['ctas']):
            for requested in (0, 2):
                ranks, correctness = profiler.sample_profile(state, args, requested, timeline, markers)
                self.assertEqual(ranks[0]['config']['requested_comm_ctas'], requested)
                self.assertEqual(correctness, {'pass': True})
        changes = [event for event in events if event[0] != 'zero']
        self.assertEqual(changes, [('setter', 0), ('refresh', 0), ('capture', 0),
                                   ('setter', 2), ('refresh', 2), ('capture', 2)])
        self.assertEqual([event[1] for event in events if event[0] == 'zero'],
                         [id(timeline), id(markers)] * (2 * (args.warmup - 2 + 1)))


if __name__ == '__main__':
    unittest.main()
