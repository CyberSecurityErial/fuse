"""CUDA-free checks for primitive timing boundaries and physical work counts."""
from contextlib import nullcontext, redirect_stderr
import copy
import ctypes as ct
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import calibrate_operators as calibration
from calibrate_operators import (COPY_PRIMITIVES, PRIMITIVES, QKV_PRIMITIVES, QKV_RESOURCE_KEYS,
                                QKV_FORWARD_PRIMITIVES, QKV_FORWARD_RESOURCE_KEYS, forward_resources,
                                allocate_forward_buffers, allocate_qkv_buffers,
                                arguments, candidate_configurations, geometry_work, prepare_primitive,
                                required_ready_bytes, sample_primitive)


class Argument(ct.Structure):
    _fields_ = [('geometry', ct.c_int32 * 13)]


def qkv_resources(**changes):
    fields = dict(zip(QKV_RESOURCE_KEYS, (3, 215040, 64, 116, 2, 256, 1, 4, 128, 32, 32, 2)))
    fields.update(changes)
    return fields


def qkv_state():
    events = []
    argument = Argument()
    argument.geometry[10] = 1

    def check(status):
        if status:
            raise RuntimeError('CPU mock native status failure')

    def tensor(name):
        return SimpleNamespace(copy_=lambda source: events.append((name, 'copy', source)),
                               fill_=lambda value: events.append((name, 'poison')),
                               data_ptr=lambda: 123456)

    state = SimpleNamespace(op=2, rank=1, world=4, m=256, device='cpu_mock_device', argument=argument,
                            case=dict(m=256, hidden=2048, q_heads=16, kv_heads=8, head_dim=128,
                                      cp=4, global_seq=1024),
                            stream=SimpleNamespace(cuda_stream=77, synchronize=lambda: events.append(('sync',))),
                            tensors={name: tensor(name) for name in ('workspace', 'out')},
                            scratch=tensor('scratch'), reference={'route': 'independent_route', 'out': 'independent_out'},
                            effective='decoded_weight', config={},
                            runtime=SimpleNamespace(check=check, copy=lambda *values: events.append(('ready_copy', *values))),
                            prepare=lambda phase, mode: lambda: events.append(('reset', phase, mode)),
                            flops=lambda phase: events.append(('flops', phase)) or 1234)
    state.arena = SimpleNamespace(ready_bytes=1024 * 1024, pointer=lambda region: 9000,
                                 write_data=lambda value: events.append(('arena_write', value)),
                                 read_data=lambda value: events.append(('arena_read', value)))
    state.scratch.view_as = lambda reference: events.append(('scratch_view', reference)) or state.scratch
    return state, events


class CalibrationTest(unittest.TestCase):
    def test_forward_cli_and_work_keep_forward_output_direction(self):
        args = arguments(['--cp', '4', '--operator', 'qkv_forward'])
        self.assertEqual(args.primitives, list(QKV_FORWARD_PRIMITIVES))
        self.assertEqual(list(QKV_FORWARD_PRIMITIVES.values()), list(range(5)))
        case = dict(m=32768, hidden=5120, n=7168, k=5120, q_heads=40, kv_heads=8,
                    head_dim=128, cp=4, global_seq=131072)
        work = geometry_work(case, 'qkv_forward')
        self.assertEqual((work['m'], work['n'], work['k']), (32768, 7168, 5120))
        self.assertEqual(work['remote_payload_bytes_per_gpu'], 2 * 32768 * 7168 * 3 // 4)
        with self.assertRaises(ValueError):
            geometry_work(dict(case, n=5120), 'qkv_forward')
        for extra in (['--iterations', '7'], ['--warmup', '2'], ['--primitives', 'copy']):
            with self.subTest(extra=extra), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                arguments(['--cp', '4', '--operator', 'qkv_forward', *extra])

    def test_forward_producer_prepares_identically_without_preloading_output(self):
        state, events = qkv_state()
        state.tensors['staging'] = SimpleNamespace(fill_=lambda value: events.append(('staging', 'poison')))
        expected = [('reset', 'forward', None), ('workspace', 'copy', 'decoded_weight')]
        for primitive in QKV_FORWARD_PRIMITIVES:
            if primitive in COPY_PRIMITIVES:
                continue
            for poison in (False, True):
                events.clear()
                prepare_primitive(state, primitive, poison=poison, operator='qkv_forward')()
                self.assertEqual(events, expected + ([('staging', 'poison')] if poison else []))

    def test_forward_copy_preloads_source_and_poison_only_destination(self):
        state, events = qkv_state()
        state.tensors['staging'] = SimpleNamespace(copy_=lambda value: events.append(('source_copy', value)))
        buffers = dict(copy_payload='deterministic_copy_payload')
        for poison in (False, True):
            events.clear()
            prepare_primitive(state, 'copy_fused_reservation', poison=poison,
                              operator='qkv_forward', buffers=buffers)()
            self.assertEqual(events, [('reset', 'forward', None)] +
                             ([('source_copy', 'deterministic_copy_payload'), ('scratch', 'poison'),
                               ('arena_write', state.scratch)] if poison else []))

    def test_forward_resource_query_checks_actual_tile_grid_and_reservation(self):
        state, _ = qkv_state()
        state.width = 4096
        config = dict(tile_m=128, tile_n=256, tile_k=64, cluster_m=2, sm_count=132, comm_ctas=4)
        changes = {}

        def query(argument, primitive, output):
            grid = 4 if primitive == 4 else 32
            values = [128, 256, 64, 2, 4, 214016, 168, grid, 2, 384, 32, 2, 16, 1, 1, 12]
            fields = dict(zip(QKV_FORWARD_RESOURCE_KEYS, values))
            fields.update(changes)
            output[:] = [fields[key] for key in QKV_FORWARD_RESOURCE_KEYS]
            return 0

        state.library = SimpleNamespace(fuse_mxfp8_test_qkv_forward_reference_config=query)
        for primitive in QKV_FORWARD_PRIMITIVES:
            resources = forward_resources(state, primitive, config)
            self.assertEqual(resources['copy_fields_apply'], primitive in COPY_PRIMITIVES)
            if primitive not in COPY_PRIMITIVES:
                self.assertEqual(resources['primitive_grid_x'], 32)
                self.assertIn(resources['compute_sm_budget'], (128, 132))
        for change in (dict(tile_n=128), dict(primitive_grid_x=130), dict(primitive_launch_cluster_m=1),
                       dict(ready_n_tiles=32), dict(primitive_registers_per_thread=0),
                       dict(copy_use_tma_store=0), dict(copy_slots=8)):
            changes = change
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                forward_resources(state, 'copy_fused_reservation', config)

    def test_forward_buffers_construct_exact_qkv_route_and_strided_epochs_once(self):
        import numpy as np

        class Tensor:
            def __init__(self, array):
                self.array = array
            def __getitem__(self, index):
                return Tensor(self.array[index])
            def __setitem__(self, index, value):
                self.array[index] = value
            def copy_(self, value):
                self.array[:] = value.array
            def clone(self):
                return Tensor(self.array.copy())
            def reshape(self, *shape):
                return Tensor(self.array.reshape(*shape))

        state, _ = qkv_state()
        state.m, state.width, state.q, state.kv = 2, 16, 8, 4
        generated = []

        def values(shape, seed):
            generated.append(seed)
            return Tensor(np.arange(np.prod(shape)).reshape(shape) + seed * 100)

        state.values = values
        state.reference['out'] = Tensor(np.empty(32))
        fake_torch = SimpleNamespace(int32=np.int32, cuda=SimpleNamespace(stream=lambda _: nullcontext()),
                                     zeros=lambda size, **kw: Tensor(np.zeros(size, dtype=kw['dtype'])),
                                     empty_like=lambda value: Tensor(np.empty_like(value.array)))
        resources = dict(ready_m_tiles=1, ready_n_tiles=2, ready_flag_stride=32)
        buffers = allocate_forward_buffers(state, [(4, {}, {'copy_fused_reservation': resources})], fake_torch)
        expected = []
        for offset, width in ((0, 2), (8, 1), (12, 1)):
            for peer in range(4):
                source = np.arange(32).reshape(2, 16) + (701 + peer) * 100
                expected.extend(source[:, offset + width:offset + 2 * width].reshape(-1))
        np.testing.assert_array_equal(buffers['copy_expected'].array, expected)
        np.testing.assert_array_equal(np.flatnonzero(buffers['ready_signaled'].array), [0, 32])
        self.assertEqual(sorted(generated), [701, 702, 703, 704])
        self.assertEqual(buffers['ready_bytes'], 256)

    def test_forward_sampling_checks_poison_and_final_result_outside_timing(self):
        state, events = qkv_state()
        state.tensors['staging'] = SimpleNamespace(copy_=lambda value: events.append(('source_copy', value)),
                                                   fill_=lambda value: events.append(('staging_poison',)))
        state.library = SimpleNamespace(fuse_mxfp8_test_qkv_forward_reference=lambda arg, primitive:
                                        events.append(('launch', primitive)) or 0)

        class Invocation:
            def __init__(self, call, launch, warmup, prepare, stream):
                self.call, self.prepare = call, prepare
            def once(self, prepare):
                prepare()
                self.call()
                events.append(('local_complete',))
            def measure(self, iterations, flops):
                events.append(('measure', flops))
                for _ in range(iterations):
                    self.prepare()
                    self.call()
                events.append(('all_rank_measure_complete',))
                return dict(p50_us=1, flops_per_gpu=flops)

        runtime = SimpleNamespace(Invocation=Invocation, aligned_prepare=lambda prepare, stream: prepare())
        torch = SimpleNamespace(cuda=SimpleNamespace(stream=lambda _: nullcontext()))
        dist = SimpleNamespace(barrier=lambda: events.append(('barrier',)))
        buffers = dict(copy_payload='payload')
        args = SimpleNamespace(operator='qkv_forward', launch='graph', warmup=10, iterations=2)
        with patch.object(calibration.bench, 'torch', torch, create=True), \
                patch.object(calibration.bench, 'dist', dist, create=True), \
                patch.object(calibration, 'check_forward_primitive',
                             side_effect=lambda *args: events.append(('verify',)) or {'passed': True}):
            for primitive, primitive_id in QKV_FORWARD_PRIMITIVES.items():
                events.clear()
                result = sample_primitive(state, primitive, args, runtime, {}, buffers)
                self.assertEqual(events.count(('launch', primitive_id)), 3)
                self.assertEqual(events.count(('verify',)), 2)
                first_verify = events.index(('verify',))
                self.assertLess(first_verify, events.index(('all_rank_measure_complete',)))
                self.assertEqual(events[-1], ('verify',))
                self.assertEqual(result['correctness_after_poison'], {'passed': True})
                self.assertEqual(result['scope'], 'no_DQ_no_other_role_no_finalize_not_F_total')
                if primitive in COPY_PRIMITIVES:
                    self.assertEqual(events[first_verify - 1], ('barrier',))
                    self.assertIn(('measure', 0), events)
                    self.assertEqual(events.count(('source_copy', 'payload')), 2)
                else:
                    self.assertIn(('measure', 1234), events)
                    self.assertIn(('flops', 'forward'), events)

    def test_geometry_counts_only_remote_wire_payload(self):
        case = dict(m=16384, hidden=2048, q_heads=16, head_dim=128, cp=8, global_seq=131072)
        work = geometry_work(case)
        self.assertEqual(work['flops_per_gpu'], 2 * 16384 * 2048 * 2048)
        self.assertEqual(work['remote_payload_bytes_per_gpu'], 2 * 16384 * 2048 * 7 // 8)
        self.assertEqual(work['staging_write_bytes_per_gpu'], 2 * 16384 * 2048)
        self.assertEqual(work['compulsory_gemm_bytes_per_gpu'], 2 * (2 * 16384 * 2048 + 2048 ** 2))

    def test_flattened_geometry_required(self):
        case = dict(m=16384, hidden=2048, q_heads=16, head_dim=128, cp=8, global_seq=131072)
        for change in (dict(cp=2), dict(m=0), dict(hidden=-1), dict(global_seq=65536)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                geometry_work(dict(case, **change))

    def test_default_protocol_and_primitive_set(self):
        args = arguments(['--cp', '8'])
        self.assertEqual((args.warmup, args.iterations, args.launch), (10, 50, 'graph'))
        self.assertEqual(args.comm_ctas, [4, 8, 16, 24, 32])
        self.assertEqual(PRIMITIVES, {'compute_subgrid': 0, 'copy': 1, 'compute_fullgrid': 2})
        self.assertEqual(args.operator, 'oproj_forward')
        self.assertEqual(args.primitives, list(PRIMITIVES))

    def test_reject_zero_sample_count(self):
        with self.assertRaises(SystemExit):
            arguments(['--cp', '4', '--iterations', '0'])

    def test_compute_preload_uses_independent_reference(self):
        calls = []
        tensors = {key: SimpleNamespace(copy_=lambda value, key=key: calls.append((key, 'copy', value)),
                                       fill_=lambda value, key=key: calls.append((key, 'fill')))
                   for key in ('workspace', 'staging', 'out')}
        state = SimpleNamespace(tensors=tensors, effective='decoded_reference',
                                reference={'route': 'independent_routed_input'},
                                prepare=lambda phase, mode: lambda: calls.append(('reset', phase, mode)))
        prepare_primitive(state, 'compute_subgrid', poison=True)()
        self.assertEqual(calls, [('reset', 'forward', None),
                                 ('workspace', 'copy', 'decoded_reference'),
                                 ('staging', 'copy', 'independent_routed_input'), ('out', 'fill')])
        calls.clear()
        prepare_primitive(state, 'compute_subgrid')()
        self.assertEqual(calls, [('reset', 'forward', None)])

    def test_copy_does_not_preload_routed_staging(self):
        calls = []
        state = SimpleNamespace(effective='decoded_reference', tensors={
            'workspace': SimpleNamespace(copy_=lambda value: calls.append(('workspace', value))),
            'staging': SimpleNamespace(fill_=lambda value: calls.append(('staging', 'poison')))},
            prepare=lambda phase, mode: lambda: None)
        prepare_primitive(state, 'copy', poison=True)()
        self.assertEqual(calls, [('workspace', 'decoded_reference'), ('staging', 'poison')])

    def test_qkv_cli_ids_and_subset_do_not_change_oproj_defaults(self):
        args = arguments(['--cp', '8', '--operator', 'qkv_backward'])
        self.assertEqual(QKV_PRIMITIVES, {'compute_bare_subgrid': 0, 'copy': 1,
                                         'compute_ready_preloaded_subgrid': 2,
                                         'compute_bare_fullgrid': 3, 'compute_ready_preloaded_fullgrid': 4,
                                         'copy_fused_reservation': 5})
        self.assertEqual(args.primitives, list(QKV_PRIMITIVES))
        subset = arguments(['--cp', '8', '--operator', 'qkv_backward', '--primitives', 'copy,compute_bare_fullgrid'])
        self.assertEqual(subset.primitives, ['copy', 'compute_bare_fullgrid'])
        for extra in (['--primitives', 'compute_subgrid'], ['--primitives', 'copy,copy'],
                      ['--primitives', ''], ['--iterations', '7'], ['--warmup', '2']):
            with self.subTest(extra=extra), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                arguments(['--cp', '4', '--operator', 'qkv_backward', *extra])

    def test_qkv_work_uses_all_packed_q_k_v_heads(self):
        state, _ = qkv_state()
        work = geometry_work(state.case, 'qkv_backward')
        self.assertEqual((work['m'], work['n'], work['k']), (256, 2048, 4096))
        self.assertEqual(work['flops_per_gpu'], 2 * 256 * 2048 * 4096)
        self.assertEqual(work['remote_payload_bytes_per_gpu'], 2 * 256 * 4096 * 3 // 4)
        self.assertIn('lower bounds', work['byte_definition'])
        with self.assertRaises(ValueError):
            geometry_work(dict(state.case, b_mnk=[256, 4096, 2048]), 'qkv_backward')

    def test_ready_allocation_is_one_actual_prefix_and_preserves_uint32_epoch_bits(self):
        state, events = qkv_state()
        state.argument.geometry[10] = -1
        configurations = [(4, {}, {'compute_bare_subgrid': qkv_resources()}),
                          (16, {}, {'compute_ready_preloaded_subgrid': qkv_resources(ready_block_m=64)})]
        allocations = []
        fake_tensor = SimpleNamespace(data_ptr=lambda: 456)
        torch = SimpleNamespace(int32='int32', cuda=SimpleNamespace(stream=lambda _: nullcontext()),
                                full=lambda *args, **kwargs: allocations.append((args, kwargs)) or fake_tensor)
        buffers = allocate_qkv_buffers(state, configurations, torch)
        self.assertEqual(buffers['bytes'], 4 * 32 * 32 * 4)
        self.assertEqual(buffers['epoch'], 0xffffffff)
        self.assertEqual(allocations, [(((4096,), -1), {'dtype': 'int32', 'device': state.device})])
        self.assertLess(buffers['bytes'], state.arena.ready_bytes)
        self.assertIsNone(allocate_qkv_buffers(state, [(4, {}, {'copy': qkv_resources()})], torch))
        self.assertIsNone(allocate_qkv_buffers(state, [(4, {}, {'copy_fused_reservation': qkv_resources()})], torch))
        self.assertEqual(len(allocations), 1)
        state.arena.ready_bytes = 64
        with self.assertRaises(ValueError):
            required_ready_bytes(state, qkv_resources())

    def test_bare_and_ready_prepare_the_same_bound_arena_and_integer_flags_every_time(self):
        state, events = qkv_state()
        resources = qkv_resources()
        buffers = dict(ready=SimpleNamespace(data_ptr=lambda: 456), bytes=16384, epoch=1)
        expected = [('reset', 'data', 'deferred'), ('workspace', 'copy', 'decoded_weight'),
                    ('arena_write', 'independent_route'), ('ready_copy', 9000, 456, 8192, 77)]
        for primitive in (name for name in QKV_PRIMITIVES if name not in COPY_PRIMITIVES):
            prepare = prepare_primitive(state, primitive, operator='qkv_backward', resources=resources, buffers=buffers)
            for _ in range(2):
                events.clear()
                prepare()
                self.assertEqual(events, expected)
        events.clear()
        prepare_primitive(state, 'compute_bare_subgrid', poison=True, operator='qkv_backward',
                          resources=resources, buffers=buffers)()
        self.assertEqual(events, expected + [('out', 'poison')])
        with self.assertRaises(ValueError):
            prepare_primitive(state, 'compute_bare_subgrid', operator='qkv_backward',
                              resources=resources, buffers=dict(buffers, epoch=2))

    def test_qkv_copy_poison_writes_arena_not_unbound_staging_or_ready_prefill(self):
        state, events = qkv_state()
        for primitive in COPY_PRIMITIVES:
            events.clear()
            prepare_primitive(state, primitive, poison=True, operator='qkv_backward')()
            self.assertEqual(events, [('reset', 'data', 'deferred'), ('scratch', 'poison'),
                                      ('arena_write', state.scratch)])
            events.clear()
            prepare_primitive(state, primitive, operator='qkv_backward')()
            self.assertEqual(events, [('reset', 'data', 'deferred')])

    def test_qkv_native_metadata_preserves_actual_copy_cluster_and_rejects_query_errors(self):
        state, events = qkv_state()
        state.requested = 0

        def setter(requested):
            state.requested = requested
            return 0

        def refresh(requested_comm_ctas):
            self.assertEqual(state.requested, requested_comm_ctas)
            state.config = dict(comm_ctas=requested_comm_ctas or 4, requested_comm_ctas=requested_comm_ctas,
                                tile_m=128, tile_n=256, tile_k=64, cluster_m=2, sm_count=132)

        def query(argument, primitive, output):
            fields = qkv_resources(primitive_launch_cluster_m=1 if primitive == 1 else 2,
                                   primitive_grid_x=state.config['comm_ctas'] if primitive in (1, 5) else 116)
            output[:] = [fields[key] for key in QKV_RESOURCE_KEYS]
            return 0

        state.refresh_config = refresh
        state.library = SimpleNamespace(fuse_mxfp8_test_set_comm_ctas=setter,
                                        fuse_mxfp8_test_qkv_reference_config=query)
        dist = SimpleNamespace(all_gather_object=lambda output, value:
                               output.__setitem__(slice(None), [copy.deepcopy(value)] * state.world))
        args = SimpleNamespace(comm_ctas=[4, 16], operator='qkv_backward', primitives=list(QKV_PRIMITIVES))
        configurations = candidate_configurations(state, args, dist)
        self.assertEqual(len(configurations), 2)
        for requested, selected, primitives in configurations:
            self.assertEqual(selected['cluster_m'], 2)
            self.assertEqual(primitives['copy']['primitive_launch_cluster_m'], 1)
            self.assertEqual(primitives['copy']['primitive_grid_x'], requested)
            self.assertTrue(primitives['copy']['copy_fields_apply'])
            self.assertTrue(primitives['copy_fused_reservation']['copy_fields_apply'])
            self.assertEqual(primitives['copy_fused_reservation']['primitive_launch_cluster_m'], 2)
            self.assertEqual(primitives['copy_fused_reservation']['primitive_grid_x'], requested)
            self.assertEqual(primitives['compute_bare_subgrid']['selected_gemm_stages'], 3)
            self.assertFalse(primitives['compute_bare_subgrid']['copy_fields_apply'])
        for primitive, changes in (('copy', dict(primitive_dynamic_smem_bytes=0)),
                                   ('compute_bare_subgrid', dict(primitive_dynamic_smem_bytes=0)),
                                   ('copy', dict(copy_use_tma=0)), ('copy', dict(copy_slots=0)),
                                   ('copy_fused_reservation', dict(primitive_launch_cluster_m=1)),
                                   ('copy_fused_reservation', dict(copy_use_tma=0))):
            def invalid_query(argument, primitive_id, output):
                status = query(argument, primitive_id, output)
                if primitive_id == QKV_PRIMITIVES[primitive]:
                    for key, value in changes.items():
                        output[QKV_RESOURCE_KEYS.index(key)] = value
                return status

            state.library.fuse_mxfp8_test_qkv_reference_config = invalid_query
            with self.subTest(primitive=primitive, changes=changes), self.assertRaises(RuntimeError):
                candidate_configurations(state, args, dist)
        state.library.fuse_mxfp8_test_qkv_reference_config = lambda *args: 1
        with self.assertRaises(RuntimeError):
            candidate_configurations(state, args, dist)

    def test_copy_result_is_read_only_after_rank_timing_collective(self):
        state, events = qkv_state()
        state.library = SimpleNamespace(fuse_mxfp8_test_qkv_reference=lambda arg, primitive:
                                        events.append(('launch_primitive', primitive)) or 0)

        class Invocation:
            def __init__(self, call, launch, warmup, prepare, stream):
                self.call, self.prepare = call, prepare
                events.append(('capture', launch))

            def once(self, prepare):
                prepare()
                self.call()

            def measure(self, iterations, flops):
                for _ in range(iterations):
                    self.prepare()
                    self.call()
                events.append(('timing_all_rank_collective_complete',))
                return dict(p50_us=1, flops_per_gpu=flops)

        def error(actual, expected, exact=False):
            events.append(('verify', actual, expected, exact))
            return {'all_ranks_finite': True, 'max_abs': 0, 'relative_rmse': 0}

        runtime = SimpleNamespace(Invocation=Invocation, aligned_prepare=lambda prepare, stream: prepare(),
                                  check_error=error)
        torch = SimpleNamespace(cuda=SimpleNamespace(stream=lambda _: nullcontext()))
        dist = SimpleNamespace(barrier=lambda: events.append(('all_rank_barrier',)))
        args = SimpleNamespace(operator='qkv_backward', launch='graph', warmup=10, iterations=2)
        with patch.object(calibration.bench, 'torch', torch, create=True), \
                patch.object(calibration.bench, 'dist', dist, create=True):
            for primitive in COPY_PRIMITIVES:
                events.clear()
                cluster = 1 if primitive == 'copy' else 2
                resources = qkv_resources(primitive_launch_cluster_m=cluster)
                result = sample_primitive(state, primitive, args, runtime, resources)
                self.assertLess(events.index(('timing_all_rank_collective_complete',)),
                                events.index(('arena_read', state.scratch)))
                self.assertIn(('scratch_view', 'independent_route'), events)
                self.assertEqual(events[-1], ('verify', state.scratch, 'independent_route', True))
                self.assertEqual(events.count(('all_rank_barrier',)), 2)
                self.assertEqual(result['primitive_id'], QKV_PRIMITIVES[primitive])
                self.assertEqual(result['resources']['primitive_launch_cluster_m'], cluster)
                self.assertEqual(result['scope'], 'no_DQ_no_W_no_finalize_not_B_total')
                self.assertFalse(any(event[0] == 'ready_copy' for event in events))

    def test_all_compute_ids_time_only_the_reference_call_and_propagate_launch_errors(self):
        state, events = qkv_state()
        launches = []
        state.library = SimpleNamespace(fuse_mxfp8_test_qkv_reference=lambda argument, primitive:
                                        launches.append(primitive) or 0)

        class Invocation:
            def __init__(self, call, launch, warmup, prepare, stream):
                self.call, self.prepare = call, prepare

            def once(self, prepare):
                prepare()
                self.call()

            def measure(self, iterations, flops):
                for _ in range(iterations):
                    self.prepare()
                    self.call()
                return dict(p50_us=2, flops_per_gpu=flops)

        verified = []
        runtime = SimpleNamespace(Invocation=Invocation, aligned_prepare=lambda prepare, stream: prepare(),
                                  check_error=lambda actual, expected: verified.append((actual, expected)) or
                                  {'all_ranks_finite': True, 'max_abs': 0.1, 'relative_rmse': .001})
        torch = SimpleNamespace(cuda=SimpleNamespace(stream=lambda _: nullcontext()))
        args = SimpleNamespace(operator='qkv_backward', launch='graph', warmup=10, iterations=2)
        buffers = dict(ready=SimpleNamespace(data_ptr=lambda: 456), bytes=8192, epoch=1)
        with patch.object(calibration.bench, 'torch', torch, create=True):
            for primitive, primitive_id in QKV_PRIMITIVES.items():
                if primitive in COPY_PRIMITIVES:
                    continue
                events.clear()
                launches.clear()
                record = sample_primitive(state, primitive, args, runtime, qkv_resources(), buffers)
                self.assertEqual(launches, [primitive_id] * 3)
                self.assertEqual(record['timing']['flops_per_gpu'], 1234)
                self.assertEqual(record['ready_prepublication_bytes'], 8192)
                self.assertEqual(record['ready_epoch'], 1)
                self.assertEqual(events.count(('ready_copy', 9000, 456, 8192, 77)), 4)
                self.assertIn(('flops', 'data'), events)
                self.assertFalse(any(event[0] == 'arena_read' for event in events))
            self.assertEqual(verified, [(state.tensors['out'], 'independent_out')] * 4)
            state.library.fuse_mxfp8_test_qkv_reference = lambda *args: 1
            with self.assertRaises(RuntimeError):
                sample_primitive(state, 'compute_bare_subgrid', args, runtime, qkv_resources(), buffers)


if __name__ == '__main__':
    unittest.main()
