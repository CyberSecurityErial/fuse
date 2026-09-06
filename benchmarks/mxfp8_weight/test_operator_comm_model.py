"""CUDA-free contracts for explicitly installing frozen forward service models."""

from contextlib import redirect_stderr
import copy
import ctypes as ct
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import operator_bench as bench
import operator_sweep as sweep
import profile_operators as profiler


def fixture():
    return dict(schema='mxfp8-oproj-service-model-v1', complete=True, frozen=True,
                models=[dict(world_size=4, sm_count=132,
                             coefficients=dict(a=1000.0, e=0.0, b=0.1, t=0.02),
                             launch_prior_us=5.0, minimum_gain=0.10,
                             domain=dict(tile_families=[[128, 256, 64, 2], [128, 320, 64, 2]]))])


class Call:
    def __init__(self, function):
        self.function, self.calls = function, []

    def __call__(self, *args):
        self.calls.append(args)
        return self.function(*args)


class Library:
    """Only a fake C ABI; none of these numbers are native measurements."""
    def __init__(self):
        self.native = [0.0] * 8
        self.selected = [12, 128, 256, 64, 2, 132, 0, 0, 0, 0]
        self.fuse_mxfp8_test_set_oproj_comm_model = Call(self.install)
        self.fuse_mxfp8_test_get_oproj_comm_model = Call(lambda output: self.copy(output, self.native))
        self.fuse_mxfp8_test_config = Call(lambda argument, output: self.copy(output, self.selected))
        self.fuse_mxfp8_test_wgrad_config = Call(
            lambda argument, output: self.copy(output, [1, 128, 256, 64, 2, 3, 215040, 64]))

    @staticmethod
    def copy(output, values):
        output[:] = values
        return 0

    def install(self, cp, sm, values):
        self.native = [cp, sm, *values]
        return 0


def check(status):
    if status:
        raise RuntimeError(f'Fake CUDA status: {status}')


class OperatorCommModelTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'model.json'
        self.runtime = SimpleNamespace(check=check)

    def write(self, document=None):
        self.path.write_text(json.dumps(fixture() if document is None else document))
        return self.path

    def install(self, library, workers=None):
        self.write()
        model = bench.load_oproj_comm_model(self.path, 4, 132)
        return bench.configure_oproj_comm_model(self.path, 4, 132, library, self.runtime,
                                                {model['source']: model['sha256']}, workers)

    def test_cli_is_opt_in_and_sweeps_preserve_the_old_auto(self):
        self.assertIsNone(bench.arguments([]).oproj_comm_model)
        self.assertIsNone(profiler.arguments(['--cp', '4']).oproj_comm_model)
        self.assertIsNone(sweep.arguments(['--cp', '4']).oproj_comm_model)
        for arguments, prefix in ((bench.arguments, []), (profiler.arguments, ['--cp', '4'])):
            args = arguments(prefix + ['--oproj-comm-model', str(self.path)])
            self.assertEqual(args.oproj_comm_model, self.path)
        for kind in ('comm', 'wgrad'):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                sweep.arguments(['--cp', '4', '--sweep-kind', kind,
                                 '--oproj-comm-model', str(self.path)])

    def test_default_requires_no_model_file_or_new_library_api(self):
        self.assertIsNone(bench.configure_oproj_comm_model(None, 4, 132, object(), object(), {}))
        self.assertIsNone(bench.read_oproj_comm_model(SimpleNamespace(), object()))

    def test_exact_cp_selection_is_not_registry_model_selection(self):
        document = fixture()
        other = copy.deepcopy(document['models'][0])
        other.update(world_size=8, model='unused_registry_name')
        other['coefficients']['a'] = 2000.0
        document['models'].insert(0, other)
        self.write(document)
        selected = bench.load_oproj_comm_model(self.path, 4, 132)
        self.assertEqual(selected['coefficients']['a'], 1000.0)
        self.assertEqual(bench.load_oproj_comm_model(self.path, 8, 132)['coefficients']['a'], 2000.0)
        self.assertEqual(selected['sha256'], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertNotIn('models', selected)
        self.assertEqual(selected['scope'], 'oproj_forward_only_manual_comm_ctas_take_precedence')

    def test_missing_or_ambiguous_cp_is_rejected_even_for_different_card_groups(self):
        for models in ([], [dict(fixture()['models'][0], world_size=8)],
                       [fixture()['models'][0], dict(fixture()['models'][0], cuda_visible_devices='other')]):
            with self.subTest(models=models), self.assertRaises(ValueError):
                bench.load_oproj_comm_model(self.write(dict(fixture(), models=models)), 4, 132)

    def test_complete_frozen_schema_is_required(self):
        for changes in (dict(schema='unknown'), dict(complete=False), dict(complete=1),
                        dict(frozen=False), dict(frozen=None), dict(models={}), dict(models=[None])):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                bench.load_oproj_comm_model(self.write(dict(fixture(), **changes)), 4, 132)

    def test_cp_sm_and_both_supported_tile_families_are_required(self):
        self.write()
        for cp, sm in ((2, 132), (4, 120)):
            with self.subTest(cp=cp, sm=sm), self.assertRaises(ValueError):
                bench.load_oproj_comm_model(self.path, cp, sm)
        for changes in (dict(sm_count=120), dict(sm_count=132.0), dict(world_size=4.0),
                        dict(domain={}), dict(domain=dict(tile_families=[[128, 256, 64, 2]])),
                        dict(domain=dict(tile_families=[[128, 128, 64, 2], [128, 320, 64, 2]]))):
            document = fixture()
            document['models'][0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                bench.load_oproj_comm_model(self.write(document), 4, 132)

    def test_coefficients_are_finite_and_obey_the_native_contract(self):
        for field in ('a', 'e', 'b', 't', 'launch_prior_us', 'minimum_gain'):
            bad_values = [None, True, '1', -1, float('nan'), float('inf'), 10**400]
            if field in ('a', 'b', 't'):
                bad_values.append(0)
            if field == 'minimum_gain':
                bad_values.append(1)
            for value in bad_values:
                document = fixture()
                model = document['models'][0]
                (model['coefficients'] if field in ('a', 'e', 'b', 't') else model)[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    bench.load_oproj_comm_model(self.write(document), 4, 132)

    def test_model_sha_is_part_of_shared_run_source_hashes(self):
        raw = b'explicit model fixture'
        path = self.path
        with patch.object(Path, 'read_bytes', lambda current: raw if current == path else b'source'), \
                patch.object(bench, 'forward_matrix', return_value=[]), \
                patch.object(bench, 'backward_matrix', return_value=[]):
            args = SimpleNamespace(library=Path('/not-opened/library.so'), backends='fuse',
                                   oproj_comm_model=path)
            sources = bench.source_hashes(args)
        self.assertEqual(sources[str(path)], hashlib.sha256(raw).hexdigest())

    def test_install_once_uses_exact_double_abi_and_checks_worker_agreement(self):
        library = Library()
        workers = SimpleNamespace(all_gather_object=lambda output, value:
                                  output.__setitem__(slice(None), [value] * 4))
        model = self.install(library, workers)
        self.assertEqual(library.native, [4, 132, 1000.0, 0.0, 0.1, 0.02, 5.0, 0.10])
        self.assertEqual(len(library.fuse_mxfp8_test_set_oproj_comm_model.calls), 1)
        self.assertEqual(bench.read_oproj_comm_model(library, self.runtime), model)
        self.assertEqual(len(library.fuse_mxfp8_test_set_oproj_comm_model.calls), 1)
        library.native[2] += 1
        with self.assertRaises(RuntimeError):
            bench.read_oproj_comm_model(library, self.runtime)

    def test_changed_file_or_mismatched_workers_cannot_proceed(self):
        library = Library()
        with self.assertRaises(RuntimeError):
            bench.configure_oproj_comm_model(self.write(), 4, 132, library, self.runtime, {})
        self.assertFalse(library.fuse_mxfp8_test_set_oproj_comm_model.calls)
        workers = SimpleNamespace(all_gather_object=lambda output, value:
                                  output.__setitem__(slice(None), [value] * 3 + [dict(value, sha256='wrong')]))
        with self.assertRaises(RuntimeError):
            self.install(library, workers)

    def test_native_status_is_not_ignored(self):
        library = Library()
        library.fuse_mxfp8_test_set_oproj_comm_model = Call(lambda *args: 1)
        with self.assertRaises(RuntimeError):
            self.install(library)

    def test_refresh_labels_fallback_honestly_manual_priority_and_other_ops_unchanged(self):
        class Arguments(ct.Structure):
            _fields_ = [('geometry', ct.c_int32 * 13)]

        library = Library()
        self.install(library)
        query = library.fuse_mxfp8_test_get_oproj_comm_model
        with patch.dict('sys.modules', test_operators=SimpleNamespace(Arguments=Arguments)):
            for op in (0, 1, 2, 3):
                state = bench.OperatorCase.__new__(bench.OperatorCase)
                state.op, state.world, state.library, state.runtime = op, 4, library, self.runtime
                state.argument, state.config = Arguments(), {}
                previous_queries = len(query.calls)
                state.refresh_config()
                self.assertEqual(len(query.calls) - previous_queries, int(op == 1))
                expected = 'calibrated_model_or_domain_fallback' if op == 1 else 'existing_bf16_production_auto'
                self.assertEqual(state.config['dispatch'], expected)
                self.assertEqual('oproj_comm_model' in state.config, op == 1)
                if op == 1:
                    library.selected[0] = 8
                    state.refresh_config(requested_comm_ctas=8)
                    self.assertEqual(state.config['dispatch'], 'explicit_comm_existing_bf16_tile_auto')
                    self.assertEqual(state.config['comm_ctas'], 8)
                    self.assertIn('not_legacy_auto', state.config['oproj_comm_model']['auto_reference'])


def qkv_forward_fixture():
    return dict(schema='mxfp8-qkv-forward-service-model-v1', complete=True, frozen=True,
                models=[dict(world_size=4, sm_count=132, minimum_gain=0.0,
                             coefficients=dict(compute_gflop_sm_us=171.676, route_slot_task_us=5.913),
                             domain=dict(min_m=32768, max_m=131072, min_n=4096, max_n=18432,
                                         min_k=2048, max_k=16384, m_multiple=256, n_multiple=256,
                                         k_multiple=64, head_dim=128, tile_families=[[128, 256, 64, 2]],
                                         comm_ctas=list(range(4, 25, 2))))])


class QkvForwardCommModelTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / 'model.json'
        self.runtime = SimpleNamespace(check=check)

    def write(self, document=None):
        self.path.write_text(json.dumps(qkv_forward_fixture() if document is None else document))
        return self.path

    def test_forward_model_is_opt_in_and_does_not_require_new_symbols_by_default(self):
        self.assertIsNone(bench.arguments([]).qkv_comm_model)
        self.assertIsNone(profiler.arguments(['--cp', '4']).qkv_comm_model)
        self.assertIsNone(bench.configure_qkv_forward_comm_model(None, 4, 132, object(), object(), {}))
        self.assertIsNone(bench.read_qkv_forward_comm_model(SimpleNamespace(), object()))
        for parse, prefix in ((bench.arguments, []), (profiler.arguments, ['--cp', '4'])):
            self.assertEqual(parse(prefix + ['--qkv-comm-model', str(self.path)]).qkv_comm_model, self.path)

    def test_forward_loader_requires_exact_domain_and_finite_coefficients(self):
        self.write()
        loaded = bench.load_qkv_forward_comm_model(self.path, 4, 132)
        self.assertEqual(loaded['operator'], 'qkv_forward')
        self.assertEqual(bench.qkv_forward_comm_model_values(loaded), [171.676, 5.913, 0.0])
        for cp, sm in ((8, 132), (4, 120)):
            with self.subTest(cp=cp, sm=sm), self.assertRaises(ValueError):
                bench.load_qkv_forward_comm_model(self.path, cp, sm)
        changes = [('schema', 'wrong'), ('complete', 1), ('frozen', False), ('models', [])]
        for key, value in changes:
            document = qkv_forward_fixture()
            document[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                bench.load_qkv_forward_comm_model(self.write(document), 4, 132)
        for field in ('compute_gflop_sm_us', 'route_slot_task_us', 'minimum_gain'):
            for value in (None, True, '1', -1, float('nan'), float('inf'), 10**400,
                          1 if field == 'minimum_gain' else 0):
                document = qkv_forward_fixture()
                model = document['models'][0]
                (model if field == 'minimum_gain' else model['coefficients'])[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    bench.load_qkv_forward_comm_model(self.write(document), 4, 132)
        for change in (dict(min_m=1024), dict(head_dim=64), dict(comm_ctas=[4, 8]),
                       dict(tile_families=[[128, 320, 64, 2]]), dict(m_multiple=256.0)):
            document = qkv_forward_fixture()
            document['models'][0]['domain'].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                bench.load_qkv_forward_comm_model(self.write(document), 4, 132)

    def test_forward_install_and_metadata_leave_other_operators_unchanged(self):
        library = Library()
        native = [0.] * 5

        def setter(cp, sm, values):
            native[:] = [cp, sm, *values]
            return 0

        library.fuse_mxfp8_test_set_qkv_forward_comm_model = Call(setter)
        library.fuse_mxfp8_test_get_qkv_forward_comm_model = Call(lambda output: Library.copy(output, native))
        model = bench.load_qkv_forward_comm_model(self.write(), 4, 132)
        sources = {model['source']: model['sha256']}
        installed = bench.configure_qkv_forward_comm_model(self.path, 4, 132, library, self.runtime, sources)
        self.assertEqual(native, [4, 132, 171.676, 5.913, 0.0])
        self.assertEqual(bench.read_qkv_forward_comm_model(library, self.runtime), installed)
        with self.assertRaises(RuntimeError):
            bench.configure_qkv_forward_comm_model(self.path, 4, 132, library, self.runtime, {})
        class Arguments(ct.Structure):
            _fields_ = [('geometry', ct.c_int32 * 13)]

        with patch.dict('sys.modules', test_operators=SimpleNamespace(Arguments=Arguments)):
            for op in range(4):
                state = bench.OperatorCase.__new__(bench.OperatorCase)
                state.op, state.world, state.library, state.runtime = op, 4, library, self.runtime
                state.argument, state.config = Arguments(), {}
                state.refresh_config()
                self.assertEqual('qkv_forward_comm_model' in state.config, op == 0)
                self.assertEqual(state.config['dispatch'], 'calibrated_model_or_domain_fallback'
                                 if op == 0 else 'existing_bf16_production_auto')
                if op == 0:
                    state.refresh_config(requested_comm_ctas=12)
                    self.assertEqual(state.config['dispatch'], 'explicit_comm_existing_bf16_tile_auto')
        native[2] += 1
        with self.assertRaises(RuntimeError):
            bench.read_qkv_forward_comm_model(library, self.runtime)


if __name__ == '__main__':
    unittest.main()
