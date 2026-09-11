"""Synthetic host-only tests of direct service intervals, NOT GPU measurements."""

from copy import deepcopy
import json
import io
from pathlib import Path
import shutil
import subprocess
import tempfile
import tarfile
import unittest
from unittest import mock

import plan_sm103_mxfp8 as plan


def fixture(comm=16):
    """Complete artificial capture, with phase-shifted sparse publications."""
    key = dict(m=4096, n=1536, k=2048, world=4, sm_count=148, capability=103,
               comm_ctas=comm, compute_ctas=148-comm, tile_m=128, tile_n=256, tile_k=128,
               epilogue_n=32, stages=4, cluster_ctas=1, raster=0, resolved_swizzle=1,
               max_swizzle_size=1, dynamic_smem_bytes=187392, q_heads=4, kv_heads=4,
               head_dim=128, weight_preparation='comm', rank_swizzle='off')
    workers, steps, panels, mt = 8*comm, key['k']//4, key['n']//256, key['m']//128
    captures = {}
    for stage in plan.STAGES:
        def record(kind, **values):
            return dict(schema=plan.SCHEMA, kind=kind, stage=stage, rank=0, iteration=0, **values)
        rows = captures[stage] = {kind: [] for kind in plan.KINDS}
        rows['config'].append(record('config', **key, captured_rank=0, capture_epochs=1,
            validation_generations=2, warmup=10, samples=50, clock='globaltimer', clock_unit='ns',
            event_boundary='host_api_inclusive_eager_cuda_event',
            gpu_uuid='GPU-01234567-89ab-cdef-0123-456789abcdef',
            control_samples_ms=[1.]*50, instrumented_samples_ms=[1.1]*50,
            quant_phase_steps=int(stage == 'QR_phase1'), delayed_panel=5,
            delay_ns=40000 if stage == 'C_delay2' else 10000))
        cta_count = 148 if stage.startswith('C_') else comm
        rows['cta'] = [record('cta', index=i, begin=100, setup_done=200, end=60000) for i in range(cta_count)]
        if stage.startswith('C_'):
            for logical in range(mt*panels):
                m, n = logical % mt, logical // mt
                cta, ordinal = comm+logical % (148-comm), logical // (148-comm)
                begin = 1000+ordinal*200
                values = dict(index=m*panels+n, cta=cta, warp=4, m=m, n=n,
                    first_load=begin, load_return=begin+40, store_begin=begin+90,
                    ready_after=begin+100, wait_begin=0, wait_end=0)
                if stage != 'C_allready' and n == 5:
                    release = rows['config'][0]['delay_ns']
                    values.update(first_load=8000, load_return=release+40, store_begin=release+90,
                                  ready_after=release+100, wait_begin=8050, wait_end=release+5)
                rows['tile'].append(record('tile', **values))
            if stage != 'C_allready':
                release = rows['config'][0]['delay_ns']
                rows['panel'].append(record('panel', index=5, release_begin=release, release=release+4))
            continue
        task_count = mt*panels*4
        if stage != 'Q':
            for index in range(task_count):
                ordinal, worker = divmod(index, workers)
                begin = 1000+ordinal*500
                rows['route'].append(record('route', index=index, cta=worker % comm, warp=worker//comm,
                    row=(index//(key['n']//128))*64, column=(index % (key['n']//128))*128,
                    rows=64, columns=128, peer=0, segment=0, begin=begin, ready=begin+20,
                    g2s_begin=begin+25, g2s_done=begin+100, s2g_begin=begin+110,
                    s2g_read_done=begin+200, copy_end=begin+210))
            for worker in range(workers):
                rows['route'].append(record('route', index=task_count+worker, cta=worker % comm,
                    warp=worker//comm, begin=50000, s2g_read_done=50010))
        if stage == 'R':
            continue
        for index in range(steps*panels):
            ordinal, worker = divmod(index, workers)
            if stage == 'Q':
                begin = 1000+ordinal*30
            else:
                shifted = ordinal-int(stage == 'QR_phase1')
                if shifted < 0:
                    begin = 900
                elif shifted//2 < (task_count+workers-1-worker)//workers:
                    begin = 1000+(shifted//2)*500+(50 if shifted % 2 == 0 else 140)
                else:
                    begin = 10000+ordinal*30
            step = index % steps
            delta = step//workers+1 if step+workers >= steps else 0
            rows['quant'].append(record('quant', index=index, cta=worker % comm, warp=worker//comm,
                panel=index//steps, groups=32, arrival_chunks=delta, begin=begin, quant_done=begin+10,
                end=begin+(16 if delta else 12), warp_join_done=begin+11 if delta else 0,
                arrival_done=begin+13 if delta else 0, release=begin+15 if step == steps-1 else 0))
    return key, captures


def point(comm=16):
    key, captures = fixture(comm)
    result = plan.reduce_capture(key, captures)
    # The fixture has a smaller M to keep host tests cheap. Header tests only
    # exercise its schema, not reinterpret synthetic values as measurements.
    result['physical_key']['m'] = 131072//key['world']
    result['bulk_services']['reference_m'] = result['physical_key']['m']
    result['bulk_services']['measured_global_seq'] = 131072
    result['provenance'] = dict(run_id='SYNTHETIC_NOT_MEASURED', binary_sha256='a'*64,
                                environment_fingerprint='b'*64)
    return result


class DirectServicePlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.key, cls.captures = fixture()

    def test_bulk_services_exclude_finalize_and_keep_all_quant_workers(self):
        captures = deepcopy(self.captures)
        for stage in ('Q', 'R', 'QR'):
            for cta in captures[stage]['cta']:
                cta['end'] = 999999  # A finalize or thread0 exit is not the local boundary.
        captures['Q']['quant'][-1]['end'] = 100000
        result = plan.reduce_bulk_services(self.key, captures)
        self.assertAlmostEqual(result['compute_us'], (60000-200)/1000)
        self.assertAlmostEqual(result['quant_us'], (100000-200)/1000)
        self.assertAlmostEqual(result['route_us'], (50010-200)/1000)
        self.assertLess(result['quant_route_us'], 999)
        self.assertEqual(result['reference_m'], self.key['m'])

    def test_sidecar_can_be_reaudited_without_redundant_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            raw = b'synthetic archive payload\n'
            with tarfile.open(directory/'artifacts.tar.gz', 'w:gz') as archive:
                member = tarfile.TarInfo('control/services-rank-0.jsonl')
                member.size = len(raw)
                archive.addfile(member, io.BytesIO(raw))
            actual, path = plan.read_probe_sidecar(directory, 1)
            self.assertEqual(actual, raw)
            self.assertFalse(path.exists())  # Read-only; no duplicate extraction.
            path.parent.mkdir(parents=True)
            path.write_bytes(raw)
            self.assertEqual(plan.read_probe_sidecar(directory, 1)[0], raw)
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'differs'):
                plan.read_probe_sidecar(directory, 1)

    def test_full_sparse_services_and_closed_origins(self):
        result = plan.reduce_capture(self.key, self.captures)
        self.assertEqual(result['status'], 'validation_pending', result['unavailable_reasons'])
        services = result['services']
        self.assertEqual(set(services), set(plan.SERVICE_FIELDS))
        self.assertAlmostEqual(services['quant_publish_us'], .006)
        self.assertAlmostEqual(services['quant_g2s_us'], .040)
        self.assertAlmostEqual(services['quant_s2g_us'], .040)
        self.assertAlmostEqual(services['g2s_mixed_us'], .090)
        self.assertAlmostEqual(services['s2g_mixed_us'], .100)
        self.assertAlmostEqual(services['quant_g2s_publish_us'], .046)
        self.assertAlmostEqual(services['quant_s2g_publish_us'], .046)
        self.assertAlmostEqual(services['g2s_publish_us'], .090)
        self.assertAlmostEqual(services['s2g_publish_us'], .100)
        self.assertEqual((services['issue_us'], services['join_us']), (0, 0))
        self.assertAlmostEqual(services['tile_latency_us'], .100)
        self.assertEqual(result['publication_accounting'], 'sparse_separate')
        self.assertFalse(result['stages']['Q']['event_timings_used_for_service_coefficients'])

    def test_always_publish_is_included_not_fabricated_zero(self):
        key, captures = fixture(64)
        result = plan.reduce_capture(key, captures)
        self.assertEqual(result['status'], 'validation_pending', result['unavailable_reasons'])
        self.assertEqual(result['publication_accounting'], 'every_chunk_included')
        self.assertEqual(result['services']['quant_publish_us'], 0)
        self.assertAlmostEqual(result['services']['quant_us'], .030)
        self.assertAlmostEqual(result['services']['quant_g2s_us'], .046)
        self.assertNotIn('quant_publish_us', result['statistics'])
        for direction in ('g2s', 's2g'):
            self.assertEqual(result['services']['quant_'+direction+'_publish_us'],
                             result['services']['quant_'+direction+'_us'])
            self.assertEqual(result['services'][direction+'_publish_us'],
                             result['services'][direction+'_mixed_us'])

    def test_bulk_reference_ignores_per_publication_latency(self):
        sample = plan.reduce_capture(self.key, self.captures)
        expected = plan.score_bulk_point(sample, self.key['m'] * 4)
        changed = deepcopy(sample)
        for field in ('quant_publish_us', 'quant_g2s_publish_us', 'quant_s2g_publish_us',
                      'g2s_publish_us', 's2g_publish_us'):
            changed['services'][field] *= 1000
        self.assertEqual(plan.score_bulk_point(changed, self.key['m'] * 4), expected)
        self.assertEqual(expected['production_us'], sample['bulk_services']['quant_route_us'] +
                         3 * sample['bulk_services']['route_us'])
        self.assertEqual(plan.score_bulk_point(sample, self.key['m'])['production_us'],
                         sample['bulk_services']['quant_route_us'])
        with self.assertRaisesRegex(ValueError, 'long-sequence'):
            plan.score_bulk_point(sample, self.key['m'] // 2)

    def test_publishing_dma_cost_is_independent_of_q_only(self):
        captures = deepcopy(self.captures)
        for stage in ('QR', 'QR_phase1'):
            for chunk in captures[stage]['quant']:
                if chunk['arrival_chunks'] and chunk['begin'] < 10000:
                    chunk['end'] += 20  # Synthetic extra DMA-context publication time.
        result = plan.reduce_capture(self.key, captures)
        self.assertEqual(result['status'], 'validation_pending', result['unavailable_reasons'])
        self.assertAlmostEqual(result['services']['quant_publish_us'], .006)
        for direction in ('g2s', 's2g'):
            self.assertAlmostEqual(result['services']['quant_'+direction+'_publish_us'], .066)
            self.assertAlmostEqual(result['services']['quant_'+direction+'_us'], .040)

    def test_unobserved_publishing_context_is_unavailable(self):
        # With four chunks/worker/panel, unshifted QR publishes only in S2G.
        # No G2S-publication observation means unavailable, not Q-only fallback.
        captures = {k:v for k,v in self.captures.items() if k != 'QR_phase1'}
        result = plan.reduce_capture(self.key, captures)
        self.assertEqual(result['status'], 'unavailable')
        self.assertIn('unmeasurable service: quant_g2s_publish_us', result['unavailable_reasons'])
        self.assertNotIn('quant_g2s_publish_us', result['services'])

    def test_complete_publications_and_clock_order_required(self):
        for mutation in ('missing', 'delta', 'rank', 'order'):
            with self.subTest(mutation=mutation):
                captures = deepcopy(self.captures)
                quant = captures['Q']['quant']
                if mutation == 'missing': quant.pop()
                if mutation == 'delta': quant[511]['arrival_chunks'] += 1
                if mutation == 'rank': quant[0]['rank'] = 1
                if mutation == 'order': quant[0]['quant_done'] = quant[0]['begin']-1
                with self.assertRaises(ValueError):
                    plan.reduce_capture(self.key, captures)

    def test_missing_delayed_service_is_not_filled_from_fused(self):
        captures = {k:v for k,v in self.captures.items() if k != 'C_delay2'}
        result = plan.reduce_capture(self.key, captures)
        self.assertEqual(result['status'], 'unavailable')
        self.assertTrue(any('C_delay2' in text for text in result['unavailable_reasons']))

    def test_eager_times_cannot_change_coefficients(self):
        original = plan.reduce_capture(self.key, self.captures)
        captures = deepcopy(self.captures)
        for rows in captures.values():
            rows['config'][0]['instrumented_samples_ms'] = [100.]*50
        self.assertEqual(plan.reduce_capture(self.key, captures)['services'], original['services'])

    def test_raster_is_strict_integer_protocol(self):
        for value in ('along_m', True, -1, 2):
            with self.assertRaises(ValueError): plan.physical_key(self.key | {'raster': value})
        self.assertEqual(plan.physical_key(self.key | {'raster': 1})['raster'], 1)

    def test_drain_owner_when_task_count_not_multiple_workers(self):
        key, captures = fixture(20)
        self.assertNotEqual((key['m']//128)*(key['n']//256)*4 % (8*20), 0)
        result = plan.reduce_capture(key, captures)
        self.assertEqual(result['status'], 'validation_pending', result['unavailable_reasons'])

    def test_one_epoch_parser_and_uuid(self):
        def encoded(captures):
            return '\n'.join(json.dumps(row) for rows in captures.values()
                             for kind in plan.KINDS for row in rows[kind]).encode()
        key, _ = plan.read_capture(data=encoded(self.captures))
        self.assertEqual(key, self.key)
        captures = deepcopy(self.captures)
        captures['R']['config'][0]['gpu_uuid'] = 'GPU-11111111-89ab-cdef-0123-456789abcdef'
        with self.assertRaises(ValueError): plan.read_capture(data=encoded(captures))

    def test_export_domain_and_sources(self):
        measured = point()
        text = plan.export_cpp([measured])
        self.assertIn('validation_pending_', text)
        self.assertIn('p.m_min = 32768; p.m_max = 131072;', text)
        self.assertIn('p.k_interpolation_group = 0;', text)
        self.assertNotIn('Oproj', text)
        with self.assertRaises(ValueError): plan.export_cpp([measured, measured])
        second = point(64)
        second['provenance']['binary_sha256'] = 'c'*64
        with self.assertRaises(ValueError): plan.export_cpp([measured, second])

    def test_bulk_export_contains_only_decision_services(self):
        measured = point()
        text = plan.export_cpp([measured])
        self.assertIn('p.bulk.reference_m = 32768;', text)
        self.assertIn('p.bulk.quant_route_us =', text)
        self.assertIn('p.bulk.route_us =', text)
        self.assertIn('p.services.tile_cycle_us =', text)
        self.assertNotIn('p.services.quant_publish_us', text)
        self.assertNotIn('p.services.quant_g2s_us', text)
        measured['bulk_services']['reference_k'] += 128
        with self.assertRaisesRegex(ValueError, 'reference mismatch'):
            plan.export_cpp([measured])

    def test_generated_declarations_compile(self):
        compiler = shutil.which('clang++') or shutil.which('g++')
        if not compiler: self.skipTest('No host C++ compiler')
        root = Path(__file__).resolve().parents[1]
        source = (root/'csrc/operators/sm103/detail/model_calibration.cuh').read_text()
        start = source.index('struct Mxfp8QkvBulkServices')
        point_start = source.index('struct Mxfp8QkvCalibrationPoint')
        end = source.index('\n};', point_start) + len('\n};')
        services = 'struct Mxfp8QkvServices { ' + ''.join('double '+name+' = 0; ' for name in plan.SERVICE_FIELDS) + '};\n'
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'header.cc'
            path.write_text('#include <array>\n#include <cstdint>\n'+services+source[start:end]
                +plan.export_cpp([point()])+'int main() { return kMxfp8QkvCalibrationPoints.size()!=1; }\n')
            result = subprocess.run([compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror', '-fsyntax-only', str(path)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_production_host_score_bridge_and_temporary_cleanup(self):
        compiler = shutil.which('c++') or shutil.which('clang++') or shutil.which('g++')
        if not compiler: self.skipTest('No host C++ compiler')
        points = [point(16), point(64)]
        original = deepcopy(points)
        directories = []
        real_temporary = tempfile.TemporaryDirectory
        def tracked(*args, **kwargs):
            temporary = real_temporary(*args, **kwargs)
            directories.append(Path(temporary.name))
            return temporary
        with mock.patch.object(plan.tempfile, 'TemporaryDirectory', side_effect=tracked):
            result = plan.score_points(points, compiler=compiler)
        self.assertEqual(points, original)  # No coefficient/domain fitting or mutation.
        self.assertEqual(result['source'], 'production_cpp_select_mxfp8_qkv_plan')
        self.assertFalse(result['performance_accepted'])
        self.assertFalse(result['coefficient_fit'])
        self.assertEqual([row['global_seq'] for row in result['rows']], [131072, 262144, 524288])
        for row in result['rows']:
            self.assertEqual(row['m'], row['global_seq']//4)
            self.assertEqual([r['requested_comm_ctas'] for r in row['candidates']], [16, 64])
            best = min(row['candidates'], key=lambda r: (r['predicted_us'], r['selected_comm_ctas']))
            auto = row['auto']
            self.assertEqual(auto['status'], 'Success')
            self.assertEqual(auto['requested_comm_ctas'], 0)
            self.assertEqual(auto['selected_comm_ctas'], best['selected_comm_ctas'])
            self.assertEqual(auto['predicted_us'], best['predicted_us'])
            self.assertEqual(auto['events'], 0)
            self.assertTrue(auto['whole_service_model'])
            self.assertGreaterEqual(auto['predicted_us'], max(auto[field] for field in
                ('compute_finish_us', 'copy_finish_us')))
            self.assertIsNone(auto['exposed_feed_us'])
            self.assertIsNone(auto['output_tail_us'])
            for candidate, anchor in zip(row['candidates'], points):
                reference = plan.score_bulk_point(anchor, row['m'])
                self.assertAlmostEqual(candidate['predicted_us'], reference['score_us'])
        self.assertTrue(directories)
        self.assertTrue(all(not directory.exists() for directory in directories))

    def test_host_bridge_keeps_missing_calibration_missing(self):
        if not (shutil.which('c++') or shutil.which('clang++') or shutil.which('g++')):
            self.skipTest('No host C++ compiler')
        missing = point()
        missing.update(status='unavailable', unavailable_reasons=['SYNTHETIC missing service'])
        result = plan.score_points([missing], global_sequences=(131072,))
        row = result['rows'][0]
        for prediction in row['candidates'] + [row['auto']]:
            self.assertEqual(prediction['status'], 'UnsupportedCalibration')
            self.assertIsNone(prediction['predicted_us'])
            self.assertIsNone(prediction['selected_comm_ctas'])
        with self.assertRaises(ValueError): plan.score_points([point()], global_sequences=(1048576,))
        unsafe = point()
        unsafe['provenance']['run_id'] = 'comment\ncode'
        with self.assertRaises(ValueError): plan.score_points([unsafe])

    def test_unavailable_clears_stale_header_and_pair_failure_restores(self):
        measured = point()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            plan.write_outputs([measured], output)
            old = {p.name:p.read_bytes() for p in output.iterdir()}
            real_replace, calls = plan.os.replace, 0
            def fail_second(source, target):
                nonlocal calls
                calls += 1
                if calls == 2: raise OSError('synthetic replacement failure')
                return real_replace(source, target)
            with mock.patch.object(plan.os, 'replace', side_effect=fail_second):
                with self.assertRaises(OSError): plan.write_outputs([measured], output)
            self.assertEqual({p.name:p.read_bytes() for p in output.iterdir()}, old)
            measured['status'] = 'unavailable'
            plan.write_outputs([measured], output)
            self.assertIn('Mxfp8QkvCalibrationPoint, 0>', (output/'calibration-current.inc').read_text())


if __name__ == '__main__':
    unittest.main()
