"""Small synthetic curves; no GPU, cloud, or external calibration dependency."""

import copy
import unittest

import plan_sm103_oproj as planner


def calibration():
    runs = []
    for k in (8192, 16384):
        geometry = dict(seq_local=32768, hidden=8192, q_width=k, world=4)
        run = dict(run_id=f'fixture-k{k}', source_id='s' * 64, node='09',
            environment_fingerprint='e' * 64, diagnostic_only=False, geometry=geometry,
            build=dict(profile=False, mpi=True, build_inputs='b' * 64, binary_sha256='d' * 64,
                       environment_fingerprint='e' * 64),
            config=dict(calibrate='1', profile='0', validation_self_test='0', launch='graph',
                        causal='1', input_generator='gpu_philox'), candidates=[])
        for candidate, comm in enumerate((8, 16, 32), 1):
            compute = 148 - comm
            waves = planner.model.ceil_div(8192, compute)
            cycle = {8: 10., 16: 12., 32: 30.}[comm] * k / 8192
            bandwidth = {8: 200., 16: 1200., 32: 1600.}[comm] * (1 if k == 8192 else .8)
            base = dict(candidate=candidate, **planner.FIXED, performance_accepted=True,
                m=32768, n=8192, k=k, world=4, sm_count=148, comm_ctas=comm,
                tile_policy='m128n256', tile_m=128, tile_n=256, tile_k=64, cluster_ctas=1,
                raster='along_m', raster_requested='along_m', schedule_schema='explicit_v1',
                max_swizzle_size=8, effective_swizzle_size=8, swizzle=8, padded_m_tiles=256,
                padded_n_tiles=32, has_padding=False, scheduled_work_tiles_derived=8192,
                production_compute_ctas_derived=[compute] * 4, work_tiles_derived=8192,
                problem_route_payload_bytes=2 * 32768 * k, sampling_mode='quick_1_5', formal_eligible=False,
                production_resources=[dict(rank=rank, tile_m=128, tile_n=256, tile_k=64, threads=256,
                                           dynamic_smem=214016) for rank in range(4)])
            for component in ('fused', 'compute_reference', 'copy_reference'):
                row = copy.deepcopy(base)
                row.update(component=component,
                    measurement_role='production' if component == 'fused' else 'calibration',
                    compute_ctas_derived=[0 if component == 'copy_reference' else compute] * 4,
                    executed_gemm_flops=0 if component == 'copy_reference' else 2 * 32768 * 8192 * k,
                    executed_route_payload_bytes=0 if component == 'compute_reference' else 2 * 32768 * k,
                    timing=dict(p50_ms=(cycle * waves / 1000 if component == 'compute_reference' else
                                       2 * 32768 * k / bandwidth / 1e6 if component == 'copy_reference' else 1.0)))
                run['candidates'].append(row)
        runs.append(run)
    return dict(schema=planner.CALIBRATION_SCHEMA, runs=runs)


def cases(k=12288):
    return dict(schema=planner.CASES_SCHEMA, skipped=[{'id': 'explicit-oom'}], cases=[
        dict(id='synthetic', model='arbitrary_label', m=32768, n=8192, k=k, world=4, sm_count=148,
             historical_pflops=999, candidates=[dict(tile_policy='m128n256', raster='along_m',
                                                   max_swizzle_size=8, comm_ctas=[8, 16, 32])])])


class OProjPlannerTests(unittest.TestCase):
    def test_k_interpolation_and_effective_service_units(self):
        data = planner.read_calibration(calibration())
        key = (4, 148, 'm128n256', 'along_m', 8, 8, 16)
        values, evidence = planner.interpolate(data['anchors'][key], 12288)
        self.assertAlmostEqual(values['tile_cycle_us'], 18.)
        self.assertAlmostEqual(values['copy_bandwidth_gb_s'], 1080.)
        self.assertEqual([row['k'] for row in evidence], [8192, 16384])
        exact, evidence = planner.interpolate(data['anchors'][key], 8192)
        self.assertAlmostEqual(exact['tile_cycle_us'], 12.)
        self.assertEqual(len(evidence), 1)

    def test_candidate_ranking_uses_both_delivery_and_compute(self):
        result = planner.plan(calibration(), cases())
        row = result['cases'][0]
        self.assertEqual(row['top2'][0]['comm_ctas'], 16)
        self.assertEqual(len(row['top2']), 2)
        self.assertFalse(result['globally_optimal'])
        self.assertFalse(result['fitted_to_fused'])
        self.assertEqual(result['input_metadata']['skipped'], [{'id': 'explicit-oom'}])
        self.assertEqual(row['metadata']['model'], 'arbitrary_label')
        for candidate in row['candidates']:
            self.assertGreater(candidate['prediction']['production_consumption_ratio'], 0)
            self.assertIsNone(candidate['measured_startup_us'])
            self.assertEqual(candidate['prediction']['service_basis'], 'amortized_full_boundary')
            self.assertNotIn('launch_us', candidate['prediction'])

    def test_fused_or_model_labels_cannot_fit_or_rank(self):
        first = planner.plan(calibration(), cases())
        changed, changed_cases = calibration(), cases()
        for run in changed['runs']:
            for row in run['candidates']:
                if row['component'] == 'fused':
                    row['timing']['p50_ms'] *= 1000
        changed_cases['cases'][0].update(model='unrelated', historical_pflops=.0001)
        second = planner.plan(changed, changed_cases)
        self.assertEqual(first['cases'][0]['top2'], second['cases'][0]['top2'])
        self.assertEqual([c['prediction'] for c in first['cases'][0]['candidates']],
                         [c['prediction'] for c in second['cases'][0]['candidates']])
        self.assertNotEqual(first['calibration_reconstruction_checks'], second['calibration_reconstruction_checks'])

    def test_compute_budget_curve_is_not_linear_sm_scaling(self):
        result = planner.plan(calibration(), cases(8192))['cases'][0]['candidates']
        self.assertEqual([c['service_estimates']['tile_cycle_us'] for c in result], [10., 12., 30.])
        self.assertNotAlmostEqual(result[1]['service_estimates']['tile_cycle_us'], 10 * 140 / 132)

    def test_missing_cp_budget_collective_or_schedule_is_unsupported(self):
        data = planner.read_calibration(calibration())
        case = cases()['cases'][0]
        spec = case['candidates'][0]
        for case_edit, spec_edit, comm in (({'world': 8}, {}, 16), ({}, {}, 24),
                 ({}, {'tile_policy': 'm128n256k64e32'}, 16), ({}, {'raster': 'along_n'}, 16),
                 ({}, {'max_swizzle_size': 4}, 16), ({'layout': 'other'}, {}, 16)):
            with self.subTest(case_edit=case_edit, spec_edit=spec_edit, comm=comm):
                self.assertEqual(planner.predict(data, case | case_edit, spec | spec_edit, comm)['status'], 'unsupported')

    def test_missing_k_bracket_and_extrapolation_are_unsupported(self):
        document = calibration()
        document['runs'] = document['runs'][:1]
        data = planner.read_calibration(document)
        case = cases()['cases'][0]
        self.assertEqual(planner.predict(data, case, case['candidates'][0], 16)['status'], 'unsupported')
        data = planner.read_calibration(calibration())
        for k in (4096, 32768):
            self.assertEqual(planner.predict(data, case | {'k': k}, case['candidates'][0], 16)['status'], 'unsupported')

    def test_cross_mn_is_marked_unvalidated(self):
        document = cases()
        document['cases'][0].update(m=65536, n=4096)
        result = planner.plan(calibration(), document)['cases'][0]['candidates'][0]
        self.assertTrue(result['cross_mn'])
        self.assertFalse(result['externally_validated'])

    def test_invalid_or_mismatched_calibration_is_rejected(self):
        mutations = [lambda d: d['runs'][0]['candidates'].pop(),
            lambda d: d['runs'][0]['candidates'][1].update(n=4096),
            lambda d: d['runs'][0]['candidates'][1].update(compute_ctas_derived=[148] * 4),
            lambda d: d['runs'][0]['candidates'][1]['production_resources'][0].update(dynamic_smem=100),
            lambda d: d['runs'][0]['candidates'][2]['timing'].update(p50_ms=float('nan')),
            lambda d: d['runs'][1]['build'].update(binary_sha256='other'),
            lambda d: d['runs'].append(copy.deepcopy(d['runs'][0]))]
        for mutate in mutations:
            document = calibration()
            mutate(document)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                planner.read_calibration(document)

    def test_script_source_changes_preserve_build_partition_and_evidence(self):
        document = calibration()
        document['runs'][1]['source_id'] = 'new-script-snapshot'
        result = planner.plan(document, cases())['cases'][0]['candidates'][0]
        self.assertEqual([row['source_id'] for row in result['evidence']],
                         ['s' * 64, 'new-script-snapshot'])

    def test_same_policy_cannot_hide_a_changed_resource_signature(self):
        document = calibration()
        for row in document['runs'][1]['candidates']:
            for resource in row['production_resources']:
                resource['dynamic_smem'] += 128
        with self.assertRaisesRegex(ValueError, 'Ambiguous actual tile signature'):
            planner.read_calibration(document)

    def test_reconstruction_error_and_budget_hit_statistics(self):
        checks = [dict(geometry={'m': 32768}, schedule={'raster': 'along_m'},
                       comm_ctas=comm, predicted_us=predicted, observed_fused_us=observed,
                       relative_error=predicted / observed - 1)
                  for comm, predicted, observed in ((8, 120., 100.), (16, 110., 120.), (32, 150., 160.))]
        result = planner.reconstruction_summary(checks)
        self.assertEqual(result['points'], 3)
        self.assertEqual(result['budget_groups'], 1)
        self.assertEqual(result['five_budget_groups'], 0)
        self.assertEqual(result['top1_hits'], 0)
        self.assertEqual(result['top2_hits'], 1)
        self.assertEqual(result['top2_hit_rate'], 1.)
        self.assertAlmostEqual(result['max_absolute_relative_error'], .2)
        self.assertAlmostEqual(result['mean_absolute_relative_error'], (.2 + 1/12 + .0625) / 3)
        checks[2].update(observed_fused_us=80., relative_error=150/80-1)
        self.assertEqual(planner.reconstruction_summary(checks)['top2_hits'], 0)

    def test_invalid_cases_and_duplicate_candidates_are_rejected(self):
        for edit in ({'m': True}, {'k': 0}, {'global_seq': 123}):
            document = cases()
            document['cases'][0].update(edit)
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                planner.plan(calibration(), document)
        document = cases()
        document['cases'][0]['candidates'][0]['comm_ctas'].append(16)
        with self.assertRaisesRegex(ValueError, 'Duplicate physical candidate'):
            planner.plan(calibration(), document)


if __name__ == '__main__':
    unittest.main()
