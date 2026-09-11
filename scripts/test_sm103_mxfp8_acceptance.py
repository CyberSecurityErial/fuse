"""Synthetic result-summary contracts; these fixtures are not GPU measurements."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import summarize_sm103_mxfp8_fused as report
from tune_sm103_mxfp8_fused import auto_acceptance_plan


def result(sequence=131072, automatic=True, comm=32, time_ms=.27):
    world, n, k = 8, 4096, 2048
    m = sequence // world
    row = dict(m=m, n=n, k=k, world=world, global_seq=sequence, run_id='synthetic-auto', candidate_id=3,
        component='fused', precision='MXFP8_E4M3_UE8M0_accFP32_BF16out', boundary=report.BOUNDARIES['fused'],
        p50_ms=time_ms, p95_ms=time_ms, pflops_per_rank=2*m*n*k/time_ms/1e12,
        raw_maxrank_ms=[time_ms]*50, half_drift=0., warmup_calls=10, epilogue_n=32,
        validation='both_payloads_and_postmeasurement_full_numeric_and_route',
        binary_sha256='a'*64, environment_fingerprint='b'*64, artifact_sha256='c'*64, source_id='d'*64,
        configuration=dict(comm_sm=str(comm), tile='m128n256', tile_m='128', tile_n='256', tile_k='128',
            raster='along_m', effective_swizzle_size='8', max_swizzle_size='8', dynamic_smem='220160',
            scheduled_compute_ctas=str(148-comm)), communication_selection=dict(mode='explicit'))
    if automatic:
        version = 'validation_pending_SYNTHETIC'
        row['communication_selection'] = dict(mode='runtime_model', requested_comm_ctas=0,
            launch_comm_ctas=0, resolved_comm_ctas=comm, model_version=version,
            rank_queries=[dict(rank=str(rank), requested_comm='0', launch_comm='0',
                resolved_comm=str(comm), comm_sm=str(comm), model_version=version) for rank in range(world)])
    return row


def history(sequence=131072, confirmed=True):
    old = result(sequence, automatic=False, comm=72, time_ms=.25)
    old.update(run_id='synthetic-confirm', binary_sha256='e'*64)
    old['configuration'].update(raster='along_n', effective_swizzle_size='1', max_swizzle_size='1')
    winner = deepcopy(old)
    winner['run_id'] = 'synthetic-search'
    row = {key: old[key] for key in report.COMPARISON_KEY}
    row.update(pure_reference_id=f'synthetic_s{sequence}_cp8', winner=winner,
               confirmation=old if confirmed else None)
    return dict(schema='sm103_mxfp8_fused_tuning_v1', rows=[row])


class AcceptanceSummaryTests(unittest.TestCase):
    def test_user_gate_is_strict_per_point_and_requires_complete_same_gemm(self):
        def table(ratios):
            return dict(rows=[dict(paired_manual={'evidence': True}, same_historical_gemm=True,
                auto_over_paired_manual=r) for r in ratios])
        self.assertTrue(report.acceptance_gate(table([.95]))['passed'])
        self.assertTrue(report.acceptance_gate(table([.901, 1.01]))['passed'])
        self.assertFalse(report.acceptance_gate(table([.90, 1.1]))['passed'])
        self.assertFalse(report.acceptance_gate(table([.94]))['passed'])
        missing = table([.98, .99])
        missing['rows'][1]['paired_manual'] = None
        self.assertFalse(report.acceptance_gate(missing)['complete'])
        different = table([.99])
        different['rows'][0]['same_historical_gemm'] = False
        self.assertFalse(report.acceptance_gate(different)['passed'])
        for value in (0, float('nan'), float('inf')):
            with self.assertRaises(ValueError): report.acceptance_gate(table([value]))

    def test_manual_archive_retains_confirmation_without_grid_history(self):
        old = history()
        old['rows'][0]['candidates'] = ['unwanted search detail']
        old['rows'].extend(history(524288, confirmed=False)['rows'])
        with tempfile.TemporaryDirectory() as temporary:
            saved, paths = report.archive_manual_best(old, {'sha256': 'f'*64}, Path(temporary))
            self.assertEqual(saved['rows'][0]['confirmation'], old['rows'][0]['confirmation'])
            self.assertNotIn('candidates', saved['rows'][0])
            self.assertEqual(set(saved['rows'][0]['winner']), {'run_id', 'configuration', 'epilogue_n'})
            self.assertIsNone(saved['rows'][1]['confirmation'])
            self.assertEqual({p.name for p in paths}, {'manual-best-current.json', 'manual-best-current.md'})
            report._check_confirmation(saved['rows'][0], saved['rows'][0]['confirmation'])

    def test_full_plan_keeps_old_oom_and_fixed_historical_gemm(self):
        old = history()
        confirmed = old['rows'][0]['confirmation']
        confirmed['run_id'] = '20260911-010101-abcdef'
        absent = deepcopy(old['rows'][0])
        absent.update(global_seq=524288, m=65536, confirmation=None, winner=None,
                      run_id='20260911-010102-abcdef', status='resource_skipped')
        old['rows'].append(absent)
        pure = dict(rows=[dict(m=65536, n=4096, k=2048, winner=dict(config='m128n256k128e32s0sw8M'))])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for row, run_id in ((old['rows'][0], confirmed['run_id']), (absent, absent['run_id'])):
                folder = root/run_id
                folder.mkdir()
                (folder/'job.json').write_text(json.dumps(dict(world=row['world'], global_seq=row['global_seq'],
                    hidden=row['k'], q_heads=16, kv_heads=8, head_dim=128)))
            tasks = auto_acceptance_plan(old, pure, root)
            self.assertEqual(len(tasks), 2)
            self.assertEqual((tasks[0]['raster'], tasks[0]['max_swizzle_size'], tasks[0]['manual_comm']),
                             ('along_n', 1, 72))
            self.assertEqual(tasks[1]['gemm_source'], 'independent_pure_gemm_winner')
            self.assertEqual(tasks[1]['raster'], 'along_m')
            self.assertIsNone(tasks[1]['manual_comm'])
            old['rows'].append(deepcopy(absent))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                auto_acceptance_plan(old, pure, root)

    def test_paired_budget_replay_never_selects_live_fastest(self):
        auto = result()
        replay = result(automatic=False, comm=72, time_ms=.26)
        faster = result(automatic=False, comm=16, time_ms=.20)
        summary = report.build_acceptance_summary([faster, replay, auto], history(), {})
        row = summary['rows'][0]
        self.assertEqual(row['paired_manual'], replay)
        self.assertAlmostEqual(row['auto_over_paired_manual'], .26/.27)
        self.assertFalse(row['same_historical_gemm'])
        self.assertEqual(row['paired_comparison'], 'same_run_historical_budget_on_current_gemm')
        self.assertEqual(summary['coverage']['paired_points'], 1)
        old = history()
        for entry in ('winner', 'confirmation'):
            old['rows'][0][entry]['configuration'].update(
                raster='along_m', effective_swizzle_size='8', max_swizzle_size='8')
        same = report.build_acceptance_summary([replay, auto], old, {})['rows'][0]
        self.assertTrue(same['same_historical_gemm'])
        self.assertEqual(same['paired_comparison'], 'same_run_historical_best_config')
        for edit in ('other_run', 'other_epilogue', 'other_layout'):
            altered = deepcopy(replay)
            if edit == 'other_run': altered['run_id'] = 'another-run'
            if edit == 'other_epilogue': altered['epilogue_n'] = 64
            if edit == 'other_layout': altered['configuration']['raster'] = 'along_n'
            missing = report.build_acceptance_summary([altered, auto], old, {})['rows'][0]
            self.assertIsNone(missing['paired_manual'])
        with self.assertRaises(ValueError):
            report.build_acceptance_summary([replay, deepcopy(replay), auto], history(), {})

    def test_main_table_only_auto_against_confirmed_best(self):
        candidates = [result(automatic=False, comm=16), result(automatic=False, comm=32), result()]
        summary = report.build_acceptance_summary(candidates, history(), {'sha256': 'f'*64})
        self.assertEqual(len(summary['audited_candidates']), 3)
        self.assertEqual(len(summary['rows']), 1)
        row = summary['rows'][0]
        self.assertAlmostEqual(row['auto_over_confirmed_best'], .25/.27)
        self.assertEqual(row['confirmed_best']['run_id'], 'synthetic-confirm')
        self.assertEqual(row['domain_role'], 'calibration')
        self.assertFalse(row['same_binary'])
        text = report.acceptance_markdown(summary)
        self.assertIn('E32 / M / 8 / 32', text)
        self.assertIn('E32 / N / 1 / 72', text)
        self.assertNotIn('E32 / M / 8 / 16', text)
        self.assertEqual(summary['acceptance_status'], 'incomplete_paired_coverage')
        holdout = report.build_acceptance_summary([result(262144)], history(262144), {})
        self.assertEqual(holdout['rows'][0]['domain_role'], 'holdout')

    def test_missing_confirmation_never_falls_back_to_search_winner(self):
        summary = report.build_acceptance_summary([result()], history(confirmed=False), {})
        row = summary['rows'][0]
        self.assertIsNone(row['confirmed_best'])
        self.assertIsNone(row['auto_over_confirmed_best'])
        self.assertEqual(row['comparison'], 'missing_historical_confirmation')

    def test_missing_auto_does_not_shrink_full_table_coverage(self):
        old = history()
        old['rows'].extend(history(262144)['rows'])
        summary = report.build_acceptance_summary([result()], old, {})
        self.assertEqual(summary['coverage']['target_points'], 2)
        self.assertEqual(summary['coverage']['auto_points'], 1)
        missing = summary['rows'][1]
        self.assertIsNone(missing['auto'])
        self.assertIsNone(missing['auto_over_confirmed_best'])
        self.assertEqual(missing['comparison'], 'missing_auto_measurement')
        self.assertIn('1 / 2', report.acceptance_markdown(summary))

    def test_invalid_zero_evidence_geometry_or_historical_values_rejected(self):
        for edit in ('positive_launch', 'missing_rank', 'duplicate_auto', 'historical_value', 'geometry'):
            current, old = [result()], history()
            if edit == 'positive_launch': current[0]['communication_selection']['launch_comm_ctas'] = 32
            if edit == 'missing_rank': current[0]['communication_selection']['rank_queries'].pop()
            if edit == 'duplicate_auto': current.append(deepcopy(current[0]))
            if edit == 'historical_value': old['rows'][0]['confirmation']['pflops_per_rank'] *= 2
            if edit == 'geometry': old['rows'][0]['confirmation']['m'] += 128
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                report.build_acceptance_summary(current, old, {})

    def test_archive_audits_all_candidates_before_replacing_pair(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, output = root/'run', root/'autotune'
            run.mkdir(); output.mkdir()
            (run/'job.json').write_text(json.dumps({'auto_mxfp8_comm': True}))
            old = root/'fused-current.json'
            old.write_text(json.dumps(history()))
            (output/'acceptance-current.md').write_text('previous')
            candidates = [result(automatic=False, comm=16), result(automatic=False, comm=32), result()]
            with mock.patch.object(report.l20d, 'fused_candidates', return_value=([16, 32, 0], ['m128n256'], [])):
                with mock.patch.object(report, 'audit_run', side_effect=[candidates[0], ValueError('bad manual candidate')]):
                    with self.assertRaises(ValueError): report.archive_acceptance([run], old, output)
                self.assertEqual((output/'acceptance-current.md').read_text(), 'previous')
                with mock.patch.object(report, 'audit_run', side_effect=candidates) as audit:
                    summary, paths = report.archive_acceptance([run], old, output)
                    self.assertEqual(audit.call_count, 3)
            self.assertEqual({p.name for p in paths}, {'acceptance-current.md', 'acceptance-current.json'})
            self.assertEqual(len(summary['audited_candidates']), 3)
            self.assertEqual({p.name for p in output.iterdir()}, {'acceptance-current.md', 'acceptance-current.json'})


if __name__ == '__main__':
    unittest.main()
