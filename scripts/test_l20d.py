"""Local archive and recovery contracts; no screen, network, or GPU calls."""
import contextlib
import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unittest
from unittest import mock

import l20d


class WorkflowContracts(unittest.TestCase):
    def test_qkv_rank_swizzle_build_isolation(self):
        plain = self.fused_job(stage='fused-build', mpi=True)
        rotated = plain | dict(qkv_rank_swizzle=True)
        l20d.validate_job(rotated)
        self.assertNotEqual(l20d.fused_build_dir(plain), l20d.fused_build_dir(rotated))
        self.assertIn('-DFUSE_SM103_QKV_RANK_SWIZZLE=OFF', l20d.fused_argv(plain)[-1])
        self.assertIn('-DFUSE_SM103_QKV_RANK_SWIZZLE=ON', l20d.fused_argv(rotated)[-1])
        for change in (dict(profile=True), dict(stage='doctor'), dict(qkv_rank_swizzle=1)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                l20d.validate_job(rotated | change)

    def test_actual_cpp_counter_guard_accepts_mpi_owned_launch(self):
        compiler = shutil.which('c++')
        if not compiler:
            self.skipTest('host C++ compiler unavailable')
        source = (l20d.REPO / 'benchmarks/sm103/fused_bf16.cu').read_text()
        begin = source.index('  if (!options.counter_component.empty() || !options.counter_direction.empty()) {')
        guard = source[begin:source.index('  if (has_seq_local && has_global_seq)', begin)]
        unit = self.root / 'counter_guard.cpp'
        unit.write_text('''#include <string>
#include <vector>
#include <stdexcept>
namespace fused_mpi { bool enabled=false; }
struct Options {
 bool calibrate=true, profile=false, validation_self_test=false;
 std::string launch="eager", host_launch="sequential", counter_component="fused", counter_direction="qkv";
 std::vector<int> comm_sm_list{12}, qkv_policy_list{1}, oproj_policy_list{1};
};
void check(const Options& options) {\n''' + guard + '''}
int main() {
 Options o; check(o);
 fused_mpi::enabled=true; o.host_launch="mpi_process"; check(o);
 o.counter_component="compute_reference"; check(o);
 o.counter_component="copy_reference"; check(o);
 o.host_launch="sequential";
 try { check(o); return 1; } catch(const std::runtime_error&) {}
 o.host_launch="mpi_process"; o.calibrate=false;
 try { check(o); return 2; } catch(const std::runtime_error&) {}
 return 0;
}
''')
        executable = self.root / 'counter_guard'
        subprocess.run([compiler, '-std=c++17', str(unit), '-o', str(executable)], check=True, capture_output=True)
        subprocess.run([str(executable)], check=True, capture_output=True)

    def test_fused_counters_preserve_peer_concurrency_and_reject_ambiguous_ranges(self):
        job = self.fused_job(calibrate=True, directions='qkv', fused_counters='fused')
        l20d.validate_job(job)
        args = l20d.fused_argv(job)
        self.assertEqual(args[args.index('--counter-component') + 1], 'fused')
        self.assertEqual(args[args.index('--counter-direction') + 1], 'qkv')
        wrapped = l20d.fused_counter_argv(self.root, args)
        for key, value in (('--replay-mode', 'app-range'), ('--devices', '0'),
                           ('--launch-count', '1'),
                           ('--cache-control', 'none'), ('--clock-control', 'none')):
            self.assertEqual(wrapped[wrapped.index(key) + 1], value)
        self.assertNotIn('--profile-from-start', wrapped)
        l20d.validate_job(job | {'mpi': True, 'fused_counter_replay': 'application'})
        app = l20d.fused_counter_argv(self.root, ['mpiexec', '-n', '8', 'bench'], 'ncu', 'application')
        self.assertEqual(app[app.index('--target-processes') + 1], 'all')
        self.assertEqual(app[app.index('--profile-from-start') + 1], 'off')
        self.assertEqual(wrapped[wrapped.index('--nvtx-include') + 1], 'fuse_communication_counters/')
        l20d.validate_job(job | {'fused_counter_tool': 'nsys'})
        sampled = l20d.fused_counter_argv(self.root, args, 'nsys')
        for option in ('--gpu-metrics-devices=0', '--capture-range=cudaProfilerApi',
                       '--capture-range-end=stop', '--gpu-metrics-frequency=10000'):
            self.assertIn(option, sampled)
        self.assertNotIn('--replay-mode', sampled)
        with self.assertRaises(ValueError):
            l20d.validate_job(self.fused_job(fused_counter_tool='nsys'))
        for changes in ({'mpi': True}, {'calibrate': False}, {'profile': True},
                        {'directions': 'qkv,oproj'}, {'fused_launch': 'graph'},
                        {'host_launch': 'per_gpu_thread'}, {'stage': 'fused-build'},
                        {'comm_sm_list': '8,16'}, {'fused_counters': 'unknown'},
                        {'validation_self_test': True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                l20d.validate_job(job | changes)

    def test_fused_counter_export_rejects_missing_duplicate_and_nonfinite_metrics(self):
        (self.root / 'fused-counters.ncu-rep').write_bytes(b'fixture')
        header = ['ID', 'Kernel Name', *l20d.FUSED_COUNTER_METRICS]
        def export(rows):
            output = io.StringIO()
            import csv
            writer = csv.writer(output)
            writer.writerow(header)
            writer.writerows(rows)
            return subprocess.CompletedProcess([], 0, output.getvalue(), '')
        row = ['0', 'range', *['1.25'] * len(l20d.FUSED_COUNTER_METRICS)]
        with mock.patch.object(l20d, 'run_counter_tool', return_value=export([row])):
            l20d.collect_fused_counters(self.root, {})
        for rows in ([], [row, row], [row[:-1] + ['nan']], [row[:-1] + ['n/a']]):
            with self.subTest(rows=rows), mock.patch.object(l20d, 'run_counter_tool', return_value=export(rows)):
                with self.assertRaises((RuntimeError, ValueError)):
                    l20d.collect_fused_counters(self.root, {})

    def test_cutlass_counters_are_single_geometry_diagnostics_not_comparison_samples(self):
        matrix = {'schema': 'sm103_gemm_matrix_v1', 'shapes': [dict(id='long', m=16384, n=18432, k=16384)]}
        job = self.fused_job(stage='gemm-probe', launches='eager', gemm_matrix_payload=matrix,
                             compare_cutlass=True, cutlass_counters=True)
        l20d.validate_job(job)
        args = l20d.gemm_probe_argv(job, self.root)
        self.assertIn('--cutlass-counters', args)
        self.assertNotIn('--cutlass-counters', l20d.gemm_probe_argv(job | {'cutlass_counters': False}, self.root))
        for change in ({'compare_cutlass': False}, {'cutlass_counters': 1},
                       {'stage': 'fused-smoke'}, {'launches': 'graph'},
                       {'gemm_matrix_payload': matrix | {'shapes': matrix['shapes'] +
                            [dict(id='second', m=8192, n=18432, k=16384)]}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                l20d.validate_job(job | change)
        profile = l20d.cutlass_counter_argv(self.root, args)
        self.assertEqual(profile[0], 'ncu')
        self.assertEqual(profile[-len(args):], args)
        for option, value in (('--nvtx-include', 'fuse_cutlass_1sm_counters/'),
                              ('--launch-count', '1'), ('--replay-mode', 'kernel'),
                              ('--clock-control', 'none'), ('--cache-control', 'none')):
            self.assertEqual(profile[profile.index(option) + 1], value)
        self.assertNotIn('--set', profile)
        self.assertNotIn('--force-overwrite', profile)
        self.assertEqual(profile[profile.index('--metrics') + 1], ','.join(l20d.CUTLASS_COUNTER_METRICS))

    def test_counter_query_fails_closed_and_preserves_version_and_requested_metrics(self):
        version = subprocess.CompletedProcess([], 0, 'NCU version test\n', '')
        query = subprocess.CompletedProcess([], 0, '\n'.join(l20d.CUTLASS_COUNTER_METRICS), '')
        with mock.patch.object(l20d, 'run_counter_tool', side_effect=[version, query]) as queried:
            l20d.prepare_cutlass_counters(self.root, {})
        query_args = queried.call_args_list[1][0][0]
        self.assertEqual(query_args[query_args.index('--metrics') + 1],
                         ','.join(metric.split('.', 1)[0] for metric in l20d.CUTLASS_COUNTER_METRICS))
        contract = json.loads((self.root / 'cutlass-counter-contract.json').read_text())
        self.assertTrue(contract['diagnostic_only'])
        self.assertFalse(contract['performance_accepted'])
        self.assertEqual(contract['selected_calls'], 1)
        self.assertEqual(contract['selected_sm_mode'], 1)
        bad = subprocess.CompletedProcess([], 0, 'No metrics found', '')
        with mock.patch.object(l20d, 'run_counter_tool', side_effect=[version, bad]), self.assertRaisesRegex(RuntimeError, 'unavailable'):
            l20d.prepare_cutlass_counters(self.root, {})
        with self.assertRaisesRegex(RuntimeError, 'no counter report'):
            l20d.collect_cutlass_counters(self.root, {})
        (self.root / 'cutlass-counters.ncu-rep').write_bytes(b'opaque test report')
        raw_csv = 'ID,Kernel Name,' + ','.join(l20d.CUTLASS_COUNTER_METRICS) + '\n'
        raw_csv += '0,selected_kernel,' + ','.join(['1'] * len(l20d.CUTLASS_COUNTER_METRICS)) + '\n'
        imported = subprocess.CompletedProcess([], 0, raw_csv, 'warning\n')
        with mock.patch.object(l20d, 'run_counter_tool', return_value=imported) as run:
            l20d.collect_cutlass_counters(self.root, {})
        self.assertIn('--import', run.call_args[0][0])
        self.assertEqual((self.root / 'cutlass-counters.csv').read_text(), imported.stdout)
        for raw in ('ID,Kernel Name\n0,kernel\n', raw_csv + raw_csv.splitlines()[-1] + '\n',
                    raw_csv.replace('0,selected_kernel,1,', '0,selected_kernel,nan,')):
            with mock.patch.object(l20d, 'run_counter_tool', return_value=subprocess.CompletedProcess([], 0, raw, '')):
                with self.assertRaises((RuntimeError, ValueError)):
                    l20d.collect_cutlass_counters(self.root, {})

    def test_graph_is_explicit_mpi_and_has_distinct_measurement_contract(self):
        eager = self.fused_job(mpi=True)
        graph = eager | {'fused_launch': 'graph'}
        l20d.validate_job(graph)
        self.assertNotIn('--launch', l20d.fused_argv(eager))
        argv = l20d.fused_argv(graph)
        self.assertEqual(argv[argv.index('--launch') + 1], 'graph')
        self.assertEqual(l20d.fused_mpi_metadata('graph'), dict(launch='graph',
            collector='mpi_graph_rank_events_v1', boundary='mpi_graph_maxrank_cudaevent',
            graph_epoch_mode='recapture_update_v1'))
        self.assertEqual(l20d.fused_mpi_metadata(), dict(launch='eager',
            collector='mpi_rank_events_v1', boundary='mpi_eager_maxrank_cudaevent'))
        for change in ({'mpi': False}, {'profile': True}, {'stage': 'fused-build'},
                       {'stage': 'gemm-probe'}, {'fused_launch': 'eager,graph'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                l20d.validate_job(graph | change)

    def test_epilogue_probe_is_explicit_profile_cta_n256_only(self):
        job = self.fused_job(profile=True, profile_detail='cta', qkv_policy='m128n256k64e32',
                             qkv_epilogue_probe=True)
        l20d.validate_job(job)
        self.assertIn('--qkv-epilogue-probe', l20d.fused_argv(job))
        self.assertNotIn('--qkv-epilogue-probe', l20d.fused_argv(self.fused_job()))
        for change in ({'stage': 'fused-build'}, {'profile': False}, {'profile_detail': 'full'},
                       {'qkv_policy': 'm128n128'}, {'mpi': True}, {'qkv_epilogue_probe': 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                l20d.validate_job(job | change)

    def test_cutlass_budget_validation_and_forwarding(self):
        matrix = {'schema': 'sm103_gemm_matrix_v1', 'shapes': [dict(id='small', m=128, n=256, k=64)]}
        job = self.fused_job(stage='gemm-probe', launches='eager', gemm_matrix_payload=matrix,
                             compare_cutlass=True)
        for budget in (116, 132, 140):
            configured = job | {'cutlass_sm_budget': budget}
            l20d.validate_job(configured)
            argv = l20d.gemm_probe_argv(configured, self.root)
            self.assertEqual(argv[argv.index('--cutlass-sm-budget') + 1], str(budget))
        for budget in (0, -1, 149, True, 1.5, 131):
            with self.assertRaises(ValueError):
                l20d.validate_job(job | {'cutlass_sm_budget': budget})
        l20d.validate_job(job | {'cutlass_sm_budget': 131, 'cutlass_counters': True})
        with self.assertRaises(ValueError):
            l20d.validate_job(job | {'cutlass_sm_budget': 132, 'compare_cutlass': False})

    def test_cutlass_probe_is_explicit_isolated_and_never_changes_default_lt(self):
        matrix = {'schema': 'sm103_gemm_matrix_v1', 'shapes': [dict(id='small', m=128, n=256, k=64)]}
        job = self.fused_job(stage='gemm-probe', launches='eager', gemm_matrix_payload=matrix)
        self.assertNotIn('--compare-cutlass-library', l20d.gemm_probe_argv(job, self.root))
        l20d.validate_job(job | {'compare_cutlass': True})
        argv = l20d.gemm_probe_argv(job | {'compare_cutlass': True}, self.root)
        self.assertEqual(argv[argv.index('--compare-cutlass-library') + 1], str(l20d.cutlass_probe_library()))
        self.assertIn('/build/sm103-cutlass/', str(l20d.cutlass_probe_library()))
        build = l20d.cutlass_probe_build_argv()[2]
        self.assertIn('--target fuse_sm103_cutlass_bf16', build)
        self.assertIn('-DFUSE_SM103_BUILD_CUTLASS_BF16=ON', build)
        self.assertNotIn('--clean-first', build)
        self.assertNotIn('--cutlass-swizzle-size', argv)
        self.assertNotIn('--cutlass-epilogue-n', argv)
        self.assertNotIn('--cutlass-1sm-cluster-m', argv)
        self.assertNotIn('--cutlass-full-check', argv)
        full = job | {'compare_cutlass': True, 'cutlass_full_check': True}
        l20d.validate_job(full)
        self.assertIn('--cutlass-full-check', l20d.gemm_probe_argv(full, self.root))
        for change in ({'compare_cutlass': False}, {'cutlass_full_check': 1},
                       {'gemm_matrix_payload': {'schema': 'sm103_gemm_matrix_v1',
                            'shapes': [dict(id='too_large', m=16384, n=16384, k=16384)]}}):
            with self.assertRaises(ValueError):
                l20d.validate_job(full | change)
        for cluster in (1, 2):
            configured = job | {'compare_cutlass': True, 'cutlass_1sm_cluster_m': cluster}
            l20d.validate_job(configured)
            args = l20d.gemm_probe_argv(configured, self.root)
            self.assertEqual(args[args.index('--cutlass-1sm-cluster-m') + 1], str(cluster))
        for configured in (job | {'cutlass_1sm_cluster_m': 1},
                           job | {'compare_cutlass': True, 'cutlass_1sm_cluster_m': True},
                           job | {'compare_cutlass': True, 'cutlass_1sm_cluster_m': 2, 'cutlass_epilogue_n': 64},
                           job | {'compare_cutlass': True, 'cutlass_1sm_cluster_m': 4}):
            with self.assertRaises(ValueError):
                l20d.validate_job(configured)
        for epilogue in (32, 64):
            configured = job | {'compare_cutlass': True, 'cutlass_epilogue_n': epilogue}
            l20d.validate_job(configured)
            args = l20d.gemm_probe_argv(configured, self.root)
            self.assertEqual(args[args.index('--cutlass-epilogue-n') + 1], str(epilogue))
        for configured in (job | {'cutlass_epilogue_n': 32},
                           job | {'compare_cutlass': True, 'cutlass_epilogue_n': True},
                           job | {'compare_cutlass': True, 'cutlass_epilogue_n': 128}):
            with self.assertRaises(ValueError):
                l20d.validate_job(configured)
        for swizzle in (1, 2, 4, 8):
            configured = job | {'compare_cutlass': True, 'cutlass_swizzle_size': swizzle}
            l20d.validate_job(configured)
            args = l20d.gemm_probe_argv(configured, self.root)
            self.assertEqual(args[args.index('--cutlass-swizzle-size') + 1], str(swizzle))
        for configured in (job | {'cutlass_swizzle_size': 1},
                           job | {'compare_cutlass': True, 'cutlass_swizzle_size': True},
                           job | {'compare_cutlass': True, 'cutlass_swizzle_size': 3}):
            with self.assertRaises(ValueError):
                l20d.validate_job(configured)
        for change in ({'stage': 'fused-smoke'}, {'launches': 'graph'}, {'launches': 'eager,graph'},
                       {'gemm_matrix_payload': None}, {'compare_cutlass': 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                l20d.validate_job(job | {'compare_cutlass': True} | change)

    def test_cutlass_probe_receipt_binds_native_inputs_node_environment_and_driver(self):
        library = self.root / 'libprobe.so'
        library.write_bytes(b'not executed')
        job = self.fused_job(files={'benchmarks/sm103/CMakeLists.txt': 'cmake',
                                   'csrc/baselines/sm103/cutlass_bf16.cu': 'native'})
        with mock.patch.object(l20d, 'cutlass_probe_library', return_value=library), \
                mock.patch.object(l20d, 'read_command', return_value='libcuda.so.1 => /driver/libcuda.so.1 (0x1234)'):
            receipt = l20d.cutlass_probe_build_receipt(job, 'environment')
            self.assertEqual(receipt['inputs'], job['files'])
            self.assertEqual(receipt['node'], '0a')
            self.assertEqual(receipt['environment_fingerprint'], 'environment')
            self.assertNotIn('0x1234', receipt['dynamic_dependencies'])
            self.assertEqual(receipt['library_sha256'], hashlib.sha256(b'not executed').hexdigest())
        with mock.patch.object(l20d, 'cutlass_probe_library', return_value=library), \
                mock.patch.object(l20d, 'read_command', return_value='libcuda.so.1 => not found'), \
                self.assertRaisesRegex(RuntimeError, 'driver linkage'):
            l20d.cutlass_probe_build_receipt(job, 'environment')

    def test_gemm_matrix_is_explicit_portable_and_does_not_change_single_probe(self):
        payload = {'schema': 'sm103_gemm_matrix_v1', 'shapes': [
            {'id': 'qkv-cp8-s131072', 'm': 16384, 'n': 18432, 'k': 16384},
            {'id': 'qkv-cp4-s65536', 'm': 16384, 'n': 18432, 'k': 16384}]}
        job = self.fused_job(stage='gemm-probe', directions='qkv,oproj', launches='eager',
                             gemm_matrix_payload=payload)
        l20d.validate_job(job)
        self.assertEqual(l20d.gemm_probe_shapes(job), payload['shapes'])
        argv = l20d.gemm_probe_argv(job, Path('/control'))
        self.assertEqual(argv[argv.index('--matrix-json') + 1], '/control/gemm-matrix.json')
        for option in ('--m', '--n', '--k'):
            self.assertNotIn(option, argv)
        for change in ({'schema': 'wrong'}, {'shapes': []}, {'shapes': payload['shapes'] * 2},
                       {'shapes': [payload['shapes'][0] | {'m': True}]},
                       {'shapes': [payload['shapes'][0] | {'k': 0}]},
                       {'shapes': [payload['shapes'][0] | {'id': '../escape'}]}):
            with self.assertRaises(ValueError):
                l20d.validate_gemm_matrix(payload | change)
        with self.assertRaises(ValueError):
            l20d.validate_job(self.fused_job(gemm_matrix='/local/path'))

    def fused_job(self, **changes):
        return dict(node='0a', stage='fused-smoke', world=8, seq_local=256,
                    comm_sm=8, devices='0,1,2,3,4,5,6,7', profile=False,
                    qkv_policy='m128n128', files={}) | changes

    def test_mpi_is_explicit_fused_only_and_uses_separate_target(self):
        for stage in l20d.FUSED_STAGES:
            job = self.fused_job(stage=stage, mpi=True)
            l20d.validate_job(job)
            self.assertEqual(l20d.fused_build_dir(job).name, 'sm103-fused-mpi')
            self.assertEqual(l20d.fused_binary(job).name, 'fused_bf16_mpi')
        build = l20d.fused_argv(self.fused_job(stage='fused-build', mpi=True))[2]
        for flag in ('-DFUSE_BUILD_MPI_BENCH=ON', '-DMPI_CXX_COMPILER=' + str(l20d.MPI_PREFIX / 'bin/mpicxx'),
                     '-DMPIEXEC_EXECUTABLE=' + str(l20d.MPI_PREFIX / 'bin/mpiexec'), '--target fused_bf16_mpi'):
            self.assertIn(flag, build)
        smoke = l20d.fused_argv(self.fused_job(mpi=True, calibrate=True, cpu_oracle=True))
        self.assertNotIn('--host-launch', smoke)
        self.assertIn('--calibrate', smoke)
        self.assertIn('--cpu-oracle', smoke)
        for changes in ({'stage': 'smoke'}, {'stage': 'gemm-probe'}, {'profile': True},
                        {'host_launch': 'per_gpu_thread'}, {'host_launch_explicit': True}, {'mpi': 'yes'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                l20d.validate_job(self.fused_job(mpi=True) | changes)

    def test_mpi_cli_rejects_even_explicit_default_host_mode_before_any_external_work(self):
        for mode in ('sequential', 'per_gpu_thread'):
            with mock.patch.object(sys, 'argv', ['l20d.py', 'run', 'fused-smoke', '--mpi', '--host-launch', mode]), \
                    mock.patch.object(l20d, 'source_package', side_effect=AssertionError('No packaging')), \
                    mock.patch.object(l20d, 'command', side_effect=AssertionError('No external command')), \
                    mock.patch.object(l20d, 'mc_copy', side_effect=AssertionError('No cloud')), \
                    mock.patch.object(l20d, 'submit', side_effect=AssertionError('No submit')), \
                    self.assertRaisesRegex(ValueError, 'host-launch'):
                l20d.main()

    def test_mpi_toolchain_receipt_hashes_private_tools_libs_and_fixed_transport(self):
        prefix = self.root / 'mpi'
        for name in ('include/mpi.h', 'bin/mpicxx', 'bin/mpiexec', 'bin/hydra_pmi_proxy', 'lib/libmpi.so.12'):
            path = prefix / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('explicit test fixture: ' + name)
            path.chmod(0o700)
        with mock.patch.object(l20d, 'MPI_PREFIX', prefix), \
                mock.patch.object(l20d, 'read_command', return_value='/workspace/gcc12/g++\n12.3.0') as query:
            first = l20d.mpi_toolchain_receipt()
            self.assertEqual(first['overrides'], {'MPICH_CXX': '/workspace/gcc12/g++', 'UCX_TLS': 'sm,self'})
            self.assertIn(str(prefix / 'lib/libmpi.so.12'), first['files'])
            (prefix / 'lib/libmpi.so.12').write_text('changed test library')
            self.assertNotEqual(first, l20d.mpi_toolchain_receipt())
            query.return_value = '/workspace/gcc/g++\n11.4.0'
            with self.assertRaisesRegex(RuntimeError, 'GCC 12'):
                l20d.mpi_toolchain_receipt()

    def test_mpi_build_receipt_rejects_node_library_and_binary_mixing(self):
        job = self.fused_job(mpi=True)
        prefix = self.root / 'mpi'
        prefix.mkdir()
        dependency = f'libmpi.so.12 => {prefix}/libmpi.so.12 (0x1234)'
        with mock.patch.object(l20d, 'REMOTE', self.root), mock.patch.object(l20d, 'MPI_PREFIX', prefix), \
                mock.patch.object(l20d, 'mpi_toolchain_receipt', return_value={'files': {'libmpi': 'a'}}) as identity, \
                mock.patch.object(l20d, 'read_command', return_value=dependency) as linkage:
            binary = l20d.fused_binary(job)
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b'explicit compiled-binary test fixture')
            binary.chmod(0o700)
            recorded = l20d.fused_build_receipt(job, 'mpi-env')
            self.assertTrue(recorded['mpi'])
            l20d.write_json(binary.parent / '.l20d-build.json', recorded)
            l20d.check_fused_build(job, 'mpi-env', self.root / 'accepted')
            for changed in (job | {'node': '09'}, job | {'mpi': False}):
                with self.assertRaises(RuntimeError):
                    l20d.check_fused_build(changed, 'mpi-env', self.root / 'rejected')
            identity.return_value = {'files': {'libmpi': 'b'}}
            with self.assertRaises(RuntimeError):
                l20d.check_fused_build(job, 'mpi-env', self.root / 'rejected')
            linkage.return_value = 'libmpi.so.12 => /system/libmpi.so.12 (0x1234)'
            with self.assertRaisesRegex(RuntimeError, 'private prefix'):
                l20d.fused_build_receipt(job, 'mpi-env')

    def test_mpi_rank_logs_are_preserved_and_merge_does_not_claim_global_order(self):
        paths = l20d.mpi_rank_logs(self.root, 2, 4)
        merged = self.root / 'attempt2.log'
        merged.write_bytes(b'launcher diagnostics\n')
        for row in paths:
            row['stdout'].write_text(f'device,rank={row["rank"]},cuda_device=0\n')
            row['stderr'].write_text(f'explicit rank {row["rank"]} stderr fixture\n')
        l20d.merge_mpi_logs(paths, merged, self.root, 2, True)
        manifest = json.loads((self.root / 'mpi-logs-attempt2.json').read_text())
        self.assertTrue(manifest['complete'])
        self.assertFalse(manifest['numeric_or_performance_accepted'])
        self.assertEqual(manifest['ordering'], 'rank_then_stream_not_global_chronological')
        content = merged.read_bytes()
        for row in manifest['ranks']:
            original = (self.root / row['path']).read_bytes()
            self.assertEqual(content[row['merged_begin']:row['merged_end']], original)
            self.assertEqual(row['sha256'], hashlib.sha256(original).hexdigest())
        paths[1]['stdout'].write_text('')
        with self.assertRaisesRegex(RuntimeError, 'missing startup/log evidence'):
            l20d.merge_mpi_logs(paths, self.root / 'incomplete.log', self.root, 3, True)

    def test_remote_mpi_uses_fork_private_env_and_keeps_per_rank_logs(self):
        job = self.fused_job(run_id='mpi-smoke', experiment='mpi-test', source_id='source',
                             timeout=300, world=4, mpi=True)
        job_path = self.root / 'mpi-job.json'
        job_path.write_text(json.dumps(job))
        control, remote = self.root / 'control', self.root / 'remote'
        remote.mkdir()
        folder = control / 'jobs/mpi-smoke'
        toolchain = {'overrides': {'MPICH_CXX': '/private/gcc12/g++', 'UCX_TLS': 'sm,self'}}
        def fake_process(*args, **kwargs):
            for row in l20d.mpi_rank_logs(folder, 1, 4):
                row['stdout'].write_text(f'device,rank={row["rank"]},cuda_device={row["rank"]}\n')
                row['stderr'].write_text('')
            return mock.Mock(wait=mock.Mock(return_value=0))
        with mock.patch.object(l20d, 'REMOTE', remote), mock.patch.object(l20d, 'CONTROL', control), \
                mock.patch.object(l20d.socket, 'gethostname', return_value=l20d.NODES['0a'][1]), \
                mock.patch.object(l20d, 'install_source'), \
                mock.patch.object(l20d, 'environment_receipt', return_value={'overrides': {}}), \
                mock.patch.object(l20d, 'mpi_toolchain_receipt', return_value=toolchain), \
                mock.patch.object(l20d, 'check_fused_build'), \
                mock.patch.object(l20d, 'check_fused_devices', return_value='GPU-a,GPU-b,GPU-c,GPU-d'), \
                mock.patch.object(l20d, 'fused_telemetry', return_value=contextlib.nullcontext()), \
                mock.patch.object(l20d, 'mc_copy'), \
                mock.patch.object(l20d.subprocess, 'Popen', side_effect=fake_process) as popen, \
                mock.patch.dict(os.environ, UCX_TLS='rc'), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(l20d.remote(job_path), 0)
            self.assertEqual(os.environ['UCX_TLS'], 'rc')
        argv = popen.call_args.args[0]
        self.assertIn('MPICH_CXX=/private/gcc12/g++', argv)
        self.assertIn('UCX_TLS=sm,self', argv)
        self.assertIn(str(l20d.MPI_PREFIX / 'bin/mpiexec'), argv)
        self.assertEqual(argv[argv.index('-launcher') + 1], 'fork')
        self.assertEqual(argv[argv.index('-n') + 1], '4')
        self.assertTrue(argv[argv.index('-outfile-pattern') + 1].endswith('rank-%r.stdout.log'))
        self.assertEqual(popen.call_args.kwargs['env']['UCX_TLS'], 'sm,self')
        state = json.loads((folder / 'status.json').read_text())
        self.assertEqual(state['state'], 'succeeded')
        self.assertTrue((folder / 'mpi-runtime-attempt1.json').is_file())
        with tarfile.open(state['artifact_local']) as archive:
            self.assertIn('control/mpi-attempt1-rank-3.stdout.log', archive.getnames())

    def test_gemm_probe_uses_per_rank_geometry_but_is_single_gpu_pure_bf16(self):
        job = self.fused_job(stage='gemm-probe', directions='qkv', launches='eager',
                             global_seq=131072, seq_local=None, hidden=16384, q_heads=128)
        l20d.validate_job(job)
        self.assertEqual(l20d.gemm_probe_geometry(job), (16384, 18432, 16384))
        self.assertEqual(l20d.gemm_probe_geometry(job | {'directions': 'oproj'}), (16384, 16384, 16384))
        argv = l20d.gemm_probe_argv(job, Path('/probe'))
        self.assertEqual(argv[argv.index('--precisions') + 1], 'bf16')
        self.assertEqual(argv[argv.index('--output') + 1], '/probe/gemm-probe.json')
        for name, expected in (('--warmup', '10'), ('--iterations', '50'),
                               ('--tune-warmup', '10'), ('--tune-iterations', '50')):
            self.assertEqual(argv[argv.index(name) + 1], expected)
        for node in l20d.NODES:
            self.assertEqual(l20d.validate_job(job | {'node': node}, l20d.NODES[node][1]), node)
        for changes in ({'directions': 'qkv,oproj'}, {'profile': True}, {'launches': ''}):
            with self.assertRaises(ValueError):
                l20d.validate_job(job | changes)

    def test_scheduler_experiments_are_single_values_and_preserve_defaults(self):
        default = self.fused_job()
        for option in ('--max-swizzle-size', '--qkv-raster', '--oproj-raster'):
            self.assertNotIn(option, l20d.fused_argv(default))
        for mpi in (False, True):
            job = default | dict(mpi=mpi, max_swizzle_size=4, qkv_raster='along_n',
                                 oproj_raster='along_m', calibrate=True)
            l20d.validate_job(job)
            self.assertEqual(l20d.fused_candidates(job), l20d.fused_candidates(default))
            argv = l20d.fused_argv(job)
            for option, expected in (('--max-swizzle-size', '4'), ('--qkv-raster', 'along_n'),
                                     ('--oproj-raster', 'along_m')):
                self.assertEqual(argv[argv.index(option) + 1], expected)
            self.assertIn('--calibrate', argv)
        for change in ({'max_swizzle_size': True}, {'max_swizzle_size': 0}, {'max_swizzle_size': 3},
                       {'max_swizzle_size': '4'}, {'qkv_raster': 'AlongM'}, {'oproj_raster': 'auto'},
                       {'stage': 'gemm-probe', 'max_swizzle_size': 4},
                       {'stage': 'fused-build', 'oproj_raster': 'along_m'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                l20d.validate_job(default | change)

    def test_scheduler_padding_is_explicit_and_profile_rejects_only_padded_cases(self):
        self.assertEqual(l20d.fused_scheduler_geometry(128, 128, 128, 8), (1, 1, 1, False))
        self.assertEqual(l20d.fused_scheduler_geometry(256, 256, 128, 8), (2, 2, 2, False))
        self.assertEqual(l20d.fused_scheduler_geometry(384, 384, 128, 8), (4, 4, 4, True))
        self.assertEqual(l20d.fused_scheduler_geometry(768, 768, 128, 8), (8, 8, 8, True))
        self.assertEqual(l20d.fused_scheduler_geometry(1024, 1024, 128, 8), (8, 8, 8, False))
        job = self.fused_job(seq_local=384, hidden=1024, q_heads=8, kv_heads=8, max_swizzle_size=4)
        l20d.validate_job(job)
        with self.assertRaisesRegex(ValueError, 'padded swizzle'):
            l20d.validate_job(job | {'profile': True})
        l20d.validate_job(job | {'seq_local': 1024, 'profile': True})

    def test_fused_explicit_candidate_lists_are_deduplicated_and_forwarded(self):
        job = self.fused_job(comm_sm_list='8,16,8',
                             qkv_policy_list='auto,m128n256,m128n128')
        self.assertEqual(l20d.fused_candidates(job), ([8, 16], ['m128n128', 'm128n256'], ['m128n128']))
        argv = l20d.fused_argv(job)
        self.assertNotIn('--comm-sm', argv)
        self.assertEqual(argv[argv.index('--comm-sm-list')+1], '8,16')
        self.assertEqual(argv[argv.index('--qkv-policy-list')+1], 'm128n128,m128n256')
        self.assertNotIn('--qkv-policy-list', l20d.fused_argv(self.fused_job()))
        both = job | {'oproj_policy_list': 'm128n256,auto,m128n128'}
        self.assertEqual(l20d.fused_candidates(both)[2], ['m128n256', 'm128n128'])
        argv = l20d.fused_argv(both)
        self.assertEqual(argv[argv.index('--oproj-policy-list')+1], 'm128n256,m128n128')

    def test_fused_candidate_lists_reject_invalid_values_and_profile_grids(self):
        for changes in ({'comm_sm_list': ''}, {'comm_sm_list': '0,16'},
                        {'comm_sm_list': '8,-1'}, {'comm_sm_list': '8,1.5'},
                        {'comm_sm_list': '8,'}, {'comm_sm_list': '1025'},
                        {'qkv_policy_list': 'auto,nope'}, {'qkv_policy_list': ''},
                        {'comm_sm_list': '8,16', 'profile': True},
                        {'qkv_policy_list': 'm128n128,m128n256', 'profile': True},
                        {'oproj_policy_list': 'm128n128,m128n256', 'profile': True},
                        {'oproj_policy_list': 'm128n160'}, {'oproj_policy_list': ''},
                        {'oproj_policy': 'm128n256_cluster_m2'},
                        {'stage': 'smoke', 'oproj_policy_list': 'm128n256'},
                        {'stage': 'smoke', 'oproj_policy': 'm128n256'},
                        {'stage': 'smoke', 'comm_sm_list': '8,16'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                l20d.validate_job(self.fused_job(**changes))
        l20d.validate_job(self.fused_job(profile=True, qkv_policy_list='auto,m128n128'))

    def test_profile_detail_is_explicit_diagnostic_mode_and_bounds_memory(self):
        for detail in ('full', 'cta'):
            job = self.fused_job(profile=True, profile_detail=detail)
            l20d.validate_job(job)
            argv = l20d.fused_argv(job)
            self.assertEqual(argv[argv.index('--profile-detail') + 1], detail)
            for changes in ({'profile': False}, {'stage': 'fused-build'}, {'stage': 'sweep'}):
                with self.assertRaisesRegex(ValueError, 'profile detail requires'):
                    l20d.validate_job(job | changes)
        with self.assertRaisesRegex(ValueError, 'Unknown profile detail'):
            l20d.validate_job(self.fused_job(profile=True, profile_detail='none'))
        job = self.fused_job(profile=True, global_seq=131072, seq_local=None, hidden=16384)
        full = l20d.fused_device_memory(job)
        cta = l20d.fused_device_memory(job | {'profile_detail': 'cta'})
        self.assertEqual(cta['profile_bytes'], 1 << 20)
        self.assertGreater(full['profile_bytes'], cta['profile_bytes'])
        self.assertEqual(full['buffer_bytes'], cta['buffer_bytes'])
        self.assertNotIn('--profile-detail', l20d.fused_argv(self.fused_job(profile=True)))

    def test_blackwell_tiles_preserve_legacy_k_and_explicit_epilogue_identity(self):
        expected = {'auto': (128, 128, 64), 'm128n256': (128, 256, 64),
                    'm128n128k128': (128, 128, 128), 'm128n256k64e32': (128, 256, 64),
                    'm128n256k128e32': (128, 256, 128), 'm128n256k64e64': (128, 256, 64)}
        for policy, shape in expected.items():
            self.assertEqual(l20d.fused_policy_tile(policy), shape)
        with self.assertRaises(ValueError):
            l20d.fused_policy_tile('m128n256k128')  # Auto epilogue does not fit two stages.
        job = self.fused_job(oproj_policy_list='m128n256,m128n256k64e32,m128n256k128e32')
        self.assertEqual(len(l20d.fused_candidates(job)[2]), 3)
        l20d.validate_job(job)
        l20d.validate_job(self.fused_job(qkv_policy='m128n256k64e64'))
        with self.assertRaises(ValueError):
            l20d.validate_job(self.fused_job(oproj_policy='m128n256k64e64'))
        # K64 remains legal for a 64-element peer shard; K128 is explicit only.
        narrow = self.fused_job(world=4, q_heads=4, kv_heads=4, head_dim=64)
        l20d.validate_job(narrow)
        with self.assertRaisesRegex(ValueError, 'selected tile K'):
            l20d.validate_job(narrow | {'oproj_policy': 'm128n128k128'})

    def test_fused_candidate_cli_does_not_silently_combine_single_and_list(self):
        for flags in (['--comm-sm', '8', '--comm-sm-list', '8,16'],
                      ['--qkv-policy', 'auto', '--qkv-policy-list', 'm128n256'],
                      ['--qkv-policy', 'm128n128', '--qkv-policy-list', 'm128n256'],
                      ['--oproj-policy', 'auto', '--oproj-policy-list', 'm128n256']):
            with mock.patch.object(sys, 'argv', ['l20d.py', 'run', 'fused-smoke', '--node', '0a', *flags]), \
                    mock.patch.object(l20d, 'source_package', side_effect=AssertionError('CLI test must not package a job')), \
                    mock.patch.object(l20d, 'command', side_effect=AssertionError('CLI test must not run external commands')), \
                    mock.patch.object(l20d, 'mc_copy', side_effect=AssertionError('CLI test must not access cloud')), \
                    mock.patch.object(l20d, 'submit', side_effect=AssertionError('CLI test must not submit a job')), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                l20d.main()
            self.assertEqual(error.exception.code, 2)

    def replay_payload(self):
        bench = l20d.baseline_executor(l20d.REPO).bench
        case = next(bench.cases(argparse.Namespace(directions='qkv', models='production_qwen_dense',
            seqs=(131072,), cps=(8,), devices='0,1,2,3,4,5,6,7')))
        entries = []
        for backend in ('cublaslt_nccl', 'te_ub'):
            config = bench.initial_configs(backend)[0]
            group = bench.group_key(case, backend, 'eager')
            entries.append(dict(group=group, case=dict(case), backend=backend, launch='eager', config=config,
                source=dict(selection_stage='sweep', status='complete', independently_remeasured=False,
                    expected_candidates=len(bench.initial_configs(backend)),
                    validated_candidates=len(bench.initial_configs(backend)), warmup=10, samples=50,
                    p50_ms=.5, p95_ms=.6, remote_raw=f'/root/experiment/sweep/{group}_{bench.digest(config)}.json',
                    raw_sha256='a'*64, rank_sha256={str(rank): 'b'*64 for rank in range(8)})))
        return dict(schema='sm103_baseline_replay_input_v1', source_node='09', source_fingerprint='1'*16,
                    source_plan_sha256='2'*64, winners_sha256='3'*64, entries=entries)

    def test_fused_validation_diagnostics_are_explicit_and_small_shape_only(self):
        for option in ('cpu_oracle', 'validation_self_test'):
            job = self.fused_job(**{option: True})
            self.assertIn('--' + option.replace('_', '-'), l20d.fused_argv(job))
            for changes in ({'stage': 'fused-build'}, {'stage': 'smoke'},
                            {'seq_local': None, 'global_seq': 131072}):
                with self.subTest(option=option, changes=changes), self.assertRaises(ValueError):
                    l20d.validate_job(job | changes)

    def test_fused_gpu_input_generator_is_explicit_and_not_a_baseline_option(self):
        self.assertNotIn('--input-generator', l20d.fused_argv(self.fused_job()))
        job = self.fused_job(input_generator='gpu_philox')
        self.assertEqual(l20d.fused_argv(job)[-2:], ['--input-generator', 'gpu_philox'])
        for changes in ({'input_generator': 'zero'}, {'stage': 'smoke'}, {'stage': 'fused-build'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                l20d.validate_job(job | changes)

    def test_fused_host_launch_mode_is_explicit_and_not_a_baseline_option(self):
        self.assertNotIn('--host-launch', l20d.fused_argv(self.fused_job()))
        job = self.fused_job(host_launch='per_gpu_thread')
        self.assertEqual(l20d.fused_argv(job)[-2:], ['--host-launch', 'per_gpu_thread'])
        for changes in ({'host_launch': 'spawn_each_sample'}, {'stage': 'smoke'},
                        {'stage': 'fused-build'}, {'stage': 'baseline-replay'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                l20d.validate_job(job | changes)
        l20d.validate_job(job | {'profile': True})

    def test_fused_calibration_is_explicit_and_separate_from_diagnostic_profiling(self):
        self.assertNotIn('--calibrate', l20d.fused_argv(self.fused_job()))
        job = self.fused_job(calibrate=True, host_launch='per_gpu_thread')
        self.assertIn('--calibrate', l20d.fused_argv(job))
        memory = l20d.fused_device_memory(job)
        self.assertEqual(memory['calibration_flag_bytes'], memory['flag_bytes'])
        self.assertEqual(l20d.fused_device_memory(job | {'calibrate': False})['calibration_flag_bytes'], 0)
        l20d.validate_job(job | {'cpu_oracle': True})
        for changes in ({'profile': True}, {'validation_self_test': True},
                        {'stage': 'fused-build'}, {'stage': 'baseline-replay'}, {'stage': 'smoke'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                l20d.validate_job(job | changes)

    def test_replay_rejects_pending_unknown_geometry_backend_launch_config_and_duplicates(self):
        payload = self.replay_payload()
        l20d.validate_replay_payload(payload, '0,1,2,3,4,5,6,7', l20d.REPO)
        mutations = [lambda entry: entry['source'].update(status='pending'),
                     lambda entry: entry['source'].update(validated_candidates=1),
                     lambda entry: entry['case'].update(hidden=1024),
                     lambda entry: entry['case'].update(model='unknown'),
                     lambda entry: entry.update(backend='unknown'),
                     lambda entry: entry.update(launch='unknown'),
                     lambda entry: entry['config'].update(channels=99),
                     lambda entry: entry['config'].update(untrusted_option=True),
                     lambda entry: entry['source'].update(rank_sha256={}),
                     lambda entry: entry['source'].update(remote_raw='/Mac/not-a-source.json')]
        for mutate in mutations:
            bad = copy.deepcopy(payload)
            mutate(bad['entries'][0])
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                l20d.validate_replay_payload(bad, '0,1,2,3,4,5,6,7', l20d.REPO)
        for bad in (payload | {'entries': []}, payload | {'entries': payload['entries']*2},
                    payload | {'source_node': '0a'}, payload | {'source_fingerprint': 'wrong'}):
            with self.assertRaises(ValueError):
                l20d.validate_replay_payload(bad, '0,1,2,3,4,5,6,7', l20d.REPO)

    def test_replay_allows_both_nodes_without_old_result_import_or_silent_other_stage_options(self):
        job = dict(stage='baseline-replay', node='0a', winners='/local/verified/winners.json')
        for node in l20d.NODES:
            self.assertEqual(l20d.validate_job(job | {'node': node}, l20d.NODES[node][1]), node)
        for changes in ({'winners': None}, {'reuse_experiment': 'node1'},
                        {'profile': True}, {'executor': 'serial'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                l20d.validate_job(job | changes)
        with self.assertRaisesRegex(ValueError, 'only valid'):
            l20d.validate_job(job | {'stage': 'smoke'})
        l20d.validate_job(job | {'oproj_layout': 'causal_dual_chunk_v1'})
        for changes in ({'oproj_layout': 'unknown'},
                        {'stage': 'fused-smoke', 'oproj_layout': 'causal_dual_chunk_v1'}):
            with self.assertRaises(ValueError):
                l20d.validate_job(job | changes)

    def test_replay_plan_has_two_frozen_configs_new_provenance_and_stable_gemm_cache(self):
        payload = self.replay_payload()
        bench = l20d.baseline_executor(l20d.REPO).bench
        remote = self.root / 'remote'
        library = remote / 'build/sm103/libfuse_cublaslt_runner.so'
        library.parent.mkdir(parents=True)
        library.write_bytes(b'node2 library')
        original_make_job = bench.make_job

        def make_job(*args):
            with mock.patch.object(bench, 'ROOT', remote), \
                    mock.patch.object(bench, 'HERE', remote / 'benchmarks/sm103'):
                return original_make_job(*args)

        job = dict(node='0a', stage='baseline-replay', devices='0,1,2,3,4,5,6,7',
                   baseline_replay=payload, source_id='source-a', job_timeout=600)
        with mock.patch.object(l20d, 'REMOTE', remote), \
                mock.patch.object(l20d, 'baseline_executor', return_value=argparse.Namespace(bench=bench)), \
                mock.patch.object(bench, 'fingerprint', return_value='measurement-id'), \
                mock.patch.object(bench, 'make_job', side_effect=make_job), \
                mock.patch.object(l20d, 'read_command', return_value='\n'.join(f'{rank}, GPU-{rank}' for rank in range(8))):
            plans = []
            for label, source, environment, node in (
                    ('a', 'source-a', 'env-a', '0a'), ('b', 'source-b', 'env-a', '0a'),
                    ('c', 'source-b', 'env-b', '0a'), ('d', 'source-a', 'env-a', '09')):
                results = remote / 'results/sm103' / label
                argv = l20d.baseline_replay_plan(job | {'source_id': source, 'node': node},
                                               environment, results, self.root / label)
                self.assertEqual(argv[-2:], ['--job-timeout', '600'])
                self.assertIn('--plan', argv)
                self.assertNotIn('--import-plan', argv)
                self.assertNotIn('--stage', argv)
                plans.append(json.loads((results / 'replay_plan.json').read_text()))
                self.assertEqual(plans[-1]['node'], node)
                for name in ('baseline-devices.json', 'baseline-replay.json'):
                    self.assertEqual(json.loads((self.root / label / name).read_text())['node'], node)
        a, b, c, d = plans
        self.assertEqual(len(a['jobs']), 2)
        self.assertFalse(a['imports_source_measurements'])
        self.assertFalse(a['communication_search'])
        self.assertNotEqual(a['fingerprint'], b['fingerprint'])
        self.assertNotEqual(b['fingerprint'], c['fingerprint'])
        self.assertNotEqual(a['fingerprint'], d['fingerprint'])
        cache = lambda plan: plan['jobs'][0]['env']['FUSE_CUBLASLT_CACHE_DIR']
        self.assertEqual(cache(a), cache(b))
        self.assertNotEqual(cache(b), cache(c))
        self.assertNotEqual(cache(a), cache(d))
        self.assertEqual(d['source_winners']['source_node'], '09')
        self.assertFalse(d['imports_source_measurements'])
        for actual, supplied in zip(a['jobs'], payload['entries']):
            self.assertEqual(actual['config'], supplied['config'])
            self.assertEqual(actual['case'], supplied['case'])
            self.assertEqual(actual['env']['CUDA_VISIBLE_DEVICES'], ','.join(f'GPU-{rank}' for rank in range(8)))
            self.assertTrue(actual['output'].startswith(str(remote / 'results/sm103/a/baseline-replay')))
            self.assertNotIn('/Users/', json.dumps(actual))

    def test_replay_cli_embeds_payload_not_mac_path_or_old_measurements(self):
        payload = self.replay_payload()
        local = self.root / 'local'

        def package(folder):
            archive = folder / 'source.tar.gz'
            archive.write_bytes(b'new source')
            return archive, 'new-source', {}

        with mock.patch.object(sys, 'argv', ['l20d.py', 'run', 'baseline-replay', '--node', '0a',
                '--winners', '/Users/admin/verified/winners.json', '--dry-run']), \
                mock.patch.object(l20d, 'LOCAL', local), mock.patch.object(l20d, 'source_package', side_effect=package), \
                mock.patch.object(l20d, 'load_replay_winners', return_value=payload) as load, \
                mock.patch.object(l20d, 'read_command', return_value='head'), \
                mock.patch.object(l20d, 'submit') as submit, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(l20d.main(), 0)
        job = json.loads(next(local.glob('*/job.json')).read_text())
        self.assertNotIn('winners', job)
        self.assertEqual(job['baseline_replay'], payload)
        self.assertEqual(len(job['baseline_replay']['entries']), 2)
        load.assert_called_once_with('/Users/admin/verified/winners.json', '0,1,2,3,4,5,6,7')
        submit.assert_not_called()

    def test_remote_replay_uses_locked_runner_batch_plan_and_normal_fetch(self):
        payload = self.replay_payload()
        executor = l20d.baseline_executor(l20d.REPO)
        remote = self.root / 'remote-replay'
        remote.mkdir()
        control = self.root / 'control-replay'
        job = dict(node='0a', stage='baseline-replay', devices='0,1,2,3,4,5,6,7',
            baseline_replay=payload, source_id='node2-source', run_id='replay-test',
            experiment='fixed-winners', timeout=300, job_timeout=600)
        job_path = self.root / 'replay-job.json'
        job_path.write_text(json.dumps(job))
        proc = mock.Mock()
        proc.wait.return_value = 0

        def plan(_job, environment, results, folder):
            path = results / 'replay_plan.json'
            l20d.write_json(path, dict(environment_fingerprint=environment, source_winners=payload, jobs=[]))
            l20d.write_json(folder / 'baseline-replay.json', {'plan': str(path)})
            return [l20d.PYTHON, '-u', str(remote / 'scripts/sm103_batch.py'), '--plan', str(path)]

        with mock.patch.object(l20d, 'REMOTE', remote), mock.patch.object(l20d, 'CONTROL', control), \
                mock.patch.object(l20d, 'baseline_executor', return_value=executor), \
                mock.patch.object(l20d.socket, 'gethostname', return_value=l20d.NODES['0a'][1]), \
                mock.patch.object(l20d, 'install_source') as install, \
                mock.patch.object(l20d, 'environment_receipt', return_value={'host': 'node2', 'overrides': {}}), \
                mock.patch.object(l20d, 'baseline_replay_plan', side_effect=plan) as replay, \
                mock.patch.object(l20d, 'fused_telemetry') as fusion_monitor, \
                mock.patch.object(l20d, 'mc_copy'), \
                mock.patch.object(l20d.subprocess, 'Popen', return_value=proc) as popen, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(l20d.remote(job_path), 0)
        install.assert_called_once()
        replay.assert_called_once()
        fusion_monitor.assert_not_called()
        self.assertIn('--plan', popen.call_args.args[0])
        self.assertNotIn('--import-plan', popen.call_args.args[0])
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        folder = control / 'jobs/replay-test'
        state = json.loads((folder / 'status.json').read_text())
        self.assertEqual((state['node'], state['state']), ('0a', 'succeeded'))
        archive = folder / 'artifacts-attempt1.tar.gz'
        with mock.patch.object(l20d, 'LOCAL', self.root / 'fetched-replay'), \
                mock.patch.object(l20d, 'get_status', return_value=state), \
                mock.patch.object(l20d, 'mc_copy', side_effect=lambda _src, dst: shutil.copy2(archive, dst)), \
                contextlib.redirect_stdout(io.StringIO()):
            l20d.fetch('replay-test')
        fetched = self.root / 'fetched-replay/replay-test/artifacts-attempt1'
        self.assertTrue((fetched / 'results/replay_plan.json').is_file())
        self.assertTrue((fetched / 'control/baseline-replay.json').is_file())

    def test_fused_telemetry_uses_selected_physical_ids_not_uuid_mask(self):
        monitor = mock.Mock()
        executor = argparse.Namespace(telemetry=monitor)
        with mock.patch.object(l20d, 'baseline_executor', return_value=executor):
            l20d.fused_telemetry(self.fused_job(world=4, devices='7,5,3,1,0,2,4,6'), self.root)
        monitor.assert_called_once_with('7,5,3,1', self.root / 'gpu-telemetry.csv')

    def test_fused_telemetry_failure_is_failed_and_csv_is_archived(self):
        for phase in ('enter', 'exit'):
            with self.subTest(phase=phase):
                control = self.root / ('control-' + phase)
                remote = self.root / ('remote-' + phase)
                remote.mkdir()
                job = self.fused_job(run_id='telemetry-' + phase, experiment='test', source_id='source-id', timeout=300)
                path = self.root / ('job-' + phase + '.json')
                path.write_text(json.dumps(job))
                proc = mock.Mock()
                proc.wait.return_value = 0

                @contextlib.contextmanager
                def fail_telemetry(_job, folder):
                    (folder / 'gpu-telemetry.csv').write_text('timestamp,index,power.draw\n')
                    if phase == 'enter':
                        raise RuntimeError('telemetry unavailable')
                    yield
                    raise RuntimeError('telemetry unavailable')

                with mock.patch.object(l20d, 'REMOTE', remote), mock.patch.object(l20d, 'CONTROL', control), \
                        mock.patch.object(l20d.socket, 'gethostname', return_value=l20d.NODES['0a'][1]), \
                        mock.patch.object(l20d, 'install_source'), \
                        mock.patch.object(l20d, 'environment_receipt', return_value={'overrides': {}}), \
                        mock.patch.object(l20d, 'check_fused_build'), \
                        mock.patch.object(l20d, 'check_fused_devices', return_value='GPU-0'), \
                        mock.patch.object(l20d, 'fused_telemetry', side_effect=fail_telemetry), \
                        mock.patch.object(l20d, 'mc_copy'), \
                        mock.patch.object(l20d.subprocess, 'Popen', return_value=proc) as popen, \
                        contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(l20d.remote(path), 1)
                self.assertEqual(popen.call_count, 0 if phase == 'enter' else 1)
                if phase == 'exit':
                    proc.wait.assert_called_once_with(timeout=30)
                folder = control / 'jobs' / job['run_id']
                state = json.loads((folder / 'status.json').read_text())
                self.assertEqual(state['state'], 'failed')
                self.assertIn('telemetry unavailable', state['error'])
                with tarfile.open(folder / 'artifacts-attempt1.tar.gz') as archive:
                    self.assertIn('control/gpu-telemetry.csv', archive.getnames())

    def test_node_defaults_preserve_old_jobs_and_map_is_immutable(self):
        self.assertEqual(l20d.validate_job({'stage': 'sweep'}, 'l20d-xerkjfcp-0001'), '09')
        self.assertEqual(l20d.validate_job(self.fused_job(), 'l20d-ebed3kz6-0000'), '0a')
        with self.assertRaises(TypeError):
            l20d.NODES['0a'] = ('wrong-screen', 'wrong-host')
        with self.assertRaisesRegex(ValueError, 'Unknown node'):
            l20d.validate_job({'stage': 'build', 'node': 'other'})
        with self.assertRaisesRegex(RuntimeError, 'Wrong host'):
            l20d.validate_job(self.fused_job(), 'l20d-xerkjfcp-0001')

    def test_exact_screen_resolution_accepts_macos_rc1_and_avoids_prefix_collision(self):
        listing = ('There are screens on:\n\t87512.L20D_screen2\t(Attached)\n'
                   '\t56103.L20D_screen\t(Attached)\n\t64286.L20D_aux\t(Detached)\n')
        with mock.patch.object(l20d.subprocess, 'run', return_value=
                               subprocess.CompletedProcess(['screen', '-ls'], 1, listing, '')):
            self.assertEqual(l20d.resolve_screen('09'), '56103.L20D_screen')
            self.assertEqual(l20d.resolve_screen('0a'), '87512.L20D_screen2')
        for listing in ('87512.L20D_screen2\t(Attached)\n',
                        '11.L20D_screen\t(Attached)\n22.L20D_screen\t(Detached)\n'):
            with self.subTest(listing=listing), mock.patch.object(l20d.subprocess, 'run', return_value=
                    subprocess.CompletedProcess(['screen', '-ls'], 0, listing, '')):
                with self.assertRaisesRegex(RuntimeError, 'Expected exactly one'):
                    l20d.resolve_screen('09')

    def test_screen_readiness_checks_selected_hostname_and_last_line(self):
        with mock.patch.object(l20d, 'resolve_screen', return_value='87512.L20D_screen2'), \
                mock.patch.object(l20d, 'screen_snapshot',
                                  return_value='root@l20d-ebed3kz6-0000 ~/workspace_wct#\n') as snapshot:
            self.assertEqual(l20d.screen_ready('0a'), '87512.L20D_screen2')
            snapshot.assert_called_once_with('87512.L20D_screen2')
        for last_line in ('root@l20d-xerkjfcp-0001 ~/workspace_wct#', 'BATCH 1/380', '验证码:'):
            with mock.patch.object(l20d, 'resolve_screen', return_value='87512.L20D_screen2'), \
                    mock.patch.object(l20d, 'screen_snapshot', return_value=last_line):
                with self.assertRaisesRegex(RuntimeError, 'no command was sent'):
                    l20d.screen_ready('0a')

    def test_screen_snapshot_waits_for_one_async_hardcopy_without_resending(self):
        with mock.patch.object(l20d, 'command') as command, \
                mock.patch.object(Path, 'read_text', side_effect=[FileNotFoundError(), '', 'root@host ~#\n']), \
                mock.patch.object(l20d.time, 'sleep'):
            self.assertEqual(l20d.screen_snapshot('56103.L20D_screen'), 'root@host ~#\n')
        command.assert_called_once()
        self.assertEqual(command.call_args[0][0][1:3], ['-S', '56103.L20D_screen'])
        with mock.patch.object(l20d, 'command') as command, \
                mock.patch.object(Path, 'read_text', side_effect=FileNotFoundError()), \
                mock.patch.object(l20d.time, 'monotonic', side_effect=[0, 3]):
            with self.assertRaisesRegex(RuntimeError, 'no terminal command was sent'):
                l20d.screen_snapshot('56103.L20D_screen')
        command.assert_called_once()

    def test_submit_targets_one_full_screen_id_and_checks_node_hostname(self):
        job = self.fused_job(run_id='node2-submit')
        with mock.patch.object(l20d, 'screen_ready', return_value='87512.L20D_screen2') as ready, \
                mock.patch.object(l20d, 'command') as command, contextlib.redirect_stdout(io.StringIO()):
            l20d.submit(job)
        ready.assert_called_once_with('0a')
        for call in command.call_args_list:
            self.assertEqual(call.args[0][1:3], ['-S', '87512.L20D_screen2'])
        injected = command.call_args_list[-1].args[0][-1]
        self.assertIn('test "$(hostname)" = l20d-ebed3kz6-0000', injected)
        self.assertNotIn('l20d-xerkjfcp-0001', injected)

    def routed_job(self, node, stage):
        extra = ({'winners': '/local/verified/winners.json'} if stage == 'baseline-replay' else
                 {'directions': 'qkv', 'launches': 'eager'} if stage == 'gemm-probe' else {})
        return self.fused_job(node=node, stage=stage, run_id=f'{node}-{stage}',
                              experiment=f'{node}-{stage}', source_id='same-source', **extra)

    def test_routed_stages_target_only_the_selected_screen_and_hostname(self):
        for node, (screen, hostname) in l20d.NODES.items():
            for stage in (*l20d.FUSED_STAGES, 'baseline-replay', 'gemm-probe'):
                job = self.routed_job(node, stage)
                screen_id = '12345.' + screen
                with self.subTest(node=node, stage=stage), \
                        mock.patch.object(l20d, 'screen_ready', return_value=screen_id) as ready, \
                        mock.patch.object(l20d, 'command') as command, \
                        mock.patch.object(l20d, 'mc_copy', side_effect=AssertionError('No cloud access')), \
                        contextlib.redirect_stdout(io.StringIO()):
                    l20d.submit(job)
                ready.assert_called_once_with(node)
                self.assertEqual(len(command.call_args_list), 2)
                for call in command.call_args_list:
                    self.assertEqual(call.args[0][1:3], ['-S', screen_id])
                injected = command.call_args_list[-1].args[0][-1]
                self.assertIn(f'test "$(hostname)" = {hostname}', injected)
                other = '0a' if node == '09' else '09'
                self.assertNotIn(l20d.NODES[other][1], injected)

    def test_routed_stages_reject_wrong_host_before_workspace_mutation(self):
        for node in l20d.NODES:
            other = '0a' if node == '09' else '09'
            for stage in (*l20d.FUSED_STAGES, 'baseline-replay', 'gemm-probe'):
                path = self.root / 'job.json'
                path.write_text(json.dumps(self.routed_job(node, stage)))
                control = self.root / f'control-{node}-{stage}'
                with self.subTest(node=node, stage=stage), \
                        mock.patch.object(l20d, 'CONTROL', control), \
                        mock.patch.object(l20d.socket, 'gethostname', return_value=l20d.NODES[other][1]), \
                        mock.patch.object(l20d, 'command', side_effect=AssertionError('No external command')), \
                        mock.patch.object(l20d, 'mc_copy', side_effect=AssertionError('No cloud access')), \
                        mock.patch.object(l20d, 'install_source') as install, \
                        self.assertRaisesRegex(RuntimeError, 'Wrong host'):
                    l20d.remote(path)
                install.assert_not_called()
                self.assertFalse(control.exists())

    def test_routed_stages_still_obey_the_selected_workspace_lock(self):
        import fcntl
        for node in l20d.NODES:
            control = self.root / f'locked-{node}'
            control.mkdir()
            with (control / 'workspace.lock').open('a+') as owner:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                for stage in (*l20d.FUSED_STAGES, 'baseline-replay', 'gemm-probe'):
                    job = self.routed_job(node, stage)
                    path = self.root / 'job.json'
                    path.write_text(json.dumps(job))
                    with self.subTest(node=node, stage=stage), \
                            mock.patch.object(l20d, 'CONTROL', control), \
                            mock.patch.object(l20d.socket, 'gethostname', return_value=l20d.NODES[node][1]), \
                            mock.patch.object(l20d, 'command', side_effect=AssertionError('No external command')), \
                            mock.patch.object(l20d, 'mc_copy'), \
                            mock.patch.object(l20d, 'install_source') as install, \
                            mock.patch.object(l20d.subprocess, 'Popen') as popen, \
                            contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(l20d.remote(path), 75)
                    install.assert_not_called()
                    popen.assert_not_called()
                    state = json.loads((control / 'jobs' / job['run_id'] / 'status.json').read_text())
                    self.assertEqual((state['node'], state['state'], state['phase']), (node, 'failed', 'lock'))

    def test_cli_default_node_remains_09_without_packaging_or_external_calls(self):
        for stage in ('sweep', *l20d.FUSED_STAGES, 'baseline-replay', 'gemm-probe'):
            with self.subTest(stage=stage), \
                    mock.patch.object(sys, 'argv', ['l20d.py', 'run', stage]), \
                    mock.patch.object(l20d, 'validate_job', side_effect=RuntimeError('Captured CLI')) as validate, \
                    mock.patch.object(l20d, 'source_package', side_effect=AssertionError('No packaging')), \
                    mock.patch.object(l20d, 'command', side_effect=AssertionError('No external command')), \
                    mock.patch.object(l20d, 'mc_copy', side_effect=AssertionError('No cloud access')), \
                    mock.patch.object(l20d, 'submit', side_effect=AssertionError('No submission')), \
                    self.assertRaisesRegex(RuntimeError, 'Captured CLI'):
                l20d.main()
            self.assertEqual(validate.call_args.args[0]['node'], '09')

    def test_fused_build_flags_and_profile_directory_are_independent(self):
        for profile in (False, True):
            job = self.fused_job(stage='fused-build', profile=profile)
            argv = l20d.fused_argv(job)
            self.assertEqual(argv[:2], ['bash', '-c'])
            self.assertIn('-DFUSE_ARCH=sm103', argv[2])
            self.assertIn('-DFUSE_BUILD_KERNELS=ON', argv[2])
            self.assertIn('-DFUSE_BUILD_BASELINES=OFF', argv[2])
            self.assertIn('-DCUTLASS_ROOT=' + l20d.CUTLASS, argv[2])
            self.assertIn('-DFUSE_ENABLE_PROFILING=' + ('ON' if profile else 'OFF'), argv[2])
            self.assertEqual(l20d.fused_build_dir(job).name,
                             'sm103-fused-profile' if profile else 'sm103-fused')

    def test_fused_smoke_argv_preserves_geometry_and_instrumentation(self):
        job = self.fused_job(world=4, seq_local=258, comm_sm=12, causal=True, profile=True)
        argv = l20d.fused_argv(job)
        self.assertEqual(argv[1:], ['--world', '4', '--comm-sm', '12', '--seq-local', '258',
                                   '--hidden', '1024', '--q-heads', '32', '--kv-heads', '8',
                                   '--head-dim', '128', '--timeout-seconds', '60',
                                   '--causal', '--profile'])
        self.assertTrue(argv[0].endswith('/sm103-fused-profile/fused_bf16'))
        self.assertEqual(l20d.fused_devices(job), ['0', '1', '2', '3'])
        for changes in ({'world': 2}, {'devices': '0,1,2,2'}, {'devices': '0,1,2,8'},
                        {'comm_sm': 0}, {'seq_local': 4097}, {'seq_local': 257},
                        {'qkv_policy': 'wave_time_model'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                l20d.validate_job(job | changes)

    def test_fused_geometry_preserves_defaults_and_accepts_long_shapes(self):
        old_job = self.fused_job()
        defaults = l20d.fused_geometry(old_job)
        self.assertEqual((defaults['world'], defaults['seq_local'], defaults['global_seq'],
                          defaults['hidden'], defaults['q_heads'], defaults['kv_heads'],
                          defaults['head_dim'], defaults['timeout_seconds']),
                         (8, 256, 2048, 1024, 32, 8, 128, 60))
        del old_job['seq_local']
        self.assertEqual(l20d.fused_geometry(old_job), defaults)
        self.assertEqual(l20d.fused_geometry(old_job | {'seq_local': None}), defaults)
        for seq, hidden, heads, expected in ((65536, 2048, 16, (8192, 2048, 4096)),
                                             (131072, 4096, 32, (16384, 4096, 6144))):
            job = self.fused_job(seq_local=None, global_seq=seq, hidden=hidden, q_heads=heads,
                                 causal=True, timeout_seconds=300)
            before = dict(job)
            l20d.validate_job(job)
            shape = l20d.fused_geometry(job)
            self.assertEqual((shape['seq_local'], shape['q_width'], shape['projection_width']), expected)
            self.assertEqual(job, before)
        l20d.validate_job(self.fused_job(seq_local=8192))

    def test_fused_shape_contract_rejects_conflicts_alignment_and_integer_limits(self):
        for changes in ({'global_seq': 65536}, {'global_seq': 65537, 'seq_local': None},
                        {'global_seq': 0, 'seq_local': None}, {'seq_local': 0},
                        {'seq_local': 1 << 31}, {'global_seq': (1 << 31), 'seq_local': None},
                        {'seq_local': (1 << 31) - 1}, {'hidden': 7}, {'head_dim': 7},
                        {'q_heads': 17}, {'kv_heads': 4}, {'head_dim': 0},
                        {'q_heads': 8, 'head_dim': 8}, {'timeout_seconds': 0},
                        {'timeout_seconds': 1 << 31}, {'hidden': True},
                        {'q_heads': (1 << 31) - 1, 'kv_heads': (1 << 31) - 1,
                         'head_dim': (1 << 31) - 1},
                        {'global_seq': 2147483640, 'seq_local': None, 'hidden': 2147483640},
                        {'global_seq': 24, 'seq_local': None, 'causal': True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                l20d.validate_job(self.fused_job(**changes))

    def test_fused_long_argv_keeps_global_sequence_and_watchdog_distinct(self):
        job = self.fused_job(seq_local=None, global_seq=65536, hidden=2048, q_heads=16,
                             kv_heads=8, head_dim=128, causal=True, timeout_seconds=300,
                             timeout=900)
        argv = l20d.fused_argv(job)
        self.assertEqual(argv[1:], ['--world', '8', '--comm-sm', '8', '--global-seq', '65536',
                                   '--hidden', '2048', '--q-heads', '16', '--kv-heads', '8',
                                   '--head-dim', '128', '--timeout-seconds', '300', '--causal'])
        self.assertNotIn('--seq-local', argv)
        self.assertEqual(job['timeout'], 900)

    def test_fused_cli_resolves_omitted_local_sequence_without_masking_conflict(self):
        base = ['l20d.py', 'run', 'fused-smoke', '--node', '0a']
        for extra, expected in (([], (256, 2048)),
                                (['--world', '8', '--global-seq', '65536', '--hidden', '2048',
                                  '--q-heads', '16', '--timeout-seconds', '300', '--timeout', '900'],
                                 (8192, 65536))):
            with mock.patch.object(sys, 'argv', base + extra), \
                    mock.patch.object(l20d, 'validate_job', side_effect=RuntimeError('parsed')) as validate:
                with self.assertRaisesRegex(RuntimeError, 'parsed'):
                    l20d.main()
            job = validate.call_args.args[0]
            shape = l20d.fused_geometry(job)
            self.assertEqual((shape['seq_local'], shape['global_seq']), expected)
            if extra:
                self.assertEqual((job['timeout_seconds'], job['timeout']), (300, 900))
        with mock.patch.object(sys, 'argv', base + ['--global-seq', '65536', '--seq-local', '8192']), \
                mock.patch.object(l20d, 'source_package') as package, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                l20d.main()
            self.assertEqual(error.exception.code, 2)
            package.assert_not_called()

    def test_fused_device_estimate_counts_both_boundaries_and_references(self):
        job = self.fused_job(seq_local=None, global_seq=65536, hidden=2048, q_heads=16)
        memory = l20d.fused_device_memory(job)
        m, h, p, q = 8192, 2048, 4096, 2048
        expected_buffers = 2 * (3 * m * (h + p + q) + h * (p + q))
        self.assertEqual(memory['buffer_bytes'], expected_buffers)
        self.assertEqual(expected_buffers, 408 << 20)
        self.assertEqual(memory['flag_bytes'], 4 * 32 * (64 * 64 + 64 * 8 + 8 + 1))
        self.assertEqual(memory['profile_bytes'], 0)
        self.assertEqual(memory['headroom_bytes'], 512 << 20)
        self.assertEqual(memory['minimum_free_bytes'], 2 << 30)
        self.assertFalse(memory['guarantees_fit'])
        self.assertIn('host reference RAM', memory['note'])

    def test_fused_device_estimate_scales_with_shape_and_profiling(self):
        job = self.fused_job(seq_local=None, global_seq=524288, hidden=4096, q_heads=32)
        memory = l20d.fused_device_memory(job)
        instrumented = l20d.fused_device_memory(job | {'profile': True})
        self.assertGreater(memory['minimum_free_bytes'], 4 << 30)
        self.assertGreater(instrumented['minimum_free_bytes'], memory['minimum_free_bytes'])
        self.assertEqual(instrumented['buffer_bytes'], memory['buffer_bytes'])
        self.assertEqual(instrumented['profile_bytes'], 512 * 32 * 256 + (1 << 20))
        self.assertEqual(memory['headroom_bytes'],
                         (memory['buffer_bytes'] + memory['flag_bytes'] + 9) // 10)

    def test_fused_build_receipt_ignores_controller_docs_but_rejects_stale_binary(self):
        job = self.fused_job(files={'CMakeLists.txt': 'cmake', 'include/fuse/kernel.h': 'header',
                                  'csrc/operators/sm103/entry.cu': 'entry', 'scripts/l20d.py': 'old'})
        same_build = job | {'files': job['files'] | {'scripts/l20d.py': 'new', 'docs/notes.md': 'new',
                                                   'csrc/operators/sm103/README.md': 'new'}}
        self.assertEqual(l20d.fused_build_inputs(job), l20d.fused_build_inputs(same_build))
        validation_header = 'benchmarks/sm103/fused_validation.cuh'
        with_validation = job | {'files': job['files'] | {validation_header: 'gpu-validation-v1'}}
        changed_validation = job | {'files': job['files'] | {validation_header: 'gpu-validation-v2'}}
        self.assertNotEqual(l20d.fused_build_inputs(job), l20d.fused_build_inputs(with_validation))
        self.assertNotEqual(l20d.fused_build_inputs(with_validation),
                            l20d.fused_build_inputs(changed_validation))
        inputs_header = 'benchmarks/sm103/fused_inputs.cuh'
        with_inputs = job | {'files': job['files'] | {inputs_header: 'gpu-input-v1'}}
        changed_inputs = job | {'files': job['files'] | {inputs_header: 'gpu-input-v2'}}
        self.assertNotEqual(l20d.fused_build_inputs(with_inputs), l20d.fused_build_inputs(changed_inputs))
        host_header = 'benchmarks/sm103/fused_host_launch.h'
        with_host = job | {'files': job['files'] | {host_header: 'host-launch-v1'}}
        changed_host = job | {'files': job['files'] | {host_header: 'host-launch-v2'}}
        self.assertNotEqual(l20d.fused_build_inputs(with_host), l20d.fused_build_inputs(changed_host))
        with mock.patch.object(l20d, 'REMOTE', self.root), \
                mock.patch.object(l20d, 'read_command',
                    return_value='libcuda.so.1 => /driver/libcuda.so.1 (0x1234)') as linkage:
            build = l20d.fused_build_dir(job)
            build.mkdir(parents=True)
            binary = build / 'fused_bf16'
            binary.write_bytes(b'compiled executable')
            binary.chmod(0o700)
            receipt = l20d.fused_build_receipt(job, 'env-id')
            self.assertEqual(receipt['dynamic_dependencies'], 'libcuda.so.1 => /driver/libcuda.so.1')
            self.assertEqual(linkage.call_args.args[0][-1], str(binary))
            self.assertIn('source /root/workspace_wct/env.sh', linkage.call_args.args[0][2])
            l20d.write_json(build / '.l20d-build.json', receipt)
            linkage.return_value = 'libcuda.so.1 => /driver/libcuda.so.1 (0xabcd)'
            l20d.check_fused_build(same_build, 'env-id', self.root / 'job')
            self.assertEqual(json.loads((self.root / 'job/fused-build.json').read_text()), receipt)
            # Even identical paths, inputs, environment, dependencies and bytes
            # cannot turn a copied node0a build receipt into node09 validation.
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                l20d.check_fused_build(same_build | {'node': '09'}, 'env-id', self.root / 'other-node')
            self.assertFalse((self.root / 'other-node/fused-build.json').exists())
            linkage.return_value = 'libcuda.so.1 => /different-driver/libcuda.so.1 (0xabcd)'
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                l20d.check_fused_build(same_build, 'env-id', self.root / 'job')
            linkage.return_value = 'libcuda.so.1 => /driver/libcuda.so.1 (0x1234)'
            binary.write_bytes(b'different executable')
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                l20d.check_fused_build(job, 'env-id', self.root / 'job')

    def gpu_csv(self, devices, *, utilization=0, free=4096):
        return '\n'.join(f'{device}, GPU-uuid{device}, {utilization}, {free}, 12345, 1980, 9000, 80'
                         for device in devices)

    def test_gpu_guard_checks_only_selected_physical_devices_without_pid_gate(self):
        job = self.fused_job(world=4, devices='7,5,3,1,0,2,4,6')
        # nvidia-smi may emit index-sorted rows, not selection order.
        raw = self.gpu_csv(['1', '3', '5', '7'])
        with mock.patch.object(l20d, 'read_command', return_value=raw) as read, \
                mock.patch.object(l20d.time, 'sleep'):
            visible = l20d.check_fused_devices(job, self.root / 'job')
        self.assertEqual(visible, 'GPU-uuid7,GPU-uuid5,GPU-uuid3,GPU-uuid1')
        self.assertEqual(read.call_count, 3)
        for call in read.call_args_list:
            self.assertIn('--id=7,5,3,1', call.args[0])
            self.assertNotIn('compute-apps', ' '.join(call.args[0]))
        receipt = json.loads((self.root / 'job/gpu-before.json').read_text())
        self.assertEqual(receipt['node'], '0a')
        self.assertEqual(len(receipt['observations']), 3)
        self.assertEqual(receipt['memory_estimate']['minimum_free_bytes'], 2 << 30)

    def test_gpu_guard_rejects_long_shape_below_estimated_need_and_records_estimate(self):
        job = self.fused_job(seq_local=None, global_seq=524288, hidden=4096, q_heads=32)
        memory = l20d.fused_device_memory(job)
        required_mib = (memory['minimum_free_bytes'] + (1 << 20) - 1) // (1 << 20)
        for free, accepted in ((4096, False), (required_mib, True)):
            with self.subTest(free=free), \
                    mock.patch.object(l20d, 'read_command', return_value=self.gpu_csv(
                        list('01234567'), free=free)), mock.patch.object(l20d.time, 'sleep'), \
                    mock.patch.object(l20d.os, 'killpg') as kill:
                if accepted:
                    self.assertEqual(l20d.check_fused_devices(job, self.root / 'long-check'),
                                     ','.join('GPU-uuid' + str(rank) for rank in range(8)))
                else:
                    with self.assertRaisesRegex(RuntimeError, 'estimated requirement'):
                        l20d.check_fused_devices(job, self.root / 'long-check')
                kill.assert_not_called()
                receipt = json.loads((self.root / 'long-check/gpu-before.json').read_text())
                self.assertEqual(receipt['memory_estimate'], memory)

    def test_gpu_guard_retains_failure_snapshot_and_never_kills(self):
        for utilization, free, error in ((25, 4096, 'is computing'), (0, 1000, '2 GiB')):
            with self.subTest(utilization=utilization, free=free), \
                    mock.patch.object(l20d, 'read_command', return_value=self.gpu_csv(
                        ['0', '1', '2', '3'], utilization=utilization, free=free)), \
                    mock.patch.object(l20d.os, 'killpg') as kill, mock.patch.object(l20d.time, 'sleep'):
                with self.assertRaisesRegex(RuntimeError, error):
                    l20d.check_fused_devices(self.fused_job(world=4), self.root / 'failed-check')
                kill.assert_not_called()
                self.assertTrue((self.root / 'failed-check/gpu-before.json').exists())

    def test_gpu_guard_waits_for_three_consecutive_idle_samples(self):
        idle = self.gpu_csv(list('0123'))
        busy = self.gpu_csv(list('0123'), utilization=60)
        with mock.patch.object(l20d, 'read_command', side_effect=[idle, busy, idle, busy, idle, idle, idle]) as read, \
                mock.patch.object(l20d.time, 'sleep'), mock.patch.object(l20d.os, 'killpg') as kill:
            self.assertEqual(l20d.check_fused_devices(self.fused_job(world=4), self.root / 'quiet-check'),
                             ','.join('GPU-uuid' + str(rank) for rank in range(4)))
        self.assertEqual(read.call_count, 7)
        kill.assert_not_called()

    def test_remote_fused_smoke_uses_child_env_and_persists_node_receipt(self):
        job = self.fused_job(run_id='remote-smoke', experiment='smoke-test',
                             source_id='source-id', timeout=300)
        path = self.root / 'job.json'
        path.write_text(json.dumps(job))
        remote = self.root / 'remote'
        remote.mkdir()
        control = self.root / 'control'
        env_receipt = {'host': l20d.NODES['0a'][1], 'overrides': {'CPATH': '/own/headers'}}
        proc = mock.Mock()
        proc.wait.return_value = 0
        with mock.patch.object(l20d, 'REMOTE', remote), \
                mock.patch.object(l20d, 'CONTROL', control), \
                mock.patch.object(l20d.socket, 'gethostname', return_value=l20d.NODES['0a'][1]), \
                mock.patch.object(l20d, 'install_source'), \
                mock.patch.object(l20d, 'environment_receipt', return_value=env_receipt), \
                mock.patch.object(l20d, 'check_fused_build') as check_build, \
                mock.patch.object(l20d, 'check_fused_devices', return_value='GPU-a,GPU-b') as check_gpus, \
                mock.patch.object(l20d, 'fused_telemetry', return_value=contextlib.nullcontext()), \
                mock.patch.object(l20d, 'mc_copy'), \
                mock.patch.object(l20d.subprocess, 'Popen', return_value=proc) as popen, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(l20d.remote(path), 0)
        check_build.assert_called_once()
        check_gpus.assert_called_once()
        argv = popen.call_args.args[0]
        self.assertEqual(argv[:4], ['bash', '-c',
                         'source /root/workspace_wct/env.sh && exec "$@"', 'l20d'])
        self.assertTrue(argv[4].endswith('/build/sm103-fused/fused_bf16'))
        child_env = popen.call_args.kwargs['env']
        self.assertEqual(child_env['CUDA_VISIBLE_DEVICES'], 'GPU-a,GPU-b')
        self.assertEqual(child_env['FUSE_QKV_GEMM_POLICY'], 'm128n128')
        self.assertEqual(child_env['FUSE_SM103_OPROJ_POLICY'], 'auto')
        self.assertEqual(child_env['CPATH'], '/own/headers')
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        state = json.loads((control / 'jobs/remote-smoke/status.json').read_text())
        self.assertEqual((state['node'], state['state']), ('0a', 'succeeded'))
        self.assertIn('artifact_sha256', state)

    def test_watch_short_job_completion_latency_preserves_sweep_cadence(self):
        for stage in ('fused-build', 'fused-smoke', 'baseline-replay'):
            self.assertEqual([l20d.watch_interval(stage, t)
                              for t in (0, 14.9, 15, 119.9, 120, 3600)],
                             [2, 2, 5, 5, 15, 15])
        for stage in ('sweep', 'formal', None):
            self.assertEqual([l20d.watch_interval(stage, t) for t in (0, 15, 120)],
                             [2, 30, 30])

    def test_watch_suppresses_heartbeats_preserves_terminal_and_fetches(self):
        states = [dict(run_id='watch-test', state='running', phase='sweep', attempt=1, elapsed_s=t)
                  for t in (3, 33, 63)]
        states.append(states[-1] | dict(state='succeeded', phase='finished', elapsed_s=65,
                                      artifact_sha256='verified-by-fetch'))
        output = io.StringIO()
        with mock.patch.object(l20d, 'LOCAL', self.root), \
                mock.patch.object(l20d, 'get_status', side_effect=states), \
                mock.patch.object(l20d.time, 'sleep'), \
                mock.patch.object(l20d, 'screen_snapshot') as screen, \
                mock.patch.object(l20d, 'submit') as submit, \
                mock.patch.object(l20d, 'fetch') as fetch, \
                contextlib.redirect_stdout(output):
            self.assertEqual(l20d.wait_for_run('watch-test'), 0)
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([x['phase'] for x in events], ['sweep', 'finished'])
        self.assertEqual(len((self.root/'watch-test/watch/events.jsonl').read_text().splitlines()), 2)
        screen.assert_not_called()
        submit.assert_not_called()
        fetch.assert_called_once_with('watch-test')

    def test_watch_stale_receipt_never_reads_screen_or_resubmits(self):
        with mock.patch.object(l20d, 'LOCAL', self.root), \
                mock.patch.object(l20d, 'get_status', return_value={'state': 'awaiting-status'}), \
                mock.patch.object(l20d.time, 'monotonic', side_effect=[0, 181]), \
                mock.patch.object(l20d, 'screen_snapshot') as screen, \
                mock.patch.object(l20d, 'submit') as submit:
            with self.assertRaisesRegex(RuntimeError, 'heartbeat is stale'):
                l20d.wait_for_run('stale-test')
        screen.assert_not_called()
        submit.assert_not_called()

    def test_watch_failure_fetches_diagnostics_without_retry(self):
        state = dict(run_id='failed-test', state='failed', phase='finished', attempt=1,
                     elapsed_s=39, artifact_sha256='verified-by-fetch')
        with mock.patch.object(l20d, 'LOCAL', self.root), \
                mock.patch.object(l20d, 'get_status', return_value=state), \
                mock.patch.object(l20d, 'fetch') as fetch, \
                mock.patch.object(l20d, 'submit') as submit, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(l20d.wait_for_run('failed-test'), 1)
        fetch.assert_called_once_with('failed-test')
        submit.assert_not_called()

    def test_progress_relay_drains_without_waiting_for_heartbeat(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'progress.log'
            path.write_text('verbose payload\nRUN 1/2 shape\nDONE 1/2 shape p50=0.1ms\n')
            stop = threading.Event()
            stop.set()
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                l20d.relay_progress(path, stop)
            self.assertEqual(output.getvalue(), 'RUN 1/2 shape\nDONE 1/2 shape p50=0.1ms\n')

    def test_fused_progress_is_live_but_raw_profile_stays_in_log(self):
        path = self.root / 'fused-progress.log'
        concise = ('config,world=8\ninput,lhs,seed=1\ncorrectness,QKV,max_abs=0\n'
                   'warmup,GEMM_A2A,iteration=1\nsample,GEMM_A2A,max_rank_ms=0.1\n'
                   'summary,GEMM_A2A,p50_ms=0.1\nprofile_resources,rank=0,threads=256\n'
                   'component_resources,GEMM_A2A,component=compute_reference,compute_budget=132\n'
                   'profile_dispatch,GEMM_A2A,host_launch=per_gpu_thread,api_begin_skew_us=4\n'
                   'PASS: both BF16 boundaries\n')
        path.write_text(concise + 'profile_cta,rank=0,cta=0,start=100\nprofile_peer,rank=0,index=0\n')
        stop = threading.Event()
        stop.set()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            l20d.relay_progress(path, stop)
        self.assertEqual(output.getvalue(), concise)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='fuse-l20d-test-')
        self.addCleanup(temporary.cleanup)
        # macOS may spell the temporary directory through the /var symlink.
        self.root = Path(temporary.name).resolve()

    def test_old_snapshot_content_changes_refresh_mtime_but_identical_files_do_not(self):
        archive = self.archive('old-source.tar.gz', [('kernel.cu', b'new kernel', False),
                                                   ('unchanged.h', b'same header', False)])
        remote, folder = self.root / 'remote', self.root / 'job'
        remote.mkdir(); folder.mkdir()
        (remote / 'kernel.cu').write_bytes(b'old kernel')
        (remote / 'unchanged.h').write_bytes(b'same header')
        for name in ('kernel.cu', 'unchanged.h'):
            os.utime(remote / name, (1000, 1000))
        job = dict(run_id='old-snapshot', source_id='new-content', archive_sha256=l20d.sha(archive),
                   files={name: hashlib.sha256(data).hexdigest() for name, data in
                          [('kernel.cu', b'new kernel'), ('unchanged.h', b'same header')]})
        before = time.time() - 1
        with mock.patch.object(l20d, 'REMOTE', remote), \
                mock.patch.object(l20d, 'mc_copy', side_effect=lambda _src, dst: shutil.copy2(archive, dst)):
            l20d.install_source(job, folder)
            changed_time = (remote / 'kernel.cu').stat().st_mtime_ns
            self.assertGreater(changed_time / 1e9, before)
            self.assertEqual((remote / 'unchanged.h').stat().st_mtime, 1000)
            self.assertEqual((folder / 'source-before/kernel.cu').read_bytes(), b'old kernel')
            l20d.install_source(job, folder)
            self.assertEqual((remote / 'kernel.cu').stat().st_mtime_ns, changed_time)

    def test_forced_rebuild_is_explicit_and_local_to_fused_build(self):
        job = self.fused_job(stage='fused-build')
        self.assertNotIn('--clean-first', l20d.fused_argv(job)[2])
        self.assertIn('--clean-first', l20d.fused_argv(job | {'rebuild': True})[2])
        for stage in ('fused-smoke', 'smoke', 'baseline-replay'):
            with self.subTest(stage=stage), self.assertRaises(ValueError):
                l20d.validate_job(job | {'stage': stage, 'rebuild': True})

    def archive(self, name, entries):
        path = self.root / name
        with tarfile.open(path, 'w:gz') as archive:
            for member, payload, is_link in entries:
                info = tarfile.TarInfo(member)
                if is_link:
                    info.type = tarfile.SYMTYPE
                    info.linkname = payload
                    archive.addfile(info)
                else:
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
        return path

    def artifact_state(self, artifact):
        return dict(run_id='example-run', state='succeeded', attempt=1,
                    artifact='mock-cloud/artifacts.tar.gz',
                    artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())

    def test_unpack_preserves_repository_relative_symlink(self):
        archive = self.archive('source.tar.gz', [
            ('benchmarks/sm103/result.txt', b'complete\n', False),
            ('benchmarks/sm103a', 'sm103', True),
        ])
        target = self.root / 'source'
        l20d.unpack(archive, target)
        self.assertTrue((target / 'benchmarks/sm103a').is_symlink())
        self.assertEqual((target / 'benchmarks/sm103a/result.txt').read_bytes(), b'complete\n')

    def test_unpack_rejects_escaping_members_and_links(self):
        outside = self.root / 'outside.txt'
        outside.write_bytes(b'keep me')
        entries = [
            ('../outside.txt', b'changed', False),
            (str(outside), b'changed', False),
            ('escape', '../outside.txt', True),
            ('escape', str(outside), True),
        ]
        for index, entry in enumerate(entries):
            with self.subTest(entry=entry):
                archive = self.archive(f'unsafe-{index}.tar.gz', [entry])
                target = self.root / f'extracted-{index}'
                with self.assertRaises(ValueError):
                    l20d.unpack(archive, target)
                self.assertEqual(outside.read_bytes(), b'keep me')
                self.assertFalse((target / 'escape').is_symlink())

    def test_source_snapshot_includes_dirty_files_and_hashes_archived_bytes(self):
        repo = self.root / 'repo'
        repo.mkdir()
        tracked = repo / 'kernel.py'
        tracked.write_bytes(b'indexed version\n')
        (repo / '.gitignore').write_text('ignored.txt\n')
        subprocess.run(['git', 'init', '-q', str(repo)], check=True, capture_output=True)
        subprocess.run(['git', '-C', str(repo), 'add', 'kernel.py', '.gitignore'],
                       check=True, capture_output=True)
        tracked.write_bytes(b'dirty working version\n')
        (repo / 'new.py').write_bytes(b'untracked source\n')
        (repo / 'ignored.txt').write_bytes(b'not source\n')
        (repo / 'compat.py').symlink_to('kernel.py')
        destination = self.root / 'package'
        destination.mkdir()
        original_add = tarfile.TarFile.add

        def add_then_edit(archive, name, *args, **kwargs):
            original_add(archive, name, *args, **kwargs)
            if Path(name) == tracked:
                tracked.write_bytes(b'edited after snapshot\n')

        with mock.patch.object(l20d, 'REPO', repo), \
                mock.patch.object(tarfile.TarFile, 'add', new=add_then_edit):
            archive_path, source_id, manifest = l20d.source_package(destination)
        with tarfile.open(archive_path) as archive:
            contents = archive.extractfile('kernel.py').read()
            self.assertEqual(contents, b'dirty working version\n')
            self.assertEqual(archive.extractfile('new.py').read(), b'untracked source\n')
            self.assertTrue(archive.getmember('compat.py').issym())
            self.assertNotIn('ignored.txt', archive.getnames())
        self.assertNotEqual(contents, tracked.read_bytes())
        self.assertEqual(manifest['kernel.py'], hashlib.sha256(contents).hexdigest())
        self.assertEqual(manifest['compat.py'], 'link:kernel.py')
        self.assertEqual(len(source_id), 64)

    def test_interrupted_fetch_does_not_certify_partial_extraction(self):
        archive = self.archive('artifact.tar.gz', [('control/result.json', b'{}\n', False)])
        state = self.artifact_state(archive)
        local = self.root / 'local'

        def interrupt_unpack(_archive, target):
            target.mkdir(parents=True)
            (target / 'partial').write_bytes(b'incomplete')
            raise OSError('simulated interrupted extraction')

        with mock.patch.object(l20d, 'LOCAL', local), \
                mock.patch.object(l20d, 'get_status', return_value=state), \
                mock.patch.object(l20d, 'mc_copy', side_effect=lambda _src, dst: shutil.copy2(archive, dst)):
            with mock.patch.object(l20d, 'unpack', side_effect=interrupt_unpack):
                with self.assertRaisesRegex(OSError, 'interrupted'):
                    l20d.fetch(state['run_id'])
            folder = local / state['run_id']
            self.assertFalse((folder / 'fetched.json').exists())
            self.assertFalse((folder / 'artifacts-attempt1').exists())
            with contextlib.redirect_stdout(io.StringIO()):
                l20d.fetch(state['run_id'])
        self.assertEqual((folder / 'artifacts-attempt1/control/result.json').read_bytes(), b'{}\n')
        self.assertFalse((folder / 'artifacts-attempt1/partial').exists())
        self.assertEqual(json.loads((folder / 'fetched.json').read_text()), state)

    def test_checksum_failure_preserves_artifact_and_blocks_cleanup(self):
        archive = self.archive('artifact.tar.gz', [('control/result.json', b'{}\n', False)])
        before = archive.read_bytes()
        state = self.artifact_state(archive) | {'artifact_sha256': '0' * 64}
        local = self.root / 'local'
        with mock.patch.object(l20d, 'LOCAL', local), \
                mock.patch.object(l20d, 'get_status', return_value=state), \
                mock.patch.object(l20d, 'mc_copy', side_effect=lambda _src, dst: shutil.copy2(archive, dst)), \
                mock.patch.object(l20d, 'command') as external_command:
            with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'):
                l20d.fetch(state['run_id'])
            folder = local / state['run_id']
            self.assertFalse((folder / 'fetched.json').exists())
            self.assertEqual((folder / 'artifacts.tar.gz').read_bytes(), before)
            with mock.patch.object(sys, 'argv', ['l20d.py', 'clean', state['run_id'], '--execute']):
                with self.assertRaisesRegex(RuntimeError, 'Fetch and verify'):
                    l20d.main()
            external_command.assert_not_called()
        self.assertEqual(archive.read_bytes(), before)

    def test_invalid_run_identifiers_cannot_form_cloud_prefixes(self):
        for value in ('', '..', '../other', '/absolute', 'a/b', '-option',
                      'a; command', 'name\n', 'x' * 102):
            with self.subTest(value=value), self.assertRaises(ValueError):
                l20d.prefix(value)
        self.assertTrue(l20d.prefix('20260906-120000-ab12cd').endswith('/20260906-120000-ab12cd'))


if __name__ == '__main__':
    unittest.main()
