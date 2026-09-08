"""Local-only synthetic receipt/log corruption tests; no CUDA/cloud dependencies."""

import csv
import io
import json
from pathlib import Path
import struct
import tarfile
import tempfile
import unittest

import summarize_sm103_fused as summary


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def line(record_kind, label=None, **values):
    def scalar(value):
        return format(value, '.9g') if isinstance(value, float) else str(value)
    parts = [record_kind] + ([] if label is None else [label])
    return ','.join(parts + [f'{key}={scalar(value)}' for key, value in values.items()])


class FusedSummaryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='fuse-summary-test-')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / 'run'
        self.control = self.root / 'artifacts-attempt1/control'
        self.control.mkdir(parents=True)
        self.log = self.control / 'attempt1.log'
        self.make_fixture()

    def write_json(self, path, value):
        path.write_text(json.dumps(value))

    def make_fixture(self, diagnostic=False, profile=False, calibrate=False, host_launch=None, profile_schema=None,
                     profile_detail=None, seq_local=256, schedule=None):
        manifest = {'CMakeLists.txt': 'a' * 64, 'benchmarks/sm103/fused_bf16.cu': 'b' * 64}
        job = dict(run_id='20260906-120000-abcdef', stage='fused-smoke', node='0a', experiment='fixture',
                   world=4, seq_local=seq_local, hidden=1024, q_heads=8, kv_heads=4, head_dim=128,
                   timeout_seconds=60, comm_sm=8, qkv_policy='auto', oproj_policy='auto',
                   devices='0,1,2,3,4,5,6,7', causal=True, input_generator='gpu_philox',
                   profile=profile, cpu_oracle=False, validation_self_test=diagnostic,
                   files=manifest, source_id=summary.json_digest(manifest))
        if schedule is not None:
            job.update(schedule)
        if calibrate:
            job['calibrate'] = True
        if host_launch is not None:
            job['host_launch'] = host_launch
        if profile_detail is not None:
            job['profile_detail'] = profile_detail
        environment = {'host': 'l20d-ebed3kz6-0000', 'compiler': 'fixture'}
        env_id = summary.json_digest(environment)
        status = {key: job[key] for key in ('run_id', 'node', 'stage', 'experiment', 'source_id')}
        status.update(attempt=1, state='running', phase='collect', exit_code=0, work_exit_code=0,
                      environment_fingerprint=env_id)
        self.write_json(self.root / 'job.json', job)
        self.write_json(self.control / 'job.json', job)
        self.write_json(self.control / 'status.json', status)
        self.write_json(self.control / 'source-installed.json', {'source_id': job['source_id'], 'files': manifest})
        self.write_json(self.control / 'environment.json', environment | {'fingerprint': env_id})
        self.write_json(self.control / 'fused-build.json', dict(node='0a', profile=profile,
            binary='/root/workspace_wct/fuse/build/sm103-fused' + ('-profile' if profile else '') + '/fused_bf16',
            binary_sha256='c' * 64, build_inputs=summary.fused_build_inputs(job), environment_fingerprint=env_id))
        self.write_json(self.control / 'gpu-before.json', dict(node='0a', selected=['0', '1', '2', '3'],
            observations=[{'devices': [{'index': str(rank), 'uuid': f'GPU-fixture-{rank}'} for rank in range(4)]}]))
        telemetry = 'timestamp, index, clocks.current.sm [MHz], clocks.current.memory [MHz], power.draw [W]\n'
        for tick in range(2):
            for rank in range(4):
                telemetry += f'2026/09/06 12:00:00.{tick}00, {rank}, 2000 MHz, 3996 MHz, 800 W\n'
        (self.control / 'gpu-telemetry.csv').write_text(telemetry)
        self.log.write_text(self.make_log(job, profile_schema))
        self.fetched = status | {'state': 'succeeded', 'phase': 'finished'}
        self.repack()

    def make_log(self, job, profile_schema=None):
        shape = summary.fused_geometry(job)
        m = shape['seq_local']
        diagnostic = job['validation_self_test']
        values = [line('config', world=4, comm_sm=8, global_seq=m * 4, seq_local=m, q_heads=8,
                      kv_heads=4, head_dim=128, hidden=1024, causal=1, seed=20260906, warmup=10, samples=50,
                      input_generator='gpu_philox', profile=int(job['profile']), timeout_seconds=60,
                      cpu_oracle=int(diagnostic), validation_self_test=int(diagnostic), candidates=2)]
        if job.get('calibrate'):
            values[0] += ',calibrate=1'
        if 'host_launch' in job:
            values[0] += ',host_launch=' + job['host_launch']
        if profile_schema is not None:
            values[0] += ',profile_schema=' + profile_schema
        for rank in range(4):
            values.append(line('device', rank=rank, name='NVIDIA L20D', runtime_cc='10.3', sms=148))
            for label, elements, seed in (
                ('QKV-weight', 1024 * 2048, 20260917), ('OProj-weight', 1024 * 1024, 20260923)):
                values.append(self.input_row(label, elements, seed, .02, rank=rank))
        for generation in range(2):
            for rank in range(4):
                for label, offset in (('QKV-activation', 1000), ('OProj-activation', 2000)):
                    values.append(self.input_row(label, m * 1024,
                        20260906 + generation * 100003 + rank * 101 + offset, .125,
                        generation=generation, rank=rank))
            for candidate, direction in enumerate(summary.DIRECTIONS, 1):
                context = dict(candidate=candidate, comm_sm=8, tile='m128n128', generation=generation)
                for rank in range(4):
                    values.append(line('candidate', direction, **context, state='resolved', rank=rank,
                                       tile_m=128, tile_n=128, tile_k=64, threads=256, dynamic_smem=230400))
                    n = shape['projection_width'] if direction == 'GEMM_A2A' else shape['hidden']
                    values.append(line('correctness', direction, **context, rank=rank, validator='gpu_full',
                        elements=m * n, checked=m * n, nonfinite=0, max_abs=0,
                        relative_l2=0, mismatches=0, atol=.01, rtol=.01))
                    route_elements = m * (n if direction == 'GEMM_A2A' else shape['q_width'])
                    values.append(line('route', direction, **context, rank=rank, validator='gpu_full',
                        elements=route_elements, checked=route_elements, nonfinite=0, bitwise_mismatches=0))
                if generation == 0:
                    if diagnostic:
                        q, kv = m * shape['q_width'], m * shape['kv_width']
                        faults = [('local_finite_error', 0), ('local_nan', q + 2 * kv - 1), ('route_nan', q + 2 * kv - 1)] + \
                            [('route_segment_boundary', index) for index in (0, q - 1, q, q + kv - 1, q + kv, q + 2 * kv - 1)] \
                            if direction == 'GEMM_A2A' else [('staging_tail_bit', q - 1), ('staging_nan', 0),
                                ('output_tail_finite_error', m * shape['hidden'] - 1), ('output_nan', 0)]
                        for fault, location in faults:
                            values.append(line('validation_self_test', direction, **context, fault=fault,
                                rank=0, index=location, gpu_cpu_detected=1, restored_full_clean=1, diagnostic_only=1))
                    else:
                        values.extend(self.timing_rows(direction, context))
                if diagnostic:
                    values.append(line('validation_oracle', direction, **context, full_cpu_match=1))
                if job.get('calibrate'):
                    values.extend(self.component_resource_rows(direction, context | {'component': 'fused'}, m))
                    for component, kind in (('compute_reference', 'correctness'), ('copy_reference', 'route')):
                        reference_context = context | {'component': component}
                        values.extend(self.component_resource_rows(direction, reference_context, m))
                        width = n if kind == 'correctness' else (n if direction == 'GEMM_A2A' else shape['q_width'])
                        def validation(phase):
                            for rank in range(4):
                                extra = dict(max_abs=0, relative_l2=0, mismatches=0, atol=.01, rtol=.01) \
                                    if kind == 'correctness' else dict(bitwise_mismatches=0)
                                values.append(line(kind, direction, **reference_context, validation_phase=phase,
                                    rank=rank, validator='gpu_full', elements=m * width, checked=m * width,
                                    nonfinite=0, **extra))
                        validation('pre')
                        if generation == 0:
                            values.extend(self.timing_rows(direction, reference_context))
                            validation('post')
        for candidate, direction in enumerate(summary.DIRECTIONS, 1):
            values.append(line('candidate_verified', direction, candidate=candidate, comm_sm=8,
                tile='m128n128', payload_generations=2, full_numeric=1, full_route=1,
                performance_accepted=int(not diagnostic)))
            if job.get('calibrate'):
                for component in ('compute_reference', 'copy_reference'):
                    values.append(line('candidate_verified', direction, candidate=candidate, comm_sm=8,
                        tile='m128n128', component=component, payload_generations=2,
                        full_numeric=int(component == 'compute_reference'), full_route=int(component == 'copy_reference'),
                        performance_accepted=1))
        if job['profile']:
            for direction in summary.DIRECTIONS:
                for rank in range(4):
                    values.append(line('profile_host', direction, rank=rank, instrumented_warmup=10,
                        host_launch=job.get('host_launch', 'sequential'),
                        production_launch_us=10, instrumented_launch_us=11,
                        production_event_us=5000, instrumented_event_us=5500))
                phases = ('instrumented', 'host_stages') if profile_schema == 'host_stages_v1' else (None,)
                for phase in phases:
                    if phase == 'host_stages':
                        values.extend(self.host_stage_rows(direction, job.get('host_launch', 'sequential')))
                    for rank in range(4):
                        context = {} if phase is None else {'profile_phase': phase}
                        n = shape['projection_width'] if direction == 'GEMM_A2A' else shape['hidden']
                        values.append(line('correctness', direction, rank=rank, validator='gpu_full', **context,
                            elements=m * n, checked=m * n, nonfinite=0, max_abs=0,
                            relative_l2=0, mismatches=0, atol=.01, rtol=.01))
                        route_n = n if direction == 'GEMM_A2A' else shape['q_width']
                        values.append(line('route', direction, rank=rank, validator='gpu_full', **context,
                            elements=m * route_n, checked=m * route_n, nonfinite=0, bitwise_mismatches=0))
            values.append(line('profile_cta', 'GEMM_A2A', rank=0, cta=0, role='comm'))
        if job.get('profile_detail') is not None:
            if job['profile_detail'] == 'full':
                values.append(line('profile_peer', rank=0, index=0, release=100))
            values = [row + ',profile_detail=' + job['profile_detail']
                      if row.startswith(('config,', 'profile', 'host_stage,')) else row for row in values]
        if diagnostic:
            values += [line('input_oracle', generator='gpu_philox', check=kind, full_bitwise_match=1)
                       for kind in ('cross_rank_weights', 'same_seed_repeat')]
            values += [line('input_oracle', generator='gpu_philox', check='oproj_reference_gather',
                           rank=rank, elements=m * shape['q_width'], full_cpu_bitwise_match=1)
                       for generation in range(2) for rank in range(4)]
        if any(key in job for key in summary.SCHEDULE_REQUEST_FIELDS):
            values = self.add_schedule_records(values, job)
        values.append('PASS: both BF16 boundaries, complete routes, changed payloads')
        return '\n'.join(values) + '\n'

    def host_stage_rows(self, direction, host_launch):
        rows = []
        for sample in range(50):
            for rank in range(4):
                row = line('host_stage', direction, sample=sample, rank=rank, epoch=120 + sample,
                    host_launch=host_launch, warmup=10, samples=50, profile_phase='host_stages',
                    profile_schema='host_stages_v1', performance_accepted=0, clock='steady_clock',
                    clock_overhead_subtracted=0, api_boundary='after_start_event_before_end_event',
                    stage_begin='policy_entry', descriptor_timing='subset_of_communication_prepare',
                    status=0, stage_mask=255, complete=1, api_us=10, api_prefix_us=1, library_return_us=1,
                    api_suffix_us=1, **{name + '_us': 1 for name in summary.HOST_STAGES},
                    local_descriptor_us=.1, peer_descriptors_us=.2, local_descriptor_count=1,
                    peer_descriptor_count=4, api_begin_us=100 + rank * 5, api_end_us=110 + rank * 5,
                    api_begin_skew_us=15, api_end_skew_us=15, all_enqueued_us=125, event_us=5000)
                rows.append(row + ',kind=production_kernel_diagnostic')
        return rows

    def component_resource_rows(self, direction, context, m=256):
        component = context['component']
        tiles = ((m + 127) // 128) * (16 if direction == 'GEMM_A2A' else 8)
        return [line('component_resources', direction, **context, rank=rank, tile_m=128, tile_n=128,
            tile_k=64, raster='along_m' if direction == 'GEMM_A2A' else 'along_n', compute_budget=140,
            scheduled_compute_ctas=0 if component == 'copy_reference' else tiles,
            scheduled_comm_ctas=0 if component == 'compute_reference' else 8,
            production_threads=256, production_dynamic_smem=230400,
            reference_resources='not_applicable' if component == 'fused' else 'unknown') for rank in range(4)]

    def add_schedule_records(self, rows, job):
        """Synthetic records use independently calculated, exact integer geometry."""
        maximum = job.get('max_swizzle_size', 1)
        requested = {direction: job.get(key, 'heuristic') for direction, key in
                     zip(summary.DIRECTIONS, ('qkv_raster', 'oproj_raster'))}
        effective_raster = {direction: ('along_m' if direction == 'GEMM_A2A' else 'along_n')
                            if value == 'heuristic' else value for direction, value in requested.items()}
        result = []
        for original in rows:
            parts = original.split(',')
            if parts[0] == 'config':
                original += ',' + ','.join(f'{key}={value}' for key, value in dict(
                    max_swizzle_size=maximum, qkv_raster=requested['GEMM_A2A'], oproj_raster=requested['A2A_GEMM'],
                    qkv_effective_raster=effective_raster['GEMM_A2A'],
                    oproj_effective_raster=effective_raster['A2A_GEMM']).items())
            elif parts[0] in ('candidate', 'component_resources'):
                direction = parts[1]
                fields = dict(part.split('=', 1) for part in parts[2:])
                m_tiles = (job['seq_local'] + 127) // 128
                n_tiles = 16 if direction == 'GEMM_A2A' else 8
                minimum = min(m_tiles, n_tiles)
                effective = next(size for size, threshold in ((8, 6), (4, 3), (2, 2), (1, 1))
                                 if size <= maximum and minimum >= threshold)
                pm = ((m_tiles + effective - 1) // effective) * effective
                pn = ((n_tiles + effective - 1) // effective) * effective
                fields.update(raster=effective_raster[direction], max_swizzle_size=maximum,
                              effective_swizzle_size=effective, padded_m_tiles=pm, padded_n_tiles=pn,
                              scheduled_compute_ctas=0 if fields.get('component') == 'copy_reference' else min(140, pm * pn))
                original = line(parts[0], direction, **fields)
            result.append(original)
        return result

    def input_row(self, label, count, seed, magnitude, **context):
        return line('input', label, **context, seed=seed, count=count, generator='gpu_philox',
                    algorithm='curand_philox4x32_10', mapping='thread_subsequence_v1', blocks=256,
                    threads=256, offset=0, distribution='uniform', lower=-magnitude, upper=magnitude,
                    min=-summary.bf16_round(magnitude), max=summary.bf16_round(magnitude),
                    mean=0, rms=magnitude / 3**.5, std=magnitude / 3**.5, finite=count, nonzero_fraction=1)

    def make_epilogue_fixture(self, seq_local=256):
        self.make_fixture(profile=True, profile_schema='host_stages_v1', profile_detail='cta',
                          host_launch='per_gpu_thread', seq_local=seq_local, schedule=dict(max_swizzle_size=1))
        job = json.loads((self.control / 'job.json').read_text())
        job.update(qkv_epilogue_probe=True, qkv_policy='m128n256k64e32')
        job['files']['benchmarks/sm103/fused_bf16.cu'] = next(iter(summary.LEGACY_EPILOGUE_JOIN_HARNESS_SHA256))
        job['source_id'] = summary.json_digest(job['files'])
        for path in (self.control / 'job.json', self.root / 'job.json'):
            self.write_json(path, job)
        self.write_json(self.control / 'source-installed.json', {'source_id': job['source_id'], 'files': job['files']})
        status = json.loads((self.control / 'status.json').read_text())
        status['source_id'] = self.fetched['source_id'] = job['source_id']
        self.write_json(self.control / 'status.json', status)
        build = json.loads((self.control / 'fused-build.json').read_text())
        build['build_inputs'] = summary.fused_build_inputs(job)
        self.write_json(self.control / 'fused-build.json', build)
        m_tiles, n_tiles = (seq_local + 127) // 128, 8
        total_tiles = m_tiles * n_tiles
        compute = min(total_tiles, 140)
        original = []
        for row in self.log.read_text().splitlines():
            if row.startswith('config,'):
                row += ',qkv_epilogue_probe=1'
            if ',GEMM_A2A,' in row:
                row = row.replace('tile=m128n128,', 'tile=m128n256k64e32,').replace('tile_n=128,', 'tile_n=256,')
                if row.startswith('candidate,'):
                    parts = row.split(',')
                    parts = ['padded_n_tiles=8' if part.startswith('padded_n_tiles=') else
                             'scheduled_compute_ctas=' + str(compute) if part.startswith('scheduled_compute_ctas=') else part
                             for part in parts]
                    row = ','.join(parts)
            original.append(row)
        rows = []
        for rank in range(4):
            for mode in summary.EPILOGUE_MODES:
                rows.append(line('epilogue_resources', rank=rank, kind=mode,
                    schema='qkv_epilogue_cta_v1', clock='globaltimer', clock_unit='ns',
                    store_interval='issuing_lane_base_call_including_accumulator_wait',
                    drain_interval='issuing_warp_global_wait_and_warp_join', record_bytes=96,
                    regs=96 if mode == 'epilogue_telemetry' else 80, local_bytes=16 if mode == 'epilogue_telemetry' else 0,
                    static_smem=1024, dynamic_smem=227328, max_threads=256, cluster_ctas=1,
                    tile_m=128, tile_n=256, tile_k=64, performance_accepted=0))
        # Final sample 49 executes reverse mode order. Validation is emitted
        # immediately; samples are buffered until all four validations finish.
        for phase in tuple('epilogue_' + mode for mode in summary.EPILOGUE_MODES[::-1]) + ('epilogue_record',):
            for rank in range(4):
                common = dict(rank=rank, profile_phase=phase, validator='gpu_full', elements=seq_local * 2048,
                              checked=seq_local * 2048, nonfinite=0)
                rows.append(line('correctness', 'GEMM_A2A', **common,
                                 max_abs=0, relative_l2=0, mismatches=0, atol=.01, rtol=.01))
                rows.append(line('route', 'GEMM_A2A', **common, bitwise_mismatches=0))
        for sample in range(50):
            for slot, mode in enumerate(summary.EPILOGUE_MODES if sample % 2 == 0 else summary.EPILOGUE_MODES[::-1]):
                for rank in range(4):
                    rows.append(line('epilogue_sample', rank=rank, sample=sample, kind=mode,
                        epoch=200 + sample * 3 + slot, warmup=10, samples=50,
                        host_launch='per_gpu_thread', launch='eager', process_layout='single_process',
                        performance_accepted=0, final_sample_poisoned=int(sample == 49),
                        event_ms=f32(1 + .1 * summary.EPILOGUE_MODES.index(mode) + .01 * rank + .0001 * sample)))
        for rank in range(4):
            for cta in range(8, 8 + compute):
                worker = cta - 8
                count = (total_tiles - worker + compute - 1) // compute
                start = 10000 + rank * 10000 + cta
                rows.append(line('epilogue_cta', rank=rank, cta=cta, epoch=350,
                    schema='qkv_epilogue_cta_v1', clock='globaltimer', clock_unit='ns',
                    tile_count=count, first_m_tile=worker % m_tiles, first_n_tile=worker // m_tiles, first_batch=0,
                    store_ns_sum=count * 20, store_ns_max=20, drain_ns_sum=count * 10, drain_ns_max=10,
                    first_store_begin=start + 100, first_store_end=start + 120, first_drain_end=start + 130,
                    first_ready_after=start + 133, last_ready_after=start + 133 + (count - 1) * 200,
                    cta_start=start, cta_role_done=start + 1000, cta_end=start + 1100, performance_accepted=0))
        position = next(i for i, row in enumerate(original) if row.startswith('profile_host,A2A_GEMM,rank=0,'))
        self.log.write_text('\n'.join(original[:position] + rows + original[position:]) + '\n')
        self.repack()
        return job, self.log.read_text()

    def timing_rows(self, direction, context):
        records = []
        def raw(kind, phase, round_id, index, epoch, delta=0):
            times = [f32(5 + rank / 10 + delta) for rank in range(4)]
            records.append(line(kind, direction, **context, phase=phase, round=round_id,
                index=index, epoch=epoch, maxrank_ms=max(times),
                **{f'rank{rank}_ms': value for rank, value in enumerate(times)}))
            return max(times)
        for index in range(10):
            raw('warmup', 'initial', -1, index, index + 2)
        for window in range(3):
            for rank in range(4):
                average = 5 + rank / 10
                records.append(line('warmup', direction, **context, phase='convergence', rank=rank,
                    window=window, calls=10, epoch=21 + window * 10, ms_per_call=average,
                    accumulated_cuda_ms=average * 10 * (window + 1), ready=int(window == 2)))
        for index in range(10):
            raw('warmup', 'sample_cadence', -1, index, 42 + index)
        records.append(line('sample', direction, **context, round=0, state='collecting', count=50))
        samples = [raw('sample', 'measurement', 0, index, 52 + index, index * .0001) for index in range(50)]
        records.append(line('sample', direction, **context, round=0, state='complete',
                            half_drift=summary.drift(samples), stable_5pct=1))
        records.append(line('summary', direction, **context, verification='pending', warmup=10,
            additional_warmup_calls=30, minimum_warmup_cuda_ms=100, warmup_wall_s=.2,
            converged_all_ranks=1, sample_cadence_warmup=10, samples=50, selected_round=0,
            warmup_p50_ms=f32(5.3), warmup_p95_ms=f32(5.3), p50_ms=summary.percentile(samples, .5),
            p95_ms=summary.percentile(samples, .95), half_drift=summary.drift(samples), stable_5pct=1,
            collector='per_epoch_rank_events_v2', boundary='single_process_eager_maxrank_cudaevent'))
        return records

    def repack(self, unsafe=False):
        archive = self.root / 'artifacts.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            for path in sorted(self.control.iterdir()):
                tar.add(path, arcname='control/' + path.name)
            if unsafe:
                entry = tarfile.TarInfo('../escape')
                entry.size = 1
                tar.addfile(entry, io.BytesIO(b'x'))
        self.fetched['artifact_sha256'] = summary.file_digest(archive)
        self.write_json(self.root / 'fetched.json', self.fetched)

    def change_log(self, change):
        self.log.write_text(change(self.log.read_text()))
        self.repack()

    def make_mpi_fixture(self, calibrate=False, schedule=None, seq_local=256):
        self.make_fixture(calibrate=calibrate, schedule=schedule, seq_local=seq_local)
        job = json.loads((self.control / 'job.json').read_text())
        job.update(mpi=True, host_launch='sequential')
        for path in (self.control / 'job.json', self.root / 'job.json'):
            self.write_json(path, job)
        toolchain = {'overrides': {'MPICH_CXX': '/fixture/g++', 'UCX_TLS': 'sm,self'}}
        environment = {'host': 'l20d-ebed3kz6-0000', 'compiler': 'fixture', 'mpi_toolchain': toolchain}
        env_id = summary.json_digest(environment)
        self.write_json(self.control / 'environment.json', environment | {'fingerprint': env_id})
        status = json.loads((self.control / 'status.json').read_text())
        status['environment_fingerprint'] = env_id
        self.write_json(self.control / 'status.json', status)
        self.fetched['environment_fingerprint'] = env_id
        binary = '/root/workspace_wct/fuse/build/sm103-fused-mpi/fused_bf16_mpi'
        self.write_json(self.control / 'fused-build.json', dict(node='0a', profile=False, mpi=True,
            binary=binary, binary_sha256='c' * 64, build_inputs=summary.fused_build_inputs(job),
            environment_fingerprint=env_id, mpi_toolchain=toolchain))
        self.write_json(self.control / 'mpi-runtime-attempt1.json', dict(schema='sm103_mpi_runtime_v1',
            node='0a', world=4, process_layout='mpi_one_process_per_gpu', host_launch='mpi_process',
            launch='eager', collector='mpi_rank_events_v1', boundary='mpi_eager_maxrank_cudaevent',
            overrides=toolchain['overrides'], argv=[
                '/root/workspace_wct/toolchain/mpich-5.0.1.post1/bin/mpiexec', '-launcher', 'fork', '-n', '4', binary]))
        streams = {rank: [] for rank in range(4)}
        for text in self.log.read_text().splitlines():
            text = text.replace('per_epoch_rank_events_v2', 'mpi_rank_events_v1').replace(
                'single_process_eager_maxrank_cudaevent', 'mpi_eager_maxrank_cudaevent')
            kind = text.split(',')[0]
            if kind == 'config':
                text += ',host_launch=mpi_process,process_layout=mpi_one_process_per_gpu,launch=eager'
            fields = dict(part.split('=', 1) for part in text.split(',')[1:] if '=' in part)
            native = kind in ('device', 'input', 'candidate', 'component_resources') or (
                kind == 'input_oracle' and 'rank' in fields)
            streams[int(fields['rank']) if native else 0].append(text)
        for rank, lines in streams.items():
            (self.control / f'mpi-attempt1-rank-{rank}.stdout.log').write_text('\n'.join(lines) + '\n')
            (self.control / f'mpi-attempt1-rank-{rank}.stderr.log').write_bytes(b'')
        self.merge_mpi_fixture()

    def merge_mpi_fixture(self, prefix=b'', suffix=b''):
        merged, records = bytearray(prefix), []
        for rank in range(4):
            for stream in ('stdout', 'stderr'):
                name = f'mpi-attempt1-rank-{rank}.{stream}.log'
                data = (self.control / name).read_bytes()
                merged.extend(f'\n# MPI rank={rank} stream={stream}; original={name}\n'.encode())
                begin = len(merged)
                merged.extend(data)
                records.append(dict(rank=rank, stream=stream, path=name, present=True, rank_started=True,
                    bytes=len(data), sha256=summary.digest(data), merged_begin=begin, merged_end=len(merged)))
        merged.extend(suffix)
        self.log.write_bytes(merged)
        job = json.loads((self.control / 'job.json').read_text())
        self.write_json(self.control / 'mpi-logs-attempt1.json', dict(schema='sm103_mpi_rank_logs_v1',
            ordering='rank_then_stream_not_global_chronological',
            collector='mpi_graph_rank_events_v1' if job.get('fused_launch') == 'graph' else 'mpi_rank_events_v1',
            merged_log='attempt1.log', merged_sha256=summary.digest(merged), ranks=records,
            complete=True, numeric_or_performance_accepted=False))
        self.repack()

    def move_mpi_record(self, prefix, source=0, target=1, stream='stdout'):
        source_path = self.control / f'mpi-attempt1-rank-{source}.stdout.log'
        rows = source_path.read_text().splitlines()
        moved = next(row for row in rows if row.startswith(prefix))
        rows.remove(moved)
        source_path.write_text('\n'.join(rows) + '\n')
        target_path = self.control / f'mpi-attempt1-rank-{target}.{stream}.log'
        target_path.write_text(target_path.read_text() + moved + '\n')
        self.merge_mpi_fixture()

    def make_graph_fixture(self, calibrate=False):
        self.make_mpi_fixture(calibrate=calibrate)
        job = json.loads((self.control / 'job.json').read_text())
        job['fused_launch'] = 'graph'
        for path in (self.control / 'job.json', self.root / 'job.json'):
            self.write_json(path, job)
        path = self.control / 'mpi-runtime-attempt1.json'
        runtime = json.loads(path.read_text())
        runtime.update(launch='graph', collector='mpi_graph_rank_events_v1',
                       boundary='mpi_graph_maxrank_cudaevent', graph_epoch_mode=summary.GRAPH_EPOCH_MODE)
        runtime['argv'] += ['--launch', 'graph']
        self.write_json(path, runtime)
        for rank in range(4):
            path = self.control / f'mpi-attempt1-rank-{rank}.stdout.log'
            rows, grouped = [], {}
            for original in path.read_text().splitlines():
                parts = original.split(',')
                kind = parts[0]
                fields = dict(part.split('=', 1) for part in parts[1:] if '=' in part)
                text = original.replace('collector=mpi_rank_events_v1', 'collector=mpi_graph_rank_events_v1').replace(
                    'boundary=mpi_eager_maxrank_cudaevent', 'boundary=mpi_graph_maxrank_cudaevent')
                if kind == 'config':
                    text = text.replace('launch=eager', 'launch=graph') + ',collector=mpi_graph_rank_events_v1'
                if kind in ('summary', 'candidate_verified'):
                    text += ',launch=graph'
                if kind in ('config', 'summary', 'candidate_verified'):
                    text += ',graph_epoch_mode=' + summary.GRAPH_EPOCH_MODE
                rows.append(text)
                if 'candidate' in fields and 'generation' in fields:
                    key = fields['candidate'], fields.get('component', 'fused'), fields['generation']
                    grouped.setdefault(key, []).append((len(rows) - 1, original, fields, parts[1]))
            additions = {}
            for (candidate, component, generation), selected in grouped.items():
                position, _, fields, direction = selected[-1]
                extra = []
                if rank == 0 and component == 'fused' and generation == '0':
                    extra += [original + ',validation_phase=post' for _, original, _, _ in selected
                              if original.startswith(('correctness,', 'route,'))]
                first = 102 if component == 'fused' and generation == '1' else 1
                calls = 101 if generation == '0' else 1  # precheck + 10+30+10 warmups + 50 samples.
                extra.append(line('graph_prepare', direction, candidate=candidate, component=component,
                    generation=generation, comm_sm=fields['comm_sm'], tile=fields['tile'], rank=rank,
                    launch='graph', graph_epoch_mode=summary.GRAPH_EPOCH_MODE, calls=calls,
                    first_epoch=first, last_epoch=first + calls - 1, wall_s=.01,
                    includes='capture_inspect_instantiate_initial_upload_sync_update', gpu_sample_time=0))
                additions[position] = extra
            path.write_text('\n'.join(item for position, text in enumerate(rows)
                                      for item in [text] + additions.get(position, [])) + '\n')
        self.merge_mpi_fixture()
        return job, self.log.read_text()

    def test_graph_fcr_receipts_samples_and_actual_preparations(self):
        self.make_graph_fixture(calibrate=True)
        run = summary.audit_run(self.root)
        self.assertEqual(len(run['candidates']), 6)
        for row in run['candidates']:
            self.assertEqual(row['launch'], 'graph')
            self.assertEqual(row['collector'], 'mpi_graph_rank_events_v1')
            self.assertEqual(row['graph_epoch_mode'], 'recapture_update_v1')
            preparation = row['graph_preparation']
            self.assertEqual([entry['calls'] for entry in preparation], [101, 1])
            self.assertEqual(preparation[1]['first_epoch'], 102 if row['component'] == 'fused' else 1)
            self.assertEqual(len(preparation[0]['rank_host_preparation']), 4)
        out = Path(self.directory.name) / 'graph-summary'
        summary.summarize([self.root], out)
        with (out / 'summary.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual({row['launch'] for row in rows}, {'graph'})
        self.assertEqual({row['graph_epoch_mode'] for row in rows}, {'recapture_update_v1'})

    def test_graph_missing_or_inconsistent_launch_metadata_rejected(self):
        job, text = self.make_graph_fixture()
        for prefix in ('config,', 'summary,', 'candidate_verified,'):
            original = next(row for row in text.splitlines() if row.startswith(prefix))
            for field in ('launch=graph', 'graph_epoch_mode=recapture_update_v1'):
                changed = original.replace(',' + field, '', 1)
                with self.subTest(prefix=prefix, field=field), self.assertRaisesRegex(ValueError, 'Graph|Sampling'):
                    summary.audit_log(text.replace(original, changed, 1), job)
        for changed_job in (job | {'fused_launch': 'eager'}, job | {'fused_launch': 'other'},
                            job | {'mpi': False}, job | {'profile': True}):
            with self.subTest(job=changed_job), self.assertRaises(ValueError):
                summary.audit_log(text, changed_job)

    def test_graph_prepare_field_domain_and_launch_epoch_corruptions_rejected(self):
        job, text = self.make_graph_fixture(calibrate=True)
        original = next(row for row in text.splitlines() if row.startswith('graph_prepare,'))
        for field in ('rank', 'generation', 'component', 'calls', 'first_epoch', 'last_epoch', 'wall_s',
                      'gpu_sample_time', 'graph_epoch_mode', 'includes', 'launch'):
            changed = ','.join(part for part in original.split(',') if not part.startswith(field + '='))
            with self.subTest(missing=field), self.assertRaises(ValueError):
                summary.audit_log(text.replace(original, changed, 1), job)
        mutations = [('calls=101', 'calls=102'), ('last_epoch=101', 'last_epoch=102'),
                     ('first_epoch=1', 'first_epoch=2'), ('gpu_sample_time=0', 'gpu_sample_time=1'),
                     ('wall_s=0.01', 'wall_s=-1'), ('rank=0', 'rank=4'),
                     ('graph_epoch_mode=recapture_update_v1', 'graph_epoch_mode=replay_static_epoch'),
                     ('includes=capture_inspect_instantiate_initial_upload_sync_update', 'includes=launch')]
        for before, after in mutations:
            with self.subTest(change=after), self.assertRaises(ValueError):
                summary.audit_log(text.replace(original, original.replace(before, after), 1), job)
        for changed in (text.replace(original + '\n', '', 1), text.replace(original, original + '\n' + original, 1),
                        text.replace(original, original + ',unknown=1', 1),
                        text.replace('graph_prepare,', 'graph_unregistered,', 1)):
            with self.assertRaises(ValueError):
                summary.audit_log(changed, job)

    def test_graph_fused_post_validation_is_required_and_after_samples(self):
        job, text = self.make_graph_fixture()
        for prefix in ('correctness,', 'route,'):
            original = next(row for row in text.splitlines() if row.startswith(prefix) and 'validation_phase=post' in row)
            with self.assertRaisesRegex(ValueError, 'Missing rank/payload'):
                summary.audit_log(text.replace(original + '\n', '', 1), job)
            moved = text.replace(original + '\n', '', 1).replace('warmup,GEMM_A2A,', original + '\nwarmup,GEMM_A2A,', 1)
            with self.assertRaisesRegex(ValueError, 'post-validation'):
                summary.audit_log(moved, job)

    def test_graph_fused_continuity_and_reference_reset_cannot_be_forged(self):
        job, text = self.make_graph_fixture(calibrate=True)
        for component in ('fused', 'compute_reference', 'copy_reference'):
            changed = []
            for row in text.splitlines():
                if row.startswith('graph_prepare,GEMM_A2A,') and f'component={component},' in row and 'generation=1,' in row:
                    fields = dict(part.split('=', 1) for part in row.split(',')[2:] if '=' in part)
                    value = int(fields['first_epoch']) + 1
                    row = row.replace('first_epoch=' + fields['first_epoch'], 'first_epoch=' + str(value)).replace(
                        'last_epoch=' + fields['last_epoch'], 'last_epoch=' + str(value))
                changed.append(row)
            with self.subTest(component=component), self.assertRaisesRegex(ValueError, 'contiguous|baseline'):
                summary.audit_log('\n'.join(changed) + '\n', job)

    def test_graph_preparation_belongs_to_its_original_rank_stream(self):
        self.make_graph_fixture()
        self.move_mpi_record('graph_prepare,GEMM_A2A,')
        self.assert_invalid('another rank|local order')

    def test_graph_continuity_covers_multiple_candidates_per_direction(self):
        rows = []
        for direction in summary.DIRECTIONS:
            for index, first, last, changed_epoch in ((1, 1, 101, 205), (2, 102, 204, 206)):
                rows.append(dict(candidate=index, direction=direction, component='fused',
                    graph_preparation=[dict(first_epoch=first, last_epoch=last),
                                       dict(first_epoch=changed_epoch, last_epoch=changed_epoch)]))
        summary.audit_graph_epoch_continuity(rows)
        rows[1]['graph_preparation'][0]['first_epoch'] += 1
        with self.assertRaisesRegex(ValueError, 'contiguous'):
            summary.audit_graph_epoch_continuity(rows)

    def test_graph_preparation_must_precede_final_acceptance(self):
        job, text = self.make_graph_fixture()
        original = next(row for row in text.splitlines() if row.startswith('graph_prepare,GEMM_A2A,')
                        and 'generation=1,' in row and 'rank=0,' in row)
        rows = text.replace(original + '\n', '', 1).splitlines()
        final = next(i for i, row in enumerate(rows) if row.startswith('candidate_verified,GEMM_A2A,'))
        rows.insert(final + 1, original)
        with self.assertRaisesRegex(ValueError, 'follows final acceptance'):
            summary.audit_log('\n'.join(rows) + '\n', job)

    def test_graph_receipt_epoch_mode_collector_and_cli_are_required(self):
        for field in ('graph_epoch_mode', 'collector', 'argv'):
            self.make_graph_fixture()
            path = self.control / 'mpi-runtime-attempt1.json'
            runtime = json.loads(path.read_text())
            if field == 'argv':
                runtime[field] = runtime[field][:-2]
            else:
                runtime.pop(field)
            self.write_json(path, runtime)
            self.repack()
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'MPI'):
                summary.audit_run(self.root)

    def test_epilogue_join_requires_explicit_new_source_and_preserves_old_v21(self):
        job, text = self.make_epilogue_fixture()
        old = summary.audit_log(text, job)
        self.assertEqual(old['epilogue_diagnostics']['role_timestamp_join_schema'], 'legacy_unlogged')
        new_job = job | {'files': job['files'] | {'benchmarks/sm103/fused_bf16.cu': 'd' * 64}}
        with self.assertRaisesRegex(ValueError, 'Missing/unknown epilogue'):
            summary.audit_log(text, new_job)
        changed = []
        joins = dict(production='none', role_telemetry='legacy_bar_sync', epilogue_telemetry='cta_popc256_dependency')
        for row in text.splitlines():
            if row.startswith('epilogue_resources,'):
                fields = dict(part.split('=', 1) for part in row.split(',')[1:])
                row += ',role_timestamp_join=' + joins[fields['kind']]
            changed.append(row)
        explicit = '\n'.join(changed) + '\n'
        current = summary.audit_log(explicit, new_job)
        self.assertEqual(current['epilogue_diagnostics']['role_timestamp_join_schema'], 'explicit_v1')
        with self.assertRaisesRegex(ValueError, 'Missing/unknown epilogue'):
            summary.audit_log(explicit, job)
        with self.assertRaisesRegex(ValueError, 'join mismatch'):
            summary.audit_log(explicit.replace('role_timestamp_join=cta_popc256_dependency',
                                              'role_timestamp_join=legacy_bar_sync', 1), new_job)

    def assert_invalid(self, pattern):
        out = Path(self.directory.name) / 'summary'
        with self.assertRaisesRegex(ValueError, pattern):
            summary.summarize([self.root], out)
        self.assertFalse(out.exists())

    def test_valid_export_recomputes_raw_samples_and_records_explicit_old_defaults(self):
        out = Path(self.directory.name) / 'summary'
        result = summary.summarize([self.control.parent], out)
        self.assertEqual(result['performance_rows'], 2)
        run = result['runs'][0]
        self.assertEqual(run['schema_defaults'], {'host_launch': 'sequential', 'component': 'fused'})
        self.assertEqual(len(run['input_statistics']), 24)
        candidate = run['candidates'][0]
        self.assertEqual(len(candidate['timing']['rounds'][0]['rank_ms']), 50)
        self.assertEqual(candidate['compute_ctas_derived'], [32] * 4)
        self.assertEqual(candidate['schedule_schema'], 'legacy_fixed_v1')
        self.assertEqual((candidate['raster_requested'], candidate['raster'], candidate['max_swizzle_size'],
                          candidate['swizzle']), ('heuristic', 'along_m', 1, 1))
        self.assertAlmostEqual(candidate['timing']['p50_ms'], 5.30245, places=5)
        with (out / 'summary.csv').open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 2)

    def test_selected_pass_wording_preserves_full_candidate_audit(self):
        self.change_log(lambda text: text.replace('PASS: both BF16', 'PASS: selected BF16'))
        self.assertEqual(len(summary.audit_run(self.root)['candidates']), 2)
        self.change_log(lambda text: '\n'.join(row for row in text.splitlines()
            if not row.startswith('candidate_verified,A2A_GEMM,')) + '\n')
        with self.assertRaises(ValueError):
            summary.audit_run(self.root)

    def test_single_direction_requires_only_selected_evidence_but_never_missing_payloads(self):
        for direction, label, excluded in (('oproj', 'A2A_GEMM', 'QKV'),
                                            ('qkv', 'GEMM_A2A', 'OProj')):
            with self.subTest(direction=direction):
                self.make_fixture()
                job = json.loads((self.control / 'job.json').read_text())
                job['fused_direction'] = direction
                for path in (self.control / 'job.json', self.root / 'job.json'):
                    self.write_json(path, job)
                other = 'GEMM_A2A' if label == 'A2A_GEMM' else 'A2A_GEMM'
                def select(text):
                    rows = [row for row in text.splitlines() if f',{other},' not in row
                            and not row.startswith('input,' + excluded + '-')]
                    return '\n'.join(rows).replace('config,', f'config,fused_direction={direction},') \
                        .replace('candidates=2', 'candidates=1').replace('candidate=2,', 'candidate=1,') + '\n'
                self.change_log(select)
                result = summary.audit_run(self.root)
                self.assertEqual(len(result['candidates']), 1)
                self.assertEqual(len(result['input_statistics']), 12)
                self.change_log(lambda text: '\n'.join(row for row in text.splitlines()
                    if not (row.startswith('input,') and 'generation=1' in row)) + '\n')
                with self.assertRaises(ValueError):
                    summary.audit_run(self.root)

    def test_rank_swizzle_marker_must_match_job(self):
        self.change_log(lambda text: text.replace('config,', 'config,qkv_rank_swizzle=rank_n_band_v1,', 1))
        with self.assertRaisesRegex(ValueError, 'rank swizzle job/config mismatch'):
            summary.audit_run(self.root)
        self.change_log(lambda text: text.replace('qkv_rank_swizzle=rank_n_band_v1', 'qkv_rank_swizzle=off'))
        self.assertTrue(all(row['qkv_rank_swizzle'] == 'off'
                            for row in summary.audit_run(self.root)['candidates']))

    def test_schedule_nondefaults_export_actual_geometry_for_fused_and_both_references(self):
        self.make_fixture(calibrate=True, seq_local=1024, schedule=dict(
            max_swizzle_size=8, qkv_raster='along_n', oproj_raster='along_m'))
        out = Path(self.directory.name) / 'summary'
        report = summary.summarize([self.root], out)
        candidates = report['runs'][0]['candidates']
        self.assertEqual(len(candidates), 6)
        for row in candidates:
            qkv = row['direction'] == 'GEMM_A2A'
            self.assertEqual(row['schedule_schema'], 'explicit_v1')
            self.assertEqual(row['raster_requested'], 'along_n' if qkv else 'along_m')
            self.assertEqual(row['raster'], row['raster_requested'])
            self.assertEqual((row['max_swizzle_size'], row['effective_swizzle_size'], row['swizzle']), (8, 8, 8))
            self.assertEqual((row['padded_m_tiles'], row['padded_n_tiles']), (8, 16 if qkv else 8))
            self.assertFalse(row['has_padding'])
            self.assertEqual(row['production_compute_ctas_derived'], [128 if qkv else 64] * 4)
            self.assertEqual(row['compute_ctas_derived'], [0] * 4 if row['component'] == 'copy_reference'
                             else row['production_compute_ctas_derived'])
        with (out / 'summary.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row['swizzle'] == row['max_swizzle_size'] == '8' for row in rows))
        self.assertEqual({row['raster'] for row in rows}, {'along_m', 'along_n'})

    def test_epilogue_diagnostics_are_complete_and_never_performance_rows(self):
        self.make_epilogue_fixture()
        out = Path(self.directory.name) / 'summary'
        report = summary.summarize([self.root], out)
        self.assertEqual(report['performance_rows'], 2)
        run = report['runs'][0]
        probe = run['epilogue_diagnostics']
        self.assertFalse(probe['performance_accepted'])
        self.assertFalse(probe['durations_summed_across_ctas'])
        self.assertTrue(probe['base_store_includes_accumulator_wait'])
        self.assertEqual(run['profile_validation_records'], 64)
        self.assertEqual(run['profile_diagnostic_records']['epilogue_sample'], 600)
        self.assertEqual(run['profile_diagnostic_records']['epilogue_cta'], 64)
        self.assertEqual(len(probe['resources']), 12)
        self.assertEqual([row['mode'] for row in probe['modes']], list(summary.EPILOGUE_MODES))
        for mode in probe['modes']:
            self.assertEqual(mode['samples'], 50)
            self.assertEqual(len(mode['sample_lines']), 200)
        for rank in probe['ranks']:
            self.assertEqual((rank['compute_ctas'], rank['tiles'], rank['record_epoch']), (16, 16, 350))
            self.assertEqual(rank['first_ready_from_first_compute_cta_ns'], 133)
            self.assertEqual(rank['drain_fraction_of_own_cta_role']['p50'], .01)
        self.assertTrue(any(row['local_bytes'] == '16' for row in probe['resources']))
        with (out / 'summary.csv').open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 2)

    def test_epilogue_option_and_raw_contract_fail_closed(self):
        job, text = self.make_epilogue_fixture()
        for updates in (dict(qkv_epilogue_probe=1), dict(qkv_epilogue_probe=False), dict(profile=False),
                        dict(profile_detail='full'), dict(qkv_policy='m128n128'), dict(mpi=True)):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                summary.audit_log(text, job | updates)
        with self.assertRaisesRegex(ValueError, 'job/config'):
            summary.audit_log(text.replace(',qkv_epilogue_probe=1', '', 1), job)
        with self.assertRaisesRegex(ValueError, 'Unexpected QKV epilogue'):
            summary.audit_log(text.replace(',qkv_epilogue_probe=1', ',qkv_epilogue_probe=0', 1),
                              job | {'qkv_epilogue_probe': False})
        with self.assertRaisesRegex(ValueError, 'Unknown epilogue record'):
            summary.audit_log(text.replace('epilogue_resources,', 'epilogue_unregistered,', 1), job)
        for prefix, fields in summary.EPILOGUE_FIELDS.items():
            original = next(row for row in text.splitlines() if row.startswith(prefix + ','))
            for field in fields:
                raw_field = 'kind' if field == 'diagnostic_kind' else field
                changed = ','.join(part for part in original.split(',') if not part.startswith(raw_field + '='))
                with self.subTest(prefix=prefix, missing=field), self.assertRaisesRegex(ValueError, 'Missing/unknown epilogue'):
                    summary.audit_log(text.replace(original, changed, 1), job)
            with self.assertRaisesRegex(ValueError, 'Missing/unknown epilogue'):
                summary.audit_log(text.replace(original, original + ',surprise=1', 1), job)
            with self.assertRaisesRegex(ValueError, 'Invalid/accepted epilogue'):
                summary.audit_log(text.replace(original, original.replace('performance_accepted=0', 'performance_accepted=1'), 1), job)

    def test_epilogue_resources_and_samples_require_complete_rank_mode_domains(self):
        job, text = self.make_epilogue_fixture()
        for prefix in summary.EPILOGUE_FIELDS:
            original = next(row for row in text.splitlines() if row.startswith(prefix + ','))
            for changed in ('', original + '\n' + original, original.replace('rank=0', 'rank=3')):
                with self.subTest(prefix=prefix, change=changed), self.assertRaisesRegex(ValueError, 'Missing/duplicate'):
                    summary.audit_log(text.replace(original, changed, 1), job)
        for prefix in ('epilogue_resources', 'epilogue_sample'):
            original = next(row for row in text.splitlines() if row.startswith(prefix + ','))
            with self.assertRaisesRegex(ValueError, 'Missing/duplicate'):
                summary.audit_log(text.replace(original, original.replace('kind=production', 'kind=unknown'), 1), job)
        original = next(row for row in text.splitlines() if row.startswith('epilogue_resources,'))
        for old, new in (('record_bytes=96', 'record_bytes=128'), ('clock_unit=ns', 'clock_unit=cycles'),
                         ('tile_n=256', 'tile_n=128'), ('cluster_ctas=1', 'cluster_ctas=2'),
                         ('store_interval=issuing_lane_base_call_including_accumulator_wait', 'store_interval=pure_epilogue')):
            with self.subTest(new=new), self.assertRaisesRegex(ValueError, 'Epilogue clock|Epilogue resource'):
                summary.audit_log(text.replace(original, original.replace(old, new), 1), job)

    def test_epilogue_buffered_samples_use_epochs_not_validation_print_order(self):
        job, text = self.make_epilogue_fixture()
        rows = text.splitlines()
        first_sample = next(row for row in rows if row.startswith('epilogue_sample,'))
        self.assertGreater(rows.index(first_sample), max(i for i, row in enumerate(rows) if 'profile_phase=epilogue_' in row))
        for old, new in (('epoch=200', 'epoch=201'), ('sample=0', 'sample=1'), ('warmup=10', 'warmup=9'),
                         ('samples=50', 'samples=49'), ('process_layout=single_process', 'process_layout=mpi_one_process_per_gpu'),
                         ('host_launch=per_gpu_thread', 'host_launch=sequential'), ('event_ms=1', 'event_ms=0'),
                         ('final_sample_poisoned=0', 'final_sample_poisoned=1')):
            with self.subTest(new=new), self.assertRaisesRegex(ValueError, 'Epilogue epoch|Missing/duplicate'):
                summary.audit_log(text.replace(first_sample, first_sample.replace(old, new), 1), job)
        last = next(row for row in rows if row.startswith('epilogue_sample,') and ',sample=49,' in row)
        with self.assertRaisesRegex(ValueError, 'Epilogue epoch'):
            summary.audit_log(text.replace(last, last.replace('final_sample_poisoned=1', 'final_sample_poisoned=0'), 1), job)

    def test_epilogue_all_four_full_validation_phases_are_required_and_ordered(self):
        job, text = self.make_epilogue_fixture()
        for phase in summary.EPILOGUE_PHASES:
            for kind in ('correctness', 'route'):
                original = next(row for row in text.splitlines() if row.startswith(kind + ',') and 'profile_phase=' + phase + ',' in row)
                with self.subTest(phase=phase, kind=kind), self.assertRaisesRegex(ValueError, 'Missing profile final'):
                    summary.audit_log(text.replace(original + '\n', '', 1), job)
        old = next(row for row in text.splitlines() if row.startswith('correctness,') and 'profile_phase=epilogue_record,' in row)
        changed = text.replace(old + '\n', '', 1).replace('epilogue_resources,', old + '\nepilogue_resources,', 1)
        with self.assertRaisesRegex(ValueError, 'Epilogue final validation phase order'):
            summary.audit_log(changed, job)
        # The original instrumented/host-stages domains remain required too.
        old = next(row for row in text.splitlines() if row.startswith('route,') and 'profile_phase=instrumented,' in row)
        with self.assertRaisesRegex(ValueError, 'Missing profile final'):
            summary.audit_log(text.replace(old + '\n', '', 1), job)

    def test_epilogue_cta_counts_timestamps_sums_and_maxima_are_consistent(self):
        job, text = self.make_epilogue_fixture(seq_local=4096)
        valid = summary.audit_log(text, job)
        self.assertTrue(all(row['tiles'] == 256 and row['compute_ctas'] == 140 for row in valid['epilogue_diagnostics']['ranks']))
        original = next(row for row in text.splitlines() if row.startswith('epilogue_cta,'))
        values = dict(part.split('=', 1) for part in original.split(',')[1:])
        changes = dict(tile_count='3', epoch='351', first_m_tile='32', first_n_tile='8', first_batch='1',
                       first_store_begin=values['cta_end'], first_ready_after=values['first_store_begin'],
                       last_ready_after=values['cta_end'], store_ns_sum='41', store_ns_max='19',
                       drain_ns_sum='1000', drain_ns_max='21', cta_start=str(2**64))
        for field, value in changes.items():
            modified = original.replace(field + '=' + values[field], field + '=' + value)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Epilogue tile|epilogue CTA clock|Epilogue duration'):
                summary.audit_log(text.replace(original, modified, 1), job)

    def test_schedule_thresholds_downgrade_padding_and_grid_budget_are_exact(self):
        for m_tiles, expected_effective, expected_m in ((1, 1, 1), (2, 2, 2), (3, 4, 4),
                (5, 4, 8), (6, 8, 8), (8, 8, 8), (16, 8, 16), (20, 8, 24)):
            with self.subTest(m_tiles=m_tiles):
                self.make_fixture(calibrate=True, seq_local=128 * m_tiles, schedule=dict(max_swizzle_size=8))
                candidates = summary.audit_run(self.root)['candidates']
                for row in candidates:
                    n_tiles = 16 if row['direction'] == 'GEMM_A2A' else 8
                    self.assertEqual((row['max_swizzle_size'], row['effective_swizzle_size'], row['swizzle']),
                                     (8, expected_effective, expected_effective))
                    self.assertEqual((row['padded_m_tiles'], row['padded_n_tiles']), (expected_m, n_tiles))
                    self.assertEqual(row['has_padding'], expected_m != m_tiles)
                    self.assertEqual(row['work_tiles_derived'], m_tiles * n_tiles)
                    self.assertEqual(row['scheduled_work_tiles_derived'], expected_m * n_tiles)
                    self.assertEqual(row['production_compute_ctas_derived'], [min(140, expected_m * n_tiles)] * 4)
                    self.assertEqual(row['problem_gemm_flops'], 2 * row['m'] * row['n'] * row['k'])
        # Exercise padding on N independently from M, including the threshold
        # changing with min(Mtiles, Ntiles), without borrowing the implementation.
        schedule = dict(schema='explicit_v1', max_swizzle_size=8,
                        requested_rasters={'GEMM_A2A': 'heuristic'}, effective_rasters={'GEMM_A2A': 'along_m'})
        for n_tiles, expected_effective, expected_n in ((1, 1, 1), (2, 2, 2), (3, 4, 4), (5, 4, 8), (6, 8, 8)):
            row = summary.resolved_schedule(schedule, 'GEMM_A2A', 1024, n_tiles * 128, 128)
            self.assertEqual((row['effective_swizzle_size'], row['padded_n_tiles']), (expected_effective, expected_n))

    def test_schedule_requests_and_every_config_field_are_checked(self):
        self.make_fixture(schedule=dict(max_swizzle_size=8, qkv_raster='along_n', oproj_raster='heuristic'))
        job = json.loads((self.control / 'job.json').read_text())
        text = self.log.read_text()
        for updates in (dict(max_swizzle_size=3), dict(max_swizzle_size=True), dict(max_swizzle_size='8'),
                        dict(max_swizzle_size=4), dict(qkv_raster='auto'), dict(qkv_raster='along_m'),
                        dict(oproj_raster='along_m')):
            with self.subTest(updates=updates), self.assertRaisesRegex(ValueError, 'job|Job/config'):
                summary.audit_log(text, job | updates)
        original = text.splitlines()[0]
        for field in summary.SCHEDULE_CONFIG_FIELDS:
            parts = original.split(',')
            modified = ','.join(part for part in parts if not part.startswith(field + '='))
            with self.subTest(missing=field), self.assertRaisesRegex(ValueError, 'Missing explicit scheduling config'):
                summary.audit_log(text.replace(original, modified, 1), job)
        for field in ('qkv_effective_raster', 'oproj_effective_raster'):
            modified = original.replace(field + '=along_n', field + '=along_m')
            with self.subTest(effective=field), self.assertRaisesRegex(ValueError, 'Job/config scheduling mismatch'):
                summary.audit_log(text.replace(original, modified, 1), job)

    def test_schedule_every_rank_generation_and_component_requires_all_actual_fields(self):
        self.make_fixture(calibrate=True, seq_local=384, schedule=dict(max_swizzle_size=8))
        job = json.loads((self.control / 'job.json').read_text())
        text = self.log.read_text()
        for kind, component in (('candidate', None), ('component_resources', 'fused'),
                                ('component_resources', 'compute_reference'), ('component_resources', 'copy_reference')):
            for generation in (0, 1):
                original = next(row for row in text.splitlines() if row.startswith(kind + ',GEMM_A2A,') and
                                f'generation={generation},' in row and 'rank=3,' in row and
                                (component is None or f'component={component},' in row))
                parts = original.split(',')
                for field in summary.SCHEDULE_ROW_FIELDS:
                    item = next(part for part in parts if part.startswith(field + '='))
                    for wrong in (None, 'along_n' if field == 'raster' else str(int(item.split('=')[1]) + 1)):
                        modified = ','.join(part for part in parts if part != item) if wrong is None else \
                            original.replace(item, field + '=' + wrong)
                        with self.subTest(kind=kind, component=component, generation=generation, field=field, wrong=wrong):
                            with self.assertRaisesRegex(ValueError, 'scheduling resource|Scheduling resources'):
                                summary.audit_log(text.replace(original, modified, 1), job)

    def test_new_default_logs_cannot_hide_missing_fields_as_legacy(self):
        self.make_fixture(schedule=dict(max_swizzle_size=1, qkv_raster='heuristic', oproj_raster='heuristic'))
        job = json.loads((self.control / 'job.json').read_text())
        text = self.log.read_text()
        stripped = '\n'.join(','.join(part for part in row.split(',')
                             if part.split('=')[0] not in set(summary.SCHEDULE_CONFIG_FIELDS + summary.SCHEDULE_ROW_FIELDS))
                             for row in text.splitlines())
        with self.assertRaisesRegex(ValueError, 'Missing explicit scheduling config'):
            summary.audit_log(stripped, job)
        old_job = {key: value for key, value in job.items() if key not in summary.SCHEDULE_REQUEST_FIELDS}
        current_source_job = old_job | {'files': old_job['files'] |
            {'benchmarks/sm103/fused_bf16.cu': next(iter(summary.SCHEDULE_HARNESS_SHA256))}}
        with self.assertRaisesRegex(ValueError, 'Missing explicit scheduling config'):
            summary.audit_log(stripped, current_source_job)
        # A lone marker on an otherwise old-looking log also cannot downgrade.
        marker = stripped.replace(',state=resolved,', ',state=resolved,max_swizzle_size=1,', 1)
        with self.assertRaisesRegex(ValueError, 'Missing explicit scheduling config'):
            summary.audit_log(marker, old_job)

    def test_profile_scheduling_rejects_padding_but_accepts_actual_downshift(self):
        self.make_fixture(profile=True, seq_local=384, schedule=dict(max_swizzle_size=8))
        with self.assertRaisesRegex(ValueError, 'Profiling does not support swizzle-padded'):
            summary.audit_run(self.root)
        self.make_fixture(profile=True, seq_local=128, schedule=dict(max_swizzle_size=8))
        report = summary.audit_run(self.root)
        self.assertTrue(all(row['max_swizzle_size'] == 8 and row['swizzle'] == 1 for row in report['candidates']))
        job = json.loads((self.control / 'job.json').read_text())
        text = self.log.read_text()
        original = next(row for row in text.splitlines() if row.startswith('candidate,GEMM_A2A,') and 'rank=0,' in row)
        unscoped = original.replace(',generation=0', '')
        valid = text.replace('profile_host,GEMM_A2A,rank=0,', unscoped + '\nprofile_host,GEMM_A2A,rank=0,', 1)
        summary.audit_log(valid, job)
        with self.assertRaisesRegex(ValueError, 'Scheduling resources'):
            summary.audit_log(valid.replace(unscoped, unscoped.replace('max_swizzle_size=8', 'max_swizzle_size=1'), 1), job)

    def test_mpi_rank_stream_schedule_matches_job_for_both_payloads_and_all_components(self):
        self.make_mpi_fixture(calibrate=True, seq_local=1024,
                              schedule=dict(max_swizzle_size=8, qkv_raster='along_n', oproj_raster='along_m'))
        result = summary.audit_run(self.root)
        self.assertTrue(all(row['host_launch'] == 'mpi_process' and row['swizzle'] == 8 for row in result['candidates']))
        path = self.control / 'mpi-attempt1-rank-3.stdout.log'
        text = path.read_text()
        original = next(row for row in text.splitlines() if row.startswith('component_resources,A2A_GEMM,')
                        and 'component=compute_reference,' in row and 'generation=1,' in row)
        path.write_text(text.replace(original, original.replace('raster=along_m', 'raster=along_n'), 1))
        self.merge_mpi_fixture()
        self.assert_invalid('Scheduling resources')

    def test_diagnostic_run_cannot_publish_performance(self):
        self.make_fixture(diagnostic=True)
        out = Path(self.directory.name) / 'summary'
        report = summary.summarize([self.root], out)
        self.assertEqual(report['performance_rows'], 0)
        self.assertEqual(report['diagnostic_rows'], 2)
        self.assertEqual(len((out / 'summary.csv').read_text().splitlines()), 1)

    def test_profile_records_remain_separate_from_production_samples(self):
        self.make_fixture(profile=True)
        result = summary.audit_run(self.root)
        self.assertEqual(result['profile_diagnostic_records'], {'profile_host': 8, 'profile_cta': 1})
        self.assertEqual(len(result['candidates']), 2)

    def test_cta_only_and_full_diagnostics_have_distinct_peer_contracts(self):
        for detail in ('cta', 'full'):
            with self.subTest(detail=detail):
                self.make_fixture(profile=True, profile_schema='host_stages_v1', profile_detail=detail)
                result = summary.audit_run(self.root)
                self.assertEqual(result['profile_detail'], detail)
                self.assertEqual(result['profile_diagnostic_records'].get('profile_peer', 0), int(detail == 'full'))
                self.assertEqual(result['profile_validation_records'], 32)
                self.assertFalse(result['host_stage_diagnostics']['performance_accepted'])
                job = json.loads((self.control / 'job.json').read_text())
                text = self.log.read_text()
                other = 'full' if detail == 'cta' else 'cta'
                with self.assertRaisesRegex(ValueError, 'Profile detail config mismatch'):
                    summary.audit_log(text, job | {'profile_detail': other})
                for prefix in ('profile_cta,', 'profile_host,', 'host_stage,'):
                    rows = text.splitlines()
                    index = next(i for i, row in enumerate(rows) if row.startswith(prefix))
                    rows[index] = rows[index].replace(',profile_detail=' + detail, '')
                    with self.assertRaisesRegex(ValueError, 'Per-record profile detail mismatch'):
                        summary.audit_log('\n'.join(rows), job)
                if detail == 'cta':
                    text = text.replace('PASS:', 'profile_peer,rank=0,index=0,profile_detail=cta\nPASS:')
                    message = 'CTA-only diagnostics must not contain peer traces'
                else:
                    text = '\n'.join(row for row in text.splitlines() if not row.startswith('profile_peer,'))
                    message = 'Full diagnostics require peer traces'
                with self.assertRaisesRegex(ValueError, message):
                    summary.audit_log(text, job)

    def test_host_stages_remain_diagnostic_with_two_validation_phases(self):
        self.make_fixture(profile=True, profile_schema='host_stages_v1', host_launch='per_gpu_thread')
        out = Path(self.directory.name) / 'summary'
        report = summary.summarize([self.root], out)
        result = report['runs'][0]
        self.assertEqual(report['performance_rows'], 2)
        self.assertEqual(result['profile_diagnostic_records']['host_stage'], 400)
        self.assertEqual(result['profile_validation_records'], 32)
        self.assertFalse(result['host_stage_diagnostics']['performance_accepted'])
        self.assertEqual(len(result['host_stage_diagnostics']['groups']), 8)
        self.assertEqual(result['host_stage_diagnostics']['groups'][0]['p50_us']['api_us'], 10)
        self.assertEqual(len((out / 'summary.csv').read_text().splitlines()), 3)

    def test_host_stages_require_both_complete_validation_phases(self):
        self.make_fixture(profile=True, profile_schema='host_stages_v1')
        for phase in ('instrumented', 'host_stages'):
            with self.subTest(phase=phase):
                job = json.loads((self.control / 'job.json').read_text())
                text = self.make_log(job, 'host_stages_v1')
                text = '\n'.join(r for r in text.splitlines() if not (
                    r.startswith('route,GEMM_A2A,rank=3,') and 'profile_phase=' + phase in r)) + '\n'
                with self.assertRaisesRegex(ValueError, 'Missing profile final'):
                    summary.audit_log(text, job)

    def test_host_stage_corruptions_rejected(self):
        self.make_fixture(profile=True, profile_schema='host_stages_v1')
        job = json.loads((self.control / 'job.json').read_text())
        text = self.log.read_text()
        cases = (
            (',kind=production_kernel_diagnostic', ',kind=production_performance', 'diagnostic contract'),
            (',performance_accepted=0,clock=', ',performance_accepted=1,clock=', 'Incomplete/accepted'),
            (',stage_mask=255,', ',stage_mask=127,', 'Incomplete/accepted'),
            (',complete=1,api_us=', ',complete=0,api_us=', 'Incomplete/accepted'),
            (',api_us=10,', ',api_us=11,', 'duration sum'),
            (',api_begin_skew_us=15,', ',api_begin_skew_us=17,', 'rank skew'),
            (',all_enqueued_us=125,', ',all_enqueued_us=120,', 'enqueue completion'),
            (',epoch=120,host_launch=', ',epoch=119,host_launch=', 'rank epochs differ'),
            (',event_us=5000,kind=', ',event_us=nan,kind=', 'Nonfinite'),
            (',local_descriptor_us=0.1,', ',local_descriptor_us=2,', 'prepare subset'),
            (',profile_schema=host_stages_v1\n', ',profile_schema=future_v2\n', 'Unsupported profile schema'),
        )
        for old, new, message in cases:
            with self.subTest(field=old):
                self.assertIn(old, text)
                with self.assertRaisesRegex(ValueError, message):
                    summary.audit_log(text.replace(old, new, 1), job)

    def test_missing_duplicate_and_unversioned_host_stages_rejected(self):
        self.make_fixture(profile=True, profile_schema='host_stages_v1')
        job = json.loads((self.control / 'job.json').read_text())
        text = self.log.read_text()
        record = next(r for r in text.splitlines() if r.startswith('host_stage,'))
        for replacement in ('', record + '\n' + record + '\n'):
            with self.subTest(replacement=bool(replacement)):
                with self.assertRaisesRegex(ValueError, 'Missing/duplicate host-stage'):
                    summary.audit_log(text.replace(record + '\n', replacement, 1), job)
        with self.assertRaisesRegex(ValueError, 'Missing profile final'):
            summary.audit_log(text.replace(',profile_schema=host_stages_v1\n', '\n', 1), job)

    def test_host_stage_validation_order_and_duplicate_rejected(self):
        self.make_fixture(profile=True, profile_schema='host_stages_v1')
        job = json.loads((self.control / 'job.json').read_text())
        text = self.log.read_text()
        record = next(r for r in text.splitlines() if r.startswith('route,GEMM_A2A,rank=0,') and
                      'profile_phase=instrumented' in r)
        with self.assertRaisesRegex(ValueError, 'Missing profile final'):
            summary.audit_log(text.replace(record, record + '\n' + record, 1), job)
        moved = text.replace(record + '\n', '', 1).replace('PASS:', record + '\nPASS:', 1)
        with self.assertRaisesRegex(ValueError, 'must follow instrumented validation'):
            summary.audit_log(moved, job)

    def test_missing_rank_timing(self):
        self.change_log(lambda s: s.replace(',rank3_ms=5.30000019', '', 1))
        self.assert_invalid('rank timing')

    def test_forged_maxrank(self):
        self.change_log(lambda s: s.replace('maxrank_ms=5.30000019', 'maxrank_ms=9', 1))
        self.assert_invalid('maxrank differs')

    def test_nan_rejected(self):
        self.change_log(lambda s: s.replace('rank0_ms=5', 'rank0_ms=nan', 1))
        self.assert_invalid('Nonfinite')

    def test_missing_payload_rank_validation(self):
        self.change_log(lambda s: '\n'.join(r for r in s.splitlines() if not (
            r.startswith('route,GEMM_A2A') and 'generation=1,rank=3' in r)) + '\n')
        self.assert_invalid('Missing rank/payload')

    def test_unfinished_candidate(self):
        self.change_log(lambda s: '\n'.join(r for r in s.splitlines()
            if not r.startswith('candidate_verified,GEMM_A2A')) + '\n')
        self.assert_invalid('Candidate incomplete')

    def test_numerical_mismatch(self):
        self.change_log(lambda s: s.replace('mismatches=0', 'mismatches=1', 1))
        self.assert_invalid('Numerical mismatch')

    def test_missing_input_finite_count(self):
        self.change_log(lambda s: s.replace(',finite=2097152', '', 1))
        self.assert_invalid('missing finite')

    def test_duplicate_input_statistics(self):
        self.change_log(lambda s: s.replace(s.splitlines()[2], s.splitlines()[2] + '\n' + s.splitlines()[2], 1))
        self.assert_invalid('duplicate input')

    def test_wrong_recorded_percentile(self):
        self.change_log(lambda s: s.replace('p50_ms=5.30244994', 'p50_ms=9', 1))
        self.assert_invalid('p50_ms')

    def test_unknown_component_rejected_not_coerced_to_fused(self):
        self.change_log(lambda s: s.replace('summary,GEMM_A2A,', 'summary,GEMM_A2A,component=invented,', 1))
        self.assert_invalid('Unsupported measurement component')

    def test_v3_eventsync_collector_is_preserved_not_relabeled(self):
        self.change_log(lambda s: s.replace('per_epoch_rank_events_v2', 'per_epoch_rank_events_v3_eventsync'))
        result = summary.audit_run(self.root)
        self.assertEqual({r['collector'] for r in result['candidates']}, {'per_epoch_rank_events_v3_eventsync'})

    def test_unknown_collector_rejected(self):
        self.change_log(lambda s: s.replace('per_epoch_rank_events_v2', 'faster_unvalidated_sampler'))
        self.assert_invalid('Unsupported or incomplete sampling contract')

    def test_component_requires_explicit_job_config_opt_in(self):
        self.change_log(lambda s: s.replace('summary,GEMM_A2A,', 'summary,GEMM_A2A,component=copy_reference,', 1))
        self.assert_invalid('Component not enabled')

    def test_threaded_launch_is_distinct_and_record_mismatch_rejected(self):
        self.make_fixture(host_launch='per_gpu_thread')
        result = summary.audit_run(self.root)
        self.assertEqual(result['candidates'][0]['host_launch'], 'per_gpu_thread')
        self.change_log(lambda s: s.replace('summary,GEMM_A2A,', 'summary,GEMM_A2A,host_launch=sequential,', 1))
        self.assert_invalid('Per-record host launch mismatch')

    def test_calibration_has_three_separate_verified_components(self):
        self.make_fixture(calibrate=True)
        out = Path(self.directory.name) / 'summary'
        report = summary.summarize([self.root], out)
        self.assertTrue(report['independent_reference_components_present'])
        self.assertEqual(report['performance_rows'], 6)
        candidates = report['runs'][0]['candidates']
        self.assertEqual({c['component'] for c in candidates}, set(summary.COMPONENTS))
        self.assertTrue(all(len(c['component_resources']) == 8 for c in candidates))
        self.assertTrue(all(c['measurement_role'] == 'calibration' for c in candidates if c['component'] != 'fused'))

    def test_mpi_root_pass_precedes_later_rank_streams_without_global_time_assumption(self):
        self.make_mpi_fixture(calibrate=True)
        text = self.log.read_text()
        self.assertLess(text.index('PASS:'), text.index('device,rank=1,'))
        result = summary.audit_run(self.root)
        self.assertEqual(len(result['candidates']), 6)
        self.assertEqual({c['host_launch'] for c in result['candidates']}, {'mpi_process'})
        self.assertEqual({c['collector'] for c in result['candidates']}, {'mpi_rank_events_v1'})
        self.assertEqual({c['component'] for c in result['candidates']}, set(summary.COMPONENTS))

    def test_mpi_missing_rank_stream_and_hash_or_range_corruption_rejected(self):
        for field, value, pattern in (('sha256', 'd' * 64, 'bytes/hash/merged range'),
                                      ('bytes', 1, 'bytes/hash/merged range'),
                                      ('merged_begin', 0, 'bytes/hash/merged range'),
                                      ('present', False, 'rank stream missing'),
                                      ('rank_started', False, 'rank stream missing')):
            with self.subTest(field=field):
                self.make_mpi_fixture()
                path = self.control / 'mpi-logs-attempt1.json'
                manifest = json.loads(path.read_text())
                manifest['ranks'][2][field] = value
                self.write_json(path, manifest)
                self.repack()
                self.assert_invalid(pattern)
        self.make_mpi_fixture()
        (self.control / 'mpi-attempt1-rank-3.stdout.log').unlink()
        self.repack()
        self.assert_invalid('rank stream missing')

    def test_mpi_cross_node_and_collector_contract_rejected(self):
        for name, field, value, pattern in (
            ('mpi-runtime-attempt1.json', 'node', '09', 'runtime contract mismatch: node'),
            ('mpi-runtime-attempt1.json', 'collector', 'per_epoch_rank_events_v3_eventsync', 'collector'),
            ('mpi-logs-attempt1.json', 'collector', 'per_epoch_rank_events_v3_eventsync', 'merged-log'),
            ('fused-build.json', 'node', '09', 'Build receipt')):
            with self.subTest(name=name, field=field):
                self.make_mpi_fixture()
                path = self.control / name
                data = json.loads(path.read_text())
                data[field] = value
                self.write_json(path, data)
                self.repack()
                self.assert_invalid(pattern)
        self.make_mpi_fixture()
        path = self.control / 'mpi-attempt1-rank-0.stdout.log'
        path.write_text(path.read_text().replace('collector=mpi_rank_events_v1', 'collector=per_epoch_rank_events_v3_eventsync'))
        self.merge_mpi_fixture()
        self.assert_invalid('sampling contract')

    def test_mpi_root_pass_must_be_final_in_root_original_stream(self):
        self.make_mpi_fixture()
        self.move_mpi_record('PASS:')
        self.assert_invalid('final harness PASS')
        self.make_mpi_fixture()
        path = self.control / 'mpi-attempt1-rank-0.stdout.log'
        path.write_text(path.read_text() + 'root still running\n')
        self.merge_mpi_fixture()
        self.assert_invalid('final harness PASS')

    def test_mpi_nonroot_cannot_replace_root_metrics_or_acceptance(self):
        for prefix in ('sample,', 'warmup,', 'summary,', 'candidate_verified,', 'correctness,', 'route,'):
            with self.subTest(prefix=prefix):
                self.make_mpi_fixture()
                self.move_mpi_record(prefix)
                self.assert_invalid('nonroot emitted root-only')

    def test_mpi_native_records_cannot_move_between_rank_streams(self):
        for prefix in ('input,', 'candidate,', 'component_resources,'):
            with self.subTest(prefix=prefix):
                self.make_mpi_fixture(calibrate=True)
                root = self.control / 'mpi-attempt1-rank-0.stdout.log'
                record = next(row for row in root.read_text().splitlines() if row.startswith(prefix))
                nonroot = self.control / 'mpi-attempt1-rank-1.stdout.log'
                nonroot.write_text(nonroot.read_text() + record + '\n')
                self.merge_mpi_fixture()
                self.assert_invalid('native record belongs to another rank')

    def test_mpi_native_payload_order_checked_only_within_original_rank(self):
        self.make_mpi_fixture()
        path = self.control / 'mpi-attempt1-rank-1.stdout.log'
        rows = path.read_text().splitlines()
        indices = [i for i, row in enumerate(rows) if row.startswith('candidate,GEMM_A2A,')]
        rows[indices[0]], rows[indices[1]] = rows[indices[1]], rows[indices[0]]
        path.write_text('\n'.join(rows) + '\n')
        self.merge_mpi_fixture()
        self.assert_invalid('native payload generations are out of local order')

    def test_mpi_stderr_and_unowned_merged_bytes_cannot_supply_metrics(self):
        self.make_mpi_fixture()
        self.move_mpi_record('candidate_verified,', target=0, stream='stderr')
        self.assert_invalid('stderr contains harness evidence')
        for placement, pattern in (('prefix', 'merged gap'), ('suffix', 'merged tail')):
            with self.subTest(placement=placement):
                self.make_mpi_fixture()
                root = self.control / 'mpi-attempt1-rank-0.stdout.log'
                rows = root.read_text().splitlines()
                moved = next(row for row in rows if row.startswith('candidate_verified,'))
                rows.remove(moved)
                root.write_text('\n'.join(rows) + '\n')
                self.merge_mpi_fixture(**{placement: (moved + '\n').encode()})
                self.assert_invalid(pattern)

    def test_mpi_original_stream_tampering_is_rejected_even_with_new_archive_hash(self):
        self.make_mpi_fixture()
        path = self.control / 'mpi-attempt1-rank-2.stdout.log'
        path.write_text(path.read_text().replace('sms=148', 'sms=147', 1))
        self.repack()
        self.assert_invalid('bytes/hash/merged range')

    def test_selected_k128_requires_matching_compute_reference_resources(self):
        self.make_fixture(calibrate=True)
        job = json.loads((self.control / 'job.json').read_text())
        job.update(qkv_policy='m128n128k128', oproj_policy='m128n128k128')
        text = self.log.read_text().replace('tile=m128n128,', 'tile=m128n128k128,').replace('tile_k=64,', 'tile_k=128,')
        result = summary.audit_log(text, job)
        self.assertEqual({c['tile_k'] for c in result['candidates']}, {128})
        record = next(row for row in text.splitlines() if row.startswith('component_resources,')
                      and 'component=compute_reference,' in row)
        with self.assertRaisesRegex(ValueError, 'Component resources'):
            summary.audit_log(text.replace(record, record.replace('tile_k=128,', 'tile_k=64,'), 1), job)
        with self.assertRaisesRegex(ValueError, 'Resolved tile/resources'):
            summary.audit_log(text.replace('tile_k=128,', 'tile_k=64,', 1), job)

    def test_mixed_selected_tiles_are_resolved_per_candidate_not_last_log_row(self):
        self.make_fixture(calibrate=True)
        job = json.loads((self.control / 'job.json').read_text())
        job['qkv_policy'] = 'm128n128k128'
        text = '\n'.join(row.replace('tile=m128n128,', 'tile=m128n128k128,').replace('tile_k=64,', 'tile_k=128,')
                         if ',GEMM_A2A,' in row else row for row in self.log.read_text().splitlines())
        result = summary.audit_log(text, job)
        for candidate in result['candidates']:
            self.assertEqual(candidate['tile_k'], 128 if candidate['direction'] == 'GEMM_A2A' else 64)

    def test_calibration_missing_post_validation_rejected(self):
        self.make_fixture(calibrate=True)
        self.change_log(lambda s: '\n'.join(r for r in s.splitlines() if not (
            r.startswith('route,GEMM_A2A') and 'component=copy_reference,validation_phase=post,rank=3' in r)) + '\n')
        self.assert_invalid('Missing rank/payload')

    def test_calibration_false_compute_budget_rejected(self):
        self.make_fixture(calibrate=True)
        self.change_log(lambda s: s.replace('compute_budget=140', 'compute_budget=148', 1))
        self.assert_invalid('Component resources disagree')

    def test_copy_reference_cannot_claim_numeric_verification(self):
        self.make_fixture(calibrate=True)
        self.change_log(lambda s: s.replace('component=copy_reference,payload_generations=2,full_numeric=0',
                                           'component=copy_reference,payload_generations=2,full_numeric=1', 1))
        self.assert_invalid('Candidate incomplete')

    def test_second_payload_cannot_precede_measurement(self):
        def move(text):
            rows = text.splitlines()
            moved = [r for r in rows if 'generation=1' in r and r.startswith(('candidate,', 'correctness,', 'route,'))]
            rows = [r for r in rows if r not in moved]
            at = next(i for i, r in enumerate(rows) if r.startswith('warmup,'))
            return '\n'.join(rows[:at] + moved + rows[at:]) + '\n'
        self.change_log(move)
        self.assert_invalid('Second payload/final acceptance')

    def test_self_test_requires_every_real_fault(self):
        self.make_fixture(diagnostic=True)
        self.change_log(lambda s: '\n'.join(r for r in s.splitlines() if 'fault=local_nan,' not in r) + '\n')
        self.assert_invalid('Incomplete checker self-test')

    def test_cpu_oracle_option_requires_actual_confirmation(self):
        self.make_fixture(diagnostic=True)
        self.change_log(lambda s: '\n'.join(r for r in s.splitlines() if not r.startswith('validation_oracle,')) + '\n')
        self.assert_invalid('CPU oracle confirmation')

    def test_profile_job_requires_actual_profiles(self):
        self.make_fixture(profile=True)
        self.change_log(lambda s: '\n'.join(r for r in s.splitlines() if not r.startswith('profile_')) + '\n')
        self.assert_invalid('Missing profile host/CTA')

    def test_failed_terminal_receipt_rejected(self):
        self.fetched['state'] = 'failed'
        self.write_json(self.root / 'fetched.json', self.fetched)
        self.assert_invalid('successful terminal')

    def test_source_installed_mismatch(self):
        path = self.control / 'source-installed.json'
        data = json.loads(path.read_text())
        data['files']['CMakeLists.txt'] = 'd' * 64
        self.write_json(path, data)
        self.repack()
        self.assert_invalid('Installed source')

    def test_build_fingerprint_mismatch(self):
        path = self.control / 'fused-build.json'
        data = json.loads(path.read_text())
        data['build_inputs'] = 'd' * 64
        self.write_json(path, data)
        self.repack()
        self.assert_invalid('Build receipt')

    def test_extracted_log_tampering_rejected(self):
        self.log.write_text(self.log.read_text() + 'tampered\n')
        self.assert_invalid('differs from archive')

    def test_archive_tampering_rejected(self):
        archive = self.root / 'artifacts.tar.gz'
        archive.write_bytes(archive.read_bytes() + b'tampered')
        self.assert_invalid('SHA256 mismatch')

    def test_unsafe_archive_path_rejected(self):
        self.repack(unsafe=True)
        self.assert_invalid('Unsafe archive member')

    def test_symlink_evidence_rejected(self):
        other = Path(self.directory.name) / 'other.log'
        other.write_bytes(self.log.read_bytes())
        self.log.unlink()
        self.log.symlink_to(other)
        self.assert_invalid('Unsafe or missing evidence')

    def test_existing_output_not_overwritten(self):
        out = Path(self.directory.name) / 'summary'
        out.mkdir()
        (out / 'user.txt').write_text('preserve')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            summary.summarize([self.root], out)
        self.assertEqual((out / 'user.txt').read_text(), 'preserve')

    def test_duplicate_run_not_double_counted(self):
        with self.assertRaisesRegex(ValueError, 'duplicate input runs'):
            summary.summarize([self.root, self.control.parent], Path(self.directory.name) / 'summary')


if __name__ == '__main__':
    unittest.main()
