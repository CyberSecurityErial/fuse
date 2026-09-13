"""Read-only MXFP8 MPI audit for fused and independent C/R/Q candidate batches.

python3 scripts/summarize_sm103_mxfp8_fused.py [--component COMPONENT] RUN...
The legacy BF16 summary schema is NOT used for MXFP8 performance rows. Its
receipt/sample primitives are reused with a workspace-aware MPI adapter.
The default remains fused only. C/R/Q are separate calibration boundaries, not
additional fused results or primitive mixed-quantization service measurements.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import tempfile

import l20d
import summarize_sm103_fused as sf

BOUNDARIES = {
    'fused': 'prequantized_A + BF16_W_quantization + MXFP8_GEMM + BF16_A2A',
    'compute_reference': 'prequantized_A_and_W + MXFP8_GEMM_to_local_BF16',
    'copy_reference': 'ready_local_BF16 + BF16_A2A_without_GEMM_or_quantization',
    'quantize_reference': 'BF16_W_to_MXFP8 + panel_ready_reset_and_publication_without_GEMM_or_A2A',
}
OPROJ_BOUNDARIES = {
    'fused': 'MXFP8_A_and_SFA_A2A + BF16_W_quantization + MXFP8_GEMM_to_BF16',
    'compute_reference': 'prepared_MXFP8_A_and_W + GEMM_to_BF16_without_readiness_or_producers',
    'copy_reference': 'MXFP8_A_and_SFA_A2A_without_GEMM_or_weight_quantization',
    'producer_reference': 'MXFP8_A_and_SFA_A2A + BF16_W_quantization_without_GEMM',
}
MXFP8_COMPONENTS = tuple(dict.fromkeys((*BOUNDARIES, *OPROJ_BOUNDARIES)))


def window_effective_swizzle(configuration):
    """Derived OprojTileOrder width, distinct from CUTLASS's base width.

    AlongN clips the minor (M) swizzle to H; AlongM clips it to P.
    This describes traversal, not an observed performance improvement.
    """
    width = int(configuration['effective_swizzle_size'])
    h = int(configuration.get('oproj_m_window_tiles', 0))
    p = int(configuration.get('oproj_n_group_tiles', 0))
    # Match OprojTileOrder::make's disabled-window fallback exactly.
    if h <= 0 or p <= 0 or h & (h - 1) or p & (p - 1):
        return width
    return min(width, h if configuration['raster'] == 'along_n' else p)


def postprocess_configuration(job, config):
    """Normalize nullable CLI defaults, then bind requested semantics to execution."""
    qkv = job.get('qkv_postprocess') or 'none'
    rope = job.get('rope_policy') or 'qwen3'
    epsilon = job.get('norm_epsilon')
    epsilon = 1.e-6 if epsilon is None else epsilon
    sf.require(qkv in ('none', 'rope', 'qknorm_rope') and
               config.get('qkv_postprocess', 'none') == qkv,
               'QKV postprocess job/config mismatch')
    sf.require(rope in ('qwen3', 'llama31') and config.get('rope_policy', 'qwen3') == rope,
               'RoPE policy job/config mismatch')
    sf.close(float(config.get('norm_epsilon', 1.e-6)), float(epsilon), 'Norm epsilon')
    for key in ('qkv_postprocess_separate', 'oproj_postnorm', 'oproj_postnorm_overlap', 'oproj_postnorm_separate'):
        sf.require(int(config.get(key, 0)) == int(bool(job.get(key))),
                   'Postnorm job/config mismatch: ' + key)
    return dict(qkv=qkv, oproj_residual_rmsnorm=bool(job.get('oproj_postnorm')),
                qkv_separate=bool(job.get('qkv_postprocess_separate')),
                oproj_overlap=bool(job.get('oproj_postnorm_overlap')),
                oproj_separate=bool(job.get('oproj_postnorm_separate')),
                norm_epsilon=epsilon, rope_policy=rope)


def audit_auto_selection(rows, job, config):
    """Resolve requested zeros using native rank evidence, not inferred timings."""
    comm, qkv_policies, oproj_policies = l20d.fused_candidates(job)
    policies = oproj_policies if job.get('fused_direction') == 'oproj' else qkv_policies
    expected = [(c, policy) for policy in policies for c in comm]
    enabled = job.get('auto_mxfp8_comm', False)
    sf.require(type(enabled) is bool and int(config.get('auto_mxfp8_comm', 0)) == int(enabled),
               'MXFP8 auto job/config mismatch')
    records = [r for r in rows if r['kind'] == 'auto_comm']
    automatic = {i for i, (c, _) in enumerate(expected, 1) if c == 0}
    sf.require(bool(automatic) == enabled, 'MXFP8 auto request/candidate mismatch')
    if not enabled:
        sf.require(not records, 'Unexpected MXFP8 auto evidence')
        return expected, {}
    fields = {'kind', 'line', 'label', 'candidate', 'comm_sm', 'tile', 'rank',
              'model_version', 'requested_comm', 'resolved_comm', 'launch_comm',
              'query_us', 'repeat_query_us'}
    groups = {index: {} for index in automatic}
    world = int(job['world'])
    for row in records:
        sf.require(set(row) == fields, 'MXFP8 auto record fields mismatch')
        index, rank, budget = (int(row[k]) for k in ('candidate', 'rank', 'comm_sm'))
        sf.require(index in groups and 0 <= rank < world and rank not in groups[index],
                   'Duplicate or unexpected MXFP8 auto candidate/rank')
        sf.require(row['label'] == ('A2A_GEMM' if job.get('fused_direction') == 'oproj' else 'GEMM_A2A')
                   and row['tile'] == expected[index - 1][1]
                   and row['requested_comm'] == row['launch_comm'] == '0'
                   and int(row['resolved_comm']) == budget and 0 < budget < 148,
                   'MXFP8 auto launch/resolved budget mismatch')
        sf.require(row['model_version'] and not any(s in row['model_version'].lower() for s in ('empty','unmeasured')),
                   'MXFP8 auto has no measured calibration version')
        sf.finite(row['query_us'], 'auto query_us', 0)
        sf.finite(row['repeat_query_us'], 'auto repeat_query_us', 0)
        bindings = [r for r in rows if r['kind'] == 'candidate'
                    and int(r['candidate']) == index and int(r['rank']) == rank]
        sf.require(bindings and row['line'] < min(r['line'] for r in bindings),
                   'MXFP8 auto selection must precede native binding')
        groups[index][rank] = row
    metadata = {}
    for index, ranks in groups.items():
        sf.require(set(ranks) == set(range(world)), 'Missing MXFP8 auto rank')
        identities = {(r['comm_sm'], r['model_version']) for r in ranks.values()}
        sf.require(len(identities) == 1, 'MXFP8 auto rank budget/calibration disagreement')
        budget, version = next(iter(identities))
        expected[index - 1] = (int(budget), expected[index - 1][1])
        metadata[index] = dict(mode='runtime_model', requested_comm_ctas=0,
            launch_comm_ctas=0, resolved_comm_ctas=int(budget), model_version=version,
            rank_queries=[ranks[rank] for rank in range(world)])
    return expected, metadata


def audit_component(rows, job, config, shape, candidate_id, component, epilogue_n, *, repeat_launches=0):
    """Audit one boundary; shared production bindings do not become C/R/Q work."""
    sf.require(component in (OPROJ_BOUNDARIES if job.get('fused_direction') == 'oproj' else BOUNDARIES),
               'Unknown MXFP8 measurement component')
    calibrate, world = bool(job.get('calibrate')), int(job['world'])
    sf.require(component == 'fused' or calibrate, 'Reference requires calibration job')
    oproj = job.get('fused_direction') == 'oproj'
    label = 'A2A_GEMM' if oproj else 'GEMM_A2A'
    selected = [r for r in rows if r.get('label') == label
                and r.get('candidate') == str(candidate_id)]
    candidate = [r for r in selected if r.get('component') == component]
    resolved = [r for r in selected if r['kind'] == 'candidate' and r.get('component') == 'fused']
    sf.require(not any(r['kind'] == 'candidate' and r.get('component') != 'fused' for r in selected),
               'Reference cannot replace shared production binding')
    expected, _ = audit_auto_selection(rows, job, config)
    sf.require(1 <= candidate_id <= len(expected), 'Candidate index mismatch')
    expected_comm, expected_tile = expected[candidate_id - 1]
    sf.require(all(int(r['comm_sm']) == expected_comm and r['tile'] == expected_tile for r in selected),
               'Candidate differs from requested budget/tile')
    timing = sf.audit_timing(candidate, world, mpi=True, launch='graph')
    domains = ('correctness', 'route') if component in ('fused', 'producer_reference') else (
        ('route',) if component == 'copy_reference' else ('correctness',))
    checked = [r for r in candidate if r['kind'] in ('correctness', 'route')]
    checks = {}
    for row in checked + resolved:
        key = row['kind'], int(row['generation']), row.get('validation_phase', 'pre'), int(row['rank'])
        sf.require(key not in checks, 'Duplicate per-rank component check/binding')
        checks[key] = row
    expected = {('candidate', g, 'pre', rank) for g in (0, 1) for rank in range(world)}
    expected.update((kind, g, phase, rank) for kind in domains for g in (0, 1)
                    for phase in (('pre', 'post') if g == 0 else ('pre',)) for rank in range(world))
    sf.require(set(checks) == expected, 'Missing before/after-measurement and changed-payload checks')
    for row in checked:
        width = (shape['hidden'] if row['kind'] == 'correctness' else shape['q_width']) if oproj else shape['projection_width']
        if job.get('qkv_postprocess') and row['kind'] == 'route' and 'atol' in row:
            # Q/K RMSNorm has an independent FP64 numerical oracle. Do not
            # mislabel its tolerance comparison as the old byte-exact route.
            sf.require(not oproj and job['qkv_postprocess'] in ('rope', 'qknorm_rope'),
                       'Invalid postprocessing validation boundary')
            sf.audit_validation(dict(row, kind='correctness'), shape['seq_local'] * width)
        else:
            sf.audit_validation(row, shape['seq_local'] * width)
    first_warmup = min(r['line'] for r in candidate if r['kind'] == 'warmup')
    for kind in domains:
        for rank in range(world):
            sf.require(checks[kind, 0, 'pre', rank]['line'] < first_warmup
                       and checks[kind, 0, 'post', rank]['line'] > timing['summary_line']
                       and checks[kind, 1, 'pre', rank]['line'] > timing['summary_line'],
                       'Checks do not bracket measurement or changed payload precedes measurement')
    quant_checks = [r for r in candidate if r['kind'] in ('quant_validation', 'producer_validation')]
    if component in ('quantize_reference', 'producer_reference'):
        sf.require(all(r['kind'] == ('producer_validation' if component == 'producer_reference' else 'quant_validation')
                       for r in quant_checks), 'Wrong operand validation boundary')
        sf.require(Counter((int(r['generation']), r.get('validation_phase', 'pre')) for r in quant_checks)
                   == Counter(((0, 'pre'), (0, 'post'), (1, 'pre'))),
                   'Missing quantization represented-operand validation')
        for row in quant_checks:
            sf.require(row.get('method') == 'represented_operands_gemm' and row.get('prepare_repeated') == '0'
                       and row.get('compute_outside_timing') == '1' and 'rank' not in row,
                       'Quantization validation overwrites workspace or includes timed GEMM')
            generation, phase = int(row['generation']), row.get('validation_phase', 'pre')
            sf.require(all(row['line'] < checks['correctness', generation, phase, rank]['line']
                           for rank in range(world)), 'Quantization validation follows its numerical checks')
            sf.require(row['line'] < first_warmup if (generation, phase) == (0, 'pre')
                       else row['line'] > timing['summary_line'], 'Quantization validation enters measurement')
    else:
        sf.require(not quant_checks, 'Quantization validation attached to another boundary')
    sf.require(all(r.get('generation') == '0' for r in candidate
                   if r['kind'] in ('warmup', 'sample', 'summary')), 'Unexpected timed payload generation')
    verified = [r for r in candidate if r['kind'] == 'candidate_verified']
    acceptance = dict(payload_generations='2', full_numeric=str(int('correctness' in domains)),
                      full_route=str(int('route' in domains)), performance_accepted='1', launch='graph',
                      graph_epoch_mode=sf.GRAPH_EPOCH_MODE)
    sf.require(len(verified) == 1 and all(verified[0].get(k) == v for k, v in acceptance.items()),
               'Missing final component acceptance')
    sf.require(verified[0]['line'] > max(r['line'] for r in checked), 'Acceptance precedes checks')
    preparation = sf.audit_graph_preparation(candidate, timing, checks, domains, world,
                                             repeat_launches=repeat_launches)
    if component != 'fused':
        sf.require(all(r['first_epoch'] == 1 for r in preparation), 'Reference epoch was not reset')

    devices = [r for r in rows if r['kind'] == 'device']
    sf.require(Counter(int(r['rank']) for r in devices) == Counter(range(world))
               and all(r.get('runtime_cc') == '10.3' and int(r['sms']) == 148 for r in devices),
               'Missing rank/device budget identity')
    schedule = sf.resolved_schedule(sf.audit_schedule_config(job, config, rows), label,
                                    shape['seq_local'], shape['hidden'] if oproj else shape['projection_width'], 256)
    compute = min(schedule['scheduled_work_tiles_derived'], 148 - expected_comm)
    sf.require(compute > 0 and expected_comm > 0, 'Invalid compute/communication budget')
    fields = ('comm_sm', 'tile', 'tile_m', 'tile_n', 'tile_k', 'raster', 'max_swizzle_size',
              'effective_swizzle_size', 'scheduled_compute_ctas', 'dynamic_smem')
    configs = {tuple(r[k] for k in fields) for r in resolved}
    sf.require(len(configs) == 1, 'Resolved configurations disagree between ranks/payloads')
    window = {key: job.get(key, 0) for key in ('oproj_m_window_tiles', 'oproj_n_group_tiles')}
    for row in resolved:
        sf.require(row.get('state') == 'resolved' and tuple(int(row[k]) for k in ('tile_m', 'tile_n', 'tile_k'))
                   == (128, 256, 128) and int(row['threads']) == 256 and int(row['dynamic_smem']) > 0,
                   'Production MXFP8 tile/resources mismatch')
        sf.audit_schedule_row(row, schedule, compute)
        sf.require(all(int(row.get(key, 0)) == value for key, value in window.items()),
                   'Production OProj window differs from requested configuration')
    resources = [r for r in candidate if r['kind'] == 'component_resources']
    sf.require(Counter((int(r['generation']), int(r['rank'])) for r in resources)
               == (Counter((g, r) for g in (0, 1) for r in range(world)) if calibrate else Counter()),
               'Missing/unexpected component resources')
    for row in resources:
        production = checks['candidate', int(row['generation']), 'pre', int(row['rank'])]
        component_compute = compute if component in ('fused', 'compute_reference') else 0
        sf.audit_schedule_row(row, schedule, component_compute)
        sf.require(all(int(row.get(key, 0)) == value for key, value in window.items()),
                   'Reference OProj window differs from production configuration')
        sf.require(all(row[k] == production[k] for k in ('tile_m', 'tile_n', 'tile_k'))
                   and int(row['compute_budget']) == 148 - expected_comm
                   and int(row['scheduled_compute_ctas']) == component_compute
                   and int(row['scheduled_comm_ctas']) == (0 if component == 'compute_reference' else expected_comm)
                   and row['production_threads'] == production['threads']
                   and row['production_dynamic_smem'] == production['dynamic_smem']
                   and row['reference_resources'] == ('not_applicable' if component == 'fused' else 'unknown')
                   and row.get('reference_precision') == 'mxfp8'
                   and int(row['epilogue_n']) == epilogue_n
                   and row.get('reference_resource_contract') == 'production_threads_and_dynamic_smem',
                   'Component resources/precision differ from production contract')
        if component != 'fused':
            sf.require(row.get('reference_weight_preparation') == (
                'inside_timing' if component in ('quantize_reference','producer_reference') else 'outside_timing'),
                'Reference weight preparation boundary mismatch')
    configuration = dict(zip(fields, next(iter(configs)))) | window
    configuration['window_effective_swizzle_size_derived'] = window_effective_swizzle(configuration)
    return timing, configuration, resources, preparation


def audit_mpi(job, records, data, attempt):
    runtime = records[f'mpi-runtime-attempt{attempt}.json']
    manifest = records[f'mpi-logs-attempt{attempt}.json']
    world = int(job['world'])
    collector = 'mpi_graph_rank_events_v1'
    for key, value in dict(schema='sm103_mpi_runtime_v1', node=job['node'], world=world,
                           process_layout='mpi_one_process_per_gpu', host_launch='mpi_process',
                           launch='graph', collector=collector,
                           boundary='mpi_graph_maxrank_cudaevent',
                           graph_epoch_mode=sf.GRAPH_EPOCH_MODE).items():
        sf.require(runtime.get(key) == value, f'MPI runtime mismatch: {key}')
    sf.require(runtime['overrides'] == records['environment.json']['mpi_toolchain']['overrides']
               and runtime['overrides'].get('UCX_TLS') == 'sm,self', 'MPI toolchain mismatch')
    argv = runtime['argv']
    prefix = str(l20d.workspace_path(job['workspace']) / 'toolchain/mpich-5.0.1.post1')
    sf.require(argv[:5] == [prefix+'/bin/mpiexec', '-launcher', 'fork', '-n', str(world)]
               and records['fused-build.json']['binary'] in argv, 'MPI launch identity mismatch')
    sf.require(argv.count('--launch') == 1 and argv[argv.index('--launch')+1] == 'graph',
               'MPI launch must be graph')
    merged = data[f'attempt{attempt}.log']
    sf.require(manifest['schema'] == 'sm103_mpi_rank_logs_v1' and manifest['complete'] is True
               and manifest['collector'] == collector
               and manifest['ordering'] == 'rank_then_stream_not_global_chronological'
               and manifest['merged_log'] == f'attempt{attempt}.log'
               and manifest['merged_sha256'] == sf.digest(merged), 'MPI log manifest mismatch')
    streams = manifest['ranks']
    expected = [(rank, stream) for rank in range(world) for stream in ('stdout', 'stderr')]
    sf.require([(r['rank'],r['stream']) for r in streams] == expected, 'MPI rank streams incomplete')
    previous = 0
    for entry in streams:
        rank, stream = entry['rank'], entry['stream']
        name = f'mpi-attempt{attempt}-rank-{rank}.{stream}.log'
        raw = data[name]
        begin, end = entry['merged_begin'], entry['merged_end']
        sf.require(entry['path'] == name and entry['present'] is True and entry['rank_started'] is True
                   and entry['sha256'] == sf.digest(raw) and entry['bytes'] == len(raw)
                   and previous <= begin <= end <= len(merged) and merged[begin:end] == raw,
                   'MPI rank evidence differs from manifest/merged bytes')
        outside, diag = sf.parse_log(merged[previous:begin].decode(), completion='none', components=MXFP8_COMPONENTS)
        sf.require(not outside and not diag, 'Unowned MPI log evidence')
        rows, diag = sf.parse_log(raw.decode(), completion='last' if rank == 0 and stream == 'stdout' else 'none',
                                  components=MXFP8_COMPONENTS)
        if stream == 'stderr':
            sf.require(not rows and not diag, 'Harness evidence in stderr')
        else:
            devices = [r for r in rows if r['kind'] == 'device']
            sf.require(len(devices) == 1 and int(devices[0]['rank']) == rank, 'Wrong native rank identity')
            for row in rows:
                native = row['kind'] in ('device','input','candidate','component_resources','graph_prepare','auto_comm')
                sf.require(int(row['rank']) == rank if native else rank == 0, 'Wrong MPI evidence owner')
        previous = end
    outside, diag = sf.parse_log(merged[previous:].decode(), completion='none', components=MXFP8_COMPONENTS)
    sf.require(not outside and not diag, 'Unowned MPI log tail')


def audit_run(directory, candidate_id=1, component='fused'):
    sf.require(component in MXFP8_COMPONENTS, 'Unknown MXFP8 measurement component')
    directory = Path(directory).resolve()
    requested = sf.json_bytes(sf.read_bytes(directory/'job.json', directory))
    sf.require(requested.get('mxfp8') and requested.get('mpi') and not requested.get('profile')
               and requested.get('fused_direction') in ('qkv', 'oproj') and requested.get('fused_launch') == 'graph'
               and not requested.get('quick'), 'Not formal MXFP8 forward MPI')
    oproj = requested.get('fused_direction') == 'oproj'
    sf.require(not oproj or component in OPROJ_BOUNDARIES, 'Unknown OProj boundary')
    sf.require(component == 'fused' or requested.get('calibrate'), 'Reference requires calibration job')
    original_workspace, original_mpi_audit = str(l20d.WORKSPACE), sf.audit_mpi_receipts
    try:
        l20d.configure_workspace(requested['workspace'])
        sf.audit_mpi_receipts = audit_mpi
        job, receipts, data, evidence = sf.read_receipts(directory)
    finally:
        sf.audit_mpi_receipts = original_mpi_audit
        l20d.configure_workspace(original_workspace)
    attempt = receipts['status.json']['attempt']
    rows, diagnostics = sf.parse_log(data[f'attempt{attempt}.log'].decode(), completion='any', components=MXFP8_COMPONENTS)
    sf.require(not diagnostics, 'Profiling records in formal measurement')
    configs = [r for r in rows if r['kind'] == 'config']
    sf.require(len(configs) == 1, 'Missing/duplicate config')
    config, shape = configs[0], l20d.fused_geometry(job)
    world = int(job['world'])
    for field in ('world','global_seq','seq_local','hidden','q_heads','kv_heads','head_dim'):
        sf.require(int(config[field]) == shape[field], 'Geometry mismatch: '+field)
    sf.require(config['input_generator'] == 'gpu_philox' and config['sampling_mode'] == 'formal_10_50'
               and int(config['warmup']) >= 10 and int(config['samples']) == 50
               and int(config['seed']) == job.get('seed',20260906) and config['profile'] == '0'
               and config['launch'] == 'graph' and config['process_layout'] == 'mpi_one_process_per_gpu'
               and config['fused_direction'] == requested['fused_direction'], 'Measurement config mismatch')
    postprocess = postprocess_configuration(job, config)
    sf.require(int(config.get('calibrate', 0)) == int(bool(job.get('calibrate'))),
               'Calibration job/config mismatch')
    sf.require(all(int(config.get(key, 0)) == job.get(key, 0)
                   for key in ('oproj_m_window_tiles', 'oproj_n_group_tiles')),
               'OProj window job/config mismatch')
    rank_swizzle = 'rank_n_band_v1' if job.get('qkv_rank_swizzle') else 'off'
    sf.require(config.get('qkv_rank_swizzle', 'off') == rank_swizzle, 'Rank swizzle job/config mismatch')
    comm, qkv, op = l20d.fused_candidates(job)
    policies = op if oproj else qkv
    sf.require(int(config['candidates']) == len(comm)*len(policies) and 1 <= candidate_id <= len(comm)*len(policies),
               'Candidate count/index mismatch')
    precision = [line for line in data[f'attempt{attempt}.log'].decode().splitlines()
                 if line.startswith('precision,mxfp8,')]
    sf.require(len(precision) == 1, 'Missing precision metadata')
    fields_precision = dict(item.split('=',1) for item in precision[0].split(',')[2:])
    sf.require(len(fields_precision) == len(precision[0].split(',')) - 2, 'Duplicate precision metadata')
    for key, value in dict(input='mxfp8', weight='bf16', output='bf16', accumulator='fp32',
                           scale='ue8m0', group_k='32', tile='128x256x128',
                           includes_activation_quantization='0').items():
        sf.require(fields_precision.get(key) == value, 'Precision metadata mismatch: ' + key)
    epilogue_n = int(fields_precision.get('epilogue_n',64))
    sf.require(epilogue_n == (job.get('mxfp8_epilogue_n') or 64), 'Epilogue metadata mismatch')
    sf.require(fields_precision['includes_weight_quantization'] == '1', 'Not dynamic weight boundary')
    if job.get('oproj_postnorm_separate') or job.get('qkv_postprocess_separate'):
        sf.require(fields_precision.get('kernel_nodes') == '2', 'Separate boundary must contain two kernels')
    sf.require(fields_precision.get('weight_preparation') == (job.get('mxfp8_weight_preparation') or 'comm'),
               'Weight preparation job/config mismatch')
    # Original BF16 input validation still covers master setup and both activation
    # generations. MXFP8 adds regenerated master weights in both generations.
    residuals = [r for r in rows if r['kind']=='input' and r.get('label')=='OProj-residual']
    reference_refinements = []
    repeat_checks = []
    if job.get('oproj_postnorm'):
        root_text = data[f'mpi-attempt{attempt}-rank-0.stdout.log'].decode()
        repeats = [dict(field.split('=',1) for field in line.split(',')[2:])
                   for line in root_text.splitlines() if line.startswith('postnorm_repeat,A2A_GEMM,')]
        if repeats:
            seen = set()
            for r in repeats:
                key = (int(r['candidate']), int(r['generation']), r.get('validation_phase','pre'),
                       int(r['repeat']), int(r['rank']))
                sf.require(key not in seen, 'Duplicate postnorm repeatability check')
                seen.add(key)
                sf.require(int(r['checked']) == 2*shape['seq_local']*shape['hidden'] and
                           int(r['bitwise_mismatches']) == 0 and
                           r['outputs'] == 'normalized_and_residual_sum' and
                           r['reference'] == 'pre_measurement_snapshot', 'Postnorm bitwise repeatability failed')
                if key[0] == candidate_id: repeat_checks.append(r)
            expected_repeats = {(c,g,'pre',i,r) for c in range(1,len(comm)*len(policies)+1)
                                for g in (0,1) for i in range(3) for r in range(world)}
            expected_repeats |= {(c,0,'post',0,r) for c in range(1,len(comm)*len(policies)+1)
                                 for r in range(world)}
            sf.require(seen == expected_repeats, 'Incomplete postnorm repeatability coverage')
        for rank in range(world):
            text = data[f'mpi-attempt{attempt}-rank-{rank}.stdout.log'].decode()
            checks = [dict(field.split('=', 1) for field in line.split(',')[1:])
                      for line in text.splitlines() if line.startswith('postnorm_residual,')]
            sf.require(len(checks) == 3 * len(comm) * len(policies),
                       'Missing residual-output pre/post-measurement or changed-payload checks')
            for check in checks:
                sf.require(int(check['rank']) == rank and int(check['mismatches']) == 0 and
                           int(check['checked']) == shape['seq_local'] * shape['hidden'],
                           'Residual-output validation failed or incomplete')
                sf.finite(check['max_abs'], 'residual-output max_abs', 0)
            for line in text.splitlines():
                if not line.startswith('postnorm_reference_refinement,'):
                    continue
                refinement = dict(field.split('=', 1) for field in line.split(',')[1:])
                sf.require(int(refinement['rank']) == rank and
                           0 <= int(refinement['row']) < shape['seq_local'] and
                           refinement['oracle'] == 'fp64_dot_and_norm' and
                           refinement['scope'] == 'whole_row' and
                           float(refinement['atol']) == float(refinement['rtol']) == 0.01 and
                           int(refinement['before_mismatches']) > 0 and
                           int(refinement['after_mismatches']) >= 0 and
                           int(refinement['nonfinite']) == 0,
                           'Invalid high-precision reference refinement')
                reference_refinements.append(refinement)
        sf.require(Counter(int(r['rank']) for r in residuals) == Counter(range(world)),
                   'Missing/duplicate postnorm residual input')
        for r in residuals:
            count = shape['seq_local'] * shape['hidden']
            sf.require(int(r['count']) == int(r['finite']) == count and
                       int(r['seed']) == int(config['seed']) + 71 + int(r['rank']) and
                       r['generator'] == 'gpu_philox' and float(r['min']) < 0 < float(r['max']) and
                       0.99 < float(r['nonzero_fraction']) <= 1 and float(r['rms']) > 0,
                       'Invalid postnorm residual statistics')
    else:
        sf.require(not residuals, 'Unexpected postnorm input on an ordinary boundary')
    sf.audit_inputs([r for r in rows if not(r['kind']=='input' and
        r.get('label') in ('MXFP8-weight','OProj-residual'))], config, shape)
    weights = [r for r in rows if r['kind']=='input' and r.get('label')=='MXFP8-weight']
    sf.require(Counter((int(r['generation']),int(r['rank'])) for r in weights)
               == Counter((g,r) for g in (0,1) for r in range(world)), 'Missing regenerated master weights')
    for r in weights:
        count = shape['hidden'] * (shape['q_width'] if oproj else shape['projection_width'])
        sf.require(int(r['count']) == count and int(r['finite']) == count
                   and int(r['seed']) == (int(config['seed'])+int(r['generation'])*100003+11)%2**32
                   and r['generator']=='gpu_philox' and r['algorithm']=='curand_philox4x32_10'
                   and r['mapping']=='thread_subsequence_v1' and r['distribution']=='uniform'
                   and int(r['blocks'])==256 and int(r['threads'])==256 and int(r['offset'])==0,
                   'Regenerated master weight contract mismatch')
        sf.close(float(r['lower']),-.02,'weight lower');sf.close(float(r['upper']),.02,'weight upper')
        bound = sf.bf16_round(.02)
        sf.require(-bound-1e-9 <= float(r['min']) < float(r['max']) <= bound+1e-9
                   and 0<float(r['nonzero_fraction'])<=1 and float(r['rms'])>0 and float(r['std'])>0,
                   'Degenerate master weight statistics')
        sf.close(float(r['std'])**2,max(0.,float(r['rms'])**2-float(r['mean'])**2),'weight variance')
    timing, configuration, resources, preparation = audit_component(
        rows, job, config, shape, candidate_id, component, epilogue_n,
        repeat_launches=3 if repeat_checks else 0)
    _, automatic = audit_auto_selection(rows, job, config)
    telemetry = sf.audit_telemetry(data['gpu-telemetry.csv'],receipts['gpu-before.json'],job)
    n, k = (shape['hidden'], shape['q_width']) if oproj else (shape['projection_width'], shape['hidden'])
    flops = 2*shape['seq_local']*n*k
    executed_flops = flops if component in ('fused', 'compute_reference') else 0
    return dict(run_id=job['run_id'],candidate_id=candidate_id,component=component,
        measurement_role=('separate_postprocess_reference' if job.get('oproj_postnorm_separate') or
                          job.get('qkv_postprocess_separate') else
                          'production' if component == 'fused' else 'calibration'),
        epilogue_n=epilogue_n,experiment=job['experiment'],world=world,global_seq=shape['global_seq'],
        m=shape['seq_local'],n=n,k=k,
        q_heads=shape['q_heads'],kv_heads=shape['kv_heads'],head_dim=shape['head_dim'],
        qkv_rank_swizzle=rank_swizzle,weight_preparation=fields_precision['weight_preparation'],
        precision={'copy_reference': 'MXFP8_A_and_SFA_copy' if oproj else 'BF16_copy',
                   'producer_reference': 'MXFP8_A_and_SFA_copy_plus_BF16_W_quantization',
                   'quantize_reference': 'BF16_to_MXFP8_E4M3_UE8M0'}.get(
            component, 'MXFP8_E4M3_UE8M0_accFP32_BF16out'),
        boundary=(OPROJ_BOUNDARIES if oproj else BOUNDARIES)[component] + (
            '+BF16_residual_add+full_hidden_RMSNorm' if job.get('oproj_postnorm') else
            '+' + postprocess['qkv'] if postprocess['qkv'] != 'none' else ''),
        postprocess=postprocess,
        reference_refinements_run=reference_refinements,
        bitwise_repeatability=repeat_checks,
        executed_gemm_flops=executed_flops,
        p50_ms=timing['p50_ms'],p95_ms=timing['p95_ms'],
        pflops_per_rank=executed_flops/timing['p50_ms']/1e12 if executed_flops else None,
        half_drift=timing['half_drift'],selected_round=timing['selected_round'],warmup_calls=timing['warmup_calls'],
        raw_maxrank_ms=timing['rounds'][-1]['maxrank_ms'],configuration=configuration,
        communication_selection=automatic.get(candidate_id, dict(mode='explicit')),
        component_resources=resources,graph_preparation=preparation,
        source_id=job['source_id'],quantization_sha256=job['files']['csrc/operators/sm103/detail/quantization.cuh'],
        binary_sha256=receipts['fused-build.json']['binary_sha256'],
        environment_fingerprint=receipts['environment.json']['fingerprint'],
        artifact_sha256=evidence['artifacts.tar.gz']['sha256'],telemetry=telemetry,
        validation='both_payloads_and_postmeasurement_full_' + (
            'numeric_and_route' if component == 'fused' else 'route' if component == 'copy_reference' else 'numeric'),
        binary_identity='remote_build_attestation; local binary bytes not available')


def build_service_summary(results):
    """Group audited whole-component times, never reinterpret them as services."""
    sf.require(results, 'No component results')
    binaries = {r['binary_sha256'] for r in results}
    environments = {r['environment_fingerprint'] for r in results}
    sf.require(len(binaries) == len(environments) == 1, 'Cannot mix binary/environment identities')
    geometry_fields = ('m', 'n', 'k', 'world', 'global_seq', 'q_heads', 'kv_heads', 'head_dim',
                       'epilogue_n', 'qkv_rank_swizzle', 'weight_preparation')
    config_fields = ('comm_sm', 'tile', 'tile_m', 'tile_n', 'tile_k', 'raster', 'max_swizzle_size',
                     'effective_swizzle_size', 'scheduled_compute_ctas', 'dynamic_smem')
    groups = {}
    for result in results:
        component = result['component']
        sf.require(component in BOUNDARIES and result['boundary'] == BOUNDARIES[component],
                   'Unregistered component boundary')
        key = tuple(result[field] for field in geometry_fields) + tuple(
            result['configuration'][field] for field in config_fields)
        if key not in groups:
            groups[key] = {field: result[field] for field in geometry_fields} | {
                'configuration': dict(result['configuration']), 'components': {}}
        components = groups[key]['components']
        sf.require(component not in components, 'Duplicate physical candidate/component; select one run explicitly')
        components[component] = result
    rows = []
    for row in groups.values():
        components = row['components']
        sf.require(set(components) == set(BOUNDARIES), 'Missing F/C/R/Q for a physical candidate')
        sf.require(len({(r['run_id'], r['candidate_id']) for r in components.values()}) == 1,
                   'Candidate boundaries must come from the same audited run')
        rows.append(row)
    rows.sort(key=lambda r: (r['n'], r['k'], r['global_seq'], r['world'], int(r['configuration']['comm_sm']),
                            r['epilogue_n'], r['configuration']['raster'],
                            int(r['configuration']['effective_swizzle_size'])))
    return dict(schema='sm103_mxfp8_component_services_v1', calibration_scope='aggregate_only',
        note='F/C/R/Q are independently timed complete boundaries. They cannot populate the 16 primitive '
             'model coefficients or identify mixed DMA/quantization latency; Q/R execute no GEMM FLOPs.',
        binary_sha256=next(iter(binaries)), environment_fingerprint=next(iter(environments)),
        source_ids=sorted({r['source_id'] for r in results}),
        run_ids=sorted({r['run_id'] for r in results}), rows=rows)


def service_markdown(summary):
    lines = ['# MXFP8 F/C/Q/R aggregate measurements', '',
             'Scope: **aggregate_only**. These complete-boundary times cannot fill the 16 primitive model '
             'coefficients. Q/R have no GEMM PFLOPS. All times are Graph MPI max-rank p50, in μs; '
             'F/C throughput is PFLOPS per rank.', '',
             'F: fused dynamic-weight quantization + GEMM + A2A; C: prequantized-operand GEMM; '
             'Q: weight quantization + ready reset/publication; R: BF16 output A2A. '
             'Q numerical validation executes C outside timing without repeating weight preparation.', '',
             '| S | CP | M × N × K | Comm CTA | Epi / raster / swizzle | F μs | C μs | Q μs | R μs | F P | C P |',
             '|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for row in summary['rows']:
        c, r = row['components'], row['configuration']
        times = [c[key]['p50_ms'] * 1000 for key in ('fused', 'compute_reference', 'quantize_reference', 'copy_reference')]
        lines.append(f"| {row['global_seq']} | {row['world']} | {row['m']} × {row['n']} × {row['k']} | "
                     f"{r['comm_sm']} | {row['epilogue_n']} / {r['raster']} / {r['effective_swizzle_size']} | "
                     + ' | '.join(f'{value:.3f}' for value in times) +
                     f" | {c['fused']['pflops_per_rank']:.3f} | {c['compute_reference']['pflops_per_rank']:.3f} |")
    lines += ['', f"Binary SHA256: `{summary['binary_sha256']}`", '',
              f"Environment fingerprint: `{summary['environment_fingerprint']}`", '',
              'The JSON companion retains every audited component, source identity and 50 raw selected-round samples.', '']
    return '\n'.join(lines)


def _write_summary_pair(directory, summary, basename, markdown):
    """Stage both small outputs after full audit; restore prior pair on write error."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    contents = {basename + '.json': json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n',
                basename + '.md': markdown(summary)}
    paths = {name: directory / name for name in contents}
    for path in paths.values():
        sf.require(not path.is_symlink() and (not path.exists() or path.is_file()), 'Unsafe summary output')
    with tempfile.TemporaryDirectory(prefix='.' + basename + '-', dir=directory) as temporary:
        temporary = Path(temporary)
        existed, replaced = {}, []
        for name, text in contents.items():
            staged = temporary / name
            with staged.open('w') as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            existed[name] = paths[name].exists()
            if existed[name]:
                (temporary / (name + '.old')).write_bytes(paths[name].read_bytes())
        try:
            for name, path in paths.items():
                os.replace(temporary / name, path)
                replaced.append(name)
        except OSError:
            for name in reversed(replaced):
                if existed[name]:
                    os.replace(temporary / (name + '.old'), paths[name])
                else:
                    paths[name].unlink()
            raise
    return tuple(paths.values())


def write_service_summary(directory, summary):
    return _write_summary_pair(directory, summary, 'services-current', service_markdown)


COMPARISON_KEY = ('m', 'n', 'k', 'world', 'global_seq')
# Presentation aliases only; never imported by a runtime model or selector.
MODEL_LABELS = {(4096, 2048): ('QwenDense',),
    (10240, 8192): ('Qwen2.5 72B', 'Llama 3.1 70B'),
    (18432, 16384): ('Llama 3.1 405B',), (43008, 14336): ('BLOOM 176B',),
    (36864, 7168): ('Kimi K3 QKV-only',), (9216, 4096): ('Qwen3 235B',)}


def _comparison_key(row):
    sf.require(all(type(row.get(k)) is int and row[k] > 0 for k in COMPARISON_KEY),
               'Invalid comparison geometry')
    sf.require(row['m'] * row['world'] == row['global_seq'], 'Comparison sequence/CP mismatch')
    return tuple(row[k] for k in COMPARISON_KEY)


def _check_confirmation(row, entry):
    """Check retained confirmation arithmetic/identity, not recreate lost receipts."""
    sf.require(_comparison_key(entry) == _comparison_key(row), 'Historical confirmation geometry mismatch')
    sf.require(entry.get('boundary') == BOUNDARIES['fused']
               and entry.get('precision') == 'MXFP8_E4M3_UE8M0_accFP32_BF16out'
               and entry.get('validation') == 'both_payloads_and_postmeasurement_full_numeric_and_route',
               'Historical confirmation boundary/validation mismatch')
    samples = entry.get('raw_maxrank_ms')
    sf.require(isinstance(samples, list) and len(samples) == 50 and entry.get('warmup_calls', 0) >= 10,
               'Historical confirmation lacks 10+50 evidence')
    samples = [sf.finite(v, 'historical raw sample', 0) for v in samples]
    sf.require(all(v > 0 for v in samples), 'Nonpositive historical sample')
    sf.close(entry['p50_ms'], sf.percentile(samples, .5), 'historical p50')
    sf.close(entry['p95_ms'], sf.percentile(samples, .95), 'historical p95')
    sf.close(entry['half_drift'], sf.drift(samples), 'historical drift')
    sf.close(entry['pflops_per_rank'], 2 * row['m'] * row['n'] * row['k'] / entry['p50_ms'] / 1e12,
             'historical PFLOPS')
    for field in ('binary_sha256', 'environment_fingerprint', 'artifact_sha256', 'source_id'):
        sf.require(isinstance(entry.get(field), str) and re.fullmatch(r'[0-9a-f]{64}', entry[field]),
                   'Missing historical identity: ' + field)
    winner = row.get('winner')
    sf.require(winner and winner.get('run_id') != entry.get('run_id')
               and winner.get('configuration') == entry.get('configuration')
               and winner.get('epilogue_n') == entry.get('epilogue_n'),
               'Historical best lacks independent confirmation of the selected configuration')
    config = entry['configuration']
    sf.require(config.get('raster') in ('along_m', 'along_n')
               and int(config.get('effective_swizzle_size', 0)) in (1, 2, 4, 8)
               and 0 < int(config.get('comm_sm', 0)) < 148, 'Invalid historical layout/budget')


def build_acceptance_summary(results, historical, historical_evidence):
    """Auto versus independently confirmed historical winners, never grid minima."""
    sf.require(results and historical.get('schema') == 'sm103_mxfp8_fused_tuning_v1',
               'Missing current results or unregistered historical table')
    bins = {r['binary_sha256'] for r in results}
    environments = {r['environment_fingerprint'] for r in results}
    sf.require(len(bins) == len(environments) == 1, 'Current Auto candidates mix binaries/environments')
    history = {}
    for row in historical['rows']:
        key = _comparison_key(row)
        sf.require(key not in history, 'Duplicate historical comparison geometry')
        confirmation = row.get('confirmation')
        if confirmation: _check_confirmation(row, confirmation)
        history[key] = row
    rows, seen, versions = [], set(), set()
    for result in results:
        if result.get('component') != 'fused': continue
        selection = result.get('communication_selection', {})
        if selection.get('mode') != 'runtime_model': continue
        key = _comparison_key(result)
        sf.require(key not in seen, 'Duplicate Auto geometry; choose one run explicitly, do not winner-select repeats')
        seen.add(key)
        config, world = result['configuration'], result['world']
        budget = int(config['comm_sm'])
        queries = selection.get('rank_queries', [])
        version = selection.get('model_version')
        sf.require(selection.get('requested_comm_ctas') == selection.get('launch_comm_ctas') == 0
                   and selection.get('resolved_comm_ctas') == budget and 0 < budget < 148
                   and version and 'empty' not in version.lower()
                   and len(queries) == world and {int(r['rank']) for r in queries} == set(range(world)),
                   'Auto lacks valid requested-zero/all-rank evidence')
        for query in queries:
            sf.require(query.get('requested_comm') == query.get('launch_comm') == '0'
                       and int(query['resolved_comm']) == int(query['comm_sm']) == budget
                       and query['model_version'] == version, 'Auto native rank evidence mismatch')
        versions.add(version)
        sf.require(result['boundary'] == BOUNDARIES['fused'] and
                   result['validation'] == 'both_payloads_and_postmeasurement_full_numeric_and_route',
                   'Auto fused boundary/validation mismatch')
        old = history.get(key)
        confirmed = old.get('confirmation') if old else None
        auto = {k: v for k, v in result.items() if k not in
                ('raw_maxrank_ms', 'telemetry', 'component_resources', 'graph_preparation')}
        sequence = result['global_seq']
        # Re-run the PRESELECTED historical budget, not the fastest live manual
        # candidate. Same run => identical native binary, payloads and GEMM
        # launch configuration; only the explicit communication budget differs.
        # A historical layout change is reported, never silently attributed to c.
        paired = []
        gemm_fields = ('tile', 'tile_m', 'tile_n', 'tile_k', 'raster',
                       'effective_swizzle_size', 'max_swizzle_size', 'dynamic_smem')
        if confirmed:
            for candidate in results:
                if (candidate.get('component') != 'fused' or candidate['run_id'] != result['run_id']
                        or _comparison_key(candidate) != key
                        or candidate.get('communication_selection', {}).get('mode') != 'explicit'):
                    continue
                cc = candidate['configuration']
                if (int(cc['comm_sm']) == int(confirmed['configuration']['comm_sm'])
                        and candidate['epilogue_n'] == result['epilogue_n']
                        and all(cc.get(f) == config.get(f) for f in gemm_fields)):
                    paired.append(candidate)
        sf.require(len(paired) <= 1, 'Duplicate same-run historical-budget replay; do not pick fastest')
        replay = paired[0] if paired else None
        same_historical_gemm = (confirmed['epilogue_n'] == result['epilogue_n'] and
            all(confirmed['configuration'].get(f) == config.get(f) for f in gemm_fields)) if confirmed else None
        rows.append(dict(zip(COMPARISON_KEY, key), model=(old or {}).get('pure_reference_id', ''),
            domain_role='calibration' if sequence == 131072 else 'holdout' if sequence in (262144, 524288)
                else 'outside_declared_domain', auto=auto, confirmed_best=confirmed,
            auto_over_confirmed_best=(result['pflops_per_rank'] / confirmed['pflops_per_rank'] if confirmed else None),
            comparison='historical_reference_not_paired' if confirmed else 'missing_historical_confirmation',
            same_historical_gemm=same_historical_gemm,
            paired_manual=replay,
            auto_over_paired_manual=(result['pflops_per_rank'] / replay['pflops_per_rank'] if replay else None),
            paired_comparison=('same_run_historical_best_config' if same_historical_gemm else
                'same_run_historical_budget_on_current_gemm') if replay else 'missing_paired_replay',
            same_binary=(result['binary_sha256'] == confirmed['binary_sha256'] if confirmed else None),
            same_environment=(result['environment_fingerprint'] == confirmed['environment_fingerprint'] if confirmed else None)))
    sf.require(rows and len(versions) == 1, 'Missing Auto or mixed Auto model versions')
    # Cover every historical target, including an old OOM with no confirmed
    # denominator. Never shrink the scope to successful or confirmed runs.
    for key, old in history.items():
        if key in seen: continue
        sequence = old['global_seq']
        rows.append(dict(zip(COMPARISON_KEY, key), model=old.get('pure_reference_id', ''),
            domain_role='calibration' if sequence == 131072 else 'holdout' if sequence in (262144, 524288)
                else 'outside_declared_domain', auto=None, confirmed_best=old.get('confirmation'),
            auto_over_confirmed_best=None, comparison='missing_auto_measurement',
            same_historical_gemm=None, paired_manual=None, auto_over_paired_manual=None,
            paired_comparison='missing_auto_measurement',
            same_binary=None, same_environment=None))
    rows.sort(key=lambda r: (r['n'], r['k'], r['world'], r['global_seq']))
    summary = dict(schema='sm103_mxfp8_autotune_acceptance_v1',
        model_version=next(iter(versions)),
        binary_sha256=next(iter(bins)), environment_fingerprint=next(iter(environments)),
        historical_table=historical_evidence,
        historical_audit='retained_confirmation_samples_and_identity; original historical receipts not reaudited',
        limitations='Historical confirmation, not a simultaneous paired run; binary, layout and resources may differ. '
                    'Paired columns separately replay the preselected historical budget in the Auto run; '
                    'a different historical GEMM layout is explicitly marked. '
                    'A similar throughput is not proof of model accuracy or a passing holdout. '
                    'Missing confirmations stay empty; search winners never replace them.',
        coverage=dict(target_points=len(rows), auto_points=len(seen),
            matched_points=sum(r['auto'] is not None and r['confirmed_best'] is not None for r in rows),
            paired_points=sum(r['paired_manual'] is not None for r in rows),
            missing_auto_points=sum(r['auto'] is None for r in rows),
            historical_confirmed_points=sum(bool(r.get('confirmation')) for r in historical['rows']),
            historical_unconfirmed_points=sum(not r.get('confirmation') for r in historical['rows'])),
        rows=rows, audited_candidates=results)
    summary['acceptance_gate'] = acceptance_gate(summary)
    summary['acceptance_status'] = ('passed_user_thresholds' if summary['acceptance_gate']['passed']
        else 'incomplete_paired_coverage' if not summary['acceptance_gate']['complete']
        else 'below_user_thresholds')
    return summary


def acceptance_markdown(summary):
    def layout(row):
        if not row: return '—'
        c = row['configuration']
        return f"E{row['epilogue_n']} / {'M' if c['raster'] == 'along_m' else 'N'} / {c['effective_swizzle_size']} / {c['comm_sm']}"
    lines = ['# MXFP8 QKVProj Auto 对历史确认最优', '',
        f"Auto 已测 {summary['coverage']['auto_points']} / {summary['coverage']['target_points']} 点；未测点留空。", '',
        '每卡 PFLOPS；Graph 10+50，双 payload 全数值/路由校验。Auto 真正传入 comm_ctas=0。', '',
        '128K 是标定点，256K/512K 是 holdout。验收分母为同场重放的预选手工最优配置；历史原值另外保留。', '',
        '覆盖数按物理形状去重；Qwen2.5 72B / Llama 3.1 70B 同形状复用测量，分别展示。', '',
        '| 模型 / N×K | 序列 | CP | 用途 | Auto P | 历史确认最优 P | Auto / 历史 | 同场手工 P | Auto / 同场 | Auto E/raster/sw/c | 历史 E/raster/sw/c | 历史GEMM同配置 |',
        '|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|---|']
    for row in summary['rows']:
        auto, best = row['auto'], row['confirmed_best']
        model = re.sub(r'_s[0-9]+_cp[0-9]+$', '', row['model']) or f"{row['n']}×{row['k']}"
        best_p = f"{best['pflops_per_rank']:.3f}" if best else '—'
        auto_p = f"{auto['pflops_per_rank']:.3f}" if auto else '—'
        ratio = f"{row['auto_over_confirmed_best']:.3f}×" if auto and best else '—'
        replay = row.get('paired_manual')
        replay_p = f"{replay['pflops_per_rank']:.3f}" if replay else '—'
        paired_ratio = f"{row['auto_over_paired_manual']:.3f}×" if replay else '—'
        same_gemm = ('是' if row.get('same_historical_gemm') else '否') if auto and best else '—'
        for label in MODEL_LABELS.get((row['n'], row['k']), (model,)):
            lines.append(f"| {label} | {row['global_seq']//1024}K | {row['world']} | {row['domain_role']} | "
                f"{auto_p} | {best_p} | {ratio} | {replay_p} | {paired_ratio} | {layout(auto)} | {layout(best)} | {same_gemm} |")
    gate = acceptance_gate(summary)
    if gate['geometric_mean'] is not None:
        lines += ['', f"同配置同场配对 {gate['paired_points']}/{gate['required_points']}；"
            f"几何平均 {gate['geometric_mean']:.3%}，最低 {gate['minimum']:.3%}。"
            f"用户验收要求：几何平均 ≥95%，每点严格 >90%；{'通过' if gate['passed'] else '尚未通过'}。",
            '以上阈值只用于结果验收，不输入性能模型，也不用于拟合策略。']
    lines += ['', summary['limitations'], '',
        '历史表只读取 confirmation 独立复测，不取搜索最小值；历史原始回执未在本次重新审计。', '',
        f"Auto binary SHA256: `{summary['binary_sha256']}`", '',
        f"模型版本：`{summary['model_version']}`", '',
        'JSON 保留本轮全部候选的完整审计结果、50 个样本及历史确认记录；Markdown 只展示 Auto 对最优。', '']
    return '\n'.join(lines)


def acceptance_gate(summary):
    """User-specified REPORT thresholds, never inputs to the offline policy."""
    eligible = [r for r in summary['rows'] if r.get('paired_manual') and r.get('same_historical_gemm')]
    ratios = [sf.finite(r['auto_over_paired_manual'], 'paired retention', 0) for r in eligible]
    sf.require(all(r > 0 for r in ratios), 'Nonpositive paired retention')
    gm = math.exp(sum(math.log(r) for r in ratios)/len(ratios)) if ratios else None
    minimum = min(ratios) if ratios else None
    complete = len(ratios) == len(summary['rows']) and bool(ratios)
    return dict(reference='same_run_preselected_manual_best_configuration',
        required_geometric_mean=0.95, required_per_point_strictly_greater_than=0.90,
        paired_points=len(ratios), required_points=len(summary['rows']), complete=complete,
        geometric_mean=gm, minimum=minimum,
        passed=complete and gm >= .95 and minimum > .90)


def archive_manual_best(historical, source_evidence, directory):
    """Keep confirmed manual SOTA configurations/results, not sweep history."""
    rows = []
    for row in historical['rows']:
        saved = {k:row[k] for k in COMPARISON_KEY}
        saved.update(pure_reference_id=row.get('pure_reference_id', ''), pure_pflops=row.get('pure_pflops'))
        confirmed = row.get('confirmation')
        if confirmed:
            _check_confirmation(row, confirmed)
            saved.update(confirmation=confirmed, winner={k:row['winner'][k]
                for k in ('run_id', 'configuration', 'epilogue_n')})
        else:
            saved.update(confirmation=None, winner=None, status='missing_independent_confirmation')
        rows.append(saved)
    summary = dict(schema='sm103_mxfp8_fused_tuning_v1', rows=rows, source_evidence=source_evidence,
        scope='Independently confirmed best from finite offline searches; no global-optimum claim. '
              'Same-run replay measurements do not replace these confirmations by picking the faster repeat.')
    def markdown(data):
        lines = ['# MXFP8 QKVProj 手工最优 SOTA', '',
            '只保留独立确认的最优配置与结果，不展示搜索过程。每卡 PFLOPS；Graph 10+50。', '',
            '| 模型 | 序列 | CP | PFLOPS | E / raster / swizzle / comm CTA | 确认 run |',
            '|---|---:|---:|---:|---|---|']
        for row in sorted(data['rows'], key=lambda r:(r['n'],r['k'],r['world'],r['global_seq'])):
            best = row['confirmation']
            value = f"{best['pflops_per_rank']:.4f}" if best else '—'
            c = best['configuration'] if best else None
            config = (f"{best['epilogue_n']} / {c['raster']} / {c['effective_swizzle_size']} / {c['comm_sm']}"
                if c else '—')
            for model in MODEL_LABELS.get((row['n'],row['k']), (row['pure_reference_id'],)):
                lines.append(f"| {model} | {row['global_seq']//1024}K | {row['world']} | {value} | "
                    f"{config} | {best['run_id'] if best else '—'} |")
        return '\n'.join(lines + ['', data['scope'], '',
            'JSON 保留全部 50 个原始样本、p50/p95、数值/路由验收、source/binary/environment/artifact 哈希。', ''])
    return summary, _write_summary_pair(directory, summary, 'manual-best-current', markdown)


def archive_acceptance(run_paths, historical_path, output):
    results = []
    for path in run_paths:
        job = sf.json_bytes(Path(path, 'job.json').read_bytes())
        sf.require(job.get('auto_mxfp8_comm') is True, 'Comparison requires Auto runs')
        comm, qkv, _ = l20d.fused_candidates(job)
        for candidate in range(1, len(comm) * len(qkv) + 1):
            for component in (BOUNDARIES if job.get('calibrate') else ('fused',)):
                results.append(audit_run(path, candidate, component))
    historical_path = Path(historical_path).resolve()
    raw = sf.read_bytes(historical_path, historical_path.parent)
    summary = build_acceptance_summary(results, sf.json_bytes(raw),
        dict(path=str(historical_path), sha256=sf.digest(raw)))
    paths = _write_summary_pair(output, summary, 'acceptance-current', acceptance_markdown)
    return summary, paths


def archive_services(run_paths, output):
    # Never create/replace the destination while an input remains unaudited.
    # Full raw logs/artifacts stay in their original runs; only audit results
    # and selected raw timing samples are retained in this unique table.
    results = []
    for path in run_paths:
        job = sf.json_bytes(Path(path, 'job.json').read_bytes())
        comm, qkv, _ = l20d.fused_candidates(job)
        for candidate in range(1, len(comm) * len(qkv) + 1):
            results.extend(audit_run(path, candidate, component) for component in BOUNDARIES)
    summary = build_service_summary(results)
    paths = write_service_summary(output, summary)
    return summary, paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component', choices=(*MXFP8_COMPONENTS, 'all'), default='fused')
    parser.add_argument('--output', type=Path, help='Explicit destination for the unique aggregate F/C/R/Q table')
    parser.add_argument('--compare-best', type=Path, help='Historical fused-current.json with independent confirmations')
    parser.add_argument('runs', nargs='+')
    args = parser.parse_args()
    if args.compare_best:
        if not args.output or args.component != 'fused':
            parser.error('--compare-best requires --output and the default fused component')
        summary, paths = archive_acceptance(args.runs, args.compare_best, args.output)
        print(f"Audited {len(summary['audited_candidates'])} candidate boundaries; "
              f"{summary['coverage']['auto_points']}/{summary['coverage']['target_points']} Auto points (comparison only)")
        for path in paths: print(path)
        raise SystemExit(0)
    if args.component == 'all' or args.output:
        if args.component != 'all' or not args.output:
            parser.error('--component all and --output must be used together')
        summary, paths = archive_services(args.runs, args.output)
        print(f"Audited {len(summary['run_ids'])} runs / {len(summary['rows'])} physical candidates / "
              f"{4 * len(summary['rows'])} boundaries (aggregate_only)")
        for path in paths:
            print(path)
        raise SystemExit(0)
    for path in args.runs:
        job = json.loads((Path(path)/'job.json').read_text())
        comm, qkv, _ = l20d.fused_candidates(job)
        for index in range(1,len(comm)*len(qkv)+1):
            result=audit_run(path,index,args.component)
            print(json.dumps({k:v for k,v in result.items() if k not in ('raw_maxrank_ms','telemetry')},separators=(',',':')))
