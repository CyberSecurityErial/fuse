#!/usr/bin/env python3
"""Reproducible, local-only full-history BF16 comparison; never launch a GPU job.

CLI: --campaign campaign.json --output NEW_DIRECTORY
Campaign schema sm103_bf16_report_campaign_v1:
  manifest: {path, sha256}; placement_overrides: [{path, sha256}]
  source_sweep: {path: fetched_run_directory, fingerprint}
  fused: [{path, source_id, launch}]; baselines: [{path, groups?: [...]}]
  pure: [{path}]
Baseline groups are exact original plan group names; omission selects all.
All paths are relative to the campaign, or absolute. Source lists are explicit;
there is no latest-run discovery or best-of-versions selection. Empty lists are
useful for internal coverage checks; incomplete campaigns publish evidence only.
"""

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
import tarfile
import tempfile

import summarize_sm103_fused as fused
import summarize_sm103_sweep as sweep

bench = sweep.bench
require, close = fused.require, fused.close
SCHEMA = 'sm103_bf16_report_campaign_v1'
LAUNCHES = ('eager', 'graph')
BACKENDS = ('cublaslt_nccl', 'te_ub')
POOL = {(tile, comm) for tile in ('m128n128', 'm128n256k64e32')
        for comm in (8, 12, 16, 24, 32)}
LAYOUTS = {'qkv': 'qkv_source_rank_major_v1', 'oproj': 'causal_dual_chunk_v1'}
GEOMETRY = ('world', 'global_seq', 'hidden', 'q_heads', 'kv_heads', 'head_dim')
# These archived controllers predate the receipt's explicit node field. Their
# plans/device receipts still bind node0a, and the envelope verifies hostname.
LEGACY_REPLAY_CONTROLLERS = {
    'accc60e59ad24498f92480dadc8e8a0f3db48a17485659c3c418047402223727',
    '7c5619c71c6c9b61567820691e7f6571affb0d687480b74501426e11206d8957',
    'bde198597b50e1565b2e7c1bc838ac7a7bf2c8a997be11f269f584dcd75361db',
    '697078257abe192ae2c737e36cb26f36b58a681ea7fdb62c57f72520a0f1831b',
    '06751dd56c82d009123a3680144f7d8e15381533215b4b0539df72069b7a4f0b',
}
LEGACY_QKV_REPLAY_CONTROLLER = 'accc60e59ad24498f92480dadc8e8a0f3db48a17485659c3c418047402223727'


def read(path):
    path = Path(path)
    return fused.json_bytes(fused.read_bytes(path, path.parent.resolve()))


def evidence(path, **fields):
    return dict(path=str(path), sha256=fused.file_digest(path), **fields)


def samples_stats(samples):
    require(isinstance(samples, list) and len(samples) == 50 and
            all(fused.finite(x, 'sample', 0) > 0 for x in samples), 'Expected 50 positive samples')
    return {f'p{int(q * 100)}_ms': fused.percentile(samples, q) for q in (.5, .95)}


def audit_stability(record, samples, *, pure=False):
    """Both collectors select the FIRST stable round, not the fastest round."""
    require(record['initial_warmup'] == 10 and record['iterations'] == 50 and
            record['relative_range_limit'] == .05, 'Sampling contract changed')
    windows = record['window_ms_per_call']
    require(len(windows) >= 3 and all(fused.finite(x, 'warm window', 0) > 0 for x in windows),
            'Missing positive warmup windows')
    require((max(windows[-3:]) - min(windows[-3:])) / statistics.median(windows[-3:]) <= .05,
            'Warmup did not converge')
    budget = record['minimum_warmup_cuda_ms']
    require(budget == 100 if pure else budget in (0, 100), 'Unexpected warmup budget')
    require(record['additional_warmup_cuda_ms'] >= budget and
            record['warmup_converged' if pure else 'converged_all_ranks'] is True,
            'Insufficient measured warmup')
    require(record['collector'] == ('single_gpu_primed_events_v2' if pure else
                                    'primed_events_vector_max_v1'), 'Unknown collector')
    rounds, selected = record['measurement_rounds'], record['selected_round']
    require(type(selected) is int and 0 <= selected == len(rounds) - 1 < 3,
            'Invalid first-stable round index')
    for index, row in enumerate(rounds):
        xs = row['samples_ms']
        samples_stats(xs)
        drift = abs(statistics.median(xs[:25]) - statistics.median(xs[25:])) / statistics.median(xs)
        close(row['half_p50_relative_drift'], drift, 'sample drift')
        require(drift > .05 if index < selected else drift <= .05, 'Not the first stable round')
        if pure:
            require(len(row['sample_cadence_warmup_ms']) == 10 and
                    all(fused.finite(x, 'cadence', 0) > 0 for x in row['sample_cadence_warmup_ms']),
                    'Missing round cadence warmup')
    require(rounds[selected]['samples_ms'] == samples, 'Selected samples differ from raw output')
    close(record['sample_half_p50_relative_drift'], rounds[selected]['half_p50_relative_drift'], 'accepted drift')
    return dict(selected_round=selected, rejected_rounds=selected,
                maxrank_samples_repeated_on_ranks=not pure)


def audit_tensor(tensor, *, pure=False, magnitude=None):
    key = 'sample_nonzero_fraction' if pure else 'nonzero_fraction'
    require(tensor['dtype'] == 'torch.bfloat16' and .99 < tensor[key] <= 1 and
            tensor['sample_min'] < 0 < tensor['sample_max'] and
            fused.finite(tensor['sample_std'], 'input std', 0) > 0, 'Invalid random BF16 inputs')
    if magnitude is not None:
        require(max(abs(tensor['sample_min']), abs(tensor['sample_max'])) <= magnitude * 1.01,
                'Random input amplitude changed')


def audit_envelope(path):
    """Shared baseline/pure receipt adapter; all archived files are bound once.

    Fused runs keep their existing stricter binary/MPI auditor. Legacy sweep
    predates explicit node metadata; only its verified hostname resolves node.
    """
    root = Path(path).resolve()
    require(not Path(path).is_symlink(), 'Symlink run directory')
    receipt = read(root / 'fetched.json')
    require(receipt.get('state') == 'succeeded' and receipt.get('phase') == 'finished' and
            receipt.get('exit_code') == receipt.get('work_exit_code') == 0 and
            not any(receipt.get(k) for k in ('error', 'collection_error')), 'Run not successfully fetched')
    attempt = fused.integer(receipt['attempt'], 'attempt', 1)
    artifact = root / f'artifacts-attempt{attempt}'
    archive = root / 'artifacts.tar.gz'
    require(fused.file_digest(archive) == receipt['artifact_sha256'], 'Archive hash mismatch')
    before = fused.file_identity(archive.stat())
    files, seen = {}, set()
    with tarfile.open(archive, 'r:gz') as stream:
        for member in stream:
            name = fused.safe_member(member.name)
            require(name not in seen and (member.isfile() or member.isdir()), 'Duplicate/link archive member')
            seen.add(name)
            if member.isfile():
                actual = fused.read_bytes(artifact / name, root)
                require(actual == stream.extractfile(member).read(), f'Archive extraction differs: {name}')
                files[name] = {'path': str(artifact / name), 'sha256': fused.digest(actual)}
    require(before == fused.file_identity(archive.stat()), 'Archive changed during audit')
    def control(name):
        require('control/' + name in files, f'Missing archived control: {name}')
        return read(artifact / 'control' / name)
    job, status, env = control('job.json'), control('status.json'), control('environment.json')
    for key in ('run_id', 'stage', 'experiment', 'source_id'):
        require(job[key] == receipt[key] == status[key], f'Receipt mismatch: {key}')
    require(status['attempt'] == attempt and status['exit_code'] == status['work_exit_code'] == 0 and
            (status['state'], status['phase']) in (('running', 'collect'), ('succeeded', 'finished')) and
            not status.get('error'), 'Archived status failed')
    require(job['source_id'] == fused.json_digest(job['files']) and
            control('source-installed.json') == {'source_id': job['source_id'], 'files': job['files']},
            'Source manifest mismatch')
    require(not (root / 'job.json').exists() or read(root / 'job.json') == job, 'Local job mismatch')
    nodes = [node for node, (_, host) in fused.NODES.items() if env['host'] == host]
    require(len(nodes) == 1, 'Unknown actual hostname')
    node = nodes[0]
    for record in (job, receipt, status):
        require(record.get('node', node) == node, 'Node/hostname mismatch')
    env_id = fused.json_digest({k: v for k, v in env.items() if k != 'fingerprint'})
    require(env_id == env['fingerprint'] == receipt['environment_fingerprint'] == status['environment_fingerprint'],
            'Environment fingerprint mismatch')
    return dict(root=root, artifact=artifact, job=job, files=files, node=node,
                run_id=job['run_id'], source_id=job['source_id'], environment_fingerprint=env_id,
                archive=evidence(archive), environment_libraries=env.get('libraries'))


def origin(run, path, **fields):
    relative = str(Path(path).relative_to(run['artifact']))
    require(relative in run['files'], 'Evidence is not an archived member')
    return dict(run_id=run['run_id'], source_id=run['source_id'], node=run['node'],
                environment_fingerprint=run['environment_fingerprint'],
                raw=run['files'][relative], **fields)


def audit_pure_run(run):
    require(run['job']['stage'] == 'gemm-probe', 'Expected gemm-probe run')
    control = run['artifact'] / 'control'
    data, contract = read(control / 'gemm-probe.json'), read(control / 'gemm-probe-contract.json')
    require(data['schema'] == 'sm103_gemm_matrix_v1' and data['state'] == 'succeeded' and
            data['measurement'] == contract['measurement'] == 'single_gpu_pure_cublaslt' and
            data['compute'] == '10.3' and data['sms'] == 148 and data['output_dtype'] == 'bf16' and
            data['measured_ranks'] == 1 and data['distributed_boundary_measured'] is False and
            data['cublas_classic_measured'] is False, 'Not a pure BF16 matrix protocol')
    require(data['measurement_protocol'] == 'single_gpu_pure_gemm_stable_v2' and
            fused.SHA256.fullmatch(data['library_sha256']) and
            data['library_sha256'] == contract['library_sha256'], 'Missing pure protocol/library identity')
    matrix_path = control / 'gemm-matrix.json'
    matrix = read(matrix_path)
    require(matrix == run['job']['gemm_matrix_payload'] and
            data['matrix_sha256'] == fused.file_digest(matrix_path) and
            contract['node'] == run['node'] and contract['physical_devices'] == ['0'] and
            contract['shapes'] == matrix['shapes'], 'Pure matrix/placement contract mismatch')
    before = read(control / 'gpu-before.json')
    require(before['node'] == run['node'] and before['selected'] == ['0'] and before['observations'] and
            all(len(o['devices']) == 1 and o['devices'][0]['index'] == '0' and
                o['devices'][0]['uuid'] == contract['cuda_visible_devices'] for o in before['observations']),
            'Pure GPU identity mismatch')
    expected = defaultdict(list)
    for shape in matrix['shapes']:
        expected[tuple(shape[k] for k in ('m', 'n', 'k'))].append(shape['id'])
    require(len(expected) == data['unique_geometries'] == len(data['geometries']) and
            data['logical_shapes'] == len(matrix['shapes']), 'Pure geometry coverage mismatch')
    launches = run['job']['launches'].split(',')
    require(set(launches) <= set(LAUNCHES) and len(set(launches)) == len(launches), 'Pure launch request')
    rows, seen = [], set()
    for gi, geometry in enumerate(data['geometries']):
        mnk = tuple(geometry['shape'][k] for k in ('m', 'n', 'k'))
        require(mnk in expected and mnk not in seen and geometry['aliases'] == expected[mnk] and
                geometry['state'] == 'succeeded', 'Pure geometry/aliases mismatch')
        seen.add(mnk)
        inputs = geometry['inputs']
        require(inputs['seed'] == 103 and inputs['distribution'] == 'uniform' and
                inputs['activation_magnitude'] == .125 and inputs['weight_magnitude'] == .02 and
                inputs['magnitude_meaning'] == 'uniform_half_range', 'Pure random-input protocol')
        for tensor, size, magnitude in [('activation', [mnk[0], mnk[2]], .125), ('weight', [mnk[1], mnk[2]], .02)]:
            audit_tensor(inputs[tensor], pure=True, magnitude=magnitude)
            require(inputs[tensor]['shape'] == size and inputs[tensor]['sample_count'] == min(4096, math.prod(size)),
                    'Pure input geometry/statistics mismatch')
        require(Counter(r['launch'] for r in geometry['results']) == Counter(launches), 'Pure launch coverage')
        plans = []
        for ri, result in enumerate(geometry['results']):
            require(result['precision'] == 'bf16' and result['warmup'] == result['tune_warmup'] == 10 and
                    result['tune_iterations'] == 50, 'Pure BF16 10+50 required')
            stats = samples_stats(result['samples_ms'])
            for key, value in stats.items():
                close(result[key], value, 'Pure ' + key)
            sampling = audit_stability(result['measurement'], result['samples_ms'], pure=True)
            correct = result['correctness']
            require(correct['checked_values'] == min(64, mnk[0]) * min(64, mnk[1]) and
                    0 <= fused.finite(correct['relative_rms'], 'relative RMS') <= .02 and
                    fused.finite(correct['max_abs'], 'maximum error', 0) >= 0, 'Pure correctness failed')
            plan = result['tuning']
            require(plan['requested'] == 256 and 0 < plan['valid'] <= plan['returned'] <= 256 and
                    len(plan['candidates']) == plan['valid'] and plan['precision'] == 16 and
                    plan['math_sms'] == 0 and plan['beta'] == 0 and plan['graph_tuning'] == 0 and
                    plan['replay'] == result['launch'], 'Pure algorithm plan contract')
            require(all(fused.finite(c['tune_ms'], 'algorithm tune time', 0) > 0 for c in plan['candidates']),
                    'Invalid heuristic candidate')
            close(plan['best_ms'], min(c['tune_ms'] for c in plan['candidates']), 'Pure chosen algorithm')
            plans.append({k: v for k, v in plan.items() if k != 'replay'})
            close(result['pflops_per_gpu_p50'], 2 * math.prod(mnk) / stats['p50_ms'] / 1e12, 'Pure PFLOPS')
            rows.append(dict(kind='pure', node=run['node'], mnk=mnk, launch=result['launch'], **stats,
                samples_ms=result['samples_ms'], sampling=sampling,
                provenance=origin(run, control / 'gemm-probe.json',
                    pointer=f'/geometries/{gi}/results/{ri}', library_sha256=data['library_sha256'],
                    gpu_uuid=contract['cuda_visible_devices'], algorithm=plan['algorithm']),
                scope='single_gpu_diagnostic_not_distributed_maxrank'))
        require(all(plan == plans[0] for plan in plans), 'Pure Graph silently retuned algorithm')
    return rows


def audit_baseline_measurement(run, job, folder):
    case = job['case']
    require(job['warmup'] == 10 and job['iterations'] == 50 and
            job['env']['FUSE_SM103_MEASUREMENT'] == 'v2', 'Baseline sampling protocol')
    path = run['artifact'] / 'results' / folder / Path(job['output']).name
    require(str(path.relative_to(run['artifact'])) in run['files'], 'Unarchived baseline output')
    stats, data = bench.read_measurement(path, job), read(path)
    metric = bench.METRICS[case['direction'], job['backend']]
    samples = data['samples_ms']
    samples = samples[metric] if isinstance(samples, dict) else samples
    for key, value in stats.items():
        close(samples_stats(samples)[key], value, 'Baseline recomputed sample')
        close(data['results'][metric][key], value, 'Baseline stored percentile')
    mnk = data.get('gemm_shape') or data['gemm_shapes']['qkv']
    require(all(mnk[k] == case[k] for k in ('m', 'n', 'k')), 'Baseline MNK differs')
    correct_fields = ({'max_abs', 'remote_recv_mismatches', 'remote_recv_mismatches_by_peer'}
                      if job['backend'] == 'te_ub' else
                      {'cublaslt_vs_torch_mm_max_abs', 'packed_qkv_mismatches' if case['direction'] == 'qkv'
                       else 'inverse_a2a_reference_mismatches'})
    if job['backend'] == 'te_ub' and case['direction'] == 'oproj':
        correct_fields.add('self_pack_mismatches')
    require(correct_fields <= data['correctness'].keys(), 'Missing numeric/route correctness checks')
    if job['backend'] == 'te_ub':
        require(len(data['correctness']['remote_recv_mismatches_by_peer']) == case['cp'], 'Missing peer correctness')
    for key, value in data['correctness'].items():
        for number in value if isinstance(value, list) else [value]:
            require(number == 0 if 'mismatch' in key else 0 <= fused.finite(number, key) <= .01,
                    'Baseline correctness tolerance')
    config = job['config']
    if job['backend'] == 'te_ub':
        actual = data['config']
        mapping = dict(comm_sm='num_comm_sm', streams='num_streams', push='push', use_ce='use_ce',
                       pack_block='pack_block', pack_warps='pack_warps', reverse='reverse')
        require(all(actual[v] == config[k] for k, v in mapping.items()) and
                actual['math_sm'] == 148 - config['comm_sm'] and
                actual['tune_warmup'] == 10 and actual['tune_iters'] == 50 and
                actual['cuda_graph'] == (job['launch'] == 'graph'), 'Actual Userbuffers configuration differs')
        if case['direction'] == 'qkv':
            require(actual['local_first'] == config['local_first'], 'Userbuffers local-first differs')
        if job['launch'] == 'graph':
            require('post_graph_max_abs' in data['correctness'], 'Missing Graph output check')
    else:
        require(all(data['environment'].get(k) == v for k, v in job['env'].items() if k.startswith('NCCL_')) and
                all(data[k] == config[k] for k in ('pack_block', 'pack_warps')) and
                data['nccl_high_priority'] == config['high_priority'], 'Actual NCCL configuration differs')
    rank_evidence, sampling = [], []
    for rank in range(case['cp']):
        rp = path.with_suffix(f'.rank{rank}.json')
        md = read(rp)
        # The original full sweep predates rank direction/layout fields. Its
        # immutable full plan plus direction-specific raw metric binds direction;
        # modern replay metadata must contain it explicitly.
        legacy_direction = folder == 'sweep' or (case['direction'] == 'qkv' and
            run.get('job', {}).get('files', {}).get('scripts/l20d.py') == LEGACY_QKV_REPLAY_CONTROLLER)
        require(md['rank'] == rank and md.get('direction', case['direction'] if legacy_direction else None) ==
                case['direction'] and md['device']['sm_count'] == 148 and
                md['cuda_visible_devices'] == job['env']['CUDA_VISIBLE_DEVICES'] and
                md['cublaslt_tune_launch'] == job['launch'] and len(md['measurement_records']) == 1,
                'Baseline rank metadata mismatch')
        record = md['measurement_records'][0]
        require(record['launch'] == job['launch'] and len(record['sample_cadence_warmup_ms']) == 10 and
                record['graph_replay_warmup'] == (10 if job['launch'] == 'graph' else 0), 'Baseline warmup/launch')
        sampling.append(audit_stability(record, samples))
        require(md['input_statistics'] and md['cublaslt_plans'] and
                all(0 < p['valid'] <= p['returned'] for p in md['cublaslt_plans']), 'Missing input/plan evidence')
        for inputs in md['input_statistics']:
            require(inputs['distribution'] == 'uniform' and
                    inputs['seed'] == (3109 if case['direction'] == 'qkv' else 2701) + rank,
                    'Baseline input generator/seed differs')
            for name, tensor in inputs['tensors'].items():
                audit_tensor(tensor, magnitude=.125 if name == 'activation' else .02)
        rank_evidence.append(origin(run, rp, pointer='/measurement_records/0'))
    return dict(kind=job['backend'], node=run['node'], case=case, launch=job['launch'],
                layout=(LAYOUTS['qkv'] if case['direction'] == 'qkv' else
                        data.get('oproj_layout', 'legacy')), config=config, **stats, samples_ms=samples,
                provenance=origin(run, path, pointer='/samples_ms/' + metric if isinstance(data['samples_ms'], dict)
                                  else '/samples_ms', group=job['group'], ranks=rank_evidence),
                sampling=sampling, devices=job['env']['CUDA_VISIBLE_DEVICES'])


class Sources:
    """One audit per explicit run and one source-winner verification per group."""

    def __init__(self, source):
        self.envelopes, self.winners, self.fused = {}, {}, {}
        self.source = self.envelope(source['path'])
        require(self.source['job']['stage'] == 'sweep' and self.source['node'] == '09', 'Source is not node09 sweep')
        results = self.source['artifact'] / 'results'
        self.plan, raw = sweep.load_plan(results, source['fingerprint'])
        self.plan_hash = fused.digest(raw)
        self.groups = defaultdict(list)
        for job in self.plan['jobs']:
            self.groups[job['group']].append(job)

    def envelope(self, path):
        key = str(Path(path).resolve())
        if key not in self.envelopes:
            print('AUDIT receipt ' + key, flush=True)
            self.envelopes[key] = audit_envelope(key)
        return self.envelopes[key]

    def winner(self, group):
        if group not in self.winners:
            require(group in self.groups, 'Group missing from full source sweep')
            values = [(sweep.measurement_row(self.source['artifact'] / 'results', self.plan, job), job)
                      for job in self.groups[group]]
            row, job = min(values, key=lambda pair: (pair[0]['p50_ms'], pair[0]['config_json']))
            accepted = audit_baseline_measurement(self.source, job, 'sweep')
            self.winners[group] = row, job, accepted
        return self.winners[group]

    def baseline(self, entry):
        require(set(entry) <= {'path', 'groups'}, 'Unknown baseline selector')
        run = self.envelope(entry['path'])
        if run['job']['stage'] == 'sweep':
            require(run['root'] == self.source['root'], 'Undeclared source sweep')
            groups = entry.get('groups', list(self.groups))
            require(len(groups) == len(set(groups)), 'Duplicate baseline groups')
            return [self.winner(group)[2] for group in groups]
        require(run['job']['stage'] == 'baseline-replay', 'Unexpected baseline stage')
        path = run['artifact'] / 'results/replay_plan.json'
        plan = read(path)
        payload = run['job']['baseline_replay']
        control = read(run['artifact'] / 'control/baseline-replay.json')
        devices = read(run['artifact'] / 'control/baseline-devices.json')
        executor = read(run['artifact'] / 'results/replay_plan.executor.json')
        controller = run['job'].get('files', {}).get('scripts/l20d.py')
        receipt_node = control.get('node', '0a' if controller in LEGACY_REPLAY_CONTROLLERS else None)
        require(plan['stage'] == 'baseline-replay' and plan['precision'] == 'bf16' and
                plan['node'] == devices['node'] == receipt_node == run['node'] and
                plan['source_id'] == run['source_id'] and
                plan['environment_fingerprint'] == run['environment_fingerprint'] and
                plan['fingerprint'] == control['fingerprint'] == executor['measurement_fingerprint'] and
                plan['source_winners'] == payload and plan['imports_source_measurements'] is False and
                plan['communication_search'] is False, 'Replay receipt/provenance differs')
        require(payload['schema'] == 'sm103_baseline_replay_input_v1' and payload['source_node'] == '09' and
                payload['source_fingerprint'] == control['source_fingerprint'] == self.plan['fingerprint'] and
                payload['source_plan_sha256'] == self.plan_hash, 'Replay source plan differs')
        entries = {row['group']: row for row in payload['entries']}
        require(len(entries) == len(payload['entries']) and
                Counter(j['group'] for j in plan['jobs']) == Counter(entries.keys()), 'Replay plan coverage')
        legacy_qkv = controller == LEGACY_QKV_REPLAY_CONTROLLER and all(
            j['case']['direction'] == 'qkv' for j in plan['jobs'])
        plan_layout = plan.get('oproj_layout', 'legacy' if legacy_qkv else None)
        job_layout = run['job'].get('oproj_layout', 'legacy' if legacy_qkv else None)
        require(plan_layout in bench.OPROJ_LAYOUTS and plan_layout == job_layout, 'Replay layout request differs')
        worlds = [j['case']['cp'] for j in plan['jobs']]
        require(worlds and all(type(cp) is int and cp in (4, 8) for cp in worlds), 'Invalid replay world size')
        physical, uuids = devices['physical'], devices['cuda_visible_devices'].split(',')
        require(isinstance(physical, list) and len(physical) == len(set(physical)) == len(uuids) ==
                len(set(uuids)) == max(worlds) and all(index in {str(i) for i in range(8)} for index in physical) and
                all(uuid.startswith('GPU-') for uuid in uuids) and
                run['job']['devices'].split(',')[:max(worlds)] == physical, 'Replay device inventory differs')
        selected = entry.get('groups', list(entries))
        require(len(selected) == len(set(selected)) and set(selected) <= entries.keys(), 'Replay requested groups')
        rows = []
        for job in plan['jobs']:
            group = job['group']
            original = entries[group]
            row, prior, accepted = self.winner(group)
            source = original['source']
            require(all(job[k] == original[k] == prior[k] for k in ('case', 'backend', 'launch', 'config')) and
                    job['source_winner'] == source and source['selection_stage'] == 'sweep' and
                    source['status'] == 'complete' and source['independently_remeasured'] is False and
                    source['expected_candidates'] == source['validated_candidates'] == len(self.groups[group]) and
                    source['warmup'] == 10 and source['samples'] == 50 and
                    source['raw_sha256'] == accepted['provenance']['raw']['sha256'] and
                    source['rank_sha256'] == {str(i): r['raw']['sha256'] for i, r in enumerate(accepted['provenance']['ranks'])},
                    'Replay changed original finite-grid winner')
            for name in ('p50_ms', 'p95_ms'):
                close(source[name], row[name], 'Source winner ' + name)
            # The receipt owns the maximum CP group; each worker is launched on
            # exactly its CP-sized ordered prefix, including in mixed CP4/8 runs.
            cp = job['case']['cp']
            require(job['env']['CUDA_VISIBLE_DEVICES'].split(',') == uuids[:cp], 'Replay devices request differs')
            measured = audit_baseline_measurement(run, job, 'baseline-replay')
            measured['physical_devices'] = physical[:cp]
            measured['provenance']['source_winner'] = accepted['provenance']['raw']
            if group in selected:
                rows.append(measured)
        return rows

    def fusion(self, entry):
        require(set(entry) == {'path', 'source_id', 'launch'} and entry['launch'] in LAUNCHES,
                'Fused entry requires explicit source and launch')
        key = str(Path(entry['path']).resolve())
        require(key not in self.fused, 'Fused run appears twice')
        print('AUDIT fused ' + key, flush=True)
        run = fused.audit_run(key)
        self.fused[key] = run
        require(run['source_id'] == entry['source_id'] and run['build']['mpi'] is True and
                run['build']['profile'] is False and run['diagnostic_only'] is False and
                run['config']['process_layout'] == 'mpi_one_process_per_gpu', 'Non-production/non-MPI fused run')
        rows = []
        for direction, label in [('qkv', 'GEMM_A2A'), ('oproj', 'A2A_GEMM')]:
            candidates = [c for c in run['candidates'] if c['direction'] == label and c['component'] == 'fused']
            require(Counter((c['tile_policy'], c['comm_ctas']) for c in candidates) == Counter(POOL),
                    'Incomplete or changed ten-candidate pool')
            require(all(c['performance_accepted'] and c['launch'] == entry['launch'] and
                        c['layout'] == LAYOUTS[direction] and c['max_swizzle_size'] == 1 and
                        c['effective_swizzle_size'] == 1 and c['raster_requested'] == 'heuristic'
                        for c in candidates), 'Fused launch/layout/scheduling differs')
            best = min(candidates, key=lambda c: (c['timing']['p50_ms'], c['tile_policy'], c['comm_ctas']))
            shape = run['geometry']
            case = dict(direction=direction, seq=shape['global_seq'], cp=shape['world'],
                        **{k: shape[k] for k in ('hidden', 'q_heads', 'kv_heads', 'head_dim')},
                        **{k: best[k] for k in ('m', 'n', 'k')})
            rows.append(dict(kind='fused', node=run['node'], case=case, launch=entry['launch'],
                layout=best['layout'], p50_ms=best['timing']['p50_ms'], p95_ms=best['timing']['p95_ms'],
                config=dict(tile=best['tile_policy'], comm_ctas=best['comm_ctas']),
                provenance=dict(run_id=run['run_id'], source_id=run['source_id'], node=run['node'],
                    binary_sha256=run['build']['binary_sha256'], environment_fingerprint=run['environment_fingerprint'],
                    candidate=best['candidate'], raw=run['evidence'][next(k for k in run['evidence']
                        if k.startswith('control/attempt') and k.endswith('.log'))],
                    timing=best['timing'], all_candidates=[dict(candidate=c['candidate'], tile=c['tile_policy'],
                        comm_ctas=c['comm_ctas'], p50_ms=c['timing']['p50_ms'], p95_ms=c['timing']['p95_ms'])
                        for c in candidates]),
                physical_devices=[d['physical_index'] for d in run['telemetry']['devices']],
                gpu_uuids=[d['uuid'] for d in run['telemetry']['devices']]))
        return rows


def logical_id(case):
    return f"{case['direction']}/{case['model']}/s{case['seq']}/cp{case['cp']}"


def physical_key(case):
    # OProj KV-head count does not participate in its route; QKV needs it.
    return tuple(case[k] for k in ('direction', 'seq', 'cp', 'hidden', 'q_heads')) + (
        case['kv_heads'] if case['direction'] == 'qkv' else 0, case['head_dim'])


def load_campaign(path):
    path = Path(path).resolve()
    campaign = read(path)
    require(set(campaign) == {'schema', 'manifest', 'placement_overrides', 'source_sweep', 'fused', 'baselines', 'pure'} and
            campaign['schema'] == SCHEMA, 'Unknown campaign schema/fields')
    def resolve(value):
        return str((path.parent / value).resolve())
    def pinned(reference):
        require(set(reference) == {'path', 'sha256'}, 'Expected pinned data reference')
        reference['path'] = resolve(reference['path'])
        require(fused.file_digest(Path(reference['path'])) == reference['sha256'], 'Campaign data hash differs')
        return read(reference['path'])
    manifest = pinned(campaign['manifest'])
    scope = manifest['scope']
    require(scope['logical_direction_geometry_rows'] == 192 and scope['physical_direction_geometry_rows'] == 168 and
            scope['requested_launches'] == list(LAUNCHES), 'Not full historical report scope')
    args = argparse.Namespace(directions='qkv,oproj', models='', seqs=scope['historical_sequence_lengths'],
                              cps=(4, 8), devices='0,1,2,3,4,5,6,7')
    cases = list(bench.cases(args))
    require(len(cases) == 192 and len({physical_key(c) for c in cases}) == 168, 'Historical definitions changed')
    cp_nodes = {cp: node for node, cp in scope['node_world'].items()}
    require(cp_nodes == {4: '09', 8: '0a'}, 'Unexpected default placement')
    nodes = {physical_key(c): cp_nodes[c['cp']] for c in cases}
    changed = set()
    for reference in campaign['placement_overrides']:
        override = pinned(reference)
        geometry = override['geometry']
        matching = [c for c in cases if all(geometry[k] == c[{'world': 'cp', 'global_seq': 'seq'}.get(k, k)]
                    for k in GEOMETRY)]
        require(matching and override['destination_node'] in fused.NODES and
                override['cross_node_ratios_permitted'] is False, 'Invalid placement geometry')
        for key in {physical_key(case) for case in matching}:
            require(key not in changed and nodes[key] == override['original_node'], 'Conflicting placement')
            changed.add(key)
            nodes[key] = override['destination_node']
        # destination_state is descriptive historical text, NEVER completion.
    campaign['source_sweep']['path'] = resolve(campaign['source_sweep']['path'])
    require(set(campaign['source_sweep']) == {'path', 'fingerprint'}, 'Unknown source-sweep fields')
    for name in ('fused', 'baselines', 'pure'):
        require(isinstance(campaign[name], list), 'Run catalog must be a list')
        for entry in campaign[name]:
            entry['path'] = resolve(entry['path'])
        require(len({entry['path'] for entry in campaign[name]}) == len(campaign[name]), 'Duplicate run catalog entry')
    for launch in LAUNCHES:
        require(len({e['source_id'] for e in campaign['fused'] if e['launch'] == launch}) <= 1,
                'Multiple fused source versions within one launch campaign')
    return campaign, cases, nodes


def join_measurements(cases, placements, rows):
    """No best-of-runs: duplicate physical cells must be identical alias data."""
    index, excluded = {}, []
    for row in rows:
        if row['kind'] == 'pure':
            key = ('pure', row['node'], tuple(row['mnk']), row['launch'])
        else:
            physical = physical_key(row['case'])
            if physical not in placements or placements[physical] != row['node']:
                excluded.append(dict(reason='outside historical placement', provenance=row['provenance']))
                continue
            require(row['layout'] == LAYOUTS[row['case']['direction']], 'Legacy/incompatible route in requested cell')
            key = (row['kind'], row['node'], physical, row['launch'])
        require(row['launch'] in LAUNCHES, 'Unknown launch')
        if key in index:
            prior = index[key]
            require(row['kind'] in BACKENDS and row['provenance']['run_id'] == prior['provenance']['run_id'] and
                    all(row[k] == prior[k] for k in ('p50_ms', 'p95_ms', 'samples_ms', 'config')),
                    'Ambiguous duplicate measurement; choose source explicitly, not fastest')
            prior.setdefault('alias_provenance', []).append(row['provenance'])
        else:
            index[key] = row
    table, missing = [], []
    for case in cases:
        node, physical = placements[physical_key(case)], physical_key(case)
        mnk = tuple(case[k] for k in ('m', 'n', 'k'))
        for launch in LAUNCHES:
            entries = {kind: index.get((kind, node, mnk if kind == 'pure' else physical, launch))
                       for kind in ('fused', *BACKENDS, 'pure')}
            absent = [kind for kind, row in entries.items() if row is None]
            if absent:
                missing.append(dict(case=logical_id(case), node=node, launch=launch, missing=absent))
                continue
            f = entries['fused']
            for backend in BACKENDS:
                b = entries[backend]
                require(all(b['case'][k] == case[k] for k in ('m', 'n', 'k')), 'Same-route baseline MNK differs')
                physical_devices = b.get('physical_devices', b['devices'].split(','))
                require(physical_devices == f['physical_devices'], 'Baseline/fused physical rank group differs')
                if b['devices'].startswith('GPU-'):
                    require(b['devices'].split(',') == f['gpu_uuids'], 'Baseline/fused GPU UUIDs differ')
            require(all(f['case'][k] == case[k] for k in ('m', 'n', 'k')), 'Fused MNK differs')
            values = dict(case=case, node=node, launch=launch, measurements=entries)
            values['ratios'] = dict(te_ub_x=entries['te_ub']['p50_ms'] / f['p50_ms'],
                strong_x=min(entries[b]['p50_ms'] for b in BACKENDS) / f['p50_ms'],
                pure_percent_diagnostic=100 * entries['pure']['p50_ms'] / f['p50_ms'])
            values['pflops_per_gpu'] = {kind: 2 * math.prod(mnk) / entry['p50_ms'] / 1e12
                                       for kind, entry in entries.items()}
            table.append(values)
    return table, missing, excluded


def aggregate_ratios(table):
    """Summarize already paired cells, with one vote per historical label.

    Physical aliases remain separate votes; this is not a pooled-sample metric
    or a geometry-deduplicated mean. Missing reports must not call this helper.
    """
    identities = Counter((logical_id(c['case']), c['launch']) for c in table)
    require(len(table) == len(identities) == 384 and all(n == 1 for n in identities.values()) and
            Counter((c['launch'], c['case']['direction']) for c in table) ==
            Counter({(launch, direction): 96 for launch in LAUNCHES for direction in ('qkv', 'oproj')}),
            'Aggregates require the complete logical matrix')
    result = dict(schema='sm103_bf16_aggregates_v1',
        weighting='one_equal_weight_per_historical_logical_row; physical aliases retain separate votes',
        formula='exp(sum(log(same_node_paired_p50_ratio)) / logical_count)',
        scope_definitions=dict(all='all 192 historical logical rows per launch', long='global S >= 65536'),
        ratio_definitions=dict(te_ub_x='TE Userbuffers p50 / fused p50',
                               strong_x='min(TE Userbuffers, cuBLASLt+NCCL) p50 / fused p50'),
        badcase_definition='ratio below its direction GM within the same launch/scope; overall unions the direction lists',
        equality_roundoff_relative_tolerance=1e-12, samples_pooled=False,
        physical_geometry_reweighted=False, performance_model_evaluated=False, goal_achievement_evaluated=False,
        launches={})
    for launch in LAUNCHES:
        launch_cells = [c for c in table if c['launch'] == launch]
        require(len({physical_key(c['case']) for c in launch_cells}) == 168,
                'Aggregates require complete physical coverage')
        scopes = result['launches'][launch] = {}
        for scope in ('all', 'long'):
            scoped = [c for c in launch_cells if scope == 'all' or c['case']['seq'] >= 65536]
            groups = scopes[scope] = {}
            for direction in ('qkv', 'oproj', 'overall'):
                cells = [c for c in scoped if direction == 'overall' or c['case']['direction'] == direction]
                require(cells, 'Empty aggregate direction/scope')
                values = groups[direction] = dict(logical_count=len(cells),
                    physical_count=len({physical_key(c['case']) for c in cells}),
                    geometric_mean={}, below_direction_geomean={})
                for metric in ('te_ub_x', 'strong_x'):
                    ratios = [fused.finite(c['ratios'][metric], metric, 0) for c in cells]
                    require(all(ratio > 0 for ratio in ratios), 'Nonpositive speedup ratio')
                    values['geometric_mean'][metric] = math.exp(math.fsum(math.log(ratio) for ratio in ratios) / len(ratios))
                    if direction == 'overall':
                        below = [case for name in ('qkv', 'oproj')
                                 for case in groups[name]['below_direction_geomean'][metric]]
                    else:
                        threshold = values['geometric_mean'][metric]
                        below = [dict(case=logical_id(c['case']), direction=direction, node=c['node'],
                            ratio=ratio, direction_geometric_mean=threshold,
                            fused_run=c['measurements']['fused']['provenance']['run_id'])
                            for c, ratio in zip(cells, ratios) if ratio < threshold and
                            not math.isclose(ratio, threshold, rel_tol=1e-12, abs_tol=0)]
                    values['below_direction_geomean'][metric] = sorted(below, key=lambda c: (c['ratio'], c['case']))
    return result


def write_report(output, report, table):
    output = Path(output).absolute()
    require(not output.exists() and not output.is_symlink(), 'Refusing to overwrite report')
    if report['complete']:
        report['aggregates'] = aggregate_ratios(table)
    else:
        require('aggregates' not in report, 'An incomplete report cannot publish aggregates')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.bf16-report-', dir=output.parent) as temporary:
        staged = Path(temporary) / 'complete'
        staged.mkdir()
        (staged / 'evidence.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        if report['complete']:
            columns = ['direction', 'model', 'seq', 'cp', 'hidden', 'q_heads', 'kv_heads', 'head_dim', 'm', 'n', 'k',
                       'node', 'launch']
            for kind in ('fused', *BACKENDS, 'pure'):
                columns += [kind + '_' + field for field in ('p50_ms', 'p95_ms', 'pflops_per_gpu', 'run', 'source')]
            columns += ['te_ub_x', 'strong_x', 'pure_percent_diagnostic', 'fused_config']
            with (staged / 'table.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                for cell in table:
                    row = cell['case'] | {'node': cell['node'], 'launch': cell['launch']} | cell['ratios']
                    for kind, measured in cell['measurements'].items():
                        row |= {kind + '_p50_ms': measured['p50_ms'], kind + '_p95_ms': measured['p95_ms'],
                                kind + '_pflops_per_gpu': cell['pflops_per_gpu'][kind],
                                kind + '_run': measured['provenance']['run_id'],
                                kind + '_source': measured['provenance']['source_id']}
                    row['fused_config'] = json.dumps(cell['measurements']['fused']['config'], sort_keys=True)
                    writer.writerow(row)
            lines = ['# SM103 BF16 全量性能表', '',
                '192 个历史逻辑方向/shape，168 个物理方向几何；Eager/Graph 分开选有限 10 候选。', '',
                '每个性能格为 **p50 / p95 ms；PFLOP/s/GPU**。吞吐仅计 GEMM 2MNK，边界耗时包含完整路由。',
                '自研为 MPI 一进程一卡；基线为 torchrun 一进程一卡。逐样本 max-rank 后计算 p50/p95。',
                '“纯 GEMM %”是同节点同 MNK 单卡 cuBLASLt p50 / 自研边界 p50，仅诊断参考，不是分布式强基线。',
                '纯 GEMM Graph 复用 Eager 选出的算法。有限搜索不证明全局最优或泛化模型达标。',
                'Eager 与 Graph 源码版本分别记录；不要将二者差值直接归因为算子优化。完整来源见 evidence.json/table.csv。', '']
            lines += ['几何平均（每格：相对 TE Userbuffers / 相对强基线）：历史逻辑行等权，物理别名保留权重；非模型或目标达标判定。', '',
                '| 启动 | 范围 | 逻辑/物理数 | QKV | OProj | 整体 |', '|---|---|---|---|---|---|']
            for launch, scopes in report['aggregates']['launches'].items():
                for scope, groups in scopes.items():
                    overall = groups['overall']
                    values = [launch, '全历史' if scope == 'all' else '长序列 S≥65536',
                              f"{overall['logical_count']}/{overall['physical_count']}"]
                    values += [f"{groups[d]['geometric_mean']['te_ub_x']:.3f}× / {groups[d]['geometric_mean']['strong_x']:.3f}×"
                               for d in ('qkv', 'oproj', 'overall')]
                    lines.append('| ' + ' | '.join(values) + ' |')
            lines += ['', '低于同范围、同方向几何平均的 case 清单见 evidence.json 的 aggregates；Eager/Graph 不混合。', '']
            by_case = defaultdict(dict)
            for cell in table:
                by_case[logical_id(cell['case'])][cell['launch']] = cell
            def timing(cell, kind):
                m = cell['measurements'][kind]
                return f"{m['p50_ms']:.6f}/{m['p95_ms']:.6f}; {cell['pflops_per_gpu'][kind]:.3f}"
            for direction in ('qkv', 'oproj'):
                lines += [f'## {"GEMM+A2A (QKV)" if direction == "qkv" else "A2A+GEMM (OProj)"}', '',
                    '| CP | 模型 | S | M×N×K | Hq/Hkv/D | 节点 | Eager 自研 | Graph 自研 | Eager/TEUB | Graph/TEUB | Eager/强基线 | Graph/强基线 | Eager 纯 GEMM % | Graph 纯 GEMM % | Eager cuBLASLt+NCCL | Graph cuBLASLt+NCCL | Eager TE Userbuffers | Graph TE Userbuffers | Eager 纯 cuBLASLt | Graph 纯 cuBLASLt | 配置 E/G | run E/G |',
                    '|' + '---|' * 22]
                for values in by_case.values():
                    eager, graph = values['eager'], values['graph']
                    c = eager['case']
                    if c['direction'] != direction:
                        continue
                    cells = [str(c['cp']), c['model'], str(c['seq']), '×'.join(str(c[k]) for k in ('m','n','k')),
                             '/'.join(str(c[k]) for k in ('q_heads','kv_heads','head_dim')), eager['node'],
                             timing(eager, 'fused'), timing(graph, 'fused')]
                    for ratio in ('te_ub_x', 'strong_x', 'pure_percent_diagnostic'):
                        cells += [f"{v['ratios'][ratio]:.3f}{'%' if ratio.endswith('diagnostic') else '×'}"
                                  for v in (eager, graph)]
                    for kind in (*BACKENDS, 'pure'):
                        cells += [timing(v, kind) for v in (eager, graph)]
                    cells += [' / '.join(f"{v['measurements']['fused']['config']['tile']},c{v['measurements']['fused']['config']['comm_ctas']}"
                                        for v in (eager, graph)),
                              ' / '.join(v['measurements']['fused']['provenance']['run_id'] for v in (eager, graph))]
                    lines.append('| ' + ' | '.join(cells) + ' |')
                lines.append('')
            (staged / 'table.md').write_text('\n'.join(lines) + '\n')
        require(not output.exists(), 'Output appeared during audit')
        staged.rename(output)


def build_report(campaign_path, output):
    require(not Path(output).exists() and not Path(output).is_symlink(), 'Refusing to overwrite report')
    campaign, cases, placements = load_campaign(campaign_path)
    sources = Sources(campaign['source_sweep'])
    require({logical_id(j['case']): j['case'] for j in sources.plan['jobs']} ==
            {logical_id(case): case for case in cases}, 'Historical cases differ from pinned full source plan')
    rows = []
    for entry in campaign['fused']:
        rows.extend(sources.fusion(entry))
    for entry in campaign['baselines']:
        rows.extend(sources.baseline(entry))
    for entry in campaign['pure']:
        require(set(entry) == {'path'}, 'Unknown pure selector')
        rows.extend(audit_pure_run(sources.envelope(entry['path'])))
    table, missing, excluded = join_measurements(cases, placements, rows)
    complete = not missing and len(table) == 384
    report = dict(schema='sm103_bf16_report_v1', complete=complete, model_fitted=False, globally_optimal=False,
        scope=dict(logical_rows=192, physical_rows=168, launch_cells=384, physical_launch_cells=336),
        campaign=evidence(Path(campaign_path).resolve()), inputs=campaign, missing=missing,
        complete_logical_launch_cells=len(table), excluded=excluded,
        receipt_evidence=[dict(run_id=r['run_id'], node=r['node'], source_id=r['source_id'],
            environment_fingerprint=r['environment_fingerprint'], archive=r['archive'],
            preserved_failures=[v for k, v in r['files'].items() if 'noise' in k or '.failed' in k])
            for r in sources.envelopes.values()],
        limitations=['Original finite sweep winners; canonical replay is not another communication search.',
            'Baseline fingerprint is receipt-bound; standalone historical library binary is not archived in every run.',
            'Legacy sweep records physical GPU indices, not UUIDs; modern replays also verify UUIDs.',
            'Pure percentages use single GPU0 events, not distributed max-rank statistics.'],
        measurements=table if complete else rows)
    write_report(output, report, table)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = build_report(args.campaign, args.output)
    except (ValueError, KeyError, OSError, tarfile.TarError, TypeError) as error:
        parser.exit(1, f'BF16 report rejected: {error}\n')
    print(json.dumps(dict(complete=report['complete'], cells=report['complete_logical_launch_cells'],
                         missing=len(report['missing']), output=str(args.output))))
    return 0 if report['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
