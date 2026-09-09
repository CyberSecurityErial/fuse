#!/usr/bin/env python3
"""Summarize compact independent-C tile timelines, without summing parallel CTAs.

The last-observed-completion CTA is a proxy for the critical compute chain,
not proof of a CUDA kernel critical path. Its gaps are observer intervals,
not measured tensor-core idle cycles or a measured cuBLASLt gap difference.
"""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import summarize_sm103_fused as audit


def summarize(text):
    chains, samples = [], {0: [], 1: []}
    for line in text.splitlines():
        if not line.startswith(('profile_gemm_gap,', 'profile_gemm_gap_sample,')):
            continue
        fields = dict(item.split('=', 1) for item in line.split(',')[1:])
        if fields['component'] != 'compute_reference' or fields['rank'] != '0':
            raise ValueError('unexpected measurement component/rank')
        if line.startswith('profile_gemm_gap_sample,'):
            samples[int(fields['instrumented'])].append(float(fields['event_ms']))
        else:
            row = {key: int(value) for key, value in fields.items()
                   if key not in ('component', 'trace_event_ms')}
            row['trace_event_ms'] = float(fields['trace_event_ms'])
            if row['service_ns'] + row['signed_gap_ns'] != row['span_ns']:
                raise ValueError('CTA time accounting does not close')
            chains.append(row)
    if not chains or any(len(values) != 50 for values in samples.values()):
        raise ValueError('incomplete diagnostic (requires all chains and 10+50 sampling)')
    if len({row['cta'] for row in chains}) != len(chains):
        raise ValueError('duplicate CTA')
    critical = max(chains, key=lambda row: row['last_completion_ns'])
    plain, observed = (statistics.median(samples[i]) for i in (0, 1))
    stability = {str(i): abs(statistics.median(values[:25]) - statistics.median(values[25:]))
                 / statistics.median(values) for i, values in samples.items()}
    return dict(schema='sm103_gemm_gap_observation_v1', component='compute_reference',
                launch='eager', rank=0, cta_count=len(chains),
                tile_count=sum(row['tiles'] for row in chains),
                plain_p50_ms=plain, instrumented_p50_ms=observed,
                instrumentation_overhead_pct=100*(observed/plain-1),
                sample_half_drift=stability, sampling_stable=all(x <= .05 for x in stability.values()),
                last_completion_cta=critical['cta'],
                last_completion_chain=critical,
                observed_chain_gap_fraction=critical['signed_gap_ns']/(critical['trace_event_ms']*1e6),
                cublaslt_gap_difference_measured=False,
                causal_throughput_loss_share=None, chains=chains, samples_ms=samples)


def read_observation(run):
    """Receipt-bound input only; do not trust editable gaps.json summaries."""
    job, records, data, evidence = audit.read_receipts(run)
    audit.require(job.get('oproj_gap_probe') is True and job.get('profile') is True and
                  job.get('directions') == 'oproj' and not job.get('mpi'), 'Wrong gap probe scope')
    text = data[f'attempt{records["status.json"]["attempt"]}.log'].decode()
    resources = []
    for line in text.splitlines():
        if line.startswith('candidate,A2A_GEMM,'):
            fields = dict(item.split('=', 1) for item in line.split(',')[2:])
            resources.append(fields)
    fields = ('tile', 'tile_m', 'tile_n', 'tile_k', 'threads', 'dynamic_smem', 'raster',
              'max_swizzle_size', 'effective_swizzle_size', 'scheduled_compute_ctas', 'oproj_comm_layout')
    audit.require(resources and all(all(field in row for field in fields) for row in resources),
                  'Gap profile lacks complete resolved GEMM configuration')
    signatures = {tuple(row[field] for field in fields) for row in resources}
    audit.require(len(signatures) == 1, 'Multiple physical GEMM configurations in one gap trace')
    resolved = dict(zip(fields, signatures.pop()))
    shape = audit.fused_geometry(job)
    gpu = records['gpu-before.json']
    physical = gpu['selected'][0]
    devices = [row for row in gpu['observations'][0]['devices'] if row['index'] == physical]
    audit.require(len(devices) == 1 and devices[0].get('uuid'), 'Missing gap rank0 GPU identity')
    identity = dict(m=shape['seq_local'], n=shape['hidden'], k=shape['q_width'], cp=shape['world'],
        compute_sms=int(resolved.pop('scheduled_compute_ctas')), node=job['node'], source_id=job['source_id'],
        binary_sha256=records['fused-build.json']['binary_sha256'],
        build_inputs=records['fused-build.json']['build_inputs'],
        environment_fingerprint=records['environment.json']['fingerprint'],
        physical_gpu_uuid=devices[0]['uuid'], causal=bool(job.get('causal')), **resolved)
    measured = summarize(text)
    measured.update(run_id=job['run_id'], source_id=job['source_id'], profile_identity=identity,
        raw_log_sha256=hashlib.sha256(data[f'attempt{records["status.json"]["attempt"]}.log']).hexdigest(),
        artifact_sha256=evidence['artifacts.tar.gz']['sha256'])
    return identity, measured


def comparison(table, run_root):
    """Require an explicit, complete profile_identity on each comparison row.

    It identifies the intended diagnostic build/configuration, not the historical
    production binary (profiling necessarily uses a different build). Older tables
    without this identity remain unjoined. MNK and a compute budget are insufficient
    to distinguish tile/raster/swizzle, route layout, CP or device/source changes.
    """
    measurements = {}
    for path in sorted(run_root.glob('*/gaps.json')):
        root = path.parent.resolve()
        job = audit.json_bytes(audit.read_bytes(path.with_name('job.json'), root))
        if not job.get('oproj_gap_probe'):
            continue
        receipt = audit.json_bytes(audit.read_bytes(path.with_name('fetched.json'), root))
        if receipt['state'] != 'succeeded':
            continue
        identity, measured = read_observation(path.parent)
        key = json.dumps(identity, sort_keys=True)
        audit.require(key not in measurements, 'Duplicate gap profile identity; select explicit evidence first')
        measurements[key] = measured
    rows = []
    for original in table['rows']:
        row = {key: original.get(key) for key in ('model', 'cp', 'seq', 'm', 'n', 'k',
                'compute_sms', 'control', 'a_ms', 'b_ms', 'retained_pct')}
        loss = None if original['a_ms'] is None else 1-original['b_ms']/original['a_ms']
        row.update(historical_loss_pct=None if loss is None else 100*loss,
                   measured_causal_loss_share_pct=None)
        identity = original.get('profile_identity')
        required = next(iter(measurements.values()))['profile_identity'].keys() if measurements else ()
        if not isinstance(identity, dict) or not required or set(identity) != set(required):
            row['join_status'] = 'unjoined_missing_identity'
            rows.append(row)
            continue
        audit.require(all(identity[key] == original[key] for key in ('m', 'n', 'k', 'cp', 'compute_sms')),
                      'Comparison row/profile identity geometry or budget mismatch')
        measured = measurements.get(json.dumps(identity, sort_keys=True))
        row['join_status'] = 'matched' if measured else 'unjoined_identity_mismatch'
        if measured:
            gap = measured['observed_chain_gap_fraction']
            row.update(run_id=measured['run_id'], profile_identity=identity, sampling_stable=measured['sampling_stable'],
                       cta_count=measured['cta_count'], tile_count=measured['tile_count'],
                       plain_eager_ms=measured['plain_p50_ms'],
                       observed_gap_us=measured['last_completion_chain']['signed_gap_ns']/1000,
                       observed_gap_fraction_pct=100*gap,
                       instrumentation_overhead_pct=measured['instrumentation_overhead_pct'],
                       # Scale comparison only: Eager instrumented gap fraction versus
                       # historical Graph loss, NOT a causal attribution/bound.
                       gap_to_historical_loss_scale_pct=(100*gap/loss if loss and loss > 0 else None))
        rows.append(row)
    return dict(schema='sm103_gemm_gap_comparison_v1', rows=rows,
                observations=list(measurements.values()),
                caveat='Eager diagnostic versus historical Graph a/b. Gap/loss is a scale comparison, not measured attribution. Independent C only; no fusion/communication gap claim.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--log', type=Path)
    mode.add_argument('--table', type=Path)
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; do not overwrite evidence')
    if args.table and args.run_root is None:
        parser.error('--table requires --run-root')
    result = (summarize(args.log.read_text()) if args.log else
              comparison(json.loads(args.table.read_text()), args.run_root))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    if args.table:
        keys = list(dict.fromkeys(key for row in result['rows'] for key in row))
        with args.output.with_suffix('.csv').open('w') as out:
            writer = csv.DictWriter(out, fieldnames=keys)
            writer.writeheader()
            writer.writerows(result['rows'])
        print(json.dumps(dict(rows=len(result['rows']),
                              covered=sum('run_id' in row for row in result['rows']))))
        return
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('chains', 'samples_ms', 'last_completion_chain')}))


if __name__ == '__main__':
    main()
