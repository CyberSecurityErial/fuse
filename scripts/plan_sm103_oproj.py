#!/usr/bin/env python3
"""Plan measured-budget OProj candidates offline; never launch GPUs or select a runtime policy.

Cases: {"schema":"sm103_oproj_model_cases_v1", "cases":[{"id":"case",
 "m":32768,"n":8192,"k":12288,"world":4,"sm_count":148,"candidates":[
 {"tile_policy":"m128n256","raster":"along_m","max_swizzle_size":8,
  "comm_ctas":[8,16,24,32,48]}]}]}.

Calibration contains complete run objects already audited by summarize_sm103_fused.
Only same-candidate independent C/R totals supply service estimates. C/waves and
copy-slot workload/R are *amortized effective* services, not measured steady-state
Tensor Core/transport rates. Each anchor's actual queue determines its busiest
slot; this removes double-counted slot imbalance, not real startup or contention.
The target queue still determines its own imbalance. The model adds no separately
calibrated startup/drain term; this does not assert real startup/drain is zero.
F is used only in calibration reconstruction checks, never to fit or rank.
Export an independent C/R-only C++ table with --calibration FILE --export-cpp HEADER;
this mode does not accept --cases or --output and never overwrites an existing file.
"""

import argparse
from bisect import bisect_left
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks/sm103'))
import fused_model as model


CALIBRATION_SCHEMA = 'sm103_oproj_primitive_calibration_runs_v1'
CASES_SCHEMA = 'sm103_oproj_model_cases_v1'
POLICIES = ('m128n256', 'm128n256k64e32')
COPY_SERVICE_MODEL = 'oproj_static_slot_service_v2'
FIXED = dict(precision='bf16_accfp32_bf16', layout='causal_dual_chunk_v1',
             oproj_comm_layout='rows', launch='graph', direction='A2A_GEMM')
PAIR_FIELDS = ('m', 'n', 'k', 'world', 'sm_count', 'comm_ctas', 'tile_policy', 'tile_m',
               'tile_n', 'tile_k', 'cluster_ctas', 'raster', 'max_swizzle_size', 'swizzle')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, name):
    require(type(value) is int and 0 < value <= 2**31 - 1, f'{name}: positive int32 required')
    return value


def positive(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
            f'{name}: positive finite number required')
    return float(value)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def swizzle_for(m, n, maximum):
    minimum = min(model.ceil_div(m, 128), model.ceil_div(n, 256))
    return (8 if maximum >= 8 and minimum >= 6 else 4 if maximum >= 4 and minimum >= 3
            else 2 if maximum >= 2 and minimum >= 2 else 1)


def bucket_key(row):
    return tuple(row[key] for key in ('world', 'sm_count', 'tile_policy', 'raster',
                                      'max_swizzle_size', 'swizzle', 'comm_ctas'))


def resource_signature(row):
    resources = model.primitive_row_resources(row)
    # Preserve unknown collective internals instead of deriving them from a name.
    return dict(policy=row['tile_policy'], cluster_ctas=row['cluster_ctas'],
                production=resources['production'], actual_epilogue_tile=None,
                ab_stages=None, accumulator_stages=None)


def rows_copy_geometry(k, world, comm):
    """The existing BF16 pull path: four 48-KiB warp slots per comm CTA."""
    row_bytes = 2 * k // world
    rows = min(128, 48 * 1024 // row_bytes)
    require(rows > 0, 'Vector copy fallback is not calibrated')
    chunks, slots = model.ceil_div(128, rows), 4 * comm
    return dict(copy_chunks=chunks, copy_slots=slots, cohort_m_tiles=max(1, slots // chunks),
                copy_chunk_bytes=[row_bytes * min(rows, 128 - i * rows) for i in range(chunks)])


def calibrate_copy_slots(m, n, world, compute, raster, swizzle, geometry, r_us):
    """Invert the fixed-slot service model using independent R only.

    R already includes the slowest slot's chunk/tail distribution. Using
    unique_bytes/R as aggregate slot bandwidth would charge that imbalance a
    second time. Instead, per-slot service = max(slot_bytes)/R; multiply by the
    slot count for the scorer's aggregate-bandwidth convention. This guarantees
    anchor R reconstruction under the model, not a measured physical bandwidth
    or a guarantee that slots remain equally serviced during fused execution.
    """
    mt, nt = model.ceil_div(m, 128), model.ceil_div(n, 256)
    chunks, slots = geometry['copy_chunks'], geometry['copy_slots']
    padded = model.ceil_div(mt, swizzle) * swizzle * model.ceil_div(nt, swizzle) * swizzle
    require(max(padded, mt * world * chunks, slots) <= 1_000_000, 'Copy calibration plan is too large')
    _, _, tasks, _ = model._oproj_delivery_plan(
        mt, nt, swizzle, compute, raster, world, chunks, geometry['cohort_m_tiles'])
    loads = [0] * slots
    for index, (_, _, chunk) in enumerate(tasks):
        loads[index % slots] += geometry['copy_chunk_bytes'][chunk]
    payload, busiest = sum(loads), max(loads)
    return dict(model=COPY_SERVICE_MODEL, copy_slots=slots, copy_chunks=chunks,
        max_slot_bytes=busiest, payload_bytes=payload,
        imbalance_factor=busiest * slots / payload,
        observed_payload_gb_s=payload / r_us / 1000,
        copy_bandwidth_gb_s=busiest * slots / r_us / 1000)


def read_calibration(document):
    require(document.get('schema') == CALIBRATION_SCHEMA and document.get('runs'),
            'Expected nonempty audited primitive calibration runs')
    anchors, run_ids, contract, signatures = defaultdict(dict), set(), None, {}
    for run in document['runs']:
        run_id, build, config = run['run_id'], run['build'], run['config']
        require(run_id not in run_ids, 'Duplicate calibration run')
        run_ids.add(run_id)
        require(run.get('diagnostic_only') is False and build.get('profile') is False and
                build.get('mpi') is True and config.get('calibrate') == '1' and
                config.get('profile') == config.get('validation_self_test') == '0' and
                config.get('launch') == 'graph' and config.get('causal') == '1',
                'Calibration must be non-profile MPI Graph independent C/R')
        current = {key: run[key] for key in ('node', 'environment_fingerprint')}
        current.update({key: build[key] for key in ('build_inputs', 'binary_sha256')})
        current['input_generator'] = config['input_generator']
        require(all(current.values()) and build['environment_fingerprint'] == current['environment_fingerprint'],
                'Missing/mismatched calibration execution identity')
        require(contract is None or contract == current, 'Mixed calibration node/build/environment contracts')
        contract = current
        groups = defaultdict(dict)
        for row in run['candidates']:
            component = row.get('component')
            require(component in ('fused', 'compute_reference', 'copy_reference'), 'Unknown calibration component')
            group = groups[integer(row['candidate'], 'candidate')]
            require(component not in group, 'Duplicate calibration candidate/component')
            group[component] = row
        for group in groups.values():
            require({'compute_reference', 'copy_reference'} <= set(group), 'Missing same-candidate C/R pair')
            c, r = group['compute_reference'], group['copy_reference']
            require(all(c.get(key) == r.get(key) for key in PAIR_FIELDS), 'C/R candidate geometry mismatch')
            for row in (c, r):
                require(row.get('performance_accepted') is True and row.get('measurement_role') == 'calibration' and
                        all(row.get(key) == value for key, value in FIXED.items()), 'Unsupported C/R measurement scope')
                require(row['tile_policy'] in POLICIES and
                        (row['tile_m'], row['tile_n'], row['tile_k'], row['cluster_ctas']) == (128, 256, 64, 1),
                        'Uncalibrated tile/collective signature')
                model.primitive_schedule(row)
                require(row['world'] in (4, 8) and row['sm_count'] == 148 and
                        all(row[key] == run['geometry'][field] for key, field in
                            (('m', 'seq_local'), ('n', 'hidden'), ('k', 'q_width'), ('world', 'world'))),
                        'Candidate/run geometry mismatch')
            signature = resource_signature(c)
            require(signature == resource_signature(r), 'C/R production resource signature mismatch')
            key = bucket_key(c)
            require(key not in signatures or signatures[key] == signature, 'Ambiguous actual tile signature')
            signatures[key] = signature
            m, n, k, world, comm = (integer(c[field], field) for field in ('m', 'n', 'k', 'world', 'comm_ctas'))
            require(8192 <= k <= 16384 and m % 128 == 0 and k % (64 * world) == 0 and comm < 148,
                    'Calibration outside supported K/ready/budget range')
            work = model.ceil_div(m, 128) * model.ceil_div(n, 256)
            scheduled = c['scheduled_work_tiles_derived']
            compute = min(scheduled, 148 - comm)
            require(c['compute_ctas_derived'] == [compute] * world and
                    r['compute_ctas_derived'] == [0] * world and
                    c['production_compute_ctas_derived'] == r['production_compute_ctas_derived'] == [compute] * world and
                    c['work_tiles_derived'] == r['work_tiles_derived'] == work,
                    'Independent reference changed the compute budget/work')
            payload = 2 * m * k
            require(c['executed_gemm_flops'] == 2 * m * n * k and c['executed_route_payload_bytes'] == 0 and
                    r['executed_gemm_flops'] == 0 and r['executed_route_payload_bytes'] == payload and
                    c['problem_route_payload_bytes'] == r['problem_route_payload_bytes'] == payload,
                    'Independent C/R work boundary mismatch')
            require(k not in anchors[key], 'Duplicate K anchor: select explicit audited evidence first')
            c_us = positive(c['timing']['p50_ms'], 'C p50') * 1000
            r_us = positive(r['timing']['p50_ms'], 'R p50') * 1000
            copy_service = calibrate_copy_slots(m, n, world, compute, c['raster'], c['swizzle'],
                                               rows_copy_geometry(k, world, comm), r_us)
            require(copy_service['payload_bytes'] == payload, 'Copy plan changed the independent R payload')
            anchors[key][k] = dict(m=m, n=n, k=k, candidate=c['candidate'], run_id=run_id,
                source_id=run['source_id'], signature=signature, compute_ctas=compute,
                c_us=c_us, r_us=r_us, tile_cycle_us=c_us / model.ceil_div(scheduled, compute),
                copy_bandwidth_gb_s=copy_service['copy_bandwidth_gb_s'], copy_service=copy_service,
                sampling_mode=c['sampling_mode'], formal_eligible=c['formal_eligible'],
                fused_p50_us=(positive(group['fused']['timing']['p50_ms'], 'F p50') * 1000
                              if 'fused' in group else None))
    require(anchors, 'No independent C/R anchors')
    return dict(contract=contract, anchors=dict(anchors), calibration_sha256=digest(document))


def interpolate(points, k):
    ks = sorted(points)
    require(8192 <= k <= 16384 and ks[0] <= k <= ks[-1], 'K outside calibrated bracket; no extrapolation')
    index = bisect_left(ks, k)
    selected = [points[k]] if k in points else [points[ks[index - 1]], points[ks[index]]]
    alpha = 0.0 if len(selected) == 1 else (k - selected[0]['k']) / (selected[1]['k'] - selected[0]['k'])
    values = {field: selected[0][field] * (1 - alpha) + selected[-1][field] * alpha
              for field in ('tile_cycle_us', 'copy_bandwidth_gb_s')}
    return values, selected


def predict(calibration, case, candidate, comm):
    m, n, k, world, sm = (integer(case[field], field) for field in ('m', 'n', 'k', 'world', 'sm_count'))
    require('global_seq' not in case or case['global_seq'] == m * world, 'Case global_seq differs from M*CP')
    maximum = integer(candidate['max_swizzle_size'], 'max_swizzle_size')
    params = dict(tile_policy=candidate['tile_policy'], raster=candidate['raster'],
                  max_swizzle_size=maximum, comm_ctas=integer(comm, 'comm_ctas'))
    unsupported = dict(status='unsupported', parameters=params)
    if any(case.get(key, value) != value for key, value in FIXED.items()):
        return unsupported | {'reason': 'Uncalibrated precision/route/launch'}
    if not (candidate['tile_policy'] in POLICIES and candidate['raster'] in ('along_m', 'along_n') and
            maximum in (1, 2, 4, 8) and world in (4, 8) and sm == 148 and comm < sm and
            m % 128 == 0 and k % (world * 64) == 0):
        return unsupported | {'reason': 'Uncalibrated collective or unsupported ready/scheduler geometry'}
    resolved = swizzle_for(m, n, maximum)
    key = bucket_key(case | params | {'swizzle': resolved})
    if key not in calibration['anchors']:
        return unsupported | {'reason': 'No exact CP/collective/raster/swizzle/communication-budget calibration'}
    try:
        service, evidence = interpolate(calibration['anchors'][key], k)
    except ValueError as error:
        return unsupported | {'reason': str(error)}
    scheduled = (model.ceil_div(model.ceil_div(m, 128), resolved) * resolved *
                 model.ceil_div(model.ceil_div(n, 256), resolved) * resolved)
    compute = min(scheduled, sm - comm)
    if any(anchor['compute_ctas'] != compute for anchor in evidence):
        return unsupported | {'reason': 'Actual compute CTA budget differs from the independent C anchors'}
    try:
        copy_geometry = rows_copy_geometry(k, world, comm)
    except ValueError as error:
        return unsupported | {'reason': str(error)}
    score = model.score_oproj_schedule(m=m, n=n, k=k, world=world, sm_count=sm, comm_ctas=comm,
        tile_m=128, tile_n=256, tile_k=64, raster=params['raster'], resolved_swizzle=resolved,
        **copy_geometry, **service, launch_us=0.0, copy_start_us=0.0, tail_us=0.0,
        service_basis='amortized_full_boundary')
    for field in ('launch_us', 'copy_start_us', 'tail_us'):
        score.pop(field)  # API zero addends are not separately measured zero-latency estimates.
    return dict(status='predicted', parameters=params | {'resolved_swizzle': resolved}, prediction=score,
        service_estimates=service, signature=evidence[0]['signature'],
        evidence=[{key: anchor[key] for key in ('run_id', 'source_id', 'candidate', 'm', 'n', 'k',
                  'compute_ctas', 'c_us', 'r_us', 'copy_service', 'sampling_mode', 'formal_eligible')} for anchor in evidence],
        interpolation='exact_K' if len(evidence) == 1 else 'linear_K_bracket',
        cross_mn=any((anchor['m'], anchor['n']) != (m, n) for anchor in evidence),
        externally_validated=False, measured_startup_us=None,
        service_semantics='effective_cycle_and_slot_service_amortized_from_independent_totals',
        copy_service_model=COPY_SERVICE_MODEL,
        caveats=['C/waves includes launch, drain and load imbalance; not a measured steady Tensor Core cycle',
                 'R-derived slot service includes local traffic and startup; it is not observed aggregate payload bandwidth',
                 'anchor slot imbalance is inverted once; target geometry retains its own slot imbalance',
                 'no separate startup/drain addend: uncalibrated, not measured zero',
                 'cross-M/N reuse and joint resource contention need external fused validation'])


def reconstruction_summary(checks):
    """Validate frozen predictions against F, without feeding results back into selection."""
    groups = defaultdict(list)
    for row in checks:
        groups[digest(row['geometry'] | row['schedule'])].append(row)
    comparisons = []
    for rows in groups.values():
        require(len({row['comm_ctas'] for row in rows}) == len(rows), 'Duplicate reconstruction budget')
        predicted = sorted(rows, key=lambda row: (row['predicted_us'], row['comm_ctas']))
        best_observed = min(row['observed_fused_us'] for row in rows)
        winners = [row['comm_ctas'] for row in rows if row['observed_fused_us'] == best_observed]
        top2 = [row['comm_ctas'] for row in predicted[:2]]
        comparisons.append(dict(geometry=rows[0]['geometry'], schedule=rows[0]['schedule'],
            budgets=[row['comm_ctas'] for row in rows], observed_best_comm_ctas=winners,
            predicted_top2_comm_ctas=top2, top1_hit=top2[0] in winners,
            top2_hit=bool(set(top2) & set(winners))))
    errors = [abs(row['relative_error']) for row in checks]
    return dict(scope='calibration_reconstruction_only_not_external_validation', points=len(checks),
        mean_absolute_relative_error=sum(errors) / len(errors) if errors else None,
        median_absolute_relative_error=model.percentile(errors, .5) if errors else None,
        p95_absolute_relative_error=model.percentile(errors, .95) if errors else None,
        max_absolute_relative_error=max(errors) if errors else None,
        mean_signed_relative_error=sum(row['relative_error'] for row in checks) / len(checks) if checks else None,
        budget_groups=len(comparisons), five_budget_groups=sum(len(row['budgets']) == 5 for row in comparisons),
        top1_hits=sum(row['top1_hit'] for row in comparisons), top2_hits=sum(row['top2_hit'] for row in comparisons),
        top2_hit_rate=sum(row['top2_hit'] for row in comparisons) / len(comparisons) if comparisons else None,
        groups=comparisons)


def plan(calibration_document, cases_document):
    calibration = read_calibration(calibration_document)
    require(cases_document.get('schema') == CASES_SCHEMA and isinstance(cases_document.get('cases'), list),
            'Expected explicit model cases schema')
    results, ids = [], set()
    for case in cases_document['cases']:
        require(isinstance(case.get('id'), str) and case['id'] and case['id'] not in ids, 'Missing/duplicate case id')
        ids.add(case['id'])
        require(isinstance(case.get('candidates'), list) and case['candidates'], 'Explicit candidate list required')
        candidates, seen = [], set()
        for spec in case['candidates']:
            require(isinstance(spec.get('comm_ctas'), list) and spec['comm_ctas'], 'Explicit communication budgets required')
            for comm in spec['comm_ctas']:
                key = spec['tile_policy'], spec['raster'], spec['max_swizzle_size'], comm
                require(key not in seen, 'Duplicate physical candidate')
                seen.add(key)
                candidates.append(predict(calibration, case, spec, comm))
        ranked = sorted((row for row in candidates if row['status'] == 'predicted'),
                        key=lambda row: row['prediction']['score_us'])
        metadata = {key: value for key, value in case.items()
                    if key not in ('id', 'm', 'n', 'k', 'world', 'sm_count', 'candidates')}
        results.append(dict(id=case['id'], metadata=metadata,
                            geometry={key: case[key] for key in ('m', 'n', 'k', 'world', 'sm_count')},
                            candidates=candidates, top2=[row['parameters'] for row in ranked[:2]]))
    checks = []
    for key, points in calibration['anchors'].items():
        world, sm, policy, raster, maximum, _, comm = key
        for anchor in points.values():
            if anchor['fused_p50_us'] is None:
                continue
            case = dict(m=anchor['m'], n=anchor['n'], k=anchor['k'], world=world, sm_count=sm)
            candidate = dict(tile_policy=policy, raster=raster, max_swizzle_size=maximum)
            predicted = predict(calibration, case, candidate, comm)['prediction']['score_us']
            checks.append(dict(run_id=anchor['run_id'], candidate=anchor['candidate'], predicted_us=predicted,
                geometry=case, schedule=candidate, comm_ctas=comm,
                observed_fused_us=anchor['fused_p50_us'], relative_error=predicted / anchor['fused_p50_us'] - 1))
    return dict(schema='sm103_oproj_model_plan_v1', fixed_scope=FIXED, execution_contract=calibration['contract'],
        calibration_sha256=calibration['calibration_sha256'], cases_sha256=digest(cases_document),
        input_metadata={key: value for key, value in cases_document.items() if key not in ('schema', 'cases')},
        cases=results, calibration_reconstruction_checks=checks, fitted_to_fused=False,
        calibration_reconstruction_summary=reconstruction_summary(checks),
        globally_optimal=False, externally_validated=False, runtime_selector=False,
        model_code_sha256=digest(Path(model.__file__).read_text()),
        planner_code_sha256=digest(Path(__file__).read_text()), copy_service_model=COPY_SERVICE_MODEL,
        assumptions=['K-only interpolation within measured [8192,16384] brackets',
                     'exact measured compute budget per candidate; no throughput proportional to SM assumption',
                     'M/N transfer is an unvalidated effective-cycle approximation',
                     'available actual resource signature is enforced; epilogue/AB/TMEM internals remain unknown',
                     'F reconstruction is diagnostic only; top2 are model proposals for measurement'])


def export_cpp(document):
    """Export primitive anchors, never a shape-winner table or F-derived digest."""
    cr_document = dict(schema=document.get('schema'), runs=[run | {'candidates': [
        row for row in run['candidates'] if row.get('component') != 'fused']}
        for run in document.get('runs', [])])
    calibration = read_calibration(cr_document)
    points, evidence, identities = [], [], set()
    for key, anchors in sorted(calibration['anchors'].items()):
        world, _, policy, raster, _, swizzle, comm = key
        for k, anchor in sorted(anchors.items()):
            point = dict(world=world, policy_index=POLICIES.index(policy), along_n=raster == 'along_n',
                swizzle=swizzle, comm_ctas=comm, k=k, reference_m=anchor['m'], reference_n=anchor['n'],
                compute_ctas=anchor['compute_ctas'], tile_cycle_us=anchor['tile_cycle_us'],
                copy_slot_bandwidth_gb_s=anchor['copy_bandwidth_gb_s'])
            identity = tuple(point[name] for name in
                             ('world', 'policy_index', 'along_n', 'swizzle', 'comm_ctas', 'k'))
            require(identity not in identities, 'Duplicate physical C++ calibration anchor')
            identities.add(identity)
            points.append(point)
            evidence.append({name: anchor[name] for name in ('run_id', 'source_id', 'candidate',
                'm', 'n', 'k', 'compute_ctas', 'c_us', 'r_us', 'copy_service', 'signature',
                'sampling_mode', 'formal_eligible')})
    version = digest(dict(model=COPY_SERVICE_MODEL, scope=FIXED,
                          execution=calibration['contract'], points=points, evidence=evidence))
    runs = sorted({row['run_id'] for row in evidence})
    require(all(run and all(c.isascii() and (c.isalnum() or c in '._-') for c in run) for run in runs),
            'Unsafe calibration run identifier')
    lines = ['// SPDX-License-Identifier: BSD-3-Clause', '#pragma once', '#include <cstdint>', '',
        '// Generated by scripts/plan_sm103_oproj.py --export-cpp; do not edit by hand.',
        '// Independent C/R only; every physical anchor is retained, with no F/winner fitting.',
        '// Domain: 148 SM, CUDA-reported compute capability 10.3 (sm_103a), BF16/FP32/BF16,',
        '// CUTLASS 57e3cfb47a2d9e0d46eb6335c3dc411498efa198, CP4/8, causal rows-pull, Graph.',
        '// Match collective/raster/resolved swizzle and actual compute budget; do not extrapolate.',
        '// C/waves and max-slot-bytes/R are amortized services, including startup/drain.',
        '// Aggregate slot service is NOT pure NVLink or observed aggregate payload bandwidth.',
        '// The target queue retains its own slot imbalance; fused contention needs validation.',
        '// policy_index: 0 = m128n256, 1 = m128n256k64e32.',
        '// Version hashes C/R values, geometry, resources, execution identity and source evidence.',
        '// Primitive source runs:'] + [f'//   {run}' for run in runs] + [
        '', 'namespace fuse::detail {', '', 'struct OprojCalibrationPoint {',
        '  int32_t world, policy_index;', '  bool along_n;',
        '  int32_t swizzle, comm_ctas, k, reference_m, reference_n, compute_ctas;',
        '  double tile_cycle_us, copy_slot_bandwidth_gb_s;', '};', '',
        f'inline constexpr char kOprojCalibrationVersion[] = "{version}";',
        'inline constexpr OprojCalibrationPoint kOprojCalibrationPoints[] = {']
    for point in points:
        values = [('true' if value else 'false') if type(value) is bool else
                  format(value, '.17g') if type(value) is float else str(value) for value in point.values()]
        lines.append('  {' + ', '.join(values) + '},')
    lines.extend(['};', '', '}  // namespace fuse::detail', ''])
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--cases', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--export-cpp', type=Path)
    args = parser.parse_args(argv)
    if args.export_cpp:
        if args.cases or args.output:
            parser.error('--export-cpp cannot be combined with --cases or --output')
        header = export_cpp(json.loads(args.calibration.read_text()))
        args.export_cpp.parent.mkdir(parents=True, exist_ok=True)
        with args.export_cpp.open('x') as stream:
            stream.write(header)
        print(json.dumps(dict(export_cpp=str(args.export_cpp))))
        return
    if not args.cases or not args.output:
        parser.error('Planning requires both --cases and --output, or use --export-cpp alone')
    result = plan(json.loads(args.calibration.read_text()), json.loads(args.cases.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.output), cases=len(result['cases']),
                         predicted=sum(c['status'] == 'predicted' for r in result['cases'] for c in r['candidates']))))


if __name__ == '__main__':
    main()
