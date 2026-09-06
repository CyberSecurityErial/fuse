#!/usr/bin/env python3
"""Freeze an experimental QKV-F steady-state role model, not a production default.

Only measured compute/route CTA envelopes enter the fit. Route includes producer
readiness and is NOT bare communication: fit max(measured_compute, route_service),
then predict max(predicted_compute, route_service). DQ, finalize, complete forward
latency and measured winners are never fitting inputs. Math/validation use the
standard library; only fitting imports NumPy/SciPy. No CUDA imports or launches.

Example:
  python qkv_forward_service_model.py --calibration calibration_summary.json --output model.json
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics


SCHEMA = 'mxfp8-qkv-forward-service-model-v1'
CALIBRATION_SCHEMA = 'mxfp8-route-compute-calibration-v1'
SCOPE = 'QKV_forward_CP4_fixed_tile_steady_role_envelopes_not_complete_F_or_2F2B'
TILE_KEYS = ('tile_m', 'tile_n', 'tile_k', 'cluster_m')
TILE = (128, 256, 64, 2)
COMM_CANDIDATES = tuple(range(4, 25, 2))
COEFFICIENT_KEYS = ('compute_gflop_sm_us', 'route_slot_task_us')
COPY_SLOTS, COPY_ROWS, COPY_COLUMNS = 12, 64, 128
EXPOSED_FRACTION = .05
REQUIRED_SOURCES = (
    'benchmarks/QKVproj+a2a/qkv_shape_bench.py',
    'benchmarks/mxfp8_weight/operator_bench.py',
    'benchmarks/mxfp8_weight/operator_bridge.cu',
    'benchmarks/mxfp8_weight/operator_perf_runtime.py',
    'benchmarks/mxfp8_weight/operator_profile.cuh',
    'benchmarks/mxfp8_weight/profile_operators.py',
    'benchmarks/mxfp8_weight/profile_report.py',
    'csrc/operators/ulysses_sm90/api/forward.cuh',
    'csrc/operators/ulysses_sm90/api/policy.cuh',
    'csrc/operators/ulysses_sm90/detail/core.cuh',
    'csrc/operators/ulysses_sm90/detail/gemm_a2a.cuh',
    'csrc/operators/ulysses_sm90/detail/launch.cuh',
    'include/fuse/operators/primitives/gemm_a2a.h',
    'include/fuse/schedule/cutlass_pipeline.cuh',
    'include/fuse/schedule/persistent_gemm.cuh',
)


def integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def finite(value, name, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')
    return value


def expect(record, expected, name):
    for key, value in expected.items():
        if (key not in record or record[key] != value or
                (type(value) in (int, bool) and type(record[key]) is not type(value))):
            raise ValueError(f'{name}.{key} differs from the declared contract')


def sha256(value, name):
    if (not isinstance(value, str) or len(value) != 64 or
            any(character not in '0123456789abcdef' for character in value)):
        raise ValueError(f'{name} must be a lowercase SHA256 digest')
    return value


def features(case, config, comm_ctas=None):
    """Work for the fixed tile, not integer waves or estimated HBM transactions.

    The optional c changes only the resource split of this already-fixed family.
    No native auto selector is assumed to preserve its tile when c changes.
    """
    m, n, k = [integer(case[key], key) for key in ('m', 'n', 'k')]
    q, kv = [integer(case[key], key) for key in ('q_heads', 'kv_heads')]
    expect(case, dict(direction='gemm_a2a', cp=4, batch=1, layout='rank_major',
                      head_dim=128, global_seq=4 * m, hidden=k), 'case')
    expect(config, dict(sm_count=132, raster='n', swizzle=1, gemm_input='bf16',
                        route_layout='rank_major'), 'config')
    if tuple(integer(config[key], key) for key in TILE_KEYS) != TILE:
        raise ValueError('Only M128N256K64/C2 is calibrated')
    if (max(m, n, k) >= 2**31 or m < 32768 or m % 256 or n % 256 or k % 64 or
            q % 4 or kv % 4 or q % kv or n != (q + 2 * kv) * 128):
        raise ValueError('Geometry is outside the long, aligned head128 CP4 family')
    for record in (case, config):
        if any(record.get(key, False) is not False for key in
               ('causal_load_balanced', 'qkv_peer_interleaved', 'defer_v_a2a')):
            raise ValueError('Only the ordinary complete rank-major QKV route is calibrated')
    c = integer(config['comm_ctas'] if comm_ctas is None else comm_ctas, 'comm_ctas')
    if c not in COMM_CANDIDATES:
        raise ValueError('Explicit communication CTAs must be even and in [4,24]')
    return dict(m=m, n=n, k=k, compute_ctas=132 - c, comm_ctas=c,
                work_gflop=2 * m * n * k / 1e9,
                route_tasks=m * n // (COPY_ROWS * COPY_COLUMNS),
                route_slots=COPY_SLOTS * c)


def service_times(feature, coefficients):
    a, tau = [finite(coefficients[key], key, positive=True) for key in COEFFICIENT_KEYS]
    compute = a * feature['work_gflop'] / feature['compute_ctas']
    route = tau * feature['route_tasks'] / feature['route_slots']
    finite(compute, 'predicted compute', positive=True)
    finite(route, 'predicted route service', positive=True)
    return dict(compute_us=compute, route_service_us=route,
                role_envelope_us=max(compute, route))


def validate_model(model):
    expect(model, dict(world_size=4, sm_count=132), 'model')
    if finite(model['minimum_gain'], 'minimum_gain') != 0:
        raise ValueError('This predeclared formula has no gain gate')
    if set(model['coefficients']) != set(COEFFICIENT_KEYS):
        raise ValueError('The steady model has exactly two coefficients')
    for key in COEFFICIENT_KEYS:
        finite(model['coefficients'][key], key, positive=True)
    domain = model['domain']
    expect(domain, dict(tile_families=[list(TILE)], head_dim=128,
                        m_multiple=256, n_multiple=256, k_multiple=64,
                        comm_ctas=list(COMM_CANDIDATES)), 'domain')
    keys = {'min_m', 'max_m', 'min_n', 'max_n', 'min_k', 'max_k',
            'm_multiple', 'n_multiple', 'k_multiple', 'head_dim', 'tile_families', 'comm_ctas'}
    if set(domain) != keys:
        raise ValueError('The native domain has an exact physical-field contract')
    for key in ('m', 'n', 'k'):
        low, high = [integer(domain[prefix + key], key) for prefix in ('min_', 'max_')]
        if low > high or low % domain[key + '_multiple'] or high % domain[key + '_multiple']:
            raise ValueError(f'Invalid aligned {key} calibration range')
    return model


def predict(case, config, model, comm_ctas=None):
    """Use a caller-supplied frozen model; reject unsupported extrapolation."""
    validate_model(model)
    feature = features(case, config, comm_ctas)
    for key in ('m', 'n', 'k'):
        low, high = [model['domain'][prefix + key] for prefix in ('min_', 'max_')]
        if not low <= feature[key] <= high:
            raise ValueError(f'{key} is outside the supplied calibration domain')
    return service_times(feature, model['coefficients'])


def select(case, config, model):
    """Minimize the declared score, not observations; exact ties retain baseline.

    minimum_gain is zero: there is no fitted gain gate, shape winner table,
    automatic calibration lookup, or mutation of production dispatch.
    """
    scores = {c: predict(case, config, model, c)['role_envelope_us'] for c in COMM_CANDIDATES}
    baseline = integer(config['comm_ctas'], 'baseline comm_ctas')
    if baseline not in scores:
        raise ValueError('The resolved baseline must be inside the declared candidate set')
    selected = baseline
    for c in COMM_CANDIDATES:
        if scores[c] < scores[selected]:
            selected = c
    a, tau = [model['coefficients'][key] for key in COEFFICIENT_KEYS]
    ratio = tau * 1e9 / (2 * COPY_ROWS * COPY_COLUMNS * COPY_SLOTS * a * case['k'])
    return dict(comm_ctas=selected, baseline_comm_ctas=baseline, switched=selected != baseline,
                baseline_score_us=scores[baseline], predicted_role_envelope_us=scores[selected],
                continuous_crossing_ctas=132 * ratio / (1 + ratio),
                candidate_scores_us=[dict(comm_ctas=c, role_envelope_us=scores[c])
                                     for c in COMM_CANDIDATES])


def calibration_rows(document):
    """Audit complete fixed-family role data; never read whole-F latency fields.

    The old profile's descriptive auto/fixed-tile labels are not authoritative:
    the explicit N256 environment and every rank's resolved tile are checked.
    One macro capture per point is diagnostic, NOT a 10+50 timing experiment.
    """
    try:
        return _calibration_rows(document)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f'Missing or malformed QKV-F role calibration: {error}') from error


def _calibration_rows(document):
    expect(document, dict(schema=CALIBRATION_SCHEMA, complete=True, profiling_only=True,
                          reported_duration_unit='us'), 'calibration')
    args, metadata = document['args'], document['metadata']
    expect(args, dict(cp=4, operators='qkv_forward', launch='graph', weight_mode='immediate',
                      oproj_comm_model=None, aggregate_only=True), 'args')
    expect(metadata, dict(kind='diagnostic_not_formal_performance', cc=[9, 0],
                          launch='graph', weight_mode='immediate'), 'metadata')
    expect(metadata['policy_environment'], dict(FUSE_QKV_GEMM_POLICY='m128n256'), 'environment')
    for key in ('native_device_name', 'cuda_visible_devices'):
        if not isinstance(metadata[key], str) or not metadata[key].strip():
            raise ValueError(f'Missing {key}')
    devices = metadata['cuda_visible_devices'].split(',')
    if len(devices) != 4 or len(set(devices)) != 4:
        raise ValueError('Exactly four distinct visible devices are required')
    if document['clock_origin'] != 'independent per rank; cross-rank absolute subtraction is invalid':
        raise ValueError('Independent GPU clock origins must be declared')
    sources = metadata['sources']
    if not isinstance(sources, dict) or not set(REQUIRED_SOURCES) <= sources.keys():
        raise ValueError('Critical benchmark/native source hashes are missing')
    for name, digest in sources.items():
        if not isinstance(name, str) or not name:
            raise ValueError('Malformed source name')
        sha256(digest, name)
    library_hash = sha256(metadata['library_sha256'], 'library')
    library_sources = [value for name, value in sources.items() if name.endswith('/' + Path(args['library']).name)]
    if library_sources != [library_hash]:
        raise ValueError('Library SHA differs from the source manifest')

    requests = args['comm_ctas']
    if (not isinstance(requests, list) or len(requests) < 2 or
            len(set(requests)) != len(requests)):
        raise ValueError('Calibration needs distinct declared communication requests')
    for c in requests:
        if integer(c, 'requested c') not in COMM_CANDIDATES:
            raise ValueError('Calibration c is outside the candidate domain')
    expect(metadata, dict(requested_comm_ctas=requests), 'metadata')
    ids, samples = document['selected_case_ids'], document['samples']
    if (not isinstance(ids, list) or not ids or len(set(ids)) != len(ids) or
            not all(isinstance(value, str) and value for value in ids)):
        raise ValueError('Selected case IDs must be nonempty and unique')
    expected = integer(document['expected_samples'], 'expected_samples')
    if (not isinstance(samples, list) or len(samples) != expected or
            expected != len(ids) * len(requests) or integer(args['max_cases'], 'max_cases') < expected):
        raise ValueError('Incomplete declared case × CTA matrix')
    sequences = [int(value) for value in args['seqs'].split(',')]
    if len(sequences) < 2 or len(set(sequences)) != len(sequences):
        raise ValueError('At least two distinct sequence lengths are needed to audit drift')
    rows, seen, physical, cases = [], set(), set(), {}
    for sample in samples:
        expect(sample, dict(operator='qkv_forward'), 'sample')
        case, c, ranks = sample['case'], sample['requested_comm_ctas'], sample['ranks']
        if case['id'] not in ids or c not in requests or case['global_seq'] not in sequences:
            raise ValueError('Undeclared case, sequence or CTA request')
        identity = (case['id'], c)
        if identity in seen:
            raise ValueError('Duplicate case/request')
        seen.add(identity)
        if case['id'] in cases and cases[case['id']] != case:
            raise ValueError('The same case ID changed geometry or metadata')
        cases[case['id']] = case
        expect(case, dict(registry='benchmarks/QKVproj+a2a/qkv_shape_bench.py',
                          visible_devices=metadata['cuda_visible_devices']), 'case')
        if case['id'] != f'gemm_a2a/{case["model"]}/s{case["global_seq"]}/cp4':
            raise ValueError('Case identity and physical metadata disagree')
        if not isinstance(ranks, list) or len(ranks) != 4:
            raise ValueError('Every point needs all four ranks')
        config = ranks[0]['config']
        feature = features(case, config, c)
        expect(config, dict(comm_ctas=c, requested_comm_ctas=c, world_size=4,
                            weight_block_size=32, payload='e4m3', scale='e8m0',
                            weight_axis='original_forward_K', gemm_accumulator='fp32',
                            alpha=1, beta=0, weight_mnk=None,
                            forward_or_data_mnk=[case[key] for key in ('m', 'n', 'k')],
                            weight_workspace_bytes=2 * case['n'] * case['k']), 'config')
        epoch = integer(config['epoch'], 'epoch')
        ready = integer(config['ready_elements'], 'ready_elements')
        if epoch >= 2**32 or ready < (case['m'] // 128) * (case['n'] // 256) * 32:
            raise ValueError('Invalid ready epoch or insufficient tile-ready storage')
        if integer(config['peer_arena_bytes'], 'peer_arena_bytes') < 2 * case['m'] * case['n'] + 4 * ready + 512:
            raise ValueError('Peer arena cannot contain declared data/ready/done regions')
        measured = []
        for rank, record in enumerate(ranks):
            expect(record, dict(rank=rank, observed_ctas=132, route_ctas=c, compute_ctas=132-c), 'rank')
            if record['config'] != config:
                raise ValueError('Ranks disagree on actual configuration')
            integer(record['origin_ns'], 'GPU origin')
            compute, route, overlap = [finite(record[key], key, positive=key != 'overlap_us')
                                       for key in ('compute_role_us', 'route_role_us', 'overlap_us')]
            if overlap > min(compute, route) + 1e-8:
                raise ValueError('Overlap exceeds a role envelope')
            measured.append((compute, route, route - overlap))
        correctness = sample['correctness']
        if set(correctness) != {'weight_dequant', 'output', 'local_gemm'}:
            raise ValueError('Missing forward/DQ correctness boundaries')
        for name, check in correctness.items():
            expect(check, dict(all_ranks_finite=True), 'correctness')
            absolute = finite(check['max_abs'], 'max_abs')
            relative = finite(check['relative_rmse'], 'relative_rmse')
            if relative > .005 or (name == 'weight_dequant' and (absolute != 0 or relative != 0)):
                raise ValueError('Exact DQ or BF16 correctness failed')
        geometry = (case['n'], case['k'])
        key = (case['m'], *geometry, c)
        if key in physical:
            raise ValueError('Duplicate physical measurement under different labels')
        physical.add(key)
        compute, route, uncovered = [max(values[index] for values in measured) for index in range(3)]
        rows.append(dict(case=copy.deepcopy(case), config=copy.deepcopy(config), geometry=geometry,
                         features=feature, compute_us=compute, route_envelope_us=route,
                         route_uncovered_us=uncovered))
    if seen != {(name, c) for name in ids for c in requests}:
        raise ValueError('Missing declared case/request')
    geometries = {row['geometry'] for row in rows}
    if set(args['model'].split(',')) != {row['case']['model'] for row in rows}:
        raise ValueError('Selected model labels differ from the declared filter')
    if len(geometries) < 3 or any({row['case']['global_seq'] for row in rows if row['geometry'] == geometry}
                                 != set(sequences) for geometry in geometries):
        raise ValueError('Need >=3 actual (N,K) geometries, each at every declared length')
    return rows


def fit_services(rows):
    """Equal-weight log residuals; measured compute censors latent route service."""
    import numpy as np
    from scipy.optimize import least_squares

    if not rows or not any(row['route_uncovered_us'] / row['route_envelope_us'] > EXPOSED_FRACTION for row in rows):
        raise ValueError('Route service is unidentifiable without communication-exposed points')
    a = math.exp(statistics.fmean(math.log(row['compute_us'] * row['features']['compute_ctas'] /
                                         row['features']['work_gflop']) for row in rows))
    def residual(parameters):
        return np.array([math.log(max(row['compute_us'], parameters[0] * row['features']['route_tasks'] /
                                     row['features']['route_slots']) / row['route_envelope_us']) for row in rows])
    result = least_squares(residual, [6.5], bounds=(0, np.inf), x_scale='jac',
                           ftol=1e-12, xtol=1e-12, gtol=1e-12)
    if not result.success:
        raise ValueError(f'Route fit did not converge: {result.message}')
    coefficients = dict(zip(COEFFICIENT_KEYS, (a, float(result.x[0]))))
    for key, value in coefficients.items():
        finite(value, key, positive=True)
    return coefficients


def relative_errors(rows, coefficients):
    errors = {key: [] for key in ('compute', 'conditional_route_envelope',
                                  'conditional_route_communication_exposed', 'joint_role_envelope')}
    for row in rows:
        prediction = service_times(row['features'], coefficients)
        ratio = max(row['compute_us'], prediction['route_service_us']) / row['route_envelope_us']
        errors['compute'].append(prediction['compute_us'] / row['compute_us'])
        errors['conditional_route_envelope'].append(ratio)
        errors['joint_role_envelope'].append(prediction['role_envelope_us'] /
                                             max(row['compute_us'], row['route_envelope_us']))
        if row['route_uncovered_us'] / row['route_envelope_us'] > EXPOSED_FRACTION:
            errors['conditional_route_communication_exposed'].append(ratio)
    return errors


def error_summary(errors):
    return {name: dict(points=len(ratios),
                       mape_percent=100 * statistics.fmean(abs(value - 1) for value in ratios) if ratios else None,
                       max_absolute_percentage_error=100 * max(abs(value - 1) for value in ratios) if ratios else None,
                       geometric_signed_bias_percent=100 * math.expm1(statistics.fmean(map(math.log, ratios))) if ratios else None)
            for name, ratios in errors.items()}


def fit_calibration(path):
    """Read one immutable calibration and return an explicitly loadable model."""
    path = Path(path)
    raw = path.read_bytes()
    document = json.loads(raw)
    rows = calibration_rows(document)
    coefficients = fit_services(rows)
    metadata = document['metadata']
    domain = dict(tile_families=[list(TILE)], head_dim=128, comm_ctas=list(COMM_CANDIDATES),
                  m_multiple=256, n_multiple=256, k_multiple=64)
    for key in ('m', 'n', 'k'):
        domain['min_' + key] = min(row['case'][key] for row in rows)
        domain['max_' + key] = max(row['case'][key] for row in rows)
    model = dict(world_size=4, sm_count=132, minimum_gain=0, coefficients=coefficients, domain=domain)
    folds, heldout = [], {key: [] for key in relative_errors([], coefficients)}
    for geometry in sorted({row['geometry'] for row in rows}):
        train = [row for row in rows if row['geometry'] != geometry]
        test = [row for row in rows if row['geometry'] == geometry]
        frozen = fit_services(train)
        errors = relative_errors(test, frozen)
        for key in heldout:
            heldout[key].extend(errors[key])
        case, config = test[0]['case'], test[0]['config']
        full_choice = select(case, config, model)['comm_ctas']
        fold_model = dict(model, coefficients=frozen)
        fold_choice = select(case, config, fold_model)['comm_ctas']
        folds.append(dict(held_out_nk=list(geometry), model_labels=sorted({row['case']['model'] for row in test}),
                          train_points=len(train), test_points=len(test), coefficients=frozen,
                          full_fit_comm_ctas=full_choice, heldout_fit_comm_ctas=fold_choice,
                          selection_changed=full_choice != fold_choice, errors=error_summary(errors)))
    lengths, cross_length = [], []
    for length in sorted({row['case']['global_seq'] for row in rows}):
        same = [row for row in rows if row['case']['global_seq'] == length]
        different = [row for row in rows if row['case']['global_seq'] != length]
        fitted = fit_services(same)
        by_geometry = [next(row for row in same if row['geometry'] == geometry)
                       for geometry in sorted({row['geometry'] for row in same})]
        lengths.append(dict(global_seq=length, points=len(same), coefficients=fitted,
                            errors=error_summary(relative_errors(same, fitted)),
                            selections=[dict(nk=list(row['geometry']), **select(row['case'], row['config'],
                                                                              dict(model, coefficients=fitted)))
                                        for row in by_geometry]))
        cross_length.append(dict(train_global_seq=length,
                                 test_global_sequences=sorted({row['case']['global_seq'] for row in different}),
                                 errors=error_summary(relative_errors(different, fitted))))
    import numpy
    import scipy
    model.update(
        calibration_path=str(path.resolve()), calibration_sha256=hashlib.sha256(raw).hexdigest(),
        source_hashes=copy.deepcopy(metadata['sources']), library_sha256=metadata['library_sha256'],
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        native_device_name=metadata['native_device_name'], cc=metadata['cc'],
        cuda_visible_devices=metadata['cuda_visible_devices'],
        policy_environment=copy.deepcopy(metadata['policy_environment']),
        calibration_metadata=copy.deepcopy({key: value for key, value in metadata.items() if key != 'sources'}),
        clock_origin=document['clock_origin'], gpu_clock_snapshot=metadata.get('gpu_clock_snapshot'),
        clock_scope='Independent per-rank durations only. Without a clock snapshot, stable clocks are not established.',
        coefficient_units=dict(compute_gflop_sm_us='microseconds_times_compute_SM_per_GFLOP',
                               route_slot_task_us='effective_microseconds_per_16KiB_task_per_slot_not_bare_copy'),
        fixed_family=dict(gemm_stages=4, raster='n', swizzle=1, route_layout='rank_major', batch=1,
                          activation='bf16', weight='offline_mxfp8_dequantized_to_bf16',
                          accumulator='fp32', head_dim=128, route_slots_per_cta=COPY_SLOTS),
        domain_interpretation='calibrated_physical_bounding_box_not_every_geometry_measured',
        formulas=dict(compute='a*(2*M*N*K/1e9)/(132-c)', route_service='tau*(M*N/8192)/(12*c)',
                      route_fit_target='max(measured_compute, predicted_route_service)',
                      selection_score='max(predicted_compute, predicted_route_service)',
                      exact_tie='retain_resolved_baseline', minimum_gain='zero; no gain gate'),
        fitting=dict(objective='sum_squared_log_prediction_over_measurement',
                     weighting='equal_weight_per_physical_point', compute_solver='closed_form_log_mean',
                     route_solver='scipy.optimize.least_squares', route_initial=[6.5], bounds=[0, 'infinity'],
                     x_scale='jac', ftol=1e-12, xtol=1e-12, gtol=1e-12,
                     numpy_version=numpy.__version__, scipy_version=scipy.__version__,
                     complete_forward_times_used=False, measured_winners_used=False, fixed_intercept_us=0),
        training=dict(points=len(rows), geometries=len(folds),
                      global_sequences=sorted({row['case']['global_seq'] for row in rows}),
                      measured_comm_ctas=sorted({row['features']['comm_ctas'] for row in rows}),
                      role_statistic='max_of_per_rank_role_envelopes_from_one_macro_capture_not_p50',
                      errors=error_summary(relative_errors(rows, coefficients))),
        leave_one_geometry_out=dict(grouping='actual_N_K_not_model_name', errors=error_summary(heldout), folds=folds),
        length_drift=dict(separate_length_fits=lengths, cross_length_validation=cross_length),
        route_identifiability=dict(exposed_threshold_fraction=EXPOSED_FRACTION,
                                   definition='max_rank(route-overlap)/max_rank(route)>threshold',
                                   by_comm_ctas=[dict(comm_ctas=c, measured_points=sum(row['features']['comm_ctas'] == c for row in rows),
                                                     communication_exposed_points=sum(row['features']['comm_ctas'] == c and
                                                         row['route_uncovered_us'] / row['route_envelope_us'] > EXPOSED_FRACTION
                                                         for row in rows))
                                                 for c in sorted({row['features']['comm_ctas'] for row in rows})]),
        limitations=[
            'Frozen means experiment parameters fixed, not a validated production default or milestone.',
            'Route includes producer readiness/queueing; neither its envelope nor this fitted latent service is bare NVLink.',
            'Low conditional route error can be dominated by compute-censored points; retain the exposed subset errors.',
            'High-c bandwidth is not identifiable when all its route observations are compute-censored.',
            'Absolute compute residuals, including the largest N/K geometry, and length drift are not clipped or discarded.',
            'DQ, finalize, complete F latency and measured winner choices are excluded; no launch/fill/wave-tail correction.',
            'Tile/stages and topology are fixed; another library/card group requires explicit validation.',
            'Predicted unmeasured even CTA points are interpolation hypotheses, not measured configurations.'
        ])
    validate_model(model)
    return dict(schema=SCHEMA, complete=True, frozen=True, model_status='experimental_role_calibration',
                production_policy_written=False, requires_explicit_loading=True, milestone_evidence=False,
                scope=SCOPE, operator='qkv_forward', models=[model])


def load_model(path):
    """Explicitly load the single model entry; no lookup or native setter."""
    artifact = json.loads(Path(path).read_text())
    expect(artifact, dict(schema=SCHEMA, complete=True, frozen=True,
                          production_policy_written=False, requires_explicit_loading=True,
                          milestone_evidence=False, scope=SCOPE, operator='qkv_forward'), 'artifact')
    if not isinstance(artifact['models'], list) or len(artifact['models']) != 1:
        raise ValueError('This artifact must contain exactly one CP4 model')
    return validate_model(artifact['models'][0])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.resolve() == args.calibration.resolve():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    result = fit_calibration(args.calibration)
    encoded = json.dumps(result, indent=2, allow_nan=False) + '\n'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        output.write(encoded)
    print(f'Wrote {args.output}: {result["models"][0]["training"]["points"]} diagnostic role points; not production-enabled')


if __name__ == '__main__':
    main()
