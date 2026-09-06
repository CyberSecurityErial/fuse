#!/usr/bin/env python3
"""Fit an experimental OProj F service model from matched primitive timings.

The math/selection API uses only the standard library. Fitting lazily imports
SciPy and NumPy; it never imports CUDA or reads fused winners. A calibration is
explicitly supplied by the caller, not selected from a model name or CP alone.

Example:
  python operator_model.py --calibration cp4.json cp8.json --output model.json
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


READY_M = 128
STAGE_BYTES = 48 * 1024
ISSUERS = 4
INTERCEPT_US = 5.0
COMM_CANDIDATES = (4, 8, 12, 16, 24, 32)
RISK_THRESHOLD = 0.10
PRIMITIVES = ('compute_subgrid', 'copy')
TILE_KEYS = ('tile_m', 'tile_n', 'tile_k', 'cluster_m')


def ceil_div(numerator, denominator):
    return (numerator + denominator - 1) // denominator


def positive_int(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f'{name} must be a positive integer: {value}')
    return value


def features(case, config, comm_ctas=None):
    """Physical features for an already-resolved, homogeneous bulk configuration.

    An optional c must equal config.comm_ctas: changing c without resolving the
    native tile again would create an unmeasured configuration. The route here
    has seq_local=M and source_row_begin=0, as in the homogeneous OProj runner.
    """
    m, n, k, cp = [positive_int(case[key], key) for key in ('m', 'n', 'k', 'cp')]
    tm, tn, tk, cluster = [positive_int(config[key], key) for key in TILE_KEYS]
    sm = positive_int(config['sm_count'], 'sm_count')
    c = positive_int(config['comm_ctas'], 'comm_ctas')
    if comm_ctas is not None and comm_ctas != c:
        raise ValueError('Resolve the native tile/config for the requested c first')
    if c >= sm or c % cluster or (sm - c) % cluster:
        raise ValueError('Invalid communication/compute cluster split')
    if case.get('direction', 'a2a_gemm') != 'a2a_gemm' or case.get('batch', 1) != 1:
        raise ValueError('Only homogeneous OProj F with flattened M is modeled')
    if case.get('seq_local', m) != m or case.get('source_row_begin', 0) != 0:
        raise ValueError('Heterogeneous source ranges are not calibrated')
    if config.get('gemm_input', 'bf16') != 'bf16' or config.get('weight_block_size', 32) != 32:
        raise ValueError('This model is for MXFP8 block32 weights and BF16 GEMM')
    layout = case.get('layout', config.get('route_layout'))
    if layout not in ('contiguous', 'causal_paired'):
        raise ValueError('An explicit supported route layout is required')
    if config.get('route_layout', layout) != layout:
        raise ValueError('Case and native route layout disagree')
    if READY_M % tm or k % cp or (k // cp) % tk:
        raise ValueError('Ready-M and per-peer K must align with the GEMM tile')
    row_bytes = 2 * (k // cp)
    comm_rows = min(READY_M, STAGE_BYTES // row_bytes)
    if (comm_rows == 0 or row_bytes % 16 or m % READY_M or
            (layout == 'causal_paired' and m % (2 * READY_M))):
        raise ValueError('The native 48 KiB bulk path is not legal for this case')

    mt, nt = ceil_div(m, tm), ceil_div(n, tn)
    compute_ctas = sm - c
    workers = compute_ctas // cluster
    waves = ceil_div(ceil_div(mt, cluster) * nt, workers)
    kpad = ceil_div(k, tk) * tk
    ready_tiles = ceil_div(m, READY_M)
    # Same two ceil operations and clamp as a2a_lhs_comm_m_window().
    window = min(ready_tiles, max(1, ceil_div(ceil_div(compute_ctas, nt), READY_M // tm)))
    if 'comm_m_window' in config and config['comm_m_window'] != window:
        raise ValueError('Native m_window differs from the modeled frontier schedule')
    if config.get('copy_schedule', 'fused_frontier_window') != 'fused_frontier_window':
        raise ValueError('Copy must use the matched fused frontier-window schedule')
    chunks = ceil_div(READY_M, comm_rows)
    # Tasks include the local peer; remote payload does not.
    tasks = ready_tiles * cp * chunks
    remote_bytes = 2 * m * k * (cp - 1) // cp
    return dict(compute_ctas=compute_ctas, compute_clusters=workers, waves=waves,
                padded_tile_tflops=2 * tm * tn * kpad / 1e12,
                normalized_tile_area=tm * tn / 32768,
                remote_payload_bytes=remote_bytes, remote_mib=remote_bytes / 2**20,
                row_bytes=row_bytes, comm_rows=comm_rows, chunks_per_ready_tile=chunks,
                copy_tasks=tasks, task_waves=ceil_div(tasks, ISSUERS * c),
                ready_m_tiles=ready_tiles, m_window=window,
                window_batches=ceil_div(ready_tiles, window))


def service_times(feature, coefficients, launch_prior_us=INTERCEPT_US):
    a, e, b, t = [coefficients[key] for key in ('a', 'e', 'b', 't')]
    if any(not math.isfinite(value) or value < 0 for value in (a, e, b, t)):
        raise ValueError('Service coefficients must be finite and nonnegative')
    if not math.isfinite(launch_prior_us) or launch_prior_us < 0:
        raise ValueError('Launch prior must be finite and nonnegative, not an observed latency')
    compute = launch_prior_us + feature['waves'] * (
        a * feature['padded_tile_tflops'] + e * feature['normalized_tile_area'])
    copy = launch_prior_us + math.hypot(b * feature['remote_mib'], t * feature['task_waves'])
    score = max(compute, copy) + min(compute, copy) / feature['window_batches']
    if not all(math.isfinite(value) and value > 0 for value in (compute, copy, score)):
        raise ValueError('Nonfinite or nonpositive model prediction')
    return dict(compute_us=compute, copy_us=copy, candidate_score_us=score)


def predict(case, config, model, comm_ctas=None):
    """Predict using an explicitly chosen card-group calibration, never a lookup."""
    domain = model['domain']
    if case['cp'] != model['world_size'] or config['sm_count'] != model['sm_count']:
        raise ValueError('CP/SM budget differs from the supplied calibration')
    if [config[key] for key in TILE_KEYS] not in domain['tile_families']:
        raise ValueError('The resolved tile family was not calibrated')
    return service_times(features(case, config, comm_ctas), model['coefficients'], model['launch_prior_us'])


def guarded_select(case, baseline_config, candidate_configs, model,
                   min_improvement=None):
    """Require >=10% predicted latency reduction, a risk gate, not a fitted value.

    All configs must already be resolved by the native selector. Invalid inputs
    are errors, not silently omitted candidates. The baseline wins exact ties.
    """
    if min_improvement is None:
        min_improvement = model['minimum_gain']
    if not math.isfinite(min_improvement) or not 0 <= min_improvement < 1:
        raise ValueError('Risk threshold must be finite and in [0, 1)')
    baseline = predict(case, baseline_config, model)['candidate_score_us']
    best, best_score = baseline_config, baseline
    for config in candidate_configs:
        if config['comm_ctas'] not in COMM_CANDIDATES:
            raise ValueError('Candidate c is outside the declared experimental set')
        score = predict(case, config, model)['candidate_score_us']
        if score < best_score:
            best, best_score = config, score
    switch = best_score < baseline and best_score <= (1 - min_improvement) * baseline
    return dict(selected_config=dict(best if switch else baseline_config), switched=switch,
                baseline_score_us=baseline, best_candidate_score_us=best_score,
                predicted_reduction_fraction=1 - best_score / baseline,
                risk_threshold=min_improvement,
                threshold_definition='predicted_latency_reduction_not_fitted',
                reason='risk_threshold_met' if switch else 'retain_baseline')


def calibration_rows(document):
    """Reject incomplete, mismatched, short-search, or failed primitive data."""
    if document.get('schema') != 'mxfp8-oproj-independent-primitives-v1' or not document.get('complete'):
        raise ValueError('A complete matched-primitive calibration is required')
    args = document['args']
    if args['launch'] != 'graph' or args['warmup'] < 10 or args['iterations'] < 50:
        raise ValueError('Calibration requires graph timings with at least 10+50')
    if document['timing']['sample_statistic'] != 'sample_wise_max_across_ranks':
        raise ValueError('Calibration must use sample-wise rank maxima')
    samples = document['samples']
    if not samples or len(samples) != document['expected_samples']:
        raise ValueError('Calibration sample count is incomplete')
    rows, seen, domains = [], set(), set()
    for sample in samples:
        case, config = sample['case'], sample['config']
        feature = features(case, config)
        if config.get('compute_scheduler') != 'stock_persistent_same_tile_and_SM_budget_no_ready_waits':
            raise ValueError('Compute must reserve the matched SM budget without ready waits')
        domains.add((case['cp'], config['sm_count']))
        identity = (case['m'], case['n'], case['k'], config['comm_ctas'])
        if identity in seen:
            raise ValueError('Duplicate physical calibration point')
        seen.add(identity)
        if sample['work']['remote_payload_bytes_per_gpu'] != feature['remote_payload_bytes']:
            raise ValueError('Recorded remote payload disagrees with the physical features')
        times = []
        for name in PRIMITIVES:
            primitive = sample['primitives'][name]
            correct, timing = primitive['correctness'], primitive['timing']
            errors = (correct['max_abs'], correct['relative_rmse'])
            if correct['all_ranks_finite'] is not True or any(
                    not math.isfinite(value) or value < 0 for value in errors):
                raise ValueError('Independent primitive errors must be finite and nonnegative on every rank')
            # Match operator_perf_runtime.check_error for BF16 outputs. Copy
            # additionally uses exact=True; GEMM need not be bitwise identical.
            if correct['relative_rmse'] > 0.005 or (name == 'copy' and correct['max_abs'] != 0):
                raise ValueError('Independent primitive failed BF16 tolerance or exact copy correctness')
            values = timing['samples_us']
            ranks = timing['rank_samples_us']
            if len(values) != args['iterations'] or len(ranks) != case['cp']:
                raise ValueError('Missing timed samples or ranks')
            if any(len(rank) != len(values) for rank in ranks):
                raise ValueError('Rank sample lengths disagree')
            if any(not math.isfinite(value) or value <= 0 for rank in ranks for value in rank):
                raise ValueError('Primitive timings must be finite and positive')
            if values != [max(rank[index] for rank in ranks) for index in range(len(values))]:
                raise ValueError('Recorded samples are not the per-sample rank maxima')
            if not math.isclose(timing['p50_us'], statistics.median(values), rel_tol=1e-10):
                raise ValueError('Recorded p50 differs from the measured sample median')
            times.append(timing['p50_us'])
        rows.append(dict(geometry=(case['n'], case['k']), features=feature, times=times))
    if len(domains) != 1 or len({row['geometry'] for row in rows}) < 3:
        raise ValueError('Use one CP/SM domain and at least three distinct (N,K) geometries per file')
    return rows


def fit_services(rows):
    # Prediction and CLI help remain usable without SciPy, NumPy, or CUDA.
    import numpy as np
    from scipy.optimize import least_squares

    x = np.array([[row['features']['waves'] * row['features']['padded_tile_tflops'],
                   row['features']['waves'] * row['features']['normalized_tile_area'],
                   row['features']['remote_mib'], row['features']['task_waves']] for row in rows])
    y = np.array([row['times'] for row in rows])
    compute = least_squares(lambda p: np.log((INTERCEPT_US + x[:, :2] @ p) / y[:, 0]),
                            [180000., 4.], bounds=(0, np.inf), x_scale='jac')
    copy = least_squares(lambda p: np.log((INTERCEPT_US + np.hypot(x[:, 2] * p[0],
                                                                x[:, 3] * p[1])) / y[:, 1]),
                         [3.4, 6.], bounds=(0, np.inf), x_scale='jac')
    if not compute.success or not copy.success:
        raise ValueError(f'Service fit did not converge: {compute.message}; {copy.message}')
    return dict(zip(('a', 'e', 'b', 't'), map(float, [*compute.x, *copy.x])))


def relative_errors(rows, coefficients):
    errors = []
    for row in rows:
        predicted = service_times(row['features'], coefficients)
        errors.append([abs(predicted[key] / actual - 1) for key, actual in
                       zip(('compute_us', 'copy_us'), row['times'])])
    return errors


def error_summary(errors):
    return {name: dict(mape_percent=100 * statistics.mean(row[index] for row in errors),
                       max_absolute_percentage_error=100 * max(row[index] for row in errors))
            for index, name in enumerate(PRIMITIVES)}


def fit_calibration(path):
    path = Path(path)
    raw = path.read_bytes()
    document = json.loads(raw)
    rows = calibration_rows(document)
    coefficients = fit_services(rows)
    folds, heldout_errors = [], []
    for geometry in sorted({row['geometry'] for row in rows}):
        train = [row for row in rows if row['geometry'] != geometry]
        test = [row for row in rows if row['geometry'] == geometry]
        frozen = fit_services(train)
        errors = relative_errors(test, frozen)
        heldout_errors.extend(errors)
        folds.append(dict(held_out_nk=list(geometry), train_points=len(train), test_points=len(test),
                          coefficients=frozen, errors=error_summary(errors)))
    configs = [sample['config'] for sample in document['samples']]
    return dict(calibration_path=str(path.resolve()), calibration_sha256=hashlib.sha256(raw).hexdigest(),
                source_hashes=document['sources'], devices=document['devices'],
                cuda_visible_devices=document['cuda_visible_devices'],
                policy_environment=document['policy_environment'],
                gpu_clock_snapshot=document.get('gpu_clock_snapshot'),
                clock_scope='Measured card group and run state only; CP is not a universal hardware coefficient. '
                            'Without a recorded clock snapshot, stable clocks are not established. '
                            'Runtime product-name strings are retained verbatim, not used for selection.',
                world_size=document['samples'][0]['case']['cp'], sm_count=configs[0]['sm_count'],
                launch_prior_us=INTERCEPT_US, minimum_gain=RISK_THRESHOLD,
                domain=dict(tile_families=[list(tile) for tile in sorted({tuple(g[key] for key in TILE_KEYS)
                                                                         for g in configs})]),
                training=dict(points=len(rows), geometries=len(folds),
                              global_sequences=sorted({s['case']['global_seq'] for s in document['samples']}),
                              comm_ctas=sorted({g['comm_ctas'] for g in configs}),
                              warmup=document['args']['warmup'], iterations=document['args']['iterations'],
                              errors=error_summary(relative_errors(rows, coefficients))),
                coefficients=coefficients,
                leave_one_geometry_out=dict(grouping='actual_N_K_not_model_name',
                                            errors=error_summary(heldout_errors), folds=folds))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--overwrite', action='store_true', help='Explicitly replace the one output JSON')
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    inputs = [path.resolve() for path in args.calibration]
    if len(set(inputs)) != len(inputs) or args.output.resolve() in inputs:
        raise ValueError('Calibration inputs must be unique and separate from the output')
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f'Refusing to replace {args.output}; use --overwrite explicitly')
    result = dict(schema='mxfp8-oproj-service-model-v1', complete=True, frozen=True,
                  production_policy_written=False, milestone_evidence=False,
                  scope='OProj_F_experimental_selection_score_not_2F2B_total',
                  tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  fitting=dict(objective='independent_compute_and_copy_sum_squared_log_prediction_over_measurement',
                               weighting='equal_weight_per_physical_measurement_row', nonnegative=['a', 'e', 'b', 't'],
                               solver='scipy.optimize.least_squares; bounds=(0,inf); x_scale=jac; default tolerances',
                               initial_compute=[180000., 4.], initial_copy=[3.4, 6.],
                               fixed_intercept_us=INTERCEPT_US, intercept_is_measured_launch=False,
                               fused_winners_used=False),
                  feature_constants=dict(ready_m=READY_M, bulk_stage_bytes=STAGE_BYTES, issuers=ISSUERS),
                  coefficient_units=dict(a='us_per_padded_tile_TFLOP_per_compute_wave',
                                         e='us_per_normalized_tile_area_per_compute_wave',
                                         b='effective_us_per_remote_MiB',
                                         t='effective_us_per_copy_task_wave'),
                  candidate_comm_ctas=list(COMM_CANDIDATES), risk_threshold=RISK_THRESHOLD,
                  risk_threshold_definition='Require >=10% predicted latency reduction; conservative, not fitted',
                  fused_score='max(Tc,Tr)+min(Tc,Tr)/ceil(ready_m_tiles/m_window)',
                  limitations=['Explicit calibration selection does not verify current clocks/topology.',
                               'Only native-resolved calibrated tile families and the legal homogeneous bulk path.',
                               'Window fill/drain is a candidate hypothesis, not an exact wait/contention decomposition.',
                               'DQ is omitted as candidate-invariant; no full-operator or milestone claim.',
                               'Short-search winners and fused timings are never fitting inputs.'],
                  original_experiment_context='The user-reported H200 CP8 experiment included devices fixed near '
                                              '1500 MHz; CP4 used a different card group. This history is not a '
                                              'clock measurement or an assumption about future calibration inputs.',
                  models=[fit_calibration(path) for path in args.calibration])
    # Record the numerical environment without making it a dependency of math imports.
    import numpy
    import scipy
    result['fitting']['numpy_version'] = numpy.__version__
    result['fitting']['scipy_version'] = scipy.__version__
    encoded = json.dumps(result, indent=2, allow_nan=False) + '\n'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w' if args.overwrite else 'x') as output:
        output.write(encoded)
    print(f'Wrote {args.output}: {len(result["models"])} explicit card-group calibrations')


if __name__ == '__main__':
    main()
