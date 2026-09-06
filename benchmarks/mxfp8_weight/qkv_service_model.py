"""Validate QKV dgrad primitive measurements and derive physical work features.

Validation/features use the standard library; diagnostic fitting lazily imports
NumPy/SciPy. No dispatch, CUDA imports, or file output. The supported native
family is BF16 M128N256K64/C2/stage4 with 128-wide heads.
Copy task depths and compute cluster waves are integer work counts, not times.
Preloaded-ready minus bare GEMM is adapter_overhead, not communication wait.
The two copy launch reservations remain distinct, even with identical work.
"""

import copy
import math
import statistics


SCHEMA = 'mxfp8-qkv-backward-independent-primitives-v1'
LEGACY_PRIMITIVES = {'compute_bare_subgrid': 0, 'copy': 1,
                     'compute_ready_preloaded_subgrid': 2, 'compute_bare_fullgrid': 3,
                     'compute_ready_preloaded_fullgrid': 4}
PRIMITIVES = dict(LEGACY_PRIMITIVES, copy_fused_reservation=5)
COPY_PRIMITIVES = ('copy', 'copy_fused_reservation')
TILE_KEYS = ('tile_m', 'tile_n', 'tile_k', 'cluster_m')
TILE = (128, 256, 64, 2)
READY_M, READY_STRIDE, COPY_SLOTS = 128, 32, 12
COPY_STAGE_BYTES = 64 * 128 * 2
COPY_SMEM_BYTES = COPY_SLOTS * (COPY_STAGE_BYTES + 8)
# sizeof(QkvBackwardN256Binding::Kernel::SharedStorage), not a fitted value.
FUSED_SMEM_BYTES = 214016
SCOPE = 'no_DQ_no_W_no_finalize_not_B_total'
FIT_INTERCEPT_US = 5.


def ceil_div(numerator, denominator):
    return (numerator + denominator - 1) // denominator


def integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}: {value}')
    return value


def finite(value, name, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')
    return value


def expect(record, expected, name):
    for key, value in expected.items():
        actual = record.get(key)
        if (key not in record or actual != value or
                (type(value) in (int, bool) and type(actual) is not type(value))):
            raise ValueError(f'{name}.{key} disagrees with the primitive contract')


def features(case, config, primitive, resources):
    """Use an actual native configuration; never resolve a tile from a label.

    Route work includes local destinations, while remote payload excludes them.
    Each of twelve warp slots reuses one 16 KiB stage for two ordered chunks per
    32 KiB task. Neither chunk issue nor task-depth waves imply synchronized
    service time. Compute waves use the queried, problem-truncated C2 grid.
    Native register counts are retained, never inferred from the tile.
    """
    if primitive not in PRIMITIVES:
        raise ValueError(f'Unsupported QKV primitive: {primitive}')
    m, n, cp, q, kv, d = [integer(case.get(key), key) for key in
                          ('m', 'hidden', 'cp', 'q_heads', 'kv_heads', 'head_dim')]
    heads, k = q + 2 * kv, (q + 2 * kv) * d
    layout = case.get('layout')
    if (cp not in (4, 8) or d != 128 or m % READY_M or n % 8 or
            q % cp or kv % cp or q % kv or
            layout not in ('contiguous', 'causal_paired')):
        raise ValueError('Geometry is outside the supported homogeneous head128 TMA domain')
    expect(case, dict(operator='qkv', batch=1, global_seq=m * cp, b_mnk=[m, n, k]), 'case')
    if tuple(integer(config.get(key), key) for key in TILE_KEYS) != TILE:
        raise ValueError('Only the resolved M128N256K64/C2 family is supported')
    sm, c = [integer(config.get(key), key) for key in ('sm_count', 'comm_ctas')]
    if sm % 2 or c % 2 or c >= sm:
        raise ValueError('Communication/compute split must preserve positive C2 budgets')
    requested = integer(config.get('requested_comm_ctas'), 'requested_comm_ctas', 0)
    if requested not in (0, c):
        raise ValueError('Requested communication CTAs differ from the native config')
    expect(config, dict(weight_block_size=32, payload='e4m3', scale='e8m0',
                        gemm_input='bf16', gemm_accumulator='fp32',
                        weight_axis='original_forward_K', route_layout=layout,
                        forward_or_data_mnk=[m, n, k], raster='n', swizzle=1,
                        copy_schedule='native_qkv_peer_head_route',
                        compute_scheduler='matched_tile_subgrid_or_fullgrid_bare_or_preloaded_ready',
                        primitive_scope=SCOPE), 'config')
    epoch = integer(config.get('epoch'), 'epoch')
    if epoch >= 2**32:
        raise ValueError('Ready epoch must fit uint32')
    ready_tiles = m // READY_M
    ready_bytes = ready_tiles * heads * READY_STRIDE * 4
    if ready_bytes > 4 * integer(config.get('ready_elements'), 'ready_elements'):
        raise ValueError('Actual ready prefix exceeds the recorded arena capacity')

    mt, nt = ceil_div(m, TILE[0]), ceil_div(n, TILE[1])
    cluster_tiles = ceil_div(mt, TILE[3]) * nt
    tasks = ready_tiles * heads
    feature = dict(m=m, n=n, k=k, cp=cp, sm_count=sm, comm_ctas=c,
                   packed_heads=heads, local_packed_heads=heads // cp,
                   q_heads=q, kv_heads=kv, head_dim=d, ready_m_tiles=ready_tiles,
                   ready_prefix_bytes=ready_bytes, flops_per_gpu=2 * m * n * k,
                   staging_write_bytes_per_gpu=2 * m * k,
                   remote_payload_bytes_per_gpu=2 * m * k * (cp - 1) // cp,
                   compulsory_gemm_bytes_per_gpu=2 * (m * k + n * k + m * n),
                   copy_tasks=tasks, copy_stage_bytes=COPY_STAGE_BYTES,
                   copy_task_bytes=READY_M * d * 2, copy_chunks_per_task=2,
                   copy_slots_per_cta=COPY_SLOTS,
                   copy_slot_task_depth=ceil_div(tasks, COPY_SLOTS * c),
                   copy_slot_chunk_depth=2 * ceil_div(tasks, COPY_SLOTS * c),
                   copy_tma_loads=2 * tasks, copy_tma_stores=2 * tasks,
                   copy_min_tasks_per_cta=tasks // c,
                   copy_max_tasks_per_cta=ceil_div(tasks, c),
                   output_m_tiles=mt, output_n_tiles=nt, cluster_work_tiles=cluster_tiles,
                   heads_per_output_tile=heads, k_tiles_per_head=d // TILE[2],
                   padded_flops_per_output_tile=2 * TILE[0] * TILE[1] * k,
                   padded_flops_per_head_per_output_tile=2 * TILE[0] * TILE[1] * d,
                   padded_flops_per_k_tile=2 * TILE[0] * TILE[1] * TILE[2])
    copy = primitive in COPY_PRIMITIVES
    fused_copy = primitive == 'copy_fused_reservation'
    expect(resources, dict(selected_gemm_stages=4, ready_block_m=READY_M,
                           ready_flag_stride=READY_STRIDE, packed_heads=heads,
                           k_tiles_per_head=d // TILE[2], threads_per_cta=384,
                           copy_fields_apply=copy, copy_use_tma=int(copy),
                           copy_slots=COPY_SLOTS if copy else 0,
                           primitive_launch_cluster_m=1 if primitive == 'copy' else TILE[3]), 'resources')
    grid = integer(resources.get('primitive_grid_x'), 'primitive_grid_x')
    smem = integer(resources.get('primitive_dynamic_smem_bytes'), 'primitive_dynamic_smem_bytes')
    registers = integer(resources.get('primitive_registers_per_thread'), 'primitive_registers_per_thread')
    if registers > 255:
        raise ValueError('Native registers per thread exceed the SM90 limit')
    if copy:
        if grid != c or smem != (FUSED_SMEM_BYTES if fused_copy else COPY_SMEM_BYTES):
            raise ValueError('Copy grid/SMEM differs from the actual twelve-slot primitive')
        feature['copy_launch_variant'] = ('fused_cluster2_reservation' if fused_copy else
                                          'standalone_cluster1')
    else:
        budget = sm if primitive.endswith('fullgrid') else sm - c
        # For this C2/swizzle1 SM90 family, the scheduler caps the grid at
        # min(SM budget, cluster-padded work CTAs), then the wrapper flattens it.
        if grid != min(budget, cluster_tiles * TILE[3]):
            raise ValueError('Compute grid differs from the matched native C2 budget/work cap')
        if smem != FUSED_SMEM_BYTES:
            raise ValueError('Compute must preserve the actual fused shared-memory reservation')
        feature.update(compute_budget_ctas=budget, compute_grid_ctas=grid,
                       compute_grid_clusters=grid // TILE[3],
                       compute_cluster_waves=ceil_div(cluster_tiles, grid // TILE[3]))
    feature['native_resources'] = dict(resources)
    return feature


def checked_timing(primitive, iterations, cp, flops):
    correct, timing = primitive['correctness'], primitive['timing']
    expect(correct, dict(all_ranks_finite=True), 'correctness')
    for key in ('max_abs', 'relative_rmse'):
        finite(correct[key], key)
    if correct['relative_rmse'] > 0.005 or (flops == 0 and correct['max_abs'] != 0):
        raise ValueError('Primitive failed BF16 tolerance or exact copy correctness')
    values, ranks = timing['samples_us'], timing['rank_samples_us']
    if (not isinstance(values, list) or not isinstance(ranks, list) or
            len(values) != iterations or len(ranks) != cp or
            any(not isinstance(rank, list) or len(rank) != iterations for rank in ranks)):
        raise ValueError('Incomplete timing samples or rank vectors')
    for value in values + [value for rank in ranks for value in rank]:
        finite(value, 'sample_us', positive=True)
    if values != [max(rank[index] for rank in ranks) for index in range(iterations)]:
        raise ValueError('Samples are not sample-wise rank maxima')
    ordered, index = sorted(values), (iterations - 1) * .95
    lower = math.floor(index)
    p95 = ordered[lower] + (ordered[math.ceil(index)] - ordered[lower]) * (index - lower)
    expected = dict(p50_us=statistics.median(values), p95_us=p95,
                    mean_us=statistics.fmean(values),
                    tflops_per_gpu=flops / statistics.median(values) / 1e6)
    expect(timing, dict(flops_per_gpu=flops), 'timing')
    for key, value in expected.items():
        if not math.isclose(finite(timing[key], key), value, rel_tol=1e-10, abs_tol=1e-9):
            raise ValueError(f'Recorded {key} disagrees with the raw timing vectors')
    return expected


def calibration_rows(document):
    """Reject partial/search/failed/mismatched data; return compact measured rows.

    An explicitly requested primitive subset is allowed, but every requested
    primitive and CTA point must be present. Keep the input document for source
    hashes, device identities, clocks and raw samples: rows are not a new report
    or standalone provenance artifact. No fused winner data is consumed.
    """
    try:
        return _calibration_rows(document)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f'Missing or malformed QKV calibration data: {error}') from error


def _calibration_rows(document):
    expect(document, dict(schema=SCHEMA, complete=True, operator='qkv_backward',
                          phase='data', weight_mode='deferred', production_policy_written=False,
                          milestone_evidence=False,
                          scope='independent_services_' + SCOPE,
                          readiness='int32_epoch_preloaded_after_reset_into_actual_strided_flag_prefix',
                          preparation='bare_and_ready_compute_share_identical_preparation_each_sample',
                          staging='peer_arena_data_bound_by_Arguments_not_unbound_tensors_staging'), 'report')
    args = document['args']
    cp = integer(args['cp'], 'cp')
    iterations = integer(args['iterations'], 'iterations', 50)
    integer(args['warmup'], 'warmup', 10)
    expect(args, dict(operator='qkv_backward', launch='graph'), 'args')
    selected, requests = args['primitives'], args['comm_ctas']
    declared = document['primitive_ids']
    if (declared not in (LEGACY_PRIMITIVES, PRIMITIVES) or
            any(type(value) is not int for value in declared.values())):
        raise ValueError('Unsupported native primitive ID mapping')
    if (not isinstance(selected, list) or not selected or len(set(selected)) != len(selected) or
            not set(selected) <= declared.keys() or not isinstance(requests, list) or not requests or
            len(set(requests)) != len(requests)):
        raise ValueError('Requested primitive/CTA lists must be nonempty, unique and supported')
    for request in requests:
        if integer(request, 'comm_ctas request', 0) % 2:
            raise ValueError('Communication requests must be zero or positive even counts')
    expect(document['timing'], dict(sample_statistic='sample_wise_max_across_ranks',
                                    included='one_existing_BF16_compute_or_route_primitive'), 'timing')
    excluded = {'DQ', 'W', 'finalize', 'warmup', 'other_primitive',
                'allocation_IPC_capture_references', 'staging_preload_control_reset_CPU_barrier',
                'integer_ready_prefix_restore'}
    if not excluded <= set(document['timing']['excluded']):
        raise ValueError('Timing scope does not exclude preparation and other operator phases')
    sources = document['sources']
    if not isinstance(sources, dict) or not sources or any(
            not isinstance(key, str) or not isinstance(value, str) or len(value) != 64 or
            any(char not in '0123456789abcdef' for char in value) for key, value in sources.items()):
        raise ValueError('Source/library SHA256 provenance is missing or malformed')
    if not isinstance(document['gpu_clock_snapshot'], str) or not document['gpu_clock_snapshot'].strip():
        raise ValueError('The card-group clock snapshot is required')
    devices = document['devices']
    if len(devices) != cp:
        raise ValueError('Device metadata is missing ranks')
    for rank, device in enumerate(devices):
        expect(device, dict(rank=rank, cc=[9, 0]), 'device')
        integer(device['sm_count'], 'device sm_count')
    sms = {device['sm_count'] for device in devices}
    if cp not in (4, 8) or len(sms) != 1:
        raise ValueError('Use one homogeneous CP4/8 SM90 card group per calibration')
    sm = next(iter(sms))
    samples = document['samples']
    count = integer(document['expected_samples'], 'expected_samples')
    if not isinstance(samples, list) or len(samples) != count:
        raise ValueError('Incomplete calibration samples')
    expect(document, dict(expected_primitive_measurements=count * len(selected)), 'report')
    rows, seen, geometries, native = [], set(), {}, {}
    compute_smem = None
    for sample in samples:
        case, config, primitives = sample['case'], sample['config'], sample['primitives']
        expect(case, dict(cp=cp), 'case')
        expect(config, dict(sm_count=sm), 'config')
        if set(primitives) != set(selected):
            raise ValueError('A requested primitive is missing or an undeclared primitive is present')
        request = config['requested_comm_ctas']
        if request not in requests:
            raise ValueError('Sample uses an undeclared communication request')
        physical = tuple(case[key] for key in ('m', 'hidden', 'q_heads', 'kv_heads', 'head_dim', 'layout'))
        identity = physical + (request,)
        if identity in seen:
            raise ValueError('Duplicate physical geometry/request point')
        seen.add(identity)
        geometries.setdefault(physical, set()).add(request)
        measured = {}
        for name in selected:
            primitive = primitives[name]
            feature = features(case, config, name, primitive['resources'])
            work = {key: feature[key] for key in ('m', 'n', 'k', 'cp', 'flops_per_gpu',
                    'remote_payload_bytes_per_gpu', 'staging_write_bytes_per_gpu',
                    'compulsory_gemm_bytes_per_gpu')}
            expect(sample['work'], work, 'work')
            copy = name in COPY_PRIMITIVES
            expect(primitive, dict(primitive_id=PRIMITIVES[name], scope=SCOPE,
                                   included='copy_only' if copy else 'BF16_GEMM_only',
                                   ready_prepublication_bytes=0 if copy else feature['ready_prefix_bytes'],
                                   ready_epoch=None if copy else config['epoch'],
                                   ready_prepublication=('not_used_by_copy' if copy else
                                                        'same_int32_epoch_prefix_for_bare_and_preloaded')),
                   'primitive')
            resources = primitive['resources']
            variant = name.replace('subgrid', 'grid').replace('fullgrid', 'grid')
            signature = {key: value for key, value in resources.items()
                         if key not in ('primitive_grid_x', 'packed_heads')}
            if variant in native and signature != native[variant]:
                raise ValueError('The same native primitive has inconsistent resource metadata')
            native[variant] = signature
            if not copy:
                if compute_smem is not None and compute_smem != resources['primitive_dynamic_smem_bytes']:
                    raise ValueError('Bare/ready compute changed the fused shared-memory reservation')
                compute_smem = resources['primitive_dynamic_smem_bytes']
            measured[name] = dict(features=feature, timing=checked_timing(
                primitive, iterations, cp, 0 if copy else feature['flops_per_gpu']))
        overhead = {}
        for grid in ('subgrid', 'fullgrid'):
            bare, ready = f'compute_bare_{grid}', f'compute_ready_preloaded_{grid}'
            if bare in measured and ready in measured:
                # Independent runs, not paired samples and not a measured wait.
                overhead[grid] = measured[ready]['timing']['p50_us'] - measured[bare]['timing']['p50_us']
        rows.append(dict(case=dict(case), config=dict(config), primitives=measured,
                         adapter_overhead_us=overhead))
    if any(points != set(requests) for points in geometries.values()):
        raise ValueError('A geometry is missing a requested communication candidate')
    return rows


def fit_calibration(document, primitive):
    """Fit ONE checked primitive and leave each actual (N,K) geometry out.

    Each measured point has equal weight; the target is its median of
    sample-wise rank maxima, not the individual rank samples. Copies with
    different launch reservations and bare/preloaded or sub/full-grid compute
    are never pooled. The fixed 5 us intercept is a prior, not measured launch
    latency. Results are diagnostic only, even when held-out error is small.

    Accept an in-memory raw document, not a path. Callers retain the original
    input/file hash when archiving: source hashes below identify the measured
    implementation, not a serialization of this document.
    """
    if primitive not in PRIMITIVES:
        raise ValueError(f'Unsupported QKV primitive: {primitive}')
    rows = calibration_rows(document)
    if any(primitive not in row['primitives'] for row in rows):
        raise ValueError(f'Requested primitive was not measured: {primitive}')
    geometries = [(row['primitives'][primitive]['features']['n'],
                   row['primitives'][primitive]['features']['k']) for row in rows]
    groups = sorted(set(geometries))
    if len(groups) < 3:
        raise ValueError('Fitting requires at least three distinct actual (N,K) geometries')

    # Ordinary imports/raw validation remain usable without fitting packages.
    import numpy as np
    import scipy
    from scipy.optimize import least_squares

    is_copy = primitive in COPY_PRIMITIVES
    names, initial = (('b', 't'), [3.4, 30.]) if is_copy else (('a', 'e'), [180000., 4.])
    inputs, targets = [], []
    for row in rows:
        measured = row['primitives'][primitive]
        feature = measured['features']
        if is_copy:
            inputs.append([feature['remote_payload_bytes_per_gpu'] / 2**20,
                           feature['copy_slot_task_depth']])
        else:
            waves = feature['compute_cluster_waves']
            inputs.append([waves * feature['padded_flops_per_output_tile'] / 1e12,
                           waves * row['config']['tile_m'] * row['config']['tile_n'] / 32768])
        targets.append(measured['timing']['p50_us'])
    x, y = np.array(inputs, dtype=float), np.array(targets, dtype=float)

    def predict(parameters, values):
        service = (np.hypot(values[:, 0] * parameters[0], values[:, 1] * parameters[1])
                   if is_copy else values @ parameters)
        return FIT_INTERCEPT_US + service

    def solve(mask):
        result = least_squares(lambda p: np.log(predict(p, x[mask]) / y[mask]),
                               initial, bounds=(0, np.inf), x_scale='jac')
        if not result.success:
            raise ValueError(f'Primitive fit did not converge: {result.message}')
        coefficients = dict(zip(names, map(float, result.x)))
        for name, value in coefficients.items():
            finite(value, f'fitted coefficient {name}')
        return result.x, dict(coefficients=coefficients,
                             solver_result=dict(status=int(result.status), nfev=int(result.nfev),
                                                cost=float(result.cost), optimality=float(result.optimality)))

    def errors(parameters, mask):
        values = abs(predict(parameters, x[mask]) / y[mask] - 1)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite primitive prediction or relative error')
        return values.tolist()

    def summary(values):
        return dict(mape_percent=100 * statistics.mean(values),
                    max_absolute_percentage_error=100 * max(values))

    all_points = np.ones(len(rows), dtype=bool)
    parameters, fitted = solve(all_points)
    folds, heldout_errors = [], []
    for geometry in groups:
        heldout = np.array([item == geometry for item in geometries], dtype=bool)
        frozen, fold = solve(~heldout)
        values = errors(frozen, heldout)
        heldout_errors.extend(values)
        folds.append(dict(held_out_nk=list(geometry), train_points=int((~heldout).sum()),
                          test_points=int(heldout.sum()), errors=summary(values), **fold))
    feature = rows[0]['primitives'][primitive]['features']
    return dict(schema='mxfp8-qkv-primitive-service-fit-v1', model_status='diagnostic_only',
                production_policy_written=False, milestone_evidence=False,
                primitive=primitive, primitive_id=PRIMITIVES[primitive], scope=SCOPE,
                source_hashes=copy.deepcopy(document['sources']), devices=copy.deepcopy(document['devices']),
                gpu_clock_snapshot=document['gpu_clock_snapshot'],
                cuda_visible_devices=document.get('cuda_visible_devices'),
                policy_environment=copy.deepcopy(document.get('policy_environment')),
                clock_scope='Measured card group and run state only; CP is not a universal hardware coefficient.',
                domain=dict(cp=feature['cp'], sm_count=feature['sm_count'], tile=list(TILE),
                            native_resources={key: value for key, value in feature['native_resources'].items()
                                              if key not in ('primitive_grid_x', 'packed_heads')},
                            copy_launch_variant=feature.get('copy_launch_variant'),
                            actual_nk=[list(group) for group in groups],
                            m=sorted({row['case']['m'] for row in rows}),
                            layouts=sorted({row['case']['layout'] for row in rows})),
                fitting=dict(objective='sum_squared_natural_log_prediction_over_measured_p50',
                             weighting='equal_weight_per_physical_measurement_row',
                             target='p50_us_of_sample_wise_rank_maxima',
                             formula=('5 + hypot(b * remote_MiB, t * copy_slot_task_depth)' if is_copy else
                                      '5 + compute_cluster_waves * (a * tile_TFLOP + e * normalized_tile_area)'),
                             feature_units=(['remote_bytes / 2**20', 'ceil(copy_tasks / (12 * comm_ctas))'] if is_copy else
                                            ['waves * padded_flops_per_output_tile / 1e12',
                                             'waves * tile_m * tile_n / 32768']),
                             coefficient_units=(dict(b='us_per_remote_MiB', t='us_per_copy_slot_task_depth') if is_copy else
                                                dict(a='us_per_CTA_output_tile_TFLOP', e='us_per_normalized_tile_area')),
                             fixed_intercept_us=FIT_INTERCEPT_US, intercept_is_measured_launch=False,
                             fused_winners_used=False, solver='scipy.optimize.least_squares',
                             method='trf', jac='2-point', loss='linear', x_scale='jac',
                             initial=initial, bounds=[0, 'inf'], ftol=1e-8, xtol=1e-8, gtol=1e-8,
                             max_nfev=None, numpy_version=np.__version__, scipy_version=scipy.__version__),
                training=dict(points=len(rows), geometries=len(groups),
                              global_sequences=sorted({row['case']['global_seq'] for row in rows}),
                              comm_ctas=sorted({row['config']['comm_ctas'] for row in rows}),
                              warmup=document['args']['warmup'], iterations=document['args']['iterations'],
                              errors=summary(errors(parameters, all_points))),
                leave_one_geometry_out=dict(grouping='actual_N_K_not_model_name',
                                            weighting='equal_weight_per_held_out_measurement_row',
                                            errors=summary(heldout_errors), folds=folds),
                limitations=['No candidate scoring, dispatch, or production acceptance is implied.',
                             'Copy service includes TMA/ready publication, not bare NVLink bandwidth.',
                             'Matched reservations do not reproduce concurrent compute contention.',
                             'Preloaded-ready versus bare timing is adapter overhead, not measured producer waiting.'],
                **fitted)
