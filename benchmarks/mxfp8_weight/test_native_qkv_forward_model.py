#!/usr/bin/env python3
"""Explicit QKV-F native/Python policy parity; no operator or GPU kernel runs.

Only --library/--model/--output main initializes one CUDA metadata-query device.
CP4/CP8 are logical routes, not distributed execution. Import/test discovery is
CPU-only. Every candidate's actual native tile is checked before Python scoring.
"""

import argparse
import ctypes as ct
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import operator_bench as bench
import qkv_forward_service_model as service_model
import test_native_operator_model as common


__test__ = False
CONFIG_KEYS = common.CONFIG_KEYS
require, check, compact_config = common.require, common.check, common.compact_config


class NativeQueries(common.NativeQueries):
    """Reuse the existing metadata ABI; bind no launch entry points."""

    def __init__(self, library, argument_type):
        super().__init__(library, argument_type)
        for name, arguments in {
            'fuse_mxfp8_test_set_qkv_forward_comm_model': [ct.c_int32, ct.c_int32, ct.POINTER(ct.c_double)],
            'fuse_mxfp8_test_get_qkv_forward_comm_model': [ct.POINTER(ct.c_double)],
        }.items():
            function = getattr(library, name)
            function.argtypes, function.restype = arguments, ct.c_int

    def argument(self, case, op):
        argument = super().argument(case, op)
        argument.route_flags = case.get('route_flags', 0)
        return argument

    def get_model(self):
        values = (ct.c_double * 5)()
        check(self.library.fuse_mxfp8_test_get_qkv_forward_comm_model(values))
        return list(values)

    def model(self, model=None):
        setter = self.library.fuse_mxfp8_test_set_qkv_forward_comm_model
        if model is None:
            check(setter(0, 0, None))
            expected = [0.] * 5
        else:
            values = bench.qkv_forward_comm_model_values(model)
            check(setter(model['world_size'], model['sm_count'], (ct.c_double * 3)(*values)))
            expected = [model['world_size'], model['sm_count'], *values]
        require(self.get_model() == expected, 'QKV model set/get or disable roundtrip differs')


def domain_reason(case, config, model):
    """Known fallback guards only; do not swallow unexpected Python errors.

    The metadata bridge fixes contiguous BF16, AlongN, swizzle1, batch/L=1,
    local_heads=Q/CP, channel_count=1 and an ordinary full QKV route. Fields it
    cannot express are explicitly listed as unexercised in the report.
    """
    if case['cp'] != model['world_size'] or config['sm_count'] != model['sm_count']:
        return 'calibration_world_or_sm_mismatch'
    if not 0 <= case.get('rank', 0) < case['cp']:
        return 'invalid_route_rank'
    q, kv, d, cp = [case[key] for key in ('q_heads', 'kv_heads', 'head_dim', 'cp')]
    if q <= 0 or kv <= 0 or q % kv or q % cp or kv % cp or d != 128:
        return 'unsupported_head128_shards'
    if (case['n'] != (q + 2 * kv) * d or case['k'] != case['hidden'] or
            case['global_seq'] != case['m'] * cp or case['batch'] != 1):
        return 'inconsistent_physical_geometry'
    if case['layout'] != 'rank_major' or case.get('route_flags', 0) & 3:
        return 'unsupported_route_layout_or_flags'
    domain = model['domain']
    for key in ('m', 'n', 'k'):
        if (not domain['min_' + key] <= case[key] <= domain['max_' + key] or
                case[key] % domain[key + '_multiple']):
            return f'{key}_outside_calibration_domain'
    if [config[key] for key in service_model.TILE_KEYS] != list(service_model.TILE):
        return 'actual_native_tile_not_N256_C2'
    if config['comm_ctas'] not in domain['comm_ctas']:
        return 'baseline_or_candidate_comm_outside_declared_domain'
    return None


def prediction_config(config):
    return dict(config, raster='n', swizzle=1, gemm_input='bf16', route_layout='rank_major')


def python_selection(case, baseline, candidates, model):
    """Apply native tile eligibility before calling the frozen physical math."""
    reason = domain_reason(case, baseline, model)
    diagnostics, eligible = [], []
    for config in candidates:
        excluded = domain_reason(case, config, model)
        record = dict(comm_ctas=config['comm_ctas'], actual_config=compact_config(config),
                      domain_fallback=excluded)
        if excluded is None:
            record.update(service_model.predict(case, prediction_config(config), model))
            eligible.append(config)
        diagnostics.append(record)
    if reason is not None:
        return dict(selected_config=baseline, switched=False, domain_fallback=reason), diagnostics

    baseline_score = service_model.predict(case, prediction_config(baseline), model)['role_envelope_us']
    selected, best = baseline, baseline_score
    for config in eligible:
        score = service_model.predict(case, prediction_config(config), model)['role_envelope_us']
        if score < best:
            selected, best = config, score
    # When all candidates really are the fixed family, also exercise the
    # frozen selector directly. A tile-ineligible c is not a failed test.
    if len(eligible) == len(service_model.COMM_CANDIDATES):
        pure = service_model.select(case, prediction_config(baseline), model)
        require(pure['comm_ctas'] == selected['comm_ctas'], 'Frozen selector disagrees with filtered scoring')
    return dict(selected_config=selected, switched=selected['comm_ctas'] != baseline['comm_ctas'],
                domain_fallback=None, baseline_score_us=baseline_score,
                selected_score_us=best), diagnostics


def parity_case(native, case, model):
    argument = native.argument(case, 0)
    native.comm(0)
    native.model()
    baseline = native.config(argument)
    candidates = []
    for c in service_model.COMM_CANDIDATES:
        native.comm(c)
        config = native.config(argument)
        require(config['comm_ctas'] == c, f'Manual c={c} changed in disabled query')
        candidates.append(config)
    expected, diagnostics = python_selection(case, baseline, candidates, model)
    native.comm(0)
    native.model(model)
    actual = native.config(argument)
    require(actual == expected['selected_config'], f'{case["id"]}: native={actual}, Python={expected}')
    for config in candidates:
        native.comm(config['comm_ctas'])
        require(native.config(argument) == config, 'Manual CTA override lost precedence over QKV model')
    native.comm(0)
    native.model()
    require(native.config(argument) == baseline, 'Disabling QKV model failed to restore auto')
    return dict(case_id=case['id'], cp=case['cp'], rank=case.get('rank', 0),
                mnk=[case[key] for key in ('m', 'n', 'k')], layout=case['layout'],
                route_flags=case.get('route_flags', 0), origin=case.get('origin', 'published_registry'),
                baseline=compact_config(baseline), candidates=diagnostics, selected=compact_config(actual),
                python_selected=compact_config(expected['selected_config']), switched=expected['switched'],
                domain_fallback=expected['domain_fallback'],
                manual_priority_pass=True, disable_restores_auto=True, passed=True)


def unaffected_case(native, case, model):
    op = bench.OPERATORS.index(bench.operator_name(case))
    require(op in (1, 2, 3), 'Noninterference scope is the other three operators')
    argument = native.argument(case, op)
    native.comm(0)
    native.model()
    baseline = native.config(argument)
    native.model(model)
    actual = native.config(argument)
    require(actual == baseline, f'QKV-F model changed {bench.OPERATORS[op]}: {case["id"]}')
    native.model()
    require(native.config(argument) == baseline, 'Disable changed another operator')
    return dict(case_id=case['id'], operator=bench.OPERATORS[op], cp=case['cp'],
                native_config=compact_config(actual), enabled_equals_disabled=True, passed=True)


def setter_contract(native, model, reference):
    native.comm(0)
    native.model(model)
    initial = native.get_model()
    values = bench.qkv_forward_comm_model_values(model)
    setter = native.library.fuse_mxfp8_test_set_qkv_forward_comm_model
    invalid = [(f'world={world}', world, 132, values) for world in (-1, 1, 2, 3, 5, 8, 16)]
    invalid += [(f'sm={sm}', 4, sm, values) for sm in (0, 130, 134)]
    invalid.append(('null_values', 4, 132, None))
    for index in range(3):
        for label, value in (('negative', -1.), ('nan', math.nan), ('infinity', math.inf)):
            changed = list(values)
            changed[index] = value
            invalid.append((f'parameter{index}_{label}', 4, 132, changed))
    for index in (0, 1):
        changed = list(values)
        changed[index] = 0.
        invalid.append((f'parameter{index}_zero', 4, 132, changed))
    for margin in (1., 2.):
        invalid.append((f'minimum_gain={margin}', 4, 132, values[:2] + [margin]))
    rejected = []
    for label, world, sm, candidate in invalid:
        status = setter(world, sm, None if candidate is None else (ct.c_double * 3)(*candidate))
        require(status != 0, f'Invalid QKV model setter accepted {label}')
        require(native.get_model() == initial, f'Invalid QKV model setter mutated state: {label}')
        rejected.append(label)
    require(native.library.fuse_mxfp8_test_get_qkv_forward_comm_model(None) != 0, 'Null getter accepted')
    require(native.get_model() == initial, 'Invalid getter changed the model')
    argument = native.argument(reference, 0)
    output = (ct.c_int32 * len(CONFIG_KEYS))()
    require(native.library.fuse_mxfp8_test_config(None, output) != 0, 'Null config input accepted')
    require(native.library.fuse_mxfp8_test_config(ct.byref(argument), None) != 0, 'Null config output accepted')
    native.comm(12)
    before = native.config(argument)
    for c in (-2, -1, 1, 3, 132, 134):
        require(native.library.fuse_mxfp8_test_set_comm_ctas(c) != 0, f'Invalid c={c} accepted')
        require(native.config(argument) == before, 'Invalid CTA setter mutated override')
        require(native.get_model() == initial, 'CTA setter changed QKV model')
    native.comm(0)
    native.model()
    return dict(rejected_model_inputs=rejected, invalid_setters_preserve_state=True,
                null_getter_and_config_rejected=True, disable_roundtrip=True,
                margin_parity_scope='frozen_zero_margin_only', passed=True)


def extra_cases(reference):
    """Unseen physical combinations and legal metadata-only negative controls."""
    cases = []
    for label, changes, origin in (
            ('unseen_K6144_N5120', dict(m=65536, q_heads=24, hidden=6144), 'unseen_in_domain_geometry'),
            ('unseen_K8192_N10240', dict(m=65536, q_heads=64, hidden=8192), 'unseen_in_domain_geometry'),
            ('unseen_K7168_N7168', dict(m=98304, q_heads=40, hidden=7168), 'unseen_in_domain_geometry'),
            ('m_below_domain', dict(m=32512), 'metadata_only_domain_negative'),
            ('m_above_domain', dict(m=131328), 'metadata_only_domain_negative'),
            ('m_unaligned', dict(m=32769), 'metadata_only_domain_negative'),
            ('n_below_domain', dict(q_heads=8), 'metadata_only_domain_negative'),
            ('n_above_domain', dict(q_heads=136), 'metadata_only_domain_negative'),
            ('k_below_domain', dict(hidden=1984), 'metadata_only_domain_negative'),
            ('k_above_domain', dict(hidden=16448), 'metadata_only_domain_negative'),
            ('k_not_multiple64', dict(hidden=2080), 'metadata_only_domain_negative'),
            ('head_dim64', dict(q_heads=48, head_dim=64), 'metadata_only_domain_negative'),
            ('causal_route', dict(layout='causal_paired'), 'metadata_only_domain_negative'),
            ('peer_interleaved', dict(route_flags=1), 'metadata_only_domain_negative'),
            ('cyclic_peers', dict(route_flags=2), 'metadata_only_domain_negative'),
            ('rank_out_of_range', dict(rank=4), 'metadata_only_domain_negative')):
        case = dict(reference, **changes)
        case.update(id=f'synthetic/{label}', origin=origin, global_seq=case['m'] * case['cp'],
                    n=(case['q_heads'] + 2 * case['kv_heads']) * case['head_dim'], k=case['hidden'])
        cases.append(case)
    for rank in (1, 2, 3):
        cases.append(dict(reference, rank=rank, id=f'synthetic/valid_logical_rank{rank}',
                          origin='additional_logical_rank_not_multi_gpu_execution'))
    return cases


def source_hashes(args):
    sources = bench.source_hashes(SimpleNamespace(library=args.library, backends='fuse',
                                                oproj_comm_model=None, qkv_comm_model=args.model))
    for path in (Path(__file__), Path(service_model.__file__), Path(common.__file__),
                 Path(common.service_model.__file__)):
        sources[str(path.relative_to(bench.ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return sources


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, default=bench.ROOT / 'build-mxfp8/libfuse_mxfp8_torch_bridge.so')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    if args.output.resolve() in (args.model.resolve(), args.library.resolve()):
        raise ValueError('Output must be separate from model/library inputs')
    environment = {name: os.environ.get(name) for name in bench.POLICY_ENV_VARIABLES}
    if any(value is not None for value in environment.values()):
        raise ValueError(f'Parity requires the unchanged policy environment: {environment}')
    sources = source_hashes(args)
    model = service_model.load_model(args.model)
    loaded = bench.load_qkv_forward_comm_model(args.model, 4, 132)
    require(model['coefficients'] == loaded['coefficients'] and model['domain'] == loaded['domain'],
            'Benchmark loader differs from frozen Python model')
    require(sources.get(loaded['source']) == loaded['sha256'], 'Model changed since fingerprinting')
    # No CUDA or Torch imports occur before this explicit entry point.
    import torch
    from test_operators import Arguments

    if not 0 <= args.device < torch.cuda.device_count():
        raise ValueError('Metadata query device is not visible')
    torch.cuda.set_device(args.device)
    props = torch.cuda.get_device_properties(args.device)
    if (props.major, props.minor, props.multi_processor_count) != (9, 0, 132):
        raise ValueError('The frozen model needs one SM90/SM132 metadata-query device')
    native = NativeQueries(ct.CDLL(str(args.library.resolve())), Arguments)
    check(native.library.fuse_mxfp8_test_set_oproj_comm_model(0, 0, None))
    check(native.library.fuse_mxfp8_test_set_wgrad_policy(0))
    all_cases = bench.forward_matrix() + bench.backward_matrix()
    qkv = [case for case in all_cases if bench.operator_name(case) == 'qkv_forward']
    others = [case for case in all_cases if bench.operator_name(case) != 'qkv_forward']
    long_cp4 = [case for case in qkv if case['cp'] == 4 and case['global_seq'] >= 131072]
    require(len(long_cp4) == 24 and {case['global_seq'] for case in long_cp4} == {131072, 262144, 524288},
            'Published CP4 long registry is not the expected 8 × 3 matrix')
    reference = long_cp4[0]
    extras = extra_cases(reference)
    calibration_nk = {tuple(fold['held_out_nk']) for fold in model['leave_one_geometry_out']['folds']}
    require(all((case['n'], case['k']) not in calibration_nk for case in extras
                if case['origin'] == 'unseen_in_domain_geometry'), 'An unseen control was in calibration')
    report = dict(schema='mxfp8-native-qkv-forward-policy-parity-v1', complete=False, passed=False,
                  scope='host_metadata_policy_parity_not_operator_correctness_or_performance',
                  operator='qkv_forward', kernel_launches=0, tensors_allocated=0,
                  no_multi_gpu_execution_claim=True, logical_world_sizes=[4, 8],
                  device=dict(index=args.device, reported_name=props.name, cc=[props.major, props.minor],
                              sm_count=props.multi_processor_count), sources=sources,
                  model_sha256=loaded['sha256'], library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                  source_hash_scope='Current files and loaded binary; not a build provenance attestation',
                  model=loaded, policy_environment=environment, config_fields=list(CONFIG_KEYS),
                  torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  registry_coverage=dict(qkv_rows=len(qkv), cp4_long_rows=len(long_cp4),
                                         other_operator_rows=len(others), extra_controls=len(extras)),
                  unexercised_native_guards=['AlongM/heuristic raster', 'noncontiguous strides',
                                            'dtype/transposition', 'channel_count/packed_source_row/defer_v',
                                            'nonzero_margin_extension', 'nondefault_env_baseline_c_outside_4_to_24'],
                  contracts=[], cases=[], noninterference=[])
    active = None
    try:
        active = 'setter_contract'
        report['contracts'].append(setter_contract(native, loaded, reference))
        for case in qkv + extras:
            active = case['id']
            report['cases'].append(parity_case(native, case, model))
        for case in others:
            active = case['id']
            report['noninterference'].append(unaffected_case(native, case, loaded))
        require(source_hashes(args) == sources, 'Source/model/library changed during parity validation')
        report['coverage'] = dict(parity_cases=len(report['cases']),
                                  switched=sum(row['switched'] for row in report['cases']),
                                  domain_fallbacks=sum(row['domain_fallback'] is not None for row in report['cases']),
                                  ineligible_candidate_tiles=sum(candidate['domain_fallback'] == 'actual_native_tile_not_N256_C2'
                                      for row in report['cases'] for candidate in row['candidates']),
                                  other_operator_checks=len(report['noninterference']))
        report.update(complete=True, passed=True)
    except Exception as error:
        report['failure'] = dict(case=active, type=type(error).__name__, message=str(error))
        raise
    finally:
        try:
            native.comm(0)
            native.model()
        except Exception as error:
            report.update(complete=False, passed=False, cleanup_failure=str(error))
            raise
        finally:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('x') as output:
                json.dump(report, output, separators=(',', ':'), allow_nan=False)
                output.write('\n')
    print(f'QKV policy parity PASS: {len(report["cases"])} cases, '
          f'{len(report["noninterference"])} other-operator checks; {args.output}')


if __name__ == '__main__':
    main()
