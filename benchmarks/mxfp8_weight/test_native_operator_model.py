#!/usr/bin/env python3
"""Explicit native/Python policy parity, not operator correctness or performance.

Run this script deliberately with --library --model --output. Importing it or
discovering CPU tests never initializes CUDA. The script uses one CUDA device
only for native metadata queries: CP4/8 are logical route sizes, not launched
process groups. There are no tensors, kernel launches, IPC arenas, or timings.
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
import operator_model as service_model


__test__ = False  # Explicit script, including under pytest discovery.
CONFIG_KEYS = ('comm_ctas', 'tile_m', 'tile_n', 'tile_k', 'cluster_m', 'sm_count',
               'policy_enum', 'weight_tile_m', 'weight_tile_n', 'weight_tile_k')


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def check(status):
    if status != 0:
        raise RuntimeError(f'Native metadata API returned CUDA status {status}')


class NativeQueries:
    """Bind only host setters/getters and metadata queries, never launch APIs."""

    def __init__(self, library, argument_type):
        self.library, self.argument_type = library, argument_type
        signatures = {
            'fuse_mxfp8_test_set_oproj_comm_model': [ct.c_int32, ct.c_int32, ct.POINTER(ct.c_double)],
            'fuse_mxfp8_test_get_oproj_comm_model': [ct.POINTER(ct.c_double)],
            'fuse_mxfp8_test_set_comm_ctas': [ct.c_int32],
            'fuse_mxfp8_test_set_wgrad_policy': [ct.c_int32],
            'fuse_mxfp8_test_config': [ct.POINTER(argument_type), ct.POINTER(ct.c_int32)],
        }
        for name, signature in signatures.items():
            function = getattr(library, name)
            function.argtypes, function.restype = signature, ct.c_int

    def argument(self, case, op):
        argument = self.argument_type()
        argument.geometry[:] = [op, case.get('rank', 0), case['cp'], case['m'], case['hidden'],
                                case['q_heads'], case['kv_heads'], case['head_dim'], 1,
                                int(case['layout'] == 'causal_paired'), 1, 0, 0]
        argument.alpha, argument.beta = 1, 0
        # All pointer fields remain zero; metadata queries must not dereference them.
        return argument

    def config(self, argument):
        output = (ct.c_int32 * len(CONFIG_KEYS))()
        check(self.library.fuse_mxfp8_test_config(ct.byref(argument), output))
        return dict(zip(CONFIG_KEYS, output))

    def comm(self, requested):
        check(self.library.fuse_mxfp8_test_set_comm_ctas(requested))

    def get_model(self):
        values = (ct.c_double * 8)()
        check(self.library.fuse_mxfp8_test_get_oproj_comm_model(values))
        return list(values)

    def model(self, model=None):
        if model is None:
            check(self.library.fuse_mxfp8_test_set_oproj_comm_model(0, 0, None))
            expected = [0., 0., 0., 0., 0., 0., 5., .10]
        else:
            values = bench.oproj_comm_model_values(model)
            check(self.library.fuse_mxfp8_test_set_oproj_comm_model(
                model['world_size'], model['sm_count'], (ct.c_double * 6)(*values)))
            expected = [model['world_size'], model['sm_count'], *values]
        require(self.get_model() == expected, 'Native model set/get or disable roundtrip mismatch')


def compact_config(config):
    return [config[key] for key in CONFIG_KEYS]


def domain_reason(case, config, model):
    """Known native fallback conditions, not a catch-all for Python exceptions."""
    if case['cp'] != model['world_size'] or config['sm_count'] != model['sm_count']:
        return 'calibration_world_or_sm_mismatch'
    if not 0 <= case.get('rank', 0) < case['cp'] or case['q_heads'] % case['cp']:
        return 'invalid_route_head_shards_or_rank'
    if case['n'] % 8 or case['k'] % 8:
        return 'unsupported_bf16_problem_alignment'
    if case['k'] % case['cp'] or (case['k'] // case['cp']) % 64:
        return 'per_peer_K_not_tile64_aligned'
    row_bytes = 2 * (case['k'] // case['cp'])
    if (row_bytes <= 0 or row_bytes % 16 or row_bytes > service_model.STAGE_BYTES or
            case['m'] % 128 or (case['layout'] == 'causal_paired' and case['m'] % 256)):
        return 'non_bulk_route'
    if [config[key] for key in service_model.TILE_KEYS] not in model['domain']['tile_families']:
        return 'uncalibrated_native_tile_family'
    return None


def python_selection(case, baseline, candidates, model):
    """Native score(ineligible)=infinity; ineligible baseline keeps legacy auto."""
    reason = domain_reason(case, baseline, model)
    diagnostics, eligible = [], []
    for config in candidates:
        excluded = domain_reason(case, config, model)
        item = dict(requested_comm_ctas=config['comm_ctas'], config=compact_config(config),
                    domain_fallback=excluded)
        if excluded is None:
            item.update(service_model.predict(case, config, model))
            eligible.append(config)
        diagnostics.append(item)
    if reason is not None:
        return dict(selected_config=baseline, switched=False, domain_fallback=reason), diagnostics
    result = service_model.guarded_select(case, baseline, eligible, model)
    result['domain_fallback'] = None
    return result, diagnostics


def parity_case(native, case, model):
    argument = native.argument(case, 1)
    native.model()
    native.comm(0)
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
    require(actual == expected['selected_config'],
            f'{case["id"]}/{case["layout"]}: native={actual}, Python={expected}')
    for config in candidates:
        native.comm(config['comm_ctas'])
        require(native.config(argument) == config, 'Manual communication override lost precedence over model')
    native.comm(0)
    native.model()
    require(native.config(argument) == baseline, 'Disabling model did not restore the original auto config')
    return dict(case_id=case['id'], cp=case['cp'], mnk=[case['m'], case['n'], case['k']],
                layout=case['layout'], origin=case.get('origin', 'published_registry'),
                baseline=compact_config(baseline), candidates=diagnostics,
                selected=compact_config(actual), python_selected=compact_config(expected['selected_config']),
                switched=expected['switched'], domain_fallback=expected['domain_fallback'],
                predicted_reduction_fraction=expected.get('predicted_reduction_fraction'),
                manual_priority_pass=True, disable_restores_auto=True, passed=True)


def unaffected_case(native, case, model):
    op = bench.OPERATORS.index(bench.operator_name(case))
    require(op in (0, 2, 3), 'Only the other three operators belong in the noninterference test')
    argument = native.argument(case, op)
    native.comm(0)
    native.model()
    baseline = native.config(argument)
    native.model(model)
    actual = native.config(argument)
    require(actual == baseline, f'OProj model changed {bench.OPERATORS[op]}: {case["id"]}')
    native.model()
    require(native.config(argument) == baseline, 'Disable changed another operator')
    return dict(case_id=case['id'], operator=bench.OPERATORS[op], cp=case['cp'],
                native_config=compact_config(actual), enabled_equals_disabled=True, passed=True)


def setter_contract(native, model, reference_case):
    native.comm(0)
    native.model(model)
    initial = native.get_model()
    values = bench.oproj_comm_model_values(model)
    setter = native.library.fuse_mxfp8_test_set_oproj_comm_model
    rejected = []
    invalid = [(f'world={world}', world, 132, values) for world in (-1, 1, 2, 3, 5, 16)]
    invalid += [(f'sm={sm}', model['world_size'], sm, values) for sm in (0, 130, 134)]
    invalid.append(('null_values', model['world_size'], 132, None))
    for index in range(6):
        for label, value in (('negative', -1.), ('nan', math.nan), ('infinity', math.inf)):
            changed = list(values)
            changed[index] = value
            invalid.append((f'parameter{index}_{label}', model['world_size'], 132, changed))
    for index in (0, 2, 3):
        changed = list(values)
        changed[index] = 0.
        invalid.append((f'parameter{index}_zero', model['world_size'], 132, changed))
    for value in (1., 2.):
        invalid.append((f'minimum_gain={value}', model['world_size'], 132, values[:5] + [value]))
    for label, world, sm, candidate in invalid:
        status = setter(world, sm, None if candidate is None else (ct.c_double * 6)(*candidate))
        require(status != 0, f'Invalid model setter was accepted: {label}')
        require(native.get_model() == initial, f'Invalid model setter mutated state: {label}')
        rejected.append(label)
    require(native.library.fuse_mxfp8_test_get_oproj_comm_model(None) != 0,
            'Null model getter was accepted')
    require(native.get_model() == initial, 'Invalid getter changed the model')
    argument = native.argument(reference_case, 1)
    output = (ct.c_int32 * 10)()
    require(native.library.fuse_mxfp8_test_config(None, output) != 0, 'Null config input was accepted')
    require(native.library.fuse_mxfp8_test_config(ct.byref(argument), None) != 0,
            'Null config output was accepted')
    native.comm(16)
    selected = native.config(argument)
    invalid_comm = (-2, -1, 1, 3, 132, 134)
    for c in invalid_comm:
        require(native.library.fuse_mxfp8_test_set_comm_ctas(c) != 0, f'Invalid c={c} was accepted')
        require(native.config(argument) == selected, 'Invalid CTA setter changed the active override')
        require(native.get_model() == initial, 'CTA setter changed the active model')
    native.comm(0)
    native.model()
    return dict(cp=model['world_size'], rejected_model_inputs=rejected,
                rejected_comm_inputs=list(invalid_comm), invalid_setters_preserve_state=True,
                null_getter_and_config_rejected=True, disable_roundtrip=True, passed=True)


def fallback_cases(reference):
    """Small metadata-only negative cases; none are run as operators."""
    cases = []
    for label, changes in (
            ('partial_ready_tile', dict(m=129)),
            ('partial_causal_half', dict(m=128)),
            ('peer_k_unaligned', dict(q_heads=16, head_dim=132)),
            ('bulk_row_exceeds_stage', dict(q_heads=16, head_dim=32768)),
            ('rank_out_of_range', dict(rank=reference['cp']))):
        case = dict(reference, **changes)
        case.update(id=f'synthetic/{label}/cp{case["cp"]}', origin='metadata_only_domain_negative',
                    global_seq=case['m'] * case['cp'], k=case['q_heads'] * case['head_dim'])
        cases.append(case)
    return cases


def source_hashes(args):
    sources = bench.source_hashes(SimpleNamespace(library=args.library, backends='fuse',
                                                oproj_comm_model=args.model))
    for path in (Path(__file__), Path(service_model.__file__)):
        sources[str(path.relative_to(bench.ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return sources


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, default=bench.ROOT / 'build-mxfp8/libfuse_mxfp8_torch_bridge.so')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', type=int, default=0, help='One metadata-query CUDA device; no operator launches')
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    if args.output.exists():
        raise FileExistsError(f'Refusing to replace {args.output}; choose a new explicit output')
    if args.output.resolve() in (args.library.resolve(), args.model.resolve()):
        raise ValueError('Output must be separate from model/library inputs')
    environment = {name: os.environ.get(name) for name in bench.POLICY_ENV_VARIABLES}
    if any(value is not None for value in environment.values()):
        raise ValueError(f'Use an unmodified baseline policy environment for parity: {environment}')
    sources = source_hashes(args)
    # All CUDA imports/initialization live behind the explicit script entry.
    import torch
    from test_operators import Arguments

    if not 0 <= args.device < torch.cuda.device_count():
        raise ValueError('The requested metadata-query device is not visible')
    torch.cuda.set_device(args.device)
    props = torch.cuda.get_device_properties(args.device)
    if (props.major, props.minor, props.multi_processor_count) != (9, 0, 132):
        raise ValueError('This frozen-model parity test requires an SM90/SM132 query device')
    models = {}
    for world in (4, 8):
        loaded = bench.load_oproj_comm_model(args.model, world, props.multi_processor_count)
        require(sources.get(loaded['source']) == loaded['sha256'], 'Model changed since source fingerprinting')
        models[world] = dict(loaded, domain=dict(tile_families=loaded['tile_families']))
    native = NativeQueries(ct.CDLL(str(args.library.resolve())), Arguments)
    check(native.library.fuse_mxfp8_test_set_wgrad_policy(0))
    forwards = bench.forward_matrix()
    oproj = [case for case in forwards if bench.operator_name(case) == 'oproj_forward']
    other = [case for case in forwards if bench.operator_name(case) != 'oproj_forward'] + bench.backward_matrix()
    report = dict(schema='mxfp8-native-oproj-policy-parity-v1', complete=False, passed=False,
                  scope='correctness_not_performance', kernel_launches=0, tensors_allocated=0,
                  device=dict(index=args.device, reported_name=props.name, cc=[props.major, props.minor],
                              sm_count=props.multi_processor_count),
                  logical_world_sizes=[4, 8], no_multi_gpu_execution_claim=True,
                  sources=sources, model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                  library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                  source_hash_scope='Current source files and loaded binary; not a build provenance attestation',
                  models=[models[world] for world in (4, 8)], config_fields=list(CONFIG_KEYS),
                  torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  policy_environment=environment, contracts=[], cases=[], noninterference=[],
                  registry_coverage=dict(oproj_published_rows=len(oproj), other_operator_rows=len(other),
                                         sequences=sorted({case['global_seq'] for case in oproj}),
                                         published_layouts=sorted({case['layout'] for case in oproj}),
                                         extra_layout='contiguous_from_same_geometry_not_new_benchmark_rows'))
    active_case = None
    try:
        for world in (4, 8):
            reference = next(case for case in oproj if case['cp'] == world and case['m'] >= 16384)
            active_case = dict(id='native_setter_contract', cp=world)
            report['contracts'].append(setter_contract(native, models[world], reference))
            selected = [case for case in oproj if case['cp'] == world]
            variants = [dict(case, layout=layout, origin=('published_registry' if layout == case['layout']
                                                         else 'additional_api_layout'))
                        for case in selected for layout in ('causal_paired', 'contiguous')]
            for case in variants + fallback_cases(reference):
                active_case = dict(id=case['id'], cp=world, layout=case['layout'])
                report['cases'].append(parity_case(native, case, models[world]))
            # A valid model for the other CP must leave this route at legacy auto.
            foreign = models[8 if world == 4 else 4]
            mismatch = dict(reference, id=f'synthetic/wrong_model_world/cp{world}',
                            origin='metadata_only_domain_negative')
            active_case = dict(id=mismatch['id'], cp=world, layout=mismatch['layout'])
            report['cases'].append(parity_case(native, mismatch, foreign))
            for case in other:
                if case['cp'] == world:
                    active_case = dict(id=case['id'], cp=world, layout=case['layout'])
                    report['noninterference'].append(unaffected_case(native, case, models[world]))
        require(source_hashes(args) == sources, 'A source, model, or binary changed during policy validation')
        report['coverage'] = {str(world): dict(
            parity_cases=sum(row['cp'] == world for row in report['cases']),
            switched=sum(row['cp'] == world and row['switched'] for row in report['cases']),
            domain_fallbacks=sum(row['cp'] == world and row['domain_fallback'] is not None
                                 for row in report['cases']),
            other_operator_checks=sum(row['cp'] == world for row in report['noninterference']))
            for world in (4, 8)}
        report.update(complete=True, passed=True)
    except Exception as error:
        report['failure'] = dict(case=active_case, type=type(error).__name__, message=str(error))
        raise
    finally:
        try:
            native.comm(0)
            native.model()
        except Exception as error:
            report.update(complete=False, passed=False,
                          cleanup_failure=dict(type=type(error).__name__, message=str(error)))
            raise
        finally:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('x') as output:
                json.dump(report, output, separators=(',', ':'), allow_nan=False)
                output.write('\n')
    print(f'Policy parity PASS: {len(report["cases"])} OProj/layout/domain checks, '
          f'{len(report["noninterference"])} other-operator checks; {args.output}')


if __name__ == '__main__':
    main()
