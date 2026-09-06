#!/usr/bin/env python3
"""Benchmark production SM90 MXFP8-weight 2F2B, one torchrun process per GPU.

The matrix is exactly the published BF16/MXFP8 baseline matrix. T is a single
flattened token count; there is no training-batch knob. The baseline's legacy
``global_seq`` IDs are retained for exact result joins. This runner does not
repeat the completed TEUB/cuBLASLt search.
"""

import argparse
import ctypes as ct
from datetime import timedelta
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

from matrix import ROOT, full_matrix as forward_matrix
from backward_matrix import full_matrix as backward_matrix


OPERATORS = ('qkv_forward', 'oproj_forward', 'qkv_backward', 'oproj_backward')
POLICY_ENV_VARIABLES = ('FUSE_QKV_COMM_POLICY', 'FUSE_QKV_GEMM_POLICY',
                        'FUSE_A2A_LHS_COMM_POLICY', 'FUSE_NVLINK_BIDIR_GBPS')


def initialize_runtime():
    """Shared by formal timing and diagnostics; list/CPU tests stay CUDA-free."""
    global torch, dist, rt
    import torch
    import torch.distributed as dist
    import operator_perf_runtime as rt


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--list-only', action='store_true')
    parser.add_argument('--cp', type=int, choices=(4, 8))
    parser.add_argument('--operators', default=','.join(OPERATORS))
    parser.add_argument('--model', default='production_qwen_dense',
                        help='comma-separated existing registry names')
    parser.add_argument('--seqs', default='1024')
    parser.add_argument('--case-ids', type=Path, help='JSON list of exact published case IDs')
    parser.add_argument('--launches', default='eager,graph')
    parser.add_argument('--weight-modes', default='immediate,deferred')
    parser.add_argument('--backends', default='fuse,pure_cublas')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--oproj-comm-model', type=Path,
                        help='explicit frozen OProj service model; omitted keeps the existing baseline')
    parser.add_argument('--qkv-comm-model', type=Path,
                        help='explicit frozen QKV forward service model; omitted keeps the existing baseline')
    parser.add_argument('--library', type=Path,
                        default=ROOT / 'build-mxfp8/libfuse_mxfp8_torch_bridge.so')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'results/mxfp8_weight/operators/pilot.json')
    return parser.parse_args(argv)


def operator_name(case):
    if 'direction' in case:
        return 'qkv_forward' if case['direction'] == 'gemm_a2a' else 'oproj_forward'
    return case['operator'] + '_backward'


def select_cases(args, world=None):
    operators = args.operators.split(',')
    if len(set(operators)) != len(operators) or not set(operators) <= set(OPERATORS):
        raise ValueError('--operators must be unique canonical 2F2B names')
    rows = forward_matrix() + backward_matrix()
    ids = None
    if args.case_ids:
        ids = json.loads(args.case_ids.read_text())
        if not isinstance(ids, list) or len(ids) != len(set(ids)):
            raise ValueError('--case-ids must contain a unique JSON list')
        unknown = set(ids) - {row['id'] for row in rows}
        if unknown:
            raise ValueError(f'Unknown matrix IDs: {sorted(unknown)}')
    cp = world or args.cp
    if world and args.cp and args.cp != world:
        raise ValueError('--cp must equal torchrun WORLD_SIZE')
    models = set(args.model.split(','))
    if ids is None and not args.full:
        unknown_models = models - {row['model'] for row in rows if operator_name(row) in operators}
        if unknown_models:
            raise ValueError(f'Unknown selected operator model names: {sorted(unknown_models)}')
    selected = [row for row in rows if operator_name(row) in operators
                and (cp is None or row['cp'] == cp)
                and (row['id'] in ids if ids is not None else
                     args.full or (row['model'] in models and
                                   row['global_seq'] in set(map(int, args.seqs.split(',')))))]
    if not selected:
        raise ValueError('No selected cases')
    if any(row['batch'] != 1 or row['m'] * row['cp'] != row['global_seq'] for row in selected):
        raise ValueError('Registry no longer has the required flattened-token semantics')
    return selected


def source_hashes(args):
    paths = [Path(__file__), Path(__file__).with_name('operator_perf_runtime.py'),
             Path(__file__).with_name('operator_bridge.cu'),
             Path(__file__).with_name('operator_profile.cuh'),
             Path(__file__).with_name('operator_reference.cuh'),
             Path(__file__).with_name('test_operators.py'),
             Path(__file__).with_name('matrix.py'), Path(__file__).with_name('backward_matrix.py'),
             ROOT / 'csrc/operators/ulysses_sm90.cu',
             *sorted((ROOT / 'csrc/operators/ulysses_sm90').glob('**/*.cuh')),
             *sorted((ROOT / 'include/fuse').glob('**/*.h')), args.library]
    paths.extend(sorted((ROOT / 'include/fuse').glob('**/*.cuh')))
    if getattr(args, 'oproj_comm_model', None) is not None:
        paths.append(Path(args.oproj_comm_model).resolve())
    if getattr(args, 'qkv_comm_model', None) is not None:
        paths.append(Path(args.qkv_comm_model).resolve())
    for filename in ('classic_gemm.py', 'classic_gemm.cu'):
        path = Path(__file__).with_name(filename)
        if path.exists() and 'pure_cublas' in args.backends.split(','):
            paths.append(path)
    if 'pure_cublas' in args.backends.split(','):
        paths.append(ROOT / 'build-mxfp8-bench/libmxfp8_classic_gemm.so')
    config_helper = Path(__file__).with_name('operator_config.py')
    if config_helper.exists():
        paths.append(config_helper)
    for row in forward_matrix() + backward_matrix():
        paths.append(ROOT / row['registry'])
    return {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path):
            hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(set(paths))}


def load_oproj_comm_model(path, world, sm_count):
    """Select one explicit card-group calibration, never a registry model name."""
    if world not in (4, 8) or sm_count != 132:
        raise ValueError('OProj service models require CP4/8 and SM132')
    path = Path(path).resolve()
    raw = path.read_bytes()
    document = json.loads(raw)
    if (not isinstance(document, dict) or
            document.get('schema') != 'mxfp8-oproj-service-model-v1' or
            document.get('complete') is not True or document.get('frozen') is not True):
        raise ValueError('A complete, frozen mxfp8-oproj-service-model-v1 report is required')
    models = document.get('models')
    if not isinstance(models, list) or any(not isinstance(row, dict) for row in models):
        raise ValueError('OProj service models must be a list of calibration objects')
    matches = [row for row in models if row.get('world_size') == world]
    if len(matches) != 1:
        raise ValueError('Choose a file with exactly one model for this CP/card group')
    model = matches[0]
    if (type(model.get('world_size')) is not int or type(model.get('sm_count')) is not int or
            model['sm_count'] != sm_count):
        raise ValueError('OProj service model CP/SM metadata does not match this worker')
    domain = model.get('domain')
    families = domain.get('tile_families') if isinstance(domain, dict) else None
    expected_families = [[128, 256, 64, 2], [128, 320, 64, 2]]
    if (not isinstance(families, list) or
            any(not isinstance(tile, list) or any(type(value) is not int for value in tile)
                for tile in families) or sorted(families) != expected_families):
        raise ValueError('OProj service model must cover exactly the N256/N320, K64, C2 families')
    coefficients = model.get('coefficients')
    if not isinstance(coefficients, dict):
        raise ValueError('OProj service model is missing coefficients')
    values = [coefficients.get(key) for key in ('a', 'e', 'b', 't')]
    values += [model.get('launch_prior_us'), model.get('minimum_gain')]
    try:
        valid_values = all(type(value) in (int, float) and math.isfinite(value) and value >= 0
                           for value in values)
    except OverflowError:
        valid_values = False
    if not valid_values:
        raise ValueError('OProj service coefficients/prior/margin must be finite and nonnegative')
    if any(values[index] == 0 for index in (0, 2, 3)) or values[5] >= 1:
        raise ValueError('OProj service a/b/t must be positive and minimum_gain must be less than one')
    return dict(schema=document['schema'],
                source=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                sha256=hashlib.sha256(raw).hexdigest(), world_size=world, sm_count=sm_count,
                coefficients=dict(zip(('a', 'e', 'b', 't'), values[:4])),
                launch_prior_us=values[4], minimum_gain=values[5], tile_families=expected_families,
                auto_reference='explicit_calibrated_model_or_domain_fallback_not_legacy_auto',
                scope='oproj_forward_only_manual_comm_ctas_take_precedence')


def oproj_comm_model_values(model):
    return [model['coefficients'][key] for key in ('a', 'e', 'b', 't')] + [
        model['launch_prior_us'], model['minimum_gain']]


def read_oproj_comm_model(library, runtime):
    """Verify installed host policy outside timing; old libraries need no new API."""
    model = vars(library).get('_fuse_oproj_comm_model')
    if model is None:
        return None
    query = library.fuse_mxfp8_test_get_oproj_comm_model
    query.argtypes = [ct.POINTER(ct.c_double)]
    query.restype = ct.c_int
    values = (ct.c_double * 8)()
    runtime.check(query(values))
    if list(values) != [model['world_size'], model['sm_count'], *oproj_comm_model_values(model)]:
        raise RuntimeError('Native OProj service model changed or differs from the explicit file')
    return dict(model)


def configure_oproj_comm_model(path, world, sm_count, library, runtime, sources, workers=None):
    """Install once per worker before creating cases/capturing any Graph."""
    if path is None:
        return None
    model = load_oproj_comm_model(path, world, sm_count)
    if sources.get(model['source']) != model['sha256']:
        raise RuntimeError('OProj service model changed since the run source fingerprint was taken')
    setter = library.fuse_mxfp8_test_set_oproj_comm_model
    setter.argtypes = [ct.c_int32, ct.c_int32, ct.POINTER(ct.c_double)]
    setter.restype = ct.c_int
    values = (ct.c_double * 6)(*oproj_comm_model_values(model))
    runtime.check(setter(world, sm_count, values))
    library._fuse_oproj_comm_model = model
    read_oproj_comm_model(library, runtime)
    if workers is not None:
        installed = [None] * world
        workers.all_gather_object(installed, model)
        if any(other != model for other in installed):
            raise RuntimeError('Workers did not install the same explicit OProj service model')
    return model


def load_qkv_forward_comm_model(path, world, sm_count):
    """Explicit CP4 long-sequence calibration; no model-name or winner lookup."""
    if world != 4 or sm_count != 132:
        raise ValueError('QKV forward service calibration requires CP4 and SM132')
    path = Path(path).resolve()
    raw = path.read_bytes()
    document = json.loads(raw)
    if (not isinstance(document, dict) or
            document.get('schema') != 'mxfp8-qkv-forward-service-model-v1' or
            document.get('complete') is not True or document.get('frozen') is not True):
        raise ValueError('A complete, frozen mxfp8-qkv-forward-service-model-v1 report is required')
    models = document.get('models')
    if not isinstance(models, list) or len(models) != 1 or not isinstance(models[0], dict):
        raise ValueError('QKV forward calibration must contain exactly one CP4 card-group model')
    model = models[0]
    if (type(model.get('world_size')) is not int or type(model.get('sm_count')) is not int or
            model['world_size'] != world or model['sm_count'] != sm_count):
        raise ValueError('QKV forward model CP/SM metadata does not match this worker')
    # Keep this contract synchronized with the native selector's measured
    # physical domain, not with a list of registry names or tested winners.
    expected_domain = dict(min_m=32768, max_m=131072, min_n=4096, max_n=18432,
                           min_k=2048, max_k=16384, m_multiple=256, n_multiple=256,
                           k_multiple=64, head_dim=128,
                           tile_families=[[128, 256, 64, 2]], comm_ctas=list(range(4, 25, 2)))
    domain = model.get('domain')
    if (not isinstance(domain, dict) or domain != expected_domain or
            any(type(domain[key]) is not int for key in expected_domain
                if key not in ('tile_families', 'comm_ctas')) or
            any(type(value) is not int for value in domain['comm_ctas']) or
            any(type(value) is not int for tile in domain['tile_families'] for value in tile)):
        raise ValueError('QKV forward model domain differs from the native long-sequence N256 contract')
    coefficients = model.get('coefficients')
    if not isinstance(coefficients, dict):
        raise ValueError('QKV forward service coefficients are missing')
    keys = ('compute_gflop_sm_us', 'route_slot_task_us')
    values = [coefficients.get(key) for key in keys] + [model.get('minimum_gain')]
    try:
        valid = all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in values)
    except OverflowError:
        valid = False
    if not valid or values[0] == 0 or values[1] == 0 or values[2] >= 1:
        raise ValueError('QKV forward service rates must be positive, with a finite margin in [0,1)')
    return dict(schema=document['schema'], operator='qkv_forward', world_size=world, sm_count=sm_count,
                source=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                sha256=hashlib.sha256(raw).hexdigest(), domain=domain,
                coefficients=dict(zip(keys, values[:2])), minimum_gain=values[2],
                auto_reference='explicit_calibrated_model_or_domain_fallback_not_legacy_auto',
                scope='qkv_forward_only_manual_comm_ctas_take_precedence')


def qkv_forward_comm_model_values(model):
    return [model['coefficients'][key] for key in ('compute_gflop_sm_us', 'route_slot_task_us')] + [
        model['minimum_gain']]


def read_qkv_forward_comm_model(library, runtime):
    model = vars(library).get('_fuse_qkv_forward_comm_model')
    if model is None:
        return None
    query = library.fuse_mxfp8_test_get_qkv_forward_comm_model
    query.argtypes, query.restype = [ct.POINTER(ct.c_double)], ct.c_int
    values = (ct.c_double * 5)()
    runtime.check(query(values))
    if list(values) != [model['world_size'], model['sm_count'], *qkv_forward_comm_model_values(model)]:
        raise RuntimeError('Native QKV forward service model differs from the explicit calibration')
    return dict(model)


def configure_qkv_forward_comm_model(path, world, sm_count, library, runtime, sources, workers=None):
    if path is None:
        return None
    model = load_qkv_forward_comm_model(path, world, sm_count)
    if sources.get(model['source']) != model['sha256']:
        raise RuntimeError('QKV forward service calibration changed since source fingerprinting')
    setter = library.fuse_mxfp8_test_set_qkv_forward_comm_model
    setter.argtypes = [ct.c_int32, ct.c_int32, ct.POINTER(ct.c_double)]
    setter.restype = ct.c_int
    values = (ct.c_double * 3)(*qkv_forward_comm_model_values(model))
    runtime.check(setter(world, sm_count, values))
    library._fuse_qkv_forward_comm_model = model
    read_qkv_forward_comm_model(library, runtime)
    if workers is not None:
        installed = [None] * world
        workers.all_gather_object(installed, model)
        if any(other != model for other in installed):
            raise RuntimeError('Workers did not install the same QKV forward service model')
    return model


def checkpoint(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class OperatorCase:
    def __init__(self, case, library, runtime):
        from test_operators import Arguments

        self.case, self.library, self.runtime = case, library, runtime
        self.name = operator_name(case)
        self.op = OPERATORS.index(self.name)
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
        self.stream = torch.cuda.Stream(device=self.device)
        self.m, self.h = case['m'], case['hidden']
        self.q = case['q_heads'] * case['head_dim']
        self.kv = case['kv_heads'] * case['head_dim']
        self.width = self.q + 2 * self.kv if self.op in (0, 2) else self.q
        self.wr, self.wc = ((self.width, self.h) if self.op in (0, 2) else (self.h, self.q))
        self.causal = case['layout'] == 'causal_paired'
        self.tensors = {}
        self.data_cache = {}
        with torch.cuda.stream(self.stream):
            self._inputs_and_reference()
            # Conservative bound across the existing BF16 dispatch tiles.
            # Ready flags have 32 uint32 padding; M>=64 and N>=32 cover both
            # output-tile and per-head (D=128) readiness indexing.
            ready = ((self.m + 63) // 64) * ((max(self.wr, self.wc) + 31) // 32) * 32
            self.arena = rt.PeerArena(runtime, self.m * self.width * 2, ready, self.stream)
            if self.op == 1:
                self.arena.write_data(self.tensors['lhs'])
            self.stream.synchronize()
            dist.barrier()
        self.argument = Arguments()
        self.argument.geometry[:] = [self.op, self.rank, self.world, self.m, self.h,
                                     case['q_heads'], case['kv_heads'], case['head_dim'],
                                     1, int(self.causal), 1, 0, 0]
        names = ('lhs', 'gk', 'gv', 'saved', 'q', 's', 'workspace', 'staging', 'out', 'dw')
        pointers = [self.tensors[key].data_ptr() if key in self.tensors else 0 for key in names]
        if self.op == 1:
            pointers[0] = self.arena.pointer('data')
        if self.op == 2:
            pointers[7] = self.arena.pointer('data')
        if self.op in (0, 3):
            pointers[8] = self.arena.pointer('data')
        self.argument.tensors[:] = pointers + [self.arena.pointer('ready'),
                                               self.arena.pointer('done'), self.stream.cuda_stream]
        for region, field in (('data', 'peer_data'), ('ready', 'peer_ready'), ('done', 'peer_done')):
            getattr(self.argument, field)[:self.world] = [self.arena.pointer(region, peer)
                                                        for peer in range(self.world)]
        self.argument.alpha, self.argument.beta = 1, 0
        self.library.fuse_mxfp8_test_launch.argtypes = [ct.POINTER(Arguments)]
        self.library.fuse_mxfp8_test_launch.restype = ct.c_int
        self.config = dict(dispatch='existing_bf16_production_auto', weight_block_size=32,
                           payload='e4m3', scale='e8m0', weight_axis='original_forward_K',
                           gemm_input='bf16', gemm_accumulator='fp32', alpha=1,
                           route_layout=case['layout'], epoch=1,
                           ready_elements=ready, peer_arena_bytes=self.arena.bytes,
                           weight_workspace_bytes=self.wr * self.wc * 2,
                           forward_or_data_mnk=([self.m, self.wr, self.wc] if self.op < 2
                                                else [self.m, self.wc, self.wr]),
                           weight_mnk=[self.wr, self.wc, self.m] if self.op >= 2 else None)
        self.refresh_config()

    def refresh_config(self, requested_comm_ctas=0):
        """Read the same native selectors after a test-only CTA override."""
        from test_operators import Arguments

        self.library.fuse_mxfp8_test_config.argtypes = [ct.POINTER(Arguments), ct.POINTER(ct.c_int32)]
        self.library.fuse_mxfp8_test_config.restype = ct.c_int
        selected = (ct.c_int32 * 10)()
        self.runtime.check(self.library.fuse_mxfp8_test_config(ct.byref(self.argument), selected))
        self.config.update(requested_comm_ctas=requested_comm_ctas, comm_ctas=selected[0],
                           dispatch=('explicit_comm_existing_bf16_tile_auto' if requested_comm_ctas
                                     else 'existing_bf16_production_auto'),
                           tile_m=selected[1], tile_n=selected[2], tile_k=selected[3],
                           cluster_m=selected[4], sm_count=selected[5], policy_enum=selected[6],
                           raster='n', swizzle=1, gemm_policy_request='auto')
        if self.op == 0:
            model = read_qkv_forward_comm_model(self.library, self.runtime)
            if model is not None:
                if model['world_size'] != self.world or model['sm_count'] != selected[5]:
                    raise RuntimeError('QKV forward case does not match the installed service-model domain')
                self.config['qkv_forward_comm_model'] = model
                if not requested_comm_ctas:
                    self.config['dispatch'] = 'calibrated_model_or_domain_fallback'
        if self.op == 1:
            model = read_oproj_comm_model(self.library, self.runtime)
            if model is not None:
                if model['world_size'] != self.world or model['sm_count'] != selected[5]:
                    raise RuntimeError('OProj case does not match the installed service-model domain')
                self.config['oproj_comm_model'] = model
                if not requested_comm_ctas:
                    self.config['dispatch'] = 'calibrated_model_or_domain_fallback'
        if self.op >= 2:
            query = self.library.fuse_mxfp8_test_wgrad_config
            query.argtypes = [ct.POINTER(Arguments), ct.POINTER(ct.c_int32)]
            query.restype = ct.c_int
            weight = (ct.c_int32 * 8)()
            self.runtime.check(query(ct.byref(self.argument), weight))
            self.config['weight_gemm'] = dict(policy_enum=weight[0], tile_m=weight[1],
                                              tile_n=weight[2], tile_k=weight[3],
                                              cluster_m=weight[4], stages=weight[5],
                                              dynamic_smem_bytes=weight[6],
                                              registers_per_thread=weight[7],
                                              raster='n', swizzle=1, output_dtype='fp32')

    def values(self, shape, seed):
        return rt.deterministic(shape, seed, self.device)

    def rows(self, rank):
        return rt.global_rows(rank, self.m, self.world, self.causal, self.device)

    def _inputs_and_reference(self):
        t = self.tensors
        weight = self.values((self.wr, self.wc), 501)
        t['q'], t['s'] = rt.quantize_offline(weight)
        self.effective = rt.reference_dequant(t['q'], t['s'])
        del weight
        t['workspace'] = torch.empty_like(self.effective)
        t['staging'] = torch.empty((self.m, self.width), dtype=torch.bfloat16, device=self.device)
        self.reference = {}
        if self.op == 0:
            t['lhs'] = self.values((self.m, self.h), 401 + self.rank)
            self.reference['gemm_input'] = t['lhs']
            expected = torch.empty(self.m * self.width, dtype=torch.bfloat16, device=self.device)
            widths = ((0, self.q // self.world), (self.q, self.kv // self.world),
                      (self.q + self.kv, self.kv // self.world))
            local_weight = torch.cat([self.effective[offset + self.rank * width:
                                                       offset + (self.rank + 1) * width]
                                      for offset, width in widths])
            for peer in range(self.world):
                source = t['lhs'] if peer == self.rank else self.values((self.m, self.h), 401 + peer)
                projected = source @ local_weight.T
                column, base = 0, 0
                for _, width in widths:
                    expected[base + peer * self.m * width:base + (peer + 1) * self.m * width].copy_(
                        projected[:, column:column + width].reshape(-1))
                    column += width
                    base += self.m * self.world * width
            self.reference['out'] = expected
            self.reference['local'] = t['lhs'] @ self.effective.T
        elif self.op == 1:
            slab = self.q // self.world
            t['lhs'] = self.values((self.m * self.world, slab), 401 + self.rank)
            routed = torch.empty((self.m, self.q), dtype=torch.bfloat16, device=self.device)
            rows = self.rows(self.rank)
            for peer in range(self.world):
                source = t['lhs'] if peer == self.rank else self.values((self.m * self.world, slab), 401 + peer)
                routed[:, peer * slab:(peer + 1) * slab].copy_(source[rows])
            self.reference['gemm_input'] = routed
            self.reference['route'] = routed
            self.reference['out'] = routed @ self.effective.T
        elif self.op == 2:
            rows = self.rows(self.rank)
            t['saved'] = self.values((self.m, self.h), 401 + self.rank)
            dy = torch.empty((self.m, self.width), dtype=torch.bfloat16, device=self.device)
            for key, offset, width, seed in (('lhs', 0, self.q // self.world, 101),
                                            ('gk', self.q, self.kv // self.world, 201),
                                            ('gv', self.q + self.kv, self.kv // self.world, 301)):
                t[key] = self.values((self.m * self.world, width), seed + self.rank)
                for peer in range(self.world):
                    source = t[key] if peer == self.rank else self.values((self.m * self.world, width), seed + peer)
                    dy[:, offset + peer * width:offset + (peer + 1) * width].copy_(source[rows])
            self.reference['gemm_input'] = dy
            self.reference['route'] = dy
            self.reference['out'] = dy @ self.effective
        else:
            t['lhs'] = self.values((self.m, self.h), 101 + self.rank)
            t['saved'] = self.values((self.m, self.q), 401 + self.rank)
            self.reference['gemm_input'] = t['lhs']
            self.reference['local'] = t['lhs'] @ self.effective
            slab = self.q // self.world
            expected = torch.empty((self.m * self.world, slab), dtype=torch.bfloat16, device=self.device)
            for peer in range(self.world):
                source = t['lhs'] if peer == self.rank else self.values((self.m, self.h), 101 + peer)
                expected[self.rows(peer)] = source @ self.effective[:, self.rank * slab:(self.rank + 1) * slab]
            self.reference['out'] = expected.reshape(-1)
        t['out'] = torch.empty_like(self.reference['out'])
        if self.op >= 2:
            self.reference['dw'] = torch.mm(self.reference['gemm_input'].T, t['saved'], out_dtype=torch.float32)
            t['dw'] = torch.empty_like(self.reference['dw'])
        self.scratch = torch.empty(self.m * self.width, dtype=torch.bfloat16, device=self.device)

    def launch(self, phase, mode):
        argument = type(self.argument).from_buffer_copy(self.argument)
        argument.beta = int(mode == 'deferred')
        argument.geometry[11] = int(phase in ('data', 'weight') or mode == 'deferred')
        argument.geometry[12] = int(phase == 'weight')

        def call():
            self.runtime.check(self.library.fuse_mxfp8_test_launch(ct.byref(argument)))

        return call

    def prepare(self, phase, mode, poison=False):
        def reset():
            self.arena.reset_control()
            if self.op >= 2 and (phase in ('weight', 'total') or poison):
                self.tensors['dw'].fill_(0.125 if mode == 'deferred' else 0)
            if poison:
                self.tensors['workspace'].fill_(float('nan'))
                self.tensors['staging'].fill_(float('nan'))
                self.tensors['out'].fill_(float('nan'))
                if self.op in (0, 2, 3) and phase != 'weight':
                    self.scratch.fill_(float('nan'))
                    self.arena.write_data(self.scratch)
        return reset

    def launch_boundary(self, phase, mode):
        """Match the measured boundary, including explicit deferred B then W."""
        if phase == 'total' and mode == 'deferred':
            data, weight = self.launch('data', mode), self.launch('weight', mode)

            def call():
                data()
                weight()
            return call
        return self.launch(phase, mode)

    def verify(self, phase, mode, accumulated=1):
        result = {}
        with torch.cuda.stream(self.stream):
            if phase != 'weight':
                result['weight_dequant'] = rt.check_error(self.tensors['workspace'], self.effective, exact=True)
                if self.op in (0, 2, 3):
                    self.arena.read_data(self.scratch)
                if self.op in (0, 3):
                    actual = self.scratch
                else:
                    actual = self.tensors['out']
                result['output'] = rt.check_error(actual, self.reference['out'])
                if self.op == 2:
                    result['route'] = rt.check_error(self.scratch.view(self.m, self.width),
                                                    self.reference['route'], exact=True)
                elif self.op == 1:
                    result['route'] = rt.check_error(self.tensors['staging'], self.reference['route'], exact=True)
                else:
                    result['local_gemm'] = rt.check_error(self.tensors['staging'], self.reference['local'])
            if self.op >= 2 and phase in ('weight', 'total'):
                expected = self.reference['dw'] * accumulated
                if mode == 'deferred':
                    expected = expected + 0.125
                result['weight_gradient'] = rt.check_error(self.tensors['dw'], expected)
        return result

    def flops(self, phase):
        one = 2 * self.m * self.wr * self.wc
        return one * (2 if phase == 'total' and self.op >= 2 else 1)

    def fuse_record(self, launch, mode, args):
        phases = ('forward',) if self.op < 2 else ('data', 'weight', 'total')
        record = dict(backend='fuse', launch=launch, weight_mode=mode,
                      grad_dtype='fp32' if self.op >= 2 else None,
                      beta=int(mode == 'deferred'),
                      warmup=args.warmup, iterations=args.iterations,
                      config=dict(self.config, beta=int(mode == 'deferred')), correctness={})
        for phase in phases:
            cache_key = ('fuse', launch)
            if phase == 'data' and cache_key in self.data_cache:
                record['data'], checks = self.data_cache[cache_key]
                record['correctness'].update(checks)
                record['data_reused_across_weight_modes'] = True
                continue
            function = self.launch_boundary(phase, mode)
            if phase == 'weight' and self.op == 2:
                with torch.cuda.stream(self.stream):
                    self.arena.write_data(self.reference['gemm_input'])
                    self.stream.synchronize()
                    dist.barrier()
            invocation = rt.Invocation(function, launch, args.warmup,
                                       self.prepare(phase, mode), self.stream)
            # Poison only relevant outputs after warmup/capture. W-only must
            # retain its input dQKV staging and does not consume MXFP8 weight.
            invocation.once(self.prepare(phase, mode, poison=True))
            record['correctness'][phase] = self.verify(phase, mode)
            record[phase] = invocation.measure(args.iterations, self.flops(phase))
            # Measured outputs are checked, not just the pre-timing warmup.
            record['correctness'][phase + '_after_samples'] = self.verify(phase, mode)
            if phase == 'data':
                self.data_cache[cache_key] = (record['data'], {
                    key: value for key, value in record['correctness'].items()
                    if key in ('data', 'data_after_samples')})
                record['data_reused_across_weight_modes'] = False
            if phase == 'weight' and mode == 'deferred':
                invocation.once(self.prepare(phase, mode))
                invocation.once(self.arena.reset_control)
                record['correctness']['nonzero_beta1_twice'] = self.verify(phase, mode, accumulated=2)
            del invocation
        if self.op >= 2:
            record['total_semantic'] = 'actual_B_then_W_not_ZeroBubble_scheduler'
            record['isolated_sum_us'] = record['data']['p50_us'] + record['weight']['p50_us']
        return record

    def pure_cublas_record(self, launch, mode, args):
        from classic_gemm import ClassicGemm

        lhs, weight = self.reference['gemm_input'], self.effective
        output = torch.empty((self.m, self.wr if self.op < 2 else self.wc),
                             dtype=torch.bfloat16, device=self.device)
        data = ClassicGemm(lhs, weight, output, tb=self.op < 2)
        dw = torch.empty_like(self.reference['dw']) if self.op >= 2 else None
        wgemm = (ClassicGemm(lhs, self.tensors['saved'], dw, ta=True,
                            beta=int(mode == 'deferred')) if self.op >= 2 else None)
        record = dict(backend='pure_cublas', launch=launch, weight_mode=mode,
                      grad_dtype='fp32' if self.op >= 2 else None,
                      beta=int(mode == 'deferred'),
                      warmup=args.warmup, iterations=args.iterations,
                      config=dict(library='classic_cublas_not_cublaslt',
                                  dq_included=False, communication_included=False,
                                  data=getattr(data, 'config', {}),
                                  weight=getattr(wgemm, 'config', {}) if wgemm else None),
                      correctness={})
        for phase in (('forward',) if self.op < 2 else ('data', 'weight', 'total')):
            cache_key = ('pure_cublas', launch)
            if phase == 'data' and cache_key in self.data_cache:
                record['data'], checks = self.data_cache[cache_key]
                record['correctness']['data'] = checks
                record['data_reused_across_weight_modes'] = True
                continue
            def function():
                if phase != 'weight':
                    data()
                if phase in ('weight', 'total'):
                    wgemm()

            def prepare():
                if dw is not None and phase in ('weight', 'total'):
                    dw.fill_(0.125 if mode == 'deferred' else 0)

            invocation = rt.Invocation(function, launch, args.warmup, prepare, self.stream)
            record[phase] = invocation.measure(args.iterations, self.flops(phase))
            with torch.cuda.stream(self.stream):
                metrics = {}
                if phase != 'weight':
                    expected = self.reference.get('local', self.reference['out'])
                    metrics['gemm_output'] = rt.check_error(output, expected)
                if phase in ('weight', 'total'):
                    metrics['weight_gradient'] = rt.check_error(
                        dw, self.reference['dw'] + (0.125 if mode == 'deferred' else 0))
                record['correctness'][phase] = metrics
                if phase == 'data':
                    self.data_cache[cache_key] = (record['data'], metrics)
                    record['data_reused_across_weight_modes'] = False
            del invocation
        if self.op >= 2:
            record['total_semantic'] = 'actual_GEMM_only_B_then_W_no_DQ_no_A2A'
            record['isolated_sum_us'] = record['data']['p50_us'] + record['weight']['p50_us']
        data.close()
        if wgemm:
            wgemm.close()
        return record

    def close(self):
        self.arena.close()


def main():
    args = arguments()
    for field, permitted in (('launches', {'eager', 'graph'}),
                              ('weight_modes', {'immediate', 'deferred'}),
                              ('backends', {'fuse', 'pure_cublas'})):
        items = getattr(args, field).split(',')
        if not set(items) <= permitted or len(items) != len(set(items)):
            raise ValueError(f'Invalid --{field.replace("_", "-")}')
    if min(args.warmup, args.iterations) < 1:
        raise ValueError('Warmup and iteration counts must be positive')
    if args.list_only:
        rows = select_cases(args)
        print(json.dumps(dict(settings=len(rows), cases=rows), indent=2))
        return

    initialize_runtime()

    if int(os.environ.get('WORLD_SIZE', 1)) not in (4, 8):
        raise RuntimeError('Use torchrun --nproc-per-node=4 or 8; one process per GPU is required')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    dist.init_process_group('gloo', timeout=timedelta(minutes=30))
    rank, world = dist.get_rank(), dist.get_world_size()
    selected = select_cases(args, world)
    props = torch.cuda.get_device_properties(local_rank)
    metadata = dict(rank=rank, local_rank=local_rank, name=props.name,
                    cc=[props.major, props.minor], sm_count=props.multi_processor_count,
                    total_memory_bytes=props.total_memory,
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    devices = [None] * world
    dist.all_gather_object(devices, metadata)
    clock_snapshot = [None]
    if rank == 0:
        clock_snapshot[0] = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,uuid,clocks.max.sm,clocks.current.sm,clocks.max.memory,power.limit',
             '--format=csv'], text=True)
    dist.broadcast_object_list(clock_snapshot, src=0)
    if any(device['cc'] != [9, 0] for device in devices):
        raise RuntimeError('This production benchmark targets SM90')
    report = dict(schema='mxfp8-production-operators-v1', devices=devices,
                  gpu_clock_snapshot=clock_snapshot[0],
                  policy_environment={key: os.environ.get(key) for key in POLICY_ENV_VARIABLES},
                  args={key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items() if key != 'resume'},
                  sources=source_hashes(args), torch=torch.__version__, cuda=torch.version.cuda,
                  semantic='offline_mxfp8_weight_runtime_dq_bf16_tensor_core_gemm',
                  token_semantic='T_is_flattened_tokens_no_training_batch',
                  timing=dict(sample_statistic='sample_wise_max_across_ranks',
                              clocks='CUDA_events_one_operation_per_sample',
                              graph='one_operation_capture_then_replay',
                              included=['software_weight_dequant_in_Fuse_F_and_B',
                                        'BF16_GEMM', 'BF16_A2A_in_Fuse_F_and_B'],
                              excluded=['offline_weight_quantization', 'references_and_correctness',
                                        'allocation_IPC_bootstrap_capture_and_JIT',
                                        'ready_done_epoch_reset_and_CPU_Gloo_barrier',
                                        'main_grad_initialization_for_each_beta1_sample'],
                              pure_cublas='GEMM_only_no_DQ_no_A2A'),
                  selected_case_ids=[case['id'] for case in selected], cases=[], complete=False)
    if args.resume and args.output.exists():
        old = json.loads(args.output.read_text())
        for key in ('schema', 'devices', 'args', 'sources', 'torch', 'cuda', 'selected_case_ids'):
            if old.get(key) != report[key]:
                raise RuntimeError(f'Resume fingerprint mismatch: {key}')
        report = old
        report['complete'] = False
    elif args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}; use --resume or a new output')
    done = {case['case']['id'] for case in report['cases']}
    started = time.monotonic()
    previous_wall = report.get('wall_seconds', 0)
    runtime = rt.CudaRuntime()
    library = ct.CDLL(str(args.library))
    try:
        library.fuse_mxfp8_test_profiling_enabled.restype = ct.c_int
        if library.fuse_mxfp8_test_profiling_enabled() != 0:
            raise RuntimeError('Formal benchmark refuses a profiling-enabled library')
        model = configure_oproj_comm_model(args.oproj_comm_model, world, props.multi_processor_count,
                                           library, runtime, report['sources'], workers=dist)
        if model is not None:
            report['oproj_comm_model'] = model
        qkv_model = configure_qkv_forward_comm_model(args.qkv_comm_model, world, props.multi_processor_count,
                                                     library, runtime, report['sources'], workers=dist)
        if qkv_model is not None:
            report['qkv_forward_comm_model'] = qkv_model
        for case in selected:
            if case['id'] in done:
                continue
            if rank == 0:
                print(f'[{len(done) + 1}/{len(selected)}] {case["id"]}', flush=True)
            state = OperatorCase(case, library, runtime)
            record = dict(case=case, operator=state.name, records=[])
            try:
                for backend in args.backends.split(','):
                    for launch in args.launches.split(','):
                        for mode in (args.weight_modes.split(',') if state.op >= 2 else (None,)):
                            method = state.fuse_record if backend == 'fuse' else state.pure_cublas_record
                            record['records'].append(method(launch, mode, args))
                report['cases'].append(record)
                done.add(case['id'])
                report['wall_seconds'] = previous_wall + time.monotonic() - started
                if rank == 0:
                    checkpoint(args.output, report)
            finally:
                state.close()
                del state
                gc.collect()
                torch.cuda.empty_cache()
        report['complete'] = len(done) == len(selected)
        report['wall_seconds'] = previous_wall + time.monotonic() - started
        if rank == 0:
            checkpoint(args.output, report)
            print(f'Completed {len(done)} settings: {args.output}', flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
