#!/usr/bin/env python3
"""Grouped MXFP8-weight backward benchmark. See BACKWARD.md for scope."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

from backward_matrix import ROOT, full_matrix


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--full', action='store_true')
    p.add_argument('--list-only', action='store_true')
    p.add_argument('--model', default='production_qwen_dense')
    p.add_argument('--seqs', default='1024')
    p.add_argument('--operator', choices=('both', 'qkv', 'oproj'), default='both')
    p.add_argument('--backends', default='cublaslt_nccl,teub')
    p.add_argument('--grad-dtypes', default='fp32')
    p.add_argument('--launches', default='eager,graph')
    p.add_argument('--weight-modes', default='immediate,deferred')
    p.add_argument('--ub-sms', default='4,8,16')
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--iterations', type=int, default=50)
    p.add_argument('--sweep-warmup', type=int, default=3)
    p.add_argument('--sweep-iters', type=int, default=12)
    p.add_argument('--candidates', type=int, default=32)
    p.add_argument('--phase', choices=('sweep', 'formal'), default='formal')
    p.add_argument('--skip-sweep', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--validation', action='store_true')
    p.add_argument('--launch-policy', type=Path, help='case ID -> launches selected for this NCCL profile')
    p.add_argument('--case-ids', type=Path)
    p.add_argument('--output', type=Path, required=True)
    return p.parse_args()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_case(c, args, helper, cpu):
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    m, wn, wk = c['m'], *c['weight_shape']
    qkv = c['operator'] == 'qkv'
    seq_local = c['global_seq'] // world
    causal = c['layout'] == 'causal_paired'
    weight = rt.deterministic((wn, wk), 501, device)
    payload, scales = rt.quantize_offline(weight)
    reference_w = rt.reference_dequant(payload, scales)
    del weight
    weight_workspace = torch.empty_like(reference_w)
    saved = rt.deterministic((m, wk), 401 + rank, device)
    dy = torch.empty((m, wn), device=device, dtype=torch.bfloat16)
    dx = torch.empty((m, wk), device=device, dtype=torch.bfloat16)
    qwidth, kvwidth = c['q_heads'] * c['head_dim'], c['kv_heads'] * c['head_dim']
    slab = (wn if qkv else wk) // world
    shape = (world, m, slab)
    rows = rt.global_rows(rank=rank, local_tokens=m, batch=c['batch'], world=world,
                          causal=causal, device=device)
    if qkv:
        qlocal, kvlocal = qwidth // world, kvwidth // world
        gradients = [rt.deterministic((c['batch'] * c['global_seq'], width), seed + rank, device)
                     for width, seed in ((qlocal, 101), (kvlocal, 201), (kvlocal, 301))]
        reference_dy = torch.empty_like(dy)
        for peer in range(world):
            for offset, width, seed in ((0, qlocal, 101), (qwidth, kvlocal, 201),
                                        (qwidth + kvwidth, kvlocal, 301)):
                full = rt.deterministic((c['batch'] * c['global_seq'], width), seed + peer, device)
                reference_dy[:, offset + peer*width:offset + (peer+1)*width].copy_(full[rows])
                del full
        dy.copy_(reference_dy)
        reference_dx = reference_dy @ reference_w
        routed = dx
        reference_routed = reference_dx
    else:
        dy.copy_(rt.deterministic((m, wn), 101 + rank, device))
        reference_dy = dy
        gradients = []
        routed = torch.empty((c['batch'] * c['global_seq'], slab), device=device, dtype=torch.bfloat16)
        reference_routed = torch.empty_like(routed)
        for peer in range(world):
            peer_dy = rt.deterministic((m, wn), 101 + peer, device)
            peer_rows = rt.global_rows(rank=peer, local_tokens=m, batch=c['batch'], world=world,
                                       causal=causal, device=device)
            reference_routed[peer_rows] = peer_dy @ reference_w[:, rank*slab:(rank+1)*slab]
            del peer_dy
        reference_dx = dy @ reference_w
    # Independent torch cuBLAS reference, FP32 output without promoting huge
    # saved activations to FP32. Actual autograd is exercised by the small gate.
    reference_dw = (torch.mm(reference_dy.T, saved, out_dtype=torch.float32)
                    if args.phase != 'sweep' or args.validation else None)
    autograd_check = None
    if args.validation:
        ax = saved.float().requires_grad_()
        aw = reference_w.float().requires_grad_()
        torch.nn.functional.linear(ax, aw).backward(reference_dy.float())
        autograd_check = dict(dgrad=rt.check_error(reference_dx, ax.grad.to(torch.bfloat16), cpu),
                              wgrad=rt.check_error(reference_dw, aw.grad, cpu))
        del ax, aw
    def dequant():
        rt.dequant_kernel[(triton.cdiv(wn*wk, 2048),)](payload, scales, weight_workspace, wn*wk, 2048)
    dequant()
    if not torch.equal(weight_workspace, reference_w):
        raise RuntimeError('weight DQ differs from independent original-axis reference')
    plan_b = Gemm(dy, weight_workspace, dx, candidates=args.candidates)
    wplans = {}
    outputs = {}
    for dtype in (args.grad_dtypes.split(',') if args.phase != 'sweep' else []):
        outputs[dtype] = torch.empty((wn, wk), device=device,
                                     dtype=torch.float32 if dtype == 'fp32' else torch.bfloat16)
        for mode in args.weight_modes.split(','):
            beta = int(mode == 'deferred')
            wplans[dtype, mode] = Gemm(dy, saved, outputs[dtype], ta=True, beta=beta,
                                        candidates=args.candidates)
    records, search = [], []
    for backend in args.backends.split(','):
        candidates = [0] if backend == 'cublaslt_nccl' else [int(x) for x in args.ub_sms.split(',')]

        def make(sms):
            ub = rt.Userbuffers(shape, helper, device, rank, world, sms) if sms else None
            send = ub.send if ub else torch.empty(shape, dtype=torch.bfloat16, device=device)
            recv = ub.recv if ub else torch.empty_like(send)
            self_send, self_recv = send[rank], recv[rank]
            def exchange():
                if ub:
                    ub.begin()
                    for i, peer in enumerate(ub.peers):
                        ub.send_peer(peer, i)
                    ub.receive_and_join()
                    # Legacy unpack consumes all sources; UB self transfer is
                    # local, and this explicit copy stays inside B timing.
                    self_recv.copy_(self_send)
                else:
                    dist.all_to_all_single(recv, send)
            def data():
                if qkv:
                    rt._qkv_inverse_pack_kernel[(triton.cdiv(send.numel(), 512),)](
                        *gradients, send, send.numel(), m, seq_local, world,
                        qlocal, kvlocal, slab, causal, 512, num_warps=4)
                    exchange()
                    rt._qkv_inverse_unpack_kernel[(triton.cdiv(dy.numel(), 512),)](
                        recv, dy, dy.numel(), m, world, qlocal, kvlocal, slab,
                        qwidth, kvwidth, wn, 512, num_warps=4)
                    dequant()
                    plan_b(dy, weight_workspace, dx)
                else:
                    dequant()
                    plan_b(dy, weight_workspace, dx)
                    rt._oproj_route_pack_kernel[(triton.cdiv(send.numel(), 512),)](
                        dx, send, send.numel(), m, wk, slab, 512, num_warps=4)
                    exchange()
                    rt._oproj_route_unpack_kernel[(triton.cdiv(recv.numel(), 512),)](
                        recv, routed, recv.numel(), m, seq_local, world, slab, causal, 512, num_warps=4)
            def poison():
                for t in (weight_workspace, dx, send, recv, routed):
                    t.fill_(float('nan'))
                if qkv:
                    dy.fill_(float('nan'))
            def check():
                route = rt.check_error(dy, reference_dy, cpu) if qkv else rt.check_error(routed, reference_routed, cpu)
                if qkv and route['max_abs'] != 0:
                    raise RuntimeError(f'QKV inverse route must be exact: {route}')
                return dict(route=route, dgrad=rt.check_error(dx, reference_dx, cpu))
            return data, poison, check, ub

        # Select B communication policy with short data-only measurements.
        # It is independent of dW dtype/beta; reuse this decision across them.
        choices = {}
        if args.skip_sweep:
            if backend != 'cublaslt_nccl':
                raise ValueError('skip-sweep is only valid for externally selected NCCL policies')
            choices = {launch: (0, 0) for launch in args.launches.split(',')}
            candidates = []
        for sms in candidates:
            data, poison, check, ub = make(sms)
            for launch in args.launches.split(','):
                measured = rt.measure(data, launch, args.sweep_warmup, args.sweep_iters, cpu, poison)
                correctness = check()
                search.append(dict(backend=backend, launch=launch, sms=sms,
                                   data=measured, correctness=correctness))
                if launch not in choices or measured['p50_us'] < choices[launch][0]:
                    choices[launch] = (measured['p50_us'], sms)
            torch.cuda.synchronize(); dist.barrier(group=cpu)
            del data, poison, check, ub
            gc.collect()
        if args.phase == 'sweep':
            continue
        for launch, (_, sms) in choices.items():
            data, poison, check, ub = make(sms)
            b = rt.measure(data, launch, args.warmup, args.iterations, cpu, poison)
            bcheck = check()
            for dtype in args.grad_dtypes.split(','):
                out = outputs[dtype]
                for mode in args.weight_modes.split(','):
                    beta = int(mode == 'deferred')
                    plan_w = wplans[dtype, mode]
                    def weight_phase():
                        plan_w(dy, saved, out, beta=beta)
                    initial = rt.deterministic((wn, wk), 901 + rank, device).to(out.dtype)
                    out.copy_(initial)
                    expected = initial
                    accumulate_checks = []
                    for _ in range(2):
                        plan_w(dy, saved, out, beta=1)
                        expected = (expected.float() + reference_dw).to(out.dtype)
                        accumulate_checks.append(rt.check_error(out, expected, cpu))
                    del initial, expected
                    w = rt.measure(weight_phase, launch, args.warmup, args.iterations, cpu, lambda: out.zero_())
                    def expected_output():
                        if not beta:
                            return reference_dw.to(out.dtype)
                        expected = torch.zeros_like(out)
                        for _ in range(args.iterations):
                            expected = (expected.float() + reference_dw).to(out.dtype)
                        return expected
                    wcheck = rt.check_error(out, expected_output(), cpu)
                    def full():
                        data()
                        weight_phase()
                    def poison_full():
                        poison()
                        out.zero_()
                    total = rt.measure(full, launch, args.warmup, args.iterations, cpu, poison_full)
                    fullcheck = check()
                    fullcheck['wgrad'] = rt.check_error(out, expected_output(), cpu)
                    records.append(dict(backend=backend, launch=launch, weight_mode=mode,
                        grad_dtype=dtype, beta=beta, sms=sms, data=b, weight=w, total=total,
                        isolated_sum_samples_us=[x+y for x,y in zip(b['samples_us'], w['samples_us'])],
                        data_reused_across_grad_dtype_and_mode=True,
                        correctness=dict(data=bcheck, weight=wcheck, full=fullcheck,
                                         nonzero_beta1_twice=accumulate_checks),
                        b_plan=plan_b.info, w_plan=plan_w.info))
            torch.cuda.synchronize(); dist.barrier(group=cpu)
            del data, poison, check, ub
            gc.collect()
    plan_b.close()
    for plan in wplans.values():
        plan.close()
    return dict(case=c, records=records, search=search, autograd_check=autograd_check)


def main():
    args = arguments()
    matrix = full_matrix()
    if args.list_only:
        print(json.dumps(matrix, indent=2)); return
    for value, allowed in ((args.backends, {'cublaslt_nccl', 'teub'}),
                           (args.grad_dtypes, {'bf16', 'fp32'}),
                           (args.launches, {'eager', 'graph'}),
                           (args.weight_modes, {'immediate', 'deferred'})):
        if not set(value.split(',')) <= allowed:
            raise ValueError(value)
    global torch, dist, triton, rt, Gemm
    import torch
    import torch.distributed as dist
    import triton
    import backward_runtime as rt
    from backward_gemm import Gemm, LIBRARY
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group('nccl', device_id=torch.device('cuda', torch.cuda.current_device()))
    cpu = dist.new_group(backend='gloo')
    world, rank = dist.get_world_size(), dist.get_rank()
    helper = rt.tex.CommOverlapHelper(dist.group.WORLD, dist.group.WORLD)
    selected = [r for r in matrix if r['cp'] == world and
                (args.operator == 'both' or r['operator'] == args.operator) and
                (args.full or (r['model'] == args.model and r['global_seq'] in map(int, args.seqs.split(','))))]
    if args.case_ids:
        ids = set(json.loads(args.case_ids.read_text()))
        selected = [r for r in matrix if r['id'] in ids and r['cp'] == world]
        if {r['id'] for r in selected} != ids:
            raise ValueError('unknown/wrong CP case IDs')
    if args.validation:
        selected = []
        for operator in ('qkv', 'oproj'):
            template = next(c for c in matrix if c['operator'] == operator and c['cp'] == world)
            for batch in (1, 2):
                for layout in ('rank_major', 'causal_paired'):
                    wn, wk = (1024, 128) if operator == 'qkv' else (128, 512)
                    m = 128 * batch
                    selected.append(dict(template, id=f'validation/{operator}/cp{world}/b{batch}/{layout}',
                        model='validation', global_seq=128*world, batch=batch, m=m,
                        hidden=128, q_heads=16, kv_heads=8, head_dim=32,
                        projection=1024 if operator == 'qkv' else 512,
                        weight_shape=[wn, wk], b_mnk=[m, wk, wn], w_mnk=[wn, wk, m], layout=layout))
    if not selected:
        raise ValueError('empty selection')
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    identity = dict(rank=rank, name=props.name, cc=[props.major, props.minor],
                    sms=props.multi_processor_count, total_memory=props.total_memory,
                    uuid=str(props.uuid))
    devices = [None] * world
    dist.all_gather_object(devices, identity, group=cpu)
    files = [Path(__file__), Path(rt.__file__), Path(__file__).with_name('backward_matrix.py'),
             Path(__file__).with_name('backward_gemm.py'), Path(__file__).with_name('backward_gemm.cu'), LIBRARY,
             ROOT / 'benchmarks/backward/backward_shape_bench.py']
    report = dict(schema='mxfp8-backward-v1', devices=devices, cases=[],
                  args={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                  sources={str(p.relative_to(ROOT)):sha(p) for p in files},
                  environment={k:v for k,v in os.environ.items() if k.startswith('NCCL_') or k == 'CUDA_VISIBLE_DEVICES'},
                  semantic='offline_mxfp8_original_weight_axis_runtime_dq_bf16_gemm',
                  plan_metadata='rank0 locally autotuned; timings and correctness cover all ranks',
                  timing='GPU events; CPU barrier outside sample; samplewise rank-max',
                  complete=False)
    launches_by_id = json.loads(args.launch_policy.read_text()) if args.launch_policy else None
    if launches_by_id is not None:
        if set(launches_by_id) != {c['id'] for c in selected}:
            raise ValueError('launch policy must exactly match requested case IDs')
        report['launch_policy'] = launches_by_id
    original_launches = args.launches
    if args.resume and args.output.exists():
        old = json.loads(args.output.read_text())
        for key in ('schema', 'devices', 'args', 'sources', 'environment', 'launch_policy'):
            if old.get(key) != report.get(key):
                raise RuntimeError(f'resume fingerprint changed: {key}')
        report = old
        completed = {x['case']['id'] for x in report['cases']}
        selected = [c for c in selected if c['id'] not in completed]
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists() and not args.resume:
            raise FileExistsError(f'refusing to overwrite {args.output}')
    if args.validation:
        expected = torch.ones(8192, device='cuda', dtype=torch.float32)
        actual = expected.clone()
        for value in (float('nan'), float('inf'), 100.):
            actual.copy_(expected)
            if rank == 1:
                actual[4097] = value
            failed = False
            try:
                rt.check_error(actual, expected, cpu)
            except RuntimeError:
                failed = True
            flags = [None] * world
            dist.all_gather_object(flags, failed, group=cpu)
            if not all(flags):
                raise RuntimeError(f'negative correctness gate failed: {flags}')
        report['one_rank_corruption_gate'] = 'passed_nan_inf_finite'
    def checkpoint():
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(args.output)
    for c in selected:
        args.launches = ','.join(launches_by_id[c['id']]) if launches_by_id else original_launches
        start = time.monotonic()
        report['cases'].append(run_case(c, args, helper, cpu))
        if rank == 0:
            checkpoint()
            print(f"DONE {c['id']} rows={len(report['cases'][-1]['records'])} seconds={time.monotonic()-start:.1f}", flush=True)
        gc.collect(); torch.cuda.empty_cache()
    report['complete'] = True
    if rank == 0:
        checkpoint()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
