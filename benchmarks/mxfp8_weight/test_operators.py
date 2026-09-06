"""Production 2F2B correctness, not a performance benchmark.

Independent CPU quantization/routing and FP32 GEMM references. Like the existing
backward_torch_autograd test, Python owns storage and calls a test-only C ABI.
All ranks are launched before any rank is synchronized. Graph replays explicitly
reset peer epochs between invocations; this is not a production epoch scheduler.
"""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


class Arguments(ct.Structure):
    _fields_ = [("geometry", ct.c_int32 * 13), ("tensors", ct.c_uint64 * 13),
                ("peer_data", ct.c_uint64 * 8), ("peer_ready", ct.c_uint64 * 8),
                ("peer_done", ct.c_uint64 * 8), ("alpha", ct.c_float),
                ("beta", ct.c_float), ("route_flags", ct.c_int32)]


def check(status):
    if status:
        raise RuntimeError(f"production CUDA API returned {status}")


def wgrad_policies(value):
    try:
        policies = [int(item) for item in value.split(',')]
    except ValueError as error:
        raise argparse.ArgumentTypeError('WGrad policies must be comma-separated integers') from error
    if len(set(policies)) != len(policies) or not set(policies) <= set(range(7)):
        raise argparse.ArgumentTypeError('Use unique WGrad policy IDs from 0,1,2,3,4,5,6')
    return policies


def wgrad_config(lib, argument):
    selected = (ct.c_int32 * 8)()
    check(lib.fuse_mxfp8_test_wgrad_config(ct.byref(argument), selected))
    keys = ('policy_enum', 'tile_m', 'tile_n', 'tile_k', 'cluster_m',
            'stages', 'dynamic_smem_bytes', 'registers_per_thread')
    return dict(zip(keys, selected))


def wgrad_policy_test(lib, device_count):
    """Check the native selector contract; resources come from the actual kernel."""
    tiles = {1: (128, 256, 64, 2), 2: (128, 128, 64, 2),
             3: (128, 128, 128, 2), 4: (128, 256, 64, 1),
             5: (128, 192, 64, 2), 6: (128, 256, 32, 2)}
    stages = {1: 3, 2: 5, 3: 2, 4: 3, 5: 4, 6: 6}
    shared_bytes = {1: 215040, 2: 231424, 3: 198656, 4: 215040,
                    5: 231424, 6: 215040}
    records = []
    try:
        for rank in range(device_count):
            torch.cuda.set_device(rank)
            props = torch.cuda.get_device_properties(rank)
            argument = Arguments()
            argument.geometry[0] = 2
            configs = {}
            for requested in range(7):
                check(lib.fuse_mxfp8_test_set_wgrad_policy(requested))
                config = wgrad_config(lib, argument)
                actual = requested or 1
                assert config['policy_enum'] == actual, (rank, requested, config)
                assert tuple(config[key] for key in ('tile_m', 'tile_n', 'tile_k', 'cluster_m')) == tiles[actual]
                assert config['stages'] == stages[actual], config
                assert config['dynamic_smem_bytes'] == shared_bytes[actual], config
                assert 0 < config['dynamic_smem_bytes'] <= props.shared_memory_per_block_optin, config
                assert 0 < config['registers_per_thread'] <= 255, config
                # QKV and OProj W use the same native policy visitor.
                argument.geometry[0] = 3
                assert wgrad_config(lib, argument) == config
                argument.geometry[0] = 2
                for invalid in (-1, 7):
                    assert lib.fuse_mxfp8_test_set_wgrad_policy(invalid) != 0, invalid
                    assert wgrad_config(lib, argument) == config, 'invalid setter mutated the active policy'
                configs[requested] = config
                records.append(dict(device=rank, requested_wgrad_policy=requested, weight_gemm=config))
            assert configs[0] == configs[1], 'auto must resolve to the explicit baseline kernel'
            output = (ct.c_int32 * 8)()
            assert lib.fuse_mxfp8_test_wgrad_config(None, output) != 0
            assert lib.fuse_mxfp8_test_wgrad_config(ct.byref(argument), None) != 0
            for unsupported_op in (0, 1, 4):
                argument.geometry[0] = unsupported_op
                assert lib.fuse_mxfp8_test_wgrad_config(ct.byref(argument), output) != 0
        print('WGrad policy 0..6 + invalid setters/queries + native resources: PASS', flush=True)
        return dict(passed=True, tested_policies=list(range(7)), invalid_policies=[-1, 7],
                    invalid_setter_preserves_policy=True, auto_matches_explicit_policy=1,
                    resource_source='native_kernel_traits_and_cudaFuncGetAttributes_not_estimates',
                    records=records)
    finally:
        check(lib.fuse_mxfp8_test_set_wgrad_policy(0))


def close(actual, expected, label, fp32=False):
    actual, expected = actual.cpu().float(), expected.float()
    assert actual.shape == expected.shape, label
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), label
    delta = actual - expected
    relative = delta.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-8)
    assert relative < (1e-4 if fp32 else 0.004), (label, float(relative), float(delta.abs().max()))


def quantize(weight):
    blocks = weight.float().reshape(weight.shape[0], -1, 32)
    exponent = torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1e-30) / 448))
    exponent = exponent.clamp(-127, 127)
    scale = torch.exp2(exponent)
    payload = (blocks / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    effective = (payload.float() * scale[..., None]).reshape_as(weight).bfloat16()
    return payload.reshape_as(weight).contiguous(), (exponent + 127).to(torch.uint8), effective


def rows_for(rank, m, world, causal):
    # A single flattened token extent; no training-batch or sequence boundaries.
    if causal:
        half = m // 2
        return torch.cat([torch.arange(rank * half, (rank + 1) * half),
                          torch.arange((2 * world - rank - 1) * half,
                                       (2 * world - rank) * half)])
    return torch.arange(rank * m, (rank + 1) * m)


def conversion_test(lib):
    # All 256 payload encodings x all 256 scale encodings, including E8M0=0,
    # NaN, BF16 overflow/underflow, subnormal ties and signed zero.
    payload = torch.arange(256, dtype=torch.uint8).repeat_interleave(32).repeat(256)
    scales = torch.arange(256, dtype=torch.uint8).repeat_interleave(256)
    reference = payload.view(torch.float8_e4m3fn).double() * torch.exp2(
        scales.repeat_interleave(32).double() - 127)
    reference[scales.repeat_interleave(32) == 255] = float('nan')
    reference = reference.bfloat16()
    q, s = payload.cuda(0), scales.cuda(0)
    out = torch.empty(payload.numel(), dtype=torch.bfloat16, device=0)
    stream = torch.cuda.current_stream(0).cuda_stream
    check(lib.fuse_mxfp8_test_dequant(q.data_ptr(), s.data_ptr(), out.data_ptr(),
                                    256, 8192, out.numel() * 2, stream))
    actual = out.cpu()
    finite = ~torch.isnan(reference)
    assert torch.equal(torch.isnan(actual), torch.isnan(reference))
    assert torch.equal(actual.view(torch.int16)[finite], reference.view(torch.int16)[finite])
    for rows, columns, size in [(0, 8192, out.numel()*2), (256, 8191, out.numel()*2),
                                (256, 8192, out.numel()*2-1)]:
        assert lib.fuse_mxfp8_test_dequant(q.data_ptr(), s.data_ptr(), out.data_ptr(),
                                         rows, columns, size, stream) != 0
    assert lib.fuse_mxfp8_test_dequant(q.data_ptr(), s.data_ptr(), q.data_ptr(),
                                     256, 8192, out.numel()*2, stream) != 0
    print('all E4M3/E8M0 encodings + invalid workspace: PASS', flush=True)


def run_case(lib, op, world, causal, m=128, h=256, wgrad_policy=0):
    if op < 2 and wgrad_policy:
        raise ValueError('Forward correctness must not repeat WGrad policies')
    check(lib.fuse_mxfp8_test_set_wgrad_policy(wgrad_policy))
    qh, kvh, d = 16, 8, 128
    a, packed = qh*d, (qh+2*kvh)*d
    width = packed if op in (0, 2) else a
    wr, wc = (packed, h) if op in (0, 2) else (h, a)
    generator = torch.Generator().manual_seed(100 + op)

    def values(*shape):
        return (torch.randn(shape, generator=generator) * 0.08).bfloat16()

    payload, scales, effective = quantize(values(wr, wc))
    x = values(m*world, h if op in (0, 2, 3) else a)
    gradient = values(m*world, packed if op == 2 else h)
    qparts = [gradient[:, :a], gradient[:, a:a+kvh*d], gradient[:, a+kvh*d:]]
    attention = values(m*world, a)
    all_rows = [rows_for(r, m, world, causal) for r in range(world)]
    buffers, arguments, streams = [], [], []
    for rank in range(world):
        torch.cuda.set_device(rank)
        stream = torch.cuda.Stream(device=rank)
        streams.append(stream)
        rows = all_rows[rank]
        if op == 0:
            lhs, saved = x[rows], x[rows]
        elif op == 1:
            lhs = x[:, rank*(a//world):(rank+1)*(a//world)].contiguous()
            saved = x[rows]
        elif op == 2:
            lhs = qparts[0][:, rank*(a//world):(rank+1)*(a//world)].contiguous()
            saved = x[rows]
        else:
            lhs, saved = gradient[rows], attention[rows]
        b = dict(lhs=lhs.cuda(rank), saved=saved.cuda(rank), q=payload.cuda(rank),
                 s=scales.cuda(rank), workspace=torch.empty((wr, wc), device=rank, dtype=torch.bfloat16),
                 staging=torch.empty((m, width), device=rank, dtype=torch.bfloat16),
                 out=torch.empty((m, h) if op in (1, 2) else (m*world*width//world,),
                                 device=rank, dtype=torch.bfloat16),
                 dw=torch.empty((wr, wc), device=rank, dtype=torch.float32),
                 ready=torch.zeros(262144, device=rank, dtype=torch.int32),
                 done=torch.zeros(world*32, device=rank, dtype=torch.int32))
        if op == 2:
            for key, part in zip(('gk', 'gv'), qparts[1:]):
                b[key] = part[:, rank*(kvh*d//world):(rank+1)*(kvh*d//world)].contiguous().cuda(rank)
        buffers.append(b)
        arg = Arguments()
        arg.geometry[:] = [op, rank, world, m, h, qh, kvh, d, 1, int(causal), 1, 0, 0]
        arg.tensors[:] = [b[key].data_ptr() if key in b else 0 for key in
                          ('lhs', 'gk', 'gv', 'saved', 'q', 's', 'workspace', 'staging',
                           'out', 'dw', 'ready', 'done')] + [stream.cuda_stream]
        arg.alpha, arg.beta = 0.75, 0
        arguments.append(arg)
    for arg in arguments:
        arg.peer_data[:world] = [b['lhs' if op == 1 else 'staging' if op == 2 else 'out'].data_ptr()
                                 for b in buffers]
        arg.peer_ready[:world] = [b['ready'].data_ptr() for b in buffers]
        arg.peer_done[:world] = [b['done'].data_ptr() for b in buffers]
    if op == 0:
        # Legacy QKV forward is rank-major only; the new API must reject a
        # causal request before launching any peer-writing work.
        torch.cuda.set_device(0)
        arguments[0].geometry[9] = 1
        assert lib.fuse_mxfp8_test_launch(ct.byref(arguments[0])) != 0
        arguments[0].geometry[9] = 0
    if op < 2:
        torch.cuda.set_device(0)
        arguments[0].route_flags = 1 if op == 0 else 2
        assert lib.fuse_mxfp8_test_launch(ct.byref(arguments[0])) != 0
        arguments[0].route_flags = 0
    weight_configs = []
    if op >= 2:
        for rank, argument in enumerate(arguments):
            torch.cuda.set_device(rank)
            config = wgrad_config(lib, argument)
            assert config['policy_enum'] == (wgrad_policy or 1), config
            weight_configs.append(dict(rank=rank, config=config))

    def sync():
        for rank in range(world):
            torch.cuda.synchronize(rank)

    def reset():
        for rank, b in enumerate(buffers):
            with torch.cuda.device(rank):
                for key in ('workspace', 'staging', 'out'):
                    b[key].fill_(float('nan'))
                b['dw'].fill_(0.125)
                b['ready'].zero_()
                b['done'].zero_()
        sync()  # All poison operations finish before ANY peer can write.

    def launch(rank):
        with torch.cuda.device(rank):
            status = lib.fuse_mxfp8_test_launch(ct.byref(arguments[rank]))
            if status:
                raise RuntimeError(f'op={op} geometry={list(arguments[rank].geometry)} CUDA status={status}')

    def verify(beta, accumulated=1):
        for rank, b in enumerate(buffers):
            rows = all_rows[rank]
            close(b['workspace'], effective, 'effective weight')
            if op == 0:
                projected = (0.75 * x.float() @ effective.float().T).bfloat16()
                parts = projected.split([a, kvh*d, kvh*d], dim=1)
                expected = torch.cat([p[:, rank*(p.shape[1]//world):(rank+1)*(p.shape[1]//world)].flatten()
                                      for p in parts])
            elif op == 1:
                expected = (0.75 * x[rows].float() @ effective.float().T).bfloat16()
                close(b['staging'], x[rows], 'OProj F routed input')
            elif op == 2:
                expected = (0.75 * gradient[rows].float() @ effective.float()).bfloat16()
                close(b['staging'], gradient[rows], 'QKV B routed gradient')
            else:
                local = (0.75 * gradient.float() @ effective.float()).bfloat16()
                close(b['staging'], local[rows], 'OProj B local dA')
                expected = local[:, rank*(a//world):(rank+1)*(a//world)].flatten()
            close(b['out'], expected, f'op{op} rank{rank}')
            if op >= 2:
                saved = x[rows] if op == 2 else attention[rows]
                expected_dw = 0.75 * gradient[rows].float().T @ saved.float()
                close(b['dw'], expected_dw * accumulated + (0.125 if beta else 0), 'FP32 dW', fp32=True)

    # Warmup populates existing BF16 launch caches before graph capture.
    reset()
    if op >= 2:
        for arg in arguments:
            arg.geometry[11] = 1
    for rank in range(world):
        launch(rank)
    sync()
    if op >= 2:
        for rank, arg in enumerate(arguments):
            arg.geometry[12] = 1
            launch(rank)
            arg.geometry[11] = 0
            arg.geometry[12] = 0
        sync()
    verify(0)
    for graph_mode in (False, True):
        for deferred in ((False, True) if op >= 2 else (False,)):
            for arg in arguments:
                arg.geometry[11] = int(deferred)
                arg.beta = float(deferred)
            reset()
            graphs = []
            for rank in range(world):
                if graph_mode:
                    with torch.cuda.device(rank):
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=streams[rank]):
                            launch(rank)
                        graphs.append(graph)
            for replay in range(2):
                reset()
                for rank in range(world):
                    if graph_mode:
                        with torch.cuda.device(rank), torch.cuda.stream(streams[rank]):
                            graphs[rank].replay()
                    else:
                        launch(rank)
                sync()
                if deferred:
                    for rank, arg in enumerate(arguments):
                        assert torch.all(buffers[rank]['dw'] == 0.125), 'deferred B touched main_grad'
                        arg.geometry[12] = 1
                        if graph_mode:
                            with torch.cuda.device(rank):
                                weight_graph = torch.cuda.CUDAGraph()
                                with torch.cuda.graph(weight_graph, stream=streams[rank]):
                                    launch(rank)
                                with torch.cuda.stream(streams[rank]):
                                    weight_graph.replay()
                                    weight_graph.replay()
                                streams[rank].synchronize()
                        else:
                            launch(rank)
                            launch(rank)  # beta=1 twice from nonzero main_grad.
                        arg.geometry[12] = 0
                    sync()
                verify(deferred, 2 if deferred else 1)
            del graphs
    suffix = f' Wpolicy={wgrad_policy}' if op >= 2 else ''
    print(f'op={op} CP{world} T={m*world} causal={causal} M={m} H={h}{suffix}: PASS', flush=True)
    return dict(operator=('qkv_forward', 'oproj_forward', 'qkv_backward', 'oproj_backward')[op],
                cp=world, flattened_tokens=m*world, causal=causal, local_tokens=m, hidden=h,
                q_heads=qh, kv_heads=kvh, head_dim=d, passed=True,
                launches=['eager', 'graph'], graph_replays=2,
                backward_modes=['immediate', 'deferred'] if op >= 2 else [],
                nonzero_beta0_overwrite=op >= 2, nonzero_beta1_twice=op >= 2,
                requested_wgrad_policy=wgrad_policy if op >= 2 else None,
                weight_gemm_configs=weight_configs)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, default=ROOT/'build-mxfp8/libfuse_mxfp8_torch_bridge.so')
    parser.add_argument('--cps', default='4,8')
    parser.add_argument('--wgrad-policies', type=wgrad_policies, default='0',
                        help='backward cases run once per requested policy; forward cases are never repeated')
    parser.add_argument('--output', type=Path,
                        default=ROOT/'results/mxfp8_weight/validation/operator_correctness.json')
    return parser.parse_args(argv)


def main():
    args = arguments()
    worlds = list(map(int, args.cps.split(',')))
    if len(set(worlds)) != len(worlds) or not set(worlds) <= {4, 8}:
        raise ValueError('--cps must contain unique CP values 4 and/or 8')
    assert torch.cuda.device_count() >= max(worlds)
    lib = ct.CDLL(str(args.library))
    lib.fuse_mxfp8_test_launch.argtypes = [ct.POINTER(Arguments)]
    lib.fuse_mxfp8_test_dequant.argtypes = [ct.c_uint64]*3 + [ct.c_int]*2 + [ct.c_uint64]*2
    lib.fuse_mxfp8_test_set_wgrad_policy.argtypes = [ct.c_int32]
    lib.fuse_mxfp8_test_set_wgrad_policy.restype = ct.c_int
    lib.fuse_mxfp8_test_wgrad_config.argtypes = [ct.POINTER(Arguments), ct.POINTER(ct.c_int32)]
    lib.fuse_mxfp8_test_wgrad_config.restype = ct.c_int
    conversion_test(lib)
    policy_validation = wgrad_policy_test(lib, max(worlds))
    results = []
    for world in worlds:
        # torch's allocator enables peer access lazily; explicitly enable it
        # before the raw-pointer production launches touch peer allocations.
        cudart = ct.CDLL('libcudart.so.12')
        cudart.cudaDeviceEnablePeerAccess.argtypes = [ct.c_int, ct.c_uint]
        for rank in range(world):
            torch.cuda.set_device(rank)
            for peer in range(world):
                if rank == peer:
                    continue
                status = cudart.cudaDeviceEnablePeerAccess(peer, 0)
                assert status in (0, 704), status
                cudart.cudaGetLastError()
        for causal, m, h in ((False, 128, 256), (True, 256, 512)):
            for op in range(4):
                for policy in (args.wgrad_policies if op >= 2 else [0]):
                    results.append(run_case(lib, op, world, causal if op != 0 else False, m, h, policy))
    check(lib.fuse_mxfp8_test_set_wgrad_policy(0))
    sources = [Path(__file__), Path(__file__).with_name('operator_bridge.cu'),
               Path(__file__).with_name('operator_reference.cuh'),
               Path(__file__).with_name('operator_profile.cuh'),
               ROOT/'csrc/operators/ulysses_sm90.cu',
               *sorted((ROOT/'csrc/operators/ulysses_sm90').glob('**/*.cuh')),
               *sorted((ROOT/'include/fuse').glob('**/*.h')),
               *sorted((ROOT/'include/fuse').glob('**/*.cuh'))]
    report = dict(kind='production_operator_correctness_not_performance', passed=True,
                  conversion_encoding_pairs=65536, cases=results,
                  requested_wgrad_policies=args.wgrad_policies, wgrad_policy_validation=policy_validation,
                  library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                  sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  devices=[dict(index=i, name=torch.cuda.get_device_name(i),
                                cc=list(torch.cuda.get_device_capability(i)))
                           for i in range(max(worlds))])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print('MXFP8 SM90 production 2F2B: PASS', flush=True)


if __name__ == '__main__':
    main()
