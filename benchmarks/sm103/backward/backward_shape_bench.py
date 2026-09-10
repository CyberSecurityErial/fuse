#!/usr/bin/env python3
"""BF16 backward geometry catalog, derived from the forward projection catalog.

For stored W[O,I] and token shard T=S/CP:
  forward: Y[T,O] = X[T,I] W^T
  dgrad:  dX[T,I] = dY[T,O] W       (NN, no materialized transpose)
  wgrad:  dW[O,I] = dY^T[O,T] X    (TN, beta=0)

CP defines the local GEMM geometry, not a measurement of distributed routing.
Wgrad here is a local partial gradient; any CP reduction is outside this probe.
The special-route TODO in projection_shapes.py applies equally to backward:
the adjoint of replicated segments requires reduction, not ordinary all-to-all.
"""
import argparse
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('sm103_backward_forward_catalog', HERE.parent / 'bench.py')
forward = importlib.util.module_from_spec(spec)
spec.loader.exec_module(forward)
projections = forward._projection_shapes
SEQUENCES = (16384, 65536, 131072, 262144, 524288)
CPS = (4, 8)


def projection_catalog():
    """Retain historical labels and all pinned production projection groups."""
    rows = []
    for direction in ('qkv', 'oproj'):
        for name, model in forward.load_shapes(direction).MODELS.items():
            if direction == 'qkv' and name in ('kimi_k3_kda', 'kimi_linear_48b_kda'):
                # Six shared-input linears, not the historical three-only QKV.
                continue
            rows.append(dict(model=name, projection=direction, direction=direction,
                input_width=model.hidden if direction == 'qkv' else model.attention_width,
                output_width=model.qkv_width if direction == 'qkv' else model.hidden,
                route='head_sequence', q_heads=model.q_heads,
                kv_heads=getattr(model, 'kv_heads', None), head_dim=model.head_dim))
    for name in projections.SOURCES:
        if name in ('qwen25_72b', 'qwen3_235b', 'bloom_176b'):
            continue  # Included with the historical labels above.
        for p in projections.grouped_projections(name):
            direction = 'oproj' if p['id'].endswith('_o') else 'qkv'
            # KDA O is already represented by its historical model label.
            if p['id'] == 'kda_o':
                continue
            rows.append(dict(model=name, projection=p['id'], direction=direction,
                input_width=p['k'], output_width=p['n'], route='segmented_todo',
                source=projections.SOURCES[name]))
    return rows


def cases(sequences=SEQUENCES, cps=CPS):
    if not sequences or len(set(sequences)) != len(sequences) or any(
            type(s) is not int or s < 16384 for s in sequences):
        raise ValueError('Production sequences must be unique integers >=16K')
    if not cps or len(set(cps)) != len(cps) or any(type(cp) is not int or cp not in CPS for cp in cps):
        raise ValueError('CP must be a unique selection of 4 and 8')
    rows = []
    for p in projection_catalog():
        for seq in sequences:
            for cp in cps:
                if seq % cp:
                    raise ValueError('Sequence must divide evenly across CP')
                t, i, o = seq // cp, p['input_width'], p['output_width']
                route_issue = ('segmented_routing_not_implemented' if p['route'] == 'segmented_todo'
                    else 'nondivisible_heads' if p['q_heads'] % cp or (
                        p['direction'] == 'qkv' and p['kv_heads'] % cp) else None)
                for phase, shape, layout in (('dgrad', (t, i, o), 'nn'), ('wgrad', (o, i, t), 'tn')):
                    rows.append(dict(p, id=f'{p["model"]}.{p["projection"]}.{phase}.s{seq}.cp{cp}',
                        phase=phase, seq=seq, cp=cp, m=shape[0], n=shape[1], k=shape[2],
                        operand_layout=layout, beta=0, accumulator_dtype='fp32', output_dtype='bf16',
                        fused_state='not_measured', route_issue=route_issue,
                        pure_gemm_state='pending', includes_cp_gradient_reduction=False))
    return rows


def export(output, sequences=SEQUENCES, cps=CPS, batch_size=256):
    if not 1 <= batch_size <= 256:
        raise ValueError('Batch size must be 1..256')
    rows = cases(sequences, cps)
    # Batch complete alias groups together, so equal geometry is measured once
    # per layout even when several models or S/CP pairs share it.
    batches = []
    for layout in ('nn', 'tn'):
        groups = {}
        for row in rows:
            if row['operand_layout'] == layout:
                groups.setdefault((row['m'], row['n'], row['k']), []).append(row)
        batch = []
        for group in groups.values():
            if batch and len(batch) + len(group) > batch_size:
                batches.append((layout, batch))
                batch = []
            if len(group) > batch_size:
                raise ValueError('Batch size cannot split an alias group')
            batch.extend(group)
        if batch:
            batches.append((layout, batch))
    output.mkdir(parents=True, exist_ok=False)
    files = []
    for index, (layout, batch) in enumerate(batches, 1):
        filename = f'{index:02d}-{layout}.json'
        payload = dict(schema='sm103_gemm_matrix_v1', shapes=[
            {key: row[key] for key in ('id', 'm', 'n', 'k')} for row in batch])
        (output / filename).write_text(json.dumps(payload, indent=2) + '\n')
        files.append(dict(file=filename, operand_layout=layout, logical_cases=len(batch),
            unique_geometries=len({(r['m'], r['n'], r['k']) for r in batch})))
    manifest = dict(schema='sm103_backward_catalog_v1', sequences=sequences, cps=cps,
        precision='bf16', launch='graph', warmup=10, samples=50, candidates=32,
        measurement='single_gpu_pure_cublaslt_not_distributed_backward',
        cases=rows, batches=files)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return dict(cases=len(rows), projection_groups=len(projection_catalog()), batches=len(files),
        unique_gemms=sum(x['unique_geometries'] for x in files), output=str(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seqs', default=','.join(map(str, SEQUENCES)))
    parser.add_argument('--cps', default='4,8')
    args = parser.parse_args()
    print(json.dumps(export(args.output, tuple(map(int, args.seqs.split(','))),
                            tuple(map(int, args.cps.split(',')))), ensure_ascii=False))
