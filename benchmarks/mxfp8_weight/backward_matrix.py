"""Backward coverage comes from the backward registry, never forward inference."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / 'benchmarks/backward/backward_shape_bench.py'


def legacy_registry():
    name = 'mxfp8_legacy_backward'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REGISTRY)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def full_matrix():
    old = legacy_registry()
    rows = []
    for operator, models in (('qkv', old.QKV_MODELS), ('oproj', old.OPROJ_MODELS)):
        for model in models.values():
            for seq in old.SEQUENCES:
                for cp in old.CONTEXT_PARALLEL:
                    m = seq // cp
                    projection = model.qkv_width if operator == 'qkv' else model.attention_width
                    wn, wk = ((projection, model.hidden) if operator == 'qkv'
                              else (model.hidden, projection))
                    rows.append(dict(
                        id=f'{operator}_backward/{model.name}/s{seq}/cp{cp}',
                        operator=operator, model=model.name, global_seq=seq,
                        cp=cp, batch=1, m=m, hidden=model.hidden,
                        q_heads=model.q_heads, kv_heads=model.kv_heads,
                        head_dim=model.head_dim, projection=projection,
                        weight_shape=[wn, wk], b_mnk=[m, wk, wn],
                        w_mnk=[wn, wk, m], layout='causal_paired',
                        visible_devices=old.VISIBLE_DEVICES[cp],
                        registry=str(REGISTRY.relative_to(ROOT))))
    assert len(rows) == 192 and len({r['id'] for r in rows}) == 192
    return rows


def expected_keys(grad_dtypes=('fp32',)):
    return {(r['id'], backend, launch, mode, dtype)
            for r in full_matrix()
            for backend in ('cublaslt_nccl', 'teub')
            for launch in ('eager', 'graph')
            for mode in ('immediate', 'deferred')
            for dtype in grad_dtypes}


if __name__ == '__main__':
    import json
    print(json.dumps(dict(settings=192, rows_per_grad_dtype=1536,
                         expected_rows=len(expected_keys()), cases=full_matrix()), indent=2))
