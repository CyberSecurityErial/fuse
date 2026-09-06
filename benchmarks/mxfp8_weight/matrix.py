"""Reuse the published SM90 shape registries without copying their geometry."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def registry(direction):
    relative = {
        "gemm_a2a": "QKVproj+a2a/qkv_shape_bench.py",
        "a2a_gemm": "a2a+Oproj/oproj_shape_bench.py",
    }[direction]
    name = "mxfp8_legacy_" + direction
    spec = importlib.util.spec_from_file_location(name, ROOT / "benchmarks" / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def full_matrix():
    rows = []
    for direction in ("gemm_a2a", "a2a_gemm"):
        old = registry(direction)
        args = SimpleNamespace(models=",".join(old.DEFAULT_MODELS),
                               seqs=old.SEQUENCES, cps=old.CONTEXT_PARALLEL)
        for model, seq, cp in old.cases(args):
            rows.append(dict(
                id=f"{direction}/{model.name}/s{seq}/cp{cp}", direction=direction,
                model=model.name, global_seq=seq, cp=cp, batch=1,
                m=seq // cp,
                n=model.qkv_width if direction == "gemm_a2a" else model.hidden,
                k=model.hidden if direction == "gemm_a2a" else model.attention_width,
                hidden=model.hidden, q_heads=model.q_heads,
                kv_heads=getattr(model, "kv_heads", 0), head_dim=model.head_dim,
                visible_devices=old.VISIBLE_DEVICES[cp],
                layout="rank_major" if direction == "gemm_a2a" else "causal_paired",
                registry=str(Path(old.__file__).relative_to(ROOT)),
            ))
    assert len(rows) == 192, "Published matrix changed: audit before benchmarking"
    assert len({r["id"] for r in rows}) == len(rows)
    return rows
