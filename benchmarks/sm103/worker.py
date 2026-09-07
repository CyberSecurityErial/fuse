#!/usr/bin/env python3
"""Run one architecture-neutral boundary worker with SM103 preflight/provenance."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
WORKERS = {
    ("qkv", "cublaslt_nccl"): "sm90/QKVproj+a2a/te_nccl_baseline.py",
    ("oproj", "cublaslt_nccl"): "sm90/a2a+Oproj/te_nccl_baseline.py",
    ("qkv", "te_ub"): "sm90/QKVproj+a2a/te_userbuffers_qkv.py",
    ("oproj", "te_ub"): "sm90/a2a+Oproj/te_userbuffers_oproj.py",
}
UB_METHODS = (
    "configure_userbuffers_p2p", "userbuffers_p2p_send", "userbuffers_p2p_recv",
    "get_userbuffers_send_stream", "get_communication_stream", "get_buffer",
)


def canonical_oproj_pack():
    """Same timed UB pack pass/signature; only the sequence routing changes."""
    import triton
    import triton.language as tl

    @triton.jit
    def pack(source, packed, numel: tl.constexpr, seq: tl.constexpr,
             seq_local: tl.constexpr, chunk_tokens: tl.constexpr,
             k_local: tl.constexpr, world: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < numel
        k = offsets % k_local
        rows_per_peer = numel // (world * k_local)
        m = (offsets // k_local) % rows_per_peer
        target = offsets // (k_local * rows_per_peer)
        batch = m // seq_local
        local_m = m % seq_local
        first = local_m < chunk_tokens
        chunk = tl.where(first, target, 2 * world - target - 1)
        in_chunk = tl.where(first, local_m, local_m - chunk_tokens)
        source_token = batch * seq + chunk * chunk_tokens + in_chunk
        tl.store(packed + offsets,
                 tl.load(source + source_token * k_local + k, mask=mask), mask=mask)

    return pack


def configure_oproj_layout(boundary, direction, backend, layout):
    """Opt-in adapter for the imported worker, never edit the SM90 source.

    Legacy OProj passes TE's *before* indices into the *after* route, while
    legacy UB has another chunk order. Canonical mode uses TE's native inverse
    indices and the equivalent UB pack. No input conversion is hoisted out of
    timing; communication, receive unpack and all GEMMs remain untouched.
    """
    import measurement
    layout = measurement.oproj_layout(layout)
    if direction != 'oproj' or layout == 'legacy':
        return
    if backend == 'cublaslt_nccl':
        from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import (
            get_seq_chunk_ids_for_reordering_after_attn,
        )
        # The legacy caller's symbol name says "before", but this branch only
        # calls inverse A2A. Rebind in this module, not in the shared TE module.
        boundary.get_seq_chunk_ids_for_reordering_before_attn = (
            get_seq_chunk_ids_for_reordering_after_attn)
    elif backend == 'te_ub':
        boundary._pack_inverse_a2a_kernel = canonical_oproj_pack()
    else:
        raise ValueError(f'unknown OProj backend: {backend!r}')


def load_boundary(direction, backend, layout):
    path = ROOT / 'benchmarks' / WORKERS[direction, backend]
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('sm103_boundary', path)
    boundary = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = boundary
    spec.loader.exec_module(boundary)
    configure_oproj_layout(boundary, direction, backend, layout)
    return boundary


def record_result_layout(output, metadata):
    """Publish the executed contract before the final per-rank receipt."""
    if metadata['rank'] != 0 or metadata['direction'] != 'oproj':
        return
    data = json.loads(output.read_text())
    data['oproj_layout'] = metadata['oproj_layout']
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=("qkv", "oproj"), required=True)
    parser.add_argument("--backend", choices=("cublaslt_nccl", "te_ub"), required=True)
    parser.add_argument("--expected-sms", type=int, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args, remaining = parser.parse_known_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import measurement
    layout = measurement.oproj_layout()
    try:
        import torch
        import triton
        import transformer_engine
        import transformer_engine_torch as tex
    except ImportError as error:
        raise SystemExit(
            f"Missing benchmark dependency: {error}. Use a workspace-local Python "
            "environment containing CUDA-enabled torch, Triton and the adapted TE build."
        ) from error
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    device = torch.cuda.get_device_properties(rank)
    if (device.major, device.minor) != (10, 3):
        raise SystemExit(f"Expected CUDA Runtime SM103, got {device.major}.{device.minor}")
    if device.multi_processor_count != args.expected_sms:
        raise SystemExit("SM count differs from tuning plan; regenerate with --sm-count")
    if args.backend == "te_ub":
        cls = getattr(tex, "CommOverlapP2P", None)
        missing = [name for name in UB_METHODS if not hasattr(cls, name)]
        if missing:
            raise SystemExit(f"TE is missing the historical UB extensions: {missing}")
    versions = {}
    for package in ("torch", "triton", "transformer-engine", "transformer-engine-torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "source build (inspect module path)"
    metadata = {
        "schema": "sm103_baseline_v1", "rank": rank,
        "direction": args.direction, "oproj_layout": layout,
        "device": {"name": device.name, "compute_capability": "10.3",
                   "sm_count": device.multi_processor_count,
                   "total_memory": device.total_memory},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cublaslt_tune_launch": "graph" if os.environ.get("FUSE_CUBLASLT_TUNE_GRAPH") == "1" else "eager",
        "cuda": torch.version.cuda, "packages": versions,
        "te_module": transformer_engine.__file__, "tex_module": tex.__file__,
    }
    measurement.reset()
    if args.preflight_only:
        print(json.dumps(metadata, indent=2))
        return
    # Each torchrun subprocess gets its own import path; identical historical
    # module names from the two directories never coexist in one interpreter.
    worker = ROOT / "benchmarks" / WORKERS[(args.direction, args.backend)]
    sys.argv = [str(worker), *remaining]
    boundary = load_boundary(args.direction, args.backend, layout)
    boundary.main()
    output = Path(remaining[remaining.index("--json-out") + 1])
    record_result_layout(output, metadata)
    metadata.update(measurement_records=measurement.RECORDS, input_statistics=measurement.INPUTS)
    output.with_suffix(f".rank{rank}.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
