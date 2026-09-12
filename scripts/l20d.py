#!/usr/bin/env python3
"""Mac -> mc -> existing L20D_screen. Thin, durable task runner; stdlib only."""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from types import MappingProxyType
import uuid

REPO = Path(__file__).resolve().parents[1]
LOCAL = Path('/Users/admin/workspace/fuse_midfile/l20d')
WORKSPACE = Path('/root/workspace_wct')  # Legacy receipts retain their original location.
REMOTE = Path('/root/workspace_wct/fuse')
CONTROL = Path('/root/workspace_wct/.l20d')
PYTHON = '/root/workspace_wct/bench-env/bin/python'
CLOUD = 'arsenal-it-bucket/arsenal-it-bucket/wct/fuse'
# Jobs may select a node, never supply an arbitrary hostname or screen target.
NODES = MappingProxyType({
    '09': ('L20D_screen', 'l20d-xerkjfcp-0001'),
    '0a': ('L20D_screen2', 'l20d-ebed3kz6-0000'),
})
CUTLASS = '/root/workspace_wct/deps/cutlass-57e3cfb47a2d9e0d46eb6335c3dc411498efa198'
MPI_PREFIX = Path('/root/workspace_wct/toolchain/mpich-5.0.1.post1')
MPI_TRANSPORT = 'sm,self'
BLACKWELL_TILE_VARIANTS = ('m128n128k128', 'm128n256k64e32', 'm128n256k128e32')
QKV_POLICIES = ('auto', 'm128n64', 'm128n128', 'm128n160', 'm128n192', 'm128n256') + BLACKWELL_TILE_VARIANTS + ('m128n256k64e64',)
OPROJ_POLICIES = ('auto', 'm128n128', 'm128n256') + BLACKWELL_TILE_VARIANTS
FUSED_STAGES = ('fused-build', 'fused-smoke')
DONE = {'succeeded', 'failed'}
STAGES = ('doctor', 'build', 'te-build', 'ub-check', 'batch-check', 'matrix-check', 'cache-check', 'smoke', 'sweep', 'refine', 'formal', 'summary', 'overhead', 'baseline-replay', 'gemm-probe', 'gemm-cutlass-build', 'transport-probe', *FUSED_STAGES)
BASELINE_CASE_FIELDS = ('direction', 'model', 'seq', 'cp', 'hidden', 'q_heads', 'kv_heads',
                        'head_dim', 'm', 'n', 'k')
CUTLASS_COUNTER_RANGE = 'fuse_cutlass_1sm_counters'
CUTLASS_COUNTER_METRICS = (
    'gpu__time_duration.sum', 'dram__bytes_read.sum', 'lts__t_sectors_op_read.sum',
    'lts__t_sector_hit_rate.pct',
    'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed',
    'sm__cycles_elapsed.avg.per_second',
)
FUSED_COUNTER_METRICS = CUTLASS_COUNTER_METRICS + (
    'dram__bytes_write.sum', 'lts__t_sectors_op_write.sum',
    'lts__throughput.avg.pct_of_peak_sustained_elapsed',
    'nvlrx__bytes_data_user.sum', 'nvltx__bytes_data_user.sum',
)


def workspace_path(value):
    """Only a named user-owned workspace; never a home/root directory itself."""
    if not isinstance(value, str) or not re.fullmatch(
            r'/(?:root|home/[a-z_][a-z0-9_-]*)/workspace_wct', value):
        raise ValueError('Workspace must be /root/workspace_wct or /home/USER/workspace_wct')
    return Path(value)


def configure_workspace(value):
    global WORKSPACE, REMOTE, CONTROL, PYTHON, CUTLASS, MPI_PREFIX
    WORKSPACE = workspace_path(value)
    REMOTE, CONTROL = WORKSPACE / 'fuse', WORKSPACE / '.l20d'
    PYTHON = str(WORKSPACE / 'bench-env/bin/python')
    CUTLASS = str(WORKSPACE / 'deps/cutlass-57e3cfb47a2d9e0d46eb6335c3dc411498efa198')
    MPI_PREFIX = WORKSPACE / 'toolchain/mpich-5.0.1.post1'


def workspace_user():
    return 'root' if WORKSPACE.parent == Path('/root') else WORKSPACE.parent.name


def source_environment():
    return 'source ' + shlex.quote(str(WORKSPACE / 'env.sh'))


def command(argv, **kwargs):
    kwargs.setdefault('timeout', 60)
    return subprocess.run([str(a) for a in argv], check=True, **kwargs)


def read_command(argv):
    return command(argv, capture_output=True, text=True).stdout.strip()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def relay_progress(path, stop):
    """Stream concise progress locally; cloud heartbeat remains independent."""
    with Path(path).open(errors='replace') as reader:
        while True:
            line = reader.readline()
            if line:
                if line.startswith(('EXECUTOR ', 'BATCH ', 'RUN ', 'DONE ', 'RETRY ', 'BATCH_AB ', 'IMPORTED ',
                                    'config,', 'resolved_qkv,', 'device,', 'input,', 'correctness,', 'route,',
                                    'candidate,', 'auto_comm,', 'candidate_verified,', 'component_resources,', 'stage_time,',
                                    'validation_oracle,', 'validation_self_test,', 'validation_error,',
                                    'input_oracle,',
                                    'warmup,', 'sample,',
                                    'summary,', 'precision,', 'counter_epoch,', 'warning,', 'profile_resources,', 'profile_host,',
                                    'profile_dispatch,', 'PASS:', 'fused_bf16:', 'bf16 ',
                                    'BACKWARD ', 'backward_validation ', 'B: p50=', 'W: p50=')):
                    print(line.rstrip(), flush=True)
            elif stop.is_set():
                return
            else:
                stop.wait(.1)


def identifier(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}', value):
        raise ValueError(f'Invalid identifier: {value!r}')
    return value


def prefix(run_id):
    return f'{CLOUD}/transfers/automation/{identifier(run_id)}'


def mc_copy(src, dst, timeout=120):
    command(['mc', 'cp', src, dst], stdout=subprocess.DEVNULL, timeout=timeout)


def job_node(job):
    node = job.get('node', '09')  # Pre-node jobs/resume keep their original target.
    if node not in NODES:
        raise ValueError(f'Unknown node: {node!r}')
    return node


def fused_devices(job):
    devices = job.get('devices', '0,1,2,3,4,5,6,7').split(',')
    world = job.get('world', 8)
    if (world not in (4, 8) or len(devices) < world or
            len(set(devices)) != len(devices) or
            any(not re.fullmatch(r'[0-7]', device) for device in devices)):
        raise ValueError('Fused smoke requires world 4/8 and distinct physical GPU indices 0–7')
    return devices[:world]


def fused_policy_tile(policy):
    """Physical M/N/K, not a cluster aggregate; e32/e64 select epilogue N."""
    if policy not in QKV_POLICIES:
        raise ValueError(f'Unknown fused tile policy: {policy}')
    match = re.fullmatch(r'm128n(64|128|160|192|256)(?:k(64|128))?(?:e(?:32|64))?',
                         'm128n128' if policy == 'auto' else policy)
    return 128, int(match.group(1)), int(match.group(2) or 64)


def fused_candidates(job):
    """Explicit same-shape candidates; auto zero is resolved only by the C++ API."""
    auto = job.get('auto_oproj_comm', False)
    mxfp8_auto = job.get('auto_mxfp8_comm', False)
    if auto and (job.get('comm_sm_list') is not None or job.get('comm_sm', 0) not in (None, 0)):
        raise ValueError('--auto-oproj-comm excludes explicit --comm-sm/list')
    comm = ['0'] if auto else ([str(job.get('comm_sm', 8))] if job.get('comm_sm_list') is None else job['comm_sm_list'].split(','))
    zero_request = auto or (mxfp8_auto and job.get('comm_sm_list') is None and job.get('comm_sm', 0) in (None, 0))
    if mxfp8_auto and zero_request:
        comm = ['0']
    if not comm or any(not re.fullmatch(r'[0-9]+', value) or not (0 if zero_request else 1) <= int(value) <= 1024 for value in comm):
        raise ValueError('Communication CTA list must contain positive integers')
    comm = list(dict.fromkeys(int(value) for value in comm))
    if mxfp8_auto and 0 not in comm:
        comm.append(0)  # Unresolved production candidate, after explicit controls.
    def policies(direction, supported):
        single, multiple = direction + '_policy', direction + '_policy_list'
        values = [job.get(single, 'auto')] if job.get(multiple) is None else job[multiple].split(',')
        if not values or any(value not in supported for value in values):
            raise ValueError(f'Unknown {direction} policy in candidate list')
        return list(dict.fromkeys('m128n128' if value == 'auto' else value for value in values))

    qkv = policies('qkv', QKV_POLICIES)
    oproj = policies('oproj', OPROJ_POLICIES)
    if job.get('profile') and (len(comm) != 1 or len(qkv) != 1 or len(oproj) != 1):
        raise ValueError('Profiling requires one candidate; avoid collecting redundant full timelines')
    return comm, qkv, oproj


def fused_geometry(job):
    """Resolve the same integer/shape contract as fused_bf16, without CUDA."""
    limit = (1 << 31) - 1
    if job.get('quick') and (job.get('stage') != 'fused-smoke' or job.get('profile') or
            job.get('validation_self_test') or job.get('fused_counters') or job.get('oproj_gap_probe')):
        raise ValueError('--quick requires non-profile fused-smoke without diagnostics')
    direction = job.get('fused_direction', 'both')
    if direction not in ('both', 'qkv', 'oproj'):
        raise ValueError('Invalid --fused-direction')
    if direction != 'both' and (job.get('stage') != 'fused-smoke' or (job.get('profile') and not (job.get('backward') or job.get('mxfp8')))):
        raise ValueError('--fused-direction requires non-profile fused-smoke')
    if job.get('profile') and job.get('directions') == 'oproj':
        direction = 'oproj'

    def positive(value, name):
        if type(value) is not int or not 1 <= value <= limit:
            raise ValueError(f'{name} requires a positive int32 value')
        return value

    world = job.get('world', 8)
    if world not in (4, 8):
        raise ValueError('--world must be 4 or 8')
    seq_local, global_seq = job.get('seq_local'), job.get('global_seq')
    if seq_local is not None and global_seq is not None:
        raise ValueError('--seq-local and --global-seq are mutually exclusive')
    if global_seq is not None:
        global_seq = positive(global_seq, '--global-seq')
        if global_seq % world:
            raise ValueError('--global-seq must be divisible by --world')
        seq_local = global_seq // world
    else:
        seq_local = positive(256 if seq_local is None else seq_local, '--seq-local')
        global_seq = seq_local * world
    hidden = positive(job.get('hidden', 1024), '--hidden')
    q_heads = positive(job.get('q_heads', 32), '--q-heads')
    kv_heads = positive(job.get('kv_heads', 8), '--kv-heads')
    head_dim = positive(job.get('head_dim', 128), '--head-dim')
    timeout_seconds = positive(job.get('timeout_seconds', 60), '--timeout-seconds')
    q_width, kv_width = q_heads * head_dim, kv_heads * head_dim
    projection_width = q_width + 2 * kv_width
    if global_seq > limit or projection_width > limit:
        raise ValueError('Global sequence or packed projection width exceeds int32')
    if ((direction != 'oproj' and (q_heads % kv_heads or kv_heads % world)) or q_heads % world or
            head_dim % 8 or hidden % 8 or (q_width // world) % 64):
        raise ValueError('Requires GQA/CP-divisible heads, BF16 8-element alignment, '
                         'and OProj K shards divisible by 64')
    _, _, oproj_policies = fused_candidates(job)
    if job.get('auto_oproj_comm'):
        if (job.get('stage') != 'fused-smoke' or direction != 'oproj' or
                job.get('fused_launch') != 'graph' or not job.get('mpi') or not job.get('causal') or
                job.get('profile') or job.get('validation_self_test') or job.get('compute_only') or job.get('fused_counters') or
                job.get('oproj_comm_layout', 'rows') != 'rows' or job.get('max_swizzle_size', 1) not in (4, 8) or
                any(policy not in ('m128n256', 'm128n256k64e32') for policy in oproj_policies)):
            raise ValueError('Automatic OProj CTAs require non-profile Graph causal rows with explicit N256/e32 and sw4/8')
        if (seq_local % 256 or not 8192 <= q_width <= 16384 or
                fused_scheduler_geometry(seq_local, hidden, 256, job['max_swizzle_size'])[0] < 4):
            raise ValueError('Automatic OProj CTA geometry outside calibrated ready/K/swizzle domain')
    if any((q_width // world) % fused_policy_tile(policy)[2] for policy in oproj_policies):
        raise ValueError('OProj K shards must be divisible by every explicitly selected tile K')
    m_tiles = (seq_local + 127) // 128
    if (m_tiles * ((projection_width + 63) // 64) > limit or
            m_tiles * ((hidden + 127) // 128) > limit or m_tiles * world > limit):
        raise ValueError('Tile grid or profiling capacity exceeds int32')
    for width in (hidden, q_width, projection_width):
        if max(2 * seq_local * width, 2 * hidden * width) > (1 << 63) - 1:
            raise ValueError('Tensor bytes exceed addressable object size')
    if job.get('causal', False) and seq_local % 2:
        raise ValueError('--causal requires an even --seq-local')
    if (job.get('cpu_oracle') or job.get('validation_self_test')) and max(
            seq_local * hidden, seq_local * projection_width, seq_local * q_width) > 4194304:
        raise ValueError('CPU oracle/self-test is restricted to small diagnostic shapes')
    return dict(world=world, seq_local=seq_local, global_seq=global_seq, hidden=hidden,
                q_heads=q_heads, kv_heads=kv_heads, head_dim=head_dim, q_width=q_width,
                kv_width=kv_width, projection_width=projection_width, timeout_seconds=timeout_seconds)


def fused_device_memory(job):
    """Per-rank device estimate; not an OOM guarantee or a host-RAM check."""
    shape = fused_geometry(job)
    m, h, p, q = (shape[key] for key in ('seq_local', 'hidden', 'projection_width', 'q_width'))
    qkv = (job.get('fused_direction', 'both') != 'oproj' and
           not (job.get('profile') and job.get('directions') == 'oproj'))
    oproj = job.get('fused_direction', 'both') != 'qkv'
    # Count only selected directions, including their independent full references.
    buffer_bytes = 2 * (qkv * (m * h + 3 * m * p + h * p) +
                        oproj * (3 * m * q + 2 * m * h + h * q))
    if job.get('mxfp8'):
        # Packed FP8 A/W + independent BF16 decoded-oracle A/W + native scales.
        if oproj:
            buffer_bytes += 4 * m * q + 3 * h * q
            buffer_bytes += (((m * shape['world'] + 127) // 128) *
                             ((q // shape['world'] + 127) // 128) +
                             ((m + 127) // 128 + (h + 127) // 128) * ((q + 127) // 128)) * 512 + 2048
            buffer_bytes += (2 * ((h + 255) // 256) + ((m + 127) // 128) * 8) * 32 * 4
        else:
            buffer_bytes += 3 * (m * h + p * h)
            buffer_bytes += ((m + 127) // 128 + (p + 127) // 128) * ((h + 127) // 128) * 512 + 1024
            buffer_bytes += 2 * ((p + 255) // 256) * 32 * 4 + 512  # Per-call panel counters/ready.
    m_tiles = (m + 127) // 128
    flag_bytes = 4 * 32 * (qkv * (m_tiles * ((p + 63) // 64) + shape['world']) +
                          oproj * (m_tiles * shape['world'] + 1))
    # Calibration duplicates only the ready/done flags, never full tensors.
    calibration_flag_bytes = flag_bytes if (job.get('calibrate') or job.get('mxfp8_service_probe')) else 0
    profile_bytes = 0
    if job.get('profile', False):
        peer_capacity = max(m_tiles * ((h + 127) // 128), m_tiles * shape['world'])
        # Current peer events fit in 256 bytes; 1 MiB also covers CTA records.
        # Update this allowance if the profiling record/layout contract changes.
        profile_bytes = (0 if job.get('profile_detail') == 'cta' else peer_capacity * 256) + (1 << 20)
        if job.get('mxfp8'):
            # Mxfp8QuantRecord includes publication timestamps and the aggregated
            # chunk contribution (72 bytes); Mxfp8WaitRecord remains 24 bytes.
            quant_n, quant_k = (h, q) if not qkv else (p, h)
            profile_bytes += ((quant_n + 255) // 256) * (quant_k // 4) * 72 + m_tiles * ((quant_n + 255) // 256) * 24
            if job.get('mxfp8_service_probe'):
                profile_bytes += m_tiles * ((p + 255) // 256) * 48 + 148 * 24 + ((p + 255) // 256) * 16
        if job.get('oproj_pipeline_probe') or job.get('oproj_gap_probe'):
            profile_bytes += 192 << 20  # Bounded pipeline probe, conservative per-rank allowance.
    allocated_bytes = buffer_bytes + flag_bytes + calibration_flag_bytes + profile_bytes
    headroom_bytes = max(512 << 20, (allocated_bytes + 9) // 10)
    return dict(geometry=shape, buffer_bytes=buffer_bytes, flag_bytes=flag_bytes,
                calibration_flag_bytes=calibration_flag_bytes,
                profile_bytes=profile_bytes, headroom_bytes=headroom_bytes,
                minimum_free_bytes=max(2 << 30, allocated_bytes + headroom_bytes),
                guarantees_fit=False,
                note='Device estimate only; CUDA/cuBLAS overhead and host reference RAM remain runtime constraints')


def fused_scheduler_geometry(m, n, tile_n, max_swizzle_size):
    """Pinned CUTLASS static cluster-1 scheduler padding, not logical work."""
    m_tiles, n_tiles = (m + 127) // 128, (n + tile_n - 1) // tile_n
    minimum = min(m_tiles, n_tiles)
    effective = (8 if max_swizzle_size >= 8 and minimum >= 6 else
                 4 if max_swizzle_size >= 4 and minimum >= 3 else
                 2 if max_swizzle_size >= 2 and minimum >= 2 else 1)
    padded_m = ((m_tiles + effective - 1) // effective) * effective
    padded_n = ((n_tiles + effective - 1) // effective) * effective
    return effective, padded_m, padded_n, (m_tiles != padded_m or n_tiles != padded_n)


def validate_job(job, hostname=None):
    search = job.get('mxfp8_gemm_search', False)
    if search and (job['stage'] not in ('build', 'gemm-probe') or
            (job['stage'] == 'gemm-probe' and
             (job.get('gemm_precision') != 'mxfp8' or not job.get('gemm_sm_budget')))):
        raise ValueError('MXFP8 CUTLASS search requires build or MXFP8 gemm-probe with explicit compute budget')
    if job.get('workspace') is not None:
        workspace_path(job['workspace'])
    if job.get('gemm_precision', 'bf16') == 'mxfp8':
        if (job['stage'] != 'gemm-probe' or job.get('launches') != 'graph' or
                job.get('gemm_operand_layout', 'nt') != 'nt' or any(job.get(k) for k in
                ('compare_cutlass', 'cutlass_counters', 'cublaslt_counters')) or
                (job.get('gemm_sm_budget') and not search) or
                job.get('cublaslt_sm_target') is not None):
            raise ValueError('Pure MXFP8 requires Graph NT full-device GEMM without BF16 diagnostics')
    if job.get('mxfp8_prequantized') and not job.get('mxfp8'):
        raise ValueError('--mxfp8-prequantized requires --mxfp8')
    if job.get('mxfp8_weight_preparation') and not job.get('mxfp8'):
        raise ValueError('--mxfp8-weight-preparation requires --mxfp8')
    if job.get('mxfp8_epilogue_n') is not None and (
            not job.get('mxfp8') or job['mxfp8_epilogue_n'] not in (32, 64)):
        raise ValueError('--mxfp8-epilogue-n requires MXFP8 and 32 or 64')
    window = tuple(job.get(key, 0) for key in ('oproj_m_window_tiles', 'oproj_n_group_tiles'))
    if any(type(value) is not int or value < 0 or value > 2**31 - 1 for value in window):
        raise ValueError('OProj window dimensions must be nonnegative int32 values')
    if any(window) and (any(value == 0 or value & (value - 1) for value in window) or
            window[0] * window[1] > 2**31 - 1 or not job.get('mxfp8') or
            job['stage'] != 'fused-smoke' or job.get('fused_direction') != 'oproj' or
            job.get('auto_mxfp8_comm') or job.get('auto_oproj_comm') or
            (job.get('comm_sm') is None and job.get('comm_sm_list') is None)):
        raise ValueError('OProj windows require two positive power-of-two tile dimensions and explicit MXFP8 OProj communication CTAs')
    if type(job.get('auto_mxfp8_comm', False)) is not bool:
        raise ValueError('Automatic MXFP8 CTA selection must be a boolean')
    if type(job.get('mxfp8_service_probe', False)) is not bool:
        raise ValueError('MXFP8 service probe must be a boolean')
    if job.get('mxfp8_service_probe') and (
            not job.get('mxfp8') or job['stage'] != 'fused-smoke' or not job.get('profile') or
            job.get('mpi') or job.get('world', 8) not in (4, 8) or job.get('fused_launch', 'eager') != 'eager' or
            job.get('host_launch') != 'per_gpu_thread' or
            job.get('profile_detail', 'full') not in (None, 'full') or job.get('calibrate') or
            job.get('auto_mxfp8_comm') or job.get('auto_oproj_comm') or job.get('mxfp8_prequantized') or
            job.get('mxfp8_weight_preparation', 'comm') not in (None, 'comm') or
            job.get('qkv_raster') not in ('along_m', 'along_n')):
        raise ValueError('MXFP8 service probe requires single-process CP4/8 full profiling with explicit ordinary comm/layout')
    if job.get('auto_mxfp8_comm') and (
            not job.get('mxfp8') or job['stage'] != 'fused-smoke' or not job.get('mpi') or
            job.get('fused_launch') != 'graph' or job.get('profile') or job.get('auto_oproj_comm') or
            job.get('mxfp8_prequantized') or job.get('mxfp8_weight_preparation', 'comm') not in (None, 'comm') or
            job.get('oproj_raster' if job.get('fused_direction') == 'oproj' else 'qkv_raster') not in ('along_m', 'along_n')):
        raise ValueError('Automatic MXFP8 CTAs require MPI Graph dynamic-weight ordinary comm and explicit raster')
    if job.get('mxfp8'):
        if job['stage'] not in FUSED_STAGES or any(job.get(k) for k in (
                'backward', 'quick', 'compute_only', 'cpu_oracle',
                'validation_self_test', 'fused_counters', 'auto_oproj_comm', 'qkv_rank_swizzle')):
            raise ValueError('MXFP8 baseline requires isolated forward build/smoke without BF16 tuning/diagnostics')
        if job['stage'] == 'fused-smoke' and job.get('fused_direction') not in ('qkv', 'oproj'):
            raise ValueError('MXFP8 requires one forward direction')
        if job['stage'] == 'fused-smoke' and job.get('fused_direction') == 'oproj':
            if (job.get('mxfp8_prequantized') or
                    job.get('mxfp8_weight_preparation', 'comm') not in (None, 'comm') or
                    job.get('oproj_policy_list') != 'm128n256' or
                    job.get('oproj_comm_layout', 'rows') != 'rows'):
                raise ValueError('MXFP8 OProj requires explicit comm, m128n256/rows and no QKV diagnostics')
            shape = fused_geometry(job)
            if (shape['q_width'] // shape['world'] % 128 or shape['seq_local'] % 128 or
                    (job.get('causal') and shape['seq_local'] % 256)):
                raise ValueError('MXFP8 OProj requires K128 peer shards and complete M128 sequence chunks')
        if job.get('calibrate') and (job.get('mxfp8_prequantized') or
                job.get('mxfp8_weight_preparation', 'comm') not in (None, 'comm')):
            raise ValueError('MXFP8 C/Q/R calibration requires dynamic-weight ordinary communication warps')
        if job.get('profile') and (job.get('mpi') or job.get('mxfp8_prequantized') or
                job.get('oproj_gap_probe') or
                (job['stage'] == 'fused-smoke' and job.get('directions') != job.get('fused_direction'))):
            raise ValueError('MXFP8 profiling requires single-process dynamic weight, matching direction, no BF16 gap probe')
        if job.get('qkv_policy', 'auto') not in ('auto', 'm128n256') or job.get('qkv_policy_list') not in (None, 'm128n256'):
            raise ValueError('MXFP8 baseline uses the fixed M128/N256/K128 collective')
        if job.get('hidden', 1024) % 128:
            raise ValueError('MXFP8 baseline hidden dimension must be divisible by 128')
        if job.get('fused_direction') != 'oproj' and job.get('head_dim', 128) != 128:
            raise ValueError('MXFP8 communication-side quantization requires the head_dim=128 TMA route')
        if job.get('mxfp8_weight_preparation') not in (None, 'comm', 'all', 'comm_warp'):
            raise ValueError('MXFP8 weight preparation requires comm, all, or comm_warp')
    if job.get('backward_gemm_sweep') and not (job.get('backward') and job.get('profile') and job.get('mpi')):
        raise ValueError('Backward GEMM sweep requires isolated backward profile MPI build')
    if job.get('backward_matrix') or job.get('backward_matrix_payload'):
        if not job.get('backward') or not job.get('mpi') or job['stage']!='fused-smoke':
            raise ValueError('Backward matrix requires backward MPI measurement')
    if job.get('backward'):
        if job['stage'] not in FUSED_STAGES or any(job.get(k) for k in ('quick','calibrate','fused_counters','auto_qkv_comm','auto_oproj_comm','compute_only','producer_only')):
            raise ValueError('Backward baseline requires isolated build/smoke without forward tuning or diagnostics')
        if job.get('profile') and not job.get('backward_gemm_sweep') and (not job.get('mpi') or job.get('backward_matrix') or job.get('backward_matrix_payload')):
            raise ValueError('Backward role profiling requires one MPI case')
        if job['stage'] == 'fused-smoke' and not job.get('mpi') and job.get('world',8) != 8:
            raise ValueError('Shared backward smoke validates CP4 and CP8 with eight visible GPUs')
        if job['stage'] == 'fused-smoke' and job.get('mpi'):
            if job.get('fused_direction') not in ('qkv','oproj') or job.get('warmup',10)<10 or job.get('iterations',50)<50:
                raise ValueError('Backward MPI requires one direction and at least 10+50')
    node = job_node(job)
    if type(job.get('auto_oproj_comm', False)) is not bool:
        raise ValueError('Automatic OProj CTA selection must be a boolean')
    if job.get('auto_oproj_comm') and job['stage'] != 'fused-smoke':
        raise ValueError('Automatic OProj CTAs require fused-smoke')
    if job.get('compute_only') and (job['stage'] != 'fused-smoke' or not job.get('calibrate') or
            job.get('fused_direction') != 'oproj' or job.get('profile') or job.get('fused_counters')):
        raise ValueError('Compute-only requires non-profile OProj calibration')
    if job.get('oproj_gap_probe') and (job['stage'] != 'fused-smoke' or
            not job.get('profile') or job.get('mpi') or job.get('directions') != 'oproj' or
            job.get('profile_detail', 'full') not in (None, 'full') or job.get('oproj_pipeline_probe')):
        raise ValueError('OProj gaps require standalone full OProj profiling without K-stage probes')
    if job.get('cublaslt_counters') and (job['stage'] != 'gemm-probe' or
            job.get('compare_cutlass') or job.get('launches') != 'graph' or
            len(gemm_probe_shapes(job)) != 1):
        raise ValueError('cuBLASLt counters require one standalone Graph geometry')
    direction = job.get('fused_direction', 'both')
    if direction not in ('both', 'qkv', 'oproj') or (direction != 'both' and
            (job['stage'] != 'fused-smoke' or (job.get('profile') and not (job.get('backward') or job.get('mxfp8'))))):
        raise ValueError('--fused-direction requires non-profile fused-smoke and both/qkv/oproj')
    if job['stage'] not in STAGES:
        raise ValueError('Unknown stage')
    if job.get('oproj_comm_layout', 'rows') not in ('rows', 'columns'):
        raise ValueError('OProj communication layout must be rows or columns')
    if job.get('fused_counter_tool', 'ncu') not in ('ncu', 'nsys'):
        raise ValueError('Unknown fused counter tool')
    if job.get('fused_counter_replay', 'app-range') not in ('app-range', 'range', 'application'):
        raise ValueError('Fused counter replay must preserve concurrent kernels')
    if job.get('fused_counter_tool', 'ncu') != 'ncu' and not job.get('fused_counters'):
        raise ValueError('Explicit fused counter tool requires fused counters')
    if job.get('fused_counters') is not None:
        if (job['stage'] != 'fused-smoke' or job.get('profile') or
                not job.get('calibrate') or job.get('validation_self_test') or
                job.get('fused_launch', 'eager') != 'eager' or
                job.get('host_launch', 'sequential') != 'sequential' or
                job.get('directions', 'qkv') not in ('qkv', 'oproj') or
                job['fused_counters'] not in ('fused', 'compute_reference', 'copy_reference')):
            raise ValueError('Fused counters require single-process eager calibration and one direction/component')
        if any(len(items) != 1 for items in fused_candidates(job)):
            raise ValueError('Fused counters require one communication budget and one tile per direction')
        if job.get('mpi', False) != (job.get('fused_counter_replay', 'app-range') == 'application'):
            raise ValueError('Application replay requires MPI; range replay uses a single process')
        if job.get('mpi') and job.get('fused_counter_tool', 'ncu') != 'ncu':
            raise ValueError('MPI counter experiment uses NCU only')
    if not isinstance(job.get('mpi', False), bool):
        raise ValueError('MPI selection must be a boolean')
    if type(job.get('qkv_rank_swizzle', False)) is not bool:
        raise ValueError('QKV rank swizzle selection must be a boolean')
    if job.get('qkv_rank_swizzle') and (job['stage'] not in FUSED_STAGES or job.get('profile')):
        raise ValueError('QKV rank swizzle experiment requires a non-profile fused stage')
    launch = job.get('fused_launch', 'eager')
    if launch not in ('eager', 'graph'):
        raise ValueError('Fused launch must be eager or graph')
    if launch == 'graph' and (job['stage'] != 'fused-smoke' or not job.get('mpi') or (job.get('profile') and not job.get('backward'))):
        raise ValueError('Fused Graph requires fused-smoke --mpi without profiling')
    if job.get('mpi'):
        if job['stage'] not in FUSED_STAGES:
            raise ValueError('--mpi is only valid for fused-build/fused-smoke')
        if (job.get('profile') and not job.get('backward')) or job.get('host_launch_explicit') or job.get('host_launch', 'sequential') != 'sequential':
            raise ValueError('MPI does not support profile or host-launch overrides')
    if job.get('oproj_layout', 'legacy') not in ('legacy', 'causal_dual_chunk_v1'):
        raise ValueError('Unknown OProj baseline layout')
    if job.get('oproj_layout', 'legacy') != 'legacy' and job['stage'] not in (
            'baseline-replay', 'smoke', 'sweep', 'refine', 'formal', 'summary'):
        raise ValueError('OProj layout override requires a baseline measurement stage')
    if job['stage'] not in FUSED_STAGES and any(job.get(key) is not None for key in
                                             ('comm_sm_list', 'qkv_policy_list', 'oproj_policy_list')):
        raise ValueError('Fused candidate lists are only valid for fused stages')
    if job['stage'] not in FUSED_STAGES and job.get('oproj_policy', 'auto') != 'auto':
        raise ValueError('OProj fused tile is only valid for fused stages')
    if job['stage'] != 'fused-smoke' and (job.get('cpu_oracle') or job.get('validation_self_test')):
        raise ValueError('CPU oracle/self-test is only valid for fused-smoke')
    if job.get('input_generator', 'cpu_mt19937') not in ('cpu_mt19937', 'gpu_philox'):
        raise ValueError('Unknown fused input generator')
    if job.get('input_generator', 'cpu_mt19937') != 'cpu_mt19937' and job['stage'] != 'fused-smoke':
        raise ValueError('Fused input generator override is only valid for fused-smoke')
    if job.get('host_launch', 'sequential') not in ('sequential', 'per_gpu_thread'):
        raise ValueError('Unknown fused host launch mode')
    if job.get('host_launch', 'sequential') != 'sequential' and job['stage'] != 'fused-smoke':
        raise ValueError('Fused host launch override is only valid for fused-smoke')
    swizzle = job.get('max_swizzle_size', 1)
    if type(swizzle) is not int or swizzle not in (1, 2, 4, 8):
        raise ValueError('Fused max swizzle size must be 1, 2, 4 or 8')
    rasters = [job.get(direction + '_raster', 'heuristic') for direction in ('qkv', 'oproj')]
    if any(value not in ('heuristic', 'along_m', 'along_n') for value in rasters):
        raise ValueError('Unknown fused raster order')
    if (swizzle != 1 or any(value != 'heuristic' for value in rasters)) and job['stage'] != 'fused-smoke':
        raise ValueError('Fused scheduler overrides are only valid for fused-smoke')
    if job.get('calibrate') and (job['stage'] != 'fused-smoke' or
                                 job.get('profile') or job.get('validation_self_test')):
        raise ValueError('Calibration requires fused-smoke without profile or fault-injection self-test')
    if job.get('profile_detail') is not None:
        if job['profile_detail'] not in ('full', 'cta'):
            raise ValueError('Unknown profile detail')
        if job['stage'] != 'fused-smoke' or not job.get('profile'):
            raise ValueError('Explicit profile detail requires fused-smoke --profile')
    if type(job.get('oproj_pipeline_probe', False)) is not bool:
        raise ValueError('OProj pipeline probe selection must be a boolean')
    if job.get('oproj_pipeline_probe') and (job['stage'] != 'fused-smoke' or
            not job.get('profile') or (job.get('profile_detail') or 'full') != 'full' or
            job.get('directions') != 'oproj' or job.get('mpi') or job.get('qkv_epilogue_probe')):
        raise ValueError('OProj pipeline probe requires single-process full OProj profiling')
    if type(job.get('qkv_epilogue_probe', False)) is not bool:
        raise ValueError('QKV epilogue probe must be a boolean')
    if job.get('qkv_epilogue_probe'):
        _, qkv, _ = fused_candidates(job)
        if (job['stage'] != 'fused-smoke' or not job.get('profile') or
                job.get('profile_detail') != 'cta' or qkv != ['m128n256k64e32']):
            raise ValueError('QKV epilogue probe requires fused-smoke --profile --profile-detail cta and QKV m128n256k64e32')
    if job.get('rebuild') and job['stage'] != 'fused-build':
        raise ValueError('Explicit rebuild is only valid for fused-build')
    if job['stage'] == 'baseline-replay':
        if job.get('reuse_experiment') or job.get('profile') or job.get('executor', 'batch') != 'batch':
            raise ValueError('Baseline replay uses fixed winner jobs: no import, profile or serial search')
        if not job.get('baseline_replay') and not job.get('winners'):
            raise ValueError('Baseline replay requires --winners from a verified local summary')
        if job.get('baseline_replay'):
            validate_replay_payload(job['baseline_replay'], job.get('devices', '0,1,2,3,4,5,6,7'),
                                    REMOTE if hostname is not None else REPO)
    elif job.get('winners') or job.get('baseline_replay'):
        raise ValueError('--winners is only valid for baseline-replay')
    operand_layout = job.get('gemm_operand_layout', 'nt')
    gemm_candidates = job.get('gemm_candidates')
    if operand_layout not in ('nt', 'nn', 'tn') or (operand_layout != 'nt' and
            (job['stage'] != 'gemm-probe' or job.get('compare_cutlass') or
             job.get('cutlass_counters') or job.get('cublaslt_counters'))):
        raise ValueError('Backward operand views require standalone gemm-probe')
    if gemm_candidates is not None and (type(gemm_candidates) is not int or
            not 1 <= gemm_candidates <= 1024 or job['stage'] != 'gemm-probe'):
        raise ValueError('GEMM candidates require gemm-probe and 1..1024')
    if job['stage'] == 'gemm-probe':
        matrix = job.get('gemm_matrix') or job.get('gemm_matrix_payload')
        if job.get('profile') or (not matrix and job.get('directions') not in ('qkv', 'oproj')):
            raise ValueError('GEMM probe requires one direction qkv/oproj and no profiling')
        if job.get('launches') not in ('eager', 'graph', 'eager,graph'):
            raise ValueError('GEMM probe requires eager/graph launch selection')
        fused_devices(job)
        fused_geometry(job)
        if job.get('gemm_matrix_payload'):
            validate_gemm_matrix(job['gemm_matrix_payload'])
    elif job.get('gemm_matrix') or job.get('gemm_matrix_payload'):
        raise ValueError('--gemm-matrix is only valid for gemm-probe')
    lt_target = job.get('cublaslt_sm_target')
    if lt_target is not None and (type(lt_target) is not int or not 0 <= lt_target <= 148
            or job['stage'] != 'gemm-probe' or job.get('launches') != 'graph'
            or job.get('compare_cutlass')):
        raise ValueError('cuBLASLt SM target requires standalone Graph gemm-probe and a value in 0..148')
    gemm_budget = job.get('gemm_sm_budget')
    if gemm_budget is not None and (type(gemm_budget) is not int or not 1 <= gemm_budget <= 148
            or job['stage'] != 'gemm-probe' or job.get('launches') != 'graph'
            or job.get('compare_cutlass') or lt_target not in (None, gemm_budget)):
        raise ValueError('GEMM SM budget requires standalone Graph gemm-probe and matching target in 1..148')
    if type(job.get('compare_cutlass', False)) is not bool:
        raise ValueError('CUTLASS comparison selection must be a boolean')
    if job.get('compare_cutlass') and (job['stage'] != 'gemm-probe' or
            job.get('launches') != 'eager' or not (job.get('gemm_matrix') or job.get('gemm_matrix_payload'))):
        raise ValueError('CUTLASS comparison requires gemm-probe with an explicit matrix and eager only')
    cutlass_budget = job.get('cutlass_sm_budget')
    if cutlass_budget is not None and (type(cutlass_budget) is not int or
            not 1 <= cutlass_budget <= 148 or not job.get('compare_cutlass')):
        raise ValueError('CUTLASS SM budget requires comparison and a value in 1..148')
    if cutlass_budget and not job.get('cutlass_counters') and cutlass_budget % 2:
        raise ValueError('Three-way comparison includes cluster2 and needs an even SM budget')
    cutlass_swizzle = job.get('cutlass_swizzle_size')
    if cutlass_swizzle is not None and (type(cutlass_swizzle) is not int or
            cutlass_swizzle not in (1, 2, 4, 8) or not job.get('compare_cutlass')):
        raise ValueError('CUTLASS swizzle requires --compare-cutlass and a value of 1, 2, 4 or 8')
    cutlass_epilogue = job.get('cutlass_epilogue_n')
    if cutlass_epilogue is not None and (type(cutlass_epilogue) is not int or
            cutlass_epilogue not in (32, 64) or not job.get('compare_cutlass')):
        raise ValueError('CUTLASS epilogue N requires --compare-cutlass and a value of 32 or 64')
    cutlass_cluster = job.get('cutlass_1sm_cluster_m')
    if cutlass_cluster is not None and (type(cutlass_cluster) is not int or
            cutlass_cluster not in (1, 2) or not job.get('compare_cutlass')):
        raise ValueError('CUTLASS 1-SM cluster M requires --compare-cutlass and a value of 1 or 2')
    if cutlass_cluster == 2 and cutlass_epilogue not in (None, 32):
        raise ValueError('CUTLASS 1-SM cluster M=2 requires the implemented E32 collective')
    if type(job.get('cutlass_full_check', False)) is not bool:
        raise ValueError('CUTLASS full-check selection must be a boolean')
    if job.get('cutlass_full_check'):
        if not job.get('compare_cutlass'):
            raise ValueError('CUTLASS full-check requires the explicit pure-GEMM comparison path')
        matrix = job.get('gemm_matrix_payload')
        if matrix is not None and any(max(row['m'] * row['n'], row['m'] * row['k'], row['n'] * row['k']) > 4194304
                for row in validate_gemm_matrix(matrix)['shapes']):
            raise ValueError('CUTLASS full-check limits every tensor to 4194304 elements')
    if type(job.get('cutlass_counters', False)) is not bool:
        raise ValueError('CUTLASS counters selection must be a boolean')
    if job.get('cutlass_counters'):
        if not job.get('compare_cutlass'):
            raise ValueError('CUTLASS counters require the explicit pure-GEMM comparison path')
        matrix = job.get('gemm_matrix_payload')
        if matrix is not None and len(validate_gemm_matrix(matrix)['shapes']) != 1:
            raise ValueError('CUTLASS counters require exactly one GEMM geometry')
    if job['stage'] in FUSED_STAGES:
        fused_devices(job)
        shape = fused_geometry(job)
        _, qkv, oproj = fused_candidates(job)
        for width, policies in ((shape['projection_width'], qkv), (shape['hidden'], oproj)):
            for policy in policies:
                _, tile_n, _ = fused_policy_tile(policy)
                _, pm, pn, padded = fused_scheduler_geometry(shape['seq_local'], width, tile_n, swizzle)
                if pm * pn > (1 << 31) - 1:
                    raise ValueError('Padded fused scheduling grid exceeds int32')
                if job.get('profile') and not job.get('mxfp8_service_probe') and padded:
                    raise ValueError('Profile does not yet validate padded swizzle CTA ownership; use an unpadded geometry')
        if not (job.get('auto_oproj_comm') or job.get('auto_mxfp8_comm')) and not 1 <= job.get('comm_sm', 8) <= 1024:
            raise ValueError('Invalid --comm-sm')
        if job.get('qkv_policy', 'auto') not in QKV_POLICIES:
            raise ValueError('Unknown QKV policy')
        if job.get('oproj_policy', 'auto') not in OPROJ_POLICIES:
            raise ValueError('Unknown OProj policy')
    if hostname is not None and hostname != NODES[node][1]:
        raise RuntimeError(f'Wrong host for node {node}: expected {NODES[node][1]}, got {hostname}')
    return node


def baseline_executor(root):
    """Load the existing stdlib planner/executor; importing it launches no GPU."""
    spec = importlib.util.spec_from_file_location('_l20d_baseline_executor',
                                                  Path(root) / 'scripts/sm103_batch.py')
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def validate_replay_payload(payload, devices, root):
    """Validate embedded data only; never execute source commands or open Mac paths."""
    if (not isinstance(payload, dict) or payload.get('schema') != 'sm103_baseline_replay_input_v1' or
            payload.get('source_node') != '09' or not isinstance(payload.get('entries'), list) or
            not payload['entries'] or any(not isinstance(entry, dict) for entry in payload['entries'])):
        raise ValueError('Expected nonempty completed node09 sweep winners')
    for key, size in (('source_fingerprint', 16), ('source_plan_sha256', 64), ('winners_sha256', 64)):
        if not isinstance(payload.get(key), str) or not re.fullmatch('[0-9a-f]{' + str(size) + '}', payload[key]):
            raise ValueError(f'Invalid replay provenance: {key}')
    bench = baseline_executor(root).bench
    seen = set()
    for entry in payload['entries']:
        case, backend, launch, config = (entry.get(key) for key in ('case', 'backend', 'launch', 'config'))
        if not isinstance(case, dict) or set(case) != set(BASELINE_CASE_FIELDS):
            raise ValueError('Incomplete or unknown baseline case fields')
        if (case['direction'] not in ('qkv', 'oproj') or backend not in ('cublaslt_nccl', 'te_ub') or
                launch not in ('eager', 'graph') or
                not isinstance(case['model'], str) or case['cp'] not in (4, 8)):
            raise ValueError('Unknown baseline direction/backend/launch/model/CP')
        if any(type(case[key]) is not int or case[key] <= 0 for key in BASELINE_CASE_FIELDS[2:]):
            raise ValueError('Baseline geometry requires positive integers')
        fused_devices(dict(world=case['cp'], devices=devices))
        try:
            expected_case = next(bench.cases(argparse.Namespace(directions=case['direction'],
                models=case['model'], seqs=(case['seq'],), cps=(case['cp'],), devices=devices)))
        except (KeyError, ValueError, StopIteration) as error:
            raise ValueError('Unknown or invalid baseline case') from error
        if case != expected_case:
            raise ValueError('Baseline MNK/head geometry differs from its model label')
        group = bench.group_key(case, backend, launch)
        if entry.get('group') != group or group in seen:
            raise ValueError('Invalid or duplicate replay winner group')
        seen.add(group)
        allowed = {bench.digest(item) for item in bench.initial_configs(backend)}
        if not isinstance(config, dict) or bench.digest(config) not in allowed:
            raise ValueError('Replay config is not a supported original sweep candidate')
        origin = entry.get('source', {})
        if (not isinstance(origin, dict) or origin.get('selection_stage') != 'sweep' or origin.get('status') != 'complete' or
                origin.get('validated_candidates') != len(allowed) or
                origin.get('expected_candidates') != len(allowed) or
                origin.get('warmup', 0) < 10 or origin.get('samples', 0) < 50):
            raise ValueError('Pending or unvalidated groups cannot be replay winners')
        for key in ('p50_ms', 'p95_ms'):
            value = origin.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('Invalid source winner latency')
        if origin['p95_ms'] < origin['p50_ms']:
            raise ValueError('Invalid source winner percentiles')
        if not isinstance(origin.get('remote_raw'), str):
            raise ValueError('Invalid source raw provenance path')
        raw = PurePosixPath(origin['remote_raw'])
        if (not raw.is_absolute() or '..' in raw.parts or raw.parent.name != 'sweep' or
                raw.name != f'{group}_{bench.digest(config)}.json'):
            raise ValueError('Invalid source raw provenance path')
        hashes = origin.get('rank_sha256', {})
        if not isinstance(hashes, dict) or set(hashes) != {str(rank) for rank in range(case['cp'])} or any(
                not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value) for value in
                [origin.get('raw_sha256', ''), *hashes.values()]):
            raise ValueError('Source raw/all-rank provenance hashes are incomplete')
    return bench


def load_replay_winners(path, devices):
    """Recheck only requested complete groups, then embed portable data in job.json."""
    from summarize_sm103_sweep import load_plan, measurement_row
    path = Path(path).resolve()
    raw_winners = path.read_bytes()
    rows = json.loads(raw_winners)
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError('Expected a nonempty winners.json array')
    coverage = json.loads((path.parent / 'coverage.json').read_text())
    if coverage.get('schema') != 'sm103_partial_sweep_summary_v1':
        raise ValueError('Replay requires the verified complete-group summary provenance')
    results = Path(coverage['results']).resolve()
    plan, raw_plan = load_plan(results, coverage['fingerprint'])
    if hashlib.sha256(raw_plan).hexdigest() != coverage['plan_sha256']:
        raise ValueError('Source plan SHA differs from the verified summary')
    requested = [row.get('group') for row in rows]
    if any(not isinstance(group, str) for group in requested) or len(set(requested)) != len(requested):
        raise ValueError('Invalid or duplicate winner groups')
    jobs = {group: [] for group in requested}
    for item in plan['jobs']:
        if item['group'] in jobs:
            jobs[item['group']].append(item)
    entries = []
    for row in rows:
        group_jobs = jobs[row['group']]
        if not group_jobs:
            raise ValueError('Winner group does not exist in the original plan')
        measured = [measurement_row(results, plan, item) for item in group_jobs]
        best = min(measured, key=lambda item: (item['p50_ms'], item['config_json']))
        if row != best:
            raise ValueError('Supplied winner differs from the complete-group measured winner')
        raw = Path(best['raw'])
        entries.append(dict(group=best['group'], case={key: best[key] for key in BASELINE_CASE_FIELDS},
            backend=best['backend'], launch=best['launch'], config=json.loads(best['config_json']),
            source=dict(selection_stage='sweep', status='complete',
                independently_remeasured=False, expected_candidates=len(group_jobs),
                validated_candidates=len(group_jobs), warmup=best['warmup'], samples=best['samples'],
                p50_ms=best['p50_ms'], p95_ms=best['p95_ms'], remote_raw=best['remote_raw'],
                raw_sha256=sha(raw), rank_sha256={str(rank): sha(raw.with_suffix(f'.rank{rank}.json'))
                                               for rank in range(best['cp'])})))
    payload = dict(schema='sm103_baseline_replay_input_v1', source_node='09',
        source_fingerprint=plan['fingerprint'], source_plan_sha256=coverage['plan_sha256'],
        winners_sha256=hashlib.sha256(raw_winners).hexdigest(), entries=entries)
    validate_replay_payload(payload, devices, REPO)
    return payload


def baseline_replay_plan(job, env_id, results, folder):
    """Replay fixed configs on the selected node; never import source measurements."""
    node = job_node(job)
    bench = validate_replay_payload(job['baseline_replay'], job['devices'], REMOTE)
    count = max(entry['case']['cp'] for entry in job['baseline_replay']['entries'])
    selected = fused_devices(dict(world=count, devices=job['devices']))
    mapping = list(csv.reader(read_command(['nvidia-smi', '--id=' + ','.join(selected),
        '--query-gpu=index,uuid', '--format=csv,noheader,nounits']).splitlines()))
    by_index = {row[0].strip(): row[1].strip() for row in mapping if len(row) == 2}
    if set(by_index) != set(selected) or len(mapping) != len(selected) or any(
            not value.startswith('GPU-') for value in by_index.values()):
        raise RuntimeError('Replay did not resolve exactly the selected physical GPU UUIDs')
    visible = ','.join(by_index[index] for index in selected)
    write_json(folder / 'baseline-devices.json', dict(node=node, physical=selected, cuda_visible_devices=visible))
    args = argparse.Namespace(results=results, stage='baseline-replay', devices=visible,
        library=REMOTE / 'build/sm103/libfuse_cublaslt_runner.so', python=PYTHON, sm_count=148, te_root=None,
        oproj_layout=job.get('oproj_layout', 'legacy'))
    if not args.library.is_file():
        raise RuntimeError(f'Build the node {node} cuBLASLt baseline library before replay')
    measurement_fingerprint = bench.fingerprint(args)
    fingerprint = bench.digest(dict(measurement=measurement_fingerprint, node=node,
        source_id=job['source_id'], environment_fingerprint=env_id, oproj_layout=args.oproj_layout))
    # Fusion/controller-only edits must not force the baseline's GEMM heuristic
    # search again. Its own source/library/environment and UUID still bind cache.
    args.cache_namespace = bench.digest(dict(measurement=measurement_fingerprint,
        node=node, environment_fingerprint=env_id))
    entries = job['baseline_replay']['entries']
    jobs = [bench.make_job(args, entry['case'], entry['backend'], entry['launch'], entry['config'])
            for entry in entries]
    for item, entry in zip(jobs, entries):
        item['source_winner'] = entry['source']
    plan = dict(schema='sm103_baseline_v1', stage='baseline-replay', precision='bf16', sm_count=148,
        node=node, fingerprint=fingerprint, source_id=job['source_id'], environment_fingerprint=env_id,
        source_winners=job['baseline_replay'], imports_source_measurements=False,
        communication_search=False, local_gemm_cache_miss_may_tune=True,
        oproj_layout=args.oproj_layout,
        selection_note=('historical winner configs replayed on canonical layout; not a new-layout search'
                        if args.oproj_layout != 'legacy' else 'historical winner configs replayed unchanged'),
        jobs=jobs)
    path = results / 'replay_plan.json'
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError('Replay plan changed; use a new experiment, not old measurements')
    elif results.exists() and any(results.iterdir()):
        raise ValueError('Replay requires a new result experiment or its identical replay plan')
    else:
        write_json(path, plan)
    write_json(folder / 'baseline-replay.json', dict(node=node, plan=str(path), fingerprint=fingerprint,
        candidates=len(jobs), source_fingerprint=job['baseline_replay']['source_fingerprint']))
    return [PYTHON, '-u', str(REMOTE / 'scripts/sm103_batch.py'), '--plan', str(path),
            '--job-timeout', str(job['job_timeout'])]


def fused_telemetry(job, folder):
    # Use physical nvidia-smi IDs, not the UUID-valued CUDA_VISIBLE_DEVICES mask.
    return baseline_executor(REMOTE).telemetry(','.join(fused_devices(job)), folder / 'gpu-telemetry.csv')


def resolve_screen(node='09'):
    name = NODES[job_node({'node': node})][0]
    # macOS screen can return 1 even while successfully listing live sessions.
    listing = subprocess.run(['screen', '-ls'], capture_output=True, text=True, timeout=20).stdout
    matches = re.findall(r'^\s*(\d+\.' + re.escape(name) + r')\s', listing, re.MULTILINE)
    if len(matches) != 1:
        raise RuntimeError(f'Expected exactly one {name} session; found {len(matches)}. No command sent.')
    return matches[0]


def screen_snapshot(screen_id=None):
    screen_id = screen_id or resolve_screen()
    with tempfile.TemporaryDirectory(prefix='fuse-screen-') as tmp:
        out = Path(tmp) / 'screen.txt'
        command(['screen', '-S', screen_id, '-p', '0', '-X', 'hardcopy', out])
        # macOS screen can acknowledge -X before its hardcopy file appears.
        # Wait only for that one read-only request; never re-send a terminal
        # command or interpret a missing snapshot as an idle prompt.
        deadline = time.monotonic() + 2
        while True:
            try:
                snapshot = out.read_text(errors='replace')
                if snapshot:
                    return snapshot
            except FileNotFoundError:
                pass
            if time.monotonic() >= deadline:
                raise RuntimeError('Screen hardcopy did not become readable; no terminal command was sent')
            time.sleep(.02)


def screen_ready(node='09'):
    screen_id = resolve_screen(node)
    snapshot = screen_snapshot(screen_id)
    host = NODES[node][1]
    lines = [line.strip() for line in snapshot.splitlines() if line.strip()]
    user = workspace_user()
    ending = r'#' if user == 'root' else r'\$'
    if not lines or not re.fullmatch(re.escape(user + '@' + host) + r' .*' + ending, lines[-1]):
        raise RuntimeError('Screen is not at the expected idle cluster prompt. '
                           f'Inspect {screen_id} window 0; no command was sent.\n' +
                           '\n'.join(lines[-4:]))
    return screen_id


def source_package(directory):
    # Includes dirty tracked files and nonignored untracked source, not merely HEAD.
    names = command(['git', '-C', REPO, 'ls-files', '-z', '--cached', '--others',
                     '--exclude-standard'], capture_output=True).stdout.decode().split('\0')
    selected = []
    for name in sorted(set(names)):
        if not name or PurePosixPath(name).parts[0] in {'build', 'results', 'logs', '.l20d', '.git'}:
            continue
        path = REPO / name
        if not os.path.lexists(path) or any(p.is_symlink() for p in path.parents if p != REPO):
            continue
        if path.is_dir() and not path.is_symlink():
            continue
        if path.is_symlink() and not path.resolve().is_relative_to(REPO):
            raise ValueError(f'Source symlink escapes repository: {name}')
        selected.append((name, path))
    archive = directory / 'source.tar.gz'
    with tarfile.open(archive, 'w:gz', dereference=False) as tar:
        for name, path in selected:
            tar.add(path, arcname=name, recursive=False)
    # Hash the snapshot itself; concurrent local edits cannot desynchronize it.
    with tarfile.open(archive) as tar:
        manifest = {item.name: ('link:' + item.linkname if item.issym() else
                    hashlib.sha256(tar.extractfile(item).read()).hexdigest())
                    for item in tar.getmembers()}
    source_id = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return archive, source_id, manifest


def frozen_source_package(directory, run_id):
    """Reuse exact local source bytes while Mac development moves on.

    Only the source snapshot is inherited. Runtime options, receipts, controller
    and device checks remain those of the new job; this is not result reuse.
    """
    parent = LOCAL / identifier(run_id)
    job = json.loads((parent / 'job.json').read_text())
    archive = directory / 'source.tar.gz'
    shutil.copyfile(parent / 'source.tar.gz', archive)
    if sha(archive) != job['archive_sha256']:
        raise ValueError('Frozen source archive digest mismatch')
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        if any(not (item.isfile() or item.issym()) for item in members):
            raise ValueError('Unexpected frozen source member type')
        files = {item.name: ('link:' + item.linkname if item.issym() else
                 hashlib.sha256(tar.extractfile(item).read()).hexdigest()) for item in members}
    source_id = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if len(files) != len(members) or files != job['files'] or source_id != job['source_id']:
        raise ValueError('Frozen source manifest mismatch')
    return archive, source_id, files, job.get('git_head')


def submit(job):
    if job.get('workspace') is not None:
        configure_workspace(job['workspace'])
    node = validate_job(job)
    screen_id = screen_ready(node)
    dest = CONTROL / 'jobs' / job['run_id']
    cloud = prefix(job['run_id'])
    remote_command = (
        f'test "$(hostname)" = {shlex.quote(NODES[node][1])} && '
        f'test "$(id -un)" = {shlex.quote(workspace_user())} && mkdir -p {shlex.quote(str(dest))} && '
        f'timeout 60 mc cp {shlex.quote(cloud + "/runner.py")} {shlex.quote(str(dest / "runner.py"))} && '
        f'timeout 60 mc cp {shlex.quote(cloud + "/job.json")} {shlex.quote(str(dest / "job.json"))} && '
        f'{PYTHON} {shlex.quote(str(dest / "runner.py"))} remote {shlex.quote(str(dest / "job.json"))}'
    )
    command(['screen', '-S', screen_id, '-X', 'select', '0'])
    command(['screen', '-S', screen_id, '-p', '0', '-X', 'stuff', remote_command + '\r'])
    print(f"SUBMITTED {job['run_id']} ({job['stage']}, node {node})", flush=True)


def get_status(run_id):
    result = subprocess.run(['mc', 'cat', prefix(run_id) + '/status.json'], capture_output=True, timeout=20)
    if result.returncode:
        return {'run_id': run_id, 'state': 'awaiting-status'}
    return json.loads(result.stdout)


def fetch(run_id):
    started = time.perf_counter()
    state = get_status(run_id)
    if state['state'] not in DONE or 'artifact_sha256' not in state:
        raise RuntimeError('No completed artifact published yet; use status.')
    folder = LOCAL / run_id
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / 'artifacts.tar.gz'
    mc_copy(state['artifact'], out)
    if sha(out) != state['artifact_sha256']:
        raise RuntimeError('Artifact checksum mismatch; remote copy retained.')
    extracted = folder / f"artifacts-attempt{state.get('attempt', 1)}"
    if not extracted.exists():
        with tempfile.TemporaryDirectory(prefix='extract-', dir=folder) as temporary:
            staged = Path(temporary) / 'complete'
            unpack(out, staged)
            staged.rename(extracted)
    write_json(folder / 'fetched.json', state)
    timings_path = folder / 'timings.json'
    timings = json.loads(timings_path.read_text()) if timings_path.exists() else {}
    timings['fetch_and_extract_s'] = time.perf_counter() - started
    write_json(timings_path, timings)
    print(f'FETCHED {extracted}', flush=True)
    return state


def watch_interval(stage, elapsed):
    if elapsed < 15:
        return 2
    # A terminal receipt is uploaded immediately, independently of the 30 s
    # running heartbeat. Avoid a 30 s completion-discovery gap on short jobs.
    if stage in ('fused-build', 'fused-smoke', 'baseline-replay'):
        return 5 if elapsed < 120 else 15
    return 30  # Preserve the long sweep's existing observation cadence.


def wait_for_run(run_id, minimum_attempt=1):
    # Polling belongs to this small controller, not the agent. The remote screen
    # still receives every RUN/DONE line; stdout here contains only events.
    previous_receipt = None
    previous_event = None
    started = time.monotonic()
    changed_at = started
    folder = LOCAL / identifier(run_id) / 'watch'
    while True:
        try:
            state = get_status(run_id)
        except (subprocess.SubprocessError, OSError, ValueError):
            # A transient read failure is not a benchmark failure. Leave the last
            # receipt intact and let the bounded stale-heartbeat check report it.
            state = {'run_id': run_id, 'state': 'awaiting-status'}
        receipt = tuple(state.get(k) for k in ('state', 'phase', 'attempt', 'elapsed_s'))
        if state['state'] != 'awaiting-status' and receipt != previous_receipt:
            previous_receipt = receipt
            changed_at = time.monotonic()
            write_json(folder / 'latest.json', state)
        event = tuple(state.get(k) for k in ('state', 'phase', 'attempt'))
        if event != previous_event and state['state'] != 'awaiting-status':
            summary = {k: state.get(k) for k in ('run_id', 'state', 'phase', 'attempt', 'elapsed_s', 'exit_code')}
            print(json.dumps(summary, ensure_ascii=False), flush=True)
            with (folder / 'events.jsonl').open('a') as stream:
                stream.write(json.dumps(summary, ensure_ascii=False) + '\n')
            previous_event = event
        if time.monotonic() - changed_at > 180:
            raise RuntimeError('Remote heartbeat is stale for 180 seconds. '
                               'No task was resubmitted or terminated. '
                               f'Last local receipt: {folder / "latest.json"}')
        # A transient read has no attempt field. Its timeout is measured from
        # the last changed receipt above, not from the start of a long job.
        if state['state'] == 'awaiting-status':
            pass
        elif state.get('attempt', 0) < minimum_attempt:
            if time.monotonic() - started > 180:
                raise RuntimeError('No new remote receipt after 180 seconds. '
                                   'No screen read, resubmission, or termination was attempted.')
        elif state['state'] in DONE:
            if 'artifact_sha256' in state:
                fetch(run_id)
            return 0 if state['state'] == 'succeeded' else 1
        time.sleep(watch_interval(state.get('stage'), time.monotonic() - started))


def unpack(archive, target):
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        for item in tar.getmembers():
            p = PurePosixPath(item.name)
            if p.is_absolute() or '..' in p.parts or not (item.isfile() or item.isdir() or item.issym()):
                raise ValueError(f'Unsafe archive member: {item.name}')
            if not (target / item.name).resolve().is_relative_to(target.resolve()):
                raise ValueError(f'Archive path escapes destination: {item.name}')
            if item.issym():
                link = PurePosixPath(item.linkname)
                if link.is_absolute() or not (target / p.parent / item.linkname).resolve().is_relative_to(target.resolve()):
                    raise ValueError(f'Unsafe archive link: {item.name}')
            tar.extract(item, target)


def install_source(job, folder):
    archive = folder / 'source.tar.gz'
    mc_copy(prefix(job['run_id']) + '/source.tar.gz', archive)
    if sha(archive) != job['archive_sha256']:
        raise RuntimeError('Source checksum mismatch')
    stage = folder / 'source'
    if not stage.exists():
        unpack(archive, stage)
    backup = folder / 'source-before'
    # Preflight the entire managed file set before replacing any source file.
    for name in job['files']:
        src, dst = stage / name, REMOTE / name
        actual = 'link:' + os.readlink(src) if src.is_symlink() else sha(src)
        if actual != job['files'][name]:
            raise RuntimeError(f'Staged source does not match manifest: {name}')
        if not dst.parent.resolve().is_relative_to(REMOTE.resolve()):
            raise RuntimeError(f'Remote parent escapes repository: {name}')
        if dst.is_dir() and not dst.is_symlink():
            raise RuntimeError(f'Remote directory conflicts with file/link: {name}; explicit migration required')
    for name in job['files']:
        src, dst = stage / name, REMOTE / name
        if os.path.lexists(dst):
            if src.is_symlink() and dst.is_symlink() and os.readlink(src) == os.readlink(dst):
                continue
            if not src.is_symlink() and not dst.is_symlink() and sha(src) == sha(dst):
                continue
            old = backup / name
            old.parent.mkdir(parents=True, exist_ok=True)
            if not os.path.lexists(old):
                shutil.copy2(dst, old, follow_symlinks=False)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_symlink():
            if os.path.lexists(dst):
                dst.unlink()  # Exact managed source path, backed up above.
            dst.symlink_to(os.readlink(src))
        else:
            # Replacing a symlink must not follow it and overwrite another file.
            fd, temp_name = tempfile.mkstemp(prefix='.l20d-', dir=dst.parent)
            os.close(fd)
            temp = Path(temp_name)
            shutil.copy2(src, temp)
            # Snapshot mtimes can precede existing build outputs (or be zero).
            # Content changed, so timestamp-based Ninja must see it as new.
            # Unchanged files above retain their mtime and incremental cache.
            os.utime(temp, None)
            temp.replace(dst)
    write_json(folder / 'source-installed.json', {'source_id': job['source_id'], 'files': job['files']})


def environment_receipt(require_te=True):
    packages = {}
    for name in ('torch', 'triton', 'transformer-engine', 'pydantic', 'importlib-metadata'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = 'missing'
    libraries = {}
    loaded = {}
    if require_te:
        loaded = json.loads(read_command([PYTHON, '-c',
            'import json,transformer_engine,transformer_engine.pytorch,transformer_engine_torch; '
            'print(json.dumps({"te":transformer_engine.__file__,"tex":transformer_engine_torch.__file__}))']))
        libraries[str(Path(loaded['tex']).resolve())] = sha(loaded['tex'])
        for site in (Path(PYTHON).parent.parent / 'lib/python3.12/site-packages',):
            for lib in (site / 'transformer_engine').glob('*.so'):
                libraries[str(lib.resolve())] = sha(lib)
    tools = {}
    for name, path in [('nvcc', '/usr/local/cuda/bin/nvcc'), ('ptxas', '/usr/local/cuda/bin/ptxas')]:
        tools[name] = read_command([path, '--version'])
    tools['gcc'] = read_command(['bash', '-c',
        source_environment() + ' && command -v gcc && gcc --version'])
    tools['driver'] = read_command(['nvidia-smi', '--id=0', '--query-gpu=driver_version',
                                   '--format=csv,noheader'])
    overrides = {'TRITON_PTXAS_PATH': '/usr/local/cuda/bin/ptxas', 'UB_SKIPMC': '1'}
    headers = WORKSPACE / 'deps/python-headers-3.12.7'
    if headers.is_dir():
        overrides['CPATH'] = str(headers)
    return dict(host=socket.gethostname(), python=sys.version, executable=sys.executable,
                workspace=str(WORKSPACE), te_required=require_te,
                packages=packages, libraries=libraries, modules=loaded, compilers=tools,
                overrides=overrides)


def fused_build_dir(job):
    name = ('sm103-fused-mpi' if job.get('mpi') else
            'sm103-fused-profile' if job.get('profile', False) else 'sm103-fused')
    if job.get('qkv_rank_swizzle'):
        name += '-rank-swizzle'
    if job.get('mxfp8'):
        name += '-mxfp8'
    if job.get('backward'):
        name += '-backward'
        if job.get('mpi') and job.get('profile'):
            name += '-profile'
    return REMOTE / 'build' / name


def fused_binary(job):
    if job.get('mxfp8'):
        return fused_build_dir(job) / ('fused_mxfp8_mpi' if job.get('mpi') else 'fused_mxfp8')
    if job.get('backward'):
        return fused_build_dir(job) / ('backward_mpi_bench' if job.get('mpi') else 'backward_smoke')
    return fused_build_dir(job) / ('fused_bf16_mpi' if job.get('mpi') else 'fused_bf16')


def mpi_toolchain_receipt():
    """Workspace-only MPI identity; no installation or global environment edits."""
    compiler = read_command(['bash', '-c',
        source_environment() + ' && command -v g++ && g++ -dumpfullversion']).splitlines()
    if len(compiler) != 2 or not compiler[0].startswith('/') or compiler[1].split('.')[0] != '12':
        raise RuntimeError('MPI requires the workspace environment GCC 12 C++ compiler')
    required = [MPI_PREFIX / name for name in ('include/mpi.h', 'bin/mpicxx', 'bin/mpiexec', 'bin/hydra_pmi_proxy')]
    if any(not path.is_file() for path in required):
        raise RuntimeError('Workspace MPI development prefix is incomplete; run its explicit setup first')
    if any(not os.access(path, os.X_OK) for path in required[1:]):
        raise RuntimeError('Workspace MPI tools are not executable')
    libraries = sorted({path.resolve() for path in MPI_PREFIX.rglob('*.so*') if path.is_file()})
    if not any(path.name.startswith('libmpi.so') for path in libraries):
        raise RuntimeError('Workspace libmpi is missing')
    identities = {}
    for path in required + libraries:
        if not path.resolve().is_relative_to(MPI_PREFIX.resolve()):
            raise RuntimeError(f'MPI tool/library escapes the private prefix: {path}')
        identities[str(path)] = sha(path)
    return dict(prefix=str(MPI_PREFIX), files=identities, cxx=compiler[0], cxx_version=compiler[1],
                overrides={'MPICH_CXX': compiler[0], 'UCX_TLS': MPI_TRANSPORT},
                transport='same_host_cpu_control_only', cuda_ipc='explicit_harness_memory_handles')


def fused_build_inputs(job):
    # Documentation/controller edits do not invalidate an otherwise current binary.
    selected = {name: digest for name, digest in job['files'].items()
                if name in ('CMakeLists.txt', 'benchmarks/sm103/fused_bf16.cu') or
                (job.get('backward') and name.startswith('benchmarks/sm90/backward/') and name.endswith('.cu')) or
                (job.get('backward') and name.startswith('benchmarks/sm103/backward/') and name.endswith('.cuh')) or
                (name.startswith('benchmarks/sm103/fused_') and
                 PurePosixPath(name).suffix in ('.h', '.hpp', '.cuh')) or
                (name.startswith(('cmake/', 'include/', 'csrc/operators/sm103/')) and
                 PurePosixPath(name).suffix in ('.cmake', '.h', '.hpp', '.cuh', '.cu', '.cpp', '.inl'))}
    return hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()


def fused_argv(job):
    validate_job(job)
    build = fused_build_dir(job)
    if job['stage'] == 'fused-build':
        configure = ['cmake', '-S', str(REMOTE), '-B', str(build), '-G', 'Ninja',
                     '-DFUSE_ARCH=sm103', '-DFUSE_BUILD_KERNELS=ON', '-DFUSE_BUILD_BASELINES=OFF',
                     '-DFUSE_ENABLE_PROFILING=' + ('ON' if job.get('profile', False) else 'OFF'),
                     '-DFUSE_SM103_QKV_RANK_SWIZZLE=' + ('ON' if job.get('qkv_rank_swizzle') else 'OFF'),
                     '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc',
                     '-DCUTLASS_ROOT=' + CUTLASS]
        if job.get('mpi'):
            configure += ['-DFUSE_BUILD_MPI_BENCH=ON', '-DMPI_CXX_COMPILER=' + str(MPI_PREFIX / 'bin/mpicxx'),
                          '-DMPIEXEC_EXECUTABLE=' + str(MPI_PREFIX / 'bin/mpiexec')]
        compile_command = ['cmake', '--build', str(build), '--target', fused_binary(job).name, '--parallel', '4']
        if job.get('rebuild'):
            compile_command.append('--clean-first')
        return ['bash', '-c', shlex.join(configure) + ' && exec ' + shlex.join(compile_command)]
    if job['stage'] != 'fused-smoke':
        raise ValueError('Expected a fused stage')
    if job.get('backward'):
        if not job.get('mpi'):
            return [str(fused_binary(job))]
        s = fused_geometry(job)
        argv = [str(fused_binary(job)), '--operator', job['fused_direction'],
                '--m', str(s['seq_local']), '--hidden', str(s['hidden']),
                '--q-heads', str(s['q_heads']), '--kv-heads', str(s['kv_heads']),
                '--head-dim', str(s['head_dim']), '--comm-ctas', str(job.get('comm_sm') or 16),
                '--gemm-policy', 'm128n128', '--weight-mode', 'deferred', '--weight-beta', '0',
                '--launch', job.get('fused_launch','eager'), '--warmup', str(job.get('warmup',10)),
                '--iterations', str(job.get('iterations',50)), '--check']
        if job.get('causal'): argv.append('--causal-load-balanced')
        return argv
    shape = fused_geometry(job)
    comm, qkv, oproj = fused_candidates(job)
    argv = [str(fused_binary(job)), '--world', str(shape['world'])]
    if job.get('mxfp8_prequantized'):
        argv.append('--mxfp8-prequantized')
    if job.get('mxfp8_weight_preparation'):
        argv += ['--mxfp8-weight-preparation', job['mxfp8_weight_preparation']]
    if job.get('mxfp8_epilogue_n') is not None:
        argv += ['--mxfp8-epilogue-n', str(job['mxfp8_epilogue_n'])]
    if job.get('mxfp8_service_probe'):
        argv.append('--mxfp8-service-probe')
    if job.get('quick'):
        argv += ['--quick']
    if job.get('fused_direction', 'both') != 'both':
        argv += ['--fused-direction', job['fused_direction']]
    if job.get('fused_launch', 'eager') != 'eager':
        argv += ['--launch', job['fused_launch']]
    if job.get('auto_oproj_comm'):
        argv.append('--auto-oproj-comm')
    elif job.get('auto_mxfp8_comm'):
        argv.append('--auto-mxfp8-comm')
        manual = [value for value in comm if value > 0]
        if manual:
            argv += ['--comm-sm-list', ','.join(map(str, manual))]
    elif job.get('comm_sm_list') is not None:
        argv += ['--comm-sm-list', ','.join(map(str, comm))]
    else:
        argv += ['--comm-sm', str(comm[0])]
    for direction, policies in (('qkv', qkv), ('oproj', oproj)):
        if job.get(direction + '_policy_list') is not None:
            argv += ['--' + direction + '-policy-list', ','.join(policies)]
    sequence_key = 'global_seq' if job.get('global_seq') is not None else 'seq_local'
    argv += ['--' + sequence_key.replace('_', '-'), str(shape[sequence_key])]
    for key in ('hidden', 'q_heads', 'kv_heads', 'head_dim', 'timeout_seconds'):
        argv += ['--' + key.replace('_', '-'), str(shape[key])]
    if job.get('causal', False):
        argv.append('--causal')
    if job.get('profile', False):
        argv.append('--profile')
        if job.get('directions') in ('qkv', 'oproj'):
            argv += ['--profile-direction', job['directions']]
        if job.get('profile_detail') is not None:
            argv += ['--profile-detail', job['profile_detail']]
    for key in ('cpu_oracle', 'validation_self_test', 'calibrate', 'compute_only', 'qkv_epilogue_probe', 'oproj_pipeline_probe', 'oproj_gap_probe'):
        if job.get(key):
            argv.append('--' + key.replace('_', '-'))
    if job.get('fused_counters'):
        argv += ['--counter-component', job['fused_counters'],
                 '--counter-direction', job.get('directions', 'qkv')]
    for key, default in (('max_swizzle_size', 1), ('qkv_raster', 'heuristic'), ('oproj_raster', 'heuristic'),
                         ('oproj_comm_layout', 'rows'), ('oproj_m_window_tiles', 0), ('oproj_n_group_tiles', 0)):
        if job.get(key, default) != default:
            argv += ['--' + key.replace('_', '-'), str(job[key])]
    if job.get('input_generator', 'cpu_mt19937') != 'cpu_mt19937':
        argv += ['--input-generator', job['input_generator']]
    if job.get('host_launch', 'sequential') != 'sequential':
        argv += ['--host-launch', job['host_launch']]
    return argv


def fused_build_receipt(job, env_id):
    binary = fused_binary(job)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f'Missing fused executable: {binary}; run fused-build first')
    # Inspect this task's executable under the same workspace-local environment
    # as the benchmark. In particular, record where a direct libcuda link resolves;
    # a successful link against a toolkit stub is not runtime-driver validation.
    dependencies = read_command(['bash', '-c',
        source_environment() + ' && exec ldd "$1"', 'l20d-link-audit', str(binary)])
    # Loader addresses are randomized between inspections; compare resolutions,
    # not ASLR virtual addresses, when validating an existing build receipt.
    dependencies = re.sub(r' \(0x[0-9a-fA-F]+\)', '', dependencies)
    recorded = dict(node=job_node(job), profile=job.get('profile', False), binary=str(binary),
                binary_sha256=sha(binary), build_inputs=fused_build_inputs(job),
                environment_fingerprint=env_id, dynamic_dependencies=dependencies)
    if job.get('qkv_rank_swizzle'):
        recorded['qkv_rank_swizzle'] = 'rank_n_band_v1'
    if job.get('mpi'):
        if 'not found' in dependencies or 'libmpi.so' not in dependencies:
            raise RuntimeError('MPI executable has unresolved or absent libmpi linkage')
        for line in dependencies.splitlines():
            if any(name in line for name in ('libmpi.', 'libucp.', 'libucs.', 'libuct.', 'libucm.', 'libfabric.')):
                match = re.search(r'=>\s+(/\S+)', line)
                if not match or not Path(match[1]).resolve().is_relative_to(MPI_PREFIX.resolve()):
                    raise RuntimeError('MPI/UCX/OFI dependency did not resolve inside the private prefix')
        recorded.update(mpi=True, mpi_toolchain=mpi_toolchain_receipt())
    return recorded


def check_fused_build(job, env_id, folder):
    path = fused_build_dir(job) / '.l20d-build.json'
    if not path.is_file():
        raise RuntimeError('No verified fused-build receipt; run fused-build first (with the same --profile setting)')
    recorded = json.loads(path.read_text())
    if recorded != fused_build_receipt(job, env_id):
        raise RuntimeError('Fused build inputs/environment/binary changed; run incremental fused-build first')
    write_json(folder / 'fused-build.json', recorded)


def mpi_rank_logs(folder, attempt, world):
    return [dict(rank=rank, **{stream: Path(folder) / f'mpi-attempt{attempt}-rank-{rank}.{stream}.log'
                              for stream in ('stdout', 'stderr')}) for rank in range(world)]


def mpi_launch_argv(job, argv, folder, attempt):
    """Hydra fork launcher; ranks inherit the same explicitly selected GPU set."""
    return [str(MPI_PREFIX / 'bin/mpiexec'), '-launcher', 'fork', '-n', str(job['world']),
            '-outfile-pattern', str(Path(folder) / f'mpi-attempt{attempt}-rank-%r.stdout.log'),
            '-errfile-pattern', str(Path(folder) / f'mpi-attempt{attempt}-rank-%r.stderr.log'), *argv]


def fused_mpi_metadata(launch='eager'):
    if launch not in ('eager', 'graph'):
        raise ValueError('Unknown fused MPI launch')
    return dict(launch=launch,
                collector='mpi_graph_rank_events_v1' if launch == 'graph' else 'mpi_rank_events_v1',
                boundary=f'mpi_{launch}_maxrank_cudaevent') | (
                    dict(graph_epoch_mode='recapture_update_v1') if launch == 'graph' else {})


def merge_mpi_logs(paths, log_path, folder, attempt, require_complete, launch='eager'):
    """Keep original per-rank bytes. Concatenation is not cross-rank time order."""
    records = []
    complete = True
    with Path(log_path).open('ab') as merged:
        for row in paths:
            stdout = row['stdout']
            rank_started = stdout.is_file() and f'device,rank={row["rank"]},'.encode() in stdout.read_bytes()
            complete = complete and rank_started
            for stream in ('stdout', 'stderr'):
                path = row[stream]
                present = path.is_file()
                complete = complete and present
                merged.write(f'\n# MPI rank={row["rank"]} stream={stream}; original={path.name}\n'.encode())
                begin = merged.tell()
                if present:
                    with path.open('rb') as source:
                        shutil.copyfileobj(source, merged)
                records.append(dict(rank=row['rank'], stream=stream, path=path.name, present=present,
                    sha256=sha(path) if present else None, bytes=path.stat().st_size if present else 0,
                    merged_begin=begin, merged_end=merged.tell(), rank_started=rank_started))
    write_json(Path(folder) / f'mpi-logs-attempt{attempt}.json', dict(schema='sm103_mpi_rank_logs_v1',
        ordering='rank_then_stream_not_global_chronological', collector=fused_mpi_metadata(launch)['collector'],
        merged_log=Path(log_path).name, merged_sha256=sha(log_path), ranks=records,
        complete=complete, numeric_or_performance_accepted=False))
    if require_complete and not complete:
        raise RuntimeError('MPI exited zero but at least one rank has missing startup/log evidence')


def gemm_probe_geometry(job):
    """Single-GPU diagnostic at the per-rank GEMM geometry, not a CP run."""
    shape = fused_geometry(job)
    qkv = job['directions'] == 'qkv'
    m = shape['seq_local']
    n = shape['projection_width'] if qkv else shape['hidden']
    k = shape['hidden'] if qkv else shape['q_width']
    return m, n, k


def validate_gemm_matrix(payload):
    if not isinstance(payload, dict) or payload.get('schema') != 'sm103_gemm_matrix_v1':
        raise ValueError('Unknown GEMM matrix schema')
    shapes = payload.get('shapes')
    if not isinstance(shapes, list) or not 1 <= len(shapes) <= 256:
        raise ValueError('GEMM matrix requires 1..256 explicit geometry records')
    names = set()
    for row in shapes:
        if not isinstance(row, dict) or set(row) != {'id', 'm', 'n', 'k'}:
            raise ValueError('GEMM matrix record requires exactly id/m/n/k')
        if not isinstance(row['id'], str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,160}', row['id']):
            raise ValueError('Invalid GEMM geometry id')
        if row['id'] in names:
            raise ValueError('Repeated GEMM geometry id')
        names.add(row['id'])
        if any(type(row[key]) is not int or not 1 <= row[key] <= 2**31 - 1 for key in ('m', 'n', 'k')):
            raise ValueError('GEMM dimensions must be positive int32 values')
    return payload


def gemm_probe_shapes(job):
    if job.get('gemm_matrix_payload'):
        return validate_gemm_matrix(job['gemm_matrix_payload'])['shapes']
    m, n, k = gemm_probe_geometry(job)
    return [dict(id=job['directions'], m=m, n=n, k=k)]


def gemm_probe_argv(job, folder):
    if job.get('gemm_precision') == 'mxfp8':
        if job.get('mxfp8_gemm_search'):
            return [str(REMOTE / 'build/sm103-mxfp8-search/mxfp8_cutlass_search'),
                    str(folder / 'mxfp8-matrix.txt'), '1', str(job['gemm_sm_budget'])]
        return [str(REMOTE / 'build/sm103/mxfp8_gemm_bench'), str(folder / 'mxfp8-matrix.txt'),
                str(job.get('gemm_candidates') or 32)]
    argv = [PYTHON, '-u', str(REMOTE / 'benchmarks/sm103/GEMM/cublaslt_bench.py'),
            '--precisions', 'bf16',
            '--launches', job['launches'], '--warmup', '10', '--iterations', '50',
            '--tune-warmup', '10', '--tune-iterations', '50',
            '--library', str(REMOTE / 'build/sm103/libfuse_sm103_cublaslt.so'),
            '--output', str(folder / 'gemm-probe.json')]
    if job.get('gemm_operand_layout', 'nt') != 'nt':
        argv += ['--operand-layout', job['gemm_operand_layout']]
    if job.get('gemm_candidates') is not None:
        argv += ['--candidates', str(job['gemm_candidates'])]
    if job.get('cublaslt_sm_target') is not None:
        argv += ['--cublaslt-sm-target', str(job['cublaslt_sm_target'])]
    if job.get('gemm_sm_budget') is not None:
        argv += ['--gemm-sm-budget', str(job['gemm_sm_budget'])]
    if job.get('gemm_matrix_payload'):
        argv += ['--matrix-json', str(folder / 'gemm-matrix.json')]
    else:
        m, n, k = gemm_probe_geometry(job)
        argv += ['--m', str(m), '--n', str(n), '--k', str(k)]
    if job.get('compare_cutlass'):
        argv += ['--compare-cutlass-library', str(cutlass_probe_library())]
        if job.get('cutlass_swizzle_size') is not None:
            argv += ['--cutlass-swizzle-size', str(job['cutlass_swizzle_size'])]
        if job.get('cutlass_epilogue_n') is not None:
            argv += ['--cutlass-epilogue-n', str(job['cutlass_epilogue_n'])]
        if job.get('cutlass_1sm_cluster_m') is not None:
            argv += ['--cutlass-1sm-cluster-m', str(job['cutlass_1sm_cluster_m'])]
        if job.get('cutlass_sm_budget') is not None:
            argv += ['--cutlass-sm-budget', str(job['cutlass_sm_budget'])]
        if job.get('cutlass_full_check'):
            argv += ['--cutlass-full-check']
        if job.get('cutlass_counters'):
            argv += ['--cutlass-counters']
    return argv


def cutlass_counter_argv(folder, argv):
    # NVTX selects exactly the warmed 1-SM call, not Lt/2-SM with similar names.
    # launch-count limits target calls, NOT the hardware-counter replay passes.
    return ['ncu', '--replay-mode', 'kernel', '--nvtx',
            '--nvtx-include', CUTLASS_COUNTER_RANGE + '/', '--launch-count', '1',
            '--clock-control', 'none', '--cache-control', 'none',
            '--metrics', ','.join(CUTLASS_COUNTER_METRICS),
            '--export', str(folder / 'cutlass-counters'), *argv]


def cublaslt_counter_argv(folder, argv):
    return ['ncu', '--replay-mode', 'kernel', '--nvtx',
            '--nvtx-include', 'fuse_cublaslt_counters/', '--launch-count', '1',
            '--clock-control', 'none', '--cache-control', 'none',
            '--metrics', ','.join(CUTLASS_COUNTER_METRICS),
            '--export', str(folder / 'cutlass-counters'), *argv, '--cublaslt-counters']


def run_counter_tool(argv, env):
    return command(['bash', '-c', source_environment() + ' && exec "$@"',
                    'l20d-counter', *argv], env=env, capture_output=True, text=True)


def fused_counter_argv(folder, argv, tool='ncu', replay='app-range'):
    # The complete application is replayed: serial kernel replay can deadlock
    # peer finalizers and destroys the concurrency this experiment measures.
    # Only device 0 is sampled. All peers execute, but application replay may
    # perturb their overlap; counters alone do not establish L2 contention.
    if tool == 'nsys':
        return ['nsys', 'profile', '--trace=cuda', '--sample=none', '--cpuctxsw=none',
                '--capture-range=cudaProfilerApi', '--capture-range-end=stop',
                '--gpu-metrics-devices=0', '--gpu-metrics-frequency=10000',
                '--output', str(folder / 'fused-counters'), *argv]
    if replay == 'application':
        return ['ncu', '--replay-mode', 'application', '--target-processes', 'all',
                '--profile-from-start', 'off', '--nvtx',
                '--nvtx-include', 'fuse_communication_counters/',
                '--devices', '0', '--launch-count', '1', '--clock-control', 'none',
                '--cache-control', 'none', '--metrics', ','.join(FUSED_COUNTER_METRICS),
                '--export', str(folder / 'fused-counters'), *argv]
    return ['ncu', '--replay-mode', replay, '--nvtx',
            '--nvtx-include', 'fuse_communication_counters/',
            '--devices', '0', '--launch-count', '1', '--clock-control', 'none',
            '--cache-control', 'none', '--metrics', ','.join(FUSED_COUNTER_METRICS),
            '--export', str(folder / 'fused-counters'), *argv]


def prepare_fused_counters(folder, env, job):
    if job.get('fused_counter_tool', 'ncu') == 'nsys':
        version = run_counter_tool(['nsys', '--version'], env).stdout
        (folder / 'nsys-version.txt').write_text(version)
        write_json(folder / 'fused-counter-contract.json', dict(
            schema='sm103_fused_sampled_counter_contract_v1', diagnostic_only=True,
            performance_accepted=False, component=job['fused_counters'],
            direction=job.get('directions', 'qkv'), profiled_device=0,
            active_devices=job['world'], selected_ranges=1, tool='nsys',
            sample_frequency_hz=10000, replay_mode='none',
            metrics='runtime metric set; availability audited from exported SQLite',
            scope='one warmed full epoch; temporal GPU0 samples, not compute-CTA attribution'))
        return
    version = run_counter_tool(['ncu', '--version'], env).stdout
    (folder / 'ncu-version.txt').write_text(version)
    bases = list(dict.fromkeys(metric.split('.', 1)[0] for metric in FUSED_COUNTER_METRICS))
    query = run_counter_tool(['ncu', '--query-metrics', '--query-metrics-mode', 'suffix',
                              '--metrics', ','.join(bases)], env)
    (folder / 'ncu-metrics-query.txt').write_text(query.stdout + query.stderr)
    if any(metric not in query.stdout for metric in FUSED_COUNTER_METRICS):
        raise RuntimeError('Requested fused counter metrics unavailable; inspect query')
    write_json(folder / 'fused-counter-contract.json', dict(
        schema='sm103_fused_counter_contract_v1', diagnostic_only=True, performance_accepted=False,
        component=job['fused_counters'], direction=job.get('directions', 'qkv'),
        profiled_device=0, active_devices=job['world'], selected_ranges=1,
        metrics=list(FUSED_COUNTER_METRICS), replay_mode=job.get('fused_counter_replay', 'app-range'),
        nvtx_range='fuse_communication_counters',
        clock_control='none', cache_control='none',
        scope=('first GPU0 kernel inside warmed epoch; peer overlap unverified, excludes finalization'
               if job.get('mpi') else
               'one warmed full epoch including peer finalization; aggregate, not compute-CTA attribution')))


def collect_fused_counters(folder, env, tool='ncu'):
    if tool == 'nsys':
        report = folder / 'fused-counters.nsys-rep'
        if not report.is_file() or report.stat().st_size == 0:
            raise RuntimeError('Nsight Systems produced no report')
        exported = run_counter_tool(['nsys', 'export', '--type=sqlite', '--output',
                                     str(folder / 'fused-counters.sqlite'), str(report)], env)
        (folder / 'nsys-export.log').write_text(exported.stdout + exported.stderr)
        return
    report = folder / 'fused-counters.ncu-rep'
    if not report.is_file() or report.stat().st_size == 0:
        raise RuntimeError('NCU produced no fused counter report')
    imported = run_counter_tool(['ncu', '--import', str(report), '--page', 'raw', '--csv',
                                  '--metrics', ','.join(FUSED_COUNTER_METRICS)], env)
    (folder / 'fused-counters.csv').write_text(imported.stdout)
    (folder / 'ncu-import.log').write_text(imported.stderr)
    rows = list(csv.reader(imported.stdout.splitlines()))
    header = next((row for row in rows if 'ID' in row and
                   all(metric in row for metric in FUSED_COUNTER_METRICS)), None)
    if header is None:
        raise RuntimeError('NCU range export is missing requested metrics')
    data = [dict(zip(header, row)) for row in rows if len(row) == len(header) and
            row[header.index('ID')].isdigit()]
    if len(data) != 1:
        raise RuntimeError('Expected exactly one profiled range')
    for metric in FUSED_COUNTER_METRICS:
        value = float(data[0][metric].replace(',', ''))
        if not math.isfinite(value) or value < 0:
            raise RuntimeError(f'Invalid fused counter metric: {metric}')


def prepare_cutlass_counters(folder, env):
    version = run_counter_tool(['ncu', '--version'], env).stdout
    (folder / 'ncu-version.txt').write_text(version)
    # NCU 2025.3 filters suffix queries by BASE names; fully suffixed names
    # returned only a device heading. Profiling below still requests the exact
    # six full metrics, and every one must be present in this query's output.
    bases = list(dict.fromkeys(metric.split('.', 1)[0] for metric in CUTLASS_COUNTER_METRICS))
    query = run_counter_tool(['ncu', '--query-metrics', '--query-metrics-mode', 'suffix',
                              '--metrics', ','.join(bases)], env)
    (folder / 'ncu-metrics-query.txt').write_text(query.stdout + query.stderr)
    if any(metric not in query.stdout for metric in CUTLASS_COUNTER_METRICS):
        raise RuntimeError('Requested NCU metrics unavailable; inspect query, no silent substitution')
    write_json(folder / 'cutlass-counter-contract.json', dict(
        schema='sm103_cutlass_counter_contract_v1', diagnostic_only=True, performance_accepted=False,
        nvtx_range=CUTLASS_COUNTER_RANGE, selected_calls=1, selected_sm_mode=1,
        metrics=list(CUTLASS_COUNTER_METRICS), replay_mode='kernel',
        clock_control='none', cache_control='none',
        replay_passes='reported by NCU, not limited by selected_calls'))


def collect_cutlass_counters(folder, env):
    report = folder / 'cutlass-counters.ncu-rep'
    if not report.is_file() or report.stat().st_size == 0:
        raise RuntimeError('NCU produced no counter report; do not accept the diagnostic')
    imported = run_counter_tool(['ncu', '--import', str(report), '--page', 'raw', '--csv',
                                  '--metrics', ','.join(CUTLASS_COUNTER_METRICS)], env)
    (folder / 'cutlass-counters.csv').write_text(imported.stdout)
    (folder / 'ncu-import.log').write_text(imported.stderr)
    rows = list(csv.reader(imported.stdout.splitlines()))
    header = next((row for row in rows if 'ID' in row and 'Kernel Name' in row), None)
    if header is None or any(metric not in header for metric in CUTLASS_COUNTER_METRICS):
        raise RuntimeError('NCU raw export is missing requested metrics')
    data = [dict(zip(header, row)) for row in rows if len(row) == len(header) and
            row[header.index('ID')].isdigit()]
    if len(data) != 1:
        raise RuntimeError('NCU must export exactly one selected kernel')
    for metric in CUTLASS_COUNTER_METRICS:
        value = float(data[0][metric].replace(',', ''))
        if not math.isfinite(value) or value < 0:
            raise RuntimeError(f'NCU metric is unavailable or invalid: {metric}')
    # Preserve raw profiler units/values. In particular, duration is not a
    # production sample and aggregate tensor activity is not effective FLOPs.


def cutlass_probe_library():
    return REMOTE / 'build/sm103-cutlass/libfuse_sm103_cutlass_bf16.so'


def cutlass_probe_build_argv():
    configure = ['cmake', '-S', str(REMOTE / 'benchmarks/sm103'),
                 '-B', str(cutlass_probe_library().parent), '-G', 'Ninja',
                 '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CUDA_ARCHITECTURES=103a',
                 '-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc',
                 '-DFUSE_SM103_BUILD_CUTLASS_BF16=ON', '-DCUTLASS_ROOT=' + CUTLASS]
    build = ['cmake', '--build', str(cutlass_probe_library().parent),
             '--target', 'fuse_sm103_cutlass_bf16', '--parallel', '4']
    return ['bash', '-c', shlex.join(configure) + ' && exec ' + shlex.join(build)]


def cutlass_probe_build_receipt(job, env_id):
    names = ('benchmarks/sm103/CMakeLists.txt', 'csrc/baselines/sm103/cutlass_bf16.cu')
    if any(name not in job['files'] for name in names):
        raise RuntimeError('CUTLASS probe native build inputs missing from snapshot')
    library = cutlass_probe_library()
    if not library.is_file():
        raise RuntimeError('Build the optional CUTLASS GEMM probe library first')
    dependencies = read_command(['bash', '-c',
        source_environment() + ' && exec ldd "$1"', 'cutlass-probe-link', str(library)])
    dependencies = re.sub(r' \(0x[0-9a-fA-F]+\)', '', dependencies)
    if 'not found' in dependencies or 'libcuda.so' not in dependencies:
        raise RuntimeError('CUTLASS probe has unresolved or absent driver linkage')
    return dict(node=job_node(job), environment_fingerprint=env_id, library=str(library),
                library_sha256=sha(library), inputs={name: job['files'][name] for name in names},
                dynamic_dependencies=dependencies)


def mxfp8_search_receipt(job, env_id):
    binary = REMOTE / 'build/sm103-mxfp8-search/mxfp8_cutlass_search'
    names = {name: digest for name, digest in job['files'].items()
             if name.startswith(('include/fuse/', 'csrc/operators/sm103/',
                                 'benchmarks/sm103/GEMM/', 'benchmarks/sm103/fused_'))
             or name in ('benchmarks/sm103/CMakeLists.txt', 'csrc/baselines/sm103/cublaslt_training.cu')}
    return dict(node=job_node(job), environment_fingerprint=env_id,
                binary=str(binary), binary_sha256=sha(binary), inputs=names,
                library_sha256=sha(binary.parent / 'libfuse_sm103_cublaslt.so'))


def check_fused_devices(job, folder, devices=None, memory=None):
    devices = fused_devices(job) if devices is None else devices
    memory = fused_device_memory(job) if memory is None else memory
    fields = ('index', 'uuid', 'utilization.gpu', 'memory.free', 'memory.used',
              'clocks.sm', 'clocks.mem', 'power.draw')
    observations = []
    quiet_samples = 0
    last_busy = None
    # A brief occupied observation is not permission to run through another
    # workload, nor a reason to submit a stream of failed jobs. Wait within
    # this job for the original three consecutive idle samples (<=5%); never
    # lower the threshold, ignore an active rank, or stop its processes.
    for sample in range(31):
        raw = read_command(['nvidia-smi', '--id=' + ','.join(devices),
                            '--query-gpu=' + ','.join(fields), '--format=csv,noheader,nounits'])
        rows = [dict(zip(fields, (value.strip() for value in row))) for row in csv.reader(raw.splitlines())]
        observations.append(dict(observed_at=time.time(), devices=rows))
        # Preserve failed checks too. Existing process/PID presence is never a gate.
        write_json(folder / 'gpu-before.json', dict(node=job_node(job), selected=devices,
                                                   memory_estimate=memory, observations=observations))
        by_index = {row.get('index'): row for row in rows}
        if len(rows) != len(devices) or set(by_index) != set(devices):
            raise RuntimeError('GPU observation did not cover exactly the selected physical devices')
        busy = []
        for device in devices:
            row = by_index[device]
            utilization, free = float(row['utilization.gpu']), float(row['memory.free'])
            if not math.isfinite(utilization) or not math.isfinite(free):
                raise RuntimeError(f'GPU {device} returned invalid utilization/memory telemetry')
            if utilization > 5:
                busy.append((device, utilization))
            if free * (1 << 20) < memory['minimum_free_bytes']:
                required_mib = memory['minimum_free_bytes'] / (1 << 20)
                raise RuntimeError(f'GPU {device} has {free:g} MiB free; estimated requirement '
                                   f'{required_mib:.1f} MiB (minimum 2 GiB); no processes were stopped')
            if not row.get('uuid', '').startswith('GPU-'):
                raise RuntimeError(f'GPU {device} did not return a valid UUID')
        if busy:
            if last_busy is None:
                print('warning,gpu_wait=waiting_for_three_idle_samples,max_observations=31', flush=True)
            last_busy = busy[0]
            quiet_samples = 0
        else:
            quiet_samples += 1
            if quiet_samples == 3:
                break
        if sample == 30:
            device, utilization = last_busy
            raise RuntimeError(f'No sustained idle window; last busy observation: GPU {device} '
                               f'is computing ({utilization}%); no processes were stopped')
        time.sleep(1)
    # UUIDs make nvidia-smi physical indices unambiguous despite CUDA ordinal order.
    return ','.join(by_index[device]['uuid'] for device in devices)


def remote(job_path):
    import fcntl
    import signal
    job = json.loads(Path(job_path).read_text())
    if job.get('workspace') is not None:
        configure_workspace(job['workspace'])
        import pwd
        if pwd.getpwuid(os.geteuid()).pw_name != workspace_user():
            raise RuntimeError('Job user does not own the selected workspace')
        if WORKSPACE.resolve() != WORKSPACE or WORKSPACE.stat().st_uid != os.geteuid():
            raise RuntimeError('Workspace is not an owned real directory')
    identifier(job['run_id'])
    identifier(job['experiment'])
    node = validate_job(job, socket.gethostname())
    folder = CONTROL / 'jobs' / job['run_id']
    folder.mkdir(parents=True, exist_ok=True)
    status_path = folder / 'status.json'
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    if previous.get('state') == 'succeeded':
        print('Already succeeded; fetch the existing result.')
        return 0
    state = dict(run_id=job['run_id'], node=node, stage=job['stage'], experiment=job['experiment'],
                 source_id=job['source_id'], state='running', phase='lock',
                 attempt=previous.get('attempt', 0) + 1, started_at=time.time())
    cloud = prefix(job['run_id'])
    def publish(**changes):
        next_phase = changes.get('phase', state['phase'])
        now = time.time()
        if next_phase != state['phase']:
            state.setdefault('timings_s', {})[state['phase']] = round(now - state.get('phase_started', state['started_at']), 4)
            state['phase_started'] = now
        state.update(changes, elapsed_s=round(time.time() - state['started_at'], 1))
        write_json(status_path, state)
        try:
            mc_copy(status_path, cloud + '/status.json', timeout=20)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            print(f'Status upload failed; local receipt: {status_path}', flush=True)
        print(f"[{job['run_id']}] {state['phase']}: {state['state']} ({state['elapsed_s']}s)", flush=True)
    lock = (CONTROL / 'workspace.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        publish(state='failed', error='Another managed task owns the workspace lock', exit_code=75)
        lock.close()
        return 75
    log_path = folder / f"attempt{state['attempt']}.log"
    if previous.get('collection_error') and previous.get('artifact_local'):
        # Retry only transfer, preserving the original computation outcome.
        work_rc = previous['work_exit_code']
        artifact = Path(previous['artifact_local'])
        state.update(work_exit_code=work_rc, artifact_local=str(artifact))
        publish(phase='collect')
        try:
            if artifact.parent != folder or not artifact.is_file():
                raise RuntimeError('Expected local artifact missing; inspect receipt')
            object_name = f"{CLOUD}/results/automation/{job['run_id']}/{artifact.name}"
            mc_copy(artifact, object_name)
            state.update(artifact=object_name, artifact_sha256=sha(artifact))
            publish(phase='finished', state='succeeded' if work_rc == 0 else 'failed', exit_code=work_rc)
            return work_rc
        except Exception as error:
            publish(phase='finished', state='failed', exit_code=1, collection_error=str(error))
            return 1
        finally:
            lock.close()
    rc = 1
    mpi_logs = None
    try:
        publish(phase='sync')
        installed = folder / 'source-installed.json'
        if not installed.exists():
            install_source(job, folder)
        else:
            # Resume never re-labels a newer dirty checkout as the original source.
            for name, value in job['files'].items():
                path = REMOTE / name
                actual = 'link:' + os.readlink(path) if path.is_symlink() else sha(path)
                if actual != value:
                    raise RuntimeError(f'Source changed since this run: {name}; submit a new run')
        receipt = environment_receipt(require_te=job['stage'] not in
            (*FUSED_STAGES, 'build', 'gemm-probe', 'gemm-cutlass-build', 'transport-probe'))
        if job.get('mpi'):
            receipt['mpi_toolchain'] = mpi_toolchain_receipt()
        env_id = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
        old_env = folder / 'environment.json'
        if old_env.exists() and json.loads(old_env.read_text())['fingerprint'] != env_id:
            raise RuntimeError('Environment changed since this run; preserve old results and submit a new experiment')
        write_json(folder / 'environment.json', receipt | {'fingerprint': env_id})
        state['environment_fingerprint'] = env_id
        env = os.environ.copy()
        env.update(receipt['overrides'], FUSE_ENV_FINGERPRINT=env_id, OMP_NUM_THREADS='1')
        if job.get('mpi'):
            env.update(receipt['mpi_toolchain']['overrides'])
        stage = job['stage']
        results = REMOTE / 'results/sm103' / job['experiment']
        if stage == 'doctor':
            argv = [PYTHON, str(REMOTE / 'scripts/l20d_doctor.py'), str(folder / 'doctor.json')]
            env['CUDA_VISIBLE_DEVICES'] = '0'
        elif stage == 'overhead':
            argv = [PYTHON, str(REMOTE / 'scripts/bench_l20d_workflow.py'),
                    '--output', str(folder / f"overhead-attempt{state['attempt']}.json")]
        elif stage == 'build':
            argv = ['bash', str(REMOTE / 'scripts/build_sm103_bench.sh')]
            if job.get('mxfp8_gemm_search'):
                build_dir = REMOTE / 'build/sm103-mxfp8-search'
                configure = ['cmake', '-S', str(REMOTE / 'benchmarks/sm103'), '-B', str(build_dir),
                    '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CUDA_ARCHITECTURES=103a',
                    '-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc',
                    '-DFUSE_SM103_BUILD_MXFP8_SEARCH=ON', '-DCUTLASS_ROOT=' + CUTLASS]
                compile_argv = ['cmake', '--build', str(build_dir), '--target', 'mxfp8_cutlass_search', '--parallel', '2']
                argv = ['bash', '-c', shlex.join(configure) + ' && exec ' + shlex.join(compile_argv)]
        elif stage == 'gemm-cutlass-build':
            argv = cutlass_probe_build_argv()
        elif stage == 'transport-probe':
            devices = fused_devices(job)[:2]
            env['CUDA_VISIBLE_DEVICES'] = check_fused_devices(job, folder, devices,
                dict(minimum_free_bytes=2 << 30, note='Two-GPU transport microbenchmark, no GEMM'))
            build = REMOTE / 'build/sm103-transport'
            build.mkdir(parents=True, exist_ok=True)
            binary = build / 'transport_probe'
            compile_argv = ['/usr/local/cuda/bin/nvcc', '-O3', '-std=c++17', '-arch=sm_103a',
                '-I' + CUTLASS + '/include', str(REMOTE / 'benchmarks/sm103/transport_probe.cu'),
                '-o', str(binary)]
            write_json(folder / 'transport-contract.json', dict(
                hardware_user_description='B300 (reported name may be L20D)', physical_devices=devices,
                source_id=job['source_id'], direction='visible GPU1 -> visible GPU0 pull',
                warmup=10, samples=50, repetitions_per_sample=32, slots=32,
                endpoints=['remote GMEM -> local SMEM', 'remote GMEM -> local GMEM'],
                concurrency=[[1,1],[20,4]], cache='warm cyclic working set; no forced flush',
                latency='per-warp globaltimer, includes issue/wait/join; not intrinsic hardware latency',
                bandwidth='G2G aggregate includes loop/timer/kernel overhead; G2S per-worker service rate only'))
            argv = ['bash', '-c', shlex.join(compile_argv) + ' && exec ' +
                    shlex.join([str(binary), str(folder / 'transport.csv')])]
        elif stage in FUSED_STAGES:
            argv = fused_argv(job)
            if stage == 'fused-smoke':
                check_fused_build(job, env_id, folder)
                if job.get('mxfp8_service_probe'):
                    argv += ['--mxfp8-service-output', str(folder / 'services-rank-0.jsonl')]
                backward_memory = None
                if job.get('backward') and job.get('mpi'):
                    s = fused_geometry(job)
                    m,h = s['seq_local'],s['hidden']
                    w = (s['q_heads']+(2*s['kv_heads'] if job['fused_direction']=='qkv' else 0))*s['head_dim']
                    elements = 2*m*w+2*m*h+2*w*h
                    if job['fused_direction']=='oproj': elements = 2*m*h+3*m*w+2*w*h
                    backward_memory = dict(minimum_free_bytes=2*elements+(2<<30), note='backward live tensors plus bounded reference and 2 GiB headroom')
                    if job.get('backward_matrix_payload'):
                        backward_memory = dict(minimum_free_bytes=2<<30, note='per-case collective device memory checks in backward batch')
                        fields=('id','direction','m','hidden','q_heads','kv_heads','head_dim')
                        (folder/'backward-matrix.txt').write_text(''.join(
                            ' '.join([str(row[k]) for k in fields]+[str(row.get('tile_n',128)),
                                str(row.get('epilogue_n',0)),str(row.get('swizzle',1)),
                                str(int(row.get('along_m',False)))])+'\n' for row in job['backward_matrix_payload']))
                        argv += ['--case-matrix',str(folder/'backward-matrix.txt'),
                                 '--json-prefix',str(folder/'backward-')]
                    else:
                        argv += ['--json-out',str(folder/'backward-result.json')]
                    if job.get('backward_gemm_sweep'):
                        argv += ['--gemm-sweep']
                    elif job.get('profile'):
                        argv += ['--role-profile','--trace-out',str(folder/'backward-perfetto.json')]
                env['CUDA_VISIBLE_DEVICES'] = check_fused_devices(job, folder, memory=backward_memory)
                env['FUSE_QKV_GEMM_POLICY'] = job.get('qkv_policy', 'auto')
                env['FUSE_SM103_OPROJ_POLICY'] = job.get('oproj_policy', 'auto')
                env['FUSE_SM103_OPROJ_COMM_LAYOUT'] = job.get('oproj_comm_layout', 'rows')
                if job.get('fused_counters') and not job.get('mpi'):
                    prepare_fused_counters(folder, env, job)
                    argv = fused_counter_argv(folder, argv, job.get('fused_counter_tool', 'ncu'),
                                              job.get('fused_counter_replay', 'app-range'))
                if job.get('mpi'):
                    mpi_logs = mpi_rank_logs(folder, state['attempt'], job['world'])
                    for row in mpi_logs:
                        for stream in ('stdout', 'stderr'):
                            row[stream].touch(exist_ok=False)
                    argv = mpi_launch_argv(job, argv, folder, state['attempt'])
                    if job.get('fused_counters'):
                        prepare_fused_counters(folder, env, job)
                        argv = fused_counter_argv(folder, argv, 'ncu', 'application')
                    write_json(folder / f'mpi-runtime-attempt{state["attempt"]}.json', dict(
                        schema='sm103_mpi_runtime_v1', node=node, world=job['world'],
                        process_layout='mpi_one_process_per_gpu', host_launch='mpi_process',
                        **fused_mpi_metadata(job.get('fused_launch', 'eager')),
                        overrides=receipt['mpi_toolchain']['overrides'], argv=argv,
                        rank_stdout_pattern=f'mpi-attempt{state["attempt"]}-rank-%r.stdout.log',
                        rank_stderr_pattern=f'mpi-attempt{state["attempt"]}-rank-%r.stderr.log'))
        elif stage == 'baseline-replay':
            argv = baseline_replay_plan(job, env_id, results, folder)
        elif stage == 'gemm-probe':
            argv = gemm_probe_argv(job, folder)
            library = REMOTE / 'build/sm103/libfuse_sm103_cublaslt.so'
            if not library.is_file():
                raise RuntimeError('The existing SM103 cuBLASLt diagnostic library is missing')
            cutlass_receipt = None
            if job.get('compare_cutlass'):
                cutlass_receipt = cutlass_probe_build_receipt(job, env_id)
                saved_receipt = cutlass_probe_library().parent / '.l20d-build.json'
                if not saved_receipt.is_file() or json.loads(saved_receipt.read_text()) != cutlass_receipt:
                    raise RuntimeError('CUTLASS diagnostic build changed; run gemm-cutlass-build before comparison')
                write_json(folder / 'cutlass-probe-build.json', cutlass_receipt)
            shapes = gemm_probe_shapes(job)
            if job.get('gemm_precision') == 'mxfp8':
                binary = REMOTE / 'build/sm103/mxfp8_gemm_bench'
                if job.get('mxfp8_gemm_search'):
                    binary = REMOTE / 'build/sm103-mxfp8-search/mxfp8_cutlass_search'
                    receipt = mxfp8_search_receipt(job, env_id)
                    saved = binary.parent / '.l20d-build.json'
                    if not saved.is_file() or json.loads(saved.read_text()) != receipt:
                        raise RuntimeError('MXFP8 search build changed; rebuild the isolated target')
                    write_json(folder / 'mxfp8-search-build.json', receipt)
                if not binary.is_file():
                    raise RuntimeError('Native MXFP8 benchmark missing; run build first')
                (folder / 'mxfp8-matrix.txt').write_text(''.join(
                    ' '.join(str(row[k]) for k in ('id', 'm', 'n', 'k')) + '\n' for row in shapes))
            if job.get('gemm_matrix_payload'):
                write_json(folder / 'gemm-matrix.json', job['gemm_matrix_payload'])
            # BF16 operands/output plus FP32 checker temporaries and workspace.
            maximum_elements = max(row['m']*row['k'] + row['n']*row['k'] + row['m']*row['n']
                                   for row in shapes)
            memory = dict(minimum_free_bytes=max(8 << 30, 8 * maximum_elements + (1 << 30)),
                          guarantees_fit=False, note='Single GPU pure-GEMM diagnostic allowance')
            if job.get('gemm_precision') == 'mxfp8':
                memory = dict(minimum_free_bytes=2 << 30,
                    note='Native MXFP8 checks each case against its live buffers plus 2 GiB before allocation')
            if job.get('gemm_operand_layout', 'nt') != 'nt':
                # BF16 views have no packing allocation. Row-selected checker
                # uses only 64 rows of each operand, including long-K wgrad.
                minimum = max(2*(r['m']*r['k']+r['n']*r['k']+r['m']*r['n']) +
                    4*(min(64,r['m'])+min(64,r['n']))*r['k'] for r in shapes)
                memory = dict(minimum_free_bytes=minimum+(2 << 30), guarantees_fit=False,
                              note='BF16 zero-copy backward views and bounded checker plus 2 GiB headroom')
            selected = fused_devices(job)[:1]
            env['CUDA_VISIBLE_DEVICES'] = check_fused_devices(job, folder, selected, memory)
            write_json(folder / 'gemm-probe-contract.json', dict(
                node=node, measurement=('single_gpu_cutlass_counters' if job.get('cutlass_counters') else
                    'single_gpu_cutlass_cublaslt_comparison' if cutlass_receipt else
                    'single_gpu_pure_cutlass_search' if job.get('mxfp8_gemm_search') else
                    'single_gpu_pure_cublaslt'), physical_devices=selected,
                cuda_visible_devices=env['CUDA_VISIBLE_DEVICES'],
                shape=({key: shapes[0][key] for key in ('m', 'n', 'k')}
                       if not job.get('gemm_matrix_payload') else None),
                shapes=shapes, geometry_cp=(job['world'] if not job.get('gemm_matrix_payload') else None),
                operand_layout=job.get('gemm_operand_layout', 'nt'),
                transpose_materialized=False, candidates_requested=job.get('gemm_candidates'),
                measured_ranks=1, math_sms=job.get('gemm_sm_budget') or job.get('cublaslt_sm_target') or 0,
                requested_gemm_sm_budget=job.get('gemm_sm_budget'),
                precision=job.get('gemm_precision', 'bf16'),
                sm_budget_enforcement=('persistent_grid_one_resident_cta_per_sm' if job.get('mxfp8_gemm_search') else
                                       'full_device_no_restriction' if job.get('gemm_precision') == 'mxfp8' else
                                       'cuda_green_context' if job.get('gemm_sm_budget') else
                                       'cublaslt_heuristic_hint_not_hard_partition'),
                library_sha256=sha(library), correctness=('full_output_two_payloads' if
                    job.get('gemm_precision') == 'mxfp8' else 'existing_evenly_spaced_64x64_check'),
                cublas_classic_measured=False, distributed_boundary_measured=False))
            if job.get('cutlass_counters'):
                prepare_cutlass_counters(folder, env)
                argv = cutlass_counter_argv(folder, argv)
            elif job.get('cublaslt_counters'):
                prepare_cutlass_counters(folder, env)
                write_json(folder / 'cutlass-counter-contract.json', dict(
                    schema='sm103_cublaslt_counter_contract_v1', backend='cublaslt',
                    diagnostic_only=True, performance_accepted=False,
                    nvtx_range='fuse_cublaslt_counters', selected_calls=1,
                    metrics=list(CUTLASS_COUNTER_METRICS), replay_mode='kernel',
                    clock_control='none', cache_control='none'))
                argv = cublaslt_counter_argv(folder, argv)
        elif stage == 'te-build':
            argv = ['bash', str(REMOTE / 'scripts/build_sm103_te.sh')]
        elif stage == 'ub-check':
            env['CUDA_VISIBLE_DEVICES'] = job['devices']
            argv = [PYTHON, '-m', 'torch.distributed.run', '--standalone',
                    '--nproc-per-node=8', str(REMOTE / 'scripts/check_sm103_ub.py'),
                    '--output', str(folder / 'ub-check.json')]
        elif stage == 'batch-check':
            argv = [PYTHON, '-u', str(REMOTE / 'scripts/check_sm103_batch.py'), '--results', str(results)]
        elif stage == 'matrix-check':
            argv = [PYTHON, '-u', str(REMOTE / 'scripts/check_sm103_matrix.py'), '--results', str(results),
                    '--directions', job['directions']]
        elif stage == 'cache-check':
            argv = [PYTHON, '-u', str(REMOTE / 'scripts/check_sm103_cache.py'), '--results', str(results)]
        else:
            entry = 'scripts/sm103_batch.py' if job.get('executor', 'serial') == 'batch' else 'benchmarks/sm103/bench.py'
            argv = [PYTHON, '-u', str(REMOTE / entry), '--stage', stage,
                    '--python', PYTHON, '--results', str(results), '--execute',
                    '--models', job['models'], '--seqs', job['seqs'], '--cps', job['cps'],
                    '--devices', job['devices'], '--backends', job['backends'],
                    '--directions', job['directions'], '--launches', job['launches'],
                    '--oproj-layout', job.get('oproj_layout', 'legacy'),
                    '--job-timeout', str(job['job_timeout'])]
            if job.get('reuse_experiment'):
                if job.get('executor') != 'batch':
                    raise ValueError('result import requires the batching executor')
                argv += ['--import-plan', str(REMOTE / 'results/sm103' / job['reuse_experiment'] / f'{stage}_plan.json')]
        # Effective compiler/toolchain overrides are confined to this child.
        if job.get('mpi'):
            # Apply after env.sh as well, so no inherited UCX default can select
            # RDMA for this same-host CPU-only MPI control plane.
            argv = ['env', *[f'{key}={value}' for key, value in receipt['mpi_toolchain']['overrides'].items()], *argv]
        argv = ['bash', '-c', source_environment() + ' && exec "$@"', 'l20d', *argv]
        publish(phase=stage)
        with log_path.open('w') as log, contextlib.ExitStack() as monitors:
            if stage == 'fused-smoke':
                monitors.enter_context(fused_telemetry(job, folder))
            elif stage in ('gemm-probe', 'transport-probe'):
                monitors.enter_context(baseline_executor(REMOTE).telemetry(
                    ','.join(fused_devices(job)[:2]) if stage == 'transport-probe' else fused_devices(job)[0],
                    folder / 'gpu-telemetry.csv'))
            proc = subprocess.Popen(argv, cwd=REMOTE, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            progress_stop = threading.Event()
            progress_path = mpi_logs[0]['stdout'] if mpi_logs else log_path
            progress_thread = threading.Thread(target=relay_progress, args=(progress_path, progress_stop), daemon=True)
            progress_thread.start()
            try:
                while True:
                    try:
                        rc = proc.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        tail = progress_path.read_text(errors='replace').splitlines()[-12:]
                        (folder / 'tail.txt').write_text('\n'.join(tail) + '\n')
                        try:
                            mc_copy(folder / 'tail.txt', cloud + '/tail.txt', timeout=20)
                        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                            pass
                        publish(last_lines=tail[-4:])
                        if time.time() - state['started_at'] > job['timeout']:
                            raise TimeoutError('Task timeout; terminating only its own process group')
            except BaseException:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                raise
            finally:
                progress_stop.set()
                progress_thread.join(timeout=2)
        if rc:
            state['error'] = '\n'.join(log_path.read_text(errors='replace').splitlines()[-30:])
        elif stage == 'build' and job.get('mxfp8_gemm_search'):
            receipt = mxfp8_search_receipt(job, env_id)
            write_json(REMOTE / 'build/sm103-mxfp8-search/.l20d-build.json', receipt)
            write_json(folder / 'mxfp8-search-build.json', receipt)
        elif stage == 'fused-build':
            build_receipt = fused_build_receipt(job, env_id)
            write_json(fused_build_dir(job) / '.l20d-build.json', build_receipt)
            write_json(folder / 'fused-build.json', build_receipt)
        elif stage == 'gemm-cutlass-build':
            build_receipt = cutlass_probe_build_receipt(job, env_id)
            write_json(cutlass_probe_library().parent / '.l20d-build.json', build_receipt)
            write_json(folder / 'cutlass-probe-build.json', build_receipt)
        elif stage == 'gemm-probe' and (job.get('cutlass_counters') or job.get('cublaslt_counters')):
            collect_cutlass_counters(folder, env)
        elif stage == 'fused-smoke' and job.get('fused_counters'):
            collect_fused_counters(folder, env, job.get('fused_counter_tool', 'ncu'))
    except Exception as error:
        rc = rc or 1
        state['error'] = f'{type(error).__name__}: {error}'
        with log_path.open('a') as log:
            log.write(state['error'] + '\n')
    finally:
        if mpi_logs:
            try:
                merge_mpi_logs(mpi_logs, log_path, folder, state['attempt'], require_complete=rc == 0,
                               launch=job.get('fused_launch', 'eager'))
                if rc and not state.get('error'):
                    state['error'] = f'MPI exited {rc}; inspect preserved rank stdout/stderr and launcher log'
            except Exception as error:
                rc = rc or 1
                state['error'] = f'MPI log collection failed: {error}'
        state['work_exit_code'] = rc
        publish(phase='collect', exit_code=rc)
        try:
            artifact = folder / f"artifacts-attempt{state['attempt']}.tar.gz"
            with tarfile.open(artifact, 'w:gz') as tar:
                for file in folder.iterdir():
                    if file.is_file() and (file.suffix in ('.json', '.log', '.txt', '.csv') or
                            (job.get('mxfp8_service_probe') and file.name == 'services-rank-0.jsonl') or
                            (job.get('backward_gemm_sweep') and file.name.endswith('.gemm-sweep.jsonl')) or
                            ((job.get('cublaslt_sm_target') is not None or job.get('gemm_sm_budget'))
                             and file.name.startswith('gemm-probe-')
                             and file.name.endswith('-launch.dot')) or
                            (job.get('cutlass_counters') and file.name == 'cutlass-counters.ncu-rep') or
                            (job.get('fused_counters') and file.name in
                             ('fused-counters.ncu-rep', 'fused-counters.nsys-rep', 'fused-counters.sqlite'))):
                        tar.add(file, arcname='control/' + file.name)
                results = REMOTE / 'results/sm103' / job['experiment']
                if results.exists():
                    tar.add(results, arcname='results')
            object_name = f"{CLOUD}/results/automation/{job['run_id']}/{artifact.name}"
            state['artifact_local'] = str(artifact)
            mc_copy(artifact, object_name)
            state.update(artifact=object_name, artifact_sha256=sha(artifact))
        except Exception as error:
            state['collection_error'] = str(error)
            rc = rc or 1
        try:
            publish(phase='finished', state='succeeded' if rc == 0 else 'failed')
        finally:
            lock.close()
    return rc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    run = sub.add_parser('run')
    run.add_argument('stage', choices=STAGES)
    run.add_argument('--node', choices=tuple(NODES), default='09')
    run.add_argument('--workspace', type=str, help='Remote user workspace; saved in the job for safe resume')
    run.add_argument('--source-run', help='Reuse the verified local source snapshot of this run, not current Mac edits')
    run.add_argument('--profile', action='store_true', help='fused stages: separate instrumented build/run')
    run.add_argument('--mpi', action='store_true', help='fused stages: optional one-process-per-GPU MPI target')
    run.add_argument('--backward', action='store_true', help='reuse BF16 reverse-route harness, separate build directory')
    run.add_argument('--backward-matrix', help='BF16 backward same-CP case list; persistent MPI ranks')
    run.add_argument('--mxfp8', action='store_true', help='QKV MXFP8 activation + BF16 W -> fused weight quantization/GEMM/BF16 A2A')
    run.add_argument('--mxfp8-prequantized', action='store_true', help='MXFP8 diagnostic: exclude weight preparation')
    run.add_argument('--mxfp8-epilogue-n', type=int, choices=(32, 64),
                     help='MXFP8 CUTLASS epilogue subtile; default 64 preserves baseline')
    run.add_argument('--mxfp8-weight-preparation', choices=('comm', 'all', 'comm_warp'), help='Weight quantization by communication CTAs (default), all CTAs at startup, or comm_warp: warps 0..3 route immediately; warps 4..7 quantize then join routing (warp_then_route_v1)')
    run.add_argument('--backward-gemm-sweep', action='store_true', help='isolated pure NN GEMM candidate sweep; production overlap unchanged')
    run.add_argument('--fused-launch', choices=('eager', 'graph'), default='eager',
                     help='fused-smoke: Graph is explicit MPI-only, with epoch preparation outside CUDA events')
    run.add_argument('--fused-direction', choices=('both', 'qkv', 'oproj'), default='both',
                     help='non-profile fused-smoke: allocate, validate and measure only selected direction')
    run.add_argument('--profile-detail', choices=('full', 'cta'),
                     help='fused-smoke --profile: full peer trace (default), or CTA-only diagnostics')
    run.add_argument('--oproj-pipeline-probe', action='store_true',
                     help='full OProj profiling: GPU0 three-worker ready/MMA/epilogue diagnostic')
    run.add_argument('--qkv-epilogue-probe', action='store_true',
                     help='private N256/K64/e32 QKV epilogue diagnostic; requires profile and CTA detail')
    run.add_argument('--rebuild', action='store_true', help='fused-build: explicitly rebuild this workspace-local build directory')
    run.add_argument('--qkv-rank-swizzle', action='store_true',
                     help='experimental QKV rank-dependent N-band rotation; separate non-profile build')
    run.add_argument('--world', type=int, choices=(4, 8), default=8, help='fused smoke rank count')
    sequence = run.add_mutually_exclusive_group()
    sequence.add_argument('--seq-local', type=int, help='fused smoke rows per rank (default 256)')
    sequence.add_argument('--global-seq', type=int, help='fused smoke global rows, divisible by world')
    run.add_argument('--hidden', type=int, default=1024, help='fused smoke hidden width')
    run.add_argument('--q-heads', type=int, default=32, help='fused smoke Q head count')
    run.add_argument('--kv-heads', type=int, default=8, help='fused smoke KV head count')
    run.add_argument('--head-dim', type=int, default=128, help='fused smoke head width')
    run.add_argument('--timeout-seconds', type=int, default=60,
                     help='fused smoke process-local watchdog; --timeout remains the controller job limit')
    communication = run.add_mutually_exclusive_group()
    communication.add_argument('--comm-sm', type=int, help='fused smoke explicit communication CTA count (default 8)')
    communication.add_argument('--comm-sm-list', help='fused smoke: explicit same-process CTA candidates')
    communication.add_argument('--auto-oproj-comm', action='store_true',
                               help='OProj Graph: exercise runtime automatic CTA selection with explicit GEMM layout')
    run.add_argument('--auto-mxfp8-comm', action='store_true',
                     help='MXFP8 Graph QKV/OProj: runtime auto CTA candidate, optionally paired with --comm-sm/list')
    run.add_argument('--mxfp8-service-probe', action='store_true',
                     help='MXFP8 single-process CP4/8 profile: independent C/Q/R/QR services; GPU0-only detail artifact')
    tiles = run.add_mutually_exclusive_group()
    tiles.add_argument('--qkv-policy', choices=QKV_POLICIES, help='fused smoke tile (default auto)')
    tiles.add_argument('--qkv-policy-list', help='fused smoke: explicit same-process QKV tile candidates')
    oproj_tiles = run.add_mutually_exclusive_group()
    oproj_tiles.add_argument('--oproj-policy', choices=OPROJ_POLICIES,
                            help='fused smoke A2A tile (default auto=N128)')
    oproj_tiles.add_argument('--oproj-policy-list', help='fused smoke: independent OProj tile candidates')
    run.add_argument('--oproj-comm-layout', choices=('rows', 'columns'), default='rows',
                     help='fused smoke: one OProj communication layout per run, independent of GEMM tile policy')
    run.add_argument('--oproj-m-window-tiles', type=int, default=0,
                     help='MXFP8 OProj: bounded M tile window; requires --oproj-n-group-tiles and explicit comm CTAs')
    run.add_argument('--oproj-n-group-tiles', type=int, default=0,
                     help='MXFP8 OProj: N tile group inside each M window; both window dimensions default to disabled')
    run.add_argument('--causal', action='store_true', help='fused smoke: OProj two-chunk gather')
    run.add_argument('--cpu-oracle', action='store_true', help='fused smoke: small-shape CPU/GPU full-validation cross-check')
    run.add_argument('--validation-self-test', action='store_true', help='fused smoke: small-shape corruption/NaN detection and restoration checks')
    run.add_argument('--input-generator', choices=('cpu_mt19937', 'gpu_philox'), default='cpu_mt19937',
                     help='fused smoke: explicit random input/statistics preparation; not a timed boundary change')
    run.add_argument('--host-launch', choices=('sequential', 'per_gpu_thread'),
                     help='fused smoke: explicit host enqueue experiment; preserve full per-rank CUDA-event boundary')
    run.add_argument('--calibrate', action='store_true',
                     help='fused smoke: append independently validated compute/copy calibration for each explicit candidate')
    run.add_argument('--compute-only', action='store_true',
                     help='OProj calibration: measure only independent GEMM, never launch fused/copy kernels')
    run.add_argument('--quick', action='store_true',
                     help='fused-smoke screening: 1 warmup + 5 samples, no convergence/retries; not formal data')
    run.add_argument('--fused-counters', choices=('fused', 'compute_reference', 'copy_reference'),
                     help='single-process calibrated diagnostic: one warmed app-range on GPU0, all peers active; no performance samples')
    run.add_argument('--fused-counter-tool', choices=('ncu', 'nsys'), default='ncu',
                     help='explicit diagnostic tool: NCU app-range or Nsight Systems temporal GPU0 metrics')
    run.add_argument('--fused-counter-replay', choices=('app-range', 'range', 'application'), default='app-range',
                     help='NCU concurrent-range replay strategy; neither serializes individual kernels')
    run.add_argument('--max-swizzle-size', type=int, choices=(1, 2, 4, 8), default=1,
                     help='fused smoke: one explicit GEMM scheduling value shared by F/C, not a candidate grid')
    run.add_argument('--qkv-raster', choices=('heuristic', 'along_m', 'along_n'), default='heuristic',
                     help='fused smoke: QKV work traversal (heuristic preserves AlongM)')
    run.add_argument('--oproj-raster', choices=('heuristic', 'along_m', 'along_n'), default='heuristic',
                     help='fused smoke: OProj work traversal (heuristic preserves AlongN)')
    run.add_argument('--experiment', type=identifier)
    run.add_argument('--winners', help='baseline-replay: local verified complete-group winners.json; all supplied rows replayed')
    run.add_argument('--gemm-matrix', help='gemm-probe: explicit local MNK matrix JSON, reused within one GPU process')
    run.add_argument('--mxfp8-gemm-search', action='store_true',
                     help='build/gemm-probe: isolated CUTLASS grid plus top-2 neighbor search, no fused changes')
    run.add_argument('--gemm-precision', choices=('bf16', 'mxfp8'), default='bf16',
                     help='Pure GEMM probe: MXFP8 uses the no-Torch native matrix runner')
    run.add_argument('--gemm-operand-layout', choices=('nt', 'nn', 'tn'), default='nt',
                     help='gemm-probe: forward NT, backward dgrad NN or wgrad TN zero-copy operands')
    run.add_argument('--gemm-candidates', type=int,
                     help='gemm-probe: explicit cuBLASLt candidate limit (default unchanged)')
    run.add_argument('--compare-cutlass', action='store_true',
                     help='gemm-probe BF16/eager matrix: independent stock 1-SM/2-SM/Lt diagnostic, not fused reference')
    run.add_argument('--cutlass-swizzle-size', type=int, choices=(1, 2, 4, 8),
                     help='explicit runtime swizzle for both stock CUTLASS plans; requires --compare-cutlass')
    run.add_argument('--cutlass-epilogue-n', type=int, choices=(32, 64),
                     help='explicit epilogue N for stock CUTLASS plans; requires --compare-cutlass')
    run.add_argument('--cutlass-1sm-cluster-m', type=int, choices=(1, 2),
                     help='stock 1-SM MMA cluster M only (M2 requires E32); leaves 2-SM/Lt unchanged; requires --compare-cutlass')
    run.add_argument('--cutlass-sm-budget', type=int,
                     help='explicit CUTLASS persistent CTA budget, not an SM affinity partition')
    run.add_argument('--cublaslt-sm-target', type=int,
                     help='Graph gemm-probe: Lt SM heuristic target (0=full device), saves actual launch graph')
    run.add_argument('--gemm-sm-budget', type=int,
                     help='Graph gemm-probe: exact process-local green-context SM budget')
    run.add_argument('--cutlass-full-check', action='store_true',
                     help='explicit small CUTLASS comparison: full output checks, tensors limited to 4194304 elements')
    run.add_argument('--cutlass-counters', action='store_true',
                     help='single-geometry 1-SM NCU diagnostic only, never a performance sample')
    run.add_argument('--cublaslt-counters', action='store_true',
                     help='single-geometry tuned Graph NCU diagnostic, not a benchmark result')
    run.add_argument('--oproj-gap-probe', action='store_true',
                     help='full independent compute CTA tile-gap coverage, compact profile diagnostic only')
    run.add_argument('--oproj-layout', choices=('legacy', 'causal_dual_chunk_v1'), default='legacy',
                     help='baseline-replay: explicit inverse-A2A physical input contract')
    run.add_argument('--models', default='production_qwen_dense')
    run.add_argument('--seqs', default='16384')
    run.add_argument('--cps', default='8')
    run.add_argument('--devices', default='0,1,2,3,4,5,6,7')
    run.add_argument('--backends', default='cublaslt_nccl,te_ub')
    run.add_argument('--directions', default='qkv,oproj')
    run.add_argument('--launches', default='eager,graph')
    run.add_argument('--job-timeout', type=int, default=600)
    run.add_argument('--executor', choices=('batch', 'serial'), default='batch')
    run.add_argument('--reuse-experiment', type=identifier,
                     help='import validated same-fingerprint/stage measurements into a wider plan')
    run.add_argument('--timeout', type=int, default=14400)
    run.add_argument('--no-wait', action='store_true')
    run.add_argument('--dry-run', action='store_true')
    for name in ('status', 'watch', 'fetch', 'resume', 'clean'):
        s = sub.add_parser(name)
        s.add_argument('run_id', type=identifier)
        if name == 'clean':
            s.add_argument('--execute', action='store_true')
        if name == 'status':
            s.add_argument('--tail', action='store_true')
    internal = sub.add_parser('remote', help=argparse.SUPPRESS)
    internal.add_argument('job_path')
    args = p.parse_args()
    if args.action == 'run':
        # argparse can ignore an explicit value identical to its default when
        # checking mutually exclusive actions (e.g. interned int 8). Resolve
        # defaults only after parsing, so --comm-sm 8 still conflicts with list.
        if args.comm_sm is None:
            args.comm_sm = 0 if (args.auto_oproj_comm or args.auto_mxfp8_comm) else 8
        if args.qkv_policy is None:
            args.qkv_policy = 'auto'
        if args.oproj_policy is None:
            args.oproj_policy = 'auto'
        args.host_launch_explicit = args.host_launch is not None
        if args.host_launch is None:
            args.host_launch = 'sequential'
    if args.action == 'remote':
        return remote(args.job_path)
    if args.action == 'run':
        if args.workspace is not None:
            configure_workspace(args.workspace)
        validate_job(vars(args))
        backward_payload = None
        if args.backward_matrix:
            backward_payload=json.loads(Path(args.backward_matrix).read_text())
            if not isinstance(backward_payload,list) or not 1<=len(backward_payload)<=256:
                raise ValueError('Backward matrix must contain 1..256 cases')
            seen=set()
            for row in backward_payload:
                if not re.fullmatch(r'[a-zA-Z0-9_.-]{1,160}',row.get('id','')) or row['id'] in seen:
                    raise ValueError('Invalid or duplicate backward case ID')
                seen.add(row['id'])
                if row.get('direction') not in ('qkv','oproj'):
                    raise ValueError('Invalid backward direction')
                if (row.get('tile_n',128) not in (128,256) or row.get('epilogue_n',0) not in (0,64) or
                    (row.get('epilogue_n',0)==64 and row.get('tile_n',128)!=256) or
                    row.get('swizzle',1) not in (1,2,4,8) or type(row.get('along_m',False)) is not bool):
                    raise ValueError('Invalid backward GEMM tuning')
                if any(type(row.get(k)) is not int or not 0<row[k]<1<<30 for k in ('m','hidden','q_heads','kv_heads','head_dim')):
                    raise ValueError('Invalid backward geometry')
                fused_geometry(vars(args)|dict(seq_local=row['m'],global_seq=None,
                    fused_direction=row['direction'],**{k:row[k] for k in ('hidden','q_heads','kv_heads','head_dim')}))
        replay_payload = load_replay_winners(args.winners, args.devices) if args.stage == 'baseline-replay' else None
        gemm_payload = (validate_gemm_matrix(json.loads(Path(args.gemm_matrix).read_text()))
                        if args.gemm_matrix else None)
        if args.stage in ('sweep', 'refine', 'formal', 'summary') and not args.experiment:
            p.error('Tuning stages require --experiment NAME shared across stages')
        run_id = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
        folder = LOCAL / run_id
        folder.mkdir(parents=True)
        started = time.perf_counter()
        if args.source_run:
            archive, source_id, files, git_head = frozen_source_package(folder, args.source_run)
        else:
            archive, source_id, files = source_package(folder)
            git_head = read_command(['git', '-C', REPO, 'rev-parse', 'HEAD'])
        timings = {'pack_s': time.perf_counter() - started}
        job = vars(args) | dict(run_id=run_id, experiment=args.experiment or run_id,
                               source_id=source_id, archive_sha256=sha(archive), files=files,
                               git_head=git_head)
        if replay_payload is not None:
            job.pop('winners', None)  # Mac path is not a remote input; embed verified portable data.
            job['baseline_replay'] = replay_payload
        if gemm_payload is not None:
            job.pop('gemm_matrix', None)
            job['gemm_matrix_payload'] = gemm_payload
        if backward_payload is not None:
            job.pop('backward_matrix',None)
            job['backward_matrix_payload']=backward_payload
        write_json(folder / 'job.json', job)
        shutil.copy2(__file__, folder / 'runner.py')
        print(json.dumps({k: job[k] for k in ('run_id', 'node', 'stage', 'experiment', 'source_id')}, indent=2))
        print(f'Source package: {archive.stat().st_size} bytes, {len(files)} files', flush=True)
        if args.dry_run:
            write_json(folder / 'timings.json', timings)
            return 0
        checked = time.perf_counter()
        screen_ready(args.node)
        timings['screen_check_s'] = time.perf_counter() - checked
        upload_started = time.perf_counter()
        for name in ('source.tar.gz', 'job.json', 'runner.py'):
            mc_copy(folder / name, prefix(run_id) + '/' + name)
        timings['upload_s'] = time.perf_counter() - upload_started
        submitted = time.perf_counter()
        submit(job)
        timings['submit_s'] = time.perf_counter() - submitted
        write_json(folder / 'timings.json', timings)
        return 0 if args.no_wait else wait_for_run(run_id)
    if args.action == 'status':
        print(json.dumps(get_status(args.run_id), indent=2, ensure_ascii=False))
        if args.tail:
            subprocess.run(['mc', 'cat', prefix(args.run_id) + '/tail.txt'])
    elif args.action == 'watch':
        return wait_for_run(args.run_id)
    elif args.action == 'fetch':
        fetch(args.run_id)
    elif args.action == 'resume':
        job = json.loads((LOCAL / args.run_id / 'job.json').read_text())
        previous = get_status(args.run_id)
        if previous['state'] == 'running':
            return wait_for_run(args.run_id)
        if previous['state'] == 'succeeded':
            fetch(args.run_id)
            return 0
        submit(job)
        return wait_for_run(args.run_id, minimum_attempt=previous.get('attempt', 0) + 1)
    elif args.action == 'clean':
        receipt = LOCAL / args.run_id / 'fetched.json'
        if not receipt.exists():
            raise RuntimeError('Fetch and verify artifacts before cleaning transfers')
        state = json.loads(receipt.read_text())
        if state['state'] != 'succeeded':
            raise RuntimeError('Keep failed-run transfer packages for diagnosis/resume')
        objects = [prefix(args.run_id) + '/' + name for name in
                   ('source.tar.gz', 'job.json', 'runner.py', 'status.json', 'tail.txt')]
        print('\n'.join(objects))
        if args.execute:
            for obj in objects:
                result = subprocess.run(['mc', 'rm', obj], capture_output=True, text=True, timeout=60)
                if result.returncode and 'object does not exist' not in result.stderr.lower():
                    raise RuntimeError(f'Cleanup failed for {obj}: {result.stderr.strip()}')
                if result.returncode == 0:
                    print(result.stdout.strip())
            write_json(LOCAL / args.run_id / 'cleaned.json', {'objects': objects, 'time': time.time()})
            print('Removed only this run transfer objects; local copies and cloud artifacts retained.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
