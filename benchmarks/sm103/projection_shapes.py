#!/usr/bin/env python3
"""Pinned attention projections; CPU-only catalog and pure-GEMM matrix export.

Dimensions are per token before TP: Y[M,N] = X[M,K] W[N,K]^T.
M=S/CP is a token-shard GEMM probe, not an assertion that an arbitrary
projection supports the existing Ulysses all-to-all layout. No remote code runs.
"""
import argparse
import json


# name: (repository, revision). Retain provenance without copying model code.
SOURCES = {
    'qwen35_397b': ('Qwen/Qwen3.5-397B-A17B', '8472618112abcbd45acbcdc58436aff4233c23f7'),
    'glm5': ('zai-org/GLM-5', 'c183ef8c61faee82855eca1ed9bb3a9a7ce3b0b2'),
    'qwen25_72b': ('Qwen/Qwen2.5-72B', 'efba10c8e54e91e0d9570ab5f7b51a958474d4cb'),
    'qwen3_235b': ('Qwen/Qwen3-235B-A22B', '8efa61729e24bd65b1d152b5ab5409052aa80e65'),
    'bloom_176b': ('bigscience/bloom', '7f10a99ce7c08f03c7719a586cb2cbda1433ac05'),
    'deepseek_v3': ('deepseek-ai/DeepSeek-V3', 'e815299b0bcbac849fa540c768ef21845365c9eb'),
    'kimi_k3': ('moonshotai/Kimi-K3', 'f831ab66814297da540d832a5235f8e904f29d06'),
    'kimi_linear_48b': ('moonshotai/Kimi-Linear-48B-A3B-Instruct', 'e1df551a447157d4658b573f9a695d57658590e9'),
}

# Only conventional equal-Q/K/V-head-dimension projections enter the existing
# boundary planner. MHA is represented honestly as KV heads == Q heads.
# The benchmark is bias-free BF16 geometry, not full checkpoint inference.
# TODO(sm103-segmented-routing): extend KDA/MLA (and gated projections) with
# explicit segment offsets and ownership, not synthetic Q/KV head counts.
# Pack only projections sharing the exact input; head-owned segments scatter
# to their head rank, while shared latent segments replicate to their consumers.
# Decide the MLA exchange boundary relative to normalization/up-projection first;
# transmitting latents and transmitting expanded K/V are different benchmarks.
# Forward/backward must share grouping and layout metadata; the adjoint of
# replication sums gradients. Non-divisible heads require explicit ragged ranges,
# not silent padding. Validate full routes/numerics before publishing fused rows.
# Deferred: this campaign measures large/long-sequence throughput of existing
# routes only; grouped pure-GEMM results do not prove a fused adapter exists.
BOUNDARY_MODELS = {
    'qwen25_72b': (8192, 64, 8, 128),
    'qwen3_235b': (4096, 64, 4, 128),
    'bloom_176b': (14336, 112, 112, 128),
    # KDA Q/K/V are equal-width Linear projections. Pack only these three
    # before canonical token/head exchange; convolution, gates and recurrence
    # are separate operations, not secretly included in this boundary score.
    'kimi_k3_kda': (7168, 96, 96, 128),
    'kimi_linear_48b_kda': (2304, 32, 32, 128),
}
LONG_SEQUENCES = (65536, 131072, 262144, 524288)


def projections(model):
    """Return actual Linear geometries, preserving repeated-operation labels.

    Do not combine low-rank stages across RMSNorm/activation. Q/K/V are
    separate probe labels for separate Linear operations; BLOOM retains its
    native packed QKV. Candidate packing belongs to the boundary benchmark.
    """
    if model not in SOURCES:
        raise ValueError(f'Unknown projection model: {model}')
    rows = []

    def add(name, n, k):
        rows.append(dict(id=name, n=n, k=k))

    def mla(prefix, h, heads, q_rank, nope=128, value=128):
        if q_rank is None:
            add(prefix + 'q', heads * (nope + 64), h)
        else:
            add(prefix + 'q_a', q_rank, h)
            add(prefix + 'q_b', heads * (nope + 64), q_rank)
        add(prefix + 'kv_a', 512 + 64, h)
        add(prefix + 'kv_b', heads * (nope + value), 512)
        add(prefix + 'o', h, heads * value)

    if model in BOUNDARY_MODELS:
        h, q, kv, dim = BOUNDARY_MODELS[model]
        if model == 'bloom_176b':
            add('qkv_native_packed', (q + 2 * kv) * dim, h)
        else:
            add('q', q * dim, h)
            add('k', kv * dim, h)
            add('v', kv * dim, h)
        add('o', h, q * dim)
    elif model == 'qwen35_397b':
        # Gated DeltaNet has unequal key/value head counts, not ordinary GQA.
        add('gdn_qkv', 2 * 16 * 128 + 64 * 128, 4096)
        add('gdn_z', 64 * 128, 4096)
        add('gdn_beta', 64, 4096)
        add('gdn_a', 64, 4096)
        add('gdn_o', 4096, 64 * 128)
        # Native full-attention q_proj includes the output gate (factor two).
        add('full_q_gate', 32 * 256 * 2, 4096)
        add('full_k', 2 * 256, 4096)
        add('full_v', 2 * 256, 4096)
        add('full_o', 4096, 32 * 256)
    elif model == 'glm5':
        mla('mla_', 6144, 64, 2048, nope=192, value=256)
        add('index_q', 32 * 128, 2048)
        add('index_k', 128, 6144)
        add('index_weights', 32, 6144)
    elif model == 'deepseek_v3':
        mla('mla_', 7168, 128, 1536)
    elif model in ('kimi_k3', 'kimi_linear_48b'):
        h, heads = (7168, 96) if model == 'kimi_k3' else (2304, 32)
        width = heads * 128
        for name in ('q', 'k', 'v'):
            add('kda_' + name, width, h)
        add('kda_f_a', 128, h)
        add('kda_f_b', width, 128)
        add('kda_beta', heads, h)
        if model == 'kimi_k3':
            add('kda_g', width, h)
        else:
            add('kda_g_a', 128, h)
            add('kda_g_b', width, 128)
        add('kda_o', h, width)
        mla('mla_', h, heads, 1536 if model == 'kimi_k3' else None)
        if model == 'kimi_k3':
            add('mla_g', width, h)
    else:
        raise ValueError(f'No verified projection mapping for {model}')
    return rows


def grouped_projections(model):
    """Concatenate weights only for projections sharing the exact input tensor.

    Downstream low-rank projections remain separate even when their K matches:
    f_a(x) and g_a(x), or normalized Q/KV latents, are different tensors.
    This describes a legal packed GEMM, not an implemented communication layout.
    """
    native = projections(model)
    groups = []
    if model in ('qwen25_72b', 'qwen3_235b'):
        groups = [('qkv_packed', ('q', 'k', 'v'))]
    elif model in ('kimi_k3', 'kimi_linear_48b'):
        gate = 'kda_g' if model == 'kimi_k3' else 'kda_g_a'
        groups = [('kda_input_packed', ('kda_q', 'kda_k', 'kda_v', 'kda_f_a', 'kda_beta', gate))]
        mla = ('mla_q_a', 'mla_kv_a', 'mla_g') if model == 'kimi_k3' else ('mla_q', 'mla_kv_a')
        groups.append(('mla_input_packed', mla))
    elif model in ('deepseek_v3', 'glm5'):
        groups = [('mla_input_packed', ('mla_q_a', 'mla_kv_a'))]
    elif model == 'qwen35_397b':
        groups = [('gdn_input_packed', ('gdn_qkv', 'gdn_z', 'gdn_beta', 'gdn_a')),
                  ('full_input_packed', ('full_q_gate', 'full_k', 'full_v'))]
    by_name = {p['id']: p for p in native}
    consumed, packed = set(), []
    for name, members in groups:
        parts = [by_name[x] for x in members]
        if len({p['k'] for p in parts}) != 1 or consumed.intersection(members):
            raise ValueError('Invalid shared-input projection group')
        consumed.update(members)
        packed.append(dict(id=name, n=sum(p['n'] for p in parts), k=parts[0]['k']))
    return packed + [p for p in native if p['id'] not in consumed]


def matrix(models, sequences=LONG_SEQUENCES, cps=(4, 8), *, grouped=False):
    rows = []
    if not models or len(set(models)) != len(models):
        raise ValueError('Select nonempty, unique model names')
    if not sequences or len(set(sequences)) != len(sequences):
        raise ValueError('Select nonempty, unique sequence lengths')
    if not cps or len(set(cps)) != len(cps) or any(cp not in (4, 8) for cp in cps):
        raise ValueError('CP must be 4 and/or 8')
    for model in models:
        for seq in sequences:
            if type(seq) is not int or seq <= 0 or any(seq % cp for cp in cps):
                raise ValueError('Sequence must be positive and divisible by CP')
            for cp in cps:
                for p in (grouped_projections(model) if grouped else projections(model)):
                    rows.append(dict(id=f'{model}.{p["id"]}.s{seq}.cp{cp}',
                                     m=seq // cp, n=p['n'], k=p['k']))
    if len(rows) > 256:
        raise ValueError('Existing GEMM runner accepts at most 256 rows; split by model')
    return dict(schema='sm103_gemm_matrix_v1', shapes=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', required=True, help='Comma-separated names from SOURCES')
    parser.add_argument('--seqs', default=','.join(map(str, LONG_SEQUENCES)))
    parser.add_argument('--cps', default='4,8')
    parser.add_argument('--grouped', action='store_true', help='Pack verified shared-input projections')
    args = parser.parse_args()
    print(json.dumps(matrix(args.models.split(','), tuple(map(int, args.seqs.split(','))),
                            tuple(map(int, args.cps.split(','))), grouped=args.grouped), indent=2))
