#!/usr/bin/env python3
"""BF16 EP grouped-GEMM catalog and reproducible, feasible routing workload.

No model loading, CUDA, remote code, or measurement. M is received rows per
expert, not sequence length or original tokens per rank. FC1 includes packed
gate/up; FC2 returns branch outputs, without top-k weighting or reduction.
"""

import argparse
import hashlib
import json
import random


TOKEN_COUNTS = (1, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512,
                1024, 2048, 4096, 8192)

# H, expert F, routed E, top-k. Shared experts are outside this boundary.
MODELS = {
    'deepseek_v41_flash': (5120, 2304, 384, 6),
    'deepseek_v4_flash': (4096, 2048, 256, 6),
    'deepseek_v4_pro': (7168, 3072, 384, 6),
    'deepseek_v32': (7168, 2048, 256, 8),
    'glm5': (6144, 2048, 256, 8),
    'glm52': (6144, 2048, 256, 8),
    'glm53': (6144, 2048, 256, 8),
    'glm53_flash': (4096, 2048, 288, 8),
    'glm47': (5120, 1536, 160, 8),
    'glm45': (5120, 1536, 160, 8),
    'glm45_air': (4096, 1408, 128, 8),
    'glm47_flash': (2048, 1536, 64, 4),
    'qwen35_397b': (4096, 1024, 512, 10),
    'qwen35_122b': (3072, 1024, 256, 8),
    'qwen3_235b': (4096, 1536, 128, 8),
    'kimi_k25': (7168, 2048, 384, 8),
    'mimo_v2_flash': (4096, 2048, 256, 8),
    'mimo_v25': (4096, 2048, 256, 8),
    'mimo_v25_pro': (6144, 2048, 384, 8),
    'mixtral_8x7b': (4096, 14336, 8, 2),
    'mixtral_8x22b': (6144, 16384, 8, 2),
}

SOURCES = {
    'deepseek_v41_flash': ('deepseek-ai/DeepSeek-V4.1-Flash', 'dba1be0a40aa45a94ad051997016db3960a90277'),
    'deepseek_v4_flash': ('deepseek-ai/DeepSeek-V4-Flash', '60d8d70770c6776ff598c94bb586a859a38244f1'),
    'deepseek_v4_pro': ('deepseek-ai/DeepSeek-V4-Pro', 'b5968e9190ef611bbf34a7229255be88a0e937c1'),
    'deepseek_v32': ('deepseek-ai/DeepSeek-V3.2', 'a7e62ac04ecb2c0a54d736dc46601c5606cf10a6'),
    'glm5': ('zai-org/GLM-5', 'c183ef8c61faee82855eca1ed9bb3a9a7ce3b0b2'),
    'glm52': ('zai-org/GLM-5.2', 'cf457fa734ab149ffef225f80893eb38c6ff5cdc'),
    'glm53': ('zai-org/GLM-5.3', 'aca966e4e02791568aa6a4ced368624b3d897f42'),
    'glm53_flash': ('zai-org/GLM-5.3-Flash', 'eb9eb208eb0d988989d07a6a12d0fdeb5f52574a'),
    'glm47': ('zai-org/GLM-4.7', '602d01efcdd332c5238ca4bcede555defbe83eb7'),
    'glm45': ('zai-org/GLM-4.5', 'cbb2c7cfb52fa128a9660cb1a7a78e017899e115'),
    'glm45_air': ('zai-org/GLM-4.5-Air', 'a24ceef6ce4f3536971efe9b778bdaa1bab18daa'),
    'glm47_flash': ('zai-org/GLM-4.7-Flash', '7dd20894a642a0aa287e9827cb1a1f7f91386b67'),
    'qwen35_397b': ('Qwen/Qwen3.5-397B-A17B', '8472618112abcbd45acbcdc58436aff4233c23f7'),
    'qwen35_122b': ('Qwen/Qwen3.5-122B-A10B', 'dc4d348443bc740c68e2d77492492c11606384d5'),
    'qwen3_235b': ('Qwen/Qwen3-235B-A22B', '8efa61729e24bd65b1d152b5ab5409052aa80e65'),
    'kimi_k25': ('moonshotai/Kimi-K2.5', '4d01dfe0332d63057c186e0b262165819efb6611'),
    'mimo_v2_flash': ('XiaomiMiMo/MiMo-V2-Flash', '1afd314a2406c282e0956375c34a676501c78649'),
    'mimo_v25': ('XiaomiMiMo/MiMo-V2.5', '63651580ca774f8504f676040460aed3e1244ac1'),
    'mimo_v25_pro': ('XiaomiMiMo/MiMo-V2.5-Pro', '21d1ecfecd7bd70f31be25ca49d7edd21f003659'),
    'mixtral_8x7b': ('mistralai/Mixtral-8x7B-v0.1', 'fc7ac94680e38d7348cfa806e51218e6273104b0'),
    'mixtral_8x22b': ('mistralai/Mixtral-8x22B-v0.1', 'e1cd34ff1747406fb2277635ed25242803009bc2'),
}


def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def balanced_route(experts, topk, ep, target_rows, seed=20260916):
    """Compact, invertible route; each token selects top-k DISTINCT experts.

    Enumerate logical branches b = token * topk + slot. An expert permutation
    maps b % E to the destination expert; b // E is its row. Integer rounding
    happens in ORIGINAL token count, never in expert padding or dropped branches.
    Store the actual permutation, so reproducibility does not rely on future
    Python PRNG implementation details.
    """
    if ep not in (4, 8) or experts <= 0 or experts % ep:
        raise ValueError('EP must be 4/8 and divide the routed expert count')
    if not 0 < topk <= experts or target_rows < 0:
        raise ValueError('invalid top-k or per-expert row target')
    tokens = ceil_div(experts * target_rows, ep * topk)
    branches = ep * tokens * topk
    order = list(range(experts))
    random.Random(seed).shuffle(order)
    inverse = [0] * experts
    counts = [0] * experts
    for residue, expert in enumerate(order):
        inverse[expert] = residue
        counts[expert] = branches // experts + (residue < branches % experts)
    local_experts = experts // ep
    rank_rows = [sum(counts[r * local_experts:(r + 1) * local_experts])
                 for r in range(ep)]
    return dict(protocol='cyclic-distinct-topk-v1', ep=ep, experts=experts,
                topk=topk, tokens_per_rank=tokens, target_rows=target_rows,
                branch_count=branches, expert_order=order,
                expert_residue=inverse, expert_counts=counts, rank_rows=rank_rows)


def source_branch(route, expert, row):
    """Destination (global expert, expert row) -> source (rank, token, slot)."""
    if not 0 <= expert < route['experts'] or not 0 <= row < route['expert_counts'][expert]:
        raise ValueError('expert row outside the actual routed workload')
    branch = row * route['experts'] + route['expert_residue'][expert]
    global_token, slot = divmod(branch, route['topk'])
    rank, token = divmod(global_token, route['tokens_per_rank'])
    return rank, token, slot


def destination_branch(route, rank, token, slot):
    """Source (rank, token, slot) -> destination (global expert, expert row)."""
    if not (0 <= rank < route['ep'] and 0 <= token < route['tokens_per_rank']
            and 0 <= slot < route['topk']):
        raise ValueError('source branch outside the actual routed workload')
    branch = (rank * route['tokens_per_rank'] + token) * route['topk'] + slot
    row, residue = divmod(branch, route['experts'])
    return route['expert_order'][residue], row


def geometry(model, direction):
    h, f, experts, topk = MODELS[model]
    if direction == 'dispatch':
        n, k = 2 * f, h
    elif direction == 'combine':
        n, k = h, f
    else:
        raise ValueError('direction must be dispatch or combine')
    return n, k, experts, topk


def cases(models=None, token_counts=TOKEN_COUNTS, eps=(4, 8),
          directions=('dispatch', 'combine')):
    """Deduplicate identical measured work, while retaining every logical label.

    The route is data, NOT a kernel specialization or runtime routing restriction.
    Real fused operators must also accept arbitrary valid expert-row mappings.
    """
    models = list(MODELS) if models is None else list(models)
    if not models or not token_counts or not eps or not directions:
        raise ValueError('empty benchmark selection')
    if any(m not in MODELS for m in models):
        raise ValueError('unknown model')
    if any(m <= 0 for m in token_counts):
        raise ValueError('zero experts are correctness cases, not FLOP benchmarks')
    physical = {}
    for model in models:
        for direction in directions:
            n, k, experts, topk = geometry(model, direction)
            for ep in eps:
                for target in token_counts:
                    route = balanced_route(experts, topk, ep, target)
                    # Target and model name are NOT physical properties.
                    key = (direction, n, k, experts, topk, ep,
                           route['tokens_per_rank'])
                    if key not in physical:
                        manifest = dict(route)
                        manifest.pop('target_rows')
                        payload = json.dumps(manifest, sort_keys=True,
                                             separators=(',', ':')).encode()
                        physical[key] = dict(
                            id=f'{direction}-n{n}-k{k}-e{experts}-top{topk}'
                               f'-ep{ep}-t{route["tokens_per_rank"]}',
                            direction=direction, n=n, k=k, route=manifest,
                            route_sha256=hashlib.sha256(payload).hexdigest(),
                            aliases=[], effective_flops=2 * route['branch_count'] * n * k,
                            status='not_measured', precision='bf16', schedule='ctasp')
                    physical[key]['aliases'].append(dict(model=model, target_rows=target))
    return list(physical.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=tuple(MODELS))
    parser.add_argument('--tokens', nargs='+', type=int, default=TOKEN_COUNTS)
    parser.add_argument('--ep', nargs='+', type=int, choices=(4, 8), default=(4, 8))
    parser.add_argument('--direction', nargs='+', choices=('dispatch', 'combine'),
                        default=('dispatch', 'combine'))
    parser.add_argument('--json', action='store_true', help='emit the full CPU plan')
    args = parser.parse_args()
    rows = cases(args.models, args.tokens, args.ep, args.direction)
    if args.json:
        print(json.dumps(dict(schema='grouped-bf16-plan-v1', cases=rows), indent=2))
    else:
        print(f'{len(rows)} physical cases; '
              f'{sum(len(r["aliases"]) for r in rows)} logical model/token cases; '
              'CTASP; reference=tuned-pure-cuBLAS; not measured')


if __name__ == '__main__':
    main()
