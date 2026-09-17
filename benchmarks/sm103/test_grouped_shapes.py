"""CPU-only shape/route tests; no CUDA correctness or performance claim."""

from collections import Counter
from pathlib import Path
import re
import unittest

import grouped_shapes as g


class GroupedShapes(unittest.TestCase):
    def test_catalog_matches_frozen_configs(self):
        # Prevent drift between the reviewed, source-linked catalog and planner.
        doc = (Path(__file__).parent / 'GROUPED_GEMM_STUDY.md').read_text()
        rows = re.findall(r'https://huggingface.co/([^/]+/[^/]+)/blob/([a-f0-9]{40})/config.json\)'
                          r' \| (\d+) \| (\d+) \| (\d+) \| (\d+) \|', doc)
        self.assertEqual(len(rows), 21)
        known = {(repo, rev): tuple(map(int, dims)) for repo, rev, *dims in rows}
        self.assertEqual(set(g.MODELS), set(g.SOURCES))
        for name, dims in g.MODELS.items():
            self.assertEqual(dims, known[g.SOURCES[name]])
        self.assertEqual(len(set(g.MODELS.values())), 17)

    def test_fc1_fc2_and_flops(self):
        for model, (h, f, experts, topk) in g.MODELS.items():
            self.assertEqual(g.geometry(model, 'dispatch'), (2*f, h, experts, topk))
            self.assertEqual(g.geometry(model, 'combine'), (h, f, experts, topk))
            rows = g.cases([model], (96,), (8,))
            self.assertEqual(rows[0]['effective_flops'], 2*rows[1]['effective_flops'])
            for case in rows:
                self.assertEqual(case['effective_flops'],
                                 2*sum(case['route']['expert_counts'])*case['n']*case['k'])

    def test_every_catalog_token_is_covered(self):
        rows = g.cases()
        expected = {(m, t, ep, d) for m in g.MODELS for t in g.TOKEN_COUNTS
                    for ep in (4, 8) for d in ('dispatch', 'combine')}
        observed = [(a['model'], a['target_rows'], c['route']['ep'], c['direction'])
                    for c in rows for a in c['aliases']]
        self.assertEqual(set(observed), expected)
        self.assertEqual(len(observed), len(expected))
        self.assertEqual(len(rows), len({c['id'] for c in rows}))
        self.assertTrue(all(c['status'] == 'not_measured' for c in rows))

    def test_balanced_route_and_inverse(self):
        for experts, topk in ((8, 2), (64, 4), (128, 8), (288, 8), (384, 6), (512, 10)):
            for ep in (4, 8):
                for target in (0, 1, 8, 17):
                    r = g.balanced_route(experts, topk, ep, target)
                    counts = Counter()
                    seen = set()
                    for rank in range(ep):
                        for token in range(r['tokens_per_rank']):
                            choices = set()
                            for slot in range(topk):
                                expert, row = g.destination_branch(r, rank, token, slot)
                                self.assertNotIn(expert, choices)
                                self.assertNotIn((expert, row), seen)
                                choices.add(expert)
                                seen.add((expert, row))
                                counts[expert] += 1
                                self.assertEqual(g.source_branch(r, expert, row), (rank, token, slot))
                    self.assertEqual([counts[e] for e in range(experts)], r['expert_counts'])
                    self.assertEqual(len(seen), r['branch_count'])
                    self.assertEqual(sum(r['rank_rows']), r['branch_count'])
                    self.assertLessEqual(max(r['expert_counts'])-min(r['expert_counts']), 1)

    def test_large_counts_without_materializing_payload(self):
        for dims in set(g.MODELS.values()):
            for ep in (4, 8):
                r = g.balanced_route(dims[2], dims[3], ep, 8192)
                for expert, count in enumerate(r['expert_counts']):
                    for row in (0, count-1):
                        self.assertEqual(g.destination_branch(r, *g.source_branch(r, expert, row)),
                                         (expert, row))

    def test_dedup_requires_experts_and_topk(self):
        rows = g.cases(['glm5', 'glm52', 'glm53'], (128,), (8,), ('dispatch',))
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]['aliases']), 3)
        rows = g.cases(['deepseek_v32', 'kimi_k25'], (128,), (8,), ('dispatch',))
        self.assertEqual(len(rows), 2)
        rows = g.cases(['deepseek_v4_flash', 'mimo_v2_flash'], (128,), (8,), ('dispatch',))
        self.assertEqual(len(rows), 2)

    def test_seed_and_bounds(self):
        a = g.balanced_route(128, 8, 8, 128, 1)
        self.assertEqual(a, g.balanced_route(128, 8, 8, 128, 1))
        self.assertNotEqual(a['expert_order'], g.balanced_route(128, 8, 8, 128, 2)['expert_order'])
        for values in ((8, 9, 8, 1), (8, 0, 8, 1), (8, 2, 3, 1), (9, 2, 8, 1), (8, 2, 8, -1)):
            with self.assertRaises(ValueError):
                g.balanced_route(*values)
        for branch in ((-1, 0, 0), (8, 0, 0), (0, a['tokens_per_rank'], 0), (0, 0, 8)):
            with self.assertRaises(ValueError):
                g.destination_branch(a, *branch)
        with self.assertRaises(ValueError):
            g.source_branch(a, 0, a['expert_counts'][0])
        for selection in (dict(models=[]), dict(token_counts=(0,)), dict(directions=('invalid',))):
            with self.assertRaises(ValueError):
                g.cases(**selection)


if __name__ == '__main__':
    unittest.main()
