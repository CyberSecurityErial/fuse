import unittest
import gc
import itertools
import types
from unittest import mock
import measurement
from measurement import stable_window, collect_samples


class WarmupContracts(unittest.TestCase):
    def test_collection_primes_events_and_reduces_vector_once(self):
        events = []
        def event(**kwargs):
            item = mock.Mock()
            item.elapsed_time.return_value = 0.25
            events.append(item)
            return item
        dist = types.ModuleType('torch.distributed')
        dist.barrier = mock.Mock()
        dist.all_reduce = mock.Mock()
        dist.ReduceOp = types.SimpleNamespace(MAX='max')
        torch = types.ModuleType('torch')
        torch.distributed = dist
        torch.cuda = types.SimpleNamespace(Event=event, synchronize=mock.Mock())
        torch.float64 = 'float64'
        tensor = mock.Mock()
        tensor.tolist.return_value = [0.25] * 50
        torch.tensor = mock.Mock(return_value=tensor)
        fn = mock.Mock()
        original_gc = gc.isenabled()
        with mock.patch.dict('sys.modules', {'torch': torch, 'torch.distributed': dist}):
            self.assertEqual(collect_samples(fn, 50, 'device'), [0.25] * 50)
        self.assertEqual(fn.call_count, 50)
        self.assertEqual(dist.barrier.call_count, 50)
        self.assertTrue(all(item.record.call_count == 2 for item in events))
        torch.tensor.assert_called_once_with([0.25] * 50, dtype='float64', device='device')
        dist.all_reduce.assert_called_once_with(tensor, op='max')
        self.assertEqual(gc.isenabled(), original_gc)

    def test_convergence_requires_three_positive_finite_close_windows(self):
        self.assertFalse(stable_window([1, 1]))
        self.assertFalse(stable_window([0, 0, 0]))
        self.assertFalse(stable_window([1, float('nan'), 1]))
        self.assertFalse(stable_window([1, 1, 1.1]))
        self.assertTrue(stable_window([8, 1, 1.01, 1.02]))


class OProjLayoutContracts(unittest.TestCase):
    """CPU-only routing proof; the oracle indexes the original global sequence."""

    @staticmethod
    def sources(world, batch, chunk_rows, local_heads, head_dim):
        # Mixed-radix IDs uniquely identify peer/batch/chunk/row/head/dimension.
        return [
            [
                [
                    [
                        (((((peer * batch + b) * (2 * world) + chunk)
                            * chunk_rows + row) * local_heads + head)
                          * head_dim + d) + 1
                        for head in range(local_heads) for d in range(head_dim)
                    ]
                    for chunk in range(2 * world) for row in range(chunk_rows)
                ]
                for b in range(batch)
            ]
            for peer in range(world)
        ]

    @staticmethod
    def route(sources, pairs, chunk_rows):
        # Pack two chunks for each destination, A2A exchanges source peers,
        # then unpack concatenates source-peer head slabs (not sequence rows).
        return [
            [
                [value for source in sources
                 for value in source[b][chunk * chunk_rows + row]]
                for b in range(len(sources[0]))
                for chunk in pair for row in range(chunk_rows)
            ]
            for pair in pairs
        ]

    @staticmethod
    def canonical(sources, chunk_rows):
        # Independent global-row definition: the first local half starts at
        # rank*chunk_rows; the second starts at S-(rank+1)*chunk_rows.
        seq = len(sources[0][0])
        return [
            [
                [value for source in sources
                 for value in source[b][
                     rank * chunk_rows + row if row < chunk_rows
                     else seq - (rank + 1) * chunk_rows + row - chunk_rows
                 ]]
                for b in range(len(sources[0])) for row in range(2 * chunk_rows)
            ]
            for rank in range(len(sources))
        ]

    @staticmethod
    def project(routed):
        # Exact integer GEMM, with nonconstant signed weights and three columns.
        width = len(routed[0][0])
        weights = [[((k * (column + 3) + column) % 13) - 6
                    for k in range(width)] for column in range(3)]
        return [[[sum(value * weight for value, weight in zip(row, column))
                  for column in weights] for row in rank] for rank in routed]

    @staticmethod
    def canonical_projection(sources, chunk_rows, local_heads, head_dim):
        # Reference GEMM addresses global rows and global heads directly. It
        # does not consume either chunk_pairs() or the emulated routed buffer.
        world = len(sources)
        seq = len(sources[0][0])
        output = []
        for rank in range(world):
            rank_output = []
            for b in range(len(sources[0])):
                for row in range(2 * chunk_rows):
                    global_row = (rank * chunk_rows + row if row < chunk_rows
                                  else seq - (rank + 1) * chunk_rows
                                  + row - chunk_rows)
                    values = [0, 0, 0]
                    for peer in range(world):
                        for head in range(local_heads):
                            for d in range(head_dim):
                                k = (peer * local_heads + head) * head_dim + d
                                value = sources[peer][b][global_row][head * head_dim + d]
                                for column in range(3):
                                    values[column] += value * (
                                        ((k * (column + 3) + column) % 13) - 6)
                    rank_output.append(values)
            output.append(rank_output)
        return output

    def test_explicit_legacy_and_canonical_pairs(self):
        golden = {
            4: {
                'te_ub': ((0, 7), (2, 5), (4, 3), (6, 1)),
                'cublaslt_nccl': ((0, 2), (4, 6), (7, 5), (3, 1)),
                'canonical': ((0, 7), (1, 6), (2, 5), (3, 4)),
            },
            8: {
                'te_ub': ((0, 15), (2, 13), (4, 11), (6, 9),
                          (8, 7), (10, 5), (12, 3), (14, 1)),
                'cublaslt_nccl': ((0, 2), (4, 6), (8, 10), (12, 14),
                                 (15, 13), (11, 9), (7, 5), (3, 1)),
                'canonical': ((0, 15), (1, 14), (2, 13), (3, 12),
                              (4, 11), (5, 10), (6, 9), (7, 8)),
            },
        }
        for world, expected in golden.items():
            for backend in ('te_ub', 'cublaslt_nccl'):
                with self.subTest(world=world, backend=backend):
                    self.assertEqual(measurement.oproj_chunk_pairs(world, backend),
                                     expected[backend])
                    self.assertEqual(measurement.oproj_chunk_pairs(world, backend, 'legacy'),
                                     expected[backend])
                    self.assertEqual(measurement.oproj_chunk_pairs(
                        world, backend, 'causal_dual_chunk_v1'), expected['canonical'])
        # The pure layout helper accepts all positive integer world sizes;
        # hardware/benchmark CP4/8 restrictions belong to their callers.
        for world, backend, layout in itertools.product(
                (1, 2, 3, 5), ('te_ub', 'cublaslt_nccl'),
                ('legacy', 'causal_dual_chunk_v1')):
            pairs = measurement.oproj_chunk_pairs(world, backend, layout)
            self.assertIsInstance(pairs, tuple)
            self.assertEqual(len(pairs), world)
            self.assertTrue(all(isinstance(pair, tuple) and len(pair) == 2 for pair in pairs))
            self.assertEqual(sorted(itertools.chain.from_iterable(pairs)), list(range(2 * world)))

    def test_unknown_layout_backend_and_invalid_world_rejected(self):
        for world in (0, -1, True, False, 4.0, '4', None):
            with self.subTest(world=world), self.assertRaises((ValueError, TypeError)):
                measurement.oproj_chunk_pairs(world, 'te_ub')
        for backend in ('', 'nccl', 'unknown', None):
            with self.subTest(backend=backend), self.assertRaises((ValueError, TypeError)):
                measurement.oproj_chunk_pairs(4, backend)
        for layout in ('', 'canonical', 'unknown', None):
            with self.subTest(layout=layout), self.assertRaises((ValueError, TypeError)):
                measurement.oproj_chunk_pairs(4, 'te_ub', layout)

    def test_canonical_marker_routes_and_integer_gemm_match_global_oracle(self):
        for world, batch, chunk_rows, local_heads, head_dim in itertools.product(
                (4, 8), (1, 2), (1, 3, 5), (1, 2, 3), (1, 2, 5)):
            with self.subTest(world=world, batch=batch, chunk_rows=chunk_rows,
                              local_heads=local_heads, head_dim=head_dim):
                sources = self.sources(world, batch, chunk_rows, local_heads, head_dim)
                expected = self.canonical(sources, chunk_rows)
                projected = self.canonical_projection(sources, chunk_rows, local_heads, head_dim)
                for backend in ('te_ub', 'cublaslt_nccl'):
                    pairs = measurement.oproj_chunk_pairs(world, backend, 'causal_dual_chunk_v1')
                    routed = self.route(sources, pairs, chunk_rows)
                    self.assertEqual(routed, expected)
                    self.assertEqual(self.project(routed), projected)

    def test_legacy_routes_are_not_canonical_on_unchanged_input(self):
        for world in (4, 8):
            sources = self.sources(world, 2, 3, 2, 5)
            expected = self.canonical(sources, 3)
            routes = {}
            for backend in ('te_ub', 'cublaslt_nccl'):
                routes[backend] = self.route(
                    sources, measurement.oproj_chunk_pairs(world, backend), 3)
                self.assertNotEqual(routes[backend], expected)
                self.assertNotEqual(self.project(routes[backend]), self.project(expected))
            self.assertNotEqual(routes['te_ub'], routes['cublaslt_nccl'])

    def test_explicit_input_permutation_makes_legacy_routes_canonical(self):
        for world, batch, chunk_rows in itertools.product((4, 8), (1, 2), (1, 3, 5)):
            sources = self.sources(world, batch, chunk_rows, 2, 3)
            expected = self.canonical(sources, chunk_rows)
            expected_projection = self.canonical_projection(sources, chunk_rows, 2, 3)
            for backend in ('te_ub', 'cublaslt_nccl'):
                # Independent legacy packing definitions construct a concrete
                # input permutation. No such permutation means no equivalence.
                before_order = list(range(0, 2 * world, 2)) + list(range(2 * world - 1, 0, -2))
                permutation = [None] * (2 * world)
                for rank in range(world):
                    legacy = ((2 * rank, 2 * world - 2 * rank - 1) if backend == 'te_ub'
                              else before_order[2 * rank:2 * rank + 2])
                    permutation[legacy[0]] = rank
                    permutation[legacy[1]] = 2 * world - rank - 1
                self.assertEqual(sorted(permutation), list(range(2 * world)))
                reordered = [
                    [[source[b][chunk * chunk_rows + row]
                      for chunk in permutation for row in range(chunk_rows)]
                     for b in range(batch)] for source in sources
                ]
                actual = self.route(reordered, measurement.oproj_chunk_pairs(world, backend), chunk_rows)
                self.assertEqual(actual, expected)
                self.assertEqual(self.project(actual), expected_projection)
