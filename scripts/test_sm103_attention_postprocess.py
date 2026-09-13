"""Host invariants for the experimental head-local norm/RoPE layout.

These tests do not substitute for CUDA numerical/Graph validation.
"""
import math
import random
import struct
import unittest


def bf16(x):
    bits = struct.unpack('<I', struct.pack('<f', x))[0]
    bits = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) << 16
    return struct.unpack('<f', struct.pack('<I', bits))[0]


class HeadLayoutTests(unittest.TestCase):
    def test_postprocess_audit_normalizes_null_cli_defaults(self):
        from summarize_sm103_mxfp8_fused import postprocess_configuration
        job=dict(qkv_postprocess=None,norm_epsilon=None,rope_policy=None)
        self.assertEqual(postprocess_configuration(job,{}),dict(qkv='none',
            oproj_residual_rmsnorm=False,oproj_overlap=False,oproj_separate=False,qkv_separate=False,
            norm_epsilon=1.e-6,rope_policy='qwen3'))

    def test_postprocess_audit_requires_executed_semantics(self):
        from summarize_sm103_mxfp8_fused import postprocess_configuration
        for key,value in [('qkv_postprocess','qknorm_rope'),('norm_epsilon',1.e-5),
                          ('rope_policy','llama31'),('oproj_postnorm',True),
                          ('oproj_postnorm_overlap',True),('oproj_postnorm_separate',True),
                          ('qkv_postprocess_separate',True)]:
            with self.subTest(key=key):
                with self.assertRaises(ValueError):postprocess_configuration({key:value},{})
                config={key:str(int(value)) if isinstance(value,bool) else str(value)}
                postprocess_configuration({key:value},config)

    def test_each_head_feature_has_one_owner(self):
        visits = []
        for lane in range(32):
            for row in range(lane // 2, 64, 16):
                for element in range(64):
                    visits.append((row, lane % 2 * 32 + element % 32 + element // 32 * 64))
        self.assertEqual(len(visits), 64 * 128)
        self.assertEqual(set(visits), {(r, f) for r in range(64) for f in range(128)})

    def test_separated_received_head_tasks_cover_qk_once_and_preserve_v(self):
        for world,q_heads,k_heads in ((4,64,4),(4,64,8),(8,128,8)):
            rows, q, k = 128, q_heads//world, k_heads//world
            addressed = []
            for task in range((rows//64)*(q+k)):
                row,head = (task//(q+k))*64,task%(q+k)
                segment = int(head>=q)
                width = (k if segment else q)*128
                offset = rows*q*128 if segment else 0
                feature = (head-q if segment else head)*128
                for r in range(row,row+64):
                    addressed.extend(offset+r*width+feature+i for i in range(128))
            self.assertEqual(len(addressed),rows*(q+k)*128)
            self.assertEqual(set(addressed),set(range(rows*(q+k)*128)))

    def test_separated_epoch_second_phase_cannot_satisfy_next_raw_route(self):
        for epoch in (1,2,1000,(2**32-2)//2):
            self.assertLess(2*epoch,2*epoch+1)
            self.assertLess(2*epoch+1,2*(epoch+1))
            self.assertLess(2*epoch+1,2**32)

    def test_rope_partners_are_in_the_same_lane(self):
        for lane in range(32):
            for element in range(64):
                f = lane % 2 * 32 + element % 32 + element // 32 * 64
                other = element ^ 32
                paired = lane % 2 * 32 + other % 32 + other // 32 * 64
                self.assertEqual(paired, (f + 64) % 128)

    def test_vector_alignment(self):
        for row in (0, 1, 63, 32767):
            for head_lane in range(2):
                for half in (0, 64):
                    self.assertEqual(((row * 128 + head_lane * 32 + half) * 2) % 16, 0)

    def test_position_is_not_destination_rank(self):
        # A source token keeps its global position when its heads are spread
        # over different destination ranks. The public API receives that table.
        source_rank, local_row, rows = 3, 19, 32768
        position = source_rank * rows + local_row
        for destination in range(4):
            self.assertEqual(position, 98323)
            if destination != source_rank:
                self.assertNotEqual(position, destination * rows + local_row)

    def test_rope_bf16_products_are_not_one_fp32_fma(self):
        found = False
        for i in range(1, 128):
            x, y = bf16(i / 17), bf16((i + 1) / 19)
            c, s = bf16(math.cos(i)), bf16(math.sin(i))
            if bf16(bf16(x * c) - bf16(y * s)) != bf16(x * c - y * s):
                found = True
                break
        self.assertTrue(found)

    def test_residual_output_must_survive_normalization(self):
        residual = [bf16(0.25), bf16(-0.5), bf16(1.), bf16(2.)]
        projected = [bf16(0.5), bf16(0.25), bf16(-0.75), bf16(-1.)]
        added = [bf16(a + b) for a, b in zip(residual, projected)]
        inverse = 1 / math.sqrt(sum(x * x for x in added) / len(added) + 1.e-6)
        normalized = [bf16(x * inverse) for x in added]
        self.assertNotEqual(added, normalized)
        self.assertEqual(added, [0.75, -0.25, 0.25, 1.])

    def test_oproj_cta_owns_the_complete_hidden_row(self):
        for width in (4096, 5120, 8192, 14336, 16384):
            columns = [col+i for thread in range(256)
                       for col in range(thread*8, width, 256*8) for i in range(8)]
            self.assertEqual(len(columns), width)
            self.assertEqual(set(columns), set(range(width)))
            self.assertLessEqual(width*2, 220160-8*4)
        rows = [row for cta in range(148) for row in range(cta, 32768, 148)]
        self.assertEqual(len(rows), 32768)
        self.assertEqual(set(rows), set(range(32768)))

    def test_fp64_reference_resolves_disputed_bf16_rounding(self):
        # Measured disputed dots: refine the independent reference, not the
        # acceptance threshold or actual output. Cancellation magnifies one
        # projection ULP after RMSNorm.
        for dot, residual, inverse, expected_sum, expected in (
            (-0.252929809, 0.1962890625, 6.35997677, -0.0576171875, -0.40234375),
            (-0.323242636, 0.0556640625, 6.33338451, -0.26953125, -1.875),
        ):
            added = bf16(bf16(dot) + residual)
            normalized = bf16(bf16(added * inverse) * 1.09375)
            self.assertEqual(added, expected_sum)
            self.assertEqual(normalized, expected)
            # A real output corruption is not repaired by changing reference.
            self.assertGreater(abs((normalized + 0.25) - expected), 0.01 + 0.01 * abs(expected))

    def test_residual_epilogue_must_round_projection_first(self):
        differences = 0
        for i in range(1, 100):
            accumulator = i / 113.0
            residual = bf16(-i / 127.0)
            staged = bf16(bf16(accumulator) + residual)
            # alpha=beta=1 in BF16 compute converts acc before multiply/add.
            epilogue = bf16(bf16(1.0 * bf16(accumulator)) + bf16(1.0 * residual))
            self.assertEqual(epilogue, staged)
            differences += staged != bf16(accumulator + residual)
        self.assertGreater(differences, 0)

    def test_overlap_tasks_own_rows_once_and_require_all_n_tiles(self):
        rng = random.Random(13)
        for rows in (128, 32768, 65536, 131072):
            for comm in (20, 32, 48):
                # Atomic eight-row claims in arbitrary completion order;
                # finished GEMM CTAs join the pool without duplicate owners.
                workers = list(range(comm))
                seen, next_row = [], 0
                while next_row < rows:
                    if next_row >= rows//2:
                        workers = list(range(148))
                    owner = rng.choice(workers)
                    self.assertTrue(0 <= owner < 148)
                    row = next_row
                    next_row += 8
                    self.assertEqual(row//128, (row+7)//128)
                    seen.extend(range(row, row+8))
                self.assertEqual(len(seen), rows)
                self.assertEqual(set(seen), set(range(rows)))
        for width in (4096, 8192, 16384):
            columns = [col+i for lane in range(128)
                       for col in range(lane*8, width, 128*8) for i in range(8)]
            self.assertEqual(sorted(columns), list(range(width)))
            ready = [True] * (width//256)
            for n in range(len(ready)):
                ready[n] = False
                self.assertFalse(all(ready))
                ready[n] = True
            self.assertTrue(all(ready))

    def test_norm_cache_fits_reclaimed_smem_without_increasing_resource_floor(self):
        shared_lower_bound = 4*(48*1024+8)
        cache_end = 4*16384*2
        partial_begin = shared_lower_bound-33*4
        self.assertLess(cache_end, partial_begin)
        self.assertEqual(partial_begin % 4, 0)

    def test_parallel_ready_checks_cover_exactly_the_same_n_flags(self):
        for n_tiles in (1,16,31,32,33,64):
            visits=[n for lane in range(32) for n in range(lane,n_tiles,32)]
            self.assertEqual(sorted(visits),list(range(n_tiles)))
            # CTA completion requires all active lanes, not merely lane zero.
            for last_ready in range(n_tiles):
                flags=[n != last_ready for n in range(n_tiles)]
                lanes=[all(flags[n] for n in range(lane,n_tiles,32)) for lane in range(32)]
                self.assertFalse(all(lanes))

    def test_async_norm_inputs_do_not_overlap_or_change_element_ownership(self):
        capacity = 4*(48*1024+8)-33*4
        for width in (4096,5120,8192,14336,16384):
            staging_bytes = 2*4*width*2
            if staging_bytes > capacity:
                self.assertGreater(width,8192)
                continue
            copied = [i for t in range(256) for i in range(t,4*width//8,256)]
            self.assertEqual(sorted(copied),list(range(4*width//8)))
            self.assertEqual(4*width*2 % 16,0)
            self.assertLessEqual(staging_bytes,capacity)
            for row in range(4):
                for t in range(64):
                    for half in range(4):
                        for col in range((t+half*64)*8,width,2048):
                            projection = row*width+col
                            residual = 4*width+projection
                            self.assertLess(projection+7,4*width)
                            self.assertGreaterEqual(residual,4*width)
                            self.assertLess(residual+7,8*width)

    def test_norm_tree_is_independent_of_cta_identity(self):
        for width in (4096,8192,16384):
            ownership = []
            for cta in (0,31,147):
                logical = {}
                for thread in range(256):
                    logical[thread] = [col+i for col in range(thread*8,width,256*8)
                                       for i in range(8)]
                ownership.append(logical)
            # Identical per-logical-lane ordered inputs, then identical warp
            # shuffle tree and eight-warp partial order, whatever task owner.
            self.assertEqual(ownership[0],ownership[1])
            self.assertEqual(ownership[0],ownership[2])
            self.assertEqual(sorted(x for v in ownership[0].values() for x in v),list(range(width)))

    def test_capacity_selected_row_group_preserves_task_coverage(self):
        for capacity in (0,196508,220028):
            for width in (4096,5120,8192,14336,16384):
                row_bytes = 2*width*2
                rows = 2 if 4*row_bytes > capacity >= 2*row_bytes else 4
                if rows == 2:
                    self.assertLessEqual(rows*row_bytes,capacity)
                tasks = [i+j for i in range(0,8,rows) for j in range(rows)]
                self.assertEqual(tasks,list(range(8)))
                self.assertLessEqual(rows*8,32)
        self.assertEqual(2*2*16384*2,131072)

    def test_row_cohorts_preserve_virtual256_order(self):
        for width in (4096,8192,16384):
            original = {lane: [] for lane in range(256)}
            for col in range(width):
                original[(col//8)%256].append(col)
            for rows_per_cta in (2,4):
                threads = 256//rows_per_cta
                for row in range(rows_per_cta):
                    visited = []
                    for thread in range(threads):
                        for half in range(rows_per_cta):
                            virtual = thread + half*threads
                            actual = [col+i for col in range(virtual*8,width,256*8)
                                      for i in range(8)]
                            self.assertEqual(actual,original[virtual])
                            self.assertEqual(row*8+half*(threads//32)+thread//32,row*8+virtual//32)
                            visited.extend(actual)
                    self.assertEqual(sorted(visited),list(range(width)))

    def test_graph_repeat_launches_are_accounted_but_not_samples(self):
        import summarize_sm103_fused as summary
        records = []
        for g,first,calls,line in ((0,1,64,201),(1,65,4,211)):
            records.append(dict(kind='graph_prepare',label='A2A_GEMM',candidate='1',comm_sm='32',
                tile='m128n256',component='fused',generation=str(g),rank='0',line=line,
                launch='graph',graph_epoch_mode=summary.GRAPH_EPOCH_MODE,
                first_epoch=str(first),last_epoch=str(first+calls-1),calls=str(calls),
                wall_s='0.01',gpu_sample_time='0',
                includes='capture_inspect_instantiate_initial_upload_sync_update'))
        records.append(dict(kind='candidate_verified',line=300))
        checks = {('candidate',g,'pre',0): dict(line=1) for g in (0,1)}
        checks.update({('correctness',0,'pre',0):dict(line=2),
                       ('correctness',0,'post',0):dict(line=200),
                       ('correctness',1,'pre',0):dict(line=210)})
        timing = dict(warmup_calls=10,rounds=[dict(maxrank_ms=[1.]*50,epoch_first=15,epoch_last=64)])
        result = summary.audit_graph_preparation(records,timing,checks,('correctness',),1,repeat_launches=3)
        self.assertEqual([r['calls'] for r in result],[64,4])
        with self.assertRaisesRegex(ValueError,'actual launch count'):
            summary.audit_graph_preparation(records,timing,checks,('correctness',),1)


if __name__ == '__main__':
    unittest.main()
