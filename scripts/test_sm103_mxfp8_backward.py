"""Backward preparation/contract checks; these do not prove CUDA correctness."""
from collections import Counter
from pathlib import Path
import unittest
import summarize_sm103_mxfp8_backward as backward_summary

ROOT = Path(__file__).resolve().parents[1]


class Mxfp8BackwardContracts(unittest.TestCase):
    def test_transposed_quantization_register_groups_cover_k32(self):
        stores, groups = Counter(), Counter()
        for warp in range(8):
            for lane in range(32):
                row = 32 * warp + lane
                for k in range(32):
                    stores[row, k] += 1
                groups[row] += 1
            for k in range(32):
                banks = {}
                for lane in range(32):
                    word = (k*264+warp*32+lane)//2
                    banks.setdefault(word % 32,set()).add(word)
                # Two adjacent BF16 halves share one32-bit word, not two
                # different words contending for the same shared-memory bank.
                self.assertTrue(all(len(words)==1 for words in banks.values()))
        self.assertEqual(set(stores), {(r,k) for r in range(256) for k in range(32)})
        self.assertEqual(set(stores.values()), {1})
        self.assertEqual(groups, Counter(range(256)))

    def test_backward_log_parser_rejects_unowned_or_ambiguous_records(self):
        rows=backward_summary.parse(b'device,rank=3,sm=148,compute=10.3\n')
        self.assertEqual(rows,[dict(kind='backward_device',rank='3',sm='148',compute='10.3')])
        for raw in (b'FAIL bad\n',b'backward_unknown rank=0\n',
                    b'backward_verified pflops=nan\n',b'backward_device rank=0 rank=1\n'):
            with self.subTest(raw=raw),self.assertRaises(ValueError):
                backward_summary.parse(raw)

    def test_transpose_tile_ownership_and_padded_scale_rows(self):
        for rows, k in ((1,128), (33,128), (128,256), (129,128), (256,384), (257,128), (384,256)):
            padded = (rows+127)//128*128
            stores, scales = Counter(), Counter()
            for rb in range(0,padded,256):
                for kb in range(0,k,32):
                    tile = {}
                    for warp in range(8):
                        for lane in range(32):
                            for i in range(warp,32,8):
                                # Label each scalar by its original physical
                                # source coordinate, not its floating value.
                                for j in range(8):
                                    r=lane*8+j
                                    tile[i,r] = (kb+i,rb+r) if rb+r<rows else None
                    for thread in range(256):
                        row=rb+thread
                        for i in range(32):
                            col=kb+i
                            self.assertEqual(tile[i,thread],(col,row) if row<rows else None)
                            if row<rows: stores[row,col]+=1
                        if row<padded: scales[row,kb//32]+=1
            self.assertEqual(set(stores),{(r,c) for r in range(rows) for c in range(k)})
            self.assertEqual(set(scales),{(r,c) for r in range(padded) for c in range(k//32)})
            self.assertEqual(set(stores.values()),{1})
            self.assertEqual(set(scales.values()),{1})

    def test_backward_boundaries_include_preparation_and_true_weight_gradient(self):
        api=(ROOT/'csrc/operators/sm103/api/backward_mxfp8.cuh').read_text()
        header=(ROOT/'include/fuse/operators/ulysses/oproj_backward.h').read_text()
        self.assertIn('prepare_mxfp8_transpose(d.weight, w.b, w.sfb, g.n, g.k, g.n',api)
        self.assertIn('prepare_mxfp8_transpose(d.gradient, w.lhs, w.sfa, g.m, g.k, g.m',api)
        self.assertIn('prepare_mxfp8_transpose(d.input, w.rhs.b, w.rhs.sfb, g.n, g.k, g.n',api)
        self.assertIn('d.saved_attention, d.grad_weight, d.alpha, d.beta',api)
        self.assertIn('Mxfp8GemmFamily<256, 128, EpilogueN, 0, Bf16>::PureGemm',api)
        self.assertIn('args.epilogue.thread.beta = d.beta',api)
        self.assertIn('launch_oproj_backward_mxfp8_weight(p.weight, stream)',api)
        self.assertIn('Cross-CP dWo reduction is caller-owned',header)
        self.assertIn('straight-through',header)
        self.assertIn('upstream dY quantization is not',header)

    def test_qkv_weight_shares_preparation_without_relabeling_heads(self):
        api=(ROOT/'csrc/operators/sm103/api/backward_mxfp8.cuh').read_text()
        header=(ROOT/'include/fuse/operators/ulysses/qkv_backward.h').read_text()
        self.assertIn('(p.q_heads + 2 * p.kv_heads) * p.head_dim',api)
        self.assertIn('p.hidden, p.local_tokens, 1',api)
        self.assertIn('d.dqkv_staging, d.saved_input, d.grad_weight, d.alpha, d.beta',api)
        self.assertIn('backward_mxfp8_weight_impl<EpilogueN, Prepare>',api)
        self.assertIn('cross-CP dW summation remains caller-owned',header)
        self.assertIn('This W entry alone does not represent a complete QKV backward',header)

    def test_existing_input_type_and_ready_protocol_are_not_faked(self):
        api=(ROOT/'csrc/operators/sm103/api/backward_mxfp8.cuh').read_text()
        self.assertIn('Base::can_implement(args, false)',api)
        self.assertIn('args.gemm.mainloop.weight_ready = nullptr',api)
        self.assertIn('args.gemm.epilogue.n_tiles = ceil_div(g.n, 256)',api)
        self.assertNotIn('reinterpret_cast<Bf16*>',api)
        self.assertNotIn('cudaMalloc',api)
        self.assertNotIn('cudaDeviceSynchronize',api)

    def test_qkv_inverse_route_packs_heads_without_changing_k32_groups(self):
        for world,q,kv in ((4,8,8),(8,16,8),(4,64,8),(8,128,8)):
            for batch in (1,2):
                for causal in (False,True):
                    seq=256
                    for rank in range(world):
                        destinations=Counter()
                        for row in range(batch*seq):
                            b,local=divmod(row,seq)
                            if causal:
                                c=rank if local<seq//2 else 2*world-rank-1
                                source=b*seq*world+c*(seq//2)+local%(seq//2)
                            else:
                                source=b*seq*world+rank*seq+local
                            for head in range(q+2*kv):
                                kind=0 if head<q else (1 if head<q+kv else 2)
                                group=head-(0 if kind==0 else (q if kind==1 else q+kv))
                                count=(q if kind==0 else kv)//world
                                peer,own=divmod(group,count)
                                self.assertLess(peer,world)
                                # A ready task cannot cross a causal row jump
                                # or change a native aligned M128/K128 SF atom.
                                self.assertEqual(source%128,row%128)
                                destinations[row,head]+=1
                                self.assertEqual(peer*count+own,group)
                        self.assertEqual(len(destinations),batch*seq*(q+2*kv))
                        self.assertEqual(set(destinations.values()),{1})

    def test_qkv_backward_keeps_bf16_w_lease_and_complete_ready(self):
        api=(ROOT/'csrc/operators/sm103/api/backward_mxfp8.cuh').read_text()
        route=(ROOT/'csrc/operators/sm103/detail/backward.cuh').read_text()
        header=(ROOT/'include/fuse/operators/ulysses/qkv_backward.h').read_text()
        self.assertIn('main.k_tiles_per_peer=1;main.epoch=1',api)
        qkv=api[api.index('cudaError_t qkv_backward_mxfp8_data_impl'):]
        self.assertNotIn('WeightReadyMainloop',qkv)
        self.assertNotIn('main.weight_ready',qkv)
        self.assertIn('d.peer_dqkv_staging[d.rank]!=w.dqkv_staging',api)
        self.assertIn('detail::InputProductionKernel<Base,Comm>',api)
        self.assertIn('Mxfp8QkvBackwardPullComm',route)
        self.assertIn('source.master_q',route)
        self.assertIn('cooperative_groups::this_grid().sync()',route)
        self.assertIn('ALL ranks finish B',header)
        self.assertIn('No KV replication/reduction',header)
        self.assertGreater(header.index('struct Mxfp8QkvBackwardInput'),
                           header.index('using Bf16QkvBackwardDataParams'))

    def test_qkv_async_transport_slices_preserve_one_complete_head(self):
        for elements_per_vector in (16,8):
            owners=Counter()
            for start in range(0,128,16):
                for lane in range(32):
                    for i in range(lane,16*128//elements_per_vector,32):
                        row=start+i//(128//elements_per_vector)
                        col=i%(128//elements_per_vector)*elements_per_vector
                        for j in range(elements_per_vector):owners[row,col+j]+=1
            self.assertEqual(set(owners),{(r,c) for r in range(128) for c in range(128)})
            self.assertEqual(set(owners.values()),{1})
        text=(ROOT/'csrc/operators/sm103/detail/backward.cuh').read_text()
        route=text[text.index('struct Mxfp8QkvBackwardPullComm'):]
        self.assertIn('cute::cp_async_wait<0>()',route)
        self.assertEqual(route.count('detail::store_release_system('),1)
        self.assertLess(route.index('cute::cp_async_wait<0>()'),route.index('detail::store_release_system('))

    def test_static_head_index_preserves_every_ready_boundary(self):
        for heads in (24,72,80,144,288,336):
            for prologue in (1,2,3,4):
                dynamic=[]; specialized=[]
                for start,end in ((0,prologue),(prologue,heads)):
                    for k in range(start,end):
                        dynamic.append((k//1,min(end-k,1-k%1)))
                        specialized.append((k,1))
                self.assertEqual(dynamic,specialized)
                self.assertEqual([p for p,_ in specialized],list(range(heads)))
        text=(ROOT/'csrc/operators/sm103/detail/cutlass_pipeline.cuh').read_text()
        self.assertIn('args.k_tiles_per_peer == StaticKTilesPerPeer',text)
        self.assertIn('int StaticKTilesPerPeer = 0',text)

    def test_smoke_cannot_be_mistaken_for_full_performance(self):
        text=(ROOT/'benchmarks/sm103/backward/mxfp8_smoke.cu').read_text()
        self.assertIn('curandStatePhilox4_32_10_t',text)
        self.assertIn('generation<2',text)
        self.assertIn('performance=not_measured',text)
        self.assertIn('double sum = 0',text)
        self.assertIn('dA inverse route bytes',text)
        self.assertIn('deferred B modified dW',text)

    def test_formal_boundary_and_reference_are_independent_and_complete(self):
        text=(ROOT/'benchmarks/sm103/backward/mxfp8_mpi_bench.cu').read_text()
        ref=(ROOT/'benchmarks/sm103/backward/mxfp8_reference.cuh').read_text()
        self.assertIn('const double flops=4.*o.m*o.h*o.width()',text)
        self.assertIn('WeightGradientMode::kImmediate',text)
        self.assertIn('r.validate(o,generation,"pre")',text)
        self.assertIn('r.validate(o,generation,"post")',text)
        self.assertIn('collect(50,"measurement",round)',text)
        self.assertIn('graph.committed_epoch()+1',text)
        self.assertIn('fused_inputs::generate',text)
        self.assertIn('cudaIpcOpenMemHandle',text)
        self.assertIn('reference.validate({dy,o.h,1},{weight,1,route.a}',text)
        self.assertIn('reference.validate({dy,1,o.h},{attention,1,route.a}',text)
        self.assertIn('frexpf(amax, &exponent)',ref)
        self.assertIn('CUBLAS_COMPUTE_32F_PEDANTIC',ref)
        self.assertIn('kRows = 128, kChunk = 4096',ref)
        self.assertNotIn('ptr_SFA',ref)
        self.assertNotIn('ptr_SFB',ref)
        self.assertNotIn('quantize_mxfp8_transposed_operand',ref)

    def test_isolated_components_preserve_full_boundary_and_native_epoch(self):
        text=(ROOT/'benchmarks/sm103/backward/mxfp8_mpi_bench.cu').read_text()
        api=(ROOT/'csrc/operators/sm103/api/backward_mxfp8.cuh').read_text()
        self.assertIn('template <int EpilogueN, bool Prepare = true>',api)
        self.assertIn('if constexpr (Prepare)',api)
        self.assertIn('oproj_backward_mxfp8_weight_impl<32, false>',api)
        self.assertIn('Component::kData,Component::kWeight,Component::kWeightCompute',text)
        self.assertIn('component==Component::kFull || component==Component::kData',text)
        weight=text.index('if(component==Component::kWeight)return fuse::launch_oproj_backward_mxfp8_weight')
        epoch=text.index('params.data.projection.epoch=epoch;',weight)
        self.assertLess(weight,epoch)
        self.assertLess(text.index('if(component==Component::kWeightCompute)',weight),epoch)
        self.assertIn('graph.reset(r.native_epoch())',text)
        self.assertIn('component==Component::kData?r.native_epoch():0',text)
        self.assertIn('(flops/2)/(value.p50*1e12)',text)
        self.assertIn('r.validate(o,generation,"post");results.push_back(result)',text)

    def test_prepared_dx_preserves_ready_budget_and_weight_workspace_lease(self):
        api=(ROOT/'csrc/operators/sm103/api/backward_mxfp8.cuh').read_text()
        harness=(ROOT/'benchmarks/sm103/backward/mxfp8_mpi_bench.cu').read_text()
        route=(ROOT/'csrc/operators/sm103/detail/backward.cuh').read_text()
        header=(ROOT/'include/fuse/operators/ulysses/qkv_backward.h').read_text()
        self.assertIn('using Comm=Mxfp8QkvBackwardPullComm<!Prepare>',api)
        self.assertIn('qkv_backward_mxfp8_data_impl<32,false>',api)
        self.assertIn('if constexpr (Prepared) return',route)
        self.assertIn('compute-CTA budget',header)
        self.assertIn('Do not run W on this scratch',header)
        self.assertIn('components.insert(components.begin()+1,Component::kDataCompute)',harness)
        self.assertIn('std::vector<L>{L::kCooperativeDynamic}',harness)

    def test_qkv_formal_reference_reads_original_peer_planes(self):
        text=(ROOT/'benchmarks/sm103/backward/mxfp8_mpi_bench.cu').read_text()
        ref=(ROOT/'benchmarks/sm103/backward/mxfp8_reference.cuh').read_text()
        self.assertIn('struct QkvGradient',ref)
        self.assertIn('source[owner][kind][int64_t(global)*width+column]',ref)
        self.assertIn('reference.validate_views(original_qkv',text)
        self.assertIn('original_BF16_route=included',text)
        self.assertIn('input_lease=all_ranks_until_B_complete',text)
        self.assertIn('(heads+(qkv?2*kv_heads:0))*128',text)


if __name__ == '__main__':
    unittest.main()
