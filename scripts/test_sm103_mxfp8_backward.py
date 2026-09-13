"""Backward preparation/contract checks; these do not prove CUDA correctness."""
from collections import Counter
from pathlib import Path
import unittest
import summarize_sm103_mxfp8_backward as backward_summary

ROOT = Path(__file__).resolve().parents[1]


class Mxfp8BackwardContracts(unittest.TestCase):
    def test_transposed_quantization_subgroups_cover_k32_without_bank_conflicts(self):
        stores, groups = Counter(), Counter()
        for warp in range(8):
            for lane in range(32):
                row, k = 4 * warp + lane // 8, 4 * (lane % 8)
                for i in range(4):
                    stores[row, k+i] += 1
                if lane % 8 == 0:
                    groups[row] += 1
            for i in range(4):
                banks = [(4*(lane % 8)+i+4*warp+lane//8) % 32 for lane in range(32)]
                self.assertEqual(len(set(banks)), 32)
        self.assertEqual(set(stores), {(r,k) for r in range(32) for k in range(32)})
        self.assertEqual(set(stores.values()), {1})
        self.assertEqual(groups, Counter(range(32)))

    def test_backward_log_parser_rejects_unowned_or_ambiguous_records(self):
        rows=backward_summary.parse(b'device,rank=3,sm=148,compute=10.3\n')
        self.assertEqual(rows,[dict(kind='backward_device',rank='3',sm='148',compute='10.3')])
        for raw in (b'FAIL bad\n',b'backward_unknown rank=0\n',
                    b'backward_verified pflops=nan\n',b'backward_device rank=0 rank=1\n'):
            with self.subTest(raw=raw),self.assertRaises(ValueError):
                backward_summary.parse(raw)

    def test_transpose_tile_ownership_and_padded_scale_rows(self):
        for rows, k in ((1,128), (33,128), (128,256), (129,128), (256,384)):
            padded = (rows+127)//128*128
            stores, scales = Counter(), Counter()
            for rb in range(0,padded,32):
                for kb in range(0,k,32):
                    tile = {}
                    for warp in range(8):
                        for lane in range(32):
                            for i in range(warp,32,8):
                                # Label each scalar by its original physical
                                # source coordinate, not its floating value.
                                tile[i,lane] = (kb+i,rb+lane) if rb+lane<rows else None
                    for warp in range(8):
                        for lane in range(32):
                            for i in range(warp,32,8):
                                row,col=rb+i,kb+lane
                                self.assertEqual(tile[lane,i],(col,row) if row<rows else None)
                                if row<rows: stores[row,col]+=1
                                if lane==0: scales[row,col//32]+=1
            self.assertEqual(set(stores),{(r,c) for r in range(rows) for c in range(k)})
            self.assertEqual(set(scales),{(r,c) for r in range(padded) for c in range(k//32)})
            self.assertEqual(set(stores.values()),{1})
            self.assertEqual(set(scales.values()),{1})

    def test_backward_boundaries_include_preparation_and_true_weight_gradient(self):
        api=(ROOT/'csrc/operators/sm103/api/backward_mxfp8.cuh').read_text()
        header=(ROOT/'include/fuse/operators/ulysses/oproj_backward.h').read_text()
        self.assertIn('prepare_mxfp8_transpose(d.weight, w.b, w.sfb, g.n, g.k, g.n',api)
        self.assertIn('prepare_mxfp8_transpose(d.grad_output, w.lhs, w.sfa, g.m, g.k, g.m',api)
        self.assertIn('prepare_mxfp8_transpose(d.saved_attention, w.rhs.b, w.rhs.sfb, g.n, g.k, g.n',api)
        self.assertIn('Mxfp8GemmFamily<256, 128, EpilogueN, 0, Bf16>::PureGemm',api)
        self.assertIn('args.epilogue.thread.beta = d.beta',api)
        self.assertIn('launch_oproj_backward_mxfp8_weight(p.weight, stream)',api)
        self.assertIn('Cross-CP dWo reduction is caller-owned',header)
        self.assertIn('straight-through',header)
        self.assertIn('upstream dY quantization is not',header)

    def test_existing_input_type_and_ready_protocol_are_not_faked(self):
        api=(ROOT/'csrc/operators/sm103/api/backward_mxfp8.cuh').read_text()
        self.assertIn('Base::can_implement(args, false)',api)
        self.assertIn('args.gemm.mainloop.weight_ready = nullptr',api)
        self.assertIn('args.gemm.epilogue.n_tiles = ceil_div(g.n, 256)',api)
        self.assertNotIn('reinterpret_cast<Bf16*>',api)
        self.assertNotIn('cudaMalloc',api)
        self.assertNotIn('cudaDeviceSynchronize',api)

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
        self.assertIn('const double flops=4.*o.m*o.h*(o.heads*128)',text)
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
        self.assertIn('graph.reset(r.params.data.projection.epoch)',text)
        self.assertIn('component==Component::kData?r.params.data.projection.epoch:0',text)
        self.assertIn('(flops/2)/(value.p50*1e12)',text)
        self.assertIn('r.validate(o,generation,"post");results.push_back(result)',text)


if __name__ == '__main__':
    unittest.main()
