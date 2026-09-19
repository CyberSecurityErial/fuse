// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "gemm.cuh"
#include "communication.cuh"
#include <cute/arch/copy_sm90_tma.hpp>
#include <algorithm>

namespace fuse::detail {

// Dispatch -> GEMM: producers gather an entire M panel through K, then publish
// ONE ready flag. Every N consumer of that panel shares the same delivery unit.
//   source(rank, token) -> expert rows [m*TileM, (m+1)*TileM), columns [0,K)
//                           CTA join -> ready[expert,m] -> GEMM N tiles
// Arbitrary source rows remain separate requests; only destination rows can
// be combined. Copy staging never changes the full-panel delivery boundary.
template <int TileM = 128>
struct GroupedDispatchComm {
  using SM100_BULK_COPY_S2G = cute::SM90_BULK_COPY_S2G;
  using SM100_BULK_COPY_G2S = cute::SM90_BULK_COPY_G2S;
  static constexpr int kMinThreads = 256;
  static constexpr int kVectorsPerThread = 8;
  static constexpr int kStageBytes = 24 * 1024;
  static constexpr size_t kRowAddressBytes = TileM*sizeof(const Bf16*);
  static constexpr size_t kStagingBudget = std::min({
      sizeof(Bf16GroupedGemmTypes<128,64>::DispatchGemm::SharedStorage),
      sizeof(Bf16GroupedGemmTypes<128,128>::DispatchGemm::SharedStorage),
      sizeof(Bf16GroupedGemmTypes<256,64>::DispatchGemm::SharedStorage),
      sizeof(Bf16GroupedGemmTypes<256,128>::DispatchGemm::SharedStorage)});
  static constexpr int kBulkSlots = std::min(size_t(kMinThreads/32),
      (kStagingBudget-kRowAddressBytes)/(kStageBytes+sizeof(uint64_t)));
  static constexpr size_t SharedStorageBytes =
      kBulkSlots*(kStageBytes+sizeof(uint64_t))+kRowAddressBytes;
  static constexpr bool kNeedsGridFinalize = true;
  static constexpr bool kReuseIdleComputeCtas = true;
  using Arguments = GroupedCommArguments;
  using Params = Arguments;

  static bool can_implement(const Arguments& a) {
    return valid_grouped_transport(a.params) && a.params.first_wave_panels &&
        a.params.effective_comm_ctas;
  }
  static Params to_underlying_arguments(const Arguments& a) { return a; }

  CUTLASS_DEVICE void operator()(const Params& args, char* smem, int comm, int comm_ctas) {
#if FUSE_ENABLE_PROFILING
    GroupedCommSummary summary{};
#endif
    const int64_t panels=args.params.order.row_tiles();
    const int64_t tail=panels%comm_ctas;
    if (args.params.balance_tail && panels>=comm_ctas && tail) {
      // Keep all complete batches unchanged. Split ONLY the final incomplete
      // batch into 8-row pieces (one row per warp), distributed across CTAs.
      // Example P=72,C=20: 60 full blocks, then 12*16 row pieces; each CTA
      // copies 9 or 10 pieces instead of 12 CTAs copying 128 rows and 8 idling.
      // There is no grid barrier at this transition: each CTA advances after
      // its own prefix. The last contributing CTA releases the original
      // full-K/128-row ready flag; GEMM does not gain extra ready checks.
      copy_selected<false>(args,smem,comm,comm_ctas,1,0,panels-tail
#if FUSE_ENABLE_PROFILING
          ,summary
#endif
      );
      copy_selected<true>(args,smem,comm,comm_ctas,TileM/(blockDim.x/32),panels-tail,panels
#if FUSE_ENABLE_PROFILING
          ,summary
#endif
      );
#if FUSE_ENABLE_PROFILING
      publish_summary(args.params,comm,summary);
#endif
      return;
    }
    const int first_wave=args.params.first_wave_panels?*args.params.first_wave_panels:0;
    const int splits = args.params.arrivals ? grouped_dispatch_splits(
        panels,comm_ctas,first_wave,int64_t(TileM)*args.params.columns*sizeof(Bf16),
        TileM,blockDim.x/32) : 1;
    if (splits>1) copy_selected<true>(args,smem,comm,comm_ctas,splits,0,panels
#if FUSE_ENABLE_PROFILING
        ,summary
#endif
    );
    else copy_selected<false>(args,smem,comm,comm_ctas,1,0,panels
#if FUSE_ENABLE_PROFILING
        ,summary
#endif
    );
#if FUSE_ENABLE_PROFILING
    publish_summary(args.params,comm,summary);
#endif
  }

#if FUSE_ENABLE_PROFILING
  CUTLASS_DEVICE void publish_summary(
      const GroupedCommParams& p, int comm, const GroupedCommSummary& summary) {
    const int warp=threadIdx.x/32;
    const int index=comm*(blockDim.x/32)+warp;
    if ((threadIdx.x&31)==0 && p.profile.comm_summary &&
        index<p.profile.comm_summary_capacity)
      p.profile.comm_summary[index]=summary;
  }
#endif

  // Each warp owns one stage. Different warps overlap their batches; a warp
  // completes its own S2G before reusing its stage. Gathered source rows stay
  // separate requests, while a contiguous destination batch uses one store.
  //   warp0: G2S -> S2G -> wait -> G2S ...
  //   warp1:   G2S -> S2G -> wait -> G2S ...
  //   all destination writes complete -> CTA join -> ONE full-panel ready.
  template <bool CpAsyncG2S>
  CUTLASS_DEVICE void copy_staged(const GroupedCommParams& p, char* smem,
      int expert, int first_row, int valid_rows, int64_t source_begin
#if FUSE_ENABLE_PROFILING
      ,GroupedCommSummary& summary
#endif
  ) {
    int batch_rows=1;
    while (batch_rows < TileM && int64_t(batch_rows)*2*p.columns*sizeof(Bf16) <= kStageBytes)
      batch_rows*=2;
    // A short stripe must not leave most copy slots idle merely because a
    // stage can hold many rows. Split its complete rows among available
    // leaders; this depends on actual work, never expert buffer capacity.
    while(batch_rows>1 && (valid_rows+batch_rows-1)/batch_rows<kBulkSlots)
      batch_rows/=2;
    const int lane=threadIdx.x%32, warp=threadIdx.x/32;
    // Independent warp leaders keep multiple batches in flight. Slots are
    // limited by the existing GEMM SMEM allocation, NOT extra communication
    // CTAs. Each leader owns its stages and completion barrier. There is
    // no CTA-wide join between batches: only the final full-panel join below
    // copy_staged makes all leaders' destination writes visible to ready.
    // With three slots: 0,3,6,... / 1,4,7,... / 2,5,8,... batch indices.
    // These queues partition ONE panel; none may prefetch the next panel.
    if(warp>=kBulkSlots) return;
    // Resolve this warp's independent source rows in parallel, before the
    // leader issues copies. The two dependent global reads (route, peer base)
    // no longer sit between each pair of bulk-copy instructions. Generic
    // shared memory holds only addresses, not payload; warp synchronization
    // makes these addresses visible to the leader. No extra CTA rendezvous,
    // no next-panel prefetch, and no change to batch ownership or readiness.
    auto** row_sources=reinterpret_cast<const Bf16**>(smem+
        kBulkSlots*(kStageBytes+sizeof(uint64_t)));
    const int batch_shift=__ffs(batch_rows)-1;  // batch_rows is a power of two
#if FUSE_ENABLE_PROFILING
    const uint64_t address_begin=read_global_timer();
#endif
    for(int owned=lane;;owned+=32) {
      const int row=(owned>>batch_shift)*kBulkSlots*batch_rows+
          warp*batch_rows+(owned&(batch_rows-1));
      if(row>=valid_rows) break;
      const auto route=p.source[source_begin+first_row+row];
      row_sources[row]=p.input[route.rank]+int64_t(route.token)*p.columns;
    }
    __syncwarp();
#if FUSE_ENABLE_PROFILING
    if(lane==0) {
      ++summary.panels;
      summary.address_ns+=read_global_timer()-address_begin;
    }
#endif
    auto* barrier=reinterpret_cast<uint64_t*>(smem+kStageBytes*kBulkSlots)+warp;
    if constexpr (!CpAsyncG2S) {
      if(lane!=0) return;
      cute::initialize_barrier(*barrier,1);
      cutlass::arch::fence_barrier_init();
      fence_proxy_async_global();
    }
    int phase=0;
    for(int begin=warp*batch_rows;begin<valid_rows;begin+=kBulkSlots*batch_rows) {
      const int count=valid_rows-begin<batch_rows ? valid_rows-begin : batch_rows;
      // Reusing this SMEM stage only requires the async engine to finish
      // reading its source. Destination-global visibility is stronger and is
      // required only once, at the full-panel publication boundary below.
      // Keeping the two waits distinct avoids serializing every batch on the
      // previous batch's NVLink/global write completion.
#if FUSE_ENABLE_PROFILING
      uint64_t stamp=lane==0?read_global_timer():0;
#endif
      if(lane==0) cute::tma_store_wait<0>();
      if constexpr (CpAsyncG2S) __syncwarp();
#if FUSE_ENABLE_PROFILING
      if(lane==0) {
        summary.store_wait_ns+=read_global_timer()-stamp;
        stamp=read_global_timer();
      }
#endif
      auto* buffer=reinterpret_cast<Bf16*>(smem+warp*kStageBytes);
#if FUSE_ENABLE_PROFILING
      uint64_t g2s_issued=0;
#endif
      if constexpr (CpAsyncG2S) {
        for(int r=0;r<count;++r) {
          auto* src=reinterpret_cast<const uint4*>(row_sources[begin+r]);
          auto* dst=reinterpret_cast<uint4*>(buffer+int64_t(r)*p.columns);
          for(int i=lane;i<p.columns/8;i+=32)
            cute::SM80_CP_ASYNC_CACHEGLOBAL<uint4>::copy(src[i],dst[i]);
        }
        cute::cp_async_fence();
#if FUSE_ENABLE_PROFILING
        if(lane==0) g2s_issued=read_global_timer();
#endif
        cute::cp_async_wait<0>();
        __syncwarp();
      } else {
        cute::set_barrier_transaction_bytes(*barrier,count*p.columns*sizeof(Bf16));
        for(int r=0;r<count;++r) {
          const Bf16* src=row_sources[begin+r];
          SM100_BULK_COPY_G2S::copy(src,barrier,buffer+int64_t(r)*p.columns,p.columns*sizeof(Bf16));
        }
#if FUSE_ENABLE_PROFILING
        g2s_issued=read_global_timer();
#endif
      }
#if FUSE_ENABLE_PROFILING
      if(lane==0) summary.g2s_issue_ns+=g2s_issued-stamp;
#endif
      if constexpr (!CpAsyncG2S) cute::wait_barrier(*barrier,phase);
#if FUSE_ENABLE_PROFILING
      const uint64_t g2s_done=lane==0?read_global_timer():0;
      if(lane==0) {
        summary.g2s_wait_ns+=g2s_done-g2s_issued;
        summary.g2s_ns+=g2s_done-stamp;
        stamp=read_global_timer();
      }
#endif
      if(lane==0) {
        phase^=1;
        cute::tma_store_fence();
        const int logical_row=first_row+begin;
        const int physical_row=p.buffer_m ? logical_row%(p.buffer_m*TileM) : logical_row;
        SM100_BULK_COPY_S2G::copy(buffer,p.output[expert]+int64_t(physical_row)*p.columns,
            count*p.columns*sizeof(Bf16));
        cute::tma_store_arrive();
      }
#if FUSE_ENABLE_PROFILING
      if(lane==0) {
        summary.store_issue_ns+=read_global_timer()-stamp;
        ++summary.batches;
        summary.bytes+=uint64_t(count)*p.columns*sizeof(Bf16);
      }
#endif
    }
#if FUSE_ENABLE_PROFILING
    const uint64_t final_wait=lane==0?read_global_timer():0;
#endif
    if(lane==0) tma_store_wait_all();
#if FUSE_ENABLE_PROFILING
    if(lane==0) summary.store_wait_ns+=read_global_timer()-final_wait;
#endif
    if constexpr (!CpAsyncG2S) {
      const uint32_t address=cute::cast_smem_ptr_to_uint(barrier);
      asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(address) : "memory");
    }
  }

  template <bool SharedPanels, bool CpAsyncG2S>
  CUTLASS_DEVICE void copy(const Params& args, char* smem, int comm, int comm_ctas, int splits,
      int64_t begin, int64_t end
#if FUSE_ENABLE_PROFILING
      ,GroupedCommSummary& summary
#endif
  ) {
    const auto& p = args.params;
    const uint32_t epoch = p.epoch_ptr ? *p.epoch_ptr : p.epoch;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int warps = blockDim.x / 32;
    const int slots = SharedPanels ? splits : 1;
    // Example: one 128-row panel and 16 available CTAs, 8 warps/CTA.
    //   CTA0: rows 0..7   CTA1: rows 8..15   ... CTA15: rows 120..127
    //          \____________ full K for every row _____________/
    //                  last arrival -> ONE panel ready
    // Complete batches: splits=1, original panel-strided queue, no atomics.
    // This changes producer parallelism, not delivery/consumer granularity.
    // Compile the ordinary queue separately: no division by a runtime '1',
    // stripe predicates or arrival instructions in the large-workload hot loop.
    for (int64_t work = comm; work < (end-begin) * slots; work += comm_ctas) {
      const int64_t task = begin+(SharedPanels ? work / slots : work);
      const int stripe = SharedPanels ? int(work % slots) : 0;
      const int expert = p.order.expert_for_row_tile(task);
      const int m = int(task - p.order.row_tile_offsets[expert]);
      const int64_t start = p.row_offsets[expert];
      const int64_t rows = p.row_offsets[expert + 1] - start;
      const int tail = int(rows - int64_t(m)*TileM);
      const int groups = ((tail < TileM ? tail : TileM) + warps - 1) / warps;
      const int producers = SharedPanels ? (slots < groups ? slots : groups) : 1;
      if (stripe >= producers) continue;  // CTA-uniform; no empty arrivals
      // Full-buffer mode never waits for a consumer. With bounded storage,
      // (expert,m) reuses (expert,m-buffer_m)'s physical slot only after ALL
      // N output tiles have consumed that old input. Compute traverses complete
      // M windows before reusing them (GroupedTileOrder::window_m):
      //   fill window 0 -> consume all N -> reuse slots for window 1 -> ...
      // Each producer/consumer's logical tasks advance monotonically. No task
      // can require overwriting an input still needed by an earlier task.
      if (p.buffer_m && m >= p.buffer_m) {
        if (threadIdx.x == 0)
          while (load_acquire_gpu(p.consumed + task - p.buffer_m) < uint32_t(p.order.n_tiles))
            __nanosleep(64);
        __syncthreads();
      }
#if FUSE_ENABLE_PROFILING
      const uint64_t profile_begin = p.profile.panels && threadIdx.x == 0 ? read_global_timer() : 0;
#endif
      // Shared tail stripes keep their original row ownership and arrival.
      // Small experts are left unchanged while staged candidates are
      // evaluated for M>128. No host count readback or model-name dispatch.
      const bool staged=rows>TileM &&
          int64_t(p.columns)*sizeof(Bf16)<=kStageBytes;
      if(staged) {
        if constexpr(SharedPanels) {
          // Each owner's 8-row stripe is contiguous, but consecutive stripes
          // may belong to other CTAs. Never bulk-store across that gap.
          const int stop=tail<TileM?tail:TileM;
          for(int offset=stripe*warps;offset<stop;offset+=warps*producers)
            copy_staged<CpAsyncG2S>(p,smem,expert,m*TileM+offset,
                stop-offset<warps?stop-offset:warps,start
#if FUSE_ENABLE_PROFILING
                ,summary
#endif
            );
        } else copy_staged<CpAsyncG2S>(p,smem,expert,m*TileM,tail<TileM?tail:TileM,start
#if FUSE_ENABLE_PROFILING
            ,summary
#endif
        );
      }
      else for (int row = m * TileM + stripe * warps + warp;
           row < (m + 1) * TileM && row < rows; row += warps * producers) {
        const auto route = p.source[start + row];
        const Bf16* src = p.input[route.rank] + int64_t(route.token) * p.columns;
        const int physical_row = p.buffer_m ? row % (p.buffer_m*TileM) : row;
        Bf16* dst = p.output[expert] + int64_t(physical_row) * p.columns;
        // Issue independent remote loads before consuming their registers in
        // local stores. Batched vectors/thread expose independent reads while
        // retaining the original CTA budget and full-panel ready boundary.
        // A deeper batch also increases live registers. Evaluate generated
        // loads/spills and complete fusion timing when selecting this depth.
        // Each instruction remains warp-coalesced; tail vectors are masked.
        for (int col = lane * 8; col < p.columns; col += kVectorsPerThread * 32 * 8) {
          uint4 values[kVectorsPerThread];
          CUTLASS_PRAGMA_UNROLL
          for (int i=0;i<kVectorsPerThread;++i)
            if (col+i*32*8 < p.columns)
              values[i]=*reinterpret_cast<const uint4*>(src+col+i*32*8);
          CUTLASS_PRAGMA_UNROLL
          for (int i=0;i<kVectorsPerThread;++i)
            if (col+i*32*8 < p.columns)
              *reinterpret_cast<uint4*>(dst+col+i*32*8)=values[i];
        }
      }
      // Release must follow the writes of every lane, not only the leader.
      __syncthreads();
      if (threadIdx.x == 0) {
        const bool complete = producers == 1 || arrive_grouped_panel(p.arrivals+task)+1 == producers;
        if (complete) {
#if FUSE_ENABLE_PROFILING
          const uint64_t release_begin = p.profile.panels ? read_global_timer() : 0;
#endif
          store_release_gpu(p.ready + task * kReadyFlagStride, epoch);
#if FUSE_ENABLE_PROFILING
          if (p.profile.panels && task < p.profile.panel_capacity)
            p.profile.panels[task] = {profile_begin, release_begin, read_global_timer(), comm, expert, m};
#endif
        }
      }
    }
  }

  // The copy-engine choice is invocation-static. Branch once before entering
  // the panel loop so neither implementation pays a per-batch method check.
  template <bool SharedPanels>
  CUTLASS_DEVICE void copy_selected(const Params& args, char* smem, int comm,
      int comm_ctas, int splits, int64_t begin, int64_t end
#if FUSE_ENABLE_PROFILING
      ,GroupedCommSummary& summary
#endif
  ) {
    if (args.params.cp_async_g2s)
      copy<SharedPanels,true>(args,smem,comm,comm_ctas,splits,begin,end
#if FUSE_ENABLE_PROFILING
          ,summary
#endif
      );
    else
      copy<SharedPanels,false>(args,smem,comm,comm_ctas,splits,begin,end
#if FUSE_ENABLE_PROFILING
          ,summary
#endif
      );
  }

  CUTLASS_DEVICE void finalize(const Params& args) {
    finalize_grouped_transport(args.params);
  }
};

static_assert(GroupedDispatchComm<>::SharedStorageBytes <= sizeof(Bf16GroupedGemmTypes<128,64>::DispatchGemm::SharedStorage));
static_assert(GroupedDispatchComm<>::SharedStorageBytes <= sizeof(Bf16GroupedGemmTypes<128,128>::DispatchGemm::SharedStorage));
static_assert(GroupedDispatchComm<>::SharedStorageBytes <= sizeof(Bf16GroupedGemmTypes<256,64>::DispatchGemm::SharedStorage));
static_assert(GroupedDispatchComm<>::SharedStorageBytes <= sizeof(Bf16GroupedGemmTypes<256,128>::DispatchGemm::SharedStorage));

template <int TileN = 128, int TileK = 64, bool SwapAB = false, bool TrimTokens = true>
using A2AGroupedGemm = GroupedMonolithicGemm<
    typename Bf16GroupedGemmTypes<TileN, TileK, SwapAB, TrimTokens>::DispatchGemm, GroupedDispatchComm<>>;

template <int TileN = 128, int TileK = 64>
using RetiringA2AGroupedGemm = GroupedMonolithicGemm<
    typename Bf16GroupedRetiringGemmTypes<TileN,TileK>::DispatchGemm,
    GroupedDispatchComm<256>,GroupedExplicitPreparePolicy,true>;

static_assert(GroupedDispatchComm<256>::SharedStorageBytes <=
    sizeof(Bf16GroupedRetiringGemmTypes<128,64>::DispatchGemm::SharedStorage));
static_assert(GroupedDispatchComm<256>::SharedStorageBytes <=
    sizeof(Bf16GroupedRetiringGemmTypes<256,64>::DispatchGemm::SharedStorage));

}  // namespace fuse::detail
