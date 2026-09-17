// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "fuse/arch/common.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace grouped_measurement {

inline void check(cudaError_t status, const char* phase="grouped timer") {
  if (status != cudaSuccess) throw std::runtime_error(std::string(phase)+": "+cudaGetErrorString(status));
}

// Benchmark gate is OUTSIDE sample events. Every rank's complete measurement
// graph is submitted before any rank can pass this gate, so host serial enqueue
// latency cannot appear as communication latency on early ranks.
// The operator's own preparation and cross-rank completion remain INSIDE events.
static __global__ void gate(uint32_t* flags, int rank, int world) {
  if (threadIdx.x != 0) return;
  const uint32_t epoch = fuse::detail::load_acquire_system(flags+rank*fuse::kReadyFlagStride)+1;
  fuse::detail::store_release_system(flags+rank*fuse::kReadyFlagStride,epoch);
  for (int peer=0;peer<world;++peer)
    while (fuse::detail::load_acquire_system(flags+peer*fuse::kReadyFlagStride)<epoch)
      __nanosleep(64);
}

struct Operation { int device; cudaStream_t stream; cudaGraph_t graph; };

class Timer {
  struct Rank {
    Operation op{};
    cudaEvent_t start{}, end{};
    cudaGraph_t graph{};
    cudaGraphExec_t exec{};
  };
  std::vector<Rank> ranks_;
  uint32_t* flags_ = nullptr;
  int root_ = 0;
 public:
  Timer(const std::vector<Operation>& ops, int repetitions=1) : ranks_(ops.size()) {
    if (ops.empty() || repetitions<=0) throw std::runtime_error("empty timing workload");
    try {
    root_=ops[0].device;
    check(cudaSetDevice(root_));
    check(cudaMalloc(reinterpret_cast<void**>(&flags_),ops.size()*fuse::kReadyFlagStride*sizeof(uint32_t)));
    check(cudaMemset(flags_,0,ops.size()*fuse::kReadyFlagStride*sizeof(uint32_t)));
    int world=ops.size();
    for (int index=0;index<world;++index) {
      auto& r=ranks_[index]; r.op=ops[index];
      check(cudaSetDevice(r.op.device));
      check(cudaEventCreate(&r.start)); check(cudaEventCreate(&r.end));
      check(cudaGraphCreate(&r.graph,0));
      cudaKernelNodeParams kernel{};
      kernel.func=reinterpret_cast<void*>(gate);
      kernel.gridDim=dim3(1); kernel.blockDim=dim3(32);
      void* args[]={&flags_,&index,&world}; kernel.kernelParams=args;
      cudaGraphNode_t previous{}, next{};
      check(cudaGraphAddKernelNode(&previous,r.graph,nullptr,0,&kernel));
      check(cudaGraphAddEventRecordNode(&next,r.graph,&previous,1,r.start)); previous=next;
      for (int i=0;i<repetitions;++i) {
        check(cudaGraphAddChildGraphNode(&next,r.graph,&previous,1,r.op.graph),"timer child graph"); previous=next;
      }
      check(cudaGraphAddEventRecordNode(&next,r.graph,&previous,1,r.end));
      check(cudaGraphInstantiate(&r.exec,r.graph,nullptr,nullptr,0),"timer instantiate");
      check(cudaGraphUpload(r.exec,r.op.stream),"timer upload");
    }
    for (const auto& r:ranks_) {
      check(cudaSetDevice(r.op.device)); check(cudaStreamSynchronize(r.op.stream));
    }
    } catch(...) { release(); throw; }
  }
  Timer(const Timer&)=delete;
  ~Timer() { release(); }
 private:
  void release() noexcept {
    for (auto& r:ranks_) {
      cudaSetDevice(r.op.device);
      if(r.exec) cudaGraphExecDestroy(r.exec);
      if(r.graph) cudaGraphDestroy(r.graph);
      if(r.start) cudaEventDestroy(r.start);
      if(r.end) cudaEventDestroy(r.end);
    }
    cudaSetDevice(root_); if(flags_) cudaFree(flags_);
  }
 public:
  std::vector<float> sample() {
    for (const auto& r:ranks_) {
      check(cudaSetDevice(r.op.device)); check(cudaGraphLaunch(r.exec,r.op.stream));
    }
    std::vector<float> times;
    for (const auto& r:ranks_) {
      check(cudaSetDevice(r.op.device)); check(cudaStreamSynchronize(r.op.stream));
      float ms=0; check(cudaEventElapsedTime(&ms,r.start,r.end));
      if (!(ms>0)) throw std::runtime_error("invalid grouped sample duration");
      times.push_back(ms);
    }
    return times;
  }
};

inline double percentile(std::vector<double> values, double q) {
  std::sort(values.begin(),values.end());
  const double position=q*(values.size()-1);
  const size_t lo=size_t(position), hi=std::min(lo+1,values.size()-1);
  return values[lo]+(values[hi]-values[lo])*(position-lo);
}

struct Samples {
  struct Round {
    std::vector<std::vector<float>> ranks_ms;
    double drift;
    int warmup;
  };
  std::vector<Round> rounds;
  std::vector<std::vector<float>> ranks_ms;
  double p50=0, p95=0, drift=0;
  int warmup=0;
};

inline Samples measure(const std::vector<Operation>& ops, bool warm_clocks=true) {
  // Repeated GPU work warms clocks without thousands of host round trips.
  // Formal samples below still contain exactly ONE complete operation.
  Samples out;
  if(warm_clocks) {
    Timer warm(ops,32);
    std::vector<double> elapsed(ops.size(),0);
    for (int round=0;round<4096;++round) {
      const auto times=warm.sample();
      for(size_t r=0;r<times.size();++r) elapsed[r]+=times[r];
      out.warmup+=32;
      if (*std::min_element(elapsed.begin(),elapsed.end())>=100.) break;
    }
    if (*std::min_element(elapsed.begin(),elapsed.end())<100.)
      throw std::runtime_error("grouped warmup budget exhausted");
  }
  Timer timer(ops);
  // First stable round, never the fastest retry. Preserve every rejected
  // round for audit. If all three drift, caller reports stable=0, not a winner.
  for(int round=0;round<3;++round) {
    for(int i=0;i<10;++i) timer.sample();
    out.warmup+=10;
    std::vector<double> maximum;
    out.ranks_ms.clear();
    for(int i=0;i<50;++i) {
      auto times=timer.sample();
      maximum.push_back(*std::max_element(times.begin(),times.end()));
      out.ranks_ms.push_back(std::move(times));
    }
    out.p50=percentile(maximum,0.5); out.p95=percentile(maximum,0.95);
    const double first=percentile(std::vector<double>(maximum.begin(),maximum.begin()+25),0.5);
    const double last=percentile(std::vector<double>(maximum.begin()+25,maximum.end()),0.5);
    out.drift=std::abs(last-first)/std::max(first,last);
    out.rounds.push_back({out.ranks_ms,out.drift,out.warmup});
    if(out.drift<=0.05) break;
  }
  return out;
}

}  // namespace grouped_measurement
