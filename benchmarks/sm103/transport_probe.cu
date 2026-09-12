// SPDX-License-Identifier: BSD-3-Clause
// B300 remote-read microbenchmark, separate from fused-operator performance.
// G2S compares a common SMEM endpoint. G2G compares a common local-GMEM endpoint:
// LD/ST is direct, cp.async uses SMEM then stores, TMA uses G2S then bulk S2G.
// Timers include software issue/wait/join; zero-byte subtraction is NOT an
// instruction's intrinsic hardware latency. No quantization or GEMM runs here.
#include <cuda_runtime.h>
#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int kSlots = 32, kRepeats = 32, kStride = 128 * 1024;
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
__device__ uint64_t timer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
  return t;
}
__host__ __device__ uint32_t pattern(size_t i) {
  uint32_t x = static_cast<uint32_t>(i) ^ 0x5ea71349u;
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu;
  return x ^ (x >> 16);
}
__global__ void fill(uint32_t* data, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += size_t{gridDim.x} * blockDim.x) data[i] = pattern(i);
}
template<int Method, bool G2G>
__global__ void transfer(const char* remote, char* output, uint64_t* times, int bytes) {
  extern __shared__ __align__(128) char storage[];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int warps = blockDim.x / 32, worker = blockIdx.x * warps + warp;
  const int stage_bytes = bytes ? bytes : 16;
  auto* stage = reinterpret_cast<uint4*>(storage + warp * stage_bytes);
  auto* barrier = reinterpret_cast<uint64_t*>(storage + warps * stage_bytes) + warp;
  if (lane == 0) {
    cute::initialize_barrier(*barrier, 1);
    cutlass::arch::fence_barrier_init();
  }
  __syncwarp();
  int phase = 0;
  for (int repeat = 0; repeat < kRepeats; ++repeat) {
    const size_t offset = (size_t{worker} * kSlots + repeat % kSlots) * kStride;
    const auto* src = reinterpret_cast<const uint4*>(remote + offset);
    auto* dst = reinterpret_cast<uint4*>(output + offset);
    __syncwarp();
    const uint64_t begin = timer();
    if (bytes) {
      if constexpr (Method == 0 && G2G) {
        // Independent vector reads precede stores, as in the production SF path.
        for (int i = lane; i < bytes / 16; i += 32 * 4) {
          uint4 values[4];
#pragma unroll
          for (int j = 0; j < 4; ++j)
            if (i + j * 32 < bytes / 16) values[j] = src[i + j * 32];
#pragma unroll
          for (int j = 0; j < 4; ++j)
            if (i + j * 32 < bytes / 16) dst[i + j * 32] = values[j];
        }
        __threadfence();
      } else {
        if constexpr (Method == 0) {
          for (int i = lane; i < bytes / 16; i += 32) stage[i] = src[i];
        } else if constexpr (Method == 1 || Method == 3) {
          for (int i = lane; i < bytes / 16; i += 32)
            cute::SM80_CP_ASYNC_CACHEGLOBAL<uint4>::copy(src[i], stage[i]);
          cute::cp_async_fence();
          cute::cp_async_wait<0>();
        } else if (lane == 0) {
          cute::set_barrier_transaction_bytes(*barrier, bytes);
          cute::SM90_BULK_COPY_G2S::copy(src, barrier, stage, bytes);
          cute::wait_barrier(*barrier, phase);
          phase ^= 1;
        }
        __syncwarp();
        if constexpr (G2G) {
          if constexpr (Method == 2 || Method == 3) {
            if (lane == 0) {
              cute::tma_store_fence();
              cute::SM90_BULK_COPY_S2G::copy(stage, dst, bytes);
              cute::tma_store_arrive();
              asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
            }
          } else {
            for (int i = lane; i < bytes / 16; i += 32) dst[i] = stage[i];
            __threadfence();
          }
        }
      }
    }
    __syncwarp();
    const uint64_t end = timer();
    if (lane == 0) times[worker * kRepeats + repeat] = end - begin;
  }
  // Preserve and validate the last G2S tile outside its measured span.
  if constexpr (!G2G) {
    auto* dst = reinterpret_cast<uint4*>(output +
        (size_t{worker} * kSlots + (kRepeats - 1) % kSlots) * kStride);
    for (int i = lane; i < bytes / 16; i += 32) dst[i] = stage[i];
  }
  __syncwarp();
  if (lane == 0) cutlass::arch::ClusterBarrier::invalidate(barrier);
}
__global__ void validate(const char* output, int bytes, int workers, bool g2g, unsigned* errors) {
  const int slots = g2g ? kSlots : 1;
  const size_t count = size_t{workers} * slots * (bytes / 4);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += size_t{gridDim.x} * blockDim.x) {
    const int word = i % (bytes / 4), slot = (i / (bytes / 4)) % slots;
    const size_t worker = i / (bytes / 4) / slots;
    const size_t idx = (worker * kSlots + (g2g ? slot : kSlots - 1)) * (kStride / 4) + word;
    if (reinterpret_cast<const uint32_t*>(output)[idx] != pattern(idx)) atomicAdd(errors, 1u);
  }
}
double quantile(std::vector<double> values, double q) {
  std::sort(values.begin(), values.end());
  const double p = q * (values.size() - 1);
  const size_t i = static_cast<size_t>(p);
  return values[i] + (values[std::min(i + 1, values.size() - 1)] - values[i]) * (p - i);
}
template<int Method, bool G2G>
void measure(std::ofstream& csv, const char* src, char* dst, uint64_t* times,
             unsigned* errors, int bytes, int ctas, int warps) {
  const int workers = ctas * warps;
  const int smem = warps * (bytes ? bytes : 16) + warps * sizeof(uint64_t);
  check(cudaFuncSetAttribute(transfer<Method,G2G>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  auto launch = [&] { transfer<Method,G2G><<<ctas,warps*32,smem>>>(src,dst,times,bytes); };
  for (int i = 0; i < 10; ++i) launch();
  check(cudaGetLastError()); check(cudaDeviceSynchronize());
  std::vector<uint64_t> raw(workers * kRepeats);
  std::vector<double> latency, batch;
  cudaEvent_t begin, end;
  check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
  for (int i = 0; i < 50; ++i) {
    check(cudaEventRecord(begin)); launch(); check(cudaEventRecord(end));
    check(cudaEventSynchronize(end));
    float ms = 0; check(cudaEventElapsedTime(&ms,begin,end)); batch.push_back(ms*1000.0);
    check(cudaMemcpy(raw.data(),times,raw.size()*sizeof(uint64_t),cudaMemcpyDeviceToHost));
    for (auto ns : raw) latency.push_back(ns / 1000.0);
  }
  check(cudaMemset(errors,0,sizeof(unsigned)));
  if (bytes) validate<<<256,256>>>(dst,bytes,workers,G2G,errors);
  unsigned failed = 0;
  check(cudaMemcpy(&failed,errors,sizeof(unsigned),cudaMemcpyDeviceToHost));
  if (failed) throw std::runtime_error("transport bytewise validation failed");
  const double p50 = quantile(latency,.5), p95 = quantile(latency,.95);
  const double batch_us = quantile(batch,.5);
  const double drift = quantile(std::vector<double>(batch.begin()+25,batch.end()),.5) /
      quantile(std::vector<double>(batch.begin(),batch.begin()+25),.5) - 1;
  csv << (G2G?"g2g":"g2s") << ',' << ctas << ',' << warps << ','
      << (Method==0?"ldst":Method==1?"cpasync":Method==2?"tma":"cpasync_tma") << ',' << bytes << ','
      << p50 << ',' << p95 << ',' << batch_us << ','
      << (bytes ? bytes / p50 / 1000.0 : 0) << ','
      << (G2G ? double(bytes)*workers*kRepeats/batch_us/1000.0 : 0) << ',' << drift << ",0\n";
  csv.flush();
  check(cudaEventDestroy(begin)); check(cudaEventDestroy(end));
}
}
int main(int argc,char** argv) try {
  if (argc != 2) throw std::runtime_error("usage: transport_probe output.csv");
  int count=0, access=0; check(cudaGetDeviceCount(&count));
  if (count != 2) throw std::runtime_error("exactly two visible GPUs required");
  check(cudaDeviceCanAccessPeer(&access,0,1));
  if (!access) throw std::runtime_error("GPU0 cannot access peer GPU1");
  cudaDeviceProp props{}; check(cudaGetDeviceProperties(&props,0));
  if (props.major != 10 || props.minor != 3) throw std::runtime_error("expected runtime SM103");
  std::printf("DEVICE runtime_sm=%d%d sms=%d reported_name=%s; user identifies B300\n",props.major,props.minor,props.multiProcessorCount,props.name);
  constexpr size_t allocation = size_t{80}*kSlots*kStride;
  char *src=nullptr,*dst=nullptr; uint64_t* times=nullptr; unsigned* errors=nullptr;
  check(cudaSetDevice(1)); check(cudaMalloc(&src,allocation));
  fill<<<2048,256>>>(reinterpret_cast<uint32_t*>(src),allocation/4); check(cudaDeviceSynchronize());
  check(cudaSetDevice(0)); check(cudaDeviceEnablePeerAccess(1,0));
  check(cudaMalloc(&dst,allocation)); check(cudaMemset(dst,0,allocation));
  check(cudaMalloc(&times,80*kRepeats*sizeof(uint64_t))); check(cudaMalloc(&errors,sizeof(unsigned)));
  check(cudaFuncSetAttribute(transfer<2,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,4*16384+32));
  const auto heat_start=std::chrono::steady_clock::now();
  do {
    transfer<2,true><<<20,128,4*16384+32>>>(src,dst,times,16384);
    check(cudaDeviceSynchronize());
  } while(std::chrono::duration<double>(std::chrono::steady_clock::now()-heat_start).count()<.25);
  std::ofstream csv(argv[1]);
  if (!csv) throw std::runtime_error("cannot open CSV");
  csv << "endpoint,ctas,warps_per_cta,method,bytes,latency_p50_us,latency_p95_us,batch_p50_us,per_worker_gbps,aggregate_gbps,half_drift,bitwise_errors\n";
  std::vector<int> sizes{0,128,256,384,512,640,768,896,1024,1280,1536,1792,2048,2560,3072,3584,4096};
  for(int n=5120;n<=16384;n+=1024) sizes.push_back(n);
  for(int n=18432;n<=49152;n+=2048) sizes.push_back(n);
  for(int n=57344;n<=131072;n+=8192) sizes.push_back(n);
  int done=0;
  for(int ctas:{1,20}) for(int bytes:sizes) {
    const int warps=ctas==1?1:4;
    if(bytes*warps>192*1024) continue;
    measure<0,false>(csv,src,dst,times,errors,bytes,ctas,warps);
    measure<1,false>(csv,src,dst,times,errors,bytes,ctas,warps);
    measure<2,false>(csv,src,dst,times,errors,bytes,ctas,warps);
    measure<0,true>(csv,src,dst,times,errors,bytes,ctas,warps);
    measure<1,true>(csv,src,dst,times,errors,bytes,ctas,warps);
    measure<2,true>(csv,src,dst,times,errors,bytes,ctas,warps);
    measure<3,true>(csv,src,dst,times,errors,bytes,ctas,warps);
    std::printf("DONE transport %d bytes=%d ctas=%d warps=%d full_check=passed\n",++done,bytes,ctas,warps); std::fflush(stdout);
  }
  check(cudaFree(errors)); check(cudaFree(times)); check(cudaFree(dst));
  check(cudaDeviceDisablePeerAccess(1)); check(cudaSetDevice(1)); check(cudaFree(src));
  return 0;
} catch(const std::exception& e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
