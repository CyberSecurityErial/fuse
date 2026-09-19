// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "fuse/operators/primitives/grouped_gemm.h"
#include <algorithm>
#include <random>
#include <stdexcept>
#include <vector>

// Benchmark-only scalar oracles. No production tile order, ready indexing or
// communication helper is reused here. All elements, including unused capacity
// and unselected branch slots, are checked; only summaries return to the CPU.
namespace grouped_validation {

// Host-only generated routes for correctness, never the performance workload.
// Each pair has identical branches, inputs and weights; the odd replay reverses
// rows within every expert. The independent oracle must follow that permutation.
// Across pairs, the SAME captured graph sees dense, skewed, empty and sparse
// counts. Full-capacity receive storage bounds even the most concentrated route.
inline std::vector<std::vector<fuse::GroupedTokenSource>> property_routes(
    int world, int experts, int tokens, int topk, uint32_t seed, int replay) {
  const int total = world * experts;
  if (world <= 0 || experts <= 0 || tokens <= 0 || topk <= 0 || topk > total || replay < 0)
    throw std::invalid_argument("invalid grouped property workload");
  std::mt19937 rng(seed + uint32_t(replay / 2) * 0x9e3779b9u);
  std::vector<std::vector<fuse::GroupedTokenSource>> rows(total);
  std::vector<int> destinations(total), sources(world * tokens);
  for (int e = 0; e < total; ++e) destinations[e] = e;
  for (int t = 0; t < world * tokens; ++t) sources[t] = t;
  std::shuffle(destinations.begin(), destinations.end(), rng);
  std::shuffle(sources.begin(), sources.end(), rng);
  const int mode = (replay / 2) % 4;
  const int boundaries[] = {1,7,8,9,15,16,17,63,64,65,127,128,129,191,192,193,255,256,257};
  int active = world * tokens;
  if (mode == 1) active = std::min(active, boundaries[rng() % 19]);
  if (mode == 2) active = 0;
  if (mode == 3) active = int(rng() % (uint32_t(active) + 1));
  for (int i = 0; i < active; ++i) {
    // Sampling without replacement keeps expert choices distinct per token.
    // In skew mode the first topk experts receive every selected token.
    if (mode != 1) std::shuffle(destinations.begin(), destinations.end(), rng);
    for (int slot = 0; slot < topk; ++slot)
      rows[destinations[slot]].push_back({sources[i] / tokens, sources[i] % tokens, slot});
  }
  for (auto& expert : rows) {
    std::shuffle(expert.begin(), expert.end(), rng);
    if (replay % 2) std::reverse(expert.begin(), expert.end());
  }
  return rows;
}

struct Peers { const fuse::Bf16* data[fuse::kMaxWorldSize]{}; };
struct BranchOwner { int rank = -1, expert = -1, row = -1; };
struct Result {
  unsigned long long numeric_errors = 0, route_errors = 0, tail_errors = 0;
  float max_abs = 0;
};

template <bool Combine>
static __global__ void reference_lhs(fuse::Bf16* oracle, fuse::Bf16* lhs,
    int experts, int capacity, int k, const int64_t* offsets,
    const fuse::GroupedTokenSource* routes, Peers inputs) {
  const uint64_t count = uint64_t(experts) * capacity * k;
  for (uint64_t i = uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
       i < count; i += uint64_t(gridDim.x)*blockDim.x) {
    const int expert = i / (uint64_t(capacity)*k);
    const int row = (i / k) % capacity;
    fuse::Bf16 value = fuse::Bf16::bitcast(0x7f7f);
    if (row < offsets[expert+1]-offsets[expert]) {
      if constexpr (Combine) value = oracle[i];
      else {
        const auto src = routes[offsets[expert]+row];
        value = inputs.data[src.rank][uint64_t(src.token)*k+i%k];
      }
    }
    oracle[i] = value;
    if constexpr (Combine) lhs[i] = value;
  }
}

static __global__ void check_lhs(const fuse::Bf16* lhs, const fuse::Bf16* oracle,
    uint64_t count, Result* result) {
  unsigned long long errors = 0;
  for (uint64_t i = uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
       i < count; i += uint64_t(gridDim.x)*blockDim.x)
    errors += lhs[i].raw() != oracle[i].raw();
  if (errors) atomicAdd(&result->route_errors, errors);
}

// After circular reuse, each physical row must contain its last logical writer.
// Earlier rows are checked through every GEMM result against the full oracle.
static __global__ void check_bounded_lhs(const fuse::Bf16* lhs, const fuse::Bf16* oracle,
    int experts, int capacity, int buffer_rows, int k, const int64_t* offsets, Result* result) {
  unsigned long long errors=0;
  const uint64_t count=uint64_t(experts)*buffer_rows*k;
  for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=uint64_t(gridDim.x)*blockDim.x) {
    const int e=i/(uint64_t(buffer_rows)*k), row=(i/k)%buffer_rows;
    const int64_t rows=offsets[e+1]-offsets[e];
    uint16_t expected=0x7f7f;
    if(row<rows) {
      const int64_t last=row+(rows-1-row)/buffer_rows*buffer_rows;
      expected=oracle[(uint64_t(e)*capacity+last)*k+i%k].raw();
    }
    errors+=lhs[i].raw()!=expected;
  }
  if(errors) atomicAdd(&result->route_errors,errors);
}

static __global__ void check_gemm(const fuse::Bf16* output, const fuse::Bf16* reference,
    int experts, int capacity, int n, const int64_t* offsets, Result* result) {
  unsigned long long numeric = 0, tails = 0;
  float maximum = 0;
  const uint64_t count = uint64_t(experts)*capacity*n;
  for (uint64_t i = uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
       i < count; i += uint64_t(gridDim.x)*blockDim.x) {
    const int expert = i / (uint64_t(capacity)*n);
    const int row = (i / n) % capacity;
    if (row < offsets[expert+1]-offsets[expert]) {
      const float actual = float(output[i]), expected = float(reference[i]);
      const float error = fabsf(actual-expected);
      numeric += !isfinite(actual) || !isfinite(expected) ||
          error > 0.002f + 0.008f*fabsf(expected);
      maximum = fmaxf(maximum, error);
    } else tails += output[i].raw() != 0x7f7f;
  }
  if (numeric) atomicAdd(&result->numeric_errors, numeric);
  if (tails) atomicAdd(&result->tail_errors, tails);
  atomicMax(reinterpret_cast<unsigned int*>(&result->max_abs), __float_as_uint(maximum));
}

static __global__ void check_branches(const fuse::Bf16* received, uint64_t count,
    const BranchOwner* owners, Peers expert_outputs, int capacity, int n, Result* result) {
  unsigned long long errors = 0;
  for (uint64_t i = uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
       i < count; i += uint64_t(gridDim.x)*blockDim.x) {
    const auto owner = owners[i/n];
    uint16_t expected = 0x7f7f;
    if (owner.rank >= 0)
      expected = expert_outputs.data[owner.rank][
          (uint64_t(owner.expert)*capacity+owner.row)*n+i%n].raw();
    errors += received[i].raw() != expected;
  }
  if (errors) atomicAdd(&result->route_errors, errors);
}

}  // namespace grouped_validation
