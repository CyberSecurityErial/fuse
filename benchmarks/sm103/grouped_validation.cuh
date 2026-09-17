// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "fuse/operators/primitives/grouped_gemm.h"

// Benchmark-only scalar oracles. No production tile order, ready indexing or
// communication helper is reused here. All elements, including unused capacity
// and unselected branch slots, are checked; only summaries return to the CPU.
namespace grouped_validation {

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
