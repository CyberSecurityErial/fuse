// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "fuse/system_barrier.cuh"

#include <cutlass/cutlass.h>

#include <cstdint>

namespace fuse::detail {

// Single-lane wait for use inside CUTLASS's elect_one_sync() producer branch.
// Do not add a warp barrier here: the other lanes do not enter that branch.
CUTLASS_DEVICE void wait_acquire_gpu_single_lane(
    const uint32_t* flag,
    uint32_t target) {
#pragma unroll 1
  while (load_acquire_gpu(flag) < target) {
    __nanosleep(64);
  }
}

CUTLASS_DEVICE uint64_t read_ready_timer() {
  uint64_t value = 0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

template <bool Instrumented>
struct PeerAcquireRecorder {
  CUTLASS_DEVICE explicit PeerAcquireRecorder(uint64_t*) {}
  CUTLASS_DEVICE void record(int32_t) {}
};

template <>
struct PeerAcquireRecorder<true> {
  uint64_t* timestamps = nullptr;

  CUTLASS_DEVICE explicit PeerAcquireRecorder(uint64_t* values)
      : timestamps(values) {}

  CUTLASS_DEVICE void record(int32_t group) {
    if (timestamps) {
      timestamps[group] = read_ready_timer();
    }
  }
};

// Sequential K-group adapter used by the generic A2A -> GEMM path.
template <class Iterator, bool Instrumented = false>
class ReadyKIterator : private PeerAcquireRecorder<Instrumented> {
 public:
  CUTLASS_DEVICE ReadyKIterator(
      Iterator iterator,
      const uint32_t* ready,
      int32_t ready_group_stride,
      uint32_t target,
      int32_t tiles_per_group,
      int32_t first_k_tile,
      int32_t work_tile_count,
      uint64_t* acquire_timestamps = nullptr)
      : PeerAcquireRecorder<Instrumented>(acquire_timestamps),
        iterator_(iterator),
        ready_(ready),
        target_(target),
        ready_group_stride_(ready_group_stride),
        group_(first_k_tile / tiles_per_group),
        tiles_left_in_group_(
            tiles_per_group - first_k_tile % tiles_per_group),
        tiles_per_group_(tiles_per_group),
        work_tiles_left_(work_tile_count) {}

  CUTLASS_DEVICE decltype(auto) operator*() const { return *iterator_; }

  CUTLASS_DEVICE ReadyKIterator& operator++() {
    ++iterator_;
    --work_tiles_left_;
    --tiles_left_in_group_;
    if (tiles_left_in_group_ == 0 && work_tiles_left_ > 0) {
      ++group_;
      tiles_left_in_group_ = tiles_per_group_;
      wait_acquire_gpu_single_lane(
          ready_ + static_cast<int64_t>(group_) * ready_group_stride_,
          target_);
      this->record(group_);
    }
    return *this;
  }

 private:
  Iterator iterator_;
  const uint32_t* ready_;
  uint32_t target_;
  int32_t ready_group_stride_;
  int32_t group_;
  int32_t tiles_left_in_group_;
  int32_t tiles_per_group_;
  int32_t work_tiles_left_;
};

template <bool Instrumented = false, class Iterator>
CUTLASS_DEVICE auto make_ready_k_iterator(
    Iterator iterator,
    const uint32_t* ready,
    int32_t ready_group_stride,
    uint32_t target,
    int32_t tiles_per_group,
    int32_t first_k_tile,
    int32_t work_tile_count,
    uint64_t* acquire_timestamps = nullptr) {
  return ReadyKIterator<Iterator, Instrumented>(
      iterator, ready, ready_group_stride, target, tiles_per_group,
      first_k_tile, work_tile_count, acquire_timestamps);
}

}  // namespace fuse::detail
