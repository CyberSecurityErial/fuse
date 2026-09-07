// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Private host diagnostics. Nothing is declared or executed in production
// builds; enabled builds record only when the owning launch thread binds a sink.
#if FUSE_ENABLE_PROFILING
#include <chrono>
#include <cstdint>

namespace fuse::detail {

enum class HostLaunchStage : unsigned {
  // Includes the initial Comm::Arguments zero/parameter copy. Communication
  // validation and descriptor construction belong to the next interval.
  kPolicyDeviceValidation,
  kCommunicationPrepare,
  kArguments,
  kImplementWorkspace,
  kLowerParameters,
  kLaunchSetup,
  kCudaEnqueue,
  kEnd,
};

struct HostLaunchRecord {
  uint64_t outer_begin_ns = 0;
  uint64_t outer_end_ns = 0;
  uint64_t stamp_ns[8]{};
  uint64_t api_return_ns = 0;
  uint64_t descriptor_ns[2]{}; // Local descriptor / peer descriptor loop.
  uint32_t descriptor_count[2]{}; // Attempted driver encodes, including failures.
  uint32_t stage_mask = 0;
  int32_t status = -1;
  bool protocol_error = false;
};

// Exactly one definition in entry.cu, shared with the benchmark translation
// unit. Never use an anonymous/header-static sink: those split the records.
extern thread_local HostLaunchRecord* host_launch_sink;

inline uint64_t host_timestamp_ns() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
}

inline void mark_host_launch_stage(HostLaunchStage stage) {
  if (auto* record = host_launch_sink) {
    const unsigned index = static_cast<unsigned>(stage);
    const uint32_t bit = uint32_t{1} << index;
    record->protocol_error |= record->stage_mask != bit - 1;
    record->stamp_ns[index] = host_timestamp_ns();
    record->stage_mask |= bit;
  }
}

class HostLaunchBinding {
 public:
  explicit HostLaunchBinding(HostLaunchRecord* record) : previous_(host_launch_sink) {
    if (record) *record = {};
    host_launch_sink = record;
  }
  ~HostLaunchBinding() { host_launch_sink = previous_; }
  HostLaunchBinding(const HostLaunchBinding&) = delete;
  HostLaunchBinding& operator=(const HostLaunchBinding&) = delete;
 private:
  HostLaunchRecord* previous_;
};

class HostLaunchScope {
 public:
  HostLaunchScope() : record_(host_launch_sink) {
    mark_host_launch_stage(HostLaunchStage::kPolicyDeviceValidation);
  }
  ~HostLaunchScope() {
    if (record_) record_->api_return_ns = host_timestamp_ns();
  }
  template <class Status>
  Status finish(Status status) {
    if (record_) record_->status = static_cast<int32_t>(status);
    return status;
  }
 private:
  HostLaunchRecord* record_;
};

class HostDescriptorScope {
 public:
  explicit HostDescriptorScope(unsigned group)
      : record_(host_launch_sink), group_(group), begin_(record_ ? host_timestamp_ns() : 0) {}
  ~HostDescriptorScope() {
    if (record_) record_->descriptor_ns[group_] += host_timestamp_ns() - begin_;
  }
  void attempted() {
    if (record_) ++record_->descriptor_count[group_];
  }
 private:
  HostLaunchRecord* record_;
  unsigned group_;
  uint64_t begin_;
};

}  // namespace fuse::detail

#define FUSE_SM103_HOST_BEGIN() ::fuse::detail::HostLaunchScope host_launch_scope
#define FUSE_SM103_HOST_MARK(stage) \
  ::fuse::detail::mark_host_launch_stage(::fuse::detail::HostLaunchStage::stage)
#define FUSE_SM103_HOST_RETURN(status) return host_launch_scope.finish(status)
#define FUSE_SM103_HOST_DESCRIPTOR_SCOPE(name, group) \
  ::fuse::detail::HostDescriptorScope name(group)
#define FUSE_SM103_HOST_DESCRIPTOR_ATTEMPT(name) name.attempted()
#else
#define FUSE_SM103_HOST_BEGIN() ((void)0)
#define FUSE_SM103_HOST_MARK(stage) ((void)0)
#define FUSE_SM103_HOST_RETURN(status) return (status)
#define FUSE_SM103_HOST_DESCRIPTOR_SCOPE(name, group) ((void)0)
#define FUSE_SM103_HOST_DESCRIPTOR_ATTEMPT(name) ((void)0)
#endif
