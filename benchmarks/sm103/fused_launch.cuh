// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <condition_variable>
#include <cstdint>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace fused_launch {

struct Result {
  int rank = -1;
  std::exception_ptr error;
};

// One main-thread caller, one persistent worker per rank. A callback only
// enqueues work: its successful return must include recording the end event.
// GPU completion and the process watchdog remain the caller's responsibility.
class Team {
 public:
  using Function = void (*)(void*, int);

  explicit Team(int ranks) {
    if (ranks <= 0) throw std::invalid_argument("launch team requires positive ranks");
    workers_.reserve(ranks);
    try {
      for (int rank = 0; rank < ranks; ++rank) workers_.emplace_back([this, rank] { worker(rank); });
    } catch (...) {
      stop(); // No jobs have been submitted during construction.
      throw;
    }
  }

  Team(const Team&) = delete;
  Team& operator=(const Team&) = delete;
  ~Team() { stop(); }

  Result dispatch(Function function, void* context) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (failure_.error) return failure_;
    if (stopping_ || in_flight_ || !function) {
      return {-1, std::make_exception_ptr(std::runtime_error("invalid launch team dispatch"))};
    }
    function_ = function;
    context_ = context;
    completed_ = 0;
    in_flight_ = true;
    ++generation_; // Mailbox generation is independent of either GPU ready epoch.
    work_.notify_all();
    done_.wait(lock, [&] { return completed_ == workers_.size() || failure_.error; });
    in_flight_ = false;
    return failure_;
  }

  // Call only on the normal path after GPU completion. An enqueue failure
  // may leave another worker in a CUDA call: the harness must fatal-exit
  // without unwinding through this destructor or attempting a blocking join.
  void stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    work_.notify_all();
    for (auto& thread : workers_) if (thread.joinable()) thread.join();
  }

 private:
  void worker(int rank) {
    uint64_t observed_generation = 0;
    while (true) {
      Function function;
      void* context;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        work_.wait(lock, [&] { return stopping_ || generation_ != observed_generation; });
        if (stopping_) return;
        observed_generation = generation_;
        function = function_;
        context = context_;
      }
      std::exception_ptr error;
      try {
        function(context, rank);
      } catch (...) {
        error = std::current_exception();
      }
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (error && !failure_.error) failure_ = {rank, error};
        ++completed_;
      }
      done_.notify_one();
    }
  }

  std::mutex mutex_;
  std::condition_variable work_, done_;
  std::vector<std::thread> workers_;
  Function function_ = nullptr;
  void* context_ = nullptr;
  uint64_t generation_ = 0;
  size_t completed_ = 0;
  bool stopping_ = false, in_flight_ = false;
  Result failure_;
};

}  // namespace fused_launch
