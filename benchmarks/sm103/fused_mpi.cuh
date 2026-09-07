// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifndef FUSE_BENCH_MPI
#define FUSE_BENCH_MPI 0
#endif
#if FUSE_BENCH_MPI
#include <cuda_runtime.h>
#include <mpi.h>
#endif

// Benchmark-only transport. The default executable has no MPI dependency.
namespace fused_mpi {

inline constexpr bool enabled = FUSE_BENCH_MPI != 0;
inline int process_rank = 0;
inline int process_world = 1;
inline int local_device = 0;

struct RankRange {
  struct Iterator {
    int value;
    int operator*() const { return value; }
    Iterator& operator++() { ++value; return *this; }
    bool operator!=(Iterator other) const { return value != other.value; }
  };
  int first, last;
  Iterator begin() const { return {first}; }
  Iterator end() const { return {last}; }
};

inline RankRange owned_ranks(int world) {
  return enabled ? RankRange{process_rank, process_rank + 1} : RankRange{0, world};
}
inline bool root() { return !enabled || process_rank == 0; }
inline bool owns(int rank) { return !enabled || rank == process_rank; }
inline int device_for(int rank) { return enabled ? local_device : rank; }

inline std::ostream& root_output() {
  class Discard : public std::streambuf {
    int_type overflow(int_type value) override { return traits_type::not_eof(value); }
  };
  static Discard discarded;
  static std::ostream quiet(&discarded);
  return root() ? std::cout : quiet;
}

#if FUSE_BENCH_MPI
inline MPI_Comm communicator = MPI_COMM_NULL;
inline bool initialized = false;

inline void check(int status, const char* operation) {
  if (status == MPI_SUCCESS) return;
  char message[MPI_MAX_ERROR_STRING]{};
  int length = 0;
  MPI_Error_string(status, message, &length);
  throw std::runtime_error(std::string(operation) + ": " + std::string(message, length));
}

inline void initialize(int& argc, char**& argv) {
  int provided = MPI_THREAD_SINGLE;
  check(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided), "MPI_Init_thread");
  initialized = true;
  check(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN), "MPI error handler");
  if (provided < MPI_THREAD_FUNNELED) throw std::runtime_error("MPI_THREAD_FUNNELED is required");
  int total = 0;
  check(MPI_Comm_rank(MPI_COMM_WORLD, &process_rank), "MPI rank");
  check(MPI_Comm_size(MPI_COMM_WORLD, &total), "MPI size");
  check(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, process_rank,
                           MPI_INFO_NULL, &communicator), "MPI shared communicator");
  check(MPI_Comm_set_errhandler(communicator, MPI_ERRORS_RETURN), "MPI local error handler");
  check(MPI_Comm_rank(communicator, &process_rank), "MPI local rank");
  check(MPI_Comm_size(communicator, &process_world), "MPI local size");
  if (process_world != total || (total != 4 && total != 8)) {
    throw std::runtime_error("MPI benchmark requires CP4/8 on one host");
  }
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || (count != 1 && count < total)) {
    throw std::runtime_error("each MPI rank must see one GPU or the whole selected GPU set");
  }
  local_device = count == 1 ? 0 : process_rank;
  char bus_id[32]{};
  if (cudaSetDevice(local_device) != cudaSuccess ||
      cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), local_device) != cudaSuccess) {
    throw std::runtime_error("cannot identify MPI rank GPU");
  }
  std::vector<char> identities(total * sizeof(bus_id));
  check(MPI_Allgather(bus_id, sizeof(bus_id), MPI_BYTE, identities.data(),
                      sizeof(bus_id), MPI_BYTE, communicator), "MPI GPU identities");
  for (int a = 0; a < total; ++a) {
    for (int b = 0; b < a; ++b) {
      if (std::string(identities.data() + a * sizeof(bus_id)) ==
          std::string(identities.data() + b * sizeof(bus_id))) {
        throw std::runtime_error("MPI ranks must own distinct physical GPUs");
      }
    }
  }
}

inline void barrier() { check(MPI_Barrier(communicator), "MPI barrier"); }

inline bool any(bool value) {
  int local = value, result = 0;
  check(MPI_Allreduce(&local, &result, 1, MPI_INT, MPI_MAX, communicator), "MPI any");
  return result != 0;
}

// Collect only owned entries. Wire records are trivial, same-binary objects;
// tensor payloads never travel through this small control-plane operation.
template <class T>
void gather_owned(std::vector<T>& entries) {
  static_assert(std::is_trivially_copyable_v<T>);
  static_assert(sizeof(T) <= static_cast<size_t>(std::numeric_limits<int>::max()));
  if (entries.size() != static_cast<size_t>(process_world)) {
    throw std::runtime_error("MPI rank record count differs from world");
  }
  const T local = entries[process_rank];
  check(MPI_Allgather(&local, sizeof(T), MPI_BYTE, entries.data(), sizeof(T), MPI_BYTE,
                      communicator), "MPI rank records");
}

inline void agree(const std::string& value) {
  uint64_t size = value.size(), root_size = size;
  check(MPI_Bcast(&root_size, 1, MPI_UINT64_T, 0, communicator), "MPI contract size");
  if (root_size > 1024 * 1024) throw std::runtime_error("MPI control contract is too large");
  std::string reference = root() ? value : std::string(root_size, '\0');
  check(MPI_Bcast(reference.data(), static_cast<int>(root_size), MPI_BYTE, 0, communicator),
        "MPI control contract");
  if (any(value != reference)) throw std::runtime_error("MPI ranks disagree on benchmark contract");
}

inline void finalize() {
  if (!initialized) return;
  check(MPI_Comm_free(&communicator), "MPI communicator free");
  check(MPI_Finalize(), "MPI finalize");
  initialized = false;
}

inline void abort(int code) {
  if (initialized) MPI_Abort(MPI_COMM_WORLD, code);
}
#else
inline void initialize(int&, char**&) {}
inline void barrier() {}
inline bool any(bool value) { return value; }
template <class T> void gather_owned(std::vector<T>&) {}
inline void agree(const std::string&) {}
inline void finalize() {}
inline void abort(int) {}
#endif

}  // namespace fused_mpi
