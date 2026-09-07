// SPDX-License-Identifier: BSD-3-Clause
// CPU-only toolchain check. CUDA IPC is validated by fused_bf16_mpi separately.
#include <mpi.h>

#include <cstdlib>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) return 1;
  int rank = 0, world = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  const int expected = argc == 2 ? std::atoi(argv[1]) : 0;
  std::vector<int> ranks(world, -1);
  int sum = 0;
  bool passed = expected > 0 && world == expected;
  passed &= MPI_Allgather(&rank, 1, MPI_INT, ranks.data(), 1, MPI_INT,
                          MPI_COMM_WORLD) == MPI_SUCCESS;
  passed &= MPI_Allreduce(&rank, &sum, 1, MPI_INT, MPI_SUM,
                          MPI_COMM_WORLD) == MPI_SUCCESS;
  for (int peer = 0; peer < world; ++peer) passed &= ranks[peer] == peer;
  passed &= sum == world * (world - 1) / 2;
  int local = passed ? 1 : 0, all = 0;
  MPI_Allreduce(&local, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (rank == 0) {
    std::cout << "mpi_cpu_smoke,world=" << world << ",allgather_allreduce="
              << (all ? "PASS" : "FAIL") << '\n';
  }
  MPI_Finalize();
  return all ? 0 : 1;
}
