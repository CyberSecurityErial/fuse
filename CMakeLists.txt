cmake_minimum_required(VERSION 3.24)
project(fuse_attn_cp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 90a)

set(CUTLASS_ROOT
    "/home/chen/workspace/source_code/TransformerEngine/3rdparty/cutlass"
    CACHE PATH "Local CUTLASS source root")
option(
  FUSE_ENABLE_PROFILING
  "Build diagnostic %globaltimer profiling kernels"
  OFF)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

add_library(
  fuse_kernels STATIC
  csrc/operators/ulysses_sm90.cu)
if(FUSE_ENABLE_PROFILING)
  target_compile_definitions(fuse_kernels PUBLIC FUSE_ENABLE_PROFILING=1)
else()
  target_compile_definitions(fuse_kernels PUBLIC FUSE_ENABLE_PROFILING=0)
endif()
target_include_directories(
  fuse_kernels
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUTLASS_ROOT}/include
    ${CUTLASS_ROOT}/tools/util/include)
target_link_libraries(fuse_kernels PUBLIC CUDA::cudart)
target_compile_options(
  fuse_kernels
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;--expt-relaxed-constexpr;--expt-extended-lambda;-lineinfo>)
set_target_properties(fuse_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

add_executable(fuse_smoke "benchmarks/a2a+Oproj/fuse_smoke.cu")
target_link_libraries(fuse_smoke PRIVATE fuse_kernels CUDA::cublas CUDA::cudart)
target_compile_options(
  fuse_smoke
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;--expt-relaxed-constexpr;--expt-extended-lambda;-lineinfo>)

add_executable(fuse_bench "benchmarks/a2a+Oproj/fuse_bench.cu")
target_link_libraries(
  fuse_bench PRIVATE fuse_kernels CUDA::cublas CUDA::cublasLt CUDA::cudart)
target_compile_options(
  fuse_bench
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;--expt-relaxed-constexpr;--expt-extended-lambda;-lineinfo>)

add_executable(qkvproj_a2a_smoke "benchmarks/QKVproj+a2a/fuse_smoke.cu")
target_link_libraries(
  qkvproj_a2a_smoke PRIVATE fuse_kernels CUDA::cublas CUDA::cudart)
target_compile_options(
  qkvproj_a2a_smoke
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;--expt-relaxed-constexpr;--expt-extended-lambda;-lineinfo>)

add_executable(qkvproj_a2a_bench "benchmarks/QKVproj+a2a/fuse_bench.cu")
target_link_libraries(
  qkvproj_a2a_bench PRIVATE fuse_kernels CUDA::cublas CUDA::cublasLt CUDA::cudart)
target_compile_options(
  qkvproj_a2a_bench
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;--expt-relaxed-constexpr;--expt-extended-lambda;-lineinfo>)

add_executable(backward_smoke "benchmarks/backward/backward_smoke.cu")
target_link_libraries(backward_smoke PRIVATE fuse_kernels CUDA::cudart)
target_compile_options(
  backward_smoke
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;--expt-relaxed-constexpr;--expt-extended-lambda;-lineinfo>)

# The focused one-process-per-GPU benchmark is optional so the core library and
# single-process tools remain buildable on systems without an MPI development
# package. CUDA IPC supplies the peer data/control mappings; MPI is bootstrap
# and host-side sample reduction only.
find_package(MPI QUIET COMPONENTS CXX)
if(MPI_CXX_FOUND)
  add_executable(
    qkvproj_a2a_mpi_bench "benchmarks/QKVproj+a2a/fuse_mpi_bench.cu")
  target_link_libraries(
    qkvproj_a2a_mpi_bench PRIVATE fuse_kernels CUDA::cudart MPI::MPI_CXX)
  target_compile_options(
    qkvproj_a2a_mpi_bench
    PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;--expt-relaxed-constexpr;--expt-extended-lambda;-lineinfo>)
  add_executable(
    backward_mpi_bench "benchmarks/backward/backward_mpi_bench.cu")
  target_link_libraries(
    backward_mpi_bench PRIVATE fuse_kernels CUDA::cudart MPI::MPI_CXX)
  target_compile_options(
    backward_mpi_bench
    PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;--expt-relaxed-constexpr;--expt-extended-lambda;-lineinfo>)
  add_executable(
    backward_cublas_mpi_bench
    "benchmarks/backward/backward_cublas_mpi_bench.cu")
  target_link_libraries(
    backward_cublas_mpi_bench
    PRIVATE CUDA::cublas CUDA::cudart MPI::MPI_CXX)
  target_compile_options(
    backward_cublas_mpi_bench
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>)
else()
  message(STATUS "MPI C++ not found; skipping MPI benchmark targets")
endif()

add_executable(cublaslt_bench "benchmarks/GEMM/cublaslt_bench.cu")
target_link_libraries(cublaslt_bench PRIVATE CUDA::cublasLt CUDA::cudart)
target_compile_options(
  cublaslt_bench
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>)

add_library(fuse_cublaslt_runner SHARED csrc/baselines/cublaslt_runner.cu)
target_link_libraries(fuse_cublaslt_runner PRIVATE CUDA::cublasLt CUDA::cudart)
target_compile_options(
  fuse_cublaslt_runner
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>)

add_executable(
  heterogeneous_cp_plan_test
  "benchmarks/heterogeneous_cp/plan_test.cpp")
target_link_libraries(heterogeneous_cp_plan_test PRIVATE fuse_kernels)

add_executable(
  heterogeneous_cp_weighted_bench
  "benchmarks/heterogeneous_cp/weighted_bench.cu")
target_link_libraries(
  heterogeneous_cp_weighted_bench
  PRIVATE fuse_kernels CUDA::cudart Threads::Threads)
target_compile_options(
  heterogeneous_cp_weighted_bench
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;--expt-relaxed-constexpr;--expt-extended-lambda;-lineinfo>)
