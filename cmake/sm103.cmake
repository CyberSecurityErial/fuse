# Blackwell BF16 fused kernels. The standalone benchmark build under
# benchmarks/sm103 remains usable without building or installing this library.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CUTLASS_ROOT "" CACHE PATH "Local CUTLASS source root with SM100 BF16 collectives")
option(FUSE_ENABLE_PROFILING "Build diagnostic role telemetry kernels" OFF)
option(FUSE_SM103_QKV_RANK_SWIZZLE "Experiment: rank-dependent QKV producer/consumer N-band rotation" OFF)
option(FUSE_BUILD_KERNELS "Build the SM103 BF16 fused operators" ON)
option(FUSE_BUILD_GROUPED_KERNELS "Build independent SM103 BF16 grouped operators" OFF)
option(FUSE_BUILD_BASELINES "Build the independent SM103 cuBLASLt benchmarks" ON)
option(FUSE_BUILD_MPI_BENCH "Build the same full-validation harness with MPI Eager execution" OFF)
if(FUSE_SM103_QKV_RANK_SWIZZLE AND FUSE_ENABLE_PROFILING)
  message(FATAL_ERROR "Rank-rotated profiling ownership is not implemented; disable rank swizzle or profiling")
endif()
if((FUSE_BUILD_KERNELS OR FUSE_BUILD_GROUPED_KERNELS) AND NOT EXISTS
    "${CUTLASS_ROOT}/include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp")
  message(FATAL_ERROR "Set CUTLASS_ROOT to a local Blackwell-capable CUTLASS checkout")
endif()
find_package(CUDAToolkit 13.0 REQUIRED)

if(FUSE_BUILD_KERNELS)
  add_library(fuse_kernels STATIC csrc/operators/sm103/entry.cu)
  if(FUSE_ENABLE_PROFILING)
    target_compile_definitions(fuse_kernels PUBLIC FUSE_ENABLE_PROFILING=1)
  else()
    target_compile_definitions(fuse_kernels PUBLIC FUSE_ENABLE_PROFILING=0)
  endif()
  target_compile_definitions(fuse_kernels PUBLIC FUSE_ARCH_SM103=1)
  target_compile_definitions(fuse_kernels PUBLIC
    FUSE_SM103_QKV_RANK_SWIZZLE=$<BOOL:${FUSE_SM103_QKV_RANK_SWIZZLE}>)
  # CUDA 13's default CUTLASS wrapper resolves the driver entry point on each
  # call. Keep this direct-call variant private to the SM103 implementation;
  # the explicit driver link must propagate to consumers of this static library.
  target_compile_definitions(fuse_kernels PRIVATE CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1)
  target_include_directories(fuse_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUTLASS_ROOT}/include
    ${CUTLASS_ROOT}/tools/util/include)
  target_link_libraries(fuse_kernels PUBLIC CUDA::cudart CUDA::cuda_driver)
  target_compile_options(fuse_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;--expt-extended-lambda;-lineinfo>)
  set_target_properties(fuse_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

  find_package(Threads REQUIRED)
  # Reuse the actual BF16 reverse-route correctness harness, unchanged.
  add_executable(backward_smoke benchmarks/sm90/backward/backward_smoke.cu)
  target_link_libraries(backward_smoke PRIVATE fuse_kernels CUDA::cudart)
  target_compile_options(backward_smoke PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr>)
  add_executable(backward_mxfp8_smoke benchmarks/sm103/backward/mxfp8_smoke.cu)
  target_link_libraries(backward_mxfp8_smoke PRIVATE fuse_kernels CUDA::cudart CUDA::cublas)
  target_compile_options(backward_mxfp8_smoke PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr>)
  add_executable(fused_bf16 benchmarks/sm103/fused_bf16.cu)
  target_link_libraries(fused_bf16 PRIVATE fuse_kernels CUDA::cublas CUDA::cudart Threads::Threads)
  target_compile_options(fused_bf16 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;--expt-extended-lambda;-lineinfo>)
  # Same validation/measurement harness, explicit precision target. No copied
  # communication benchmark or mixing of BF16 and MXFP8 result boundaries.
  add_executable(fused_mxfp8 benchmarks/sm103/fused_bf16.cu)
  target_compile_definitions(fused_mxfp8 PRIVATE FUSE_BENCH_MXFP8=1)
  target_link_libraries(fused_mxfp8 PRIVATE fuse_kernels CUDA::cublas CUDA::cudart Threads::Threads)
  target_compile_options(fused_mxfp8 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;--expt-extended-lambda;-lineinfo>)
  if(FUSE_BUILD_MPI_BENCH)
    find_package(MPI REQUIRED COMPONENTS CXX)
    add_executable(backward_mxfp8_mpi benchmarks/sm103/backward/mxfp8_mpi_bench.cu)
    target_compile_definitions(backward_mxfp8_mpi PRIVATE FUSE_BENCH_MPI=1)
    target_link_libraries(backward_mxfp8_mpi PRIVATE fuse_kernels CUDA::cudart CUDA::cublas MPI::MPI_CXX)
    target_compile_options(backward_mxfp8_mpi PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr>)
    add_executable(fused_mxfp8_mpi benchmarks/sm103/fused_bf16.cu)
    target_compile_definitions(fused_mxfp8_mpi PRIVATE FUSE_BENCH_MXFP8=1 FUSE_BENCH_MPI=1)
    target_link_libraries(fused_mxfp8_mpi PRIVATE
      fuse_kernels CUDA::cublas CUDA::cudart Threads::Threads MPI::MPI_CXX)
    target_compile_options(fused_mxfp8_mpi PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;--expt-extended-lambda;-lineinfo>)
    add_executable(backward_mpi_bench benchmarks/sm90/backward/backward_mpi_bench.cu)
    target_link_libraries(backward_mpi_bench PRIVATE fuse_kernels CUDA::cudart CUDA::cublas MPI::MPI_CXX)
    target_compile_options(backward_mpi_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr>)
    add_executable(fused_bf16_mpi benchmarks/sm103/fused_bf16.cu)
    target_compile_definitions(fused_bf16_mpi PRIVATE FUSE_BENCH_MPI=1)
    target_link_libraries(fused_bf16_mpi PRIVATE
      fuse_kernels CUDA::cublas CUDA::cudart Threads::Threads MPI::MPI_CXX)
    target_compile_options(fused_bf16_mpi PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;--expt-extended-lambda;-lineinfo>)
  endif()
endif()

if(FUSE_BUILD_GROUPED_KERNELS)
  set(DEEPGEMM_ROOT "" CACHE PATH "Optional pinned DeepGEMM source for benchmark-only native reference")
  add_library(fuse_grouped_kernels STATIC csrc/operators/sm103/grouped_entry.cu)
  target_include_directories(fuse_grouped_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include ${CUTLASS_ROOT}/include ${CUTLASS_ROOT}/tools/util/include)
  target_compile_definitions(fuse_grouped_kernels PRIVATE CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1)
  target_compile_definitions(fuse_grouped_kernels PUBLIC FUSE_ENABLE_PROFILING=$<BOOL:${FUSE_ENABLE_PROFILING}>)
  target_link_libraries(fuse_grouped_kernels PUBLIC CUDA::cudart CUDA::cuda_driver)
  target_compile_options(fuse_grouped_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;--expt-extended-lambda;-lineinfo>)
  set_target_properties(fuse_grouped_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
  add_executable(grouped_bf16 benchmarks/sm103/grouped_bf16.cu
    csrc/baselines/sm103/cublaslt_training.cu)
  target_link_libraries(grouped_bf16 PRIVATE fuse_grouped_kernels CUDA::cublas CUDA::cublasLt)
  target_compile_options(grouped_bf16 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;--expt-extended-lambda;-lineinfo>)
  if(DEEPGEMM_ROOT)
    add_library(grouped_deepgemm STATIC csrc/baselines/sm103/deepgemm_grouped.cu)
    target_include_directories(grouped_deepgemm PRIVATE ${DEEPGEMM_ROOT}/deep_gemm/include
      ${DEEPGEMM_ROOT}/third-party/cutlass/include)
    target_link_libraries(grouped_deepgemm PUBLIC CUDA::cudart CUDA::cuda_driver)
    target_compile_options(grouped_deepgemm PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;--expt-extended-lambda;-lineinfo>)
    set_target_properties(grouped_deepgemm PROPERTIES CUDA_STANDARD 20)
    target_compile_definitions(grouped_bf16 PRIVATE FUSE_GROUPED_EXTERNAL=1)
    target_link_libraries(grouped_bf16 PRIVATE grouped_deepgemm)
  endif()
endif()

if(FUSE_BUILD_BASELINES)
  add_subdirectory(benchmarks/sm103)
endif()
