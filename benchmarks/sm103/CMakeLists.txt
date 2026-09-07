cmake_minimum_required(VERSION 3.24)
# Independent baseline build: no Hopper WGMMA translation units are linked.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 103a CACHE STRING "CUDA target architecture")
endif()
project(fuse_sm103a_bench LANGUAGES CXX CUDA)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
find_package(CUDAToolkit 13.0 REQUIRED)
get_filename_component(FUSE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
add_library(fuse_cublaslt_runner SHARED "${FUSE_ROOT}/csrc/baselines/cublaslt_runner.cu")
target_link_libraries(fuse_cublaslt_runner PRIVATE CUDA::cublasLt CUDA::cudart)
add_executable(cublaslt_bench "${FUSE_ROOT}/benchmarks/GEMM/cublaslt_bench.cu")
target_link_libraries(cublaslt_bench PRIVATE CUDA::cublasLt CUDA::cudart)
add_executable(cublaslt_smoke cublaslt_smoke.cu)
target_link_libraries(cublaslt_smoke PRIVATE fuse_cublaslt_runner CUDA::cudart)
foreach(target fuse_cublaslt_runner cublaslt_bench cublaslt_smoke)
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>)
endforeach()
add_library(fuse_sm103_cublaslt SHARED "${FUSE_ROOT}/csrc/baselines/sm103/cublaslt_training.cu")
target_link_libraries(fuse_sm103_cublaslt PRIVATE CUDA::cublasLt CUDA::cudart)
target_compile_options(fuse_sm103_cublaslt PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>)

# Optional, independent pure-GEMM experiment. Existing baseline builds do not
# require CUTLASS and do not compile either of these template instances.
option(FUSE_SM103_BUILD_CUTLASS_BF16 "Build isolated 1-SM/2-SM BF16 GEMM plans" OFF)
if(FUSE_SM103_BUILD_CUTLASS_BF16)
  set(CUTLASS_ROOT "" CACHE PATH "Existing Blackwell-capable CUTLASS checkout")
  set(_fuse_cutlass_builder "${CUTLASS_ROOT}/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl")
  if(NOT EXISTS "${_fuse_cutlass_builder}" OR NOT EXISTS
      "${CUTLASS_ROOT}/tools/util/include/cutlass/util/packed_stride.hpp")
    message(FATAL_ERROR "Set CUTLASS_ROOT to the existing Blackwell CUTLASS checkout")
  endif()
  file(SHA256 "${_fuse_cutlass_builder}" _fuse_cutlass_builder_sha256)
  add_library(fuse_sm103_cutlass_bf16 SHARED
    "${FUSE_ROOT}/csrc/baselines/sm103/cutlass_bf16.cu")
  target_include_directories(fuse_sm103_cutlass_bf16 PRIVATE
    "${CUTLASS_ROOT}/include" "${CUTLASS_ROOT}/tools/util/include")
  target_compile_definitions(fuse_sm103_cutlass_bf16 PRIVATE
    CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1
    FUSE_SM103_CUTLASS_BUILDER_SHA256="${_fuse_cutlass_builder_sha256}")
  target_compile_options(fuse_sm103_cutlass_bf16 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;--expt-extended-lambda;-lineinfo>)
  target_link_libraries(fuse_sm103_cutlass_bf16 PRIVATE CUDA::cudart CUDA::cuda_driver)
  set_target_properties(fuse_sm103_cutlass_bf16 PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endif()
