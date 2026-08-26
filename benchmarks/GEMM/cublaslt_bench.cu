#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#define CUDA_CHECK(expr) check_cuda((expr), #expr, __FILE__, __LINE__)
#define CUBLAS_CHECK(expr) check_cublas((expr), #expr, __FILE__, __LINE__)

void check_cuda(cudaError_t status, const char* expr, const char* file, int line) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(file) + ":" + std::to_string(line) + " " + expr + ": " +
        cudaGetErrorString(status));
  }
}

void check_cublas(cublasStatus_t status, const char* expr, const char* file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
        std::string(file) + ":" + std::to_string(line) + " " + expr +
        ": cuBLAS status " + std::to_string(static_cast<int>(status)));
  }
}

struct Options {
  int m = 4096;
  int n = 10240;
  int k = 8192;
  int batches = 1;
  int warmup = 10;
  int iterations = 30;
  int tune_warmup = 2;
  int tune_iterations = 5;
  int workspace_mib = 32;
  int candidates = 64;
};

int positive(const char* value, const char* name) {
  const int result = std::stoi(value);
  if (result <= 0) {
    throw std::runtime_error(std::string(name) + " must be positive");
  }
  return result;
}

Options parse_options(int argc, char** argv) {
  Options result;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto take = [&](const char* name) {
      if (++i == argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      return argv[i];
    };
    if (arg == "--m") result.m = positive(take("--m"), "--m");
    else if (arg == "--n") result.n = positive(take("--n"), "--n");
    else if (arg == "--k") result.k = positive(take("--k"), "--k");
    else if (arg == "--batches") result.batches = positive(take("--batches"), "--batches");
    else if (arg == "--warmup") result.warmup = positive(take("--warmup"), "--warmup");
    else if (arg == "--iterations") result.iterations = positive(take("--iterations"), "--iterations");
    else if (arg == "--tune-warmup") result.tune_warmup = positive(take("--tune-warmup"), "--tune-warmup");
    else if (arg == "--tune-iterations") result.tune_iterations = positive(take("--tune-iterations"), "--tune-iterations");
    else if (arg == "--workspace-mib") result.workspace_mib = positive(take("--workspace-mib"), "--workspace-mib");
    else if (arg == "--candidates") result.candidates = positive(take("--candidates"), "--candidates");
    else throw std::runtime_error("unknown argument: " + arg);
  }
  return result;
}

__global__ void fill_bf16(__nv_bfloat16* data, int64_t size, int seed) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < size;
       i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float value = static_cast<float>((i * 17 + seed * 13) % 19 - 9) / 32.0f;
    data[i] = __float2bfloat16(value);
  }
}

struct Descriptors {
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t a = nullptr;
  cublasLtMatrixLayout_t b = nullptr;
  cublasLtMatrixLayout_t c = nullptr;
  cublasLtMatrixLayout_t d = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
};

Descriptors make_descriptors(const Options& o, size_t workspace_bytes) {
  Descriptors desc;
  CUBLAS_CHECK(cublasLtMatmulDescCreate(
      &desc.operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  const cublasOperation_t trans_a = CUBLAS_OP_N;
  const cublasOperation_t trans_b = CUBLAS_OP_T;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      desc.operation, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      desc.operation, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));

  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&desc.a, CUDA_R_16BF, o.m, o.k, o.k));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&desc.b, CUDA_R_16BF, o.n, o.k, o.k));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&desc.c, CUDA_R_16BF, o.m, o.n, o.n));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&desc.d, CUDA_R_16BF, o.m, o.n, o.n));
  const cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
  for (auto layout : {desc.a, desc.b, desc.c, desc.d}) {
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
  }
  if (o.batches > 1) {
    const int32_t batch_count = o.batches;
    const int64_t stride_a = static_cast<int64_t>(o.m) * o.k;
    const int64_t stride_b = static_cast<int64_t>(o.n) * o.k;
    const int64_t stride_d = static_cast<int64_t>(o.m) * o.n;
    for (auto layout : {desc.a, desc.b, desc.c, desc.d}) {
      CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
          layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
          &batch_count, sizeof(batch_count)));
    }
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        desc.a, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &stride_a, sizeof(stride_a)));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        desc.b, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &stride_b, sizeof(stride_b)));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        desc.c, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &stride_d, sizeof(stride_d)));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        desc.d, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &stride_d, sizeof(stride_d)));
  }
  CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&desc.preference));
  CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      desc.preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspace_bytes, sizeof(workspace_bytes)));
  return desc;
}

void destroy_descriptors(Descriptors& desc) {
  if (desc.preference) cublasLtMatmulPreferenceDestroy(desc.preference);
  if (desc.d) cublasLtMatrixLayoutDestroy(desc.d);
  if (desc.c) cublasLtMatrixLayoutDestroy(desc.c);
  if (desc.b) cublasLtMatrixLayoutDestroy(desc.b);
  if (desc.a) cublasLtMatrixLayoutDestroy(desc.a);
  if (desc.operation) cublasLtMatmulDescDestroy(desc.operation);
}

cublasStatus_t matmul(
    cublasLtHandle_t handle,
    const Descriptors& desc,
    const cublasLtMatmulAlgo_t& algo,
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* d,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  return cublasLtMatmul(
      handle,
      desc.operation,
      &alpha,
      a,
      desc.a,
      b,
      desc.b,
      &beta,
      d,
      desc.c,
      d,
      desc.d,
      &algo,
      workspace,
      workspace_bytes,
      stream);
}

float time_algo(
    cublasLtHandle_t handle,
    const Descriptors& desc,
    const cublasLtMatmulAlgo_t& algo,
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* d,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream,
    int warmup,
    int iterations) {
  for (int i = 0; i < warmup; ++i) {
    if (matmul(handle, desc, algo, a, b, d, workspace, workspace_bytes, stream) !=
        CUBLAS_STATUS_SUCCESS) {
      return -1.0f;
    }
  }
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    cudaGetLastError();
    return -1.0f;
  }
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < iterations; ++i) {
    if (matmul(handle, desc, algo, a, b, d, workspace, workspace_bytes, stream) !=
        CUBLAS_STATUS_SUCCESS) {
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      return -1.0f;
    }
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return elapsed / iterations;
}

int algo_attribute(
    const cublasLtMatmulAlgo_t& algo,
    cublasLtMatmulAlgoConfigAttributes_t attribute) {
  int value = -1;
  size_t written = 0;
  const cublasStatus_t status = cublasLtMatmulAlgoConfigGetAttribute(
      &algo, attribute, &value, sizeof(value), &written);
  return status == CUBLAS_STATUS_SUCCESS ? value : -1;
}

struct Result {
  int candidate = -1;
  float milliseconds = 0.0f;
  cublasLtMatmulHeuristicResult_t heuristic{};
};

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    CUDA_CHECK(cudaSetDevice(0));
    int low_priority = 0;
    int high_priority = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&low_priority, &high_priority));
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithPriority(
        &stream, cudaStreamNonBlocking, high_priority));
    cublasLtHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasLtCreate(&handle));

    const int64_t a_count =
        static_cast<int64_t>(options.batches) * options.m * options.k;
    const int64_t b_count =
        static_cast<int64_t>(options.batches) * options.n * options.k;
    const int64_t d_count =
        static_cast<int64_t>(options.batches) * options.m * options.n;
    __nv_bfloat16* a = nullptr;
    __nv_bfloat16* b = nullptr;
    __nv_bfloat16* d = nullptr;
    void* workspace = nullptr;
    const size_t workspace_bytes =
        static_cast<size_t>(options.workspace_mib) << 20;
    CUDA_CHECK(cudaMalloc(&a, a_count * sizeof(*a)));
    CUDA_CHECK(cudaMalloc(&b, b_count * sizeof(*b)));
    CUDA_CHECK(cudaMalloc(&d, d_count * sizeof(*d)));
    CUDA_CHECK(cudaMalloc(&workspace, workspace_bytes));
    fill_bf16<<<4096, 256, 0, stream>>>(a, a_count, 1);
    fill_bf16<<<4096, 256, 0, stream>>>(b, b_count, 2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    Descriptors desc = make_descriptors(options, workspace_bytes);
    std::vector<cublasLtMatmulHeuristicResult_t> candidates(options.candidates);
    int returned = 0;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
        handle,
        desc.operation,
        desc.a,
        desc.b,
        desc.c,
        desc.d,
        desc.preference,
        options.candidates,
        candidates.data(),
        &returned));
    if (returned == 0) {
      throw std::runtime_error("cuBLASLt returned no heuristic candidates");
    }

    std::vector<Result> results;
    for (int i = 0; i < returned; ++i) {
      if (candidates[i].state != CUBLAS_STATUS_SUCCESS ||
          candidates[i].workspaceSize > workspace_bytes) {
        continue;
      }
      const float milliseconds = time_algo(
          handle,
          desc,
          candidates[i].algo,
          a,
          b,
          d,
          workspace,
          candidates[i].workspaceSize,
          stream,
          options.tune_warmup,
          options.tune_iterations);
      if (milliseconds > 0.0f) {
        results.push_back({i, milliseconds, candidates[i]});
      }
    }
    if (results.empty()) {
      throw std::runtime_error("all cuBLASLt heuristic candidates failed");
    }
    std::sort(results.begin(), results.end(), [](const Result& x, const Result& y) {
      return x.milliseconds < y.milliseconds;
    });
    const double flops = 2.0 * options.m * options.n * options.k * options.batches;
    std::cout << "cuBLASLt BF16 autotune M=" << options.m << " N=" << options.n
              << " K=" << options.k << " L=" << options.batches
              << " workspace=" << options.workspace_mib << " MiB"
              << " candidates=" << returned << "\n";
    const int report = std::min<int>(10, results.size());
    for (int i = 0; i < report; ++i) {
      const auto& result = results[i];
      const auto& algo = result.heuristic.algo;
      std::cout << "candidate=" << result.candidate
                << " algo=" << algo_attribute(algo, CUBLASLT_ALGO_CONFIG_ID)
                << " tile=" << algo_attribute(algo, CUBLASLT_ALGO_CONFIG_TILE_ID)
                << " stages=" << algo_attribute(algo, CUBLASLT_ALGO_CONFIG_STAGES_ID)
                << " splitK=" << algo_attribute(algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM)
                << " swizzle=" << algo_attribute(algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING)
                << " workspace=" << result.heuristic.workspaceSize
                << " tune_ms=" << std::fixed << std::setprecision(4)
                << result.milliseconds
                << " TFLOPS=" << std::setprecision(1)
                << flops / result.milliseconds / 1.0e9 << "\n";
    }

    const auto& best = results.front().heuristic;
    const float milliseconds = time_algo(
        handle,
        desc,
        best.algo,
        a,
        b,
        d,
        workspace,
        best.workspaceSize,
        stream,
        options.warmup,
        options.iterations);
    if (milliseconds <= 0.0f) {
      throw std::runtime_error("selected cuBLASLt algorithm failed formal timing");
    }
    std::cout << "BEST mean=" << std::fixed << std::setprecision(4)
              << milliseconds << " ms TFLOPS/GPU=" << std::setprecision(1)
              << flops / milliseconds / 1.0e9 << "\n";

    destroy_descriptors(desc);
    CUBLAS_CHECK(cublasLtDestroy(handle));
    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(d));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "cublaslt_bench: " << error.what() << "\n";
    return 1;
  }
}
