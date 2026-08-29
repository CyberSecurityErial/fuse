#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#define CUDA_CHECK(expr) check_cuda((expr), #expr, __FILE__, __LINE__)
#define CUBLAS_CHECK(expr) check_cublas((expr), #expr, __FILE__, __LINE__)
#define MPI_CHECK(expr) check_mpi((expr), #expr, __FILE__, __LINE__)

void check_cuda(
    cudaError_t status,
    const char* expression,
    const char* file,
    int line) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(file) + ":" + std::to_string(line) + " " + expression +
        ": " + cudaGetErrorString(status));
  }
}

void check_cublas(
    cublasStatus_t status,
    const char* expression,
    const char* file,
    int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
        std::string(file) + ":" + std::to_string(line) + " " + expression +
        ": cuBLAS status " + std::to_string(static_cast<int>(status)));
  }
}

void check_mpi(
    int status,
    const char* expression,
    const char* file,
    int line) {
  if (status != MPI_SUCCESS) {
    char message[MPI_MAX_ERROR_STRING]{};
    int length = 0;
    MPI_Error_string(status, message, &length);
    throw std::runtime_error(
        std::string(file) + ":" + std::to_string(line) + " " + expression +
        ": " + std::string(message, static_cast<size_t>(length)));
  }
}

enum class OperatorKind { kQkv, kOproj };

struct Options {
  OperatorKind operator_kind = OperatorKind::kQkv;
  int m = 128;
  int hidden = 4096;
  int q_heads = 16;
  int kv_heads = 8;
  int head_dim = 128;
  int warmup = 10;
  int iterations = 50;
  bool check = false;
  bool help = false;
  std::string json_out;
};

int positive(const std::string& value, const char* name) {
  const long long parsed = std::stoll(value);
  if (parsed <= 0 || parsed > 0x7fffffffLL) {
    throw std::runtime_error(std::string(name) + " must be a positive int");
  }
  return static_cast<int>(parsed);
}

Options parse_options(int argc, char** argv) {
  Options result;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto take = [&](const char* name) {
      if (++index == argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      return std::string(argv[index]);
    };
    if (argument == "--operator") {
      const std::string value = take("--operator");
      if (value == "qkv") {
        result.operator_kind = OperatorKind::kQkv;
      } else if (value == "oproj") {
        result.operator_kind = OperatorKind::kOproj;
      } else {
        throw std::runtime_error("--operator must be qkv or oproj");
      }
    } else if (argument == "--m") {
      result.m = positive(take("--m"), "--m");
    } else if (argument == "--hidden") {
      result.hidden = positive(take("--hidden"), "--hidden");
    } else if (argument == "--q-heads") {
      result.q_heads = positive(take("--q-heads"), "--q-heads");
    } else if (argument == "--kv-heads") {
      result.kv_heads = positive(take("--kv-heads"), "--kv-heads");
    } else if (argument == "--head-dim") {
      result.head_dim = positive(take("--head-dim"), "--head-dim");
    } else if (argument == "--warmup") {
      result.warmup = positive(take("--warmup"), "--warmup");
    } else if (argument == "--iterations") {
      result.iterations = positive(take("--iterations"), "--iterations");
    } else if (argument == "--check") {
      result.check = true;
    } else if (argument == "--no-check") {
      result.check = false;
    } else if (argument == "--json-out") {
      result.json_out = take("--json-out");
    } else if (argument == "--help" || argument == "-h") {
      result.help = true;
    } else {
      throw std::runtime_error("unknown argument: " + argument);
    }
  }
  return result;
}

void print_usage(const char* program) {
  std::cout
      << "Usage: " << program << " [options]\n"
      << "  --operator qkv|oproj\n"
      << "  --m M --hidden H --q-heads HQ --kv-heads HKV --head-dim D\n"
      << "  --warmup N --iterations N --check|--no-check --json-out PATH\n";
}

__global__ void fill_bf16(
    __nv_bfloat16* values,
    int64_t elements,
    int seed) {
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float value = static_cast<float>(
        (index * 17 + static_cast<int64_t>(seed) * 13) % 23 - 11) / 32.0f;
    values[index] = __float2bfloat16(value);
  }
}

struct Context {
  MPI_Comm local_comm = MPI_COMM_NULL;
  int rank = 0;
  int world = 1;
  int local_rank = 0;
  int local_world = 1;
};

Context initialize_context() {
  Context result;
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &result.rank));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &result.world));
  MPI_CHECK(MPI_Comm_split_type(
      MPI_COMM_WORLD,
      MPI_COMM_TYPE_SHARED,
      result.rank,
      MPI_INFO_NULL,
      &result.local_comm));
  MPI_CHECK(MPI_Comm_rank(result.local_comm, &result.local_rank));
  MPI_CHECK(MPI_Comm_size(result.local_comm, &result.local_world));
  if (result.world != result.local_world) {
    throw std::runtime_error("pure cuBLAS benchmark requires one node");
  }
  int devices = 0;
  CUDA_CHECK(cudaGetDeviceCount(&devices));
  if (devices < result.local_world) {
    throw std::runtime_error("not enough visible CUDA devices");
  }
  CUDA_CHECK(cudaSetDevice(result.local_rank));
  return result;
}

struct Runtime {
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cublasHandle_t handle = nullptr;
};

Runtime make_runtime() {
  Runtime result;
  CUDA_CHECK(cudaStreamCreateWithFlags(
      &result.stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaEventCreate(&result.start));
  CUDA_CHECK(cudaEventCreate(&result.stop));
  CUBLAS_CHECK(cublasCreate(&result.handle));
  CUBLAS_CHECK(cublasSetStream(result.handle, result.stream));
  CUBLAS_CHECK(cublasSetMathMode(result.handle, CUBLAS_TENSOR_OP_MATH));
  return result;
}

void destroy_runtime(Runtime& runtime) {
  if (runtime.handle) CUBLAS_CHECK(cublasDestroy(runtime.handle));
  if (runtime.stop) CUDA_CHECK(cudaEventDestroy(runtime.stop));
  if (runtime.start) CUDA_CHECK(cudaEventDestroy(runtime.start));
  if (runtime.stream) CUDA_CHECK(cudaStreamDestroy(runtime.stream));
}

// Row-major D[M,N] = A[M,K] * B[K,N]. cuBLAS sees the transposed
// column-major equation D^T[N,M] = B^T[N,K] * A^T[K,M], without copying.
void launch_row_major_nn(
    cublasHandle_t handle,
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* d,
    int m,
    int n,
    int k) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  CUBLAS_CHECK(cublasGemmEx(
      handle,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      n,
      m,
      k,
      &alpha,
      b,
      CUDA_R_16BF,
      n,
      a,
      CUDA_R_16BF,
      k,
      &beta,
      d,
      CUDA_R_16BF,
      n,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// Row-major D[R,C] = G^T[R,M] * X[M,C]. X is already the column-major
// view [C,M]; G is transposed in the descriptor, so neither operand moves.
void launch_transpose_left(
    cublasHandle_t handle,
    const __nv_bfloat16* gradient,
    const __nv_bfloat16* saved_input,
    __nv_bfloat16* grad_weight,
    int rows,
    int columns,
    int tokens,
    float beta) {
  const float alpha = 1.0f;
  CUBLAS_CHECK(cublasGemmEx(
      handle,
      CUBLAS_OP_N,
      CUBLAS_OP_T,
      columns,
      rows,
      tokens,
      &alpha,
      saved_input,
      CUDA_R_16BF,
      columns,
      gradient,
      CUDA_R_16BF,
      rows,
      &beta,
      grad_weight,
      CUDA_R_16BF,
      columns,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

template <class Launch>
std::vector<float> time_max_rank(
    const Options& options,
    const Context& context,
    const Runtime& runtime,
    Launch launch) {
  for (int step = 0; step < options.warmup; ++step) {
    MPI_CHECK(MPI_Barrier(context.local_comm));
    launch();
    CUDA_CHECK(cudaStreamSynchronize(runtime.stream));
  }
  std::vector<float> samples;
  samples.reserve(options.iterations);
  for (int step = 0; step < options.iterations; ++step) {
    MPI_CHECK(MPI_Barrier(context.local_comm));
    CUDA_CHECK(cudaEventRecord(runtime.start, runtime.stream));
    launch();
    CUDA_CHECK(cudaEventRecord(runtime.stop, runtime.stream));
    CUDA_CHECK(cudaEventSynchronize(runtime.stop));
    float local_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(
        &local_ms, runtime.start, runtime.stop));
    float critical_ms = 0.0f;
    MPI_CHECK(MPI_Allreduce(
        &local_ms,
        &critical_ms,
        1,
        MPI_FLOAT,
        MPI_MAX,
        context.local_comm));
    samples.push_back(critical_ms);
  }
  return samples;
}

float percentile(std::vector<float> values, double quantile) {
  std::sort(values.begin(), values.end());
  const double position = (values.size() - 1) * quantile;
  const size_t lower = static_cast<size_t>(std::floor(position));
  const size_t upper = static_cast<size_t>(std::ceil(position));
  const double weight = position - lower;
  return static_cast<float>(
      values[lower] * (1.0 - weight) + values[upper] * weight);
}

struct Summary {
  double mean = 0.0;
  float p50 = 0.0f;
  float p95 = 0.0f;
  float minimum = 0.0f;
  float maximum = 0.0f;
};

Summary summarize(const std::vector<float>& samples) {
  if (samples.empty()) {
    throw std::runtime_error("cannot summarize empty samples");
  }
  return {
      std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size(),
      percentile(samples, 0.50),
      percentile(samples, 0.95),
      *std::min_element(samples.begin(), samples.end()),
      *std::max_element(samples.begin(), samples.end()),
  };
}

float bf16_to_float(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

std::vector<uint16_t> download_bf16(
    const __nv_bfloat16* pointer,
    int64_t elements) {
  std::vector<uint16_t> result(static_cast<size_t>(elements));
  CUDA_CHECK(cudaMemcpy(
      result.data(),
      pointer,
      static_cast<size_t>(elements) * sizeof(uint16_t),
      cudaMemcpyDeviceToHost));
  return result;
}

uint64_t validate_b(
    const std::vector<uint16_t>& gradient,
    const std::vector<uint16_t>& weight,
    const std::vector<uint16_t>& output,
    int m,
    int rows,
    int columns) {
  uint64_t mismatches = 0;
  for (int row = 0; row < m; ++row) {
    for (int column = 0; column < columns; ++column) {
      float expected = 0.0f;
      for (int k = 0; k < rows; ++k) {
        expected += bf16_to_float(
            gradient[static_cast<size_t>(row) * rows + k]) *
            bf16_to_float(weight[static_cast<size_t>(k) * columns + column]);
      }
      const float actual = bf16_to_float(
          output[static_cast<size_t>(row) * columns + column]);
      mismatches += std::abs(actual - expected) > 0.125f;
    }
  }
  return mismatches;
}

uint64_t validate_w(
    const std::vector<uint16_t>& gradient,
    const std::vector<uint16_t>& saved_input,
    const std::vector<uint16_t>& output,
    int tokens,
    int rows,
    int columns,
    float expected_scale = 1.0f) {
  uint64_t mismatches = 0;
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < columns; ++column) {
      float expected = 0.0f;
      for (int token = 0; token < tokens; ++token) {
        expected += bf16_to_float(
            gradient[static_cast<size_t>(token) * rows + row]) *
            bf16_to_float(
                saved_input[static_cast<size_t>(token) * columns + column]);
      }
      expected *= expected_scale;
      const float actual = bf16_to_float(
          output[static_cast<size_t>(row) * columns + column]);
      mismatches += std::abs(actual - expected) > 0.125f;
    }
  }
  return mismatches;
}

void write_summary(
    std::ostream& output,
    const char* name,
    const Summary& summary,
    const std::vector<float>& samples,
    bool comma) {
  output << "  \"" << name << "\": {\n"
         << "    \"mean_ms\": " << summary.mean << ",\n"
         << "    \"p50_ms\": " << summary.p50 << ",\n"
         << "    \"p95_ms\": " << summary.p95 << ",\n"
         << "    \"min_ms\": " << summary.minimum << ",\n"
         << "    \"max_ms\": " << summary.maximum << ",\n"
         << "    \"samples_ms\": [";
  for (size_t index = 0; index < samples.size(); ++index) {
    if (index) output << ", ";
    output << samples[index];
  }
  output << "]\n  }" << (comma ? "," : "") << "\n";
}

int run(const Options& options, const Context& context) {
  if (options.m <= 0 || options.hidden <= 0 || options.q_heads <= 0 ||
      options.kv_heads <= 0 || options.head_dim <= 0 ||
      options.q_heads % context.world != 0 ||
      options.kv_heads % context.world != 0) {
    throw std::runtime_error("invalid shape or head geometry");
  }
  const int projection_width = options.operator_kind == OperatorKind::kQkv
      ? (options.q_heads + 2 * options.kv_heads) * options.head_dim
      : options.q_heads * options.head_dim;
  const int weight_rows = options.operator_kind == OperatorKind::kQkv
      ? projection_width
      : options.hidden;
  const int weight_columns = options.operator_kind == OperatorKind::kQkv
      ? options.hidden
      : projection_width;
  const int64_t gradient_elements =
      static_cast<int64_t>(options.m) * weight_rows;
  const int64_t matrix_elements =
      static_cast<int64_t>(weight_rows) * weight_columns;
  const int64_t data_elements =
      static_cast<int64_t>(options.m) * weight_columns;

  Runtime runtime = make_runtime();
  __nv_bfloat16* gradient = nullptr;
  __nv_bfloat16* weight_or_grad = nullptr;
  __nv_bfloat16* saved_or_data = nullptr;
  CUDA_CHECK(cudaMalloc(&gradient, gradient_elements * sizeof(*gradient)));
  CUDA_CHECK(cudaMalloc(
      &weight_or_grad, matrix_elements * sizeof(*weight_or_grad)));
  CUDA_CHECK(cudaMalloc(&saved_or_data, data_elements * sizeof(*saved_or_data)));
  fill_bf16<<<4096, 256, 0, runtime.stream>>>(
      gradient, gradient_elements, 101 + context.rank);
  fill_bf16<<<4096, 256, 0, runtime.stream>>>(
      weight_or_grad, matrix_elements, 211 + context.rank);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemsetAsync(
      saved_or_data, 0, data_elements * sizeof(*saved_or_data),
      runtime.stream));
  CUDA_CHECK(cudaStreamSynchronize(runtime.stream));

  const auto b_samples = time_max_rank(
      options,
      context,
      runtime,
      [&] {
        launch_row_major_nn(
            runtime.handle,
            gradient,
            weight_or_grad,
            saved_or_data,
            options.m,
            weight_columns,
            weight_rows);
      });
  const auto b_summary = summarize(b_samples);
  std::vector<uint16_t> host_gradient;
  std::vector<uint16_t> host_saved;
  uint64_t local_mismatches = 0;
  if (options.check) {
    const int64_t work =
        static_cast<int64_t>(options.m) * weight_rows * weight_columns;
    if (work > 100000000) {
      throw std::runtime_error("--check is limited to a small GEMM");
    }
    host_gradient = download_bf16(gradient, gradient_elements);
    const auto host_weight = download_bf16(weight_or_grad, matrix_elements);
    host_saved = download_bf16(saved_or_data, data_elements);
    local_mismatches += validate_b(
        host_gradient,
        host_weight,
        host_saved,
        options.m,
        weight_rows,
        weight_columns);
  }
  const auto w_beta0_samples = time_max_rank(
      options,
      context,
      runtime,
      [&] {
        launch_transpose_left(
            runtime.handle,
            gradient,
            saved_or_data,
            weight_or_grad,
            weight_rows,
            weight_columns,
            options.m,
            0.0f);
      });
  const auto w_beta0_summary = summarize(w_beta0_samples);
  if (options.check) {
    const auto host_grad_weight =
        download_bf16(weight_or_grad, matrix_elements);
    local_mismatches += validate_w(
        host_gradient,
        host_saved,
        host_grad_weight,
        options.m,
        weight_rows,
        weight_columns,
        1.0f);
  }
  const auto w_beta1_samples = time_max_rank(
      options,
      context,
      runtime,
      [&] {
        launch_transpose_left(
            runtime.handle,
            gradient,
            saved_or_data,
            weight_or_grad,
            weight_rows,
            weight_columns,
            options.m,
            1.0f);
      });
  const auto w_beta1_summary = summarize(w_beta1_samples);
  if (options.check) {
    const auto host_accumulated =
        download_bf16(weight_or_grad, matrix_elements);
    local_mismatches += validate_w(
        host_gradient,
        host_saved,
        host_accumulated,
        options.m,
        weight_rows,
        weight_columns,
        static_cast<float>(1 + options.warmup + options.iterations));
  }
  uint64_t global_mismatches = 0;
  MPI_CHECK(MPI_Allreduce(
      &local_mismatches,
      &global_mismatches,
      1,
      MPI_UINT64_T,
      MPI_SUM,
      context.local_comm));
  if (global_mismatches != 0) {
    throw std::runtime_error(
        "cuBLAS layout check mismatches=" +
        std::to_string(global_mismatches));
  }
  std::vector<float> total_beta0_samples(options.iterations);
  std::vector<float> total_beta1_samples(options.iterations);
  for (int index = 0; index < options.iterations; ++index) {
    total_beta0_samples[index] = b_samples[index] + w_beta0_samples[index];
    total_beta1_samples[index] = b_samples[index] + w_beta1_samples[index];
  }
  const auto total_beta0_summary = summarize(total_beta0_samples);
  const auto total_beta1_summary = summarize(total_beta1_samples);

  if (context.rank == 0) {
    const double total_flops =
        4.0 * options.m * weight_rows * weight_columns;
    std::cout << "classic cuBLAS backward "
              << (options.operator_kind == OperatorKind::kQkv ? "QKV" : "OProj")
              << " B=" << options.m << "x" << weight_columns << "x"
              << weight_rows << " W=" << weight_rows << "x"
              << weight_columns << "x" << options.m
              << " beta0/beta1 p50=" << std::fixed << std::setprecision(4)
              << total_beta0_summary.p50 << "/" << total_beta1_summary.p50
              << " ms TFLOPS/GPU="
              << std::setprecision(1)
              << total_flops / total_beta0_summary.p50 / 1.0e9 << "/"
              << total_flops / total_beta1_summary.p50 / 1.0e9 << "\n";
    if (!options.json_out.empty()) {
      const std::string temporary = options.json_out + ".tmp";
      std::ofstream output(temporary);
      if (!output) {
        throw std::runtime_error("cannot open JSON output: " + temporary);
      }
      output << std::setprecision(10)
             << "{\n"
             << "  \"mode\": \"classic_cublas_backward_mpi\",\n"
             << "  \"schema\": \"v10_backward_cublas_v2\",\n"
             << "  \"operator\": \""
             << (options.operator_kind == OperatorKind::kQkv ? "qkv" : "oproj")
             << "\",\n"
             << "  \"world_size\": " << context.world << ",\n"
             << "  \"warmup\": " << options.warmup << ",\n"
             << "  \"iterations\": " << options.iterations << ",\n"
             << "  \"correctness\": \""
             << (options.check ? "cpu_reference" : "not_run") << "\",\n"
             << "  \"timing\": \"per-sample max-rank CUDA event\",\n"
             << "  \"shape\": {\"local_tokens\": " << options.m
             << ", \"hidden\": " << options.hidden
             << ", \"projection_width\": " << projection_width << "},\n"
             << "  \"b_mnk\": [" << options.m << ", " << weight_columns
             << ", " << weight_rows << "],\n"
             << "  \"w_mnk\": [" << weight_rows << ", "
             << weight_columns << ", " << options.m << "],\n";
      write_summary(output, "b_gemm", b_summary, b_samples, true);
      write_summary(
          output, "w_gemm_beta0", w_beta0_summary, w_beta0_samples, true);
      write_summary(
          output, "w_gemm_beta1", w_beta1_summary, w_beta1_samples, true);
      write_summary(
          output,
          "total_beta0",
          total_beta0_summary,
          total_beta0_samples,
          true);
      write_summary(
          output,
          "total_beta1",
          total_beta1_summary,
          total_beta1_samples,
          false);
      output << "}\n";
      output.close();
      if (std::rename(temporary.c_str(), options.json_out.c_str()) != 0) {
        throw std::runtime_error("cannot publish JSON output");
      }
    }
  }

  CUDA_CHECK(cudaFree(saved_or_data));
  CUDA_CHECK(cudaFree(weight_or_grad));
  CUDA_CHECK(cudaFree(gradient));
  destroy_runtime(runtime);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  int initialized = 0;
  int finalized = 0;
  int rank = -1;
  try {
    int provided = MPI_THREAD_SINGLE;
    MPI_CHECK(MPI_Init_thread(
        &argc, &argv, MPI_THREAD_FUNNELED, &provided));
    initialized = 1;
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    if (provided < MPI_THREAD_FUNNELED) {
      throw std::runtime_error("MPI does not provide MPI_THREAD_FUNNELED");
    }
    const Options options = parse_options(argc, argv);
    if (options.help) {
      if (rank == 0) print_usage(argv[0]);
      MPI_CHECK(MPI_Finalize());
      return 0;
    }
    Context context = initialize_context();
    const int result = run(options, context);
    MPI_CHECK(MPI_Comm_free(&context.local_comm));
    MPI_CHECK(MPI_Finalize());
    return result;
  } catch (const std::exception& error) {
    std::cerr << "backward_cublas_mpi_bench"
              << (rank >= 0 ? " rank=" + std::to_string(rank) : "")
              << ": " << error.what() << "\n";
    if (initialized) {
      MPI_Finalized(&finalized);
      if (!finalized) MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return 1;
  }
}
