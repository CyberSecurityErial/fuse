#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

extern "C" {
void* fuse_cublaslt_bf16_create(int, int64_t, int64_t, int64_t,
    const void*, const void*, void*, void*, int, int, uint64_t);
int fuse_cublaslt_bf16_run_beta(void*, const void*, const void*, void*, void*, float);
void fuse_cublaslt_bf16_destroy(void*);
const char* fuse_cublaslt_last_error();
}

#define CUDA_CHECK(call) do { auto status = (call); if (status != cudaSuccess) \
  throw std::runtime_error(cudaGetErrorString(status)); } while (0)

__global__ void fill(__nv_bfloat16* data, int count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) data[i] = __float2bfloat16(1.0f);
}

int main() {
  try {
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    if (props.major != 10 || props.minor != 3)
      throw std::runtime_error("This smoke requires CUDA Runtime compute capability 10.3");
    constexpr int m = 128, n = 256, k = 128;
    __nv_bfloat16 *a, *b, *d;
    CUDA_CHECK(cudaMalloc(&a, m * k * sizeof(*a)));
    CUDA_CHECK(cudaMalloc(&b, n * k * sizeof(*b)));
    CUDA_CHECK(cudaMalloc(&d, m * n * sizeof(*d)));
    fill<<<(m*k+255)/256,256>>>(a, m*k);
    fill<<<(n*k+255)/256,256>>>(b, n*k);
    CUDA_CHECK(cudaGetLastError());
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaDeviceSynchronize());
    void* plan = fuse_cublaslt_bf16_create(0, m, n, k, a, b, d, stream, 2, 5, 64ull<<20);
    if (!plan) throw std::runtime_error(fuse_cublaslt_last_error());
    auto run = [&](float beta) {
      if (!fuse_cublaslt_bf16_run_beta(plan, a, b, d, stream, beta))
        throw std::runtime_error(fuse_cublaslt_last_error());
    };
    auto verify = [&](float expected) {
      CUDA_CHECK(cudaStreamSynchronize(stream));
      std::vector<__nv_bfloat16> host(m*n);
      CUDA_CHECK(cudaMemcpy(host.data(), d, host.size()*sizeof(*d), cudaMemcpyDeviceToHost));
      for (auto value : host)
        if (__bfloat162float(value) != expected) throw std::runtime_error("BF16 result mismatch");
    };
    run(0); verify(k);
    run(1); verify(2*k);
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    run(0); run(1);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphUpload(executable, stream));
    for (int i = 0; i < 3; ++i) {
      CUDA_CHECK(cudaGraphLaunch(executable, stream));
      verify(2*k);
    }
    CUDA_CHECK(cudaGraphExecDestroy(executable));
    CUDA_CHECK(cudaGraphDestroy(graph));
    fuse_cublaslt_bf16_destroy(plan);
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d)); CUDA_CHECK(cudaFree(b)); CUDA_CHECK(cudaFree(a));
    std::printf("PASS SM103a cuBLASLt: tuned plan, beta=0/1, three Graph replays, %d values each\n", m*n);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
