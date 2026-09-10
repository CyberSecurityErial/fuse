// Included inside the shared backward harness namespace. SM90 builds do not
// include this file. References use bounded row slabs, never another full-size
// output or FP32 copy. Both payloads validate every GEMM and routed element.
struct BackwardRouteReference {
  const Bf16* peer[8]{};
  int m, width, q_heads, kv_heads, head_dim, rank, world, batch, seed_offset;
  bool qkv, causal;
};

__device__ Bf16 backward_random_value(int64_t index, int seed) {
  uint64_t x = static_cast<uint64_t>(index) + (static_cast<uint64_t>(seed) << 32);
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  x ^= x >> 31;
  return Bf16((static_cast<int>(x & 65535) - 32768) / 131072.0f);
}

__global__ void check_backward_route(
    BackwardRouteReference p, const Bf16* output, unsigned long long* errors) {
  unsigned long long bad = 0;
  const int local_seq = p.m / p.batch;
  const int global_seq = local_seq * p.world;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < int64_t(p.m) * p.width; i += int64_t(gridDim.x) * blockDim.x) {
    Bf16 expected;
    if (p.qkv) {
      const int row = i / p.width, feature = i % p.width;
      const int batch = row / local_seq, local_row = row % local_seq;
      int global_row = p.rank * local_seq + local_row;
      if (p.causal) {
        const int half = local_seq / 2;
        global_row = (local_row < half ? p.rank : 2*p.world-p.rank-1)*half + local_row%half;
      }
      global_row += batch * global_seq;
      const int head = feature / p.head_dim;
      const int segment = head < p.q_heads ? 0 : (head < p.q_heads+p.kv_heads ? 1 : 2);
      const int heads = segment == 0 ? p.q_heads : p.kv_heads;
      const int segment_head = head - (segment == 0 ? 0 : p.q_heads+(segment-1)*p.kv_heads);
      const int heads_per_rank = heads / p.world;
      const int source = segment_head / heads_per_rank;
      const int64_t source_index = int64_t(global_row)*heads_per_rank*p.head_dim +
          (segment_head%heads_per_rank)*p.head_dim + feature%p.head_dim;
      expected = backward_random_value(source_index, source+101+100*segment+p.seed_offset);
    } else {
      // Output is [batch, global sequence, this rank's contiguous head shard].
      const int shard = p.width/p.world;
      const int row = i/shard, column = i%shard;
      const int batch = row/global_seq, seq = row%global_seq;
      int source = seq/local_seq, local_row = seq%local_seq;
      if (p.causal) {
        const int half = local_seq/2, chunk = seq/half;
        source = chunk < p.world ? chunk : 2*p.world-chunk-1;
        local_row = (chunk < p.world ? 0 : half) + seq%half;
      }
      expected = p.peer[source][int64_t(batch*local_seq+local_row)*p.width+p.rank*shard+column];
    }
    bad += reinterpret_cast<const uint16_t*>(output)[i] !=
        *reinterpret_cast<const uint16_t*>(&expected);
  }
  if (bad) atomicAdd(errors, bad);
}

__global__ void check_backward_gemm(
    const Bf16* actual, const Bf16* expected, int rows, int cols,
    int stride, unsigned long long* errors) {
  unsigned long long bad = 0;
  for (int64_t i = int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
       i < int64_t(rows)*cols; i += int64_t(gridDim.x)*blockDim.x) {
    float a = float(actual[(i/cols)*stride+i%cols]), b = float(expected[i]);
    // BF16 rounding plus FP32 reduction-order tolerance, fixed for all shapes.
    bad += !isfinite(a) || !isfinite(b) || fabsf(a-b) > 0.02f + 0.02f*fabsf(b);
  }
  if (bad) atomicAdd(errors, bad);
}

void refill_backward(const Options& o, const RankContext& c,
                     const Runtime& r, Buffers& b, int offset) {
  if (o.operator_kind == OperatorKind::kQkv) {
    fill(b.grad_q, int64_t(o.m)*o.q_heads*o.head_dim, c.rank+101+offset,r.stream);
    fill(b.grad_k, int64_t(o.m)*o.kv_heads*o.head_dim,c.rank+201+offset,r.stream);
    fill(b.grad_v, int64_t(o.m)*o.kv_heads*o.head_dim,c.rank+301+offset,r.stream);
  } else fill(b.grad_output,int64_t(o.m)*o.hidden,c.rank+401+offset,r.stream);
  fill(b.weight,b.grad_weight_elements,c.rank+501+offset,r.stream);
  const int width = (o.operator_kind == OperatorKind::kQkv ? o.hidden : o.q_heads*o.head_dim);
  fill(b.saved_input,int64_t(o.m)*width,c.rank+601+offset,r.stream);
  CUDA_CHECK(cudaStreamSynchronize(r.stream));
}

void validate_full_backward(const Options& o, const RankContext& c,
                            const Runtime& r, const Buffers& b, int offset, bool gemm_only = false) {
  if (o.weight_beta != 0) throw std::runtime_error("SM103 formal baseline requires beta=0; beta=1 is checked by backward_smoke");
  const bool qkv = o.operator_kind == OperatorKind::kQkv;
  const int width = (o.q_heads+(qkv ? 2*o.kv_heads : 0))*o.head_dim;
  unsigned long long* errors = allocate<unsigned long long>(1);
  CUDA_CHECK(cudaMemsetAsync(errors,0,sizeof(*errors),r.stream));
  BackwardRouteReference route{};
  route.m=o.m; route.width=width; route.q_heads=o.q_heads; route.kv_heads=o.kv_heads;
  route.head_dim=o.head_dim; route.rank=c.rank; route.world=c.world;
  route.batch=o.batch; route.seed_offset=offset; route.qkv=qkv; route.causal=o.causal_load_balanced;
  if (!gemm_only) {
  if (!qkv) {
    cudaIpcMemHandle_t local{};
    std::vector<cudaIpcMemHandle_t> handles(c.world);
    CUDA_CHECK(cudaIpcGetMemHandle(&local,b.local_intermediate));
    MPI_CHECK(MPI_Allgather(&local,sizeof(local),MPI_BYTE,handles.data(),sizeof(local),MPI_BYTE,c.local_comm));
    for(int peer=0;peer<c.world;++peer) {
      if(peer==c.rank) route.peer[peer]=b.local_intermediate;
      else CUDA_CHECK(cudaIpcOpenMemHandle((void**)&route.peer[peer],handles[peer],cudaIpcMemLazyEnablePeerAccess));
    }
  }
  check_backward_route<<<4096,256,0,r.stream>>>(route,b.owned_output,errors);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(r.stream));
  MPI_CHECK(MPI_Barrier(c.local_comm));
  if (!qkv) for(int peer=0;peer<c.world;++peer)
    if(peer!=c.rank) CUDA_CHECK(cudaIpcCloseMemHandle((void*)route.peer[peer]));
  }

  cublasHandle_t handle{};
  auto blas = [](cublasStatus_t status) { if(status!=CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("backward cuBLAS reference failed: "+std::to_string(int(status))); };
  blas(cublasCreate(&handle)); blas(cublasSetStream(handle,r.stream));
  const int n=qkv ? o.hidden : width, k=qkv ? width : o.hidden;
  const int slab=128;
  Bf16* reference=allocate<Bf16>(int64_t(slab)*std::max(n,width));
  float alpha=1.f,beta=0.f;
  const Bf16* input=qkv ? b.local_intermediate : b.grad_output;
  // NN dX: row-major [M,K] [K,N]. cuBLAS sees C^T=B^T A^T.
  for(int row=0;row<o.m;row+=slab) {
    int count=std::min(slab,o.m-row);
    blas(cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,count,k,&alpha,
      b.weight,CUDA_R_16BF,n,input+int64_t(row)*k,CUDA_R_16BF,k,&beta,
      reference,CUDA_R_16BF,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    check_backward_gemm<<<512,256,0,r.stream>>>(b.grad_input+int64_t(row)*n,reference,count,n,n,errors);
  }
  // TN dW: [K,M] [M,N], with the first operand a zero-copy transpose.
  const int wm=qkv ? width : o.hidden, wn=qkv ? o.hidden : width;
  for(int row=0;!gemm_only && row<wm;row+=slab) {
    int count=std::min(slab,wm-row);
    blas(cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_T,wn,count,o.m,&alpha,
      b.saved_input,CUDA_R_16BF,wn,input+row,CUDA_R_16BF,wm,&beta,
      reference,CUDA_R_16BF,wn,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    check_backward_gemm<<<512,256,0,r.stream>>>(b.grad_weight+int64_t(row)*wn,reference,count,wn,wn,errors);
  }
  CUDA_CHECK(cudaGetLastError());
  unsigned long long local_errors=0,total_errors=0;
  CUDA_CHECK(cudaMemcpyAsync(&local_errors,errors,sizeof(local_errors),cudaMemcpyDeviceToHost,r.stream));
  CUDA_CHECK(cudaStreamSynchronize(r.stream));
  MPI_CHECK(MPI_Allreduce(&local_errors,&total_errors,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,c.local_comm));
  CUDA_CHECK(cudaFree(reference)); CUDA_CHECK(cudaFree(errors)); blas(cublasDestroy(handle));
  if(total_errors) throw std::runtime_error("full backward numerical/route mismatch: "+std::to_string(total_errors));
  if(c.rank==0 && !gemm_only) std::cout << "backward_validation payload=" << offset << " full_gemm_and_route=PASS\n" << std::flush;
}
