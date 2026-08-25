// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>

namespace fuse {

#if defined(__CUDACC__)
#define FUSE_ROUTE_HOST_DEVICE __host__ __device__
#else
#define FUSE_ROUTE_HOST_DEVICE
#endif

enum class RouteKind : int32_t {
  kHeadToSequence = 0,
  kQkvGqaPack = 2,
};

enum class RouteDirection : int32_t {
  kForward = 0,
  kInverse = 1,
};

struct UlyssesRoute {
  int32_t world_size = 1;
  int32_t rank = 0;
  int32_t batch = 1;
  int32_t global_seq = 0;
  int32_t seq_local = 0;
  int32_t q_heads = 0;
  int32_t kv_heads = 0;
  int32_t local_heads = 0;
  int32_t head_dim = 0;
  int32_t channel_count = 1;
  // QKV_GQA_PACK may consume an offline-prepacked projection weight/output:
  // within Q and K, physical heads are [local_head][owner_rank]. V remains
  // in its standard contiguous segment. The final routed Q/K/V layout is
  // unchanged.
  bool qkv_peer_interleaved = false;
  // QKV_GQA_PACK normally routes Q, K, and V into the consumer's contiguous
  // post-A2A tensors. When enabled, only Q/K are routed and V remains in the
  // producer's local GEMM output for a later pull-based consumer.
  bool defer_v_a2a = false;
  // Standard causal CP assigns two discontiguous global sequence chunks to
  // each rank. Routes set this bit to map rank-local rows to those chunks
  // without a separate index-select adapter.
  bool causal_load_balanced = false;
  // In inverse head-to-sequence A2A, place source peer
  // (rank + k_shard) % world into consecutive GEMM K shards. The output
  // projection weight must use the same rank-local cyclic K prepack. This
  // makes the first shard local and balances every communication wave across
  // source GPUs without an online permutation.
  bool cyclic_peer_order = false;
  // Optional packed-varlen mapping for HEAD_TO_SEQUENCE input routes.
  // packed_source_row[local_row] gives the row in the sequence-major global
  // packed input that belongs to this rank-local packed row.  Bulk copies may
  // coalesce at most packed_row_granularity consecutive rows; callers must
  // choose a granularity that never crosses a sequence boundary.
  const int32_t* packed_source_row = nullptr;
  int32_t packed_row_granularity = 0;
  RouteKind kind = RouteKind::kHeadToSequence;
  RouteDirection direction = RouteDirection::kForward;
};

struct QkvGqaAddress {
  int32_t owner_rank = -1;
  int32_t local_feature = -1;
  int32_t segment = -1;  // 0=Q, 1=K, 2=V
  int32_t logical_head = -1;
  int32_t head_offset = -1;

  FUSE_ROUTE_HOST_DEVICE constexpr bool valid() const { return owner_rank >= 0; }
};

// Maps [Q_heads, K_heads, V_heads] packed features to the rank-local packed
// feature axis. Head ownership is contiguous. Uneven head splits are rejected.
FUSE_ROUTE_HOST_DEVICE constexpr QkvGqaAddress map_qkv_gqa_feature(
    const UlyssesRoute& route,
    int32_t global_feature) {
  if (route.world_size <= 0 || route.q_heads <= 0 || route.kv_heads <= 0 ||
      route.head_dim <= 0 || route.q_heads % route.world_size != 0 ||
      route.kv_heads % route.world_size != 0 || global_feature < 0 ||
      global_feature >= (route.q_heads + 2 * route.kv_heads) * route.head_dim) {
    return {};
  }
  const int32_t q_width = route.q_heads * route.head_dim;
  const int32_t kv_width = route.kv_heads * route.head_dim;
  const int32_t segment = global_feature < q_width
      ? 0
      : (global_feature < q_width + kv_width ? 1 : 2);
  const int32_t segment_feature = global_feature -
      (segment == 0 ? 0 : q_width + (segment == 2 ? kv_width : 0));
  const int32_t head = segment_feature / route.head_dim;
  const int32_t offset = segment_feature - head * route.head_dim;
  const int32_t local_heads =
      (segment == 0 ? route.q_heads : route.kv_heads) / route.world_size;
  const int32_t owner = head / local_heads;
  const int32_t local_head = head - owner * local_heads;
  const int32_t q_local_width = route.q_heads / route.world_size * route.head_dim;
  const int32_t kv_local_width = route.kv_heads / route.world_size * route.head_dim;
  const int32_t segment_base =
      segment == 0 ? 0 : q_local_width + (segment == 2 ? kv_local_width : 0);
  return {owner, segment_base + local_head * route.head_dim + offset,
          segment, head, offset};
}

FUSE_ROUTE_HOST_DEVICE constexpr int32_t qkv_gqa_global_feature(
    const UlyssesRoute& route,
    int32_t owner_rank,
    int32_t local_feature) {
  if (route.world_size <= 0 || owner_rank < 0 || owner_rank >= route.world_size ||
      route.q_heads % route.world_size != 0 ||
      route.kv_heads % route.world_size != 0 || route.head_dim <= 0) {
    return -1;
  }
  const int32_t q_local_width = route.q_heads / route.world_size * route.head_dim;
  const int32_t kv_local_width = route.kv_heads / route.world_size * route.head_dim;
  const int32_t local_width = q_local_width + 2 * kv_local_width;
  if (local_feature < 0 || local_feature >= local_width) {
    return -1;
  }
  const int32_t segment = local_feature < q_local_width
      ? 0
      : (local_feature < q_local_width + kv_local_width ? 1 : 2);
  const int32_t segment_local = local_feature -
      (segment == 0 ? 0 : q_local_width + (segment == 2 ? kv_local_width : 0));
  const int32_t heads_per_rank =
      (segment == 0 ? route.q_heads : route.kv_heads) / route.world_size;
  const int32_t global_head =
      owner_rank * heads_per_rank + segment_local / route.head_dim;
  const int32_t offset = segment_local % route.head_dim;
  const int32_t q_width = route.q_heads * route.head_dim;
  const int32_t kv_width = route.kv_heads * route.head_dim;
  const int32_t segment_base =
      segment == 0 ? 0 : q_width + (segment == 2 ? kv_width : 0);
  return segment_base + global_head * route.head_dim + offset;
}

#undef FUSE_ROUTE_HOST_DEVICE

}  // namespace fuse
