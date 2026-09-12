#!/usr/bin/env python3
"""CPU checks for SM103 communication routing, task coverage and harness options.

The C++ probe compiles the real public fuse/layout/ulysses.h with a host C++
compiler. Harness probes compile the actual host-only options/candidate code,
using a two-byte storage alias solely for its BF16 allocation-size checks.
The real main's scheduling is also exercised with explicit CPU stand-ins for
CUDA work; it does not test the stand-ins as GPU implementations.
Validation probes include the real private header and execute its BF16 bit
conversion, element-address oracle and statistics observers/merges on the CPU.
Input probes check real uniform scaling, statistics and OProj gather addresses;
they do not execute or emulate the Philox RNG or its CUDA generation kernel.
Launch probes execute the real host thread team, including publication and
failure handling; they do not submit CUDA work or test GPU launch concurrency.
Host-stage probes link the actual private TLS definition across translation
units and execute the real rank-enqueue/diagnostic collectors with CUDA stand-ins.
They test timer ownership/coverage, not real driver costs or descriptor encoding.
Tile-policy probes execute the real selectors/bindings and peer-K validator
with opaque GEMM type stand-ins; they do not instantiate CUTLASS collectives.
Timeline probes execute the real first-observation recorder and profile flow
with scalar atomics/timers and CUDA stand-ins, not GPU ordering or timings.
The remaining checks model the communication task/address formulas in Python.
These tests do not compile a CUDA kernel or validate TMA, memory ordering,
cooperative occupancy, cross-GPU completion, or CUDA Graph execution.

Run: python3 scripts/test_sm103_fusion_routes.py
"""

from collections import Counter
from itertools import product
import math
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CPP_PROBE = r"""
#include "fuse/layout/ulysses.h"
#include <iostream>

int main() {
  int world, q_heads, kv_heads, head_dim, feature;
  while (std::cin >> world >> q_heads >> kv_heads >> head_dim >> feature) {
    fuse::UlyssesRoute route;
    route.world_size = world;
    route.q_heads = q_heads;
    route.kv_heads = kv_heads;
    route.head_dim = head_dim;
    const auto address = fuse::map_qkv_gqa_feature(route, feature);
    const int inverse = address.valid()
        ? fuse::qkv_gqa_global_feature(route, address.owner_rank, address.local_feature)
        : -1;
    std::cout << address.owner_rank << ' ' << address.local_feature << ' '
              << address.segment << ' ' << address.logical_head << ' '
              << address.head_offset << ' ' << inverse << '\n';
  }
}
"""


def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def qkv_sequence_row(rank, world, seq_local, row):
    """SM90/SM103 packed-QKV contract: concatenate source ranks along sequence."""
    batch, local = divmod(row, seq_local)
    return batch * world * seq_local + rank * seq_local + local


def oproj_sequence_row(rank, world, seq_local, row, causal):
    """OProj gather alone maps each rank to its two causal sequence chunks."""
    batch, local = divmod(row, seq_local)
    if causal:
        half = seq_local // 2
        chunk = rank if local < half else 2 * world - rank - 1
        sequence = chunk * half + local % half
    else:
        sequence = rank * seq_local + local
    return batch * world * seq_local + sequence


def a2a_bulk_tasks(m, world, comm_rows, m_window, comm_ctas):
    """Yield each scheduled (M tile, peer slot, row chunk) exactly as a CTA does."""
    m_tiles = ceil_div(m, 128)
    chunks = ceil_div(128, comm_rows)
    window_m = min(m_window, m_tiles)
    full_windows = m_tiles // window_m
    tasks_per_window = window_m * world * chunks
    full_tasks = full_windows * tasks_per_window
    for slot, cta in product(range(4), range(comm_ctas)):
        for task in range(slot * comm_ctas + cta, m_tiles * world * chunks, comm_ctas * 4):
            if task < full_tasks:
                window, inside = divmod(task, tasks_per_window)
                peer, in_peer = divmod(inside, window_m * chunks)
                tile_m = window * window_m + in_peer // chunks
            else:
                tail_m = m_tiles - full_windows * window_m
                peer, in_peer = divmod(task - full_tasks, tail_m * chunks)
                tile_m = full_windows * window_m + in_peer // chunks
            yield tile_m, peer, in_peer % chunks


def compile_host_probe(test_class, source, name, *flags):
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler or shutil.which(compiler[0]) is None:
        raise unittest.SkipTest("a host C++ compiler is required for real-source checks")
    directory = tempfile.TemporaryDirectory(prefix="fuse-sm103-route-test-")
    test_class.addClassCleanup(directory.cleanup)
    probe = Path(directory.name) / name
    # Supply source on stdin. Only compiler-generated temporary artifacts
    # are written, and TemporaryDirectory owns their complete lifecycle.
    result = subprocess.run(
        [*compiler, "-std=c++17", "-O2", "-I", str(ROOT / "include"),
         "-include", str(ROOT / "benchmarks/sm103/fused_mpi.cuh"), *flags,
         "-x", "c++", "-", "-o", str(probe)],
        input=source, text=True, capture_output=True, timeout=60,
    )
    if result.returncode:
        raise AssertionError(f"{name} failed to compile:\n{result.stderr}")
    return probe


class PublicCppLayout(unittest.TestCase):
    """These tests execute real public C++ head-layout functions, without CUDA."""

    @classmethod
    def setUpClass(cls):
        cls.probe = compile_host_probe(cls, CPP_PROBE, "public-layout-probe")

    def query(self, cases):
        result = subprocess.run(
            [str(self.probe)], input="".join(" ".join(map(str, case)) + "\n" for case in cases),
            text=True, capture_output=True, check=True, timeout=30,
        )
        values = [tuple(map(int, line.split())) for line in result.stdout.splitlines()]
        self.assertEqual(len(values), len(cases))
        return values

    def test_real_public_head_mapping_matches_qkv_copy_chunks(self):
        checked = 0
        for world, q_local, head_dim in product((1, 2, 4, 8), (1, 2, 4),
                                                (8, 64, 96, 128, 136, 192, 256)):
            q_heads, kv_heads = world * q_local, world
            width = (q_heads + 2 * kv_heads) * head_dim
            actual = self.query([(world, q_heads, kv_heads, head_dim, f) for f in range(width)])
            seen = Counter()
            for peer in range(world):
                for segment, local_heads, source_base, packed_base in (
                    (0, q_local, 0, 0),
                    (1, 1, q_heads * head_dim, q_local * head_dim),
                    (2, 1, (q_heads + kv_heads) * head_dim, (q_local + 1) * head_dim),
                ):
                    for local_head in range(local_heads):
                        for offset in range(0, head_dim, 128):
                            feature = source_base + (peer * local_heads + local_head) * head_dim + offset
                            for column in range(min(128, head_dim - offset)):
                                global_feature = feature + column
                                expected = (
                                    peer, packed_base + local_head * head_dim + offset + column,
                                    segment, peer * local_heads + local_head,
                                    offset + column, global_feature,
                                )
                                self.assertEqual(actual[global_feature], expected)
                                seen[global_feature] += 1
            self.assertEqual(seen, Counter({feature: 1 for feature in range(width)}))
            checked += 1
        self.assertEqual(checked, 84)

    def test_real_public_layout_rejects_invalid_features_and_splits(self):
        cases = [(0, 8, 8, 128, 0), (4, 6, 4, 128, 0), (8, 16, 4, 128, 0),
                 (4, 8, 4, 0, 0), (4, 8, 4, 128, -1), (4, 8, 4, 128, 2048)]
        for result in self.query(cases):
            self.assertEqual(result, (-1, -1, -1, -1, -1, -1))


class TilePolicyHostContracts(unittest.TestCase):
    """Real policy dispatch and peer validation; no CUDA or GEMM emulation."""

    @classmethod
    def setUpClass(cls):
        launch = (ROOT / "csrc/operators/sm103/detail/launch.cuh").read_text()
        bindings = launch[launch.index("namespace fuse {"):launch.index("using DeviceInfo =")]
        comm = (ROOT / "csrc/operators/sm103/detail/a2a_gemm.cuh").read_text()
        constants = comm[comm.index("template <\n    int32_t ReadyBlockM"):comm.index("  struct Arguments")]
        validator = comm[comm.index("  static bool supported_params("):
                         comm.index("  static cudaError_t initialize(")]
        gemm = (ROOT / "csrc/operators/sm103/detail/gemm.cuh").read_text()
        problem_check = gemm[gemm.index("__host__ __device__ constexpr int64_t a_row_stride("):
                             gemm.index("inline auto raster_option(")]
        public = (ROOT / "include/fuse/operators/primitives/a2a_gemm.h").read_text()
        params = public[public.index("enum class A2ALhsGemmPolicy"):
                        public.index("// Explicit precision name")]
        source = r"""
#include "fuse/layout/gemm.h"
#include "fuse/layout/ulysses.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <type_traits>
#define __host__
#define __device__
#define CUTLASS_HOST_DEVICE
using cudaError_t = int;
enum { cudaSuccess = 0, cudaErrorInvalidValue = 1, cudaErrorNotSupported = 2 };
namespace cute {
template <int M, int N, int K> struct Shape { static constexpr int dims[3] = {M,N,K}; };
template <int I, class S> constexpr int size(S) { return S::dims[I]; }
struct SM90_BULK_COPY_G2S {};
struct SM90_BULK_COPY_S2G {};
struct SM90_TMA_LOAD_3D {};
struct SM90_TMA_STORE_3D {};
}
namespace fuse {
using Bf16 = uint16_t;
constexpr int kMaxWorldSize = 8, kAlignment = 8;
""" + params + problem_check + constants + validator + r"""
};
// Opaque type identities preserve N/K/epilogue choices, without pretending
// to compile or execute any CUTLASS collective or tensor instruction.
template <int N, int K = 64, int E = 0> struct Bf16GemmTypes {
  using Identity = Bf16GemmTypes;
  using TileShape = cute::Shape<128,N,K>;
  static constexpr int epilogue_n = E;
  struct OutputGemm { using Identity = Bf16GemmTypes; using TileShape = typename Identity::TileShape; };
  struct PureGemm { using Identity = Bf16GemmTypes; using TileShape = typename Identity::TileShape; };
};
template <int N, int K = 64, int E = 0> struct A2ALhsGemmTypes : Bf16GemmTypes<N,K,E> {
  using Gemm = typename Bf16GemmTypes<N,K,E>::OutputGemm;
  using TelemetryGemm = Gemm;
  using TelemetryPureGemm = Gemm;
};
template <int N, int K, int E> using Mxfp8GemmFamily = Bf16GemmTypes<N,K,E>;
template <int E> using Mxfp8A2ALhsGemmTypes = A2ALhsGemmTypes<256,128,E>;
struct Mxfp8A2ALhsInputComm { static constexpr int kReadyBlockM = 128, kTileK = 128; };
template <bool Profile> using Mxfp8A2ALhsInputCommT = Mxfp8A2ALhsInputComm;
template <int N> struct QkvComm { static constexpr int kBlockM = 128, kBlockN = N; };
using QkvGqaPackCommN64 = QkvComm<64>;
using QkvGqaPackComm = QkvComm<128>;
using QkvGqaPackCommSmallInterleaved = QkvComm<128>;
using QkvGqaPackCommN160 = QkvComm<160>;
using QkvGqaPackCommN192 = QkvComm<192>;
using QkvGqaPackCommWide = QkvComm<256>;
using Mxfp8QkvGqaPackComm = QkvComm<256>;
namespace detail {
template <class G, class C> struct MonolithicGemm {};
template <class K, class P, bool Profile = false> struct InputProductionKernel {};
template <class K> struct RoleTelemetryKernel {};
}
template <class G, class C> struct GemmA2ARoleTelemetryKernel {};
}
""" + bindings + r"""
} }
template <bool Oproj, class Binding>
cudaError_t emit(int shard) {
  using Gemm = typename Binding::Gemm;
  using Pure = typename Binding::PureGemm;
  using Tile = typename Binding::TileShape;
  static_assert(std::is_same_v<typename Gemm::Identity, typename Pure::Identity>);
  bool accepted = true;
  if constexpr (Oproj) {
    using Comm = typename Binding::Comm;
#if FUSE_ENABLE_PROFILING
    static_assert(Comm::kTileK == Binding::TelemetryComm::kTileK);
#endif
    alignas(16) fuse::Bf16 data[8]{};
    uint32_t ready = 0;
    fuse::A2AGemmParams p;
    p.gemm.m = 128; p.gemm.n = 256; p.gemm.k = 4 * shard;
    p.route.world_size = 4; p.route.rank = 0; p.route.batch = 1;
    p.route.seq_local = 128; p.route.global_seq = 512;
    p.route.q_heads = 4; p.route.local_heads = 1; p.route.head_dim = shard;
    p.route.kind = fuse::RouteKind::kHeadToSequence;
    p.route.direction = fuse::RouteDirection::kInverse; p.route.channel_count = 1;
    p.num_comm_ctas = 8; p.epoch = 1; p.input_epoch = 1;
    p.input_staging = data; p.rhs_nt = data; p.output = data; p.ready = &ready;
    for (int peer = 0; peer < 4; ++peer) { p.peer_input[peer] = data; p.peer_input_ready[peer] = &ready; }
    accepted = Comm::supported_params(p);
  }
  std::cout << cute::size<0>(Tile{}) << ' ' << cute::size<1>(Tile{}) << ' '
            << cute::size<2>(Tile{}) << ' ' << Gemm::Identity::epilogue_n << ' ' << accepted << '\n';
  return cudaSuccess;
}
int main(int argc, char** argv) {
  if (argc != 5) return 3;
  const bool oproj = std::strcmp(argv[1], "oproj") == 0;
  const char* variable = oproj ? "FUSE_SM103_OPROJ_POLICY" : "FUSE_QKV_GEMM_POLICY";
  if (std::strcmp(argv[2], "unset") == 0) ::unsetenv(variable);
  else ::setenv(variable, argv[2], 1);
  const int shard = std::atoi(argv[3]);
  if (oproj) {
    fuse::OprojGemmPolicy policy{};
    int status = fuse::select_oproj_gemm_policy(&policy);
    auto visitor = [&](auto tag) { return emit<true, typename decltype(tag)::type>(shard); };
    return status ? status : fuse::visit_oproj_forward_policy(policy, visitor);
  }
  fuse::QkvGemmPolicy policy{};
  int status = fuse::select_qkv_gemm_policy(&policy);
  auto visitor = [&](auto tag) { return emit<false, typename decltype(tag)::type>(shard); };
  return status ? status : fuse::visit_qkv_forward_policy(policy, std::atoi(argv[4]) != 0, visitor);
}
"""
        cls.probes = {profiling: compile_host_probe(cls, source, f"tile-policy-{profiling}",
            f"-DFUSE_ENABLE_PROFILING={profiling}", "-Wall", "-Wextra", "-Werror")
            for profiling in (0, 1)}

    def run_policy(self, direction, policy, shard=128, interleaved=0, profiling=0):
        return subprocess.run([str(self.probes[profiling]), direction, policy, str(shard), str(interleaved)],
                              text=True, capture_output=True, timeout=10)

    def test_real_bindings_keep_complete_compute_and_telemetry_tile_identity(self):
        new = {"m128n128k128": (128, 128, 128, 0, 1),
               "m128n256k64e32": (128, 256, 64, 32, 1),
               "m128n256k128e32": (128, 256, 128, 32, 1)}
        for direction, profiling in product(("qkv", "oproj"), (0, 1)):
            policies = {"unset": (128, 128, 64, 0, 1), "auto": (128, 128, 64, 0, 1),
                        "m128n128": (128, 128, 64, 0, 1), "m128n256": (128, 256, 64, 0, 1), **new}
            if direction == "qkv":
                policies.update({f"m128n{n}": (128, n, 64, 0, 1) for n in (64, 160, 192)})
                policies["m128n256k64e64"] = (128, 256, 64, 64, 1)
            for policy, expected in policies.items():
                with self.subTest(direction=direction, policy=policy, profiling=profiling):
                    result = self.run_policy(direction, policy, profiling=profiling)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(tuple(map(int, result.stdout.split())), expected)

    def test_real_peer_validator_preserves_64_aligned_shards_only_for_k64(self):
        for profiling, shard in product((0, 1), (64, 128, 192, 256, 384, 2048)):
            for policy in ("auto", "m128n256", "m128n128k128", "m128n256k64e32", "m128n256k128e32"):
                result = self.run_policy("oproj", policy, shard=shard, profiling=profiling)
                self.assertEqual(result.returncode, 0, result.stderr)
                k = 128 if "k128" in policy else 64
                self.assertEqual(int(result.stdout.split()[-1]), int(shard % k == 0))

    def test_real_dispatch_rejects_unregistered_and_new_interleaved_variants(self):
        for direction, policy in product(("qkv", "oproj"), ("", "m128n256k128", "m128n128k64e32", "m128n512")):
            self.assertNotEqual(self.run_policy(direction, policy).returncode, 0)
        for policy in ("m128n128k128", "m128n256k64e32", "m128n256k128e32", "m128n256k64e64"):
            self.assertNotEqual(self.run_policy("qkv", policy, interleaved=1).returncode, 0)
        self.assertEqual(self.run_policy("qkv", "auto", interleaved=1).returncode, 0)
        self.assertNotEqual(self.run_policy("oproj", "m128n256k64e64").returncode, 0)
        for policy in ("m128n128k64e64", "m128n256k128e64", "m128n256k64e128"):
            self.assertNotEqual(self.run_policy("qkv", policy).returncode, 0)


class HarnessHostContracts(unittest.TestCase):
    """Execute the actual harness parser/candidate builder, not a Python copy.

    CLI and real-main scheduling checks use mocked CUDA work. They do not
    execute candidate kernels, device payload changes or GPU epoch protocols.
    """

    @classmethod
    def setUpClass(cls):
        source = (ROOT / "benchmarks/sm103/fused_bf16.cu").read_text()
        # Named boundaries intentionally fail if the real implementation moves;
        # never fall back to duplicated parser/candidate logic in this test.
        host_types = source[source.index("constexpr int kWarmup ="):
                            source.index("\nvoid check_cuda(")]
        parser = source[source.index("Options parse_options("):
                        source.index("\nstruct RankRuntime")]
        probe_source = r"""
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
// Storage-width substitute only: no CUDA/BF16 arithmetic is tested here.
using Bf16 = uint16_t;
#include "fuse/layout/gemm.h"
""" + host_types + parser + r"""
int main(int argc, char** argv) {
  try {
    const auto options = parse_options(argc, argv);
    std::cout << "shape " << options.world << ' ' << options.seq_local << ' '
              << options.global_seq() << ' ' << options.hidden << ' '
              << options.q_heads << ' ' << options.kv_heads << ' '
              << options.head_dim << ' ' << options.timeout_seconds << ' '
              << options.seed << ' ' << options.causal << ' ' << options.profile << '\n';
    std::cout << "comm";
    for (const auto value : options.comm_sm_list) std::cout << ' ' << value;
    std::cout << "\npolicy";
    for (const auto& value : options.qkv_policy_list) std::cout << ' ' << value;
    std::cout << "\noproj_policy";
    for (const auto& value : options.oproj_policy_list) std::cout << ' ' << value;
    std::cout << "\ndiagnostics " << options.cpu_oracle << ' '
              << options.validation_self_test << '\n';
    std::cout << "generator " << options.input_generator << '\n';
    std::cout << "host_launch " << options.host_launch << '\n';
    std::cout << "calibration " << options.calibrate << ' '
              << component_name(options.component) << '\n';
    std::cout << "profile_detail " << options.profile_detail << '\n';
    fuse::GemmProblem qkv, oproj;
    apply_schedule(qkv, options, Direction::kQkv);
    apply_schedule(oproj, options, Direction::kOproj);
    const auto qs = schedule_geometry(ceil_div(options.seq_local, 128),
        ceil_div(options.projection_width(), std::stoi(options.qkv_policy_list.front().substr(5))), qkv.max_swizzle_size);
    const auto os = schedule_geometry(ceil_div(options.seq_local, 128),
        ceil_div(options.hidden, std::stoi(options.oproj_policy_list.front().substr(5))), oproj.max_swizzle_size);
    std::cout << "schedule " << options.max_swizzle_size << ' ' << options.qkv_raster << ' ' << options.oproj_raster
              << ' ' << effective_raster(qkv, Direction::kQkv) << ' ' << effective_raster(oproj, Direction::kOproj)
              << ' ' << qs.effective_swizzle_size << ' ' << qs.padded_m_tiles << ' ' << qs.padded_n_tiles
              << ' ' << os.effective_swizzle_size << ' ' << os.padded_m_tiles << ' ' << os.padded_n_tiles << '\n';
    std::cout << "comm_layout " << options.oproj_comm_layout << '\n';
    for (const auto& candidate : make_candidates(options)) {
      std::cout << "candidate " << direction_name(candidate.direction) << ' '
                << candidate.comm_sm << ' ' << candidate.tile_policy << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
"""
        cls.probes = {
            profiling: compile_host_probe(cls, probe_source, f"harness-host-{profiling}",
                                          f"-DFUSE_ENABLE_PROFILING={profiling}")
            for profiling in (0, 1)
        }
        # Execute the real main's control flow with explicit CPU stand-ins for
        # CUDA work. This validates scheduling/acceptance, not device behavior.
        flow_source = probe_source[:probe_source.index("int main(")] + r"""
#include "fused_validation.cuh"
#include "fused_inputs.cuh"
#include "fused_launch.cuh"
#include <atomic>
#include <chrono>
#include <csignal>
#include <exception>
#include <iomanip>
#include <memory>
#include <unistd.h>
struct RankRuntime {};
void resolve_auto_candidates(std::vector<RankRuntime>&, std::vector<Candidate>&) {
  throw std::runtime_error("auto CTA resolution requires its dedicated API stand-in test");
}
void bind_graph(std::vector<RankRuntime>&, const Options& options, uint32_t epoch) {
  if (options.launch == "graph") std::cout << "mock,graph_bind," << epoch << '\n';
}
void report_graph_preparation(const std::vector<RankRuntime>&, const Options& options,
                              Direction, const std::string& context) {
  if (options.launch == "graph") std::cout << "mock,graph_prepare" << context << '\n';
}
uint32_t payload_generation = 0, reference_generation = 99, last_epoch[2] = {};
uint32_t reference_last_epoch[2][2] = {};
int last_selected_direction = -1;
int selected_comm[2] = {};
std::string selected_policy[2];
std::string selected_input_generator;
std::string selected_host_launch;
fused_launch::Team* selected_launch_team = nullptr;
std::atomic<int> started_workers{0}, stopped_workers{0};
struct WorkerLifetime {
  WorkerLifetime() { ++started_workers; }
  ~WorkerLifetime() { ++stopped_workers; }
};
void check_policy_environment(Direction direction, const std::string& expected) {
  const char* name = direction == Direction::kQkv
      ? "FUSE_QKV_GEMM_POLICY" : "FUSE_SM103_OPROJ_POLICY";
  const char* actual = std::getenv(name);
  if (!actual || actual != expected) throw std::runtime_error("wrong directional policy environment");
}
void timeout_handler(int) {}
std::vector<RankRuntime> create_runtimes(const Options& options) {
  check_policy_environment(Direction::kQkv, options.qkv_policy_list.front());
  check_policy_environment(Direction::kOproj, options.oproj_policy_list.front());
  const char* layout = std::getenv("FUSE_SM103_OPROJ_COMM_LAYOUT");
  if (!layout || layout != options.oproj_comm_layout) throw std::runtime_error("wrong communication layout environment");
  std::cout << "mock,setup\n";
  selected_input_generator = options.input_generator;
  selected_host_launch = options.host_launch;
  std::cout << "mock,setup_generator," << selected_input_generator << '\n';
  std::cout << "mock,setup_policy,qkv=" << std::getenv("FUSE_QKV_GEMM_POLICY")
            << ",oproj=" << std::getenv("FUSE_SM103_OPROJ_POLICY") << '\n';
  return std::vector<RankRuntime>(options.world);
}
std::unique_ptr<fused_launch::Team> start_launch_team(
    std::vector<RankRuntime>& runtimes, const Options& options) {
  if (options.host_launch != selected_host_launch) throw std::runtime_error("launch mode changed");
  std::cout << "mock,launch_team," << options.host_launch << '\n';
  if (options.host_launch == "sequential") return nullptr;
  if (options.host_launch != "per_gpu_thread") throw std::runtime_error("unknown launch mode");
  auto team = std::make_unique<fused_launch::Team>(static_cast<int>(runtimes.size()));
  const auto initialized = team->dispatch([](void*, int) {
    thread_local WorkerLifetime lifetime;
    (void)lifetime;
  }, nullptr);
  if (initialized.error) std::rethrow_exception(initialized.error);
  selected_launch_team = team.get();
  return team;
}
void set_inputs(std::vector<RankRuntime>&, const Options& options, uint32_t generation) {
  if (options.input_generator != selected_input_generator) throw std::runtime_error("input generator changed");
  payload_generation = generation;
  std::cout << "mock,input," << generation << '\n';
  std::cout << "mock,input_generator," << generation << ',' << options.input_generator << '\n';
}
void prepare_references(std::vector<RankRuntime>&, const Options& options) {
  if (options.input_generator != selected_input_generator) throw std::runtime_error("reference generator changed");
  reference_generation = payload_generation;
  std::cout << "mock,reference," << payload_generation << '\n';
  std::cout << "mock,reference_generator," << payload_generation << ',' << options.input_generator << '\n';
}
void select_candidate(std::vector<RankRuntime>&, const Candidate& candidate, const std::string& context) {
  // The GPU selection stub delegates environment mutation to the real helper.
  set_tile_policy(candidate.direction, candidate.tile_policy);
  last_selected_direction = candidate.direction == Direction::kQkv ? 0 : 1;
  selected_comm[last_selected_direction] = candidate.comm_sm;
  selected_policy[last_selected_direction] = candidate.tile_policy;
  check_policy_environment(candidate.direction, candidate.tile_policy);
  std::cout << "mock,select" << context << '\n';
}
void poison_outputs(std::vector<RankRuntime>&, const Options&, Direction direction) {
  std::cout << "mock,poison," << direction_name(direction) << '\n';
}
void describe_component(std::vector<RankRuntime>&, const Options& options,
                        Direction direction, const std::string& context) {
  std::cout << "mock,component," << direction_name(direction) << context
            << ",selected_component=" << component_name(options.component) << '\n';
}
void reset_calibration(std::vector<RankRuntime>&, const Options& options, Direction direction) {
  if (!options.calibrate) throw std::runtime_error("unexpected calibration reset");
  const int index = direction == Direction::kQkv ? 0 : 1;
  reference_last_epoch[index][0] = reference_last_epoch[index][1] = 0;
  std::cout << "mock,calibration_reset," << direction_name(direction)
            << ",generation=" << payload_generation << '\n';
}
void prepare_component(std::vector<RankRuntime>&, const Options& options, Direction direction) {
  if (!options.calibrate || options.component == MeasurementComponent::kFused) {
    throw std::runtime_error("unexpected reference preparation");
  }
  std::cout << "mock,prepare_component," << direction_name(direction)
            << ",component=" << component_name(options.component)
            << ",generation=" << payload_generation << '\n';
}
void run_epoch(std::vector<RankRuntime>& runtimes, const Options& options, Direction direction, uint32_t epoch) {
  if (options.host_launch != selected_host_launch ||
      (options.host_launch == "per_gpu_thread") != (selected_launch_team != nullptr)) {
    throw std::runtime_error("epoch lost launch mode/team");
  }
  if (selected_launch_team) {
    std::vector<int> called(runtimes.size());
    const auto result = selected_launch_team->dispatch([](void* pointer, int rank) {
      ++static_cast<std::vector<int>*>(pointer)->at(rank);
    }, &called);
    if (result.error) std::rethrow_exception(result.error);
    for (int count : called) if (count != 1) throw std::runtime_error("bad host rank dispatch");
  }
  const int index = direction == Direction::kQkv ? 0 : 1;
  auto& observed = options.component == MeasurementComponent::kFused ? last_epoch[index]
      : reference_last_epoch[index][options.component == MeasurementComponent::kComputeReference ? 0 : 1];
  if (epoch != ++observed) throw std::runtime_error("noncontiguous component host epoch");
  std::cout << "mock,epoch," << direction_name(direction) << ',' << epoch << '\n';
  std::cout << "mock,component_epoch," << direction_name(direction)
            << ",component=" << component_name(options.component)
            << ",generation=" << payload_generation << ",epoch=" << epoch << '\n';
}
void validate(std::vector<RankRuntime>&, const Options& options, Direction direction, const std::string& context) {
  if (reference_generation != payload_generation) throw std::runtime_error("stale reference");
  std::cout << "mock,validate" << context << '\n' << std::flush;
  std::cout << "mock,component_validate," << direction_name(direction) << context
            << ",selected_component=" << component_name(options.component) << '\n' << std::flush;
  if (payload_generation == 1 && std::getenv("FUSE_TEST_FAIL_SECOND_PAYLOAD")) {
    throw std::runtime_error("injected second-payload failure");
  }
  if (options.component == MeasurementComponent::kFused &&
      std::getenv("FUSE_TEST_FAIL_GRAPH_POST") &&
      context.find(",validation_phase=post") != std::string::npos) {
    throw std::runtime_error("injected Graph last-output failure");
  }
  const char* fail_component = std::getenv("FUSE_TEST_FAIL_CALIBRATION_COMPONENT");
  if (fail_component && options.component != MeasurementComponent::kFused &&
      std::string(fail_component) == component_name(options.component) &&
      context.find(",validation_phase=post") != std::string::npos) {
    throw std::runtime_error("injected calibration post-sample failure");
  }
}
void validation_self_test(std::vector<RankRuntime>&, const Options&, Direction direction,
                          const std::string& context) {
  std::cout << "mock,self_test," << direction_name(direction) << context << '\n' << std::flush;
  if (std::getenv("FUSE_TEST_FAIL_SELF_TEST")) {
    throw std::runtime_error("injected checker self-test failure");
  }
}
void benchmark(std::vector<RankRuntime>& runtimes, const Options& options, Direction direction,
               uint32_t& epoch, const std::string& context) {
  // Only exercise the shared epoch reference; no timing model is claimed.
  for (int index = 0; index < 3; ++index) run_epoch(runtimes, options, direction, ++epoch);
  std::cout << "mock,benchmark" << context << '\n';
  std::cout << "mock,component_benchmark," << direction_name(direction) << context
            << ",selected_component=" << component_name(options.component) << '\n';
}
void profile_compute_gaps(std::vector<RankRuntime>&, const Options&) {}
void profile(std::vector<RankRuntime>& runtimes, const Options& options,
             Direction direction, uint32_t& epoch) {
  const int index = direction == Direction::kQkv ? 0 : 1;
  const auto& policy = (index == 0 ? options.qkv_policy_list : options.oproj_policy_list).front();
  if (last_selected_direction != index || selected_policy[index] != policy ||
      selected_comm[index] != options.comm_sm) {
    throw std::runtime_error("profile did not reselect its own candidate");
  }
  check_policy_environment(direction, policy);
  run_epoch(runtimes, options, direction, ++epoch);
  std::cout << "mock,profile," << direction_name(direction) << ",tile=" << policy << '\n';
}
void profile_qkv_epilogue(std::vector<RankRuntime>&, const Options&, uint32_t&) {}
void destroy_runtimes(std::vector<RankRuntime>& runtimes, const Options& options) {
  const int expected = options.host_launch == "per_gpu_thread" ? runtimes.size() : 0;
  if (started_workers != expected || stopped_workers != expected) {
    throw std::runtime_error("runtime destroyed before launch workers joined");
  }
  std::cout << "mock,launch_workers_joined," << stopped_workers << '\n';
  std::cout << "mock,cleanup\n";
}
""" + source[source.index("struct StageTimer {"):
             source.index("template <class T>\ndouble percentile")] + source[source.index("int main("):]
        cls.flow_probe = compile_host_probe(cls, flow_source, "harness-main-flow",
                                            "-DFUSE_ENABLE_PROFILING=1", "-pthread", "-I",
                                            str(ROOT / "benchmarks/sm103"))
        # Execute the real main's Graph branch with explicit transport stand-ins.
        # The separate MPI seam probe exercises the unmodified MPI parser.
        graph_flow = flow_source.replace("Options parse_options(", "Options parse_flow_options(", 1)
        point = graph_flow.index("int main(")
        graph_flow = graph_flow[:point] + r'''
Options parse_options(int argc, char** argv) {
  auto options = parse_flow_options(argc, argv);
  options.launch = "graph";
  return options;
}
''' + graph_flow[point:]
        cls.graph_flow_probe = compile_host_probe(cls, graph_flow, "harness-graph-main-flow",
            "-DFUSE_ENABLE_PROFILING=1", "-pthread", "-I", str(ROOT / "benchmarks/sm103"))

    def invoke(self, *arguments, profiling=0, policy_env=None, oproj_policy_env=None, comm_layout_env=None):
        environment = os.environ.copy()
        environment.pop("FUSE_QKV_GEMM_POLICY", None)
        environment.pop("FUSE_SM103_OPROJ_POLICY", None)
        environment.pop("FUSE_SM103_OPROJ_COMM_LAYOUT", None)
        if policy_env is not None:
            environment["FUSE_QKV_GEMM_POLICY"] = policy_env
        if oproj_policy_env is not None:
            environment["FUSE_SM103_OPROJ_POLICY"] = oproj_policy_env
        if comm_layout_env is not None:
            environment["FUSE_SM103_OPROJ_COMM_LAYOUT"] = comm_layout_env
        return subprocess.run([str(self.probes[profiling]), *arguments],
                              env=environment, text=True, capture_output=True, timeout=10)

    def query(self, *arguments, **kwargs):
        result = self.invoke(*arguments, **kwargs)
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertGreaterEqual(len(lines), 8, result.stdout)
        self.assertEqual(lines[0].split()[0], "shape")
        self.assertEqual(lines[1].split()[0], "comm")
        self.assertEqual(lines[2].split()[0], "policy")
        self.assertEqual(lines[3].split()[0], "oproj_policy")
        self.assertEqual(lines[4].split()[0], "diagnostics")
        self.assertEqual(lines[5].split()[0], "generator")
        self.assertEqual(lines[6].split()[0], "host_launch")
        self.assertEqual(lines[7].split()[0], "calibration")
        candidates = []
        for line in lines[11:]:
            tag, direction, comm_sm, policy = line.split()
            self.assertEqual(tag, "candidate")
            candidates.append((direction, int(comm_sm), policy))
        return {"shape": tuple(map(int, lines[0].split()[1:])),
                "comm": list(map(int, lines[1].split()[1:])),
                "policy": lines[2].split()[1:], "candidates": candidates,
                "oproj_policy": lines[3].split()[1:],
                "diagnostics": tuple(map(int, lines[4].split()[1:])),
                "generator": lines[5].split()[1],
                "host_launch": lines[6].split()[1],
                "calibration": (int(lines[7].split()[1]), lines[7].split()[2]),
                "profile_detail": lines[8].split()[1],
                "schedule": tuple(lines[9].split()[1:]),
                "comm_layout": lines[10].split()[1]}

    def assert_rejected(self, *arguments, **kwargs):
        result = self.invoke(*arguments, **kwargs)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertTrue(result.stderr.strip())
        self.assertNotIn("candidate ", result.stdout)

    def test_defaults_preserve_single_candidate_per_direction_and_shape(self):
        actual = self.query()
        self.assertEqual(actual["shape"], (4, 256, 1024, 1024, 32, 8, 128, 60,
                                           20260906, 0, 0))
        self.assertEqual(actual["comm"], [8])
        self.assertEqual(actual["policy"], ["m128n128"])
        self.assertEqual(actual["oproj_policy"], ["m128n128"])
        self.assertEqual(actual["diagnostics"], (0, 0))
        self.assertEqual(actual["generator"], "cpu_mt19937")
        self.assertEqual(actual["host_launch"], "sequential")
        self.assertEqual(actual["calibration"], (0, "fused"))
        self.assertEqual(actual["profile_detail"], "full")
        self.assertEqual(actual["schedule"][:5], ("1", "heuristic", "heuristic", "along_m", "along_n"))
        self.assertEqual(actual["candidates"], [("GEMM_A2A", 8, "m128n128"),
                                                ("A2A_GEMM", 8, "m128n128")])

    def test_profile_detail_is_explicit_and_requires_profiling(self):
        self.assertEqual(self.query("--profile", profiling=1)["profile_detail"], "full")
        for detail in ("full", "cta"):
            for arguments in (("--profile", "--profile-detail", detail),
                              ("--profile-detail", detail, "--profile")):
                self.assertEqual(self.query(*arguments, profiling=1)["profile_detail"], detail)
                self.assert_rejected(*arguments, profiling=0)
            self.assert_rejected("--profile-detail", detail, profiling=1)
        for value in ("", "peer", "CTA", "full,cta"):
            self.assert_rejected("--profile", "--profile-detail", value, profiling=1)
        self.assert_rejected("--profile", "--profile-detail", profiling=1)

    def test_epilogue_probe_is_explicit_profile_only_with_one_real_policy(self):
        args = ("--profile", "--profile-detail", "cta", "--qkv-policy-list", "m128n256k64e32",
                "--qkv-epilogue-probe")
        self.query(*args, profiling=1)
        self.assert_rejected(*args, profiling=0)
        for extra in ((), ("--profile",), ("--profile", "--profile-detail", "cta"),
                      ("--profile", "--qkv-policy-list", "m128n256k64e32")):
            self.assert_rejected("--qkv-epilogue-probe", *extra, profiling=1)

    def test_schedule_options_share_actual_problem_fields_without_expanding_candidates(self):
        for swizzle, qkv, oproj in product((1, 2, 4, 8), ("heuristic", "along_m", "along_n"),
                                         ("heuristic", "along_m", "along_n")):
            result = self.query("--max-swizzle-size", str(swizzle), "--qkv-raster", qkv,
                                "--oproj-raster", oproj)
            self.assertEqual(result["schedule"][:5], (str(swizzle), qkv, oproj,
                "along_m" if qkv == "heuristic" else qkv, "along_n" if oproj == "heuristic" else oproj))
            self.assertEqual(len(result["candidates"]), 2)

    def test_schedule_cli_invalid_values_are_rejected_before_any_candidate(self):
        for value in ("0", "3", "5", "16", "-1", "2,4", "", "4x", "4294967296"):
            self.assert_rejected("--max-swizzle-size", value)
        self.assert_rejected("--max-swizzle-size")
        for option in ("--qkv-raster", "--oproj-raster"):
            for value in ("auto", "m", "AlongM", "along_m,along_n", ""):
                self.assert_rejected(option, value)
            self.assert_rejected(option)

    def test_oproj_comm_layout_is_independent_of_tiles_and_cli_overrides_environment(self):
        args = ("--comm-sm-list", "8,16", "--qkv-policy-list", "auto,m128n256",
                "--oproj-policy-list", "m128n128,m128n256k128e32", "--calibrate")
        default = self.query(*args)
        self.assertEqual(default["comm_layout"], "rows")
        for layout in ("rows", "columns"):
            actual = self.query(*args, "--oproj-comm-layout", layout, comm_layout_env="invalid")
            self.assertEqual(actual["comm_layout"], layout)
            self.assertEqual(actual["candidates"], default["candidates"])
            self.assertEqual(self.query(*args, comm_layout_env=layout)["comm_layout"], layout)
        for value in ("", "Rows", "rows,columns", "invalid"):
            self.assert_rejected("--oproj-comm-layout", value)
            self.assert_rejected(comm_layout_env=value)
        self.assert_rejected("--oproj-comm-layout")

    def test_fixed_comm_layout_reaches_setup_config_and_mpi_agreement(self):
        for layout in ("rows", "columns"):
            result = self.run_flow("--oproj-comm-layout", layout, "--comm-sm-list", "8,16",
                "--oproj-policy-list", "m128n128,m128n256", "--calibrate", graph=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            config = next(row for row in result.stdout.splitlines() if row.startswith("config,"))
            self.assertIn("oproj_comm_layout=" + layout, config)
            self.assertIn("candidates=6", config)
        source = (ROOT / "benchmarks/sm103/fused_bf16.cu").read_text()
        agreement = source[source.index("std::ostringstream contract;"):source.index("fused_mpi::agree(contract.str());")]
        self.assertIn("options.oproj_comm_layout", agreement)

    def test_effective_swizzle_downshifts_and_padding_is_not_hidden(self):
        # Actual parser/helper results at CUTLASS's min-tile thresholds 2/3/6.
        for m_tiles, effective, padded in ((1, 1, 1), (2, 2, 2), (3, 4, 4), (5, 4, 8),
                                          (6, 8, 8), (8, 8, 8)):
            actual = self.query("--seq-local", str(m_tiles * 128), "--max-swizzle-size", "8")
            self.assertEqual(actual["schedule"][5:],
                             tuple(map(str, (effective, padded, 48, effective, padded, 8))))
        actual = self.query("--seq-local", "384", "--hidden", "640", "--max-swizzle-size", "8")
        self.assertEqual(actual["schedule"][5:], ("4", "4", "48", "4", "4", "8"))

    def test_profile_padding_rejection_does_not_disable_production_or_calibration(self):
        padded = ("--seq-local", "384", "--hidden", "640", "--max-swizzle-size", "8")
        self.query(*padded)
        self.query(*padded, "--calibrate")
        result = self.invoke(*padded, "--profile", profiling=1)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--profile does not support swizzle-padded M/N tiles", result.stderr)
        self.assertNotIn("candidate ", result.stdout)
        # Requested8 but actual1 is legitimate, as is an unpadded actual8.
        for rows in (128, 1024):
            self.query("--seq-local", str(rows), "--max-swizzle-size", "8", "--profile", profiling=1)

    def test_schedule_is_in_real_mpi_agreement_and_raw_config(self):
        result = self.run_flow("--calibrate", "--max-swizzle-size", "4",
                               "--qkv-raster", "along_n", "--oproj-raster", "along_m")
        self.assertEqual(result.returncode, 0, result.stderr)
        config = next(line for line in result.stdout.splitlines() if line.startswith("config,"))
        for field in ("max_swizzle_size=4", "qkv_raster=along_n", "oproj_raster=along_m",
                      "qkv_effective_raster=along_n", "oproj_effective_raster=along_m"):
            self.assertIn(field, config)
        source = (ROOT / "benchmarks/sm103/fused_bf16.cu").read_text()
        agreement = source[source.index("std::ostringstream contract;"):source.index("fused_mpi::agree(contract.str());")]
        for field in ("options.max_swizzle_size", "options.qkv_raster", "options.oproj_raster"):
            self.assertIn(field, agreement)
        self.assertIn("apply_schedule(runtime.qkv.gemm, options, Direction::kQkv);", source)
        self.assertIn("apply_schedule(runtime.oproj.gemm, options, Direction::kOproj);", source)

    def test_lists_stably_deduplicate_before_direction_specific_expansion(self):
        policies = ["m128n256", "m128n64", "m128n128", "m128n192", "m128n160"]
        actual = self.query("--comm-sm-list", "12,8,12,4,8", "--qkv-policy-list",
                            "m128n256,m128n64,auto,m128n128,m128n192,m128n160,auto",
                            "--oproj-policy-list", "m128n256,auto,m128n128,m128n256")
        self.assertEqual(actual["comm"], [12, 8, 4])
        self.assertEqual(actual["policy"], policies)
        self.assertEqual(actual["oproj_policy"], ["m128n256", "m128n128"])
        expected = [("GEMM_A2A", comm, policy) for policy in policies for comm in (12, 8, 4)]
        expected += [("A2A_GEMM", comm, policy) for policy in ("m128n256", "m128n128")
                     for comm in (12, 8, 4)]
        self.assertEqual(actual["candidates"], expected)
        self.assertEqual(len(actual["candidates"]), (5 + 2) * 3)
        self.assertEqual(len(actual["candidates"]), len(set(actual["candidates"])))

    def test_new_k_epilogue_policies_are_explicit_and_deduplicate_without_aliasing(self):
        policies = ["m128n128k128", "m128n256k64e32", "m128n256k128e32"]
        value = ",".join([*policies, policies[0]])
        actual = self.query("--qkv-policy-list", value, "--oproj-policy-list", value)
        self.assertEqual(actual["policy"], policies)
        self.assertEqual(actual["oproj_policy"], policies)
        self.assertEqual(actual["candidates"], [(direction, 8, policy)
            for direction, policy in product(("GEMM_A2A", "A2A_GEMM"), policies)])
        for policy in policies:
            env = self.query(policy_env=policy, oproj_policy_env=policy)
            self.assertEqual(env["policy"], [policy])
            self.assertEqual(env["oproj_policy"], [policy])
            profile = self.query("--profile", "--qkv-policy-list", policy,
                                 "--oproj-policy-list", policy, profiling=1)
            self.assertEqual(profile["policy"], [policy])
            self.assertEqual(profile["oproj_policy"], [policy])
        for option in ("--qkv-policy-list", "--oproj-policy-list"):
            for invalid in ("m128n256k128", "m128n128k64e32", "m128n256k128e64"):
                self.assert_rejected(option, invalid)
        actual = self.query("--qkv-policy-list", "m128n256k64e32,m128n256k64e64")
        self.assertEqual(actual["policy"], ["m128n256k64e32", "m128n256k64e64"])
        self.assert_rejected("--oproj-policy-list", "m128n256k64e64")

    def test_explicit_scalar_and_list_communication_options_are_mutually_exclusive(self):
        for arguments in (("--comm-sm", "8", "--comm-sm-list", "8"),
                          ("--comm-sm-list", "8", "--comm-sm", "8")):
            with self.subTest(arguments=arguments):
                self.assert_rejected(*arguments)
        self.assertEqual(self.query("--comm-sm", "12")["comm"], [12])

    def test_communication_list_rejects_invalid_tokens_without_partial_candidates(self):
        for value in ("", ",8", "8,", "8,,12", "0", "-1", "1025", "2147483648",
                      "4294967296", "18446744073709551616", "8.0", "eight", "8,0"):
            with self.subTest(value=value):
                self.assert_rejected("--comm-sm-list", value)
        self.assert_rejected("--comm-sm-list")
        # These are host bounds only; 1024 still requires a runtime SM check.
        self.assertEqual(self.query("--comm-sm-list", "1,1024")["comm"], [1, 1024])

    def test_policy_list_rejects_empty_or_unsupported_tokens(self):
        for value in ("", ",auto", "auto,", "auto,,m128n64", "m128n320",
                      "M128N128", "fp8", "auto,unknown"):
            with self.subTest(value=value):
                self.assert_rejected("--qkv-policy-list", value)
        self.assert_rejected("--qkv-policy-list")

    def test_policy_environment_default_and_explicit_override(self):
        self.assertEqual(self.query(policy_env="m128n64")["policy"], ["m128n64"])
        self.assertEqual(self.query(policy_env="auto")["policy"], ["m128n128"])
        self.assert_rejected(policy_env="unknown")
        actual = self.query("--qkv-policy-list", "auto,m128n192", policy_env="unknown")
        self.assertEqual(actual["policy"], ["m128n128", "m128n192"])

    def test_oproj_policy_list_rejects_empty_and_non_oproj_tiles(self):
        for value in ("", ",auto", "auto,", "auto,,m128n256", "m128n64", "m128n160",
                      "m128n192", "m128n320", "M128N128", "fp8", "auto,unknown"):
            with self.subTest(value=value):
                self.assert_rejected("--oproj-policy-list", value)
        self.assert_rejected("--oproj-policy-list")

    def test_oproj_environment_is_direction_specific_and_explicit_list_overrides_it(self):
        actual = self.query(policy_env="m128n64", oproj_policy_env="m128n256")
        self.assertEqual(actual["policy"], ["m128n64"])
        self.assertEqual(actual["oproj_policy"], ["m128n256"])
        self.assertEqual(actual["candidates"], [("GEMM_A2A", 8, "m128n64"),
                                                ("A2A_GEMM", 8, "m128n256")])
        self.assertEqual(self.query(oproj_policy_env="auto")["oproj_policy"], ["m128n128"])
        for value in ("unknown", "m128n64", "m128n160", "auto,m128n256"):
            with self.subTest(value=value):
                self.assert_rejected(oproj_policy_env=value)
        actual = self.query("--oproj-policy-list", "m128n256,auto,m128n128",
                            oproj_policy_env="unknown")
        self.assertEqual(actual["oproj_policy"], ["m128n256", "m128n128"])
        self.assert_rejected("--qkv-policy-list", "auto", oproj_policy_env="unknown")
        self.assert_rejected("--oproj-policy-list", "auto", policy_env="unknown")

    def test_input_generator_requires_an_explicit_supported_name(self):
        for generator in ("cpu_mt19937", "gpu_philox"):
            self.assertEqual(self.query("--input-generator", generator)["generator"], generator)
        for generator in ("", "gpu", "mt19937", "GPU_PHILOX", "cpu_mt19937,gpu_philox"):
            with self.subTest(generator=generator):
                self.assert_rejected("--input-generator", generator)
        self.assert_rejected("--input-generator")

    def test_profile_requires_enabled_build_and_one_effective_candidate_per_direction(self):
        self.assert_rejected("--profile")
        actual = self.query("--profile", "--comm-sm-list", "8,8", "--qkv-policy-list",
                            "auto,m128n128", profiling=1)
        self.assertEqual(actual["shape"][-1], 1)
        self.assertEqual(len(actual["candidates"]), 2)
        self.assert_rejected("--profile", "--comm-sm-list", "8,12", profiling=1)
        self.assert_rejected("--profile", "--qkv-policy-list", "auto,m128n64", profiling=1)
        self.assert_rejected("--profile", "--oproj-policy-list", "auto,m128n256", profiling=1)
        single_wide = self.query("--profile", "--oproj-policy-list", "m128n256,m128n256",
                                 profiling=1)
        self.assertEqual(single_wide["candidates"], [("GEMM_A2A", 8, "m128n128"),
                                                     ("A2A_GEMM", 8, "m128n256")])
        self.query("--profile", "--oproj-policy-list", "auto,m128n128", profiling=1)

    def test_host_launch_requires_a_supported_explicit_name(self):
        for mode in ("sequential", "per_gpu_thread"):
            self.assertEqual(self.query("--host-launch", mode)["host_launch"], mode)
        for mode in ("", "parallel", "per-gpu-thread", "PER_GPU_THREAD",
                     "sequential,per_gpu_thread"):
            with self.subTest(mode=mode):
                self.assert_rejected("--host-launch", mode)
        self.assert_rejected("--host-launch")

    def test_calibration_is_explicit_and_excludes_profile_and_self_test(self):
        for mode in ("sequential", "per_gpu_thread"):
            with self.subTest(mode=mode):
                actual = self.query("--calibrate", "--cpu-oracle", "--host-launch", mode)
                self.assertEqual(actual["calibration"], (1, "fused"))
                self.assertEqual(actual["diagnostics"], (1, 0))
                self.assertEqual(actual["host_launch"], mode)
                self.assertEqual(len(actual["candidates"]), 2)
        for exclusive in ("--profile", "--validation-self-test"):
            for arguments in (("--calibrate", exclusive), (exclusive, "--calibrate")):
                with self.subTest(arguments=arguments):
                    self.assert_rejected(*arguments, profiling=1)

    def test_training_shapes_preserve_global_local_geometry(self):
        for sequence, hidden, q_heads in ((65536, 2048, 16), (131072, 4096, 32)):
            with self.subTest(sequence=sequence):
                actual = self.query("--world", "8", "--global-seq", str(sequence),
                                    "--hidden", str(hidden), "--q-heads", str(q_heads),
                                    "--kv-heads", "8", "--head-dim", "128")
                self.assertEqual(actual["shape"][:7],
                                 (8, sequence // 8, sequence, hidden, q_heads, 8, 128))

    def test_invalid_geometry_and_overflow_still_rejected(self):
        for arguments in (("--global-seq", "65537", "--world", "8"),
                          ("--seq-local", "256", "--global-seq", "1024"),
                          ("--seq-local", "2147483647"),
                          ("--q-heads", "2147483647"),
                          ("--head-dim", "2147483647"),
                          ("--kv-heads", "0"), ("--q-heads", "12"),
                          ("--hidden", "7"), ("--world", "8", "--head-dim", "8"),
                          ("--causal", "--seq-local", "3")):
            with self.subTest(arguments=arguments):
                self.assert_rejected(*arguments)

    def test_cpu_diagnostics_are_explicit_and_self_test_implies_cpu_oracle(self):
        self.assertEqual(self.query("--cpu-oracle")["diagnostics"], (1, 0))
        self.assertEqual(self.query("--validation-self-test")["diagnostics"], (1, 1))
        self.assertEqual(self.query("--validation-self-test", "--cpu-oracle")["diagnostics"], (1, 1))

    def test_cpu_diagnostic_cap_is_per_rank_elements_not_a_production_shape_limit(self):
        # QKV is largest here: 8192 * 512 == 4194304 elements per rank.
        shape = ("--world", "4", "--q-heads", "8", "--kv-heads", "4",
                 "--head-dim", "32", "--hidden", "64")
        for option in ("--cpu-oracle", "--validation-self-test"):
            with self.subTest(option=option):
                self.query(*shape, "--seq-local", "8192", option)
                self.assert_rejected(*shape, "--seq-local", "8193", option)
        self.query(*shape, "--seq-local", "8193")
        # A large hidden activation must also trigger the diagnostic-only cap.
        wide = ("--seq-local", "512", "--hidden", "8200")
        self.assert_rejected(*wide, "--cpu-oracle")
        self.query(*wide)

    def run_flow(self, *arguments, fail_second_payload=False, fail_self_test=False,
                 fail_calibration_component=None, graph=False, fail_graph_post=False):
        environment = os.environ.copy()
        environment.pop("FUSE_QKV_GEMM_POLICY", None)
        environment.pop("FUSE_SM103_OPROJ_POLICY", None)
        environment.pop("FUSE_SM103_OPROJ_COMM_LAYOUT", None)
        environment.pop("FUSE_TEST_FAIL_SECOND_PAYLOAD", None)
        environment.pop("FUSE_TEST_FAIL_SELF_TEST", None)
        environment.pop("FUSE_TEST_FAIL_CALIBRATION_COMPONENT", None)
        environment.pop("FUSE_TEST_FAIL_GRAPH_POST", None)
        if fail_second_payload:
            environment["FUSE_TEST_FAIL_SECOND_PAYLOAD"] = "1"
        if fail_self_test:
            environment["FUSE_TEST_FAIL_SELF_TEST"] = "1"
        if fail_calibration_component is not None:
            environment["FUSE_TEST_FAIL_CALIBRATION_COMPONENT"] = fail_calibration_component
        if fail_graph_post:
            environment["FUSE_TEST_FAIL_GRAPH_POST"] = "1"
        return subprocess.run([str(self.graph_flow_probe if graph else self.flow_probe), *arguments], env=environment,
                              text=True, capture_output=True, timeout=10)

    def test_graph_is_explicit_and_requires_the_mpi_target(self):
        self.query("--launch", "eager")
        for value in ("graph", "invalid", "Graph", ""):
            self.assert_rejected("--launch", value)
        self.assert_rejected("--launch")

    def test_graph_main_binds_every_candidate_component_payload_and_checks_actual_last_sample(self):
        result = self.run_flow("--calibrate", "--comm-sm-list", "8,16", graph=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        # Four physical candidates, three components, two payloads. Each
        # gen0 benchmark must validate before any next epoch/poison/prepare.
        self.assertEqual(result.stdout.count("mock,graph_bind,"), 24)
        self.assertEqual(result.stdout.count("mock,graph_prepare,"), 24)
        self.assertEqual(result.stdout.count("candidate_verified,"), 12)
        self.assertIn("collector=mpi_graph_rank_events_v1", result.stdout)
        for index, line in enumerate(lines):
            if not line.startswith("mock,benchmark,"):
                continue
            expected = "mock,validate" + line.removeprefix("mock,benchmark") + ",validation_phase=post"
            position = lines.index(expected, index + 1)
            between = lines[index + 1:position]
            self.assertFalse(any(value.startswith(("mock,epoch,", "mock,poison,",
                "mock,prepare_component,", "mock,input,", "mock,select,")) for value in between))
        accepted = [line for line in lines if line.startswith("candidate_verified,")]
        self.assertTrue(all("graph_epoch_mode=recapture_update_v1" in line for line in accepted))

    def test_graph_last_sample_failure_cannot_be_hidden_by_a_later_launch_or_payload(self):
        result = self.run_flow("--calibrate", graph=True, fail_graph_post=True)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("injected Graph last-output failure", result.stderr)
        self.assertEqual(result.stdout.count("mock,epoch,"), 4)
        self.assertNotIn("mock,prepare_component,", result.stdout)
        self.assertNotIn("mock,input,1", result.stdout)
        self.assertNotIn("candidate_verified,", result.stdout)
        self.assertNotIn("PASS:", result.stdout)

    def test_real_main_reuses_each_generation_and_accepts_only_after_all_checks(self):
        result = self.run_flow("--comm-sm-list", "8,16", "--qkv-policy-list", "m128n64,m128n256",
                               "--oproj-policy-list", "m128n256,auto")
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertEqual(lines.count("mock,setup"), 1)
        self.assertIn("mock,setup_policy,qkv=m128n64,oproj=m128n256", lines)
        self.assertEqual(lines.count("mock,cleanup"), 1)
        self.assertEqual([line for line in lines if line.startswith("mock,input,")],
                         ["mock,input,0", "mock,input,1"])
        self.assertEqual([line for line in lines if line.startswith("mock,reference,")],
                         ["mock,reference,0", "mock,reference,1"])
        checks = [line for line in lines if line.startswith("mock,validate,")]
        self.assertEqual(len(checks), 16)
        for index, line in enumerate(checks):
            self.assertIn(f",candidate={index % 8 + 1},", line)
            self.assertIn(f",generation={index // 8}", line)
        self.assertEqual(sum(line.startswith("mock,poison,") for line in lines), 16)
        summaries = [line for line in lines if line.startswith("mock,benchmark,")]
        expected_tiles = [tile for tile in ("m128n64", "m128n256", "m128n256", "m128n128")
                          for _ in range(2)]
        self.assertEqual(len(summaries), 8)
        for index, (line, tile) in enumerate(zip(summaries, expected_tiles)):
            self.assertIn(f",candidate={index + 1},", line)
            self.assertIn(f",comm_sm={(8, 16)[index % 2]},", line)
            self.assertIn(f",tile={tile},generation=0", line)
        accepted = [index for index, line in enumerate(lines) if line.startswith("candidate_verified,")]
        self.assertEqual(len(accepted), 8)
        self.assertTrue(all(",tile=" in lines[index] for index in accepted))
        self.assertGreater(min(accepted), lines.index("mock,cleanup"))
        self.assertTrue(lines[-1].startswith("PASS:"))

    def test_real_main_second_payload_failure_never_accepts_candidates(self):
        result = self.run_flow("--comm-sm-list", "8,16", fail_second_payload=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("injected second-payload failure", result.stderr)
        self.assertIn("mock,benchmark,", result.stdout)
        self.assertIn("status=failed", result.stdout)
        self.assertNotIn("candidate_verified,", result.stdout)
        self.assertNotIn("PASS:", result.stdout)

    def test_real_main_single_configuration_keeps_both_profile_directions(self):
        result = self.run_flow("--profile", "--qkv-policy-list", "m128n64",
                               "--oproj-policy-list", "m128n256", "--comm-sm", "12")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("mock,setup_policy,qkv=m128n64,oproj=m128n256", result.stdout)
        self.assertIn("mock,profile,GEMM_A2A,tile=m128n64", result.stdout)
        self.assertIn("mock,profile,A2A_GEMM,tile=m128n256", result.stdout)
        self.assertEqual(result.stdout.count("mock,select,"), 6)
        self.assertEqual(result.stdout.count("candidate_verified,"), 2)

    def test_real_main_self_test_is_once_per_direction_and_never_a_performance_result(self):
        result = self.run_flow("--validation-self-test", "--comm-sm-list", "8,16",
                               "--qkv-policy-list", "auto,m128n64")
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self_tests = [line for line in lines if line.startswith("mock,self_test,")]
        self.assertEqual(len(self_tests), 2)
        self.assertTrue(self_tests[0].startswith("mock,self_test,GEMM_A2A,"))
        self.assertTrue(self_tests[1].startswith("mock,self_test,A2A_GEMM,"))
        self.assertEqual(sum(line.startswith("mock,validate,") for line in lines), 12)
        self.assertNotIn("mock,benchmark,", result.stdout)
        accepted = [line for line in lines if line.startswith("candidate_verified,")]
        self.assertEqual(len(accepted), 6)
        self.assertTrue(all("performance_accepted=0" in line for line in accepted))

    def test_real_main_failed_checker_self_test_prevents_acceptance(self):
        result = self.run_flow("--validation-self-test", fail_self_test=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("injected checker self-test failure", result.stderr)
        self.assertNotIn("mock,benchmark,", result.stdout)
        self.assertNotIn("candidate_verified,", result.stdout)
        self.assertNotIn("PASS:", result.stdout)

    def test_real_main_cpu_oracle_keeps_regular_measurement(self):
        result = self.run_flow("--cpu-oracle")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("mock,benchmark,"), 2)
        self.assertNotIn("mock,self_test,", result.stdout)
        accepted = [line for line in result.stdout.splitlines() if line.startswith("candidate_verified,")]
        self.assertEqual(len(accepted), 2)
        self.assertTrue(all("performance_accepted=1" in line for line in accepted))

    def test_real_main_preserves_gpu_generator_through_both_payload_generations(self):
        result = self.run_flow("--input-generator", "gpu_philox")
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertIn(",input_generator=gpu_philox", lines[0])
        self.assertIn("mock,setup_generator,gpu_philox", lines)
        for stage in ("input", "reference"):
            actual = [line for line in lines if line.startswith(f"mock,{stage}_generator,")]
            self.assertEqual(actual, [f"mock,{stage}_generator,0,gpu_philox",
                                      f"mock,{stage}_generator,1,gpu_philox"])
        self.assertEqual(result.stdout.count("mock,benchmark,"), 2)
        self.assertEqual(result.stdout.count("candidate_verified,"), 2)

    def test_real_main_uses_one_persistent_launch_team_and_joins_before_cleanup(self):
        for mode in ("sequential", "per_gpu_thread"):
            with self.subTest(mode=mode):
                result = self.run_flow("--host-launch", mode, "--profile",
                                       "--input-generator", "gpu_philox")
                self.assertEqual(result.returncode, 0, result.stderr)
                lines = result.stdout.splitlines()
                self.assertIn(f",host_launch={mode}", lines[0])
                self.assertEqual(lines.count(f"mock,launch_team,{mode}"), 1)
                self.assertIn(f"mock,launch_workers_joined,{4 if mode == 'per_gpu_thread' else 0}",
                              lines)
                self.assertEqual(result.stdout.count("mock,profile,"), 2)
                self.assertEqual(result.stdout.count("candidate_verified,"), 2)

    def test_real_main_parallel_failure_still_exits_without_acceptance(self):
        result = self.run_flow("--host-launch", "per_gpu_thread", fail_second_payload=True)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("injected second-payload failure", result.stderr)
        self.assertNotIn("mock,cleanup", result.stdout)
        self.assertNotIn("candidate_verified,", result.stdout)
        self.assertNotIn("PASS:", result.stdout)

    def test_real_main_calibrates_each_component_with_independent_epochs_and_scoped_acceptance(self):
        result = self.run_flow("--calibrate", "--cpu-oracle", "--host-launch", "per_gpu_thread",
                               "--comm-sm-list", "8,16", "--qkv-policy-list", "m128n64,m128n256")
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()

        def records(prefix):
            return [dict(field.split("=", 1) for field in line.split(",") if "=" in field)
                    for line in lines if line.startswith(prefix)]

        checks = records("mock,component_validate,")
        expected = [(candidate, generation, component, phase)
                    for generation in (0, 1) for candidate in range(1, 7)
                    for component, phase in (
                        (("fused", None), ("compute_reference", "pre"),
                         ("compute_reference", "post"), ("copy_reference", "pre"),
                         ("copy_reference", "post")) if generation == 0 else
                        (("fused", None), ("compute_reference", "pre"), ("copy_reference", "pre")))]
        self.assertEqual([(int(row["candidate"]), int(row["generation"]),
                           row["selected_component"], row.get("validation_phase")) for row in checks], expected)
        for row in checks:
            self.assertEqual(row["selected_component"], row["component"])
        samples = records("mock,component_benchmark,")
        self.assertEqual([(int(row["candidate"]), row["selected_component"], row["generation"])
                          for row in samples],
                         [(candidate, component, "0") for candidate in range(1, 7)
                          for component in ("fused", "compute_reference", "copy_reference")])
        # CPU benchmark stand-in advances three epochs. Post-sample validation
        # must not launch again; every reference sequence is 1,2,3,4 or only 1.
        epochs = records("mock,component_epoch,")
        for generation in (0, 1):
            for component in ("compute_reference", "copy_reference"):
                actual = [int(row["epoch"]) for row in epochs
                          if int(row["generation"]) == generation and row["component"] == component]
                self.assertEqual(actual, ([1, 2, 3, 4] if generation == 0 else [1]) * 6)
        self.assertEqual(len(records("mock,calibration_reset,")), 12)
        self.assertEqual(len(records("mock,prepare_component,")), 24)
        accepted = records("candidate_verified,")
        self.assertEqual(len(accepted), 18)
        expected_scopes = {"fused": ("1", "1"), "compute_reference": ("1", "0"),
                           "copy_reference": ("0", "1")}
        for row in accepted:
            self.assertEqual((row["full_numeric"], row["full_route"]), expected_scopes[row["component"]])
            self.assertEqual(row["payload_generations"], "2")
            self.assertEqual(row["performance_accepted"], "1")
        first_acceptance = next(index for index, line in enumerate(lines)
                                if line.startswith("candidate_verified,"))
        self.assertGreater(first_acceptance, lines.index("mock,cleanup"))
        self.assertEqual(result.stdout.count("mock,input,"), 2)
        self.assertEqual(result.stdout.count("mock,reference,"), 2)

    def test_real_main_reference_post_sample_failure_prevents_all_acceptance(self):
        for component in ("compute_reference", "copy_reference"):
            with self.subTest(component=component):
                result = self.run_flow("--calibrate", fail_calibration_component=component)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("injected calibration post-sample failure", result.stderr)
                self.assertIn(f",component={component},validation_phase=post", result.stdout)
                self.assertIn(f",selected_component={component}", result.stdout)
                self.assertNotIn("candidate_verified,", result.stdout)
                self.assertNotIn("PASS:", result.stdout)

    def test_compute_only_never_launches_fused_or_copy_components(self):
        result = self.run_flow('--calibrate', '--compute-only', '--fused-direction', 'oproj')
        self.assertEqual(result.returncode, 0, result.stderr)
        epochs = [line for line in result.stdout.splitlines() if line.startswith('mock,component_epoch,')]
        self.assertTrue(epochs)
        self.assertTrue(all(',component=compute_reference,' in line for line in epochs))
        accepted = [line for line in result.stdout.splitlines() if line.startswith('candidate_verified,')]
        self.assertTrue(accepted)
        self.assertTrue(all(',component=compute_reference,' in line for line in accepted))


class NoResidualHostContracts(unittest.TestCase):
    """Execute the shared real argument builder, not a numerical GEMM model."""

    @classmethod
    def setUpClass(cls):
        gemm = (ROOT / "csrc/operators/sm103/detail/gemm.cuh").read_text()
        helpers = gemm[gemm.index("__host__ __device__ constexpr int64_t a_row_stride("):
                       gemm.index("// Backward dgrad reads the stored forward weight")]
        launch = (ROOT / "csrc/operators/sm103/detail/launch.cuh").read_text()
        begin = launch.index("template <class Kernel, class Input = Bf16>\ntypename Kernel::Arguments gemm_arguments(")
        # Extract this function only, not unrelated helpers inserted after it.
        arguments = launch[begin:launch.index("\n}\n", begin) + 3]
        source = r"""
#include "fuse/layout/gemm.h"
#include <cstdint>
#include <stdexcept>
#include <tuple>
#define __host__
#define __device__
namespace cute {
struct _1 {};
template <class... T> auto make_shape(T... values) { return std::make_tuple(values...); }
template <class... T> auto make_stride(T... values) { return std::make_tuple(values...); }
}
namespace cutlass::gemm { enum class GemmUniversalMode { kGemm }; }
namespace fuse {
using Bf16 = uint16_t;
constexpr int kAlignment = 8;
namespace detail {
struct PersistentTileSchedulerSm100Monolithic { enum RasterOrderOptions { AlongM, AlongN }; };
}
""" + helpers + r"""
struct DeviceInfo { int device = 3, sm_count = 148; };
struct ProbeGemm {
  struct Arguments {
    cutlass::gemm::GemmUniversalMode mode;
    std::tuple<int, int, int, int> problem_shape;
    using Stride = std::tuple<int64_t, cute::_1, int64_t>;
    struct { const Bf16* ptr_A; const Bf16* ptr_B; Stride dA, dB; } mainloop;
    struct {
      struct { float alpha, beta; } thread;
      const void* ptr_C;
      Bf16* ptr_D;
      Stride dC, dD;
    } epilogue;
    struct { int device_id, sm_count; } hw_info;
    struct {
      int block_offset, max_swizzle_size;
      detail::PersistentTileSchedulerSm100Monolithic::RasterOrderOptions raster_order;
    } scheduler;
  };
};
""" + arguments + r"""
}
int main() {
  using namespace fuse;
  using Raster = detail::PersistentTileSchedulerSm100Monolithic;
  GemmProblem p;
  p.m = 128; p.n = 256; p.k = 2048;
  alignas(16) Bf16 a[8]{}, b[8]{}, d[8]{};
  DeviceInfo info;
  for (bool padded : {false, true}) {
    if (padded) { p.stride_a.row = 2072; p.stride_b.row = 2080; p.stride_d.row = 264; }
    for (auto fallback : {GemmRaster::kAlongM, GemmRaster::kAlongN}) {
      for (auto requested : {GemmRaster::kHeuristic, GemmRaster::kAlongM, GemmRaster::kAlongN}) {
        for (int swizzle : {1, 2, 4, 8}) {
          p.raster = requested;
          p.max_swizzle_size = swizzle;
          const auto effective = requested == GemmRaster::kHeuristic ? fallback : requested;
          for (float alpha : {-2.0f, 0.0f, .25f, 1.0f}) {
            const auto args = gemm_arguments<ProbeGemm>(p, a, b, d, alpha, 12, info, fallback);
            if (args.mode != cutlass::gemm::GemmUniversalMode::kGemm ||
                args.problem_shape != std::make_tuple(128, 256, 2048, 1) ||
                args.mainloop.ptr_A != a || args.mainloop.ptr_B != b || args.epilogue.ptr_D != d ||
                args.epilogue.thread.alpha != alpha || args.epilogue.thread.beta != 0.0f ||
                args.epilogue.ptr_C != nullptr ||
                std::get<0>(args.mainloop.dA) != (padded ? 2072 : 2048) ||
                std::get<0>(args.mainloop.dB) != (padded ? 2080 : 2048) ||
                std::get<0>(args.epilogue.dD) != (padded ? 264 : 256) ||
                args.hw_info.device_id != 3 || args.hw_info.sm_count != 136 ||
                args.scheduler.block_offset != 12 || args.scheduler.max_swizzle_size != swizzle ||
                args.scheduler.raster_order != (effective == GemmRaster::kAlongM ? Raster::AlongM : Raster::AlongN)) {
              throw std::runtime_error("no-residual argument contract");
            }
          }
        }
      }
    }
  }
}
"""
        cls.probe = compile_host_probe(cls, source, "no-residual-arguments", "-Wall", "-Wextra", "-Werror")

    def test_real_builder_keeps_beta_zero_null_c_alpha_and_geometry(self):
        subprocess.run([str(self.probe)], check=True, timeout=10)
        launch = (ROOT / "csrc/operators/sm103/detail/launch.cuh").read_text()
        reference = (ROOT / "csrc/operators/sm103/api/reference.cuh").read_text()
        self.assertEqual(launch.count("gemm_arguments<Gemm>("), 2)
        self.assertIn("auto args = input.template arguments<Gemm>(params, info);", launch)
        self.assertEqual(reference.count("auto args = gemm_arguments<Gemm>("), 1)


class CtaTimelineHostContracts(unittest.TestCase):
    """Real recorder/profile control flow, with explicit CPU-only substitutes."""

    @classmethod
    def setUpClass(cls):
        timeline = (ROOT / "include/fuse/profiling/timeline.cuh").read_text()
        structs = timeline[timeline.index("struct A2AGemmCtaTimeline {"):
                           timeline.index("}  // namespace fuse")]
        header = r"""
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>
namespace fuse { constexpr int kMaxWorldSize = 8;
""" + structs + "}\n"
        pipeline = (ROOT / "csrc/operators/sm103/detail/cutlass_pipeline.cuh").read_text()
        begin = pipeline.index(" private:\n", pipeline.index("struct A2ALhsReadyMainloop")) + len(" private:\n")
        recorder = pipeline[begin:pipeline.index("\n  const Params* params_;", begin)]
        recorder_probe = header + r"""
#define CUTLASS_DEVICE
namespace cute { using std::get; }
struct CtaIndex { int x = 3; };
CtaIndex blockIdx;
uint64_t clock_reads = 0;
int cta_atomics = 0, peer_atomics = 0;
void* active_address = nullptr;
uint64_t read_global_timer() { return ++clock_reads; }
unsigned long long atomicCAS(unsigned long long* pointer, unsigned long long old, unsigned long long value) {
  if (pointer == active_address) ++cta_atomics; else ++peer_atomics;
  const auto previous = *pointer;
  if (previous == old) *pointer = value;
  return previous;
}
struct Params {
  fuse::A2AGemmCtaTimeline* timeline;
  int timeline_capacity;
  fuse::A2AGemmPeerTimeline* peer_timeline;
  int peer_timeline_capacity;
  int m_tiles = 2, n_tiles = 3;
};
template <bool Instrumented> struct Probe {
  const Params* params_;
""" + recorder + r"""
};
void require(bool condition) { if (!condition) throw std::runtime_error("recorder assertion"); }
int main(int argc, char** argv) {
#if FUSE_ENABLE_PROFILING
  const bool full = argc > 1 && std::string(argv[1]) == "full";
  fuse::A2AGemmCtaTimeline timeline[4]{};
  fuse::A2AGemmPeerTimeline peers[6]{};
  Params params{timeline, 4, full ? peers : nullptr, full ? 6 : 0};
  active_address = &timeline[3].active_start;
  Probe<true> probe{&params};
  for (auto tile : {std::make_tuple(-1, 0, 0, 0), std::make_tuple(2, 0, 0, 0),
                    std::make_tuple(0, 3, 0, 0), std::make_tuple(0, 0, 0, 1)}) {
    probe.record_peer_acquire(tile, 0);
  }
  require(!clock_reads && !cta_atomics && !peer_atomics);
  for (int m = 0; m < 2; ++m) for (int n = 0; n < 3; ++n) for (int peer = 0; peer < 4; ++peer) {
    const auto tile = std::make_tuple(m, n, 0, 0);
    probe.record_peer_acquire(tile, peer);
    const auto first = peers[m * 3 + n].acquire[peer];
    probe.record_peer_acquire(tile, peer); // Prologue/remainder or cached observation.
    if (full) require(first != 0 && peers[m * 3 + n].acquire[peer] == first);
  }
  require(cta_atomics == 1 && timeline[3].active_start == 1);
  require(clock_reads == (full ? 48u : 1u) && peer_atomics == (full ? 48 : 0));
  if (full) for (int i = 0; i < 6; ++i) {
    require(peers[i].valid && peers[i].m_tile == i / 3 && peers[i].n_tile == i % 3);
  }
  const auto clocks_before = clock_reads;
  Probe<false> production{&params};
  production.record_peer_acquire(std::make_tuple(0, 0, 0, 0), 0);
  require(clock_reads == clocks_before && cta_atomics == 1);
  timeline[3].active_start = 0;
  Probe<true> next_launch{&params};
  next_launch.record_peer_acquire(std::make_tuple(0, 0, 0, 0), 0);
  require(cta_atomics == 2 && timeline[3].active_start == clocks_before + 1);
#else
  (void)argc; (void)argv;
  static_assert(sizeof(Probe<false>) == sizeof(const Params*));
#endif
}
"""
        cls.recorders = {profiling: compile_host_probe(cls, recorder_probe, f"cta-recorder-{profiling}",
            f"-DFUSE_ENABLE_PROFILING={profiling}", "-Wall", "-Wextra", "-Werror")
            for profiling in (0, 1)}
        harness = (ROOT / "benchmarks/sm103/fused_bf16.cu").read_text()
        begin = harness.index("    if (options.profile", harness.index("create_runtimes("))
        allocation = harness[begin:harness.index("\n#endif\n  }\n  finish_all", begin)]
        begin = harness.index("void profile(")
        profile = harness[begin:harness.index("\n}\n", begin) + len("\n}\n")]
        timing = harness[harness.index("struct HostLaunchTiming {"):
                         harness.index("\nstruct RankLaunch {")]
        route_header = (ROOT / "include/fuse/profiling/qkv_route.cuh").read_text()
        route_record = route_header[route_header.index("struct QkvRouteTimeline {"):
                                    route_header.index("\ncudaError_t")]
        pipeline_header = (ROOT / "include/fuse/profiling/sm103/oproj.cuh").read_text()
        pipeline_records = pipeline_header[pipeline_header.index("struct OprojReadyRecord {"):
                                           pipeline_header.index("// Bind only on the host")]
        profile_probe = (header + '\nnamespace fuse {\n' + route_record + '}\n' +
                         '\nnamespace fuse::detail {\n' + pipeline_records + '}\n') + r"""
constexpr int kWarmup = 10;
enum class Direction { kQkv, kOproj };
const char* direction_name(Direction d) { return d == Direction::kQkv ? "GEMM_A2A" : "A2A_GEMM"; }
struct Options {
  bool profile = true;
  bool mxfp8_service_probe = false;
  bool qkv_epilogue_probe = false;
  bool oproj_pipeline_probe = false;
  bool oproj_gap_probe = false;
  int max_swizzle_size = 1;
  int world = 4, comm_sm = 1, seq_local = 128, hidden = 128;
  int q_heads = 8, kv_heads = 4;
  std::string profile_direction = "oproj"; // This probe exercises CTA/peer records.
  unsigned timeout_seconds = 0;
  std::string host_launch = "sequential", profile_detail = "full";
  int projection_width() const { return 384; }
  int q_width() const { return 512; }
};
namespace fuse::detail { struct QkvEpilogueRecord {}; }
struct RankRuntime {
  int device = 0, sm_count = 4, stream = 0, peer_capacity = 0;
  struct { int gemm = 0; struct { bool defer_v_a2a = false; } route; } qkv;
  fuse::QkvRouteTimeline* qkv_route_timeline = nullptr;
  int qkv_route_capacity = 0;
  fuse::A2AGemmCtaTimeline* timeline = nullptr;
  fuse::A2AGemmPeerTimeline* peer_timeline = nullptr;
  fuse::detail::QkvEpilogueRecord* qkv_epilogue = nullptr;
  fuse::detail::OprojPipelineView oproj_pipeline{};
};
namespace fuse {
struct Traits { int block_m = 128, block_n = 128, block_k = 64; };
Traits cutlass_kernel_traits() { return {}; }
template <class... T> Traits qkv_cutlass_kernel_traits(T...) { return {}; }
template <class T> int query_gemm_a2a_route_timeline_capacity(const T&, int* capacity) { *capacity = 64; return 0; }
int query_a2a_gemm_role_resources(A2AGemmRoleResources* resources) {
  resources->threads_per_cta = 256; resources->registers_per_thread = 64;
  resources->telemetry_registers_per_thread = 64; resources->dynamic_smem_bytes = 200000;
  return 0;
}
}
int ceil_div(int a, int b) { return a / b + (a % b != 0); }
int checked_product(int a, int b) { return a * b; }
int peer_allocations = 0, peer_clears = 0, peer_downloads = 0, checks = 0, host_stages = 0;
int epoch_calls = 0;
uint32_t last_epoch = 17;
std::string failure;
std::vector<void*> peer_buffers;
template <class T> T* allocate(RankRuntime&, int count) {
  auto* pointer = new T[count]{};
  if constexpr (std::is_same_v<T, fuse::A2AGemmPeerTimeline>) {
    ++peer_allocations; peer_buffers.push_back(pointer);
  }
  return pointer;
}
#define CUDA_CHECK(expression) do { if (expression) throw std::runtime_error("CUDA stand-in failure"); } while (0)
void allocate_profile(RankRuntime& runtime, const Options& options) {
  const int rank = runtime.device;
""" + allocation + r"""
}
unsigned alarm(unsigned) { return 0; }
int cudaSetDevice(int) { return 0; }
int cudaMemsetAsync(void* pointer, int value, size_t bytes, int) {
  if (!pointer) throw std::runtime_error("null diagnostic clear");
  if (std::find(peer_buffers.begin(), peer_buffers.end(), pointer) != peer_buffers.end()) ++peer_clears;
  std::memset(pointer, value, bytes); return 0;
}
void finish_all(std::vector<RankRuntime>&) {}
template <class T> std::vector<T> download(const T* pointer, int count) {
  if (!pointer) throw std::runtime_error("null diagnostic download");
  if constexpr (std::is_same_v<T, fuse::A2AGemmPeerTimeline>) ++peer_downloads;
  return {pointer, pointer + count};
}
""" + timing + r"""
std::vector<float> run_epoch(std::vector<RankRuntime>& ranks, const Options& options,
    Direction direction, uint32_t epoch, bool instrumented = false,
    std::vector<double>* launch = nullptr, HostLaunchTiming* timing = nullptr) {
  if (epoch != ++last_epoch) throw std::runtime_error("discontinuous diagnostic epoch");
  ++epoch_calls;
  if (launch) launch->assign(options.world, 1.0);
  if (timing) {
    timing->api_begin_us.assign(options.world, 1.0);
    timing->api_end_us.assign(options.world, 2.0);
  }
  if (instrumented) for (auto& rank : ranks) {
    for (int cta = 0; cta < (direction == Direction::kQkv ? 4 : 2); ++cta) {
      auto& event = rank.timeline[cta];
      event.start = 100; event.end = 200; event.active_start = 110;
      event.role_done = 150; event.grid_sync_done = 160; event.fence_done = 170; event.publish_done = 180;
      for (auto& stamp : event.source_ready) stamp = 190;
    }
    if (failure == "cta") rank.timeline[1].active_start = 0;
    for (int i = 0; i < rank.peer_capacity; ++i) {
      auto& event = rank.peer_timeline[i];
      event.comm_valid = 1; event.task_begin = 100; event.input_ready = 110;
      event.publish_issue = 120; event.release = 130; event.valid = 1;
      for (auto& stamp : event.acquire) stamp = 140;
    }
  }
  return std::vector<float>(options.world, .5f);
}
void validate(std::vector<RankRuntime>&, const Options&, Direction, const char* context) {
  ++checks;
  if (failure == context) throw std::runtime_error("injected full-validation failure");
}
void profile_host_stages(std::vector<RankRuntime>&, const Options&, Direction, uint32_t& epoch) {
  if (checks != 1 + 2 * host_stages) throw std::runtime_error("instrumented output overwritten before validation");
  ++host_stages; epoch += 60; last_epoch += 60;
}
""" + profile + r"""
int main(int argc, char** argv) {
  Options options;
  if (argc > 1) options.profile_detail = argv[1];
  if (argc > 2) failure = argv[2];
  std::vector<RankRuntime> ranks(options.world);
  try {
    for (auto& rank : ranks) allocate_profile(rank, options);
    uint32_t epoch = 17;
    profile(ranks, options, Direction::kQkv, epoch);
    profile(ranks, options, Direction::kOproj, epoch);
    const bool full = options.profile_detail == "full";
    if (peer_allocations != (full ? 4 : 0) || peer_clears != (full ? 88 : 0) ||
        peer_downloads != (full ? 4 : 0) || checks != 4 || host_stages != 2 ||
        epoch_calls != 24 || epoch != 161) return 3;
    for (auto& rank : ranks) { delete[] rank.timeline; delete[] rank.peer_timeline; }
    std::cout << "PASS profile " << options.profile_detail << '\n';
    return 0;
  } catch (const std::exception& error) {
    for (auto& rank : ranks) { delete[] rank.timeline; delete[] rank.peer_timeline; }
    std::cerr << error.what() << '\n';
    return 1;
  }
}
"""
        cls.profile_probe = compile_host_probe(cls, profile_probe, "cta-profile-flow",
            "-DFUSE_ENABLE_PROFILING=1", "-DFUSE_BENCH_MXFP8=0", "-Wall", "-Wextra", "-Werror")

    def test_first_cta_stamp_is_latched_but_full_peer_observations_remain(self):
        for profiling, detail in product((0, 1), ("full", "cta")):
            subprocess.run([str(self.recorders[profiling]), detail], check=True, timeout=10)
        source = (ROOT / "csrc/operators/sm103/detail/cutlass_pipeline.cuh").read_text()
        call = source.index("record_peer_acquire(tile_coord, peer);")
        self.assertIn("if (threadIdx.x % 32 == 0)", source[call - 60:call])
        # The readiness-cache block ends before telemetry. Cache hits still
        # reach the real recorder exercised above; no warp ordering is emulated.
        cache_end = source.index("acquired_peer_ = peer;")
        self.assertIn("}\n\n#if FUSE_ENABLE_PROFILING", source[cache_end:call])

    def test_cta_mode_omits_all_peer_work_and_preserves_complete_profile_flow(self):
        for detail in ("full", "cta"):
            result = subprocess.run([str(self.profile_probe), detail], text=True,
                                    capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count("profile_cta,"), 24)
            self.assertEqual(result.stdout.count("profile_host,"), 8)
            self.assertEqual(result.stdout.count("profile_peer,"), 16 if detail == "full" else 0)
            for line in result.stdout.splitlines():
                if line.startswith("profile_"):
                    self.assertIn(f",profile_detail={detail}", line)

    def test_cta_mode_still_rejects_missing_active_stamp_and_both_validation_failures(self):
        for failure in ("cta", ",profile_phase=instrumented", ",profile_phase=host_stages"):
            result = subprocess.run([str(self.profile_probe), "cta", failure],
                                    text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertNotIn("PASS profile", result.stdout)
            self.assertIn("missing compute active_start" if failure == "cta" else
                          "injected full-validation failure", result.stderr)


class HostStageContracts(unittest.TestCase):
    """Run real private host timers/TLS and the diagnostic collector, not CUDA."""

    @classmethod
    def setUpClass(cls):
        cls.header_dir = ROOT / "include"
        entry = (ROOT / "csrc/operators/sm103/entry.cu").read_text()
        start = entry.index("namespace fuse::detail {\nthread_local HostLaunchRecord*")
        definition = entry[start:entry.index("#endif", start)]
        support = ('#include "fuse/profiling/sm103/host.cuh"\n'
                   'namespace fuse::detail { struct OprojPipelineView; }\n') + definition + r"""
void record_stages(int status) {
  FUSE_SM103_HOST_BEGIN();
  FUSE_SM103_HOST_MARK(kCommunicationPrepare);
  {
    FUSE_SM103_HOST_DESCRIPTOR_SCOPE(local, 0);
    FUSE_SM103_HOST_DESCRIPTOR_ATTEMPT(local);
  }
  {
    FUSE_SM103_HOST_DESCRIPTOR_SCOPE(peers, 1);
    for (int peer = 0; peer < 24; ++peer) FUSE_SM103_HOST_DESCRIPTOR_ATTEMPT(peers);
  }
  if (status) { host_launch_scope.finish(status); return; }
  FUSE_SM103_HOST_MARK(kArguments);
  FUSE_SM103_HOST_MARK(kImplementWorkspace);
  FUSE_SM103_HOST_MARK(kLowerParameters);
  FUSE_SM103_HOST_MARK(kLaunchSetup);
  FUSE_SM103_HOST_MARK(kCudaEnqueue);
  FUSE_SM103_HOST_MARK(kEnd);
  host_launch_scope.finish(0);
}
"""
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        if not compiler or shutil.which(compiler[0]) is None:
            raise unittest.SkipTest("a host C++ compiler is required")
        directory = tempfile.TemporaryDirectory(prefix="fuse-host-stage-test-")
        cls.addClassCleanup(directory.cleanup)
        cls.support_object = Path(directory.name) / "stage-support.o"
        result = subprocess.run(
            [*compiler, "-std=c++17", "-O2", "-DFUSE_ENABLE_PROFILING=1", "-I",
             str(cls.header_dir), "-x", "c++", "-c", "-", "-o", str(cls.support_object)],
            input=support, text=True, capture_output=True, timeout=60)
        if result.returncode:
            raise AssertionError(result.stderr)
        source = r"""
#include "fuse/profiling/sm103/host.cuh"
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
void record_stages(int);
void require(bool condition) { if (!condition) throw std::runtime_error("host-stage assertion"); }
int main(int argc, char** argv) {
  using namespace fuse::detail;
  const std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "cross_tu") {
    HostLaunchRecord record;
    { HostLaunchBinding binding(&record); record_stages(0); }
    require(host_launch_sink == nullptr && record.status == 0 && record.stage_mask == 255);
    require(!record.protocol_error && record.descriptor_count[0] == 1 && record.descriptor_count[1] == 24);
    for (int i = 0; i < 7; ++i) require(record.stamp_ns[i] <= record.stamp_ns[i + 1]);
    require(record.api_return_ns >= record.stamp_ns[7]);
  } else if (mode == "disabled_sink") {
    record_stages(0);
    require(host_launch_sink == nullptr);
    HostLaunchRecord outer;
    HostLaunchBinding binding(&outer);
    { HostLaunchBinding disabled(nullptr); record_stages(0); }
    require(host_launch_sink == &outer && outer.stage_mask == 0 && outer.api_return_ns == 0);
  } else if (mode == "errors") {
    HostLaunchRecord record;
    { HostLaunchBinding binding(&record); record_stages(17); }
    require(record.status == 17 && record.stage_mask == 3 && !record.protocol_error);
    require(record.api_return_ns >= record.stamp_ns[1]);
    try {
      HostLaunchBinding binding(&record);
      HostLaunchScope scope;
      throw 19;
    } catch (int) {}
    require(host_launch_sink == nullptr && record.stage_mask == 1 && record.status == -1);
    { HostLaunchBinding binding(&record); HostLaunchScope scope;
      mark_host_launch_stage(HostLaunchStage::kArguments); }
    require(record.protocol_error);
  } else if (mode == "thread_isolation") {
    std::array<HostLaunchRecord, 8> records;
    std::array<std::thread, 8> workers;
    for (int rank = 0; rank < 8; ++rank) workers[rank] = std::thread([&, rank] {
      require(host_launch_sink == nullptr);
      for (int i = 0; i < 50; ++i) {
        HostLaunchBinding binding(&records[rank]);
        record_stages(rank);
      }
      require(host_launch_sink == nullptr);
    });
    for (auto& thread : workers) thread.join();
    for (int rank = 0; rank < 8; ++rank) {
      require(records[rank].status == rank && records[rank].descriptor_count[1] == 24);
      require(records[rank].stage_mask == (rank ? 3u : 255u));
    }
    require(host_launch_sink == nullptr);
  } else return 2;
  std::cout << "PASS " << mode << '\n';
}
"""
        cls.probe = compile_host_probe(cls, source, "host-stage-tls", "-pthread",
            "-DFUSE_ENABLE_PROFILING=1", "-I", str(cls.header_dir), str(cls.support_object))
        disabled = r"""
#include "fuse/profiling/sm103/host.cuh"
int operation() {
  FUSE_SM103_HOST_BEGIN();
  FUSE_SM103_HOST_MARK(intentionally_undefined_stage);
  FUSE_SM103_HOST_DESCRIPTOR_SCOPE(intentionally_undefined_name, undefined_group);
  FUSE_SM103_HOST_DESCRIPTOR_ATTEMPT(intentionally_undefined_name);
  FUSE_SM103_HOST_RETURN(0);
}
int main() { return operation(); }
"""
        cls.disabled_probe = compile_host_probe(cls, disabled, "host-stage-macro-off",
            "-DFUSE_ENABLE_PROFILING=0", "-I", str(cls.header_dir), "-Wall", "-Wextra", "-Werror")
        harness = (ROOT / "benchmarks/sm103/fused_bf16.cu").read_text()
        diagnostic = harness[harness.index("void profile_host_stages("):
                             harness.index("\nvoid profile_compute_gaps(")]
        timing = harness[harness.index("struct HostLaunchTiming {"):
                         harness.index("\nstruct RankLaunch {")]
        collector = r"""
#include "fuse/profiling/sm103/host.cuh"
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
void record_stages(int);
namespace fuse { constexpr int kMaxWorldSize = 8; }
constexpr int kWarmup = 10, kSamples = 50, cudaSuccess = 0;
enum class MeasurementComponent { kFused, kComputeReference };
enum class Direction { kQkv, kOproj };
const char* direction_name(Direction d) { return d == Direction::kQkv ? "GEMM_A2A" : "A2A_GEMM"; }
struct Options {
  bool profile = true;
  MeasurementComponent component = MeasurementComponent::kFused;
  int world = 8;
  std::string host_launch = "sequential";
  std::string profile_detail = "full";
};
struct RankRuntime {};
int warm_calls = 0, diagnostic_calls = 0;
bool incomplete = false;
""" + timing + r"""
std::vector<float> run_epoch(std::vector<RankRuntime>&, const Options& options, Direction,
    uint32_t, bool gpu_profile = false, std::vector<double>* launch = nullptr,
    HostLaunchTiming* dispatch = nullptr, fuse::detail::HostLaunchRecord* records = nullptr) {
  if (gpu_profile) throw std::runtime_error("host diagnostic enabled GPU traces");
  if (!records) { ++warm_calls; return {}; }
  ++diagnostic_calls;
  if (launch->size() != 8 || dispatch->api_begin_us.size() != 8 || dispatch->api_end_us.size() != 8) {
    throw std::runtime_error("records not preallocated");
  }
  for (int rank = 0; rank < options.world; ++rank) {
    fuse::detail::HostLaunchBinding binding(&records[rank]);
    records[rank].outer_begin_ns = fuse::detail::host_timestamp_ns();
    record_stages(0);
    records[rank].outer_end_ns = fuse::detail::host_timestamp_ns();
    if (incomplete && diagnostic_calls == 2 && rank == 3) records[rank].stage_mask = 127;
    (*launch)[rank] = (records[rank].outer_end_ns - records[rank].outer_begin_ns) * .001;
    dispatch->api_begin_us[rank] = rank * 10;
    dispatch->api_end_us[rank] = rank * 10 + (*launch)[rank];
  }
  dispatch->all_enqueued_us = 100;
  return std::vector<float>(options.world, .5f);
}
""" + diagnostic + r"""
int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  Options options;
  std::vector<RankRuntime> ranks(8);
  uint32_t epoch = 17;
  incomplete = mode == "incomplete";
  if (mode == "disabled") options.profile = false;
  if (mode == "component") options.component = MeasurementComponent::kComputeReference;
  try {
    profile_host_stages(ranks, options, Direction::kQkv, epoch);
    profile_host_stages(ranks, options, Direction::kOproj, epoch);
    if (epoch != 137 || warm_calls != 20 || diagnostic_calls != 100) return 4;
    std::cout << "PASS collector\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    if ((mode == "disabled" || mode == "component") && (warm_calls || diagnostic_calls)) return 5;
    return 1;
  }
}
"""
        cls.collector_probe = compile_host_probe(cls, collector, "host-stage-collector",
            "-DFUSE_ENABLE_PROFILING=1", "-I", str(cls.header_dir), str(cls.support_object))
        enqueue = harness[harness.index("struct HostLaunchTiming {"):
                          harness.index("\nvoid cublas_nt(")]
        launch_probe = r"""
#include "fuse/profiling/sm103/host.cuh"
#include "fused_launch.cuh"
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>
void record_stages(int);
enum class MeasurementComponent { kFused, kComputeReference, kCopyReference };
enum class Direction { kQkv, kOproj };
struct Options {
  unsigned timeout_seconds = 10;
  MeasurementComponent component = MeasurementComponent::kFused;
  int comm_sm = 24;
  std::string host_launch = "per_gpu_thread";
};
namespace fuse {
struct Params { uint32_t epoch = 0; void* ready = nullptr; void* peer_route_done_epoch[8]{}; };
namespace detail {
struct OprojPipelineView { void* tiles = nullptr; };
struct OprojPipelineBinding { explicit OprojPipelineBinding(const OprojPipelineView*) {} };
}
template <class... T> int launch_batched_cutlass_reference(T...) { return 0; }
template <class... T> int launch_gemm_a2a_copy_reference(T...) { return 0; }
template <class... T> int launch_a2a_gemm_cutlass_reference(T...) { return 0; }
template <class... T> int launch_a2a_gemm_copy_reference(T...) { return 0; }
template <class... T> int launch_gemm_a2a_role_telemetry(T...) { return 0; }
template <class... T> int launch_gemm_a2a_route_telemetry(T...) { return 0; }
namespace detail { template <class... T> int launch_qkv_epilogue_telemetry(T...) { return 0; } }
template <class... T> int launch_a2a_gemm_cutlass_role_telemetry(T...) { return 0; }
int launch_gemm_a2a_cutlass(Params, int) { record_stages(0); return 0; }
int launch_a2a_gemm_cutlass(Params, int) { record_stages(0); return 0; }
}
struct RankRuntime {
  int device = 0, stream = 0, start = 0, end = 0, sm_count = 148, peer_capacity = 0;
  void* timeline = nullptr;
  void* peer_timeline = nullptr;
  void* qkv_route_timeline = nullptr;
  int qkv_route_capacity = 0;
  void* qkv_epilogue = nullptr;
  void* calibration_qkv_ready = nullptr;
  void* calibration_oproj_ready = nullptr;
  void* calibration_route_done = nullptr;
  fuse::Params qkv, oproj;
  fuse::detail::OprojPipelineView oproj_pipeline{};
  fused_launch::Team* launch_team = nullptr;
};
int cudaSetDevice(int) { return 0; }
int cudaEventRecord(int, int) { return 0; }
int cudaEventElapsedTime(float* output, int, int) { *output = .5f; return 0; }
#define CUDA_CHECK(operation) do { if (operation) throw std::runtime_error("CUDA stand-in failed"); } while (0)
void wait_all(std::vector<RankRuntime>&) {}
void check_enqueue(const fused_launch::Result& result) { if (result.error) std::rethrow_exception(result.error); }
""" + enqueue + r"""
int main() {
  Options options;
  fused_launch::Team team(8);
  std::vector<RankRuntime> runtimes(8);
  for (auto& runtime : runtimes) runtime.launch_team = &team;
  for (const char* mode : {"sequential", "per_gpu_thread"}) {
    options.host_launch = mode;
    std::array<fuse::detail::HostLaunchRecord, 8> records;
    std::vector<double> launch;
    HostLaunchTiming timing;
    for (Direction direction : {Direction::kQkv, Direction::kOproj}) {
      run_epoch(runtimes, options, direction, 101, false, &launch, &timing, records.data());
      for (const auto& record : records) {
        if (record.stage_mask != 255 || record.status || record.protocol_error ||
            record.outer_begin_ns > record.stamp_ns[0] || record.outer_end_ns < record.api_return_ns) return 3;
      }
      run_epoch(runtimes, options, direction, 102);
    }
  }
  const auto result = team.dispatch([](void*, int) {
    if (fuse::detail::host_launch_sink) throw std::runtime_error("worker retained sink after return");
  }, nullptr);
  check_enqueue(result);
  team.stop();
  if (fuse::detail::host_launch_sink) return 4;
  ::alarm(0);
}
"""
        cls.enqueue_probe = compile_host_probe(cls, launch_probe, "host-stage-enqueue",
            "-DFUSE_ENABLE_PROFILING=1", "-pthread", "-I", str(cls.header_dir),
            "-I", str(ROOT / "benchmarks/sm103"), str(cls.support_object), "-Wall", "-Wextra", "-Werror")

    def test_single_sink_is_shared_across_real_translation_units(self):
        for mode in ("cross_tu", "disabled_sink", "errors", "thread_isolation"):
            with self.subTest(mode=mode):
                result = subprocess.run([str(self.probe), mode], text=True, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), "PASS " + mode)

    def test_macro_off_needs_no_sink_type_linkage_or_argument_symbols(self):
        subprocess.run([str(self.disabled_probe)], check=True, timeout=10)

    def test_profile_checks_instrumented_output_before_host_diagnostics_overwrite_it(self):
        source = (ROOT / "benchmarks/sm103/fused_bf16.cu").read_text()
        profile = source[source.index("void profile("):source.index("\nvoid destroy_runtimes(")]
        # Source-order regression, not a substitute for real GPU correctness:
        # the host-only collector cannot validate an instrumented GPU output.
        self.assertEqual(profile.count("validate(runtimes, options, direction,"), 2)
        before = profile.index('validate(runtimes, options, direction, ",profile_phase=instrumented");')
        diagnostics = profile.index("profile_host_stages(runtimes, options, direction, epoch);")
        after = profile.index('validate(runtimes, options, direction, ",profile_phase=host_stages");')
        self.assertLess(before, diagnostics)
        self.assertLess(diagnostics, after)
        self.assertIn('<< ",profile_schema=" << (options.profile ? "host_stages_v1" : "none")', source)

    def test_real_enqueue_and_epoch_bind_the_owning_thread_and_clear_after_diagnostics(self):
        subprocess.run([str(self.enqueue_probe)], check=True, timeout=15)

    def test_real_diagnostic_collects_10_plus_50_without_gpu_traces_and_keeps_epochs(self):
        result = subprocess.run([str(self.collector_probe), "normal"], text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = [line for line in result.stdout.splitlines() if line.startswith("host_stage,")]
        self.assertEqual(len(lines), 2 * 50 * 8)
        records = [dict(field.split("=", 1) for field in line.split(",")[2:]) for line in lines]
        for record in records:
            self.assertEqual(record["complete"], "1")
            self.assertEqual(record["performance_accepted"], "0")
            self.assertEqual(record["local_descriptor_count"], "1")
            self.assertEqual(record["peer_descriptor_count"], "24")
            self.assertEqual(record["profile_phase"], "host_stages")
            self.assertEqual(record["profile_schema"], "host_stages_v1")
        self.assertEqual(sorted({int(row["epoch"]) for row in records}),
                         list(range(28, 78)) + list(range(88, 138)))

    def test_real_collector_rejects_missing_stage_or_nonprofile_reference_mode(self):
        for mode in ("incomplete", "disabled", "component"):
            with self.subTest(mode=mode):
                result = subprocess.run([str(self.collector_probe), mode], text=True, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertTrue(result.stderr)
                if mode == "incomplete":
                    self.assertIn(",complete=0,", result.stdout)
                else:
                    self.assertNotIn("host_stage,", result.stdout)


class LaunchHostContracts(unittest.TestCase):
    """Execute the real persistent CPU team; no CUDA calls or GPU timing."""

    @classmethod
    def setUpClass(cls):
        source = r"""
#include "fused_launch.cuh"
#include <chrono>
#include <condition_variable>
#include <exception>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
void require_success(const fused_launch::Result& result) {
  if (result.error) std::rethrow_exception(result.error);
}
struct Identity {
  int generation = 0;
  std::vector<int> calls, observed;
  std::vector<std::thread::id> threads;
  explicit Identity(int ranks) : calls(ranks), observed(ranks), threads(ranks) {}
};
struct Gate {
  std::mutex mutex;
  std::condition_variable ready;
  int entered = 0;
  bool release = false;
  bool nonstandard = false;
  std::vector<int> values{0, 0};
};
void blocked_callback(void* pointer, int rank) {
  auto& gate = *static_cast<Gate*>(pointer);
  {
    std::unique_lock<std::mutex> lock(gate.mutex);
    ++gate.entered;
    gate.ready.notify_all();
    gate.ready.wait(lock, [&] { return gate.release; });
  }
  gate.values.at(rank) = 101 + rank;
}
void failing_callback(void* pointer, int rank) {
  auto& gate = *static_cast<Gate*>(pointer);
  if (rank == 1) {
    blocked_callback(pointer, rank);
    return;
  }
  std::unique_lock<std::mutex> lock(gate.mutex);
  gate.ready.wait(lock, [&] { return gate.entered == 1; });
  lock.unlock();
  if (gate.nonstandard) throw 17;
  throw std::runtime_error("rank-zero injection");
}
void release(Gate& gate) {
  std::lock_guard<std::mutex> lock(gate.mutex);
  gate.release = true;
  gate.ready.notify_all();
}
int main(int argc, char** argv) {
  try {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "identity") {
      const int ranks = std::stoi(argv[2]);
      fused_launch::Team team(ranks);
      Identity state(ranks);
      const auto main_thread = std::this_thread::get_id();
      for (int generation = 1; generation <= 128; ++generation) {
        state.generation = generation;
        require_success(team.dispatch([](void* pointer, int rank) {
          auto& state = *static_cast<Identity*>(pointer);
          require(rank >= 0 && rank < static_cast<int>(state.calls.size()), "invalid rank");
          const auto thread = std::this_thread::get_id();
          if (state.calls[rank]) require(state.threads[rank] == thread, "worker replaced");
          state.threads[rank] = thread;
          require(state.calls[rank] + 1 == state.generation, "skipped/repeated generation");
          ++state.calls[rank];
          state.observed[rank] = state.generation * 100 + rank;
        }, &state));
        for (int rank = 0; rank < ranks; ++rank) {
          require(state.calls[rank] == generation, "callback not complete");
          require(state.observed[rank] == generation * 100 + rank, "publication lost");
          require(state.threads[rank] != main_thread, "callback ran on caller");
          for (int peer = 0; peer < rank; ++peer) {
            require(state.threads[rank] != state.threads[peer], "ranks shared a worker");
          }
        }
      }
      team.stop();
      team.stop();
    } else if (mode == "publication" || mode == "inflight") {
      fused_launch::Team team(2);
      Gate gate;
      auto job = std::async(std::launch::async, [&] {
        return team.dispatch(blocked_callback, &gate);
      });
      {
        std::unique_lock<std::mutex> lock(gate.mutex);
        gate.ready.wait(lock, [&] { return gate.entered == 2; });
      }
      const bool returned_early = job.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
      fused_launch::Result overlapping;
      if (mode == "inflight") overlapping = team.dispatch(blocked_callback, &gate);
      release(gate);
      require_success(job.get());
      team.stop();
      require(!returned_early, "dispatch returned before callbacks completed");
      if (mode == "inflight") require(bool(overlapping.error), "overlapping dispatch accepted");
      require(gate.values == std::vector<int>({101, 102}), "callback writes unpublished");
    } else if (mode == "exception" || mode == "nonstandard") {
      fused_launch::Team team(2);
      Gate gate;
      gate.nonstandard = mode == "nonstandard";
      // Rank 1 cannot finish until dispatch returns and the caller releases it.
      // Thus a successful test proves error return does not wait for all done.
      const auto result = team.dispatch(failing_callback, &gate);
      release(gate);
      team.stop();
      require(result.rank == 0 && bool(result.error), "failure rank/exception lost");
      bool correct_exception = false;
      try {
        std::rethrow_exception(result.error);
      } catch (const std::runtime_error& error) {
        correct_exception = !gate.nonstandard && std::string(error.what()) == "rank-zero injection";
      } catch (int value) {
        correct_exception = gate.nonstandard && value == 17;
      }
      require(correct_exception, "exception not preserved");
      require(gate.values[1] == 102, "stop failed to join released worker");
    } else if (mode == "boundaries") {
      for (int ranks : {0, -1}) {
        bool rejected = false;
        try { fused_launch::Team invalid(ranks); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "invalid rank count accepted");
      }
      fused_launch::Team team(1);
      const auto invalid = team.dispatch(nullptr, nullptr);
      require(bool(invalid.error), "null callback accepted");
      team.stop();
      team.stop();
      const auto stopped = team.dispatch([](void*, int) {}, nullptr);
      require(bool(stopped.error), "dispatch after stop accepted");
      // A never-dispatched team's destructor must wake and join idle workers.
      { fused_launch::Team idle(2); }
    } else {
      throw std::runtime_error("unknown probe mode");
    }
    std::cout << "PASS " << mode << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
"""
        cls.probe = compile_host_probe(cls, source, "launch-host", "-pthread", "-I",
                                       str(ROOT / "benchmarks/sm103"))

    def run_case(self, mode, *arguments):
        result = subprocess.run([str(self.probe), mode, *map(str, arguments)],
                                text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), f"PASS {mode}")

    def test_fixed_rank_workers_and_exactly_once_generation_across_128_rounds(self):
        for ranks in (1, 2, 8):
            with self.subTest(ranks=ranks):
                self.run_case("identity", ranks)

    def test_success_waits_for_callback_end_and_publishes_writes(self):
        self.run_case("publication")

    def test_inflight_dispatch_is_rejected_without_corrupting_active_generation(self):
        self.run_case("inflight")

    def test_worker_error_returns_before_another_rank_is_released(self):
        self.run_case("exception")

    def test_nonstandard_worker_exception_is_preserved(self):
        self.run_case("nonstandard")

    def test_invalid_constructor_null_callback_and_idempotent_stop(self):
        self.run_case("boundaries")


class MpiEagerHostContracts(unittest.TestCase):
    """Compile the real MPI transport/parser/IPC/enqueue with explicit CPU mocks.

    These probes validate ownership, arguments and fail-closed branches, not
    actual MPI collectives, CUDA IPC visibility, or cross-process GPU ordering.
    """

    @classmethod
    def setUpClass(cls):
        directory = tempfile.TemporaryDirectory(prefix="fuse-mpi-host-mocks-")
        cls.addClassCleanup(directory.cleanup)
        cls.headers = Path(directory.name)
        (cls.headers / "mpi.h").write_text(r'''
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
using MPI_Comm = int;
enum { MPI_COMM_NULL=0, MPI_COMM_WORLD=1, MPI_SUCCESS=0, MPI_ERRORS_RETURN=0,
       MPI_THREAD_SINGLE=0, MPI_THREAD_FUNNELED=1, MPI_COMM_TYPE_SHARED=0,
       MPI_INFO_NULL=0, MPI_BYTE=1, MPI_INT=2, MPI_UINT64_T=3, MPI_MAX=0,
       MPI_MAX_ERROR_STRING=256 };
inline int env(const char* name, int fallback) { const char* x=std::getenv(name); return x?std::atoi(x):fallback; }
inline int mpi_barriers=0, mpi_gathers=0, mpi_aborts=0;
inline int MPI_Init_thread(int*, char***, int, int* provided) { *provided=1; return 0; }
inline int MPI_Comm_set_errhandler(int,int) { return 0; }
inline int MPI_Comm_rank(int,int* rank) { *rank=env("MOCK_RANK",2); return 0; }
inline int MPI_Comm_size(int,int* size) { *size=env("MOCK_WORLD",4); return 0; }
inline int MPI_Comm_split_type(int,int,int,int,int* comm) { *comm=2; return 0; }
inline int MPI_Comm_free(int* comm) { *comm=0; return 0; }
inline int MPI_Finalize() { return 0; }
inline int MPI_Error_string(int,char* out,int* size) { std::strcpy(out,"mock MPI failure"); *size=16; return 0; }
inline int MPI_Barrier(int) { ++mpi_barriers; return 0; }
inline int MPI_Abort(int,int) { ++mpi_aborts; return 0; }
inline int MPI_Allreduce(const void* input,void* output,int,int,int,int) {
  *static_cast<int*>(output)=*static_cast<const int*>(input) || env("MOCK_REMOTE_FAILURE",0); return 0;
}
inline int MPI_Bcast(void* data,int count,int type,int,int) {
  if (type==MPI_BYTE && count) {
    std::memcpy(data,"same physical geometry and candidate contract",count);
    if(env("MOCK_CONTRACT_MISMATCH",0)) static_cast<char*>(data)[0]='!';
  }
  return 0;
}
inline int MPI_Allgather(const void* input,int bytes,int,void* output,int,int,int) {
  ++mpi_gathers;
  for (int rank=0; rank<env("MOCK_WORLD",4); ++rank) {
    char* target=static_cast<char*>(output)+rank*bytes;
    if (bytes==32) std::snprintf(target,32,"GPU-%d",env("MOCK_DUPLICATE",0)?0:rank);
    else if (bytes==sizeof(float)) { const float value=(rank+1)*.25f; std::memcpy(target,&value,bytes); }
    else std::memcpy(target,input,bytes);
  }
  return 0;
}
''')
        (cls.headers / "cuda_runtime.h").write_text(r'''
#pragma once
#include <cstdint>
#include <cstring>
#include "mpi.h"
using cudaStream_t=void*; using cudaEvent_t=void*; using cublasHandle_t=void*;
using cudaError_t=int;
enum { cudaSuccess=0, cudaIpcMemLazyEnablePeerAccess=1, cudaMemcpyDeviceToHost=2 };
struct cudaIpcMemHandle_t { uintptr_t value=0; };
inline int last_device=-1, opened=0, exported=0, event_records=0, event_waits=0, stream_waits=0;
inline int cudaGetDeviceCount(int* count) { *count=env("MOCK_VISIBLE",1); return 0; }
inline int cudaSetDevice(int device) { last_device=device; return 0; }
inline int cudaDeviceGetPCIBusId(char* output,int bytes,int) { std::snprintf(output,bytes,"GPU-%d",env("MOCK_RANK",2)); return 0; }
inline int cudaIpcGetMemHandle(cudaIpcMemHandle_t* out,void* pointer) { ++exported; out->value=reinterpret_cast<uintptr_t>(pointer); return 0; }
inline int cudaIpcOpenMemHandle(void** out,cudaIpcMemHandle_t value,unsigned flag) {
  if(flag!=cudaIpcMemLazyEnablePeerAccess) return 1;
  ++opened; *out=reinterpret_cast<void*>(value.value); return 0;
}
inline int cudaEventRecord(void*,void*) { ++event_records; return 0; }
inline int cudaEventSynchronize(void*) { ++event_waits; return 0; }
inline int cudaStreamSynchronize(void*) { ++stream_waits; return 0; }
inline int cudaEventElapsedTime(float* output,void*,void*) { *output=(env("MOCK_RANK",2)+1)*.25f; return 0; }
inline int cudaMemcpy(void* output,const void* input,size_t bytes,int) { std::memcpy(output,input,bytes); return 0; }
''')
        harness = (ROOT / "benchmarks/sm103/fused_bf16.cu").read_text()
        host_types = harness[harness.index("constexpr int kWarmup ="):harness.index("\nvoid check_cuda(")]
        parser = harness[harness.index("Options parse_options("):harness.index("\nstruct RankRuntime")]
        runtime = harness[harness.index("struct RankRuntime {"):harness.index("\nvoid check_enqueue(")]
        wait_ipc = harness[harness.index("void wait_all("):harness.index("\nstd::vector<RankRuntime> create_runtimes(")]
        enqueue = harness[harness.index("struct HostLaunchTiming {"):harness.index("\nvoid cublas_nt(")]
        gather = harness[harness.index("void prepare_cpu_gather("):harness.index("\nvoid prepare_references(")]
        sequence = harness[harness.index("int global_sequence("):harness.index("\nvoid wait_all(")]
        source = r'''
#include "fused_launch.cuh"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <sstream>
#include <unistd.h>
using Bf16=uint16_t;
#include "fuse/layout/gemm.h"
namespace fused_validation { struct Scratch {}; }
namespace fused_inputs { struct Scratch {}; }
namespace fuse {
struct Params {
  GemmProblem gemm{};
  uint32_t epoch=0; uint32_t* ready=nullptr;
  Bf16* local_output=nullptr; Bf16* rhs_nt=nullptr;
  uint32_t* peer_route_done_epoch[8]{};
};
using GemmA2AParams=Params; using A2AGemmParams=Params;
inline int launches=0, reserved=0; inline bool capturing=false;
inline uint32_t last_epoch=0; inline uint32_t* last_ready=nullptr;
inline GemmProblem last_problem{};
int launch_gemm_a2a_cutlass(Params p,void*) { if (!capturing) ++launches; last_epoch=p.epoch; last_ready=p.ready; last_problem=p.gemm; return 0; }
int launch_a2a_gemm_cutlass(Params p,void* s) { return launch_gemm_a2a_cutlass(p,s); }
int launch_batched_cutlass_reference(Params p,void* s,int budget) { reserved=budget; return launch_gemm_a2a_cutlass(p,s); }
int launch_a2a_gemm_cutlass_reference(Params p,void* s,int budget) { return launch_batched_cutlass_reference(p,s,budget); }
int launch_gemm_a2a_copy_reference(Params p,void* s) { return launch_gemm_a2a_cutlass(p,s); }
int launch_a2a_gemm_copy_reference(Params p,void* s) { return launch_gemm_a2a_cutlass(p,s); }
}
// CUDA Graph itself is separately tested using the real helper and API mocks.
// Here the transport stand-in checks capture/enqueue/barrier/event ordering.
namespace fused_graph {
class Operation {
  int device_, before_barriers_=0, before_events_=0;
  void* stream_; uint32_t committed_, prepared_=0;
public:
  Operation(int device, void* stream, uint32_t epoch)
      : device_(device), stream_(stream), committed_(epoch) {}
  void reset(uint32_t epoch) { committed_=epoch; prepared_=0; }
  uint32_t committed_epoch() const { return committed_; }
  template<class F> void prepare(uint32_t epoch, F&& callback) {
    if (prepared_ || epoch != committed_+1 || last_device != device_)
      throw std::runtime_error("bad Graph prepare epoch/owner");
    before_barriers_=mpi_barriers; before_events_=event_records;
    const int before_launches=fuse::launches;
    fuse::capturing=true; const int status=callback(epoch,stream_); fuse::capturing=false;
    if (status || fuse::launches != before_launches || mpi_barriers != before_barriers_ ||
        event_records != before_events_) throw std::runtime_error("capture submitted work/events");
    prepared_=epoch;
  }
  void launch() {
    if (!prepared_ || mpi_barriers != before_barriers_+1 || event_records != before_events_+1)
      throw std::runtime_error("Graph replay lost prepare/barrier/start-event ordering");
    ++fuse::launches; committed_=prepared_; prepared_=0;
  }
};
}
#define CUDA_CHECK(x) do { if(x) throw std::runtime_error("mock CUDA failure"); } while(0)
void check_enqueue(const fused_launch::Result& value) { if(value.error) std::rethrow_exception(value.error); }
template<class T> std::vector<T> download(const T* source,size_t count) { return {source,source+count}; }
''' + host_types + parser + runtime + wait_ipc + enqueue + sequence + gather + r'''
int main(int argc,char** argv) {
  try {
    static_assert(kWarmup==10 && kSamples==50);
    fused_mpi::initialize(argc,argv);
    Options options=parse_options(argc,argv);
    fused_mpi::agree("same physical geometry and candidate contract");
    if (options.world!=env("MOCK_WORLD",4) || options.host_launch!="mpi_process") return 2;
    const int rank=fused_mpi::process_rank;
    std::vector<RankRuntime> ranks(options.world);
    auto& owned=ranks[rank]; owned.device=fused_mpi::device_for(rank);
    Bf16 data[8]{}; uint32_t flags[8]{};
    owned.qkv.local_output=data; owned.peer_output=data+1; owned.peer_input=data+2;
    owned.route_done=flags; owned.input_ready=flags+1;
    owned.calibration_route_done=options.calibrate ? flags+2 : nullptr;
    owned.qkv.rhs_nt=data+3; owned.oproj.rhs_nt=data+4;
    owned.calibration_qkv_ready=flags+3; owned.calibration_oproj_ready=flags+4;
    apply_schedule(owned.qkv.gemm,options,Direction::kQkv);
    apply_schedule(owned.oproj.gemm,options,Direction::kOproj);
    finish_all(ranks); exchange_peer_buffers(ranks,options);
    const int fields=5+options.calibrate+(options.cpu_oracle && options.input_generator=="gpu_philox" ? 2:0);
    if(exported!=fields || opened!=(options.world-1)*fields || event_waits!=1) return 3;
    for(int peer=0;peer<options.world;++peer) {
      if(!ranks[peer].qkv.local_output || !ranks[peer].peer_input || !ranks[peer].route_done ||
         !ranks[peer].input_ready || (options.calibrate && !ranks[peer].calibration_route_done)) return 4;
      if(peer!=rank && (ranks[peer].device!=-1 || !ranks[peer].allocations.empty())) return 5;
    }
    for(auto direction:{Direction::kQkv,Direction::kOproj}) {
      for(auto component:{MeasurementComponent::kFused,MeasurementComponent::kComputeReference,
                          MeasurementComponent::kCopyReference}) {
        options.component=component; options.comm_sm=24;
        bind_graph(ranks,options,86);
        const int repeats=options.launch=="graph"?3:1;
        for (int iteration=0; iteration<repeats; ++iteration) {
        const uint32_t epoch=87+iteration;
        const auto times=run_epoch(ranks,options,direction,epoch);
        for(int peer=0;peer<options.world;++peer) if(times[peer]!=(peer+1)*.25f) return 6;
        if(fuse::last_epoch!=epoch || last_device!=owned.device) return 7;
        const auto& scheduled=direction==Direction::kQkv?owned.qkv.gemm:owned.oproj.gemm;
        if(fuse::last_problem.raster!=scheduled.raster ||
           fuse::last_problem.max_swizzle_size!=scheduled.max_swizzle_size) return 12;
        if(component!=MeasurementComponent::kFused && fuse::last_ready!=
            (direction==Direction::kQkv?owned.calibration_qkv_ready:owned.calibration_oproj_ready)) return 8;
        }
        report_graph_preparation(ranks,options,direction,",component=mock");
        if (options.launch=="graph" && (owned.graph->committed_epoch()!=89 ||
            owned.graph_prepare_calls!=3)) return 13;
      }
    }
    const bool graph=options.launch=="graph";
    if(fuse::launches!=(graph?18:6) || fuse::reserved!=24 || event_records!=(graph?43:13) ||
       event_waits!=(graph?25:7) || stream_waits!=(graph?6:0)) return 9;
    options.seq_local=6; options.q_heads=options.world; options.head_dim=8;
    std::vector<std::vector<Bf16>> inputs(options.world);
    for(int peer=0;peer<options.world;++peer) {
      inputs[peer].resize(options.seq_local*options.q_width());
      for(size_t i=0;i<inputs[peer].size();++i) inputs[peer][i]=peer*1000+i;
      ranks[peer].peer_input=inputs[peer].data();
    }
    for(bool causal:{false,true}) {
      options.causal=causal; std::vector<Bf16> expected;
      prepare_cpu_gather(expected,ranks,options,rank);
      for(int row=0;row<options.seq_local;++row) {
        const int global=!causal?rank*6+row:(row<3?rank*3+row:(2*options.world-1-rank)*3+row-3);
        for(int peer=0;peer<options.world;++peer) for(int column=0;column<8;++column) {
          if(expected[row*options.q_width()+peer*8+column]!=inputs[peer][global*8+column]) return 10;
        }
      }
    }
    if(fused_mpi::any(false)!=bool(env("MOCK_REMOTE_FAILURE",0))) return 11;
    fused_mpi::finalize(); ::alarm(0);
    std::cout<<"PASS MPI host contracts rank="<<rank<<" device="<<owned.device<<'\n';
  } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; fused_mpi::abort(1); return 1; }
}
'''
        cls.probe = compile_host_probe(cls, source, "mpi-eager-host", "-DFUSE_BENCH_MPI=1",
            "-DFUSE_ENABLE_PROFILING=0", "-I", str(cls.headers), "-I", str(ROOT / "benchmarks/sm103"),
            "-pthread", "-Wall", "-Wextra", "-Werror")

    def run_probe(self, *arguments, **overrides):
        environment = dict(os.environ, MOCK_RANK="2", MOCK_WORLD="4", MOCK_VISIBLE="1")
        for name in ("FUSE_QKV_GEMM_POLICY", "FUSE_SM103_OPROJ_POLICY"):
            environment.pop(name, None)
        environment.update({name: str(value) for name, value in overrides.items()})
        return subprocess.run([str(self.probe), *arguments], env=environment,
                              text=True, capture_output=True, timeout=10)

    def test_real_mpi_seam_keeps_logical_rank_distinct_from_device_and_all_components(self):
        for world in (4, 8):
            for visible in (1, world):
                with self.subTest(world=world, visible=visible):
                    result = self.run_probe("--calibrate", MOCK_WORLD=world, MOCK_VISIBLE=visible)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(f"device={0 if visible == 1 else 2}", result.stdout)

    def test_nondefault_schedule_reaches_fused_compute_and_copy_without_losing_fields(self):
        for world in (4, 8):
            for swizzle, qkv_raster, oproj_raster in ((8, "along_n", "along_m"),
                                                    (4, "heuristic", "along_n")):
                with self.subTest(world=world, swizzle=swizzle, qkv=qkv_raster, oproj=oproj_raster):
                    result = self.run_probe("--calibrate", "--max-swizzle-size", str(swizzle),
                        "--qkv-raster", qkv_raster, "--oproj-raster", oproj_raster,
                        MOCK_WORLD=world)
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_real_ipc_optional_weight_oracle_and_remote_expiry_vote(self):
        result = self.run_probe("--cpu-oracle", "--input-generator", "gpu_philox", MOCK_REMOTE_FAILURE=1)
        # A remote failure also makes the initial contract agreement fail closed.
        self.assertNotEqual(result.returncode, 0)
        result = self.run_probe("--cpu-oracle", "--input-generator", "gpu_philox")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_mpi_rejects_unsupported_execution_and_inconsistent_rank_contracts(self):
        for arguments, overrides in ((["--world", "8"], {}), (["--host-launch", "sequential"], {}),
            (["--profile"], {}), (["--launch", "invalid"], {}),
            (["--launch", "graph", "--profile"], {}), ([], {"MOCK_DUPLICATE": 1}),
            ([], {"MOCK_WORLD": 2}), ([], {"MOCK_VISIBLE": 2}), ([], {"MOCK_CONTRACT_MISMATCH": 1})):
            with self.subTest(arguments=arguments, overrides=overrides):
                self.assertNotEqual(self.run_probe(*arguments, **overrides).returncode, 0)

    def test_graph_prepares_before_barrier_and_events_and_replays_all_components(self):
        for world in (4, 8):
            result = self.run_probe("--launch", "graph", "--calibrate", MOCK_WORLD=world)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count("graph_prepare,"), 6)
            self.assertEqual(result.stdout.count("calls=3,first_epoch=87,last_epoch=89"), 6)
            self.assertIn("gpu_sample_time=0", result.stdout)

    def test_shared_harness_retains_full_validation_epoch_and_cleanup_protocols(self):
        source = (ROOT / "benchmarks/sm103/fused_bf16.cu").read_text()
        self.assertNotIn("cudaSetDevice(rank)", source)
        self.assertNotIn("cudaSetDevice(0)", source)
        self.assertIn("fused_mpi::gather_owned(results);", source)
        self.assertIn("!fused_mpi::any(warm_wall_seconds() >= 5)", source)
        cleanup = source[source.index("void destroy_runtimes("):source.index("int main(")]
        self.assertLess(cleanup.index("finish_all(runtimes)"), cleanup.index("cudaIpcCloseMemHandle"))
        self.assertLess(cleanup.index("owner.graph.reset()"), cleanup.index("cudaIpcCloseMemHandle"))
        self.assertLess(cleanup.index("fused_mpi::barrier()"), cleanup.index("cudaFree(allocation)"))
        cmake = (ROOT / "cmake/sm103.cmake").read_text()
        self.assertIn('option(FUSE_BUILD_MPI_BENCH "Build the same full-validation harness with MPI Eager execution" OFF)', cmake)
        self.assertIn("add_executable(fused_bf16_mpi benchmarks/sm103/fused_bf16.cu)", cmake)


class CompletionWaitHostContracts(unittest.TestCase):
    """Execute the real completion loop with explicit CUDA-call stand-ins."""

    @classmethod
    def setUpClass(cls):
        source = (ROOT / 'benchmarks/sm103/fused_bf16.cu').read_text()
        body = source[source.index('void wait_all('):source.index('void finish_all(')]
        probe = r'''
#include <vector>
#include <stdexcept>
struct RankRuntime { int device, end; };
std::vector<int> calls;
int fail_event = -1;
int cudaSetDevice(int rank) { calls.push_back(rank); return 0; }
int cudaEventSynchronize(int event) { calls.push_back(event); return event == fail_event; }
#define CUDA_CHECK(expr) do { if (expr) throw std::runtime_error("CUDA failure"); } while (0)
''' + body + r'''
int main() {
  std::vector<RankRuntime> ranks{{0,10}, {1,11}, {2,12}, {3,13}};
  wait_all(ranks);
  if (calls != std::vector<int>{0,10,1,11,2,12,3,13}) return 1;
  calls.clear(); fail_event = 11;
  try { wait_all(ranks); return 2; } catch (const std::runtime_error&) {}
  return calls == std::vector<int>{0,10,1,11} ? 0 : 3;
}
'''
        cls.probe = compile_host_probe(cls, probe, 'completion-wait', '-Wall', '-Wextra', '-Werror')

    def test_actual_events_are_waited_on_without_poll_sleeps_and_errors_propagate(self):
        result = subprocess.run([str(self.probe)], timeout=10, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class ValidationHostContracts(unittest.TestCase):
    """Real validation-header arithmetic only; GPU reductions remain untested."""

    @classmethod
    def setUpClass(cls):
        source = r"""
#include "fused_validation.cuh"
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
using fused_validation::Stats;
void print(const Stats& value) {
  std::cout << std::setprecision(17) << value.checked << ' ' << value.mismatches << ' '
            << value.nonfinite << ' ' << value.error_square << ' ' << value.reference_square
            << ' ' << value.max_abs << ' ' << value.first_index << ' '
            << value.first_actual << ' ' << value.first_expected << '\n';
}
int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "map") {
    fused_validation::QkvOracle oracle{};
    uint64_t index;
    while (std::cin >> oracle.world >> oracle.seq_local >> oracle.q_width >> oracle.kv_width
                    >> oracle.destination >> index) {
      const auto address = oracle.address(index);
      std::cout << address.rank << ' ' << address.offset << '\n';
    }
  } else if (mode == "load") {
    fused_validation::QkvOracle oracle{};
    std::cin >> oracle.world >> oracle.seq_local >> oracle.q_width >> oracle.kv_width >> oracle.destination;
    const uint64_t count = oracle.seq_local * (oracle.q_width + 2 * oracle.kv_width);
    std::vector<std::vector<uint16_t>> sources(oracle.world, std::vector<uint16_t>(count));
    for (int rank = 0; rank < oracle.world; ++rank) {
      for (uint64_t index = 0; index < count; ++index) {
        sources[rank][index] = static_cast<uint16_t>((rank * count + index) * 257 + 19);
      }
      oracle.source[rank] = sources[rank].data();
    }
    for (uint64_t index = 0; index < count; ++index) std::cout << oracle(index) << '\n';
  } else if (mode == "convert") {
    unsigned value;
    while (std::cin >> value) {
      std::cout << std::setprecision(17)
                << fused_validation::bf16_to_float(static_cast<uint16_t>(value)) << '\n';
    }
  } else if (mode == "counters") {
    auto a = Stats::zero(), b = Stats::zero();
    a.checked = (uint64_t{1} << 32) + 3;
    a.mismatches = (uint64_t{1} << 32) + 1;
    a.nonfinite = (uint64_t{1} << 32);
    b.checked = 11;
    b.mismatches = 7;
    b.nonfinite = 5;
    b.first_index = (uint64_t{1} << 33) + 2;
    b.first_actual = 0x3f80;
    b.first_expected = 0;
    a.merge(b);
    print(a);
  } else if (mode == "numeric" || mode == "bits") {
    auto total = Stats::zero();
    Stats parts[2] = {Stats::zero(), Stats::zero()};
    unsigned actual, expected;
    uint64_t index;
    while (std::cin >> actual >> expected >> index) {
      auto& part = parts[index % 2];
      if (mode == "numeric") {
        total.observe_numeric(actual, expected, index);
        part.observe_numeric(actual, expected, index);
      } else {
        total.observe_bits(actual, expected, index);
        part.observe_bits(actual, expected, index);
      }
    }
    print(total);
    auto merged = Stats::zero();
    merged.merge(parts[1]);
    merged.merge(Stats::zero());
    merged.merge(parts[0]);
    print(merged);
  } else return 2;
  return 0;
}
"""
        cls.probe = compile_host_probe(cls, source, "validation-header-host", "-I",
                                       str(ROOT / "benchmarks/sm103"))

    def query(self, mode, cases=()):
        result = subprocess.run([str(self.probe), mode],
                                input="".join(" ".join(map(str, row)) + "\n" for row in cases),
                                text=True, capture_output=True, check=True, timeout=30)
        return result.stdout.splitlines()

    @staticmethod
    def bf16(value):
        return struct.unpack("<f", struct.pack("<I", value << 16))[0]

    @staticmethod
    def forward_addresses(world, rows, q_width, kv_width, destination):
        """Independent source-to-packed traversal, not the header's inverse formula."""
        expected = {}
        packed_base = 0
        for width, source_base in ((q_width, 0), (kv_width, q_width),
                                   (kv_width, q_width + kv_width)):
            local = width // world
            for source, row, feature in product(range(world), range(rows), range(local)):
                packed = packed_base + (source * rows + row) * local + feature
                offset = row * (q_width + 2 * kv_width) + source_base + destination * local + feature
                expected[packed] = (source, offset)
            packed_base += world * rows * local
        return expected

    def statistics(self, mode, samples):
        lines = self.query(mode, samples)
        self.assertEqual(len(lines), 2)
        records = []
        for line in lines:
            fields = line.split()
            records.append(tuple(map(int, fields[:3])) + tuple(map(float, fields[3:6])) +
                           tuple(map(int, fields[6:])))
        # Different merge order may change floating sums, never integer evidence.
        for index, (serial, merged) in enumerate(zip(*records)):
            if index in (3, 4):
                self.assertTrue(math.isclose(serial, merged, rel_tol=1e-12, abs_tol=1e-30))
            else:
                self.assertEqual(serial, merged)
        return records[0]

    def test_real_qkv_inverse_mapping_matches_complete_forward_layouts(self):
        cases, expected = [], []
        for world, rows, q_local, dimension in product((1, 4, 8), (1, 3), (1, 2), (8, 16)):
            q_width, kv_width = world * q_local * dimension, world * dimension
            for destination in range(world):
                mapping = self.forward_addresses(world, rows, q_width, kv_width, destination)
                self.assertEqual(sorted(mapping), list(range(rows * (q_width + 2 * kv_width))))
                for index, address in sorted(mapping.items()):
                    cases.append((world, rows, q_width, kv_width, destination, index))
                    expected.append(address)
        actual = [tuple(map(int, line.split())) for line in self.query("map", cases)]
        self.assertEqual(actual, expected)

    def test_real_qkv_mapping_preserves_large_64bit_segment_and_row_offsets(self):
        cases, expected = [], []
        world, q_width, kv_width = 8, 4096, 1024
        for rows in (8192, 16384, 1048576):
            for destination in (0, 7):
                packed_base = 0
                for width, source_base in ((q_width, 0), (kv_width, q_width),
                                           (kv_width, q_width + kv_width)):
                    local = width // world
                    for source, row, feature in product((0, 7), (0, rows - 1), (0, local - 1)):
                        index = packed_base + (source * rows + row) * local + feature
                        cases.append((world, rows, q_width, kv_width, destination, index))
                        expected.append((source, row * (q_width + 2 * kv_width) +
                                         source_base + destination * local + feature))
                    packed_base += world * rows * local
        self.assertTrue(any(case[-1] > 2**32 for case in cases))
        self.assertTrue(any(address[-1] > 2**32 for address in expected))
        self.assertEqual([tuple(map(int, line.split())) for line in self.query("map", cases)], expected)

    def test_real_qkv_oracle_loads_the_addressed_source_bits(self):
        world, rows, q_width, kv_width = 4, 3, 64, 32
        count = rows * (q_width + 2 * kv_width)
        for destination in range(world):
            mapping = self.forward_addresses(world, rows, q_width, kv_width, destination)
            expected = [((source * count + offset) * 257 + 19) & 0xffff
                        for _, (source, offset) in sorted(mapping.items())]
            actual = list(map(int, self.query("load", [(world, rows, q_width, kv_width, destination)])))
            self.assertEqual(actual, expected)

    def test_real_bf16_conversion_preserves_finite_values_and_special_encodings(self):
        values = (0, 0x8000, 1, 0x007f, 0x0080, 0x3f80, 0xbf80, 0x7f7f, 0xff7f,
                  0x7f80, 0xff80, 0x7fc0, 0xffff)
        for bits, line in zip(values, self.query("convert", [(value,) for value in values])):
            actual, expected = float(line), self.bf16(bits)
            if math.isnan(expected):
                self.assertTrue(math.isnan(actual))
            else:
                self.assertEqual(actual, expected)
                if expected == 0:
                    self.assertEqual(math.copysign(1, actual), math.copysign(1, expected))

    def test_numeric_tolerance_first_failure_and_merge_use_real_stats(self):
        samples = [(0, 0x8000, 0), (0x3f80, 0x3f80, 1), (0x3c23, 0, 2),
                   (0x3c24, 0, 3), (0x3f82, 0x3f80, 4), (0x3f83, 0x3f80, 5),
                   (0xbf82, 0xbf80, 6), (0xbf83, 0xbf80, 7)]
        stats = self.statistics("numeric", list(reversed(samples)))
        self.assertEqual(stats[:3], (8, 3, 0))
        self.assertEqual(stats[6:], (3, 0x3c24, 0))
        errors = [abs(self.bf16(a) - self.bf16(b)) for a, b, _ in samples]
        self.assertAlmostEqual(stats[3], sum(value * value for value in errors))
        self.assertAlmostEqual(stats[4], sum(self.bf16(b)**2 for _, b, _ in samples))
        self.assertEqual(stats[5], max(errors))

    def test_nonfinite_actual_or_reference_never_passes_numeric_or_bitwise_checks(self):
        for mode in ("numeric", "bits"):
            for special in (0x7f80, 0xff80, 0x7fc0, 0xffff):
                for actual, expected in ((special, 0), (0, special), (special, special)):
                    with self.subTest(mode=mode, actual=actual, expected=expected):
                        stats = self.statistics(mode, [(actual, expected, 2**33 + 7)])
                        self.assertEqual(stats[:3], (1, 1, 1))
                        self.assertEqual(stats[6:], (2**33 + 7, actual, expected))
                        self.assertEqual(stats[3:5], (0, 0))
                        if mode == "numeric":
                            self.assertEqual(stats[5], math.inf)
        mixed = self.statistics("numeric", [(0x3f80, 0x3f80, 8), (0x7fc0, 0, 3),
                                             (0x3f80, 0, 9), (0, 0x7f80, 1)])
        self.assertEqual(mixed[:3], (4, 3, 2))
        self.assertEqual(mixed[3:6], (1, 1, math.inf))
        self.assertEqual(mixed[6:], (1, 0, 0x7f80))

    def test_bitwise_checks_detect_signed_zero_and_single_bit_poison(self):
        samples = [(0, 0x8000, 8), (0x3f80, 0x3f81, 2), (1, 0, 9), (0x3f80, 0x3f80, 1)]
        stats = self.statistics("bits", samples)
        self.assertEqual(stats[:3], (4, 3, 0))
        self.assertEqual(stats[6:], (2, 0x3f80, 0x3f81))
        self.assertEqual(self.statistics("numeric", [(0, 0x8000, 8)])[:3], (1, 0, 0))

    def test_numeric_statistics_avoid_float_square_overflow_and_reject_difference_overflow(self):
        maximum = self.bf16(0x7f7f)
        stats = self.statistics("numeric", [(0x7f7f, 0, 0), (0x7f7f, 0x7f7f, 1)])
        self.assertEqual(stats[:3], (2, 1, 0))
        self.assertTrue(math.isfinite(stats[3]) and math.isfinite(stats[4]))
        self.assertEqual(stats[3:5], (maximum * maximum, maximum * maximum))
        overflow = self.statistics("numeric", [(0x7f7f, 0xff7f, 0)])
        self.assertEqual(overflow[:3], (1, 1, 1))
        self.assertEqual(overflow[3:6], (0, 0, math.inf))

    def test_stats_zero_identity_and_64bit_counters(self):
        empty = self.statistics("numeric", [])
        self.assertEqual(empty[:6], (0, 0, 0, 0, 0, 0))
        self.assertEqual(empty[6], 2**64 - 1)
        fields = self.query("counters")[0].split()
        self.assertEqual(tuple(map(int, fields[:3])), (2**32 + 14, 2**32 + 8, 2**32 + 5))
        self.assertEqual(tuple(map(int, fields[6:])), (2**33 + 2, 0x3f80, 0))


class InputHostContracts(unittest.TestCase):
    """Real input-header statistics/mapping, not Philox or CUDA execution."""

    @classmethod
    def setUpClass(cls):
        source = r"""
#include "fused_inputs.cuh"
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
using fused_inputs::Stats;
void print(const Stats& value) {
  std::cout << std::setprecision(17) << value.count << ' ' << value.finite << ' '
            << value.nonzero << ' ' << value.sum << ' ' << value.square_sum << ' '
            << value.minimum << ' ' << value.maximum << '\n';
}
int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "stats") {
    auto total = Stats::zero();
    Stats parts[2] = {Stats::zero(), Stats::zero()};
    uint32_t bits;
    unsigned index = 0;
    while (std::cin >> bits) {
      float value;
      std::memcpy(&value, &bits, sizeof(value));
      total.observe(value);
      parts[index++ % 2].observe(value);
    }
    print(total);
    auto merged = Stats::zero();
    merged.merge(parts[1]);
    merged.merge(Stats::zero());
    merged.merge(parts[0]);
    print(merged);
  } else if (mode == "counters") {
    auto a = Stats::zero(), b = Stats::zero();
    a.count = (uint64_t{1} << 32) + 3;
    a.finite = (uint64_t{1} << 32) + 2;
    a.nonzero = (uint64_t{1} << 32) + 1;
    b.count = 11;
    b.finite = 7;
    b.nonzero = 5;
    a.merge(b);
    print(a);
  } else if (mode == "uniform") {
    uint32_t bits;
    float magnitude;
    while (std::cin >> bits >> magnitude) {
      std::cout << std::setprecision(17) << fused_inputs::uniform_value(bits, magnitude) << '\n';
    }
  } else if (mode == "map") {
    fused_inputs::OprojGather gather{};
    uint64_t index;
    while (std::cin >> gather.world >> gather.seq_local >> gather.q_width
                    >> gather.destination >> gather.causal >> index) {
      const auto address = gather.address(index);
      std::cout << address.rank << ' ' << address.offset << '\n';
    }
  } else if (mode == "load") {
    fused_inputs::OprojGather gather{};
    std::cin >> gather.world >> gather.seq_local >> gather.q_width >> gather.destination >> gather.causal;
    const uint64_t count = gather.seq_local * gather.q_width;
    std::vector<std::vector<uint16_t>> sources(gather.world, std::vector<uint16_t>(count));
    for (int rank = 0; rank < gather.world; ++rank) {
      for (uint64_t index = 0; index < count; ++index) {
        sources[rank][index] = static_cast<uint16_t>((rank * count + index) * 257 + 31);
      }
      gather.source[rank] = sources[rank].data();
    }
    for (uint64_t index = 0; index < count; ++index) std::cout << gather(index) << '\n';
  } else return 2;
  return 0;
}
"""
        cls.probe = compile_host_probe(cls, source, "inputs-header-host", "-I",
                                       str(ROOT / "benchmarks/sm103"))

    def query(self, mode, cases=()):
        result = subprocess.run([str(self.probe), mode],
                                input="".join(" ".join(map(str, row)) + "\n" for row in cases),
                                text=True, capture_output=True, check=True, timeout=30)
        return result.stdout.splitlines()

    @staticmethod
    def f32(value):
        return struct.unpack("<f", struct.pack("<f", value))[0]

    @staticmethod
    def float_bits(value):
        return struct.unpack("<I", struct.pack("<f", value))[0]

    def statistics(self, values):
        records = []
        for line in self.query("stats", [(value,) for value in values]):
            fields = line.split()
            records.append(tuple(map(int, fields[:3])) + tuple(map(float, fields[3:])))
        self.assertEqual(len(records), 2)
        for index, (serial, merged) in enumerate(zip(*records)):
            if index in (3, 4):
                self.assertTrue(math.isclose(serial, merged, rel_tol=1e-12, abs_tol=1e-30))
            else:
                self.assertEqual(serial, merged)
        return records[0]

    @staticmethod
    def owned_rows(world, rows, destination, causal):
        if not causal:
            return range(destination * rows, (destination + 1) * rows)
        half = rows // 2
        return [row for chunk in (destination, 2 * world - destination - 1)
                for row in range(chunk * half, (chunk + 1) * half)]

    @classmethod
    def forward_gather(cls, world, rows, width, destination, causal):
        """Enumerate each source shard into independently selected row slices."""
        shard = width // world
        expected = {}
        for row, global_row in enumerate(cls.owned_rows(world, rows, destination, causal)):
            for source, feature in product(range(world), range(shard)):
                expected[row * width + source * shard + feature] = (source, global_row * shard + feature)
        return expected

    def test_uniform_transform_uses_high_24_bits_and_bounded_endpoints(self):
        bits = (0, 1, 255, 256, 511, 0x7fffffff, 0x80000000, 0x800000ff,
                0xffffff00, 0xffffffff)
        cases = [(value, magnitude) for magnitude in (0.125, 0.02) for value in bits]
        actual = list(map(float, self.query("uniform", cases)))
        expected = [self.f32(((value >> 8) / 16777216.0 - 0.5) *
                             self.f32(2 * self.f32(magnitude))) for value, magnitude in cases]
        self.assertEqual(actual, expected)
        for index, magnitude in enumerate((0.125, 0.02)):
            values = actual[index * len(bits):(index + 1) * len(bits)]
            self.assertEqual(values[:3], [-self.f32(magnitude)] * 3)
            self.assertEqual(values[6:8], [0.0, 0.0])
            self.assertEqual(values[-2], values[-1])
            self.assertTrue(all(-self.f32(magnitude) <= value < self.f32(magnitude) for value in values))

    def test_input_stats_count_finite_nonzero_and_skip_invalid_aggregates(self):
        finite = (-0.125, -0.02, -0.0, 0.0, 0.02, 0.125)
        values = [self.float_bits(value) for value in finite] + [0x7f800000, 0xff800000, 0x7fc00000]
        stats = self.statistics(values)
        self.assertEqual(stats[:3], (9, 6, 4))
        expected = [self.f32(value) for value in finite]
        self.assertAlmostEqual(stats[3], sum(expected))
        self.assertAlmostEqual(stats[4], sum(value * value for value in expected))
        self.assertEqual(stats[5:], (-0.125, 0.125))

    def test_input_stats_zero_identity_invalid_only_and_64bit_counts(self):
        self.assertEqual(self.statistics([]), (0, 0, 0, 0, 0, math.inf, -math.inf))
        self.assertEqual(self.statistics([0x7fc00000, 0x7f800000, 0xff800000]),
                         (3, 0, 0, 0, 0, math.inf, -math.inf))
        fields = self.query("counters")[0].split()
        self.assertEqual(tuple(map(int, fields[:3])), (2**32 + 14, 2**32 + 9, 2**32 + 6))

    def test_input_square_statistics_do_not_overflow_at_finite_float_limit(self):
        maximum = struct.unpack("<f", struct.pack("<I", 0x7f7fffff))[0]
        stats = self.statistics([0x7f7fffff, 0xff7fffff])
        self.assertEqual(stats[:4], (2, 2, 2, 0))
        self.assertTrue(math.isfinite(stats[4]))
        self.assertEqual(stats[4], 2 * maximum * maximum)
        self.assertEqual(stats[5:], (-maximum, maximum))

    def test_real_oproj_gather_covers_all_ranks_causal_chunks_and_tail_rows(self):
        cases, expected = [], []
        for world, shard, causal in product((1, 4, 8), (1, 8), (False, True)):
            width = world * shard
            for rows in ((2, 6, 130) if causal else (1, 3, 129)):
                for destination in range(world):
                    mapping = self.forward_gather(world, rows, width, destination, causal)
                    self.assertEqual(sorted(mapping), list(range(rows * width)))
                    for index, address in sorted(mapping.items()):
                        cases.append((world, rows, width, destination, int(causal), index))
                        expected.append(address)
        actual = [tuple(map(int, line.split())) for line in self.query("map", cases)]
        self.assertEqual(actual, expected)

    def test_real_oproj_gather_preserves_large_64bit_indices_and_offsets(self):
        cases, expected = [], []
        world, width = 8, 4096
        shard = width // world
        for rows, destination, causal in product((131072, 2**21), range(world), (False, True)):
            half = rows // 2
            selected_rows = (0, half - 1, half, rows - 1)
            # Independent ownership intervals; avoid materializing a huge tensor.
            intervals = ((destination * half, (2 * world - destination - 1) * half)
                         if causal else (destination * rows, destination * rows + half))
            for row, source, feature in product(selected_rows, (0, 7), (0, shard - 1)):
                global_row = intervals[row // half] + row % half
                index = row * width + source * shard + feature
                cases.append((world, rows, width, destination, int(causal), index))
                expected.append((source, global_row * shard + feature))
        self.assertTrue(any(case[-1] > 2**32 for case in cases))
        self.assertTrue(any(address[-1] > 2**32 for address in expected))
        self.assertEqual([tuple(map(int, line.split())) for line in self.query("map", cases)], expected)

    def test_real_oproj_gather_reads_source_bits_in_both_row_orders(self):
        world, rows, width = 4, 6, 64
        count = rows * width
        for destination, causal in product(range(world), (False, True)):
            mapping = self.forward_gather(world, rows, width, destination, causal)
            expected = [((source * count + offset) * 257 + 31) & 0xffff
                        for _, (source, offset) in sorted(mapping.items())]
            actual = list(map(int, self.query("load", [(world, rows, width, destination, int(causal))])))
            self.assertEqual(actual, expected)


class CommunicationModels(unittest.TestCase):
    """Address/count models only: passing is not GPU synchronization evidence."""

    def test_a2a_bulk_task_coverage_and_empty_tail_arrivals(self):
        checked = 0
        for m, world, rows, window, ctas in product(
            (1, 63, 128, 129, 384, 513), (1, 4, 8), (3, 32, 48, 128),
            (1, 3, 32), (1, 7, 12, 147),
        ):
            m_tiles, chunks = ceil_div(m, 128), ceil_div(128, rows)
            tasks = list(a2a_bulk_tasks(m, world, rows, window, ctas))
            expected = Counter({key: 1 for key in product(range(m_tiles), range(world), range(chunks))})
            self.assertEqual(Counter(tasks), expected)
            copied_rows = Counter()
            arrivals = Counter()
            for tile, peer, chunk in tasks:
                begin = tile * 128 + chunk * rows
                copied_rows[tile, peer] += max(0, min(rows, 128 - chunk * rows, m - begin))
                arrivals[tile, peer] += 1  # Includes tasks containing no valid rows.
            for tile, peer in product(range(m_tiles), range(world)):
                self.assertEqual(copied_rows[tile, peer], min(128, m - tile * 128))
                self.assertEqual(arrivals[tile, peer], chunks)
            checked += 1
        self.assertEqual(checked, 864)

    def test_qkv_sequence_mapping_matches_rank_major_boundary(self):
        # The SM90 QKV smoke and SM103 measurement.qkv_route_check concatenate
        # source ranks. Interleaved weights/deferred V change features, not rows.
        checked = 0
        for world, seq, batch in product((1, 2, 4, 8), (1, 3, 63, 64, 130), (1, 2)):
            all_rows = []
            for rank in range(world):
                expected = [b * world * seq + rank * seq + local
                            for b in range(batch) for local in range(seq)]
                actual = [qkv_sequence_row(rank, world, seq, row)
                          for row in range(batch * seq)]
                self.assertEqual(actual, expected)
                all_rows.extend(actual)
            self.assertEqual(sorted(all_rows), list(range(batch * world * seq)))
            checked += 1
        self.assertEqual(checked, 40)

    def test_oproj_sequence_mapping_matches_independent_chunk_partition(self):
        checked = 0
        for world, seq, batch, causal in product(
            (1, 2, 4, 8), (2, 6, 64, 128, 130), (1, 2), (False, True),
        ):
            all_rows = []
            for rank in range(world):
                chunks = (rank, 2 * world - rank - 1) if causal else (2 * rank, 2 * rank + 1)
                expected = [b * world * seq + chunk * (seq // 2) + row
                            for b in range(batch) for chunk in chunks for row in range(seq // 2)]
                actual = [oproj_sequence_row(rank, world, seq, row, causal)
                          for row in range(batch * seq)]
                self.assertEqual(actual, expected)
                all_rows.extend(actual)
            self.assertEqual(sorted(all_rows), list(range(batch * world * seq)))
            checked += 1
        self.assertEqual(checked, 80)

    def test_qkv_and_causal_oproj_have_distinct_sequence_contracts(self):
        self.assertEqual(qkv_sequence_row(1, 4, 8, 4), 12)
        self.assertEqual(oproj_sequence_row(1, 4, 8, 4, True), 24)
        for world in (1, 2, 4, 8):
            for rank in range(world):
                for row in range(16):
                    self.assertEqual(qkv_sequence_row(rank, world, 8, row),
                                     oproj_sequence_row(rank, world, 8, row, False))

    def test_qkv_eight_warp_work_coverage_in_both_rasters(self):
        for world, m_chunks, heads, ctas, along_n in product(
            (1, 4, 8), (1, 3, 5), (1, 3, 9), (1, 7, 12, 147), (False, True),
        ):
            seen = Counter()
            tasks = world * m_chunks * heads
            for slot, cta in product(range(8), range(ctas)):
                for work in range(slot * ctas + cta, tasks, ctas * 8):
                    peer_work, peer = divmod(work, world)
                    if along_n:
                        m, head = divmod(peer_work, heads)
                    else:
                        head, m = divmod(peer_work, m_chunks)
                    seen[peer, m, head] += 1
            self.assertEqual(seen, Counter({key: 1 for key in
                product(range(world), range(m_chunks), range(heads))}))

    def test_copy_chunks_wait_for_every_overlapping_producer_tile(self):
        crossed_n = False
        for dim, base, row_begin, rows in product((8, 96, 128, 136, 192, 256),
                                                 (0, 8, 96, 128), (0, 64, 128), (1, 64, 128)):
            for offset in range(0, dim, 128):
                first = base + offset
                count = min(128, dim - offset)
                producer_m = range(row_begin // 128, (row_begin + rows - 1) // 128 + 1)
                producer_n = range(first // 128, (first + count - 1) // 128 + 1)
                signals = set(product(producer_m, producer_n))
                touched = {(row // 128, column // 128)
                           for row in range(row_begin, row_begin + rows)
                           for column in range(first, first + count)}
                self.assertEqual(signals, touched)
                crossed_n |= len(producer_n) > 1
        self.assertTrue(crossed_n)

    def test_qkv_vector_warps_wait_for_every_producer_of_one_m_tile(self):
        for tile_m, physical_feature, columns in product(
            (0, 1, 17), (0, 8, 96, 128, 264), (8, 64, 128),
        ):
            first = physical_feature // 128
            last = (physical_feature + columns - 1) // 128
            signals = [(tile_m, first + item)
                       for warp in range(8)
                       for item in range(warp, last - first + 1, 8)]
            self.assertEqual(Counter(signals), Counter({(tile_m, n): 1
                                                       for n in range(first, last + 1)}))

    def test_cyclic_k_shards_require_the_same_weight_permutation(self):
        for world, rank in [(world, rank) for world in (1, 4, 8) for rank in range(world)]:
            activation = [2 * peer + 1 for peer in range(world)]
            weight = [3 * peer + 2 for peer in range(world)]
            order = [(rank + peer_slot) % world for peer_slot in range(world)]
            self.assertEqual(sorted(order), list(range(world)))
            reference = sum(a * b for a, b in zip(activation, weight))
            self.assertEqual(sum(activation[peer] * weight[peer] for peer in order), reference)
            if world > 1 and rank != 0:
                self.assertNotEqual(sum(activation[peer] * weight[slot]
                                        for slot, peer in enumerate(order)), reference)


if __name__ == "__main__":
    unittest.main(verbosity=2)
