"""CPU arithmetic contracts for the SM103 fused BF16 implementation.

These executable models check exact coverage and producer/consumer agreement
against independent Cartesian or elementwise oracles. They do not compile or
execute the CUDA headers, establish memory ordering, prove GPU residency, or
validate numerical GEMM results. CUDA compilation and device tests remain
separate, required validation after the user's source review.

The scheduler model follows CUTLASS 57e3cfb's one-CTA static mapping, including
its swizzle padding and both raster directions. The optional CUTLASS_ROOT host
probe executes the real CUDA 13 wrapper macro with driver/runtime stand-ins;
it checks branch selection and argument/error forwarding, not the driver ABI,
linking against libcuda, descriptor validity, or performance. Optional source
contracts also check CUTLASS's delayed-store flush before ready publication;
they do not execute TMA or prove device memory ordering. Run with:
    python3 scripts/test_sm103_fusion_contract.py
"""

from collections import Counter
from dataclasses import dataclass
from itertools import product
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest


TILE_M = 128
TILE_N = 128
TILE_K = 64
READY_STRIDE = 32
UINT32_MAX = (1 << 32) - 1
ROOT = Path(__file__).resolve().parents[1]


class DriverCallBuildContracts(unittest.TestCase):
    def test_direct_driver_branch_is_private_to_sm103_static_library(self):
        source = (ROOT / "cmake/sm103.cmake").read_text()
        commands = re.findall(r"target_compile_definitions\s*\(([^)]*)\)", source)
        direct = [command.split() for command in commands
                  if "CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL" in command]
        self.assertEqual(direct, [["fuse_kernels", "PRIVATE",
                                  "CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1"]])
        links = re.findall(r"target_link_libraries\s*\(([^)]*)\)", source)
        self.assertEqual([command.split() for command in links if "CUDA::cuda_driver" in command],
                         [["fuse_kernels", "PUBLIC", "CUDA::cudart", "CUDA::cuda_driver"]])
        library = source.index("add_library(fuse_kernels STATIC")
        definition = source.index("target_compile_definitions(fuse_kernels PRIVATE")
        self.assertLess(library, definition)
        self.assertLess(definition, source.index("find_package(Threads REQUIRED)"))
        for path in (ROOT / "CMakeLists.txt", *sorted((ROOT / "cmake").glob("*.cmake")),
                     ROOT / "benchmarks/sm103/CMakeLists.txt"):
            if path == ROOT / "benchmarks/sm103/CMakeLists.txt":
                # The optional Blackwell-only pure-GEMM experiment also needs
                # this private branch; no preexisting baseline target inherits it.
                baseline = path.read_text()
                commands = re.findall(r"target_compile_definitions\s*\(([^)]*)\)", baseline)
                direct = [command.split() for command in commands
                          if "CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL" in command]
                self.assertEqual(len(direct), 1)
                self.assertEqual(direct[0][:3], ["fuse_sm103_cutlass_bf16", "PRIVATE",
                                               "CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1"])
                prefix = baseline.split("if(FUSE_SM103_BUILD_CUTLASS_BF16)")[0]
                self.assertNotIn("CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL", prefix)
                continue
            if path != ROOT / "cmake/sm103.cmake":
                self.assertNotIn("CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL", path.read_text(), str(path))

    def test_actual_cuda13_wrapper_branch_and_error_forwarding(self):
        cutlass = os.environ.get("CUTLASS_ROOT")
        if not cutlass:
            self.skipTest("set CUTLASS_ROOT to exercise the real local CUTLASS wrapper")
        header = (Path(cutlass) / "include/cutlass/cuda_host_adapter.hpp").read_text()
        begin = header.index("#define CUTLASS_CUDA_DRIVER_STRINGIFY(tok)")
        end = header.index("#if (__CUDACC_VER_MAJOR__ >= 12)", begin)
        wrapper = header[begin:end]
        declaration = "CUTLASS_CUDA_DRIVER_WRAPPER_DECL(cuTensorMapEncodeTiled, 12000);"
        self.assertIn(declaration, header[end:])
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        if not compiler or shutil.which(compiler[0]) is None:
            self.skipTest("a host C++ compiler is required")
        probe = r"""
#include <cstring>
#include <initializer_list>
using CUresult = int;
using cudaError_t = int;
enum cudaDriverEntryPointQueryResult { cudaDriverEntryPointSuccess, queryFailure };
enum { cudaSuccess = 0, cudaEnableDefault = 0, CUDA_ERROR_UNKNOWN = 999 };
int queries = 0, calls = 0, failure = 0;
CUresult cuTensorMapEncodeTiled(int* output, int value) {
  ++calls;
  *output = value;
  return value == 17 ? 0 : 73;
}
using PFN_cuTensorMapEncodeTiled_v12000 = decltype(&cuTensorMapEncodeTiled);
cudaError_t cudaGetDriverEntryPointByVersion(const char* name, void** result,
    int version, int flags, cudaDriverEntryPointQueryResult* status) {
  ++queries;
  if (std::strcmp(name, "cuTensorMapEncodeTiled") || version != 12000 || flags != 0) return 8;
  *status = failure == 2 ? queryFailure : cudaDriverEntryPointSuccess;
  *result = reinterpret_cast<void*>(&cuTensorMapEncodeTiled);
  return failure == 1 ? 7 : cudaSuccess;
}
""" + wrapper + declaration + r"""
int main() {
#if defined(CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL)
  constexpr bool direct = true;
#else
  constexpr bool direct = false;
#endif
  for (failure = 0; failure < 3; ++failure) {
    for (int value : {17, 29}) {
      int output = -1;
      int previous_calls = calls, previous_queries = queries;
      CUresult result = call_cuTensorMapEncodeTiled(&output, value);
      bool invoked = direct || failure == 0;
      if (queries != previous_queries + (direct ? 0 : 1) ||
          calls != previous_calls + (invoked ? 1 : 0) ||
          output != (invoked ? value : -1) ||
          result != (invoked ? (value == 17 ? 0 : 73) : CUDA_ERROR_UNKNOWN)) return 1;
    }
  }
}
"""
        # CUTLASS uses defined(), not the macro's truth value: even =0 selects
        # the direct branch. A is undefined; B is explicitly =1 in the build.
        with tempfile.TemporaryDirectory(prefix="fuse-driver-wrapper-test-") as directory:
            for name, defines in (("indirect", []), ("direct", ["-DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1"]),
                                  ("defined_zero", ["-DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=0"])):
                with self.subTest(branch=name):
                    binary = Path(directory) / name
                    result = subprocess.run(
                        [*compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                         "-D__CUDACC_VER_MAJOR__=13", *defines,
                         "-x", "c++", "-", "-o", str(binary)],
                        input=probe, text=True, capture_output=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    subprocess.run([str(binary)], check=True, timeout=10)


def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


@dataclass(frozen=True)
class StaticTiles:
    """Arithmetic equivalent of SM100's static one-CTA tile coordinates."""

    m_tiles: int
    n_tiles: int
    batches: int = 1
    raster: str = "n"
    max_swizzle: int = 1

    @property
    def swizzle(self):
        smaller = min(self.m_tiles, self.n_tiles)
        for width, minimum in ((8, 6), (4, 3), (2, 2)):
            if self.max_swizzle >= width and smaller >= minimum:
                return width
        return 1

    @property
    def padded(self):
        return (ceil_div(self.m_tiles, self.swizzle) * self.swizzle,
                ceil_div(self.n_tiles, self.swizzle) * self.swizzle)

    @property
    def total(self):
        m, n = self.padded
        return m * n * self.batches

    def decode(self, linear):
        if linear < 0 or linear >= self.total:
            raise ValueError("invalid work tile")
        padded_m, padded_n = self.padded
        batch, index = divmod(linear, padded_m * padded_n)
        major_size = padded_n if self.raster == "n" else padded_m
        extra, offset = divmod(index, self.swizzle)
        minor_group, major = divmod(extra, major_size)
        minor = minor_group * self.swizzle + offset
        m, n = (minor, major) if self.raster == "n" else (major, minor)
        return batch, m, n


def role_assignments(total_work, total_sms, comm_ctas):
    """A physical communication prefix followed by resident compute workers."""
    if total_work <= 0 or comm_ctas <= 0 or comm_ctas >= total_sms:
        raise ValueError("nonempty communication and compute budgets required")
    compute = min(total_work, total_sms - comm_ctas)
    return {
        physical: tuple(range(physical - comm_ctas, total_work, compute))
        for physical in range(comm_ctas, comm_ctas + compute)
    }


def valid_initial_worker(physical, comm_ctas, compute, total_work):
    return (comm_ctas >= 0 and compute > 0 and physical >= comm_ctas and
            physical - comm_ctas < compute and
            physical - comm_ctas < total_work)


def slot_tasks(task_count, comm_ctas, slots):
    """Independent warp slots can be used with arbitrary integer CTA counts."""
    return {
        (cta, slot): range(slot * comm_ctas + cta, task_count, comm_ctas * slots)
        for cta, slot in product(range(comm_ctas), range(slots))
    }


def arrival_task(task, m_tiles, world, chunks, m_window):
    """Bulk communication's peer-major windows, including its short last one."""
    window = min(m_window, m_tiles)
    full_windows, tail_tiles = divmod(m_tiles, window)
    per_window = window * world * chunks
    full_tasks = full_windows * per_window
    if task < full_tasks:
        window_id, inside = divmod(task, per_window)
        peer, inside_peer = divmod(inside, window * chunks)
        tile, chunk = divmod(inside_peer, chunks)
        return window_id * window + tile, peer, chunk
    peer, inside_peer = divmod(task - full_tasks, tail_tiles * chunks)
    tile, chunk = divmod(inside_peer, chunks)
    return full_windows * window + tile, peer, chunk


def arrival_rows(tile, chunk, comm_rows, total_rows):
    local_begin = chunk * comm_rows
    begin = tile * TILE_M + local_begin
    rows = max(0, min(comm_rows, TILE_M - local_begin, total_rows - begin))
    return range(begin, begin + rows)


def peer_segments(first_k, count, k_tiles_per_peer):
    """Calls to CUTLASS load split only at peer boundaries, retaining K order."""
    while count:
        peer, inside_peer = divmod(first_k, k_tiles_per_peer)
        take = min(count, k_tiles_per_peer - inside_peer)
        yield peer, range(first_k, first_k + take)
        first_k += take
        count -= take


def ready_index(batch, m, n, m_tiles, n_tiles):
    return ((batch * m_tiles + m) * n_tiles + n) * READY_STRIDE


def read_footprint(row, rows, column, columns, n_tiles, batch=0, m_tiles=1):
    """Communication waits for every producer tile intersecting its rectangle."""
    return {
        ready_index(batch, m, n, m_tiles, n_tiles)
        for m in range(row // TILE_M, (row + rows - 1) // TILE_M + 1)
        for n in range(column // TILE_N, (column + columns - 1) // TILE_N + 1)
    }


@dataclass(frozen=True)
class HeadChunk:
    source: int
    destination: int
    width: int
    count: int
    segment: int


def qkv_chunk(peer, chunk_id, world, q_heads, kv_heads, head_dim):
    q_local, kv_local = q_heads // world, kv_heads // world
    chunks_per_head = ceil_div(head_dim, TILE_N)
    head_slot, inner = divmod(chunk_id, chunks_per_head)
    offset = inner * TILE_N
    segment = 0 if head_slot < q_local else (1 if head_slot < q_local + kv_local else 2)
    local_head = head_slot - (0 if segment == 0 else q_local + (kv_local if segment == 2 else 0))
    local_heads = q_local if segment == 0 else kv_local
    segment_base = 0 if segment == 0 else q_heads + (kv_heads if segment == 2 else 0)
    return HeadChunk(
        (segment_base + peer * local_heads + local_head) * head_dim + offset,
        local_head * head_dim + offset,
        local_heads * head_dim,
        min(TILE_N, head_dim - offset),
        segment,
    )


@dataclass(frozen=True)
class QkvGeometry:
    producer_n: int
    m_tiles_per_task: int
    copy_n: int


# Producer tiles, vector-task grouping and copy chunks are separate geometries.
# These are the five supported GEMM->A2A bindings; A2A->GEMM stays N128.
QKV_GEOMETRIES = (
    QkvGeometry(64, 4, 128),
    QkvGeometry(128, 4, 128),
    QkvGeometry(160, 4, 160),
    QkvGeometry(192, 4, 192),
    QkvGeometry(256, 1, 256),
)


@dataclass(frozen=True)
class QkvShape:
    """Valid dense BF16 route geometry, not a tensor-map encoder model."""

    world: int
    head_dim: int
    seq_local: int
    batch: int = 1
    q_local_heads: int = 2
    kv_local_heads: int = 1
    defer_v: bool = False
    interleaved: bool = False

    @property
    def rows(self):
        return self.batch * self.seq_local

    @property
    def columns(self):
        return self.world * (self.q_local_heads + 2 * self.kv_local_heads) * self.head_dim

    @property
    def bulk(self):
        # The actual tensor-map path only supports a complete 128-wide head.
        return self.head_dim == 128


@dataclass(frozen=True)
class QkvCopyTask:
    work: int
    cta: int
    slot: int
    row: int
    rows: int
    column: int
    columns: int
    peer: int
    segment: int
    local_column: int


def qkv_family_tasks(shape, geometry, raster, comm_ctas):
    """Model the scalar CTA loop or the eight independent bulk warp loops."""
    if raster not in ("m", "n") or comm_ctas <= 0:
        raise ValueError("explicit raster and positive communication budget required")
    if shape.interleaved and (geometry.producer_n != 128 or raster != "m"):
        raise ValueError("peer-interleaved QKV has only the N128/AlongM binding")
    route_heads = shape.q_local_heads + (1 if shape.defer_v else 2) * shape.kv_local_heads
    chunks_per_head = ceil_div(shape.head_dim, geometry.copy_n)
    head_chunks = route_heads * chunks_per_head
    row_block = 64 if shape.bulk else TILE_M * geometry.m_tiles_per_task
    row_groups = ceil_div(shape.rows, row_block)
    task_count = shape.world * row_groups * head_chunks
    slots = 8 if shape.bulk else 1
    for (cta, slot), jobs in slot_tasks(task_count, comm_ctas, slots).items():
        for work in jobs:
            rank_work, peer = divmod(work, shape.world)
            if raster == "n":
                row_group, head_chunk = divmod(rank_work, head_chunks)
            else:
                head_chunk, row_group = divmod(rank_work, row_groups)
            head_slot, chunk = divmod(head_chunk, chunks_per_head)
            segment = (0 if head_slot < shape.q_local_heads else
                       1 if head_slot < shape.q_local_heads + shape.kv_local_heads else 2)
            local_head = head_slot - (0 if segment == 0 else
                                     shape.q_local_heads + (shape.kv_local_heads if segment == 2 else 0))
            local_heads = shape.q_local_heads if segment == 0 else shape.kv_local_heads
            physical_head = (local_head * shape.world + peer
                             if shape.interleaved and segment < 2 else
                             peer * local_heads + local_head)
            segment_heads = (0 if segment == 0 else
                             shape.q_local_heads + (shape.kv_local_heads if segment == 2 else 0))
            head_offset = chunk * geometry.copy_n
            row = row_group * row_block
            yield QkvCopyTask(
                work, cta, slot, row, min(row_block, shape.rows - row),
                (segment_heads * shape.world + physical_head) * shape.head_dim + head_offset,
                min(geometry.copy_n, shape.head_dim - head_offset), peer, segment,
                local_head * shape.head_dim + head_offset,
            )


def qkv_family_waits(shape, geometry, task):
    """One entry per waiting warp, not per lane, as in wait_acquire_system."""
    n_tiles = ceil_div(shape.columns, geometry.producer_n)
    first_n = task.column // geometry.producer_n
    last_n = (task.column + task.columns - 1) // geometry.producer_n
    if shape.bulk:
        return [((task.row // TILE_M) * n_tiles + n) * READY_STRIDE
                for n in range(first_n, last_n + 1)]
    producer_columns = last_n - first_n + 1
    m_group = task.row // (TILE_M * geometry.m_tiles_per_task)
    waits = []
    for warp in range(8):
        for item in range(warp, geometry.m_tiles_per_task * producer_columns, 8):
            local_m, local_n = divmod(item, producer_columns)
            tile_m = m_group * geometry.m_tiles_per_task + local_m
            if tile_m < ceil_div(shape.rows, TILE_M):
                waits.append((tile_m * n_tiles + first_n + local_n) * READY_STRIDE)
    return waits


def qkv_element_owner_oracle(shape, geometry, rows, columns):
    """Independent Cartesian projection of every read element onto its owner."""
    row_owners = {row // TILE_M for row in rows}
    column_owners = {column // geometry.producer_n for column in columns}
    n_tiles = ceil_div(shape.columns, geometry.producer_n)
    return {(m * n_tiles + n) * READY_STRIDE
            for m, n in product(row_owners, column_owners)}


def qkv_source_column_oracle(shape):
    """Start at physical columns, independently invert their rank/head layout."""
    result = {}
    segment_begin = 0
    for segment, local_heads in enumerate(
            (shape.q_local_heads, shape.kv_local_heads, shape.kv_local_heads)):
        segment_columns = shape.world * local_heads * shape.head_dim
        if not (shape.defer_v and segment == 2):
            for inside in range(segment_columns):
                physical_head, component = divmod(inside, shape.head_dim)
                if shape.interleaved and segment < 2:
                    local_head, peer = divmod(physical_head, shape.world)
                else:
                    peer, local_head = divmod(physical_head, local_heads)
                result[segment_begin + inside] = (
                    peer, segment, local_head * shape.head_dim + component)
        segment_begin += segment_columns
    return result


def qkv_task_destination(shape, task, source_rank, row, column_offset):
    """Linear destination address used by the communication task model."""
    batch, local_row = divmod(row, shape.seq_local)
    destination_row = batch * shape.world * shape.seq_local + source_rank * shape.seq_local + local_row
    q_width = shape.q_local_heads * shape.head_dim
    kv_width = shape.kv_local_heads * shape.head_dim
    local_column = task.local_column + column_offset
    if shape.defer_v:
        return (destination_row * (q_width + kv_width) +
                (q_width if task.segment == 1 else 0) + local_column)
    segment_offset = (0 if task.segment == 0 else
                      shape.rows * shape.world * (q_width + (kv_width if task.segment == 2 else 0)))
    width = q_width if task.segment == 0 else kv_width
    return segment_offset + destination_row * width + local_column


def qkv_destination_element_oracle(shape):
    """Enumerate the public destination layout in physical memory order.

    Unlike the task decoder, this has no copy chunks, GEMM tiles, raster,
    warp slots or linear-address formula. Each destination element names
    its unique (source rank, flattened source row, physical source column).
    Q/K/V are separate segments, or row-interleaved Q/K when V is deferred.
    Both layouts keep source ranks contiguous; QKV does not causal-scatter.
    """
    inverse = {target: source for source, target in qkv_source_column_oracle(shape).items()}
    segments = (0, 1) if shape.defer_v else (0, 1, 2)
    local_widths = tuple(heads * shape.head_dim for heads in
                         (shape.q_local_heads, shape.kv_local_heads, shape.kv_local_heads))
    row_coordinates = tuple(product(range(shape.batch), range(shape.world), range(shape.seq_local)))
    result = {}
    for peer in range(shape.world):
        if shape.defer_v:
            coordinates = ((segment, batch, rank, row, column)
                           for batch, rank, row in row_coordinates for segment in segments
                           for column in range(local_widths[segment]))
        else:
            coordinates = ((segment, batch, rank, row, column)
                           for segment in segments for batch, rank, row in row_coordinates
                           for column in range(local_widths[segment]))
        for address, (segment, batch, rank, row, column) in enumerate(coordinates):
            result[peer, address] = (
                rank, batch * shape.seq_local + row, inverse[peer, segment, column])
    return result


def qkv_v_completion_waits(shape, geometry):
    """V producer completion only; it is not the later global Q/K finalize."""
    n_tiles = ceil_div(shape.columns, geometry.producer_n)
    first_v = shape.world * (shape.q_local_heads + shape.kv_local_heads) * shape.head_dim
    first_n = first_v // geometry.producer_n
    v_tiles = n_tiles - first_n
    return [((item // v_tiles) * n_tiles + first_n + item % v_tiles) * READY_STRIDE
            for warp in range(8)
            for item in range(warp, ceil_div(shape.rows, TILE_M) * v_tiles, 8)]


class RoleGridContractTests(unittest.TestCase):
    def test_every_payload_and_padded_tile_has_exactly_one_worker(self):
        shapes = ((1, 1, 1), (127, 129, 1), (129, 257, 1),
                  (1024, 4096, 1), (769, 897, 3), (16385, 257, 2))
        for shape, raster, swizzle, comm in product(
                shapes, ("m", "n"), (1, 2, 4, 8), (1, 3, 12, 33, 147)):
            with self.subTest(shape=shape, raster=raster, swizzle=swizzle, comm=comm):
                m, n, batches = shape
                tiles = StaticTiles(ceil_div(m, TILE_M), ceil_div(n, TILE_N),
                                    batches, raster, swizzle)
                assignments = role_assignments(tiles.total, 148, comm)
                linear = [index for worker in assignments.values() for index in worker]
                self.assertEqual(Counter(linear), Counter(range(tiles.total)))
                observed = Counter(tiles.decode(index) for index in linear)
                padded_m, padded_n = tiles.padded
                expected = Counter(product(range(batches), range(padded_m), range(padded_n)))
                self.assertEqual(observed, expected)
                payload = {coord for coord in observed
                           if coord[1] < tiles.m_tiles and coord[2] < tiles.n_tiles}
                self.assertEqual(payload, set(product(
                    range(batches), range(tiles.m_tiles), range(tiles.n_tiles))))
                for physical, worker in assignments.items():
                    self.assertTrue(valid_initial_worker(
                        physical, comm, len(assignments), tiles.total))
                    self.assertTrue(worker)

    def test_communication_budget_does_not_change_payload_coverage(self):
        for sms, comm, work in product((8, 16, 148), (1, 3, 7), (1, 2, 37, 1021)):
            workers = role_assignments(work, sms, comm)
            self.assertLessEqual(len(workers) + comm, sms)
            self.assertEqual(min(workers), comm)
            self.assertTrue(all(physical >= comm for physical in workers))
            self.assertEqual(sorted(i for jobs in workers.values() for i in jobs), list(range(work)))

    def test_communication_and_initially_empty_workers_are_rejected(self):
        self.assertFalse(valid_initial_worker(2, 3, 4, 17))
        self.assertTrue(valid_initial_worker(3, 3, 4, 17))
        self.assertTrue(valid_initial_worker(6, 3, 4, 17))
        self.assertFalse(valid_initial_worker(7, 3, 4, 17))
        self.assertFalse(valid_initial_worker(4, 3, 10, 1))
        self.assertFalse(valid_initial_worker(3, 3, 0, 1))
        for work, sms, comm in ((0, 148, 12), (16, 8, 8), (16, 8, 0)):
            with self.assertRaises(ValueError):
                role_assignments(work, sms, comm)

    def test_original_physical_grid_stride_would_leave_holes(self):
        # Regression witness for the bug motivating the custom scheduler.
        total, comm, compute = 97, 3, 5
        wrong = {i for worker in range(compute)
                 for i in range(worker, total, compute + comm)}
        correct = {i for jobs in role_assignments(total, compute + comm, comm).values()
                   for i in jobs}
        self.assertNotEqual(wrong, set(range(total)))
        self.assertEqual(correct, set(range(total)))


class ArrivalContractTests(unittest.TestCase):
    def test_bulk_windows_and_arbitrary_cta_counts_cover_every_arrival(self):
        for rows, world, comm_rows, window, comm in product(
                (1, 127, 129, 513), (4, 8), (7, 32, 48, 128), (1, 2, 7), (3, 12)):
            with self.subTest(rows=rows, world=world, chunk=comm_rows, window=window, comm=comm):
                m_tiles = ceil_div(rows, TILE_M)
                chunks = ceil_div(TILE_M, comm_rows)
                tasks = m_tiles * world * chunks
                assigned = [task for jobs in slot_tasks(tasks, comm, 4).values() for task in jobs]
                decoded = [arrival_task(task, m_tiles, world, chunks, window) for task in assigned]
                self.assertEqual(Counter(decoded), Counter(product(
                    range(m_tiles), range(world), range(chunks))))
                arrivals = Counter((m, peer) for m, peer, _ in decoded)
                self.assertEqual(set(arrivals.values()), {chunks})

    def test_real_rows_copied_once_but_empty_tail_still_arrives(self):
        for rows, comm_rows in product((1, 31, 33, 127, 129, 255, 257), (7, 32, 48, 128)):
            m_tiles = ceil_div(rows, TILE_M)
            chunks = ceil_div(TILE_M, comm_rows)
            spans = [arrival_rows(m, chunk, comm_rows, rows)
                     for m, chunk in product(range(m_tiles), range(chunks))]
            self.assertEqual(Counter(row for span in spans for row in span), Counter(range(rows)))
            self.assertEqual(len(spans), m_tiles * chunks)
        tail = [arrival_rows(1, chunk, 32, 129) for chunk in range(4)]
        self.assertEqual([len(span) for span in tail], [1, 0, 0, 0])
        self.assertLess(sum(bool(span) for span in tail), 4,
                        "Dropping empty arrivals would keep the GEMM waiting forever.")

    def test_consecutive_epochs_reach_the_fixed_arrival_target(self):
        for comm_rows in (7, 32, 48, 128):
            chunks = ceil_div(TILE_M, comm_rows)
            count = 0
            for epoch in range(1, 6):
                target = epoch * chunks
                for _ in range(chunks):
                    self.assertLess(count, target)
                    count += 1
                self.assertEqual(count, target)
            maximum_epoch = UINT32_MAX // chunks
            self.assertLessEqual(maximum_epoch * chunks, UINT32_MAX)
            self.assertGreater((maximum_epoch + 1) * chunks, UINT32_MAX)

    def test_fixed_epoch_replay_cannot_reuse_the_same_ready_state(self):
        # A passing arithmetic test exposes this prohibited caller protocol;
        # it does not claim Graph replay has been implemented or validated.
        chunks, epoch = 4, 3
        old_count = chunks * epoch
        self.assertGreaterEqual(old_count, epoch * chunks)
        self.assertLess(old_count, (epoch + 1) * chunks)

    def test_prologue_and_remainder_wait_for_their_own_first_peer(self):
        for world, per_peer in product((1, 4, 8), (1, 2, 7, 32)):
            total = world * per_peer
            for prologue in sorted({0, 1, min(3, total), min(per_peer + 1, total), total}):
                calls = ((0, prologue), (prologue, total - prologue))
                segments = [segment for first, count in calls
                            for segment in peer_segments(first, count, per_peer)]
                self.assertEqual([k for _, indices in segments for k in indices], list(range(total)))
                for peer, indices in segments:
                    self.assertTrue(all(k // per_peer == peer for k in indices))
                if prologue < total:
                    first_peer, _ = next(peer_segments(prologue, total - prologue, per_peer))
                    self.assertEqual(first_peer, prologue // per_peer)

    def test_k64_and_k128_segments_never_load_across_an_unacquired_peer(self):
        for world, shard, tile_k in product((4, 8), (128, 384, 2048), (64, 128)):
            per_peer = shard // tile_k
            total = world * per_peer
            for prologue in sorted({1, min(3, total), min(per_peer + 1, total)}):
                segments = [segment for first, count in ((0, prologue), (prologue, total - prologue))
                            for segment in peer_segments(first, count, per_peer)]
                elements = [element for _, tiles in segments for tile in tiles
                            for element in range(tile * tile_k, (tile + 1) * tile_k)]
                self.assertEqual(elements, list(range(world * shard)))
                for peer, tiles in segments:
                    self.assertTrue(all(tile * tile_k >= peer * shard and
                                        (tile + 1) * tile_k <= (peer + 1) * shard for tile in tiles))


class GemmTileParameterContracts(unittest.TestCase):
    def test_e64_is_one_explicit_qkv_candidate_with_unchanged_ready_tile(self):
        gemm = (ROOT / "csrc/operators/sm103/detail/gemm.cuh").read_text()
        self.assertIn("EpilogueN != 64 || (BlockN == 256 && BlockK == 64)", gemm)
        self.assertIn("EpilogueN == 64, cute::Shape<cute::_128, cute::_64>", gemm)
        self.assertIn("Epilogue::DispatchPolicy::StagesD == 2", gemm)
        self.assertIn("OutputGemm::AccumulatorPipelineStageCount == 2", gemm)
        self.assertIn("PureGemm::AccumulatorPipelineStageCount == 2", gemm)
        # These are assertions on real instantiated CUDA types, not estimates
        # substituted into resource metadata. NVCC still has to execute them.
        self.assertIn("Mainloop::DispatchPolicy::Stages == 4", gemm)
        launch = (ROOT / "csrc/operators/sm103/detail/launch.cuh").read_text()
        self.assertIn("using QkvForwardN256K64E64Binding =\n"
                      "    GemmA2AKernelBinding<Bf16GemmTypes<256, 64, 64>, QkvGqaPackCommWide>;", launch)
        self.assertNotIn("OprojForwardN256K64E64Binding", launch)
        self.assertIn("using PureGemm = typename GemmTypes::PureGemm;", launch)
        traits = (ROOT / "csrc/operators/sm103/api/policy.cuh").read_text()
        self.assertIn("traits = kernel_traits<typename Binding::Kernel>();", traits)
        self.assertIn("visit_qkv_forward_policy(policy, route.qkv_peer_interleaved, read_traits)", traits)

    def test_e64_reduces_subtile_iterations_without_room_for_a_fifth_ab_stage(self):
        n, k, m, capacity = 256, 64, 128, 232448
        ab_stage_bytes = 2 * (m + n) * k
        for epilogue_n, subtiles, d_bytes in ((32, 8, 16384), (64, 4, 32768)):
            self.assertEqual(n // epilogue_n, subtiles)
            self.assertEqual(2 * m * epilogue_n * min(subtiles, 2), d_bytes)
            self.assertLess(4 * ab_stage_bytes + d_bytes, capacity)
            self.assertGreater(5 * ab_stage_bytes + d_bytes, capacity)
        self.assertEqual(min(4, 512 // n), 2)
        # Payload lower bounds only: actual alignment, barriers, registers and
        # resident occupancy are reported by the CUDA build/runtime checks.

    def test_real_family_threads_k_and_epilogue_through_one_compute_definition(self):
        source = (ROOT / "csrc/operators/sm103/detail/gemm.cuh").read_text()
        family = source[source.index("struct Bf16GemmTypes {"):]
        self.assertIn("static constexpr int kTileK = BlockK;", family)
        self.assertIn("EpilogueN == 32 || BlockN == 160", family)
        self.assertIn("using Dense = Bf16GemmTypes<BlockN, BlockK, EpilogueN, SwapAB>;", family)
        self.assertIn("using PureGemm = typename Dense::PureGemm;", family)
        # Source wiring guard only. Real collective types/resources still
        # require NVCC; the host policy probe intentionally uses opaque types.
        self.assertIn("ProblemShape, Mainloop, Epilogue, detail::MonolithicPersistentScheduler", family)

    def test_both_families_use_no_c_without_changing_bf16_or_float_compute(self):
        source = (ROOT / "csrc/operators/sm103/detail/gemm.cuh").read_text()
        family = source[source.index("struct Bf16GemmTypes {"):source.index("using N64TileShape")]
        self.assertIn("using Element = Bf16;", family)
        self.assertIn("using Accumulator = float;", family)
        self.assertIn("using ElementC = void;", family)
        self.assertIn("using OutputLayout = cute::conditional_t<SwapAB, cutlass::layout::ColumnMajor, LayoutD>;", family)
        self.assertIn("Accumulator, Accumulator,\n      ElementC, OutputLayout, kAlignment,\n"
                      "      Element, OutputLayout, kAlignment,", family)
        self.assertIn("Element, LayoutA, kAlignment,\n      Element, LayoutB, kAlignment,", family)
        self.assertIn("Mainloop::DispatchPolicy::Stages == 4", family)
        self.assertIn("OutputGemm::MaxThreadsPerBlock == 256", family)
        inverse = source[source.index("struct A2ALhsGemmTypes {"):]
        self.assertIn("using Dense = Bf16GemmTypes<BlockN, BlockK, EpilogueN, SwapAB>;", inverse)
        self.assertEqual(inverse.count("typename Dense::Epilogue"), 2)  # Fused + telemetry.
        for path, name in (("a2a_gemm.h", "A2AGemmParams"), ("gemm_a2a.h", "GemmA2AParams")):
            public = (ROOT / "include/fuse/operators/primitives" / path).read_text()
            params = public[public.index(f"struct {name} {{"):]
            params = params[:params.index("\n};")]
            self.assertIn("float alpha = 1.0f;", params)
            self.assertNotRegex(params, r"\b(beta|ptr_C)\b")

    def test_no_c_storage_does_not_imply_five_n256_k64_stages(self):
        # Lower bounds, excluding barriers, callback storage and alignment.
        # Fitting here is necessary, not proof of actual CUTLASS compilation.
        capacity = 232448  # fixed CUTLASS SM100 builder capacity
        candidates = ((128, 128, 3), (256, 64, 4), (256, 128, 2))
        for n, k, ab_stages in candidates:
            epilogue_n = 32
            epi_tiles = n // epilogue_n
            stages_d = min(epi_tiles, 2)
            epilogue = 128 * epilogue_n * 2 * stages_d
            stage = 2 * (128 + n) * k
            self.assertEqual(n % epilogue_n, 0)
            self.assertEqual(epilogue, 16 * 1024)
            self.assertLess(ab_stages * stage + epilogue, capacity)
            self.assertGreater((ab_stages + 1) * stage + epilogue, capacity)

    def test_ready_publication_keeps_full_completion_after_base_store(self):
        pipeline = (ROOT / "csrc/operators/sm103/detail/cutlass_pipeline.cuh").read_text()
        store = pipeline[pipeline.index("    auto states = Base::template store<ReuseTmem>("):]
        base = store.index("auto states = Base::template store<ReuseTmem>(")
        issuing_warp = store.index("if (epilogue_thread < 32)")
        drain = store.index("fuse::detail::tma_store_wait_all();")
        sync = store.index("__syncwarp();", drain)
        publish = store.index("fuse::detail::store_release_gpu(", sync)
        self.assertLess(base, issuing_warp)
        self.assertLess(issuing_warp, drain)
        self.assertLess(drain, sync)
        self.assertLess(sync, publish)
        common = (ROOT / "include/fuse/arch/common.cuh").read_text()
        wait = common[common.index("void tma_store_wait_all() {"):]
        wait = wait[:wait.index("\n}")]
        self.assertIn("cp.async.bulk.wait_group 0;", wait)
        self.assertNotIn("wait_group.read", wait)

    def test_cutlass_void_c_delays_only_subtiles_and_flushes_before_each_store_returns(self):
        cutlass = os.environ.get("CUTLASS_ROOT")
        if not cutlass:
            self.skipTest("set CUTLASS_ROOT to check the actual delayed-store source contract")
        root = Path(cutlass) / "include/cutlass/epilogue/collective"
        builder = (root / "builders/sm100_builder.inl").read_text()
        self.assertIn("constexpr bool DelayTmaStore = is_void_v<ElementC_>;", builder)
        epilogue = (root / "sm100_epilogue_tma_warpspecialized.hpp").read_text()
        flushes = list(re.finditer(
            r"if constexpr \(DelayTmaStore\)\s*{\s*// Issue TMA stores for the last subtile\s*"
            r"tma_store_fn\(epi_m_prev, epi_n_prev\);\s*}", epilogue))
        self.assertEqual(len(flushes), 2)  # Accumulator-pipeline and tensor overloads.
        first_return = epilogue.index("return cute::make_tuple(", flushes[0].end())
        invoke_loop = epilogue.index("epi_loop_fn(cst_callbacks);", flushes[0].end())
        self.assertLess(invoke_loop, first_return)
        self.assertLess(first_return, flushes[1].start())
        second_return = epilogue.index("return cute::make_tuple(", flushes[1].end())
        self.assertLess(second_return, epilogue.index("store_tail(", second_return))
        stores = list(re.finditer(r"auto tma_store_fn =", epilogue))
        self.assertEqual(len(stores), 2)
        for store, flush in zip(stores, flushes):
            issue = epilogue.index("copy(params.tma_store_d,", store.start())
            commit = epilogue.index("store_pipeline.producer_commit(", issue)
            self.assertLess(issue, commit)
            self.assertLess(commit, flush.start())


class TelemetryContractTests(unittest.TestCase):
    """Diagnostic index/lifetime models, not device timer or atomic tests."""

    def test_peer_allocation_covers_two_distinct_index_spaces(self):
        for m_tiles, n_tiles, world in product((1, 3, 17), (1, 5, 33), (1, 4, 8)):
            releases = {m * world + peer for m, peer in
                        product(range(m_tiles), range(world))}
            acquires = {m * n_tiles + n for m, n in
                        product(range(m_tiles), range(n_tiles))}
            capacity = max(m_tiles * world, m_tiles * n_tiles)
            self.assertEqual(releases, set(range(m_tiles * world)))
            self.assertEqual(acquires, set(range(m_tiles * n_tiles)))
            self.assertEqual(max(releases | acquires), capacity - 1)
            # An overlapping record is intentional; release and acquire
            # occupy different struct fields, not the same logical event.
            self.assertTrue(releases & acquires)

    def test_padded_n_must_not_alias_next_row_diagnostic(self):
        m_tiles, n_tiles = 3, 5
        self.assertEqual(0 * n_tiles + n_tiles, 1 * n_tiles + 0)
        valid = [(m, n) for m, n in product(range(4), range(8))
                 if m < m_tiles and n < n_tiles]
        self.assertEqual(len({m * n_tiles + n for m, n in valid}), 15)

    def test_cached_acquire_still_records_each_output_tile_first_observation(self):
        for world in (1, 4, 8):
            observed = {}
            acquired = None
            time = 1
            for n, prologue in product(range(3), (True, False)):
                # A repeated first peer across the prologue/remainder is
                # possible. The ready cache must not suppress telemetry.
                peers = [0] if prologue else list(range(world))
                for peer in peers:
                    if acquired != (0, peer):
                        acquired = (0, peer)
                    observed.setdefault((0, n, peer), time)
                    time += 1
            self.assertEqual(set(observed), set(product((0,), range(3), range(world))))
            self.assertEqual(observed[(0, 0, 0)], 1)

    def test_resetting_diagnostics_does_not_reset_cumulative_ready(self):
        chunks, last_epoch = 4, 9
        ready = chunks * last_epoch
        diagnostic = {"release": 100, "acquire": 101}
        diagnostic = dict.fromkeys(diagnostic, 0)
        self.assertEqual(diagnostic, {"release": 0, "acquire": 0})
        self.assertLess(ready, chunks * (last_epoch + 1))
        # A post-publication release sample may follow a consumer acquire;
        # software timestamps need not order as release_sample <= acquire.
        publish, acquire, release_sample = 100, 101, 102
        self.assertLess(publish, acquire)
        self.assertLess(acquire, release_sample)


class QkvReadyContractTests(unittest.TestCase):
    def test_publication_indices_do_not_alias_across_tiles_or_batches(self):
        for batches, m_tiles, n_tiles in ((1, 1, 1), (2, 3, 7), (4, 17, 5)):
            indices = [ready_index(b, m, n, m_tiles, n_tiles)
                       for b, m, n in product(range(batches), range(m_tiles), range(n_tiles))]
            self.assertEqual(indices, list(range(0, batches * m_tiles * n_tiles * READY_STRIDE, READY_STRIDE)))

    def test_all_intersecting_tiles_are_required_at_both_boundaries(self):
        footprint = read_footprint(127, 2, 120, 16, n_tiles=3, m_tiles=2)
        self.assertEqual(footprint, {0, READY_STRIDE, 3 * READY_STRIDE, 4 * READY_STRIDE})

    def test_qkv_head_chunks_cover_source_and_peer_segments_exactly_once(self):
        for world, head_dim in product((2, 4, 8), (8, 80, 128, 160, 256)):
            q_heads, kv_heads = 4 * world, world
            per_peer_chunks = 6 * ceil_div(head_dim, TILE_N)
            source_coverage = Counter()
            for peer in range(world):
                destination_coverage = Counter()
                for chunk_id in range(per_peer_chunks):
                    head = qkv_chunk(peer, chunk_id, world, q_heads, kv_heads, head_dim)
                    source_coverage.update(range(head.source, head.source + head.count))
                    destination_coverage.update((head.segment, col)
                                                for col in range(head.destination, head.destination + head.count))
                expected = Counter((segment, col) for segment, heads in enumerate((4, 1, 1))
                                   for col in range(heads * head_dim))
                self.assertEqual(destination_coverage, expected)
            self.assertEqual(source_coverage, Counter(range((q_heads + 2 * kv_heads) * head_dim)))

    def test_qkv_consumer_flags_match_elementwise_producer_owners(self):
        for world, head_dim in product((4, 8), (80, 128, 160)):
            q_heads, kv_heads = 4 * world, world
            n_tiles = ceil_div((q_heads + 2 * kv_heads) * head_dim, TILE_N)
            per_peer_chunks = 6 * ceil_div(head_dim, TILE_N)
            for peer, chunk, (row, rows) in product(
                    range(world), range(per_peer_chunks), ((0, 1), (127, 2), (250, 7))):
                head = qkv_chunk(peer, chunk, world, q_heads, kv_heads, head_dim)
                waited = read_footprint(row, rows, head.source, head.count, n_tiles, m_tiles=3)
                elementwise = {
                    ready_index(0, r // TILE_M, col // TILE_N, 3, n_tiles)
                    for r in range(row, row + rows)
                    for col in range(head.source, head.source + head.count)
                }
                self.assertEqual(waited, elementwise)


class QkvFamilyContractTests(unittest.TestCase):
    """Five-width arithmetic contracts, not compiled kernel/resource tests."""

    def assert_complete_route(self, shape, geometry, raster, comm_ctas):
        oracle = qkv_source_column_oracle(shape)
        self.assertEqual(len(set(oracle.values())), len(oracle))
        spans_by_column = {}
        waited_union = set()
        for task in qkv_family_tasks(shape, geometry, raster, comm_ctas):
            columns = range(task.column, task.column + task.columns)
            self.assertEqual([oracle.get(column) for column in columns],
                             [(task.peer, task.segment, task.local_column + offset)
                              for offset in range(task.columns)])
            for column in columns:
                spans_by_column.setdefault(column, []).append((task.row, task.row + task.rows))
            waits = qkv_family_waits(shape, geometry, task)
            expected_waits = qkv_element_owner_oracle(
                shape, geometry, range(task.row, task.row + task.rows), columns)
            self.assertEqual(Counter(waits), Counter(expected_waits))
            waited_union.update(waits)
        self.assertEqual(set(spans_by_column), set(oracle))
        # For every individual source column, its complete row intervals must
        # partition [0, M) without gaps/overlap. This proves exact elementwise
        # coverage without allocating the potentially huge dense M x N oracle.
        for spans in {tuple(sorted(spans)) for spans in spans_by_column.values()}:
            self.assertEqual(spans[0][0], 0)
            self.assertEqual(spans[-1][1], shape.rows)
            self.assertTrue(all(begin < end for begin, end in spans))
            self.assertTrue(all(left[1] == right[0] for left, right in zip(spans, spans[1:])))
        self.assertEqual(waited_union, qkv_element_owner_oracle(
            shape, geometry, range(shape.rows), oracle))

    def test_five_widths_cover_routes_once_and_wait_exact_producer_owners(self):
        for geometry, world, head_dim, raster, defer_v, seq in product(
                QKV_GEOMETRIES, (4, 8), (64, 128, 160, 192, 256, 320),
                ("m", "n"), (False, True), (3, 513)):
            with self.subTest(geometry=geometry, world=world, head_dim=head_dim,
                              raster=raster, defer_v=defer_v, seq=seq):
                shape = QkvShape(world, head_dim, seq, batch=2 if seq == 3 else 1,
                                 defer_v=defer_v)
                self.assert_complete_route(shape, geometry, raster, 3 if defer_v else 7)

    def test_interleaved_qk_and_standard_v_have_only_the_n128_along_m_binding(self):
        geometry = QKV_GEOMETRIES[1]
        for world, head_dim, defer_v in product((4, 8), (64, 128, 160, 192, 256, 320), (False, True)):
            # Two K/V heads per peer makes the K interleave nontrivial too.
            shape = QkvShape(world, head_dim, 65, batch=2, q_local_heads=4,
                             kv_local_heads=2, defer_v=defer_v, interleaved=True)
            with self.subTest(world=world, head_dim=head_dim, defer_v=defer_v):
                self.assert_complete_route(shape, geometry, "m", 3)
                with self.assertRaises(ValueError):
                    list(qkv_family_tasks(shape, geometry, "n", 3))
                for other in QKV_GEOMETRIES:
                    if other.producer_n != 128:
                        with self.assertRaises(ValueError):
                            list(qkv_family_tasks(shape, other, "m", 3))

    def test_each_bulk_slot_or_scalar_cta_gets_exactly_its_strided_tasks(self):
        for geometry, world, head_dim, raster, comm in product(
                QKV_GEOMETRIES, (4, 8), (128, 160), ("m", "n"), (1, 3, 12, 147)):
            shape = QkvShape(world, head_dim, 513)
            tasks = list(qkv_family_tasks(shape, geometry, raster, comm))
            m_tiles = ceil_div(shape.rows, TILE_M)
            row_groups = (ceil_div(shape.rows, 64) if shape.bulk else
                          ceil_div(m_tiles, geometry.m_tiles_per_task))
            heads = shape.q_local_heads + 2 * shape.kv_local_heads
            count = world * row_groups * heads * ceil_div(head_dim, geometry.copy_n)
            self.assertEqual(Counter(task.work for task in tasks), Counter(range(count)))
            slots = 8 if shape.bulk else 1
            for task in tasks:
                self.assertEqual(task.cta, task.work % comm)
                self.assertEqual(task.slot, (task.work // comm) % slots)

    def test_scalar_threads_cover_every_valid_vector_including_group_and_head_tails(self):
        for geometry, head_dim, rows in product(QKV_GEOMETRIES, (64, 160, 320), (1, 129, 513)):
            shape = QkvShape(4, head_dim, rows)
            tasks = list(qkv_family_tasks(shape, geometry, "m", 3))
            # First/last tasks and shortest head chunk cover full groups, short
            # M tails and the head remainder without expanding every big tile.
            selected = {tasks[0], tasks[-1], min(tasks, key=lambda task: task.columns)}
            for task in selected:
                self.assertGreater(task.rows, 0)
                self.assertLessEqual(task.row + task.rows, rows)
                self.assertEqual(task.column % 8, 0)
                self.assertEqual(task.columns % 8, 0)
                vectors_per_row = task.columns // 8
                count = task.rows * vectors_per_row
                indices = [index for thread in range(256) for index in range(thread, count, 256)]
                self.assertEqual(Counter(indices), Counter(range(count)))
                coordinates = {(task.row + index // vectors_per_row, index % vectors_per_row)
                               for index in indices}
                self.assertEqual(coordinates, set(product(
                    range(task.row, task.row + task.rows), range(vectors_per_row))))
            expected_tail = rows - ((rows - 1) // (TILE_M * geometry.m_tiles_per_task)) * (
                TILE_M * geometry.m_tiles_per_task)
            self.assertEqual(min(task.rows for task in tasks), expected_tail)

    def test_bulk_transfer_stays_64_by_128_independent_of_producer_geometry(self):
        for geometry, (batch, seq) in product(
                QKV_GEOMETRIES, ((1, 1), (1, 63), (1, 64), (1, 65), (2, 128), (2, 129), (64, 65))):
            shape = QkvShape(4, 128, seq, batch=batch)
            use_tma_store = shape.rows % 64 == 0 and seq % 64 == 0
            tasks = list(qkv_family_tasks(shape, geometry, "n", 3))
            for task in tasks:
                self.assertEqual(task.columns, 128)
                self.assertEqual(task.row % 64, 0)
                self.assertEqual(task.rows, min(64, shape.rows - task.row))
                self.assertEqual(task.row // TILE_M, (task.row + task.rows - 1) // TILE_M)
                if use_tma_store:
                    self.assertEqual(task.rows, 64)
                    # A 2D TMA store must not cross a local-sequence boundary:
                    # there is a gap to the next batch in this rank's output.
                    self.assertEqual(task.row // seq, (task.row + 63) // seq)
            if batch == 64 and seq == 65:
                self.assertEqual(shape.rows % 64, 0)
                self.assertFalse(use_tma_store,
                                 "A full M tile alone cannot authorize a contiguous TMA store.")

    def test_unaligned_copy_chunks_wait_for_two_or_three_producer_columns(self):
        witnesses = ((64, 128, 0, {0, 1}),
                     (160, 128, 128, {0, 1}),
                     (192, 128, 128, {0, 1}),
                     (64, 160, 160, {2, 3, 4}),
                     (128, 160, 160, {1, 2}),
                     (160, 192, 192, {1, 2}))
        for producer_n, head_dim, column, owners in witnesses:
            geometry = next(item for item in QKV_GEOMETRIES if item.producer_n == producer_n)
            shape = QkvShape(4, head_dim, 513)
            task = next(task for task in qkv_family_tasks(shape, geometry, "m", 3)
                        if task.row == 0 and task.column == column)
            n_tiles = ceil_div(shape.columns, producer_n)
            waits = qkv_family_waits(shape, geometry, task)
            self.assertEqual({flag // READY_STRIDE % n_tiles for flag in waits}, owners)
            self.assertEqual(Counter(waits), Counter(qkv_element_owner_oracle(
                shape, geometry, range(task.rows), range(column, column + task.columns))))

    def test_grouped_waits_never_wait_for_nonexistent_tail_producers(self):
        for geometry, rows in product(QKV_GEOMETRIES, (1, 127, 128, 129, 257, 513)):
            shape = QkvShape(4, 160, rows)
            n_tiles = ceil_div(shape.columns, geometry.producer_n)
            for task in qkv_family_tasks(shape, geometry, "n", 3):
                waits = qkv_family_waits(shape, geometry, task)
                waited_m = {flag // READY_STRIDE // n_tiles for flag in waits}
                self.assertEqual(waited_m, {row // TILE_M for row in range(task.row, task.row + task.rows)})
                self.assertLess(max(waited_m), ceil_div(rows, TILE_M))

    def test_deferred_v_waits_cover_v_producers_not_a_fabricated_remote_v_route(self):
        for geometry, world, head_dim in product(QKV_GEOMETRIES, (4, 8), (64, 128, 160, 192, 256, 320)):
            shape = QkvShape(world, head_dim, 129, defer_v=True)
            v_begin = world * (shape.q_local_heads + shape.kv_local_heads) * head_dim
            completion = qkv_v_completion_waits(shape, geometry)
            self.assertEqual(Counter(completion), Counter(qkv_element_owner_oracle(
                shape, geometry, range(shape.rows), range(v_begin, shape.columns))))
            qk_waits = set()
            for task in qkv_family_tasks(shape, geometry, "m", 3):
                self.assertLess(task.segment, 2)
                self.assertLessEqual(task.column + task.columns, v_begin)
                qk_waits.update(qkv_family_waits(shape, geometry, task))
            self.assertEqual(qk_waits | set(completion), qkv_element_owner_oracle(
                shape, geometry, range(shape.rows), range(shape.columns)))
        # A tile can straddle the QK/V boundary. Waiting for it from both
        # paths is correct and does not mean that V was remotely copied.
        geometry = QKV_GEOMETRIES[2]
        shape = QkvShape(4, 64, 129, defer_v=True)
        qk_waits = {flag for task in qkv_family_tasks(shape, geometry, "m", 3)
                    for flag in qkv_family_waits(shape, geometry, task)}
        self.assertTrue(qk_waits & set(qkv_v_completion_waits(shape, geometry)))

    def test_explicit_elementwise_destination_oracle_covers_all_source_ranks_and_batches(self):
        cases = [(geometry, QkvShape(4, head_dim, 3, batch=2, defer_v=defer_v))
                 for geometry, head_dim, defer_v in product(QKV_GEOMETRIES, (128, 160), (False, True))]
        cases += [(QKV_GEOMETRIES[1], QkvShape(8, 64, 2, batch=2, q_local_heads=4,
                                             kv_local_heads=2, defer_v=defer_v, interleaved=True))
                  for defer_v in (False, True)]
        for geometry, shape in cases:
            with self.subTest(geometry=geometry, shape=shape):
                actual = {}
                counts = Counter()
                raster = "n" if shape.bulk else "m"
                for task in qkv_family_tasks(shape, geometry, raster, 3):
                    for rank, row, offset in product(range(shape.world),
                                                     range(task.row, task.row + task.rows),
                                                     range(task.columns)):
                        destination = (task.peer, qkv_task_destination(shape, task, rank, row, offset))
                        counts[destination] += 1
                        actual[destination] = (rank, row, task.column + offset)
                self.assertEqual(set(counts.values()), {1})
                self.assertEqual(actual, qkv_destination_element_oracle(shape))

    def test_each_geometry_scheduler_publishes_every_actual_ready_tile_once(self):
        for geometry, raster, swizzle, comm in product(QKV_GEOMETRIES, ("m", "n"), (1, 8), (3, 147)):
            shape = QkvShape(8, 160, 513)
            m_tiles = ceil_div(shape.rows, TILE_M)
            n_tiles = ceil_div(shape.columns, geometry.producer_n)
            tiles = StaticTiles(m_tiles, n_tiles, raster=raster, max_swizzle=swizzle)
            published = Counter()
            for jobs in role_assignments(tiles.total, 148, comm).values():
                for work in jobs:
                    _, m, n = tiles.decode(work)
                    if m < m_tiles and n < n_tiles:
                        published[(m * n_tiles + n) * READY_STRIDE] += 1
            expected = Counter(range(0, m_tiles * n_tiles * READY_STRIDE, READY_STRIDE))
            self.assertEqual(published, expected)

    def test_family_mma_epilogue_and_accumulator_arithmetic_constraints(self):
        # CUTLASS 57e3cfb: 1SM BF16 MMA supports M64/128, N multiples of
        # eight <=256. Its no-C BF16 Auto epilogue targets N32; explicit
        # N160/e32 bindings retain N32. TMEM has 128x512 cells.
        # These arithmetic checks cannot establish actual SharedStorage
        # size, mainloop StageCount>=2, occupancy, or CUDA compilability.
        self.assertEqual([geometry.producer_n for geometry in QKV_GEOMETRIES], [64, 128, 160, 192, 256])
        for geometry, stages, raw_stage_kib in zip(QKV_GEOMETRIES, (4, 4, 3, 2, 2), (24, 32, 36, 40, 48)):
            n = geometry.producer_n
            self.assertIn(TILE_M, (64, 128))
            self.assertEqual(n % 8, 0)
            self.assertLessEqual(n, 256)
            self.assertEqual(TILE_K % 8, 0)
            epilogue_n = 32
            self.assertEqual(n % epilogue_n, 0)
            self.assertEqual(min(4, 512 // n), stages)
            self.assertLessEqual(stages * TILE_M * n, 128 * 512)
            self.assertEqual(2 * TILE_K * (TILE_M + n), raw_stage_kib * 1024)


if __name__ == "__main__":
    unittest.main()
