#!/usr/bin/env python3
"""PyTorch-autograd conformance for the two fused backward boundaries.

The production kernels are called through a tiny raw-pointer C ABI.  PyTorch
owns every allocation and computes the independent reference with
``torch.nn.functional.linear`` followed by native autograd.  Route references
use only index_select/cat/index_copy, so this test checks the public layout
contract in addition to DGrad, WGrad, and beta=1 accumulation.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

# cuBLAS needs this setting before its first handle is created in order to
# guarantee deterministic GEMM accumulation.  The native reference and the
# fused kernels can then be compared repeatedly without an algorithm-change
# false positive.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F


PTR = ctypes.c_uint64
PTR_ARRAY = ctypes.POINTER(PTR)


@dataclass(frozen=True)
class Case:
    name: str
    world: int
    local_tokens: int
    hidden: int
    batch: int
    q_heads: int
    kv_heads: int
    head_dim: int
    causal: bool

    @property
    def global_tokens(self) -> int:
        return self.local_tokens * self.world

    @property
    def qkv_width(self) -> int:
        return (self.q_heads + 2 * self.kv_heads) * self.head_dim

    @property
    def attention_width(self) -> int:
        return self.q_heads * self.head_dim


CASES = (
    Case("cp4_rank_major", 4, 16, 256, 1, 16, 8, 128, False),
    Case("cp8_causal", 8, 16, 256, 1, 16, 8, 128, True),
    Case("cp4_batch2_causal", 4, 32, 256, 2, 16, 8, 128, True),
    Case("cp8_wide_rank_major", 8, 16, 512, 1, 40, 8, 128, False),
)


class Bridge:
    def __init__(self, path: Path) -> None:
        self.library = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        self.library.fuse_backward_bridge_last_error.restype = ctypes.c_char_p
        self.library.fuse_backward_bridge_enable_peer_access.argtypes = [ctypes.c_int]
        self.library.fuse_backward_bridge_enable_peer_access.restype = ctypes.c_int

        self.library.fuse_backward_bridge_qkv_ready_elements.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.library.fuse_backward_bridge_qkv_ready_elements.restype = ctypes.c_int64
        self.library.fuse_backward_bridge_oproj_ready_elements.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.library.fuse_backward_bridge_oproj_ready_elements.restype = ctypes.c_int64

        self.library.fuse_backward_bridge_qkv_data.argtypes = [
            ctypes.c_int,
            PTR,
            PTR,
            PTR,
            PTR_ARRAY,
            PTR_ARRAY,
            PTR_ARRAY,
            PTR,
            PTR,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint32,
            ctypes.c_int,
            PTR,
        ]
        self.library.fuse_backward_bridge_qkv_data.restype = ctypes.c_int
        self.library.fuse_backward_bridge_qkv_weight.argtypes = [
            ctypes.c_int,
            PTR,
            PTR,
            PTR,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            PTR,
        ]
        self.library.fuse_backward_bridge_qkv_weight.restype = ctypes.c_int

        self.library.fuse_backward_bridge_oproj_data.argtypes = [
            ctypes.c_int,
            PTR,
            PTR,
            PTR,
            PTR_ARRAY,
            PTR_ARRAY,
            PTR,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint32,
            ctypes.c_int,
            PTR,
        ]
        self.library.fuse_backward_bridge_oproj_data.restype = ctypes.c_int
        self.library.fuse_backward_bridge_oproj_weight.argtypes = [
            ctypes.c_int,
            PTR,
            PTR,
            PTR,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            PTR,
        ]
        self.library.fuse_backward_bridge_oproj_weight.restype = ctypes.c_int

    def check(self, status: int, operation: str) -> None:
        if status:
            raw = self.library.fuse_backward_bridge_last_error()
            detail = raw.decode() if raw else f"CUDA status {status}"
            raise RuntimeError(f"{operation}: {detail}")

    def enable_peer_access(self, world: int) -> None:
        self.check(
            self.library.fuse_backward_bridge_enable_peer_access(world),
            "enable peer access",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bridge",
        type=Path,
        required=True,
        help="path to libfuse_backward_torch_bridge.so",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(case.name for case in CASES),
        help="run only selected cases; repeat the option to select more than one",
    )
    return parser.parse_args()


def values(shape: tuple[int, ...], seed: int, device: int) -> torch.Tensor:
    elements = math.prod(shape)
    with torch.cuda.device(device):
        base = torch.arange(elements, dtype=torch.int64, device=device)
        data = ((base * 17 + seed * 23) % 13 - 6).to(torch.float32) / 64.0
        return data.to(torch.bfloat16).view(shape)


def addresses(tensors: list[torch.Tensor]) -> ctypes.Array[PTR]:
    return (PTR * len(tensors))(*(tensor.data_ptr() for tensor in tensors))


def stream_address(device: int) -> int:
    return int(torch.cuda.current_stream(device).cuda_stream)


def synchronize(world: int) -> None:
    for device in range(world):
        torch.cuda.synchronize(device)


def global_rows(case: Case, owner: int, device: int) -> torch.Tensor:
    sequence_local = case.local_tokens // case.batch
    rows: list[int] = []
    for local_row in range(case.local_tokens):
        batch_index, row = divmod(local_row, sequence_local)
        if not case.causal:
            global_row = (
                batch_index * sequence_local * case.world
                + owner * sequence_local
                + row
            )
        else:
            chunk_rows = sequence_local // 2
            chunk = owner if row < chunk_rows else 2 * case.world - owner - 1
            global_row = (
                batch_index * sequence_local * case.world
                + chunk * chunk_rows
                + row % chunk_rows
            )
        rows.append(global_row)
    return torch.tensor(rows, dtype=torch.int64, device=device)


def compare(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    label: str,
    atol: float,
) -> dict[str, float | int | str]:
    difference = (actual.float() - expected.float()).abs()
    max_abs = float(difference.max().item()) if difference.numel() else 0.0
    mismatches = int(torch.count_nonzero(difference > atol).item())
    if mismatches:
        raise AssertionError(
            f"{label}: mismatches={mismatches}, max_abs={max_abs}, atol={atol}"
        )
    return {
        "tensor": label,
        "elements": actual.numel(),
        "mismatches": mismatches,
        "max_abs": max_abs,
        "atol": atol,
    }


def qkv_torch_reference(case: Case) -> dict[str, list[torch.Tensor]]:
    q_local = case.q_heads // case.world * case.head_dim
    kv_local = case.kv_heads // case.world * case.head_dim
    grad_q = [
        values((case.global_tokens, q_local), 101 + rank, rank)
        for rank in range(case.world)
    ]
    grad_k = [
        values((case.global_tokens, kv_local), 201 + rank, rank)
        for rank in range(case.world)
    ]
    grad_v = [
        values((case.global_tokens, kv_local), 301 + rank, rank)
        for rank in range(case.world)
    ]
    saved_input = [
        values((case.local_tokens, case.hidden), 401 + rank, rank)
        .detach()
        .requires_grad_(True)
        for rank in range(case.world)
    ]
    weight = [
        values((case.qkv_width, case.hidden), 501 + rank, rank)
        .detach()
        .requires_grad_(True)
        for rank in range(case.world)
    ]

    dqkv: list[torch.Tensor] = []
    dx: list[torch.Tensor] = []
    dw: list[torch.Tensor] = []
    for owner in range(case.world):
        rows = global_rows(case, owner, owner)
        q_parts = [tensor.index_select(0, rows.to(rank)).to(owner) for rank, tensor in enumerate(grad_q)]
        k_parts = [tensor.index_select(0, rows.to(rank)).to(owner) for rank, tensor in enumerate(grad_k)]
        v_parts = [tensor.index_select(0, rows.to(rank)).to(owner) for rank, tensor in enumerate(grad_v)]
        local_dqkv = torch.cat((*q_parts, *k_parts, *v_parts), dim=1).contiguous()
        output = F.linear(saved_input[owner], weight[owner])
        local_dx, local_dw = torch.autograd.grad(
            output,
            (saved_input[owner], weight[owner]),
            grad_outputs=local_dqkv,
        )
        dqkv.append(local_dqkv)
        dx.append(local_dx)
        dw.append(local_dw)
    return {
        "grad_q": grad_q,
        "grad_k": grad_k,
        "grad_v": grad_v,
        "saved_input": [tensor.detach() for tensor in saved_input],
        "weight": [tensor.detach() for tensor in weight],
        "dqkv": dqkv,
        "dx": dx,
        "dw": dw,
    }


def run_qkv(
    bridge: Bridge,
    case: Case,
    mode: str,
) -> list[dict[str, float | int | str]]:
    reference = qkv_torch_reference(case)
    ready_elements = bridge.library.fuse_backward_bridge_qkv_ready_elements(
        case.local_tokens,
        case.hidden,
        case.batch,
        case.q_heads,
        case.kv_heads,
        case.head_dim,
        case.world,
        8,
        int(case.causal),
    )
    if ready_elements <= 0:
        raise RuntimeError(f"invalid QKV ready size: {ready_elements}")
    staging = [
        torch.full(
            (case.local_tokens, case.qkv_width),
            float("nan"),
            dtype=torch.bfloat16,
            device=rank,
        )
        for rank in range(case.world)
    ]
    grad_input = [
        torch.full(
            (case.local_tokens, case.hidden),
            float("nan"),
            dtype=torch.bfloat16,
            device=rank,
        )
        for rank in range(case.world)
    ]
    ready = [
        torch.zeros(ready_elements, dtype=torch.int32, device=rank)
        for rank in range(case.world)
    ]
    done = [
        torch.zeros(case.world * 32, dtype=torch.int32, device=rank)
        for rank in range(case.world)
    ]
    if mode == "immediate":
        grad_weight = [
            torch.full_like(reference["weight"][rank], float("nan"))
            for rank in range(case.world)
        ]
    else:
        grad_weight = [
            values(reference["weight"][rank].shape, 901 + rank, rank)
            for rank in range(case.world)
        ]
    initial_weight = [tensor.clone() for tensor in grad_weight]
    staging_addresses = addresses(staging)
    ready_addresses = addresses(ready)
    done_addresses = addresses(done)

    for rank in range(case.world):
        status = bridge.library.fuse_backward_bridge_qkv_data(
            rank,
            reference["grad_q"][rank].data_ptr(),
            reference["grad_k"][rank].data_ptr(),
            reference["grad_v"][rank].data_ptr(),
            staging_addresses,
            ready_addresses,
            done_addresses,
            reference["weight"][rank].data_ptr(),
            grad_input[rank].data_ptr(),
            case.local_tokens,
            case.hidden,
            case.batch,
            case.q_heads,
            case.kv_heads,
            case.head_dim,
            case.world,
            rank,
            8,
            1,
            int(case.causal),
            stream_address(rank),
        )
        bridge.check(status, f"QKV data rank {rank}")
        if mode == "immediate":
            status = bridge.library.fuse_backward_bridge_qkv_weight(
                rank,
                staging[rank].data_ptr(),
                reference["saved_input"][rank].data_ptr(),
                grad_weight[rank].data_ptr(),
                case.local_tokens,
                case.hidden,
                case.q_heads,
                case.kv_heads,
                case.head_dim,
                0.0,
                stream_address(rank),
            )
            bridge.check(status, f"QKV immediate weight rank {rank}")
    synchronize(case.world)

    checks: list[dict[str, float | int | str]] = []
    for rank in range(case.world):
        checks.append(compare(staging[rank], reference["dqkv"][rank], label=f"qkv/{mode}/rank{rank}/route", atol=0.0))
        checks.append(compare(grad_input[rank], reference["dx"][rank], label=f"qkv/{mode}/rank{rank}/dX", atol=0.125))

    if mode == "deferred":
        for rank in range(case.world):
            status = bridge.library.fuse_backward_bridge_qkv_weight(
                rank,
                staging[rank].data_ptr(),
                reference["saved_input"][rank].data_ptr(),
                grad_weight[rank].data_ptr(),
                case.local_tokens,
                case.hidden,
                case.q_heads,
                case.kv_heads,
                case.head_dim,
                1.0,
                stream_address(rank),
            )
            bridge.check(status, f"QKV deferred weight rank {rank}")
        synchronize(case.world)
        for rank in range(case.world):
            expected = (initial_weight[rank].float() + reference["dw"][rank].float()).to(torch.bfloat16)
            checks.append(compare(grad_weight[rank], expected, label=f"qkv/{mode}/rank{rank}/main_grad_once", atol=0.25))
        for rank in range(case.world):
            status = bridge.library.fuse_backward_bridge_qkv_weight(
                rank,
                staging[rank].data_ptr(),
                reference["saved_input"][rank].data_ptr(),
                grad_weight[rank].data_ptr(),
                case.local_tokens,
                case.hidden,
                case.q_heads,
                case.kv_heads,
                case.head_dim,
                1.0,
                stream_address(rank),
            )
            bridge.check(status, f"QKV second accumulated weight rank {rank}")
        synchronize(case.world)
        for rank in range(case.world):
            expected = (
                initial_weight[rank].float() + 2.0 * reference["dw"][rank].float()
            ).to(torch.bfloat16)
            checks.append(compare(grad_weight[rank], expected, label=f"qkv/{mode}/rank{rank}/main_grad_twice", atol=0.25))
    else:
        for rank in range(case.world):
            checks.append(compare(grad_weight[rank], reference["dw"][rank], label=f"qkv/{mode}/rank{rank}/dW", atol=0.125))
    return checks


def oproj_torch_reference(case: Case) -> dict[str, list[torch.Tensor]]:
    local_width = case.attention_width // case.world
    head_attention = [
        values((case.global_tokens, local_width), 601 + rank, rank)
        for rank in range(case.world)
    ]
    grad_output = [
        values((case.local_tokens, case.hidden), 701 + rank, rank)
        for rank in range(case.world)
    ]
    weight = [
        values((case.hidden, case.attention_width), 801 + rank, rank)
        .detach()
        .requires_grad_(True)
        for rank in range(case.world)
    ]
    saved_attention: list[torch.Tensor] = []
    local_da: list[torch.Tensor] = []
    dw: list[torch.Tensor] = []
    for owner in range(case.world):
        rows = global_rows(case, owner, owner)
        parts = [tensor.index_select(0, rows.to(rank)).to(owner) for rank, tensor in enumerate(head_attention)]
        attention = torch.cat(parts, dim=1).contiguous().detach().requires_grad_(True)
        output = F.linear(attention, weight[owner])
        da, local_dw = torch.autograd.grad(
            output,
            (attention, weight[owner]),
            grad_outputs=grad_output[owner],
        )
        saved_attention.append(attention.detach())
        local_da.append(da)
        dw.append(local_dw)

    peer_da = [
        torch.empty(
            (case.global_tokens, local_width),
            dtype=torch.bfloat16,
            device=rank,
        )
        for rank in range(case.world)
    ]
    for destination in range(case.world):
        expected = torch.empty_like(peer_da[destination])
        for owner in range(case.world):
            rows = global_rows(case, owner, destination)
            shard = local_da[owner][
                :, destination * local_width : (destination + 1) * local_width
            ].to(destination)
            expected.index_copy_(0, rows, shard)
        peer_da[destination] = expected
    return {
        "head_attention": head_attention,
        "grad_output": grad_output,
        "weight": [tensor.detach() for tensor in weight],
        "saved_attention": saved_attention,
        "local_da": local_da,
        "peer_da": peer_da,
        "dw": dw,
    }


def run_oproj(
    bridge: Bridge,
    case: Case,
    mode: str,
) -> list[dict[str, float | int | str]]:
    reference = oproj_torch_reference(case)
    ready_elements = bridge.library.fuse_backward_bridge_oproj_ready_elements(
        case.local_tokens,
        case.hidden,
        case.batch,
        case.q_heads,
        case.head_dim,
        case.world,
        8,
        int(case.causal),
    )
    if ready_elements <= 0:
        raise RuntimeError(f"invalid OProj ready size: {ready_elements}")
    local_da = [
        torch.full_like(reference["saved_attention"][rank], float("nan"))
        for rank in range(case.world)
    ]
    peer_da = [
        torch.full_like(reference["peer_da"][rank], float("nan"))
        for rank in range(case.world)
    ]
    ready = [
        torch.zeros(ready_elements, dtype=torch.int32, device=rank)
        for rank in range(case.world)
    ]
    done = [
        torch.zeros(case.world * 32, dtype=torch.int32, device=rank)
        for rank in range(case.world)
    ]
    if mode == "immediate":
        grad_weight = [
            torch.full_like(reference["weight"][rank], float("nan"))
            for rank in range(case.world)
        ]
    else:
        grad_weight = [
            values(reference["weight"][rank].shape, 1001 + rank, rank)
            for rank in range(case.world)
        ]
    initial_weight = [tensor.clone() for tensor in grad_weight]
    peer_da_addresses = addresses(peer_da)
    done_addresses = addresses(done)

    for rank in range(case.world):
        status = bridge.library.fuse_backward_bridge_oproj_data(
            rank,
            reference["grad_output"][rank].data_ptr(),
            reference["weight"][rank].data_ptr(),
            local_da[rank].data_ptr(),
            peer_da_addresses,
            done_addresses,
            ready[rank].data_ptr(),
            case.local_tokens,
            case.hidden,
            case.batch,
            case.q_heads,
            case.head_dim,
            case.world,
            rank,
            8,
            1,
            int(case.causal),
            stream_address(rank),
        )
        bridge.check(status, f"OProj data rank {rank}")
        if mode == "immediate":
            status = bridge.library.fuse_backward_bridge_oproj_weight(
                rank,
                reference["grad_output"][rank].data_ptr(),
                reference["saved_attention"][rank].data_ptr(),
                grad_weight[rank].data_ptr(),
                case.local_tokens,
                case.hidden,
                case.q_heads,
                case.head_dim,
                0.0,
                stream_address(rank),
            )
            bridge.check(status, f"OProj immediate weight rank {rank}")
    synchronize(case.world)

    checks: list[dict[str, float | int | str]] = []
    for rank in range(case.world):
        checks.append(compare(local_da[rank], reference["local_da"][rank], label=f"oproj/{mode}/rank{rank}/local_dA", atol=0.125))
        checks.append(compare(peer_da[rank], reference["peer_da"][rank], label=f"oproj/{mode}/rank{rank}/route", atol=0.125))

    if mode == "deferred":
        for rank in range(case.world):
            status = bridge.library.fuse_backward_bridge_oproj_weight(
                rank,
                reference["grad_output"][rank].data_ptr(),
                reference["saved_attention"][rank].data_ptr(),
                grad_weight[rank].data_ptr(),
                case.local_tokens,
                case.hidden,
                case.q_heads,
                case.head_dim,
                1.0,
                stream_address(rank),
            )
            bridge.check(status, f"OProj deferred weight rank {rank}")
        synchronize(case.world)
        for rank in range(case.world):
            expected = (initial_weight[rank].float() + reference["dw"][rank].float()).to(torch.bfloat16)
            checks.append(compare(grad_weight[rank], expected, label=f"oproj/{mode}/rank{rank}/main_grad_once", atol=0.25))
        for rank in range(case.world):
            status = bridge.library.fuse_backward_bridge_oproj_weight(
                rank,
                reference["grad_output"][rank].data_ptr(),
                reference["saved_attention"][rank].data_ptr(),
                grad_weight[rank].data_ptr(),
                case.local_tokens,
                case.hidden,
                case.q_heads,
                case.head_dim,
                1.0,
                stream_address(rank),
            )
            bridge.check(status, f"OProj second accumulated weight rank {rank}")
        synchronize(case.world)
        for rank in range(case.world):
            expected = (
                initial_weight[rank].float() + 2.0 * reference["dw"][rank].float()
            ).to(torch.bfloat16)
            checks.append(compare(grad_weight[rank], expected, label=f"oproj/{mode}/rank{rank}/main_grad_twice", atol=0.25))
    else:
        for rank in range(case.world):
            checks.append(compare(grad_weight[rank], reference["dw"][rank], label=f"oproj/{mode}/rank{rank}/dW", atol=0.125))
    return checks


def main() -> None:
    args = parse_args()
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    selected = set(args.case or (case.name for case in CASES))
    cases = [case for case in CASES if case.name in selected]
    if torch.cuda.device_count() < max(case.world for case in cases):
        raise RuntimeError("the selected conformance cases require eight CUDA devices")
    bridge = Bridge(args.bridge.resolve())
    bridge.enable_peer_access(max(case.world for case in cases))
    results: list[dict[str, object]] = []
    for case in cases:
        for operator, runner in (("qkv", run_qkv), ("oproj", run_oproj)):
            for mode in ("immediate", "deferred"):
                checks = runner(bridge, case, mode)
                maximum = max(float(check["max_abs"]) for check in checks)
                results.append(
                    {
                        "case": case.name,
                        "operator": operator,
                        "mode": mode,
                        "status": "pass",
                        "max_abs": maximum,
                        "checks": checks,
                    }
                )
                print(
                    f"PASS {case.name:20s} {operator:5s} {mode:9s} "
                    f"checks={len(checks):2d} max_abs={maximum:.6f}",
                    flush=True,
                )
    payload = {
        "schema": "v10_backward_torch_autograd_v1",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "tf32": False,
        "reference": "torch.nn.functional.linear forward + torch.autograd.grad; native torch route construction",
        "cases": [asdict(case) for case in cases],
        "results": results,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.json_out.with_suffix(args.json_out.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(args.json_out)


if __name__ == "__main__":
    main()
