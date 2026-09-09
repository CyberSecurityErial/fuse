#!/usr/bin/env python3
"""Offline SM103 BF16 candidate experiment; never a runtime selector.

Consumes the JSON/CSV pair emitted by summarize_sm103_fused.py. The upstream
summary owns raw-artifact/correctness auditing; this module checks its model
input contract and records the exact input hashes, without reopening GPU logs.
Only synthetic tests may fit before the separately authorized calibration run.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import re
from types import MappingProxyType


DIRECTIONS = ("GEMM_A2A", "A2A_GEMM")
TILES = {"GEMM_A2A": (64, 128, 160, 192, 256), "A2A_GEMM": (128, 256)}
REGULARIZATIONS = (0.01, 0.1, 1.0, 10.0)
MODEL_FORMS = ("log_ridge", "nonnegative_time_ridge")
CALIBRATION_SEQUENCES = (131072, 262144)
CALIBRATION_COMMS = (8, 12, 16, 24, 32, 48)
# Audit the pre-registered experiment, not an input to features or prediction.
CALIBRATION_FAMILIES = ((2048, 16, 8, 128), (4096, 32, 8, 128), (16384, 128, 8, 128))
FEATURES = (
    ("wave_flops", "FLOP/worker", "W * 2 * 128 * tile_n * Kp"),
    ("wave_tile_bytes", "bytes/worker", "W * (2*Kp*(128+tile_n) + 2*128*tile_n); traffic proxy, not measured HBM bytes"),
    ("route_waves", "task waves", "ceil(route_tasks / (active_slots * comm_ctas))"),
    ("remote_bytes", "bytes/rank", "2*M*(QKV:N, OProj:K)*(CP-1)/CP"),
    ("comm_compute_ratio", "dimensionless", "comm_ctas / actual_compute_ctas"),
    ("ready_geometry", "direction-specific", "QKV: mean producer flags per 64x128 copy; OProj: inverse M-window width"),
)
SCOPE = {
    "precision": "bf16_accfp32_bf16", "runtime_cc": "10.3", "sm_count": 148,
    "threads": 256, "cluster_ctas": 1, "swizzle": 1, "batch": 1,
    "head_dim": 128, "global_sequence_range": [65536, 524288],
    "causal_local_m_multiple": 256, "oproj_copy": "bulk_only",
    "layout": {"GEMM_A2A": "qkv_source_rank_major_v1", "A2A_GEMM": "causal_dual_chunk_v1"},
    "not_claimed": "No arbitrary-shape, short-sequence, other-precision, Graph, or runtime-selector validation",
}
CSV_COLUMNS = (
    "run_id,node,source_id,binary_sha256,environment_fingerprint,direction,component,measurement_role,"
    "world,global_seq,seq_local,hidden,q_heads,kv_heads,head_dim,m,n,k,layout,host_launch,launch,collector,"
    "precision,sm_count,candidate,comm_ctas,tile_policy,tile_m,tile_n,tile_k,problem_gemm_flops,"
    "problem_route_payload_bytes,problem_remote_payload_bytes,executed_gemm_flops,executed_route_payload_bytes,"
    "p50_ms,p95_ms,half_drift"
).split(",")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ceil_div(a, b):
    return (a + b - 1) // b


def integer(row, key):
    value = row.get(key)
    require(type(value) is int and value > 0, f"{key} must be a positive integer")
    return value


def positive(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
            f"Invalid positive finite {name}")
    return float(value)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def hash_value(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "Invalid provenance hash")
    return value


def problem_key(row):
    """Physical candidate group; OProj does not depend on KV metadata."""
    base = (row["direction"], row["global_seq"], row["world"], row["m"], row["n"], row["k"], row["layout"])
    return base + ((row["q_heads"], row["kv_heads"], row["head_dim"]) if row["direction"] == "GEMM_A2A" else ())


def family_key(row):
    """No sequence, CP, tile, comm, model name, rank, or random seed."""
    if row["direction"] == "GEMM_A2A":
        return (row["hidden"], row["q_heads"] * row["head_dim"], row["kv_heads"] * row["head_dim"], row["head_dim"])
    return (row["hidden"], row["k"])


def candidate_key(row):
    return problem_key(row) + (row["tile_n"], row["comm_ctas"])


def physical_features(row):
    """Six exact deterministic proxies. No C/R times or model/tile categories.

    T=ceil(M/128)*ceil(N/Bn), G=min(T,SM-c), W=ceil(T/G), Kp=ceil(K/64)*64.
    QKV uses 8 independent 64x128 copy slots; MTilesPerTask affects only the
    excluded vector fallback. OProj uses four 48-KiB bulk slots, cumulative
    M128/peer arrivals, and the production consumer-derived M window.
    """
    direction = row.get("direction")
    require(direction in DIRECTIONS, "Unsupported direction")
    m, n, k, world, hidden, qh, kvh, dim, sm, comm, bn = (
        integer(row, key) for key in ("m", "n", "k", "world", "hidden", "q_heads", "kv_heads",
                                     "head_dim", "sm_count", "comm_ctas", "tile_n"))
    sequence = integer(row, "global_seq")
    require(world in (4, 8) and sm == 148 and comm < sm, "Unsupported CP/SM/communication budget")
    require(row.get("seq_local") == m and sequence == m * world and 65536 <= sequence <= 524288,
            "Unsupported long-sequence geometry")
    require(m % 256 == 0 and dim == 128 and hidden % 8 == 0 and
            qh % world == kvh % world == qh % kvh == 0, "Unsupported causal BF16 head/row geometry")
    require(bn in TILES[direction] and row.get("tile_m") == 128 and row.get("tile_k") == 64 and
            row.get("tile_policy") == f"m128n{bn}", "Unsupported or inconsistent actual tile")
    require(row.get("cluster_ctas") == row.get("swizzle") == 1 and row.get("batch", 1) == 1,
            "Only one-SM, cluster1, swizzle1, batch1 is modeled")
    require(not any(row.get(field) for field in ("qkv_peer_interleaved", "defer_v_a2a", "cyclic_peer_order",
                                               "packed_source_row", "packed_row_granularity")) and
            row.get("channel_count", 1) == 1, "Unsupported extended route semantics")
    require(row.get("layout") == SCOPE["layout"][direction] and
            row.get("raster") == ("along_m" if direction == "GEMM_A2A" else "along_n") and
            row.get("precision") == SCOPE["precision"] and row.get("launch") == "eager",
            "Unsupported layout/raster/precision/launch")
    q_width, projection = qh * dim, (qh + 2 * kvh) * dim
    require((n, k) == ((projection, hidden) if direction == "GEMM_A2A" else (hidden, q_width)) and
            q_width // world % 64 == 0, "GEMM dimensions do not match the route")
    require(max(m, n, k, sequence) <= 2**31 - 1, "Dimensions exceed the operator int32 contract")
    m_tiles, n_tiles, kp = ceil_div(m, 128), ceil_div(n, bn), ceil_div(k, 64) * 64
    tiles = m_tiles * n_tiles
    require(max(m_tiles * ceil_div(projection, 64), m_tiles * ceil_div(hidden, 128), m_tiles * world) <= 2**31 - 1 and
            all(max(2*m*width, 2*hidden*width) <= 2**63 - 1 for width in (hidden, q_width, projection)),
            "Geometry exceeds the operator flag/address capacity")
    compute = min(tiles, sm - comm)
    waves = ceil_div(tiles, compute)
    if direction == "GEMM_A2A":
        tasks, slots = ceil_div(m, 64) * (qh + 2 * kvh), 8
        # Head starts advance by 128. The flag-count pattern repeats after
        # bn/gcd(bn,128) heads, so no tensor-sized enumeration is necessary.
        heads = qh + 2 * kvh
        period = bn // math.gcd(bn, 128)
        spans = [((head * 128 + 127) // bn - head * 128 // bn + 1) for head in range(period)]
        ready = (sum(spans) * (heads // period) + sum(spans[:heads % period])) / heads
    else:
        row_bytes = 2 * k // world
        comm_rows = min(128, (48 * 1024) // row_bytes)
        require(comm_rows > 0, "OProj vector fallback is outside the calibrated copy scope")
        chunks = ceil_div(128, comm_rows)
        tasks, slots = m_tiles * world * chunks, 4
        m_window = min(m_tiles, ceil_div(sm - comm, n_tiles))
        ready = 1.0 / m_window
    payload = 2 * m * (n if direction == "GEMM_A2A" else k)
    return (
        waves * 2 * 128 * bn * kp,
        waves * (2 * kp * (128 + bn) + 2 * 128 * bn),
        ceil_div(tasks, slots * comm),
        payload * (world - 1) // world,
        comm / compute,
        ready,
    )


@dataclass
class Observation:
    row: dict
    p50_ms: float
    p95_ms: float
    run_id: str
    reference_ms: dict


@dataclass
class Dataset:
    observations: list
    contract: dict
    resource_contract: dict
    provenance: list


def read_summary(directory):
    """Fail closed on incompatible, duplicate, or unpaired summary inputs."""
    directory = Path(directory)
    json_path, csv_path = directory / "summary.json", directory / "summary.csv"
    json_bytes, csv_bytes = json_path.read_bytes(), csv_path.read_bytes()
    report = json.loads(json_bytes)
    require(report.get("schema") == "sm103_fused_verified_v1" and report.get("model_fitted") is False and
            report.get("globally_optimal") is False and report.get("diagnostic_rows") == 0,
            "Expected a verified, non-diagnostic upstream summary")
    csv_rows = {}
    reader = csv.DictReader(io.StringIO(csv_bytes.decode()))
    require(reader.fieldnames == CSV_COLUMNS, "Unexpected upstream CSV schema")
    for row in reader:
        key = (row["run_id"], int(row["candidate"]), row["component"])
        require(key not in csv_rows, "Duplicate CSV candidate/component")
        csv_rows[key] = row
    observations, contract, resources, sources = [], None, {}, []
    physical, components, component_contracts, seen_csv, run_ids = set(), {}, {}, set(), set()
    for run in report["runs"]:
        require(not run["diagnostic_only"] and run["node"] == "0a" and not run["build"]["profile"],
                "Only production node2 evidence is modeled")
        run_id, config, build = run["run_id"], run["config"], run["build"]
        require(run_id not in run_ids, "Duplicate run")
        run_ids.add(run_id)
        require(config.get("profile") == "0" and config.get("validation_self_test") == "0" and
                config.get("causal") == "1" and int(config["warmup"]) >= 10 and int(config["samples"]) >= 50,
                "Wrong diagnostics/layout/measurement config")
        base = {"source_id": hash_value(run["source_id"]),
                "binary_sha256": hash_value(build["binary_sha256"]),
                "build_inputs": hash_value(build["build_inputs"]),
                "environment_fingerprint": hash_value(run["environment_fingerprint"]),
                "input_generator": config["input_generator"], "host_launch": config["host_launch"]}
        require(base["input_generator"] in ("cpu_mt19937", "gpu_philox") and
                base["host_launch"] in ("sequential", "per_gpu_thread") and
                build["environment_fingerprint"] == base["environment_fingerprint"], "Invalid execution provenance")
        sources.append({"run_id": run_id, **base})
        for row in run["candidates"]:
            require(row["performance_accepted"] is True, "Unaccepted performance row")
            key = (run_id, row["candidate"], row["component"])
            require(key in csv_rows and key not in seen_csv, "Missing/duplicate JSON candidate/component")
            seen_csv.add(key)
            timing, tabular = row["timing"], csv_rows[key]
            for field, value in tabular.items():
                expected = ({**run, **row, **timing, "binary_sha256": build["binary_sha256"]}).get(field)
                require(expected is not None, f"Unknown/missing CSV field: {field}")
                matches = float(value) == expected if type(expected) in (float, int) else value == str(expected)
                require(matches, f"JSON/CSV disagreement: {field}")
            p50, p95 = positive(timing["p50_ms"], "p50"), positive(timing["p95_ms"], "p95")
            require(p95 >= p50 and 0 <= timing["half_drift"] <= 0.05, "Invalid timing summary")
            component = row["component"]
            require(component in ("fused", "compute_reference", "copy_reference"), "Unknown component")
            identity = (run_id, row["candidate"])
            require(component not in components.setdefault(identity, {}), "Duplicate reference component")
            components[identity][component] = p50
            require(row["collector"] == timing["collector"], "Component timing collector mismatch")
            pairing_fields = (
                "direction", "world", "global_seq", "seq_local", "hidden", "q_heads", "kv_heads", "head_dim",
                "m", "n", "k", "tile_policy", "tile_m", "tile_n", "tile_k", "comm_ctas", "cluster_ctas",
                "raster", "swizzle", "sm_count", "sm_counts", "layout", "host_launch", "collector", "precision", "launch")
            component_contracts.setdefault(identity, {})[component] = {field: row.get(field) for field in pairing_fields}
            if component != "fused":
                require(row["measurement_role"] == "calibration", "Wrong reference role")
                continue
            require(row["measurement_role"] == "production", "Wrong fused measurement role")
            physical_features(row)
            boundary = {**base, **{field: row[field] for field in ("host_launch", "collector", "precision", "launch")}}
            require(boundary["host_launch"] == config["host_launch"] and
                    row["collector"] == timing["collector"] == "per_epoch_rank_events_v3_eventsync",
                    "Wrong launch/collector boundary")
            require(contract is None or boundary == contract, "Mixed source/build/environment/measurement boundary")
            contract = boundary
            world = row["world"]
            require(len(run["devices"]) == world and
                    {(int(d["rank"]), d["runtime_cc"], int(d["sms"])) for d in run["devices"]} ==
                    {(rank, "10.3", 148) for rank in range(world)} and row["sm_counts"] == [148] * world,
                    "Unsupported device resources")
            for field in ("world", "global_seq", "seq_local", "hidden", "q_heads", "kv_heads", "head_dim"):
                require(row[field] == run["geometry"][field] == int(config[field]), "Run/candidate geometry mismatch")
            tiles = ceil_div(row["m"], 128) * ceil_div(row["n"], row["tile_n"])
            compute = min(tiles, 148 - row["comm_ctas"])
            require(row["work_tiles_derived"] == tiles and
                    row["production_compute_ctas_derived"] == [compute] * world and
                    row["production_waves_derived"] == [ceil_div(tiles, compute)] * world,
                    "Recorded compute geometry disagrees with the actual tile")
            recorded = row["production_resources"]
            require(len(recorded) == world and {int(r["rank"]) for r in recorded} == set(range(world)),
                    "Missing/duplicate resource rank")
            resource_values = set()
            for resource in recorded:
                require(int(resource["threads"]) == 256 and int(resource["dynamic_smem"]) > 0 and
                        all(int(resource[field]) == row[field] for field in ("tile_m", "tile_n", "tile_k")),
                        "Unsupported or inconsistent production resource binding")
                resource_values.add((int(resource["threads"]), int(resource["dynamic_smem"])))
            require(len(resource_values) == 1, "Heterogeneous rank resources")
            resource_key = f'{row["direction"]}/n{row["tile_n"]}'
            resource = list(resource_values.pop())
            require(resource_key not in resources or resources[resource_key] == resource,
                    "Resource binding changed across geometries")
            resources[resource_key] = resource
            require(candidate_key(row) not in physical, "Duplicate physical geometry candidate (including aliases)")
            physical.add(candidate_key(row))
            observations.append(Observation(row, p50, p95, run_id, components[identity]))
    require(observations and seen_csv == set(csv_rows) and len(seen_csv) == report["performance_rows"],
            "Empty or unpaired summary rows")
    for identity, values in components.items():
        require("fused" in values, "Reference without fused candidate")
        paired = component_contracts[identity]
        require(all(value == paired["fused"] for value in paired.values()),
                "Reference geometry/tile/measurement boundary differs from its fused candidate")
    return Dataset(observations, contract, resources,
                   [{"summary_json": str(json_path.absolute()), "summary_json_sha256": digest(json_bytes),
                     "summary_csv": str(csv_path.absolute()), "summary_csv_sha256": digest(csv_bytes), "runs": sources}])


def calibration_contract(observations):
    expected = {(direction, seq, world, *family)
                for direction in DIRECTIONS for seq in CALIBRATION_SEQUENCES for world in (4, 8)
                for family in CALIBRATION_FAMILIES}
    require(len(observations) == 504 and len({candidate_key(o.row) for o in observations}) == 504,
            "Calibration requires 504 unique fused candidates")
    require(all(set(o.reference_ms) == {"fused", "compute_reference", "copy_reference"} for o in observations),
            "Pre-registered calibration must retain all C/R diagnostic rows")
    actual, grids = set(), {}
    for observation in observations:
        row = observation.row
        actual.add((row["direction"], row["global_seq"], row["world"], row["hidden"],
                    row["q_heads"], row["kv_heads"], row["head_dim"]))
        grids.setdefault(problem_key(row), set()).add((row["tile_n"], row["comm_ctas"]))
    require(actual == expected, "Training must contain exactly the pre-registered 12 problems in both directions")
    for key, grid in grids.items():
        require(grid == {(tile, comm) for tile in TILES[key[0]] for comm in CALIBRATION_COMMS},
                "Incomplete or changed pre-registered candidate grid")


def solve(matrix, rhs):
    """Small pivoted dense solve; positive ridge makes the design full rank."""
    a = [list(row) + [value] for row, value in zip(matrix, rhs)]
    size = len(a)
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(a[row][column]))
        require(abs(a[pivot][column]) > 1e-14, "Singular ridge system")
        a[column], a[pivot] = a[pivot], a[column]
        scale = a[column][column]
        a[column] = [value / scale for value in a[column]]
        for row in range(size):
            if row != column:
                factor = a[row][column]
                a[row] = [value - factor * other for value, other in zip(a[row], a[column])]
    return [row[-1] for row in a]


def fit_ridge(observations, regularization):
    require(observations and regularization > 0 and math.isfinite(regularization), "Invalid ridge fit input")
    require(len({o.row["direction"] for o in observations}) == 1, "Fit directions separately")
    groups = {}
    for observation in observations:
        key = problem_key(observation.row)
        groups[key] = groups.get(key, 0) + 1
    weights = [1 / (len(groups) * groups[problem_key(o.row)]) for o in observations]
    features = [[math.log1p(value) for value in physical_features(o.row)] for o in observations]
    means = [sum(w * x[j] for w, x in zip(weights, features)) for j in range(6)]
    scales = [math.sqrt(sum(w * (x[j] - means[j]) ** 2 for w, x in zip(weights, features))) for j in range(6)]
    scales = [value if value > 1e-12 else 1.0 for value in scales]
    design = [[1.0] + [(x[j] - means[j]) / scales[j] for j in range(6)] for x in features]
    matrix, rhs = [[0.0] * 7 for _ in range(7)], [0.0] * 7
    for weight, x, observation in zip(weights, design, observations):
        target = math.log(positive(observation.p50_ms, "fit p50"))
        for i in range(7):
            rhs[i] += weight * x[i] * target
            for j in range(7):
                matrix[i][j] += weight * x[i] * x[j]
    for j in range(1, 7):
        matrix[j][j] += regularization
    coefficients = solve(matrix, rhs)
    require(all(math.isfinite(value) for value in coefficients), "Nonfinite model coefficients")
    return {"form": "log_ridge", "direction": observations[0].row["direction"], "regularization": regularization,
            "transform": "log1p(raw), training-fold weighted standardization",
            "target": "log(p50_ms)", "means": means, "scales": scales, "coefficients": coefficients}


def solve_nonnegative(matrix, rhs):
    """Exact small convex quadratic via support enumeration, not model search.

    Minimize b' A b - 2 r' b for b>=0. Every feasible support solution is
    checked against the full KKT conditions; at most 128 supports exist.
    Ridge supplies positive curvature even for collinear physical proxies.
    """
    size = len(rhs)
    require(0 < size <= 7 and len(matrix) == size and all(len(row) == size for row in matrix),
            "Nonnegative solver supports at most seven variables")
    require(all(math.isfinite(value) for row in matrix for value in row) and
            all(math.isfinite(value) for value in rhs), "Nonfinite quadratic system")
    tolerance = 1e-9 * max(1.0, max(abs(value) for value in rhs))
    for mask in range(1 << size):
        support = [index for index in range(size) if mask & (1 << index)]
        values = solve([[matrix[i][j] for j in support] for i in support],
                       [rhs[i] for i in support]) if support else []
        if any(value < 0 for value in values):
            continue
        coefficients = [0.0] * size
        for index, value in zip(support, values):
            coefficients[index] = value
        gradient = [sum(matrix[i][j] * coefficients[j] for j in range(size)) - rhs[i]
                    for i in range(size)]
        violation = max((abs(value) if coefficients[index] > 0 else max(0.0, -value))
                        for index, value in enumerate(gradient))
        if violation <= tolerance:
            return coefficients, violation
    raise ValueError("No KKT-valid nonnegative ridge solution")


def fit_nonnegative_ridge(observations, regularization):
    """Additive raw-proxy time model with relative squared-error weights.

    Each problem group has equal total weight. Features and target use their
    training-fold weighted RMS, with no centering. Normalizing target time
    makes the fixed ridge grid invariant to milliseconds versus microseconds.
    Objective: sum w*((Z*b-t)/t)^2 + lambda*sum(b[1:]^2), all b>=0.
    The intercept is nonnegative but unpenalized. No C/R lower bound exists.
    """
    require(observations and regularization > 0 and math.isfinite(regularization), "Invalid ridge fit input")
    require(len({o.row["direction"] for o in observations}) == 1, "Fit directions separately")
    groups = {}
    for observation in observations:
        key = problem_key(observation.row)
        groups[key] = groups.get(key, 0) + 1
    weights = [1 / (len(groups) * groups[problem_key(o.row)]) for o in observations]
    features = [physical_features(o.row) for o in observations]
    targets = [positive(o.p50_ms, "fit p50") for o in observations]
    scales = [math.sqrt(sum(weight * feature[j] ** 2 for weight, feature in zip(weights, features)))
              for j in range(6)]
    scales = [value if value > 0 else 1.0 for value in scales]
    target_scale = math.sqrt(sum(weight * target ** 2 for weight, target in zip(weights, targets)))
    matrix, rhs = [[0.0] * 7 for _ in range(7)], [0.0] * 7
    for weight, features_, target in zip(weights, features, targets):
        x = [1.0] + [value / scale for value, scale in zip(features_, scales)]
        t = target / target_scale
        relative_weight = weight / (t*t)
        for i in range(7):
            rhs[i] += relative_weight * x[i] * t
            for j in range(7):
                matrix[i][j] += relative_weight * x[i] * x[j]
    for j in range(1, 7):
        matrix[j][j] += regularization
    coefficients, violation = solve_nonnegative(matrix, rhs)
    return {"form": "nonnegative_time_ridge", "direction": observations[0].row["direction"],
            "regularization": regularization, "transform": "raw / training-fold weighted RMS; no centering",
            "target": "p50_ms", "target_scale": target_scale, "scales": scales,
            "coefficients": coefficients, "kkt_max_violation": violation}


def fit_model(observations, regularization, form):
    require(form in MODEL_FORMS, "Unknown pre-defined model form")
    fit = fit_ridge if form == "log_ridge" else fit_nonnegative_ridge
    return fit(observations, regularization)


def predict(model, row):
    require(row["direction"] == model["direction"], "Wrong model direction")
    form = model.get("form", "log_ridge")  # The frozen v1 report predates this explicit field.
    require(form in MODEL_FORMS, "Unknown pre-defined model form")
    if form == "nonnegative_time_ridge":
        features = [1.0] + [value / scale for value, scale in zip(physical_features(row), model["scales"])]
        value = model["target_scale"] * sum(weight * feature for weight, feature in zip(model["coefficients"], features))
        return positive(value, "prediction")
    x = [1.0] + [(math.log1p(value) - mean) / scale for value, mean, scale in
                 zip(physical_features(row), model["means"], model["scales"])]
    value = math.exp(sum(weight * feature for weight, feature in zip(model["coefficients"], x)))
    return positive(value, "prediction")


def percentile(values, fraction):
    values = sorted(values)
    require(values, "Empty metric")
    index = (len(values) - 1) * fraction
    left = int(index)
    right = min(left + 1, len(values) - 1)
    return values[left] + (values[right] - values[left]) * (index - left)


def stats(values):
    return {"median": percentile(values, 0.5), "p95": percentile(values, 0.95), "max": max(values)}


def evaluate(observations, predictions):
    """Preserve every true candidate; C/R are audit-only, including F<C."""
    require(len(observations) == len(predictions), "Prediction count mismatch")
    groups, errors, log_errors = {}, [], []
    for observation, estimate in zip(observations, predictions):
        positive(estimate, "evaluation prediction")
        errors.append(estimate / observation.p50_ms - 1)
        log_errors.append(math.log(estimate / observation.p50_ms))
        row = observation.row
        refs = observation.reference_ms
        references = {key: refs.get(key) for key in ("compute_reference", "copy_reference")}
        reference_max = max(references.values()) if all(v is not None for v in references.values()) else None
        groups.setdefault(problem_key(row), []).append({
            "run_id": observation.run_id, "candidate": row["candidate"], "tile_n": row["tile_n"],
            "comm_ctas": row["comm_ctas"], "actual_p50_ms": observation.p50_ms,
            "actual_p95_ms": observation.p95_ms, "predicted_p50_ms": estimate,
            **references, "f_minus_max_cr_ms": None if reference_max is None else observation.p50_ms - reference_max})
    results, regrets = [], []
    for key, candidates in sorted(groups.items()):
        selected = min(candidates, key=lambda c: (c["predicted_p50_ms"], c["comm_ctas"], c["tile_n"]))
        best = min(candidates, key=lambda c: (c["actual_p50_ms"], c["comm_ctas"], c["tile_n"]))
        regret = selected["actual_p50_ms"] / best["actual_p50_ms"] - 1
        regrets.append(regret)
        results.append({"problem": list(key), "selected_candidate": selected["candidate"],
                        "selected_tile_n": selected["tile_n"], "selected_comm_ctas": selected["comm_ctas"],
                        "grid_best_p50_ms": best["actual_p50_ms"], "regret_fraction": regret,
                        "candidates": candidates})
    return {"relative_prediction_error_fraction": {"absolute": stats([abs(x) for x in errors]),
                                                    "signed_median": percentile(errors, 0.5)},
            "log_rmse": math.sqrt(sum(x*x for x in log_errors) / len(log_errors)),
            "regret_fraction": stats(regrets), "groups": results}


def choose_regularization(observations, form="log_ridge"):
    families = sorted({family_key(o.row) for o in observations})
    require(len(families) >= 2, "Regularization selection needs at least two geometry families")
    scores = []
    for value in REGULARIZATIONS:
        group_losses = []
        for held in families:
            train = [o for o in observations if family_key(o.row) != held]
            model = fit_model(train, value, form)
            losses = {}
            for observation in observations:
                if family_key(observation.row) == held:
                    ratio = predict(model, observation.row) / observation.p50_ms
                    loss = (math.log(ratio) if form == "log_ridge" else ratio - 1) ** 2
                    losses.setdefault(problem_key(observation.row), []).append(loss)
            group_losses.extend(sum(values) / len(values) for values in losses.values())
        metric = "mean_group_log_squared_error" if form == "log_ridge" else "mean_group_relative_squared_error"
        scores.append({"regularization": value, metric: sum(group_losses) / len(group_losses)})
    selected = min(scores, key=lambda s: (s[metric], -s["regularization"]))
    return selected["regularization"], scores


def nested_lofo(observations, form="log_ridge"):
    families = sorted({family_key(o.row) for o in observations})
    require(len(families) >= 3, "Nested family CV needs at least three actual geometry families")
    predictions, ordered, folds = [], [], []
    for held in families:
        train = [o for o in observations if family_key(o.row) != held]
        test = [o for o in observations if family_key(o.row) == held]
        regularization, scores = choose_regularization(train, form)
        model = fit_model(train, regularization, form)
        estimates = [predict(model, o.row) for o in test]
        ordered.extend(test)
        predictions.extend(estimates)
        folds.append({"held_family": list(held), "training_families": [list(f) for f in families if f != held],
                      "regularization": regularization, "inner_selection": scores})
    return {"method": "nested_leave_actual_geometry_family_out", "folds": folds, **evaluate(ordered, predictions)}


def experiment(training, heldout=(), form="nonnegative_time_ridge"):
    require(form in MODEL_FORMS, "Unknown pre-defined model form")
    calibration_contract(training.observations)
    train_keys = {problem_key(o.row) for o in training.observations}
    external_keys = set()
    for dataset in heldout:
        require(dataset.contract == training.contract and dataset.resource_contract == training.resource_contract,
                "External evidence uses a different execution/resource contract")
        keys = {problem_key(o.row) for o in dataset.observations}
        require(not (keys & train_keys) and not (keys & external_keys), "Held-out geometry overlaps calibration/another held-out input")
        external_keys.update(keys)
        grids = {}
        for observation in dataset.observations:
            row = observation.row
            grids.setdefault(problem_key(row), set()).add((row["tile_n"], row["comm_ctas"]))
        for key, grid in grids.items():
            require(grid == {(tile, comm) for tile in TILES[key[0]] for comm in CALIBRATION_COMMS},
                    "Held-out candidate grid differs from the pre-registered comparison")
    report = {"schema": "sm103_fused_model_experiment_v1", "runtime_selector_installed": False,
              "globally_optimal": False, "scope": SCOPE, "features": FEATURES,
              "feature_revision": "physical_six_v1", "regularizations": REGULARIZATIONS,
              "model_form": form,
              "objective": ("equal problem-group weighted squared log p50 error; intercept unpenalized"
                            if form == "log_ridge" else
                            "equal problem-group weighted squared relative p50 error; nonnegative coefficients/intercept; intercept unpenalized"),
              "development_note": ("v1 pre-defined calibration experiment" if form == "log_ridge" else
                                   "v2 development iteration motivated by v1 training-family CV failures; CV is not untouched external validation"),
              "selection_tie_break": "lower comm, then lower tile_n; no measured winner in prediction",
              "execution_contract": training.contract, "resource_contract": training.resource_contract,
              "training_provenance": training.provenance,
              "heldout_provenance": [p for d in heldout for p in d.provenance], "directions": {}}
    for direction in DIRECTIONS:
        observations = [o for o in training.observations if o.row["direction"] == direction]
        cv = nested_lofo(observations, form)
        regularization, scores = choose_regularization(observations, form)
        model = fit_model(observations, regularization, form)
        external = [o for d in heldout for o in d.observations if o.row["direction"] == direction]
        external_report = None
        if external:
            estimates = [predict(model, o.row) for o in external]
            external_report = evaluate(external, estimates)
            train_families = {family_key(o.row) for o in observations}
            train_sequences = {o.row["global_seq"] for o in observations}
            strata = {}
            for observation, estimate in zip(external, estimates):
                family = "seen_family" if family_key(observation.row) in train_families else "new_family"
                sequence = "seen_sequence" if observation.row["global_seq"] in train_sequences else "new_sequence"
                selected = strata.setdefault(f"{family}/{sequence}", ([], []))
                selected[0].append(observation)
                selected[1].append(estimate)
            external_report["generalization_strata"] = {
                name: {key: value for key, value in evaluate(rows, predictions).items() if key != "groups"}
                for name, (rows, predictions) in strata.items()}
        report["directions"][direction] = {
            "training_family_cv": cv, "final_regularization_selection": scores, "model": model,
            "external_heldout": external_report}
    report["experiment_code_sha256"] = digest(Path(__file__).read_bytes())
    return report


# Primitive estimates are deliberately separate from the frozen v1/v2 fits.
# No B300 timings are supplied here: callers must provide measured calibration.
PRIMITIVE_SCHEDULE_FIELDS = (
    "schedule_schema", "raster_requested", "raster", "max_swizzle_size",
    "effective_swizzle_size", "swizzle", "padded_m_tiles", "padded_n_tiles",
    "has_padding", "scheduled_work_tiles_derived",
)


@dataclass
class PrimitiveDataset:
    """Audited observations partitioned by execution contract, not fitted anchors."""
    records: list
    execution_contracts: dict
    provenance: dict


def primitive_schedule(row):
    """Check the current cluster-1 static scheduler; predict no reuse benefit."""
    m_tiles = ceil_div(integer(row, "m"), 128)
    n_tiles = ceil_div(integer(row, "n"), integer(row, "tile_n"))
    maximum = integer(row, "max_swizzle_size")
    require(maximum in (1, 2, 4, 8), "Unsupported requested swizzle")
    minimum = min(m_tiles, n_tiles)
    effective = (8 if maximum >= 8 and minimum >= 6 else
                 4 if maximum >= 4 and minimum >= 3 else
                 2 if maximum >= 2 and minimum >= 2 else 1)
    requested = row.get("raster_requested")
    require(requested in ("heuristic", "along_m", "along_n"), "Unsupported requested raster")
    raster = ("along_m" if row["direction"] == "GEMM_A2A" else "along_n") if requested == "heuristic" else requested
    padded_m, padded_n = ceil_div(m_tiles, effective) * effective, ceil_div(n_tiles, effective) * effective
    require(padded_m * padded_n <= 2**31 - 1, "Scheduled work exceeds int32")
    schema = row.get("schedule_schema")
    require(schema in ("legacy_fixed_v1", "explicit_v1") and
            (schema != "legacy_fixed_v1" or (maximum == 1 and requested == "heuristic")),
            "Invalid scheduler provenance")
    schedule = dict(schedule_schema=schema, raster_requested=requested, raster=raster,
                    max_swizzle_size=maximum, effective_swizzle_size=effective, swizzle=effective,
                    padded_m_tiles=padded_m, padded_n_tiles=padded_n,
                    has_padding=(padded_m != m_tiles or padded_n != n_tiles),
                    scheduled_work_tiles_derived=padded_m * padded_n)
    require(all(type(row.get(key)) is type(value) and row[key] == value for key, value in schedule.items()),
            "Actual schedule differs from requested/lowered geometry")
    return schedule


def primitive_candidate_key(row):
    """No model labels; preserve distinct tile, schedule and communication choices."""
    return comparison_problem_key(row) + (
        row["tile_policy"], row["tile_m"], row["tile_n"], row["tile_k"], row["comm_ctas"],
        tuple(row[key] for key in PRIMITIVE_SCHEDULE_FIELDS),
        tuple(row["production_compute_ctas_derived"]),
    )


def primitive_metadata_int(row, field, minimum=0):
    value = row.get(field)
    require(type(value) is int or (isinstance(value, str) and value.isdecimal()),
            f"Missing/integer metadata: {field}")
    value = int(value)
    require(value >= minimum, f"Out-of-range metadata: {field}")
    return value


def primitive_row_resources(row):
    """Only production resources are known; reference/collective details stay unknown."""
    world = integer(row, "world")
    resources = row.get("production_resources")
    require(isinstance(resources, list) and len(resources) == world and
            {primitive_metadata_int(r, "rank") for r in resources} == set(range(world)),
            "Missing/duplicate production resource rank")
    bindings = set()
    for resource in resources:
        fields = ("tile_m", "tile_n", "tile_k", "threads", "dynamic_smem")
        binding = tuple(primitive_metadata_int(resource, key, 1) for key in fields)
        require(binding[:3] == (128, row["tile_n"], 64) and binding[3] == 256 and binding[4] % 128 == 0,
                "Production resource binding differs from candidate")
        bindings.add(binding)
    require(len(bindings) == 1, "Heterogeneous production resource binding")
    production = dict(zip(fields, bindings.pop()))
    return {"production": production,
            "actual_component": production if row["component"] == "fused" else None,
            "requested_epilogue_n": 32 if row["tile_policy"] == "m128n256k64e32" else None,
            "actual_epilogue_tile": None, "element_c": None, "ab_stages": None,
            "accumulator_stages": None}


def read_primitive_summary(directory):
    """Consume current verified F/C/R summaries, without fitting or re-auditing GPU logs.

    MPI Eager only; N128 auto and explicit N256/K64/E32 are recognized policy
    requests. Actual epilogue/AB/TMEM metadata is absent from this summary and
    must not be inferred from policy names or borrowed from native diagnostics.
    Different source/node/build contracts remain separate partitions. A C/R
    total is an independent reference observation, never a TC service anchor.
    """
    directory = Path(directory)
    paths = [directory / name for name in ("summary.json", "summary.csv")]
    json_bytes, csv_bytes = (path.read_bytes() for path in paths)
    report = json.loads(json_bytes)
    require(report.get("schema") == "sm103_fused_verified_v1" and
            report.get("model_fitted") is False and report.get("globally_optimal") is False and
            report.get("diagnostic_rows") == 0, "Expected verified non-diagnostic fused summary")
    reader = csv.DictReader(io.StringIO(csv_bytes.decode()))
    required = set(CSV_COLUMNS) | set(PRIMITIVE_SCHEDULE_FIELDS) | {"graph_epoch_mode"}
    require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames) and
            required <= set(reader.fieldnames), "Missing/duplicate primitive CSV columns")
    tabular = {}
    for value in reader:
        require(None not in value and all(item is not None for item in value.values()), "Malformed primitive CSV row")
        key = value["run_id"], int(value["candidate"]), value["component"]
        require(key not in tabular, "Duplicate primitive CSV candidate/component")
        tabular[key] = value
    records, contracts, runs, seen, physical, paired = [], {}, [], set(), set(), {}
    for run in report["runs"]:
        run_id, config, build = run["run_id"], run["config"], run["build"]
        require(run_id not in {r["run_id"] for r in runs}, "Duplicate primitive run")
        require(run.get("diagnostic_only") is False and build.get("profile") is False and build.get("mpi") is True and
                config.get("process_layout") == "mpi_one_process_per_gpu" and
                config.get("host_launch") == "mpi_process" and config.get("launch") == "eager" and
                config.get("profile") == config.get("validation_self_test") == "0" and
                primitive_metadata_int(config, "warmup", 10) == 10 and
                primitive_metadata_int(config, "samples", 50) == 50,
                "Primitive input requires production MPI Eager 10+50")
        require(config.get("calibrate") in ("0", "1") and config.get("causal") in ("0", "1"),
                "Missing calibration/route request")
        contract = primitive_execution_contract({
            **{key: run[key] for key in ("source_id", "environment_fingerprint", "node")},
            **{key: build[key] for key in ("binary_sha256", "build_inputs")},
            "runtime_cc": "10.3", "sm_count": 148, "precision": "bf16_accfp32_bf16",
            "launch": "eager", "host_launch": "mpi_process", "collector": "mpi_rank_events_v1",
            "input_generator": config["input_generator"]})
        require(build["environment_fingerprint"] == contract["environment_fingerprint"], "Build environment mismatch")
        contract_id = digest(json.dumps(contract, sort_keys=True).encode())
        contracts[contract_id] = contract
        runs.append({"run_id": run_id, "contract_id": contract_id})
        world = integer(run["geometry"], "world")
        devices = run["devices"]
        require(len(devices) == world and
                {(primitive_metadata_int(d, "rank"), d["runtime_cc"], primitive_metadata_int(d, "sms", 1))
                 for d in devices} == {(rank, "10.3", 148) for rank in range(world)}, "Unsupported rank resources")
        for row in run["candidates"]:
            component = row.get("component")
            require(component in ("fused", "compute_reference", "copy_reference") and
                    row.get("performance_accepted") is True and
                    row.get("measurement_role") == ("production" if component == "fused" else "calibration"),
                    "Unaccepted or incorrectly classified primitive observation")
            key = run_id, integer(row, "candidate"), component
            require(key in tabular and key not in seen, "Missing/duplicate primitive JSON component")
            seen.add(key)
            flat = run | row | row["timing"] | {"binary_sha256": build["binary_sha256"]}
            for field, value in tabular[key].items():
                require(field in flat, f"Unknown primitive CSV field: {field}")
                expected = flat[field]
                matches = (value == "" if expected is None else
                           float(value) == expected if type(expected) in (int, float) else value == str(expected))
                require(matches, f"Primitive JSON/CSV disagreement: {field}")
            require(all(row.get(k) == contract[k] for k in ("host_launch", "collector", "precision", "launch")) and
                    row.get("graph_epoch_mode") is None and row["timing"]["collector"] == contract["collector"],
                    "Primitive observation execution mismatch")
            for field in ("world", "global_seq", "seq_local", "hidden", "q_heads", "kv_heads", "head_dim"):
                require(integer(row, field) == run["geometry"][field] == primitive_metadata_int(config, field, 1),
                        "Primitive candidate/run geometry mismatch")
            comparison_problem_key(row)
            policy_tiles = {"m128n128": 128, "m128n256k64e32": 256}
            require(row.get("tile_policy") in policy_tiles and row.get("tile_n") == policy_tiles[row["tile_policy"]] and
                    row.get("tile_m") == 128 and row.get("tile_k") == 64 and row.get("cluster_ctas") == 1 and
                    row.get("sm_counts") == [148] * world and integer(row, "sm_count") == 148,
                    "Unsupported primitive tile/resource scope")
            require(not any(row.get(field) for field in ("cyclic_peer_order", "packed_row_granularity")) and
                    row.get("channel_count", 1) == 1, "Unsupported extended route")
            expected_layout = ("qkv_source_rank_major_v1" if row["direction"] == "GEMM_A2A" else
                               "causal_dual_chunk_v1" if config["causal"] == "1" else "sequence_rank_major_v1")
            require(row["layout"] == expected_layout, "Route layout differs from request")
            schedule = primitive_schedule(row)
            requested = run["scheduling"]
            require(schedule["schedule_schema"] == requested["schema"] and
                    schedule["max_swizzle_size"] == requested["max_swizzle_size"] and
                    schedule["raster_requested"] == requested["requested_rasters"][row["direction"]] and
                    schedule["raster"] == requested["effective_rasters"][row["direction"]],
                    "Candidate/run scheduling mismatch")
            comm = integer(row, "comm_ctas")
            require(comm < 148, "Invalid communication budget")
            compute = min(schedule["scheduled_work_tiles_derived"], 148 - comm)
            require(row["production_compute_ctas_derived"] == [compute] * world and
                    row["compute_ctas_derived"] == [0 if component == "copy_reference" else compute] * world and
                    row["production_waves_derived"] == [ceil_div(schedule["scheduled_work_tiles_derived"], compute)] * world and
                    row["work_tiles_derived"] == ceil_div(row["m"], 128) * ceil_div(row["n"], row["tile_n"]),
                    "Actual compute budget/work geometry mismatch")
            resources = primitive_row_resources(row)
            flops = 2 * row["m"] * row["n"] * row["k"]
            payload = 2 * row["m"] * (row["n"] if row["direction"] == "GEMM_A2A" else row["k"])
            require(row["problem_gemm_flops"] == flops and row["problem_route_payload_bytes"] == payload and
                    row["problem_remote_payload_bytes"] == payload * (world - 1) // world and
                    row["executed_gemm_flops"] == (0 if component == "copy_reference" else flops) and
                    row["executed_route_payload_bytes"] == (0 if component == "compute_reference" else payload),
                    "Component executed-work scope mismatch")
            timing = row["timing"]
            p50, p95 = positive(timing["p50_ms"], "primitive p50"), positive(timing["p95_ms"], "primitive p95")
            require(p95 >= p50 and 0 <= nonnegative(timing["half_drift"], "primitive drift") <= .05,
                    "Invalid primitive timing")
            candidate_key = primitive_candidate_key(row)
            physical_key = contract_id, candidate_key, component
            require(physical_key not in physical, "Duplicate physical primitive candidate/component")
            physical.add(physical_key)
            pairing = candidate_key, resources["production"]
            bucket = paired.setdefault((run_id, row["candidate"]), {"expected": {"fused"} if config["calibrate"] == "0"
                                                                   else {"fused", "compute_reference", "copy_reference"}})
            bucket[component] = pairing
            records.append({"run_id": run_id, "contract_id": contract_id, "candidate_key": candidate_key,
                "row": row, "schedule": schedule, "resources": resources,
                "p50_ms": p50, "p95_ms": p95, "component": component,
                "measurement_scope": "fused_boundary" if component == "fused" else "independent_reference",
                "long_sequence": row["global_seq"] >= 65536,
                "tc_service_us": None, "calibration_eligible": False,
                "calibration_limitation": "Totals are not steady TC service; actual collective metadata is incomplete"})
    require(records and seen == set(tabular) and len(records) == report["performance_rows"],
            "Empty/unpaired primitive summary")
    for values in paired.values():
        require(set(values) - {"expected"} == values["expected"] and
                all(values[key] == values["fused"] for key in values["expected"]),
                "Missing or mismatched F/C/R physical candidate")
    return PrimitiveDataset(records, contracts, {
        "summary_json": str(paths[0].absolute()), "summary_json_sha256": digest(json_bytes),
        "summary_csv": str(paths[1].absolute()), "summary_csv_sha256": digest(csv_bytes),
        "runs": runs, "raw_artifact_audit": "owned_by_summarize_sm103_fused; not repeated here",
        "model_fitted": False})


@dataclass(frozen=True)
class CollectiveSignature:
    tile_m: int
    tile_n: int
    tile_k: int
    epilogue_m: int
    epilogue_n: int
    ab_stages: int
    dynamic_smem_bytes: int
    threads: int = 256
    cluster_ctas: int = 1
    precision: str = "bf16_accfp32_bf16"
    element_c: str = "void"

    def __post_init__(self):
        for field in ("tile_m", "tile_n", "tile_k", "epilogue_m", "epilogue_n",
                      "ab_stages", "dynamic_smem_bytes", "threads", "cluster_ctas"):
            integer(vars(self), field)
        require((self.tile_m, self.tile_k, self.epilogue_m, self.epilogue_n) == (128, 64, 128, 32) and
                self.tile_n in (128, 256) and self.threads == 256 and self.cluster_ctas == 1 and
                self.precision == "bf16_accfp32_bf16" and self.element_c == "void",
                "Primitive scope is the current one-SM N128/N256 K64 no-C BF16 collective")
        require(self.ab_stages >= 2 and (self.tile_n != 256 or self.ab_stages == 4) and
                self.dynamic_smem_bytes % 128 == 0, "Invalid actual collective resources")


def primitive_problem_key(row):
    """Physical workload, independent of labels, policy spelling and SM split."""
    direction = row.get("direction")
    require(direction in DIRECTIONS, "Unsupported direction")
    m, n, k, world = (integer(row, key) for key in ("m", "n", "k", "world"))
    require(world in (4, 8) and max(m, n, k) <= 2**31 - 1, "Unsupported CP/dimension range")
    allowed = (("qkv_source_rank_major_v1",) if direction == "GEMM_A2A" else
               ("causal_dual_chunk_v1", "sequence_rank_major_v1"))
    require(row.get("layout") in allowed, "Unsupported physical route layout")
    return direction, m, n, k, world, row["layout"]


def primitive_execution_contract(contract):
    """Keep hardware, kernel provenance and measurement cadence inseparable."""
    fields = ("source_id", "binary_sha256", "build_inputs", "environment_fingerprint",
              "node", "runtime_cc", "sm_count", "precision", "launch", "host_launch",
              "collector", "input_generator")
    require(set(contract) == set(fields), "Incomplete/unknown primitive execution contract")
    for field in fields[:4]:
        hash_value(contract[field])
    require(contract["node"] in ("09", "0a") and contract["runtime_cc"] == "10.3" and
            type(contract["sm_count"]) is int and contract["sm_count"] == 148 and
            contract["precision"] == "bf16_accfp32_bf16" and contract["launch"] == "eager" and
            contract["input_generator"] in ("cpu_mt19937", "gpu_philox"), "Unsupported primitive execution scope")
    host, collector = contract["host_launch"], contract["collector"]
    require((host in ("sequential", "per_gpu_thread") and collector == "per_epoch_rank_events_v3_eventsync") or
            (host == "mpi_process" and collector == "mpi_rank_events_v1"), "Unknown launch/collector pairing")
    return dict(contract)


def nonnegative(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            f"Invalid nonnegative finite {name}")
    return float(value)


@dataclass(frozen=True)
class PrimitiveCalibration:
    """Explicit steady tile service in us; not fitted end-to-end F winners.

    Each (K, tile_us) anchor excludes the once-per-launch overhead. The caller
    supplies actual collective resources and the measured validity domain.
    No K, worker-budget or matrix-size extrapolation is performed. Anchor
    extraction and its raw-data audit belong to the calibration producer.
    """
    signature: CollectiveSignature
    k_anchors: tuple
    launch_us: float | None
    compute_ctas_range: tuple
    m_range: tuple
    n_range: tuple
    execution_contract: dict
    calibration_problems: tuple
    evidence_sha256: str

    def __post_init__(self):
        require(isinstance(self.signature, CollectiveSignature), "Missing actual collective signature")
        object.__setattr__(self, "execution_contract", MappingProxyType(
            primitive_execution_contract(self.execution_contract)))
        object.__setattr__(self, "k_anchors", tuple(tuple(item) for item in self.k_anchors))
        object.__setattr__(self, "calibration_problems", tuple(tuple(item) for item in self.calibration_problems))
        for field in ("compute_ctas_range", "m_range", "n_range"):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        hash_value(self.evidence_sha256)
        require(len(self.k_anchors) >= 2, "At least two explicit K anchors are required")
        previous = 0
        for k, service in self.k_anchors:
            require(type(k) is int and previous < k <= 2**31 - 1 and k % self.signature.tile_k == 0,
                    "K anchors must be ordered, unique and tile-aligned")
            positive(service, "tile service us")
            previous = k
        if self.launch_us is not None:
            nonnegative(self.launch_us, "launch us")
        for bounds in (self.compute_ctas_range, self.m_range, self.n_range):
            require(len(bounds) == 2 and all(type(x) is int and x > 0 for x in bounds) and
                    bounds[0] <= bounds[1], "Invalid measured primitive domain")
        require(self.compute_ctas_range[1] <= self.execution_contract["sm_count"], "Worker domain exceeds device")
        require(self.calibration_problems and
                len(set(self.calibration_problems)) == len(self.calibration_problems), "Missing/duplicate calibration problems")
        for key in self.calibration_problems:
            require(len(key) == 6 and primitive_problem_key(dict(zip(
                ("direction", "m", "n", "k", "world", "layout"), key))) == key, "Invalid calibration problem key")
            require(key[0] == self.calibration_problems[0][0] and
                    self.m_range[0] <= key[1] <= self.m_range[1] and
                    self.n_range[0] <= key[2] <= self.n_range[1] and
                    self.k_anchors[0][0] <= key[3] <= self.k_anchors[-1][0],
                    "Calibration workload outside its declared direction/domain")

    @classmethod
    def from_dict(cls, value):
        """Consume an explicit JSON-compatible calibration, without fitting."""
        value = dict(value)
        require(value.pop("schema", None) == "sm103_primitive_calibration_v1", "Unknown primitive calibration schema")
        value["signature"] = CollectiveSignature(**value["signature"])
        for field in ("k_anchors", "calibration_problems"):
            value[field] = tuple(tuple(item) for item in value[field])
        for field in ("compute_ctas_range", "m_range", "n_range"):
            value[field] = tuple(value[field])
        return cls(**value)


def estimate_compute_roof(row, calibration, *, signature, execution_contract, heldout=False):
    """Continuous Q/g estimate; an unmeasured launch term stays unknown.

    A held-out flag checks physical workload separation, not proof that the
    prediction is accurate. The kernel signature is actual metadata supplied
    by the caller; tile_policy strings and measured C/R/F never enter arithmetic.
    """
    require(type(heldout) is bool, "heldout must be a bool")
    require(signature == calibration.signature, "Actual collective differs from calibration")
    require(primitive_execution_contract(execution_contract) == calibration.execution_contract,
            "Primitive execution contract mismatch")
    key = primitive_problem_key(row)
    require(key[0] == calibration.calibration_problems[0][0] and
            (key[4], key[5]) in {(p[4], p[5]) for p in calibration.calibration_problems},
            "Uncalibrated direction/CP/layout")
    for field in ("source_id", "node", "precision", "launch", "host_launch", "collector"):
        require(field not in row or row[field] == execution_contract[field], "Row execution contract mismatch")
    require(not heldout or key not in calibration.calibration_problems, "Held-out physical workload overlaps calibration")
    require(all(integer(row, field) == getattr(signature, field) for field in ("tile_m", "tile_n", "tile_k")) and
            row.get("precision") == signature.precision and row.get("cluster_ctas") == 1 and
            row.get("swizzle") == 1 and row.get("batch", 1) == 1 and
            row.get("raster") == ("along_m" if key[0] == "GEMM_A2A" else "along_n"),
            "Problem/collective geometry mismatch")
    require(not any(row.get(field) for field in ("qkv_peer_interleaved", "defer_v_a2a", "cyclic_peer_order",
                                               "packed_source_row", "packed_row_granularity")) and
            row.get("channel_count", 1) == 1, "Uncalibrated extended route semantics")
    _, m, n, k, world, _ = key
    require(k % signature.tile_k == 0 and
            (key[0] != "A2A_GEMM" or k % (world * signature.tile_k) == 0), "Unsupported K/peer boundary")
    sm, comm = integer(row, "sm_count"), integer(row, "comm_ctas")
    require(sm == execution_contract["sm_count"] and comm < sm, "Invalid physical SM split")
    if "production_resources" in row:
        resources = row["production_resources"]
        require(len(resources) == world and {int(r["rank"]) for r in resources} == set(range(world)) and
                all(int(r["threads"]) == signature.threads and
                    int(r["dynamic_smem"]) == signature.dynamic_smem_bytes and
                    all(int(r[field]) == getattr(signature, field) for field in ("tile_m", "tile_n", "tile_k"))
                    for r in resources), "Recorded resources differ from actual collective")
    tiles = ceil_div(m, signature.tile_m) * ceil_div(n, signature.tile_n)
    workers = min(tiles, sm - comm)
    for actual, domain in ((workers, calibration.compute_ctas_range), (m, calibration.m_range), (n, calibration.n_range)):
        require(domain[0] <= actual <= domain[1], "Outside measured primitive domain")
    anchors = calibration.k_anchors
    require(anchors[0][0] <= k <= anchors[-1][0], "K extrapolation is not calibrated")
    for (low_k, low_us), (high_k, high_us) in zip(anchors, anchors[1:]):
        if k <= high_k:
            tile_us = low_us + (high_us - low_us) * (k - low_k) / (high_k - low_k)
            break
    waves = tiles / workers
    steady_us = waves * tile_us
    return {"model": "continuous_tile_service_v1", "tile_service_us": tile_us,
            "work_tiles": tiles, "compute_ctas": workers, "wave_equivalents": waves,
            "integer_wave_correction_applied": False, "steady_compute_us": steady_us,
            "launch_us": calibration.launch_us,
            "compute_us": None if calibration.launch_us is None else steady_us + calibration.launch_us,
            "tile_flops": 2 * signature.tile_m * signature.tile_n * k,
            "tile_bytes_proxy": 2 * k * (signature.tile_m + signature.tile_n) + 2 * signature.tile_m * signature.tile_n,
            "heldout_workload": heldout, "externally_validated": False,
            "in_calibration_workloads": key in calibration.calibration_problems,
            "compute_roof_is_lower_bound": False,
            "signature": dict(vars(signature)), "execution_contract": dict(calibration.execution_contract),
            "interpolation_k": [low_k, high_k],
            "calibration_evidence_sha256": calibration.evidence_sha256}


def score_pipeline(direction, *, compute_service_us=None, route_service_us=None,
                   first_compute_us=None, first_route_us=None, tail_us=None,
                   readiness="unknown", producer_wait_us=None):
    """Explicit ideal-overlap scenario, never an inferred TC-stall duration.

    Services exclude duplicate launch/final-ack costs; raw standalone C/R
    totals are not automatically such services. `sufficient` must come from
    feed-frontier evidence (OProj) or drain evidence (QKV), not R<C. The tail
    is only work outside the relaxed overlap expression. None means unmeasured.
    """
    require(direction in DIRECTIONS and readiness in ("unknown", "sufficient", "insufficient"),
            "Invalid pipeline direction/readiness")
    values = dict(compute_service_us=compute_service_us, route_service_us=route_service_us,
                  first_compute_us=first_compute_us, first_route_us=first_route_us,
                  tail_us=tail_us, producer_wait_us=producer_wait_us)
    for name, value in values.items():
        if value is not None:
            nonnegative(value, name)
    c, r = compute_service_us, route_service_us
    first = first_compute_us if direction == "GEMM_A2A" else first_route_us
    if c is not None and first_compute_us is not None:
        require(first_compute_us <= c, "First compute is already part of total compute")
    if r is not None and first_route_us is not None:
        require(first_route_us <= r, "First route is already part of total route")
    known = c is not None and r is not None and first is not None
    relaxed = (max(c, first + r) if direction == "GEMM_A2A" else max(first + c, r)) if known else None
    complete = known and tail_us is not None and readiness == "sufficient"
    return {"direction": direction, "constraint": "drain" if direction == "GEMM_A2A" else "feed",
            **values, "readiness": readiness, "relaxed_overlap_us": relaxed,
            "score_us": relaxed + tail_us if complete else None,
            "status": "ideal_scenario" if complete else "insufficient" if readiness == "insufficient" else "unknown",
            "full_overlap_established": False, "theoretical_upper_speedup": None,
            "producer_wait_is_tensor_core_stall": False}


def _oproj_delivery_plan(m_tiles, n_tiles, swizzle, compute_ctas, raster,
                         world, chunks, cohort_m_tiles):
    """Enumerate the current cluster-1 scheduler and communication priorities.

    Invalid padded GEMM coordinates retain their logical index but do no MMA.
    Unlike the device decoder this explicit oracle needs no prefix inversion.
    """
    pm, pn = ceil_div(m_tiles, swizzle) * swizzle, ceil_div(n_tiles, swizzle) * swizzle
    along_n = raster == "along_n"
    tiles, windows, seen = [], {}, set()
    for group in range(0, pm if along_n else pn, swizzle):
        for major in range(pn if along_n else pm):
            for offset in range(swizzle):
                m, n = (group + offset, major) if along_n else (major, group + offset)
                logical = len(tiles)
                valid = m < m_tiles and n < n_tiles
                tiles.append((m, n) if valid else None)
                if valid and m not in seen:
                    windows.setdefault(logical // compute_ctas, []).append(m)
                    seen.add(m)
    use_cohorts = not along_n and pn > swizzle
    copies = []
    for ms in windows.values():
        width = cohort_m_tiles if use_cohorts else max(swizzle, ceil_div(len(ms), world))
        groups = [ms[start:start + width] for start in range(0, len(ms), width)]
        if use_cohorts:
            cells = ((group, peer) for group in range(len(groups)) for peer in range(world))
        else:
            cells = ((diagonal - peer, peer)
                     for diagonal in range(len(groups) + world - 1)
                     for peer in range(world) if 0 <= diagonal - peer < len(groups))
        for group, peer in cells:
            copies.extend((m, peer, chunk) for m in groups[group] for chunk in range(chunks))
    require(len(seen) == m_tiles and len(copies) == m_tiles * world * chunks,
            "Incomplete OProj delivery plan")
    return tiles, windows, copies, use_cohorts


def score_oproj_schedule(*, m, n, k, world, sm_count, comm_ctas,
                         tile_m, tile_n, tile_k, raster, resolved_swizzle,
                         copy_chunks, copy_slots, cohort_m_tiles,
                         tile_cycle_us, copy_bandwidth_gb_s,
                         launch_us, copy_start_us, tail_us,
                         copy_chunk_bytes=None, include_trace=False,
                         service_basis="measured_cycle_and_steady_copy"):
    """Offline OProj service model, separate from the older calibrated APIs.

    All service constants are supplied by the caller from measured evidence;
    this function neither fits winners nor provides B300 timing defaults.
    ``tile_cycle_us`` is the no-feed-wait start-to-start interval, including
    necessary epilogue/pipeline work. It is split uniformly across peer K
    segments: no exact producer-prefetch, TMEM or Tensor Core busy claim.
    Supply the measured cycle for THIS actual compute-CTA budget. SM count
    changes the schedule below, never a linear throughput extrapolation;
    a caller using a measured budget curve retains its diminishing returns.

    ``copy_bandwidth_gb_s`` is the aggregate useful delivery rate for ALL A
    bytes (decimal GB/s), not remote-only NVLink bandwidth. A slot receives
    1/copy_slots of that rate. Fixed-stride tasks model last-chunk readiness,
    not dynamic bandwidth sharing or contention feedback. Use a measured
    steady service rate, not a total time already charged with startup/tail.
    An optional byte vector specifies the actual chunks of one full (M,peer);
    otherwise equal chunk sizes are an explicit approximation.

    launch_us is common startup; copy_start_us is additional producer startup;
    tail_us is only the unmodeled final drain. All are charged once. M must
    consist of complete M128 ready units; padded N work retains static CTA
    ownership. Non-cluster1, cyclic peer ordering and other precisions are not
    modeled. This is a prediction, not proof of overlap or a runtime selector.
    ``amortized_full_boundary`` instead accepts C/waves and A_bytes/R from
    independent whole-boundary measurements. It requires all three additional
    startup/tail terms to be zero to avoid adding them again; their separately
    measured values remain UNKNOWN, not measured zero. Such effective services
    do not become observed per-tile cycles or pure fabric bandwidth.
    """
    geometry = dict(m=m, n=n, k=k, world=world, sm_count=sm_count, comm_ctas=comm_ctas,
                    tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
                    resolved_swizzle=resolved_swizzle, copy_chunks=copy_chunks,
                    copy_slots=copy_slots, cohort_m_tiles=cohort_m_tiles)
    for field in geometry:
        require(integer(geometry, field) <= 2**31 - 1, "OProj model geometry exceeds int32")
    require(world in (4, 8) and comm_ctas < sm_count and tile_m == 128 and
            tile_n in (128, 256) and tile_k in (64, 128) and m % tile_m == 0 and
            k % (world * tile_k) == 0, "Unsupported OProj model geometry/ready boundary")
    require(raster in ("along_m", "along_n") and resolved_swizzle in (1, 2, 4, 8),
            "Unsupported resolved OProj schedule")
    require(copy_slots in (comm_ctas, 4 * comm_ctas), "Copy slots differ from vector/bulk CTA capacity")
    require(type(include_trace) is bool, "include_trace must be a bool")
    cycle = positive(tile_cycle_us, "tile cycle us")
    bandwidth = positive(copy_bandwidth_gb_s, "copy payload bandwidth GB/s")
    launch = nonnegative(launch_us, "common launch us")
    copy_start = nonnegative(copy_start_us, "copy startup us")
    tail = nonnegative(tail_us, "final tail us")
    require(service_basis in ("measured_cycle_and_steady_copy", "amortized_full_boundary"),
            "Unknown OProj service calibration basis")
    amortized = service_basis == "amortized_full_boundary"
    require(not amortized or launch == copy_start == tail == 0,
            "Amortized services cannot add startup/tail a second time")
    mt, nt = m // tile_m, ceil_div(n, tile_n)
    pm, pn = ceil_div(mt, resolved_swizzle) * resolved_swizzle, ceil_div(nt, resolved_swizzle) * resolved_swizzle
    require(pm * pn <= 1_000_000 and mt * world * copy_chunks <= 1_000_000 and copy_slots <= 1_000_000,
            "OProj model enumeration exceeds bounded offline scope")
    compute = min(pm * pn, sm_count - comm_ctas)
    ready_bytes = 2 * tile_m * k // world
    if copy_chunk_bytes is None:
        chunk_bytes = [ready_bytes / copy_chunks] * copy_chunks
    else:
        require(isinstance(copy_chunk_bytes, (tuple, list)) and len(copy_chunk_bytes) == copy_chunks,
                "Copy chunk byte vector has the wrong length")
        require(all(type(value) is int and value > 0 for value in copy_chunk_bytes) and
                sum(copy_chunk_bytes) == ready_bytes, "Copy chunks do not cover one complete (M,peer)")
        chunk_bytes = list(copy_chunk_bytes)
    tiles, windows, copies, use_cohorts = _oproj_delivery_plan(
        mt, nt, resolved_swizzle, compute, raster, world, copy_chunks, cohort_m_tiles)
    # Physical slot s processes task s, s+S, ... independently; no cohort barrier.
    slots = [launch + copy_start] * copy_slots
    releases = [[launch + copy_start] * world for _ in range(mt)]
    arrivals = [[0] * world for _ in range(mt)]
    for index, (mi, peer, chunk) in enumerate(copies):
        slot = index % copy_slots
        slots[slot] += chunk_bytes[chunk] / bandwidth * (copy_slots / 1000.0)
        releases[mi][peer] = max(releases[mi][peer], slots[slot])
        arrivals[mi][peer] += 1
    require(all(count == copy_chunks for peers in arrivals for count in peers),
            "Ready unit missing a copy arrival")
    copy_end = max(slots)
    peer_service = cycle / world
    worker_finishes, worker_waits, worker_services, worker_initial_waits = [], [], [], []
    phase_finishes, phase_services, phase_counts = [], [], []
    waits_by_peer = [0.0] * world
    consumption = []
    first_peers = []
    for worker in range(compute):
        now, waited, service, phase_service, phase_count = launch, 0.0, 0.0, 0.0, 0
        phase_end, initial_wait = launch, 0.0
        first = True
        for logical in range(worker, len(tiles), compute):
            coordinate = tiles[logical]
            if coordinate is None:
                continue
            mi, ni = coordinate
            if first:
                first_peers.append(releases[mi][0])
                initial_wait = max(0.0, releases[mi][0] - launch)
                first = False
            begin, tile_wait = now, 0.0
            for peer in range(world):
                wait = max(0.0, releases[mi][peer] - now)
                now += wait + peer_service
                waited += wait
                tile_wait += wait
                waits_by_peer[peer] += wait
            service += cycle
            # AlongM has a front-loaded N-band phase. AlongN interleaves N
            # reuse while progressing through M, so its feed phase is the
            # whole computation, not a fictitious early global N band.
            if raster == "along_n" or ni < resolved_swizzle:
                phase_service += cycle
                phase_count += 1
                phase_end = now
            if include_trace:
                consumption.append(dict(worker=worker, logical_tile=logical, m_tile=mi, n_tile=ni,
                                        begin_us=begin, finish_us=now, feed_wait_us=tile_wait))
        worker_finishes.append(now)
        worker_waits.append(waited)
        worker_services.append(service)
        worker_initial_waits.append(initial_wait)
        phase_finishes.append(phase_end)
        phase_services.append(phase_service)
        phase_counts.append(phase_count)
    critical_worker = max(range(compute), key=lambda worker: worker_finishes[worker])
    compute_end = worker_finishes[critical_worker]
    first_ready = min(first_peers)
    work_tiles = mt * nt
    waves = ceil_div(work_tiles, compute)
    # Keep the old integer-wave/start/drain estimate alongside the explicit DAG.
    # Padding can distribute valid tiles unevenly; the actual strided worker
    # service provides another necessary bound within this service model.
    wave_service = waves * cycle
    ideal = max(first_ready + wave_service, copy_end) + tail
    strided_service = max(worker_services)
    critical_path = max(compute_end, copy_end) + tail
    score = max(ideal, critical_path)
    payload = 2 * m * k
    production_span = copy_end - launch
    require(math.isfinite(production_span) and production_span > 0,
            "Copy service duration cannot be represented safely")
    phase_service = max(phase_services)
    demand_rate = payload / phase_service / 1000.0
    delivery_rate = payload / production_span / 1000.0
    result = dict(
        model="oproj_static_feed_service_v1", direction="A2A_GEMM", geometry=geometry,
        raster=raster, compute_ctas=compute, padded_m_tiles=pm, padded_n_tiles=pn,
        valid_work_tiles=work_tiles, scheduled_work_tiles=len(tiles), integer_waves=waves,
        delivery_policy="ready_cohorts" if use_cohorts else "group_peer_diagonal",
        first_use_windows=len(windows), base_cohort_m_tiles=max(1, copy_slots // copy_chunks),
        copy_tasks=len(copies), unique_payload_bytes=payload,
        remote_payload_bytes=payload * (world - 1) // world,
        tile_cycle_us=cycle, peer_service_us=peer_service, copy_bandwidth_gb_s=bandwidth,
        tile_cycle_budget_assumed=compute, linear_sm_throughput_scaling_applied=False,
        service_basis=service_basis,
        separately_calibrated_startup_tail_us=None if amortized else
            dict(launch=launch, copy_start=copy_start, tail=tail),
        launch_us=launch, copy_start_us=copy_start, tail_us=tail,
        copy_finish_us=copy_end, compute_finish_us=compute_end,
        first_ready_us=first_ready, wave_compute_service_us=wave_service,
        strided_compute_service_us=strided_service,
        ideal_overlap_us=ideal, critical_path_us=critical_path, score_us=score,
        critical_worker=critical_worker, critical_worker_service_us=worker_services[critical_worker],
        critical_worker_feed_wait_us=worker_waits[critical_worker],
        critical_worker_initial_wait_us=worker_initial_waits[critical_worker],
        critical_worker_later_feed_wait_us=max(0.0, worker_waits[critical_worker] - worker_initial_waits[critical_worker]),
        exposed_feed_us=max(0.0, compute_end - first_ready - strided_service),
        worker_wait_sum_us=sum(worker_waits), peer_wait_sum_us=waits_by_peer,
        first_n_band_compute_fraction=min(resolved_swizzle, nt) / nt,
        first_n_band_is_temporal_phase=(raster == "along_m"),
        feed_phase="first_n_band" if raster == "along_m" else "whole_gemm",
        feed_phase_compute_fraction=min(resolved_swizzle, nt) / nt if raster == "along_m" else 1.0,
        feed_phase_tiles=sum(phase_counts), feed_phase_ideal_compute_us=phase_service,
        feed_phase_predicted_finish_us=max(phase_finishes),
        feed_phase_requires_all_unique_a=True,
        feed_phase_demand_gb_s=demand_rate, effective_delivery_gb_s=delivery_rate,
        production_consumption_ratio=delivery_rate / demand_rate,
        whole_workload_production_consumption_ratio=delivery_rate / (payload / strided_service / 1000.0),
        services_supplied_by_caller=True, externally_validated=False,
        worker_wait_sum_is_global_stall=False, tensor_core_busy_inferred=False,
        assumptions=["uniform peer-K share of the caller-supplied tile service",
                     "caller supplies budget-matched measured services; no across-budget SM scaling",
                     "effective C/waves and A_bytes/R include amortized startup/tail" if amortized else
                     "measured no-feed-wait cycle and independently separated steady copy service",
                     "fixed slot bandwidth; no contention/prefetch feedback simulation",
                     "whole (M,peer) release after every original chunk; no new ready granularity",
                     "integer-wave score is a model scenario, not a measured hardware lower bound",
                     "equal copy chunks" if copy_chunk_bytes is None else "explicit copy chunk bytes"])
    require(all(math.isfinite(value) for value in (score, demand_rate, delivery_rate)),
            "OProj service arithmetic is not finite")
    if include_trace:
        result.update(copy_order=copies, ready_us_by_m_peer=releases,
                      first_use_m_windows=list(windows.values()),
                      worker_finish_us=worker_finishes, worker_service_us=worker_services,
                      worker_feed_wait_us=worker_waits, tile_consumption=consumption)
    return result


def comparison_problem_key(row):
    """Bind the normalized boundary to real model dimensions, not its label."""
    key = primitive_problem_key(row)
    require(row.get("batch", 1) == 1 and not any(row.get(field) for field in
            ("qkv_peer_interleaved", "defer_v_a2a", "packed_source_row", "packed_row_granularity")),
            "Comparison requires the complete unmodified forward route")
    direction, m, n, k, world, _ = key
    hidden, q_heads, kv_heads, head_dim = (integer(row, field) for field in
                                         ("hidden", "q_heads", "kv_heads", "head_dim"))
    require(integer(row, "seq_local") == m and integer(row, "global_seq") == m * world,
            "Comparison sequence/M/CP mismatch")
    require(q_heads % world == kv_heads % world == q_heads % kv_heads == 0 and head_dim % 8 == 0,
            "Comparison head geometry mismatch")
    expected = ((q_heads + 2 * kv_heads) * head_dim, hidden) if direction == "GEMM_A2A" else (hidden, q_heads * head_dim)
    require((n, k) == expected, "Comparison MNK differs from actual projection dimensions")
    # Q/K/V segment boundaries matter even when their combined N is equal.
    return key + ((q_heads, kv_heads, head_dim) if direction == "GEMM_A2A" else ())


def comparison_pair(row):
    """Validate an upstream-audited normalized MPI/torch-distributed pair.

    Common fields describe execution semantics. Each collector retains its
    own source/evidence and RNG provenance; neither seeds nor binaries must
    match. This function consumes accepted p50s, never combines rank samples.
    """
    common = row.get("comparison_contract")
    require(isinstance(common, dict) and set(common) ==
            {"node", "precision", "launch", "process_layout", "boundary", "cadence", "input_distribution"} and
            row.get("baseline_comparison_contract") == common, "Missing/mixed comparison contract")
    require(common["node"] in ("09", "0a") and common["precision"] == "bf16_accfp32_bf16" and
            common["launch"] == "eager" and common["process_layout"] == "one_process_per_gpu" and
            common["boundary"] == "full_direction_maxrank_cuda_event", "Unsupported common comparison boundary")
    require(common["cadence"] == dict(minimum_warmup=10, samples=50, minimum_warmup_cuda_ms=100,
                                     warmup_range_limit=.05, half_drift_limit=.05, maximum_rounds=3,
                                     barrier="per_sample", selection="first_stable_round") and
            common["input_distribution"] == dict(kind="uniform", activation_amplitude=.125, weight_amplitude=.02),
            "Unmatched comparison cadence/input distribution")
    fused, baseline = row.get("measurement_provenance"), row.get("baseline_measurement_provenance")
    for provenance in (fused, baseline):
        require(isinstance(provenance, dict) and set(provenance) ==
                {"collector", "host_launch", "backend", "run_id", "source_id", "evidence_sha256", "input_generator", "seed"},
                "Missing collector provenance")
        require(isinstance(provenance["run_id"], str) and provenance["run_id"] and
                type(provenance["seed"]) is int and provenance["seed"] >= 0, "Invalid run/RNG provenance")
        hash_value(provenance["source_id"])
        hash_value(provenance["evidence_sha256"])
    require((fused["collector"], fused["host_launch"], fused["backend"]) ==
            ("mpi_rank_events_v1", "mpi_process", "fuse") and
            fused["input_generator"] in ("cpu_mt19937", "gpu_philox") and
            (baseline["collector"], baseline["host_launch"]) ==
            ("primed_events_vector_max_v1", "torch_distributed") and
            baseline["backend"] in ("te_ub", "cublaslt_nccl") and baseline["input_generator"] == "torch_uniform",
            "Unsupported collector/process pairing")
    require(all(row.get(field) == fused[field] for field in ("host_launch", "collector")) and
            all(field not in row or row[field] == common[field] for field in ("node", "precision", "launch")),
            "Incumbent comparison provenance mismatch")
    baseline_problem = row.get("baseline_problem")
    require(isinstance(baseline_problem, dict), "Missing actual baseline geometry")
    key = comparison_problem_key(row)
    require(comparison_problem_key(baseline_problem) == key, "Baseline physical geometry/layout differs")
    if key[0] == "A2A_GEMM":
        # Legacy NCCL/UB have different dual-chunk permutations, neither the
        # fused rank-major layout nor the explicitly aligned causal contract.
        require(baseline_problem.get("oproj_layout") == "causal_dual_chunk_v1" and
                key[5] == "causal_dual_chunk_v1" and row["seq_local"] % 2 == 0,
                "Baseline native OProj layout is not the matched causal permutation")
    return common, key, fused, baseline


def badcase_targets(measurements):
    """Direction GM and min(GM, supplied ideal cap) for accepted paired p50s.

    Formal comparisons currently admit only the verified MPI fused and
    torch-distributed baseline collectors. Old single-process results remain
    valid diagnostics in their original reports, not this MPI evaluation set.
    """
    groups, seen, contract = {}, set(), None
    for row in measurements:
        current_contract, key, fused_provenance, baseline_provenance = comparison_pair(row)
        require(key not in seen, "Duplicate physical workload (including model aliases)")
        seen.add(key)
        require(row.get("performance_accepted") is True, "Unaccepted incumbent measurement")
        require(row.get("baseline_performance_accepted") is True, "Unaccepted baseline measurement")
        require(contract is None or contract == current_contract, "Missing/mixed comparison contract")
        contract = current_contract
        speedup = positive(positive(row.get("baseline_ms"), "baseline ms") /
                           positive(row.get("fused_ms"), "fused ms"), "paired speedup")
        upper = row.get("upper_speedup")
        if upper is not None:
            positive(upper, "supplied ideal upper speedup")
        groups.setdefault(key[0], []).append({"problem": key, "speedup": speedup, "upper_speedup": upper,
                                             "fused_provenance": dict(fused_provenance),
                                             "baseline_provenance": dict(baseline_provenance)})
    require(groups, "Empty badcase evaluation set")
    result = {}
    for direction, rows in groups.items():
        mean = math.exp(sum(math.log(row["speedup"]) for row in rows) / len(rows))
        for row in rows:
            inconsistent = row["upper_speedup"] is not None and row["upper_speedup"] < row["speedup"]
            target = None if row["upper_speedup"] is None or inconsistent else min(mean, row["upper_speedup"])
            row.update(below_direction_geomean=row["speedup"] < mean, target_speedup=target,
                       target_status="inconsistent_upper" if inconsistent else "unknown" if target is None else "supplied_ideal_cap",
                       upper_below_observed=inconsistent,
                       positive_log_gap=None if target is None else max(0.0, math.log(target / row["speedup"])))
        result[direction] = {"count": len(rows), "geomean_speedup": mean, "workloads": rows}
    return {"comparison_contract": contract, "directions": result, "full_historical_coverage_claimed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inspect", "fit"))
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, nargs="*", default=[])
    parser.add_argument("--model-form", choices=MODEL_FORMS, default="nonnegative_time_ridge")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    require((args.mode == "fit") == (args.output is not None), "Only explicit fit writes one --output report")
    require(args.mode == "fit" or not args.heldout, "Inspect does not consume held-out measurements")
    training = read_summary(args.summary)
    if args.mode == "inspect":
        print(json.dumps({"fitted": False, "fused_rows": len(training.observations),
                          "execution_contract": training.contract, "provenance": training.provenance,
                          "families": {d: sorted({family_key(o.row) for o in training.observations if o.row["direction"] == d})
                                       for d in DIRECTIONS}}, indent=2))
        return
    require(not args.output.exists() and not args.output.is_symlink(), "Refusing to overwrite a model report")
    result = experiment(training, [read_summary(path) for path in args.heldout], args.model_form)
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(encoded)
    print(f"Offline experiment written: {args.output}; no runtime selector change")


if __name__ == "__main__":
    main()
