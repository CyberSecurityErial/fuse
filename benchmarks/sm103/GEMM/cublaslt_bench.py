#!/usr/bin/env python3
"""Single-GPU cuBLASLt search; optionally reuse one process across BF16 shapes."""
import argparse
import contextlib
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import time

import torch
from cublaslt import Library, Operand, Plan, check_gemm

PROTOCOL = "single_gpu_pure_gemm_stable_v3_launch_tuned"
COUNTER_RANGE = "fuse_cutlass_1sm_counters"


def percentile(samples, q):
    values = sorted(samples)
    position = (len(values) - 1) * q
    lo, hi = int(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def matrix_shapes(payload):
    """Deduplicate exact M/N/K only; retain every logical identifier."""
    if (not isinstance(payload, dict) or payload.get("schema") != "sm103_gemm_matrix_v1"
            or not isinstance(payload.get("shapes"), list) or not 1 <= len(payload["shapes"]) <= 256):
        raise ValueError("expected sm103_gemm_matrix_v1 with 1..256 shapes")
    groups, identifiers = {}, set()
    for row in payload["shapes"]:
        if not isinstance(row, dict) or set(row) != {"id", "m", "n", "k"}:
            raise ValueError("each matrix row requires exactly id/m/n/k")
        label = row["id"]
        if (not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", label)
                or label in identifiers):
            raise ValueError("matrix ids must be unique nonempty safe labels")
        if any(type(row[key]) is not int or not 1 <= row[key] <= 2**31 - 1 for key in ("m", "n", "k")):
            raise ValueError("matrix dimensions must be positive int32 values")
        identifiers.add(label)
        key = tuple(row[name] for name in ("m", "n", "k"))
        group = groups.setdefault(key, {"shape": dict(zip(("m", "n", "k"), key)), "aliases": []})
        group["aliases"].append(label)
    return list(groups.values())


class MeasurementFailure(RuntimeError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def collect_samples(run, count):
    # Prime lazy CUDA handles outside timed intervals. There is no distributed
    # initialization, communication, quantization or correctness in this region.
    pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
             for _ in range(count)]
    for start, end in pairs:
        start.record()
        end.record()
    torch.cuda.synchronize()
    gc_enabled = gc.isenabled()
    try:
        gc.disable()
        for start, end in pairs:
            start.record()
            run()
            end.record()
            end.synchronize()
    finally:
        if gc_enabled:
            gc.enable()
    samples = [start.elapsed_time(end) for start, end in pairs]
    if any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("invalid CUDA event sample")
    return samples


def measure(run, warmup, iterations):
    evidence = {"protocol": PROTOCOL, "minimum_warmup_cuda_ms": 100,
                "relative_range_limit": .05, "warmup_timeout_s": 5,
                "initial_warmup": warmup, "iterations": iterations,
                "window_ms_per_call": [], "additional_warmup_calls": 0,
                "additional_warmup_cuda_ms": 0.0, "measurement_rounds": [],
                "collector": "single_gpu_primed_events_v2"}
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    end.record()
    end.synchronize()
    begun, count = time.monotonic(), 10
    while True:
        start.record()
        for _ in range(count):
            run()
        end.record()
        end.synchronize()
        elapsed = start.elapsed_time(end)
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise MeasurementFailure("invalid CUDA warmup time", evidence)
        evidence["additional_warmup_calls"] += count
        evidence["additional_warmup_cuda_ms"] += elapsed
        windows = evidence["window_ms_per_call"]
        windows.append(elapsed / count)
        stable = (len(windows) >= 3 and
                  (max(windows[-3:]) - min(windows[-3:])) / percentile(windows[-3:], .5) <= .05)
        if evidence["additional_warmup_cuda_ms"] >= 100 and stable:
            break
        if time.monotonic() - begun >= 5:
            raise MeasurementFailure("warmup did not converge within 5s", evidence)
        count = max(10, min(1000, math.ceil(20 / max(windows[-1], .001))))
    evidence["warmup_wall_s"] = time.monotonic() - begun
    evidence["warmup_converged"] = True
    for attempt in range(3):
        # Each round has 10+50 cadence. Keep rejected rounds; never select the
        # fastest round or relax the threshold after noisy measurements.
        cadence = collect_samples(run, warmup)
        samples = collect_samples(run, iterations)
        middle = iterations // 2
        first, second = percentile(samples[:middle], .5), percentile(samples[middle:], .5)
        drift = abs(second - first) / percentile(samples, .5)
        evidence["measurement_rounds"].append({"sample_cadence_warmup_ms": cadence,
            "samples_ms": samples, "half_p50_relative_drift": drift})
        if drift <= .05:
            evidence["selected_round"] = attempt
            evidence["sample_half_p50_relative_drift"] = drift
            return samples, evidence
    raise MeasurementFailure("measurement drift exceeds 5% in all 3 rounds", evidence)


def input_tensor(shape, magnitude, uniform):
    if not uniform:
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16) * magnitude
    tensor = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
    flat = tensor.view(-1)
    # Match training bounds and FP32-uniform -> BF16 conversion. PyTorch's
    # generator sequence is not bit-identical to the fused harness.
    for begin in range(0, flat.numel(), 1 << 20):
        target = flat[begin:begin + (1 << 20)]
        values = torch.empty(target.numel(), device="cuda", dtype=torch.float32)
        target.copy_(values.uniform_(-magnitude, magnitude))
    return tensor


def input_statistics(tensor):
    flat = tensor.reshape(-1)
    sample = flat[::max(1, flat.numel() // 4096)][:4096].float()
    nonzero = int(torch.count_nonzero(sample).item())
    if not bool(torch.isfinite(sample).all()) or nonzero == 0:
        raise ValueError("nonfinite or all-zero sampled GEMM input")
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sample_count": sample.numel(),
            "sample_nonzero_fraction": nonzero / sample.numel(), "sample_min": sample.min().item(),
            "sample_max": sample.max().item(), "sample_mean": sample.mean().item(),
            "sample_std": sample.std(unbiased=False).item()}


def prepare_inputs(geometry, uniform):
    m, n, k = (geometry["shape"][key] for key in ("m", "n", "k"))
    torch.manual_seed(103)
    source = input_tensor((m, k), .125, uniform)
    weight = input_tensor((n, k), .02, uniform)
    geometry["inputs"] = {"seed": 103, "generator": "torch_cuda",
        "distribution": "uniform" if uniform else "normal",
        "activation_magnitude": .125, "weight_magnitude": .02,
        "magnitude_meaning": "uniform_half_range" if uniform else "normal_stddev",
        "fused_bitwise_identical": False,
        "activation": input_statistics(source), "weight": input_statistics(weight)}
    return source, weight


def prepare_backward_inputs(geometry, uniform, operand_layout):
    if operand_layout not in ('nn', 'tn'):
        raise ValueError('backward operands require NN or TN layout')
    m, n, k = (geometry["shape"][key] for key in ("m", "n", "k"))
    torch.manual_seed(103)
    # Logical operation stays X[M,K] * W[N,K]^T. Only allocation/view strides
    # change: NN dgrad reads stored forward weights; TN wgrad reads dY^T/X.
    source = input_tensor((k, m) if operand_layout == 'tn' else (m, k), .125, uniform)
    weight = input_tensor((k, n), .02, uniform)
    # Inspect physical storage before creating views; reshape(logical_view)
    # would silently materialize a long-sequence transpose just for statistics.
    source_stats, weight_stats = input_statistics(source), input_statistics(weight)
    if operand_layout == 'tn':
        source = source.T
    weight = weight.T
    geometry["inputs"] = {"seed": 103, "generator": "torch_cuda",
        "distribution": "uniform" if uniform else "normal",
        "activation_magnitude": .125, "weight_magnitude": .02,
        "magnitude_meaning": "uniform_half_range" if uniform else "normal_stddev",
        "fused_bitwise_identical": False,
        "operand_layout": operand_layout, "transpose_materialized": False,
        "activation_stride": list(source.stride()), "weight_stride": list(weight.stride()),
        "activation_storage": source_stats, "weight_storage": weight_stats,
        "activation_shape": list(source.shape), "weight_shape": list(weight.shape)}
    return source, weight


def run_geometry(a, lib, geometry, index, total):
    m, n, k = (geometry["shape"][key] for key in ("m", "n", "k"))
    sm_target = getattr(a, "cublaslt_sm_target", None)
    budget = getattr(a, "sm_budget_context", None)
    if getattr(a, 'operand_layout', 'nt') != 'nt':
        source, weight = prepare_backward_inputs(geometry, a.matrix_json is not None, a.operand_layout)
    else:
        source, weight = prepare_inputs(geometry, a.matrix_json is not None)
    for precision in a.precisions.split(","):
        x, w = Operand(lib, source, precision), Operand(lib, weight, precision)
        output = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        print(f"{precision} RUN {index}/{total} shape={m}x{n}x{k} "
              f"aliases={','.join(geometry['aliases'])} candidates={a.candidates}", flush=True)
        plan = None
        try:
            for launch in a.launches.split(","):
                # Tune in the measured launch mode, without reallocating inputs.
                # Eager and Graph may select different algorithms.
                if plan is not None:
                    plan.close()
                    plan = None
                # Native candidate capture cannot use CUDA's legacy default
                # stream. Join input initialization, then tune on a side stream.
                tuning_stream = (budget.stream if budget else torch.cuda.Stream()) if launch == 'graph' else None
                if tuning_stream is not None:
                    torch.cuda.synchronize()
                with (torch.cuda.stream(tuning_stream) if tuning_stream is not None
                      else contextlib.nullcontext()):
                    plan = Plan(lib, x, w, output, candidates=a.candidates, workspace_mib=a.workspace_mib,
                                warmup=a.tune_warmup, iterations=a.tune_iterations, graph=launch == 'graph',
                                math_sms=sm_target or 0)
                if tuning_stream is not None:
                    tuning_stream.synchronize()
                correctness = check_gemm(plan)
                run, graph = plan.run, None
                launch_evidence = None
                try:
                    if launch == "graph":
                        for _ in range(a.warmup):
                            plan.run()
                        torch.cuda.synchronize()
                        # Keep the exact measured graph for a post-timing launch
                        # audit. SM_COUNT_TARGET is a heuristic hint, not affinity.
                        graph = (torch.cuda.CUDAGraph(keep_graph=True) if sm_target is not None
                                 else torch.cuda.CUDAGraph())
                        with (torch.cuda.graph(graph, stream=budget.stream) if budget
                              else torch.cuda.graph(graph)):
                            plan.run()
                        if sm_target is not None:
                            graph.instantiate()
                        run = graph.replay
                        # Check the captured callable, not a fresh eager GEMM
                        # that would overwrite and mask a broken graph output.
                        correctness = check_gemm(argparse.Namespace(
                            x=plan.x, weight=plan.weight, output=plan.output, run=run))
                    samples, evidence = measure(run, a.warmup, a.iterations)
                    if getattr(a, "cublaslt_counters", False):
                        # Select the warmed, tuned callable (including its green
                        # context), never an unrelated heuristic/eager launch.
                        torch.cuda.synchronize()
                        torch.cuda.nvtx.range_push("fuse_cublaslt_counters")
                        try:
                            run()
                            torch.cuda.synchronize()
                        finally:
                            torch.cuda.nvtx.range_pop()
                        geometry["counter_diagnostic"] = {
                            "backend": "cublaslt", "launch": launch,
                            "diagnostic_only": True, "range_launches": 1,
                            "nvtx_range": "fuse_cublaslt_counters",
                            "correctness_post": check_gemm(argparse.Namespace(
                                x=plan.x, weight=plan.weight, output=plan.output,
                                run=lambda: plan.output))}
                    if graph is not None and sm_target is not None:
                        dot = a.output.with_name(f"{a.output.stem}-{index}-{precision}-launch.dot")
                        graph.debug_dump(str(dot))
                        if not dot.is_file() or not dot.stat().st_size:
                            raise RuntimeError("measured GEMM launch audit was not written")
                        launch_evidence = {"file": dot.name,
                            "sha256": hashlib.sha256(dot.read_bytes()).hexdigest(),
                            "source": "measured_cuda_graph_verbose_dot",
                            "hard_sm_partition_verified": False}
                except MeasurementFailure as error:
                    geometry["failed_measurement"] = {"precision": precision, "launch": launch,
                                                       "error": str(error), "measurement": error.evidence}
                    raise
                finally:
                    del run, graph
                p50 = percentile(samples, .5)
                record = {"precision": precision, "launch": launch, "warmup": a.warmup,
                    "samples_ms": samples, "p50_ms": p50, "p95_ms": percentile(samples, .95),
                    "correctness": correctness, "tuning": dict(plan.info, replay=launch),
                    "tune_warmup": a.tune_warmup, "tune_iterations": a.tune_iterations,
                    "measurement": evidence, "tflops_p50": 2 * m * n * k / p50 / 1e9,
                    "pflops_per_gpu_p50": 2 * m * n * k / p50 / 1e12}
                if sm_target is not None:
                    record["sm_budget"] = {"requested_sm_target": sm_target,
                        "enforcement": "cublaslt_heuristic_hint_not_hard_partition",
                        "launch_evidence": launch_evidence}
                    if budget:
                        record["sm_budget"].update(budget.info)
                geometry["results"].append(record)
                print(f"{precision} DONE {index}/{total} shape={m}x{n}x{k} launch={launch} "
                      f"p50={p50:.6f}ms p95={record['p95_ms']:.6f}ms "
                      f"PFLOPS/GPU={record['pflops_per_gpu_p50']:.6f} valid={plan.info['valid']}", flush=True)
        finally:
            if plan is not None:
                plan.close()
        del plan, x, w, output
    # No GPU objects escape this function. Reuse the allocator across shapes;
    # do not retain every shape's buffers or empty its cache after each sample.


def run_cutlass_comparison(a, lib, geometry, index, total, cutlass_library):
    """Fixed three-way BF16 diagnostic; never replaces a pure-Lt table entry.

    All plans share the same actual input/output buffers. Forward/reverse
    blocks expose order effects instead of selecting the faster repetition.
    The existing stable sampler and numerical checker are reused unchanged.
    """
    from cutlass import Plan as CutlassPlan
    m, n, k = (geometry["shape"][key] for key in ("m", "n", "k"))
    source, weight = prepare_inputs(geometry, True)
    x, w = Operand(lib, source, "bf16"), Operand(lib, weight, "bf16")
    output = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    plans = {}
    active = {"phase": "create", "backend": "cublaslt"}
    geometry["comparison"] = {"scope": "single_gpu_stock_gemm_not_fused_C_reference",
        "same_input_buffers": True, "same_output_buffer": True,
        "physical_sm_budget": "all" if not getattr(a, "cutlass_sm_budget", 0) else "per_backend",
        "cutlass_sm_budget": getattr(a, "cutlass_sm_budget", 0),
        "cublaslt_sm_budget": "all",
        "blocks": [["cublaslt", "cutlass_1sm", "cutlass_2sm"],
                   ["cutlass_2sm", "cutlass_1sm", "cublaslt"]],
        "repetition_selection": "none_all_blocks_retained", "cutlass_schedule": "static_persistent",
        "cutlass_max_swizzle_size": a.cutlass_swizzle_size,
        "cutlass_epilogue_n": a.cutlass_epilogue_n,
        "cutlass_1sm_cluster_m": a.cutlass_1sm_cluster_m,
        "full_check_requested": a.cutlass_full_check}
    try:
        plans["cublaslt"] = Plan(lib, x, w, output, candidates=a.candidates,
            workspace_mib=a.workspace_mib, warmup=a.tune_warmup, iterations=a.tune_iterations, graph=False)
        for mode in (1, 2):
            active = {"phase": "create", "backend": f"cutlass_{mode}sm"}
            plans[f"cutlass_{mode}sm"] = CutlassPlan(cutlass_library, x, w, output,
                                                   sm_mode=mode, max_swizzle_size=a.cutlass_swizzle_size,
                                                   epilogue_n=a.cutlass_epilogue_n,
                                                   cluster_m=a.cutlass_1sm_cluster_m if mode == 1 else 2,
                                                   sm_budget=getattr(a, 'cutlass_sm_budget', 0))
        for block, order in enumerate(geometry["comparison"]["blocks"]):
            for backend in order:
                plan = plans[backend]
                active = {"phase": "correctness_pre", "backend": backend, "comparison_block": block}
                print(f"bf16 RUN {index}/{total} shape={m}x{n}x{k} backend={backend} block={block}", flush=True)
                # A previous backend's correct output must not mask a no-op.
                output.fill_(float("nan"))
                correctness_pre = check_gemm(plan, full=True) if a.cutlass_full_check else check_gemm(plan)
                try:
                    active["phase"] = "measure"
                    samples, evidence = measure(plan.run, a.warmup, a.iterations)
                except MeasurementFailure as error:
                    geometry["failed_measurement"] = {"precision": "bf16", "launch": "eager",
                        "backend": backend, "comparison_block": block,
                        "error": str(error), "measurement": error.evidence}
                    raise
                active.update(phase="correctness_post", samples_ms=samples, measurement=evidence)
                # Inspect the last timed result; a fresh GEMM must not mask it.
                saved_output = argparse.Namespace(
                    x=plan.x, weight=plan.weight, output=plan.output, run=lambda: plan.output)
                correctness_post = (check_gemm(saved_output, full=True) if a.cutlass_full_check
                                    else check_gemm(saved_output))
                p50 = percentile(samples, .5)
                record = {"precision": "bf16", "launch": "eager", "backend": backend,
                    "comparison_block": block, "warmup": a.warmup, "samples_ms": samples,
                    "p50_ms": p50, "p95_ms": percentile(samples, .95), "measurement": evidence,
                    "correctness": correctness_pre, "correctness_post": correctness_post,
                    "plan": dict(plan.info), "tuning_performed": backend == "cublaslt",
                    "tune_warmup": a.tune_warmup if backend == "cublaslt" else None,
                    "tune_iterations": a.tune_iterations if backend == "cublaslt" else None,
                    "tflops_p50": 2 * m * n * k / p50 / 1e9,
                    "pflops_per_gpu_p50": 2 * m * n * k / p50 / 1e12}
                geometry["results"].append(record)
                print(f"bf16 DONE {index}/{total} shape={m}x{n}x{k} backend={backend} block={block} "
                      f"p50={p50:.6f}ms p95={record['p95_ms']:.6f}ms "
                      f"PFLOPS/GPU={record['pflops_per_gpu_p50']:.6f}", flush=True)
    except Exception as error:
        # Keep failed post-check samples as diagnostic evidence, never an
        # accepted result. Likewise identify plan creation/pre-check failures.
        geometry.setdefault("failed_measurement", dict(active, error=str(error), performance_accepted=False))
        raise
    finally:
        for plan in reversed(list(plans.values())):
            plan.close()


def warmup_counters(run, warmup):
    """Same warmup as measure(), without entering its formal sample rounds.

    Keep the production sampler unchanged; an AST contract test checks this
    short diagnostic copy against its warmup body.
    """
    evidence = {"protocol": "single_gpu_counter_warmup_v1", "minimum_warmup_cuda_ms": 100,
                "relative_range_limit": .05, "warmup_timeout_s": 5,
                "initial_warmup": warmup, "window_ms_per_call": [],
                "additional_warmup_calls": 0, "additional_warmup_cuda_ms": 0.0}
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    end.record()
    end.synchronize()
    begun, count = time.monotonic(), 10
    while True:
        start.record()
        for _ in range(count):
            run()
        end.record()
        end.synchronize()
        elapsed = start.elapsed_time(end)
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise MeasurementFailure("invalid CUDA warmup time", evidence)
        evidence["additional_warmup_calls"] += count
        evidence["additional_warmup_cuda_ms"] += elapsed
        windows = evidence["window_ms_per_call"]
        windows.append(elapsed / count)
        stable = (len(windows) >= 3 and
                  (max(windows[-3:]) - min(windows[-3:])) / percentile(windows[-3:], .5) <= .05)
        if evidence["additional_warmup_cuda_ms"] >= 100 and stable:
            break
        if time.monotonic() - begun >= 5:
            raise MeasurementFailure("warmup did not converge within 5s", evidence)
        count = max(10, min(1000, math.ceil(20 / max(windows[-1], .001))))
    evidence["warmup_wall_s"] = time.monotonic() - begun
    evidence["warmup_converged"] = True
    return evidence


def run_cutlass_counters(a, lib, geometry, cutlass_library):
    """One warmed 1-SM invocation for external NCU; never a performance sample."""
    from cutlass import Plan as CutlassPlan
    m, n, k = (geometry["shape"][key] for key in ("m", "n", "k"))
    diagnostic = geometry["counter_diagnostic"] = {
        "diagnostic_only": True, "state": "running", "phase": "input",
        "backend": "cutlass_1sm", "precision": "bf16", "launch": "eager",
        "scope": "single_gpu_stock_gemm_not_fused_C_reference",
        "physical_sm_budget": getattr(a, "cutlass_sm_budget", 0) or "all",
        "nvtx_range": COUNTER_RANGE,
        "nvtx_range_kind": "push_pop", "range_launches": 0,
        "cutlass_max_swizzle_size": a.cutlass_swizzle_size,
        "cutlass_epilogue_n": a.cutlass_epilogue_n,
        "cutlass_1sm_cluster_m": a.cutlass_1sm_cluster_m,
        "full_check_requested": a.cutlass_full_check,
        "tuning_performed": False, "formal_sampling_performed": False,
        "ncu_metrics_verified": False,
        "correctness_scope": ("existing_check_gemm_full_small_bf16" if a.cutlass_full_check
                              else "existing_check_gemm_max64x64_evenly_spaced_values"),
        "cache_preparation": "steady_warmup_then_output_nan_fill; no_explicit_cache_flush"}
    plan = None
    try:
        source, weight = prepare_inputs(geometry, True)
        x, w = Operand(lib, source, "bf16"), Operand(lib, weight, "bf16")
        output = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        diagnostic["phase"] = "create"
        plan = CutlassPlan(cutlass_library, x, w, output, sm_mode=1,
                           max_swizzle_size=a.cutlass_swizzle_size, epilogue_n=a.cutlass_epilogue_n,
                           cluster_m=a.cutlass_1sm_cluster_m,
                           sm_budget=getattr(a, 'cutlass_sm_budget', 0))
        try:
            diagnostic["plan"] = dict(plan.info)
            print(f"bf16 COUNTERS RUN shape={m}x{n}x{k} backend=cutlass_1sm "
                  f"swizzle={a.cutlass_swizzle_size} epilogue_n={a.cutlass_epilogue_n} "
                  f"cluster_m={a.cutlass_1sm_cluster_m}", flush=True)
            diagnostic["phase"] = "correctness_pre"
            output.fill_(float("nan"))
            diagnostic["correctness_pre"] = (check_gemm(plan, full=True) if a.cutlass_full_check
                                               else check_gemm(plan))
            diagnostic["phase"] = "warmup"
            diagnostic["warmup"] = warmup_counters(plan.run, a.warmup)
            diagnostic["phase"] = "nvtx_launch"
            # A no-op must not reuse the preceding warmup's correct result.
            # This output-only fill changes cache state and is recorded above.
            output.fill_(float("nan"))
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push(COUNTER_RANGE)
            try:
                plan.run()
            finally:
                torch.cuda.nvtx.range_pop()
            diagnostic["range_launches"] = 1
            torch.cuda.synchronize()
            diagnostic["phase"] = "correctness_post"
            saved_output = argparse.Namespace(
                x=plan.x, weight=plan.weight, output=plan.output, run=lambda: plan.output)
            diagnostic["correctness_post"] = (check_gemm(saved_output, full=True) if a.cutlass_full_check
                                                else check_gemm(saved_output))
            diagnostic["phase"] = "cleanup"
        finally:
            # Also wait before freeing a plan if enqueue or validation failed.
            try:
                torch.cuda.synchronize()
            finally:
                plan.close()
        diagnostic.update(state="succeeded", phase="complete")
        print(f"bf16 COUNTERS DONE shape={m}x{n}x{k} nvtx_range={COUNTER_RANGE} "
              "range_launches=1 diagnostic_only=1", flush=True)
    except Exception as error:
        diagnostic.update(state="failed", error=str(error))
        if isinstance(error, MeasurementFailure):
            diagnostic["warmup"] = error.evidence
        raise


def save_result(path, result, *, initial=False):
    if initial:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as output:
            json.dump(result, output, indent=2)
            output.write("\n")
    else:
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("x") as output:
            json.dump(result, output, indent=2)
            output.write("\n")
        temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name, default in (("m", None), ("n", None), ("k", None), ("warmup", 10), ("iterations", 50),
                          ("candidates", 256), ("workspace-mib", 256), ("tune-warmup", 10), ("tune-iterations", 50)):
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--matrix-json", type=Path)
    p.add_argument("--precisions")
    p.add_argument('--operand-layout', choices=('nt', 'nn', 'tn'), default='nt',
                   help='BF16 row-major logical GEMM: forward NT, dgrad NN, wgrad TN; zero-copy views')
    p.add_argument("--launches", default="eager,graph")
    p.add_argument("--library", type=Path)
    p.add_argument("--compare-cutlass-library", type=Path,
                   help="explicit BF16 matrix/eager comparison: cuBLASLt vs stock CUTLASS 1-SM/2-SM")
    p.add_argument("--cutlass-counters", action="store_true",
                   help="counter-only 1-SM NVTX launch; requires comparison library and one BF16/eager geometry")
    p.add_argument("--cublaslt-counters", action="store_true",
                   help="NCU diagnostic of the warmed tuned Graph; timings are not benchmark results")
    p.add_argument("--cutlass-swizzle-size", type=int, choices=(1, 2, 4, 8),
                   help="explicit max swizzle for both CUTLASS plans only (default: 1)")
    p.add_argument("--cutlass-epilogue-n", type=int, choices=(32, 64),
                   help="explicit epilogue N for both CUTLASS plans only (default: 32)")
    p.add_argument("--cutlass-1sm-cluster-m", type=int, choices=(1, 2),
                   help="1-SM MMA multicast cluster M only; 2-SM stays cluster2 (default: 1)")
    p.add_argument("--cutlass-sm-budget", type=int, default=0,
                   help="persistent CUTLASS CTA budget; zero uses all SMs, no SM affinity")
    p.add_argument("--cublaslt-sm-target", type=int,
                   help="explicit Graph-only Lt SM heuristic target; 0=full device; audit launch grid")
    p.add_argument("--gemm-sm-budget", type=int,
                   help="Graph-only process-local green-context SM budget; exact count required")
    p.add_argument("--cutlass-full-check", action="store_true",
                   help="explicit small BF16 comparison: check every output, outside timing")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or a.output.with_name(a.output.name + ".tmp").exists():
        p.error("output or checkpoint exists; choose a fresh file")
    if a.precisions is None:
        a.precisions = "bf16" if a.matrix_json else "bf16,fp8,fp4"
    if not set(a.precisions.split(",")) <= {"bf16", "fp8", "fp4"}:
        p.error("invalid precision")
    if not set(a.launches.split(",")) <= {"eager", "graph"}:
        p.error("invalid launch")
    if a.operand_layout != 'nt' and (a.precisions != 'bf16' or a.compare_cutlass_library or
                                    a.cutlass_counters or a.cublaslt_counters):
        p.error('backward operand views require standalone BF16 GEMM without counters')
    if (len(set(a.launches.split(","))) != len(a.launches.split(",")) or
            len(set(a.precisions.split(","))) != len(a.precisions.split(","))):
        p.error("duplicate precision or launch")
    if min(a.warmup, a.tune_warmup) < 10 or min(a.iterations, a.tune_iterations) < 50:
        p.error("at least 10+50 required for tuning and formal measurement")
    if not 1 <= a.candidates <= 1024 or a.workspace_mib < 0:
        p.error("invalid candidates/workspace")
    if a.compare_cutlass_library and (not a.matrix_json or a.precisions != "bf16" or a.launches != "eager"):
        p.error("CUTLASS comparison requires explicit BF16 matrix and eager only")
    if a.cutlass_sm_budget < 0 or (a.cutlass_sm_budget and not a.compare_cutlass_library):
        p.error("--cutlass-sm-budget requires comparison library and a nonnegative budget")
    if a.cublaslt_sm_target is not None and (a.cublaslt_sm_target < 0 or a.launches != "graph"
                                           or a.compare_cutlass_library):
        p.error("--cublaslt-sm-target requires standalone Graph GEMM and a nonnegative target")
    if a.gemm_sm_budget is not None:
        if (a.gemm_sm_budget < 1 or a.launches != 'graph' or a.compare_cutlass_library or
                a.cublaslt_sm_target not in (None, a.gemm_sm_budget)):
            p.error('--gemm-sm-budget requires standalone Graph and a matching Lt target')
        a.cublaslt_sm_target = a.gemm_sm_budget
    if a.cutlass_counters and not a.compare_cutlass_library:
        p.error("--cutlass-counters requires --compare-cutlass-library")
    if a.cublaslt_counters and (a.compare_cutlass_library or a.launches != 'graph'
                              or a.precisions != 'bf16'):
        p.error("--cublaslt-counters requires standalone BF16 Graph")
    if a.cutlass_swizzle_size is not None and not a.compare_cutlass_library:
        p.error("--cutlass-swizzle-size requires --compare-cutlass-library")
    if a.cutlass_swizzle_size is None:
        a.cutlass_swizzle_size = 1
    if a.cutlass_epilogue_n is not None and not a.compare_cutlass_library:
        p.error("--cutlass-epilogue-n requires --compare-cutlass-library")
    if a.cutlass_epilogue_n is None:
        a.cutlass_epilogue_n = 32
    if a.cutlass_1sm_cluster_m is not None and not a.compare_cutlass_library:
        p.error("--cutlass-1sm-cluster-m requires --compare-cutlass-library")
    if a.cutlass_1sm_cluster_m is None:
        a.cutlass_1sm_cluster_m = 1
    if a.cutlass_1sm_cluster_m == 2 and a.cutlass_epilogue_n != 32:
        p.error("1-SM cluster2 comparison requires --cutlass-epilogue-n 32")
    if a.cutlass_full_check and not a.compare_cutlass_library:
        p.error("--cutlass-full-check requires --compare-cutlass-library")
    if a.matrix_json:
        if any(value is not None for value in (a.m, a.n, a.k)) or a.precisions != "bf16":
            p.error("matrix mode requires BF16 and no single-shape m/n/k overrides")
        try:
            matrix_data = a.matrix_json.read_bytes()
            shapes = matrix_shapes(json.loads(matrix_data))
        except (OSError, ValueError) as error:
            p.error(str(error))
    else:
        a.m, a.n, a.k = (value if value is not None else default
                         for value, default in zip((a.m, a.n, a.k), (512, 4096, 2048)))
        if min(a.m, a.n, a.k) < 1:
            p.error("positive dimensions required")
        shapes = [{"shape": dict(m=a.m, n=a.n, k=a.k), "aliases": ["single"]}]
    if a.cutlass_counters and len(shapes) != 1:
        p.error("--cutlass-counters requires exactly one distinct M/N/K geometry")
    if a.cutlass_full_check:
        for geometry in shapes:
            m, n, k = (geometry['shape'][key] for key in ('m', 'n', 'k'))
            if max(m * n, m * k, n * k) > 4_194_304:
                p.error("--cutlass-full-check requires each buffer at most 4194304 elements")
    prop = torch.cuda.get_device_properties(0)
    if (prop.major, prop.minor) != (10, 3):
        p.error("requires Runtime compute 10.3")
    if a.cublaslt_sm_target is not None and a.cublaslt_sm_target > prop.multi_processor_count:
        p.error("cuBLASLt SM target exceeds the Runtime SM count")
    lib = Library(a.library)
    cutlass_library = None
    if a.compare_cutlass_library:
        from cutlass import Library as CutlassLibrary
        cutlass_library = CutlassLibrary(a.compare_cutlass_library)
    result = {"schema": "sm103_gemm_matrix_v1" if a.matrix_json else "sm103_gemm_v1",
        "operand_layout": a.operand_layout, "transpose_materialized": False,
        "state": "running", "measurement_protocol": PROTOCOL,
        "torch": torch.__version__, "cuda": torch.version.cuda, "compute": "10.3",
        "sms": prop.multi_processor_count, "device": prop.name,
        "library": str(lib.path), "library_sha256": hashlib.sha256(lib.path.read_bytes()).hexdigest(),
        "output_dtype": "bf16", "measurement": "single_gpu_pure_cublaslt",
        "measured_ranks": 1, "distributed_boundary_measured": False, "cublas_classic_measured": False,
        "quantization_timing": "excluded; prequantized operands; NVFP4 tensor scale=1"}
    if a.cublaslt_sm_target is not None:
        # Reduced-budget diagnostics must not overwrite unrestricted table rows.
        result.update(cublaslt_sm_target=a.cublaslt_sm_target,
                      comparison_scope="sm_target_diagnostic" if a.cublaslt_sm_target else "full_device",
                      hard_sm_partition_verified=False)
    if cutlass_library:
        result.update(schema="sm103_gemm_comparison_v1", measurement="single_gpu_cutlass_cublaslt_comparison",
                      cutlass_library=str(cutlass_library.path),
                      cutlass_library_sha256=hashlib.sha256(cutlass_library.path.read_bytes()).hexdigest(),
                      quantization_timing="not_applicable_bf16_only")
    if a.cutlass_counters:
        result.update(schema="sm103_cutlass_counters_v1", diagnostic_only=True,
                      measurement="single_gpu_cutlass_1sm_counters",
                      measurement_protocol="single_gpu_nvtx_counter_only_v1",
                      frontend_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      nvtx_range=COUNTER_RANGE, formal_sampling_performed=False,
                      ncu_metrics_verified=False)
    if a.cublaslt_counters:
        result.update(diagnostic_only=True, performance_accepted=False,
                      measurement="single_gpu_cublaslt_counters",
                      nvtx_range="fuse_cublaslt_counters", ncu_metrics_verified=False)
    if a.matrix_json:
        result.update(matrix_sha256=hashlib.sha256(matrix_data).hexdigest(),
                      logical_shapes=sum(len(item["aliases"]) for item in shapes),
                      unique_geometries=len(shapes), geometries=[])
    else:
        result.update(shape=shapes[0]["shape"], results=[])
    save_result(a.output, result, initial=True)
    a.sm_budget_context = None
    try:
        if a.gemm_sm_budget is not None:
            from cublaslt import SmBudget
            a.sm_budget_context = SmBudget(lib, a.gemm_sm_budget)
            result.update(sm_budget=a.sm_budget_context.info, comparison_scope='sm_budget_diagnostic')
        for index, shape in enumerate(shapes, 1):
            geometry = dict(shape, results=[], state="running")
            if a.cutlass_counters:
                del geometry["results"]
            if a.matrix_json:
                result["geometries"].append(geometry)
            else:
                geometry["results"] = result["results"]
            try:
                if a.cutlass_counters:
                    run_cutlass_counters(a, lib, geometry, cutlass_library)
                elif cutlass_library:
                    run_cutlass_comparison(a, lib, geometry, index, len(shapes), cutlass_library)
                else:
                    with (torch.cuda.stream(a.sm_budget_context.stream) if a.sm_budget_context
                          else contextlib.nullcontext()):
                        run_geometry(a, lib, geometry, index, len(shapes))
            except Exception:
                geometry["state"] = "failed"
                if not a.matrix_json:
                    result.update(inputs=geometry.get("inputs"), failed_measurement=geometry.get("failed_measurement"))
                raise
            geometry["state"] = "succeeded"
            if not a.matrix_json:
                result["inputs"] = geometry["inputs"]
            save_result(a.output, result)
        result["state"] = "succeeded"
    except Exception as error:
        result.update(state="failed", error=str(error))
        raise
    finally:
        try:
            if a.sm_budget_context:
                a.sm_budget_context.close()
        finally:
            save_result(a.output, result)


if __name__ == "__main__":
    main()
