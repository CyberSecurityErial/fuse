"""Synthetic offline model contract tests. No cloud/GPU or real-data fitting."""

import copy
import ast
import csv
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fused_model", ROOT / "benchmarks/sm103/fused_model.py")
model = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = model
spec.loader.exec_module(model)


def row(direction="GEMM_A2A", hidden=2048, q_heads=16, sequence=131072, world=8, tile=128, comm=32):
    m = sequence // world
    n, k = ((q_heads + 16) * 128, hidden) if direction == "GEMM_A2A" else (hidden, q_heads * 128)
    tiles = model.ceil_div(m, 128) * model.ceil_div(n, tile)
    compute = min(tiles, 148 - comm)
    payload = 2 * m * (n if direction == "GEMM_A2A" else k)
    return dict(candidate=1, direction=direction, component="fused", measurement_role="production",
        performance_accepted=True, world=world, global_seq=sequence, seq_local=m, hidden=hidden,
        q_heads=q_heads, kv_heads=8, head_dim=128, m=m, n=n, k=k,
        tile_policy=f"m128n{tile}", tile_m=128, tile_n=tile, tile_k=64, comm_ctas=comm,
        cluster_ctas=1, swizzle=1, raster="along_m" if direction == "GEMM_A2A" else "along_n",
        layout=model.SCOPE["layout"][direction], launch="eager", precision=model.SCOPE["precision"],
        collector="per_epoch_rank_events_v3_eventsync", host_launch="per_gpu_thread",
        sm_count=148, sm_counts=[148] * world, work_tiles_derived=tiles,
        production_compute_ctas_derived=[compute] * world,
        production_waves_derived=[model.ceil_div(tiles, compute)] * world,
        problem_gemm_flops=2*m*n*k, problem_route_payload_bytes=payload,
        problem_remote_payload_bytes=payload*(world-1)//world,
        executed_gemm_flops=2*m*n*k, executed_route_payload_bytes=payload,
        production_resources=[dict(rank=str(rank), tile_m="128", tile_n=str(tile), tile_k="64",
                                   threads="256", dynamic_smem="222208") for rank in range(world)],
        timing=dict(p50_ms=1.0, p95_ms=1.1, half_drift=.01, collector="per_epoch_rank_events_v3_eventsync"))


def observation(value, time=None):
    features = model.physical_features(value)
    if time is None:
        # Synthetic target is a predetermined function of physical features,
        # not recorded GPU performance or a tile/comm winner table.
        time = math.exp(-4 + .10*math.log1p(features[0]) + .02*math.log1p(features[1]) +
                        .04*math.log1p(features[2]) + .01*math.log1p(features[3]) +
                        .12*math.log1p(features[4]) + .03*math.log1p(features[5]))
    return model.Observation(value, time, time*1.1, "synthetic", {"fused": time, "compute_reference": time*1.2,
                                                              "copy_reference": time*.4})


def calibration():
    rows = []
    for direction in model.DIRECTIONS:
        for hidden, heads, _, _ in model.CALIBRATION_FAMILIES:
            for sequence in model.CALIBRATION_SEQUENCES:
                for world in (4, 8):
                    for tile in model.TILES[direction]:
                        for comm in model.CALIBRATION_COMMS:
                            value = row(direction, hidden, heads, sequence, world, tile, comm)
                            value["candidate"] = len(rows) + 1
                            rows.append(observation(value))
    return rows


def summary_fixture(values=None):
    values = values or [row()]
    grouped = {}
    for value in values:
        shape = tuple(value[k] for k in ("world", "global_seq", "seq_local", "hidden", "q_heads", "kv_heads", "head_dim"))
        grouped.setdefault(shape, []).append(copy.deepcopy(value))
    runs = []
    for index, (shape, candidates) in enumerate(grouped.items()):
        geometry = dict(zip(("world", "global_seq", "seq_local", "hidden", "q_heads", "kv_heads", "head_dim"), shape))
        config = {k: str(v) for k, v in geometry.items()}
        config.update(profile="0", validation_self_test="0", causal="1", warmup="10", samples="50",
                      input_generator="gpu_philox", host_launch="per_gpu_thread")
        components = []
        for candidate in candidates:
            for component in ("fused", "compute_reference", "copy_reference"):
                value = copy.deepcopy(candidate)
                value["component"] = component
                value["measurement_role"] = "production" if component == "fused" else "calibration"
                if component == "compute_reference":
                    value["timing"]["p50_ms"] = 1.2
                    value["timing"]["p95_ms"] = 1.3
                    value["executed_route_payload_bytes"] = 0
                if component == "copy_reference":
                    value["timing"]["p50_ms"] = .4
                    value["timing"]["p95_ms"] = .5
                    value["executed_gemm_flops"] = 0
                components.append(value)
        runs.append(dict(run_id=f"synthetic-{index}", node="0a", source_id="a"*64,
            environment_fingerprint="b"*64, config=config, geometry=geometry, diagnostic_only=False,
            devices=[dict(rank=str(rank), runtime_cc="10.3", sms="148") for rank in range(shape[0])],
            build=dict(profile=False, binary_sha256="c"*64, build_inputs="d"*64, environment_fingerprint="b"*64),
            candidates=components))
    return dict(schema="sm103_fused_verified_v1", model_fitted=False, globally_optimal=False,
                diagnostic_rows=0, performance_rows=sum(len(run["candidates"]) for run in runs), runs=runs)


def write_fixture(directory, report):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps(report))
    with (directory / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=model.CSV_COLUMNS)
        writer.writeheader()
        for run in report["runs"]:
            for candidate in run["candidates"]:
                values = run | candidate | candidate["timing"] | {"binary_sha256": run["build"]["binary_sha256"]}
                writer.writerow({key: values[key] for key in model.CSV_COLUMNS})


class PhysicalFeaturesTests(unittest.TestCase):
    def test_qkv_tasks_bytes_and_n64_spans_are_exact(self):
        value = row(tile=64)
        f = model.physical_features(value)
        self.assertEqual(f[2], model.ceil_div(256*32, 8*32))
        self.assertEqual(f[3], 2*16384*4096*7//8)
        self.assertEqual(f[5], 2)
        for tile in model.TILES["GEMM_A2A"]:
            value = row(tile=tile)
            expected = sum((start+127)//tile - start//tile + 1 for start in range(0, 4096, 128))/32
            self.assertEqual(model.physical_features(value)[5], expected)

    def test_oproj_actual_windows_and_48k_chunk_rounding(self):
        narrow = model.physical_features(row("A2A_GEMM", tile=128))
        wide = model.physical_features(row("A2A_GEMM", tile=256))
        self.assertEqual((narrow[5], wide[5]), (1/8, 1/15))
        self.assertEqual(narrow[2], model.ceil_div(128*8*2, 4*32))
        large = row("A2A_GEMM", hidden=16384, q_heads=128, world=4)
        # 8192-byte peer rows: 6 rows/chunk and ceil(128/6)=22 arrivals.
        self.assertEqual(model.physical_features(large)[2], model.ceil_div(256*4*22, 4*32))

    def test_features_never_use_model_labels_times_or_tile_categories(self):
        value = row()
        expected = model.physical_features(value)
        value.update(model="winner-forbidden", timing={"p50_ms": float("nan")},
                     compute_reference=99999, copy_reference=-123, winner=True)
        self.assertEqual(model.physical_features(value), expected)

    def test_explicit_unsupported_scope_and_false_geometry(self):
        changes = (dict(head_dim=256), dict(sm_count=132), dict(tile_n=320), dict(tile_m=64),
                   dict(cluster_ctas=2), dict(swizzle=2), dict(layout="unknown"), dict(launch="graph"),
                   dict(precision="fp8"), dict(world=2), dict(m=257), dict(n=1), dict(comm_ctas=148),
                   dict(m=True), dict(global_seq=1024), dict(batch=2), dict(defer_v_a2a=True),
                   dict(qkv_peer_interleaved=True), dict(cyclic_peer_order=True), dict(channel_count=2))
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                model.physical_features(row() | change)

    def test_family_and_alias_grouping(self):
        a = row("A2A_GEMM", hidden=4096, q_heads=32)
        b = a | dict(kv_heads=16, model="representative_small")
        self.assertEqual(model.problem_key(a), model.problem_key(b))
        self.assertEqual(model.family_key(a), model.family_key(b))
        same_m = row("A2A_GEMM", hidden=4096, q_heads=32, sequence=262144, world=8)
        other_cp = row("A2A_GEMM", hidden=4096, q_heads=32, sequence=131072, world=4)
        self.assertEqual(same_m["m"], other_cp["m"])
        self.assertNotEqual(model.problem_key(same_m), model.problem_key(other_cp))
        self.assertEqual(model.family_key(same_m), model.family_key(other_cp))


class SummaryInputTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="fused-model-test-")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)

    def load(self, report):
        write_fixture(self.path, report)
        return model.read_summary(self.path)

    def test_paired_input_hashes_resources_and_negative_reference_residual(self):
        data = self.load(summary_fixture())
        self.assertEqual(len(data.observations), 1)
        self.assertEqual(data.provenance[0]["summary_json_sha256"], model.digest((self.path/"summary.json").read_bytes()))
        observation = data.observations[0]
        report = model.evaluate([observation], [observation.p50_ms])
        self.assertAlmostEqual(report["groups"][0]["candidates"][0]["f_minus_max_cr_ms"], -.2)

    def test_csv_corruption_and_missing_column_are_rejected(self):
        write_fixture(self.path, summary_fixture())
        path = self.path / "summary.csv"
        original = path.read_text()
        path.write_text(original.replace("1.0,1.1,0.01", "2.0,1.1,0.01"))
        with self.assertRaisesRegex(ValueError, "JSON/CSV"):
            model.read_summary(self.path)
        path.write_text(original.replace("p50_ms", "bogus"))
        with self.assertRaisesRegex(ValueError, "CSV schema"):
            model.read_summary(self.path)

    def test_mixed_boundary_and_bad_resources_are_rejected(self):
        for field, value in (("host_launch", "sequential"), ("source_id", "e"*64),
                             ("binary_sha256", "e"*64), ("environment_fingerprint", "e"*64)):
            report = summary_fixture([row(), row(sequence=262144)])
            second = report["runs"][1]
            if field == "host_launch":
                second["config"][field] = value
                for candidate in second["candidates"]:
                    candidate[field] = value
            elif field == "binary_sha256":
                second["build"][field] = value
            else:
                second[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.load(report)
        for change in (dict(threads="384"), dict(tile_n="64"), dict(dynamic_smem="0")):
            report = summary_fixture()
            report["runs"][0]["candidates"][0]["production_resources"][0].update(change)
            with self.assertRaises(ValueError):
                self.load(report)

    def test_duplicate_geometry_alias_is_rejected(self):
        a = row("A2A_GEMM", hidden=4096, q_heads=32)
        b = copy.deepcopy(a)
        b["candidate"], b["kv_heads"] = 2, 16
        with self.assertRaisesRegex(ValueError, "Duplicate physical"):
            self.load(summary_fixture([a, b]))

    def test_diagnostic_incomplete_and_invalid_time_are_rejected(self):
        for field, value in (("performance_accepted", False), ("timing", dict(p50_ms=float("nan"), p95_ms=1.1,
                                                                            half_drift=.01, collector="bad"))):
            report = summary_fixture()
            report["runs"][0]["candidates"][0][field] = value
            with self.assertRaises(ValueError):
                self.load(report)
        report = summary_fixture()
        report["diagnostic_rows"] = 1
        with self.assertRaises(ValueError):
            self.load(report)

    def test_references_must_match_fused_geometry_and_measurement_boundary(self):
        for field, value in (("collector", "unknown_reference_collector"), ("comm_ctas", 24),
                             ("n", 8192), ("host_launch", "sequential")):
            report = summary_fixture()
            reference = report["runs"][0]["candidates"][1]
            reference[field] = value
            if field == "collector":
                reference["timing"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Reference geometry"):
                self.load(report)

    def test_fused_only_external_summary_is_allowed(self):
        report = summary_fixture()
        report["runs"][0]["candidates"] = report["runs"][0]["candidates"][:1]
        report["performance_rows"] = 1
        data = self.load(report)
        self.assertEqual(set(data.observations[0].reference_ms), {"fused"})


class RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = calibration()

    def test_registration_and_grid_fail_closed(self):
        model.calibration_contract(self.rows)
        for rows in (self.rows[:-1], self.rows + [self.rows[0]]):
            with self.assertRaises(ValueError):
                model.calibration_contract(rows)

    def test_ridge_constant_and_collinear_design(self):
        observations = [observation(row(comm=c), 2.0) for c in model.CALIBRATION_COMMS]
        fitted = model.fit_ridge(observations, .1)
        for observation_ in observations:
            self.assertAlmostEqual(model.predict(fitted, observation_.row), 2.0, places=10)
        self.assertEqual(len(fitted["coefficients"]), 7)
        self.assertTrue(all(math.isfinite(x) for x in fitted["coefficients"]))

    def test_synthetic_known_log_relation_is_recovered(self):
        observations = [o for o in self.rows if o.row["direction"] == "GEMM_A2A"]
        fitted = model.fit_ridge(observations, 1e-10)
        for observation_ in observations:
            self.assertAlmostEqual(model.predict(fitted, observation_.row), observation_.p50_ms, places=6)

    def test_fold_standardization_and_fit_never_see_outer_family(self):
        observations = [o for o in self.rows if o.row["direction"] == "A2A_GEMM"]
        original_fit = model.fit_ridge
        calls = []
        def record_fit(rows, regularization):
            calls.append({model.family_key(o.row) for o in rows})
            return original_fit(rows, regularization)
        with mock.patch.object(model, "fit_ridge", side_effect=record_fit):
            result = model.nested_lofo(observations)
        for index, fold in enumerate(result["folds"]):
            # Four lambdas x two inner-held families, plus one outer fit.
            outer_calls = calls[index*9:(index+1)*9]
            self.assertEqual(len(outer_calls), 9)
            for families in outer_calls:
                self.assertNotIn(tuple(fold["held_family"]), families)
        self.assertEqual(len(result["groups"]), 12)
        self.assertIn("absolute", result["relative_prediction_error_fraction"])

    def test_no_measured_reference_values_in_predictions(self):
        observations = [o for o in self.rows if o.row["direction"] == "A2A_GEMM"]
        fitted = model.fit_ridge(observations, 1.0)
        before = [model.predict(fitted, o.row) for o in observations]
        altered = copy.deepcopy(observations)
        for o in altered:
            o.reference_ms = {"compute_reference": 1e20, "copy_reference": -1}
        after = [model.predict(model.fit_ridge(altered, 1.0), o.row) for o in altered]
        self.assertEqual(before, after)

    def test_regret_and_signed_prediction_error(self):
        observations = [observation(row(comm=8), 1), observation(row(comm=16), 2)]
        result = model.evaluate(observations, [3, .5])
        self.assertEqual(result["regret_fraction"]["max"], 1.0)
        self.assertEqual(result["groups"][0]["selected_comm_ctas"], 16)
        self.assertAlmostEqual(result["relative_prediction_error_fraction"]["signed_median"], .625)

    def test_external_alias_and_boundary_overlap_rejected_before_fit(self):
        train = model.Dataset(self.rows, {"source": "fixed"}, {}, [])
        external = model.Dataset([self.rows[0]], train.contract, {}, [])
        with mock.patch.object(model, "fit_ridge", side_effect=AssertionError("must not fit")):
            with self.assertRaisesRegex(ValueError, "overlaps"):
                model.experiment(train, [external])
            external.contract = {"source": "changed"}
            with self.assertRaisesRegex(ValueError, "different execution"):
                model.experiment(train, [external])

    def test_synthetic_complete_experiment_separates_external_sequence_test(self):
        train = model.Dataset(self.rows, {"source": "fixed"}, {}, [])
        external = []
        for direction in model.DIRECTIONS:
            for tile in model.TILES[direction]:
                for comm in model.CALIBRATION_COMMS:
                    value = row(direction, sequence=65536, tile=tile, comm=comm)
                    value["candidate"] = len(external) + 1
                    external.append(observation(value))
        heldout = model.Dataset(external, train.contract, {}, [])
        report = model.experiment(train, [heldout])
        self.assertFalse(report["runtime_selector_installed"])
        for direction in model.DIRECTIONS:
            result = report["directions"][direction]
            self.assertEqual(len(result["model"]["coefficients"]), 7)
            self.assertEqual(len(result["training_family_cv"]["groups"]), 12)
            self.assertEqual(set(result["external_heldout"]["generalization_strata"]), {"seen_family/new_sequence"})
        json.dumps(report, allow_nan=False)

    def test_inspect_never_fits_and_fit_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="fused-model-cli-test-") as directory:
            path = Path(directory)
            write_fixture(path, summary_fixture())
            with mock.patch.object(model, "fit_ridge", side_effect=AssertionError("inspect cannot fit")), \
                    mock.patch("sys.stdout", new_callable=io.StringIO) as output:
                model.main(["inspect", "--summary", str(path)])
                self.assertFalse(json.loads(output.getvalue())["fitted"])
                occupied = path / "report.json"
                occupied.write_text("user-owned")
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    model.main(["fit", "--summary", str(path), "--output", str(occupied)])
                self.assertEqual(occupied.read_text(), "user-owned")


class NonnegativeTimeTests(unittest.TestCase):
    def test_boundary_solution_satisfies_full_kkt(self):
        # Unconstrained solution is (-1,2); constrained optimum is (0,1.5).
        coefficients, violation = model.solve_nonnegative([[2., 1.], [1., 2.]], [0., 3.])
        self.assertEqual(coefficients, [0., 1.5])
        self.assertLess(violation, 1e-12)
        zeros, _ = model.solve_nonnegative([[1., 0.], [0., 1.]], [-1., -2.])
        self.assertEqual(zeros, [0., 0.])
        with self.assertRaises(ValueError):
            model.solve_nonnegative([[1.] * 8 for _ in range(8)], [1.] * 8)

    def test_relative_error_weighting_and_unpenalized_nonnegative_intercept(self):
        observations = [observation(row(), 1), observation(row(), 10)]
        fitted = model.fit_nonnegative_ridge(observations, 1.0)
        # Equal-group relative least squares constant: sum(1/y)/sum(1/y^2).
        expected = (1 + 1/10) / (1 + 1/100)
        self.assertAlmostEqual(model.predict(fitted, row()), expected, places=10)
        self.assertEqual(fitted["coefficients"][1:], [0.] * 6)
        self.assertGreater(fitted["coefficients"][0], 0)
        self.assertNotIn("means", fitted)
        for raw, scale in zip(model.physical_features(row()), fitted["scales"]):
            self.assertAlmostEqual(raw, scale)

    def test_rms_scaling_is_time_unit_invariant(self):
        observations = calibration()[:60]
        original = model.fit_nonnegative_ridge(observations, .1)
        converted = copy.deepcopy(observations)
        for value in converted:
            value.p50_ms *= 1000
        changed = model.fit_nonnegative_ridge(converted, .1)
        for left, right in zip(original["coefficients"], changed["coefficients"]):
            self.assertAlmostEqual(left, right, places=9)
        for value in observations:
            self.assertAlmostEqual(model.predict(changed, value.row)/1000,
                                   model.predict(original, value.row), places=9)

    def test_known_positive_raw_linear_cost_is_recovered(self):
        observations = calibration()
        observations = [value for value in observations if value.row["direction"] == "GEMM_A2A"]
        weights = (1e-12, 1e-9, .0003, 3e-10, .05, .02)
        for value in observations:
            value.p50_ms = .02 + sum(weight*feature for weight, feature in
                                     zip(weights, model.physical_features(value.row)))
        fitted = model.fit_nonnegative_ridge(observations, 1e-10)
        self.assertTrue(all(value >= 0 for value in fitted["coefficients"]))
        for value in observations:
            self.assertAlmostEqual(model.predict(fitted, value.row)/value.p50_ms, 1, places=6)

    def test_no_reference_floor_or_reference_input(self):
        observations = [observation(row(comm=comm), 1.0) for comm in model.CALIBRATION_COMMS]
        fitted = model.fit_nonnegative_ridge(observations, .01)
        for value in observations:
            self.assertLess(model.predict(fitted, value.row), value.reference_ms["compute_reference"])
        altered = copy.deepcopy(observations)
        for value in altered:
            value.reference_ms = {"compute_reference": 1e10, "copy_reference": 1e-10}
        changed = model.fit_nonnegative_ridge(altered, .01)
        self.assertEqual(changed, fitted)
        with self.assertRaisesRegex(ValueError, "Unknown pre-defined"):
            model.predict(fitted | {"form": "unknown"}, observations[0].row)

    def test_nested_nonnegative_training_never_sees_outer_family(self):
        observations = [value for value in calibration() if value.row["direction"] == "A2A_GEMM"]
        original = model.fit_nonnegative_ridge
        calls = []
        def fit(rows, regularization):
            calls.append({model.family_key(value.row) for value in rows})
            return original(rows, regularization)
        with mock.patch.object(model, "fit_nonnegative_ridge", side_effect=fit):
            report = model.nested_lofo(observations, "nonnegative_time_ridge")
        for index, fold in enumerate(report["folds"]):
            for families in calls[index*9:(index+1)*9]:
                self.assertNotIn(tuple(fold["held_family"]), families)
            self.assertEqual({score["regularization"] for score in fold["inner_selection"]},
                             set(model.REGULARIZATIONS))
            self.assertTrue(all("mean_group_relative_squared_error" in score for score in fold["inner_selection"]))


def primitive_contract():
    return dict(source_id="a"*64, binary_sha256="b"*64, build_inputs="c"*64,
                environment_fingerprint="d"*64, node="0a", runtime_cc="10.3", sm_count=148,
                precision="bf16_accfp32_bf16", launch="eager", host_launch="per_gpu_thread",
                collector="per_epoch_rank_events_v3_eventsync", input_generator="gpu_philox")


def primitive_fixture(direction="GEMM_A2A", tile=128, launch_us=3.0):
    # Explicit synthetic service data, never claimed to be B300 measurements.
    signature = model.CollectiveSignature(128, tile, 64, 128, 32, 6 if tile == 128 else 4,
                                          222208 if tile == 128 else 230400)
    value = dict(schema="sm103_primitive_calibration_v1", signature=vars(signature),
                 k_anchors=[[2048, 4.0], [16384, 18.0]], launch_us=launch_us,
                 compute_ctas_range=[16, 148], m_range=[128, 131072], n_range=[128, 32768],
                 execution_contract=primitive_contract(), evidence_sha256="e"*64,
                 calibration_problems=[[direction, 128, 4096, k, 8, model.SCOPE["layout"][direction]]
                                       for k in (2048, 16384)])
    return value


def primitive_row(direction="GEMM_A2A", tile=128, **changes):
    value = row(direction, tile=tile) | changes
    signature = model.PrimitiveCalibration.from_dict(primitive_fixture(direction, tile)).signature
    for resource in value["production_resources"]:
        resource["dynamic_smem"] = str(signature.dynamic_smem_bytes)
    return value


class PrimitiveComputeTests(unittest.TestCase):
    def estimate(self, value=None, calibration=None, **options):
        value = value or primitive_row()
        calibration = calibration or model.PrimitiveCalibration.from_dict(primitive_fixture())
        return model.estimate_compute_roof(value, calibration, signature=calibration.signature,
                                          execution_contract=primitive_contract(), **options)

    def test_continuous_waves_and_k_interpolation_have_explicit_units(self):
        result = self.estimate(primitive_row(k=4096))
        self.assertEqual(result["tile_service_us"], 6.0)
        self.assertEqual(result["work_tiles"], 4096)
        self.assertEqual(result["compute_ctas"], 116)
        self.assertAlmostEqual(result["wave_equivalents"], 4096/116)
        self.assertAlmostEqual(result["compute_us"], 3 + 6*4096/116)
        self.assertEqual(result["tile_flops"], 2*128*128*4096)
        self.assertEqual(result["tile_bytes_proxy"], 2*4096*256 + 2*128*128)
        self.assertFalse(result["integer_wave_correction_applied"])
        self.assertFalse(result["compute_roof_is_lower_bound"])
        doubled = self.estimate(primitive_row(k=4096, m=32768))
        self.assertAlmostEqual(doubled["steady_compute_us"], 2*result["steady_compute_us"])

    def test_both_real_tile_geometries_and_unknown_launch(self):
        for direction in model.DIRECTIONS:
            for tile in (128, 256):
                calibration = model.PrimitiveCalibration.from_dict(primitive_fixture(direction, tile, None))
                result = self.estimate(primitive_row(direction, tile), calibration)
                self.assertGreater(result["steady_compute_us"], 0)
                self.assertIsNone(result["compute_us"])
                self.assertIsNone(result["launch_us"])

    def test_names_and_measured_reference_times_do_not_affect_estimates(self):
        expected = self.estimate()
        changed = primitive_row(model="never-dispatch-on-this", tile_policy="equivalent-alias",
                                timing={"p50_ms": float("nan")}, compute_reference=-1, copy_reference=1e30)
        self.assertEqual(expected, self.estimate(changed))

    def test_anchors_are_explicit_immutable_and_not_extrapolated(self):
        fixture = primitive_fixture()
        calibration = model.PrimitiveCalibration.from_dict(fixture)
        fixture["k_anchors"][0][1] = 99
        fixture["execution_contract"]["source_id"] = "f"*64
        self.assertEqual(self.estimate(calibration=calibration)["tile_service_us"], 4)
        with self.assertRaises(TypeError):
            calibration.execution_contract["source_id"] = "f"*64
        for k in (1024, 16448):
            with self.subTest(k=k), self.assertRaisesRegex(ValueError, "extrapolation"):
                self.estimate(primitive_row(k=k))
        for anchors in ([[2048, 4]], [[4096, 5], [2048, 4]], [[2048, 4], [2048, 5]],
                        [[2048, float("nan")], [16384, 5]], [[2048, -1], [16384, 5]]):
            with self.subTest(anchors=anchors), self.assertRaises(ValueError):
                model.PrimitiveCalibration.from_dict(primitive_fixture() | {"k_anchors": anchors})

    def test_actual_resources_and_execution_contract_fail_closed(self):
        calibration = model.PrimitiveCalibration.from_dict(primitive_fixture())
        wrong = model.PrimitiveCalibration.from_dict(primitive_fixture(tile=256)).signature
        with self.assertRaisesRegex(ValueError, "collective differs"):
            model.estimate_compute_roof(primitive_row(), calibration, signature=wrong,
                                       execution_contract=primitive_contract())
        for field, value in (("source_id", "f"*64), ("node", "09"), ("host_launch", "sequential")):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "contract mismatch"):
                model.estimate_compute_roof(primitive_row(), calibration, signature=calibration.signature,
                                           execution_contract=primitive_contract() | {field: value})
        changed = primitive_row()
        changed["production_resources"][0]["dynamic_smem"] = "230400"
        with self.assertRaisesRegex(ValueError, "Recorded resources"):
            self.estimate(changed)
        for change in (dict(ab_stages=3, tile_n=256), dict(tile_k=128), dict(element_c="bf16"),
                       dict(epilogue_n=64), dict(threads=384), dict(tile_n=True)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                model.CollectiveSignature(**(primitive_fixture()["signature"] | change))

    def test_prediction_and_heldout_domains_are_physical(self):
        for changes in (dict(m=262144), dict(n=65536), dict(comm_ctas=145), dict(world=4),
                        dict(layout="unknown"), dict(k=2056), dict(tile_k=128), dict(defer_v_a2a=True),
                        dict(collector="mpi_rank_events_v1"), dict(direction="A2A_GEMM")):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.estimate(primitive_row(**changes))
        result = self.estimate(heldout=True)
        self.assertTrue(result["heldout_workload"])
        self.assertFalse(result["externally_validated"])
        # A new label, policy spelling or SM split is not a held-out workload.
        for comm in (32, 64):
            with self.assertRaisesRegex(ValueError, "overlaps calibration"):
                self.estimate(primitive_row(m=128, comm_ctas=comm, model="new-alias"), heldout=True)

    def test_json_calibration_roundtrip_and_mpi_contract_are_explicit(self):
        fixture = json.loads(json.dumps(primitive_fixture()))
        fixture["execution_contract"].update(host_launch="mpi_process", collector="mpi_rank_events_v1")
        calibration = model.PrimitiveCalibration.from_dict(fixture)
        value = primitive_row(host_launch="mpi_process", collector="mpi_rank_events_v1")
        result = model.estimate_compute_roof(value, calibration, signature=calibration.signature,
                                             execution_contract=fixture["execution_contract"])
        json.dumps(result, allow_nan=False)
        self.assertEqual(result["calibration_evidence_sha256"], "e"*64)
        with self.assertRaises(ValueError):
            model.PrimitiveCalibration.from_dict(fixture | {"schema": "unknown"})


def primitive_summary_fixture(values=None, *, calibrate=True, maximum=1, raster="heuristic", node="0a"):
    """Current summary field names, with explicitly synthetic MPI timings."""
    values = copy.deepcopy(values or [row()])
    for index, value in enumerate(values, 1):
        value["candidate"] = index
        if value["tile_n"] == 256:
            value["tile_policy"] = "m128n256k64e32"
    report = summary_fixture(values)
    for run in report["runs"]:
        run["node"] = node
        run["build"]["mpi"] = True
        run["config"].update(process_layout="mpi_one_process_per_gpu", host_launch="mpi_process",
                             launch="eager", calibrate=str(int(calibrate)))
        requested = {direction: raster for direction in model.DIRECTIONS}
        effective = {direction: ("along_m" if direction == "GEMM_A2A" else "along_n")
                     if raster == "heuristic" else raster for direction in model.DIRECTIONS}
        run["scheduling"] = dict(schema="explicit_v1", max_swizzle_size=maximum,
                                 requested_rasters=requested, effective_rasters=effective)
        if not calibrate:
            run["candidates"] = [c for c in run["candidates"] if c["component"] == "fused"]
        for value in run["candidates"]:
            value.update(host_launch="mpi_process", collector="mpi_rank_events_v1", graph_epoch_mode=None)
            value["timing"]["collector"] = "mpi_rank_events_v1"
            m_tiles, n_tiles = model.ceil_div(value["m"], 128), model.ceil_div(value["n"], value["tile_n"])
            minimum = min(m_tiles, n_tiles)
            swizzle = (8 if maximum >= 8 and minimum >= 6 else 4 if maximum >= 4 and minimum >= 3 else
                       2 if maximum >= 2 and minimum >= 2 else 1)
            pm, pn = model.ceil_div(m_tiles, swizzle)*swizzle, model.ceil_div(n_tiles, swizzle)*swizzle
            value.update(schedule_schema="explicit_v1", raster_requested=raster, raster=effective[value["direction"]],
                         max_swizzle_size=maximum, effective_swizzle_size=swizzle, swizzle=swizzle,
                         padded_m_tiles=pm, padded_n_tiles=pn, has_padding=(pm != m_tiles or pn != n_tiles),
                         scheduled_work_tiles_derived=pm*pn)
            compute = min(pm*pn, 148-value["comm_ctas"])
            value["production_compute_ctas_derived"] = [compute]*value["world"]
            value["compute_ctas_derived"] = [0 if value["component"] == "copy_reference" else compute]*value["world"]
            value["production_waves_derived"] = [model.ceil_div(pm*pn, compute)]*value["world"]
            for resource in value["production_resources"]:
                resource["dynamic_smem"] = "214016"
    report["performance_rows"] = sum(len(r["candidates"]) for r in report["runs"])
    return report


def write_primitive_fixture(directory, report):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps(report))
    fields = list(dict.fromkeys(model.CSV_COLUMNS + list(model.PRIMITIVE_SCHEDULE_FIELDS) + ["graph_epoch_mode"]))
    with (directory / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in report["runs"]:
            for candidate in run["candidates"]:
                flat = run | candidate | candidate["timing"] | {"binary_sha256": run["build"]["binary_sha256"]}
                writer.writerow({key: flat[key] for key in fields})


class PrimitiveSummaryInputTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="fused-primitive-input-test-")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)

    def load(self, report):
        write_primitive_fixture(self.path, report)
        with mock.patch.object(model, "fit_model", side_effect=AssertionError("no fitting")), \
                mock.patch.object(model, "read_summary", side_effect=AssertionError("frozen reader")), \
                mock.patch.object(model, "physical_features", side_effect=AssertionError("frozen features")):
            return model.read_primitive_summary(self.path)

    def test_legacy_model_and_estimator_function_asts_remain_frozen(self):
        original = {
            'require', 'ceil_div', 'integer', 'positive', 'digest', 'hash_value', 'problem_key', 'family_key',
            'candidate_key', 'physical_features', 'Observation', 'Dataset', 'read_summary', 'calibration_contract',
            'solve', 'fit_ridge', 'solve_nonnegative', 'fit_nonnegative_ridge', 'fit_model', 'predict', 'percentile',
            'stats', 'evaluate', 'choose_regularization', 'nested_lofo', 'experiment', 'CollectiveSignature',
            'primitive_problem_key', 'primitive_execution_contract', 'nonnegative', 'PrimitiveCalibration',
            'estimate_compute_roof', 'score_pipeline', 'comparison_problem_key', 'comparison_pair', 'badcase_targets', 'main'}
        nodes = [node for node in ast.parse(Path(model.__file__).read_text()).body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in original]
        for root in nodes:
            for node in ast.walk(root):
                if getattr(node, "type_params", None) == []:
                    node._fields = tuple(field for field in node._fields if field != "type_params")
        self.assertEqual(len(nodes), len(original))
        self.assertEqual(model.digest('\n'.join(ast.dump(n, include_attributes=False) for n in nodes).encode()),
                         'd2985ca5f52eac1cf1efcd7b322adaf89fd18e4659bd915140435c17617aaf1d')

    def test_mpi_eager_fcr_are_observations_not_tc_service_or_fitted_anchors(self):
        data = self.load(primitive_summary_fixture([row(), row(tile=256)]))
        self.assertEqual(len(data.records), 6)
        self.assertEqual({r["component"] for r in data.records}, {"fused", "compute_reference", "copy_reference"})
        self.assertEqual(len(data.execution_contracts), 1)
        self.assertEqual(next(iter(data.execution_contracts.values()))["collector"], "mpi_rank_events_v1")
        for record in data.records:
            self.assertIsNone(record["tc_service_us"])
            self.assertFalse(record["calibration_eligible"])
            for field in ("ab_stages", "accumulator_stages", "actual_epilogue_tile", "element_c"):
                self.assertIsNone(record["resources"][field])
            if record["component"] != "fused":
                self.assertIsNone(record["resources"]["actual_component"])
                self.assertEqual(record["measurement_scope"], "independent_reference")
        n256 = next(r for r in data.records if r["row"]["tile_n"] == 256)
        self.assertEqual(n256["resources"]["requested_epilogue_n"], 32)
        self.assertEqual(n256["resources"]["production"]["dynamic_smem"], 214016)
        self.assertLess(data.records[0]["p50_ms"], data.records[1]["p50_ms"])  # Preserve F<C, no clipping.
        self.assertEqual(data.provenance["summary_json_sha256"], model.digest((self.path/"summary.json").read_bytes()))
        self.assertEqual(data.provenance["summary_csv_sha256"], model.digest((self.path/"summary.csv").read_bytes()))
        json.dumps(vars(data), allow_nan=False)

    def test_full_f_only_dataset_does_not_invent_references(self):
        values = [row(direction, sequence=sequence, tile=tile, comm=comm)
                  for direction in model.DIRECTIONS for sequence in (1024, 131072)
                  for tile in (128, 256) for comm in (8, 12, 16, 24, 32)]
        data = self.load(primitive_summary_fixture(values, calibrate=False))
        self.assertEqual(len(data.records), 40)
        self.assertEqual({r["component"] for r in data.records}, {"fused"})
        self.assertEqual(sum(r["long_sequence"] for r in data.records), 20)
        self.assertTrue(all(not r["calibration_eligible"] for r in data.records))

    def test_swizzle_raster_tile_and_budget_candidates_stay_distinct(self):
        reports = [primitive_summary_fixture([row(), row(tile=256), row(comm=8)], maximum=maximum, raster=raster)
                   for maximum, raster in ((1, "heuristic"), (4, "heuristic"), (4, "along_n"))]
        merged = reports[0]
        for index, report in enumerate(reports[1:], 1):
            report["runs"][0]["run_id"] = f"schedule-{index}"
            merged["runs"] += report["runs"]
            merged["performance_rows"] += report["performance_rows"]
        data = self.load(merged)
        fused = [r for r in data.records if r["component"] == "fused"]
        self.assertEqual(len({r["candidate_key"] for r in fused}), 9)
        self.assertEqual({r["schedule"]["effective_swizzle_size"] for r in fused}, {1, 4})

    def test_small_requested_swizzle_is_not_conflated_with_effective_or_padding(self):
        for sequence, expected, padding in ((512, 1, False), (2056, 4, True)):
            report = primitive_summary_fixture([row(sequence=sequence)], maximum=4)
            data = self.load(report)
            schedule = data.records[0]["schedule"]
            self.assertEqual(schedule["max_swizzle_size"], 4)
            self.assertEqual(schedule["effective_swizzle_size"], expected)
            self.assertEqual(schedule["has_padding"], padding)

    def test_source_and_node_are_explicit_partitions_not_merged_samples(self):
        report = primitive_summary_fixture()
        other = copy.deepcopy(report["runs"][0])
        other["run_id"] = "other-source"
        other["source_id"] = "f"*64
        report["runs"].append(other)
        third = copy.deepcopy(other)
        third.update(run_id="other-node", node="09")
        report["runs"].append(third)
        report["performance_rows"] = 9
        data = self.load(report)
        self.assertEqual(len(data.execution_contracts), 3)
        self.assertEqual(len(data.records), 9)
        self.assertEqual(len({r["candidate_key"] for r in data.records}), 1)

    def test_duplicate_physical_geometry_or_oproj_alias_is_rejected(self):
        a = row("A2A_GEMM")
        b = a | dict(model="another-label", kv_heads=16)
        report = primitive_summary_fixture([a, b])
        with self.assertRaisesRegex(ValueError, "Duplicate physical"):
            self.load(report)

    def test_graph_profiles_native_counters_and_nonmpi_are_rejected(self):
        for mutation in ("graph", "profile", "selftest", "nonmpi", "native", "counter", "fp8", "unaccepted"):
            report = primitive_summary_fixture()
            run = report["runs"][0]
            if mutation == "graph": run["config"]["launch"] = "graph"
            if mutation == "profile": run["build"]["profile"] = True
            if mutation == "selftest": run["config"]["validation_self_test"] = "1"
            if mutation == "nonmpi": run["build"]["mpi"] = False
            if mutation == "native": report["schema"] = "sm103_gemm_comparison_v1"
            if mutation == "counter": report["schema"] = "sm103_cutlass_counters_v1"
            if mutation == "fp8": run["candidates"][0]["precision"] = "fp8"
            if mutation == "unaccepted": run["candidates"][0]["performance_accepted"] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.load(report)

    def test_mismatched_fcr_scope_resources_schedule_and_compute_are_rejected(self):
        changes = [dict(component="unknown"), dict(tile_policy="m128n256k64e64"), dict(tile_policy="m128n256"),
                   dict(tile_k=128), dict(comm_ctas=12), dict(layout="unknown"), dict(raster="along_n"),
                   dict(effective_swizzle_size=4), dict(padded_m_tiles=132), dict(collector="unknown"),
                   dict(production_compute_ctas_derived=[140]*8), dict(compute_ctas_derived=[0]*8),
                   dict(executed_route_payload_bytes=128), dict(graph_epoch_mode="recapture_update_v1")]
        for change in changes:
            report = primitive_summary_fixture()
            report["runs"][0]["candidates"][1].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load(report)
        report = primitive_summary_fixture()
        report["runs"][0]["candidates"][1]["production_resources"][0]["dynamic_smem"] = "230400"
        with self.assertRaisesRegex(ValueError, "Heterogeneous"):
            self.load(report)

    def test_missing_reference_and_extra_csv_rows_cannot_be_accepted(self):
        report = primitive_summary_fixture()
        report["runs"][0]["candidates"].pop()
        report["performance_rows"] = 2
        with self.assertRaisesRegex(ValueError, "Missing or mismatched"):
            self.load(report)
        write_primitive_fixture(self.path, primitive_summary_fixture())
        path = self.path/"summary.csv"
        lines = path.read_text().splitlines()
        path.write_text('\n'.join(lines + [lines[1]]) + '\n')
        with self.assertRaisesRegex(ValueError, "Duplicate primitive CSV"):
            model.read_primitive_summary(self.path)

    def test_csv_schedule_corruption_and_missing_schema_fields_are_rejected(self):
        for mutation in ("schedule", "missing", "duplicate_header", "extra"):
            write_primitive_fixture(self.path, primitive_summary_fixture())
            path = self.path/"summary.csv"
            rows = list(csv.reader(io.StringIO(path.read_text())))
            if mutation == "schedule": rows[1][rows[0].index("effective_swizzle_size")] = "4"
            if mutation == "missing":
                index = rows[0].index("schedule_schema")
                for value in rows: value.pop(index)
            if mutation == "duplicate_header": rows[0][-1] = rows[0][0]
            if mutation == "extra":
                rows[0].append("invented_field")
                for value in rows[1:]: value.append("0")
            with path.open("w", newline="") as stream:
                csv.writer(stream).writerows(rows)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                model.read_primitive_summary(self.path)


class PhysicalPipelineTests(unittest.TestCase):
    def test_missing_feed_first_or_tail_never_establishes_overlap(self):
        for direction in model.DIRECTIONS:
            for extra in ({}, {"first_compute_us": 2, "first_route_us": 1}, {"tail_us": 0}):
                result = model.score_pipeline(direction, compute_service_us=100, route_service_us=10, **extra)
                self.assertEqual(result["status"], "unknown")
                self.assertIsNone(result["score_us"])
                self.assertFalse(result["full_overlap_established"])
                self.assertIsNone(result["theoretical_upper_speedup"])
        result = model.score_pipeline("A2A_GEMM", compute_service_us=100, route_service_us=10,
                                       first_route_us=1, tail_us=0, readiness="insufficient")
        self.assertEqual(result["status"], "insufficient")
        self.assertIsNone(result["score_us"])

    def test_qkv_first_compute_is_not_added_twice(self):
        for route, expected in ((10, 100), (99, 104)):
            result = model.score_pipeline("GEMM_A2A", compute_service_us=100, route_service_us=route,
                                           first_compute_us=5, tail_us=2, readiness="sufficient")
            self.assertEqual(result["relaxed_overlap_us"], expected)
            self.assertEqual(result["score_us"], expected+2)
            self.assertEqual(result["constraint"], "drain")
            self.assertEqual(result["status"], "ideal_scenario")

    def test_oproj_startup_and_producer_wait_are_not_tc_stall(self):
        arguments = dict(compute_service_us=100, route_service_us=20, first_route_us=5,
                         tail_us=1, readiness="sufficient")
        result = model.score_pipeline("A2A_GEMM", **arguments)
        self.assertEqual(result["score_us"], 106)
        waited = model.score_pipeline("A2A_GEMM", producer_wait_us=1000, **arguments)
        self.assertEqual(waited["score_us"], result["score_us"])
        self.assertFalse(waited["producer_wait_is_tensor_core_stall"])
        self.assertEqual(waited["constraint"], "feed")
        for changes in (dict(first_route_us=21), dict(compute_service_us=float("nan")),
                        dict(tail_us=-1), dict(readiness=True)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                model.score_pipeline("A2A_GEMM", **(arguments | changes))


class BadcaseTargetTests(unittest.TestCase):
    def measured(self, direction, sequence, speedup, upper=None):
        # Normalized schema from real MPI and measurement.py metadata; times
        # below remain synthetic, not a claim that these paired jobs ran.
        value = row(direction, sequence=sequence) | dict(host_launch="mpi_process", collector="mpi_rank_events_v1")
        contract = dict(node="0a", launch="eager", precision="bf16_accfp32_bf16",
                        process_layout="one_process_per_gpu", boundary="full_direction_maxrank_cuda_event",
                        cadence=dict(minimum_warmup=10, samples=50, minimum_warmup_cuda_ms=100,
                                     warmup_range_limit=.05, half_drift_limit=.05, maximum_rounds=3,
                                     barrier="per_sample", selection="first_stable_round"),
                        input_distribution=dict(kind="uniform", activation_amplitude=.125, weight_amplitude=.02))
        fields = ("direction", "m", "n", "k", "world", "layout", "hidden", "q_heads", "kv_heads",
                  "head_dim", "seq_local", "global_seq")
        baseline_problem = {key: value[key] for key in fields}
        if direction == "A2A_GEMM":
            baseline_problem["oproj_layout"] = "causal_dual_chunk_v1"
        fused = dict(collector="mpi_rank_events_v1", host_launch="mpi_process", backend="fuse",
                     run_id="normalized-mpi-fixture", source_id="a"*64, evidence_sha256="b"*64,
                     input_generator="gpu_philox", seed=20260906)
        baseline = dict(collector="primed_events_vector_max_v1", host_launch="torch_distributed", backend="te_ub",
                        run_id="normalized-baseline-fixture", source_id="c"*64, evidence_sha256="d"*64,
                        input_generator="torch_uniform", seed=2701)
        return value | dict(baseline_ms=2*speedup, fused_ms=2, upper_speedup=upper,
                            baseline_performance_accepted=True, comparison_contract=contract,
                            baseline_comparison_contract=copy.deepcopy(contract), baseline_problem=baseline_problem,
                            measurement_provenance=fused, baseline_measurement_provenance=baseline)

    def test_direction_geomeans_and_min_target_do_not_hide_unknown_caps(self):
        values = [self.measured("GEMM_A2A", 65536, .5, .75),
                  self.measured("GEMM_A2A", 131072, 1),
                  self.measured("GEMM_A2A", 262144, 2, 1.9),
                  self.measured("A2A_GEMM", 65536, 2, 10)]
        result = model.badcase_targets(values)
        qkv = result["directions"]["GEMM_A2A"]
        self.assertEqual(qkv["geomean_speedup"], 1)
        self.assertEqual(qkv["workloads"][0]["target_speedup"], .75)
        self.assertAlmostEqual(qkv["workloads"][0]["positive_log_gap"], math.log(1.5))
        self.assertTrue(qkv["workloads"][0]["below_direction_geomean"])
        self.assertIsNone(qkv["workloads"][1]["target_speedup"])
        self.assertTrue(qkv["workloads"][2]["upper_below_observed"])
        self.assertEqual(qkv["workloads"][2]["target_status"], "inconsistent_upper")
        self.assertIsNone(qkv["workloads"][2]["positive_log_gap"])
        self.assertEqual(result["directions"]["A2A_GEMM"]["geomean_speedup"], 2)
        self.assertFalse(result["full_historical_coverage_claimed"])
        json.dumps(result, allow_nan=False)

    def test_aliases_mixed_boundaries_and_unaccepted_measurements_rejected(self):
        value = self.measured("A2A_GEMM", 65536, 1)
        with self.assertRaisesRegex(ValueError, "Duplicate physical"):
            model.badcase_targets([value, value | dict(model="different-label", kv_heads=16)])
        for changes in (dict(performance_accepted=False), dict(baseline_performance_accepted=False),
                        dict(baseline_ms=float("nan")), dict(upper_speedup=0),
                        dict(baseline_comparison_contract={}), dict(comparison_contract={})):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                model.badcase_targets([value | changes])
        other = self.measured("A2A_GEMM", 131072, 2)
        other["comparison_contract"]["node"] = "09"
        other["baseline_comparison_contract"]["node"] = "09"
        with self.assertRaisesRegex(ValueError, "mixed comparison"):
            model.badcase_targets([value, other])

    def test_real_collector_names_and_different_rng_provenance_are_retained(self):
        for backend in ("te_ub", "cublaslt_nccl"):
            value = self.measured("A2A_GEMM", 262144, 1.2)
            value["baseline_measurement_provenance"]["backend"] = backend
            result = model.badcase_targets([value])["directions"]["A2A_GEMM"]["workloads"][0]
            self.assertEqual(result["fused_provenance"]["collector"], "mpi_rank_events_v1")
            self.assertEqual(result["baseline_provenance"]["collector"], "primed_events_vector_max_v1")
            self.assertNotEqual(result["fused_provenance"]["seed"], result["baseline_provenance"]["seed"])
            self.assertNotEqual(result["fused_provenance"]["source_id"], result["baseline_provenance"]["source_id"])
            self.assertEqual(result["baseline_provenance"]["backend"], backend)
        for side, change in (("baseline_measurement_provenance", dict(collector="mpi_rank_events_v1")),
                             ("measurement_provenance", dict(collector="per_epoch_rank_events_v3_eventsync",
                                                             host_launch="per_gpu_thread")),
                             ("baseline_measurement_provenance", dict(host_launch="mpi_process")),
                             ("baseline_measurement_provenance", dict(evidence_sha256="missing"))):
            value = self.measured("GEMM_A2A", 65536, 1)
            value[side].update(change)
            with self.subTest(side=side, change=change), self.assertRaises(ValueError):
                model.badcase_targets([value])

    def test_baseline_geometry_and_native_layout_cannot_be_relabelled(self):
        for change in (dict(m=16384), dict(n=8192), dict(world=4), dict(global_seq=131072),
                       dict(direction="GEMM_A2A"), dict(oproj_layout="legacy"),
                       dict(layout="sequence_rank_major_v1"), dict(defer_v_a2a=True)):
            value = self.measured("A2A_GEMM", 262144, 1)
            value["baseline_problem"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                model.badcase_targets([value])
        # The same total N can still have different Q/K/V segment boundaries.
        value = self.measured("GEMM_A2A", 65536, 1)
        for problem in (value, value["baseline_problem"]):
            problem.update(q_heads=32, kv_heads=8, n=6144)
        value["baseline_problem"].update(q_heads=16, kv_heads=16)
        with self.assertRaisesRegex(ValueError, "physical geometry"):
            model.badcase_targets([value])

    def test_common_cadence_distribution_and_process_layout_are_checked(self):
        for change in (dict(process_layout="single_process"), dict(launch="graph"), dict(precision="fp8"),
                       dict(boundary="compute_only"), dict(cadence={}),
                       dict(input_distribution=dict(kind="uniform", activation_amplitude=0, weight_amplitude=.02))):
            value = self.measured("GEMM_A2A", 65536, 1)
            value["comparison_contract"].update(change)
            value["baseline_comparison_contract"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                model.badcase_targets([value])
        value = self.measured("GEMM_A2A", 65536, 1)
        value["baseline_comparison_contract"]["cadence"]["barrier"] = "per_round"
        with self.assertRaisesRegex(ValueError, "comparison contract"):
            model.badcase_targets([value])


class OprojFeedScheduleTests(unittest.TestCase):
    """Synthetic service constants; exact queue arithmetic checked against C++."""

    @staticmethod
    def arguments(**changes):
        values = dict(m=128 * 13, n=256 * 7, k=4096, world=4,
                      sm_count=148, comm_ctas=8, tile_m=128, tile_n=256, tile_k=64,
                      raster="along_m", resolved_swizzle=4, copy_chunks=11,
                      copy_slots=32, cohort_m_tiles=2, tile_cycle_us=40.0,
                      copy_bandwidth_gb_s=400.0, launch_us=0.0,
                      copy_start_us=0.0, tail_us=0.0)
        values.update(changes)
        return values

    def test_exact_queue_matches_current_device_decoder(self):
        # Reuse the existing host-only C++ stand-in, not a second copy of the
        # device decoder. No CUDA compiler, GPU, cloud or screen is involved.
        path = ROOT / "scripts/test_sm103_producer_consumer.py"
        spec = importlib.util.spec_from_file_location("oproj_queue_oracle", path)
        oracle = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(oracle)
        host = oracle.ProducerConsumerTests
        host.setUpClass()
        self.addCleanup(host.doClassCleanups)
        import subprocess
        for mt, nt, swizzle, comm, slots, chunks, world in (
                (13, 7, 4, 8, 32, 11, 4),
                (128, 16, 8, 16, 64, 11, 4),
                (128, 28, 4, 16, 64, 16, 4),
                (33, 8, 4, 24, 96, 6, 8),
                (13, 3, 4, 3, 3, 11, 8)):
            for raster in ("along_m", "along_n"):
                with self.subTest(mt=mt, nt=nt, raster=raster, comm=comm, world=world):
                    result = model.score_oproj_schedule(**self.arguments(
                        m=mt*128, n=nt*256, resolved_swizzle=swizzle, raster=raster,
                        comm_ctas=comm, copy_slots=slots, copy_chunks=chunks, world=world,
                        cohort_m_tiles=max(1, slots//chunks), include_trace=True))
                    expected = [tuple(map(int, line.split())) for line in subprocess.check_output(
                        [str(host.binary), "input", str(mt), str(nt), str(swizzle),
                         str(int(raster == "along_n")), str(148-comm), str(world),
                         str(chunks), str(slots)], text=True).splitlines()]
                    self.assertEqual(result["copy_order"], expected)
                    self.assertEqual(len(set(expected)), mt*world*chunks)

    def test_tail_cohort_padding_and_static_worker_stride(self):
        result = model.score_oproj_schedule(**self.arguments(cohort_m_tiles=3, include_trace=True))
        self.assertEqual(result["delivery_policy"], "ready_cohorts")
        self.assertEqual((result["padded_m_tiles"], result["padded_n_tiles"]), (16, 8))
        self.assertEqual(result["scheduled_work_tiles"], 128)
        self.assertEqual(result["valid_work_tiles"], 91)
        self.assertEqual(result["copy_order"][-44:], [(12, p, c) for p in range(4) for c in range(11)])
        coordinates = set()
        for tile in result["tile_consumption"]:
            self.assertEqual(tile["worker"], tile["logical_tile"] % result["compute_ctas"])
            coordinates.add((tile["m_tile"], tile["n_tile"]))
        self.assertEqual(coordinates, {(m, n) for m in range(13) for n in range(7)})
        for finish, service, wait in zip(result["worker_finish_us"], result["worker_service_us"],
                                         result["worker_feed_wait_us"]):
            self.assertAlmostEqual(finish, service + wait)

    def test_full_ready_uses_last_chunk_and_peer_k_order(self):
        result = model.score_oproj_schedule(**self.arguments(
            m=128, n=128, k=256, tile_n=128, comm_ctas=1, copy_slots=1,
            resolved_swizzle=1, copy_chunks=2, cohort_m_tiles=1,
            copy_chunk_bytes=[4096, 12288], copy_bandwidth_gb_s=1,
            include_trace=True))
        for actual, expected in zip(result["ready_us_by_m_peer"][0], (16.384, 32.768, 49.152, 65.536)):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(result["compute_finish_us"], 75.536)
        self.assertAlmostEqual(result["critical_worker_feed_wait_us"], 35.536)
        self.assertAlmostEqual(result["critical_worker_later_feed_wait_us"], 19.152)
        self.assertGreater(result["score_us"], result["ideal_overlap_us"])
        self.assertFalse(result["tensor_core_busy_inferred"])

    def test_independent_slots_can_complete_later_peer_before_earlier_chunk(self):
        result = model.score_oproj_schedule(**self.arguments(
            m=128, n=128, k=256, tile_n=128, comm_ctas=1, copy_slots=4,
            resolved_swizzle=1, copy_chunks=2, cohort_m_tiles=1,
            copy_chunk_bytes=[4096, 12288], copy_bandwidth_gb_s=1,
            include_trace=True))
        self.assertEqual(result["ready_us_by_m_peer"][0], [49.152, 49.152, 98.304, 98.304])

    def test_fast_delivery_reduces_to_integer_wave_service(self):
        result = model.score_oproj_schedule(**self.arguments(
            m=32768, n=4096, resolved_swizzle=8, comm_ctas=16, copy_slots=64,
            cohort_m_tiles=5, copy_bandwidth_gb_s=1e12, launch_us=3, tail_us=2))
        self.assertEqual(result["integer_waves"], 32)
        self.assertAlmostEqual(result["score_us"], 3 + 32*40 + 2, places=5)
        self.assertAlmostEqual(result["exposed_feed_us"], 0, places=5)

    def test_budget_changes_compute_window_and_copy_cohort_together(self):
        old = model.score_oproj_schedule(**self.arguments(
            m=32768, n=4096, resolved_swizzle=8, comm_ctas=16, copy_slots=64,
            cohort_m_tiles=5, include_trace=True))
        new = model.score_oproj_schedule(**self.arguments(
            m=32768, n=4096, resolved_swizzle=8, comm_ctas=24, copy_slots=96,
            cohort_m_tiles=8, include_trace=True))
        self.assertEqual((old["compute_ctas"], new["compute_ctas"]), (132, 124))
        self.assertEqual((len(old["first_use_m_windows"][0]), len(new["first_use_m_windows"][0])), (17, 16))
        self.assertEqual((old["base_cohort_m_tiles"], new["base_cohort_m_tiles"]), (5, 8))
        self.assertNotEqual(old["copy_order"], new["copy_order"])
        self.assertEqual(set(old["copy_order"]), set(new["copy_order"]))

    def test_along_n_has_no_fictitious_early_global_n_band(self):
        result = model.score_oproj_schedule(**self.arguments(raster="along_n"))
        self.assertEqual(result["feed_phase"], "whole_gemm")
        self.assertEqual(result["feed_phase_compute_fraction"], 1)
        self.assertEqual(result["feed_phase_tiles"], result["valid_work_tiles"])
        self.assertEqual(result["feed_phase_ideal_compute_us"], result["strided_compute_service_us"])
        self.assertFalse(result["first_n_band_is_temporal_phase"])
        same = model.score_oproj_schedule(**self.arguments(raster="along_n", cohort_m_tiles=10))
        self.assertEqual(result["score_us"], same["score_us"])

    def test_more_n_reuse_does_not_duplicate_a_bytes(self):
        narrow = model.score_oproj_schedule(**self.arguments(m=32768, n=4096, k=8192, resolved_swizzle=8))
        wide = model.score_oproj_schedule(**self.arguments(m=32768, n=8192, k=8192, resolved_swizzle=8))
        self.assertEqual(narrow["unique_payload_bytes"], 512*1024*1024)
        self.assertEqual(wide["unique_payload_bytes"], narrow["unique_payload_bytes"])
        self.assertEqual(wide["remote_payload_bytes"], 384*1024*1024)
        self.assertEqual(narrow["copy_tasks"], wide["copy_tasks"])
        self.assertEqual(narrow["copy_finish_us"], wide["copy_finish_us"])
        self.assertEqual((narrow["feed_phase_compute_fraction"], wide["feed_phase_compute_fraction"]), (.5, .25))

    def test_start_tail_and_ratio_use_one_common_time_origin(self):
        base = model.score_oproj_schedule(**self.arguments())
        shifted = model.score_oproj_schedule(**self.arguments(launch_us=10, tail_us=7))
        self.assertAlmostEqual(shifted["score_us"], base["score_us"] + 17)
        self.assertAlmostEqual(shifted["production_consumption_ratio"], base["production_consumption_ratio"])
        self.assertAlmostEqual(base["production_consumption_ratio"],
                               base["feed_phase_ideal_compute_us"] / base["copy_finish_us"])
        self.assertNotIn("copy_order", base)
        self.assertFalse(base["externally_validated"])
        json.dumps(base, allow_nan=False)

    def test_budget_matched_cycles_are_not_linearly_scaled_by_sms(self):
        small = model.score_oproj_schedule(**self.arguments(comm_ctas=16, copy_slots=64, tile_cycle_us=43))
        large = model.score_oproj_schedule(**self.arguments(comm_ctas=32, copy_slots=128, tile_cycle_us=41))
        self.assertEqual((small["tile_cycle_us"], large["tile_cycle_us"]), (43, 41))
        self.assertEqual(large["wave_compute_service_us"], large["integer_waves"] * 41)
        self.assertFalse(large["linear_sm_throughput_scaling_applied"])

    def test_amortized_services_keep_unmeasured_startup_unknown(self):
        result = model.score_oproj_schedule(**self.arguments(service_basis="amortized_full_boundary"))
        self.assertIsNone(result["separately_calibrated_startup_tail_us"])
        for term in ("launch_us", "copy_start_us", "tail_us"):
            with self.subTest(term=term), self.assertRaisesRegex(ValueError, "second time"):
                model.score_oproj_schedule(**self.arguments(service_basis="amortized_full_boundary", **{term: 1}))

    def test_invalid_geometry_services_and_chunk_vectors_fail_closed(self):
        for changes in (dict(m=129), dict(k=4097), dict(comm_ctas=148), dict(copy_chunks=0),
                        dict(copy_slots=33), dict(cohort_m_tiles=0), dict(resolved_swizzle=3),
                        dict(raster="heuristic"), dict(tile_cycle_us=0), dict(copy_bandwidth_gb_s=float("inf")),
                        dict(launch_us=-1), dict(copy_start_us=-1), dict(tail_us=float("nan")),
                        dict(copy_chunk_bytes=[1]), dict(copy_chunk_bytes=[1]*11),
                        dict(include_trace=1), dict(m=128_000_000), dict(world=3),
                        dict(sm_count=2_000_002, comm_ctas=2_000_001, copy_slots=2_000_001)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                model.score_oproj_schedule(**self.arguments(**changes))


if __name__ == "__main__":
    unittest.main()
