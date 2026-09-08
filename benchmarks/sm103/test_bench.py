"""CPU-only checks of benchmark contracts; no CUDA packages needed."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

import bench
import worker


class BenchmarkContracts(unittest.TestCase):
    def args(self, **overrides):
        return argparse.Namespace(**({
            "directions": "qkv,oproj", "models": "", "seqs": (1024,4096,16384,131072,262144,524288),
            "cps": (4,8), "devices": "0,1,2,3,4,5,6,7", "sm_count": 148,
            "stage": "formal", "results": Path("/tmp/sm103-test-unused"),
            "library": Path("/tmp/sm103-test-unused/lib.so"), "python": "python3", "te_root": None,
        } | overrides))

    def test_historical_shape_contract(self):
        cases = list(bench.cases(self.args()))
        self.assertEqual(len(cases), 192)
        for direction in ("qkv", "oproj"):
            self.assertEqual(sum(c["direction"] == direction for c in cases), 96)
        qkv = next(c for c in cases if c["direction"] == "qkv" and c["model"] == "llama3_8b")
        self.assertEqual((qkv["m"], qkv["n"], qkv["k"]), (256, 6144, 4096))

    def test_production_matrix_excludes_1k_4k(self):
        self.assertEqual(bench.PRODUCTION_SEQUENCES, (16384, 131072, 262144, 524288))
        cases = list(bench.cases(self.args(seqs=bench.PRODUCTION_SEQUENCES)))
        self.assertEqual(len(cases), 128)
        self.assertFalse(any(c['seq'] in (1024, 4096) for c in cases))
        for direction in ('qkv', 'oproj'):
            self.assertEqual(sum(c['direction'] == direction for c in cases), 64)

    def test_formal_ub_uses_actual_sm_budget_and_launch(self):
        args = self.args()
        case = next(bench.cases(args))
        config = bench.initial_configs("te_ub")[0]
        eager = bench.make_job(args, case, "te_ub", "eager", config)
        graph = bench.make_job(args, case, "te_ub", "graph", config)
        self.assertEqual(eager["command"][eager["command"].index("--math-sm")+1], "144")
        self.assertIn("--no-cuda-graph", eager["command"])
        self.assertIn("--cuda-graph", graph["command"])
        self.assertEqual(eager["env"]["FUSE_CUBLASLT_TUNE_GRAPH"], "0")
        self.assertEqual(graph["env"]["FUSE_CUBLASLT_TUNE_GRAPH"], "1")
        self.assertEqual((graph["warmup"], graph["iterations"]), (10,50))
        self.assertNotEqual(eager["output"], graph["output"])
        self.assertNotIn("--local-first", bench.make_job(
            args, case | {"direction": "oproj"}, "te_ub", "eager", config)["command"])

    def test_nccl_search_space_and_geometry_validation(self):
        configs = bench.initial_configs("cublaslt_nccl")
        self.assertEqual(len({bench.digest(c) for c in configs}), 48)
        self.assertEqual(len(bench.refine(configs[0], "cublaslt_nccl", "qkv")), 16)
        with self.assertRaises(ValueError):
            list(bench.cases(self.args(cps=(3,))))

    def test_oproj_layout_is_opt_in_and_fingerprinted(self):
        args = self.args(directions='oproj')
        case = next(bench.cases(args))
        config = bench.initial_configs('te_ub')[0]
        old = bench.make_job(args, case, 'te_ub', 'eager', config)
        self.assertEqual(old['env']['FUSE_SM103_OPROJ_LAYOUT'], 'legacy')
        fp = bench.fingerprint(args)
        args.oproj_layout = 'causal_dual_chunk_v1'
        new = bench.make_job(args, case, 'te_ub', 'eager', config)
        self.assertEqual(new['env']['FUSE_SM103_OPROJ_LAYOUT'], args.oproj_layout)
        self.assertEqual(old['command'], new['command'])
        self.assertNotEqual(fp, bench.fingerprint(args))
        qkv = bench.make_job(args, case | {'direction': 'qkv'}, 'te_ub', 'eager', config)
        self.assertEqual(qkv['env']['FUSE_SM103_OPROJ_LAYOUT'], 'legacy')
        args.oproj_layout = 'unknown'
        with self.assertRaisesRegex(ValueError, 'layout'):
            bench.make_job(args, case, 'te_ub', 'eager', config)

    def test_oproj_reader_requires_matching_raw_and_every_rank_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'raw.json'
            job = {'case': {'direction': 'oproj', 'cp': 2}, 'backend': 'te_ub',
                   'iterations': 3, 'launch': 'eager'}
            data = {'world_size': 2, 'launch': 'eager', 'samples_ms': [1, 2, 3],
                    'correctness': {'route_mismatches': 0}}
            metadata = {'device': {'compute_capability': '10.3'}}
            path.write_text(json.dumps(data))
            for rank in range(2):
                path.with_suffix(f'.rank{rank}.json').write_text(json.dumps(metadata))
            # Historical receipts have no layout field; only legacy may read them.
            self.assertEqual(bench.read_measurement(path, job)['p50_ms'], 2)
            job['env'] = {'FUSE_SM103_OPROJ_LAYOUT': 'causal_dual_chunk_v1'}
            with self.assertRaisesRegex(ValueError, 'result layout'):
                bench.read_measurement(path, job)
            worker.record_result_layout(path, {'rank': 0, 'direction': 'oproj',
                                              'oproj_layout': 'causal_dual_chunk_v1'})
            for rank in range(2):
                with self.assertRaisesRegex(ValueError, f'rank {rank} layout'):
                    bench.read_measurement(path, job)
                path.with_suffix(f'.rank{rank}.json').write_text(json.dumps(
                    metadata | {'oproj_layout': 'causal_dual_chunk_v1'}))
            self.assertEqual(bench.read_measurement(path, job)['p50_ms'], 2)
            for layout in ('legacy', 'unknown'):
                job['env']['FUSE_SM103_OPROJ_LAYOUT'] = layout
                with self.assertRaisesRegex(ValueError, 'layout'):
                    bench.read_measurement(path, job)

    def test_oproj_adapter_only_rebinds_requested_boundary(self):
        import measurement
        before, after, old_pack, new_pack = (object() for _ in range(4))
        boundary = types.SimpleNamespace(
            get_seq_chunk_ids_for_reordering_before_attn=before,
            _pack_inverse_a2a_kernel=old_pack)
        name = 'transformer_engine.pytorch.attention.dot_product_attention.context_parallel'
        te = types.ModuleType(name)
        te.get_seq_chunk_ids_for_reordering_after_attn = after
        with (mock.patch.dict(sys.modules, {name: te, 'measurement': measurement}),
              mock.patch.object(worker, 'canonical_oproj_pack', return_value=new_pack) as pack):
            worker.configure_oproj_layout(boundary, 'oproj', 'te_ub', 'legacy')
            worker.configure_oproj_layout(boundary, 'qkv', 'te_ub', 'causal_dual_chunk_v1')
            self.assertIs(boundary._pack_inverse_a2a_kernel, old_pack)
            pack.assert_not_called()
            worker.configure_oproj_layout(boundary, 'oproj', 'cublaslt_nccl', 'causal_dual_chunk_v1')
            self.assertIs(boundary.get_seq_chunk_ids_for_reordering_before_attn, after)
            self.assertIs(te.get_seq_chunk_ids_for_reordering_after_attn, after)
            worker.configure_oproj_layout(boundary, 'oproj', 'te_ub', 'causal_dual_chunk_v1')
            self.assertIs(boundary._pack_inverse_a2a_kernel, new_pack)
            with self.assertRaisesRegex(ValueError, 'layout'):
                worker.configure_oproj_layout(boundary, 'oproj', 'te_ub', 'unknown')

    def test_canonical_pack_actual_kernel_scalar_cpu_emulation(self):
        # Execute the real kernel body one lane at a time; no CUDA/Triton import.
        class Pointer:
            def __init__(self, data, offset=0):
                self.data, self.offset = data, offset
            def __add__(self, offset):
                return Pointer(self.data, self.offset + offset)
        position = [0]
        tl = types.ModuleType('triton.language')
        tl.constexpr = object()
        tl.program_id = lambda _: position[0]
        tl.arange = lambda begin, end: 0
        tl.where = lambda condition, a, b: a if condition else b
        tl.load = lambda pointer, mask: pointer.data[pointer.offset] if mask else 0
        def store(pointer, value, mask):
            if mask:
                pointer.data[pointer.offset] = value
        tl.store = store
        triton = types.ModuleType('triton')
        triton.language, triton.jit = tl, lambda fn: fn
        with mock.patch.dict(sys.modules, {'triton': triton, 'triton.language': tl}):
            pack = worker.canonical_oproj_pack()
            for world in (4, 8):
                for batch in (1, 2):
                    chunk_rows, kl = 3, 15
                    seq, local = 2*world*chunk_rows, 2*chunk_rows
                    source = list(range(batch*seq*kl))
                    output = [None]*len(source)
                    for offset in range(len(source)+2):
                        position[0] = offset
                        pack(Pointer(source), Pointer(output), len(source), seq,
                             local, chunk_rows, kl, world, 1)
                    expected = []
                    for rank in range(world):
                        for b in range(batch):
                            for row in range(local):
                                global_row = (rank*chunk_rows+row if row < chunk_rows else
                                    (2*world-rank-1)*chunk_rows+row-chunk_rows)
                                begin = (b*seq+global_row)*kl
                                expected.extend(source[begin:begin+kl])
                    self.assertEqual(output, expected)

    def test_adapted_ub_accepts_arbitrary_grid(self):
        args = self.args()
        case = next(bench.cases(args))
        configs = bench.initial_configs("te_ub")
        self.assertEqual([c["comm_sm"] for c in configs], [4,8,12,16,20,24])
        for sm in range(1,33):
            bench.make_job(args, case, "te_ub", "eager", configs[0] | {"comm_sm": sm})
        with self.assertRaisesRegex(ValueError, "UB_MAX_SM"):
            bench.make_job(args, case, "te_ub", "eager", configs[0] | {"comm_sm": 64})

    def test_reduction_and_rejection_of_invalid_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.json"
            job = {"case": {"direction": "qkv", "cp": 2}, "backend": "te_ub",
                   "iterations": 3, "launch": "graph"}
            data = {"world_size": 2, "launch": "graph", "samples_ms": [1,10,2],
                    "correctness": {"route_mismatches": 0}}
            for rank in range(2):
                path.with_suffix(f".rank{rank}.json").write_text(json.dumps(
                    {"device": {"compute_capability": "10.3"}}))
            path.write_text(json.dumps(data))
            stats = bench.read_measurement(path, job)
            self.assertEqual(stats["p50_ms"], 2)
            self.assertAlmostEqual(stats["p95_ms"], 9.2)
            for change in ({"samples_ms": [1,2]}, {"launch": "eager"},
                           {"correctness": {"route_mismatches": 1}}):
                path.write_text(json.dumps(data | change))
                with self.assertRaises(ValueError):
                    bench.read_measurement(path, job)

    def test_launch_selection_and_default_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            for launches, expected in ((None, 8), ("graph", 4)):
                destination = Path(directory) / (launches or "default")
                command = ["bench.py", "--stage", "smoke", "--results", str(destination)]
                if launches:
                    command.extend(["--launches", launches])
                with mock.patch.object(sys, "argv", command), contextlib.redirect_stdout(io.StringIO()):
                    bench.main()
                plan = json.loads((destination / "smoke_plan.json").read_text())
                self.assertEqual(len(plan["jobs"]), expected)
                self.assertTrue(all(job["case"]["cp"] == 8 for job in plan["jobs"]))
                if launches:
                    self.assertEqual({job["launch"] for job in plan["jobs"]}, {"graph"})

    def test_stage_checks_once_and_resume_skips_gpu_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            library = destination / "lib.so"
            library.write_bytes(b"test-library")
            command = ["bench.py", "--stage", "smoke", "--launches", "graph", "--execute",
                       "--results", str(destination), "--library", str(library)]
            rows = "\n".join(f"{i}, GPU-{i}, 0, 32768, 0, 1500, 1000, 50" for i in range(8))
            observation = subprocess.CompletedProcess([], 0, stdout=rows)

            def finish_job(argv, env, log, timeout):
                self.assertEqual(timeout, 600)
                Path(argv[argv.index("--json-out") + 1]).write_text("{}")

            with (mock.patch.object(sys, "argv", command), mock.patch.dict(os.environ, {}, clear=True),
                  mock.patch.object(bench.subprocess, "run", return_value=observation) as query,
                  mock.patch.object(bench.time, "sleep") as sleep,
                  mock.patch.object(bench, "run_job", side_effect=finish_job) as run,
                  mock.patch.object(bench, "read_measurement", return_value={}),
                  contextlib.redirect_stdout(io.StringIO())):
                bench.main()
                self.assertEqual(query.call_count, 3 + 4)
                self.assertEqual(sleep.call_count, 2)
                self.assertEqual(run.call_count, 4)
                query.reset_mock()
                bench.main()
                query.assert_not_called()
            self.assertEqual(len(json.loads((destination / "smoke.gpu-before.json").read_text())), 3)
            for path in (destination / "smoke").glob("*.gpu-before.json"):
                self.assertEqual(len(json.loads(path.read_text())), 1)

    def test_snapshots_allow_previous_job_utilization_but_check_memory(self):
        result = subprocess.CompletedProcess([], 0, stdout="0, GPU-0, 90, 4096, 0, 1, 1, 1")
        with mock.patch.object(bench.subprocess, "run", return_value=result), mock.patch.object(bench.time, "sleep") as sleep:
            with self.assertRaisesRegex(ValueError, "is computing"):
                bench.observe_devices("0", samples=1)
            self.assertEqual(len(bench.observe_devices("0", samples=1, check_idle=False)), 1)
            result.stdout = "0, GPU-0, 0, 1000, 0, 1, 1, 1"
            with self.assertRaisesRegex(ValueError, "less than 2 GiB"):
                bench.observe_devices("0", samples=1, check_idle=False)
            sleep.assert_not_called()

    def test_optional_environment_fingerprint(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            absent = bench.fingerprint(self.args())
            os.environ["FUSE_ENV_FINGERPRINT"] = "runtime-a"
            runtime_a = bench.fingerprint(self.args())
            self.assertNotEqual(absent, runtime_a)
            os.environ["FUSE_ENV_FINGERPRINT"] = "runtime-b"
            self.assertNotEqual(runtime_a, bench.fingerprint(self.args()))
            os.environ["FUSE_ENV_FINGERPRINT"] = ""
            self.assertEqual(absent, bench.fingerprint(self.args()))

    def test_timeout_and_failure_log_are_actionable(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            log_path = output.with_suffix(".log")
            started = time.monotonic()
            with log_path.open("w") as log:
                with self.assertRaises(subprocess.TimeoutExpired):
                    bench.run_job([sys.executable, "-c", "import time; time.sleep(60)"],
                                  os.environ.copy(), log, .1)
            self.assertLess(time.monotonic() - started, 5)
            log_path.write_text("".join(f"line-{i}\n" for i in range(40)) + "Missing dependency: example_package\n")
            job = {"output": str(output), "group": "cp8-graph", "config": {}, "command": ["benchmark"]}
            error = subprocess.CalledProcessError(7, job["command"])
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                bench.report_job_failure(job, error, 600)
            self.assertIn("Missing dependency: example_package", stderr.getvalue())
            self.assertNotIn("line-10\n", stderr.getvalue())
            self.assertIn("line-11\n", stderr.getvalue())
            failure = json.loads(output.with_suffix(".failed.json").read_text())
            self.assertEqual(failure["returncode"], 7)
            self.assertEqual(failure["log"], str(log_path))

    def test_timeout_kills_group_after_leader_exits(self):
        process = mock.MagicMock()
        process.pid = 12345
        process.wait.side_effect = [subprocess.TimeoutExpired(["job"], 1), 0, 0]
        with (mock.patch.object(bench.subprocess, "Popen") as popen,
              mock.patch.object(bench.os, "killpg") as killpg):
            popen.return_value.__enter__.return_value = process
            with self.assertRaises(subprocess.TimeoutExpired):
                bench.run_job(["job"], {}, io.StringIO(), 1)
            self.assertEqual(killpg.call_args_list, [
                mock.call(12345, signal.SIGTERM), mock.call(12345, 0), mock.call(12345, signal.SIGKILL)])

    def test_sigterm_cleans_up_owned_subprocess(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "child.log"
            worker = "import os,time; print(os.getpid(), flush=True); time.sleep(60)"
            outer = (
                f"import os,signal,sys; sys.path.insert(0, {str(bench.HERE)!r}); import bench; "
                "signal.signal(signal.SIGTERM, bench.handle_termination); "
                f"log=open({str(log_path)!r}, 'w'); "
                f"bench.run_job([sys.executable, '-c', {worker!r}], os.environ.copy(), log, 60)"
            )
            process = subprocess.Popen([sys.executable, "-c", outer], stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, start_new_session=True)
            child_pid = None
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    if log_path.exists() and log_path.read_text().strip():
                        child_pid = int(log_path.read_text().strip())
                        break
                    time.sleep(.01)
                self.assertIsNotNone(child_pid, "test child did not start")
                process.terminate()
                process.wait(timeout=5)
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                if child_pid:
                    try:
                        os.killpg(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass


if __name__ == "__main__":
    unittest.main()
