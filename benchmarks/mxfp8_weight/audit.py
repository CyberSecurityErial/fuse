"""Audit collected best-tested rows against the exact legacy matrix (no GPU)."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from matrix import full_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-full", action="store_true")
    args = parser.parse_args()
    cases = {r["id"]: r for r in full_matrix()}
    expected = {(key, b, l) for key in cases for b in ("cublaslt_nccl", "teub")
                for l in ("eager", "graph")}
    seen, rows, errors = set(), [], []
    for path in args.inputs:
        report = json.loads(path.read_text())
        source_reports = {}
        for source, digest in report.get("source_reports", {}).items():
            if not Path(source).is_file() or hashlib.sha256(Path(source).read_bytes()).hexdigest() != digest:
                errors.append(f"{path}: source report missing/modified: {source}")
            else:
                source_reports[source] = json.loads(Path(source).read_text())
        if (report["warmup"], report["iterations"]) != (10, 50):
            errors.append(f"{path}: not formal 10/50 sampling")
        for item in report["cases"]:
            case = item["case"]
            if case != cases.get(case["id"]):
                errors.append(f'{case["id"]}: geometry/metadata differs from legacy matrix')
            if report["cuda_visible_devices"] != case["visible_devices"]:
                errors.append(f'{case["id"]}: physical GPU group differs from legacy matrix')
            for row in item["best_tested"]:
                key = (case["id"], row["backend"], row["launch"])
                if source_reports:
                    source = source_reports.get(row.get("source_run"))
                    if source is None:
                        errors.append(f"No authenticated source run: {key}")
                    else:
                        matches = [r for c in source["cases"] if c["case"] == case
                                   for r in c["best_tested"] if (r["backend"], r["launch"]) == key[1:]]
                        if len(matches) != 1 or any(row.get(k) != v for k, v in matches[0].items()):
                            errors.append(f"Combined row differs from its raw source: {key}")
                if key in seen or key not in expected:
                    errors.append(f"Duplicate or unexpected result: {key}")
                seen.add(key)
                samples, ranks = row["samples_us"], row["rank_samples_us"]
                valid = (len(samples) == 50 and len(ranks) == case["cp"] and
                         all(len(r) == 50 for r in ranks) and
                         all(math.isfinite(x) and x > 0 for x in samples))
                if not valid or samples != [max(v) for v in zip(*ranks)]:
                    errors.append(f"Invalid sample-wise rank-max timing: {key}")
                if "warmup" in row and (row["warmup"], row["iterations"], row["phase"]) != (10, 50, "formal"):
                    errors.append(f"Not independent formal sampling: {key}")
                if len(samples) == 50:
                    ordered = sorted(samples)
                    p50 = (ordered[24] + ordered[25]) / 2
                    p95 = ordered[46] * .45 + ordered[47] * .55
                    if not math.isclose(row["p50_us"], p50) or not math.isclose(row["p95_us"], p95):
                        errors.append(f"Stored percentiles differ from raw samples: {key}")
                for gate in ("correctness", "correctness_after_samples"):
                    value = row[gate]["relative_rmse"]
                    if not math.isfinite(value) or value > .005:
                        errors.append(f"Failed {gate}: {key}")
                rows.append(dict(id=key[0], backend=key[1], launch=key[2],
                                 global_seq=case["global_seq"], cp=case["cp"],
                                 m=case["m"], n=case["n"], k=case["k"],
                                 p50_us=row["p50_us"], p95_us=row["p95_us"],
                                 ub_comm_sms=row["ub_comm_sms"],
                                 schedule=row.get("config", {}).get("schedule", "full"),
                                 config=json.dumps(row.get("config", {}), sort_keys=True),
                                 nccl_profile=row.get("nccl_profile", ""),
                                 source=row.get("source_run", str(path))))
    missing = sorted(expected - seen)
    result = dict(expected_settings=192, expected_rows=len(expected),
                  collected_rows=len(seen & expected), complete=not missing and not errors,
                  errors=errors, missing=missing)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if rows:
        with args.output.with_suffix(".csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({k: v for k, v in result.items() if k != "missing"}, indent=2))
    if errors or (args.require_full and missing):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
