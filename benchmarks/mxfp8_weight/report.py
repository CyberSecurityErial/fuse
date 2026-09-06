"""Render the complete 192-setting table from authenticated formal results."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

from matrix import full_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    paths = [args.directory / f"combined_cp{cp}.json" for cp in (4, 8)]
    subprocess.run([sys.executable, str(Path(__file__).with_name("audit.py")),
                    *map(str, paths), "--output", str(args.directory / "coverage.json"),
                    "--require-full"], check=True)
    reports = [json.loads(p.read_text()) for p in paths]
    data = {c["case"]["id"]: c for report in reports for c in report["cases"]}
    lines = ["# MXFP8 权重 SM90 基线：完整正式结果", "",
             "192 个旧矩阵配置 × 2 后端 × 2 执行模式 = 768 条结果；全部通过覆盖审计。",
             "权重离线 E4M3+E8M0（每 32 元素），每次运行时反量化为 BF16；激活、GEMM、A2A、输出均为 BF16。",
             "时延包含反量化、GEMM、打包、通信及解包；不含离线量化、分配、编译、调参和校验。",
             "10 次预热、50 次独立正式采样，每个样本取所有 rank 的最大值；表中为 p50，单位 µs。",
             "TEUB/NCCL 的选择均来自独立短采样；这里的“最佳”仅限 README 列出的有限候选，不宣称全局最优。", "",
             "## 硬件与执行口径", "",
             "沿用用户指定的 H200 机器和原 CP 卡组。环境将运行时名称显示为 NVIDIA L20X；原始名称不改写。",
             "CUDA 实测属性及 UUID 保存在 JSON 中，不用显示名称推断硬件计算能力。", ""]
    for report in reports:
        device = report["devices"][0]
        lines.append(f'- CP{len(report["devices"])}：物理卡 {report["cuda_visible_devices"]}；'
                     f'CUDA CC {device["cc"]}，{device["sms"]} SM，'
                     f'{device["memory_bytes"] / 2**30:.2f} GiB/卡。')
    lines += ["", "Eager CUDA event 时延可能包含主机提交间隙；Graph 为单次完整算子 replay，不能混为模型 E2E 吞吐。",
              "旧 bench 的 shape/布局/采样数/rank-max 保持一致，但采样同步工具改为 CPU barrier，历史数字不当作同次 A/B。", ""]
    summary = []
    for direction in ("gemm_a2a", "a2a_gemm"):
        cases = [c for c in full_matrix() if c["direction"] == direction]
        lines += [f"## {'Gemm→A2A' if direction == 'gemm_a2a' else 'A2A→Gemm'}：96 个配置", "",
                  "加速比 = NCCL 时延 / TEUB 时延，>1 表示 TEUB 更快。", "",
                  "| 模型 | 全局 S | CP | M×N×K | NCCL Eager | TEUB Eager | 加速 | NCCL Graph | TEUB Graph | 加速 |",
                  "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
        speeds = {"eager": [], "graph": []}
        for case in cases:
            rows = {(r["backend"], r["launch"]): r for r in data[case["id"]]["best_tested"]}
            numbers = []
            for launch in ("eager", "graph"):
                nccl = rows["cublaslt_nccl", launch]["p50_us"]
                teub = rows["teub", launch]["p50_us"]
                speeds[launch].append(nccl / teub)
                numbers += [f"{nccl:.3f}", f"{teub:.3f}", f"{nccl/teub:.3f}×"]
            lines.append(f'| {case["model"]} | {case["global_seq"]} | {case["cp"]} | '
                         f'{case["m"]}×{case["n"]}×{case["k"]} | ' + " | ".join(numbers) + " |")
        for launch, values in speeds.items():
            summary.append(dict(direction=direction, launch=launch,
                                teub_faster=sum(v > 1 for v in values), settings=len(values),
                                unweighted_geomean_speedup=math.exp(sum(map(math.log, values))/len(values))))
        lines.append("")
    maximum_abs = max(r["correctness_after_samples"]["max_abs"] for c in data.values() for r in c["best_tested"])
    maximum_rrmse = max(r["correctness_after_samples"]["relative_rmse"] for c in data.values() for r in c["best_tested"])
    lines += ["## 校验与可追溯性", "",
              f"正式结果对独立 BF16 参考的最大绝对误差：{maximum_abs:.8g}；最大相对 RMSE：{maximum_rrmse:.8g}。",
              "量化相对原始权重的误差另存于每个 case 的 weight_quantization_error，不与实现误差混用。",
              "coverage.csv 保存 768 条 p50/p95、完整配置和原始文件路径；JSON 保留每张卡的所有样本、cuBLASLt 算法及源文件哈希。",
              "source_reports 哈希与每条汇总行到原始行的映射均由 audit.py 验证。", ""]
    (args.directory / "REPORT.md").write_text("\n".join(lines) + "\n")
    (args.directory / "statistics.json").write_text(json.dumps(dict(summary=summary,
         max_abs=maximum_abs, max_relative_rmse=maximum_rrmse), indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
