"""Audit both suites and validation evidence before claiming four operators."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from backward_matrix import ROOT
from backward_report import audit, correctness_errors
from run_suite import PROFILES, write


def forward_evidence(directory):
    """Supplement the forward row audit with source and selection checks."""
    errors, evidence = [], {}
    for cp in (4,8):
        combined = json.loads((directory/f'combined_cp{cp}.json').read_text())
        for source, sha in combined['source_reports'].items():
            path = Path(source)
            if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                errors.append(f'forward raw report modified: {source}')
            raw = json.loads(path.read_text())
            if not raw.get('sources'):
                errors.append(f'forward report missing code provenance: {source}')
            for code, expected_sha in raw.get('sources', {}).items():
                target = Path(code)
                if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != expected_sha:
                    errors.append(f'forward benchmark source changed: {code}')
            evidence[source] = sha
        surveys = {}
        for profile in PROFILES:
            raw = json.loads((directory/f'cublaslt_nccl_sweep_{profile}_cp{cp}.json').read_text())
            if (raw['warmup'],raw['iterations']) != (3,12):
                errors.append(f'forward NCCL short protocol changed: {profile}/CP{cp}')
            for c in raw['cases']:
                for r in c['best_tested']:
                    surveys.setdefault((c['case']['id'],r['launch']), []).append((r['p50_us'],profile))
        policy_rows = json.loads((directory/f'nccl_policy_cp{cp}.json').read_text())
        policy = {(r['id'],r['launch']):r['profile'] for r in policy_rows}
        expected = {(c['case']['id'],l) for c in combined['cases'] for l in ('eager','graph')}
        if set(policy) != expected or len(policy_rows) != len(expected):
            errors.append(f'forward CP{cp} policy coverage mismatch')
        for key, profile in policy.items():
            candidates = surveys.get(key, [])
            if len(candidates) != len(PROFILES) or min(candidates)[1] != profile:
                errors.append(f'forward NCCL not independent sweep winner: {key}')
        for c in combined['cases']:
            for r in c['best_tested']:
                if not all(math.isfinite(v) and v > 0 for rank in r['rank_samples_us'] for v in rank):
                    errors.append(f'forward nonfinite rank timing: {c["case"]["id"]}')
                key = (c['case']['id'],r['launch'])
                if r['backend'] == 'cublaslt_nccl':
                    if r['nccl_profile'] != policy.get(key):
                        errors.append(f'forward formal profile differs from policy: {key}')
                else:
                    candidates = [s for s in c['sweep_records'] if s['launch']==r['launch'] and s['backend']=='teub']
                    if not candidates or min(candidates, key=lambda s:s['p50_us'])['config'] != r['config']:
                        errors.append(f'forward TEUB not independent sweep winner: {key}')
                    if any((s['warmup'],s['iterations'],s['phase']) != (3,12,'sweep') for s in candidates):
                        errors.append(f'forward TEUB short protocol changed: {key}')
    return errors, evidence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--forward', type=Path, default=ROOT/'results/mxfp8_weight/full_v2')
    p.add_argument('--backward', type=Path, default=ROOT/'results/mxfp8_weight/backward_full_v1')
    p.add_argument('--validation', type=Path, default=ROOT/'results/mxfp8_weight/backward_validation_v2')
    args = p.parse_args()
    subprocess.run([sys.executable, str(Path(__file__).with_name('report.py')), str(args.forward)], check=True)
    forward = json.loads((args.forward/'coverage.json').read_text())
    backward, rows = audit(args.backward)
    errors, forward_sources = forward_evidence(args.forward)
    if not forward['complete'] or forward['collected_rows'] != 768:
        errors.append('forward is not complete 192 settings/768 rows')
    if not backward['complete'] or backward['collected_rows'] != 1536:
        errors.append('backward is not complete 192 settings/1536 rows')
    validation = []
    for cp in (4,8):
        path = args.validation/f'cp{cp}.json'
        if not path.exists():
            errors.append(f'missing validation CP{cp}'); continue
        r = json.loads(path.read_text())
        if not r['complete'] or r.get('one_rank_corruption_gate') != 'passed_nan_inf_finite':
            errors.append(f'incomplete validation CP{cp}')
        for source, digest in r['sources'].items():
            if hashlib.sha256((ROOT/source).read_bytes()).hexdigest() != digest:
                errors.append(f'validation source changed: {source}')
        expected = {(op,b,layout) for op in ('qkv','oproj') for b in (1,2)
                    for layout in ('rank_major','causal_paired')}
        got = {(c['case']['operator'], c['case']['batch'], c['case']['layout']) for c in r['cases']}
        if got != expected or len(r['cases']) != 8:
            errors.append(f'incomplete validation layout/batch matrix CP{cp}')
        for c in r['cases']:
            if c.get('autograd_check') is None:
                errors.append('missing independent actual autograd')
            else:
                errors += correctness_errors(c['autograd_check'])
            expected_rows = {(b,l,m) for b in ('cublaslt_nccl','teub')
                             for l in ('eager','graph') for m in ('immediate','deferred')}
            if {(x['backend'],x['launch'],x['weight_mode']) for x in c['records']} != expected_rows:
                errors.append('incomplete validation backends/launches/modes')
            for x in c['records']:
                errors += correctness_errors(x['correctness'])
        validation.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               cases=len(r['cases']), records=sum(len(c['records']) for c in r['cases'])))
    summary = dict(complete=not errors, forward_rows=forward['collected_rows'],
                   backward_rows=backward['collected_rows'], total_settings=384,
                   total_rows=2304, errors=errors, validation=validation,
                   forward_coverage_sha256=hashlib.sha256((args.forward/'coverage.json').read_bytes()).hexdigest(),
                   forward_source_reports=forward_sources,
                   backward_audit=backward)
    output = args.backward/'four_operator_coverage.json'
    write(output, summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('backward_audit','validation')}, indent=2))
    if errors:
        raise SystemExit(1)
    text = '''# MXFP8 权重量化：四算子基线交付范围

四算子覆盖与正确性证据已重新审计；这是 benchmark，不是生产 Fuse 内核。

| 算子 | 原 SM90 配置数 | 正式结果数 | 输出语义 |
|---|---:|---:|---|
| QKV 前向：GEMM→A2A | 96 | 384 | BF16 Q/K/V |
| OProj 前向：A2A→GEMM | 96 | 384 | BF16 输出 |
| QKV 反向：A2A→dX，dW | 96 | 768 | BF16 dX，FP32 main_grad |
| OProj 反向：dA→A2A，dW | 96 | 768 | BF16 dA，FP32 main_grad |
| 合计 | 384 | 2304 | 前向与反向不混计 |

每个算子沿用各自旧 SM90 注册表：8 组模型 × 6 档全局序列 × CP4/CP8。
序列为 1024/4096/16384/131072/262144/524288；batch=1，M=S/CP。
前向各有两个后端和 Eager/Graph；反向再区分普通 beta=0 与分离 B/W beta=1。

权重在计时外离线量化为 MXFP8。每次使用权重的调用，在计时内软件反量化到
BF16 workspace，再做 BF16 GEMM；没有原生 MXFP8 Tensor Core，也没有 token
量化或通信压缩。反向 dW 不读取权重，不额外添加反量化或优化器成本。

新反向主表采用用户确认的 FP32 dW/main_grad。旧 SM90 反向的 main_grad 是
BF16；shape、CP、seqlen、路由和计时边界对齐，不把梯度精度变化伪装成“只改量化”。
前向 QKV 的正式布局是 rank-major，反向两算子的正式布局是 causal paired，
均匹配各自旧测试；本交付没有声称四张表来自一次训练 E2E。

所有正式项为 10 次预热、50 个样本，先逐样本取跨 rank 最大值，再算 p50/p95。
反向单独记录 B、W 和顺序 B→W 全程；独立两段的样本和另列，不拿它冒充实测全程。
deferred 顺序全程只是 beta=1 基线，不是 ZeroBubble 调度收益。

搜索是声明的有限候选，不是全参数穷举。反向通信按 B 阶段短扫选优；cuBLASLt
按实际 dtype/beta 独立选择算法，再对优胜通信配置做正式复测。前反向的候选集
分别见各自 README，不能声称两者做了完全相同的编排优化。

硬件沿用用户指定 H200 和原 CP 卡组；JSON 保留环境原始名称及 CC9.0/132SM。
TEUB 指 TE Userbuffers P2P + cuBLASLt。表内加速比只比较同语义的 TEUB/NCCL；
不是相对旧 BF16 的纯量化收益，也不是 Fuse 或推理框架 E2E 收益。

## 可审计产物

* 前向：`../full_v2/REPORT.md`、`coverage.csv`、原始 rank 样本和候选结果。
* 反向：`REPORT.md`、`coverage.csv`、`coverage.json`、原始短扫及独立正式结果。
* 四算子证据：`four_operator_coverage.json`，含前向覆盖哈希、反向审计及正确性文件哈希。
* 扩展正确性：`../backward_validation_v2/cp4.json` 与 `cp8.json`，覆盖两种布局、
  batch=1/2、实际 autograd、非零梯度两次累加和单 rank NaN/Inf/错误传播。
'''
    (args.backward/'FOUR_OPERATORS.md').write_text(text)


if __name__ == '__main__':
    main()
