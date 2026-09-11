#!/usr/bin/env python3
"""Consume audited pure GEMM winners, then tune fused communication budgets.

The kernel has no model-name table. This offline experiment preserves each
MNK winner as a seed, jointly explores neighboring layout/worker choices, and retains a paired
E64/M/sw1/comm16 control. Every candidate uses full Graph 10+50 and two payloads.
Run via the normal Mac/mc/screen controller, never a direct remote connection.
"""
import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import sys

from summarize_sm103_mxfp8_fused import audit_run

ROOT = Path(__file__).resolve().parents[1]


def gemm_config(row):
    match = re.fullmatch(r'm128n256k128e(32|64)s0sw([1248])([MN])', row['winner']['config'])
    if not match:
        raise ValueError('Winner requires an unregistered fused collective: '+row['id'])
    return int(match[1]), int(match[2]), 'along_m' if match[3] == 'M' else 'along_n'


def layout_neighbors(config):
    epilogue, swizzle, raster = config
    widths = (1, 2, 4, 8)
    index = widths.index(swizzle)
    result = [(epilogue, swizzle, 'along_n' if raster == 'along_m' else 'along_m')]
    for i in (index-1, index+1):
        if 0 <= i < len(widths):
            result.append((epilogue, widths[i], raster))
    return result


def auto_acceptance_plan(historical, pure_gemm, run_root):
    """Full fixed-GEMM matrix, with one PRESELECTED manual budget plus Auto.

    This builds benchmark arguments only; it is not a runtime policy or a new
    search. Confirmed historical layouts stay explicit inputs. A point without
    a confirmed fused baseline uses its independent GEMM winner and leaves the
    manual comparison empty. In particular an old OOM does not delete a target.
    """
    from summarize_sm103_mxfp8_fused import _check_confirmation, _comparison_key
    pure = {(r['m'], r['n'], r['k']): r for r in pure_gemm['rows']}
    tasks, seen = [], set()
    for row in historical['rows']:
        key = _comparison_key(row)
        if key in seen:
            raise ValueError('Duplicate Auto acceptance target')
        seen.add(key)
        confirmed = row.get('confirmation')
        if confirmed:
            _check_confirmation(row, confirmed)
            c = confirmed['configuration']
            config = confirmed['epilogue_n'], int(c['max_swizzle_size']), c['raster']
            manual = int(c['comm_sm'])
            run_id = confirmed['run_id']
        else:
            config = gemm_config(pure[(row['m'], row['n'], row['k'])])
            manual, run_id = None, row['run_id']
        if not re.fullmatch(r'[0-9]{8}-[0-9]{6}-[a-z0-9]+', run_id):
            raise ValueError('Invalid source run ID')
        job = json.loads((Path(run_root) / run_id / 'job.json').read_text())
        # Labels in historical logical aliases can mention another sequence;
        # actual geometry, not a parsed label, is the authoritative join key.
        if (job['world'] != row['world'] or job['global_seq'] != row['global_seq'] or
                job['hidden'] != row['k'] or
                (job['q_heads'] + 2*job['kv_heads']) * job['head_dim'] != row['n']):
            raise ValueError('Historical job/acceptance geometry mismatch')
        tasks.append(dict(model=row['pure_reference_id'], m=row['m'], n=row['n'], k=row['k'],
            **{f:job[f] for f in ('world', 'global_seq', 'hidden', 'q_heads', 'kv_heads', 'head_dim')},
            epilogue_n=config[0], max_swizzle_size=config[1], raster=config[2], manual_comm=manual,
            gemm_source='historical_confirmed_layout' if confirmed else 'independent_pure_gemm_winner'))
    return sorted(tasks, key=lambda t:(t['n'], t['k'], t['world'], t['global_seq']))


def render_summary(report):
    """Render only paired results/configuration, not the tuning history."""
    names = {(4096,2048):'QwenDense', (10240,8192):'Qwen72 / Llama70',
             (18432,16384):'Llama405B', (43008,14336):'BLOOM176B',
             (36864,7168):'Kimi K3 QKV-only', (9216,4096):'Qwen3 235B'}
    valid=[r for r in report['rows'] if r.get('confirmation')]
    gm=math.expm1(sum(math.log1p(r['confirmed_gain']) for r in valid)/len(valid)) if valid else 0
    lines=['# MXFP8 QKVProj 手工联合调优', '',
           f"独立复测 {len(valid)} 点；相对同 binary 旧配置，几何平均提升 {gm:.2%}。", '',
           '单位：每卡 GEMM-equivalent PFLOPS。Graph 10+50，双随机 payload 全数值/路由校验。',
           '边界：已量化 A + 每次调用内部 BF16 W 量化 + MXFP8 GEMM + BF16 A2A。',
           '共同 GEMM 配置：M128/N256/K128、E32、CUTLASS auto stages；表内显式给出布局与通信 CTA。',
           '旧配置：E64、AlongM/swizzle1、16 通信 CTA。复测值不与搜索最小值择优拼接。', '',
           '| 模型 | 序列 | CP | 旧配置 P | 手工复测 P | 提升 | 纯 cuBLASLt P | 占纯 GEMM | 布局 / swizzle / 通信CTA |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in sorted(report['rows'],key=lambda r:(r['n'],r['k'],r['world'],r['global_seq'])):
        prefix=f"| {names.get((r['n'],r['k']),r['pure_reference_id'])} | {r['global_seq']//1024}K | {r['world']} |"
        if r.get('status')=='resource_skipped':
            lines.append(prefix+f" — | — | 显存不足，未启动 | {r['pure_pflops']:.3f} | — | — |")
            continue
        c=r.get('confirmation')
        if not c:
            lines.append(prefix+' — | — | 待复测 | — | — | — |')
            continue
        config=c['configuration']; direction='M' if config['raster']=='along_m' else 'N'
        gain=r['confirmed_gain']; color='🟢' if gain>=0 else '🔴'
        lines.append(prefix+f" {r['baseline']['pflops_per_rank']:.3f} | {c['pflops_per_rank']:.3f} | "
                     f"{color} {gain:+.2%} | {r['pure_pflops']:.3f} | {c['pflops_per_rank']/r['pure_pflops']:.1%} | "
                     f"{direction} / {config['max_swizzle_size']} / {config['comm_sm']} |")
    lines += ['', '说明：', '',
              '- 这是本次有限候选中的手工配置，不是全局最优声明，也未实现运行时自动选优。',
              '- 前15点完整网格/邻域；后续按用户提速要求复用 N/K 族有效布局，保留逐点测量和校验。',
              '- 纯 cuBLASLt 是之前的满148 SM参考；采样器与融合不同，百分比不是硬件理论 MFU。',
              '- Qwen72/Llama70 为同形状复用；Kimi 仅 QKV，不代表完整 KDA；Qwen3 CP8路由未适配，不在原33点矩阵内。',
              '- BLOOM CP4/512K预检：GPU0空闲39457 MiB，预计需48745.9 MiB，按要求跳过；未停其他任务。',
              '- 每点精确参数、原始样本、source/binary/environment哈希和run ID见同名 JSON；原始证据在 l20d 对应run归档。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--current', type=Path, required=True)
    parser.add_argument('--gemm', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--node', choices=('09', '0a'), default='09')
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--fast', action='store_true', help='Reuse measured family winners as a bounded joint candidate pool')
    parser.add_argument('--confirm', action='store_true', help='Independently remeasure the fixed winners; no new search')
    args = parser.parse_args()
    args.resume = args.resume or args.confirm
    current = json.loads(args.current.read_text())
    pure = json.loads(args.gemm.read_text())
    winners = {(r['m'], r['n'], r['k']): r for r in pure['rows']}
    rows = current['fusion_rows']
    for row in rows:
        gemm_config(winners[(row['m'],row['n'],row['k'])])
    print(f'PLAN {len(rows)} fused shapes: '+('fixed-winner confirmation only' if args.confirm else
          'paired control + CTA grid + joint raster/swizzle/CTA neighbors'), flush=True)
    if not args.execute:
        return
    if args.output.exists() and not args.resume:
        raise ValueError('Output exists; inspect completed work before explicitly continuing')
    report = dict(schema='sm103_mxfp8_fused_tuning_v1', rows=[],
                  reference=str(args.current), pure_winners=str(args.gemm),
                  measurement='Graph10+50, two payloads, full validation, paired baseline',
                  scope='pure-GEMM seed; joint raster/swizzle/CTA grid and local neighbors')
    if args.resume:
        report = json.loads(args.output.read_text())
        if report['reference'] != str(args.current) or report['pure_winners'] != str(args.gemm):
            raise ValueError('Resume input provenance changed')
    if args.confirm and (not report.get('complete') or len(report['rows'])!=len(rows)):
        raise ValueError('Finish the matrix search before confirming winners')
    run_root = args.current.parents[2]/'l20d'
    cached_jobs = []
    for path in run_root.glob('*/job.json'):
        job = json.loads(path.read_text())
        if job.get('experiment','').startswith('mxfp8-fused-tune-'):
            cached_jobs.append((path.parent,job))

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(args.output)

    def run(row, label, epilogue, swizzle, raster, budgets):
        old = json.loads((Path(row['raw_evidence']).parent.parent/'job.json').read_text())
        experiment = f"mxfp8-fused-tune-{row['n']}-{row['k']}-s{row['global_seq']}-cp{row['world']}-{label}"
        cmd = [sys.executable, str(ROOT/'scripts/l20d.py'), 'run', 'fused-smoke',
               '--node',args.node,'--workspace',args.workspace,'--mxfp8','--mpi',
               '--fused-direction','qkv','--fused-launch','graph','--input-generator','gpu_philox',
               '--qkv-policy-list','m128n256','--mxfp8-weight-preparation','comm',
               '--timeout','900','--timeout-seconds','120',
               '--experiment',experiment,'--mxfp8-epilogue-n',str(epilogue),
               '--max-swizzle-size',str(swizzle),'--qkv-raster',raster,
               '--comm-sm-list',','.join(map(str,budgets))]
        for key in ('world','global_seq','hidden','q_heads','kv_heads','head_dim'):
            cmd += ['--'+key.replace('_','-'), str(old[key])]
        expected = {key:old[key] for key in ('world','global_seq','hidden','q_heads','kv_heads','head_dim')}
        expected.update(experiment=experiment,mxfp8_epilogue_n=epilogue,max_swizzle_size=swizzle,
                        qkv_raster=raster,comm_sm_list=','.join(map(str,budgets)),workspace=args.workspace,node=args.node)
        for directory,job in sorted(cached_jobs,key=lambda item:item[0].name):
            if label == 'confirm' and job.get('experiment') != experiment:
                continue  # A search sample is not an independent confirmation.
            previous_budgets=[int(v) for v in (job.get('comm_sm_list') or str(job.get('comm_sm'))).split(',')]
            if (all(job.get(k)==v for k,v in expected.items() if k not in ('experiment','comm_sm_list'))
                    and set(budgets)<=set(previous_budgets) and (directory/'fetched.json').exists()):
                if json.loads((directory/'fetched.json').read_text()).get('state') != 'succeeded':
                    continue
                result = [audit_run(directory,previous_budgets.index(c)+1) for c in budgets]
                anchor = report['rows'][0]['baseline'] if report['rows'] else result[0]
                if all(r['binary_sha256']==anchor['binary_sha256'] and
                       r['environment_fingerprint']==anchor['environment_fingerprint'] for r in result):
                    print('REUSE',directory.name,label,flush=True)
                    return result
        process = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        match = re.search(r'"run_id": "([0-9]{8}-[0-9]{6}-[a-z0-9]+)"', process.stdout)
        if not match:
            raise RuntimeError(process.stdout[-2000:])
        directory = args.current.parents[2]/'l20d'/match[1]
        # The canonical midfile root is shared by controller runs and tables.
        if not directory.exists():
            directory = Path('/Users/admin/workspace/fuse_midfile/l20d')/match[1]
        if process.returncode:
            print('FAILED', match[1], process.stdout[-1200:], flush=True)
            raise RuntimeError('Controller failure; preserve evidence and diagnose before retry')
        result = [audit_run(directory, i+1) for i in range(len(budgets))]
        if report['rows']:
            anchor=report['rows'][0]['baseline']
            if any(r['binary_sha256']!=anchor['binary_sha256'] or
                   r['environment_fingerprint']!=anchor['environment_fingerprint'] for r in result):
                raise ValueError('Measured binary/environment changed across the comparison')
        for r in result:
            r.pop('telemetry', None)
            print(f"  {label} comm={r['configuration']['comm_sm']} E{epilogue} {raster}/sw{swizzle} "
                  f"p50={r['p50_ms']:.6f}ms p95={r['p95_ms']:.6f}ms {r['pflops_per_rank']:.3f}P", flush=True)
        return result

    for index, row in enumerate(rows, 1):
        if index <= len(report['rows']):
            previous=report['rows'][index-1]
            if any(previous[k]!=row[k] for k in ('m','n','k','world','global_seq')):
                raise ValueError('Resume matrix order changed')
            if not args.confirm or previous.get('confirmation') or previous.get('status')=='resource_skipped':
                continue
        print(f"SHAPE {index}/{len(rows)} N={row['n']} K={row['k']} S={row['global_seq']} CP={row['world']}", flush=True)
        config = gemm_config(winners[(row['m'],row['n'],row['k'])])
        if args.confirm:
            entry=report['rows'][index-1]; winner=entry['winner']; c=winner['configuration']
            confirmed=run(row,'confirm',winner['epilogue_n'],int(c['max_swizzle_size']),
                          c['raster'],[int(c['comm_sm'])])[0]
            entry['confirmation']=confirmed
            entry['confirmed_gain']=entry['baseline']['p50_ms']/confirmed['p50_ms']-1
            save()
            print(f"CONFIRMED {index}/{len(rows)} {confirmed['pflops_per_rank']:.3f}P "
                  f"({entry['confirmed_gain']:+.2%} vs control)",flush=True)
            continue
        baseline = run(row,'baseline',64,1,'along_m',[16])[0]
        if args.fast:
            family = sorted((r for r in report['rows'] if r.get('winner') and (r['n'],r['k'])==(row['n'],row['k'])),
                            key=lambda r:abs(math.log2(r['m']/row['m']))+.25*(r['world']!=row['world']))
            layouts = [config]
            for r in family:
                w=r['winner']; c=w['configuration']
                alternative=(w['epilogue_n'],int(c['max_swizzle_size']),c['raster'])
                if alternative not in layouts: layouts.append(alternative)
                if len(layouts)==2: break
            candidates=[]
            for i,layout in enumerate(layouts):
                matching=[r for r in family if (r['winner']['epilogue_n'],
                    int(r['winner']['configuration']['max_swizzle_size']),r['winner']['configuration']['raster'])==layout]
                center=int((matching or family)[0]['winner']['configuration']['comm_sm']) if family else 24
                budgets=sorted({max(8,center-8),center,min(80,center+8)})
                candidates+=run(row,f'pool{i}',*layout,budgets)
            best=min(candidates,key=lambda x:x['p50_ms'])
            if best['p50_ms'] > baseline['p50_ms']:
                c=best['configuration']
                candidates+=run(row,'pool-expand',best['epilogue_n'],int(c['max_swizzle_size']),
                                c['raster'],[8,16,24,32,48,64])
                best=min(candidates,key=lambda x:x['p50_ms'])
            entry=dict(m=row['m'],n=row['n'],k=row['k'],world=row['world'],global_seq=row['global_seq'],
                       pure_reference_id=row['pure_reference_id'],pure_pflops=row['pure_pflops'],
                       baseline=baseline,winner=best,candidates=candidates,
                       search_mode='family_seeded_joint_pool',gain=baseline['p50_ms']/best['p50_ms']-1)
            report['rows'].append(entry); save()
            print(f"DONE {index}/{len(rows)} {baseline['pflops_per_rank']:.3f} -> {best['pflops_per_rank']:.3f}P "
                  f"({entry['gain']:+.2%}), comm={best['configuration']['comm_sm']}",flush=True)
            continue
        candidates = run(row,'grid',*config,[8,16,24,32,48,64])
        best = min(candidates,key=lambda x:x['p50_ms'])
        center = int(best['configuration']['comm_sm'])
        # BF16's GEMM-driven contract is reused by the MXFP8 binding:
        # each layout changes the shared resolved producer order, quantization
        # panel order and route-copy dependency inverse together. A budget
        # changes compute-grid stride and communication slots together. Probe
        # a budget neighborhood for EACH layout, not only at the seed's winner.
        layout_budgets = sorted({max(4,center-8),center,min(80,center+8)})
        for i, alternative in enumerate(layout_neighbors(config)):
            candidates += run(row,f'layout{i}',*alternative,layout_budgets)
        best = min(candidates,key=lambda x:x['p50_ms'])
        config = (best['epilogue_n'], int(best['configuration']['max_swizzle_size']),
                  best['configuration']['raster'])
        center = int(best['configuration']['comm_sm'])
        sampled = {int(r['configuration']['comm_sm']) for r in candidates
                   if (r['epilogue_n'],int(r['configuration']['max_swizzle_size']),r['configuration']['raster']) == config}
        neighbors = sorted({max(4,center-4),center+4} - sampled)
        if center >= 64 and 80 not in sampled:
            neighbors += [80]
        if neighbors:
            candidates += run(row,'neighbors',*config,neighbors)
        best = min(candidates,key=lambda x:x['p50_ms'])
        # Offline selection is not a learned runtime model. Keep regressions
        # visible; never substitute the baseline and claim the new path improved.
        entry = dict(m=row['m'],n=row['n'],k=row['k'],world=row['world'],global_seq=row['global_seq'],
                     pure_reference_id=row['pure_reference_id'],pure_pflops=row['pure_pflops'],
                     baseline=baseline,winner=best,candidates=candidates,
                     gain=baseline['p50_ms']/best['p50_ms']-1)
        report['rows'].append(entry)
        save()
        print(f"DONE {index}/{len(rows)} {baseline['pflops_per_rank']:.3f} -> {best['pflops_per_rank']:.3f}P "
              f"({entry['gain']:+.2%}), comm={best['configuration']['comm_sm']}", flush=True)
    report['complete'] = True
    if args.confirm:
        report['confirmed_complete']=True
    save()


if __name__ == '__main__':
    main()
