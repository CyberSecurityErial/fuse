"""Render measured B300 transport curves using only Python's standard library.

No fitted crossover or hardware-latency claim: preserve endpoint, concurrency,
timing overhead, actual observed points, validation and source receipt.
"""
import argparse
import csv
import html
import json
import math
from pathlib import Path
import shutil


def render(rows):
    width, height = 1440, 900
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<style>text{font-family:Arial,sans-serif;fill:#243347;font-size:13px} .title{font-size:22px;font-weight:600}</style>']
    def label(x, y, value, extra=''):
        svg.append(f'<text x="{x}" y="{y}" {extra}>{html.escape(value)}</text>')
    label(28,35,'B300: LD/ST vs cp.async vs TMA — remote GPU1 → local GPU0', 'class="title"')
    label(28,60,'Warm cyclic buffers; no GEMM/quantization. 10 warmups + 50 samples × 32 transfers. Runtime SM103; reported name may be L20D.')
    colors = {'ldst':'#2864dc','cpasync':'#db7a12','tma':'#119267','cpasync_tma':'#9847be'}
    panels = [('g2s','latency_p50_us','G2S service latency (us)'),
              ('g2g','latency_p50_us','Complete G2G copy latency (us)'),
              ('g2g','aggregate_gbps','G2G aggregate useful bandwidth (GB/s)')]
    for row_idx,ctas in enumerate((1,20)):
        warps=1 if ctas==1 else 4
        for col,(endpoint,metric,title) in enumerate(panels):
            x0,y0=70+col*470,155+row_idx*340
            w,h=385,235
            data=[r for r in rows if r['ctas']==ctas and r['endpoint']==endpoint and r['bytes']>0]
            xmin,xmax=min(r['bytes'] for r in data),max(r['bytes'] for r in data)
            top=max(r['latency_p95_us'] if metric=='latency_p50_us' else r[metric] for r in data)*1.08
            bottom=min(r[metric] for r in data)*.85 if metric=='latency_p50_us' else 0
            def x(v): return x0+math.log(v/xmin)/math.log(xmax/xmin)*w
            def y(v):
                fraction=math.log(v/bottom)/math.log(top/bottom) if bottom else v/top
                return y0+h-fraction*h
            label(x0,y0-34,f'{ctas} CTA × {warps} warp/CTA | {title}')
            ticks=[v for v in (.25,.5,1,2,4,8,16,32,64,128,256) if bottom<=v<=top] if bottom else [top*j/5 for j in range(6)]
            for v in ticks:
                svg.append(f'<path d="M{x0},{y(v)}h{w}" stroke="#e3e8f0"/>')
                label(x0-8,y(v)+4,f'{v:.2f}' if top<20 else f'{v:.0f}', 'text-anchor="end"')
            for v in (128,512,2048,8192,32768,131072):
                if v>xmax: continue
                svg.append(f'<path d="M{x(v)},{y0}v{h}" stroke="#edf0f5"/>')
                label(x(v),y0+h+22,f'{v/1024:g}K','text-anchor="middle"')
            label(x0+w/2,y0+h+43,'Bytes per warp transfer (log scale)','text-anchor="middle"')
            for method,color in colors.items():
                series=sorted((r for r in data if r['method']==method),key=lambda r:r['bytes'])
                if not series: continue
                for field,dash in ((metric,''),('latency_p95_us','stroke-dasharray="3 4"')) if metric=='latency_p50_us' else ((metric,''),):
                    points=' '.join(f'{x(r["bytes"]):.2f},{y(r[field]):.2f}' for r in series)
                    svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.8" {dash}/>')
                for r in series:
                    fill='white' if abs(r['half_drift'])>.05 else color
                    payload=html.escape(json.dumps(r),quote=True)
                    tip=f'{method} | {r["bytes"]} B | p50 {r["latency_p50_us"]:.3f} us | p95 {r["latency_p95_us"]:.3f} us'
                    svg.append(f'<circle data-point="{payload}" cx="{x(r["bytes"]):.2f}" cy="{y(r[metric]):.2f}" r="2.5" fill="{fill}" stroke="{color}"><title>{html.escape(tip)}</title></circle>')
    for i,(method,color) in enumerate(colors.items()):
        x0=60+i*190
        svg.append(f'<path d="M{x0},825h30" stroke="{color}" stroke-width="3"/>')
        label(x0+40,830,method)
    label(825,830,'Log latency. Solid: p50; dashed: p95; hollow: drift >5%.')
    label(28,862,'Latency includes issue/wait/warp join. G2G: LD/ST direct; cp.async → SMEM → stores; TMA G2S → TMA S2G.')
    label(28,884,'Bandwidth uses complete G2G batch time, including timer/loop/kernel overhead. These are two-GPU curves, not 8-GPU A2A bandwidth.')
    svg.append('</svg>')
    return '\n'.join(svg)


def interactive(svg):
    """Self-contained hover inspection: no CDN, server, or Python dependency."""
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>B300 搬运时延与带宽</title>
<style>
body{margin:0;background:#eef2f7;color:#243347;font:15px system-ui,sans-serif}
main{max-width:1600px;margin:20px auto;padding:0 18px}
.hint{margin:12px 0;line-height:1.8}a{color:#2864dc}
.chart{background:white;border-radius:12px;overflow:auto;box-shadow:0 2px 12px #17233a14}
svg{width:100%;height:auto;display:block;min-width:1000px}
#tooltip{position:fixed;display:none;z-index:3;pointer-events:none;background:#17243a;
color:white;padding:14px 18px;border-radius:9px;line-height:1.7;box-shadow:0 4px 18px #0004;min-width:250px}
#tooltip b{color:#91d7ff}#tooltip small{color:#c1cee0}
</style><main><div class="hint"><strong>B300 实测交互图</strong> — 鼠标靠近采样点查看数值；点击固定，再点击解除。
<br>两卡链路，非八卡A2A；单笔大小≠序列长度。无外部依赖，可离线打开。
<a href="README.md">计时口径与时延表</a> · <a href="transport.csv" download>CSV</a>
<br>LD/ST 是此微测的向量化实现，不代表所有 LD/ST 实现的性能上限；“启动固有延迟”不能由这项测量严格分离。</div>
<div class="chart">''' + svg + '''</div></main><div id="tooltip" role="status" aria-live="polite"></div>
<script>
const svg=document.querySelector('svg'), tip=document.querySelector('#tooltip');
const points=[...svg.querySelectorAll('circle[data-point]')].map(node=>({node,
 x:+node.getAttribute('cx'),y:+node.getAttribute('cy'),data:JSON.parse(node.dataset.point)}));
let pinned=false,active=null;
function hide(){tip.style.display='none';if(active)active.node.setAttribute('r',2.5);active=null;}
document.querySelector('.chart').addEventListener('pointermove',event=>{
 if(pinned)return;
 const p=svg.createSVGPoint();p.x=event.clientX;p.y=event.clientY;
 const local=p.matrixTransform(svg.getScreenCTM().inverse());
 let best=null,distance=12*12;
 for(const point of points){const d=(point.x-local.x)**2+(point.y-local.y)**2;if(d<distance){best=point;distance=d;}}
 if(!best){hide();return;}
 if(active)active.node.setAttribute('r',2.5);active=best;best.node.setAttribute('r',5);
 const r=best.data,fmt=v=>Number(v).toFixed(3);
 const endpoint=r.endpoint==='g2s'?'远端 GMEM → 本地 SMEM':'远端 GMEM → 本地 GMEM';
 tip.innerHTML=`<b>${r.method} · ${r.bytes/1024} KiB</b><br>${r.bytes.toLocaleString()} bytes / warp<br>${endpoint}<br>`+
 `${r.ctas} CTA × ${r.warps_per_cta} warp/CTA<br>p50：<b>${fmt(r.latency_p50_us)} μs</b><br>p95：${fmt(r.latency_p95_us)} μs<br>`+
 (r.endpoint==='g2g'?`整批有效带宽：<b>${fmt(r.aggregate_gbps)} GB/s</b><br>`:'')+
 `单warp等效速率：${fmt(r.per_worker_gbps)} GB/s<br>整批 p50：${fmt(r.batch_p50_us)} μs<br>`+
 `<small>前后半整批时延漂移：${(r.half_drift*100).toFixed(2)}%<br>逐字校验：${r.bitwise_errors===0?'通过':'失败'}</small>`;
 tip.style.display='block';
 tip.style.left=Math.max(4,Math.min(event.clientX+16,innerWidth-tip.offsetWidth-12))+'px';
 tip.style.top=Math.max(4,Math.min(event.clientY+16,innerHeight-tip.offsetHeight-12))+'px';
});
document.querySelector('.chart').addEventListener('pointerleave',()=>{if(!pinned)hide();});
document.querySelector('.chart').addEventListener('click',()=>{if(active)pinned=!pinned;});
document.addEventListener('keydown',e=>{if(e.key==='Escape'){pinned=false;hide();}});
</script></html>'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--run-id',required=True)
    a=p.parse_args()
    rows=list(csv.DictReader(a.csv.open()))
    integer=('ctas','warps_per_cta','bytes','bitwise_errors')
    text=('endpoint','method')
    for row in rows:
        for k in row:
            if k not in text: row[k]=int(row[k]) if k in integer else float(row[k])
        if row['bitwise_errors'] or any(not math.isfinite(v) for v in row.values() if isinstance(v,float)):
            raise ValueError('Invalid transport evidence')
    keys={(r['endpoint'],r['ctas'],r['method'],r['bytes']) for r in rows}
    if len(keys)!=len(rows): raise ValueError('Duplicate transport point')
    for ctas in (1,20):
        sizes={r['bytes'] for r in rows if r['ctas']==ctas}
        if len(sizes)<40 or 0 not in sizes or 32768 not in sizes:
            raise ValueError('Missing dense transport coverage')
        if any((e,ctas,m,b) not in keys for e in ('g2s','g2g') for m in ('ldst','cpasync','tma') for b in sizes):
            raise ValueError('Unpaired transport methods')
        if any(('g2g',ctas,'cpasync_tma',b) not in keys for b in sizes):
            raise ValueError('Missing mixed-path control')
    a.output.mkdir(parents=True,exist_ok=True)
    if a.csv.resolve()!=(a.output/'transport.csv').resolve(): shutil.copy2(a.csv,a.output/'transport.csv')
    svg=render(rows)
    (a.output/'latency-bandwidth.svg').write_text(svg)
    (a.output/'index.html').write_text(interactive(svg))
    lines=['# B300 搬运时延与带宽','',f'证据 run：`{a.run_id}`。',
        '', '[交互HTML：悬停看数值](index.html) · [完整折线图](latency-bandwidth.svg) · [全量数据 CSV](transport.csv)',
        '', '两卡：可见 GPU1 → GPU0；1 CTA×1 warp 与20 CTA×4 warp。随机整数位模式，完整输出逐字校验；10次预热＋50样本，每样本32次搬运。',
        '32个循环地址槽、无强制刷缓存；不含GEMM/量化。CUDA报告SM103，B300身份来自集群说明。',
        '', '“固有启动时延”不能从这项实验严格分离。下面报告空计时基线与128B小消息：后者减去前者仍包含真实搬运、等待和同步，不是纯指令启动时延。',
        '', '| 终点 | CTA×warp | 方式 | 空基线 μs | 128B p50 μs | 扣空基线 μs | 128B p95 μs |',
        '|---|---:|---|---:|---:|---:|---:|']
    by={(r['endpoint'],r['ctas'],r['method'],r['bytes']):r for r in rows}
    for e in ('g2s','g2g'):
        for c in (1,20):
            for m in (('ldst','cpasync','tma') if e=='g2s' else ('ldst','cpasync','tma','cpasync_tma')):
                zero,small=by[e,c,m,0],by[e,c,m,128]
                lines.append(f'| {e} | {c}×{1 if c==1 else 4} | {m} | {zero["latency_p50_us"]:.3f} | {small["latency_p50_us"]:.3f} | {max(0,small["latency_p50_us"]-zero["latency_p50_us"]):.3f} | {small["latency_p95_us"]:.3f} |')
    lines+=['','G2S是同一SMEM终点；G2G是同一本地GMEM终点，但路径不同：LD/ST直达，cp.async经SMEM后普通store，TMA经SMEM后TMA store；cpasync_tma是cp.async加载加TMA写回。',
        '这些是具体软件实现的实测曲线，不是三类指令的硬件极限：本微测LD/ST G2G每lane四个独立128bit读取后写回；G2S为向量load/store。',
        'p50/p95来自warp内globaltimer；聚合带宽用G2G整批CUDA event时间，包含循环/记录/启动开销，不能与单warp服务速率混淆。',
        'G2S最后一个tile在计时区间外存出校验；G2G检查全部32槽。图中空心点表示整批前后半时延漂移超过5%，不得据此拟合精确切换阈值。',
        '曲线可作条带长度的起点参考，不能直接替代八卡融合并发下的测量；SF重排和ready发布不在此图中。','']
    (a.output/'README.md').write_text('\n'.join(lines))
    print(f'rows={len(rows)} sizes={len({r["bytes"] for r in rows})} drift_gt_5pct={sum(abs(r["half_drift"])>.05 for r in rows)} output={a.output}')


if __name__=='__main__': main()
