#!/usr/bin/env python3
"""Render Graph-only views of the canonical CSV (requires reportlab)."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('table', type=Path)
    args = parser.parse_args()
    with args.table.open() as stream:
        rows = [r for r in csv.DictReader(stream) if r['launch'] == 'graph']
    assert rows and all(float(r['p50_ms']) > 0 for r in rows)
    backends = ('te_ub', 'cublaslt_nccl', 'cublaslt_gemm')
    configs, groups = {}, defaultdict(dict)
    for row in rows:
        config = json.loads(row['config_json'])
        if row['backend'] == 'cublaslt_gemm':
            config = {k: config[k] for k in ('algorithm', 'beta', 'math_sms',
                'workspace_capacity', 'library_sha256', 'replay')}
            full = json.loads(row['config_json'])
            config['selected'] = [{k: c[k] for k in ('algorithm', 'tile', 'split_k', 'workspace_bytes')}
                                  for c in full['selected_candidate_records']]
        identity = json.dumps(dict(backend=row['backend'], config=config), sort_keys=True)
        configs.setdefault(identity, f'C{len(configs)+1:03d}')
        row['config_id'] = configs[identity]
        key = (row['direction'], row['model'], int(row['cp']), int(row['seq']), row['node'])
        assert row['backend'] not in groups[key], 'Ambiguous measurement domain'
        groups[key][row['backend']] = row
    assert all(set(g) == set(backends) for g in groups.values())
    styles = getSampleStyleSheet()
    styles['Normal'].fontSize = 8
    styles['Normal'].leading = 11
    styles['Normal'].wordWrap = 'CJK'
    story, md = [], ['# SM103 BF16 baseline archive — Graph only', '']
    intro = ('Graph only | BF16 | 10 warmups + 50 samples | PFLOPS per GPU. '
             'TEUB and cuBLASLt+NCCL include communication (distributed max-rank timing). '
             'cuBLASLt GEMM is a single-GPU compute-only reference, NOT classic cuBLAS. '
             'Best observed candidates, not global optima or independent winner retests. '
             'Historical archive; no new GPU measurements were performed for this PDF.')
    story += [Paragraph('SM103 BF16 — Graph baseline archive', styles['Title']),
              Paragraph(intro, styles['Normal']), Spacer(1, 12)]
    md += [intro, '', f'{len(rows)} records / {len(groups)} shape-CP groups.', '']
    story += [Paragraph(f'{len(rows)} records / {len(groups)} shape-CP groups. '
                        'Config IDs are defined in the appendix. Full provenance remains in table.csv.', styles['Normal']),
              PageBreak()]
    header = ['CP', 'S', 'Node', 'M / N / K', 'TEUB\nms / P', 'Config',
              'Lt+NCCL\nms / P', 'Config', 'Lt GEMM\nms / P', 'Config']
    for direction, model in sorted({(k[0], k[1]) for k in groups}):
        title = f'{direction.upper()} — {model}'
        story += [Paragraph(title, styles['Heading1'])]
        note = 'QKV Projection -> A2A' if direction == 'qkv' else 'A2A -> OProj (causal_dual_chunk_v1)'
        story += [Paragraph(note, styles['Normal']), Spacer(1, 10)]
        md += ['## '+title, '', note, '', '| '+' | '.join(h.replace('\n', ' ') for h in header)+' |',
               '|'+ '|'.join(['---']*len(header))+'|']
        data = [header]
        for key, group in sorted(groups.items()):
            if key[:2] != (direction, model):
                continue
            row = group['te_ub']
            values = [str(key[2]), str(key[3]), key[4], '/'.join(row[k] for k in ('m', 'n', 'k'))]
            for backend in backends:
                r = group[backend]
                values += [f"{float(r['p50_ms']):.4f} / {float(r['pflops_per_gpu']):.3f}", r['config_id']]
            data.append(values)
            md.append('| '+' | '.join(values)+' |')
        table = Table(data, colWidths=[25, 48, 33, 133, 95, 40, 95, 40, 95, 40], repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#17324d')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8), ('LEADING', (0, 0), (-1, -1), 11),
            ('TOPPADDING', (0, 0), (-1, -1), 7), ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#edf2f6')]),
            ('GRID', (0, 0), (-1, -1), .3, colors.lightgrey)]))
        story += [table, Spacer(1, 12), Paragraph('Each value is p50 milliseconds / PFLOPS per GPU. '
            'p95 and source hashes are available in the canonical CSV. Node 09 and node 0a are not interchangeable.', styles['Normal']), PageBreak()]
        md += ['']
    story += [Paragraph('Configuration appendix', styles['Heading1'])]
    md += ['## Configuration appendix', '']
    for identity, cid in configs.items():
        description = json.loads(identity)
        text = json.dumps(description['config'], sort_keys=True)
        story += [Paragraph(f'{cid} — {description["backend"]}', styles['Heading3']),
                  Paragraph(escape(text), styles['Normal']), Spacer(1, 6)]
        md += [f'### {cid} — {description["backend"]}', '', '```json',
               json.dumps(description['config'], indent=2, sort_keys=True), '```', '']
    caveat = ('cuBLASLt algorithm metadata are library-version-specific. Historical records do not '
              'contain a complete serialized algorithm descriptor; exact replay may require retuning '
              'with the recorded library. Config IDs omit search history and candidate timing.')
    story += [Spacer(1, 10), Paragraph(caveat, styles['Normal'])]
    md += [caveat, '']
    target = args.table.parent / 'graph.pdf'
    def footer(canvas, doc):
        canvas.setFont('Helvetica', 8)
        canvas.drawString(30, 20, 'SM103 BF16 | Graph only | Source: table.csv')
        canvas.drawRightString(810, 20, str(doc.page))
    SimpleDocTemplate(str(target), pagesize=landscape(A4), leftMargin=30, rightMargin=30,
                      topMargin=30, bottomMargin=35).build(story, onFirstPage=footer, onLaterPages=footer)
    args.table.with_name('graph.md').write_text('\n'.join(md))
    print(json.dumps({'pdf': str(target), 'records': len(rows), 'configs': len(configs)}))


if __name__ == '__main__':
    main()
