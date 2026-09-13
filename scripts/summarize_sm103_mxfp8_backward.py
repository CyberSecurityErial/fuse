"""Read-only audit of complete MXFP8 projection B+W Graph measurements.

Reuses archived source/build/environment verification, not forward performance
parsing. Every rank's own stream, both payloads, full checks, warmup convergence
and all accepted/rejected raw samples are required. Does not write a winner.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

import l20d
import summarize_sm103_fused as sf

KINDS={'backward_device','backward_config','backward_input','backward_validation',
       'backward_sample','backward_warmup','backward_rejected','backward_verified','backward_complete'}

def parse(raw):
    rows=[]
    for line in raw.decode().splitlines():
        if not line.strip():
            continue
        if line.startswith('device,'):
            line='backward_device '+line.removeprefix('device,').replace(',',' ')
        fields=line.split()
        sf.require(fields[0] in KINDS, 'Unexpected backward log record: '+line[:120])
        row={'kind':fields[0]}
        for field in fields[1:]:
            sf.require('=' in field, 'Malformed backward field')
            key,value=field.split('=',1)
            sf.require(key not in row, 'Duplicate backward field')
            sf.require(value.lower() not in ('nan','inf','-inf','+inf','infinity'), 'Nonfinite backward evidence')
            row[key]=value
        rows.append(row)
    return rows


def audit_mpi(job,records,data,attempt):
    world=job['world']
    runtime=records[f'mpi-runtime-attempt{attempt}.json']
    manifest=records[f'mpi-logs-attempt{attempt}.json']
    for key,value in dict(schema='sm103_mpi_runtime_v1',node=job['node'],world=world,
            process_layout='mpi_one_process_per_gpu',host_launch='mpi_process',
            launch='graph',collector='mpi_graph_rank_events_v1',boundary='mpi_graph_maxrank_cudaevent',
            graph_epoch_mode=sf.GRAPH_EPOCH_MODE).items():
        sf.require(runtime.get(key)==value,'MPI metadata mismatch: '+key)
    sf.require(runtime['overrides']==records['environment.json']['mpi_toolchain']['overrides'] and
               runtime['overrides'].get('UCX_TLS')=='sm,self','MPI toolchain mismatch')
    argv=runtime['argv']
    prefix=str(l20d.workspace_path(job['workspace'])/'toolchain/mpich-5.0.1.post1/bin/mpiexec')
    sf.require(argv[:5]==[prefix,'-launcher','fork','-n',str(world)] and
               records['fused-build.json']['binary'] in argv,'MPI executable mismatch')
    merged=data[f'attempt{attempt}.log']
    sf.require(manifest['schema']=='sm103_mpi_rank_logs_v1' and manifest['complete'] is True and
               manifest['merged_sha256']==sf.digest(merged) and
               manifest['ordering']=='rank_then_stream_not_global_chronological' and
               manifest['collector']=='mpi_graph_rank_events_v1','MPI merge contract mismatch')
    entries=manifest['ranks']
    sf.require([(r['rank'],r['stream']) for r in entries]==
        [(r,s) for r in range(world) for s in ('stdout','stderr')],'Incomplete MPI streams')
    previous=0
    for entry in entries:
        rank,stream=entry['rank'],entry['stream']
        name=f'mpi-attempt{attempt}-rank-{rank}.{stream}.log'
        raw=data[name];begin,end=entry['merged_begin'],entry['merged_end']
        sf.require(entry['path']==name and entry['present'] and entry['rank_started'] and
                   entry['sha256']==sf.digest(raw) and entry['bytes']==len(raw) and
                   previous<=begin<=end<=len(merged) and merged[begin:end]==raw,
                   'MPI rank bytes mismatch')
        sf.require(b'backward_' not in merged[previous:begin], 'Unowned backward evidence')
        if stream=='stderr':
            sf.require(not raw.strip(),'MPI rank stderr is not empty')
        else:
            rows=parse(raw)
            devices=[r for r in rows if r['kind']=='backward_device']
            sf.require(len(devices)==1 and int(devices[0]['rank'])==rank,'Native rank startup mismatch')
            for row in rows:
                if row['kind'] in ('backward_device','backward_input','backward_validation'):
                    sf.require(int(row['rank'])==rank,'Wrong native record owner')
                else:
                    sf.require(rank==0,'Nonroot emitted root-only record')
        previous=end
    sf.require(b'backward_' not in merged[previous:],'Unowned backward tail')


def audit_run(directory,component='full'):
    directory=Path(directory).resolve()
    request=sf.json_bytes(sf.read_bytes(directory/'job.json',directory))
    sf.require(request.get('mxfp8') and request.get('backward') and request.get('mpi') and
               request.get('fused_direction') in ('oproj','qkv') and request.get('fused_launch')=='graph' and
               not request.get('profile') and not request.get('quick'),'Not complete MXFP8 backward Graph')
    sf.require(component in ('full','data','weight','weight_compute','data_compute','data_gemm') and
               (component=='full' or request.get('calibrate')),'Backward component requires calibration')
    old_workspace,old_audit=str(l20d.WORKSPACE),sf.audit_mpi_receipts
    try:
        l20d.configure_workspace(request['workspace']);sf.audit_mpi_receipts=audit_mpi
        job,receipts,data,evidence=sf.read_receipts(directory)
    finally:
        l20d.configure_workspace(old_workspace);sf.audit_mpi_receipts=old_audit
    attempt=receipts['status.json']['attempt']
    rows=[]
    for rank in range(job['world']):rows+=parse(data[f'mpi-attempt{attempt}-rank-{rank}.stdout.log'])
    def selected(kind,**fields):
        timed=kind in ('backward_validation','backward_warmup','backward_sample','backward_verified','backward_rejected')
        return [r for r in rows if r['kind']==kind and (not timed or r.get('component','full')==component)
                and all(r.get(k)==str(v) for k,v in fields.items())]
    configs=selected('backward_config');sf.require(len(configs)==1,'Missing/duplicate backward configuration')
    c=configs[0];shape=l20d.fused_geometry(job)
    qkv=job['fused_direction']=='qkv'
    if component=='data_compute':
        sf.require(qkv and c.get('data_compute_reference')=='1',
                   'Missing exact prepared dX diagnostic boundary')
    if component=='data_gemm':
        sf.require(qkv and c.get('data_gemm_reference')=='1',
                   'Missing same-rank bare dX diagnostic boundary')
    raster=job[job['fused_direction']+'_raster']
    width=shape['projection_width'] if qkv else shape['q_width']
    sf.require(c.get('op')==job['fused_direction']+'_mxfp8','Wrong backward operator')
    if qkv:
        sf.require(int(c['q_heads'])==shape['q_heads'] and int(c['kv_heads'])==shape['kv_heads'],
                   'Wrong packed QKV head geometry')
    sf.require(int(c.get('calibrate',0))==int(bool(job.get('calibrate'))),'Calibration job/config mismatch')
    if component=='weight_compute':
        sf.require(c.get('weight_compute_reference')=='1','Missing prepared dW diagnostic boundary')
    for key, expected in dict(
            weight_epilogue=job.get('backward_weight_epilogue_n') or (job.get('mxfp8_epilogue_n') or 32),
            weight_swizzle=job.get('backward_weight_swizzle') or job.get('max_swizzle_size',1),
            weight_along_m=int((job.get('backward_weight_raster') or raster)=='along_m')).items():
        fallback={'weight_epilogue':'epilogue','weight_swizzle':'swizzle','weight_along_m':'along_m'}[key]
        sf.require(int(c.get(key,c[fallback]))==expected,'Independent dW config mismatch: '+key)
    for key,value in dict(M=shape['seq_local'],H=shape['hidden'],A=width,world=job['world'],
            comm=job['comm_sm'],epilogue=job.get('mxfp8_epilogue_n') or 32,
            swizzle=job.get('max_swizzle_size',1),along_m=int(raster=='along_m'),
            causal=int(bool(job.get('causal'))),kernels=5).items():
        sf.require(int(c[key])==value,'Backward configuration mismatch: '+key)
    sf.require(c['launch']=='graph' and c['weight_mode']=='immediate' and
               c['timed']==('weight_quant_inverse_QKV_dX_dQKV_quant_X_quant_dW' if qkv else
                           'weight_quant_dA_route_dY_quant_A_quant_dW') and
               c['upstream_dQ_dK_dV_quant' if qkv else 'upstream_dY_quant']=='excluded' and
               c['CP_dW_reduce']=='caller_owned','Wrong backward boundary')
    if qkv:
        sf.require(c.get('original_BF16_route')=='included' and c.get('input_lease')=='all_ranks_until_B_complete',
                   'Missing original-gradient route/input lease boundary')
    m,h,a,world=shape['seq_local'],shape['hidden'],width,job['world']
    flops=4*m*h*a;sf.close(float(c['flops_per_rank']),flops,'Backward FLOPs')
    if component!='full':flops//=2
    payloads=[]
    for gen in (0,1):
        inputs=selected('backward_input',generation=gen)
        tensors=('dQ','dK','dV','W','saved_X') if qkv else ('dY','W','saved_A')
        sf.require(Counter((int(r['rank']),r['tensor']) for r in inputs)==
                   Counter((rank,tensor) for rank in range(world) for tensor in tensors),'Incomplete inputs')
        for r in inputs:
            rank=int(r['rank'])
            counts=({'dQ':m*shape['q_width'],'dK':m*shape['kv_width'],'dV':m*shape['kv_width'],
                     'W':h*a,'saved_X':m*h} if qkv else {'dY':m*h,'W':h*a,'saved_A':m*a})
            seeds=({'dQ':1234+rank,'dK':2234+rank,'dV':3234+rank,'W':5678,'saved_X':9012+rank}
                   if qkv else {'dY':1234+rank,'W':5678,'saved_A':9012+rank})
            count=counts[r['tensor']];seed=seeds[r['tensor']]+gen*100
            sf.require(r['generator']=='gpu_philox' and int(r['seed'])==seed and
                       int(r['count'])==int(r['finite'])==count and 0<int(r['nonzero'])<=count and
                       math.isfinite(float(r['square_sum'])) and float(r['square_sum'])>0,'Invalid Philox payload')
        checks=selected('backward_validation',generation=gen)
        sf.require(Counter((int(r['rank']),r['phase']) for r in checks)==
                   Counter((rank,phase) for rank in range(world) for phase in ('pre','post')),'Missing full pre/post checks')
        for r in checks:
            for output,count in (('B',m*h if qkv else m*a),('W',h*a),('route',m*a)):
                sf.require(int(r[output+'_checked'])==count and int(r[output+'_mismatch'])==0,'Incomplete/failed gradient validation')
        windows=selected('backward_warmup',generation=gen)
        for rank in range(world):
            w=[r for r in windows if int(r['rank'])==rank]
            sf.require(len(w)>=3 and [int(r['window']) for r in w]==list(range(len(w))), 'Incomplete convergence windows')
            total=0
            for r in w:
                sf.require(int(r['calls'])>0 and float(r['ms_per_call'])>0,'Invalid convergence call')
                total+=int(r['calls'])*float(r['ms_per_call'])
                sf.close(total,float(r['accumulated_cuda_ms']),'Accumulated warmup CUDA time')
            last=[float(r['ms_per_call']) for r in w[-3:]]
            sf.require(total>=100 and w[-1]['ready']=='1' and
                       (max(last)-min(last))/sf.percentile(last,.5)<=.0500001,'Rank did not converge')
        samples=selected('backward_sample',generation=gen)
        sf.require(all(r['phase'] in ('initial','sample_cadence','measurement') for r in samples),
                   'Unknown backward sampling phase')
        for phase in ('initial','sample_cadence'):
            sf.require([int(r['index']) for r in samples if r['phase']==phase]==list(range(10)), 'Missing 10-call warmup/cadence')
        for r in samples:
            times=[float(r[f'rank{rank}_ms']) for rank in range(world)]
            sf.require(all(math.isfinite(t) and t>0 for t in times),'Invalid raw rank sample')
            sf.close(max(times),float(r['maxrank_ms']),'Max-rank event sample')
        finals=selected('backward_verified',generation=gen);sf.require(len(finals)==1,'Missing verified payload')
        final=finals[0];chosen=int(final['selected_round']);sf.require(0<=chosen<3,'Invalid round')
        measured=[r for r in samples if r['phase']=='measurement']
        sf.require(len(measured)==50*(chosen+1),'Missing/excess formal samples')
        for round_id in range(chosen+1):
            rr=[r for r in measured if int(r['round'])==round_id]
            sf.require([int(r['index']) for r in rr]==list(range(50)),'Incomplete 50-call round')
            sf.require(all(int(rr[i]['epoch'])==int(rr[0]['epoch'])+i for i in range(50)),'Noncontiguous measured epochs')
            values=[float(r['maxrank_ms']) for r in rr];p50=sf.percentile(values,.5)
            drift=abs(sf.percentile(values[:25],.5)-sf.percentile(values[25:],.5))/p50
            sf.require(drift<=.0500001 if round_id==chosen else drift>.05,'Not first stable measurement round')
        sf.close(p50,float(final['p50_ms']),'Backward p50');sf.close(drift,float(final['half_drift']),'Backward drift')
        sf.close(sf.percentile(values,.95),float(final['p95_ms']),'Backward p95')
        sf.close(flops/(p50*1e12),float(final['pflops']),'Backward throughput')
        sf.require(final['verification']=='pass','Backward payload not verified')
        payloads.append(dict(generation=gen,p50_ms=p50,pflops=flops/(p50*1e12),half_drift=drift))
    complete=selected('backward_complete')
    sf.require(len(complete)==1 and complete[0].get('verification')=='pass' and
               complete[0].get('payloads')=='2' and complete[0].get('boundary')=='immediate_B_W','Missing final complete boundary')
    # Equal-weight the two payload times, never pick the faster random payload.
    p50=sum(x['p50_ms'] for x in payloads)/2
    return dict(run=job['run_id'],source=job['source_id'],binary=receipts['fused-build.json']['binary_sha256'],
        artifact=evidence['artifacts.tar.gz']['sha256'],environment=receipts['environment.json']['fingerprint'],
        configuration=c,component=component,
        boundary={'full':'immediate_B_W_five_kernels','data':('W_quant_inverse_QKV_dX_two_kernels' if qkv else
                  'W_quant_dA_inverse_A2A_two_kernels'),
                  'weight':('dQKV_quant_saved_X_quant_dW_three_kernels' if qkv else
                            'dY_quant_saved_A_quant_dW_three_kernels'),
                  'weight_compute':'prepared_dW_GEMM_one_kernel_no_quantization',
                  'data_compute':'prepared_dX_same_budget_acquire_adapter_no_quantization_or_transport',
                  'data_gemm':'prepared_dX_same_budget_stock_collective_no_adapter_quantization_or_transport'}[component],payloads=payloads,
        p50_ms=p50,pflops=flops/(p50*1e12),verification='full_two_payload_pre_post_numeric_and_route')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('runs',nargs='+',type=Path)
    parser.add_argument('--component',choices=('full','data','weight','weight_compute','data_compute','data_gemm'),default='full')
    args=parser.parse_args();print(json.dumps([audit_run(run,args.component) for run in args.runs],indent=2))
