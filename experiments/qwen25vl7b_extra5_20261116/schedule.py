"""Eight-card supplemental evaluation, sharing headroom with InternVL jobs."""
import csv
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo
import psutil

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
OUT=ROOT/'outputs'/BASE.name
SOURCE=BASE/'source'
VARIANTS=json.loads((OUT/'variants.json').read_text())
DEFS=json.loads((OUT/'extra_benchmarks.json').read_text())['benchmarks']
TASKS=['mmvp','mmbench_en_dev_static','naturalbench','mmtbench_val','mmerealworld']
KEYS={d['task']:d['primary_metric'] for d in DEFS}
FACTORS={d['task']:d['score_to_percent'] for d in DEFS}
PREP=json.loads((OUT/'benchmark_preparation.json').read_text())['tasks']

def read(path,default=None):return json.loads(path.read_text()) if path.exists() else default
def save(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)

def build_jobs():
    jobs=[]
    def job(v,task,smoke=False,dependencies=()):
        key=v['name']+'__'+task+('__smoke' if smoke else '')
        directory=OUT/v['name']/'eval'/(task+('_smoke' if smoke else ''))
        command=[sys.executable,'-u',str(BASE/'run_eval.py'),'--model','qwen2_5_vl','--model_args',
            'pretrained=/data/lkp/models/Qwen2.5-VL-7B-Instruct,torch_dtype=float16,attn_implementation=sdpa,min_pixels=200704,max_pixels=1605632',
            '--tasks',task,'--device','npu','--batch_size','1','--seed','42',
            '--gen_kwargs','temperature=0,do_sample=False,max_new_tokens=256','--bootstrap_iters','1000',
            '--log_samples','--output_path',str(directory),'--verbosity','DEBUG']
        if v['state']:command+=['--quantized_state',v['state'],'--w_bit',str(v['w']),'--a_bit',str(v['a'])]
        if smoke:command+=['--limit','2']
        jobs.append({'key':key,'variant':v['name'],'task':task,'smoke':smoke,'command':command,
            'eval_dir':str(directory),'dependencies':list(dependencies)})
        return key
    gates={}
    for v in [VARIANTS[0],next(v for v in VARIANTS if v['w']==3),next(v for v in VARIANTS if v['a']==8)]:
        gates[(v['w'],v['a'])]=job(v,'mmvp',True)
    for v in VARIANTS:
        for task in TASKS:job(v,task,dependencies=[gates[(v['w'],v['a'])]])
    return jobs

def result(job):
    expected=2 if job['smoke'] else PREP[job['task']]['rows']
    for path in sorted(Path(job['eval_dir']).rglob('*_results.json'),reverse=True):
        data=read(path,{})
        metrics=data.get('results',{}).get(job['task'],{})
        score=metrics.get(KEYS[job['task']])
        if data.get('n-samples',{}).get(job['task'],{}).get('effective')==expected and isinstance(score,(int,float)) and math.isfinite(score):return path,data
    return None

def resources():
    r=subprocess.run(['/data/lkp/envs/ascend-nputop/bin/nputop','--json'],capture_output=True,text=True,timeout=25,check=True)
    value=json.loads(r.stdout);save(OUT/'resource_snapshot.json',value)
    return {int(d['card']):d['metrics']['memory_total']['value']-d['metrics']['memory_used']['value'] for d in value['devices']}

def report(jobs,state):
    counts={s:sum(v['status']==s for v in state.values()) for s in ['pending','running','complete','failed','dependency_failed']}
    save(OUT/'queue_summary.json',counts)
    scores={};secondary=[]
    for job in jobs:
        if job['smoke'] or state[job['key']]['status']!='complete':continue
        found=result(job)
        if not found:continue
        metrics=found[1]['results'][job['task']]
        scores.setdefault(job['variant'],{})[job['task']]=metrics[KEYS[job['task']]]*FACTORS[job['task']]
        for key,value in metrics.items():
            if isinstance(value,(int,float)) and 'stderr' not in key:
                secondary.append({'variant':job['variant'],'task':job['task'],'metric':key,'native_value':value})
    rows=[{'variant':v['name'],**scores.get(v['name'],{})} for v in VARIANTS]
    with (OUT/'summary_extra5.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['variant',*TASKS]);writer.writeheader();writer.writerows(rows)
    with (OUT/'summary_extra_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['variant','task','metric','native_value']);writer.writeheader();writer.writerows(secondary)
    original={}
    for directory in ['lat_awq_general_20261107','lat_awq_general_seed3407_20261111']:
        with (ROOT/'outputs'/directory/'summary.csv').open() as f:
            original.update({row['variant']:row for row in csv.DictReader(f)})
    core=['mmmu_val','ai2d','scienceqa','textvqa_val','seedbench','mmstar','cv_bench','vizwiz_vqa_val']
    with (OUT/'summary_all13.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['variant',*core,*TASKS,'average_8'])
        writer.writeheader()
        for v in VARIANTS:
            prior=original[v['name']]
            writer.writerow({'variant':v['name'],**{k:prior[k] for k in core+['average_8']},**scores.get(v['name'],{})})
    lines=['# Qwen2.5-VL-7B 新增五项补测','',
        f'更新时间：{datetime.now(ZoneInfo("Asia/Shanghai")):%Y-%m-%d %H:%M:%S}（北京时间）',
        '33 组：FP16 + 四种子（42/123/2026/3407）× W3A16/W4A8 × A/B/C/D；共 165 项正式评测，另加 3 项两例入口验证。',
        '卡 0–7 与 InternVL 共卡；每卡至多一个补测进程，准入至少 34 GiB 空闲显存，补测进程 allocator 上限 44%。',
        '共享显存不足的任务保留日志，等待 InternVL 释放该卡后以独占预算重试。',
        f'任务状态：{counts}；正式评测完成 {sum(not j["smoke"] and state[j["key"]]["status"]=="complete" for j in jobs)}/165。','',
        '运行中：']
    lines += [f'- NPU {v["npu"]}：{key}；PID {v["pid"]}；显存预算 {v.get("memory_fraction")}' for key,v in state.items() if v['status']=='running']
    lines+=['','错误与待独占重试：']
    lines += [f'- {k}：{v.get("error", "")}；{v["status"]}' for k,v in state.items() if v.get('error')]
    lines+=['','原八项结果和 Avg8 保留于原实验目录；补测不重新量化，不改变 seed、图像分辨率或计分。',
        'MMBench 为 EN dev/static；MMT 为 VAL；MME-RealWorld 为完整英文集；MMVP 为 pair accuracy；NaturalBench 为 G_ACC。']
    tmp=OUT/'PROGRESS.md.tmp';tmp.write_text('\n'.join(lines)+'\n');tmp.replace(OUT/'PROGRESS.md')

def main():
    lock=(OUT/'scheduler.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (OUT/'scheduler.pid').write_text(str(os.getpid())+'\n')
    jobs=build_jobs();save(OUT/'plan.json',jobs);old=read(OUT/'status.json',{});state={};running={}
    for job in jobs:
        key=job['key'];prior=old.get(key,{})
        state[key]={'status':'complete' if result(job) else 'pending',**({'exclusive_required':True} if prior.get('exclusive_required') else {})}
        if prior.get('status')=='running' and psutil.pid_exists(prior.get('pid',-1)):
            p=psutil.Process(prior['pid'])
            if p.status()!=psutil.STATUS_ZOMBIE and p.cmdline()==job['command']:state[key]=prior;running[key]=(p,None)
    while True:
        for key,(process,log) in list(running.items()):
            if isinstance(process,subprocess.Popen):code=process.poll();finished=code is not None
            else:
                try:finished=not process.is_running() or process.status()==psutil.STATUS_ZOMBIE
                except psutil.NoSuchProcess:finished=True
                code=None
            if not finished:continue
            job=next(j for j in jobs if j['key']==key)
            if code in (0,None) and result(job):state[key].update(status='complete',finished=time.time())
            else:
                tail=Path(state[key]['log']).read_text(errors='replace')[-12000:]
                oom='out of memory' in tail.lower()
                retry=oom and not state[key].get('exclusive_required')
                state[key].update(status='pending' if retry else 'failed',exclusive_required=oom,
                    error='Shared-memory OOM; queued for exclusive retry' if retry else tail[-1200:])
            if log:log.close()
            del running[key]
        try:free=resources()
        except Exception as error:
            print('Resource collection failed; launches deferred:',error,flush=True);save(OUT/'status.json',state);report(jobs,state);time.sleep(10);continue
        intern=read(ROOT/'outputs/internvl2_8b_methods_20261115/status.json',{})
        busy={v['npu']:key for key,v in intern.items() if v['status']=='running'}
        intern_pending=any(v['status']=='pending' for v in intern.values())
        used={state[k]['npu'] for k in running}
        for job in jobs:
            key=job['key'];v=state[key]
            if v['status']!='pending':continue
            deps=[state[d]['status'] for d in job['dependencies']]
            if any(d in ['failed','dependency_failed'] for d in deps):v.update(status='dependency_failed',error='Entry validation failed');continue
            if any(d!='complete' for d in deps):continue
            exclusive=v.get('exclusive_required',False)
            candidates=[card for card in range(8) if card not in used and free.get(card,0)>=34*1024**3
                and (not exclusive or card not in busy and not intern_pending) and ('__quant' not in busy.get(card,''))]
            if not candidates:continue
            card=max(candidates,key=lambda c:free[c]);fraction=0.90 if exclusive else 0.44
            env=dict(os.environ,ASCEND_RT_VISIBLE_DEVICES=str(card),QWEN_EXTRA_MEMORY_FRACTION=str(fraction),
                HF_DATASETS_CACHE='/data/lkp/datasets/llava_onevision_hf_cache',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4',
                TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled',ASCEND_GLOBAL_LOG_LEVEL='3',ASCEND_SLOG_PRINT_TO_STDOUT='0')
            path=OUT/'logs'/(key+'.log');log=path.open('a');process=subprocess.Popen(job['command'],cwd=SOURCE,env=env,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            state[key]={**v,'status':'running','pid':process.pid,'npu':card,'log':str(path),'started':time.time(),'memory_fraction':fraction}
            running[key]=(process,log);used.add(card);print('Started',key,'NPU',card,'PID',process.pid,flush=True)
        save(OUT/'status.json',state);report(jobs,state)
        if all(v['status'] in ['complete','failed','dependency_failed'] for v in state.values()):
            save(OUT/'completion.json',{'finished':time.time(),'counts':read(OUT/'queue_summary.json'),
                'all_successful':all(v['status']=='complete' for v in state.values())});return
        time.sleep(10)

if __name__=='__main__':main()
