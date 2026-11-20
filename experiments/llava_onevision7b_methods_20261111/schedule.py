"""Resumable FP16/all-method queue restricted to six physical Ascend NPUs."""
import bootstrap
import csv
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import psutil

OUT=bootstrap.OUT
DEVICES=[1,3,4,5,6,7]
CORE_TASKS=['mmmu_val','ai2d','scienceqa','textvqa_val','seedbench','mmstar','cv_bench','vizwiz_vqa_val']
EXTRA_DEFINITIONS=json.loads((OUT/'extra_benchmarks.json').read_text())['benchmarks']
EXTRA_TASKS=[d['task'] for d in EXTRA_DEFINITIONS]
TASKS=CORE_TASKS+EXTRA_TASKS
KEYS={'mmmu_val':'mmmu_acc,none','ai2d':'exact_match,flexible-extract','scienceqa':'exact_match,none',
      'textvqa_val':'exact_match,none','seedbench':'seed_image,none','mmstar':'average,none',
      'cv_bench':'cv_bench_acc,none','vizwiz_vqa_val':'exact_match,none'}
KEYS.update({d['task']:d['primary_metric'] for d in EXTRA_DEFINITIONS})
SCORE_TO_PERCENT={t:100.0 for t in CORE_TASKS}
SCORE_TO_PERCENT.update({d['task']:d['score_to_percent'] for d in EXTRA_DEFINITIONS})
VARIANTS=[{'name':'fp16','method':'fp16','w':16,'a':16}]
VARIANTS += [{'name':f'{m}_w3a16','method':m,'w':3,'a':16} for m in ['rtn','awq','gptq','mbq','qig','lat_awq']]
VARIANTS += [{'name':f'{m}_w4a8','method':m,'w':4,'a':8} for m in ['rtn','smoothquant','mbq','qig','lat_awq']]


def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def save(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def build_jobs():
    jobs=[]
    def add(key,command,dependencies=(),**metadata):
        jobs.append({'key':key,'command':[sys.executable,'-u',*command],
            'dependencies':list(dependencies),**metadata})
    def evaluate(v,task,smoke=False,dependencies=()):
        name=v['name']+('_smoke' if smoke and v['name']!='fp16' else '')
        output=OUT/name/'eval'/(task+('_smoke' if smoke else ''))
        args=f'pretrained={bootstrap.MODEL},torch_dtype=float16,attn_implementation=sdpa'
        if v['name']!='fp16':
            args+=f',state_path={OUT/name/"quantized_state.pt"},w_bit={v["w"]},a_bit={v["a"]}'
        command=[str(bootstrap.BASE/'evaluate_model.py'),'--model','llava_onevision','--model_args',args,
            '--tasks',task,'--device','npu','--batch_size','1','--seed','42',
            '--gen_kwargs','temperature=0,do_sample=False,max_new_tokens=256',
            '--bootstrap_iters','1000','--log_samples','--output_path',str(output),'--verbosity','DEBUG']
        if smoke:command+=['--limit','2']
        key=v['name']+'__'+task+('__smoke' if smoke else '')
        add(key,command,dependencies,kind='eval',variant=v['name'],task=task,
            smoke=smoke,eval_dir=str(output),priority=0 if smoke else 2)
        return key
    for n in [2,128]:
        add(f'calibration_n{n}',[str(bootstrap.BASE/'run_model.py'),'calibration','--n',str(n)],
            kind='calibration',n=n,priority=0)
    baseline_smoke=evaluate(VARIANTS[0],'ai2d',smoke=True)
    smoke_gates={}
    for v in VARIANTS[1:]:
        key=v['name']+'__quant__smoke'
        add(key,[str(bootstrap.BASE/'run_model.py'),'quant','--method',v['method'],'--w',str(v['w']),
            '--a',str(v['a']),'--n','2'],['calibration_n2',baseline_smoke],
            kind='quant',variant=v['name'],n=2,smoke=True,priority=0)
        smoke_gates[v['name']]=evaluate(v,'ai2d',smoke=True,dependencies=[key])
    for task in TASKS:
        evaluate(VARIANTS[0],task,dependencies=[baseline_smoke])
    for v in VARIANTS[1:]:
        key=v['name']+'__quant'
        add(key,[str(bootstrap.BASE/'run_model.py'),'quant','--method',v['method'],'--w',str(v['w']),
            '--a',str(v['a'])],['calibration_n128',smoke_gates[v['name']]],
            kind='quant',variant=v['name'],n=128,smoke=False,priority=1)
        for task in TASKS:evaluate(v,task,dependencies=[key])
    return jobs


def result(job):
    prepared=read(OUT/'benchmark_preparation.json',{}).get('tasks',{}).get(job['task'],{})
    if not prepared.get('ready'):return None
    expected=2 if job['smoke'] else prepared['rows']
    for path in sorted(Path(job['eval_dir']).rglob('*_results.json'),reverse=True):
        data=read(path,{})
        if KEYS[job['task']] not in data.get('results',{}).get(job['task'],{}):continue
        if data.get('n-samples',{}).get(job['task'],{}).get('effective')!=expected:continue
        return path,data
    return None


def completed(job):
    if job['kind']=='calibration':
        path=OUT/f'calibration_n{job["n"]}_seed42.pt'
        return path.exists() and read(path.with_suffix('.meta.json'),{}).get('cases')==job['n']
    if job['kind']=='quant':
        path=OUT/(job['variant']+('_smoke' if job['smoke'] else ''))
        return (path/'quantized_state.pt').exists() and read(path/'quantization.json',{}).get('complete',False)
    return result(job) is not None


def report(jobs,state):
    counts={s:sum(v['status']==s for v in state.values()) for s in ['pending','running','complete','failed','dependency_failed']}
    scores={};extra_metrics=[]
    for job in jobs:
        if job['kind']!='eval' or job['smoke'] or state[job['key']]['status']!='complete':continue
        _,data=result(job)
        metrics=data['results'][job['task']]
        scores.setdefault(job['variant'],{})[job['task']]=float(metrics[KEYS[job['task']]])*SCORE_TO_PERCENT[job['task']]
        if job['task'] in EXTRA_TASKS:
            definition=read(OUT/'benchmark_preparation.json')['tasks'][job['task']]
            for metric,value in metrics.items():
                if isinstance(value,(int,float)) and 'stderr' not in metric:
                    extra_metrics.append({'variant':job['variant'],'benchmark':job['task'],'metric':metric,
                        'native_value':value,'score_percent':value*SCORE_TO_PERCENT[job['task']],
                        'calibration_overlap_occurrences':definition.get('calibration_overlap_occurrences',0)})
    rows=[]
    for v in VARIANTS:
        values=scores.get(v['name'],{})
        rows.append({'variant':v['name'],**values,
            'average_8':sum(values[t] for t in CORE_TASKS)/8 if all(t in values for t in CORE_TASKS) else ''})
    temporary=OUT/'summary.csv.tmp'
    with temporary.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['variant',*TASKS,'average_8'])
        writer.writeheader();writer.writerows(rows)
    temporary.replace(OUT/'summary.csv')
    temporary=OUT/'summary_extra_metrics.csv.tmp'
    with temporary.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['variant','benchmark','metric','native_value','score_percent','calibration_overlap_occurrences'])
        writer.writeheader();writer.writerows(extra_metrics)
    temporary.replace(OUT/'summary_extra_metrics.csv')
    core_complete=sum(bool(r['average_8']) for r in rows)
    full_complete=sum(all(t in r for t in TASKS) for r in rows)
    baseline=next(r for r in rows if r['variant']=='fp16')
    result_lines=['# LLaVA-OneVision-7B 方法比较','',
        f'核心八项已完成：{core_complete}/12；全部十三项已完成：{full_complete}/12。',
        '仅使用完整测试集结果；每组共享相同评测设置；分数单位为百分数。',
        '校准 seed42，128 例；原生图像处理；仅量化语言解码器线性层。','',
        '| 方法 | '+ ' | '.join(TASKS)+' | Avg8 | 相对 FP16（百分点） |',
        '|---|'+'---:|'*(len(TASKS)+2)]
    for row in rows:
        values=[f'{row[t]:.4f}' if t in row else '' for t in TASKS]
        average=f'{row["average_8"]:.4f}' if row['average_8']!='' else ''
        delta=f'{row["average_8"]-baseline["average_8"]:+.4f}' if row['average_8']!='' and baseline['average_8']!='' else ''
        result_lines.append('| '+row['variant']+' | '+' | '.join([*values,average,delta])+' |')
    result_lines+=['','Avg8 始终仅计算原有八项；新增五项单独列分，不混入原平均分。',
        'MMBench 为 EN dev/static；MMT-Bench 为公开 VAL/官方 exact_matching；MMVP 主指标为 pair accuracy；NaturalBench 主指标为 G_ACC，四项完整指标见 summary_extra_metrics.csv。',
        'W4A16 按用户要求不运行；AWQ/GPTQ 的 W4A8 不受本仓库权重量化实现支持。',
        '低比特为伪量化精度实验，保存的张量仍为 FP16，不能据此声称真实部署压缩或加速。']
    preparation=read(OUT/'benchmark_preparation.json',{}).get('tasks',{})
    for task in EXTRA_TASKS:
        overlap=preparation.get(task,{}).get('calibration_overlap_occurrences',0)
        if overlap:result_lines.append(f'校准图像重复标记：{task} 有 {overlap} 次图像与校准集完全重复；保留完整测试集并明确标记，不能视为无重叠测试。')
    temporary=OUT/'RESULTS.md.tmp';temporary.write_text('\n'.join(result_lines)+'\n');temporary.replace(OUT/'RESULTS.md')
    assets=read(OUT/'download_status.json',{})
    checked=read(OUT/'method_validation.json',{})
    status='全部完成' if counts['complete']==len(jobs) else '运行中'
    if not counts['pending'] and not counts['running'] and (counts['failed'] or counts['dependency_failed']):status='已结束，存在失败项'
    if read(OUT/'model_integrity.json',{}).get('status')!='passed':status='下载及校验模型'
    elif checked.get('status')!='passed':status='验证量化方法适配'
    lines=['# LLaVA-OneVision-7B 全方法评测进度','',
        f'状态：{status}；更新时间：{datetime.now(ZoneInfo("Asia/Shanghai")):%Y-%m-%d %H:%M:%S}（北京时间）。','',
        '物理 NPU：1、3、4、5、6、7；0、2 留给 seed3407 消融。',
        'W3A16：RTN、AWQ、GPTQ、MBQ、QIG、LAT-AWQ。',
        'W4A8：RTN、SmoothQuant、MBQ、QIG、LAT-AWQ。另含 FP16，共 12 组，每组原八项及新增五项基准。',
        '校准：seed42、128 张已审计的 ShareGPT4V COCO train2017 图像；需要校准的方法共用 OneVision 缓存，RTN 不使用校准数据。',
        '模型：llava-hf/llava-onevision-qwen2-7b-ov-hf，固定 revision；原生 anyres_max_9、官方聊天模板、FP16/SDPA、greedy/256 token。','',
        f'任务状态：{counts}；核心八项完成 {core_complete}/12，全部十三项完成 {full_complete}/12。',
        f'模型文件：{len(assets.get("completed_files",[]))}/{assets.get("required_files","?")}；状态 {assets.get("status","waiting")}。',
        f'微型真实解码器方法验证：{checked.get("status","waiting")}。','',
        f'独立小模型端到端评测入口验证：{read(OUT/"integration_validation.json",{}).get("status","waiting")}。','',
        '运行中：']
    lines += [f'- {k}：NPU {v.get("npu")}，PID {v.get("pid")}' for k,v in state.items() if v['status']=='running']
    lines += ['','失败任务：']+[f'- {k}：{v.get("error")}' for k,v in state.items() if v['status'] in ['failed','dependency_failed']]
    lines += ['','每种方法/位宽先通过完整 checkpoint 两例量化、严格权重重载及生成评测，再启动正式量化。',
        '完整核心八项齐全后才填写 Avg8；新增五项单列，失败和缺失不计为零。全部为伪量化精度实验，权重仍以 FP16 张量保存。','',
        '新增基准准备：']
    extra=read(OUT/'extra_benchmark_preparation.json',{}).get('tasks',{})
    for definition in EXTRA_DEFINITIONS:
        entry=extra.get(definition['task'],{})
        lines.append(f'- {definition["label"]}：{entry.get("status","queued")}；文件 {entry.get("files_downloaded",0)}/{len(definition["files"])}；样本 {entry.get("rows","待准备")}。')
        if entry.get('error'):lines.append('  错误：'+entry['error'])
        if entry.get('calibration_overlap_occurrences',0):lines.append(f'  校准图像重复 {entry["calibration_overlap_occurrences"]} 次，完整测试集保留并明确标记。')
    temporary=OUT/'PROGRESS.md.tmp';temporary.write_text('\n'.join(lines)+'\n');temporary.replace(OUT/'PROGRESS.md')
    save(OUT/'queue_summary.json',counts)


def main():
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'scheduler.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (OUT/'scheduler.pid').write_text(str(os.getpid())+'\n')
    jobs=build_jobs();save(OUT/'plan.json',jobs)
    previous=read(OUT/'status.json',{});state={};running={}
    for job in jobs:
        key=job['key'];old=previous.get(key,{})
        state[key]={'status':'complete' if completed(job) else 'pending'}
        if old.get('status')=='running' and psutil.pid_exists(old.get('pid',-1)):
            process=psutil.Process(old['pid'])
            if process.status()!=psutil.STATUS_ZOMBIE and process.cmdline()==job['command']:
                state[key]=old;running[key]=(process,None)
                print('Adopted',key,'PID',process.pid,flush=True)
    by_key={j['key']:j for j in jobs}
    while True:
        for key,(process,log) in list(running.items()):
            if isinstance(process,subprocess.Popen):
                code=process.poll();finished=code is not None
            else:
                try:finished=not process.is_running() or process.status()==psutil.STATUS_ZOMBIE
                except psutil.NoSuchProcess:finished=True
                code=None
            if not finished:continue
            success=(code in (0,None)) and completed(by_key[key])
            state[key].update(status='complete' if success else 'failed',finished=time.time(),exit_code=code)
            if not success:state[key]['error']='Process failed or expected complete artifact missing; inspect '+state[key]['log']
            if log:log.close()
            del running[key]
            print('Finished',key,state[key]['status'],flush=True)
        free=[d for d in DEVICES if d not in {state[k]['npu'] for k in running}]
        ready=(read(OUT/'model_integrity.json',{}).get('status')=='passed'
            and read(OUT/'method_validation.json',{}).get('status')=='passed'
            and read(OUT/'integration_validation.json',{}).get('status')=='passed')
        for job in sorted(jobs,key=lambda j:j['priority']):
            key=job['key']
            if state[key]['status']!='pending':continue
            dependencies=[state[d]['status'] for d in job['dependencies']]
            if any(v in ('failed','dependency_failed') for v in dependencies):
                state[key]={'status':'dependency_failed','error':'A prerequisite failed'}
                continue
            if not ready or not free or any(v!='complete' for v in dependencies):continue
            if job['kind']=='eval' and not read(OUT/'benchmark_preparation.json',{}).get('tasks',{}).get(job['task'],{}).get('ready'):continue
            device=free.pop(0)
            env=dict(os.environ,ASCEND_RT_VISIBLE_DEVICES=str(device),OMP_NUM_THREADS='8',
                QIG_SCALE_SEARCH_BATCH_SIZE='8',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',
                WANDB_MODE='disabled',ASCEND_GLOBAL_LOG_LEVEL='3',ASCEND_SLOG_PRINT_TO_STDOUT='0')
            path=OUT/'logs'/(key+'.log');log=path.open('a',buffering=1)
            process=subprocess.Popen(job['command'],cwd=bootstrap.SOURCE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            state[key]={'status':'running','npu':device,'pid':process.pid,'started':time.time(),'command':job['command'],'log':str(path)}
            running[key]=(process,log)
            print('Started',key,'NPU',device,'PID',process.pid,flush=True)
        save(OUT/'status.json',state);report(jobs,state)
        if all(v['status'] in ('complete','failed','dependency_failed') for v in state.values()):
            save(OUT/'completion.json',{'finished':time.time(),'counts':read(OUT/'queue_summary.json'),
                'all_successful':all(v['status']=='complete' for v in state.values())})
            return
        time.sleep(10)


if __name__=='__main__':
    main()
