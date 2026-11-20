"""Resumable four-NPU experiment queue; medical NPUs are never allocated here."""
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

BASE = Path(__file__).resolve().parent
SOURCE = BASE / 'source'
OUT = Path('/data/lkp/VAQ/outputs/lat_awq_general_20261107')
PYTHON = '/data/lkp/envs/vaq-general/bin/python'
MODEL = '/data/lkp/models/Qwen2.5-VL-7B-Instruct'
DEVICES = [0, 2, 3, 4]
TASKS = ['mmmu_val', 'ai2d', 'scienceqa', 'textvqa_val', 'seedbench', 'mmstar', 'cv_bench', 'vizwiz_vqa_val']
KEYS = {'mmmu_val': 'mmmu_acc,none', 'ai2d': 'exact_match,flexible-extract', 'scienceqa': 'exact_match,none',
        'textvqa_val': 'exact_match,none', 'seedbench': 'seed_image,none', 'mmstar': 'average,none',
        'cv_bench': 'cv_bench_acc,none', 'vizwiz_vqa_val': 'exact_match,none'}


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def save(path, value):
    temporary = path.with_suffix('.tmp');temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2));temporary.replace(path)


def build_jobs():
    jobs = []
    def add(key, command, dependencies=(), **kw):
        jobs.append(dict(key=key, command=[PYTHON, '-u', *command], dependencies=list(dependencies), **kw))
    def calibration(n, seed):
        key = f'calibration_n{n}_seed{seed}'
        add(key, ['general_run.py', 'calibration', '--n', str(n), '--seed', str(seed)],
            artifact=str(OUT / f'{key}.pt'), needs_calibration=True, needs_data_audit=n > 2)
        return key
    n2 = calibration(2, 42)
    def evaluate(variant, task, dependencies=(), smoke=False, w=16, a=16):
        key = f'{variant}__{task}' + ('__smoke' if smoke else '')
        output = OUT / variant / 'eval' / (task + ('_smoke' if smoke else ''))
        cmd = ['main.py', '--model', 'qwen2_5_vl', '--model_args',
            f'pretrained={MODEL},torch_dtype=float16,attn_implementation=sdpa,min_pixels=200704,max_pixels=1605632',
            '--tasks', task, '--device', 'npu', '--batch_size', '1', '--seed', '42',
            '--gen_kwargs', 'temperature=0,do_sample=False,max_new_tokens=256', '--bootstrap_iters', '1000',
            '--log_samples', '--output_path', str(output), '--verbosity', 'DEBUG']
        if variant != 'fp16':
            cmd += ['--quantized_state', str(OUT / variant / 'quantized_state.pt'), '--w_bit', str(w), '--a_bit', str(a)]
        if smoke:
            cmd += ['--limit', '2']
        add(key, cmd, dependencies, task=task, eval_dir=str(output), variant=variant, smoke=smoke)
        return key
    eval_smoke = evaluate('fp16', 'ai2d', smoke=True)
    smoke_keys = []
    for w, a in [(3, 16), (4, 8)]:
        variant = f'w{w}a{a}_D_seed42_smoke'
        key = f'{variant}__quant'
        add(key, ['general_run.py', 'quant', '--w', str(w), '--a', str(a), '--variant', 'D', '--n', '2'], [n2],
            artifact=str(OUT / variant / 'quantization.json'))
        smoke_keys.append(evaluate(variant, 'ai2d', [key, eval_smoke], smoke=True, w=w, a=a))
    # Shared caches are generated once per seed, reused by all four module settings.
    for seed in [42, 123, 2026]:
        cache_key = calibration(128, seed)
        for w, a in [(3, 16), (4, 8)]:
            for setting in 'ABCD':
                variant = f'w{w}a{a}_{setting}_seed{seed}'
                key = variant + '__quant'
                add(key, ['general_run.py', 'quant', '--w', str(w), '--a', str(a), '--variant', setting,
                    '--seed', str(seed)], [cache_key, smoke_keys[0 if a == 16 else 1]], artifact=str(OUT / variant / 'quantization.json'))
                for task in TASKS:
                    evaluate(variant, task, [key, eval_smoke], w=w, a=a)
    for task in TASKS:
        evaluate('fp16', task, [eval_smoke])
    return jobs


def result(job):
    for path in sorted(Path(job['eval_dir']).rglob('*results*.json'), reverse=True):
        value = read(path, {})
        if job['task'] in value.get('results', {}):
            return path, value
    return None


def completed(job):
    if 'artifact' in job:
        return Path(job['artifact']).exists()
    return result(job) is not None


def report(jobs, state):
    counters = {s: sum(item['status'] == s for item in state.values()) for s in ['pending', 'running', 'complete', 'failed']}
    summary = {}
    for job in jobs:
        if not job.get('eval_dir') or job.get('smoke') or state[job['key']]['status'] != 'complete':
            continue
        found = result(job)
        if not found:
            continue
        path, data = found
        metrics = data['results'][job['task']]
        desired = KEYS[job['task']]
        # Fail visibly on unexpected metric names; do not substitute another score.
        if desired not in metrics:
            summary.setdefault(job['variant'], {})[job['task'] + '_metric_error'] = str(list(metrics))
            continue
        summary.setdefault(job['variant'], {})[job['task']] = float(metrics[desired]) * 100
    rows = []
    for variant, scores in summary.items():
        row = {'variant': variant, **scores}
        row['average_8'] = sum(scores[t] for t in TASKS) / 8 if all(t in scores for t in TASKS) else ''
        rows.append(row)
    with (OUT / 'summary.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=['variant', *TASKS, 'average_8'], extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    averaged = []
    for w, a in [(3, 16), (4, 8)]:
        for setting in 'ABCD':
            entries = [summary.get(f'w{w}a{a}_{setting}_seed{seed}', {}) for seed in [42, 123, 2026]]
            if not all(all(task in entry for task in TASKS) for entry in entries):continue
            import statistics
            values = [sum(entry[task] for task in TASKS) / 8 for entry in entries]
            averaged.append({'precision': f'w{w}a{a}', 'variant': setting, 'calibration_seeds': '42,123,2026',
                             'average_8_mean': statistics.mean(values), 'average_8_std': statistics.stdev(values),
                             **{task: statistics.mean(entry[task] for entry in entries) for task in TASKS}})
    with (OUT / 'summary_three_seeds.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=['precision', 'variant', 'calibration_seeds', *TASKS, 'average_8_mean', 'average_8_std'])
        writer.writeheader();writer.writerows(averaged)
    assets = read(OUT / 'download_status.json', {})
    calibration = read(OUT / 'calibration_preparation.json', {})
    benchmarks = read(OUT / 'benchmark_preparation.json', {})
    lines = ['# LAT-AWQ 通用模型消融进度', '', 'Qwen2.5-VL-7B-Instruct；NPU 0、2、3、4；内镜使用 1、5、6、7。', '',
        '配置：A=关闭 S/L，B=仅 S，C=仅 L，D=完整方法；W3A16/W4A8；校准种子 42/123/2026，每次 128 张。',
        '所有组共享校准样本、有效 token 掩码、搜索网格、图像分辨率、生成参数和计分。结果包含 FP16 参照。', '',
        f'任务状态：{counters}', f'模型下载：{assets.get("status")}，完成 {len(assets.get("completed_files", []))}/{assets.get("required_files", "?")} 个文件；校验 {read(OUT / "model_integrity.json", {}).get("status", "waiting")}。',
        f'校准图像：{calibration.get("completed_images", 0)}/{calibration.get("required_images", "?")}；状态 {calibration.get("status")}。',
        f'已准备基准：{list(benchmarks.get("tasks", {}))}',
        f'校准/评测图像重复检查：{read(OUT / "data_audit.json", {}).get("status", "waiting")}。', '', '运行中：']
    lines += [f'- {key}，NPU {item.get("npu")}，PID {item.get("pid")}' for key, item in state.items() if item['status'] == 'running']
    lines += ['', '失败任务：'] + [f'- {key}：{item.get("error")}' for key, item in state.items() if item['status'] == 'failed']
    (OUT / 'PROGRESS.md').write_text('\n'.join(lines) + '\n')
    save(OUT / 'queue_summary.json', counters)


def main():
    OUT.mkdir(exist_ok=True)
    lock = (OUT / 'scheduler.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (OUT / 'scheduler.pid').write_text(str(os.getpid()))
    jobs = build_jobs();save(OUT / 'plan.json', jobs)
    previous = read(OUT / 'status.json', {})
    state = {};running = {}
    for job in jobs:
        key = job['key'];old = previous.get(key, {})
        state[key] = {'status': 'complete' if completed(job) else 'pending'}
        if not completed(job) and old.get('status') == 'running' and psutil.pid_exists(old.get('pid', -1)):
            process = psutil.Process(old['pid'])
            if process.status() != psutil.STATUS_ZOMBIE and process.cmdline() == job['command']:
                state[key] = old;running[key] = (process, None)
    while True:
        for key, (process, log) in list(running.items()):
            if isinstance(process, subprocess.Popen):
                done = process.poll() is not None
                code = process.returncode
            else:
                try:done = not process.is_running() or process.status() == psutil.STATUS_ZOMBIE
                except psutil.NoSuchProcess:done = True
                code = None
            if not done:
                continue
            job = next(j for j in jobs if j['key'] == key)
            state[key].update(status='complete' if completed(job) else 'failed', finished=time.time(), exit_code=code)
            if not completed(job):state[key]['error'] = 'Process ended without complete artifact; inspect logs/' + key + '.log'
            if log:log.close()
            del running[key]
        assets = read(OUT / 'download_status.json', {})
        calib = read(OUT / 'calibration_preparation.json', {})
        benchmarks = read(OUT / 'benchmark_preparation.json', {}).get('tasks', {})
        free = [device for device in DEVICES if device not in [state[key]['npu'] for key in running]]
        for job in jobs:
            if not free or read(OUT / 'model_integrity.json', {}).get('status') != 'passed':break
            key = job['key']
            if state[key]['status'] != 'pending' or any(state[d]['status'] != 'complete' for d in job['dependencies']):continue
            if job.get('needs_calibration') and calib.get('status') != 'ready':continue
            if job.get('needs_data_audit') and read(OUT / 'data_audit.json', {}).get('status') != 'passed':continue
            if job.get('task') and job['task'] not in benchmarks:continue
            device = free.pop(0)
            env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=str(device), OMP_NUM_THREADS='8',
                QIG_SCALE_SEARCH_BATCH_SIZE='8', TOKENIZERS_PARALLELISM='false', ASCEND_GLOBAL_LOG_LEVEL='3',
                ASCEND_SLOG_PRINT_TO_STDOUT='0', HF_HUB_OFFLINE='1', WANDB_MODE='disabled')
            logpath = OUT / 'logs' / (key + '.log');logpath.parent.mkdir(exist_ok=True)
            log = logpath.open('a', buffering=1)
            process = subprocess.Popen(job['command'], cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT)
            state[key].update(status='running', npu=device, pid=process.pid, started=time.time(), command=job['command'])
            running[key] = (process, log)
            print('Started', key, 'NPU', device, 'PID', process.pid, flush=True)
        save(OUT / 'status.json', state);report(jobs, state)
        if all(item['status'] == 'complete' for item in state.values()):
            with (OUT / 'paired_analysis.log').open('a') as log:
                subprocess.run([PYTHON, '-u', str(BASE / 'analyze_general.py')], cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT, check=True)
            break
        time.sleep(15)


if __name__ == '__main__':
    main()
