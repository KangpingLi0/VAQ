"""Run the original ablation implementation for seed 3407 on physical NPUs 0,2."""
import csv
from datetime import datetime
import importlib.util
import json
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[1]
PREVIOUS = BASE.parent / 'lat_awq_general_20261107'
OUT = REPO / 'outputs' / BASE.name
SEED = 3407
DEVICES = [0, 2]

spec = importlib.util.spec_from_file_location('original_scheduler', PREVIOUS / 'schedule_general.py')
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)
queue.BASE = BASE
queue.OUT = OUT
queue.DEVICES = DEVICES


def build_jobs():
    jobs = []

    def add(key, command, dependencies=(), **metadata):
        jobs.append(dict(key=key, command=[queue.PYTHON, '-u', *command], dependencies=list(dependencies), **metadata))

    calibration = f'calibration_n128_seed{SEED}'
    add(calibration, [str(BASE / 'run_seed.py'), 'calibration', '--n', '128', '--seed', str(SEED)],
        artifact=str(OUT / f'{calibration}.pt'), needs_calibration=True)
    # Keep the original ordering and every evaluation argument, including eval seed 42.
    for w, a in [(3, 16), (4, 8)]:
        for setting in 'ABCD':
            variant = f'w{w}a{a}_{setting}_seed{SEED}'
            quantization = variant + '__quant'
            add(quantization, [str(BASE / 'run_seed.py'), 'quant', '--w', str(w), '--a', str(a),
                '--variant', setting, '--seed', str(SEED)], [calibration],
                artifact=str(OUT / variant / 'quantization.json'), needs_data_audit=True)
            for task in queue.TASKS:
                output = OUT / variant / 'eval' / task
                command = ['main.py', '--model', 'qwen2_5_vl', '--model_args',
                    f'pretrained={queue.MODEL},torch_dtype=float16,attn_implementation=sdpa,min_pixels=200704,max_pixels=1605632',
                    '--tasks', task, '--device', 'npu', '--batch_size', '1', '--seed', '42',
                    '--gen_kwargs', 'temperature=0,do_sample=False,max_new_tokens=256', '--bootstrap_iters', '1000',
                    '--log_samples', '--output_path', str(output), '--verbosity', 'DEBUG',
                    '--quantized_state', str(OUT / variant / 'quantized_state.pt'), '--w_bit', str(w), '--a_bit', str(a)]
                add(variant + '__' + task, command, [quantization],
                    task=task, eval_dir=str(output), variant=variant, smoke=False)
    return jobs


def report(jobs, state):
    counts = {key: sum(value['status'] == key for value in state.values())
              for key in ['pending', 'running', 'complete', 'failed']}
    scores = {}
    for job in jobs:
        if not job.get('eval_dir') or state[job['key']]['status'] != 'complete':
            continue
        _, data = queue.result(job)
        metric = data['results'][job['task']][queue.KEYS[job['task']]]
        scores.setdefault(job['variant'], {})[job['task']] = float(metric) * 100
    rows = []
    for variant, values in scores.items():
        rows.append({'variant': variant, **values,
            'average_8': sum(values[task] for task in queue.TASKS) / 8 if all(task in values for task in queue.TASKS) else ''})
    temporary = OUT / 'summary.csv.tmp'
    with temporary.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=['variant', *queue.TASKS, 'average_8'])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(OUT / 'summary.csv')
    preparation = queue.read(OUT / 'calibration_preparation.json', {})
    audit = queue.read(OUT / 'data_audit.json', {})
    completed_models = sum(bool(row['average_8']) for row in rows)
    lines = [
        '# LAT-AWQ seed3407 消融进度', '',
        f'更新时间：{datetime.now(ZoneInfo("Asia/Shanghai")):%Y-%m-%d %H:%M:%S}（北京时间）。', '',
        'Qwen2.5-VL-7B-Instruct；仅使用物理 NPU 0、2。',
        '校准种子 3407，128 张 COCO train2017 ShareGPT4V 图像；四组共享同一缓存。',
        'A=关闭 S/L，B=仅 S，C=仅 L，D=完整 S+L；W3A16、W4A8 共八组，每组完整八项基准。',
        '沿用原实验隔离源码、量化算法和评测参数；推理解码种子仍固定为 42。', '',
        f'任务状态：{counts}；完整评测组数：{completed_models}/8。',
        f'校准图像：{preparation.get("completed_images", 0)}/128；准备状态：{preparation.get("status", "waiting")}。',
        f'校准/测试图像重复检查：{audit.get("status", "waiting")}，已检查 {len(audit.get("tasks", {}))}/8 项基准。', '',
        '运行中：',
    ]
    lines += [f'- {key}，NPU {value.get("npu")}，PID {value.get("pid")}' for key, value in state.items() if value['status'] == 'running']
    lines += ['', '失败任务：']
    lines += [f'- {key}：{value.get("error")}' for key, value in state.items() if value['status'] == 'failed']
    lines += ['', '完整八项结果齐全后才计算 average_8。完成结果见 summary.csv；不覆盖原三种子汇总。']
    temporary = OUT / 'PROGRESS.md.tmp'
    temporary.write_text('\n'.join(lines) + '\n')
    temporary.replace(OUT / 'PROGRESS.md')
    queue.save(OUT / 'queue_summary.json', counts)


if __name__ == '__main__':
    queue.build_jobs = build_jobs
    queue.report = report
    queue.main()
