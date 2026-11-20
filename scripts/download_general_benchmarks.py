"""Fetch only evaluation splits, retaining the benchmark's original examples."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys

import requests

sys.path.insert(0, str(Path(__file__).parent))
from download_general_assets import download

OUT = Path('/data/lkp/VAQ/outputs/lat_awq_general_20261107')
DATA = Path('/data/lkp/datasets/general_benchmarks')


def main():
    definitions = [
        ('mmmu_val', 'lmms-lab/MMMU', 'validation', lambda name: name.startswith('data/validation-')),
        ('ai2d', 'lmms-lab/ai2d', 'test', lambda name: name.startswith('data/test-')),
        ('scienceqa', 'lmms-lab/ScienceQA', 'test', lambda name: name.startswith('ScienceQA-FULL/test-')),
        ('textvqa_val', 'lmms-lab/textvqa', 'validation', lambda name: name.startswith('data/validation-')),
        ('vizwiz_vqa_val', 'lmms-lab/VizWiz-VQA', 'val', lambda name: name.startswith('data/val-')),
        ('mmstar', 'Lin-Chen/MMStar', 'val', lambda name: name == 'mmstar.parquet'),
        ('cv_bench', 'nyu-visionx/CV-Bench', 'test', lambda name: name in ['test_2d.parquet', 'test_3d.parquet']),
    ]
    status = {'status': 'downloading', 'tasks': {}, 'seedbench': 'official image-only cohort preparation pending'}
    jobs = []
    for task, repo, split, predicate in definitions:
        meta = json.loads((OUT / 'dataset_metadata' / (repo.replace('/', '--') + '.json')).read_text())
        paths = [item['rfilename'] for item in meta['siblings'] if predicate(item['rfilename']) and item['rfilename'].endswith('.parquet')]
        if not paths:
            raise RuntimeError(f'No files selected for {task}')
        targets = []
        for name in paths:
            target = DATA / task / Path(name).name
            target.parent.mkdir(parents=True, exist_ok=True)
            targets.append(str(target))
            jobs.append((task, f"https://huggingface.co/datasets/{repo}/resolve/{meta['sha']}/{name}", target))
        status['tasks'][task] = {'repo': repo, 'revision': meta['sha'], 'split': split,
                                 'files': targets, 'required': len(targets), 'downloaded': 0}
    seed_meta = json.loads((OUT / 'dataset_metadata' / 'AILab-CVC--SEED-Bench.json').read_text())
    for name in ['SEED-Bench.json', 'SEED-Bench-image.zip']:
        target = DATA / 'seedbench' / name;target.parent.mkdir(parents=True, exist_ok=True)
        jobs.append(('seedbench', f"https://huggingface.co/datasets/AILab-CVC/SEED-Bench/resolve/{seed_meta['sha']}/{name}", target))
    status['seedbench_revision'] = seed_meta['sha']
    status_path = OUT / 'benchmark_download_status.json'
    def save():
        temporary = status_path.with_suffix('.tmp');temporary.write_text(json.dumps(status, indent=2));temporary.replace(status_path)
    save()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(download, url, target): task for task, url, target in jobs}
        for future in as_completed(futures):
            future.result();task = futures[future]
            if task in status['tasks']:
                status['tasks'][task]['downloaded'] += 1
            save()
    status['status'] = 'downloaded'
    status['seedbench'] = 'official image archive and annotations downloaded'
    save()


if __name__ == '__main__':
    main()
