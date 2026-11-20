"""Prepare local benchmark inputs, retaining upstream prompts and score functions."""
import io
import json
from pathlib import Path
import re
import time
import zipfile

from datasets import Dataset, Features, Image, List, Value, load_dataset
from PIL import Image as PILImage

OUT = Path('/data/lkp/VAQ/outputs/lat_awq_general_20261107')
DATA = Path('/data/lkp/datasets/general_benchmarks')
TASKS = Path('/data/lkp/VAQ/experiments/lat_awq_general_20261107/source/3rdparty/lmms-eval-new/lmms_eval/tasks')
YAMLS = {
    'mmmu_val': 'mmmu/mmmu_val.yaml', 'ai2d': 'ai2d/ai2d.yaml',
    'scienceqa': 'scienceqa/scienceqa.yaml', 'textvqa_val': 'textvqa/_default_template_textvqa_yaml',
    'vizwiz_vqa_val': 'vizwiz_vqa/_default_template_vqa_yaml', 'mmstar': 'mmstar/mmstar.yaml',
    'cv_bench': 'cv_bench/cv_bench.yaml', 'seedbench': 'seedbench/seedbench.yaml',
}


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2));temporary.replace(path)


def local_yaml(task, files, split):
    path = TASKS / YAMLS[task]
    backup = path.with_name(path.name + '.upstream')
    if not backup.exists():
        backup.write_text(path.read_text())
    text = backup.read_text()
    # Only replace loading locations; leave prompts, split names and scoring intact.
    text = re.sub(r'^dataset_path:.*$', 'dataset_path: parquet', text, flags=re.M)
    text = re.sub(r'^dataset_name:.*\n?', '', text, flags=re.M)
    text = re.sub(r'^dataset_kwargs:\n(?:[ \t]+[^\n]*\n?)*', '', text, flags=re.M)
    text += '\ndataset_kwargs:\n  data_files:\n    ' + split + ':\n'
    text += ''.join('      - ' + json.dumps(str(file)) + '\n' for file in files)
    path.write_text(text)


def seed_images():
    folder = DATA / 'seedbench'
    target = folder / 'image_test.parquet'
    if target.exists():
        return target
    questions = json.loads((folder / 'SEED-Bench.json').read_text())['questions']
    questions = [q for q in questions if q['data_type'] == 'image']
    with zipfile.ZipFile(folder / 'SEED-Bench-image.zip') as archive:
        by_stem = {}
        for name in archive.namelist():
            if not name.endswith('/') and Path(name).suffix.lower() in {'', '.png', '.jpg', '.jpeg', '.webp'}:
                stem = Path(name).stem
                if stem in by_stem:
                    raise RuntimeError(f'Ambiguous image ID: {stem}')
                by_stem[stem] = name
        def examples():
            # Open in the generator so datasets can fingerprint its closure.
            with zipfile.ZipFile(str(folder / 'SEED-Bench-image.zip')) as images:
                for question in questions:
                    row = {k: str(question[k]) for k in ['question', 'question_id', 'answer', 'choice_a', 'choice_b', 'choice_c', 'choice_d', 'data_type']}
                    stem = Path(str(question['data_id'])).stem
                    if stem not in by_stem:
                        raise RuntimeError(f'Missing SEED image: {stem}')
                    content = images.read(by_stem[stem])
                    with PILImage.open(io.BytesIO(content)) as image:
                        image.verify()
                    row['image'] = [{'bytes': content, 'path': by_stem[stem]}]
                    yield row
        features = Features({key: Value('string') for key in ['question', 'question_id', 'answer', 'choice_a', 'choice_b', 'choice_c', 'choice_d', 'data_type']})
        features['image'] = List(Image())
        dataset = Dataset.from_generator(examples, features=features, keep_in_memory=False)
        temporary = target.with_suffix('.tmp');dataset.to_parquet(str(temporary));temporary.replace(target)
    return target


def main():
    status_path = OUT / 'benchmark_preparation.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {'status': 'waiting_downloads', 'tasks': {}}
    while len(status['tasks']) < 8:
        downloaded = json.loads((OUT / 'benchmark_download_status.json').read_text())
        for task, definition in downloaded['tasks'].items():
            if task in status['tasks'] or not all(Path(p).exists() for p in definition['files']):
                continue
            dataset = load_dataset('parquet', data_files={definition['split']: definition['files']}, split=definition['split'])
            assert len(dataset) > 0
            files = definition['files']
            # Official MMStar parquet stores JPEG bytes without an HF Image feature.
            # Restore that type so its original doc_to_visual receives PIL images.
            if task == 'mmstar':
                dataset = dataset.cast_column('image', Image())
                converted = DATA / task / 'mmstar_images.parquet'
                temporary = converted.with_suffix('.tmp');dataset.to_parquet(str(temporary));temporary.replace(converted)
                files = [str(converted)]
            local_yaml(task, files, definition['split'])
            status['tasks'][task] = {**definition, 'files': files, 'rows': len(dataset), 'ready': True, 'fingerprint': dataset._fingerprint}
            save(status_path, status)
            print(task, len(dataset), 'ready', flush=True)
        folder = DATA / 'seedbench'
        if 'seedbench' not in status['tasks'] and (folder / 'SEED-Bench-image.zip').exists() and (folder / 'SEED-Bench.json').exists():
            file = seed_images()
            dataset = load_dataset('parquet', data_files={'test': [str(file)]}, split='test')
            local_yaml('seedbench', [file], 'test')
            status['tasks']['seedbench'] = {'rows': len(dataset), 'ready': True, 'files': [str(file)], 'split': 'test',
                'cohort': 'all official image questions; seed_image metric', 'revision': downloaded['seedbench_revision'], 'fingerprint': dataset._fingerprint}
            save(status_path, status)
        time.sleep(10)
    status['status'] = 'ready';save(status_path, status)


if __name__ == '__main__':
    main()
