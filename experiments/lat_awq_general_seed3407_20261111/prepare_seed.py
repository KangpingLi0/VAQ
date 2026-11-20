"""Select the additional seed once, fetch its images, and check test overlap."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[1]
OUT = REPO / 'outputs' / BASE.name
PREVIOUS = REPO / 'outputs' / 'lat_awq_general_20261107'
ROOT = Path('/data/lkp/datasets/sharegpt4v')
SEED = 3407


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def prepare():
    fetcher = load_module('original_calibration_preparation', REPO / 'scripts/prepare_general_calibration.py')
    source = ROOT / 'sharegpt4v_instruct_gpt4-vision_cap100k.json'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    previous = json.loads((PREVIOUS / 'calibration_preparation.json').read_text())
    assert digest == previous['source_sha256'], 'ShareGPT4V source changed'
    unique = {}
    for row in json.loads(source.read_text()):
        if row.get('image', '').startswith('coco/train2017/'):
            unique.setdefault(row['image'], row)
    rows = list(unique.values())
    # Verify the selection algorithm against an existing seed before extending it.
    reference = [rows[i]['id'] for i in np.random.default_rng(2026).permutation(len(rows))[:128]]
    assert reference == previous['selection_ids']['calibration_n128_seed2026']
    selected = [dict(rows[i]) for i in np.random.default_rng(SEED).permutation(len(rows))[:128]]
    for row in selected:
        row['image'] = str(ROOT / row['image'])
    assert len({row['image'] for row in selected}) == 128
    destination = OUT / f'calibration_n128_seed{SEED}.json'
    if destination.exists():
        assert json.loads(destination.read_text()) == selected, 'Existing seed selection changed'
    else:
        save(destination, selected)
    status = {
        'status': 'downloading_images', 'seed': SEED, 'source_rows': len(rows),
        'source_sha256': digest, 'required_images': 128, 'completed_images': 0,
        'selection_ids': {f'calibration_n128_seed{SEED}': [row['id'] for row in selected]},
        'selection_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
        'selection_policy': 'Original unique COCO train2017 image order; default_rng(seed).permutation; first 128; no score-based filtering',
    }
    save(OUT / 'calibration_preparation.json', status)
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(fetcher.fetch_image, str(Path(row['image']).relative_to(ROOT))) for row in selected]
        for future in as_completed(futures):
            future.result()
            status['completed_images'] += 1
            if status['completed_images'] % 8 == 0:
                save(OUT / 'calibration_preparation.json', status)
                print('Calibration images', status['completed_images'], '/128', flush=True)
    status['status'] = 'ready'
    save(OUT / 'calibration_preparation.json', status)
    return selected, status['selection_sha256']


def audit(selected, selection_sha256):
    from datasets import Image, List, load_dataset
    helpers = load_module('original_image_audit', REPO / 'scripts/audit_general_data.py')
    destination = OUT / 'data_audit.json'
    status = json.loads(destination.read_text()) if destination.exists() else {}
    if status:
        assert status['selection_sha256'] == selection_sha256, 'Audit selection changed'
    else:
        status = {
            'status': 'auditing', 'seed': SEED, 'selection_sha256': selection_sha256,
            'calibration_unique_images': 128, 'tasks': {},
            'comparison': 'SHA256 of original bytes and RGB size+decoded pixels, same as prior experiment',
        }
    references = {}
    for row in selected:
        for digest in helpers.hashes(Path(row['image']).read_bytes()):
            references.setdefault(digest, []).append(row['id'])
    save(destination, status)
    prepared = json.loads((OUT / 'benchmark_preparation.json').read_text())
    for task, definition in prepared['tasks'].items():
        if task in status['tasks']:
            continue
        dataset = load_dataset('parquet', data_files={definition['split']: definition['files']}, split=definition['split'])
        columns = [key for key, feature in dataset.features.items() if isinstance(feature, Image)
                   or isinstance(feature, List) and isinstance(feature.feature, Image)
                   or key == 'image' and str(feature) == "Value('binary')"]
        if not columns:
            raise ValueError(f'No image columns: {task}')
        count = 0
        seen = set()
        overlaps = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            for batch in dataset.select_columns(columns).data.table.to_batches(max_chunksize=64):
                contents = [content for row in batch.to_pylist() for value in row.values() for content in helpers.images(value)]
                for raw, pixels in pool.map(helpers.hashes, contents):
                    count += 1
                    seen.add(raw)
                    matches = references.get(raw, []) + references.get(pixels, [])
                    if matches:
                        overlaps.append({'benchmark_image_sha256': raw, 'calibration_ids': matches})
        status['tasks'][task] = {
            'question_rows': len(dataset), 'image_occurrences': count,
            'unique_encoded_images': len(seen), 'overlaps': overlaps,
        }
        save(destination, status)
        print('Audit', task, count, 'images; overlaps', len(overlaps), flush=True)
    status['status'] = 'overlap_found' if any(v['overlaps'] for v in status['tasks'].values()) else 'passed'
    status['finished'] = time.time()
    save(destination, status)
    assert status['status'] == 'passed', 'Calibration/test image overlap found; quantization remains blocked'


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'preparation.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    selected, digest = prepare()
    audit(selected, digest)
