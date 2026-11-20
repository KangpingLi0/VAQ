"""Check calibration/evaluation image overlap without reading model predictions."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import time

from datasets import Image, List, load_dataset
from PIL import Image as PILImage

OUT = Path('/data/lkp/VAQ/outputs/lat_awq_general_20261107')


def hashes(content):
    raw = hashlib.sha256(content).hexdigest()
    with PILImage.open(io.BytesIO(content)) as image:
        image = image.convert('RGB')
        pixels = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
    return raw, pixels


def images(value):
    if value is None:
        return
    if isinstance(value, bytes):
        yield value
    elif isinstance(value, dict):
        if value.get('bytes') is not None:
            yield value['bytes']
        elif value.get('path'):
            yield Path(value['path']).read_bytes()
    elif isinstance(value, list):
        for item in value:
            yield from images(item)


def save(status):
    path = OUT / 'data_audit.json';temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2));temporary.replace(path)


def main():
    while not (OUT / 'calibration_n128_seed2026.json').exists():time.sleep(10)
    selected = {}
    for seed in [42, 123, 2026]:
        for row in json.loads((OUT / f'calibration_n128_seed{seed}.json').read_text()):
            selected.setdefault(row['image'], []).append(seed)
    references = {}
    for path, seeds in selected.items():
        for digest in hashes(Path(path).read_bytes()):
            references.setdefault(digest, []).append({'image': path, 'seeds': seeds})
    status = {'status': 'auditing', 'calibration_unique_images': len(selected), 'tasks': {},
              'comparison': 'SHA256 of original bytes and of RGB size+decoded pixels; excludes exact duplicates, not all semantic similarity'}
    save(status)
    while len(status['tasks']) < 8:
        prepared = json.loads((OUT / 'benchmark_preparation.json').read_text())
        for task, definition in prepared['tasks'].items():
            if task in status['tasks']:continue
            dataset = load_dataset('parquet', data_files={definition['split']: definition['files']}, split=definition['split'])
            columns = [key for key, feature in dataset.features.items() if isinstance(feature, Image)
                       or isinstance(feature, List) and isinstance(feature.feature, Image)
                       or key == 'image' and str(feature) == "Value('binary')"]
            if not columns:raise ValueError(f'No image columns: {task}')
            table = dataset.select_columns(columns).data.table
            count = 0;seen = set();overlaps = []
            with ThreadPoolExecutor(max_workers=8) as pool:
                for batch in table.to_batches(max_chunksize=64):
                    contents = [content for row in batch.to_pylist() for value in row.values() for content in images(value)]
                    for raw, pixels in pool.map(hashes, contents):
                        count += 1;seen.add(raw)
                        matches = references.get(raw, []) + references.get(pixels, [])
                        if matches:overlaps.append({'benchmark_image_sha256': raw, 'calibration_matches': matches})
            status['tasks'][task] = {'question_rows': len(dataset), 'image_occurrences': count,
                                     'unique_encoded_images': len(seen), 'overlaps': overlaps}
            save(status);print(task, count, 'overlaps', len(overlaps), flush=True)
        time.sleep(10)
    status['status'] = 'overlap_found' if any(t['overlaps'] for t in status['tasks'].values()) else 'passed'
    save(status)


if __name__ == '__main__':main()
