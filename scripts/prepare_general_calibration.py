"""Prepare fixed, image-disjoint calibration subsets without looking at benchmark scores."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image

ROOT = Path('/data/lkp/datasets/sharegpt4v')
OUT = Path('/data/lkp/VAQ/outputs/lat_awq_general_20261107')


def fetch_image(relative):
    target = ROOT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        with Image.open(target) as image:
            image.verify()
        return relative
    name = Path(relative).name
    temporary = target.with_suffix('.download')
    for base in ['https://images.cocodataset.org/train2017/', 'https://s3.amazonaws.com/images.cocodataset.org/train2017/']:
        result = subprocess.run(['curl', '-sS', '-L', '--fail', '--retry', '2', '--connect-timeout', '15',
                                 '--max-time', '90', '-o', str(temporary), base + name], capture_output=True)
        if result.returncode == 0:
            try:
                with Image.open(temporary) as image:
                    image.verify()
                temporary.replace(target)
                return relative
            except Exception:
                pass
    raise RuntimeError(f'Image download failed: {relative}')


def main():
    path = ROOT / 'sharegpt4v_instruct_gpt4-vision_cap100k.json'
    while not path.exists():
        time.sleep(5)
    rows = json.loads(path.read_text())
    rows = [row for row in rows if row.get('image', '').startswith('coco/train2017/')]
    # Keep only one conversation per image so nominal N is also the image count.
    unique = {}
    for row in rows:
        unique.setdefault(row['image'], row)
    rows = list(unique.values())
    selections = {}
    selected_images = set()
    for seed in [42, 123, 2026]:
        indices = np.random.default_rng(seed).permutation(len(rows))[:256]
        selected = [rows[index] for index in indices]
        for n in ([32, 64, 128, 256] if seed == 42 else [128]):
            selections[f'calibration_n{n}_seed{seed}'] = selected[:n]
        selected_images.update(row['image'] for row in selected)
    remainder = [row for row in rows if row['image'] not in selected_images]
    dev_indices = np.random.default_rng(1729).permutation(len(remainder))[:128]
    selections['development_n128'] = [remainder[index] for index in dev_indices]
    selected_images.update(row['image'] for row in selections['development_n128'])
    status = {'status': 'downloading_images', 'source_rows': len(rows), 'required_images': len(selected_images),
              'completed_images': 0, 'selection_ids': {name: [row['id'] for row in selected] for name, selected in selections.items()}}
    def save():
        target = OUT / 'calibration_preparation.json'
        temporary = target.with_suffix('.tmp');temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2));temporary.replace(target)
    save()
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(fetch_image, image) for image in sorted(selected_images)]
        for future in as_completed(futures):
            future.result();status['completed_images'] += 1
            if status['completed_images'] % 10 == 0:
                save()
    for name, selected in selections.items():
        data = []
        for row in selected:
            row = dict(row)
            row['image'] = str(ROOT / row['image'])
            data.append(row)
        (OUT / (name + '.json')).write_text(json.dumps(data, ensure_ascii=False, indent=2))
    status['source_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    status['status'] = 'ready'
    save()


if __name__ == '__main__':
    main()
