"""Verify pinned public model files against sizes and LFS SHA256 objects."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

OUT = Path('/data/lkp/VAQ/outputs/lat_awq_general_20261107')
MODEL = Path('/data/lkp/models/Qwen2.5-VL-7B-Instruct')


def verify(item):
    path = MODEL / item['path']
    while not path.exists():time.sleep(5)
    if path.stat().st_size != item['size']:raise ValueError(f'Wrong file size: {path}')
    actual = None
    if 'lfs' in item:
        digest = hashlib.sha256()
        with path.open('rb') as file:
            for chunk in iter(lambda: file.read(16 * 1024 * 1024), b''):digest.update(chunk)
        actual = digest.hexdigest()
        if actual != item['lfs']['oid']:raise ValueError(f'SHA256 mismatch: {path}')
    return {'file': str(path), 'bytes': item['size'], 'lfs_sha256': actual}


def main():
    items = json.loads((OUT / 'model_files.json').read_text())
    items = [item for item in items if item.get('type') == 'file' and item['path'] not in ['README.md', '.gitattributes']]
    with ThreadPoolExecutor(max_workers=4) as pool:verified = list(pool.map(verify, items))
    value = {'status': 'passed', 'revision': 'cc594898137f460bfe9f0759e9844b3ce807cfb5', 'files': verified}
    temporary = OUT / 'model_integrity.tmp';temporary.write_text(json.dumps(value, indent=2));temporary.replace(OUT / 'model_integrity.json')


if __name__ == '__main__':main()
