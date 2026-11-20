"""Download and verify one fixed revision of the HF OneVision checkpoint."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
from pathlib import Path
import time

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[1]
OUT = REPO / 'outputs' / BASE.name
MODEL = Path('/data/lkp/models/llava-onevision-qwen2-7b-ov-hf')


def main():
    MODEL.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((OUT / 'model_info.json').read_text())
    spec = importlib.util.spec_from_file_location('original_downloader', REPO / 'scripts/download_general_assets.py')
    downloader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(downloader)
    downloader.ROOT = OUT
    files = [f for f in metadata['siblings'] if f['rfilename'].endswith(('.json', '.txt', '.safetensors'))]
    status = {'status': 'downloading', 'repository': metadata['id'], 'revision': metadata['sha'],
              'required_files': len(files), 'completed_files': [], 'model_dir': str(MODEL)}

    def save():
        temporary = OUT / 'download_status.tmp'
        temporary.write_text(json.dumps(status, indent=2) + '\n')
        temporary.replace(OUT / 'download_status.json')

    def fetch(item):
        name = item['rfilename']
        destination = MODEL / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        downloader.download(f"https://huggingface.co/{metadata['id']}/resolve/{metadata['sha']}/{name}", destination)
        assert destination.stat().st_size == item['size'], f'Unexpected size: {name}'
        expected = item.get('lfs', {}).get('sha256')
        if expected:
            digest = hashlib.sha256()
            with destination.open('rb') as handle:
                for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b''):
                    digest.update(chunk)
            assert digest.hexdigest() == expected, f'Checksum mismatch: {name}'
        print('Verified', name, flush=True)
        return name

    save()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch, item) for item in files]
        for future in as_completed(futures):
            status['completed_files'].append(future.result())
            save()
    status['status'] = 'ready'
    status['finished'] = time.time()
    save()
    (OUT / 'model_integrity.json').write_text(json.dumps({'status': 'passed', 'revision': metadata['sha'],
        'verification': 'All file sizes checked; every weight shard SHA256 matched the pinned Hub manifest'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
