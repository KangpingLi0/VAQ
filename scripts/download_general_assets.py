"""Download the public general VLM and calibration annotations with resume support."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import time

ROOT = Path('/data/lkp/VAQ/outputs/lat_awq_general_20261107')
ROOT.mkdir(parents=True, exist_ok=True)
(ROOT / 'download_logs').mkdir(exist_ok=True)
MODEL = Path('/data/lkp/models/Qwen2.5-VL-7B-Instruct')
CALIB = Path('/data/lkp/datasets/sharegpt4v')
MODEL.mkdir(parents=True, exist_ok=True)
CALIB.mkdir(parents=True, exist_ok=True)


def download(url, destination):
    if destination.exists():
        return str(destination)
    temporary = destination.with_name(destination.name + '.download')
    log = ROOT / 'download_logs' / (destination.name + '.log')
    for attempt in range(6):
        actual_url = url + ('&' if '?' in url else '?') + f'download=true&run={int(time.time())}'
        with log.open('a') as handle:
            result = subprocess.run(['curl', '--location', '--fail', '--connect-timeout', '20',
                                     '--max-time', '14400', '--retry', '3', '--continue-at', '-',
                                     '--output', str(temporary), actual_url], stdout=handle, stderr=handle)
        if result.returncode == 0:
            temporary.replace(destination)
            return str(destination)
        time.sleep(min(30, 3 * (attempt + 1)))
    raise RuntimeError(f'Download failed: {destination}; see {log}')


def main():
    model_info = json.loads(Path('/data/lkp/qwen25vl_model_info.json').read_text())
    calib_info = json.loads(Path('/data/lkp/sharegpt4v_info.json').read_text())
    jobs = []
    for item in model_info['siblings']:
        name = item['rfilename']
        if name in ('.gitattributes', 'README.md'):
            continue
        jobs.append((f"https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/{model_info['sha']}/{name}", MODEL / name))
    annotation = 'sharegpt4v_instruct_gpt4-vision_cap100k.json'
    jobs.insert(0, (f"https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/resolve/{calib_info['sha']}/{annotation}", CALIB / annotation))
    metadata = {'model_revision': model_info['sha'], 'calibration_revision': calib_info['sha'],
                'model_dir': str(MODEL), 'calibration_annotation': str(CALIB / annotation),
                'status': 'downloading', 'completed_files': [], 'required_files': len(jobs)}
    status_path = ROOT / 'download_status.json'
    def save():
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        temporary.replace(status_path)
    save()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download, url, target) for url, target in jobs]
        for future in as_completed(futures):
            metadata['completed_files'].append(future.result())
            save()
    metadata['status'] = 'model_and_annotations_downloaded'
    save()


if __name__ == '__main__':
    main()
