"""Report the new seed and a separate four-seed descriptive summary."""
import csv
import json
from pathlib import Path
import statistics
import time

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1] / 'outputs'
OUT = ROOT / BASE.name
PREVIOUS = ROOT / 'lat_awq_general_20261107'
TASKS = ['mmmu_val', 'ai2d', 'scienceqa', 'textvqa_val', 'seedbench', 'mmstar', 'cv_bench', 'vizwiz_vqa_val']


def main():
    rows = {r['variant']: r for r in csv.DictReader((OUT / 'summary.csv').open())}
    old = {r['variant']: r for r in csv.DictReader((PREVIOUS / 'summary.csv').open())}
    assert len(rows) == 8 and all(r['average_8'] for r in rows.values())
    contrasts = []
    combined = []
    for precision in ['w3a16', 'w4a8']:
        settings = {v: rows[f'{precision}_{v}_seed3407'] for v in 'ABCD'}
        for task in [*TASKS, 'average_8']:
            values = {v: float(settings[v][task]) for v in 'ABCD'}
            contrasts.append({'precision': precision, 'task': task, **values,
                **{f'D-{v}': values['D'] - values[v] for v in 'ABC'},
                'D_beats_A_B_C': all(values['D'] > values[v] for v in 'ABC')})
        for setting in 'ABCD':
            entries = [old[f'{precision}_{setting}_seed{s}'] for s in [42, 123, 2026]] + [settings[setting]]
            averages = [float(entry['average_8']) for entry in entries]
            combined.append({'precision': precision, 'variant': setting, 'calibration_seeds': '42,123,2026,3407',
                **{task: statistics.mean(float(entry[task]) for entry in entries) for task in TASKS},
                'average_8_mean': statistics.mean(averages), 'average_8_std': statistics.stdev(averages)})
    for filename, data in [('contrasts_seed3407.csv', contrasts), ('summary_four_seeds.csv', combined)]:
        with (OUT / filename).open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    state = json.loads((OUT / 'status.json').read_text())
    assert all(value['status'] == 'complete' for value in state.values())
    (OUT / 'completion.json').write_text(json.dumps({
        'finished': time.time(), 'calibration_seed': 3407, 'devices': [0, 2],
        'complete_jobs': len(state), 'complete_models': 8, 'all_successful': True,
        'analysis': 'Descriptive new-seed contrasts and a separate four-seed mean/std summary; no new significance claim',
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
