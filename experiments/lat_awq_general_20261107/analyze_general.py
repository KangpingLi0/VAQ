"""Report every core ablation, three calibration seeds and paired uncertainty."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from schedule_general import KEYS, OUT, SOURCE, TASKS, result

SEEDS = [42, 123, 2026]
CONTRASTS = {'B-A': [-1, 1, 0, 0], 'C-A': [-1, 0, 1, 0],
             'D-A': [-1, 0, 0, 1], 'D-B': [0, -1, 0, 1],
             'D-C': [0, 0, -1, 1], 'interaction_D-B-C+A': [1, -1, -1, 1]}


def samples(variant, task):
    job = {'eval_dir': str(OUT / variant / 'eval' / task), 'task': task}
    found = result(job)
    if not found:raise FileNotFoundError(f'Missing result: {variant}/{task}')
    path, aggregate = found
    sample_path = path.with_name(path.name.replace('_results.json', f'_samples_{task}.jsonl'))
    rows = [json.loads(line) for line in sample_path.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda row: row['doc_id'])
    if len({row['doc_id'] for row in rows}) != len(rows):raise ValueError('Duplicate document IDs')
    return rows, aggregate['results'][task][KEYS[task]]


def case_score(task, value):
    if task == 'mmmu_val':
        from lmms_eval.tasks.mmmu.utils import evaluate_mmmu
        return evaluate_mmmu([value])[1]['acc']
    if task == 'mmstar':return value['score']
    if task == 'cv_bench':return float(value['is_correct'])
    if task == 'seedbench':return float(value['pred'].lower().strip() == value['answer'].lower().strip())
    return float(value)


def task_data(w, a, task):
    reference = None;matrix = []
    metric = KEYS[task].split(',')[0]
    strata = None
    for seed in SEEDS:
        settings = []
        for setting in 'ABCD':
            variant = f'w{w}a{a}_{setting}_seed{seed}'
            rows, aggregate = samples(variant, task)
            # Upstream doc_hash can contain PIL repr memory addresses. Pair by the
            # fixed dataset fingerprint/row ID and verify target + rendered input.
            identities = [(row['doc_id'], row['target'], row.get('input')) for row in rows]
            if reference is None:
                reference = identities
                strata = [row[metric]['l2_category'] for row in rows] if task == 'mmstar' else ['all'] * len(rows)
            if identities != reference:raise ValueError(f'Unpaired cases: {variant}/{task}')
            scores = np.array([case_score(task, row[metric]) for row in rows])
            groups = [np.where(np.array(strata) == label)[0] for label in sorted(set(strata))]
            reconstructed = np.mean([scores[group].mean() for group in groups])
            if abs(reconstructed - aggregate) > 1e-4:
                raise ValueError(f'Per-case aggregation disagrees: {variant}/{task}: {reconstructed} vs {aggregate}')
            settings.append(scores)
        matrix.append(settings)
    return np.array(matrix), groups


def bootstrap(matrix, groups, rng, iterations):
    # matrix=[calibration seeds, cases]. Resample the same test cases for all
    # seeds, within MMStar's official macro-average strata.
    distribution = np.zeros((iterations, len(SEEDS)))
    for group in groups:
        for start in range(0, iterations, 32):
            size = min(32, iterations - start)
            indices = rng.choice(group, size=(size, len(group)), replace=True)
            distribution[start:start + size] += matrix[:, indices].mean(axis=2).T / len(groups)
    return distribution


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--iterations', type=int, default=10000)
    args = parser.parse_args();rng = np.random.default_rng(1729)
    output = {'bootstrap_iterations': args.iterations, 'ci': 'descriptive 95% percentile intervals; no multiple-comparison significance claim',
        'resampling': 'paired calibration-seed and benchmark-case bootstrap; shared seed draws across 8 tasks; MMStar stratified by l2 category',
        'calibration_seeds': SEEDS, 'precision': {}}
    for w, a in [(3, 16), (4, 8)]:
        datasets = {task: task_data(w, a, task) for task in TASKS}
        precision = {}
        seed_draws = rng.integers(0, len(SEEDS), size=(args.iterations, len(SEEDS)))
        for name, coefficients in CONTRASTS.items():
            coefficients = np.array(coefficients)
            distributions = [];per_task = {};seed_deltas = []
            for task, (data, groups) in datasets.items():
                differences = (data * coefficients[None, :, None]).sum(axis=1)
                by_seed = np.mean([differences[:, group].mean(axis=1) for group in groups], axis=0) * 100
                distribution = bootstrap(differences, groups, rng, args.iterations) * 100
                distribution = np.take_along_axis(distribution, seed_draws, axis=1).mean(axis=1)
                per_task[task] = {'mean_delta_pp': float(by_seed.mean()), 'seed_std_pp': float(by_seed.std(ddof=1)),
                                  'per_seed_delta_pp': by_seed.tolist(), 'paired_ci95_pp': np.quantile(distribution, [.025, .975]).tolist()}
                distributions.append(distribution);seed_deltas.append(by_seed)
            by_seed = np.array(seed_deltas).mean(axis=0)
            distribution = np.array(distributions).mean(axis=0)
            precision[name] = {'tasks': per_task, 'average_8': {'mean_delta_pp': float(by_seed.mean()),
                'seed_std_pp': float(by_seed.std(ddof=1)), 'per_seed_delta_pp': by_seed.tolist(),
                'paired_ci95_pp': np.quantile(distribution, [.025, .975]).tolist()}}
        output['precision'][f'w{w}a{a}'] = precision
    (OUT / 'paired_analysis.json').write_text(json.dumps(output, ensure_ascii=False, indent=2))
    rows = [{'precision': precision, 'contrast': contrast, **data['average_8']} for precision, comparisons in output['precision'].items() for contrast, data in comparisons.items()]
    with (OUT / 'paired_analysis.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


if __name__ == '__main__':main()
