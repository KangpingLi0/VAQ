import argparse
import json
import math
import random
import os

from eval_multi_level import (
    calculate_metrics,
    build_label_set_from_cases,
    compute_label_confusions,
    compute_micro_macro_metrics,
    extract_keywords_from_conclusion,
    extract_keywords_from_description,
    load_keyword_data,
    parse_medical_report,
)


def get_case_id(item):
    for key in ("id", "image_id", "case_id", "sample_id"):
        value = item.get(key)
        if value is not None and value != "":
            return str(value)
    return None


def index_cases(items):
    indexed = {}
    dupes = []
    missing = 0
    for idx, item in enumerate(items):
        case_id = get_case_id(item)
        if case_id is None:
            missing += 1
            case_id = f"__idx__{idx}"
        if case_id in indexed:
            dupes.append(case_id)
            continue
        indexed[case_id] = item
    return indexed, dupes, missing


def compute_case_metrics(item, keyword_description, keyword_conclusion, keyword_disease):
    pred = item.get("pred", "")
    label = item.get("label", "")

    parse_pred = parse_medical_report(pred)
    desc_pred_l1, desc_pred_l2, desc_pred_l3 = extract_keywords_from_description(
        parse_pred, keyword_description
    )
    conclu_pred_l1, conclu_pred_l2, conclu_pred_l3 = extract_keywords_from_conclusion(
        parse_pred, keyword_conclusion
    )
    disease_pred_l1, _, _ = extract_keywords_from_conclusion(parse_pred, keyword_disease)

    parse_label = parse_medical_report(label)
    desc_label_l1, desc_label_l2, desc_label_l3 = extract_keywords_from_description(
        parse_label, keyword_description
    )
    conclu_label_l1, conclu_label_l2, conclu_label_l3 = extract_keywords_from_conclusion(
        parse_label, keyword_conclusion
    )
    disease_label_l1, _, _ = extract_keywords_from_conclusion(parse_label, keyword_disease)

    overall_l1 = calculate_metrics(
        desc_pred_l1 + conclu_pred_l1, desc_label_l1 + conclu_label_l1
    )
    desc_l1 = calculate_metrics(desc_pred_l1, desc_label_l1)
    conclu_l1 = calculate_metrics(conclu_pred_l1, conclu_label_l1)
    overall_l2 = calculate_metrics(
        desc_pred_l2 + conclu_pred_l2, desc_label_l2 + conclu_label_l2
    )
    overall_l3 = calculate_metrics(
        desc_pred_l3 + conclu_pred_l3, desc_label_l3 + conclu_label_l3
    )

    disease_pred_set = set(disease_pred_l1)
    disease_label_set = set(disease_label_l1)
    miss = 1.0 if not disease_label_set.issubset(disease_pred_set) else 0.0
    recall_all = 1.0 - miss
    exact_match = 1.0 if disease_pred_set == disease_label_set else 0.0
    gt_count = len(disease_label_set)

    return {
        "overall_l1": overall_l1,
        "desc_l1": desc_l1,
        "conclu_l1": conclu_l1,
        "overall_l2": overall_l2,
        "overall_l3": overall_l3,
        "disease_pred_set": disease_pred_set,
        "disease_label_set": disease_label_set,
        "case_miss": miss,
        "case_recall_all": recall_all,
        "case_exact_match": exact_match,
        "case_gt_count": gt_count,
    }


def select_metric_index(metric_name):
    metric_name = metric_name.lower()
    if metric_name == "precision":
        return 0
    if metric_name == "recall":
        return 1
    if metric_name == "f1":
        return 2
    raise ValueError(f"Unknown metric: {metric_name}")


def select_micro_macro_index(metric_name):
    metric_name = metric_name.lower()
    if metric_name == "recall":
        return 0  # sensitivity
    if metric_name == "precision":
        return 1  # specificity (closest available)
    if metric_name == "f1":
        return 2
    raise ValueError(f"Unknown metric: {metric_name}")


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def compute_disease_metric(cases, metric_idx, label):
    micro, macro = compute_micro_macro_metrics(
        compute_label_confusions(build_label_set_from_cases(cases), cases)
    )
    return micro[metric_idx] if label == "micro" else macro[metric_idx]


def paired_significance_on_cases(
    baseline_cases, ours_cases, metric_idx, label, args, idx_name
):
    if not baseline_cases:
        print(
            f"{idx_name}: ours_mean=nan, baseline_mean=nan, mean_diff=nan, "
            "p_perm=nan, ours_ci95=(nan,nan), baseline_ci95=(nan,nan), diff_ci95=(nan,nan)"
        )
        return

    base_metric = compute_disease_metric(baseline_cases, metric_idx, label)
    ours_metric = compute_disease_metric(ours_cases, metric_idx, label)
    diff_mean = ours_metric - base_metric

    rng = random.Random(args.seed)
    n = len(baseline_cases)

    if n > 0 and args.bootstrap > 0:
        base_boot = []
        ours_boot = []
        diff_boot = []
        for _ in range(args.bootstrap):
            sample_idx = [rng.randrange(n) for _ in range(n)]
            base_sample = [baseline_cases[i] for i in sample_idx]
            ours_sample = [ours_cases[i] for i in sample_idx]
            b_metric = compute_disease_metric(base_sample, metric_idx, label)
            o_metric = compute_disease_metric(ours_sample, metric_idx, label)
            base_boot.append(b_metric)
            ours_boot.append(o_metric)
            diff_boot.append(o_metric - b_metric)
        base_boot.sort()
        ours_boot.sort()
        diff_boot.sort()
        lo = int(0.025 * args.bootstrap)
        hi = int(0.975 * args.bootstrap) - 1
        base_ci = (base_boot[lo], base_boot[hi])
        ours_ci = (ours_boot[lo], ours_boot[hi])
        diff_ci = (diff_boot[lo], diff_boot[hi])
    else:
        base_ci = (float("nan"), float("nan"))
        ours_ci = (float("nan"), float("nan"))
        diff_ci = (float("nan"), float("nan"))

    if n > 0 and args.permutations > 0:
        count = 0
        for _ in range(args.permutations):
            perm_base = []
            perm_ours = []
            for base_case, ours_case in zip(baseline_cases, ours_cases):
                if rng.random() < 0.5:
                    perm_base.append(base_case)
                    perm_ours.append(ours_case)
                else:
                    perm_base.append(ours_case)
                    perm_ours.append(base_case)
            p_base = compute_disease_metric(perm_base, metric_idx, label)
            p_ours = compute_disease_metric(perm_ours, metric_idx, label)
            if abs(p_ours - p_base) >= abs(diff_mean):
                count += 1
        p_val = count / args.permutations
    else:
        p_val = float("nan")

    print(
        f"{idx_name}: ours_mean={ours_metric:.6f}, baseline_mean={base_metric:.6f}, "
        f"mean_diff={diff_mean:.6f}, p_perm={p_val:.6g}, "
        f"ours_ci95=({ours_ci[0]:.6f},{ours_ci[1]:.6f}), "
        f"baseline_ci95=({base_ci[0]:.6f},{base_ci[1]:.6f}), "
        f"diff_ci95=({diff_ci[0]:.6f},{diff_ci[1]:.6f})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Paired two-sided t-test for multi-level metrics by case id."
    )
    parser.add_argument(
        "baseline_file",
        nargs="?",
        default="results_final_final/medgemma_27B_huaxi_union_2018_2019_2020_2021_test_review_5000cases.json",
        help="Baseline JSON file (default: medgemma).",
    )
    parser.add_argument(
        "ours_file",
        nargs="?",
        default="results_final_final/ours_huaxi_union_2018_2019_2020_2021_test_review_5000cases.json",
        help="Ours JSON file (default: ours).",
    )
    parser.add_argument(
        "--metric",
        choices=["precision", "recall", "f1"],
        default="f1",
        help="Which metric component to test (default: f1).",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Bootstrap repetitions for CI (default: 1000).",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=1000,
        help="Permutation repetitions for paired test (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap/permutation (default: 42).",
    )
    args = parser.parse_args()

    with open(args.baseline_file, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    with open(args.ours_file, "r", encoding="utf-8") as f:
        ours = json.load(f)

    keyword_description, keyword_conclusion, keyword_disease = load_keyword_data()

    baseline_index, baseline_dupes, baseline_missing = index_cases(baseline)
    ours_index, ours_dupes, ours_missing = index_cases(ours)

    common_ids = sorted(set(baseline_index) & set(ours_index))
    if not common_ids:
        raise SystemExit("No matching ids found between the two files.")

    metric_idx = select_metric_index(args.metric)
    metric_keys = ["overall_l1", "desc_l1", "conclu_l1", "overall_l2", "overall_l3"]
    case_metric_keys = [
        "conclu_level1_case_miss_rate",
        "conclu_level1_case_recall_all",
        "conclu_level1_case_exact_match",
    ]
    bucket_metric_keys = []
    for label_count in (1, 2, 3, 4):
        bucket_label = "gt3" if label_count == 4 else str(label_count)
        bucket_metric_keys.extend(
            [
                f"conclu_level1_case_gt_count_{bucket_label}_miss_rate",
                f"conclu_level1_case_gt_count_{bucket_label}_case_recall_all",
                f"conclu_level1_case_gt_count_{bucket_label}_exact_match",
            ]
        )

    all_metric_keys = metric_keys + case_metric_keys + bucket_metric_keys
    baseline_values = {key: [] for key in all_metric_keys}
    ours_values = {key: [] for key in all_metric_keys}
    baseline_disease_cases = []
    ours_disease_cases = []
    label_mismatch = 0
    label_count_mismatch = 0

    for case_id in common_ids:
        base_item = baseline_index[case_id]
        ours_item = ours_index[case_id]
        if base_item.get("label") != ours_item.get("label"):
            label_mismatch += 1

        base_metrics = compute_case_metrics(
            base_item, keyword_description, keyword_conclusion, keyword_disease
        )
        ours_metrics = compute_case_metrics(
            ours_item, keyword_description, keyword_conclusion, keyword_disease
        )

        for key in metric_keys:
            baseline_values[key].append(base_metrics[key][metric_idx])
            ours_values[key].append(ours_metrics[key][metric_idx])

        baseline_values["conclu_level1_case_miss_rate"].append(base_metrics["case_miss"])
        ours_values["conclu_level1_case_miss_rate"].append(ours_metrics["case_miss"])
        baseline_values["conclu_level1_case_recall_all"].append(base_metrics["case_recall_all"])
        ours_values["conclu_level1_case_recall_all"].append(ours_metrics["case_recall_all"])
        baseline_values["conclu_level1_case_exact_match"].append(base_metrics["case_exact_match"])
        ours_values["conclu_level1_case_exact_match"].append(ours_metrics["case_exact_match"])

        base_gt = base_metrics["case_gt_count"]
        ours_gt = ours_metrics["case_gt_count"]
        if base_gt != ours_gt:
            label_count_mismatch += 1

        for metric_prefix, base_value, ours_value in (
            ("miss_rate", base_metrics["case_miss"], ours_metrics["case_miss"]),
            ("case_recall_all", base_metrics["case_recall_all"], ours_metrics["case_recall_all"]),
            ("exact_match", base_metrics["case_exact_match"], ours_metrics["case_exact_match"]),
        ):
            if base_gt > 0:
                bucket = 4 if base_gt >= 4 else base_gt
                bucket_label = "gt3" if bucket == 4 else str(bucket)
                key = f"conclu_level1_case_gt_count_{bucket_label}_{metric_prefix}"
                baseline_values[key].append(base_value)
                ours_values[key].append(ours_value)

        baseline_disease_cases.append(
            {"pred": base_metrics["disease_pred_set"], "label": base_metrics["disease_label_set"]}
        )
        ours_disease_cases.append(
            {"pred": ours_metrics["disease_pred_set"], "label": ours_metrics["disease_label_set"]}
        )

    print(f"Baseline: {os.path.basename(args.baseline_file)}")
    print(f"Ours: {os.path.basename(args.ours_file)}")
    print(f"Matched cases: {len(common_ids)}")
    if baseline_dupes or ours_dupes:
        print(f"Duplicate ids ignored - baseline: {len(baseline_dupes)}, ours: {len(ours_dupes)}")
    if baseline_missing or ours_missing:
        print(
            "Missing ids (used index fallback) - "
            f"baseline: {baseline_missing}, ours: {ours_missing}"
        )
    if label_mismatch:
        print(f"Label mismatches on matched ids: {label_mismatch}")
    if label_count_mismatch:
        print(f"Label count mismatches on matched ids: {label_count_mismatch}")
    print(f"Metric: {args.metric}")
    print("")

    for key in all_metric_keys:
        base_vals = baseline_values[key]
        ours_vals = ours_values[key]
        diffs = [o - b for o, b in zip(ours_vals, base_vals)]
        base_mean = mean(base_vals)
        ours_mean = mean(ours_vals)
        diff_mean = mean(diffs)

        rng = random.Random(args.seed)
        n = len(diffs)

        # Paired bootstrap on case indices
        if n > 0 and args.bootstrap > 0:
            base_boot = []
            ours_boot = []
            diff_boot = []
            for _ in range(args.bootstrap):
                sample_idx = [rng.randrange(n) for _ in range(n)]
                base_sample = [base_vals[i] for i in sample_idx]
                ours_sample = [ours_vals[i] for i in sample_idx]
                base_boot.append(mean(base_sample))
                ours_boot.append(mean(ours_sample))
                diff_boot.append(mean([o - b for o, b in zip(ours_sample, base_sample)]))
            base_boot.sort()
            ours_boot.sort()
            diff_boot.sort()
            lo = int(0.025 * args.bootstrap)
            hi = int(0.975 * args.bootstrap) - 1
            base_ci = (base_boot[lo], base_boot[hi])
            ours_ci = (ours_boot[lo], ours_boot[hi])
            diff_ci = (diff_boot[lo], diff_boot[hi])
        else:
            base_ci = (float("nan"), float("nan"))
            ours_ci = (float("nan"), float("nan"))
            diff_ci = (float("nan"), float("nan"))

        # Paired permutation test on mean difference
        if n > 0 and args.permutations > 0:
            count = 0
            for _ in range(args.permutations):
                perm_diffs = []
                for o, b in zip(ours_vals, base_vals):
                    if rng.random() < 0.5:
                        perm_diffs.append(o - b)
                    else:
                        perm_diffs.append(b - o)
                perm_mean = mean(perm_diffs)
                if abs(perm_mean) >= abs(diff_mean):
                    count += 1
            p_val = count / args.permutations
        else:
            p_val = float("nan")

        print(
            f"{key}: ours_mean={ours_mean:.6f}, baseline_mean={base_mean:.6f}, "
            f"mean_diff={diff_mean:.6f}, p_perm={p_val:.6g}, "
            f"ours_ci95=({ours_ci[0]:.6f},{ours_ci[1]:.6f}), "
            f"baseline_ci95=({base_ci[0]:.6f},{base_ci[1]:.6f}), "
            f"diff_ci95=({diff_ci[0]:.6f},{diff_ci[1]:.6f})"
        )

    micro_macro_idx = select_micro_macro_index(args.metric)
    for label, idx_name in (
        ("micro", "conclu_level1_disease_micro"),
        ("macro", "conclu_level1_disease_macro"),
    ):
        paired_significance_on_cases(
            baseline_disease_cases,
            ours_disease_cases,
            micro_macro_idx,
            label,
            args,
            idx_name,
        )

    for metric_name, metric_idx in (("sensitivity", 0), ("specificity", 1)):
        for label, idx_name in (
            ("micro", f"conclu_level1_disease_micro_{metric_name}"),
            ("macro", f"conclu_level1_disease_macro_{metric_name}"),
        ):
            paired_significance_on_cases(
                baseline_disease_cases,
                ours_disease_cases,
                metric_idx,
                label,
                args,
                idx_name,
            )


if __name__ == "__main__":
    main()
