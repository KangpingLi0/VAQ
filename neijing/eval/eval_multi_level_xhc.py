import argparse
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def load_keyword_data():
    base_dir = "/work/data/public/最终评测方法"
    with open(os.path.join(base_dir, "部位描述关键词.json"), "r", encoding="utf-8") as f:
        keyword_description = json.load(f)
    with open(os.path.join(base_dir, "诊断结论关键词_hzh.json"), "r", encoding="utf-8") as f:
        keyword_conclusion = json.load(f)
    with open(os.path.join(base_dir, "疾病分类关键词.json"), "r", encoding="utf-8") as f:
        keyword_disease = json.load(f)
    return keyword_description, keyword_conclusion, keyword_disease


def parse_medical_report(report):
    report_lines = report.split("\n")
    parsed_info = {}

    for line in report_lines:
        if line.startswith("- "):
            part = line[2:].split("：")[0]
            try:
                description = line[2:].split("：")[1]
            except IndexError:
                continue
            parsed_info[part] = description
        if "诊断结论" in line:
            try:
                diagnosis = line.split("：")[1]
            except IndexError:
                continue
            parsed_info["诊断结论"] = diagnosis

    return parsed_info


def extract_keywords_from_description(parsed_report, keyword_data):
    extracted_keywords_level1 = []
    extracted_keywords_level2 = []
    extracted_keywords_level3 = []

    for part, description in parsed_report.items():
        description_parts = description.split("，")

        for part_class, part_keywords in keyword_data.items():
            if part not in part_class:
                continue

            for keyword_class, keywords in part_keywords.items():
                for keyword, synonym_info in keywords.items():
                    all_synonyms = synonym_info["同义词"] + [keyword]
                    for synonym in all_synonyms:
                        for i, desc in enumerate(description_parts):
                            if "未见" in desc or "无" in desc:
                                continue
                            if "配合不出现" in synonym_info and synonym_info["配合不出现"] in desc:
                                continue
                            if (
                                "配合使用" in synonym_info
                                and synonym_info["配合使用"]
                                and synonym_info["配合使用"] not in description
                            ):
                                continue
                            if synonym in desc:
                                description_parts[i] = desc.replace(synonym, "")
                                if "含义" in synonym_info and synonym_info["含义"]:
                                    extracted_keywords_level1.append(
                                        f"{part}-{synonym_info['含义']}"
                                    )
                                else:
                                    extracted_keywords_level1.append(f"{part}-{keyword}")

                                if keyword_class != "病变类":
                                    extracted_keywords_level2.append(
                                        f"{part}-{keyword_class}"
                                    )
                                else:
                                    if "含义" in synonym_info and synonym_info["含义"]:
                                        extracted_keywords_level2.append(
                                            f"{part}-{synonym_info['含义']}"
                                        )
                                    else:
                                        extracted_keywords_level2.append(f"{part}-{keyword}")

                                extracted_keywords_level3.append(f"{part}-异常")

    return (
        extracted_keywords_level1,
        extracted_keywords_level2,
        extracted_keywords_level3,
    )


def extract_keywords_from_conclusion(parsed_report, keyword_data):
    extracted_keywords_level1 = []
    extracted_keywords_level2 = []
    extracted_keywords_level3 = []

    if "诊断结论" not in parsed_report:
        return extracted_keywords_level1, extracted_keywords_level2, extracted_keywords_level3

    conclusion = parsed_report["诊断结论"]
    conclusion_parts = conclusion.split("；")

    for part_or_disease, part_keywords in keyword_data.items():
        if part_or_disease == "子部位":
            continue
        for keyword_class, keywords in part_keywords.items():
            for keyword, synonym_info in keywords.items():
                all_synonyms = synonym_info["同义词"] + [keyword]
                for synonym in all_synonyms:
                    for i, conclusion_part in enumerate(conclusion_parts):
                        if "未见" in conclusion_part or "无" in conclusion_part:
                            continue
                        if (
                            "配合不出现" in synonym_info
                            and synonym_info["配合不出现"] in conclusion_part
                        ):
                            continue
                        if (
                            "配合使用" in synonym_info
                            and synonym_info["配合使用"]
                            and synonym_info["配合使用"] not in conclusion
                        ):
                            continue
                        if synonym in conclusion_part:
                            conclusion_parts[i] = conclusion_part.replace(synonym, "")
                            if "含义" in synonym_info and synonym_info["含义"]:
                                extracted_keywords_level1.append(
                                    f"诊断结论-{synonym_info['含义']}"
                                )
                            else:
                                extracted_keywords_level1.append(
                                    f"诊断结论-{keyword}"
                                )

                            if keyword_class != "病变类":
                                extracted_keywords_level2.append(
                                    f"诊断结论-{keyword_class}"
                                )
                            else:
                                if "含义" in synonym_info and synonym_info["含义"]:
                                    extracted_keywords_level2.append(
                                        f"诊断结论-{synonym_info['含义']}"
                                    )
                                else:
                                    extracted_keywords_level2.append(
                                        f"诊断结论-{keyword}"
                                    )

                            extracted_keywords_level3.append("诊断结论-异常")

    return (
        extracted_keywords_level1,
        extracted_keywords_level2,
        extracted_keywords_level3,
    )


def calculate_metrics(pred_keywords, label_keywords):
    set_pred = set(pred_keywords)
    set_label = set(label_keywords)

    if not set_pred and not set_label:
        return 1.0, 1.0, 1.0

    tp = len(set_pred.intersection(set_label))
    fp = len(set_pred - set_label)
    fn = len(set_label - set_pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def summarize_confusion_metrics(counts):
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    return sensitivity, specificity, f1


def build_label_set_from_cases(cases):
    labels = set()
    for item in cases:
        labels.update(item["pred"])
        labels.update(item["label"])
    return labels


def compute_label_confusions(labels, cases):
    label_counts = {label: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for label in labels}
    for item in cases:
        pred_set = item["pred"]
        label_set = item["label"]
        for label in labels:
            pred_has = label in pred_set
            true_has = label in label_set
            if pred_has and true_has:
                label_counts[label]["tp"] += 1
            elif pred_has and not true_has:
                label_counts[label]["fp"] += 1
            elif (not pred_has) and true_has:
                label_counts[label]["fn"] += 1
            else:
                label_counts[label]["tn"] += 1
    return label_counts


def compute_micro_macro_metrics(label_counts):
    total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    macro_sens = []
    macro_spec = []
    macro_f1 = []
    for counts in label_counts.values():
        for k in total:
            total[k] += counts[k]
        sens, spec, f1 = summarize_confusion_metrics(counts)
        macro_sens.append(sens)
        macro_spec.append(spec)
        macro_f1.append(f1)

    micro_sens, micro_spec, micro_f1 = summarize_confusion_metrics(total)
    macro_sens_avg = sum(macro_sens) / len(macro_sens) if macro_sens else 0.0
    macro_spec_avg = sum(macro_spec) / len(macro_spec) if macro_spec else 0.0
    macro_f1_avg = sum(macro_f1) / len(macro_f1) if macro_f1 else 0.0
    return (micro_sens, micro_spec, micro_f1), (macro_sens_avg, macro_spec_avg, macro_f1_avg)


def compute_case_level_metrics(cases):
    total = len(cases)
    miss_count = 0
    exact_match_count = 0
    for item in cases:
        pred_set = item["pred"]
        label_set = item["label"]
        if not label_set.issubset(pred_set):
            miss_count += 1
        if pred_set == label_set:
            exact_match_count += 1
    miss_rate = miss_count / total if total > 0 else 0.0
    case_recall_all = (total - miss_count) / total if total > 0 else 0.0
    exact_match_acc = exact_match_count / total if total > 0 else 0.0
    return miss_rate, case_recall_all, exact_match_acc


def group_cases_by_label_count(cases):
    grouped = defaultdict(list)
    for item in cases:
        label_count = len(item["label"])
        if label_count == 0:
            continue
        grouped[label_count].append(item)
    return grouped


def compute_multi_level_metrics(results, keyword_description, keyword_conclusion, keyword_disease):
    all_precision_dic = defaultdict(list)
    all_recall_dic = defaultdict(list)
    all_f1_dic = defaultdict(list)
    conclu_level1_cases = []
    conclu_level2_cases = []

    for case in results:
        pred = case.get("pred", "")
        label = case.get("label", "")

        parse_pred = parse_medical_report(pred)
        desc_pred_l1, desc_pred_l2, desc_pred_l3 = extract_keywords_from_description(
            parse_pred, keyword_description
        )
        conclu_pred_l1, conclu_pred_l2, conclu_pred_l3 = extract_keywords_from_conclusion(
            parse_pred, keyword_conclusion
        )
        disease_pred_l1, disease_pred_l2, _ = extract_keywords_from_conclusion(
            parse_pred, keyword_disease
        )

        parse_label = parse_medical_report(label)
        desc_label_l1, desc_label_l2, desc_label_l3 = extract_keywords_from_description(
            parse_label, keyword_description
        )
        conclu_label_l1, conclu_label_l2, conclu_label_l3 = extract_keywords_from_conclusion(
            parse_label, keyword_conclusion
        )
        disease_label_l1, disease_label_l2, _ = extract_keywords_from_conclusion(
            parse_label, keyword_disease
        )

        conclu_level1_cases.append(
            {"pred": set(disease_pred_l1), "label": set(disease_label_l1)}
        )
        conclu_level2_cases.append(
            {"pred": set(disease_pred_l2), "label": set(disease_label_l2)}
        )

        overall_l1 = calculate_metrics(desc_pred_l1 + conclu_pred_l1, desc_label_l1 + conclu_label_l1)
        desc_l1 = calculate_metrics(desc_pred_l1, desc_label_l1)
        conclu_l1 = calculate_metrics(conclu_pred_l1, conclu_label_l1)
        overall_l2 = calculate_metrics(desc_pred_l2 + conclu_pred_l2, desc_label_l2 + conclu_label_l2)
        overall_l3 = calculate_metrics(desc_pred_l3 + conclu_pred_l3, desc_label_l3 + conclu_label_l3)

        all_precision_dic["overall_level1"].append(overall_l1[0])
        all_recall_dic["overall_level1"].append(overall_l1[1])
        all_f1_dic["overall_level1"].append(overall_l1[2])

        all_precision_dic["desc_level1"].append(desc_l1[0])
        all_recall_dic["desc_level1"].append(desc_l1[1])
        all_f1_dic["desc_level1"].append(desc_l1[2])

        all_precision_dic["conclu_level1"].append(conclu_l1[0])
        all_recall_dic["conclu_level1"].append(conclu_l1[1])
        all_f1_dic["conclu_level1"].append(conclu_l1[2])

        all_precision_dic["overall_level2"].append(overall_l2[0])
        all_recall_dic["overall_level2"].append(overall_l2[1])
        all_f1_dic["overall_level2"].append(overall_l2[2])

        all_precision_dic["overall_level3"].append(overall_l3[0])
        all_recall_dic["overall_level3"].append(overall_l3[1])
        all_f1_dic["overall_level3"].append(overall_l3[2])

    def _clean_data(values):
        return [x for x in values if x is not None and not np.isnan(x)]

    def _tukey_stats(values, label):
        data = _clean_data(values)
        if not data:
            print(f"{label}: no data")
            return
        min_val = np.min(data)
        q1 = np.percentile(data, 25)
        median = np.percentile(data, 50)
        q3 = np.percentile(data, 75)
        max_val = np.max(data)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        whisker_low = np.min([x for x in data if x >= lower], initial=min_val)
        whisker_high = np.max([x for x in data if x <= upper], initial=max_val)
        print(f"{label} - 最小值: {min_val}")
        print(f"{label} - Q1 (25%): {q1}")
        print(f"{label} - 中位数 (50%): {median}")
        print(f"{label} - Q3 (75%): {q3}")
        print(f"{label} - 最大值: {max_val}")
        print(f"{label} - 四分位距(IQR): {iqr}")
        print(f"{label} - Tukey whisker low/high: {whisker_low}, {whisker_high}")

    _tukey_stats(all_f1_dic["overall_level1"], "overall_level1")
    _tukey_stats(all_f1_dic["overall_level2"], "overall_level2")
    _tukey_stats(all_f1_dic["overall_level3"], "overall_level3")

    plot_data = [
        _clean_data(all_f1_dic["overall_level1"]),
        _clean_data(all_f1_dic["overall_level2"]),
        _clean_data(all_f1_dic["overall_level3"]),
    ]
    if any(plot_data):
        os.makedirs("results_eval", exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.boxplot(plot_data, whis=1.5, labels=["L1", "L2", "L3"], showfliers=True)
        ax.set_title("Overall Level F1 (Tukey Whisker)")
        ax.set_ylabel("F1")
        fig.tight_layout()
        fig.savefig(os.path.join("results_eval", "boxplot_overall_levels.png"), dpi=150)
        plt.close(fig)
    # lines = []
    # for key in all_precision_dic.keys():
    #     avg_precision = sum(all_precision_dic[key]) / len(all_precision_dic[key])
    #     avg_recall = sum(all_recall_dic[key]) / len(all_recall_dic[key])
    #     avg_f1 = sum(all_f1_dic[key]) / len(all_f1_dic[key])
    #     lines.append(
    #         f"{key} - Average Precision: {avg_precision:.4f}, "
    #         f"Average Recall: {avg_recall:.4f}, Average F1 Score: {avg_f1:.4f}"
    #     )

    # conclu_level1_labels = build_label_set_from_cases(conclu_level1_cases)
    # conclu_level1_counts = compute_label_confusions(conclu_level1_labels, conclu_level1_cases)
    # conclu_level1_micro, conclu_level1_macro = compute_micro_macro_metrics(conclu_level1_counts)
    # conclu_level1_miss, conclu_level1_case_recall, conclu_level1_exact = compute_case_level_metrics(
    #     conclu_level1_cases
    # )

    # lines.append(
    #     "conclu_level1_disease_micro - Sensitivity: "
    #     f"{conclu_level1_micro[0]:.4f}, Specificity: {conclu_level1_micro[1]:.4f}, "
    #     f"F1 Score: {conclu_level1_micro[2]:.4f}"
    # )
    # lines.append(
    #     "conclu_level1_disease_macro - Sensitivity: "
    #     f"{conclu_level1_macro[0]:.4f}, Specificity: {conclu_level1_macro[1]:.4f}, "
    #     f"F1 Score: {conclu_level1_macro[2]:.4f}"
    # )
    # lines.append(
    #     "conclu_level1_case - Miss rate: "
    #     f"{conclu_level1_miss:.4f}, Case Recall@all: "
    #     f"{conclu_level1_case_recall:.4f}, Exact Match Accuracy: "
    #     f"{conclu_level1_exact:.4f}"
    # )

    # grouped = group_cases_by_label_count(conclu_level1_cases)
    # lines.append("conclu_level1_case_by_gt_count:")
    # bucketed = {1: [], 2: [], 3: [], 4: []}
    # for label_count, cases_group in grouped.items():
    #     if label_count >= 4:
    #         bucketed[4].extend(cases_group)
    #     elif label_count in bucketed:
    #         bucketed[label_count].extend(cases_group)
    # for label_count in (1, 2, 3, 4):
    #     cases_group = bucketed[label_count]
    #     miss, recall_all, exact = compute_case_level_metrics(cases_group)
    #     label_text = ">3" if label_count == 4 else str(label_count)
    #     lines.append(
    #         f"gt_count={label_text} (n={len(cases_group)}) - "
    #         f"Miss rate: {miss:.4f}, Case Recall@all: {recall_all:.4f}, "
    #         f"Exact Match Accuracy: {exact:.4f}"
    #     )

    # lines.append("conclu_level1_disease_per_label:")
    # for label in sorted(conclu_level1_counts.keys()):
    #     counts = conclu_level1_counts[label]
    #     sens, spec, f1 = summarize_confusion_metrics(counts)
    #     lines.append(
    #         f"{label} - Sensitivity: {sens:.4f}, Specificity: {spec:.4f}, "
    #         f"F1 Score: {f1:.4f}, tp: {counts['tp']}, tn: {counts['tn']}, "
    #         f"fp: {counts['fp']}, fn: {counts['fn']}"
    #     )

    # return lines


# def coco_caption_eval(results, annotation_file):
#     from pycocoevalcap.eval import COCOEvalCap
#     from pycocotools.coco import COCO

#     coco = COCO(annotation_file)
#     if results and "caption" in results[0]:
#         coco_results = results
#     else:
#         coco_results = [
#             {"image_id": x.get("image_id"), "caption": x.get("pred", ""), "pred": x.get("pred", "")}
#             for x in results
#         ]
#     coco_result = coco.loadRes(coco_results)
#     coco_eval = COCOEvalCap(coco, coco_result)
#     coco_eval.params["image_id"] = coco_result.getImgIds()
#     coco_eval.evaluate()
#     return coco_eval.eval


# def write_report(lines, results_file):
#     os.makedirs("results_eval", exist_ok=True)
#     base_name = os.path.splitext(os.path.basename(results_file))[0]
#     output_path = os.path.join("results_eval", f"{base_name}.txt")
#     with open(output_path, "w", encoding="utf-8") as f:
#         for line in lines:
#             f.write(line)
#             f.write("\n")
#     return output_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-level metrics.")
    parser.add_argument("--results_file",type=str, default="results_final_final/ours_huaxi_union_2018_2019_2020_2021_test_review_5000cases.json")
    # parser.add_argument(
    #     "annotation_file",
    #     nargs="?",
    #     default=None,
    # )
    args = parser.parse_args()

    with open(args.results_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    keyword_description, keyword_conclusion, keyword_disease = load_keyword_data()
    lines = compute_multi_level_metrics(
        results, keyword_description, keyword_conclusion, keyword_disease
    )

    # if args.annotation_file:
    #     coco_eval = coco_caption_eval(results, args.annotation_file)
    #     lines.append("coco_caption_metrics:")
    #     for key in sorted(coco_eval.keys()):
    #         lines.append(f"{key}: {coco_eval[key]:.6f}")

    # output_path = write_report(lines, args.results_file)
    # for line in lines:
        # print(line)
    # print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()
