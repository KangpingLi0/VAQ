import argparse
import csv
import re
from pathlib import Path


DEFAULT_REPORTS = [
    "eval/results_eval/vlm_compressor_checkpoint1660.txt",
    "eval/results_eval/qwen2_5vl_7b_lora_huaxi2021_r128_a256_lr2e4_ckpt1270_merged.txt",
    "eval/results_eval/huaxi_qwen2_5vl_gptq_w4a16_checkpoint1660_gptq_w4a16_g128.txt",
    "eval/results_eval/huaxi_qwen2_5vl_awq_w4a16_checkpoint1660_awq_w4a16_g128.txt",
]

DEFAULT_OUTPUT = "eval/results_eval/huaxi_test3100_four_models_multi_level_summary.csv"

FIELDS = [
    "model",
    "n",
    "overall_level1_precision",
    "overall_level1_recall",
    "overall_level1_f1",
    "desc_level1_precision",
    "desc_level1_recall",
    "desc_level1_f1",
    "conclu_level1_precision",
    "conclu_level1_recall",
    "conclu_level1_f1",
    "overall_level2_precision",
    "overall_level2_recall",
    "overall_level2_f1",
    "overall_level3_precision",
    "overall_level3_recall",
    "overall_level3_f1",
    "conclu_level1_disease_micro_sensitivity",
    "conclu_level1_disease_micro_specificity",
    "conclu_level1_disease_micro_f1",
    "conclu_level1_disease_macro_sensitivity",
    "conclu_level1_disease_macro_specificity",
    "conclu_level1_disease_macro_f1",
    "conclu_level1_case_miss_rate",
    "conclu_level1_case_recall_all",
    "conclu_level1_case_exact_match",
    "report",
]

METRIC_RE = re.compile(
    r"^(.*?) - Average Precision: ([0-9.]+), "
    r"Average Recall: ([0-9.]+), Average F1 Score: ([0-9.]+)$"
)
DISEASE_RE = re.compile(
    r"^conclu_level1_disease_(micro|macro) - Sensitivity: ([0-9.]+), "
    r"Specificity: ([0-9.]+), F1 Score: ([0-9.]+)$"
)
CASE_RE = re.compile(
    r"^conclu_level1_case - Miss rate: ([0-9.]+), "
    r"Case Recall@all: ([0-9.]+), Exact Match Accuracy: ([0-9.]+)$"
)


def parse_report(report_path):
    report_path = Path(report_path)
    row = {"model": report_path.stem, "n": "", "report": str(report_path)}

    with report_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("sample_count:"):
                row["n"] = line.split(":", 1)[1].strip()
                continue

            metric_match = METRIC_RE.match(line)
            if metric_match:
                key = metric_match.group(1)
                row[f"{key}_precision"] = metric_match.group(2)
                row[f"{key}_recall"] = metric_match.group(3)
                row[f"{key}_f1"] = metric_match.group(4)
                continue

            disease_match = DISEASE_RE.match(line)
            if disease_match:
                key = f"conclu_level1_disease_{disease_match.group(1)}"
                row[f"{key}_sensitivity"] = disease_match.group(2)
                row[f"{key}_specificity"] = disease_match.group(3)
                row[f"{key}_f1"] = disease_match.group(4)
                continue

            case_match = CASE_RE.match(line)
            if case_match:
                row["conclu_level1_case_miss_rate"] = case_match.group(1)
                row["conclu_level1_case_recall_all"] = case_match.group(2)
                row["conclu_level1_case_exact_match"] = case_match.group(3)

    return row


def print_core_table(rows):
    print(
        "| model | n | overall_l1 F1 | conclu_l1 F1 | overall_l2 F1 | "
        "overall_l3 F1 | disease_micro F1 | disease_macro F1 | case exact |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            "| {model} | {n} | {overall_level1_f1} | {conclu_level1_f1} | "
            "{overall_level2_f1} | {overall_level3_f1} | "
            "{conclu_level1_disease_micro_f1} | "
            "{conclu_level1_disease_macro_f1} | "
            "{conclu_level1_case_exact_match} |".format(**row)
        )


def main():
    parser = argparse.ArgumentParser(
        description="Summarize eval_multi_level txt reports into one CSV."
    )
    parser.add_argument(
        "--reports",
        nargs="*",
        default=DEFAULT_REPORTS,
        help="Report txt files. Defaults to the four huaxi_test3100 model reports.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output CSV path.",
    )
    args = parser.parse_args()

    rows = [parse_report(report) for report in args.reports]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved summary to: {output_path}")
    print_core_table(rows)


if __name__ == "__main__":
    main()
