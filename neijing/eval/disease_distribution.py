import argparse
import json
from collections import Counter

from eval.eval_multi_level import load_keyword_data, parse_medical_report, extract_keywords_from_conclusion


def extract_assistant_report(messages):
    for msg in messages:
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def count_diseases(data, keyword_disease, use_level=1, per_case_unique=True):
    counter = Counter()
    for item in data:
        report = extract_assistant_report(item.get("messages", []))
        if not report:
            continue
        parsed = parse_medical_report(report)
        level1, level2, _ = extract_keywords_from_conclusion(parsed, keyword_disease)
        labels = level1 if use_level == 1 else level2
        if per_case_unique:
            labels = set(labels)
        counter.update(labels)
    return counter


def main():
    parser = argparse.ArgumentParser(description="Count disease distribution using eval_multi_level logic.")
    parser.add_argument(
        "--input",
        default="/work/data/xhc/code/infer_any_thing/llama_factory_huaxi_union_2018_2019_2020_2021_train_sampled_review_42506cases.json",
        help="Path to JSON list file",
    )
    parser.add_argument("--level", type=int, choices=[1, 2], default=1, help="Disease label level")
    parser.add_argument(
        "--count-all",
        action="store_true",
        help="Count all occurrences (not per-case unique)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=50,
        help="Print top N diseases (0 to print all)",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional output path to save full counts as JSON",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    _, _, keyword_disease = load_keyword_data()

    counter = count_diseases(
        data,
        keyword_disease,
        use_level=args.level,
        per_case_unique=not args.count_all,
    )

    items = counter.most_common() if args.top == 0 else counter.most_common(args.top)
    for label, count in items:
        print(f"{label}\t{count}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(dict(counter), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
