import argparse
import math
import os
import random
import sys


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def paired_permutation_pvalue(ours_vals, base_vals, permutations, seed):
    diffs = [o - b for o, b in zip(ours_vals, base_vals)]
    obs = mean(diffs)
    if not diffs or all(math.isclose(d, 0.0) for d in diffs):
        return float("nan"), obs
    rng = random.Random(seed)
    count = 0
    for _ in range(permutations):
        perm_diffs = []
        for o, b in zip(ours_vals, base_vals):
            if rng.random() < 0.5:
                perm_diffs.append(o - b)
            else:
                perm_diffs.append(b - o)
        if abs(mean(perm_diffs)) >= abs(obs):
            count += 1
    return count / permutations, obs


def paired_bootstrap_ci(ours_vals, base_vals, bootstrap, seed):
    n = len(ours_vals)
    if n == 0 or bootstrap <= 0:
        nan = float("nan")
        return (nan, nan), (nan, nan), (nan, nan)
    rng = random.Random(seed)
    ours_boot = []
    base_boot = []
    diff_boot = []
    for _ in range(bootstrap):
        sample_idx = [rng.randrange(n) for _ in range(n)]
        o = [ours_vals[i] for i in sample_idx]
        b = [base_vals[i] for i in sample_idx]
        ours_boot.append(mean(o))
        base_boot.append(mean(b))
        diff_boot.append(mean([ov - bv for ov, bv in zip(o, b)]))
    ours_boot.sort()
    base_boot.sort()
    diff_boot.sort()
    lo = int(0.025 * bootstrap)
    hi = int(0.975 * bootstrap) - 1
    return (ours_boot[lo], ours_boot[hi]), (base_boot[lo], base_boot[hi]), (diff_boot[lo], diff_boot[hi])


def compute_kappa(acc_vals, ns_vals):
    ns_valid = [n for n in ns_vals if isinstance(n, int) and n > 0]
    if len(ns_valid) != len(ns_vals) or not ns_valid:
        return float("nan"), float("nan")
    acc = mean(acc_vals)
    p_chance = sum(1.0 / n for n in ns_valid) / len(ns_valid)
    if p_chance >= 1:
        return p_chance, float("nan")
    kappa = (acc - p_chance) / (1 - p_chance)
    return p_chance, kappa


def build_case_metrics(evaluator, results_file):
    data = evaluator.load_predictions(results_file)
    if not data:
        return {}, {}

    per_case = {}
    per_case_meta = {}
    error_dict = {"id_not_found": 0, "index_parse_error": 0, "index_out_of_bounds": 0}

    for item in data:
        case_id = item["image_id"]
        if case_id not in evaluator.id2images:
            error_dict["id_not_found"] += 1
            continue

        gt_idx = evaluator._parse_image_index(item["label"])
        pred_idx = evaluator._parse_image_index(item["pred"])
        if gt_idx is None or pred_idx is None:
            error_dict["index_parse_error"] += 1
            continue

        imgs = evaluator.id2images[case_id]
        if gt_idx >= len(imgs) or pred_idx >= len(imgs):
            error_dict["index_out_of_bounds"] += 1
            continue

        gt_path = evaluator._norm_path(imgs[gt_idx])
        pred_path = evaluator._norm_path(imgs[pred_idx])

        N = evaluator.id2length.get(case_id, 0)
        sim = float(evaluator.sim_dict.get(gt_path, {}).get(pred_path, 0.0))
        hit = 1.0 if (gt_path == pred_path or sim >= 0.9999) else 0.0

        topk_hits = {}
        for k in (1, 3, 5):
            if N < k:
                topk_hits[k] = None
            else:
                topk_hits[k] = 1.0 if evaluator._is_topk_hit(gt_path, pred_path, k) else 0.0

        per_case[case_id] = {
            "acc": hit,
            "sim": sim,
            "top1": topk_hits[1],
            "top3": topk_hits[3],
            "top5": topk_hits[5],
        }
        per_case_meta[case_id] = {"N": N}

    if any(v > 0 for v in error_dict.values()):
        print("  [错误统计] ", end="")
        for k, v in error_dict.items():
            if v > 0:
                print(f"{k}: {v}", end=" ")
        print("")

    return per_case, per_case_meta


def main():
    parser = argparse.ArgumentParser(
        description="Paired two-sided permutation test for image selection metrics."
    )
    parser.add_argument(
        "--testfile",
        default="/work/data/lxr/code/选图-小模型/数据构建/huaxi数据-同H800/中科院合肥肿瘤医院数据_真实世界_测试集_v2.json",
        help="Test source JSON file.",
    )
    parser.add_argument(
        "--sim_path",
        default="/work/data/lxr/code/选图-小模型/查看相似度/similarity_dict_选图hfcas.json",
        help="Similarity dict JSON file.",
    )
    parser.add_argument(
        "--ours",
        default="/work/data/lxr/code/选图-小模型/数据构建/huaxi数据-同H800/ours结果/中科院合肥肿瘤医院数据_真实世界_测试集_v2_H800结果.jsonl",
        help="Ours results file.",
    )
    parser.add_argument(
        "--medgemma",
        default="/work/data/lxr/code/选图-小模型/数据构建/huaxi数据-同H800/medgemma27B结果/medgemma_27B_中科院合肥肿瘤医院数据_真实世界_测试集_v2_rewrite.json",
        help="MedGemma 27B results file.",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=1000,
        help="Permutation repetitions (default: 1000).",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Bootstrap repetitions for 95%% CI (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    args = parser.parse_args()

    eval_dir = "/work/data/lxr/code/选图-小模型/数据构建/code"
    if eval_dir not in sys.path:
        sys.path.append(eval_dir)

    try:
        from 评测_rewrite import ImageSelectionEvaluator
    except Exception as exc:
        raise SystemExit(f"Failed to import ImageSelectionEvaluator: {exc}") from exc
    SYNC_PATH = "/work/model/xieqiang/推理和评测/ascend_llama_factory_hfcas_union_2022_05_to_2023_test_1122cases_gt.json"

    evaluator = ImageSelectionEvaluator(testfile=args.testfile, sim_path=args.sim_path, xieqiang_gt_path=SYNC_PATH)

    ours_cases, ours_meta = build_case_metrics(evaluator, args.ours)
    base_cases, base_meta = build_case_metrics(evaluator, args.medgemma)

    common_ids = sorted(set(ours_cases) & set(base_cases))
    if not common_ids:
        raise SystemExit("No matching case ids between ours and medgemma.")

    metrics = ["acc", "sim", "top1", "top3", "top5"]
    print(f"Matched cases: {len(common_ids)}")
    print(f"Ours: {os.path.basename(args.ours)}")
    print(f"MedGemma: {os.path.basename(args.medgemma)}")
    print("")

    for metric in metrics:
        ours_vals = []
        base_vals = []
        for case_id in common_ids:
            o = ours_cases[case_id][metric]
            b = base_cases[case_id][metric]
            if o is None or b is None:
                continue
            ours_vals.append(o)
            base_vals.append(b)

        p_val, diff_mean = paired_permutation_pvalue(
            ours_vals, base_vals, args.permutations, args.seed
        )
        ours_ci, base_ci, diff_ci = paired_bootstrap_ci(
            ours_vals, base_vals, args.bootstrap, args.seed
        )
        print(
            f"{metric}: ours_mean={mean(ours_vals):.6f}, "
            f"medgemma_mean={mean(base_vals):.6f}, "
            f"mean_diff={diff_mean:.6f}, p_perm={p_val:.6g}, n={len(ours_vals)}, "
            f"ours_ci95=({ours_ci[0]:.6f},{ours_ci[1]:.6f}), "
            f"medgemma_ci95=({base_ci[0]:.6f},{base_ci[1]:.6f}), "
            f"diff_ci95=({diff_ci[0]:.6f},{diff_ci[1]:.6f})"
        )

    # Kappa (recompute per resample / permutation)
    ours_acc = []
    base_acc = []
    ns_vals = []
    for case_id in common_ids:
        ours_acc.append(ours_cases[case_id]["acc"])
        base_acc.append(base_cases[case_id]["acc"])
        ns_vals.append(int(base_meta[case_id]["N"]))

    _, base_kappa = compute_kappa(base_acc, ns_vals)
    _, ours_kappa = compute_kappa(ours_acc, ns_vals)
    diff_mean = ours_kappa - base_kappa

    if len(ns_vals) > 0 and args.bootstrap > 0:
        rng = random.Random(args.seed)
        base_boot = []
        ours_boot = []
        diff_boot = []
        for _ in range(args.bootstrap):
            sample_idx = [rng.randrange(len(ns_vals)) for _ in range(len(ns_vals))]
            b_acc = [base_acc[i] for i in sample_idx]
            o_acc = [ours_acc[i] for i in sample_idx]
            n_s = [ns_vals[i] for i in sample_idx]
            _, b_k = compute_kappa(b_acc, n_s)
            _, o_k = compute_kappa(o_acc, n_s)
            base_boot.append(b_k)
            ours_boot.append(o_k)
            diff_boot.append(o_k - b_k)
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

    if len(ns_vals) > 0 and args.permutations > 0:
        rng = random.Random(args.seed)
        count = 0
        for _ in range(args.permutations):
            p_base = []
            p_ours = []
            for b, o in zip(base_acc, ours_acc):
                if rng.random() < 0.5:
                    p_base.append(b)
                    p_ours.append(o)
                else:
                    p_base.append(o)
                    p_ours.append(b)
            _, b_k = compute_kappa(p_base, ns_vals)
            _, o_k = compute_kappa(p_ours, ns_vals)
            if abs(o_k - b_k) >= abs(diff_mean):
                count += 1
        p_val = count / args.permutations
    else:
        p_val = float("nan")

    print(
        f"kappa: ours_mean={ours_kappa:.6f}, medgemma_mean={base_kappa:.6f}, "
        f"mean_diff={diff_mean:.6f}, p_perm={p_val:.6g}, n={len(ns_vals)}, "
        f"ours_ci95=({ours_ci[0]:.6f},{ours_ci[1]:.6f}), "
        f"medgemma_ci95=({base_ci[0]:.6f},{base_ci[1]:.6f}), "
        f"diff_ci95=({diff_ci[0]:.6f},{diff_ci[1]:.6f})"
    )


if __name__ == "__main__":
    main()
