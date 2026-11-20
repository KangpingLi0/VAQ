"""Publish human-readable progress while the experiment scheduler is alive."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

from run_experiment import DEFAULT_OUT


def read_json(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pid", type=int, required=True)
    args = parser.parse_args()
    root = args.output
    while True:
        if not (root / "status.json").exists():
            time.sleep(5)
            continue
        state = read_json(root / "status.json")
        plan = read_json(root / "experiment_plan.json")
        alive = Path(f"/proc/{args.pid}").exists()
        counts = state["counts"]
        complete = not (counts["pending"] or counts["running"])
        status = "全部成功完成" if complete and not counts["failed"] else "运行中" if alive else "调度器已停止，请查看日志"
        if complete and counts["failed"]:
            status = "任务已结束，存在失败项"
        lines = ["# 内镜报告量化评测进度", "",
                 f"状态：{status}。更新时间：{datetime.now(ZoneInfo('Asia/Shanghai')):%Y-%m-%d %H:%M:%S}（北京时间）。", "",
                 "校准：512 例 / 13,642 张图；测试：2,989 例 / 93,809 张图。", "",
                 f"子任务：完成 {counts['complete']}，运行 {counts['running']}，等待 {counts['pending']}，失败 {counts['failed']}。", "",
                 "| 模型 | 量化 | 已生成病例 | 医学指标 | NLG 指标 |",
                 "|---|---|---:|---|---|"]
        for variant in plan["variants"]:
            name = variant["name"]
            folder = root / name
            case_count = 0
            for shard in range(8):
                done = folder / f"done.part{shard:02d}.json"
                progress = folder / f"progress.part{shard:02d}.json"
                if done.exists():
                    case_count += read_json(done)["cases"]
                elif progress.exists():
                    case_count += read_json(progress)["completed"]
            quant = "FP16" if name == "fp16" else state["jobs"][name + ".quant"]["status"]
            medical = "完成" if (folder / "medical_metrics.json").exists() else "等待"
            nlg = "完成" if (folder / "nlg_metrics.json").exists() else "等待"
            lines.append(f"| {name} | {quant} | {case_count}/2989 | {medical} | {nlg} |")
        lines += ["", "AWQ/GPTQ 的 W4A8 在本仓库中不支持，已明确排除。",
                  "SmoothQuant 仅保留 W4A8，W4A16/W3A16 按要求排除。",
                  "低比特为伪量化精度；权重以 FP16 张量保存。", "",
                  "完成的指标逐项写入 [汇总表](summary.csv) 和 [结果报告](RESULTS.md)。"]
        temporary = root / "PROGRESS.md.tmp"
        temporary.write_text("\n".join(lines) + "\n")
        temporary.replace(root / "PROGRESS.md")
        if complete or not alive:
            return
        time.sleep(15)


if __name__ == "__main__":
    main()
