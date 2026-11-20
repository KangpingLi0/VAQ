"""Run the full experiment DAG on eight independent NPUs, with restart support."""
from __future__ import annotations

import argparse
import csv
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from run_experiment import DEFAULT_OUT, load_json, save_json, load_module

RUNNER = Path(__file__).with_name("run_experiment.py")
METHODS = ("rtn", "smoothquant", "awq", "gptq", "mbq", "qig", "lat_awq")
BITS = ((4, 16), (3, 16), (4, 8))


def build_jobs(output, shards):
    jobs = {}
    jobs["calibration"] = {"mode": "calibration", "dependencies": [], "device": True}
    variants = [{"name": "fp16", "method": "fp16", "w": 16, "a": 16}]
    unsupported = []
    excluded = []
    for w, a in BITS:
        for method in METHODS:
            name = f"{method}_w{w}a{a}"
            if a < 16 and method in ("awq", "gptq"):
                unsupported.append({"variant": name, "reason": "repository implementation is weight-only"})
                continue
            if method == "smoothquant" and a >= 16:
                excluded.append({"variant": name, "reason": "user requested SmoothQuant weight-and-activation quantization only"})
                continue
            variants.append({"name": name, "method": method, "w": w, "a": a})
    baseline_jobs = [f"fp16.infer.{shard:02d}" for shard in range(shards)]
    for variant in variants:
        quant_id = variant["name"] + ".quant"
        if variant["method"] != "fp16":
            jobs[quant_id] = {
                "mode": "quant", "dependencies": [] if variant["method"] == "rtn" else ["calibration"],
                "device": True, "variant": variant,
            }
        part_ids = []
        for shard in range(shards):
            job_id = f"{variant['name']}.infer.{shard:02d}"
            part_ids.append(job_id)
            jobs[job_id] = {
                "mode": "infer", "variant": variant, "shard": shard, "shards": shards,
                "dependencies": [] if variant["method"] == "fp16" else [quant_id, *baseline_jobs],
                "device": True,
            }
        jobs[variant["name"] + ".metrics"] = {
            "mode": "metrics", "variant": variant, "dependencies": part_ids, "device": False,
        }
    save_json(output / "experiment_plan.json", {
        "variants": variants, "unsupported": unsupported, "excluded": excluded, "calibration_cases": 512,
        "test_cases": 2989, "image_size": [224, 224], "all_images": True,
        "representation": "fake quantization; weights stored as FP16 tensors",
    })
    return jobs


def completed_artifact(output, job):
    if job["mode"] == "calibration":
        return (output / "calibration.pt").exists() and (output / "calibration.meta.json").exists()
    run = output / job["variant"]["name"]
    if job["mode"] == "quant":
        return (run / "quantized_state.pt").exists() and (run / "quantization.json").exists()
    if job["mode"] == "infer":
        return (run / f"done.part{job['shard']:02d}.json").exists()
    return (run / "medical_metrics.json").exists() and (run / "nlg_metrics.json").exists()


def summarize(output, jobs):
    summary_module = load_module("medical_summary", "/data/lkp/eval/summarize_multi_level_results.py")
    rows = []
    variants = load_json(output / "experiment_plan.json")["variants"]
    for variant in variants:
        run = output / variant["name"]
        if not (run / "medical_metrics.txt").exists():
            continue
        row = summary_module.parse_report(run / "medical_metrics.txt")
        row["model"] = variant["name"]
        row.update(load_json(run / "nlg_metrics.json") if (run / "nlg_metrics.json").exists() else {})
        if (run / "predictions.json").exists():
            predictions = load_json(run / "predictions.json")
            row["empty_reports"] = sum(not x["pred"] for x in predictions)
            row["token_limit_reports"] = sum(x["hit_token_limit"] for x in predictions)
        rows.append(row)
    columns = list(summary_module.FIELDS) + sorted({key for row in rows for key in row} - set(summary_module.FIELDS))
    temporary = output / "summary.csv.tmp"
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output / "summary.csv")
    save_json(output / "summary.json", rows)
    lines = ["# checkpoint-1660 量化评测", "", f"已完成指标汇总：{len(rows)}/{len(variants)}。本表仅列已完成的评测。", "",
             "校准 512 例；测试 2,989 例。全部图片统一为 224×224。",
             "W/A 为仓库的伪量化精度，保存的权重张量仍是 FP16。", "",
             "| 模型 | 例数 | 总体 L1 F1 | 诊断 L1 F1 | 疾病 micro F1 | 疾病 macro F1 | 病例完全匹配 | BLEU-4 | CIDEr |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        keys = ["model", "n", "overall_level1_f1", "conclu_level1_f1", "conclu_level1_disease_micro_f1",
                "conclu_level1_disease_macro_f1", "conclu_level1_case_exact_match", "Bleu_4", "CIDEr"]
        lines.append("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |")
    lines.extend(["", "AWQ、GPTQ 的 W4A8：仓库实现不支持，未运行。",
                  "SmoothQuant 仅评测 W4A8，W4A16、W3A16 按用户要求排除。",
                  "", "任务状态见 status.json，各模型目录保留预测、权重、量化参数和完整指标。"])
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")


class AdoptedProcess:
    """Observe a task left running by an earlier scheduler without restarting it."""

    def __init__(self, process, output, job):
        self.process = process
        self.pid = process.pid
        self.output = output
        self.job = job

    def poll(self):
        try:
            if self.process.is_running() and self.process.status() != psutil.STATUS_ZOMBIE:
                return None
        except psutil.NoSuchProcess:
            pass
        return 0 if completed_artifact(self.output, self.job) else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max-searches", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--reserve", default="", help="NPU:PID pairs for existing trials; release when PID exits")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "scheduler.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (output / "scheduler.pid").write_text(str(os.getpid()) + "\n")
    devices = [int(device) for device in args.devices.split(",")]
    reservations = {int(pair.split(":")[0]): int(pair.split(":")[1])
                    for pair in args.reserve.split(",") if pair}
    jobs = build_jobs(output, args.shards)
    previous = load_json(output / "status.json") if (output / "status.json").exists() else {"jobs": {}}
    state = {}
    running = {}
    for name, job in jobs.items():
        old = previous["jobs"].get(name, {})
        if old.get("status") == "running" and old.get("pid"):
            try:
                process = psutil.Process(old["pid"])
                if process.status() != psutil.STATUS_ZOMBIE and process.cmdline() == old.get("command"):
                    log = open(old["log"], "a", buffering=1)
                    running[name] = (AdoptedProcess(process, output, job), log)
                    state[name] = {**old, "adopted": True}
                    print("adopted", name, "pid", process.pid, "npu", old.get("npu"), flush=True)
                    continue
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        state[name] = {"status": "complete" if completed_artifact(output, job) else "pending"}
    summarize(output, jobs)
    while True:
        for name, (process, log) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            state[name].update(status="complete" if code == 0 and completed_artifact(output, jobs[name]) else "failed",
                               exit_code=code, finished=time.time())
            print(name, state[name]["status"], flush=True)
            del running[name]
            summarize(output, jobs)
        used = {state[name].get("npu") for name in running}
        used.update(device for device, pid in reservations.items() if Path(f"/proc/{pid}").exists())
        searches = sum(jobs[name]["mode"] == "quant" and jobs[name]["variant"]["method"] != "rtn"
                       for name in running)
        metrics_count = sum(jobs[name]["mode"] == "metrics" for name in running)
        candidates = sorted(jobs, key=lambda name: (
            0 if name == "calibration" else
            1 if jobs[name].get("variant", {}).get("method") == "fp16" else
            2 if jobs[name]["mode"] == "metrics" else
            3 if jobs[name]["mode"] == "infer" else 4,
            list(jobs).index(name),
        ))
        for name in candidates:
            job = jobs[name]
            if state[name]["status"] != "pending":
                continue
            dependencies = [state[dependency]["status"] for dependency in job["dependencies"]]
            if any(status in ("failed", "dependency_failed") for status in dependencies):
                state[name]["status"] = "dependency_failed"
                continue
            if any(status != "complete" for status in dependencies):
                continue
            is_search = job["mode"] == "quant" and job["variant"]["method"] != "rtn"
            if is_search and searches >= args.max_searches:
                continue
            if job["mode"] == "metrics" and metrics_count >= 1:
                continue
            free = [device for device in devices if device not in used]
            if job["device"] and not free:
                continue
            device = free[0] if job["device"] else None
            command = [sys.executable, "-u", str(RUNNER), job["mode"], "--output", str(output),
                       "--batch-size", str(args.batch_size)]
            if "variant" in job:
                variant = job["variant"]
                command.extend(["--method", variant["method"], "--w-bit", str(variant["w"]), "--a-bit", str(variant["a"])])
            if job["mode"] == "infer":
                command.extend(["--shard", str(job["shard"]), "--shards", str(job["shards"])])
            if job["mode"] == "metrics":
                command.append("--nlp")
            environment = dict(os.environ)
            environment.update(OMP_NUM_THREADS="8", QIG_SCALE_SEARCH_BATCH_SIZE="8",
                               ASCEND_SLOG_PRINT_TO_STDOUT="0", ASCEND_GLOBAL_LOG_LEVEL="3",
                               TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1")
            environment["JAVA_TOOL_OPTIONS"] = "-XX:ActiveProcessorCount=8"
            if device is not None:
                environment["ASCEND_RT_VISIBLE_DEVICES"] = str(device)
            logfile = output / "logs" / (name + ".log")
            logfile.parent.mkdir(exist_ok=True)
            log = logfile.open("a", buffering=1)
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       cwd=RUNNER.parents[2], env=environment)
            running[name] = (process, log)
            state[name].update(status="running", npu=device, pid=process.pid, started=time.time(),
                               command=command, log=str(logfile))
            used.add(device)
            searches += int(is_search)
            metrics_count += int(job["mode"] == "metrics")
            print("started", name, "npu", device, flush=True)
        counts = {status: sum(job["status"] == status for job in state.values())
                  for status in ("pending", "running", "complete", "failed", "dependency_failed")}
        save_json(output / "status.json", {"updated": time.time(), "counts": counts, "jobs": state})
        if not running and counts["pending"] == 0:
            summarize(output, jobs)
            save_json(output / "completion.json", {"finished": time.time(), "counts": counts,
                      "all_successful": not (counts["failed"] or counts["dependency_failed"])})
            print("finished", counts, flush=True)
            sys.exit(1 if counts["failed"] or counts["dependency_failed"] else 0)
        time.sleep(5)


if __name__ == "__main__":
    main()
