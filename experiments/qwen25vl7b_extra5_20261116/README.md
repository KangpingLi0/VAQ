# Qwen2.5-VL-7B 新增五项补测

补测 MMBench EN dev/static、MMT-Bench VAL、完整英文 MME-RealWorld、MMVP、NaturalBench test。
覆盖 FP16 及全部现有消融 checkpoint：四种子 42/123/2026/3407 × W3A16/W4A8 × A/B/C/D，合计 33 组、165 项正式评测，另设 FP16/W3A16/W4A8 各一项两例 MMVP 入口验证。

复用原始量化状态，严格重载完整参数及 activation wrapper buffer，不重新量化。原冻结 Qwen 源码复制到独立 source，仅补入上一轮验证过的五项任务配置及计分函数。数据均复用固定快照。MMVP 报告 pair accuracy；NaturalBench 主指标 G_ACC，扩展指标单列；MMBench 原计分已经是百分数，不重复缩放。

评测沿用原协议：Qwen2.5-VL-7B-Instruct、FP16、SDPA、min_pixels=200704/max_pixels=1605632、batch1、greedy/256新 token、评测 seed42。校准种子与评测 seed 分开记录。

物理 NPU 0–7 共享：InternVL 原进程继续运行，每卡至多新增一个 Qwen 补测进程；准入要求至少 34 GiB 空闲 HBM，且该卡没有 InternVL 量化任务。补测进程 allocator 限为总显存的44%。共卡显存不足的补测任务等待 InternVL 释放该卡，再以90%预算独占重试。保留首次失败日志，不删样本、不降低分辨率、不将缺失记为零。

`schedule.py` 独占锁保护，并支持接管存活的同命令进程。资源采集失败时停止新任务准入，已有任务继续。

输出目录 `/data/lkp/VAQ/outputs/qwen25vl7b_extra5_20261116/`：

- `PROGRESS.md`、`status.json`、`queue_summary.json`：运行进度与卡号。
- `summary_extra5.csv`、`summary_extra_metrics.csv`：新增五项及扩展指标。
- `summary_all13.csv`：拼接原八项分数与新增五项，原 Avg8 保持不变。
- `variants.json`、`plan.json`：33 个模型状态路径与逐项命令。
- `resource_snapshot.json`：最近一次八卡资源采样。
- `audit_extra_all_seeds.json`：四种子校准图像与新增评测图像的字节/RGB完全重复检查；审计不删改完整评测集。

FP16 与低比特均通过同一入口、提示和计分。低比特为伪量化精度实验，不据此推断实际部署压缩率或推理加速。
