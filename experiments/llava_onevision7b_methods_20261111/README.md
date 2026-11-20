# LLaVA-OneVision-7B：FP16 与全部指定量化方法

固定使用物理 NPU **1、3、4、5、6、7**。NPU 0、2 留给正在运行的 seed3407 消融。

| 位宽 | 方法 |
|---|---|
| FP16 | 原始模型参照 |
| W3A16 | RTN、AWQ、GPTQ、MBQ、QIG、LAT-AWQ |
| W4A8 | RTN、SmoothQuant、MBQ、QIG、LAT-AWQ |

共 12 组，原八项为 MMMU val、AI2D、ScienceQA、TextVQA val、SEED-Bench image、MMStar、CV-Bench、VizWiz val。

2026-11-11 追加五项：MMBench EN dev/static、MMT-Bench VAL、完整英文 MME-RealWorld、MMVP、NaturalBench test。每组共十三项，合计 **156 项完整评测**。不运行 W4A16，不替换不支持的 AWQ/GPTQ W4A8。

## 新增五项的协议

| 基准 | 划分与主指标 | 额外保留 |
|---|---|---|
| MMBench | EN dev，原 lmms-eval static/CircularEval；主分数已是百分数 | 本地逐样本结果和选项轮换一致性评分 |
| MMT-Bench | OpenGVLab 公开 VAL，原始拼图 TSV；官方 VLMEvalKit exact_matching 的 Overall accuracy | 原始 question/index/category/l2-category 和提取的选项 |
| MME-RealWorld | 完整英文导出，原 lmms-eval accuracy | 原有细分类统计；不替换为 lite |
| MMVP | 固定修正版导出，300 题、150 对；主指标 pair accuracy | 单题 accuracy |
| NaturalBench | 作者的 lmms-eval test 导出，7600 题、1900 个四元组；主指标 G_ACC | Q_ACC、I_ACC、Acc |

所有数据固定 revision 并核对文件大小及 LFS SHA256。NaturalBench 原配置中答案字段 `answer` 与导出字段 `Answer`、指标名 `ACC` 与函数返回的 `Acc` 不一致，已只修正 YAML 字段映射，计分函数及四元组公式保留原实现。MMT-Bench 使用原始 VAL 图像呈现，不换成 MI 版本；提示和提取函数来自保留在 `benchmark_reference/` 的官方 VLMEvalKit 源码。

新增数据由 `prepare_extra_benchmarks.py` 独立下载、检查和定位；每项就绪后单独解锁其 12 个评测任务，不阻塞原八项。MME-RealWorld 的完整数据约 30.9 GB。正式模型仍在下载时，仅先准备较小的 MMVP，其余数据在模型完成后下载，优先保障模型启动。

记录校准图像与新增测试图像的原始字节/RGB 完全重复情况。保留用户要求的完整测试群体；若存在重复，会在准备清单、进度、结果报告和扩展指标 CSV 中明确标记，不静默删题或重新选择校准种子。

## 模型、校准与评测

- 模型为 `llava-hf/llava-onevision-qwen2-7b-ov-hf`，revision `0d50680527681998e456c7b78950205bedd8a068`。四个权重分片均校验 SHA256，所有所需文件检查大小。
- 复用此前已审计的 seed42、128 张 ShareGPT4V COCO train2017 图像及对话选择。重新用 OneVision 的实际视觉编码器、投影器和图像展开流程生成统一校准缓存。需要校准的方法使用同一缓存；RTN 不使用校准数据。不复用 Qwen2.5-VL 的特征。
- 图像处理沿用固定版 HF processor 的原生 `anyres_max_9`，使用官方聊天模板。校准标签仅覆盖助手回答及 EOS；视觉、用户提示和 padding 均不作为回答标签。
- 仅量化实际 Qwen2 解码器的线性层；视觉编码器、多模态投影器、嵌入和 LM head 保留 FP16。`model_bridge.py` 通过共享原始模块的 Qwen2ForCausalLM 视图接入已有量化实现，不转换或复制待量化权重。
- W3A16 分组大小 128；W4A8 使用原实现的 per-channel 权重量化及 per-token 激活量化。MBQ/QIG 开启其 reweight；LAT-AWQ 开启 S/L，IG 32 步，混合系数 0.5。各方法原有损失和搜索逻辑保留。
- 缩放搜索会原位修改权重的方法，在应用搜索结果前恢复原始解码器，再应用一次 scale，避免重复缩放。
- 评测用固定版 lmms-eval 原有任务提示和计分函数、本地同一数据快照；FP16、SDPA、batch=1、greedy、最多 256 新 token，评测 seed42。支持单图、多图和无图题目。
- 保存完整参数和 activation wrapper buffer。每种配置的两例完整 checkpoint 验证通过后才启动正式量化，验证包括严格重载保存的量化权重及实际生成。生成 logits 非有限值会直接报错。

## 输出与恢复

输出目录：`/data/lkp/VAQ/outputs/llava_onevision7b_methods_20261111/`。

- `PROGRESS.md`、`status.json`、`queue_summary.json`：实时任务、PID、卡号和失败状态。
- `experiment_plan.json`、`plan.json`：用户指定的方法矩阵和 192 个含校准/验证的任务。
- `extra_benchmarks.json`、`extra_benchmark_preparation.json`：新增数据的固定版本、下载、样本数与检查状态。
- `summary.csv`、`RESULTS.md`：原八项齐全后计算 Avg8；新增五项单列，不混入 Avg8，缺失不填零。
- `summary_extra_metrics.csv`：保留 MMVP 的双指标及 NaturalBench 的四项指标等扩展结果；MMBench 分数不会重复乘 100。
- 各模型目录保留量化权重、配置、缩放结果和逐样本评测记录；`logs/` 保留全部执行日志。
- `model_info.json`、`source_checksums.json`、`model_integrity.json`、`method_validation.json`：固定 revision、源码校验和及验证记录。

`schedule.py` 使用独占锁，支持接管同命令的运行进程并按完整产物恢复；失败依赖明确标为 dependency_failed。模型未完成下载校验、或微型真实解码器方法验证未通过时，不启动完整模型任务。

所有低比特结果为伪量化精度实验，张量仍以 FP16 保存，不能据此声称部署压缩率或推理加速。
