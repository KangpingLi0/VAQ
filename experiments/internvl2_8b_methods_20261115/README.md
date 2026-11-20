# InternVL2-8B：八卡 FP16 与全部指定量化方法

物理 Ascend NPU **0–7**，独立调度队列。上一轮 LLaVA 和 seed3407 已完成。

| 位宽 | 方法 |
|---|---|
| FP16 | 原始模型 |
| W3A16 | RTN、AWQ、GPTQ、MBQ、QIG、LAT-AWQ |
| W4A8 | RTN、SmoothQuant、MBQ、QIG、LAT-AWQ |

共 **12 组 × 13 项 = 156 项正式评测**。含校准、两例量化及生成验证共 192 个任务。不运行 W4A16。

评测集沿用上一轮全部固定快照：MMMU val、AI2D、ScienceQA、TextVQA val、SEED-Bench image、MMStar、CV-Bench、VizWiz val，以及 MMBench EN dev/static、MMT-Bench VAL、完整英文 MME-RealWorld、MMVP、NaturalBench test。

Avg8 只计算原八项，新增五项单独报告。MMBench 原指标已是百分数，不重复乘 100；MMVP 主指标为配对准确率，NaturalBench 主指标为 G_ACC，扩展指标保存在 summary_extra_metrics.csv。缺失或失败不填零。

## 模型及协议

- 官方模型 [OpenGVLab/InternVL2-8B](https://huggingface.co/OpenGVLab/InternVL2-8B)，固定 revision `6fb9ad6924f69424e57fab2ab061d707688f0296`。四个权重分片和 tokenizer.model 检查 SHA256，全部必需文件检查大小；官方 Python 代码按该 revision 下载，不修改。
- 原生 InternViT 与 InternLM2；448 像素，官方动态分块 min1/max12，加缩略图，ImageNet 标准化，官方 internlm2-chat 模板。校准与评测使用同一图像处理。
- FP16、原生 eager attention、batch1、greedy、最多 256 个新 token、评测 seed42。支持单图、多图和纯文本。
- 固定此前已审计的 seed42、128 张 ShareGPT4V COCO train2017 图像和对话；用本模型重新生成多模态嵌入，所有校准方法共享缓存；RTN 不用校准。标签仅覆盖助手回答及结束符，排除图像、提示和 padding。
- 只量化 InternLM2 解码层的 5 个线性模块：attention.wqkv、attention.wo、feed_forward.w1/w2/w3。视觉编码器、投影器、词嵌入及输出头保持 FP16。
- W3A16 分组 128；W4A8 沿用 per-channel 权重、per-token 激活。保留仓库方法原有搜索、损失和掩码；MBQ/QIG 开启 reweight；LAT-AWQ 开启 S/L，IG32，混合系数 0.5。
- AWQ/MBQ/QIG/LAT-AWQ 搜索后先恢复原始解码器，再应用一次结果，避免重复缩放。保存完整权重与 activation wrapper buffer，评测严格重载。

官方 2024 年 InternLM2 代码使用 tuple KV 缓存，与本环境 Transformers 4.57 的新生成接口不同。`multimodal.py` 用不变的原生 forward 和 tuple 缓存逐 token 贪心生成，保持官方模板结束符；仅投影最后一个 token，以减少无用 logits。测试核对带缓存生成与完整无缓存前向逐 token 一致，并核对多模态嵌入、logits 和 loss 与官方模型 forward 一致。未替换注意力公式。

2026-11-15 显存修复：梯度收集时暂时将无 dropout 的原生解码器设为 train，使官方逐层 gradient checkpointing 真正生效，结束前向后恢复原模式。新任务的 scale-search micro-batch 设为 1，仍使用全部 128 例。`memory_compat.py` 在无梯度长序列前向中按 256 个 query 分块计算原生 eager attention；每个 query 仍关注完整 K/V，保留 FP32 softmax、原始掩码和 KV 缓存，不截断图像或文本。损失、输入梯度、带 padding 的 prefill 和缓存解码 logits 已通过两项回归测试。原始失败历史保存在 `status.before_memory_fix_20261115.json` 和 `recovery_memory_20261115.json`，完整模型重试单独观察。

## 校验、启动及恢复

`test_bridge.py` 用两层原生 InternLM2 检查多模态输入/标签、单图/多图/纯文本缓存生成一致性、量化范围及完整状态重载。`validate_methods.py` 在 NPU 上逐项运行 11 个真实量化路径；`integration_check.py` 验证实际 CLI、校准、RTN、生成、计分及样本输出。

独立合成小模型只写入 `integration_only/`，其分数永不计入正式结果。完成模型哈希和两类适配校验后，调度器自动开始完整模型任务；每个量化配置的两例完整模型量化、严格重载和生成成功后，才启动对应正式量化。

调度器用独占锁、按完整产物恢复并接管同命令的存活进程，固定分配 NPU 0–7。模型下载保留完整的 HTTP Range 分块，可续传。数据和 Arrow 缓存位于 /data 卷，避免占满系统盘。

主要输出位于 `/data/lkp/VAQ/outputs/internvl2_8b_methods_20261115/`：

- `PROGRESS.md`、`status.json`、`queue_summary.json`：进度、运行 PID、卡号和错误。
- `summary.csv`、`RESULTS.md`、`summary_extra_metrics.csv`：完整评测与扩展指标。
- `experiment_plan.json`、`plan.json`：方法矩阵及逐项任务命令。
- `model_info.json`、`model_integrity.json`、`source_checksums.json`：模型版本、校验及本地源码散列。
- `logs/`、各方法目录：量化配置、scale、完整状态和逐样本结果。

所有低比特配置均为伪量化精度实验，保存张量仍为 FP16；这些结果不代表真实部署的压缩率或推理加速。
