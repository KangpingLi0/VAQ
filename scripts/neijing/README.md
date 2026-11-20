# checkpoint-1660 内镜报告量化与评测

运行环境为 `/data/lkp/envs/vaq-neijing`，八张 Ascend 910B2 各运行独立任务。
模型为 `/data/lkp/最后使用版本/checkpoint-1660`；评测函数和关键词来自 `/data/lkp/eval`。

校准集使用全部 512 例、13,642 张图。测试 JSON 实际为 2,989 例、93,809 张图。
全部图像按校准数据分辨率统一为 224×224，使用 checkpoint 的 slow processor，
每例保留全部图像。贪心解码，最多生成 1024 token，repetition_penalty 为 1.05。
嵌入、视觉编码器、lm_head 保持 FP16，仅量化语言解码器的 196 个线性层。

比较 FP16，以及 RTN、SmoothQuant、AWQ、GPTQ、MBQ、QIG、LAT-AWQ 的
W4A16/W3A16/W4A8；AWQ、GPTQ 的本仓库实现仅支持权重量化，排除其 W4A8。
SmoothQuant 按用户要求只运行 W4A8，排除 W4A16/W3A16。
正式比较共 17 个量化配置和 FP16，共 18 组。
W4A16/W3A16 的分组大小为 128；
W4A8 按各方法原实现使用 per-channel weight 和 per-token activation。
MBQ/QIG 开启 reweight，LAT-AWQ 开启 token-aware saliency 和 token-weighted loss。

这是仓库原有伪量化方法，低比特数值以 FP16 张量存储，不能据此宣称打包后的模型大小或 INT4/INT8 推理加速。
每个模型目录的 `quantized_state.pt` 包含参数及 activation wrapper 的 buffer；
再次运行 runner 会先重建所需模块再严格加载 state dict。

```bash
# 创建可重复的数据路径映射，并校验所有图片和病例 ID。
/data/lkp/envs/vaq-neijing/bin/python scripts/neijing/prepare_data.py --hash-weights

# 完整运行 / 中断后的续跑。独占锁阻止重复启动调度器。
bash scripts/neijing/run_all.sh
```

输出目录：`outputs/neijing_checkpoint1660_20261107/`。

- `status.json`：所有任务状态、PID、NPU、命令和日志位置。
- `experiment_plan.json`：完整方法矩阵及排除原因。
- `summary.csv`、`summary.json`、`RESULTS.md`：完成后的指标，逐项更新。
- 各模型目录：量化参数、完整 state dict、逐例预测、医学指标、NLG 指标。
- `data_validation.json`、`input_checksums.json`：数据数量、重叠检查和输入校验和。

调度器先生成完整校准缓存和 FP16 预测。视觉网络未被量化，因此 FP16 阶段保存的
视觉特征可被其他方法复用；原始 input_ids、image_grid_thw 和多模态 RoPE 处理保持一致。
该复用不会跳过任何图片。量化搜索最多同时运行三个，控制主机内存峰值。
每例预测即时写入 JSONL，每个分片完成后核验条数，再执行指标计算。
失败任务会明确保留 failed 状态，依赖任务不会被标为完成。
大批量校准的首层输入捕获在 CPU 上准备完整 attention mask，
随后在 NPU 上以每批 8 例进行层搜索，避免把 512 例的完整 mask 同时放进 HBM。
搜索结束后恢复原始 checkpoint，再应用一次 scale，防止重复缩放。

NLG 直接调用 `/data/lkp/eval/NLG指标计算.py`，使用其 COCO/PTB 分词及
BLEU-1/2/3/4、METEOR、ROUGE-L、CIDEr、SPICE 计算流程。
本地使用 Java 8；pycocoevalcap 的 SPICE 调用移除了 `-cache` 参数，
避免其捆绑的 x86 LMDB 动态库在 aarch64 上加载失败。此更改只关闭缓存。
