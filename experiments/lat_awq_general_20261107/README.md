# LAT-AWQ 通用模型消融

运行入口：`schedule_general.py`。状态、日志、配置与结果统一写入 `/data/lkp/VAQ/outputs/lat_awq_general_20261107`。
队列只使用物理 NPU 0、2、3、4，内镜队列使用 1、5、6、7。

先完成 FP16 与两个位宽完整方法的小样本运行验证，再执行正式四组消融：

| 组 | token-aware saliency S | token-weighted loss L |
|---|---|---|
| A | 关闭 | 关闭 |
| B | 开启 | 关闭 |
| C | 关闭 | 开启 |
| D | 开启 | 开启 |

W3A16、W4A8 各运行 3 份固定校准子集，种子为 42、123、2026；每份 128 张 COCO train2017 的 ShareGPT4V 图像及对话。
同一份校准缓存由四组复用；IG 32 步、IQR 系数 1.5、混合 λ=0.5、缩放搜索网格一致。
不依据测试分数选择子集、随机种子、检查点或评测指标。

八项基准为 MMMU validation、AI2D test、ScienceQA test、TextVQA validation、SEED-Bench 全部 image 问题、MMStar val、CV-Bench test、VizWiz val。
SEED 图像问题采用官方图像包转为与原适配器相同的字段，报告 `seed_image`。
保留固定版 lmms-eval 的提示词与计分函数，仅将数据加载路径改为本地 parquet。
各组使用 FP16、SDPA、相同动态图像范围（200704–1605632 像素）、batch=1、贪心生成上限 256 token。
`logits_to_keep=1` 避免保存不参与生成的历史位置 logits，对所有组一致。
完整 8 项都有结果后才填写平均值。

隔离源码中的公平性修正：四组的激活统计、IG、IQR、重建损失均使用相同有效视觉与回答 token，排除 padding，每例校准样本等权。
W4A8 量化分支沿用当前实现的 stop-gradient 行为，没有额外引入 STE。
`reweight=False` 对所有组一致，因为该参数未用于当前 LAT-AWQ 的缩放损失。
这些设置可能与已有对比表不同，D 组必须重跑，不能直接复制旧表数值。

模型、数据的固定 revision 记录在下载与准备清单中；每次量化保存 configuration.json、完整缩放结果和搜索日志。
量化权重以 FP16 张量表示（伪量化精度实验），不用于声称实际存储压缩或加速。
负增益、失败与缺失结果都应保留。最终以 3 个校准种子的均值和标准差报告；逐样本预测供配对 bootstrap 检验。

2026-11-07 运行检查修正：W3A16 的 A/B 原先先用 FP16 作差，C/D 先转 FP32 再作差，存在损失算术精度的混杂。
现统一四组使用严格的 FP32 token MSE 实现，L 仅决定权重是否均匀。
受影响的 seed42 A/B 搜索已归档并重跑；C/D 的损失实现不变，W4A8 四组原本已共用同一实现。
归档目录中的搜索记录不进入正式汇总。
