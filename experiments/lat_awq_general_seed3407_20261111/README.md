# LAT-AWQ 追加校准种子 3407

使用物理 NPU **0、2**，运行 W3A16、W4A8 的 A/B/C/D 八组消融，每组完整评测原有八项基准。

校准集从与原实验相同的 ShareGPT4V COCO train2017 去重图像列表中，以 `np.random.default_rng(3407).permutation` 固定选取前 128 张。四组、两个位宽共享该缓存。准备脚本复核原始标注校验和、图像完整性，以及全部八项基准的编码字节和解码 RGB 图像重复检查；通过重复检查后才开始量化。

执行原实验的隔离源码 `../lat_awq_general_20261107/source`，仅通过包装入口改变输出路径；IG 步数、缩放网格、S/L 开关、有效 token 掩码、像素范围、解码和计分设置均沿用原实验。3407 是校准和量化种子，评测入口的推理种子仍固定为 42。已有 smoke 和 FP16 参照复用已完成实验。

输出：`/data/lkp/VAQ/outputs/lat_awq_general_seed3407_20261111/`。

- `PROGRESS.md`、`status.json`：实时进度和 PID/NPU。
- `summary.csv`：新种子八组结果，完整八项齐全后计算平均。
- `contrasts_seed3407.csv`：完成后逐项比较 D−A、D−B、D−C。
- `summary_four_seeds.csv`：完成后单独计算四种子的均值和样本标准差。
- `experiment.json`、`source_checksums.json`：配置和源码校验和。
- `logs/`：准备、校准、量化和评测日志。

新结果单独保存，原三种子结果不被覆盖。所有方法均为伪量化精度实验。
