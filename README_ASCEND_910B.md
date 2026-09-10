# 在多台昇腾 910B 32G 机器上复现

本文说明如何在其他昇腾 910B 32G 机器上复现本仓库的 QIG 量化搜索、伪量化推理和 `lmms-eval` 评测。

> 重要：当前代码是**单进程、单 NPU 卡运行一次任务**，不是跨机器分布式量化。多台机器应各自运行不同方法、数据分片或评测任务；不需要配置 HCCL。QIG 的 `--pseudo_quant` 仍以浮点张量模拟低比特量化，W4/W3 不等于模型只占 4/3 bit 显存。

## 1. 复现基线与已知边界

主仓库基线：

```text
repository: https://github.com/KangpingLi0/VAQ.git
branch:     npu_adapt
commit:     79a3c55
```

本机确认可用的参考软件栈：

```text
CPU architecture: aarch64
OS:               Ubuntu 22.04.5 LTS
NPU:              Ascend 910B2
npu-smi:          25.2.0
CANN toolkit:     8.2.RC1
Python:           3.11.14
torch:            2.6.0
torch_npu:        2.6.0
torchvision:      0.21.0
transformers:     4.57.1
accelerate:       1.11.0
datasets:         4.4.1
qwen-vl-utils:    0.0.14
lmms-eval commit: 3b61454351157d6553a0ea8a57b81e6867809389
```

参考机的 910B2 实际是 64 GiB HBM，并非 32 GiB。因此 32 GiB 上的命令采用保守参数，但目前不能声称所有模型和完整评测已经在 32 GiB 卡上逐项验证。建议先用 Qwen2.5-VL 3B 完成 2 条样本的 smoke test，再逐步增大校准集。7B/8B 的 QIG 搜索在 32 GiB 上不保证可用，72B/26B 等大模型不在单张 32 GiB 卡的复现范围内。

另外，根目录的 `requirements.txt` 是原 CUDA 项目的依赖文件，其中 `torch==2.8.0` 和通用 `triton` 不应直接覆盖当前 Ascend 环境。Ascend 机器必须先按 CANN/torch-npu 兼容矩阵安装成套版本。

## 2. 每台机器需要准备什么

每台机器都需要：

1. 相同的 Ascend 驱动、固件、CANN 和 `torch_npu`/`torch` 版本。
2. 本仓库相同 commit。
3. 指定 commit 的 `lmms-eval`，并应用仓库提供的 NPU 补丁。
4. 完整且相同的模型目录，包括权重、配置、tokenizer、processor 和模型自定义代码。
5. 相同的校准 JSON、图片和评测数据；正式复现时还要保持样本顺序、随机种子、图片分辨率与 prompt 一致。
6. 独立的缓存、日志和输出目录。多台机器不要并发写同一个输出目录或同一个 `.pt` 文件。

模型权重、数据集、Hugging Face 缓存、量化 scale 和日志均被 `.gitignore` 排除，不会随 GitHub 仓库下载，必须单独同步或重新下载。

推荐目录布局如下；路径可以改变，但不要再依赖本机用户名：

```text
${WORK_ROOT}/QIG                         # 代码
${WORK_ROOT}/envs/qig-lmms-py311        # Python 环境（可选）
${WORK_ROOT}/models/qwen2.5-vl-3b       # 模型
${WORK_ROOT}/datasets/calib              # 校准 JSON
${WORK_ROOT}/datasets/images             # 校准/推理图片
${WORK_ROOT}/datasets/eval               # 评测数据
${WORK_ROOT}/cache                       # HF/CANN/pip 等缓存
${WORK_ROOT}/experiments/qig             # scale、日志和结果
```

`scripts/qig/env_qig.sh` 默认使用 `/work/model/lkp`，并主动拒绝 `/home/lkp`。其他机器应先导出自己的 `WORK_ROOT` 和 `REPO_DIR`。

## 3. 系统与 NPU 预检

先确认目标机的架构、驱动和 CANN：

```bash
uname -m
cat /etc/os-release
npu-smi info
cat /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

建议目标机与参考栈完全一致。至少必须保证 CANN、PyTorch 和 torch-npu 是官方兼容组合；只让 `torch==torch_npu` 的版本号看起来相同并不足够。

## 4. 拉取代码和外部依赖

```bash
export WORK_ROOT=/data/qig-reproduce
export REPO_DIR=${WORK_ROOT}/QIG

mkdir -p "${WORK_ROOT}"
git clone --branch npu_adapt --single-branch \
  https://github.com/KangpingLi0/VAQ.git "${REPO_DIR}"
cd "${REPO_DIR}"
git checkout 79a3c55
test "$(git rev-parse HEAD)" = "79a3c5504febf3e6307fcd515fb00672a181b964"

git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git \
  3rdparty/lmms-eval-new
git -C 3rdparty/lmms-eval-new checkout \
  3b61454351157d6553a0ea8a57b81e6867809389
git -C 3rdparty/lmms-eval-new apply --check \
  "${REPO_DIR}/patches/lmms-eval-3b614543-ascend.patch"
git -C 3rdparty/lmms-eval-new apply \
  "${REPO_DIR}/patches/lmms-eval-3b614543-ascend.patch"
```

该补丁包含 Qwen2-VL、Qwen2.5-VL、Qwen3-VL 和 InternVL2 的 NPU 加载兼容修改。不要把本机 benchmark YAML 中写死的 `/work/model/lkp/...` 数据路径复制到新机器；数据路径应在目标机单独配置。

## 5. 准备 Python 环境

### 方案 A：同架构、同 CANN 机器直接打包环境（推荐）

在已经跑通的源机器上：

```bash
conda install -n qig-lmms-py311 conda-pack
conda pack -n qig-lmms-py311 -o qig-lmms-py311.tar.gz
sha256sum qig-lmms-py311.tar.gz > qig-lmms-py311.tar.gz.sha256
```

把两个文件复制到目标机，然后执行：

```bash
cd "${WORK_ROOT}"
sha256sum -c qig-lmms-py311.tar.gz.sha256
mkdir -p "${WORK_ROOT}/envs/qig-lmms-py311"
tar -xzf qig-lmms-py311.tar.gz -C "${WORK_ROOT}/envs/qig-lmms-py311"
"${WORK_ROOT}/envs/qig-lmms-py311/bin/conda-unpack"
export PYTHON_BIN="${WORK_ROOT}/envs/qig-lmms-py311/bin/python"

# conda-pack 中的 editable 安装可能仍记录源机器路径；在目标机重新绑定。
cd "${REPO_DIR}"
"${PYTHON_BIN}" -m pip install --no-deps -e 3rdparty/lmms-eval-new
"${PYTHON_BIN}" -m pip install --no-deps -e .
```

Conda 环境打包不能替代目标机的 Ascend 驱动和 CANN 安装，并且源、目标机器都应是 aarch64。

### 方案 B：重新创建环境

如果不能打包环境，先根据目标机 CANN 版本，从昇腾官方/内部 wheel 源安装匹配的 `torch==2.6.0`、`torch_npu==2.6.0` 和 `torchvision==0.21.0`。确认 NPU 可用后，再安装 Python 依赖：

```bash
conda create -n qig-lmms-py311 python=3.11 -y
conda activate qig-lmms-py311

# 此处先安装与 CANN 8.2.RC1 匹配的 torch/torch_npu/torchvision wheel。
# 不要先执行根目录的 pip install -r requirements.txt。

python -m pip install \
  transformers==4.57.1 accelerate==1.11.0 datasets==4.4.1 \
  evaluate==0.4.6 qwen-vl-utils==0.0.14 numpy==1.26.4 \
  einops==0.8.1 sentencepiece==0.2.1 safetensors==0.7.0 \
  decord easydict imageio seaborn tqdm rouge jieba fuzzywuzzy \
  xopen anthropic packaging ninja wandb loguru hf_transfer

python -m pip install -e 3rdparty/lmms-eval-new
python -m pip install -e .
export PYTHON_BIN="$(command -v python)"
```

安装 `lmms-eval` 后要再次检查 `pip` 没有自动升级/降级 torch 栈。若内部环境已经提供 `torchvision_npu`，也应保持它与 torchvision 和 torch-npu 的版本一致。当前 Ascend 路径不依赖 CUDA `flash-attn` 或通用 Triton。

## 6. 初始化运行目录与环境变量

每次打开新 shell 都执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export WORK_ROOT=/data/qig-reproduce
export REPO_DIR=${WORK_ROOT}/QIG
export PYTHON_BIN=${WORK_ROOT}/envs/qig-lmms-py311/bin/python

cd "${REPO_DIR}"
source scripts/qig/env_qig.sh

export PYTHONPATH="${REPO_DIR}/3rdparty/lmms-eval-new:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=0
export QIG_SCALE_SEARCH_BATCH_SIZE=1
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
```

如果需要联网下载模型/数据，不要设置 offline 变量。完全离线运行时再设置：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
```

验证 Python 确实使用了 NPU，而不是静默回落到 CPU/CUDA：

```bash
"${PYTHON_BIN}" - <<'PY'
import torch
import torch_npu

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("available:", torch.npu.is_available())
print("count:", torch.npu.device_count())
assert torch.npu.is_available()
x = torch.ones(2, device="npu")
print("npu add:", (x + x).cpu())
PY
```

还应确认加载的是刚刚补丁后的 `lmms-eval`：

```bash
"${PYTHON_BIN}" - <<'PY'
import lmms_eval
print(lmms_eval.__file__)
PY
```

输出路径必须位于 `${REPO_DIR}/3rdparty/lmms-eval-new`，否则需要检查 `PYTHONPATH` 或 editable install。

## 7. 同步模型和数据

### 7.1 模型

从 Hugging Face/ModelScope 下载，或从源机器同步完整的模型快照。以 Qwen2.5-VL 3B 为例：

```bash
export MODEL_PATH=${MODEL_ROOT}/qwen2.5-vl-3b
test -f "${MODEL_PATH}/config.json"
test -f "${MODEL_PATH}/tokenizer_config.json"
find "${MODEL_PATH}" -name '*.safetensors' -type f | head
```

不要只复制 `.safetensors`；processor/tokenizer/config 缺失也会导致加载失败。自训练 checkpoint 还必须同步其对应的 processor/tokenizer 文件。

### 7.2 校准数据

校准 JSON 支持 ShareGPT4V/COCO 风格。`image`/`images` 中的相对路径会与 `--image_folder` 拼接：

```json
[
  {
    "id": "calib-0",
    "image": ["images/example.jpg"],
    "conversations": [
      {"from": "human", "value": "<image>\n请描述图片。"},
      {"from": "gpt", "value": "一段参考回答。"}
    ]
  }
]
```

例如上面的文件使用 `--image_folder "${DATA_ROOT}"` 时，图片应位于 `${DATA_ROOT}/images/example.jpg`。

### 7.3 简单推理数据

`inference.py` 不接收独立的 image root，所以 `images` 建议写目标机上的绝对路径：

```json
[
  {
    "id": "eval-0",
    "images": ["/data/qig-reproduce/datasets/images/example.jpg"],
    "question": "请简要描述图片。"
  }
]
```

正式 `lmms-eval` 还需要每个 task 的数据集。在线模式让 `datasets` 写入 `${HF_DATASETS_CACHE}`；离线模式应同步完整 HF cache，或在目标机的 task YAML 中把 `dataset_path` 改成该机器的本地数据集路径。

### 7.4 校验文件一致性

源机器生成清单：

```bash
(cd "${MODEL_PATH}" && find . -type f -print0 | sort -z | xargs -0 sha256sum) \
  > "${WORK_ROOT}/model.sha256"
(cd "${DATA_ROOT}" && find . -type f -print0 | sort -z | xargs -0 sha256sum) \
  > "${WORK_ROOT}/data.sha256"
```

目标机校验：

```bash
(cd "${MODEL_PATH}" && sha256sum -c "${WORK_ROOT}/model.sha256")
(cd "${DATA_ROOT}" && sha256sum -c "${WORK_ROOT}/data.sha256")
```

数据量很大时至少校验模型全部文件、校准 JSON、校准图片清单和已有 scale 文件。

## 8. 32G 单卡最小 smoke test

先使用 3B 模型、2 条校准样本、224×224、每条最多 1 张图：

```bash
export MODEL_PATH=${MODEL_ROOT}/qwen2.5-vl-3b
export CALIB_DATA=${DATA_ROOT}/calib/calib.json
export IMAGE_FOLDER=${DATA_ROOT}
export EVAL_PAIRS=${DATA_ROOT}/eval/eval.json
export RUN_DIR=${EXP_ROOT}/smoke_$(hostname)
export CALIB_CACHE=${RUN_DIR}/calib_n2_224.pt
export SCALE_PATH=${RUN_DIR}/qig_w4a16_n2.pt

mkdir -p "${RUN_DIR}"
cd "${REPO_DIR}"

ASCEND_RT_VISIBLE_DEVICES=0 QIG_SCALE_SEARCH_BATCH_SIZE=1 \
"${PYTHON_BIN}" -W ignore main_quant.py \
  --device npu \
  --model qwen2_5_vl \
  --model_args "pretrained=${MODEL_PATH},torch_dtype=float16,min_pixels=50176,max_pixels=50176" \
  --calib_data coco \
  --data_path "${CALIB_DATA}" \
  --image_folder "${IMAGE_FOLDER}" \
  --n_samples 2 \
  --micro_batch_size 1 \
  --calib_image_size 224 \
  --calib_max_images 1 \
  --calib_cache_path "${CALIB_CACHE}" \
  --method qig \
  --run_process \
  --w_bit 4 \
  --a_bit 16 \
  --w_group 128 \
  --reweight \
  --loss_mode mae \
  --seed 42 \
  --scale_path "${SCALE_PATH}" \
  2>&1 | tee "${RUN_DIR}/quant.log"

test -s "${SCALE_PATH}"
! grep -E 'Traceback|RuntimeError|out of memory|OOM' "${RUN_DIR}/quant.log"
```

然后复用 scale 做两条样本推理：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
"${PYTHON_BIN}" -W ignore inference.py \
  --device npu \
  --model qwen2_5_vl \
  --model_args "pretrained=${MODEL_PATH},torch_dtype=float16,min_pixels=50176,max_pixels=50176" \
  --calib_data none \
  --method qig \
  --pseudo_quant \
  --w_bit 4 \
  --a_bit 16 \
  --w_group 128 \
  --reweight \
  --loss_mode mae \
  --scale_path "${SCALE_PATH}" \
  --infer_pairs "${EVAL_PAIRS}" \
  --save_path "${RUN_DIR}/predictions.json" \
  --max_new_tokens 64 \
  --temperature 0 \
  2>&1 | tee "${RUN_DIR}/inference.log"

test -s "${RUN_DIR}/predictions.json"
```

运行时在另一个终端持续观察：

```bash
watch -n 1 npu-smi info
```

## 9. 从 smoke 扩展到正式校准

通过后先单独生成可复用的 128 条校准缓存：

```bash
export RUN_DIR=${EXP_ROOT}/qwen25vl3b_qig_w4a16_n128
export CALIB_CACHE=${RUN_DIR}/calib_n128_224.pt
export SCALE_PATH=${RUN_DIR}/qig_w4a16_n128.pt
mkdir -p "${RUN_DIR}"

ASCEND_RT_VISIBLE_DEVICES=0 \
"${PYTHON_BIN}" -W ignore main_quant.py \
  --device npu \
  --model qwen2_5_vl \
  --model_args "pretrained=${MODEL_PATH},torch_dtype=float16,min_pixels=50176,max_pixels=50176" \
  --calib_data coco \
  --data_path "${CALIB_DATA}" \
  --image_folder "${IMAGE_FOLDER}" \
  --n_samples 128 \
  --micro_batch_size 1 \
  --calib_image_size 224 \
  --calib_max_images 1 \
  --calib_cache_path "${CALIB_CACHE}" \
  --calib_cache_only \
  2>&1 | tee "${RUN_DIR}/build_calib_cache.log"
```

再用缓存搜索 scale：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 QIG_SCALE_SEARCH_BATCH_SIZE=1 \
"${PYTHON_BIN}" -W ignore main_quant.py \
  --device npu \
  --model qwen2_5_vl \
  --model_args "pretrained=${MODEL_PATH},torch_dtype=float16,min_pixels=50176,max_pixels=50176" \
  --calib_cache_path "${CALIB_CACHE}" \
  --calib_cache_n_samples 128 \
  --method qig \
  --run_process \
  --w_bit 4 \
  --a_bit 16 \
  --w_group 128 \
  --reweight \
  --loss_mode mae \
  --seed 42 \
  --scale_path "${SCALE_PATH}" \
  2>&1 | tee "${RUN_DIR}/quant.log"
```

为了可比较，不要在机器之间改变 `n_samples`、图像尺寸、图片数、模型 checkpoint、校准 JSON 顺序、method 参数或 seed。已有校准缓存只应与生成它的模型和预处理参数配套使用。

## 10. `lmms-eval` 评测

确认目标 task 的数据已经下载或本地 YAML 已正确配置后，一次先跑一个 task：

```bash
export TASK=mmmu_val
export EVAL_DIR=${RUN_DIR}/eval_${TASK}
mkdir -p "${EVAL_DIR}"

ASCEND_RT_VISIBLE_DEVICES=0 \
"${PYTHON_BIN}" -W ignore main.py \
  --device npu \
  --model qwen2_5_vl \
  --model_args "pretrained=${MODEL_PATH},torch_dtype=float16,min_pixels=50176,max_pixels=50176" \
  --tasks "${TASK}" \
  --batch_size 1 \
  --method qig \
  --pseudo_quant \
  --w_bit 4 \
  --a_bit 16 \
  --w_group 128 \
  --reweight \
  --loss_mode mae \
  --scale_path "${SCALE_PATH}" \
  --output_path "${EVAL_DIR}" \
  --log_samples \
  --log_samples_suffix "$(hostname)_qig_w4a16" \
  2>&1 | tee "${EVAL_DIR}/eval.log"
```

先加 `--limit 2` 做 task 级 smoke test，确认结果 JSON 中确实包含 task 和有效样本数，再去掉 `--limit` 跑全量。

## 11. 多台机器如何分工

推荐做法：

- 机器 A 负责唯一的 QIG scale 搜索；完成后对 `.pt` 做 SHA256，并分发给其他机器。
- 机器 B/C/D 各跑不同的 benchmark task，或使用互不重叠的推理 JSON 分片。
- 如果比较 QIG/AWQ/MBQ/RTN，让不同机器运行不同 method，但必须固定模型、校准集和预处理参数。
- 输出目录带上 hostname、method、bit、seed 和 task，避免共享存储覆盖。

scale 可以复用的前提是以下项目完全一致：模型文件、模型 adapter、method、W/A bit、group size、校准数据和顺序、图片预处理、loss/reweight/distort 参数以及代码 commit。不能只看 scale 文件名判断兼容性。

每个 run 建议保存元数据：

```bash
mkdir -p "${RUN_DIR}/metadata"
git rev-parse HEAD > "${RUN_DIR}/metadata/qig_commit.txt"
git -C 3rdparty/lmms-eval-new rev-parse HEAD \
  > "${RUN_DIR}/metadata/lmms_eval_commit.txt"
git -C 3rdparty/lmms-eval-new diff \
  > "${RUN_DIR}/metadata/lmms_eval_applied.patch"
"${PYTHON_BIN}" -m pip freeze > "${RUN_DIR}/metadata/pip_freeze.txt"
npu-smi info > "${RUN_DIR}/metadata/npu_smi.txt"
cp "${CALIB_DATA}" "${RUN_DIR}/metadata/calib.json"
sha256sum "${MODEL_PATH}"/*.safetensors "${CALIB_DATA}" "${SCALE_PATH}" \
  > "${RUN_DIR}/metadata/input_checksums.txt"
```

## 12. 32G 显存调优顺序

发生 OOM 时按以下顺序处理：

1. 确认是 3B 模型并使用 FP16；先不要尝试 7B/8B。
2. 保持 `QIG_SCALE_SEARCH_BATCH_SIZE=1` 和 `--micro_batch_size 1`。
3. 使用 `min_pixels=50176,max_pixels=50176`、`--calib_image_size 224`、`--calib_max_images 1`。
4. smoke 阶段把 `--n_samples`/`--calib_cache_n_samples` 降到 1 或 2。
5. 推理保持 `--batch_size 1`，降低 `--max_new_tokens`。
6. 确认同一张卡没有其他进程占用 HBM。

降低 `w_bit` 通常不会按比例降低本项目的峰值显存，因为这里执行的是伪量化搜索。若 3B、单样本、224×224、batch=1 仍然 OOM，需要记录完整日志和峰值显存，再做 CPU offload/算子级内存优化；当前 NPU adapter 不支持通过 `device_map=auto` 跨卡切分来解决。

## 13. 常见问题

| 现象 | 优先检查 |
| --- | --- |
| `torch.npu.is_available()` 为 false | 是否 source CANN 环境；驱动/CANN/torch-npu 是否匹配；是否用了错误 Python |
| `Unexpected kwargs: {'torch_dtype': ...}` | `lmms-eval` NPU 补丁未应用，或实际 import 了另一个 lmms_eval |
| 日志出现 `cuda` / `device_map` 错误 | 命令缺少 `--device npu`，或 lmms-eval 补丁/PYTHONPATH 不正确 |
| `ModuleNotFoundError: lmms_eval` | 安装 editable 包，或设置本文的 `PYTHONPATH` |
| 模型加载时联网失败 | 模型快照不完整；先下载完整文件，离线时设置 offline 变量 |
| benchmark 找不到数据 | HF cache 未完整同步，或 task YAML 的 `dataset_path` 仍指向其他机器 |
| NPU OOM | 按上一节降低模型、图片尺寸、样本 batch 和生成长度；检查其他进程 |
| scale 存在但结果异常 | 核对 scale 与模型、method、bit、group、calibration 和代码 commit 是否一致 |
| 退出码为 0 但结果不完整 | 检查结果 JSON 的 task、有效样本数，并搜索日志中的 Traceback/OOM |

## 14. 最终验收清单

- [ ] 所有机器的 `npu-smi`、CANN、torch、torch-npu 和关键 Python 包版本一致。
- [ ] 所有机器都在 `npu_adapt` 的同一 commit。
- [ ] `lmms-eval` commit 一致且 NPU 补丁已应用。
- [ ] 模型、校准集、图片、评测集和 scale 的 SHA256 一致。
- [ ] 3B、2 样本、224×224 smoke quant 成功生成非空 scale。
- [ ] smoke inference 生成非空 predictions，日志无 Traceback/OOM。
- [ ] task 级 `--limit 2` 评测成功后再运行全量。
- [ ] 每台机器使用独立输出目录，并保存 commit、环境、命令、日志和 checksum。
- [ ] 汇总结果前核对 task 的有效样本数，而不是只看进程退出码。

本机特定的历史实验、目录和脚本说明仍见 [`docs/QIG_REPRODUCE.md`](docs/QIG_REPRODUCE.md)；在其他机器复现时应优先以本文为准。
