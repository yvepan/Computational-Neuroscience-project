# T1 MRI 去噪

本仓库保存 T1 MRI 去噪实验的代码、最终评估结果、可视化图和实验报告。任务目标是从加噪 T1 MRI 预测对应的干净 T1 MRI。

## 链接

- GitHub 仓库：https://github.com/yvepan/Computational-Neuroscience-project
- 模型权重：https://drive.google.com/file/d/1GytKqEMeoEHM6e87vdCDSrrdNKzpEPVe/view?usp=drive_link

## 目录结构

```text
src/                    训练、评估、模型、损失函数和数据集代码
outputs/                最终指标、训练曲线和可视化结果
report/                 实验报告 PDF 与 LaTeX 源文件
config.yaml             实验配置
split.json              固定病例级 train/val/test 划分
requirements.txt        Python 依赖
weights/                权重下载说明
```

## 权重文件

模型权重未直接放入 GitHub 仓库。下载 Google Drive 中的压缩包后，将文件放置为：

```text
ckpt/best.pt
ckpt/last.pt
```

其中 `best.pt` 用于最终评估和出图，`last.pt` 用于断点续训。

## 主要结果

Residual U-Net 在 60 个测试病例上的结果：

```text
MAE  = 0.0057 +/- 0.0011
RMSE = 0.0072 +/- 0.0014
PSNR = 43.01 +/- 1.82 dB
SSIM = 0.9959 +/- 0.0015
```

完整指标表位于：

```text
outputs/metrics_summary.csv
```

实验报告位于：

```text
report/mri_denoising_report.pdf
```

## 复现步骤

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 准备数据：将 T1 MRI 数据（加噪 / 干净 NIfTI 配对）放好后，编辑 `config.yaml` 中的 `data_dir`，使其指向你本地的数据目录。`split.json` 已固定病例级 train/val/test 划分，无需改动即可复现同样的划分。

3. 生成预处理缓存：

```bash
python src/preprocess_cache.py --config config.yaml
```

4. 训练模型（默认在存在 `ckpt/last.pt` 时自动断点续训；加 `--resume none` 从头训练）：

```bash
python src/train.py --config config.yaml
```

5. 若只想复现评估结果，可先按上文“权重文件”一节从 Google Drive 下载权重并放到 `ckpt/best.pt`，再在测试集上评估（生成 `outputs/metrics_*.csv`）：

```bash
python src/evaluate.py --config config.yaml --ckpt ckpt/best.pt
```

6. 生成定性可视化对比图（写入 `outputs/figs/`）：

```bash
python src/visualize.py --config config.yaml --ckpt ckpt/best.pt --num-cases 6
```

7. 汇总统计检验与方法对比图：

```bash
python src/analyze_results.py
```

> 所有脚本均从仓库根目录运行；脚本内部使用 `from common import ...`，因此请保持以 `python src/xxx.py` 的方式调用。

## 说明

预处理缓存未放入仓库，因为体积较大，且可由原始 NIfTI 文件重新生成。训练脚本默认在存在 `ckpt/last.pt` 时自动断点续训。
