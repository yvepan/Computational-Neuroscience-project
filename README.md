# T1 MRI Denoising

This repository contains the code and final results for a T1 MRI denoising
experiment. The task is to map noisy T1 MRI volumes to clean T1 MRI volumes.

## Project Structure

```text
src/                    Training, evaluation, model, loss, and dataset code
outputs/                Final metrics, training curves, and visualisations
report/                 Final report PDF and LaTeX source
config.yaml             Experiment configuration
split.json              Fixed case-level train/val/test split
requirements.txt        Python dependencies
weights/                Checkpoint download note
```

Large model checkpoints are not tracked in this repository. They should be
downloaded separately and placed under:

```text
ckpt/best.pt
ckpt/last.pt
```

## Main Result

Residual U-Net on the 60-case test set:

```text
MAE  = 0.0057 +/- 0.0011
RMSE = 0.0072 +/- 0.0014
PSNR = 43.01 +/- 1.82 dB
SSIM = 0.9959 +/- 0.0015
```

The full result table is available in:

```text
outputs/metrics_summary.csv
```

The report is available in:

```text
report/mri_denoising_report.pdf
```

## Notes

The preprocessed cache is not included because it is large and can be regenerated
from the raw NIfTI files. The training script resumes from `ckpt/last.pt` by
default when the checkpoint exists.
