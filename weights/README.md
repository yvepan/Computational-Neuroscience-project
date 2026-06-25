# Model Checkpoints

Model checkpoints are not committed to this repository because they are large.

Expected local layout after download:

```text
ckpt/best.pt
ckpt/last.pt
```

Use `ckpt/best.pt` for final evaluation and visualisation. Use `ckpt/last.pt`
only when continuing training from the last saved epoch.

The checkpoint files prepared for external upload are:

```text
best.pt
last.pt
SHA256SUMS.txt
```
