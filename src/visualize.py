from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import ensure_dirs, infer_volume_25d, load_cached_case, load_config, make_or_load_split, project_path
from model import ResidualUNet2D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--ckpt", default="ckpt/best.pt")
    parser.add_argument("--num-cases", type=int, default=6)
    return parser.parse_args()


def save_axial(caseid: str, noisy: np.ndarray, pred: np.ndarray, clean: np.ndarray, mask: np.ndarray, out_path) -> None:
    slices = np.where(mask.mean(axis=(0, 1)) > 0.01)[0]
    k = int(slices[len(slices) // 2]) if slices.size else noisy.shape[2] // 2
    err = np.abs(np.clip(pred[:, :, k], 0, 1) - np.clip(clean[:, :, k], 0, 1))
    m = mask[:, :, k]
    err_vmax = float(np.percentile(err[m], 99)) if np.any(m) else float(np.percentile(err, 99))
    err_vmax = max(err_vmax, 0.02)
    ims = [
        np.clip(noisy[:, :, k], 0, 1),
        np.clip(pred[:, :, k], 0, 1),
        np.clip(clean[:, :, k], 0, 1),
        err,
    ]
    titles = ["noisy", "U-Net denoised", "clean", "|error|"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
    for ax, im, title in zip(axes, ims, titles):
        vmax = err_vmax if title == "|error|" else 1.0
        h = ax.imshow(np.rot90(im), cmap="gray" if title != "|error|" else "magma", vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
        if title == "|error|":
            fig.colorbar(h, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{caseid} axial slice {k}")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_orthogonal(caseid: str, noisy: np.ndarray, pred: np.ndarray, clean: np.ndarray, mask: np.ndarray, out_path) -> None:
    coords = np.argwhere(mask)
    center = coords.mean(axis=0).astype(int) if coords.size else np.asarray(noisy.shape) // 2
    views = [
        ("axial", lambda v: v[:, :, center[2]]),
        ("sagittal", lambda v: v[center[0], :, :]),
        ("coronal", lambda v: v[:, center[1], :]),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(10, 10), constrained_layout=True)
    for r, (view_name, take) in enumerate(views):
        for c, (name, vol) in enumerate([("noisy", noisy), ("U-Net", pred), ("clean", clean)]):
            axes[r, c].imshow(np.rot90(np.clip(take(vol), 0, 1)), cmap="gray", vmin=0, vmax=1)
            axes[r, c].set_title(f"{view_name} {name}")
            axes[r, c].axis("off")
    fig.suptitle(f"{caseid} orthogonal views")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    split = make_or_load_split(cfg)
    cache_dir = project_path(cfg["cache_dir"])
    figs_dir = project_path(cfg["outputs_dir"]) / "figs"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualUNet2D(base=int(cfg["train"]["base_channels"])).to(device)
    ckpt = torch.load(project_path(args.ckpt), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    amp_enabled = bool(cfg["train"]["amp"]) and device.type == "cuda"

    selected = split["test"][: int(args.num_cases)]
    for i, caseid in enumerate(selected):
        c = load_cached_case(cache_dir, caseid)
        noisy = c["noisy"].astype(np.float32)
        clean = c["clean"].astype(np.float32)
        pred = infer_volume_25d(model, noisy, device, batch_slices=int(cfg["eval"]["batch_slices"]), amp=amp_enabled)
        save_axial(caseid, noisy, pred, clean, c["mask"], figs_dir / f"{caseid}_axial.png")
        if i == 0:
            save_orthogonal(caseid, noisy, pred, clean, c["mask"], figs_dir / f"{caseid}_orthogonal.png")
    print(f"Wrote figures to {figs_dir}")


if __name__ == "__main__":
    main()
