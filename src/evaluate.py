from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_nl_means, estimate_sigma
from tqdm import tqdm

from common import (
    ensure_dirs,
    infer_volume_25d,
    load_cached_case,
    load_config,
    load_manifest,
    masked_metrics,
    make_or_load_split,
    project_path,
)
from model import ResidualUNet2D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--ckpt", default="ckpt/best.pt")
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def gaussian_grid_search(val_ids: list[str], cache_dir: Path, sigmas: list[float]) -> float:
    best_sigma, best_psnr = sigmas[0], -float("inf")
    for sigma in sigmas:
        psnrs = []
        for caseid in tqdm(val_ids, desc=f"gaussian sigma={sigma}", leave=False):
            c = load_cached_case(cache_dir, caseid)
            pred = gaussian_filter(c["noisy"].astype(np.float32), sigma=float(sigma))
            psnrs.append(masked_metrics(pred, c["clean"].astype(np.float32), c["mask"])["psnr"])
        score = float(np.mean(psnrs))
        if score > best_psnr:
            best_psnr = score
            best_sigma = float(sigma)
    print(f"Best Gaussian sigma on val: {best_sigma} (PSNR={best_psnr:.3f})")
    return best_sigma


def estimate_slice_sigma(sl: np.ndarray) -> float:
    try:
        return float(np.mean(estimate_sigma(sl, channel_axis=None)))
    except ImportError:
        high = (
            sl[:-2, 1:-1]
            + sl[2:, 1:-1]
            + sl[1:-1, :-2]
            + sl[1:-1, 2:]
            - 4.0 * sl[1:-1, 1:-1]
        )
        return float(np.median(np.abs(high - np.median(high))) / 0.6745 / np.sqrt(20.0))


def nlm_volume(noisy: np.ndarray, patch_size: int, patch_distance: int, h_factor: float) -> np.ndarray:
    out = np.empty_like(noisy, dtype=np.float32)
    for k in range(noisy.shape[2]):
        sl = np.clip(noisy[:, :, k].astype(np.float32), 0, 1)
        sigma = estimate_slice_sigma(sl)
        out[:, :, k] = denoise_nl_means(
            sl,
            h=h_factor * max(sigma, 1e-6),
            sigma=max(sigma, 1e-6),
            patch_size=patch_size,
            patch_distance=patch_distance,
            fast_mode=True,
            preserve_range=True,
            channel_axis=None,
        ).astype(np.float32)
    return out


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    groups = [("all", lambda r: True)]
    for b in ("low", "mid", "high"):
        groups.append((b, lambda r, b=b: r["noise_bin"] == b))
    for method in sorted({r["method"] for r in rows}):
        for group_name, pred in groups:
            subset = [r for r in rows if r["method"] == method and pred(r)]
            if not subset:
                continue
            item = {"method": method, "group": group_name, "n": len(subset)}
            for metric in ("mae", "rmse", "psnr", "ssim"):
                vals = np.asarray([float(r[metric]) for r in subset], dtype=np.float64)
                item[f"{metric}_mean"] = f"{vals.mean():.6f}"
                item[f"{metric}_std"] = f"{vals.std(ddof=1) if len(vals) > 1 else 0.0:.6f}"
                item[f"{metric}_mean_std"] = f"{vals.mean():.4f}+/-{(vals.std(ddof=1) if len(vals) > 1 else 0.0):.4f}"
            summary.append(item)
    return summary


def noise_bins(test_ids: list[str], manifest: dict[str, dict[str, str]]) -> dict[str, str]:
    vals = np.asarray([float(manifest[c]["rician_sigma"]) for c in test_ids], dtype=np.float64)
    q1, q2 = np.quantile(vals, [1 / 3, 2 / 3])
    bins = {}
    for cid, v in zip(test_ids, vals):
        bins[cid] = "low" if v <= q1 else ("mid" if v <= q2 else "high")
    return bins


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    split = make_or_load_split(cfg)
    cache_dir = project_path(cfg["cache_dir"])
    outputs_dir = project_path(cfg["outputs_dir"])
    data_dir = project_path(cfg["data_dir"])
    eval_cfg = cfg["eval"]

    gaussian_sigma = gaussian_grid_search(split["val"], cache_dir, [float(x) for x in eval_cfg["gaussian_sigma_grid"]])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualUNet2D(base=int(cfg["train"]["base_channels"])).to(device)
    ckpt_path = project_path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    amp_enabled = bool(cfg["train"]["amp"]) and device.type == "cuda"

    manifest = load_manifest(data_dir)
    bins = noise_bins(split["test"], manifest)
    rows = []
    methods = ["Noisy", "Gaussian", "NLM", "ResidualUNet"]
    for caseid in tqdm(split["test"], desc="test"):
        c = load_cached_case(cache_dir, caseid)
        noisy = c["noisy"].astype(np.float32)
        clean = c["clean"].astype(np.float32)
        mask = c["mask"]
        preds = {
            "Noisy": noisy,
            "Gaussian": gaussian_filter(noisy, sigma=gaussian_sigma).astype(np.float32),
            "NLM": nlm_volume(
                noisy,
                patch_size=int(eval_cfg["nlm_patch_size"]),
                patch_distance=int(eval_cfg["nlm_patch_distance"]),
                h_factor=float(eval_cfg["nlm_h_factor"]),
            ),
            "ResidualUNet": infer_volume_25d(model, noisy, device, batch_slices=int(eval_cfg["batch_slices"]), amp=amp_enabled),
        }
        for method in methods:
            m = masked_metrics(preds[method], clean, mask)
            rows.append(
                {
                    "caseid": caseid,
                    "method": method,
                    "noise_bin": bins[caseid],
                    "gaussian_sigma": manifest[caseid]["gaussian_sigma"],
                    "rician_sigma": manifest[caseid]["rician_sigma"],
                    "mae": f"{m['mae']:.8f}",
                    "rmse": f"{m['rmse']:.8f}",
                    "psnr": f"{m['psnr']:.8f}",
                    "ssim": f"{m['ssim']:.8f}",
                }
            )

    per_case_fields = ["caseid", "method", "noise_bin", "gaussian_sigma", "rician_sigma", "mae", "rmse", "psnr", "ssim"]
    write_rows(outputs_dir / "metrics_per_case.csv", rows, per_case_fields)
    summary = summarize(rows)
    summary_fields = [
        "method",
        "group",
        "n",
        "mae_mean",
        "mae_std",
        "mae_mean_std",
        "rmse_mean",
        "rmse_std",
        "rmse_mean_std",
        "psnr_mean",
        "psnr_std",
        "psnr_mean_std",
        "ssim_mean",
        "ssim_std",
        "ssim_mean_std",
    ]
    write_rows(outputs_dir / "metrics_summary.csv", summary, summary_fields)
    print(f"Wrote {outputs_dir / 'metrics_per_case.csv'}")
    print(f"Wrote {outputs_dir / 'metrics_summary.csv'}")


if __name__ == "__main__":
    main()
