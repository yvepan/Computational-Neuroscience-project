"""Post-hoc analysis: paired significance tests + summary figures from metrics_per_case.csv."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from common import load_config, project_path

METHODS = ["Noisy", "Gaussian", "NLM", "ResidualUNet"]
MAIN = "ResidualUNet"
BINS = ["low", "mid", "high"]
COLORS = {"Noisy": "#9e9e9e", "Gaussian": "#ef6c00", "NLM": "#1e88e5", "ResidualUNet": "#2e7d32"}


def load_per_case(path: Path):
    by_method = defaultdict(dict)        # method -> {caseid -> row}
    bin_of = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_method[r["method"]][r["caseid"]] = r
            bin_of[r["caseid"]] = r["noise_bin"]
    return by_method, bin_of


def paired_array(by_method, m, metric, cases):
    return np.array([float(by_method[m][c][metric]) for c in cases], dtype=np.float64)


def run_tests(by_method, outputs_dir: Path):
    cases = sorted(by_method[MAIN].keys())
    rows = []
    for metric in ("psnr", "ssim", "rmse", "mae"):
        u = paired_array(by_method, MAIN, metric, cases)
        for base in ("Noisy", "Gaussian", "NLM"):
            b = paired_array(by_method, base, metric, cases)
            diff = u - b
            t_stat, t_p = stats.ttest_rel(u, b)
            try:
                w_stat, w_p = stats.wilcoxon(u, b)
            except ValueError:
                w_stat, w_p = float("nan"), float("nan")
            dz = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else float("nan")
            rows.append({
                "metric": metric.upper(),
                "comparison": f"{MAIN} vs {base}",
                "mean_diff": f"{diff.mean():+.4f}",
                "std_diff": f"{diff.std(ddof=1):.4f}",
                "cohen_dz": f"{dz:.3f}",
                "wins_n": f"{int((diff > 0).sum())}/{len(diff)}" if metric in ("psnr", "ssim") else f"{int((diff < 0).sum())}/{len(diff)}",
                "t_p": f"{t_p:.2e}",
                "wilcoxon_p": f"{w_p:.2e}",
            })
    fields = ["metric", "comparison", "mean_diff", "std_diff", "cohen_dz", "wins_n", "t_p", "wilcoxon_p"]
    out = outputs_dir / "stats_tests.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    print(f"Wrote {out}")
    for r in rows:
        print(f"  {r['metric']:5s} {r['comparison']:26s} diff={r['mean_diff']} wins={r['wins_n']} dz={r['cohen_dz']} wilcoxon_p={r['wilcoxon_p']}")
    return rows


def fig_bars(by_method, outputs_dir: Path):
    cases = sorted(by_method[MAIN].keys())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, metric, label in ((axes[0], "psnr", "PSNR (dB)"), (axes[1], "ssim", "SSIM")):
        means, stds = [], []
        for m in METHODS:
            v = paired_array(by_method, m, metric, cases)
            means.append(v.mean()); stds.append(v.std(ddof=1))
        bars = ax.bar(METHODS, means, yerr=stds, capsize=4,
                      color=[COLORS[m] for m in METHODS], alpha=0.9)
        ax.set_ylabel(label); ax.set_title(f"Test-set {label} by method (mean +/- std, n=60)")
        ax.grid(True, axis="y", alpha=0.3)
        lo = min(means) - max(stds) * 1.5
        ax.set_ylim(bottom=max(0, lo))
        for bar, mu in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{mu:.2f}" if metric == "psnr" else f"{mu:.4f}",
                    ha="center", va="bottom", fontsize=9)
    out = outputs_dir / "fig_method_bars.png"
    fig.savefig(out, dpi=180); plt.close(fig)
    print(f"Wrote {out}")


def fig_gain_vs_noise(by_method, bin_of, outputs_dir: Path):
    bin_cases = {b: sorted([c for c, bb in bin_of.items() if bb == b]) for b in BINS}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    # left: absolute PSNR across noise bins (skip Gaussian to keep scale readable)
    ax = axes[0]
    for m in ("Noisy", "NLM", "ResidualUNet"):
        ys = [paired_array(by_method, m, "psnr", bin_cases[b]).mean() for b in BINS]
        ax.plot(BINS, ys, marker="o", linewidth=2, label=m, color=COLORS[m])
    ax.set_ylabel("PSNR (dB)"); ax.set_xlabel("Noise level (rician_sigma tercile)")
    ax.set_title("PSNR vs noise level"); ax.grid(True, alpha=0.3); ax.legend()
    # right: UNet gain over noisy per bin
    ax = axes[1]
    gains = []
    for b in BINS:
        u = paired_array(by_method, "ResidualUNet", "psnr", bin_cases[b]).mean()
        n = paired_array(by_method, "Noisy", "psnr", bin_cases[b]).mean()
        gains.append(u - n)
    bars = ax.bar(BINS, gains, color="#2e7d32", alpha=0.9)
    ax.set_ylabel("PSNR gain over Noisy (dB)"); ax.set_xlabel("Noise level")
    ax.set_title("UNet improvement grows with noise"); ax.grid(True, axis="y", alpha=0.3)
    for bar, g in zip(bars, gains):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"+{g:.2f}", ha="center", va="bottom", fontsize=10)
    out = outputs_dir / "fig_gain_vs_noise.png"
    fig.savefig(out, dpi=180); plt.close(fig)
    print(f"Wrote {out}")


def main():
    cfg = load_config("config.yaml")
    outputs_dir = project_path(cfg["outputs_dir"])
    by_method, bin_of = load_per_case(outputs_dir / "metrics_per_case.csv")
    run_tests(by_method, outputs_dir)
    fig_bars(by_method, outputs_dir)
    fig_gain_vs_noise(by_method, bin_of, outputs_dir)


if __name__ == "__main__":
    main()
