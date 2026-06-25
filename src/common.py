from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import yaml
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter, label
from skimage.filters import threshold_otsu
from skimage.metrics import structural_similarity as sk_ssim


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = project_path(path or "config.yaml")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def ensure_dirs(cfg: dict[str, Any]) -> None:
    for key in ("cache_dir", "ckpt_dir", "outputs_dir"):
        project_path(cfg[key]).mkdir(parents=True, exist_ok=True)
    (project_path(cfg["outputs_dir"]) / "figs").mkdir(parents=True, exist_ok=True)


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    def as_rng_tensor(value: Any) -> torch.ByteTensor:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().to(torch.uint8)
        return torch.as_tensor(value, dtype=torch.uint8, device="cpu")

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(as_rng_tensor(state["torch"]))
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([as_rng_tensor(s) for s in state["cuda"]])


def list_case_ids(data_dir: Path) -> list[str]:
    ids = []
    for p in data_dir.iterdir():
        if p.is_dir() and (p / "T1_noisy.nii.gz").exists() and (p / "T1_clean.nii.gz").exists():
            ids.append(p.name)
    return sorted(ids)


def make_or_load_split(cfg: dict[str, Any]) -> dict[str, list[str]]:
    split_path = project_path(cfg["split_path"])
    if split_path.exists():
        with split_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    data_dir = project_path(cfg["data_dir"])
    case_ids = list_case_ids(data_dir)
    expected = sum(cfg["split"].values())
    if len(case_ids) != expected:
        raise RuntimeError(f"Expected {expected} cases, found {len(case_ids)} in {data_dir}")
    rng = random.Random(int(cfg["seed"]))
    rng.shuffle(case_ids)
    n_train = int(cfg["split"]["train"])
    n_val = int(cfg["split"]["val"])
    split = {
        "train": sorted(case_ids[:n_train]),
        "val": sorted(case_ids[n_train:n_train + n_val]),
        "test": sorted(case_ids[n_train + n_val:]),
    }
    with split_path.open("w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    return split


def load_nifti(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32), dtype=np.float32)


def brain_mask(noisy: np.ndarray) -> np.ndarray:
    sm = gaussian_filter(noisy.astype("float32"), 1.0)
    thr = max(threshold_otsu(sm), 0.06 * np.percentile(noisy, 99.5))
    m = sm > thr
    m = binary_closing(m, iterations=2)
    m = binary_fill_holes(m)
    lab, n = label(m)
    if n > 1:
        m = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    return m.astype(bool)


def norm_params(noisy: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    v = noisy[mask]
    if v.size == 0:
        raise RuntimeError("Empty brain mask; cannot compute normalization parameters.")
    lo, hi = np.percentile(v, 1), np.percentile(v, 99.5)
    return float(lo), float(max(hi, lo + 1e-6))


def normalize_with_params(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return ((x.astype(np.float32) - lo) / (hi - lo)).astype(np.float32)


def nonempty_axial_slices(mask: np.ndarray, min_fraction: float = 0.01) -> np.ndarray:
    keep = [k for k in range(mask.shape[2]) if float(mask[:, :, k].mean()) > min_fraction]
    return np.asarray(keep, dtype=np.int16)


def load_cached_case(cache_dir: Path, caseid: str) -> dict[str, Any]:
    path = cache_dir / f"{caseid}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing cache file {path}. Run: python src/preprocess_cache.py")
    with np.load(path, allow_pickle=False) as z:
        return {
            "noisy": z["noisy"],
            "clean": z["clean"],
            "mask": z["mask"].astype(bool),
            "slices": z["slices"].astype(np.int32),
            "lo": float(z["lo"]),
            "hi": float(z["hi"]),
        }


def pad_to_multiple_tensor(x: torch.Tensor, multiple: int = 16) -> tuple[torch.Tensor, tuple[int, int]]:
    h, w = x.shape[-2:]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph or pw:
        x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, (ph, pw)


@torch.no_grad()
def infer_volume_25d(
    model: torch.nn.Module,
    noisy: np.ndarray,
    device: torch.device,
    batch_slices: int = 24,
    amp: bool = True,
) -> np.ndarray:
    model.eval()
    h, w, d = noisy.shape
    preds: list[np.ndarray] = []
    autocast_enabled = amp and device.type == "cuda"
    for start in range(0, d, batch_slices):
        xs = []
        for k in range(start, min(d, start + batch_slices)):
            km1 = max(k - 1, 0)
            kp1 = min(k + 1, d - 1)
            xs.append(np.stack([noisy[:, :, km1], noisy[:, :, k], noisy[:, :, kp1]], axis=0))
        x = torch.from_numpy(np.stack(xs).astype(np.float32)).to(device, non_blocking=True)
        x, (ph, pw) = pad_to_multiple_tensor(x, 16)
        with torch.amp.autocast("cuda", enabled=autocast_enabled):
            y = model(x)
        if ph:
            y = y[..., :-ph, :]
        if pw:
            y = y[..., :-pw]
        preds.append(y[:, 0].float().cpu().numpy())
    return np.concatenate(preds, axis=0).transpose(1, 2, 0).astype(np.float32)


def masked_metrics(pred: np.ndarray, clean: np.ndarray, mask: np.ndarray, dr: float = 1.0) -> dict[str, float]:
    p = np.clip(pred.astype(np.float32), 0, 1)
    c = np.clip(clean.astype(np.float32), 0, 1)
    e = (p - c)[mask]
    mse = float(np.mean(e ** 2))
    mae = float(np.mean(np.abs(e)))
    rmse = math.sqrt(max(mse, 0.0))
    psnr = 10.0 * math.log10((dr ** 2) / max(mse, 1e-12))
    _, smap = sk_ssim(c, p, data_range=dr, full=True)
    return {"mae": mae, "rmse": rmse, "psnr": psnr, "ssim": float(smap[mask].mean())}


def append_csv(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_manifest(data_dir: Path) -> dict[str, dict[str, str]]:
    manifest = data_dir / "manifest.csv"
    rows: dict[str, dict[str, str]] = {}
    with manifest.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["caseid"]] = row
    return rows


class Timer:
    def __init__(self) -> None:
        self.start = time.time()

    @property
    def hours(self) -> float:
        return (time.time() - self.start) / 3600.0

    @property
    def seconds(self) -> float:
        return time.time() - self.start
