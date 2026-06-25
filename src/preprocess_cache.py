from __future__ import annotations

import argparse

import numpy as np
from tqdm import tqdm

from common import (
    brain_mask,
    ensure_dirs,
    list_case_ids,
    load_config,
    load_nifti,
    make_or_load_split,
    nonempty_axial_slices,
    norm_params,
    normalize_with_params,
    project_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    split = make_or_load_split(cfg)
    data_dir = project_path(cfg["data_dir"])
    cache_dir = project_path(cfg["cache_dir"])
    case_ids = list_case_ids(data_dir)
    split_ids = set(split["train"] + split["val"] + split["test"])
    if set(case_ids) != split_ids:
        raise RuntimeError("Data case IDs do not match split.json. Remove split.json only if you intend to rebuild it.")

    min_fraction = float(cfg["preprocess"]["mask_slice_min_fraction"])
    for caseid in tqdm(case_ids, desc="preprocess"):
        out = cache_dir / f"{caseid}.npz"
        if out.exists():
            continue
        case_dir = data_dir / caseid
        noisy = load_nifti(case_dir / "T1_noisy.nii.gz")
        clean = load_nifti(case_dir / "T1_clean.nii.gz")
        if noisy.shape != clean.shape:
            raise RuntimeError(f"Shape mismatch for {caseid}: noisy={noisy.shape}, clean={clean.shape}")
        mask = brain_mask(noisy)
        lo, hi = norm_params(noisy, mask)
        noisy_n = normalize_with_params(noisy, lo, hi)
        clean_n = normalize_with_params(clean, lo, hi)
        slices = nonempty_axial_slices(mask, min_fraction)
        if slices.size == 0:
            raise RuntimeError(f"No non-empty axial slices for {caseid}")
        np.savez(
            out,
            noisy=noisy_n.astype(np.float16),
            clean=clean_n.astype(np.float16),
            mask=mask.astype(bool),
            slices=slices,
            lo=np.asarray(lo, dtype=np.float32),
            hi=np.asarray(hi, dtype=np.float32),
        )
    print(f"Done. split={project_path(cfg['split_path']).resolve()} cache={cache_dir.resolve()}")


if __name__ == "__main__":
    main()
