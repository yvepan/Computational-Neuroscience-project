from __future__ import annotations

import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from common import load_cached_case


class MRIDenoiseTrainDataset(Dataset):
    def __init__(
        self,
        case_ids: list[str],
        cache_dir: Path,
        patch_size: int = 192,
        samples_per_epoch: int = 24000,
        lru_cases: int = 8,
    ) -> None:
        self.case_ids = list(case_ids)
        self.cache_dir = Path(cache_dir)
        self.patch_size = int(patch_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.lru_cases = int(lru_cases)
        self.case_block = max(1, int(np.ceil(self.samples_per_epoch / max(len(self.case_ids), 1))))
        self._cache: OrderedDict[str, dict] = OrderedDict()

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        caseid = self.case_ids[(idx // self.case_block) % len(self.case_ids)]
        c = self._get_case(caseid)
        noisy = c["noisy"]
        clean = c["clean"]
        mask = c["mask"]
        k = int(random.choice(c["slices"]))
        km1 = max(k - 1, 0)
        kp1 = min(k + 1, noisy.shape[2] - 1)
        x = np.stack([noisy[:, :, km1], noisy[:, :, k], noisy[:, :, kp1]], axis=0).astype(np.float32)
        y = clean[:, :, k][None].astype(np.float32)
        m = mask[:, :, k]
        x, y = self._random_patch(x, y, m)
        if random.random() < 0.5:
            x = x[:, ::-1, :].copy()
            y = y[:, ::-1, :].copy()
        if random.random() < 0.5:
            x = x[:, :, ::-1].copy()
            y = y[:, :, ::-1].copy()
        return torch.from_numpy(x), torch.from_numpy(y)

    def _get_case(self, caseid: str) -> dict:
        if caseid in self._cache:
            self._cache.move_to_end(caseid)
            return self._cache[caseid]
        case = load_cached_case(self.cache_dir, caseid)
        self._cache[caseid] = case
        if len(self._cache) > self.lru_cases:
            self._cache.popitem(last=False)
        return case

    def _random_patch(self, x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ps = self.patch_size
        h, w = mask.shape
        pad_h = max(ps - h, 0)
        pad_w = max(ps - w, 0)
        if pad_h or pad_w:
            before_h, before_w = pad_h // 2, pad_w // 2
            after_h, after_w = pad_h - before_h, pad_w - before_w
            x_mode = "reflect" if x.shape[1] > 1 and x.shape[2] > 1 else "edge"
            y_mode = "reflect" if y.shape[1] > 1 and y.shape[2] > 1 else "edge"
            x = np.pad(x, ((0, 0), (before_h, after_h), (before_w, after_w)), mode=x_mode)
            y = np.pad(y, ((0, 0), (before_h, after_h), (before_w, after_w)), mode=y_mode)
            mask = np.pad(mask, ((before_h, after_h), (before_w, after_w)), mode="constant")
            h, w = mask.shape

        ys, xs = np.where(mask)
        for _ in range(20):
            if ys.size:
                j = random.randrange(ys.size)
                cy = int(ys[j])
                cx = int(xs[j])
                top = min(max(cy - random.randrange(ps), 0), h - ps)
                left = min(max(cx - random.randrange(ps), 0), w - ps)
            else:
                top = random.randint(0, h - ps)
                left = random.randint(0, w - ps)
            if mask[top:top + ps, left:left + ps].mean() > 0:
                return x[:, top:top + ps, left:left + ps], y[:, top:top + ps, left:left + ps]

        top = random.randint(0, h - ps)
        left = random.randint(0, w - ps)
        return x[:, top:top + ps, left:left + ps], y[:, top:top + ps, left:left + ps]


class MRIDenoiseCaseDataset(Dataset):
    def __init__(self, case_ids: list[str], cache_dir: Path) -> None:
        self.case_ids = list(case_ids)
        self.cache_dir = Path(cache_dir)

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> tuple[str, dict]:
        caseid = self.case_ids[idx]
        return caseid, load_cached_case(self.cache_dir, caseid)
