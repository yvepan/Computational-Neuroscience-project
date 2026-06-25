from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _gaussian_window(window_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w2d = (g[:, None] @ g[None, :]).expand(channels, 1, window_size, window_size).contiguous()
    return w2d


def ssim_value(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    channels = x.shape[1]
    window = _gaussian_window(window_size, sigma, channels, x.device, x.dtype)
    pad = window_size // 2
    mu_x = F.conv2d(x, window, padding=pad, groups=channels)
    mu_y = F.conv2d(y, window, padding=pad, groups=channels)
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, window, padding=pad, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=pad, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=channels) - mu_xy
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2))
    return ssim_map.mean()


class SSIMLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - ssim_value(pred, target, data_range=1.0)


class GradientLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        ky = kx.transpose(2, 3).contiguous()
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        kx = self.kx.to(device=pred.device, dtype=pred.dtype)
        ky = self.ky.to(device=pred.device, dtype=pred.dtype)
        px = F.conv2d(pred, kx, padding=1)
        py = F.conv2d(pred, ky, padding=1)
        tx = F.conv2d(target, kx, padding=1)
        ty = F.conv2d(target, ky, padding=1)
        return F.l1_loss(px, tx) + F.l1_loss(py, ty)


class DenoiseLoss(nn.Module):
    def __init__(self, ssim_weight: float = 0.2, grad_weight: float = 0.1, use_gradient: bool = True) -> None:
        super().__init__()
        self.ssim_weight = float(ssim_weight)
        self.grad_weight = float(grad_weight)
        self.use_gradient = bool(use_gradient)
        self.ssim = SSIMLoss()
        self.grad = GradientLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.clamp(0.0, 1.0)
        target = target.clamp(0.0, 1.0)
        loss = F.l1_loss(pred, target) + self.ssim_weight * self.ssim(pred, target)
        if self.use_gradient:
            loss = loss + self.grad_weight * self.grad(pred, target)
        return loss
