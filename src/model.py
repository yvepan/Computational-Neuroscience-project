from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        groups = min(8, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.norm1(self.conv1(x)), inplace=True)
        y = self.norm2(self.conv2(y))
        return F.relu(y + self.skip(x), inplace=True)


class ResidualUNet2D(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 1, base: int = 48) -> None:
        super().__init__()
        ch = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc1 = ResBlock(in_ch, ch[0])
        self.enc2 = ResBlock(ch[0], ch[1])
        self.enc3 = ResBlock(ch[1], ch[2])
        self.enc4 = ResBlock(ch[2], ch[3])
        self.bottleneck = ResBlock(ch[3], ch[4])
        self.pool = nn.MaxPool2d(2)
        self.up4 = ResBlock(ch[4] + ch[3], ch[3])
        self.up3 = ResBlock(ch[3] + ch[2], ch[2])
        self.up2 = ResBlock(ch[2] + ch[1], ch[1])
        self.up1 = ResBlock(ch[1] + ch[0], ch[0])
        self.head = nn.Conv2d(ch[0], out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        u4 = F.interpolate(b, size=e4.shape[-2:], mode="bilinear", align_corners=False)
        u4 = self.up4(torch.cat([u4, e4], dim=1))
        u3 = F.interpolate(u4, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        u3 = self.up3(torch.cat([u3, e3], dim=1))
        u2 = F.interpolate(u3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.up2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(u2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up1(torch.cat([u1, e1], dim=1))
        residual_noise = self.head(u1)
        return x[:, 1:2] - residual_noise
