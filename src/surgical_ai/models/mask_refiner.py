"""Mask-refinement network: a U-Net with an ImageNet ResNet-34 encoder that takes the image crop, SAM's mask logits and the box as
input and predicts a correction added to SAM's logits. The output layer starts at zero, so the untrained network returns SAM's
mask unchanged and training can only move away from it where that lowers the loss. Logits outside the box are forced to -12
(a mask never leaves its ground-truth box)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

LOGIT_CLIP = 12.0
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class _Up(nn.Module):
    def __init__(self, cin: int, skip: int, cout: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin + skip, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class MaskRefiner(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet34(weights=weights)
        stem = nn.Conv2d(5, 64, 7, 2, 3, bias=False)  # RGB + SAM logit + box mask
        with torch.no_grad():
            stem.weight.zero_()
            stem.weight[:, :3] = r.conv1.weight
        self.stem = nn.Sequential(stem, r.bn1, r.relu)
        self.pool, self.l1, self.l2, self.l3, self.l4 = r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4
        self.up3, self.up2 = _Up(512, 256, 256), _Up(256, 128, 128)
        self.up1, self.up0 = _Up(128, 64, 64), _Up(64, 64, 32)
        self.refine = nn.Sequential(nn.Conv2d(32 + 5, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.out = nn.Conv2d(32, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, img: torch.Tensor, sam_logit: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        """img (B,3,S,S) in [0,1]; sam_logit and box (B,1,S,S) (box a 0/1 rectangle). Returns refined logits (B,1,S,S)."""
        x = torch.cat([(img - self.mean) / self.std, sam_logit / 6.0, box], dim=1)
        s0 = self.stem(x)
        s1 = self.l1(self.pool(s0))
        s2 = self.l2(s1)
        s3 = self.l3(s2)
        s4 = self.l4(s3)
        d = self.up3(s4, s3)
        d = self.up2(d, s2)
        d = self.up1(d, s1)
        d = self.up0(d, s0)
        d = F.interpolate(d, size=x.shape[-2:], mode="bilinear", align_corners=False)
        delta = self.out(self.refine(torch.cat([d, x], dim=1)))
        refined = (sam_logit + delta).clamp(-LOGIT_CLIP, LOGIT_CLIP)
        return torch.where(box > 0.5, refined, torch.full_like(refined, -LOGIT_CLIP))


def box_channel(boxes: torch.Tensor, geom: torch.Tensor, size: int) -> torch.Tensor:
    """Rectangle mask (B,1,S,S) of each box in its crop. boxes (B,4) xyxy and geom (B,3) (cx, cy, side) in frame pixels."""
    cx, cy, s = geom[:, 0:1], geom[:, 1:2], geom[:, 2:3]
    ax = (torch.arange(size, device=boxes.device, dtype=torch.float32)[None] + 0.5) / size  # crop coordinate in [0,1]
    xs = cx - s / 2 + ax * s
    ys = cy - s / 2 + ax * s
    in_x = (xs >= boxes[:, 0:1]) & (xs <= boxes[:, 2:3])
    in_y = (ys >= boxes[:, 1:2]) & (ys <= boxes[:, 3:4])
    return (in_y[:, :, None] & in_x[:, None, :]).float()[:, None]
