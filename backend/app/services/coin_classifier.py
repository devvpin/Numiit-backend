"""
app/services/coin_classifier.py

Self-contained ArcFace neural architecture and preprocessing pipeline for
Western Kshatrapa coin classification in the NumiIT backend.
"""

from __future__ import annotations

import math
from typing import Any
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ArcMarginProduct(nn.Module):
    """
    Additive Angular Margin for ArcFace (Deng et al., CVPR 2019).
    Calculates cos(theta + m) for target class and cos(theta) for other classes.
    """
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.50):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input: torch.Tensor, label: torch.Tensor | None = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        if label is None:
            return cosine * self.s

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).clamp(0, 1)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class ImprovCoinNet(nn.Module):
    """
    ArcFace Metric Learning Network with ResNet-18 Backbone.
    Produces hyper-compact intra-class clusters and wide inter-class angular separation.
    """
    def __init__(self, num_classes: int = 29, backbone: str = "resnet18", embedding_dim: int = 512,
                 s: float = 30.0, m: float = 0.50, pretrained: bool = False):
        super(ImprovCoinNet, self).__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes

        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
        else:
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet34(weights=weights)

        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Proven ArcFace embedding projection head
        self.embedding_head = nn.Sequential(
            nn.Linear(num_ftrs, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.PReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # Hyperspherical Angular Margin Head
        self.arcface_head = ArcMarginProduct(in_features=embedding_dim, out_features=num_classes, s=s, m=m)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat = self.embedding_head(feat)
        return F.normalize(feat, p=2, dim=1)

    def forward(self, x: torch.Tensor, label: torch.Tensor | None = None) -> torch.Tensor:
        features = self.extract_features(x)
        logits = self.arcface_head(features, label)
        return logits


def enhance_coin_image(rgb_image: np.ndarray, target_size: int = 224) -> np.ndarray:
    """
    High-fidelity numismatic contrast enhancement pipeline:
    1. Luminance CLAHE in LAB color space to enhance faint Brahmi relief without tinting patina.
    2. High-pass unsharp mask for low-resolution thumbnails to preserve stroke edges.
    3. High-quality bicubic resampling to target dimensions.
    """
    # 1. LAB Luminance CLAHE
    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    # 2. Resolution-Adaptive Unsharp Masking
    h, w = enhanced_rgb.shape[:2]
    if min(h, w) < 220:
        blurred = cv2.GaussianBlur(enhanced_rgb, (0, 0), sigmaX=1.0)
        sharpened = cv2.addWeighted(enhanced_rgb, 1.35, blurred, -0.35, 0)
        enhanced_rgb = np.clip(sharpened, 0, 255).astype(np.uint8)

    # 3. High-order Bicubic Interpolation
    resized = cv2.resize(enhanced_rgb, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    return resized
