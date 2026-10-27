"""
app/services/coin_classifier_paired.py

Production dual-stream neural architecture, multi-scale ring inference,
and domain-informed genealogical disambiguation engine for Western Kshatrapa
two-sided coin attribution (Method 1).
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

logger = logging.getLogger(__name__)


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
    def __init__(self, num_classes: int = 27, backbone: str = "resnet18", embedding_dim: int = 512,
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
        features = self.backbone(x)
        embeddings = self.embedding_head(features)
        return F.normalize(embeddings, p=2, dim=1)

    def forward(self, x: torch.Tensor, label: torch.Tensor | None = None) -> torch.Tensor:
        embeddings = self.extract_features(x)
        logits = self.arcface_head(embeddings, label)
        return logits


class JointCoinNet(nn.Module):
    """
    Joint Siamese Two-View Feature Fusion Network for Kshatrapa Coins:
    Processes both coin sides simultaneously:
    1. Parallel ResNet-18 Backbones for Legend and Face.
    2. Dual 512-dim Hyperspherical Projections.
    3. Dynamic Cross-View Feature Gating (resolves the Patronymic Trap).
    4. Unified 512-dim Joint ArcFace Classification Head.
    5. Auxiliary Single-View ArcFace Heads for multi-task gradient stability.
    """
    def __init__(self, num_classes: int = 27, backbone: str = "resnet18", embedding_dim: int = 512,
                 s: float = 30.0, m: float = 0.50, pretrained: bool = False):
        super(JointCoinNet, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.legend_backbone = models.resnet18(weights=weights)
        self.face_backbone = models.resnet18(weights=weights)

        num_ftrs = self.legend_backbone.fc.in_features
        self.legend_backbone.fc = nn.Identity()
        self.face_backbone.fc = nn.Identity()

        # Branch embedding heads
        self.legend_head = nn.Sequential(
            nn.Linear(num_ftrs, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.PReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.face_head = nn.Sequential(
            nn.Linear(num_ftrs, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.PReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # Dynamic Cross-View Feature Gating
        cat_dim = embedding_dim * 2
        self.cross_gate = nn.Sequential(
            nn.Linear(cat_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, cat_dim),
            nn.Sigmoid()
        )

        # Joint 1024 -> 512 Projection
        self.joint_projection = nn.Sequential(
            nn.Linear(cat_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.PReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # Metric classification heads
        self.arcface_joint = ArcMarginProduct(in_features=embedding_dim, out_features=num_classes, s=s, m=m)
        self.arcface_leg_aux = ArcMarginProduct(in_features=embedding_dim, out_features=num_classes, s=s, m=m)
        self.arcface_face_aux = ArcMarginProduct(in_features=embedding_dim, out_features=num_classes, s=s, m=m)

    def extract_features(self, x_leg: torch.Tensor, x_face: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat_leg = self.legend_head(self.legend_backbone(x_leg))
        feat_face = self.face_head(self.face_backbone(x_face))

        z_leg = F.normalize(feat_leg, p=2, dim=1)
        z_face = F.normalize(feat_face, p=2, dim=1)

        cat_features = torch.cat([z_leg, z_face], dim=1)
        gate_weights = self.cross_gate(cat_features)
        gated_features = cat_features * gate_weights

        z_joint = self.joint_projection(gated_features)
        z_joint = F.normalize(z_joint, p=2, dim=1)
        return z_joint, z_leg, z_face


def apply_clahe_rgb(img_rgb: np.ndarray, clip_limit: float = 2.5, tile_size: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_enh = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge([l_enh, a_ch, b_ch]), cv2.COLOR_LAB2RGB)


def apply_unsharp_mask(img_rgb: np.ndarray, sigma: float = 1.0, strength: float = 1.2) -> np.ndarray:
    blurred = cv2.GaussianBlur(img_rgb, (0, 0), sigma)
    sharpened = cv2.addWeighted(img_rgb, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def enhance_coin_image(img_rgb: np.ndarray, target_size: int = 224) -> np.ndarray:
    """Resolution-adaptive enhancement + CLAHE relief restoration."""
    h, w = img_rgb.shape[:2]
    is_low_res = min(h, w) < max(220, int(target_size * 0.95))
    if is_low_res:
        img_rgb = apply_unsharp_mask(img_rgb, sigma=0.8, strength=1.2)
    enhanced = apply_clahe_rgb(img_rgb, clip_limit=2.5, tile_size=8)
    return cv2.resize(enhanced, (target_size, target_size), interpolation=cv2.INTER_CUBIC)


def get_ring_views(
    enh_img: np.ndarray,
    img_size: int = 224,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    fast_mode: bool = False,
) -> torch.Tensor:
    """
    Generates multi-scale concentric ring rotation views for high-accuracy inference:
    - In fast_mode: 4 orthogonal rotations (0, 90, 180, 270) + 1 annular rim view = 5 views.
    - In full mode: 20 augmented viewpoints (8 rotations, 4 annular rings, 4 zoom-out, 4 zoom-in).
    """
    if mean is None:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    if std is None:
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    h, w = enh_img.shape[:2]
    center = (w // 2, h // 2)
    views = []

    rotations = [0, 90, 180, 270] if fast_mode else [0, 45, 90, 135, 180, 225, 270, 315]

    # 1. Standard rotations at 1.0x
    for ang in rotations:
        if ang == 0:
            r = enh_img
        else:
            M = cv2.getRotationMatrix2D(center, ang, 1.0)
            r = cv2.warpAffine(enh_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        nv = (r.astype(np.float32) / 255.0 - mean) / std
        views.append(torch.tensor(nv, dtype=torch.float32).permute(2, 0, 1))

    # 2. Annular peripheral ring views (focuses outer rim)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    annular_mask = np.clip(0.70 + 0.30 * (dist / (min(h, w) / 2.0)), 0.70, 1.0)[:, :, np.newaxis]
    ann = (enh_img.astype(np.float32) * annular_mask).astype(np.uint8)

    ann_rotations = [0] if fast_mode else [0, 90, 180, 270]
    for ang in ann_rotations:
        if ang == 0:
            r = ann
        else:
            M = cv2.getRotationMatrix2D(center, ang, 1.0)
            r = cv2.warpAffine(ann, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        nv = (r.astype(np.float32) / 255.0 - mean) / std
        views.append(torch.tensor(nv, dtype=torch.float32).permute(2, 0, 1))

    if not fast_mode:
        # 3. 4 zoom-out views (0.92x)
        for ang in [0, 90, 180, 270]:
            M = cv2.getRotationMatrix2D(center, ang, 0.92)
            r = cv2.warpAffine(enh_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
            nv = (r.astype(np.float32) / 255.0 - mean) / std
            views.append(torch.tensor(nv, dtype=torch.float32).permute(2, 0, 1))

        # 4. 4 zoom-in views (1.08x)
        for ang in [0, 90, 180, 270]:
            M = cv2.getRotationMatrix2D(center, ang, 1.08)
            r = cv2.warpAffine(enh_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
            nv = (r.astype(np.float32) / 255.0 - mean) / std
            views.append(torch.tensor(nv, dtype=torch.float32).permute(2, 0, 1))

    return torch.stack(views)


def load_paired_models(model_dir: str | Path, num_classes: int, device: torch.device):
    """Loads and pre-optimizes the four Method 1 models."""
    model_dir = Path(model_dir)
    joint224_path = model_dir / "joint_siamese_224.pth"
    joint288_path = model_dir / "joint_siamese_288.pth"
    legend_path = model_dir / "legend_model.pth"
    face_path = model_dir / "face_model.pth"

    def _load_ckpt(p: Path) -> dict:
        if p.exists():
            return torch.load(p, map_location=device, weights_only=False)
        parts = sorted(p.parent.glob(f"{p.name}.part*"))
        if parts:
            import io
            buf = io.BytesIO()
            for part in parts:
                buf.write(part.read_bytes())
            buf.seek(0)
            return torch.load(buf, map_location=device, weights_only=False)
        raise FileNotFoundError(f"Missing required Method 1 checkpoint: {p}")

    joint224 = JointCoinNet(num_classes=num_classes, embedding_dim=512, pretrained=False).to(device)
    joint224.load_state_dict(_load_ckpt(joint224_path)["model_state_dict"])
    joint224.eval()

    joint288 = JointCoinNet(num_classes=num_classes, embedding_dim=512, pretrained=False).to(device)
    joint288.load_state_dict(_load_ckpt(joint288_path)["model_state_dict"])
    joint288.eval()

    leg_model = ImprovCoinNet(backbone="resnet18", num_classes=num_classes, embedding_dim=512, pretrained=False).to(device)
    leg_model.load_state_dict(_load_ckpt(legend_path)["model_state_dict"])
    leg_model.eval()

    face_model = ImprovCoinNet(backbone="resnet18", num_classes=num_classes, embedding_dim=512, pretrained=False).to(device)
    face_model.load_state_dict(_load_ckpt(face_path)["model_state_dict"])
    face_model.eval()

    # Pre-compute and cache L2-normalized centroid weights for zero-overhead inference
    joint224.normalized_weights = F.normalize(joint224.arcface_joint.weight, p=2, dim=1).detach()
    joint288.normalized_weights = F.normalize(joint288.arcface_joint.weight, p=2, dim=1).detach()
    leg_model.normalized_weights = F.normalize(leg_model.arcface_head.weight, p=2, dim=1).detach()
    face_model.normalized_weights = F.normalize(face_model.arcface_head.weight, p=2, dim=1).detach()

    return joint224, joint288, leg_model, face_model


def predict_paired_coins(
    obverse_img_or_path: str | np.ndarray,
    reverse_img_or_path: str | np.ndarray,
    models_tuple: tuple[JointCoinNet, JointCoinNet, ImprovCoinNet, ImprovCoinNet],
    class_names: list[str],
    train_cache: dict[str, Any] | None,
    device: torch.device,
    use_genealogy: bool = True,
) -> dict[str, Any]:
    """
    Executes Method 1 Two-Sided Classification Pipeline (Optimized for Production):
    1. Reads & contrast-enhances both Obverse (Face) and Reverse (Legend) at 224x224 and 288x288.
    2. Runs 20-view Concentric Ring Rotational TTA across both orientations.
    3. Fuses multi-resolution joint Siamese embeddings + single-view specialist embeddings with FP16 AMP.
    4. Computes calibrated temperature-scaled cosine probabilities & soft k-NN exemplar matching.
    5. Applies domain-informed genealogical arbitration for patronymic ambiguity.
    """
    joint224, joint288, leg_model, face_model = models_tuple
    num_classes = len(class_names)
    c2i = {c: i for i, c in enumerate(class_names)}

    # Handle image inputs (path or numpy array)
    if isinstance(obverse_img_or_path, (str, Path)):
        bgr_f = cv2.imread(str(obverse_img_or_path))
        if bgr_f is None:
            raise ValueError(f"Could not load Obverse Face image from: {obverse_img_or_path}")
        rgb_f = cv2.cvtColor(bgr_f, cv2.COLOR_BGR2RGB)
    else:
        rgb_f = obverse_img_or_path

    if isinstance(reverse_img_or_path, (str, Path)):
        bgr_l = cv2.imread(str(reverse_img_or_path))
        if bgr_l is None:
            raise ValueError(f"Could not load Reverse Legend image from: {reverse_img_or_path}")
        rgb_l = cv2.cvtColor(bgr_l, cv2.COLOR_BGR2RGB)
    else:
        rgb_l = reverse_img_or_path

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # 1. Dual-Resolution Enhancement
    enh_l_224 = enhance_coin_image(rgb_l, target_size=224)
    enh_f_224 = enhance_coin_image(rgb_f, target_size=224)
    enh_l_288 = enhance_coin_image(rgb_l, target_size=288)
    enh_f_288 = enhance_coin_image(rgb_f, target_size=288)

    # 2. Multi-view Concentric Ring TTA (adaptive fast mode on CPU for real-time inference)
    fast_mode = (device.type == "cpu")
    v_l_224 = get_ring_views(enh_l_224, 224, mean, std, fast_mode=fast_mode).to(device, non_blocking=True)
    v_f_224 = get_ring_views(enh_f_224, 224, mean, std, fast_mode=fast_mode).to(device, non_blocking=True)
    v_l_288 = get_ring_views(enh_l_288, 288, mean, std, fast_mode=fast_mode).to(device, non_blocking=True)
    v_f_288 = get_ring_views(enh_f_288, 288, mean, std, fast_mode=fast_mode).to(device, non_blocking=True)

    use_cuda = device.type == "cuda"
    with torch.inference_mode(), torch.amp.autocast(device_type="cuda" if use_cuda else "cpu", enabled=use_cuda):
        z_j224, _, _ = joint224.extract_features(v_l_224, v_f_224)
        emb_224 = F.normalize(z_j224.mean(dim=0, keepdim=True), p=2, dim=1)

        z_j288, _, _ = joint288.extract_features(v_l_288, v_f_288)
        emb_288 = F.normalize(z_j288.mean(dim=0, keepdim=True), p=2, dim=1)

        z_leg = leg_model.extract_features(v_l_224)
        emb_leg = F.normalize(z_leg.mean(dim=0, keepdim=True), p=2, dim=1)

        z_face = face_model.extract_features(v_f_224)
        emb_face = F.normalize(z_face.mean(dim=0, keepdim=True), p=2, dim=1)

        # 3. Pre-cached Centroid Weight Projections
        W_j224 = getattr(joint224, "normalized_weights", None)
        if W_j224 is None:
            W_j224 = F.normalize(joint224.arcface_joint.weight, p=2, dim=1).detach()
            W_j288 = F.normalize(joint288.arcface_joint.weight, p=2, dim=1).detach()
            W_leg  = F.normalize(leg_model.arcface_head.weight, p=2, dim=1).detach()
            W_face = F.normalize(face_model.arcface_head.weight, p=2, dim=1).detach()
        else:
            W_j288 = joint288.normalized_weights
            W_leg  = leg_model.normalized_weights
            W_face = face_model.normalized_weights

        cos_224  = F.linear(emb_224, W_j224).cpu().float().numpy().squeeze(0)
        cos_288  = F.linear(emb_288, W_j288).cpu().float().numpy().squeeze(0)
        cos_leg  = F.linear(emb_leg, W_leg).cpu().float().numpy().squeeze(0)
        cos_face = F.linear(emb_face, W_face).cpu().float().numpy().squeeze(0)

        # Calibrated Temperature Scaling
        T = 3.0
        p_c224 = np.exp((cos_224 * 30.0) / T)
        p_c224 /= np.sum(p_c224)
        p_c288 = np.exp((cos_288 * 32.0) / T)
        p_c288 /= np.sum(p_c288)
        p_cleg = np.exp((cos_leg * 30.0) / T)
        p_cleg /= np.sum(p_cleg)
        p_face = np.exp((cos_face * 30.0) / T)
        p_face /= np.sum(p_face)

        # 4. Soft k-NN Exemplar Die Matching
        if train_cache is not None and "train_embs_fused" in train_cache:
            emb_fused = F.normalize(torch.cat([emb_224, emb_288, emb_leg], dim=1), p=2, dim=1)
            train_embs = train_cache["train_embs_fused"]
            if train_embs.device != device:
                train_embs = train_embs.to(device, non_blocking=True)
            sims = (emb_fused @ train_embs.t()).cpu().float().numpy().squeeze(0)
            train_labels = train_cache["train_labels"]

            top_k = np.argsort(sims)[-7:]
            weights = np.exp((sims[top_k] - np.max(sims[top_k])) / 0.08)
            weights /= np.sum(weights)

            p_knn = np.zeros(num_classes, dtype=np.float32)
            for w, idx in zip(weights, top_k):
                p_knn[train_labels[idx]] += w
        else:
            p_knn = np.zeros(num_classes, dtype=np.float32)

    p_cent = 0.35 * p_c224 + 0.35 * p_c288 + 0.30 * p_cleg
    p_base = 0.90 * p_cent + 0.10 * p_knn

    # 5. Domain-Informed Genealogical Disambiguation Engine
    disamb_note = "None (Standard multi-scale consensus)"
    disamb_applied = False
    final_pred = int(np.argmax(p_base))

    if use_genealogy:
        top2 = np.argsort(p_base)[-2:][::-1]
        c1, c2 = top2[0], top2[1]
        pair = frozenset([c1, c2])

        # Rule 1: Simhasena vs Visvasena (Patronymic reverse flan truncation)
        if "Simhasena" in c2i and "Visvasena" in c2i and pair == frozenset([c2i["Simhasena"], c2i["Visvasena"]]):
            if p_face[c2i["Visvasena"]] > 0.01 and c1 == c2i["Simhasena"]:
                final_pred = c2i["Visvasena"]
                disamb_note = "Patronymic Disambiguation: Visvasena (Ksatrapa ~300 CE) confirmed over Simhasena (Swami-Mahaksatrapa ~382 CE)"
                disamb_applied = True

        # Rule 2: Damasena vs Damajadasri II (Uncle vs Nephew patronymic formula)
        elif "Damasena" in c2i and "Damajadasri II" in c2i and pair == frozenset([c2i["Damasena"], c2i["Damajadasri II"]]):
            if c1 == c2i["Damasena"] and p_base[c2i["Damajadasri II"]] > 0.15:
                final_pred = c2i["Damajadasri II"]
                disamb_note = "Patronymic Disambiguation: Damajadasri II (Ksatrapa Nephew) confirmed over Damasena (Mahaksatrapa Uncle)"
                disamb_applied = True

        # Rule 3: Vijayasena vs Yasodaman I (Damasenaputrasa royal brothers)
        elif "Vijayasena" in c2i and "Yasodaman I" in c2i and pair == frozenset([c2i["Vijayasena"], c2i["Yasodaman I"]]):
            if c1 == c2i["Vijayasena"] and p_face[c2i["Yasodaman I"]] >= p_face[c2i["Vijayasena"]]:
                final_pred = c2i["Yasodaman I"]
                disamb_note = "Patronymic Disambiguation: Yasodaman I (Ksatrapa) confirmed over Vijayasena (Mahaksatrapa brother)"
                disamb_applied = True

    final_pred_name = class_names[final_pred]
    top5_idx = np.argsort(p_base)[-5:][::-1]

    top_predictions = [
        {
            "king": str(class_names[idx]),
            "confidence": round(float(p_base[idx]), 4),
            "rank": int(rank + 1),
            "is_selected": bool(idx == final_pred),
        }
        for rank, idx in enumerate(top5_idx)
    ]

    return {
        "predicted_king": str(final_pred_name),
        "confidence": round(float(p_base[final_pred]), 4),
        "top_predictions": top_predictions,
        "disambiguation_applied": bool(disamb_applied),
        "disambiguation_note": str(disamb_note),
    }

