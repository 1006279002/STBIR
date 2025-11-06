from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    denom = torch.clamp(torch.norm(x, dim=dim, keepdim=True), min=eps)
    return x / denom


@dataclass
class LossConfig:
    temperature: float
    triplet_margin: float
    fusion_margin: float
    weight_info_nce: float
    weight_triplet: float
    weight_fusion: float


class SketchEncoder(nn.Module):
    """ResNet-based sketch encoder with projection to CLIP dimensionality."""

    def __init__(self, backbone: str = "resnet50", output_dim: int = 512, pretrained: bool = True):
        super().__init__()
        if backbone != "resnet50":
            raise ValueError(f"Unsupported sketch backbone: {backbone}")
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = models.resnet50(weights=weights)
        in_features = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.backbone = resnet
        self.projection = nn.Linear(in_features, output_dim)
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.projection(features)


class MultiStageSBIRModel(nn.Module):
    """Stage-wise training wrapper around CLIP and a sketch encoder."""

    def __init__(
        self,
        clip_model: nn.Module,
        feature_dim: int = 512,
        sketch_backbone: str = "resnet50",
        sketch_pretrained: bool = True,
        fusion_strategy: str = "mean",
    ):
        super().__init__()
        self.clip = clip_model
        self.feature_dim = feature_dim
        self.sketch_encoder = SketchEncoder(sketch_backbone, output_dim=feature_dim, pretrained=sketch_pretrained)
        self.fusion_strategy = fusion_strategy

        self.register_buffer("clip_mean", CLIP_MEAN.view(1, 3, 1, 1))
        self.register_buffer("clip_std", CLIP_STD.view(1, 3, 1, 1))

    # ------------------------------------------------------------------
    # Stage control
    # ------------------------------------------------------------------
    def configure_stage(self, target: str) -> None:
        target = target.lower()
        self.clip.eval()
        self.clip.requires_grad_(False)
        self.sketch_encoder.eval()
        self.sketch_encoder.requires_grad_(False)

        if target == "sketch":
            self.sketch_encoder.train()
            self.sketch_encoder.requires_grad_(True)
            self.sketch_encoder.float()
        elif target == "image":
            self.clip.float()
            self.clip.visual.train()
            self.clip.visual.requires_grad_(True)
        elif target == "text":
            # promote full CLIP stack to float32 so encode_text uses consistent dtype
            self.clip.float()
            self.clip.visual.eval()
            self.clip.visual.requires_grad_(False)

            self.clip.transformer.train()
            self.clip.token_embedding.requires_grad_(True)
            self.clip.transformer.requires_grad_(True)
            if hasattr(self.clip, "ln_final"):
                self.clip.ln_final.train()
                self.clip.ln_final.requires_grad_(True)
            if hasattr(self.clip, "text_projection"):
                self.clip.text_projection.requires_grad_(True)
        else:
            raise ValueError(f"Unknown training target: {target}")

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.clip.encode_image(images)
        return l2_normalize(feats.float())

    def encode_texts(self, tokens: torch.Tensor) -> torch.Tensor:
        feats = self.clip.encode_text(tokens)
        return l2_normalize(feats.float())

    def encode_sketches(self, sketches: torch.Tensor) -> torch.Tensor:
        feats = self.sketch_encoder(sketches)
        return l2_normalize(feats.float())

    # ------------------------------------------------------------------
    # Forward for training
    # ------------------------------------------------------------------
    def forward_stage(
        self,
        batch: Dict[str, torch.Tensor],
        target: str,
        loss_cfg: LossConfig,
    ) -> Dict[str, torch.Tensor]:
        images = batch["images"]
        texts = batch["texts"]
        sketches = batch["sketches_resnet"]

        image_features = self.encode_images(images)
        text_features = self.encode_texts(texts)
        sketch_features = self.encode_sketches(sketches)

        neg_images, neg_image_mask = self._encode_negative_images(batch["neg_images"])
        neg_texts, neg_text_mask = self._encode_negative_texts(batch["neg_texts"])
        neg_sketches, neg_sketch_mask = self._encode_negative_sketches(batch["neg_sketches_resnet"])

        target = target.lower()
        if target == "sketch":
            anchor = self._fuse_features([image_features, text_features])
            positives = sketch_features
            negatives = neg_sketches
            mask = neg_sketch_mask
            fusion_loss = images.new_tensor(0.0)
        elif target == "image":
            anchor = self._fuse_features([text_features, sketch_features])
            positives = image_features
            negatives = neg_images
            mask = neg_image_mask
            fusion_loss = self._fusion_alignment_loss(
                images,
                batch["sketches_clip"],
                batch["neg_sketches_clip"],
                image_features,
                loss_cfg.fusion_margin,
            )
        elif target == "text":
            anchor = self._fuse_features([image_features, sketch_features])
            positives = text_features
            negatives = neg_texts
            mask = neg_text_mask
            fusion_loss = images.new_tensor(0.0)
        else:
            raise ValueError(f"Unknown target stage: {target}")

        info = self._info_nce(anchor, positives, negatives, mask, loss_cfg.temperature)
        triplet = self._triplet_cosine(anchor, positives, negatives, mask, loss_cfg.triplet_margin)

        total = (
            loss_cfg.weight_info_nce * info
            + loss_cfg.weight_triplet * triplet
            + loss_cfg.weight_fusion * fusion_loss
        )

        return {
            "loss": total,
            "info_nce": info,
            "triplet": triplet,
            "fusion": fusion_loss,
            "features": {
                "image": image_features,
                "text": text_features,
                "sketch": sketch_features,
            },
        }

    # ------------------------------------------------------------------
    # Negative encoders
    # ------------------------------------------------------------------
    def _encode_negative_images(self, negatives: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if negatives.dim() == 4:
            negatives = negatives.unsqueeze(1)
        if negatives.size(1) == 0:
            zeros = negatives.new_zeros((negatives.size(0), 0, self.feature_dim))
            mask = negatives.new_zeros((negatives.size(0), 0), dtype=torch.bool)
            return zeros, mask
        b, k = negatives.size(0), negatives.size(1)
        flat = negatives.reshape(b * k, *negatives.shape[2:])
        feats = self.encode_images(flat)
        feats = feats.view(b, k, -1)
        mask = negatives.reshape(b, k, -1).abs().sum(dim=-1) > 0
        return feats, mask

    def _encode_negative_texts(self, negatives: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if negatives.dim() == 2:
            negatives = negatives.unsqueeze(1)
        if negatives.size(1) == 0:
            zeros = negatives.new_zeros((negatives.size(0), 0, self.feature_dim))
            mask = negatives.new_zeros((negatives.size(0), 0), dtype=torch.bool)
            return zeros, mask
        b, k = negatives.size(0), negatives.size(1)
        flat = negatives.reshape(b * k, -1)
        feats = self.encode_texts(flat)
        feats = feats.view(b, k, -1)
        mask = negatives.sum(dim=-1) > 0
        return feats, mask

    def _encode_negative_sketches(self, negatives: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if negatives.dim() == 4:
            negatives = negatives.unsqueeze(1)
        if negatives.size(1) == 0:
            zeros = negatives.new_zeros((negatives.size(0), 0, self.feature_dim))
            mask = negatives.new_zeros((negatives.size(0), 0), dtype=torch.bool)
            return zeros, mask
        b, k = negatives.size(0), negatives.size(1)
        flat = negatives.reshape(b * k, *negatives.shape[2:])
        feats = self.encode_sketches(flat)
        feats = feats.view(b, k, -1)
        mask = negatives.reshape(b, k, -1).abs().sum(dim=-1) > 0
        return feats, mask

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------
    def _info_nce(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
        mask: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        total = anchor.new_tensor(0.0)
        valid = 0
        for idx in range(anchor.size(0)):
            neg_feat = negatives[idx]
            neg_mask = mask[idx] if mask.numel() else None
            if neg_mask is not None:
                neg_feat = neg_feat[neg_mask]
            if neg_feat.size(0) == 0:
                continue
            logits = torch.cat([positive[idx].unsqueeze(0), neg_feat], dim=0)
            logits = anchor[idx].unsqueeze(0) @ logits.t()
            logits = logits.squeeze(0) / temperature
            labels = torch.zeros(1, device=anchor.device, dtype=torch.long)
            total = total + F.cross_entropy(logits.unsqueeze(0), labels)
            valid += 1
        if valid == 0:
            return anchor.new_tensor(0.0)
        return total / valid

    def _triplet_cosine(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
        mask: torch.Tensor,
        margin: float,
    ) -> torch.Tensor:
        total = anchor.new_tensor(0.0)
        count = 0
        pos_sim = F.cosine_similarity(anchor, positive)
        for idx in range(anchor.size(0)):
            neg_feat = negatives[idx]
            neg_mask = mask[idx] if mask.numel() else None
            if neg_mask is not None:
                neg_feat = neg_feat[neg_mask]
            if neg_feat.size(0) == 0:
                continue
            neg_sim = F.cosine_similarity(anchor[idx].unsqueeze(0), neg_feat)
            loss = torch.clamp(margin + neg_sim - pos_sim[idx], min=0.0)
            total = total + loss.sum()
            count += neg_feat.size(0)
        if count == 0:
            return anchor.new_tensor(0.0)
        return total / count

    def _fusion_alignment_loss(
        self,
        images: torch.Tensor,
        sketches_clip: torch.Tensor,
        neg_sketches_clip: torch.Tensor,
        image_features: torch.Tensor,
        margin: float,
    ) -> torch.Tensor:
        fused = self._fuse_quad(images, sketches_clip)
        fused_features = self.encode_images(fused)
        pos_loss = 1.0 - F.cosine_similarity(fused_features, image_features).mean()

        neg_loss = images.new_tensor(0.0)
        if neg_sketches_clip.size(1) > 0:
            neg_first = neg_sketches_clip[:, 0]
            fused_neg = self._fuse_quad(images, neg_first)
            fused_neg_features = self.encode_images(fused_neg)
            neg_sim = F.cosine_similarity(fused_neg_features, image_features)
            neg_loss = torch.clamp(neg_sim - margin, min=0.0).mean()
        return pos_loss + neg_loss

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def _fuse_features(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(features, dim=0)
        if self.fusion_strategy == "mean":
            fused = stacked.mean(dim=0)
        elif self.fusion_strategy == "sum":
            fused = stacked.sum(dim=0)
        else:
            raise ValueError(f"Unsupported fusion strategy: {self.fusion_strategy}")
        return l2_normalize(fused)

    def _fuse_quad(self, image: torch.Tensor, sketch: torch.Tensor) -> torch.Tensor:
        image_denorm = self._denormalize_clip(image)
        sketch_denorm = self._denormalize_clip(sketch)
        fused = image_denorm.clone()
        h, w = fused.shape[-2:]
        h2, w2 = h // 2, w // 2
        fused[..., :h2, w2:] = sketch_denorm[..., :h2, w2:]
        fused[..., h2:, :w2] = sketch_denorm[..., h2:, :w2]
        return self._normalize_clip(fused)

    def _denormalize_clip(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.clip_std.to(tensor.device, tensor.dtype) + self.clip_mean.to(tensor.device, tensor.dtype)

    def _normalize_clip(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.clip_mean.to(tensor.device, tensor.dtype)) / self.clip_std.to(tensor.device, tensor.dtype)


__all__ = ["MultiStageSBIRModel", "LossConfig"]