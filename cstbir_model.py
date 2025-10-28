from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src.clip.model import LayerNorm, VisionTransformer


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """DICE 损失，假设 pred 已通过 sigmoid。"""
    pred_flat = pred.reshape(pred.size(0), -1)
    target_flat = target.reshape(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    score = (2 * intersection + eps) / (union + eps)
    return 1.0 - score.mean()


class SketchEncoder(VisionTransformer):
    """与 CLIP ViT 参数一致的素描编码器，输入为单通道。"""

    def __init__(self, base_visual: VisionTransformer):
        super().__init__(
            input_resolution=base_visual.input_resolution,
            patch_size=base_visual.conv1.kernel_size[0],
            width=base_visual.conv1.out_channels,
            layers=base_visual.transformer.layers,
            heads=base_visual.transformer.resblocks[0].attn.num_heads,
            output_dim=base_visual.proj.shape[1],
            input_channels=1,
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class CSTBIRModel(nn.Module):
    """三模态 CSTBIR 模型，对 CLIP 进行扩展。"""

    def __init__(
        self,
        clip_model: nn.Module,
        num_classes: int,
        detector_grid: int = 7,
        detector_boxes: int = 2,
        reconstruction_channels: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        self.clip = clip_model
        self.num_classes = num_classes
        self.detector_grid = detector_grid
        self.detector_boxes = detector_boxes

        if not hasattr(self.clip.visual, "forward_with_patches"):
            raise ValueError("视觉编码器缺少 forward_with_patches")

        self.embed_dim = self.clip.text_projection.shape[1]
        self.vision_width = self.clip.visual.conv1.out_channels
        self.patch_size = self.clip.visual.conv1.kernel_size[0]
        self.grid_size = int((self.clip.visual.input_resolution // self.patch_size))

        self.sketch_encoder = SketchEncoder(self.clip.visual)

        self.text_classifier = nn.Linear(self.embed_dim, num_classes)
        self.image_classifier = nn.Linear(self.embed_dim, num_classes)

        detector_out_channels = detector_boxes * 5 + num_classes
        self.detector_head = nn.Sequential(
            nn.Conv2d(self.vision_width, self.vision_width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.vision_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.vision_width, detector_out_channels, kernel_size=1)
        )

        if reconstruction_channels is None:
            reconstruction_channels = (
                self.vision_width // 2,
                self.vision_width // 4,
                self.vision_width // 8,
                self.vision_width // 16,
            )

        recon_layers = []
        in_ch = self.vision_width
        for out_ch in reconstruction_channels:
            recon_layers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            recon_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False))
            recon_layers.append(nn.BatchNorm2d(out_ch))
            recon_layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        recon_layers.append(nn.Conv2d(in_ch, 1, kernel_size=1))
        self.reconstruction_head = nn.Sequential(*recon_layers)

        clip_device = self.clip.visual.conv1.weight.device
        clip_dtype = self.clip.visual.conv1.weight.dtype
        self.sketch_encoder = self.sketch_encoder.to(device=clip_device, dtype=clip_dtype)
        self.text_classifier = self.text_classifier.to(device=clip_device, dtype=torch.float32)
        self.image_classifier = self.image_classifier.to(device=clip_device, dtype=torch.float32)
        self.detector_head = self.detector_head.to(device=clip_device, dtype=torch.float32)
        self.reconstruction_head = self.reconstruction_head.to(device=clip_device, dtype=torch.float32)
        self.bce_loss = nn.BCEWithLogitsLoss().to(device=clip_device, dtype=torch.float32)

        for module in self.sketch_encoder.modules():
            if isinstance(module, LayerNorm):
                module.float()
        for module in self.detector_head.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.float()
        for module in self.reconstruction_head.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.float()

    def forward(self, images: torch.Tensor, texts: torch.Tensor, sketches: torch.Tensor) -> Dict[str, torch.Tensor]:
        dtype = self.clip.dtype
        image_proj, _, _, image_sequence = self.clip.visual.forward_with_patches(
            images.type(dtype)
        )
        text_features = self.clip.encode_text(texts)
        sketch_proj, sketch_cls_token, _, _ = self.sketch_encoder.forward_with_patches(sketches.type(dtype))

        image_proj_fp32 = torch.nan_to_num(image_proj.float())
        text_features_fp32 = torch.nan_to_num(text_features.float())
        sketch_proj_fp32 = torch.nan_to_num(sketch_proj.float())

        image_sequence_fp32 = torch.nan_to_num(image_sequence.float())
        sketch_cls_token_fp32 = torch.nan_to_num(sketch_cls_token.float())

        attention_scores = torch.matmul(image_sequence_fp32, sketch_cls_token_fp32.unsqueeze(-1)).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = torch.nan_to_num(attention_weights)
        attended_sequence_fp32 = attention_weights.unsqueeze(-1) * image_sequence_fp32

        attended_patches = attended_sequence_fp32[:, 1:, :]
        h_avg = torch.nan_to_num(attended_sequence_fp32.mean(dim=1))
        image_attended_proj = h_avg.to(self.clip.visual.proj.dtype) @ self.clip.visual.proj
        image_attended_proj = torch.nan_to_num(image_attended_proj.float())

        normalized_image = self._normalize(image_proj_fp32)
        normalized_text = self._normalize(text_features_fp32)
        logits_per_image = normalized_image @ normalized_text.t()
        logit_scale = self.clip.logit_scale.float().exp()
        logits_per_image = logits_per_image * logit_scale
        logits_per_text = logits_per_image.t()

        image_cls_logits = self.image_classifier(image_attended_proj)
        text_cls_logits = self.text_classifier(text_features_fp32)

        patch_map_fp32 = attended_patches.permute(0, 2, 1).reshape(
            images.size(0), self.vision_width, self.grid_size, self.grid_size
        )
        patch_map_fp32 = torch.nan_to_num(patch_map_fp32)
        detection_map = self.detector_head(
            F.adaptive_avg_pool2d(patch_map_fp32, (self.detector_grid, self.detector_grid))
        )

        reconstruction_logits = self.reconstruction_head(patch_map_fp32)
        reconstruction_logits = torch.nan_to_num(reconstruction_logits)

        attention_patches = attention_weights[:, 1:].reshape(
            images.size(0), self.grid_size, self.grid_size
        )

        return {
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
            "image_cls_logits": image_cls_logits,
            "text_cls_logits": text_cls_logits,
            "detection_map": detection_map,
            "reconstruction_logits": reconstruction_logits,
            "attention_map": attention_patches,
            "image_features": image_proj_fp32,
            "text_features": text_features_fp32,
            "sketch_features": sketch_proj_fp32,
        }

    def reconstruction_loss(self, recon_logits: torch.Tensor, target_sketch: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
        recon_logits = recon_logits.float()
        target_sketch = target_sketch.float()
        bce = self.bce_loss(recon_logits, target_sketch)
        dice = dice_loss(torch.sigmoid(recon_logits), target_sketch)
        return alpha * bce + beta * dice

    @staticmethod
    def _normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        x = torch.nan_to_num(x.float())
        denom = torch.clamp(x.norm(dim=-1, keepdim=True), min=eps)
        return x / denom

    def detection_loss(self, prediction: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        if target is None:
            return prediction.new_tensor(0.0)
        return F.mse_loss(prediction.float(), target.float())