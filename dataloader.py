from __future__ import annotations

import io
import json
import multiprocessing as mp
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src import clip

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class ManifestEntry:
    idx: int
    id: str
    category: str
    category_id: int
    virtual_class: Optional[str]
    virtual_class_id: int
    image_path: Path
    text: str
    positive_sketches: List[Path]
    negative_sketches: List[Path]
    negative_images: List[Path]
    negative_texts: List[str]


class SharedNoiseState:
    def __init__(self, prob: float, strength: float) -> None:
        self._prob = mp.Value("d", float(prob))
        self._strength = mp.Value("d", float(strength))

    def set(self, prob: float, strength: float) -> None:
        with self._prob.get_lock():
            self._prob.value = float(prob)
        with self._strength.get_lock():
            self._strength.value = float(strength)

    def get(self) -> Tuple[float, float]:
        with self._prob.get_lock():
            prob = float(self._prob.value)
        with self._strength.get_lock():
            strength = float(self._strength.value)
        return prob, strength


class CurriculumNoiseSchedule:
    def __init__(self, config: Dict[str, object]) -> None:
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enable", True))
        self.start_epoch = int(cfg.get("start_epoch", 0))
        self.initial_prob = float(cfg.get("initial_prob", 0.0))
        self.max_prob = float(cfg.get("max_prob", 0.5))
        self.initial_strength = float(cfg.get("initial_strength", 0.2))
        self.max_strength = float(cfg.get("max_strength", 1.0))
        self.ramp_epochs = int(cfg.get("ramp_epochs", 5))

        max_epoch = cfg.get("max_epoch")
        try:
            max_epoch_int = None if max_epoch is None else int(max_epoch)
        except (TypeError, ValueError):
            max_epoch_int = None
        if max_epoch_int is not None:
            ramp = max_epoch_int - self.start_epoch
            if ramp > 0:
                self.ramp_epochs = ramp

        per_stage = cfg.get("per_stage", {})
        self.per_stage: Dict[str, Dict[str, object]] = {
            str(name).lower(): dict(value)
            for name, value in per_stage.items()
            if isinstance(value, dict)
        }

    def compute(self, epoch: int, total_epochs: Optional[int], stage: Optional[str]) -> Tuple[float, float]:
        if not self.enabled:
            return 0.0, 0.0
        stage_cfg = self._resolve_stage(stage)
        if not bool(stage_cfg.get("enable", True)):
            return 0.0, 0.0
        start_epoch = int(stage_cfg.get("start_epoch", self.start_epoch))
        initial_prob = float(stage_cfg.get("initial_prob", self.initial_prob))
        max_prob = float(stage_cfg.get("max_prob", self.max_prob))
        initial_strength = float(stage_cfg.get("initial_strength", self.initial_strength))
        max_strength = float(stage_cfg.get("max_strength", self.max_strength))
        ramp_epochs = stage_cfg.get("ramp_epochs")
        if ramp_epochs is None:
            ramp_epochs = self.ramp_epochs
            stage_max_epoch = stage_cfg.get("max_epoch")
            if stage_max_epoch is not None:
                stage_max_epoch = int(stage_max_epoch)
                ramp = stage_max_epoch - start_epoch
                if ramp > 0:
                    ramp_epochs = ramp
        ramp_epochs = int(ramp_epochs) if ramp_epochs is not None else 1
        if ramp_epochs <= 0:
            ramp_epochs = max(1, int(total_epochs) if total_epochs is not None else 1)

        if epoch <= start_epoch:
            return initial_prob, initial_strength

        progress = (epoch - start_epoch) / float(ramp_epochs)
        progress = max(0.0, min(1.0, progress))

        prob = initial_prob + (max_prob - initial_prob) * progress
        strength = initial_strength + (max_strength - initial_strength) * progress
        return prob, strength

    def _resolve_stage(self, stage: Optional[str]) -> Dict[str, object]:
        if not stage:
            return {}
        return self.per_stage.get(stage.lower(), {})


class CurriculumNoiseAugmentor:
    def __init__(self, config: Dict[str, object]) -> None:
        self.config = config
        self.image_cfg: Dict[str, object] = dict(config.get("image", {}))
        self.sketch_cfg: Dict[str, object] = dict(config.get("sketch", {}))

        if not self.image_cfg:
            self.image_cfg = {
                "gaussian_sigma": [0.0, 0.12],
                "blur_radius": [0.0, 1.2],
                "jpeg_quality": [45, 85],
                "cutout_ratio": [0.05, 0.12],
            }
        if not self.sketch_cfg:
            self.sketch_cfg = {
                "gaussian_sigma": [0.0, 0.18],
                "blur_radius": [0.0, 0.6],
                "contrast": [0.85, 1.2],
                "cutout_ratio": [0.02, 0.06],
            }

    def apply(self, img: Image.Image, kind: str, prob: float, strength: float) -> Image.Image:
        if prob <= 0.0 or strength <= 0.0:
            return img
        if random.random() >= prob:
            return img
        if kind == "sketch":
            return self._apply_sketch_noise(img, strength)
        return self._apply_image_noise(img, strength)

    def _apply_image_noise(self, img: Image.Image, strength: float) -> Image.Image:
        cfg = self.image_cfg
        sigma = self._interp(cfg.get("gaussian_sigma"), strength)
        if sigma > 0:
            img = self._gaussian_noise(img, sigma)

        radius = self._interp(cfg.get("blur_radius"), strength)
        if radius > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        quality_range = cfg.get("jpeg_quality")
        if quality_range:
            quality = int(self._interp(quality_range, 1.0 - strength, clamp=True))
            quality = max(10, min(quality, 95))
            img = self._jpeg_compress(img, quality)

        cutout_ratio = self._interp(cfg.get("cutout_ratio"), strength)
        if cutout_ratio > 0:
            img = self._random_cutout(img, cutout_ratio)

        return img

    def _apply_sketch_noise(self, img: Image.Image, strength: float) -> Image.Image:
        cfg = self.sketch_cfg
        sigma = self._interp(cfg.get("gaussian_sigma"), strength)
        if sigma > 0:
            img = self._gaussian_noise(img, sigma)

        radius = self._interp(cfg.get("blur_radius"), strength)
        if radius > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        contrast_range = cfg.get("contrast")
        if contrast_range:
            factor = self._interp(contrast_range, strength)
            if factor > 0:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(factor)

        cutout_ratio = self._interp(cfg.get("cutout_ratio"), strength)
        if cutout_ratio > 0:
            img = self._random_cutout(img, cutout_ratio, fill=255)

        return img

    @staticmethod
    def _interp(values: Optional[Sequence[float]], strength: float, clamp: bool = False) -> float:
        if not values:
            return 0.0
        if isinstance(values, (int, float)):
            return float(values)
        start, end = float(values[0]), float(values[-1])
        strength = max(0.0, min(1.0, strength)) if clamp else strength
        return start + (end - start) * strength

    @staticmethod
    def _gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
        array = np.asarray(img).astype(np.float32)
        noise = np.random.normal(0.0, sigma * 255.0, array.shape).astype(np.float32)
        array = np.clip(array + noise, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(array)

    @staticmethod
    def _jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        degraded = Image.open(buffer)
        degraded.load()
        return degraded.convert(img.mode)

    @staticmethod
    def _random_cutout(img: Image.Image, ratio: float, fill: Optional[int] = None) -> Image.Image:
        ratio = max(0.0, min(ratio, 0.8))
        if ratio <= 0.0:
            return img
        width, height = img.size
        cut_w = max(1, int(width * (ratio ** 0.5)))
        cut_h = max(1, int(height * (ratio ** 0.5)))
        x0 = random.randint(0, max(0, width - cut_w))
        y0 = random.randint(0, max(0, height - cut_h))
        x1 = min(width, x0 + cut_w)
        y1 = min(height, y0 + cut_h)
        draw = ImageDraw.Draw(img)
        if img.mode == "RGB":
            fill_color = (fill, fill, fill) if fill is not None else (0, 0, 0)
        else:
            fill_color = fill if fill is not None else 0
        draw.rectangle([x0, y0, x1, y1], fill=fill_color)
        return img



def default_sketch_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

def load_manifest(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


class SBIRDataset(Dataset):
    """Dataset that yields positive/negative tuples for SBIR training."""

    def __init__(
        self,
        manifest_path: Path,
        clip_preprocess,
        sketch_transform: Optional[transforms.Compose] = None,
        negatives_per_modality: Optional[Dict[str, int]] = None,
        clip_context_length: int = 77,
        category_mapping: Optional[Dict[str, int]] = None,
        virtual_mapping: Optional[Dict[str, int]] = None,
        curriculum_noise: Optional[Dict[str, object]] = None,
        apply_noise: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.clip_preprocess = clip_preprocess
        self.sketch_transform = sketch_transform or default_sketch_transform(224)
        self.neg_counts = negatives_per_modality or {"sketch": 4, "image": 4, "text": 4}
        self.context_length = clip_context_length
        self._noise_schedule: Optional[CurriculumNoiseSchedule] = None
        self._noise_state: Optional[SharedNoiseState] = None
        self._noise_augmentor: Optional[CurriculumNoiseAugmentor] = None
        self._noise_enabled = False

        if curriculum_noise and apply_noise:
            schedule_cfg = self._extract_schedule_cfg(curriculum_noise)
            augment_cfg = self._extract_augment_cfg(curriculum_noise)
            self._noise_schedule = CurriculumNoiseSchedule(schedule_cfg)
            if self._noise_schedule.enabled:
                self._noise_state = SharedNoiseState(
                    prob=self._noise_schedule.initial_prob,
                    strength=self._noise_schedule.initial_strength,
                )
                self._noise_augmentor = CurriculumNoiseAugmentor(augment_cfg)
                self._noise_enabled = True

        manifest_rows = load_manifest(self.manifest_path)
        category_map: Dict[str, int] = {str(k): int(v) for k, v in (category_mapping or {}).items()}
        virtual_map: Dict[str, int] = {str(k): int(v) for k, v in (virtual_mapping or {}).items()}

        self.entries: List[ManifestEntry] = []
        seen_category_ids = set()
        seen_categories = set()
        seen_virtual_ids = set()
        seen_virtual_classes = set()

        for idx, row in enumerate(manifest_rows):
            category_id = self._resolve_category_id(row, category_map)
            vclass_raw = row.get("virtual_class")
            if vclass_raw is not None:
                vclass = str(vclass_raw)
                vclass_id = virtual_map.setdefault(vclass, len(virtual_map))
            else:
                vclass = None
                vclass_id = -1

            entry = ManifestEntry(
                idx=idx,
                id=row["id"],
                category=row.get("category", "unknown"),
                category_id=category_id,
                virtual_class=vclass,
                virtual_class_id=vclass_id,
                image_path=Path(row["image"]).resolve(),
                text=row["text"],
                positive_sketches=[Path(p).resolve() for p in row.get("positive_sketches", [])],
                negative_sketches=[Path(p).resolve() for p in row.get("negative_sketches", [])],
                negative_images=[Path(p).resolve() for p in row.get("negative_images", [])],
                negative_texts=list(row.get("negative_texts", [])),
            )

            if not entry.positive_sketches:
                raise ValueError(f"entry {entry.id} missing positive sketches")

            seen_category_ids.add(entry.category_id)
            seen_categories.add(entry.category)
            if entry.virtual_class_id >= 0:
                seen_virtual_ids.add(entry.virtual_class_id)
                if entry.virtual_class is not None:
                    seen_virtual_classes.add(entry.virtual_class)

            self.entries.append(entry)

        self.category_to_index = category_map
        self.category_ids_present = frozenset(seen_category_ids)
        self.categories_present = frozenset(seen_categories)
        self.num_categories = len(self.categories_present)

        self.virtual_class_to_index = virtual_map
        self.virtual_class_ids_present = frozenset(seen_virtual_ids)
        self.virtual_classes_present = frozenset(seen_virtual_classes)
        self.num_virtual_classes = len(self.virtual_classes_present)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        entry = self.entries[index]
        image_tensor = self._load_image(entry.image_path)
        text_tokens = self._tokenize(entry.text)

        sketch_path = random.choice(entry.positive_sketches)
        sketch_resnet = self._load_sketch(sketch_path, to_clip=False)
        sketch_clip = self._load_sketch(sketch_path, to_clip=True)

        neg_sketch_paths = self._sample(entry.negative_sketches, self.neg_counts.get("sketch", 0))
        neg_image_paths = self._sample(entry.negative_images, self.neg_counts.get("image", 0))
        neg_text_values = self._sample(entry.negative_texts, self.neg_counts.get("text", 0))

        neg_sketch_resnet = self._load_sketch_batch(neg_sketch_paths, to_clip=False, like=sketch_resnet)
        neg_sketch_clip = self._load_sketch_batch(neg_sketch_paths, to_clip=True, like=sketch_clip)
        neg_image_tensor = self._load_image_batch(neg_image_paths, like=image_tensor)
        neg_text_tokens = self._tokenize_batch(neg_text_values)

        return {
            "image": image_tensor,
            "text_tokens": text_tokens,
            "sketch_resnet": sketch_resnet,
            "sketch_clip": sketch_clip,
            "neg_images": neg_image_tensor,
            "neg_text_tokens": neg_text_tokens,
            "neg_sketches_resnet": neg_sketch_resnet,
            "neg_sketches_clip": neg_sketch_clip,
            "category_id": entry.category_id,
            "virtual_class_id": entry.virtual_class_id,
            "meta": {
                "id": entry.id,
                "category": entry.category,
                "category_id": entry.category_id,
                "virtual_class": entry.virtual_class,
                "virtual_class_id": entry.virtual_class_id,
                "image_path": str(entry.image_path),
                "sketch_path": str(sketch_path),
                "negative_image_paths": [str(p) for p in neg_image_paths],
                "negative_sketch_paths": [str(p) for p in neg_sketch_paths],
                "negative_texts": neg_text_values,
                "text": entry.text,
            },
        }

    def _load_image(self, path: Path) -> torch.Tensor:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = self._apply_noise_if_needed(img, kind="image")
            return self.clip_preprocess(img)

    def _load_image_batch(self, paths: Sequence[Path], like: torch.Tensor) -> torch.Tensor:
        if not paths:
            return like.new_zeros((0,) + like.shape)
        tensors = [self._load_image(p) for p in paths]
        return torch.stack(tensors, dim=0)

    def _load_sketch(self, path: Path, to_clip: bool) -> torch.Tensor:
        with Image.open(path) as img:
            img = img.convert("RGB")
            if to_clip:
                img = self._apply_noise_if_needed(img, kind="sketch")
                return self.clip_preprocess(img)
            img = self._apply_noise_if_needed(img, kind="sketch")
            return self.sketch_transform(img)

    def _load_sketch_batch(self, paths: Sequence[Path], to_clip: bool, like: torch.Tensor) -> torch.Tensor:
        if not paths:
            return like.new_zeros((0,) + like.shape)
        tensors = [self._load_sketch(p, to_clip=to_clip) for p in paths]
        return torch.stack(tensors, dim=0)

    def _tokenize(self, text: str) -> torch.Tensor:
        return clip.tokenize([text], context_length=self.context_length, truncate=True).squeeze(0)

    def _tokenize_batch(self, texts: Sequence[str]) -> torch.Tensor:
        if not texts:
            return torch.zeros((0, self.context_length), dtype=torch.long)
        tokens = clip.tokenize(list(texts), context_length=self.context_length, truncate=True)
        return tokens

    @staticmethod
    def _sample(sequence: Sequence, count: int) -> List:
        if count <= 0 or not sequence:
            return []
        if len(sequence) >= count:
            return random.sample(list(sequence), count)
        return [random.choice(list(sequence)) for _ in range(count)]

    @staticmethod
    def _resolve_category_id(row: Dict[str, object], mapping: Dict[str, int]) -> int:
        category = str(row.get("category", "unknown"))
        if "category_id" in row:
            category_id = int(row["category_id"])
            if category in mapping and mapping[category] != category_id:
                raise ValueError(
                    f"category id mismatch for {category}: manifest {category_id} vs mapping {mapping[category]}"
                )
            mapping.setdefault(category, category_id)
            return category_id
        if category in mapping:
            return mapping[category]
        category_id = len(mapping)
        mapping[category] = category_id
        return category_id

    @staticmethod
    def _extract_schedule_cfg(config: Dict[str, object]) -> Dict[str, object]:
        if not isinstance(config, dict):
            return {}
        schedule = config.get("schedule")
        if isinstance(schedule, dict):
            return dict(schedule)
        keys = {
            "enable",
            "start_epoch",
            "initial_prob",
            "max_prob",
            "initial_strength",
            "max_strength",
            "ramp_epochs",
            "max_epoch",
            "per_stage",
        }
        return {key: config[key] for key in keys if key in config}

    @staticmethod
    def _extract_augment_cfg(config: Dict[str, object]) -> Dict[str, object]:
        if not isinstance(config, dict):
            return {}
        augment = config.get("augment")
        if isinstance(augment, dict):
            return dict(augment)
        return {key: config[key] for key in ("image", "sketch") if key in config}

    def update_noise(self, epoch: int, total_epochs: Optional[int] = None, stage: Optional[str] = None) -> None:
        if not self._noise_enabled or self._noise_schedule is None or self._noise_state is None:
            return
        prob, strength = self._noise_schedule.compute(epoch, total_epochs, stage)
        self._noise_state.set(prob, strength)

    def noise_parameters(self) -> Optional[Tuple[float, float]]:
        if not self._noise_enabled or self._noise_state is None:
            return None
        return self._noise_state.get()

    def _apply_noise_if_needed(self, img: Image.Image, kind: str) -> Image.Image:
        if not self._noise_enabled or self._noise_state is None or self._noise_augmentor is None:
            return img
        prob, strength = self._noise_state.get()
        return self._noise_augmentor.apply(img, kind, prob, strength)


def sbir_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    texts = torch.stack([item["text_tokens"] for item in batch], dim=0)
    sketches_resnet = torch.stack([item["sketch_resnet"] for item in batch], dim=0)
    sketches_clip = torch.stack([item["sketch_clip"] for item in batch], dim=0)

    neg_images = _pad_and_stack(batch, key="neg_images", like=images)
    neg_texts = _pad_and_stack(batch, key="neg_text_tokens", like=texts)
    neg_sketch_resnet = _pad_and_stack(batch, key="neg_sketches_resnet", like=sketches_resnet)
    neg_sketch_clip = _pad_and_stack(batch, key="neg_sketches_clip", like=sketches_clip)

    category_ids = torch.tensor([item["category_id"] for item in batch], dtype=torch.long)
    virtual_class_ids = torch.tensor([item["virtual_class_id"] for item in batch], dtype=torch.long)
    meta = [item["meta"] for item in batch]

    return {
        "images": images,
        "texts": texts,
        "sketches_resnet": sketches_resnet,
        "sketches_clip": sketches_clip,
        "neg_images": neg_images,
        "neg_texts": neg_texts,
        "neg_sketches_resnet": neg_sketch_resnet,
        "neg_sketches_clip": neg_sketch_clip,
        "category_ids": category_ids,
        "virtual_class_ids": virtual_class_ids,
        "meta": meta,
    }


def _pad_and_stack(batch: List[Dict[str, torch.Tensor]], key: str, like: torch.Tensor) -> torch.Tensor:
    tensors = [item[key] for item in batch]
    if not tensors:
        return like.new_zeros((0,))
    max_len = max(t.size(0) for t in tensors)
    if max_len == 0:
        return like.new_zeros((len(batch), 0) + like.shape[1:])

    padded = []
    for tensor in tensors:
        if tensor.size(0) == max_len:
            padded.append(tensor)
            continue
        pad_shape = (max_len - tensor.size(0),) + tensor.size()[1:]
        pad_tensor = tensor.new_zeros(pad_shape)
        padded.append(torch.cat([tensor, pad_tensor], dim=0))
    return torch.stack(padded, dim=0)


__all__ = ["SBIRDataset", "sbir_collate", "default_sketch_transform", "load_manifest"]
