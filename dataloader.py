from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src import clip

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class ManifestEntry:
    """In-memory representation of a manifest row."""

    idx: int
    id: str
    category: str
    image_path: Path
    text: str
    positive_sketches: List[Path]
    negative_sketches: List[Path]
    negative_images: List[Path]
    negative_texts: List[str]


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
    """Dataset that yields positive/negative triplets for each modality."""

    def __init__(
        self,
        manifest_path: Path,
        clip_preprocess,
        sketch_transform: Optional[transforms.Compose] = None,
        negatives_per_modality: Optional[Dict[str, int]] = None,
        clip_context_length: int = 77,
    ):
        self.manifest_path = Path(manifest_path)
        self.clip_preprocess = clip_preprocess
        self.sketch_transform = sketch_transform or default_sketch_transform(224)
        self.neg_counts = negatives_per_modality or {"sketch": 4, "image": 4, "text": 4}
        self.context_length = clip_context_length

        manifest_rows = load_manifest(self.manifest_path)
        self.entries: List[ManifestEntry] = []
        for idx, row in enumerate(manifest_rows):
            entry = ManifestEntry(
                idx=idx,
                id=row["id"],
                category=row.get("category", "unknown"),
                image_path=Path(row["image"]).resolve(),
                text=row["text"],
                positive_sketches=[Path(p).resolve() for p in row.get("positive_sketches", [])],
                negative_sketches=[Path(p).resolve() for p in row.get("negative_sketches", [])],
                negative_images=[Path(p).resolve() for p in row.get("negative_images", [])],
                negative_texts=list(row.get("negative_texts", [])),
            )
            if not entry.positive_sketches:
                raise ValueError(f"entry {entry.id} missing positive sketches")
            self.entries.append(entry)

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
            "meta": {
                "id": entry.id,
                "category": entry.category,
                "image_path": str(entry.image_path),
                "sketch_path": str(sketch_path),
                "negative_image_paths": [str(p) for p in neg_image_paths],
                "negative_sketch_paths": [str(p) for p in neg_sketch_paths],
                "negative_texts": neg_text_values,
                "text": entry.text,
            },
        }

    # ------------------------------------------------------------------
    # loading helpers
    # ------------------------------------------------------------------
    def _load_image(self, path: Path) -> torch.Tensor:
        with Image.open(path) as img:
            img = img.convert("RGB")
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
                return self.clip_preprocess(img)
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


def sbir_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    texts = torch.stack([item["text_tokens"] for item in batch], dim=0)
    sketches_resnet = torch.stack([item["sketch_resnet"] for item in batch], dim=0)
    sketches_clip = torch.stack([item["sketch_clip"] for item in batch], dim=0)

    neg_images = _pad_and_stack(batch, key="neg_images", like=images)
    neg_texts = _pad_and_stack(batch, key="neg_text_tokens", like=texts)
    neg_sketch_resnet = _pad_and_stack(batch, key="neg_sketches_resnet", like=sketches_resnet)
    neg_sketch_clip = _pad_and_stack(batch, key="neg_sketches_clip", like=sketches_clip)

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
