import json
import os
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from src import clip

class CSTBIR_dataset:
    """支持文本、图像与素描三模态的数据管线。"""

    _offset_cache: Dict[str, Dict[int, int]] = {}
    _label_to_idx: Dict[str, int] = {}
    _idx_to_label: List[str] = []
    _sketch_root: Optional[str] = None

    def __init__(self, data_path: str, image_files_path: str, sketch_files_path: str, batch_size: int, split: str, preprocess):
        self.batch_size = batch_size
        self.preprocess = preprocess
        self.image_root = image_files_path
        self.sketch_root = sketch_files_path

        full_df = pd.read_json(data_path, lines=False)
        full_df = full_df.dropna(subset=["label", "sketch", "image", "text", "split"])

        if not CSTBIR_dataset._label_to_idx:
            labels = sorted(full_df["label"].unique())
            CSTBIR_dataset._label_to_idx = {label: idx for idx, label in enumerate(labels)}
            CSTBIR_dataset._idx_to_label = labels

        self.data = full_df[full_df["split"] == split].reset_index(drop=True)
        if self.data.empty:
            raise RuntimeError(f"未找到 split={split} 的数据样本。")

        self.records: List[Dict[str, object]] = []
        label_sketch_indices: Dict[str, set] = defaultdict(set)

        self._register_sketch_root()

        for row in self.data.itertuples(index=False):
            label = getattr(row, "label")
            sketch_name = getattr(row, "sketch")
            sketch_label, sketch_idx = self._parse_sketch_name(sketch_name)
            label_id = CSTBIR_dataset._label_to_idx[label]

            record = {
                "image_path": os.path.join(self.image_root, getattr(row, "image")),
                "text": getattr(row, "text"),
                "label": label,
                "label_id": label_id,
                "sketch_label": sketch_label,
                "sketch_idx": sketch_idx,
            }
            self.records.append(record)
            label_sketch_indices[sketch_label].add(sketch_idx)

        self.n_samples = len(self.records)
        self._build_conflict_maps()

        for sketch_label, indices in label_sketch_indices.items():
            self._ensure_offsets(sketch_label, indices)

        self._sketch_cache: "OrderedDict[Tuple[str, int], torch.Tensor]" = OrderedDict()
        self._sketch_cache_limit = 2048
        self._image_cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self._image_cache_limit = 2048
        self._text_cache: Dict[str, torch.Tensor] = {}

    @staticmethod
    def label_to_id(label: str) -> int:
        return CSTBIR_dataset._label_to_idx[label]

    @staticmethod
    def id_to_label(idx: int) -> str:
        return CSTBIR_dataset._idx_to_label[idx]

    def _build_conflict_maps(self):
        self.image2sampleidx: Dict[str, List[int]] = defaultdict(list)
        self.text2image: Dict[str, List[str]] = defaultdict(list)
        self.image2text: Dict[str, List[str]] = defaultdict(list)

        for idx, record in enumerate(self.records):
            image_path = record["image_path"]
            text = record["text"]
            self.image2sampleidx[image_path].append(idx)
            self.image2text[image_path].append(text)
            self.text2image[text].append(image_path)

    def __len__(self):
        return max(1, self.n_samples // self.batch_size)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        image = self._get_image_tensor(record["image_path"])
        text = self._get_text_tokens(record["text"])
        sketch = self._get_sketch_tensor(record["sketch_label"], record["sketch_idx"])  # (1, 224, 224)
        label_id = record["label_id"]
        return image, text, sketch, label_id

    def is_text_conflict(self, text: str, images: List[str]) -> bool:
        return bool(set(self.text2image[text]) & set(images))

    def is_image_conflict(self, image: str, texts: List[str]) -> bool:
        return bool(set(self.image2text[image]) & set(texts))

    def get_samples(self) -> Dict[str, torch.Tensor]:
        selected_images: List[str] = []
        selected_texts: List[str] = []
        batch_images: List[torch.Tensor] = []
        batch_texts: List[torch.Tensor] = []
        batch_sketches: List[torch.Tensor] = []
        batch_labels: List[int] = []

        while len(selected_images) < self.batch_size:
            sample_idx = np.random.randint(0, self.n_samples)
            record = self.records[sample_idx]
            image_path = record["image_path"]
            text = record["text"]

            if image_path in selected_images or text in selected_texts:
                continue
            if self.is_text_conflict(text, selected_images):
                continue
            if self.is_image_conflict(image_path, selected_texts):
                continue

            selected_images.append(image_path)
            selected_texts.append(text)

            batch_images.append(self._get_image_tensor(image_path))
            batch_texts.append(self._get_text_tokens(text))
            batch_sketches.append(self._get_sketch_tensor(record["sketch_label"], record["sketch_idx"]))
            batch_labels.append(record["label_id"])

        images_tensor = torch.stack(batch_images, dim=0)
        texts_tensor = torch.stack(batch_texts, dim=0)
        sketches_tensor = torch.stack(batch_sketches, dim=0)
        labels_tensor = torch.tensor(batch_labels, dtype=torch.long)

        return {
            "images": images_tensor,
            "texts": texts_tensor,
            "sketches": sketches_tensor,
            "labels": labels_tensor,
        }

    @staticmethod
    def _parse_sketch_name(name: str) -> Tuple[str, int]:
        stem = Path(name).stem
        if "_" not in stem:
            raise ValueError(f"无法解析素描文件名: {name}")
        label, index = stem.split("_", 1)
        return label, int(index)

    @classmethod
    def _ensure_offsets(cls, label: str, indices: set):
        cache = cls._offset_cache.setdefault(label, {})
        missing = set(indices) - set(cache.keys())
        if not missing:
            return

        if cls._sketch_root is None:
            raise RuntimeError("请先设置素描根目录。")
        ndjson_path = Path(cls._sketch_root) / f"{label}.ndjson"
        if not ndjson_path.exists():
            raise FileNotFoundError(f"未找到素描 ndjson 文件: {ndjson_path}")

        with ndjson_path.open("rb") as reader:
            offset = reader.tell()
            for line_idx, _ in enumerate(reader):
                if line_idx in missing:
                    cache[line_idx] = offset
                    if len(cache) == len(indices):
                        break
                offset = reader.tell()

    def _register_sketch_root(self):
        if CSTBIR_dataset._sketch_root is None:
            CSTBIR_dataset._sketch_root = self.sketch_root

    def _get_sketch_tensor(self, label: str, index: int) -> torch.Tensor:
        key = (label, index)
        cached = self._sketch_cache.get(key)
        if cached is not None:
            self._sketch_cache.move_to_end(key)
            return cached.clone()

        offset = self._offset_cache[label][index]
        ndjson_path = Path(self.sketch_root) / f"{label}.ndjson"
        with ndjson_path.open("rb") as reader:
            reader.seek(offset)
            line = reader.readline().decode("utf-8").strip()
        record = json.loads(line)
        drawing = record["drawing"]
        tensor = self._drawing_to_tensor(drawing)

        self._sketch_cache[key] = tensor
        if len(self._sketch_cache) > self._sketch_cache_limit:
            self._sketch_cache.popitem(last=False)
        return tensor.clone()

    @staticmethod
    def _drawing_to_tensor(drawing: List[List[List[int]]], size: int = 224, line_width: int = 3) -> torch.Tensor:
        image = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(image)
        scale = (size - 1) / 255.0

        for stroke_x, stroke_y in drawing:
            if not stroke_x or not stroke_y:
                continue
            points = [(x * scale, y * scale) for x, y in zip(stroke_x, stroke_y)]
            if len(points) == 1:
                draw.point(points[0], fill=255)
            else:
                draw.line(points, fill=255, width=line_width, joint="curve")

        array = np.array(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0).contiguous()
        return tensor

    def _get_image_tensor(self, image_path: str) -> torch.Tensor:
        cached = self._image_cache.get(image_path)
        if cached is not None:
            self._image_cache.move_to_end(image_path)
            return cached.clone()

        tensor = self.preprocess(Image.open(image_path))
        self._image_cache[image_path] = tensor
        if len(self._image_cache) > self._image_cache_limit:
            self._image_cache.popitem(last=False)
        return tensor.clone()

    def _get_text_tokens(self, text: str) -> torch.Tensor:
        cached = self._text_cache.get(text)
        if cached is None:
            cached = clip.tokenize(text).squeeze(0)
            self._text_cache[text] = cached
        return cached.clone()


__all__ = ["CSTBIR_dataset"]
