"""Utility to construct positive and negative samples for SBIR training.

This script reads the raw dataset splits defined in ``config.yaml`` and writes
JSONL manifest files for train/val/test splits. Each manifest row contains the
paths required to build three-modal tuples together with negative candidates
for sketch, image, and text modalities.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import yaml
from tqdm import tqdm


@dataclass
class RawSample:
    """Single SBIR sample before negative assignment."""

    uid: str
    category: str
    split: str
    image_path: Path
    sketch_paths: List[Path]
    text: str
    source: str
    virtual_class: Optional[str] = None

    def to_record(
        self,
        root: Path,
        sketch_negatives: Sequence[Path],
        image_negatives: Sequence[Path],
        text_negatives: Sequence[str],
    ) -> Dict[str, object]:
        return {
            "id": self.uid,
            "category": self.category,
            "split": self.split,
            "source": self.source,
            "image": relpath(self.image_path, root),
            "text": self.text,
            "positive_sketches": [relpath(p, root) for p in self.sketch_paths],
            "negative_sketches": [relpath(p, root) for p in sketch_negatives],
            "negative_images": [relpath(p, root) for p in image_negatives],
            "negative_texts": list(text_negatives),
            "virtual_class": self.virtual_class,
        }


def relpath(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def read_text_pairs(path: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            # Expect ``image\ttext`` format.
            parts = raw.split("\t", maxsplit=1)
            if len(parts) == 1:
                image_name = parts[0].strip()
                text = ""
            else:
                image_name, text = parts
            pairs.append((image_name.strip(), text.strip()))
    return pairs


def load_sketches(sketch_root: Path, image_name: str) -> List[Path]:
    stem = Path(image_name).stem
    candidates = sorted(sketch_root.glob(f"{stem}_*.png"))
    return candidates


def collect_noise(noise_dir: Optional[str]) -> List[Path]:
    if not noise_dir:
        return []
    path = Path(noise_dir).resolve()
    if not path.exists():
        return []
    return sorted(p for p in path.glob("*.png"))


def resolve_virtual_class_dir(category: str, split_cfg: Dict[str, object]) -> Optional[Path]:
    explicit = split_cfg.get("virtual_class_dir")
    if explicit:
        candidate = Path(str(explicit)).resolve()
        if candidate.exists():
            return candidate
    image_dir = split_cfg.get("image_dir")
    if not image_dir:
        return None
    image_root = Path(str(image_dir)).resolve()
    for parent in image_root.parents:
        candidate = (parent / f"{category}_cls").resolve()
        if candidate.exists():
            return candidate
    return None


def build_virtual_class_lookup(root: Optional[Path]) -> Dict[str, str]:
    if not root or not root.exists():
        return {}
    mapping: Dict[str, str] = {}
    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir():
            continue
        cluster = cls_dir.name
        for file_path in cls_dir.iterdir():
            if not file_path.is_file():
                continue
            mapping.setdefault(file_path.name, cluster)
    return mapping


def make_cluster_key(category: str, virtual_class: Optional[str]) -> Optional[str]:
    if not virtual_class:
        return None
    return f"{category}::{virtual_class}"


def split_train_val(samples: List[RawSample], ratio: float, split_name: str) -> Tuple[List[RawSample], List[RawSample]]:
    if ratio <= 0.0 or len(samples) < 2:
        return samples, []
    n_val = max(1, int(round(len(samples) * ratio)))
    random.shuffle(samples)
    val_samples = []
    for _ in range(n_val):
        val_samples.append(samples.pop())
    for sample in val_samples:
        sample.split = "val"
    for sample in samples:
        sample.split = split_name
    return samples, val_samples


def sample_paths(candidates: Sequence[Path], exclude: Iterable[Path], count: int) -> List[Path]:
    exclude_set = {p.resolve() for p in exclude}
    filtered = [p for p in candidates if p.resolve() not in exclude_set]
    if not filtered:
        return []
    if len(filtered) >= count:
        return random.sample(filtered, count)
    # Allow repetition when the candidate pool is small.
    return [random.choice(filtered) for _ in range(count)]


def sample_texts(candidates: Sequence[str], current: str, count: int) -> List[str]:
    filtered = [c for c in candidates if c != current]
    if not filtered:
        return []
    if len(filtered) >= count:
        return random.sample(filtered, count)
    return [random.choice(filtered) for _ in range(count)]


def build_negative_pools(samples: List[RawSample], noise_paths: Sequence[Path]) -> Tuple[List[Path], List[Path], List[str]]:
    sketch_pool: List[Path] = []
    image_pool: List[Path] = []
    text_pool: List[str] = []
    for sample in samples:
        sketch_pool.extend(sample.sketch_paths)
        image_pool.append(sample.image_path)
        text_pool.append(sample.text)
    sketch_pool.extend(noise_paths)
    return sketch_pool, image_pool, text_pool


def assign_negatives(
    samples: List[RawSample],
    cluster_images: Dict[str, Sequence[Path]],
    cluster_sketches: Dict[str, Sequence[Path]],
    global_sketch_pool: Sequence[Path],
    global_image_pool: Sequence[Path],
    text_pool: Sequence[str],
    neg_cfg: Dict[str, int],
) -> List[Dict[str, object]]:
    root = Path.cwd().resolve()
    records: List[Dict[str, object]] = []
    for sample in tqdm(samples, desc="assign negatives", leave=False):
        cluster = make_cluster_key(sample.category, sample.virtual_class)

        sketch_count = neg_cfg.get("sketch", 0)
        image_count = neg_cfg.get("image", 0)

        sketch_candidates = cluster_sketches.get(cluster, ()) if cluster else ()
        image_candidates = cluster_images.get(cluster, ()) if cluster else ()

        sketch_neg = sample_paths(sketch_candidates, sample.sketch_paths, sketch_count)
        if not sketch_neg and sketch_count > 0:
            sketch_neg = sample_paths(global_sketch_pool, sample.sketch_paths, sketch_count)

        image_neg = sample_paths(image_candidates, [sample.image_path], image_count)
        if not image_neg and image_count > 0:
            image_neg = sample_paths(global_image_pool, [sample.image_path], image_count)

        text_neg = sample_texts(text_pool, sample.text, neg_cfg.get("text", 0))
        records.append(sample.to_record(root, sketch_neg, image_neg, text_neg))
    return records


def generate_manifests(config: Dict[str, object], seed: int) -> None:
    random.seed(seed)

    data_cfg = config["data"]
    datasets_cfg = data_cfg["datasets"]
    val_ratio = float(data_cfg.get("val_ratio", 0.0))
    neg_cfg = data_cfg.get("negatives_per_modality", {})

    train_samples: List[RawSample] = []
    val_samples: List[RawSample] = []
    test_samples: List[RawSample] = []

    train_noise: List[Path] = []
    test_noise: List[Path] = []

    cluster_images_by_split: Dict[str, DefaultDict[str, Set[Path]]] = {
        "train": defaultdict(set),
        "val": defaultdict(set),
        "test": defaultdict(set),
    }
    cluster_sketches_by_split: Dict[str, DefaultDict[str, Set[Path]]] = {
        "train": defaultdict(set),
        "val": defaultdict(set),
        "test": defaultdict(set),
    }

    def register_samples(samples_to_register: Sequence[RawSample]) -> None:
        for sample in samples_to_register:
            key = make_cluster_key(sample.category, sample.virtual_class)
            if not key:
                continue
            cluster_images_by_split[sample.split][key].add(sample.image_path)
            cluster_sketches_by_split[sample.split][key].update(sample.sketch_paths)

    for category, cfg in tqdm(datasets_cfg.items(), desc="categories"):
        train_cfg = cfg.get("train", {})
        test_cfg = cfg.get("test", {})

        train_text_pairs = read_text_pairs(Path(train_cfg["text"]).resolve())
        train_sketch_root = Path(train_cfg["sketch_dir"]).resolve()
        train_image_root = Path(train_cfg["image_dir"]).resolve()
        train_noise.extend(collect_noise(train_cfg.get("noise_dir")))

        virtual_dir = resolve_virtual_class_dir(category, train_cfg)
        test_virtual_dir = resolve_virtual_class_dir(category, test_cfg)
        virtual_lookup = build_virtual_class_lookup(virtual_dir)
        if test_virtual_dir and test_virtual_dir != virtual_dir:
            virtual_lookup.update(build_virtual_class_lookup(test_virtual_dir))

        category_train_samples: List[RawSample] = []
        for image_name, text in tqdm(train_text_pairs, desc=f"train:{category}", leave=False):
            image_path = (train_image_root / image_name).resolve()
            sketch_paths = load_sketches(train_sketch_root, image_name)
            if not sketch_paths:
                continue
            uid = f"{category}:{Path(image_name).stem}"
            virtual_class = virtual_lookup.get(Path(image_name).name)
            category_train_samples.append(
                RawSample(
                    uid=uid,
                    category=category,
                    split="train",
                    image_path=image_path,
                    sketch_paths=sketch_paths,
                    text=text,
                    source="train",
                    virtual_class=virtual_class,
                )
            )

        kept_train, kept_val = split_train_val(category_train_samples, val_ratio, "train")
        train_samples.extend(kept_train)
        register_samples(kept_train)
        val_samples.extend(kept_val)
        register_samples(kept_val)

        test_text_pairs = read_text_pairs(Path(test_cfg["text"]).resolve())
        test_sketch_root = Path(test_cfg["sketch_dir"]).resolve()
        test_image_root = Path(test_cfg["image_dir"]).resolve()
        test_noise.extend(collect_noise(test_cfg.get("noise_dir")))

        category_test_samples: List[RawSample] = []
        for image_name, text in tqdm(test_text_pairs, desc=f"test:{category}", leave=False):
            image_path = (test_image_root / image_name).resolve()
            sketch_paths = load_sketches(test_sketch_root, image_name)
            if not sketch_paths:
                continue
            uid = f"{category}:{Path(image_name).stem}"
            virtual_class = virtual_lookup.get(Path(image_name).name)
            category_test_samples.append(
                RawSample(
                    uid=uid,
                    category=category,
                    split="test",
                    image_path=image_path,
                    sketch_paths=sketch_paths,
                    text=text,
                    source="test",
                    virtual_class=virtual_class,
                )
            )

        test_samples.extend(category_test_samples)
        register_samples(category_test_samples)

    # Prepare negative pools per split.
    train_sketch_pool, train_image_pool, train_text_pool = build_negative_pools(train_samples, train_noise)
    val_sketch_pool, val_image_pool, val_text_pool = build_negative_pools(val_samples, train_noise)
    test_sketch_pool, test_image_pool, test_text_pool = build_negative_pools(test_samples, test_noise)

    manifests = data_cfg.get("manifests", {})

    cluster_images_by_split = {
        split: {key: tuple(sorted(paths, key=lambda p: p.as_posix())) for key, paths in split_map.items()}
        for split, split_map in cluster_images_by_split.items()
    }
    cluster_sketches_by_split = {
        split: {key: tuple(sorted(paths, key=lambda p: p.as_posix())) for key, paths in split_map.items()}
        for split, split_map in cluster_sketches_by_split.items()
    }

    if train_samples:
        train_records = assign_negatives(
            train_samples,
            cluster_images_by_split["train"],
            cluster_sketches_by_split["train"],
            train_sketch_pool,
            train_image_pool,
            train_text_pool,
            neg_cfg,
        )
        write_manifest(Path(manifests["train"]), train_records)
    if val_samples:
        val_records = assign_negatives(
            val_samples,
            cluster_images_by_split["val"],
            cluster_sketches_by_split["val"],
            val_sketch_pool,
            val_image_pool,
            val_text_pool,
            neg_cfg,
        )
        write_manifest(Path(manifests["val"]), val_records)
    else:
        # Ensure we still produce an empty file when no val split is requested.
        write_manifest(Path(manifests["val"]), [])
    if test_samples:
        test_records = assign_negatives(
            test_samples,
            cluster_images_by_split["test"],
            cluster_sketches_by_split["test"],
            test_sketch_pool,
            test_image_pool,
            test_text_pool,
            neg_cfg,
        )
        write_manifest(Path(manifests["test"]), test_records)


def write_manifest(path: Path, records: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} rows -> {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SBIR manifest files with positive/negative tuples.")
    parser.add_argument("--config", default="config.yaml", help="Path to project configuration YAML file.")
    parser.add_argument("--seed", type=int, default=110, help="Random seed used during sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    generate_manifests(config, args.seed)


if __name__ == "__main__":
    main()
