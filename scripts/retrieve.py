"""Image retrieval evaluation using the trained multi-stage SBIR model."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cstbir_model import MultiStageSBIRModel
from dataloader import default_sketch_transform, load_manifest
from src import clip
from utils import load_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval performance on the test manifest")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt", help="Path to model checkpoint")
    parser.add_argument("--device", default=None, help="Device override")
    parser.add_argument("--output", default="outputs/retrieval/results.json", help="Where to store retrieval stats")
    return parser.parse_args()


def load_image(path: Path, preprocess) -> torch.Tensor:
    with Image.open(path) as img:
        return preprocess(img.convert("RGB"))


def load_sketch(path: Path, transform) -> torch.Tensor:
    with Image.open(path) as img:
        return transform(img.convert("RGB"))


def compute_image_features(paths: Sequence[Path], preprocess, model: MultiStageSBIRModel, device: torch.device) -> Dict[Path, torch.Tensor]:
    features: Dict[Path, torch.Tensor] = {}
    batch: List[torch.Tensor] = []
    refs: List[Path] = []
    with torch.no_grad():
        for path in tqdm(paths, desc="images", leave=False):
            batch.append(load_image(path, preprocess))
            refs.append(path)
            if len(batch) == 32:
                feats = model.encode_images(torch.stack(batch).to(device))
                for ref, feat in zip(refs, feats):
                    features[ref] = feat.cpu()
                batch.clear()
                refs.clear()
        if batch:
            feats = model.encode_images(torch.stack(batch).to(device))
            for ref, feat in zip(refs, feats):
                features[ref] = feat.cpu()
    return features


def compute_text_feature(text: str, model: MultiStageSBIRModel, device: torch.device) -> torch.Tensor:
    tokens = clip.tokenize([text], truncate=True)
    with torch.no_grad():
        return model.encode_texts(tokens.to(device)).squeeze(0).cpu()


def compute_sketch_feature(paths: Sequence[Path], transform, model: MultiStageSBIRModel, device: torch.device) -> torch.Tensor:
    if not paths:
        raise ValueError("Query has no positive sketch")
    tensors = torch.stack([load_sketch(p, transform) for p in paths])
    with torch.no_grad():
        feats = model.encode_sketches(tensors.to(device))
    return feats.mean(dim=0).cpu()


def summarize_metrics(ranks: Sequence[int], ap_all: Sequence[float], recall_ks: Sequence[int]) -> Dict[str, object]:
    recall_keys = [f"R@{int(k)}" for k in recall_ks]
    summary: Dict[str, object] = {
        "num_queries": len(ranks),
        "median_rank": float("nan"),
        "map": {"mAP@all": float("nan")},
        "accuracy": {"acc@1": float("nan"), "acc@5": float("nan")},
        "recall_at": {key: float("nan") for key in recall_keys},
    }

    if not ranks:
        return summary

    ranks_tensor = torch.tensor(ranks, dtype=torch.float32)
    summary["median_rank"] = float(torch.median(ranks_tensor).item())

    mean_ap_all = torch.tensor(ap_all, dtype=torch.float32).mean().item() if ap_all else float("nan")
    summary["map"]["mAP@all"] = float("nan") if math.isnan(mean_ap_all) else float(round(mean_ap_all * 100.0, 2))

    total = float(len(ranks))
    acc1 = 100.0 * sum(1 for r in ranks if r == 1) / total
    acc5 = 100.0 * sum(1 for r in ranks if r <= 5) / total
    summary["accuracy"]["acc@1"] = float(round(acc1, 2))
    summary["accuracy"]["acc@5"] = float(round(acc5, 2))

    recall_at: Dict[str, float] = {}
    for key, k in zip(recall_keys, recall_ks):
        correct = sum(1 for r in ranks if r <= k)
        recall_at[key] = float(round(100.0 * correct / total, 2))
    summary["recall_at"] = recall_at

    return summary


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    set_seed(config.get("seed", 110))

    device_name = args.device or config.get("training", {}).get("device", "cuda")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    clip_model, preprocess = clip.load(config["model"]["clip_model"], device=device, jit=False)
    model_cfg = config["model"]

    checkpoint_path = Path(args.checkpoint)
    state = torch.load(checkpoint_path, map_location=device)
    saved_model_state = state.get("model", {})
    saved_classifier = saved_model_state.get("cls_head.weight")
    num_classes = saved_classifier.size(0) if saved_classifier is not None else 0

    model = MultiStageSBIRModel(
        clip_model=clip_model,
        feature_dim=model_cfg.get("feature_dim", 512),
        sketch_backbone=model_cfg.get("sketch_backbone", "resnet50"),
        sketch_pretrained=model_cfg.get("sketch_pretrained", True),
        fusion_strategy=model_cfg.get("fusion_strategy", "mean"),
        num_classes=num_classes,
    ).to(device)

    incompatible = model.load_state_dict(saved_model_state, strict=False)
    if incompatible.missing_keys:
        print(f"warning: missing keys when loading checkpoint: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        print(f"warning: unexpected keys ignored from checkpoint: {incompatible.unexpected_keys}")
    model.eval()

    manifest_path = Path(config["data"]["manifests"]["test"])
    rows = load_manifest(manifest_path)

    unique_images = sorted({Path(row["image"]).resolve() for row in rows})
    image_features = compute_image_features(unique_images, preprocess, model, device)

    image_matrix = torch.stack([image_features[path] for path in unique_images])
    image_index = {path: idx for idx, path in enumerate(unique_images)}

    sketch_transform = default_sketch_transform(config["data"].get("image_size", 224))

    ranks: List[int] = []
    ap_all: List[float] = []
    detailed: List[Dict[str, object]] = []
    category_buckets: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"ranks": [], "ap_all": []})
    # recall@K settings: read from config if provided, else default to [10,20,50,100]
    recall_ks = config.get("evaluation", {}).get("recall_at", [10, 20, 50, 100])
    # ensure integers and sorted
    recall_ks = sorted(int(k) for k in recall_ks)

    for row in tqdm(rows, desc="queries"):
        image_path = Path(row["image"]).resolve()
        text = row["text"]
        sketch_paths = [Path(p).resolve() for p in row.get("positive_sketches", [])]

        text_feat = compute_text_feature(text, model, device)
        sketch_feat = compute_sketch_feature(sketch_paths, sketch_transform, model, device)

        query = F.normalize(text_feat + sketch_feat, dim=0)
        sims = torch.matmul(query.unsqueeze(0), image_matrix.t()).squeeze(0)
        sorted_idx = torch.argsort(sims, descending=True)
        target_idx = image_index[image_path]
        rank = (sorted_idx == target_idx).nonzero(as_tuple=False).item() + 1

        ranks.append(rank)
        ap_all.append(1.0 / rank)
        category = row.get("category", "unknown")
        category_buckets[category]["ranks"].append(rank)
        category_buckets[category]["ap_all"].append(1.0 / rank)
        detailed.append(
            {
                "id": row["id"],
                "image": str(image_path),
                "text": text,
                "rank": int(rank),
            }
        )

    overall_metrics = summarize_metrics(ranks, ap_all, recall_ks)
    per_category_metrics: Dict[str, Dict[str, object]] = {}
    for category, bucket in category_buckets.items():
        per_category_metrics[category] = summarize_metrics(bucket["ranks"], bucket["ap_all"], recall_ks)

    median_rank = overall_metrics["median_rank"]
    mean_ap_all_pct = overall_metrics["map"]["mAP@all"]
    acc1 = overall_metrics["accuracy"]["acc@1"]
    acc5 = overall_metrics["accuracy"]["acc@5"]
    recall_at = overall_metrics["recall_at"]

    output = {
        "median_rank": median_rank,
        "num_queries": overall_metrics["num_queries"],
        "recall_at": recall_at,
        "map": overall_metrics["map"],
        "accuracy": overall_metrics["accuracy"],
        "per_category": per_category_metrics,
        "details": detailed[: min(50, len(detailed))],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)

    print(f"Median rank: {median_rank:.2f} over {overall_metrics['num_queries']} queries")
    if math.isnan(mean_ap_all_pct):
        print("mAP@all: nan")
    else:
        print(f"mAP@all: {mean_ap_all_pct:.2f}%")
    if math.isnan(acc1):
        print("acc@1: nan")
    else:
        print(f"acc@1: {acc1:.2f}%")
    if math.isnan(acc5):
        print("acc@5: nan")
    else:
        print(f"acc@5: {acc5:.2f}%")
    # print recall@K
    for k in recall_ks:
        key = f"R@{k}"
        val = recall_at.get(key)
        if val is None or (isinstance(val, float) and (val != val)):
            print(f"{key}: nan")
        else:
            # val is stored as percentage already
            print(f"{key}: {val:.2f}%")

    if per_category_metrics:
        print("Per-category metrics:")
        for category in sorted(per_category_metrics):
            metrics = per_category_metrics[category]
            cat_median = metrics["median_rank"]
            cat_map = metrics["map"].get("mAP@all", float("nan"))
            cat_acc1 = metrics["accuracy"].get("acc@1", float("nan"))
            cat_acc5 = metrics["accuracy"].get("acc@5", float("nan"))
            cat_recall = metrics["recall_at"]

            cat_median_str = "nan" if math.isnan(cat_median) else f"{cat_median:.2f}"
            cat_map_str = "nan" if math.isnan(cat_map) else f"{cat_map:.2f}%"
            cat_acc1_str = "nan" if math.isnan(cat_acc1) else f"{cat_acc1:.2f}%"
            cat_acc5_str = "nan" if math.isnan(cat_acc5) else f"{cat_acc5:.2f}%"

            recall_parts: List[str] = []
            for k in recall_ks:
                key = f"R@{k}"
                val = cat_recall.get(key, float("nan"))
                if isinstance(val, float) and math.isnan(val):
                    recall_parts.append(f"{key}=nan")
                else:
                    recall_parts.append(f"{key}={val:.2f}%")
            recall_str = ", ".join(recall_parts)

            print(
                f"  {category} (n={metrics['num_queries']}): median_rank={cat_median_str}, "
                f"mAP@all={cat_map_str}, acc@1={cat_acc1_str}, acc@5={cat_acc5_str}, {recall_str}"
            )

    print(f"Saved sample retrieval details to {output_path}")


if __name__ == "__main__":
    main()
