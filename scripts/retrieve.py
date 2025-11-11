"""Image retrieval evaluation using the trained multi-stage SBIR model."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime
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


def summarize_metrics(ranks: Sequence[int], ap_all: Sequence[float]) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "num_queries": len(ranks),
        "median_rank": float("nan"),
        "map": {"mAP@all": float("nan")},
        "accuracy": {"acc@1": float("nan"), "acc@5": float("nan"), "acc@10": float("nan")},
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
    acc10 = 100.0 * sum(1 for r in ranks if r <= 10) / total
    summary["accuracy"]["acc@1"] = float(round(acc1, 2))
    summary["accuracy"]["acc@5"] = float(round(acc5, 2))
    summary["accuracy"]["acc@10"] = float(round(acc10, 2))

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

        top_indices = sorted_idx[:10].tolist()
        top_candidates = [unique_images[idx].name for idx in top_indices]

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
                "top10_images": top_candidates,
            }
        )

    overall_metrics = summarize_metrics(ranks, ap_all)
    per_category_metrics: Dict[str, Dict[str, object]] = {}
    for category, bucket in category_buckets.items():
        per_category_metrics[category] = summarize_metrics(bucket["ranks"], bucket["ap_all"])

    median_rank = overall_metrics["median_rank"]
    mean_ap_all_pct = overall_metrics["map"]["mAP@all"]
    acc1 = overall_metrics["accuracy"]["acc@1"]
    acc5 = overall_metrics["accuracy"]["acc@5"]
    acc10 = overall_metrics["accuracy"].get("acc@10", float("nan"))

    output = {
        "median_rank": median_rank,
        "num_queries": overall_metrics["num_queries"],
        "map": overall_metrics["map"],
        "accuracy": overall_metrics["accuracy"],
        "per_category": per_category_metrics,
        "details": detailed,
    }

    run_record = dict(output)
    run_record["timestamp"] = datetime.now().isoformat(timespec="seconds")
    run_record["config"] = str(args.config)
    run_record["checkpoint"] = str(args.checkpoint)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    past_runs: List[Dict[str, object]] = []
    if output_path.exists():
        try:
            with output_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
        except json.JSONDecodeError:
            existing = None
        if isinstance(existing, list):
            past_runs = existing
        elif isinstance(existing, dict):
            if "history" in existing and isinstance(existing["history"], list):
                past_runs = list(existing["history"])
            else:
                past_runs = [existing]

    past_runs.append(run_record)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(past_runs, handle, ensure_ascii=False, indent=2)

    if per_category_metrics:
        print("Per-category metrics:")
        header = ["Category", "Count", "Median Rank", "mAP@all", "acc@1", "acc@5", "acc@10"]
        rows: List[List[str]] = []
        for category in sorted(per_category_metrics):
            metrics = per_category_metrics[category]
            cat_median = metrics["median_rank"]
            cat_map = metrics["map"].get("mAP@all", float("nan"))
            cat_acc1 = metrics["accuracy"].get("acc@1", float("nan"))
            cat_acc5 = metrics["accuracy"].get("acc@5", float("nan"))
            cat_acc10 = metrics["accuracy"].get("acc@10", float("nan"))

            rows.append(
                [
                    category,
                    str(metrics["num_queries"]),
                    "nan" if math.isnan(cat_median) else f"{cat_median:.2f}",
                    "nan" if math.isnan(cat_map) else f"{cat_map:.2f}%",
                    "nan" if math.isnan(cat_acc1) else f"{cat_acc1:.2f}%",
                    "nan" if math.isnan(cat_acc5) else f"{cat_acc5:.2f}%",
                    "nan" if math.isnan(cat_acc10) else f"{cat_acc10:.2f}%",
                ]
            )

        column_widths = [len(col) for col in header]
        if rows:
            column_widths = [max(len(header[i]), max(len(row[i]) for row in rows)) for i in range(len(header))]
        header_line = " | ".join(header[i].ljust(column_widths[i]) for i in range(len(header)))
        separator = "-+-".join("-" * column_widths[i] for i in range(len(header)))
        print(header_line)
        print(separator)
        for row in rows:
            print(" | ".join(row[i].ljust(column_widths[i]) for i in range(len(header))))

    table_header = ["Metric", "Value"]
    table_rows = [
        ["Median Rank", "nan" if math.isnan(median_rank) else f"{median_rank:.2f}"],
        ["mAP@all", "nan" if math.isnan(mean_ap_all_pct) else f"{mean_ap_all_pct:.2f}%"],
        ["acc@1", "nan" if math.isnan(acc1) else f"{acc1:.2f}%"],
        ["acc@5", "nan" if math.isnan(acc5) else f"{acc5:.2f}%"],
        ["acc@10", "nan" if math.isnan(acc10) else f"{acc10:.2f}%"],
        ["#Queries", str(overall_metrics["num_queries"])]
    ]
    col_widths = [max(len(table_header[i]), max(len(row[i]) for row in table_rows)) for i in range(2)]
    print("\nOverall metrics:")
    print(" | ".join(table_header[i].ljust(col_widths[i]) for i in range(2)))
    print("-+-".join("-" * col_widths[i] for i in range(2)))
    for row in table_rows:
        print(" | ".join(row[i].ljust(col_widths[i]) for i in range(2)))

    print(f"Saved sample retrieval details to {output_path}")


if __name__ == "__main__":
    main()
