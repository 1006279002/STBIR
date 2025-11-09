"""Stage-wise training script for the SBIR multi-modal retrieval model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cstbir_model import LossConfig, MultiStageSBIRModel
from dataloader import SBIRDataset, sbir_collate
from src import clip
from utils import AverageMeter, load_config, move_to_device, save_checkpoint, set_seed

from scripts.generate_manifest import main as generate_manifest_main
from scripts.retrieve import main as retrieve_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-stage SBIR retrieval model")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration YAML file.")
    parser.add_argument("--device", default=None, help="Force device override (e.g., cpu, cuda:0)")
    parser.add_argument("--stage", default=None, help="Run only the specified stage (sketch|image|text)")
    parser.add_argument("--resume", default=None, help="Resume checkpoint path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    set_seed(config.get("seed", 110))

    device_name = args.device or config.get("training", {}).get("device", "cuda")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    clip_model, preprocess = clip.load(config["model"]["clip_model"], device=device, jit=False)

    dataset_cfg = config["data"]
    manifests = dataset_cfg["manifests"]
    curriculum_noise_cfg = dataset_cfg.get("curriculum_noise")

    train_dataset = SBIRDataset(
        Path(manifests["train"]),
        clip_preprocess=preprocess,
        negatives_per_modality=dataset_cfg.get("negatives_per_modality"),
        curriculum_noise=curriculum_noise_cfg,
        apply_noise=True,
    )
    val_dataset = SBIRDataset(
        Path(manifests["val"]),
        clip_preprocess=preprocess,
        negatives_per_modality=dataset_cfg.get("negatives_per_modality"),
        category_mapping=train_dataset.category_to_index,
        virtual_mapping=train_dataset.virtual_class_to_index,
        curriculum_noise=curriculum_noise_cfg,
        apply_noise=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"].get("num_workers", 4),
        pin_memory=True,
        collate_fn=sbir_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"].get("num_workers", 4),
        pin_memory=True,
        collate_fn=sbir_collate,
    )

    model_cfg = config["model"]
    num_classes = train_dataset.num_virtual_classes or train_dataset.num_categories
    model = MultiStageSBIRModel(
        clip_model=clip_model,
        feature_dim=model_cfg.get("feature_dim", 512),
        sketch_backbone=model_cfg.get("sketch_backbone", "resnet50"),
        sketch_pretrained=model_cfg.get("sketch_pretrained", True),
        fusion_strategy=model_cfg.get("fusion_strategy", "mean"),
        num_classes=num_classes,
    ).to(device)

    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])

    loss_cfg_raw = config["loss"]
    weight_cfg = loss_cfg_raw["weights"]
    loss_cfg = LossConfig(
        temperature=float(loss_cfg_raw["temperature"]),
        triplet_margin=float(loss_cfg_raw["triplet_margin"]),
        fusion_margin=float(loss_cfg_raw["fusion_margin"]),
        weight_info_nce=float(weight_cfg["info_nce"]),
        weight_triplet=float(weight_cfg["triplet"]),
        weight_fusion=float(weight_cfg["fusion"]),
        weight_cls=float(weight_cfg.get("cls", 0.0)),
    )

    stages = config["training"]["stages"]
    if args.stage:
        stages = [stage for stage in stages if stage["target"].lower() == args.stage.lower()]
        if not stages:
            raise ValueError(f"Stage {args.stage} not found in configuration")

    ckpt_dir = Path(config["training"]["checkpoint"]["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for stage in stages:
        target = stage["target"].lower()
        epochs = int(stage["epochs"])
        lr = float(stage.get("lr", config["training"].get("initial_lr", 1e-4)))
        weight_decay = float(stage.get("weight_decay", 1e-4))

        model.configure_stage(target)

        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=lr,
            weight_decay=weight_decay,
        )

        stage_best = float("inf")

        for epoch in range(epochs):
            train_dataset.update_noise(epoch=epoch, total_epochs=epochs, stage=target)
            noise_state = train_dataset.noise_parameters()
            if noise_state is not None:
                prob, strength = noise_state
                print(
                    f"Noise schedule | stage {target} epoch {epoch}: prob={prob:.3f} strength={strength:.3f}"
                )
            train_stats = run_epoch(model, train_loader, optimizer, device, loss_cfg, target, train=True)
            val_stats = run_epoch(model, val_loader, optimizer, device, loss_cfg, target, train=False)

            val_loss = val_stats["loss"]
            is_best = val_loss < stage_best
            if is_best:
                stage_best = val_loss
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "stage": target,
                "epoch": epoch,
                "best_val": stage_best,
            }
            
            if is_best:
                save_checkpoint(checkpoint, ckpt_dir / "best.pt")

            train_cls = train_stats.get("cls", 0.0)
            val_cls = val_stats.get("cls", 0.0)
            print(
                f"Stage {target}, epoch {epoch}: train {train_stats['loss']:.4f} "
                f"(cls {train_cls:.4f}) val {val_stats['loss']:.4f} (cls {val_cls:.4f})"
            )



def run_epoch(
    model: MultiStageSBIRModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_cfg: LossConfig,
    target: str,
    train: bool,
) -> Dict[str, float]:
    model.train() if train else model.eval()
    meter = {
        "loss": AverageMeter(),
        "info_nce": AverageMeter(),
        "triplet": AverageMeter(),
        "fusion": AverageMeter(),
        "cls": AverageMeter(),
    }

    for batch in tqdm(loader, desc="train" if train else "eval", leave=False):
        with torch.set_grad_enabled(train):
            batch = move_to_device(batch, device)
            outputs = model.forward_stage(batch, target, loss_cfg)
            loss = outputs["loss"]

            if not torch.isfinite(loss):
                print(
                    f"warning: encountered non-finite loss during {target} stage | "
                    f"train={train} skipping batch"
                )
                continue

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        batch_size = batch["images"].size(0)
        meter["loss"].update(loss.item(), batch_size)
        meter["info_nce"].update(outputs["info_nce"].item(), batch_size)
        meter["triplet"].update(outputs["triplet"].item(), batch_size)
        meter["fusion"].update(outputs["fusion"].item(), batch_size)
        cls_value = outputs.get("cls")
        if cls_value is None:
            cls_item = 0.0
        else:
            cls_item = cls_value.item()
        meter["cls"].update(cls_item, batch_size)

    return {key: value.avg for key, value in meter.items()}


if __name__ == "__main__":
    main()
    retrieve_main()

