import math
import os
import queue
import random
import threading
from contextlib import contextmanager
from typing import Dict, Iterator

import torch
from torch import nn, optim
from tqdm import tqdm

from cstbir_model import CSTBIRModel
from dataloader import CSTBIR_dataset
from src import clip
from utils import convert_models_to_fp32, load_config


@contextmanager
def prefetch_batches(dataset: CSTBIR_dataset, num_batches: int, max_prefetch: int):
    if max_prefetch <= 1:
        def _direct_iterator() -> Iterator[Dict[str, torch.Tensor]]:
            for _ in range(num_batches):
                yield dataset.get_samples()

        yield _direct_iterator()
        return

    sentinel = object()
    batch_queue: "queue.Queue[Dict[str, torch.Tensor]]" = queue.Queue(max_prefetch)
    stop_event = threading.Event()

    def producer():
        try:
            for _ in range(num_batches):
                if stop_event.is_set():
                    break
                batch = dataset.get_samples()
                batch_queue.put(batch)
        finally:
            batch_queue.put(sentinel)

    worker = threading.Thread(target=producer, daemon=True)
    worker.start()
    try:
        def _iterator() -> Iterator[Dict[str, torch.Tensor]]:
            while True:
                item = batch_queue.get()
                if item is sentinel:
                    break
                yield item

        yield _iterator()
    finally:
        stop_event.set()
        worker.join()

random.seed(110)

def compute_contrastive_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor, ground_truth: torch.Tensor, loss_img, loss_txt) -> torch.Tensor:
    return (loss_img(logits_per_image.float(), ground_truth) + loss_txt(logits_per_text.float(), ground_truth)) / 2.0


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def main():
    config = load_config("config.yaml")

    save_dir = config["training"]["save_model_path"]
    os.makedirs(save_dir, exist_ok=True)

    use_gpu = config["training"].get("gpu", True) and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    base_model, preprocess = clip.load(config["model"]["model_name"], device=device, jit=False)

    print("loading dataset...")
    train_data = CSTBIR_dataset(
        config["data"]["dataset_path"],
        config["data"]["images_path"],
        config["data"]["sketch_path"],
        config["model"]["batch_size"],
        config["data"]["train_split_name"],
        preprocess,
    )
    val_data = CSTBIR_dataset(
        config["data"]["dataset_path"],
        config["data"]["images_path"],
        config["data"]["sketch_path"],
        config["model"]["batch_size"],
        config["data"]["val_split_name"],
        preprocess,
    )

    n_train_samples = len(train_data) * config["model"]["batch_size"]
    n_val_samples = len(val_data) * config["model"]["batch_size"]
    print("loaded dataset")
    print(f"# train samples: {n_train_samples}")
    print(f"# val samples: {n_val_samples}")

    num_classes = len(CSTBIR_dataset._idx_to_label)

    detector_cfg = config.get("detector", {})
    model = CSTBIRModel(
        base_model,
        num_classes=num_classes,
        detector_grid=detector_cfg.get("grid_size", 7),
        detector_boxes=detector_cfg.get("boxes_per_cell", 2),
    ).to(device)

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    cls_loss = nn.CrossEntropyLoss()

    loss_weights = config.get("loss", {})
    w_contrastive = float(loss_weights.get("contrastive_weight", 1.0))
    w_text_cls = float(loss_weights.get("text_cls_weight", 1.0))
    w_image_cls = float(loss_weights.get("image_cls_weight", 1.0))
    w_detection = float(loss_weights.get("detection_weight", 1.0))
    w_recon = float(loss_weights.get("reconstruction_weight", 1.0))
    sr_alpha = float(loss_weights.get("sr_bce_weight", 0.5))
    sr_beta = float(loss_weights.get("sr_dice_weight", 0.5))

    lr_raw = config.get("training", {}).get("learning_rate", config.get("model", {}).get("learning_rate", 5e-5))
    try:
        lr = float(lr_raw)
    except Exception:
        lr = 5e-5
    weight_decay = float(config["model"].get("weight_decay", 0.0))

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    epochs = int(config.get("training", {}).get("epochs", 1))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_data) * epochs)

    softmax = nn.Softmax(dim=1)

    best_te_loss = 1e5
    best_te_acc = 0.0
    best_epoch = -1

    def evaluate(split_name: str, dataset: CSTBIR_dataset):
        model.eval()
        total_loss = 0.0
        total_acc = 0.0
        steps = 0
        prefetch_size = int(config["data"].get("prefetch_batches", 4))
        total_steps = len(dataset)
        with torch.no_grad():
            with prefetch_batches(dataset, total_steps, prefetch_size) as iterator:
                for batch in tqdm(iterator, leave=False, desc=f"{split_name} eval", total=total_steps, miniters=1):
                    batch = move_batch_to_device(batch, device)
                    outputs = model(batch["images"], batch["texts"], batch["sketches"])
                    ground_truth = torch.arange(batch["images"].size(0), device=device)

                    contrastive = compute_contrastive_loss(
                        outputs["logits_per_image"], outputs["logits_per_text"], ground_truth, loss_img, loss_txt
                    )
                    text_cls = cls_loss(outputs["text_cls_logits"].float(), batch["labels"])
                    image_cls = cls_loss(outputs["image_cls_logits"].float(), batch["labels"])
                    detection = model.detection_loss(outputs["detection_map"])
                    reconstruction = model.reconstruction_loss(outputs["reconstruction_logits"], batch["sketches"], sr_alpha, sr_beta)

                    total = (
                        w_contrastive * contrastive
                        + w_text_cls * text_cls
                        + w_image_cls * image_cls
                        + w_detection * detection
                        + w_recon * reconstruction
                    )

                    total_loss += total.item()
                    preds = torch.argmax(softmax(outputs["logits_per_text"].float()), dim=1)
                    total_acc += torch.sum(preds == ground_truth).item()
                    steps += 1

        avg_loss = total_loss / max(1, steps)
        avg_acc = total_acc / (steps * dataset.batch_size)
        return avg_loss, avg_acc

    te_loss, te_acc = evaluate("val", val_data)
    print(f"initial val_loss {te_loss:.4f}, val_acc {te_acc * 100:.2f}")

    for epoch in range(epochs):
        print(
            f"running epoch {epoch}, best val loss {best_te_loss:.4f} best val acc {best_te_acc * 100:.2f} after epoch {best_epoch}"
        )
        model.train()
        step = 0
        tr_loss = 0.0
        tr_acc = 0.0

        prefetch_size = int(config["data"].get("prefetch_batches", 4))
        total_steps = len(train_data)
        with prefetch_batches(train_data, total_steps, prefetch_size) as iterator:
            for batch in tqdm(iterator, desc="train", total=total_steps, miniters=1):
                batch = move_batch_to_device(batch, device)
                optimizer.zero_grad()

                if hasattr(model.clip, "logit_scale"):
                    with torch.no_grad():
                        upper = math.log(100.0)
                        model.clip.logit_scale.data.clamp_(max=upper)

                outputs = model(batch["images"], batch["texts"], batch["sketches"])
                ground_truth = torch.arange(batch["images"].size(0), device=device)

                contrastive = compute_contrastive_loss(
                    outputs["logits_per_image"], outputs["logits_per_text"], ground_truth, loss_img, loss_txt
                )
                text_cls = cls_loss(outputs["text_cls_logits"].float(), batch["labels"])
                image_cls = cls_loss(outputs["image_cls_logits"].float(), batch["labels"])
                detection = model.detection_loss(outputs["detection_map"])
                reconstruction = model.reconstruction_loss(outputs["reconstruction_logits"], batch["sketches"], sr_alpha, sr_beta)

                total_loss = (
                    w_contrastive * contrastive
                    + w_text_cls * text_cls
                    + w_image_cls * image_cls
                    + w_detection * detection
                    + w_recon * reconstruction
                )

                preds = torch.argmax(softmax(outputs["logits_per_text"].float()), dim=1)
                tr_acc += torch.sum(preds == ground_truth).item()

                loss_components = {
                    "contrastive": contrastive,
                    "text_cls": text_cls,
                    "image_cls": image_cls,
                    "detection": detection,
                    "reconstruction": reconstruction,
                    "total": total_loss,
                }
                non_finite_terms = [name for name, value in loss_components.items() if not torch.isfinite(value)]
                if non_finite_terms:
                    print(
                        "warning: encounter non-finite values -> "
                        + ", ".join(
                            f"{name}={loss_components[name].detach().float().mean().item():.4e}" for name in non_finite_terms
                        ),
                        flush=True,
                    )
                    if hasattr(model.clip, "logit_scale"):
                        with torch.no_grad():
                            upper = math.log(100.0)
                            model.clip.logit_scale.data.clamp_(max=upper)
                    continue

                total_loss.backward()
                tr_loss += total_loss.item()

                if device.type == "cpu":
                    optimizer.step()
                    with torch.no_grad():
                        if hasattr(model.clip, "logit_scale"):
                            upper = math.log(100.0)
                            model.clip.logit_scale.data.clamp_(max=upper)
                    scheduler.step()
                else:
                    convert_models_to_fp32(model.clip)
                    optimizer.step()
                    with torch.no_grad():
                        if hasattr(model.clip, "logit_scale"):
                            upper = math.log(100.0)
                            model.clip.logit_scale.data.clamp_(max=upper)
                    scheduler.step()
                    clip.model.convert_weights(model.clip)

                step += 1

        tr_loss /= max(1, step)
        tr_acc /= n_train_samples

        te_loss, te_acc = evaluate("val", val_data)

        if te_loss < best_te_loss:
            best_te_loss = te_loss
            best_epoch = epoch
        if te_acc > best_te_acc:
            best_te_acc = te_acc

        checkpoint_path = os.path.join(save_dir, f"model_checkpoint_{epoch}.pt")
        torch.save(model.state_dict(), checkpoint_path)

        print(
            f"epoch {epoch}, tr_loss {tr_loss:.4f}, val_loss {te_loss:.4f}, tr_acc {tr_acc * 100:.2f}, val_acc {te_acc * 100:.2f}"
        )

    torch.save(model.state_dict(), os.path.join(save_dir, "last_model.pt"))


if __name__ == "__main__":
    main()

