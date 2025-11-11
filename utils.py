from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_to_device(batch: Dict[str, Any], device: torch.device, non_blocking: bool = False) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=non_blocking)
        elif isinstance(value, list):
            moved[key] = [item.to(device, non_blocking=non_blocking) if isinstance(item, torch.Tensor) else item for item in value]
        elif isinstance(value, dict):
            moved[key] = move_to_device(value, device, non_blocking=non_blocking)
        else:
            moved[key] = value
    return moved


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.total = 0.0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def save_checkpoint(state: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


__all__ = [
    "AverageMeter",
    "load_config",
    "move_to_device",
    "save_checkpoint",
    "save_json",
    "set_seed",
]
