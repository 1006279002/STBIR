from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator

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


class CUDAPrefetcher(Iterator[Dict[str, Any]]):
    """Asynchronously prefetches batches to GPU using a dedicated CUDA stream."""

    def __init__(self, loader: Iterable[Dict[str, Any]], device: torch.device) -> None:
        if device.type != "cuda":
            raise ValueError("CUDAPrefetcher requires a CUDA device")
        self._loader_iter = iter(loader)
        self._device = device
        self._stream = torch.cuda.Stream(device=device)
        self._next_batch: Dict[str, Any] | None = None
        self._preload()

    def _preload(self) -> None:
        try:
            batch = next(self._loader_iter)
        except StopIteration:
            self._next_batch = None
            return
        with torch.cuda.stream(self._stream):
            self._next_batch = move_to_device(batch, self._device, non_blocking=True)

    def __iter__(self) -> "CUDAPrefetcher":
        return self

    def __next__(self) -> Dict[str, Any]:
        current_stream = torch.cuda.current_stream(self._device)
        current_stream.wait_stream(self._stream)
        if self._next_batch is None:
            raise StopIteration
        batch = self._next_batch
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                value.record_stream(current_stream)
        self._preload()
        return batch


def batch_iterator(
    loader: Iterable[Dict[str, Any]],
    device: torch.device,
    prefetch_to_gpu: bool = False,
) -> Iterable[Dict[str, Any]]:
    if prefetch_to_gpu and device.type == "cuda":
        prefetcher = CUDAPrefetcher(loader, device)

        def generator() -> Iterator[Dict[str, Any]]:
            while True:
                try:
                    yield next(prefetcher)
                except StopIteration:
                    break

        return generator()
    non_blocking = device.type == "cuda"
    for batch in loader:
        yield move_to_device(batch, device, non_blocking=non_blocking)


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
    "batch_iterator",
    "CUDAPrefetcher",
    "load_config",
    "move_to_device",
    "save_checkpoint",
    "save_json",
    "set_seed",
]
