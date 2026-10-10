"""GPU-first device selection and safe cache cleanup."""

from __future__ import annotations

import gc

import torch


def select_device(mode: str = "auto") -> tuple[torch.device, str]:
    normalized = mode.lower()
    if normalized not in {"auto", "cuda", "cpu"}:
        raise ValueError("Device mode must be auto, cuda, or cpu.")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU execution was requested but is unavailable.")
    if normalized in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda"), "GPU"
    return torch.device("cpu"), "CPU fallback"


def release_device_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
