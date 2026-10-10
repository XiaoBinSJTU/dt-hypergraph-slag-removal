"""Deterministic synthetic inputs for core-algorithm smoke testing only."""

from __future__ import annotations

import torch

from .device import release_device_cache, select_device
from .losses import CompositeLoss
from .models import DigitalTwinHypergraphModel


def _normalize(operator: torch.Tensor) -> torch.Tensor:
    degree = operator.sum(dim=1).clamp_min(1.0e-8)
    inverse_root = degree.rsqrt()
    return inverse_root[:, None] * operator * inverse_root[None, :]


def synthetic_relation_views(channels: int = 6) -> torch.Tensor:
    """Create three non-sensitive relation operators for interface validation."""
    identity = torch.eye(channels)
    chain = torch.eye(channels)
    for index in range(channels - 1):
        chain[index, index + 1] = 0.45
        chain[index + 1, index] = 0.45
    global_view = 0.15 * torch.ones(channels, channels) + 0.85 * torch.eye(channels)
    return torch.stack((identity, _normalize(chain), _normalize(global_view)))


def synthetic_knowledge() -> dict[str, torch.Tensor]:
    """Create deterministic templates that do not encode industrial records."""
    grid = torch.linspace(-1.0, 1.0, 48).reshape(4, 12)
    return {
        "state_templates": grid,
        "impact_templates": torch.cos(1.7 * grid),
        "action_templates": torch.sin(2.1 * grid),
        "rule_paths": torch.tensor(
            [
                [1.0, 0.8, 0.5, 0.2, 0.1, 0.0],
                [0.2, 0.8, 1.0, 0.7, 0.3, 0.1],
                [0.0, 0.2, 0.6, 1.0, 0.8, 0.4],
                [0.1, 0.1, 0.3, 0.6, 0.9, 1.0],
            ],
            dtype=torch.float32,
        ),
    }


def build_synthetic_case(
    device: torch.device,
) -> tuple[DigitalTwinHypergraphModel, dict[str, torch.Tensor]]:
    """Build a small deterministic cascade case without loading external data."""
    batch_size, steps, channels = 4, 16, 6
    time = torch.linspace(0.0, 1.0, steps, device=device).view(1, steps, 1)
    frequency = torch.arange(1, channels + 1, device=device).view(1, 1, channels)
    phase = torch.arange(batch_size, device=device).view(batch_size, 1, 1) * 0.17
    offsets = torch.linspace(-0.12, 0.12, channels, device=device).view(1, 1, channels)
    signals = (
        torch.sin(2.0 * torch.pi * time * frequency + phase)
        + 0.15 * torch.cos(torch.pi * time * (frequency + 1.0))
        + offsets
    )

    model = DigitalTwinHypergraphModel(
        channels=channels,
        classes=2,
        relations=synthetic_relation_views(channels),
        hidden_dim=16,
        method="proposed",
        local_task=True,
        module_map=torch.arange(channels),
        propagation_coefficient=0.85,
        variant="full",
        knowledge=synthetic_knowledge(),
    ).to(device)

    module_states = torch.sigmoid(signals.mean(dim=1))
    torque_scale = torch.tensor([0.31, 0.27, 0.20, 0.14], device=device)
    batch = {
        "signals": signals,
        "labels": torch.tensor([0, 1, 0, 1], device=device),
        "module_states": module_states,
        "torque_offsets": module_states[:, 2:6] * torque_scale,
    }
    return model, batch


def run_smoke_step(device_mode: str = "auto") -> dict[str, object]:
    """Run a real forward, composite loss, and backward pass."""
    torch.manual_seed(20250318)
    device, device_label = select_device(device_mode)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(20250318)

    model, batch = build_synthetic_case(device)
    model.train()
    output = model(batch["signals"])
    criterion = CompositeLoss(
        reconstruction_weight=0.02,
        torque_weight=1.0,
        propagation_weight=1.0,
    ).to(device)
    loss, _ = criterion(output, batch)
    if not torch.isfinite(loss):
        raise RuntimeError("The synthetic smoke-test loss is not finite.")
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("The synthetic smoke-test gradients are not finite.")

    result = {
        "status": "PASS",
        "device_mode": device_label,
        "forward_shape": tuple(output["logits"].shape),
        "loss_backward": "PASS",
    }
    del loss, output, batch, criterion, model, gradients
    release_device_cache(device)
    return result
