"""Differentiable temporal features for sensor-node representations."""

from __future__ import annotations

import torch


def temporal_node_features(signals: torch.Tensor) -> torch.Tensor:
    """Return mean, deviation, slope, spectral peak, shock, and energy per channel."""
    if signals.ndim != 3:
        raise ValueError("Signals must have shape [batch, time, channels].")
    _, steps, _ = signals.shape
    mean = signals.mean(dim=1)
    deviation = signals.std(dim=1, unbiased=False)
    centered_time = torch.linspace(-1.0, 1.0, steps, device=signals.device, dtype=signals.dtype)
    slope = (signals * centered_time.view(1, -1, 1)).mean(dim=1)
    spectrum = torch.fft.rfft(signals, dim=1).abs()
    spectral_peak = spectrum[:, 1:, :].amax(dim=1) if spectrum.shape[1] > 1 else spectrum[:, 0, :]
    shock = torch.diff(signals, dim=1).abs().amax(dim=1)
    energy = torch.sqrt(torch.clamp(signals.square().mean(dim=1), min=1.0e-8))
    return torch.stack((mean, deviation, slope, spectral_peak, shock, energy), dim=-1)


def normalize_contributions(errors: torch.Tensor) -> torch.Tensor:
    denominator = errors.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    return errors / denominator
