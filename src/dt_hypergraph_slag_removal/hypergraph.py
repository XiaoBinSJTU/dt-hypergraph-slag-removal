"""Data-driven relation views and physical-semantic propagation operators."""

from __future__ import annotations

import numpy as np
import torch


def _absolute_correlation(values: np.ndarray) -> np.ndarray:
    correlation = np.corrcoef(values, rowvar=False)
    correlation = np.nan_to_num(np.abs(correlation), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(correlation, 1.0)
    return correlation.astype(np.float32)


def _retain_edges(scores: np.ndarray, threshold: float, neighbors: int = 2) -> np.ndarray:
    channels = scores.shape[0]
    retained = np.where(scores >= threshold, scores, 0.0).astype(np.float32)
    for row in range(channels):
        indices = np.argsort(scores[row])[-(neighbors + 1) :]
        retained[row, indices] = scores[row, indices]
    retained = np.maximum(retained, retained.T)
    np.fill_diagonal(retained, 1.0)
    return retained


def normalize_adjacency(adjacency: np.ndarray) -> np.ndarray:
    degree = np.maximum(adjacency.sum(axis=1), 1.0e-8)
    inverse_root = 1.0 / np.sqrt(degree)
    return (inverse_root[:, None] * adjacency * inverse_root[None, :]).astype(np.float32)


def build_relation_views(signals: np.ndarray, thresholds: list[float] | tuple[float, ...]) -> torch.Tensor:
    """Construct similarity, nonlinear, and lag-direction relation operators."""
    if signals.ndim != 3:
        raise ValueError("Signals must have shape [samples, time, channels].")
    samples, steps, channels = signals.shape
    flat = signals.reshape(samples * steps, channels)
    similarity_scores = _absolute_correlation(flat)
    nonlinear_scores = _absolute_correlation(np.square(flat))
    current = signals[:, :-1, :].reshape(samples * (steps - 1), channels)
    future = signals[:, 1:, :].reshape(samples * (steps - 1), channels)
    directional = np.zeros((channels, channels), dtype=np.float32)
    for source in range(channels):
        source_values = current[:, source]
        source_scale = max(float(source_values.std()), 1.0e-8)
        for target in range(channels):
            target_values = future[:, target]
            target_scale = max(float(target_values.std()), 1.0e-8)
            centered_product = (source_values - source_values.mean()) * (target_values - target_values.mean())
            directional[source, target] = abs(float(centered_product.mean()) / (source_scale * target_scale))
    directional_scores = np.maximum(directional, directional.T)
    tau_similarity, tau_nonlinear, tau_directional = [float(value) for value in thresholds]
    views = (
        _retain_edges(similarity_scores, max(0.0, 1.0 - tau_similarity)),
        _retain_edges(nonlinear_scores, tau_nonlinear),
        _retain_edges(directional_scores, tau_directional),
    )
    return torch.from_numpy(np.stack([normalize_adjacency(view) for view in views], axis=0))


def physical_relation_matrices() -> torch.Tensor:
    """Return mechanical, transmission, and thermal module relation operators."""
    mechanical = np.eye(6, dtype=np.float32)
    transmission = np.eye(6, dtype=np.float32)
    thermal = np.eye(6, dtype=np.float32)
    mechanical_weights = (0.94, 0.88, 0.74, 0.56, 0.34)
    transmission_weights = (0.82, 0.76, 0.64, 0.48, 0.30)
    thermal_weights = (0.68, 0.60, 0.55, 0.47, 0.40)
    for index in range(5):
        mechanical[index, index + 1] = mechanical[index + 1, index] = mechanical_weights[index]
        transmission[index, index + 1] = transmission[index + 1, index] = transmission_weights[index]
        thermal[index, index + 1] = thermal[index + 1, index] = thermal_weights[index]
    mechanical[0, 2] = mechanical[2, 0] = 0.63
    transmission[2, 4] = transmission[4, 2] = 0.42
    thermal[0, 3] = thermal[3, 0] = 0.38
    matrices = [normalize_adjacency(item) for item in (mechanical, transmission, thermal)]
    return torch.from_numpy(np.stack(matrices, axis=0))


def propagate_states(
    initial: torch.Tensor,
    operators: torch.Tensor,
    gates: torch.Tensor,
    coefficient: float,
    steps: int = 8,
) -> torch.Tensor:
    """Iterate the reported convex propagation update on a batched operator."""
    operator = torch.einsum("bv,vij->bij", gates, operators)
    state = initial
    for _ in range(steps):
        state = coefficient * torch.bmm(operator, state.unsqueeze(-1)).squeeze(-1) + (1.0 - coefficient) * initial
    return state.clamp(0.0, 1.0)
