"""Composite training loss for state, reconstruction, source, and torque tasks."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional


class CompositeLoss(nn.Module):
    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        reconstruction_weight: float = 0.0,
        torque_weight: float = 0.0,
        propagation_weight: float = 0.0,
        module_weight: float = 0.15,
    ) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.reconstruction_weight = float(reconstruction_weight)
        self.torque_weight = float(torque_weight)
        self.propagation_weight = float(propagation_weight)
        self.module_weight = float(module_weight)

    def forward(self, output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        classification = functional.cross_entropy(output["logits"], batch["labels"], weight=self.class_weights)
        total = classification
        values = {"classification": float(classification.detach().cpu())}
        if self.reconstruction_weight > 0.0:
            normal_mask = batch["labels"] == 0
            prediction = output["reconstruction"][normal_mask] if normal_mask.any() else output["reconstruction"]
            target = output["reconstruction_target"][normal_mask] if normal_mask.any() else output["reconstruction_target"]
            reconstruction = functional.mse_loss(prediction, target)
            total = total + self.reconstruction_weight * reconstruction
            values["reconstruction"] = float(reconstruction.detach().cpu())
        if self.torque_weight > 0.0 and "torque_offsets" in batch and "torque_offsets" in output:
            torque = functional.smooth_l1_loss(output["torque_offsets"], batch["torque_offsets"])
            total = total + self.torque_weight * torque
            values["torque"] = float(torque.detach().cpu())
        if self.propagation_weight > 0.0 and "module_states" in batch and "module_states" in output:
            propagation = functional.smooth_l1_loss(output["module_states"], batch["module_states"])
            total = total + self.propagation_weight * propagation
            values["propagation"] = float(propagation.detach().cpu())
        if "module_labels" in batch and "module_logits" in output:
            valid = batch["module_labels"] >= 0
            if valid.any():
                module = functional.cross_entropy(output["module_logits"][valid], batch["module_labels"][valid])
                total = total + self.module_weight * module
                values["module"] = float(module.detach().cpu())
        values["total"] = float(total.detach().cpu())
        return total, values
