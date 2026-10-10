"""Multi-view part model and three-level digital-twin cascade model."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as functional

from .features import normalize_contributions, temporal_node_features
from .hypergraph import physical_relation_matrices, propagate_states


class DigitalTwinHypergraphModel(nn.Module):
    """Compact implementation of part, module, and knowledge cascade reasoning."""

    def __init__(
        self,
        channels: int,
        classes: int,
        relations: torch.Tensor,
        hidden_dim: int,
        method: str,
        local_task: bool = False,
        module_map: torch.Tensor | None = None,
        propagation_coefficient: float = 0.85,
        variant: str = "full",
        knowledge: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if relations.shape != (3, channels, channels):
            raise ValueError("Three channel-relation operators are required.")
        self.channels = channels
        self.classes = classes
        self.hidden_dim = hidden_dim
        self.method = method
        self.local_task = local_task
        self.variant = variant
        self.propagation_coefficient = float(propagation_coefficient)
        self.register_buffer("relations", relations.float())
        self.input_projection = nn.Linear(6, hidden_dim)
        self.view_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(3)])
        self.view_score = nn.Linear(hidden_dim, 1)
        self.proposed_fusion = nn.Linear(3 * hidden_dim, hidden_dim)
        self.view_classifiers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, classes)
                )
                for _ in range(3)
            ]
        )
        self.state_classifier = nn.Sequential(
            nn.Linear(12, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, classes)
        )
        self.ksd_residual_classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, classes)
        )
        self.reconstruction_decoder = nn.Linear(hidden_dim, 6)
        self.base_classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, classes),
        )
        if local_task:
            if module_map is None or module_map.numel() != channels:
                raise ValueError("A channel-to-module map is required for local tasks.")
            self.register_buffer("module_map", module_map.long())
            physical = physical_relation_matrices().float()
            uniform = (physical > 0).float()
            uniform = uniform / uniform.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
            self.register_buffer("physical_operators", physical)
            self.register_buffer("uniform_physical_operators", uniform)
            self.module_state_head = nn.Linear(hidden_dim, 1)
            nn.init.constant_(self.module_state_head.bias, -1.5)
            self.module_classifier = nn.Linear(hidden_dim, 1)
            torque_ratios = torch.tensor([0.31, 0.27, 0.20, 0.14], dtype=torch.float32) / 0.40
            self.torque_scale_logits = nn.Parameter(torch.logit(torque_ratios.clamp(0.01, 0.99)))
            self.baseline_torque_head = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 4), nn.Sigmoid()
            )
            self.knowledge_query = nn.Linear(2 * hidden_dim, hidden_dim)
            self.state_template_projection = nn.Linear(12, hidden_dim, bias=False)
            self.impact_template_projection = nn.Linear(12, hidden_dim, bias=False)
            self.action_template_projection = nn.Linear(12, hidden_dim, bias=False)
            knowledge = knowledge or {}
            self.register_buffer("state_templates", torch.as_tensor(knowledge.get("state_templates", torch.eye(4, 12))).float())
            self.register_buffer("impact_templates", torch.as_tensor(knowledge.get("impact_templates", torch.eye(4, 12))).float())
            self.register_buffer("action_templates", torch.as_tensor(knowledge.get("action_templates", torch.eye(4, 12))).float())
            self.register_buffer("rule_paths", torch.as_tensor(knowledge.get("rule_paths", torch.ones(1, 6))).float())
            cascade_width = 2 + 4 + 3 + 4
            self.cascade_classifier = nn.Sequential(
                nn.Linear(cascade_width, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(hidden_dim, classes),
            )

    def _encode_views(self, node_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        projected = torch.relu(self.input_projection(node_features))
        encoded_views = []
        for view in range(3):
            propagated = torch.einsum("ij,bjh->bih", self.relations[view], projected)
            encoded_views.append(torch.relu(self.view_layers[view](propagated)))
        stacked = torch.stack(encoded_views, dim=1)
        if self.method == "phsl_hgnn":
            weights = torch.tensor([1.0, 0.0, 0.0], device=stacked.device, dtype=stacked.dtype)
            fused = stacked[:, 0]
        elif self.method == "ksd_hgnn":
            weights = torch.tensor([0.5, 0.5, 0.0], device=stacked.device, dtype=stacked.dtype)
            fused = torch.tanh(0.5 * stacked[:, 0] + 0.5 * stacked[:, 1])
        elif self.method == "spatial_temporal_meta_hypergraph":
            union = torch.amax(self.relations, dim=0)
            fused = torch.relu(self.view_layers[0](torch.einsum("ij,bjh->bih", union, projected)))
            weights = torch.full((3,), 1.0 / 3.0, device=stacked.device, dtype=stacked.dtype)
        elif self.method == "phg_mtnet":
            weights = torch.full((3,), 1.0 / 3.0, device=stacked.device, dtype=stacked.dtype)
            fused = stacked.mean(dim=1)
        elif self.method == "proposed":
            scores = torch.stack([self.view_score(stacked[:, view].mean(dim=1)).squeeze(-1) for view in range(3)], dim=1)
            weights = torch.softmax(scores, dim=1)
            adaptive = torch.einsum("bv,bvch->bch", weights, stacked)
            concatenated = torch.cat([stacked[:, view] for view in range(3)], dim=-1)
            fused = torch.relu(self.proposed_fusion(concatenated)) + 0.5 * adaptive + 0.5 * stacked.mean(dim=1)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        if weights.ndim == 1:
            weights = weights.unsqueeze(0).expand(node_features.shape[0], -1)
        return fused, weights, stacked

    @staticmethod
    def _pool(node_embeddings: torch.Tensor, method: str) -> torch.Tensor:
        mean = node_embeddings.mean(dim=1)
        if method == "phsl_hgnn":
            second = mean
        elif method == "ksd_hgnn":
            second = node_embeddings.std(dim=1, unbiased=False)
        else:
            second = node_embeddings.amax(dim=1)
        return torch.cat((mean, second), dim=1)

    def _aggregate_modules(self, values: torch.Tensor) -> torch.Tensor:
        output_shape = (values.shape[0], 6) + tuple(values.shape[2:])
        aggregated = torch.zeros(output_shape, device=values.device, dtype=values.dtype)
        aggregated.index_add_(1, self.module_map, values)
        counts = torch.bincount(self.module_map, minlength=6).to(values.device, values.dtype).clamp_min(1.0)
        reshape = (1, 6) + (1,) * (values.ndim - 2)
        return aggregated / counts.view(reshape)

    def _template_score(self, query: torch.Tensor, templates: torch.Tensor, projection: nn.Linear) -> torch.Tensor:
        projected_templates = functional.normalize(projection(templates), dim=1)
        normalized_query = functional.normalize(query, dim=1)
        return torch.matmul(normalized_query, projected_templates.T).amax(dim=1, keepdim=True)

    def forward(self, signals: torch.Tensor) -> dict[str, torch.Tensor]:
        node_features = temporal_node_features(signals)
        node_embeddings, view_weights, encoded_views = self._encode_views(node_features)
        reconstruction = self.reconstruction_decoder(node_embeddings)
        node_errors = (node_features - reconstruction).square().mean(dim=-1)
        node_contributions = normalize_contributions(node_errors)
        pooled = self._pool(node_embeddings, self.method)
        output: dict[str, torch.Tensor] = {
            "reconstruction": reconstruction,
            "reconstruction_target": node_features,
            "node_contributions": node_contributions,
            "view_weights": view_weights,
        }
        view_logits = None
        state_logits = None
        ksd_logits = None
        state_summary = torch.cat((node_features.mean(dim=1), node_features.amax(dim=1)), dim=1)
        if self.method == "proposed":
            logits_by_view = torch.stack(
                [self.view_classifiers[view](self._pool(encoded_views[:, view], "phg_mtnet")) for view in range(3)],
                dim=1,
            )
            adaptive_logits = torch.einsum("bv,bvk->bk", view_weights, logits_by_view)
            paired_logits = logits_by_view[:, :2].mean(dim=1)
            view_logits = 0.5 * adaptive_logits + 0.5 * paired_logits
            state_logits = self.state_classifier(state_summary)
            ksd_nodes = torch.tanh(0.5 * encoded_views[:, 0] + 0.5 * encoded_views[:, 1])
            ksd_logits = self.ksd_residual_classifier(self._pool(ksd_nodes, "ksd_hgnn"))
        if not self.local_task:
            output["logits"] = (
                self.base_classifier(pooled)
                if view_logits is None
                else ksd_logits + 0.35 * (self.base_classifier(pooled) + view_logits + state_logits)
            )
            return output

        module_embeddings = self._aggregate_modules(node_embeddings)
        module_contributions = self._aggregate_modules(node_contributions.unsqueeze(-1)).squeeze(-1)
        initial_states = torch.sigmoid(self.module_state_head(module_embeddings).squeeze(-1))
        module_logits = self.module_classifier(module_embeddings).squeeze(-1) + 3.0 * module_contributions
        if self.method != "proposed":
            output.update(
                {
                    "logits": self.base_classifier(pooled),
                    "module_states": initial_states,
                    "module_logits": module_logits,
                    "torque_offsets": 0.35 * self.baseline_torque_head(pooled),
                    "path_consistency": torch.zeros((signals.shape[0], 1), device=signals.device),
                }
            )
            return output

        if self.variant == "fixed_propagation":
            gates = torch.full((signals.shape[0], 3), 1.0 / 3.0, device=signals.device, dtype=signals.dtype)
        else:
            state_rule_scores = torch.stack(
                (
                    node_features[:, :, 2].abs().mean(dim=1),
                    node_features[:, :, 3].mean(dim=1),
                    node_features[:, :, 4].mean(dim=1),
                ),
                dim=1,
            )
            state_rule_scores = state_rule_scores / state_rule_scores.detach().mean(dim=0, keepdim=True).clamp_min(1.0e-8)
            state_rule_gates = state_rule_scores / state_rule_scores.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
            gates = state_rule_gates
        operators = self.uniform_physical_operators if self.variant == "uniform_physical_weights" else self.physical_operators
        module_states = propagate_states(initial_states, operators, gates, self.propagation_coefficient)
        physical_torque = module_states[:, 2:6] * (0.40 * torch.sigmoid(self.torque_scale_logits)).unsqueeze(0)
        data_torque = 0.35 * self.baseline_torque_head(pooled)
        torque_offsets = 0.25 * physical_torque + 0.75 * data_torque
        query = torch.tanh(self.knowledge_query(pooled))
        knowledge_scores = torch.cat(
            (
                self._template_score(query, self.state_templates, self.state_template_projection),
                self._template_score(query, self.impact_templates, self.impact_template_projection),
                self._template_score(query, self.action_templates, self.action_template_projection),
            ),
            dim=1,
        )
        normalized_paths = functional.normalize(self.rule_paths, dim=1)
        all_path_scores = torch.matmul(functional.normalize(module_states, dim=1), normalized_paths.T)
        path_consistency = all_path_scores.amax(dim=1, keepdim=True)
        path_profile = all_path_scores[:, :4]
        if self.variant == "no_path_consistency":
            path_consistency = torch.zeros_like(path_consistency)
            path_profile = torch.zeros_like(path_profile)
            knowledge_scores = knowledge_scores.mean(dim=1, keepdim=True).expand(-1, 3)
        else:
            knowledge_scores = knowledge_scores * path_consistency
        module_summary = torch.stack((module_states.mean(dim=1), module_states.amax(dim=1)), dim=1)
        cascade_features = torch.cat(
            (module_summary, torque_offsets, knowledge_scores, path_profile), dim=1
        )
        output.update(
            {
                "logits": self.cascade_classifier(cascade_features),
                "module_states": module_states,
                "module_logits": module_logits,
                "torque_offsets": torque_offsets,
                "path_consistency": path_consistency,
            }
        )
        return output
