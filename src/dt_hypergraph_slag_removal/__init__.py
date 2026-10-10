"""Core reference implementation for digital-twin hypergraph cascade reasoning."""

from .device import release_device_cache, select_device
from .features import normalize_contributions, temporal_node_features
from .hypergraph import (
    build_relation_views,
    normalize_adjacency,
    physical_relation_matrices,
    propagate_states,
)
from .losses import CompositeLoss
from .models import DigitalTwinHypergraphModel

__all__ = [
    "CompositeLoss",
    "DigitalTwinHypergraphModel",
    "build_relation_views",
    "normalize_adjacency",
    "normalize_contributions",
    "physical_relation_matrices",
    "propagate_states",
    "release_device_cache",
    "select_device",
    "temporal_node_features",
]

__version__ = "0.1.0"
