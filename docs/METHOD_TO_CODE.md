# Method-to-Code Map

This map connects the manuscript's algorithmic stages to the public implementation. It describes source-code behavior.

| Method stage | Public implementation | Main entry points |
| --- | --- | --- |
| Temporal part representation | `src/dt_hypergraph_slag_removal/features.py` | `temporal_node_features`, `normalize_contributions` |
| Multi-view relation construction | `src/dt_hypergraph_slag_removal/hypergraph.py` | `build_relation_views`, `normalize_adjacency` |
| Part-level relation propagation and fusion | `src/dt_hypergraph_slag_removal/models.py` | `DigitalTwinHypergraphModel._encode_views` |
| Module-level physical-semantic propagation | `src/dt_hypergraph_slag_removal/hypergraph.py`, `models.py` | `physical_relation_matrices`, `propagate_states`, `DigitalTwinHypergraphModel.forward` |
| State-impact-action knowledge matching | `src/dt_hypergraph_slag_removal/models.py` | `DigitalTwinHypergraphModel._template_score`, `DigitalTwinHypergraphModel.forward` |
| Joint optimization objectives | `src/dt_hypergraph_slag_removal/losses.py` | `CompositeLoss.forward` |
| Hardware-neutral execution | `src/dt_hypergraph_slag_removal/device.py` | `select_device`, `release_device_cache` |
| Non-sensitive smoke test | `src/dt_hypergraph_slag_removal/demo.py`, `examples/minimal_demo.py` | `build_synthetic_case`, `run_smoke_step` |

## Information flow

1. A batched signal tensor is transformed into six temporal descriptors per channel.
2. Three relation operators propagate part embeddings and are fused by learned view scores.
3. Channel states are aggregated into six module states.
4. Physical-semantic operators propagate module states through the cascade.
5. Structured state, impact, action, and rule-path templates contribute to the final decision logits.
6. The composite objective can jointly supervise the decision, reconstruction, torque, and propagation paths.

## Public interface assumptions

- Input tensors use the shape `[batch, time, channels]`.
- The reference cascade expects three channel-relation operators.
- The local cascade maps channels to six modules.
