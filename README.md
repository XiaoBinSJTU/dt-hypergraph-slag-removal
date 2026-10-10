# Digital Twin Hypergraph Cascade Modeling for End-Effector Slag Removal Decisions in Steelmaking Slag-Handling Robots

[Bin Xiao on GitHub](https://github.com/XiaoBinSJTU)

## Paper Information

- **Title:** *Digital twin hypergraph cascade modeling for end-effector slag removal decisions in steelmaking slag-handling robots*
- **Authors:** Bin Xiao et al. 
- **Status:** Manuscript Under Review 
- **Repository type:** Research Reference Implementation

## Graphical Abstract

### Motivation

![Robot degradation motivation](https://raw.githubusercontent.com/XiaoBinSJTU/XiaoBinSJTU/main/assets/graphical-abstracts/07-rcim-robot-degradation.svg)

### Overall Framework

![Digital twin hypergraph cascade framework](https://raw.githubusercontent.com/XiaoBinSJTU/XiaoBinSJTU/main/assets/graphical-abstracts/07-rcim-cascade-framework.svg)

## Research Highlights

The method organizes slag-handling robot signals as a digital-twin hypergraph cascade. Multi-view part relations capture complementary temporal dependencies; physical-semantic propagation connects part evidence to module states; and structured knowledge matching links states, impacts, and actions to a final decision. This public package exposes those differentiable operators and their interfaces while deliberately excluding industrial records and manuscript-level experimental claims.

## Method Overview

### 1. Temporal part representation

Each sensor channel is represented by six differentiable descriptors: mean, deviation, trend, spectral peak, shock response, and energy. These descriptors form node features for the relation views.

### 2. Multi-view relation modeling

Similarity, nonlinear-coupling, and lag-direction views are converted into normalized relation operators. The model propagates node embeddings independently through each view.

### 3. Adaptive part-level fusion

Learned view scores combine the relation-specific embeddings. The implementation retains both the adaptive fusion and the per-view decision path used by the proposed model.

### 4. Module-level cascade propagation

Channel evidence is aggregated into six module states. Mechanical, transmission, and thermal operators propagate the states through a bounded iterative update, and the resulting representation supports torque-offset estimation.

### 5. Knowledge-guided decision

Projected state, impact, action, and rule-path templates are matched against the learned query. Their scores are combined with propagated module states to form the final decision logits.

## Method-to-Code Map

| Method component | File | Class or function |
| --- | --- | --- |
| Temporal node descriptors | `src/dt_hypergraph_slag_removal/features.py` | `temporal_node_features` |
| Relation-view construction | `src/dt_hypergraph_slag_removal/hypergraph.py` | `build_relation_views` |
| Physical-semantic propagation | `src/dt_hypergraph_slag_removal/hypergraph.py` | `physical_relation_matrices`, `propagate_states` |
| Three-level cascade | `src/dt_hypergraph_slag_removal/models.py` | `DigitalTwinHypergraphModel` |
| Joint objectives | `src/dt_hypergraph_slag_removal/losses.py` | `CompositeLoss` |
| Synthetic smoke-test adapter | `src/dt_hypergraph_slag_removal/demo.py` | `run_smoke_step` |

See [METHOD_TO_CODE.md](docs/METHOD_TO_CODE.md) for the full mapping and tensor conventions.

## Quick Start

The commands below are intended for Windows PowerShell with Python 3.10 or newer.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python examples\minimal_demo.py
python -m pytest -q
```

To force a device mode:

```powershell
python examples\minimal_demo.py --device cpu
python examples\minimal_demo.py --device cuda
```

The default `auto` mode selects a compatible GPU when available and otherwise uses the CPU path.

## Author

[Bin Xiao — XiaoBinSJTU](https://github.com/XiaoBinSJTU)
