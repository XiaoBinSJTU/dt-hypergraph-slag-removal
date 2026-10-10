"""Forward-path checks using synthetic inputs only."""

import torch

from dt_hypergraph_slag_removal.demo import build_synthetic_case


def test_core_forward_shapes_and_constraints() -> None:
    model, batch = build_synthetic_case(torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        output = model(batch["signals"])

    assert output["logits"].shape == (4, 2)
    assert output["module_states"].shape == (4, 6)
    assert output["torque_offsets"].shape == (4, 4)
    assert torch.isfinite(output["logits"]).all()
    assert torch.all((output["module_states"] >= 0.0) & (output["module_states"] <= 1.0))
    assert torch.allclose(output["view_weights"].sum(dim=1), torch.ones(4), atol=1.0e-6)
