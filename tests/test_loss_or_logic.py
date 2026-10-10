"""Loss and backward checks using synthetic inputs only."""

from dt_hypergraph_slag_removal.demo import run_smoke_step


def test_composite_loss_and_backward() -> None:
    result = run_smoke_step("cpu")
    assert result["status"] == "PASS"
    assert result["forward_shape"] == (4, 2)
    assert result["loss_backward"] == "PASS"
