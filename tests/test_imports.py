"""Import checks for the public package."""


def test_public_imports() -> None:
    from dt_hypergraph_slag_removal import CompositeLoss, DigitalTwinHypergraphModel

    assert CompositeLoss.__name__ == "CompositeLoss"
    assert DigitalTwinHypergraphModel.__name__ == "DigitalTwinHypergraphModel"
