"""Run the core cascade with synthetic inputs for demonstration only."""

from __future__ import annotations

import argparse

from dt_hypergraph_slag_removal.demo import run_smoke_step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    result = run_smoke_step(args.device)
    print(f"demo_status={result['status']}")
    print(f"device_mode={result['device_mode']}")
    print(f"forward_shape={result['forward_shape']}")
    print(f"loss_backward={result['loss_backward']}")


if __name__ == "__main__":
    main()
