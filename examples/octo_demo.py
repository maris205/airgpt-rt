#!/usr/bin/env python3
"""Run one Octo BridgeV2 action-chunk request through the AirGPT SDK."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from airgpt_sdk import Client


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("instruction")
    parser.add_argument("--base-url", default="http://127.0.0.1:8003")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    with Client(args.base_url, timeout=180) as client:
        result = client.robot.action(
            "octo",
            args.image,
            args.instruction,
            metadata={"seed": args.seed, "dataset": "bridge_dataset"},
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
