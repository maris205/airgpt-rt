#!/usr/bin/env python3
"""Run one image-conditioned world-model request through the AirGPT SDK."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from airgpt_sdk import Client


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("action")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--num-frames", type=int, default=9)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    with Client(args.base_url, timeout=240) as client:
        result = client.world.predict(
            "ltx_video",
            image=args.image,
            action=args.action,
            num_frames=args.num_frames,
            seed=args.seed,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
