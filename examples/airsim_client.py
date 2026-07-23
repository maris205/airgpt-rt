#!/usr/bin/env python3
"""Map a BridgeV2 action to an AirSim NED-frame command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from airgpt_sdk import Client


def bridge_action_to_airsim(action: list[float]) -> dict[str, Any]:
    if len(action) != 7:
        raise ValueError("Bridge action must contain 7 values")
    dx, dy, dz, droll, dpitch, dyaw, gripper = action
    return {
        "ned_delta_position_m": {"north": dx, "east": dy, "down": -dz},
        "body_delta_rotation_rad": {"roll": droll, "pitch": dpitch, "yaw": dyaw},
        "gripper": gripper,
        "frame_note": "Assumes Bridge +Z-up coordinates mapped to AirSim NED +Z-down",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("instruction")
    parser.add_argument("--model", default="octo")
    parser.add_argument("--base-url", default="http://127.0.0.1:8003")
    args = parser.parse_args()
    with Client(args.base_url, timeout=180) as client:
        result = client.robot.action(
            args.model,
            args.image,
            args.instruction,
            metadata={"seed": 0, "dataset": "bridge_dataset"} if args.model == "octo" else {},
        )
    print(
        json.dumps(
            {"model_result": result, "airsim_command": bridge_action_to_airsim(result["action"])},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
