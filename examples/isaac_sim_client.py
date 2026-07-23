#!/usr/bin/env python3
"""Map a BridgeV2 action to an Isaac-style end-effector command.

The script only depends on the AirGPT SDK. Isaac Sim installation and transport are
intentionally left to the simulator application.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from airgpt_sdk import Client


def bridge_action_to_isaac(action: list[float]) -> dict[str, Any]:
    if len(action) != 7:
        raise ValueError("Bridge action must contain 7 values")
    return {
        "ee_delta_position_m": action[:3],
        "ee_delta_rotation_axis_angle_rad": action[3:6],
        "gripper": action[6],
        "gripper_convention": "BridgeV2 dataset convention",
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
            {"model_result": result, "isaac_command": bridge_action_to_isaac(result["action"])},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
