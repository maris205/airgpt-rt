#!/usr/bin/env python3
"""Run native Octo inference and record load, compile, and warm latency metrics."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import flax
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import tensorflow as tf
from PIL import Image

# TensorFlow is used by Octo for file and text utilities. Keep GPU ownership in JAX.
tf.config.set_visible_devices([], "GPU")

from octo.model.octo_model import OctoModel  # noqa: E402


DEFAULT_MODEL = "hf://rail-berkeley/octo-small-1.5"
DEFAULT_INSTRUCTION = "pick up the fork"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--dataset", default="bridge_dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/m0/octo"))
    return parser.parse_args()


def gpu_memory_mib() -> dict[str, int] | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        used, free, total = (int(value.strip()) for value in output.split(","))
        return {"used_mib": used, "free_mib": free, "total_mib": total}
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def block_until_ready(tree: Any) -> Any:
    return jax.tree_map(
        lambda value: value.block_until_ready() if isinstance(value, jax.Array) else value,
        tree,
    )


def main() -> None:
    args = parse_args()
    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError(f"Octo requires a JAX GPU backend; found {jax.devices()}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image = np.asarray(Image.open(args.image).convert("RGB").resize((256, 256)))
    image = image[np.newaxis, np.newaxis, ...]
    observation = {
        "image_primary": image,
        "timestep_pad_mask": np.array([[True]]),
    }

    report: dict[str, Any] = {
        "model": args.model,
        "instruction": args.instruction,
        "dataset": args.dataset,
        "image": str(args.image),
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "flax": flax.__version__,
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "jax_devices": [str(device) for device in jax.devices()],
            "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        },
        "gpu_before_load": gpu_memory_mib(),
    }

    load_start = time.perf_counter()
    model = OctoModel.load_pretrained(args.model)
    block_until_ready(model.params)
    report["model_load_seconds"] = round(time.perf_counter() - load_start, 3)
    report["gpu_after_load"] = gpu_memory_mib()

    task = model.create_tasks(texts=[args.instruction])
    statistics = model.dataset_statistics[args.dataset]["action"]

    def predict(key: jax.Array) -> Any:
        return model.sample_actions(
            observation,
            task,
            unnormalization_statistics=statistics,
            rng=key,
        )

    compile_start = time.perf_counter()
    first_action = block_until_ready(predict(jax.random.PRNGKey(0)))
    report["first_inference_seconds"] = round(time.perf_counter() - compile_start, 3)
    report["gpu_after_first_inference"] = gpu_memory_mib()

    warm_start = time.perf_counter()
    warm_action = block_until_ready(predict(jax.random.PRNGKey(1)))
    report["warm_inference_seconds"] = round(time.perf_counter() - warm_start, 3)
    report["gpu_after_warm_inference"] = gpu_memory_mib()
    report["action"] = np.asarray(first_action).tolist()
    report["action_shape"] = list(np.asarray(first_action).shape)
    report["warm_action_finite"] = bool(np.isfinite(np.asarray(warm_action)).all())

    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report_path={report_path}")


if __name__ == "__main__":
    main()
