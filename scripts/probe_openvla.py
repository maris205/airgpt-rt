#!/usr/bin/env python3
"""Run one native OpenVLA action prediction and record compatibility metrics."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import transformers
from PIL import Image, ImageDraw
from transformers import AutoModelForVision2Seq, AutoProcessor


DEFAULT_MODEL = "openvla/openvla-7b"
DEFAULT_INSTRUCTION = "pick up the red cube"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--unnorm-key", default="bridge_orig")
    parser.add_argument(
        "--attention",
        choices=("sdpa", "eager", "flash_attention_2"),
        default="sdpa",
        help="SDPA avoids OpenVLA issue #58 without requiring FlashAttention.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/m0/openvla"))
    return parser.parse_args()


def make_test_image(path: Path) -> Image.Image:
    image = Image.new("RGB", (640, 480), (205, 210, 215))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 300, 640, 480), fill=(142, 105, 70))
    draw.rectangle((255, 205, 385, 335), fill=(205, 35, 40), outline=(90, 15, 20), width=5)
    draw.ellipse((430, 235, 525, 330), fill=(235, 235, 225), outline=(80, 80, 80), width=4)
    draw.rectangle((100, 250, 190, 330), fill=(45, 105, 180), outline=(20, 45, 90), width=4)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return image


def gpu_snapshot() -> dict[str, Any]:
    free, total = torch.cuda.mem_get_info()
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_gib": round(total / 1024**3, 3),
        "free_gib": round(free / 1024**3, 3),
        "allocated_gib": round(torch.cuda.memory_allocated() / 1024**3, 3),
        "reserved_gib": round(torch.cuda.memory_reserved() / 1024**3, 3),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    }


def nvidia_driver() -> str | None:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the OpenVLA compatibility probe")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.image or args.output_dir / "synthetic_tabletop.png"
    image = Image.open(image_path).convert("RGB") if args.image else make_test_image(image_path)
    prompt = f"In: What action should the robot take to {args.instruction.lower()}?\nOut:"

    process = psutil.Process()
    report: dict[str, Any] = {
        "model": args.model,
        "instruction": args.instruction,
        "prompt": prompt,
        "unnorm_key": args.unnorm_key,
        "attention_implementation": args.attention,
        "image": str(image_path),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "driver": nvidia_driver(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        },
        "gpu_before_load": gpu_snapshot(),
        "rss_before_load_gib": round(process.memory_info().rss / 1024**3, 3),
    }

    processor_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    report["processor_load_seconds"] = round(time.perf_counter() - processor_start, 3)

    torch.cuda.reset_peak_memory_stats()
    model_start = time.perf_counter()
    model = AutoModelForVision2Seq.from_pretrained(
        args.model,
        attn_implementation=args.attention,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda:0")
    torch.cuda.synchronize()
    report["model_load_seconds"] = round(time.perf_counter() - model_start, 3)
    report["gpu_after_load"] = gpu_snapshot()
    report["rss_after_load_gib"] = round(process.memory_info().rss / 1024**3, 3)

    inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats()
    inference_start = time.perf_counter()
    with torch.inference_mode():
        action = model.predict_action(
            **inputs,
            unnorm_key=args.unnorm_key,
            do_sample=False,
        )
    torch.cuda.synchronize()
    report["inference_seconds"] = round(time.perf_counter() - inference_start, 3)
    report["gpu_after_inference"] = gpu_snapshot()
    report["action"] = np.asarray(action).tolist()
    report["action_shape"] = list(np.asarray(action).shape)

    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report_path={report_path}")


if __name__ == "__main__":
    main()
