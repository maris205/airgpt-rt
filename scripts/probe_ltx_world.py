#!/usr/bin/env python3
"""Generate a short conditioned future video with LTX-Video and record metrics."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import diffusers
import psutil
import torch
import transformers
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video
from PIL import Image


DEFAULT_PROMPT = (
    "A robot gripper moves toward the fork on the kitchen counter, closes around the "
    "fork, and lifts it slightly. The camera remains fixed and the motion is smooth."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default="blurry, jittery, distorted, camera movement")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/m0/world"))
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


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the LTX world-model probe")
    if args.frames % 8 != 1:
        raise ValueError("LTX frame count must be 8n + 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(args.image).convert("RGB")
    process = psutil.Process()
    report: dict[str, Any] = {
        "model": str(args.model),
        "image": str(args.image),
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "height": args.height,
        "width": args.width,
        "num_frames": args.frames,
        "num_inference_steps": args.steps,
        "seed": args.seed,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "diffusers": diffusers.__version__,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "bf16_supported": torch.cuda.is_bf16_supported(),
        },
        "gpu_before_load": gpu_memory_mib(),
        "rss_before_load_gib": round(process.memory_info().rss / 1024**3, 3),
    }

    load_start = time.perf_counter()
    pipe = LTXImageToVideoPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    report["model_load_seconds"] = round(time.perf_counter() - load_start, 3)
    report["rss_after_load_gib"] = round(process.memory_info().rss / 1024**3, 3)
    report["gpu_after_load"] = gpu_memory_mib()

    torch.cuda.reset_peak_memory_stats()
    generation_start = time.perf_counter()
    with torch.inference_mode():
        frames = pipe(
            image=image,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            num_inference_steps=args.steps,
            guidance_scale=3.0,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
            max_sequence_length=128,
        ).frames[0]
    torch.cuda.synchronize()
    report["generation_seconds"] = round(time.perf_counter() - generation_start, 3)
    report["gpu_peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    report["gpu_peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    report["gpu_after_generation"] = gpu_memory_mib()
    report["output_frame_count"] = len(frames)
    report["output_frame_size"] = list(frames[0].size) if frames else None

    video_path = args.output_dir / "future.mp4"
    export_to_video(frames, str(video_path), fps=8)
    report["video"] = str(video_path)
    report["video_size_mib"] = round(video_path.stat().st_size / 1024**2, 3)

    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report_path={report_path}")


if __name__ == "__main__":
    main()
