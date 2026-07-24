from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import uvicorn

from .app import create_app
from .config import DEFAULT_CONFIG_PATH, Settings
from .registry import ModelRegistry


DEFAULT_PORTS = {
    "openvla": 8000,
    "ltx_video": 8001,
    "dreamer_cartpole": 8002,
    "octo": 8003,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="airgpt", description="AirGPT local model runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="start one model service")
    serve.add_argument("--model", required=True)
    serve.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    serve.add_argument("--model-path")
    serve.add_argument("--revision")
    serve.add_argument("--device")
    serve.add_argument("--dtype")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int)
    serve.add_argument("--worker-mode", choices=("real", "mock"), default="real")
    serve.add_argument("--max-concurrency", type=int, default=1)
    serve.add_argument("--inference-timeout", type=float, default=240.0)
    serve.add_argument("--artifact-dir", type=Path)
    serve.add_argument("--cache-dir", type=Path)
    serve.add_argument("--log-level", default="INFO")
    serve.add_argument("--lazy", action="store_true", help="load on first inference request")
    serve.add_argument("--trust-remote-code", action="store_true")

    models = subparsers.add_parser("models", help="inspect registered models")
    model_subparsers = models.add_subparsers(dest="models_command", required=True)
    model_list = model_subparsers.add_parser("list", help="list model manifests")
    model_list.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    model_list.add_argument("--json", action="store_true")
    model_info = model_subparsers.add_parser("info", help="show one model manifest")
    model_info.add_argument("model")
    model_info.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser


def _serve(args: argparse.Namespace) -> int:
    ModelRegistry(args.config, selected_model=args.model)
    if args.cache_dir:
        os.environ["HF_HOME"] = str(args.cache_dir)
    port = args.port or DEFAULT_PORTS.get(args.model, 8000)
    settings = Settings(
        model_config_path=args.config,
        selected_model=args.model,
        model_path_override=args.model_path,
        revision_override=args.revision,
        device_override=args.device,
        dtype_override=args.dtype,
        trust_remote_code=args.trust_remote_code,
        preload_model=not args.lazy,
        artifact_dir=args.artifact_dir or Settings.artifact_dir,
        worker_mode=args.worker_mode,
        log_level=args.log_level,
        max_concurrent_inference=args.max_concurrency,
        inference_timeout_seconds=args.inference_timeout,
    )
    uvicorn.run(
        create_app(settings),
        host=args.host,
        port=port,
        log_level=args.log_level.lower(),
    )
    return 0


def _models(args: argparse.Namespace) -> int:
    registry = ModelRegistry(args.config)
    if args.models_command == "info":
        print(json.dumps(registry.get(args.model).model_dump(), indent=2, sort_keys=True))
        return 0
    models = [
        {"name": name, **spec.model_dump()} for name, spec in registry.items() if spec.enabled
    ]
    if args.json:
        print(json.dumps({"models": models}, indent=2, sort_keys=True))
        return 0
    print(f"{'MODEL':24} {'TYPE':12} {'BACKEND':16} {'DEVICE':10} READY")
    for model in models:
        ready = str(model["metadata"].get("runtime_ready", False)).lower()
        print(
            f"{model['name']:24} {model['type']:12} {model['backend']:16} "
            f"{model['device']:10} {ready}"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "serve":
        return _serve(args)
    return _models(args)


if __name__ == "__main__":
    raise SystemExit(main())
