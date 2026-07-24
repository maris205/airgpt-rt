from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import uvicorn

from .app import create_app
from .config import DEFAULT_CONFIG_PATH, Settings
from .hub import HubClient, HubError, launch_command
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

    hub = subparsers.add_parser("hub", help="search the AirGPT embodied model catalog")
    hub_subparsers = hub.add_subparsers(dest="hub_command", required=True)
    hub_search = hub_subparsers.add_parser("search", help="search verified Hub manifests")
    hub_search.add_argument("query", nargs="?", default="")
    hub_search.add_argument("--task", choices=("vla", "world", "state_world"))
    hub_search.add_argument("--verified-only", action="store_true")
    hub_search.add_argument("--limit", type=int, default=20)
    hub_search.add_argument("--json", action="store_true")
    _add_catalog_options(hub_search)
    hub_info = hub_subparsers.add_parser("info", help="show one Hub manifest")
    hub_info.add_argument("model")
    _add_catalog_options(hub_info)

    pull = subparsers.add_parser("pull", help="download a pinned model from the Hub")
    pull.add_argument("model")
    pull.add_argument("--cache-dir", type=Path)
    pull.add_argument("--local-dir", type=Path)
    pull.add_argument("--revision")
    pull.add_argument("--dry-run", action="store_true")
    pull.add_argument("--json", action="store_true")
    _add_catalog_options(pull)
    return parser


def _add_catalog_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--catalog-url")
    parser.add_argument("--catalog-file", type=Path)


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


def _hub_client(args: argparse.Namespace) -> HubClient:
    return HubClient(catalog_url=args.catalog_url, catalog_file=args.catalog_file)


def _hub(args: argparse.Namespace) -> int:
    client = _hub_client(args)
    if args.hub_command == "info":
        print(json.dumps(client.get(args.model).model_dump(), indent=2, sort_keys=True))
        return 0
    if args.limit < 1:
        raise HubError("--limit must be at least 1")
    models = client.search(
        args.query,
        task=args.task,
        verified_only=args.verified_only,
        limit=args.limit,
    )
    if args.json:
        print(
            json.dumps(
                {"catalog": client.source, "models": [model.model_dump() for model in models]},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    print(f"{'MODEL':24} {'TASK':12} {'STATUS':12} {'VRAM':>6} SOURCE")
    for model in models:
        print(
            f"{model.slug:24} {model.task:12} {model.verification.status:12} "
            f"{model.hardware.recommended_vram_gb:>4}GB {model.source.repo_id}"
        )
    return 0


def _pull(args: argparse.Namespace) -> int:
    client = _hub_client(args)
    if args.dry_run:
        model = client.get(args.model)
        path: Path | str = args.local_dir or "<downloaded-path>"
    else:
        model, path = client.pull(
            args.model,
            cache_dir=args.cache_dir,
            local_dir=args.local_dir,
            revision=args.revision,
        )
    payload = {
        "model": model.slug,
        "repo_id": model.source.repo_id,
        "revision": args.revision or model.source.revision,
        "path": str(path),
        "launch_command": launch_command(model, path),
        "dry_run": args.dry_run,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Model: {payload['repo_id']}@{payload['revision']}")
        print(f"Path: {payload['path']}")
        print(f"Run: {payload['launch_command']}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "serve":
            return _serve(args)
        if args.command == "models":
            return _models(args)
        if args.command == "hub":
            return _hub(args)
        return _pull(args)
    except HubError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
