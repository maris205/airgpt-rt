# M5 Release And Local Console

M5 turns the model service into a local v1 candidate for lab and classroom use. It
adds a vLLM-like single-model launcher, an offline-capable web console, and Docker
profiles that keep the PyTorch, JAX, TensorFlow, and Diffusers stacks isolated.

## CLI

Install the package in the environment that owns the selected runtime:

```bash
python -m pip install -e .
airgpt models list
airgpt models info octo
```

Start one model process:

```bash
airgpt serve --model openvla --device cuda:0 --port 8000
airgpt serve --model octo --device cuda:0 --port 8003
airgpt serve --model ltx_video --device cuda:0 --port 8001
airgpt serve --model dreamer_cartpole --device cpu --port 8002
```

`serve` preloads the selected model by default. Add `--lazy` to load it on the first
request. `--model-path`, `--revision`-compatible registry metadata, `--dtype`,
`--cache-dir`, `--max-concurrency`, and `--inference-timeout` are available as common
operational flags. Runtime-specific model parameters remain in the registry metadata.

The CLI intentionally starts one model per process. This avoids loading incompatible
JAX/PyTorch/TensorFlow dependency stacks into one Python interpreter and makes GPU
memory ownership predictable.

## Web console

Every service exposes the console at `/ui/` and redirects `/` there. It supports:

- model status, load, unload, and refresh;
- image/instruction action tests;
- image-conditioned world-model tests;
- observation/action-sequence state rollouts;
- action bars, frame/video previews, compact rollout metrics, and raw JSON responses.

The console is static HTML/CSS/JavaScript bundled with the Python package. It has no
Node build step and no external CDN dependency, so it works on an offline lab network.
It is an operations console, not an authentication boundary; keep it on a trusted local
network in v1.

## Docker

The AutoDL development container does not provide Docker Engine, so image builds cannot
be verified there. The repository includes one Dockerfile per runtime and Compose
profiles for a Docker-capable NVIDIA host:

```bash
docker compose --profile console up --build
docker compose --profile octo up --build
docker compose --profile openvla up --build
docker compose --profile world up --build
docker compose --profile dreamer up --build
```

Each profile is isolated, has a `/health/live` check, mounts model caches and artifacts,
and never bakes model weights into the image. Do not start multiple GPU profiles on one
32G card unless the measured memory budget has been checked for that combination.

## AutoDL release-candidate validation

On 2026-07-24, the single-model CLI preloaded and passed the live API smoke test for all
four v1 runtimes on the AutoDL host:

| Model | Runtime device | Result |
|---|---|---|
| OpenVLA 7B | NVIDIA vGPU-32GB | deterministic 7D action, pass |
| Octo Small 1.5 | NVIDIA vGPU-32GB | deterministic action chunk, pass |
| LTX-Video 0.9.5 | NVIDIA vGPU-32GB | 9-frame artifact and MP4, pass |
| DreamerV3 CartPole | CPU | 5-step latent and decoded rollout, pass |

The runtimes were started sequentially to preserve predictable GPU ownership. Startup
preloading runs through Starlette's worker threadpool because the pinned Octo/Orbax stack
uses `nest_asyncio`, which cannot patch Uvicorn's `uvloop` event loop directly.

## Release boundary

M5 does not include authentication, remote scheduling, Kubernetes, or the future AirGPT
Hub. The Hub remains a separate product that can later publish verified manifests and
download revisions into this local runtime.
