# AirGPT Model Service v1.0 Development Plan

## Current status

- M0 complete: OpenVLA, Octo Small 1.5, and LTX-Video 0.9.5 passed native GPU probes.
- M1 complete: mock API, registry, lifecycle manager, SDK, OpenAPI, error contracts,
  tests, native startup, and Git baseline are operational.
- M2 complete: the OpenVLA Transformers adapter passed real SDK-to-API GPU inference,
  repeated-request stability, and unload verification.
- M3 complete: LTX-Video Diffusers runtime passed artifact-backed GPU API and video
  decoding checks.
- M3.1 complete: task-specific DreamerV3 CartPole state-space worker passed real
  SDK/API rollout with a pinned checkpoint and reproducible Gym 0.26 patch.
- M4 complete: isolated Octo/JAX worker, typed SDK improvements, model examples, and
  Isaac Sim/AirSim coordinate-mapping clients passed real API and contract tests.
- Next: M5, deployment hardening and release validation.

## 1. Goal and delivery scope

Build a single-node embodied-AI inference service with:

- FastAPI public API and OpenAPI documentation.
- YAML model registry and model lifecycle manager.
- Runtime/adapter separation for VLA and world models.
- OpenVLA, Octo, and one world-model integration.
- Python SDK and runnable examples.
- Native AutoDL development workflow plus Docker deployment artifacts.
- Unit, API, contract, and GPU smoke tests.

Explicitly excluded from v1: authentication, authorization, billing, GPU scheduling,
Kubernetes, Jetson, domestic accelerator support, and course-management features.

## 2. Current environment assessment

Observed on 2026-07-22:

- The repository contains only `product-v1.md`; no application code or Git metadata exists.
- AutoDL container: Ubuntu 22.04, Python 3.12.3, CUDA toolkit 12.1, PyTorch
  2.3.0+cu121.
- GPU is enabled as an AutoDL `NVIDIA vGPU-32GB` backed by an RTX 4080 SUPER (Ada,
  compute capability 8.9). It exposes 31.47 GiB to PyTorch; BF16 is supported.
- A 24 GiB allocation/write test and BF16 matrix multiplication both completed
  successfully. This is sufficient for the planned OpenVLA and Octo single-model
  inference paths, with roughly 7 GiB remaining after a 24 GiB allocation.
- Available persistent disk: about 136 GiB. This is enough for source code and a
  small number of models, but model cache retention must be controlled.
- Docker is not usable inside the current AutoDL container:
  - Docker CLI and daemon are absent.
  - No host Docker socket is mounted.
  - The container lacks `CAP_SYS_ADMIN`.
  - User namespaces/rootless containers are disabled.
- Installing Docker packages here would not make nested Docker work. Dockerfiles and
  Compose files can be authored in AutoDL, but image build and `docker compose up`
  must be verified on a Docker-capable host or CI runner.

Native Conda environments and direct Uvicorn processes will therefore be the primary
AutoDL development path.

## 3. Architecture decisions

### 3.1 Runtime separation

Do not force all VLA models through vLLM. The product document's proposed mapping is
not compatible with the actual model stacks:

- OpenVLA: Transformers/PyTorch plus model-specific action decoding.
- Octo: JAX/Octo stack.
- vLLM: optional runtime for models explicitly supported by vLLM; it is not the
  default OpenVLA or Octo backend.
- World models: Diffusers/PyTorch or model-specific runtime.

Use four runtime interfaces:

```text
BaseRuntime
|- TransformersVLARuntime
|- OctoJaxRuntime
|- VLLMRuntime             # optional/extension backend
`- DiffusionWorldRuntime
```

Each model adapter owns input conversion and output decoding. A runtime owns device,
precision, model loading, generation, and unload behavior.

### 3.2 Process and deployment layout

Use one public gateway and isolated runtime workers:

```text
Client -> FastAPI Gateway -> VLA Worker (PyTorch or JAX)
                        `-> World Worker (Diffusers/PyTorch)
```

This separation avoids Python dependency conflicts between PyTorch, JAX, vLLM, and
world-model packages. During the first CPU/mock milestone, all components may run in
one process behind the same interfaces. Final Compose deployment uses separate
services and internal HTTP health/inference contracts.

### 3.3 Model lifecycle

- Registry configuration is declarative; secrets and cache paths come from env vars.
- Models are loaded lazily unless marked for preload.
- v1 defaults to one loaded model per worker to prevent accidental GPU OOM.
- Manager state: `registered`, `loading`, `ready`, `failed`, `unloading`.
- Loading is serialized per worker; inference concurrency is bounded with a semaphore.
- OOM and load failures return structured errors and leave the worker recoverable.

### 3.4 API contract adjustments

Keep the requested public routes and add operational endpoints:

- `GET /health/live`
- `GET /health/ready`
- `GET /v1/models`
- `GET /v1/models/{model}`
- `POST /v1/models/{model}/load`
- `POST /v1/models/{model}/unload`
- `POST /v1/robot/action`
- `POST /v1/world/predict`

Robot input accepts an image as base64 or multipart upload through SDK helpers. World
model input must include valid conditioning data supported by the selected model,
such as an initial image/frame plus prompt/action. A `scene` string alone is not
sufficient for most video prediction models and must not be treated as a universal
contract.

Large generated frames should be written to an artifact directory and returned as
URLs/paths plus metadata. Returning long videos as JSON base64 is reserved for small
test payloads.

## 4. Proposed repository layout

```text
airgpt/
|- pyproject.toml
|- README.md
|- .env.example
|- Dockerfile.gateway
|- Dockerfile.vla
|- Dockerfile.world
|- docker-compose.yml
|- configs/
|  `- models.yaml
|- src/airgpt_service/
|  |- main.py
|  |- api/
|  |- core/
|  |- schemas/
|  |- runtimes/
|  |- adapters/
|  |  |- vla/
|  |  `- world/
|  `- workers/
|- src/airgpt_sdk/
|- examples/
|  |- openvla_demo.py
|  |- octo_demo.py
|  |- world_demo.py
|  |- isaac_sim_client.py
|  `- airsim_client.py
|- tests/
|  |- unit/
|  |- api/
|  |- contract/
|  `- gpu/
`- scripts/
   |- dev.sh
   |- smoke_api.sh
   `- verify_gpu.py
```

Use `pyproject.toml` with optional dependency groups rather than one global
`requirements.txt`:

- `gateway`: FastAPI, Pydantic, Uvicorn, HTTP client, YAML.
- `vla-openvla`: PyTorch/Transformers/OpenVLA dependencies.
- `vla-octo`: JAX/Octo dependencies.
- `world`: Diffusers and the selected world-model dependencies.
- `dev`: pytest, pytest-asyncio, HTTP test client, Ruff, mypy.

Exact versions are frozen only after the GPU compatibility spike.

## 5. Milestones

### M0 - Compatibility spike and contract freeze (1-2 working days, GPU required)

Tasks:

- Record the enabled AutoDL GPU, VRAM, driver, CUDA visibility, and free disk in the
  compatibility report; repeat the check after instance/image changes.
- Create isolated environments for OpenVLA, Octo, and candidate world model.
- Run one upstream/native inference for each model before wrapping it in the service.
- Measure model download size, peak VRAM, load time, inference latency, and output
  format.
- Confirm model licenses, Hugging Face access requirements, and whether remote code is
  required.
- Select the v1 world model based on the actual GPU. Prefer the smallest model that
  produces a meaningful conditioned future-frame demo. Treat NVIDIA Cosmos as an
  optional target if hardware/license requirements are too high.
- Freeze normalized action and world-prediction schemas from observed outputs.

Exit criteria:

- OpenVLA produces a decoded action on one sample.
- Octo produces an action on one sample, or a documented blocker and replacement
  decision is approved.
- One world model produces frames on one sample.
- A compatibility matrix with pinned versions and measured resource needs exists.

### M1 - CPU/mock service skeleton (2 working days, no GPU)

Tasks:

- Initialize Git and Python package structure.
- Implement settings, logging, error schema, request IDs, and startup/shutdown hooks.
- Implement YAML registry validation and model metadata schemas.
- Implement model manager state machine and mock runtimes/adapters.
- Add model, action, world, liveness, and readiness routes.
- Generate OpenAPI examples matching the SDK.
- Add unit and API tests that never download model weights.

Exit criteria:

- `pytest` passes without GPU or network access.
- Uvicorn starts locally and `/docs`, health, models, robot, and world mock APIs work.
- Invalid model, invalid image, timeout, model-load failure, and busy/OOM-style errors
  have stable HTTP status codes and machine-readable bodies.

### M2 - OpenVLA production path (2-3 working days, GPU required)

Tasks:

- Implement `TransformersVLARuntime` device/dtype/loading lifecycle.
- Implement OpenVLA image/instruction preprocessing and action decoding adapter.
- Add deterministic sample fixture and GPU integration test.
- Add request serialization, inference timeout, concurrency limit, and OOM cleanup.
- Record action dimensions, normalization rules, and robot-specific metadata rather
  than returning an unexplained float list.

Exit criteria:

- `/v1/robot/action` returns a validated OpenVLA action for the reference image.
- Repeated requests do not continually increase GPU memory.
- Load failure and CUDA OOM are surfaced cleanly without crashing the gateway.

### M3 - World-model production path (2-3 working days, GPU required)

Tasks:

- Implement selected world runtime and adapter from the M0 decision.
- Add artifact storage, output metadata, cleanup/TTL, and maximum generation limits.
- Implement synchronous inference for the v1 demo; keep the internal service boundary
  ready for later asynchronous jobs.
- Add GPU smoke test using low resolution and few frames.

Exit criteria:

- `/v1/world/predict` produces viewable future frames from documented conditioning
  input.
- Output resolution, frame count, seed, latency, and model name are returned.
- Generated artifacts are bounded and automatically cleaned up.

### M4 - Octo, SDK, and simulator examples (2-3 working days, GPU for Octo)

Status: **complete**. The implementation and measured exit criteria are documented in
[`docs/M4_OCTO_AND_SIMULATORS.md`](docs/M4_OCTO_AND_SIMULATORS.md).

Tasks:

- Implement isolated Octo/JAX worker and normalized action adapter.
- Implement typed Python SDK with timeout, retries for safe GET/load operations,
  image-file helpers, and structured exceptions.
- Add OpenVLA, Octo, and world examples.
- Add Isaac Sim and AirSim client examples that demonstrate coordinate/action mapping;
  simulator installation is not bundled into this service.
- Add gateway-to-worker contract tests.

Exit criteria:

- `client.robot.action(...)`, `client.world.predict(...)`, and model listing work.
- OpenVLA and Octo responses share an envelope but retain model-specific action
  metadata.
- Simulator examples run against mock mode and are documented for real mode.

### M3.1 - DreamerV3 state-space worker (2-3 working days, task-specific)

Status: **complete**. The implementation and measured exit criteria are documented in
[`docs/M3_1_DREAMER.md`](docs/M3_1_DREAMER.md).

Tasks:

- Freeze one Gymnasium/robot observation and action contract with a reproducible
  checkpoint and normalization metadata.
- Add a separate `state_world` registry type and `/v1/state/predict` request/response
  envelope for latent rollouts.
- Keep the RSSM worker isolated from the Diffusers environment; expose optional decoded
  observations only when the matching decoder is available.

Exit criteria:

- A known task produces deterministic latent rollouts from an observation and action
  sequence, with shape, horizon, uncertainty, and checkpoint metadata returned.
- The service does not imply that a task-specific Dreamer checkpoint is a universal
  image-to-video world model.

### M5 - Deployment hardening and release (2 working days)

Tasks:

- Build separate gateway, VLA, and world images with pinned base images.
- Add Compose profiles so mock/gateway can run without GPU and real workers request
  NVIDIA GPU devices.
- Mount model cache and generated artifacts as volumes; never bake weights into images.
- Add health checks, restart policy, resource limits, env configuration, and log paths.
- Run lint, type checks, unit/API tests, GPU smoke tests, and end-to-end SDK tests.
- Complete README, API examples, troubleshooting, model licenses, and resource table.

Exit criteria:

- On a Docker-capable NVIDIA host, `docker compose up` starts healthy services and the
  SDK completes one robot and one world inference.
- In AutoDL, equivalent native worker commands pass the same smoke tests.
- A clean machine can reproduce the deployment from documented steps and pinned
  versions.

## 6. GPU usage plan

GPU does not need to stay enabled for API, registry, SDK, mock adapter, and most test
development. Enable it for M0, M2, M3, the Octo portion of M4, and the final GPU smoke
suite.

Initial capacity assumptions to validate in M0:

| Workload | Planning target | Notes |
|---|---:|---|
| OpenVLA 7B | Fits the available 31.5 GiB | Start with BF16; quantization is not required for the first correctness path. |
| Octo | Fits the available 31.5 GiB | Disable or cap JAX preallocation. |
| Small video/world model | Expected to fit at low resolution/frame count | Prefer a 2B-class or otherwise memory-efficient image/video-conditioned checkpoint. |
| NVIDIA Cosmos class model | Checkpoint dependent | Only a small/2B-class checkpoint is a reasonable 32 GiB target; larger 7B/14B video checkpoints are not a v1 baseline. |

Do not load OpenVLA, Octo, and the world model on one GPU simultaneously in v1.

## 7. Testing strategy

- Unit: registry validation, manager transitions, adapters, SDK encoding, errors.
- API: all routes with mock runtimes, validation and payload size limits.
- Contract: gateway-to-worker request/response compatibility.
- GPU smoke: one minimal real inference per supported model, marked and opt-in.
- Stability: repeated inference, unload/reload, timeout, invalid input, and simulated
  OOM/load failure.
- End-to-end: SDK -> gateway -> worker -> artifact/action response.
- Docker: build and Compose tests on an external NVIDIA Docker host/CI, not this
  restricted AutoDL container.

CI must not download multi-gigabyte weights. Real-model tests use a pre-populated cache
and explicit `RUN_GPU_TESTS=1`.

## 8. Delivery schedule and gates

Estimated implementation: 11-15 working days after model access and a suitable GPU are
available.

| Gate | Deliverable | Decision |
|---|---|---|
| G0 | Compatibility report | Pin runtimes/models or change unsupported target. |
| G1 | Mock API/SDK contract | Freeze public v1 request/response envelopes. |
| G2 | OpenVLA demo | Accept robot-action path. |
| G3 | World demo | Accept world-model choice and conditioning schema. |
| G4 | Octo + simulators | Accept second VLA and integration examples. |
| G5 | Native + Docker release | Tag v1.0 release candidate. |

## 9. Main risks and mitigations

- **vLLM mismatch:** keep it optional and use the model's supported native runtime.
- **Octo dependency conflict:** isolate JAX in its own environment/container.
- **World API underspecified:** freeze conditioning schema after a real model spike.
- **Cosmos resource cost:** use a smaller world model for v1 if the selected AutoDL GPU
  cannot run Cosmos; keep the adapter boundary stable.
- **Model output incompatibility:** normalized envelope plus model/robot-specific action
  metadata; do not pretend all action spaces are identical.
- **GPU OOM:** lazy loading, one model per worker, bounded concurrency, unload endpoint,
  low-resolution smoke settings.
- **Disk pressure:** shared Hugging Face cache, artifact TTL, documented cache cleanup,
  no model weights in Git or images.
- **Docker unavailable on AutoDL:** develop and test natively there; verify containers
  on a host with Docker Engine, NVIDIA Container Toolkit, and GPU access.

## 10. v1 definition of done

v1 is complete only when:

- OpenVLA, Octo, and one selected world model each pass a real inference smoke test.
- Public APIs, Python SDK, model lifecycle, health endpoints, and examples are usable.
- Mock tests pass without GPU; GPU tests are reproducible from a pre-populated cache.
- Native AutoDL startup is documented and tested.
- Docker Compose deployment is tested on a Docker-capable NVIDIA host.
- Known model limitations, action schemas, licenses, VRAM, disk usage, and latency are
  documented from measurements rather than assumptions.
