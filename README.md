# AirGPT Embodied AI Model Service

Unified FastAPI service for VLA, image/video world-model, and task-specific state-space
inference. Mock mode covers the full public contract without GPU access. Real mode
supports OpenVLA, Octo, and LTX-Video on CUDA plus a DreamerV3 CartPole worker in an
isolated CPU JAX environment.

## Start

```bash
python -m pip install -e '.[dev]'
./scripts/dev.sh
```

Open `http://127.0.0.1:8000/docs` for the generated API documentation.

Open `http://127.0.0.1:8000/ui/` for the local runtime console.

### Single-model CLI

The v1 launcher starts one isolated runtime process, similar to the operational UX of
vLLM:

```bash
airgpt models list
airgpt serve --model openvla --port 8000
```

Use `--lazy` to defer model loading until the first request. See
[`docs/M5_RELEASE.md`](docs/M5_RELEASE.md) for local console and Docker profiles.

### OpenVLA GPU mode

The `airgpt-openvla` Conda environment and model cache must exist first. In the current
AutoDL workspace they were created and validated during M0.

```bash
./scripts/run_openvla.sh
./scripts/smoke_openvla_api.sh
```

`configs/models.yaml` currently uses local-cache-only loading to make startup
independent of Hugging Face connectivity.

### LTX-Video GPU mode

Run the world worker on port `8001` so it can coexist with the OpenVLA worker:

```bash
./scripts/run_ltx_world.sh
./scripts/smoke_ltx_api.sh
```

The response contains `/artifacts/...` URLs for individual frames and the generated
MP4. The default artifact directory is `artifacts/runtime-world`.

### Octo GPU mode

Run the isolated Octo/JAX worker on port `8003`:

```bash
./scripts/setup_octo.sh
./scripts/run_octo.sh
./scripts/smoke_octo_api.sh
```

Octo returns a common 7-DoF action plus its native action chunk in metadata. See
[`docs/M4_OCTO_AND_SIMULATORS.md`](docs/M4_OCTO_AND_SIMULATORS.md) for normalization,
cache, and simulator frame assumptions.

### DreamerV3 state-space mode

Set up the pinned official source and run the CartPole state worker on port `8002`:

```bash
./scripts/setup_dreamer.sh
./scripts/run_dreamer.sh
./scripts/smoke_dreamer_api.sh
```

See [`docs/M3_1_DREAMER.md`](docs/M3_1_DREAMER.md) for the fixed observation/action
contract and the task-specific checkpoint limitation.

## Test

```bash
pytest
ruff check src tests examples
```

## API

- `GET /health/live`
- `GET /health/ready`
- `GET /v1/models`
- `GET /v1/models/{model}`
- `POST /v1/models/{model}/load`
- `POST /v1/models/{model}/unload`
- `POST /v1/robot/action`
- `POST /v1/world/predict`
- `POST /v1/state/predict`

## SDK

```python
from pathlib import Path

from airgpt_sdk import Client

with Client("http://127.0.0.1:8000") as client:
    result = client.robot.action(
        model="openvla",
        image=Path("image.jpg"),
        instruction="pick up the cup",
    )
```

## Runtime Status

- OpenVLA 7B: production Transformers adapter and real GPU API passed M2.
- Octo Small 1.5: isolated JAX worker and real API action-chunk smoke passed M4.
- LTX-Video 0.9.5: Diffusers runtime and artifact-backed GPU API passed M3.
- Cosmos: adapter boundary reserved; checkpoint access is currently gated.
- DreamerV3 CartPole: M3.1 state-space worker passed real SDK/API rollout; see
  [`docs/M3_1_DREAMER.md`](docs/M3_1_DREAMER.md).

AutoDL development uses native Conda environments. Docker artifacts will be validated
on a host with Docker Engine and NVIDIA Container Toolkit.
