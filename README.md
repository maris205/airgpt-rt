# AirGPT Embodied AI Model Service

Unified FastAPI service for VLA and world-model inference. Mock mode covers the full
public contract without GPU access. Real mode currently supports OpenVLA and LTX-Video
on CUDA in their isolated Conda environments.

## Start

```bash
python -m pip install -e '.[dev]'
./scripts/dev.sh
```

Open `http://127.0.0.1:8000/docs` for the generated API documentation.

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

## Test

```bash
pytest
ruff check src tests
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
- Octo Small 1.5: M0 GPU probe passed; isolated JAX worker pending M4.
- LTX-Video 0.9.5: Diffusers runtime and artifact-backed GPU API passed M3.
- Cosmos: adapter boundary reserved; checkpoint access is currently gated.
- DreamerV3: recommended follow-on state-space worker; see `docs/DREAMER_V3_EVALUATION.md`.

AutoDL development uses native Conda environments. Docker artifacts will be validated
on a host with Docker Engine and NVIDIA Container Toolkit.
