# AirGPT Embodied AI Model Service

Unified FastAPI service for VLA and world-model inference. The current M1 build uses
deterministic mock runtimes to freeze the public API before connecting the GPU workers
validated in M0.

## Start

```bash
python -m pip install -e '.[dev]'
./scripts/dev.sh
```

Open `http://127.0.0.1:8000/docs` for the generated API documentation.

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

- OpenVLA 7B: M0 GPU probe passed; production adapter pending M2.
- Octo Small 1.5: M0 GPU probe passed; isolated JAX worker pending M4.
- LTX-Video 0.9.5: M0 GPU probe passed; Diffusers worker pending M3.
- Cosmos: adapter boundary reserved; checkpoint access is currently gated.

AutoDL development uses native Conda environments. Docker artifacts will be validated
on a host with Docker Engine and NVIDIA Container Toolkit.

