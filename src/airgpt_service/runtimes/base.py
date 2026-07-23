from __future__ import annotations

from typing import Protocol

from ..schemas import (
    RobotActionResponse,
    RobotActionRequest,
    WorldPredictRequest,
    WorldPredictResponse,
)


class Runtime(Protocol):
    device: str

    def load(self) -> None: ...

    def unload(self) -> None: ...


class VLAAdapter(Runtime, Protocol):
    def predict_action(self, request: RobotActionRequest) -> RobotActionResponse: ...


class WorldAdapter(Runtime, Protocol):
    def predict_world(self, request: WorldPredictRequest) -> WorldPredictResponse: ...
