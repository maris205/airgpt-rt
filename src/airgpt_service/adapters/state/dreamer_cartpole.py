from __future__ import annotations

import numpy as np

from ...errors import InvalidInputError
from ...schemas import StatePredictRequest


class DreamerCartPoleAdapter:
    observation_dim = 4
    action_count = 2

    def preprocess(self, request: StatePredictRequest) -> tuple[np.ndarray, np.ndarray]:
        if len(request.observation) != self.observation_dim:
            raise InvalidInputError(
                f"CartPole observation must contain {self.observation_dim} values"
            )
        observation = np.asarray(request.observation, dtype=np.float32)
        if not np.isfinite(observation).all():
            raise InvalidInputError("observation values must be finite")
        actions = []
        for action in request.actions:
            if isinstance(action, list):
                raise InvalidInputError("CartPole actions must be scalar integers")
            if action not in (0, 1):
                raise InvalidInputError("CartPole actions must be 0 or 1")
            actions.append(action)
        return observation, np.asarray(actions, dtype=np.int32)
