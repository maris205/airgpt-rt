import pytest

from airgpt_service.adapters.state.dreamer_cartpole import DreamerCartPoleAdapter
from airgpt_service.errors import InvalidInputError
from airgpt_service.schemas import StatePredictRequest


def test_dreamer_cartpole_adapter_validates_task_contract():
    adapter = DreamerCartPoleAdapter()
    request = StatePredictRequest(
        model="dreamer_cartpole",
        observation=[0.1, 0.2, 0.3, 0.4],
        actions=[0, 1, 0],
    )
    observation, actions = adapter.preprocess(request)
    assert observation.shape == (4,)
    assert actions.tolist() == [0, 1, 0]

    with pytest.raises(InvalidInputError):
        adapter.preprocess(request.model_copy(update={"actions": [2]}))
