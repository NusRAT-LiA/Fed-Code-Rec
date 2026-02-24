import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import flwr as fl
from flwr.common import FitRes, Status, Code
from fl_server.server import CustomFedAvg, ModelCheckpointer

@pytest.fixture
def mock_checkpointer(tmp_path):
    mock = MagicMock(spec=ModelCheckpointer)
    mock.model_path = tmp_path / "global_model.pth"
    return mock

def test_gradient_aggregation_and_saving(mock_checkpointer):
    """
    Scenario: Two clients send different weight updates.
    The server must average them and save a new .pth file.
    """
    weights_client_a = [np.ones((768, 768), dtype=np.float32) * 1.0, np.ones((768,), dtype=np.float32) * 1.0]
    weights_client_b = [np.ones((768, 768), dtype=np.float32) * 3.0, np.ones((768,), dtype=np.float32) * 3.0]
    from flwr.server.client_proxy import ClientProxy
    cp1, cp2 = MagicMock(spec=ClientProxy), MagicMock(spec=ClientProxy)
    
    def create_fit_res(weights):
        params = fl.common.ndarrays_to_parameters(weights)
        return FitRes(status=Status(code=Code.OK, message="S"), parameters=params, num_examples=100, metrics={})
    results = [(cp1, create_fit_res(weights_client_a)), (cp2, create_fit_res(weights_client_b))]

    
    def fake_save_action(parameters, round_num):
        mock_checkpointer.model_path.touch()

    mock_checkpointer.save_model.side_effect = fake_save_action

    strategy = CustomFedAvg(
        checkpointer=mock_checkpointer,
        min_fit_clients=2,
        min_available_clients=2
    )

    aggregated_params, _ = strategy.aggregate_fit(server_round=1, results=results, failures=[])

    assert aggregated_params is not None
   
    assert Path(mock_checkpointer.model_path).exists()