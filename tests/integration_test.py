import sys
sys.path.append(".")
from Orchestrator import Orchestrator

from clothPredictionModule import FashionNet, FashionNetAdapter
import pytest
import torch

@pytest.fixture
def orchestrator():
    return Orchestrator()


def test_run_workflow(orchestrator, monkeypatch):
    img_num = "3"
    def prepare_model_mock():
        return FashionNet()

    class Prediction():
        def __init__(self):
            self.data = torch.Tensor([[ -3.2093,  -2.1909, -12.9942,  -7.2977, -13.6355, -30.4276,  -3.6275, -4.0288,  -0.2310,  -4.7135]])

    monkeypatch.setattr(FashionNetAdapter, 'prepare_model', prepare_model_mock())
    monkeypatch.setattr(FashionNet, '__call__', lambda _, __: Prediction())
    
    prediction = orchestrator.run(img_num)
    assert type(prediction) is int

