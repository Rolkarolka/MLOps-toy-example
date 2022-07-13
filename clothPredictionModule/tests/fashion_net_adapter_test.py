import pytest

from clothPredictionModule import FashionNetAdapter
from torchvision import datasets

@pytest.fixture
def model():
    return FashionNetAdapter()

def test_model_predict(model):
    image = datasets.FashionMNIST('./data', download=True, train=False)[3]
    prediction = model.predict(image)
    assert type(prediction) is int
