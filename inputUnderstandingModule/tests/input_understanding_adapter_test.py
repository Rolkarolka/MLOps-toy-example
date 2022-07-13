import pytest

from inputUnderstandingModule import InputUnderstandingAdapter
from torchvision import datasets

@pytest.fixture
def input_understanding():
    return InputUnderstandingAdapter()

def test_prepare_input(input_understanding):
    processed_input = input_understanding.process("3")
    assert type(processed_input) is tuple
