from src.model import Model
import pytest
import torch


def test_error_on_wrong_shape():
    model = Model(784, 128, 64, 10)
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        model(torch.randn(1,2,3))
      