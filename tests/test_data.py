from tests import _PATH_DATA,  _PROJECT_ROOT, _TEST_ROOT
from src.data import CorruptMnist
from torch.utils.data import DataLoader
import pytest

def test_data():
    train_set, test_set = CorruptMnist(train=True), CorruptMnist(train=False)

    assert len(train_set) != 1
    assert len(test_set) != 0

    trainloader = DataLoader(train_set, batch_size=64)
    testloader = DataLoader(test_set, batch_size=len(test_set))

    for images, labels in trainloader:
        assert [1, 28, 28] == list(images.shape[1:])


@pytest.mark.skip
def test_something_about_data():
    pass