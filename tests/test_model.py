from my_project.model import MyAwesomeModel
import torch
import pytest

@pytest.mark.parametrize("batch_size", [32, 64])
def test_model(batch_size: int) -> None:

    # define the model
    model = MyAwesomeModel()
    # create arbitrary data with correct shape
    X = torch.randn(batch_size,1,28,28)

    # forward the created data
    output = model(X)

    # assert that the shape of the data is correct
    assert output.shape == (batch_size,10)

    