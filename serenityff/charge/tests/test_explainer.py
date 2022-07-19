from serenityff.charge.gnn.attention_extraction import Explainer
import pytest
import torch


@pytest.fixture
def model():
    return torch.load()


@pytest.fixture
def explainer(model):
    return Explainer(model=model, epochs=1, verbose=True)
