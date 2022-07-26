from typing import Optional, Sequence

import torch

from serenityff.charge.gnn.utils import CustomData


class Trainer:
    def __init__(self) -> None:
        pass

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    def data(self) -> Sequence[CustomData]:
        return self._data

    @property
    def train_data(self) -> Sequence[CustomData]:
        return self._train_data

    @property
    def test_data(self) -> Sequence[CustomData]:
        return self._test_data

    @property
    def eval_data(self) -> Sequence[CustomData]:
        return self._eval_data

    def train_model(
        self,
        epochs,
        batchsize: Optional[int] = 64,
        device: Optional[torch.device] = torch.device("cpu"),
    ):
        assert self.train_data
        assert self.model
        assert self.optimizer

        self.model.train()

        for _ in range(epochs):
            pass
