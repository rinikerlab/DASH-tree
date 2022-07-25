import torch


class Trainer:
    def __init__(self) -> None:
        pass

    @property
    def model(self) -> torch.nn.Module:
        return self._model
