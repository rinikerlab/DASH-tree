from typing import Callable, List, Optional, Sequence

import torch

from serenityff.charge.gnn.utils import CustomData, get_graph_from_mol, mols_from_sdf


class Trainer:
    def __init__(
        self,
        device: Optional[str] = "cpu",
    ) -> None:
        self.device = device

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    def loss_function(self) -> Callable:
        return self._loss_function

    @property
    def data(self) -> Sequence[CustomData]:
        return self._data

    @model.setter
    def model(self, value: torch.nn.Module) -> None:
        if isinstance(value, torch.nn.Module):
            self._model = value
            self._model.to(self._device)
            return
        else:
            raise TypeError("model has to be a subclass of torch.nn.Module!")

    @optimizer.setter
    def optimizer(self, value: torch.optim.Optimizer) -> None:
        if isinstance(value, torch.optim.Optimizer):
            self._optimizer = value
            return
        else:
            raise TypeError("Optimizer has to be a subclass of torch.optim.Optimizer")

    @loss_function.setter
    def loss_function(self, value: Callable) -> None:
        if isinstance(value, Callable):
            self._loss_function = value
            return
        else:
            raise TypeError("loss_function has to be of type callable")

    @device.setter
    def device(self, value: torch.device):
        if isinstance(value, torch.device):
            self._device = value
            self._update_device()
            return
        elif isinstance(value, str):
            if value in ["cpu", "cuda"]:
                self._device = torch.device(value)
                self._update_device()
                return
            else:
                raise ValueError("Device has to be 'cpu' or 'cuda'!")
        else:
            raise TypeError("device has to be of type str or torch.device")

    @data.setter
    def data(self, value: Sequence[CustomData]):
        self._data = value

    def _update_device(self) -> None:
        """
        Moves model and data to the device specified in self.device.
        """
        try:
            self.model.to(self.device)
        except AttributeError:
            pass
        return

    def gen_graphs_from_sdf(
        self,
        sdf_file: str,
        allowable_set: Optional[List[int]] = [
            "C",
            "N",
            "O",
            "F",
            "P",
            "S",
            "Cl",
            "Br",
            "I",
            "H",
        ],
    ) -> None:
        mols = mols_from_sdf(sdf_file)
        self.data = [get_graph_from_mol(mol, allowable_set) for mol in mols]
        return

    def load_graphs_from_pt(self, pt_file: str) -> None:
        self.data = torch.load(pt_file)
        return
