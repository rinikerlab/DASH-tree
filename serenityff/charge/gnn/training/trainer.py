import os
from typing import Callable, List, Optional, Sequence, Union
from warnings import warn

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import torch
from torch_geometric.data import DataLoader

from serenityff.charge.gnn.utils import (
    CustomData,
    get_graph_from_mol,
    mols_from_sdf,
    split_data_Kfold,
    split_data_random,
)


class Trainer:
    def __init__(
        self,
        device: Union[torch.device, Literal["cpu", "cuda"]] = "cpu",
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

    @property
    def train_data(self) -> torch.utils.data.Subset:
        return self._train_data

    @property
    def eval_data(self) -> torch.utils.data.Subset:
        return self._eval_data

    @property
    def save_prefix(self) -> str:
        return self._save_prefix

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
    def device(self, value: Union[torch.device, Literal["cpu", "cuda"]]):
        if isinstance(value, torch.device):
            self._device = value
            self._update_device()
            return
        elif isinstance(value, str):
            if value.lower() in ["cpu", "cuda"]:
                self._device = torch.device(value.lower())
                self._update_device()
                return
            else:
                raise ValueError("Device has to be 'cpu' or 'cuda'!")
        else:
            raise TypeError("device has to be of type str or torch.device")

    @data.setter
    def data(self, value: Sequence[CustomData]):
        self._data = value

    @train_data.setter
    def train_data(self, value: torch.utils.data.Subset) -> None:
        self._train_data = value
        return

    @eval_data.setter
    def eval_data(self, value: torch.utils.data.Subset) -> None:
        self._eval_data = value
        return

    @save_prefix.setter
    def save_prefix(self, value: str) -> None:
        dir = os.path.dirname(value)
        if os.path.isdir(dir):
            self._save_prefix = value
        else:
            os.makedirs(dir)
            self._save_prefix = value
        return

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

    def _random_split(self, train_ratio: Optional[float] = 0.8) -> None:
        self.train_data, self.eval_data = split_data_random(data_list=self.data, train_ratio=train_ratio)
        return

    def _kfold_split(
        self,
        n_splits: Optional[int] = 5,
        split: Optional[int] = 0,
    ) -> None:
        self.train_data, self.eval_data = split_data_Kfold(
            data_list=self.data,
            n_splits=n_splits,
            split=split,
        )
        return

    def prepare_training_data(
        self,
        split_type: Optional[Literal["random", "kfold"]] = "random",
        train_ratio: Optional[float] = 0.8,
        n_splits: Optional[int] = 5,
        split: Optional[int] = 0,
    ) -> None:
        try:
            self.data
        except AttributeError:
            warn("No data has been loaded to this trainer. Load Data firstt!")
            return
        if split_type.lower() == "random":
            self._random_split(train_ratio=train_ratio)
            return
        elif split_type.lower() == "kfold":
            self._kfold_split(n_splits=n_splits, split=split)
            return
        else:
            raise NotImplementedError(f"split_type {split_type} is not implemented yet.")

    def validate_model(self) -> List[float]:
        self.model.eval()
        on_gpu = self.device == torch.device("cuda")
        val_loss = []
        loader = DataLoader(self.eval_data, batch_size=32)
        for data in loader:
            data.to(self.device)
            prediction = self.model(
                data.x,
                data.edge_index,
                data.batch,
                data.edge_attr,
                data.molecule_charge,
            )
            loss = self.loss_function(prediction, data.y)
            val_loss.append(np.mean(loss.to("cpu").tolist()))
            del data, prediction, loss
            if on_gpu:
                torch.cuda.empty_cache()
        return np.mean(val_loss)

    def _save_training_data(self, loss: Sequence[float], eval_loss: Sequence[float], batch_size: int) -> None:
        np.save(arr=loss, file=f"{self.save_prefix}_train_loss.dat")
        np.save(arr=eval_loss, file=f"{self.save_prefix}_eval_loss.dat")

    def train_model(
        self,
        epochs: int,
        batch_size: Optional[int] = 64,
    ):
        try:
            self.train_data
            self.eval_data
            self.optimizer
            self.loss_function
        except AttributeError as e:
            raise e(
                "Make sure, train data has been prepared and that an optimizer\
                     and a loss_function have been set in this instance!"
            )
        on_gpu = self.device == torch.device("cuda")
        losses = []
        eval_losses = []

        for _ in range(epochs):
            self.model.train()
            train_loss = []
            loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)

            for data in loader:
                self.optimizer.zero_grad()
                data.to(self.device)
                prediction = self.model(
                    data.x,
                    data.edge_index,
                    data.batch,
                    data.edge_attr,
                    data.molecule_charge,
                )
                loss = self.loss_function(prediction, data.y)
                loss.backward()
                self.optimizer.step()

                losses.append(np.mean(loss.to("cpu").tolist()))
                train_loss.append(np.mean(loss.to("cpu").tolist()))
                if on_gpu:
                    torch.cuda.empty_cache()
            eval_losses.append(self.validate_model())

        self._save_training_data(losses, eval_losses, batch_size)
        return losses, eval_losses
