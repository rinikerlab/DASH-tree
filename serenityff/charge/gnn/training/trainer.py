import contextlib
import os
from typing import Callable, List, Optional, OrderedDict, Sequence, Tuple, Union
from warnings import warn

from tqdm import tqdm


try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from serenityff.charge.gnn.utils import (
    ChargeCorrectedNodeWiseAttentiveFP,
    NodeWiseAttentiveFP,
    CustomData,
    get_graph_from_mol,
    mols_from_sdf,
    split_data_Kfold,
    split_data_random,
    split_data_smiles,
)
from serenityff.charge.utils import Molecule, NotInitializedError


class Trainer:
    """
    Trainer class for the GNN. Holds the model, optimizer, loss function,
    data, and training and evaluation data.

    Can be used on CPU or GPU.

    Offers convinient parsing of molecule data to the GNN.
    """

    def __init__(
        self,
        device: Union[torch.device, Literal["cpu", "cuda", "available"]] = "available",
        loss_function: Optional[Callable] = torch.nn.functional.mse_loss,
        physicsInformed: Optional[bool] = True,
        seed: Optional[int] = 161311,
        verbose: bool = False,
    ) -> None:
        self.device = device
        self.loss_function = loss_function
        self.physicsInformed = physicsInformed
        self.seed = seed
        self.verbose = verbose

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
    def physicsInformed(self) -> bool:
        return self._physicsInformed

    @property
    def seed(self) -> int:
        return self._seed

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

    @property
    def verbose(self) -> bool:
        return self._verbose

    @property
    def _on_gpu(self) -> bool:
        return self.device == torch.device("cuda")

    @model.setter
    def model(self, value: Union[str, torch.nn.Module]) -> None:
        if isinstance(value, str):
            try:
                load = torch.load(value, map_location=torch.device("cpu"))
            except FileNotFoundError as e:
                raise e
            try:
                load.state_dict()
                self._model = value
            except AttributeError:
                if self.physicsInformed:
                    self._model = ChargeCorrectedNodeWiseAttentiveFP()
                else:
                    self._model = NodeWiseAttentiveFP()
                self._model.load_state_dict(load)

        elif isinstance(value, torch.nn.Module):
            self._model = value
        elif isinstance(value, OrderedDict):
            if self.physicsInformed:
                self._model = ChargeCorrectedNodeWiseAttentiveFP()
            else:
                self._model = NodeWiseAttentiveFP()
            self._model.load_state_dict(value)
        else:
            raise TypeError(
                "model has to be either of type torch.nn.Module, OrderedDict, \
                    or the str path to a .pt model holding either of the aforementioned types."
            )
        self._update_device()
        return

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

    @physicsInformed.setter
    def physicsInformed(self, value: bool) -> None:
        if isinstance(value, bool):
            self._physicsInformed = value
            return
        else:
            raise TypeError("physicsInformed has to be of type bool")

    @seed.setter
    def seed(self, value: int) -> None:
        if isinstance(value, int):
            self._seed = value
            return
        else:
            raise TypeError("seed has to be of type int")

    @device.setter
    def device(self, value: Union[torch.device, Literal["cpu", "cuda", "available"]]):
        if isinstance(value, torch.device):
            self._device = value
            self._update_device()
            return
        elif isinstance(value, str):
            # I think that would be a nice conveniance option
            if value == "available":
                if torch.cuda.is_available():
                    value = "cuda"
                else:
                    value = "cpu"
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

    @verbose.setter
    def verbose(self, value: bool) -> None:
        self._verbose = value

    def _update_device(self) -> None:
        """
        Moves model to the device specified in self.device.
        """
        with contextlib.suppress(AttributeError):
            self.model.to(self.device)

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
        verbose: bool = verbose,
        sdf_property_name: str = "MBIScharge",
    ) -> None:
        """
        Creates pytorch geometric graphs using the custom featurizer for all molecules in a sdf file. 'MolFileAlias' in the sdf is taken
        as the ground truth value, generate your input sdf file accordingly.

        Args:
            sdf_file (str): path to .sdf file holding the molecules.
            allowable_set (Optional[List[int]], optional): Allowable atom types. Defaults to [ "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "H", ].
        """
        mols = mols_from_sdf(sdf_file)
        self.data = [
            get_graph_from_mol(
                mol=mol,
                index=index,
                sdf_property_name=sdf_property_name,
                allowable_set=allowable_set,
            )
            for index, mol in tqdm(
                enumerate(mols),
                disable=not verbose,
                desc="Creating molecular graphs from sdf file entries",
                total=len(mols),
            )
        ]
        self.data = [d for d in self.data if d is not None]
        return

    def load_graphs_from_pt(self, pt_file: str) -> None:
        """
        Loads pytorch geometric graphs from a .pt file.

        Args:
            pt_file (str): path to .pt file.
        """
        self.data = torch.load(pt_file)
        return

    def prepare_training_data(
        self,
        split_type: Optional[Literal["random", "kfold", "smiles"]] = "random",
        train_ratio: Optional[float] = 0.8,
        n_splits: Optional[int] = 5,
        split: Optional[int] = 0,
        seed: Optional[int] = None,
    ) -> None:
        """
        Splits training data into test data and eval data. At the moment, random, kfold and smiles split are implemented.

        Args:
            split_type (Optional[Literal[&quot;random&quot;, &quot;kfold&quot;]], optional): What split type you want. Defaults to "random".
            train_ratio (Optional[float], optional): ratio of train/eval in random split. Defaults to 0.8.
            n_splits (Optional[int], optional): number of splits in the kfold split. Defaults to 5.
            split (Optional[int], optional): which of the n_splits you want. Defaults to 0.
            seed (Optional[int], optional): random number seed for splits

        Raises:
            NotImplementedError: If a splittype other than 'random', 'kfold' or 'smiles' is chosen.
        """
        try:
            self.data
        except AttributeError:
            warn("No data has been loaded to this trainer. Load Data firstt!")
            return
        seed = self.seed if seed is None else seed
        if split_type.lower() == "random":
            self.train_data, self.eval_data = split_data_random(data_list=self.data, train_ratio=train_ratio, seed=seed)
            return
        if split_type.lower() == "kfold":
            self.train_data, self.eval_data = split_data_Kfold(
                data_list=self.data, n_splits=n_splits, split=split, seed=seed
            )
            return
        if split_type.lower() == "smiles":
            self.train_data, self.eval_data = split_data_smiles(data_list=self.data, train_ratio=train_ratio, seed=seed)
            return
        raise NotImplementedError(f"split_type {split_type} is not implemented yet.")

    def _save_training_data(
        self,
        loss: Sequence[float],
        eval_loss: Sequence[float],
    ) -> None:
        """
        Saves losses to numpy files.

        Args:
            loss (Sequence[float]): train loss.
            eval_loss (Sequence[float]): eval loss.
        """
        np.save(arr=loss, file=f"{self.save_prefix}_train_loss")
        np.save(arr=eval_loss, file=f"{self.save_prefix}_eval_loss")

    def _is_initialized(self) -> bool:
        """
        Checks if this instance of trainer has all attributes needed for it to train a model.

        Raises:
            NotInitializedError: Thrown if something is yet missing.

        Returns:
            bool: True if everything is initialized.
        """
        try:
            self.train_data
            self.eval_data
            self.optimizer
            self.loss_function
            return True
        except AttributeError:
            raise NotInitializedError(
                "Make sure, train data has been prepared and that an optimizer\
                     and a loss_function have been set in this instance!"
            )

    def validate_model(self, verbose: bool = verbose) -> List[float]:
        """
        predicts values for self.eval_data and returns the losses.

        Returns:
            List[float]: eval losses for self.eval_data.
        """
        try:
            self._is_initialized()
        except NotInitializedError as e:
            raise e
        self.model.eval()
        val_loss = []
        loader = DataLoader(self.eval_data, batch_size=64)
        for data in tqdm(loader, disable=not verbose, desc="Validating model", leave=False):
            data.to(self.device)
            prediction = self.model(
                data.x,
                data.edge_index,
                data.batch,
                data.edge_attr,
                data.molecule_charge,
                data.torch_indices,
            )
            loss = self.loss_function(torch.squeeze(prediction), data.y)
            val_loss.append(np.mean(loss.to("cpu").tolist()))
            del data, prediction, loss
            if self._on_gpu:
                torch.cuda.empty_cache()
        return np.mean(val_loss)

    def train_model(
        self,
        epochs: int,
        batch_size: Optional[int] = 64,
        verbose: Optional[bool] = verbose,
        save_after_every_step: bool = False,
    ) -> Tuple[Sequence[float]]:
        """
        Trains self.model if everything is initialized.

        Args:
            epochs (int): epochs to be trained.
            batch_size (Optional[int], optional): batchsize to be used in training. Defaults to 64.
        Raises:
            NotInitializedError: Raised in first two lines.

        Returns:
            Tuple[Sequence[float]]: train and eval losses.

        """
        try:
            self._is_initialized()
        except NotInitializedError as e:
            raise e
        train_loss = []
        eval_losses = []
        if verbose:
            print(f"Training model with {len(self.train_data)} molecules.", flush=True)
        for epo in tqdm(
            range(epochs),
            disable=not verbose,
            desc=f"Training for {epochs} epochs",
            leave=False,
        ):
            self.model.train()
            losses = []
            loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
            for data in tqdm(
                loader,
                disable=not verbose,
                desc=f"Training with {len(loader)} batches",
                leave=False,
            ):
                self.optimizer.zero_grad()
                data.to(self.device)
                data.validate()
                prediction = self.model(
                    data.x,
                    data.edge_index,
                    data.batch,
                    data.edge_attr,
                    data.molecule_charge,
                    data.torch_indices,
                )

                loss = self.loss_function(torch.squeeze(prediction), data.y)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.to("cpu").tolist())
                del data, prediction, loss
                if self._on_gpu:
                    torch.cuda.empty_cache()
            eval_losses.append(self.validate_model())
            train_loss.append(np.mean(losses))
            if save_after_every_step:
                self._save_training_data(train_loss, eval_losses)
                self.save_model_statedict()
                self.save_model()
            if tqdm.write:
                print(
                    f"Epoch: {epo}/{epochs} - Train Loss: {train_loss[-1]:.2E} - Eval Loss: {eval_losses[-1]:.2E}",
                    flush=True,
                )

        self._save_training_data(train_loss, eval_losses)
        self.save_model()
        self.save_model_statedict()
        return train_loss, eval_losses

    def predict(
        self,
        data: Union[Molecule, Sequence[Molecule], CustomData, Sequence[CustomData]],
        verbose: bool = verbose,
    ) -> Sequence[Sequence[float]]:
        """
        Predict values for graphs given in data using self.model.

        Args:
            data (Union[Molecule, Sequence[Molecule], CustomData, Sequence[CustomData]]): data to be predict values for.

        Raises:
            NotInitializedError: If self.model is not set yet.
            TypeError: If Input is neither a rdkit molecule or torch_geometric graph (or sequences of them)

        Returns:
            Sequence[Sequence[float]]: predictions made by self.model
        """
        try:
            self.model
        except AttributeError:
            raise NotInitializedError("load a model before predicting!")
        if not isinstance(data, list):
            data = [data]
        if isinstance(data[0], Molecule):
            graphs = [get_graph_from_mol(mol, index, no_y=False) for index, mol in enumerate(data)]
        elif isinstance(data[0], CustomData):
            graphs = data
        else:
            raise TypeError("Input has to be a Sequence or single rdkit molecule or a CustomData graph.")
        if verbose:
            print(f"Predicting values for {len(data)} molecules.")
        loader = DataLoader(graphs, batch_size=1, shuffle=False)
        predictions = []
        self.model.eval()
        for data in tqdm(loader, disable=not verbose, desc="Predicting values", leave=False):
            data.to(self.device)
            predictions.append(
                self.model(
                    data.x,
                    data.edge_index,
                    data.batch,
                    data.edge_attr,
                    data.molecule_charge,
                    data.torch_indices,
                )
                .to("cpu")
                .tolist()
            )
            del data
            if self._on_gpu:
                torch.cuda.empty_cache()
        return predictions

    def save_model_statedict(self, name: Optional[str] = "_model_sd.pt", verbose: bool = verbose) -> None:
        """
        Saves a models statedict to self.save_prefix + name

        Args:
            name (Optional[str], optional): name the model to be saved under. Defaults to "_model_sd.pt".
        """
        try:
            self.model
        except AttributeError:
            raise NotInitializedError("No model initialized, cannot save nothing ;^)")
        torch.save(self.model.state_dict(), f"{self.save_prefix}{name}")
        if verbose:
            print(f"Models statedict saved to {self.save_prefix}{name}")

    def save_model(self, name: Optional[str] = "_model.pt", verbose: bool = verbose) -> None:
        """
        Saves a models statedict to self.save_prefix + name

        Args:
            name (Optional[str], optional): name the model to be saved under. Defaults to "_model_sd.pt".
        """
        try:
            self.model
        except AttributeError:
            raise NotInitializedError("No model initialized, cannot save nothing ;^)")
        torch.save(self.model, f"{self.save_prefix}{name}")
        if verbose:
            print(f"Models statedict saved to {self.save_prefix}{name}")
