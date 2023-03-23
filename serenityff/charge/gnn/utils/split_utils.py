from math import ceil
from typing import List, Optional, Sequence, Tuple

import torch
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupShuffleSplit

from .custom_data import CustomData


def get_split_numbers(N: int, train_ratio: Optional[float] = 0.8) -> List[float]:
    """
    Basic function determining absolute split values for a dataset containing N datapoints.

    Args:
        N (int): size of the data set.
        train_ratio (int, optional): how much of the data set is train set between 0 and 1. Defaults to .8.

    Returns:
        List[float]: numbers to split the data set.
    """
    train = ceil(N * train_ratio)
    test = N - train
    return [train, test]


def split_data_random(
    data_list: List[CustomData],
    train_ratio: Optional[float] = 0.8,
    seed: Optional[int] = 13,
) -> Tuple[torch.utils.data.Subset]:
    """
    Splits a List of CustomData in two Subsets, having a ration of train_ratio.
    A seed for the randomnumber generator can be specified.

    Args:
        data_list (List[CustomData]): List of Data objects
        train_ratio (Optional[float], optional): ratio of traindata over testdata. Defaults to .8.
        seed (Optional[int], optional): Seed for random number generator. Defaults to 13.

    Returns:
        Tuple[Subset]: train and test subset.
    """
    N_tot = len(data_list)
    numbers = get_split_numbers(N=N_tot, train_ratio=train_ratio)
    randn_generator = torch.Generator().manual_seed(seed)
    train_data, test_data = torch.utils.data.random_split(
        dataset=data_list,
        lengths=numbers,
        generator=randn_generator,
    )
    return train_data, test_data


def split_data_smiles(
    data_list: List[CustomData],
    train_ratio: Optional[float] = 0.8,
    seed: Optional[int] = 13,
) -> Tuple[torch.utils.data.Subset]:
    """
    Splits a List of CustomData in two Subsets, having a ration of train_ratio.
    A seed for the randomnumber generator can be specified.
    Difference to random split: Conformers of same molecule are not split between the sets

    Args:
        data_list (List[CustomData]): List of Data objects
        train_ratio (Optional[float], optional): ratio of traindata over testdata. Defaults to .8.
        seed (Optional[int], optional): Seed for random number generator. Defaults to 13.

    Returns:
        Tuple[Subset]: train and test subset.
    """
    smiles_list = []
    train_data = []
    test_data = []
    for mol in data_list:
        smiles_list.append(mol["smiles"])
    gss = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
    for train_idx, test_idx in gss.split(data_list, groups=smiles_list):
        train_data = [data_list[i] for i in train_idx]
        test_data = [data_list[i] for i in test_idx]
    train_data = torch.utils.data.Subset(train_data, indices=list(range(0, len(train_data))))
    test_data = torch.utils.data.Subset(test_data, indices=list(range(0, len(test_data))))
    return train_data, test_data


def split_data_Kfold(data_list: Sequence[CustomData], n_splits: int = 5, split: int = 0, seed: int = 161311) -> Tuple[Sequence[CustomData]]:
    """
        Performs a kfold split on a List of CustomData objects,
        returning a list contianing the training, and a list containing
        the test set.

    Args:
        data_list (Sequence[CustomData]): List to split.
        n_splits (int): Numbrer of splits
        split (int): _description_

    Returns:
        Tuple[Sequence[CustomData]]: Two Lists containing train and test data.
    """
    kfold_split = KFold(n_splits, shuffle=True, random_state=seed)
    train_idx_list, val_idx_list = [], []
    for train_idx, val_idx in kfold_split.split(data_list):
        train_idx_list.append(train_idx)
        val_idx_list.append(val_idx)
    test_data = [data_list[molecule_idx] for molecule_idx in val_idx_list[split]]
    train_data = [data_list[molecule_idx] for molecule_idx in train_idx_list[split]]

    return train_data, test_data
