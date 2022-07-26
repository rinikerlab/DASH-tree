import torch
from typing import List, Sequence, Tuple, Optional
from .custom_data import CustomData
from sklearn.model_selection import KFold, GroupShuffleSplit
from rdkit import Chem
import numpy as np
from random import shuffle
import pandas as pd


def get_split_numbers(N: int, train_ratio: Optional[float] = 0.8) -> List[float]:
    """
    Basic function determining absolute split values for a dataset containing N datapoints.

    Args:
        N (int): size of the data set.
        train_ratio (int, optional): how much of the data set is train set between 0 and 1. Defaults to .8.

    Returns:
        List[float]: numbers to split the data set.
    """
    train = int(N * train_ratio)
    test = int(N - train)
    while train + test != N:
        if train + test > N:
            test -= 1
        else:
            train += 1
    return [train, test]


def split_data_random(
    data_list: List[CustomData], train_ratio: Optional[float] = 0.8, seed: Optional[int] = 13
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


def split_data_Kfold(data_list: Sequence[CustomData], n_splits: int, split: int) -> Tuple[Sequence[CustomData]]:
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
    kfold_split = KFold(n_splits, shuffle=True, random_state=5)
    train_idx_list, val_idx_list = [], []
    for split, (train_idx, val_idx) in enumerate(kfold_split.split(data_list)):
        train_idx_list.append(train_idx)
        val_idx_list.append(val_idx)
    val_idx_split = val_idx_list[split]
    train_idx_split = train_idx_list[split]
    test_data = [data_list[molecule_idx] for molecule_idx in val_idx_split]
    train_data = [data_list[molecule_idx] for molecule_idx in train_idx_split]

    return train_data, test_data


def split_data_scaffold(data_list: Sequence[CustomData], n_splits: int, split: int) -> Tuple[Sequence[CustomData]]:
    """
    Creates A Train and Test set using the scaffold split.

    Args:
        data_list (Sequence[CustomData]): List to split
        n_splits (int): Number of Splits.
        split (int): Which split to get.

    Returns:
        Tuple[Sequence[CustomData]]: Two lists containing train and test set.
    """
    gss = GroupShuffleSplit(n_splits=n_splits, train_size=0.8, random_state=42)
    train_idx_list, val_idx_list = [], []
    smiles = [graph.smiles for graph in data_list]
    scaffold_smiles = [
        Chem.MolToSmiles(
            Chem.Scaffolds.MurckoScaffold.MakeScaffoldGeneric(
                Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(smile))
            )
        )
        for smile in smiles
    ]

    groups = np.array([np.unique(scaffold_smiles).to_list().index(smile) for smile in scaffold_smiles])
    graph_df = pd.DataFrame(data_list)
    X_data = graph_df.iloc[:, [0, 1, 2, 4, 5, 6]]
    y_data = graph_df.iloc[:, 3]
    for split, (train_idx, val_idx) in enumerate(gss.split(X_data, y_data, groups)):
        train_idx_list.append(train_idx)
        val_idx_list.append(val_idx)
    test_data = [data_list[molecule_idx] for molecule_idx in val_idx_list[split]]
    train_data = [data_list[molecule_idx] for molecule_idx in train_idx_list[split]]

    return train_data, test_data


def split_data_fragment(data_list, n_splits, split, k, N):

    x_data_all = [g_data.x for g_data in data_list]
    smiles_data = [g_data.smiles for g_data in data_list]
    # here we start with the scaffolding
    scaf_smiles = []  # the smiles of the scaffold of all the data
    mol_data = []  # the moles all the data
    scaf_mol = []  # the moles of all the scaffolds
    for i in smiles_data:
        i = Chem.MolFromSmiles(i)
        mol_data.append(i)
        scaf_smiles.append(Chem.MolToSmiles(Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(i)))
        scaf_mol.append(Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(i))

    #  get random list
    def random_list(sub, K):
        while True:
            shuffle(sub)
            yield sub[:K]

    # initializing K, N. Looks for biggest common substructure of K molecules and does this N times
    K = 5
    N = len(x_data_all)  # TODO play around with these parameters

    scaffold_sample = (
        []
    )  # here we randomly sample moles from all the scaffold. you could also sample moles from all the moles!
    # getting N random elements
    for idx in range(0, N):
        scaffold_sample.append(next(random_list(scaf_mol, K)))

    MCS = []  # these are the most common substures or fragments in MCSresults
    for i in range(0, N):
        MCS.append(Chem.rdFMCS.FindMCS(scaffold_sample[i]))

    # the issue with this function is that it creates a special datatype which can not directly be changed to mol or smiles.
    # Has to go via SMARTS
    fraction_smiles = []  # the smiles from the fragments
    for i in range(0, N):
        a = MCS[i].smartsString  # here we take the smarts
        b = Chem.MolFromSmarts(a)  # to moles
        fraction_smiles.append(Chem.MolToSmiles(b))  # and back to smiles

    fraction_smiles_unique = list(dict.fromkeys(fraction_smiles))  # deleting the duplicates from smiles
    fraction_moles_unique = []
    for smiles in fraction_smiles_unique:
        fraction_moles_unique.append(Chem.MolFromSmiles(smiles))  # unique moles

    # here all molecules are screened on all fragments and matches are stored in nested lists.
    # This  list contains duplicates of the molecules in the dataset as one molecule can have multiple fragments as substructure
    matches = []
    error_match = []
    for i in range(0, len(fraction_moles_unique)):
        try:
            matches.append([x for x in mol_data if x.HasSubstructMatch(fraction_moles_unique[i])])
        except:
            print("error")
            error_match.append(i)

    matches.sort(key=len)  # sort by length: the fragments with the least amount matches in the dataset come first

    matches_smiles_per_fragment = []  # this is needed to maintain the datastructure
    matches_smiles = (
        []
    )  # here we make a nested list, with every sublist representing a fragment, containing all the molecules that contain the fragment
    for i in range(0, len(matches)):
        for x in range(0, len(matches[i])):
            matches_smiles_per_fragment.append(Chem.MolToSmiles(matches[i][x]))
        matches_smiles.append(matches_smiles_per_fragment)
        matches_smiles_per_fragment = []  # empty the list in between runs

    # I don't understand exaclty how it works but this whole chunk of code ensures that all the molecules occur only once in the list
    visited_frag = []
    for lst in matches_smiles:
        for i in range(len(lst) - 1, -1, -1):
            if lst[i] not in visited_frag:
                visited_frag.append(lst[i])
            else:
                lst.pop(i)
    fragment_full2 = [
        ele for ele in matches_smiles if ele != []
    ]  # This ensures that the list contains all the molecules from the dataset once
    # empty lists are deleted

    fragment_indices = [*range(0, len(fragment_full2))]  # getting indices
    # making a dictionary with indices of the fragments and the smiles of the molecules linked to this fragment
    fra_smi = dict(zip(fragment_indices, fragment_full2))
    smiles_indices = [*range(0, len(smiles_data))]  # getting indices

    # Because there was an issue with two molecules given a different smiles notation in the original set from the one generated from moles later
    # Here we convert all smiles to moles and back to smiles
    smiles_data2 = []
    for smiles in smiles_data:
        moles = Chem.MolFromSmiles(smiles)
        smiles_data2.append(Chem.MolToSmiles(moles))
        moles = []
    # making a dictionary with indices and smiles of all the molecules in the dataset
    smi_id = dict(zip(smiles_indices, smiles_data2))

    # making list with indeces of the fragment connected to the molecule in the dataset
    ind_ids = []
    smiles_grouped = []
    for index in smi_id:
        for frag_id in fra_smi.keys():  # frag_id = number of the fragment
            for smi in fra_smi[frag_id]:  # smi = smiles of the molecule
                if smi == smi_id[index]:
                    ind_ids.append(frag_id)
                    smiles_grouped.append(smi)

    ############################################################
    ################### insert in code Niels ###################
    ############################################################

    groups = np.array(ind_ids)
    graph_df = pd.DataFrame(data_list)
    X_data = graph_df.iloc[:, [0, 1, 2, 4, 5]]
    y_data = graph_df.iloc[:, 3]
    train_idx_list, val_idx_list = [], []
    gss = GroupShuffleSplit(
        n_splits=n_splits, train_size=0.8, random_state=42
    )  # TODO play around with these parameters
    for split, (train_idx, val_idx) in enumerate(gss.split(X_data, y_data, groups)):
        train_idx_list.append(train_idx)
        val_idx_list.append(val_idx)
    val_idx_split = val_idx_list[split]
    train_idx_split = train_idx_list[split]
    test_data = [data_list[molecule_idx] for molecule_idx in val_idx_split]
    train_data = [data_list[molecule_idx] for molecule_idx in train_idx_split]

    return train_data, test_data
