from typing import List, Optional, Sequence

import torch
from rdkit import Chem

from serenityff.charge.gnn.utils import CustomData, MolGraphConvFeaturizer
from serenityff.charge.utils import Molecule


def mols_from_sdf(sdf_file: str, removeHs: Optional[bool] = False) -> Sequence[Molecule]:
    """
    Returns a Sequence of rdkit molecules read in from a .sdf file.

    Args:
        sdf_file (str): path to .sdf file.
        removeHs (Optional[bool], optional): Wheter to remove Hydrogens. Defaults to False.

    Returns:
        Sequence[Molecule]: rdkit mols.
    """
    return Chem.SDMolSupplier(sdf_file, removeHs=removeHs)


def get_graph_from_mol(
    mol: Molecule,
    index: int,
    sdf_property_name: Optional[str] = None,
    allowable_set: Optional[List[str]] = [
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
    no_y: Optional[bool] = False,
) -> CustomData:
    """
    Creates an pytorch_geometric Graph from an rdkit molecule.
    The graph contains following features:
        > Node Features:
            > Atom Type (as specified in allowable set)
            > formal_charge
            > hybridization
            > H acceptor_donor
            > aromaticity
            > degree
        > Edge Features:
            > Bond type
            > is in ring
            > is conjugated
            > stereo
    Args:
        mol (Molecule): rdkit molecule
        sdf_property_name (Optional[str]): Name of the property in the sdf file to be used for training.
        allowable_set (Optional[List[str]], optional): List of atoms to be \
            included in the feature vector. Defaults to \
                [ "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "H", ].

    Returns:
        CustomData: pytorch geometric Data with .smiles as an extra attribute.
    """
    grapher = MolGraphConvFeaturizer(use_edges=True)
    graph = grapher._featurize(mol, allowable_set).to_pyg_graph()
    if no_y:
        graph.y = torch.tensor(
            [0 for _ in mol.GetAtoms()],
            dtype=torch.float,
        )
    else:
        assert (
            sdf_property_name is not None
        ), "'sdf_property_name' can only be None in case you selected 'no_y'. Please provide sdf_property_name."
        value = mol.GetProp(sdf_property_name)
        graph.y = torch.tensor(list(float(x) for x in value.split("|")), dtype=torch.float)
        try:
            assert not torch.isnan(graph.y).any(), f"y found in graph {str(graph)} for molecule with value {value}."
        except AssertionError as exc:
            print(exc)
            print("this molecule is skipped")
            return None
    # TODO: Check if batch is needed, otherwise this could lead to a problem if all batches are set to 0
    # Batch will be overwritten by the DataLoader class
    graph.batch = torch.tensor([0 for _ in mol.GetAtoms()], dtype=int)
    graph.molecule_charge = Chem.GetFormalCharge(mol)
    graph.smiles = Chem.MolToSmiles(mol, canonical=True)
    graph.sdf_idx = index
    return graph
