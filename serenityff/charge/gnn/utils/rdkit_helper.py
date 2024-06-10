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
    sdf_property_name: Optional[str] = "MBIScharge",
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
) -> Optional[CustomData]:
    """
    Creates an pytorch_geometric Graph from an rdkit molecule.

    Returns None if the property is not found or contains NaN.
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

    def get_mol_prop_as_torch_tensor(prop_name: Optional[str], mol: Molecule) -> torch.Tensor:
        if prop_name is None:
            raise ValueError("Property name can not be None when no_y == False.")
        if not mol.HasProp(prop_name):
            raise ValueError(f"Property {prop_name} not found in molecule.")
        tensor = torch.tensor([float(x) for x in mol.GetProp(prop_name).split("|")], dtype=torch.float)
        if torch.isnan(tensor).any():
            raise TypeError(f"Nan found in {prop_name}.")
        return tensor

    grapher = MolGraphConvFeaturizer(use_edges=True)
    graph = grapher._featurize(mol, allowable_set).to_pyg_graph()
    if no_y:
        graph.y = torch.tensor(
            [0 for _ in mol.GetAtoms()],
            dtype=torch.float,
        )
    else:
        try:
            graph.y = get_mol_prop_as_torch_tensor(sdf_property_name, mol)
        except TypeError as exc:
            print(exc)
            return None

    graph.batch = torch.tensor([0 for _ in mol.GetAtoms()], dtype=int)
    graph.molecule_charge = Chem.GetFormalCharge(mol)
    graph.smiles = Chem.MolToSmiles(mol, canonical=True)
    graph.sdf_idx = index
    return graph
