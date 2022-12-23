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
        allowable_set (Optional[List[str]], optional): List of atoms to be \
            included in the feature vector. Defaults to \
                [ "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "H", ].

    Returns:
        CustomData: pytorch geometric Data with .smiles as an extra attribute.
    """
    grapher = MolGraphConvFeaturizer(use_edges=True)
    graph = grapher._featurize(mol, allowable_set).to_pyg_graph()
    if not no_y:
        graph.y = torch.tensor(
            [float(x) for x in mol.GetProp('MBIScharge').split("|")],
            dtype=torch.float,
        )
    else:
        graph.y = torch.tensor(
            [0 for _ in mol.GetAtoms()],
            dtype=torch.float,
        )
    graph.batch = torch.tensor([0 for _ in mol.GetAtoms()], dtype=int)
    graph.molecule_charge = Chem.GetFormalCharge(mol)
    graph.smiles = Chem.MolToSmiles(mol, canonical=True)
    graph.sdf_idx = index
    return graph
