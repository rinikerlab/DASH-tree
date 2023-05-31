from typing import List, Optional, Sequence

import torch
from rdkit import Chem
from rdkit.Chem.TorsionFingerprints import CalculateTorsionLists
from rdkit.Chem.rdMolTransforms import GetDihedralDeg

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
            [float(x) for x in mol.GetProp("MBIScharge").split("|")],
            dtype=torch.float,
        )
    else:
        graph.y = torch.tensor(
            [0 for _ in mol.GetAtoms()],
            dtype=torch.float,
        )
    # TODO: Check if batch is needed, otherwise this could lead to a problem if all batches are set to 0
    # Batch will be overwritten by the DataLoader class
    graph.batch = torch.tensor([0 for _ in mol.GetAtoms()], dtype=int)
    graph.molecule_charge = Chem.GetFormalCharge(mol)
    graph.smiles = Chem.MolToSmiles(mol, canonical=True)
    graph.sdf_idx = index
    return graph


def get_list_of_torsion_indices_tuples(mol: Molecule) -> List[tuple]:
    """
    Returns a list of tuples containing the indices of the atoms involved in a torsion.
    The list is sorted by the atom index of the first atom in the torsion.
    Args:
        mol (Molecule): rdkit molecule

    Returns:
        List[tuple]: list of tuples containing the indices of the atoms involved in a torsion.
    """
    tmp_torsion_list = CalculateTorsionLists(mol)
    torsion_indices_list = []
    for propAndImprop in tmp_torsion_list:
        for torsion in propAndImprop:
            for indices in torsion[0]:
                torsion_indices_list.append(indices)
    return torsion_indices_list


def get_torsion_angle(mol: Molecule, atom_i: int, atom_j: int, atom_k: int, atom_l: int) -> float:
    """
    Returns the torsion angle (DEG) of the atoms i, j, k, l in the rdkit molecule mol.
    Args:
        mol (Molecule): rdkit molecule
        atom_i (int): index of atom i
        atom_j (int): index of atom j
        atom_k (int): index of atom k
        atom_l (int): index of atom l

    Returns:
        float: torsion angle in degrees
    """
    conf = mol.GetConformer()
    return GetDihedralDeg(conf, atom_i, atom_j, atom_k, atom_l)


def get_all_torsion_angles(mol: Molecule) -> List[List]:
    """
    Returns a list of lists containing the torsion indices and the torsion angle (DEG) of all torsions in the rdkit molecule mol.
    Args:
        mol (Molecule): rdkit molecule

    Returns:
        List[List]: list of lists containing the torsion indices and the torsion angle (DEG) of all torsions in the rdkit molecule mol.
    """
    torsion_indices_list = get_list_of_torsion_indices_tuples(mol)
    torsion_angles_list = []
    for torsion_indices in torsion_indices_list:
        torsion_angles_list.append([torsion_indices, get_torsion_angle(mol, *torsion_indices)])
    return torsion_angles_list


def get_number_of_torsions(mol):
    return len(get_list_of_torsion_indices_tuples(mol))


def get_torsion_graph_from_mol(
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
    grapher = MolGraphConvFeaturizer(use_edges=True)
    graph = grapher._featurize(mol, allowable_set).to_pyg_graph()
    if not no_y:
        torsions = get_all_torsion_angles(mol)
        graph.y = torch.tensor(
            [float(x[1]) for x in torsions],
            dtype=torch.float,
        )
    else:
        graph.y = torch.tensor(
            [0 for _ in range(get_number_of_torsions(mol))],
            dtype=torch.float,
        )
    graph.batch = torch.tensor([0 for _ in range(get_number_of_torsions(mol))], dtype=int)
    graph.molecule_charge = Chem.GetFormalCharge(mol)
    graph.smiles = Chem.MolToSmiles(mol, canonical=True)
    graph.sdf_idx = index
    return graph
