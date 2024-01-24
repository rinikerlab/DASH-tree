# from rdkit import Chem
from rdkit.Chem.TorsionFingerprints import CalculateTorsionLists
from rdkit.Chem.rdMolTransforms import GetDihedralDeg
from serenityff.charge.utils import Molecule
from typing import List  # , Optional, Sequence


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
    return GetDihedralDeg(conf, atom_i, atom_j, atom_k, atom_l) / 180


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
