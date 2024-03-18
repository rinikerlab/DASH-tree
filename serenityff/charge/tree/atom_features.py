from typing import Any, Tuple
import numpy as np
from rdkit import Chem

from serenityff.charge.utils import Molecule


def is_atom_bonded_conjugate(atom: Chem.Atom) -> bool:
    for bond in atom.GetBonds():
        if bond.GetIsConjugated():
            return True
    return False


def get_connection_info_bond_type(molecule: Molecule, index: int, connected_to: int) -> int:
    bond = molecule.GetBondBetweenAtoms(int(index), int(connected_to))
    if bond is None:
        return -1
    elif bond.GetIsConjugated():
        return 4
    else:
        return int(bond.GetBondType())


class AtomFeatures:
    """
    AtomFeatures class is a mostly static class that contains all atom features included in the tree.
    A atom feature contains the following information:
        > Element type (e.g. C, N, O, ...)
        > Number of bonds (e.g. 1, 2, 3, ...)
        > Formal charge (e.g. -1, 0, 1, ...)
        > IsConjugated (True or False)
        > Number of Hydrogens (e.g. 0, 1, 2, ...) (includeNeighbors=True)

    It has a single atom form, or a form with the connection information like the relative index of the connected atom and the bond type.
    a atom with connection information can be called for example with the function atom_features_from_molecule_w_connection_info() and a
    example would be:
    [(6, 3, 0, False, 0), 0, 1]
    Wich is a carbon atom with 3 bonds, 0 formal charge, not aromatic and 0 hydrogens connected to an atom
    with index 0 via a single bond.
    """

    feature_list = (
        (5, 1, 0, False, 2),
        (5, 3, -1, True, 0),
        (5, 3, 0, False, 0),
        (5, 3, 0, False, 2),
        (5, 4, -1, False, 0),
        (5, 4, -1, False, 2),
        (35, 1, 0, False, 0),
        (6, 1, -1, False, 0),
        (6, 1, -1, True, 0),
        (6, 1, 0, False, 1),
        (6, 1, 0, False, 2),
        (6, 1, 0, False, 3),
        (6, 1, 0, True, 1),
        (6, 2, -1, True, 1),
        (6, 2, 0, False, 0),
        (6, 2, 0, False, 1),
        (6, 2, 0, False, 2),
        (6, 2, 0, True, 0),
        (6, 2, 0, True, 1),
        (6, 3, -1, False, 0),
        (6, 3, -1, True, 0),
        (6, 3, -1, False, 1),
        (6, 3, -1, True, 1),
        (6, 3, 0, False, 0),
        (6, 3, 0, False, 1),
        (6, 3, 0, False, 2),
        (6, 3, 0, True, 0),
        (6, 3, 0, True, 1),
        (6, 3, 0, True, 2),
        (6, 3, 1, False, 0),
        (6, 3, 1, True, 0),
        (6, 4, 0, False, 0),
        (6, 4, 0, False, 1),
        (6, 4, 0, False, 2),
        (6, 4, 0, False, 3),
        (17, 1, 0, False, 0),
        (9, 1, 0, False, 0),
        (1, 1, 0, False, 0),
        (53, 1, 0, False, 0),
        (53, 3, 0, False, 0),
        (53, 3, 0, False, 1),
        (53, 4, 0, False, 0),
        (7, 1, -1, False, 0),
        (7, 1, -1, True, 0),
        (7, 1, 0, False, 0),
        (7, 1, 0, True, 0),
        (7, 1, 0, True, 1),
        (7, 1, 0, True, 2),
        (7, 1, 1, False, 3),
        (7, 2, -1, False, 0),
        (7, 2, -1, False, 1),
        (7, 2, -1, True, 0),
        (7, 2, -1, True, 1),
        (7, 2, 0, False, 0),
        (7, 2, 0, False, 1),
        (7, 2, 0, True, 0),
        (7, 2, 0, True, 1),
        (7, 2, 1, False, 0),
        (7, 2, 1, False, 1),
        (7, 2, 1, False, 2),
        (7, 2, 1, True, 0),
        (7, 2, 1, True, 1),
        (7, 3, 0, False, 0),
        (7, 3, 0, False, 1),
        (7, 3, 0, False, 2),
        (7, 3, 0, True, 0),
        (7, 3, 0, True, 1),
        (7, 3, 0, True, 2),
        (7, 3, 1, False, 0),
        (7, 3, 1, False, 1),
        (7, 3, 1, True, 0),
        (7, 3, 1, True, 1),
        (7, 4, 1, False, 0),
        (7, 4, 1, False, 1),
        (7, 4, 1, False, 2),
        (7, 4, 1, False, 3),
        (8, 1, -1, False, 0),
        (8, 1, -1, True, 0),
        (8, 1, 0, False, 0),
        (8, 1, 0, False, 1),
        (8, 1, 0, True, 0),
        (8, 1, 0, True, 1),
        (8, 2, 0, False, 0),
        (8, 2, 0, False, 1),
        (8, 2, 0, False, 2),
        (8, 2, 0, True, 0),
        (8, 2, 0, True, 1),
        (8, 2, 0, True, 2),
        (8, 2, 1, False, 0),
        (8, 2, 1, False, 1),
        (8, 2, 1, True, 0),
        (8, 3, 1, False, 0),
        (8, 3, 1, False, 1),
        (15, 1, 0, False, 1),
        (15, 2, 0, False, 0),
        (15, 2, 0, False, 1),
        (15, 2, 0, True, 0),
        (15, 3, 0, False, 0),
        (15, 4, 0, False, 0),
        (15, 4, 0, False, 1),
        (15, 4, 0, True, 0),
        (15, 4, 0, True, 1),
        (15, 4, 1, False, 0),
        (15, 5, 0, False, 0),
        (15, 5, 0, False, 1),
        (16, 1, -1, False, 0),
        (16, 1, -1, True, 0),
        (16, 1, 0, False, 0),
        (16, 1, 0, True, 0),
        (16, 2, 0, False, 0),
        (16, 2, 0, True, 0),
        (16, 2, 0, False, 1),
        (16, 2, 1, True, 0),
        (16, 3, 0, False, 0),
        (16, 3, 0, True, 0),
        (16, 3, 1, True, 0),
        (16, 3, 1, False, 0),
        (16, 4, 0, False, 0),
        (16, 4, 0, False, 1),
        (16, 4, 0, True, 0),
        (16, 4, 1, False, 0),
        (16, 4, 1, True, 0),
    )
    afKey_2_afTuple = {k: v for k, v in enumerate(feature_list)}
    afTuple_2_afKey = {v: k for k, v in afKey_2_afTuple.items()}

    @staticmethod
    def get_number_of_features() -> int:
        return len(AtomFeatures.feature_list)

    @staticmethod
    def atom_features_from_molecule(molecule: Molecule, index: int) -> int:
        return AtomFeatures.afTuple_2_afKey[AtomFeatures.return_atom_feature_tuple_from_molecule(molecule, index)]

    @staticmethod
    def return_atom_feature_tuple_from_molecule(molecule: Molecule, index: int) -> Tuple:
        atom = molecule.GetAtomWithIdx(index)
        af_tuple = (
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            is_atom_bonded_conjugate(atom),
            atom.GetTotalNumHs(includeNeighbors=True),
        )
        return af_tuple

    @staticmethod
    def lookup_int(key: int) -> str:
        return AtomFeatures.afKey_2_afTuple[key]

    @staticmethod
    def lookup_tuple(key: Tuple) -> int:
        return AtomFeatures.afTuple_2_afKey[key]

    @staticmethod
    def string_to_tuple(key: str) -> Tuple:
        atom_type, num_bons, charge, isConjugated, numHs = key.split(" ")
        return (
            Chem.GetPeriodicTable().GetAtomicNumber(atom_type),
            int(num_bons),
            int(charge),
            isConjugated == "True",
            int(numHs),
        )

    @staticmethod
    def tuple_to_string(key: Tuple) -> str:
        return f"{Chem.GetPeriodicTable().GetElementSymbol(key[0])} {key[1]} {key[2]} {key[3]} {key[4]}"

    @staticmethod
    def lookup_sting(key: str) -> int:
        return AtomFeatures.afTuple_2_afKey[AtomFeatures.string_to_tuple(key)]

    @staticmethod
    def atom_features_from_molecule_w_connection_info(
        molecule: Molecule, index: int, connected_to: Tuple[Any] = (-1, -1)
    ) -> (int, int, int):
        connected_bond_type = (
            -1 if connected_to[1] == -1 else get_connection_info_bond_type(molecule, int(index), int(connected_to[1]))
        )
        key = AtomFeatures.atom_features_from_molecule(molecule, index)
        return (key, connected_to[0], connected_bond_type)

    @staticmethod
    def similarity(feature1: int, feature2: int) -> float:
        fp1 = AtomFeatures.lookup_int(feature1)
        fp2 = AtomFeatures.lookup_int(feature2)
        return np.sum([1 for i, j in zip(fp1, fp2) if i == j]) / len(fp1)

    @staticmethod
    def similarity_w_connection_info(feature1: int, feature2: int) -> float:
        fp1 = AtomFeatures.lookup_int(feature1[0])
        fp2 = AtomFeatures.lookup_int(feature2[0])
        ret_val = np.sum([1 for i, j in zip(fp1, fp2) if i == j])
        if feature1[1] == feature2[1]:
            ret_val += 1
        if feature1[2] == feature2[2]:
            ret_val += 1
        return ret_val / (len(fp1) + 2)
