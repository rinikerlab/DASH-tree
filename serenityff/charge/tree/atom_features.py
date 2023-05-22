from typing import Any, Sequence, Tuple, Union
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
        > Hybridization (e.g. SP, SP2, SP3, ...)
        > Aromaticity (True or False)
        > Number of Hydrogens (e.g. 0, 1, 2, ...) (includeNeighbors=True)

    It has a single atom form, or a form with the connection information like the relative index of the connected atom and the bond type.
    a atom with connection information can be called for example with the function atom_features_from_molecule_w_connection_info() and a
    example would be:
    ["C 3 0 SP2 False 0", 0, "SINGLE"]
    Wich is a carbon atom with 3 bonds, 0 formal charge, SP2 hybridization, not aromatic and 0 hydrogens connected to an atom
    with index 0 via a single bond.
    """

    feature_list = [
        "B 1 0 False 2",
        "B 3 -1 True 0",
        "B 3 0 False 0",
        "B 3 0 False 2",
        "B 4 -1 False 0",
        "B 4 -1 False 2",
        "Br 1 0 False 0",
        "C 1 -1 False 0",
        "C 1 -1 True 0",
        "C 1 0 False 1",
        "C 1 0 False 2",
        "C 1 0 False 3",
        "C 1 0 True 1",
        "C 2 -1 True 1",
        "C 2 0 False 0",
        "C 2 0 False 1",
        "C 2 0 False 2",
        "C 2 0 True 0",
        "C 2 0 True 1",
        "C 3 -1 False 0",
        "C 3 -1 True 0",
        "C 3 -1 False 1",
        "C 3 -1 True 1",
        "C 3 0 False 0",
        "C 3 0 False 1",
        "C 3 0 False 2",
        "C 3 0 True 0",
        "C 3 0 True 1",
        "C 3 0 True 2",
        "C 3 1 False 0",
        "C 3 1 True 0",
        "C 4 0 False 0",
        "C 4 0 False 1",
        "C 4 0 False 2",
        "C 4 0 False 3",
        "Cl 1 0 False 0",
        "F 1 0 False 0",
        "H 1 0 False 0",
        "I 1 0 False 0",
        "I 3 0 False 0",
        "I 3 0 False 1",
        "I 4 0 False 0",
        "N 1 -1 False 0",
        "N 1 -1 True 0",
        "N 1 0 False 0",
        "N 1 0 True 0",
        "N 1 0 True 1",
        "N 1 0 True 2",
        "N 1 1 False 3",
        "N 2 -1 False 0",
        "N 2 -1 False 1",
        "N 2 -1 True 0",
        "N 2 -1 True 1",
        "N 2 0 False 0",
        "N 2 0 False 1",
        "N 2 0 True 0",
        "N 2 0 True 1",
        "N 2 1 False 0",
        "N 2 1 False 1",
        "N 2 1 False 2",
        "N 2 1 True 0",
        "N 2 1 True 1",
        "N 3 0 False 0",
        "N 3 0 False 1",
        "N 3 0 False 2",
        "N 3 0 True 0",
        "N 3 0 True 1",
        "N 3 0 True 2",
        "N 3 1 False 0",
        "N 3 1 False 1",
        "N 3 1 True 0",
        "N 3 1 True 1",
        "N 4 1 False 0",
        "N 4 1 False 1",
        "N 4 1 False 2",
        "N 4 1 False 3",
        "O 1 -1 False 0",
        "O 1 -1 True 0",
        "O 1 0 False 0",
        "O 1 0 False 1",
        "O 1 0 True 0",
        "O 1 0 True 1",
        "O 2 0 False 0",
        "O 2 0 False 1",
        "O 2 0 False 2",
        "O 2 0 True 0",
        "O 2 0 True 1",
        "O 2 0 True 2",
        "O 2 1 False 0",
        "O 2 1 False 1",
        "O 2 1 True 0",
        "O 3 1 False 0",
        "O 3 1 False 1",
        "P 1 0 False 1",
        "P 2 0 False 0",
        "P 2 0 False 1",
        "P 2 0 True 0",
        "P 3 0 False 0",
        "P 4 0 False 0",
        "P 4 0 False 1",
        "P 4 0 True 0",
        "P 4 0 True 1",
        "P 4 1 False 0",
        "P 5 0 False 0",
        "P 5 0 False 1",
        "S 1 -1 False 0",
        "S 1 -1 True 0",
        "S 1 0 False 0",
        "S 1 0 True 0",
        "S 2 0 False 0",
        "S 2 0 True 0",
        "S 2 0 False 1",
        "S 2 1 True 0",
        "S 3 0 False 0",
        "S 3 0 True 0",
        "S 3 1 True 0",
        "S 3 1 False 0",
        "S 4 0 False 0",
        "S 4 0 False 1",
        "S 4 0 True 0",
        "S 4 1 False 0",
        "S 4 1 True 0",
    ]
    int_key_dict = {k: v for k, v in enumerate(feature_list)}
    str_key_dict = {v: k for k, v in int_key_dict.items()}

    @staticmethod
    def get_number_of_features() -> int:
        return len(AtomFeatures.feature_list)

    @staticmethod
    def atom_features_from_molecule(molecule: Molecule, index: int) -> int:
        return AtomFeatures.str_key_dict[AtomFeatures.return_atom_feature_key_from_molecule(molecule, index)]

    @staticmethod
    def return_atom_feature_key_from_molecule(molecule: Molecule, index: int) -> str:
        atom = molecule.GetAtomWithIdx(index)
        key = f"{atom.GetSymbol()} {len(atom.GetBonds())} {atom.GetFormalCharge()} {is_atom_bonded_conjugate(atom)} {atom.GetTotalNumHs(includeNeighbors=True)}"
        return key

    @staticmethod
    def atom_features_from_data(data: Sequence[Any]) -> int:
        key = f"{data[0]} {int(data[1])} {int(data[2])} {data[3]} {int(data[4])}"
        return AtomFeatures.str_key_dict[key]

    @staticmethod
    def lookup_int(key: int) -> str:
        return AtomFeatures.int_key_dict[key]

    @staticmethod
    def lookup_str(key: str) -> int:
        return AtomFeatures.str_key_dict[key]

    @staticmethod
    def get_split_feature_typed_from_key(key: Union[int, str]) -> int:
        if isinstance(key, int):
            af_string = AtomFeatures.int_key_dict[key]
        else:
            af_string = key
        element, num_bonds, charge, conjugated, num_hydrogens = af_string.split(" ")
        return (element, int(num_bonds), int(charge), True if conjugated == "True" else False, int(num_hydrogens))

    @staticmethod
    def atom_features_from_molecule_w_connection_info(
        molecule: Molecule, index: int, connected_to: Tuple[Any] = (-1, -1)
    ) -> int:
        connected_bond_type = (
            -1
            if connected_to[1] == -1
            else get_connection_info_bond_type(molecule, int(index), int(connected_to[1]))
            # int(molecule.GetBondBetweenAtoms(int(index), int(connected_to[1])).GetBondType())
        )
        key = AtomFeatures.return_atom_feature_key_from_molecule(molecule, index)
        return [AtomFeatures.str_key_dict[key], connected_to[0], connected_bond_type]

    @staticmethod
    def atom_features_from_data_w_connection_info(data: Sequence[Any]) -> int:
        try:
            connected_to = (int(data[5]), 0)
        except (TypeError, ValueError):
            connected_to = (-1, -1)
        try:
            bond_type_str = data[6]
            conenection_bond_type = (
                bond_type_str
                if bond_type_str in Chem.BondType.names
                else int(
                    Chem.rdchem.BondType.values[
                        int(bond_type_str[bond_type_str.find("(") + 1 : bond_type_str.find(")")])
                    ]
                )
            )
        except (AttributeError, ValueError):
            conenection_bond_type = -1
        key = f"{data[0]} {int(data[1])} {int(data[2])} {data[3]} {int(data[4])}"
        return [AtomFeatures.str_key_dict[key], connected_to[0], conenection_bond_type]

    @staticmethod
    def similarity(feature1: int, feature2: int) -> float:
        fp1 = AtomFeatures.lookup_int(feature1).split(" ")
        fp2 = AtomFeatures.lookup_int(feature2).split(" ")
        return np.sum([1 for i, j in zip(fp1, fp2) if i == j]) / len(fp1)

    @staticmethod
    def similarity_w_connection_info(feature1: int, feature2: int) -> float:
        fp1 = AtomFeatures.lookup_int(feature1[0]).split(" ")
        fp2 = AtomFeatures.lookup_int(feature2[0]).split(" ")
        ret_val = np.sum([1 for i, j in zip(fp1, fp2) if i == j])
        if feature1[1] == feature2[1]:
            ret_val += 1
        if feature1[2] == feature2[2]:
            ret_val += 1
        return ret_val / (len(fp1) + 2)

    @staticmethod
    def is_similar(feature1: int, feature2: int, threshold: float) -> float:
        similarity = AtomFeatures.similarity(feature1, feature2)
        return similarity >= threshold

    @staticmethod
    def is_similar_w_connection_info(feature1: int, feature2: int, threshold: float) -> float:
        similarity = AtomFeatures.similarity_w_connection_info(feature1, feature2)
        return similarity >= threshold
