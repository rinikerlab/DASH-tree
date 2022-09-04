from typing import Any, Sequence, Tuple
import numpy as np
from rdkit import Chem

from serenityff.charge.utils import Molecule


class AtomFeatures:
    feature_list = [
        "Br 1 0 SP3 False 0",
        "C 2 0 SP False 0",
        "C 2 0 SP False 1",
        "C 3 -1 SP2 False 0",
        "C 3 -1 SP2 True 0",
        "C 3 -1 SP2 True 1",
        "C 3 -1 SP3 False 0",
        "C 3 0 SP2 False 0",
        "C 3 0 SP2 False 1",
        "C 3 0 SP2 False 2",
        "C 3 0 SP2 True 0",
        "C 3 0 SP2 True 1",
        "C 3 1 SP2 False 0",
        "C 3 1 SP2 True 0",
        "C 4 0 SP3 False 0",
        "C 4 0 SP3 False 1",
        "C 4 0 SP3 False 2",
        "C 4 0 SP3 False 3",
        "Cl 1 0 SP3 False 0",
        "F 1 0 SP3 False 0",
        "H 1 0 S False 0",
        "I 1 0 SP3 False 0",
        "N 1 -1 SP2 False 0",
        "N 1 0 SP False 0",
        "N 2 0 SP2 False 0",
        "N 2 0 SP2 False 1",
        "N 2 0 SP2 True 0",
        "N 2 1 SP False 0",
        "N 3 0 SP2 False 0",
        "N 3 0 SP2 False 1",
        "N 3 0 SP2 False 2",
        "N 3 0 SP2 True 0",
        "N 3 0 SP2 True 1",
        "N 3 0 SP3 False 0",
        "N 3 0 SP3 False 1",
        "N 3 0 SP3 False 2",
        "N 3 1 SP2 False 0",
        "N 3 1 SP2 True 0",
        "N 4 1 SP3 False 0",
        "N 4 1 SP3 False 3",
        "O 1 -1 SP2 False 0",
        "O 1 -1 SP3 False 0",
        "O 1 0 SP2 False 0",
        "O 2 0 SP2 False 0",
        "O 2 0 SP2 False 1",
        "O 2 0 SP2 True 0",
        "O 2 0 SP3 False 0",
        "O 2 0 SP3 False 1",
        "O 2 1 SP2 False 0",
        "O 2 1 SP2 False 1",
        "O 2 1 SP2 True 0",
        "P 2 0 SP2 False 0",
        "P 2 0 SP2 False 1",
        "P 2 0 SP2 True 0",
        "P 3 0 SP3 False 0",
        "P 4 0 SP3 False 0",
        "P 4 0 SP3 False 1",
        "P 4 1 SP3 False 0",
        "S 1 0 SP2 False 0",
        "S 2 0 SP2 False 0",
        "S 2 0 SP2 True 0",
        "S 2 0 SP3 False 0",
        "S 2 0 SP3 False 1",
        "S 2 1 SP2 True 0",
        "S 3 0 SP2 False 0",
        "S 3 0 SP3 False 0",
        "S 3 1 SP2 True 0",
        "S 3 1 SP3 False 0",
        "S 4 0 SP3 False 0",
        "S 4 0 SP3D False 0",
        "S 4 1 SP3 False 0",
    ]
    int_key_dict = {k: v for k, v in enumerate(feature_list)}
    str_key_dict = {v: k for k, v in int_key_dict.items()}

    @staticmethod
    def atom_features_from_molecule(molecule: Molecule, index: int) -> int:
        atom = molecule.GetAtomWithIdx(index)
        key = f"{atom.GetSymbol()} {len(atom.GetBonds())} {atom.GetFormalCharge()} {str(atom.GetHybridization())} {atom.GetIsAromatic()} {atom.GetTotalNumHs(includeNeighbors=True)}"
        return AtomFeatures.str_key_dict[key]

    @staticmethod
    def atom_features_from_data(data: Sequence[Any]) -> int:
        key = f"{data[0]} {int(data[1])} {int(data[2])} {data[3]} {data[4]} {int(data[5])}"
        return AtomFeatures.str_key_dict[key]

    @staticmethod
    def lookup_int(key: int) -> str:
        return AtomFeatures.int_key_dict[key]

    @staticmethod
    def lookup_str(key: str) -> int:
        return AtomFeatures.str_key_dict[key]

    @staticmethod
    def atom_features_from_molecule_w_connection_info(
        molecule: Molecule, index: int, connected_to: Tuple[Any] = (-1, -1)
    ) -> int:
        connected_bond_type = (
            -1
            if connected_to[1] == -1
            else str(molecule.GetBondBetweenAtoms(int(index), int(connected_to[1])).GetBondType())
        )
        atom = molecule.GetAtomWithIdx(index)
        key = f"{atom.GetSymbol()} {len(atom.GetBonds())} {atom.GetFormalCharge()} {str(atom.GetHybridization())} {atom.GetIsAromatic()} {atom.GetTotalNumHs(includeNeighbors=True)}"
        return [AtomFeatures.str_key_dict[key], connected_to[0], connected_bond_type]

    @staticmethod
    def atom_features_from_data_w_connection_info(data: Sequence[Any]) -> int:
        try:
            connected_to = (int(data[6]), 0)
        except (TypeError, ValueError):
            connected_to = (-1, -1)
        try:
            bond_type_str = data[7]
            conenection_bond_type = (
                bond_type_str
                if bond_type_str in Chem.BondType.names
                else str(
                    Chem.rdchem.BondType.values[
                        int(bond_type_str[bond_type_str.find("(") + 1 : bond_type_str.find(")")])
                    ]
                )
            )
        except (AttributeError, ValueError):
            conenection_bond_type = -1
        key = f"{data[0]} {int(data[1])} {int(data[2])} {data[3]} {data[4]} {int(data[5])}"
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
