from serenityff.charge.tree.atom_features import AtomFeatures, get_connection_info_bond_type
from typing import Any, Tuple
import numpy as np
from rdkit import Chem
from serenityff.charge.utils import Molecule


class AtomFeaturesReduced(AtomFeatures):
    """
    Reduced set of atom features.
    """

    feature_list = (
        (5, 1, 0, 2),
        (5, 3, -1, 0),
        (5, 3, 0, 0),
        (5, 3, 0, 2),
        (5, 4, -1, 0),
        (5, 4, -1, 2),
        (35, 1, 0, 0),
        (6, 1, -1, 0),
        (6, 1, 0, 1),
        (6, 1, 0, 2),
        (6, 1, 0, 3),
        (6, 2, -1, 1),
        (6, 2, 0, 0),
        (6, 2, 0, 1),
        (6, 2, 0, 2),
        (6, 3, -1, 0),
        (6, 3, -1, 1),
        (6, 3, 0, 0),
        (6, 3, 0, 1),
        (6, 3, 0, 2),
        (6, 3, 1, 0),
        (6, 4, 0, 0),
        (6, 4, 0, 1),
        (6, 4, 0, 2),
        (6, 4, 0, 3),
        (17, 1, 0, 0),
        (9, 1, 0, 0),
        (1, 1, 0, 0),
        (53, 1, 0, 0),
        (53, 3, 0, 0),
        (53, 3, 0, 1),
        (53, 4, 0, 0),
        (7, 1, -1, 0),
        (7, 1, 0, 0),
        (7, 1, 0, 1),
        (7, 1, 0, 2),
        (7, 1, 1, 3),
        (7, 2, -1, 0),
        (7, 2, -1, 1),
        (7, 2, 0, 0),
        (7, 2, 0, 1),
        (7, 2, 1, 0),
        (7, 2, 1, 1),
        (7, 2, 1, 2),
        (7, 3, 0, 0),
        (7, 3, 0, 1),
        (7, 3, 0, 2),
        (7, 3, 1, 0),
        (7, 3, 1, 1),
        (7, 4, 1, 0),
        (7, 4, 1, 1),
        (7, 4, 1, 2),
        (7, 4, 1, 3),
        (8, 1, -1, 0),
        (8, 1, 0, 0),
        (8, 1, 0, 1),
        (8, 2, 0, 0),
        (8, 2, 0, 1),
        (8, 2, 0, 2),
        (8, 2, 1, 0),
        (8, 2, 1, 1),
        (8, 3, 1, 0),
        (8, 3, 1, 1),
        (15, 1, 0, 1),
        (15, 2, 0, 0),
        (15, 2, 0, 1),
        (15, 3, 0, 0),
        (15, 4, 0, 0),
        (15, 4, 0, 1),
        (15, 4, 1, 0),
        (15, 5, 0, 0),
        (15, 5, 0, 1),
        (16, 1, -1, 0),
        (16, 1, 0, 0),
        (16, 2, 0, 0),
        (16, 2, 0, 1),
        (16, 2, 1, 0),
        (16, 3, 0, 0),
        (16, 3, 1, 0),
        (16, 4, 0, 0),
        (16, 4, 0, 1),
        (16, 4, 1, 0),
    )
    afKey_2_afTuple = {k: v for k, v in enumerate(feature_list)}
    afTuple_2_afKey = {v: k for k, v in afKey_2_afTuple.items()}

    @staticmethod
    def get_number_of_features() -> int:
        return len(AtomFeaturesReduced.feature_list)

    @staticmethod
    def atom_features_from_molecule(molecule: Molecule, index: int) -> int:
        return AtomFeaturesReduced.afTuple_2_afKey[
            AtomFeaturesReduced.return_atom_feature_tuple_from_molecule(molecule, index)
        ]

    @staticmethod
    def return_atom_feature_tuple_from_molecule(molecule: Molecule, index: int) -> Tuple:
        atom = molecule.GetAtomWithIdx(index)
        af_tuple = (
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetTotalNumHs(includeNeighbors=True),
        )
        return af_tuple

    @staticmethod
    def lookup_int(key: int) -> str:
        return AtomFeaturesReduced.afKey_2_afTuple[key]

    @staticmethod
    def lookup_tuple(key: Tuple) -> int:
        return AtomFeaturesReduced.afTuple_2_afKey[key]

    @staticmethod
    def string_to_tuple(key: str) -> Tuple:
        atom_type, num_bons, charge, isConjugated, numHs = key.split(" ")
        return (
            Chem.GetPeriodicTable().GetAtomicNumber(atom_type),
            int(num_bons),
            int(charge),
            int(numHs),
        )

    @staticmethod
    def tuple_to_string(key: Tuple) -> str:
        return f"{Chem.GetPeriodicTable().GetElementSymbol(key[0])} {key[1]} {key[2]} {key[3]}"

    @staticmethod
    def lookup_sting(key: str) -> int:
        return AtomFeaturesReduced.afTuple_2_afKey[AtomFeaturesReduced.string_to_tuple(key)]

    @staticmethod
    def atom_features_from_molecule_w_connection_info(
        molecule: Molecule, index: int, connected_to: Tuple[Any] = (-1, -1)
    ) -> (int, int, int):
        connected_bond_type = (
            -1 if connected_to[1] == -1 else get_connection_info_bond_type(molecule, int(index), int(connected_to[1]))
        )
        key = AtomFeaturesReduced.atom_features_from_molecule(molecule, index)
        return (key, connected_to[0], connected_bond_type)

    @staticmethod
    def similarity(feature1: int, feature2: int) -> float:
        fp1 = AtomFeaturesReduced.lookup_int(feature1)
        fp2 = AtomFeaturesReduced.lookup_int(feature2)
        return np.sum([1 for i, j in zip(fp1, fp2) if i == j]) / len(fp1)

    @staticmethod
    def similarity_w_connection_info(feature1: int, feature2: int) -> float:
        fp1 = AtomFeaturesReduced.lookup_int(feature1[0])
        fp2 = AtomFeaturesReduced.lookup_int(feature2[0])
        ret_val = np.sum([1 for i, j in zip(fp1, fp2) if i == j])
        if feature1[1] == feature2[1]:
            ret_val += 1
        if feature1[2] == feature2[2]:
            ret_val += 1
        return ret_val / (len(fp1) + 2)
