# from serenityff.charge.tree.atom_features import AtomFeatures
from serenityff.charge.utils.rdkit_typing import Molecule
from typing import List
from serenityff.charge.tree.atom_features_reduced import AtomFeaturesReduced as AtomFeatures


def torsion_is_in_Ring(mol: Molecule, indices: List[int]) -> int:
    a, b, c, d = indices
    bond = mol.GetBondBetweenAtoms(int(b), int(c))
    return int(bond.IsInRing())


def get_canon_torsion_feature(
    af1: int, af2: int, af3: int, af4: int, max_number_afs_for_concat: int = None, useRingsInMol: Molecule = None
) -> int:
    # get the canonical torsion feature
    # defined as the feature with the smallest numbers to the left, considering the mirror symmetry
    # e.g. [1, 2, 3, 4] and [4, 3, 2, 1] are the same and the canonical torsion feature is [1, 2, 3, 4]
    # e.g. [1, 1, 2, 1] and [1, 2, 1, 1] are the same and the canonical torsion feature is [1, 1, 2, 1]
    if af1 > af4:
        af1, af2, af3, af4 = af4, af3, af2, af1  # flip to the canonical order
    elif af1 == af4 and af2 > af3:
        af1, af2, af3, af4 = af4, af3, af2, af1  # flip to the canonical order
    if max_number_afs_for_concat is None:
        max_number_afs_for_concat = AtomFeatures.get_number_of_features()
        if max_number_afs_for_concat < 100:
            max_number_afs_for_concat = 100  # set too 100 for nice look and easy to read
    concat_torsion_feature = (
        af1
        + af2 * max_number_afs_for_concat
        + af3 * max_number_afs_for_concat**2
        + af4 * max_number_afs_for_concat**3
    )
    if useRingsInMol is not None:
        concat_torsion_feature += (
            torsion_is_in_Ring(useRingsInMol, (af1, af2, af3, af4)) * max_number_afs_for_concat**4
        )
    return concat_torsion_feature
