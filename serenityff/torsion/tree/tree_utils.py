from serenityff.charge.tree.atom_features import AtomFeatures


def get_canon_torsion_feature(af1: int, af2: int, af3: int, af4: int, max_number_afs_for_concat: int = None):
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
    concat_torsion_feature = (
        af1
        + af2 * max_number_afs_for_concat
        + af3 * max_number_afs_for_concat**2
        + af4 * max_number_afs_for_concat**3
    )
    return concat_torsion_feature
