"""Factory for dash Trees with different properties loaded."""
from enum import Enum, auto


class TreeType(Enum):
    # MBIS
    DEFAULT = auto()
    # Other Charges
    AM1BCC = auto()
    RESP = auto()
    MULLIKEN = auto()
    CHARGES = auto()
    # Other
    DUALDESCRIPTORS = auto()
    C6 = auto()
    POLARIZABILITY = auto()
    DFTD4 = auto()
    DIPOLE = auto()
    # Full
    FULL = auto()


# class Forest:
#     def get_MBIS_DASH_tree(preload: bool = True, verbose: bool = True):
#         return DASHTree(preload=preload, verbose=verbose)

# def get_AM1BCC_DASH_tree(preload: bool = True, verbose: bool = True):
#     return DASHTree(preload=preload, verbose=verbose, tree_type=)
