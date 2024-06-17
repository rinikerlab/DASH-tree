"""Factory for dash Trees with different properties loaded."""
from serenityff.charge.tree.dash_tree import DASHTree
from serenityff.charge.tree.retrieve_data import DASH_PROPS_DIR


class Forest:
    @staticmethod
    def get_MBIS_DASH_tree(preload: bool = True, verbose: bool = True) -> DASHTree:
        return DASHTree(preload=preload, verbose=verbose)

    @staticmethod
    def get_AM1BCC_DASH_tree(preload: bool = True, verbose: bool = True) -> DASHTree:
        return DASHTree(
            tree_folder_path=DASH_PROPS_DIR,
            preload=preload,
            verbose=verbose,
            default_value_column="AM1BCC",
            default_std_column="AM1BCC_std",
        )

    @staticmethod
    def get_RESP1_DASH_tree(preload: bool = True, verbose: bool = True) -> DASHTree:
        return DASHTree(
            tree_folder_path=DASH_PROPS_DIR,
            preload=preload,
            verbose=verbose,
            default_value_column="RESP1",
        )

    @staticmethod
    def get_RESP2_DASH_tree(preload: bool = True, verbose: bool = True) -> DASHTree:
        return DASHTree(
            tree_folder_path=DASH_PROPS_DIR,
            preload=preload,
            verbose=verbose,
            default_value_column="RESP2",
        )

    @staticmethod
    def get_mulliken_DASH_tree(preload: bool = True, verbose: bool = True) -> DASHTree:
        return DASHTree(
            tree_folder_path=DASH_PROPS_DIR,
            preload=preload,
            verbose=verbose,
            default_value_column="mulliken",
        )

    @staticmethod
    def get_charges_DASH_tree(preload: bool = True, verbose: bool = True) -> DASHTree:
        return DASHTree(
            tree_folder_path=DASH_PROPS_DIR,
            preload=preload,
            verbose=verbose,
        )

    @staticmethod
    def get_dual_descriptor_DASH_tree(preload: bool = True, verbose: bool = True) -> DASHTree:
        return DASHTree(
            tree_folder_path=DASH_PROPS_DIR,
            preload=preload,
            verbose=verbose,
            default_value_column="dual",
        )

    @staticmethod
    def get_dipole_DASH_tree(preload: bool = True, verbose: bool = True) -> DASHTree:
        return DASHTree(
            tree_folder_path=DASH_PROPS_DIR,
            preload=preload,
            verbose=verbose,
            default_value_column="mbis_dipole_strength",
        )

    @staticmethod
    def get_C6_DASH_tree(preload: bool = True, verbose: bool = True) -> DASHTree:
        return DASHTree(
            tree_folder_path=DASH_PROPS_DIR,
            preload=preload,
            verbose=verbose,
            default_value_column="DFTD4:C6",
            default_std_column="DFTD4:C6_std",
        )

    @staticmethod
    def get_polarizability_DASH_tree(preload: bool = True, verbose: bool = True) -> DASHTree:
        return DASHTree(
            tree_folder_path=DASH_PROPS_DIR,
            preload=preload,
            verbose=verbose,
            default_value_column="DFTD4:polarizability",
            default_std_column="DFTD4:polarizability_std",
        )

    @staticmethod
    def get_full_props_DASH_tree(preload: bool = True, verbose: bool = True) -> DASHTree:
        return DASHTree(tree_folder_path=DASH_PROPS_DIR, preload=preload, verbose=verbose)
