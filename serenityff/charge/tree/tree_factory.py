"""Factory for dash Trees with different properties loaded."""
from serenityff.charge.data import dash_props_tree_path
from serenityff.charge.tree.dash_tree import DASHTree, TreeType


class Forest:
    @staticmethod
    def get_MBIS_DASH_tree(preload: bool = True, verbose: bool = True) -> DASHTree:
        return DASHTree(preload=preload, verbose=verbose, tree_type=TreeType.DEFAULT)

    @staticmethod
    def get_AM1BCC_DASH_tree(preload: bool = True, verbose: bool = True) -> DASHTree:
        return DASHTree(
            tree_folder_path=dash_props_tree_path,
            preload=preload,
            verbose=verbose,
            default_value_column="AM1BCC",
            default_std_column="AM1BCC_std",
            tree_type=TreeType.AM1BCC,
        )

    @staticmethod
    def get_RESP1_DASH_tree(preload: bool = True, verbose: bool = True) -> DASHTree:
        return DASHTree(
            tree_folder_path=dash_props_tree_path,
            preload=preload,
            verbose=verbose,
            default_value_column="resp1",
            tree_type=TreeType.RESP,
        )

    @staticmethod
    def get_RESP2_DASH_tree(preload: bool = True, verbose: bool = True) -> DASHTree:
        return DASHTree(
            tree_folder_path=dash_props_tree_path,
            preload=preload,
            verbose=verbose,
            default_value_column="resp2",
            tree_type=TreeType.RESP,
        )

    @staticmethod
    def get_mulliken_DASH_tree(preload: bool = True, verbose: bool = True) -> DASHTree:
        return DASHTree(
            tree_folder_path=dash_props_tree_path,
            preload=preload,
            verbose=verbose,
            default_value_column="mulliken",
            tree_type=TreeType.MULLIKEN,
        )

    @staticmethod
    def get_charges_DASH_tree(preload: bool = True, verbose: bool = True) -> DASHTree:
        return DASHTree(
            tree_folder_path=dash_props_tree_path,
            preload=preload,
            verbose=verbose,
            tree_type=TreeType.CHARGES,
        )

    @staticmethod
    def get_dual_descriptor_DASH_tree(preload: bool = True, verbose: bool = True) -> DASHTree:
        return DASHTree(
            tree_folder_path=dash_props_tree_path,
            preload=preload,
            verbose=verbose,
            default_value_column="dual",
            tree_type=TreeType.DUALDESCRIPTORS,
        )

    @staticmethod
    def get_dipole_DASH_tree(preload: bool = True, verbose: bool = True) -> DASHTree:
        return DASHTree(
            tree_folder_path=dash_props_tree_path,
            preload=preload,
            verbose=verbose,
            default_value_column="mbis_dipole_strength",
            tree_type=TreeType.DIPOLE,
        )

    @staticmethod
    def get_C6_DASH_tree(preload: bool = True, verbose: bool = True) -> DASHTree:
        return DASHTree(
            tree_folder_path=dash_props_tree_path,
            preload=preload,
            verbose=verbose,
            default_value_column="DFTD4:C6",
            default_std_column="DFTD4:C6_std",
            tree_type=TreeType.C6,
        )

    @staticmethod
    def get_polarizability_DASH_tree(preload: bool = True, verbose: bool = True) -> DASHTree:
        return DASHTree(
            tree_folder_path=dash_props_tree_path,
            preload=preload,
            verbose=verbose,
            default_value_column="DFTD4:polarizability",
            default_std_column="DFTD4:polarizability_std",
            tree_type=TreeType.POLARIZABILITY,
        )

    @staticmethod
    def get_full_props_DASH_tree(preload: bool = True, verbose: bool = True) -> DASHTree:
        return DASHTree(
            tree_folder_path=dash_props_tree_path,
            preload=preload,
            verbose=verbose,
            tree_type=TreeType.FULL,
        )
