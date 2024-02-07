import os
import gzip
import pickle
import pandas as pd
import numpy as np
from serenityff.charge.tree.dash_tree import DASHTree

# from serenityff.charge.tree.atom_features import AtomFeatures
from serenityff.charge.tree.atom_features_reduced import AtomFeaturesReduced
from serenityff.torsion.data import default_dash_torsion_tree_path
from serenityff.charge.utils.rdkit_typing import Molecule
from serenityff.torsion.tree.dash_utils import get_canon_torsion_feature


class DASHTorsionTree(DASHTree):
    def __init__(
        self,
        tree_folder_path: str = default_dash_torsion_tree_path,
        preload: bool = True,
        verbose: bool = True,
        num_processes: int = 1,
    ):
        super().__init__(tree_folder_path, preload, verbose, num_processes)
        self.atom_feature_type = AtomFeaturesReduced

    def load_all_trees_and_data(self):
        """
        Load all trees and data from the tree_folder_path, expects files named after the atom feature key and
        the file extension .gz for the tree and .h5 for the data
        Examples:
            0.gz, 0.h5 for the tree and data of the atom feature with key 0
        """
        if self.verbose:
            print("Loading DASH tree data")
        # find all files in self.tree_folder_path, ending with .gz
        files = os.listdir(self.tree_folder_path)
        files = [file for file in files if file.endswith(".gz")]
        file_indices = [int(file.split(".")[0]) for file in files]
        # import all files
        for i in file_indices:
            tree_path = os.path.join(self.tree_folder_path, f"{i}.gz")
            df_path = os.path.join(self.tree_folder_path, f"{i}.h5")
            self.load_tree_and_data(tree_path, df_path, branch_idx=i)
        if self.verbose:
            print(f"Loaded {len(self.tree_storage)} trees and data")

    def load_tree_and_data(self, tree_path: str, df_path: str, hdf_key: str = "df", branch_idx: int = None):
        """
        Load a tree and data from the tree_folder_path, expects files named after the atom feature key and
        the file extension .gz for the tree and .h5 for the data
        Examples:
            0.gz, 0.h5 for the tree and data of the atom feature with key 0

        Parameters
        ----------
        tree_path : str
            the path to the tree file
        df_path : str
            the path to the data file
        hdf_key : str, optional
            the key of the data in the hdf file, by default "df"
        branch_idx : int, optional
            the atom feature key of the tree branch, by default takes the key from the file name
        """
        if branch_idx is None:
            branch_idx = int(os.path.basename(tree_path).split(".")[0])
        with gzip.open(tree_path, "rb") as f:
            tree = pickle.load(f)
        df = pd.read_hdf(df_path, key=hdf_key, mode="r")
        try:
            df.drop(columns=["histogram", "max_attention", "con_type", "con_atom"], inplace=True)
        except KeyError:
            pass
        # remove empty csd arrays for memory efficiency
        try:
            df["csd"] = df["csd"].apply(lambda x: x if np.sum(x) > 0 else np.array([0]))
        except KeyError:
            pass
        self.tree_storage[branch_idx] = tree
        self.data_storage[branch_idx] = df

    def _get_init_layer(self, mol: Molecule, atom: int, max_depth: int):
        if len(atom) != 4:
            raise ValueError(f"A list of 4 atom indices is required to define a torsion angle. Got {atom} instead.")
        af1, af2, af3, af4 = [self.atom_feature_type.atom_features_from_molecule(mol, atom_i) for atom_i in atom]
        canon_init_torsion_feature = get_canon_torsion_feature(af1, af2, af3, af4, useRingsInMol=mol)
        matched_node_path = [canon_init_torsion_feature, 0]
        max_depth = max(max_depth - 3, 0)
        return canon_init_torsion_feature, matched_node_path, atom, max_depth

    def match_new_torsion(
        self,
        atoms_in_torsion: [int],
        mol: Molecule,
        max_depth: int = 16,
        attention_threshold: float = 10,
        attention_increment_threshold: float = 0,
        return_atom_indices: bool = False,
        neighbor_dict=None,
    ):
        if len(atoms_in_torsion) != 4:
            raise ValueError(
                f"A list of 4 atom indices is required to define a torsion angle. Got {atoms_in_torsion} instead."
            )
        # Shh, don't tell anyone, but we match torsions like single atoms, just with a overwriten _get_init_layer method
        return super().match_new_atom(
            atoms_in_torsion,
            mol,
            max_depth,
            attention_threshold,
            attention_increment_threshold,
            return_atom_indices,
            neighbor_dict,
        )
