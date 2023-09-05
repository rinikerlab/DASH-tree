import os
import pickle
import gzip
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Process, Manager

from serenityff.charge.tree.atom_features import AtomFeatures
from serenityff.charge.data import default_dash_tree_path
from serenityff.charge.utils.rdkit_typing import Molecule


class DASHTree:
    def __init__(
        self,
        tree_folder_path: str = default_dash_tree_path,
        preload: bool = True,
        verbose: bool = True,
        num_processes: int = 1,
    ):
        """
        Class to handle DASH trees and data

        Parameters
        ----------
        tree_folder_path : str
            Path to folder containing DASH trees and data
        preload : bool
            If True, load all trees and data into memory, if False, load on demand
        verbose : bool
            If True, print status messages
        num_processes : int
            Number of processes to use for loading and assigning molecules.
            TODO: This is currently slow and not recommended
        """
        self.tree_folder_path = tree_folder_path
        self.verbose = verbose
        self.num_processes = num_processes
        self.tree_storage = {}
        self.data_storage = {}
        if preload:
            self.load_all_trees_and_data()

    ########################################
    #   Tree import/export functions
    ########################################

    # tree file format:
    # int(id_counter), int(atom_type), int(con_atom), int(con_type), float(oldNode.attention), []children

    def load_all_trees_and_data(self):
        """
        Load all trees and data from the tree_folder_path, expects files named after the atom feature key and
        the file extension .gz for the tree and .h5 for the data
        Examples:
            0.gz, 0.h5 for the tree and data of the atom feature with key 0
        """
        if self.verbose:
            print("Loading DASH tree data")
        # import all files
        if self.num_processes <= 1:
            for i in range(AtomFeatures.get_number_of_features()):
                tree_path = os.path.join(self.tree_folder_path, f"{i}.gz")
                df_path = os.path.join(self.tree_folder_path, f"{i}.h5")
                self.load_tree_and_data(tree_path, df_path, branch_idx=i)
        else:
            self.tree_storage = Manager().dict()
            self.data_storage = Manager().dict()
            processes = []
            for i in range(AtomFeatures.get_number_of_features()):
                tree_path = os.path.join(self.tree_folder_path, f"{i}.gz")
                df_path = os.path.join(self.tree_folder_path, f"{i}.h5")
                p = Process(target=self.load_tree_and_data, args=(tree_path, df_path))
                processes.append(p)
                p.start()
            for p in processes:
                p.join()

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
        self.tree_storage[branch_idx] = tree
        self.data_storage[branch_idx] = df

    def save_all_trees_and_data(self):
        """
        Save all trees and data to the tree_folder_path with the file names {branch_idx}.gz and {branch_idx}.h5
        """
        if self.verbose:
            print(f"Saving DASH tree data to {len(self.tree_storage)} files in {self.tree_folder_path}")
        for branch_idx in tqdm(self.tree_storage):
            self.save_tree_and_data(branch_idx)

    def save_tree_and_data(self, branch_idx: int):
        """
        Save a tree branch and data to the tree_folder_path with the file names {branch_idx}.gz and {branch_idx}.h5

        Parameters
        ----------
        branch_idx : int
            Atom feature key of the tree branch to save
        """
        tree_path = os.path.join(self.tree_folder_path, f"{branch_idx}.gz")
        df_path = os.path.join(self.tree_folder_path, f"{branch_idx}.h5")
        self._save_tree_and_data(branch_idx, tree_path, df_path)

    def _save_tree_and_data(self, branch_idx: int, tree_path: str, df_path: str):
        with gzip.open(tree_path, "wb") as f:
            pickle.dump(self.tree_storage[branch_idx], f)
        self.data_storage[branch_idx].to_hdf(df_path, key="df", mode="w")

    ########################################
    #   Tree assignment functions
    ########################################

    def _pick_subgraph_expansion_node(
        self, current_node: int, branch_idx: int, possible_new_atom_features: list, possible_new_atom_idxs: list
    ):
        current_node_children = self.tree_storage[branch_idx][current_node][5]
        for child in current_node_children:
            child_tree_node = self.tree_storage[branch_idx][child]
            child_af = [child_tree_node[1], child_tree_node[2], child_tree_node[3]]
            for possible_atom_feature, possible_atom_idx in zip(possible_new_atom_features, possible_new_atom_idxs):
                if possible_atom_feature == child_af:
                    return (child, possible_atom_idx)
        return (None, None)

    def _init_neighbor_dict(self, mol: Molecule):
        neighbor_dict = {}
        for atom_idx, atom in enumerate(mol.GetAtoms()):
            neighbor_dict[atom_idx] = []
            for neighbor in atom.GetNeighbors():
                af_with_connection_info = AtomFeatures.atom_features_from_molecule_w_connection_info(
                    mol, neighbor.GetIdx(), (0, atom_idx)
                )
                neighbor_dict[atom_idx].append((neighbor.GetIdx(), af_with_connection_info))
        return neighbor_dict

    def _new_neighbors(self, neighbor_dict, connected_atoms):
        connected_atoms_set = set(connected_atoms)
        new_neighbors_afs = []
        new_neighbors = []
        for rel_atom_idx, atom_idx in enumerate(connected_atoms):
            for neighbor in neighbor_dict[atom_idx]:
                if neighbor[0] not in connected_atoms_set and neighbor[0] not in new_neighbors:
                    new_neighbors.append(neighbor[0])
                    new_neighbors_afs.append(
                        [neighbor[1][0], rel_atom_idx, neighbor[1][2]]
                    )  # fix rel_atom_idx (the atom index in the subgraph)
        return new_neighbors_afs, new_neighbors

    def _new_neighbors_atomic(self, neighbor_dict, connected_atoms, atom_idx_added):
        connected_atoms_set = set(connected_atoms)
        new_neighbors_afs = []
        new_neighbors = []
        for neighbor in neighbor_dict[atom_idx_added]:
            if neighbor[0] not in connected_atoms_set:
                new_neighbors.append(neighbor[0])
                new_neighbors_afs.append(
                    [neighbor[1][0], atom_idx_added, neighbor[1][2]]
                )  # fix rel_atom_idx (the atom index in the subgraph)
        return new_neighbors_afs, new_neighbors

    def match_new_atom(
        self,
        atom: int,
        mol: Molecule,
        max_depth: int = 16,
        attention_threshold: float = 10,
        attention_increment_threshold: float = 0,
    ):
        """
        Match a atom in a molecule to a DASH tree subgraph. The matching is done by starting at the atom and
        traversing the tree until the max_depth is reached or the attention_threshold is exceeded.
        If the attention_increment_threshold is exceeded, the matching is stopped and the current path is returned.

        Parameters
        ----------
        atom : int
            Atom index in the molecule of the atom to match
        mol : Molecule
            RDKit molecule object in which the atom is located
        max_depth : int
            Maximum depth of the tree to traverse
        attention_threshold : float
            Maximum cumulative attention value to traverse the tree
        attention_increment_threshold : float
            Minimum attention increment to stop the traversal
        """
        neighbor_dict = self._init_neighbor_dict(mol)
        init_atom_feature = AtomFeatures.atom_features_from_molecule_w_connection_info(mol, atom)
        branch_idx = init_atom_feature[0]  # branch_idx is the key of the AtomFeature without connection info
        matched_node_path = [branch_idx, 0]
        if branch_idx not in self.tree_storage:
            self.load_tree_and_data(
                os.path.join(self.tree_folder_path, f"{branch_idx}.gz"),
                os.path.join(self.tree_folder_path, f"{branch_idx}.h5"),
            )
        cummulative_attention = 0
        # Special case for H -> only connect to heavy atom and ignore H
        if mol.GetAtomWithIdx(atom).GetAtomicNum() == 1:
            h_connected_heavy_atom = mol.GetAtomWithIdx(atom).GetNeighbors()[0].GetIdx()
            init_atom_feature = AtomFeatures.atom_features_from_molecule_w_connection_info(mol, h_connected_heavy_atom)
            child, _ = self._pick_subgraph_expansion_node(0, branch_idx, [init_atom_feature], [h_connected_heavy_atom])
            matched_node_path.append(child)
            atom_indices_in_subgraph = [h_connected_heavy_atom]  # skip Hs as they are only treated implicitly
            max_depth -= 1  # reduce max_depth by 1 as we already added one node
        else:
            atom_indices_in_subgraph = [atom]
        if max_depth <= 1:
            return matched_node_path
        else:
            possible_new_atom_features, possible_new_atom_idxs = self._new_neighbors(
                neighbor_dict, atom_indices_in_subgraph
            )
            for _ in range(1, max_depth):
                child, atom = self._pick_subgraph_expansion_node(
                    matched_node_path[-1], branch_idx, possible_new_atom_features, possible_new_atom_idxs
                )
                if child is None:
                    return matched_node_path
                matched_node_path.append(child)
                atom_indices_in_subgraph.append(atom)
                node_attention = self.tree_storage[branch_idx][child][4]
                cummulative_attention += node_attention
                possible_new_atom_features_toAdd, possible_new_atom_idxs_toAdd = self._new_neighbors_atomic(
                    neighbor_dict, atom_indices_in_subgraph, atom
                )
                possible_new_atom_features.extend(possible_new_atom_features_toAdd)
                possible_new_atom_idxs.extend(possible_new_atom_idxs_toAdd)
                if cummulative_attention > attention_threshold:
                    return matched_node_path
                if node_attention < attention_increment_threshold:
                    return matched_node_path
            return matched_node_path

    def get_atom_properties(self, matched_node_path: list):
        """
        Get the properties of a atom from a matched DASH tree subgraph (node path)

        Parameters
        ----------
        matched_node_path : list
            List of node ids of the matched subgraph (node path) in the order of the traversal

        Returns
        -------
        pd.Series
            All properties of the atom which where stored in the DASH tree
        """
        branch_idx = matched_node_path[0]
        atom = matched_node_path[-1]
        if branch_idx not in self.data_storage:
            try:
                self.load_tree_and_data(
                    os.path.join(self.tree_folder_path, f"{branch_idx}.gz"),
                    os.path.join(self.tree_folder_path, f"{branch_idx}.h5"),
                )
            except Exception as e:
                print(f"Error loading tree {branch_idx}: {e}")
        df = self.data_storage[branch_idx]
        return df.iloc[atom]

    def get_molecules_partial_charges(
        self,
        mol: Molecule,
        norm_method: str = "std_weighted",
        max_depth: int = 16,
        attention_threshold: float = 10,
        attention_incremet_threshold: float = 0,
        verbose: bool = False,
        default_std_value: float = 0.1,
        chg_key: str = "result",
        chg_std_key: str = "stdDeviation",
    ):
        """
        Get the partial charges of all atoms in a molecule by matching them to DASH tree subgraphs and
        normalizing the charges of the matched atoms

        Parameters
        ----------
        mol : Molecule
            RDKit molecule object for which the partial charges should be calculated
        norm_method : str
            Method to normalize the partial charges, one of 'none', 'symmetric', 'std_weighted'
        max_depth : int
            Maximum depth of the tree to traverse
        attention_threshold : float
            Maximum cumulative attention value to traverse the tree
        attention_incremet_threshold : float
            Minimum attention increment to stop the traversal
        verbose : bool
            If True, print status messages
        default_std_value : float
            Default value to use for the standard deviation if it is 0
        chg_key : str
            Key of the partial charge in the DASH tree data
        chg_std_key : str
            Key of the partial charge standard deviation in the DASH tree data

        Returns
        -------
        dict
            Dictionary containing the partial charges, standard deviations and match depths of all atoms
        """
        return_list = []
        tree_raw_charges = []
        tree_charge_std = []
        tree_match_depth = []
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            try:
                node_path = self.match_new_atom(
                    atom_idx,
                    mol,
                    max_depth=max_depth,
                    attention_threshold=attention_threshold,
                    attention_increment_threshold=attention_incremet_threshold,
                )
                node_properties = self.get_atom_properties(node_path)
                tree_raw_charges.append(float(node_properties[chg_key]))
                tmp_tree_std = float(node_properties[chg_std_key])
                tree_charge_std.append(tmp_tree_std if tmp_tree_std > 0 else default_std_value)
                tree_match_depth.append(len(node_path) - 1)
            except Exception as e:
                print(e)
                tree_raw_charges.append(np.NaN)
                tree_charge_std.append(np.NaN)
                tree_match_depth.append(-1)

        if verbose:
            print(f"Tree raw charges: {tree_raw_charges}")
        if norm_method == "none":
            return_list = tree_raw_charges
        elif norm_method == "symmetric":
            tot_charge_tree = sum(tree_raw_charges)
            tot_charge_mol = sum([x.GetFormalCharge() for x in mol.GetAtoms()])
            return_list = [x + (tot_charge_mol - tot_charge_tree) / len(tree_raw_charges) for x in tree_raw_charges]
        elif norm_method == "std_weighted":
            tot_charge_tree = sum(tree_raw_charges)
            tot_charge_mol = sum([x.GetFormalCharge() for x in mol.GetAtoms()])
            tot_std_tree = sum(tree_charge_std)
            return_list = [
                x + (tot_charge_mol - tot_charge_tree) * (y / tot_std_tree)
                for x, y in zip(tree_raw_charges, tree_charge_std)
            ]
        else:
            raise ValueError("norm_method must be one of 'none', 'symmetric', 'std_weighted'")
        if verbose:
            print(f"Tree normalized charges: {return_list}")
        return {"charges": return_list, "std": tree_charge_std, "match_depth": tree_match_depth}
