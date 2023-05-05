import glob
import lzma
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

from serenityff.charge.tree.atom_features import AtomFeatures
from serenityff.charge.tree.node import node
from serenityff.charge.tree.tree_utils import get_possible_atom_features


class tree:
    def __init__(self):
        self.root = node(level=0)
        self.tree_lengths = defaultdict(int)
        self.max_depth = 36
        self.hasData = False

    ###############################################################################
    # General helper functions
    ###############################################################################

    def from_file(self, file_path: str, nrows=None):
        """
        Reads a tree from a file.

        Parameters
        ----------
        file_path : str
            The path to the file. (file must be readable by pandas)
        """
        self.root.from_file(file_path, nrows=nrows)
        self.hasData = True

    def _branch_from_file(self, file_path: str, verbose=False) -> node:
        """
        Reads a tree branch from a file. This is a helper function for from_folder.

        Parameters
        ----------
        file_path : str
            The path to the file. (file must be readable by pandas)
        """
        new_node = node(level=0)
        try:
            new_node.from_file(file_path)
            if verbose:
                print("Added {}".format(file_path))
        except Exception as e:
            print(e)
        return new_node

    def from_folder(self, folder_path: str, num_processes=8, verbose=False):
        """
        Reads a tree from a folder, with each file being a branch.
        This is the preferred way to read a tree, since it can be done in parallel.

        Parameters
        ----------
        folder_path : str
            The path to the folder with files for each branch.
        num_processes : int, optional
            The number of processes to use, by default 8
        verbose : bool, optional
            Whether to print the progress, by default False
        """
        self.root = node(level=0)
        all_files_in_folder = glob.glob(folder_path + "/tree*.csv")
        if verbose:
            print("Found {} files in folder".format(len(all_files_in_folder)))
        if num_processes <= 1:
            for file_path in all_files_in_folder:
                self._branch_from_file(file_path, verbose=verbose)
        else:
            # sort files by size so that the largest files are processed first
            all_files_in_folder = sorted(all_files_in_folder, key=lambda x: os.path.getsize(x))
            with Pool(num_processes) as p:
                res = p.map(self._branch_from_file, all_files_in_folder)
            for new_node in res:
                self.root.children.append(new_node)
        self.hasData = True

    def from_folder_pickle(self, folder_path: str, verbose=False):
        self.root = node(level=0)
        all_files_in_folder = glob.glob(folder_path + "/tree*.pkl")
        if verbose:
            print("Found {} files in folder".format(len(all_files_in_folder)))
        for file_path in all_files_in_folder:
            branch = pickle.load(open(file_path, "rb"))
            self.root.children.append(branch)

    def from_folder_pickle_lzma(self, folder_path: str, verbose=False):
        self.root = node(level=0)
        all_files_in_folder = glob.glob(folder_path + "/tree*.pkl")
        if verbose:
            print("Found {} files in folder".format(len(all_files_in_folder)))
        for file_path in all_files_in_folder:
            branch = pickle.load(lzma.open(file_path, "rb"))
            self.root.children.append(branch)

    def to_folder_pickle_lzmaz(self, folder_path: str, verbose=False):
        for i, branch in enumerate(self.root.children):
            pickle.dump(branch, lzma.open(f"{folder_path}/tree_{i}.pkl", "wb"))

    def update_tree_length(self):
        """
        Updates the tree_lengths dictionary.
        """
        self.tree_lengths.clear()
        self.tree_lengths = self.root.get_tree_length(self.tree_lengths)
        self.max_depth = max(self.tree_lengths.keys()) + 1

    def get_sum_nodes(self):
        """
        Returns the sum of the nodes in the tree.

        Returns
        -------
        int
            The sum of the nodes in the tree.
        """
        self.update_tree_length()
        return sum(self.tree_lengths.values())

    ###############################################################################
    # Tree assignment functions
    ###############################################################################

    def match_new_atom(self, atom, mol, max_depth=0, attention_threshold=10):
        """
        Matches a given atom in the decision tree to a node

        Parameters
        ----------
        atom : int
            atom index
        mol : rdkit.Chem.rdchem.Mol
            molecule in which the atom is located

        Returns
        -------
        list[node, list[node]]
            The final matched node and the path to the node
        """
        current_correct_node = self.root
        node_path = [self.root]
        connected_atoms = []
        total_attention = 0
        if max_depth == 0:
            max_depth = self.max_depth
        for i in range(max_depth):
            try:
                if i == 0:
                    possible_new_atom_features = [AtomFeatures.atom_features_from_molecule_w_connection_info(mol, atom)]
                    possible_new_atom_idxs = [atom]
                elif (
                    i == 1 and mol.GetAtomWithIdx(atom).GetSymbol() == "H"
                ):  # special case for H -> only connect to heavy atom and ignore H
                    h_connected_heavy_atom = (
                        mol.GetAtomWithIdx(atom).GetNeighbors()[0].GetIdx()
                    )  # get connected heavy atom
                    possible_new_atom_features = [
                        AtomFeatures.atom_features_from_molecule_w_connection_info(mol, h_connected_heavy_atom)
                    ]
                    possible_new_atom_idxs = [h_connected_heavy_atom]
                    connected_atoms = []  # reset connected atoms so that heavy atom is 0 and H is ignored
                else:
                    possible_new_atom_features, possible_new_atom_idxs = get_possible_atom_features(
                        mol, [int(x) for x in connected_atoms]
                    )
                found_match = False
                for current_node in current_correct_node.children:
                    if found_match:
                        break
                    for possible_atom_feature, possible_atom_idx in zip(
                        possible_new_atom_features, possible_new_atom_idxs
                    ):
                        if possible_atom_feature in current_node.atoms:
                            current_correct_node = current_node
                            connected_atoms.append(possible_atom_idx)
                            node_path.append(current_node)
                            total_attention += current_node.attention
                            found_match = True
                            break
                if total_attention > attention_threshold:
                    break
            except Exception as e:
                print(e)
                break
        return (current_correct_node.result, node_path)

    def match_molecule_atoms(
        self,
        mol,
        norm_method="std_weighted",
        max_depth=0,
        attention_threshold=10,
        verbose=False,
        return_raw=False,
        return_std=False,
        return_match_depth=False,
        default_std_value=0.1,
    ):
        """
        Matches all atoms in a molecule to the tree. And returns normalized results.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            molecule to be matched
        norm_method : str, optional
            The method to be used for normalization, by default "std_weighted", options are "std_weighted", "symmetric", "none"
        max_depth : int, optional
            The maximum depth to be used for matching, by default 0 (=unlimited)


        Returns
        -------
        list[float]
            A list with the normalized matched charges from the tree.
        """
        return_list = []
        tree_raw_charges = []
        tree_charge_std = []
        tree_match_depth = []
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            try:
                result, node_path = self.match_new_atom(
                    atom_idx, mol, max_depth=max_depth, attention_threshold=attention_threshold
                )
                tree_raw_charges.append(float(result))
                tmp_tree_std = float(node_path[-1].stdDeviation)
                tree_charge_std.append(tmp_tree_std if tmp_tree_std > 0 else default_std_value)
                tree_match_depth.append(len(node_path))
            except Exception as e:
                print(e)
                tree_raw_charges.append(np.NaN)
                tree_charge_std.append(np.NaN)
                tree_match_depth.append(-1)

        if verbose:
            print("Tree raw charges: {}".format(tree_raw_charges))
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
            print("Tree normalized charges: {}".format(return_list))
        return_data = [return_list]
        if return_raw:
            return_data.append(tree_raw_charges)
        if return_std:
            return_data.append(tree_charge_std)
        if return_match_depth:
            return_data.append(tree_match_depth)
        return return_data

    def _match_dataset_dev(self, mol_sup, stop=1000000, max_depth=0, attention_threshold=10):
        """
        Matches all molecules in a dataset to the tree.

        Parameters
        ----------
        mol_sup : rdkit.Chem.rdchem.MolSupplier
            The dataset to be matched.
        stop : int, optional
            A early stop for development, by default 1000000

        Returns
        -------
        pd.DataFrame
            A dataframe with the results of the match.
        """
        i = 0
        tot_list = []
        for mol in tqdm(mol_sup):
            if i >= stop:
                break
            charges = self.match_molecule_atoms(mol, i, max_depth=max_depth, attention_threshold=attention_threshold)[0]
            elements = [x.GetSymbol() for x in mol.GetAtoms()]
            atom_idxs = [x.GetIdx() for x in mol.GetAtoms()]
            mol_idx = [i] * len(charges)
            tot_list.append(
                pd.DataFrame({"mol_idx": mol_idx, "atom_idx": atom_idxs, "element": elements, "charge": charges})
            )
            i += 1
        return pd.concat(tot_list)

    def _match_dataset_with_indices_dev(self, mol_sup, indices, verbose=True):
        i = 0
        tot_list = []
        for mol in tqdm(mol_sup) if verbose else mol_sup:
            if i in indices:
                tot_list.extend(self.match_molecule_atoms(mol, i, max_depth=self.max_depth))
                i += 1
            else:
                i += 1
        return pd.DataFrame(tot_list)
