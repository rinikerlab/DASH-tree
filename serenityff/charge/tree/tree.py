import glob
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

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

    def from_folder(self, folder_path: str, verbose=False):
        self.root = node(level=0)
        all_files_in_folder = glob.glob(folder_path + "/tree*.csv")
        if verbose:
            print("Found {} files in folder".format(len(all_files_in_folder)))
        for file_path in all_files_in_folder:
            try:
                new_node = node(level=0)
                new_node.from_file(file_path)
                self.root.children.append(new_node)
                if verbose:
                    print("Added {}".format(file_path))
            except Exception as e:
                print(e)
                continue
        self.hasData = True

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

    def match_new_atom(self, atom, mol, max_depth=0):
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
        if max_depth == 0:
            max_depth = self.max_depth
        for i in range(max_depth):
            try:
                if i == 0:
                    possible_new_atom_features = [AtomFeatures.atom_features_from_molecule_w_connection_info(mol, atom)]
                    possible_new_atom_idxs = [atom]
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
                            found_match = True
                            break
            except Exception as e:
                print(e)
                break
        return (current_correct_node.result, node_path)

    def match_molecule_atoms(self, mol, norm_method="std_weighted", max_depth=0, verbose=False):
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
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            try:
                result, node_path = self.match_new_atom(atom_idx, mol, max_depth=max_depth)
                tree_raw_charges.append(float(result))
                tree_charge_std.append(float(node_path[-1].stdDeviation))
            except Exception as e:
                print(e)
                tree_raw_charges.append(np.NaN)
                tree_charge_std.append(np.NaN)

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
        return return_list

    def _match_molecules_atoms_dev(self, mol, mol_idx, max_depth=0):
        """
        Matches all atoms in a molecule to the tree. ANd returns multiple normalized results.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            molecule to be matched
        mol_idx : int
            index of the molecule in the dataset

        Returns
        -------
        list[dict[]]
            A dict for every atom with the atom properties and the result of the match.
        """
        return_list = []
        mbis_charges = mol.GetProp("MBIS_CHARGES").split("|")
        for atom in mol.GetAtoms():
            return_dict = {}
            atom_idx = atom.GetIdx()
            return_dict["mol_idx"] = int(mol_idx)
            return_dict["atom_idx"] = int(atom_idx)
            return_dict["atomtype"] = atom.GetSymbol()
            return_dict["truth"] = float(mbis_charges[atom_idx])
            try:
                result, node_path = self.match_new_atom(atom_idx, mol, max_depth=max_depth)
                return_dict["tree"] = float(result)
                return_dict["tree_std"] = node_path[-1].stdDeviation
            except Exception as e:
                print(e)
                return_dict["tree"] = np.NAN
                return_dict["tree_std"] = np.NAN
            return_list.append(return_dict)

        # normalize_charge symmetric
        tot_charge_truth = np.round(np.sum([float(x) for x in mbis_charges]))
        tot_charge_tree = np.sum([x["tree"] for x in return_list])
        for x in return_list:
            x["tree_norm1"] = x["tree"] - ((tot_charge_tree - tot_charge_truth) / mol.GetNumAtoms())

        # normalize_charge std weighted
        tot_std = np.sum([x["tree_std"] for x in return_list])
        for x in return_list:
            x["tree_norm2"] = x["tree"] + (tot_charge_truth - tot_charge_tree) * (x["tree_std"] / tot_std)

        return return_list

    def _match_dataset_dev(self, mol_sup, stop=1000000, max_depth=0):
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
            tot_list.extend(self._match_molecules_atoms_dev(mol, i, max_depth=max_depth))
            i += 1
        return pd.DataFrame(tot_list)

    def _match_dataset_with_indices_dev(self, mol_sup, indices, verbose=True):
        i = 0
        tot_list = []
        for mol in tqdm(mol_sup) if verbose else mol_sup:
            if i in indices:
                tot_list.extend(self._match_molecules_atoms_dev(mol, i, max_depth=self.max_depth))
                i += 1
            else:
                i += 1
        return pd.DataFrame(tot_list)
