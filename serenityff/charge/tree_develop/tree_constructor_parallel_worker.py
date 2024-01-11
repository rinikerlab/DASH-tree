# import datetime
import traceback
from typing import List
import numpy as np
import pandas as pd

# from serenityff.charge.tree.atom_features import AtomFeatures

from serenityff.charge.tree.tree_utils import (
    get_connected_atom_with_max_attention,
    get_connected_neighbor,
    get_possible_connected_new_atom,
)
from serenityff.charge.tree_develop.develop_node import DevelopNode

try:
    from multiprocessing import Pool
except ImportError:
    print("multiprocessing not installed, parallelization not possible")


class Tree_constructor_parallel_worker:
    """
    This class is the worker for Tree_constructor. It will build the tree for preprocessed data in parallel.
    For more information see Tree_constructor.
    """

    def __init__(
        self,
        df_af_split,
        matrices,
        feature_dict,
        roots,
        bond_matrices,
        num_layers_to_build,
        attention_percentage,
        verbose=False,
        logger=None,
        node_type=DevelopNode,
    ):
        self.df_af_split = df_af_split
        self.matrices = matrices
        self.feature_dict = feature_dict
        self.roots = roots
        self.bond_matrices = bond_matrices
        self.num_layers_to_build = num_layers_to_build
        self.attention_percentage = attention_percentage
        self.verbose = verbose
        self.node_type = node_type
        if logger is None:
            self.loggingBuild = False
        else:
            self.loggingBuild = True
            self.logger = logger

    def _match_atom(self, layer, line):
        current_node = self.roots[line["atom_feature"]]
        for i in range(layer):
            for child in current_node.children:
                if np.all(child.atom_features == line[i]):
                    current_node = child
                    break
        return current_node

    def _find_matching_child(self, children, matrix, bond_matrix, indices, mol_index):
        possible_new_atoms = []
        for i in get_possible_connected_new_atom(matrix, indices):
            rel, abs = get_connected_neighbor(matrix, i, indices)
            tmp_feature = self.feature_dict[mol_index][i]
            tmp_bond_type = bond_matrix[abs][i]
            possible_new_atoms.append(
                (
                    i,
                    [tmp_feature, rel, tmp_bond_type],
                ),
            )
        for child in children:
            for idx, possible_new_atom in possible_new_atoms:
                if possible_new_atom == child.atom_features:
                    return child, idx
        return None, None

    def _try_to_add_new_node(self, line, layer):
        """
        Try to add a new node to the tree. Either by creating a new node as child of the last matching node or by
        adding the properties to a already existing node.

        Parameters
        ----------
        line : pd.Series
            The line of the dataframe that is currently processed (containing the atom information for the current node)
        matrix : dict[int, np.ndarray]
            The adjacency matrices of all molecules
        layer : int
            The current layer of the tree

        Returns
        -------
        new_atom_feature : list
            The atom features of the new node (or the matching node)
        """
        try:
            connected_atoms = line["connected_atoms"]
            truth_value = float(line["truth"])
            connected_atom_max_attention = line["connected_atom_max_attention"]
            indices = np.array(connected_atoms)
            mol_index = line["mol_index"]
            matrix = self.matrices[mol_index]
            bond_matrix = self.bond_matrices[mol_index]
            try:
                current_node = self._match_atom(layer, line)
                matching_child, new_atom_idx = self._find_matching_child(
                    current_node.children,
                    matrix,
                    bond_matrix,
                    indices,
                    mol_index,
                )
                if matching_child is not None:
                    matching_child.truth_values.append(truth_value)
                    matching_child.attention_values.append(float(line["node_attentions"][new_atom_idx]))
                    line.iloc[layer] = matching_child.atom_features
                    connected_atoms.append(new_atom_idx)
                    return matching_child.atom_features
                else:
                    current_atom_idx = int(line["connected_atom_max_attention_idx"])
                    rel, abs = get_connected_neighbor(matrix, current_atom_idx, indices)
                    new_atom_feature = [
                        self.feature_dict[mol_index][current_atom_idx],
                        rel,
                        bond_matrix[current_atom_idx][abs],
                    ]
                    new_node = self.node_type(
                        atom_features=new_atom_feature,
                        level=layer + 1,
                        truth_values=truth_value,
                        attention_values=connected_atom_max_attention,
                        parent_attention=current_node.parent_attention + np.max(current_node.attention_values),
                    )
                    current_node.add_child(new_node)
                    line.iloc[layer] = new_node.atom_features
                    connected_atoms.append(current_atom_idx)
                return new_atom_feature
            except Exception as e:
                if False:  # use this to debug
                    print(f"TODO in AF {line['atom_feature']} - Layer {layer} - {e}")
                # fill connected atoms with all atoms in the molecule
                num_atoms = len(self.feature_dict[mol_index])
                line["connected_atoms"] = list(range(num_atoms))
                return [[-1, -1, -1], -1, -1]
        except Exception as e:
            print(e)
            return [[-1, -1, -1], -1, -1]

    def _find_matching_child_h(self, children, mol_index, hconnec):
        for child in children:
            possible_new_atom = [self.feature_dict[mol_index][hconnec], -1, -1]

            if possible_new_atom == child.atom_features:
                return child, hconnec
        return None, None

    def _add_node_conn_to_hydrogen(self, line):
        connected_atoms = line["connected_atoms"]
        truth_value = float(line["truth"])
        mol_index = line["mol_index"]
        connected_atom_max_attention = float(line["connected_atom_max_attention"])
        try:
            current_node = self._match_atom(1, line)
            matching_child, new_atom_idx = self._find_matching_child_h(
                current_node.children,
                mol_index,
                line["h_connectivity"],
            )
            if matching_child is not None:
                matching_child.truth_values.append(truth_value)
                matching_child.attention_values.append(float(line["node_attentions"][new_atom_idx]))
                line.iloc[1] = matching_child.atom_features
                connected_atoms.append(new_atom_idx)
                return matching_child.atom_features
            else:
                current_atom_idx = int(line["connected_atom_max_attention_idx"])
                new_atom_feature = [self.feature_dict[mol_index][current_atom_idx], -1, -1]
                new_node = self.node_type(
                    atom_features=new_atom_feature,
                    level=1 + 1,
                    truth_values=truth_value,
                    attention_values=connected_atom_max_attention,
                    parent_attention=current_node.parent_attention + np.max(current_node.attention_values),
                )
                current_node.add_child(new_node)
                line.iloc[1] = new_node.atom_features
                connected_atoms.append(current_atom_idx)
            return new_atom_feature
        except Exception as e:
            raise e

    def _build_layer_1(self, af: int, df_work: pd.DataFrame):
        # 1st layer, different from other layers due to the implicit hydrogens
        ci, ca = [], []
        df_work["total_connected_attention"] = [
            np.sum(row["node_attentions"][row["connected_atoms"]]) for _, row in df_work.iterrows()
        ]
        for _, row in df_work.iterrows():
            if row["atomtype"] == "H":
                i, a = (
                    row["h_connectivity"],
                    row["node_attentions"][row["h_connectivity"]],
                )
            else:
                i, a = get_connected_atom_with_max_attention(
                    matrix=self.matrices[int(row["mol_index"])],
                    attentions=np.array(row["node_attentions"]),
                    indices=np.array(row["connected_atoms"]),
                )
            ci.append(i)
            ca.append(a)
        df_work["connected_atom_max_attention_idx"] = ci
        df_work["connected_atom_max_attention"] = ca
        df_work.sort_values(by="connected_atom_max_attention", ascending=False, inplace=True)
        df_work[1] = [
            self._add_node_conn_to_hydrogen(
                row,
            )
            if row["atomtype"] == "H"
            else self._try_to_add_new_node(
                row,
                1,
            )
            for _, row in df_work.iterrows()
        ]

    def _build_tree_single_AF(self, af: int, df_work: pd.DataFrame):
        """
        Main function to build the tree for a single atom feature (AF)
        Loops over the layers and works through the dataframe (sorted by attention) to attach new nodes to the tree

        Parameters
        ----------
        af : int
            Atom feature to build the tree for
        df_work : pd.DataFrame
            Dataframe containing the data for the atom feature

        Returns
        -------
        DevelopNode
            Root node of this AF
        """
        try:
            self._build_layer_1(af=af, df_work=df_work)
            if self.verbose:
                print(f"AF={af} - Layer {1} done", flush=True)
                print(f"children layer 1: {self.roots[af].children}", flush=True)
            for layer in range(2, self.num_layers_to_build):
                try:
                    df_work["total_connected_attention"] = [
                        np.sum(row["node_attentions"][row["connected_atoms"]]) for _, row in df_work.iterrows()
                    ]
                    df_work = df_work.loc[df_work["total_connected_attention"] < self.attention_percentage]
                    ai, a = [], []
                    for _, row in df_work.iterrows():
                        x, y = get_connected_atom_with_max_attention(
                            matrix=self.matrices[int(row["mol_index"])],
                            attentions=np.array(row["node_attentions"]),
                            indices=np.array(row["connected_atoms"]),
                        )
                        ai.append(x)
                        a.append(y)
                    df_work["connected_atom_max_attention_idx"] = ai
                    df_work["connected_atom_max_attention"] = a
                    df_work = df_work.loc[df_work["connected_atom_max_attention_idx"].notnull()]
                    df_work.sort_values(by="connected_atom_max_attention", ascending=False, inplace=True)
                    df_work[layer] = [self._try_to_add_new_node(row, layer) for _, row in df_work.iterrows()]
                except Exception as e:
                    print(f"Error in AF {af} - Layer {layer} - {e}")
                    print(traceback.format_exc())
                    break
            try:
                del ai, a
                for layer in range(2, self.num_layers_to_build):
                    df_work.drop(layer, axis=1, inplace=True)
                df_work.drop("total_connected_attention", axis=1, inplace=True)
                df_work.drop("connected_atom_max_attention_idx", axis=1, inplace=True)
                df_work.drop("connected_atom_max_attention", axis=1, inplace=True)
                df_work.drop("connected_atoms", axis=1, inplace=True)
                del df_work
            except Exception as e:
                print(f"Error in AF {af} - Deleting error - {e}")
            print(f"AF {af} done")
            return self.roots[af]
        except Exception as e:
            print(f"Error in AF {af} - {e}")
            print(traceback.format_exc())
            try:
                return self.roots[af]
            except Exception:
                return DevelopNode()

    def build_tree(self, num_processes: int = 6, af_list: List[int] = None):
        if af_list is None:
            af_list = self.df_af_split.keys()  # list(range(AtomFeatures.get_number_of_features()))
        all_args = [(x, self.df_af_split[x]) for x in af_list]
        res = []
        if num_processes == 1:
            res = [self._build_tree_single_AF(*x) for x in all_args]
        else:
            with Pool(num_processes) as p:
                res = p.starmap(self._build_tree_single_AF, all_args)
        self.root = DevelopNode()
        self.root.children = res
        if self.verbose:
            print(f"tree in build, {len(self.root.children)} children", flush=True)
            print(f"child 0 in build, {self.root.children[0]}", flush=True)
        self.root.update_average()
