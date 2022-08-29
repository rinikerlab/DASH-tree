import datetime
import pickle
import random

import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from serenityff.charge.gnn.utils import mols_from_sdf
from serenityff.charge.tree.atom_features import AtomFeatures
from serenityff.charge.tree.node import node
from serenityff.charge.tree.tree_utils import (
    create_new_node_from_develop_node,
    get_connected_atom_with_max_attention,
    get_connected_neighbor,
    get_possible_connected_new_atom,
)
from serenityff.charge.tree_develop.develop_node import DevelopNode


class Tree_constructor:
    def __init__(
        self,
        df_path: str,
        sdf_suplier: str,
        nrows: int = 1000,
        attention_percentage: float = 0.99,
        data_split: float = 0.2,
        seed: int = 42,
        num_layers_to_build=24,
    ):
        self.sdf_suplier = mols_from_sdf(sdf_suplier)
        self.sdf_suplier_path = sdf_suplier

        self.original_df = pd.read_csv(
            df_path,
            usecols=[
                "atomtype",
                "mol_index",
                "idx_in_mol",
                "node_attentions",
                "truth",
            ],
            nrows=nrows,
        )

        random.seed(seed)
        test_set = random.sample(
            self.original_df.mol_index.unique().tolist(),
            int(len(self.original_df.mol_index.unique()) * data_split),
        )
        self.df = self.original_df.loc[~self.original_df.mol_index.isin(test_set)].copy()
        self.test_df = self.original_df.loc[self.original_df.mol_index.isin(test_set)].copy()

        delattr(self, "original_df")
        self.df["node_attentions"] = self.df["node_attentions"].apply(eval)
        self.df["node_attentions"] = self.df["node_attentions"].apply(lambda x: (np.array(x) / sum(x)))
        self.attention_percentage = attention_percentage
        self.df["connected_atoms"] = self.df.apply(lambda x: [x["idx_in_mol"]], axis=1)
        self.df["h_connectivity"] = self.df.apply(lambda x: self._get_hydrogen_connectivity(x), axis=1)
        self.df["total_connected_attention"] = self.df.apply(
            lambda x: x["node_attentions"][x["idx_in_mol"]], axis=1, dtype=pd.Int16Dtype
        )

        self.num_layers_to_build = num_layers_to_build
        self.root = DevelopNode()
        self.new_root = node(level=0)

        print(f"Number of train mols: {len(self.df.mol_index.unique())}")
        print(f"Number of test mols: {len(self.test_df.mol_index.unique())}")

    def _get_hydrogen_connectivity(self, line) -> int:
        if line["atomtype"] == "H":
            return int(
                np.where(
                    Chem.GetAdjacencyMatrix(mols_from_sdf(self.sdf_suplier_path, removeHs=False)[line["mol_index"]])[
                        line["idx_in_mol"]
                    ]
                )[0].item()
            )
        else:
            return np.NAN

    def _create_atom_features(self, line):
        return AtomFeatures.from_molecule(self.sdf_suplier[line["mol_index"]], line["idx_in_mol"])

    def _create_adjacency_matrices(self):
        print("Creating Adjacency matrices:")
        self.matrices = []
        for mol in tqdm(mols_from_sdf(self.sdf_suplier_path, removeHs=True)):
            matrix = np.array(Chem.GetAdjacencyMatrix(mol), np.bool_)
            np.fill_diagonal(matrix, True)
            self.matrices.append(matrix)

    def create_tree_level_0(self):
        print("Preparing Dataframe:")
        self.df[0] = self.df.apply(self._create_atom_features, axis=1)
        self.df.apply(
            lambda x: self.root.add_node(
                [
                    DevelopNode(
                        atom_features=x[0],
                        level=1,
                        truth_values=x["truth"],
                        attention_values=x["node_attentions"][x["idx_in_mol"]],
                    )
                ]
            ),
            axis=1,
        )
        self._create_adjacency_matrices()
        print(f"{datetime.datetime.now()}\tLayer 0 done")

    def _match_atom(self, layer, line):
        current_node = self.root
        for i in range(layer):
            for child in current_node.children:
                if child.atom_features == line[i]:
                    return child

    def _find_matching_child(self, children, matrix, indices, mol):
        for child in children:
            possible_new_atoms = [
                (
                    i,
                    AtomFeatures.from_molecule(
                        mol,
                        int(i),
                        connected_to=get_connected_neighbor(matrix, i, indices),
                    ),
                )
                for i in get_possible_connected_new_atom(matrix, indices)
            ]
            for idx, possible_new_atom in possible_new_atoms:
                if possible_new_atom == child.atom_features:
                    return child, idx

    def _try_to_add_new_node(self, line, matrix, mol, layer):
        connected_atoms = line["connected_atoms"]
        truth_value = line["truth"]
        connected_atom_max_attention = line["connected_atom_max_attention"]
        indices = np.array(connected_atoms)
        try:
            current_node = self._match_atom(layer, line)
            matching_child, new_atom_idx = self._find_matching_child(current_node.children, matrix, indices, mol)
            if matching_child is not None:
                matching_child.truth_values = np.append(matching_child.truth_values, truth_value)
                matching_child.attention_values = np.append(
                    matching_child.attention_values, line["node_attentions"][new_atom_idx]
                )
                line[layer] = matching_child.atom_features
                connected_atoms.append(new_atom_idx)
                return matching_child.atom_features
            else:
                current_atom_idx = int(line["connected_atom_max_attention_idx"])
                new_atom_feature = AtomFeatures.from_molecule(
                    mol,
                    current_atom_idx,
                    connected_to=get_connected_neighbor(
                        matrix=matrix,
                        idx=current_atom_idx,
                        indices=indices,
                    ),
                )
                new_node = DevelopNode(
                    atom_features=new_atom_feature,
                    level=layer + 1,
                    truth_values=truth_value,
                    attention_values=connected_atom_max_attention,
                    parent_attention=current_node.parent_attention + np.max(current_node.attention_values),
                )
                current_node.add_child(new_node)
                line[layer] = new_node.atom_features
                connected_atoms.append(current_atom_idx)
            return new_atom_feature
        except Exception as e:
            raise e

    def _add_new_node_level_1(self, line, matrix, mol):
        # TODO: all
        return

    def _build_layer_1(self):
        self.df_work = self.df_work.loc[self.df_work["total_connected_attention"] < self.attention_percentage]
        (self.df_work["connected_atom_max_attention_idx"], self.df_work["connected_atom_max_attention"],) = zip(
            *self.df_work.apply(
                lambda x: (
                    int(x["h_connectivity"]),
                    x["node_attentions"][int(x["h_connectivity"])],
                )
                if x["atomtype"] == "H"
                else get_connected_atom_with_max_attention(
                    matrix=self.matrices[int(x["mol_index"])],
                    attentions=np.array(x["node_attentions"]),
                    indices=np.array(x["connected_atoms"]),
                ),
                axis=1,
            )
        )
        self.df_work.sort_values(by="connected_atom_max_attention", ascending=False, inplace=True)
        self.df_work[1] = self.df_work.apply(
            lambda x: self._try_to_add_new_node(
                x,
                self.matrices[x["mol_index"]],
                self.sdf_suplier[int(x["mol_index"])],
                1,
            ),
            axis=1,
        )

    def build_tree(self):
        self.df_work = self.df.copy(deep=True)
        for i in range(1, self.num_layers_to_build):
            if i == 1:
                self._build_layer_1()
            else:
                self.df_work["total_connected_attention"] = self.df_work.apply(
                    lambda x: np.sum(x["node_attentions"][x["connected_atoms"]]), axis=1
                )

                self.df_work = self.df_work.loc[self.df_work["total_connected_attention"] < self.attention_percentage]

                (self.df_work["connected_atom_max_attention_idx"], self.df_work["connected_atom_max_attention"],) = zip(
                    *self.df_work.apply(
                        lambda x: get_connected_atom_with_max_attention(
                            matrix=self.matrices[int(x["mol_index"])],
                            attentions=np.array(x["node_attentions"]),
                            indices=np.array(x["connected_atoms"]),
                        ),
                        axis=1,
                    )
                )
                self.df_work.sort_values(by="connected_atom_max_attention", ascending=False, inplace=True)
                self.df_work[i] = self.df_work.apply(
                    lambda x: self._try_to_add_new_node(
                        x,
                        self.matrices[x["mol_index"]],
                        self.sdf_suplier[int(x["mol_index"])],
                        i,
                    ),
                    axis=1,
                )
            print(f"{datetime.datetime.now()}\tLayer {i} done", flush=True)

        self.root.update_average()

    def convert_tree_to_node(self, delDevelop=False):
        self.new_root = create_new_node_from_develop_node(self.root)
        if delDevelop:
            del self.root
            self.root = None

    def calculate_tree_length(self):
        self.tree_length = self.new_root.calculate_tree_length()

    def pickle_tree(self, file_name):
        with open(file_name, "wb") as f:
            pickle.dump(self.root, f)

    def write_tree_to_file(self, file_name):
        self.new_root.to_file(file_name)
