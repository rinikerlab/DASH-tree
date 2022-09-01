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
        nrows: int,
        read_engine: str = "python",
        attention_percentage: float = 0.99,
        data_split: float = 0.2,
        seed: int = 42,
        num_layers_to_build=24,
        sanitize=False,
        verbose=False,
    ):
        if verbose:
            print(f"{datetime.datetime.now()}\tInitializing Tree_constructor")
        self.sdf_suplier = mols_from_sdf(sdf_suplier)
        self.sdf_suplier_wo_h = mols_from_sdf(sdf_suplier, removeHs=True)

        if verbose:
            print(f"{datetime.datetime.now()}\tMols imported, starting df import")

        self.original_df = (
            pd.read_csv(
                df_path,
                usecols=[
                    "atomtype",
                    "mol_index",
                    "idx_in_mol",
                    "node_attentions",
                    "truth",
                ],
                engine="pyarrow",
            )
            if read_engine == "pyarrow"
            else pd.read_csv(
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
        )

        if verbose:
            print(f"{datetime.datetime.now()}\tdf imported, starting data spliting")

        random.seed(seed)
        unique_mols = self.original_df.mol_index.unique().tolist()
        test_set = random.sample(
            unique_mols,
            int(len(unique_mols) * data_split),
        )
        self.df = self.original_df.loc[~self.original_df.mol_index.isin(test_set)].copy()
        self.test_df = self.original_df.loc[self.original_df.mol_index.isin(test_set)].copy()

        delattr(self, "original_df")
        self.df["node_attentions"] = self.df["node_attentions"].apply(eval)

        h, c, t, n = [], [], [], []

        if verbose:
            print(f"{datetime.datetime.now()}\tStarting table filling")

        self.tempmatrix = Chem.GetAdjacencyMatrix(self.sdf_suplier[0])

        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
            n.append(np.array(row["node_attentions"]) / sum(row["node_attentions"]))
            h.append(self._get_hydrogen_connectivity(row))
            c.append(([] if row["atomtype"] == "H" else [row["idx_in_mol"]]))
            t.append(row["node_attentions"][row["idx_in_mol"]])

        self.df["h_connectivity"] = h
        self.df["connected_atoms"] = c
        self.df["total_connected_attention"] = t
        self.df["node_attentions"] = n

        del h, c, n, t

        delattr(self, "tempmatrix")

        self.attention_percentage = attention_percentage
        self.num_layers_to_build = num_layers_to_build
        self.feature_dict = dict()
        self.root = DevelopNode()
        self.new_root = node(level=0)

        print(f"Number of train mols: {len(self.df.mol_index.unique())}")
        print(f"Number of test mols: {len(self.test_df.mol_index.unique())}")

    def _get_hydrogen_connectivity(self, line) -> int:
        if line["idx_in_mol"] == 0:
            self.tempmatrix = Chem.GetAdjacencyMatrix(self.sdf_suplier[line["mol_index"]])
        if line["atomtype"] == "H":
            return int(np.where(self.tempmatrix[line["idx_in_mol"]])[0].item())
        else:
            return -1

    def _create_atom_features(self, line):
        return AtomFeatures.atom_features_from_molecule_w_connection_info(
            self.sdf_suplier[line["mol_index"]], line["idx_in_mol"]
        )

    def _create_adjacency_matrices(self):
        bonddict = {v: k for k, v in Chem.rdchem.BondType.values.items()}
        print("Creating Adjacency matrices:")
        self.matrices = []
        self.bond_matrices = []
        for mol in tqdm(self.sdf_suplier_wo_h):
            matrix = np.array(Chem.GetAdjacencyMatrix(mol), np.bool_)
            np.fill_diagonal(matrix, True)
            self.matrices.append(matrix)
            matrix = matrix.astype(np.int8)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if i == j:
                        continue
                    if matrix[i][j]:
                        matrix[i][j] = bonddict[mol.GetBondBetweenAtoms(int(i), int(j)).GetBondType()]
            self.bond_matrices.append(matrix)

    def create_tree_level_0(self):
        print("Preparing Dataframe:")
        features = []
        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
            feat = self._create_atom_features(row)
            features.append(feat)
            if row["idx_in_mol"] == 0:
                self.feature_dict[row["mol_index"]] = dict()
                self.feature_dict[row["mol_index"]][row["idx_in_mol"]] = feat[0]
            else:
                self.feature_dict[row["mol_index"]][row["idx_in_mol"]] = feat[0]

            self.root.add_node(
                [
                    DevelopNode(
                        atom_features=feat,
                        level=1,
                        truth_values=row["truth"],
                        attention_values=row["node_attentions"][row["idx_in_mol"]],
                    )
                ]
            )
        self.df[0] = features
        self._create_adjacency_matrices()
        print(f"{datetime.datetime.now()}\tLayer 0 done")

    def _match_atom(self, layer, line):
        current_node = self.root
        for i in range(layer):
            for child in current_node.children:
                if (child.atom_features == line[i]).all():
                    return child

    def _find_matching_child(self, children, matrix, indices, mol_index):
        for child in children:
            possible_new_atoms = []
            for i in get_possible_connected_new_atom(matrix, indices):
                rel, abs = get_connected_neighbor(matrix, i, indices)
                possible_new_atoms.append(
                    (
                        i,
                        np.array([self.feature_dict[mol_index][i], rel, self.bond_matrices[mol_index][abs][i]]),
                    ),
                )
            for idx, possible_new_atom in possible_new_atoms:
                if (possible_new_atom == child.atom_features).all():
                    return child, idx
        return None, None

    def _try_to_add_new_node(self, line, matrix, layer):
        connected_atoms = line["connected_atoms"]
        truth_value = line["truth"]
        connected_atom_max_attention = line["connected_atom_max_attention"]
        indices = np.array(connected_atoms)
        mol_index = line["mol_index"]
        try:
            current_node = self._match_atom(layer, line)
            matching_child, new_atom_idx = self._find_matching_child(
                current_node.children,
                matrix,
                indices,
                mol_index,
            )
            if matching_child is not None:
                matching_child.truth_values = np.append(matching_child.truth_values, truth_value)
                matching_child.attention_values = np.append(
                    matching_child.attention_values,
                    line["node_attentions"][new_atom_idx],
                )
                line.iloc[layer] = matching_child.atom_features
                connected_atoms.append(new_atom_idx)
                return matching_child.atom_features
            else:
                current_atom_idx = int(line["connected_atom_max_attention_idx"])
                rel, abs = get_connected_neighbor(matrix, current_atom_idx, indices)
                new_atom_feature = np.array(
                    [
                        self.feature_dict[mol_index][current_atom_idx],
                        rel,
                        self.bond_matrices[mol_index][current_atom_idx][abs],
                    ],
                    dtype=np.int64,
                )
                new_node = DevelopNode(
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
            raise e

    def _find_matching_child_h(self, children, mol_index, hconnec):
        for child in children:
            possible_new_atom = np.array([self.feature_dict[mol_index][hconnec], -1, -1], dtype=np.int64)

            if (possible_new_atom == child.atom_features).all():
                return child, hconnec
        return None, None

    def _add_node_conn_to_hydrogen(self, line):
        connected_atoms = line["connected_atoms"]
        truth_value = line["truth"]
        mol_index = line["mol_index"]
        connected_atom_max_attention = line["connected_atom_max_attention"]
        try:
            current_node = self._match_atom(1, line)
            matching_child, new_atom_idx = self._find_matching_child_h(
                current_node.children,
                mol_index,
                line["h_connectivity"],
            )
            if matching_child is not None:
                matching_child.truth_values = np.append(matching_child.truth_values, truth_value)
                matching_child.attention_values = np.append(
                    matching_child.attention_values,
                    line["node_attentions"][new_atom_idx],
                )
                line.iloc[1] = matching_child.atom_features
                connected_atoms.append(new_atom_idx)
                return matching_child.atom_features
            else:
                current_atom_idx = int(line["connected_atom_max_attention_idx"])
                new_atom_feature = np.array([self.feature_dict[mol_index][current_atom_idx], -1, -1], dtype=np.int64)
                new_node = DevelopNode(
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

    def _build_layer_1(self):
        ci, ca = [], []
        self.df_work["total_connected_attention"] = [
            np.sum(row["node_attentions"][row["connected_atoms"]]) for _, row in self.df_work.iterrows()
        ]
        self.df_work = self.df_work.loc[self.df_work["total_connected_attention"] < self.attention_percentage]
        for _, row in self.df_work.iterrows():
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
        self.df_work["connected_atom_max_attention_idx"] = ci
        self.df_work["connected_atom_max_attention"] = ca
        self.df_work.sort_values(by="connected_atom_max_attention", ascending=False, inplace=True)
        self.df_work[1] = [
            self._add_node_conn_to_hydrogen(
                row,
            )
            if row["atomtype"] == "H"
            else self._try_to_add_new_node(
                row,
                self.matrices[row["mol_index"]],
                1,
            )
            for _, row in self.df_work.iterrows()
        ]

    def build_tree(self):
        self.df_work = self.df.copy(deep=True)
        self._build_layer_1()
        print(f"{datetime.datetime.now()}\tLayer {1} done", flush=True)
        for layer in range(2, self.num_layers_to_build):
            self.df_work["total_connected_attention"] = [
                np.sum(row["node_attentions"][row["connected_atoms"]]) for _, row in self.df_work.iterrows()
            ]
            self.df_work = self.df_work.loc[self.df_work["total_connected_attention"] < self.attention_percentage]
            ai, a = [], []
            for _, row in self.df_work.iterrows():
                x, y = get_connected_atom_with_max_attention(
                    matrix=self.matrices[int(row["mol_index"])],
                    attentions=np.array(row["node_attentions"]),
                    indices=np.array(row["connected_atoms"]),
                )
                ai.append(x)
                a.append(y)

            self.df_work["connected_atom_max_attention_idx"] = ai
            self.df_work["connected_atom_max_attention"] = a
            self.df_work = self.df_work.loc[self.df_work["connected_atom_max_attention_idx"].notnull()]
            if self.df_work.shape[0] == 0:
                break
            self.df_work.sort_values(by="connected_atom_max_attention", ascending=False, inplace=True)
            self.df_work[layer] = [
                self._try_to_add_new_node(row, self.matrices[row["mol_index"]], layer)
                for _, row in self.df_work.iterrows()
            ]
            print(f"{datetime.datetime.now()}\tLayer {layer} done", flush=True)

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
