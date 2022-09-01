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
        sanitize=False,
    ):
        # Get sdfs of all molecules
        self.sdf_suplier = mols_from_sdf(sdf_file=sdf_suplier)

        #  get the dataset with all the information
        self.original_df = pd.read_csv(
            df_path,
            usecols=[
                "atomtype",
                "smiles",
                "mol_index",
                "idx_in_mol",
                "node_attentions",
                "prediction",
                "truth",
            ],
            nrows=nrows,
        )
        if sanitize:
            self._clean_molecule_indices_in_df()
        #  split the dataset into build and test
        random.seed(seed)
        test_set = random.sample(
            self.original_df.mol_index.unique().tolist(),
            int(len(self.original_df.mol_index.unique()) * data_split),
        )
        self.df = self.original_df.loc[~self.original_df.mol_index.isin(test_set)].copy()
        self.test_df = self.original_df.loc[self.original_df.mol_index.isin(test_set)].copy()
        num_train_mols = len(self.df.mol_index.unique())
        num_test_mols = len(self.test_df.mol_index.unique())
        print(f"Number of train mols: {num_train_mols}")
        print(f"Number of test mols: {num_test_mols}")
        delattr(self, "original_df")
        self.df["node_attentions"] = self.df["node_attentions"].apply(eval)
        self.df["node_attentions"] = self.df["node_attentions"].apply(
            lambda x: (np.array(x) / sum(x)).tolist()
        )  # normalize node attentions
        self.attention_percentage = attention_percentage
        self.df["connected_atoms"] = self.df.apply(lambda x: [], axis=1)
        self.df["connected_attentions"] = self.df.apply(lambda x: [], axis=1)

        self.num_layers_to_build = num_layers_to_build

        self.root = DevelopNode()
        self.new_root = node(level=0)

    def _clean_molecule_indices_in_df(self):
        molecule_idx_in_df = self.original_df.mol_index.unique().tolist()
        for mol_idx in molecule_idx_in_df:
            number_of_atoms_in_mol_df = len(self.original_df.loc[self.original_df.mol_index == mol_idx])
            number_of_atoms_in_mol_sdf = self.sdf_suplier[mol_idx].GetNumAtoms()
            if number_of_atoms_in_mol_df > number_of_atoms_in_mol_sdf:
                if number_of_atoms_in_mol_df <= self.sdf_suplier[mol_idx + 1].GetNumAtoms():
                    self.original_df.loc[self.original_df.mol_index >= mol_idx, "mol_index"] += 1
                elif number_of_atoms_in_mol_df <= self.sdf_suplier[mol_idx + 2].GetNumAtoms():
                    self.original_df.loc[self.original_df.mol_index >= mol_idx, "mol_index"] += 2
                elif number_of_atoms_in_mol_df <= self.sdf_suplier[mol_idx + 3].GetNumAtoms():
                    self.original_df.loc[self.original_df.mol_index >= mol_idx, "mol_index"] += 3
                else:
                    raise ValueError(f"Molecule {mol_idx} could not be sanitized")
            else:
                pass

    def try_to_add_new_node(self, line, matrix, mol, layer):
        connected_atoms = line["connected_atoms"]
        truth_value = line["truth"]
        connected_attentions = line["connected_attentions"]
        connected_atom_max_attention = line["connected_atom_max_attention"]

        try:
            current_node = self.root
            for i in range(layer):
                for child in current_node.children:
                    if child.atom_features == line[i]:
                        current_node = child
                        break
            matching_child = None
            new_atom_idx = None
            for child in current_node.children:
                possible_new_atoms_idx = list(
                    get_possible_connected_new_atom(matrix, indices=np.array(connected_atoms))
                )
                possible_new_atoms = [
                    (
                        i,
                        AtomFeatures.from_molecule(
                            mol,
                            int(i),
                            connected_to=get_connected_neighbor(
                                matrix=matrix, idx=i, indices=np.array(connected_atoms)
                            ),
                        ),
                    )
                    for i in possible_new_atoms_idx
                ]
                for idx, possible_new_atom in possible_new_atoms:
                    if possible_new_atom == child.atom_features:
                        matching_child = child
                        new_atom_idx = idx
                        break
                if matching_child is not None:
                    break
            if matching_child is not None:
                matching_child.truth_values = np.append(matching_child.truth_values, truth_value)
                attention_value = line["node_attentions"][new_atom_idx]
                matching_child.attention_values = np.append(matching_child.attention_values, attention_value)
                line[i] = matching_child.atom_features
                connected_atoms.append(new_atom_idx)
                connected_attentions.append(attention_value)
                return matching_child.atom_features
            else:
                current_atom_idx = int(line["connected_atom_max_attention_idx"])
                connectedTo = get_connected_neighbor(
                    matrix=matrix,
                    idx=current_atom_idx,
                    indices=np.array(connected_atoms),
                )
                new_atom_feature = AtomFeatures.from_molecule(mol, current_atom_idx, connected_to=connectedTo)
                new_node = DevelopNode(
                    atom_features=new_atom_feature,
                    level=layer + 1,
                    truth_values=truth_value,
                    attention_values=connected_atom_max_attention,
                    parent_attention=current_node.parent_attention + np.max(current_node.attention_values),
                )
                current_node.add_child(new_node)
                line[i] = new_node.atom_features
                connected_atoms.append(current_atom_idx)
                connected_attentions.append(connected_atom_max_attention)
            return new_atom_feature
        except Exception as e:
            raise e

    def _create_feature_0_in_table(self, line):
        mol = self.sdf_suplier[line["mol_index"]]
        idx = line["idx_in_mol"]
        atom = AtomFeatures.from_molecule(mol, idx)
        line["connected_atoms"] = line["connected_atoms"].append(idx)
        line["connected_attentions"] = line["connected_attentions"].append(line["node_attentions"][idx])
        return atom

    def create_tree_level_0(self):
        self.df[0] = self.df.apply(self._create_feature_0_in_table, axis=1)
        self.df.apply(
            lambda x: self.root.add_node(
                [
                    DevelopNode(
                        atom_features=x[0],
                        level=1,
                        truth_values=x["truth"],
                        attention_values=x["connected_attentions"][0],
                    )
                ]
            ),
            axis=1,
        )
        self.matrices = []
        for mol in tqdm(self.sdf_suplier):
            matrix = np.array(Chem.GetAdjacencyMatrix(mol), np.bool_)
            np.fill_diagonal(matrix, True)
            self.matrices.append(matrix)
        self.df["total_connected_attention"] = self.df["connected_attentions"].apply(np.sum)
        print(f"{datetime.datetime.now()}\tLayer 0 done")

    def build_tree(self):
        self.df_work = self.df.copy(deep=True)
        for i in range(1, self.num_layers_to_build):
            self.df_work["total_connected_attention"] = self.df_work["connected_attentions"].apply(np.sum)

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
                lambda x: self.try_to_add_new_node(
                    x,
                    self.matrices[int(x["mol_index"])],
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
