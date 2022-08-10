import pickle
from rdkit import Chem
import numpy as np
import pandas as pd
import random
import datetime

from serenityff.charge.tree.atom_features import atom_features
from serenityff.charge.tree_develop.develop_node import develop_node
from serenityff.charge.tree.node import node
from serenityff.charge.tree.tree_utils import (
    get_connected_atom_with_max_attention,
    get_possible_connected_new_atom,
    get_connected_neighbor,
    create_new_node_from_develop_node,
)


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
        # Get sdfs of all molecules
        self.sdf_suplier = Chem.SDMolSupplier(sdf_suplier, removeHs=False)

        # get the dataset with all the information
        self.original_df = pd.read_csv(
            df_path,
            usecols=["atomtype", "smiles", "mol_index", "idx_in_mol", "node_attentions", "prediction", "truth"],
            nrows=nrows,
        )

        # split the dataset into build and test
        random.seed(seed)
        test_set = random.sample(
            self.original_df.mol_index.unique().tolist(), int(len(self.original_df.mol_index.unique()) * data_split)
        )
        self.df = self.original_df.loc[~self.original_df.mol_index.isin(test_set)].copy()
        self.test_df = self.original_df.loc[self.original_df.mol_index.isin(test_set)].copy()

        # prepare df to build the tree
        self.df["node_attentions"] = self.df["node_attentions"].apply(eval)
        self.df["node_attentions"] = self.df["node_attentions"].apply(
            lambda x: (np.array(x) / sum(x)).tolist()
        )  # normalize node attentions
        self.attention_percentage = attention_percentage
        self.df["connected_atoms"] = self.df.apply(lambda x: [], axis=1)
        self.df["connected_attentions"] = self.df.apply(lambda x: [], axis=1)

        self.num_layers_to_build = num_layers_to_build

        self.root = develop_node()
        self.new_root = node()

    def try_to_add_new_node(self, line, mol, layer):
        try:
            current_node = self.root
            for i in range(layer):
                for child in current_node.children:
                    if child.atom == line[i]:
                        current_node = child
                        break
            matching_child = None
            new_atom_idx = None
            for child in current_node.children:
                possible_new_atoms_idx = get_possible_connected_new_atom(mol, connected_atoms=line["connected_atoms"])
                possible_new_atoms = [
                    (
                        i,
                        atom_features(
                            mol,
                            i,
                            connectedTo=get_connected_neighbor(
                                atom_idx=i, mol=mol, connected_atoms=line["connected_atoms"]
                            ),
                        ),
                    )
                    for i in possible_new_atoms_idx
                ]
                for idx, possible_new_atom in possible_new_atoms:
                    if possible_new_atom == child.atom:
                        matching_child = child
                        new_atom_idx = idx
                        break
                if matching_child is not None:
                    break
            if matching_child is not None:
                matching_child.data = np.append(matching_child.data, line["truth"])
                attention_value = line["node_attentions"][new_atom_idx]
                matching_child.attention_data = np.append(matching_child.attention_data, attention_value)
                line[i] = matching_child.atom
                line["connected_atoms"].append(new_atom_idx)
                line["connected_attentions"].append(attention_value)
                return matching_child.atom
            else:
                current_atom_idx = int(line["connected_atom_max_attention_idx"])
                connectedTo = get_connected_neighbor(
                    atom_idx=current_atom_idx, mol=mol, connected_atoms=line["connected_atoms"]
                )
                new_atom_feature = atom_features(mol, current_atom_idx, connectedTo=connectedTo)
                new_node = develop_node(
                    atom=new_atom_feature,
                    level=layer + 1,
                    data=line["truth"],
                    attention_data=line["connected_atom_max_attention"],
                    parent_attention=current_node.parent_attention + np.max(current_node.attention_data),
                )
                current_node.add_child(new_node)
                line[i] = new_node.atom
                line["connected_atoms"].append(current_atom_idx)
                line["connected_attentions"].append(line["node_attentions"][current_atom_idx])
            return new_atom_feature
        except Exception:
            # raise e
            return None

    def _create_feature_0_in_table(self, line):
        mol = self.sdf_suplier[line["mol_index"]]
        idx = line["idx_in_mol"]
        atom = atom_features(mol, idx)
        line["connected_atoms"] = line["connected_atoms"].append(idx)
        line["connected_attentions"] = line["connected_attentions"].append(line["node_attentions"][idx])
        return atom

    def create_tree_level_0(self):
        self.df[0] = self.df.apply(self._create_feature_0_in_table, axis=1)
        self.df.apply(
            lambda x: self.root.add_node(
                [develop_node(atom=x[0], level=1, data=x["truth"], attention_data=x["connected_attentions"][0])]
            ),
            axis=1,
        )
        self.df["total_connected_attention"] = self.df["connected_attentions"].apply(np.sum)
        print(f"{datetime.datetime.now()}\tLayer 0 done")

    def _work_through_df_and_build_tree_step(self, df_local, layer, attention_percentage):
        df_local["total_connected_attention"] = df_local["connected_attentions"].apply(np.sum)
        df_work = df_local[df_local["total_connected_attention"] < attention_percentage].copy(deep=True)

        df_work["connected_atom_max_attention"] = df_work.apply(
            lambda x: get_connected_atom_with_max_attention(
                atom_idx=x["idx_in_mol"],
                layer=0,
                mol=self.sdf_suplier[x["mol_index"]],
                node_attentions=x["node_attentions"],
            )[1],
            axis=1,
        )
        df_work["connected_atom_max_attention_idx"] = df_work.apply(
            lambda x: get_connected_atom_with_max_attention(
                atom_idx=x["idx_in_mol"],
                layer=0,
                mol=self.sdf_suplier[x["mol_index"]],
                node_attentions=x["node_attentions"],
            )[0],
            axis=1,
        )
        df_work.sort_values(by="connected_atom_max_attention", ascending=False, inplace=True)

        df_work[layer] = df_work.apply(
            lambda x: self.try_to_add_new_node(x, self.sdf_suplier[x["mol_index"]], layer), axis=1
        )
        return df_work

    def build_tree(self):
        self.df_work = self.df.copy(deep=True)
        for i in range(1, self.num_layers_to_build):
            self.df_work["total_connected_attention"] = self.df_work["connected_attentions"].apply(np.sum)

            self.df_work = self.df_work.loc[self.df_work["total_connected_attention"] < self.attention_percentage]

            # self.df_work["connected_atom_max_attention"] = self.df_work.apply(lambda x: get_connected_atom_with_max_attention(atom_idx=int(x["idx_in_mol"]), layer=0, mol=self.sdf_suplier[int(x["mol_index"])], node_attentions=x["node_attentions"])[1], axis=1)
            # self.df_work["connected_atom_max_attention_idx"] = self.df_work.apply(lambda x: get_connected_atom_with_max_attention(atom_idx=int(x["idx_in_mol"]), layer=0, mol=self.sdf_suplier[int(x["mol_index"])], node_attentions=x["node_attentions"])[0], axis=1)

            self.df_work["connected_atom_max_attention_idx"], self.df_work["connected_atom_max_attention"] = zip(
                *self.df_work.apply(
                    lambda x: get_connected_atom_with_max_attention(
                        atom_idx=int(x["idx_in_mol"]),
                        layer=0,
                        mol=self.sdf_suplier[int(x["mol_index"])],
                        node_attentions=x["node_attentions"],
                    ),
                    axis=1,
                )
            )
            self.df_work.sort_values(by="connected_atom_max_attention", ascending=False, inplace=True)

            self.df_work[i] = self.df_work.apply(
                lambda x: self.try_to_add_new_node(x, self.sdf_suplier[int(x["mol_index"])], i), axis=1
            )
            print(f"{datetime.datetime.now()}\tLayer {i} done", flush=True)

        self.root.update_average()

    def convert_tree_to_node(self):
        self.new_root = create_new_node_from_develop_node(self.root, self.new_root)

    def pickle_tree(self, file_name):
        with open(file_name, "wb") as f:
            pickle.dump(self.root, f)

    def write_tree_to_file(self, file_name):
        self.new_root.to_file(file_name)
