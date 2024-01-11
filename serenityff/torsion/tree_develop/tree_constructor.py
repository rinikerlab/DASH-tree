import numpy as np
import pandas as pd
from serenityff.torsion.tree_develop.develop_node import DevelopNode
from serenityff.charge.tree_develop.tree_constructor import Tree_constructor
from serenityff.charge.gnn.utils.rdkit_helper import get_all_torsion_angles
from serenityff.torsion.tree.dash_utils import get_canon_torsion_feature
from serenityff.torsion.tree.tree_utils import get_DASH_tree_from_DEV_tree


class Torsion_tree_constructor(Tree_constructor):
    def __init__(
        self,
        df_path: str,
        sdf_suplier: str,
        nrows: int = None,
        attention_percentage: float = 1000,
        data_split: float = 0.1,
        seed: int = 42,
        num_layers_to_build=24,
        sanitize=False,
        sanitize_charges=False,
        verbose=False,
        loggingBuild=False,
        split_indices_path=None,
        save_cleaned_df_path=None,
        save_feature_dict_path=None,
    ):
        super().__init__(
            df_path,
            sdf_suplier,
            nrows,
            attention_percentage,
            data_split,
            seed,
            num_layers_to_build,
            sanitize,
            sanitize_charges,
            verbose,
            loggingBuild,
            split_indices_path,
            save_cleaned_df_path,
            save_feature_dict_path,
        )
        self.node_type = DevelopNode
        self.create_torsion_df()
        self.create_correct_root_children()

    def create_torsion_df(self):
        df_list = []
        set_of_mol_indices_in_df = set(self.df["mol_index"].values)
        num_torsions_found = 0
        for mol_index, mol in enumerate(self.sdf_suplier):
            if mol_index not in set_of_mol_indices_in_df:
                continue
            torsion_angles_list = get_all_torsion_angles(mol)
            # add torsion number to the torsion_angles_list
            torsion_angles_list = [
                (torsion_indices, torsion_angle, torsion_number)
                for torsion_number, (torsion_indices, torsion_angle) in enumerate(torsion_angles_list)
            ]
            for torsion_indices, torsion_angle, torsion_number in torsion_angles_list:
                num_torsions_found += 1
                # find the four atoms in seld.df and combine them
                a1, a2, a3, a4 = torsion_indices
                df_mol = self.df[self.df["mol_index"] == mol_index]
                df_a1 = df_mol[df_mol["idx_in_mol"] == a1].iloc[0]
                df_a2 = df_mol[df_mol["idx_in_mol"] == a2].iloc[0]
                df_a3 = df_mol[df_mol["idx_in_mol"] == a3].iloc[0]
                df_a4 = df_mol[df_mol["idx_in_mol"] == a4].iloc[0]
                # average node_attentions
                node_attentions = np.mean(
                    [
                        df_a1["node_attentions"],
                        df_a2["node_attentions"],
                        df_a3["node_attentions"],
                        df_a4["node_attentions"],
                    ],
                    axis=0,
                )
                new_line = df_a1.copy(deep=True)
                new_line["node_attentions"] = node_attentions
                new_line["truth"] = torsion_angle
                new_line["connected_atoms"] = [a1, a2, a3, a4]
                new_line["idx_in_mol"] = torsion_number
                af1 = df_a1["atom_feature"]
                af2 = df_a2["atom_feature"]
                af3 = df_a3["atom_feature"]
                af4 = df_a4["atom_feature"]
                new_line["atom_feature"] = get_canon_torsion_feature(af1, af2, af3, af4)
                df_list.append(new_line)
        self.df = pd.concat(df_list, axis=1).T
        if self.verbose:
            print(f"Found {num_torsions_found} torsions in the dataset")
            print(f"Created a dataframe with {self.df.shape} torsions")

    def create_correct_root_children(self):
        unique_afs_in_df = self.df.atom_feature.unique().tolist()
        self.roots = {}
        for af in unique_afs_in_df:
            self.roots[af] = DevelopNode(atom_features=[af, -1, -1], level=1)

    def convert_tree_to_node(self, delDevelop=False, tree_folder_path: str = "./"):
        """
        Helper function to convert develop nodes to normal nodes
        """
        self.new_tree = get_DASH_tree_from_DEV_tree(self.root, tree_folder_path=tree_folder_path)
