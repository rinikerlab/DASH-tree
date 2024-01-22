import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
from serenityff.torsion.tree_develop.develop_node import DevelopNode
from serenityff.charge.tree_develop.tree_constructor import Tree_constructor
from serenityff.charge.gnn.utils.rdkit_helper import get_all_torsion_angles
from serenityff.torsion.tree.dash_utils import get_canon_torsion_feature
from serenityff.torsion.tree.tree_utils import get_DASH_tree_from_DEV_tree
from serenityff.charge.tree.atom_features_reduced import AtomFeaturesReduced


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
        load_torsion_df_path=None,
        atom_feature_type=AtomFeaturesReduced,
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
            atom_feature_type,
        )
        self.node_type = DevelopNode
        self.atom_feature_type = atom_feature_type
        if verbose:
            print(f"{datetime.datetime.now()}\tCharge constructor build, creating torsion df", flush=True)
        if load_torsion_df_path is None:
            self.create_torsion_df()
            self.df.to_csv("torsion_df.csv", index=False)
        else:
            self.df = pd.read_csv(load_torsion_df_path)
            self.df["node_attentions"] = self.df["node_attentions"].apply(eval)
            self.df["connected_atoms"] = self.df["connected_atoms"].apply(eval)
        if verbose:
            print(f"{datetime.datetime.now()}\tTorison df created, creating root children", flush=True)
        self.create_correct_root_children()

    def create_torsion_df(self):
        df_list = []
        set_of_mol_indices_in_df = set(self.df["mol_index"].values)
        num_torsions_found = 0
        errors_found = 0
        # fix df indices
        self.df.reset_index(drop=True, inplace=True)
        mol_2_df_dict = self.df.groupby("mol_index").groups
        for mol_index, mol in enumerate(self.sdf_suplier):
            if self.verbose:
                if mol_index % 10000 == 0:
                    print(
                        f"{datetime.datetime.now()}\t{mol_index}/{len(self.sdf_suplier)} molecules processed",
                        flush=True,
                    )
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
                try:
                    group = mol_2_df_dict[mol_index]
                    df_mol = self.df.iloc[group]
                    df_a1 = df_mol[df_mol["idx_in_mol"] == a1].iloc[0]
                    df_a2 = df_mol[df_mol["idx_in_mol"] == a2].iloc[0]
                    df_a3 = df_mol[df_mol["idx_in_mol"] == a3].iloc[0]
                    df_a4 = df_mol[df_mol["idx_in_mol"] == a4].iloc[0]
                except IndexError:
                    errors_found += 1
                    continue
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
            print(f"Found {errors_found} errors in the dataset")
            print(f"Created a dataframe with {self.df.shape} torsions")

    def create_correct_root_children(self):
        unique_afs_in_df = self.df.atom_feature.unique().tolist()
        self.roots = {}
        for af in unique_afs_in_df:
            self.roots[af] = DevelopNode(atom_features=[af, -1, -1], level=1)
        if self.verbose:
            print(f"Created {len(self.roots)} root children of {self.atom_feature_type.get_number_of_features()} af's")

    def create_tree_level_0(self, save_dfs_prefix: str = None):
        """
        Creates the tree level 0 for all atom features

        First level is a separate function due to the different function body and option to restart building from here
        """
        print("Preparing Dataframe:")
        self.df_af_split = {}
        self.unique_afs_in_df = self.df.atom_feature.unique().tolist()
        if self.verbose:
            print(f"Number of unique atom features in df: {len(self.unique_afs_in_df)}")
        for af in self.unique_afs_in_df:
            self.df_af_split[af] = self.df.loc[self.df.atom_feature == af].copy()

        print("Creating Tree Level 0:")
        for af in tqdm(self.unique_afs_in_df):
            df_work = self.df_af_split[af]
            current_node = self.roots[af]
            try:
                truth_values = df_work["truth"].to_list()
                # attention_values = df_work.apply(lambda x: np.sum(x["node_attentions"][x["connected_atoms"]]), axis=1).to_list()
                attention_values = [0] * len(truth_values)
                current_node.truth_values = truth_values
                current_node.attention_values = attention_values
                current_node.update_average()
            except (KeyError, AttributeError):
                pass
            df_work[0] = df_work["atom_feature"]
            if save_dfs_prefix is not None:
                df_work.to_csv(f"{save_dfs_prefix}_layer_0_{af}.csv")
        print(f"{datetime.datetime.now()}\tLayer 0 done")

    def convert_tree_to_node(self, delDevelop=False, tree_folder_path: str = "./"):
        """
        Helper function to convert develop nodes to normal nodes
        """
        self.new_tree = get_DASH_tree_from_DEV_tree(self.root, tree_folder_path=tree_folder_path)
