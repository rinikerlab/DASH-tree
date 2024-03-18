import datetime
import pickle
import random
import os
import logging
import time
from typing import NoReturn

import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from collections import defaultdict

from serenityff.charge.tree.atom_features import (
    AtomFeatures,
    get_connection_info_bond_type,
)
from serenityff.charge.tree.node import Node
from serenityff.charge.tree.tree_utils import get_DASH_tree_from_DEV_tree
from serenityff.charge.tree_develop.develop_node import DevelopNode
from serenityff.charge.tree_develop.tree_constructor_parallel_worker import (
    Tree_constructor_parallel_worker,
)
from serenityff.charge.tree_develop.tree_constructor_singleJB_worker import (
    Tree_constructor_singleJB_worker,
)


class Tree_constructor:
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
        atom_feature_type=AtomFeatures,
    ) -> None:
        """Initialize Tree_constructor object for DASH tree building (SerenityFF)
        This constructor can:
        I)  load a dataframe from a csv file and a sdf file
        II) sanitize the dataframe
        III) split the dataframe into a train and test set (randomly or by indices provided)
        IV) create all adjacency matrices and atom features for all molecules
        V) prepare everything for the tree building (seperate functions "create_tree_level_0" and "build_tree" are needed for the actual tree building)

        Parameters
        ----------
        df_path : str
            path to the csv file containing the dataframe with the attention data (format see examples)
        sdf_suplier : str
            path to the sdf file containing the molecules
        nrows : int, optional
            number of rows to read from the csv file, by default None (read all)
        attention_percentage : float, optional
            make attention_percentage to build the tree, by default 1000
        data_split : float, optional
            random split of the data into train and test set, by default 0.1
        seed : int, optional
            seed for the random split, by default 42
        num_layers_to_build : int, optional
            max number of layers to build, by default 24
        sanitize : bool, optional
            check if sdf and df are consistent, by default False
        sanitize_charges : bool, optional
            check if charges are consistent, by default False
        verbose : bool, optional
            Do you want to read a story? Once upon a time... , by default False
        loggingBuild : bool, optional
            Debugging, by default False
        split_indices_path : _type_, optional
            Provide indices for the split (to be consistent with GNN), overrides random split, by default None
        save_cleaned_df_path : _type_, optional
            Debug option for skipping steps, by default None
        save_feature_dict_path : _type_, optional
            Debug option for skipping steps, by default None
        """
        # init
        self.node_type = DevelopNode
        self.atom_feature_type = atom_feature_type
        if loggingBuild:
            self.loggingBuild = True
            logging.basicConfig(
                filename=os.path.dirname(df_path) + "/tree_constructor.log",
                filemode="a",
                format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                level=logging.DEBUG,
            )
            self.logger = logging.getLogger("TreeConstructor")
            self.logger.setLevel(logging.DEBUG)
        else:
            self.loggingBuild = False

        self.verbose = verbose
        if verbose:
            print(f"{datetime.datetime.now()}\tInitializing Tree_constructor", flush=True)
        self.sdf_suplier = Chem.SDMolSupplier(sdf_suplier, removeHs=False)
        self.sdf_suplier_wo_h = Chem.SDMolSupplier(sdf_suplier, removeHs=True)
        self.feature_dict = dict()

        # load data
        if verbose:
            print(
                f"{datetime.datetime.now()}\tMols imported, starting df import",
                flush=True,
            )
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

        # clean data
        if sanitize:
            if verbose:
                print(f"{datetime.datetime.now()}\tSanitizing", flush=True)
            self._clean_molecule_indices_in_df()
            if sanitize_charges:
                if verbose:
                    print(f"{datetime.datetime.now()}\tCheck charge sanity", flush=True)
                self._check_charge_sanity()

        if save_cleaned_df_path is not None:
            if verbose:
                print(f"{datetime.datetime.now()}\tSaving cleaned df", flush=True)
            self.original_df.to_csv(save_cleaned_df_path, index=False)

        if verbose:
            print(
                f"{datetime.datetime.now()}\tdf imported, starting data spliting",
                flush=True,
            )

        # split data in train and validation set
        random.seed(seed)
        if split_indices_path is None:
            unique_mols = self.original_df.mol_index.unique().tolist()
            test_set = random.sample(
                unique_mols,
                int(len(unique_mols) * data_split),
            )
            test_set = set(test_set)
        else:
            if verbose:
                print(
                    f"{datetime.datetime.now()}\tUsing split indices from {split_indices_path}",
                    flush=True,
                )
            df_test_set = pd.read_csv(split_indices_path)
            test_set = df_test_set["sdf_idx"].tolist()
            test_set = [int(i) for i in test_set]
            test_set = set(test_set)
        if verbose:
            print(f"{datetime.datetime.now()}\tSplitting data", flush=True)
        self.df = self.original_df.loc[~self.original_df.mol_index.isin(test_set)].copy()
        self.test_df = self.original_df.loc[self.original_df.mol_index.isin(test_set)].copy()

        if verbose:
            print(f"{datetime.datetime.now()}\tData split, delete original", flush=True)
        delattr(self, "original_df")
        self.df["node_attentions"] = self.df["node_attentions"].apply(eval)

        # create feature dict and adjacency matrix
        h, c, t, n, af = [], [], [], [], []
        if verbose:
            print(f"{datetime.datetime.now()}\tStarting table filling", flush=True)
        self.tempmatrix = Chem.GetAdjacencyMatrix(self.sdf_suplier[0])
        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
            n.append(np.array(row["node_attentions"]) / sum(row["node_attentions"]))
            h.append(self._get_hydrogen_connectivity(row))
            c.append(([] if row["atomtype"] == "H" else [row["idx_in_mol"]]))
            t.append(row["node_attentions"][row["idx_in_mol"]])
            tmp_af = self.atom_feature_type.atom_features_from_molecule(
                self.sdf_suplier[row["mol_index"]], row["idx_in_mol"]
            )
            af.append(tmp_af)
            if row["idx_in_mol"] == 0:
                self.feature_dict[row["mol_index"]] = dict()
                self.feature_dict[row["mol_index"]][row["idx_in_mol"]] = tmp_af
            else:
                self.feature_dict[row["mol_index"]][row["idx_in_mol"]] = tmp_af
        self.df["h_connectivity"] = h
        self.df["connected_atoms"] = c
        self.df["total_connected_attention"] = t
        self.df["node_attentions"] = n
        self.df["atom_feature"] = af
        del h, c, n, t, af
        delattr(self, "tempmatrix")

        # Advanced/Debug options
        if save_feature_dict_path is not None:
            if verbose:
                print(f"{datetime.datetime.now()}\tSaving feature dict", flush=True)
            with open(save_feature_dict_path, "wb") as f:
                pickle.dump(self.feature_dict, f)

        # Initialize tree root
        self.attention_percentage = attention_percentage
        self.num_layers_to_build = num_layers_to_build
        self.roots = {}
        for af in self.atom_feature_type.feature_list:
            af_key = self.atom_feature_type.afTuple_2_afKey[af]
            self.roots[af_key] = self.node_type(atom_features=[af_key, -1, -1], level=1)
        self.new_root = Node(level=0)

        if verbose:
            print(
                f"{datetime.datetime.now()}\tTable filled, starting adjacency matrix creation",
                flush=True,
            )
        self._create_adjacency_matrices()

        print(f"Number of train mols: {len(self.df.mol_index.unique())}")
        print(f"Number of test mols: {len(self.test_df.mol_index.unique())}")

    def _clean_molecule_indices_in_df(self):
        """
        Brute force method to clean molecule indices in df and sdf_suplier
        """
        num_missmatches = 0
        molecule_idx_in_df = self.original_df.mol_index.unique().tolist()
        for mol_index in molecule_idx_in_df:
            number_of_atoms_in_mol_df = len(self.original_df.loc[self.original_df.mol_index == mol_index])
            number_of_atoms_in_mol_sdf = self.sdf_suplier[mol_index].GetNumAtoms()
            if number_of_atoms_in_mol_df > number_of_atoms_in_mol_sdf:
                num_missmatches += 1
                for i in range(5):
                    if number_of_atoms_in_mol_df <= self.sdf_suplier[mol_index + 1 + i].GetNumAtoms():
                        self.original_df.loc[self.original_df.mol_index >= mol_index, "mol_index"] += 1 + i
                        break
                    if i == 4:
                        self._raise_index_missmatch_error(
                            mol_index,
                            number_of_atoms_in_mol_df,
                            number_of_atoms_in_mol_sdf,
                        )
            else:
                pass
        if self.verbose:
            print(f"Number of missmatches found in sanitizing: {num_missmatches}")

    def _raise_index_missmatch_error(
        self, mol_index, number_of_atoms_in_mol_df, number_of_atoms_in_mol_sdf
    ) -> NoReturn:
        print(
            f"Molecule {mol_index} has {number_of_atoms_in_mol_df} atoms in df and {number_of_atoms_in_mol_sdf} atoms in sdf"
        )
        print(f"shifted mol has {self.sdf_suplier[mol_index+1].GetNumAtoms()} atoms")
        print("--------------------------------------------------")
        print(self.original_df.loc[self.original_df.mol_index == mol_index])
        print("--------------------------------------------------")
        print(self.original_df.loc[self.original_df.mol_index == mol_index].iloc[0].smiles)
        print(Chem.MolToSmiles(self.sdf_suplier[mol_index]))
        print(Chem.MolToSmiles(self.sdf_suplier[mol_index + 1]))
        print("--------------------------------------------------")
        raise ValueError(f"Number of atoms in df and sdf are not the same for molecule {mol_index}")

    def _check_charge_sanity(self) -> None:
        """
        Checks if the charges in the original_df are reasonable. Deletes mols with unphysical charges
        """
        self.wrong_charged_mols_list = []
        indices_to_drop = []
        for mol_index in tqdm(self.original_df.mol_index.unique()):
            df_with_mol_index = self.original_df.loc[self.original_df.mol_index == mol_index]
            charges = df_with_mol_index.truth.values
            elements = df_with_mol_index.atomtype.values
            for element, charge in zip(elements, charges):
                self._check_charges(element, charge, indices_to_drop, df_with_mol_index, mol_index)
        self.original_df.drop(indices_to_drop, inplace=True)
        if self.verbose:
            print(
                f"Number of wrong charged mols: {len(self.wrong_charged_mols_list)} of {len(self.original_df.mol_index.unique())} mols"
            )

    def _check_charges(self, element, charge, indices_to_drop, df_with_mol_index, mol_index) -> None:
        """
        Thresholds for reasonable charges
        """
        check_charge_dict_temp = {
            "H": (-0.01, 1.01),
            "C": (-2, 4),
            "N": (-4, 6),
            "O": (-4, 6),
            "S": (-10, 10),
            "P": (-10, 10),
            "F": (-10, 0.01),
            "Cl": (-10, 0.01),
            "Br": (-10, 0.01),
            "I": (-10, 0.01),
        }
        check_charge_dict = defaultdict(lambda: (-10, 10), check_charge_dict_temp)
        lower_bound, upper_bound = check_charge_dict[element]
        if charge < lower_bound or charge > upper_bound:
            indices_to_drop.extend(df_with_mol_index.index.to_list())
            self.wrong_charged_mols_list.append(mol_index)

    def _get_hydrogen_connectivity(self, line) -> int:
        """
        Returns the index of the atom to which the hydrogen is connected else -1
        """
        if line["idx_in_mol"] == 0:
            self.tempmatrix = Chem.GetAdjacencyMatrix(self.sdf_suplier[line["mol_index"]])
        if line["atomtype"] == "H":
            try:
                return int(np.where(self.tempmatrix[line["idx_in_mol"]])[0].item())
            except ValueError:
                return -1
        else:
            return -1

    def _create_atom_features(self, line):
        return self.atom_feature_type.atom_features_from_molecule_w_connection_info(
            self.sdf_suplier[line["mol_index"]], line["idx_in_mol"]
        )

    def _create_single_adjacency_matrix(self, mol: Chem.Mol) -> np.ndarray:
        """
        Create storage of all adjacency matrices with bond type information for later lookup without rdkit
        """
        matrix = np.array(Chem.GetAdjacencyMatrix(mol), np.bool_)
        np.fill_diagonal(matrix, True)
        self.matrices.append(matrix)
        matrix = matrix.astype(np.int8)
        for i in range(matrix.shape[0]):
            for j in np.arange(i + 1, matrix.shape[1]):
                if matrix[i][j]:
                    matrix[i][j] = get_connection_info_bond_type(mol, i, j)
                    matrix[j][i] = matrix[i][j]
        # warning for future developers. Don't try to be smart and use sparse matrices. It will be slower
        return matrix

    def _create_adjacency_matrices(self):
        """
        For all mols:
            (Create storage of all adjacency matrices with bond type information for later lookup without rdkit)
        """
        print("Creating Adjacency matrices:")
        self.matrices = []
        self.bond_matrices = []
        for mol in tqdm(self.sdf_suplier_wo_h):
            matrix = self._create_single_adjacency_matrix(mol)
            self.bond_matrices.append(matrix)

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
                attention_values = df_work.apply(lambda x: x["node_attentions"][x["idx_in_mol"]], axis=1).to_list()
                current_node.truth_values = truth_values
                current_node.attention_values = attention_values
                current_node.update_average()
            except (KeyError, AttributeError):
                pass
            df_work[0] = df_work["atom_feature"]
            if save_dfs_prefix is not None:
                df_work.to_csv(f"{save_dfs_prefix}_layer_0_{af}.csv")
        print(f"{datetime.datetime.now()}\tLayer 0 done")

    def _build_with_seperate_slurm_jobs(self, tree_worker: Tree_constructor_parallel_worker):
        """
        Helper function to build each branch with a different slurm job
        """
        pickle_path = "tree_worker.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(tree_worker, f)
        out_folder = "tree_out"
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        for af in self.unique_afs_in_df:
            # for af in range(self.atom_feature_type.get_number_of_features() + 2):
            try:
                temp = pickle.load(open(f"{out_folder}/{af}.pkl", "rb"))
                assert temp is not None
                assert temp.level is not None
            except (FileNotFoundError, AssertionError):
                Tree_constructor_singleJB_worker.run_singleJB(pickle_path, af)
        time.sleep(200)
        num_slurm_jobs = int(os.popen("squeue | grep  't_' | wc -l").read())
        while num_slurm_jobs > 0:
            time.sleep(200)
            num_slurm_jobs = int(os.popen("squeue | grep  't_' | wc -l").read())
        # collect all pickle files
        for af in self.unique_afs_in_df:
            # for af in range(self.atom_feature_type.get_number_of_features() + 2):
            try:
                with open(f"{out_folder}/{af}.pkl", "rb") as f:
                    self.root.children.append(pickle.load(f))
            except FileNotFoundError:
                print(f"File {af}.pkl not found")

    def build_tree(self, num_processes=1, build_with_sperate_jobs=False):
        """
        Builds the tree from the level 0 nodes (make sure to run create_tree_level_0 before)

        Different parallelization options are available
        """
        tree_worker = Tree_constructor_parallel_worker(
            df_af_split=self.df_af_split,
            matrices=self.matrices,
            feature_dict=self.feature_dict,
            roots=self.roots,
            bond_matrices=self.bond_matrices,
            num_layers_to_build=self.num_layers_to_build,
            attention_percentage=self.attention_percentage,
            verbose=self.verbose,
            logger=[self.logger if self.loggingBuild else None],
            node_type=self.node_type,
        )
        if build_with_sperate_jobs:
            self._build_with_seperate_slurm_jobs(tree_worker)
        else:
            tree_worker.build_tree(num_processes=num_processes)
            self.root = tree_worker.root

    def convert_tree_to_node(self, delDevelop=False, tree_folder_path: str = "./"):
        """
        Helper function to convert develop nodes to normal nodes
        """
        # self.new_root = create_new_node_from_develop_node(self.root)
        # if delDevelop:
        #    del self.root
        #    self.root = None
        get_DASH_tree_from_DEV_tree(self.root, tree_folder_path=tree_folder_path)

    def calculate_tree_length(self):
        self.tree_length = self.new_root.calculate_tree_length()

    def pickle_tree(self, file_name):
        with open(file_name, "wb") as f:
            pickle.dump(self.root, f)

    def write_tree_to_file(self, file_name):
        self.new_root.to_file(file_name)
