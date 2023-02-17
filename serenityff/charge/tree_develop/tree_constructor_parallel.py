import datetime
import pickle
import random
import os
import logging

import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from serenityff.charge.gnn.utils import mols_from_sdf
from serenityff.charge.tree.atom_features import AtomFeatures
from serenityff.charge.tree.node import node
from serenityff.charge.tree.tree_utils import (
    create_new_node_from_develop_node,
)
from serenityff.charge.tree_develop.develop_node import DevelopNode
from serenityff.charge.tree_develop.tree_constructor_parallel_worker import Tree_constructor_parallel_worker


class Tree_constructor:
    def __init__(
        self,
        df_path: str,
        sdf_suplier: str,
        nrows: int,
        attention_percentage: float = 0.99,
        data_split: float = 0.2,
        seed: int = 42,
        num_layers_to_build=24,
        sanitize=False,
        verbose=False,
        loggingBuild=False,
        split_indices_path=None,
    ):
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
            print(f"{datetime.datetime.now()}\tInitializing Tree_constructor")
        self.sdf_suplier = mols_from_sdf(sdf_suplier, removeHs=False)
        self.sdf_suplier_wo_h = mols_from_sdf(sdf_suplier, removeHs=True)
        self.feature_dict = dict()
        # self.dask_client = Client()
        # if verbose:
        #    print(self.dask_client)

        if verbose:
            print(f"{datetime.datetime.now()}\tMols imported, starting df import")

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
        if sanitize:
            if verbose:
                print(f"{datetime.datetime.now()}\tSanitizing")
            self._clean_molecule_indices_in_df()
            if verbose:
                print(f"{datetime.datetime.now()}\tCheck charge sanity")
            self._check_charge_sanity()

        if verbose:
            print(f"{datetime.datetime.now()}\tdf imported, starting data spliting")

        random.seed(seed)
        if split_indices_path is None:
            unique_mols = self.original_df.mol_index.unique().tolist()
            test_set = random.sample(
                unique_mols,
                int(len(unique_mols) * data_split),
            )
        else:
            df_test_set = pd.read_csv(split_indices_path)
            test_set = df_test_set["sdf_idx"].tolist()
        self.df = self.original_df.loc[~self.original_df.mol_index.isin(test_set)].copy()
        self.test_df = self.original_df.loc[self.original_df.mol_index.isin(test_set)].copy()

        delattr(self, "original_df")
        self.df["node_attentions"] = self.df["node_attentions"].apply(eval)

        h, c, t, n, af = [], [], [], [], []

        if verbose:
            print(f"{datetime.datetime.now()}\tStarting table filling")

        self.tempmatrix = Chem.GetAdjacencyMatrix(self.sdf_suplier[0])

        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
            n.append(np.array(row["node_attentions"]) / sum(row["node_attentions"]))
            h.append(self._get_hydrogen_connectivity(row))
            c.append(([] if row["atomtype"] == "H" else [row["idx_in_mol"]]))
            t.append(row["node_attentions"][row["idx_in_mol"]])
            tmp_af = AtomFeatures.atom_features_from_molecule(self.sdf_suplier[row["mol_index"]], row["idx_in_mol"])
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

        self.attention_percentage = attention_percentage
        self.num_layers_to_build = num_layers_to_build
        self.roots = {}
        for af in AtomFeatures.feature_list:
            af_key = AtomFeatures.lookup_str(af)
            self.roots[af_key] = DevelopNode(atom_features=[af_key, -1, -1], level=1)
        self.new_root = node(level=0)

        self._create_adjacency_matrices()

        print(f"Number of train mols: {len(self.df.mol_index.unique())}")
        print(f"Number of test mols: {len(self.test_df.mol_index.unique())}")

    def _clean_molecule_indices_in_df(self):
        molecule_idx_in_df = self.original_df.mol_index.unique().tolist()
        for mol_index in molecule_idx_in_df:
            number_of_atoms_in_mol_df = len(self.original_df.loc[self.original_df.mol_index == mol_index])
            number_of_atoms_in_mol_sdf = self.sdf_suplier[mol_index].GetNumAtoms()
            if number_of_atoms_in_mol_df > number_of_atoms_in_mol_sdf:
                if number_of_atoms_in_mol_df <= self.sdf_suplier[mol_index + 1].GetNumAtoms():
                    self.original_df.loc[self.original_df.mol_index >= mol_index, "mol_index"] += 1
                elif number_of_atoms_in_mol_df <= self.sdf_suplier[mol_index + 2].GetNumAtoms():
                    self.original_df.loc[self.original_df.mol_index >= mol_index, "mol_index"] += 2
                elif number_of_atoms_in_mol_df <= self.sdf_suplier[mol_index + 3].GetNumAtoms():
                    self.original_df.loc[self.original_df.mol_index >= mol_index, "mol_index"] += 3
                elif number_of_atoms_in_mol_df <= self.sdf_suplier[mol_index + 4].GetNumAtoms():
                    self.original_df.loc[self.original_df.mol_index >= mol_index, "mol_index"] += 4
                elif number_of_atoms_in_mol_df <= self.sdf_suplier[mol_index + 5].GetNumAtoms():
                    self.original_df.loc[self.original_df.mol_index >= mol_index, "mol_index"] += 5
                else:
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
            else:
                pass

    def _check_charge_sanity(self):
        self.wrong_charged_mols_list = []
        indices_to_drop = []
        for mol_index in tqdm(self.original_df.mol_index.unique()):
            df_with_mol_index = self.original_df.loc[self.original_df.mol_index == mol_index]
            charges = df_with_mol_index.truth.values
            elements = df_with_mol_index.atomtype.values
            for element, charge in zip(elements, charges):
                if element == "H" and charge < -0.01:
                    indices_to_drop.extend(df_with_mol_index.index.to_list())
                    self.wrong_charged_mols_list.append(mol_index)
                    break
                elif element == "C" and (charge < -2 or charge > 4):
                    indices_to_drop.extend(df_with_mol_index.index.to_list())
                    self.wrong_charged_mols_list.append(mol_index)
                    break
                elif element == "N" and (charge < -4 or charge > 6):
                    indices_to_drop.extend(df_with_mol_index.index.to_list())
                    self.wrong_charged_mols_list.append(mol_index)
                    break
                elif element == "O" and (charge < -4 or charge > 6):
                    indices_to_drop.extend(df_with_mol_index.index.to_list())
                    self.wrong_charged_mols_list.append(mol_index)
                    break
        self.original_df.drop(indices_to_drop, inplace=True)
        if self.verbose:
            print(
                f"Number of wrong charged mols: {len(self.wrong_charged_mols_list)} of {len(self.original_df.mol_index.unique())} mols"
            )

    def _get_hydrogen_connectivity(self, line) -> int:
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
        return AtomFeatures.atom_features_from_molecule_w_connection_info(
            self.sdf_suplier[line["mol_index"]], line["idx_in_mol"]
        )

    def _create_single_adjacency_matrix(self, mol: Chem.Mol) -> np.ndarray:
        bonddict = {v: k for k, v in Chem.rdchem.BondType.values.items()}
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
        return matrix

    def _create_adjacency_matrices(self):
        print("Creating Adjacency matrices:")
        self.matrices = []
        self.bond_matrices = []
        for mol in tqdm(self.sdf_suplier_wo_h):  # i, mol in enumerate(self.sdf_suplier_wo_h):  #
            matrix = self._create_single_adjacency_matrix(mol)
            self.bond_matrices.append(matrix)
        # self.bond_matrices = self.dask_client.gather(self.bond_matrices)

    def create_tree_level_0(self):
        print("Preparing Dataframe:")
        self.df_af_split = {}
        for af in range(AtomFeatures.get_number_of_features()):
            self.df_af_split[af] = self.df.loc[self.df.atom_feature == af].copy()

        print("Creating Tree Level 0:")
        for af in tqdm(range(AtomFeatures.get_number_of_features())):
            df_work = self.df_af_split[af]
            current_node = self.roots[af]
            try:
                truth_values = df_work["truth"].to_list()
                attention_values = df_work.apply(lambda x: x["node_attentions"][x["idx_in_mol"]], axis=1).to_list()
                current_node.truth_values = truth_values
                current_node.attention_values = attention_values
                current_node.update_average()
            except (KeyError, AttributeError):
                # print(f"Layer 0: {af} has no truth values")
                pass
            df_work[0] = df_work["atom_feature"]
        print(f"{datetime.datetime.now()}\tLayer 0 done")

    def build_tree(self, num_processes):
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
        )
        tree_worker.build_tree(num_processes=num_processes)
        self.root = tree_worker.root

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
