import glob
import lzma
import os
import pickle

# from collections import defaultdict

# import numpy as np
import pandas as pd
from tqdm import tqdm

# from multiprocessing import Pool
# from multiprocessing import Process, Manager

# from newick import loads, Node, read, write

# from serenityff.charge.tree.atom_features import AtomFeatures
# from serenityff.charge.tree.node import node
# from serenityff.charge.tree.tree_utils import get_possible_atom_features
from serenityff.charge.data import default_dash_tree_path


class DASHTree:
    def __init__(self, tree_folder_path=default_dash_tree_path, preload=True, verbose=True, num_processes=4):
        self.tree_folder_path = tree_folder_path
        self.verbose = verbose
        self.num_processes = num_processes
        self.tree_storage = {}
        self.data_storage = {}
        if preload:
            self.load_all_trees_and_data()

    def load_all_trees_and_data(self):
        tree_paths = glob.glob(os.path.join(self.tree_folder_path, "*.lzma"))
        df_paths = glob.glob(os.path.join(self.tree_folder_path, "*.h5"))
        if self.verbose:
            print(f"Loading DASH tree data from {len(tree_paths)} files in {self.tree_folder_path}")
        for tree_path, df_path in tqdm(zip(tree_paths, df_paths), total=len(tree_paths)):
            self.load_tree_and_data(tree_path, df_path)
        # with Pool(self.num_processes) as p:
        #    p.starmap(self._load_tree_and_data, tqdm(zip(tree_paths, df_paths), total=len(tree_paths)))

    def _load_tree_and_data_star(self, tree_path, df_path, return_dict):
        branch_idx = int(os.path.basename(tree_path).split(".")[0])
        with lzma.open(tree_path, "rb") as f:
            tree = pickle.load(f)
        df = pd.read_hdf(df_path)
        return_dict["tree_storage"][branch_idx] = tree
        return_dict["data_storage"][branch_idx] = df

    def load_tree_and_data(self, tree_path, df_path):
        branch_idx = int(os.path.basename(tree_path).split(".")[0])
        with lzma.open(tree_path, "rb") as f:
            tree = pickle.load(f)
        df = pd.read_hdf(df_path)
        self.tree_storage[branch_idx] = tree
        self.data_storage[branch_idx] = df
