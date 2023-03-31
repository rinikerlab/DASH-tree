import datetime
import os
import logging

import pandas as pd
from rdkit import Chem

from serenityff.charge.tree.atom_features import AtomFeatures
from serenityff.charge.tree.node import node
from serenityff.charge.tree_develop.develop_node import DevelopNode
from serenityff.charge.tree_develop.tree_constructor_parallel_worker import Tree_constructor_parallel_worker
from serenityff.charge.tree_develop.tree_constructor_parallel import Tree_constructor

# from scipy import sparse


class Tree_constructor_branch(Tree_constructor):
    # TODO: Add description
    def __init__(
        self,
        df_path: str,
        sdf_suplier: str,
        attention_percentage: float = 0.99,
        num_layers_to_build=24,
        verbose=False,
        loggingBuild=False,
        af_number=0,
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
            print(f"{datetime.datetime.now()}\tInitializing Tree_constructor", flush=True)
        self.sdf_suplier = Chem.SDMolSupplier(sdf_suplier, removeHs=False)
        self.sdf_suplier_wo_h = Chem.SDMolSupplier(sdf_suplier, removeHs=True)
        self.feature_dict = dict()

        if verbose:
            print(f"{datetime.datetime.now()}\tMols imported, starting df import", flush=True)

        self.df = pd.read_csv(df_path)

        self.attention_percentage = attention_percentage
        self.num_layers_to_build = num_layers_to_build
        self.af_number = af_number
        self.roots = {}
        for af in AtomFeatures.feature_list:
            af_key = AtomFeatures.lookup_str(af)
            self.roots[af_key] = DevelopNode(atom_features=[af_key, -1, -1], level=1)
        self.new_root = node(level=0)

        if verbose:
            print(f"{datetime.datetime.now()}\tTable filled, starting adjacency matrix creation", flush=True)
        self._create_adjacency_matrices()

    def build_tree(self, num_processes=1, build_with_sperate_jobs=False):
        # TODO: Ich faende es vielleicht besser, wenn man hier create_tree_level_0 triggered und nicht
        # manuell aufrufen muss. Oder man checkt ob es schon getriggered wurde.
        tree_worker = Tree_constructor_parallel_worker(
            df_af_split={self.af_number: self.df},
            matrices=self.matrices,
            feature_dict=self.feature_dict,
            roots=self.roots,
            bond_matrices=self.bond_matrices,
            num_layers_to_build=self.num_layers_to_build,
            attention_percentage=self.attention_percentage,
            verbose=self.verbose,
            logger=[self.logger if self.loggingBuild else None],
        )
        tree_worker.build_tree(num_processes=1, af_list=[self.af_number])
        self.root = tree_worker.root
